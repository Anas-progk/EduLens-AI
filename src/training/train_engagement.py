"""
Complete training pipeline -- v4 (Three-Phase Balanced Sampling).

ROOT CAUSE DIAGNOSIS (from v3 threshold sweep evidence):
  P(Engaged) stats: mean=0.627, p10=0.505, p25=0.560
  NE F1 = 0.000 at ALL thresholds below 0.40.
  Best NE F1 = 0.097 at threshold=0.482.
  The model outputs P(Engaged) > 0.50 for virtually every sample.

  WHY: With batch_size=32 and 5.8% NE frequency, each training batch contains
  only ~1.86 NE samples on average.  Some batches have ZERO NE samples.
  Focal Loss requires probability calibration to work -- it downweights easy
  examples.  But if NE examples barely appear, there is nothing to focus on.
  The model converges to a stable local minimum: "predict Engaged always"
  (94.2% training accuracy).  No loss function, no matter how strong, can
  escape this minimum when the gradient from 2 NE samples per batch is
  overwhelmed by 30 Engaged samples.

  PROOF: v3 used gamma=3.0, alpha_power=0.75 (8.9:1 ratio), prior bias init,
  class-conditional augmentation -- and still NE F1 = 0.097.
  The loss WAS strong enough; the EXPOSURE was not.

THREE-PHASE FIX:
  The KEY rule: sampler + UNIFORM alpha  = no double-correction (safe).
                sampler + STRONG alpha   = double-correction (v1 crash).

  Phase 1 [BOOTSTRAP] -- 10 epochs, frozen backbone, 50:50 sampler:
    ~16 NE samples per batch instead of ~2.
    Uniform Focal alpha (sampler provides balance, alpha would double-correct).
    Head learns to separate NE from Engaged in ImageNet feature space.
    Strong aug on NE prevents memorisation of the 2,656 NE frames.
    Expected: NE F1 = 0.25-0.45 (from 0.09 in v3).

  Phase 2 [ALIGN] -- 12 epochs, unfrozen, 25% NE sampler:
    ~8 NE per batch (4x natural frequency).  Very low LRs.
    Backbone features slowly align to engagement cues.
    Uniform alpha -- sampler still handles NE exposure.
    Expected: NE F1 = 0.30-0.55, stable.

  Phase 3 [CALIBRATE] -- 8 epochs, unfrozen, real distribution, no sampler:
    Real 94:6 batches.  Strong Focal alpha (power=0.75).
    Calibrates probability estimates to the real-world distribution.
    Threshold sweep continues each epoch for best-checkpoint selection.
    Expected: NE F1 = 0.25-0.45 (calibrated).  Macro F1 target: 0.62-0.75.

KEPT FROM v3:
  - Prior bias initialisation: log([0.06, 0.94]) prevents wasted epochs
  - Per-epoch numpy threshold sweep: model saved on threshold-adjusted F1
  - Class-conditional augmentation: NE frames get stronger transforms
  - 8-tuple run_epoch return (includes P(Engaged) for threshold sweep)
"""

import math
import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

from src.config import CONFIG
from src.data.dataset import EngagementDataset
from src.models.swin_model import (
    build_engagement_model,
    freeze_backbone, unfreeze_all,
    get_optimizer_frozen, get_optimizer_unfrozen,
)


# =============================================================================
# PHASE CONFIGURATION
# All phase-specific hyperparameters are defined here, not scattered in CONFIG.
# Modify this block to experiment with different strategies.
# =============================================================================

PHASES = [
    # -------------------------------------------------------------------------
    # PHASE 1: BOOTSTRAP
    # Goal: force the model to learn what "Not Engaged" looks like.
    # Frozen backbone = only the 0.2M head trains, backbone serves as extractor.
    # 50:50 sampler = ~16 NE per batch.  Uniform alpha = no double-correction.
    # gamma=1.5 (mild) because the sampler already provides NE exposure.
    # -------------------------------------------------------------------------
    {
        'name'          : 'BOOTSTRAP',
        'epochs'        : CONFIG['phase1_epochs'],   # 10
        'frozen'        : True,
        'ne_ratio'      : 0.50,        # 50% NE per batch
        'focal_gamma'   : 1.5,
        'focal_alpha_pw': 0.0,         # UNIFORM -- sampler balances, no double-correction
        'lr_head'       : CONFIG['phase1_lr_head'],
        'lr_backbone'   : None,
        'wd'            : CONFIG['phase1_wd'],
        'warmup_epochs' : 0,           # no warmup: fresh head benefits from full LR
    },
    # -------------------------------------------------------------------------
    # PHASE 2: ALIGN
    # Goal: backbone features adapt to engagement cues without losing Phase 1 gains.
    # 25% NE sampler (still 4x natural) with very low LRs -- preserve head knowledge.
    # Uniform alpha still -- sampler provides enough NE signal.
    # gamma=2.0 (moderate) for slightly harder example focus.
    # -------------------------------------------------------------------------
    {
        'name'          : 'ALIGN',
        'epochs'        : CONFIG['phase2_epochs'],   # 12
        'frozen'        : False,
        'ne_ratio'      : 0.25,        # 25% NE per batch
        'focal_gamma'   : 2.0,
        'focal_alpha_pw': 0.0,         # UNIFORM -- sampler still provides balance
        'lr_head'       : CONFIG['phase2_lr_head'],
        'lr_backbone'   : CONFIG['phase2_lr_backbone'],
        'wd'            : CONFIG['phase2_wd'],
        'warmup_epochs' : CONFIG['phase2_warmup'],   # 2
    },
    # -------------------------------------------------------------------------
    # PHASE 3: CALIBRATE
    # Goal: calibrate probability estimates to the real 94:6 distribution.
    # No sampler -- the model already knows NE features from Phases 1+2.
    # Strong alpha (power=0.75, ~8.9:1) compensates for natural imbalance.
    # Very low LRs -- just shift the decision boundary, don't relearn features.
    # Early stopping active (Phases 1+2 always complete fully).
    # -------------------------------------------------------------------------
    {
        'name'          : 'CALIBRATE',
        'epochs'        : CONFIG['phase3_epochs'],   # 8
        'frozen'        : False,
        'ne_ratio'      : None,        # no sampler -- real 94:6 distribution
        'focal_gamma'   : 3.0,
        'focal_alpha_pw': 0.75,        # ~8.9:1 NE:E -- strong but not 16:1
        'lr_head'       : CONFIG['phase3_lr_head'],
        'lr_backbone'   : CONFIG['phase3_lr_backbone'],
        'wd'            : CONFIG['phase3_wd'],
        'warmup_epochs' : 0,
    },
]


# =============================================================================
# FOCAL LOSS
# =============================================================================

class FocalLoss(nn.Module):
    """
    FL = -alpha_t * (1 - p_t)^gamma * log(p_t)

    alpha: per-class weight tensor (shape: num_classes).
           With balanced sampler use alpha=ones (uniform) to avoid double-correction.
           Without sampler use dampened inverse-frequency alpha.
    gamma: focusing parameter. Higher = more focus on hard examples.
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer('alpha', alpha)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets,
                             weight=self.alpha.to(inputs.device),
                             reduction='none')
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def compute_focal_alpha(labels_array, power: float = 0.5) -> torch.Tensor:
    """
    Dampened inverse-frequency class weights.
    power=0.0 -> uniform [1, 1]
    power=0.5 -> sqrt(inv_freq), 4:1 for 94:6
    power=0.75 -> inv_freq^0.75, 8.9:1 for 94:6
    Returns tensor normalized so mean=1.
    """
    counts  = np.bincount(labels_array, minlength=2).astype(float)
    weights = (len(labels_array) / (2.0 * counts)) ** power
    return torch.FloatTensor(weights / weights.mean())


# =============================================================================
# EARLY STOPPING
# =============================================================================

class EarlyStopping:
    def __init__(self, patience=8, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = None
        self.counter   = 0

    def step(self, score) -> bool:
        if self.best is None or score > self.best + self.min_delta:
            self.best    = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def reset(self):
        self.best    = None
        self.counter = 0


# =============================================================================
# FAST THRESHOLD SWEEP  (numpy, <5 ms per epoch)
# =============================================================================

def sweep_threshold_fast(probs, labels, n_steps=25):
    """
    Find P(Engaged) threshold maximising macro-F1.
    Range 0.05-0.70 covers all realistic optima for imbalanced classifiers.
    Returns (best_threshold: float, best_macro_f1: float).
    """
    probs  = np.asarray(probs,  dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.05, 0.70, n_steps):
        preds = (probs >= t).astype(np.int32)
        f1    = f1_score(labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# =============================================================================
# CORE EPOCH RUNNER  (8-tuple return)
# =============================================================================

def run_epoch(model, loader, optimizer, criterion, scaler, device,
              grad_clip=0.3, training=True):
    """
    One full pass (train or eval).

    Returns (8-tuple):
      avg_loss, accuracy(%), macro_f1, f1_not_engaged, f1_engaged,
      all_labels (list), all_preds (list), all_probs (list of P(Engaged))

    all_probs is populated only during eval (empty list during training).
    """
    model.train() if training else model.eval()
    total_loss, counted, correct, n_samples = 0.0, 0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    use_amp = device.startswith('cuda') and scaler is not None
    desc    = '  Train' if training else '  Val  '

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast(device_type=device.split(':')[0], enabled=use_amp):
                out  = model(images)
                loss = criterion(out, labels)

            if not torch.isfinite(loss):
                print(f'  [WARN] non-finite loss {loss.item():.4f}, skipping batch')
                if training:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if training:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            preds       = out.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            n_samples  += labels.size(0)
            total_loss += loss.item()
            counted    += 1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if not training:
                all_probs.extend(
                    torch.softmax(out, dim=1)[:, 1].cpu().numpy())

    if counted == 0:
        return float('nan'), 0., 0., 0., 0., [], [], []

    avg_loss = total_loss / counted
    accuracy = 100.0 * correct / max(1, n_samples)
    f1s      = f1_score(all_labels, all_preds, average=None,
                        labels=[0, 1], zero_division=0)
    macro_f1 = float(np.mean(f1s))
    return avg_loss, accuracy, macro_f1, float(f1s[0]), float(f1s[1]), \
           all_labels, all_preds, all_probs


# =============================================================================
# HELPERS
# =============================================================================

def save_checkpoint(path, model, optimizer, epoch, val_f1, threshold):
    torch.save({
        'epoch'              : epoch,
        'model_state_dict'   : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1'             : val_f1,
        'decision_threshold' : threshold,
        'config'             : CONFIG,
    }, path)


def make_lambda_sched(optimizer, warmup_epochs, total_epochs, eta_min=1e-4):
    """Single LambdaLR: linear warmup then cosine decay."""
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return 0.1 + 0.9 * ep / max(1, warmup_epochs)
        p = (ep - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return eta_min + (1 - eta_min) * 0.5 * (1 + math.cos(math.pi * p))
    return LambdaLR(optimizer, lr_lambda)


def make_train_loader(train_ds, ne_ratio, cfg):
    """Build DataLoader for a training phase (sampler or shuffle)."""
    if ne_ratio is not None:
        sampler = EngagementDataset.build_sampler(train_ds.labels,
                                                   ne_proportion=ne_ratio)
        return DataLoader(train_ds,
                          batch_size  = cfg['batch_size'],
                          sampler     = sampler,
                          shuffle     = False,
                          num_workers = cfg['num_workers'],
                          pin_memory  = cfg['pin_memory'],
                          persistent_workers=False,
                          drop_last   = True)
    else:
        return DataLoader(train_ds,
                          batch_size  = cfg['batch_size'],
                          shuffle     = True,
                          num_workers = cfg['num_workers'],
                          pin_memory  = cfg['pin_memory'],
                          persistent_workers=False,
                          drop_last   = True)


# =============================================================================
# MAIN
# =============================================================================

def train():
    device = CONFIG['device']
    SEP    = '=' * 70

    total_epochs = sum(ph['epochs'] for ph in PHASES)
    print('\n' + SEP)
    print('  ENGAGEMENT CLASSIFIER  --  v4 (Three-Phase Balanced Sampling)')
    print(f'  Device  : {device}')
    print(f'  Phases  : ' + '  |  '.join(
        f"{ph['name']}({ph['epochs']}ep)" for ph in PHASES))
    print(f'  Total   : {total_epochs} epochs')
    print(SEP)

    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # -------------------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------------------
    print('\nLoading datasets...')
    train_ds = EngagementDataset(CONFIG['train_csv'], split='train',
                                  image_size=CONFIG['image_size'])
    val_ds   = EngagementDataset(CONFIG['val_csv'],   split='val',
                                  image_size=CONFIG['image_size'])

    # Val loader is ALWAYS the real distribution (no sampler, no shuffle)
    val_loader = DataLoader(val_ds,
                            batch_size  = CONFIG['batch_size'],
                            shuffle     = False,
                            num_workers = CONFIG['num_workers'],
                            pin_memory  = CONFIG['pin_memory'],
                            persistent_workers=False)

    # -------------------------------------------------------------------------
    # Model + Prior Bias Init
    # -------------------------------------------------------------------------
    model = build_engagement_model(
        num_classes    = CONFIG['num_classes'],
        pretrained     = True,
        drop_rate      = CONFIG['drop_rate'],
        drop_path_rate = CONFIG['drop_path_rate'],
    )

    # Prior bias: model starts at real class distribution instead of 50/50.
    # Without this, the first 2-3 epochs are wasted learning log(94/6).
    counts    = np.bincount(train_ds.labels, minlength=2).astype(float)
    log_prior = np.log(counts / counts.sum())
    with torch.no_grad():
        model.head[-1].bias.data = torch.tensor(log_prior, dtype=torch.float32)
    print(f'\n  Prior bias init : {np.round(log_prior, 4)}  (log class freqs)')
    model = model.to(device)

    scaler = GradScaler() if device.startswith('cuda') else None

    best_val_f1    = 0.0
    best_threshold = CONFIG['eval_threshold']
    best_ckpt      = os.path.join(CONFIG['save_dir'], CONFIG['best_model_name'])
    history        = {k: [] for k in ('phase', 'train_loss', 'val_loss',
                                       'train_f1', 'val_f1', 'val_ne_f1',
                                       'val_eng_f1', 'thresh', 'thresh_f1')}
    global_ep = 0

    # =========================================================================
    # MAIN TRAINING LOOP -- three phases
    # =========================================================================

    for ph_idx, phase in enumerate(PHASES):
        ph_name   = phase['name']
        ph_epochs = phase['epochs']
        ph_ne     = phase['ne_ratio']

        print('\n' + '-' * 70)
        print(f'  PHASE {ph_idx+1}: {ph_name}  ({ph_epochs} epochs)')
        sampler_str = (f'{ph_ne*100:.0f}% NE per batch'
                       if ph_ne is not None else 'real 94:6 distribution')
        alpha_desc  = ('uniform' if phase['focal_alpha_pw'] == 0.0
                       else f'power={phase["focal_alpha_pw"]}')
        print(f'  Sampler : {sampler_str}')
        print(f'  Loss    : Focal(gamma={phase["focal_gamma"]}, alpha={alpha_desc})')
        if phase['frozen']:
            print(f'  Backbone: FROZEN  |  Head LR: {phase["lr_head"]:.0e}')
        else:
            print(f'  Backbone LR: {phase["lr_backbone"]:.0e}  '
                  f'|  Head LR: {phase["lr_head"]:.0e}')
        print('-' * 70)

        # ── Build per-phase DataLoader ─────────────────────────────────────
        train_loader = make_train_loader(train_ds, ph_ne, CONFIG)

        # ── Build per-phase criterion ──────────────────────────────────────
        alpha = compute_focal_alpha(train_ds.labels,
                                     power=phase['focal_alpha_pw']).to(device)
        criterion = FocalLoss(alpha=alpha, gamma=phase['focal_gamma'])

        ratio_str = f'{alpha[0].item():.2f}:{alpha[1].item():.2f}'
        print(f'  Alpha   : [{ratio_str}]  '
              f'(NE/E ratio: {alpha[0].item()/alpha[1].item():.2f}x)')

        # ── Freeze / unfreeze backbone ─────────────────────────────────────
        if phase['frozen']:
            model = freeze_backbone(model)
            optimizer = get_optimizer_frozen(model, phase['lr_head'], phase['wd'])
        else:
            if ph_idx > 0 and PHASES[ph_idx - 1]['frozen']:
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
            model = unfreeze_all(model)
            optimizer = get_optimizer_unfrozen(
                model,
                backbone_lr  = phase['lr_backbone'],
                head_lr      = phase['lr_head'],
                weight_decay = phase['wd'],
            )

        # ── LR Scheduler ──────────────────────────────────────────────────
        wu = phase['warmup_epochs']
        if wu > 0:
            sched = make_lambda_sched(optimizer, wu, ph_epochs)
        else:
            lr_ref = phase['lr_head']
            sched  = CosineAnnealingLR(optimizer, T_max=ph_epochs,
                                        eta_min=lr_ref * 0.05)

        # ── Early stopping only in Phase 3 ────────────────────────────────
        stopper = (EarlyStopping(CONFIG['patience'], CONFIG['min_delta'])
                   if ph_name == 'CALIBRATE' else None)

        # ── Epoch loop ────────────────────────────────────────────────────
        for epoch in range(1, ph_epochs + 1):
            global_ep += 1

            tr_loss, _, tr_f1, tr_ne, _, _, _, _ = run_epoch(
                model, train_loader, optimizer, criterion, scaler,
                device, CONFIG['grad_clip'], training=True)
            sched.step()

            va_loss, _, va_f1, va_ne, va_eng, va_lbl, _, va_probs = \
                run_epoch(model, val_loader, None, criterion, None,
                          device, training=False)

            ep_thresh, ep_f1 = sweep_threshold_fast(va_probs, va_lbl)

            # Track history
            lr_now = optimizer.param_groups[0]['lr']
            for k, v in [('phase', ph_name), ('train_loss', tr_loss),
                          ('val_loss', va_loss), ('train_f1', tr_f1),
                          ('val_f1', va_f1), ('val_ne_f1', va_ne),
                          ('val_eng_f1', va_eng), ('thresh', ep_thresh),
                          ('thresh_f1', ep_f1)]:
                history[k].append(v)

            flag = ''
            if ep_f1 > best_val_f1:
                best_val_f1    = ep_f1
                best_threshold = ep_thresh
                save_checkpoint(best_ckpt, model, optimizer, global_ep,
                                ep_f1, threshold=ep_thresh)
                flag = '  <- BEST'

            print('  Ep {:02d}/{:02d} [{}]'
                  '  Tr={:.4f}(f1={:.3f})'
                  '  | Va={:.4f} f1@t={:.2f}->{:.4f}'
                  '  [NE={:.3f} E={:.3f}]'
                  '  LR={:.1e}{}'.format(
                  global_ep, total_epochs, ph_name[:3],
                  tr_loss, tr_f1,
                  va_loss, ep_thresh, ep_f1,
                  va_ne, va_eng,
                  lr_now, flag))

            if stopper is not None and stopper.step(ep_f1):
                print(f'\n  Early stopping: no improvement for '
                      f'{CONFIG["patience"]} epochs.')
                break

    # =========================================================================
    # POST-TRAINING: fine-grained threshold verification on best checkpoint
    # =========================================================================
    print('\n' + '-' * 70)
    print('  POST-TRAINING THRESHOLD VERIFICATION (35 steps, fresh pass)')
    print('-' * 70)

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    all_probs, all_lbl = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='  Verify', leave=False):
            logits = model(images.to(device))
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_lbl.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_lbl   = np.array(all_lbl)

    best_f1_final, best_t_final = 0.0, 0.5
    sweep_rows = []
    for t in np.linspace(0.05, 0.75, 35):
        preds = (all_probs >= t).astype(int)
        f1    = f1_score(all_lbl, preds, average='macro', zero_division=0)
        f1s   = f1_score(all_lbl, preds, average=None, labels=[0,1], zero_division=0)
        sweep_rows.append((t, f1, f1s[0], f1s[1]))
        if f1 > best_f1_final:
            best_f1_final, best_t_final = f1, float(t)

    # Print final sweep summary
    t05_f1 = next((r[1] for r in sweep_rows if abs(r[0]-0.5) < 0.03), float('nan'))
    print(f'  threshold=0.50 (argmax)   -> Macro F1 = {t05_f1:.4f}')
    print(f'  threshold={best_t_final:.2f} (optimal)  -> Macro F1 = {best_f1_final:.4f}')

    opt_preds = (all_probs >= best_t_final).astype(int)
    print(f'\nClassification report at threshold={best_t_final:.2f}:')
    print(classification_report(all_lbl, opt_preds,
                                 target_names=['Not Engaged', 'Engaged'],
                                 digits=4))

    # Update checkpoint with final threshold
    ckpt['decision_threshold']  = best_t_final
    ckpt['val_f1_at_threshold'] = best_f1_final
    torch.save(ckpt, best_ckpt)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print('\n' + SEP)
    print('  TRAINING COMPLETE  (v4 Three-Phase)')
    print(f'  Best threshold-F1 (training)  : {best_val_f1:.4f}')
    print(f'  Best threshold-F1 (post-check): {best_f1_final:.4f}  '
          f'(t={best_t_final:.2f})')
    print(f'  Checkpoint: {best_ckpt}')
    print(SEP)

    return history, best_t_final


if __name__ == '__main__':
    train()
