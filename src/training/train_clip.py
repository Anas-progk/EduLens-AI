"""
train_clip.py -- Three-phase clip-level engagement training (v2 -- overfitting-hardened).

Architecture: Swin-Tiny backbone + TemporalTransformer + classification head
Input:        (B=4, T=8, C=3, H=224, W=224) clips
Output:       (B, 2) engagement logits

KEY LESSONS FROM RUNS 4 + 5
============================
Run 4 (original merged_train.csv, no synthetic):
  BOOTSTRAP Ep03: macro_F1=0.8485, NE_F1=0.7785  <- BEST ever checkpoint
  ALIGN Ep06-13:  val F1 oscillated 0.58-0.69     <- full unfreeze -> overfitting
  CALIBRATE:      val F1 ~0.60-0.62               <- further degradation

Run 5 (934 aggressive synthetic NE):
  BOOTSTRAP Ep03: macro_F1=0.7621, NE_F1=0.6281   <- WORSE than run 4
  Same degradation in ALIGN/CALIBRATE.

Root causes identified:
  1. BOOTSTRAP had no early stopping -> ran past its peak (ep03), then ep05 collapsed.
  2. ALIGN unfroze all 27.5M backbone params on ~1000 clips -> catastrophic memorisation.
     Train F1 -> 0.984 while val F1 dropped 0.85 -> 0.60.
  3. Aggressive synthetic data taught the model "distortion = NE" not "behaviour = NE".

Fixes in v2:
  1. BOOTSTRAP now has early stopping (patience=3) - saves best, stops when done.
  2. ALIGN uses SELECTIVE UNFREEZE: only last Swin stage (~8.6M) + temporal+head.
     Total trainable in ALIGN ~15.9M vs 34.8M -> far less memorisation capacity.
     patience=2: exits immediately if no improvement in 2 epochs.
  3. CALIBRATE refreezes backbone. Only temporal+head fine-tune at very low LR.
  4. Cosine LR decay within every phase -> prevents late-epoch instability.
  5. Per-phase patience stored in PHASES dict for full control.

Target: macro F1 >= 0.82, NE F1 >= 0.72.

Usage (Colab):
  python src/training/train_clip.py
"""

import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CONFIG
from src.data.clip_dataset import ClipDataset
from src.models.swin_clip_model import (
    build_clip_model,
    freeze_backbone,
    unfreeze_all,
    partial_unfreeze_last_stages,
    get_optimizer_frozen,
    get_optimizer_unfrozen,
    get_optimizer_partial,
)


# ============================================================
# Phase definitions  (v2 -- overfitting-hardened)
# ============================================================

PHASES = [
    # Phase 1: BOOTSTRAP
    # Backbone frozen. Temporal head + classifier learn engagement dynamics.
    # 50:50 NE sampler gives balanced signal from the start.
    # early_stop=True with patience=3: stops after best epoch instead of
    # always running 5 epochs. In run 4, best was Ep03; with patience=3
    # we'd stop at Ep05 (still saving the ep03 checkpoint).
    # Cosine LR decay from 3e-4 down to ~1.5e-5 over 7 epochs.
    {
        "name"            : "BOOTSTRAP",
        "epochs"          : 7,
        "frozen"          : True,
        "partial_unfreeze": False,
        "ne_ratio"        : 0.50,
        "focal_gamma"     : 1.5,
        "focal_alpha_pw"  : 0.0,
        "lr_head"         : 3e-4,
        "warmup_epochs"   : 0,
        "early_stop"      : True,
        "patience"        : 3,
    },
    # Phase 2: ALIGN
    # SELECTIVE unfreeze: only Swin stage-3 (768-dim, ~8.6M) + temporal + head.
    # Early stages frozen -> preserves generalised visual features.
    # Backbone LR=2e-6 (very low) -- minimal perturbation of ImageNet weights.
    # patience=2: if ALIGN can't improve BOOTSTRAP best in 2 epochs, exit.
    {
        "name"            : "ALIGN",
        "epochs"          : 6,
        "frozen"          : False,
        "partial_unfreeze": True,
        "n_stages"        : 1,
        "ne_ratio"        : None,
        "focal_gamma"     : 2.0,
        "focal_alpha_pw"  : 0.0,
        "lr_backbone"     : 2e-6,
        "lr_temporal"     : 8e-6,
        "lr_head"         : 1.5e-5,
        "warmup_epochs"   : 1,
        "early_stop"      : True,
        "patience"        : 2,
    },
    # Phase 3: CALIBRATE
    # Backbone refrozen. Only temporal+head adjust at minimal LR.
    # Goal: fine-tune decision boundary to natural class distribution.
    # alpha_pw=0.25: mild NE up-weighting without collapsing NE precision.
    {
        "name"            : "CALIBRATE",
        "epochs"          : 6,
        "frozen"          : True,
        "partial_unfreeze": False,
        "ne_ratio"        : None,
        "focal_gamma"     : 2.0,
        "focal_alpha_pw"  : 0.25,
        "lr_head"         : 5e-6,
        "warmup_epochs"   : 0,
        "early_stop"      : True,
        "patience"        : 3,
    },
]


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)"""

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs     = torch.exp(log_probs)

        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - target_probs) ** self.gamma
        alpha_t      = self.alpha[targets]

        loss = -alpha_t * focal_weight * target_log_probs
        return loss.mean()


def make_focal_loss(train_ds, alpha_power: float, gamma: float, device) -> FocalLoss:
    """Build FocalLoss with alpha derived from class frequencies."""
    counts = np.bincount(train_ds.labels, minlength=2).astype(float)

    if alpha_power == 0.0:
        alpha = torch.ones(2, dtype=torch.float32)
    else:
        inv_freq = 1.0 / counts
        inv_freq = inv_freq ** alpha_power
        inv_freq = inv_freq / inv_freq.sum() * 2
        alpha = torch.tensor(inv_freq, dtype=torch.float32)

    alpha  = alpha.to(device)
    ne_pct = counts[0] / counts.sum() * 100
    print(f"  FocalLoss: gamma={gamma:.1f}  alpha_pw={alpha_power:.2f}  "
          f"alpha=[{alpha[0]:.3f}, {alpha[1]:.3f}]  NE={ne_pct:.1f}%")
    return FocalLoss(alpha=alpha, gamma=gamma)


# ============================================================
# DataLoader factory
# ============================================================

def make_train_loader(train_ds, ne_ratio, cfg: dict) -> DataLoader:
    """Training DataLoader with optional NE-balanced sampler."""
    if ne_ratio is not None:
        sampler = ClipDataset.build_sampler(train_ds.labels, ne_proportion=ne_ratio)
        return DataLoader(
            train_ds,
            batch_size  = cfg["clip_batch_size"],
            sampler     = sampler,
            num_workers = cfg["num_workers"],
            pin_memory  = cfg["pin_memory"],
            drop_last   = True,
        )
    else:
        return DataLoader(
            train_ds,
            batch_size  = cfg["clip_batch_size"],
            shuffle     = True,
            num_workers = cfg["num_workers"],
            pin_memory  = cfg["pin_memory"],
            drop_last   = True,
        )


# ============================================================
# Threshold sweep
# ============================================================

def sweep_threshold_fast(probs: np.ndarray, labels: np.ndarray, n_steps: int = 40):
    """Fast numpy threshold sweep. Returns (best_threshold, best_macro_f1)."""
    thresholds = np.linspace(0.10, 0.80, n_steps)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


# ============================================================
# LR scheduling: warmup + cosine decay
# ============================================================

def get_lr_scale(ep: int, warmup_epochs: int, total_epochs: int) -> float:
    """
    Combined linear warmup + cosine decay.
    - First warmup_epochs: linearly ramp from 0.1 to 1.0
    - Remaining: cosine decay from 1.0 to 0.05 (floor)
    """
    if warmup_epochs > 0 and ep < warmup_epochs:
        return 0.1 + 0.9 * (ep / warmup_epochs)
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    progress      = (ep - warmup_epochs) / cosine_epochs
    cosine_val    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return max(cosine_val, 0.05)


# ============================================================
# Single epoch
# ============================================================

def run_epoch(model, loader, criterion, optimizer, device, is_train: bool, grad_clip: float):
    """Run one training or evaluation epoch."""
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for clips, labels in loader:
            clips  = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(clips)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()

            with torch.no_grad():
                probs = F.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            total_loss += loss.item()
            n_batches  += 1

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    avg_loss  = total_loss / max(n_batches, 1)
    macro_f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=[0, 1])
    ne_f1     = float(per_class[0]) if len(per_class) > 0 else 0.0
    e_f1      = float(per_class[1]) if len(per_class) > 1 else 0.0

    return avg_loss, macro_f1, ne_f1, e_f1, all_probs, all_labels


# ============================================================
# Prior bias initialisation
# ============================================================

def init_prior_bias(model, train_ds):
    """Initialise classification head bias to log-prior."""
    counts = np.bincount(train_ds.labels, minlength=2).astype(float)
    priors = counts / counts.sum()
    priors = np.clip(priors, 1e-6, 1 - 1e-6)
    log_prior = np.log(priors)
    with torch.no_grad():
        model.head[-1].bias.data = torch.tensor(
            log_prior, dtype=torch.float32, device=model.head[-1].bias.device
        )
    print(f"  Prior bias init: NE={priors[0]:.4f}  E={priors[1]:.4f}")


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train(cfg: dict = None):
    if cfg is None:
        cfg = CONFIG

    clip_cfg = {
        "clip_batch_size": cfg.get("clip_batch_size", 4),
        "n_frames"       : cfg.get("n_frames_clip", 8),
        "num_workers"    : cfg.get("num_workers", 0),
        "pin_memory"     : cfg.get("pin_memory", True),
        "image_size"     : cfg.get("image_size", 224),
        "num_classes"    : cfg.get("num_classes", 2),
        "drop_rate"      : cfg.get("drop_rate", 0.30),
        "drop_path_rate" : cfg.get("drop_path_rate", 0.15),
        "grad_clip"      : cfg.get("grad_clip", 0.3),
        "save_dir"       : cfg.get("save_dir", "weights"),
        "patience"       : cfg.get("patience", 8),
        "min_delta"      : cfg.get("min_delta", 0.001),
        "train_csv"      : cfg.get("train_csv", "data/splits/merged_train.csv"),
        "val_csv"        : cfg.get("val_csv",   "data/splits/custom_val.csv"),
    }

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"\n{'='*65}")
    print(f"  CLIP-LEVEL ENGAGEMENT TRAINING v2 (overfitting-hardened)")
    print(f"  Swin-Tiny + TemporalTransformer")
    print(f"  Device  : {device}")
    print(f"  TrainCSV: {clip_cfg['train_csv']}")
    print(f"{'='*65}")

    # Datasets
    print("\n[DATA]")
    train_ds = ClipDataset(
        clip_cfg["train_csv"], split="train",
        image_size=clip_cfg["image_size"], n_frames=clip_cfg["n_frames"],
    )
    val_ds = ClipDataset(
        clip_cfg["val_csv"], split="val",
        image_size=clip_cfg["image_size"], n_frames=clip_cfg["n_frames"],
    )
    val_loader = DataLoader(
        val_ds, batch_size=clip_cfg["clip_batch_size"],
        shuffle=False, num_workers=clip_cfg["num_workers"],
        pin_memory=clip_cfg["pin_memory"],
    )

    # Model
    print("\n[MODEL]")
    model = build_clip_model(
        num_classes    = clip_cfg["num_classes"],
        pretrained     = True,
        drop_rate      = clip_cfg["drop_rate"],
        drop_path_rate = clip_cfg["drop_path_rate"],
        n_frames       = clip_cfg["n_frames"],
    ).to(device)

    init_prior_bias(model, train_ds)

    # Checkpointing
    os.makedirs(clip_cfg["save_dir"], exist_ok=True)
    best_ckpt      = os.path.join(clip_cfg["save_dir"], "best_clip_model.pth")
    best_macro_f1  = 0.0
    best_threshold = 0.5
    global_epoch   = 0

    # Training phases
    for phase in PHASES:
        pname          = phase["name"]
        abbr           = pname[:3].upper()
        phase_patience = phase.get("patience", clip_cfg["patience"])
        es_counter     = 0
        n_epochs       = phase["epochs"]

        print(f"\n{'='*65}")
        print(f"  Phase: {pname}  (max {n_epochs} epochs | patience={phase_patience})")
        print(
            f"  frozen={phase['frozen']}  "
            f"partial_unfreeze={phase.get('partial_unfreeze', False)}  "
            f"ne_ratio={phase['ne_ratio']}  "
            f"gamma={phase['focal_gamma']}  alpha_pw={phase['focal_alpha_pw']}"
        )
        print(f"{'='*65}")

        # Freeze / unfreeze
        if phase["frozen"]:
            freeze_backbone(model)
        elif phase.get("partial_unfreeze", False):
            partial_unfreeze_last_stages(model, n_stages=phase.get("n_stages", 1))
        else:
            unfreeze_all(model)

        # Criterion
        criterion = make_focal_loss(
            train_ds, alpha_power=phase["focal_alpha_pw"],
            gamma=phase["focal_gamma"], device=device,
        )

        # DataLoader
        train_loader = make_train_loader(train_ds, phase["ne_ratio"], clip_cfg)

        # Optimizer
        if phase["frozen"]:
            optimizer = get_optimizer_frozen(model, lr=phase["lr_head"], weight_decay=1e-2)
        elif phase.get("partial_unfreeze", False):
            optimizer = get_optimizer_partial(
                model,
                backbone_lr=phase["lr_backbone"],
                temporal_lr=phase["lr_temporal"],
                head_lr    =phase["lr_head"],
                weight_decay=1e-2,
            )
        else:
            optimizer = get_optimizer_unfrozen(
                model,
                backbone_lr=phase["lr_backbone"],
                temporal_lr=phase["lr_temporal"],
                head_lr    =phase["lr_head"],
                weight_decay=1e-2,
            )

        # Store base LRs for cosine decay
        for pg in optimizer.param_groups:
            pg["base_lr"] = pg["lr"]

        warmup = phase.get("warmup_epochs", 0)

        for ep in range(n_epochs):
            t0 = time.time()
            global_epoch += 1

            # Warmup + cosine LR scale
            lr_scale = get_lr_scale(ep, warmup, n_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = pg["base_lr"] * lr_scale

            # Train
            tr_loss, tr_f1, _, _, _, _ = run_epoch(
                model, train_loader, criterion, optimizer,
                device, is_train=True, grad_clip=clip_cfg["grad_clip"]
            )

            # Validate
            val_loss, _, _, _, val_probs, val_labels = run_epoch(
                model, val_loader, criterion, None,
                device, is_train=False, grad_clip=0.0
            )

            # Threshold sweep
            best_t, val_macro_f1 = sweep_threshold_fast(val_probs, val_labels)
            val_preds = (val_probs >= best_t).astype(int)
            per_class = f1_score(
                val_labels, val_preds, average=None, zero_division=0, labels=[0, 1]
            )
            ne_f1 = float(per_class[0]) if len(per_class) > 0 else 0.0
            e_f1  = float(per_class[1]) if len(per_class) > 1 else 0.0

            elapsed    = time.time() - t0
            current_lr = optimizer.param_groups[-1]["lr"]

            print(
                f"Ep {global_epoch:02d} [{abbr}] "
                f"Tr={tr_loss:.4f}(f1={tr_f1:.3f}) | "
                f"Va={val_loss:.4f} f1@t={best_t:.2f}->{val_macro_f1:.4f} "
                f"[NE={ne_f1:.3f} E={e_f1:.3f}] "
                f"LR={current_lr:.1e} t={elapsed:.0f}s"
            )

            # Save checkpoint if best
            if val_macro_f1 > best_macro_f1 + clip_cfg["min_delta"]:
                best_macro_f1  = val_macro_f1
                best_threshold = best_t
                es_counter     = 0
                torch.save(
                    {
                        "epoch"       : global_epoch,
                        "phase"       : pname,
                        "model_state" : model.state_dict(),
                        "macro_f1"    : val_macro_f1,
                        "ne_f1"       : ne_f1,
                        "threshold"   : best_threshold,
                        "n_frames"    : clip_cfg["n_frames"],
                        "drop_rate"   : clip_cfg["drop_rate"],
                        "drop_path"   : clip_cfg["drop_path_rate"],
                    },
                    best_ckpt,
                )
                print(
                    f"  *** New best: macro_F1={val_macro_f1:.4f}  "
                    f"NE_F1={ne_f1:.4f}  threshold={best_threshold:.2f}  -> saved ***"
                )
            else:
                es_counter += 1

            # Per-phase early stopping
            if phase.get("early_stop", False) and es_counter >= phase_patience:
                print(
                    f"  Early stopping [{pname}]: no improvement for "
                    f"{phase_patience} epoch(s). Best so far: {best_macro_f1:.4f}"
                )
                break

    # Final summary
    print(f"\n{'='*65}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best val macro-F1 : {best_macro_f1:.4f}")
    print(f"  Best threshold    : {best_threshold:.2f}")
    print(f"  Checkpoint        : {best_ckpt}")
    print(f"{'='*65}")

    print("\n[FINAL VAL REPORT -- best checkpoint]")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    final_criterion = make_focal_loss(
        train_ds,
        alpha_power=PHASES[-1]["focal_alpha_pw"],
        gamma=PHASES[-1]["focal_gamma"],
        device=device,
    )
    _, _, _, _, val_probs, val_labels = run_epoch(
        model, val_loader, final_criterion, None,
        device, is_train=False, grad_clip=0.0
    )
    val_preds = (val_probs >= best_threshold).astype(int)
    print(classification_report(
        val_labels, val_preds,
        target_names=["Not Engaged", "Engaged"],
        digits=4,
    ))

    return model, best_threshold


if __name__ == "__main__":
    train()
