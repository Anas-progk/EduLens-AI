# =============================================================================
# STANDALONE THRESHOLD SWEEP  — paste into a Colab cell and run immediately
# No retraining needed.  Loads your existing checkpoint, sweeps thresholds
# on the val set, and prints the full classification report at the optimal cut.
#
# Expected output:
#   threshold=0.50  ->  Macro-F1 ≈ 0.52  (what you already have)
#   threshold=0.20-0.40  ->  Macro-F1 ≈ 0.62-0.70  (estimated improvement)
# =============================================================================

import sys, os, math
sys.path.append('/content')   # adjust if your project root differs

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

# ── Project imports ────────────────────────────────────────────────────────────
from src.config import CONFIG
from src.data.dataset import EngagementDataset
from src.models.swin_model import build_engagement_model

# =============================================================================
# SETTINGS  (edit these two paths if needed)
# =============================================================================
CHECKPOINT = CONFIG['save_dir'] + '/' + CONFIG['best_model_name']
# e.g.  '/content/drive/MyDrive/engagement_weights/best_engagement_model.pth'

VAL_CSV    = CONFIG['val_csv']
DEVICE     = CONFIG['device']
IMAGE_SIZE = CONFIG['image_size']
BATCH_SIZE = CONFIG['batch_size']

# How many threshold steps to try between 0.05 and 0.75
N_STEPS    = 35

# =============================================================================
# LOAD MODEL
# =============================================================================
print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)

model = build_engagement_model(
    num_classes    = CONFIG['num_classes'],
    pretrained     = False,          # weights come from checkpoint
    drop_rate      = CONFIG['drop_rate'],
    drop_path_rate = CONFIG['drop_path_rate'],
)
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(DEVICE)
model.eval()

saved_epoch = ckpt.get('epoch', '?')
saved_f1    = ckpt.get('val_f1', float('nan'))
saved_thr   = ckpt.get('decision_threshold', 0.5)
print(f"  Checkpoint epoch : {saved_epoch}")
print(f"  Saved val F1     : {saved_f1:.4f}  (argmax threshold={saved_thr:.2f})")

# =============================================================================
# BUILD VAL LOADER  (deterministic — no augmentation)
# =============================================================================
val_ds = EngagementDataset(VAL_CSV, split='val', image_size=IMAGE_SIZE)
val_loader = DataLoader(
    val_ds,
    batch_size   = BATCH_SIZE,
    shuffle      = False,
    num_workers  = 2,
    pin_memory   = DEVICE.startswith('cuda'),
    persistent_workers = False,
)

# =============================================================================
# COLLECT PROBABILITIES
# =============================================================================
print("\nRunning inference on val set...")
all_probs  = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="  Inference"):
        images = images.to(DEVICE)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1)[:, 1]   # P(Engaged)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)

print(f"\n  Total val frames  : {len(all_labels):,}")
print(f"  Not Engaged (0)   : {(all_labels==0).sum():,}  "
      f"({100*(all_labels==0).mean():.1f}%)")
print(f"  Engaged (1)       : {(all_labels==1).sum():,}  "
      f"({100*(all_labels==1).mean():.1f}%)")
print(f"\n  P(Engaged) stats:")
print(f"    mean  = {all_probs.mean():.4f}")
print(f"    median= {np.median(all_probs):.4f}")
print(f"    p10   = {np.percentile(all_probs, 10):.4f}")
print(f"    p25   = {np.percentile(all_probs, 25):.4f}")

# =============================================================================
# THRESHOLD SWEEP
# =============================================================================
print(f"\n{'─'*60}")
print(f"  THRESHOLD SWEEP  ({N_STEPS} steps, 0.05 → 0.75)")
print(f"{'─'*60}")
print(f"  {'Threshold':>9}  {'Macro-F1':>9}  {'NE F1':>9}  {'E F1':>9}  "
      f"{'NE Prec':>9}  {'NE Rec':>9}")
print(f"  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")

best_macro_f1 = 0.0
best_thresh   = 0.5
best_results  = {}

for t in np.linspace(0.05, 0.75, N_STEPS):
    preds    = (all_probs >= t).astype(int)
    macro_f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
    f1_each  = f1_score(all_labels, preds, average=None, labels=[0,1], zero_division=0)

    from sklearn.metrics import precision_score, recall_score
    ne_prec = precision_score(all_labels, preds, pos_label=0, zero_division=0)
    ne_rec  = recall_score(all_labels,    preds, pos_label=0, zero_division=0)

    marker = " ◄" if macro_f1 > best_macro_f1 else ""
    print(f"  {t:>9.3f}  {macro_f1:>9.4f}  {f1_each[0]:>9.4f}  {f1_each[1]:>9.4f}  "
          f"{ne_prec:>9.4f}  {ne_rec:>9.4f}{marker}")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_thresh   = t
        best_results  = {
            'preds'    : preds,
            'macro_f1' : macro_f1,
            'ne_f1'    : f1_each[0],
            'eng_f1'   : f1_each[1],
        }

# =============================================================================
# BEST THRESHOLD REPORT
# =============================================================================
print(f"\n{'='*60}")
print(f"  BEST THRESHOLD : {best_thresh:.3f}  →  Macro-F1 = {best_macro_f1:.4f}")
print(f"  (was {saved_thr:.3f} → {saved_f1:.4f} with argmax,  "
      f"gain = {best_macro_f1 - saved_f1:+.4f})")
print(f"{'='*60}")
print(f"\nClassification report at threshold={best_thresh:.3f}:")
print(classification_report(
    all_labels, best_results['preds'],
    target_names=['Not Engaged', 'Engaged'],
    digits=4,
))

# =============================================================================
# SAVE UPDATED THRESHOLD INTO CHECKPOINT
# =============================================================================
ckpt['decision_threshold']   = float(best_thresh)
ckpt['val_f1_at_threshold']  = float(best_macro_f1)
torch.save(ckpt, CHECKPOINT)
print(f"✅  Checkpoint updated with decision_threshold={best_thresh:.3f}")
print(f"   (val_f1_at_threshold={best_macro_f1:.4f})")
print(f"\n   Add to CONFIG:  'eval_threshold': {best_thresh:.2f}")
