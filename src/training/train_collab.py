"""
train_collab.py -- Phase 2 collaboration head training.

Trains ONLY the CollaborationHead (2.4M params).
The SwinClipModel (engagement backbone) is NEVER updated — Phase 1 weights stay frozen.

Training flow:
  Phase 2-A: Bootstrap CollaborationHead
    - Features pre-extracted (frozen backbone)
    - Pure head training on cached features
    - AdamW, BCE with pos_weight, early stopping (patience=10)
    - Expected: 25-40 epochs

  Phase 2-B (OPTIONAL — only if val F1 < 0.60):
    - This should NOT be needed with good data
    - Emergency: unfreeze ONLY engagement temporal transformer at LR=1e-7
    - Keep backbone (Swin layers) frozen even in this phase

Usage:
  # Standard training (recommended — run on Colab for speed)
  python src/training/train_collab.py

  # With explicit paths
  python src/training/train_collab.py --splits_dir data/collab_splits/ --epochs 50

  # Resume from checkpoint
  python src/training/train_collab.py --resume weights/collab_checkpoint.pth
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.collaboration_head import build_collab_head, collab_loss
from src.data.collab_dataset import CollabPairDataset, SPLITS_DIR


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "splits_dir":   SPLITS_DIR,
    "weights_dir":  "weights",
    "batch_size":   16,        # Small enough for CPU (feature-level, no backbone)
    "lr":           3e-4,      # AdamW LR for CollaborationHead
    "weight_decay": 1e-4,
    "epochs":       60,
    "patience":     10,        # Early stop patience
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "collab_threshold": 0.50,  # Sigmoid threshold for binary decision (tuned at end)
    "pos_weight":   1.5,       # BCEWithLogitsLoss pos_weight (adjust if class imbalanced)
    "seed":         42,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate(model, loader, device, threshold=0.50):
    """
    Evaluate CollaborationHead on a dataloader.
    Returns dict of metrics.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for feat_A, feat_B, signals, labels in loader:
            feat_A  = feat_A.to(device)
            feat_B  = feat_B.to(device)
            signals = signals.to(device)

            logits = model(feat_A, feat_B, signals)   # (B,)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    probs_arr  = np.array(all_probs)
    labels_arr = np.array(all_labels, dtype=int)
    preds_arr  = (probs_arr >= threshold).astype(int)

    f1   = f1_score(labels_arr, preds_arr, zero_division=0)
    prec = precision_score(labels_arr, preds_arr, zero_division=0)
    rec  = recall_score(labels_arr, preds_arr, zero_division=0)
    acc  = float((preds_arr == labels_arr).mean())
    cm   = confusion_matrix(labels_arr, preds_arr)

    return {
        "f1":        f1,
        "precision": prec,
        "recall":    rec,
        "accuracy":  acc,
        "cm":        cm,
        "n_pos_pred": int(preds_arr.sum()),
        "n_pos_true": int(labels_arr.sum()),
    }


def threshold_sweep(model, loader, device):
    """
    Find the best classification threshold on validation set.
    Tests thresholds from 0.30 to 0.80 and returns the one with max F1.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for feat_A, feat_B, signals, labels in loader:
            logits = model(feat_A.to(device), feat_B.to(device), signals.to(device))
            all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    labels_arr = np.array(all_labels, dtype=int)
    probs_arr  = np.array(all_probs)

    best_thresh = 0.50
    best_f1     = 0.0

    for thresh in np.arange(0.30, 0.81, 0.05):
        preds = (probs_arr >= thresh).astype(int)
        f1    = f1_score(labels_arr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = float(thresh)

    return best_thresh, best_f1


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(config: dict):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device     = config["device"]
    splits_dir = config["splits_dir"]
    weights_dir = Path(config["weights_dir"])
    weights_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Phase 2 — CollaborationHead Training")
    print(f"  Device:     {device}")
    print(f"  Splits dir: {splits_dir}")
    print(f"{'='*60}\n")

    # ── Load datasets ──────────────────────────────────────────────────────
    try:
        train_ds = CollabPairDataset(split='train', splits_dir=splits_dir)
        val_ds   = CollabPairDataset(split='val',   splits_dir=splits_dir)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nHave you completed the data pipeline?")
        print("  Step 1: python src/data/collab_video_processor.py")
        print("  Step 2: python src/data/collab_annotator.py")
        print("  Step 3: python src/data/collab_dataset.py --extract_features --model_path weights/best_clip_model.pth")
        print("  Step 4: python src/data/collab_dataset.py --build_splits")
        return

    if len(train_ds) < 50:
        print(f"WARNING: Only {len(train_ds)} training pairs. Target ≥ 200.")
        print("Continue annotating more pairs before training.")

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── Build model ────────────────────────────────────────────────────────
    model = build_collab_head().to(device)

    # Compute pos_weight from dataset balance
    pos_weight = train_ds.get_class_weights()
    pos_weight = min(pos_weight, 3.0)   # cap at 3x to avoid extreme weighting
    print(f"\n  Dataset pos_weight (N/C ratio): {pos_weight:.2f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = config["lr"],
        weight_decay = config["weight_decay"],
    )

    # Cosine annealing LR: gradually reduces LR toward 0 at epoch=epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )

    # ── Resume from checkpoint ─────────────────────────────────────────────
    start_epoch  = 1
    best_val_f1  = 0.0
    best_thresh  = 0.50
    no_improve   = 0

    if config.get("resume"):
        ckpt_path = config["resume"]
        if Path(ckpt_path).exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch  = ckpt.get("epoch", 1) + 1
            best_val_f1  = ckpt.get("best_val_f1", 0.0)
            best_thresh  = ckpt.get("best_thresh", 0.50)
            no_improve   = ckpt.get("no_improve", 0)
            print(f"  Resumed from {ckpt_path} (epoch {start_epoch-1}, val_f1={best_val_f1:.4f})")
        else:
            print(f"  WARNING: checkpoint {ckpt_path} not found, starting fresh")

    print(f"\n{'Ep':>4}  {'TrainLoss':>9}  {'ValF1':>7}  {'Prec':>6}  "
          f"{'Rec':>6}  {'Acc':>6}  {'LR':>8}  {'Best':>7}")
    print("-" * 65)

    # ── Training epochs ────────────────────────────────────────────────────
    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for feat_A, feat_B, signals, labels in train_loader:
            feat_A  = feat_A.to(device)
            feat_B  = feat_B.to(device)
            signals = signals.to(device)
            labels  = labels.to(device)

            logits = model(feat_A, feat_B, signals)
            loss   = collab_loss(logits, labels, pos_weight=pos_weight)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (small head can have explosive gradients on early epochs)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        lr_now   = optimizer.param_groups[0]['lr']

        # Evaluate on val
        val_metrics = evaluate(model, val_loader, device, threshold=best_thresh)
        val_f1   = val_metrics["f1"]
        val_prec = val_metrics["precision"]
        val_rec  = val_metrics["recall"]
        val_acc  = val_metrics["accuracy"]

        # Track best
        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            no_improve  = 0
            # Update threshold every 5 epochs
            if epoch % 5 == 0:
                best_thresh, _ = threshold_sweep(model, val_loader, device)
            # Save best model
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1":        best_val_f1,
                "best_thresh":        best_thresh,
                "no_improve":         no_improve,
                "config":             config,
            }, str(weights_dir / "best_collab_model.pth"))
        else:
            no_improve += 1

        marker = " ★" if improved else ""

        print(f"{epoch:>4}  {avg_loss:>9.4f}  {val_f1:>7.4f}  {val_prec:>6.4f}  "
              f"{val_rec:>6.4f}  {val_acc:>6.4f}  {lr_now:>8.2e}  "
              f"{best_val_f1:>7.4f}{marker}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1":        best_val_f1,
                "best_thresh":        best_thresh,
                "no_improve":         no_improve,
                "config":             config,
            }, str(weights_dir / "collab_checkpoint.pth"))

        # Early stopping
        if no_improve >= config["patience"]:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {no_improve} epochs)")
            break

    # ── Final evaluation ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Best val F1: {best_val_f1:.4f}  (threshold={best_thresh:.2f})")

    # Load best model for final eval
    best_ckpt = weights_dir / "best_collab_model.pth"
    if best_ckpt.exists():
        ckpt = torch.load(str(best_ckpt), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    # Final threshold sweep
    best_thresh, best_thresh_f1 = threshold_sweep(model, val_loader, device)
    print(f"  Best threshold (sweep): {best_thresh:.2f}  F1={best_thresh_f1:.4f}")

    # Final metrics at best threshold
    val_final = evaluate(model, val_loader, device, threshold=best_thresh)
    print(f"\n  Validation metrics (threshold={best_thresh:.2f}):")
    print(f"    F1:        {val_final['f1']:.4f}")
    print(f"    Precision: {val_final['precision']:.4f}")
    print(f"    Recall:    {val_final['recall']:.4f}")
    print(f"    Accuracy:  {val_final['accuracy']:.4f}")
    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    print(f"    [TN FP]")
    print(f"    [FN TP]")
    print(f"    {val_final['cm'].tolist()}")

    # Update checkpoint with best threshold
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_val_f1":      best_val_f1,
        "best_thresh":      best_thresh,
        "config":           config,
    }, str(weights_dir / "best_collab_model.pth"))

    print(f"\n  Saved: {weights_dir}/best_collab_model.pth")

    # Performance assessment
    if val_final['f1'] >= 0.75:
        print(f"\n  ✓ EXCELLENT: F1={val_final['f1']:.3f} ≥ 0.75 — model is ready for inference")
    elif val_final['f1'] >= 0.65:
        print(f"\n  ✓ GOOD: F1={val_final['f1']:.3f} — acceptable for deployment")
        print(f"    Consider annotating 100+ more pairs to push above 0.75")
    else:
        print(f"\n  ⚠ LOW: F1={val_final['f1']:.3f} < 0.65")
        print(f"    Likely causes:")
        print(f"      1. Too few training pairs (need ≥ 300 usable C+N pairs)")
        print(f"      2. Label noise (review annotation guidelines)")
        print(f"      3. Class imbalance (check C vs N ratio)")

    print(f"\nNext step:")
    print(f"  python run_collab_inference.py")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 collaboration head")
    parser.add_argument("--splits_dir",  default=DEFAULT_CONFIG["splits_dir"])
    parser.add_argument("--weights_dir", default=DEFAULT_CONFIG["weights_dir"])
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--patience",    type=int,   default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--pos_weight",  type=float, default=DEFAULT_CONFIG["pos_weight"])
    parser.add_argument("--device",      default=DEFAULT_CONFIG["device"])
    parser.add_argument("--resume",      default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config.update(vars(args))

    train(config)


if __name__ == "__main__":
    main()
