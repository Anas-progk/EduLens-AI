"""
eval_engagement_honest.py -- score best_clip_model.pth on the UNTOUCHED test set.

WHY: the headline "92.4% / macro-F1 0.9019" is the **validation** score of the val-selected epoch at
the val-selected **0.75 threshold** (`train_clip.py` prints it as "[FINAL VAL REPORT]"). It is therefore
(a) selection-optimistic — the epoch AND the threshold were both chosen to maximise macro-F1 on that same
val set — and (b) mislabeled as "test". The real held-out test videos (VID_5,6,8,17,20) were NEVER scored.

The split itself is clean: train(18) / val(5) / test(5) videos are fully disjoint (no shared
video / clip / frame / person), so this is NOT leakage — it is just the wrong number reported.

This script computes the honest generalisation number: take the FIXED val-selected threshold from the
checkpoint and apply it to custom_test (no re-tuning on test). It also re-scores val as a sanity check
that it reproduces ~0.9019.

Run on Colab (GPU + checkpoint + custom_dataset/processed frames present):
    python src/eval/eval_engagement_honest.py \
        --ckpt /content/drive/MyDrive/engagement_weights/best_clip_model.pth
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.data.clip_dataset import ClipDataset
from src.models.swin_clip_model import build_clip_model


@torch.no_grad()
def infer(model, csv, split, device, bs=4):
    ds = ClipDataset(csv, split=split, image_size=224, n_frames=8)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    P, Y = [], []
    model.eval()
    for clips, labels in dl:
        logits = model(clips.to(device))
        P.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())   # P(Engaged)
        Y.extend(labels.numpy())
    return np.array(P), np.array(Y), len(ds)


def report(name, P, Y, t):
    pred = (P >= t).astype(int)
    mf = f1_score(Y, pred, average="macro", zero_division=0)
    per = f1_score(Y, pred, average=None, labels=[0, 1], zero_division=0)
    acc = accuracy_score(Y, pred)
    print(f"\n[{name}]  threshold={t:.2f}  clips={len(Y)}  (NE={int((Y == 0).sum())} / E={int((Y == 1).sum())})")
    print(f"  macro-F1 = {mf:.4f}   NE-F1 = {per[0]:.4f}   E-F1 = {per[1]:.4f}   accuracy = {acc:.4f}")
    print(classification_report(Y, pred, target_names=["Not Engaged", "Engaged"], digits=4, zero_division=0))
    return mf, per[0], acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="weights/best_clip_model.pth")
    ap.add_argument("--val", default="data/splits/custom_val.csv")
    ap.add_argument("--test", default="data/splits/custom_test.csv")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(a.ckpt, map_location=device)
    thr = float(ckpt.get("threshold", 0.5)) if isinstance(ckpt, dict) else 0.5
    sd = ckpt["model_state"] if (isinstance(ckpt, dict) and "model_state" in ckpt) else ckpt
    model = build_clip_model(num_classes=2, pretrained=False, drop_rate=0.30,
                             drop_path_rate=0.15, n_frames=8).to(device)
    model.load_state_dict(sd)
    print(f"loaded {a.ckpt}  |  device {device}  |  val-selected threshold = {thr:.2f}")

    # 1) sanity — reproduce the reported validation number
    Pv, Yv, _ = infer(model, a.val, "val", device)
    print("\n" + "#" * 70)
    print("#  SANITY: VALIDATION  (should reproduce the reported ~0.9019 / 92.4%)")
    print("#" * 70)
    report("VAL @ val-threshold", Pv, Yv, thr)

    # 2) the honest number — untouched test set, FIXED val-selected threshold
    Pt, Yt, _ = infer(model, a.test, "test", device)
    print("\n" + "#" * 70)
    print("#  HONEST HELD-OUT TEST  (VID_5,6,8,17,20 — never used for training OR selection)")
    print("#" * 70)
    mft, _, acct = report("TEST @ val-threshold (THE honest number)", Pt, Yt, thr)
    report("TEST @ 0.50 argmax (reference)", Pt, Yt, 0.50)

    print("\n" + "=" * 70)
    print(f"VERDICT INPUT:  honest TEST macro-F1 = {mft:.4f}  (acc {acct:.4f})  vs reported VAL 0.9019")
    print("  • Gap small  -> 92.4% was essentially honest (just relabel val→test).")
    print("  • Gap large  -> 92.4% was validation-optimistic; report the TEST number instead.")
    print("=" * 70)


if __name__ == "__main__":
    main()
