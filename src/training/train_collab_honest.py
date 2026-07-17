"""
train_collab_honest.py -- The honest, de-confounded Phase 2 trainer + evaluator.

RUN THIS ON COLAB (where the full .npy feature cache + the balanced video VID_(4)
features live). It is self-contained except for CollaborationHead.

WHAT IT FIXES (vs every previous run)
-------------------------------------
1. DUPLICATION: collapses 6,869 clip-rows -> 884 UNIQUE undirected pairs
   (one pair = one sample; (A,B)/(B,A) merged; near-duplicate frames pooled).
2. DEAD SIGNALS: feeds REAL relational signals (trajectory correlation, turn-taking,
   joint activity) computed from each pair's feature time-series -- the model finally
   has something collaborative to learn that is NOT scene identity.
3. CONFOUNDED EVAL: the balanced video VID_(4) is held out as TEST, where a
   scene-memorizing model scores only ~chance. The number on VID_(4) is the one
   you report and defend.
4. COLLAPSE: stable BCE (label smoothing + pos_weight), output-bias init, cosine LR,
   grad clip, and a collapse guard. No focal loss.

BUILT-IN ABLATION (this IS the feature probe):
   Trains TWICE -- with signals ON and with signals OFF (neutral) -- so you can see
   exactly how much the relational signals add on the honest VID_(4) test. If
   signals-ON >> signals-OFF, the relational signals are carrying the collaboration
   signal that the frozen engagement features cannot.

USAGE
-----
  python src/training/train_collab_honest.py \
      --index data/collab_cache/feature_index_33.csv \
      --cache data/collab_cache \
      --test_video "VID_ (4)" --epochs 80

  # single run, signals on only:
  python src/training/train_collab_honest.py --no_ablation
"""

import os, sys, json, argparse, numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "src")
from data.collab_pairs import load_pairs, honest_split, add_symmetric, SIGNAL_NAMES

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from models.collaboration_head import CollaborationHead, collab_loss
except Exception:
    from src.models.collaboration_head import CollaborationHead, collab_loss


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def prf(y, pred):
    """Per-class precision/recall/F1 for binary {0=N,1=C} + macro-F1."""
    out = {}
    for cls in (0, 1):
        tp = int(((pred == cls) & (y == cls)).sum())
        fp = int(((pred == cls) & (y != cls)).sum())
        fn = int(((pred != cls) & (y == cls)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[cls] = (p, r, f)
    macro_f1 = (out[0][2] + out[1][2]) / 2
    return out, macro_f1


def confusion(y, pred):
    return np.array([[int(((y == i) & (pred == j)).sum()) for j in (0, 1)] for i in (0, 1)])


# ---------------------------------------------------------------------------
# tensors
# ---------------------------------------------------------------------------

def to_tensors(pairs, sig_mean, sig_std, use_signals=True):
    A = torch.tensor(np.stack([p["pooled_A"] for p in pairs]), dtype=torch.float32)
    B = torch.tensor(np.stack([p["pooled_B"] for p in pairs]), dtype=torch.float32)
    if use_signals:
        S = (np.stack([p["signals"] for p in pairs]) - sig_mean) / sig_std
    else:
        S = np.zeros((len(pairs), len(SIGNAL_NAMES)), dtype=np.float32)  # neutral -> ablation
    S = torch.tensor(S, dtype=torch.float32)
    y = torch.tensor([p["label"] for p in pairs], dtype=torch.float32)
    return A, B, S, y


def scene_majority_f1(test_pairs):
    """macro-F1 a scene-only predictor gets on TEST (predict the test video's majority).
    On a balanced test this is ~0.34-0.40 -> the bar real understanding must clear."""
    from collections import defaultdict
    vid = defaultdict(lambda: [0, 0])
    for p in test_pairs:
        vid[p["video"]][p["label"]] += 1
    y = np.array([p["label"] for p in test_pairs])
    pred = np.array([1 if vid[p["video"]][1] >= vid[p["video"]][0] else 0 for p in test_pairs])
    return prf(y, pred)[1]


# ---------------------------------------------------------------------------
# train one configuration
# ---------------------------------------------------------------------------

def run(splits, cfg, use_signals, tag):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train = add_symmetric(splits["train"])
    val, test = splits["val"], splits["test"]

    sig = np.stack([p["signals"] for p in train])
    sig_mean, sig_std = sig.mean(0), sig.std(0) + 1e-6

    Atr, Btr, Str, ytr = to_tensors(train, sig_mean, sig_std, use_signals)
    Ava, Bva, Sva, yva = to_tensors(val, sig_mean, sig_std, use_signals)
    Ate, Bte, Ste, yte = to_tensors(test, sig_mean, sig_std, use_signals)

    c_rate = float(ytr.mean())
    pos_weight = (1 - c_rate) / max(c_rate, 1e-3)            # balance C vs N
    pos_weight = float(np.clip(pos_weight, 0.5, 2.0))

    model = CollaborationHead(signal_dim=len(SIGNAL_NAMES), dropout=cfg["dropout"]).to(dev)
    with torch.no_grad():                                    # bias init -> no collapse
        model.mlp[-1].bias.fill_(float(np.log(c_rate / (1 - c_rate))))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    loader = DataLoader(TensorDataset(Atr, Btr, Str, ytr), batch_size=cfg["bs"], shuffle=True)

    best_f1, best_state, best_thr, wait, collapse = -1, None, 0.5, 0, 0
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for a, b, s, yb in loader:
            a, b, s, yb = a.to(dev), b.to(dev), s.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = collab_loss(model(a, b, s), yb, pos_weight=pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Ava.to(dev), Bva.to(dev), Sva.to(dev))).cpu().numpy()
        yv = yva.numpy().astype(int)
        # collapse guard
        if abs(pv[yv == 1].mean() - pv[yv == 0].mean()) < 0.02:
            collapse += 1
            if collapse >= 8:
                print(f"  [{tag}] collapse guard tripped at ep{ep}; stopping"); break
        else:
            collapse = 0
        # threshold sweep on val (macro-F1, require F1_N>0.05 so it can't cheat to all-C)
        best_t, best_vf1 = 0.5, -1
        for t in np.linspace(0.30, 0.70, 41):
            _, mf1 = prf(yv, (pv >= t).astype(int))
            f1n = prf(yv, (pv >= t).astype(int))[0][0][2]
            if mf1 > best_vf1 and f1n > 0.05:
                best_vf1, best_t = mf1, t
        if best_vf1 > best_f1:
            best_f1, best_thr = best_vf1, best_t
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print(f"  [{tag}] early stop at ep{ep} (best val macroF1={best_f1:.3f})"); break

    # ---- evaluate best on TEST (the honest VID_(4) number) ----
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pt = torch.sigmoid(model(Ate.to(dev), Bte.to(dev), Ste.to(dev))).cpu().numpy()
    yt = yte.numpy().astype(int)
    pred = (pt >= best_thr).astype(int)
    per, macro = prf(yt, pred)
    acc = float((pred == yt).mean())
    cm = confusion(yt, pred)
    return {
        "tag": tag, "use_signals": use_signals, "thr": float(best_thr),
        "val_macroF1": float(best_f1), "test_macroF1": float(macro), "test_acc": acc,
        "test_F1_N": per[0][2], "test_F1_C": per[1][2],
        "test_prec_N": per[0][0], "test_rec_N": per[0][1],
        "test_prec_C": per[1][0], "test_rec_C": per[1][1],
        "confusion": cm.tolist(), "n_test": len(yt), "model": model.state_dict(),
        "sig_mean": sig_mean.tolist(), "sig_std": sig_std.tolist(),
    }


def report(r, scene_f1):
    print(f"\n{'='*60}\nRESULT [{r['tag']}]  signals={'ON' if r['use_signals'] else 'OFF'}")
    print(f"{'='*60}")
    print(f"  threshold (from val)     : {r['thr']:.2f}")
    print(f"  TEST macro-F1 (VID_4)    : {r['test_macroF1']:.3f}   "
          f"<-- vs scene-only baseline {scene_f1:.3f}")
    print(f"  TEST accuracy            : {r['test_acc']:.3f}   (n={r['n_test']})")
    print(f"  F1  N={r['test_F1_N']:.3f}  C={r['test_F1_C']:.3f}")
    print(f"  N: P={r['test_prec_N']:.3f} R={r['test_rec_N']:.3f}   "
          f"C: P={r['test_prec_C']:.3f} R={r['test_rec_C']:.3f}")
    print(f"  confusion [rows=true N,C][cols=pred N,C]: {r['confusion']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/collab_cache/feature_index_33.csv")
    ap.add_argument("--cache", default="data/collab_cache")
    ap.add_argument("--test_video", default="VID_ (4)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--out", default="weights/best_collab_honest.pth")
    ap.add_argument("--no_ablation", action="store_true", help="train signals-ON only")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = dict(epochs=args.epochs, patience=args.patience, lr=args.lr,
               wd=args.wd, bs=args.bs, dropout=args.dropout)

    print("="*60 + "\nHONEST COLLAB TRAINING\n" + "="*60)
    pairs = load_pairs(args.index, args.cache)
    splits = honest_split(pairs, test_videos=(args.test_video,))
    if len(splits["test"]) == 0:
        print(f"\nERROR: TEST video '{args.test_video}' has no pairs/features here.")
        print("       Run this on Colab where VID_(4) features exist.")
        return
    scene_f1 = scene_majority_f1(splits["test"])
    print(f"\nScene-only macro-F1 on TEST (the bar to beat): {scene_f1:.3f}")

    configs = [True] if args.no_ablation else [True, False]
    results = []
    for use_sig in configs:
        tag = "signals_ON" if use_sig else "signals_OFF"
        print(f"\n----- training {tag} -----")
        r = run(splits, cfg, use_sig, tag)
        report(r, scene_f1)
        results.append(r)

    # save the signals-ON model
    best = next(r for r in results if r["use_signals"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": best["model"], "threshold": best["thr"],
                "sig_mean": best["sig_mean"], "sig_std": best["sig_std"],
                "signal_names": SIGNAL_NAMES}, args.out)
    print(f"\nSaved signals-ON model -> {args.out}")

    if len(results) == 2:
        on = next(r for r in results if r["use_signals"])["test_macroF1"]
        off = next(r for r in results if not r["use_signals"])["test_macroF1"]
        print(f"\n{'='*60}\nABLATION (honest VID_4 test)\n{'='*60}")
        print(f"  signals OFF (engagement features only): macro-F1 {off:.3f}")
        print(f"  signals ON  (+ relational signals)    : macro-F1 {on:.3f}")
        print(f"  lift from relational signals          : {on-off:+.3f}")
        print(f"  scene-only baseline                   : {scene_f1:.3f}")
        if on - off > 0.04:
            print("  => relational signals add real, confound-resistant signal. KEEP them.")
        if off <= scene_f1 + 0.03:
            print("  => engagement features alone ~ scene baseline: they are largely")
            print("     collab-blind (expected: engagement was trained to SUPPRESS talking).")


if __name__ == "__main__":
    main()
