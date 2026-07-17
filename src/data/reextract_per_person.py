"""
reextract_per_person.py -- Rebuild the collaboration feature cache as TRUE PER-PERSON
features, fixing the Stage-0 degeneracy where the on-disk data/collab_cache/ stored a
single WHOLE-FRAME vector replicated onto every student (feat_A == feat_B for all 883
pairs; all tracks in a frame collapsed to ONE vector).

WHAT WAS WRONG (proven in Stage-0)
----------------------------------
The cached 768-d vectors in data/collab_cache/*_A.npy / *_B.npy are scene/whole-frame
features, not per-person crops. Within any (video, frame) all tracked people shared ONE
identical vector. That makes pair-level collaboration structurally impossible:
  - pair LOVO macro-F1 = 0.499 (chance), within-VID_(4) ceiling = 0.55 ~= shuffle floor.
The breakage is ONLY in the cached .npy. Everything needed to fix it is correct on disk:
  - per-person crops:  data/collab_raw/crops/{video}/{track}/clip_XXXX/frame_000N.jpg (distinct)
  - catalog:           data/collab_raw/pair_catalog_33.csv  (clip_dir_A != clip_dir_B, 100%)
  - checkpoint:        weights/best_clip_model.pth  (backbone./temporal./head. keys load cleanly)
  - extractor code:    FeatureExtractorWrapper.forward (per-frame backbone -> temporal pool)
So the original cache is STALE (an older whole-frame build). Re-running the CORRECT extractor
on the per-person crops produces a valid cache.

WHAT THIS SCRIPT DOES
---------------------
1. Loads the frozen engagement backbone and ASSERTS the checkpoint actually loaded
   (aborts if backbone weights are missing -> avoids silently caching random-net garbage).
2. SELF-TEST: extracts two DIFFERENT track crops from the same (video, frame) and asserts
   their features are NOT identical. If they are, the environment/model is wrong and we stop
   BEFORE wasting time on a full extraction.
3. Drives extraction from the EXISTING feature_index.csv (authoritative pair_ids + labels) so
   the new cache is a 1:1 drop-in: same 6869 clip-rows -> same 883 unique pairs, same splits.
   Clip directories come from the catalog (joined on pair_id). Features are cached per UNIQUE
   clip_dir (a person-clip recurs across many pairs) so each person-clip is encoded once.
4. Writes data/collab_cache_fresh/{pair_id}_A.npy, _B.npy and a fresh feature_index.csv.
5. POST-CHECKS: verifies feat_A != feat_B across pairs and that different tracks in a frame
   now have DISTINCT vectors. Prints the fraction still-identical (must be ~0).

COLAB USAGE (GPU)
-----------------
  # after cloning the repo and mounting weights:
  !python src/data/reextract_per_person.py \
      --index   data/collab_cache/feature_index.csv \
      --catalog data/collab_raw/pair_catalog_33.csv \
      --model_path weights/best_clip_model.pth \
      --out_dir data/collab_cache_fresh
  # quick smoke test first (50 pairs):  add  --limit 50

Then bring data/collab_cache_fresh/ back and we re-run ceiling_probe.py + the honest LOVO
evaluator pointed at --cache data/collab_cache_fresh to see if the VID_(4) ceiling rises
above ~0.56 (the Stage-2 go/no-go).
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

import numpy as np

# --- make repo modules importable whether run from repo root or src/data ------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in (str(_ROOT), str(_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# imports of the EXISTING, correct extractor code (reuse -> guaranteed consistency)
# ---------------------------------------------------------------------------

def _load_repo_pieces():
    import torch  # noqa
    try:
        from src.models.swin_clip_model import build_clip_model
        from src.models.collaboration_head import build_feature_extractor
        from src.data.collab_dataset import _extract_clip_feature, build_val_transform, CLIP_LEN
    except ImportError:
        from models.swin_clip_model import build_clip_model
        from models.collaboration_head import build_feature_extractor
        from data.collab_dataset import _extract_clip_feature, build_val_transform, CLIP_LEN
    return (build_clip_model, build_feature_extractor,
            _extract_clip_feature, build_val_transform, CLIP_LEN)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _norm_dir(raw, crops_root):
    """Catalog stores Windows paths like 'data\\collab_raw\\crops\\VID_ (6)\\1\\clip_0003'.
    Normalise slashes and resolve under crops_root (default repo root)."""
    p = str(raw).replace("\\", "/").strip()
    cand = Path(crops_root) / p
    return cand


def load_checkpoint_strict(build_clip_model, model_path, device):
    """Load engagement backbone; ABORT if the backbone did not actually load."""
    import torch
    if not Path(model_path).exists():
        sys.exit(f"[FATAL] checkpoint not found: {model_path}")
    model = build_clip_model(num_classes=2, pretrained=False)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    # strip a possible DataParallel 'module.' prefix
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if k.startswith("backbone.")]
    if missing:
        print(f"[FATAL] {len(missing)} backbone.* keys MISSING from checkpoint load.")
        print("        The backbone would be random -> features meaningless. Aborting.")
        print("        first few:", missing[:5])
        sys.exit(1)
    n_loaded = sum(1 for _ in state)
    print(f"[ckpt ] loaded {n_loaded} tensors; backbone fully matched "
          f"(unexpected={len(result.unexpected_keys)}, non-backbone missing="
          f"{len(result.missing_keys) - len(missing)})")
    return model


def self_test_distinct(extract_fn, jobs, crops_root, transform, extractor, device, CLIP_LEN):
    """Find two DIFFERENT track dirs in the same (video, start_frame) and prove their
    features differ. Catches a whole-frame / constant-output regression up front."""
    by_frame = {}
    for j in jobs:
        key = (j["video_id"], j["pair_id"].rsplit("_f", 1)[-1])
        by_frame.setdefault(key, set()).update([j["dir_A"], j["dir_B"]])
    pick = None
    for key, dirs in by_frame.items():
        ds = [d for d in dirs if _norm_dir(d, crops_root).is_dir()]
        if len(ds) >= 2:
            pick = ds[:2]; break
    if not pick:
        print("[selftest] WARN: could not find two co-frame dirs; skipping distinctness test")
        return
    f1 = extract_fn(str(_norm_dir(pick[0], crops_root)), extractor, transform, device)
    f2 = extract_fn(str(_norm_dir(pick[1], crops_root)), extractor, transform, device)
    f1 = np.asarray(f1, np.float64); f2 = np.asarray(f2, np.float64)
    l2 = float(np.linalg.norm(f1 - f2))
    cos = float((f1 @ f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-9))
    print(f"[selftest] two different tracks, same frame:  |A-B|={l2:.4f}  cos={cos:.4f}")
    if l2 < 1e-6 or cos > 0.9999:
        print("[FATAL] two DIFFERENT people produced IDENTICAL features.")
        print("        The extractor is collapsing input -> do NOT trust a full run. Aborting.")
        sys.exit(1)
    print("[selftest] PASS -- per-person features are distinct.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Re-extract TRUE per-person collab features.")
    ap.add_argument("--index",   default="data/collab_cache/feature_index.csv",
                    help="authoritative pair_ids + labels to reproduce")
    ap.add_argument("--catalog", default="data/collab_raw/pair_catalog_33.csv",
                    help="provides clip_dir_A / clip_dir_B per pair_id")
    ap.add_argument("--crops_root", default=".",
                    help="root that the catalog's relative crop dirs resolve against")
    ap.add_argument("--model_path", default="weights/best_clip_model.pth")
    ap.add_argument("--out_dir", default="data/collab_cache_fresh")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="only process first N pairs (smoke test)")
    args = ap.parse_args()

    (build_clip_model, build_feature_extractor,
     _extract_clip_feature, build_val_transform, CLIP_LEN) = _load_repo_pieces()
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[init ] device={device}  out_dir={args.out_dir}")

    # ---- authoritative pairs + labels from the existing index ----
    index_rows = list(csv.DictReader(open(args.index, newline="")))
    if not index_rows:
        sys.exit(f"[FATAL] empty index: {args.index}")
    # de-dup pair_ids but preserve label + meta (first occurrence wins)
    pair_meta = {}
    for r in index_rows:
        pair_meta.setdefault(r["pair_id"], r)
    print(f"[index] {len(index_rows)} clip-rows -> {len(pair_meta)} unique pair_ids")

    # ---- crop dirs from catalog ----
    cat = {r["pair_id"]: r for r in csv.DictReader(open(args.catalog, newline=""))}
    jobs = []
    miss_cat = miss_dir = 0
    for pid, meta in pair_meta.items():
        c = cat.get(pid)
        if c is None:
            miss_cat += 1; continue
        dA, dB = c.get("clip_dir_A"), c.get("clip_dir_B")
        if not _norm_dir(dA, args.crops_root).is_dir() or not _norm_dir(dB, args.crops_root).is_dir():
            miss_dir += 1; continue
        jobs.append({"pair_id": pid, "label": meta["label"], "video_id": meta["video_id"],
                     "frame_w": meta.get("frame_w", 848), "frame_h": meta.get("frame_h", 480),
                     "track_id_A": meta.get("track_id_A", 0), "track_id_B": meta.get("track_id_B", 1),
                     "dir_A": dA, "dir_B": dB})
    print(f"[join ] jobs={len(jobs)}  (missing-in-catalog={miss_cat}, missing-crop-dir={miss_dir})")
    if args.limit:
        jobs = jobs[:args.limit]
        print(f"[limit] smoke test: first {len(jobs)} pairs only")
    if not jobs:
        sys.exit("[FATAL] no extractable pairs after join")

    # ---- model ----
    model = load_checkpoint_strict(build_clip_model, args.model_path, device)
    extractor = build_feature_extractor(model).to(device).eval()
    transform = build_val_transform()

    # ---- up-front self-test ----
    self_test_distinct(_extract_clip_feature, jobs, args.crops_root,
                       transform, extractor, device, CLIP_LEN)

    # ---- extract, caching per UNIQUE clip_dir (person-clip recurs across pairs) ----
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    feat_cache = {}  # normalised clip_dir -> 768 vector

    def feat_for(dir_raw):
        key = str(_norm_dir(dir_raw, args.crops_root))
        v = feat_cache.get(key)
        if v is None:
            v = _extract_clip_feature(key, extractor, transform, device)
            v = None if v is None else np.asarray(v, np.float32)
            feat_cache[key] = v
        return v

    t0 = time.time(); done = skipped = 0
    new_index = []
    for i, j in enumerate(jobs):
        fA = feat_for(j["dir_A"]); fB = feat_for(j["dir_B"])
        if fA is None or fB is None:
            skipped += 1; continue
        pA = out / f"{j['pair_id']}_A.npy"; pB = out / f"{j['pair_id']}_B.npy"
        np.save(str(pA), fA); np.save(str(pB), fB)
        new_index.append({"pair_id": j["pair_id"], "label": j["label"],
                          "feat_A": str(pA), "feat_B": str(pB), "video_id": j["video_id"],
                          "frame_w": j["frame_w"], "frame_h": j["frame_h"],
                          "track_id_A": j["track_id_A"], "track_id_B": j["track_id_B"]})
        done += 1
        if (i + 1) % 200 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{len(jobs)}  ({rate:.1f} pair/s, {len(feat_cache)} uniq clip-feats)")

    # ---- write fresh index ----
    idx_path = out / "feature_index.csv"
    with open(idx_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(new_index[0].keys()))
        w.writeheader(); w.writerows(new_index)
    print(f"[save ] {done} pairs cached, {skipped} skipped; index -> {idx_path}")
    print(f"[save ] {len(feat_cache)} unique person-clip features encoded "
          f"in {time.time()-t0:.0f}s")

    # ---- POST-CHECKS: the degeneracy must be gone ----
    print("\n[verify] re-loading a sample to confirm the fix ...")
    rng = np.random.default_rng(0)
    sample = rng.choice(len(new_index), size=min(200, len(new_index)), replace=False)
    n_identical = 0
    for k in sample:
        r = new_index[int(k)]
        a = np.load(r["feat_A"]); b = np.load(r["feat_B"])
        if float(np.linalg.norm(a - b)) < 1e-6:
            n_identical += 1
    frac = n_identical / len(sample)
    print(f"[verify] pairs with feat_A == feat_B: {n_identical}/{len(sample)} ({frac*100:.1f}%)")

    # within-(video,frame) distinctness across tracks
    by_frame = {}
    for r in new_index:
        key = (r["video_id"], r["pair_id"].rsplit("_f", 1)[-1])
        by_frame.setdefault(key, []).append(r["feat_A"])
    multi = [(k, v) for k, v in by_frame.items() if len(v) >= 3][:5]
    for k, paths in multi:
        vs = [np.load(p) for p in paths[:6]]
        norms = {round(float(np.linalg.norm(x)), 3) for x in vs}
        print(f"[verify] frame {k}: {len(vs)} tracks -> {len(norms)} distinct feature norms")

    if frac > 0.05:
        print("\n[verify] WARNING: features still look degenerate. Investigate before training.")
    else:
        print("\n[verify] PASS -- per-person features are distinct. Cache is a valid drop-in.")
        print(f"[next ] re-run:  python src/training/ceiling_probe.py "
              f"(rebuild npz with --cache {args.out_dir} first)")
        print(f"[next ] re-run:  python src/training/train_collab_video_level.py "
              f"--index {idx_path} --cache {args.out_dir} --l2 1.0 --min_pairs 3")


if __name__ == "__main__":
    main()
