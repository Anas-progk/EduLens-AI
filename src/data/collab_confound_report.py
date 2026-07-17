"""
collab_confound_report.py -- Measure the SCENE-LABEL CONFOUND in collab data.

WHY THIS EXISTS
---------------
A collaboration model can "cheat" by memorizing WHICH VIDEO a pair came from
instead of learning whether the pair is actually collaborating. This happens
when a whole video is labeled almost entirely one class (e.g. every pair in
VID_7 is "C"). Then "predict the video's majority class" already scores high
WITHOUT any collaboration understanding. That fake score is the confound.

This tool quantifies it with one number: the SCENE-ONLY BASELINE.
  scene_baseline = sum_over_videos( max(C_count, N_count) ) / total
  = the accuracy of a model that knows ONLY which video each pair is from
    and predicts that video's majority label.

  - scene_baseline ~ 50%  -> videos are balanced; the data forces real learning. GOOD.
  - scene_baseline ~ 95%  -> label is almost determined by the video. The model
                             will memorize scenes and FALL APART on new footage. BAD.

It also tells you, concretely:
  * which videos are BALANCED enough to use for an HONEST val/test split, and
  * which PURE videos are inflating the confound and should be re-checked / re-annotated.

USAGE
-----
  python src/data/collab_confound_report.py                          # auto-find catalog
  python src/data/collab_confound_report.py --csv data/collab_raw/pair_catalog_merged.csv
  python src/data/collab_confound_report.py --csv data/collab_cache/feature_index.csv

Run it again after each annotation session to watch the confound number drop.
"""

import os
import csv
import argparse
from collections import defaultdict


# Candidate input files, most-canonical first.
DEFAULT_CANDIDATES = [
    "data/collab_raw/pair_catalog_merged.csv",
    "data/collab_cache/feature_index.csv",
    "data/collab_raw/pair_catalog.csv",
]


def _find_default_csv():
    for p in DEFAULT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def _load_labeled(csv_path):
    """Return list of (video_id, label) for rows labeled C or N. Dedupe to unique pairs."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], []

    cols = rows[0].keys()
    has_pairkey = {"video_id", "track_id_A", "track_id_B"} <= set(cols)

    # Collapse multiple clips of the same person-pair into one vote (majority label),
    # so near-duplicate frames don't distort the confound estimate.
    pair_votes = defaultdict(list)
    clip_level = []
    for r in rows:
        lab = (r.get("label", "") or "").strip()
        if lab not in ("C", "N"):
            continue
        vid = r.get("video_id", "UNKNOWN")
        clip_level.append((vid, lab))
        if has_pairkey:
            key = (vid, str(r.get("track_id_A")), str(r.get("track_id_B")))
            pair_votes[key].append(lab)

    if has_pairkey:
        pair_level = []
        for (vid, a, b), votes in pair_votes.items():
            maj = "C" if votes.count("C") >= votes.count("N") else "N"
            pair_level.append((vid, maj))
        return pair_level, clip_level
    # No pair key (e.g. feature_index without track ids) -> use clip level for both.
    return clip_level, clip_level


def _per_video(pairs):
    vid = defaultdict(lambda: [0, 0])   # video -> [C, N]
    for v, l in pairs:
        vid[v][0 if l == "C" else 1] += 1
    return vid


def _classify(c, n):
    t = c + n
    if c == 0:
        return "pure-N"
    if n == 0:
        return "pure-C"
    frac = c / t
    return "BALANCED" if 0.35 <= frac <= 0.65 else "mixed"


def report(csv_path):
    pairs, clips = _load_labeled(csv_path)
    if not pairs:
        print(f"No C/N labels found in {csv_path}")
        return

    unit = "person-pairs" if len(pairs) != len(clips) else "clips"
    vid = _per_video(pairs)

    total = sum(c + n for c, n in vid.values())
    nC = sum(c for c, n in vid.values())
    nN = total - nC
    scene_correct = sum(max(c, n) for c, n in vid.values())
    scene_baseline = scene_correct / total
    blind = max(nC, nN) / total

    SEP = "=" * 74
    print(f"\n{SEP}")
    print(f"COLLAB CONFOUND REPORT   ({os.path.basename(csv_path)})")
    print(SEP)
    print(f"Counting unique {unit}: {total}   (from {len(clips)} labeled clips)")
    print(f"Class balance: C={nC} ({nC/total*100:.0f}%)   N={nN} ({nN/total*100:.0f}%)")
    print(f"Videos: {len(vid)}")

    # ---- per-video table -------------------------------------------------
    print(f"\n{'video':<22}{'C':>5}{'N':>5}{'tot':>6}{'C%':>6}  kind")
    print("-" * 52)
    balanced, pure = [], []
    for v in sorted(vid, key=lambda x: -(vid[x][0] + vid[x][1])):
        c, n = vid[v]
        t = c + n
        kind = _classify(c, n)
        if kind == "BALANCED":
            balanced.append((v, c, n))
        elif kind in ("pure-C", "pure-N"):
            pure.append((v, c, n, t))
        print(f"{v:<22}{c:>5}{n:>5}{t:>6}{c/t*100:>5.0f}%  {kind}")

    # ---- headline numbers ------------------------------------------------
    print(f"\n{SEP}")
    print("HEADLINE")
    print(SEP)
    print(f"  Blind majority baseline : {blind*100:5.1f}%   (ignores everything, predicts '{('C' if nC>=nN else 'N')}')")
    print(f"  SCENE-ONLY baseline     : {scene_baseline*100:5.1f}%   <-- THE CONFOUND. Drive this toward 50%.")
    print(f"      A model that knows ONLY which video a pair is from scores this much")
    print(f"      with ZERO collaboration understanding. Your honest model must beat it")
    print(f"      on BALANCED videos to prove it learned anything real.")

    if scene_baseline >= 0.85:
        verdict = "SEVERE confound. Label ~= video. Honest generalization is near-impossible yet."
    elif scene_baseline >= 0.72:
        verdict = "HIGH confound. Eval on pure videos will look great but won't generalize."
    elif scene_baseline >= 0.62:
        verdict = "MODERATE confound. Usable if val/test are the BALANCED videos below."
    else:
        verdict = "LOW confound. Data supports honest evaluation."
    print(f"  Verdict: {verdict}")

    # ---- honest-eval feasibility ----------------------------------------
    print(f"\n{SEP}")
    print("HONEST EVAL FEASIBILITY")
    print(SEP)
    if balanced:
        bal_clips = sum(c + n for _, c, n in balanced)
        print(f"  BALANCED videos (35-65% C) -- USE THESE FOR val/test:")
        for v, c, n in balanced:
            print(f"    {v:<22} C={c:<4} N={n:<4} ({c/(c+n)*100:.0f}% C)")
        print(f"  -> {len(balanced)} balanced videos, {bal_clips} pairs available for honest eval.")
        print(f"     On these, scene-memorization scores only ~{max( [max(c,n)/(c+n) for _,c,n in balanced] )*100:.0f}% or less,")
        print(f"     so beating that margin = REAL collaboration understanding.")
    else:
        print("  *** NO balanced videos (none are 35-65% C). ***")
        print("  Every video is dominated by one class, so ANY val/test split lets the")
        print("  model win by guessing the scene's majority. You CANNOT measure honest")
        print("  collaboration accuracy on this data yet. Fix the data first (below).")

    # ---- re-annotation worklist -----------------------------------------
    print(f"\n{SEP}")
    print("CONFOUND-REDUCTION WORKLIST  (highest impact first)")
    print(SEP)
    pure.sort(key=lambda x: -x[3])
    big_pure = [p for p in pure if p[3] >= 8]
    if big_pure:
        print("  These PURE videos inject 'scene = label'. Re-open each in the annotator")
        print("  and check per-pair: are ALL pairs truly the same class, or was it labeled")
        print("  on autopilot? Adding the minority-class pairs that exist breaks the confound.")
        print(f"\n  {'video':<22}{'pairs':>6}  currently")
        for v, c, n, t in big_pure:
            cls = "all-C" if n == 0 else "all-N"
            print(f"    {v:<22}{t:>6}  {cls}")
        print(f"\n  Rule of thumb: a real classroom video almost never has EVERY pair")
        print(f"  collaborating or EVERY pair ignoring each other. If a 'pure' video has")
        print(f"  20+ pairs, it almost certainly contains both -> re-annotate it carefully.")
    else:
        print("  No large pure videos. Confound is driven by mild imbalance, not pure scenes.")

    print(f"\n{SEP}")
    print(f"TARGET: get SCENE-ONLY baseline under ~62% and have >=2 BALANCED videos")
    print(f"        for val/test. Then the honest split + stable training give a number")
    print(f"        you can defend.")
    print(SEP + "\n")

    return {
        "scene_baseline": scene_baseline,
        "blind": blind,
        "n_balanced_videos": len(balanced),
        "total": total,
    }


def main():
    ap = argparse.ArgumentParser(description="Measure scene-label confound in collab data.")
    ap.add_argument("--csv", default=None, help="catalog or feature_index CSV")
    args = ap.parse_args()

    path = args.csv or _find_default_csv()
    if not path or not os.path.exists(path):
        print("ERROR: no input CSV found. Pass --csv <path>. Looked for:")
        for p in DEFAULT_CANDIDATES:
            print(f"   {p}")
        return
    report(path)


if __name__ == "__main__":
    main()
