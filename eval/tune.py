"""
Phase 7 — offline weight tuning against the hand-labeled eval set.

What it does
------------
Loads the precomputed feature table, applies the rubric via the SAME scorer used
at rank time, produces a FULL ranking of all 100k, and scores it against
eval/labels_batch1.csv using the official composite (eval/evaluate.py).

Why a full ranking (not the top-100 submission): subset-mode NDCG orders ALL the
judged candidates, but the low-tier labels (stuffers, honeypots) are correctly
buried far below rank 100. We must see their true positions, so we rank everyone.

Two trustworthy facts about the knobs (see features.py / scorer.py):
  * SEMANTIC_WEIGHT and all rubric weights are applied at score time -> free to sweep.
  * behavioral_multiplier is baked into the parquet, BUT it = floor + (1-floor)*w with
    a KNOWN old floor (0.55). So we can invert w and re-apply any new floor exactly,
    letting us tune behavioral aggressiveness without the 90s rebuild.

With only 17 labels this is a sensitivity check, not a fit: we want knobs where
NDCG@10 is already high and FLAT (robust), and we stop the moment it plateaus.
Overfitting 17 points would be the failure mode, not under-tuning.

Usage:
    python eval/tune.py                  # baseline + sweeps
"""
from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import ranker.scorer as scorer            # noqa: E402  (mutable SEMANTIC_WEIGHT)
from ranker.rubric import load_rubric     # noqa: E402
from evaluate import load_labels, evaluate  # noqa: E402

OLD_FLOOR = 0.55  # behavioral.multiplier_floor baked into features.parquet


def remap_behavioral(df: pd.DataFrame, new_floor: float) -> pd.DataFrame:
    """Re-apply a different behavioral floor to the baked multiplier, exactly."""
    if new_floor == OLD_FLOOR:
        return df
    w = ((df["behavioral_multiplier"] - OLD_FLOOR) / (1.0 - OLD_FLOOR)).clip(0.0, 1.0)
    df = df.copy()
    df["behavioral_multiplier"] = new_floor + (1.0 - new_floor) * w
    return df


def ranked_ids(df: pd.DataFrame, rubric: dict, semantic_weight: float, beh_floor: float) -> list:
    """Full ranking (all rows) under the given knobs -> candidate_ids best-first."""
    scorer.SEMANTIC_WEIGHT = semantic_weight
    d = remap_behavioral(df, beh_floor)
    d = scorer.score_dataframe(d, rubric)
    d = d.sort_values(by=["score", "candidate_id"], ascending=[False, True], kind="mergesort")
    return d["candidate_id"].tolist()


def measure(df, rubric, labels, semantic_weight, beh_floor) -> dict:
    ids = ranked_ids(df, rubric, semantic_weight, beh_floor)
    return {
        "subset": evaluate(ids, labels, mode="subset"),
        "full": evaluate(ids, labels, mode="full"),
    }


def row(label, m) -> str:
    s, f = m["subset"], m["full"]
    return (f"{label:<28} "
            f"subset NDCG@10={s['ndcg@10']:.4f} comp={s['composite']:.4f} | "
            f"full NDCG@10={f['ndcg@10']:.4f} comp={f['composite']:.4f}")


def main() -> int:
    rubric = load_rubric(os.path.join(ROOT, "rubric.yaml"))
    df = pd.read_parquet(os.path.join(ROOT, "artifacts", "features.parquet"))
    labels = load_labels(os.path.join(HERE, "labels_batch1.csv"))

    base_sem = 6.0   # current scorer.SEMANTIC_WEIGHT
    base_floor = OLD_FLOOR

    base = measure(df, rubric, labels, base_sem, base_floor)
    print("=" * 96)
    print(row(f"BASELINE (sem={base_sem}, floor={base_floor})", base))
    print(f"  judged in ranking: {int(base['subset']['n_ranked_judged'])}/{len(labels)}")
    print("=" * 96)

    print("\n-- sweep: behavioral floor (aggressiveness; lower = punishes unavailability harder) --")
    for floor in [0.35, 0.45, 0.55, 0.65, 0.75]:
        print("  " + row(f"floor={floor}", measure(df, rubric, labels, base_sem, floor)))

    print("\n-- sweep: SEMANTIC_WEIGHT (TF-IDF cosine -> fit units) --")
    for sem in [3.0, 4.5, 6.0, 8.0, 10.0]:
        print("  " + row(f"sem={sem}", measure(df, rubric, labels, sem, base_floor)))

    print("\n-- sweep: domain core_title_match weight (the #1 anti-stuffer lever) --")
    orig = rubric["domain"]["weights"]["core_title_match"]
    for w in [2.0, 3.0, 4.0, 5.0]:
        rubric["domain"]["weights"]["core_title_match"] = w
        print("  " + row(f"core_title={w}", measure(df, rubric, labels, base_sem, base_floor)))
    rubric["domain"]["weights"]["core_title_match"] = orig

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
