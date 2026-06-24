"""
Redrob hackathon — rank-time entry point.

Single command that produces the submission CSV (Stage-3 reproduction target):

    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Behavior:
  * If --features <parquet> exists, load it (fast path, ~1s) -- used for tuning.
  * Otherwise build features from --candidates (full path, ~90s) -- self-contained
    reproduction, still well inside the 5-minute / CPU / no-network budget.

Constraints honored: CPU-only, no network, <=5 min, <=16 GB. The scoring itself is a
vectorized weighted sum over a feature table (~1s); the only cost is the optional
one-time feature build.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ranker.build import build_features_df          # noqa: E402
from ranker.rubric import load_rubric               # noqa: E402
from ranker.scorer import score_dataframe, rank_top_n  # noqa: E402
from ranker.reasoning import add_reasoning           # noqa: E402

REQUIRED_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]


def get_features(args, rubric) -> pd.DataFrame:
    if args.features and os.path.exists(args.features):
        print(f"loading cached features: {args.features}", flush=True)
        return pd.read_parquet(args.features)
    if not args.candidates:
        sys.exit("ERROR: no cached --features and no --candidates to build from.")
    print(f"building features from {args.candidates} ...", flush=True)
    df, meta = build_features_df(args.candidates, rubric, limit=args.limit)
    print(f"  built {meta['n_candidates']} rows in {meta['build_seconds']}s", flush=True)
    return df


def main(argv):
    ap = argparse.ArgumentParser(description="Redrob candidate ranker (rank-time).")
    ap.add_argument("--candidates", help="path to candidates.jsonl")
    ap.add_argument("--rubric", default=os.path.join(os.path.dirname(__file__), "rubric.yaml"))
    ap.add_argument("--features", default=os.path.join(os.path.dirname(__file__), "artifacts", "features.parquet"),
                    help="cached feature parquet (used if it exists)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="debug cap on candidates when building")
    ap.add_argument("--no-reasoning", action="store_true", help="emit empty reasoning column (Phase 6 fills it)")
    args = ap.parse_args(argv)

    t0 = time.time()
    rubric = load_rubric(os.path.abspath(args.rubric))
    df = get_features(args, rubric)

    df = score_dataframe(df, rubric)
    ranked = rank_top_n(df, n=args.top)

    if args.no_reasoning:
        ranked["reasoning"] = ""   # tuning runs skip reasoning for speed
    else:
        add_reasoning(ranked)      # Phase 6: fact-grounded reasoning per row

    out = ranked[REQUIRED_COLUMNS].copy()
    out.to_csv(args.out, index=False, encoding="utf-8")

    # sanity report
    print(f"\nwrote {args.out}  ({len(out)} rows)")
    print(f"score range: {out['score'].min():.3f} .. {out['score'].max():.3f}")
    print(f"monotonic non-increasing: {bool((out['score'].diff().dropna() <= 1e-9).all())}")
    print(f"total time: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
