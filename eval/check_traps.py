"""Phase 5 verification: trap/honeypot leakage in the ranked top-N.

Loads the cached features, applies the real scorer, and reports how many of the
top-N carry each trap/consistency flag. The hard line is the hackathon's: honeypot
DQ rate in top 100 must stay well under 10%. Run after any scorer/rubric change.

    python eval/check_traps.py            # top 100 from artifacts/features.parquet
"""
from __future__ import annotations

import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from ranker.rubric import load_rubric            # noqa: E402
from ranker.scorer import score_dataframe, rank_top_n  # noqa: E402

# flag -> human label; honeypot_hard is the DQ line, the rest are consistency softs.
TRAP_FLAGS = {
    "honeypot_hard": "honeypot (hard floor)",
    "cons_role_gt_career": "role longer than career",
    "cons_expert_zero_usage": "expert proficiency, ~0 months used",
    "cons_skilldur_gt_career": "skill duration exceeds career",
    "cons_tenuresum_gt_exp": "tenure sum exceeds experience",
    "dq_stuffer": "keyword stuffer",
    "dq_consulting_only": "consulting-firm-only",
    "dq_title_chaser": "title chaser",
    "dq_cv_primary": "CV/speech primary",
}


def main(argv) -> int:
    n = int(argv[0]) if argv else 100
    rubric = load_rubric(os.path.join(ROOT, "rubric.yaml"))
    df = pd.read_parquet(os.path.join(ROOT, "artifacts", "features.parquet"))
    df = score_dataframe(df, rubric)
    top = rank_top_n(df, n=n)

    print(f"top {n} trap/consistency leakage (pool totals in parens):\n")
    for flag, label in TRAP_FLAGS.items():
        if flag not in df.columns:
            continue
        in_top = int(top[flag].sum())
        pool = int(df[flag].sum())
        pct = 100.0 * in_top / n
        print(f"  {label:38s} {in_top:3d}/{n}  ({pct:4.1f}%)   pool={pool}")

    hp = 100.0 * int(top["honeypot_hard"].sum()) / n
    line = "PASS" if hp < 10.0 else "FAIL"
    print(f"\nhoneypot DQ rate in top {n}: {hp:.1f}%  -> {line} (< 10% required)")
    return 0 if hp < 10.0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
