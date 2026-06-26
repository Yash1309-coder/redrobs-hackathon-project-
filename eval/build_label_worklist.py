"""
Build the batch-2 labeling worklist (active learning).

The eval set is most informative where the MODEL is least certain and where a label
changes a real decision: the in/out boundary of the top-100, and the contested middle
where corroborated-but-not-retrieval profiles (T1/T2) sit next to genuine search/recsys
ones (T3/T4). Labelling random profiles wastes human time on the obvious tails.

This script ranks all candidates with the SAME scorer used at rank time, then samples
~30 candidates concentrated around the top-100 boundary plus a few anchors, EXCLUDING
the ones already labelled in batch 1. For each it pre-extracts the raw profile facts
(verbatim) into a review sheet. The human reads the facts and fills the blank `tier`
(0-4) and `reason` columns -- the labels stay 100% human judgment; the script only
chooses WHO to look at and lays the facts out.

Usage:
    python eval/build_label_worklist.py \
        --candidates "<path>/candidates.jsonl" \
        --out eval/labels_batch2_TODO.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from ranker.rubric import load_rubric          # noqa: E402
from ranker.scorer import score_dataframe       # noqa: E402

TARGET_N = 30


def load_facts(path: str) -> dict:
    """candidate_id -> compact fact dict, read verbatim from the profile."""
    facts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            p = c.get("profile", {}) or {}
            s = c.get("redrob_signals", {}) or {}
            hist = c.get("career_history", []) or []
            companies = ", ".join(h.get("company") or "" for h in hist[:3])
            skills = [sk.get("name") or "" for sk in (c.get("skills") or [])]
            facts[c.get("candidate_id")] = {
                "title": p.get("current_title") or "",
                "industry": p.get("current_industry") or "",
                "yoe": p.get("years_of_experience"),
                "companies": companies,
                "skills": ", ".join(s_ for s_ in skills[:10] if s_),
                "resp": s.get("recruiter_response_rate"),
                "last_active": s.get("last_active_date") or "",
                "open": s.get("open_to_work_flag"),
                "github": s.get("github_activity_score"),
            }
    return facts


def pick_boundary(ranked: pd.DataFrame, labeled: set, n: int) -> pd.DataFrame:
    """Sample where labels matter most: dense around the top-100 cut, a few anchors."""
    pool = ranked[~ranked["candidate_id"].isin(labeled)].reset_index(drop=True)
    pool = pool[pool["honeypot_hard"] == 0]  # honeypots already handled; don't waste labels
    pool = pool.reset_index(drop=True)

    # rank position in the full ordering (1 = best)
    pool["pos"] = pool.index + 1
    bands = {
        "anchor_top":   pool[pool["pos"].between(5, 30)],     # calibrate the T4 ceiling
        "boundary_in":  pool[pool["pos"].between(70, 100)],   # last picks that made the cut
        "boundary_out": pool[pool["pos"].between(101, 160)],  # first picks that missed -- contested
        "mid":          pool[pool["pos"].between(161, 400)],  # T1/T2 territory
    }
    quotas = {"anchor_top": 5, "boundary_in": 8, "boundary_out": 10, "mid": 7}
    picks = []
    for name, q in quotas.items():
        b = bands[name]
        if len(b):
            step = max(1, len(b) // q)
            picks.append(b.iloc[::step].head(q).assign(band=name))
    out = pd.concat(picks).drop_duplicates("candidate_id").head(n)
    return out


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--features", default=os.path.join(ROOT, "artifacts", "features.parquet"))
    ap.add_argument("--labels", default=os.path.join(HERE, "labels_batch1.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "labels_batch2_TODO.csv"))
    ap.add_argument("--n", type=int, default=TARGET_N)
    args = ap.parse_args(argv)

    rubric = load_rubric(os.path.join(ROOT, "rubric.yaml"))
    df = pd.read_parquet(args.features)
    df = score_dataframe(df, rubric)
    ranked = df.sort_values(["score", "candidate_id"], ascending=[False, True],
                            kind="mergesort").reset_index(drop=True)

    labeled = set(pd.read_csv(args.labels)["candidate_id"])
    picks = pick_boundary(ranked, labeled, args.n)
    facts = load_facts(args.candidates)

    rows = []
    for _, r in picks.iterrows():
        cid = r["candidate_id"]
        fa = facts.get(cid, {})
        rows.append({
            "candidate_id": cid,
            "tier": "",          # <-- HUMAN fills 0-4
            "archetype": "",     # <-- HUMAN fills (optional)
            "reason": "",        # <-- HUMAN fills
            "_model_pos": int(r["pos"]),
            "_band": r["band"],
            "_model_score": round(float(r["score"]), 3),
            "_title": fa.get("title"),
            "_industry": fa.get("industry"),
            "_yoe": fa.get("yoe"),
            "_companies": fa.get("companies"),
            "_skills": fa.get("skills"),
            "_resp": fa.get("resp"),
            "_last_active": fa.get("last_active"),
            "_open_to_work": fa.get("open"),
            "_github": fa.get("github"),
            "_domain_corroborated": int(r.get("domain_corroborated", 0)),
            "_dq_stuffer": int(r.get("dq_stuffer", 0)),
            "_dq_consulting": int(r.get("dq_consulting_only", 0)),
            "_dq_cv_primary": int(r.get("dq_cv_primary", 0)),
        })

    fields = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {args.out}  ({len(rows)} candidates to label)")
    print("Columns prefixed '_' are model context (read-only). Fill 'tier' (0-4),")
    print("optionally 'archetype' and 'reason'. Then build labels_batch2.csv from the filled rows.")
    by_band = {}
    for r in rows:
        by_band[r["_band"]] = by_band.get(r["_band"], 0) + 1
    print(f"band distribution: {by_band}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
