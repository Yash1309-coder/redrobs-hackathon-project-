"""
Offline ranking-evaluation harness for the Redrob hackathon.

Why this exists
---------------
There is no live leaderboard and only 3 submissions. We cannot tune by submitting.
So we build our own graded ground truth (hand-labeled tiers 0-4) and measure every
change against it offline. The JD explicitly asks for engineers who design ranking
evaluation (NDCG/MRR/MAP, offline-to-online) -- this module IS that skill, demonstrated.

Metrics (mirroring the official composite from submission_spec Section 4)
    composite = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10
where:
    - relevance grade = the labeled tier (0..4), used for graded NDCG.
    - "relevant" (for MAP / P@10) = tier >= 3, matching the spec's "tier 3+" definition.

Two evaluation modes
--------------------
full   : score a full ranking (all 100k, or a top-N ranking) against the labels.
         Candidates not present in the labels are treated as relevance 0. This mirrors
         the real eval (full hidden ground truth) but is only a proxy while our label
         set is sparse -- an unlabeled candidate ranked high is assumed irrelevant,
         which can understate a genuinely good pick we simply haven't labeled yet.
subset : restrict to labeled candidates only and measure how well the model *orders
         the ones we have judged*. More trustworthy with sparse labels; use this as
         the primary signal until the label set is large.

Dependencies: numpy only. No network, no pandas.

Usage
-----
    # Sanity check the metric math (ideal ranking should score ~1.0):
    python eval/evaluate.py --labels eval/labels_batch1.csv --selftest

    # Evaluate a ranking. RANKING may be either:
    #   (a) a submission CSV with a 'candidate_id' column (rank order = file order or 'rank'), or
    #   (b) a plain text file with one candidate_id per line (best first).
    python eval/evaluate.py --labels eval/labels_batch1.csv --ranking submission.csv
    python eval/evaluate.py --labels eval/labels_batch1.csv --ranking submission.csv --mode subset
"""
from __future__ import annotations
import argparse
import csv
import sys
from typing import Dict, List, Sequence

import numpy as np

RELEVANT_TIER = 3  # tier >= 3 counts as "relevant" for MAP / P@10 (per spec)


# --------------------------------------------------------------------------- IO

def load_labels(path: str) -> Dict[str, int]:
    """Read labels CSV -> {candidate_id: tier}. Requires 'candidate_id' and 'tier'."""
    labels: Dict[str, int] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "candidate_id" not in reader.fieldnames or "tier" not in reader.fieldnames:
            raise ValueError("labels file must have 'candidate_id' and 'tier' columns")
        for row in reader:
            cid = (row["candidate_id"] or "").strip()
            if not cid:
                continue
            labels[cid] = int(row["tier"])
    if not labels:
        raise ValueError(f"no labels loaded from {path}")
    return labels


def load_ranking(path: str) -> List[str]:
    """
    Read a ranking -> ordered list of candidate_ids (best first).

    Accepts a CSV with a 'candidate_id' column (ordered by a 'rank' column if present,
    else by file order), or a plain one-id-per-line text file.
    """
    with open(path, newline="", encoding="utf-8") as f:
        head = f.read(4096)
        f.seek(0)
        if "candidate_id" in head:
            reader = csv.DictReader(f)
            rows = [r for r in reader if (r.get("candidate_id") or "").strip()]
            if rows and "rank" in (reader.fieldnames or []):
                rows.sort(key=lambda r: int(r["rank"]))
            return [r["candidate_id"].strip() for r in rows]
        return [ln.strip() for ln in f if ln.strip()]


# ----------------------------------------------------------------------- metrics

def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with the 2^rel - 1 gain formulation."""
    g = np.asarray(gains, dtype=float)
    if g.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, g.size + 2))
    return float(np.sum((np.power(2.0, g) - 1.0) * discounts))


def ndcg_at_k(rels: Sequence[int], k: int) -> float:
    """NDCG@k given the graded relevances in ranked order."""
    rels = list(rels)
    if not rels:
        return 0.0
    actual = dcg(rels[:k])
    ideal = dcg(sorted(rels, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def average_precision(rels: Sequence[int]) -> float:
    """AP with binary relevance (tier >= RELEVANT_TIER). Single-query => this is MAP."""
    rels = list(rels)
    n_rel = sum(1 for r in rels if r >= RELEVANT_TIER)
    if n_rel == 0:
        return 0.0
    hits, score = 0, 0.0
    for i, r in enumerate(rels, start=1):
        if r >= RELEVANT_TIER:
            hits += 1
            score += hits / i
    return score / n_rel


def precision_at_k(rels: Sequence[int], k: int) -> float:
    rels = list(rels)[:k]
    if not rels:
        return 0.0
    return sum(1 for r in rels if r >= RELEVANT_TIER) / k


def composite(m: Dict[str, float]) -> float:
    return 0.50 * m["ndcg@10"] + 0.30 * m["ndcg@50"] + 0.15 * m["map"] + 0.05 * m["p@10"]


# --------------------------------------------------------------------- evaluation

def relevances_in_order(ranked_ids: Sequence[str], labels: Dict[str, int], mode: str) -> List[int]:
    """Map a ranked id list to graded relevances under the chosen mode."""
    if mode == "subset":
        return [labels[cid] for cid in ranked_ids if cid in labels]
    # full mode: unlabeled -> relevance 0
    return [labels.get(cid, 0) for cid in ranked_ids]


def evaluate(ranked_ids: Sequence[str], labels: Dict[str, int], mode: str = "subset") -> Dict[str, float]:
    rels = relevances_in_order(ranked_ids, labels, mode)
    m = {
        "ndcg@10": ndcg_at_k(rels, 10),
        "ndcg@50": ndcg_at_k(rels, 50),
        "map": average_precision(rels),
        "p@10": precision_at_k(rels, 10),
    }
    m["composite"] = composite(m)
    m["n_ranked_judged"] = float(sum(1 for cid in ranked_ids if cid in labels))
    return m


def print_report(m: Dict[str, float], title: str = "") -> None:
    if title:
        print(f"\n{title}")
    print(f"  NDCG@10   : {m['ndcg@10']:.4f}   (weight 0.50)")
    print(f"  NDCG@50   : {m['ndcg@50']:.4f}   (weight 0.30)")
    print(f"  MAP       : {m['map']:.4f}   (weight 0.15)")
    print(f"  P@10      : {m['p@10']:.4f}   (weight 0.05)")
    print(f"  --------------------------------")
    print(f"  COMPOSITE : {m['composite']:.4f}")
    print(f"  judged candidates in ranking: {int(m['n_ranked_judged'])}")


# -------------------------------------------------------------------------- main

def selftest(labels: Dict[str, int]) -> None:
    """Verify the math: ideal ordering -> NDCG ~1.0; reversed -> lower. No model needed."""
    ids_by_tier = sorted(labels, key=lambda c: labels[c], reverse=True)
    ideal = evaluate(ids_by_tier, labels, mode="subset")
    worst = evaluate(list(reversed(ids_by_tier)), labels, mode="subset")
    print_report(ideal, "SELF-TEST  ideal ordering (expect NDCG@10 = 1.0)")
    print_report(worst, "SELF-TEST  worst ordering (expect much lower)")
    assert abs(ideal["ndcg@10"] - 1.0) < 1e-9, "ideal NDCG@10 should be 1.0"
    assert ideal["composite"] >= worst["composite"], "ideal should beat worst"
    tiers = sorted(set(labels.values()))
    counts = {t: sum(1 for v in labels.values() if v == t) for t in tiers}
    print(f"\nlabel distribution by tier: {counts}  (total {len(labels)})")
    print("self-test passed.")


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Offline ranking eval (NDCG/MAP/P@k).")
    ap.add_argument("--labels", required=True, help="labels CSV (candidate_id,tier,...)")
    ap.add_argument("--ranking", help="ranking CSV or one-id-per-line text file")
    ap.add_argument("--mode", choices=["full", "subset"], default="subset")
    ap.add_argument("--selftest", action="store_true", help="run metric self-test and exit")
    args = ap.parse_args(argv)

    labels = load_labels(args.labels)

    if args.selftest:
        selftest(labels)
        return 0

    if not args.ranking:
        print("nothing to evaluate. pass --ranking <file> or --selftest.", file=sys.stderr)
        return 2

    ranked = load_ranking(args.ranking)
    m = evaluate(ranked, labels, mode=args.mode)
    print_report(m, f"EVAL  ranking={args.ranking}  mode={args.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
