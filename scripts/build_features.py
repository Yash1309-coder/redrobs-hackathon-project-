"""
OFFLINE build step (network OK, no clock). Produces the artifacts the fast rank-time
scorer consumes:

    artifacts/features.parquet   - one row per candidate: all structured rubric signals
                                   + semantic_sim (TF-IDF cosine to the JD reference)
    artifacts/build_meta.json    - reference date, row count, vectorizer params, versions

Why TF-IDF here (Phase 3 hybrid decision): zero heavy deps, builds in seconds, fully
reproducible. The semantic layer is a drop-in: swapping in BGE embeddings later only
changes how `semantic_sim` is computed; every downstream phase is unaffected.

Usage:
    python scripts/build_features.py \
        --candidates "<path>/candidates.jsonl" \
        --rubric rubric.yaml \
        --out artifacts/
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# make src/ importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ranker.features import extract_features, _relevant_skill_tokens  # noqa: E402
from ranker.rubric import load_rubric  # noqa: E402
from ranker.textbuild import candidate_text, reference_text  # noqa: E402


def iter_candidates(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def compute_reference_date(records) -> dt.date:
    """Reference 'today' = max last_active_date in the pool (deterministic, no clock)."""
    best = None
    for c in records:
        d = (c.get("redrob_signals", {}) or {}).get("last_active_date") or ""
        try:
            day = dt.date.fromisoformat(d[:10])
            if best is None or day > best:
                best = day
        except Exception:
            continue
    return best or dt.date(2026, 6, 1)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--rubric", default="rubric.yaml")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (debug)")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    rubric = load_rubric(os.path.abspath(args.rubric))
    rel_tokens = _relevant_skill_tokens(rubric)

    t0 = time.time()
    print("loading candidates ...", flush=True)
    records = list(iter_candidates(args.candidates))
    if args.limit:
        records = records[: args.limit]
    n = len(records)
    print(f"  {n} candidates loaded in {time.time()-t0:.1f}s", flush=True)

    ref_date = compute_reference_date(records)
    print(f"reference date (max last_active): {ref_date.isoformat()}", flush=True)

    # ---- semantic layer: TF-IDF cosine to the JD reference text ----
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    print("building candidate text + TF-IDF ...", flush=True)
    texts = [candidate_text(c) for c in records]
    ref = reference_text(rubric)
    vec = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=2, max_features=50000, sublinear_tf=True,
    )
    X = vec.fit_transform(texts)              # (n, vocab)
    qv = vec.transform([ref])                 # (1, vocab)
    sims = linear_kernel(qv, X).ravel()       # cosine (tfidf is L2-normalized)
    print(f"  tfidf vocab={len(vec.vocabulary_)}  sim[min/mean/max]="
          f"{sims.min():.3f}/{sims.mean():.3f}/{sims.max():.3f}", flush=True)

    # ---- structured features ----
    print("extracting structured features ...", flush=True)
    rows = []
    te = time.time()
    for i, c in enumerate(records):
        row = extract_features(c, rubric, ref_date, rel_tokens)
        row["semantic_sim"] = float(sims[i])
        rows.append(row)
        if (i + 1) % 20000 == 0:
            print(f"  {i+1}/{n} ({time.time()-te:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    out_parquet = os.path.join(args.out, "features.parquet")
    df.to_parquet(out_parquet, index=False)

    meta = {
        "n_candidates": n,
        "reference_date": ref_date.isoformat(),
        "semantic_layer": "tfidf-1.2gram-50k",
        "tfidf_vocab": len(vec.vocabulary_),
        "feature_columns": list(df.columns),
        "build_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "build_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nwrote {out_parquet}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    print(f"total build time: {time.time()-t0:.1f}s")
    # quick sanity: flag prevalences
    for col in ["dq_stuffer", "dq_consulting_only", "dq_cv_primary", "honeypot_hard", "domain_corroborated"]:
        if col in df:
            print(f"  {col:22s} sum={int(df[col].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
