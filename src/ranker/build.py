"""Shared offline-build logic: candidates.jsonl + rubric -> features DataFrame.

Used by both scripts/build_features.py (writes the artifact) and rank.py (rebuilds
from scratch when no cached artifact is present, for Stage-3 reproduction). Keeping it
in one place means the build that produces the cache and the build that reproduces it
can never drift apart.
"""
from __future__ import annotations

import datetime as dt
import json
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from .features import extract_features, _relevant_skill_tokens
from .textbuild import candidate_text, reference_text


def iter_candidates(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def compute_reference_date(records: List[Dict[str, Any]]) -> dt.date:
    """Reference 'today' = max last_active_date in the pool (deterministic, no clock)."""
    best: Optional[dt.date] = None
    for c in records:
        d = (c.get("redrob_signals", {}) or {}).get("last_active_date") or ""
        try:
            day = dt.date.fromisoformat(d[:10])
            if best is None or day > best:
                best = day
        except Exception:
            continue
    return best or dt.date(2026, 6, 1)


def _semantic_sims(records: List[Dict[str, Any]], rubric: Dict[str, Any]):
    """TF-IDF cosine of each candidate's narrative text to the JD reference text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel

    texts = [candidate_text(c) for c in records]
    ref = reference_text(rubric)
    vec = TfidfVectorizer(
        lowercase=True, stop_words="english", ngram_range=(1, 2),
        min_df=2, max_features=50000, sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    qv = vec.transform([ref])
    sims = linear_kernel(qv, X).ravel()
    return sims, len(vec.vocabulary_)


def build_features_df(candidates_path: str, rubric: Dict[str, Any],
                      limit: int = 0, verbose: bool = True):
    """Build the full feature DataFrame and a small meta dict."""
    t0 = time.time()
    records = list(iter_candidates(candidates_path))
    if limit:
        records = records[:limit]
    n = len(records)
    ref_date = compute_reference_date(records)
    rel_tokens = _relevant_skill_tokens(rubric)

    sims, vocab = _semantic_sims(records, rubric)

    rows = []
    for i, c in enumerate(records):
        row = extract_features(c, rubric, ref_date, rel_tokens)
        row["semantic_sim"] = float(sims[i])
        rows.append(row)
        if verbose and (i + 1) % 20000 == 0:
            print(f"  features {i+1}/{n} ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    meta = {
        "n_candidates": n,
        "reference_date": ref_date.isoformat(),
        "semantic_layer": "tfidf-1.2gram-50k",
        "tfidf_vocab": vocab,
        "feature_columns": list(df.columns),
        "build_seconds": round(time.time() - t0, 1),
    }
    return df, meta
