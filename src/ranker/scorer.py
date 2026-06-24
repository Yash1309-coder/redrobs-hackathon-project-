"""Rank-time scoring: features.parquet + rubric weights -> per-candidate score.

This is the heart of the competition deliverable. It is intentionally pure arithmetic
over a precomputed feature table -- no model, no network -- so it runs in ~1s on CPU,
far inside the 5-minute budget. All weights live in rubric.yaml; this module just
applies the composition defined there (design.md S3.2 / rubric.yaml S10):

    base_fit   = domain + experience + must_haves + nice_to_haves
                 + location_fit + product_fit + semantic_sim
    final      = base_fit * behavioral_multiplier - penalties
    penalties  = disqualifiers (soft) + soft-consistency
    honeypot_hard -> score floored to the bottom regardless of base_fit
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

# Semantic layer weight (TF-IDF cosine is ~0..0.25 here; scale it into fit units).
SEMANTIC_WEIGHT = 6.0
HONEYPOT_FLOOR_SCORE = -100.0


def _domain_score(df: pd.DataFrame, rubric: Dict[str, Any]) -> pd.Series:
    w = rubric["domain"]["weights"]
    s = (
        df["core_title_match"] * w["core_title_match"]
        + df["core_term_in_summary"].clip(upper=3) * w["core_term_in_summary"]
        + df["core_skill_count"].clip(upper=6) * w["core_skill_present"]
    )
    # off-domain penalty only when it's the dominant subdomain
    s = s + df["off_domain_dominant"] * w["off_domain_dominant_penalty"]
    return s


def _must_have_score(df: pd.DataFrame, rubric: Dict[str, Any]) -> pd.Series:
    s = pd.Series(0.0, index=df.index)
    for mh in rubric["must_haves"]:
        col = f"mh_{mh['id']}"
        if col in df:
            s = s + df[col] * mh["weight"]
    return s


def _nice_to_have_score(df: pd.DataFrame, rubric: Dict[str, Any]) -> pd.Series:
    s = pd.Series(0.0, index=df.index)
    for nh in rubric["nice_to_haves"]:
        col = f"nh_{nh['id']}"
        if col in df:
            s = s + df[col] * nh["weight"]
    return s


def _experience_score(df: pd.DataFrame, rubric: Dict[str, Any]) -> pd.Series:
    w = rubric["experience"]["weights"]
    # experience_fit already encodes band membership as {1.0, .7, .35, -.5};
    # map onto weights so tuning stays in rubric.yaml.
    fit = df["experience_fit"]
    out = pd.Series(0.0, index=df.index)
    out = out.mask(fit >= 1.0, w["in_ideal"])
    out = out.mask((fit >= 0.7) & (fit < 1.0), w["in_scope"])
    out = out.mask((fit >= 0.35) & (fit < 0.7), w["near_scope"])
    out = out.mask(fit < 0, w["out_of_band_penalty"])
    return out


def _penalties(df: pd.DataFrame, rubric: Dict[str, Any]) -> pd.Series:
    pen = pd.Series(0.0, index=df.index)
    dq = {d["id"]: d for d in rubric["disqualifiers"]}
    mapping = {
        "dq_stuffer": "incoherence_keyword_stuffer",
        "dq_consulting_only": "consulting_firm_only",
        "dq_title_chaser": "title_chaser",
        "dq_architect": "architect_stopped_coding",
        "dq_langchain_only": "langchain_only_recent_llm",
        "dq_cv_primary": "cv_speech_robotics_primary",
        "dq_closed_source": "closed_source_no_validation",
    }
    for col, rid in mapping.items():
        if col in df and rid in dq:
            pen = pen + df[col] * dq[rid]["penalty"]   # penalty is negative
    # soft consistency penalties (not the hard honeypot floor)
    cons = {r["id"]: r for r in rubric["consistency"]["rules"]}
    if "cons_skilldur_gt_career" in df and cons.get("skill_duration_exceeds_career", {}).get("soft"):
        pen = pen + df["cons_skilldur_gt_career"] * -0.5
    if "cons_tenuresum_gt_exp" in df and cons.get("tenure_sum_exceeds_experience", {}).get("soft"):
        pen = pen + df["cons_tenuresum_gt_exp"] * -0.5
    return pen


def score_dataframe(df: pd.DataFrame, rubric: Dict[str, Any]) -> pd.DataFrame:
    """Add 'base_fit', 'penalties', 'score' columns. Returns the same df."""
    loc_w = rubric["location"]["weights"]  # location_fit already in weight units
    base = (
        _domain_score(df, rubric)
        + _experience_score(df, rubric)
        + _must_have_score(df, rubric)
        + _nice_to_have_score(df, rubric)
        + df["location_fit"]
        + df["product_fit"]
        + df["semantic_sim"] * SEMANTIC_WEIGHT
    )
    df["base_fit"] = base
    df["penalties"] = _penalties(df, rubric)
    final = base * df["behavioral_multiplier"] + df["penalties"]
    # hard honeypot floor: forced to the bottom regardless of fit
    final = final.mask(df["honeypot_hard"] == 1, HONEYPOT_FLOOR_SCORE)
    df["score"] = final
    return df


def rank_top_n(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Sort by score desc, deterministic tie-break by candidate_id asc, take top n."""
    ranked = df.sort_values(
        by=["score", "candidate_id"], ascending=[False, True], kind="mergesort"
    ).head(n).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked
