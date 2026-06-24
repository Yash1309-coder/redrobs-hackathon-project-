"""Turn a candidate record into normalized text for the semantic layer (TF-IDF now,
BGE embeddings later). Also builds the JD/rubric reference text the candidate text is
compared against.

We deliberately weight the *career descriptions and summary* (what they actually did)
over the raw skills list (which the stuffers game). The skills are included once; the
narrative is what carries semantic signal for plain-language Tier-5s.
"""
from __future__ import annotations

from typing import Any, Dict, List


def _join(parts: List[str]) -> str:
    return " ".join(p for p in parts if p).strip()


def candidate_text(c: Dict[str, Any]) -> str:
    """Concatenate the narrative-heavy fields of a candidate profile."""
    p = c.get("profile", {}) or {}
    parts: List[str] = [
        p.get("headline") or "",
        p.get("summary") or "",
        p.get("current_title") or "",
        p.get("current_industry") or "",
    ]
    for h in c.get("career_history", []) or []:
        parts.append(h.get("title") or "")
        parts.append(h.get("description") or "")
    # skills once (names only) — present but not dominant
    parts.append(" ".join((s.get("name") or "") for s in c.get("skills", []) or []))
    return _join(parts)


def reference_text(rubric: Dict[str, Any]) -> str:
    """Build the JD/rubric reference text that candidates are scored against.

    This is the 'ideal candidate' description in the JD's own language plus the
    rubric's core/must-have vocabulary, so cosine similarity rewards genuine
    in-domain narratives even when the buzzword tags are absent.
    """
    dom = rubric.get("domain", {})
    must = rubric.get("must_haves", [])
    core_terms = dom.get("core_terms", []) or []
    core_titles = dom.get("core_titles", []) or []
    mh_terms: List[str] = []
    for mh in must:
        det = mh.get("detect", {}) or {}
        mh_terms += det.get("summary_terms", []) or []
        mh_terms += det.get("skills_any", []) or []
    ideal = (
        "Senior AI engineer with production experience building search, retrieval, "
        "ranking and recommendation systems at product companies. Hybrid retrieval "
        "(BM25 plus dense vector recall), embeddings, vector databases, and rigorous "
        "ranking evaluation with NDCG, MRR and MAP, offline to online. Shipped "
        "end-to-end systems to real users at scale. Strong Python."
    )
    return _join([ideal] + core_titles + core_terms + mh_terms)
