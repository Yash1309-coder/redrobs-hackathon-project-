"""Phase 6 — fact-grounded reasoning, templated deterministically at rank time.

Every clause traces to a column on the candidate's feature row (years, title, named
skills the profile actually lists, fired must-have detections, behavioral signals). No
network, no LLM, no hallucinated skills/employers. Variation is real (driven by each
candidate's distinct facts) plus deterministic phrasing rotation keyed on candidate_id,
so output stays byte-identical across reruns.

Tone tracks rank: confident at the top, honest about caveats lower down.
"""
from __future__ import annotations

from typing import Any, Mapping

# must-have column -> short human phrase (ordered by JD priority; first missing = the gap we name)
_MUST_HAVES = [
    ("mh_embeddings_retrieval_production", "production embeddings/retrieval"),
    ("mh_vector_db_hybrid_search", "vector-DB & hybrid search"),
    ("mh_ranking_evaluation_literacy", "ranking evaluation (NDCG/MRR/MAP)"),
    ("mh_shipped_at_scale", "shipping end-to-end at scale"),
    ("mh_strong_python_engineering", "strong Python engineering"),
]

# deterministic phrasing pools (indexed by candidate hash) so rows don't read identically
_LEAD_CORROBORATED = [
    "{y}y {title} whose title and narrative corroborate genuine retrieval/ranking work",
    "{y}y {title} with title-and-summary evidence of real in-domain work",
    "{y}y {title}; the career text backs up the AI/retrieval signal, not just skill tags",
]
_LEAD_WEAK = [
    "{y}y {title} with relevant signal but lighter domain corroboration",
    "{y}y {title}; in-domain evidence present but not dominant",
    "{y}y {title} adjacent to the role rather than squarely in it",
]
_CONNECTORS = ["Brings", "Evidence of", "Shows"]


def _h(cid: Any, k: int) -> int:
    """Stable, clock-free index in [0,k) from a candidate id."""
    return sum(bytes(str(cid), "utf-8")) % k


def _clean(s: Any) -> str:
    """Trim and drop the U+FFFD mojibake that exists in some source summaries."""
    return " ".join(str(s or "").replace("�", " ").split())


def generate_reasoning(row: Mapping[str, Any], rank: int) -> str:
    cid = row.get("candidate_id")
    y = row.get("years_of_experience") or 0
    ystr = f"{y:.1f}".rstrip("0").rstrip(".")
    title = _clean(row.get("current_title")) or "engineer"
    corroborated = int(row.get("domain_corroborated") or 0) == 1

    lead_pool = _LEAD_CORROBORATED if corroborated else _LEAD_WEAK
    lead = lead_pool[_h(cid, len(lead_pool))].format(y=ystr, title=title)

    # strengths: fired must-haves (grounded -- detection ran on their own text/skills)
    hits = [phrase for col, phrase in _MUST_HAVES if int(row.get(col) or 0) == 1]
    skills = _clean(row.get("matched_skills"))
    parts = [lead + "."]
    if hits:
        conn = _CONNECTORS[_h(cid, len(_CONNECTORS))]
        # rotate the window of shown must-haves so rows that fire the same set still read differently
        off = _h(cid, len(hits))
        shown = (hits[off:] + hits[:off])[:3]
        body = ", ".join(shown)
        if skills:
            body += f"; lists {skills}"
        parts.append(f"{conn} {body}.")
    elif skills:
        parts.append(f"Lists {skills}.")

    # one honest gap: the highest-priority must-have that did NOT fire
    gap = next((phrase for col, phrase in _MUST_HAVES if int(row.get(col) or 0) == 0), None)
    if gap and rank > 15:
        parts.append(f"Gap: no explicit {gap}.")
    elif (row.get("product_fit") or 0) < 0:
        parts.append("Caveat: services-industry background, not product.")

    # availability tail from real signals
    tail = []
    if row.get("open_to_work_flag"):
        tail.append("open to work")
    rr = row.get("recruiter_response_rate")
    if rr is not None and rr >= 0.5:
        tail.append(f"responsive to recruiters ({rr:.0%})")
    npd = row.get("notice_period_days")
    if npd is not None and npd <= 30:
        tail.append("short notice period")
    if tail:
        parts.append((", ".join(tail)).capitalize() + ".")

    return " ".join(parts)


def add_reasoning(ranked, reasoning_col: str = "reasoning"):
    """Fill the reasoning column on a ranked top-N DataFrame (in place)."""
    ranked[reasoning_col] = [
        generate_reasoning(r, int(r["rank"])) for r in ranked.to_dict("records")
    ]
    return ranked


if __name__ == "__main__":  # ponytail: one runnable check on the non-trivial logic
    base = {"years_of_experience": 7.2, "current_title": "ML Engineer", "domain_corroborated": 1,
            "mh_embeddings_retrieval_production": 1, "mh_ranking_evaluation_literacy": 1,
            "matched_skills": "FAISS, pgvector", "open_to_work_flag": True}
    a, b = ({**base, "candidate_id": "c1"}, {**base, "candidate_id": "c2"})
    ra, rb = generate_reasoning(a, 1), generate_reasoning(b, 1)
    assert ra == generate_reasoning(a, 1), "must be deterministic"
    assert "FAISS" in ra and "pgvector" in ra, "must name the candidate's real skills"
    # no skill claimed that the profile doesn't list
    none = generate_reasoning({**base, "candidate_id": "c3", "matched_skills": ""}, 1)
    assert "FAISS" not in none and "lists" not in none.lower(), "no hallucinated skills"
    # honest gap surfaces for a low-ranked candidate missing a must-have
    gap = generate_reasoning({"candidate_id": "c4", "years_of_experience": 5, "current_title": "Dev",
                              "domain_corroborated": 0, "mh_embeddings_retrieval_production": 0}, 80)
    assert "Gap:" in gap, "must admit gaps honestly at lower ranks"
    print("reasoning self-check OK:", ra)
