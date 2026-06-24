"""Structured feature extraction: candidate record + rubric -> raw signal dict.

Design choice (tracker decision 2026-06-21): this module emits RAW / lightly-normalized
signals only. The rank-time scorer applies the rubric WEIGHTS. That separation means
Phase 7 weight-tuning re-runs only the cheap scorer, never this extraction.

Every signal here traces to a concrete profile field, so the reasoning generator
(Phase 6) and the Stage-5 interview can point at exactly why a candidate scored as it did.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------- small helpers

def _low(x: Any) -> str:
    return (x or "").lower().strip() if isinstance(x, str) else ""


def _contains_any(haystack: str, needles: List[str]) -> bool:
    return any(n in haystack for n in needles)


def _count_any(haystack: str, needles: List[str]) -> int:
    return sum(1 for n in needles if n in haystack)


def _days_between(ref: dt.date, date_str: str) -> Optional[int]:
    try:
        d = dt.date.fromisoformat((date_str or "")[:10])
        return (ref - d).days
    except Exception:
        return None


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _ramp(v: float, poor: float, ideal: float) -> float:
    """Linear 0->1 as v goes poor->ideal (handles either direction)."""
    if ideal == poor:
        return 0.0
    return _clamp((v - poor) / (ideal - poor))


# ----------------------------------------------------------------- field access

def _skill_names(c: Dict[str, Any]) -> List[str]:
    return [_low(s.get("name")) for s in c.get("skills", []) or []]


def _companies(c: Dict[str, Any]) -> List[str]:
    return [_low(h.get("company")) for h in c.get("career_history", []) or []]


def _career_text(c: Dict[str, Any]) -> str:
    p = c.get("profile", {}) or {}
    parts = [_low(p.get("summary")), _low(p.get("headline"))]
    parts += [_low(h.get("description")) for h in c.get("career_history", []) or []]
    parts += [_low(h.get("title")) for h in c.get("career_history", []) or []]
    return " ".join(parts)


def _relevant_skill_tokens(rubric: Dict[str, Any]) -> List[str]:
    """All skill tokens we treat as 'relevant retrieval/ML' for coherence checks."""
    dom = rubric.get("domain", {})
    toks = set(dom.get("core_terms", []) or [])
    for mh in rubric.get("must_haves", []):
        toks.update((mh.get("detect", {}) or {}).get("skills_any", []) or [])
    return [str(t).lower() for t in toks]


# ----------------------------------------------------------------- domain / coherence

def _domain_signals(c: Dict[str, Any], rubric: Dict[str, Any], rel_tokens: List[str]) -> Dict[str, Any]:
    dom = rubric.get("domain", {})
    p = c.get("profile", {}) or {}
    title = _low(p.get("current_title"))
    head = _low(p.get("headline"))
    title_head = title + " " + head
    summ_career = _low(p.get("summary")) + " " + _career_text(c)
    skills = " ".join(_skill_names(c))

    core_titles = [t.lower() for t in dom.get("core_titles", []) or []]
    core_terms = [t.lower() for t in dom.get("core_terms", []) or []]
    off_terms = [t.lower() for t in dom.get("off_domain_terms", []) or []]

    core_title_match = _contains_any(title_head, core_titles)
    core_term_in_summary = _count_any(summ_career, core_terms)
    core_skill_count = sum(1 for sk in _skill_names(c) if any(tok in sk for tok in rel_tokens))
    off_domain_count = _count_any(title_head + " " + skills, off_terms)
    # CV/speech/robotics is the candidate's PRIMARY domain only when it shows up in the
    # title/headline (not just as a couple of stray skill tags) AND in-domain signal is
    # weak. Validated in Phase 2: requiring the title term drops false positives ~50x.
    off_in_title = _contains_any(title_head, off_terms)
    off_domain_dominant = off_domain_count >= 2 and core_skill_count <= 2 and off_in_title

    # corroboration: the make-or-break signal vs keyword stuffers
    domain_corroborated = bool(
        core_title_match
        or _contains_any(summ_career, ["retrieval", "ranking", "recommend", "semantic search",
                                       "information retrieval", "search relevance", "nlp"])
    )
    return {
        "core_title_match": int(core_title_match),
        "core_term_in_summary": core_term_in_summary,
        "core_skill_count": core_skill_count,
        "off_domain_count": off_domain_count,
        "off_domain_dominant": int(off_domain_dominant),
        "domain_corroborated": int(domain_corroborated),
    }


# ----------------------------------------------------------------- experience

def _experience_fit(yoe: float, rubric: Dict[str, Any]) -> float:
    e = rubric.get("experience", {})
    if e["ideal_min"] <= yoe <= e["ideal_max"]:
        return 1.0
    if e["inscope_min"] <= yoe <= e["inscope_max"]:
        return 0.7
    if (e["hard_low"] <= yoe < e["inscope_min"]) or (e["inscope_max"] < yoe <= e["hard_high"]):
        return 0.35
    return -0.5  # out of band (too junior / manager drift) -> penalty signal


# ----------------------------------------------------------------- must / nice

def _detect_block(c: Dict[str, Any], det: Dict[str, Any], career_text: str,
                  skill_names: List[str], signals: Dict[str, Any]) -> float:
    """Return 1.0 if ANY listed evidence is present, else 0.0."""
    if not det:
        return 0.0
    if _contains_any(career_text, [t.lower() for t in det.get("summary_terms", []) or []]):
        return 1.0
    sk_any = [t.lower() for t in det.get("skills_any", []) or []]
    if any(tok in sk for tok in sk_any for sk in skill_names):
        return 1.0
    for sig, thr in (det.get("signal_min", {}) or {}).items():
        v = signals.get(sig)
        if v is not None and v >= thr:
            return 1.0
    for ind in [t.lower() for t in det.get("industry_any", []) or []]:
        if ind in _low(c.get("profile", {}).get("current_industry")):
            return 1.0
    return 0.0


# ----------------------------------------------------------------- disqualifiers

def _disqualifiers(c: Dict[str, Any], rubric: Dict[str, Any], dom: Dict[str, Any],
                   rel_tokens: List[str]) -> Dict[str, int]:
    p = c.get("profile", {}) or {}
    title = _low(p.get("current_title"))
    summ = _low(p.get("summary"))
    skills = _skill_names(c)
    companies = _companies(c)
    career = c.get("career_history", []) or []
    yoe = p.get("years_of_experience") or 0
    dq = {d["id"]: d for d in rubric.get("disqualifiers", [])}
    out: Dict[str, int] = {}

    # incoherence / keyword stuffer
    d = dq.get("incoherence_keyword_stuffer", {}).get("detect", {})
    stuffer_titles = [t.lower() for t in d.get("stuffer_titles", []) or []]
    rel_skill_count = sum(1 for sk in skills if any(tok in sk for tok in rel_tokens))
    out["dq_stuffer"] = int(
        rel_skill_count >= 3 and not dom["domain_corroborated"] and _contains_any(title, stuffer_titles)
    )

    # consulting-firm-only (check WHOLE history; product experience anywhere clears it)
    d = dq.get("consulting_firm_only", {}).get("detect", {})
    consult = [t.lower() for t in d.get("consulting_firms", []) or []]
    out["dq_consulting_only"] = int(
        bool(companies) and all(any(k in comp for k in consult) for comp in companies)
    )

    # title-chaser: many short tenures, no long anchor
    durs = [h.get("duration_months") or 0 for h in career]
    if len(durs) >= 3:
        durs_sorted = sorted(durs)
        median = durs_sorted[len(durs_sorted) // 2]
        out["dq_title_chaser"] = int(median < 18 and max(durs) < 30)
    else:
        out["dq_title_chaser"] = 0

    # architect / lead who stopped coding
    d = dq.get("architect_stopped_coding", {}).get("detect", {})
    hands = [t.lower() for t in d.get("hands_on_terms", []) or []]
    out["dq_architect"] = int(
        any(k in title for k in ["architect", "tech lead", "engineering manager", "director"])
        and not _contains_any(summ, hands)
    )

    # langchain-only / recent LLM, no pre-LLM ML
    d = dq.get("langchain_only_recent_llm", {}).get("detect", {})
    fw = [t.lower() for t in d.get("framework_skills", []) or []]
    ai_skill_durs = [s.get("duration_months") or 0 for s in c.get("skills", []) or []
                     if any(tok in _low(s.get("name")) for tok in rel_tokens + fw)]
    has_fw = any(tok in sk for tok in fw for sk in skills)
    out["dq_langchain_only"] = int(
        has_fw and bool(ai_skill_durs) and max(ai_skill_durs) < 12
    )

    # CV/speech/robotics primary without NLP/IR
    out["dq_cv_primary"] = int(dom["off_domain_dominant"] == 1 and dom["core_title_match"] == 0)

    # closed-source no validation (weak)
    s = c.get("redrob_signals", {}) or {}
    out["dq_closed_source"] = int(
        yoe >= 6 and (s.get("github_activity_score") == -1) and not (c.get("certifications") or [])
        and not s.get("linkedin_connected", False)
    )
    return out


# ----------------------------------------------------------------- location / company

def _location_fit(c: Dict[str, Any], rubric: Dict[str, Any]) -> float:
    loc = rubric.get("location", {})
    p = c.get("profile", {}) or {}
    s = c.get("redrob_signals", {}) or {}
    location = _low(p.get("location"))
    country = _low(p.get("country"))
    targets = [t.lower() for t in loc.get("target_cities", []) or []]
    w = loc.get("weights", {})
    if _contains_any(location, targets):
        return w.get("target_city", 1.0)
    if country == "india":
        return w.get("india_other", 0.3)
    if s.get("willing_to_relocate"):
        return w.get("non_india_willing_relocate", -0.5)
    return w.get("non_india_not_relocating", -1.5)


def _product_fit(c: Dict[str, Any], rubric: Dict[str, Any]) -> float:
    pc = rubric.get("product_company", {})
    prod = [t.lower() for t in pc.get("product_industries", []) or []]
    w = pc.get("weights", {})
    cur_ind = _low(c.get("profile", {}).get("current_industry"))
    hist_inds = [_low(h.get("industry")) for h in c.get("career_history", []) or []]
    if _contains_any(cur_ind, prod):
        return w.get("product_now", 0.8)
    if any(_contains_any(i, prod) for i in hist_inds):
        return w.get("product_in_history", 0.5)
    return w.get("services_only", -0.5)


# ----------------------------------------------------------------- behavioral

def _behavioral_multiplier(c: Dict[str, Any], rubric: Dict[str, Any], ref_date: dt.date) -> float:
    b = rubric.get("behavioral", {})
    s = c.get("redrob_signals", {}) or {}
    f = b.get("factors", {})
    score, wsum = 0.0, 0.0

    def add(weight: float, val: float):
        nonlocal score, wsum
        score += weight * val
        wsum += weight

    rr = f.get("recruiter_response_rate", {})
    add(rr.get("weight", 0), _ramp(s.get("recruiter_response_rate") or 0, rr.get("poor", 0.1), rr.get("ideal", 0.85)))

    la = f.get("last_active_recency_days", {})
    days = _days_between(ref_date, s.get("last_active_date") or "")
    if days is not None:
        add(la.get("weight", 0), _ramp(days, la.get("poor", 180), la.get("ideal", 14)))

    ow = f.get("open_to_work_flag", {})
    add(ow.get("weight", 0), 1.0 if s.get("open_to_work_flag") else 0.0)

    ic = f.get("interview_completion_rate", {})
    add(ic.get("weight", 0), _ramp(s.get("interview_completion_rate") or 0, ic.get("poor", 0.3), ic.get("ideal", 0.85)))

    sv = f.get("saved_by_recruiters_30d", {})
    add(sv.get("weight", 0), _ramp(s.get("saved_by_recruiters_30d") or 0, sv.get("poor", 0), sv.get("ideal", 30)))

    # notice period
    npd = s.get("notice_period_days")
    npr = b.get("notice_period", {})
    if npd is not None:
        if npd <= npr.get("great_max_days", 30):
            nv = 1.0
        elif npd <= npr.get("ok_max_days", 60):
            nv = 0.6
        elif npd <= npr.get("penalty_over_days", 90):
            nv = 0.3
        else:
            nv = 0.0
        add(npr.get("weight", 0.15), nv)

    weighted = score / wsum if wsum else 0.0
    floor = b.get("multiplier_floor", 0.55)
    return floor + (1.0 - floor) * weighted


# ----------------------------------------------------------------- consistency / honeypot

def _consistency(c: Dict[str, Any], rubric: Dict[str, Any]) -> Dict[str, int]:
    p = c.get("profile", {}) or {}
    yoe = p.get("years_of_experience") or 0
    skills = c.get("skills", []) or []
    career = c.get("career_history", []) or []

    role_gt_career = any((h.get("duration_months") or 0) > yoe * 12 + 24 for h in career)
    expert_low = [s for s in skills if s.get("proficiency") == "expert" and (s.get("duration_months") or 0) <= 3]
    expert_zero_usage = len(expert_low) >= 3
    skilldur_gt_career = any((s.get("duration_months") or 0) > yoe * 12 + 18 for s in skills)
    tenure_sum = sum((h.get("duration_months") or 0) for h in career)
    tenuresum_gt_exp = yoe > 0 and tenure_sum > yoe * 12 * 1.6

    honeypot_hard = role_gt_career or expert_zero_usage  # high-precision floor
    return {
        "cons_role_gt_career": int(role_gt_career),
        "cons_expert_zero_usage": int(expert_zero_usage),
        "cons_skilldur_gt_career": int(skilldur_gt_career),
        "cons_tenuresum_gt_exp": int(tenuresum_gt_exp),
        "honeypot_hard": int(honeypot_hard),
    }


# ----------------------------------------------------------------- public API

def extract_features(c: Dict[str, Any], rubric: Dict[str, Any], ref_date: dt.date,
                     rel_tokens: Optional[List[str]] = None) -> Dict[str, Any]:
    """Return the full raw-signal row for one candidate."""
    if rel_tokens is None:
        rel_tokens = _relevant_skill_tokens(rubric)
    p = c.get("profile", {}) or {}
    s = c.get("redrob_signals", {}) or {}
    yoe = float(p.get("years_of_experience") or 0)

    row: Dict[str, Any] = {"candidate_id": c.get("candidate_id")}

    dom = _domain_signals(c, rubric, rel_tokens)
    row.update(dom)
    row["years_of_experience"] = yoe
    row["experience_fit"] = _experience_fit(yoe, rubric)

    career_text = _career_text(c)
    skill_names = _skill_names(c)
    for mh in rubric.get("must_haves", []):
        row[f"mh_{mh['id']}"] = _detect_block(c, mh.get("detect", {}), career_text, skill_names, s)
    for nh in rubric.get("nice_to_haves", []):
        row[f"nh_{nh['id']}"] = _detect_block(c, nh.get("detect", {}), career_text, skill_names, s)

    row.update(_disqualifiers(c, rubric, dom, rel_tokens))
    row["location_fit"] = _location_fit(c, rubric)
    row["product_fit"] = _product_fit(c, rubric)
    row["behavioral_multiplier"] = _behavioral_multiplier(c, rubric, ref_date)
    row.update(_consistency(c, rubric))

    # raw signals kept for reasoning generation + debugging
    row["recruiter_response_rate"] = s.get("recruiter_response_rate")
    row["open_to_work_flag"] = bool(s.get("open_to_work_flag"))
    row["notice_period_days"] = s.get("notice_period_days")
    row["github_activity_score"] = s.get("github_activity_score")

    # raw fact fields for Phase 6 reasoning (verbatim from the profile -> no hallucination)
    row["current_title"] = p.get("current_title") or ""
    row["current_industry"] = p.get("current_industry") or ""
    matched = [s_.get("name") for s_ in (c.get("skills") or [])
               if any(tok in _low(s_.get("name")) for tok in rel_tokens)]
    # cap by whole skills (not chars) so reasoning never truncates a name mid-word
    row["matched_skills"] = ", ".join(list(dict.fromkeys(m for m in matched if m))[:6])
    return row
