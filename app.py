"""
Streamlit Cloud sandbox — interactive demo of the Recruiter-Brain ranker.

Loads the committed feature artifact (artifacts/features.parquet, 3.5M), applies the
frozen rubric weights, and shows the top-N candidates with their fact-grounded reasoning.
Same scoring path as rank.py, just rendered. CPU-only, no network.

Run locally:  streamlit run app.py
"""
import os
import sys

import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from ranker.rubric import load_rubric             # noqa: E402
from ranker.scorer import score_dataframe, rank_top_n  # noqa: E402
from ranker.reasoning import add_reasoning         # noqa: E402

# friendly names for the five must-have signals (mh_* columns)
MUST_HAVES = {
    "mh_embeddings_retrieval_production": "Embeddings/retrieval in prod",
    "mh_vector_db_hybrid_search": "Vector DB / hybrid search",
    "mh_ranking_evaluation_literacy": "Ranking eval (NDCG/MRR)",
    "mh_strong_python_engineering": "Strong Python eng",
    "mh_shipped_at_scale": "Shipped at scale",
}


@st.cache_data
def get_scored() -> pd.DataFrame:
    """Full 100k scored once; the funnel metrics need the whole pool, cards need top-100."""
    rubric = load_rubric(os.path.join(HERE, "rubric.yaml"))
    df = pd.read_parquet(os.path.join(HERE, "artifacts", "features.parquet"))
    return score_dataframe(df, rubric)


@st.cache_data
def get_ranked() -> pd.DataFrame:
    ranked = rank_top_n(get_scored(), n=100)
    add_reasoning(ranked)
    return ranked


def chip(text: str, kind: str = "skill") -> str:
    bg = {"skill": "#1e2a3a", "good": "#13351f", "warn": "#3a2a13"}[kind]
    fg = {"skill": "#9fc5e8", "good": "#7fd99a", "warn": "#e8c07f"}[kind]
    return (f"<span style='background:{bg};color:{fg};padding:2px 8px;border-radius:10px;"
            f"font-size:0.78rem;margin:2px;display:inline-block'>{text}</span>")


st.set_page_config(page_title="Recruiter-Brain Ranker", layout="wide")
st.title("Recruiter-Brain Candidate Ranker")
st.caption(
    "Redrob — top candidates for one JD, scored by a frozen JD-understanding rubric. "
    "CPU, no network, deterministic."
)

scored = get_scored()
ranked = get_ranked()

# --- defense funnel: the differentiator. 100k in, traps filtered, clean top-100 out.
st.subheader("From 100,000 profiles to a defensible top 100")
c = st.columns(5)
c[0].metric("Candidate pool", f"{len(scored):,}")
c[1].metric("Keyword stuffers", f"{int((scored['dq_stuffer'] == 1).sum()):,}", help="Title/career contradict AI skills — demoted")
c[2].metric("Consulting-only", f"{int((scored['dq_consulting_only'] == 1).sum()):,}", help="Services-firm careers — demoted")
c[3].metric("Honeypots floored", f"{int((scored['honeypot_hard'] == 1).sum()):,}", help="Internally impossible timelines — hard floor")
c[4].metric("Top-100 corroborated", f"{int(ranked['domain_corroborated'].sum())}/100",
            help="Skills backed by title + career, not keyword veneer")

st.divider()
left, right = st.columns([3, 1])
n = right.slider("Show top N", min_value=5, max_value=100, value=20, step=5)
right.download_button(
    "⬇ Download top-100 submission.csv",
    ranked[["candidate_id", "rank", "score", "reasoning"]].to_csv(index=False),
    file_name="submission.csv",
    mime="text/csv",
    use_container_width=True,
)

with left:
    for _, r in ranked.head(n).iterrows():
        with st.container(border=True):
            head = st.columns([1, 5])
            head[0].markdown(f"### #{int(r['rank'])}")
            head[0].markdown(f"**{r['score']:.2f}** pts")
            title = f"**{r['current_title']}** · {r['current_industry']} · {r['years_of_experience']:.1f}y"
            head[1].markdown(title)
            head[1].markdown(r["reasoning"])

            skills = [s.strip() for s in str(r["matched_skills"]).split(",") if s.strip()]
            fired = [v for k, v in MUST_HAVES.items() if r.get(k, 0) >= 1]
            badges = "".join(chip(s) for s in skills)
            badges += "".join(chip(f"✓ {f}", "good") for f in fired)
            if r.get("open_to_work_flag"):
                badges += chip("✓ open to work", "good")
            if r.get("penalties", 0) > 0:
                badges += chip(f"− {r['penalties']:.1f} penalty", "warn")
            head[1].markdown(badges, unsafe_allow_html=True)

            comp = (f"base {r['base_fit']:.1f}  ·  behavioral ×{r['behavioral_multiplier']:.2f}  "
                    f"·  {len(fired)}/5 must-haves  ·  resp {r['recruiter_response_rate']:.2f}")
            head[1].caption(comp)
