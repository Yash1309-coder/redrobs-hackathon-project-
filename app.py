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


@st.cache_data
def get_ranked() -> pd.DataFrame:
    rubric = load_rubric(os.path.join(HERE, "rubric.yaml"))
    df = pd.read_parquet(os.path.join(HERE, "artifacts", "features.parquet"))
    df = score_dataframe(df, rubric)
    ranked = rank_top_n(df, n=100)
    add_reasoning(ranked)
    return ranked[["candidate_id", "rank", "score", "reasoning"]]


st.set_page_config(page_title="Recruiter-Brain Ranker", layout="wide")
st.title("Recruiter-Brain Candidate Ranker")
st.caption(
    "Redrob — top candidates for one JD, scored by a frozen JD-understanding rubric. "
    "CPU, no network, deterministic."
)

ranked = get_ranked()
n = st.slider("Show top N", min_value=5, max_value=100, value=20, step=5)
st.dataframe(ranked.head(n), use_container_width=True, hide_index=True)
st.download_button(
    "Download full top-100 submission.csv",
    ranked.to_csv(index=False),
    file_name="submission.csv",
    mime="text/csv",
)
