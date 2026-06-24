"""Recruiter-Brain candidate ranker for the Redrob hackathon.

Package layout:
    rubric.py     - load + access the frozen rubric.yaml (the JD-understanding layer)
    textbuild.py  - turn a candidate record into normalized text (for TF-IDF / embeddings)
    features.py   - extract per-candidate structured signals from the rubric

The offline build (scripts/build_features.py) produces artifacts; the rank-time
scorer (Phase 4) consumes them and stays model-free and fast.
"""
