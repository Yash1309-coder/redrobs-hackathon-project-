# Recruiter-Brain Candidate Ranker

Redrob hackathon — *Intelligent Candidate Discovery & Ranking*. Ranks 100,000
candidates against a single JD and emits the top 100 with fact-grounded
reasoning, on CPU, in under 5 minutes, with no network at rank time.

**The edge is a JD-understanding layer, not keyword matching.** A frozen,
human-reviewed rubric (`rubric.yaml`) encodes what the role actually needs;
the scorer rewards skill↔title↔career coherence (the #1 signal in this pool,
where keyword stuffers outnumber corroborated candidates ~10:1) and demotes
stuffers, consulting-only careers, and internally-inconsistent honeypots.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+. CPU-only. No GPU, no API keys.

## Reproduce the submission (single command)

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

- **From scratch** (no cached features): builds the feature table from
  `candidates.jsonl` then scores — ~110s total on a 16-core CPU, well inside
  the 5-minute budget.
- **Fast path**: if `artifacts/features.parquet` exists it is loaded instead of
  rebuilt (~1s to score). Delete `artifacts/features.parquet` to force a clean
  from-scratch reproduction.

Output `submission.csv` has columns `candidate_id,rank,score,reasoning`, 100
rows, ranks 1–100, scores strictly non-increasing.

## Two-stage architecture (why it fits the budget)

```
candidates.jsonl ──► [offline build]  ──► artifacts/features.parquet (100k × 42 raw signals)
                     scripts/build_features.py        │
                                                      ▼
                     rubric.yaml ──► [rank-time]  ──► submission.csv
                                     rank.py (vectorized weighted sum, ~1s)
```

`features.parquet` holds **raw** signals; the scorer applies rubric **weights**.
Tuning re-runs only the fast scorer, never the 90s extraction. The semantic
layer is TF-IDF (zero heavy deps, fully reproducible); it is a drop-in for BGE
embeddings without touching any downstream phase.

## Layout

| Path | What |
|------|------|
| `rank.py` | rank-time entry point (build-if-needed → score → reason → CSV) |
| `rubric.yaml` | frozen, human-reviewed JD rubric — the signature artifact |
| `src/ranker/` | `rubric` (load), `textbuild`, `features`, `build`, `scorer`, `reasoning` |
| `scripts/build_features.py` | offline feature/embedding build (network OK here) |
| `artifacts/` | `features.parquet`, `build_meta.json` |
| `eval/` | `evaluate.py` (NDCG/MAP/P@k), `tune.py`, `check_traps.py`, `labels_batch1.csv` |
| `design.md`, `tracker.md` | approach + phase-by-phase decision log |

## Evaluation

Hand-labeled eval set (no leaderboard) scored with `eval/evaluate.py`:

- **Subset NDCG@10 = 0.9968** (NDCG@50 0.9968, MAP 0.9667) — flat across every
  weight knob, i.e. saturated and robust, not overfit.
- **Top-100 honeypot rate = 0%** (pool contains 40) — verified, re-runnable:
  `python eval/check_traps.py` (exits non-zero if it ever crosses 10%).
- Top-100: 100/100 domain-corroborated, YoE median 6.6 (in the 5–9 band).

```bash
python eval/tune.py          # full + subset NDCG/MAP against the labeled set
python eval/check_traps.py   # honeypot / trap leakage gate on the current submission
```
