# Project Tracker — Recruiter-Brain Candidate Ranker

**Hackathon:** Redrob — Intelligent Candidate Discovery & Ranking
**Goal:** top-X finalist (reach the defend-your-work interview)
**Design:** see [design.md](design.md)
**Last updated:** 2026-06-25

---

## Status at a glance

| Phase | Status | Notes |
|-------|--------|-------|
| 0. Brainstorm & design | ✅ Done | design.md written |
| 1. Data understanding & eval set | ✅ Done | data profiled; batch 1 labeled (17); eval harness built + self-tested ✅ |
| 2. JD rubric (LLM-assisted, frozen) | ✅ Done | rubric.yaml authored + detection validated (0 false-positives on good candidates) |
| 3. Offline feature + embedding build | ✅ Done | features.parquet (100k×39) built in 92s; TF-IDF semantic layer; flags reconcile with Phase 2 |
| 4. Rank-time scorer | ✅ Done | rank.py end-to-end; rebuild 109s (<<5min); validate_submission passes; 0 traps in top 100 |
| 5. Honeypot / consistency defense | ✅ Done | top-100 honeypot rate 0% (verified, `eval/check_traps.py`) |
| 6. Reasoning generation | ✅ Done | fact-grounded, 100/100 unique, honest gaps; `src/ranker/reasoning.py` |
| 7. Tune against eval set | ✅ Done | subset NDCG@10 **0.9968** (ceiling, robust); config locked, no overfit |
| 8. Repo + deck + sandbox | 🟡 In progress | README/reqs/metadata/git/deck.pdf ✅; sandbox deployed but **app is private** — must set to public |
| 9. Validate & submit | 🟡 In progress | validator PASS + honeypot 0% (re-verified); awaiting public sandbox, then portal submit |

Legend: ⬜ todo · 🟡 in progress · ✅ done · ⛔ blocked

---

## Phase 1 — Data understanding & eval set
- [x] Confirm 100,000 rows (already unpacked, 487 MB)
- [x] Skim 30+ profiles across the quality spectrum
- [x] Identify trap examples (stuffers, honeypots, CV-subdomain, consulting, inactive) — see findings below
- [~] **Hand-label 60-100 candidates into tiers 0-4** — batch 1 done (17) in `eval/labels_batch1.csv`; batch 2 should target T1/T3 borderlines
- [x] Write the eval harness: NDCG@10, NDCG@50, MAP, P@10 — `eval/evaluate.py` (numpy-only, self-test passes: ideal NDCG@10=1.0)

### Phase 1 findings (measured on real data)
- 100k rows; 75% India; YoE median 6.8. General talent pool, not AI engineers.
- **8,545** have 3+ relevant skills, but only **755** have a corroborating title → **7,789 keyword stuffers (~10:1 noise)**.
- **Dominant signal = skill↔title↔career coherence**, ahead of raw semantic similarity. Highest lever on NDCG@10.
- Honeypots catchable via internal consistency (single role months > total experience; e.g. 166mo role vs 9.9y). ~9,745 consulting-only careers.
- Stuffer fingerprint: unrelated title + short-duration AI skills + boilerplate summary + often open_to_work=false.
- True top tier: RecSys/Search/NLP/ML eng @ product co, 5-9y, hybrid-retrieval/eval in summary, resp 0.85+, active, open. ~342 India candidates clear the strong-fit bar.
- Subdomain nuance: some corroborated "AI" titles are Computer Vision — JD demotes CV/speech-without-NLP/IR.

## Phase 2 — JD rubric (the signature move)
- [x] Extract full JD (incl. ideal-profile + hackathon-note sections)
- [x] Author structured rubric draft (domain, experience, must-haves, nice-to-haves, disqualifiers, location, product-company, behavioral, consistency)
- [x] **Human review/edit every line** — grounded in measured data values
- [x] Freeze to version-controlled `rubric.yaml`
- [x] Map each disqualifier to a concrete detection signal
- [x] Confirm disqualifier patterns fire on real profiles (validation below)

### Phase 2 validation (rules tested on labeled set + full pool)
- Detection on 17 labeled candidates: **0 false-positives** on the 4 top-fits / T3 / 4 adjacents; all 4 stuffers→STUFFER, all 3 honeypots→HONEYPOT, CV/Wipro→CONSULT.
- Prevalence across 100k: stuffers 5,466 · consulting-only 9,745 · CV-primary 86 · **corroborated-relevant 541** (the realistic top-100 universe).
- **Honeypot strategy:** use high-precision impossibility rules R1 (`role>career`, 19) + R2 (`expert proficiency with ~0 months used`, 21) as a hard FLOOR (union ~40, all genuinely impossible). Reject broad rules (expert-heavy≥5=167, skill-dur>career=5,429) — they demote legit seniors. Don't chase all ~80; domain+coherence scoring buries the rest. Verify actual top-100 honeypot rate in Phase 5.

## Phase 3 — Offline build (network OK, no clock)  ✅
- [x] Semantic layer decision: **hybrid** — TF-IDF now (zero deps, seconds), BGE drop-in later
- [x] Extract structured features → `artifacts/features.parquet` (100k × 39 cols)
- [x] Semantic similarity → `semantic_sim` column (TF-IDF cosine to JD reference text)
- [x] Build script `scripts/build_features.py` regenerates artifacts from candidates + rubric
- [x] Fixed `dq_cv_primary` over-firing (require off-domain term in title, not skill tags)
- [ ] BGE embedding upgrade — deferred; revisit in Phase 7 if eval set demands it

### Phase 3 outputs
- `src/ranker/`: `rubric.py` (load), `textbuild.py` (candidate/JD text), `features.py` (all signals)
- `artifacts/features.parquet` (100k×39), `artifacts/build_meta.json`
- Full build = 92s on 16-core CPU. Flag counts reconcile with Phase 2 (stuffer 5512, consulting 9745, honeypot_hard 40).
- **Architecture note:** features.parquet holds RAW signals; the scorer applies rubric WEIGHTS. So Phase 7 tuning re-runs only the fast scorer, never this 92s extraction.

## Phase 4 — Rank-time scorer (≤5 min, CPU, no net)  ✅
- [x] Load rubric + artifacts — `rank.py` + `src/ranker/scorer.py`
- [x] Implement scoring composition: `base_fit × behavioral − penalties`; honeypot_hard → hard floor
- [x] Sort, take top 100, deterministic tie-break (score desc, candidate_id asc — matches validator)
- [x] Emit CSV: `candidate_id,rank,score,reasoning` (100 rows, ranks 1-100, scores non-increasing)
- [x] **Timed on CPU** — full rebuild **109s** (<<300s budget); cached path 0.8s

### Phase 4 validation
- `validate_submission.py` → **"Submission is valid."** on both cached and rebuild outputs.
- Cached-path and from-scratch-rebuild CSVs are **byte-identical** (reproducible; deterministic).
- Score range 17.13..20.74, strictly monotonic; **100/100 unique scores** (tie-break correct but unexercised).
- **Top-100 trap leakage:** honeypot 0, stuffer 0, consulting-only 0, CV-primary 0, title-chaser 1. Well under the 10% honeypot-DQ line.
- Top-100 quality: **100/100 domain-corroborated**, YoE median 6.6 (in 5-9 band), 90/100 open_to_work.
- `reasoning` column currently empty by design — Phase 6 populates it.

## Phase 5 — Honeypot / consistency defense  ✅
- [x] Consistency penalty: role-vs-career (`cons_role_gt_career`), proficiency-vs-years (`cons_expert_zero_usage`), skills-vs-tenure (`cons_skilldur_gt_career` + `cons_tenuresum_gt_exp`). Built in Phase 3 `_consistency()`; honeypot_hard → floor, skilldur/tenuresum → soft −0.5.
- [x] Verify honeypot rate in top 100 well under 10% — **0%** via `eval/check_traps.py` (PASS).
- [x] No special-cased IDs — floor/penalty fall out of `scorer.py` naturally (confirmed by read).

### Phase 5 validation (`python eval/check_traps.py`)
- Top-100 honeypot (hard-floor) rate: **0/100 (0.0%)** — pool has 40. PASS (<10%).
- Other consistency/trap flags in top 100: skilldur>career 15 (benign soft signal — fires on legit self-taught seniors, kept soft on purpose, Phase 2 decision), title-chaser 1; role>career / expert-zero-usage / tenuresum / stuffer / consulting / CV-primary all **0**.
- `eval/check_traps.py` re-runnable after any scorer/rubric change; exits non-zero if honeypot rate ever crosses 10%.

## Phase 6 — Reasoning generation  ✅
- [x] Template reasoning from real profile facts — `src/ranker/reasoning.py`, deterministic at rank time (no network/LLM). Uses years, `current_title`, `matched_skills` (verbatim profile skills), fired must-haves, behavioral signals. Added `current_title`/`current_industry`/`matched_skills` raw cols to `features.py` (parquet now 100k×42); wired into `rank.py` (skipped under `--no-reasoning` for tuning runs).
- [x] Acknowledge gaps honestly; tone tracks rank — top ranks confident; rank>15 names the top missing must-have; services-industry caveat where it applies.
- [x] Variation across rows — **100/100 unique**; driven by each candidate's distinct skills + deterministic phrasing rotation (lead pool + must-have window) keyed on candidate_id. Not name-insertion.
- [x] No hallucinated skills/employers — every clause traces to a row column; skills named only from the profile's own list. Asserted in `reasoning.py` self-check (`python src/ranker/reasoning.py`).

### Phase 6 validation
- `validate_submission.py` → **"Submission is valid."** with reasoning populated.
- Rebuild flag counts unchanged (honeypot 40, stuffer 5512) → scores identical → **Phase 5 re-check still 0%**.
- Reasoning is deterministic (cached fast path reproduces byte-identical text in ~1s).

## Phase 7 — Tune  ✅
- [x] Run scorer on hand-labeled eval set — `eval/tune.py` builds a FULL 100k ranking and scores it via `evaluate.py` (subset = trustworthy with sparse labels; full = sparse-label artifact, treats every unlabeled candidate as tier-0).
- [x] Tune weights, especially behavioral aggressiveness — swept behavioral floor, SEMANTIC_WEIGHT, core_title_match, experience in_scope, retrieval must-have. **Subset NDCG@10 stays 0.9968 across every knob** → the trustworthy metric is saturated and robust.
- [x] Lock the config once NDCG@10 plateaus — **locked the existing hand-set weights; no changes made.**

### Phase 7 validation (`python eval/tune.py`)
- All 17 labels present in the ranking. Subset (orders the judged set): **NDCG@10 0.9968, NDCG@50 0.9968, MAP 0.9667, composite 0.9674.**
- Ordering: the 4 tier-4 top-fits rank above everything judged; honeypots floored to the bottom (−100); stuffers/consulting buried (pos 12k-99k).
- **Single residual inversion:** tier-2 `CAND_0000273` (BYJU'S ML eng, 5.8y in-band, predictive-modeling) sits one slot above tier-3 `CAND_0051615` (Meta search eng, 4.6y below-band, real RAG/retrieval). The human ranked domain-relevance over experience-band; the model can't separate them (both `core_title_match=1`).
- **Why no weight change:** forcing experience `in_scope` 1.2→0.4 and retrieval must-have 1.6→2.8 both leave NDCG@10 at *exactly* 0.9968 — the pair won't flip without global surgery that risks the clean tier-4 block + the 541-candidate corroborated pool. Bending global weights to fix one borderline pair on a 17-label set = overfitting. **Real fix = batch-2 labels (T1/T3 borderlines), already scheduled in Phase 1.** Decision logged below.
- Config unchanged → submission byte-identical → `eval/check_traps.py` re-run: honeypot **0/100 (PASS)**.
- Note: `full`-mode NDCG@10 IS knob-sensitive (0.25→0.33) but is **not** an optimization target — it just rewards pushing our 4 labeled tier-4s above unlabeled good picks we haven't judged. Optimizing it = overfitting to the label sample, not the role.

## Phase 8 — Repo + deck + sandbox (Stages 3-5)  🟡
- [x] Clean README: setup + single reproduce command — `README.md` (reproduce verified end-to-end, validator PASS)
- [x] `requirements.txt` with pinned versions — numpy/pandas/scikit-learn/pyarrow/PyYAML on py3.10.11
- [x] `submission_metadata.yaml` at repo root — technical fields measured; `# FILL` left for team_name/phone/github_repo/sandbox_link
- [x] **Real git history** — 9 genuine phase-by-phase commits (Phase 0→8), no single dump; 465MB data dir gitignored
- [x] Deck → PDF — `deck.md` → `deck.pdf` (Chrome headless print; one slide per page; approach, rubric, scoring, eval, sample reasoning)
- [~] Hosted sandbox — **Streamlit Cloud** chosen; `app.py` written + smoke-tested headless (100 rows, reasoning OK). **Deploy pending** (push repo → Streamlit Cloud → paste URL into `submission_metadata.yaml`)

## Phase 9 — Validate & submit
- [ ] Run `validate_submission.py` — zero errors
- [ ] Re-check common rejections (99/101 rows, rank starts at 0, dup ids, equal scores, increasing scores, wrong extension)
- [ ] Submit CSV + portal metadata + repo + sandbox link
- [ ] Keep 1 submission in reserve (3 max, last valid counts)

---

## Decisions log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-06-21 | Edge = JD-understanding layer | Aligns with the prompt's core thesis: understand the role, don't match keywords |
| 2026-06-21 | Rubric = LLM-assisted, frozen config | Smart + defensible + no network at rank time |
| 2026-06-21 | Disqualifiers = soft penalties, not hard filters | Avoid nuking plain-language Tier-5s; just demote polished-fakes |
| 2026-06-21 | Build our own hand-labeled eval set | No leaderboard; also demonstrates the eval skill the JD demands |
| 2026-06-21 | Domain coherence (title/career corroborates skills) is the #1 feature | 10:1 stuffer ratio measured; weight title-match (3.0) >> skill-tag (0.4) |
| 2026-06-21 | Use `current_industry` (not company name) for product-vs-services | Filler/consulting names dominate company head; industry is robust |
| 2026-06-21 | Honeypot defense = high-precision R1+R2 floor, not broad recall | Broad consistency rules demote legit seniors; precision > recall here |
| 2026-06-25 | Lock hand-set weights; don't tune further on 17 labels | Subset NDCG@10 saturated at 0.9968 and flat across all knobs; the lone residual inversion needs batch-2 labels, not weight-bending. `full`-mode NDCG is a sparse-label artifact, not a target. |

## Open questions
- Embedding model final choice (BGE vs E5) — decide in Phase 3
- ~~Behavioral-signal weight — settle empirically in Phase 7~~ **Settled: floor 0.55 kept; subset NDCG@10 flat for floor 0.35-0.65, only degrades at 0.75.**
- Whether any disqualifier ever needs a hard gate — revisit if traps leak into top 10

## Risks / landmines
- ⛔ Honeypot rate > 10% in top 100 → instant DQ
- ⛔ Rank step > 5 min or uses network → DQ at Stage 3
- ⛔ Flat git history / LLM-only code → fail Stage 4
- ⛔ Format violations → auto-reject before scoring
