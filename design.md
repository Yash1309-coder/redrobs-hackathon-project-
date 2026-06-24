# Design: Recruiter-Brain Candidate Ranker

**Hackathon:** Redrob — Intelligent Candidate Discovery & Ranking Challenge
**Goal:** Reach top-X (defend-your-work interview), not just a score.
**Status:** DRAFT
**Last updated:** 2026-06-21

---

## 1. Problem Statement

Rank the top 100 candidates out of a 100,000-candidate pool against a single released
job description (Senior AI Engineer, Founding Team @ Redrob), best-fit first, each with
a 1-2 sentence honest, fact-grounded justification.

The real problem is not search. It is **judgment under traps and constraints**:

- The JD is mostly *negative* signal (disqualifiers, anti-patterns). A keyword/embedding
  matcher ranks the people the JD explicitly rejects.
- The dataset is adversarial: ~80 honeypots (internally-impossible profiles), keyword
  stuffers, plain-language Tier-5s (great candidates with no buzzwords), behavioral twins.
- Hard compute limits at rank time: **≤5 min, 16 GB RAM, CPU-only, no network.**
- Multi-stage human evaluation (code reproduction, manual review, 30-min interview)
  designed so "AI-only paste-and-pray" submissions fail.

## 2. What Makes This Win ("the recruiter brain")

A single **structured fit rubric**, extracted from the JD, drives three things at once:
ranking, honeypot avoidance, and reasoning generation. One system, fully defensible.

The scoring is top-heavy — **NDCG@10 = 50% of the composite** — so the entire design
optimizes for getting the *top 10 exactly right*, not for being marginally better at
rank 80.

| Composite metric | Weight |
|------------------|--------|
| NDCG@10          | 0.50   |
| NDCG@50          | 0.30   |
| MAP              | 0.15   |
| P@10             | 0.05   |

## 3. Architecture

The compute constraint forces a clean split: intelligence is built **offline**;
rank time is fast and model-free.

```
OFFLINE (build time, network OK, no clock)        RANK TIME (<=5 min, CPU, no net)
-----------------------------------------         ------------------------------
JD --LLM--> rubric.yaml (human-reviewed, frozen)  load rubric.yaml
candidates.jsonl --> embeddings.npy (per cand)    load embeddings.npy + features
candidates.jsonl --> features.parquet             score(candidate, rubric) -> float
  (skills, tenure, company-type, consistency)     sort, take top 100
JD rubric elements --> rubric_vectors.npy         generate reasoning from facts
                                                  write submission.csv
```

Rank-time loads precomputed vectors/features and does dot products + arithmetic.
No embedding model is loaded at rank time, so the 5-minute budget is never at risk.
(Spec 10.3 explicitly allows shipping precomputed artifacts, or a script that builds them.)

### 3.1 The JD-understanding layer (signature move)

The JD is turned into a structured rubric **once, offline**, using an LLM, then
**human-reviewed and frozen** into a version-controlled `rubric.yaml`. Network is used
only at build time (allowed); rank time reads the static file. This is the artifact we
open and defend in the interview.

The rubric has three buckets. The **negatives are the differentiator** — most teams
encode only positives.

**A. Must-haves (semantic + structured)**
- Embeddings-based retrieval experience (production)
- Vector DB / hybrid search infrastructure
- Strong Python / code quality
- Ranking evaluation literacy (NDCG, MRR, MAP, offline-to-online)

**B. Disqualifiers — each maps to a *detectable pattern*, not a vibe**

| Anti-pattern (from JD)        | Detection signal in structured data |
|-------------------------------|-------------------------------------|
| Research-only                 | All career_history titles/companies academic/lab; no "production/shipped/users" language |
| Consulting-firm-only          | Every `company` in {TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, ...}; no product company ever |
| Title-chaser                  | Tenure consistently < ~18 months across many roles |
| Architect who stopped coding  | Recent titles "Architect/Tech Lead" + summary lacks hands-on signals |
| LangChain-only / <12mo LLM    | AI skills all `duration_months` < 12, no pre-LLM ML history |
| CV/speech/robotics, no NLP/IR | Skill set centered on vision/speech; no retrieval/NLP |

These act as **soft penalties / gating** that pull polished-on-paper candidates *out of
the top 10* — exactly what a recruiter does reading past buzzwords.

**C. Behavioral modifiers (redrob_signals)**
A perfect-on-paper candidate who is inactive and unresponsive is not hireable. Multiply
skill-fit by an availability/engagement factor built from `recruiter_response_rate`,
`last_active_date`, `open_to_work_flag`, `interview_completion_rate`.

### 3.2 Scoring model (transparent composition)

Conceptually (not final code):

```
final = semantic_fit
        * must_have_coverage
        * behavioral_availability
        - disqualifier_penalties
        - inconsistency_penalty
```

Every term traces to a fact in the profile. A black-box ranker that scores marginally
higher but cannot be explained loses this hackathon at Stage 5.

### 3.3 Honeypot / consistency defense

An internal-consistency check demotes impossible profiles and keyword stuffers using the
same machinery:
- experience vs company age (8 yrs at a 3-yr-old company)
- proficiency vs years-used ("expert" in 10 skills with 0 years)
- skill count vs total tenure

This keeps honeypots out of the top 100 (**>10% in top 100 = disqualification at Stage 3**)
and strengthens the "we actually read profiles" story. We do **not** special-case
honeypots by ID; they fall out of the consistency penalty naturally.

**Measured honeypot strategy (Phase 2):** precision beats recall. Two high-precision
impossibility rules — `role_duration > total_experience` (19 hits) and `expert
proficiency with ~0 months used` (21 hits) — are genuinely impossible and used as a
hard floor (~40 candidates). Broad rules (expert-heavy≥5 = 167, skill-dur>career =
5,429) are rejected: they demote legitimate senior engineers and cost score. We do not
chase all ~80 honeypots — they're low-quality profiles that domain + coherence scoring
already buries; the floor is a safety net. Actual top-100 honeypot rate is verified in
Phase 5.

### 3.4 Reasoning generation

Because each candidate carries structured fit-scores and flags, reasoning is templated
from **real facts** deterministically at rank time (no network):
specific (years, title, named skills, signal values), honest about gaps, varied, and
rank-consistent. This is precisely the Stage-4 manual-review rubric.

## 4. Approaches Considered

### Approach A — Hand-authored rubric
Read the JD as a recruiter; hand-write rubric + detection rules.
- **Pros:** maximum defensibility, fully transparent, no LLM dependence.
- **Cons:** manual, may miss nuance.

### Approach B — LLM-assisted, frozen config  **(CHOSEN)**
LLM extracts the JD into a structured rubric offline; human reviews/edits and freezes it
to `rubric.yaml` that rank-time reads.
- **Pros:** best blend of smart + defensible + fast; network only at build time;
  clean artifact to defend.
- **Cons:** must genuinely review the LLM output, not rubber-stamp it.

### Approach C — General automated JD parser
A reusable parser that turns *any* JD into a rubric.
- **Pros:** strongest "this is a real product" story.
- **Cons:** harder, riskier, may underfit THIS JD's specific negatives.

**Recommended:** Approach B. It captures LLM smarts on JD reading while leaving us a
transparent, human-owned artifact for Stages 4-5.

## 5. Validation Strategy (no leaderboard → build our own eval)

There is no live leaderboard and a 3-submission cap. We validate by methodology:

1. Hand-label ~60-100 candidates into relevance tiers (0-4) against the JD.
2. Measure NDCG@10 / NDCG@50 / MAP / P@10 offline on that set.
3. Tune rubric weights (esp. behavioral aggressiveness) against the offline eval.

The JD *explicitly* demands "designing evaluation frameworks for ranking systems." Doing
this IS demonstrating the exact skill being hired for — a strong interview moment.

## 6. Why This Wins Each Stage

- **Stage 2 (score):** semantic recall finds the right people; disqualifier penalties +
  behavioral multipliers sharpen the top-10 (where 50% of the score lives).
- **Stage 3 (sandbox + honeypots):** precomputed artifacts → fast & reproducible;
  consistency penalty keeps honeypots out of top 100.
- **Stage 4 (reasoning + git):** reasoning from real facts → specific, honest, varied;
  commit in real increments (no single dump).
- **Stage 5 (interview):** every number is explainable. We built a system, not a prompt.

## 6.5 Data Findings (Phase 1 — verified on the real 100k)

Measured directly from `candidates.jsonl`, these reshape the priorities:

- **100,000 rows. 75% India**, rest spread across USA/Australia/Canada/UK/etc. YoE median 6.8 (p25 3.9, p75 9.9) — the 5-9 band is the bulk.
- **The pool is a general talent pool, not AI engineers.** Skills are near-uniformly distributed (figma, salesforce, accounting rank as high as kafka/airflow). The relevant population is the needle.
- **"Has the skills" is NOT discriminative.** 8,545 candidates carry 3+ relevant retrieval skills, but only **755** also have a corroborating title; **7,789** are title-mismatched (keyword stuffers). That **~10:1 noise ratio is the keyword-stuffer trap, quantified.**
- **The dominant signal is skill ↔ title ↔ career coherence.** Example stuffers found: a 1.9y Operations Manager, a Graphic Designer @ TCS, a Mechanical Engineer, and a Customer Support rep all listing the full FAISS/RAG/embeddings stack — with those skills dated only 4-18 months and boilerplate "recently excited about AI" summaries. A recruiter laughs; cosine similarity ranks them top-10.
- **Stuffer fingerprint:** unrelated current_title + AI skills with short `duration_months` + generic summary ("driving outcomes in my domain... recently excited about AI") + often `open_to_work=false`.
- **Honeypots are catchable by internal-consistency, confirmed.** Real examples: single career roles of 166 / 171 / 144 months that exceed the candidate's total stated experience. A `single_role_months > years_of_experience*12 + margin` check flags them cleanly. Also: expert-proficiency-heavy profiles (≥5 "expert" skills; only 1,311 "expert" tags exist in the whole pool, so this is a strong outlier). ~9,745 careers are consulting-firm-only.
- **True top-tier looks like:** RecSys/Search/NLP/ML Engineer at a product company (Swiggy, CRED, Paytm, Meta, Zoho), 5-9y, summary explicitly mentioning hybrid retrieval / ranking / NDCG-MRR eval, high `recruiter_response_rate` (0.85+), recently active, `open_to_work=true`. Roughly **342** India-based candidates clear a corroborated-title + 4.5-9.5y + 4+ relevant-skills bar — the top-100 lives in a refined slice of these.
- **Subdomain nuance is real:** several corroborated "AI" titles are **Computer Vision Engineers** — which the JD explicitly demotes (CV/speech without NLP/IR). The rubric must separate NLP/IR/retrieval/recsys/search from CV/speech/robotics.

**Implication for the rubric:** a `coherence` term (does title/career/summary corroborate the claimed skills?) is the highest-leverage feature, ahead of raw semantic similarity. This is the single biggest lever on NDCG@10.

First hand-labeled batch (17 candidates spanning all tiers) lives in [eval/labels_batch1.csv](eval/labels_batch1.csv).

## 7. Open Questions

- **Embedding model:** BGE-small/base or E5 (JD name-drops both). Offline, so CPU/GPU
  both fine; pick by retrieval quality.
- **Behavioral weight:** too high buries a perfect skills-match who is merely inactive.
  The hand-labeled eval set settles this empirically.
- **Soft vs hard on disqualifiers:** default soft penalties; revisit only if honeypots
  or stuffers leak into the top 10 on the eval set.

## 8. Compute & Submission Constraints (must not violate)

- Rank step ≤ 5 min wall-clock, ≤ 16 GB RAM, CPU-only, no network, ≤ 5 GB disk.
- Output: `team_xxx.csv`, columns `candidate_id,rank,score,reasoning`, exactly 100 rows,
  ranks 1-100 each once, scores non-increasing, every id exists in candidates.jsonl.
- Run the provided `validate_submission.py` before every upload.
- 3 submissions max; last valid one counts.

## 9. Deliverables

1. **GitHub repo** — clean, reproducible. README with a single command:
   `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`.
   Includes precomputed artifacts (or a build script), `requirements.txt`,
   `submission_metadata.yaml`.
2. **Deck → PDF** — approach, why, how it works (the rubric, the scoring composition,
   the eval, sample reasoning).
3. **Ranked CSV** — the top-100 submission.
4. **Sandbox link** — hosted env (HF Spaces / Streamlit / Replit / Colab / Docker / Binder).

## 10. The Assignment (next real-world action)

Before any ranking code: **hand-label 20 candidates** from `sample_candidates.json`
against the JD into tiers 0-4, writing one honest sentence each on why. This bootstraps
the eval set, pressure-tests the rubric buckets against real profiles, and tells you
immediately whether the disqualifier patterns are detectable in the actual data.

## 11. What We Noticed (founder signal)

- You picked "win top-X, go deep" and "JD understanding layer" — you optimized for the
  defensible system over the quick score. That's the exact instinct this challenge rewards.
- You chose "LLM-assisted, frozen config" over both the lazy (full-LLM) and the heroic
  (general parser) paths. Reading the tradeoff correctly is the skill the JD is hiring for.
