# Roadmap if the LLM mouth is permanent — grounded in our record

*2026-09-19. Answer to the owner's question: "we leaned on an LLM mouth before and still hit the same limits —
what should the roadmap actually be if we keep it?" Produced by an 8-agent deep-RAG workflow over our findings +
adversarial verification (verdict SOUND-WITH-CORRECTIONS; corrections folded in below). Live board:
[GAP_CLOSURE_MISSION.md](../../GAP_CLOSURE_MISSION.md). Supersedes the mouth sections of
[2026-09-18-strategic-plan-...](2026-09-18-strategic-plan-faithfulness-tradeoff-and-the-data-lever.md). Two items
need OWNER RATIFICATION (marked ⚖️).*

## The one-line answer
The mouth was never the bottleneck. Keeping the LLM stops being "retire the mouth" and becomes "prove the brain
is the load-bearing cognizer under it." The owner's remembered limits live in three different places and we kept
re-attacking the wrong one.

## 1. Premise check — did the LLM mouth also hit these limits? Mostly NO (grounded)
The named limits decompose by WHERE they live:
- **Fluency / broad open prose = the MOUTH, and the LLM SUCCEEDS at it.** The fluency WALL was hit every time we
  tried to RETIRE the LLM: the from-scratch spiking own-voice mouth falls *below a fair trigram* on broad-domain
  WikiText-103 (2026-09-04-fluency-scale-wt103-linattn-below-trigram), every architecture lever came back flat
  (2026-09-05-mouth-objective-lever-flat...architecture-is-wrong-axis-NO-GO), and raw scale is falsified
  (2026-09-18-scaling-go-no-go... + today's decisive extended-d384 `NO-GO-CAPACITY-SATURATED`, flat 0.0045 nats
  over 8× tokens, +0.17 above the band). **"We leaned on an LLM and still hit fluency limits" is not what
  happened — we hit them whenever we took the LLM away.**
- **Scaffold burndown = INTEGRATION/purity, and it IS the mouth — as a ledger fact, not a capability fact.** ~56 of
  ~69 production rows are `BLOCKED:neural-render` because the brain does not yet render its own PROSE (Qwen OR a host
  template — SVO/RF-composer). The content, recall, affect, GNW ignition behind those rows are already brain-driven
  and on-by-default. The mouth blocks the RETIREMENT column, not the capability.
- **Learning / growth retention = the BRAIN, recurs regardless of mouth.** Real walls are memory-architecture:
  pure-potentiation BTSP degrades weak-cue recall under a graded instrument (2026-08-27-graded-recall...NOGO); the
  replay generator is still host `np.mean`, not a spiking CA3 (2026-08-20-sleep-replay...GO-on-the-principle); the
  deep off-diagonal decorrelation wall is unbuilt (2026-06-15-natural-learning...locality-wall). No mouth decision
  touches these.
- **Hallucination = mostly solved, with ONE real residual the mouth choice DOES bear on (verifier correction).**
  The ABSTAIN path is generator-agnostic (byte-identical honest-abstain regardless of generator,
  2026-09-04-onebrain-stage1...GO; prompt-only honesty categorically FAILED, 2026-08-21, so the post-hoc moat
  stays). BUT Qwen's KNOWN-topic open-PROSE confabulation is NOT caught by the post-filter — NP_ENTAILMENT changed
  0/12 real replies (2026-09-01-open-ended-bundle-moat-safety-soak) because free prose is outside its parse scope;
  it was contained only by BYPASSING Qwen for lexicon-covered topics (spiking render_fact_sentence, 2026-09-02).
  **A permanent open-prose Qwen mouth leaves this residual LIVE on the out-of-lexicon slice — a tracked cost, not
  "solved."**

## 2. Solved vs active-frontier vs genuine-wall
| Capability | Status |
|---|---|
| Honest-abstain / hallucination on lexicon-covered topics | ✅ solved, generator-agnostic |
| Open-prose confabulation (out-of-lexicon, Qwen free prose) | 🧱 tracked residual of a permanent open-prose mouth |
| Fluency via LLM mouth | ✅ (Qwen is the default renderer today) |
| Fluency via a *spiking* mouth at deployable scale | 🧱 GENUINE WALL — falsified 3 ways; do not re-attempt |
| Between-turn learn/retain, curated regime | ✅ 43-seed/120-day GO; sleep-replay default-on |
| Retention under graded instrument / raw corpus | 🧱 memory-architecture wall (pure-potentiation NO-GO; off-diagonal) |
| Broad knowledge (store breadth) | 🔬 corpus-provisioning problem, not capacity |
| Open-ended self-driven discussion | 🧱 unbuilt organs (forward model, pragmatics, episodic) |
| Faculties load-bearing on the LIVE mouth | 🔬 partial — affect NO-GO live (2026-09-04), curiosity-calibration HELD (2026-09-17, now fixed+validating) |

## 3. STOP doing (falsified levers — do not re-attempt)
- Scaling a spiking mouth to broad fluency (below-trigram; architecture flat; raw scale falsified).
- Buying hardware for fluency (explicit NO-GO 2026-09-18).
- Prompt-only honesty (categorically failed; the post-hoc moat is mandatory).
- Pure-potentiation BTSP as the retention write (graded NO-GO).
- ⚖️ Treating **% scaffold_retired** as the headline metric — it structurally cannot move while the mouth is kept,
  so it reads as permanent failure and drives us back to the falsified retirement chase. **Relaxing this formal
  success condition (GOAL: done = production-default + scaffold-retired) needs owner ratification.**

## 4. ⚖️ The new #1 metric (proposed, replaces "retire the mouth")
**Load-bearing fraction: the % of production faculty rows where LESIONING the brain's contribution provably
changes the reply** (vary state → reply differs; lesion → difference vanishes). Encodes the standing bar
(feedback_faculties_must_drive_not_observe) + the scaffold-acceptance condition
(feedback_scaffold_ok_as_conditioned_articulation_if_faculties_load_bearing). The mouth becomes acceptable BY
DEFINITION once this is high: the LLM words, the brain decides. This is the instrument that keeps the concession
honest.

## 5. The roadmap (mouth-independent, ordered)
**NEAR (weeks, cheap/de-risked):**
1. Settle the decisive scaling test (done: NO-GO-CAPACITY-SATURATED, 1 seed; 6-seed confirm pending) → ⚖️ formally
   accept Qwen as permanent conditioned articulation and CLOSE the mouth-retirement arc (don't leave it a silent
   aspiration).
2. Re-instrument the ledger around the load-bearing metric: split the ~56 blocked rows into "content brain-driven,
   wording outsourced → ACCEPTED" vs "content still host → real work." Only the second set is genuine scaffold.
3. Ship Stage-3 single brain-state-driven reply dispatch (record calls it "just do it").

**MID (the real open work — mouth-independent):**
4. **HARD GATE on calling the concession "principled" (verifier correction — sequence this FIRST, blocking):**
   close the two measured faculty-drive gaps — affect-hollow on the live mouth (2026-09-04) + the
   curiosity-calibration flip (2026-09-17, fixed + validating this session). Until these pass lesion tests, the
   mouth is decorated, not driven.
5. Retention architecture: add the heterosynaptic-depression / competitive-normalization arm (verify it is the
   graded-NO-GO's own named fix before building); build a spiking CA3 autoassociative store to replace host
   `np.mean` replay.
6. Breadth = curation, not scale: provision the knowledge-store corpus (capacity is not the ceiling); any residual
   mouth-fluency work is data-quality/curation at Chinchilla-matched ratio — gap-analysis on existing infra
   (corpus_stream / curriculum / tokcache), not greenfield.

**LONG (disclosed residuals, months, owner-gated):**
7. Off-diagonal decorrelation / dendritic-plus-lateral architecture — the deep real-corpus retention wall (verify
   the exact rank figures against a raw artifact before quoting; the finding reports +0.35 peak vs +0.518 offline).
8. The three unbuilt organs for true open-endedness (forward model, pragmatics/consequence-of-speaking, episodic
   memory) — independent of the mouth decision; do not gate on it.
9. Named permanent host residuals under the LLM-mouth choice, disclosed not hidden: the moat's own decision logic
   (`moat-verify: BLOCKED:neural-render`); the open-prose hallucination residual (§1); mouth substrate-fusion is
   FEASIBLE + local on the 3090 (~11.9 GB dense; VRAM is not the wall — wall-clock is), deprioritized not blocked
   (verifier correction to an earlier VRAM overclaim).

## 6. The honest tension — principled scaffold or concession?
It is a **permanent concession on the language-PRODUCTION faculty** (Broca/motor-speech is a real human faculty we
would outsource to a transformer) — a genuine, disclosed hole in the "complete + faithful emulation" bet, not a
caveat to wave away. It is made PRINCIPLED — not a cheat — ONLY by the load-bearing guarantee: the LLM is confined
to FORM and never asserts its own pretrained CONTENT; the honesty read-outs + moat are generator-agnostic; cognition
stays on the one spiking substrate; and we bank the falsified method (spiking-mouth-to-fluency) rather than abandon
the capability (fluent grounded honest conversation). **The moment the brain's faculties stop demonstrably driving
the reply, the LLM becomes the cognizer and we are an LLM-plausible-text system wearing a brain — the exact thing
the project was founded to reject.** That is why §4's metric is load-bearing, and why §5.4 is a hard gate.

## 7. Owner ratifications — BOTH RATIFIED 2026-09-19
- ✅ **RATIFIED: Qwen is the permanent conditioned-articulation mouth** — the spiking-mouth-to-fluency arc is CLOSED
  as falsified (banked method, not abandoned capability).
- ✅ **RATIFIED: "load-bearing fraction" is the new #1 metric** in place of "% scaffold_retired."

This is now the ACTIVE arc. Immediate build order: (a) build the load-bearing-fraction instrument (lesion-verified,
reusing the regression-battery per-faculty probe infra); (b) re-instrument the ledger (ACCEPTED vs real-work split);
(c) close the two faculty-drive gaps (affect-hollow-live + the curiosity-calibration flip, validating this session)
as the HARD GATE. Heavy runs go via tools/memcap.sh; no long local runs while the owner games.
