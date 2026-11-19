# Novelty-gated inquiry ("it asks about what is new to it", milestone 11-30): record and design

Written 2026-11-18. Capability: Curiosity. Produced by a record sweep (4 readers: brain/ code, old-track record,
biology, whole transcripts), three independent designs and six skeptic reviews (workflow wf_f4d7d55f-1b8). Below is
the synthesis as returned, unedited. It is a DRAFT: stage 0 (`experiments/novelty_stage0.py`) runs first and its
numbers decide whether the stage-1 build and pre-registration go ahead as written.

## Stage 0 (added 11-18, before the 1.5B run)
- Measurement only, no switch: 12 matched frames x {common, rare-real, coined} item; the cortex's surprisal of the
  item from the birth state; the same item in a follow-up frame after the brain heard and answered the statement
  (habituation); and the unchanged brain's reply (asks about the item: "?" and the item named).
- 0.1B smoke, seed 42: coined vs common AUROC 1.00, coined vs rare-real 0.99 (summed nats; first token 0.68),
  per-token surprisal C 4.2 / R 5.3 / N 9.0 nats, repeat-mention drop N 22.7 nats; asks about the item C 0.25,
  R 0.08, N 0.25 (no difference). Example: "We had frunkets for dinner last night." -> "That sounds like a great night!"

## Synthesis (workflow output)

**Winner: test-first, with grafts.** It keeps test-first's bank, arms and difference-in-differences criteria. From signal-first it takes habituation driven by the working state and the shuffled-arousal null. Learned-behaviour's sleep method becomes the stage-2 fallback, with its own flaws fixed. The changes below answer every fatal flaw the four judges raised.

**How each fatal flaw is fixed**
- **Novelty is rarity, not "new to me".** The signal is now N = the largest per-word surprisal read from the working state, minus `cue_response`. Read from the working state, a repeat within the day gets less surprising (habituation). `cue_response` is the existing gain the consolidated synapses give a cue, so whatever was slept gets less surprising across days. This reuses an existing asset instead of building a new one. N against R (rare-real items) becomes a formal criterion.
- **The bar is fitted on the test distribution.** The bar is now the 95th percentile of N on common-item and chat turns only, a false-alarm rate. Coined items are never used to fit it. Dev also uses a different syllable inventory, 12 other frames and seeds 1140-1142.
- **The familiarity gate may be broken** (it is lexical, frames are shared, and it may not fire on repeats). A stage-0 check measures it before any building. The gate is used only if its AUROC is at least 0.70. Otherwise habituation comes from the working state alone, with no host gate.
- **Echo questions, judge noise, 20% coverage.** The blind judge scores 100% of replies. Q2 states that an echo ("Quaxlin?") or a generic "What does it mean?" counts as no. Judge noise is measured on dev with two judges.
- **Power.** Each class gets 12 turns with 4 samples per turn, so 48 per class per seed. Criteria are pooled with a seed-bootstrap 90% CI, plus a sign test of at least 5 of 6 seeds.
- **The "?" axis may be tiny or empty.** It is harvested from the banked 11-18/11-16 day-1 transcripts (16-25 asks per 72 turns per seed, across 26+ seeds). Each class needs at least 40 replies. Two axes go to dev: all asks, and asks that name the item.
- **Read order and numerics.** With the switch off, the code keeps today's single read. With it on, the order is recall/reinstate, then the user turn is read from `active`, then N, then the delta, then the `Assistant:` header. `reply_monitor` resamples from the same logits and state, so it keeps the delta.
- **Scope creep.** The `salience` change to storage is dropped.
- **Train/test mismatch in the stage-2 fallback.** Replay injects D at the same point as at test (after the user turn). It stores an exchange log, and answered asks on low-drive turns are kept out of the no-D contrast set.

**(A) Record brief**
- **What we knew:** the 1.5B cortex already asks generic follow-ups on 16-25 of 72 day-1 tell turns ("What does it mean?" for Henrietta). Additive state deltas at 1.5B moved style, and random directions moved it too (mood iterations 1-3). Hippocampal familiarity is word-overlap and failed to generalise (abstain 11-03, M1 AUROC 0.40-0.59). Few-example nights distil unconditionally (K2, teacher-night 11-17).
- **What is new:** a surprise signal from the cortex's own prediction error, which PREREQUISITES lists as "available but unused", gating a state drive.
- **Prerequisites met:** cortex logits, `cue_response`, `add_delta`, the `Affect` axis pattern.
- **Missing:** a learned gain (it is a scaffold here) and a place to ask (only the reply).
- **Expected:** stage 0 shows N separates invented from common items, and invented from rare-real only partly. Stage 1 is most likely to fail P3 (RAND or SHUF about equal to ON), which would be a verdict on state-delta steering. About 30% chance of an all-pass.

**(B) Build plan (smallest first)**
1. `brain/cortex.py` `read_full(text, state)`: logits for every token, the state, and per-token NLL. `LiveCortex` override uses `p(ids, st)[0]`, as `cue_response` does. About 20 lines.
2. `brain/inquiry.py`:
   - `novelty(brain, text, st)`: max-word NLL minus `cue_response`, gate optional. About 30 lines.
   - `build_axis(rows, variant)`: from banked replies, labelled by trailing "?" and, for the item variant, item token present; minimum count of 40. About 40 lines.
   - `enabled` lesion switch. About 10 lines.
3. `brain/brain.py`:
   - `Brain(inquiry=False, inquiry_gain, inquiry_bar, inquiry_axis)`.
   - The split read in `hear` from line 415, used only when the switch is on.
   - `info["novelty"]`, `info["drive"]`.
   - About 30 lines.
4. `brain/eval/open_conv_bank.py`: the C/R/N Latin-square bank, repeats, nonsense, dev bank. About 60 lines. `brain/eval/blind_ask_judge.py` (Haiku, ZDR path). About 80 lines. Arm flags in `open_conv.py`. About 25 lines.
5. `brain/tests/test_inquiry.py`:
   - off is byte-identical to today;
   - `enabled=False` is identical to off;
   - the split read and the single read agree on CPU fp32.
   - About 40 lines.
6. `SCAFFOLDS.md`: 3 rows (bar, gain, "?" label), each with a target column. `PREREQUISITES.md`: Curiosity row.
7. **Smoke:** 0.1B, seed 42, 10 turns, OFF and ON, on the GPU queue. Then stage 0 (signals only, no generation, 1.5B, pool, about 1 h).

**(C) Pre-registration draft**

**Pre-registration: novelty-gated inquiry drive (1.5B), milestone 11-30 "it asks about what is new to it"**

**Method.**
- **Bank:** 24 frames × {C, R, N}, rotated by Latin square across seeds. Twelve of each class per conversation, 6 repeat-N turns about 8 turns after the explanation, 4 nonsense turns, and chat turns in between. About 50 turns.
- **Stage 0 (no generation):** AUROC of N for N vs C, N vs R, and first vs repeat. The gate is kept only if its AUROC on first vs repeat is at least 0.70.
- **Dev:** seeds 1140-1142, dev bank. Choose the axis variant and gain from {4, 8, 16, 32} as the largest gain whose coherence is no more than 0.3 below OFF. Then freeze.
- **Test:** seeds 42/43/44/100/101/102.
  - Arms: OFF; ON; RAND (matched norm per layer, 3 draws); SHUF (real axis, ON's firing rate, random turns).
  - Day 2 runs for OFF and ON only.
  - Pool CPU fp32, queued after the mood iteration 4 test. About 30 node-hours.
- **Scoring:** AskItem = judge Q2 yes, where echoes and generic asks do not count. Regex "?" plus item token is reported as a cross-check.

**Criteria (all required; Δ = ON − OFF).**
- **P1:** DiD [AskItem(N) − AskItem(C)] Δ ≥ 0.20 pooled, CI lower bound > 0, and positive on at least 5 of 6 seeds.
- **P1b:** DiD [AskItem(N) − AskItem(R)] Δ ≥ 0.10 pooled.
- **P2:** |Δ AskItem(C)| ≤ 0.05, two-sided. Δ AskItem(R) ≤ +0.10.
- **P3:** DiD for RAND and for SHUF ≤ 0.5 × ON's DiD.
- **P4:** ON AskItem(repeat N) ≤ AskItem(C) + 0.05, and ≤ first-N rate − 0.15.
- **P5:** coherence of ON is no more than 0.3 below OFF. Day-2 inventions plus intrusions rise by at most 2 per seed. Named-right falls by at most 1 of 6. Asks on nonsense turns ≤ asks on N turns.
- **P6:** a whole-transcript read, blind to arm, finds no new defect class.
- **Lesion:** `enabled=False` is byte-identical to OFF on 6 of 6 seeds. Replies differ on at least 50% of N turns.

**Decision branches.**
- **All hold:** land-result skeptic, then flip `inquiry` on for 1.5B and report through the weekly. The milestone is recorded as met at scaffold level (bar and gain are host-set).
- **P1 holds, P3 fails:** a verdict on additive steering. Next is stage 2, sleep-learned asking, on fresh seeds.
- **P1 holds, P1b fails:** the signal measures rarity. Next is the post-sleep `cue_response` term with a day-1 night before testing.
- **P4 fails:** habituation fails. Report it and try the S_all slept familiarity.
- **OFF already shows N−C ≥ 0.20:** narrowed. Report that the capability is present from pretraining.

**(D) Open questions only the evidence can settle**
- Is max-word NLL at 1.5B much higher for "Hogmanay" or "bassoon" than for "Max", and how close is it to coined words? This is the stage-0 N-vs-R AUROC.
- Does the RWKV working state lower a coined word's surprisal on its second mention enough for habituation without the hippocampal gate?
- Does the split read reproduce the single read byte for byte on CPU fp32?
- Are there at least 40 item-naming asks among the banked transcripts, and does that axis carry reference to the item rather than question form only?
- What is the judge's Q2 agreement rate (two judges, plus the owner's 10% spot-check), and is it tight enough for a 0.20 threshold?
- Do RAND or SHUF move AskItem at the chosen gain, as random directions moved mood?

I edited nothing. Grounding: `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/brain.py` (`hear` at line 386, `cue_response` at 145, `knows` at 458, `sleep_extra` at 300/572), `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/PREREQUISITES.md` (Curiosity at line 101), `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/research/2026-11-18-sall-open-conv-1.5b.md` (style model).
## Stage 0 result (11-18, 1.5B, six seeds)
Signal AUROC 1.00 (N vs C and N vs R, summed nats), habituation AUROC 1.00; behaviour shows no selectivity
(asks what it is ~10/72 N vs ~8/72 C) and invents meanings for coined items. Stage 1 goes ahead; target measure =
asks what the new thing is (blind reader) + invented meanings must not rise. `results/2026-11-18-novelty-stage0-1.5b.md`.
