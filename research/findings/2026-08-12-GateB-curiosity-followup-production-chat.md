---
type: finding
status: contributing
date: 2026-08-12
mechanism: D3 CURIOSITY (crave-drive) WIRED into the DEFAULT /api/brain-chat turn — an honest FOLLOW-UP QUESTION on a NOVEL topic. On an ABSTAIN (the no-confab moat refused -> the brain holds no answer -> a maximal epistemic gap) the brain's own gap feeds the `from_novelty` neuromodulator (already committed additive/default-off in sim/) -> an excitability_drive on a spiking ASK pool -> the ASK pool SPIKES; the wanting is read DIRECTLY off cp_firing_states[ask] (reuse-by-import from research/runners/curiosity_production_organ.py -> the DR-1 crave-drive, on-bridge 6-seed CPU GO + 6/6-SAFE in Stage-A step-3; corr(gap,SPIKING-want)=+0.996 reproduced numpy-CPU). When the ASK pool CRAVES (want>=threshold) the brain APPENDS an honest FOLLOW-UP QUESTION — crave, don't refuse. The moat is INVERTED, not broken: the answer stays an abstain (never a confabulated fact); the added text is unambiguously a QUESTION. Default-ON, moat-safe, lesion-load-bearing, NO sim/ edit.
lane: Gate-B / D3 · Curiosity (crave, don't refuse — an honest follow-up question on a novel topic)
lane_ref: D3
verdict: GO / WIRED (production-integration). Single-process synchronous in-process verify on the real /api/brain-chat handler (SIM_BACKEND=numpy, GPU-free stub renderer, tiny-demo; the curiosity block runs in BOTH the rich and single-fact paths). 24/24 verify checks pass. The load-bearing on-bridge number reproduced numpy-CPU: corr(gap, cp_firing_states[ask])=+0.996, lesion->0 asks.
seed-waiver: production-INTEGRATION verify of an already-6-seed / 6/6-SAFE faculty (the DR-1 crave-drive: on-bridge 6-seed CPU GO 2026-07-23/24 `_curiosity_seek_learn_onbridge_derisk.py`; crave-on-spikes 6/6-SAFE in the Stage-A step-3 integration 2026-08-07). This doc verifies the deterministic WIRING glue on the real handler (single process, one seed=42 organ) + reproduces the load-bearing corr/lesion numpy-CPU. Lesion + flag-off + additivity arms are decisive on the single wired seed.
artifacts:
  - research/findings/raw/_gateB_curiosity_production_verify.json
---

# Gate-B / D3: an honest curiosity FOLLOW-UP QUESTION on a novel topic (crave, don't refuse)

**Status:** GO / WIRED. On a topic the brain does NOT hold (an ABSTAIN), it no longer only refuses — it
CRAVES to learn and ASKS: "I don't know about that. My curiosity is piqued — I haven't learned about
`<topic>` yet: what can you tell me about `<topic>`?" The decision to ask is a genuinely-SPIKING ASK-pool
firing read, not a host `if abstain` flag.

## Per-GO VERIFICATION of the D3 claims (verify-first — the E1 mis-read lesson)

The burn-down D3 row claimed three GOs. Verified against the runners' OWN verdicts, only ONE is a genuinely-
spiking, wireable mechanism:

- **Curiosity-inversion (the crave-DRIVE): REAL + spiking + wireable.** The DR-1 on-bridge realization
  (`_curiosity_seek_learn_onbridge_derisk.py`) is genuinely spiking: the epistemic-gap scalar
  `current_novelty_signal` drives the `from_novelty` neuromodulator -> an `excitability_drive` on a spiking
  ASK pool -> the wanting is read off `cp_firing_states[ask]`. Reproduced numpy-CPU (seed 42, `--smoke`):
  **corr(gap, SPIKING-want) = +0.996**, lesion (`curiosity_excit_sensitivity=0`) **-> 0 asks**,
  permuted corr -0.61, yoked masters 1<3 — the runner's OWN verdict is **GO**. The one `sim/` edit it needs
  (`from_novelty` in `sim/neuromodulators.py` + the `current_novelty_signal`/`novelty_baseline` config
  fields) is ALREADY committed in `main`, additive + default-off + byte-identical when unused -> **NO new
  `sim/` edit** to wire. **This is the wired deliverable.**
- **Learning-progress SELECTION: host-formula shortcut + on-bridge seed-fragile.** The LP-max ask SELECTOR
  (2026-08-07) is explicitly "a CPU numpy PROXY: the LP traces are numpy EMAs, not spiking pools", and the
  on-bridge substrate-memory promotion of the LP pools is NEGATIVE/PARTIAL (1/6, seed-fragile). NOT wireable
  as a spiking mechanism, and NOT NEEDED here (a single-topic chat follow-up asks about the one gap the user
  surfaced — there is no multi-armed budget to allocate).
- **Curiosity-VETO: host-formula (survives the critic lesion).** The reward-omission veto's default value is
  a HOST ELP TD tracker — proven a shortcut because it SURVIVED the GABA_B critic lesion 6/6
  (2026-07-31: "the striosome is not load-bearing"). The `--spiking-veto` variant reads the striosome value
  but is a separate, more fragile path. NOT wireable as-is, and NOT NEEDED (a per-turn chat follow-up has no
  noisy-TV budget to protect).

So the wire uses ONLY the crave-DRIVE (the genuinely-spiking, 6-seed / 6/6-SAFE part). The SELECTOR + VETO
remain the named next rungs, honestly un-wired.

## The wire (reuse-by-import; NO `sim/` edit)

`research/runners/curiosity_production_organ.py` builds ONE co-resident curiosity bridge
(`build_curiosity_bridge`, reused from the DR-1 de-risk) and calibrates a curious-vs-incurious ASK-pool
firing threshold from a NOVEL vs FAMILIAR novelty battery. On an ABSTAIN the brain's own epistemic gap (it
holds no answer -> a maximal-novelty scalar) is delivered as the `from_novelty` drive; the settled ASK-pool
rate off `cp_firing_states[ask]` IS the wanting. A rate above the calibrated threshold appends an honest
FOLLOW-UP QUESTION. `webapp/server.py brain_chat` runs the read ONLY on an abstain (both the rich and the
single-fact paths), so it never manufactures a fact, changes an answer, or flips an abstain into an assert —
it only APPENDS a QUESTION (the moat INVERTED, not broken; the abstain still confabulates nothing).

The NOVELTY DERIVATION (the abstain = the brain's own memory read: it holds no answer) is a declared host
boundary — the SAME uncertainty signal that drives the no-confab moat (the DR-1 inversion). The ASK-pool
firing read + the threshold decision are the load-bearing SPIKING part — exactly the surprise organ's pattern
(host sensory encoding + spiking mismatch read) and the metacog organ's (host evidence derivation + spiking
balance read). The wh-FRAME of the question is a fixed host language scaffold (like the body acting on motor
output); the topic CONTENT is the concept the user surfaced.

## Verify — 24/24 (real handler, numpy-CPU). Artifact `research/findings/raw/_gateB_curiosity_production_verify.json`

<!--derived-->

(The numbers below are rounded reads of the cited verify artifact, whose full-precision values are ground truth.)

- **NOVEL abstain -> honest curiosity follow-up.** "what does the dragon breathe" -> "I don't know about
  that. My curiosity is piqued — I haven't learned about dragon yet: what can you tell me about dragon?"
  (ASK-pool want **129.2 Hz** >= threshold **65.9** -> curious=True; topic="dragon"). The "don't know" abstain
  text is UNTOUCHED — the moat still confabulates nothing; the added text is unambiguously a QUESTION.
- **FAMILIAR recall -> NO follow-up (out of scope).** "what does the dog chase" -> "...cat..." (abstained=False
  -> curiosity is null, no suffix). Curiosity is scoped to abstains, so every recall turn is byte-identical.
- **LESION-LOAD-BEARING.** `BRAIN_CURIOSITY_LESION=1` removes the curiosity drive PATHWAY
  (`curiosity_excit_sensitivity=0` -> the `from_novelty` modulator no longer drives the ASK pool); the SAME
  novel abstain's want COLLAPSES **129.2 -> 5.4 Hz** (< threshold) -> curious=False -> NO follow-up (bare
  "I don't know about that."). So the follow-up is caused by the SPIKING ASK-pool firing, not by the abstain
  flag.
- **FLAG-OFF byte-identical.** `BRAIN_CURIOSITY=0` -> curiosity null; the abstain is the bare
  "I don't know about that." **Additivity proven:** default answer == flag-off answer + the exact follow-up
  suffix (`additive:default==off+suffix` = True).
- **SINGLE-FACT path (`rich=False`)** also appends the follow-up on abstain and leaves recalls untouched.
- **NO-REGRESSION with ALL organs default-ON.** recall, abstain, and the affect / surprise (D2) /
  comprehension (D4) / metacog (E1) / world-model (E2) trace keys are all present and unchanged on the wired
  turn (the curiosity block is additive + scoped to abstains).

## Honest residuals (declared — the mission's named next rungs, not faked)

- **ONLY the crave-DRIVE is wired.** The learning-progress SELECTOR (which of several concepts to ask) is a
  CPU-proxy host formula with seed-fragile on-bridge memory (1/6), and the noisy-TV VETO is a host ELP TD
  tracker (survives the critic lesion) — NEITHER is wired. A single-topic chat follow-up needs neither
  (there is one gap, the one the user surfaced). Those are the named next rungs.
- **NOVELTY = the ABSTAIN** (a BINARY epistemic gap: the brain holds the concept or it does not), a declared
  host boundary; a graded Bogacz-Brown familiarity-gate novelty is the next rung. Curiosity is scoped to
  ABSTAINS (the clearest novelty); a low-confidence RECALL is handled by the metacog hedge (E1), so curiosity
  on a low-confidence recall is a named next rung (not double-fired here).
- **The wh-FRAME is a host language scaffold** (the fixed "what can you tell me about `<topic>`?"); only the
  topic CONTENT is the concept the user surfaced (a host content-word extractor, like the surprise organ's
  assertion extractor — it picks WHICH word to frame, never WHETHER to ask).
- **CO-RESIDENT** on its own curiosity/ASK bridge ALONGSIDE the recall composer — rides on the one-brain
  merge (burn-down #1), exactly as the affect/surprise/comprehension/metacog/world-model organs do.

A FUNCTIONAL curiosity CORRELATE (an ASK-pool drive that tracks the epistemic gap), NOT a claim of subjective
wanting.

## Escape / lesion knobs

```
BRAIN_CURIOSITY=0          # disable -> byte-identical oracle (no crave read, no follow-up)
BRAIN_CURIOSITY_LESION=1   # remove the curiosity drive pathway -> the ASK-pool want collapses -> no follow-up
```
