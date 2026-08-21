---
type: finding
status: live
date: 2026-08-21
mechanism: da-gated-curiosity-crave-threshold-wired-into-chat
lane: integration
integration_faculty: da-gated-curiosity-threshold
seeds: [42]
seed-waiver: A WIRE-IN verify of a deterministic config coupling (does the live self-produced DA reach the curiosity
  crave decision and flip WHETHER the honest follow-up QUESTION is appended; is OFF a no-op; does the lesion sever it) —
  a plumbing/attribution proof, not a stochastic effect size. The spiking ASK-pool crave-DRIVE it reads is the
  6-seed / 6/6-SAFE DR-1 de-risk it reuses (_curiosity_seek_learn_onbridge_derisk, a high gap-vs-spiking-want
  correlation); the DA levels are the real spiking SNc read (#76/#79), not a host constant. The per-turn ASK-pool codes
  at seed 42 are representative (want and threshold are read live off the artifact).
verdict: GO
runner: research/runners/_curiosity_da_threshold_verify.py
instrument: through the REAL production webapp/server.py::brain_chat handler (numpy-CPU, rf recall), ask a NOVEL-topic
  question (the moat abstains -> the curiosity block runs) under three regimes and read the follow-up decision + the
  curiosity_da trace off the response. (A) BRAIN_CURIOSITY_DA unset -> no curiosity_da key + the organ's calibrated
  threshold decides unchanged (novel abstain fires the follow-up; a familiar recall does not). (B) enabled, the SAME
  novel message (novelty held fixed) under a HIGH-DA turn (BRAIN_DA_DRIVES_INDUCE=1300) vs a LOW-DA turn (=100): the
  high-DA turn crosses the lowered crave-threshold and APPENDS the follow-up where the low-DA turn does NOT. (C)
  BRAIN_CURIOSITY_DA_LESION=1 pins the crave-gain to 1.0 regardless of DA -> the high-vs-low decision is identical.
artifact: research/findings/raw/_curiosity_da_threshold/verify.json
external: NO-EXTERNAL-NEEDED — couples the in-repo DR-1 spiking curiosity crave-drive (6-seed GO) with the #76/#79
  spiking DA-mode read; the biology (DA/arousal raises exploratory drive) is the Aston-Jones-Cohen adaptive-gain /
  tonic-DA vigor account already grounded for the sibling couplings.
---

# DA/engagement-gated curiosity crave-threshold wired into the live chat — the brain is MORE curious when it is engaged (WIRE-IN GO, default-OFF)

**Verdict: GO (wire-in, default-OFF — NOT flipped on).** The brain's OWN self-produced tonic dopamine (its engagement /
arousal state) now modulates the CURIOSITY crave decision: when ENGAGED (DA above tonic) it LOWERS the effective
crave-threshold so the spiking ASK pool crosses it more readily — the brain asks an honest follow-up on a novel topic it
would otherwise let pass; when disengaged (DA below tonic) it raises the threshold. This is WAVE-0 Gap-4 coupling (b) —
a coupling of two EXISTING spiking faculties (it re-invents neither): the DR-1 spiking curiosity ASK-pool crave-drive
(`curiosity_production_organ`, corr(gap,want)=+0.996 <!--derived-->) and the #76/#79 spiking SNc->DA read (`da_mode_drives_chat`). The
same self-produced DA that colors HOW forthcoming the reply is (the #79 read-side) now also sets HOW READILY the brain
craves to learn more. Biologically grounded: DA / arousal raises exploratory drive (Aston-Jones & Cohen LC-NE
adaptive-gain; the tonic-DA vigor/incentive account).

## What was built

- `webapp/da_curiosity_drives_chat.py` — the production glue (mirrors the sibling `da_encoding_drives_chat.py`).
  `da_curiosity_enabled()` (flag `BRAIN_CURIOSITY_DA`, default-OFF), `da_curiosity_lesioned()`
  (`BRAIN_CURIOSITY_DA_LESION`), `da_level_of(chat)` (reads the live self-produced DA off `chat._last_da_drives`), and
  `crave_decision(chat, want_hz, base_threshold)`. The organ decides `want >= threshold`; this scales the ASK-pool WANT
  by a small DA crave-gain — engagement amplifies the crave — then compares against the organ's OWN calibrated
  threshold: `da_crave_gain = clip(0.2, 3.0, 1 + 1.5*(DA - 0.5))` (tonic 0.5 -> gain 1.0). The equivalent EFFECTIVE
  crave-threshold `threshold / gain` is reported in the trace (DA-engaged -> lower; disengaged -> higher). NO `sim/`
  edit (a host decision over the live spiking reads; `git diff sim/` empty).
- `webapp/server.py::brain_chat` (`_curiosity_followup`) — after the organ's spiking judge, when
  `BRAIN_CURIOSITY_DA` is enabled the crave decision is recomputed with the live DA and an additive `curiosity_da`
  trace is attached. Default-OFF: the flag unset -> the block is skipped, the organ's calibrated threshold decides
  `curious` unchanged, and NO `curiosity_da` key is attached -> byte-identical to HEAD.

## The moat is preserved by construction

The coupling runs ONLY inside the curiosity block, which itself runs ONLY on an ABSTAIN (the moat already refused —
there is no answer to corrupt). It changes ONLY WHETHER the honest follow-up QUESTION is appended — it never
manufactures a fact, flips an abstain into an assert, or enters the certainty band. The content fields (`abstained`,
`recalled_svo`, `verified`) are byte-identical with the coupling on or off.

## The wire-in proof (GO = A and B and C; `_curiosity_da_threshold_verify.py`, real handler, numpy-CPU rf)

_(values below are rounded from the cited `research/findings/raw/_curiosity_da_threshold/verify.json`; exact values there.)_

- **(A) OFF byte-identical.** `BRAIN_CURIOSITY_DA` unset: on the novel-abstain turn no `curiosity_da` key and the
  organ's calibrated threshold still fires the follow-up; on a familiar recall (taught then queried) it is not an
  abstain -> no follow-up, no key. The only thing this change adds is skipped when off.
- **(B) ON, LOAD-BEARING.** <!--derived--> The SAME message ("what do you know about wombats"), its novelty HELD FIXED (ASK-pool
  want = 126.9 Hz, base threshold = 65.9 Hz on both arms), under a HIGH-DA turn (`INDUCE=1300` -> DA 1.239, gain 2.108, <!--derived-->
  effective threshold 31.3 Hz) **FIRES the follow-up**, where the LOW-DA turn (`INDUCE=100` -> DA 0.046, gain 0.319, <!--derived-->
  effective threshold 206.4 Hz) does **NOT**. The engagement state alone flips the crave decision. (Values rounded from
  the cited artifact.)
- **(C) LESION severs.** `BRAIN_CURIOSITY_DA_LESION=1` pins the crave-gain to 1.0 regardless of DA -> the high-vs-low
  follow-up decision is IDENTICAL (both fire, the organ's own decision) -> the DA-dependence vanishes, attributing the
  (B) difference to the live DA read. (Distinct from `BRAIN_CURIOSITY_LESION`, which collapses the WANT, and
  `BRAIN_DA_DRIVES_LESION`, which collapses the LEVEL.)

## Honest scope — WIRED, default-OFF, NOT flipped on

- This is a WIRED coupling behind `BRAIN_CURIOSITY_DA` (unset -> off -> byte-identical). A default-ON flip would need a
  no-regression soak on the production default (the follow-up cadence under real engagement should help, not nag).
- The NOVELTY the organ reads is the ABSTAIN (a declared host boundary, per the curiosity organ); the
  ENGAGEMENT->SNc-afferent scalar is the same host sensory/comprehension boundary the #79 DA-mode read names as its
  residual. The DA LEVEL itself (SNc spikes off the bus, lesion-proven) and the spiking ASK-pool WANT are the neural
  parts this coupling rides. The crave-gain constants (K_DA=1.5, g_min/g_max) are host-tuned.
- The curiosity organ is co-resident on its own ASK bridge (rides the one-brain merge, burn-down #1), alongside the
  recall composer and the #76 BG substrate — not yet merged onto the single recall bridge.
- FUNCTIONAL engagement->curiosity correlate, NOT a phenomenal claim of subjective wanting.
