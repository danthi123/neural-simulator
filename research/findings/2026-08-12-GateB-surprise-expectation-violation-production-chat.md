---
status: live
lane: gap#1
date: 2026-08-12
type: finding
integration_faculty: surprise-monitor
seed-waiver: production-INTEGRATION verify of an already-6/6-GO faculty (the D2 de-risk `2026-08-12-spiking-expectation-violation-surprise-conversational-6seed-GO-mechanism.md`). This doc verifies the deterministic WIRING glue on the real /api/brain-chat handler (single process, one seed=42 organ at the robust operating point), not a new statistical GO; the 6-seed evidence is the cited de-risk. Confirm/contradict/lesion arms are decisive on the single wired seed.
---

# Gate-B (D2): EXPECTATION-VIOLATION / SURPRISE wired into the production /api/brain-chat turn — the brain honestly NOTICES ("that surprises me — I'd learned <stored>") when an assertion contradicts what it holds, default-on, moat-safe, lesion-load-bearing (WIRED)

**Date:** 2026-08-12
**Status:** GO / WIRED (production-integration). Single-process synchronous in-process verify on the real
`/api/brain-chat` handler (`SIM_BACKEND=numpy`, `BRAIN_COMPOSER_KIND=rf`, stub renderer, rich=False). All 8 verify checks pass.

## What changed

When the user ASSERTS a fact `(agent, action, patient)` for which the brain ALREADY HOLDS a stored
`(agent,action)→patient` association, a genuinely-SPIKING predictive-coding MISMATCH unit runs and, on a firing
surprise, the brain PREPENDS an honest functional NOTICE — *"That surprises me — my mismatch monitor fired: I'd
learned that <agent> <action> <stored>."* — the owner's "understanding of consequences / expectation". The signal
is a windowed `cp_firing_states[surprise]` rate, NOT a host `recalled==asserted` string compare.

Additive, no `sim/` edit. Two pieces:

- **`research/runners/surprise_production_organ.py`** (new) — the production-integration glue. It REUSES the
  adversarially-verified D2 faculty (`_spiking_expectation_rpe_derisk.py`, 6/6 GO at the robust operating point,
  lesion-decisive): cue `(agent,action)` --Hebbian topographic--> `patient_expected` (an FS/PV-like interneuron
  delivering GABA_A SUBTRACTIVE inhibition = the recalled prediction); `patient_asserted` --excitation--> `surprise`
  (RS pyramidal). CONFIRM cancels (~0 Hz), CONTRADICT/NOVEL fires. Built once + trained (learning then FROZEN) at
  the robust `cue_to_expected_weight=0.8` operating point; patient concepts map to circuit blocks on demand; the
  confirm-vs-contradict threshold is calibrated at build. The EXPECTED patient is RECALLED by the brain's own
  spiking recall (`what_does`), the ASSERTED patient is the sensory drive — the mismatch is a firing read.
- **`webapp/server.py`** `brain_chat` — per turn, AFTER the D4 comprehension gate and BEFORE the rich/single split:
  extract the assertion; if the brain holds a stored `(agent,action)→patient` (`what_does` truthy), run the organ;
  on `surprised` PREPEND the honest notice to the turn's answer. `BRAIN_SURPRISE=0` fully disables (byte-identical
  oracle); `BRAIN_SURPRISE_LESION=1` zeroes the prediction→surprise edges for the load-bearing test.

## Verify (SYNCHRONOUS, in-process, real `/api/brain-chat` handler, numpy-CPU, rich=False, 8/8 checks PASS, 42 s)

Artifact: `research/findings/raw/_gateB_surprise_production_verify.json` (all numbers below). De-risk numbers (6/6 GO,
confirm 0.3–2.5 Hz vs violate 7.5–9.9 Hz) are quoted from
`research/findings/2026-08-12-spiking-expectation-violation-surprise-conversational-6seed-GO-mechanism.md`. <!--derived-->

Threshold calibrated at build ≈ **2.73 Hz** (confirm ≈ 0.2 Hz vs contradict ≈ 5.44 Hz).

| turn | surprise read | behaviour |
|---|---|---|
| teach `wolf hunt deer` (no prior expectation) | surprise:null (skipped) | `The wolf hunts deer.` |
| CONFIRM `wolf hunt deer` (stored==asserted) | **0.00 Hz** < thr → surprised=False | no notice → `The wolf hunts deer.` |
| CONTRADICT `wolf hunt rock` (stored=deer, asserted=rock) | **5.61 Hz** ≥ thr → **surprised=True** | honest notice prepended → *"That surprises me — my mismatch monitor fired: I'd learned that wolf hunt deer. The wolf hunts rock."* |

CONTRADICT (5.61 Hz) ≥ **3×** CONFIRM (0.00 Hz) — a real separation.

**LESION (`BRAIN_SURPRISE_LESION=1`):** on the SAME CONFIRM input `wolf hunt deer` (stored==asserted==deer) that is
**0.00 Hz** intact, zeroing the `patient_expected→surprise` prediction edges removes the subtractive inhibition, so
the confirm now FIRES at **4.92 Hz** (surprised=True → the notice fires even on a confirmed assertion). Restoring the
prediction returns it to **0.00 Hz** — **load-bearing** (the confirm/contradict separation is caused by the spiking
prediction, not a fixed input artifact).

**FLAG-OFF (`BRAIN_SURPRISE=0`):** `surprise` is null and no notice fires — the byte-identical oracle.

## Honest residuals (declared; each rides an existing burn-down row)

- **CO-RESIDENT:** the mismatch unit runs on its OWN circuit bridge, ALONGSIDE the recall composer, not merged onto
  the single recall bridge — the remaining one-brain consolidation step (**burn-down #1**), exactly as the affect organ.
- **PRECISION BOUNDARY** (the de-risk's mapped residual): at LOW prediction gain the GO drops to 3/6 (the
  divisive-normalization / gain-match companion process is proxied by a fixed weight). Wired at the ROBUST operating
  point (`cue_to_expected_weight=0.8`), 6/6-GO with headroom. Fully-learned all-to-all CA3 recall + homeostatic gain
  precision are the named next rungs.
- **INFLECTION:** the `(agent,action)` recall + patient-block mapping key on surface tokens (light tolerance); a
  fully inflection-robust lookup rides on the same lemmatization work as D4.

## Repro

```
SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf   # POST /api/brain-chat (or in-process brain_chat)
#  teach:      "wolf hunt deer"       (brain now expects (wolf,hunt)->deer)
#  confirm:    "wolf hunt deer"       -> ~0 Hz, no notice
#  contradict: "wolf hunt rock"       -> firing surprise -> "That surprises me — I'd learned that wolf hunt deer."
#  lesion:  BRAIN_SURPRISE_LESION=1 (removes the prediction -> confirm fires too)  ;  disable: BRAIN_SURPRISE=0
```
