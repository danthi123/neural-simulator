---
type: finding
status: contributing
date: 2026-08-12
mechanism: predictive-coding-mismatch
---

<!-- Scope: GO on the spiking mismatch MECHANISM (6/6 at the robust operating point,
     research/findings/raw/_spiking_expectation_rpe_6seed.json, verdict GO). The low-prior
     artifact (_spiking_expectation_rpe_6seed_lowprior.json, verdict UNDEFINED) is the PRECISION
     SENSITIVITY point, cited to CHARACTERISE the boundary — the GO does NOT rest on it. -->

# A genuine SPIKING expectation-violation (surprise) signal, readable at conversation time — 6/6 GO on the mismatch mechanism (lesion-decisive), boundary on the fully-LEARNED mapping

## Headline

A minimal spiking **predictive-coding mismatch unit** produces a **genuine spiking surprise
signal** that separates an assertion which VIOLATES the brain's stored expectation (a
contradiction or a novel fact) from one that CONFIRMS it — **6/6 seeds**, the signal a
`cp_firing_states` firing-rate READ (never a host subtraction), readable WITHIN the assertion
window (conversation time), and **lesion-decisive** (removing the spiking prediction collapses
the separation and raises the confirm-surprise from ~1 Hz to ~9 Hz). This is the
expectation/"understanding-of-consequences" faculty as a spiking signal, not a host compare.
Runner: `research/runners/_spiking_expectation_rpe_derisk.py` (CPU/numpy, ~600 neurons, NO
`sim/` edit).

## The honest boundary this sits at (is there already a spiking RPE? — yes, but a DIFFERENT one)

The project ALREADY has a genuine spiking reward-prediction-error: the limbic core / SNc fires
`delta = r - V` entirely in spikes (`2026-06-18-limbic-core-rpe-battery-GO`, 6/6, lesion-confirmed),
and the multicue learner replaced its host `err` with SNc firing (`2026-06-19-multicue-learning-firm-and-neural-reward`).
**But every existing spiking RPE is over a SCALAR REWARD MAGNITUDE (the Schultz delta).** NONE
reads the brain's stored (agent,action)->patient association, recalls the EXPECTED patient, and
fires on a SEMANTIC CONTENT contradiction ("the dog eats grass" vs the stored "(dog,eats)->meat").
That comparison, as the conversational pipeline stands, would be a host
`recalled_patient == asserted_patient` string compare in Python — a **shortcut** under the
brain-based-only bar. This finding de-risks the genuinely SPIKING replacement.

## Mechanism — a predictive-coding error unit in the patient-concept space

Predictive coding (Rao & Ballard 1999; Bastos et al. 2012): an error unit fires the part of the
feed-forward input NOT explained by the top-down prediction — `error = [actual - prediction]_+`,
the prediction delivered as SUBTRACTIVE inhibition by an interneuron. The direct analogue of the
SNc's `delta = r - V`, but over CONTENT:

```
 cue (agent,action) --Hebbian, topographic--> patient_expected (FS/PV interneuron; the recalled
                                               expectation)  --GABA_A perisomatic (the prediction)-->
 patient_asserted --EXC (topographic c->block c)------------------------------------------------> surprise (RS)
   (the asserted patient, as sensory drive = the legit teacher boundary)      its FIRING RATE = surprise
```

- CONFIRM (assert == expected): surprise block c gets excitation AND matching prediction
  inhibition -> cancel -> surprise ~0.
- CONTRADICT (assert = j != expected i): block j is excited but NOT inhibited (the prediction
  inhibits block i) -> surprise block j FIRES -> high.
- NOVEL (assert an out-of-repertoire patient): un-inhibited -> FIRES.

The surprise pool's TOTAL windowed rate is the conversation-time readout ("am I surprised?").
The prediction PRECEDES the deviant (cue-only pre-phase establishes the expectation, then the
assertion arrives) — the mismatch-negativity protocol.

## Results — GO at the robust operating point (`cue->expected` gain 0.8), 6 seeds, CPU

`--seeds 42,43,44,100,101,102 --cue-to-expected-weight 0.8` (raw: `research/findings/raw/_spiking_expectation_rpe_6seed.json`):

| seed | recall Hz | confirm Hz | contradict Hz (x) | novel Hz (x) | GO |
|---|---|---|---|---|---|
| 42  | 14.1 | 0.40 | 9.05 (22.8x) | 9.94 (25.0x) | Y |
| 43  | 15.1 | 0.30 | 9.19 (30.9x) | 8.63 (29.0x) | Y |
| 44  |  8.5 | 2.54 | 8.89 ( 3.5x) | 8.61 ( 3.4x) | Y |
| 100 | 11.9 | 0.58 | 9.80 (17.0x) | 8.27 (14.4x) | Y |
| 101 | 10.8 | 0.63 | 7.94 (12.5x) | 9.33 (14.7x) | Y |
| 102 |  9.7 | 0.75 | 7.54 (10.0x) | 8.79 (11.7x) | Y |

**INTACT GO 6/6** (gate: contradict AND novel each >= 3x confirm, AND contradict >= 5 Hz).

**LESION (decisive, 3/3):** zero the `patient_expected->surprise` edges -> no prediction ->
contradict/confirm ratio -> 1.0 AND the confirm-surprise RISES `1.08 -> 9.04 Hz`. The spiking
prediction is load-bearing — the surprise on a CONFIRMED assertion is exactly the part the
prediction cancelled, not a fixed input artifact.

**BRAIN-BASED:** `current_reward_signal == 0` (asserted in the runner); the signal is read only
from `cp_firing_states[surprise]`; no Python subtraction of the asserted vs expected codes exists.

## The wall / companion process (quantified) — PRECISION, and the fully-LEARNED mapping

At a LOW prediction gain (`--cue-to-expected-weight 0.4`, raw `research/findings/raw/_spiking_expectation_rpe_6seed_lowprior.json`) intact
GO drops to **3/6**: the seeds whose recall is weak (5-7 Hz, from per-neuron threshold
heterogeneity) do not fully cancel confirm (confirm 2.7-4.6 Hz) -> ratio < 3x. Lesion stays
decisive (confirm `2.43 -> 9.04 Hz`). So the separation robustness scales with the **gain match
between the recalled-prediction inhibition and the asserted excitation** — the PRECISION /
divisive-normalization the animal regulates with inhibitory gain control (PV/SST) + neuromodulation
(NE/ACh), which we proxied with a fixed weight. At gain 0.8 the prediction is strong + uniform
enough for 6/6; the honest next mechanism is a HOMEOSTATIC intrinsic-plasticity precision on the
prediction pool so a low-prior (genuinely learning-sensitive) regime also reaches 6/6.

**Scope of "learned" (the second boundary).** The `cue->patient_expected` **mapping is
TOPOGRAPHIC** (cue block i -> prediction block i); the association STRENGTH is Hebbian-learned
(rate-window Hebbian + Miller-MacKay subtractive normalization — the built-in competition that
prevented Hebbian runaway). Because a topographic prior alone predicts, the UNTRAINED control does
NOT collapse at gain 0.8 (it does partially at gain 0.4). A fully-LEARNED all-to-all mapping —
where untrained/permute would be decisive — needs the CA3 pattern-separation / competition
companion process (`2026-06-05-D-cue-recall-RESOLVED`, already GO). That integration (the sparse
distributed recall sourcing the prediction) is the characterized next rung, not a wall.

## Wireable into the live turn

Yes. Given the conversational pipeline's stored (agent,action)->patient association and an
incoming asserted patient (both already spiking representations), this unit adds a `surprise`
error pool whose windowed rate is a live "expectation-violation" read — usable to (a) NOTICE
("my mismatch monitor reads this as surprising"), an honest functional self-report, and (b) gate
learning by surprise (route `surprise` firing to a phasic neuromodulator / the surprise-LR-boost,
cf. `2026-04-26-surprise-lr-boost`). It moves from a burn-down item (D2/E) to a wireable spiking
faculty: the mismatch mechanism is GO; the fully-emergent learned mapping + homeostatic precision
are the named follow-ons.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._spiking_expectation_rpe_derisk \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_spiking_expectation_rpe_6seed.json
# the precision boundary (low prior, 3/6):
SIM_BACKEND=numpy python -m research.runners._spiking_expectation_rpe_derisk \
    --seeds 42,43,44,100,101,102 --cue-to-expected-weight 0.4 \
    --out research/findings/raw/_spiking_expectation_rpe_6seed_lowprior.json
```

## Substrate notes earned building this (verified 2026-08-12)

- **FS interneuron + GABA_B routing produced a WRONG-SIGN (net excitatory) effect** on this
  substrate; MSN-D1 + GABA_B (the limbic template) and FS + GABA_A both inhibit correctly. The
  prediction pool uses **FS + GABA_A** (low rheobase -> the learned recall fires it; fast
  perisomatic subtractive inhibition).
- **Deterministic regime** (`enable_ou_process=False`, `enable_conductance_noise=False`) is
  required for a controllable operating point — OU background (100 pA) alone fired the pools at
  ~128 Hz and compressed the dynamic range.
- A **hard state reset between trials** (a 20-step settle cannot quiesce a 500 Hz FS pool) removes
  a recency contamination that otherwise made multi-fact recall non-selective.

## Provenance
- Builds on: `2026-06-18-limbic-core-rpe-battery-GO.md` (spiking SNc delta=r-V, the reward-magnitude
  RPE this is NOT), `2026-06-19-multicue-learning-firm-and-neural-reward.md` (spiking RPE reward),
  `2026-06-05-D-cue-recall-RESOLVED-sparse-heteroassoc.md` (the learned recall, the next rung),
  `2026-04-26-surprise-lr-boost.md` (surprise-gated plasticity, the wiring target).
- Reused engine mechanisms: brain-region framework, per-pathway receptors, rate-window Hebbian +
  `hebbian_mean_subtract` (Miller-MacKay normalization). NO `sim/` edit.
- Predictive coding: Rao & Ballard 1999 Nat Neurosci; Bastos et al. 2012 Neuron (canonical
  microcircuit, deep-layer predictions inhibit superficial error units).
