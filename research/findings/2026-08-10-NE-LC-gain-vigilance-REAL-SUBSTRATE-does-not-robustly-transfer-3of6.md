---
type: finding
status: contributing
date: 2026-08-10
mechanism: neuromodulators
lane: A5
seeds: [42, 43, 44, 100, 101, 102]
instrument: population spike-count d-prime (signal-present vs signal-absent trials) on a real 2-population spiking bridge, swept across NE synaptic_gain levels; controls = byte-identical-when-off (sha256 of per-trial spike counts, gain=1.0 vs subsystem-disabled) + rate-matched additive excitability_drive (must NOT reproduce the d' lift).
---

# NE / LC multiplicative gain on the REAL substrate — does NOT robustly transfer (3/6 GO), an honest negative that maps the substrate limit

The idealized-200-LIF probe (section 2 of
[`2026-08-10-parallel-push-results-...`](2026-08-10-parallel-push-results-activity-silent-WM-GO-NE-gain-positive-CA3-transmission-refuted-3rd.md),
clean monotone d' 2.04 -> 3.14 -> 4.25) is here PROMOTED to a REAL `SimulationBridge`. The gain is delivered by the
sim's own `NeuromodulatorManager.compute_synaptic_gain_multiplier()` scaling `effective_synaptic_strength` at
`sim/bridge.py:8167` — NOT a host multiply on the readout, so this is a genuine real-substrate test. Biology:
Aston-Jones & Cohen 2005 LC-NE adaptive gain (Annu. Rev. Neurosci. 28:403-450). **Verdict: the clean idealized
POSITIVE does NOT robustly transfer — REAL-SUBSTRATE 3/6 GO (NO-GO on the >=5/6 bar).** This negative is the
deliverable: it maps where the substrate's own heterogeneity breaks an idealization.

## Method (real substrate, brain-based gain — NO `sim/` edit, config-only)

<!--derived-->

- 2-population spiking net (Izhikevich RS): `n_in=40` INPUT -> `n_tgt=160` TARGET, dense E->E synapses
  (`inject_explicit_wiring`, `weight=3.0`), plasticity frozen (weights verified drift-free). Input driven by
  `base_drive=320 pA` constant external current + `sig_drive=18 pA` extra on signal-present trials (the weak
  signal). OU background noise restricted to the TARGET pop (`cp_ou_neuron_mask`, `ou_std=60 pA`) = the FIXED
  intrinsic spike-generation floor, summed AFTER the synaptic matvec (`sim/bridge.py:7581`) so `synaptic_gain`
  does NOT scale it.
- NE modulator: `NeuromodulatorConfig(name="NE", targets=[synaptic_gain scope=all sensitivity=1.0])`; gain =
  1 + conc, set per trial via `set_concentration`. Readout = TARGET population spike COUNT per trial; d' between
  120 signal-present and 120 signal-absent trials from the ACTUAL counts. `SIM_BACKEND=cupy`, `cfg.seed` set
  explicitly. All mutable per-neuron state (incl. the ADAPTIVE `cp_neuron_firing_thresholds`) is reset per trial.

## Result — the lift is SEED-FRAGILE (3/6 GO)

<!--derived-->

Per-seed d' across the NE gain sweep g = 1.0 / 1.5 / 2.0 / 2.5 (mean target spike count in parens):

| seed | g=1.0 | g=1.5 | g=2.0 | g=2.5 | monotone | mult>add | byte-ident | verdict |
|------|-------|-------|-------|-------|----------|----------|------------|---------|
| 42  | 0.81 (116) | 1.30 (161) | 1.96 (209) | 2.86 (256) | yes | yes (add d'=0.81) | yes | **GO** |
| 43  | 0.18 (73)  | 0.25 (95)  | 0.54 (118) | 0.98 (142) | yes | yes (add d'=0.06) | yes | **GO** |
| 102 | 0.80 (157) | 1.45 (206) | 1.78 (248) | 2.23 (290) | yes | yes (add d'=0.86) | yes | **GO** |
| 44  | 0.04 (83)  | -0.02 (90) | -0.23 (97) | -0.61 (107)| no  | no                | **NO** (max|diff|=2) | fail |
| 100 | 0.32 (116) | 0.51 (147) | 0.45 (176) | 0.36 (204) | no  | no                | yes | fail |
| 101 | -0.43 (24) | -0.38 (19) | -0.40 (16) | -0.36 (15) | yes (flat/neg) | no     | yes | fail |

Aggregate: **n_go 3/6, n_monotone 4/6, n_mult_beats_add 3/6, n_byte_identical 5/6, GO_ALL=False.**

`attributable_to("NE multiplicative gain d-prime: gain-high vs gain-off", d_hi, d_lo)` per seed: 42 → **+0.716**,
43 → **+0.818**, 102 → **+0.640** (the gain OWNS the lift on the GO seeds); 100 → +0.116, 101 → **-0.171**,
44 → +1.073 but onto a d' that is itself NEGATIVE (a lift toward less-negative, not detection). On the 3 GO seeds
the multiplicative gain also beats the rate-matched additive `excitability_drive` control by 0.92–2.05 d' (42:
2.86 vs 0.81; 43: 0.98 vs 0.06; 102: 2.23 vs 0.86) — where it works, it works for the RIGHT reason
(multiplicative, not a DC offset).

## Why it does not transfer — the companion process the probe replaced with a constant

<!--derived-->

The idealized LIF probe was a HOMOGENEOUS population at a single hand-tuned operating point (fixed background,
fixed sigma, no per-neuron variation). The real substrate runs three things the probe replaced with constants:
per-neuron **Izhikevich heterogeneity**, the **OU** background process, and per-neuron **adaptive thresholds**.
With a SINGLE global operating point (`weight`, `base_drive`, `ou_std`) these push different seeds to very
different baselines: seed 101's target barely fires (rate 15–24 over 120 steps — noise-floor sparse, d' negative),
and its rate FALLS as gain rises; seed 44 sits where more gain drives d' NEGATIVE. Multiplicatively scaling the
synaptic drive only sharpens detection when the neuron is already on the SENSITIVE part of its f-I curve — a
property the real substrate does NOT guarantee across heterogeneity from one global set-point. This is the
recurring wall pattern (`docs/FAILURE_GATE_MATRIX.md` / the "companion process" finding): **the operating point is
implicit in the animal**, held by a homeostatic set-point the idealization omitted.

## Byte-identical-when-off: 5/6 exact; seed 44 breaks by 2 spikes = a harness reset residual, NOT an off-path leak

<!--derived-->

The gain=1.0 no-op path IS bit-identical to the subsystem-disabled bridge — proven directly, including on the one
failing seed. Two diagnostics on seed 44:
- **TEST A** — a FRESH subsystem-ON bridge at gain=1.0 (NO prior trials) vs a FRESH subsystem-OFF bridge, same 40
  trials: **max|count-diff| = 0.0, 0/40 trials differ.** The guarded-off `synaptic_gain` path (`bridge.py:8168`
  sets `effective_connections_matrix = self.cp_connections`, the identical object the off-path uses) does NOT
  leak, and enabling the subsystem draws no extra RNG. Two subsystem-OFF bridges are likewise exact (0.0).
- **TEST B** — the ON bridge AFTER running the full gain sweep + additive control (the runner's actual order) vs a
  fresh OFF bridge: **max|diff| = 2.0 on 1/40 trials** — reproduces the run's seed-44 break.

So the break is **residual reused-bridge STATE CONTAMINATION**, not the gain and not the neuromodulator off-path.
Trials reuse one bridge and reset per-neuron state each trial (incl. the ADAPTIVE `cp_neuron_firing_thresholds`,
`cp_neuron_activity_ema`, `cp_last_spike_time`); that reset made 5/6 seeds EXACT, but at least one more mutable
state array is not restored, and for seed 44's particular trajectory it flips 2 spikes on 1 trial. It is an
INSTRUMENT artifact of the reused-bridge harness — the fresh-bridge no-op is exact — and it does not change any
mean or the verdict. **Honest open caveat:** I did not enumerate the last un-reset array (diminishing returns);
the clean fix is a fresh bridge per trial (slower) or completing the reset set. I therefore do NOT claim
byte-identical holds universally — it holds exactly on 5/6, and the 6th is a characterized harness residual.

## Next lever (a negative launches the next mechanism)

Pair the NE gain with the **homeostatic companion process** the probe omitted: an intrinsic-plasticity / firing-rate
set-point (or a co-modulated threshold) that first places each heterogeneous target neuron on the sensitive part of
its f-I curve, THEN applies the NE multiplicative gain. Hypothesis: NE-gain detection becomes robust (>=5/6)
once the operating point is homeostatically fixed per neuron rather than by one global drive — because in vivo LC-NE
acts on a cortex already held at a set-point by intrinsic/synaptic scaling. Concrete build: add a short
rate-homeostasis warm-up (adjust per-neuron excitability to a target baseline rate at g=1.0) before the sweep, or
co-vary an adaptive-threshold term with the gain, then re-run the 6-seed gate.

## Provenance

- Runner: `research/runners/_ne_lc_gain_vigilance_realbridge_derisk.py` (config-only, no `sim/` edit).
- Artifact: `research/findings/raw/lanes/ne_realbridge_6seed.json` (+ `research/findings/raw/lanes/ne_realbridge_6seed.json.prov.json`).
- Reproduce: `SIM_BACKEND=cupy python -m research.runners._ne_lc_gain_vigilance_realbridge_derisk --seeds 42 43 44 100 101 102 --n-trials 120 --T 120 --out research/findings/raw/lanes/ne_realbridge_6seed.json`
