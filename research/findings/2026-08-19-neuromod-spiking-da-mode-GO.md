---
type: finding
status: live
date: 2026-08-19
mechanism: spiking-da-nucleus-self-drives-the-mode
lane: neuromodulation
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO
board_task: 76
instrument: on the fixed-anatomy nav basal-ganglia spiking substrate, make the dopamine LEVEL come from the substrate's OWN spiking DA nucleus (the snc population, IZH2007_DOPAMINE) instead of a host-set concentration. ONE modulator on the subsystem bus (dopamine_mode) reads the SNc mean firing rate each step via the from_region_firing_signed production rule -> concentration, and drives excitability_drive on D1(+)/D2(-) -> the #64 Go/NoGo reconfiguration. The SNc nucleus is itself driven by a reward/context afferent (appetitive = strong depolarising current to snc -> SNc bursts -> DA up -> Go; aversive = weak/zero -> SNc sub-tonic -> DA down -> NoGo). Measure the FUNCTIONAL connectivity F in each context with the board-#63 perturb-and-measure probe. GO gate (all, 6 seeds): SELF-DRIVEN SNc fires more + DA level higher in appetitive than aversive (gap>0 every seed); SAME WIRING max|W_app-W_ave|==0; RECONFIG spearman(F_app,F_ave)<0.9 max over seeds AND excluding the snc row/col; DOUBLE DISSOCIATION >=1 edge opened only in appetitive/Go AND >=1 only in aversive/NoGo, every seed; DA-NUCLEUS load-bearing: silence the SNc nucleus -> F_app==F_ave byte-for-byte (max|dF|==0); determinism byte-identical F at fixed cfg.seed. No set_concentration for dopamine_mode anywhere in the loop.
artifact: research/findings/raw/neuromod_spiking_da/nsd_6seed.json
---

# The brain's OWN spiking dopamine nucleus decides its mode from reward/context — self-driven (6-seed GO)

**Verdict: GO (6/6 seeds).** Board #64 showed a dopaminergic modulator RECONFIGURES the effective circuit
on fixed wiring, but its DA LEVEL was HOST-SET (a manual concentration = the state). This closes that limit.
The dopamine level is now produced by the substrate's OWN spiking DA nucleus — the `snc` population
(`IZH2007_DOPAMINE`) — whose firing rate the neuromodulator bus reads each step; the nucleus is itself
driven by a reward/context afferent. The whole loop runs on spikes and synapses with NO host DA knob:

> reward/context afferent → **spiking SNc nucleus fires** → DA concentration (bus) → D1(+)/D2(−)
> excitability → DIRECT/INDIRECT effective-circuit reconfiguration.

Appetitive context makes the SNc nucleus burst → it self-produces a HIGH dopamine level → the DIRECT/Go
pathway carries signal. Aversive context makes the SNc nucleus go sub-tonic → it self-produces a LOW
dopamine level → the INDIRECT/NoGo pathway carries signal. Silencing the SNc nucleus makes the
reward/context change unable to switch the mode at all (F byte-identical). All-spiking / all-synaptic;
reuse-by-import of the board-#64 modulator config path and the board-#63 probe; NO `sim/` edit.

## Question + sources

Board #64 (`2026-08-19-neuromod-reconfiguration-GO.md`) established the Bargmann/Marder capability — a
neuromodulator reconfigures the effective circuit on fixed wiring (BioEssays 34:458–465, 2012; Neuron
76:1–11, 2012). Its honest limit was that the DA LEVEL was a scalar the runner set. Board task #76 ("let
the brain's own dopamine decide its mode") asks to remove that host knob: the DA level must come from a
SPIKING dopaminergic nucleus that reads reward/context itself. The dopamine nucleus computes a SIGNED
reward-prediction-error code — burst above tonic on better-than-expected outcomes, dip below tonic on
worse (Schultz, Science 275:1593–1599, 1997; J. Neurophysiol. 80, 1998). <!--derived--> The DA level then
gates the striatal direct/indirect projection systems (Albin–DeLong–Penney; Gerfen & Surmeier, Ann. Rev.
Neurosci. 34:441–466, 2011). <!--derived-->

## Mechanism (all spiking / all synaptic; on the neuromodulator bus; no `sim/` edit)

<!--derived-->
The DA nucleus is the substrate's `snc` region — 10 Izhikevich `IZH2007_DOPAMINE` neurons. In this
substrate `snc` has NO outgoing synapses (its afferents are `gpi→snc` and `str_striosome→snc`), so it is a
genuine READ-OUT nucleus: its firing rate is what the bus transduces into the tonic DA level, with no
synaptic short-circuit from `snc` to the D1/D2 MSNs. ONE modulator on the subsystem's own bus
(`sim.neuromodulators.NeuromodulatorManager`), `dopamine_mode`, carries both halves of the loop:

- **SNc firing → DA level (the new part).** Production rule `from_region_firing_signed` on
  `source_regions=["snc"]` reads the SNc mean firing fraction each step (EMA, `window_ms=50`), thresholds
  at the neutral/tonic SNc rate (`threshold=0.07`), and drives the concentration ABOVE baseline when SNc
  bursts / BELOW baseline when SNc is sub-tonic — the Schultz-1998 signed dopamine code. This rule already
  existed in the bus for the spiking-SNc actor-critic; it is reused unchanged.
- **DA level → reconfiguration (the #64 target).** Targets `excitability_drive` `scope="group:str_D1"`
  (sensitivity +1000 pA/unit, D1R Gs excitatory) and `scope="group:str_D2"` (−1000 pA/unit, D2R Gi
  inhibitory) — the same reconfiguration lever #64 used, now fed by the SNc-produced level.

**The concentration is NEVER set by the runner** — grep confirms no `set_concentration` for `dopamine_mode`
in the loop. It is produced entirely by SNc firing through the bus's own `step()`. The measurement loop is
fully live: each step adds the SNc-produced excitability drive to the input, steps the substrate, then lets
the bus read the resulting SNc firing and update the level.

**Reward/context afferent.** Appetitive = a strong depolarising current to the SNc nucleus (800 pA → SNc
fires more) ; aversive = zero (SNc sub-tonic). This afferent is the environment's reward/context signal
reaching the DA nucleus. What is CLOSED here is that the DA LEVEL is now brain-derived (set by SNc spikes).
Computing the reward/context scalar itself from the brain's own sensory stream is a SEPARATE faculty and
remains an environmental input — see residuals.

## Read-out (board #63 perturb-and-measure, reuse-by-import)

For each context we run the live loop and, for each region TYPE A, perturb A (+400 pA), settle, record the
signed Δ firing-rate of every other TYPE B → a state-specific FUNCTIONAL matrix F over the 12 TYPES. All
plasticity + noise + heterogeneity OFF (isolation: measure propagation on a FIXED graph). `cfg.seed` seeds
the substrate (the seed trap — not `actual_seed_used`). Operating point keeps the striatal tone at 40 pA so
the D1/D2 MSNs sit near threshold and the DA level GATES which pathway a cortical drive recruits — the #64
reconfiguring regime (at tone 250 DA degrades to gain-only; see residuals).

## Result — the SNc nucleus self-drives the level, and the level reconfigures the circuit (6/6)

<!--derived-->
Per-seed spearman(F_app,F_ave) = 0.782, 0.762, 0.812, 0.753, 0.811, 0.762. Pooled:

| metric | pooled | reading |
|---|---|---|
| **SNc firing** appetitive / aversive | **0.141 / 0.006** (gap>0 every seed) | the nucleus fires far more under appetitive context |
| **DA level (conc) SELF-PRODUCED** appetitive / aversive | **0.887 / 0.083** (gap min 0.795) | SNc firing sets a HIGH level (Go) vs LOW (NoGo) — no host knob |
| spearman(F_app, F_ave), union-nonzero edges | **mean 0.780, max 0.812** | rank correlation < 1 on every seed ⇒ reconfiguration, not gain |
| edges OPENED only in appetitive/Go (min / mean) | 6 / 7.17 | direct-pathway edges present only in Go |
| edges OPENED only in aversive/NoGo (min / mean) | 6 / 7.17 | indirect / hyperdirect edges present only in NoGo |
| spearman with the **SNc nucleus SILENCED** | **1.000** | without the nucleus the two contexts are the SAME network |
| max\|F_app − F_ave\| with the nucleus silenced | **0.0** | byte-for-byte identical ⇒ 100% attributable to the DA nucleus |
| max\|W_app − W_ave\| (anatomy) | **0.0** | wiring is IDENTICAL across contexts — a purely functional switch |
| determinism: byte-identical F on re-run @ fixed seed | True | — |

**The DOUBLE DISSOCIATION (edges common to ALL 6 seeds):**

- **Opened only in appetitive/Go (DIRECT pathway):** `cortex→thal`, `gpi→thal`, `str_D1→thal`,
  `str_D1→str_PV_FSI`. The direct route disinhibits the thalamus (str_D1 inhibits gpi → gpi releases thal),
  so a cortical drive reaches thal/motor — the same signature as #64.
- **Opened only in aversive/NoGo:** `snc→gpe`, `snc→gpe_arky`, `snc→str_D1`, `snc→str_D2`,
  `thal→str_striosome`. The `snc→X` edges are the LIVE self-drive signature: in the low-DA aversive state,
  perturbing the nucleus itself pushes DA up and reconfigures the circuit — an edge that CANNOT exist in
  #64's host-set static design, and a direct demonstration that the loop is live.

## Robustness — the reconfiguration is genuinely in the D1/D2 circuit, not the nucleus self-perturbation

<!--derived-->
Removing the `snc` row AND column entirely (so no `snc→X` or `X→snc` edge can contribute), the
reconfiguration persists on every seed: spearman(F_app,F_ave) EXCL snc = 0.860, 0.797, 0.846, 0.816, 0.864,
0.817 (mean 0.833, max 0.864 — all < 0.9), with the canonical direct/indirect double dissociation intact.
The **Go-opened** non-snc edges common to all 6 seeds are `cortex→thal`, `str_D1→thal`, `gpi→thal`,
`str_D1→str_PV_FSI` (plus `cortex→motor`, `str_D1→motor` on most seeds) — the clean direct-pathway
signature. The **NoGo-opened** non-snc edges include `str_D2→gpe`, `str_D2→str_PV_FSI`, `gpe→stn`,
`cortex→stn` (per-seed) and `thal→str_striosome` (all seeds) — the indirect / stopping signature. The
self-driven DA level reconfigures the striatal direct/indirect systems themselves, exactly as #64; the
`snc→X` feedback edges are additional, not the source of the effect.

## Why this is self-driven, not a host knob (the anti-cheats ARE the result)

<!--derived-->
1. **No host concentration knob (grep-verified).** The runner contains no `set_concentration` for
   `dopamine_mode`. The level is produced by `NeuromodulatorManager.step()` reading SNc firing every step
   via `from_region_firing_signed`. The only host inputs are the baseline tones and the reward/context
   afferent current to `snc` (the environmental signal reaching the nucleus).
2. **DA-nucleus load-bearing (lesion dissociation).** Silence the SNc nucleus (clamp its input to a fixed
   sub-firing current, context-independent) and the reward/context change can no longer reach the DA level:
   the self-produced level collapses to 0.046 for BOTH contexts and F_app == F_ave byte-for-byte
   (spearman 1.000, max|dF|=0). The mode switch is 100% attributable to the nucleus (`attributable_to`
   treatment 0.780 vs control 1.000 across seeds).
3. **Same wiring.** max|W_app − W_ave| = 0 exactly — pathways are never touched. A functional
   reconfiguration on FIXED structure, driven by a brain-derived level.

## Honest residuals / scope

<!--derived-->
- **The reward/context is still an environmental scalar afferent to the SNc nucleus.** This task closes the
  DA-LEVEL host knob (the level is now set by SNc spikes, not `set_concentration`); it does NOT close the
  reward-computation faculty — computing the appetitive/aversive signal from the brain's own sensory stream
  is a separate brain-based-only target. The afferent-to-a-DA-nucleus form is faithful (VTA/SNc receive
  reward-carrying afferents), but the scalar's ORIGIN is host, and is documented as such.
- **The aversive/NoGo opened-edge set common to all seeds is dominated by `snc→X` self-drive edges**, not
  the classic `str_D2→gpe` indirect edge (which appears per-seed and in the excl-snc analysis, common only
  to a subset). This is an honest, expected consequence of the live loop: perturbing the nucleus in the
  low-DA state shifts the level. The genuine D1/D2 direct/indirect reconfiguration is confirmed by the
  excl-snc robustness check above.
- **Operating-point dependence (inherited from #64, and bounded).** The reconfiguring regime requires the
  MSNs near threshold (str tone 40) and the SNc→DA parameters tuned so the SNc-produced level spans the #64
  reconfiguring band. At str tone 250 the switch degrades to gain-only (#64). The SNc→DA gain
  (`sensitivity`/`threshold`/`tau`/`window`) sets how a firing-rate deviation maps to a concentration
  deviation; it is tuned to land appetitive high / aversive low, then the anti-cheats (lesion byte-identity,
  W-unchanged, attribution) guard the claim regardless of the tuning.
- **De-risk, not production integration.** Additive / default-off (pure runner reuse, no `sim/` edit). The
  next rung is wiring `dopamine_mode` + the SNc read into the live default brain so its own DA nucleus sets
  the mode moment-to-moment during a conversation (rest/focus/appetitive/aversive), and driving the SNc
  afferent from a brain-computed reward rather than an environmental scalar.

## Reproduce

```bash
SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -m research.runners._neuromod_spiking_da_mode_derisk \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/neuromod_spiking_da/nsd_6seed.json
# exploration (one seed, tune the SNc->DA operating point):
#   ... --explore --seeds 42 --app-snc 800 --ave-snc 0 --da-threshold 0.07 --da-sens 70
```

Runner: `research/runners/_neuromod_spiking_da_mode_derisk.py` (reuse-by-import of
`_neuromod_reconfiguration_derisk` for the D1/D2 target + edge analysis and `_perturb_and_measure_derisk`
for the probe). Raw + provenance sidecar:
`research/findings/raw/neuromod_spiking_da/nsd_6seed.json(.prov.json)`.
