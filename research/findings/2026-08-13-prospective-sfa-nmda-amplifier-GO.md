---
type: finding
status: go
date: 2026-08-13
lane: prospective
mechanism: pmem-sfa-nmda-coincidence-amplifier
runner: research/runners/_pmem_sfa_nmda_amplifier_derisk.py
artifacts:
  - research/findings/raw/_pmem_sfa_nmda_amplifier.json
  - research/findings/raw/_pmem_sfa_nmda_amplifier.json.prov.json
---

# Prospective memory `fire_on_cue` CLOSES 6/6: a supralinear NMDA/dendritic-plateau COINCIDENCE amplifier is the load-bearing lever (SFA is not); the intention-latch + cue-monitor arc is complete

**Verdict: GO at the pre-registered 6-seed gate (6/6 pass EVERY clause; need 5/6).** Against the two `fire_on_cue`
misses [H] relocated to a coincidence SEPARATION/GAIN deficit
(`2026-08-13-prospective-perpool-homeostat-BOUNDARY.md`), I built the mission's named mechanism — spike-frequency
adaptation (SFA) at a timescale between the coincidence read and the hold, PLUS a supralinear NMDA/dendritic-plateau
coincidence amplifier. `fire_on_cue` rises **4/6 → 6/6**, every silence clause STAYS **6/6**, and every seed's
fire/silent ratio clears the absolute-window requirement (0.20/0.06 = **3.33**) with margin (min **7.42**). CPU,
reuse-by-import of the [H] `HomeostaticProspectiveMemory` + the FROZEN gate, NO `sim/` edit.

**But the ablation OVERTURNS the pre-registered mechanism split: the supralinear NMDA plateau ALONE closes the
whole residual (6/6); SFA is neither necessary nor sufficient.** The brief predicted SFA would fix seed 44 (the
sustained-runaway seed) via timescale-separation and the NMDA amplifier would fix seed 100 (the gain deficit).
Half right: the plateau fixes seed 100 as predicted — and ALSO fixes seed 44, because seed 44's miss was a
bias-SUPPRESSED coincidence, which the coincidence-specific plateau amplifies directly. SFA does not do the seed-44
job. This is the honest, and stronger, result: one mechanism closes it.

Artifact: `research/findings/raw/_pmem_sfa_nmda_amplifier.json` (provenance sidecar beside it).

## The mechanism (label-free; NO `sim/` edit; reuse-by-import)

A POOL-GATED regenerative NMDA plateau on each `rel_X` cue-monitor pool. Each step reads the pool's REAL mean NMDA
conductance `cp_conductance_g_nmda`; when it crosses a pool threshold `theta[X]`, ALL pool neurons get a uniform
supralinear boost `min(plateau_g * (g_pool − theta), plateau_cap)` pA. `theta[X]` is calibrated STAGE-2 (biases
frozen), label-free, to `1.05 ×` the max POOL-MEAN g_nmda produced by either SINGLE input — cue-alone over the
30-step cue read AND held-alone over the full ~300-step hold. Because a single input never lifts the pool-mean over
threshold, no single-input silence condition boosts; only the COINCIDENCE (both feedforward inputs summed) crosses,
so the plateau fires the pool-wide regenerative event and the JOINT response is amplified. This is a faithful
dendritic-NMDA-plateau read on a substrate conductance, biology-bound to
`research/biology/dendritic-plateau-coincidence-burst.md` (Kandel 6e / Larkum et al. 1999: distal input alone
gives "only a very small" somatic response; paired with proximal it triggers a plateau) and to the NMDA receptor
as a molecular coincidence detector (Kandel 6e; Schiller, Major, Koester & Schiller 2000, NMDA spikes in basal
dendrites).

I chose POOL-gating (all-or-none, spreads across the pool) over a per-neuron graded boost after a diagnostic: a
per-neuron threshold gives ~0 boost to the high-rheobase LAGGARD neurons (their own coincidence g_nmda barely
exceeds their own single-input g_nmda), so a graded boost could not lift them and seed 44's pool ceilinged. The
pool-wide regenerative event lifts the laggards — which is also the more faithful NMDA-spike/plateau (a
regenerative branch event, not a graded per-synapse gain).

SFA is present in the ARM-1 build per the brief (a per-neuron K-adaptation current, normalized low-pass of the
pool's own spikes, subtracted; timescale `sfa_tau=100` steps, between the 30-step read and the ~300-step hold, so
the SUSTAINED single input adapts while the fresh cue transient is preserved). The ablation below shows it is not
load-bearing for this residual.

## Results — 6 seeds 42/43/44/100/101/102, N=5

The gate (thresholds + per-seed clause logic) is IMPORTED from the parent runner and the substrate class is
monkey-patched, so every arm is scored by the SAME code.

<!--derived-->
| clause | ARM2: [H] homeostat (control) | ARM1: SFA + NMDA plateau |
|---|---|---|
| persistence | 6/6 | **6/6** |
| no_fire_before | 6/6 | **6/6** |
| no_fire_wrongcue | 6/6 | **6/6** |
| no_intention_silent | 6/6 | **6/6** |
| lesion_holds | 6/6 | **6/6** |
| lesion_forgets | 6/6 | **6/6** |
| separation | 5/6 | **6/6** |
| **fire_on_cue** | **4/6** | **6/6** |
| seeds passing ALL clauses | **4/6** | **6/6** |

<!--derived-->
The ARM-2 control reproduces the [H] 4/6 BOUNDARY EXACTLY (seeds 44 and 100 fail `fire_on_cue`; seed 44 `fireA`
0.157 ratio 3.40, seed 100 `fireB` 0.085 ratio 1.94) — validating the like-for-like instrument. Under ARM-1 every
seed fires on its true cue with a large margin, and every silence read stays sub-ceiling (`max_silent` 0.042–0.050 <!--derived-->
< 0.06). Per-seed `fire_min` / fire-silent ratio (ARM-1): 42 `0.387`/`9.05`, 43 `0.378`/`8.96`, 44 `0.346`/`7.42`,
100 `0.361`/`8.22`, 101 `0.373`/`7.46`, 102 `0.386`/`8.91`. The two [H] targets are specifically rescued: seed 100
`0.085 → 0.361`, seed 44 `0.157 → 0.346`.

## Attribution — the plateau is the load-bearing lever; SFA is not (ablation, N=5, homeostat ON in all)

<!--derived-->
| mechanism | seed 44 `fire_min` (pass) | seed 100 `fire_min` (pass) |
|---|---|---|
| SFA-only (plateau OFF) | 0.137 (FAIL) | 0.081 (FAIL) |
| **Plateau-only (SFA OFF)** | **0.354 (PASS)** | **0.361 (PASS)** |
| BOTH-OFF ([H] homeostat) | 0.157 (FAIL) | 0.085 (FAIL) |
| BOTH-ON (the ARM-1 build) | 0.346 (PASS) | 0.361 (PASS) |

<!--derived-->
SFA-only rescues NEITHER hard seed (seed 44 `0.157 → 0.137`, slightly WORSE; seed 100 `0.085 → 0.081`, unchanged).
Plateau-only rescues BOTH, and a full 6-seed plateau-only run is a clean **6/6** (per-seed ratios 7.67–9.39, all
silence 6/6). Adding SFA to the plateau changes almost nothing (seed 44 `0.354 → 0.346`; seed 100 identical
`0.361`). So the pre-registered "SFA fixes the runaway, NMDA fixes the gain" split is refuted: the coincidence
amplifier owns the closure of both. The `attributable_to` on MEAN fire reads only 43% owned by the ARM-1
manipulation (57% is the shared [H] homeostat, present in both arms) — expected and non-fatal, because 4 seeds
already pass under the homeostat; the amplifier's deliverable is the SEED-SPECIFIC rescue of the last two, which
the control confirms still fail without it.

## Why the plateau fixes seed 44 (not SFA)

Seed 44's [H] miss was NOT an intrinsic gain deficit (its separation ratio 3.40 fit the window) — its held-alone
input runs away over the sustained hold, forcing the homeostat to a conservative bias that SUPPRESSES the
coincidence. Both failure modes — seed 100's weak joint response and seed 44's bias-suppressed joint response —
are "the coincidence is too weak against a fixed threshold". The plateau amplifies exactly the coincidence
(pool-mean g_nmda above the single-input ceiling) and so overcomes BOTH: seed 44 passes under plateau-only even at
its conservative [H] bias (−1116/−1416 pA). The bias-relaxation SFA provides during calibration (seed 44 to
≈ −1006/−1293) turned out not to be the operative lever.

## Honest scope + anti-cheats verified

- **Brain-based / label-free:** the plateau threshold references only the pool's OWN single-input NMDA
  conductance, never which cue is correct; the boost gates on the pool's own `cp_conductance_g_nmda`. All firing
  reads are `cp_firing_states`.
- **The named cheat did NOT happen:** "an amplifier that fires on single inputs". The `void_if` guard checks every
  silence clause stays 6/6; it is clear (`silence_regressed=[]`). The per-pool threshold is pinned ABOVE the worst
  single input, so single-input conditions never boost — verified: no_fire_before / no_fire_wrongcue /
  no_intention_silent / lesion_holds / lesion_forgets all 6/6, `max_silent` 0.042–0.050 < 0.06 on every seed <!--derived-->. The
  diagnostic (label-based, report only) shows single-input pool-mean g_nmda < theta < coincidence g_nmda with a
  positive margin (~24–32) on every seed/pool.
- **Instrument:** the ARM-2 control reproduces the parent [H] 4/6 exactly (same code path), so the ARM-1 deltas
  are attributable to the amplifier, not a substrate change.
- **HOST-SCAFFOLD, FLAGGED (unchanged from [H]/parent):** the cue→action CONTENT binding is installed
  synaptically; the plateau boost (and the SFA current) are host-injected current-injection PROXIES for the
  dendritic NMDA-plateau conductance and the K-adaptation conductance — the same class of flagged proxy as the
  parent's tonic-inhibition bias. The MECHANISM (a supralinear coincidence gate reading a real NMDA conductance,
  label-free) is brain-based.

## What this closes

The prospective-memory arc (a spiking PFC intention LATCH holding a deferred intention across N intervening
distractor turns + a BA10-style spiking CUE-MONITOR that releases it only on the right cue) is now a full, faithful
6/6: latch, persistence, cue-specific release, wrong-cue / no-intention / lesion silence, AND the release-amplitude
`fire_on_cue` that the parent + [H] left open. The next biologization (unchanged, named by the parent) is LEARNING
the cue→action content binding via one-shot Hebbian potentiation at intention-formation (Gollwitzer
implementation-intentions), replacing the flagged synaptic install.

BANKED METHOD: SFA (timescale-separation adaptation) is NOT the lever for a bias-suppressed / weak coincidence —
the operative mechanism is a coincidence-SPECIFIC supralinear amplifier (a pool-gated NMDA plateau) that overcomes
both a gain deficit and a conservative-bias suppression while leaving single-input silence untouched by
construction.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._pmem_sfa_nmda_amplifier_derisk --seed 100 --smoke   # target-seed smoke
SIM_BACKEND=numpy python -m research.runners._pmem_sfa_nmda_amplifier_derisk --derisk             # 6 seeds, ARM1+ARM2
```
