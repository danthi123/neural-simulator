---
type: finding
status: boundary
date: 2026-08-13
lane: prospective
mechanism: pmem-perpool-homeostat
runner: research/runners/_pmem_perpool_homeostat_derisk.py
artifacts:
  - research/findings/raw/_pmem_perpool_homeostat.json
  - research/findings/raw/_pmem_perpool_homeostat.json.prov.json
---

# Prospective memory `fire_on_cue`: a per-pool intrinsic-homeostatic set-point GUARANTEES silence and normalizes the operating point — but the residual is a SEPARATION/gain deficit, not the operating-point spread

**Verdict: BOUNDARY at the pre-registered 6-seed gate (4/6; need 5/6). NOT a GO.** Against the prospective-memory
`fire_on_cue` residual (finding `2026-08-13-prospective-memory-intention-latch-cue-monitor-derisk.md`: 3/6, the
release amplitude spread ~4x across seeds under a CONSTANT tonic bias), I built the named surpass — a per-`rel`-pool
intrinsic-plasticity homeostatic SET-POINT on each cue-monitor pool's tonic-inhibition bias (Hengen et al. 2016,
"firing rates return to a precise, cell-autonomous set point"; Styr et al. 2019, activity set-points via intrinsic-
excitability mechanisms; Turrigiano 2011). It WORKS as an operating-point control: every silence clause STAYS 6/6
(the anti-cheat holds — no spurious fires), `fire_on_cue` lifts 3/6 -> 4/6, and the operating-point-limited seed 102
is rescued. But it does NOT reach GO: two seeds fail `fire_on_cue` for a reason the set-point CANNOT fix — a
coincidence SEPARATION/gain deficit — which relocates and refines the parent finding's "subtractive/threshold, not
gain" diagnosis. CPU, 131 s, reuse-by-import of the parent substrate + FROZEN gate, NO `sim/` edit.

Artifact: `research/findings/raw/_pmem_perpool_homeostat.json` (provenance sidecar `.prov.json` beside it).

## The mechanism (the mission's named surpass; label-free; NO `sim/` edit)

Each `rel_X` cue-monitor pool gets a per-pool homeostatic set-point on its tonic-inhibition bias (the parent's
`rel_bias_pA`, a single CONSTANT, is the proxy for the per-pool excitability control biology runs alongside). An
intrinsic-plasticity update `bias_X += eta * (r_set - r_single_X)` adapts each pool's bias so its WORST SUSTAINED
single input settles at a sub-threshold set-point `r_set=0.045 < SILENT_MAX=0.06`. Label-free: it references the
pool's own single-input response, never which cue is correct. Calibrated once per seed (cached), then FROZEN for the
trial. The single-input reference is the MAX of the two silence conditions over their ACTUAL gate exposure — cue-alone
over the 30-step cue read AND held-alone over the full N-turn hold (the `rel` accumulator ramps SLOWLY across the
~300-step hold; a from-rest/short probe under-reads it 0.006 vs 0.062 <!--derived--> and leaves the pool too excitable). Because the
worst single input is pinned sub-threshold PER POOL, every single-input silence condition is sub-threshold by
construction; only the COINCIDENCE (held + cue) may cross. A divisive-normalization FS partner (Carandini & Heeger
2012) was the other named option; it is the wrong tool for the operating-point sub-problem (it cannot lift a hypo pool
over threshold) — see the residual below for where it (and its limits) return.

The substrate ALREADY carries this mechanism natively (`cfg.homeostasis_target_rate=0.02` + threshold adaptation),
but at the default slow timescale (tau ~5000 steps) it barely moves within a short trial — which is why the parent's
constant-bias run (nominal homeostasis ON) still spread 4x. This runner supplies the fast, per-pool, reference-
calibrated version.

## Results — 6 seeds 42/43/44/100/101/102, N=5, HOMEOSTAT-ON vs HOMEOSTAT-OFF (parent constant bias) internal control

The gate (thresholds + per-seed clause logic) is IMPORTED from the parent runner and the substrate class is
monkey-patched, so both arms are scored by the SAME code; the ONLY difference is the per-pool homeostat.

<!--derived-->
| clause | OFF (parent constant bias) | ON (homeostat) |
|---|---|---|
| persistence | 6/6 | **6/6** |
| no_fire_before | 6/6 | **6/6** |
| no_fire_wrongcue | 6/6 | **6/6** |
| no_intention_silent | 6/6 | **6/6** |
| lesion_holds | 6/6 | **6/6** |
| lesion_forgets | 6/6 | **6/6** |
| separation | 5/6 | 5/6 |
| **fire_on_cue** | **3/6** | **4/6** |
| seeds passing ALL clauses | **3/6** | **4/6** |

<!--derived-->
The OFF arm reproduces the parent's 3/6 BOUNDARY exactly (seeds 42/43/101 pass), validating the like-for-like
control. The homeostat holds ALL silence 6/6 (`silence_regressed=[]`, `void_if` anti-cheat clear — it did NOT raise
gains until everything fires; max-silent per seed is 0.041-0.046, all under the 0.06 ceiling with margin) and adds
seed 102 (`fire_min` 0.198 -> 0.266). Per-seed correct-cue release ON vs OFF: 42 `0.269/0.267`, 43 `0.256/0.211`,
44 `0.157/0.164`, 100 `0.085/0.001`, 101 `0.237/0.217`, 102 `0.266/0.198`. The operating-point normalization is
explicit: to equalize the single-input read to ~0.045 the homeostat sets biases spanning **-245 to -1416 pA** across
seeds/pools (vs the parent's one constant -1050) — a per-pool operating-point correction of up to ~1.3 nA.

## The residual is SEPARATION/gain, not the operating point (this refines the parent's diagnosis)

The parent named the residual "a subtractive/threshold deficit, not a gain deficit" and named this per-pool
homeostat as the single-variable fix. Building it shows the operating-point control is necessary and holds silence,
but the two remaining `fire_on_cue` failures are NOT operating-point spread — the set-point reaches its limit for a
DIFFERENT reason per seed, both a coincidence SEPARATION deficit the required absolute window (`FIRE_THR/SILENT_MAX`
= 0.20/0.06 = ratio >= 3.33) exposes:

- **seed 100 — a genuine coincidence-GAIN deficit.** `fire_min/max_silent` = 0.085/0.044 = **1.94x** <!--derived--> — BELOW the 3.33
  the absolute window requires. Even at near-ceiling excitability (bias -245 pA, the homeostat raised it ~800 pA), the
  JOINT held+cue response is intrinsically weak (`fireB` 0.085). A set-point/threshold shift cannot manufacture
  separation that is not in the pool's F-I curve. (It DID lift seed 100 ~140x, 0.0006 -> 0.085, so the operating-point
  correction is real — it is just not the binding constraint here.)
- **seed 44 — a sustained-runaway-forced conservative bias.** Its held-alone input RUNS AWAY over the sustained hold
  (held-probe 0.05, the binding single-input), so the homeostat must set the bias BELOW baseline (-1116/-1416 pA vs
  -1050) to hold silence, which SUPPRESSES the coincidence (`fireA` 0.157 < 0.20). Its separation ratio is 3.41 <!--derived-->
  (it fits the window) — the miss is the fire-suppressing conservative bias the runaway forces, i.e. the recurrent
  sustained runaway, not the operating point.

Both point at the SAME missing companion: a mechanism that DECOUPLES the sustained single input from the transient
coincidence (raising the effective fire/silent ratio) and/or AMPLIFIES the coincidence — a class BEYOND the operating-
point set-point this de-risk supplies. `attributable_to` reads only **16.8%** of the mean-fire ON-vs-OFF difference
as owned by the homeostat: the coincidence already fires on most seeds under the constant bias, so the homeostat's
deliverable is the label-free SILENCE GUARANTEE + operating-point normalization + rescuing the OP-limited seed — not a
wholesale fire increase.

## Honest scope + anti-cheats verified

- **Brain-based / label-free:** the homeostat adapts each pool's tonic-inhibition set-point to the pool's OWN
  single-input firing toward a rate target (intrinsic-excitability homeostasis), never to which cue is correct. All
  reads are `cp_firing_states`.
- **The cheat did NOT happen:** the named risk was "a homeostat that raises all gains until everything fires". The
  `void_if` guard checks every silence clause stays 6/6; it is clear (`silence_regressed=[]`). The homeostat pins the
  strongest SINGLE input sub-threshold by construction, so raising a pool's excitability lifts the COINCIDENCE without
  lifting any single-input silence condition — verified: no_fire_before / no_fire_wrongcue / no_intention_silent /
  lesion_holds / lesion_forgets all 6/6, max-silent 0.041-0.046 < 0.06 on every seed. <!--derived-->
- **Instrument:** the internal HOMEOSTAT-OFF arm reproduces the parent 3/6 exactly (same code path), so the ON-arm
  deltas are attributable to the homeostat, not to a substrate change; the `Verdict` preconditions (silence held,
  fire OFF->ON, ON-vs-OFF control separation) travel with the artifact.
- **HOST-SCAFFOLD, FLAGGED (unchanged from parent):** the cue->action CONTENT binding is installed synaptically; the
  mechanism (hold-across-turns + cue-gated release + per-pool operating-point homeostasis) is brain-based.

## Next single-variable mechanism (named; a DIFFERENT class than the set-point)

The residual is the coincidence SEPARATION/gain sub-problem; the operating-point sub-problem is resolved (silence 6/6,
fire 3/6->4/6, biases span 1.2 nA to normalize). The next mechanism must raise the effective fire/silent ratio:

1. **Spike-frequency ADAPTATION on each `rel` pool** (Kv/M-current AHP; Kandel 6e excitability regulation) with a
   timescale between the coincidence read (~30 steps) and the hold (~300 steps): adapts away the SUSTAINED single-
   input baseline (rescues the seed-44-type runaway — lets the frozen bias rise and the coincidence fire) while the
   fresh cue-onset transient still crosses. This is the companion the pure set-point lacks: it decouples sustained
   from transient, which a subtractive bias cannot.
2. **A supralinear NMDA/dendritic COINCIDENCE amplifier** for the seed-100-type gain deficit (ratio 1.94): a
   plateau/dendritic-spike coincidence nonlinearity multiplies the JOINT held+cue drive specifically. NOTE a divisive-
   normalization FS partner (Carandini & Heeger 2012) is the WRONG tool for seed 100 — dividing an already-weak
   coincidence reduces it further; divisive normalization helps only the sustained-suppression side (option 1).

BANKED METHOD: a per-pool intrinsic-homeostatic bias set-point resolves the operating-point spread and GUARANTEES
silence label-free, but cannot close a coincidence separation/gain deficit. Closure is NOT deferred — it moves to the
adaptation + coincidence-amplification class above.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._pmem_perpool_homeostat_derisk --smoke     # 1 seed, N=3, <60s
SIM_BACKEND=numpy python -m research.runners._pmem_perpool_homeostat_derisk --derisk    # 6 seeds, ON+OFF, ~131s
```
