---
status: live
type: finding
lane: laneC
date: 2026-09-04
---

# GNW thought-swap eviction gets an LC-NE ADAPTIVE-GAIN readout (WAVE-1 one-brain integration): a locus-coeruleus-like spiking population, driven by the SAME mismatch/salience trigger that already fires the swap, sets the eviction boost's GAIN — GO 5/6 seed-level (6/6 pooled on every criterion but one, which misses narrowly on already-near-zero residuals — see below): graded (LOW>=BASE>=HIGH vacate speed, 6/6), lesionable (LC ablated -> sluggish on every completing seed + outright STICKY — never completes in the same window — on 3/6), and readout-load-bearing (zeroing the gain coefficient at HIGH's own tonic drive reproduces the floor outcome despite lc firing just as much, 6/6)

**Date:** 2026-09-04
**Runner:** `research/runners/_gnw_lc_ne_gain_swap_derisk.py` (reuse-by-import of the ENTIRE swap/eviction substrate from `_gnw_neural_swap_intention_derisk.py` / `_gnw_active_overwrite_derisk.py` / `_gnw_recurrence_weaken_swap_derisk.py` / `_gnw_neural_vacancy_gate_derisk.py`; **NO `sim/` edit**; additive; the ONE new mechanism is a `lc` region + a `mm_ALL -> lc` synapse).
**Backend:** CPU (numpy), cost-routed per the WAVE-1 plan (no GPU). **Seeds:** 42/43/44/100/101/102.
**Verdict:** seed-level **GO on 5/6** (seed 44 lands `UNDEFINED`, not a negative — see below); **pooled GO** — every one of the ten pooled preconditions holds at >=5/6 or 6/6 (`tools.verdict.Verdict`, earned not asserted).
**Artifacts:** `research/findings/raw/_gnw_lc_ne_gain_swap_6seed.json` (+ `.prov.json` sidecar), calibration `research/findings/raw/_gnw_lc_ne_gain_swap.json`.
**Reproduce:**
```
SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -u -m research.runners._gnw_lc_ne_gain_swap_derisk --calibrate --seed 42
SIM_BACKEND=numpy OMP_NUM_THREADS=2 python -u -m research.runners._gnw_lc_ne_gain_swap_derisk --six-seed \
    --json research/findings/raw/_gnw_lc_ne_gain_swap_6seed.json
```
**Builds on / cites:** the swap DECISION+EVICTION+ADMISSION chain this extends,
[`2026-08-19-gnw-neural-swap-intention-GO.md`](2026-08-19-gnw-neural-swap-intention-GO.md) (a spiking
mismatch/salience detector's rate sets a FIXED-gain STD boost via `eff_boost = min(MAX_BOOST, BOOST_GAIN *
mm_rate)`; wired to production as `webapp/gnw_thought_swap.py`) and its own antecedents,
[`2026-08-19-gnw-recurrence-weaken-swap-GO.md`](2026-08-19-gnw-recurrence-weaken-swap-GO.md) (the
Mongillo-Barak-Tsodyks STD eviction effector, `MultiLoopSTD`) and
[`2026-08-19-gnw-neural-vacancy-gate-GO.md`](2026-08-19-gnw-neural-vacancy-gate-GO.md) (the admission half). The
banked NO-GO this finding deliberately does NOT reuse:
[`2026-08-18-gnw-active-overwrite-NOGO.md`](2026-08-18-gnw-active-overwrite-NOGO.md)'s per-slot LATERAL WTA
inhibition lever (break-in/lockout catch-22) — see "What 'WTA suppression' means here" below.
Biology: **Bouret & Sara 2005**, *Trends Neurosci* 28(11):574-582 (phasic LC-NE network reset); **Aston-Jones &
Cohen 2005**, *Annu Rev Neurosci* 28:403-450 (tonic/phasic adaptive gain); **Devauges & Sara 1990**, *Behav Brain
Res* 39(1):19-28 (raising LC-NE firing speeds an attentional shift) — all three verified against PubMed at build
time (PMIDs 16165227, 16022602, 2167690), not quoted from memory. Recorded at
[`research/biology/locus-coeruleus-ne-adaptive-gain-swap.md`](../biology/locus-coeruleus-ne-adaptive-gain-swap.md).
Corpus-first (`before_you_build.sh`) run and logged before building — no prior attempt at this exact mechanism
existed in the record.

## What was built (the ONE new mechanism; everything else reused unchanged)

The existing swap-intention substrate is imported verbatim: the disjoint supra-critical workspace + divisive-norm
pool + tonic thalamus (the emergent one-coalition-at-a-time competition), the neural vacancy gate (occ/gate), the
spiking mismatch/prediction detector (mm/pred) that decides WHEN to swap, and `MultiLoopSTD` — the short-term-
depression EFFECTOR that actually performs the eviction (the incumbent's own sustained firing depletes its
recurrent loop below the sustain knee; it self-evicts). None of this is rebuilt or edited.
Added, additively: a 60-neuron excitatory `lc` region (feedforward, no internal recurrence — matching `mm`'s own
design) that receives (1) a TONIC external current (`ne_tonic_pa`, the graded/lesion independent variable — the
animal's baseline arousal state) and (2) a dense E_TO_E synapse from EVERY pattern's `mm_k` (Bouret & Sara's
phasic "network reset": LC bursts precisely when a salient mismatch is being detected, regardless of which content
is mismatching). `lc`'s own windowed spiking rate, normalized by its measured dynamic range, REPLACES the fixed
`BOOST_GAIN` constant at the exact slot it already occupied:
`boost_gain_eff = GAIN_FLOOR + NE_GAIN_SPAN * (lc_rate_windowed / LC_RATE_REF)`, then
`eff_boost = min(MAX_BOOST, boost_gain_eff * mm_rate_windowed)` — the SAME formula shape as the base finding, one
constant now a spiking readout instead of a host float.
Operating point (frozen from a `--calibrate` run on seed 42, not re-tuned per seed): `GAIN_FLOOR=0.30`,
`NE_GAIN_SPAN=0.45`, `LC_RATE_REF=0.1761` (lc's own measured rate at a 1400 pA tonic drive, mm silent),
`ne_tonic_pa` in {0 (LESION), 250 (LOW), 550 (BASE), 1400 (HIGH)} pA, `w_mm_lc=1.2`. `GAIN_FLOOR` sits BELOW the
pre-existing production `BOOST_GAIN=1.0` by design (see the biology entry): a circuit whose adaptive-gain source
is silent should be the SLOW end of the dimension Devauges & Sara 1990 measured raising NE speeds up, not merely
"the old default".

## What "WTA suppression" means on this substrate (an honest terminology note)

The task named "the existing WTA suppression mechanism" as the eviction substrate to reuse. This workspace's
one-coalition-at-a-time property is functionally winner-take-all (`n_ignited` never exceeds 1; evicting the loser
IS the depression-driven collapse the mismatch population triggers) but it is NOT a separate mutual-lateral-
inhibition circuit of the kind this repo built elsewhere (the BG action-selector's D1/GPi race; the affect-marker
assemblies' FSI cross-inhibition, `2026-08-28-affect-marker-spiking-wta-derisk.md`). A literal per-slot LATERAL WTA
inhibition lever WAS tried for this exact swap task and is a BANKED NEGATIVE
([`2026-08-18-gnw-active-overwrite-NOGO.md`](2026-08-18-gnw-active-overwrite-NOGO.md): WTA strong enough to give
single-content selectivity LOCKS OUT the challenger before it can trigger eviction — the break-in/lockout
catch-22). The mechanism that actually ships the swap, and that this finding gain-modulates, is the STD/divisive-
normalization eviction effector `MultiLoopSTD` — reused UNCHANGED, per "do not rebuild it".

## Result — a graded, lesionable adaptive gain on 6/6 seeds (from the cited 6-seed artifact)

<!--derived-->

| seed | LESION (a_vacate / swapped) | LOW a_vacate | BASE a_vacate | HIGH a_vacate | graded speed | graded clean | lesion sluggish/sticky | readout LB | speedup attr. |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 239 / ✓ | 141 | 127 | 107 | ✓ | ✓ | ✓ (slower) | ✓ | 56.3% |
| 43 | never / ✗ STICKY | 209 | 182 | 140 | ✓ | ✓ | ✓ (sticky) | ✓ | 100.0% |
| 44 | 252 / ✓ | 141 | 127 | 98 | ✓ | ✗ (0.0033→0.0047) | ✓ (slower) | ✓ | 63.5% |
| 100 | 296 / ✗ STICKY | 185 | 153 | 126 | ✓ | ✓ | ✓ (sticky) | ✓ | 88.1% |
| 101 | never / ✗ STICKY | 194 | 169 | 141 | ✓ | ✓ | ✓ (sticky) | ✓ | 100.0% |
| 102 | 227 / ✓ | 147 | 133 | 111 | ✓ | ✓ | ✓ (slower) | ✓ | 56.5% |

`a_vacate` = the step at which the incumbent's windowed rate drops below the ignition threshold (lower = faster).
"never/STICKY" = the incumbent never vacates within the full 320-step window BASE/HIGH complete in comfortably.
`graded speed` requires `LOW>=BASE>=HIGH` strictly, `graded clean` requires `HIGH old_residual_post <= BASE's`,
`readout LB` is the readout-off control described below. `speedup attr.` is `tools.lab.attributable_to` on
(evict_steps − a_vacate) between HIGH (intact gain readout) and the readout-off control (same tonic drive as HIGH,
gain coefficient zeroed) — i.e. what fraction of the SPEEDUP survives when the readout, not the tonic drive, is
cut.

Every seed shows the SAME ordering: LESION slowest (or never), then LOW, BASE, HIGH progressively faster —
`graded_speed` (strict monotonic) holds on **6/6**. `graded_cleanliness` (HIGH's final incumbent residual at or
below BASE's) holds on **5/6**: seed 44 misses (BASE=0.0033, HIGH=0.0047 <!--derived: their difference, 0.0014,
is not itself an artifact field-->), both values already two
orders of magnitude below the 0.167 ignition threshold — an honest miss on an already-clean pair, not a
counter-example to the graded story (a genuine final-state-cleanliness gradient does not appear once a swap is
already comfortably past the collapse knee, because `MultiLoopSTD`'s collapse is an all-or-none Rung-1 bistable
transition: once the incumbent crosses the sustain knee it falls to the SAME rest branch regardless of margin;
the gradient that IS robust — and the one this de-risk's core claim rests on — is SPEED, not final residual).

## Anti-cheats (each measured, `tools.verdict.Verdict` earns the verdict rather than asserting it)

- **LC floor verified (6/6):** the LESION arm's OWN measured `lc_peak` stays below 0.02 on every seed — a genuine
  ablation (the population does not fire), not merely a zeroed input on an intact circuit left to be inferred.
- **Lesion gain pinned at floor (6/6):** `gain_max == GAIN_FLOOR` exactly in the LESION arm on every seed —
  `boost_gain_eff` never rises above the floor regardless of mm's own salience-driven firing.
- **Lesionable / sluggish-or-sticky (6/6):** on every seed the LESION arm is either markedly SLOWER than BASE (3/6:
  seeds 42/44/102, 227-252 steps vs 127-133) or does not complete AT ALL within the 320-step window BASE/HIGH
  complete comfortably in (3/6: seeds 43/100/101) — the task's own hedge ("sluggish/sticky") is not a hedge here,
  it is a real split the substrate itself produces across seeds.
- **Readout-load-bearing, the NON-CIRCULAR control (6/6):** a `readout_off` arm receives the SAME strong tonic
  drive as HIGH (`ne_tonic_pa=1400`, so `lc` fires just as much — verified: `lc_peak` in readout_off is
  comparable to or EXCEEDS HIGH's on every seed) but `ne_gain_span=0` zeroes lc's CONTRIBUTION to the gain
  (`boost_gain_eff` pinned at `GAIN_FLOOR`, verified exactly on all 6). `readout_off` reproduces the LESION-like
  slow/stuck outcome DESPITE lc firing — on 3/6 seeds it does not even complete the swap (43/100/101, matching the
  exact seeds where LESION itself is stuck). This rules out "lc firing incidentally helps via some other route":
  `lc` has NO outgoing projection besides the (host-arithmetic) gain read; the only path from its activity to the
  outcome is the coefficient this control zeroes.
- **No host workspace reset (6/6)** in the CONTINUOUS demonstration — its own freshly-built, untouched substrate
  (mirroring how the base finding's own headline is the first operation on ITS substrate), `isolate=False`,
  `host_workspace_reset_calls==0`, and it swaps (BASE operating point).
- **Determinism (6/6):** build-twice-same-seed `_izh_hash` match.
- **A real bug found and fixed mid-arc (banked again):** the SAME footgun the recurrence-weaken-swap finding
  banked (`RecurrenceDepression` snapshots its `base` recurrent weights from `cp_connections.data` AT
  CONSTRUCTION) reappeared here: the readout-off control's `MultiLoopSTD` was originally constructed AFTER
  low/base/high had already run on the shared substrate, silently capturing a depressed "base" and inflating its
  apparent speed (172 steps) toward HIGH's. Fixed identically — every `MultiLoopSTD` instance, including the
  readout-off control's, is now constructed up front on the freshly-built substrate before any arm runs (readout-
  off's `a_vacate` moved from 172 to 227-297 across seeds once fixed, matching LESION's stuck/slow character far
  more closely, which is what should happen since both pin the same floor gain).
- **Not an RNG-prefix confound:** `_izh_hash` is verified IDENTICAL between the `w_mm_lc=1.2` (intact) and
  `w_mm_lc=0.0` (lesioned) builds at the same seed — the LESION arm's separate build is not comparing different
  neurons, only a different synapse.

## Brain-based check (named honestly, per the task's own ask)

`lc`'s RATE is genuine spiking activity: real Izhikevich neurons on a real `SimulationBridge`, driven by a real
synapse from `mm`'s spikes — not a host-computed scalar. The COUPLING from that rate into
`boost_gain_eff = GAIN_FLOOR + NE_GAIN_SPAN * ne_level` is HOST ARITHMETIC: there is no engine primitive for "one
population's firing rate sets another synapse population's short-term-plasticity release-probability gain". This
is not a new gap this mechanism introduces — it is the SAME already-disclosed residual in
`webapp/gnw_thought_swap.py` ("the mm->boost COUPLING is host arithmetic... a functional correlate only"), now
extended one link further upstream: mm-rate -> lc-rate is a real synapse; lc-rate -> boost_gain is the identical
KIND of host read-out mm-rate -> boost_gain already was in the finding this extends. **Honest verdict: the NE
LEVEL is brain-based (neurons/synapses); the GAIN READOUT is a documented host-arithmetic shortcut, consistent
with (not worse than) the precedent it extends.** The engine primitive that would close this (a population's
spikes directly setting another population's STP `U`) does not exist; named here as the residual to convert.

## Honest limits / remaining scaffolds (named, not claimed closed — this is a runner-level de-risk)

1. The gain readout is host arithmetic (above) — the one honest residual, matching precedent exactly.
2. `GAIN_FLOOR`/`NE_GAIN_SPAN`/`LC_RATE_REF`/the four tonic operating points are empirical calibrations on this
   substrate, frozen from one seed's `--calibrate` run (the same convention every prior rung in this arc uses),
   not biology-required constants.
3. `MAX_BOOST=0.16` (inherited unchanged) caps the readout, so HIGH's speed/cleanliness advantage over BASE
   narrows as both approach saturation — a genuine, reported operating window, not a knife-edge, the same
   character every prior rung in this arc has reported at its own edges.
4. The coalitions, gate/occ/mm/pred pools, and now `lc`, are hand-wired dense frozen populations, not
   self-organized (inherited, unchanged from the finding this extends).
5. **This is a de-risk at the runner level; it is NOT wired to production** (`/api/brain-chat`). The existing
   swap already IS wired via `webapp/gnw_thought_swap.py` at a fixed `BOOST_GAIN`; routing that wiring's gain
   through this mechanism (replacing the fixed constant with a per-turn-persistent `lc` population's rate) is the
   named next rung, out of scope for this WAVE-1 de-risk task.

## Files

Runner: `research/runners/_gnw_lc_ne_gain_swap_derisk.py`. Biology:
`research/biology/locus-coeruleus-ne-adaptive-gain-swap.md`. 6-seed artifact:
`research/findings/raw/_gnw_lc_ne_gain_swap_6seed.json` (+ `.prov.json`). Calibration:
`research/findings/raw/_gnw_lc_ne_gain_swap.json`.
