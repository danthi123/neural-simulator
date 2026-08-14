---
type: finding
status: live
date: 2026-08-13
mechanism: onebrain-composer-merge
---

# One-brain merge — the RECALL COMPOSER bridge + the SURPRISE organ on ONE shared spiking substrate (GO)

**Date:** 2026-08-13 · **Status:** GO (6-seed 42/43/44/100/101/102) — the production RF-phasor recall
COMPOSER (the organ `/api/brain-chat` is built around: `query_patient` + the no-confab moat) and the D2
SURPRISE organ (Izhikevich + Hebbian + homeostasis + the two merge flags) run on ONE `SimulationBridge`
(one `cp_membrane_potential_v`, 5 regions), with the composer recall + the no-confab moat + the surprise
read ALL byte-identical (max delta 0.0) to their standalone references, and a composer store+query
byte-ISOLATED from the surprise organ's slice. A load-bearing cross-organ synapse lives in the shared
pool. NO `sim/` edit. This de-risks the DEEPEST one-substrate rung — the composer is what every organ reads.

## What this de-risks (the mission)

Production now has 4 Gate-B organs sharing substrates across 2 pools (surprise+world-model,
metacog+pragmatic), but BOTH pools were separate from the recall COMPOSER bridge. Merging an organ WITH
the composer is the deepest one-substrate step because the composer is the central spiking organ every
organ reads. This lane DE-RISKS it (NOT a production flip): can the SURPRISE organ share ONE bridge with
the RF-phasor composer, byte-identical, moat intact, with a genuine cross-organ synapse?

## The structure — ONE bridge, 5 regions, two codes on one pool

- **Surprise organ (Izhikevich):** `cue` → `patient_expected`(FS/PV, GABA_A) → `surprise` ← `patient_asserted`.
  1056 neurons, Hebbian topographic recall + homeostasis + `per_region_threshold_heterogeneity` +
  `per_region_homeostasis_isolation`. Its `_step` runs the full Izhikevich `_run_one_simulation_step`.
- **Composer (RF-phasor):** the production `RFPhasorComposer` on a masked SLICE (448 neurons; the CAPSTONE
  index-shift port — rebase bind/unbind/bundle conns by `rf_base`, kick with `neuron_mask`=composer-slice,
  read the slice). Its ops use `rf_resonate_steps`, the FAST PATH that BYPASSES `_run_one_simulation_step`
  and (masked) writes ONLY the composer slice's v/u. Its complex weights (`cp_rf_w_*`) live in composer-region
  rows/cols only.

One shared `cp_membrane_potential_v` (N=1504=1056 surprise + 448 composer) holds BOTH organs (asserted
in-code: one pool, contiguous composer span). Two organs, two codes (Izhikevich rate vs RF phasor), one pool.

## Why byte-identity holds (the mechanism)

The two organs read through DIFFERENT machinery on the SAME neuron array:
- The composer's RF ops never call `_run_one_simulation_step`, never read `cp_neuron_firing_thresholds` /
  the Hebbian / homeostasis code, and (masked) write only the composer slice. So its recall is INVARIANT to
  the surprise organ's Izhikevich state AND to Hebbian/homeostasis being ON — byte-identical to a standalone
  `RFPhasorComposer` (its own per-op RF bridges).
- The surprise organ's Izhikevich `_step` touches every neuron, but the composer region carries no pathway
  to/from the surprise regions (byte-identity config), stays at REST (undriven), and is FROZEN by
  `per_region_homeostasis_isolation` — so the surprise read is byte-identical to the standalone surprise
  organ (rung-1: `2026-08-13-one-brain-merge-CLOSED-per-region-threshold.md` + `...-homeostasis-GO.md`).

## Result (`_onebrain_composer_merge_derisk.py`, 6-seed; `--seed`-driven, `--seeds 42,43,44,100,101,102`)

Facts `[dog→chase→cat, owl→eat→mouse, wolf→hunt→deer]`; unstored cue `lion roar` → the moat must abstain.

| Axis | Verdict | Detail |
|---|---|---|
| one shared neuron pool (composer + surprise) | 6/6 | N=1504 = 1056 surprise + 448 composer, one `cp_membrane_potential_v`, contiguous composer span |
| determinism (`cfg.seed` incl. thresholds) | 6/6 | two fresh builds at one seed → identical v / connections / thresholds |
| COMPOSER recall byte-identical (shared vs isolated) | 6/6 | `['cat','mouse','deer']` == isolated, every seed; recall CORRECT (== stored patients) |
| no-confab MOAT preserved (unstored → abstain) | 6/6 | shared `query_patient('lion','roar')` == None == isolated |
| composer op byte-ISOLATED from surprise slice | 6/6 | a composer store+query leaves surprise v/u/thresholds byte-identical (max err 0.0) |
| SURPRISE read byte-identical (merged vs solo) | 6/6 | max err 0.0 Hz over confirm/contradict/novel per-fact rates |
| surprise faculty alive (contradict ≫ confirm) | 6/6 | separation 5.4–61.3× (byte-identical of a LIVE organ, not a dead one) |
| **MERGE byte-identity GO** | **6/6** | one pool + determinism + both reads byte-identical + moat + op-isolation |
| cross-organ synapse LOAD-BEARING (current-driven source) | 6/6 | composer→surprise edge: intact +92.9…+99.3 Hz, lesion +0.0 Hz, attribution frac 1.0 |

## The cross-organ synapse — load-bearing on the pool; recall-driven is the next rung

A `composer → surprise` edge in the shared `cp_connections` IS load-bearing when its source (composer-region)
neurons emit Izhikevich SPIKES: driving the composer block with current raises the surprise read by
~93–99 Hz on a CONFIRM (where surprise is otherwise ~0), and LESIONING the edge (weight→0, `plastic=False` so
the lesion holds) collapses that interaction to +0.0 Hz (attribution frac 1.0). This proves the pool is
genuinely ONE and a same-code cross-organ synapse acts across it.

BUT the composer's RF-phasor RECALL leaves those neurons in a PHASE state (|Z|~1, not an Izhikevich spike
train), and its `rf_resonate_steps` fast path never traverses `cp_connections` — so the composer's actual
recall does NOT natively drive the edge (measured: RF-recall interaction −0.12…+0.00 Hz, i.e. inert). The
precise obstacle is the RF-phasor ↔ spike-rate CODE gap: the two organs share the pool but speak different
codes, so a `cp_connections` synapse (traversed only by Izhikevich `_step`) cannot be driven by an RF read,
and RF ops (which skip `_step`) cannot be perturbed by it.

**Named engine feature (the surpass, no-defer):** a PHASE→SPIKE TRANSDUCER region — generalize the composer's
EXISTING spiking-cleanup RF-membrane→Izhikevich-WTA read (`_spiking_cleanup` / `_izh_bank`, already validated)
into a first-class shared-bridge primitive, so the composer's recall READOUT emits a spike rate that drives
the cross-organ synapse. This is an in-repo prototype, not an external unknown, so no capability is walled —
it is a scoped next rung. `NO-EXTERNAL-NEEDED:` the transducer already exists as the composer's spiking-cleanup
stage; the rung is to route it onto the shared bridge.

## Read-out — the deepest one-substrate rung is de-risked

⇒ the recall COMPOSER (RF bind/unbind/cleanup + the no-confab moat) can share ONE spiking substrate with a
full production organ (surprise: Izhikevich + Hebbian + homeostasis + the merge flags), every read
byte-identical, the moat intact, a composer op byte-isolated from the organ's slice, and the shared pool
carries a load-bearing cross-organ synapse. The one remaining nuance — driving that synapse FROM the
composer's own recall — is a characterized, in-repo-prototyped next rung (the phase→spike transducer), NOT a
wall.

**Honest scope:** (1) this is a DE-RISK, not a production flip (the composer stays its own bridge in
`/api/brain-chat`; this proves the merge is byte-safe when done). (2) The load-bearing cross-organ synapse is
driven by a current stand-in for the transducer's output; the composer's RECALL driving it is the named next
rung. (3) The composer's fact-store here is the numpy-kb idealization (its documented "principled
idealization"; the RF resonate ops themselves ARE on the shared bridge). (4) `cross_frac` via
`tools.lab.attributable_to`; the lesion holds (pathway `plastic=False`, Hebbian frozen during reads).

CI/repro: `SIM_BACKEND=numpy python -m research.runners._onebrain_composer_merge_derisk --seeds
42,43,44,100,101,102 --out research/findings/raw/_onebrain_composer_merge_6seed.json`. Runner:
`research/runners/_onebrain_composer_merge_derisk.py` (`--seed`, `--seeds`, `--D-cmp`, `--cross-weight`).
De-risk chain: composer+WKV CAPSTONE (`2026-07-20-single-shared-substrate-CAPSTONE-...`) → 2-organ merge
(`2026-08-13-one-brain-merge-CLOSED-per-region-threshold.md`) → THIS (composer + a production organ, one pool).
