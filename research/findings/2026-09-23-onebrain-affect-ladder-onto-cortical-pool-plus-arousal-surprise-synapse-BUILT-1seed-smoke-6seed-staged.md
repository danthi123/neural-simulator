---
type: finding
status: partial
date: 2026-09-23
lane: one-brain/migration (charter D3)
mechanism: the Gate-B affect ladder (the production affect organ) moved onto the shared cortical pool as a 12th organ,
  plus a fixed cross-region synapse from its own latched arousal rungs onto the D2 surprise pool
seeds: [42]
runner: research/runners/_onebrain_affect_pool_verify.py
artifacts:
  - research/findings/raw/_onebrain_affect_pool/smoke_seed42.json
  - research/findings/raw/_onebrain_affect_pool/calibrate_seed42.json
builds_on:
  - research/findings/2026-09-17-onebrain-wave3-organ-merge-ALL-11-organs-one-pool-GO.md
  - research/findings/2026-08-13-per-region-ou-wiring-affect-GO.md
  - research/findings/2026-09-02-crossedge-arousal-surprise-derisk-PARTIAL-smoke-go.md
---

# D3: the affect organ onto the shared cortical pool, with an arousal-to-surprise synapse. Built, 1-seed smoke passes, 6-seed gate staged

**Status: PARTIAL.** This is a 1-seed smoke plus a 1-seed weight sweep. The pre-registered 6-seed gate is on the
mini-PC pool and has not been harvested. Everything here is DEFAULT-OFF. It is not wired into production by default,
and no default was flipped.

**Fix round (2026-09-23, after the adversarial review of `7b46d761e`).** Four corrections, detailed in the section of
that name below: the "gain-like, not a DC bias" claim is withdrawn; `XEDGE_W` = 0.05 is a hand-set constant, not a
calibrated value; the arm-X instrument now measures the surprise organ at its production operating point; and the
arm-X gate now scores a change of the organ's own surprise verdict instead of a 0.10 Hz rate floor. Arm M (the
answer-preservation gate, 6 seeds) is unchanged and still part of GO.

## Why affect is the next organ

The 11-organ Wave-3 pool (default-ON since `8ee5e6817`) holds the cortical organs. Among the charter-D3 core set,
the affect organ still ran on its own bridge in production. `affect_production_organ.read_differential` dispatches to
`_appraisal_interoceptive_ladder_derisk.get_ladder(seed)`, which is a standalone bridge of about 690 neurons. The
other organs that are still off the pool were less ready:

- `d6_multiref_wm` and `prospective_memory` run per session in production, so a process-shared pool would leak state
  between sessions.
- `source_provenance` needs an online re-open for its incremental `encode_fact`.

## What was built (commits `1fe3f56e`, `f9e03e44`, `f77556db`; no `sim/` edit)

- `research/runners/onebrain_affect_pool.py`
  - **`AFFECT_DESCRIPTOR`** is the production ladder's exact region and pathway spec, reused by import. A unit test
    checks that it matches the standalone ladder.
  - **`PoolAffectLadder`** runs the same settle, ramp, drive-off and read protocol on the pool slice. It uses a
    local per-neuron OU stream and local Hebbian-off inside `sequence_isolation`. The appraisal reaches the rungs
    only through the relay synapses, and the anti-cheat assertion is kept.
  - **`AROUSAL_XEDGE`** connects all 160 arousal-rung neurons to the `surprise` pool at a fixed, hand-set weight
    of 0.05. It sits behind the transmission gate `affect_arousal_to_surprise`.
- **Flags.** `BRAIN_ONEBRAIN_AFFECT_POOL` routes the production affect read and `get_merged_cortical_pool` onto the
  12-organ pool. `BRAIN_ONEBRAIN_AFFECT_XEDGE` adds the synapse. Both default to OFF and are read from an
  import-light module.
- **`CrossEdge.transmission_gate`.** An optional field on `CrossEdge`. When it is `None`, the dense dict is
  unchanged (tested).

## Measured (seed 42, numpy, N=7692 neurons on the 12-organ pool)

The smoke is `research/findings/raw/_onebrain_affect_pool/smoke_seed42.json`. All 7 of its checks pass:

- **Co-residence.** Affect read on the 12-organ pool matches affect alone on the superset config, with max |delta| = 0.0.
- **Tone levels.** The graded tone levels match the standalone production ladder at every production appraisal:
  (-3, -2, 0, 2, 3) on both.
- **Alive checks.** Correct sign at |a| >= 0.5. The neutral read is exactly 0.0. The affect_out lesion gives 0.0.
- **Determinism.** Two reads give identical results.
- **Edge isolation.** The arousal synapse leaves the ladder's own read unchanged (max |delta| = 0.0).

The continuous differentials differ from standalone, as expected: per-neuron het and OU are realized differently.
Pool vs standalone (rounded from the smoke artifact) <!--derived-->: -0.0772 vs -0.0833, -0.0394 vs -0.0389, 0.0347 vs 0.0378, 0.0800 vs 0.0864.

<!--derived-->
| effective arousal->surprise weight | contradict shift, a=+1 vs a=0 (Hz) | shift, a=-1 (Hz) | confirm at a=+1 (Hz; threshold 2.629) |
|---|---|---|---|
| 0.02 | +0.094 | +0.094 | 0.007 (same as base) |
| 0.05 (the hand-set `XEDGE_W`) | +0.224 | +0.224 | 0.007 (same as base) |
| 0.10 | +0.376 | +0.376 | 0.022 |
| 0.20 | +0.644 | +0.644 | 0.029 |
| 0.40 | +4.398 | +4.398 | 3.299 (confirm over threshold; runaway) |

Source: `research/findings/raw/_onebrain_affect_pool/calibrate_seed42.json`. The latched arousal rungs fire at 78.4 Hz during the surprise window, and the
relays are silent there (asserted on every step). At w=0.05 the contradict rate rises and confirm stays at its
0.007 Hz floor. That does NOT show a gain effect: confirm sits at a ~0 Hz floor, where a sub-threshold additive
(DC) depolarization would also leave it unchanged, and at w=0.4 confirm jumps to 3.30 Hz, which looks more like
DC. Gain vs DC was not tested and no claim is made. This sweep was also measured at a non-production operating
point (see the fix round), and at 600 pA every contradiction was already flagged at a=0 (all 8 per-block rates
2.66 to 5.21 Hz vs threshold 2.63), so no verdict could change.

## Honest residuals

1. **Mechanism level only.** Production organ reads are isolated per organ. The surprise read hard-resets the
   bridge, and the affect read runs inside `sequence_isolation`. So in a live turn the held arousal does not yet
   reach the surprise verdict. The next rung is a turn-scoped shared-state protocol. Until that exists, this synapse
   is not load-bearing on a reply.
2. **Symmetric by construction.** The +1 and -1 shifts are identical because the arousal relay is driven by
   |appraisal|. So "valence-independent" is a property of the design, not a discovery.
3. **Thin +0.5 margin.** At a=+0.5 the pool differential of 0.0347 is only 0.0047 above <!--derived--> the level-2 boundary of 0.03.
   The M3 check, where tone levels must equal standalone, could flip on another seed.
4. **Hebbian clip trap.** Found while building: with the pool's global Hebbian on, one read on an affect-only config
   clipped the `plastic=False` ladder edges to `hebbian_max_weight` (28 -> 1.0). This is the known plasticity BOUND
   TRAP. The read now turns Hebbian off locally, which is faithful because the standalone ladder has it off. M5
   checks the ladder weights across the whole 12-organ lifecycle.
5. **Seed 42 was the v1 sweep seed** and is also one of the 6 gate seeds. The sweep no longer sets anything: the
   weight is hand-set, and the v2 diagnostic sweep runs on non-gate seed 7.
6. **No same-count control.** No arm adds the same number of excitatory synapses from non-affect cells. So the arm-X
   result, if it passes, shows that the ladder's own held arousal, through this synapse, changes the surprise
   verdict. It does not show anything specific to arousal beyond "extra excitation onto surprise".
7. **Production strength is saturated.** Production drives every assertion at 600 pA. The v1 seed-42 read flagged
   every contradiction at that strength, so the v2 functional check reads at a marginal (weaker) strength. The
   production-strength verdict changes are reported, not gated. Making the edge matter at production needs graded
   assertion evidence, for example assertion drive set by the comprehension parser's confidence.

## Fix round (2026-09-23)

1. **Gain claim withdrawn** (review issue 1). "Gain-like, not a DC bias" is removed from this finding and from the
   module docstring. LC-NE adaptive gain (Aston-Jones & Cohen 2005) stays only as the biological motivation. The v2
   runner reports mean contradict Hz per assertion strength at a=0 and a=+-1 as raw data, with no gain claim.
2. **`XEDGE_W` provenance** (issue 2). 0.05 was hard-coded in the build commit `1fe3f56ef` (08:59:38). The seed-42
   sweep started at 08:55:41 and took about 1268 s, so it finished after the weight was set, and its result
   (`f77556db`) agreed with it. The weight is a hand-set constant. `--calibrate` is now a diagnostic on non-gate
   seed 7 and does not set it (`onebrain_affect_pool.py` comment corrected in `0f1f35ff1`).
3. **Operating point** (issue 3). v1 ran the surprise window inside the ladder's 8 pA OU on every pool neuron,
   while the production surprise read is noise-free. `local_ou(scope="affect")` now confines the OU current to the
   affect organ's neurons through the engine's `cp_ou_neuron_mask` seam. The surprise read uses the organ's own
   `_drive_read` sequence with `_step`. New check X0 requires that, at a=0 and at every tested strength, each trial's
   verdict equals the verdict of the organ's literal production read path.
4. **Functional gate** (issue 4). The 0.10 Hz floor (about 2% of a saturated 4.5 Hz baseline) is replaced by X1. S*
   is the largest grid strength at which the a=0 brain flags at most half the contradictions, chosen from the a=0
   read only. At S*, the held arousal must newly flag at least max(2, ceil(n/4)) contradictions at a=+1 and at
   a=-1. No S* means UNDEFINED, which is not a pass. X2 requires no confirm false alarm and no lost detection at the
   production 600 pA. Lesion, no-edge, intero-null, byte-off, held-arousal, determinism and OU-scope checks are
   relabelled I1-I7 integrity smokes: required, but they pass largely by construction.
5. **Order robustness** (issue 8). `local_ou` now saves and restores the prior OU state instead of setting it to
   None (unit test `test_local_ou_restores_prior_ou_state_and_scope_mask`).
6. **Process** (issues 5-7). The shared `research/queue/.lane_waiver` written by the build agent was deleted. The
   ARM-M seed-42 job was moved up to the ARM-M block at the head of `pool.queue`. The v1 arm-X jobs on pool42
   (seeds 42, 44, 100) were stopped and the queued v1 seed-43 line removed; their instrument is superseded. The
   isolated-revision corpus gap is closed on main by the provisioner's corpus sync, and `tools/pool_sync.sh` now
   harvests revision-dir results.

The amended gate was committed in `cd5288018` before any v2 run, with an AMENDMENT LOG listing what had been seen.

## The GO gate (pre-registered in the runner docstring, not yet scored)

The literal command is the `--aggregate` line in the runner's docstring. It runs over the per-seed `verify_M_seed*`
and `verify_X_seed*` JSON files in the `_onebrain_affect_pool` raw directory; those files do not exist yet.
`aggregate` ignores arm-X checks from any record without the v2 `x_instrument` tag.

GO needs every one of M1-M7, X0-X2 and I1-I7 to hold on all 6 seeds. A missing arm counts as not passed.
