---
type: finding
status: live
claim_check: measured
date: 2026-09-24
lane: onebrain-integration
mechanism: BRAIN_XEDGE_IN_WAVE3 (default OFF) grows the d6 w{k}->sel cross-edge inside the production merged cortical
  pool, so the live comprehension organ and every per-session d6 organ share one spiking pool with the learned edge
seeds: [7]
prereg: docs/plans/2026-09-24-xedge-in-wave3-PREREG.md
runner: research/runners/_xedge_in_wave3_verify.py
artifacts:
  - research/findings/raw/_xedge_wave3_probe/s42.json
  - research/findings/raw/_xedge_wave3_probe/s7_flag_on_head.json
  - research/findings/raw/_xedge_wave3_probe/s7_flag_off_head.json
  - research/findings/raw/_xedge_in_wave3/s7/g1_compare.json
  - research/findings/raw/_xedge_in_wave3/s7/g1_a2_primary_compare.json
  - research/findings/raw/_xedge_in_wave3/s7/g1_a2_flagonly_compare.json
  - research/findings/raw/_xedge_in_wave3/s7/g1_on.json
  - research/findings/raw/_xedge_in_wave3/s7/g2t_compare.json
  - research/findings/raw/_xedge_in_wave3/s7/g2l_compare.json
  - research/findings/raw/_xedge_in_wave3/s7/g2p_compare.json
  - research/findings/raw/_xedge_in_wave3/s7/g3_selftest.json
verdict: de-risk (seed 7) only. G0 routing reconciled with the flag on and unchanged with it off. G1 as registered is a
  FAIL. The equal read-pass control (amendment A2) reads all 11 organs identical. G2-transient and G3 pass. The flag
  stays OFF; the 6-seed battery is staged for the orchestrator.
---

# BRAIN_XEDGE_IN_WAVE3, seed-7 de-risk: G1 as registered is a FAIL on an unequal read-pass comparator; the equal-pass control reads identical

seed-waiver: this is the ONE dev seed (7) the pre-registration allows the lane. Seeds 42/43/44/100/101/102 were not
run. Nothing here is a 6-seed verdict and nothing is flipped. Terms checked against docs/TERMS.md (byte-identical is
used only where an exact repr-level compare was made in the data; lesion only where the lesioned read was measured).

## Why this lane exists

The S02 identity probe (seed 42, main-equivalent `8e1bbc470`, shipped defaults,
`research/findings/raw/_xedge_wave3_probe/s42.json`) read CONFIRMED-SEVERED:
- the production comprehension organ is NOT the xedge holder's `comp_organ` (it rides the Wave-3 merged pool);
- the per-session d6 organ's `shared=` is the separate xedge pool, NOT `get_merged_cortical_pool`.

So in production the learned d6 -> comprehension cross-edge lives on a pool that the live comprehension read never
touches, until the two routes are reconciled. (This is read off the routing booleans; no production turn was
instrumented for it.) The flag `BRAIN_XEDGE_IN_WAVE3` (commit `58eaecd5c`, default OFF) grows the edge inside
`get_merged_cortical_pool` instead, with a per-session d6 sub-slice and a hard reset of transient activity when the
owning session changes.

## Results (seed 7, numpy, AWS pool1/pool2)

| gate | artifact | result |
|---|---|---|
| G0 routing, flag ON | `research/findings/raw/_xedge_wave3_probe/s7_flag_on_head.json` | RECONCILED: (a) and (b) both True, at the merged head `c108ba01c` |
| G0 routing, flag OFF | `research/findings/raw/_xedge_wave3_probe/s7_flag_off_head.json` | CONFIRMED-SEVERED, the same booleans as main |
| G1 as registered | `research/findings/raw/_xedge_in_wave3/s7/g1_compare.json` | FAIL (status NO-GO), see below |
| G1 amendment A2, primary | `research/findings/raw/_xedge_in_wave3/s7/g1_a2_primary_compare.json` | all 11 organs identical |
| G1 amendment A2, flag only | `research/findings/raw/_xedge_in_wave3/s7/g1_a2_flagonly_compare.json` | all 11 organs identical |
| G2-transient (gating) | `research/findings/raw/_xedge_in_wave3/s7/g2t_compare.json` | PASS |
| G2-learning (informational) | `research/findings/raw/_xedge_in_wave3/s7/g2l_compare.json` | diffs confined to the shared learned weight |
| G2-pregrown (A1, non-gating) | `research/findings/raw/_xedge_in_wave3/s7/g2p_compare.json` | PASS |
| G3 load-bearing selftest | `research/findings/raw/_xedge_in_wave3/s7/g3_selftest.json` | PASS |

### G1: the registered FAIL, and why it is an instrument confound

The registered comparison put ON-exercised (a SECOND `_isolated_reads` pass, after 6 credited turns, 2 per-session d6
loads and 2 focused comprehension reads) against OFF-build (a FIRST pass). Four organs differed and every answer was
unchanged: causal_whatif 16.67 (`directed_fwd_BtoD`), worldmodel 18.75 (`expect[+1].pred_pos`), surprise
`calib.pred_gain_max` and source_provenance `content_7` by small amounts. The registered verdict stands as a FAIL.

`_isolated_reads` constructs every organ afresh, and these four organs train or encode on the SHARED pool when they are
constructed. So the second pass is not comparable with the first on EITHER arm. Amendment A2 (committed `733f8314f`,
before the control arms ran) added a build pass + second pass arm with no exercise, on both flags
(`g1_off2.json`, `g1_on2.json` in the same folder). Recomputed from the raw files:
- OFF-build equals OFF2-build and ON-build equals ON2-build for all 11 organs, across separate processes.
- The second pass ALONE moves the same four organs by the same amounts on BOTH flags. That is exactly the
  registered FAIL row, so the pass count accounts for the whole registered delta.
- A2-primary (ON-exercised vs OFF-second): all 11 organs identical, answers equal. The exercise had moved the
  cross-edge from 0.05 to 1.2593 (`w2->A`) and 1.2507 (`w0->P`), recorded in `research/findings/raw/_xedge_in_wave3/s7/g1_on.json`.
- A2-flag-only (ON-second vs OFF-second): all 11 organs identical.

The read values are serialised with `repr(float)`, so "identical" here is a bitwise compare in the data. At seed 7 the
in-pool cross-edge and its exercise leave every other pooled organ's reads byte-identical once the read-pass count is
equal. Provenance note: the off2/on2 arms ran on the `d4da11af6` tree with only the runner replaced by the `32ff5d7ff`
version (the sidecars record the digest mismatch; the file's sha256 equals the branch head's runner). g1_off/g1_on ran on
the same `d4da11af6` tree, so the substrate code is the same on both sides of every compare.

### G2: two concurrent sessions, interleaved

Session A holds {wolf, dog}, session B holds {cat, bird}. In the interleaved run every compared field (d6 judge,
comprehension judge, repair_target, hold-query readout) equals the alone run for both sessions, no output of either
session names the other's referents, and the session guard fired 5 times interleaved and 0 times alone. With learning on,
interleaving changed only the process-shared learned weight (`w0->A` 0.8603 interleaved vs 0.4602 alone), which is
by design. With 24 credited turns first (`w2->A` 4.1639, `w0->P` 4.1329) the transient compare still passes.

Limit of this test, stated: the cross-edge's focus is POSITIONAL (slot `w0`, declared in the prereg §5). Both sessions
read with focus `w0` and get the same comprehension margin, so a leak THROUGH the edge would carry slot information, not
referent content. The content check that can fail is the per-session d6 recovery on the shared pool, and it recovered
exactly each session's own referents in all four runs.

### G3: the grown in-pool edge is load-bearing

After the 80-turn in-pool curriculum (`w2->A` 8.6868, `w0->P` 8.6079, the uncredited edges at 0.05),
`_selftest_loadbearing` gives max |dNet| 0.05 intact and exactly 0.0 with the cross-edge zeroed, so the whole measured
effect is attributable to the edge. `_selftest_livelearn` after growth: 5 of 5 repair-role decisions flip intact, 0 of 5
lesioned. The same selftest on the fresh holder (all edges 0.05) reads 0.0 intact, as expected before growth.

## Pre-existing defect seen on both arms (not caused by this flag)

In the two-session chat smoke through the real `brain_chat` handler, the curiosity -> d6 cross-edge build fails with
`KeyError 'cue'` on BOTH flag settings. Cause, reproduced locally: `onebrain_merge_framework._cached_spec` caches specs
on `(descriptor.key, seed)`, and the Wave-2 verify descriptor for curiosity (which renames `cue` to `cur_cue`) shares the
key `curiosity` with `REGISTRY['curiosity']`. Whichever builds first in a process fills the cache for the other. This
lives on main and is left to a separate fix.

## What this does not claim

- No 6-seed verdict and no flip. The flag stays OFF.
- It does not show the reply follows the held referent's CONTENT; the focus is positional.
- It says nothing about the affect-pool combination (`xedge_in_wave3_enabled()` is False under
  `BRAIN_ONEBRAIN_AFFECT_POOL=1`).

## Next

The orchestrator's 6-seed battery (seeds 42/43/44/100/101/102) at the branch head, per seed: G0 on/off, G1 off2 + on
(compare with pairs `build:build,second:exercised`), G2-transient alone_A/alone_B/interleaved + compare, G3 selftest.
