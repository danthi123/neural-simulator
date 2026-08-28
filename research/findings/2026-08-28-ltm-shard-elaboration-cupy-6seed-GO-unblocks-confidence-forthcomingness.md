---
type: finding
status: positive
date: 2026-08-28
lane: knowledge-integration
verdict: LTM-shard elaboration (`BRAIN_ELABORATE_FROM_LTM_SHARD`, additive/default-OFF, already merged `1b64d563`) is confirmed cupy 6-seed GO — cross-backend confirmation of the prior numpy 3-seed GO. All six checks pass on all 6 seeds: byte_identical_off, load_bearing, varying_shard_changes_elaboration, lesion_reverts, moat, confidence_cap_engages (undefined_reasons empty). The composer now reads the ROUTED long-term-memory shard (not just the recent-turn buffer tier) when elaborating a reply, on the real cupy substrate. This directly UNBLOCKS the confidence->forthcomingness #94 re-test, whose sole remaining blocker was "elaboration reads only the buffer tier, never the routed LTM shard, so the hedge-cap had nothing to trim on the true production floor".
mechanism: cross-backend (cupy 6-seed) confirmation of the LTM-shard elaboration read; the composer elaborates from the routed LTM shard, load-bearing + moat-safe + byte-identical-off
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_ltm_shard_elab/verify_ltm_shard_elab_6seed_cupy.json
runner: research/findings/raw/_ltm_shard_elab/verify_ltm_shard_elab.py
---

# LTM-shard elaboration: cupy 6-seed GO — cross-backend confirmed, unblocks confidence->forthcomingness #94

Artifact: `research/findings/raw/_ltm_shard_elab/verify_ltm_shard_elab_6seed_cupy.json` (cupy, 6 seeds; via the gpu_queue, `VERDICT: GO -- all seeds pass every check`, DONE rc=0 at 14:04).

## What this confirms

`BRAIN_ELABORATE_FROM_LTM_SHARD` (additive, DEFAULT-OFF, byte-identical when off) was merged numpy-3-seed-GO (`1b64d563`, already on `main`). This is its cupy 6-seed cross-backend confirmation. Per-check, all 6 seeds pass:

- **byte_identical_off** — flag off => output byte-identical to pre-change.
- **load_bearing** + **varying_shard_changes_elaboration** — with the flag on, the elaboration content DEPENDS on the routed LTM shard: vary the shard and the elaboration changes.
- **lesion_reverts** — cut the shard read and the change vanishes (the shard is genuinely driving, not decorating).
- **moat** — every elaborated claim stays backed (no fabrication introduced).
- **confidence_cap_engages** — the hedge-cap still trims correctly on the elaborated floor.

## What it unblocks (the next rung)

The confidence->forthcomingness faculty (#94) was reverted to default-OFF because its coupling had nothing to act on: on the true un-overridden production floor, elaboration reached only the recent-turn buffer tier, so high-vs-low confidence produced identical replies (content-exhaustion, not a confidence failure — see `2026-08-27-confidence-forthcomingness-retest-PARTIAL`). With LTM-shard elaboration now reaching PAST the buffer tier (cupy-GO here), the confidence->forthcomingness coupling has real content to trim on the production floor. NEXT RUNG: re-test confidence->forthcomingness with `BRAIN_ELABORATE_FROM_LTM_SHARD=1` on the true floor, through the real `/api/brain-chat` handler; if high-vs-low confidence now yields a genuinely different number of grounded sentences on the production floor (not just under the test-override), it is flip-viable. This also deepens knowledge-in-chat (#66): the composer now draws on routed long-term knowledge, not just the last few turns.

Honest scope: this confirms the elaboration READ is load-bearing + safe; it does NOT itself flip confidence->forthcomingness (that is the named next rung, its own verify). Default-OFF unchanged.
