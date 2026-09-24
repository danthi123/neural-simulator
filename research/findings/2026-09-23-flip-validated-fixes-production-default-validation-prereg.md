---
type: finding
status: live
date: 2026-09-23
lane: load-bearing-fraction
mechanism: PRE-REGISTRATION of the production-default validation for three fixes flipped default-ON on branch research/flip-validated-fixes (BRAIN_EPISODIC_STORE_VERIFY, BRAIN_PMEM_FACILITATION, BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE). Two 6-seed sharded batteries with no fix flags passed, one with thin probes and one with adequate probes.
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. Filed in its own commit before any flipdefaults-thin or flipdefaults-adequate shard existed. No result is claimed here.
runner: research/runners/load_bearing_fraction.py (sharded by tools/lb_shard.py; guarded by tools/assert_flipped_defaults.py)
artifacts:
  - research/findings/raw/_load_bearing/_shards/allfixes2/aggregate.json
  - research/findings/raw/_load_bearing/load_bearing_6seed_aggregate.json
  - research/findings/raw/_load_bearing/_6seed_aggregate.txt
  - research/findings/raw/_flip_validated_fixes/chat_sha_verdict.json
---

# Flip validated fixes: production-default validation, PRE-REGISTRATION

**Filed 2026-09-23, in its own commit, before any shard of either battery below was run.** Seeds 42 43 44 100 101 102.
The job lists are committed AFTER this document, as
`research/findings/raw/_load_bearing/_shards/flipdefaults-thin/JOBS.txt` and
`research/findings/raw/_load_bearing/_shards/flipdefaults-adequate/JOBS.txt`.

## What changed, and what was already seen

The owner authorized Claude to make validated default-flips (2026-09-23; `docs/plans/2026-09-23-autonomous-charter.md`
§5). Branch `research/flip-validated-fixes` changes three reader defaults from OFF to ON. An explicit `=0` still forces
each one off.

| flag | reader constant | faculty row |
|---|---|---|
| BRAIN_EPISODIC_STORE_VERIFY | `_STORE_VERIFY_DEFAULT_ON` in research/runners/_episodic_dap_dialogue_memory.py | episodic-memory |
| BRAIN_PMEM_FACILITATION | `_PMEM_FACILITATION_DEFAULT_ON` in research/runners/prospective_memory_production_organ.py | prospective-memory |
| BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE | `_ABSTAIN_AT_TIE_DEFAULT_ON` in research/runners/source_provenance_honesty.py | source-provenance-honesty |

Seen before this was written:

- **Each fix's own 6-seed verification.** Episodic: `research/findings/raw/_lbf_fix_episodic_store/verdict.json`.
  Prospective memory: `research/findings/raw/_pmem_facilitation.json`. Source provenance: finding
  `2026-09-22-source-provenance-abstain-at-tie-load-bearing-6-6.md`.
- **The combined battery `allfixes2`.** It ran adequate probes with all four fix flags set, including
  BRAIN_PMEM_OP_STABILIZER, which is NOT flipped. Robust core 24, union 25, mean fraction 0.949, no incomplete
  faculties. All three flipped faculties were load-bearing 6/6 there.
- **The thin-probe 6-seed baseline from 2026-09-20.** It ran on AWS as a single process per seed, not sharded, and with
  none of the fixes. Robust core 14, mean 0.5897. source-provenance-honesty and affect-marker-spiking-wta were each
  load-bearing in 4 of 6 seeds. episodic-memory and prospective-memory were 0/6, which a later finding traced to the
  thin probe, not to the brain.
- **The flag-OFF chat identity check** (`research/findings/raw/_flip_validated_fixes/chat_sha_verdict.json`). A scripted
  10-turn `/api/brain-chat` session (tiny-demo brain, numpy, one process per tree) ran with all three flags set to
  `0`. Every turn returned HTTP 200, and all 10 per-turn sha256 values matched between this branch and the pinned
  pre-flip commit 033e385b8. The session hash was the same in both trees. That is the OFF-arm check; the criteria
  below do not govern it. Its reach is limited. The session forms a reminder and fires it (turns 3 and 6, per the recorded
  answers). It includes single-fact recall turns, where the source-provenance organ (default-ON since 2026-09-01)
  runs; the artifact stores hashes and answers, not the `provenance` key, so that path is not separately confirmed. It does NOT reach the episodic store, because on numpy the handler
  defers the episodic WRITE to cupy, so the turn-7 recall abstains in both trees.
- **The same session with the flags UNSET (the new default), branch only.** It differs from the OFF arm on turns 3, 5
  and 6, the prospective-memory turns. The answer text is the same on those turns, so the change is in other response
  fields. This is informational only.

## The two batteries

Both pass NO fix flag, so each mechanism runs at its production default. Every job line starts with
`.venv/bin/python tools/assert_flipped_defaults.py &&`. That guard exits 1 on a revision without the flip, or when any
of the three flags is set in the environment. Without it, a job dispatched from a pre-flip checkout would silently
measure the old OFF defaults and still look valid.

```
python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag flipdefaults-thin     --no-fixes --probe-set thin
python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag flipdefaults-adequate --no-fixes
python tools/lb_shard.py aggregate --tag flipdefaults-thin
python tools/lb_shard.py aggregate --tag flipdefaults-adequate
```

Each battery is 31 faculties x 6 seeds = 186 shards, numpy backend, run at a revision that contains the flip (the
branch head or its merge). The shard's provenance sidecar records the git SHA.

## Pass criteria — ADEQUATE battery (`flipdefaults-adequate`), compared against `allfixes2`

- **A1 (the flipped faculties).** episodic-memory, prospective-memory and source-provenance-honesty are each
  load-bearing in 6 of 6 seeds, null-control clean, with no UNRELIABLE seed.
- **A2 (no regression).** For every faculty in the allfixes2 aggregate (cited above), `n_load_bearing` is not lower than in
  allfixes2. numpy shards are deterministic, so no tolerance is allowed.
- **A3 (completeness).** `incomplete_faculties` is empty.

Predictions. A1 for episodic-memory and source-provenance-honesty: pass (same code path as allfixes2, now reached
through the default). A1 for prospective-memory is the **open question**. allfixes2 also set
BRAIN_PMEM_OP_STABILIZER, and this battery does not. The facilitation-only organ gate was 6/6, but seed 44's margin
there was thin (+0.011 <!--derived--> above FIRE_THR, per the adversarial correction 73d1a2cbc of finding
2026-09-22-prospective-memory-facilitation-load-bearing-6seed.md). A seed-44 miss here would be a real result: the fix does not hold alone at the
production default.

## Pass criteria — THIN battery (`flipdefaults-thin`), compared against the 2026-09-20 thin baseline

- **T1 (no regression).** Every faculty in the thin robust core 14 is load-bearing 6/6. No other faculty's
  load-bearing seed count falls below its 2026-09-20 count.
- **T2 (the tie fix under thin probes).** source-provenance-honesty is load-bearing 6/6, up from 4/6. The thin
  baseline missed on seeds 44 and 102 (`research/findings/raw/_load_bearing/load_bearing_s44.json`,
  `load_bearing_s102.json`), the same two seeds the fix was built for. The fix acts on the lesion arm's zero-signal
  tie, which no probe flag touches, so it predicts 6/6.
- **Not gated: episodic-memory and prospective-memory.** The thin probe does not exercise them: it asks for recall in a
  fresh session with nothing stored, and it has no intervening-turn cue. Both are predicted to stay 0/6. That is an
  instrument limit, not a regression. They are reported only.

## Decision rule

- **All criteria pass:** the flips stand. The three ledger `subflag_flip_2026_09_23` notes change from "pending" to
  validated, and the branch is eligible to merge after adversarial review.
- **A1 fails for one flag:** revert that flag's default before merging. The other two are judged on their own
  criteria.
- **A2 or T1 fails for a faculty that is not flipped:** re-run that faculty with all three flags set to `0` at the same
  revision (`--extra-env BRAIN_EPISODIC_STORE_VERIFY=0 BRAIN_PMEM_FACILITATION=0 BRAIN_SOURCE_PROV_ABSTAIN_AT_TIE=0`,
  tag `<tag>-off`, without the guard prefix). If the OFF run also misses, the cause is not the flip. Candidates are the
  thin baseline being monolithic rather than sharded, or other code merged since 2026-09-20; record it as that. If the
  OFF run matches the baseline, the flip caused it: revert the flag responsible, bisecting the three if needed.
- A result is UNDEFINED, not a pass, for any faculty whose shard is missing or UNRELIABLE.

## Not measured by either battery

- **Latency.** The episodic store now runs at least one extra recall read per stored topic, plus a re-form on every
  failed lap (capped at 8). No artifact measures its wall-clock cost in a chat turn yet.
- **Adversarial review.** None of the three mechanism merges records an explicit `safe_to_merge` verdict. The review of
  this branch has to supply it.
