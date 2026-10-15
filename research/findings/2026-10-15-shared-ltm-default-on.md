---
type: finding
status: confirmed
claim_check: measured (A1/A2 per research/coordination/2026-10-13-shared-ltm-flip-JOBS.md Amendment A over the 18 gate shards at e08895cd3; Hermes's score research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_A_score.json; Claude re-scored the same shards independently 2026-10-15 and got the same verdicts and the same moved key)
date: 2026-10-15
lane: one-brain memory plan step 1 (owner-approved 2026-10-13)
mechanism: BRAIN_SHARED_LTM (one read-only imported knowledge store per server process, shared by every conversation through per-conversation SharedLTMView), RNG isolation forced on with it
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_A_score.json
  - research/coordination/2026-10-13-shared-ltm-flip-JOBS.md
verdict: PASS (A1 6/6, A2 6/6). Sharing is now DEFAULT ON in the server; BRAIN_SHARED_LTM=0 is the escape.
---

# Shared imported knowledge store: default ON

**What changed.** `webapp/server.py` `_SHARED_LTM_DEFAULT_ON` is now True. Every conversation reads ONE copy of the
imported knowledge store (the wikidata_100k prior, about 5.44 GiB) instead of loading its own. RNG isolation is
forced on with it, so a lazy shard build cannot reseed the chat's global noise or carry one conversation's history
into another's. The store is read-only through every view; a write raises. `BRAIN_SHARED_LTM=0` restores the
pre-change per-conversation load, call for call.

**Evidence (registered before data; JOBS file, Amendment A; score `research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_A_score.json`).**
- A1 (strict, blind): `on` equals `offiso` (sharing OFF, isolation ON) on every per-turn record, after dropping only
  wall-clock, latency and RSS fields, on 6/6 gate seeds. Sharing itself changes nothing.
- A2 (declared non-blind): `offiso` vs `off`: every reply-level field identical (reply text, `answer`,
  `abstained`, `recalled_svo`) on 6/6 seeds. The one moved key is `bc_a.d6_hebbian.held.readout` on 5/6 seeds
  (e.g. seed 42: 365.8 vs 375.1). That is the declared read-out change of RNG isolation: the old path reset the
  global noise to the seed-42 stream on every LTM bridge build, and the readout sampled that noise. No reply moves.
- Claude's independent re-score of the 18 shards matched both verdicts and found the same single moved key on the
  same five seeds (seed 44: no differing key).

**History.** The first flip battery (registered 2026-10-13, prereg d8e72021f) FAILED its bar on that same readout
key on 5/6 seeds. Amendment A (c8759a385) split sharing from isolation before any offiso data, and this result
attributes the whole difference to isolation, with sharing byte-neutral.

**What this does not claim.** Sharing is memory engineering of a declared prior (owner 2026-10-13: the imported
store is a declared prior, not learned knowledge). It is not a faculty and adds no ledger row: no reply depends on
whether the store is shared, which is the point of A1. It does not claim that the knowledge was learned, and it
changes no recall content. Steps 2-3 of the memory plan (M0 turn lock: on; M2 one brain: default OFF) are separate.

**Tests.** `tests/test_shared_ltm.py` pins the new default (`test_default_is_on_with_rng_isolation`) and the escape
(`test_escape_off_makes_the_pre_change_call`); 117 passed with test_one_brain and test_brain_turn_lock, and 75 passed
across the other chat-brain tests (arm renderer, session release, cross-session persistence, rank2 loop, tiered
store), numpy, 2026-10-15.
