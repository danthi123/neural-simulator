---
type: finding
status: live
mechanism: gnw-consensus-ltm-exempt-production-flip broad no-regression + moat soak
lane: E-language / knowledge-in-chat
date: 2026-08-27
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_ltm_exempt_broad_flip_soak.json
runner: research/runners/_ltm_exempt_broad_flip_soak.py
---

# The LTM-exemption production flip, broadened — 90 nonexistent probes x 90 LTM facts x 48 buffer facts, 6 seeds, GO 6/6

**Verdict: GO (6/6 seeds).** [`2026-08-27-ltm-exempt-production-flip-knowledge-answers-live-by-default.md`](2026-08-27-ltm-exempt-production-flip-knowledge-answers-live-by-default.md)
shipped `BRAIN_GNW_ORGANB_LTM_EXEMPT` as the production default (`e259fb45a`) on the strength of two 6-seed
de-risks that each checked a HANDFUL of facts (1 anchor LTM probe + 2 more from shard 0, 1 fixed fake fact, 1
buffer fact, per seed). This soak broadens that exact same proof to **90 nonexistent probes, 90 genuine LTM
facts pulled from 53 distinct entities spread across the corpus, and 48 conversational-buffer facts**, on BOTH the
2-organ and 3-organ bus, at the same 6 seeds — with **zero breaches of any kind**. It does not change the
mechanism; it hardens confidence in the default already live.

## What this reuses vs adds

`research/runners/_ltm_exempt_broad_flip_soak.py` calls the exact same combine-level entry points the two
de-risks called — `webapp.gnw_two_organ_bus.two_organ_combine` / `webapp.gnw_three_organ_bus.three_organ_combine`
— with `organb_ltm_exempt` passed as an explicit **function parameter**, never read from the environment inside
the check. No check logic is re-derived. What's new is scale and breadth:

- **MOAT (the critical property):** `--n-fake-probes 15` synthetic (agent,action) pairs PER SEED (90 total), each
  verified absent from the store by a direct `composer.query_patient` sanity check before being probed, must
  abstain with `abstain_reason == "primary_recall_miss"` on both buses.
- **COMMIT:** `--n-ltm-facts 15` genuine facts per seed pulled **round-robin across all 75 shards** of the shipped
  `wikidata_core_15k` bundle (`find_ltm_facts_broad`), not just shard 0 as the de-risks did — 53 distinct real
  agent entities were exercised across the 6 seeds (`chelsea_fc`, `john_lenin`, `joseph_franz_haydn`,
  `princeton_u`, `canada_portal`, ... — a genuine corpus-breadth sample, not one local cluster).
  Each must commit the exact expected patient with the flag ON, and abstain with the flag OFF, on both buses.
- **BUFFER:** `--n-buffer-facts 8` freshly-taught conversational facts per seed (48 total), each compared
  byte-for-byte (`committed` / `organ_b_confirmed` / `organ_b_surprise_hz` / `organ_c_votes` /
  `organ_c_real_vocab_known`) flag on vs off, on both buses.
- **THE `=0` ESCAPE:** a direct, substrate-free env-level check — `BRAIN_GNW_ORGANB_LTM_EXEMPT` genuinely unset
  reads `True` (today's production default); explicit `"0"` reads `False`; explicit `"1"` reads `True`.

SIM_BACKEND=numpy, the tiny-demo + shipped LTM bundle (the identical light path both de-risks used) — no GPU
brain, per the memory-safety constraint for this task.

## Result — clean sweep, every axis, every seed

| seed | moat (fake probes, both buses abstain `primary_recall_miss`) | commit (LTM facts, both buses) | buffer (taught facts, byte-identical both buses) | seed verdict |
|---|---|---|---|---|
| 42  | 15/15 | 15/15 | 8/8 | GO |
| 43  | 15/15 | 15/15 | 8/8 | GO |
| 44  | 15/15 | 15/15 | 8/8 | GO |
| 100 | 15/15 | 15/15 | 8/8 | GO |
| 101 | 15/15 | 15/15 | 8/8 | GO |
| 102 | 15/15 | 15/15 | 8/8 | GO |
| **total** | **90/90** | **90/90** | **48/48** | **GO 6/6** |

`n_dropped_unresolvable_ltm_probes = 0` on every seed — every one of the 90 sampled LTM triples resolved through
the store's own direct read (none hit the separate `*_portal`/`*_core` key-routing gap the parent finding names as
an honest residual; that gap is about USER-typed surface forms, not the store's own internal keys this soak
queries against). Env-level escape: `unset=True` (today's default), `"0"=False`, `"1"=True` — all correct.

**Sample commit rows (seed 42, showing the corpus breadth):**

| agent | action | expected patient | 2-organ commit | 3-organ commit |
|---|---|---|---|---|
| `chelsea_fc` | `country` | `united_kingom` | `united_kingom` | `united_kingom` |
| `phyllis_virginia_daniels` | `instance_of` | `human_specie` | `human_specie` | `human_specie` |
| `the_tar_heel_state` | `country` | `u_s_of_a` | `u_s_of_a` | `u_s_of_a` |
| `princeton_u` | `country` | `u_s_of_a` | `u_s_of_a` | `u_s_of_a` |
| `audio_master_plus_series` | `country` | `u_s_of_a` | `u_s_of_a` | `u_s_of_a` |

## Why zero moat breaches is the number that matters

Every one of the 90 fake probes carries a unique, seed-and-index-salted synthetic name
(`definitely_not_a_stored_entity_{seed}_{i}_{rand}` / `..._relation_...`), so this is not the same single fixed
fake fact re-asked 90 times — it is 90 genuinely distinct absent-fact queries. Organ A's own forward-recall miss
(`primary_recall_miss`) short-circuits before organ B or C is ever consulted, on both buses, on every probe, at
every seed — confirming by construction (not by re-deriving the mechanism) that the exemption lever the flip made
default-ON cannot manufacture an answer for anything the brain never stored, at this larger sample size.

## An off-arm staleness class this runner is immune to by construction

`research/FAILURE_LOG.md`'s 2026-08-27 entries document six flip-soaks whose OFF arm used
`os.environ.pop("BRAIN_*", None)`, which silently went stale (kept reading as OFF-then-actually-ON) the moment
each flag's production default flipped to `"1"`. This runner never touches the environment for the OFF arm inside
a check — `two_organ_combine`/`three_organ_combine` take `organb_ltm_exempt` as an explicit keyword argument, so
`ltm_exempt=False` is unconditionally the OFF behavior regardless of what the module-level default reads. The
runner's own `escape_check()` is the ONLY place that touches the environment, and it asserts the unset/`"0"`/`"1"`
mapping directly rather than assuming it.

## Honest residuals (unchanged from the parent finding)

- The `*_portal`/`*_core` key-routing gap the parent finding names (some country entities keyed differently than
  the bare surface form a user types) is a fact-store routing issue, not a consensus-combine defect — this soak's
  90 LTM probes all resolved through the store's own internal keys, so it does not exercise that gap; it remains
  open, as reported.
- This is a numpy, tiny-demo soak (memory-safety constraint for this task, matching both de-risks' own setup) —
  it does not additionally verify the cupy production path end-to-end; the existing cupy verification for
  `gnw-two-organ-bus` (PI ledger row) is unaffected and untouched by this arc.

## Pool staging

The mini-PC pool (`pool40`/`pool41`/`pool42`) checkouts are all currently on `main` at `65c5a334` — behind even
the flip commit (`e259fb45a`), let alone this branch — so `tools/sweep_pool.sh` cannot run this runner until it is
merged (confirmed live via `ssh pool40/41/42 'cd ~/derisk-pool/sim && git log --oneline -1'`). This soak was
therefore run **locally** (numpy, 6 seeds, the result above) to bank a verdict now. Ready command to stage the
same 6-seed sweep across the pool once merged (one seed per job, round-robin across the 3 nodes; verified by
dry-running `sweep_pool.sh`'s own cartesian-expansion script):

```bash
tools/sweep_pool.sh raw/ltm_exempt_broad_soak \
  'research.runners._ltm_exempt_broad_flip_soak --seeds {V} --n-ltm-facts 15 --n-fake-probes 15 --n-buffer-facts 8' \
  'seed=42,43,44,100,101,102'
```

## Reproduce

```bash
SIM_BACKEND=numpy OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u \
    -m research.runners._ltm_exempt_broad_flip_soak \
    --seeds 42 43 44 100 101 102 --n-ltm-facts 15 --n-fake-probes 15 --n-buffer-facts 8 \
    --json research/findings/raw/_ltm_exempt_broad_flip_soak.json
```
