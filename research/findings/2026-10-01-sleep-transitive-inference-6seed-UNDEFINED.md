---
type: finding
status: live
date: 2026-10-01
lane: prove-who-owns-the-computation
mechanism: offline-replay associative inference (sleep transitive inference) on the spiking CA3 replay store
seeds: [42, 43, 44, 100, 101, 102]
---

# Sleep transitive inference — 6-seed gate: UNDEFINED

**PROVISIONAL**: written by the local model from the registered scorer's output; not yet inspected by Claude.

Family verdict: **UNDEFINED** (the scorer's word, never another). pm_verdict: **UNDEFINED**.

Cites `research/findings/raw/_sleep_transitive_inference/aggregate.json` and the six
`research/findings/raw/_sleep_transitive_inference/seed{42,43,44,100,101,102}.json`.

## Per-seed (scorer output)

| seed | verdict | U-rule / false STI gates (from `per_seed`) |
|---|---|---|
| 42 | UNDEFINED | U6 the weight-permutation control lost the premises or flooded the foils: STI3 has no valid control |
| 43 | NO-GO | STI3b_order_necessary, STI7_premises_survive |
| 44 | NO-GO | STI3b_order_necessary, STI7_premises_survive |
| 100 | NO-GO | STI7_premises_survive |
| 101 | UNDEFINED | U6 the weight-permutation control lost the premises or flooded the foils: STI3 has no valid control |
| 102 | NO-GO | STI3b_order_necessary, STI7_premises_survive |

pm_verdict (its own line): **UNDEFINED** — PM1_what_matters_kept, PM2_replay_follows_strength, PM3_detail_fades
all false on the four defined seeds; it never changes the STI word.

## Necessity (label, k of the defined seeds)

Necessity is computed over the four defined seeds (42 and 101 are UNDEFINED, so they contribute no gates).

| mechanism | necessary on | label |
|---|---|---|
| offline_activity | 4 of 4 | contributing |
| swr_replay_propagation | 4 of 4 | contributing |
| weight_change_structure | 4 of 4 | contributing |
| replay_coreactivation | 4 of 4 | contributing |
| temporal_order | 1 of 4 | contributing |
| shared_item | 4 of 4 | contributing |
| pattern_completion | 0 of 4 | not necessary |
| adaptation | 4 of 4 | contributing |
| plasticity_during_replay | 4 of 4 | contributing |

## U6 fires (the weight permutation is not a valid control) — next method (prereg section 10)

Permute within classes of synapses of equal pre-night weight (so premise synapses exchange changes only with each
other) as the next control.
