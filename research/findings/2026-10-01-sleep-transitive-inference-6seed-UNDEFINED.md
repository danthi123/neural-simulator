---
type: finding
status: live
date: 2026-10-01
lane: prove-who-owns-the-computation
mechanism: offline-replay associative inference (sleep transitive inference) on the spiking CA3 replay store
seeds: [42, 43, 44, 100, 101, 102]
---

# Sleep transitive inference — 6-seed gate: UNDEFINED

**PROVISIONAL**: written by the local model from the registered scorer's output. Inspected by Claude 2026-10-01:
the registered grader re-derives `aggregate.json` byte-identical from the six seed records, and its selftest passes.
See "Claude review" at the end.

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

## Claude review (2026-10-01)

**What the four defined seeds show.** Gates STI0-STI6 hold on all four: STI0, STI1, STI2, STI3, STI4, STI5 and
STI6. In words:
- the premises are learned;
- the night writes the A->D link;
- the replay lesion removes the link and keeps the premises;
- a permuted night does not write it;
- the crossed chain moves the inference with the chain;
- no false inference;
- the A->D weight rises.

So on this CA3 store, the night integrates the two premises relationally. Two gates fail:
- **STI7 fails on all four, as registered.** The night that creates A->D erodes the premises. Section 8 predicted this
  from the dev data.
- **STI3b fails on 3 of 4: not predicted.** A night replayed in reverse order also writes the inference, so the
  asymmetric rule's order is not what does the work. This is consistent with co-activation-driven integration, the
  regime of the symmetric CA3-CA3 rule (Mishra 2016).

**The family word.** The registered rule gives UNDEFINED, because seeds 42 and 101 fire U6 (the `wperm` control lost
the premises or flooded the foils). The family could not read GO even if both seeds were rescued, because GO needs
all six seeds and four are NO-GO. Under prereg section 10 this is a NO-GO for the method (asymmetric-STDP night,
PV loop on, no downscaling), not for the capability.

**The ruling** is `T3-sleep-ti` in the runbook. In short:
- No standalone U6 re-run or U6 prereg: it cannot change the outcome.
- The next family registers section 10's STI7 companions with the within-class permutation as its STI3 control from
  the start.
- The first companion is the symmetric CA3-CA3 rule. STI3b's failure makes it the cheapest faithful step.
- Then sleep downscaling.

