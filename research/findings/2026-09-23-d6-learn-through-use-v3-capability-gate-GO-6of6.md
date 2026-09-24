---
type: finding
status: live
date: 2026-09-23
lane: D6-learn-and-grow
mechanism: D6 gate v3 (capability gate) — a local spiking Hebbian store write (BRAIN_D6_HEBBIAN_STORE=1 with BRAIN_D6_ENGRAM_VOCAB=1 and BRAIN_D6_ENGRAM_READTIME=1, all default-OFF) replacing the host pattern copy for in-conversation fact acquisition, 6 seeds, numpy
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO 6/6 on the pre-registered K1-K7 (every criterion on every seed). The later recall reply depends on the synaptic write and on the synapse read at probe time; the host record of the fact is inert; the lesion changes only the write; the null is clean. The EXPO_H exposure-matched SECONDARY (K3e/K4e, non-scoring) PASSES on all 6 seeds (measured 2026-09-23 23:05, see the section below).
runner: research/runners/d6_learn_through_use_lb.py
builds_on:
  - research/findings/2026-09-23-d6-learn-through-use-v3-PREREGISTRATION-capability-gate.md
  - research/findings/2026-09-23-d6-learn-through-use-v3-s42-all-criteria-pass-INCOMPLETE-1of6-plus-v1-pool-banked.md (superseded by this finding)
artifacts:
  - research/findings/raw/_d6_learn_through_use_v3/d6_ltu_v3_6seed_verdict.json
  - research/findings/raw/_d6_learn_through_use_v3/d6_ltu_v3_6seed_verdict_with_expo.json
  - research/findings/raw/_d6_learn_through_use_v3/s{42,43,44,100,101,102}_{USE_H,USE_H_REP,USE_D,FREEZE_H,SHUF_H,NOREC_H,ABL_H}.json
---

# D6 learn-through-use, gate v3: GO 6/6 — the recall reply depends on the in-conversation synaptic write <!--derived-->

The question the v3 gate registered: when the brain is taught a fact in conversation ("the wolf hunts the deer") and
later asked "what does the wolf hunt", does the reply depend on the **synaptic write** the teach turn made, rather
than on a host record of the fact? Scored with the pre-registration's verbatim command
(`--score-only --variant capability --arm-dir research/findings/raw/_d6_learn_through_use_v3 --seeds 42 43 44 100 101 102`)
→ `research/findings/raw/_d6_learn_through_use_v3/d6_ltu_v3_6seed_verdict.json`: `n_defined 6, n_go 6, GO true`.

## Result, per criterion (all 6 seeds identical in outcome)
<!--derived from research/findings/raw/_d6_learn_through_use_v3/d6_ltu_v3_6seed_verdict.json -->

| criterion | seeds passing | what it tests |
|---|---|---|
| K1 learns | 6/6 | USE_H's probe recalls `[wolf, hunt, deer]` |
| K2 use-specific | 6/6 | the shuffled-teach arm (SHUF_H) does not recall deer; the double dissociation holds |
| K3 the reply depends on the write | 6/6 | FREEZE_H (same input and host code, write rate 0) abstains; no 'deer' in its reply |
| K4 read off the synapse at probe time | 6/6 | ABL_H (taught block zeroed after the teach turn) abstains |
| K5 host record inert | 6/6 | NOREC_H equals FREEZE_H on every turn (`K5_differing_turns: []` on all seeds) |
| K6 lesion is write-only | 6/6 | same encode episode, zero store writes after teach in lesion arms, zero homeostatic calls |
| K7 null | 6/6 | USE_H equals USE_H_REP on every turn (`null_diffs: 0`), no void arms |

`attributable_to_write` is +1 treatment vs +0 control on every seed (100% of the probe change is attributable to the
write). Secondary C7 (production direct copy USE_D equals USE_H on teach/d2/probe/xprobe) holds on all 6 seeds.

K5 was the pre-registration's one open prediction (the host-list familiarity leak that failed gate v1 on s42, s43
and s100). It held on every seed: removing the host record of the frozen fact changed no turn.

## Not measured yet: the EXPO_H secondary (K3e/K4e)

ADDENDUM A6/A7 of the pre-registration restored an exposure-matched arm (EXPO_H: the same content words with no
write) as a SECONDARY, non-scoring check that FREEZE_H's and ABL_H's abstentions are not produced by word exposure.
It cannot move `go` by construction (selftest `v3_missing_EXPO_is_secondary_only`). It reads **UNDEFINED on all 6
seeds** here: no EXPO_H arm had run on any seed. Six EXPO_H lines are now staged on the pool at revision
`01f7a5a4` (the D6 head with EXPO_H restored). `git diff 40e83981 01f7a5a4` touches only runner scripts: `sim/`,
`experiment/`, `webapp/` and `bridges/` are identical to the fanout revision the K arms ran at. This finding does not
claim K3e/K4e.

## Provenance
- s42: local numpy (the registered first run). s43-s102: pool41/pool42, isolated revision
  `40e839812dc5f0b3d7e403417b696b8e41e487f9`, one job per (seed, arm).
- 12 of those 35 arms first exited rc=0 with NO arm file: a pool out-of-memory episode killed the worker, and the
  runner's `run()` stored `None` and returned 0. They were re-run at the same revision (research/FAILURE_LOG.md
  2026-09-23; the runner now reports `failed_arms` and exits 4, commit `a5010cda`). No arm result was reused from a
  failed run.
- Scored on main with the verbatim command; the K1-K7 scoring code is unchanged between `40e83981` and main (the
  diff adds only the EXPO_H secondary and the `failed_arms` report). `--selftest` prints SELFTEST PASS.

## Honest scope
- Runner-level GO (docs/TERMS.md: the gate's own verdict is positive). Every D6 flag is default-OFF; no production
  default changed. This is not `wired`/`on-by-default`/`scaffold_retired`.
- One pathway (in-conversation declarative fact acquisition), one scripted 5-turn protocol, the tiny-demo brain.
  Off-protocol readers (e.g. the describe path; `ENGRAM_READTIME_NOT_ROUTED`) are not measured.
- The host shortcuts (a)-(j) declared in `research/runners/d6_hebbian_store.py` remain: the host-wired instructive
  pathway, the host phase-lock loop, the W_MAX clamp (only the phase is learned), the held-threshold, the four-reader
  read-time view, the block-to-words map. The write itself is local and spiking; these surrounding pieces are not.
- The teach-turn ack differs between FREEZE_H and USE_H ("The wolf hunts deer." vs "the wolf hunts the deer"), as the
  pre-registration predicted: it is the same-turn readback of the write, reported and not scored.

## Honesty
Functional read-outs only. "Learns" and "recalls" mean the reply's content changes with the synaptic write, measured
by lesion. No claim of felt experience.

## Addendum (2026-09-23 ~22:40): what "the synaptic write" is, measured
The capacity-curve review (branch `research/d6-capacity-curve`, workflow `wf_2a017223-975`) built the D6 write and the
host copy side by side at seed 42, N=5: the Hebbian weights match the host-copy weights with complex correlation
0.99996 (max |dw| 0.031), and all 128 synapses are saturated at W_MAX. The instructive pathway, the phase-lock loop and
the W_MAX clamp (declared shortcuts (a)-(c) above) make the local write a near-copy of the host pattern. So this GO
shows that the recall reply is carried by, and read from, a synaptic change on the substrate; it does NOT show that a
local learning rule discovers the pattern. The same review measured the store's scaling cost: one disjoint block per
fact with exact host routing, so recall cannot degrade with N, while per-operation cost grows linearly (per-fact encode
~12.7 s and a projected read-time view ~3923 s per turn at N=2000). Scaling work continues in workflow `w1di54v7l`
(an honest cost-criterion capacity curve, and a distributed store in shared synapses that can interfere).

## Secondary measured (2026-09-23 ~23:05): EXPO_H K3e/K4e pass on all 6 seeds
The six EXPO_H arms (same content words, no write; revision `01f7a5a4`, whose `sim/`, `experiment/`, `webapp/` and
`bridges/` are identical to the K-arm revision) landed and were scored with the same registered command
(`research/findings/raw/_d6_learn_through_use_v3/d6_ltu_v3_6seed_verdict_with_expo.json`; K1-K7 unchanged, GO 6/6).
On every seed: EXPO_H made no store write and has no taught block; K3e (FREEZE_H.probe == EXPO_H.probe, no 'deer')
and K4e (ABL_H.probe == EXPO_H.probe, no 'deer') both hold. The lesion arms' abstention is therefore the same reply the
brain gives after merely hearing the words — the recall is carried by the write, not by word exposure. As
pre-registered, this secondary cannot move `go`.
