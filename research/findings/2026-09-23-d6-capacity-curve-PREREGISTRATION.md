---
type: finding
status: live
date: 2026-09-23
lane: D6-learn-and-grow
mechanism: D6 capacity-curve PRE-REGISTRATION -- recall, interference, memory and time per fact as the number of facts taught in conversation grows (N in 5/50/500/2000), for the D6 local Hebbian store write (BRAIN_D6_HEBBIAN_STORE=1 with ENGRAM_VOCAB and ENGRAM_READTIME) versus the host pattern copy (all D6 flags 0), with a write-freeze null at every N
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only (filed before any run of research/runners/d6_capacity_curve.py). No result is claimed here.
runner: research/runners/d6_capacity_curve.py
builds_on:
  - research/findings/2026-09-23-d6-learn-through-use-v3-capability-gate-GO-6of6.md
  - research/findings/2026-09-05-onebrain-fact-shard-dg-ca3-sublinear-spiking-retrieval-derisk-GO.md
artifacts:
  - research/runners/d6_capacity_curve.py
  - research/findings/raw/_d6_learn_through_use_v3/d6_ltu_v3_6seed_verdict.json
  - research/findings/raw/_onebrain_fact_shard/derisk_404_6seed.json
---

# D6 capacity curve: PRE-REGISTRATION (filed before any run)

**Filed 2026-09-23, in its own commit, before any job of `research/runners/d6_capacity_curve.py` ran** (the runner
was committed one commit earlier at `28db60987`, selftest only). Seeds 42 43 44 100 101 102. Arm dir
`research/findings/raw/_d6_capacity_curve/`.

## The question

The owner asked (2026-09-23) whether the learning D6 proved "scales to the levels needed to go head to head with even
a tiny llm". D6 gate v3 is GO 6/6 on ONE taught fact in the tiny-demo brain. This instrument measures how the same
write behaves as N facts are taught: recall accuracy, interference (false and crossed recall), memory per fact, and
encode + recall time per fact.

## What was SEEN when this was written

- The D6 v3 GO (one fact, tiny-demo brain; `research/findings/raw/_d6_learn_through_use_v3/d6_ltu_v3_6seed_verdict.json`)
  and the fact-shard de-risk (404 co-resident facts, 563 MiB, direct copy write, per-block read ~0.35 s at low load;
  `research/findings/raw/_onebrain_fact_shard/derisk_404_6seed.json`). Both are in `builds_on`.
- The runner's `--selftest` (synthetic arms; no brain). No job of this instrument has run at any N, seed or arm.

## Design (the runner docstring is the full spec)

- **Lexicon**: fixed per seed, 220 agents + 60 actions + 220 patients (disjoint pools), the same at every N.
- **Facts**: one per-seed master list of 2000 SVO facts in blocks of four with controlled overlap:
  `(a,v1,p) (a,v2,q) (b,v3,p) (b,v4,q)`. Each fact has a same-subject and a same-object sibling, every
  (agent, action) pair is unique, and level N teaches the first N facts (nested prefixes).
- **Brain**: the production `OneBrainComposer` with the chat's production arguments (D=128, spiking cleanup on,
  vocab_headroom 128, integrated_loop off, fact-shard retrieval on) and k_max = N + 16. `cfg.seed` is set by the
  composer's bridge builder; the firing-threshold array is hashed per job.
- **Teach**: the chat acquisition call (`hear("a v p", polarity="AFFIRM")` inside
  `d6_hebbian_store.conversation_write`), one fact at a time.
- **Probes** (identical for every arm of a (seed, N) cell): n_probe = min(N, 100) taught facts; one near-miss per
  taught probe (`ask_yes_no(a, v, q)`, q preferably a same-subject sibling's patient); n_probe never-taught
  (agent, action) pairs over stored words (`query_patient`).

## Arms

| arm | env | role |
|---|---|---|
| HEBB | STORE=1, ENGRAM_VOCAB=1, ENGRAM_READTIME=1, FREEZE=0 | the D6 write |
| FREEZE | as HEBB, FREEZE=1 | null: same encode episode, eta=0 |
| COPY | all four D6 flags 0 | the production host pattern copy |
| HEBB_REP | as HEBB, rebuilt | determinism null, N = 5 and 50 only |

## Quantities recorded per (seed, N, arm)

recall (query_patient == taught patient); abstain; crossed recall, split into same-subject and other-fact; wrong
word; yes/no hit; routing-free block decode (all three roles decoded from the block itself); engram held; learned
mean |w| of the probed blocks; near-miss false accept ("yes"); novel false recall (any answer); the fraction of foil
queries whose DG shard was empty (a host-routing abstain); peak RSS and RSS at each stage; n_total; store synapse count;
wall time per encode (median, mean, last-decile mean); wall time per query; build time; projected read-time-view cost
per turn (N x the median per-block engram read).

## The pre-registered gate (implemented verbatim in `score_cell` / `_level_label` / `curve_verdict`)

**Cell (seed, N) is DEFINED** only if every required arm (HEBB, FREEZE, COPY, plus HEBB_REP at N in {5, 50}) exists,
has no error, taught exactly N facts, and shares the substrate threshold hash, the fact hash, the probe hash and
n_total with the other arms. Otherwise, the cell is UNDEFINED.
A DEFINED cell must also pass all four of these checks, or it becomes UNDEFINED:
- **FREEZE null**: recall <= 0.05, novel false recall <= 0.05, near-miss false accept <= 0.05, and the mean |w| of
  the probed blocks is exactly 0.
- **Lever**: HEBB mean |w| > 0.5.
- **No store write during the probe phase**, in any arm.
- **Determinism**: HEBB_REP gives the same decisions as HEBB on every probe (at N = 5 and 50).

**An arm SCALES in a cell** iff recall >= 0.90 AND novel false recall <= 0.05 AND near-miss false accept <= 0.05.

**Level label (per arm, per N)**: INCOMPLETE if fewer than 6 seeds; UNDEFINED if any seed's cell is UNDEFINED;
SCALES if 6/6 seeds scale; FAILS if 0/6; MIXED(k/6) otherwise.

**Parity label (HEBB vs COPY, per N)**: PARITY if on all 6 seeds |recall_HEBB - recall_COPY| <= 0.05 and both false
rates differ by <= 0.05; HEBB-WORSE if recall_HEBB < recall_COPY - 0.05 on >= 4 seeds; HEBB-BETTER if
recall_HEBB > recall_COPY + 0.05 on >= 4 seeds; PARITY-MIXED otherwise (the residual); UNDEFINED if any cell is.

**Curve verdict (per arm, over the levels run)**, exhaustive:
- UNDEFINED: any level is UNDEFINED or INCOMPLETE.
- SCALES-TO-Nmax: SCALES at every level.
- CEILING-BETWEEN-Nk-AND-Nk+1: SCALES at every level up to Nk, not SCALES at every level above.
- FAILS-FROM-5: not SCALES at any level.
- NON-MONOTONE: every other pattern (the residual).

Memory and time carry no threshold. They are reported as curves: per-N medians over seeds and the per-fact slope of
peak RSS between levels. Speed is secondary under the charter and never enters a verdict.

## The N = 2000 fit rule (measure, not guess)

After the N = 5 and N = 50 smoke on seed 42, one `--resource-probe` job builds the full N = 2000 bridge
(k_max 2016), teaches 5 facts under HEBB and runs three queries. Projected peak for a full N = 2000 job = the probe's
peak RSS + (per-fact RSS growth during the N = 50 smoke's teach phase x 2000) + the N = 50 smoke's probe-phase growth.
N = 2000 is dropped from the grid only if that projection exceeds BOTH 13 GB (a 15 GB pool node minus 2 GB) AND
120 GB (a 128 GB AWS node minus 8 GB). Wall time is reported but is not a drop criterion. The same rule, applied to
N = 500, decides which N = 500 jobs go to the pool and which go to AWS.

## Predictions (written before any run)

1. **Recall stays high and flat in N for HEBB and COPY (PARITY at every level); FREEZE abstains at every N.** The
   store is one block per fact: disjoint trigger -> readout synapses, D = 128 per fact. So two stored facts share no
   synapse, and storage interference between facts is zero by construction. This is a localist memory, not a
   superposition. What could still produce errors is the routing (DG bucket collisions), cleanup crowding over the
   fixed ~500-word lexicon plus 128 headroom slots, and left-over substrate state between encodes.
2. **Novel false recall about 0, but mostly credited to the host routing**: an unstored (agent, action) combination
   usually intersects to an empty DG shard, so no substrate read happens. The `novel_shard_empty` split reports how
   much. The near-miss foils (same agent and action, sibling patient) do reach the substrate decode, so they are the
   interference measure this instrument credits to the brain.
3. **The scaling wall is cost, not accuracy.** Every resonate step, and every Izhikevich step of the on-bridge parser
   during `hear`, touches all n_total neurons, and n_total grows by 129 per fact. So wall time per encode and per
   query grows about linearly in N, and teaching N facts costs O(N^2). The read-time engram view re-reads every block
   after each teach turn, so its per-turn cost grows about linearly in N. Memory per fact is predicted constant
   (bridge state per neuron).
4. If prediction 1 fails for HEBB but not for COPY at some N (HEBB-WORSE), the write is the limit. If both fail, the
   substrate store/read or the routing is the limit. The block-decode split tells which.

## What this can and cannot answer about "a tiny LLM"

This measures declarative SVO fact storage and cued recall, one fact per teach turn. It reports facts per GB of RSS
and seconds per fact taught or recalled. A small language model stores knowledge in shared weights and generates
text; this instrument measures neither of those. So the only comparison it supports is on those two axes. It makes
no claim about language competence. The capacity the brain would need for fluent conversation is a separate question.

## Declared host shortcuts (none credited to the brain)

The DG fact-shard routing is built from the host kb record (the declared DG host-rate stand-in, and d6_hebbian_store
shortcut (h)). The engram-held decision is a host threshold (shortcut (e)). The instructive pathway, the phase lock
and the W_MAX clamp are shortcuts (a) to (c). ENGRAM_VOCAB and ENGRAM_READTIME act only on chat-level readers, which
this composer-level instrument does not run. The DA encoding gain is off (g = 1; residual (f)). The lexicon is
pre-known: runtime word recruitment is a separate capacity axis. The chat's polarity extractor and verb lemmatizer are
bypassed, because facts are generated as base-form "a v p". A zero (frozen) block decodes to the codebook's first word
(`ac00`, an action), which can never match an agent cue.

## AMENDMENT LOG

- **A1 (2026-09-23, after the first smoke job only: seed 42, N = 5, arm HEBB,
  `research/findings/raw/_d6_capacity_curve/smoke/s42_N5_HEBB.json`, all five probes correct).** The near-miss row's
  `shard` field was computed on a three-role cue (agent, action, patient), but `ask_yes_no` routes on (agent, action)
  (`_fact_shard_yesno_match`). The field now records the (agent, action) shard the query actually uses. This changes
  only the reported, non-scoring `nearmiss_shard_empty`. No threshold, band, arm or probe changed. That job is
  re-run at the amended revision, and the pre-amendment file is kept as `s42_N5_HEBB.pre_A1.json`.

## Honesty

Functional read-outs only. "Learns" and "recalls" mean the recall answer changes with the synaptic write, measured
against the write-freeze null. No claim of felt experience.
