---
type: finding
status: contributing
date: 2026-08-11
mechanism: emergence-engine heterosynaptic-competition ALLOCATION (label-free anti-collision winner selection at the on-bridge HTM-TM allocate step) keeps allocation keys DISJOINT under capacity starvation, feeding the selective-write store the clean keys it needs
lane: emergence engine (recurrent spiking sequence/language cortex; roadmap L130 "scale spiking HTM Temporal-Memory generator")
instrument: research/runners/_emerge_hetero_ltd_allocation_derisk.py — an AllocLTDLearner subclass of the EMERGE-14 OnBridgeLearner that, at the ALLOCATE step, competes a new context's k winners against the codes already claimed this epoch by OTHER prev-winner SDRs and picks the k cells minimizing the MAX per-foreign-context overlap (heterosynaptic competition / lateral inhibition), plus an anti-Hebbian synaptic depression of the foreign coincidence afferents (reported). Reuse-by-import of the selective-write store's ContentStore/harvest/read/swap. hetero_ltd=False is the no-allocation-LTD baseline (= the load-bearing lesion). SIM_BACKEND=numpy (CPU); 1-seed SMOKE (6-seed command below).
artifacts:
  - research/findings/raw/_emerge_hetero_ltd_allocation/smoke_nseq3_L16_cells8_seed42.json
  - research/findings/raw/_emerge_hetero_ltd_allocation/hetero_nseq3_L16_cells8_6seed.json
  - research/runners/_emerge_hetero_ltd_allocation_derisk.py
verdict: 6-SEED GO — heterosynaptic-competition allocation rescues the regime-C starvation wall (store 0.333->1.000, all anti-cheats hold across 6 seeds)
---

# Emergence engine — HETEROSYNAPTIC-COMPETITION ALLOCATION keeps the on-bridge HTM-TM's allocation keys DISJOINT under full capacity starvation (n_cells=8), RESCUING the horizon where a content store ALONE collapsed to chance: the no-allocation-LTD baseline store is 0.333 (~chance, 1/3 contexts own a clean key) but WITH hetero-LTD the selective store recovers to 1.000 (3/3 contexts own a clean key). 1-seed SMOKE + the exact 6-seed command. <!--derived--> (0.667=2/3 and 0.333=1/3 are rounded displays of the full-precision values in the cited artifact)

## Why this (our-own-record first — the SEPARATE wall the selective-store SMOKE handed over)

<!--derived-->
`2026-08-11-emergence-engine-selective-write-store-...SMOKE` showed a SELECTIVE-WRITE content store over the HTM-TM's own
allocation keys RESTORES the interference-broken horizon WHEREVER clean allocation keys survive — but hit a hard wall at
regime C (n_cells=8, full starvation): EVERY allocation key MERGES (16/16 ambiguous), the selective gate poisons every
one, the store is empty, and it collapses to chance. It named the surpass verbatim: "a homeostatic/competitive
(heterosynaptic-LTD) allocation that keeps allocation keys disjoint under capacity pressure, feeding the selective store
the clean keys it needs." This de-risk builds + tests exactly that. CROSS-LANE: the SAME biology as the source-monitor
competitive-encoding win (`2026-08-11-source-monitor-competitive-encoding-heterosynaptic-LTD-...6seed`) — foreign/
overlapping cells' shared afferents depressed at encoding so codes stay orthogonal. One mechanism, two lanes.

## The diagnosis (MEASURED first, not assumed — dump of per-context winner SETS at regime C)

<!--derived-->
At n_cells=8, after training, the confident winner-cell SETS in the shared middle are `ctx0->{0,1,2,3}  ctx1->{4,5,6,7}
ctx2->{0,1,2,3}`. **ctx2 REPRODUCES ctx0's EXACT set.** The wall is NOT physical capacity: C(8,4)=70 distinct 4-subsets
exist, and ctx2 could take e.g. {0,1,4,5} (overlap 0.5 with each of ctx0/ctx1, EXACT-distinct from both). The wall is
that the stock allocation RULE (k freshest-committed cells, ties broken by cell index) hands the third context the first
context's cells. So the fix is an allocation COMPETITION that avoids reproducing a foreign context's exact code — the
functional outcome of heterosynaptic LTD + lateral inhibition among competing assemblies.

## Mechanism (reuse-by-import of EMERGE-14 + the selective store; NO `sim/` edit)

At the HTM-TM ALLOCATE step (a new context's prev-winner SDR matches no existing segment), the k winners are chosen to
MINIMIZE the MAX per-foreign-context overlap (greedy: spread across foreign owners first, then committed-count freshness,
then index). The competition is **LABEL-FREE** — keyed on the presynaptic winner SDR (distinct per context because the
cues drive distinct cue-column winners), not a host cue-label. Under starvation this forces ctx2 onto a DISTINCT
combination ({0,1,4,5}) instead of ctx0's {0,1,2,3}, so at least one clean allocation key survives per context. An
anti-Hebbian SYNAPTIC depression of the foreign coincidence afferents (cp_connections.data, from this context's
prev-winners into foreign-claimed cells it did not win) accompanies it and is REPORTED (`foreign_l1_depressed`); at the
sub-connected allocate step this is ~0, so the SELECTION competition is the load-bearing lever (see honesty below).

## Result (1-seed SMOKE, seed 42, n_seq=3, chance 0.333; numpy/CPU) <!--derived-->

<!--derived-->
(store/bare = branch-prediction accuracy, higher = better; clean-key-ctxs = # of the 3 contexts that own >=1 uniquely-identifying allocation key — the keys the selective store can complete from. All 0.333/0.667 in this block are rounded 1/3 and 2/3 displays of the full-precision values in the cited artifact; this section is block-scoped derived to the next heading.)

| regime | n_cells | arm | bare | STORE | store-read-lesion | always-write | permute | swap | clean-key-ctxs | foreign_l1 |
|---|---|---|---|---|---|---|---|---|---|---|
| STARVED (regime C) | 8 | NO-alloc-LTD (baseline / lesion) | 0.333 | **0.333** | — | 0.333 | 0.000 | 0.333 | **1/3** | — |
| STARVED (regime C) | 8 | **HETERO-LTD** | 0.000 | **1.000** | 0.000 | 0.000 | 0.000 | 1.000 | **3/3** | 0.000 |
| FAIR (no-harm) | 20 | NO-alloc-LTD | 0.667 | 1.000 | — | — | — | — | — | — |
| FAIR (no-harm) | 20 | HETERO-LTD | 0.667 | 1.000 | — | — | — | — | — | — |

Raw (seed 42, numpy/CPU): `research/findings/raw/_emerge_hetero_ltd_allocation/smoke_nseq3_L16_cells8_seed42.json` (both starved arms + the two fair-capacity no-harm arms + the 4/4 verdict preconditions).

**The rescue.** At regime C the no-allocation-LTD baseline store sits at chance (0.333) with only 1/3 contexts owning a
clean key (ctx1; ctx0 and ctx2 share the exact same merged code). WITH hetero-LTD every context owns a distinct clean
allocation key (3/3), so the selective store recovers to 1.000. Note bare drops to 0.000 in the hetero arm (the raw chain
still merges at late positions) — the store does ALL the recovery work from the clean EARLY (t=1) keys, exactly its
designed job.

**All anti-cheats bite (via `tools.lab`; preconditions via `tools.verdict.Verdict`, 4/4 satisfied).**
- (a) LOAD-BEARING = anti-cheat (d): the lesion of hetero-LTD IS the no-allocation-LTD baseline at the SAME capacity —
  keys re-merge (clean-key contexts 3 -> 1), the store collapses 1.000 -> 0.333. hetero BEATS baseline by +0.667.
- store-READ lesion (store load-bearing given the clean keys): 1.000 -> 0.000.
- SELECTIVITY GATES: always-write 0.000 (the ambiguous merged keys trap the read -> veto -> bare).
- ATTRIBUTABLE: permute the keys' branch labels (derangement) -> 0.000 <= chance (recovery is key->branch-driven).
- CONTEXT-DRIVEN / NO-CONFAB: swap-follows-context 1.000 (inject a different cue -> completes to the INJECTED branch).
- NO-HARM at fair capacity (n_cells=20): store 1.000 -> 1.000 (hetero is inert where capacity suffices).
- oracle 1.000 (task context-solvable), Markov floor at chance.

## Honesty — what is the load-bearing lever, and the declared residual

<!--derived-->
`foreign_l1_depressed = 0.000`: the anti-Hebbian SYNAPTIC depression is inert at the allocation step (the cross-talk
afferents start sub-connected at p_init, so there is nothing potentiated to depress there). **The load-bearing lever is
the SELECTION competition** (the anti-collision winner choice), which is the functional outcome of heterosynaptic LTD +
lateral inhibition, implemented host-side at the ALREADY host-orchestrated allocation step (the DECLARED EMERGE-9d
residual — winner-selection + committed-metric allocation were never on the substrate). This is at PARITY with the
source-monitor sibling, whose competitive encoding is likewise host-bookkept ("reads the recorded per-source encoding
activity"). The label-free, fully-synaptic, ONLINE realization (derive the foreign-code depression from the substrate's
own firing during encoding, no host claim record) is the burn-down, shared with the source-monitor lane.

## Verdict — 6-SEED GO (decisive run RUN by coordinator)

<!--derived-->
The GO question — "does hetero-LTD allocation RESCUE regime C, keeping enough allocation keys disjoint that the selective
store recovers the horizon where it previously collapsed to chance?" — is answered **yes**: baseline store 0.333
(chance) -> hetero store 1.000, with the load-bearing lesion, attribution, selectivity, no-confab, and fair-capacity
no-harm anti-cheats all holding, and clean-key contexts lifted 1/3 -> 3/3. **The 6-seed decisive sweep was RUN
(coordinator, `research/findings/raw/_emerge_hetero_ltd_allocation/hetero_nseq3_L16_cells8_6seed.json`) and the runner's
own verdict is GO** — the rescue + every anti-cheat holds across seeds 42/43/44/100/101/102. Honest scope unchanged:
`foreign_l1_depressed=0.000` — the SELECTION competition (anti-collision winner choice) is the load-bearing lever, not
the synaptic depression (inert at the sub-connected allocate step); the fully-synaptic label-free ONLINE version is the
declared burn-down, shared with the source-monitor competitive-encoding lane (a cross-lane convergence).

## The decisive 6-seed command (CPU/numpy, NOT cupy; do NOT run the sweep here)

```bash
OUTDIR=research/findings/raw/_emerge_hetero_ltd_allocation; EXT=.json
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._emerge_hetero_ltd_allocation_derisk \
  --seeds 42 43 44 100 101 102 --n-seq 3 --distance 16 --n-cells 8 --fair-cells 20 --epochs 35 \
  --out "$OUTDIR/hetero_nseq3_L16_cells8_6seed$EXT"
```

GO = at n_cells=8 the no-allocation-LTD baseline store collapses to <= chance+0.10 AND hetero store >= 0.90 and >=
baseline+0.15, with clean-key contexts = n_seq, load-bearing + attributable + selective + context-driven + fair-capacity
no-harm, all 6/6. Each point is ~30-60 s CPU/seed across the arms (the coincidence loop is launch-bound), so the
coordinator may parallelise across seeds via the pool.

## NEXT

1. Run the 6-seed sweep above -> the multi-seed rescue surface (this closes the regime-C boundary the selective-store
   SMOKE flagged as "capacity is a SEPARATE, harder wall").
2. Burn-down (shared with the source-monitor lane): the fully-synaptic, label-free ONLINE heterosynaptic LTD — derive the
   foreign-code depression from the substrate's own firing during encoding, retiring the host claim record. NO `sim/`
   edit for the de-risk; the online version may need an additive default-off substrate hook (flag it for review).

Reuse-by-import (EMERGE-14 + the selective-write store); NO `sim/` edit. 1-seed SMOKE; the 6-seed sweep is the decisive run.
