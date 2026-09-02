---
status: live
type: finding
lane: onebrain-merge
date: 2026-09-02
mechanism: onebrain-twopool-merge
---

# One-brain TWO-POOL MERGE — ORGAN-READ rung: all 4 core cortical organs read correctly off ONE shared merged pool (6-seed GO)

**Verdict: GO (6/6 seeds).** The organ-READ rung of the literal two-pool merge (Vikunja #171). It runs
the 4 core cortical organs' REAL production read pipelines on ONE shared merged spiking pool (N=2034 = D2 surprise
1056 + E2 world-model 528 + E1 metacog 290 + D pragmatic 160) and gates each organ's read against its co-resident-
alone baseline, its shipped 2-pool production baseline, and a faculty-alive check — the rung the substrate-init
merge (`2026-08-27-onebrain-twopool-merge-substrate-byte-identity-6seed-GO.md`) named as its follow-on.

## The gap on `main` this closes (why a NEW runner, not the existing framework verify)

Production runs TWO separate merged pools: pool #1 (`onebrain_merge_production.MergedSubstrate`) = surprise +
world-model (hebbian ON, param-het OFF, homeostasis ON); pool #2 (`onebrain_merge_production2.MergedSubstrate2`) =
metacog + pragmatic (frozen, param-het ON, homeostasis OFF). The declarative `onebrain_merge_framework` verifies
each pool's PAIR co-residence-invariant SEPARATELY (`--keys surprise,worldmodel` and `--keys metacog,pragmatic`) —
but had NEVER merged all four on ONE bridge, because their GLOBAL configs conflict. The all-4 merge existed only in
two now-DELETED bespoke runners (`_onebrain_twopool_merge_derisk.py` / `_onebrain_twopool_organread_verify.py`),
landed as documentation-only cherry-picks — their prior 6-seed GO is NOT reproducible on `main`. This runner
restores a RUNNABLE all-4 organ-read verify, built on the declarative `merge_organs` engine (reuse, not a bespoke
pool), so the rung reproduces.

## The reconciliation — 4 global-config conflicts, each a per-region/per-synapse seam (not a wall)

Substrate-INIT never exercised these (no training/dynamics at init); the organ-READ does. Each conflict maps to an
existing framework seam so the 4 organs share ONE superset config:

| conflict | pool #1 | pool #2 | reconciled by |
|---|---|---|---|
| `enable_hebbian_learning` | True (surprise's `judge` trains cue->expected) | False (frozen) | global True + a per-synapse `cp_plasticity_rate_gain = 0.0` FREEZE on every pool-2 INTERNAL edge (`freeze_regions`) — pool-1 Hebbian can never touch a pool-2 weight |
| `enable_parameter_heterogeneity` | False | True | the twopool MASK: global False + name-keyed per-region het on metacog/pragmatic ONLY (`param_het=True`) |
| `hebbian_max_weight` | 45 | 400 | pool-1's 45 globally (pool-2 edges frozen => never clipped) |
| `enable_homeostasis` | True (CoreSimConfig default) | False | global False + per-region `enable_homeostasis=True` on EVERY surprise/world-model region (the diffbuilder mask) |

**The homeostasis conflict was the load-bearing one, and it was SILENT.** With pool-2's global
`enable_homeostasis=False` unreconciled, the world-model went COMPLETELY DEAD on the merged pool — `pred_pos=0.0`
for both contexts, `state_neg` mis-selected to `state_pos` (both 0), pred-sign FLIPPED (`-1` where the shipped pool
reads `+1`). It is not a MergeConflict (no clashing REQUIRED value — pool-1 leaves the key at its default), so the
config union accepted it and the faculty died quietly. The per-region homeostasis mask restores the world-model
(`pred_pos≈435 Hz`, signs `+1/-1`, `states 0,2`). This is the CLAUDE.md "companion process replaced by a constant"
pattern exactly: the animal runs homeostasis alongside the forward model; dropping it globally to satisfy the
frozen pool silently silenced the organ.

**A 5th seam — PER-ORGAN READ ISOLATION (full-snapshot-restore), also silent.** With homeostasis restored the
world-model was alive but its read drifted ~0.7 Hz merged-vs-coresident. Bisected (2026-09-02): the drift is NOT a
co-residence WEIGHT coupling (constructing surprise before world-model leaves its read at `d=0.0`) — it is that
metacog/pragmatic CALIBRATION fires on the shared bridge at their construction, leaving conductance rise/slow
buffers + homeostatic accumulators that the organ's own per-CALL `read_isolation` (per-neuron only) and its
internal `_hard_reset` (v/u only) do NOT wash, and the world-model — a long-integration read under homeostasis —
integrates that residue into a sub-1-Hz shift. Closed by a FULL-SNAPSHOT-RESTORE: snapshot the pool's post-build
pristine dynamical state (all conductance buffers + homeostatic + pulse timers, NOT the trained weights), then
restore it before EVERY organ's read — so all 4 organs read from the SAME clean substrate, order-independently
(the "per-organ read isolation" the board rung names, applied identically to the merged, coresident and shipped
pools so the compare is apples-to-apples). This is the read-side twin of the gain-0 freeze: the freeze stops
pool-1's plasticity leaking into pool-2's WEIGHTS; the snapshot-restore stops any organ's calibration leaking into
another's READ.

## The gate (6 seeds 42/43/44/100/101/102, numpy CPU, bit-exact)

| gate | result |
|---|---|
| (a) ORGAN-READ byte-identity — each organ's read on the merged pool == its read CO-RESIDENT-ALONE on the merged superset config (co-residence invariance) | **6/6, all 4 organs, max delta 0.00e+00** |
| (b) FACULTY-ALIVE — each organ produces its live verdict on the merged pool (surprise contradict>confirm; world-model +/- pred signs opposite & violated>expected; metacog confidence grows with evidence; pragmatic implicature separates the scalar family) | **6/6, all 4 organs** |
| (c) ANSWER-PRESERVATION — each organ's rendered chat answer on the merged pool == its answer on the CURRENT 2-separate-pools production handler | **6/6, all 4 organs** |
| GAIN-0 FREEZE HOLDS — pool-2 internal edge weights byte-identical before vs after the full train+read lifecycle (surprise trained Hebbian on the shared bridge; pool-2 stayed frozen) | **6/6 (n≈26.3k–26.5k pool-2 edges, delta 0.00e+00)** |
| LEGACY DISCRIMINATOR — seams-OFF pool diverges merged-vs-coresident (byte-identity NOT vacuous) | **6/6 (24–25 per seed)** |

Every gate is 6/6 across seeds 42/43/44/100/101/102; `tools.verdict.Verdict` decided **GO**. The organ-read
byte-identity is EXACT (`max delta 0.00e+00`) for all four organs — surprise (`SurpriseProductionOrgan.judge`),
world-model (`WorldModelProductionOrgan.expectation` / `read_surprise`), metacog (`MetacogProductionOrgan.judge`)
and pragmatic (`PragmaticProductionOrgan.interpret`) — over the full calibration + read battery. The faculties are
demonstrably ALIVE on the merged pool (not a vacuous all-zero read): surprise fires harder on a contradiction than
a confirmation, the world-model predicts opposite valence signs for +/- context and fires more on a violated than
an expected turn, metacog's confidence margin grows with evidence, pragmatic's implicature margin separates the
scalar family. The legacy (seams-OFF) discriminator diverges 24–25 per seed, so the byte-identity is NOT a vacuous
all-zero compare.

### Byte-identical vs the SHIPPED 2 pools too (stronger than co-residence invariance)

Under the SAME per-organ read-isolation protocol applied to all three pools, the merged reads are byte-identical
not only to the co-resident-alone baseline but to the SHIPPED 2 production pools (`MergedSubstrate` #1 +
`MergedSubstrate2` #2): **shipped-read byte-identity 6/6, all 4 organs, max delta 0.00e+00.** The earlier ~0.7 Hz
world-model residual was ENTIRELY the read-order/construction leak — once every read starts from the pool's
pristine settled state, the merged pool reproduces each shipped pool's read bit-for-bit. So this is not merely
co-residence-invariant migration safety; the merged single pool is a byte-exact stand-in for the two production
pools on every organ read.

## Honest scope — MIGRATION gate, NOT the one-brain INTEGRATION goal

Byte-identity-in-ISOLATION proves co-locating the 4 organs on ONE substrate did NOT change any read — the
safe-migration gate. It deliberately FORBIDS the cross-region synaptic interaction that IS the one-brain goal: a
pool with zero cross-edges is MIGRATED, not INTEGRATED. The hand-declared block-diagonal / assembly loops / gain-0
freezes are host scaffold; the faithful end state has regions that develop connectivity and interact through
learning. **Not a production flip** — no `BRAIN_*` flag change; production keeps two pools. Per `docs/TERMS.md`,
"closed" needs production integration — this is a 6-seed GO de-risk, the organ-read half of merging the two pools.
Functional read-outs only; no phenomenal claim.

## The named next rung

The production single-`BRAIN_ONEBRAIN_MERGE`-pool flip (retire `MergedSubstrate`/`MergedSubstrate2` for one
`merge_organs([surprise, worldmodel, metacog, pragmatic], wire=True)`) is now de-risked by this gate — the safe
bulk-migration proof it was waiting on. It needs its own narrowly-scoped commit + a `webapp/server.py`
brain-chat regression pass (metacog/pragmatic are default-ON in live chat), not a rider on this verification.

## Files

- `research/runners/_onebrain_twopool_merge_organread_verify.py` — the runner (reconciled descriptors +
  merged/co-resident/shipped comparison + the 5-way gate).
- `research/runners/onebrain_merge_framework.py` — 3 ADDITIVE `MergedPool` accessors
  (`meta_surprise`/`meta_worldmodel`/`worldmodel_idx_map`) so the UNMODIFIED shipped surprise/world-model organs run
  against the pool (byte-identical when unused — no existing verify constructs a surprise/world-model organ with
  `shared=<MergedPool>`).
- Artifact: `research/findings/raw/_onebrain_twopool_merge_organread_6seed.json`.

NO `sim/` edit. All runs are tiny numpy nets (N=2034) on the CPU.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._onebrain_twopool_merge_organread_verify \
    --seeds 42,43,44,100,101,102 \
    --out research/findings/raw/_onebrain_twopool_merge_organread_6seed.json
```
