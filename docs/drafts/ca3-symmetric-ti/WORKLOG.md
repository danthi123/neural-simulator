# draft-ca3-symmetric-ti — WORKLOG

## 2026-10-03 00:31 (local model) — §5 draft choice (step 1)

**Chosen:** the symmetric broad CA3-CA3 rule (Mishra 2016) as the sleep-TI successor method, with the
within-class weight-permutation control registered as its STI3 control from the start
(RULING `T3-sleep-ti`).

**Why this, now (runbook §5 order):**
- First open research next-item in CURRENT STATE: T3's 6-seed family is NO-GO for the asymmetric-STDP night
  (`2026-10-01-sleep-transitive-inference-6seed-UNDEFINED.md`); its ruling names the successor explicitly —
  "The first companion is the symmetric CA3-CA3 rule. STI3b's failure makes it the cheapest faithful step."
- The evidence already points there: STI7 (night erodes premises) fails 4/4 as registered, and STI3b fails 3/4
  (a reverse-order night also writes the inference → order is not load-bearing → co-activation-driven
  integration, the symmetric-rule regime). Prereg §10 registers this exact companion for the STI7 pattern.
- Everything else in CURRENT STATE's next-tier is merge-gated (Claude's) or a §4 cell-wait, so no section-4
  task could move; §5 rules apply.
- Serves the stage exit indirectly (a load-bearing faculty candidate) and integration-first (a mechanism
  inside the one spiking brain, no host shortcut); biology already bound:
  `research/findings/2026-07-08-riii-ca3-attractor-formation-symmetric-hebbian.md`
  (CA3 recurrent associative LTP is symmetric Hebbian; Kandel 6e Ch 54; Marr 1971 autoassociator).

**Record consults:** `before_you_build` logged 00:30 (`.git/corpus_checks_shared.jsonl`); RAG hits read:
prereg §10 (registered next methods), the UNDEFINED finding (per-seed failure table + ruling), the
symmetric-Hebbian CA3 biology finding.

**Scope guard:** PROVISIONAL draft — dev seeds only, new runner copy `research/runners/_ca3_symmetric_ti_*.py`,
never edit `sim/`, the registered STI runner, a scorer or a gate; never claims GO; gate seeds / prereg
amendment / merge stay Claude's.

## 2026-10-03 01:12 (local model) — §5 step 3: PLAN drafted, refuted, revised

- Plan drafted by MiMo Pro delegate #1 (grounded: it re-read the UNDEFINED finding, prereg §10-§11, the
  symmetric-Hebbian biology finding, the registered runner, and the engine's `hebbian_symmetric` +
  `enforce_plastic_mask_in_hebbian` configs it will flip ON via flags): `PLAN.md` v1 — 5 arms
  (wake/sym/wpermc/dis/rev), seeds 8-11 (the ruling's named dev set), 12-row outcome-to-reading table,
  pool job shape, runner-copy delta, risks, §14 failure routing.
- Refute pass by MiMo Pro delegate #2: **REFUTE-BLOCK** on ONE violation — row 2's decisive fork
  ("cospike ≈ 0" vs "healthy") was undefined, so the table was not fully fixed before runs (§5 rule 3).
  Five non-blocking nits: §10/§11 ladder ordering + uncounted rung jobs, runner name vs the ruling's
  example, §4's §9→§7 cross-ref slip, unmapped table outcomes, "byte-identical" overstatement vs the
  enumerated delta (dropped arms, strengthened gate refusal).
- All six fixes applied (u8 condition pinned to `cospike_total == 0`; qualitative words numbered; catch-all
  row 13; ladder rungs sequenced before the 8 standard jobs and counted at 11 total; name note; §12 full
  hunk enumeration).

## 2026-10-03 04:36 — re-refute round 2 (REFUTE-BLOCK) + all fixes applied

Second refute delegate: verdict **REFUTE-BLOCK**, 2 blocking violations, and it verified all six round-1 fixes
landed in the text. Both blocks fixed per the reviewer's exact prescriptions, plus all 9 nits:

1. §7 table still not fully fixed before runs (§5 rule 3): rows 4/5/6 had unpinned predicates ("foil/other
   reads rise", "most |Δw| mass", "foil/C flooded") and rows 5/6 could fire together with contradictory
   readings. Fixed: row 4 foil pinned "> 0.05 above `wake`'s"; row 6 fully pinned (premises < 0.30 OR foil >
   0.05 above wake OR C ≥ 0.10 OR > 0.5 of |Δw| mass in size-1 classes) and declared to **override row 5**;
   row 5 gains "and row 6 does not fire"; row 4b dedup; row 7 gains §9's DI(sym)−DI(dis) ≥ 0.15 clause.
2. §10 violated RULING `iw2-provision` (quoted the forbidden `pool_provision.sh` from the 09-29 prereg) and
   `pool_queue.sh add` without `--checked`. Fixed: server touch-flow
   (`touch $P/research/queue/provision.req/$(... rev-parse research/draft-ca3-symmetric-ti)`, poll
   `/opt/host-status/sim-provision/<sha>.json`, refusal → NEEDS-CLAUDE), branch/worktree named
   (`research/draft-ca3-symmetric-ti` @ `$WT/draft-ca3-symmetric-ti`), every queue line gains the mandatory
   `--checked '<task>: <why> ; mem_gb=2 ; max_age_h=48'`.
Nits fixed: §12 hunk list enumerates `sleep`→`sym` + the --merge record-assembly mode; "9–11 jobs total"
wording; core-hours scoped to the 8 standard jobs; §2 numbers cited to dev raw records (0.591→0.433 /
0.52→0.41, `cupy_seed8.json`); §14 "healthy" → `cospike_total ≥ 1`; §11 absolute-vs-above-wake bar
disambiguated + ladder-exhaustion reworded as an §11 rung-selection outcome (not a §7 seed reading); row 2's
ambiguous WAD gloss dropped; §10 now names the branch/worktree.

**A third refute round (REFUTE-APPROVE) runs next wake** (2-delegate/wake budget spent this wake: draft was
round 1's legacy, this was round 2). No build, no runs, no queue until REFUTE-APPROVE.
- State: no runs exist yet (outcome table genuinely precedes all runs); draft folder uncommitted → committing
  choice + PLAN + this WORKLOG now through the gates.
