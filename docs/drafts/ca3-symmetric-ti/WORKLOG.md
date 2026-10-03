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

## 2026-10-03 05:38 — refute round 3: REFUTE-APPROVE ✅ + approve-round nits fixed

Third refute delegate verified all round-2 fixes landed (quoted each) and returned **REFUTE-APPROVE**: no
remaining blocking violation against non-negotiables 1–6 or §5 rules. Independently checked clean: no
sim//registered/grader/gate edits, no GO words on dev, §7/§9/§11 bars agree, rows disjoint with catch-all,
gate-seed strengthening real vs :744-745, cospike U8 pin consistent across rows 2/12 + §13/§14, engine
evidence (hebbian_symmetric config:873, bridge:1240/:10128, delta lr·(w_max−w) :10136), registered constants
match, placement follows compute-placement + iw2-provision.

Its 9 nits — the 6 actionable ones fixed now, per the reviewer's prescriptions (nits, not design changes):

1. §2's 0.591→0.433 citation was WRONG (that value is in no dev/gate record — it traces to the prereg's rounded
   q3_s7 exploration summary, PREREGISTRATION.md:301). Reverted to 0.59→0.43 with the prereg:301 citation; dev
   records cited only for direction.
2. §10's "keeps the registered backend refusal" corrected: registered refusal is gate-run-only (:748-749); the
   copy strengthens it to unconditional (§12 item 6 now says the same instead of "verbatim").
3. §12's hunk enumeration now includes the §11 DEFAULTS constants (without it, §12 forbade what §11 requires).
4. Row 5's STI3 clause gains the registered `gain > 0` guard (non-positive gain could pass it spuriously).
5. §9 STI1 line gains the registered absolute clause DI(sym) ≥ 0.15.
6. Row 4b's "premises kept" parenthetically pinned to the max(0.30, 0.9×wake) bar.
Left as reviewer judged safe: wpermc-also-infers → row 13 (never a pass), conservative NEEDS-CLAUDE
escalation on provisioning refusals, runner-name deferral (Claude confirms at review).

**Status: APPROVED for build.** Next wake: §4 build — write the runner copy
`research/runners/_ca3_symmetric_ti_symrule.py` with the enumerated §12 delta only, smoke on seed 8 (dev),
then §10's queue flow (ladder rungs first). Delegate budget: 1/2 used this wake.
- State: no runs exist yet (outcome table genuinely precedes all runs); draft folder uncommitted → committing
  choice + PLAN + this WORKLOG now through the gates.
- 2026-10-03 10:58 [ca3-symmetric-ti] §4 BUILD DONE: `research/runners/_ca3_symmetric_ti_symrule.py` (914 lines)
  written on branch `research/draft-ca3-symmetric-ti` (worktree `.claude/worktrees/draft-ca3-symmetric-ti`,
  committed **d4a533ae7** through the gates) as a copy of the registered sleep runner carrying only the §12
  delta: arms (wake/sym/wpermc/dis/rev — rev instrument-only, registered arm blocks removed); `night(hebbian=True)`
  runs STDP off + the engine's symmetric Hebbian rule + plastic mask + the §11 constants (echoed to each night
  record as `rule_echo`); `permute_dw_classes` (within-class Δw permutation + class histogram with size-1 mass
  share); `replica_sym` (same-step `lr*(w_max−w)`, clip [0,900], activity gate = `F.any | pv_any` mirroring the
  engine's all-neuron `_prev_any and _fired_any` — the PV/any trace is recorded every night step so the identity
  check is exact, not approximate); `cospike_counts` (FᵀF self-product over plastic between synapses → per-pair
  matrix + `cospike_total` for row 2's U8 read); `--merge` record assembly (U7-analogue encode-hash check FIRST,
  loud AssertionError on part disagreement, row-8 sym-vs-dis attribution computed there); gate-seed and numpy
  refusals now unconditional (§12 item 6); the sleep-vs-dis attribution block left with the arm-set reduction.
  Smokes, all on the worktree with NO brain run: py_compile rc=0; `--seed 42` refused rc=2; `SIM_BACKEND=cupy`
  refused rc=2; `--selftest` SELFTEST PASS; `--merge` good path → u7=True + attribution 0.975, mismatch path →
  AssertionError rc=1. Gates: `single-draw-null` cleared via its own prescribed waiver `# single-draw-null: ok …`
  (the seed-derived within-class permutation is the control TREATMENT and §7 row 12 forbids any single-seed
  verdict, so no row reads one draw alone); `tool-health-fresh` reported-only (`BLOCKING=False`; the auto-tune
  rot — dearpygui absent from the venv — is pre-existing on main and already a tracked repair-backlog item).
  smoke1: scipy 1.15.3 present. NEXT (§10): provisioning touch-flow for the branch, then queue the 11 jobs —
  the 3 ladder rungs (seed 8, `--set hebbian_learning_rate` ∈ {0.002, 0.006, 0.02}) FIRST, then parts a/b ×
  seeds 8–11 — `mem_gb=2`, `max_age_h=48`, `bash tools/queue_job_shape_check.sh` pre-check + `--checked`,
  full 40-char SHA. No runs exist yet.
- **§10 order 1 DONE (2026-10-03 15:56 EDT)** — provisioning touch-flow (`touch research/queue/provision.req/d4a533ae77186050e3e8bc6156e24bb81af3221b`):
  `sim-provision/<sha>.json` → **state=ok** after ~7 min ("provisioned on pool41, pool42, pool40; not on optional
  wspool (backfilled later)"; `behind_main` = docs-only 1 commit, accepted per RULING iw2-provision). Then queued
  the **3 ladder rung jobs** (PLAN §10/§11 order: rungs first) — seed 8, `--dev --arms wake sym --set
  hebbian_learning_rate=` {0.02, 0.006, 0.002} **descending**, so §11's "largest that satisfies" evaluates from the
  top as runs land; each line = the §10 template (env SIM_BACKEND=numpy + thread pins + CUDA_VISIBLE_DEVICES=),
  `queue_job_shape_check.sh` pass, then `pool_queue.sh add --checked '<task>: … ; mem_gb=2 ; max_age_h=48'` →
  **queued 3/3** (depths 195–197; `pool.queue` grep = 3). Line record: `jobs/rungs_d4a533ae7.tsv` (header + 3 rows;
  scratch copy `research/queue/ca3_rungs_d4a533ae7.tsv` untracked). NOT queued: the 8 standard jobs (4 dev seeds ×
  parts a/b) — they wait for the §11 rung freeze (non-DI criteria only; ladder exhausted without a satisfying rung
  → SPECIFICITY-FLOOD stop, §11). Next: rung outputs land → §11 selection (judgment → MiMo Pro per RULING
  models) → freeze → queue 8 standard jobs → merge+§7 reads. No numbers read yet (blind).
- **§10 rung attempt outcomes (2026-10-03 17:15 EDT)** — **1/3 landed, 2 dead-claims requeued ONCE.**
  `seed8_rung_lr0.002.json` **landed 16:39** (51 min run, `.prov.json` sidecar written; content unread — blind,
  §11 waits). `seed8_rung_lr0.02` + `seed8_rung_lr0.006` = **dead claims**: dispatched pool40 15:48:02/15:48:08,
  alive at 16:47 (pgrep), gone from pool40/41/wspool by 17:12; no output files, no exit records — and
  `sim-pool-jobs/exits.json` has **zero** ca3 rows even for the landed rung, so for these jobs file-presence +
  node liveness are the liveness signals, not exits. Same silent-vanish signature as the 14:37 event that killed
  the 4 stage1-host-share gate cells. Requeue ONCE (runbook §3 + duplicate-claim guard): both original TSV lines
  re-`add --checked`ed → "a previous identical claim exists … **no longer alive** -- queueing", `requeued=2
  failures=0` (depths 185–186; lr0.002 excluded — never requeue a landed cell). If either vanishes again without
  a file: §10 report **#4** with both attempt outcomes — no second requeue. §11 selection still ordered
  largest-first: needs lr0.02 verdict, then 0.006, then 0.002 (already in hand).
