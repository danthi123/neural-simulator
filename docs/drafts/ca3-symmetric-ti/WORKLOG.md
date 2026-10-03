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
  hunk enumeration). **Re-refute pending: next wake must run delegate #2 again and get REFUTE-APPROVE
  before any build (§4) or run.** Delegate budget: 2/2 used this wake.
- State: no runs exist yet (outcome table genuinely precedes all runs); draft folder uncommitted → committing
  choice + PLAN + this WORKLOG now through the gates.
