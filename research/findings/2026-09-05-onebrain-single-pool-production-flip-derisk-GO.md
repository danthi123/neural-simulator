---
type: finding
status: live
lane: onebrain-merge
date: 2026-09-05
mechanism: onebrain-single-pool-flip
---

# One-brain SINGLE-POOL production flip — DE-RISKED (GO) on CURRENT main: the 4 core cortical organs on ONE `merge_organs` pool preserve every live chat answer (6-seed), and the 4 newly-flipped faculties are provably untouched

**Verdict: GO to flip `BRAIN_ONEBRAIN_SINGLE_POOL` default-ON** — retire the two production pools
`MergedSubstrate`/`MergedSubstrate2` for one `merge_organs([surprise, worldmodel, metacog, pragmatic], wire=True)`
pool in the live brain-chat. **The flip is left DEFAULT-OFF in this commit (leave-ready + report):** the
production default flip is high-stakes, so the controller flips `_SINGLE_POOL_DEFAULT_ON` after review. This
finding is the `webapp/server.py` brain-chat regression the wiring commit (`db5a38745`) named as its gate
("finding follows the soak").

## Premise-check against CURRENT main (the map had moved since the two isolation de-risks)

The task framed this as building the flip up from the two isolation de-risks
(`2026-08-27-...-substrate-byte-identity-...`, `2026-09-02-onebrain-twopool-merge-organ-read-GO.md`). Verifying
against `main` (b94ab45ee) FIRST found the flip is already further along than that framing:

- **The flip is already WIRED and ready, DEFAULT-OFF.** `research/runners/onebrain_single_pool_production.py`
  (`single_pool_enabled()` gated on `BRAIN_ONEBRAIN_SINGLE_POOL`, `_SINGLE_POOL_DEFAULT_ON = False`;
  `get_single_pool(seed)` builds ONE `merge_organs(_recon_descriptors(), seed, wire=True)` from the EXACT
  organ-read-GO reconciled descriptors) landed in `db5a38745`. All 4 core organs' `get_organ()` already resolve
  `shared = get_single_pool(seed)` when the flag is ON, else the two-pool path — verified in all four:
  `surprise_production_organ.py:372-378`, `worldmodel_production_organ.py:258-263`,
  `metacog_production_organ.py:480-485`, `pragmatic_production_organ.py:243-248`. The single-pool layer WINS over
  the pairwise `BRAIN_ONEBRAIN_MERGE`/`_MERGE2` flags.
- **A brain-chat answer-preservation regression harness already exists**
  (`_onebrain_single_pool_flip_regression.py`) and had returned 6/6 GO — but on `git_sha f10763172`
  (2026-09-02), which **predates tonight's 4 production-default flips** AND several framework/organ changes since
  (`2ca5370e8` metacog de-risk, `3ed20b24a` a cupy framework snapshot fix, `ff6166369`/`8c3c34951` merge waves).
  So the prior GO artifact is **not a valid re-baseline for current main** — exactly the re-baseline the task
  flagged. This finding RE-RUNS it on b94ab45ee (the 4-flips-on main).
- **Naming:** the flip could not reuse `BRAIN_ONEBRAIN_MERGE` (already pool #1's pairwise flag on `main`); the
  distinct `BRAIN_ONEBRAIN_SINGLE_POOL` is layered above the two pairwise flags.

## Gate 1 — the 4 CORE organs' live chat answers are preserved on CURRENT main (6-seed)

`_onebrain_single_pool_flip_regression.py` runs each core organ via its REAL `get_organ()` singleton (subprocess-
isolated, flag ON vs the two-pool default), reading the LIVE chat battery each organ's chat handler calls:
surprise `judge` (surprised bool over confirm/contradict/novel), world-model `expectation`/`read_surprise`
(pred-sign + violated/expected surprised bits), metacog `judge` (confident bool over evidence 0.1/0.5/0.9),
pragmatic `interpret` (enriched scalar-implicature over some/all/none).

| backend | git_sha | seeds | result |
|---|---|---|---|
| numpy | b94ab45ee (CURRENT main, this finding) | 42,43,44,100,101,102 | **6/6 GO — all 4 organs answer-preserved every seed** |
| cupy | f10763172 (prior, stale SHA) | 42,43,44,100,101,102 | 6/6 GO (all 4 organs answer-preserved) |
| cupy | b94ab45ee (CURRENT main, backend-matched) | 42,43,44,100,101,102 | queued on gpu_queue.sh (backend-matched confirmation, pending) |

All 4 organs answer-preserved, every seed, on current main. This is the load-bearing gate: metacog + pragmatic
are default-ON in live chat, so they are the load-bearing targets, and both preserve. Artifacts (per seed):
`research/findings/raw/_sp_core_rebaseline_s*.json` (each `status: GO`, prov `git_sha b94ab45ee`).

## Gate 2 — the 4 newly-flipped faculties (2026-09-05) are NOT regressed by the flip

The flip must re-baseline against CURRENT main with the 4 overnight production-default flips ON (shared-salience,
value-choice, appraisal-interoception, GNW-stop). Two lines of evidence:

**Architectural independence (decisive).** `BRAIN_ONEBRAIN_SINGLE_POOL` is read by exactly ONE function
(`single_pool_enabled`), consumed by exactly ONE site each — the 4 CORE organs' `get_organ()`. None of the 4
flipped-faculty production modules (`shared_salience_afferent.py`, `value_choice_production_organ.py`,
`affect_production_organ.py`, `webapp/gnw_global_stop.py`) reference the flag, `single_pool_*`, `merge_organs`,
the `MergedSubstrate*` pools, or import any of the 4 core organs (grep-verified, zero hits). The flip cannot alter
a flipped faculty's code path. The one indirect channel — divergent global-RNG advancement from the ON-vs-OFF
core-organ builds — is bounded: affect/appraisal builds BEFORE the flag point in the startup sequence; gnw-stop
snapshots+restores the host `random` state around its spiking read (`gnw_global_stop.py:255-262`); the organs seed
their substrates from `cfg.seed`.

**Empirical byte-identity (`_onebrain_single_pool_flipped_faculty_independence.py`, 6-seed).** Per seed,
subprocess-isolated ON vs OFF, it builds all 4 CORE organs (reproducing the live startup order), then reads the
flipped faculties that build after them: shared-salience `read_salience`, appraisal-interoception
`read_differential`, and the value-choice substrate (hashed via the SAME `VT.build_merged` the organ builds). The
NON-VACUITY WITNESS is a hash of the FULL shared pool the core organs built — ON = the single `merge_organs` pool
(N=2034), OFF = MergedSubstrate #1 (N=6064, incl. its composer/cleanup regions) — which MUST differ, proving the two workers genuinely took different
core-build paths. **Result: 6/6 GO — all 3 empirically-tested faculties (shared-salience read,
appraisal-interoception differential, value-choice substrate hash) byte-identical ON vs OFF every seed, while the
non-vacuity pool witness diverged every seed (ON = single pool N=2034 vs OFF = MergedSubstrate #1 N=6064).** The
global-RNG state does NOT diverge ON vs OFF (recorded as
informational): the organs re-seed the global RNG from `cfg.seed`, so even the RNG channel to downstream faculties
is closed — the flipped faculties are insulated from the flip regardless. GNW-stop is covered architecturally
(host-random snapshot/restore + chat-state verdict; zero merge references; needs a live ChatBrain to exercise).

## Byte-identity where expected vs the live path (honest distinction)

- **In ISOLATION (the migration gate, already 6/6 GO):** the organ-read verify
  (`2026-09-02-onebrain-twopool-merge-organ-read-GO.md`) proved each organ reads BYTE-IDENTICALLY off the single
  `merge_organs` pool vs both co-resident-alone AND the shipped 2 production pools — under a harness full-snapshot
  read-isolation protocol (`max delta 0.00e+00`, all 4 organs, 6 seeds).
- **In the LIVE path (this flip):** the gate is ANSWER-PRESERVATION, not raw numeric byte-identity, because the
  live organs use their OWN per-call read isolation (not the harness full-restore), and a genuine shared pool
  shifts the numeric Hz/margin debug fields while preserving every classification — the characterized, expected
  cost of a shared pool, already documented for the pool-1 flip
  (`2026-08-13-onebrain-production-default-flip-SCOPED.md`). Answer-preservation is therefore the correct live gate,
  and it holds 6/6.

## Honest scope — MIGRATION flip, not the one-brain INTEGRATION goal

The single pool has ZERO cross-organ synapses by construction (the `merge_organs(..., wire=True)` block-diagonal +
gain-0 freezes + per-region masks are host scaffold that deliberately forbid the cross-region interaction that IS
the one-brain integration goal). This flip MIGRATES the 4 core organs onto one substrate byte-safely; it does not
yet INTEGRATE them. Per `docs/TERMS.md`, this is the production flip that "closes" the migration rung; the
cross-edge integration (`2026-09-02-onebrain-crossregion-integration-DESIGN-ranked-crossedges.md`) is the separate
next rung. Functional read-outs only; no phenomenal claim. NO `sim/` edit; all state in `research/runners/`; the
pools are the tiny (N=2034) nets the organ-read GO validated.

## The flip (left ready; controller applies after review)

Set `_SINGLE_POOL_DEFAULT_ON = True` in `research/runners/onebrain_single_pool_production.py` (or ship
`BRAIN_ONEBRAIN_SINGLE_POOL=1`), with `BRAIN_ONEBRAIN_SINGLE_POOL=0` the byte-identical escape to the two-pool
path. Recommended: pair the flip with the backend-matched cupy 6-seed re-baseline (queued) landing GO.

## Reproduce

```bash
# Gate 1 (core organs) — per seed, parallelized across 42,43,44,100,101,102; numpy; current main:
SIM_BACKEND=numpy python -m research.runners._onebrain_single_pool_flip_regression \
    --seeds 42 --out research/findings/raw/_sp_core_rebaseline_s42.json     # repeat per seed
# Gate 1 backend-matched (cupy) 6-seed — queued on gpu_queue.sh (SIM_BACKEND=cupy, same runner; artifact pending):
SIM_BACKEND=cupy python -m research.runners._onebrain_single_pool_flip_regression --seeds 42,43,44,100,101,102
# Gate 2 (flipped-faculty independence) — per seed, parallelized:
SIM_BACKEND=numpy python -m research.runners._onebrain_single_pool_flipped_faculty_independence \
    --seeds 42 --out research/findings/raw/_ff_s42.json                     # repeat per seed
```

## Files
- `research/runners/_onebrain_single_pool_flipped_faculty_independence.py` — NEW: the flipped-faculty independence verifier (this finding).
- `research/runners/_onebrain_single_pool_flip_regression.py` — the core-organ answer-preservation harness (pre-existing; re-baselined here).
- `research/runners/onebrain_single_pool_production.py` — the flip wiring (pre-existing, default-OFF).
- Artifacts: `research/findings/raw/_sp_core_rebaseline_s*.json` (Gate 1, 6 per-seed), `research/findings/raw/_ff_s*.json` (Gate 2, 6 per-seed); backend-matched cupy Gate 1 queued on gpu_queue.sh (pending).
