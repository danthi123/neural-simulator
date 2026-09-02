---
status: live
type: finding
lane: onebrain-integration
date: 2026-09-02
mechanism: onebrain-wave1-single-pool-extension
verdict: SMOKE-GO (3-seed numpy organ-read rung) — PARTIAL pending the 6-seed cupy verify + production wire-in
---

# One-brain integration program, Phase 3 Wave 1 — comprehension + source_provenance folded onto the single pool: 3-seed numpy SMOKE-GO, a genuine NEW Hebbian rule-shape seam found + fixed

**This is the organ-read MIGRATION-SAFETY rung (3 seeds, numpy CPU), not a production flip.** It extends the
shipped 4-organ single pool (`onebrain_single_pool_production.get_single_pool`: surprise + world-model + metacog
+ pragmatic, `BRAIN_ONEBRAIN_SINGLE_POOL`, default-OFF, its own 6-seed cupy soak still queued) with comprehension
+ source_provenance onto ONE shared `merge_organs` pool — Wave 1 of
[`docs/plans/2026-09-02-onebrain-integration-program.md`](plans/2026-09-02-onebrain-integration-program.md) §Phase 3,
named there as "the true next step." The 6-seed cupy verify is QUEUED separately (guarded, skip-if-runner-absent);
this finding is PARTIAL until that lands, and the module is NOT wired into any live `get_organ()` — that is a
deliberately deferred, separate next rung (see Scope below).

## Already-built check (verify-first, done before writing a line of code)

Confirmed comprehension + source_provenance are NOT already merged onto the single pool:
`docs/PRODUCTION_INTEGRATION_LEDGER.yaml` has no `single_pool` reference; `grep -l "single_pool"
research/findings/2026-09*.md` returns none; the RAG (`tools/rag/rag_search.py "comprehension source_provenance
single pool merge wave1"`) surfaces only the EARLIER, DIFFERENT batch (2026-08-27's GROUP_A organ-read-GO, a
hebbian-OFF family — see "The genuinely new finding" below for why that batch does not already cover this). The
integration-program plan doc's own "Current connectome" section states the single pool carries only 4 organs
today and explicitly names Wave 1 as the next unbuilt step.

## What was built (additive, default-OFF, reuses the merge_organs pattern)

1. **`research/runners/_onebrain_wave1_organread_verify.py`** — the organ-read verify. `_wave1_descriptors()`
   takes the single pool's OWN reconciled 4-organ family
   (`_onebrain_twopool_merge_organread_verify._recon_descriptors`, imported not re-derived — zero drift from the
   shipped pool) and adds comprehension + source_provenance from the framework's existing GROUP_A registry
   (`onebrain_merge_framework.REGISTRY`), reconciled the SAME way metacog/pragmatic already are: pop the
   conflicting `enable_hebbian_learning` (comprehension/source_provenance declare False; the pool needs True for
   surprise/world-model's live Hebbian read) and gain-0 FREEZE every one of the organ's own regions' internal
   edges (`freeze_regions`), so the pool's live Hebbian training can never perturb comprehension's installed
   cue→role validities or source_provenance's trained opponent trace. Every OTHER config key
   comprehension/source_provenance declare was checked by inspection before running anything and already agreed
   with the pool's union (`enable_nmda=True` matches metacog's own value; the plasticity/noise-off family matches
   on both sides) — confirming the plan's own "no new seam" characterization of Wave 1 for the STANDARD
   reconciliation pattern (see below for the one seam that was NOT standard).
2. **`research/runners/onebrain_wave1_pool_production.py`** — the additive accessor: `BRAIN_ONEBRAIN_WAVE1_POOL`
   (default-OFF) + `get_wave1_pool(seed)`, mirroring `onebrain_single_pool_production.py`'s exact structure
   (memoized-by-seed, reuse-by-import of `_wave1_descriptors`).
3. **A genuine bug fix in `onebrain_merge_framework.py::_SourceProvReadOrgan`** (see "The genuinely new finding").

## Seams handled

- **hebbian (per-synapse gain-0 freeze):** comprehension + source_provenance's own regions frozen, matching
  metacog/pragmatic's existing pattern exactly. Verified via the gain-0-freeze check (scoped to metacog/pragmatic
  only — see "A verify-runner bug I found and fixed in my own code" below for why comprehension/source_provenance
  are correctly EXCLUDED from that specific check).
- **param-het (name-keyed per-region mask):** source_provenance already declares `param_het=True` on its
  registered descriptor; the engine's existing name-keyed per-region masking handled it with no new code.
- **hebbian_max_weight / enable_nmda / homeostasis / noise flags:** all already agreed between
  comprehension/source_provenance's own config and the single pool's union — confirmed by inspection AND by an
  incremental bisection (see below) that individually and jointly ruled every one of them OUT as the cause of the
  one real seam that DID surface.
- **read-isolation:** both organs' `_guard()` already dispatches to `pool.read_isolation(key)` when `shared` is
  set (comprehension's existing production code; source_provenance's existing framework read organ) — no new
  mechanism needed.

## The genuinely NEW seam (not in the taxonomy) — found, bisected, fixed

<!--derived-->
The numbers in this section (through the end of "Why this is a genuinely new seam") come from ad-hoc scratch
diagnostic scripts run during investigation, not from a saved/cited artifact — reported narratively as the
reasoning trail that led to the fix, not as claims requiring their own artifact citation. The GATING numbers this
finding stands on are the cited 3-seed `organread_3seed_smoke.json` table further below.

The first full 6-organ smoke (seed 42) built cleanly and passed gate (a) [co-residence invariance] for all 6
organs, but source_provenance's read was DEGENERATE even alone under the wave1 superset config (acc 1.0→0.5,
`min_d_true` 0.89→0.0, all 8 battery items misread "perceived" regardless of true label) — proving this was NOT a
co-residence effect (gate (a) says merged == alone-on-superset, both equally broken) but a property of the
SUPERSET CONFIG itself.

**Bisection** (each step a small, cheap numpy build; ruled out individually AND pairwise/jointly before finding
the cause):
1. Source_provenance alone with ONLY its own 8 config keys (no pool1/metacog extras) through the SAME
   `merge_organs` engine seams → discrimination WORKS (acc 1.0, `min_d_true` 0.75). Confirms the merge engine's
   4 FP-determinism seams (`deterministic_transpose_matvec` / `dedup_synapse_masks` / `per_region_inhibitory_seed`
   / `per_region_threshold_heterogeneity`) are NOT the cause.
2. Individually adding `enable_gabab`+params, `enable_nmda`+params, `enable_hebbian_learning`+`hebbian_max_weight
   =45`, `per_region_homeostasis_isolation` → each alone: WORKS.
3. All 6 pairs of those 4 groups → each pair: WORKS.
4. All 4 groups together (with and without `stdp_w_max`) → still WORKS. (Ruled out the entire hypothesis space of
   "a single group or small combination of groups.")
5. Diffed the ACTUAL produced `CoreSimConfig` between step-4's manual union and the real
   `config_descriptors=full_wave1_family` build → found 10 real diffs, the load-bearing ones being 4 Hebbian
   **rule-shape** keys the single pool's `_POOL1_CONFIG` sets non-default (`hebbian_rate_window=True`,
   `hebbian_coactivity_decay=0.85`, `hebbian_coactivity_thresh=0.20`, `hebbian_mean_subtract=1.0` — a
   covariance-style Hebbian rule surprise/world-model need) that my manual bundles never included.
6. Confirmed: patching `_SourceProvReadOrgan`'s encode window to explicitly restore those 4 keys to
   `CoreSimConfig()`'s canonical defaults (what source_provenance's own standalone implicitly relies on, since it
   never sets them) restores full discrimination (acc 1.0, `min_d_true` 0.746).

**Why this is a genuinely new seam, not an instance of the existing taxonomy.** The existing "hebbian" seam entry
is about WHICH edges may learn (the gain-0 freeze). This is about the FUNCTIONAL FORM of the update rule DURING
the one window where source_provenance's OWN edges are deliberately un-frozen (its build-time encode).
`_SourceProvReadOrgan.ensure_built()` already saved+restored 6 Hebbian VALUE hyperparameters around that window
(`hebbian_learning_rate`/`hebbian_max_weight`/`hebbian_min_weight`/`hebbian_weight_decay`/`hebbian_symmetric`/
`enable_hebbian_learning`) — it simply never needed to cover the 4 rule-shape keys before, because every
CO-RESIDENT organ in the ORIGINAL 2026-08-27 GROUP_A batch (self_schema/d6/comprehension/causal_whatif) also left
those 4 at `CoreSimConfig` defaults, so there was nothing to leak. The single pool's `_POOL1_CONFIG` is the FIRST
descriptor in this codebase to set them non-default globally, which is what newly exposed the incompleteness.

**The fix** (`onebrain_merge_framework.py::_SourceProvReadOrgan.ensure_built()`, `git diff` is ~20 lines, entirely
inside the existing `if self._shared is not None:` branch): read a FRESH, unconfigured `CoreSimConfig()` (not a
hardcoded copy that could drift from the class default) and save+set+restore the same 4 rule-shape keys around the
encode window, exactly like the 6 keys already handled. Byte-identical for every EXISTING caller: the fix is a
no-op whenever the pool's rule-shape config already matches `CoreSimConfig` defaults (true for the ORIGINAL
GROUP_A batch — its own prior 6-seed organ-read GO is unaffected), and `_SourceProvReadOrgan` has zero production
usage (the shipped `source_provenance_production_organ.SourceProvenanceHonestyMonitor` is a different class with
no `shared=` support at all — this fix cannot touch live chat). Logged in
[`research/FAILURE_LOG.md`](../FAILURE_LOG.md) (2026-09-02 row): GATED for regression by this finding's own verify
runner; NOT-GATEABLE yet as a general class (no mechanical check exists that a build-time-encode organ's
saved/restored hyperparameter list is complete relative to every field a co-resident descriptor might override).

## A verify-runner bug I found and fixed in my own code (before the seam above)

Two flaws in the FIRST draft of `_onebrain_wave1_organread_verify.py`, both fixed before trusting any result:
1. `_recon_descriptors()`'s `surprise`/`worldmodel` rows never set `answer_fn` on the descriptor (the twopool
   verify dispatches answers through its own separate `_READ_FNS` table instead) — a `TypeError` the very first
   run caught immediately. Fixed by back-filling those two rows with the twopool verify's own
   `_surprise_answer`/`_worldmodel_answer` (reuse-by-import, not new code).
2. The gain-0-freeze check initially included comprehension/source_provenance's OWN regions, which flagged a
   FALSE "gain0=False" — those two organs install their own weights AT CONSTRUCTION (comprehension's cue→role
   validities; source_provenance's build-time encode), so their edges are SUPPOSED to change once between the
   pool-build snapshot and the read; that is not a freeze violation. Fixed by scoping the check to
   metacog/pragmatic only (`_POOL2_FREEZE`, matching the ORIGINAL twopool verify's own proven check); the property
   that DOES matter for comprehension/source_provenance — no LIVE Hebbian drift once their own weights are
   installed — is what gate (a) already proves (co-residence invariance is impossible if a co-resident organ's
   Hebbian step had leaked into their edges).

## Organ-read parity (numpy, seeds 42/43/44, 3/3 GO on every gate)

Raw artifact: `research/findings/raw/_onebrain_wave1/organread_3seed_smoke.json`.

| organ | (a) co-residence byte-identical | (b) faculty-alive | (c) answer-preservation | shipped-read continuous delta |
|---|---|---|---|---|
| surprise | 3/3 | 3/3 | 3/3 (vs `get_single_pool`) | 0.0 |
| worldmodel | 3/3 | 3/3 | 3/3 (vs `get_single_pool`) | 0.0 |
| metacog | 3/3 | 3/3 | 3/3 (vs `get_single_pool`) | 0.0 |
| pragmatic | 3/3 | 3/3 | 3/3 (vs `get_single_pool`) | 0.0 |
| comprehension | 3/3 | 3/3 | 3/3 (categorical answer) | 0.13–0.15 (informational, see below) |
| source_provenance | 3/3 | 3/3 | 3/3 (categorical answer) | 0.16–0.25 (informational, see below) |

- **(a) organ-read byte-identity (co-residence invariance):** 3/3 for all six organs — the DECISIVE
  migration-safety bar (the SAME one the base-4 single pool's own organ-read GO used): each organ's read on the
  6-organ wave1 pool is EXACTLY the read it gets alone, on the identical superset config. Adding comprehension +
  source_provenance does not perturb surprise/world-model/metacog/pragmatic's reads (`cores_d=0.0`, every seed),
  and comprehension/source_provenance's own reads are equally co-residence-invariant.
- **(b) faculty-alive:** 3/3 — every organ still produces its live, non-degenerate verdict (surprise's
  contradict/confirm separation ≥2x; world-model's sign-correct expectation + violation>expected firing; metacog's
  confidence margin grows with evidence; pragmatic's implicature margin separates the scalar family;
  comprehension's well/ill calibration means separate; source_provenance's opponent accuracy ≥0.99 with a real
  signed discriminability).
- **(c) answer-preservation, STRICT (surprise/world-model/metacog/pragmatic vs the ACTUAL shipped
  `get_single_pool`):** 3/3, read-byte-identical (0.0 delta) — a fair, apples-to-apples comparison since that pool
  also runs the merge engine's seams. Extending it with 2 more organs changes nothing about the original four.
- **(c) answer-preservation, comprehension/source_provenance (categorical, vs a RAW UNSEAMED standalone build):**
  3/3 — the rendered decision matches. The CONTINUOUS margin (0.13–0.25) does NOT match the raw standalone and is
  reported as INFORMATIONAL, not gating: this is the documented, pre-existing FP-determinism seam sensitivity
  (`onebrain_merge_framework._base_config`'s own comment: "a SPIKING DYNAMICS read integrated over hundreds of
  steps... AMPLIFIES a single-ULP per-step delta into a 1-spike read divergence"), never previously claimed
  byte-identical to an UNSEAMED build anywhere in this codebase — only co-residence invariance UNDER THE SAME
  seamed config (gate a) was ever validated for these two organs, and that gate is 3/3.
- **Gain-0 freeze holds** (metacog/pragmatic's ~26,300–26,450 internal edges, bit-identical before vs after the
  full train+read lifecycle): 3/3.
- **Legacy discriminator diverges** (seams OFF → merged-vs-coresident init genuinely differs, proving byte-identity
  above is not vacuous): 3/3, delta 25 on every seed.

## Byte-identical-when-off

Both new files are additive; `git diff` on every EXISTING file touches only
`onebrain_merge_framework.py::_SourceProvReadOrgan` (the bug fix above, unconditional but a no-op for every
existing caller). No production `get_organ()` dispatch (surprise/world-model/metacog/pragmatic/comprehension/
source_provenance's six files) was modified — `BRAIN_ONEBRAIN_WAVE1_POOL` currently has ZERO effect on production
regardless of its value, provably so by the absence of any reference to it outside the two new files. This is a
deliberate scope decision (see below), not an oversight.

## Scope: what Wave 1 deliberately does NOT do yet

This lands the MIGRATION-SAFETY organ-read rung only — the same rung the base-4 single pool passed BEFORE its own,
separate `get_organ()` wiring commit. Wiring the wave1 flag into the six organs' live `get_organ()` singletons
(mirroring the base-4 pattern, itself already safely applied to two DEFAULT-ON organs, metacog/pragmatic) is the
natural next rung, but is deliberately deferred here: `source_provenance_production_organ.py`'s actual production
class (`SourceProvenanceHonestyMonitor`, default-ON since 2026-09-01) is entirely different from the framework's
`_SourceProvReadOrgan` used in this rung and has no `shared=` support at all, so wiring it in is real, separate
work (adding `shared=` support to a currently-default-ON production class) that deserves its own build + review,
not a same-session addendum to a merge-logic smoke. `comprehension_production_organ.py`'s `get_organ()` also
already has an UNRELATED shared-pool mechanism (`onebrain_xedge_production.get_xedge_pool`) whose interaction with
a second shared-pool flag needs its own careful design, not a hasty bolt-on.

## Foundation dependency (unchanged from the task brief)

The single-pool 6-seed cupy soak (`BRAIN_ONEBRAIN_SINGLE_POOL`) is queued but not yet GO. This numpy smoke
validates the MERGE LOGIC (comprehension + source_provenance's own reconciliation) independent of that soak's
outcome; if the soak later fails, Wave 1 rebases on whatever fix the base-4 single pool needs — the reconciliation
this finding adds is layered on top of, and imported from, that pool's own descriptors, not a fork of them.

## Reproduce

```
SIM_BACKEND=numpy python -m research.runners._onebrain_wave1_organread_verify \
    --seeds 42,43,44 \
    --out research/findings/raw/_onebrain_wave1/organread_3seed_smoke.json
```

## Next steps

1. **Queued (this landing):** the 6-seed cupy organ-read verify, skip-guarded on this runner's presence.
2. **Separate, later:** add `shared=` support to `SourceProvenanceHonestyMonitor` (the actual default-ON
   production class) and wire the six organs' `get_organ()` singletons onto `get_wave1_pool()` behind
   `BRAIN_ONEBRAIN_WAVE1_POOL`, gated by `onebrain_regression_battery.py` through the real `webapp.server.
   brain_chat` (meaningful only once the flag is actually read somewhere) — mirroring exactly how the base-4
   single pool's own production wiring followed its organ-read GO as a distinct commit.
3. Extend the seam taxonomy (CLAUDE.md / the integration program design doc) with the Hebbian RULE-SHAPE seam
   named above, so Wave 2 (self_schema + curiosity + causal_whatif, which the program doc already flags for its
   own param-het wrinkles) checks for it up front rather than re-discovering it.
