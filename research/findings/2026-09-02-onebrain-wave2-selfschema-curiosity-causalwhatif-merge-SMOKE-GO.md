---
status: live
type: finding
lane: onebrain-integration
date: 2026-09-02
mechanism: onebrain-wave2-single-pool-extension
verdict: SMOKE-GO (3-seed numpy organ-read rung) — PARTIAL pending the 6-seed cupy verify + production wire-in
---

# One-brain integration program, Phase 3 Wave 2 — self_schema + curiosity + causal_whatif folded onto the wave-1 pool: 3-seed numpy SMOKE-GO, THREE genuine seams found (one not named in the plan doc) and fixed

**This is the organ-read MIGRATION-SAFETY rung (3 seeds, numpy CPU), not a production flip.** It extends the
shipped 6-organ Wave-1 pool (`onebrain_wave1_pool_production.get_wave1_pool`: surprise + world-model + metacog +
pragmatic + comprehension + source_provenance, `BRAIN_ONEBRAIN_WAVE1_POOL`, default-OFF, its own 6-seed cupy soak
still queued) with self_schema + curiosity + causal_whatif onto ONE shared `merge_organs` pool — Wave 2 of
[`docs/plans/2026-09-02-onebrain-integration-program.md`](plans/2026-09-02-onebrain-integration-program.md)
§Phase 3. The 6-seed cupy verify is QUEUED separately (guarded, skip-if-runner-absent); this finding is PARTIAL
until that lands, and the module is NOT wired into any live `get_organ()` — a deliberately deferred, separate
next rung (see Scope below), mirroring Wave 1's own sequencing exactly.

## Already-built check (verify-first, done before writing a line of code)

Confirmed self_schema + curiosity + causal_whatif are NOT already merged onto the single/wave-1 pool:
`docs/PRODUCTION_INTEGRATION_LEDGER.yaml` has no "wave2"/9-organ single-pool reference; `grep -l "wave2\|self_schema.*curiosity.*causal_whatif"
research/findings/2026-09*.md` returns none; the RAG (`tools/rag/rag_search.py "self_schema curiosity causal_whatif
single pool merge wave 2" 5 --corpus finding`) surfaces only the 2026-08-27 GROUP_A batch (a DIFFERENT, hebbian-OFF
frozen family that never co-resided self_schema with metacog) and the 2026-09-02 Wave-1 finding, which names Wave 2
as its own explicit "Next steps" #3. `git branch -a` / `git log --all --oneline | grep wave2` show no prior wave2
branch or commit. Confirmed NOT already built.

## What was built (additive, default-OFF, reuses the merge_organs pattern)

1. **`research/runners/_onebrain_wave2_organread_verify.py`** — the organ-read verify. `_wave2_descriptors()` takes
   the wave-1 pool's OWN reconciled 6-organ family (`_onebrain_wave1_organread_verify._wave1_descriptors`, imported
   not re-derived — zero drift from the shipped wave-1 pool) and adds self_schema + curiosity + causal_whatif from
   the framework's existing GROUP_A registry, each reconciled per the seams below.
2. **`research/runners/onebrain_wave2_pool_production.py`** — the additive accessor: `BRAIN_ONEBRAIN_WAVE2_POOL`
   (default-OFF) + `get_wave2_pool(seed)`, mirroring `onebrain_wave1_pool_production.py`'s exact structure
   (memoized-by-seed, reuse-by-import of `_wave2_descriptors`).

No existing production file was touched — see "Byte-identical-when-off" below.

## Seams handled — TWO matched the plan doc's category, ONE was found only by verifying rather than assuming

The task brief named one predicted seam (self_schema/metacog's `workspace` name collision) and asked me to VERIFY
each organ's own seams rather than assume that was the whole story. Enumerating every Wave-2 organ's region names
and wiring-dict keys against the wave-1 superset (rather than trusting the plan doc's single named prediction)
found THREE genuine seams, not one:

1. **self_schema/metacog REGION-NAME COLLISION** (the plan doc's own prediction, confirmed real). self_schema's
   `workspace`/`workspace_fs` regions are LITERALLY the same names metacog already owns in the wave-1 pool
   (`METACOG.regions == ("workspace", "workspace_fs", "meta_schema")`; `REGISTRY["self_schema"].regions ==
   ("workspace", "workspace_fs", "self_schema")`). The framework's spec-extraction `owner` dict raises
   `MergeConflict` on any duplicate region name across descriptors. **Fix:** self_schema's Wave-2 descriptor
   renames only the two colliding regions (`ss_workspace`/`ss_workspace_fs`; `self_schema` itself is untouched — no
   collision) via a generic `_renamed_spec` wrapper (`dataclasses.replace` on each `BrainRegion.name` +
   `RegionPathway.from_region`/`to_region`, preserving every other field — `enable_nmda`, `n_neurons`,
   `exc_fraction`, density, weight_mean — byte-for-byte).
2. **curiosity/surprise REGION-NAME COLLISION — NOT named in the plan doc.** curiosity's `cue` region
   (`REGISTRY["curiosity"].regions[0] == "cue"`) is literally the same name SURPRISE already owns
   (`SURPRISE.regions == ("cue", "patient_expected", "patient_asserted", "surprise")`). This is a second,
   undocumented collision the plan doc's "workspace" framing did not anticipate — found only by actually
   enumerating every organ's region-name footprint before assuming the plan doc's single named seam was complete.
   **Fix:** same mechanism, scoped to curiosity's own `cue` only (renamed `cur_cue`); surprise's `cue` is untouched,
   so the wave-1 6 organs' own reads/wiring keep referencing the original name — zero risk to anything already
   shipped.
3. **A THIRD, SILENT (non-`MergeConflict`-raising) collision, found only by checking the WIRING-KEY namespace, not
   just region names.** self_schema's `explicit_wiring_fn` and metacog's `explicit_wiring_fn` both emit dict keys
   `loop_0`/`loop_1` into the SAME `_install_organ_read_wiring` union (self_schema's K_CONTENTS=4 range(4) overlaps
   metacog's K_CLASSES=2 range(2) at k=0,1). `dict.update()` does not raise on this — it silently overwrites
   whichever descriptor's `explicit_wiring_fn` ran later in list order, corrupting the OTHER organ's attractor-loop
   wiring with no error at all. This is exactly the seam taxonomy's warning in CLAUDE.md: "a MergeConflict is NOT
   raised; the union accepts a default and the faculty dies quietly." **Fix:** self_schema's Wave-2 wiring keys are
   prefixed (`ss_loop_{k}`/`ss_member{k}_to_attend`), disjoint from metacog's `loop_{k}` by construction. (This
   seam would have produced a silently-corrupted metacog attractor loop that still passed a naive smoke test —
   the kind of failure the gate (a) co-residence check exists to catch, but only because gate (a) also re-checks
   metacog's OWN read on every wave-2 build, not just the three new organs'.)

**hebbian (per-synapse gain-0 freeze):** self_schema + causal_whatif's own configs declare `enable_hebbian_learning:
False`, conflicting with the pool's global `True` (surprise/world-model's live Hebbian read needs it) — reconciled
exactly like Wave 1's comprehension/source_provenance: pop the key (pool's True wins) + gain-0 freeze every one of
the organ's own regions' internal edges. curiosity declares no `config` at all (nothing to pop), but its own
regions are frozen too anyway for hygiene (not load-bearing for its read — the read only ever touches `ask`, which
has no afferents of its own in this circuit).

**param-het (name-keyed per-region mask):** self_schema and curiosity both already declare `param_het=True` on
their registered descriptors; the engine's existing per-region masking (already proven compatible with
metacog/pragmatic's OWN param_het=True path by Wave 1) handled it with no new code — confirmed by the 3/3 GO on
gate (a), since a mis-masked heterogeneity draw would show up as a co-residence delta.

**Hebbian RULE-SHAPE seam (Wave 1's own new-seam finding — checked for applicability, found NOT APPLICABLE.)**
Wave 1 found source_provenance's build-time Hebbian ENCODE window needed 4 additional saved/restored rule-shape
keys because it runs a live Hebbian update during its own construction. None of Wave 2's three organs run a live
HEBBIAN update at their own construction: self_schema's loop is a fixed `explicit_wiring_fn` weight, never trained
by any rule; causal_whatif's build-time train uses STDP+DA (a different mechanism, its own local
`stdp_a_plus`/`tau`/`w_max`, entirely disjoint from the Hebbian rule-shape keys); curiosity has no build-time
encode at all. Confirmed empirically, not just by inspection: gate (a) shows zero co-residence delta for all
three, which would not hold if a silently-leaked rule-shape key were corrupting a read.

## The "400 > 45" attractor-weight concern — VERIFIED, not assumed, to be a non-issue for this organ family

The plan doc flagged "the 400>45 attractor-weight survival" as a Wave-2 risk. Checked directly rather than
assumed: self_schema's actual installed weights are `LOOP_W=30.0` (`DEFAULT_ATTRACTOR_WEIGHT`, the SAME constant
metacog's own loop already uses inside the wave-1 pool) and `MEMBER_TO_ATTEND_W=12.0` — both already well under
the pool's `hebbian_max_weight=45.0` ceiling (from `_POOL1_CONFIG`, unchanged by Wave 2). causal_whatif's
build-time STDP+DA train caps its own xblock weights at a LOCAL `stdp_w_max=24.0` (restored after training) — also
under 45. The literal "400" in this codebase
(`_self_schema_region_derisk.build_self_schema_bridge`'s own `cfg.hebbian_max_weight = max(400.0,
attractor_weight*4.0)`, and the analogous 2026-08-27 pool-#2 `_POOL2_METACOG_CONFIG.stdp_w_max`) is a generous
STANDALONE safety margin (4x headroom above the 30-weight loop, floored at 400) — not evidence any real synapse in
this family is installed near that value, so no clip ever fires either way. Checked directly in `sim/bridge.py`:
the Hebbian clip (~L10013-10023) and the reward/homeostatic clips (~L10822-10834, ~L11157-11170) are ALL already
GATED by `cp_plasticity_rate_gain` (a 2026-07-31 fix, predates this rung) — a frozen (gain-0) region's weights are
excluded from every clip regardless of the pool's global ceiling, so the freeze above is sufficient protection
independent of the 45-vs-400 comparison.

## A self-caught bug: the gain-0 freeze check must EXCLUDE a build-time-plasticity-then-frozen organ's own region

The first 3-seed run (seed 42 alone, then re-run at seeds 42/43/44) FAILED the gain-0 freeze precondition
(`gain0=False`, measured=0/1) despite gates (a)/(b)/(c) and the legacy discriminator all passing. Root cause: my
initial `_WAVE2_FREEZE` verification tuple included `"evt"` (causal_whatif's own region) in the before/after
weight-invariance check — but `_CausalReadOrgan.ensure_built()` runs a genuine BUILD-TIME STDP+DA train of its OWN
evt slice as part of its normal operation (the trained xblock weights ARE the organ's job, exactly the same
category as source_provenance's build-time Hebbian encode in Wave 1). Wave 1's own `_frozen_edge_weights` check
already documents excluding comprehension/source_provenance from this exact check for this exact reason ("both
organs INSTALL their own weights AT CONSTRUCTION ... that is not a freeze violation, it is the organ doing its
job") — I had re-derived the mistake Wave 1 already fixed once, on a new organ. **Fix:** removed `"evt"` from the
verification scope (`_WAVE2_FREEZE`) while leaving `freeze_regions=("evt",)` on the descriptor unchanged (still
required at pool-build time and for causal_whatif's own local freeze-then-restore baseline, protecting evt from
every OTHER organ's ongoing Hebbian both before its own training and after it completes). self_schema + curiosity
install nothing at their own construction, so both correctly stay in the verification scope. After the fix, all 3
seeds pass gain-0 freeze cleanly (measured=3/3, `gain0_freeze_delta=0.0` every seed).

## Organ-read parity (numpy, seeds 42/43/44, 3/3 GO on every gate)

Raw artifact: `research/findings/raw/_onebrain_wave2/organread_3seed_smoke.json`.

| organ | (a) co-residence byte-identical | (b) faculty-alive | (c) answer-preservation | shipped-read continuous delta |
|---|---|---|---|---|
| surprise | 3/3 | 3/3 | 3/3 (vs `get_wave1_pool`) | 0.0 |
| worldmodel | 3/3 | 3/3 | 3/3 (vs `get_wave1_pool`) | 0.0 |
| metacog | 3/3 | 3/3 | 3/3 (vs `get_wave1_pool`) | 0.0 |
| pragmatic | 3/3 | 3/3 | 3/3 (vs `get_wave1_pool`) | 0.0 |
| comprehension | 3/3 | 3/3 | 3/3 (vs `get_wave1_pool`) | 0.0 |
| source_provenance | 3/3 | 3/3 | 3/3 (vs `get_wave1_pool`) | 0.0 |
| self_schema | 3/3 | 3/3 | 3/3 (categorical answer) | 0.0015–0.0040 <!--derived--> (informational, see below) |
| curiosity | 3/3 | 3/3 | 3/3 (categorical answer) | 113.7–126.9 <!--derived--> (informational, see below) |
| causal_whatif | 3/3 | 3/3 | 3/3 (categorical answer) | 0.0 |

- **(a) organ-read byte-identity (co-residence invariance):** 3/3 for all nine organs — the DECISIVE
  migration-safety bar (the same one Wave 1's own 6-organ GO used, extended to 9): each organ's read on the wave-2
  pool is EXACTLY the read it gets alone, on the identical superset config. Adding self_schema + curiosity +
  causal_whatif does not perturb the 6 wave-1-carried organs' reads (`cores_d=0.0`, every seed), and the 3 new
  organs' own reads are equally co-residence-invariant, confirming the seam-1/2/3 fixes fully resolved the
  region-name and wiring-key collisions (a residual collision would show up here as a nonzero `cores_d` or a
  degenerate faculty-alive check, not a silent pass).
- **(b) faculty-alive:** 3/3 — every organ still produces its live, non-degenerate verdict, including the 3 new
  ones: self_schema's `author_rate_self > author_rate_heard` (self-vs-heard authorship separation), curiosity's
  `want_novel_hz > want_familiar_hz` (the ASK-pool crave drive genuinely tracks the epistemic gap), causal_whatif's
  `fwd_acc >= 0.5` plus `directed_fwd_BtoD > directed_rev_DtoB` (the trained forward model predicts correctly and
  the directedness asymmetry holds).
- **(c) answer-preservation, STRICT (the 6 wave-1-carried organs vs the ACTUAL SHIPPED `get_wave1_pool`):** 3/3,
  read-byte-identical (0.0 delta) — a fair, apples-to-apples comparison since that pool also runs the merge
  engine's seams. Extending it with 3 more organs changes nothing about the six it originally carried.
- **(c) answer-preservation, self_schema/curiosity/causal_whatif (categorical, vs a RAW UNSEAMED standalone
  build):** 3/3 — the rendered decision matches every seed.
  <!--derived--> The CONTINUOUS margin (self_schema 0.0015–0.0040; curiosity 113.7–126.9 Hz; causal_whatif exactly 0.0) does NOT match a raw standalone build byte-for-byte and is
  reported as INFORMATIONAL, not gating — this is the SAME pre-existing, already-documented FP-determinism seam
  sensitivity Wave 1 named for comprehension/source_provenance (a spiking-dynamics read integrated over hundreds
  of steps amplifies a single-ULP per-step delta into a 1-spike read divergence), never previously claimed
  byte-identical to an unseamed build anywhere in this codebase for a live/OU-driven read. curiosity's larger
  absolute delta (Hz units, not a normalized margin) reflects that its read averages a spike COUNT over
  `_CUR_READ_REPS=4` repetitions of a `W_WANT`-step window — more accumulated per-step FP variance than
  self_schema's frozen single-pass read, consistent with (not contradicting) the seam's own documented mechanism.
- **Gain-0 freeze holds** (self_schema/curiosity's internal edges, `n_frozen_edges` ranging 64,592–64,802
  <!--derived--> across seeds — the exact count depends on the per-region-seamed wiring plan's density draw —
  bit-identical before vs after the full 9-organ train+read lifecycle): 3/3, `gain0_freeze_delta=0.0` every seed.
  causal_whatif's `evt` is deliberately excluded from this specific check (see above) but remains protected by
  `freeze_regions=("evt",)` at build time and by its own local freeze-then-restore.
- **Legacy discriminator** (seams OFF → merged-vs-coresident init diverges, delta~24.6-24.8 <!--derived--> every
  seed): 3/3, proving the byte-identity above is not vacuous.

## Byte-identical-when-off

Both new files are additive; `git diff --stat` on tracked files shows only `research/findings/raw/_provenance/runs.jsonl`
(the automatic per-run provenance ledger, `research/runners/__init__.py`'s existing mechanism — not a production
file) touched. No production `get_organ()` dispatch (surprise/world-model/metacog/pragmatic/comprehension/
source_provenance/self_schema/curiosity/causal_whatif's nine files) was modified, and neither was
`onebrain_merge_framework.py` or `onebrain_wave1_pool_production.py` — `BRAIN_ONEBRAIN_WAVE2_POOL` currently has
ZERO effect on production regardless of its value, provably so by the absence of any reference to it outside the
two new files.

## Scope: what Wave 2 deliberately does NOT do yet

This lands the MIGRATION-SAFETY organ-read rung only — the same rung the wave-1 pool passed BEFORE its own,
separate `get_organ()` wiring commit (still not landed either). Wiring the wave2 flag into the nine organs' live
`get_organ()` singletons is the natural next rung, deliberately deferred here for the SAME reasons Wave 1 named:
`source_provenance_production_organ.py`'s actual production class has no `shared=` support at all, and
`comprehension_production_organ.py`'s `get_organ()` already has an unrelated shared-pool mechanism whose
interaction with a THIRD shared-pool flag needs its own careful design. `self_schema_production_organ.py`,
`curiosity_production_organ.py`, and `causal_whatif_production_organ.py`'s own `get_organ()` singletons have the
same open question and are equally out of scope here.

Also out of scope, unchanged from Wave 1: cross-region interaction (the one-brain INTEGRATION goal itself) — zero
cross-region synapses are added by this rung; a pool with no cross-edges is MIGRATED, not INTEGRATED.

## Foundation dependency (unchanged from the task brief)

The wave-1 pool's own 6-seed cupy soak (`BRAIN_ONEBRAIN_WAVE1_POOL`) is queued but not yet GO. This numpy smoke
validates the Wave-2 MERGE LOGIC (self_schema/curiosity/causal_whatif's own reconciliation) independent of that
soak's outcome; if the soak later needs a fix, Wave 2 rebases on whatever the wave-1 pool needs — the
reconciliation this finding adds is layered on top of, and imported from, that pool's own descriptors, not a fork
of them.

## Reproduce

```
SIM_BACKEND=numpy python -m research.runners._onebrain_wave2_organread_verify \
    --seeds 42,43,44 \
    --out research/findings/raw/_onebrain_wave2/organread_3seed_smoke.json
```

## Next steps

1. **Queued (this landing):** the 6-seed cupy organ-read verify, skip-guarded on this runner's presence.
2. **Separate, later:** add `shared=` support to the production classes that lack it and wire the nine organs'
   `get_organ()` singletons onto `get_wave2_pool()` behind `BRAIN_ONEBRAIN_WAVE2_POOL`, gated by
   `onebrain_regression_battery.py` through the real `webapp.server.brain_chat` — mirroring how the wave-1 pool's
   own production wiring is sequenced as a distinct, later commit after its organ-read GO.
3. **Wave 3** (d6-multiref-WM + prospective-memory, per the program doc) should check its OWN region-name and
   wiring-key footprint against the full 9-organ Wave-2 superset before assuming only the plan doc's named seams
   apply — this rung found a THIRD, silent, non-`MergeConflict` seam (the wiring-key collision) that no prior
   landing's plan doc anticipated, purely by enumerating rather than assuming.
