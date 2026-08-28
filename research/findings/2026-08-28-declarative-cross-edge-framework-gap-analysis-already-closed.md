---
type: finding
status: live
date: 2026-08-28
mechanism: declarative-cross-edges-framework-gap-analysis
lane: onebrain-integration
artifacts:
  - research/findings/raw/_onebrain_declarative_crossedge_r1_repro_6seed.json
runner: research/runners/onebrain_merge_framework.py
---

# The audit's #1-ranked "make cross_edges DECLARATIVE" investment was ALREADY BUILT before this task started — gap analysis finds no genuine gap in the primary ask, so no Phase-2 layer was added

**One-line:** This task was dispatched to build a declarative `cross_edge` framework so migrating a GO faculty
into the one-brain via a learned cross-region edge becomes a cheap registry row instead of a bespoke ~37-46KB
runner (the completeness audit's `research/findings/2026-08-27-onebrain-completeness-audit.md` §4 step-2
ranking). Phase-1 verification (git log, the local RAG corpus, `tools/before_you_build.sh`) found that this
exact deliverable was **already built, merged to `main`, and verified** roughly 40 minutes after the audit that
named it — `a7728ffcc8/c15e59f5b6`, 2026-08-27 20:45-20:46, on a branch (`research/declarative-cross-edges`)
almost identically named to the one this task was told to use. It is an ancestor of this worktree's `HEAD`
(`3c5b75ed5`). **Verdict: no genuine gap in the primary ask. Per this task's own stop-criterion, no redundant
Phase-2 layer was built.** The one honest residual (F1-F4 gate genericization) and the ranked next migration
pairs are below.

## 1. What exists — verified against the actual code, not just the commit message

`research/runners/onebrain_merge_framework.py`:

- **`CrossEdge`** (frozen dataclass, line 74) — `key, source_key, source_region, target_key, target_region,
  init_weight=0.05, plastic=True, gate=None, learn_rule="rate_hebbian", freeze_rest=True`. `learn_rule` is a
  documentation field; the hebbian/reward hyperparameters still ride the existing `config_descriptors` union
  (no new hyperparameter mechanism was needed — this matches exactly what this task's brief called "the
  learning-rule/params" part of the spec).
- **`_cross_edge_dense(bridge, ce)`** (line 108) — builds the dense pre/post/weight population `inject_explicit_
  wiring` consumes, reading both endpoints off `region_manager.indices(name)`, byte-identical in construction
  shape to R1's hand-typed `_dense(...)`.
- **`cross_edges=` param on `MergedPool.__init__`/`merge_organs`** (lines 231-235, 577-593) — threaded through;
  `ensure_built()` raises `MergeConflict` if `cross_edges` is given without `wire=True` (fail loud, not silent).
- **`MergedPool.apply_cross_edge_freeze()`** (line 443) — generalizes R1Pool's 3 hand-typed whitelist-inversion
  lines (`set_plasticity_gate(GATE,1.0)` / `cp_plasticity_rate_gain[:]=0.0` / `set_plasticity_gate(GATE,1.0)`)
  to every registered edge with `freeze_rest=True`. No-op when no edge wants it.
- **Byte-identical-off**: every new code path is gated on `self.cross_edges` being truthy; the framework's
  existing independent-comparison gates (`--smoke`, `--smoke2`, `--determinism2`, `tests/test_determinism.py`)
  were re-run after the patch and stayed clean (max_init_delta=0.0, 9 passed / 2 skipped) — none of them pass
  `cross_edges`, so this is the same evidence the framework carried before, re-confirmed after.

**The reproduction proof** (`research/runners/_onebrain_declarative_crossedge_r1_repro.py`, new file):
`DeclarativeR1Pool` subclasses the bespoke `R1Pool` and overrides only `__init__` — instead of R1's hand-typed
`_dense(...)` union + 3-line whitelist, it calls `merge_organs([D6, COMP], wire=True,
cross_edges=CROSS_EDGES)` + `pool.apply_cross_edge_freeze()`. 4 `CrossEdge` rows match R1's `x_w0_sela` /
`x_w0_selp` / `x_w1_sela` / `x_w1_selp` edges exactly. Every downstream method (`_hard_reset`, `_drive`,
`train`, `amb_read`, `cross_weights`) is inherited unchanged, so the train/read protocol is provably identical
between arms — only pool construction differs. Both arms run through **R1's own unmodified imported F1-F4 +
emergence + lesion-recovers-migration functions** (reuse-by-import, not reimplementation).

**Result, 6 seeds (42/43/44/100/101/102), both arms run fresh in-process per seed**
(`research/findings/raw/_onebrain_declarative_crossedge_r1_repro_6seed.json`, `GO: true`, `n_go: 6/6`): grown-
weight max|delta| between the two construction paths = **0.000** at every seed (bit-for-bit, not merely
functionally equivalent), F2 lesion-attributable fraction reads **1.0/1.0** in both arms at every seed. This is
a stronger bar than the pre-registered check (weight max|delta|<1.0, attributable-fraction within 0.15).

**A second, independent real-world use, beyond the reproduction proof**: `research/runners/
_onebrain_integration_surprise_episodic_crossedge.py:218-223` constructs a genuinely NEW edge
(surprise→source_provenance) directly via `CrossEdge(key=GATE, source_key="surprise", ...)` +
`merge_organs([SURPRISE_LITE, SP], wire=True, cross_edges=CROSS_EDGES)` — not a reproduction of an existing
bespoke runner, an edge that never had a bespoke hand-wired form. Its F1/F3/F4/emergence/migration read clean
6/6; F2 (the read of the interaction itself) is UNDEFINED on a read-fidelity instrument crux (rate-saturation,
being worked as its own active lane — see `research/findings/2026-08-27-onebrain-surprise-episodic-crossedge-
F2-crux-UNDEFINED.md` and the ongoing non-rate-read de-risks, e.g. `f10edd601`). The construction/wiring
succeeded cleanly through the declarative path; the open problem is a read instrument, not the framework — this
is exactly the kind of case the framework was meant to make cheap, and it did.

## 2. Verdict on the primary ask

**Migration is already declarative for the wiring/construction step.** A new learned cross-region edge is now:
(a) name its `source_region`/`target_region`, (b) pick `init_weight`/`gate`/`freeze_rest`, (c) call
`merge_organs(..., wire=True, cross_edges=[CrossEdge(...), ...])` + `apply_cross_edge_freeze()`. That is a data
row plus the edge-specific train/read/lesion functions the F-gate needs — not a from-scratch ~37-46KB runner.
Two edges now go through it (the R1 reproduction and the new surprise→provenance edge), one bit-for-bit
matching a known-GO bespoke baseline and one a genuinely new construction. Per this task's explicit
stop-criterion ("if it turns out most of it exists ... STOP rather than build a redundant layer"), **no Phase-2
declarative-spec/validator layer was built** — building one now would duplicate `CrossEdge` +
`apply_cross_edge_freeze`, which already exist, match the brief's spec almost field-for-field, and are already
in second use.

## 3. The one honest residual — not built, not blocking

The **F1-F4 functional gate itself is still hand-typed per edge**: `_f1/_f2/_f3/_f4` are separately defined in
`_onebrain_integration_r1_wm_comprehension.py`, `_r2_threefactor_selforganized.py`,
`_r3_spiking_dopamine_credit.py`, `_r3v2_noncorrupting_dopamine_credit.py`, `_r3v3_functional_drive.py`,
`_r4_selfschema_provenance.py`, and `_surprise_episodic_crossedge.py` — seven near-duplicate implementations,
not one generic harness. This is real duplication, but it is a narrower and lower-leverage gap than the wiring
question the audit ranked #1: what each F-gate actually reads (comprehension margin vs. self-schema author-score
vs. surprise-provenance drive) is inherently edge-specific, and the framework's own reuse-by-import pattern
already avoids *re-implementing* a gate once it exists (the R1 repro imports R1's F-gate unchanged; a future
edge could do the same for whichever existing gate is the closest analogue). The generic instruments underneath
each hand-typed gate (`tools.lab.lever`, `attributable_to`, `bound_check`, `zero_lever_control`) are already
shared, not duplicated. Genericizing the outer F1-F4 harness (accepting `train_fn`/`read_fn`/`lesion_fn` as
parameters) is a plausible future rung, but was not attempted here — it wasn't the audit's ranked ask, none of
the seven existing edges asked for it, and speculatively building it without a second concrete consumer waiting
on it risks being exactly the "redundant layer" this task's brief warned against.

## 4. Ranked next faculty pairs (updates the audit's §4 roadmap with what has since landed)

Re-checked each of the audit's 8 ranked steps against the current repo state (git log, `research/findings/`):

1. **Strengthen + default-flip d6→comprehension** — **DONE.** `research/findings/2026-08-28-onebrain-xedge-
   production-default-flipped-ON-6seed-GO.md`: `BRAIN_ONEBRAIN_XEDGE` + `BRAIN_ONEBRAIN_XEDGE_LEARN` flipped
   True by default, FLIP_VERIFY_GO=True through the real `/api/brain-chat` handler, 6 seeds, `n_hollow=0`. (An
   earlier same-day NO-GO on this edge was a cupy backend-crash instrument artifact — `_snapshot_rest` storing
   host numpy into a device cupy array — fixed and voided per that finding's own correction banner; the ON-GO
   supersedes it.)
2. **Make cross-edges declarative** — **DONE.** This finding's subject (§1-2 above).
3. **R4 self_schema→source_provenance into production** — **PARTIALLY DONE.** `d84775aa8` wired the FROZEN R4
   edge into the live chat brain as an additive, lesion-attributable diagnostic, default-OFF
   (`BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA`). Not yet default-flipped. **Recommended next candidate #1** — closest to
   done, and directly on the honesty-boundary deliverable (authorship→"is this my own thought"), a
   prefrontal-medial self-monitoring loop consistent with the audit's Magrou/Gamanut reciprocal-pair framing.
4. **R2 three-factor (neuromod-gated) upgrade** — GO 6/6 at the framework level (`4d9978c49`, self-organized
   topology PARTIAL), **not yet** re-expressed through the now-proven `CrossEdge` path nor wired into
   production. **Recommended next candidate #2** — establishes the neuromod-gated-plasticity backbone every
   later edge (including live per-turn learning) reuses.
5. **R3 surprise→episodic/provenance encoding gate** — construction is now declarative (§1 above, second CrossEdge
   consumer) but the interaction READ is UNDEFINED on a read-fidelity instrument crux, an active separate lane
   (non-rate first-spike-latency/dispersion reads being tried, e.g. `f10edd601`). Not a wiring gap.
6. **Migrate d5_episodic (Group-C own-pool seam)** — no evidence found of this having started; still open.
7. **affect→tone/mouth edge** — ambiguous: a related but possibly-distinct lane exists
   (`research/affect-tone-spiking-mouth`, `6fb0611e8`/`515fdc651`, "Gate-B mood coloring on the spiking recall
   mouth"). Whether it routes through `OrganDescriptor`/`CrossEdge` or is a separate mechanism was **not
   verified in this pass** (out of this task's tight scope) — flag for a fresh check before ranking it against
   candidates #1/#2 above.
8. **Reciprocal / multi-edge loops** — last step, not started, correctly sequenced last.

## Files

`research/runners/onebrain_merge_framework.py` (`CrossEdge:74`, `_cross_edge_dense:108`, `MergedPool.
apply_cross_edge_freeze:443`, `merge_organs:577`) · `research/runners/_onebrain_declarative_crossedge_r1_repro.py`
(reproduction proof) · `research/runners/_onebrain_integration_surprise_episodic_crossedge.py:218-223` (second
real consumer) · `research/findings/2026-08-27-declarative-cross-edges-framework-GO.md` (the original landing
finding, read in full for this gap analysis) · `research/findings/2026-08-27-onebrain-completeness-audit.md`
(the ranking this closes, §4 steps 1-8) · `research/findings/2026-08-28-onebrain-xedge-production-default-
flipped-ON-6seed-GO.md` and its superseded `...-NO-GO.md` (step 1 status). No `sim/`/`webapp/` file touched;
no new code added; this is a verification-and-documentation finding only.
