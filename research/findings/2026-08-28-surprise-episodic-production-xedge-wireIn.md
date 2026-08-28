---
type: finding
status: live
date: 2026-08-28
mechanism: onebrain-surprise-episodic-129construction-production-wireIn
lane: onebrain-integration
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_xedge_surprise_episodic_production_frozen_6seed.json
runner: research/runners/onebrain_xedge_surprise_episodic_production.py
builds_on:
  - research/findings/2026-08-28-surprise-episodic-129construction-6seed-GO.md
  - research/findings/2026-08-28-declarative-cross-edge-framework-gap-analysis-already-closed.md
  - research/findings/2026-08-27-onebrain-r4-selfschema-provenance-production-GO.md
---

# board #129's surprise->source_provenance TWO-cross-edge construction is already DECLARATIVE (verified, no new
layer needed) and is now wired into the live chat brain as an additive, default-OFF diagnostic — lesion-attributable
6/6 through the production functions, real-HTTP no-regression PASSED, one real-HTTP load-bearing check left
uncompleted within this session's time-box (disclosed, not hidden)

**One-line:** The just-landed `surprise->source_provenance` construction
(`2026-08-28-surprise-episodic-129construction-6seed-GO.md`) turns out to already be expressed through the
declarative `CrossEdge`/`merge_organs(cross_edges=...)` framework (verified by reading
`_onebrain_surprise_episodic_129construction_derisk.py:_build_pool_129`, not assumed) — so stage 1 of this task
("declarative re-expression") required no new code, only verification + a reproduction re-run. Stage 2
(production wire-in) is new: `research/runners/onebrain_xedge_surprise_episodic_production.py` co-locates the
construction's two frozen cross-edges on a shared merge pool and attaches a live, additive diagnostic field
(`resp["surprise"]["source_provenance_crossedge"]`) to `webapp/server.py`'s existing D2 surprise block, gated
default-OFF by `BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC`. 6/6 seeds GO on the offline production self-test
(lesion-attributable, `n_hollow=0`); a real-`/api/brain-chat`-handler no-regression test PASSED (byte-identical
whole-response on an ordinary turn, flag on vs off); a second real-handler test exercising an actual live D2
surprise trigger (the load-bearing on/lesion check through the HTTP layer itself, not just the production
functions) was started but did not finish inside this session's time-box and was killed rather than left silently
unreported — named as the one open residual, not smoothed over.

## 1. Stage 1 — declarative re-expression: ALREADY TRUE, verified not assumed

`_onebrain_surprise_episodic_129construction_derisk.py:_build_pool_129` (the just-merged GO finding's own
construction) builds its pool via:

```python
from research.runners.onebrain_merge_framework import REGISTRY, CrossEdge, merge_organs
CROSS_EDGES = [
    CrossEdge(key=GATE, source_key="surprise", source_region="surprise",
             target_key="source_provenance", target_region="prov_generated", ...),
    CrossEdge(key=GATE2, source_key="surprise", source_region="patient_expected",
             target_key="source_provenance", target_region="prov_perceived", ...),
]
pool = merge_organs([SURPRISE_LITE, SP], seed=seed, wire=True, cross_edges=CROSS_EDGES)
```

Both cross-edges are `CrossEdge` registry rows through `merge_organs(cross_edges=...)` — the exact declarative
path `research/findings/2026-08-28-declarative-cross-edge-framework-gap-analysis-already-closed.md` verified
already closed the audit's #1-ranked ask. Unlike R1 (`_onebrain_declarative_crossedge_r1_repro.py`) and R4
(`_onebrain_declarative_crossedge_r4_repro.py`), there is **no bespoke hand-typed predecessor to reproduce
against** for this specific two-edge construction — the ORIGINAL single-edge blocked runner
(`_onebrain_integration_surprise_episodic_crossedge.py:218-223`) was ALSO already declarative (the gap-analysis
finding's own §1, "a second, independent real-world use"). So "does it reproduce the 6/6 GO" here means
determinism/reproducibility of the ALREADY-declarative construction, not bespoke-vs-declarative equivalence.

**Reproduction check performed this session** (not skipped): re-ran seed 42 fresh
(`SIM_BACKEND=numpy python -m research.runners._onebrain_surprise_episodic_129construction_derisk --seeds 42
--floor 0.0478`) and compared against `research/findings/raw/_onebrain_surprise_episodic_129construction_6seed.json`'s
seed-42 row: `delta_intact=0.19641086282693246`, `delta_lesion=0.01261980518111741`,
`frac_attributable=0.9357479265684131` in BOTH — bit-for-bit identical (float-exact), confirming `cfg.seed`
determinism holds for this construction across independent process invocations (the CLAUDE.md seed trap does not
apply here, as the original finding's own two-process seed-42 comparison already showed).

**Verdict on stage 1: no new declarative layer built — none was needed.** The construction was declarative from
first landing. This session's contribution to stage 1 is verification (reading the actual code, not trusting the
label) + one independent reproduction re-run, not new mechanism.

## 2. Stage 2 — production wire-in (new this session)

`research/runners/onebrain_xedge_surprise_episodic_production.py` (new, mirrors PART-1
`onebrain_xedge_production.py` and R4's `onebrain_xedge_selfschema_production.py` verbatim in shape):

- **`XedgeSurpriseEpisodicProductionPool`**: builds `SurpriseEpisodic129Pool(seed)` (reused by import, unchanged —
  the exact class the 6-seed GO validated), calls its `train_129()` once (grows both cross-edges 0.05 -> trained
  by the substrate's own Hebbian rule), then freezes (`enable_hebbian_learning=False`, already left by
  `train_129()`, plus an explicit `set_plasticity_gate(GATE/GATE2, 0.0)` for defensive parity with the PART-1/R4
  convention). No weight moves during any live turn. Grown IN-PROCESS (not a saved weight file), for the same
  cross-backend-seed-trap reason PART-1/R4 document.
- **`crossedge_provenance_shift_129(pool, hold_surprise)`**: the live-turn hook. Delegates to
  `SurpriseEpisodic129Pool.amb_read_ratio` (the construction's OWN validated F2 instrument, reused verbatim) to
  read the divisive-ratio provenance margin on the construction's fixed ambiguous item, with vs without holding
  the surprise circuit's CONTRADICT drive, and reports the shift toward GENERATED.
- **Runs on an INDEPENDENT pool instance**, not the live `surprise_production_organ` singleton the D2 notice /
  reconsolidation pipeline uses — zero shared mutable state with the already-default-ON surprise mechanism.

`webapp/server.py` (inside the existing D2 surprise block, right after `surprise_info = dict(sj)` /
the reconsolidation try/except): a new guarded, additive block reads `surprise_info["surprised"]` (the turn's
OWN live D2 verdict, already computed by the existing block) and, only when
`BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC` is set, attaches `surprise_info["source_provenance_crossedge"]` —
never touching `resp["answer"]` or any pre-existing `surprise` field. 34 lines changed, all inside the one
guarded block (`webapp/server.py:5326-5360`).

`BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC_LESION=1` zeroes BOTH cross-edges together (`masks["both_edges"]`) — the
SAME joint lesion the 6-seed research GO's own F2 gate used, not a weaker single-edge lesion.

## 3. Verification

<!--derived-->

### 3a. Offline 6-seed production self-test (0 Claude tokens, numpy CPU, ~7 min)

Artifact: `research/findings/raw/_onebrain_xedge_surprise_episodic_production_frozen_6seed.json`.

| seed | intact shift | lesion shift | frac attributable | no-signal-no-bias | n_hollow | clears registered floor (0.0478) |
|---|---|---|---|---|---|---|
| 42 | +0.2007 | +0.0166 | 0.9173 | True | 0 | True |
| 43 | +0.1860 | +0.0123 | 0.9338 | True | 0 | True |
| 44 | +0.1268 | +0.0129 | 0.8984 | True | 0 | True |
| 100 | +0.2006 | +0.0103 | 0.9484 | True | 0 | True |
| 101 | +0.1571 | +0.0061 | 0.9610 | True | 0 | True |
| 102 | +0.1484 | +0.0137 | 0.9079 | True | 0 | True |

**6/6 GO on lesion-attributability** (this wiring's own crux, exercised through the ACTUAL production functions
`crossedge_provenance_shift_129`/`get_xedge_surprise_episodic_pool` the webapp calls, just not yet through the
Flask HTTP layer itself — see 3c). **`n_hollow=0` on every seed** (board #94-class anti-hollow bar: the coupling
demonstrably drives a real, lesion-attributable, non-negligible difference). **6/6 clear the construction's own
pre-registered `F2_INTACT_FLOOR_129=0.0478`** (frozen from the research finding's own seed-7 calibration, reused
verbatim, not re-derived) under this wiring's simpler call sequence (train -> read, no F1/F3/F4 pre-conditioning
steps unlike the construction's own `run_seed`) — this wire-in's own shift values (0.127-0.201) run slightly
above the research runner's own 6-seed range (0.126-0.198), close but not identical, the expected small residual
from the simpler production call order (the SAME class of residual the R4 wire-in's own finding documents and
does not hide).

The "no signal, no bias" control (`surprise_held=False`, both intact and lesioned) reads within the construction's
own `F2_LESION_RATIO=0.34` noise floor on every seed — holding nothing produces no shift, as it must.

### 3b. Byte-identical-off — verified in the DATA, not inferred from the code

Per `docs/TERMS.md`'s bar ("asserted in the data (hash or exact compare), never inferred from reading the
code"): the EXACT conditional logic added to `webapp/server.py` was extracted and run standalone against a fake
`surprise_info` dict with the flag unset — `json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)`
returned `True` (an exact string/hash compare, not a code-reading inference). Separately, the real-HTTP
no-regression test (3c below) independently confirms the SAME property on an ordinary ORDINARY (non-surprising)
ChatBrain turn, through the real ASGI stack — `d_off == d_on` (a real dict-equality assertion on the FULL
response body).

### 3c. Through the real `/api/brain-chat` handler (2 pytest tests attempted, `tests/test_webapp_server.py`)

| test | result |
|---|---|
| `test_brain_chat_xedge_surprise_episodic_no_regression_on_ordinary_recall_turn` | **PASS** (486.76s) — an ordinary recall turn's WHOLE response is structurally IDENTICAL (`d_off == d_on`) whether the flag is off or on; `"source_provenance_crossedge" not in surprise` on both arms |
| `test_brain_chat_xedge_surprise_episodic_default_off_is_byte_identical` + `test_brain_chat_xedge_surprise_episodic_on_reads_live_crossedge_and_lesion_collapses` | **NOT COMPLETED** — started (a genuine D2-surprise-triggering turn: `"dog chase mouse"` against tiny-demo's stored `dog chase cat`), ran past 11 minutes wall-clock without finishing, and was killed to respect this session's time-box rather than left silently unreported |

**Honest residual, named not hidden**: the load-bearing on/lesion check (does the diagnostic actually
drive-then-vanish-under-lesion when a GENUINE live D2 contradiction fires, exercised through the real HTTP
handler rather than the production module's own functions directly) was not completed inside this session. The
claim it would have checked is NOT unverified in substance — §3a already exercises the identical
`crossedge_provenance_shift_129`/`get_xedge_surprise_episodic_pool` functions `webapp/server.py`'s new block
calls, 6/6 GO, `n_hollow=0` — but the SPECIFIC "a real live turn that trips D2 surprise, through the ASGI stack,
attaches the diagnostic and the diagnostic's own lesion collapses it" proof, at the HTTP layer, is not yet
banked. The two tests are already written (`tests/test_webapp_server.py`, same file as the no-regression test)
and ready to run to completion in a follow-up session; they were not deleted or weakened to force a false pass.

## 4. Honest residuals (declared, carried + new)

1. **The real-HTTP load-bearing on/lesion test did not finish this session** (§3c) — the highest-priority
   follow-up, not a correctness doubt (§3a's evidence is real, just not yet re-proven through the HTTP layer).
2. **Carried from the construction's own finding, §5, NOT re-litigated here**: an individual-edge lesion control
   found the NEW `patient_expected->prov_perceived` edge (CONFIRM-trained) reproduces nearly the full intact
   shift alone, while the ORIGINAL `surprise->prov_generated` edge (CONTRADICT-trained, validated on every other
   arm) alone reproduces almost none of it. This wire-in's `BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC_LESION` lesions
   BOTH edges together (the pre-registered, cleanly-passing gate) — it exercises the validated JOINT mechanism,
   not a resolution of that open circuit-level question.
3. **Diagnostic-only, not a coupling into an arbitrary live fact.** The cross-edges bias the construction's own
   fixed, dual-context-encoded ambiguous content pattern (a substrate stand-in), not an arbitrary recalled fact —
   identical shape to PART-1/R4's own declared residual.
4. **Not strict `self-organized`** (per `docs/TERMS.md`): cross-edge topology + training schedule are
   host-chosen/host-curated; the WEIGHT is learned by the substrate's own Hebbian rule.
5. **No ledger row added** to `docs/PRODUCTION_INTEGRATION_LEDGER.yaml` — this wiring is default-OFF (not an
   on-by-default production claim), mirroring the R4 wire-in's own precedent (also absent from that ledger).

## 5. Verdict

**Stage 1 (declarative re-expression): CONFIRMED — no new code needed, verified by reading the actual
construction code and by an independent bit-exact reproduction re-run of seed 42.**

**Stage 2 (production wire-in): reachable from `research/runners/onebrain_xedge_surprise_episodic_production.py`
and `webapp/server.py`'s D2 surprise block, default-OFF (`BRAIN_ONEBRAIN_XEDGE_SURPRISE_EPISODIC`, unset ->
byte-identical), 6/6 lesion-attributable on the offline production self-test (`n_hollow=0`), one real-HTTP
no-regression test PASSED, one real-HTTP load-bearing test started and not completed this session (disclosed).**
Flip-ready in the sense of "non-hollow 6/6" at the production-function level; NOT flip-ready in the stricter
sense of "proven through the HTTP layer under a genuine live trigger" until §3c's second test completes. Kept
default-OFF by design — a later flip-soak (after §3c completes) owns the default-ON decision, per instruction
(never autonomous). Functional read-outs only; no phenomenal-experience claim.

## 6. Ranked next

1. **Finish §3c's two real-HTTP tests** (already written, `tests/test_webapp_server.py`) — the single item
   standing between this wiring and "flip-ready, non-hollow 6/6" at the HTTP layer.
2. **Investigate the confirm-side-edge-dominance open question** (construction finding §5) at the circuit level.
3. **Migrate this edge through the framework's own F1-F4 genericization** (the gap-analysis's §3 residual: seven
   near-duplicate hand-typed F-gates) — this edge is a third concrete consumer motivating that generic harness.
4. **A later, owner-gated flip-soak** once §3c lands — mirrors the R4/PART-1 sequencing exactly.

## Files

`research/runners/onebrain_xedge_surprise_episodic_production.py` (new, production wiring module) ·
`webapp/server.py:5326-5360` (the guarded, additive attach inside the existing D2 surprise block) ·
`tests/test_webapp_server.py` (3 new tests: 1 PASSED, 2 started/not completed, all real) ·
`research/findings/raw/_onebrain_xedge_surprise_episodic_production_frozen_6seed.json` (+ `.prov.json`) — the
6-seed offline production self-test artifact (n_go=6/6, n_hollow_total=0). No `sim/` file touched.
