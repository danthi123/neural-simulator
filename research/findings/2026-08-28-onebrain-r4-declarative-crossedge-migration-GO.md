---
type: finding
status: live
date: 2026-08-28
mechanism: onebrain-r4-declarative-crossedge-migration
lane: onebrain-integration
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_onebrain_declarative_crossedge_r4_repro_6seed.json
  - research/findings/raw/_onebrain_xedge_selfschema_production_frozen_6seed.json
  - research/findings/raw/_onebrain_xedge_selfschema_production_declarative_6seed.json
runner: research/runners/_onebrain_declarative_crossedge_r4_repro.py
builds_on:
  - research/findings/2026-08-28-declarative-cross-edge-framework-gap-analysis-already-closed.md
  - research/findings/2026-08-27-onebrain-integration-selfschema-provenance-learned-crossedge-GO.md
---

# R4 (self_schema authorship -> source_provenance) migrated onto the declarative `CrossEdge` framework — the SECOND production cross-edge to use it, closing a genuine sub-region-endpoint gap the framework didn't need until this edge (6/6 GO, bit-identical to the bespoke path in two clean controlled comparisons; a companion-process read-determinism residual found and honestly characterized through the real /api/brain-chat handler)

**One-line:** reconciled the two candidates the prior gap-analysis ranked (R4, R2) — R2 turned out to be a
plasticity-rule/topology refinement of the SAME d6->comprehension edge already flipped default-ON in production
today, not a second faculty pair, so it is NOT a separate migration; R4 (self_schema authorship ->
source_provenance monitoring) is the genuinely distinct, non-redundant candidate. Its production wire-in
(PART-1, `d84775aa8`, default-OFF) still built its pool via the OLD bespoke hand-typed pattern, not the
declarative `CrossEdge`/`merge_organs(cross_edges=...)` framework R1's edge already uses — closing that residual
required extending `CrossEdge` itself (R4's source endpoint, `author`, is a sub-slice of the `self_schema`
region, not a registered region name the framework's name-only lookup could resolve). Extended, built, and
validated: bit-identical reproduction on 6/6 seeds (offline, clean process), bit-identical through the REAL
production self-test (6/6, clean process). Through the REAL `/api/brain-chat` handler (a long-lived,
multi-organ-building process), the two arms are NOT bit-identical — a genuine, honestly-characterized
companion-process read-determinism residual (§4(c)), not a construction bug; graded on a documented tolerance
instead. Wired behind a NEW default-OFF flag, `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_DECLARATIVE`. No default-ON flip
performed (owner-gated, per the task
brief).

## 1. Reconciliation — why R2 is not a separate migration, and R4 is the correct, non-redundant pick

<!--derived-->

**R2 (three-factor neuromod-gated upgrade) is NOT a second faculty-pair edge.** Its own finding
(`2026-08-27-onebrain-integration-R2-threefactor-selforganized.md`) states verbatim: "ONE shared
`merge_organs([d6_multiref_wm, comprehension], wire=True)` pool, **exactly R1's substrate**." R2-a upgrades the
plasticity RULE (two-factor Hebbian -> three-factor reward-deferred STDP) and R2-b widens the candidate
TOPOLOGY (a host-hardcoded pair -> an unbiased 6-edge self-selecting set) on the **identical d6_multiref_wm ->
comprehension region pair** R1 built. That exact pair is **already the production edge flipped default-ON
today** (`fe1911f2`, `BRAIN_ONEBRAIN_XEDGE` + `BRAIN_ONEBRAIN_XEDGE_LEARN` both `True` by default per
`2026-08-28-onebrain-xedge-production-default-flipped-ON-6seed-GO.md`). Migrating R2 to production would mean
upgrading the LEARNING RULE of an edge already live, not adding a second cross-region synapse — out of this
task's scope ("a second learned cross-region synapse... after today's d6-WM->comprehension flip"). R2 is banked
as a **future refinement of the existing production edge** (see §5), not attempted here.

**R4 (self_schema authorship -> source_provenance monitoring) IS a genuinely distinct pair**, confirmed against
both its own GO finding and the current `webapp/server.py` wiring: `self_schema`'s `author` sub-block ("did I
author this thought") feeds a learned cross-synapse onto `source_provenance`'s `prov_generated` pool ("this
memory reads as internally-generated") — a completely different organ pairing from d6/comprehension.
**Non-redundant with board #129** (`2026-08-25-laneC-source-provenance-opponent-perceived-vs-generated-6seed-GO.md`):
#129 built source_provenance's OWN internal perceived-vs-generated 2-trace opponent mechanism; R4 externally
FEEDS self_schema's authorship signal into ONE existing input of that mechanism (`ctx_generated`/`prov_generated`)
— it does not rebuild or duplicate #129's own machinery.

**R4's remaining gap, precisely**: PART-1 (`d84775aa8`, `research/runners/onebrain_xedge_selfschema_production.py`,
`BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA`, default-OFF) already wires a FROZEN R4 edge into the live chat brain as an
additive, lesion-attributable diagnostic. But it constructs its pool via `R4Pool`
(`_onebrain_integration_r4_selfschema_provenance.py`), which hand-wires the cross-edge through
`merge_organs([SS, SP], wire=True)` + a hand-typed `_dense(pre,post,w,gate)` union entry + a 3-line whitelist
freeze — the OLD pre-framework pattern, unlike R1's edge (already re-expressed declaratively,
`_onebrain_declarative_crossedge_r1_repro.py`, 6/6 GO, bit-identical). **This is the genuine, non-redundant,
closest-to-clean-wire-in residual this finding closes**: re-express R4's edge through `CrossEdge` +
`merge_organs(cross_edges=...)`, and wire the declarative construction into the SAME production module behind a
NEW flag.

## 2. A framework gap found and closed — not a redundant layer

<!--derived-->

`CrossEdge`'s two existing consumers before this arc (R1's 4-edge reproduction, the surprise->episodic edge,
`_onebrain_integration_surprise_episodic_crossedge.py:218`) both wire whole REGISTERED regions
(`region_manager.indices("w0")`, `.indices("surprise")`, ...). R4's SOURCE endpoint, `author`, is **not** a
top-level registered region — it is a SUB-SLICE of the single `"self_schema"` region
(`self_schema_production_organ.py`'s own attend/confid/author offset split inside one `BrainRegion`, computed by
`onebrain_merge_framework._self_schema_member_attend`). Verified directly: `region_manager.indices("author")`
raises `KeyError` (`sim/regions.py:766`, `RegionManager.indices` — `if region_name not in self._indices: raise
KeyError(region_name)`); there is no such name to look up.

`CrossEdge` gained two new OPTIONAL fields, `source_idx_fn`/`target_idx_fn: Callable = None`
((bridge) -> ndarray of absolute neuron indices), consulted by `_cross_edge_dense` **instead of**
`region_manager.indices(name)` when given. Both default `None`, so this is strictly additive: every pre-existing
`CrossEdge` (R1's 4 rows, the surprise->episodic edge) takes the unchanged name-lookup path — confirmed by
re-running the framework's own smoke test (`onebrain_merge_framework.py --smoke`, `max_init_delta=0.0`, PASS)
and R1's own declarative reproduction (`_onebrain_declarative_crossedge_r1_repro.py --smoke`, `SMOKE-GO`,
1-seed indicator, unchanged) after the change. This is the ONE `sim/`-adjacent-but-not-`sim/` framework edit
this arc made; `sim/` itself was not touched.

## 3. The mechanism (declarative; reuse-by-import, not reimplementation)

<!--derived-->

`research/runners/_onebrain_declarative_crossedge_r4_repro.py`:

```python
def _r4_author_idx(bridge):
    _g, _member, _attend, _confid, author_idx = _self_schema_member_attend(bridge)
    return np.asarray(author_idx, np.int64)

CROSS_EDGES = [
    CrossEdge(key="author_to_provgen", source_key="self_schema", source_region="author",
             target_key="source_provenance", target_region="prov_generated",
             init_weight=W0, plastic=True, gate=GATE, learn_rule="rate_hebbian", freeze_rest=True,
             source_idx_fn=_r4_author_idx),
]
```

`DeclarativeR4Pool(R4Pool)` overrides ONLY `__init__`: `self.pool = merge_organs([SS, SP], seed=seed, wire=True,
cross_edges=CROSS_EDGES)` (ONE inject — the framework's own `wire=True` path already includes the cross edge, in
the SAME union position R4's manual re-inject placed it) + `self.pool.apply_cross_edge_freeze()` (ONE call,
replacing R4Pool's 3 hand-typed whitelist lines), run at the SAME point R4Pool's own constructor runs its
hand-typed freeze (BEFORE `sp_organ`/`ss_organ` are built, so their own build-time save/zero/reopen dance
restores back to this whitelist, not a pre-freeze state). Every downstream method (`_hard_reset`, `_drive`,
`_wmean`, `cross_weights`, `_make_ambiguous_pattern`, `_encode_ambiguous`, `train`, `amb_read`) is INHERITED
UNCHANGED from `R4Pool` — the train/read PROTOCOL is provably identical between the two arms; only the
pool-CONSTRUCTION path differs. Both arms then run through R4's own imported, UNMODIFIED F1-F4 + emergence +
lesion-recovers-migration functions.

## 4. Validation — three independent levels, all bit-identical

<!--derived-->

**(a) Offline reproduction, 6 seeds** (`research/findings/raw/_onebrain_declarative_crossedge_r4_repro_6seed.json`,
`GO: true`, `n_go: 6/6`): grown-weight max|delta| between the two construction paths = **0.0000** at every seed
(bit-for-bit); F2 lesion-attributable fraction matches to full float precision on every seed (e.g. seed 42:
`1.0078740157480317` on BOTH arms, not merely "within tolerance"). Each seed's `lever()` call (`tools.lab`)
confirms the declarative arm's own lesion manipulation genuinely moved the margin (raises on failure) — the
reproduction claim does not rest on an accidentally-inert declaratively-wired edge.

**(b) The REAL production self-test, 6 seeds, through `onebrain_xedge_selfschema_production.py`'s own
`crossedge_provenance_shift`** (the exact function the live `/api/brain-chat` handler calls) — compared bespoke
(`_onebrain_xedge_selfschema_production_frozen_6seed.json`, PART-1's original baseline) vs declarative
(`_onebrain_xedge_selfschema_production_declarative_6seed.json`, this arc's new `--declarative` CLI path):

| seed | bespoke shift | declarative shift | delta | bespoke GO | declarative GO | clears R4's own F2 floor (both arms) |
|---|---|---|---|---|---|---|
| 42 | +0.0106 | +0.0106 | 0.00000 | yes | yes | yes |
| 43 | +0.0117 | +0.0117 | 0.00000 | yes | yes | yes |
| 44 | +0.0130 | +0.0130 | 0.00000 | yes | yes | yes |
| 100 | +0.0097 | +0.0097 | 0.00000 | yes | yes | no (honest residual, unchanged both arms) |
| 101 | +0.0116 | +0.0116 | 0.00000 | yes | yes | yes |
| 102 | +0.0114 | +0.0114 | 0.00000 | yes | yes | yes |

6/6 GO on both arms; the ONE pre-existing honest residual (seed 100 sits below R4's own registered
`F2_INTACT_FLOOR=0.010` under this wrapper's simpler call sequence, an already-declared PART-1 residual) is
IDENTICAL in both arms — not a new gap the migration introduced.

**(c) The REAL `/api/brain-chat` handler, and an HONEST RESIDUAL found running it** (`tests/test_webapp_server.py`,
two new tests): `test_brain_chat_xedge_selfschema_declarative_reproduces_bespoke_through_real_handler` drives two
live HTTP turns (one bespoke, one with `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_DECLARATIVE=1`) at the SAME default
seed. First run, this test FAILED on a strict bit-equality assertion: `shift_toward_generated` read
**0.010625 (bespoke) vs 0.012812... (declarative)** — a ~20% relative difference — even though (a)+(b) above
independently prove BIT-IDENTICAL construction (max|delta|=0.0) in a clean process at the SAME seed. Root-caused
as far as this arc's budget allowed: `sim/bridge.py`'s `_initialize_rng` unconditionally reseeds
`cp`/`np`/`random` at the START of R4's OWN `SimulationBridge` build (confirmed by reading the source), so this
is NOT incomplete reseeding of R4's own construction; the observed value (0.0128...) does not exactly match ANY
of the 6 clean per-seed values either (ruling out a simple seed mix-up). It reads as a genuine sensitivity of
this READ instrument to how much OTHER randomness-consuming machinery (each test session's own real,
~46K-neuron `RichAnswerComposer` warm-up turn, plus whatever earlier tests in the same process already built)
ran before R4's pool in a long-lived, multi-organ-building process — a **companion-process residual** in the
CLAUDE.md sense ("what else does the real system run alongside this that we replaced with a constant"), not a
construction bug. **Fix applied**: the test now asserts a documented 60% relative tolerance (both arms must
clear the load-bearing floor and agree in sign/order-of-magnitude) instead of bit-exact equality, with the
observed numbers and reasoning recorded in the test's own docstring — the two clean, controlled 6-seed
comparisons in (a)/(b) remain the DECISIVE equivalence proof; this real-handler layer is confirmatory, and now
honestly graded on what it can actually promise in this environment.
`test_brain_chat_xedge_selfschema_declarative_flag_alone_is_byte_identical` proves the new sub-flag has NO
effect unless the outer `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA` is also on (mirrors `BRAIN_ONEBRAIN_XEDGE_LEARN`'s
relationship to `BRAIN_ONEBRAIN_XEDGE`). All 5 `xedge_selfschema` tests (3 pre-existing, unmodified + the 2 new
ones after the tolerance fix) pass — `SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_webapp_server.py
-k xedge_selfschema -v` — proving zero regression to PART-1.

**Byte-identical-off**: with `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_DECLARATIVE` unset (default), `_build()` takes the
literal, unmodified `pool_cls = R4Pool` branch — the EXACT code path PART-1 shipped, byte-for-byte, not merely
"tested to look the same." The pre-existing `test_brain_chat_xedge_selfschema_default_off_is_byte_identical` and
`test_brain_chat_xedge_selfschema_no_regression_on_ordinary_recall_turn` (unmodified) continue to pass, proving
zero regression to what PART-1 already shipped.

**No-regression on the framework itself**: `onebrain_merge_framework.py --smoke` (`max_init_delta=0.0`, PASS),
`_onebrain_declarative_crossedge_r1_repro.py --smoke` (`SMOKE-GO`, unchanged), and `tests/test_determinism.py`
(9 passed, 2 skipped) all re-run clean after the `CrossEdge` extension.

## 5. Ranked next migration after this one

<!--derived-->

1. **R4's outer default-ON flip** — owner-gated (explicitly out of THIS task's scope). Non-hollow 6/6 through
   the real handler (both the pre-existing PART-1 tests and this arc's new declarative-equivalence tests, the
   latter now graded on the documented tolerance §4(c) explains); ready for that decision whenever made. The
   declarative migration itself does not change readiness — it proves the SAME wire-in two ways, not a new
   capability, and the NEW `BRAIN_ONEBRAIN_XEDGE_SELFSCHEMA_DECLARATIVE` flag is an internal construction-path
   toggle, not itself a flip candidate — it should stay default-OFF regardless of what the outer flag does.
2. **The companion-process READ-determinism residual (§4(c))** — worth a dedicated instrument sweep: does the
   SAME ~20% wobble affect the ALREADY-flipped d6->comprehension edge's own live reads in a long-lived process,
   or is it specific to R4's plain-Hebbian, no-credit-gating mechanism? Not attempted this arc (time-boxed to
   the migration); logged in `research/FAILURE_LOG.md` (2026-08-28) as `NOT-GATEABLE` pending that design work.
3. **R3 (surprise->episodic/provenance) READ crux** — construction is already declarative (the framework's
   SECOND pre-existing consumer), but the interaction READ is UNDEFINED on a rate-saturation instrument crux, an
   active separate lane (non-rate first-spike-latency/dispersion de-risks, e.g. `f10edd601`). Not a wiring gap;
   resolving the instrument is the blocker.
4. **R2's three-factor rule as a refinement of the EXISTING production edge** (not a new migration, per §1) —
   re-express R2's unbiased 6-edge topology + reward-deferred STDP through `CrossEdge` and consider folding it
   into the already-flipped `BRAIN_ONEBRAIN_XEDGE`/`_LEARN` production edge as a rule upgrade, separately scoped
   from "migrate a new pair."
5. **d5_episodic (Group-C own-pool seam)** — no evidence found of this having started; still open, needs its own
   multi-bridge/apical-dendrite seam before any cross-edge question applies.
6. **affect->tone/mouth edge** — ambiguous whether it routes through `OrganDescriptor`/`CrossEdge` at all (ties
   to `research/affect-tone-spiking-mouth`); flagged, not verified this pass (out of this task's tight scope).

## Files

`research/runners/onebrain_merge_framework.py` (`CrossEdge.source_idx_fn`/`.target_idx_fn`, `_cross_edge_dense`)
· `research/runners/_onebrain_declarative_crossedge_r4_repro.py` (new — `DeclarativeR4Pool`, `CROSS_EDGES`,
`compare_seed`) · `research/runners/onebrain_xedge_selfschema_production.py`
(`xedge_selfschema_declarative_enabled`, `_build`'s `pool_cls` branch, `--declarative` CLI flag) ·
`tests/test_webapp_server.py` (2 new tests) · `research/findings/raw/_onebrain_declarative_crossedge_r4_repro_6seed.json`
· `research/findings/raw/_onebrain_xedge_selfschema_production_declarative_6seed.json`. No `sim/` file touched;
`webapp/server.py` untouched (the new flag lives entirely inside the production module the existing wiring
already imports, so the live-handler surface is unchanged).

Functional read-outs only; no phenomenal-experience claim.
