---
type: finding
status: scoping
date: 2026-08-27
mechanism: onebrain-production-integration
lane: onebrain-integration
---

# One-brain PRODUCTION integration — SCOPING: what it takes to make the de-risked learned cross-region edges (R1/R3-v3/R4) actually DRIVE each other in the LIVE chat brain

Status: SCOPING (a plan + a file:line trace of the live path; NO code changed, NO default flipped, NO
`sim/`/`webapp/` edit). The de-risked cross-edges (R1 d6-WM→comprehension, R3-v3 DA-credit→comprehension, R4
self_schema→source_provenance) all live on the RESEARCH merge FRAMEWORK (`onebrain_merge_framework.py` +
the `_onebrain_integration_r*` runners), default-off, NOT in the live brain. This doc answers: does the live
brain use the framework (Q1); the gap to it (Q2); the smallest safe first wire-in (Q3); the blockers (Q4).

## Q1 — HOW the live production brain is built, and does it use the merge framework?

Answer: **NO. The live chat brain does NOT use `merge_organs()` and hosts NONE of the R1/R3-v3/R4 organs on a
shared pool.** Every one of the four organs builds its OWN standalone `SimulationBridge`, so there is no shared
substrate a cross-region synapse could span. Trace:

- The shared turn pipeline is `webapp.server.brain_reply(chat, req, source, cache_key)` (`webapp/server.py:4106`),
  reached by the HTTP handler `brain_chat` (`:4008`), the OpenAI shim (`:6011`), and the TUI via the thin surface
  `webapp/brain_reply.py` (`reply_over_chat` → `server.brain_reply`). One pipeline, three callers.
- The brain object is built by `_build_chat_brain` (`webapp/server.py:3621`): for the default `tiny-demo` it calls
  `brain_chat_tui._build_tiny_demo(42, ..., composer_kind=_ck)` (`:3661`) with
  `_ck = os.environ.get("BRAIN_COMPOSER_KIND", _COMPOSER_KIND_DEFAULT)` and `_COMPOSER_KIND_DEFAULT="onebrain"`
  (`:3573`). This returns a `ChatBrain` wrapping a `BrainConversationalAgent` (`:3721`).
- CRUX TERMINOLOGY TRAP: `composer_kind="onebrain"` is NOT the merge framework. It selects the genuinely-SPIKING
  RECALL COMPOSER (resonate-and-fire per query, `brain_chat_tui.py:1500-1506`, `:1553-1563`) — a different sense of
  "one brain" (one recall substrate) than `merge_organs()`/`MergedPool` (many organs, one bridge). `grep merge_organs
  webapp/` returns ZERO. The webapp never imports the framework's `merge_organs`/`OrganDescriptor`/`MergedPool`.
- The faculty organs are attached per-turn as SEPARATE modules via `get_organ()`, each building its own bridge:
  comprehension `_get_comprehension_organ` (`:2970`) → `comprehension_production_organ.get_organ` (`:483`, builds
  `ComprehensionProductionOrgan(seed)` — `shared=None`); d6 `_get_multiref_organ` (`:3303`) →
  `MultiReferentWMOrgan(seed=42)` (`d6_multiref_wm_production_organ.py:259`, `shared=None`); self_schema
  `_get_self_schema_organ` (`:3013`) → `SelfSchemaAuthorshipOrgan(seed)` (`self_schema_production_organ.py:167`,
  `shared=None`); source_provenance `_get_source_provenance_organ` (`:3000`) → `SourceProvenanceHonestyMonitor`
  (`source_provenance_honesty.py:63`). Four organs, four bridges, zero cross-region synapses.

TWO production MERGE POOLS DO exist and ARE wired into the live path — but neither hosts an R-organ and neither has
ANY cross-synapse (they are byte-identical CO-RESIDENCY, the exact "co-resident but not interacting" scaffold):

- Pool #1 (`onebrain_merge_production.py`, `merge_enabled` `:162`, `_MERGE_DEFAULT_ON=True` `:159`) — surprise +
  world-model + the recall composer + the parser share one bespoke `MergedSubstrate`. Its own docstring: "NO
  cross-organ synapse is added on this rung" (`:24`). The worldmodel organ reads it (`worldmodel_production_organ.py:256`).
- Pool #2 (`onebrain_merge_production2.py`) — metacog + pragmatic. As of 2026-08-27 it is "RETIRED-TO-A-SHIM":
  `MergedSubstrate2.ensure_built()` DELEGATES to `merge_organs([METACOG, PRAGMATIC], seed, wire=True)`. So the
  FRAMEWORK is already load-bearing in production — but ONLY for metacog+pragmatic, and with NO cross-synapse. The
  metacog organ attaches via `get_organ`: `shared = get_merged_substrate2(seed) if merge2_enabled() else None`
  (`metacog_production_organ.py:311-313`) — the exact template Q3 reuses.

## Q2 — the GAP between the framework pool (where R1/R3-v3/R4 live) and the production brain's pool

The R-cross-edges require their source+target organs CO-RESIDENT on ONE `MergedPool` so a synapse can span two
slices of one `cp_connections`. Production keeps them on separate bridges. Concretely, three gaps:

1. **No pool hosts the R-organs.** The framework `REGISTRY` (`onebrain_merge_framework.py:2067`) registers d6,
   comprehension, self_schema, source_provenance (all `supports_shared=True`, `organ_cls` bound to the shipped
   production classes), and the R-runners build `MergedPool([D6, COMP, DA], wire=True)` etc. But production's pool #1
   (surprise/worldmodel) and pool #2 (metacog/pragmatic) host none of them. To wire a cross-edge, production must
   build a MergedPool that instantiates the relevant R-organs and pass `shared=pool` into each `get_organ`.
2. **The cross-edge itself is NOT declarative yet.** `OrganDescriptor` has NO `cross_edges` field and `merge_organs`
   has NO `cross_edges=` param (grep: zero matches in the framework). The DESIGN doc's
   `cross_edges: [(src,dst,w0,rule,gate)]` (`2026-08-27-onebrain-integration-phase-DESIGN.md` §2) is an ASPIRATION.
   Today each R-runner hand-wires the cross-edge in its own `_build_pool` via `inject_explicit_wiring` (R3-v3:
   `_onebrain_integration_r3_spiking_dopamine_credit.py:189-222`) plus a whitelisted plasticity gate. So the pool
   config/organs are framework code, but the cross-edge wiring lives in a research runner, not in any production path.
3. **Lifecycle + read-protocol mismatch.** In production d6 is PER-SESSION (`_get_multiref_organ(cache_key)`,
   `server.py:3303`, one bridge per conversation) while comprehension is PROCESS-SHARED (one `_ORGAN`,
   `comprehension_production_organ.py:480`). A shared pool cannot be both. Also the R-runners drive a DIAGNOSTIC
   battery (`amb_read`, the balanced-cue ambiguity margin) whose credit-edge is GROWN over ~200 host-teacher episodes
   (`teach_agent`/`teach_patient` drives, R3-v3 runner); the live turn has neither that teacher schedule nor that
   protocol — so a first wire-in must LOAD a pre-grown FROZEN edge, not learn it live.

Migration status is good where it is closed: substrate-init + organ-read byte-identity is 6-seed GO for the six
frozen-forward organs on ONE pool (`2026-08-27-onebrain-merge-framework-organ-read-*` findings). The missing piece
is exclusively the cross-edge wiring INTO the live brain, plus the pool that co-locates the organs there.

## Q3 — the SMALLEST safe first wire-in (recommended: R3-v3 d6-WM+DA-credit → comprehension)

Recommendation: **R3-v3** first. Its two organs are BOTH shipped classes that already accept `shared=`
(`ComprehensionProductionOrgan.__init__(seed, shared=None)` `comprehension_production_organ.py:241`;
`MultiReferentWMOrgan.__init__(seed, shared=None)` `d6_multiref_wm_production_organ.py:142`), and the cross-edge
DRIVES the SAME `read_margin` (`:368`) that the live `judge()` (`:399`) consumes — so the drive is genuinely
conversation-load-bearing (a biased margin flips `comprehended`, which the pipeline already acts on:
`not_understood` repair path `server.py:5180`). R4 is worse-positioned (see Q4): its production wrapper does not
plumb `shared=`.

<!--derived-->
(the numeric floors below — `F2_INTACT_FLOOR=0.008`, the ~0.32 well/ill comprehension threshold — are restatements
of values from the R3-v3 finding + `2026-08-27-onebrain-integration-phase-DESIGN.md` §1, not new measurements; they
are the pre-registered gate floors the live wire-in would re-verify against, not results this scoping doc produced.)

Additive, DEFAULT-OFF, byte-identical-off shape (no existing behavior changes unless the flag is set):

- **Flag.** `BRAIN_ONEBRAIN_XEDGE` (env, default OFF; mirror `merge2_enabled()` style,
  `onebrain_merge_production2.py`). OFF ⇒ every organ builds standalone exactly as today (byte-identical). A new
  production module `research/runners/onebrain_xedge_production.py` owns `xedge_enabled()` + a process-shared
  `get_xedge_pool(seed)` that builds `merge_organs([D6, COMP], wire=True)` and hand-wires the FROZEN, pre-grown
  `w{k}→sel_agent/sel_patient` edge (reusing R3-v3's `_build_pool` wiring, edge plasticity gate set to 0 after load).
- **The two get_organ attach points** (the ONLY production edits, both the metacog template):
  `comprehension_production_organ.get_organ` (`:483`) → `shared = get_xedge_pool(seed) if xedge_enabled() else None`;
  `d6_multiref_wm_production_organ.get_organ` (`:259`) → same. Nothing else in `webapp/server.py` changes — the
  pipeline already calls both `get_organ`s. (d6's per-session `_get_multiref_organ` must be reconciled — see Q4.)
- **Verification (the drive-couplings discipline, memory #84/#85).** (i) LIVE load-bearing: a turn holding
  referent-0 in d6 vs referent-1, on the same ambiguous transitive, must move the comprehension `read_margin`
  toward the held referent (signed Δ ≥ the R3-v3 `F2_INTACT_FLOOR=0.008` <!--derived-->); (ii) LESION: zero the `w{k}→sel` edge
  in the live turn → Δ→~0 (the change vanishes). (iii) F1 no-regression: comprehension still separates well/ill
  (margin ≥ ~0.32 on well items) and d6 still recovers all referents. (iv) F4 moat: a no-cue item stays undecided
  regardless of WM; a CLEAR-cue item is not flipped. (v) BYTE-IDENTICAL-OFF: `BRAIN_ONEBRAIN_XEDGE=0` vs unset ⇒
  identical reads across the existing chat smoke. (vi) lesion-recovers-migration: with the edge lesioned the pool
  equals the migrated (no-cross) pool.
- **RISKS to the live brain.** (a) Instability/runaway from the added recurrence — bounded by the frozen (gain-0)
  edge on the first rung + `hebbian_max_weight` if later grown live (F3). (b) Moat leak — a WM bias manufacturing a
  false-accept; F4 is the instrument, and the edge only reweights genuine ambiguity. (c) LATENCY — a co-resident
  pool steps more neurons per read; speed is secondary per mission but a slow first turn could regress UX (measure).
  (d) A build failure in `get_xedge_pool` must DEGRADE to standalone (guard like the tiny-demo LTM attach,
  `server.py:3685`), never crash brain load. (e) Determinism: the pool's per-region seams must reproduce each organ's
  standalone init (the 6-seed migration GO is the evidence; re-verify on the shipped classes).

## Q4 — BLOCKERS (what makes this NOT a clean flag flip today)

1. **No pool co-locates the R-organs in production, and the cross-edge is non-declarative.** A production
   `get_xedge_pool` + the hand-wired frozen edge (Q2.2) must be BUILT first — it does not exist. This is new
   research-runner code (allowed), not a `sim/`/`webapp/` edit beyond the two `get_organ` lines.
2. **d6 lifecycle mismatch (per-session vs process-shared).** Comprehension is one process singleton; d6 is one
   bridge per `cache_key` (`server.py:3303`). A shared d6+comprehension pool forces a choice: make the pool
   per-session (comprehension loses its singleton; more memory), or make d6 process-shared (loses per-conversation
   referent isolation — the `_SESSION_MULTIREF` guard exists precisely to stop one chat's referents leaking into
   another, `server.py:3306`). Reconciling this is the main design decision the first wire-in must make.
3. **R4 production class is not merge-capable.** `SourceProvenanceHonestyMonitor.__init__(seed, *, lesion)`
   (`source_provenance_honesty.py:74`) builds `ProvenanceBrain(self.seed)` with NO `shared=` — while the framework's
   R4 organ is `_SourceProvReadOrgan`→`ProvenanceBrain(seed, shared=...)` (`onebrain_merge_framework.py:1227`), a
   DIFFERENT wrapper. So R4's live wire-in additionally needs the production provenance wrapper extended to forward
   `shared=` (a production edit). This is why R3-v3 is the recommended first edge, not R4.
4. **No live credit/learning signal.** R3-v3 GROWS the edge from host teacher drives; live chat has none. The first
   rung must ship a PRE-GROWN FROZEN edge (drive without live learning). Making the edge learn from raw dialogue is a
   later rung (the DESIGN R2 three-factor / neuromod-seam item), not a blocker for the first DRIVE.
5. **NOT a prerequisite:** retiring the bespoke `MergedSubstrate*` is NOT required first. Pool #2 already proves a
   framework pool runs in production beside them; a new xedge pool can coexist under its own flag, and the pools
   retire later once every cross-edge holds its F-gate (DESIGN §4 production-flip criterion).

## Files (this scoping — read-only, no code changed)

This doc PRODUCES no measurement of its own; the artifacts below are the EVIDENTIARY BASIS its premise rests on
(that R1/R3-v3/R4 are de-risked GO on the framework), cited so the claim "already de-risked" is traceable:

- `research/findings/raw/_onebrain_integration_r1_wm_comprehension_6seed.json` — R1 (d6-WM→comprehension) 6-seed GO.
- `research/findings/raw/_onebrain_integration_r3v3_functional_drive_6seed.json` — R3-v3 (DA-credit→comprehension
  functional drive) 6-seed GO; the recommended first wire-in.
- `research/findings/raw/_onebrain_integration_r4_selfschema_provenance_6seed.json` — R4
  (self_schema→source_provenance) 6-seed GO.
- `research/findings/raw/_onebrain_integration_crossedge_smoke_6seed.json` — the learned-cross-edge feasibility smoke.

- `research/findings/2026-08-27-onebrain-production-integration-SCOPING.md` — this doc.
- Live path traced: `webapp/server.py` (`_build_chat_brain:3621`, `brain_reply:4106`, the four `get_organ`
  attach points `:2970`/`:3000`/`:3013`/`:3303`, comprehension consume `:5132`), `webapp/brain_reply.py`,
  `research/runners/brain_chat_tui.py` (`_build_tiny_demo:1500`).
- Framework + R-edges: `research/runners/onebrain_merge_framework.py` (`merge_organs:479`, `REGISTRY:2067`),
  `_onebrain_integration_r3_spiking_dopamine_credit.py` (`_build_pool:170`), the R1/R3-v3/R4 findings +
  `2026-08-27-onebrain-integration-phase-DESIGN.md`.
- Production pools: `onebrain_merge_production.py` (`merge_enabled:162`), `onebrain_merge_production2.py`
  (framework-backed shim), `metacog_production_organ.py:311` (the shared-attach template).
