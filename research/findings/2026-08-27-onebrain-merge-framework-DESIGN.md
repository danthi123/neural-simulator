---
type: finding
status: live
lane: onebrain-merge
date: 2026-08-27
---

# One-brain merge FRAMEWORK — a declarative, batched N-organ merge engine (DESIGN)

Status: DESIGN (not a result; no GO claimed). Scope: a plan + a prototype skeleton
(`research/runners/onebrain_merge_framework.py`), NOT the full migration. Owner-raised: the one-brain
merge is done BESPOKE, one organ at a time — O(N) hand-written code + O(N) verify cycles for ~20 remaining
organs. This design replaces the per-organ code with ONE declarative registry + ONE parameterized verify.

## 0. What exists today (grounding — read before trusting the reframe)

Production on `main` runs TWO bespoke merged pools, each a hand-written class:

- Pool #1 `onebrain_merge_production.py:172` `MergedSubstrate` — surprise + world-model on one bridge.
  Hardcoded organ build-kwargs `_SURPRISE_KW`/`_WORLDMODEL_KW` (`:58`,`:60`); config superset inlined in
  `ensure_built` (`:230`-`:262`); `if "surprise" in self.organs:` region-append branches (`:266`-`:324`);
  block-diagonal wiring hand-called (`:337`-`:343`); `read_isolation` snapshot/restore (`:416`); the
  parser-on-pool bind `_bind_parser_onto_pool` (`:642`) uses `cp_plasticity_rate_gain=0.0` as a per-synapse
  gain-0 FREEZE primitive.
- Pool #2 `onebrain_merge_production2.py:98` `MergedSubstrate2` — metacog + pragmatic. A SEPARATE class
  because its global config conflicts (param-het ON vs OFF); `_metacog_specs`/`_pragmatic_specs` (`:117`,
  `:133`); `build_wiring_plan` union + assembly loops (`:208`-`:219`); full-snapshot restore for isolation.

Four organs merged (surprise, world-model, metacog, pragmatic). Confirmed by code: exactly those four carry a
`self._shared` branch in `ensure_built` (grep over all 26 `*_production_organ.py`); the other 22 are
own-bridge-only. Each organ takes `shared=None`; when a substrate is injected it reads its region slice off
`shared.bridge` + `shared.<organ>_idx_map()` and wraps reads in `shared.read_isolation(name)`
(`surprise_production_organ.py:203`,`:339`; `metacog_production_organ.py:243`).

A GENERALIZATION already exists on a research branch (NOT on main), and it is the seed of this framework:

- `_onebrain_twopool_merge_derisk.py` (branch `research/onebrain-twopool-merge`): `build_pool(seed, organs)`
  builds ONE superset bridge for ANY subset of the 4 organs from a generic `ORGAN_REGIONS` map + per-organ
  `_organ_specs`, reconciling the pool-1/pool-2 config conflict with per-region SEAMS
  (`per_region_parameter_heterogeneity`/`_threshold_heterogeneity`/`_wiring_seed`/`_homeostasis_isolation`)
  masked BY REGION NAME. Verdict: 6/6 GO substrate-init byte-identity (all 4 organs merged-vs-co-resident,
  `_onebrain_twopool_merge_6seed.json`).
- `_onebrain_twopool_organread_verify.py`: `MergedSubstrate4` runs all 4 organs' REAL read pipelines on the
  single pool + a GENERIC gain-0 freeze over any edge internal to a frozen region-set (`:163`-`:184`) +
  name-keyed `read_isolation` over N organs (`:262`). Smoke GO (byte-identity + answer-preservation +
  gain-0-frozen for all 4). This file is 90% of the engine already — just not packaged as a registry.

So the migration is O(N) bespoke today, but the twopool branch proves the declarative form works for 4
organs. The framework packages that form and scales it.

## 1. Organ descriptor schema

The minimal declarative record to register any organ for merge. Everything the bespoke `MergedSubstrate*`
did by hand becomes a field. A descriptor is pure data + a few small callables reused BY IMPORT from the
organ's existing de-risk builder (no new mechanism).

```python
@dataclass(frozen=True)
class OrganDescriptor:
    key: str                         # "surprise" — stable id, keys the per-region name RNG + isolation mask
    regions: tuple[str, ...]         # region NAMES this organ owns (must be DISJOINT across the pool)
    spec_fn: Callable[[int], tuple]  # seed -> (regions, pathways, meta); reuse-by-import of the de-risk builder
    config: dict                     # cfg field -> value this organ REQUIRES (unioned; conflict = engine error)
    region_flags: dict = None        # per-region BrainRegion overrides: {"workspace": {"enable_nmda": True}, ...}
    post_build: Callable = None      # (bridge, meta) -> None; topographic wiring AFTER init (block-diag / loops)
    freeze_regions: tuple = ()        # regions whose INTERNAL edges get a permanent cp_plasticity_rate_gain=0
    isolation: str = "per_slice"     # "per_slice" (snapshot/restore this slice) | "full_snapshot" (pool-2 style)
    idx_fn: Callable = None           # (bridge) -> the idx map/dict the organ's shared= read path consumes
    read_fn: Callable = None          # (organ_instance) -> dict of numeric reads (the byte-identity battery)
    answer_fn: Callable = None        # (organ_instance) -> the rendered chat-answer(s) (answer-preservation)
    organ_cls: type = None            # the shipped *_ProductionOrgan class (constructed with shared=<pool>)
```

Field-by-field mapping to today's bespoke code (each field REPLACES a hand-written thing):

| field | replaces (bespoke) | source it reuses |
|---|---|---|
| `regions` | `_SURPRISE_REGIONS` / `_METACOG_REGIONS` constants | the organ's region names |
| `spec_fn` | `build_expectation_circuit(...)` throwaway + `_metacog_specs()` | the de-risk builder, imported |
| `config` | the inlined `cfg.*=` block in `ensure_built` | one dict per organ, unioned |
| `region_flags` | `r.enable_nmda=`/`r.enable_homeostasis=` name loops | `build_pool` `:171`-`:176` |
| `post_build` | `_install_block_diagonal` x3 / assembly-loop union | `surprise`/`metacog` builder |
| `freeze_regions` | `_bind_parser_onto_pool` gain-0 + pool-2 freeze | the gain-0 primitive |
| `isolation` | `read_isolation` vs `_restore_state` | the two existing protocols |
| `idx_fn` | `surprise_idx_map()` / `metacog_idx()` | the organ's shared read contract |
| `read_fn`/`answer_fn` | `_surprise_reads` / `_metacog_reads` / notice strings | the verify batteries |

Three real organs as concrete descriptors (abbreviated; the prototype registers the first two in full):

```python
SURPRISE = OrganDescriptor(
    key="surprise",
    regions=("cue", "patient_expected", "patient_asserted", "surprise"),
    spec_fn=lambda s: _specs_from(build_expectation_circuit(s, per_region_thresh=True, **_SURPRISE_KW)),
    config={"per_region_threshold_heterogeneity": True, "per_region_homeostasis_isolation": True,
            "enable_hebbian_learning": True, "hebbian_max_weight": 45.0, "enable_homeostasis": True,
            "enable_gabab": True, "gabab_conductance_max": 0.0},
    post_build=lambda b, m: _install_surprise_block_diagonal(b, m),   # 3 block-diagonal calls
    isolation="per_slice", idx_fn=lambda b: _name_idx(b, SURPRISE.regions),
    read_fn=_surprise_reads, answer_fn=lambda o: o.judge("a","acts","beta","gamma")["surprised"],
    organ_cls=SurpriseProductionOrgan)

METACOG = OrganDescriptor(
    key="metacog",
    regions=("workspace", "workspace_fs", "meta_schema"),
    spec_fn=_metacog_specs,                                            # imported from the derisk
    config={"per_region_parameter_heterogeneity": True, "per_region_wiring_seed": True,
            "enable_parameter_heterogeneity": True, "enable_nmda": True, "enable_homeostasis": False},
    region_flags={"workspace": {"enable_nmda": True, "enable_homeostasis": True},
                  "meta_schema": {"enable_nmda": True}},
    post_build=lambda b, m: _install_assembly_loops(b),               # K dense self-recurrent loops
    freeze_regions=("workspace", "workspace_fs", "meta_schema"),      # gain-0 vs surprise's shared Hebbian
    isolation="full_snapshot", idx_fn=lambda b: _metacog_idx(b),
    read_fn=_metacog_reads, answer_fn=lambda o: o.judge(0.2)["confident"],
    organ_cls=MetacogProductionOrgan)

# PRAGMATIC — identical shape to METACOG (item/item_fs, no NMDA, item<->item_fs pathways, full_snapshot).
```

The descriptor is the ENTIRE per-organ cost. A trivial organ (frozen op-point, disjoint names, no global
conflict) is ~12 lines with no new callable. The design intent: adding an organ should be a registry ROW,
not a class.

## 2. Merge engine API

```python
def merge_organs(descriptors: list[OrganDescriptor], seed: int = 42,
                 backend: str | None = None) -> MergedPool: ...
```

`MergedPool` exposes exactly the surface the shipped organs already expect from a `shared=` substrate:
`.bridge`, `.cfg`, `.xp`, `.snap`, `.ensure_built()`, `.read_isolation(key)`, and per-organ `idx` accessors
(dispatched by `key` to the descriptor's `idx_fn`). So a shipped organ is constructed UNCHANGED as
`desc.organ_cls(seed=seed, shared=pool)` — the injection contract is already universal (every merged organ
takes `shared=`); the engine just makes the substrate generic.

Build algorithm (generalizes `MergedSubstrate.ensure_built` + `build_pool` to an N-list):

1. SPEC EXTRACTION — `for d: regions, pathways, meta[d.key] = d.spec_fn(seed)`. Reuse-by-import; throwaway
   bridges are fine because every seam keys on region NAME (crc32), so init is co-residence + RNG-order
   invariant. UNION regions/pathways in descriptor order (order only affects index base, not per-neuron init).
2. NAME-DISJOINTNESS CHECK — assert the union of `d.regions` has no dup (a rename is forbidden: the seams key
   on the name, so a collision changes a slice's init — the exact reason affect is scoped to its own pool,
   `onebrain_merge_production2.py:19`-`:25`).
3. CONFIG UNION — start from `BASE_CONFIG` (dt=1, IZHIKEVICH, GENERIC_UNSTRUCTURED, all the always-on seam
   flags), then fold each `d.config`. A key set to two DIFFERENT values is a genuine conflict → raise
   `MergeConflict(key, {d1:v1, d2:v2})`. This is where a global-config incompatibility surfaces LOUDLY at
   registration instead of silently corrupting a slice (param-het ON vs OFF; OU on vs off).
4. PER-REGION FLAGS — apply `d.region_flags` onto the matching `BrainRegion` (the diffbuilder pattern:
   `enable_homeostasis`/`enable_nmda`/`enable_heterogeneity` per region reconciles a global conflict into a
   masked one, `build_pool:171`).
5. BUILD — one `SimulationBridge`, `_initialize_simulation_data`. Per-region seeding is ALREADY generic in the
   engine (the seams read region names); nothing per-organ here.
6. POST-BUILD WIRING — `for d: d.post_build(bridge, meta[d.key])` (block-diagonal, assembly loops), in
   descriptor order, exactly as the bespoke code sequences it.
7. GAIN-0 FREEZE — union each `d.freeze_regions`; set `cp_plasticity_rate_gain=0` on every edge with BOTH
   endpoints in a frozen region (the generic form is already in `organread_verify:163`-`184`; assert no edge
   has EXACTLY one endpoint in a frozen region = no unintended cross-synapse). Gain-1 elsewhere is
   byte-identical to the ungated scalar path.
8. SNAPSHOT — `_rest_v/_rest_u` (per_slice organs) and a settled `_snapshot_state` (full_snapshot organs).

What becomes the engine, and what must be refactored:

- BECOMES the engine verbatim (promote from the twopool branch): `build_pool` (steps 1-5), the gain-0 freeze
  loop (step 7), the name-keyed `read_isolation`/`_keep_mask` (`MergedSubstrate4`). These are already generic.
- REFACTOR out of the bespoke classes into descriptor callables: the `if "surprise" in organs:` region
  branches (`MergedSubstrate:266`-`324`) → each organ's `spec_fn`+`region_flags`; the inlined block-diagonal
  (`:337`) → `post_build`; the two isolation protocols → the `isolation` field selecting the existing
  mechanism. NO new mechanism is invented — the refactor MOVES code, it does not add behavior.
- STAYS in `sim/` untouched: every seam flag + the gain-0 primitive already exist on `main` (guarded). The
  framework is pure `research/runners/` glue, exactly like the bespoke merges.

## 3. Batched verify — one runner, all organs x 6 seeds

`_onebrain_twopool_organread_verify.py` IS the prototype of this — it already loops `ALL_ORGANS`, runs each
organ's real read on the shared pool, and gates byte-identity + answer-preservation + gain-0-frozen + a
legacy discriminator, per seed. Generalize it to read the REGISTRY instead of a hardcoded 4-tuple:

```python
# research/runners/onebrain_merge_verify.py  (parameterized; pool/gpu-queue routable)
def verify(keys: list[str], seeds: list[int]) -> dict:
    descs = [REGISTRY[k] for k in keys]
    for seed in seeds:
        pool   = merge_organs(descs, seed)                 # the merged N-organ substrate
        for d in descs:
            merged = d.organ_cls(seed=seed, shared=pool)   # shipped organ on the pool
            base   = d.organ_cls(seed=seed, shared=coresident_pool(d, seed))  # organ alone, same superset cfg
            byte[d.key]   = max_delta(d.read_fn(merged), d.read_fn(base)) == 0.0
            answer[d.key] = d.answer_fn(merged) == d.answer_fn(base)
        gain0_ok = freeze_structural_and_bit_frozen(pool)  # gain array 0/1 exact + pool edges byte-frozen
        legacy   = seams_off_diverges(descs, seed)         # discriminator: not vacuous
    ...
```

Key properties, all inherited from the working organread verify:

- The BASELINE per organ is the organ ALONE on the SAME superset config (`coresident_pool`), so a non-zero
  delta isolates CO-RESIDENCE, not a config change — the standard the 2-organ and 4-organ verdicts used.
- ONE sweep gates the WHOLE registry: `--keys all --seeds 42,43,44,100,101,102`. A new organ adds a row to
  the sweep, not a new runner. This is the O(N)→O(1) win: one verify code path, N descriptors, batched.
- ROUTABLE: numpy CPU bit-exact (`SIM_BACKEND=numpy`) makes it a `tools/sweep_pool.sh` (mini-PC) job for the
  small organs; the large ones (vision/perception) route to `tools/gpu_queue.sh`. Multi-seed is
  controller-fanned `--seeds`. Zero agent tokens (cost-routing).
- The legacy discriminator (seams OFF → diverges) stays, so the compare is never vacuous (the exact anti-cheat
  `build_pool` already carries).

## 4. The integration phase — what replaces byte-identity

Byte-identity-in-isolation is the MIGRATION-SAFETY gate. It is deliberately CONSERVATIVE — and it FORBIDS
the interaction that is the one-brain GOAL. Two levels must not be conflated:

- (1) MIGRATION: co-locating an organ on the shared substrate does not BREAK it → byte-identity in isolation.
  This is the expensive, necessary, BULK gate. Run it ONCE per organ (registration), batched.
- (2) INTEGRATION: organs SHARE the substrate and INTERACT via cross-region synapses → emergent cross-faculty
  behavior. This is the actual "one brain". Byte-identity-in-isolation cannot hold here BY DEFINITION: a
  cross-synapse means organ B's state now depends on organ A, so B's read is no longer identical to B-alone.

So the design uses byte-identity as a BULK MIGRATION GATE (batched, once), then SWITCHES the validation to
FUNCTIONAL once an organ is on the pool. The integration gate (sketch):

```
FUNCTIONAL-INTEGRATION gate (per cross-region edge added):
  F1 FACULTY-STILL-WORKS: each organ's own end-to-end task metric stays >= its standalone GO bar
     (surprise 22.8x confirm/contradict separation; metacog type2_auc>=0.67; pragmatic implicature margin).
     Co-residence may perturb numbers but must NOT cross the organ's decision boundary on its own task.
  F2 INTERACTION-IS-REAL (load-bearing, both directions):
     - vary the SOURCE organ's state; the TARGET organ's read/answer must demonstrably CHANGE.
     - LESION the cross-synapse (zero the edge); the change must VANISH (the coupling caused it, not a
       confound). This is the anti-hollow test from the memory: faculties must DRIVE, not observe.
  F3 NO-RUNAWAY: the added recurrence does not destabilize the pool (firing-rate homeostat bound; the pool
     stays alive across a long multi-turn stepping burst — the read_isolation stress case).
  F4 MOAT/HONESTY preserved: an interaction never manufactures a fact, flips an abstain, or asserts
     phenomenality (the standing additive+honest constraint every organ already carries).
```

F2 is the crux and is already a project-native check (memory: "vary the faculty state, response must differ +
the diff must vanish when the coupling is lesioned"). The first integration edges are named in the org docs
(recall→surprise; affect→tone; swap→focus) — the framework's job is to make ADDING such an edge a descriptor
field (`cross_edges: [(src_region, dst_region, weight, plastic)]`) gated by F1-F4, not bespoke wiring.

Honesty caveat (the deeper scaffold note): the block-diagonal masks, assembly loops, and gain-0 freezes are
THEMSELVES host scaffold — hand-declared connectivity. The faithful end state has regions that DEVELOP their
connectivity + interact through LEARNING (self-organized cross-region synapses via the substrate's own
Hebbian/STDP/BTSP), not a hand-set mask. So this framework is a SCAFFOLD-ACCELERATED migration: cheap +
declarative to get all organs co-resident and interacting NOW, with the hand-wiring flagged for
self-organization burn-down (see §6). The framework must not let "declarative merge" ossify into a permanent
host-wired connectome.

## 5. Migration plan — remaining organs, difficulty-grouped

22 organs are own-bridge-only. Grouped by what the engine needs (config-conflict markers grepped from each
organ file; "neuromod?" flags need per-organ confirmation that the GLOBAL flag is required vs merely
referenced):

GROUP A — declarative-NOW (frozen/quiescent op-point, small net, disjoint names, config in the microcircuit
family the engine already reconciles). Register as a descriptor row; expected to pass batched byte-identity
with NO engine change:

| organ | note |
|---|---|
| comprehension | margin/competence read; microcircuit family |
| b3_noncontradiction | yes/no check; frozen |
| causal_whatif | what-if/why; confirm neuromod ref is inert |
| d5_episodic | topic/recall/recency; needs a `shared=` kwarg added first (0 refs today) |
| d6_multiref_wm | load/judge WM; has shared plumbing |
| source_provenance | provenance read; confirm neuromod ref inert |
| self_schema | author read; small |
| affective_tom | other-tone read; small |
| d3_discourse_event_register | observe; small |
| reconsolidation | window/reconsolidate; has shared plumbing |
| prospective_memory | intention/cue; needs `shared=` kwarg added |
| repair | needs `shared=` kwarg + a read entrypoint surfaced |
| curiosity | judge; confirm neuromod ref inert |

GROUP B — needs a NEW engine SEAM (one config field generalized), then declarative. Each adds ONE capability
to the config-union/region-flags machinery, reused by all later organs:

| organ | engine extension needed |
|---|---|
| activity_silent_wm | short-term-facilitation seam: per-region `enable_short_term_plasticity` + `cp_stp_u` isolation in the snapshot set |
| vocab_agnostic_spiking_generation | global-OU seam: the soft-WTA bank needs `enable_ou_process=True` + `ou_seed` per-region; co-resident with the composer slice |
| bg_action_selection | OU + salience-excitability-bias seam; two-channel race (a real commit, not a read) — verify F-gate, not byte-identity |
| sc_orienting | perception/retina drive (environment boundary) + retinotopic sheet; visuomotor read pool |

GROUP C — needs a WHOLE-SUBSYSTEM merge (own pool, not the shared microcircuit pool). Engine supports MULTIPLE
named pools; these get their own:

| organ | why its own pool |
|---|---|
| affect | build_one_brain whole co-resident brain; GLOBAL OU + neuromod; HARD name collision (workspace/workspace_fs/meta_schema vs metacog) — rename forbidden (seams key on name). Documented structural blocker, `production2.py:19` |
| value_choice | striosome_value + navcritic; DA-gated STDP + OU; a LEARNED value subsystem |
| vision_identity | Gabor/V1 front-end + coincidence-column pooler; large net, distinct neural profile; route to GPU |
| v1_selforg | self-organized V1 RF bank; develops weights via on-bridge STDP; transplant onto the perception pool |

Ordering (cheapest-first, each rung reused by the next):

1. GROUP A batch: register all ~13 as descriptors, run ONE `--keys A... --seeds x6` sweep on the pool. This
   is the immediate O(N)→O(1) payoff — ~13 organs migrated in one verify cycle instead of 13.
2. Fold the 4 already-merged organs (surprise/world-model/metacog/pragmatic) into the SAME registry, retiring
   `MergedSubstrate`/`MergedSubstrate2` to a thin `merge_organs([...])` call (proves the framework
   reproduces production byte-identically — the twopool 6/6 GO is the evidence it can).
3. GROUP B: add the three seams (STP, OU, perception-drive) one at a time, each unlocking its organ(s).
4. GROUP C: multi-pool support + the whole-subsystem organs; affect/value on their own pools.
5. INTEGRATION phase (§4): begin adding cross_edges under the F-gate, starting with the named first edges.

Honest estimate: ~13 declarative-now, ~4 need one engine seam each, ~4 need a subsystem/own-pool. The bespoke
cost was ~200 lines + a 6-seed verify PER organ; the framework makes GROUP A ~12 lines + a shared sweep.

## 6. Risks + the scaffold-retirement note

- HOST-SCAFFOLD, flagged for retirement (do NOT let the declarative merge ossify it): (a) the block-diagonal
  masks + assembly loops + per-region seams are HAND-DECLARED connectivity — the faithful end state
  self-organizes them via the substrate's own plasticity; (b) the gain-0 freeze is a host clamp standing in
  for a real E/I / homeostatic balance that would let the pool-2 edges coexist with shared Hebbian WITHOUT a
  freeze; (c) `read_isolation`'s snapshot/restore is a host correction for a spontaneous-firing coupling that
  a biological pool would damp intrinsically. Each is a documented residual, not a hidden shortcut — the
  framework should carry a `scaffold_residuals` field per descriptor so the burn-down is tracked, not lost.
- THE REFRAME'S RISK: byte-identity is necessary for migration but INSUFFICIENT for the goal, and worse, it is
  ANTAGONISTIC to the goal (it forbids interaction). The danger is declaring "one brain" at bulk byte-identity
  while every organ is still an ISLAND. Mitigation: the §4 F-gate is a FIRST-CLASS deliverable, not a
  follow-on; migration byte-identity is explicitly labeled a SAFETY gate, and a pool with zero cross-edges is
  reported as MIGRATED, not INTEGRATED.
- CONFLICT SURFACES LOUDLY, not silently: the config-union `MergeConflict` + the name-disjointness assert +
  the "no edge with exactly one frozen endpoint" assert convert the three ways a merge silently corrupts a
  slice into registration-time errors. This is the framework's main safety improvement over bespoke (where a
  conflict was caught only if the author remembered to test it).
- GROUP-C name collisions and global OU/neuromod are REAL structural limits, not laziness — the multi-pool
  design is the honest surpass (mirrors how affect already merges onto its own pool), not a deferral.

## 7. Prototype skeleton

`research/runners/onebrain_merge_framework.py` — the `OrganDescriptor` dataclass, a `merge_organs` engine
(config-union + region-union + post-build hooks + gain-0 freeze), a `REGISTRY` with surprise + world-model
registered from the on-`main` de-risk builders, and a numpy smoke that builds the engine pool and the shipped
`MergedSubstrate` and asserts INIT-array byte-identity over the organ regions (the round-trip proof). See the
file's `_smoke()` and the committed run output. This proves the schema compiles and one organ round-trips
byte-identically; it does NOT migrate the remaining organs (that is the plan above, run batched on the pool).

Smoke result artifact (numpy, seed 42):
`research/findings/raw/_onebrain_merge_framework_smoke_s42.json` — engine_N == shipped_N == 1584,
`max_init_delta = 0.0`, BYTE-IDENTICAL PASS (the descriptor->engine pool reproduces the shipped
`MergedSubstrate` per-neuron init exactly over all 11 regions).
