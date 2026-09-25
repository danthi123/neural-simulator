---
type: finding
status: descriptive
date: 2026-09-24
lane: chat-time-plasticity-audit (coordinator follow-up to 2026-09-24-plastic-mask-instrument-and-production-
  reachability.md's open question: "the ORDER organs attach to the pool during a real chat process's warmup
  determines how much of a real conversation's turns run with Hebbian enabled at all -- a live question this
  document does not resolve")
mechanism: an empirical audit, through the REAL production entry point (webapp.server.brain_chat), of which
  synapses can change DURING a live conversation, plus a default-OFF fix (a NAMED per-pathway plasticity gate
  replacing a bridge-wide cfg.enable_hebbian_learning kill switch) for the one mechanism the audit confirmed is
  a latent hazard, verified at seed 42, numpy/CPU
seeds: [42]
seed-waiver: single-seed DESCRIPTIVE audit + mechanism verification (docs/TERMS.md: a magnitude claim needs
  6 seeds; the qualitative claims here -- which cfg object is shared, which pathway is gated, whether a drift
  reads exactly 0.0 or nonzero -- do not depend on the seed). The fix's own 6-seed capability check is
  PRE-REGISTERED below and NOT run.
verdict: DESCRIPTIVE. (1) `WorldModelProductionOrgan` and `SurpriseProductionOrgan` each train their own
  pathway then set `cfg.enable_hebbian_learning = False` on the ONE cfg object the default-ON 11-organ wave3
  pool shares with comprehension/metacog/pragmatic/self_schema/curiosity/causal_whatif -- confirmed live, in
  the real production warmup order, at seed 42; SURPRISE is the organ that actually flips the switch first
  (comprehension -> surprise -> metacog -> world-model -> pragmatic), correcting the trigger finding's framing,
  which named world-model. (2) Empirically, NO faculty riding that shared pool is
  SILENTLY FROZEN today: every one of the eight is itself a documented "train once, then read a frozen
  circuit" oracle -- the kill switch currently coincides with what each of them already wants for itself. It is
  a LATENT hazard (a mechanism, not a currently-lost capability): any future faculty wired onto this pool that
  wants ONGOING chat-time Hebbian learning would be silently defeated by it, with no error and no signal. (3) A
  default-OFF fix is built and unit-tested: `BRAIN_WORLDMODEL_LOCAL_FREEZE` / `BRAIN_SURPRISE_LOCAL_FREEZE`
  replace the global kill with a NAMED per-pathway `set_plasticity_gate(..., 0.0)`, verified (off:
  byte-identical to today; on: each organ's own pathway still reads frozen AND a synthetic UNGATED probe
  pathway on the same shared bridge genuinely learns AND both organs' own functional answers are unchanged).
runner: research/runners/chat_time_plasticity_audit.py,
  tests/_chat_time_plasticity_local_freeze_scenario.py
artifacts:
  - research/findings/raw/_chat_time_plasticity_audit/s42.json
  - research/findings/raw/_chat_time_plasticity_audit/real_pool_freeze_s42.json
  - research/findings/raw/_chat_time_plasticity_local_freeze_mechanism_smoke/off_s42.json
  - research/findings/raw/_chat_time_plasticity_local_freeze_mechanism_smoke/on_s42.json
builds_on:
  - research/findings/2026-09-24-plastic-mask-instrument-and-production-reachability.md
  - research/findings/2026-09-23-d6-learn-through-use-hebbian-fact-write-s42-smoke-C3-NOGO-host-kb-familiarity-leak.md
---

# Chat-time plasticity audit: the shared wave3-pool Hebbian kill is a latent hazard, not (yet) a lost capability — and a local-gate fix (2026-09-24)

## ROUND 2 AMENDMENT (coordinator review, 2026-09-24)

Round 1 (branch `research/chat-time-plasticity-audit`, merged to main as `55823d1bd`) was reviewed
SOUND-WITH-ISSUES. Three fixes landed here (branch `research/chat-time-plasticity-audit-r2`), amending this SAME
finding in place rather than superseding it (the underlying measurements and conclusions all still hold):

1. **The fix was verified only on a synthetic 3-organ pool, never the real wave3 production pool, and the
   finding omitted the REAL safety net for the other 6 co-resident organs** (`onebrain_merge_framework.py`'s
   `_apply_gain0_freeze`, a `freeze_regions`-driven direct gain-0 freeze independent of `cfg.enable_hebbian_
   learning`) — added as new §3a, the per-faculty table's substrate/evidence columns were corrected, and a new
   `--real-pool-freeze-check` mode was added to `research/runners/chat_time_plasticity_audit.py` (§6, K6).
2. **A crash on the fallback pool**: `BRAIN_ONEBRAIN_WAVE3_POOL=0` + `BRAIN_ONEBRAIN_SINGLE_POOL=0` (both
   default-ON) reaches the legacy `onebrain_merge_production.MergedSubstrate` pool, which never threads
   `local_freeze_gate` into its own build calls, so a flag-on run raised `KeyError` on `bridge.
   set_plasticity_gate`. Fixed: `_freeze_local_or_fallback` (added to both organ modules) checks gate presence
   first and falls back to the pre-fix global kill with a logged `RuntimeWarning` instead of crashing — verified
   with a new `--mode fallback-pool` scenario + `tests/test_chat_time_plasticity_local_freeze.py::
   test_fallback_pool_does_not_crash_and_warns` (§6, K7).
3. **Wording**: the byte-identical-OFF claim leaned on "verified via `git diff`" instead of the measured OFF-arm
   data (docs/TERMS.md: byte-identical must be asserted IN THE DATA). Reworded (§5) to rest on the three measured
   OFF-arm quantities (no gate declared, `enable_hebbian_learning` False, the probe pathway killed) instead.

---

Branch `research/chat-time-plasticity-audit`. Coordinator follow-up to the 2026-09-24 plastic-mask finding,
which found that `WorldModelProductionOrgan._build_one` trains its own state->valence transition then sets
`cfg.enable_hebbian_learning = False` on the SAME cfg object the default-ON 11-organ wave3 pool
(`onebrain_wave3_pool_production.get_merged_cortical_pool`) shares with seven other production organs, and
left open "how much of a real conversation's turns run with Hebbian enabled at all."

## 1. The mechanism, confirmed in code (not inferred from a docstring)

`sim/bridge.py`'s Hebbian update block is gated FIRST by `cfg.enable_hebbian_learning` (a single boolean on the
bridge's `CoreSimConfig`); only INSIDE that gate does a per-synapse named `plasticity_gate`
(`cp_plasticity_rate_gain`, `bridge.set_plasticity_gate(name, value)`) apply as a multiplier. So
`cfg.enable_hebbian_learning = False` skips the ENTIRE Hebbian loop for the bridge — every named gate, however
it is set, is moot. `WorldModelProductionOrgan._build_one` and `SurpriseProductionOrgan._build_one`
(`research/runners/worldmodel_production_organ.py`, `research/runners/surprise_production_organ.py`) each: (a)
call `self._shared.ensure_built()` — attaching to the ONE shared `MergedSubstrate` via
`onebrain_wave3_pool_production.get_merged_cortical_pool(seed, min_wave=1)`; (b) train their own pathway
(`train_transition`/`train_expectation`); (c) set `cfg.enable_hebbian_learning = False` on that SAME shared
`cfg` object.

## 2. Attach order and when Hebbian goes off (real production warmup, seed 42, numpy)

`webapp/server.py::_warm_chat_brain`'s inner `_warm()` is what a real webapp process runs at boot, BEFORE any
`/api/brain-chat` turn. Its faculty attach order is: affect (standalone) → comprehension → surprise → metacog
→ world-model → pragmatic → value-choice (this audit skips value-choice: its own docstring warns the first
build can cost ~4 minutes, and it attaches AFTER every organ this bug can touch). `research/runners/
chat_time_plasticity_audit.py` replicates this order exactly and reads the shared pool's cfg after each step
(`research/findings/raw/_chat_time_plasticity_audit/s42.json`, `warmup` array):

| step | `enable_hebbian_learning` | `enable_stdp` | who moved it |
|---|---|---|---|
| before any organ (pool built, nothing trained) | True | False | n/a — the pool's own construction leaves Hebbian on |
| after affect | True | False | unchanged — affect is standalone, never touches this cfg |
| after comprehension | True | False | comprehension does not touch `enable_hebbian_learning` (it DOES flip its own `cue_position`/`cue_animacy`/`cue_verbfit`/`cue_lexbias` named gates from 1.0 to 0.0 here — a correct, LOCAL, per-pathway freeze of its own trained cue pathways, unrelated to the global switch) |
| after surprise | **False** | False | `SurpriseProductionOrgan._build_one`: True (internal, for its own training) → **False** — **this is the FIRST organ to flip the shared switch**, not world-model |
| after metacog | False | False | unchanged — metacog never touches this flag (a fixed NMDA-balance readout, no Hebbian pathway of its own) |
| after world-model | False | False | `WorldModelProductionOrgan._build_one`: True (internal) → False — a no-op from the OUTSIDE (surprise already set it False); world-model's own line is real but not the FIRST cause |
| after pragmatic | False | False | unchanged — pragmatic is built plasticity-OFF by design and never touches this flag |

**Correction to the trigger finding's framing:** the referenced 2026-09-24 plastic-mask finding named world-model
as the organ whose `cfg.enable_hebbian_learning = False` line does the killing. Measured here: **surprise's
identical line runs FIRST in the real attach order** (comprehension → surprise → metacog → world-model →
pragmatic) and is what actually flips the switch; world-model's own line only re-confirms an already-False
value. Both lines are equally real and equally hazardous (§5 fixes both), but the FIRST cause, empirically, is
surprise. `enable_hebbian_learning` reads **False from the end of surprise's build to the end of the process**
— no code path re-enables it globally afterward. The only per-turn mechanism that DOES move weights on this
shared bridge (the wave3 xedge cross-edge credit, §4) is immune to this because it uses a NAMED gate + a
temporarily opened `enable_stdp` window of its OWN, never touching this switch.

## 3. Six real chat turns through `webapp.server.brain_chat` (seed 42, numpy, stub renderer)

`research/runners/chat_time_plasticity_audit.py` then drives 6 turns (teach/recall/prospective-memory-
formation/intervening/prospective-memory-cue) through the exact `/api/brain-chat` handler and re-reads, after
every turn, the shared wave3 pool's whole-bridge max|dw| (bucketed by NAMED gate vs UNGATED) plus every organ
confirmed to hold its OWN standalone bridge (affect, source_provenance, prospective memory once its formation
turn builds it):

| bridge | whole-bridge max\|dw\| over 6 turns | `enable_hebbian_learning` throughout | why |
|---|---|---|---|
| shared wave3 pool | **0.0** (93,856 gated + 437,948 ungated synapses, every turn) | False | the global kill (§1); confirmed 0.0 on every gated AND ungated synapse these 6 turns touch |
| affect (standalone) | 0.0 (48,520 gated + 42,813 ungated) | False, permanently, on its own cfg (affect runs Hebbian-off by design; its own named gates — `workspace_loop_fixed`, `meta_read_fixed`, `meta_to_self_confid_fixed`, `arb_wta_fixed` — are its OWN internal frozen-readout pathways, unrelated to wave3's identically-named `workspace_loop_fixed`) | own bridge, never shared |
| source_provenance (standalone) | 28.7 (2,225 UNGATED synapses only; its own `content_learn`/`prov_learn` gated pathways read exactly 0.0) | True, permanently, on its own cfg | own bridge, own cfg, never shared — see §5 and the caveat immediately below |
| prospective memory (standalone, per-session) | 0.0 (99,696 ungated synapses; no named gates on this circuit) | False (`enable_stdp: True`) | own bridge, own cfg, never shared — see §5 and the caveat immediately below |

**Two honest caveats on these two rows, not on the wave3-pool finding.** (1) Both `source_provenance` and
`prospective_memory` build LAZILY, inside a turn, the first time the conversation needs them (turn 1's teach
and turn 4's "remind me..." respectively) — this runner's `sp_w0`/`pm_w0` baseline is taken the first time
`_SESSION_PMEM`/`_get_source_provenance_organ` returns a built organ, which is **after** that same turn's own
write already ran, so this specific instrument cannot see either organ's OWN first write (it is folded into
the baseline). This is a measurement-timing limitation of this runner, not evidence either write did not
happen. (2) The BEHAVIOR confirms both mechanisms ran correctly in this exact conversation regardless: turn
1's response carries `"provenance": {"label": "perceived", "agrees_with_encoded": true, ...}` (the encode+judge
round-trip worked), and turn 6 ("the timer rings") correctly prepends `"(Reminder — you asked me to check the
oven when the timer rings came up, and the timer rings just came up.)"` — the held intention correctly fired on
its cue, four turns after formation. (3) The 28.7 max|dw| on source_provenance's own UNGATED synapses, present
from turn 3 onward, is NOT the encode/judge mechanism (those stay at exactly 0.0 the whole run) — it lands on
edges that are declared `plastic=False` in `ProvenanceBrain`'s own wiring (the ctx↔prov / prov↔inh opponent-
inhibition pathways) but carry no NAMED plasticity gate, so they are exactly the shape of the ALREADY-DOCUMENTED
2026-09-02 defect ("the runtime Hebbian path never consulted `cp_synapse_plastic_mask`", `research/FAILURE_LOG.md`,
gated by `BRAIN_ENFORCE_PLASTIC_MASK`, default OFF) — a pre-existing, differently-caused drift on a NEW organ
this audit happened to observe, not this finding's shared-cfg mechanism. Flagged, not chased further here.

## 3a. The REAL safety net for the other 6 organs, omitted by round 1: `_apply_gain0_freeze`

Coordinator review round 2 (2026-09-24) named the gap: round 1 characterized comprehension/metacog/pragmatic/
self_schema/curiosity/causal_whatif/source_provenance as "FROZEN-BY-DESIGN" without saying WHY their edges can
survive `cfg.enable_hebbian_learning` ever reading True (which this finding's own fix makes happen). The answer
is a SEPARATE, independent mechanism from world-model/surprise's cfg switch or named gate:
`onebrain_merge_framework.MergedPool.ensure_built`'s step 7 (`_apply_gain0_freeze`, lines ~398-406/454-472) unions
every registered `OrganDescriptor.freeze_regions` at POOL-BUILD time and sets `cp_plasticity_rate_gain = 0.0`
DIRECTLY on every edge with BOTH endpoints inside that union — a raw array write, not a NAMED gate (so it never
appears in `bridge.list_plasticity_gates()`, and nothing can `set_plasticity_gate` it back open by name). Because
this gain multiplies the Hebbian delta INSIDE the same `if cfg.enable_hebbian_learning:` block (§1), it holds
these edges at exactly 0 drift NO MATTER what `cfg.enable_hebbian_learning` reads afterward — permanently, for
the life of the pool.

Tracing which organs declare `freeze_regions` (`_onebrain_wave3_organread_verify._wave3_descriptors`, reuse-by-
import) confirms comprehension, metacog, pragmatic, self_schema, curiosity, causal_whatif and source_provenance
ALL do (`_onebrain_twopool_merge_organread_verify._recon_descriptors` is where metacog/pragmatic's are set, with
a comment that states the design intent outright: `mc["enable_hebbian_learning"] = True  # match pool-1 global;
pool-2 edges gain-0 frozen` — the pool's original designers deliberately made these two organs robust to
whatever pool-#1's (surprise/world-model's) global switch does). **World-model and surprise are the ONLY two
wave3-pool organs with NO `freeze_regions` declared** — they are the SOLE reason the fragile global switch
existed in the first place, and the sole organs this finding's fix actually needed to change.

**Measured on the REAL wave3 production pool** (`research/runners/chat_time_plasticity_audit.py
--real-pool-freeze-check`, both `BRAIN_WORLDMODEL_LOCAL_FREEZE=1` and `BRAIN_SURPRISE_LOCAL_FREEZE=1`, seed 42,
`research/findings/raw/_chat_time_plasticity_audit/real_pool_freeze_s42.json`):

| quantity | measured value |
|---|---|
| `enable_hebbian_learning_after_warmup` | **True** |
| `worldmodel_gate_value` | 0.0 |
| `surprise_gate_value` | 0.0 |
| `gain0_frozen_regions_ok` (comprehension+metacog+pragmatic+self_schema+curiosity+causal_whatif+source_provenance+prospective_memory+d6_multiref_wm's own regions, 278,364 synapses) | **True** (`max\|dw\| == 0.0`) |
| `whole_bridge_max_dw` (every synapse on the pool, no bucketing) | 0.0 |

`frozen_regions_by_organ` confirms the exact region sets `_apply_gain0_freeze` protects for each of the 9 other
organs (e.g. `metacog: [workspace, workspace_fs, meta_schema]`, `comprehension: [sel_agent, sel_FS_agent,
sel_patient, sel_FS_patient, cue_position_pos, cue_position_neg, cue_animacy_pos, cue_animacy_neg,
cue_verbfit_pos, cue_verbfit_neg, cue_lexbias_pos, cue_lexbias_neg]`) and `organs_with_no_freeze_regions:
[surprise, worldmodel]` confirms those two are the only pool members without one. `whole_bridge_max_dw == 0.0`
means nothing moved ANYWHERE on the real pool across these 4 turns even with `enable_hebbian_learning` now True
— consistent with §4's finding that no REAL production faculty currently has an ungated, chat-time-plastic
pathway on this pool (the synthetic probe pathway in §5 exists precisely because the real pool has none to
demonstrate the restored capability with).

This closes round 1's gap: the fix's safety for the other 6 organs was never "no faculty happens to want ongoing
learning" alone — it is this independent, pool-build-time, un-named freeze that makes their edges immovable
regardless of the global switch. World-model/surprise are the only load-bearing case for the local-gate fix.

## 4. The per-faculty table (LIVE / FROZEN-BY-DESIGN / SILENTLY FROZEN)

Every faculty whose design docstring or a findings doc claims it learns/adapts/writes DURING a live
conversation (not merely once at build time), classified against what §1–3 actually measured:

| faculty | mechanism | substrate | classification | evidence |
|---|---|---|---|---|
| world-model (state→pred) | bridge Hebbian, `train_transition` | shared wave3 pool; **NO `freeze_regions` declared** (§3a) | **FROZEN-BY-DESIGN** (for itself) — but via the hazardous global-switch MECHANISM (§1), the ONLY safety net it has | own docstring: "TRAINED (Hebbian state->valence) then FROZEN"; code confirmed |
| surprise (cue→patient_expected) | bridge Hebbian, `train_expectation` | shared wave3 pool; **NO `freeze_regions` declared** (§3a) | **FROZEN-BY-DESIGN** (same mechanism, same sole safety net) | own docstring: "LEARN ... then FREEZE (per-turn reads never learn)" |
| metacog | static NMDA-conductance-balance readout | shared wave3 pool; `freeze_regions` gain0-frozen (§3a) | **FROZEN-BY-DESIGN** — never claims chat-time Hebbian learning, AND independently hard-frozen regardless of `enable_hebbian_learning` | own docstring: "the confidence IS a synaptic-conductance balance"; `_recon_descriptors`: "pool-2 edges gain-0 frozen" |
| pragmatic | fixed RSA circuit | shared wave3 pool; `freeze_regions` gain0-frozen (§3a) | **FROZEN-BY-DESIGN**, independently hard-frozen | own docstring: "computed ONCE at organ-build and FROZEN ... plasticity OFF, a FIXED operating point" |
| comprehension | role-competition; specific pathways separately frozen by the PRE-EXISTING named gate `workspace_loop_fixed` | shared wave3 pool; `freeze_regions=tuple(comp.regions)` gain0-frozen (§3a), ON TOP of the named gate | **FROZEN-BY-DESIGN**, TWICE independent of this bug | 2026-09-24 plastic-mask finding, §2; §3a |
| self_schema | fixed authorship-readout circuit | shared wave3 pool; `freeze_regions` gain0-frozen (§3a) | **FROZEN-BY-DESIGN**, independently hard-frozen | own docstring: reuse-by-import of a static de-risked circuit |
| curiosity | novelty-driven ASK-pool read; a graded habituation-style novelty is a DECLARED, unbuilt next rung | shared wave3 pool; `freeze_regions` gain0-frozen (§3a) | **FROZEN-BY-DESIGN** today, independently hard-frozen — but the gain0 freeze would ALSO have to be removed (not just this fix's flags flipped) before the declared next rung (graded novelty) could actually learn | own docstring: "a graded familiarity-gate novelty (Bogacz-Brown) is the next rung" (not yet built, so nothing is silently lost) |
| causal_whatif | causal curriculum trained once, `ensure_built` guard, keyed per-brain-composer | shared wave3 pool; `freeze_regions=("evt",)` gain0-frozen (§3a) | **FROZEN-BY-DESIGN** (a separate, undeclared build-once-staleness residual — out of scope here), independently hard-frozen too | code: `if self._built: return` |
| source_provenance | per-call live Hebbian encode, `encode_fact` → `ProvenanceBrain.encode(pattern, provenance, learning=True)` | **own standalone bridge/cfg** — `get_organ()` has NO `shared=` parameter at all (the WAVE3-POOL COPY of source_provenance's regions IS gain0-frozen per §3a, but production never reads that copy) | **LIVE** | `ProvenanceBrain.__init__`: `cfg.enable_hebbian_learning = True` permanently, own cfg; `encode()` opens the NAMED gates `prov_learn`/`content_learn` only for the encode call, in a `try/finally`, then re-closes them — the SAME local-per-pathway pattern this finding's fix reuses |
| prospective_memory | one-shot Hebbian cue→action binding at intention-FORMATION | **own standalone bridge, per-session** — `ProspectiveMemoryOrgan.__init__` takes no `shared=` at all | **LIVE** | code: no shared-pool path exists for this organ |
| d6_multiref_wm Hebbian store | local phase-coupled rule, its own per-write `eta` (never reads `cfg.enable_hebbian_learning`) | own path, `BRAIN_D6_HEBBIAN_STORE` default **OFF** | **FROZEN-BY-DESIGN / not-yet-default-on** (an explicit, working knob — not silently broken) | research/findings/2026-09-23-d6-learn-through-use-v3-capability-gate-GO-6of6.md |
| xedge cross-edge credit (`credit_live_turn`) | a temporarily-opened `enable_stdp` window + the named gate `wm_to_sel_r2`, re-frozen after each credited turn | **its own SEPARATE pool** (`onebrain_xedge_production.get_xedge_pool`, comprehension+d6+da_credit), `BRAIN_ONEBRAIN_XEDGE`/`_LEARN` default **ON** since 2026-08-28 | **LIVE (weights genuinely move)**, but presently a SEPARATE, already-documented architecture gap: comprehension's live read resolves the WAVE3 pool first, so this pool's cross-edge has "no path to a reply" — see that module's own docstring ("live cross-organ synapses in production = 0"). NOT this bug (different pool, different cfg); flagged here only so it is not mistaken for the same defect | `research/runners/onebrain_xedge_production.py` module docstring, "THE SEVERANCE THIS CLOSES" |
| DA-encoding / tag-capture / composer store / reconsolidation | direct host-level rewrites of `store_conns` | N/A — never routed through the bridge's central Hebbian loop | **LIVE**, unaffected by `cfg.enable_hebbian_learning` by construction (a separate "is a direct weight write itself a brain-based-only shortcut" question, out of scope here) | code read (each writes weights directly, not via `_run_one_simulation_step`'s Hebbian block) |
| D5 episodic (learn-through-use) | BTSP one-shot write + Turrigiano scaling | own standalone bridge (`EpisodicDapMemory`) — NOT one of the wave3 pool's 11 organs | **LIVE by design**, but on numpy the WRITE is latency-DEFERRED (`webapp/server.py::_episodic_store_ok`: "~510s/topic on numpy@2000" vs "~seconds on cupy") — unrelated to this bug; NOT exercised in this numpy measurement (forcing `BRAIN_EPISODIC_STORE=1` made a 6-turn run take >10 min/turn on this shared box; not worth the cost to re-confirm an already-declared, orthogonal gate) | code + docstring |

**No faculty is SILENTLY FROZEN today.** Every wave3-pool rider is honestly documented as a build-once oracle
and the audit's own measurement (§3) confirms zero chat-time drift on all of them — exactly what each one's own
design says should happen. **Correction from round 1 (coordinator review, 2026-09-24): this is NOT a coincidence
for 6 of the 8 riders** — comprehension/metacog/pragmatic/self_schema/curiosity/causal_whatif are independently,
permanently hard-frozen by `_apply_gain0_freeze` (§3a) regardless of what `cfg.enable_hebbian_learning` reads,
by original design ("match pool-1 global; pool-2 edges gain-0 frozen"). **It IS a coincidence, and the real
hazard, for the remaining 2:** world-model and surprise declare NO `freeze_regions` at all, so the fragile global
switch was their ONLY protection pre-fix — exactly the two organs this finding's local-gate fix targets. Two of
this pool's own declared "next rungs" that live on ALREADY-gain0-frozen organs — curiosity's graded habituation-
style novelty, causal_whatif's staleness fix — would need MORE than this fix's flags to ever learn chat-time
(the gain0 freeze itself would have to be relaxed too, a separate, larger change); they are not at risk from the
mechanism this finding fixes, only from the (separate, independent) gain0 freeze, which this fix does not touch.

## 5. The fix: two known-good LOCAL patterns already in this codebase, applied to world-model and surprise

The codebase already has TWO correct exemplars of "freeze one organ's own pathway without touching the shared
switch": `source_provenance_honesty.ProvenanceBrain.encode()` (opens/closes its own named gates
`prov_learn`/`content_learn` per call, `cfg.enable_hebbian_learning` stays permanently True on its own bridge)
and `onebrain_xedge_production.Wave3XedgeView.credit_live_turn` (opens a local `enable_stdp` window + the named
gate `wm_to_sel_r2` for exactly one credited step, then re-freezes). World-model and surprise had neither —
they reached for the coarsest available tool because their SHARED cfg object gave them nothing else.

**The fix** (default-OFF, additive, `NO sim/ edit`):
- `research/runners/_affective_world_model_derisk.py::build_world_model_circuit` gained
  `local_freeze_gate: bool = False`; when True, the `state→pred_pos`/`state→pred_neg` pathways are tagged with
  the new named gate `WORLDMODEL_FREEZE_GATE = "worldmodel_frozen"` (else ungated, exactly as before).
- `research/runners/_spiking_expectation_rpe_derisk.py::build_expectation_circuit` gained the same
  `local_freeze_gate` param, tagging `cue→patient_expected` with `SURPRISE_FREEZE_GATE = "surprise_frozen"`.
- `worldmodel_production_organ.worldmodel_local_freeze_enabled()` / `surprise_production_organ.
  surprise_local_freeze_enabled()` (env `BRAIN_WORLDMODEL_LOCAL_FREEZE` / `BRAIN_SURPRISE_LOCAL_FREEZE`,
  default OFF). Off: `_build_one` is UNCHANGED (`cfg.enable_hebbian_learning = False`, byte-identical). On:
  `_build_one` instead calls `bridge.set_plasticity_gate(<GATE>, 0.0)` after training and never touches
  `cfg.enable_hebbian_learning` — so it stays whatever `train_transition`/`train_expectation` left it (True).
- `onebrain_merge_framework.py`'s `_surprise_spec`/`_worldmodel_spec` (the functions the production wave3 pool
  actually calls to build these two organs' regions) thread the SAME env-gated `local_freeze_gate` into the
  wiring, so the gate reaches the real SHARED bridge, not just a standalone one.

### Test results (seed 42, numpy; `tests/test_chat_time_plasticity_local_freeze.py`, 3/3 PASS)

Scenario: `tests/_chat_time_plasticity_local_freeze_scenario.py` builds a 3-organ pool (surprise + world-model
+ a synthetic 2-region UNGATED "probe" pathway standing in for "whatever faculty wires an ongoing chat-time
Hebbian pathway onto this pool next"), trains surprise+world-model exactly as production warmup does, then
drives 300 steps of real co-activity on the probe pathway. Saved runs:
`research/findings/raw/_chat_time_plasticity_local_freeze_mechanism_smoke/{off,on}_s42.json`.

| check | OFF (today's shipped mechanism) | ON (`BRAIN_*_LOCAL_FREEZE=1`) |
|---|---|---|
| named gates declared | none (`[]`) | `worldmodel_frozen`, `surprise_frozen` |
| `enable_hebbian_learning` after both organs build | **False** | **True** |
| world-model's own state→pred max\|dw\| (11,520 synapses) | 0.0 | 0.0 (via the named gate, not the global switch) |
| surprise's own cue→patient_expected max\|dw\| (55,296 synapses) | 0.0 | 0.0 |
| the UNGATED probe pathway's max\|dw\| after 300 co-active steps (144 synapses) | **0.0** (killed by the global switch, same blast radius the audit measured) | **0.0339** (genuinely learns) |
| both organs' own functional answers (`judge()`/`expectation()`/`read_surprise()`) | — | **identical** to OFF (dict-equal) |

The OFF arm's own measured data is what supports "byte-identical to the pre-fix mechanism", not an inspection of
the diff: no named gate is declared at all (`gates_after_organ_build: []`), `enable_hebbian_learning` reads
False after both organs build (the ONE externally-visible state the pre-fix code ever produced), and the
UNGATED probe pathway is killed (`max|dw| == 0.0`) -- together these three measured quantities fully
characterize "the old bridge-wide kill and nothing else ran", which is what byte-identical-to-today means here
(docs/TERMS.md: asserted in the data, not inferred from reading the code). The ON arm proves the causal claim:
the SAME probe pathway that the global switch silently kills is free to learn once the switch is no longer
touched (`max|dw| == 0.0339 > 0`), while both organs' own reads are measured identical to the OFF arm's.

**Regression checks** (existing suites that exercise `onebrain_merge_framework.py`, unaffected since the new
code paths are default-OFF): `python -m research.runners.onebrain_merge_framework --smoke` — PASS
(`max_init_delta=0.0`, byte-identical substrate-init); `tests/test_onebrain_affect_pool.py` +
`tests/test_reward_value_afferent.py` — 73/73 PASS.

## 6. Pre-registered 6-seed capability check (NOT run, NOT queued — commands only)

The check that would decide whether to default-flip either `BRAIN_*_LOCAL_FREEZE` flag on. Criteria, all must
hold on all 6 seeds (42, 43, 44, 100, 101, 102):
- **K1 mechanism** — ON: both named gates exist and read 0.0 after both organs build; `enable_hebbian_learning`
  reads True.
- **K2 own-pathway frozen** — ON: world-model's `state→pred_{pos,neg}` and surprise's `cue→patient_expected`
  max\|dw\| == 0.0 after the probe-drive.
- **K3 capability restored** — ON: the synthetic probe pathway's max\|dw\| > 0.0 after the probe-drive; OFF:
  == 0.0 (the same instrument must show the contrast on every seed, not just 42).
- **K4 no-regression** — `surprise_answer` and `worldmodel_answer` are IDENTICAL between OFF and ON on every
  seed (both organs' functional output to their callers never changes).
- **K5 off byte-identical** — OFF: no gate named `worldmodel_frozen`/`surprise_frozen` exists at all, and
  `enable_hebbian_learning` reads False (the pre-existing mechanism, unmodified).
- **K6 real-pool safety net** (coordinator round 2, 2026-09-24 — extends the check to the REAL wave3 production
  pool, not just the synthetic 3-organ one) — ON, via `--real-pool-freeze-check`: `gain0_frozen_regions_ok` is
  True (the union of comprehension/metacog/pragmatic/self_schema/curiosity/causal_whatif/source_provenance's
  `freeze_regions` edges reads max\|dw\|==0.0 over the real chat turns) AND `worldmodel_gate_value`/
  `surprise_gate_value` both read 0.0 AND `enable_hebbian_learning_after_warmup` reads True — on EVERY seed.
- **K7 fallback pool** (coordinator round 2) — with `BRAIN_ONEBRAIN_WAVE3_POOL=0` + `BRAIN_ONEBRAIN_SINGLE_
  POOL=0` + either local-freeze flag ON, via `--mode fallback-pool`: `crashed` is False, exactly 2 `RuntimeWarning`s
  fire (naming `worldmodel_frozen`/`surprise_frozen`), `enable_hebbian_learning_after_build` reads False (the
  documented, no-worse-than-before fallback), and both organs' own functional answers are unchanged from a
  flags-off run on the SAME fallback pool — on every seed.

Commands (numpy/CPU; run only when the owner authorizes the flip decision):
```bash
bash tools/mem_ok.sh 8
mkdir -p research/findings/raw/_chat_time_plasticity_local_freeze_6seed
for s in 42 43 44 100 101 102; do
  for mode in off on; do
    SIM_BACKEND=numpy bash tools/memcap.sh 8 -- .venv/bin/python -m \
      tests._chat_time_plasticity_local_freeze_scenario --mode "$mode" --seed "$s" \
      > "research/findings/raw/_chat_time_plasticity_local_freeze_6seed/${mode}_s${s}.json"
  done
  # K7: the fallback-pool combination (BRAIN_ONEBRAIN_WAVE3_POOL=0 + BRAIN_ONEBRAIN_SINGLE_POOL=0), same seed
  SIM_BACKEND=numpy bash tools/memcap.sh 8 -- .venv/bin/python -m \
    tests._chat_time_plasticity_local_freeze_scenario --mode fallback-pool --seed "$s" \
    > "research/findings/raw/_chat_time_plasticity_local_freeze_6seed/fallback_s${s}.json"
done
# K6: the REAL wave3 production pool is much heavier (a real webapp.server.brain_chat build + turns) -- run each
# seed as its own memcapped job, not fanned in a loop, so a single seed's OOM/crash does not lose the others.
for s in 42 43 44 100 101 102; do
  bash tools/mem_ok.sh 12
  SIM_BACKEND=numpy bash tools/memcap.sh 12 -- .venv/bin/python -m research.runners.chat_time_plasticity_audit \
    --real-pool-freeze-check --seed "$s" \
    --out "research/findings/raw/_chat_time_plasticity_audit/real_pool_freeze_s${s}.json"
done
# Score: for each seed, diff off_s<seed>.json vs on_s<seed>.json against K1-K5, fallback_s<seed>.json against K7,
# and real_pool_freeze_s<seed>.json against K6 (no scorer script exists yet -- write one, or diff by hand; do
# not report a verdict without it).
```

## What this document does NOT claim

- No claim that any CURRENT production faculty behaves differently with either new flag on — none does (K4).
  The fix closes a latent hazard, not a bug affecting today's replies.
- No 6-seed evidence for the fix (its capability check is pre-registered above, not run).
- Not a claim about the xedge cross-edge's own "does it reach the reply" gap (§4's xedge row) — that is a
  pre-existing, separately-documented architecture question this audit did not re-investigate.
- Not a claim that `sc_orienting_production_organ.py`'s `cfg.enable_hebbian_learning = False` (also named in
  the task that produced this audit) is the same bug: its bridge is confirmed standalone (never shared), so it
  has no blast radius and needs no fix.
- Not a claim about `onebrain_affect_pool.py`'s `cfg.enable_hebbian_learning = False` (a DIFFERENT,
  save/restore-in-a-`finally` pattern, on a default-OFF path) or `onebrain_xedge_wave3.py`'s occurrence (also
  default-OFF) — both are candidates for the SAME local-gate fix if either flag ever defaults ON, not
  investigated further here.
- Single-seed. The qualitative claims (which cfg is shared, which pathway is gated, exact-zero vs nonzero
  drift) do not depend on the seed; a magnitude claim would.
- Not a claim that source_provenance's or prospective memory's OWN first write produced zero drift — this
  runner's baseline for both is taken after each organ's lazy first build inside the SAME turn that also writes
  it (§3's caveat), so their first write is folded into the baseline and invisible to this specific instrument.
  The behavioral evidence (§3) — a correct provenance judgment, and a correctly-fired reminder four turns after
  formation — is what supports classifying both LIVE, not a weight-drift number.
- Not a claim about the 28.7 max|dw| observed on source_provenance's own ungated synapses (§3) beyond noting it
  matches the shape of the already-gated 2026-09-02 plastic-mask defect; not independently re-verified as the
  SAME cause here.
