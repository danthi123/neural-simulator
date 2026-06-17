# Reconsolidation for the conversational agent — deep-research + reuse scoping

**Date:** 2026-06-17
**Type:** Scope-only (deep-research-first standing practice). NO build in this doc — diagnosis → ranked options → reuse map → ONE cheap-first de-risk with anti-cheats → GO/NEGATIVE meaning.
**Author:** read-only research subagent
**Roadmap slot:** the next *medium* conversational item (CYCLE 146 EXACT-NEXT, foreground track): scale-WM (done) → narrate (done) → **reconsolidation (THIS)** → productive syntax (the hard dendritic half deferred) → dendritic credit-assignment (deferred).

---

## 0. One-paragraph executive answer

The production conversational memory is **append-only**: `RFPhasorComposer.store()` pushes each fact onto a Python list (`self.kb`), and every `query_*` returns the **first** matching fact. Tell the agent "the dog went north", then correct it ("actually the dog went south"), and today you get **two contradictory facts coexisting**, with the stale one answered first. Reconsolidation is the mechanism that fixes exactly this: a stored fact, when **reactivated by a partial cue** and met with **new/mismatching information** (a prediction error at retrieval), becomes **labile** and is **updated in place** rather than duplicated. The recommended build is a **mismatch-gated in-place fact update on the existing composer KB** — i.e. realize the Osan-Tort-Amaral (2011) attractor-network reconsolidation model's two-independent-mechanisms idea (Hebbian re-store vs mismatch-driven labilization) at the composer/VSA layer, with the **prediction-error gate** sourced from the composer's own retrieval signals (the familiarity match + the unbind mismatch the moat already computes). The cheap-first de-risk (`_phaseB_reconsolidation_update_derisk.py`, numpy/CPU, 6 seeds) tests the load-bearing claim — a reactivation-gated update *replaces* the reactivated fact and is *distinguishable* from naive append, overwrite-always, and do-nothing — with the **no-prediction-error control** (re-stating the SAME fact must NOT change the memory) as the decisive anti-cheat that proves the boundary condition is real rather than last-write-wins.

---

## 1. Diagnosis

### 1.1 What is reconsolidation (biology), in one screen

A *consolidated* long-term memory is not write-once. When it is **reactivated** by a retrieval cue, it transiently returns to a **labile** (protein-synthesis-dependent) state and must **re-stabilize**; during that window it can be **updated, strengthened, or weakened** (Nader, Schafe & LeDoux 2000 — Kandel 6e Ch 53 catalog entry **J.27**; Lee 2009). Two facts make it a *controlled* mechanism rather than "overwrite on every recall":

- **It is gated by prediction error / novelty at retrieval.** A memory becomes labile **only** if reactivation includes something **new or mismatching**. A perfectly-predicted reactivation just re-stabilizes unchanged (Sevenster, Beckers & Kindt 2013, *Science* — PE is *necessary* for destabilization; Exton-McGuinness, Lee & Reichelt 2015; the "PE demarcates retrieval → reconsolidation → new learning" transition, Sinclair & Barense / Pedreira-style work). **Trust-but-verify confirmed** (see §6).
- **Update vs strengthen vs new-trace is itself content-dependent.** If reactivation repeats the prior experience → the trace **strengthens**; if it carries **novel** information → that information is **integrated into / linked to** the reactivated trace (Lee 2009; the hippocampal-content-updating demonstration, Rodriguez-Ortiz / Hupbach-style "Memory Reconsolidation Mediates the Updating of Hippocampal Memory Content"). If the mismatch is large enough, you cross into **extinction / a new parallel trace** instead of updating the old one (Osan-Tort-Amaral 2011; Sevenster 2014).

### 1.2 What "a memory updates on recall" means on THIS spiking-VSA substrate

The conversational memory has two interacting layers; reconsolidation can be realized at either, and they are not equivalent:

- **The composer KB layer (the fact store).** A fact is a role-filler **bound composite** (FHRR phasor, `_encode` → `_bundle` of `bind(role, filler)`), held in `self.kb` as `(fact_dict, composite_phases)`. "Updating on recall" here = on a cued retrieval whose new content **mismatches** the recovered filler, **rewrite that fact's composite in place** (replace its `kb` entry / re-bind the slot to the corrected filler) instead of appending a contradictory duplicate. This is the **direct, load-bearing** locus for the conversational capability — "the agent can be corrected and updates the fact" — and it is where the Osan-Amaral attractor-update analogy lands most cleanly (an FHRR composite *is* an attractor pattern; re-binding one slot is the discrete analogue of a mismatch-gated attractor update).

- **The engram / weight layer (Hebbian/STDP traces on the bridge).** A fact can also live as a Tonegawa engram tag (catalog **D.14**) or as STDP weights along `lang_input → pool` pathways. "Updating on recall" here = stimulate the tag (reactivate), then under a gated plasticity window let STDP/decay **re-store with the corrected co-firing** (strengthen the corrected association, decay the stale one). This is the more biologically literal reconsolidation (labile-then-restabilize at the synapse), but it is **slower, noisier, and harder to make multi-seed-clean** at vocab scale (the project's entire 2026-05 history shows weight-level binding is seed-fragile). It is the right *eventual* target but the **wrong first de-risk**.

**Recommendation framing:** de-risk reconsolidation at the **composer KB layer first** (cheap, deterministic, decisive), with the engram/weight-layer version named as the follow-on "biologically-literal" tier. This mirrors the project's standing pattern (host/composer scaffold first, then the spiking/synaptic conversion as a separately-validated step).

### 1.3 Conversational behaviors this unlocks

1. **In-place correction (the headline).** "The dog went north." → later → "Actually, the dog went south." The stored SVO fact's patient is **updated**; a subsequent "where did the dog go?" answers **south**, with **no** stale duplicate retained. A stateless basic LLM (Phi-3 / Llama-3.2 class) has *no* persistent fact store to correct — this is a memory behavior that distinguishes the artificial-life agent.
2. **Gist-vs-detail drift over repeated retrievals (catalog J.34, "memory imperfections as features").** Repeated reactivation+re-store can let verbatim detail decay while the role-structured gist persists — reconsolidation "makes retrieval inherently editable." This is *desirable* lossy/reconstructive memory under the owner's relaxed-moat directive, not a bug.
3. **Interference-driven, content-aware forgetting.** Updating-rather-than-appending means a corrected fact does not accumulate as clutter; the old value is overwritten (or weakened), which is the adaptive "maintaining memory relevance" function (Lee 2009).

### 1.4 The boundary condition that keeps it from "overwrite-always"

The single most important design constraint: **lability is prediction-error-gated.** Concretely, on this substrate, a reactivation should trigger an update **only when** the new content **mismatches** the stored content beyond a threshold; a reactivation that is **fully predicted** (re-stating the same fact, or asking about it without contradicting it) must **re-stabilize the memory unchanged**. Without this gate, "reconsolidation" degenerates into trivial **last-write-wins**, which is neither biological nor a capability (it is just `dict[key]=value`). The PE gate is what makes this scientifically and behaviorally meaningful — and it is the thing the de-risk's decisive anti-cheat tests (§4).

The complementary boundary: **abstention is still respected.** Updating a **previously-stored, reactivated** memory is the feature; **fabricating a never-stored fact from nothing** is still not allowed (the no-confab moat). Reconsolidation operates on the **retrieved** trace — there must *be* a matching trace to reactivate. A "correction" of a subject the agent never heard about should **abstain** (no trace to make labile → no update, and certainly no invention). The owner's 2026-06-17 reframe (`feedback_moat_not_hard_lossy_memory_ok`) explicitly *enables* this arc: lossy/reconstructive/updatable memory is acceptable; the moat is a plus to keep where free, traded for the learned-lossy path where it buys capability. Reconsolidation is squarely on the "trade for capability" side — but the update-an-existing-trace vs invent-from-nothing distinction is exactly what the de-risk characterizes.

---

## 2. Ranked biologically-grounded options

### ▶ Option A (RECOMMENDED) — Mismatch-gated in-place fact update at the composer KB (Osan-Amaral attractor-update, realized on FHRR)

- **Mechanism.** On a corrective utterance that **cues an existing fact** (matches its cue roles — e.g. same agent+action) but **supplies a mismatching filler** (different patient), measure a **prediction error** = the mismatch between the recovered filler (composer `unbind` + cleanup of the cued role) and the new asserted filler. If PE **exceeds a labilization threshold** (and is below the extinction threshold), **rewrite that fact's composite in place**: re-`bind` the corrected filler into the slot and replace the `kb` entry (optionally with a graded blend of old/new composite at moderate PE = partial update, full replace at high PE). If PE is **below threshold** (the new filler ≈ the stored one, i.e. a re-statement), **re-stabilize unchanged** (no write; optionally a tiny strengthening). The familiarity gate decides *whether a trace exists to reactivate at all* (abstain → no update, no invention).
- **Citation.** Osan, Tort & Amaral 2011, *PLoS ONE* "A Mismatch-Based Model for Memory Reconsolidation and Extinction in Attractor Networks" — **independent mechanisms** mediate Hebbian strengthening vs **mismatch-driven labilization**; reconsolidation = *updating a trace*, extinction = *a new trace*, selected by the similarity between original and reexposure. Sevenster 2013 (*Science*) for PE-necessity. Catalog **J.27** (reconsolidation), **J.34** (editable retrieval / gist). Lee 2009 (updating-vs-strengthening). Plate FHRR (the composite-as-attractor representation).
- **On THIS substrate.** Pure composer-layer logic over `RFPhasorComposer`: reuse `unbind`/`_cleanup` (PE measurement + which-fact-matches), `_bind`/`_encode` (the rewrite), `self.kb` (the store). The "attractor" is the FHRR composite; "mismatch-driven labilization" is the PE-gated decision to overwrite that composite; "Hebbian re-store" is the re-bind. **No `sim/` edit, reuse-by-import** — exactly the project's house pattern for a first de-risk.
- **Trade-offs.** (+) Cheap, deterministic, decisive, multi-seed-clean; directly delivers the headline conversational capability; the PE gate is naturally available (the moat already computes match strengths). (+) Bit-compatible default-off (append-only path preserved). (−) The update lives in the composer's `kb` list, which is itself the project's documented "composer-as-idealization" host-held store — so this reconsolidation is realized at the *idealized* layer, not yet the synapse. (That is acceptable and standard here, but must be stated honestly; the engram/weight version is Option C.)

### Option B — Engram-tag reactivation + plasticity-window re-store (the synaptic-literal tier)

- **Mechanism.** Represent a fact as a Tonegawa engram tag (D.14). To correct it: `stimulate_tag` to **reactivate** (the labile window), then under a **gated plasticity window** drive the corrected co-firing so STDP **re-stores** the corrected association and the stale one decays. The plasticity window is opened by a prediction-error neuromodulator signal (below).
- **Citation.** Catalog **D.14** (engram cells), **J.27/J.18** (reconsolidation is gene-expression/late-LTP-dependent — the *biological* labile-then-restabilize); Nader 2000. The PE-gated plasticity window maps to the existing `plasticity_window_gate` neuromodulator target (Cluster B.3 cholinergic gating) and the `from_rpe` production rule.
- **On THIS substrate.** Reuses `start_engram_recording`/`commit_engram_tag`/`stimulate_tag`/`clear_tag_drive` (already shipped, `sim/bridge.py`) + the neuromodulator subsystem's `plasticity_window_gate` + `cp_plasticity_rate_gain`. This is the **biologically literal** reconsolidation (labile-at-the-synapse).
- **Trade-offs.** (+) Genuinely synaptic; the "real" reconsolidation. (−) Weight-level binding at vocab scale is the project's most seed-fragile regime (the entire 2026-05 record); making "update-not-duplicate" multi-seed-clean at the synapse is a *much* larger, higher-variance build. **Wrong first de-risk; right eventual target.** Best pursued *after* Option A proves the capability and the PE-gate logic at the composer layer.

### Option C — Schema-gated assimilation (V_SCHEMA / Tse-2007) as the *integration* sibling of reconsolidation

- **Mechanism.** When the corrected content is **schema-consistent** (a known role-filler shape, an anchor pool that already exists), integrate it fast into the existing schema rather than forming a new trace — the same machinery the project already validated for fast novel-key binding.
- **Citation.** Tse et al. 2007 (schema-supported integration); the project's V_SCHEMA findings (`2026-05-12-V_SCHEMA-2of4-strong-hippo-BREAKTHROUGH.md` and siblings). Mechanistically adjacent to PE-gated reconsolidation: **schema-consistency is the low-PE / assimilation regime; reconsolidation-update is the moderate-PE regime; new-trace/extinction is the high-PE regime** — the same prediction-error axis Osan-Amaral formalize.
- **On THIS substrate.** V_SCHEMA is a weight-level mechanism on a hippocampus-enabled lineage (slow, 53-min bootstraps). **Not** a first de-risk; cited here because it (a) shows the project already has the *assimilation* end of the PE axis, and (b) is the natural partner for the engram-tier (Option B) build — schema-consistent corrections assimilate, schema-violating corrections reconsolidate.
- **Trade-offs.** (+) Reuses validated machinery; completes the PE-axis story. (−) Expensive, seed-fragile, weight-level; an integration *sibling*, not the reconsolidation mechanism itself.

**Why A leads:** it isolates and proves the **single load-bearing claim** (PE-gated in-place update, distinguishable from append/overwrite/nothing) at the cheapest, most decisive layer, reuses the deployed composer verbatim, and directly yields the headline conversational behavior — while B and C (the synaptic-literal and schema-assimilation tiers) are named as the honest follow-on build that A de-risks.

---

## 3. Reusable project machinery (build minimally)

| Need | Existing code | Reuse for |
|---|---|---|
| The fact store + bind/unbind/bundle/cleanup | `research/runners/rf_phasor_composer.py` — `RFPhasorComposer` (`store`, `query_agent`, `query_patient`, `_encode`, `_bind`, `_unbind_phases`, `_cleanup`, `self.kb`) | **Option A core.** `kb` is the append-only list to make updatable; `unbind`+`_cleanup` measure the prediction error (recovered filler vs asserted filler); `_bind`/`_encode` perform the in-place rewrite. |
| The no-confab abstention / familiarity signal | `RFPhasorComposer` returns `None` on no-match; `OrderedPositionWM._match_strength` / `calibrate_threshold` (`research/runners/ordered_position_wm.py`); `resonate_fire_fhrr.py` `cleanup_separated` (Bogacz-Brown familiarity gate) | **The PE gate + the "is there a trace to reactivate?" gate.** The same match-strength machinery that gates abstention gives the mismatch magnitude that gates labilization. "No trace → abstain → no invention" is reused verbatim. |
| Multi-turn / corrective dialogue surface | `research/runners/multi_turn_agent_v2.py` (`MultiTurnAgentV2`, ordered discourse buffer, `hear`/`what_does`/`reason_chain`/`narrate`) + `research/runners/multi_turn_agent.py` (`SpikingLoopContextBuffer` reactivation buffer in `content_selection_spiking.py`) | The conversational wrapper where a "correction" turn is detected and routed to the composer's reconsolidation update. `hear()` is the natural hook (it already comprehends + stores). |
| Engram reactivation (Option B) | `sim/bridge.py` — `start_engram_recording`, `commit_engram_tag`, `stimulate_tag`, `clear_tag_drive`, `get_engram_tag_indices` (catalog D.14, shipped) | The synaptic-literal reactivation/labile-window for the Option-B follow-on. `stimulate_tag` = reactivate; the gated step that follows = re-store. |
| Extinction-style weakening + consolidation (Option B) | `sim/bridge_memory.py` — `BridgeMemory.forget` (multiplicative decay along a key's outgoing synapses), `.consolidate` (SWR replay) | The "weaken the stale trace" and "re-stabilize" halves of the synaptic reconsolidation cycle. `forget(decay_rate<1)` ≈ partial labilization/weakening; `consolidate` ≈ re-stabilization. |
| The lability/PE neuromodulator gate (Option B) | `sim/neuromodulators.py` — `plasticity_window_gate` target (HIGH conc BLOCKS, LOW PERMITS plasticity), `from_rpe` and `from_error_persistence` production rules, `pause_on_reward`; runtime `bridge.set_plasticity_gate` / `cp_plasticity_rate_gain` | A **prediction-error-driven plasticity window**: `from_rpe` produces concentration on mismatch; wired to `plasticity_window_gate` it opens the labile window only when retrieval carries PE — the exact biological gating, already a built target type. |
| Schema-assimilation sibling (Option C) | V_SCHEMA runners + findings (`2026-05-12-V_SCHEMA-*`); Tse-2007 catalog | The low-PE/assimilation end of the PE axis; partner for the engram tier. |
| Catalog grounding | `sim-catalog/.../feature-catalog.md` entries **J.27** (reconsolidation), **J.34** (editable retrieval / gist / false memory), **D.14** (engram cells), **D.19** (SWR replay), **J.18** (late-LTP/CREB — the biological labile-restabilize dependency) | Citations + the explicit "Sim status: missing — could be modeled with structural plasticity + gated reconsolidation but no project goal yet" (J.34) that this arc now makes a goal. |

**Net new code for the de-risk:** one runner (`_phaseB_reconsolidation_update_derisk.py`) that subclasses/wraps `RFPhasorComposer` with an `update_on_mismatch(...)` method + the four-arm comparison harness. **Zero `sim/` edits.** If the de-risk goes GO, the production change is a small additive method on `RFPhasorComposer` (default-off → append-only path bit-preserved) plus a correction-turn hook in `MultiTurnAgentV2` — again no `sim/` edit.

---

## 4. The recommended cheap-first de-risk

**Runner:** `research/runners/_phaseB_reconsolidation_update_derisk.py` (numpy/CPU; reuse-by-import only; NO `sim/` edit; no autodiff; no protected module touched). The composer's RF ops run on CPU per-op bridges, so this is a small-wall-clock probe.

**Pre-registered, frozen design (set BEFORE any multi-seed run; never tuned to a result):**

- **Setup.** Build an `RFPhasorComposer` (seed, D=128 to match the production agent, the standard SVO vocab). Store a baseline fact set, e.g. `dog go north`, `cat run south`, plus distractors. Then deliver a **corrective utterance** that cues an existing fact and supplies a **mismatching filler**: `dog go south` (same agent+action `dog`/`go`, patient changed north→south).
- **The mechanism under test (Option A).** `update_on_mismatch(agent, action, new_patient)`: find the fact whose cue roles match (`unbind(agent)`/`unbind(action)`); measure **PE** = `1 − phase_cos(recovered_patient_phasor, new_patient_phasor)` (reuse `_unbind_phases` + the match-strength function). If **no fact matches the cue** → **abstain** (return a no-op flag; do NOT create a fact — the moat). If matches **and** `PE ≥ PE_LABILE` → **rewrite in place** (re-`bind` `new_patient` into the patient slot, replace the `kb` entry). If matches **and** `PE < PE_LABILE` (a re-statement) → **re-stabilize unchanged** (no write).
- **Four arms compared on the SAME corrective utterance** (the load-bearing distinguishability test):
  1. **RECONSOLIDATE** (the mechanism): PE-gated in-place update.
  2. **NAIVE-APPEND** (baseline = current production): `store()` the corrected fact → two `dog go {north,south}` facts coexist.
  3. **OVERWRITE-ALWAYS** (ablation of the boundary condition): rewrite on *any* cue match regardless of PE.
  4. **DO-NOTHING** (ablation of the update): ignore the correction; stale fact persists.
- **Probes after the correction.**
  - **Post-correction query** `query_patient(dog, go)` → must be **south** (the corrected value), and the KB must contain **exactly one** `dog go *` fact (no duplicate). [RECONSOLIDATE passes; NAIVE-APPEND fails the single-fact check and/or answers the stale first match; DO-NOTHING answers north.]
  - **Untouched fact** `query_patient(cat, run)` → must still be **south** (correcting one fact must not corrupt others). [All but a broken update pass; included to catch collateral damage.]
- **Anti-cheat controls (decisive):**
  - **C1 — NO-PREDICTION-ERROR control (the boundary-condition proof).** Re-state the **SAME** fact (`dog go north` again, PE≈0). RECONSOLIDATE and DO-NOTHING must leave the memory **unchanged** (the composite phasor byte-stable to tolerance; `query_patient(dog,go)`→north). **OVERWRITE-ALWAYS would (wastefully) rewrite even here** — this is the arm that exposes "is it really PE-gated, or just last-write-wins?". A mechanism that updates identically under PE≈0 and PE-high is **last-write-wins, not reconsolidation → BOUNDARY/NEGATIVE.**
  - **C2 — Moat / never-stored control.** Correct a subject **never stored** (`elephant go west`). RECONSOLIDATE must **abstain** (no fact created, `query_patient(elephant,go)`→None). Proves reconsolidation updates a *reactivated* trace and does **not** fabricate a never-stored fact (the moat-as-plus is respected).
  - **C3 — Permuted/lesion control.** (permuted) Apply the corrective filler to a **random wrong cue** (scramble which fact the PE is computed against) → the targeted fact must **not** change and the wrong fact must **not** be silently corrupted. (lesion) Disable the `unbind` PE read (feed a constant PE=0) → the update must collapse to DO-NOTHING (proves the update is *driven by* the measured mismatch, not by the call itself).
  - **C4 — Multi-seed.** Seeds **42, 43, 44, 100, 101, 102** (the standing 6-seed rule).
- **Frozen thresholds.** `PE_LABILE` set **in advance** from the measured PE separation: a true filler-change (north→south, two random phasors) sits at mean phase-cos ≈ 0 → PE ≈ 1.0; a re-statement sits at PE ≈ 0; the gate goes at the **measured midpoint** of the same-vs-different PE distributions (the `cleanup_separated`-style placement rule already used by `OrderedPositionWM.calibrate_threshold`), **not** a value chosen to pass a downstream probe. (An optional moderate-PE "near-miss" filler may be included to map the partial-update regime, but it is **not** part of the GO bar.)

**Pre-registered verdict:**

- **GO** = on **≥5/6 seeds**: (i) post-correction query returns the corrected value with **exactly one** matching fact (RECONSOLIDATE), (ii) the untouched fact is preserved, (iii) **C1 holds** (PE≈0 re-statement leaves memory unchanged — the boundary condition is real), (iv) **C2 holds** (never-stored correction abstains, no fabrication), (v) **C3 holds** (permuted/lesion collapse); AND the **four arms are cleanly separated** (RECONSOLIDATE uniquely passes all probes; NAIVE-APPEND fails the single-fact/stale-answer probe; OVERWRITE-ALWAYS fails C1; DO-NOTHING fails the correction probe) on **6/6** seeds for the moat/C1 checks.
- **BOUNDARY** = the in-place update works but the **PE gate is seed-fragile** (C1 marginal — re-statements sometimes perturb the memory), or the partial-update (moderate-PE) regime is unstable while the full-PE case is clean. (I.e. the *capability* is there but the *boundary condition* isn't robust — honest, and still informative.)
- **NEGATIVE** = the update cannot be made distinguishable from last-write-wins (C1 fails: PE≈0 changes the memory), OR the moat breaks (C2 fails: a never-stored correction fabricates a fact). Either is the deliverable map of what the substrate can/can't do.

---

## 5. What a GO / NEGATIVE means for committing the build

- **GO →** build Option A as a small additive `RFPhasorComposer.update_on_mismatch(...)` (default-off; append-only path bit-preserved) + a correction-turn hook in `MultiTurnAgentV2` (detect "actually / no, …" or a contradicting re-assertion → route to the composer update), with the existing test suites asserting the no-confab moat and the four-arm distinguishability. Reuse-by-import, no `sim/` edit. Then the honest **follow-on tier** is named: the **synaptic-literal** reconsolidation (Option B — engram reactivation + `plasticity_window_gate` PE-driven labile window + `forget`/`consolidate` re-store), with Option C (schema-assimilation) as its low-PE partner. The GO de-risk *de-risks the PE-gate logic* that the engram tier then has to realize at the synapse — same payoff structure as every prior composer→spiking conversion in this project.
- **NEGATIVE →** the composer-layer cannot cleanly separate reconsolidation from last-write-wins (most likely failure: the PE gate is not separable because the recovered-filler phasor under bundle cross-talk is too noisy to distinguish "same" from "different" reliably at the working D — a known FHRR cross-talk regime). That is a real, citable boundary: it would say in-place fact-correction needs either a cleaner-code representation (the PPMI/learned-cortex arc) or the synaptic tier directly, and would re-rank the build toward Option B/C. Honest negative = the scientific deliverable (owner standing directive).

---

## 6. Trust-but-verify ledger (load-bearing claims to confirm before building)

| Claim | Status | Note |
|---|---|---|
| **Prediction error is *necessary* for reconsolidation/destabilization** (the boundary condition) | **CONFIRMED** | Sevenster, Beckers & Kindt 2013, *Science* 339:830 — PE governs whether a reactivated fear memory destabilizes; the "PE demarcates retrieval→reconsolidation→new-learning" transition (learnmem.cshlp.org/content/21/11/580). This is the literature basis for the C1 anti-cheat. |
| **An attractor-network reconsolidation model uses mismatch to gate labilization, with independent strengthen vs labilize mechanisms; reconsolidation=update-trace, extinction=new-trace, selected by original-vs-reexposure similarity** | **CONFIRMED** | Osan, Tort & Amaral 2011, *PLoS ONE* 6(8):e23113. Direct computational precedent + the analogy that licenses Option A (the FHRR composite as the attractor; PE-gated overwrite as mismatch-driven labilization). |
| **Reconsolidation functions as memory *updating* (integrate novel info), distinct from strengthening; novel→linked/new, repeated→strengthen** | **CONFIRMED (with nuance)** | Lee 2009 "Reconsolidation: maintaining memory relevance"; Lee himself flags the *formal* updating demonstration as debated. The cleaner "updating of hippocampal memory content" demonstration is the Rodriguez-Ortiz / Hupbach-style work (PMC2991235). Attribute updating to that line, not to Lee alone. |
| **Catalog J.27 reconsolidation, J.34 editable-retrieval/gist, D.14 engram exist and are "missing/not-a-goal" in the sim** | **CONFIRMED** | Read directly from `sim-catalog/.../feature-catalog.md` (J.27 lines ~3789, J.34 ~3903, D.14 ~1248, J.18 ~3693). J.34: "could be modeled with structural plasticity + gated reconsolidation but no project goal yet" — this arc makes it a goal. |
| **`RFPhasorComposer.store` is append-only; `query_*` returns the first match (so correction → duplicate today)** | **CONFIRMED** | Read directly (`store` appends to `self.kb`; `query_patient`/`query_agent` iterate and `return` on first match). This is the gap the build fills. |
| **The neuromodulator subsystem already has `plasticity_window_gate` + `from_rpe`** (a PE-driven plasticity window exists as a target type) | **CONFIRMED** | Read directly from `sim/neuromodulators.py`. Relevant only to the Option-B follow-on, not the cheap de-risk. |
| The exact Kandel 6e page span for J.27 (Ch 53 p 1330–1334) | **Catalog-asserted, not independently re-read** | Kandel PDF not opened this pass; the catalog entry is the source. Low risk (the mechanism, not the page number, is load-bearing). |

**Sources (web):**
- [Sevenster, Beckers & Kindt 2013, Prediction Error Governs Pharmacologically Induced Amnesia for Learned Fear (Science)](https://pubmed.ncbi.nlm.nih.gov/23413355/)
- [Prediction error demarcates the transition from retrieval, to reconsolidation, to new learning (Learn. Mem. 2014)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4201815/)
- [Osan, Tort & Amaral 2011, A Mismatch-Based Model for Memory Reconsolidation and Extinction in Attractor Networks (PLoS ONE)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3149635/)
- [On the Boundary Conditions of Avoidance Memory Reconsolidation: An Attractor Network Perspective (Neural Networks 2020)](https://www.sciencedirect.com/science/article/abs/pii/S0893608020301337)
- [Lee 2009, Reconsolidation: maintaining memory relevance (Trends Neurosci.)](https://pubmed.ncbi.nlm.nih.gov/19640595/)
- [Memory Reconsolidation Mediates the Updating of Hippocampal Memory Content (the updating demonstration)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2991235/)
