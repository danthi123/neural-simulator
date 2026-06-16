# Unified-agent Stage 1: the parser-silence co-residence bug (root-cause debug)

**Date:** 2026-06-16
**Status:** root cause FALSIFIED + REVISED + localized to a structural co-residence regression; mechanism A/B in flight.
**Context:** the "unified embodied agent" arc — STEP 2b merged nav+conversation+composer onto one `SimulationBridge`; STEP-1 of the *generalization* add (`build_merged_nav_conv_bridge(co_resident_generalization=True)`) appends a structured-perception → NMDA-concept → fact stack and a rate-Hebbian convergence. The Stage-1 no-regression gate crashed.

## The symptom

`_unified_stage1_merged.py` (the Stage-1 runner) crashes in its no-regression gate:

```
agent.hear("dog chase cat")  ->  composer.store(roles["agent"], roles["action"], ...)
KeyError: 'action'
```

`parse_on_slices` builds `roles` as a **dict-comprehension keyed by the decoded role** (`nav_conv_merged_bridge.py:163`): `{role_of(pos): word[pos] for pos in range(3)}`. If two positions decode to the **same** role, a key is dropped, and `hear()`'s unsafe `roles["action"]` (vs the shipped tests' safe `.get()`) raises `KeyError`.

## Prior root cause — FALSIFIED by code reading (the debugging Iron Law)

The pre-compaction hypothesis was: *"the gen-stack `inject_explicit_wiring` rebuilds the plasticity-gate map → the parser's `parser_fixed` freeze is wiped → the gen convergence training (global Hebbian) drifts the parser weights → the parse regresses."*

This is **wrong**, provably, from the code:
- There is exactly **one** `inject_explicit_wiring` (`build_merged_nav_conv_bridge`, line 603), **before** both the parser-train pass (621) and the gen-convergence pass (637). The parser freeze (`set_plasticity_gate(PARSER_GATE, 0.0)`, line 627) is **never re-injected away**.
- `_train_merged_convergence` holds **all** non-gen synapses at `cp_plasticity_rate_gain = 0` (only perception→concept plastic) for the whole pass, and `train_convergence` (`_genfrontier_onsubstrate_convergence_derisk.py:197`) **only** sets external current + steps the bridge — it never touches `cp_connections` / `set_pathway_weights` / the gain array.
- ⇒ the parser weights are **byte-identical** gen-OFF vs gen-ON. Gen training **cannot** drift the parser.

Lesson: the pre-compaction "high-confidence" root cause did not match the code. Falsify-by-reading before any fix.

## DIAG1 — is it gen-specific, a tie, or out-of-vocab? (`_unified_stage1_parser_diag.py`, GPU)

Builds the merged bridge gen-OFF and gen-ON; decodes **both** `"dog chase cat"` (the failing sentence) and `"dog go north"` (the shipped-test sentence), 3 trials each, at default (reset 60 / test 80) vs long (reset 200 / test 160) settle.

| build | "dog chase cat" | "dog go north" |
|---|---|---|
| **gen-OFF** | parses perfectly (action ensemble ~1.18, all trials/settles) | parses perfectly |
| **gen-ON** | every position → 'agent', **all role rates EXACTLY 0.00** | same — **all role rates 0.00** |

- gen-OFF proves the parser is **positional** (word identity is irrelevant — `"chase"` out-of-vocab is a non-issue) and there is **no pre-existing tie**.
- gen-ON **silences the parser** deterministically; long settle does **not** help.
- ⇒ a genuine **structural** gen-ON co-residence regression — not weights (proven), not OU (identical post-build), not a tie.

## DIAG2 — localize the silence (`_unified_stage1_parser_diag2.py`, GPU)

One gen-ON build; five localizers:

| check | result |
|---|---|
| reproduce the read | role rates 0.00 (silent) ✓ |
| free-run firing/region | gen_concept **0.0** (NOT a runaway attractor); only gen_fact 0.05; parse_conj 0.006 |
| trained conj→role weights | **present + strong**: 720 edges, mean\|w\| 6.96, max 20.0 |
| drive the action conjunction | **conj fires (rate 20.0)** but role rates **0.00** |
| OU forced ON | does **not** rescue |
| hard membrane reset (v←−65) | does **not** rescue |

⇒ **structural net-suppression of parse_role**: the conjunction fires, the excitatory weights onto parse_role are present and strong, yet parse_role fires zero — and neither noise nor a hard reset rescues it.

## Key structural facts (from reading the builder)

- The parser has **no inhibitory neurons and zero internal density** (`parser_regions_pathways`: both `exc_fraction=1.0`, `internal_density=0.0`). parse_role's **only** designed input is the conj→role excitation. So suppression **cannot** originate inside the parser.
- The gen-stack edges are **confined** to gen regions (`_build_generalization_edges`: `pc_post = gen_concept`, `fc_post = gen_fact`) — **no** accidental wiring into parse_role.
- The merged cfg has **STP OFF** (`enable_short_term_plasticity=False`), the **neuromodulator subsystem OFF** (no `enable_neuromodulator_subsystem`), no transmission gate on the parser pathway, no `input_mean`/`divisive`/`dendritic` global terms. ⇒ the conj→role **current path is ungated and identical** gen-OFF vs gen-ON.

So by elimination from reading, the suppressor is **not**: weights, accidental wiring, the inhibitory-index set (gen adds no inhibitory neurons), OU, STP, `nm_gain`, transmission gating, or any opt-in global term. The remaining space is a **network-size-dependent / global firing effect** on parse_role.

## DIAG3 — the decisive A/B (`_unified_stage1_parser_diag3.py`, GPU, in flight)

Builds gen-OFF and gen-ON; under the **same** conj drive measures parse_role's membrane V + received `g_e`/`g_i`, plus the count of excitatory vs inhibitory synapses INTO parse_role. Distinguishes: wiring-differs / same-wiring-more-inhibition / conj-current-not-delivered / deeper-global-term.

**gen-OFF (the working reference):**
- parse_role input wiring: **720 excitatory edges, 0 inhibitory** (sum_w_exc 12417).
- driven state: **v_max 34.8** (spikes past the +30 threshold), **role_fire 0.38**, conj_fire 20.0.
- **metric caveat:** `g_e`/`g_i` both read **0.0 even while parse_role is firing** — the conj→role current is not surfaced through `cp_conductance_g_e` (delivered via a different path or consumed within the step before the read). ⇒ the decisive gen-ON metrics are **parse_role v_max** (does it reach +30?) and the **inhibitory-edge count**, NOT g_e/g_i.

**NOTE — build slowdown:** diag3's gen-OFF build took **445 s** vs ~44 s in diag1 for the same bridge (≈10×). Root-caused to **sustained desktop GPU contention** (nvidia-smi: 97–99% util, ~6 GB, from non-python Electron/browser graphics contexts — the user's own GPU work, not a CuPy leak). Every GPU run is ~10× slow while this persists.

## CPU wiring check — the suppression is NOT a wiring bug (`_unified_stage1_wiring_check.py`, no GPU)

parse_role's incoming edges are fixed at injection time, so the wiring question is answerable by the union wiring plan alone — a pure CPU computation (`RegionManager.initialize` + `build_wiring_plan`), instant and immune to the GPU contention. Result (seed 42):

| build | edges into parse_role | exc / inh | pre-region |
|---|---|---|---|
| gen-OFF | 720 | 720 / 0 | parse_conj |
| gen-ON | 720 | 720 / 0 | parse_conj |

`conj_role_wiring_identical = true`, no extra edges, no inhibitory edges. ⇒ **the parser silence is NOT a wiring bug.** Combined with diag3's weight sums it becomes a **weight-magnitude** bug:
- gen-OFF parse_role incoming sum_w = **12417** over 720 edges → mean **17.2** (and a strong tail; v_max 34.8 → fires).
- gen-ON conj→role sum = **5008** → mean **6.96**, **max EXACTLY 20.0**.

## ROOT CAUSE — the UNGATED Hebbian weight clip (`sim/bridge.py:6509`)

The per-step Hebbian block gates the weight **decay** by `cp_plasticity_rate_gain` (line 6505: a frozen pathway at gain 0 does **not** decay), but the immediately-following **clip** is **ungated**:

```python
cp.clip(self.cp_connections.data, cfg.hebbian_min_weight, cfg.hebbian_max_weight, out=...)   # line 6509 — EVERY weight
```

`_train_merged_convergence` temporarily lowers `hebbian_max_weight → 20.0` for the gen perception→concept Hebbian. During its 320 Hebbian-on steps, the ungated clip crushes the **frozen** parser's load-bearing conj→role edges (legitimately ~40–60 — the strong edges that drive the role ensembles) down to ≤20. **Smoking gun:** diag2's gen-ON conj→role **max = exactly 20.0**; clipping the strong tail collapses the mean 17.2 → 6.96, dropping parse_role below its tight firing margin (gen-OFF v_max 34.8, only ~5 mV over threshold) → silence.

This is the **same** ungated-clip foot-gun the unification 5a de-risk documented (mitigated for nav/parser by `stdp_w_max = hebbian_max_weight = 400`) — but the convergence pass re-lowers `hebbian_max` to 20, re-exposing it for the frozen pathways.

## THE FIX (runner-side, no `sim/` edit, default-off-safe)

In `_train_merged_convergence`: snapshot `bridge.cp_connections.data` before the pass; in the `finally`, byte-restore every **non-`conv_mask`** synapse (parser, nav, dlPFC, concept→fact). Only the gen perception→concept edges (`conv_mask`) legitimately train; everything else is restored verbatim, so the parser keeps its strong conj→role edges and its firing margin. The pass only runs when `co_resident_generalization=True`, so the shipped byte-identity path is untouched.

**GPU behavioral confirmation:** re-run `_unified_stage1_parser_diag` on the fixed builder (expect gen-ON now parses both sentences identically to gen-OFF; conj→role max restored > 20), then the full `_unified_stage1_merged.py` no-regression gate. (Slow under the current desktop GPU contention.)

## Why this matters (honest framing)

The no-regression gate **worked**: it caught a real co-residence regression that the byte-identity check (region bases) could not see — the parser's *function* breaks when the generalization stack shares the bridge, even though its bases, weights, and current path are unchanged. The science risk this Stage-1 was meant to test (does the perception→concept convergence survive co-residence with nav + the parser) is reachable only after this is fixed. An honest negative here (the parser cannot co-reside with the gen stack without mitigation) is itself a deliverable about what the shared substrate supports.

## Artifacts

- Runners: `research/runners/_unified_stage1_parser_diag{,2,3}.py`
- Raw: `research/findings/raw/_unified_stage1_parser_diag{,2,3}.{log,json}`
- Commits: `b424906c` (falsify + diag1), `44597864` (diag2 localization) — both remotes.
