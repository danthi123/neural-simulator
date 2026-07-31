---
type: plan
status: live
date: 2026-04-29
---

# Cluster F (cerebellum) v1 — design

**Date:** 2026-04-29
**Goal:** Implement a minimal Marr-Albus-Ito cerebellar microcircuit that runs as a parallel pathway alongside the existing A+E flagship, providing CF-gated supervised error-correction on top of A+E's structural stability.
**Why now:** Three cluster-stacking attempts (A+E, A+D, A+D+E, B.3+C v1) have shown A+E is the operational ceiling for the BG/hippocampus stack. Cluster F is structurally independent of the BG/hippocampus arc and is the **most under-built cluster** in the catalog (73× citations, 5 specialty PDFs locally readable in `sim-catalog/references/textbooks/`). High catalog ROI; orthogonal failure modes.
**Scope:** v1 is the minimum-viable cerebellar circuit that can learn the supervised remapping of (state) → (action) under the existing multi-goal cheat-5 task. Not eyeblink conditioning per se — the catalog's classical use case — but the same machinery, mapped onto our existing task.

## Catalog grounding

Primary: Kandel 6e Ch 37 (cerebellum). Specialty PDFs at `sim-catalog/references/textbooks/`:
- `cerebellum-marr/Marr-1969-cerebellar-cortex.pdf` — codon coding, original (LTP-direction) PF→PC plasticity proposal
- `cerebellum-albus/Albus-1971-cerebellar-function.pdf` — sign-flipped LTD, Perceptron framing, Albus weight update rule
- `cerebellum-marr/Hesslow-2013-classical-conditioning-motor.pdf` — current eyeblink-conditioning evidence; PC pacemaker, LTD-as-sole-mechanism critique
- `cerebellum-marr/Moore-ed-2002-NeuroscientistsGuide-ClassicalConditioning.pdf` (Hesslow & Yeo Ch 4 pp 86-146) — full microcircuit, AIP-specific lesion specificity, CR/UR dissociation as gate
- (Buzsáki 2006 cerebellar-rhythms section, supplementary)

Catalog entries (read pre-design): F.01 Purkinje, F.02 Granule/PF, F.03 Mossy-fiber afferents, F.04 Climbing fiber, F.05 PF→PC LTD (Marr-Albus-Ito with sign discrepancy), F.06 DCN, F.07 Forward/inverse internal models (deferred to v2), F.08 Eyeblink (target paradigm; v1 maps the machinery onto multi-goal).

## Architecture (v1)

Five new regions stitched into the existing brain-region framework:

| Region | n_neurons | exc_fraction | Role | Catalog ref |
|---|---|---|---|---|
| `mossy_state` | 60 | 1.0 | Mossy-fiber input (single pool v1; F.03 MF stream split deferred to v2) | F.03 |
| `granule` | 250 | 1.0 | Sparse expansion code, ~3-5% active (Marr §3, Albus §IV.A) | F.02 |
| `purkinje_{N,E,S,W}` | 60 each | 0.7 | Tonic 30-80 Hz simple-spike pacemaker; per-action pool | F.01 |
| `dcn_aip_{N,E,S,W}` | 30 each | 0.8 | Tonic 40 Hz; output gate; per-action pool (AIP-equivalent per Hesslow & Yeo) | F.06 |
| `inferior_olive` | 20 | 1.0 | Sparse ~1 Hz; CF teaching signal | F.04 |

**Per-action structure**: 4 separate `purkinje_X` and `dcn_aip_X` pools (N, E, S, W) so each cerebellar circuit can learn the correct action-specific output independently. Mirrors the existing BG cascade's per-action structure for clean composition with A.

### Pathways

**Forward pathway (state → action prediction):**
- `mossy_state → granule` — sparse expansion (density 0.05, weight 8.0, plastic=False). Marr's codon recoding: 4-5 MF claws per granule, granule fires only when all-or-most claws active.
- `granule → purkinje_{N,E,S,W}` — parallel fiber to Purkinje (density 0.30, weight 1.0 initial, **plastic=True with CF-gated LTD rule**). Each granule potentially contacts each PC pool. This is the LEARNING SITE.
- `purkinje_{N,E,S,W} → dcn_aip_{N,E,S,W}` — same-action only, INHIBITORY (PC outputs are GABAergic). High weight (15.0) so PC firing strongly silences DCN. plastic=False in v1 (Mauk's two-site plasticity deferred to v2).
- `dcn_aip_{N,E,S,W} → motor_{N,E,S,W}` — same-action only, EXCITATORY (additive contribution alongside thal_X drive). Weight 8.0. plastic=False.

**Teaching pathway (error → CF):**
- `inferior_olive → purkinje_{N,E,S,W}` — climbing fiber (density 0.05 to each PC pool — sparse 1:N target, since v1 doesn't model the strict 1:1 PC:CF mapping). High weight (50.0) so each CF event evokes a strong PC complex spike.

### Plasticity rule (new)

PF→PC LTD with CF gating, per Albus 1971 §IV.C eq.4. New helper in the runner:

```
For each (PF→PC) synapse i in the granule → purkinje pathway:
  recent_pf_activity[i] := EMA of pre-synaptic PF firing in last 50 ms
  on each CF burst (post-synaptic complex spike):
    Δw_i = -η · recent_pf_activity[i] · cf_burst_intensity
  bound: w_i clamped to [0, w_max]
```

Implementation uses the existing eligibility-trace infrastructure: `cp_eligibility_trace[i]` for PF activity EMA, with the CF burst as the "reward" signal, and a sign flip for LTD direction.

Concretely, we set the PF→PC pathway's `plasticity_gate = "cerebellum_pf_pc"` and use the existing reward-modulated update path **with sign reversal**. The CF firing triggers a momentary "reward" of −1.0 on these synapses, decreasing weights. Unpaired PF activity gets no negative reward → weights stay (Albus's stability argument requires bidirectional rules; v1 ships LTD-only, accepting asymptotic weight collapse — fix in v2 with a slow LTP for unpaired PF, F.16 basket/stellate analog).

### Climbing-fiber error signal

`inferior_olive` fires when an error is detected. The error definition for the multi-goal cheat-5 task:

**Δd > 0** between consecutive steps → IO fires (agent moved away from goal).

This is the v1 simplification of "perceived motor error". More biologically faithful would be:
- v2: `Δd > 0 AND last_action == intended` — error only when the agent's action increased distance, not when something else happened
- v3: predicted vs actual sensory state mismatch (forward model, F.07)

Implementation:
- Per-step, runner computes `delta_d = current_dist - prev_dist`.
- If `delta_d > 0`, runner injects a brief excitatory current to `inferior_olive` (analogous to motor-exploration noise injection). IO fires; CF events propagate to PC pools; PF→PC LTD events fire on synapses that were active in the recent window.

### Composition with existing flags

`--enable-cluster-f-cerebellum` is opt-in. When on, the cerebellar pathway is added to the brain regions. The DCN→motor pathway provides additive contribution to motor pools, so when A is also on (closed BG loop), motor pools receive sum of (BG-driven thal contribution) + (cerebellum-driven DCN contribution).

The hypothesis: the BG learns SLOWLY via reward-modulated STDP, providing the dominant motor output. The cerebellum learns FAST via CF-gated LTD on individual error events, providing fine-grained corrections that compose additively with BG output.

## v1 explicitly OUT-OF-scope

- **Intrinsic PC timer (F.17)** — adaptive CR latency. v1 PCs use the existing `IZH2007_FS_CORTICAL_INTERNEURON` preset as a high-firing-rate stand-in. The dedicated `HH_CEREBELLAR_PURKINJE` preset exists but adopting it requires HH dt scaling (0.05ms) which is incompatible with the rest of the simulation. v2 will revisit.
- **Basket/stellate-b plasticity (F.16)** — bidirectional Perceptron-symmetry argument. v1 LTD-only is unstable long-term; we'll accept this in v1 and address in v2.
- **Three MF input streams (F.03)** — single `mossy_state` pool in v1. v2 splits into `mossy_efference`, `mossy_proprioception`, `mossy_vestibular`.
- **Nucleo-olivary feedback (F.18)** — DCN → IO inhibition. Without this, learning won't extinguish. Acceptable for v1 evaluation (multi-goal task evaluates acquisition not extinction).
- **AIP-specific output (F.06)** — collapsed `dcn_aip` to a single per-action pool; no posterior-interpositus / dentate / fastigial separation.
- **Two-site plasticity (Mauk-Medina, F.06 supplemental)** — plastic MF→DCN synapses for slow content learning, fast PF→PC for timing. v1 plastic only at PF→PC.
- **Eyeblink conditioning task harness** — v1 evaluates on multi-goal cheat-5; eyeblink is the canonical paradigm but requires a CS-US trial structure that doesn't exist in our task. v2 could add an eyeblink runner.
- **Forward/inverse internal models (F.07)** — v1 doesn't have efference copy. Skip.

## Acid tests for v1

In order of priority:

1. **Microcircuit smoke test** (mandatory): with all flags off except `--enable-cluster-f-cerebellum`, the runner builds 5 + 4 + 4 = 13 new regions and the expected pathway count without import errors or crash. Test in `tests/test_cluster_f.py`.
2. **CF firing test** (mandatory): inject a step where `delta_d > 0` and verify IO neurons fire; PC complex spikes follow; PF→PC weights for synapses active during the trigger are reduced. Probe in `research/probes/cerebellum_cf_probe.py`.
3. **A+F multi-goal det eval** (gate): does A+F (no E) match or beat A+E (6.97 ± 0.83)? Hypothesis: the supervised-learning arm catches errors A misses, mean improves.
4. **A+E+F multi-goal det eval** (gate): does adding F to A+E improve over A+E? This is the cheat-5 closure test. If A+E+F < 6.97 with stat sig, that's a real win.
5. **Phase transition recovery** (qualitative): when a goal change happens, do CF events spike during the next ~10 steps as the agent over-shoots? Per-phase finalQ should show A+F or A+E+F adapting faster on phase 1 than A+E alone.

## Acceptance criteria

- All 5 new tests in `tests/test_cluster_f.py` pass.
- Smoke run completes without errors.
- A+F or A+E+F multi-goal det (n=6) matches or beats A+E (Welch's t > 0.5 for the better config).
- Acid test: baseline (no clusters) on the same code reproduces ~7.0-7.6 (within historical noise of documented baselines), confirming Cluster F additions are behaviorally neutral when its flag is OFF.

## Implementation sequence

1. **Design doc** (this file) — DONE.
2. **Builder function** — `build_cerebellum_regions_and_pathways()` in `research/runners/g11_bg_runner.py`.
3. **CLI flag** — `--enable-cluster-f-cerebellum` with proper plumbing.
4. **CF error-signal injection** — per-step delta_d check, conditional IO drive.
5. **PF→PC LTD** — uses existing `set_plasticity_gate("cerebellum_pf_pc", ...)` infrastructure with sign-reversed reward modulation.
6. **Tests** — `tests/test_cluster_f.py`: 5 unit tests (region count, pathway count, IO firing, PF→PC LTD, integration smoke).
7. **Eval** — n=6 baseline + n=6 A+F + n=6 A+E+F = 18 runs, multi-goal det, ~90 min.
8. **Findings** — `research/findings/2026-04-29-cluster-f-results.md` with per-seed table + decision.

## Estimated wall-clock

- Builder + flags + LTD wiring: ~3 hours of code work.
- Tests: ~1 hour.
- Eval: ~90 min GPU.
- Findings: ~30 min write-up.

Total: ~5-6 hours including verification cycles.
