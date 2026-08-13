---
type: finding
status: live
date: 2026-08-13
mechanism: one-brain-merge
---

# Merge → production, RUNG 1 (GO, opt-in): the SURPRISE + WORLD-MODEL production organs run on ONE shared spiking bridge, byte-identical to co-resident — behind a default-off flag

**Date:** 2026-08-13 · **Runner:** `research/runners/_onebrain_merge_rung1_verify.py` · **Artifact:**
`research/findings/raw/_onebrain_merge_rung1_6seed.json` (6 seeds 42/43/44/100/101/102, `SIM_BACKEND=numpy`).
**Flag:** `BRAIN_ONEBRAIN_MERGE` (default **OFF**). **NO `sim/` edit** — the two merge flags already exist on
`main` (guarded, default-off); everything here is additive `research/runners/` code, reuse-by-import.

## What this is

The FIRST production rung of the one-substrate merge. Production has been CO-RESIDENCY: every Gate-B spiking
organ builds its OWN `SimulationBridge`. The merge was de-risked byte-EXACT end-to-end
(`2026-08-13-one-brain-merge-CLOSED-per-region-threshold.md`, `...-homeostasis-GO.md`, `...-Norgan-GO.md`) but
no production organ set actually shared one pool. This migrates the TWO MOST COMPATIBLE production organs — the
D2 SURPRISE expectation-violation organ (`surprise_production_organ`, `build_expectation_circuit`) and the E2
affective WORLD-MODEL organ (`worldmodel_production_organ`, `build_world_model_circuit`) — onto ONE shared
`SimulationBridge` behind an opt-in flag. OFF → each organ builds its own bridge exactly as today. ON → both
share one `cp_` array, one step, one `cfg.seed`, with the two merge flags ON
(`per_region_threshold_heterogeneity`, `per_region_homeostasis_isolation`). No cross-organ synapse is added on
this rung (the load-bearing claim is byte-identity of the two organs' reads).

**Why this pair:** their global configs are IDENTICAL where it matters (`dt_ms=1.0`, IZHIKEVICH,
GENERIC_UNSTRUCTURED, `enable_homeostasis=True`, `enable_gabab=True` with `gabab_conductance_max=0` so GABA_B is
inert in both, the same Hebbian block, `enable_nmda` unset in both), so the config SUPERSET is a trivial union
with NO genuine single-valued conflict — contrast the mapped `dt_ms`/`homeostasis` conflict of the
expectation+Wong-Wang diffbuilder pair in `...-Norgan-GO.md`. Region names are disjoint.

## Result — 6/6 GO

Every read is verified through the SAME production organ classes (`SurpriseProductionOrgan`,
`WorldModelProductionOrgan`) and their SAME public read APIs (`judge`/`read_surprise`, `expectation`/
`read_surprise`) that `brain_chat` calls; only the substrate (one bridge vs two) differs.

| axis (6 seeds) | result | verdict |
|---|---|---|
| ONE shared pool (both organs are the SAME bridge + SAME `cp_membrane_potential_v` array, N=1584) | 6/6 | GO |
| SURPRISE organ reads byte-identical merged-vs-co-resident (max delta **0.000e+00**) | 6/6 | GO |
| WORLD-MODEL organ reads byte-identical merged-vs-co-resident (max delta **0.000e+00**) | 6/6 | GO |
| surprise faculty alive on the merged bridge (contradict/confirm separation 38.7×–146.4×) | 6/6 | GO |
| world-model faculty alive on the merged bridge (pred signs +/− opposite; violated > expected surprise) | 6/6 | GO |
| determinism (`cfg.seed`; build-twice incl. `cp_neuron_firing_thresholds`) | 6/6 | GO |
| **RUNG-1 MERGE** (pool + byte-id both + alive both + determinism) | **6/6** | **GO** |

- Byte-identity is EXACT **0.0** across the whole read battery (calibration numbers + a judge/expectation/
  surprise battery) for BOTH organs, all 6 seeds. The metric has teeth: it read **0.69 Hz** on the world-model
  organ BEFORE the read-isolation fix below, and **0.0** after — a real difference it can detect, closed to zero.
- The byte-identical organs are FUNCTIONAL (not a degenerate silent-substrate identity): surprise separations
  38.7×–146.4×, world-model predicts opposite valence signs for +/− context and fires more surprise on a
  violated turn than an expected one, all 6 seeds.

## The one non-trivial seam this rung closed: a READ-TIME homeostasis coupling (not init, not training)

INIT byte-identity (`per_region_threshold_heterogeneity`) and train-time idle-drift
(`per_region_homeostasis_isolation`) were already closed by the 2-organ de-risk, and they hold here (verified:
world-model thresholds + trained state→pred weights are byte-identical merged-vs-solo, max err 0.0). A THIRD,
READ-TIME coupling surfaced for this DIFFERENT-builder pair and is the companion-process lesson again: the
world-model's FS prediction neurons (`pred_pos`), homeostatically SILENCED during build, drop their firing
threshold to ≈ **−54.4 mV** — at the resting potential **−55.0 mV** — so they fire SPONTANEOUSLY (~14 spikes)
whenever the shared substrate is stepped, INCLUDING while the co-resident surprise organ is read. The
participation-gated `per_region_homeostasis_isolation` cannot freeze them (they participate BY firing), so on
the shared, continuously-stepped substrate they receive extra intrinsic-plasticity adaptation + refractory/
previous-firing-state advances that the standalone bridge (stepped only during its OWN reads) never undergoes.
Their own organ's `_hard_reset` does not fully clear this (it resets `cp_refractory` by the wrong name and never
touches `cp_prev_firing_states`), so a ~2-spike footprint carried into the next read (the 0.69 Hz residual).

**The fix (a read-isolation guard, additive, in `onebrain_merge_production.MergedSubstrate.read_isolation`):**
snapshot the FULL per-neuron state before a read, run it (the ACTIVE organ's slice self-adapts EXACTLY as it
does standalone — that behaviour is preserved), then RESTORE the co-resident organ's slice. Each organ's neural
evolution then depends only on its OWN reads → byte-identical to the standalone organ (there is no cross synapse,
so the restored co-resident never influenced the read). This does NOT change any single organ's read semantics
(the surprise organ is byte-identical to its solo self WITH the guard active); it removes only the cross-organ
footprint. It is a no-op for the flag-off path and for a single-organ substrate (the keep-mask covers all
neurons → nothing to restore).

## No regression (flag OFF = today)

- `BRAIN_ONEBRAIN_MERGE` default-OFF → `merge_enabled()` is `False`; both organs build their own bridge with
  `shared=None` — the standalone path is untouched.
- `brain_chat_tui --smoke` is **byte-identical** to a stashed pre-change baseline (the JSON verdict compared
  field-by-field, path-normalized: equal).
- `pytest tests/test_determinism.py -q` → **9 passed**.

## Reproduce

```bash
SIM_BACKEND=numpy python -m research.runners._onebrain_merge_rung1_verify \
    --seeds 42,43,44,100,101,102 --out research/findings/raw/_onebrain_merge_rung1_6seed.json
```

## Honest scope / non-claims

- `wired: shared-substrate for 1 pair (surprise + world-model) / on_by_default: NO (opt-in, `BRAIN_ONEBRAIN_MERGE`
  default-off — this is the FIRST rung, not the production default) / scaffold_retired: the co-residency for THIS
  pair is now retirable (the shared substrate reproduces both organs' reads bit-for-bit).` Functional read-outs
  only; no phenomenal claim.
- **"Co-resident" here means each organ on its own bridge WITH the two merge flags ON** (the same threshold +
  homeostasis scoping the merge uses) — the apples-to-apples baseline that isolates one-bridge-vs-two. It is NOT
  today's flag-off production default (which has `per_region_threshold_heterogeneity` off); that default is
  unchanged and byte-identical to before (the smoke proof).
- **ONE pair only.** This is the first rung; migrating the other production organs (comprehension/metacog/affect/
  pragmatic and the rest) is a later arc, and the diffbuilder pairs with genuine `dt_ms`/`homeostasis` conflicts
  still need the mapped per-organ scoping (`...-Norgan-GO.md` [MS]).
- **The read-time coupling generalizes.** Any organ with homeostatically-spontaneously-active neurons will need
  the read-isolation guard (or a region-scoped homeostasis-during-read gate) to merge byte-exactly; the guard
  here is the demonstrated, production-read-preserving mechanism.
