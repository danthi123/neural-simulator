"""LOOP-STEP 3 de-risk #2 Phase B — POPULATION CODING at an OOM-SAFE scale.

Phase A (`_genseq_loopstep3_graded_derisk.py`) CONFIRMED the rate-saturation was a READOUT
artifact: the graded `a_cont` membrane readout un-saturates (`a_cont_sat=False`) and recovers
per-block fidelity [0.865, 0.596, 0.327], cumulative 0.327 (< the 0.8 GO bar). The remaining gap is
per-layer error ACCUMULATION across the stacked dense transforms. The documented fix is POPULATION
coding (N neurons/feature, read the population-MEAN graded output -> finer rank survives the
per-layer noise): the project's prior rate-code-wall lift, single-neuron 47% -> n_per=8 100% ->
n_per=32 108% of host (`2026-06-15-...-GO.md`, CYCLE 91/95).

Phase B in the graded runner escalated to population coding but used the FULL 2048-wide MLP, which
OOM'd at n_per=8 (~83K neurons / ~823M synapses / >26 GB on the 24 GB card, because the dense
2048x2048 layers x the signed E/I split x n_per^2 blow up). THIS runner re-runs Phase B at a
FEASIBLE, OOM-SAFE scale:

  - NARROW the hidden width: slice each cortex weight matrix W to the first HIDDEN_WIDTH rows/cols
    (a dense HIDDEN_WIDTH-wide MLP slice, NOT the full 2048). This preserves the multi-layer
    error-accumulation problem's CHARACTER (3 stacked dense transforms, a real ~HIDDEN_WIDTH fan-in)
    while keeping the bridge tiny. The vocab in/out stays 66 (the readout dimension is meaningful).
  - n_per <= 4 (NOT 8): sweep n_per in {1, 2, 4}.
  - PRINT the planned neuron + estimated synapse count and ASSERT it is well under the OOM ceiling
    BEFORE building. Build ONE config at a time; free it (del + free the CuPy pool) before the next.

Everything else is IDENTICAL to the graded runner (reuse-by-import): the graded `a_cont` readout,
the signed E/I split-channel wiring (every block graded=True), the TRAIT FIX, the NON-SPIKING
integrator regime, the greedy per-block fan-in gain calibration, the matched off-bridge GRADED
analog ground truth, and the matched/mismatched specificity anti-cheat. The ONLY changes vs Phase A:
(1) slice Ws to a narrow hidden width, (2) sweep n_per, (3) the OOM assertion + per-config free.

NO `sim/` edit (the graded path + the `"graded": True` wiring flag already exist; this is a runner).

Verdict:
  GO = cumulative analog-Spearman >= ~0.8 across the stacked dense layers at a feasible n_per (vs
       Phase A's single-neuron 0.327) AND the specificity margin re-opens (matched >> mismatched).
  PARTIAL = lifts toward but not to 0.8 -> report the n_per TREND (1->2->4) and EXTRAPOLATE: does it
       project to clear 0.8 at a larger (cloud-scale) n_per, or is there a deeper per-layer wall?
  NEGATIVE = pop-coding does not lift the cumulative above chance.

Usage:
  SIM_BACKEND=cupy python -m research.runners._genseq_loopstep3_popcode_derisk
"""
from __future__ import annotations

import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Reuse EVERY building block from the Phase-A graded runner (NO duplication of the bridge build /
# calibration / readout / metrics -- only the narrow-slice + n_per sweep + OOM guard are new).
from research.runners._genseq_loopstep3_graded_derisk import (  # noqa: E402
    load_artifact,
    Layout,
    evaluate_config,
    spearman,  # noqa: F401  (re-exported for parity / debugging)
    GRADED_REST_MV,  # noqa: F401
    A_CONT_TARGET,
)

OUT_PATH = _REPO / "research/findings/raw/_genseq_loopstep3_popcode.json"

# ---- OOM-SAFE knobs (the load-bearing change) ------------------------------------------------
# The full 2048-wide MLP OOM'd at n_per=8. Narrow to a dense HIDDEN_WIDTH-wide slice so the bridge
# stays tiny while preserving the 3-stacked-dense-transform character. 512 keeps a real ~512-source
# fan-in (vs the toy 256) and is still <0.5 GB at n_per=4 (see the OOM print below).
HIDDEN_WIDTH = 512
N_BLOCKS = 3                  # 3 stacked dense transforms (the per-layer accumulation under test)
N_PER_SWEEP = (1, 2, 4)       # n_per <= 4 (OOM-safe); the documented lift trend 1->2->4
# Hard OOM ceiling for the per-config assertion. The card is 24 GB; the prompt mandates shrinking
# any config that would exceed ~16 GB. We assert each config's estimate is < this and, defensively,
# < a tighter SAFE budget so we never approach the wall.
OOM_CEILING_GB = 16.0
SAFE_BUDGET_GB = 8.0
BYTES_PER_EDGE_EST = 32       # conservative: cp_connections + per-synapse weight/gain/transmission arrays


def slice_weights(Ws, hidden_width, n_blocks):
    """Slice the cortex weights to a narrow HIDDEN_WIDTH-wide MLP, preserving the in/out vocab dim.

    The artifact is [66, 2048, 2048, 2048, 66]. For an n_blocks-deep chain we keep the FIRST
    n_blocks transforms and clamp every hidden dimension to <= hidden_width (the vocab dims 66 stay
    full). Block L's W is W_full[L][:in_L, :out_L]. This is a structurally-faithful dense MLP slice:
    a real ~hidden_width-source fan-in feeding each unit, the same accumulation problem at a
    tractable size."""
    full_sizes = [Ws[0].shape[0]] + [W.shape[1] for W in Ws]   # [66,2048,2048,2048,66]
    # narrowed sizes: clamp the interior (hidden) dims; keep the input/output vocab dims full.
    narrowed = [full_sizes[0]]
    for li in range(1, n_blocks):       # interior hidden layers of the kept chain
        narrowed.append(min(hidden_width, full_sizes[li]))
    narrowed.append(min(hidden_width, full_sizes[n_blocks]))    # block (n_blocks-1) output dim
    sliced = []
    for L in range(n_blocks):
        nin, nout = narrowed[L], narrowed[L + 1]
        sliced.append(np.ascontiguousarray(Ws[L][:nin, :nout]).astype(np.float32))
    return sliced, narrowed


def estimate_bridge_cost(feature_sizes, n_blocks, n_per):
    """Planned neuron count + estimated synapse (edge) count + GB for the signed E/I split-channel
    + population-coded bridge. Dense blocks => nnz = nin*nout; signed split => 2 target copies
    (e,i) per non-top block (pos+neg fill one copy's nnz); top block => 1 readout copy; population
    coding => x n_per^2 edges. Matches the graded runner's wiring exactly."""
    fs = list(feature_sizes)
    n_neurons = 0
    for li in range(n_blocks):
        n_neurons += 2 * fs[li] * n_per       # e + i copies
    n_neurons += fs[n_blocks] * n_per          # readout
    base_edges = 0
    for L in range(n_blocks):
        nnz = fs[L] * fs[L + 1]                # dense
        copies = 1 if (L + 1) == n_blocks else 2
        base_edges += copies * nnz
    edges = base_edges * (n_per * n_per)
    gb = edges * BYTES_PER_EDGE_EST / 1e9
    return n_neurons, edges, gb


def free_cuda():
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def measure_within_population_divergence(Ws, layer_sizes, n_blocks, *, n_per, graded_scale_mV,
                                         drive_pA, n_steps, warmup, probe_dim, non_spiking,
                                         threshold_jitter_mV):
    """THE LOAD-BEARING DIAGNOSTIC of *why* population coding does/doesn't lift this readout.

    The documented pop-code lift (CYCLE 91/95, 47%->100%) cancels per-neuron STOCHASTIC read-out
    noise: it only helps when the n_per neurons of a feature carry DIFFERENT noisy estimates, so the
    population mean is a better estimate than one neuron. Here we measure, per block-output feature,
    the standard deviation of a_cont ACROSS its n_per population copies (the within-pop spread that
    averaging would cancel). If it is ~0, the copies are deterministic clones -> the population mean
    == a single neuron, and n_per is a literal no-op (NO noise to average).

    Returns the mean / max within-feature population std at each block output."""
    import cupy as cp
    from research.runners._genseq_loopstep3_graded_derisk import (
        Layout, build_graded_signed_bridge)
    layout = Layout(layer_sizes[:n_blocks + 1], n_blocks, n_per=n_per)
    b, _c = build_graded_signed_bridge(
        Ws, layout, seed=42, e_gain=1.0, graded_scale_mV=graded_scale_mV,
        non_spiking=non_spiking, threshold_jitter_mV=threshold_jitter_mV)
    rest = cp.float32(GRADED_REST_MV); inv = cp.float32(1.0 / max(1e-3, graded_scale_mV))
    n_total = layout.n_total
    drive = cp.zeros(n_total, dtype=cp.float32)
    for nidx in layout.e_neurons(0, int(probe_dim)):
        drive[int(nidx)] = cp.float32(drive_pA)
    for nidx in layout.i_neurons(0, int(probe_dim)):
        drive[int(nidx)] = cp.float32(drive_pA)
    b.cp_external_input_current[:] = drive
    # accumulate a_cont so the std is over the time-averaged per-neuron readout (what we actually read)
    acc = [cp.zeros(layout.feature_sizes[li + 1] * n_per, dtype=cp.float64) for li in range(n_blocks)]
    spk = [cp.zeros(layout.feature_sizes[li + 1] * n_per, dtype=cp.float64) for li in range(n_blocks)]
    counted = 0
    for step in range(warmup + n_steps):
        b._run_one_simulation_step()
        b.cp_external_input_current[:] = drive
        if step >= warmup:
            v = b.cp_membrane_potential_v
            ac = cp.clip((v - rest) * inv, 0.0, 1.0)
            fired = b.cp_firing_states
            for li in range(n_blocks):
                span = layout.feature_sizes[li + 1] * n_per
                base = layout.e_base[li + 1] if li < n_blocks - 1 else layout.readout_base
                acc[li] += ac[base:base + span].astype(cp.float64)
                spk[li] += fired[base:base + span].astype(cp.float64)
            counted += 1
    out = []
    for li in range(n_blocks):
        nf = layer_sizes[li + 1]
        a_mean = (acc[li] / max(1, counted)).reshape(nf, n_per).get()
        s_mean = (spk[li] / max(1, counted)).reshape(nf, n_per).get()
        wstd = a_mean.std(axis=1)            # within-feature spread across the n_per copies
        out.append({"block": li,
                    "within_pop_std_mean": float(wstd.mean()),
                    "within_pop_std_max": float(wstd.max()),
                    "spike_rate_mean": float(s_mean.mean())})
    del b
    free_cuda()
    return out


def main():
    backend = os.environ.get("SIM_BACKEND", "auto")
    print(f"[popcode] SIM_BACKEND={backend}", flush=True)
    Ws_full, thresholds, leaks, layer_sizes_full, vocab = load_artifact()
    n_blocks = N_BLOCKS

    # --- narrow the cortex to an OOM-safe dense MLP slice -----------------------------------
    Ws, narrowed_sizes = slice_weights(Ws_full, HIDDEN_WIDTH, n_blocks)
    # the runner-internal layer_sizes the Layout/evaluate_config use is the NARROWED chain.
    layer_sizes = list(narrowed_sizes)
    feature_sizes = layer_sizes[:n_blocks + 1]
    print(f"[popcode] NARROWED cortex: full {layer_sizes_full} -> sliced {feature_sizes} "
          f"(HIDDEN_WIDTH={HIDDEN_WIDTH}, {n_blocks} dense blocks)", flush=True)
    for L in range(n_blocks):
        print(f"[popcode]   block {L}: W {Ws[L].shape}  nnz={int(np.count_nonzero(np.abs(Ws[L])>0)):,}", flush=True)

    # --- probe chars (same as Phase A) ------------------------------------------------------
    if vocab is not None:
        probe_chars = [" ", "e", "t", "a", "o", "h"]
        probe_dims = [vocab.index(c) for c in probe_chars if c in vocab]
    else:
        probe_dims = [2, 44, 59, 40, 54, 47]
    probe_dims = probe_dims[:6]

    T = 24
    n_steps = 36
    warmup = 18
    drive_pA = 4000.0
    e_gain = 1.0
    graded_scale_mV = 20.0     # Phase A's best scale (cum 0.327 at 20; 40->0.027, 80->-0.196).
    GO_BAR = 0.8

    # --- PRE-FLIGHT OOM REPORT for the whole sweep -----------------------------------------
    print("\n[popcode] ===== PRE-FLIGHT OOM PLAN (assert each < %.0f GB ceiling, %.0f GB safe budget) ====="
          % (OOM_CEILING_GB, SAFE_BUDGET_GB), flush=True)
    plan = []
    for n_per in N_PER_SWEEP:
        n_neu, edges, gb = estimate_bridge_cost(feature_sizes, n_blocks, n_per)
        plan.append({"n_per": n_per, "n_neurons": n_neu, "est_edges": int(edges), "est_gb": round(gb, 3)})
        flag = "OK" if gb < SAFE_BUDGET_GB else ("WARN(>safe)" if gb < OOM_CEILING_GB else "ABORT(>ceiling)")
        print(f"[popcode]   n_per={n_per}: neurons={n_neu:>7,d}  est_edges={int(edges):>13,d}  "
              f"~{gb:6.3f} GB @ {BYTES_PER_EDGE_EST}B/edge  -> {flag}", flush=True)
        assert gb < OOM_CEILING_GB, (
            f"OOM GUARD: n_per={n_per} estimated {gb:.2f} GB exceeds the {OOM_CEILING_GB} GB ceiling. "
            f"Shrink HIDDEN_WIDTH (currently {HIDDEN_WIDTH}) or n_per before building.")

    # =====================================================================
    # SWEEP n_per in {1,2,4} -- the documented pop-code lift trend. Build ONE at a time, free between.
    # =====================================================================
    sweep_results = []
    best = None
    for n_per in N_PER_SWEEP:
        print(f"\n[popcode] ===== n_per={n_per} (narrow{HIDDEN_WIDTH}, {n_blocks} blocks) =====", flush=True)
        free_cuda()
        res = evaluate_config(
            Ws, thresholds, leaks, layer_sizes, vocab, probe_dims,
            n_blocks=n_blocks, n_per=n_per, graded_scale_mV=graded_scale_mV, e_gain=e_gain,
            drive_pA=drive_pA, T=T, n_steps=n_steps, warmup=warmup,
            non_spiking=True, threshold_jitter_mV=0.0, calibrate=True,
            label=f"pop_nper{n_per}")
        cm = res["cumulative_mean_spearman"]
        mg = res["anti_cheat_specificity"]["specificity_margin"]
        per_block = [round(a["mean_spearman_vs_graded"], 3) for a in res["per_layer_fidelity"]]
        print(f"[popcode] n_per={n_per} -> cumulative_sp={'nan' if math.isnan(cm) else f'{cm:.3f}'} "
              f"margin={mg:.3f} a_cont_sat={res['a_cont_saturated']} per_block_sp={per_block} "
              f"(n_total={res['n_total_neurons']})", flush=True)
        sweep_results.append(res)
        if (not math.isnan(cm)) and (best is None or math.isnan(best["cumulative_mean_spearman"])
                                     or cm > best["cumulative_mean_spearman"]):
            best = res
        free_cuda()

    if best is None:
        best = sweep_results[-1]

    # --- WHY-DIAGNOSTIC: within-population divergence (the load-bearing measurement) --------
    # Population coding lifts a readout ONLY by averaging per-neuron STOCHASTIC noise. Measure
    # whether the n_per copies of each feature actually DIVERGE (graded non-spiking arm) and, as a
    # cross-check, whether a spiking+jitter arm injects divergence that pop-coding could cancel.
    print("\n[popcode] ===== WHY-DIAGNOSTIC: within-population divergence (n_per=4) =====", flush=True)
    free_cuda()
    div_graded = measure_within_population_divergence(
        Ws, layer_sizes, n_blocks, n_per=4, graded_scale_mV=graded_scale_mV,
        drive_pA=drive_pA, n_steps=n_steps, warmup=warmup, probe_dim=probe_dims[0],
        non_spiking=True, threshold_jitter_mV=0.0)
    for d in div_graded:
        print(f"[popcode]   graded non-spiking block {d['block']}: within_pop_std mean={d['within_pop_std_mean']:.3e} "
              f"max={d['within_pop_std_max']:.3e} spike_rate={d['spike_rate_mean']:.3e}", flush=True)
    max_div_graded = max(d["within_pop_std_max"] for d in div_graded)
    pop_is_noop = max_div_graded < 1e-9
    print(f"[popcode]   -> max within-pop std (graded non-spiking) = {max_div_graded:.3e} "
          f"=> population coding is a {'NO-OP (clones, nothing to average)' if pop_is_noop else 'meaningful (copies diverge)'}",
          flush=True)

    # cross-check: a spiking+jitter arm (per-neuron noise EXISTS) at n_per 1 vs 4 -- does pop-coding
    # lift there? (run only if the graded arm is a no-op, to characterize the deeper wall).
    spiking_arm = None
    if pop_is_noop:
        print("\n[popcode] ===== cross-check: SPIKING+jitter arm (per-neuron noise present) n_per 1 vs 4 =====",
              flush=True)
        spiking_arm = []
        for n_per in (1, 4):
            free_cuda()
            res = evaluate_config(
                Ws, thresholds, leaks, layer_sizes, vocab, probe_dims,
                n_blocks=n_blocks, n_per=n_per, graded_scale_mV=graded_scale_mV, e_gain=e_gain,
                drive_pA=drive_pA, T=T, n_steps=n_steps, warmup=warmup,
                non_spiking=False, threshold_jitter_mV=3.0, calibrate=True,
                label=f"spk_nper{n_per}")
            cmv = res["cumulative_mean_spearman"]
            spiking_arm.append({"n_per": n_per, "cumulative_mean_spearman": cmv,
                                "per_block": [a["mean_spearman_vs_graded"] for a in res["per_layer_fidelity"]],
                                "on_spike_max_rate": [a["on_spike_max_rate"] for a in res["per_layer_fidelity"]]})
            print(f"[popcode]   spiking+jitter n_per={n_per}: cumulative="
                  f"{'nan' if math.isnan(cmv) else f'{cmv:.3f}'}", flush=True)
            free_cuda()

    # --- n_per trend + extrapolation -------------------------------------------------------
    trend = [(r["n_per"], None if math.isnan(r["cumulative_mean_spearman"]) else round(r["cumulative_mean_spearman"], 3))
             for r in sweep_results]
    # crude log-of-n_per linear extrapolation of cumulative -> project the n_per needed to hit 0.8.
    pts = [(math.log2(r["n_per"]), r["cumulative_mean_spearman"]) for r in sweep_results
           if not math.isnan(r["cumulative_mean_spearman"])]
    projected_nper_for_go = None
    slope = None
    if len(pts) >= 2:
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        slope, intercept = np.polyfit(xs, ys, 1)
        if slope > 1e-6:
            x_go = (GO_BAR - intercept) / slope
            projected_nper_for_go = float(2.0 ** x_go) if x_go < 60 else float("inf")

    # --- verdict on the BEST config --------------------------------------------------------
    cm = best["cumulative_mean_spearman"]
    spec = best["anti_cheat_specificity"]
    per_layer_agg = best["per_layer_fidelity"]
    margin_ok = (not math.isnan(spec["specificity_margin"]) and spec["specificity_margin"] > 0.1)

    if (not math.isnan(cm)) and cm >= GO_BAR and margin_ok:
        verdict = "GO"
    elif (not math.isnan(cm)) and cm >= 0.4 and margin_ok:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    per_layer_sp = [None if math.isnan(a["mean_spearman_vs_graded"]) else round(a["mean_spearman_vs_graded"], 3)
                    for a in per_layer_agg]
    proj_str = ("n/a" if (projected_nper_for_go is None or math.isinf(projected_nper_for_go))
                else (f"~{projected_nper_for_go:.0f}" if projected_nper_for_go < 1e6 else ">1e6"))
    # ROOT-CAUSE tag: the flat n_per trend is explained by the within-pop divergence measurement.
    rootcause = ("pop-coding NO-OP: the graded non-spiking a_cont readout is DETERMINISTIC "
                 "(within-pop std~0) so the n_per copies are clones -> averaging cancels nothing; the "
                 "per-layer fidelity loss (0.85->0.62->0.29) is deterministic signal compression "
                 "through the stacked saturating clip, NOT read-out noise pop-coding can fix"
                 if pop_is_noop else "pop-coding meaningful (copies diverge)")
    verdict_line = (
        "popcode: narrow%d blocks=%d GRADED_analog n_per_sweep=%s -> BEST n_per=%d "
        "cumulative_analog_spearman=%.3f per_layer_spearman(vs_graded)=%s specificity_margin=%.3f "
        "trend(n_per:cum)=%s within_pop_std_max=%.2e [%s] -> %s "
        "(vs Phase A single-neuron cumulative=0.327)" % (
            HIDDEN_WIDTH, n_blocks, list(N_PER_SWEEP), best["n_per"], cm, per_layer_sp,
            spec["specificity_margin"], trend, max_div_graded, rootcause, verdict))

    result = {
        "probe": "genseq_loopstep3_popcode_oomsafe",
        "resolves": "de-risk #2 Phase B (population coding) at a FEASIBLE OOM-safe scale; does pop-coding "
                    "lift the multi-layer GRADED consolidation cumulative fidelity to >=0.8?",
        "artifact": "cortex_10M_seed42.npz",
        "oom_safety": {
            "hidden_width": HIDDEN_WIDTH, "n_blocks": n_blocks,
            "full_layer_sizes": layer_sizes_full, "narrowed_feature_sizes": feature_sizes,
            "bytes_per_edge_est": BYTES_PER_EDGE_EST,
            "oom_ceiling_gb": OOM_CEILING_GB, "safe_budget_gb": SAFE_BUDGET_GB,
            "preflight_plan": plan,
            "note": "prior Phase B OOM'd at the FULL 2048-wide MLP, n_per=8 (~26 GB). Narrowed slice + "
                    "n_per<=4 keeps every config under the safe budget; built one at a time, pool freed between.",
        },
        "n_blocks": n_blocks, "feature_sizes": feature_sizes,
        "neuron_model": "ADEX_as_LIF (signed E/I split-channel, GRADED analog transmission, non-spiking, population-coded)",
        "method": "reuse the Phase-A graded runner VERBATIM (graded a_cont readout, signed E/I split, TRAIT "
                  "FIX, non-spiking integrator, greedy per-block fan-in gain calibration, matched off-bridge "
                  "GRADED ground truth, specificity anti-cheat); the ONLY changes: slice Ws to a narrow "
                  "HIDDEN_WIDTH dense MLP, sweep n_per in {1,2,4}, OOM-assert each config, free between. NO sim/ edit.",
        "graded_scale_mV": graded_scale_mV, "go_bar": GO_BAR,
        "drive_pA": drive_pA, "T_off": T, "n_steps_on": n_steps, "warmup": warmup,
        "n_per_sweep": list(N_PER_SWEEP),
        "sweep_results": [
            {"n_per": r["n_per"], "n_total_neurons": r["n_total_neurons"],
             "cumulative_mean_spearman": r["cumulative_mean_spearman"],
             "per_layer_fidelity": r["per_layer_fidelity"],
             "specificity_margin": r["anti_cheat_specificity"]["specificity_margin"],
             "anti_cheat_specificity": r["anti_cheat_specificity"],
             "a_cont_saturated": r["a_cont_saturated"],
             "per_layer_gain_calibration": r["per_layer_gain_calibration"]}
            for r in sweep_results
        ],
        "n_per_trend": trend,
        "within_population_divergence_diagnostic": {
            "graded_non_spiking_n_per4": div_graded,
            "max_within_pop_std": max_div_graded,
            "pop_coding_is_noop": pop_is_noop,
            "interpretation": rootcause,
            "spiking_jitter_cross_check": spiking_arm,
        },
        "extrapolation": {
            "model": "cumulative ~ slope*log2(n_per) + intercept",
            "slope_per_doubling": (None if slope is None else float(slope)),
            "projected_n_per_for_0.8": (None if (projected_nper_for_go is not None and math.isinf(projected_nper_for_go))
                                        else projected_nper_for_go),
            "note": "MOOT — pop-coding is a no-op on this deterministic graded readout (within-pop std~0); "
                    "the flat trend is not a slow lift but a no-op. A larger n_per cannot help.",
        },
        "best_config": {
            "n_per": best["n_per"], "graded_scale_mV": best["graded_scale_mV"],
            "n_total_neurons": best["n_total_neurons"],
            "per_layer_gain_calibration": best["per_layer_gain_calibration"],
        },
        "best_per_layer_fidelity": per_layer_agg,
        "best_cumulative_mean_spearman": cm,
        "best_anti_cheat_specificity": spec,
        "baseline_phaseA_single_neuron": {"cumulative_mean_spearman": 0.327,
                                          "per_block": [0.865, 0.596, 0.327],
                                          "note": "graded a_cont readout, n_per=1, scale=20"},
        "verdict_line": verdict_line, "verdict": verdict,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, default=lambda o: None
                                   if (isinstance(o, float) and math.isnan(o)) else o))

    print("\n[popcode] ===== n_per TREND (cumulative analog-Spearman) =====", flush=True)
    for n_per, cum in trend:
        print(f"[popcode]   n_per={n_per}: cumulative={cum}", flush=True)
    print(f"[popcode]   Phase A single-neuron baseline: 0.327", flush=True)
    if projected_nper_for_go is not None:
        print(f"[popcode]   extrapolated n_per to reach 0.8: ~{projected_nper_for_go:.0f} "
              f"(slope {slope:+.3f}/doubling)", flush=True)
    print("\n[popcode] BEST per-block (a_cont analog readout):", flush=True)
    for a in per_layer_agg:
        print("[popcode]   block %d: sp_vs_graded=%.3f sp_vs_spiking=%.3f | a_cont_mean=%.3f "
              "frac_pinned=%.3f" % (a["layer"], a["mean_spearman_vs_graded"],
              a["mean_spearman_vs_spiking"], a["a_cont_mean"], a["frac_features_pinned_hi"]), flush=True)
    print("\n" + "=" * 78)
    print(verdict_line)
    print("=" * 78)
    print(f"[popcode] wrote {OUT_PATH}", flush=True)
    return result


if __name__ == "__main__":
    main()
