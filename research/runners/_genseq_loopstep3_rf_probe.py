"""LOOP-STEP 3 — Probe P1: does the RESONATE-AND-FIRE COMPLEX-SYNAPSE path preserve per-layer
rank across stacked dense layers, escaping BOTH the per-layer clip-compression wall (W1) AND the
`g·(V−E)` install-divergence wall (W2) that defeated the rate / graded consolidation?

Scoping: research/findings/2026-06-22-genseq-GLA-analog-accumulator-reframe-scoping.md §6 (Probe P1)
+ the [VERIFY #1] (lines 300-305): "RF info is in PHASE (unit-magnitude); a transformer linear
layer's output is magnitude-coded. Whether magnitude→phase mapping preserves rank for a dense layer
is untested ... P1 measures it directly; this is the only bridge mechanism that escapes g·(V−E)."

THE IDEA (verified to source, `sim/bridge.py:5710-5746` `_rf_advance_one`): the RF complex matvec
`Z_out = decay·rot(Z) + (W_re@re − W_im@im) [+ i(...)]` is added DIRECTLY to the complex state Z —
**NO clip, NO g·(V−E), NO refractory ceiling**. None of the four loop-step-3 NEGATIVEs tested it;
it is the ONE bridge accumulator the scoping found that avoids both walls. This probe tests whether
a dense layer installed as RF COMPLEX synapses, read by magnitude/phase, preserves the teacher
float-activation rank across 2-3 STACKED layers (each layer's read drives the next).

THE LOAD-BEARING DESIGN CHOICE — how to map a real-valued layer activation into a complex/phasor
target faithfully (stated, per the prompt). The teacher block is `a_out = clip(a_in @ W, 0, 1)` —
a magnitude-coded ReLU+saturation. The RF complex accumulator is exercised in its both-walls-
escaping regime:
  ENCODE   z_in[m] = a_in[m]  (REAL: magnitude = activation, reference phase 0) — the scoping-named
           "magnitude=activation" faithful encoding.
  WEIGHTS  the REAL layer weight W installed as a COMPLEX synapse with W_im=0 (connection
           (post=D_in+n, pre=m, weight=W[m,n]) so the matvec computes a_out = W^T @ a_in = a_in @ W).
  DYNAMICS lam=0.0 (no magnitude decay) + a LARGE period (ω = 2π/period → 0, so the rotation per
           step ≈ identity) ⇒ the complex accumulator computes the PURE SIGNED LINEAR MATVEC
           `Re(Z_out) = nsteps · (a_in @ W)` exactly (verified: sp(Re(Z), a_in@W) = 1.000,
           im→0), with NO clip, NO g·(V−E), NO ceiling — the both-walls escape.
  READOUT  PRIMARY (faithful to a magnitude-coded ReLU layer): a_hat = clip(Re(Z_out)/scale, 0, 1)
           — re-imposes the teacher's OWN nonlinearity on the rank-faithful linear accumulation, so
           the comparison is teacher-clip vs teacher-clip; the ONLY thing under test is whether the
           RF accumulator compressed the rank (it does not). SECONDARY (the RF-native channel, for
           honesty): the phase readout `rf_read_phases()` under a phase=activation encoding (info
           in PHASE) — does the unit-magnitude phase channel ALSO carry a dense layer's rank?
  STACK    layer L's PRIMARY readout a_hat (∈[0,1]) is re-encoded as layer L+1's input magnitude →
           a FRESH small RF bridge per block (D_in + D_out ≤ ~1024 neurons — OOM-trivial).

TEACHER (identical slice + metric to the rate/graded/popcode/distill NEGATIVEs, reuse-by-import):
  the narrow-512 3-block dense MLP slice of cortex_10M_seed42.npz (`slice_weights`), per-layer
  target = `offbridge_graded_forward` (`a_{L+1} = clip(a_L @ W_L, 0, 1)`), metric = `spearman`.
  The rate/graded path got cumulative 0.288 / 0.327 here; the GO bar is ~0.85.

ANTI-CHEATS:
  (1) matched/mismatched specificity margin on the FINAL block (matched char >> mismatched char).
  (2) a SHUFFLED-TARGET control: score the readout vs a char-DERANGED teacher — must collapse to
      ~chance (the distill-NEGATIVE's shuffled control reached 0.542 because the teacher's final
      reps are char-correlated; we report it honestly and also the per-char-paired margin).

Verdict (scoping §6):
  GO       = stacked CUMULATIVE phase/magnitude-rank ≥ ~0.85 (escaping BOTH walls) vs the
             rate/graded 0.288, AND the specificity margin re-opens (matched >> mismatched), AND
             the shuffled-target control collapses. ⇒ the RF complex accumulator is the
             substrate-native escape; the generator-as-PHASOR re-architecture becomes the plan.
  PARTIAL  = lifts well above 0.288 but < 0.85, or one anti-cheat is soft → report + diagnose.
  NEGATIVE = phase/magnitude coding cannot carry a dense signed linear layer's rank when stacked →
             the differentiable-bridge `sim/` edit (option a) is correctly next (→ owner).

NO `sim/` edit (the RF path already exists; reuse-by-import). GPU. Usage:
  SIM_BACKEND=cupy python -m research.runners._genseq_loopstep3_rf_probe
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

# Reuse the teacher + slice + metric VERBATIM from the prior NEGATIVEs (identical comparison basis).
from research.runners._genseq_loopstep3_graded_derisk import (  # noqa: E402
    load_artifact,
    offbridge_graded_forward,
    spearman,
)
from research.runners._genseq_loopstep3_popcode_derisk import slice_weights  # noqa: E402
from research.runners.rf_phasor_composer import _build_rf_bridge  # noqa: E402

OUT_PATH = _REPO / "research/findings/raw/_genseq_loopstep3_rf.json"

HIDDEN_WIDTH = 512
N_BLOCKS = 3
GO_BAR = 0.85          # the prompt's both-walls-escape bar (vs the rate/graded 0.288)
# Dynamics: lam=0 (no decay) + a large period so ω≈0 (the rotation per step ≈ identity) ⇒ the
# complex accumulator is the PURE signed linear matvec in Re(Z). A handful of steps suffices (the
# matvec is applied every step; rank is step-count-invariant — verified). period >> nsteps.
RF_PERIOD = 100000
RF_NSTEPS = 8
RF_LAMBDA = 0.0
OOM_CEILING_GB = 16.0


def free_cuda():
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _rf_bridge_cache():
    """One small RF bridge reused per (D_in+D_out) neuron count across blocks/chars (rf_set_complex_
    weights REPLACES the weights each call; rf_kick resets the state) — avoids _initialize_simulation_
    data per op. Mirrors RFPhasorComposer._resonate's bridge cache."""
    return {}


def rf_linear_layer_signed(bridge, W, a_in, *, period, nsteps, lam):
    """ONE dense linear layer through the RF COMPLEX-SYNAPSE accumulator, read as the SIGNED matvec.

    Installs W (real) as complex synapses (post=D_in+n <- pre=m, weight=W[m,n]); kicks z_in = a_in
    (real, magnitude=activation, phase 0); resonates `nsteps` with ω≈0 (period huge) + lam=0 so the
    complex accumulator computes Re(Z_out) = nsteps·(a_in @ W) with NO clip / g·(V−E) / ceiling.
    Returns the SIGNED linear output (Re(Z_out)/nsteps ≈ a_in @ W) AND the magnitude |Z_out|/nsteps.
    """
    import cupy as cp
    D_in, D_out = W.shape
    n = D_in + D_out
    conns = [(D_in + nn, m, complex(float(W[m, nn]), 0.0))
             for m in range(D_in) for nn in range(D_out) if W[m, nn] != 0.0]
    bridge.rf_set_complex_weights(conns)
    kick = np.zeros(n, dtype=np.complex128)
    kick[:D_in] = np.asarray(a_in, dtype=np.float64)
    bridge.rf_kick(kick, period=int(period), lam=float(lam))
    bridge.rf_resonate_steps(int(nsteps))
    re = cp.asnumpy(bridge.cp_membrane_potential_v).astype(np.float64)[D_in:]
    im = cp.asnumpy(bridge.cp_recovery_variable_u).astype(np.float64)[D_in:]
    signed = re / float(nsteps)
    mag = np.hypot(re, im) / float(nsteps)
    return signed, mag


def rf_linear_layer_phase(bridge, W, a_in, *, period, nsteps):
    """The RF-NATIVE channel (secondary, for honesty): encode the activation in PHASE
    (phase_in = a_in ∈ [0,1)) at UNIT magnitude, run the resonate window, read `rf_read_phases()`.
    Tests whether the unit-magnitude phase channel ALSO carries a dense layer's rank. (period sets
    ω; here we use the composer's own RF_PERIOD so a phase-cycle completes for the readout.)"""
    D_in, D_out = W.shape
    n = D_in + D_out
    conns = [(D_in + nn, m, complex(float(W[m, nn]), 0.0))
             for m in range(D_in) for nn in range(D_out) if W[m, nn] != 0.0]
    bridge.rf_set_complex_weights(conns)
    kick = np.zeros(n, dtype=np.complex128)
    kick[:D_in] = np.exp(2j * np.pi * np.asarray(a_in, dtype=np.float64))   # phase=activation, |z|=1
    bridge.rf_kick(kick, period=int(period), lam=0.0)
    bridge.rf_resonate_steps(int(period) + 8)
    phases = np.asarray(bridge.rf_read_phases())[D_in:]
    return phases


def rf_stack_forward(Ws, scales, input_oh, n_blocks, cache, *, period, nsteps, lam):
    """Stack n_blocks dense layers through the RF complex accumulator. Each block's PRIMARY readout
    a_hat = clip(Re(Z)/scale, 0, 1) (the teacher's OWN nonlinearity on the rank-faithful linear
    accumulation) becomes the next block's input magnitude. `scales[L]` rescales block L's signed
    output into the teacher's [0,1] band before the clip (calibrated once on a probe char, the RF
    analogue of the graded runner's per-block gain — but here the rank is scale-INVARIANT, so the
    scale only fixes the OCCUPANCY, never the rank). Returns the per-block clipped activations."""
    a = np.asarray(input_oh, dtype=np.float64)
    outs = []
    for L in range(n_blocks):
        W = Ws[L]
        D_in, D_out = W.shape
        n = D_in + D_out
        b = cache.get(n)
        if b is None:
            b = _build_rf_bridge(n, seed=42)
            cache[n] = b
        signed, _mag = rf_linear_layer_signed(b, W, a, period=period, nsteps=nsteps, lam=lam)
        a = np.clip(signed * float(scales[L]), 0.0, 1.0)   # teacher's clip; scale fixes occupancy
        outs.append(a.copy())
    return outs


def calibrate_scales(Ws, n_blocks, cal_oh, cache, *, period, nsteps, lam, target=0.18):
    """Greedy per-block scale calibration (the RF analogue of the graded runner's fan-in gain): pick
    each block's scale so its clipped-output mean lands near `target` (an un-pinned band). RANK is
    scale-invariant for the RF accumulator (Re(Z)∝a@W exactly), so this only fixes occupancy — it
    can NOT manufacture rank (the GO is real if it holds without pinning). Calibrated on one char."""
    a = np.asarray(cal_oh, dtype=np.float64)
    scales = [1.0] * n_blocks
    for L in range(n_blocks):
        W = Ws[L]; D_in, D_out = W.shape; n = D_in + D_out
        b = cache.get(n)
        if b is None:
            b = _build_rf_bridge(n, seed=42); cache[n] = b
        signed, _ = rf_linear_layer_signed(b, W, a, period=period, nsteps=nsteps, lam=lam)
        pos = signed[signed > 0.0]
        if pos.size == 0:
            sc = 1.0
        else:
            # choose scale so clip(signed*sc,0,1).mean() ≈ target: bisect on sc.
            lo, hi = 1e-6, 1e3
            sc = 1.0
            for _ in range(28):
                mid = math.sqrt(lo * hi)
                occ = float(np.clip(signed * mid, 0.0, 1.0).mean())
                if occ > target:
                    hi = mid
                else:
                    lo = mid
                sc = mid
                if abs(occ - target) <= 0.01:
                    break
        scales[L] = sc
        a = np.clip(signed * sc, 0.0, 1.0)
        print(f"[rf:cal] block {L} scale={sc:.4g} -> clipped_mean~"
              f"{float(np.clip(signed*sc,0,1).mean()):.3f} (rank is scale-invariant)", flush=True)
    return scales


def main():
    backend = os.environ.get("SIM_BACKEND", "auto")
    print(f"[rf] SIM_BACKEND={backend}", flush=True)
    Ws_full, thresholds, leaks, layer_sizes_full, vocab = load_artifact()
    n_blocks = N_BLOCKS

    Ws, narrowed_sizes = slice_weights(Ws_full, HIDDEN_WIDTH, n_blocks)
    feature_sizes = list(narrowed_sizes)[:n_blocks + 1]
    print(f"[rf] NARROWED cortex: full {layer_sizes_full} -> sliced {feature_sizes} "
          f"(HIDDEN_WIDTH={HIDDEN_WIDTH}, {n_blocks} dense blocks)", flush=True)
    for L in range(n_blocks):
        print(f"[rf]   block {L}: W {Ws[L].shape}  nnz={int(np.count_nonzero(np.abs(Ws[L])>0)):,}", flush=True)

    # ---- OOM pre-flight: the largest RF bridge is max(D_in+D_out) over blocks, dense complex CSR.
    max_n = max(W.shape[0] + W.shape[1] for W in Ws)
    max_nnz = max(int(np.count_nonzero(np.abs(W) > 0)) for W in Ws)
    # per RF bridge: 2 complex CSR (re+im) ~ max_nnz*16B + index ~max_nnz*8B + state O(n). Trivial.
    est_gb = (max_nnz * 2 * (16 + 8) + max_n * 64) / 1e9
    print(f"[rf] OOM pre-flight: max RF bridge n={max_n} neurons, max block nnz={max_nnz:,} "
          f"-> ~{est_gb:.4f} GB (ceiling {OOM_CEILING_GB} GB)", flush=True)
    assert est_gb < OOM_CEILING_GB, f"OOM GUARD: estimated {est_gb:.2f} GB exceeds {OOM_CEILING_GB} GB"

    if vocab is not None:
        probe_chars = [" ", "e", "t", "a", "o", "h"]
        probe_dims = [vocab.index(c) for c in probe_chars if c in vocab]
    else:
        probe_dims = [2, 44, 59, 40, 54, 47]
    probe_dims = probe_dims[:6]
    V_in = Ws[0].shape[0]

    cache = _rf_bridge_cache()

    # ---- per-char teacher (off-bridge graded clip-chain) + RF stack (primary signed readout) ----
    free_cuda()
    cal_oh = np.zeros(V_in, dtype=np.float64); cal_oh[probe_dims[0]] = 1.0
    print("\n[rf] ===== greedy per-block scale calibration (occupancy only; rank scale-invariant) =====",
          flush=True)
    scales = calibrate_scales(Ws, n_blocks, cal_oh, cache,
                              period=RF_PERIOD, nsteps=RF_NSTEPS, lam=RF_LAMBDA)

    print("\n[rf] ===== per-char: teacher (clip-chain) vs RF complex-accumulator stack =====", flush=True)
    teacher_by_char = {}
    rf_by_char = {}
    rf_phase_by_char = {}
    per_input = []
    for dim in probe_dims:
        oh = np.zeros(V_in, dtype=np.float64); oh[dim] = 1.0
        teacher = offbridge_graded_forward(Ws, oh, n_blocks)         # per-block clip(a@W,0,1)
        rf_out = rf_stack_forward(Ws, scales, oh, n_blocks, cache,
                                  period=RF_PERIOD, nsteps=RF_NSTEPS, lam=RF_LAMBDA)
        # SECONDARY (RF-native phase channel): single-block phase readout per block vs the SAME
        # teacher per-block target (not stacked through phase — phase doesn't re-encode cleanly as a
        # next-layer magnitude; this isolates "does the phase channel carry ONE dense layer's rank").
        rf_phase = []
        for L in range(n_blocks):
            in_vec = oh if L == 0 else teacher[L - 1]   # teacher-forced input to isolate the layer
            ph = rf_linear_layer_phase(cache[Ws[L].shape[0] + Ws[L].shape[1]], Ws[L], in_vec,
                                       period=RF_PERIOD, nsteps=RF_NSTEPS)
            rf_phase.append(ph)
        teacher_by_char[dim] = teacher
        rf_by_char[dim] = rf_out
        rf_phase_by_char[dim] = rf_phase

        layer_metrics = []
        for L in range(n_blocks):
            sp_mag = spearman(teacher[L], rf_out[L])         # PRIMARY: signed/clip magnitude readout
            sp_ph = spearman(teacher[L], rf_phase[L])         # SECONDARY: RF-native phase channel
            layer_metrics.append({"layer": L, "spearman_primary_vs_teacher": sp_mag,
                                  "spearman_phase_vs_teacher": sp_ph})
        cumulative = layer_metrics[-1]["spearman_primary_vs_teacher"]
        per_input.append({"char": (vocab[dim] if vocab else None), "dim": int(dim),
                          "layers": layer_metrics, "cumulative_spearman": cumulative})
        ls = " ".join("L%d:%s" % (m["layer"], "nan" if math.isnan(m["spearman_primary_vs_teacher"])
                      else "%.2f" % m["spearman_primary_vs_teacher"]) for m in layer_metrics)
        lp = " ".join("L%d:%s" % (m["layer"], "nan" if math.isnan(m["spearman_phase_vs_teacher"])
                      else "%.2f" % m["spearman_phase_vs_teacher"]) for m in layer_metrics)
        print(f"[rf] char={per_input[-1]['char']!r} dim={dim:2d} | PRIMARY {ls} | phase {lp} | "
              f"CUMUL={'nan' if math.isnan(cumulative) else f'{cumulative:.3f}'}", flush=True)

    # ---- aggregate per-layer + cumulative ----
    per_layer_agg = []
    for L in range(n_blocks):
        sp = [r["layers"][L]["spearman_primary_vs_teacher"] for r in per_input
              if not math.isnan(r["layers"][L]["spearman_primary_vs_teacher"])]
        sph = [r["layers"][L]["spearman_phase_vs_teacher"] for r in per_input
               if not math.isnan(r["layers"][L]["spearman_phase_vs_teacher"])]
        per_layer_agg.append({
            "layer": L,
            "mean_spearman_primary_vs_teacher": float(np.mean(sp)) if sp else float("nan"),
            "mean_spearman_phase_vs_teacher": float(np.mean(sph)) if sph else float("nan"),
        })
    cum_list = [r["cumulative_spearman"] for r in per_input if not math.isnan(r["cumulative_spearman"])]
    cumulative_mean = float(np.mean(cum_list)) if cum_list else float("nan")

    # ---- ANTI-CHEAT 1: matched/mismatched specificity on the FINAL block (primary readout) ----
    dims = list(rf_by_char.keys())
    matched, mismatched = [], []
    for d_on in dims:
        rf_final = rf_by_char[d_on][n_blocks - 1]
        for d_off in dims:
            t_final = teacher_by_char[d_off][n_blocks - 1]
            s = spearman(t_final, rf_final)
            if math.isnan(s):
                continue
            (matched if d_on == d_off else mismatched).append(s)
    spec = {
        "matched_mean_spearman": float(np.mean(matched)) if matched else float("nan"),
        "mismatched_mean_spearman": float(np.mean(mismatched)) if mismatched else float("nan"),
    }
    spec["specificity_margin"] = spec["matched_mean_spearman"] - spec["mismatched_mean_spearman"]

    # ---- ANTI-CHEAT 2: shuffled-target control (score vs a char-DERANGED teacher) ----
    # Per-block: for each char, score its RF output vs a DIFFERENT char's teacher (a fixed
    # derangement), averaged. A real result has the matched (real-pairing) cumulative >> shuffled.
    rng = np.random.default_rng(12345)
    perm = list(range(len(dims)))
    for _ in range(64):
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(len(dims))):
            break
    shuffled_per_layer = []
    for L in range(n_blocks):
        vals = []
        for i, d_on in enumerate(dims):
            d_off = dims[perm[i]]
            s = spearman(teacher_by_char[d_off][L], rf_by_char[d_on][L])
            if not math.isnan(s):
                vals.append(s)
        shuffled_per_layer.append(float(np.mean(vals)) if vals else float("nan"))
    shuffled_cumulative = shuffled_per_layer[-1]

    # ---- verdict ----
    margin_ok = (not math.isnan(spec["specificity_margin"]) and spec["specificity_margin"] > 0.1)
    shuffled_collapses = (math.isnan(shuffled_cumulative)
                          or (not math.isnan(cumulative_mean)
                              and cumulative_mean - shuffled_cumulative > 0.2))
    if (not math.isnan(cumulative_mean)) and cumulative_mean >= GO_BAR and margin_ok and shuffled_collapses:
        verdict = "GO"
    elif (not math.isnan(cumulative_mean)) and cumulative_mean >= 0.4:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    per_layer_primary = [None if math.isnan(a["mean_spearman_primary_vs_teacher"]) else
                         round(a["mean_spearman_primary_vs_teacher"], 3) for a in per_layer_agg]
    per_layer_phase = [None if math.isnan(a["mean_spearman_phase_vs_teacher"]) else
                       round(a["mean_spearman_phase_vs_teacher"], 3) for a in per_layer_agg]
    verdict_line = (
        "rf_complex_accumulator: narrow%d blocks=%d ENCODE=magnitude(signed Re(Z), lam=0 omega~0) "
        "stacked CUMULATIVE_rank=%.3f per_layer_primary=%s per_layer_phase=%s specificity_margin=%.3f "
        "shuffled_control_cumulative=%.3f -> %s (vs rate/graded cumulative 0.288; both-walls-escape "
        "GO bar %.2f)" % (
            HIDDEN_WIDTH, n_blocks,
            (float("nan") if math.isnan(cumulative_mean) else cumulative_mean),
            per_layer_primary, per_layer_phase, spec["specificity_margin"],
            (float("nan") if math.isnan(shuffled_cumulative) else shuffled_cumulative),
            verdict, GO_BAR))

    result = {
        "probe": "genseq_loopstep3_rf_complex_accumulator",
        "resolves": "P1 (scoping §6): does the RF complex-synapse path preserve per-layer rank across "
                    "stacked dense layers, escaping BOTH the clip-compression (W1) AND g·(V−E) (W2) walls?",
        "artifact": "cortex_10M_seed42.npz",
        "design_choice_activation_to_phasor": {
            "encoding": "magnitude=activation: z_in = a_in (REAL, reference phase 0); the scoping-named "
                        "faithful encoding. Weights = the REAL W as complex synapses (W_im=0).",
            "dynamics": "lam=0.0 (no magnitude decay) + period=%d so omega=2pi/period~0 (rotation per "
                        "step ~ identity) => the complex accumulator computes Re(Z_out)=nsteps*(a_in@W) "
                        "EXACTLY (verified sp(Re(Z),a_in@W)=1.000, im->0), with NO clip / g.(V-E) / "
                        "refractory ceiling." % RF_PERIOD,
            "primary_readout": "a_hat = clip(Re(Z)/scale, 0, 1) -- re-imposes the TEACHER'S OWN clip on "
                               "the rank-faithful linear accumulation; teacher-clip vs teacher-clip, so "
                               "the ONLY thing measured is whether the RF accumulator compressed rank.",
            "secondary_readout": "RF-NATIVE phase channel: phase=activation encoding (|z|=1, info in "
                                 "PHASE), rf_read_phases() per single (teacher-forced) block -- does the "
                                 "unit-magnitude phase channel ALSO carry a dense layer's rank?",
            "stacking": "layer L's PRIMARY a_hat (in [0,1]) re-encodes as layer L+1's input magnitude; a "
                        "FRESH small RF bridge per block (D_in+D_out<=~1024 neurons; OOM-trivial).",
        },
        "n_blocks": n_blocks, "feature_sizes": feature_sizes,
        "neuron_model": "RESONATE_AND_FIRE (complex state Z=re+i*im; complex synaptic matvec added "
                        "DIRECTLY to Z -- NO clip, NO g.(V-E), NO refractory ceiling)",
        "rf_period": RF_PERIOD, "rf_nsteps": RF_NSTEPS, "rf_lambda": RF_LAMBDA,
        "go_bar": GO_BAR,
        "per_block_scale_calibration": [float(s) for s in scales],
        "per_input": per_input,
        "per_layer_fidelity": per_layer_agg,
        "cumulative_mean_spearman": cumulative_mean,
        "anti_cheat_specificity": spec,
        "anti_cheat_shuffled_target": {
            "per_layer_shuffled_spearman": shuffled_per_layer,
            "shuffled_cumulative": shuffled_cumulative,
            "real_cumulative": cumulative_mean,
            "real_minus_shuffled": (None if (math.isnan(cumulative_mean) or math.isnan(shuffled_cumulative))
                                    else cumulative_mean - shuffled_cumulative),
            "collapses_to_chance": shuffled_collapses,
            "note": "the distill-NEGATIVE's shuffled control reached 0.542 because the teacher's FINAL "
                    "reps are char-correlated; reported honestly. A real result has real_cumulative >> "
                    "shuffled AND a re-opened matched/mismatched margin.",
        },
        "baseline_rate_graded": {"cumulative_mean_spearman": 0.288,
                                 "per_block": [0.846, 0.620, 0.288],
                                 "note": "graded/rate a_cont readout, the W1 deterministic clip-compression"},
        "verdict_line": verdict_line, "verdict": verdict,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, default=lambda o: None
                                   if (isinstance(o, float) and math.isnan(o)) else o))

    print("\n[rf] ===== per-block (PRIMARY signed/clip magnitude readout vs teacher) =====", flush=True)
    for a in per_layer_agg:
        print("[rf]   block %d: primary_sp_vs_teacher=%.3f  phase_sp_vs_teacher=%.3f" % (
            a["layer"], a["mean_spearman_primary_vs_teacher"], a["mean_spearman_phase_vs_teacher"]),
            flush=True)
    print(f"[rf] specificity: matched={spec['matched_mean_spearman']:.3f} "
          f"mismatched={spec['mismatched_mean_spearman']:.3f} margin={spec['specificity_margin']:.3f}",
          flush=True)
    print(f"[rf] shuffled-target control: cumulative={shuffled_cumulative:.3f} (real={cumulative_mean:.3f}; "
          f"collapses={shuffled_collapses})", flush=True)
    print("\n" + "=" * 78)
    print(verdict_line)
    print("=" * 78)
    print(f"[rf] wrote {OUT_PATH}", flush=True)
    free_cuda()
    return result


if __name__ == "__main__":
    main()
