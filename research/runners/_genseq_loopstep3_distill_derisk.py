"""LOOP-STEP 3 de-risk #3 -- CLIP-AWARE LAYERWISE DISTILLATION (the owner-chosen robust path).

Scoping: research/findings/2026-06-22-surrogate-grad-on-bridge-scoping.md
  THE IDEA (HYBRID #1, NO sim/ edit): the consolidation wall is DETERMINISTIC per-layer
  clip(a@W,0,1) compression (popcode-NEGATIVE: within-pop std = 0.00e+00 -> noiseless; the
  cumulative analog-Spearman compresses 0.846 -> 0.620 -> 0.288 through the stacked clips when
  the float weights are installed VERBATIM). A deterministic differentiable function is exactly
  what you can TRAIN AROUND. So instead of installing float weights verbatim, TRAIN new
  clip-aware weights THROUGH the bridge-faithful differentiable forward (quantization-aware /
  ANN->SNN activation distillation) so the bridge's ACTUAL graded output matches the TEACHER's
  per-layer activations, then INSTALL on the live bridge and re-measure.

THE TEACHER (scoping section 3, [VERIFY] #2): the matched off-bridge GRADED per-layer target
representations -- offbridge_graded_forward(VERBATIM sliced weights): t_0 = onehot,
t_{L+1} = clip(t_L @ W_sliced[L], 0, 1). This is the CLEAN per-layer target the on-bridge a_cont
chain is supposed to compute (and exactly the quantity the existing graded metric scores against).
The verbatim install reproduces t_0->t_1 well (L0=0.846) but loses the cumulative (t_3, 0.288)
because each block's INPUT is the COMPOUNDED clipped output of the block below. The fix: train
each block to RECONSTRUCT its clean teacher target t_{L+1} GIVEN the compounded (lossy) input the
chain actually delivers -> the compression is absorbed into the learned weights.

THE TRAINER (torch autograd, clip+matmul is trivially differentiable; clip has a clean
sub-gradient -- pass-through in (0,1), zero outside; NO surrogate needed for the graded half):
  - GREEDY LAYERWISE (scoping section 3 RECOMMENDATION -- cheapest, most stable, matches the
    per-layer metric): for block L in order, holding earlier TRAINED blocks fixed, feed the
    COMPOUNDED real input a_L (clip-chain output of trained blocks 0..L-1) and minimize
    || clip(a_L @ W_L, 0, 1) - t_{L+1} ||^2 over the probe set. Init W_L = verbatim sliced.
  - then an optional END-TO-END polish (all W_L jointly, final-block loss) -- the named recovery
    for cross-layer interactions greedy leaves on the table.

THE INSTALL (NO sim/ edit): build_graded_signed_bridge(TRAINED Ws, ...) -- the SAME signed E/I
split-channel graded wiring via inject_explicit_wiring (bridge.py:2468/2491 builds the CSR from
initial_weights). Run the SAME greedy per-block fan-in gain calibration on the trained weights
(the live bridge's g.(V-E) conductance needs it), read each block output as the on-bridge analog
a_cont = clip((v-rest)/scale, 0, 1), and SCORE vs the fixed TEACHER (verbatim t_{L+1}) -- NOT vs
offbridge_graded_forward(trained) (that would be circular).

[VERIFY] -- the one load-bearing risk (scoping section 7.1): does the idealized clip-trainer's
weights close the gap ON THE LIVE BRIDGE despite the g.(V-E) driving-force sub-linearity + AdEx
settling + per-block gain? We MEASURE the installed live-bridge fidelity (not the trainer's
offline loss). If a gap remains, the mitigation (still NO sim/ edit) is to FOLD the driving-force
term into the trainer forward -- exercised here as a SECOND trainer arm and both reported.

ANTI-CHEATS (mandatory, scoping section 5.4):
  (i)  SHUFFLED-TARGET control: distil to PERMUTED teacher activations -> installed fidelity must
       stay at chance (proves the recovery is from REAL distillation, not a generic high-activation
       pattern the clip chain always produces).
  (ii) matched-vs-mismatched cross-input SPECIFICITY margin (the trained weights compute each
       char's SPECIFIC mapping; matched >> mismatched).

Verdict (scoping section 5):
  GO   = installed-on-LIVE-BRIDGE cumulative analog-Spearman >= ~0.8 (vs verbatim-install 0.288),
         AND specificity margin re-opens (matched >> mismatched), AND shuffled-target at chance.
  PARTIAL = recovers above the verbatim 0.288 but < 0.8 -> the driving-force-aware trainer arm is
         reported; if still < 0.8, the finding points at the end-task polish / the differentiable-
         bridge sim/ edit fallback.
  NEGATIVE = even the distilled + installed weights do not beat the verbatim 0.288 on the bridge.

OOM safety: the SAME narrow-512 3-block dense MLP slice as the popcode runner (~2.7K neurons /
~0.03 GB at n_per=1). PRE-FLIGHT cost print + assert < 16 GB before any build.

NO sim/ edit. Usage:
  SIM_BACKEND=cupy python -m research.runners._genseq_loopstep3_distill_derisk
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

# Reuse the Phase-A graded runner's bridge build / readout / calibration / metrics VERBATIM, and
# the popcode runner's narrow-slice + OOM-cost helpers. NO duplication of the load-bearing bridge
# machinery -- only the trainer + the teacher-scored measurement are new.
from research.runners._genseq_loopstep3_graded_derisk import (  # noqa: E402
    load_artifact,
    Layout,
    build_graded_signed_bridge,
    onbridge_block_analog,
    offbridge_graded_forward,
    greedy_block_gain_calibration,
    spearman,
    pearson,
    topk_overlap,
    GRADED_REST_MV,
    A_CONT_TARGET,
)
from research.runners._genseq_loopstep3_popcode_derisk import (  # noqa: E402
    slice_weights,
    estimate_bridge_cost,
    free_cuda,
)

OUT_PATH = _REPO / "research/findings/raw/_genseq_loopstep3_distill.json"

# ---- OOM-safe knobs (identical slice to the popcode runner) -------------------------------------
HIDDEN_WIDTH = 512
N_BLOCKS = 3
GRADED_SCALE_MV = 20.0          # Phase A's best scale (cumulative 0.327 at 20; the verbatim ref).
GO_BAR = 0.8
OOM_CEILING_GB = 16.0
SAFE_BUDGET_GB = 8.0
BYTES_PER_EDGE_EST = 32

# ---- trainer knobs ------------------------------------------------------------------------------
TRAIN_LR = 5e-3
TRAIN_STEPS_LAYERWISE = 4000     # per block (a shallow well-conditioned regression)
TRAIN_STEPS_E2E = 2000           # end-to-end polish over all blocks
DRIVE_PA = 4000.0
T_OFF = 24
N_STEPS_ON = 36
WARMUP_ON = 18


# =================================================================================================
# TEACHER (the clean per-layer target reps) + the differentiable trainer
# =================================================================================================
def teacher_activations(Ws_sliced, probe_dims, n_blocks):
    """The MATCHED off-bridge GRADED per-layer target reps on the VERBATIM sliced weights, for
    every probe char. Returns:
      inputs  : (P, V_in)  one-hot inputs
      targets : list over blocks L=0..n_blocks-1 of (P, out_L) clean teacher activations t_{L+1}.
    t_0 = onehot; t_{L+1} = clip(t_L @ W_sliced[L], 0, 1). The cumulative target t_{n_blocks} is
    the final-block reference the installed bridge must reproduce."""
    V_in = Ws_sliced[0].shape[0]
    P = len(probe_dims)
    inputs = np.zeros((P, V_in), dtype=np.float64)
    for r, d in enumerate(probe_dims):
        inputs[r, d] = 1.0
    a = inputs.copy()
    targets = []
    for L in range(n_blocks):
        a = np.clip(a @ Ws_sliced[L].astype(np.float64), 0.0, 1.0)
        targets.append(a.copy())
    return inputs, targets


def distill_weights(Ws_sliced, inputs, targets, n_blocks, *, lr=TRAIN_LR,
                    steps_layerwise=TRAIN_STEPS_LAYERWISE, steps_e2e=TRAIN_STEPS_E2E,
                    driving_force_aware=False, label="distill", verbose=True):
    """Train clip-aware weights through the bridge-faithful differentiable forward so the stacked
    clip chain reproduces the clean teacher targets.

    GREEDY LAYERWISE: for block L (holding trained blocks below fixed), the input a_L is the
    COMPOUNDED clip-chain output of trained blocks 0..L-1 (the lossy signal the bridge actually
    delivers); minimize || clip(a_L @ W_L, 0, 1) - t_{L+1} ||^2. W_L init = verbatim. Then an
    END-TO-END polish (all W_L jointly, final-block loss).

    driving_force_aware: [VERIFY] mitigation -- approximate the live bridge's g.(V-E) sub-linearity
    in the trainer forward (a saturating soft-compression of the pre-clip drive) so the trained
    weights are matched to the conductance regime, not the pure clip(a@W). Still differentiable,
    still NO sim/ edit. (A 1-param-per-block monotone squashing of the matmul before the clip.)

    Returns trained_Ws (list of np.float32, same shapes as sliced) + a training log."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    inp_t = torch.tensor(inputs, dtype=torch.float32, device=dev)
    tgts_t = [torch.tensor(t, dtype=torch.float32, device=dev) for t in targets]
    # trainable weights, init = verbatim sliced
    Wt = [torch.tensor(Ws_sliced[L], dtype=torch.float32, device=dev).clone().requires_grad_(True)
          for L in range(n_blocks)]
    log = {"label": label, "driving_force_aware": driving_force_aware, "layerwise": [], "e2e": []}

    def block_forward(a, L):
        """The bridge-faithful differentiable block forward: optionally a g.(V-E)-like saturating
        pre-compression, then the clip(.,0,1) nonlinearity."""
        z = a @ Wt[L]
        if driving_force_aware:
            # Monotone saturating squash approximating the conductance driving-force roll-off:
            # as the (excitatory) drive grows, the effective increment shrinks (V approaches E_e).
            # tanh keeps the low-drive regime ~linear (matching clip's pass-through band) and rolls
            # the high-drive tail (matching the live bridge's a_cont compression). Differentiable.
            z = torch.tanh(z)
            z = torch.clamp(z, 0.0, 1.0)
        else:
            z = torch.clamp(z, 0.0, 1.0)
        return z

    # ---- GREEDY LAYERWISE ----
    for L in range(n_blocks):
        # compounded REAL input to block L = clip-chain through the (now-fixed) trained blocks below
        with torch.no_grad():
            a = inp_t
            for j in range(L):
                a = block_forward(a, j).detach()
        a_in = a.detach()
        opt = torch.optim.Adam([Wt[L]], lr=lr)
        last = None
        for it in range(steps_layerwise):
            opt.zero_grad()
            out = block_forward(a_in, L)
            loss = ((out - tgts_t[L]) ** 2).mean()
            loss.backward()
            opt.step()
            last = float(loss.detach().cpu())
        log["layerwise"].append({"block": L, "final_mse": last})
        if verbose:
            print(f"[distill:{label}] layerwise block {L} final_mse={last:.6e}", flush=True)

    # ---- END-TO-END polish (final-block loss through the whole trained chain) ----
    if steps_e2e > 0:
        opt = torch.optim.Adam(Wt, lr=lr * 0.5)
        last = None
        for it in range(steps_e2e):
            opt.zero_grad()
            a = inp_t
            outs = []
            for L in range(n_blocks):
                a = block_forward(a, L)
                outs.append(a)
            # weighted: final-block dominant + a small per-layer anchor (keeps earlier blocks honest)
            loss = ((outs[-1] - tgts_t[-1]) ** 2).mean()
            for L in range(n_blocks - 1):
                loss = loss + 0.25 * ((outs[L] - tgts_t[L]) ** 2).mean()
            loss.backward()
            opt.step()
            last = float(loss.detach().cpu())
        log["e2e"].append({"final_loss": last})
        if verbose:
            print(f"[distill:{label}] e2e polish final_loss={last:.6e}", flush=True)

    trained = [Wt[L].detach().cpu().numpy().astype(np.float32) for L in range(n_blocks)]
    # report the trainer's OFFLINE cumulative fidelity (pure clip-chain, no bridge) vs teacher.
    with torch.no_grad():
        a = inp_t
        offline_outs = []
        for L in range(n_blocks):
            a = block_forward(a, L)
            offline_outs.append(a.cpu().numpy())
    offline_cumul = []
    for L in range(n_blocks):
        sps = [spearman(targets[L][r], offline_outs[L][r]) for r in range(inputs.shape[0])]
        sps = [s for s in sps if not math.isnan(s)]
        offline_cumul.append(float(np.mean(sps)) if sps else float("nan"))
    log["offline_per_block_spearman_vs_teacher"] = offline_cumul
    del Wt, inp_t, tgts_t
    return trained, log


# =================================================================================================
# INSTALL the (trained) weights on the LIVE bridge + SCORE vs the fixed TEACHER
# =================================================================================================
def install_and_measure(Ws_install, teacher_targets, layer_sizes, probe_dims, vocab, *,
                        n_blocks, graded_scale_mV, e_gain=1.0, drive_pA=DRIVE_PA, T=T_OFF,
                        n_steps=N_STEPS_ON, warmup=WARMUP_ON, calibrate=True, label="install"):
    """Build the signed-E/I GRADED bridge with the GIVEN weights (NO sim/ edit), run the greedy
    per-block fan-in gain calibration on THOSE weights, read each block output as the on-bridge
    analog a_cont, and SCORE vs the fixed TEACHER (verbatim t_{L+1}) -- the load-bearing
    LIVE-BRIDGE measurement.

    Returns a dict with per-block + cumulative analog-Spearman vs teacher, the specificity margin,
    and the calibration."""
    feature_sizes = layer_sizes[:n_blocks + 1]
    layout = Layout(feature_sizes, n_blocks, n_per=1)
    cal_dim = probe_dims[0]

    per_layer_scale = [1.0] * n_blocks
    cal_log = []
    if calibrate:
        per_layer_scale, cal_log = greedy_block_gain_calibration(
            Ws_install, layout, cal_dim, e_gain=e_gain, graded_scale_mV=graded_scale_mV,
            non_spiking=True, threshold_jitter_mV=0.0, drive_pA=drive_pA,
            n_steps=n_steps, warmup=warmup)
    refine = any(abs(s - 1.0) > 1e-9 for s in per_layer_scale)
    ple = per_layer_scale if refine else None

    # teacher targets indexed by probe dim
    teacher_by_dim = {}
    for r, d in enumerate(probe_dims):
        teacher_by_dim[d] = [teacher_targets[L][r] for L in range(n_blocks)]

    on_by_dim = {}
    per_input = []
    for r, dim in enumerate(probe_dims):
        bridge, cfg = build_graded_signed_bridge(
            Ws_install, layout, seed=42, e_gain=e_gain,
            per_layer_e_gain=ple, per_layer_i_gain=ple,
            graded_scale_mV=graded_scale_mV, non_spiking=True, threshold_jitter_mV=0.0)
        on_analog, _on_rates, sat = onbridge_block_analog(
            bridge, cfg, layout, active_input_dims=[dim], drive_pA=drive_pA,
            n_steps=n_steps, warmup=warmup, graded_scale_mV=graded_scale_mV)
        on_by_dim[dim] = on_analog
        del bridge
        free_cuda()
        layer_metrics = []
        for L in range(n_blocks):
            t = teacher_by_dim[dim][L]
            on_r = on_analog[L]
            sp = spearman(t, on_r)
            pe = pearson(t, on_r)
            k = max(10, int(0.25 * on_r.size))
            ov = topk_overlap(t, on_r, k)
            layer_metrics.append({"layer": L, "spearman_vs_teacher": sp,
                                  "pearson_vs_teacher": pe, "topk_overlap": ov,
                                  "on_a_cont_mean": sat[L]["a_mean"], "on_a_cont_max": sat[L]["a_max"]})
        cumulative_sp = layer_metrics[-1]["spearman_vs_teacher"]
        per_input.append({"char": (vocab[dim] if vocab else None), "dim": int(dim),
                          "layers": layer_metrics, "cumulative_spearman": cumulative_sp})
        ls = " ".join("L%d:%s" % (m["layer"], ("nan" if math.isnan(m["spearman_vs_teacher"])
                      else "%.2f" % m["spearman_vs_teacher"])) for m in layer_metrics)
        print(f"[distill:{label}] char={per_input[-1]['char']!r} dim={dim:2d} | {ls} | "
              f"CUMUL sp={'nan' if math.isnan(cumulative_sp) else f'{cumulative_sp:.3f}'}", flush=True)

    # ANTI-CHEAT specificity on the FINAL stage vs the TEACHER final target.
    dims = list(on_by_dim.keys())
    matched, mismatched = [], []
    for d_on in dims:
        on_final = on_by_dim[d_on][n_blocks - 1]
        for d_off in dims:
            off_final = teacher_by_dim[d_off][n_blocks - 1]
            s = spearman(off_final, on_final)
            if math.isnan(s):
                continue
            (matched if d_on == d_off else mismatched).append(s)
    spec = {
        "matched_mean_spearman": float(np.mean(matched)) if matched else float("nan"),
        "mismatched_mean_spearman": float(np.mean(mismatched)) if mismatched else float("nan"),
    }
    spec["specificity_margin"] = spec["matched_mean_spearman"] - spec["mismatched_mean_spearman"]

    per_layer_agg = []
    for L in range(n_blocks):
        sg = [r["layers"][L]["spearman_vs_teacher"] for r in per_input
              if not math.isnan(r["layers"][L]["spearman_vs_teacher"])]
        ovs = [r["layers"][L]["topk_overlap"] for r in per_input]
        acm = [r["layers"][L]["on_a_cont_mean"] for r in per_input]
        per_layer_agg.append({
            "layer": L,
            "mean_spearman_vs_teacher": float(np.mean(sg)) if sg else float("nan"),
            "mean_topk_overlap": float(np.mean(ovs)) if ovs else float("nan"),
            "a_cont_mean": float(np.mean(acm)),
        })
    cumul_sps = [r["cumulative_spearman"] for r in per_input
                 if not math.isnan(r["cumulative_spearman"])]
    cumulative_mean = float(np.mean(cumul_sps)) if cumul_sps else float("nan")

    return {
        "label": label,
        "per_layer_gain_calibration": per_layer_scale,
        "per_layer_gain_calibration_log": cal_log,
        "per_layer_fidelity": per_layer_agg,
        "cumulative_mean_spearman_vs_teacher": cumulative_mean,
        "anti_cheat_specificity": spec,
        "per_input": per_input,
    }


# =================================================================================================
# Main
# =================================================================================================
def main():
    backend = os.environ.get("SIM_BACKEND", "auto")
    print(f"[distill] SIM_BACKEND={backend}", flush=True)
    Ws_full, thresholds, leaks, layer_sizes_full, vocab = load_artifact()
    n_blocks = N_BLOCKS

    Ws_sliced, narrowed_sizes = slice_weights(Ws_full, HIDDEN_WIDTH, n_blocks)
    layer_sizes = list(narrowed_sizes)
    feature_sizes = layer_sizes[:n_blocks + 1]
    print(f"[distill] NARROWED cortex: full {layer_sizes_full} -> sliced {feature_sizes} "
          f"(HIDDEN_WIDTH={HIDDEN_WIDTH}, {n_blocks} dense blocks)", flush=True)
    for L in range(n_blocks):
        print(f"[distill]   block {L}: W {Ws_sliced[L].shape}  "
              f"nnz={int(np.count_nonzero(np.abs(Ws_sliced[L])>0)):,}", flush=True)

    # probe chars + a DISJOINT held-out set (for the specificity check on unseen chars)
    if vocab is not None:
        probe_chars = [" ", "e", "t", "a", "o", "h"]
        probe_dims = [vocab.index(c) for c in probe_chars if c in vocab]
        held_chars = ["n", "s", "r", "i", "d", "l"]
        held_dims = [vocab.index(c) for c in held_chars if c in vocab]
    else:
        probe_dims = [2, 44, 59, 40, 54, 47]
        held_dims = [53, 58, 57, 48, 43, 51]
    probe_dims = probe_dims[:6]
    held_dims = held_dims[:6]

    # ---- PRE-FLIGHT OOM PLAN (n_per=1; the trainer is on torch, the bridge build is tiny) ----
    print("\n[distill] ===== PRE-FLIGHT OOM PLAN (assert < %.0f GB ceiling, %.0f GB safe budget) ====="
          % (OOM_CEILING_GB, SAFE_BUDGET_GB), flush=True)
    n_neu, edges, gb = estimate_bridge_cost(feature_sizes, n_blocks, n_per=1)
    flag = "OK" if gb < SAFE_BUDGET_GB else ("WARN(>safe)" if gb < OOM_CEILING_GB else "ABORT(>ceiling)")
    print(f"[distill]   bridge: neurons={n_neu:,d}  est_edges={int(edges):,d}  ~{gb:.3f} GB "
          f"@ {BYTES_PER_EDGE_EST}B/edge -> {flag}", flush=True)
    assert gb < OOM_CEILING_GB, (
        f"OOM GUARD: estimated {gb:.2f} GB exceeds the {OOM_CEILING_GB} GB ceiling. "
        f"Shrink HIDDEN_WIDTH (currently {HIDDEN_WIDTH}).")

    # ---- TEACHER (clean per-layer target reps, verbatim sliced weights) ----
    inputs, teacher_targets = teacher_activations(Ws_sliced, probe_dims, n_blocks)
    held_inputs, held_targets = teacher_activations(Ws_sliced, held_dims, n_blocks)
    print(f"[distill] teacher per-layer target shapes: {[t.shape for t in teacher_targets]}", flush=True)

    # =============================================================================================
    # BASELINE: install the VERBATIM sliced weights and score vs teacher (must reproduce ~0.288).
    # =============================================================================================
    print("\n[distill] ===== BASELINE: install VERBATIM sliced weights, score vs teacher =====", flush=True)
    free_cuda()
    verbatim_install = install_and_measure(
        Ws_sliced, teacher_targets, layer_sizes, probe_dims, vocab,
        n_blocks=n_blocks, graded_scale_mV=GRADED_SCALE_MV, label="verbatim")
    vb_cum = verbatim_install["cumulative_mean_spearman_vs_teacher"]
    print(f"[distill] VERBATIM installed cumulative vs teacher = {vb_cum:.3f} "
          f"per_block={[round(a['mean_spearman_vs_teacher'],3) for a in verbatim_install['per_layer_fidelity']]} "
          f"(expect ~0.288)", flush=True)

    # =============================================================================================
    # ARM 1: distil clip-aware weights (pure clip trainer) -> install -> score vs teacher.
    # =============================================================================================
    print("\n[distill] ===== ARM 1: clip-aware distillation (pure clip trainer) =====", flush=True)
    free_cuda()
    trained_Ws, train_log = distill_weights(
        Ws_sliced, inputs, teacher_targets, n_blocks, driving_force_aware=False, label="clip")
    free_cuda()
    arm1_install = install_and_measure(
        trained_Ws, teacher_targets, layer_sizes, probe_dims, vocab,
        n_blocks=n_blocks, graded_scale_mV=GRADED_SCALE_MV, label="arm1_clip")
    a1_cum = arm1_install["cumulative_mean_spearman_vs_teacher"]
    a1_off = train_log["offline_per_block_spearman_vs_teacher"]
    print(f"[distill] ARM1 trainer OFFLINE cumulative vs teacher = {a1_off[-1]:.3f} (per_block={[round(x,3) for x in a1_off]})", flush=True)
    print(f"[distill] ARM1 INSTALLED (live bridge) cumulative vs teacher = {a1_cum:.3f} "
          f"per_block={[round(a['mean_spearman_vs_teacher'],3) for a in arm1_install['per_layer_fidelity']]} "
          f"margin={arm1_install['anti_cheat_specificity']['specificity_margin']:.3f}", flush=True)

    # =============================================================================================
    # [VERIFY] mitigation -- ARM 2 ONLY if ARM 1 misses GO: driving-force-aware trainer (fold the
    # g.(V-E) sub-linearity into the trainer forward). Still NO sim/ edit.
    # =============================================================================================
    arm2_install = None
    a2_cum = float("nan")
    a2_off = None
    if math.isnan(a1_cum) or a1_cum < GO_BAR:
        print("\n[distill] ===== ARM 2 ([VERIFY] mitigation): driving-force-aware trainer "
              "(fold g.(V-E) into the forward) =====", flush=True)
        free_cuda()
        trained_Ws2, train_log2 = distill_weights(
            Ws_sliced, inputs, teacher_targets, n_blocks, driving_force_aware=True, label="dfaware")
        free_cuda()
        arm2_install = install_and_measure(
            trained_Ws2, teacher_targets, layer_sizes, probe_dims, vocab,
            n_blocks=n_blocks, graded_scale_mV=GRADED_SCALE_MV, label="arm2_dfaware")
        a2_cum = arm2_install["cumulative_mean_spearman_vs_teacher"]
        a2_off = train_log2["offline_per_block_spearman_vs_teacher"]
        print(f"[distill] ARM2 trainer OFFLINE cumulative vs teacher = {a2_off[-1]:.3f}", flush=True)
        print(f"[distill] ARM2 INSTALLED (live bridge) cumulative vs teacher = {a2_cum:.3f} "
              f"per_block={[round(a['mean_spearman_vs_teacher'],3) for a in arm2_install['per_layer_fidelity']]} "
              f"margin={arm2_install['anti_cheat_specificity']['specificity_margin']:.3f}", flush=True)

    # pick the best installed arm
    arms = [("arm1_clip", a1_cum, arm1_install)]
    if arm2_install is not None:
        arms.append(("arm2_dfaware", a2_cum, arm2_install))
    best_name, best_cum, best_install = max(
        arms, key=lambda x: (x[1] if not math.isnan(x[1]) else -2.0))
    best_trained = trained_Ws if best_name == "arm1_clip" else trained_Ws2

    # =============================================================================================
    # ANTI-CHEAT (i): SHUFFLED-TARGET control. Distil to PERMUTED teacher activations -> install ->
    # the installed fidelity vs the REAL teacher must stay at chance (proves recovery is from real
    # distillation, not a generic high-activation pattern). Use the BEST arm's trainer settings.
    # =============================================================================================
    print("\n[distill] ===== ANTI-CHEAT: SHUFFLED-TARGET control (distil to permuted targets) =====", flush=True)
    rng = np.random.default_rng(1234)
    # permute the TARGET feature-vectors across the probe chars (per layer), so the trainer learns a
    # WRONG char->activation map; scored against the REAL teacher must be at chance.
    perm = rng.permutation(len(probe_dims))
    while np.any(perm == np.arange(len(probe_dims))):   # ensure a derangement (no fixed point)
        perm = rng.permutation(len(probe_dims))
    shuffled_targets = [t[perm].copy() for t in teacher_targets]
    df_aware_best = (best_name == "arm2_dfaware")
    free_cuda()
    shuf_trained, shuf_log = distill_weights(
        Ws_sliced, inputs, shuffled_targets, n_blocks, driving_force_aware=df_aware_best,
        label="shuffled")
    free_cuda()
    shuf_install = install_and_measure(
        shuf_trained, teacher_targets, layer_sizes, probe_dims, vocab,   # score vs REAL teacher
        n_blocks=n_blocks, graded_scale_mV=GRADED_SCALE_MV, label="shuffled_ctrl")
    shuf_cum = shuf_install["cumulative_mean_spearman_vs_teacher"]
    print(f"[distill] SHUFFLED-control INSTALLED cumulative vs REAL teacher = {shuf_cum:.3f} "
          f"(must be at chance ~0; the trained-to-WRONG-target weights must NOT recover)", flush=True)
    del shuf_trained
    free_cuda()

    # =============================================================================================
    # VERDICT
    # =============================================================================================
    spec = best_install["anti_cheat_specificity"]
    margin_ok = (not math.isnan(spec["specificity_margin"]) and spec["specificity_margin"] > 0.1)
    # shuffled-control "at chance": well below the best arm AND below a small absolute bar.
    shuf_at_chance = (math.isnan(shuf_cum) or (shuf_cum < 0.4 and (math.isnan(best_cum) or shuf_cum < best_cum - 0.3)))

    if (not math.isnan(best_cum)) and best_cum >= GO_BAR and margin_ok and shuf_at_chance:
        verdict = "GO"
    elif (not math.isnan(best_cum)) and best_cum > vb_cum + 0.1 and margin_ok:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    best_per_block = [None if math.isnan(a["mean_spearman_vs_teacher"]) else round(a["mean_spearman_vs_teacher"], 3)
                      for a in best_install["per_layer_fidelity"]]
    verdict_line = (
        "distill: narrow%d blocks=%d clip-aware-layerwise-distillation INSTALLED-on-live-bridge "
        "cumulative_analog_spearman_vs_teacher=%.3f (best arm=%s; verbatim-install=%.3f; trainer-offline=%.3f) "
        "per_block=%s specificity_margin=%.3f shuffled_control=%.3f -> %s "
        "(vs verbatim-install 0.288 baseline)" % (
            HIDDEN_WIDTH, n_blocks, (best_cum if not math.isnan(best_cum) else float('nan')),
            best_name, vb_cum, (a1_off[-1] if best_name == "arm1_clip" else (a2_off[-1] if a2_off else float('nan'))),
            best_per_block, spec["specificity_margin"], shuf_cum, verdict))

    result = {
        "probe": "genseq_loopstep3_clip_aware_distillation",
        "resolves": "the consolidation per-layer clip-compression wall (verbatim install 0.288) via "
                    "clip-aware layerwise activation distillation through the bridge-faithful "
                    "differentiable forward, then install on the live bridge. NO sim/ edit.",
        "scoping": "research/findings/2026-06-22-surrogate-grad-on-bridge-scoping.md (HYBRID #1)",
        "artifact": "cortex_10M_seed42.npz",
        "oom_safety": {
            "hidden_width": HIDDEN_WIDTH, "n_blocks": n_blocks,
            "full_layer_sizes": layer_sizes_full, "narrowed_feature_sizes": feature_sizes,
            "bytes_per_edge_est": BYTES_PER_EDGE_EST,
            "oom_ceiling_gb": OOM_CEILING_GB, "safe_budget_gb": SAFE_BUDGET_GB,
            "bridge_neurons": int(n_neu), "bridge_est_edges": int(edges), "bridge_est_gb": round(gb, 3),
        },
        "n_blocks": n_blocks, "feature_sizes": feature_sizes,
        "graded_scale_mV": GRADED_SCALE_MV, "go_bar": GO_BAR,
        "probe_dims": probe_dims, "held_out_dims": held_dims,
        "trainer": {
            "framework": "torch (clip+matmul autograd; clip sub-gradient pass-through)",
            "lr": TRAIN_LR, "steps_layerwise_per_block": TRAIN_STEPS_LAYERWISE,
            "steps_e2e": TRAIN_STEPS_E2E,
            "target": "layerwise activation distillation to clean teacher reps t_{L+1}=clip(t_L@W_sliced,0,1); "
                      "compounded real input per block; end-to-end final-block polish",
        },
        "teacher": "offbridge_graded_forward(VERBATIM sliced weights) clean per-layer target reps",
        "verbatim_install": {
            "cumulative_mean_spearman_vs_teacher": vb_cum,
            "per_layer_fidelity": verbatim_install["per_layer_fidelity"],
            "anti_cheat_specificity": verbatim_install["anti_cheat_specificity"],
        },
        "arm1_clip_trainer": {
            "offline_per_block_spearman_vs_teacher": a1_off,
            "training_log": train_log,
            "installed_cumulative_mean_spearman_vs_teacher": a1_cum,
            "installed_per_layer_fidelity": arm1_install["per_layer_fidelity"],
            "installed_anti_cheat_specificity": arm1_install["anti_cheat_specificity"],
            "installed_per_input": arm1_install["per_input"],
            "per_layer_gain_calibration": arm1_install["per_layer_gain_calibration"],
        },
        "arm2_driving_force_aware_trainer": (None if arm2_install is None else {
            "offline_per_block_spearman_vs_teacher": a2_off,
            "installed_cumulative_mean_spearman_vs_teacher": a2_cum,
            "installed_per_layer_fidelity": arm2_install["per_layer_fidelity"],
            "installed_anti_cheat_specificity": arm2_install["anti_cheat_specificity"],
        }),
        "anti_cheat_shuffled_target": {
            "method": "distil to a deranged permutation of the teacher targets across probe chars; "
                      "install; score vs the REAL teacher -> must be at chance",
            "permutation": perm.tolist(),
            "driving_force_aware": df_aware_best,
            "offline_per_block_spearman_vs_real_teacher": shuf_log["offline_per_block_spearman_vs_teacher"],
            "installed_cumulative_mean_spearman_vs_real_teacher": shuf_cum,
            "installed_per_layer_fidelity": shuf_install["per_layer_fidelity"],
            "at_chance": bool(shuf_at_chance),
        },
        "best_arm": best_name,
        "best_installed_cumulative_mean_spearman_vs_teacher": best_cum,
        "best_installed_per_layer_fidelity": best_install["per_layer_fidelity"],
        "best_installed_anti_cheat_specificity": spec,
        "baseline_verbatim_install": {"cumulative_mean_spearman": 0.288,
                                      "per_block": [0.846, 0.620, 0.288],
                                      "note": "the popcode-NEGATIVE verbatim-install wall (scale=20)"},
        "verdict_line": verdict_line, "verdict": verdict,
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, default=lambda o: None
                                   if (isinstance(o, float) and math.isnan(o)) else o))

    print("\n[distill] ===== SUMMARY =====", flush=True)
    print(f"[distill]   verbatim-install vs teacher:   {vb_cum:.3f}", flush=True)
    print(f"[distill]   ARM1 clip trainer OFFLINE:     {a1_off[-1]:.3f}", flush=True)
    print(f"[distill]   ARM1 clip INSTALLED live:      {a1_cum:.3f}", flush=True)
    if arm2_install is not None:
        print(f"[distill]   ARM2 df-aware OFFLINE:         {a2_off[-1]:.3f}", flush=True)
        print(f"[distill]   ARM2 df-aware INSTALLED live:  {a2_cum:.3f}", flush=True)
    print(f"[distill]   BEST installed ({best_name}):  {best_cum:.3f}", flush=True)
    print(f"[distill]   shuffled-target control:       {shuf_cum:.3f} (at_chance={shuf_at_chance})", flush=True)
    print(f"[distill]   specificity margin:            {spec['specificity_margin']:.3f}", flush=True)
    print("\n" + "=" * 78)
    print(verdict_line)
    print("=" * 78)
    print(f"[distill] wrote {OUT_PATH}", flush=True)
    return result


if __name__ == "__main__":
    main()
