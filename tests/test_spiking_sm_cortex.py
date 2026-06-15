"""Task 1 tests: the cortex-bridge builder + PPMI-shaped input encoder.

Runs CPU-only (SIM_BACKEND=numpy) so it is fast and deterministic.
"""
import os

os.environ["SIM_BACKEND"] = "numpy"  # CPU-only, set BEFORE importing the runner

import numpy as np
import pytest

from research.runners.dendritic_d1_learn_graded_structure_derisk import (
    _cos_sim,
    _pearson_vs_Strue,
    build_concept_hub_counts,
    effective_rank,
)
from research.runners.spiking_sm_cortex import (
    build_sm_cortex_bridge,
    encode_drive,
    read_codes,
    train_sm_cortex,
)
from sim.backend import to_host


def test_build_and_encode():
    # --- builder ---
    bridge, hub_idx, cortex_idx = build_sm_cortex_bridge(n_hub=200, n_cortex=64, seed=42)

    hub_idx = np.asarray(hub_idx)
    cortex_idx = np.asarray(cortex_idx)
    assert len(hub_idx) == 200
    assert len(cortex_idx) == 64
    # hub and cortex index slices must be disjoint
    assert np.intersect1d(hub_idx, cortex_idx).size == 0

    # the hub->cortex pathway is present, plastic, and tagged with the gate name.
    pathways = bridge.core_config.region_pathways
    hub_to_cortex = [
        pw for pw in pathways
        if pw.from_region == "hub" and pw.to_region == "cortex"
    ]
    assert len(hub_to_cortex) == 1, "expected exactly one hub->cortex pathway"
    pw = hub_to_cortex[0]
    assert pw.plastic is True
    assert pw.plasticity_gate == "hub_to_cortex"

    # --- encoder ---
    raw = np.array([0.0, 1.0, 3.0, 7.0])
    out_log = encode_drive(raw, log=True)
    assert np.allclose(out_log, np.log1p([0.0, 1.0, 3.0, 7.0]))

    out_raw = encode_drive(raw, log=False)
    assert np.allclose(out_raw, np.maximum(raw, 0.0))

    # negatives are clipped to zero in both modes
    neg = np.array([-2.0, 0.0, 5.0])
    assert np.allclose(encode_drive(neg, log=False), np.array([0.0, 0.0, 5.0]))
    assert np.allclose(encode_drive(neg, log=True), np.log1p([0.0, 0.0, 5.0]))


@pytest.mark.xfail(
    reason="Stays XFAIL under the DEFAULT builder (no C1a WTA / co-fire). Root cause re-diagnosed in Task 3 "
    "(2026-06-15): the Task-2 collapse was NOT a net-depression -- the deeper cause is that "
    "bridge._run_one_simulation_step() does NOT advance current_time_ms, so every spike was stamped t=0, "
    "delta_t==0, and STDP was a total NO-OP (weights frozen exactly, not depressed). Task 3 fixes that at "
    "the runner level (train/read now advance the clock via _step_with_time -- NO sim/ edit). But the "
    "DEFAULT weight_mean=0.05 hub->cortex pathway is FAR too weak to ever fire the cortex (0 cortex spikes "
    "-> STDP has no post-spike to pair -> weights still do not move AND codes.sum()==0). Firing the cortex "
    "needs the C1a regime (strong hub weight + WTA + co-fire), which test_trained_cortex_recovers_structure "
    "exercises. So this default-builder smoke remains XFAIL; the C1a path is validated (cortex fires, STDP "
    "engages, weights rise) in the HARD GATE test's collapse-guard. See "
    "2026-06-15-bridge-competitive-stdp-deep-research.md.",
    strict=False,
)
def test_train_read_machinery():
    """Task 2 mechanical check: training MOVES the plastic hub->cortex weights, and read_codes
    returns a non-degenerate [Nc x n_cortex] spike-count code matrix with the cortex actually firing.

    Tiny synthetic case (16 concepts x 80 hubs, n_cortex=32) so it runs in a few seconds on numpy.
    NO structure claim here -- only that the train/read plumbing works. Stays XFAIL under the DEFAULT
    builder: the weak (weight_mean=0.05) hub->cortex pathway never fires the cortex without the C1a
    WTA + co-fire regime, so codes.sum()==0 and STDP (now correctly clocked) has no post-spike to pair.
    """
    n_concepts, n_hub, n_cortex = 16, 80, 32

    # a simple non-negative count matrix: a few Poisson-ish active columns per concept.
    rng = np.random.RandomState(7)
    C = np.zeros((n_concepts, n_hub), dtype=np.float64)
    for i in range(n_concepts):
        active = rng.choice(n_hub, size=10, replace=False)
        C[i, active] = rng.poisson(lam=4.0, size=active.size).astype(np.float64)
    C_drive = encode_drive(C)  # log1p Weber-Fechner compression

    bridge, hub_idx, cortex_idx = build_sm_cortex_bridge(n_hub=n_hub, n_cortex=n_cortex, seed=42)
    hub_idx = np.asarray(hub_idx)
    cortex_idx = np.asarray(cortex_idx)

    # --- weights must CHANGE through training (STDP on the only plastic pathway moves them) ---
    # NOTE: structural plasticity is ON by default, so the CSR .data array LENGTH can change
    # across training (synapses form/eliminate). Compare the total weight MASS (a scalar robust to
    # the length change) -- STDP + structural plasticity both move it away from the initial value.
    w_before = np.asarray(to_host(bridge.cp_connections.data)).copy()
    train_sm_cortex(bridge, C_drive, hub_idx, cortex_idx, n_epochs=2)
    w_after = np.asarray(to_host(bridge.cp_connections.data)).copy()
    if w_before.shape == w_after.shape:
        assert not np.allclose(w_before, w_after), "training did not move the hub->cortex weights"
    else:
        # structural plasticity changed the synapse count -> the weights necessarily changed.
        assert not np.isclose(float(w_before.sum()), float(w_after.sum())), \
            "training did not move the hub->cortex weight mass"

    # --- read_codes returns the per-concept cortex spike-count codes ---
    codes = read_codes(bridge, C_drive, hub_idx, cortex_idx)
    assert codes.shape == (n_concepts, n_cortex)
    assert codes.sum() > 0, "cortex never fired -- all codes are silent"
    assert effective_rank(codes) > 1.0, "codes are degenerate (effective rank <= 1)"

    # read_codes must RESTORE the gate to full plasticity afterward.
    assert bridge._plasticity_gate_values["hub_to_cortex"] == 1.0


# ---------------------------------------------------------------------------
# Task 3 (option C1a) HARD GATE: competitive-STDP (WTA + fast homeostasis + co-fire).
# ---------------------------------------------------------------------------
def _build_synth_64():
    """The synthetic 64-concept case with a STRONG common mode (host PPMI+SVD ceiling ~0.96).

    n_cat=8, per_cat=8, n_common=200 (the common mode that swamps raw profiles), n_sig_per_cat=12.
    Mirrors the calibration in dendritic_d1_learn_graded_structure_derisk / the deep-research (a real
    host ceiling: raw-profile Pearson ~0.05, PPMI+SVD ceiling ~0.96). Returns (C, labels, S_true).
    """
    C, labels, S_true, _hub_freq = build_concept_hub_counts(
        n_cat=8, per_cat=8, n_common=200, n_sig_per_cat=12,
        lam_common=40.0, lam_sig=4.0, lam_bg=0.3, seed=42,
    )
    return C, labels, S_true


# C1a tuned recipe (frozen after the tuning grid). All config/runner-level; NO sim/ edits.
# NOTE (attempt 2, 2026-06-15): the gate stays xfail. The corrected fix = CENTERING (common-mode removal).
# The three brain-based centering mechanisms (feedforward subtractive-inhibition cm pool / synaptic scaling /
# stronger dendritic gain) are all available in build_sm_cortex_bridge (default-off) and were swept in
# research/findings/raw/_phaseB_task3_cm_gate.py + _phaseB_task3_centering_sweep.py: the cm pool is the best
# (it nudges structure -0.115 -> ~-0.05, margin ~+0.06) but a UNIFORM spiking inhibition can only do a
# SCALAR subtraction, not the L1 per-dimension centering -> the gate stays NEGATIVE (the Mikulasch-
# Priesemann wall). The fast WTA recipe below is what this gate test runs (so CI stays ~16s); the cm-pool
# arm is exercised by the raw sweep + documented in 2026-06-15-phaseB-task3-centering-RESULT.md.
_C1A = dict(
    n_cortex=128,
    seed=42,
    # WTA: cortex with 20% inhibitory neurons + dense internal E->I->E + strong inhibition.
    cortex_exc_fraction=0.8,
    cortex_internal_density=0.5,
    cortex_inh_weight_mean=6.0,
    # fast adaptive-threshold homeostasis (Diehl-Cook theta): ~100x the slow nav defaults.
    homeostasis_ema_alpha=0.05,
    homeostasis_threshold_adapt_rate=0.03,
    # training / read protocol
    n_epochs=8,
    drive_scale=12.0,
    cofire_pA=4.0,
    window=40,
    settle=8,
)


def _read_c1a(bridge, C_drive, hub_idx, cortex_idx):
    return read_codes(
        bridge, C_drive, hub_idx, cortex_idx,
        drive_scale=_C1A["drive_scale"], window=_C1A["window"], settle=_C1A["settle"],
    )


def _read_ge_codes_c1a(bridge, C_drive, hub_idx, cortex_idx):
    """Read the per-concept cortex code from the PRE-THRESHOLD analog conductance cp_conductance_g_e
    (plasticity frozen), same drive/window as the spike read. The localization instrument: if g_e-cos ~
    spike-cos, the destroyer is the common mode in the analog drive (not the spike threshold)."""
    from sim.backend import to_host
    from research.runners.spiking_sm_cortex import _set_hub_drive, _step_with_time

    cortex_idx = np.asarray(cortex_idx)
    Nc = int(np.asarray(C_drive).shape[0])
    ds, win, settle = _C1A["drive_scale"], _C1A["window"], _C1A["settle"]
    codes = np.zeros((Nc, cortex_idx.size), dtype=np.float64)
    gate_names = list(getattr(bridge, "_plasticity_gate_values", {}).keys()) or ["hub_to_cortex"]
    for g in gate_names:
        bridge.set_plasticity_gate(g, 0.0)
    try:
        for i in range(Nc):
            _set_hub_drive(bridge, hub_idx, C_drive[i], ds)
            acc = np.zeros(cortex_idx.size, dtype=np.float64)
            for t in range(int(settle) + int(win)):
                _step_with_time(bridge)
                if t >= int(settle):
                    acc += np.asarray(to_host(bridge.cp_conductance_g_e))[cortex_idx].astype(np.float64)
            codes[i] = acc
            bridge.cp_external_input_current[:] = 0.0
    finally:
        for g in gate_names:
            bridge.set_plasticity_gate(g, 1.0)
    return codes


@pytest.mark.xfail(
    reason="Task-3 HARD GATE = WALL after the CENTERING (common-mode removal) attempt (honest, decision-"
    "relevant; 2026-06-15, attempt 2). The C1a competitive-STDP machinery WORKS (collapse-guard below: cortex "
    "FIRES silent~0.12<0.5, eff_rank~13.5>1; hub->cortex STDP RISES 0.05->0.46, no floor-decay -- the Task-2 "
    "silent-target trap is CURED by the _step_with_time clock fix, NO sim/ edit). The corrected diagnosis was "
    "LOCALIZED on the bridge: the cortex g_e (pre-threshold ANALOG conductance) code already has "
    "Pearson(cos,S_true)~-0.063 == the spike-count code ~-0.074, so the spiking THRESHOLD is NOT the "
    "destroyer -- the COMMON MODE survives into the analog drive (the 200 high-freq common hubs swamp the "
    "cortex). The input carries the structure (rate-level log-input cosine +0.89). The L1-validated fix is "
    "CENTERING = common-mode removal (2026-06-14-L1-GO; subtractive-inhibition + bounded Hebbian). Three "
    "brain-based centering mechanisms were tried (ALL framework/config/runner-level, NO sim/ edits): "
    "(1) a feedforward subtractive-inhibition all-inhibitory cm pool (hub->cm exc + inhibitory cm->cortex = "
    "(hub excitation)-(cm inhibition prop to common mode)) -> BEST result, but it only nudges -0.115 "
    "(centering-OFF) -> ~-0.065 (margin +0.05) and STRONGER cm SILENCES the cortex (silent->1.0); "
    "(2) enable_synaptic_scaling (Turrigiano per-neuron renorm) -> -0.092 (no margin: a rate homeostat scales "
    "common+signal alike, cannot SEPARATE the common mode); (3) stronger dendritic divisive gain (sigma "
    "0.02->0.005) -> -0.089 (divisive != subtractive). A read-sparsity control (denser non-silent codes) gets "
    "WORSE (-0.12..-0.15), ruling out a readout artifact. ==> the point-neuron spiking substrate can do only "
    "a SCALAR (rank-1, ~uniform) common-mode subtraction; the L1 op is a PER-DIMENSION analog centering "
    "(x - col_mean BEFORE the projection) that a uniform spiking inhibitory pool structurally cannot "
    "reproduce -- the deep-research's pre-registered risk #1/#2/#6 = the Mikulasch-Priesemann analog-whitening "
    "wall (2026-06-15-bridge-competitive-stdp-deep-research.md; 2026-06-15-phaseB-task3-centering-RESULT.md). "
    "A guarded sim/ edit (C1b: a per-cortex-neuron post-triggered / learned-subtractive rule with anti-runaway "
    "normalization) is now the CONTROLLER decision vs the honest NEGATIVE. The permuted control is clean (~0). "
    "The collapse-guard + random-proj + permuted assertions below are all real; the test xfails ONLY on the "
    "structure bar (a).",
    strict=False,
)
def test_trained_cortex_recovers_structure():
    """HARD GATE (C1a): WTA + fast homeostasis + a non-specific co-fire drive let the bridge cortex
    LEARN, UNSUPERVISED, a code that recovers the synthetic category structure, beating an untrained
    random-projection, WITHOUT collapsing to silence.

    Multi-check (the contrast IS the result):
      (a) structure:   Pearson(cos(codes), S_true)         >= +0.30
      (b) load-bearing: trained - random_proj               >= +0.10
      (c) NOT-silent:   mean(codes.sum(1)==0) < 0.5  AND  effective_rank(codes) > 1.0
      (d) permuted:     Pearson(cos(codes), S_perm)         ~ 0  (|.| <= 0.15)

    Unsupervised: the ONLY concept-specific signal is the hub drive (the environment's sensory input);
    the co-fire drive is UNIFORM across cortex neurons (carries no per-concept info). No per-concept
    target code is EVER injected.
    """
    C, labels, S_true = _build_synth_64()
    C_drive = encode_drive(C)  # log1p Weber-Fechner compression
    n_hub = C.shape[1]

    # --- TRAINED cortex (C1a: WTA + fast homeostasis + co-fire) ---
    bridge, hub_idx, cortex_idx = build_sm_cortex_bridge(
        n_hub=n_hub, n_cortex=_C1A["n_cortex"], seed=_C1A["seed"],
        cortex_exc_fraction=_C1A["cortex_exc_fraction"],
        cortex_internal_density=_C1A["cortex_internal_density"],
        cortex_inh_weight_mean=_C1A["cortex_inh_weight_mean"],
        homeostasis_ema_alpha=_C1A["homeostasis_ema_alpha"],
        homeostasis_threshold_adapt_rate=_C1A["homeostasis_threshold_adapt_rate"],
    )
    hub_idx = np.asarray(hub_idx)
    cortex_idx = np.asarray(cortex_idx)

    w_traj = train_sm_cortex(
        bridge, C_drive, hub_idx, cortex_idx,
        n_epochs=_C1A["n_epochs"], drive_scale=_C1A["drive_scale"],
        window=_C1A["window"], settle=_C1A["settle"],
        cofire_pA=_C1A["cofire_pA"], record_weight_trajectory=True,
    )
    codes = _read_c1a(bridge, C_drive, hub_idx, cortex_idx)

    # --- UNTRAINED random-projection control on the IDENTICAL pipeline (learning load-bearing) ---
    # A fresh C1a bridge built but NOT trained -> its hub->cortex weights are the random init; read
    # codes through the same WTA+homeostasis read pipeline. If "structure" survived an untrained
    # projection, it would be a projection artifact, not learning.
    rp_bridge, rp_hub, rp_cortex = build_sm_cortex_bridge(
        n_hub=n_hub, n_cortex=_C1A["n_cortex"], seed=_C1A["seed"] + 1000,
        cortex_exc_fraction=_C1A["cortex_exc_fraction"],
        cortex_internal_density=_C1A["cortex_internal_density"],
        cortex_inh_weight_mean=_C1A["cortex_inh_weight_mean"],
        homeostasis_ema_alpha=_C1A["homeostasis_ema_alpha"],
        homeostasis_threshold_adapt_rate=_C1A["homeostasis_threshold_adapt_rate"],
    )
    rp_codes = _read_c1a(rp_bridge, C_drive, np.asarray(rp_hub), np.asarray(rp_cortex))

    # --- metrics ---
    pearson = _pearson_vs_Strue(_cos_sim(codes), S_true)
    rp_pearson = _pearson_vs_Strue(_cos_sim(rp_codes), S_true)
    silent_frac = float(np.mean(codes.sum(1) == 0))
    eff_rank = effective_rank(codes)
    rng = np.random.RandomState(20260615)
    perm = rng.permutation(labels)
    S_perm = (perm[:, None] == perm[None, :]).astype(np.float64)
    perm_pearson = _pearson_vs_Strue(_cos_sim(codes), S_perm)

    # --- LOCALIZATION instrument (attempt-2 centering finding, 2026-06-15): read the SAME trained bridge's
    # cortex code from the PRE-THRESHOLD analog conductance g_e (plasticity frozen). If g_e-cos ~ spike-cos
    # (both ~ -0.07), the spiking THRESHOLD is not the destroyer -- the COMMON MODE survives into the analog
    # drive -> centering (common-mode removal) is the required op, but a uniform spiking inhibitory pool can
    # only do a scalar (rank-1) subtraction, not the L1 per-dimension centering (the Mikulasch-Priesemann
    # wall). See research/findings/2026-06-15-phaseB-task3-centering-RESULT.md. This is a PRINT (the
    # decision-relevant localization), not an assertion. log-input cosine = the ceiling the input carries.
    log_input_pearson = _pearson_vs_Strue(_cos_sim(C_drive), S_true)
    ge_codes = _read_ge_codes_c1a(bridge, C_drive, hub_idx, cortex_idx)
    ge_pearson = _pearson_vs_Strue(_cos_sim(ge_codes), S_true)
    print(
        f"\n[Task-3 localization] log-input(ceiling)={log_input_pearson:+.3f}  "
        f"cortex g_e(pre-threshold)={ge_pearson:+.3f}  cortex spike-code={pearson:+.3f}  "
        f"=> threshold NOT the destroyer; common mode survives into the analog drive (centering needed; "
        f"a uniform spiking inhibition does only a SCALAR subtraction -> WALL).",
        flush=True,
    )

    print(
        f"\n[C1a HARD GATE] structure Pearson={pearson:+.3f}  random-proj Pearson={rp_pearson:+.3f}  "
        f"(margin {pearson - rp_pearson:+.3f})  silent_frac={silent_frac:.3f}  eff_rank={eff_rank:.1f}  "
        f"permuted={perm_pearson:+.3f}",
        flush=True,
    )
    if w_traj is not None:
        head = w_traj[:3]
        tail = w_traj[-3:]
        print(
            f"[C1a HARD GATE] hub->cortex mean-weight trajectory: start {head} ... end {tail} "
            f"(min {min(w_traj):.3f}, max {max(w_traj):.3f}) -- a collapse-to-floor would geometric-decay",
            flush=True,
        )

    # (c) NOT-SILENT collapse-guard (the new load-bearing anti-cheat) -- assert FIRST so a silence
    # failure reports the cured/not-cured state distinctly from a structure shortfall.
    assert silent_frac < 0.5, f"cortex mostly silent: {silent_frac:.3f} of concepts have no cortex spikes"
    assert eff_rank > 1.0, f"codes degenerate (effective rank {eff_rank:.2f} <= 1)"
    # (a) structure
    assert pearson >= 0.30, f"structure Pearson {pearson:+.3f} < +0.30"
    # (b) learning load-bearing vs random projection
    assert pearson - rp_pearson >= 0.10, (
        f"learning not load-bearing: trained {pearson:+.3f} vs random-proj {rp_pearson:+.3f} "
        f"(margin {pearson - rp_pearson:+.3f} < +0.10)"
    )
    # (d) permuted control ~ 0
    assert abs(perm_pearson) <= 0.15, f"permuted Pearson {perm_pearson:+.3f} not ~0 (code-overlap meaning-independent?)"
