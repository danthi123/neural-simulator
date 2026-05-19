"""TDD tests for the net-new shared-rhythm SPEAR conversational runner.

Written FIRST (red before the runner lands). The decisive multi-seed CuPy
run is a later controller-only task; here we only screen that:

  (a) ``run_spear_conversational(seeds=[42], tiny_synth=True)`` runs
      end-to-end, returns a dict with ``rungs`` + ``verdict`` whose
      ``gate`` is one of the four frozen states, and NEVER raises;
  (b) every rung carries EXACTLY the five required keys with correct
      types/ranges so the frozen verdict does NOT VOID for a structural
      reason (it may legitimately FAIL on toy numbers -- fine);
  (c) no shipped module text imports torch.autograd / ``.backward``;
  (d) for a (seed,N) cell ``full`` and ``rhythm_removed`` consume the
      SAME seed and differ ONLY by the shared-rhythm controller being
      enabled vs disabled (the controller is the sole difference, a
      single flag/param threaded identically);
  (e) the decode path uses the validated neural readout (no tag-string
      parse; the runner source contains no ``.split("_")`` on tag
      names) and the moat is fed the raw firing-rate quantity.

tiny_synth shrinks pools/episodes/phase-block lengths hard so this is a
fast logic-screen smoke (toy numbers are NOT a result).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import research.runners.spear_conversational_runner as scr
from research.runners.spear_conversational_core import (
    REQUIRED_KEYS,
    spear_conversational_verdict,
)

_VALID_GATES = {
    "VOID",
    "FAIL",
    "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
    "PASS",
}


def test_runner_module_exposes_entry_point():
    assert hasattr(scr, "run_spear_conversational")
    assert callable(scr.run_spear_conversational)
    assert hasattr(scr, "main")


def test_tiny_synth_runs_end_to_end_and_is_not_structurally_void():
    """(a)+(b): a tiny-synth multi-seed run returns a well-formed dict
    the frozen verdict accepts (one of the four states, never raises,
    NOT VOID for a structural reason). We run the verdict's minimum
    seed count (3) so structural well-formedness can be screened; the
    toy numbers themselves are explicitly NOT a result."""
    result = scr.run_spear_conversational(
        seeds=[42, 43, 44], tiny_synth=True
    )

    assert isinstance(result, dict)
    assert "rungs" in result and isinstance(result["rungs"], list)
    assert len(result["rungs"]) >= 1
    assert "verdict" in result and isinstance(result["verdict"], dict)

    gate = result["verdict"]["gate"]
    assert gate in _VALID_GATES

    # Every rung must carry EXACTLY the five required keys with correct
    # types/ranges so the frozen verdict does not VOID structurally.
    for r in result["rungs"]:
        assert isinstance(r, dict)
        for k in REQUIRED_KEYS:
            assert k in r, "rung missing required key %s" % k
        assert isinstance(r["N"], int) and not isinstance(r["N"], bool)
        assert isinstance(r["n_seeds"], int) and not isinstance(
            r["n_seeds"], bool
        )
        for ak in (
            "full_acc",
            "rhythm_removed_acc",
            "abstain_correct_rhythm_removed",
        ):
            v = r[ak]
            assert isinstance(v, float) and not isinstance(v, bool)
            assert 0.0 <= v <= 1.0

    # Recompute the verdict from the raw rungs -- it must not VOID for a
    # structural/instrument reason (a legitimate FAIL on toy numbers is
    # acceptable; VOID would mean a malformed rung shape).
    recomputed = spear_conversational_verdict(result["rungs"])
    assert recomputed["gate"] in _VALID_GATES
    assert recomputed["gate"] != "VOID", (
        "tiny-synth rungs must be structurally well-formed; VOID here "
        "means a malformed rung shape, got reason=%r"
        % recomputed.get("reason")
    )


def test_tiny_synth_does_not_raise_with_loads_and_out(tmp_path):
    """(a): explicit small ladder + out-path also runs clean and writes
    a JSON the verdict accepts."""
    out = tmp_path / "spear_smoke.json"
    result = scr.run_spear_conversational(
        seeds=[42], loads=(2,), tiny_synth=True, out_path=str(out)
    )
    assert out.exists()
    assert result["verdict"]["gate"] in _VALID_GATES
    assert result.get("tiny_synth") is True
    # the smoke must explicitly disclaim its toy numbers
    assert "note" in result and "NOT a result" in result["note"]


def test_no_autograd_on_shipped_path():
    """(c): no shipped module text imports torch.autograd / .backward."""
    src = Path(scr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "torch.autograd" not in src
    assert ".backward(" not in src
    assert "import torch" not in src


def test_decode_is_neural_not_tag_string_parse():
    """(e): the runner source contains NO tag-string parse -- no
    .split("_") and no .split('_') anywhere (the answer is decoded from
    the validated neural readout, never out of an opaque tag name)."""
    src = Path(scr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert '.split("_")' not in src
    assert ".split('_')" not in src
    # The validated neural readout helpers must be the decode path.
    assert "lang_output_pattern_during_stim" in src
    # The moat must be fed the raw firing-rate ranked confidence.
    assert "abstention_gate" in src or "_abstain_gate" in src


def test_rhythm_removed_is_full_minus_only_the_shared_rhythm():
    """(d): for a (seed,N) cell, full and rhythm_removed differ ONLY by
    the shared-rhythm controller being enabled vs disabled.

    Structural assertion (no heavy run): the per-arm cell function takes
    a single boolean controller flag, and the cell driver invokes it
    once with the flag True (full) and once with the flag False
    (rhythm_removed) using the SAME seed and SAME facts/draws. We assert
    the flag is the sole differing argument by inspecting the source of
    the cell driver.
    """
    # The arm function must accept a single explicit shared-rhythm flag.
    assert hasattr(scr, "_run_arm")
    arm_sig = inspect.signature(scr._run_arm)
    assert "use_rhythm" in arm_sig.parameters, (
        "the arm runner must thread an explicit `use_rhythm` controller "
        "flag so full vs rhythm_removed differ ONLY by it"
    )

    cell_src = inspect.getsource(scr._cell)
    # full arm: use_rhythm True; rhythm_removed arm: use_rhythm False.
    assert "use_rhythm=True" in cell_src
    assert "use_rhythm=False" in cell_src
    # Both arms must be built from the SAME seed (no per-arm seed
    # perturbation) -- the cell takes one `seed` and passes it to both.
    cell_sig = inspect.signature(scr._cell)
    assert "seed" in cell_sig.parameters
    # No second RNG seed / no seed offset between the two arms.
    assert "seed + 1" not in cell_src and "seed+1" not in cell_src

    # The verdict-relevant control key must be derived from the
    # rhythm-disabled arm (the Stage-1-static reduction).
    agg_src = inspect.getsource(scr._aggregate)
    assert "rhythm_removed_acc" in agg_src
    assert "abstain_correct_rhythm_removed" in agg_src


def test_controller_has_theta_period_and_ach_phase_gate():
    """The net-new piece is a theta-phase clock + ACh phase gate. Assert
    the controller derives a theta period in STEPS from the bridge dt
    and gates plasticity via the reused neuromodulator set_concentration
    on encode vs retrieve phases (a timing controller, not a new rule).
    """
    src = Path(scr.__file__).read_text(encoding="utf-8", errors="ignore")
    # theta ~125 ms period derived from dt (not a hardcoded step count).
    assert "125" in src  # ~125 ms theta cycle
    assert "set_concentration" in src  # reused ACh phase gate
    assert "step_simulation" in src or "_run_one_simulation_step" in src
    # gamma sub-cycle indexes the dlpfc compositional slot.
    assert "gamma" in src.lower()
    assert "dlpfc" in src.lower()


# =====================================================================
#  Faithfulness pins (FIX A + FIX B + FIX C).
#  The adversarial review confirmed the prior `plasticity_window_gate`
#  (scope=all) target was FUNCTIONALLY INERT in this runner because its
#  ONLY consumer (sim/bridge.py:5577-5579) sits inside the C2 reward-mod
#  block, which is gated by `update_path_active` (bridge.py:5512-5513)
#  and never enters when `current_reward_signal=0` (this runner never
#  drives reward). The pins below enforce that the ACh modulator is now
#  routed through targets the bridge CONSUMES during encode/retrieve --
#  `plasticity_rate` (scope=all) AND `synaptic_gain` (scope=all). The
#  synaptic_gain path is consumed EVERY step (bridge.py:4877-4879 and
#  4890-4897) regardless of reward, so the gate genuinely modulates
#  forward propagation -- biology-faithful to Hasselmo SPEAR (high ACh
#  suppresses recurrent feedback during encode).
# =====================================================================


def _registered_ach_targets():
    """Build a tiny-synth bridge and return the list of ModulatorTarget
    dataclasses registered on the acetylcholine_tan modulator. Routed
    via the public substrate-build helper so we exercise the SAME
    NeuromodulatorConfig the runner actually ships.
    """
    bridge, _dims = scr._build_substrate(seed=42, tiny_synth=True)
    mgr = bridge.neuromodulator_manager
    assert mgr is not None, (
        "neuromodulator subsystem must be enabled on the runner's bridge"
    )
    # Reach into the manager's configs via its public name lookup.
    cfg = mgr._config_by_name("acetylcholine_tan")
    return list(cfg.targets)


def test_ach_targets_include_active_consumers_not_just_inert_window_gate():
    """FIX A: ACh must target consumers that are active during the
    encode/retrieve phases this runner exercises, NOT solely the C2-gated
    plasticity_window_gate (which is inert when reward=0).

    Concretely: `plasticity_rate` (scope=all) is consumed in the C2
    block AND its scope=all path is sound; `synaptic_gain` (scope=all)
    is consumed every step in the synaptic-conductance section
    (bridge.py:4877-4879 and 4890-4897) independent of reward. At least
    one of these MUST be in the registered target list.
    """
    targets = _registered_ach_targets()
    target_types = {t.target_type for t in targets}

    assert "plasticity_rate" in target_types or "synaptic_gain" in target_types, (
        "ACh modulator must target at least one of plasticity_rate / "
        "synaptic_gain (the consumers actually active during this "
        "runner's encode/retrieve phases). Found only: %r" % target_types
    )


def test_ach_targets_synaptic_gain_for_phase_dynamics():
    """FIX B: the encode/retrieve phase must modulate FORWARD DYNAMICS,
    not just (silent) plasticity. synaptic_gain (scope=all) is consumed
    every simulation step in sim/bridge.py:4877-4879 (with STP) and
    4890-4897 (no STP), so a non-trivial sensitivity here makes ACh
    genuinely change the bridge's effective synaptic strength matrix
    between encode and retrieve. This is the Hasselmo biology (high ACh
    suppresses recurrent feedback during encode; low ACh permits CA3
    pattern completion during retrieve).
    """
    targets = _registered_ach_targets()
    sg_targets = [t for t in targets if t.target_type == "synaptic_gain"]
    assert sg_targets, (
        "ACh modulator must register at least one synaptic_gain target "
        "so the encode/retrieve phase modulates forward dynamics"
    )
    # Sensitivity must be non-trivial in at least one scope=all
    # synaptic_gain target so the gate has measurable effect.
    sg_all = [t for t in sg_targets if t.scope == "all"]
    assert sg_all, (
        "at least one synaptic_gain target must have scope='all' so it "
        "is consumed by compute_synaptic_gain_multiplier()"
    )
    assert any(abs(t.sensitivity) > 1e-6 for t in sg_all), (
        "synaptic_gain sensitivity must be non-zero so the gate moves "
        "effective_synaptic_strength between encode and retrieve"
    )


def test_ach_structural_effect_on_bridge_state_pin():
    """FIX C (positive structural-effect adversarial pin): mirror the
    reviewer's 50-step constant-input probe. Build the SAME substrate,
    drive the SAME external input, step the bridge with ACh held at the
    encode setpoint vs the retrieve setpoint, and assert the resulting
    membrane state is NOT byte-identical between the two phases. This
    locks in the invariant the inert plasticity_window_gate-only target
    list violated.
    """
    from sim.backend import get_backend, to_host
    cp, _backend = get_backend()

    # Two independent substrates from the SAME seed: deterministic.
    bridge_enc, _ = scr._build_substrate(seed=42, tiny_synth=True)
    bridge_ret, _ = scr._build_substrate(seed=42, tiny_synth=True)

    # Hold ACh at the encode vs retrieve setpoints the controller uses.
    scr._set_ach(bridge_enc, scr._ACH_ENCODE_LOW)
    scr._set_ach(bridge_ret, scr._ACH_RETRIEVE_HIGH)

    # Identical constant external input on both -- a small drive on the
    # first language_input lane so propagation actually exercises the
    # synaptic_gain-modulated effective_synaptic_strength path.
    drive_pA = cp.float32(200.0)
    n_drive = min(8, bridge_enc.cp_external_input_current.shape[0])
    bridge_enc.cp_external_input_current[:n_drive] = drive_pA
    bridge_ret.cp_external_input_current[:n_drive] = drive_pA

    # Step both bridges for the SAME number of steps.
    for _ in range(50):
        # Re-pin ACh at the start of every step (the manager's exponential
        # decay toward baseline would otherwise pull them together over
        # 50 * dt_ms = 25 ms vs decay_tau_ms = 500 ms; pinning every step
        # is what the encode/retrieve phase helpers also do via the
        # set-then-step pattern).
        scr._set_ach(bridge_enc, scr._ACH_ENCODE_LOW)
        scr._set_ach(bridge_ret, scr._ACH_RETRIEVE_HIGH)
        bridge_enc._run_one_simulation_step()
        bridge_ret._run_one_simulation_step()

    # Compare a deterministic summary of membrane potential on both --
    # the sum of the first 10 entries -- and require they differ by
    # more than a tiny epsilon. If the gate were inert (the defect),
    # these would be byte-identical.
    n = min(10, bridge_enc.cp_membrane_potential_v.shape[0])
    v_enc = to_host(bridge_enc.cp_membrane_potential_v[:n])
    v_ret = to_host(bridge_ret.cp_membrane_potential_v[:n])

    diff = float(abs(float(v_enc.sum()) - float(v_ret.sum())))
    # Require a *measurable* difference. 1e-3 mV is well below biological
    # noise but well above floating-point churn -- the inert gate
    # produced byte-identical state (diff = 0.0 exactly).
    assert diff > 1e-3, (
        "ACh gate must affect bridge state across encode/retrieve "
        "setpoints. Got byte-identical membrane state (sum_diff=%.6g mV), "
        "which is the inert-gate failure mode the adversarial review "
        "flagged. v_enc.sum=%r v_ret.sum=%r"
        % (diff, float(v_enc.sum()), float(v_ret.sum()))
    )


def test_full_vs_rhythm_removed_diverge_in_underlying_readout_smoke():
    """FIX C companion: an end-to-end smoke pin. For a tiny-synth
    (seed=42, N=2) cell, the `full` and `rhythm_removed` arms differ
    only by the use_rhythm flag, but that flag must actually change
    something observable in the UNDERLYING NEURAL READOUT (the raw
    lang_output firing-rate ranked confidences fed to the moat). The
    moat itself may legitimately collapse both arms to the same
    (degenerate) acc/abstain pair at tiny-synth scale -- the toy
    numbers are explicitly NOT a result -- but the upstream neural
    state IT IS FED must reflect the gate's effect. The inert-gate
    failure mode (the defect this fix closes) would give byte-identical
    ranked confidences across the two arms.
    """
    # Two fresh substrates from the SAME seed (deterministic).
    b_full, dims = scr._build_substrate(seed=42, tiny_synth=True)
    b_rmrm, _ = scr._build_substrate(seed=42, tiny_synth=True)

    # Identically encode one fact on both bridges using the SAME
    # encode helper, then read it back via the SAME retrieve helper
    # -- the only difference being use_rhythm.
    noun, adj = ("apple", "big")
    tag = "fact_0"
    enc_steps = 8
    recall_steps = 20
    scr._theta_encode_phase(
        b_full, (noun, adj), tag, dims, enc_steps,
        gamma_idx=0, n_facts=1, use_rhythm=True,
    )
    scr._theta_encode_phase(
        b_rmrm, (noun, adj), tag, dims, enc_steps,
        gamma_idx=0, n_facts=1, use_rhythm=False,
    )
    _ans_full, ranked_full = scr._theta_retrieve_phase(
        b_full, noun, tag, dims, have_remote=True,
        recall_steps=recall_steps, use_rhythm=True,
    )
    _ans_rmrm, ranked_rmrm = scr._theta_retrieve_phase(
        b_rmrm, noun, tag, dims, have_remote=True,
        recall_steps=recall_steps, use_rhythm=False,
    )

    # ranked is List[(word, confidence, source)]. Pull the raw confidence
    # vectors (sorted by word for a stable comparison) and require
    # they are NOT pointwise identical.
    def _conf_map(ranked):
        return {w: float(c) for (w, c, _src) in ranked}

    cf = _conf_map(ranked_full)
    cr = _conf_map(ranked_rmrm)
    # If the gate is inert, the two arms read the SAME firing rates
    # off the SAME bridge dynamics and the maps are identical.
    keys_match = (set(cf.keys()) == set(cr.keys()))
    pointwise_diff = 0.0
    if keys_match:
        for k in cf:
            pointwise_diff += abs(cf[k] - cr.get(k, 0.0))
    else:
        # Disjoint keys is itself a divergence (some words decoded by
        # one arm and not the other).
        pointwise_diff = 1.0
    assert pointwise_diff > 1e-6 or not keys_match, (
        "full vs rhythm_removed produced byte-identical ranked "
        "lang_output confidences -- the gate is not changing the "
        "neural readout. ranked_full=%r ranked_rmrm=%r"
        % (ranked_full, ranked_rmrm)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
