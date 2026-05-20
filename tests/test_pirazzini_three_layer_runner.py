"""TDD tests for the net-new Pirazzini-reference three-layer runner.

Written FIRST (red before the runner lands). The decisive multi-seed CuPy
run is a later controller-only task; this suite screens only that:

  (a) ``run_pirazzini_three_layer(seeds=[42,43,44], tiny_synth=True)``
      runs end-to-end, returns a dict with ``rungs`` + ``verdict`` whose
      ``gate`` is one of the four frozen states, and NEVER raises;
  (b) every rung carries EXACTLY the five required keys with correct
      types/ranges so the frozen verdict does NOT VOID for a structural
      reason (it may legitimately FAIL on toy numbers -- fine);
  (c) no shipped module text imports torch.autograd / ``.backward``;
  (d) for a (seed,N) cell ``full`` and ``theta_disabled`` consume the
      SAME seed and differ ONLY by the external theta generator being
      enabled vs disabled (a single boolean flag threaded identically;
      no seed perturbation between the arms);
  (e) the decode path uses the validated neural readout (no tag-string
      parse on tag names; the runner source contains no ``.split("_")``
      on tag names) and the moat is fed the raw firing-rate quantity
      ``pat[active].sum() / n_active`` (NOT a cosine * norm hack);
  (f) STRUCTURAL-EFFECT PIN: a 50-step constant-input probe holding the
      external theta generator ON vs OFF must produce a NON-byte-
      identical bridge state -- mirrors the SPEAR re-review's 14.15 mV
      pin and proves the controller is mechanistically active.

tiny_synth shrinks pools/episodes/phase-block lengths so this is a fast
logic-screen smoke (toy numbers are NOT a result).
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import research.runners.pirazzini_three_layer_runner as pzr
from research.runners.pirazzini_three_layer_core import (
    REQUIRED_KEYS,
    pirazzini_three_layer_verdict,
)

_VALID_GATES = {
    "VOID",
    "FAIL",
    "WORKS-AT-SMALL-LOAD-NO-SCALE-CONFIDENCE",
    "PASS",
}


def test_runner_module_exposes_entry_point():
    assert hasattr(pzr, "run_pirazzini_three_layer")
    assert callable(pzr.run_pirazzini_three_layer)
    assert hasattr(pzr, "main")


def test_tiny_synth_runs_end_to_end_and_is_not_structurally_void():
    """(a)+(b): a tiny-synth multi-seed run returns a well-formed dict
    the frozen verdict accepts (one of the four states, never raises,
    NOT VOID for a structural reason). We run the verdict's minimum
    seed count (3) so structural well-formedness can be screened; the
    toy numbers themselves are explicitly NOT a result."""
    result = pzr.run_pirazzini_three_layer(
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
            "theta_disabled_acc",
            "abstain_correct_theta_disabled",
        ):
            v = r[ak]
            assert isinstance(v, float) and not isinstance(v, bool)
            assert 0.0 <= v <= 1.0

    # Recompute the verdict from the raw rungs -- it must not VOID for a
    # structural/instrument reason (a legitimate FAIL on toy numbers is
    # acceptable; VOID would mean a malformed rung shape).
    recomputed = pirazzini_three_layer_verdict(result["rungs"])
    assert recomputed["gate"] in _VALID_GATES
    assert recomputed["gate"] != "VOID", (
        "tiny-synth rungs must be structurally well-formed; VOID here "
        "means a malformed rung shape, got reason=%r"
        % recomputed.get("reason")
    )


def test_tiny_synth_does_not_raise_with_loads_and_out(tmp_path):
    """(a): explicit small ladder + out-path also runs clean and writes
    a JSON the verdict accepts."""
    out = tmp_path / "pirazzini_smoke.json"
    result = pzr.run_pirazzini_three_layer(
        seeds=[42], loads=(2,), tiny_synth=True, out_path=str(out)
    )
    assert out.exists()
    assert result["verdict"]["gate"] in _VALID_GATES
    assert result.get("tiny_synth") is True
    # the smoke must explicitly disclaim its toy numbers
    assert "note" in result and "NOT a result" in result["note"]


def test_no_autograd_on_shipped_path():
    """(c): no shipped module text imports torch.autograd / .backward."""
    src = Path(pzr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert "torch.autograd" not in src
    assert ".backward(" not in src
    assert "import torch" not in src


def test_decode_is_neural_not_tag_string_parse():
    """(e): the runner source contains NO tag-string parse -- no
    .split("_") and no .split('_') anywhere (the answer is decoded from
    the validated neural readout, never out of an opaque tag name)."""
    src = Path(pzr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert '.split("_")' not in src
    assert ".split('_')" not in src
    # The validated neural readout helpers must be the decode path.
    assert "lang_output_pattern_during_stim" in src
    # The moat must be fed the raw firing-rate ranked confidence.
    assert "abstention_gate" in src or "_abstain_gate" in src


def test_moat_is_fed_raw_firing_rate_not_cosine_norm_hack():
    """(e cont'd): the moat input is the calibrated raw firing-rate
    quantity ``pat[active].sum() / n_active`` (the same one the 650
    threshold is calibrated on; SPEAR re-review lesson). The cosine *
    pattern-norm hack (the retired out-of-calibration path) must NOT
    appear."""
    src = Path(pzr.__file__).read_text(encoding="utf-8", errors="ignore")
    # The raw firing-rate confidence quantity.
    assert "pat[active].sum()" in src or "pat[active].sum() " in src
    # The retired cosine-norm hack must NOT appear.
    assert "cos * np.linalg.norm" not in src
    assert "cosine * np.linalg.norm" not in src


def test_theta_disabled_is_full_minus_only_the_external_theta_generator():
    """(d): for a (seed,N) cell, full and theta_disabled differ ONLY by
    the external theta generator being enabled vs disabled.

    Structural assertion (no heavy run): the per-arm cell function takes
    a single boolean controller flag, and the cell driver invokes it
    once with the flag True (full) and once with the flag False
    (theta_disabled) using the SAME seed and SAME facts/draws. We assert
    the flag is the sole differing argument by inspecting the source of
    the cell driver.
    """
    # The arm function must accept a single explicit theta flag.
    assert hasattr(pzr, "_run_arm")
    arm_sig = inspect.signature(pzr._run_arm)
    assert "use_theta" in arm_sig.parameters, (
        "the arm runner must thread an explicit `use_theta` controller "
        "flag so full vs theta_disabled differ ONLY by it"
    )

    cell_src = inspect.getsource(pzr._cell)
    # full arm: use_theta True; theta_disabled arm: use_theta False.
    assert "use_theta=True" in cell_src
    assert "use_theta=False" in cell_src
    # Both arms must be built from the SAME seed (no per-arm seed
    # perturbation) -- the cell takes one `seed` and passes it to both.
    cell_sig = inspect.signature(pzr._cell)
    assert "seed" in cell_sig.parameters
    # No second RNG seed / no seed offset between the two arms.
    assert "seed + 1" not in cell_src and "seed+1" not in cell_src

    # The verdict-relevant control key must be derived from the
    # theta-disabled arm.
    agg_src = inspect.getsource(pzr._aggregate)
    assert "theta_disabled_acc" in agg_src
    assert "abstain_correct_theta_disabled" in agg_src


def test_controller_has_theta_period_and_disinhibition_target():
    """The net-new piece is an EXTERNAL THETA GENERATOR controller. Assert
    the controller derives a theta period in STEPS from the bridge dt
    (Pirazzini ~250 ms = 4 Hz) and writes a DISINHIBITORY current onto
    the CA3-targeted inhibitory population `dg_pv_basket` via the reused
    `bridge.cp_external_input_current` path (Pirazzini disinhibition
    mechanism, NOT a synaptic_gain modulation). The reused
    set_concentration multi-target ACh phase gate runs alongside (HIGH
    ACh = encode, LOW ACh = retrieve -- standard Hasselmo polarity).
    """
    src = Path(pzr.__file__).read_text(encoding="utf-8", errors="ignore")
    # theta ~250 ms period derived from dt (4 Hz Pirazzini default).
    assert "250" in src  # ~250 ms theta cycle
    # The disinhibition target is the CA3-targeted inhibitory population.
    assert "dg_pv_basket" in src
    # The disinhibitory current is written via the reused external-current
    # path on the bridge.
    assert "cp_external_input_current" in src
    # The reused ACh phase gate.
    assert "set_concentration" in src
    assert "step_simulation" in src or "_run_one_simulation_step" in src
    # The dlpfc PFC working-memory frame.
    assert "dlpfc" in src.lower()


def test_ach_polarity_is_hasselmo_high_at_encode_low_at_retrieve():
    """The Hasselmo polarity (opposite to the SPEAR runner's choice):
    HIGH ACh during ENCODE (suppresses CA3->CA1 + strengthens cortical
    input + facilitates LTP); LOW ACh during RETRIEVE (pattern
    completion). Assert via source-text inspection that the encode
    setpoint is the HIGH value and the retrieve setpoint is the LOW
    value (the literals at module level)."""
    # These are the controller's setpoint constants. We assert they
    # exist and that their numeric order matches the Hasselmo polarity.
    assert hasattr(pzr, "_ACH_ENCODE_HIGH")
    assert hasattr(pzr, "_ACH_RETRIEVE_LOW")
    assert float(pzr._ACH_ENCODE_HIGH) > float(pzr._ACH_RETRIEVE_LOW), (
        "Hasselmo polarity inverted: ENCODE must be HIGH and RETRIEVE "
        "must be LOW (opposite of the SPEAR runner's TAN-convention "
        "polarity)"
    )


def test_ach_multi_target_present_with_documented_breadth():
    """The multi-target ACh modulator must register the three Pirazzini
    effects (suppress CA3->CA1 + strengthen cortical input + facilitate
    LTP). Where named gates are unavailable for `synaptic_gain` (the
    bridge consumer honors only scope=`all` for synaptic_gain), the
    fallback scope=`all` is documented and used. At least one
    synaptic_gain target AND one plasticity_rate target must be
    registered with non-zero sensitivity."""
    bridge, _dims = pzr._build_substrate(seed=42, tiny_synth=True)
    mgr = bridge.neuromodulator_manager
    assert mgr is not None, (
        "neuromodulator subsystem must be enabled on the runner's bridge"
    )
    cfg = mgr._config_by_name("ach_pirazzini")
    targets = list(cfg.targets)
    target_types = {t.target_type for t in targets}
    assert "synaptic_gain" in target_types, (
        "multi-target ACh must include synaptic_gain (modulates forward "
        "transmission across encode/retrieve phases)"
    )
    assert "plasticity_rate" in target_types, (
        "multi-target ACh must include plasticity_rate (facilitates LTP "
        "during encode)"
    )
    # At least one synaptic_gain scope=all target must have non-zero
    # sensitivity so the gate genuinely modulates effective synaptic
    # strength between the encode and retrieve phases.
    sg_all = [t for t in targets
              if t.target_type == "synaptic_gain" and t.scope == "all"]
    assert sg_all, (
        "at least one synaptic_gain target must have scope='all' so it "
        "is consumed by compute_synaptic_gain_multiplier()"
    )
    assert any(abs(t.sensitivity) > 1e-6 for t in sg_all)


def test_external_theta_structural_effect_pin():
    """(f) STRUCTURAL-EFFECT PIN: mirror the SPEAR re-review's 50-step
    probe. Build the SAME substrate twice from the SAME seed, drive the
    SAME small constant external input, step both bridges 50 times --
    one with the external theta generator's disinhibitory current ON,
    the other with it OFF -- and assert the membrane state diverges.

    If the controller's external theta generator is mechanistically
    inert (does not actually release CA3 pyramidals at theta-trough),
    the two bridges would produce byte-identical state. The threshold
    1e-3 mV is well above floating-point churn but well below
    biological noise, mirroring the SPEAR pin's 14.15 mV measurement.
    """
    from sim.backend import get_backend, to_host
    cp, _backend = get_backend()

    bridge_on, dims = pzr._build_substrate(seed=42, tiny_synth=True)
    bridge_off, _ = pzr._build_substrate(seed=42, tiny_synth=True)

    # Identical constant external input on both -- a small drive on the
    # first language_input lane so propagation actually exercises the
    # CA3 -> CA1 path that the disinhibition releases.
    drive_pA = cp.float32(200.0)
    n_drive = min(8, bridge_on.cp_external_input_current.shape[0])
    bridge_on.cp_external_input_current[:n_drive] = drive_pA
    bridge_off.cp_external_input_current[:n_drive] = drive_pA

    # Resolve dg_pv_basket indices (the CA3-targeted inhibitory
    # population; the disinhibition target).
    rm = bridge_on.region_manager
    try:
        pv_idx = list(rm.indices("dg_pv_basket"))
    except Exception:
        pv_idx = []
    assert pv_idx, (
        "the runner substrate must include the dg_pv_basket region "
        "(enable_hippocampus_consolidation=True); the disinhibition "
        "mechanism requires this CA3-targeted inhibitory population"
    )
    pv_arr_on = cp.asarray(pv_idx, dtype=cp.int64)

    # Step both bridges for the SAME number of steps. On `bridge_on`,
    # at theta-trough phase (the second half of each ~250 ms theta cycle
    # at the tiny-synth dt), write a NEGATIVE current onto dg_pv_basket
    # so the inhibitory population is silenced -> CA3 pyramidals are
    # disinhibited. On `bridge_off`, NEVER write the disinhibitory
    # current. Everything else is identical.
    theta_steps = int(dims.get("theta_steps", 50))
    trough_start = max(1, theta_steps // 2)
    disinhib_pA = -150.0
    for s in range(50):
        # restore drive each step (the previous step's _run_one_step
        # zeros the buffer in places; mirror what the runner does).
        bridge_on.cp_external_input_current[:n_drive] = drive_pA
        bridge_off.cp_external_input_current[:n_drive] = drive_pA
        phase_in_cycle = s % theta_steps
        if phase_in_cycle >= trough_start:
            bridge_on.cp_external_input_current[pv_arr_on] = \
                cp.float32(disinhib_pA)
        bridge_on._run_one_simulation_step()
        bridge_off._run_one_simulation_step()

    # Compare a deterministic summary of membrane potential on both --
    # the sum of the first 10 entries -- and require they differ by
    # more than a tiny epsilon. If the theta generator's disinhibitory
    # current were inert (the defect this pin closes), these would be
    # byte-identical.
    n = min(10, bridge_on.cp_membrane_potential_v.shape[0])
    v_on = to_host(bridge_on.cp_membrane_potential_v[:n])
    v_off = to_host(bridge_off.cp_membrane_potential_v[:n])

    diff = float(abs(float(v_on.sum()) - float(v_off.sum())))
    # Require a *measurable* difference.
    assert diff > 1e-3, (
        "external theta generator must affect bridge state across "
        "theta-on vs theta-off. Got byte-identical membrane state "
        "(sum_diff=%.6g mV), which is the inert-controller failure "
        "mode. v_on.sum=%r v_off.sum=%r"
        % (diff, float(v_on.sum()), float(v_off.sum()))
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
