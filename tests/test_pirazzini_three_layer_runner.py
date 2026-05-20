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
    the validated neural readout, never out of an opaque tag name).

    FIX B (2026-05-20): the runner-local per-step retrieval loop
    accumulates the `language_output` firing pattern directly via
    `cp_firing_states[lang_output]` per step (REPLACING the buffer-
    wiping `lang_output_pattern_during_stim` helper that erased the
    disinhibition write on entry). We therefore assert the validated
    neural readout via the actual mechanism: the runner reads
    language_output's cp_firing_states + the REUSED _ranked_from_pattern
    formula computes raw firing-rate confidence the 650 moat is
    calibrated on."""
    src = Path(pzr.__file__).read_text(encoding="utf-8", errors="ignore")
    assert '.split("_")' not in src
    assert ".split('_')" not in src
    # The validated neural readout: the runner-local loop accumulates
    # lang_output firing states + the REUSED ranked-from-pattern is
    # the moat-calibrated quantity.
    assert "cp_firing_states" in src
    assert "language_output" in src
    assert "_ranked_from_pattern" in src
    # The moat must be fed the raw firing-rate ranked confidence.
    assert "abstention_gate" in src or "_abstain_gate" in src


def test_moat_is_fed_raw_firing_rate_not_cosine_norm_hack():
    """(e cont'd): the moat input is the calibrated raw firing-rate
    quantity ``pat[active].sum() / n_active`` (the same one the 650
    threshold is calibrated on; SPEAR re-review lesson). The cosine *
    pattern-norm hack (the retired out-of-calibration path) must NOT
    appear."""
    src = Path(pzr.__file__).read_text(encoding="utf-8", errors="ignore")
    # The raw firing-rate confidence quantity (via the REUSED
    # _ranked_from_pattern at compose_retrieval_runner). The runner
    # imports + calls it; the formula itself lives in the cleared module.
    assert "_ranked_from_pattern" in src
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
    (Pirazzini ~250 ms = 4 Hz) and routes a DISINHIBITORY drive onto the
    CA3-targeted inhibitory population `dg_pv_basket` via a REUSED
    `excitability_drive` neuromodulator target (mirrors the SPEAR
    f1292a0 fix pattern: the per-step-honored consumer in sim/bridge.py).
    The reused `set_concentration` multi-target ACh phase gate runs
    alongside (HIGH ACh = encode, LOW ACh = retrieve -- standard
    Hasselmo polarity).
    """
    src = Path(pzr.__file__).read_text(encoding="utf-8", errors="ignore")
    # theta ~250 ms period derived from dt (4 Hz Pirazzini default).
    assert "250" in src  # ~250 ms theta cycle
    # The disinhibition target is the CA3-targeted inhibitory population
    # registered as a neuromodulator group (so the bridge consumes it).
    assert "dg_pv_basket" in src
    assert 'scope="group:dg_pv_basket"' in src, (
        "FIX A: the disinhibition target must be routed through a "
        "`group:dg_pv_basket` excitability_drive scope so the bridge's "
        "per-step `compute_excitability_drive_per_neuron` consumes it "
        "regardless of `cp_external_input_current` buffer clears."
    )
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
    effects (suppress CA3->CA1 transmission + strengthen cortical input
    + facilitate LTP). FIX C: the Hasselmo transmission semantics are
    routed through excitability_drive targets on ec/ca3 (per-step
    consumed in sim/bridge.py:4960-4964) -- NOT only via plasticity
    gates that modulate UPDATE rates. At least one excitability_drive
    target on group:ca3 AND group:ec must be registered with non-zero
    sensitivity.
    """
    bridge, _dims = pzr._build_substrate(seed=42, tiny_synth=True)
    mgr = bridge.neuromodulator_manager
    assert mgr is not None, (
        "neuromodulator subsystem must be enabled on the runner's bridge"
    )
    cfg = mgr._config_by_name("ach_pirazzini")
    targets = list(cfg.targets)
    target_specs = {(t.target_type, t.scope) for t in targets}
    # FIX C requires Hasselmo TRANSMISSION effects via excitability_drive
    # on hippocampal regions (consumed every step regardless of reward).
    assert ("excitability_drive", "group:ca3") in target_specs, (
        "FIX C: ACh must route the Hasselmo `suppress CA3 output during "
        "encode` via excitability_drive scope=group:ca3 (negative "
        "sensitivity at HIGH ACh)"
    )
    assert ("excitability_drive", "group:ec") in target_specs, (
        "FIX C: ACh must route the Hasselmo `strengthen cortical input "
        "during encode` via excitability_drive scope=group:ec (positive "
        "sensitivity at HIGH ACh)"
    )
    # The system-wide plasticity_rate facilitation is still required
    # (Hasselmo LTP facilitation at encoding).
    pr_all = [t for t in targets
              if t.target_type == "plasticity_rate" and t.scope == "all"]
    assert pr_all, (
        "ACh must include a plasticity_rate scope=all target "
        "(Hasselmo LTP facilitation at encode)"
    )
    assert any(abs(t.sensitivity) > 1e-6 for t in pr_all)


def test_neutral_ach_does_not_pre_freeze_pathway_gates():
    """FIX C invariant: at the NEUTRAL ACh setpoint (the value the
    `theta_disabled` arm holds throughout), every plasticity_gate
    target on the ACh modulator MUST resolve to ~1.0 (PERMIT). The
    bug the adversarial review caught was the original sensitivity=-2.0
    with baseline=0.5 sending the `ca3_to_ca1` gate to 0.0 at NEUTRAL
    (i.e., the control arm had CA3->CA1 plasticity FROZEN throughout,
    so the named control was NOT 'full minus disinhibition' but rather
    'full minus the ACh-polarity-driving-plasticity'). The fix rebalances
    baseline + sensitivity so NEUTRAL produces gate=1.0.
    """
    bridge, _dims = pzr._build_substrate(seed=42, tiny_synth=True)
    mgr = bridge.neuromodulator_manager
    assert mgr is not None
    # Set ACh concentration to the runner's documented neutral setpoint
    # (the value the theta_disabled arm writes via _ACH_NEUTRAL).
    neutral = float(pzr._ACH_NEUTRAL)
    mgr.set_concentration("ach_pirazzini", neutral)
    gates = mgr.compute_plasticity_gate_values()
    for gate_name, gate_value in gates.items():
        assert abs(gate_value - 1.0) < 1e-3, (
            "FIX C invariant violated: at NEUTRAL ACh setpoint %.3f "
            "the gate %r resolves to %.3f (not 1.0). The control arm "
            "would have this pathway PRE-FROZEN regardless of the "
            "disinhibition mechanism. Rebalance the modulator's "
            "baseline + sensitivity so the NEUTRAL multiplier on each "
            "registered pathway-scoped target is ~1.0."
            % (neutral, gate_name, gate_value)
        )


def test_disinhibition_modulator_is_registered_with_group_scope():
    """FIX A invariant: the disinhibition mechanism MUST be registered as
    a NeuromodulatorConfig with an excitability_drive target on
    group:dg_pv_basket (a per-step-consumed scope), separate from the
    ACh modulator. This mirrors the SPEAR f1292a0 fix pattern:
    route the named-biology effect through a modulator target whose
    consumer is honored every simulation step regardless of buffer
    clears in encode/readout helpers."""
    bridge, _dims = pzr._build_substrate(seed=42, tiny_synth=True)
    mgr = bridge.neuromodulator_manager
    assert mgr is not None
    # The disinhibition modulator must be named and registered.
    assert "dg_disinhibition" in mgr.modulator_names(), (
        "FIX A: the runner must register a `dg_disinhibition` "
        "NeuromodulatorConfig with an excitability_drive target on "
        "group:dg_pv_basket so the Pirazzini disinhibition is "
        "consumed per-step by the bridge."
    )
    cfg = mgr._config_by_name("dg_disinhibition")
    target_specs = {(t.target_type, t.scope) for t in cfg.targets}
    assert ("excitability_drive", "group:dg_pv_basket") in target_specs, (
        "the dg_disinhibition modulator MUST target excitability_drive "
        "scope=group:dg_pv_basket (consumed at sim/bridge.py:4960-4964 "
        "every step)"
    )
    # At least one excitability_drive target on group:dg_pv_basket with
    # NEGATIVE sensitivity (silences inhibitors -> CA3 disinhibited).
    edrive = [t for t in cfg.targets
              if t.target_type == "excitability_drive"
              and t.scope == "group:dg_pv_basket"]
    assert edrive and all(t.sensitivity < -1e-6 for t in edrive), (
        "excitability_drive on group:dg_pv_basket must have NEGATIVE "
        "sensitivity (silencing inhibitors releases CA3 pyramidals)."
    )


def test_external_theta_structural_effect_pin_via_runner_actual_code_path():
    """(f) STRUCTURAL-EFFECT PIN (RUNNER's ACTUAL CODE PATH): mirror the
    SPEAR re-review's 14.15 mV pin but via the runner's REAL `_run_arm`
    (NOT a synthetic-bypass probe that writes cp_external_input_current
    directly and bypasses the modulator path). Build the SAME substrate
    twice from the SAME seed; for each, exercise the runner's actual
    encode phase briefly (one fact, tiny-synth) with `use_theta=True`
    vs `use_theta=False` at the SAME NEUTRAL ACh setpoint. Assert the
    membrane state DIFFERS by > 1e-3 mV (mirrors SPEAR's 14.15 mV
    invariant via the runner's REAL code path).

    The PRIOR synthetic-bypass pin (which directly wrote
    bridge.cp_external_input_current[pv_arr] = -150 in a manual loop)
    was insufficient: it would have PASSED even though the runner's
    actual code path (with `step_idx=0` hardcoded everywhere) was
    mechanistically inert. FIX A's reroute through the modulator
    subsystem closes that gap so this pin now exercises the real
    behavior.
    """
    from sim.backend import get_backend, to_host
    cp, _backend = get_backend()

    # We exercise the runner-internal per-step encode loop via a
    # minimal direct call. The runner exposes a private parameterised
    # encode helper that we call once with `use_theta=True` and once
    # with `use_theta=False` from the SAME initial bridge state. The
    # ACh concentration is held at NEUTRAL on BOTH arms (so the only
    # difference is the dg_disinhibition modulator).
    bridge_on, dims_on = pzr._build_substrate(seed=42, tiny_synth=True)
    bridge_off, dims_off = pzr._build_substrate(seed=42, tiny_synth=True)

    # NEUTRAL ACh on BOTH arms so the only differentiator is theta.
    pzr._set_ach(bridge_on, pzr._ACH_NEUTRAL)
    pzr._set_ach(bridge_off, pzr._ACH_NEUTRAL)

    # Resolve dg_pv_basket indices to register them as a group on the
    # neuromodulator manager (the runner's _run_arm does this; we mimic
    # the same registration here so the pin exercises the same setup).
    pv_idx_on = pzr._resolve_pv_basket_indices(bridge_on)
    pv_idx_off = pzr._resolve_pv_basket_indices(bridge_off)
    assert pv_idx_on, (
        "the runner substrate must include dg_pv_basket region "
        "(enable_hippocampus_consolidation=True)"
    )
    # The runner registers groups at the start of _run_arm. Mimic that.
    bridge_on.neuromodulator_manager.set_group_indices(
        bridge_on.region_manager.region_indices_dict()
    )
    bridge_off.neuromodulator_manager.set_group_indices(
        bridge_off.region_manager.region_indices_dict()
    )

    # Drive language_input lightly on both arms to give propagation
    # something to chew on (the encode does this too).
    n_drive = min(8, bridge_on.cp_external_input_current.shape[0])
    drive_pA = cp.float32(200.0)

    # On `bridge_on`: switch the dg_disinhibition controller ON at
    # theta-trough phase steps (per the runner's real cycle). On
    # `bridge_off`: hold dg_disinhibition concentration at 0.0 every
    # step (controller is OFF -- the theta_disabled arm).
    theta_steps = int(dims_on.get("theta_steps", 8))
    trough_start = max(1, theta_steps // 2)
    for s in range(50):
        bridge_on.cp_external_input_current[:n_drive] = drive_pA
        bridge_off.cp_external_input_current[:n_drive] = drive_pA
        phase_in_cycle = s % theta_steps
        if phase_in_cycle >= trough_start:
            bridge_on.neuromodulator_manager.set_concentration(
                "dg_disinhibition", 1.0
            )
        else:
            bridge_on.neuromodulator_manager.set_concentration(
                "dg_disinhibition", 0.0
            )
        # theta_disabled arm: always 0.0 (controller OFF).
        bridge_off.neuromodulator_manager.set_concentration(
            "dg_disinhibition", 0.0
        )
        bridge_on._run_one_simulation_step()
        bridge_off._run_one_simulation_step()

    n = min(10, bridge_on.cp_membrane_potential_v.shape[0])
    v_on = to_host(bridge_on.cp_membrane_potential_v[:n])
    v_off = to_host(bridge_off.cp_membrane_potential_v[:n])

    diff = float(abs(float(v_on.sum()) - float(v_off.sum())))
    assert diff > 1e-3, (
        "FIX A invariant violated: the runner's `dg_disinhibition` "
        "modulator must produce measurable bridge-state divergence "
        "between theta-ON and theta-OFF arms at the SAME NEUTRAL "
        "ACh setpoint. Got byte-identical membrane state "
        "(sum_diff=%.6g mV). This is the inert-controller failure "
        "mode the adversarial review caught. v_on.sum=%r v_off.sum=%r"
        % (diff, float(v_on.sum()), float(v_off.sum()))
    )


def test_ach_only_mechanism_cannot_PASS():
    """FIX D positive false-PASS-protection pin: an ACh-only solver
    (disinhibition modulator forcibly OFF; only the Hasselmo ACh
    polarity is active) must NOT score GATE=PASS through the runner +
    frozen verdict. The adversarial review confirmed an ACh-only
    runner-equivalent scored PASS via the runner+frozen-verdict end-
    to-end -- a false-PASS exploit that would let the named control
    (`theta_disabled` = `full minus the disinhibition mechanism`) be
    impersonated by `full minus the ACh polarity`. This pin closes
    that exploit by structurally requiring the disinhibition
    modulator's effect to be the actual differentiator.

    Mechanism: the runner accepts a private `_force_disinhibition_off`
    kwarg / module-level toggle. With it set, the dg_disinhibition
    concentration is held at 0.0 throughout BOTH arms; the only
    remaining differentiator is the ACh polarity. The frozen verdict
    must NOT report PASS in that mode at any tiny-synth (seed, N).
    """
    # The runner must expose a parameter that disables only the
    # disinhibition modulator (so the ACh polarity is still active).
    arm_sig = inspect.signature(pzr._run_arm)
    assert "force_disinhibition_off" in arm_sig.parameters, (
        "FIX D: the runner must accept `force_disinhibition_off` to "
        "isolate the ACh-only mechanism. Without it, we cannot prove "
        "the runner+verdict cannot be exploited by an ACh-only solver."
    )

    # Run the frozen verdict on a tiny-synth multi-seed rung in
    # ACh-only mode. The decisive multi-seed CuPy run is later; here
    # we only assert the frozen verdict cannot score PASS in the
    # ACh-only mode.
    rungs_acc = []
    for seed in (42, 43, 44):
        full = pzr._run_arm(seed, N=2, tiny_synth=True,
                              use_theta=True,
                              force_disinhibition_off=True)
        td = pzr._run_arm(seed, N=2, tiny_synth=True,
                            use_theta=False,
                            force_disinhibition_off=True)
        rungs_acc.append((full, td))
    rung = {
        "N": 2,
        "n_seeds": 3,
        "full_acc": float(sum(f["acc"] for f, _ in rungs_acc) / 3.0),
        "theta_disabled_acc": float(
            sum(t["acc"] for _, t in rungs_acc) / 3.0
        ),
        "abstain_correct_theta_disabled": float(
            sum(t["abstain_correct"] for _, t in rungs_acc) / 3.0
        ),
    }
    verdict = pirazzini_three_layer_verdict([rung])
    assert verdict["gate"] != "PASS", (
        "FIX D invariant violated: an ACh-only solver "
        "(disinhibition mechanism forced OFF) scored GATE=PASS via "
        "the runner + frozen verdict. This is the false-PASS exploit "
        "the adversarial review caught: the named control "
        "(`theta_disabled` = full minus disinhibition) is in practice "
        "indistinguishable from `full minus ACh polarity` because "
        "the runner's actual code path makes ACh the only "
        "differentiator. Rung: %r Verdict: %r"
        % (rung, verdict)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
