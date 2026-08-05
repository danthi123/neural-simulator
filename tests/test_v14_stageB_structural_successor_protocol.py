import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/specs/v14_snr_stageB_structural_successor_v2.json"
PROTOCOL_SHA256 = "c0ab042b65cb7be21b640d9a14cddb13582a1662f3f0aa3159093aa1d13792c4"


def _load():
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA256
    return json.loads(PROTOCOL.read_text())


def test_protocol_binds_predecessors_and_measured_transfer_boundaries():
    protocol = _load()
    for binding in protocol["predecessor_evidence"]:
        path = ROOT / binding["path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]

    fast_na = protocol["source_transfers"]["fast_sodium"]
    kv3 = protocol["source_transfers"]["kv3_like"]
    assert fast_na["evidence_class"] == "direct_measured_transfer"
    assert fast_na["constraints"]["recovery_fast_tau_ms"]["mean"] == 0.59
    assert fast_na["constraints"]["recovery_slow_tau_ms"]["mean"] == 35.1
    assert fast_na["constraints"]["recovery_fast_fraction"]["mean"] == 0.526
    assert "effective whole-cell conductance after morphology and space-clamp correction" in fast_na[
        "unresolved"
    ]
    assert kv3["constraints"]["rise_20_80_at_plus_40_mV_ms"]["mean"] == 0.41
    assert "Kv3 conductance density" in kv3["unresolved"]


def test_architecture_keeps_local_calcium_load_bearing_and_axial_current_conservative():
    architecture = _load()["architecture"]
    assert architecture["calcium_compartments"]["local"] == (
        "receives proximal-dendritic Cav2.2 influx and is the only calcium state allowed to drive SK"
    )
    assert "sk_activation" in architecture["proximal_dendrite"]["state"]
    assert "sk" not in architecture["soma"]["currents"]
    assert architecture["axial_coupling"]["conservation"] == (
        "equal magnitude and opposite sign before compartment area normalization"
    )
    equations = architecture["equation_contract"]
    assert equations["steady_state_forms"]["fast_na_activation_gate"].startswith(
        "m_inf(V)=A_fast_na_inf(V)^(1/3)"
    )
    assert equations["steady_state_forms"]["kv3_activation_gate"].startswith(
        "n_inf(V)=A_kv3_inf(V)^(1/4)"
    )
    assert "dV_soma/dt" in equations["soma_balance"]
    assert "dN_local/dt" in equations["calcium_mass_balance"]["local"]
    assert equations["fixed_update_order"][-1].endswith("without resetting either compartment")
    assert "bypass the generic HH" in architecture["production_bridge_ownership"][
        "double_counting_guard"
    ]
    assert "host-side spike reset, pacemaker current, or calcium-to-SK shortcut" in architecture[
        "excluded_from_v1"
    ]


def test_gate_order_forbids_calibration_before_equation_and_compartment_readiness():
    protocol = _load()
    stages = protocol["validation_ladder"]
    assert [stage["stage"] for stage in stages] == list(range(7))
    assert [stage["id"] for stage in stages[:4]] == [
        "contract_and_units",
        "isolated_fast_channel_voltage_clamp",
        "two_compartment_numerics",
        "local_bulk_calcium_separation",
    ]
    assert stages[4]["id"] == "absolute_waveform_transfer_gate"
    assert stages[5]["status"].startswith("blocked_until_stage_4")
    assert any("do not compensate with parameter search" in rule for rule in protocol["stop_rules"])
    assert any("candidate calibration or promotion" in rule for rule in protocol["scope"]["not_authorized"])


def test_causal_successor_adds_sham_and_keeps_unavailable_cohort_closed():
    stage = _load()["validation_ladder"][5]
    assert stage["id"] == "future_calibration_and_causal_confirmation"
    assert "phase-matched intact continuation" in stage["required_arms"]
    assert any("spike-phase confound" in gate for gate in stage["hard_gates"])
    assert any("remains unavailable" in gate for gate in stage["hard_gates"])


def test_resource_policy_reserves_authority_and_shared_gpu_lease():
    policy = _load()["resource_policy"]
    assert "NumPy authority" in policy["local_cpu"]
    assert "shared lease" in policy["local_gpu"]
    assert "independent NumPy replicas" in policy["mini_pc_pool"]


def test_fast_channel_commands_and_performance_budget_are_frozen():
    stages = _load()["validation_ladder"]
    clamp = stages[1]
    assert "20 ms tests from -80 through +30 mV" in clamp["source_command_protocols"][
        "fast_na_activation"
    ]
    assert "10 s prepulses from -110 through 0 mV" in clamp["source_command_protocols"][
        "kv3_steady_state_inactivation"
    ]
    assert clamp["project_operational_recovery_duration_ladder_ms"] == [
        0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0,
    ]
    assert "model prior" in clamp["kinetic_model_prior_boundary"]["tau_interpolation"]
    numerics = stages[2]
    assert numerics["fixed_environment"]["software_identity"]["cupy"] == "14.1.1"
    assert any("100000 complete simulation steps per second" in gate for gate in numerics["hard_gates"])
