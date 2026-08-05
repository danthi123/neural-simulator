import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.runners import v14_stageB_failure_diagnostic as diagnostic


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "research/specs/v14_snr_stageB_failure_diagnostic_v1.json"
PROTOCOL_SHA256 = "45ab095d0f4b3dde22729e9f4d9edb49a600179fef718b0c3d01473a497a17c0"
EXPECTED_IDS = [
    "v14-stageB-v3-sobol-0628-d2eaa93190ac",
    "v14-stageB-v3-sobol-0756-6d8e2f95b7f9",
    "v14-stageB-v3-sobol-0771-7b5ec8c73929",
    "v14-stageB-v3-sobol-0816-4f32e6b9d15a",
    "v14-stageB-v3-sobol-0869-8d3233911ad2",
    "v14-stageB-v3-sobol-0923-01dba8058443",
    "v14-stageB-v3-sobol-0961-dea82b0ba466",
    "v14-stageB-v3-sobol-0999-ccdb750c02c4",
    "v14-stageB-v3-sobol-1010-e3e90562e351",
]


def test_real_protocol_authenticates_and_rederives_exact_selection():
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == PROTOCOL_SHA256

    loaded = diagnostic.load_failure_diagnostic(
        PROTOCOL, PROTOCOL_SHA256, repository_root=ROOT
    )

    assert [row["candidate_id"] for row in loaded["selection"]] == EXPECTED_IDS
    assert [row["candidate_id"] for row in loaded["candidates"]] == EXPECTED_IDS
    assert loaded["declaration"]["arm"] == "nap_lesion"
    assert loaded["campaign_binding"]["sha256"] == (
        "b6a659bdefbc920c9a7dadf9c5a4d3f1a03778cae0460b807d153be49f75a997"
    )
    assert loaded["triage_binding"]["sha256"] == (
        "40ec99d4c8626bbdf21a968436a3612b693fef49cec5054e255fc0e8d195f703"
    )


def test_protocol_digest_is_mandatory_before_source_loading():
    with pytest.raises(diagnostic.StageBFailureDiagnosticError, match="digest"):
        diagnostic.load_failure_diagnostic(
            PROTOCOL, "0" * 64, repository_root=ROOT
        )


def test_selection_rule_rejects_unresolved_and_other_failures():
    wanted = {
        "candidate_id": "wanted",
        "candidate_sha256": "1" * 64,
        "resolved_checks": [
            {
                "gate_id": "nap-complete-lesion",
                "metric": "median_membrane_voltage_change_mV",
                "passed": False,
            },
            {"gate_id": "sk-complete-lesion", "metric": "isi_cv", "passed": True},
        ],
    }
    unresolved = json.loads(json.dumps(wanted))
    unresolved["candidate_id"] = "unresolved"
    unresolved["resolved_checks"][1]["passed"] = None
    other = json.loads(json.dumps(wanted))
    other["candidate_id"] = "other"
    other["resolved_checks"][0]["metric"] = "spike_count"
    two_failures = json.loads(json.dumps(wanted))
    two_failures["candidate_id"] = "two"
    two_failures["resolved_checks"][1]["passed"] = False

    assert diagnostic._derive_selection(
        {"candidates": [unresolved, wanted, other, two_failures]}
    ) == [{"candidate_id": "wanted", "candidate_sha256": "1" * 64}]


def test_phase_plan_has_exact_post_update_boundaries():
    protocol = json.loads(PROTOCOL.read_text())

    phases = diagnostic._phase_plan(protocol)

    assert [phase["steps"] for phase in phases] == [39_999, 20_000, 10_000, 20_001]
    assert [(phase["first_sample_number"], phase["last_sample_number"]) for phase in phases] == [
        (1, 39_999),
        (40_000, 59_999),
        (60_000, 69_999),
        (70_000, 90_000),
    ]
    assert phases[1]["first_sample_index"] == 39_999
    assert phases[2]["first_sample_index"] == 59_999
    assert phases[3]["first_sample_index"] == 69_999
    assert sum(phase["steps"] for phase in phases) == 90_000


def test_phase_plan_rejects_boundary_drift():
    protocol = json.loads(PROTOCOL.read_text())
    protocol["execution"]["phases"][2]["start_s"] = 3.00005

    with pytest.raises(diagnostic.StageBFailureDiagnosticError, match="boundary"):
        diagnostic._phase_plan(protocol)


@pytest.mark.parametrize(
    ("current_pA", "density", "bridge_numeric"),
    [
        (0.0, 0.0, 0.0),
        (-10.0, -0.5, -500_000.0),
        (-20.0, -1.0, -1_000_000.0),
        (-30.0, -1.5, -1_500_000.0),
    ],
)
def test_current_conversion_matches_stage_b_bridge_contract(
    current_pA, density, bridge_numeric
):
    assert diagnostic.current_density_uA_per_cm2(current_pA, 2000.0) == density
    assert (
        diagnostic.bridge_external_current_numeric(current_pA, 2000.0)
        == bridge_numeric
    )
    assert bridge_numeric * 1.0e-6 == density


@pytest.mark.parametrize("area", [0.0, -1.0, float("nan"), True])
def test_current_conversion_rejects_invalid_area(area):
    with pytest.raises(diagnostic.StageBFailureDiagnosticError):
        diagnostic.current_density_uA_per_cm2(-10.0, area)


def test_validate_only_never_constructs_runtime_or_writes_output(monkeypatch, capsys):
    loaded = {
        "protocol_binding": {"path": "protocol.json", "sha256": "1" * 64},
        "campaign_binding": {"path": "campaign.json", "sha256": "2" * 64},
        "triage_binding": {"path": "triage.json", "sha256": "3" * 64},
        "declaration_binding": {
            "path": "declaration.json",
            "sha256": "4" * 64,
            "self_sha256": "5" * 64,
        },
        "selection": [{"candidate_id": "c", "candidate_sha256": "6" * 64}],
        "phase_plan": ({"name": "phase", "steps": 1},),
    }
    monkeypatch.setattr(diagnostic, "load_failure_diagnostic", lambda *a, **k: loaded)
    monkeypatch.setattr(
        diagnostic,
        "_runtime_components",
        lambda: pytest.fail("validate-only loaded the GPU runtime"),
    )

    assert diagnostic.main(
        [
            "--protocol",
            "protocol.json",
            "--protocol-sha256",
            "1" * 64,
            "--validate-only",
        ]
    ) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["process_status"] == "validated_not_executed"
    assert result["scientific_verdict"] is None
    assert result["engineering_diagnostic_only"] is True


def test_execution_requires_receipt_path(capsys):
    with pytest.raises(SystemExit) as exc:
        diagnostic.main(
            [
                "--protocol",
                str(PROTOCOL),
                "--protocol-sha256",
                PROTOCOL_SHA256,
            ]
        )
    assert exc.value.code == 2
    assert "requires --output" in capsys.readouterr().err


def test_fake_runtime_executes_four_fresh_fixed_arms_and_publishes_once(
    tmp_path, monkeypatch
):
    candidate = {
        "candidate_id": "diagnostic-candidate",
        "candidate_sha256": "7" * 64,
        "region_name": "candidate_region",
        "packet": {"path": "packet.json", "sha256": "8" * 64},
        "policy": {"path": "policy.json", "sha256": "9" * 64},
    }
    phases = tuple(
        {
            "name": name,
            "start_s": start,
            "end_s": end,
            "first_sample_index": index,
            "first_sample_number": index + 1,
            "last_sample_index": index,
            "last_sample_number": index + 1,
            "steps": 1,
            "g_nap": g_nap,
            "external_current_pA": external,
        }
        for index, (name, start, end, g_nap, external) in enumerate(
            diagnostic.EXPECTED_PHASES
        )
    )
    loaded = {
        "repository_root": tmp_path,
        "protocol": {"execution": {"membrane_area_um2": 2000.0}},
        "protocol_binding": {"path": "protocol.json", "sha256": "1" * 64},
        "campaign_binding": {"path": "campaign.json", "sha256": "2" * 64},
        "triage_binding": {"path": "triage.json", "sha256": "3" * 64},
        "declaration_binding": {
            "path": "declaration.json",
            "sha256": "4" * 64,
            "self_sha256": "5" * 64,
        },
        "selection": [
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_sha256": candidate["candidate_sha256"],
            }
        ],
        "candidates": [candidate],
        "phase_plan": phases,
    }
    monkeypatch.setattr(diagnostic, "load_failure_diagnostic", lambda *a, **k: loaded)

    class FakeBridge:
        instances = []

        def __init__(self, core_config, **_kwargs):
            self.is_initialized = False
            self.external_history = []
            self.cp_membrane_potential_v = np.zeros(1, dtype=np.float32)
            self.cp_firing_states = np.zeros(1, dtype=bool)
            self.cp_gating_variable_m = np.zeros(1, dtype=np.float32)
            self.cp_gating_variable_h = np.zeros(1, dtype=np.float32)
            self.cp_gating_variable_n = np.zeros(1, dtype=np.float32)
            self.cp_snr_nap_activation = np.zeros(1, dtype=np.float32)
            self.cp_snr_nap_inactivation = np.zeros(1, dtype=np.float32)
            self.cp_snr_ca_activation = np.zeros(1, dtype=np.float32)
            self.cp_snr_ca_inactivation = np.zeros(1, dtype=np.float32)
            self.cp_snr_calcium = np.zeros(1, dtype=np.float32)
            self.cp_snr_sk_activation = np.zeros(1, dtype=np.float32)
            self.cp_snr_h_activation = np.zeros(1, dtype=np.float32)
            self.cp_snr_ionic_current_scratch = np.zeros(1, dtype=np.float32)
            self.cp_external_input_current = np.zeros(1, dtype=np.float32)
            self.cp_hh_C_m = np.ones(1, dtype=np.float32)
            self.cp_hh_g_Na_max = np.zeros(1, dtype=np.float32)
            self.cp_hh_g_K_max = np.zeros(1, dtype=np.float32)
            self.cp_hh_g_L = np.zeros(1, dtype=np.float32)
            self.cp_hh_E_Na = np.zeros(1, dtype=np.float32)
            self.cp_hh_E_K = np.zeros(1, dtype=np.float32)
            self.cp_hh_E_L = np.zeros(1, dtype=np.float32)
            self.cp_snr_g_nalcn_max = np.zeros(1, dtype=np.float32)
            self.cp_snr_g_nap_max = np.ones(1, dtype=np.float32)
            self.cp_snr_g_ca_max = np.zeros(1, dtype=np.float32)
            self.cp_snr_g_sk_max = np.zeros(1, dtype=np.float32)
            self.cp_snr_g_h_max = np.zeros(1, dtype=np.float32)
            self.snr_packet_kernel_parameters = {
                name: np.zeros(1, dtype=np.float32)
                for name in (
                    "E_nalcn_mv",
                    "E_ca_mv",
                    "E_hcn_mv",
                    "cav22_activation_power",
                )
            }
            binding = SimpleNamespace(
                packet_path=candidate["packet"]["path"],
                packet_file_sha256=candidate["packet"]["sha256"],
                authority_policy_sha256=candidate["policy"]["sha256"],
                runtime_parameters=SimpleNamespace(
                    geometry=SimpleNamespace(membrane_area_um2=2000.0)
                ),
            )
            self.snr_packet_bindings = {candidate["region_name"]: binding}
            self.core_config = core_config
            self.instances.append(self)

        def _initialize_simulation_data(self):
            self.is_initialized = True

        def _snr_direct_outputs_can_dispatch(self, _config):
            return True

        def _run_one_simulation_step(self):
            density = self.cp_external_input_current * np.float32(1.0e-6)
            self.external_history.append(float(density[0]))
            self.cp_snr_ionic_current_scratch[:] = density
            self.cp_membrane_potential_v[:] += density * np.float32(diagnostic.DT_MS)

        def clear_simulation_state_and_gpu_memory(self):
            self.is_initialized = False

    def prepare_capture(arguments):
        def execute(*values):
            for source, destination in zip(values[:14], values[14:], strict=True):
                destination[...] = source

        execute(*arguments)
        return execute

    def currents(*arguments):
        return tuple(np.zeros_like(arguments[0]) for _ in range(9))

    from tools.diagnostic_trace import save_diagnostic_trace

    runtime = SimpleNamespace(
        xp=np,
        SimulationBridge=FakeBridge,
        VisualizationConfig=SimpleNamespace,
        RuntimeState=SimpleNamespace,
        GPUConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        build_config=lambda candidates, maximum_steps, runtime: SimpleNamespace(
            candidates=candidates, maximum_steps=maximum_steps
        ),
        binding_provenance=lambda _binding: {"binding": "authenticated-fake"},
        runtime_binding_manifest_bytes=lambda _bindings: b"same-bindings",
        synchronize=lambda _xp: None,
        to_host=lambda _xp, value: np.asarray(value),
        prepare_capture=prepare_capture,
        diagnostic_currents=currents,
        save_trace=save_diagnostic_trace,
    )
    output = tmp_path / "results" / "receipt.json"

    result = diagnostic.run_failure_diagnostic(
        "protocol.json",
        "1" * 64,
        output,
        repository_root=tmp_path,
        chunk_steps=1,
        _runtime=runtime,
    )

    assert len(FakeBridge.instances) == 4
    assert [bridge.external_history for bridge in FakeBridge.instances] == [
        [0.0, 0.0, current, 0.0] for current in (0.0, -0.5, -1.0, -1.5)
    ]
    assert result["engineering_diagnostic_only"] is True
    assert result["scientific_verdict"] is None
    assert result["execution"]["adaptive_decisions"] is False
    assert result["execution"]["trace_count"] == 4
    assert result["attribution"] == {
        "label": "sustained firing attributable to NaP presence",
        "baseline_window_s": 0.00005,
        "lesion_window_s": 0.00005,
        "candidate_baseline_rate_hz": [0.0],
        "candidate_nap_lesion_rate_hz": [0.0],
        "cohort_baseline_median_rate_hz": 0.0,
        "cohort_nap_lesion_median_rate_hz": 0.0,
        "attributable_fraction": None,
    }
    assert output.read_bytes() == diagnostic._canonical_bytes(result)
    assert result["sha256"] == diagnostic._digest(
        {key: value for key, value in result.items() if key != "sha256"}
    )
    with pytest.raises(diagnostic.StageBFailureDiagnosticError, match="replace"):
        diagnostic.run_failure_diagnostic(
            "protocol.json",
            "1" * 64,
            output,
            repository_root=tmp_path,
            chunk_steps=1,
            _runtime=runtime,
        )
