import json
import os

import pytest


os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _vocal_action_credit_gate_v11_boundary as gate


def test_v11_seed_boundary_keeps_reserved_execution_sealed():
    assert gate.validate_construction_seed(gate.CONSTRUCTION_SEED) == 991
    with pytest.raises(ValueError, match="reserved seed 1 remains sealed"):
        gate.validate_construction_seed(gate.RESERVED_SEED)
    with pytest.raises(ValueError, match="formal execution is sealed"):
        gate.validate_formal_seed(gate.RESERVED_SEED)


def test_v11_topology_is_symmetric_polarized_and_nmda_scoped():
    bridge, handles = gate.build_v11_bridge(
        gate.CONSTRUCTION_SEED, recurrence_weight=0.25
    )
    audit = gate.boundary_structural_audit(bridge, handles)

    assert bridge.core_config.num_neurons == 688
    assert len(bridge.core_config.region_pathways) == 49
    assert audit["pass"] is True
    assert all(audit["checks"].values())
    assert audit["nmda_regions"] == ["action_corollary"]
    assert bridge.core_config.enable_nmda is True
    assert bridge.core_config.reward_learning_rate == 0.0


def test_v11_lesions_are_fixed_symmetric_transmission_gates():
    bridge, handles = gate.build_v11_bridge(
        gate.CONSTRUCTION_SEED,
        recurrence_weight=0.5,
        gate_values={
            gate.MOTOR_COPY_GATE: 0.0,
            gate.RECURRENCE_GATE: 0.0,
        },
    )

    assert handles["gate_values"] == {
        gate.MOTOR_COPY_GATE: 0.0,
        gate.RECURRENCE_GATE: 0.0,
        gate.PROPOSAL_STOP_GATE: 1.0,
        gate.COMMIT_STOP_GATE: 1.0,
    }
    assert bridge._transmission_gate_values[gate.MOTOR_COPY_GATE] == 0.0
    assert bridge._transmission_gate_values[gate.RECURRENCE_GATE] == 0.0


def test_cross_backend_merge_selects_first_shared_passing_weight(tmp_path):
    def artifact(backend, passes):
        return {
            "stage": "construction_backend",
            "seed": gate.CONSTRUCTION_SEED,
            "recurrence_weights": list(gate.RECURRENCE_WEIGHTS),
            "selected_weight_this_backend": next(
                (weight for weight, passed in zip(gate.RECURRENCE_WEIGHTS, passes)
                 if passed),
                None,
            ),
            "backend_info": {"backend": backend},
            "rows": [
                {"recurrence_weight": weight, "pass": passed}
                for weight, passed in zip(gate.RECURRENCE_WEIGHTS, passes)
            ],
        }

    numpy_path = tmp_path / "numpy.json"
    cupy_path = tmp_path / "cupy.json"
    numpy_path.write_text(json.dumps(artifact("numpy", [False, True, True, True])))
    cupy_path.write_text(json.dumps(artifact("cupy", [False, False, True, True])))

    merged = gate.merge_construction_artifacts(numpy_path, cupy_path)

    assert merged["construction_go"] is True
    assert merged["selected_recurrence_weight"] == 1.0
    assert merged["formal_execution_open"] is False
    assert merged["shared_pass_by_weight"] == [
        {"recurrence_weight": 0.25, "numpy_pass": False, "cupy_pass": False},
        {"recurrence_weight": 0.5, "numpy_pass": True, "cupy_pass": False},
        {"recurrence_weight": 1.0, "numpy_pass": True, "cupy_pass": True},
        {"recurrence_weight": 2.0, "numpy_pass": True, "cupy_pass": True},
    ]
