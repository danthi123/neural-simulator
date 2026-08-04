import json
import os

import pytest


os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import _vocal_action_credit_gate_v12_disinhibitory as gate


def test_v12_seed_boundary_keeps_engagement_sealed():
    assert gate.validate_construction_seed(gate.CONSTRUCTION_SEED) == 997
    with pytest.raises(ValueError, match="reserved seed 2 remains sealed"):
        gate.validate_construction_seed(gate.RESERVED_SEED)
    with pytest.raises(ValueError, match="engagement execution is sealed"):
        gate.validate_engagement_seed(gate.RESERVED_SEED)


def test_v12_topology_is_exact_symmetric_and_fast_receptor_only():
    bridge, handles = gate.build_v12_bridge(gate.CONSTRUCTION_SEED)
    audit = gate.boundary_structural_audit(bridge, handles)

    assert bridge.core_config.num_neurons == 728
    assert len(bridge.core_config.region_pathways) == 53
    assert audit["pass"] is True
    assert all(audit["checks"].values())
    assert audit["nmda_regions"] == []
    assert bridge.core_config.enable_nmda is False
    assert bridge.core_config.enable_gabab is False
    assert bridge.core_config.reward_learning_rate == 0.0


def test_v12_lesions_are_fixed_transmission_gates():
    bridge, handles = gate.build_v12_bridge(
        gate.CONSTRUCTION_SEED,
        gate_values={
            gate.GUARD_GATE: 0.0,
            gate.DISINHIBITION_GATE: 0.0,
        },
    )

    assert handles["gate_values"] == {
        gate.BACKGROUND_GATE: 1.0,
        gate.DISINHIBITOR_DRIVE_GATE: 1.0,
        gate.DISINHIBITION_GATE: 0.0,
        gate.GUARD_GATE: 0.0,
        gate.MOTOR_COPY_GATE: 1.0,
        gate.PROPOSAL_STOP_GATE: 1.0,
        gate.COMMIT_STOP_GATE: 1.0,
    }
    assert bridge._transmission_gate_values[gate.GUARD_GATE] == 0.0
    assert bridge._transmission_gate_values[gate.DISINHIBITION_GATE] == 0.0


def test_v12_cross_backend_merge_requires_shared_go(tmp_path):
    def artifact(backend, outcome):
        return {
            "stage": "construction_backend",
            "seed": gate.CONSTRUCTION_SEED,
            "reserved_seed": gate.RESERVED_SEED,
            "engagement_execution_open": False,
            "outcome": outcome,
            "construction_go": outcome == "CONSTRUCTION_GO",
            "backend_info": {"backend": backend},
        }

    numpy_path = tmp_path / "numpy.json"
    cupy_path = tmp_path / "cupy.json"
    numpy_path.write_text(json.dumps(artifact("numpy", "CONSTRUCTION_GO")))
    cupy_path.write_text(json.dumps(artifact("cupy", "CONSTRUCTION_GO")))

    merged = gate.merge_construction_artifacts(numpy_path, cupy_path)

    assert merged["construction_go"] is True
    assert merged["outcome"] == "CONSTRUCTION_GO"
    assert merged["engagement_execution_open"] is False


def test_v12_cross_backend_merge_preserves_shared_short_boundary(tmp_path):
    def artifact(backend):
        return {
            "stage": "construction_backend",
            "seed": gate.CONSTRUCTION_SEED,
            "reserved_seed": gate.RESERVED_SEED,
            "engagement_execution_open": False,
            "outcome": "CONSTRUCTION_QUALIFIED_BOUNDARY_TOO_SHORT",
            "construction_go": False,
            "backend_info": {"backend": backend},
        }

    numpy_path = tmp_path / "numpy.json"
    cupy_path = tmp_path / "cupy.json"
    numpy_path.write_text(json.dumps(artifact("numpy")))
    cupy_path.write_text(json.dumps(artifact("cupy")))

    merged = gate.merge_construction_artifacts(numpy_path, cupy_path)

    assert merged["construction_go"] is False
    assert merged["outcome"] == "CONSTRUCTION_QUALIFIED_BOUNDARY_TOO_SHORT"
