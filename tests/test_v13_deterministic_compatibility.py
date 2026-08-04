import base64
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import research.runners._v13_deterministic_compatibility as gate


def _identity(twin):
    return {
        "source_twin": twin,
        "git_sha": "source-sha",
        "complete_tree_manifest_sha256": "tree",
        "execution_manifest_sha256": "execution",
        "runner_sha256": "runner",
        "spec_sha256": "spec",
        "preregistration_sha256": "prereg",
        "deterministic_patch_id": "patch",
    }


def _contract(twin):
    baseline = twin == gate.SOURCE_TWINS[0]
    return {
        "source_twin": twin,
        "contract_valid": True,
        "brain_region_has_intrinsic_field": not baseline,
    }


def _cell(twin, seed, backend, repeat):
    identity = _identity(twin)
    hashes = {
        name: gate._sha(f"{seed}:{backend}:{name}".encode())
        for name in gate.HASH_NAMES
    }
    verdict = gate._verdict("synthetic cell", True, {"instrument valid": True})
    return {
        "schema": "v13-deterministic-compatibility-cell-v1",
        "stage": "cell",
        "source_twin": twin,
        "seed": seed,
        "backend": backend,
        "repeat": repeat,
        "git_sha": identity["git_sha"],
        "source_seal": {"manifest": {
            key: identity[key] for key in (
                "complete_tree_manifest_sha256", "execution_manifest_sha256",
                "runner_sha256", "spec_sha256", "preregistration_sha256",
            )
        }},
        "deterministic_patch": {"stable_patch_id": identity["deterministic_patch_id"]},
        "source_twin_contract": _contract(twin),
        "topology_final": {"topology_sha256": gate._sha(f"{seed}:{backend}:topology".encode())},
        "hashes": hashes,
        "process": {
            "host": "host",
            "pid": seed * 100 + (0 if backend == "numpy" else 10) + repeat,
            "invocation_uuid": f"{seed}-{backend}-{repeat}",
        },
        **verdict,
    }


def _matrix(tmp_path, twin):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in gate.SEEDS:
        for backend in gate.BACKENDS:
            for repeat in gate.REPEATS:
                path = tmp_path / f"{twin}-{seed}-{backend}-{repeat}.json"
                path.write_text(json.dumps(_cell(twin, seed, backend, repeat)))
                paths.append(path)
    return paths


def _write_bundle(tmp_path, twin):
    paths = _matrix(tmp_path / twin, twin)
    bundle = gate.merge_source_twin(paths, source_twin=twin)
    path = tmp_path / f"bundle-{twin}.json"
    path.write_text(json.dumps(bundle))
    return path, bundle


def test_locked_spec_and_patch_identity():
    assert gate.load_locked_spec()["partitions"] == {"compatibility": list(gate.SEEDS)}
    patch = gate._stable_patch("78bfb8617")
    assert patch["stable_patch_id"] == "18bd23624a3247cb0f205795081b7a540c15ed89"
    assert patch["touches_sim_bridge"] is True


def test_cell_orchestration_captures_complete_raster(monkeypatch):
    class Bridge:
        def __init__(self):
            regions = [SimpleNamespace(name="all", n_neurons=600)]
            self.core_config = SimpleNamespace(
                num_neurons=600, brain_regions=regions,
                deterministic_transpose_matvec=True, dt_ms=1.0,
            )
            self.runtime_state = SimpleNamespace(current_time_ms=0.0)
            self.cp_connections = csr_matrix(np.eye(600, dtype=np.float32))
            self.cp_external_input_current = np.zeros(600, dtype=np.float32)
            self.cp_firing_states = np.zeros(600, dtype=bool)
            self.cp_membrane_potential_v = np.arange(600, dtype=np.float32)
            self.cp_recovery_variable_u = np.arange(600, dtype=np.float32) * 2
            self.cp_conductance_g_e = np.zeros(600, dtype=np.float32)
            self.cp_conductance_g_i = np.zeros(600, dtype=np.float32)
            self.cp_intrinsic_current_pA = None
            self.step = 0

        def _run_one_simulation_step(self):
            self.cp_firing_states[:] = False
            self.cp_firing_states[self.step % 600] = True
            self.step += 1

        def clear_simulation_state_and_gpu_memory(self):
            pass

    manifest = {
        "runner_sha256": "runner", "spec_sha256": "spec",
        "preregistration_sha256": "prereg",
        "complete_tree_manifest_sha256": "tree", "execution_manifest_sha256": "execution",
    }
    monkeypatch.setattr(gate, "_source_seal", lambda root: {
        "clean_execution_inputs": True, "manifest": manifest,
    })
    monkeypatch.setattr(gate, "_stable_patch", lambda commit: {
        "stable_patch_id": gate.DETERMINISTIC_PATCH_ID, "is_ancestor_of_head": True,
        "touches_sim_bridge": True,
    })
    monkeypatch.setattr(gate, "_source_twin_contract", lambda twin: _contract(twin))
    monkeypatch.setattr(gate, "_backend_info", lambda backend: {
        "backend": backend, "device": "CPU (NumPy backend)",
    })
    monkeypatch.setattr(gate, "build_selector_bridge", lambda *args, **kwargs: Bridge())
    monkeypatch.setattr(gate, "_indices", lambda bridge, name: np.arange(4))
    monkeypatch.setattr(
        gate, "_set_equal_tonic_current",
        lambda bridge, config: bridge.cp_external_input_current.fill(0),
    )
    monkeypatch.setattr(gate, "get_backend", lambda: (np, "numpy"))
    monkeypatch.setattr(gate, "synchronize", lambda: None)
    monkeypatch.setenv("SIM_BACKEND", "numpy")
    result = gate.execute_cell(
        source_twin=gate.SOURCE_TWINS[1], seed=gate.SEEDS[0], backend="numpy",
        repeat=1, deterministic_patch_commit="patch", output_root=gate.OUTPUT_ROOT,
    )
    assert result["verdict_status"] == "GO"
    assert result["raster"]["shape"] == [300, 600]
    packed = base64.b64decode(result["raster"]["packed_base64"])
    assert len(packed) == 300 * 600 // 8
    assert set(result["hashes"]) == set(gate.HASH_NAMES)


def test_merge_exact_matrix_is_earned_go(tmp_path):
    paths = _matrix(tmp_path, gate.SOURCE_TWINS[1])
    result = gate.merge_source_twin(paths, source_twin=gate.SOURCE_TWINS[1])
    assert result["outcome"] == "DETERMINISM_GO"
    assert all(row["ok"] is True for row in result["preconditions"])


def test_merge_rejects_missing_duplicate_source_mismatch_and_unearned(tmp_path):
    paths = _matrix(tmp_path, gate.SOURCE_TWINS[1])
    with pytest.raises(ValueError, match="exactly 36"):
        gate.merge_source_twin(paths[:-1], source_twin=gate.SOURCE_TWINS[1])
    with pytest.raises(ValueError, match="duplicate"):
        gate.merge_source_twin(paths[:-1] + [paths[0]], source_twin=gate.SOURCE_TWINS[1])

    changed = json.loads(paths[0].read_text())
    changed["git_sha"] = "other-source"
    paths[0].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="source identity mismatch"):
        gate.merge_source_twin(paths, source_twin=gate.SOURCE_TWINS[1])

    changed["git_sha"] = "source-sha"
    changed.update(gate._verdict("bad cell", True, {"instrument valid": False}))
    paths[0].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="unearned"):
        gate.merge_source_twin(paths, source_twin=gate.SOURCE_TWINS[1])


def test_repeat_mismatch_is_earned_no_go(tmp_path):
    paths = _matrix(tmp_path, gate.SOURCE_TWINS[1])
    changed = json.loads(paths[0].read_text())
    changed["hashes"]["v"] = "mismatch"
    paths[0].write_text(json.dumps(changed))
    result = gate.merge_source_twin(paths, source_twin=gate.SOURCE_TWINS[1])
    assert result["outcome"] == "DETERMINISM_NO_GO"
    assert result["verdict_status"] == "NO-GO"
    assert all(row["ok"] is True for row in result["preconditions"])
    assert result["acceptance_checks"]["all_seven_hashes_exact_within_repeats"] is False


def test_compare_exact_bundles_go_and_hash_mismatch_is_earned_no_go(tmp_path):
    baseline_path, _ = _write_bundle(tmp_path, gate.SOURCE_TWINS[0])
    candidate_path, candidate = _write_bundle(tmp_path, gate.SOURCE_TWINS[1])
    result = gate.compare_source_twins(baseline_path, candidate_path)
    assert result["outcome"] == "DETERMINISTIC_COMPATIBILITY_GO"

    candidate["matrix"][f"{gate.SEEDS[0]}:numpy"]["hashes"]["u"] = "different"
    candidate_path.write_text(json.dumps(candidate))
    result = gate.compare_source_twins(baseline_path, candidate_path)
    assert result["outcome"] == "COMPATIBILITY_NO_GO"
    assert result["verdict_status"] == "NO-GO"
    assert all(row["ok"] is True for row in result["preconditions"])
    assert result["acceptance_checks"]["all_seven_hashes_exact_across_twins"] is False


def test_sequential_owned_outputs_allowed_but_other_dirt_fails(tmp_path, monkeypatch):
    root = tmp_path
    output_root = root / "research/findings/raw/v13_deterministic_compatibility"
    output_root.mkdir(parents=True)
    manifest = {
        "runner_sha256": "runner", "spec_sha256": "spec",
        "preregistration_sha256": "prereg", "complete_tree_manifest_sha256": "tree",
        "execution_manifest_sha256": "execution",
    }
    monkeypatch.setattr(gate, "ROOT", root)
    monkeypatch.setattr(gate, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(gate, "_source_manifest", lambda: manifest)

    names = []
    for repeat in (1, 2):
        name = f"cell-{gate.SOURCE_TWINS[1]}-seed{gate.SEEDS[0]}-numpy-repeat{repeat}.json"
        artifact = {
            "schema": "v13-deterministic-compatibility-cell-v1",
            "source_seal": {"manifest": manifest},
        }
        (output_root / name).write_text(json.dumps(artifact))
        (output_root / f"{name}.prov.json").write_text(json.dumps({
            "runner": "research/runners/_v13_deterministic_compatibility.py",
        }))
        names.extend([name, f"{name}.prov.json"])
        status = "\n".join(
            f"?? research/findings/raw/v13_deterministic_compatibility/{item}"
            for item in names
        )
        monkeypatch.setattr(gate, "_git", lambda *args, **kwargs: SimpleNamespace(stdout=status))
        assert gate._source_seal(output_root)["clean_execution_inputs"] is True

    status += "\n?? research/findings/raw/arbitrary.json\n?? sim/new_source.py\n?? config.json"
    monkeypatch.setattr(gate, "_git", lambda *args, **kwargs: SimpleNamespace(stdout=status))
    seal = gate._source_seal(output_root)
    assert seal["clean_execution_inputs"] is False
    assert {row["path"] for row in seal["dirty_inputs"]} == {
        "research/findings/raw/arbitrary.json", "sim/new_source.py", "config.json",
    }


def test_create_only_output_refuses_overwrite(tmp_path):
    path = tmp_path / "artifact.json"
    gate._write_json_create(path, {"value": 1})
    with pytest.raises(FileExistsError):
        gate._write_json_create(path, {"value": 2})
