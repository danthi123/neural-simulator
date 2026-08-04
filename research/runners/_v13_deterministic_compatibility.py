"""Execute and merge the preregistered V13 deterministic compatibility audit."""
from __future__ import annotations

import argparse
import base64
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import uuid

import numpy as np

from research.runners._vocal_action_selector_gate import (
    _indices, _set_equal_tonic_current, build_selector_bridge, selector_config,
)
from sim.backend import get_backend, synchronize, to_host
from sim.regions import BrainRegion
from tools.lab import assert_backend
from tools.verdict import Verdict


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = Path(__file__).resolve()
SPEC_PATH = ROOT / "research/specs/v13_tonic_output_deterministic_compatibility.json"
PREREG_PATH = ROOT / (
    "research/findings/2026-08-04-neural-vocal-credit-gateB-v13-"
    "deterministic-compatibility-correction-PREREGISTRATION.md"
)
BASELINE_REVISION = "8994b5102"
DETERMINISTIC_PATCH_ID = "18bd23624a3247cb0f205795081b7a540c15ed89"
SOURCE_TWINS = ("baseline_8994_plus_deterministic_patch", "candidate_v13")
SEEDS = (271829, 271831, 271837, 271843, 271849, 271853)
BACKENDS = ("numpy", "cupy")
REPEATS = (1, 2, 3)
STEPS, POPULATION_COUNT, PRACTICE_DRIVE_PA = 300, 600, 250.0
HASH_NAMES = ("raster", "v", "u", "g_e", "g_i", "weights", "external")
RUNTIME_DIRTY_PATHS = {
    "research/findings/raw/_provenance/runs.jsonl",
    "research/queue/.corpus_checks.jsonl",
}
OUTPUT_ROOT = ROOT / "research/findings/raw/v13_deterministic_compatibility"
BASELINE_ALLOWED_PATHS = {
    "sim/bridge.py",
    "tests/test_deterministic_sparse_matvec.py",
    "research/runners/_v13_deterministic_compatibility.py",
    "tests/test_v13_deterministic_compatibility.py",
    str(SPEC_PATH.relative_to(ROOT)),
    str(PREREG_PATH.relative_to(ROOT)),
}


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_array(value) -> str:
    return _sha(np.ascontiguousarray(np.asarray(to_host(value))).tobytes())


def _git(*args, input_bytes=None, check=True):
    return subprocess.run(
        ["git", *args], cwd=ROOT, input=input_bytes, check=check,
        capture_output=True, text=input_bytes is None,
    )


def load_locked_spec() -> dict:
    spec = json.loads(SPEC_PATH.read_text())
    checks = {
        "id": spec.get("id") == "gateB-v13-deterministic-default-off-compatibility",
        "status": spec.get("status") == "preregistered",
        "baseline": spec.get("baseline_base_revision") == BASELINE_REVISION,
        "partitions": spec.get("partitions") == {"compatibility": list(SEEDS)},
        "backends": spec.get("backends") == list(BACKENDS),
        "twins": spec.get("source_twins") == list(SOURCE_TWINS),
        "repeats": spec.get("separate_process_repetitions") == 3,
        "shape": spec.get("steps") == STEPS and spec.get("population_count") == POPULATION_COUNT,
        "drive": spec.get("practice_drive_pA") == PRACTICE_DRIVE_PA,
        "flags": spec.get("required_core_config") == {
            "deterministic_transpose_matvec": True, "enable_nmda": False,
        },
        "hashes": spec.get("required_hashes") == list(HASH_NAMES),
    }
    if not all(checks.values()):
        raise ValueError(f"runner/spec disagreement: {checks}")
    return spec


def _source_manifest() -> dict:
    raw = _git("ls-tree", "-r", "--full-tree", "HEAD").stdout.encode()
    lines = [line for line in raw.splitlines() if line]
    suffixes = (".py", ".sh", ".json", ".yaml", ".yml", ".toml")
    execution = [
        line for line in lines
        if (lambda path: path.endswith(suffixes) or Path(path).name.startswith("requirements"))(
            line.split(b"\t", 1)[1].decode()
        )
    ]
    execution_raw = b"\n".join(execution) + (b"\n" if execution else b"")
    return {
        "format": "git-ls-tree-r-v1",
        "git_tree_sha": _git("rev-parse", "HEAD^{tree}").stdout.strip(),
        "complete_tree_manifest_sha256": _sha(raw),
        "complete_tree_entry_count": len(lines),
        "execution_manifest_sha256": _sha(execution_raw),
        "execution_entry_count": len(execution),
        "runner_sha256": _sha(RUNNER_PATH.read_bytes()),
        "spec_sha256": _sha(SPEC_PATH.read_bytes()),
        "preregistration_sha256": _sha(PREREG_PATH.read_bytes()),
    }


def _artifact_owner(path: Path, manifest: dict) -> bool:
    try:
        artifact = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    schema = artifact.get("schema")
    if schema == "v13-deterministic-compatibility-cell-v1":
        owner = artifact.get("source_seal", {}).get("manifest", {})
    elif schema == "v13-deterministic-compatibility-source-bundle-v1":
        owner = artifact.get("source_identity", {})
    elif schema == "v13-deterministic-compatibility-comparison-v1":
        owner = artifact.get("executor_identity", {})
    else:
        return False
    return (
        owner.get("runner_sha256") == manifest["runner_sha256"]
        and owner.get("spec_sha256") == manifest["spec_sha256"]
    )


def _allowed_runtime_output(path: str, output_root: Path, manifest: dict) -> bool:
    try:
        relative = (ROOT / path).resolve().relative_to(output_root.resolve())
    except ValueError:
        return False
    if len(relative.parts) != 1:
        return False
    name = relative.name
    artifact_name = name[:-10] if name.endswith(".prov.json") else name
    patterns = (
        rf"cell-({'|'.join(map(re.escape, SOURCE_TWINS))})-seed({'|'.join(map(str, SEEDS))})-"
        rf"({'|'.join(BACKENDS)})-repeat[123]\.json",
        rf"bundle-({'|'.join(map(re.escape, SOURCE_TWINS))})\.json",
        r"comparison-baseline-vs-candidate\.json",
    )
    if not any(re.fullmatch(pattern, artifact_name) for pattern in patterns):
        return False
    artifact_path = output_root / artifact_name
    if not _artifact_owner(artifact_path, manifest):
        return False
    if name.endswith(".prov.json"):
        try:
            sidecar = json.loads((output_root / name).read_text())
        except (OSError, ValueError):
            return False
        return str(sidecar.get("runner", "")).endswith(
            "research/runners/_v13_deterministic_compatibility.py"
        )
    return True


def _source_seal(output_root: Path = OUTPUT_ROOT) -> dict:
    if output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError(f"output root must be the locked directory {OUTPUT_ROOT}")
    manifest = _source_manifest()
    dirty, ignored = [], []
    for line in _git("status", "--porcelain", "--untracked-files=all").stdout.splitlines():
        path = line[3:].split(" -> ")[-1]
        row = {"status": line[:2], "path": path}
        if path in RUNTIME_DIRTY_PATHS or _allowed_runtime_output(path, output_root, manifest):
            ignored.append(row)
        else:
            dirty.append(row)
    return {
        "clean_execution_inputs": not dirty,
        "dirty_inputs": dirty,
        "ignored_runtime_state": ignored,
        "output_root": str(output_root),
        "manifest": manifest,
    }


def _stable_patch(commit: str) -> dict:
    resolved = _git("rev-parse", f"{commit}^{{commit}}").stdout.strip()
    patch = _git("show", "--format=", "--binary", resolved).stdout.encode()
    patch_id = _git("patch-id", "--stable", input_bytes=patch).stdout.decode().split()
    if not patch_id:
        raise ValueError(f"commit {commit} has no stable patch-id")
    changed = _git("diff-tree", "--no-commit-id", "--name-only", "-r", resolved).stdout.splitlines()
    return {
        "commit": resolved,
        "stable_patch_id": patch_id[0],
        "is_ancestor_of_head": _git(
            "merge-base", "--is-ancestor", resolved, "HEAD", check=False
        ).returncode == 0,
        "changed_paths": sorted(changed),
        "touches_sim_bridge": "sim/bridge.py" in changed,
    }


def _source_twin_contract(source_twin: str) -> dict:
    if source_twin not in SOURCE_TWINS:
        raise ValueError(f"unknown source twin: {source_twin}")
    has_field = "intrinsic_current_pA" in {item.name for item in fields(BrainRegion)}
    if source_twin == SOURCE_TWINS[0]:
        ancestor = _git(
            "merge-base", "--is-ancestor", BASELINE_REVISION, "HEAD", check=False
        ).returncode == 0
        changed = set(_git("diff", "--name-only", f"{BASELINE_REVISION}..HEAD").stdout.splitlines())
        unexpected = sorted(changed - BASELINE_ALLOWED_PATHS)
        valid = ancestor and not has_field and not unexpected
    else:
        ancestor, changed, unexpected, valid = None, None, [], has_field
    return {
        "source_twin": source_twin,
        "base_revision": BASELINE_REVISION if source_twin == SOURCE_TWINS[0] else None,
        "base_is_ancestor": ancestor,
        "brain_region_has_intrinsic_field": has_field,
        "changed_paths_from_base": sorted(changed) if changed is not None else None,
        "unexpected_paths_from_base": unexpected,
        "contract_valid": valid,
    }


def _backend_info(requested: str) -> dict:
    if os.environ.get("SIM_BACKEND") != requested or requested not in BACKENDS:
        raise ValueError("--backend must agree with explicit SIM_BACKEND=numpy|cupy")
    assert_backend(requested, note="V13 deterministic compatibility correction")
    xp, actual = get_backend()
    if actual != requested:
        raise RuntimeError(f"requested {requested}, resolved {actual}")
    result = {
        "backend": actual, "host": platform.node(), "python": sys.version,
        "platform": platform.platform(), "numpy_version": np.__version__,
        "device": "CPU (NumPy backend)",
    }
    if actual == "cupy":
        props = xp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        result.update({
            "device": name.decode() if isinstance(name, bytes) else str(name),
            "cupy_version": xp.__version__,
            "cuda_runtime_version": int(xp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": int(xp.cuda.runtime.driverGetVersion()),
            "compute_capability": [int(props["major"]), int(props["minor"])],
            "total_global_memory_bytes": int(props["totalGlobalMem"]),
        })
    return result


def _topology(bridge) -> dict:
    matrix, layout, cursor = bridge.cp_connections, [], 0
    for region in bridge.core_config.brain_regions:
        count = int(region.n_neurons)
        layout.append({"name": region.name, "start": cursor, "stop": cursor + count})
        cursor += count
    result = {
        "shape": [int(value) for value in matrix.shape], "nnz": int(matrix.nnz),
        "indptr_sha256": _hash_array(matrix.indptr),
        "indices_sha256": _hash_array(matrix.indices), "region_layout": layout,
    }
    result["topology_sha256"] = _sha(_canonical(result))
    return result


def _intrinsic_state(bridge) -> dict:
    sentinel = object()
    value = getattr(bridge, "cp_intrinsic_current_pA", sentinel)
    return {
        "bridge_attribute_exists": value is not sentinel,
        "bridge_value_is_none": value is None,
        "bridge_state": "attribute_absent" if value is sentinel else (
            "none" if value is None else "allocated"
        ),
    }


def _pack_raster(raster) -> dict:
    raster = np.ascontiguousarray(raster, dtype=bool)
    packed = np.packbits(raster.reshape(-1), bitorder="little")
    return {
        "shape": list(raster.shape), "dtype": "bool",
        "encoding": "numpy-packbits-little-base64", "bit_count": int(raster.size),
        "packed_base64": base64.b64encode(packed.tobytes()).decode(),
    }


def _verdict(label: str, go: bool, requirements: dict[str, bool]) -> dict:
    earned = Verdict(label)
    for name, measured in requirements.items():
        earned.require(name, measured, expect=True)
    result = earned.decide(go=go, verbose=False)
    return {
        "verdict_status": result["status"], "go": result["go"],
        "preconditions": result["preconditions"],
        "undefined_reasons": result["undefined_reasons"],
        "disabled_processes": result["disabled_processes"],
    }


def _earned_go(artifact) -> bool:
    checks = artifact.get("preconditions")
    return bool(
        artifact.get("verdict_status") == "GO" and artifact.get("go") is True
        and isinstance(checks, list) and checks and all(row.get("ok") is True for row in checks)
        and not artifact.get("undefined_reasons")
    )


def _apply_drive(bridge, config, xp):
    _set_equal_tonic_current(bridge, config)
    bridge.cp_external_input_current[xp.asarray(_indices(bridge, "practice_arousal"))] = (
        xp.float32(PRACTICE_DRIVE_PA)
    )


def execute_cell(*, source_twin: str, seed: int, backend: str, repeat: int,
                 deterministic_patch_commit: str, output_root: Path = OUTPUT_ROOT) -> dict:
    load_locked_spec()
    if seed not in SEEDS:
        raise ValueError("seed is outside the six locked correction seeds")
    if backend not in BACKENDS or repeat not in REPEATS:
        raise ValueError("backend or repeat is outside the locked matrix")
    process = {
        "host": platform.node(), "pid": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "invocation_uuid": str(uuid.uuid4()),
    }
    seal, patch = _source_seal(output_root), _stable_patch(deterministic_patch_commit)
    twin, runtime = _source_twin_contract(source_twin), _backend_info(backend)
    config = selector_config("v2")
    bridge = build_selector_bridge(
        seed, config, commit_enable_nmda=False,
        core_config_updates={"enable_nmda": False, "deterministic_transpose_matvec": True},
    )
    intrinsic = _intrinsic_state(bridge)
    if source_twin == SOURCE_TWINS[0]:
        intrinsic_ok = (
            not twin["brain_region_has_intrinsic_field"]
            and intrinsic["bridge_state"] in ("attribute_absent", "none")
        )
    else:
        intrinsic_ok = (
            twin["brain_region_has_intrinsic_field"]
            and intrinsic["bridge_attribute_exists"] and intrinsic["bridge_value_is_none"]
        )
    deterministic = bool(getattr(bridge.core_config, "deterministic_transpose_matvec", False))
    initial_topology, initial_weight = _topology(bridge), _hash_array(bridge.cp_connections.data)
    xp, actual_backend = get_backend()
    _apply_drive(bridge, config, xp)
    initial_external = _hash_array(bridge.cp_external_input_current)
    practice = np.asarray(_indices(bridge, "practice_arousal"), dtype=np.int64)
    external = np.asarray(to_host(bridge.cp_external_input_current))
    drive_exact = bool(practice.size and np.all(external[practice] == np.float32(PRACTICE_DRIVE_PA)))
    raster = np.zeros((STEPS, bridge.core_config.num_neurons), dtype=bool)
    for step in range(STEPS):
        _apply_drive(bridge, config, xp)
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
        raster[step] = np.asarray(to_host(bridge.cp_firing_states), dtype=bool)
    synchronize()
    final_topology = _topology(bridge)
    hashes = {
        "raster": _hash_array(raster), "v": _hash_array(bridge.cp_membrane_potential_v),
        "u": _hash_array(bridge.cp_recovery_variable_u),
        "g_e": _hash_array(bridge.cp_conductance_g_e),
        "g_i": _hash_array(bridge.cp_conductance_g_i),
        "weights": _hash_array(bridge.cp_connections.data),
        "external": _hash_array(bridge.cp_external_input_current),
    }
    requirements = {
        "source execution inputs sealed": seal["clean_execution_inputs"],
        "deterministic patch is in source ancestry": patch["is_ancestor_of_head"],
        "deterministic patch touches sim bridge": patch["touches_sim_bridge"],
        "deterministic patch identity locked": (
            patch["stable_patch_id"] == DETERMINISTIC_PATCH_ID
        ),
        "source twin contract holds": twin["contract_valid"],
        "explicit backend resolved": actual_backend == backend,
        "RTX 3090 used for CuPy": backend != "cupy" or "3090" in runtime["device"],
        "deterministic flag read back true": deterministic,
        "intrinsic default state matches twin": intrinsic_ok,
        "complete 300x600 raster captured": raster.shape == (STEPS, POPULATION_COUNT),
        "topology immutable": initial_topology == final_topology,
        "weights immutable": initial_weight == hashes["weights"],
        "external current exact and stable": drive_exact and initial_external == hashes["external"],
        "all seven hashes recorded": set(hashes) == set(HASH_NAMES),
    }
    verdict = _verdict("V13 deterministic compatibility cell", all(requirements.values()), requirements)
    artifact = {
        "schema": "v13-deterministic-compatibility-cell-v1", "stage": "cell",
        "source_twin": source_twin, "seed": seed, "backend": backend, "repeat": repeat,
        "steps": STEPS, "population_count": POPULATION_COUNT,
        "practice_drive_pA": PRACTICE_DRIVE_PA, "process": process, "runtime": runtime,
        "git_sha": _git("rev-parse", "HEAD").stdout.strip(), "source_seal": seal,
        "deterministic_patch": patch, "source_twin_contract": twin,
        "deterministic_flag_readback": deterministic, "intrinsic_default_state": intrinsic,
        "topology_initial": initial_topology, "topology_final": final_topology,
        "initial_weight_sha256": initial_weight, "final_weight_sha256": hashes["weights"],
        "initial_external_sha256": initial_external, "final_external_sha256": hashes["external"],
        "raster": _pack_raster(raster), "hashes": hashes, **verdict,
    }
    artifact["outcome"] = "CELL_GO" if artifact["go"] else (
        "CELL_UNDEFINED" if artifact["verdict_status"] == "UNDEFINED" else "CELL_NO_GO"
    )
    bridge.clear_simulation_state_and_gpu_memory()
    return artifact


def _artifact_digest(artifact) -> str:
    return _sha(_canonical(artifact))


def _source_identity(artifact) -> dict:
    manifest = artifact["source_seal"]["manifest"]
    return {
        "source_twin": artifact["source_twin"], "git_sha": artifact["git_sha"],
        **{key: manifest[key] for key in (
            "complete_tree_manifest_sha256", "execution_manifest_sha256",
            "runner_sha256", "spec_sha256", "preregistration_sha256",
        )},
        "deterministic_patch_id": artifact["deterministic_patch"]["stable_patch_id"],
    }


def merge_source_twin(paths: list[Path], *, source_twin: str) -> dict:
    load_locked_spec()
    expected = {(s, b, r) for s in SEEDS for b in BACKENDS for r in REPEATS}
    if source_twin not in SOURCE_TWINS or len(paths) != len(expected):
        raise ValueError("merge requires one known source twin and exactly 36 artifacts")
    artifacts = [json.loads(path.read_text()) for path in paths]
    by_key, source = {}, None
    for path, artifact in zip(paths, artifacts):
        if artifact.get("stage") != "cell" or artifact.get("source_twin") != source_twin:
            raise ValueError(f"wrong cell stage/source twin: {path}")
        if not _earned_go(artifact):
            raise ValueError(f"unearned cell artifact: {path}")
        key = (artifact.get("seed"), artifact.get("backend"), artifact.get("repeat"))
        if key in by_key:
            raise ValueError(f"duplicate matrix cell: {key}")
        by_key[key] = artifact
        identity = _source_identity(artifact)
        if source is None:
            source = identity
        elif identity != source:
            raise ValueError(f"source identity mismatch: {path}")
    if set(by_key) != expected:
        raise ValueError(f"incomplete matrix; missing={sorted(expected - set(by_key))}")
    groups, hashes_exact, topology_exact, separate = {}, True, True, True
    for seed in SEEDS:
        for backend in BACKENDS:
            rows = [by_key[(seed, backend, repeat)] for repeat in REPEATS]
            hash_rows, topology_rows = [r["hashes"] for r in rows], [r["topology_final"] for r in rows]
            process_rows = {(r["process"]["host"], r["process"]["pid"]) for r in rows}
            nonces = {r["process"]["invocation_uuid"] for r in rows}
            h_ok = all(row == hash_rows[0] for row in hash_rows[1:])
            t_ok = all(row == topology_rows[0] for row in topology_rows[1:])
            p_ok = len(process_rows) == len(REPEATS) and len(nonces) == len(REPEATS)
            hashes_exact, topology_exact, separate = hashes_exact and h_ok, topology_exact and t_ok, separate and p_ok
            groups[f"{seed}:{backend}"] = {
                "hashes": hash_rows[0], "topology_sha256": topology_rows[0]["topology_sha256"],
                "repeat_hashes_exact": h_ok, "repeat_topology_exact": t_ok,
                "separate_processes": p_ok,
                "artifact_sha256": [_artifact_digest(row) for row in rows],
            }
    requirements = {
        "exact 6x2x3 matrix": set(by_key) == expected,
        "all inputs carry earned GO verdicts": all(_earned_go(row) for row in artifacts),
        "one sealed source identity": source is not None,
        "three distinct processes per cell": separate,
    }
    verdict = _verdict(
        "V13 deterministic compatibility source merge",
        hashes_exact and topology_exact and separate, requirements,
    )
    artifact = {
        "schema": "v13-deterministic-compatibility-source-bundle-v1",
        "stage": "source_twin_merge", "source_twin": source_twin,
        "source_identity": source, "source_twin_contract": artifacts[0]["source_twin_contract"],
        "matrix": groups, "input_artifacts": [str(path) for path in paths],
        "acceptance_checks": {
            "topology_exact_within_repeats": topology_exact,
            "all_seven_hashes_exact_within_repeats": hashes_exact,
        },
        **verdict,
    }
    artifact["outcome"] = "DETERMINISM_GO" if artifact["go"] else (
        "DETERMINISM_UNDEFINED" if artifact["verdict_status"] == "UNDEFINED" else "DETERMINISM_NO_GO"
    )
    return artifact


def compare_source_twins(baseline_path: Path, candidate_path: Path) -> dict:
    load_locked_spec()
    baseline, candidate = json.loads(baseline_path.read_text()), json.loads(candidate_path.read_text())
    for label, artifact, twin in (
        ("baseline", baseline, SOURCE_TWINS[0]), ("candidate", candidate, SOURCE_TWINS[1]),
    ):
        if artifact.get("stage") != "source_twin_merge" or artifact.get("source_twin") != twin:
            raise ValueError(f"{label} is not the required source-twin bundle")
        if not _earned_go(artifact):
            raise ValueError(f"{label} bundle lacks an earned determinism GO")
    expected = {f"{seed}:{backend}" for seed in SEEDS for backend in BACKENDS}
    if set(baseline["matrix"]) != expected or set(candidate["matrix"]) != expected:
        raise ValueError("source bundles do not contain the exact 6x2 matrix")
    left_id, right_id = baseline["source_identity"], candidate["source_identity"]
    comparisons, hashes_exact, topology_exact = {}, True, True
    for key in sorted(expected):
        left, right = baseline["matrix"][key], candidate["matrix"][key]
        equal = {name: left["hashes"][name] == right["hashes"][name] for name in HASH_NAMES}
        topology = left["topology_sha256"] == right["topology_sha256"]
        hashes_exact, topology_exact = hashes_exact and all(equal.values()), topology_exact and topology
        comparisons[key] = {"hash_equal": equal, "all_seven_hashes_exact": all(equal.values()),
                            "topology_exact": topology}
    left_contract, right_contract = baseline["source_twin_contract"], candidate["source_twin_contract"]
    requirements = {
        "baseline bundle earned determinism GO": _earned_go(baseline),
        "candidate bundle earned determinism GO": _earned_go(candidate),
        "baseline proves intrinsic field absent": (
            left_contract.get("contract_valid") is True
            and left_contract.get("brain_region_has_intrinsic_field") is False
        ),
        "candidate proves intrinsic field present": (
            right_contract.get("contract_valid") is True
            and right_contract.get("brain_region_has_intrinsic_field") is True
        ),
        "identical deterministic correction patch": (
            left_id["deterministic_patch_id"] == right_id["deterministic_patch_id"]
        ),
        "identical executor": left_id["runner_sha256"] == right_id["runner_sha256"],
        "identical locked spec": left_id["spec_sha256"] == right_id["spec_sha256"],
        "identical preregistration": left_id["preregistration_sha256"] == right_id["preregistration_sha256"],
    }
    verdict = _verdict(
        "V13 deterministic default-off compatibility correction",
        hashes_exact and topology_exact, requirements,
    )
    artifact = {
        "schema": "v13-deterministic-compatibility-comparison-v1",
        "stage": "cross_twin_compare", "baseline_bundle": str(baseline_path),
        "candidate_bundle": str(candidate_path), "baseline_bundle_sha256": _artifact_digest(baseline),
        "candidate_bundle_sha256": _artifact_digest(candidate),
        "deterministic_patch_id": left_id["deterministic_patch_id"],
        "executor_identity": {
            "runner_sha256": _sha(RUNNER_PATH.read_bytes()),
            "spec_sha256": _sha(SPEC_PATH.read_bytes()),
        },
        "comparisons": comparisons,
        "acceptance_checks": {
            "topology_exact_across_twins": topology_exact,
            "all_seven_hashes_exact_across_twins": hashes_exact,
        },
        **verdict,
    }
    artifact["outcome"] = "DETERMINISTIC_COMPATIBILITY_GO" if artifact["go"] else (
        "DETERMINISTIC_COMPATIBILITY_UNDEFINED"
        if artifact["verdict_status"] == "UNDEFINED" else "COMPATIBILITY_NO_GO"
    )
    return artifact


def _write_json_create(path: Path, artifact: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _require_output_path(path: Path, output_root: Path, expected_name: str):
    if output_root.resolve() != OUTPUT_ROOT.resolve():
        raise ValueError(f"output root must be the locked directory {OUTPUT_ROOT}")
    if path.resolve() != (output_root / expected_name).resolve():
        raise ValueError(f"output must be {output_root / expected_name}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    cell = sub.add_parser("cell")
    cell.add_argument("--source-twin", required=True, choices=SOURCE_TWINS)
    cell.add_argument("--seed", required=True, type=int, choices=SEEDS)
    cell.add_argument("--backend", required=True, choices=BACKENDS)
    cell.add_argument("--repeat", required=True, type=int, choices=REPEATS)
    cell.add_argument("--deterministic-patch-commit", required=True)
    cell.add_argument("--output-root", required=True, type=Path)
    cell.add_argument("--out", required=True, type=Path)
    merge = sub.add_parser("merge")
    merge.add_argument("--source-twin", required=True, choices=SOURCE_TWINS)
    merge.add_argument("--inputs", required=True, nargs="+", type=Path)
    merge.add_argument("--output-root", required=True, type=Path)
    merge.add_argument("--out", required=True, type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("--baseline", required=True, type=Path)
    compare.add_argument("--candidate", required=True, type=Path)
    compare.add_argument("--output-root", required=True, type=Path)
    compare.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.mode == "cell":
        expected = (
            f"cell-{args.source_twin}-seed{args.seed}-{args.backend}-repeat{args.repeat}.json"
        )
        _require_output_path(args.out, args.output_root, expected)
        artifact = execute_cell(
            source_twin=args.source_twin, seed=args.seed, backend=args.backend,
            repeat=args.repeat, deterministic_patch_commit=args.deterministic_patch_commit,
            output_root=args.output_root,
        )
    elif args.mode == "merge":
        _require_output_path(
            args.out, args.output_root, f"bundle-{args.source_twin}.json"
        )
        artifact = merge_source_twin(args.inputs, source_twin=args.source_twin)
    else:
        _require_output_path(
            args.out, args.output_root, "comparison-baseline-vs-candidate.json"
        )
        artifact = compare_source_twins(args.baseline, args.candidate)
    _write_json_create(args.out, artifact)
    print(json.dumps({"out": str(args.out), "outcome": artifact["outcome"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
