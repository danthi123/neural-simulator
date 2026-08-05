"""Capture source-native NMODL initialization before model transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "v14-source-model-neuron-initialization-oracle-v1"
OUTPUT_SCHEMA = "v14-source-model-neuron-initialization-observation-v1"


class OracleError(RuntimeError):
    """Raised when source-native behavior cannot be authenticated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _semantic_digest(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_bytes({key: item for key, item in value.items() if key != "sha256"}))


def _inside(root: Path, value: str, context: str) -> Path:
    relative = PurePosixPath(value)
    _require(
        isinstance(value, str) and value and not relative.is_absolute()
        and str(relative) == value and all(part not in {"", ".", ".."} for part in relative.parts),
        f"{context} path is invalid",
    )
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise OracleError(f"{context} escapes repository") from exc
    return path


def _load_spec(path: Path, expected_sha256: str, root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    _require(path.is_file() and not path.is_symlink(), "oracle spec is unavailable")
    observed = _sha256_bytes(path.read_bytes())
    _require(observed == expected_sha256, "oracle spec digest mismatch")
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OracleError("oracle spec is not valid JSON") from exc
    _require(isinstance(spec, dict) and spec.get("schema") == SCHEMA, "oracle spec schema mismatch")
    _require(spec.get("status") == "preregistered_not_executed", "oracle spec is not prospective")
    source = spec.get("source")
    _require(isinstance(source, dict), "source binding is missing")
    _require(source.get("file_name") == "rsg.mod", "only the sealed Khaliq source is authorized")
    _require(
        isinstance(source.get("url"), str) and source["url"].startswith("https://raw.githubusercontent.com/"),
        "source URL is invalid",
    )
    _require(
        isinstance(source.get("sha256"), str) and len(source["sha256"]) == 64,
        "source digest is invalid",
    )
    states = spec.get("state_order")
    _require(
        states == ["C1", "C2", "C3", "C4", "C5", "I1", "I2", "I3", "I4", "I5", "O", "B", "I6"],
        "state order differs from the NMODL declaration",
    )
    voltages = spec.get("voltages_mV")
    _require(
        isinstance(voltages, list) and voltages == [-120.0, -100.0, -90.0, -40.0, 0.0, 30.0],
        "oracle voltage ladder changed",
    )
    _require(spec.get("fixed_voltage_step_ms") == 0.001, "oracle step changed")
    runner = spec.get("runner")
    _require(isinstance(runner, dict) and set(runner) == {"path", "sha256"}, "runner binding is invalid")
    runner_path = _inside(root, runner["path"], "runner")
    _require(
        runner_path == Path(__file__).resolve() and _sha256_bytes(runner_path.read_bytes()) == runner["sha256"],
        "loaded runner does not match its sealed binding",
    )
    relative = path.relative_to(root).as_posix()
    return spec, {"path": relative, "sha256": observed}


def _download_source(spec: Mapping[str, Any]) -> bytes:
    source = spec["source"]
    with urlopen(source["url"], timeout=30) as response:  # noqa: S310 - URL is sealed and validated.
        payload = response.read()
    _require(_sha256_bytes(payload) == source["sha256"], "downloaded source digest mismatch")
    return payload


def _compile(source: bytes, directory: Path) -> Path:
    source_dir = directory / "mod"
    source_dir.mkdir()
    (source_dir / "rsg.mod").write_bytes(source)
    nrnivmodl = Path(sys.executable).with_name("nrnivmodl")
    _require(nrnivmodl.is_file(), "nrnivmodl is unavailable beside the selected interpreter")
    completed = subprocess.run(
        [str(nrnivmodl), str(source_dir)], cwd=directory, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    _require(completed.returncode == 0, f"NMODL compilation failed:\n{completed.stdout}")
    library = directory / "x86_64/libnrnmech.so"
    _require(library.is_file(), "compiled mechanism library is missing")
    return library


def _capture(spec: Mapping[str, Any], library: Path) -> tuple[str, list[dict[str, Any]]]:
    try:
        import neuron
        from neuron import h
    except ImportError as exc:
        raise OracleError("the selected interpreter does not provide NEURON") from exc

    h.nrn_load_dll(str(library))
    section = h.Section(name="source_native_khaliq_oracle")
    section.L = 10.0
    section.diam = 10.0
    section.insert("naRsg")
    clamp = h.SEClamp(section(0.5))
    clamp.dur1 = 1.0e9
    clamp.rs = 1.0e-9
    h.dt = float(spec["fixed_voltage_step_ms"])
    names = list(spec["state_order"])
    observations = []
    for voltage in spec["voltages_mV"]:
        clamp.amp1 = float(voltage)
        h.finitialize(float(voltage))
        segment = section(0.5)
        initial = [float(getattr(segment.naRsg, name)) for name in names]
        h.fadvance()
        after = [float(getattr(segment.naRsg, name)) for name in names]
        _require(all(math.isfinite(value) for value in initial + after), "NEURON produced non-finite state")
        observations.append({
            "voltage_mV": float(voltage),
            "voltage_after_step_mV": float(segment.v),
            "initial_state": dict(zip(names, initial)),
            "state_after_step": dict(zip(names, after)),
            "initial_sum": sum(initial),
            "initial_minimum": min(initial),
            "initial_negative_states": [name for name, value in zip(names, initial) if value < 0.0],
            "max_absolute_state_change_after_step": max(abs(right - left) for left, right in zip(initial, after)),
        })
    return str(neuron.__version__), observations


def run(spec_path: Path, spec_sha256: str, output_path: Path, root: Path = ROOT) -> dict[str, Any]:
    root = root.resolve()
    path = spec_path if spec_path.is_absolute() else root / spec_path
    path = path.resolve()
    spec, binding = _load_spec(path, spec_sha256, root)
    output = output_path if output_path.is_absolute() else root / output_path
    output = output.resolve()
    expected_output = _inside(root, spec["output"], "output")
    _require(output == expected_output, "output differs from the preregistration")
    _require(not output.exists(), "refusing to overwrite oracle evidence")
    source = _download_source(spec)
    with tempfile.TemporaryDirectory(prefix="v14-khaliq-neuron-oracle-") as temporary:
        library = _compile(source, Path(temporary))
        neuron_version, observations = _capture(spec, library)
    document: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "spec": binding,
        "source": dict(spec["source"]),
        "runner": dict(spec["runner"]),
        "python_executable": sys.executable,
        "neuron_version": neuron_version,
        "fixed_voltage_step_ms": spec["fixed_voltage_step_ms"],
        "observations": observations,
        "scientific_verdict": None,
        "interpretation_status": "raw_source_native_initialization_observation",
    }
    document["sha256"] = _semantic_digest(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x", encoding="ascii") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
    except FileExistsError as exc:
        raise OracleError("refusing to overwrite oracle evidence") from exc
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.spec, args.spec_sha256, args.out)
    print(json.dumps({"output": args.out.as_posix(), "sha256": result["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
