#!/usr/bin/env python3
"""Seed-free, non-scientific transport check for the V14 Stage B scorer.

This is deliberately outside the experiment harness.  It exercises two exact
candidate documents through isolated artifacts and the raw-trace scorer, but it
does not execute a model, allocate a scientific seed, or make a physiology
claim.  Its only purpose is to prove that a valid scorer NO_GO has successful
process semantics and that infrastructure errors do not acquire a verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.v14_stageB_scorer import StageBScorerError, score_raw_observations


SCHEMA = "v14-snr-stageB-readiness-dry-run-v1"
CANDIDATE_SCHEMA = "sim-adaptive-candidate-v1"
RAW_SCHEMA = "v14-snr-stageB-raw-observations-v1"
BACKEND = "numpy"
DEVICE = "cpu"
PARTITION = "readiness"
FIXTURE_RELATIVE_PATH = Path("research/fixtures/v14_snr_stageB_scorer_fixtures.json")


class ReadinessDryRunError(ValueError):
    """Raised when the deterministic readiness-only transport contract is invalid."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exact_json(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _canonical(value)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _candidate_document(candidate_id: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ReadinessDryRunError("candidate_id must be non-empty text")
    if not isinstance(parameters, Mapping) or not parameters:
        raise ReadinessDryRunError("candidate parameters must be a non-empty object")
    return {"schema": CANDIDATE_SCHEMA, "candidate_id": candidate_id, "parameters": dict(parameters)}


def _candidate_echo(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": document["candidate_id"],
        "candidate_sha256": _digest(document),
        "effective_parameters": dict(document["parameters"]),
    }


def _provenance() -> dict[str, Any]:
    return {
        "runner": "tools/v14_stageB_readiness_dry_run.py",
        "backend": BACKEND,
        "device": DEVICE,
        "partition": PARTITION,
        "seed": None,
        "reserved_seed_count": 0,
    }


def _spike_train(rate_hz: float) -> dict[str, Any]:
    """Build an uncropped, deterministic 400 ms post-burn-in trace description."""
    if rate_hz <= 0:
        raise ReadinessDryRunError("synthetic rate must be positive")
    interval = 1.0 / rate_hz
    spikes = []
    time = 0.1 + interval / 2.0
    while time < 0.5:
        spikes.append(time)
        time += interval
    return {
        "kind": "spike_train",
        "spike_times_s": spikes,
        "time_unit": "s",
        "sample_interval_s": 0.001,
        "recording_start_s": 0.0,
        "recording_end_s": 0.6,
        "burn_in_start_s": 0.0,
        "burn_in_end_s": 0.1,
        "window_start_s": 0.1,
        "window_end_s": 0.5,
    }


def _conductance_trace(peak_nS: float) -> dict[str, Any]:
    return {
        "kind": "conductance_trace",
        "time_s": [0.0, 0.001, 0.002, 0.003, 0.004, 0.005],
        "conductance_nS": [0.0, 0.0, peak_nS / 2.0, peak_nS, peak_nS / 4.0, 0.0],
        "time_unit": "s",
        "conductance_unit": "nS",
        "sample_interval_s": 0.001,
        "recording_start_s": 0.0,
        "burn_in_start_s": 0.0,
        "burn_in_end_s": 0.002,
        "window_start_s": 0.002,
        "window_end_s": 0.005,
    }


def _raw_observations(candidate: Mapping[str, Any], *, rate_hz: float, root: Path) -> dict[str, Any]:
    fixture_path = root / FIXTURE_RELATIVE_PATH
    if not fixture_path.is_file():
        raise ReadinessDryRunError(f"missing scorer fixture packet: {fixture_path}")
    return {
        "schema": RAW_SCHEMA,
        "readiness_only": {
            "synthetic": True,
            "non_scientific": True,
            "scientific_authority": "none",
            "reserved_seed_count": 0,
        },
        "backend": BACKEND,
        "device": DEVICE,
        "provenance": _provenance(),
        "adaptive_candidate": _candidate_echo(candidate),
        "fixture_packet": {
            "path": FIXTURE_RELATIVE_PATH.as_posix(),
            "sha256": _file_digest(fixture_path),
        },
        "observations": [
            {"fixture_id": "adult-autonomous-rate-observed-range", "raw": _spike_train(rate_hz)},
            {
                "fixture_id": "nalcn-lesion-ratio-4mM-model-derived",
                "raw": {
                    "kind": "paired_spike_rate_ratio",
                    "intact": _spike_train(30.0),
                    "lesion": _spike_train(17.5),
                },
            },
            {"fixture_id": "direct-pathway-unitary-peak-observed-range", "raw": _conductance_trace(1.0)},
            {"fixture_id": "pallidonigral-unitary-peak-observed-range", "raw": _conductance_trace(5.0)},
            {"fixture_id": "pallidonigral-barrage-peak-selected-range", "raw": _conductance_trace(2.0)},
        ],
    }


def _default_candidates() -> tuple[tuple[str, dict[str, Any], float], ...]:
    """Fixed readiness fixtures, not adaptive proposals or scientific parameters."""
    return (
        (
            "readiness-synthetic-in-band",
            {"readiness_trace_profile": "synthetic-in-band", "readiness_trace_version": 1},
            20.0,
        ),
        (
            "readiness-synthetic-no-go",
            {"readiness_trace_profile": "synthetic-out-of-band", "readiness_trace_version": 1},
            2.5,
        ),
    )


def _validate_candidates(candidates: Sequence[tuple[str, Mapping[str, Any], float]]) -> None:
    if len(candidates) != 2:
        raise ReadinessDryRunError("readiness dry run requires exactly two candidates")
    identifiers = [candidate[0] for candidate in candidates]
    if len(set(identifiers)) != 2:
        raise ReadinessDryRunError("readiness candidates must have distinct ids")
    if any("seed" in _canonical(candidate).decode("ascii").lower() for candidate in candidates):
        raise ReadinessDryRunError("readiness candidates must not contain seed data")


def _infrastructure_report(error: Exception, *, output_dir: Path, completed: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "readiness_only": {"synthetic": True, "non_scientific": True, "reserved_seed_count": 0},
        "process_status": "failed",
        "exit_code": 1,
        "backend": BACKEND,
        "device": DEVICE,
        "provenance": _provenance(),
        "completed_candidates": completed,
        "infrastructure_error": f"{type(error).__name__}: {error}",
        "output_dir": str(output_dir),
    }


def run_readiness_dry_run(
    output_dir: str | Path,
    *,
    root: str | Path = ROOT,
    candidates: Sequence[tuple[str, Mapping[str, Any], float]] | None = None,
) -> dict[str, Any]:
    """Run the fixed seed-free candidate transport check and write isolated artifacts.

    A returned failed report is an infrastructure result.  It deliberately has no
    ``scientific_verdict`` key, while scored physiological misses are completed
    result rows with exit-code semantic success.
    """
    root_path = Path(root).resolve()
    destination = Path(output_dir).resolve()
    selected = tuple(_default_candidates() if candidates is None else candidates)
    _validate_candidates(selected)
    if destination.exists() and any(destination.iterdir()):
        raise ReadinessDryRunError("output directory must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, Any]] = []
    try:
        for identifier, parameters, rate_hz in selected:
            candidate = _candidate_document(identifier, parameters)
            expected_echo = _candidate_echo(candidate)
            artifact_dir = destination / expected_echo["candidate_sha256"]
            if artifact_dir.exists():
                raise ReadinessDryRunError("candidate artifact directory collision")
            candidate_path = artifact_dir / "candidate.json"
            candidate_digest = _write_exact_json(candidate_path, candidate)
            _write_exact_json(artifact_dir / "candidate.json.prov.json", _provenance())
            if candidate_digest != expected_echo["candidate_sha256"]:
                raise ReadinessDryRunError("candidate artifact is not its exact canonical document")

            raw = _raw_observations(candidate, rate_hz=rate_hz, root=root_path)
            raw_path = artifact_dir / "raw-observations.json"
            raw_digest = _write_exact_json(raw_path, raw)
            score = score_raw_observations(raw, root=root_path)
            if score.get("adaptive_candidate") != expected_echo:
                raise ReadinessDryRunError("scorer candidate echo does not match isolated candidate artifact")
            verdict = "GO" if score.get("all_bounded_fixtures_passed") is True else "NO_GO"
            result = {
                "schema": SCHEMA,
                "readiness_only": {
                    "synthetic": True,
                    "non_scientific": True,
                    "scientific_authority": "none",
                    "reserved_seed_count": 0,
                },
                "process_status": "completed",
                "exit_code": 0,
                "backend": BACKEND,
                "device": DEVICE,
                "partition": PARTITION,
                "seed": None,
                "provenance": _provenance(),
                "scientific_verdict": verdict,
                "verdict_semantics": "synthetic readiness transport only; not a physiology claim",
                "adaptive_candidate": expected_echo,
                "candidate_artifact": {
                    "path": candidate_path.relative_to(destination).as_posix(),
                    "sha256": candidate_digest,
                },
                "raw_observation_artifact": {
                    "path": raw_path.relative_to(destination).as_posix(),
                    "sha256": raw_digest,
                },
                "score": score,
            }
            result_path = artifact_dir / "result.json"
            _write_exact_json(result_path, result)
            completed.append(result)
    except (OSError, ReadinessDryRunError, StageBScorerError, ValueError, TypeError) as error:
        report = _infrastructure_report(error, output_dir=destination, completed=completed)
        _write_exact_json(destination / "readiness-dry-run.json", report)
        return report

    report = {
        "schema": SCHEMA,
        "readiness_only": {"synthetic": True, "non_scientific": True, "reserved_seed_count": 0},
        "process_status": "completed",
        "exit_code": 0,
        "backend": BACKEND,
        "device": DEVICE,
        "provenance": _provenance(),
        "backend_partition_pairs": [{"backend": BACKEND, "partition": PARTITION}],
        "seeds_selected": False,
        "held_out_partitions_accessed": [],
        "candidate_count": len(completed),
        "candidate_digests": [item["adaptive_candidate"]["candidate_sha256"] for item in completed],
        "results": completed,
    }
    _write_exact_json(destination / "readiness-dry-run.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="new or empty directory for isolated artifacts")
    parser.add_argument("--root", default=str(ROOT), help="repository root that owns the scorer fixture packet")
    args = parser.parse_args(argv)
    try:
        report = run_readiness_dry_run(args.output_dir, root=args.root)
    except ReadinessDryRunError as error:
        print(json.dumps({"process_status": "failed", "exit_code": 1, "infrastructure_error": str(error)}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
