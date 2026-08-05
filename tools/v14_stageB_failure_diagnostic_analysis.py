#!/usr/bin/env python3
"""Authenticate and summarize the preregistered Stage B failure diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.diagnostic_trace import DiagnosticTraceError, load_diagnostic_trace


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA = "v14-snr-stageB-failure-diagnostic-receipt-v1"
OUTPUT_SCHEMA = "v14-snr-stageB-failure-diagnostic-analysis-v1"
EXPECTED_CURRENTS = (0.0, -10.0, -20.0, -30.0)
PHASE_SLICES = {
    "baseline": slice(29_999, 39_999),
    "immediate_post_lesion": slice(39_999, 40_399),
    "late_post_lesion": slice(49_999, 59_999),
    "pulse": slice(59_999, 69_999),
    "late_pulse": slice(67_999, 69_999),
    "release": slice(69_999, 90_000),
}


class StageBFailureAnalysisError(ValueError):
    """Raised when diagnostic evidence is incomplete or unauthenticated."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(_canonical_bytes(value))


def _sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageBFailureAnalysisError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _inside_file(root: Path, value: Any, context: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise StageBFailureAnalysisError(f"{context} path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or str(relative) != value or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise StageBFailureAnalysisError(f"{context} path is not canonical")
    unresolved = root.joinpath(*relative.parts)
    if unresolved.is_symlink():
        raise StageBFailureAnalysisError(f"{context} must not be a symbolic link")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageBFailureAnalysisError(f"{context} escapes repository root") from exc
    if not path.is_file():
        raise StageBFailureAnalysisError(f"{context} must be a regular file")
    return value, path


def _load_receipt(
    root: Path, receipt_path: str | Path, expected_sha256: str | None
) -> tuple[dict[str, str], dict[str, Any]]:
    supplied = Path(receipt_path).expanduser()
    path = (supplied if supplied.is_absolute() else root / supplied).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise StageBFailureAnalysisError("receipt escapes repository root") from exc
    _, path = _inside_file(root, relative, "receipt")
    raw = path.read_bytes()
    file_sha = _digest_bytes(raw)
    if expected_sha256 is not None and file_sha != _sha256(
        expected_sha256, "receipt sha256"
    ):
        raise StageBFailureAnalysisError("receipt digest does not match")
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageBFailureAnalysisError("receipt is not valid JSON") from exc
    if not isinstance(receipt, dict):
        raise StageBFailureAnalysisError("receipt must contain an object")
    body = {key: value for key, value in receipt.items() if key != "sha256"}
    if receipt.get("sha256") != _digest(body):
        raise StageBFailureAnalysisError("receipt self digest is invalid")
    return {"path": relative, "sha256": file_sha, "self_sha256": receipt["sha256"]}, receipt


def _validate_receipt(receipt: Mapping[str, Any]) -> tuple[list[str], dict[tuple[str, float], Mapping[str, Any]]]:
    execution = receipt.get("execution")
    selection = receipt.get("selection")
    traces = receipt.get("traces")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("process_status") != "completed"
        or receipt.get("engineering_diagnostic_only") is not True
        or receipt.get("scientific_verdict") is not None
        or receipt.get("candidate_promotion_allowed") is not False
        or receipt.get("parameter_tuning_allowed") is not False
        or receipt.get("source_equivalence_claimed") is not False
        or not isinstance(execution, Mapping)
        or execution.get("candidate_count") != 9
        or execution.get("arm_count") != 4
        or execution.get("trace_count") != 36
        or execution.get("total_steps_per_arm") != 90_000
        or not isinstance(selection, list)
        or len(selection) != 9
        or not isinstance(traces, list)
        or len(traces) != 36
    ):
        raise StageBFailureAnalysisError("receipt identity, boundary, or exact shape is invalid")
    candidate_ids = []
    identities: dict[str, str] = {}
    for row in selection:
        if not isinstance(row, Mapping) or set(row) != {"candidate_id", "candidate_sha256"}:
            raise StageBFailureAnalysisError("selection identity is invalid")
        candidate_id = row["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in identities:
            raise StageBFailureAnalysisError("selection contains an invalid candidate id")
        identities[candidate_id] = _sha256(row["candidate_sha256"], "candidate sha256")
        candidate_ids.append(candidate_id)

    indexed: dict[tuple[str, float], Mapping[str, Any]] = {}
    for trace in traces:
        if not isinstance(trace, Mapping):
            raise StageBFailureAnalysisError("trace receipt is invalid")
        candidate_id = trace.get("candidate_id")
        current = trace.get("rescue_current_pA")
        binding = trace.get("diagnostic_trace")
        key = (candidate_id, current)
        if (
            candidate_id not in identities
            or trace.get("candidate_sha256") != identities[candidate_id]
            or current not in EXPECTED_CURRENTS
            or key in indexed
            or not isinstance(binding, Mapping)
            or set(binding) != {"path", "sample_count", "sha256"}
            or binding.get("sample_count") != 90_000
        ):
            raise StageBFailureAnalysisError("trace identity or binding is invalid")
        indexed[key] = trace
    expected = {(candidate_id, current) for candidate_id in candidate_ids for current in EXPECTED_CURRENTS}
    if set(indexed) != expected:
        raise StageBFailureAnalysisError("receipt does not contain the exact candidate-current cross product")
    return candidate_ids, indexed


def _median(channels: Mapping[str, np.ndarray], name: str, window: slice) -> float:
    return float(np.median(channels[name][window]))


def _spike_count(channels: Mapping[str, np.ndarray], window: slice) -> int:
    return int(np.count_nonzero(channels["spikes"][window]))


def _first_spike_latency_ms(channels: Mapping[str, np.ndarray], window: slice) -> float | None:
    indices = np.flatnonzero(channels["spikes"][window])
    return None if not indices.size else float(indices[0] * 0.05)


def analyze_failure_diagnostic(
    receipt_path: str | Path,
    *,
    repository_root: str | Path = ROOT,
    receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Return authenticated descriptive evidence without tuning or promotion."""
    root = Path(repository_root).expanduser().resolve(strict=True)
    receipt_binding, receipt = _load_receipt(root, receipt_path, receipt_sha256)
    candidate_ids, indexed = _validate_receipt(receipt)
    loaded: dict[tuple[str, float], dict[str, np.ndarray]] = {}
    trace_digests: list[dict[str, Any]] = []
    for key, trace in indexed.items():
        binding = trace["diagnostic_trace"]
        relative, path = _inside_file(root, binding["path"], "diagnostic trace")
        try:
            time, channels = load_diagnostic_trace(path, binding["sha256"])
        except (OSError, TypeError, ValueError, DiagnosticTraceError) as exc:
            raise StageBFailureAnalysisError(
                f"diagnostic trace {key[0]} at {key[1]} pA is invalid: {exc}"
            ) from exc
        if (
            time.shape != (90_000,)
            or time[0] != 0.00005
            or time[-1] != 4.5
            or channels["spikes"].dtype != np.dtype("|b1")
        ):
            raise StageBFailureAnalysisError("diagnostic trace timing or dtype is invalid")
        loaded[key] = channels
        trace_digests.append(
            {"candidate_id": key[0], "rescue_current_pA": key[1], "path": relative, "sha256": binding["sha256"]}
        )

    rows = []
    coupling_count = 0
    rescue_count = 0
    matched_prefix_count = 0
    immediate_hyperpolarization_count = 0
    calcium_decline_count = 0
    sk_activation_decline_count = 0
    fast_na_availability_increase_count = 0
    any_release_spike_rescue_count = 0
    for candidate_id in candidate_ids:
        controls = loaded[(candidate_id, 0.0)]
        prefix_identical = all(
            all(
                np.array_equal(controls[name][:59_999], loaded[(candidate_id, current)][name][:59_999])
                for name in controls
                if name != "i_external"
            )
            for current in EXPECTED_CURRENTS[1:]
        )
        matched_prefix_count += int(prefix_identical)
        baseline_voltage = _median(controls, "post_update_voltage_mv", PHASE_SLICES["baseline"])
        immediate_voltage = _median(
            controls, "post_update_voltage_mv", PHASE_SLICES["immediate_post_lesion"]
        )
        late_voltage = _median(controls, "post_update_voltage_mv", PHASE_SLICES["late_post_lesion"])
        baseline_calcium = _median(controls, "calcium_um", PHASE_SLICES["baseline"])
        late_calcium = _median(controls, "calcium_um", PHASE_SLICES["late_post_lesion"])
        baseline_sk = _median(controls, "i_sk", PHASE_SLICES["baseline"])
        late_sk = _median(controls, "i_sk", PHASE_SLICES["late_post_lesion"])
        baseline_sk_activation = _median(
            controls, "sk_activation", PHASE_SLICES["baseline"]
        )
        late_sk_activation = _median(
            controls, "sk_activation", PHASE_SLICES["late_post_lesion"]
        )
        baseline_h = _median(controls, "fast_na_inactivation", PHASE_SLICES["baseline"])
        late_h = _median(controls, "fast_na_inactivation", PHASE_SLICES["late_post_lesion"])
        immediate_voltage_trace = controls["post_update_voltage_mv"][
            PHASE_SLICES["immediate_post_lesion"]
        ]
        immediate_min_voltage = float(np.min(immediate_voltage_trace))
        immediate_min_latency_ms = float(np.argmin(immediate_voltage_trace) * 0.05)
        immediate_hyperpolarization = immediate_min_voltage < baseline_voltage
        calcium_declined = late_calcium < baseline_calcium
        sk_activation_declined = late_sk_activation < baseline_sk_activation
        coupling_supported = (
            immediate_hyperpolarization
            and calcium_declined
            and sk_activation_declined
            and late_voltage > immediate_min_voltage
        )
        coupling_count += int(coupling_supported)
        immediate_hyperpolarization_count += int(immediate_hyperpolarization)
        calcium_decline_count += int(calcium_declined)
        sk_activation_decline_count += int(sk_activation_declined)

        rescue_rows = []
        zero_release_spikes = _spike_count(controls, PHASE_SLICES["release"])
        for current in EXPECTED_CURRENTS:
            channels = loaded[(candidate_id, current)]
            rescue_rows.append(
                {
                    "current_pA": current,
                    "late_pulse_fast_na_inactivation_median": _median(
                        channels, "fast_na_inactivation", PHASE_SLICES["late_pulse"]
                    ),
                    "pulse_spike_count": _spike_count(channels, PHASE_SLICES["pulse"]),
                    "release_spike_count": _spike_count(channels, PHASE_SLICES["release"]),
                    "first_release_spike_latency_ms": _first_spike_latency_ms(
                        channels, PHASE_SLICES["release"]
                    ),
                }
            )
        minus_30 = rescue_rows[-1]
        fast_na_availability_increased = (
            minus_30["late_pulse_fast_na_inactivation_median"]
            > rescue_rows[0]["late_pulse_fast_na_inactivation_median"]
        )
        any_release_spike_rescue = (
            minus_30["release_spike_count"] > zero_release_spikes
        )
        rescue_supported = (
            fast_na_availability_increased and any_release_spike_rescue
        )
        rescue_count += int(rescue_supported)
        fast_na_availability_increase_count += int(fast_na_availability_increased)
        any_release_spike_rescue_count += int(any_release_spike_rescue)
        rows.append(
            {
                "candidate_id": candidate_id,
                "pre_pulse_prefix_identical_across_arms": prefix_identical,
                "baseline": {
                    "voltage_median_mV": baseline_voltage,
                    "calcium_median_um": baseline_calcium,
                    "i_sk_median_uA_per_cm2": baseline_sk,
                    "fast_na_inactivation_median": baseline_h,
                },
                "immediate_post_lesion": {
                    "voltage_median_mV": immediate_voltage,
                    "voltage_delta_from_baseline_mV": immediate_voltage - baseline_voltage,
                    "minimum_voltage_mV": immediate_min_voltage,
                    "minimum_delta_from_baseline_mV": immediate_min_voltage - baseline_voltage,
                    "minimum_latency_ms": immediate_min_latency_ms,
                    "hyperpolarization_observed": immediate_hyperpolarization,
                },
                "late_post_lesion": {
                    "voltage_median_mV": late_voltage,
                    "voltage_delta_from_baseline_mV": late_voltage - baseline_voltage,
                    "calcium_median_um": late_calcium,
                    "calcium_fraction_of_baseline": late_calcium / baseline_calcium,
                    "i_sk_median_uA_per_cm2": late_sk,
                    "i_sk_fraction_of_baseline": late_sk / baseline_sk,
                    "sk_activation_median": late_sk_activation,
                    "sk_activation_fraction_of_baseline": (
                        late_sk_activation / baseline_sk_activation
                    ),
                    "fast_na_inactivation_median": late_h,
                    "spike_count": _spike_count(controls, PHASE_SLICES["late_post_lesion"]),
                },
                "coupling_collapse_direction_supported": coupling_supported,
                "minus_30_pA_release_rescue_supported": rescue_supported,
                "minus_30_pA_fast_na_availability_increased": (
                    fast_na_availability_increased
                ),
                "minus_30_pA_any_release_spike_rescue": any_release_spike_rescue,
                "rescue_arms": rescue_rows,
            }
        )

    body = {
        "schema": OUTPUT_SCHEMA,
        "process_status": "completed",
        "engineering_diagnostic_only": True,
        "scientific_verdict": None,
        "candidate_promotion_allowed": False,
        "parameter_tuning_allowed": False,
        "source_equivalence_claimed": False,
        "receipt": receipt_binding,
        "trace_count": len(trace_digests),
        "trace_bindings": trace_digests,
        "analysis_conventions": {
            "baseline_window_s": [1.5, 2.0],
            "immediate_post_lesion_window_ms": [0.0, 20.0],
            "late_post_lesion_window_ms": [500.0, 1000.0],
            "late_pulse_window_ms": [400.0, 500.0],
            "release_window_ms": [0.0, 1000.05],
            "aggregation": "per-candidate discrete median or exact spike count",
            "classification_boundary": "descriptive post-execution convention; support requires the filed direction in all nine candidates and cannot promote a candidate",
        },
        "summary": {
            "candidate_count": len(candidate_ids),
            "matched_prefix_identical_count": matched_prefix_count,
            "coupling_collapse_direction_supported_count": coupling_count,
            "minus_30_pA_release_rescue_supported_count": rescue_count,
            "immediate_hyperpolarization_observed_count": (
                immediate_hyperpolarization_count
            ),
            "late_calcium_decline_count": calcium_decline_count,
            "late_sk_activation_decline_count": sk_activation_decline_count,
            "minus_30_pA_fast_na_availability_increase_count": (
                fast_na_availability_increase_count
            ),
            "minus_30_pA_any_release_spike_rescue_count": (
                any_release_spike_rescue_count
            ),
            "coupling_collapse_direction_unanimous": coupling_count == len(candidate_ids),
            "minus_30_pA_release_rescue_unanimous": rescue_count == len(candidate_ids),
        },
        "candidates": rows,
        "interpretation_boundary": (
            "This analysis localizes an engineering failure in the current single-compartment model. "
            "The zero-pA arm controls the rescue pulse but v1 has no intact-NaP sham continuation, "
            "so spike-phase-confounded immediate lesion voltage cannot establish a causal sign. "
            "It does not establish source equivalence, biological validity, or a Stage B pass."
        ),
    }
    return {**body, "sha256": _digest(body)}


def _publish_once(path: Path, result: Mapping[str, Any]) -> None:
    data = _canonical_bytes(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    linked = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        linked = True
    except Exception:
        if linked:
            path.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--receipt-sha256")
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        result = analyze_failure_diagnostic(
            args.receipt,
            repository_root=args.repository_root,
            receipt_sha256=args.receipt_sha256,
        )
        if args.output:
            root = Path(args.repository_root).expanduser().resolve(strict=True)
            output = Path(args.output).expanduser()
            output = (output if output.is_absolute() else root / output).resolve()
            try:
                output.relative_to(root)
            except ValueError as exc:
                raise StageBFailureAnalysisError("output escapes repository root") from exc
            if output.exists() or output.is_symlink():
                raise StageBFailureAnalysisError("refusing to replace existing output")
            _publish_once(output, result)
        print(_canonical_bytes(result).decode("ascii"))
    except (OSError, TypeError, ValueError, StageBFailureAnalysisError) as exc:
        parser.exit(2, f"Stage B failure diagnostic analysis error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
