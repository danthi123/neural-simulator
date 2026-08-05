#!/usr/bin/env python3
"""Triage a complete Stage B GPU campaign without making scientific claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.snr_executable_packet import canonical_bytes
from tools.compact_trace import CompactTraceError, load_compact_trace
from tools.v14_stageB_campaign import (
    CAMPAIGN_SCHEMA,
    GPU_BATCH_RECEIPT_SCHEMA,
    PHASED_CAMPAIGN_SCHEMA,
    PHASED_GPU_BATCH_RECEIPT_SCHEMA,
)
from tools.v14_stageB_physiology_metrics import phased_nap_voltage_measure


ROOT = Path(__file__).resolve().parents[1]
TRIAGE_SCHEMA = "v14-snr-stageB-gpu-triage-v1"
PHASED_TRIAGE_SCHEMA = "v14-snr-stageB-gpu-triage-v2"
ARMS = (
    "intact_autonomous",
    "nap_lesion",
    "cav2_2_lesion",
    "sk_lesion",
    "hcn_baseline_lesion",
)


class StageBGPUTriageError(ValueError):
    """Raised when GPU campaign evidence is incomplete or unauthenticated."""


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(canonical_bytes(value))


def _sha(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StageBGPUTriageError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _repo_path(root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise StageBGPUTriageError(f"{context} path must be repository-relative")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise StageBGPUTriageError(f"{context} path must be canonical repository-relative text")
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageBGPUTriageError(f"{context} path escapes repository_root") from exc
    if path.is_symlink() or not path.is_file():
        raise StageBGPUTriageError(f"{context} must be a regular file")
    return path


def _load_json(path: Path, expected_sha256: str, context: str) -> dict[str, Any]:
    expected = _sha(expected_sha256, f"{context} digest")
    raw = path.read_bytes()
    if _digest_bytes(raw) != expected:
        raise StageBGPUTriageError(f"{context} digest does not match")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageBGPUTriageError(f"{context} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise StageBGPUTriageError(f"{context} must contain an object")
    return value


def _event_metrics(spikes: np.ndarray, times: np.ndarray) -> dict[str, Any]:
    spike_times = times[spikes]
    result: dict[str, Any] = {"spike_count": int(spike_times.size)}
    if spike_times.size < 101:
        result.update({"firing_rate_hz": None, "isi_cv": None})
        return result
    selected = spike_times[:101]
    intervals = np.diff(selected)
    if intervals.size != 100 or np.any(intervals <= 0.0):
        raise StageBGPUTriageError("101-spike trace has invalid event ordering")
    mean_interval = float(np.mean(intervals))
    result.update({
        "firing_rate_hz": 100.0 / float(selected[-1] - selected[0]),
        "isi_cv": float(np.std(intervals, ddof=0) / mean_interval),
    })
    return result


def _nap_metrics(
    spikes: np.ndarray, times: np.ndarray, voltage_mV: np.ndarray
) -> dict[str, Any]:
    """Preserve V1 metrics or recompute the complete V3 same-cell assay."""

    if len(times) != 60_000:
        return _event_metrics(spikes, times)
    if len(voltage_mV) != len(times):
        raise StageBGPUTriageError("phased NaP trace changed its voltage shape")
    dt = 0.00005
    if not np.allclose(
        times, np.arange(1, 60_001, dtype=np.float64) * dt, rtol=0.0, atol=1e-12
    ):
        raise StageBGPUTriageError("phased NaP trace changed its filed sampling schedule")
    measured = phased_nap_voltage_measure(
        times,
        voltage_mV,
        times[spikes],
        time_unit="s",
        voltage_unit="mV",
        sample_interval_s=dt,
        recording_start_s=dt,
        recording_end_s=3.0 + dt,
        burn_in_start_s=dt,
        burn_in_end_s=1.0,
        stability_comparison_start_s=1.0,
        stability_comparison_end_s=1.5,
        baseline_start_s=1.5,
        baseline_end_s=2.0,
        post_lesion_start_s=2.0,
        post_lesion_end_s=3.0,
    )
    return {
        "spike_count": measured["post_lesion_spike_count"],
        "firing_rate_hz": None,
        "isi_cv": None,
        "same_cell_phased": True,
        "baseline_stable": abs(float(measured["stability_delta_mV"])) <= 0.5,
        **measured,
    }


def _validate_phased_nap_intervention(value: Any) -> None:
    fixed = {
        "kind": "complete_intrinsic_current_lesion",
        "operation": "set_conductance_density_to_zero_between_post_update_samples",
        "target": "nap",
        "runtime_conductance_field": "cp_snr_g_nap_max",
        "conductance_density_unit": "mS/cm^2",
        "after": 0.0,
        "timestamp_s": 2.0,
        "lesion_onset_sample_index": 39_999,
        "lesion_onset_sample_number": 40_000,
        "last_intact_sample_s": 1.99995,
        "first_lesion_sample_s": 2.0,
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != set(fixed) | {"before"}
        or any(value.get(key) != expected for key, expected in fixed.items())
        or isinstance(value.get("before"), bool)
        or not isinstance(value.get("before"), (int, float))
        or not np.isfinite(float(value["before"]))
        or float(value["before"]) <= 0.0
    ):
        raise StageBGPUTriageError("phased NaP receipt does not prove the exact lesion")


def _classify(metrics: Mapping[str, Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    intact = metrics["intact_autonomous"]
    nap = metrics["nap_lesion"]
    cav = metrics["cav2_2_lesion"]
    sk = metrics["sk_lesion"]
    hcn = metrics["hcn_baseline_lesion"]

    def direction(lesion: Mapping[str, Any], key: str) -> bool | None:
        left, right = intact.get(key), lesion.get(key)
        return None if left is None or right is None else bool(float(right) > float(left))

    hcn_rate_change: float | None = None
    if intact.get("firing_rate_hz") is not None and hcn.get("firing_rate_hz") is not None:
        baseline = float(intact["firing_rate_hz"])
        if baseline <= 0.0:
            raise StageBGPUTriageError("HCN rate change is undefined for zero intact rate")
        hcn_rate_change = abs(float(hcn["firing_rate_hz"]) - baseline) / baseline
    checks = [
        {"gate_id": "nap-complete-lesion", "metric": "spike_count", "passed": nap["spike_count"] == 0},
        {"gate_id": "cav2.2-complete-lesion", "metric": "isi_cv", "passed": direction(cav, "isi_cv")},
        {"gate_id": "sk-complete-lesion", "metric": "isi_cv", "passed": direction(sk, "isi_cv")},
        {"gate_id": "hcn-complete-lesion", "metric": "lesion_spike_count", "passed": hcn["spike_count"] > 0},
        {
            "gate_id": "hcn-complete-lesion",
            "metric": "absolute_baseline_rate_change_fraction",
            "observed": hcn_rate_change,
            "passed": None if hcn_rate_change is None else hcn_rate_change <= 0.2,
        },
    ]
    if nap.get("same_cell_phased") is True:
        checks[1:1] = [
            {
                "gate_id": "nap-complete-lesion",
                "metric": "baseline_stable",
                "observed": nap["stability_delta_mV"],
                "passed": nap["baseline_stable"],
            },
            {
                "gate_id": "nap-complete-lesion",
                "metric": "median_membrane_voltage_change_mV",
                "observed": nap["post_lesion_delta_mV"],
                "passed": float(nap["post_lesion_delta_mV"]) < 0.0,
            },
        ]
    if any(item["passed"] is False for item in checks):
        classification = "engineering_fail"
    elif any(item["passed"] is None for item in checks):
        classification = "engineering_inconclusive"
    else:
        classification = "engineering_pass"
    return classification, checks


def triage_gpu_campaign(
    campaign_path: str | Path,
    campaign_sha256: str,
    results_root: str | Path,
    *,
    repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    root = Path(repository_root).expanduser().resolve(strict=True)
    supplied_campaign = Path(campaign_path).expanduser()
    campaign_file = (
        supplied_campaign if supplied_campaign.is_absolute() else root / supplied_campaign
    ).resolve()
    try:
        campaign_relative = campaign_file.relative_to(root).as_posix()
    except ValueError as exc:
        raise StageBGPUTriageError("campaign must be inside repository_root") from exc
    campaign = _load_json(campaign_file, campaign_sha256, "campaign")
    campaign_body = {key: value for key, value in campaign.items() if key != "sha256"}
    phased_campaign = campaign.get("schema") == PHASED_CAMPAIGN_SCHEMA
    batch_size = campaign.get("batch_size")
    expected_batches = None
    if phased_campaign and isinstance(batch_size, int) and not isinstance(batch_size, bool) and batch_size > 0:
        expected_batches = len(ARMS) * math.ceil(512 / batch_size)
    if (
        campaign.get("schema") not in {CAMPAIGN_SCHEMA, PHASED_CAMPAIGN_SCHEMA}
        or campaign.get("sha256") != _digest(campaign_body)
        or campaign.get("candidate_count") != 512
        or campaign.get("arm_count") != len(ARMS)
        or (
            campaign.get("batch_count")
            != (expected_batches if phased_campaign else 40)
        )
    ):
        raise StageBGPUTriageError("campaign is not the exact materialized 512-candidate screen")
    results = Path(results_root).expanduser()
    results = (results if results.is_absolute() else root / results).resolve()
    try:
        results.relative_to(root)
    except ValueError as exc:
        raise StageBGPUTriageError("results_root must be inside repository_root") from exc

    campaign_binding = {"path": campaign_relative, "sha256": _digest_bytes(campaign_file.read_bytes())}
    by_candidate: dict[str, dict[str, Any]] = {}
    declaration_count = 0
    for declaration in campaign["declarations"]:
        arm = declaration["arm"]
        index = declaration["batch_index"]
        declaration_path = _repo_path(root, declaration.get("path"), "batch declaration")
        declaration_document = _load_json(
            declaration_path, declaration.get("sha256"), "batch declaration"
        )
        expected_candidates = {
            item.get("candidate_id"): item.get("candidate_sha256")
            for item in declaration_document.get("candidates", [])
            if isinstance(item, Mapping)
        }
        if (
            declaration_document.get("sha256") != declaration.get("declaration_sha256")
            or declaration_document.get("arm") != arm
            or len(expected_candidates) != declaration.get("candidate_count")
        ):
            raise StageBGPUTriageError("batch declaration identity is invalid")
        receipt_path = results / arm / f"batch-{index:03d}" / "receipt.json"
        receipt = _load_json(receipt_path, _digest_bytes(receipt_path.read_bytes()), "GPU receipt")
        receipt_body = {key: value for key, value in receipt.items() if key != "sha256"}
        if (
            receipt.get("schema")
            not in {GPU_BATCH_RECEIPT_SCHEMA, PHASED_GPU_BATCH_RECEIPT_SCHEMA}
            or receipt.get("sha256") != _digest(receipt_body)
            or receipt.get("campaign") != campaign_binding
            or receipt.get("declaration") != {
                "path": declaration["path"],
                "sha256": declaration["sha256"],
                "declaration_sha256": declaration["declaration_sha256"],
            }
            or receipt.get("arm") != arm
            or receipt.get("batch_index") != index
            or receipt.get("engineering_screening_only") is not True
            or receipt.get("scientific_verdict") is not None
            or receipt.get("numpy_confirmation_required") is not True
        ):
            raise StageBGPUTriageError("GPU receipt changed identity or scientific boundary")
        if (receipt.get("schema") == PHASED_GPU_BATCH_RECEIPT_SCHEMA) != (
            campaign.get("schema") == PHASED_CAMPAIGN_SCHEMA
        ):
            raise StageBGPUTriageError("GPU receipt schema does not match campaign revision")
        observed_candidates = {
            item.get("candidate_id"): item.get("candidate_sha256")
            for item in receipt.get("traces", [])
            if isinstance(item, Mapping)
        }
        if observed_candidates != expected_candidates:
            raise StageBGPUTriageError("GPU receipt does not exactly cover its declaration")
        declaration_count += 1
        for trace in receipt.get("traces", []):
            candidate_id = trace.get("candidate_id")
            candidate_sha = trace.get("candidate_sha256")
            binding = trace.get("compact_trace")
            if not isinstance(candidate_id, str) or not isinstance(binding, Mapping):
                raise StageBGPUTriageError("GPU receipt has an invalid trace binding")
            archive = _repo_path(root, binding.get("path"), "compact trace")
            try:
                arrays = load_compact_trace(archive, expected_sha256=_sha(binding.get("sha256"), "compact trace"))
            except (CompactTraceError, OSError, TypeError, ValueError) as exc:
                raise StageBGPUTriageError(f"compact trace authentication failed: {exc}") from exc
            if len(arrays["time"]) != binding.get("sample_count"):
                raise StageBGPUTriageError("compact trace sample count does not match receipt")
            entry = by_candidate.setdefault(candidate_id, {
                "candidate_id": candidate_id,
                "candidate_sha256": candidate_sha,
                "metrics": {},
            })
            if entry["candidate_sha256"] != candidate_sha or arm in entry["metrics"]:
                raise StageBGPUTriageError("candidate identity is duplicated or inconsistent")
            if phased_campaign and arm == "nap_lesion":
                _validate_phased_nap_intervention(trace.get("runtime_intervention"))
            entry["metrics"][arm] = (
                _nap_metrics(arrays["spikes"], arrays["time"], arrays["voltage"])
                if arm == "nap_lesion"
                else _event_metrics(arrays["spikes"], arrays["time"])
            )

    if declaration_count != campaign["batch_count"] or len(by_candidate) != 512:
        raise StageBGPUTriageError("GPU result set is incomplete")
    output_candidates = []
    for candidate_id in sorted(by_candidate):
        entry = by_candidate[candidate_id]
        if set(entry["metrics"]) != set(ARMS):
            raise StageBGPUTriageError(f"candidate {candidate_id} is missing a GPU arm")
        classification, checks = _classify(entry["metrics"])
        output_candidates.append({
            **entry,
            "classification": classification,
            "resolved_checks": checks,
        })
    counts = Counter(item["classification"] for item in output_candidates)
    phased_nap = all(
        item["metrics"]["nap_lesion"].get("same_cell_phased") is True
        for item in output_candidates
    )
    body = {
        "schema": PHASED_TRIAGE_SCHEMA if phased_nap else TRIAGE_SCHEMA,
        "process_status": "completed",
        "engineering_screening_only": True,
        "scientific_verdict": None,
        "source_equivalence_claimed": False,
        "numpy_confirmation_required": True,
        "campaign": campaign_binding,
        "candidate_count": len(output_candidates),
        "classification_counts": dict(sorted(counts.items())),
        "candidates": output_candidates,
    }
    return {**body, "sha256": _digest(body)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--campaign-sha256", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        result = triage_gpu_campaign(
            args.campaign, args.campaign_sha256, args.results_root,
            repository_root=args.repository_root,
        )
        root = Path(args.repository_root).resolve(strict=True)
        output = Path(args.output)
        output = (output if output.is_absolute() else root / output).resolve()
        output.relative_to(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(canonical_bytes(result))
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(2, f"Stage B GPU triage failure: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
