#!/usr/bin/env python3
"""Run the preregistered Stage B V3 NaP silent-state diagnostic.

This is a post-hoc engineering diagnostic.  It cannot promote candidates or
produce a Stage B or scientific verdict.  The production Stage B campaign is
intentionally left unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if __package__ in {None, ""}:
    sys.path.insert(0, str(ROOT))

from tools.lab import attributable_to
PROTOCOL_SCHEMA = "v14-snr-stageB-failure-diagnostic-protocol-v1"
PROTOCOL_ID = "v14-stageB-v3-nap-silent-state-localizer-v1"
RECEIPT_SCHEMA = "v14-snr-stageB-failure-diagnostic-receipt-v1"
CAMPAIGN_SCHEMA = "v14-snr-stageB-screen-campaign-v2"
TRIAGE_SCHEMA = "v14-snr-stageB-gpu-triage-v2"
DT_MS = 0.05
DEFAULT_CHUNK_STEPS = 4096
MAX_CHUNK_STEPS = 65_536
EXPECTED_CURRENTS_PA = (0.0, -10.0, -20.0, -30.0)
EXPECTED_PHASES = (
    ("intact_baseline", 0.0, 2.0, "packet_value", 0.0),
    ("nap_lesion", 2.0, 3.0, 0.0, 0.0),
    ("hyperpolarizing_pulse", 3.0, 3.5, 0.0, "arm_value"),
    ("post_pulse_release", 3.5, 4.5, 0.0, 0.0),
)
EXPECTED_STATE_CHANNELS = (
    "pre_update_voltage_mV",
    "post_update_voltage_mV",
    "spikes",
    "fast_na_activation",
    "fast_na_inactivation",
    "fast_k_activation",
    "nap_activation",
    "nap_inactivation",
    "cav22_activation",
    "cav22_inactivation",
    "calcium_um",
    "sk_activation",
    "hcn_activation",
)
EXPECTED_CURRENT_CHANNELS = (
    "i_fast_na",
    "i_fast_k",
    "i_leak",
    "i_nalcn",
    "i_nap",
    "i_cav22",
    "i_sk",
    "i_hcn",
    "i_total_ionic",
    "i_external",
)
_ARCHIVE_CHANNEL_NAMES = {
    "pre_update_voltage_mV": "pre_update_voltage_mv",
    "post_update_voltage_mV": "post_update_voltage_mv",
}
_CAPTURE_STATE_FIELDS = (
    ("pre_update_voltage_mV", None),
    ("post_update_voltage_mV", "cp_membrane_potential_v"),
    ("spikes", "cp_firing_states"),
    ("fast_na_activation", "cp_gating_variable_m"),
    ("fast_na_inactivation", "cp_gating_variable_h"),
    ("fast_k_activation", "cp_gating_variable_n"),
    ("nap_activation", "cp_snr_nap_activation"),
    ("nap_inactivation", "cp_snr_nap_inactivation"),
    ("cav22_activation", "cp_snr_ca_activation"),
    ("cav22_inactivation", "cp_snr_ca_inactivation"),
    ("calcium_um", "cp_snr_calcium"),
    ("sk_activation", "cp_snr_sk_activation"),
    ("hcn_activation", "cp_snr_h_activation"),
    ("snr_effective_input_scratch", "cp_snr_ionic_current_scratch"),
)
_CURRENT_NAMES = (
    "i_fast_na",
    "i_fast_k",
    "i_leak",
    "i_nalcn",
    "i_nap",
    "i_cav22",
    "i_sk",
    "i_hcn",
    "i_total_ionic",
)


class StageBFailureDiagnosticError(ValueError):
    """Raised before unauthenticated or scientifically unsafe work can run."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise StageBFailureDiagnosticError(f"value is not canonical JSON: {exc}") from exc


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
        raise StageBFailureDiagnosticError(
            f"{context} must be a lowercase SHA-256 digest"
        )
    return value


def _repository_file(root: Path, value: Any, context: str) -> tuple[str, Path]:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\\" in value
        or "\x00" in value
        or any(ord(character) > 127 for character in value)
    ):
        raise StageBFailureDiagnosticError(
            f"{context} must be canonical repository-relative POSIX text"
        )
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or str(relative) != value
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise StageBFailureDiagnosticError(
            f"{context} must be canonical repository-relative POSIX text"
        )
    unresolved = root.joinpath(*relative.parts)
    if unresolved.is_symlink():
        raise StageBFailureDiagnosticError(f"{context} must not be a symbolic link")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise StageBFailureDiagnosticError(f"{context} escapes repository root") from exc
    if not path.is_file():
        raise StageBFailureDiagnosticError(f"{context} must be a regular file")
    return value, path


def _load_json_file(
    root: Path,
    path_value: Any,
    expected_sha256: Any,
    context: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    relative, path = _repository_file(root, path_value, f"{context} path")
    expected = _sha256(expected_sha256, f"{context} sha256")
    raw = path.read_bytes()
    if _digest_bytes(raw) != expected:
        raise StageBFailureDiagnosticError(f"{context} digest does not match")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise StageBFailureDiagnosticError(f"{context} is not valid JSON") from exc
    if not isinstance(document, dict):
        raise StageBFailureDiagnosticError(f"{context} must contain a JSON object")
    return {"path": relative, "sha256": expected}, document


def _validate_self_digest(document: Mapping[str, Any], context: str) -> None:
    observed = _sha256(document.get("sha256"), f"{context} self sha256")
    body = {key: value for key, value in document.items() if key != "sha256"}
    if observed != _digest(body):
        raise StageBFailureDiagnosticError(f"{context} self digest is invalid")


def _phase_plan(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    execution = protocol.get("execution")
    if not isinstance(execution, Mapping):
        raise StageBFailureDiagnosticError("protocol execution must be an object")
    phases = execution.get("phases")
    if not isinstance(phases, list) or len(phases) != len(EXPECTED_PHASES):
        raise StageBFailureDiagnosticError("protocol changed the exact phase family")
    dt_s = DT_MS / 1000.0
    total_steps = int(round(EXPECTED_PHASES[-1][2] / dt_s))
    plan: list[dict[str, Any]] = []
    previous_capture_count = 0
    for index, (phase, expected) in enumerate(zip(phases, EXPECTED_PHASES, strict=True)):
        if not isinstance(phase, Mapping):
            raise StageBFailureDiagnosticError(f"phase {index} must be an object")
        name, start_s, end_s, g_nap, external = expected
        if phase != {
            "name": name,
            "start_s": start_s,
            "end_s": end_s,
            "g_nap": g_nap,
            "external_current_pA": external,
        }:
            raise StageBFailureDiagnosticError(f"phase {index} changed its filed boundary")
        first_sample_number = max(1, int(round(start_s / dt_s)))
        last_sample_number = int(round(end_s / dt_s)) - (0 if index == len(phases) - 1 else 1)
        capture_count = last_sample_number - first_sample_number + 1
        if capture_count <= 0 or first_sample_number != previous_capture_count + 1:
            raise StageBFailureDiagnosticError("phase samples are not contiguous")
        plan.append(
            {
                "name": name,
                "start_s": start_s,
                "end_s": end_s,
                "first_sample_index": first_sample_number - 1,
                "first_sample_number": first_sample_number,
                "last_sample_index": last_sample_number - 1,
                "last_sample_number": last_sample_number,
                "steps": capture_count,
                "g_nap": g_nap,
                "external_current_pA": external,
            }
        )
        previous_capture_count += capture_count
    if previous_capture_count != total_steps:
        raise StageBFailureDiagnosticError("phase plan does not cover the filed duration")
    return tuple(plan)


def _derive_selection(triage: Mapping[str, Any]) -> list[dict[str, str]]:
    candidates = triage.get("candidates")
    if not isinstance(candidates, list):
        raise StageBFailureDiagnosticError("triage candidates must be a list")
    selected: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise StageBFailureDiagnosticError(f"triage candidate {index} is invalid")
        checks = candidate.get("resolved_checks")
        if not isinstance(checks, list) or any(not isinstance(check, Mapping) for check in checks):
            raise StageBFailureDiagnosticError(
                f"triage candidate {index} has invalid resolved checks"
            )
        unresolved = [check for check in checks if check.get("passed") is None]
        failed = [check for check in checks if check.get("passed") is False]
        if not unresolved and len(failed) == 1 and (
            failed[0].get("gate_id") == "nap-complete-lesion"
            and failed[0].get("metric") == "median_membrane_voltage_change_mV"
        ):
            candidate_id = candidate.get("candidate_id")
            candidate_sha = candidate.get("candidate_sha256")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise StageBFailureDiagnosticError("selected candidate id is invalid")
            selected.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_sha256": _sha256(
                        candidate_sha, f"selected candidate {candidate_id} sha256"
                    ),
                }
            )
    return selected


def _validate_protocol_shape(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("protocol_id") != PROTOCOL_ID
        or protocol.get("status") != "preregistered-engineering-diagnostic-not-executed"
    ):
        raise StageBFailureDiagnosticError("protocol identity or status is invalid")
    boundaries = protocol.get("boundaries")
    if not isinstance(boundaries, Mapping) or (
        boundaries.get("engineering_only") is not True
        or boundaries.get("scientific_verdict") is not None
        or boundaries.get("candidate_promotion_allowed") is not False
        or boundaries.get("parameter_tuning_allowed") is not False
        or boundaries.get("range_expansion_allowed") is not False
        or boundaries.get("source_equivalence_claimed") is not False
    ):
        raise StageBFailureDiagnosticError("protocol changed its scientific boundaries")
    execution = protocol.get("execution")
    if not isinstance(execution, Mapping) or (
        execution.get("backend") != "cupy"
        or execution.get("dt_ms") != DT_MS
        or execution.get("noise") is not False
        or execution.get("plasticity") is not False
        or execution.get("connections") != 0
        or execution.get("identical_initial_state_across_rescue_arms") is not True
        or execution.get("one_continuously_simulated_cell_within_each_arm") is not True
        or tuple(execution.get("rescue_current_pA", ())) != EXPECTED_CURRENTS_PA
        or execution.get("membrane_area_um2") != 2000.0
    ):
        raise StageBFailureDiagnosticError("protocol changed its filed execution settings")
    required = protocol.get("required_channels")
    if not isinstance(required, Mapping) or (
        tuple(required.get("state", ())) != EXPECTED_STATE_CHANNELS
        or tuple(required.get("current_density_uA_per_cm2", ()))
        != EXPECTED_CURRENT_CHANNELS
    ):
        raise StageBFailureDiagnosticError("protocol changed the required diagnostic channels")
    analysis = protocol.get("analysis")
    if not isinstance(analysis, Mapping) or (
        analysis.get("classification_only") is not True
        or analysis.get("no_pass_fail_gate") is not True
    ):
        raise StageBFailureDiagnosticError("protocol changed its analysis boundary")
    return _phase_plan(protocol)


def load_failure_diagnostic(
    protocol_path: str | Path,
    protocol_sha256: str,
    *,
    repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Authenticate the protocol, source evidence, and exact NaP declaration."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    supplied = Path(protocol_path).expanduser()
    protocol_file = (
        supplied if supplied.is_absolute() else root / supplied
    ).resolve()
    try:
        protocol_relative = protocol_file.relative_to(root).as_posix()
    except ValueError as exc:
        raise StageBFailureDiagnosticError("protocol escapes repository root") from exc
    protocol_binding, protocol = _load_json_file(
        root, protocol_relative, protocol_sha256, "diagnostic protocol"
    )
    phase_plan = _validate_protocol_shape(protocol)

    source = protocol.get("source_screen")
    if not isinstance(source, Mapping):
        raise StageBFailureDiagnosticError("protocol source_screen must be an object")
    campaign_ref = source.get("campaign")
    triage_ref = source.get("triage")
    if not isinstance(campaign_ref, Mapping) or not isinstance(triage_ref, Mapping):
        raise StageBFailureDiagnosticError("protocol source bindings are invalid")
    campaign_binding, campaign = _load_json_file(
        root, campaign_ref.get("path"), campaign_ref.get("sha256"), "source campaign"
    )
    triage_binding, triage = _load_json_file(
        root, triage_ref.get("path"), triage_ref.get("sha256"), "source triage"
    )
    _validate_self_digest(campaign, "source campaign")
    _validate_self_digest(triage, "source triage")
    if (
        campaign.get("schema") != CAMPAIGN_SCHEMA
        or campaign.get("candidate_count") != 512
        or campaign.get("engineering_screening_only") is not True
        or campaign.get("scientific_verdict") is not None
    ):
        raise StageBFailureDiagnosticError("source campaign identity or boundary is invalid")
    if (
        triage.get("schema") != TRIAGE_SCHEMA
        or triage.get("candidate_count") != 512
        or triage.get("engineering_screening_only") is not True
        or triage.get("scientific_verdict") is not None
        or triage.get("source_equivalence_claimed") is not False
        or not isinstance(triage.get("campaign"), Mapping)
        or triage["campaign"].get("sha256") != campaign_binding["sha256"]
    ):
        raise StageBFailureDiagnosticError("source triage identity or campaign binding is invalid")
    counts = source.get("observed_counts")
    if not isinstance(counts, Mapping) or (
        counts.get("candidate_count") != triage.get("candidate_count")
        or counts.get("engineering_fail")
        != triage.get("classification_counts", {}).get("engineering_fail")
        or counts.get("engineering_inconclusive")
        != triage.get("classification_counts", {}).get("engineering_inconclusive")
        or counts.get("engineering_pass")
        != triage.get("classification_counts", {}).get("engineering_pass", 0)
    ):
        raise StageBFailureDiagnosticError("protocol observed counts do not match triage")

    selected = _derive_selection(triage)
    selection = protocol.get("selection")
    if not isinstance(selection, Mapping) or (
        selection.get("post_hoc_disclosed") is not True
        or selection.get("candidate_count") != len(selected)
        or selection.get("candidates") != selected
        or len(selected) != 9
    ):
        raise StageBFailureDiagnosticError(
            "protocol selection does not match deterministic triage rederivation"
        )

    declarations = campaign.get("declarations")
    nap_declarations = [
        declaration
        for declaration in declarations if isinstance(declaration, Mapping)
        and declaration.get("arm") == "nap_lesion"
    ] if isinstance(declarations, list) else []
    if len(nap_declarations) != 1:
        raise StageBFailureDiagnosticError("campaign must bind exactly one NaP declaration")
    declaration_ref = nap_declarations[0]
    from research.runners.v14_stageB_batched_physiology import (
        StageBBatchedPhysiologyError,
        load_batch_declaration,
    )

    try:
        declaration = load_batch_declaration(
            declaration_ref.get("path"),
            declaration_ref.get("sha256"),
            repository_root=root,
        )
    except (OSError, TypeError, ValueError, StageBBatchedPhysiologyError) as exc:
        raise StageBFailureDiagnosticError(
            f"NaP declaration authentication failed: {exc}"
        ) from exc
    if (
        declaration.get("arm") != "nap_lesion"
        or declaration.get("self_sha256") != declaration_ref.get("declaration_sha256")
        or len(declaration.get("candidates", ())) != declaration_ref.get("candidate_count")
    ):
        raise StageBFailureDiagnosticError("NaP declaration identity is invalid")
    declared_by_id = {
        candidate["candidate_id"]: candidate for candidate in declaration["candidates"]
    }
    if len(declared_by_id) != len(declaration["candidates"]):
        raise StageBFailureDiagnosticError("NaP declaration contains duplicate candidates")
    filtered: list[dict[str, Any]] = []
    for identity in selected:
        candidate = declared_by_id.get(identity["candidate_id"])
        if candidate is None or candidate["candidate_sha256"] != identity["candidate_sha256"]:
            raise StageBFailureDiagnosticError(
                "selected candidate is absent or changed in the NaP declaration"
            )
        filtered.append(candidate)

    return {
        "repository_root": root,
        "protocol": protocol,
        "protocol_binding": protocol_binding,
        "campaign_binding": campaign_binding,
        "triage_binding": triage_binding,
        "declaration": declaration,
        "declaration_binding": {
            "path": declaration["path"],
            "sha256": declaration["sha256"],
            "self_sha256": declaration["self_sha256"],
        },
        "candidates": filtered,
        "selection": selected,
        "phase_plan": phase_plan,
    }


def current_density_uA_per_cm2(current_pA: float, membrane_area_um2: float) -> float:
    """Convert whole-cell current to the density used by the HH equation."""

    if isinstance(current_pA, bool) or not isinstance(current_pA, (int, float)):
        raise StageBFailureDiagnosticError("current_pA must be numeric")
    if (
        isinstance(membrane_area_um2, bool)
        or not isinstance(membrane_area_um2, (int, float))
        or not math.isfinite(float(membrane_area_um2))
        or float(membrane_area_um2) <= 0.0
    ):
        raise StageBFailureDiagnosticError("membrane_area_um2 must be finite and positive")
    if not math.isfinite(float(current_pA)):
        raise StageBFailureDiagnosticError("current_pA must be finite")
    return 100.0 * float(current_pA) / float(membrane_area_um2)


def bridge_external_current_numeric(current_pA: float, membrane_area_um2: float) -> float:
    """Convert pA to the bridge value whose HH path applies a 1e-6 scale."""

    return current_density_uA_per_cm2(current_pA, membrane_area_um2) * 1.0e6


def _runtime_components() -> SimpleNamespace:
    from research.runners.v14_stageB_batched_physiology import (
        _binding_provenance,
        _build_config,
        _runtime_components as stageb_runtime_components,
        _synchronize,
        _to_host,
    )

    runtime = stageb_runtime_components()
    from sim.kernels import (
        fused_snr_packet_diagnostic_currents,
        prepare_fused_snr_diagnostic_capture_into,
    )
    from tools.diagnostic_trace import save_diagnostic_trace

    runtime.build_config = _build_config
    runtime.binding_provenance = _binding_provenance
    runtime.synchronize = _synchronize
    runtime.to_host = _to_host
    runtime.prepare_capture = prepare_fused_snr_diagnostic_capture_into
    runtime.diagnostic_currents = fused_snr_packet_diagnostic_currents
    runtime.save_trace = save_diagnostic_trace
    return runtime


def _initial_state_digest(bridge: Any, runtime: SimpleNamespace) -> str:
    names = [field for _, field in _CAPTURE_STATE_FIELDS[1:-1]]
    names.extend(
        [
            "cp_hh_g_Na_max",
            "cp_hh_g_K_max",
            "cp_hh_g_L",
            "cp_snr_g_nalcn_max",
            "cp_snr_g_nap_max",
            "cp_snr_g_ca_max",
            "cp_snr_g_sk_max",
            "cp_snr_g_h_max",
        ]
    )
    digest = hashlib.sha256()
    for name in names:
        value = runtime.to_host(runtime.xp, getattr(bridge, name))
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _diagnostic_current_arguments(bridge: Any, states: Mapping[str, Any]) -> tuple[Any, ...]:
    parameters = bridge.snr_packet_kernel_parameters
    return (
        states["pre_update_voltage_mV"],
        states["fast_na_activation"],
        states["fast_na_inactivation"],
        states["fast_k_activation"],
        states["nap_activation"],
        states["nap_inactivation"],
        states["cav22_activation"],
        states["cav22_inactivation"],
        states["sk_activation"],
        states["hcn_activation"],
        bridge.cp_hh_g_Na_max,
        bridge.cp_hh_g_K_max,
        bridge.cp_hh_g_L,
        bridge.cp_snr_g_nalcn_max,
        bridge.cp_snr_g_nap_max,
        bridge.cp_snr_g_ca_max,
        bridge.cp_snr_g_sk_max,
        bridge.cp_snr_g_h_max,
        bridge.cp_hh_E_Na,
        bridge.cp_hh_E_K,
        bridge.cp_hh_E_L,
        parameters["E_nalcn_mv"],
        parameters["E_ca_mv"],
        parameters["E_hcn_mv"],
        parameters["cav22_activation_power"],
    )


def _capture_arguments(
    bridge: Any,
    pre_voltage: Any,
    outputs: Mapping[str, Any],
) -> tuple[Any, ...]:
    sources = [pre_voltage]
    sources.extend(getattr(bridge, field) for _, field in _CAPTURE_STATE_FIELDS[1:])
    destinations = [outputs[name] for name, _ in _CAPTURE_STATE_FIELDS]
    return (*sources, *destinations)


def _phase_current(phase: Mapping[str, Any], arm_current_pA: float) -> float:
    value = phase["external_current_pA"]
    return arm_current_pA if value == "arm_value" else float(value)


def _run_phase_chunks(
    bridge: Any,
    runtime: SimpleNamespace,
    phase: Mapping[str, Any],
    arm_current_pA: float,
    area_um2: np.ndarray,
    packet_nap: Any,
    chunk_steps: int,
    previous_post: Any,
    capture_executor: Any | None,
) -> tuple[dict[str, list[np.ndarray]], Any, Any, dict[str, float]]:
    xp = runtime.xp
    if phase["g_nap"] == "packet_value":
        bridge.cp_snr_g_nap_max[...] = packet_nap
    else:
        bridge.cp_snr_g_nap_max[...] = xp.float32(0.0)
    current_pA = _phase_current(phase, arm_current_pA)
    bridge_values = np.array(
        [bridge_external_current_numeric(current_pA, area) for area in area_um2],
        dtype=np.float32,
    )
    bridge.cp_external_input_current[...] = xp.asarray(bridge_values)
    exact_external_density = (
        bridge.cp_external_input_current * xp.float32(1.0e-6)
    )
    runtime.synchronize(xp)

    host_chunks: dict[str, list[np.ndarray]] = {
        name: [] for name, _ in _CAPTURE_STATE_FIELDS
    }
    host_chunks.update({name: [] for name in _CURRENT_NAMES})
    host_chunks.update(
        {
            "i_external": [],
            "snr_current_balance_residual": [],
            "membrane_current_balance_residual": [],
        }
    )
    maxima = {"snr": 0.0, "membrane": 0.0}
    for chunk_start in range(0, phase["steps"], chunk_steps):
        width = min(chunk_steps, phase["steps"] - chunk_start)
        device_states: dict[str, Any] = {}
        for name, _ in _CAPTURE_STATE_FIELDS:
            dtype = xp.bool_ if name == "spikes" else xp.float32
            device_states[name] = xp.empty((width, len(area_um2)), dtype=dtype)
        for offset in range(width):
            pre_voltage = previous_post
            bridge._run_one_simulation_step()
            row_outputs = {name: value[offset] for name, value in device_states.items()}
            arguments = _capture_arguments(bridge, pre_voltage, row_outputs)
            if capture_executor is None:
                capture_executor = runtime.prepare_capture(arguments)
            else:
                capture_executor(*arguments)
            previous_post = row_outputs["post_update_voltage_mV"]

        currents = runtime.diagnostic_currents(
            *_diagnostic_current_arguments(bridge, device_states)
        )
        device_currents = dict(zip(_CURRENT_NAMES, currents, strict=True))
        i_external = xp.broadcast_to(
            exact_external_density[None, :],
            device_states["pre_update_voltage_mV"].shape,
        )
        snr_total = (
            device_currents["i_nalcn"]
            + device_currents["i_nap"]
            + device_currents["i_cav22"]
            + device_currents["i_sk"]
            + device_currents["i_hcn"]
        )
        snr_residual = device_states["snr_effective_input_scratch"] - (
            i_external - snr_total
        )
        fast_total = (
            device_currents["i_fast_na"]
            + device_currents["i_fast_k"]
            + device_currents["i_leak"]
        )
        membrane_residual = (
            bridge.cp_hh_C_m[None, :]
            * (
                device_states["post_update_voltage_mV"]
                - device_states["pre_update_voltage_mV"]
            )
            / xp.float32(DT_MS)
            - (device_states["snr_effective_input_scratch"] - fast_total)
        )
        runtime.synchronize(xp)
        snr_host = runtime.to_host(xp, snr_residual).astype(np.float32, copy=False)
        membrane_host = runtime.to_host(xp, membrane_residual).astype(np.float32, copy=False)
        maxima["snr"] = max(maxima["snr"], float(np.max(np.abs(snr_host))))
        maxima["membrane"] = max(
            maxima["membrane"], float(np.max(np.abs(membrane_host)))
        )
        for name, values in device_states.items():
            host_chunks[name].append(runtime.to_host(xp, values).copy())
        for name, values in device_currents.items():
            host_chunks[name].append(
                runtime.to_host(xp, values).astype(np.float32, copy=False).copy()
            )
        host_chunks["i_external"].append(
            runtime.to_host(xp, i_external).astype(np.float32, copy=False).copy()
        )
        host_chunks["snr_current_balance_residual"].append(snr_host.copy())
        host_chunks["membrane_current_balance_residual"].append(membrane_host.copy())
        previous_post = previous_post.copy()

    if maxima["snr"] > 5.0e-4 or maxima["membrane"] > 5.0e-3:
        raise StageBFailureDiagnosticError(
            "diagnostic current-balance residual exceeded the engineering tolerance: "
            f"snr={maxima['snr']}, membrane={maxima['membrane']}"
        )
    return host_chunks, previous_post, capture_executor, maxima


def _archive_token(current_pA: float) -> str:
    return "zero-pa" if current_pA == 0.0 else f"minus-{abs(int(current_pA))}-pa"


def _output_path(root: Path, supplied: str | Path) -> tuple[str, Path]:
    path = Path(supplied).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise StageBFailureDiagnosticError("output must be inside repository root") from exc
    if path.is_symlink():
        raise StageBFailureDiagnosticError("output must not be a symbolic link")
    if path.exists():
        raise StageBFailureDiagnosticError("refusing to replace an existing receipt")
    return relative, path


def _publish_receipt_once(destination: Path, result: Mapping[str, Any]) -> None:
    data = _canonical_bytes(result)
    temporary: Path | None = None
    linked = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        linked = True
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if linked:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_failure_diagnostic(
    protocol_path: str | Path,
    protocol_sha256: str,
    output: str | Path,
    *,
    repository_root: str | Path = ROOT,
    chunk_steps: int = DEFAULT_CHUNK_STEPS,
    _runtime: SimpleNamespace | None = None,
) -> dict[str, Any]:
    """Execute the four fixed diagnostic arms and publish authenticated evidence."""

    if isinstance(chunk_steps, bool) or not isinstance(chunk_steps, int):
        raise StageBFailureDiagnosticError("chunk_steps must be an integer")
    if not 1 <= chunk_steps <= MAX_CHUNK_STEPS:
        raise StageBFailureDiagnosticError(
            f"chunk_steps must be in [1, {MAX_CHUNK_STEPS}]"
        )
    loaded = load_failure_diagnostic(
        protocol_path, protocol_sha256, repository_root=repository_root
    )
    root = loaded["repository_root"]
    receipt_relative, destination = _output_path(root, output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    runtime = _runtime if _runtime is not None else _runtime_components()
    xp = runtime.xp
    candidates = loaded["candidates"]
    total_steps = sum(phase["steps"] for phase in loaded["phase_plan"])
    traces: list[dict[str, Any]] = []
    created_archives: list[Path] = []
    expected_initial_digest: str | None = None
    expected_manifest_digest: str | None = None
    arm_provenance: list[dict[str, Any]] = []
    nap_firing_attribution: dict[str, Any] | None = None

    try:
        for arm_current_pA in EXPECTED_CURRENTS_PA:
            config = runtime.build_config(candidates, total_steps, runtime)
            bridge = runtime.SimulationBridge(
                core_config=config,
                viz_config=runtime.VisualizationConfig(),
                runtime_state=runtime.RuntimeState(),
                gpu_config=runtime.GPUConfig(
                    enable_profiling=False,
                    stats_sync_interval_steps=total_steps + 1,
                ),
                simulation_source_root=str(root),
            )
            try:
                bridge._initialize_simulation_data()
                if not bridge.is_initialized:
                    raise StageBFailureDiagnosticError("diagnostic bridge initialization failed")
                dispatch_check = getattr(bridge, "_snr_direct_outputs_can_dispatch", None)
                if not callable(dispatch_check) or not dispatch_check(config):
                    raise StageBFailureDiagnosticError(
                        "diagnostic bridge cannot dispatch packet SNr direct outputs"
                    )
                bindings = bridge.snr_packet_bindings
                if set(bindings) != {candidate["region_name"] for candidate in candidates}:
                    raise StageBFailureDiagnosticError(
                        "diagnostic runtime bindings do not match selected candidates"
                    )
                binding_rows = []
                area_um2 = []
                for candidate in candidates:
                    binding = bindings[candidate["region_name"]]
                    if (
                        binding.packet_path != candidate["packet"]["path"]
                        or binding.packet_file_sha256 != candidate["packet"]["sha256"]
                        or binding.authority_policy_sha256 != candidate["policy"]["sha256"]
                    ):
                        raise StageBFailureDiagnosticError(
                            f"runtime binding changed candidate {candidate['candidate_id']}"
                        )
                    area = float(binding.runtime_parameters.geometry.membrane_area_um2)
                    if area != float(loaded["protocol"]["execution"]["membrane_area_um2"]):
                        raise StageBFailureDiagnosticError(
                            "candidate membrane area changed the filed current conversion"
                        )
                    area_um2.append(area)
                    binding_rows.append(runtime.binding_provenance(binding))
                area_array = np.asarray(area_um2, dtype=np.float64)
                manifest_digest = _digest_bytes(
                    runtime.runtime_binding_manifest_bytes(bindings)
                )
                initial_digest = _initial_state_digest(bridge, runtime)
                if expected_initial_digest is None:
                    expected_initial_digest = initial_digest
                    expected_manifest_digest = manifest_digest
                elif (
                    initial_digest != expected_initial_digest
                    or manifest_digest != expected_manifest_digest
                ):
                    raise StageBFailureDiagnosticError(
                        "fresh rescue arms did not start from identical deterministic state"
                    )

                packet_nap = bridge.cp_snr_g_nap_max.copy()
                previous_post = bridge.cp_membrane_potential_v.copy()
                capture_executor = None
                arm_chunks: dict[str, list[np.ndarray]] = {}
                arm_maxima = {"snr": 0.0, "membrane": 0.0}
                phase_receipts = []
                for phase in loaded["phase_plan"]:
                    chunks, previous_post, capture_executor, maxima = _run_phase_chunks(
                        bridge,
                        runtime,
                        phase,
                        arm_current_pA,
                        area_array,
                        packet_nap,
                        chunk_steps,
                        previous_post,
                        capture_executor,
                    )
                    for name, values in chunks.items():
                        arm_chunks.setdefault(name, []).extend(values)
                    arm_maxima["snr"] = max(arm_maxima["snr"], maxima["snr"])
                    arm_maxima["membrane"] = max(
                        arm_maxima["membrane"], maxima["membrane"]
                    )
                    phase_receipts.append(
                        {
                            **dict(phase),
                            "external_current_pA": _phase_current(
                                phase, arm_current_pA
                            ),
                        }
                    )
                runtime.synchronize(xp)
            finally:
                bridge.clear_simulation_state_and_gpu_memory()

            arrays = {
                name: np.concatenate(chunks, axis=0)
                for name, chunks in arm_chunks.items()
            }
            if any(array.shape != (total_steps, len(candidates)) for array in arrays.values()):
                raise StageBFailureDiagnosticError("captured diagnostic shape is incomplete")
            if arm_current_pA == 0.0:
                baseline_phase = loaded["phase_plan"][0]
                lesion_phase = loaded["phase_plan"][1]
                baseline_stop = int(baseline_phase["last_sample_index"]) + 1
                baseline_start = max(
                    int(baseline_phase["first_sample_index"]), baseline_stop - 10_000
                )
                lesion_start = int(lesion_phase["first_sample_index"])
                lesion_stop = int(lesion_phase["last_sample_index"]) + 1
                baseline_duration_s = (baseline_stop - baseline_start) * DT_MS / 1000.0
                lesion_duration_s = (lesion_stop - lesion_start) * DT_MS / 1000.0
                baseline_rates = (
                    arrays["spikes"][baseline_start:baseline_stop].sum(axis=0)
                    / baseline_duration_s
                ).astype(np.float64)
                lesion_rates = (
                    arrays["spikes"][lesion_start:lesion_stop].sum(axis=0)
                    / lesion_duration_s
                ).astype(np.float64)
                baseline_median = float(np.median(baseline_rates))
                lesion_median = float(np.median(lesion_rates))
                nap_firing_attribution = {
                    "label": "sustained firing attributable to NaP presence",
                    "baseline_window_s": baseline_duration_s,
                    "lesion_window_s": lesion_duration_s,
                    "candidate_baseline_rate_hz": baseline_rates.tolist(),
                    "candidate_nap_lesion_rate_hz": lesion_rates.tolist(),
                    "cohort_baseline_median_rate_hz": baseline_median,
                    "cohort_nap_lesion_median_rate_hz": lesion_median,
                    "attributable_fraction": attributable_to(
                        "sustained firing attributable to NaP presence",
                        baseline_median,
                        lesion_median,
                    ),
                }
            time_s = np.arange(1, total_steps + 1, dtype="<f8") * (DT_MS / 1000.0)
            arm_token = _archive_token(arm_current_pA)
            for candidate_index, candidate in enumerate(candidates):
                archive = destination.with_name(
                    f"{destination.stem}.{candidate['candidate_id']}.{arm_token}.diagnostic.zip"
                )
                channels = {
                    _ARCHIVE_CHANNEL_NAMES.get(name, name): np.ascontiguousarray(
                        array[:, candidate_index]
                    ).astype(
                        "|b1" if name == "spikes" else "<f4", copy=False
                    )
                    for name, array in arrays.items()
                }
                archive_sha = runtime.save_trace(archive, time_s, channels)
                created_archives.append(archive)
                traces.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "candidate_sha256": candidate["candidate_sha256"],
                        "rescue_current_pA": arm_current_pA,
                        "diagnostic_trace": {
                            "path": archive.relative_to(root).as_posix(),
                            "sha256": archive_sha,
                            "sample_count": total_steps,
                        },
                    }
                )
            arm_provenance.append(
                {
                    "rescue_current_pA": arm_current_pA,
                    "fresh_bridge": True,
                    "initial_state_sha256": initial_digest,
                    "runtime_binding_manifest_sha256": manifest_digest,
                    "runtime_bindings": binding_rows,
                    "phases": phase_receipts,
                    "max_abs_snr_current_balance_residual_uA_per_cm2": arm_maxima["snr"],
                    "max_abs_membrane_current_balance_residual_uA_per_cm2": arm_maxima[
                        "membrane"
                    ],
                }
            )

        if nap_firing_attribution is None:
            raise StageBFailureDiagnosticError("NaP firing attribution was not measured")
        body = {
            "schema": RECEIPT_SCHEMA,
            "process_status": "completed",
            "backend": "cupy",
            "device": "cuda",
            "engineering_diagnostic_only": True,
            "scientific_verdict": None,
            "candidate_promotion_allowed": False,
            "parameter_tuning_allowed": False,
            "source_equivalence_claimed": False,
            "protocol": loaded["protocol_binding"],
            "source_campaign": loaded["campaign_binding"],
            "source_triage": loaded["triage_binding"],
            "source_nap_declaration": loaded["declaration_binding"],
            "selection": loaded["selection"],
            "execution": {
                "candidate_count": len(candidates),
                "arm_count": len(EXPECTED_CURRENTS_PA),
                "trace_count": len(traces),
                "dt_ms": DT_MS,
                "total_steps_per_arm": total_steps,
                "chunk_steps": chunk_steps,
                "fresh_identical_bridge_per_arm": True,
                "adaptive_decisions": False,
                "connections": 0,
                "noise": False,
                "plasticity": False,
                "state_capture_launches_per_step": 1,
                "current_decomposition": "device-side chunk-vectorized exact-state arithmetic",
                "current_units": "uA/cm^2; outward-positive",
                "sample_semantics": (
                    "pre-update voltage plus gates and currents used by that update, "
                    "paired with post-update voltage and spike state"
                ),
                "protocol_to_archive_channel_names": {
                    name: _ARCHIVE_CHANNEL_NAMES.get(name, name)
                    for name in (*EXPECTED_STATE_CHANNELS, *EXPECTED_CURRENT_CHANNELS)
                },
            },
            "attribution": nap_firing_attribution,
            "arms": arm_provenance,
            "traces": traces,
            "provenance": {
                "runner": "research/runners/v14_stageB_failure_diagnostic.py",
                "receipt_path": receipt_relative,
            },
        }
        result = {**body, "sha256": _digest(body)}
        _publish_receipt_once(destination, result)
        return result
    except Exception:
        for archive in created_archives:
            archive.unlink(missing_ok=True)
        raise


def _validation_summary(loaded: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "v14-snr-stageB-failure-diagnostic-validation-v1",
        "process_status": "validated_not_executed",
        "engineering_diagnostic_only": True,
        "scientific_verdict": None,
        "protocol": loaded["protocol_binding"],
        "source_campaign": loaded["campaign_binding"],
        "source_triage": loaded["triage_binding"],
        "source_nap_declaration": loaded["declaration_binding"],
        "selection": loaded["selection"],
        "rescue_current_pA": list(EXPECTED_CURRENTS_PA),
        "phases": [dict(phase) for phase in loaded["phase_plan"]],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--output")
    parser.add_argument("--chunk-steps", type=int, default=DEFAULT_CHUNK_STEPS)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.validate_only:
            if args.output is not None:
                raise StageBFailureDiagnosticError(
                    "--validate-only does not accept --output"
                )
            loaded = load_failure_diagnostic(
                args.protocol,
                args.protocol_sha256,
                repository_root=args.repository_root,
            )
            print(_canonical_bytes(_validation_summary(loaded)).decode("ascii"))
        else:
            if args.output is None:
                raise StageBFailureDiagnosticError(
                    "execution requires --output for the authenticated receipt"
                )
            result = run_failure_diagnostic(
                args.protocol,
                args.protocol_sha256,
                args.output,
                repository_root=args.repository_root,
                chunk_steps=args.chunk_steps,
            )
            print(_canonical_bytes(result).decode("ascii"))
    except (OSError, TypeError, ValueError, StageBFailureDiagnosticError) as exc:
        parser.exit(2, f"Stage B failure diagnostic error: {exc}\n")
    return 0


__all__ = [
    "StageBFailureDiagnosticError",
    "bridge_external_current_numeric",
    "current_density_uA_per_cm2",
    "load_failure_diagnostic",
    "run_failure_diagnostic",
]


if __name__ == "__main__":
    raise SystemExit(main())
