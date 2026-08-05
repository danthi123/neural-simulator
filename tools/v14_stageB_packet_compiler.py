"""Deterministic, non-authorizing compiler for V14 Stage B SNr packets.

The compiler can describe the exact authority artifact it expects, but cannot
write that artifact, adjudicate a packet, issue a policy, or seal a packet.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sim.snr_executable_packet import (
    PARAMETER_SCHEMA,
    SCHEMA_VERSION,
    PacketError,
    canonical_bytes,
    canonical_decimal,
    claim_document,
    load_packet,
)


TEMPLATE_SCHEMA = "v14-snr-stageB-packet-template-v1"
CANDIDATE_SCHEMA = "sim-adaptive-candidate-v1"
REQUEST_SCHEMA = "v14-snr-stageB-compilation-request-v1"
EVIDENCE_SCHEMA = "v14-snr-stageB-evidence-claims-v1"
AUTHORITY_SCHEMA = "v14-snr-stageB-authority-claims-v1"


class StageBPacketCompilerError(ValueError):
    """Raised when compilation cannot produce an exact structural packet."""


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(canonical_bytes(value))


def _sha256(value: Any, context: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise StageBPacketCompilerError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _text(value: Any, context: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(ord(character) > 127 for character in value)):
        raise StageBPacketCompilerError(f"{context} must be nonempty trimmed ASCII text")
    return value


def _decimal_string(value: Any, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise StageBPacketCompilerError(f"{context} must be a finite JSON number or decimal string")
    if isinstance(value, float) and not math.isfinite(value):
        raise StageBPacketCompilerError(f"{context} must be finite")
    if isinstance(value, str):
        try:
            return canonical_decimal(value, context)
        except PacketError as exc:
            raise StageBPacketCompilerError(str(exc)) from exc
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as exc:
        raise StageBPacketCompilerError(f"{context} is not a decimal number") from exc
    if not decimal.is_finite():
        raise StageBPacketCompilerError(f"{context} must be finite")
    if decimal == 0:
        rendered = "0"
    else:
        normalized = decimal.normalize()
        adjusted = normalized.adjusted()
        rendered = format(normalized, "f") if -6 <= adjusted <= 20 else format(normalized, "e")
        rendered = rendered.replace("e+", "e")
    try:
        return canonical_decimal(rendered, context)
    except PacketError as exc:
        raise StageBPacketCompilerError(str(exc)) from exc


def _load_pinned_json(path: str | Path, expected_sha256: str, context: str) -> tuple[Path, dict[str, Any]]:
    _sha256(expected_sha256, f"{context} expected digest")
    source = Path(path).resolve()
    try:
        raw = source.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBPacketCompilerError(f"cannot load {context}: {exc}") from exc
    if _digest_bytes(raw) != expected_sha256:
        raise StageBPacketCompilerError(f"{context} digest does not match")
    try:
        canonical = canonical_bytes(value)
    except PacketError as exc:
        raise StageBPacketCompilerError(f"{context} is not canonical JSON: {exc}") from exc
    if raw != canonical:
        raise StageBPacketCompilerError(f"{context} bytes are not canonical JSON")
    if not isinstance(value, dict):
        raise StageBPacketCompilerError(f"{context} must contain an object")
    return source, value


def _candidate(document: Any) -> dict[str, Any]:
    if not isinstance(document, Mapping) or set(document) != {"schema", "candidate_id", "parameters"}:
        raise StageBPacketCompilerError("candidate document has an invalid shape")
    if document.get("schema") != CANDIDATE_SCHEMA:
        raise StageBPacketCompilerError("candidate document has the wrong schema")
    identifier = _text(document.get("candidate_id"), "candidate_id")
    parameters = document.get("parameters")
    if not isinstance(parameters, Mapping):
        raise StageBPacketCompilerError("candidate parameters must be an object")
    return {"schema": CANDIDATE_SCHEMA, "candidate_id": identifier, "parameters": dict(parameters)}


def _template(document: Any) -> dict[str, Any]:
    if (not isinstance(document, Mapping)
            or set(document) != {"schema", "template_id", "parameter_leaves"}
            or document.get("schema") != TEMPLATE_SCHEMA):
        raise StageBPacketCompilerError("packet template has an invalid shape or schema")
    _text(document.get("template_id"), "template_id")
    groups = document.get("parameter_leaves")
    if not isinstance(groups, Mapping) or set(groups) != set(PARAMETER_SCHEMA):
        raise StageBPacketCompilerError("packet template does not contain the exact parameter groups")
    for group, schema in PARAMETER_SCHEMA.items():
        leaves = groups.get(group)
        if not isinstance(leaves, Mapping) or set(leaves) != set(schema):
            raise StageBPacketCompilerError(f"packet template group {group!r} is incomplete or widened")
    return dict(document)


def _leaf_value(
    group: str,
    parameter: str,
    specification: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any],
    used_keys: set[str],
) -> tuple[str, str, dict[str, Any], str, str]:
    common = {"mode", "unit", "uncertainty", "evidence", "authority"}
    mode = specification.get("mode")
    expected = common | ({"value"} if mode == "fixed" else {"candidate_key", "bounds", "transform"})
    if mode not in {"fixed", "searched"} or set(specification) != expected:
        raise StageBPacketCompilerError(f"template leaf {group}.{parameter} has an invalid shape")
    unit = specification.get("unit")
    if unit not in PARAMETER_SCHEMA[group][parameter]:
        raise StageBPacketCompilerError(f"template leaf {group}.{parameter} has the wrong unit")
    uncertainty = specification.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise StageBPacketCompilerError(f"template leaf {group}.{parameter} has no uncertainty contract")
    evidence = specification.get("evidence")
    authority = specification.get("authority")
    if mode == "searched":
        if evidence != "derived" or authority != "project_decision":
            raise StageBPacketCompilerError(
                f"searched leaf {group}.{parameter} must remain derived/project_decision"
            )
        key = _text(specification.get("candidate_key"), f"{group}.{parameter}.candidate_key")
        if key in used_keys or key not in candidate_parameters:
            raise StageBPacketCompilerError(f"searched candidate key {key!r} is missing or duplicated")
        used_keys.add(key)
        value = _decimal_string(candidate_parameters[key], f"candidate.parameters.{key}")
        bounds = specification.get("bounds")
        transform = specification.get("transform")
        if not isinstance(bounds, Mapping) or set(bounds) != {"low", "high"}:
            raise StageBPacketCompilerError(f"searched leaf {group}.{parameter} has invalid bounds")
        low = Decimal(_decimal_string(bounds["low"], f"{group}.{parameter}.bounds.low"))
        high = Decimal(_decimal_string(bounds["high"], f"{group}.{parameter}.bounds.high"))
        numeric = Decimal(value)
        if low > high or numeric < low or numeric > high:
            raise StageBPacketCompilerError(f"candidate value for {key!r} is outside sealed bounds")
        if transform not in {"linear", "log"} or (transform == "log" and low <= 0):
            raise StageBPacketCompilerError(f"searched leaf {group}.{parameter} has invalid transform")
    else:
        value = _decimal_string(specification.get("value"), f"{group}.{parameter}.value")
        if evidence not in {"measured", "model_prior", "derived"}:
            raise StageBPacketCompilerError(f"fixed leaf {group}.{parameter} has invalid evidence")
        compatible = {"measured": "primary_source", "model_prior": "model_source", "derived": "project_decision"}
        if authority != compatible[evidence]:
            raise StageBPacketCompilerError(f"fixed leaf {group}.{parameter} has incompatible authority")
    return value, str(unit), dict(uncertainty), str(evidence), str(authority)


def expected_authority_claims(
    *, template_id: str, template_sha256: str, candidate_sha256: str,
    claims: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe verifier output without creating or authorizing it."""
    return {
        "schema": AUTHORITY_SCHEMA,
        "template": {"template_id": template_id, "sha256": template_sha256},
        "candidate_sha256": candidate_sha256,
        "authorized_claims": claims,
    }


def compile_documents(
    template: Mapping[str, Any], candidate: Mapping[str, Any], *, template_sha256: str,
) -> dict[str, dict[str, Any]]:
    """Return the four compiler-owned documents plus the unwritten authority expectation."""
    _sha256(template_sha256, "template_sha256")
    checked_template = _template(template)
    checked_candidate = _candidate(candidate)
    candidate_raw = canonical_bytes(checked_candidate)
    candidate_sha = _digest_bytes(candidate_raw)
    packet_id = f"{checked_template['template_id']}--{checked_candidate['candidate_id']}--{candidate_sha[:16]}"
    claims: dict[str, dict[str, Any]] = {}
    leaf_inputs: dict[str, dict[str, tuple[str, str, dict[str, Any], str, str]]] = {}
    used_keys: set[str] = set()
    for group in PARAMETER_SCHEMA:
        claims[group] = {}
        leaf_inputs[group] = {}
        for parameter in PARAMETER_SCHEMA[group]:
            parts = _leaf_value(
                group, parameter, checked_template["parameter_leaves"][group][parameter],
                checked_candidate["parameters"], used_keys,
            )
            value, unit, uncertainty, evidence, authority = parts
            claims[group][parameter] = claim_document(
                packet_id, group, parameter, value, unit, uncertainty, evidence, authority
            )
            leaf_inputs[group][parameter] = parts
    if used_keys != set(checked_candidate["parameters"]):
        extra = sorted(set(checked_candidate["parameters"]) - used_keys)
        raise StageBPacketCompilerError(f"candidate has unfiled parameters: {extra}")

    evidence_document = {
        "schema": EVIDENCE_SCHEMA,
        "template": {"template_id": checked_template["template_id"], "sha256": template_sha256},
        "candidate_sha256": candidate_sha,
        "claims": claims,
    }
    evidence_sha = _digest(evidence_document)
    authority_document = expected_authority_claims(
        template_id=checked_template["template_id"], template_sha256=template_sha256,
        candidate_sha256=candidate_sha, claims=claims,
    )
    authority_sha = _digest(authority_document)
    groups: dict[str, dict[str, Any]] = {}
    for group in PARAMETER_SCHEMA:
        groups[group] = {}
        for parameter in PARAMETER_SCHEMA[group]:
            value, unit, uncertainty, evidence, authority = leaf_inputs[group][parameter]
            claim_sha = _digest(claims[group][parameter])
            groups[group][parameter] = {
                "value": value,
                "unit": unit,
                "uncertainty": uncertainty,
                "evidence": {
                    "kind": evidence, "artifact_path": "evidence-claims.json",
                    "artifact_sha256": evidence_sha,
                    "locator": f"json-pointer:/claims/{group}/{parameter}",
                    "claim_sha256": claim_sha,
                },
                "authority": {
                    "kind": authority, "artifact_path": "authority-claims.json",
                    "artifact_sha256": authority_sha,
                    "locator": f"json-pointer:/authorized_claims/{group}/{parameter}",
                    "claim_sha256": claim_sha,
                },
            }
    packet = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet_id,
        "state": "STRUCTURAL",
        "groups": groups,
        "adjudication": None,
    }
    try:
        load_packet(packet, artifact_root=Path.cwd())
    except PacketError as exc:
        raise StageBPacketCompilerError(f"compiled packet is structurally invalid: {exc}") from exc
    request = {
        "schema": REQUEST_SCHEMA,
        "template": {"template_id": checked_template["template_id"], "sha256": template_sha256},
        "candidate_sha256": candidate_sha,
        "evidence_claims_sha256": evidence_sha,
        "expected_authority_claims_sha256": authority_sha,
        "structural_packet_sha256": _digest(packet),
        "compiler_authority": "none",
        "forbidden_outputs": [
            "authority-claims.json", "adjudication.json", "authority-policy.json",
            "packet.sealed.json", "candidate-release.json",
        ],
    }
    return {
        "candidate.json": checked_candidate,
        "compilation-request.json": request,
        "evidence-claims.json": evidence_document,
        "packet.structural.json": packet,
        "expected_authority_claims": authority_document,
    }


def compile_candidate(
    template_path: str | Path, template_sha256: str, candidate_path: str | Path,
    candidate_sha256: str, output_dir: str | Path, *, repository_root: str | Path,
) -> dict[str, Any]:
    """Validate pinned inputs and write only compiler-owned artifacts."""
    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise StageBPacketCompilerError("repository_root must be a directory")
    template_source, template = _load_pinned_json(template_path, template_sha256, "packet template")
    candidate_source, candidate = _load_pinned_json(candidate_path, candidate_sha256, "candidate document")
    for path, context in ((template_source, "packet template"), (candidate_source, "candidate document")):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise StageBPacketCompilerError(f"{context} must be inside repository_root") from exc
    documents = compile_documents(template, candidate, template_sha256=template_sha256)
    destination = Path(output_dir).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:
        raise StageBPacketCompilerError("compiler output must be inside repository_root") from exc
    if destination.exists():
        raise StageBPacketCompilerError("compiler output directory must not already exist")
    destination.mkdir(parents=True)
    for name in ("candidate.json", "compilation-request.json", "evidence-claims.json", "packet.structural.json"):
        path = destination / name
        with path.open("xb") as handle:
            handle.write(canonical_bytes(documents[name]))
    return documents["compilation-request.json"]


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_candidate(
            args.template, args.template_sha256, args.candidate, args.candidate_sha256,
            args.output_dir, repository_root=args.repository_root,
        )
    except (OSError, StageBPacketCompilerError, ValueError, TypeError) as exc:
        parser.exit(2, f"Stage B packet compilation failure: {exc}\n")
    print(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
