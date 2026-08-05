"""Independent, fail-closed verifier and policy issuer for one Stage B packet.

This module deliberately reimplements the compiler's input interpretation.  It
does not import the compiler or use its validation helpers: agreement between
the two implementations is the control that permits the verifier to issue a
candidate-specific authority policy.
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
    AUTHORITY_POLICY_SCHEMA_VERSION,
    PARAMETER_SCHEMA,
    SCHEMA_VERSION,
    PacketError,
    canonical_bytes,
    canonical_decimal,
    claim_document,
    expected_adjudication_document,
    load_authority_policy_file,
    load_packet,
    load_packet_file,
    materialize_packet,
)
from sim.snr_packet_runtime import materialized_packet_sha256


TEMPLATE_SCHEMA = "v14-snr-stageB-packet-template-v1"
CANDIDATE_SCHEMA = "sim-adaptive-candidate-v1"
REQUEST_SCHEMA = "v14-snr-stageB-compilation-request-v1"
EVIDENCE_SCHEMA = "v14-snr-stageB-evidence-claims-v1"
AUTHORITY_SCHEMA = "v14-snr-stageB-authority-claims-v1"
RELEASE_SCHEMA = "v14-snr-stageB-candidate-release-v1"
INPUT_NAMES = (
    "candidate.json", "compilation-request.json", "evidence-claims.json",
    "packet.structural.json",
)
OUTPUT_NAMES = (
    "authority-claims.json", "packet.artifacts-verified.json", "adjudication.json",
    "authority-policy.json", "packet.sealed.json", "candidate-release.json",
)


class StageBPacketVerifierError(ValueError):
    """The compiler output cannot be independently authorized."""


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _digest_bytes(canonical_bytes(value))


def _sha256(value: Any, context: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise StageBPacketVerifierError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _text(value: Any, context: str) -> str:
    if (not isinstance(value, str) or not value or value != value.strip()
            or any(ord(character) > 127 for character in value)):
        raise StageBPacketVerifierError(f"{context} must be nonempty trimmed ASCII text")
    return value


def _decimal_string(value: Any, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise StageBPacketVerifierError(f"{context} must be a finite JSON number or decimal string")
    if isinstance(value, float) and not math.isfinite(value):
        raise StageBPacketVerifierError(f"{context} must be finite")
    if isinstance(value, str):
        try:
            return canonical_decimal(value, context)
        except PacketError as exc:
            raise StageBPacketVerifierError(str(exc)) from exc
    try:
        decimal = Decimal(str(value))
    except InvalidOperation as exc:
        raise StageBPacketVerifierError(f"{context} is not a decimal number") from exc
    if not decimal.is_finite():
        raise StageBPacketVerifierError(f"{context} must be finite")
    if decimal == 0:
        normalized = "0"
    else:
        normalized_decimal = decimal.normalize()
        adjusted = normalized_decimal.adjusted()
        normalized = (
            format(normalized_decimal, "f")
            if -6 <= adjusted <= 20
            else format(normalized_decimal, "e").replace("e+", "e")
        )
    try:
        return canonical_decimal(normalized, context)
    except PacketError as exc:
        raise StageBPacketVerifierError(str(exc)) from exc


def _canonical_object(path: Path, context: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBPacketVerifierError(f"cannot load {context}: {exc}") from exc
    if not isinstance(document, dict):
        raise StageBPacketVerifierError(f"{context} must contain an object")
    try:
        canonical = canonical_bytes(document)
    except PacketError as exc:
        raise StageBPacketVerifierError(f"{context} is not canonical JSON: {exc}") from exc
    if raw != canonical:
        raise StageBPacketVerifierError(f"{context} bytes are not canonical JSON")
    return document


def _pinned_template(path: str | Path, expected_sha256: str, root: Path) -> tuple[Path, dict[str, Any]]:
    _sha256(expected_sha256, "template expected digest")
    source = Path(path).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise StageBPacketVerifierError("packet template must be inside repository_root") from exc
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise StageBPacketVerifierError(f"cannot load packet template: {exc}") from exc
    if _digest_bytes(raw) != expected_sha256:
        raise StageBPacketVerifierError("packet template digest does not match")
    document = _canonical_object(source, "packet template")
    return source, document


def _template(document: Any) -> dict[str, Any]:
    if (not isinstance(document, Mapping) or set(document) != {"schema", "template_id", "parameter_leaves"}
            or document.get("schema") != TEMPLATE_SCHEMA):
        raise StageBPacketVerifierError("packet template has an invalid shape or schema")
    _text(document.get("template_id"), "template_id")
    groups = document.get("parameter_leaves")
    if not isinstance(groups, Mapping) or set(groups) != set(PARAMETER_SCHEMA):
        raise StageBPacketVerifierError("packet template does not contain the exact parameter groups")
    for group, schema in PARAMETER_SCHEMA.items():
        leaves = groups[group]
        if not isinstance(leaves, Mapping) or set(leaves) != set(schema):
            raise StageBPacketVerifierError(f"packet template group {group!r} is incomplete or widened")
    return dict(document)


def _candidate(document: Any) -> dict[str, Any]:
    if not isinstance(document, Mapping) or set(document) != {"schema", "candidate_id", "parameters"}:
        raise StageBPacketVerifierError("candidate document has an invalid shape")
    if document.get("schema") != CANDIDATE_SCHEMA:
        raise StageBPacketVerifierError("candidate document has the wrong schema")
    identifier = _text(document.get("candidate_id"), "candidate_id")
    parameters = document.get("parameters")
    if not isinstance(parameters, Mapping):
        raise StageBPacketVerifierError("candidate parameters must be an object")
    return {"schema": CANDIDATE_SCHEMA, "candidate_id": identifier, "parameters": dict(parameters)}


def _leaf(
    group: str, parameter: str, specification: Mapping[str, Any],
    candidate_parameters: Mapping[str, Any], used_keys: set[str],
) -> tuple[str, str, dict[str, Any], str, str]:
    common = {"mode", "unit", "uncertainty", "evidence", "authority"}
    mode = specification.get("mode")
    expected = common | ({"value"} if mode == "fixed" else {"candidate_key", "bounds", "transform"})
    if mode not in {"fixed", "searched"} or set(specification) != expected:
        raise StageBPacketVerifierError(f"template leaf {group}.{parameter} has an invalid shape")
    unit = specification.get("unit")
    if unit not in PARAMETER_SCHEMA[group][parameter]:
        raise StageBPacketVerifierError(f"template leaf {group}.{parameter} has the wrong unit")
    uncertainty = specification.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise StageBPacketVerifierError(f"template leaf {group}.{parameter} has no uncertainty contract")
    evidence, authority = specification.get("evidence"), specification.get("authority")
    if mode == "searched":
        if evidence != "derived" or authority != "project_decision":
            raise StageBPacketVerifierError(f"searched leaf {group}.{parameter} must remain derived/project_decision")
        key = _text(specification.get("candidate_key"), f"{group}.{parameter}.candidate_key")
        if key in used_keys or key not in candidate_parameters:
            raise StageBPacketVerifierError(f"searched candidate key {key!r} is missing or duplicated")
        used_keys.add(key)
        value = _decimal_string(candidate_parameters[key], f"candidate.parameters.{key}")
        bounds, transform = specification.get("bounds"), specification.get("transform")
        if not isinstance(bounds, Mapping) or set(bounds) != {"low", "high"}:
            raise StageBPacketVerifierError(f"searched leaf {group}.{parameter} has invalid bounds")
        low = Decimal(_decimal_string(bounds["low"], f"{group}.{parameter}.bounds.low"))
        high = Decimal(_decimal_string(bounds["high"], f"{group}.{parameter}.bounds.high"))
        if low > high or not low <= Decimal(value) <= high:
            raise StageBPacketVerifierError(f"candidate value for {key!r} is outside sealed bounds")
        if transform not in {"linear", "log"} or (transform == "log" and low <= 0):
            raise StageBPacketVerifierError(f"searched leaf {group}.{parameter} has invalid transform")
    else:
        value = _decimal_string(specification.get("value"), f"{group}.{parameter}.value")
        compatible = {"measured": "primary_source", "model_prior": "model_source", "derived": "project_decision"}
        if evidence not in compatible or authority != compatible[evidence]:
            raise StageBPacketVerifierError(f"fixed leaf {group}.{parameter} has incompatible evidence/authority")
    return value, str(unit), dict(uncertainty), str(evidence), str(authority)


def _expected_documents(template: Mapping[str, Any], candidate: Mapping[str, Any], template_sha256: str) -> dict[str, dict[str, Any]]:
    checked_template, checked_candidate = _template(template), _candidate(candidate)
    candidate_sha = _digest(checked_candidate)
    packet_id = f"{checked_template['template_id']}--{checked_candidate['candidate_id']}--{candidate_sha[:16]}"
    claims: dict[str, dict[str, Any]] = {}
    leaves: dict[str, dict[str, tuple[str, str, dict[str, Any], str, str]]] = {}
    used_keys: set[str] = set()
    for group in PARAMETER_SCHEMA:
        claims[group], leaves[group] = {}, {}
        for parameter in PARAMETER_SCHEMA[group]:
            leaf = _leaf(group, parameter, checked_template["parameter_leaves"][group][parameter], checked_candidate["parameters"], used_keys)
            value, unit, uncertainty, evidence, authority = leaf
            claims[group][parameter] = claim_document(packet_id, group, parameter, value, unit, uncertainty, evidence, authority)
            leaves[group][parameter] = leaf
    if used_keys != set(checked_candidate["parameters"]):
        raise StageBPacketVerifierError(f"candidate has unfiled parameters: {sorted(set(checked_candidate['parameters']) - used_keys)}")
    evidence = {"schema": EVIDENCE_SCHEMA, "template": {"template_id": checked_template["template_id"], "sha256": template_sha256}, "candidate_sha256": candidate_sha, "claims": claims}
    authority = {"schema": AUTHORITY_SCHEMA, "template": {"template_id": checked_template["template_id"], "sha256": template_sha256}, "candidate_sha256": candidate_sha, "authorized_claims": claims}
    evidence_sha, authority_sha = _digest(evidence), _digest(authority)
    groups: dict[str, dict[str, Any]] = {}
    for group in PARAMETER_SCHEMA:
        groups[group] = {}
        for parameter in PARAMETER_SCHEMA[group]:
            value, unit, uncertainty, evidence_kind, authority_kind = leaves[group][parameter]
            claim_sha = _digest(claims[group][parameter])
            groups[group][parameter] = {
                "value": value, "unit": unit, "uncertainty": uncertainty,
                "evidence": {"kind": evidence_kind, "artifact_path": "evidence-claims.json", "artifact_sha256": evidence_sha, "locator": f"json-pointer:/claims/{group}/{parameter}", "claim_sha256": claim_sha},
                "authority": {"kind": authority_kind, "artifact_path": "authority-claims.json", "artifact_sha256": authority_sha, "locator": f"json-pointer:/authorized_claims/{group}/{parameter}", "claim_sha256": claim_sha},
            }
    structural = {"schema_version": SCHEMA_VERSION, "packet_id": packet_id, "state": "STRUCTURAL", "groups": groups, "adjudication": None}
    try:
        load_packet(structural, artifact_root=Path.cwd())
    except PacketError as exc:
        raise StageBPacketVerifierError(f"recomputed packet is structurally invalid: {exc}") from exc
    request = {
        "schema": REQUEST_SCHEMA, "template": {"template_id": checked_template["template_id"], "sha256": template_sha256},
        "candidate_sha256": candidate_sha, "evidence_claims_sha256": evidence_sha,
        "expected_authority_claims_sha256": authority_sha, "structural_packet_sha256": _digest(structural),
        "compiler_authority": "none", "forbidden_outputs": ["authority-claims.json", "adjudication.json", "authority-policy.json", "packet.sealed.json", "candidate-release.json"],
    }
    return {"candidate.json": checked_candidate, "compilation-request.json": request, "evidence-claims.json": evidence, "packet.structural.json": structural, "authority-claims.json": authority}


def _write_once(path: Path, document: Mapping[str, Any]) -> str:
    raw = canonical_bytes(document)
    with path.open("xb") as handle:
        handle.write(raw)
    return _digest_bytes(raw)


class _OutputTransaction:
    """Remove only outputs created by this verification if issuance fails."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._created: list[Path] = []

    def __enter__(self) -> "_OutputTransaction":
        return self

    def write(self, name: str, document: Mapping[str, Any]) -> str:
        path = self._directory / name
        digest = _write_once(path, document)
        self._created.append(path)
        return digest

    def rollback(self) -> None:
        for path in reversed(self._created):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        self._created.clear()

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> bool:
        if exception_type is not None:
            self.rollback()
        return False


def _require_compiler_directory(path: str | Path, root: Path) -> Path:
    directory = Path(path).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise StageBPacketVerifierError("compiler output directory must be inside repository_root") from exc
    if not directory.is_dir():
        raise StageBPacketVerifierError("compiler output directory must be a directory")
    names = {item.name for item in directory.iterdir()}
    if names != set(INPUT_NAMES):
        raise StageBPacketVerifierError("compiler output directory must contain exactly the four compiler artifacts")
    if any((directory / name).is_symlink() or not (directory / name).is_file() for name in INPUT_NAMES):
        raise StageBPacketVerifierError("compiler artifacts must be regular, non-symlink files")
    return directory


def verify_candidate(
    template_path: str | Path, template_sha256: str, compilation_dir: str | Path, *, repository_root: str | Path,
) -> dict[str, Any]:
    """Verify one compiler directory and issue its six write-once outputs.

    The function is intentionally single-candidate: no policy or release can
    span a batch, and any existing output is a hard failure.
    """
    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise StageBPacketVerifierError("repository_root must be a directory")
    _, template = _pinned_template(template_path, template_sha256, root)
    directory = _require_compiler_directory(compilation_dir, root)
    inputs = {name: _canonical_object(directory / name, name) for name in INPUT_NAMES}
    expected = _expected_documents(template, inputs["candidate.json"], template_sha256)
    for name in INPUT_NAMES:
        if inputs[name] != expected[name]:
            raise StageBPacketVerifierError(f"{name} does not equal the independently recomputed document")

    outputs = _OutputTransaction(directory)
    # The first two writes enable a production ARTIFACTS_VERIFIED load.
    authority_sha = outputs.write("authority-claims.json", expected["authority-claims.json"])
    verified = dict(expected["packet.structural.json"])
    verified["state"] = "ARTIFACTS_VERIFIED"
    verified_sha = outputs.write("packet.artifacts-verified.json", verified)
    try:
        verified_packet = load_packet_file(
            "packet.artifacts-verified.json", artifact_root=directory,
            expected_sha256=verified_sha, authority_policy=None,
        )
    except PacketError as exc:
        outputs.rollback()
        raise StageBPacketVerifierError(f"ARTIFACTS_VERIFIED packet did not load: {exc}") from exc

    receipt = expected_adjudication_document(verified_packet)
    adjudication = {"receipt": receipt}
    adjudication_sha = outputs.write("adjudication.json", adjudication)
    receipt_sha = _digest(receipt)
    trusted_claims = []
    for group in PARAMETER_SCHEMA:
        for parameter in PARAMETER_SCHEMA[group]:
            claim = expected["authority-claims.json"]["authorized_claims"][group][parameter]
            trusted_claims.append({"authority": claim["authority"], "artifact_sha256": authority_sha, "claim_sha256": _digest(claim)})
    if len(trusted_claims) != 69:
        outputs.rollback()
        raise StageBPacketVerifierError("authority policy must contain exactly 69 leaf claims")
    policy = {
        "schema_version": AUTHORITY_POLICY_SCHEMA_VERSION,
        "policy_id": f"{verified_packet.packet_id}--authority-policy",
        "trusted_claims": trusted_claims,
        "trusted_adjudication_receipts": [receipt_sha],
    }
    policy_sha = outputs.write("authority-policy.json", policy)
    sealed = dict(verified)
    sealed["state"] = "SEALED"
    sealed["adjudication"] = {"artifact_path": "adjudication.json", "artifact_sha256": adjudication_sha, "locator": "json-pointer:/receipt", "receipt_sha256": receipt_sha}
    sealed_sha = outputs.write("packet.sealed.json", sealed)
    try:
        loaded_policy = load_authority_policy_file("authority-policy.json", artifact_root=directory, expected_sha256=policy_sha)
        sealed_packet = load_packet_file("packet.sealed.json", artifact_root=directory, expected_sha256=sealed_sha, authority_policy=loaded_policy)
        materialized = materialize_packet(sealed_packet, sealed_packet.validation_receipt)
    except PacketError as exc:
        outputs.rollback()
        raise StageBPacketVerifierError(f"SEALED packet did not load and materialize: {exc}") from exc
    release = {
        "schema": RELEASE_SCHEMA,
        "template": {"template_id": template["template_id"], "sha256": template_sha256},
        "candidate": {"candidate_id": expected["candidate.json"]["candidate_id"], "sha256": _digest(expected["candidate.json"])},
        "artifacts": {
            "compilation_request_sha256": _digest(inputs["compilation-request.json"]),
            "evidence_claims_sha256": _digest(inputs["evidence-claims.json"]),
            "authority_claims_sha256": authority_sha,
            "structural_packet_sha256": _digest(inputs["packet.structural.json"]),
            "artifacts_verified_packet_sha256": verified_sha,
            "adjudication_sha256": adjudication_sha,
            "authority_policy_sha256": policy_sha,
            "sealed_packet_sha256": sealed_sha,
            "materialized_sha256": materialized_packet_sha256(materialized),
        },
        "fitted_value_status": "Fitted values remain derived/model priors, never measurements.",
    }
    release_sha = outputs.write("candidate-release.json", release)
    return {"candidate_release_sha256": release_sha, "packet_sha256": sealed_sha, "policy_sha256": policy_sha, "materialized_sha256": materialized_packet_sha256(materialized)}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True)
    parser.add_argument("--template-sha256", required=True)
    parser.add_argument("--compilation-dir", required=True)
    parser.add_argument("--repository-root", required=True)
    args = parser.parse_args(argv)
    try:
        result = verify_candidate(args.template, args.template_sha256, args.compilation_dir, repository_root=args.repository_root)
    except (OSError, PacketError, StageBPacketVerifierError, ValueError, TypeError) as exc:
        parser.exit(2, f"Stage B packet verification failure: {exc}\n")
    print(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
