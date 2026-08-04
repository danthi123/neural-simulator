from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from sim.snr_executable_packet import (
    ADJUDICATION_SCHEMA_VERSION,
    AuthorityKind,
    AuthorityPolicy,
    ExecutablePacket,
    PARAMETER_SCHEMA,
    PacketError,
    PacketState,
    SCHEMA_VERSION,
    TrustedClaim,
    ValidationReceipt,
    canonical_bytes,
    canonical_decimal,
    claim_document,
    expected_adjudication_document,
    load_packet,
    load_packet_json,
    materialize_packet,
)


_UNCERTAINTY = {
    "kind": "not_reported",
    "lower": None,
    "upper": None,
    "unit": None,
}

_EXPECTED_SCHEMA = {
    "fast_hh": {
        "capacitance_density": "uF/cm^2",
        "sodium_conductance_density": "mS/cm^2",
        "potassium_conductance_density": "mS/cm^2",
        "leak_conductance_density": "mS/cm^2",
        "initial_voltage": "mV",
        "spike_detection_voltage": "mV",
        "initial_m": "dimensionless",
        "initial_h": "dimensionless",
        "initial_n": "dimensionless",
        "sodium_activation_q10": "dimensionless",
        "sodium_inactivation_q10": "dimensionless",
        "potassium_activation_q10": "dimensionless",
    },
    "nalcn": {"conductance_density": "mS/cm^2"},
    "nap": {
        "conductance_density": "nS/pF",
        "activation_half": "mV",
        "activation_slope": "mV",
        "activation_tau_min": "ms",
        "activation_tau_max": "ms",
        "activation_tau_half": "mV",
        "activation_tau_sigma_0": "mV",
        "activation_tau_sigma_1": "mV",
        "inactivation_half": "mV",
        "inactivation_slope": "mV",
        "inactivation_tau_min": "ms",
        "inactivation_tau_max": "ms",
        "inactivation_tau_half": "mV",
        "inactivation_tau_sigma_0": "mV",
        "inactivation_tau_sigma_1": "mV",
        "kinetic_q10": "dimensionless",
        "reference_temperature": "degC",
    },
    "cav22": {
        "conductance_density": "nS/pF",
        "activation_half": "mV",
        "activation_slope": "mV",
        "activation_tau": "ms",
        "inactivation_half": "mV",
        "inactivation_slope": "mV",
        "inactivation_tau": "ms",
        "activation_power": "dimensionless",
        "kinetic_q10": "dimensionless",
        "reference_temperature": "degC",
    },
    "hcn": {
        "conductance_density": "nS/pF",
        "activation_half": "mV",
        "activation_slope": "mV",
        "activation_tau": "ms",
        "kinetic_q10": "dimensionless",
        "reference_temperature": "degC",
    },
    "calcium": {
        "baseline": "uM",
        "decay_tau": "ms",
        "current_fraction": "dimensionless",
        "kinetic_q10": "dimensionless",
        "reference_temperature": "degC",
    },
    "sk": {
        "conductance_density": "nS/pF",
        "half_activation": "uM",
        "hill_coefficient": "dimensionless",
        "activation_tau": "ms",
        "deactivation_tau": "ms",
        "kinetic_q10": "dimensionless",
        "reference_temperature": "degC",
    },
    "geometry": {
        "membrane_area": "um^2",
        "accessible_calcium_volume": "um^3",
    },
    "ionic_env": {
        "sodium_reversal": "mV",
        "potassium_reversal": "mV",
        "leak_reversal": "mV",
        "nalcn_reversal": "mV",
        "calcium_reversal": "mV",
        "hcn_reversal": "mV",
        "extracellular_calcium": "uM",
        "calcium_valence": "dimensionless",
        "simulation_temperature": "degC",
    },
}


def _write_json(path: Path, value: object) -> bytes:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _valid_value(group: str, parameter: str) -> str:
    overrides = {
        ("fast_hh", "initial_voltage"): "-65",
        ("fast_hh", "spike_detection_voltage"): "-20",
        ("fast_hh", "initial_m"): "0.1",
        ("fast_hh", "initial_h"): "0.6",
        ("fast_hh", "initial_n"): "0.3",
        ("calcium", "baseline"): "0.1",
        ("calcium", "current_fraction"): "0.5",
        ("cav22", "activation_power"): "2",
        ("ionic_env", "sodium_reversal"): "50",
        ("ionic_env", "potassium_reversal"): "-90",
        ("ionic_env", "leak_reversal"): "-65",
        ("ionic_env", "nalcn_reversal"): "0",
        ("ionic_env", "calcium_reversal"): "120",
        ("ionic_env", "hcn_reversal"): "-30",
        ("ionic_env", "extracellular_calcium"): "2000",
        ("ionic_env", "calcium_valence"): "2",
        ("ionic_env", "simulation_temperature"): "35",
    }
    if (group, parameter) in overrides:
        return overrides[(group, parameter)]
    if parameter == "reference_temperature":
        return "35"
    if parameter.endswith("q10"):
        return "2"
    return "1"


def _packet(
    tmp_path: Path,
    *,
    state: str = "DRAFT",
    evidence: str = "derived",
    authority: str = "project_decision",
) -> dict[str, object]:
    packet_id = "snr-test-v2"
    groups: dict[str, dict[str, object]] = {}
    claims: dict[str, dict[str, object]] = {}
    for group, schema in PARAMETER_SCHEMA.items():
        group_leaves: dict[str, object] = {}
        group_claims: dict[str, object] = {}
        for parameter, permitted_units in schema.items():
            unit = next(iter(permitted_units))
            value = _valid_value(group, parameter)
            claim = claim_document(
                packet_id,
                group,
                parameter,
                value,
                unit,
                _UNCERTAINTY,
                evidence,
                authority,
            )
            claim_digest = hashlib.sha256(canonical_bytes(claim)).hexdigest()
            locator = f"json-pointer:/claims/{group}/{parameter}"
            common = {
                "artifact_path": "source.json",
                "artifact_sha256": "PENDING",
                "locator": locator,
                "claim_sha256": claim_digest,
            }
            group_leaves[parameter] = {
                "value": value,
                "unit": unit,
                "uncertainty": dict(_UNCERTAINTY),
                "evidence": {"kind": evidence, **common},
                "authority": {"kind": authority, **common},
            }
            group_claims[parameter] = claim
        groups[group] = group_leaves
        claims[group] = group_claims
    source_raw = _write_json(tmp_path / "source.json", {"claims": claims})
    source_sha = hashlib.sha256(source_raw).hexdigest()
    for leaves in groups.values():
        for leaf in leaves.values():
            leaf["evidence"]["artifact_sha256"] = source_sha
            leaf["authority"]["artifact_sha256"] = source_sha
    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet_id,
        "state": state,
        "groups": groups,
        "adjudication": None,
    }
    if state in {"SCIENTIFICALLY_RESOLVED", "SEALED"}:
        structural_document = copy.deepcopy(document)
        structural_document["state"] = "ARTIFACTS_VERIFIED"
        structural = _load_packet(structural_document, tmp_path)
        adjudication = expected_adjudication_document(structural)
        assert adjudication["schema_version"] == ADJUDICATION_SCHEMA_VERSION
        adjudication_raw = _write_json(
            tmp_path / "adjudication.json", {"receipt": adjudication}
        )
        document["adjudication"] = {
            "artifact_path": "adjudication.json",
            "artifact_sha256": hashlib.sha256(adjudication_raw).hexdigest(),
            "locator": "json-pointer:/receipt",
            "receipt_sha256": hashlib.sha256(canonical_bytes(adjudication)).hexdigest(),
        }
    return document


def _authority_policy(document: dict[str, object]) -> AuthorityPolicy:
    claims = frozenset(
        TrustedClaim(
            AuthorityKind(leaf["authority"]["kind"]),
            leaf["authority"]["artifact_sha256"],
            leaf["authority"]["claim_sha256"],
        )
        for leaves in document["groups"].values()
        for leaf in leaves.values()
        if leaf["authority"]["kind"] != "unresolved"
    )
    if not claims:
        claims = frozenset(
            {TrustedClaim(AuthorityKind.PROJECT_DECISION, "0" * 64, "1" * 64)}
        )
    adjudication = document.get("adjudication")
    receipts = (
        frozenset({adjudication["receipt_sha256"]})
        if isinstance(adjudication, dict)
        else frozenset({"0" * 64})
    )
    return AuthorityPolicy("test-review-policy-v1", claims, receipts)


def _load_packet(document: dict[str, object], tmp_path: Path):
    policy = (
        _authority_policy(document)
        if document["state"] in {"SCIENTIFICALLY_RESOLVED", "SEALED"}
        else None
    )
    return load_packet(
        document,
        artifact_root=tmp_path,
        authority_policy=policy,
    )


def _rewrite_claim_artifact(tmp_path: Path, document: dict[str, object]) -> None:
    claims: dict[str, dict[str, object]] = {}
    for group, leaves in document["groups"].items():
        claims[group] = {}
        for parameter, leaf in leaves.items():
            claim = claim_document(
                document["packet_id"],
                group,
                parameter,
                leaf["value"],
                leaf["unit"],
                leaf["uncertainty"],
                leaf["evidence"]["kind"],
                leaf["authority"]["kind"],
            )
            claims[group][parameter] = claim
            digest = hashlib.sha256(canonical_bytes(claim)).hexdigest()
            leaf["evidence"]["claim_sha256"] = digest
            leaf["authority"]["claim_sha256"] = digest
    raw = _write_json(tmp_path / "source.json", {"claims": claims})
    digest = hashlib.sha256(raw).hexdigest()
    for leaves in document["groups"].values():
        for leaf in leaves.values():
            leaf["evidence"]["artifact_sha256"] = digest
            leaf["authority"]["artifact_sha256"] = digest


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    raw = '{"schema_version":"x","schema_version":"y"}'
    with pytest.raises(PacketError, match="duplicate JSON key"):
        load_packet_json(raw, artifact_root=tmp_path)


def test_declared_schema_matches_the_complete_executable_surface() -> None:
    actual = {
        group: {parameter: next(iter(units)) for parameter, units in parameters.items()}
        for group, parameters in PARAMETER_SCHEMA.items()
    }
    assert actual == _EXPECTED_SCHEMA


@pytest.mark.parametrize(
    "bad",
    [
        "+1",
        ".5",
        "01",
        "1_0",
        "١",
        "NaN",
        "Infinity",
        "1.0",
        "-0",
        "1e0",
        "1e999",
        "1" * 65,
        "0." + "1" * 70,
    ],
)
def test_decimal_parser_rejects_noncanonical_or_unbounded_input(bad: str) -> None:
    with pytest.raises(PacketError, match="canonical|bounds|syntax|long"):
        canonical_decimal(bad)


@pytest.mark.parametrize("good", ["0", "-1", "12", "1.25", "0.000001", "1e-7", "1e21"])
def test_decimal_parser_accepts_bounded_canonical_forms(good: str) -> None:
    assert canonical_decimal(good) == good


def test_direct_packet_and_receipt_constructor_misuse_is_rejected(tmp_path: Path) -> None:
    document = _packet(tmp_path, state="SEALED")
    sealed = _load_packet(document, tmp_path)
    with pytest.raises(TypeError, match="loader-created"):
        ExecutablePacket(
            object(),
            packet_id="forged",
            state=PacketState.SEALED,
            groups={},
            artifact_root=tmp_path,
            adjudication=None,
            authority_policy=None,
        )
    with pytest.raises(TypeError):
        replace(sealed, _state=PacketState.DRAFT)
    with pytest.raises(TypeError, match="loader-issued"):
        ValidationReceipt(
            object(), "a" * 64, "b" * 64, "c" * 64, "d" * 64
        )
    with pytest.raises(AttributeError, match="immutable"):
        sealed._state = PacketState.DRAFT


def test_sealed_leaves_require_exact_receipt_and_revalidation(tmp_path: Path) -> None:
    sealed = _load_packet(_packet(tmp_path, state="SEALED"), tmp_path)
    with pytest.raises(PacketError, match="materialize_packet"):
        _ = sealed.groups
    receipt = sealed.validation_receipt
    materialized = materialize_packet(sealed, receipt)
    assert materialized.packet_id == sealed.packet_id
    assert set(materialized.groups) == set(PARAMETER_SCHEMA)
    other = _load_packet(_packet(tmp_path, state="SEALED"), tmp_path)
    with pytest.raises(PacketError, match="this packet's validation receipt"):
        materialize_packet(other, receipt)


def test_resolved_packet_requires_external_claim_and_adjudication_trust(tmp_path: Path) -> None:
    document = _packet(tmp_path, state="SEALED")
    with pytest.raises(PacketError, match="external AuthorityPolicy"):
        load_packet(document, artifact_root=tmp_path)

    policy = _authority_policy(document)
    missing_claim = next(iter(policy.trusted_claims))
    incomplete = AuthorityPolicy(
        "incomplete-test-policy",
        policy.trusted_claims - {missing_claim},
        policy.trusted_adjudication_receipts,
    )
    with pytest.raises(PacketError, match="does not approve"):
        load_packet(
            document,
            artifact_root=tmp_path,
            authority_policy=incomplete,
        )

    wrong_receipt = AuthorityPolicy(
        "wrong-receipt-test-policy",
        policy.trusted_claims,
        frozenset({"f" * 64}),
    )
    with pytest.raises(PacketError, match="does not approve adjudication"):
        load_packet(
            document,
            artifact_root=tmp_path,
            authority_policy=wrong_receipt,
        )


@pytest.mark.parametrize(
    ("group", "parameter", "bad"),
    [
        ("fast_hh", "initial_m", "2"),
        ("nap", "activation_slope", "0"),
        ("cav22", "activation_power", "1.5"),
        ("calcium", "current_fraction", "2"),
        ("geometry", "membrane_area", "-1"),
        ("ionic_env", "calcium_valence", "0"),
    ],
)
def test_nonexecutable_parameter_values_cannot_seal(
    tmp_path: Path, group: str, parameter: str, bad: str
) -> None:
    document = _packet(tmp_path)
    document["groups"][group][parameter]["value"] = bad
    with pytest.raises(PacketError, match="must be"):
        _load_packet(document, tmp_path)


def test_cross_field_invariants_are_enforced(tmp_path: Path) -> None:
    document = _packet(tmp_path)
    document["groups"]["fast_hh"]["spike_detection_voltage"]["value"] = "-80"
    _rewrite_claim_artifact(tmp_path, document)
    with pytest.raises(PacketError, match="must exceed"):
        _load_packet(document, tmp_path)

    document = _packet(tmp_path)
    document["groups"]["nap"]["activation_tau_min"]["value"] = "2"
    document["groups"]["nap"]["activation_tau_max"]["value"] = "1"
    _rewrite_claim_artifact(tmp_path, document)
    with pytest.raises(PacketError, match="must not exceed"):
        _load_packet(document, tmp_path)


def test_complete_parameter_and_unit_schema_is_enforced(tmp_path: Path) -> None:
    missing = _packet(tmp_path)
    del missing["groups"]["fast_hh"]["initial_m"]
    with pytest.raises(PacketError, match="fast_hh parameters mismatch.*initial_m"):
        _load_packet(missing, tmp_path)

    extra = _packet(tmp_path)
    extra["groups"]["nalcn"]["invented"] = copy.deepcopy(
        extra["groups"]["nalcn"]["conductance_density"]
    )
    with pytest.raises(PacketError, match="nalcn parameters mismatch.*invented"):
        _load_packet(extra, tmp_path)

    wrong_unit = _packet(tmp_path)
    wrong_unit["groups"]["geometry"]["membrane_area"]["unit"] = "cm^2"
    with pytest.raises(PacketError, match="not permitted"):
        _load_packet(wrong_unit, tmp_path)


def test_generic_locator_cannot_authorize_unrelated_values(tmp_path: Path) -> None:
    document = _packet(tmp_path, state="ARTIFACTS_VERIFIED")
    leaf = document["groups"]["nap"]["activation_half"]
    leaf["evidence"]["locator"] = (
        document["groups"]["nalcn"]["conductance_density"]["evidence"]["locator"]
    )
    with pytest.raises(PacketError, match="does not bind executable claim"):
        _load_packet(document, tmp_path)


def test_locator_content_must_bind_uncertainty_and_both_classes(tmp_path: Path) -> None:
    document = _packet(tmp_path, state="ARTIFACTS_VERIFIED")
    source = json.loads((tmp_path / "source.json").read_text(encoding="utf-8"))
    claim = source["claims"]["sk"]["half_activation"]
    claim["uncertainty"]["kind"] = "interval"
    raw = _write_json(tmp_path / "source.json", source)
    digest = hashlib.sha256(raw).hexdigest()
    for leaves in document["groups"].values():
        for leaf in leaves.values():
            leaf["evidence"]["artifact_sha256"] = digest
            leaf["authority"]["artifact_sha256"] = digest
    with pytest.raises(PacketError, match="does not bind executable claim"):
        _load_packet(document, tmp_path)


def test_claim_digest_covers_extracted_canonical_content(tmp_path: Path) -> None:
    document = _packet(tmp_path)
    document["groups"]["hcn"]["activation_tau"]["evidence"]["claim_sha256"] = "0" * 64
    with pytest.raises(PacketError, match="claim_sha256"):
        _load_packet(document, tmp_path)


def test_artifact_mutation_after_sealing_fails_materialization(tmp_path: Path) -> None:
    sealed = _load_packet(_packet(tmp_path, state="SEALED"), tmp_path)
    receipt = sealed.validation_receipt
    (tmp_path / "source.json").write_text('{"claims":{}}', encoding="utf-8")
    with pytest.raises(PacketError, match="artifact digest mismatch"):
        materialize_packet(sealed, receipt)


def test_adjudication_mutation_after_sealing_fails_materialization(tmp_path: Path) -> None:
    sealed = _load_packet(_packet(tmp_path, state="SEALED"), tmp_path)
    receipt = sealed.validation_receipt
    (tmp_path / "adjudication.json").write_text('{"receipt":{}}', encoding="utf-8")
    with pytest.raises(PacketError, match="artifact digest mismatch"):
        materialize_packet(sealed, receipt)


def test_scientific_resolution_requires_independent_complete_adjudication(tmp_path: Path) -> None:
    document = _packet(tmp_path, state="SEALED")
    adjudication = json.loads((tmp_path / "adjudication.json").read_text(encoding="utf-8"))
    adjudication["receipt"]["compatibility_decisions"].pop()
    raw = _write_json(tmp_path / "adjudication.json", adjudication)
    document["adjudication"]["artifact_sha256"] = hashlib.sha256(raw).hexdigest()
    document["adjudication"]["receipt_sha256"] = hashlib.sha256(
        canonical_bytes(adjudication["receipt"])
    ).hexdigest()
    with pytest.raises(PacketError, match="complete packet"):
        _load_packet(document, tmp_path)


def test_adjudication_cannot_reuse_a_leaf_source(tmp_path: Path) -> None:
    document = _packet(tmp_path, state="SEALED")
    document["adjudication"]["artifact_path"] = "source.json"
    with pytest.raises(PacketError, match="independent artifact"):
        _load_packet(document, tmp_path)


def test_evidence_authority_compatibility_is_enforced(tmp_path: Path) -> None:
    document = _packet(
        tmp_path,
        state="ARTIFACTS_VERIFIED",
        evidence="measured",
        authority="project_decision",
    )
    structural = _load_packet(document, tmp_path)
    adjudication = expected_adjudication_document(structural)
    raw = _write_json(tmp_path / "adjudication.json", {"receipt": adjudication})
    document["state"] = "SCIENTIFICALLY_RESOLVED"
    document["adjudication"] = {
        "artifact_path": "adjudication.json",
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "locator": "json-pointer:/receipt",
        "receipt_sha256": hashlib.sha256(canonical_bytes(adjudication)).hexdigest(),
    }
    with pytest.raises(PacketError, match="incompatible evidence/authority"):
        _load_packet(document, tmp_path)


def test_unresolved_pair_cannot_be_scientifically_resolved(tmp_path: Path) -> None:
    document = _packet(
        tmp_path,
        state="ARTIFACTS_VERIFIED",
        evidence="unresolved",
        authority="unresolved",
    )
    structural = _load_packet(document, tmp_path)
    adjudication = expected_adjudication_document(structural)
    raw = _write_json(tmp_path / "adjudication.json", {"receipt": adjudication})
    document["state"] = "SCIENTIFICALLY_RESOLVED"
    document["adjudication"] = {
        "artifact_path": "adjudication.json",
        "artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "locator": "json-pointer:/receipt",
        "receipt_sha256": hashlib.sha256(canonical_bytes(adjudication)).hexdigest(),
    }
    with pytest.raises(PacketError, match="unresolved evidence"):
        _load_packet(document, tmp_path)


def test_claimed_resolved_state_without_adjudication_fails(tmp_path: Path) -> None:
    document = _packet(tmp_path, state="ARTIFACTS_VERIFIED")
    document["state"] = "SEALED"
    with pytest.raises(PacketError, match="requires adjudication"):
        _load_packet(document, tmp_path)


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="Linux no-follow test")
def test_final_and_component_symlinks_are_rejected(tmp_path: Path) -> None:
    document = _packet(tmp_path, state="ARTIFACTS_VERIFIED")
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    source.rename(target)
    source.symlink_to(target.name)
    with pytest.raises(PacketError, match="cannot safely open artifact"):
        _load_packet(document, tmp_path)

    source.unlink()
    source.write_bytes(target.read_bytes())
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    nested = real_dir / "source.json"
    nested.write_bytes(source.read_bytes())
    alias = tmp_path / "alias"
    alias.symlink_to(real_dir.name, target_is_directory=True)
    for leaves in document["groups"].values():
        for leaf in leaves.values():
            leaf["evidence"]["artifact_path"] = "alias/source.json"
            leaf["authority"]["artifact_path"] = "alias/source.json"
    with pytest.raises(PacketError, match="cannot safely open artifact"):
        _load_packet(document, tmp_path)


@pytest.mark.parametrize("path", ["../source.json", "/tmp/source.json", "a/../source.json", "./source.json"])
def test_path_traversal_and_noncanonical_paths_are_rejected(tmp_path: Path, path: str) -> None:
    document = _packet(tmp_path)
    document["groups"]["nalcn"]["conductance_density"]["evidence"]["artifact_path"] = path
    with pytest.raises(PacketError, match="artifact_path"):
        _load_packet(document, tmp_path)


def test_nonregular_artifact_is_rejected(tmp_path: Path) -> None:
    document = _packet(tmp_path, state="ARTIFACTS_VERIFIED")
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    for leaves in document["groups"].values():
        for leaf in leaves.values():
            leaf["evidence"]["artifact_path"] = "not-a-file"
            leaf["authority"]["artifact_path"] = "not-a-file"
    with pytest.raises(PacketError, match="not a regular file"):
        _load_packet(document, tmp_path)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO unavailable")
def test_fifo_artifact_is_rejected_without_blocking(tmp_path: Path) -> None:
    document = _packet(tmp_path, state="ARTIFACTS_VERIFIED")
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    for leaves in document["groups"].values():
        for leaf in leaves.values():
            leaf["evidence"]["artifact_path"] = "fifo"
            leaf["authority"]["artifact_path"] = "fifo"
    with pytest.raises(PacketError, match="not a regular file"):
        _load_packet(document, tmp_path)


def test_nested_records_and_materialized_values_are_immutable(tmp_path: Path) -> None:
    sealed = _load_packet(_packet(tmp_path, state="SEALED"), tmp_path)
    materialized = materialize_packet(sealed, sealed.validation_receipt)
    leaf = materialized.groups["sk"]["half_activation"]
    with pytest.raises(TypeError):
        materialized.groups["sk"] = {}
    with pytest.raises(TypeError):
        materialized.groups["sk"]["half_activation"] = leaf
    with pytest.raises(AttributeError):
        leaf.value = "2"
    with pytest.raises(AttributeError):
        object.__setattr__(leaf, "value", "-999")


def test_digest_is_deterministic_across_mapping_order(tmp_path: Path) -> None:
    document = _packet(tmp_path, state="SEALED")
    first = _load_packet(document, tmp_path)
    reordered = json.loads(json.dumps(document, sort_keys=True))
    second = _load_packet(reordered, tmp_path)
    assert first.canonical_bytes == second.canonical_bytes
    assert first.sha256 == second.sha256
    assert first.structural_sha256 == second.structural_sha256


def test_interval_uncertainty_is_bound_and_immutable(tmp_path: Path) -> None:
    document = _packet(tmp_path, state="DRAFT")
    leaf = document["groups"]["hcn"]["activation_half"]
    leaf["uncertainty"] = {
        "kind": "interval",
        "lower": "-80",
        "upper": "-70",
        "unit": "mV",
    }
    _rewrite_claim_artifact(tmp_path, document)
    packet = _load_packet(document, tmp_path)
    uncertainty = packet.groups["hcn"]["activation_half"].uncertainty
    assert uncertainty.lower == "-80"
    with pytest.raises(FrozenInstanceError):
        uncertainty.lower = "-90"


def test_packet_and_locator_resource_inputs_fail_as_packet_errors(tmp_path: Path) -> None:
    with pytest.raises(PacketError, match="size limit"):
        load_packet_json("x" * (4 * 1024 * 1024 + 1), artifact_root=tmp_path)

    nested: object = "leaf"
    for _ in range(40):
        nested = {"next": nested}
    with pytest.raises(PacketError, match="nesting limit"):
        load_packet(nested, artifact_root=tmp_path)

    document = _packet(tmp_path, state="ARTIFACTS_VERIFIED")
    document["groups"]["nalcn"]["conductance_density"]["evidence"][
        "locator"
    ] = "json-pointer:/" + "9" * 5000
    with pytest.raises(PacketError):
        _load_packet(document, tmp_path)
