"""Fail-closed authority boundary for executable SNr parameter packets.

Packets and validation receipts can only be created by the loaders in this
module.  A sealed packet exposes executable leaves only through
``materialize_packet``; that boundary reopens every source and adjudication
artifact before returning an immutable snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from enum import Enum
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple


SCHEMA_VERSION = "snr-executable-packet-v2"
ADJUDICATION_SCHEMA_VERSION = "snr-adjudication-receipt-v1"
_HEX_LENGTH = 64
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_PACKET_BYTES = 4 * 1024 * 1024
_MAX_DOCUMENT_NODES = 10_000
_MAX_DOCUMENT_DEPTH = 32
_MAX_TEXT_CHARS = 4096
_MAX_PATH_CHARS = 4096
_MAX_PATH_COMPONENT_CHARS = 255
_MAX_POINTER_CHARS = 4096
_MAX_POINTER_TOKEN_CHARS = 1024
_MAX_ARRAY_INDEX_DIGITS = 12
_MAX_DECIMAL_CHARS = 64
_MAX_DECIMAL_DIGITS = 48
_MAX_DECIMAL_EXPONENT = 64
_MAX_EXPANDED_DECIMAL_CHARS = 96
_DECIMAL_RE = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:e-?(?:[1-9][0-9]*))?\Z",
    re.ASCII,
)
_CONSTRUCTION_KEY = object()


def _units(**values: str) -> Mapping[str, frozenset[str]]:
    return MappingProxyType({key: frozenset({unit}) for key, unit in values.items()})


# This is the complete scalar surface needed to replace the current hardcoded
# HH/SNr fused path.  Names map onto snr_channel_parameters records,
# CoreSimConfig/BrainRegion fields, and the physical calcium conversion inputs.
PARAMETER_SCHEMA: Mapping[str, Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "fast_hh": _units(
            capacitance_density="uF/cm^2",
            sodium_conductance_density="mS/cm^2",
            potassium_conductance_density="mS/cm^2",
            leak_conductance_density="mS/cm^2",
            initial_voltage="mV",
            spike_detection_voltage="mV",
            initial_m="dimensionless",
            initial_h="dimensionless",
            initial_n="dimensionless",
            sodium_activation_q10="dimensionless",
            sodium_inactivation_q10="dimensionless",
            potassium_activation_q10="dimensionless",
        ),
        "nalcn": _units(conductance_density="mS/cm^2"),
        "nap": _units(
            conductance_density="nS/pF",
            activation_half="mV",
            activation_slope="mV",
            activation_tau_min="ms",
            activation_tau_max="ms",
            activation_tau_half="mV",
            activation_tau_sigma_0="mV",
            activation_tau_sigma_1="mV",
            inactivation_half="mV",
            inactivation_slope="mV",
            inactivation_tau_min="ms",
            inactivation_tau_max="ms",
            inactivation_tau_half="mV",
            inactivation_tau_sigma_0="mV",
            inactivation_tau_sigma_1="mV",
            kinetic_q10="dimensionless",
            reference_temperature="degC",
        ),
        "cav22": _units(
            conductance_density="nS/pF",
            activation_half="mV",
            activation_slope="mV",
            activation_tau="ms",
            inactivation_half="mV",
            inactivation_slope="mV",
            inactivation_tau="ms",
            activation_power="dimensionless",
            kinetic_q10="dimensionless",
            reference_temperature="degC",
        ),
        "hcn": _units(
            conductance_density="nS/pF",
            activation_half="mV",
            activation_slope="mV",
            activation_tau="ms",
            kinetic_q10="dimensionless",
            reference_temperature="degC",
        ),
        "calcium": _units(
            baseline="uM",
            decay_tau="ms",
            current_fraction="dimensionless",
            kinetic_q10="dimensionless",
            reference_temperature="degC",
        ),
        "sk": _units(
            conductance_density="nS/pF",
            half_activation="uM",
            hill_coefficient="dimensionless",
            activation_tau="ms",
            deactivation_tau="ms",
            kinetic_q10="dimensionless",
            reference_temperature="degC",
        ),
        "geometry": _units(
            membrane_area="um^2",
            accessible_calcium_volume="um^3",
        ),
        "ionic_env": _units(
            sodium_reversal="mV",
            potassium_reversal="mV",
            leak_reversal="mV",
            nalcn_reversal="mV",
            calcium_reversal="mV",
            hcn_reversal="mV",
            extracellular_calcium="uM",
            calcium_valence="dimensionless",
            simulation_temperature="degC",
        ),
    }
)
REQUIRED_GROUPS = frozenset(PARAMETER_SCHEMA)


@dataclass(frozen=True, slots=True)
class NumericDomain:
    """Executable numeric domain, separate from scientific uncertainty."""

    lower: str | None = None
    upper: str | None = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True
    integer: bool = False
    nonzero: bool = False


_NONNEGATIVE = NumericDomain(lower="0")
_POSITIVE = NumericDomain(lower="0", lower_inclusive=False)
_GATE = NumericDomain(lower="0", upper="1")
_FRACTION = NumericDomain(lower="0", upper="1", lower_inclusive=False)
_Q10 = NumericDomain(lower="0.01", upper="100")
_VOLTAGE = NumericDomain(lower="-500", upper="500")
_SLOPE = NumericDomain(lower="-500", upper="500", nonzero=True)
_TEMPERATURE = NumericDomain(lower="-50", upper="100")
_POSITIVE_INTEGER = NumericDomain(
    lower="1", upper="16", integer=True
)


def _domains(**values: NumericDomain) -> Mapping[str, NumericDomain]:
    return MappingProxyType(values)


PARAMETER_DOMAINS: Mapping[str, Mapping[str, NumericDomain]] = MappingProxyType(
    {
        "fast_hh": _domains(
            capacitance_density=_POSITIVE,
            sodium_conductance_density=_NONNEGATIVE,
            potassium_conductance_density=_NONNEGATIVE,
            leak_conductance_density=_NONNEGATIVE,
            initial_voltage=_VOLTAGE,
            spike_detection_voltage=_VOLTAGE,
            initial_m=_GATE,
            initial_h=_GATE,
            initial_n=_GATE,
            sodium_activation_q10=_Q10,
            sodium_inactivation_q10=_Q10,
            potassium_activation_q10=_Q10,
        ),
        "nalcn": _domains(conductance_density=_NONNEGATIVE),
        "nap": _domains(
            conductance_density=_NONNEGATIVE,
            activation_half=_VOLTAGE,
            activation_slope=_SLOPE,
            activation_tau_min=_POSITIVE,
            activation_tau_max=_POSITIVE,
            activation_tau_half=_VOLTAGE,
            activation_tau_sigma_0=_SLOPE,
            activation_tau_sigma_1=_SLOPE,
            inactivation_half=_VOLTAGE,
            inactivation_slope=_SLOPE,
            inactivation_tau_min=_POSITIVE,
            inactivation_tau_max=_POSITIVE,
            inactivation_tau_half=_VOLTAGE,
            inactivation_tau_sigma_0=_SLOPE,
            inactivation_tau_sigma_1=_SLOPE,
            kinetic_q10=_Q10,
            reference_temperature=_TEMPERATURE,
        ),
        "cav22": _domains(
            conductance_density=_NONNEGATIVE,
            activation_half=_VOLTAGE,
            activation_slope=_SLOPE,
            activation_tau=_POSITIVE,
            inactivation_half=_VOLTAGE,
            inactivation_slope=_SLOPE,
            inactivation_tau=_POSITIVE,
            activation_power=_POSITIVE_INTEGER,
            kinetic_q10=_Q10,
            reference_temperature=_TEMPERATURE,
        ),
        "hcn": _domains(
            conductance_density=_NONNEGATIVE,
            activation_half=_VOLTAGE,
            activation_slope=_SLOPE,
            activation_tau=_POSITIVE,
            kinetic_q10=_Q10,
            reference_temperature=_TEMPERATURE,
        ),
        "calcium": _domains(
            baseline=_NONNEGATIVE,
            decay_tau=_POSITIVE,
            current_fraction=_FRACTION,
            kinetic_q10=_Q10,
            reference_temperature=_TEMPERATURE,
        ),
        "sk": _domains(
            conductance_density=_NONNEGATIVE,
            half_activation=_POSITIVE,
            hill_coefficient=_POSITIVE,
            activation_tau=_POSITIVE,
            deactivation_tau=_POSITIVE,
            kinetic_q10=_Q10,
            reference_temperature=_TEMPERATURE,
        ),
        "geometry": _domains(
            membrane_area=_POSITIVE,
            accessible_calcium_volume=_POSITIVE,
        ),
        "ionic_env": _domains(
            sodium_reversal=_VOLTAGE,
            potassium_reversal=_VOLTAGE,
            leak_reversal=_VOLTAGE,
            nalcn_reversal=_VOLTAGE,
            calcium_reversal=_VOLTAGE,
            hcn_reversal=_VOLTAGE,
            extracellular_calcium=_POSITIVE,
            calcium_valence=_POSITIVE_INTEGER,
            simulation_temperature=_TEMPERATURE,
        ),
    }
)


class PacketError(ValueError):
    """The packet cannot be trusted, sealed, or materialized."""


class PacketState(str, Enum):
    DRAFT = "DRAFT"
    STRUCTURAL = "STRUCTURAL"
    ARTIFACTS_VERIFIED = "ARTIFACTS_VERIFIED"
    SCIENTIFICALLY_RESOLVED = "SCIENTIFICALLY_RESOLVED"
    SEALED = "SEALED"


class EvidenceKind(str, Enum):
    MEASURED = "measured"
    MODEL_PRIOR = "model_prior"
    DERIVED = "derived"
    UNRESOLVED = "unresolved"


class AuthorityKind(str, Enum):
    PRIMARY_SOURCE = "primary_source"
    MODEL_SOURCE = "model_source"
    PROJECT_DECISION = "project_decision"
    UNRESOLVED = "unresolved"


class UncertaintyKind(str, Enum):
    NOT_REPORTED = "not_reported"
    INTERVAL = "interval"


_COMPATIBLE_BINDINGS = {
    EvidenceKind.MEASURED: AuthorityKind.PRIMARY_SOURCE,
    EvidenceKind.MODEL_PRIOR: AuthorityKind.MODEL_SOURCE,
    EvidenceKind.DERIVED: AuthorityKind.PROJECT_DECISION,
    EvidenceKind.UNRESOLVED: AuthorityKind.UNRESOLVED,
}


class TrustedClaim(NamedTuple):
    """One exact claim authorized by an independently maintained trust root."""

    authority: AuthorityKind
    artifact_sha256: str
    claim_sha256: str


@dataclass(frozen=True, slots=True)
class AuthorityPolicy:
    """External trust root for scientific claims and adjudication receipts.

    Packet files cannot create this authority implicitly. Production callers
    must build it from the reviewed source catalog and adjudication registry;
    choosing another policy intentionally defines a different trust domain.
    """

    policy_id: str
    trusted_claims: frozenset[TrustedClaim]
    trusted_adjudication_receipts: frozenset[str]

    def __post_init__(self) -> None:
        _text(self.policy_id, "authority_policy.policy_id")
        if not self.trusted_claims:
            raise PacketError("authority policy must approve at least one claim")
        if not self.trusted_adjudication_receipts:
            raise PacketError(
                "authority policy must approve at least one adjudication receipt"
            )
        for claim in self.trusted_claims:
            if type(claim) is not TrustedClaim:
                raise PacketError("authority policy claims must be TrustedClaim records")
            if claim.authority is AuthorityKind.UNRESOLVED:
                raise PacketError("authority policy cannot approve unresolved claims")
            _sha256(claim.artifact_sha256, "trusted claim artifact_sha256")
            _sha256(claim.claim_sha256, "trusted claim claim_sha256")
        for digest in self.trusted_adjudication_receipts:
            _sha256(digest, "trusted adjudication receipt")


@dataclass(frozen=True, slots=True)
class Uncertainty:
    kind: UncertaintyKind
    lower: str | None
    upper: str | None
    unit: str | None


@dataclass(frozen=True, slots=True)
class EvidenceBinding:
    kind: EvidenceKind
    artifact_path: str
    artifact_sha256: str
    locator: str
    claim_sha256: str


@dataclass(frozen=True, slots=True)
class AuthorityBinding:
    kind: AuthorityKind
    artifact_path: str
    artifact_sha256: str
    locator: str
    claim_sha256: str


@dataclass(frozen=True, slots=True)
class AdjudicationBinding:
    artifact_path: str
    artifact_sha256: str
    locator: str
    receipt_sha256: str


@dataclass(frozen=True, slots=True)
class ParameterLeaf:
    value: str
    unit: str
    uncertainty: Uncertainty
    evidence: EvidenceBinding
    authority: AuthorityBinding


class MaterializedUncertainty(NamedTuple):
    kind: UncertaintyKind
    lower: str | None
    upper: str | None
    unit: str | None


class MaterializedParameterLeaf(NamedTuple):
    value: str
    unit: str
    uncertainty: MaterializedUncertainty
    evidence_kind: EvidenceKind
    authority_kind: AuthorityKind


class MaterializedPacket(NamedTuple):
    """Deeply immutable value snapshot returned by the materializer."""

    packet_id: str
    packet_sha256: str
    structural_sha256: str
    groups: Mapping[str, Mapping[str, MaterializedParameterLeaf]]


class ValidationReceipt:
    """Loader token binding packet, adjudication, and external trust policy."""

    __slots__ = (
        "_packet_sha256",
        "_structural_sha256",
        "_adjudication_sha256",
        "_authority_policy_sha256",
    )

    def __new__(cls, key: object, *args: object, **kwargs: object) -> "ValidationReceipt":
        if key is not _CONSTRUCTION_KEY:
            raise TypeError("ValidationReceipt instances are loader-issued only")
        return super().__new__(cls)

    def __init__(
        self,
        key: object,
        packet_sha256: str,
        structural_sha256: str,
        adjudication_sha256: str,
        authority_policy_sha256: str,
    ) -> None:
        object.__setattr__(self, "_packet_sha256", packet_sha256)
        object.__setattr__(self, "_structural_sha256", structural_sha256)
        object.__setattr__(self, "_adjudication_sha256", adjudication_sha256)
        object.__setattr__(self, "_authority_policy_sha256", authority_policy_sha256)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ValidationReceipt is immutable")

    def __reduce__(self) -> object:
        raise TypeError("ValidationReceipt cannot be serialized")


class ExecutablePacket:
    """Loader result whose authority is rechecked before materialization.

    Python reflection is not a security boundary. The external AuthorityPolicy,
    current artifact contents, and canonical digests are the authority boundary.
    """

    __slots__ = (
        "_schema_version",
        "_packet_id",
        "_state",
        "_groups",
        "_artifact_root",
        "_adjudication",
        "_authority_policy",
        "_receipt",
        "_locked",
    )

    def __new__(cls, key: object, *args: object, **kwargs: object) -> "ExecutablePacket":
        if key is not _CONSTRUCTION_KEY:
            raise TypeError("ExecutablePacket instances are loader-created only")
        return super().__new__(cls)

    def __init__(
        self,
        key: object,
        *,
        packet_id: str,
        state: PacketState,
        groups: Mapping[str, Mapping[str, ParameterLeaf]],
        artifact_root: Path,
        adjudication: AdjudicationBinding | None,
        authority_policy: AuthorityPolicy | None,
    ) -> None:
        object.__setattr__(self, "_schema_version", SCHEMA_VERSION)
        object.__setattr__(self, "_packet_id", packet_id)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_groups", groups)
        object.__setattr__(self, "_artifact_root", artifact_root)
        object.__setattr__(self, "_adjudication", adjudication)
        object.__setattr__(self, "_authority_policy", authority_policy)
        object.__setattr__(self, "_receipt", None)
        object.__setattr__(self, "_locked", True)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ExecutablePacket is immutable")

    def __reduce__(self) -> object:
        raise TypeError("ExecutablePacket cannot be serialized")

    @property
    def schema_version(self) -> str:
        return self._schema_version

    @property
    def packet_id(self) -> str:
        return self._packet_id

    @property
    def state(self) -> PacketState:
        return self._state

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    @property
    def groups(self) -> Mapping[str, Mapping[str, ParameterLeaf]]:
        if self._state is PacketState.SEALED:
            raise PacketError(
                "sealed executable leaves require materialize_packet(packet, receipt)"
            )
        return self._groups

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_bytes(_packet_document(self))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def structural_sha256(self) -> str:
        return hashlib.sha256(canonical_bytes(_structural_document(self))).hexdigest()

    @property
    def validation_receipt(self) -> ValidationReceipt:
        receipt = self._receipt
        if self._state is not PacketState.SEALED or receipt is None:
            raise PacketError("only a successfully sealed packet has a validation receipt")
        return receipt


def canonical_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible content as deterministic ASCII UTF-8."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PacketError("content is not canonical JSON data") from exc


def claim_document(
    packet_id: str,
    group: str,
    parameter: str,
    value: str,
    unit: str,
    uncertainty: Mapping[str, object],
    evidence: str,
    authority: str,
) -> dict[str, object]:
    """Build the exact canonical content that a source locator must extract."""

    normalized_uncertainty = _load_uncertainty(uncertainty, unit, "claim.uncertainty")
    try:
        evidence_kind = EvidenceKind(evidence)
        authority_kind = AuthorityKind(authority)
    except (TypeError, ValueError) as exc:
        raise PacketError("claim has an invalid evidence or authority kind") from exc
    return {
        "authority": authority_kind.value,
        "evidence": evidence_kind.value,
        "group": _text(group, "claim.group"),
        "packet_id": _text(packet_id, "claim.packet_id"),
        "parameter": _text(parameter, "claim.parameter"),
        "uncertainty": _uncertainty_document(normalized_uncertainty),
        "unit": _text(unit, "claim.unit"),
        "value": canonical_decimal(value, "claim.value"),
    }


def claim_sha256(
    packet_id: str,
    group: str,
    parameter: str,
    value: str,
    unit: str,
    uncertainty: Mapping[str, object],
    evidence: str,
    authority: str,
) -> str:
    return hashlib.sha256(
        canonical_bytes(
            claim_document(
                packet_id,
                group,
                parameter,
                value,
                unit,
                uncertainty,
                evidence,
                authority,
            )
        )
    ).hexdigest()


def load_packet_json(
    source: str | bytes | bytearray,
    *,
    artifact_root: str | Path,
    authority_policy: AuthorityPolicy | None = None,
) -> ExecutablePacket:
    """Strictly parse JSON and validate every lifecycle claim it makes."""

    if isinstance(source, str):
        raw = source
    elif isinstance(source, (bytes, bytearray)):
        try:
            raw = bytes(source).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PacketError("packet is not valid UTF-8") from exc
    else:
        raise TypeError("source must be str, bytes, or bytearray")
    if len(raw.encode("utf-8")) > _MAX_PACKET_BYTES:
        raise PacketError("packet JSON exceeds size limit")
    try:
        document = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, PacketError, RecursionError) as exc:
        raise PacketError(f"invalid packet JSON: {exc}") from exc
    return load_packet(
        document,
        artifact_root=artifact_root,
        authority_policy=authority_policy,
    )


def load_packet(
    document: object,
    *,
    artifact_root: str | Path,
    authority_policy: AuthorityPolicy | None = None,
) -> ExecutablePacket:
    """Create an immutable packet only after validating its claimed state."""

    _validate_document_limits(document)
    root = Path(os.fspath(artifact_root))
    _verify_root_openable(root)
    obj = _object(document, "packet")
    _exact_keys(
        obj,
        {"schema_version", "packet_id", "state", "groups", "adjudication"},
        "packet",
    )
    if obj["schema_version"] != SCHEMA_VERSION:
        raise PacketError(f"unsupported schema_version: {obj['schema_version']!r}")
    packet_id = _text(obj["packet_id"], "packet.packet_id")
    try:
        state = PacketState(obj["state"])
    except (TypeError, ValueError) as exc:
        raise PacketError(f"invalid packet state: {obj['state']!r}") from exc
    groups = _load_groups(obj["groups"], packet_id)
    adjudication = _load_adjudication_binding(obj["adjudication"])
    if state in {PacketState.DRAFT, PacketState.STRUCTURAL, PacketState.ARTIFACTS_VERIFIED}:
        if adjudication is not None:
            raise PacketError(f"{state.value} packet cannot carry adjudication")
    elif adjudication is None:
        raise PacketError(f"{state.value} packet requires adjudication")
    if state in {PacketState.SCIENTIFICALLY_RESOLVED, PacketState.SEALED}:
        if type(authority_policy) is not AuthorityPolicy:
            raise PacketError(
                f"{state.value} packet requires an external AuthorityPolicy"
            )
    elif authority_policy is not None and type(authority_policy) is not AuthorityPolicy:
        raise PacketError("authority_policy must be AuthorityPolicy or None")

    packet = ExecutablePacket(
        _CONSTRUCTION_KEY,
        packet_id=packet_id,
        state=state,
        groups=groups,
        artifact_root=root,
        adjudication=adjudication,
        authority_policy=authority_policy,
    )
    _validate_structure(packet)
    if state in {
        PacketState.ARTIFACTS_VERIFIED,
        PacketState.SCIENTIFICALLY_RESOLVED,
        PacketState.SEALED,
    }:
        _verify_leaf_artifacts(packet)
    if state in {PacketState.SCIENTIFICALLY_RESOLVED, PacketState.SEALED}:
        adjudication_sha = _validate_scientific_resolution(
            packet, authority_policy
        )
    else:
        adjudication_sha = ""
    if state is PacketState.SEALED:
        receipt = ValidationReceipt(
            _CONSTRUCTION_KEY,
            packet.sha256,
            packet.structural_sha256,
            adjudication_sha,
            _authority_policy_sha256(authority_policy),
        )
        object.__setattr__(packet, "_receipt", receipt)
    return packet


def materialize_packet(
    packet: ExecutablePacket,
    receipt: ValidationReceipt,
) -> MaterializedPacket:
    """Revalidate current artifacts and return executable values immutably."""

    if type(packet) is not ExecutablePacket:
        raise PacketError("materialization requires an exact ExecutablePacket")
    if packet.state is not PacketState.SEALED:
        raise PacketError("materialization requires a SEALED packet")
    if type(receipt) is not ValidationReceipt or receipt is not packet._receipt:
        raise PacketError("materialization requires this packet's validation receipt")
    if (
        receipt._packet_sha256 != packet.sha256
        or receipt._structural_sha256 != packet.structural_sha256
    ):
        raise PacketError("validation receipt no longer binds this packet")
    if receipt._authority_policy_sha256 != _authority_policy_sha256(
        packet._authority_policy
    ):
        raise PacketError("validation receipt no longer binds its authority policy")
    _verify_leaf_artifacts(packet)
    adjudication_sha = _validate_scientific_resolution(
        packet, packet._authority_policy
    )
    if adjudication_sha != receipt._adjudication_sha256:
        raise PacketError("adjudication receipt changed after packet sealing")
    return MaterializedPacket(
        packet.packet_id,
        packet.sha256,
        packet.structural_sha256,
        _materialized_groups(packet._groups),
    )


def expected_adjudication_document(packet: ExecutablePacket) -> dict[str, object]:
    """Return the exact receipt body an independent adjudicator must sign off."""

    if type(packet) is not ExecutablePacket:
        raise TypeError("packet must be an ExecutablePacket")
    leaves = []
    decisions = []
    for group, parameters in sorted(packet._groups.items()):
        for parameter, leaf in sorted(parameters.items()):
            leaves.append(
                {
                    "authority": _binding_document(leaf.authority),
                    "evidence": _binding_document(leaf.evidence),
                    "group": group,
                    "parameter": parameter,
                }
            )
            decisions.append(
                {
                    "authority": leaf.authority.kind.value,
                    "decision": "compatible",
                    "evidence": leaf.evidence.kind.value,
                    "group": group,
                    "parameter": parameter,
                }
            )
    return {
        "compatibility_decisions": decisions,
        "leaf_bindings": leaves,
        "packet_id": packet.packet_id,
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "status": "accepted",
        "structural_packet_sha256": packet.structural_sha256,
    }


def canonical_decimal(value: object, name: str = "value") -> str:
    """Accept only the one bounded ASCII representation chosen for a number."""

    if not isinstance(value, str):
        raise PacketError(f"{name} must be a canonical decimal string")
    if len(value) > _MAX_DECIMAL_CHARS or not _DECIMAL_RE.fullmatch(value):
        raise PacketError(f"{name} must use bounded canonical ASCII decimal syntax")
    try:
        decimal = Decimal(value)
    except InvalidOperation as exc:
        raise PacketError(f"{name} is not a decimal string") from exc
    if not decimal.is_finite():
        raise PacketError(f"{name} must be finite")
    sign, digits_tuple, exponent = decimal.as_tuple()
    if len(digits_tuple) > _MAX_DECIMAL_DIGITS or abs(exponent) > _MAX_DECIMAL_EXPONENT:
        raise PacketError(f"{name} exceeds decimal size bounds")
    if decimal == 0:
        canonical = "0"
    else:
        with localcontext() as context:
            context.prec = _MAX_DECIMAL_DIGITS
            normalized = decimal.normalize(context)
        adjusted = normalized.adjusted()
        if -6 <= adjusted <= 20:
            canonical = format(normalized, "f")
        else:
            canonical = format(normalized, "e").replace("e+", "e")
    if len(canonical) > _MAX_EXPANDED_DECIMAL_CHARS:
        raise PacketError(f"{name} expanded representation is too long")
    if value != canonical:
        raise PacketError(f"{name} is not canonical; expected {canonical!r}")
    return value


def _validate_numeric_domain(
    value: str,
    domain: NumericDomain,
    name: str,
) -> None:
    decimal = Decimal(value)
    if domain.integer and decimal != decimal.to_integral_value():
        raise PacketError(f"{name} must be an integer")
    if domain.nonzero and decimal == 0:
        raise PacketError(f"{name} must be nonzero")
    if domain.lower is not None:
        lower = Decimal(domain.lower)
        if decimal < lower or (decimal == lower and not domain.lower_inclusive):
            operator = ">=" if domain.lower_inclusive else ">"
            raise PacketError(f"{name} must be {operator} {domain.lower}")
    if domain.upper is not None:
        upper = Decimal(domain.upper)
        if decimal > upper or (decimal == upper and not domain.upper_inclusive):
            operator = "<=" if domain.upper_inclusive else "<"
            raise PacketError(f"{name} must be {operator} {domain.upper}")


def _load_groups(value: object, packet_id: str) -> Mapping[str, Mapping[str, ParameterLeaf]]:
    groups_obj = _object(value, "packet.groups")
    actual_groups = set(groups_obj)
    if actual_groups != REQUIRED_GROUPS:
        raise PacketError(
            "packet groups mismatch; "
            f"missing={sorted(REQUIRED_GROUPS - actual_groups)}, "
            f"extra={sorted(actual_groups - REQUIRED_GROUPS)}"
        )
    groups: dict[str, Mapping[str, ParameterLeaf]] = {}
    for group_name, leaves_value in groups_obj.items():
        group = _text(group_name, "group name")
        leaves_obj = _object(leaves_value, f"groups.{group}")
        expected_parameters = set(PARAMETER_SCHEMA[group])
        actual_parameters = set(leaves_obj)
        if actual_parameters != expected_parameters:
            raise PacketError(
                f"group {group} parameters mismatch; "
                f"missing={sorted(expected_parameters - actual_parameters)}, "
                f"extra={sorted(actual_parameters - expected_parameters)}"
            )
        leaves: dict[str, ParameterLeaf] = {}
        for parameter_name, leaf_value in leaves_obj.items():
            parameter = _text(parameter_name, f"groups.{group} parameter name")
            leaf_obj = _object(leaf_value, f"groups.{group}.{parameter}")
            raw_unit = leaf_obj.get("unit")
            if isinstance(raw_unit, str) and raw_unit not in PARAMETER_SCHEMA[group][parameter]:
                raise PacketError(
                    f"unit {raw_unit!r} is not permitted for {group}.{parameter}"
                )
            leaves[parameter] = _load_leaf(
                leaf_obj,
                packet_id=packet_id,
                group=group,
                parameter=parameter,
            )
        groups[group] = MappingProxyType(leaves)
    return MappingProxyType(groups)


def _load_leaf(
    value: object,
    *,
    packet_id: str,
    group: str,
    parameter: str,
) -> ParameterLeaf:
    obj = _object(value, f"groups.{group}.{parameter}")
    _exact_keys(
        obj,
        {"value", "unit", "uncertainty", "evidence", "authority"},
        f"groups.{group}.{parameter}",
    )
    decimal_value = canonical_decimal(obj["value"], f"groups.{group}.{parameter}.value")
    _validate_numeric_domain(
        decimal_value,
        PARAMETER_DOMAINS[group][parameter],
        f"groups.{group}.{parameter}.value",
    )
    unit = _text(obj["unit"], f"groups.{group}.{parameter}.unit")
    uncertainty = _load_uncertainty(
        obj["uncertainty"], unit, f"groups.{group}.{parameter}.uncertainty"
    )
    evidence_obj = _binding_object(obj["evidence"], f"groups.{group}.{parameter}.evidence")
    authority_obj = _binding_object(obj["authority"], f"groups.{group}.{parameter}.authority")
    try:
        evidence_kind = EvidenceKind(evidence_obj["kind"])
        authority_kind = AuthorityKind(authority_obj["kind"])
    except (TypeError, ValueError) as exc:
        raise PacketError(f"invalid binding kind for {group}.{parameter}") from exc
    expected_claim = claim_sha256(
        packet_id,
        group,
        parameter,
        decimal_value,
        unit,
        _uncertainty_document(uncertainty),
        evidence_kind.value,
        authority_kind.value,
    )
    evidence = EvidenceBinding(
        kind=evidence_kind,
        **_binding_fields(evidence_obj, expected_claim),
    )
    authority = AuthorityBinding(
        kind=authority_kind,
        **_binding_fields(authority_obj, expected_claim),
    )
    return ParameterLeaf(decimal_value, unit, uncertainty, evidence, authority)


def _load_uncertainty(value: object, leaf_unit: str, name: str) -> Uncertainty:
    obj = _object(value, name)
    _exact_keys(obj, {"kind", "lower", "upper", "unit"}, name)
    try:
        kind = UncertaintyKind(obj["kind"])
    except (TypeError, ValueError) as exc:
        raise PacketError(f"{name}.kind is invalid") from exc
    if kind is UncertaintyKind.NOT_REPORTED:
        if any(obj[field] is not None for field in ("lower", "upper", "unit")):
            raise PacketError(f"{name} not_reported fields must be null")
        return Uncertainty(kind, None, None, None)
    lower = canonical_decimal(obj["lower"], f"{name}.lower")
    upper = canonical_decimal(obj["upper"], f"{name}.upper")
    unit = _text(obj["unit"], f"{name}.unit")
    if unit != leaf_unit:
        raise PacketError(f"{name}.unit must match the parameter unit")
    if Decimal(lower) > Decimal(upper):
        raise PacketError(f"{name} interval lower exceeds upper")
    return Uncertainty(kind, lower, upper, unit)


def _binding_object(value: object, name: str) -> Mapping[str, object]:
    obj = _object(value, name)
    _exact_keys(
        obj,
        {"kind", "artifact_path", "artifact_sha256", "locator", "claim_sha256"},
        name,
    )
    return obj


def _binding_fields(obj: Mapping[str, object], expected_claim: str) -> dict[str, str]:
    fields = {
        name: _text(obj[name], f"binding.{name}")
        for name in ("artifact_path", "artifact_sha256", "locator", "claim_sha256")
    }
    _sha256(fields["artifact_sha256"], "binding.artifact_sha256")
    _sha256(fields["claim_sha256"], "binding.claim_sha256")
    if fields["claim_sha256"] != expected_claim:
        raise PacketError("binding claim_sha256 does not match its executable claim")
    _relative_components(fields["artifact_path"])
    _json_pointer(fields["locator"])
    return fields


def _load_adjudication_binding(value: object) -> AdjudicationBinding | None:
    if value is None:
        return None
    obj = _object(value, "packet.adjudication")
    _exact_keys(
        obj,
        {"artifact_path", "artifact_sha256", "locator", "receipt_sha256"},
        "packet.adjudication",
    )
    fields = {
        name: _text(obj[name], f"packet.adjudication.{name}")
        for name in ("artifact_path", "artifact_sha256", "locator", "receipt_sha256")
    }
    _sha256(fields["artifact_sha256"], "packet.adjudication.artifact_sha256")
    _sha256(fields["receipt_sha256"], "packet.adjudication.receipt_sha256")
    _relative_components(fields["artifact_path"])
    _json_pointer(fields["locator"])
    return AdjudicationBinding(**fields)


def _validate_structure(packet: ExecutablePacket) -> None:
    actual_groups = set(packet._groups)
    if actual_groups != REQUIRED_GROUPS:
        raise PacketError(
            "packet groups mismatch; "
            f"missing={sorted(REQUIRED_GROUPS - actual_groups)}, "
            f"extra={sorted(actual_groups - REQUIRED_GROUPS)}"
        )
    for group, schema in PARAMETER_SCHEMA.items():
        leaves = packet._groups[group]
        actual_parameters = set(leaves)
        expected_parameters = set(schema)
        if actual_parameters != expected_parameters:
            raise PacketError(
                f"group {group} parameters mismatch; "
                f"missing={sorted(expected_parameters - actual_parameters)}, "
                f"extra={sorted(actual_parameters - expected_parameters)}"
            )
        for parameter, leaf in leaves.items():
            if leaf.unit not in schema[parameter]:
                raise PacketError(
                    f"unit {leaf.unit!r} is not permitted for {group}.{parameter}"
                )

    fast_hh = packet._groups["fast_hh"]
    if Decimal(fast_hh["spike_detection_voltage"].value) <= Decimal(
        fast_hh["initial_voltage"].value
    ):
        raise PacketError(
            "fast_hh.spike_detection_voltage must exceed initial_voltage"
        )
    nap = packet._groups["nap"]
    for gate in ("activation", "inactivation"):
        if Decimal(nap[f"{gate}_tau_min"].value) > Decimal(
            nap[f"{gate}_tau_max"].value
        ):
            raise PacketError(f"nap.{gate}_tau_min must not exceed tau_max")
    calcium = packet._groups["calcium"]
    ionic = packet._groups["ionic_env"]
    if Decimal(calcium["baseline"].value) >= Decimal(
        ionic["extracellular_calcium"].value
    ):
        raise PacketError(
            "calcium.baseline must be below ionic_env.extracellular_calcium"
        )


def _verify_leaf_artifacts(packet: ExecutablePacket) -> None:
    _validate_structure(packet)
    expected: list[tuple[EvidenceBinding | AuthorityBinding, dict[str, object]]] = []
    for group, leaves in packet._groups.items():
        for parameter, leaf in leaves.items():
            claim = claim_document(
                packet.packet_id,
                group,
                parameter,
                leaf.value,
                leaf.unit,
                _uncertainty_document(leaf.uncertainty),
                leaf.evidence.kind.value,
                leaf.authority.kind.value,
            )
            expected.extend(((leaf.evidence, claim), (leaf.authority, claim)))
    with _ArtifactReader(packet.artifact_root) as reader:
        for binding, claim in expected:
            raw = reader.read(binding.artifact_path)
            if hashlib.sha256(raw).hexdigest() != binding.artifact_sha256:
                raise PacketError(f"artifact digest mismatch: {binding.artifact_path}")
            extracted = _extract_json_pointer(raw, binding.locator, binding.artifact_path)
            if extracted != claim:
                raise PacketError(
                    f"artifact locator does not bind executable claim: {binding.artifact_path}"
                )
            extracted_digest = hashlib.sha256(canonical_bytes(extracted)).hexdigest()
            if extracted_digest != binding.claim_sha256:
                raise PacketError(
                    f"located claim digest mismatch: {binding.artifact_path}"
                )


def _validate_scientific_resolution(
    packet: ExecutablePacket,
    authority_policy: AuthorityPolicy | None,
) -> str:
    if type(authority_policy) is not AuthorityPolicy:
        raise PacketError("scientific resolution requires an external AuthorityPolicy")
    for group, leaves in packet._groups.items():
        for parameter, leaf in leaves.items():
            expected_authority = _COMPATIBLE_BINDINGS[leaf.evidence.kind]
            if leaf.evidence.kind is EvidenceKind.UNRESOLVED:
                raise PacketError(f"unresolved evidence for {group}.{parameter}")
            if leaf.authority.kind is not expected_authority:
                raise PacketError(
                    f"incompatible evidence/authority for {group}.{parameter}: "
                    f"{leaf.evidence.kind.value}/{leaf.authority.kind.value}"
                )
            trusted_claim = TrustedClaim(
                leaf.authority.kind,
                leaf.authority.artifact_sha256,
                leaf.authority.claim_sha256,
            )
            if trusted_claim not in authority_policy.trusted_claims:
                raise PacketError(
                    f"authority policy does not approve {group}.{parameter}"
                )
    binding = packet._adjudication
    if binding is None:
        raise PacketError("scientific resolution requires adjudication")
    leaf_paths = {
        item.artifact_path
        for leaves in packet._groups.values()
        for leaf in leaves.values()
        for item in (leaf.evidence, leaf.authority)
    }
    if binding.artifact_path in leaf_paths:
        raise PacketError("adjudication must be an independent artifact")
    with _ArtifactReader(packet.artifact_root) as reader:
        raw = reader.read(binding.artifact_path)
    if hashlib.sha256(raw).hexdigest() != binding.artifact_sha256:
        raise PacketError(f"artifact digest mismatch: {binding.artifact_path}")
    extracted = _extract_json_pointer(raw, binding.locator, binding.artifact_path)
    expected = expected_adjudication_document(packet)
    if extracted != expected:
        raise PacketError("adjudication receipt does not bind the complete packet")
    digest = hashlib.sha256(canonical_bytes(extracted)).hexdigest()
    if digest != binding.receipt_sha256:
        raise PacketError("adjudication receipt digest mismatch")
    if digest not in authority_policy.trusted_adjudication_receipts:
        raise PacketError("authority policy does not approve adjudication receipt")
    return digest


class _ArtifactReader:
    """Descriptor-relative, no-follow reader rooted at one open directory."""

    def __init__(self, root: Path):
        self._root = root
        self._root_fd = -1
        self._cache: dict[str, bytes] = {}

    def __enter__(self) -> "_ArtifactReader":
        try:
            self._root_fd = os.open(
                self._root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
        except OSError as exc:
            raise PacketError(f"cannot open artifact root safely: {self._root}") from exc
        return self

    def __exit__(self, *args: object) -> None:
        if self._root_fd >= 0:
            os.close(self._root_fd)
            self._root_fd = -1

    def read(self, relative: str) -> bytes:
        cached = self._cache.get(relative)
        if cached is not None:
            return cached
        components = _relative_components(relative)
        current_fd = os.dup(self._root_fd)
        try:
            for component in components[:-1]:
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=current_fd,
                )
                os.close(current_fd)
                current_fd = next_fd
            file_fd = os.open(
                components[-1],
                os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current_fd,
            )
            try:
                info = os.fstat(file_fd)
                if not stat.S_ISREG(info.st_mode):
                    raise PacketError(f"artifact is not a regular file: {relative}")
                if info.st_size > _MAX_ARTIFACT_BYTES:
                    raise PacketError(f"artifact exceeds size limit: {relative}")
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = os.read(file_fd, min(1024 * 1024, _MAX_ARTIFACT_BYTES + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > _MAX_ARTIFACT_BYTES:
                        raise PacketError(f"artifact exceeds size limit: {relative}")
                    chunks.append(chunk)
                raw = b"".join(chunks)
            finally:
                os.close(file_fd)
        except OSError as exc:
            raise PacketError(f"cannot safely open artifact: {relative}") from exc
        finally:
            os.close(current_fd)
        self._cache[relative] = raw
        return raw


def _verify_root_openable(root: Path) -> None:
    try:
        fd = os.open(
            root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise PacketError("artifact_root must be a no-follow directory") from exc
    os.close(fd)


def _relative_components(value: str) -> tuple[str, ...]:
    if len(value) > _MAX_PATH_CHARS:
        raise PacketError("artifact_path exceeds size limit")
    if "\\" in value or "\x00" in value:
        raise PacketError("artifact_path contains a forbidden character")
    path = PurePosixPath(value)
    components = path.parts
    if path.is_absolute() or not components or any(part in {"", ".", ".."} for part in components):
        raise PacketError("artifact_path must stay below artifact_root")
    if str(path) != value:
        raise PacketError("artifact_path must be canonical POSIX relative syntax")
    if any(len(component) > _MAX_PATH_COMPONENT_CHARS for component in components):
        raise PacketError("artifact_path component exceeds size limit")
    return components


def _json_pointer(locator: str) -> str:
    if len(locator) > _MAX_POINTER_CHARS:
        raise PacketError("artifact locator exceeds size limit")
    prefix = "json-pointer:"
    if not locator.startswith(prefix):
        raise PacketError("artifact locator must use json-pointer:")
    pointer = locator[len(prefix) :]
    if not pointer.startswith("/"):
        raise PacketError("JSON Pointer locator must begin with '/'")
    if any(
        len(token) > _MAX_POINTER_TOKEN_CHARS
        for token in pointer[1:].split("/")
    ):
        raise PacketError("JSON Pointer token exceeds size limit")
    for index, character in enumerate(pointer):
        if character == "~" and (index + 1 == len(pointer) or pointer[index + 1] not in "01"):
            raise PacketError("JSON Pointer contains an invalid escape")
    return pointer


def _extract_json_pointer(raw: bytes, locator: str, path: str) -> object:
    pointer = _json_pointer(locator)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        for encoded in pointer[1:].split("/"):
            token = encoded.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict):
                value = value[token]
            elif isinstance(value, list) and token.isascii() and token.isdigit():
                if token != "0" and token.startswith("0"):
                    raise KeyError(token)
                if len(token) > _MAX_ARRAY_INDEX_DIGITS:
                    raise KeyError(token)
                value = value[int(token)]
            else:
                raise KeyError(token)
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        PacketError,
        KeyError,
        IndexError,
        ValueError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise PacketError(f"JSON Pointer locator not found in artifact: {path}") from exc
    return value


def _structural_document(packet: ExecutablePacket) -> dict[str, object]:
    return {
        "groups": _groups_document(packet._groups),
        "packet_id": packet.packet_id,
        "schema_version": packet.schema_version,
    }


def _packet_document(packet: ExecutablePacket) -> dict[str, object]:
    return {
        **_structural_document(packet),
        "adjudication": _adjudication_document(packet._adjudication),
        "state": packet.state.value,
    }


def _groups_document(groups: Mapping[str, Mapping[str, ParameterLeaf]]) -> dict[str, object]:
    return {
        group: {
            parameter: {
                "authority": _binding_document(leaf.authority),
                "evidence": _binding_document(leaf.evidence),
                "uncertainty": _uncertainty_document(leaf.uncertainty),
                "unit": leaf.unit,
                "value": leaf.value,
            }
            for parameter, leaf in leaves.items()
        }
        for group, leaves in groups.items()
    }


def _materialized_groups(
    groups: Mapping[str, Mapping[str, ParameterLeaf]],
) -> Mapping[str, Mapping[str, MaterializedParameterLeaf]]:
    snapshot: dict[str, Mapping[str, MaterializedParameterLeaf]] = {}
    for group, leaves in groups.items():
        snapshot[group] = MappingProxyType(
            {
                parameter: MaterializedParameterLeaf(
                    leaf.value,
                    leaf.unit,
                    MaterializedUncertainty(
                        leaf.uncertainty.kind,
                        leaf.uncertainty.lower,
                        leaf.uncertainty.upper,
                        leaf.uncertainty.unit,
                    ),
                    leaf.evidence.kind,
                    leaf.authority.kind,
                )
                for parameter, leaf in leaves.items()
            }
        )
    return MappingProxyType(snapshot)


def _authority_policy_sha256(policy: AuthorityPolicy | None) -> str:
    if type(policy) is not AuthorityPolicy:
        raise PacketError("sealed packet has no external authority policy")
    document = {
        "policy_id": policy.policy_id,
        "trusted_adjudication_receipts": sorted(
            policy.trusted_adjudication_receipts
        ),
        "trusted_claims": [
            {
                "artifact_sha256": claim.artifact_sha256,
                "authority": claim.authority.value,
                "claim_sha256": claim.claim_sha256,
            }
            for claim in sorted(
                policy.trusted_claims,
                key=lambda item: (
                    item.authority.value,
                    item.artifact_sha256,
                    item.claim_sha256,
                ),
            )
        ],
    }
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def _validate_document_limits(value: object) -> None:
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_DOCUMENT_NODES:
            raise PacketError("packet document exceeds node limit")
        if depth > _MAX_DOCUMENT_DEPTH:
            raise PacketError("packet document exceeds nesting limit")
        if isinstance(item, str):
            if len(item) > _MAX_TEXT_CHARS:
                raise PacketError("packet text field exceeds size limit")
        elif isinstance(item, dict):
            stack.extend((key, depth + 1) for key in item)
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend((child, depth + 1) for child in item)


def _binding_document(binding: EvidenceBinding | AuthorityBinding) -> dict[str, str]:
    return {
        "artifact_path": binding.artifact_path,
        "artifact_sha256": binding.artifact_sha256,
        "claim_sha256": binding.claim_sha256,
        "kind": binding.kind.value,
        "locator": binding.locator,
    }


def _adjudication_document(binding: AdjudicationBinding | None) -> dict[str, str] | None:
    if binding is None:
        return None
    return {
        "artifact_path": binding.artifact_path,
        "artifact_sha256": binding.artifact_sha256,
        "locator": binding.locator,
        "receipt_sha256": binding.receipt_sha256,
    }


def _uncertainty_document(value: Uncertainty) -> dict[str, str | None]:
    return {
        "kind": value.kind.value,
        "lower": value.lower,
        "unit": value.unit,
        "upper": value.upper,
    }


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PacketError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise PacketError(f"nonfinite JSON number is forbidden: {token}")


def _object(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise PacketError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise PacketError(
            f"{name} fields mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _text(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_TEXT_CHARS
        or value != value.strip()
        or any(ord(character) > 127 for character in value)
    ):
        raise PacketError(f"{name} must be nonempty trimmed ASCII text")
    return value


def _sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _HEX_LENGTH
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PacketError(f"{name} must be a lowercase SHA-256 digest")


__all__ = [
    "ADJUDICATION_SCHEMA_VERSION",
    "AdjudicationBinding",
    "AuthorityBinding",
    "AuthorityKind",
    "AuthorityPolicy",
    "EvidenceBinding",
    "EvidenceKind",
    "ExecutablePacket",
    "MaterializedPacket",
    "MaterializedParameterLeaf",
    "MaterializedUncertainty",
    "NumericDomain",
    "PARAMETER_DOMAINS",
    "PARAMETER_SCHEMA",
    "PacketError",
    "PacketState",
    "ParameterLeaf",
    "REQUIRED_GROUPS",
    "SCHEMA_VERSION",
    "TrustedClaim",
    "Uncertainty",
    "UncertaintyKind",
    "ValidationReceipt",
    "canonical_bytes",
    "canonical_decimal",
    "claim_document",
    "claim_sha256",
    "expected_adjudication_document",
    "load_packet",
    "load_packet_json",
    "materialize_packet",
]
