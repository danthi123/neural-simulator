#!/usr/bin/env python3
"""Compile an authenticated failed analysis into a bounded research gate."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

from tools import research_escalation


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_SCHEMA = "sim-analysis-research-handoff-contract-v1"
RECEIPT_SCHEMA = "sim-analysis-research-handoff-receipt-v1"


class AnalysisResearchHandoffError(ValueError):
    """Raised when analysis evidence cannot authorize a research handoff."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _semantic_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256(value: Any, context: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise AnalysisResearchHandoffError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _inside_file(root: Path, value: Any, context: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise AnalysisResearchHandoffError(f"{context} path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or str(relative) != value or any(part in {"", ".", ".."} for part in relative.parts):
        raise AnalysisResearchHandoffError(f"{context} path is not canonical")
    unresolved = root.joinpath(*relative.parts)
    if unresolved.is_symlink():
        raise AnalysisResearchHandoffError(f"{context} must not be a symbolic link")
    path = unresolved.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AnalysisResearchHandoffError(f"{context} escapes repository root") from exc
    if not path.is_file():
        raise AnalysisResearchHandoffError(f"{context} must be a regular file")
    return value, path


def _inside_output(root: Path, value: Any, context: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise AnalysisResearchHandoffError(f"{context} path is invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or str(relative) != value or any(part in {"", ".", ".."} for part in relative.parts):
        raise AnalysisResearchHandoffError(f"{context} path is not canonical")
    path = root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AnalysisResearchHandoffError(f"{context} escapes repository root") from exc
    return value, path


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisResearchHandoffError(f"{context} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise AnalysisResearchHandoffError(f"{context} must contain an object")
    return value


def _path_value(document: Any, path: Any, context: str) -> Any:
    if not isinstance(path, list) or not path or not all(isinstance(item, (str, int)) for item in path):
        raise AnalysisResearchHandoffError(f"{context} must be a non-empty string/integer path")
    value = document
    for item in path:
        try:
            value = value[item]
        except (KeyError, IndexError, TypeError) as exc:
            raise AnalysisResearchHandoffError(f"{context} does not exist in the analysis") from exc
    return value


def _load_contract(root: Path, path_value: str | Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, str]]:
    supplied = Path(path_value).expanduser()
    path = (supplied if supplied.is_absolute() else root / supplied).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise AnalysisResearchHandoffError("contract escapes repository root") from exc
    _, path = _inside_file(root, relative, "contract")
    raw = path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != _sha256(expected_sha256, "contract sha256"):
        raise AnalysisResearchHandoffError("contract digest does not match")
    try:
        contract = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnalysisResearchHandoffError("contract is not valid JSON") from exc
    body = {key: value for key, value in contract.items() if key != "sha256"}
    if contract.get("sha256") != _semantic_digest(body):
        raise AnalysisResearchHandoffError("contract self digest is invalid")
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status") != "preregistered"
        or contract.get("scientific_verdict") is not None
        or contract.get("source_claim_acceptance_allowed") is not False
        or contract.get("successor_dispatch_allowed") is not False
    ):
        raise AnalysisResearchHandoffError("contract identity or authority is invalid")
    return contract, {"path": relative, "sha256": file_sha, "self_sha256": contract["sha256"]}


def _load_analysis(
    root: Path, binding: Any
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    if not isinstance(binding, Mapping) or set(binding) != {
        "path", "sha256", "schema", "governing_binding", "provenance"
    }:
        raise AnalysisResearchHandoffError("analysis binding is invalid")
    relative, path = _inside_file(root, binding["path"], "analysis")
    raw = path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != _sha256(binding["sha256"], "analysis sha256"):
        raise AnalysisResearchHandoffError("analysis digest does not match")
    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AnalysisResearchHandoffError("analysis is not valid JSON") from exc
    body = {key: value for key, value in analysis.items() if key != "sha256"}
    if analysis.get("schema") != binding["schema"] or analysis.get("sha256") != _semantic_digest(body):
        raise AnalysisResearchHandoffError("analysis schema or self digest is invalid")
    governing = binding["governing_binding"]
    if not isinstance(governing, Mapping) or set(governing) != {"path", "equals"}:
        raise AnalysisResearchHandoffError("analysis governing binding is invalid")
    if _path_value(analysis, governing["path"], "governing binding path") != governing["equals"]:
        raise AnalysisResearchHandoffError("analysis governing binding does not match")
    expected_governing = governing["equals"]
    if not isinstance(expected_governing, Mapping) or set(expected_governing) != {"path", "sha256"}:
        raise AnalysisResearchHandoffError("governing execution binding is invalid")
    _, governing_path = _inside_file(root, expected_governing["path"], "governing execution spec")
    if _file_digest(governing_path) != _sha256(
        expected_governing["sha256"], "governing execution spec sha256"
    ):
        raise AnalysisResearchHandoffError("governing execution spec digest does not match")
    provenance = binding["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {"path", "sha256", "runner", "backend"}:
        raise AnalysisResearchHandoffError("analysis provenance binding is invalid")
    provenance_relative, provenance_path = _inside_file(root, provenance["path"], "analysis provenance")
    provenance_raw = provenance_path.read_bytes()
    provenance_sha = hashlib.sha256(provenance_raw).hexdigest()
    if provenance_sha != _sha256(provenance["sha256"], "analysis provenance sha256"):
        raise AnalysisResearchHandoffError("analysis provenance digest does not match")
    try:
        sidecar = json.loads(provenance_raw)
    except json.JSONDecodeError as exc:
        raise AnalysisResearchHandoffError("analysis provenance is not valid JSON") from exc
    argv = sidecar.get("argv")
    if (
        sidecar.get("artifact") != relative
        or sidecar.get("runner") != provenance["runner"]
        or sidecar.get("sim_backend_requested") != provenance["backend"]
        or sidecar.get("sim_backend") != provenance["backend"]
        or not isinstance(sidecar.get("run_id"), str) or not sidecar["run_id"]
        or not isinstance(argv, list)
        or "--out" not in argv
        or argv[argv.index("--out") + 1:] == []
        or argv[argv.index("--out") + 1] != relative
    ):
        raise AnalysisResearchHandoffError("analysis provenance identity is invalid")
    return (
        analysis,
        {"path": relative, "sha256": file_sha, "self_sha256": analysis["sha256"]},
        {
            "path": provenance_relative, "sha256": provenance_sha,
            "run_id": sidecar["run_id"], "runner": sidecar["runner"],
            "git_sha": sidecar.get("git_sha"), "git_dirty": sidecar.get("git_dirty"),
        },
    )


def _validate_contract(
    contract: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], list[dict[str, str]], tuple[Path, str]]:
    if set(contract) != {
        "schema", "status", "scientific_verdict", "source_claim_acceptance_allowed",
        "successor_dispatch_allowed", "implementation", "analysis", "trigger", "research_gate",
        "prior_attempts", "questions", "receipt_output", "sha256",
    }:
        raise AnalysisResearchHandoffError("contract field set is invalid")
    implementation = contract.get("implementation")
    if not isinstance(implementation, Mapping) or set(implementation) != {"compiler", "research_escalation"}:
        raise AnalysisResearchHandoffError("implementation binding set is invalid")
    for name, binding in implementation.items():
        if not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}:
            raise AnalysisResearchHandoffError(f"{name} implementation binding is invalid")
        _, path = _inside_file(root, binding["path"], f"{name} implementation")
        if _file_digest(path) != _sha256(binding["sha256"], f"{name} implementation sha256"):
            raise AnalysisResearchHandoffError(f"{name} implementation digest does not match")
    escalation_path = _inside_file(
        root, implementation["research_escalation"]["path"], "research_escalation implementation"
    )[1]
    if Path(research_escalation.__file__).resolve() != escalation_path:
        raise AnalysisResearchHandoffError(
            "loaded research_escalation module is not the authenticated implementation"
        )
    escalation_sha = implementation["research_escalation"]["sha256"]
    trigger = contract.get("trigger")
    gate = contract.get("research_gate")
    questions = contract.get("questions")
    attempts = contract.get("prior_attempts")
    if not isinstance(trigger, Mapping) or set(trigger) != {
        "verdict_path", "allowed_verdicts", "items_path", "item_id_field",
        "failed_field", "failed_value", "expected_failed_ids", "reject_unmapped",
    }:
        raise AnalysisResearchHandoffError("trigger declaration is invalid")
    if (
        not isinstance(trigger["allowed_verdicts"], list) or not trigger["allowed_verdicts"]
        or not all(isinstance(value, str) and value for value in trigger["allowed_verdicts"])
        or not isinstance(trigger["item_id_field"], str) or not trigger["item_id_field"]
        or not isinstance(trigger["failed_field"], str) or not trigger["failed_field"]
        or trigger["reject_unmapped"] is not True
        or not isinstance(trigger["expected_failed_ids"], list) or not trigger["expected_failed_ids"]
        or len(set(trigger["expected_failed_ids"])) != len(trigger["expected_failed_ids"])
    ):
        raise AnalysisResearchHandoffError("trigger fields are invalid")
    if not isinstance(gate, Mapping) or set(gate) != {
        "slug", "title", "blocked_experiment", "wall_reason", "query", "output"
    } or not all(isinstance(gate[key], str) and gate[key].strip() for key in gate):
        raise AnalysisResearchHandoffError("research gate declaration is invalid")
    _inside_output(root, gate["output"], "research gate output")
    _inside_output(root, contract["receipt_output"], "receipt output")
    if not isinstance(attempts, list) or len(attempts) < 2:
        raise AnalysisResearchHandoffError("at least two prior attempts are required")
    authenticated_attempts = []
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, Mapping) or set(attempt) != {"path", "sha256", "summary"}:
            raise AnalysisResearchHandoffError(f"prior attempt {index} binding is invalid")
        relative, path = _inside_file(root, attempt["path"], f"prior attempt {index}")
        if _file_digest(path) != _sha256(attempt["sha256"], f"prior attempt {index} sha256"):
            raise AnalysisResearchHandoffError(f"prior attempt {index} digest does not match")
        if not isinstance(attempt["summary"], str) or not attempt["summary"].strip():
            raise AnalysisResearchHandoffError(f"prior attempt {index} summary is invalid")
        authenticated_attempts.append({"path": relative, "sha256": attempt["sha256"], "summary": attempt["summary"].strip()})
    if not isinstance(questions, list) or not questions:
        raise AnalysisResearchHandoffError("questions must be a non-empty list")
    seen = set()
    kind_counts = {"parameter": 0, "wiring": 0}
    for question in questions:
        if not isinstance(question, Mapping) or set(question) != {"id", "kind", "text", "trigger_ids"}:
            raise AnalysisResearchHandoffError("question declaration is invalid")
        if question["kind"] not in {"parameter", "wiring"} or not isinstance(question["text"], str) or not question["text"].strip():
            raise AnalysisResearchHandoffError("question kind or text is invalid")
        if not isinstance(question["id"], str) or not question["id"] or question["id"] in seen:
            raise AnalysisResearchHandoffError("question IDs must be unique non-empty strings")
        seen.add(question["id"])
        kind_counts[question["kind"]] += 1
        expected_id = ("P" if question["kind"] == "parameter" else "W") + str(
            kind_counts[question["kind"]]
        )
        if question["id"] != expected_id:
            raise AnalysisResearchHandoffError(
                "question IDs must match the research gate's sequential P/W identity"
            )
        if (
            not isinstance(question["trigger_ids"], list) or not question["trigger_ids"]
            or any(value not in trigger["expected_failed_ids"] for value in question["trigger_ids"])
        ):
            raise AnalysisResearchHandoffError("question trigger IDs are invalid")
    return dict(trigger), authenticated_attempts, (escalation_path, escalation_sha)


@contextmanager
def _handoff_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".handoff.lock")
    with lock_path.open("a+", encoding="ascii") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield


def _expected_gate_questions(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    return [
        {"id": question["id"], "kind": question["kind"], "status": "open", "text": question["text"]}
        for question in selected
    ]


def _validate_existing_gate(
    path: Path,
    gate: Mapping[str, str],
    attempts: Sequence[Mapping[str, str]],
    selected: Sequence[Mapping[str, Any]],
) -> None:
    try:
        state = research_escalation._load(path)
    except Exception as exc:
        raise AnalysisResearchHandoffError("existing partial research gate is invalid") from exc
    expected = {
        "slug": gate["slug"],
        "title": gate["title"],
        "blocked_experiment": gate["blocked_experiment"],
        "wall_reason": gate["wall_reason"],
        "query": gate["query"],
        "failed_attempts": [f"{attempt['summary']} ({attempt['path']})" for attempt in attempts],
        "questions": _expected_gate_questions(selected),
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise AnalysisResearchHandoffError("existing partial research gate does not match the contract")


def compile_handoff(
    contract_path: str | Path,
    contract_sha256: str,
    output_path: str | Path,
    *,
    repository_root: str | Path = ROOT,
) -> dict[str, Any]:
    """Create one research gate and a receipt from an authenticated failed analysis."""
    root = Path(repository_root).expanduser().resolve(strict=True)
    contract, contract_binding = _load_contract(root, contract_path, contract_sha256)
    trigger, attempts, escalation_binding = _validate_contract(contract, root)
    analysis, analysis_binding, provenance_binding = _load_analysis(root, contract["analysis"])
    verdict = _path_value(analysis, trigger["verdict_path"], "verdict path")
    if verdict not in trigger["allowed_verdicts"]:
        raise AnalysisResearchHandoffError("analysis verdict does not authorize research escalation")
    items = _path_value(analysis, trigger["items_path"], "items path")
    if not isinstance(items, list) or not all(isinstance(item, Mapping) for item in items):
        raise AnalysisResearchHandoffError("analysis trigger items are invalid")
    identifiers = []
    failed = []
    for item in items:
        identifier = item.get(trigger["item_id_field"])
        passed = item.get(trigger["failed_field"])
        if (
            not isinstance(identifier, str) or not identifier or identifier in identifiers
            or not isinstance(passed, bool)
        ):
            raise AnalysisResearchHandoffError("analysis trigger item IDs are invalid or duplicated")
        for value in item.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise AnalysisResearchHandoffError("analysis trigger item contains a non-finite value")
        identifiers.append(identifier)
        if passed == trigger["failed_value"]:
            failed.append(identifier)
    if analysis.get("failed_metric_count") != len(failed):
        raise AnalysisResearchHandoffError("analysis failed metric count is inconsistent")
    if set(failed) != set(trigger["expected_failed_ids"]):
        raise AnalysisResearchHandoffError("observed failed IDs differ from the preregistered failure set")
    selected = [
        question for question in contract["questions"]
        if set(question["trigger_ids"]).intersection(failed)
    ]
    covered = {identifier for question in selected for identifier in question["trigger_ids"]}
    if trigger["reject_unmapped"] and set(failed) - covered:
        raise AnalysisResearchHandoffError("one or more failed IDs have no preregistered research question")
    parameter_questions = [question["text"] for question in selected if question["kind"] == "parameter"]
    wiring_questions = [question["text"] for question in selected if question["kind"] == "wiring"]
    if not parameter_questions or not wiring_questions:
        raise AnalysisResearchHandoffError("selected handoff needs parameter and wiring questions")

    gate = contract["research_gate"]
    gate_relative, gate_path = _inside_output(root, gate["output"], "research gate output")
    receipt_supplied = Path(output_path).expanduser()
    receipt_path = (receipt_supplied if receipt_supplied.is_absolute() else root / receipt_supplied).resolve()
    try:
        receipt_relative = receipt_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise AnalysisResearchHandoffError("receipt output escapes repository root") from exc
    if receipt_relative != contract["receipt_output"]:
        raise AnalysisResearchHandoffError("receipt output differs from the contract")
    arguments = SimpleNamespace(
        slug=gate["slug"], title=gate["title"], blocked_experiment=gate["blocked_experiment"],
        wall_reason=gate["wall_reason"],
        failed_attempt=[f"{attempt['summary']} ({attempt['path']})" for attempt in attempts],
        parameter_question=parameter_questions, wiring_question=wiring_questions,
        query=gate["query"], output=gate_relative,
    )
    with _handoff_lock(receipt_path):
        if receipt_path.exists():
            raise AnalysisResearchHandoffError("refusing to overwrite an existing receipt")
        created_here = False
        if gate_path.exists():
            _validate_existing_gate(gate_path, gate, attempts, selected)
        else:
            try:
                created_gate = research_escalation.start(arguments, root)
            except research_escalation.EscalationError as exc:
                raise AnalysisResearchHandoffError(f"research escalation failed: {exc}") from exc
            if created_gate.resolve() != gate_path or not gate_path.is_file():
                raise AnalysisResearchHandoffError("research escalation created the wrong artifact")
            created_here = True
        gate_sha = _file_digest(gate_path)
        if _file_digest(escalation_binding[0]) != escalation_binding[1]:
            raise AnalysisResearchHandoffError(
                "research_escalation implementation changed during handoff"
            )
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "contract": contract_binding,
            "analysis": analysis_binding,
            "analysis_provenance": provenance_binding,
            "triggered_verdict": verdict,
            "failed_ids": sorted(failed),
            "selected_question_ids": [question["id"] for question in selected],
            "prior_attempts": attempts,
            "research_gate": {"path": gate_relative, "sha256": gate_sha},
            "source_claims_accepted": False,
            "successor_dispatched": False,
            "scientific_verdict": None,
        }
        receipt["sha256"] = _semantic_digest(receipt)
        try:
            _write_new(receipt_path, receipt)
        except BaseException:
            if created_here and gate_path.is_file() and _file_digest(gate_path) == gate_sha:
                gate_path.unlink()
            raise
        return receipt


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise AnalysisResearchHandoffError(f"refusing to overwrite existing output: {path}")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args(argv)
    receipt = compile_handoff(arguments.contract, arguments.contract_sha256, arguments.out)
    print(json.dumps({"output": arguments.out, "sha256": receipt["sha256"], "failed_ids": receipt["failed_ids"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
