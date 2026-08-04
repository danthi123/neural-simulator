#!/usr/bin/env python3
"""The experiment harness: you cannot get a VERDICT out of it without passing the gates.

WHY THIS EXISTS (2026-07-31, owner diagnosis). The project's bottleneck is not the science -- it is that
judgement errors and skipped checks eat most of the working time. The checks already existed: before_you_build.sh,
research_gate.sh, tools/lab.py, lane_check.py, workflow_check.sh, docs/TERMS.md, verify-go. They are OPTIONAL, and
optional checks are skipped exactly when momentum makes skipping attractive -- which is exactly when they matter.

The evidence for that is the whole record, but the sharpest single fact is this: on 2026-07-31 the ONLY check that
stopped a mistake without being invited was the pre-commit hook, which blocked a commit for over-long doc lines.
Every other check that ran that day ran because it was remembered. Every failure that day was a check not
remembered:

  * 94 GPU-hours re-deriving a NO-GO banked a week earlier   -> before_you_build.sh not run
  * a control that agreed with its treatment to 1e-9         -> no instrument validation
  * a metric a position-shuffle reproduced to 1.3%           -> no negative control on the metric
  * the FIFTH instance of the plasticity bound trap          -> pre-flight existed as PROSE for four other rules
  * an A/B whose arms differed in 3 variables, not 1         -> no one-variable assertion
  * 6 runs staged on a wrong provenance assumption           -> config never recorded, filename used as provenance

THE DESIGN RULE, and the only one that matters: this class FAILS CLOSED. `verdict()` raises unless the
experiment was pre-registered, the corpus was checked, and the instrument was validated in BOTH directions. It is
not a linter you run at the end. It is the only door to a reportable result.

    from tools.experiment import Experiment

    exp = Experiment(
        name="gap5-laps-isolation",
        lane="H · Memory",
        question="Is laps=1 (a single induction pass) the operative variable for place-specificity?",
        hypothesis="laps=1 gives shuffle-ratio > 2.0; laps=5 stays ~1.0",
        gate="permutation p < 0.05 AND ratio > 2.0, 6 seeds",
        kill="if laps=1 ALSO gives ratio ~1.0, the induction-event framing is REFUTED -- record it, do not retune",
        one_variable="laps",
        arms={"L1": dict(laps=1, dwell=30, w_max=2500), "L5": dict(laps=5, dwell=30, w_max=2500)},
    )
    exp.check_bounds(w_max=2500, weight=250)              # the bound trap, executable
    exp.validate_instrument(metric, positive=..., negative=...)   # REQUIRED, both directions
    exp.verdict(observed=..., passed=True)                # raises if any gate above was skipped
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import socket
import string
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tools.lab import bound_check, sign_budget, void_if, LeverError  # noqa: E402

# Re-exported so a caller needs ONE import to get the whole discipline, not four.
__all__ = [
    "Experiment",
    "HarnessError",
    "bound_check",
    "sign_budget",
    "void_if",
    "create_experiment_seal",
    "expand_experiment_jobs",
    "load_experiment_spec",
    "write_experiment_plan",
]


class HarnessError(LeverError):
    """Raised when a gate is skipped. Never catch this to 'keep going' -- that IS the failure mode.

    Subclasses LeverError (itself an AssertionError) so that ONE except-clause covers every guard in the
    harness AND in tools.lab. Two exception types would mean a caller could catch one and silently proceed
    past the other, which is the precise shape of failure this module exists to prevent.
    """


_PLAN_SCHEMA = "sim-experiment-plan-v1"
_SEAL_SCHEMA = "sim-experiment-seal-v1"
_MANIFEST_SCHEMA = "sim-execution-manifest-v1"
_CORPUS_SCHEMA = "sim-corpus-check-v1"
_CONTRACT_SCHEMA = "sim-experiment-job-contract-v1"
_ALLOWED_PLACEHOLDERS = frozenset(("arm", "backend", "device", "output", "partition", "seed", "spec"))
_CODE_ROOTS = ("sim", "research/runners", "experiment", "tools")
_RUNTIME_LOGS = frozenset((
    "research/findings/raw/_provenance/runs.jsonl",
    "research/queue/.corpus_checks.jsonl",
))
_DEPENDENCY_NAMES = ("requirements.txt", "requirements-dev.txt", "pyproject.toml", "setup.cfg",
                     "setup.py", "poetry.lock", "uv.lock")


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _required_text(mapping, key, context):
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"{context} requires a non-empty {key!r}")
    return value.strip()


def _heldout_partition(name):
    return "".join(ch for ch in name.lower() if ch.isalnum()).startswith("heldout")


def load_experiment_spec(path):
    """Load the shared JSON preregistration format and reject ambiguous job partitions."""
    path = Path(path)
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot load experiment spec {path}: {exc}") from None
    if not isinstance(spec, dict):
        raise HarnessError("experiment spec must be a JSON object")
    schema = _required_text(spec, "schema", "experiment spec")
    if not schema.startswith("sim-experiment-spec-"):
        raise HarnessError(f"unsupported experiment spec schema {schema!r}")
    _required_text(spec, "id", "experiment spec")

    partitions = spec.get("partitions")
    if not isinstance(partitions, dict) or not partitions:
        raise HarnessError("experiment spec requires a non-empty 'partitions' object")
    if "calibration" not in partitions:
        raise HarnessError("experiment spec must name an explicit 'calibration' seed partition")
    if not any(_heldout_partition(name) for name in partitions):
        raise HarnessError("experiment spec must name an explicit held-out seed partition")
    owners = {}
    for name, seeds in partitions.items():
        if not isinstance(name, str) or not name.strip():
            raise HarnessError("partition names must be non-empty strings")
        if not isinstance(seeds, list) or not seeds:
            raise HarnessError(f"partition {name!r} must contain at least one seed")
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise HarnessError(f"partition {name!r} contains invalid seed {seed!r}")
            if seed in owners:
                raise HarnessError(
                    f"seed {seed} overlaps partitions {owners[seed]!r} and {name!r}; partitions must be disjoint"
                )
            owners[seed] = name

    backends = spec.get("backends")
    if (not isinstance(backends, list) or not backends
            or any(not isinstance(item, str) or not item.strip() for item in backends)):
        raise HarnessError("experiment spec requires a non-empty string list 'backends'")
    if len(set(backends)) != len(backends):
        raise HarnessError("experiment spec backends must be unique")
    return spec


def _read_source_revision(root):
    path = root / ".source_revision"
    if not path.is_file():
        return None
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key:
            values[key] = value
    if values.get("source_kind") != "git_archive" or not values.get("git_sha"):
        raise HarnessError(".source_revision is incomplete or is not a git_archive identity")
    manifest = root / ".source_manifest.sha256"
    expected = values.get("source_manifest_sha256")
    if not expected or not manifest.is_file() or _sha256_file(manifest) != expected:
        raise HarnessError("exported source manifest does not match .source_revision")
    declared = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        path = PurePosixPath(relative)
        if (not separator or len(digest) != 64 or path.is_absolute()
                or ".." in path.parts or relative in declared):
            raise HarnessError("exported source manifest contains an invalid or duplicate entry")
        try:
            int(digest, 16)
        except ValueError:
            raise HarnessError("exported source manifest contains an invalid digest") from None
        declared[relative] = digest
    actual = set()
    for relative_root in ("sim", "research/runners", "experiment", "tools"):
        source_root = root / relative_root
        if not source_root.is_dir():
            continue
        for source_path in source_root.rglob("*"):
            if source_path.is_file() and "__pycache__" not in source_path.parts \
                    and source_path.suffix in (".py", ".sh"):
                actual.add(source_path.relative_to(root).as_posix())
    research_init = root / "research/__init__.py"
    if research_init.is_file():
        actual.add("research/__init__.py")
    if set(declared) != actual:
        missing = sorted(set(declared) - actual)[:3]
        extra = sorted(actual - set(declared))[:3]
        raise HarnessError(f"exported source file set differs from manifest; missing={missing}, extra={extra}")
    for relative, digest in declared.items():
        source_path = root / relative
        if not source_path.is_file() or _sha256_file(source_path) != digest:
            raise HarnessError(f"exported source digest mismatch: {relative}")
    return {"kind": "git_archive", "revision": values["git_sha"],
            "source_manifest_sha256": expected}


def _source_identity(root):
    root = Path(root).resolve()
    archive = _read_source_revision(root)
    if archive is not None:
        return archive
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise HarnessError(f"cannot identify experiment source: {exc}") from None
    return {"kind": "git", "revision": revision}


def _repository_path(relative, root, context):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.name:
        raise HarnessError(f"{context} must be a safe repository-relative path, got {relative!r}")
    absolute = Path(root, *path.parts).resolve()
    if os.path.commonpath((str(Path(root).resolve()), str(absolute))) != str(Path(root).resolve()):
        raise HarnessError(f"{context} escapes repository root: {relative!r}")
    return absolute


def _git_paths(root, args):
    try:
        raw = subprocess.run(["git", *args, "-z"], cwd=root, check=True, capture_output=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise HarnessError(f"cannot inspect Git worktree: {exc}") from None
    return {item.decode("utf-8", "surrogateescape") for item in raw.split(b"\0") if item}


def _git_dirty_paths(root):
    if _read_source_revision(root) is not None:
        return set(), set()
    changed = _git_paths(root, ["diff", "--name-only"]) | _git_paths(root, ["diff", "--cached", "--name-only"])
    untracked = _git_paths(root, ["ls-files", "--others", "--exclude-standard"])
    return changed | untracked, untracked


def _code_file_records(root):
    records = []
    paths = set()
    for relative_root in _CODE_ROOTS:
        source_root = root / relative_root
        if not source_root.is_dir():
            continue
        for path in source_root.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".sh"):
                paths.add(path.relative_to(root).as_posix())
    if (root / "research/__init__.py").is_file():
        paths.add("research/__init__.py")
    for relative in sorted(paths):
        records.append({"path": relative, "sha256": _sha256_file(root / relative)})
    return records


def _execution_contract(spec):
    execution = spec.get("execution")
    if not isinstance(execution, dict):
        raise HarnessError("spec is valid for preregistration but lacks an 'execution' contract")
    command = execution.get("command")
    if (not isinstance(command, list) or not command
            or any(not isinstance(token, str) or not token for token in command)):
        raise HarnessError("execution.command must be a non-empty JSON string list")
    if not any(token.endswith(".venv/bin/python") or token == ".venv/bin/python" for token in command):
        raise HarnessError("execution.command must use the sanctioned .venv/bin/python interpreter")
    try:
        module = command[command.index("-m") + 1]
    except (ValueError, IndexError):
        raise HarnessError("execution.command must invoke a runner with '-m research.runners.<name>'") from None
    if not module.startswith("research.runners."):
        raise HarnessError("execution.command must invoke research.runners so automatic provenance remains active")

    output = _required_text(execution, "output", "execution contract")
    fields = set()
    for value in list(command) + [output] + list(execution.get("runtime_outputs", [])):
        if not isinstance(value, str) or not value:
            raise HarnessError("execution templates must be non-empty strings")
        try:
            fields.update(field for _, field, _, _ in string.Formatter().parse(value) if field)
        except ValueError as exc:
            raise HarnessError(f"invalid execution template {value!r}: {exc}") from None
    unknown = fields - _ALLOWED_PLACEHOLDERS
    if unknown:
        raise HarnessError(f"execution templates use unsupported placeholders: {sorted(unknown)}")
    if not {"seed", "partition", "output"}.issubset(fields):
        raise HarnessError("execution templates must expose placeholders ['output', 'partition', 'seed']")

    raw_arms = execution.get("arms", spec.get("arms", ["default"]))
    arms = sorted(raw_arms) if isinstance(raw_arms, (dict, list)) else None
    if not arms or any(not isinstance(arm, str) or not arm.strip() for arm in arms):
        raise HarnessError("execution requires at least one non-empty arm name")
    if len(arms) > 1 and "arm" not in fields:
        raise HarnessError("multi-arm execution must expose an {arm} placeholder")

    targets = execution.get("targets")
    if not isinstance(targets, dict):
        raise HarnessError("execution.targets must declare a device and lane for every backend")
    normalized_targets = {}
    for backend in spec["backends"]:
        target = targets.get(backend)
        if not isinstance(target, dict):
            raise HarnessError(f"backend {backend!r} has no execution target/device declaration")
        device = _required_text(target, "device", f"backend {backend!r}")
        lane = _required_text(target, "lane", f"backend {backend!r}")
        if lane not in ("local", "gpu", "pool"):
            raise HarnessError(f"backend {backend!r} has unsupported lane {lane!r}")
        env = target.get("env", {})
        if (not isinstance(env, dict)
                or any(not isinstance(key, str) or not isinstance(value, str) for key, value in env.items())):
            raise HarnessError(f"backend {backend!r} target env must map strings to strings")
        if "SIM_BACKEND" in env and env["SIM_BACKEND"] != backend:
            raise HarnessError(f"backend {backend!r} target contradicts SIM_BACKEND={env['SIM_BACKEND']!r}")
        normalized_targets[backend] = {"device": device, "lane": lane, "env": dict(env)}
    corpus = execution.get("corpus_check")
    if not isinstance(corpus, dict):
        raise HarnessError("execution requires a machine-readable 'corpus_check' contract")
    stale = execution.get("claim_stale_seconds", 6 * 3600)
    if isinstance(stale, bool) or not isinstance(stale, int) or stale < 60:
        raise HarnessError("execution.claim_stale_seconds must be an integer >= 60")
    return command, output, arms, normalized_targets, corpus, stale


def _file_declarations(execution, key):
    values = execution.get(key, [])
    if not isinstance(values, list):
        raise HarnessError(f"execution.{key} must be a JSON list")
    declarations = []
    for item in values:
        if isinstance(item, str):
            declarations.append({"path": item})
        elif isinstance(item, dict):
            declarations.append(dict(item))
        else:
            raise HarnessError(f"execution.{key} entries must be paths or objects")
    return declarations


def _validated_corpus_check(spec, root, contract):
    relative = _required_text(contract, "path", "corpus check")
    expected = _required_text(contract, "sha256", "corpus check")
    query = _required_text(contract, "query", "corpus check")
    max_age = contract.get("max_age_seconds", 24 * 3600)
    if len(expected) != 64:
        raise HarnessError("corpus check sha256 must contain 64 hexadecimal characters")
    if isinstance(max_age, bool) or not isinstance(max_age, int) or max_age <= 0:
        raise HarnessError("corpus check max_age_seconds must be a positive integer")
    path = _repository_path(relative, root, "corpus check")
    if not path.is_file() or _sha256_file(path) != expected:
        raise HarnessError("corpus/RAG check is missing or its digest does not match the execution contract")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"corpus/RAG check is not valid machine-readable JSON: {exc}") from None
    rag = record.get("rag")
    if (record.get("schema") != _CORPUS_SCHEMA or record.get("status") != "success"
            or record.get("experiment_id") != spec["id"] or record.get("query") != query
            or not isinstance(rag, dict) or rag.get("status") != "success"
            or not isinstance(rag.get("index_digest"), str) or not rag["index_digest"]):
        raise HarnessError("corpus/RAG check does not record a successful matching retrieval")
    completed = record.get("completed_at")
    if isinstance(completed, bool) or not isinstance(completed, (int, float)):
        raise HarnessError("corpus/RAG check completed_at must be an epoch timestamp")
    age = time.time() - float(completed)
    if age < -300 or age > max_age:
        raise HarnessError(f"corpus/RAG check is stale or future-dated (age={age:.0f}s, max={max_age}s)")
    return {"path": relative, "sha256": expected, "query": query, "completed_at": completed,
            "rag_index_digest": rag["index_digest"], "max_age_seconds": max_age}


def _input_records(spec_path, spec, root, corpus):
    execution = spec["execution"]
    records = {}
    spec_relative = spec_path.relative_to(root).as_posix()
    declarations = [({"path": spec_relative}, "spec"), ({"path": corpus["path"]}, "corpus_check")]
    for name in _DEPENDENCY_NAMES:
        if (root / name).is_file():
            declarations.append(({"path": name}, "dependency"))
    declarations.extend((item, "dependency") for item in _file_declarations(execution, "dependencies"))
    declarations.extend((item, "input") for item in _file_declarations(execution, "inputs"))
    for rule in spec.get("prerequisites", []):
        if isinstance(rule, dict) and isinstance(rule.get("path"), str):
            declarations.append(({"path": rule["path"], "sha256": rule.get("sha256")}, "prerequisite"))
    for rule in spec.get("stop_rules", []):
        if isinstance(rule, dict) and isinstance(rule.get("decision_file"), str):
            decision = _repository_path(rule["decision_file"], root, "stop-rule decision")
            if decision.is_file():
                declarations.append(({"path": rule["decision_file"]}, "decision"))
    for declaration, role in declarations:
        relative = _required_text(declaration, "path", f"execution {role}")
        path = _repository_path(relative, root, f"execution {role}")
        if not path.is_file():
            raise HarnessError(f"declared execution {role} is missing: {relative}")
        digest = _sha256_file(path)
        supplied = declaration.get("sha256")
        if supplied is not None and supplied != digest:
            raise HarnessError(f"declared execution {role} has the wrong sha256: {relative}")
        prior = records.get(relative)
        if prior and prior["sha256"] != digest:
            raise HarnessError(f"execution input {relative!r} has conflicting digests")
        records.setdefault(relative, {"path": relative, "sha256": digest, "roles": []})["roles"].append(role)
    return [{**record, "roles": sorted(set(record["roles"]))} for _, record in sorted(records.items())]


def _portable_output_path(template, values, root):
    rendered = template.format_map(values)
    return PurePosixPath(rendered).as_posix(), _repository_path(rendered, root, "output")


def _runtime_paths(spec, root, output_template, arms, targets):
    paths = set(_RUNTIME_LOGS)
    extra_templates = spec["execution"].get("runtime_outputs", [])
    if not isinstance(extra_templates, list):
        raise HarnessError("execution.runtime_outputs must be a JSON list")
    spec_relative = Path(spec["_spec_path"]).relative_to(root).as_posix()
    for partition, seeds in spec["partitions"].items():
        for backend in sorted(spec["backends"]):
            for arm in arms:
                for seed in sorted(seeds):
                    values = {"arm": arm, "backend": backend, "device": targets[backend]["device"],
                              "partition": partition, "seed": seed, "spec": spec_relative, "output": ""}
                    output, _ = _portable_output_path(output_template, values, root)
                    values["output"] = output
                    paths.update((output, output + ".prov.json", output + ".claim"))
                    for template in extra_templates:
                        rendered = template.format_map(values)
                        _repository_path(rendered, root, "runtime output")
                        paths.add(PurePosixPath(rendered).as_posix())
    return sorted(paths)


def _manifest_digest(manifest):
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    return _sha256_bytes(_canonical_json(body))


def _build_execution_manifest(spec_path, spec, root):
    root = Path(root).resolve()
    spec_path = Path(spec_path).resolve()
    spec = dict(spec)
    spec["_spec_path"] = str(spec_path)
    command, output, arms, targets, corpus_contract, _ = _execution_contract(spec)
    corpus = _validated_corpus_check(spec, root, corpus_contract)
    code_files = _code_file_records(root)
    manifest = {
        "schema": _MANIFEST_SCHEMA,
        "experiment_id": spec["id"],
        "source": _source_identity(root),
        "code_files": code_files,
        "code_sha256": _sha256_bytes(_canonical_json(code_files)),
        "inputs": _input_records(spec_path, spec, root, corpus),
        "runtime_paths": _runtime_paths(spec, root, output, arms, targets),
        "corpus_check": corpus,
        "command_template": command,
    }
    manifest["sha256"] = _manifest_digest(manifest)
    return manifest


def _validate_worktree_dirt(root, manifest):
    dirty, untracked = _git_dirty_paths(root)
    trusted_untracked_inputs = {item["path"] for item in manifest["inputs"]} & untracked
    allowed = set(manifest["runtime_paths"]) | trusted_untracked_inputs
    unexpected = sorted(dirty - allowed)
    if unexpected:
        raise HarnessError(f"execution source/config has unsealed worktree changes: {unexpected[:5]}")


def _write_new_readonly_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as fh:
            fh.write(data)
    except FileExistsError:
        raise HarnessError(f"refusing to replace immutable file {path}") from None
    path.chmod(0o444)


def create_experiment_seal(spec_path, seal_path, root=ROOT):
    """Seal exact executable code, configuration, dependencies, inputs, and allowed outputs."""
    root = Path(root).resolve()
    spec_path = Path(spec_path).resolve()
    spec = load_experiment_spec(spec_path)
    manifest = _build_execution_manifest(spec_path, spec, root)
    _validate_worktree_dirt(root, manifest)
    seal = {"schema": _SEAL_SCHEMA, "experiment_id": spec["id"],
            "spec_sha256": _sha256_bytes(_canonical_json(spec)), "manifest": manifest}
    _write_new_readonly_json(seal_path, seal)
    return seal


def _validated_seal(spec_path, spec, seal_path, root):
    try:
        seal = json.loads(Path(seal_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot load experiment seal {seal_path}: {exc}") from None
    if seal.get("schema") != _SEAL_SCHEMA or seal.get("experiment_id") != spec["id"]:
        raise HarnessError("experiment seal has the wrong schema or experiment id")
    if seal.get("spec_sha256") != _sha256_bytes(_canonical_json(spec)):
        raise HarnessError("experiment spec changed after sealing")
    current = _build_execution_manifest(spec_path, spec, root)
    _validate_worktree_dirt(root, current)
    sealed = seal.get("manifest")
    if not isinstance(sealed, dict) or sealed.get("sha256") != _manifest_digest(sealed):
        raise HarnessError("experiment seal contains an invalid execution manifest digest")
    if current != sealed:
        raise HarnessError("sealed execution source, config, dependency, or input changed")
    return seal

def _check_prerequisites(spec, selected, root):
    rules = spec.get("prerequisites", [])
    if not isinstance(rules, list):
        raise HarnessError("prerequisites must be a JSON list")
    for rule in rules:
        if not isinstance(rule, dict):
            raise HarnessError("each prerequisite must be a JSON object")
        applies = rule.get("partitions", list(selected))
        if not isinstance(applies, list) or not set(applies).intersection(selected):
            continue
        rule_id = _required_text(rule, "id", "prerequisite")
        relative = _required_text(rule, "path", f"prerequisite {rule_id!r}")
        path = _repository_path(relative, root, f"prerequisite {rule_id!r}")
        if not path.is_file():
            raise HarnessError(f"prerequisite {rule_id!r} is missing: {relative}")
        expected = _required_text(rule, "sha256", f"prerequisite {rule_id!r}")
        if len(expected) != 64 or _sha256_file(path) != expected:
            raise HarnessError(f"prerequisite {rule_id!r} has the wrong sha256: {relative}")


def _check_stop_rules(spec, selected, root, spec_sha256):
    rules = spec.get("stop_rules", [])
    if not isinstance(rules, list):
        raise HarnessError("stop_rules must be a JSON list")
    for rule in rules:
        if not isinstance(rule, dict):
            raise HarnessError("each stop rule must be a JSON object")
        blocks = rule.get("blocks", [])
        if not isinstance(blocks, list) or not set(blocks).intersection(selected):
            continue
        rule_id = _required_text(rule, "id", "stop rule")
        relative = _required_text(rule, "decision_file", f"stop rule {rule_id!r}")
        decision_path = _repository_path(relative, root, f"stop rule {rule_id!r}")
        try:
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(f"stop rule {rule_id!r} has no valid decision record: {exc}") from None
        if decision.get("rule_id") != rule_id or decision.get("spec_sha256") != spec_sha256:
            raise HarnessError(f"stop rule {rule_id!r} decision does not belong to this sealed config")
        value = decision.get("decision")
        if value == "stop":
            raise HarnessError(f"stop rule {rule_id!r} recorded STOP; downstream jobs are blocked")
        if value != "continue":
            raise HarnessError(f"stop rule {rule_id!r} requires an explicit 'continue' or 'stop' decision")


def _runtime_snapshot(manifest):
    return {
        "manifest_sha256": manifest["sha256"],
        "source": manifest["source"],
        "code_sha256": manifest["code_sha256"],
        "code_file_count": len(manifest["code_files"]),
        "inputs": manifest["inputs"],
        "runtime_paths_sha256": _sha256_bytes(_canonical_json(manifest["runtime_paths"])),
        "runtime_path_count": len(manifest["runtime_paths"]),
        "corpus_check": manifest["corpus_check"],
    }


def _verify_runtime_snapshot(contract, root):
    root = Path(root).resolve()
    if contract.get("schema") != _CONTRACT_SCHEMA:
        raise HarnessError("job has an unsupported execution-contract schema")
    snapshot = contract.get("execution_snapshot")
    if not isinstance(snapshot, dict):
        raise HarnessError("job is missing its sealed execution snapshot")
    actual_source = _source_identity(root)
    expected_source = snapshot.get("source", {})
    if actual_source.get("revision") != expected_source.get("revision"):
        raise HarnessError(
            f"execution source revision mismatch: expected {expected_source.get('revision')}, "
            f"found {actual_source.get('revision')}"
        )
    code_files = _code_file_records(root)
    if (len(code_files) != snapshot.get("code_file_count")
            or _sha256_bytes(_canonical_json(code_files)) != snapshot.get("code_sha256")):
        raise HarnessError("execution code manifest differs from the sealed source")
    inputs = snapshot.get("inputs")
    if not isinstance(inputs, list):
        raise HarnessError("execution snapshot has no input manifest")
    spec = None
    spec_path = None
    for item in inputs:
        if not isinstance(item, dict) or not isinstance(item.get("roles"), list):
            raise HarnessError("execution snapshot contains an invalid input record")
        path = _repository_path(item.get("path"), root, "sealed execution input")
        if not path.is_file() or _sha256_file(path) != item.get("sha256"):
            raise HarnessError(f"sealed execution input changed or is missing: {item.get('path')}")
        if "spec" in item["roles"]:
            spec = load_experiment_spec(path)
            spec_path = path
    if spec is None or spec.get("id") != contract.get("experiment_id"):
        raise HarnessError("sealed experiment specification is missing or belongs to another experiment")
    _validated_corpus_check(spec, root, snapshot.get("corpus_check", {}))
    spec_with_path = dict(spec)
    spec_with_path["_spec_path"] = str(spec_path)
    _, output_template, arms, targets, _, _ = _execution_contract(spec_with_path)
    runtime_paths = _runtime_paths(spec_with_path, root, output_template, arms, targets)
    if (len(runtime_paths) != snapshot.get("runtime_path_count")
            or _sha256_bytes(_canonical_json(runtime_paths)) != snapshot.get("runtime_paths_sha256")):
        raise HarnessError("allowed runtime-output manifest differs from the sealed contract")
    pseudo_manifest = {"inputs": inputs, "runtime_paths": runtime_paths}
    _validate_worktree_dirt(root, pseudo_manifest)
    if contract.get("output") not in set(runtime_paths):
        raise HarnessError("job output is not an allowed path in the sealed execution manifest")
    return snapshot


def _encoded_contract(contract):
    return base64.urlsafe_b64encode(_canonical_json(contract)).decode("ascii")


def _decoded_contract(value):
    try:
        contract = json.loads(base64.urlsafe_b64decode(value.encode("ascii")))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise HarnessError(f"cannot decode execution contract: {exc}") from None
    if not isinstance(contract, dict):
        raise HarnessError("execution contract must decode to a JSON object")
    return contract


def _pid_is_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except PermissionError:
        return True
    except (ProcessLookupError, TypeError, ValueError):
        return False


def _claim_is_stale(path, stale_seconds):
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        started = float(record["started_at"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        try:
            started = path.stat().st_mtime
        except OSError:
            return True
        record = {}
    age = max(0.0, time.time() - started)
    if record.get("hostname") == socket.gethostname() and _pid_is_alive(record.get("pid")):
        return False
    return age > stale_seconds


def _acquire_claim(path, job_id, stale_seconds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not _claim_is_stale(path, stale_seconds):
            raise HarnessError(f"output claim is active: {path}")
        try:
            path.unlink()
        except OSError as exc:
            raise HarnessError(f"cannot clear stale output claim {path}: {exc}") from None
    record = {"schema": "sim-experiment-claim-v1", "job_id": job_id, "pid": os.getpid(),
              "hostname": socket.gethostname(), "started_at": time.time()}
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        raise HarnessError(f"output claim raced with another worker: {path}") from None
    with os.fdopen(descriptor, "w", encoding="utf-8") as fh:
        json.dump(record, fh, sort_keys=True)
        fh.write("\n")


def execute_job_contract(contract, command, root=ROOT):
    """Verify sealed state, own the output, run once, and release failed/stale claims."""
    root = Path(root).resolve()
    command = list(command)
    if command != contract.get("runner_command"):
        raise HarnessError("runner command differs from the digest-bound job contract")
    _verify_runtime_snapshot(contract, root)
    output = _repository_path(contract.get("output"), root, "job output")
    claim = Path(str(output) + ".claim")
    if output.exists():
        raise HarnessError(f"successful output already exists and is immutable: {contract.get('output')}")
    _acquire_claim(claim, contract.get("job_id"), contract.get("claim_stale_seconds", 6 * 3600))
    env = os.environ.copy()
    env.update(contract.get("environment", {}))
    try:
        result = subprocess.run(command, cwd=root, env=env, check=False)
        if result.returncode:
            raise HarnessError(f"experiment runner failed with exit code {result.returncode}")
        if not output.is_file():
            raise HarnessError("experiment runner exited successfully without creating its declared output")
    finally:
        try:
            claim.unlink()
        except FileNotFoundError:
            pass
    return {"job_id": contract.get("job_id"), "output": contract.get("output"), "status": "complete"}


def _job_command(contract, tokens):
    encoded = _encoded_contract(contract)
    wrapper = [".venv/bin/python", "tools/experiment.py", "execute-job", "--contract", encoded, "--", *tokens]
    return shlex.join(wrapper)


def expand_experiment_jobs(spec_path, partitions, seal_path=None, root=ROOT):
    """Expand a sealed preregistration into a deterministic backend/seed/arm job matrix."""
    root = Path(root).resolve()
    spec_path = Path(spec_path).resolve()
    try:
        spec_relative = spec_path.relative_to(root).as_posix()
    except ValueError:
        raise HarnessError("experiment spec must live inside the repository for portable dispatch") from None
    spec = load_experiment_spec(spec_path)
    if not partitions:
        raise HarnessError("choose at least one partition explicitly; held-out is never an implicit default")
    selected = sorted(set(partitions))
    unknown = set(selected) - set(spec["partitions"])
    if unknown:
        raise HarnessError(f"unknown experiment partitions: {sorted(unknown)}")
    if any(_heldout_partition(name) for name in selected) and seal_path is None:
        raise HarnessError("held-out jobs are locked until a clean source/config seal is supplied")
    seal = _validated_seal(spec_path, spec, seal_path, root) if seal_path is not None else None
    manifest = seal["manifest"] if seal else _build_execution_manifest(spec_path, spec, root)
    _validate_worktree_dirt(root, manifest)
    spec_sha256 = _sha256_bytes(_canonical_json(spec))
    _check_prerequisites(spec, selected, root)
    _check_stop_rules(spec, selected, root, spec_sha256)
    command_template, output_template, arms, targets, _, claim_stale = _execution_contract(spec)
    snapshot = _runtime_snapshot(manifest)
    reason = "corpus-check:" + manifest["corpus_check"]["sha256"][:16]

    jobs = []
    outputs = set()
    for partition in selected:
        for backend in sorted(spec["backends"]):
            target = targets[backend]
            for arm in arms:
                for seed in sorted(spec["partitions"][partition]):
                    values = {
                        "arm": arm,
                        "backend": backend,
                        "device": target["device"],
                        "partition": partition,
                        "seed": seed,
                        "spec": spec_relative,
                    }
                    output, absolute_output = _portable_output_path(output_template, {**values, "output": ""}, root)
                    values["output"] = output
                    if output in outputs:
                        raise HarnessError(f"job matrix maps more than one job to output {output!r}")
                    outputs.add(output)
                    claim_path = Path(str(absolute_output) + ".claim")
                    if claim_path.exists() and _claim_is_stale(claim_path, claim_stale):
                        claim_path.unlink()
                    if absolute_output.exists() or claim_path.exists():
                        raise HarnessError(f"refusing mutable output collision at {output!r} (or its claim)")
                    tokens = [token.format_map(values) for token in command_template]
                    identity = {
                        "experiment_id": spec["id"], "spec_sha256": spec_sha256,
                        "execution_manifest_sha256": manifest["sha256"],
                        "corpus_check_sha256": manifest["corpus_check"]["sha256"],
                        "source": manifest["source"], "partition": partition, "backend": backend,
                        "device": target["device"], "arm": arm, "seed": seed, "output": output,
                    }
                    job_id = _sha256_bytes(_canonical_json(identity))[:20]
                    environment = {"SIM_BACKEND": backend, **target["env"]}
                    contract = {"schema": _CONTRACT_SCHEMA, "job_id": job_id, **identity,
                                "execution_snapshot": snapshot, "runner_command": tokens,
                                "environment": environment, "claim_stale_seconds": claim_stale}
                    command = _job_command(contract, tokens)
                    enqueue = None
                    if target["lane"] in ("gpu", "pool"):
                        enqueue = shlex.join(["bash", "tools/queue_add.sh", target["lane"], command, reason])
                    jobs.append({
                        "schema": _PLAN_SCHEMA,
                        "job_id": job_id,
                        **identity,
                        "sealed": seal is not None,
                        "lane": target["lane"],
                        "command": command,
                        "enqueue_command": enqueue,
                        "output_claim": output + ".claim",
                        "execution_contract": contract,
                    })
    return jobs


def write_experiment_plan(jobs, plan_dir):
    """Write a fresh, read-only command and JSON manifest pair for every planned job."""
    jobs = list(jobs)
    if not jobs:
        raise HarnessError("cannot write an empty experiment plan")
    plan_dir = Path(plan_dir)
    try:
        plan_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise HarnessError(f"refusing to mutate existing plan directory {plan_dir}") from None
    for job in jobs:
        stem = f"{job['partition']}-{job['backend']}-{job['arm']}-{job['seed']}-{job['job_id']}"
        _write_new_readonly_json(plan_dir / f"{stem}.json", job)
        command = job["enqueue_command"] or job["command"]
        command_path = plan_dir / f"{stem}.command"
        with command_path.open("x", encoding="utf-8") as fh:
            fh.write(command + "\n")
        command_path.chmod(0o444)
    index = {
        "schema": _PLAN_SCHEMA,
        "experiment_id": jobs[0]["experiment_id"],
        "job_ids": [job["job_id"] for job in jobs],
        "count": len(jobs),
    }
    _write_new_readonly_json(plan_dir / "plan.json", index)
    plan_dir.chmod(0o555)
    return index


class Experiment:
    # ---------------------------------------------------------------- registration
    def __init__(self, name, lane, question, hypothesis, gate, kill, one_variable,
                 arms=None, supersedes=None, corpus_check=True):
        """Pre-registration is the CONSTRUCTOR, so an experiment cannot exist without it.

        `kill` is not optional and not decorative: a pre-registered prediction protects against retrofitting a
        conclusion, and a kill criterion protects against retuning past a refutation. Both were skipped on the
        levers that cost the most.

        `one_variable` names the SINGLE thing that differs between arms. "ONE FLAG != ONE VARIABLE" is its own
        recurring failure: `--bdsp-wmax` was one config field but two functional variables, and
        `hebbian_mean_subtract` changed the fixed point AND the weight mass AND the firing rate, which turned a
        clean-looking refutation into a confound.
        """
        for field, val in (("name", name), ("lane", lane), ("question", question),
                           ("hypothesis", hypothesis), ("gate", gate), ("kill", kill),
                           ("one_variable", one_variable)):
            if not val or not str(val).strip():
                raise HarnessError(
                    "PRE-REGISTRATION INCOMPLETE: '%s' is required. Every field here exists because omitting it "
                    "cost this project a retraction or a day of compute. If you cannot state the kill criterion, "
                    "you do not yet know what would refute you, and the run cannot be interpreted either way."
                    % field)
        self.name = name
        self.lane = lane
        self.question = question
        self.hypothesis = hypothesis
        self.gate = gate
        self.kill = kill
        self.one_variable = one_variable
        self.arms = dict(arms or {})
        self.supersedes = supersedes
        self._registered_at = time.time()
        self._instrument_ok = False
        self._instrument_report = None
        self._bounds_checked = []
        self._corpus_hits = []

        print("=" * 78)
        print("EXPERIMENT  %s   [lane %s]" % (self.name, self.lane))
        print("=" * 78)
        print("  Q      : %s" % self.question)
        print("  H      : %s" % self.hypothesis)
        print("  GATE   : %s" % self.gate)
        print("  KILL   : %s" % self.kill)
        print("  VAR    : %s" % self.one_variable)
        if self.arms:
            self._assert_one_variable()
        if corpus_check:
            self._corpus_check()

    def _assert_one_variable(self):
        """Arms must differ in exactly the declared variable. Anything else is a confound, by construction."""
        keys = set()
        for cfg in self.arms.values():
            keys |= set(cfg.keys())
        differing = []
        for k in sorted(keys):
            vals = {json.dumps(cfg.get(k), sort_keys=True, default=str) for cfg in self.arms.values()}
            if len(vals) > 1:
                differing.append(k)
        if not differing:
            raise HarnessError(
                "ARMS ARE IDENTICAL: no config key differs across %s. The A/B is void and would have produced "
                "two identical numbers that look like a result." % list(self.arms))
        extra = [k for k in differing if k != self.one_variable]
        if extra:
            raise HarnessError(
                "CONFOUNDED ARMS: you declared one_variable=%r but the arms also differ in %s. Either hold those "
                "fixed, or declare the comparison honestly as multi-variable and expect it to be uninterpretable. "
                "(Earned: a mean-subtract A/B that also changed weight mass and firing rate, so 'the rule is "
                "worse' was inseparable from '3x less weight and 4x lower rate'.)" % (self.one_variable, extra))
        print("  arms   : %s differ ONLY in %r ✔" % (list(self.arms), self.one_variable))

    def _corpus_check(self):
        """Ask the record BEFORE spending compute. Refuses on a strong hit unless explicitly superseded."""
        try:
            out = subprocess.run(
                [os.path.join(ROOT, ".venv-rag/bin/python"), "tools/rag/rag_search.py",
                 self.question, "5", "--corpus", "finding"],
                cwd=ROOT, capture_output=True, text=True, timeout=180).stdout
        except Exception as e:                                  # narrow enough to see; never silent
            print("  corpus : ⚠️  check FAILED (%s: %s) — treat this as UNCHECKED, not clean"
                  % (type(e).__name__, e))
            return
        hits = []
        for ln in out.split("\n"):
            s = ln.strip()
            if s.startswith("[") and "(finding)" in s:
                try:
                    score = float(s.split("]")[1].split()[0])
                except Exception:
                    continue
                hits.append((score, s.split("(finding)")[-1].strip()))
        self._corpus_hits = hits[:5]
        strong = [h for h in hits if h[0] > 3.0]
        for sc, path in hits[:3]:
            print("  corpus : %+.2f  %s" % (sc, path[:88]))
        if strong and not self.supersedes:
            raise HarnessError(
                "THE RECORD MAY ALREADY ANSWER THIS. Strong prior finding(s):\n    %s\n"
                "READ them first. If they genuinely do not cover this experiment, re-register with "
                "supersedes='<why this is different>'. (Earned: 94 GPU-hours spent re-deriving a NO-GO that had "
                "been banked a week earlier, in a config that also reversed the re-scope made to afford it.)"
                % "\n    ".join(p for _, p in strong[:3]))
        if self.supersedes:
            print("  corpus : superseding prior work — %s" % self.supersedes)

    # ---------------------------------------------------------------- gates
    def check_bounds(self, **pairs):
        """check_bounds(btsp_w_max=(150, 250)) or check_bounds(w_max=150, weight=250)."""
        if "weight" in pairs and len(pairs) == 2:
            (rule, bound), = [(k, v) for k, v in pairs.items() if k != "weight"]
            try:
                bound_check(rule, bound, pairs["weight"])
            except LeverError as e:
                raise HarnessError(str(e)) from None
            self._bounds_checked.append(rule)
            return self
        for rule, pair in pairs.items():
            try:
                bound_check(rule, pair[0], pair[1])
            except LeverError as e:                 # surface as ONE harness exception type
                raise HarnessError(str(e)) from None
            self._bounds_checked.append(rule)
        return self

    def validate_instrument(self, metric, positive, negative, n=30, alpha=0.05,
                            min_power=0.9, max_fpr=0.15):
        """REQUIRED before any verdict. `metric(case) -> p_value`. Both directions, measured not asserted.

        `positive` and `negative` are callables taking a draw index and returning a case the metric should and
        should not flag. Power and false-positive RATE are measured over `n` independent draws -- never judged
        from a single draw, which is how a legitimate borderline p=0.0398 was briefly mistaken for a broken gate.
        """
        pos_p, neg_p = [], []
        for i in range(int(n)):
            pos_p.append(float(metric(positive(i))))
            neg_p.append(float(metric(negative(i))))
        power = sum(1 for p in pos_p if p < alpha) / float(len(pos_p))
        fpr = sum(1 for p in neg_p if p < alpha) / float(len(neg_p))
        self._instrument_report = dict(n=int(n), alpha=alpha, power=power, fpr=fpr,
                                       neg_p_median=sorted(neg_p)[len(neg_p) // 2])
        print("  instrument: power %.3f (want >= %.2f) | FPR %.3f (want <= %.2f) over %d draws"
              % (power, min_power, fpr, max_fpr, n))
        if power < min_power:
            raise HarnessError(
                "INSTRUMENT HAS NO POWER (%.3f): it fails to detect an effect that IS present, so a NEGATIVE "
                "from it is UNINTERPRETABLE -- not a scientific null. (Earned: a control that agreed with its "
                "own treatment to 1e-9 in 29 of 36 runs while printing confident 'NOT place-specific' verdicts.)"
                % power)
        if fpr > max_fpr:
            raise HarnessError(
                "INSTRUMENT CRIES WOLF (FPR %.3f): it flags effects that are NOT present, so a POSITIVE from it "
                "is uninterpretable." % fpr)
        self._instrument_ok = True
        return self

    # ---------------------------------------------------------------- the only exit
    def verdict(self, observed, passed, notes="", artifact=None):
        """The ONLY way to a reportable result -- and it raises if any gate above was skipped."""
        if not self._instrument_ok:
            raise HarnessError(
                "NO VERDICT WITHOUT A VALIDATED INSTRUMENT. Call validate_instrument() first. A refutation needs "
                "its instrument verified exactly as much as a confirmation does; most of this project's "
                "retractions were correct measurements read through an unverified instrument.")
        rec = dict(name=self.name, lane=self.lane, question=self.question, hypothesis=self.hypothesis,
                   gate=self.gate, kill_criterion=self.kill, one_variable=self.one_variable,
                   arms=self.arms, supersedes=self.supersedes,
                   corpus_hits=[{"score": s, "path": p} for s, p in self._corpus_hits],
                   bounds_checked=self._bounds_checked, instrument=self._instrument_report,
                   observed=observed, passed=bool(passed), notes=notes,
                   registered_at=self._registered_at)
        print("-" * 78)
        print("  VERDICT: %s" % ("PASS — the pre-registered gate is met" if passed else
                                 "⛔ FAIL — did the KILL criterion fire? If so, RECORD it; do NOT retune."))
        print("  gate    : %s" % self.gate)
        print("  observed: %s" % json.dumps(observed, default=str)[:400])
        if not passed:
            print("  kill    : %s" % self.kill)
        if artifact:
            # FULL config into the artifact. A filename is NOT provenance -- recovering one run's pool_k once
            # required forensics on its synapse count because the knob existed only in the file's name.
            os.makedirs(os.path.dirname(os.path.abspath(artifact)), exist_ok=True)
            json.dump(rec, open(artifact, "w"), indent=1, default=str)
            print("  artifact: %s (full pre-registration + instrument report embedded)" % artifact)
        return rec


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Validate, seal, and expand machine-readable experiments.")
    subparsers = parser.add_subparsers(dest="action", required=True)
    seal_parser = subparsers.add_parser("seal", help="seal executable source, inputs, and exact outputs")
    seal_parser.add_argument("--spec", required=True)
    seal_parser.add_argument("--seal", required=True)
    plan_parser = subparsers.add_parser("plan", help="write immutable job commands and manifests")
    plan_parser.add_argument("--spec", required=True)
    plan_parser.add_argument("--partition", action="append", required=True)
    plan_parser.add_argument("--seal")
    plan_parser.add_argument("--plan-dir", required=True)
    execute_parser = subparsers.add_parser("execute-job", help=argparse.SUPPRESS)
    execute_parser.add_argument("--contract", required=True)
    execute_parser.add_argument("runner_command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == "seal":
        result = create_experiment_seal(args.spec, args.seal)
    elif args.action == "plan":
        jobs = expand_experiment_jobs(args.spec, args.partition, seal_path=args.seal)
        result = write_experiment_plan(jobs, args.plan_dir)
    else:
        command = args.runner_command[1:] if args.runner_command[:1] == ["--"] else args.runner_command
        result = execute_job_contract(_decoded_contract(args.contract), command)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
