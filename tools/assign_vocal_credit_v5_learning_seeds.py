#!/usr/bin/env python3
"""Assign and verify the sealed Gate B v5 learning seed partitions."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    ROOT
    / "tools"
    / "seed_manifests"
    / "vocal_action_credit_gate_v5_learning.json"
)
NAMESPACE = "neural-vocal-action-credit-v5-learning/formal-v1"
CANDIDATE_LOW = 71000
CANDIDATE_HIGH = 79999
PARTITION_SIZES = {"calibration": 2, "development": 4, "held_out": 2}
SCAN_DIRS = (ROOT / "research" / "runners", ROOT / "tests")


@dataclass(frozen=True, order=True)
class SeedDeclaration:
    path: str
    symbol: str
    seed: int


def _target_names(statement: ast.stmt) -> list[str]:
    targets: list[ast.expr] = []
    if isinstance(statement, ast.Assign):
        targets.extend(statement.targets)
    elif isinstance(statement, ast.AnnAssign):
        targets.append(statement.target)
    names = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
    return names


def _literal_ints(node: ast.AST) -> list[int]:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return [int(node.value)]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values: list[int] = []
        for element in node.elts:
            values.extend(_literal_ints(element))
        return values
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal_ints(node.left) + _literal_ints(node.right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in {"tuple", "list", "set"} and len(node.args) == 1:
            return _literal_ints(node.args[0])
        if node.func.id == "range" and 1 <= len(node.args) <= 3:
            args = [_literal_ints(argument) for argument in node.args]
            if all(len(values) == 1 for values in args):
                return list(range(*(values[0] for values in args)))
    return []


def _fallback_seed_declarations(source: str, path: Path, root: Path) -> list[SeedDeclaration]:
    """Conservatively scan seed constants in files using newer Python syntax."""
    declarations: list[SeedDeclaration] = []
    pattern = re.compile(
        r"^([A-Z][A-Z0-9_]*SEEDS?)\s*(?::[^=]+)?=\s*(.*)$"
    )
    for line in source.splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        symbol, expression = match.groups()
        for raw in re.findall(r"(?<![A-Za-z0-9_])-?\d+", expression):
            declarations.append(
                SeedDeclaration(
                    path=path.relative_to(root).as_posix(),
                    symbol=symbol,
                    seed=int(raw),
                )
            )
    return declarations


def scan_seed_declarations(root: Path = ROOT) -> list[SeedDeclaration]:
    declarations: list[SeedDeclaration] = []
    scan_dirs = (root / "research" / "runners", root / "tests")
    for scan_dir in scan_dirs:
        for path in sorted(scan_dir.rglob("*.py")):
            try:
                source = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise RuntimeError(f"cannot read seed declarations in {path}: {exc}") from exc
            try:
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                declarations.extend(_fallback_seed_declarations(source, path, root))
                continue
            for statement in tree.body:
                names = [
                    name
                    for name in _target_names(statement)
                    if name.isupper() and "SEED" in name
                ]
                if not names:
                    continue
                value = getattr(statement, "value", None)
                if value is None:
                    continue
                for name in names:
                    for seed in _literal_ints(value):
                        declarations.append(
                            SeedDeclaration(
                                path=path.relative_to(root).as_posix(),
                                symbol=name,
                                seed=seed,
                            )
                        )
    return sorted(set(declarations))


def assignment(declarations: list[SeedDeclaration]) -> dict[str, list[int]]:
    used = {declaration.seed for declaration in declarations}
    chosen: list[int] = []
    counter = 0
    required = sum(PARTITION_SIZES.values())
    width = CANDIDATE_HIGH - CANDIDATE_LOW + 1
    while len(chosen) < required:
        digest = hashlib.sha256(f"{NAMESPACE}:{counter}".encode()).digest()
        candidate = CANDIDATE_LOW + int.from_bytes(digest[:8], "big") % width
        counter += 1
        if candidate in used or candidate in chosen:
            continue
        chosen.append(candidate)
        if counter > width * 4:
            raise RuntimeError("could not find a collision-free seed partition")
    offset = 0
    partitions: dict[str, list[int]] = {}
    for name, size in PARTITION_SIZES.items():
        partitions[name] = chosen[offset : offset + size]
        offset += size
    return partitions


def declaration_digest(declarations: list[SeedDeclaration]) -> str:
    payload = [declaration.__dict__ for declaration in declarations]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def current_revision(root: Path = ROOT) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_manifest(root: Path = ROOT) -> dict:
    declarations = scan_seed_declarations(root)
    partitions = assignment(declarations)
    assigned = {seed for values in partitions.values() for seed in values}
    collisions = sorted(assigned & {item.seed for item in declarations})
    return {
        "schema_version": 1,
        "mechanism": "neural-vocal-action-credit-v5-learning",
        "namespace": NAMESPACE,
        "generator": "tools/assign_vocal_credit_v5_learning_seeds.py",
        "device": "not_applicable_seed_assignment_no_execution",
        "source_revision": current_revision(root),
        "scan_scope": ["research/runners/**/*.py", "tests/**/*.py"],
        "declaration_count": len(declarations),
        "declaration_digest": declaration_digest(declarations),
        "candidate_range": [CANDIDATE_LOW, CANDIDATE_HIGH],
        "partitions": partitions,
        "collisions": collisions,
        "formal_execution_open": False,
    }


def validate_manifest(manifest: dict, root: Path = ROOT) -> None:
    expected_partitions = assignment(scan_seed_declarations(root))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported seed manifest schema")
    if manifest.get("mechanism") != "neural-vocal-action-credit-v5-learning":
        raise ValueError("seed manifest mechanism mismatch")
    if manifest.get("namespace") != NAMESPACE:
        raise ValueError("seed manifest namespace mismatch")
    if manifest.get("partitions") != expected_partitions:
        raise ValueError(
            "seed manifest is no longer the deterministic collision-free assignment"
        )
    assigned = [
        seed
        for name in PARTITION_SIZES
        for seed in manifest["partitions"].get(name, [])
    ]
    if len(assigned) != sum(PARTITION_SIZES.values()) or len(set(assigned)) != len(assigned):
        raise ValueError("seed manifest partitions are incomplete or overlap")
    if manifest.get("collisions"):
        raise ValueError(f"seed manifest records collisions: {manifest['collisions']}")
    if manifest.get("formal_execution_open") is not False:
        raise ValueError("seed manifest must not open formal execution")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.write:
        manifest = build_manifest()
        validate_manifest(manifest)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"WROTE {MANIFEST.relative_to(ROOT)}")
        print(json.dumps(manifest["partitions"], sort_keys=True))
        return 0
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    print("VOCAL_CREDIT_V5_LEARNING_SEEDS_VALID")
    print(json.dumps(manifest["partitions"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
