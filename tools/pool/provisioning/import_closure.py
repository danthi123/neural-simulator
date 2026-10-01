#!/usr/bin/env python3
"""Static import closure of pool-run code: which third-party modules can a runner reach?

Why this exists (2026-09-30): the pool venvs were built from a hand-kept pip list in tools/pool_provision.sh
that no check compared against what pool jobs import. `httpx` was missing, so every runner that drives the live
`/api/brain-chat` handler through `starlette.testclient` (iw1_default_identity at 049d4bb0b) died at import on
the pool, while the workstation venv (which has httpx) never showed it. A top-level `import runner` does not
catch this either: the brain-chat harness imports `starlette.testclient` LAZILY, inside a function.

This walks the source tree statically (AST, no execution), from one or more entry modules, following every
first-party import it can resolve -- module-level AND function-level, `from x import submodule`, literal
`importlib.import_module("...")`, and literal `-m research.runners.X` spawn targets -- and records each
third-party import it meets, with two flags:

  eager    -- reached only through module-level imports, so `python -m <entry>` fails at start without it;
  guarded  -- inside a `try:` whose handler catches ImportError (the code has a fallback when it is absent).

Some third-party SUBMODULES need a package their own distribution does not declare as a hard dependency
(`starlette.testclient` needs `httpx`); `SUBMODULE_EXTRA_IMPORTS` names those so a closure that reaches the
submodule also reaches the extra package.

POOL COVERAGE (`pool_gaps`): a third-party module a pool runner reaches is a GAP unless
  * requirements-pool.txt provides it (a pinned distribution or one of their dependencies, resolved through
    importlib.metadata in the running environment), or
  * every site that imports it is guarded (the code has a fallback), or
  * it is NOT_ON_POOL (the GPU / LLM stack, absent on CPU nodes by design) AND reached only lazily (inside a
    function), never at module level -- an eager NOT_ON_POOL import means the runner cannot even start there.

CLI (prints JSON):
    python -m tools.pool.provisioning.import_closure --root . research.runners.load_bearing_fraction ...
    python -m tools.pool.provisioning.import_closure --queue research/queue/pool.queue.held
        (every distinct (runner, revision) in a pool queue file, each read from `git archive <revision>`, checked
         against requirements-pool.txt; exit 1 if any has a gap)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence


# Third-party submodules whose import pulls in a package the parent distribution does not require, as
# ALTERNATIVES (any one satisfies it; the first names the gap). starlette/testclient.py (1.x) tries
# `import httpx2`, falls back to `import httpx` (deprecation warning), else raises RuntimeError; fastapi.testclient
# re-exports it. starlette declares both only under its optional "full" extra.
SUBMODULE_EXTRA_IMPORTS: dict[str, tuple[str, ...]] = {
    "starlette.testclient": ("httpx", "httpx2"),
    "fastapi.testclient": ("httpx", "httpx2"),
}

# Absent on the CPU pool BY DESIGN (pool jobs run SIM_BACKEND=numpy with no GPU and no local LLM). Reaching one
# lazily is tolerated (a backend / LLM switch keeps a pool run off that path); reaching one at module level is a gap.
NOT_ON_POOL: dict[str, str] = {
    "cupy": "GPU array backend; sim.backend imports it only for SIM_BACKEND=cupy",
    "cupyx": "GPU sparse backend (ships with cupy); same switch as cupy",
    "torch": "LLM / transformer stack; pool jobs run with no GPU and no local model",
    "transformers": "LLM stack (the Qwen mouth); pool jobs run with no local model",
}

# Runners whose lines the pool runs: the gate families and the batteries they wrap. tests/test_pool_venv_deps.py
# checks every one PRESENT in the tree (most live on research branches until they merge).
POOL_RUNNERS: tuple[str, ...] = (
    "research.runners.host_share",
    "research.runners.host_share_bg_seam",
    "research.runners.iw1_default_identity",
    "research.runners.load_bearing_fraction",
    "research.runners.long_delay_credit_bg",
    "research.runners.onebrain_regression_battery",
    "research.runners.sleep_transitive_inference",
    "research.runners.stage1_c1_category",
    "research.runners.stage1_exit_gate",
    "research.runners.stage2_exit_gate",
)

_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
_IMPORT_ERRORS = frozenset({"ImportError", "ModuleNotFoundError", "Exception", "BaseException"})
_SPAWN_TARGET = re.compile(r"^(?:research\.runners|sim|webapp|tools|experiment)(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")


@dataclass(frozen=True)
class ThirdPartySite:
    target: str          # dotted name as imported, e.g. "starlette.testclient"
    importer: str        # first-party module that imports it
    lineno: int
    eager: bool
    guarded: bool

    @property
    def top(self) -> str:
        return self.target.split(".", 1)[0]


@dataclass
class Closure:
    modules: set[str] = field(default_factory=set)          # first-party modules reached (any path)
    eager_modules: set[str] = field(default_factory=set)    # first-party modules reached via module-level imports
    sites: list[ThirdPartySite] = field(default_factory=list)

    def third_party(self) -> dict[str, dict[str, object]]:
        """top-level third-party name -> {eager, guarded_everywhere, targets, first_site, alternatives}."""
        out: dict[str, dict[str, object]] = {}
        for site in self.sites:
            names = [(site.top, site.target, (site.top,))]
            for sub, extras in SUBMODULE_EXTRA_IMPORTS.items():
                if site.target == sub or site.target.startswith(sub + "."):
                    names.append((extras[0], f"{' or '.join(extras)} (needed by {sub})", extras))
            for top, target, alternatives in names:
                row = out.setdefault(top, {"eager": False, "guarded_everywhere": True, "targets": set(),
                                           "first_site": f"{site.importer}:{site.lineno}",
                                           "alternatives": list(alternatives)})
                row["eager"] = bool(row["eager"]) or (site.eager and not site.guarded)
                row["guarded_everywhere"] = bool(row["guarded_everywhere"]) and site.guarded
                row["targets"].add(target)  # type: ignore[union-attr]
        for row in out.values():
            row["targets"] = sorted(row["targets"])  # type: ignore[arg-type]
        return dict(sorted(out.items()))


class _Resolver:
    def __init__(self, root: Path):
        self.root = root
        self._cache: dict[str, Path | None] = {}
        self.first_party_tops = {
            p.stem if p.suffix == ".py" else p.name
            for p in root.iterdir()
            if (p.is_dir() and not p.name.startswith(".") and any(p.glob("*.py")))
            or (p.suffix == ".py" and p.is_file())
        }

    def is_first_party(self, dotted: str) -> bool:
        return dotted.split(".", 1)[0] in self.first_party_tops

    def path(self, dotted: str) -> Path | None:
        """Source file for a first-party module (package -> __init__.py); None if it is not a module file."""
        if dotted not in self._cache:
            base = self.root.joinpath(*dotted.split("."))
            found: Path | None = None
            if base.with_suffix(".py").is_file():
                found = base.with_suffix(".py")
            elif (base / "__init__.py").is_file():
                found = base / "__init__.py"
            self._cache[dotted] = found
        return self._cache[dotted]

    def is_package(self, dotted: str) -> bool:
        return self.root.joinpath(*dotted.split(".")).is_dir()


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    for node in types:
        name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", None)
        if name in _IMPORT_ERRORS:
            return True
    return False


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING")


def _walk(node: ast.AST, eager: bool, guarded: bool) -> Iterator[tuple[ast.AST, bool, bool]]:
    """Yield (import-ish node, eager, guarded) for every import in `node`'s subtree."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.Import, ast.ImportFrom)):
            yield child, eager, guarded
        elif isinstance(child, ast.Call):
            yield child, eager, guarded
            yield from _walk(child, eager, guarded)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child, eager, guarded
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            yield from _walk(child, False, guarded)
        elif isinstance(child, ast.If) and _is_type_checking(child.test):
            yield from _walk(ast.Module(body=child.orelse, type_ignores=[]), eager, guarded)
        elif isinstance(child, ast.Try) or type(child).__name__ == "TryStar":
            catches = any(_catches_import_error(h) for h in child.handlers)  # type: ignore[attr-defined]
            yield from _walk(ast.Module(body=child.body, type_ignores=[]), eager, guarded or catches)  # type: ignore[attr-defined]
            for handler in child.handlers:  # type: ignore[attr-defined]
                yield from _walk(handler, eager, guarded)
            yield from _walk(ast.Module(body=[*child.orelse, *child.finalbody], type_ignores=[]),  # type: ignore[attr-defined]
                             eager, guarded)
        else:
            yield from _walk(child, eager, guarded)


def _resolve_relative(module: str, is_pkg: bool, level: int, name: str | None) -> str:
    parts = module.split(".")
    base = parts if is_pkg else parts[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)]
    return ".".join([*base, name] if name else base)


def import_closure(root: str | Path, entries: Iterable[str]) -> Closure:
    resolver = _Resolver(Path(root).resolve())
    closure = Closure()
    # queue of (module, eager)
    queue: list[tuple[str, bool]] = []

    def enqueue(dotted: str, eager: bool) -> None:
        parts = dotted.split(".")
        for i in range(1, len(parts) + 1):   # importing a.b.c runs a/__init__ and a/b/__init__ too
            name = ".".join(parts[:i])
            if resolver.path(name) is None:
                continue
            if eager and name not in closure.eager_modules:
                closure.eager_modules.add(name)
                closure.modules.add(name)
                queue.append((name, True))
            elif name not in closure.modules:
                closure.modules.add(name)
                queue.append((name, False))

    for entry in entries:
        if resolver.path(entry) is None:
            raise FileNotFoundError(f"entry module not found under {resolver.root}: {entry}")
        enqueue(entry, True)

    seen_sites: set[tuple[str, str, int, bool, bool]] = set()
    while queue:
        module, module_eager = queue.pop()
        source = resolver.path(module)
        assert source is not None
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        except SyntaxError:
            continue
        is_pkg = source.name == "__init__.py"
        for node, eager, guarded in _walk(tree, module_eager, False):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (_resolve_relative(module, is_pkg, node.level, node.module) if node.level
                        else (node.module or ""))
                if not base:
                    continue
                targets = [base]
                for alias in node.names:
                    if alias.name != "*":
                        sub = f"{base}.{alias.name}"
                        if resolver.is_first_party(sub) and resolver.path(sub) is not None:
                            targets.append(sub)
                        elif not resolver.is_first_party(sub):
                            targets.append(sub)   # e.g. `from starlette import testclient`
            elif isinstance(node, ast.Call):
                func = node.func
                fname = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
                if fname in {"import_module", "__import__"} and node.args and isinstance(node.args[0], ast.Constant) \
                        and isinstance(node.args[0].value, str) and not node.args[0].value.startswith("."):
                    targets = [node.args[0].value]
            elif isinstance(node, ast.Constant):
                value = node.value
                if isinstance(value, str) and _SPAWN_TARGET.match(value) and resolver.path(value) is not None:
                    enqueue(value, False)   # a subprocess spawn target: its own process, reached at run time
                continue
            for target in targets:
                if not target:
                    continue
                if resolver.is_first_party(target):
                    enqueue(target, eager)
                elif target.split(".", 1)[0] not in _STDLIB:
                    key = (target, module, getattr(node, "lineno", 0), eager, guarded)
                    if key not in seen_sites:
                        seen_sites.add(key)
                        closure.sites.append(ThirdPartySite(target, module, getattr(node, "lineno", 0), eager, guarded))
    return closure


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_names(path: str | Path) -> list[str]:
    """Distribution names pinned in a requirements-pool.txt (`name==version` lines)."""
    names = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if line and not line.startswith("#"):
            names.append(_normalize(re.split(r"[=<>!~;\[ ]", line, maxsplit=1)[0]))
    return names


class RequirementNotInstalled(LookupError):
    """A requirement cannot be resolved because it is not installed in the environment running the check."""


def provided_modules(requirements: Iterable[str]) -> set[str]:
    """Top-level import names the given distributions and their (non-extra) dependencies provide, per
    importlib.metadata of the RUNNING environment (which must have the pinned set installed)."""
    from importlib import metadata

    todo = [_normalize(n) for n in requirements]
    direct = set(todo)
    dists: set[str] = set()
    missing: list[str] = []
    while todo:
        name = todo.pop()
        if name in dists:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            if name in direct:
                missing.append(name)
            continue   # a dependency behind an environment marker that does not apply here
        dists.add(name)
        for requirement in dist.requires or []:
            if "extra ==" in requirement.replace("extra==", "extra =="):
                continue
            match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
            if match:
                todo.append(_normalize(match.group(0)))
    if missing:
        raise RequirementNotInstalled(f"not installed here, cannot resolve: {sorted(missing)}")
    from tools.pool.provisioning.pool_venv_check import top_level_modules   # the same inference the node runs

    return {module for name in dists for module in top_level_modules(name)}


def pool_gaps(closure: Closure, provided: set[str]) -> dict[str, dict[str, object]]:
    """Third-party modules the closure reaches that a pool venv built from `provided` would not satisfy."""
    gaps: dict[str, dict[str, object]] = {}
    for top, row in closure.third_party().items():
        if any(name in provided for name in row["alternatives"]) or row["guarded_everywhere"]:  # type: ignore[union-attr]
            continue
        if top in NOT_ON_POOL:
            if not row["eager"]:
                continue
            reason = f"imported at module level but NOT_ON_POOL ({NOT_ON_POOL[top]}): the runner cannot start"
        else:
            reason = "not provided by requirements-pool.txt"
        gaps[top] = dict(row, reason=reason)
    return gaps


_QUEUE_REVISION = re.compile(r"derisk-pool/revisions/([0-9a-f]{40})")
_QUEUE_RUNNER = re.compile(r"-m\s+(research\.runners\.[A-Za-z0-9_]+)")
_ARCHIVE_PATHS = ("sim", "webapp", "research/__init__.py", "research/runners", "tools", "experiment")


def queue_pairs(queue_file: str | Path) -> list[tuple[str, str]]:
    """Distinct (runner module, revision sha) pairs in a pool queue file; unpinned lines use 'HEAD'."""
    pairs: list[tuple[str, str]] = []
    for line in Path(queue_file).read_text(encoding="utf-8").splitlines():
        runner = _QUEUE_RUNNER.search(line)
        if not runner:
            continue
        revision = _QUEUE_REVISION.search(line)
        pair = (runner.group(1), revision.group(1) if revision else "HEAD")
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def check_queue(queue_file: str | Path, requirements: str | Path, repo: str | Path = ".") -> dict[str, object]:
    provided = provided_modules(requirement_names(requirements))
    results: dict[str, object] = {}
    by_revision: dict[str, list[str]] = {}
    for runner, revision in queue_pairs(queue_file):
        by_revision.setdefault(revision, []).append(runner)
    for revision, runners in by_revision.items():
        with tempfile.TemporaryDirectory(prefix="pool-closure-") as tmp:
            archive = subprocess.run(["git", "-C", str(repo), "archive", revision, *_ARCHIVE_PATHS],
                                     check=True, capture_output=True).stdout
            subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
            for runner in runners:
                key = f"{runner}@{revision[:9]}"
                try:
                    closure = import_closure(tmp, [runner])
                except FileNotFoundError as exc:
                    results[key] = {"error": str(exc)}
                    continue
                results[key] = {"gaps": pool_gaps(closure, provided),
                                "reached_not_on_pool": sorted(t for t in closure.third_party() if t in NOT_ON_POOL),
                                "optional_guarded": sorted(t for t, r in closure.third_party().items()
                                                           if r["guarded_everywhere"] and t not in provided)}
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=".")
    parser.add_argument("--queue", help="check every (runner, revision) in this pool queue file instead")
    parser.add_argument("--requirements", default="requirements-pool.txt")
    parser.add_argument("entries", nargs="*", help="dotted entry modules, e.g. research.runners.load_bearing_fraction")
    args = parser.parse_args(argv)
    if args.queue:
        results = check_queue(args.queue, args.requirements, args.root)
        json.dump(results, sys.stdout, indent=1, sort_keys=True)
        print()
        return 1 if any(isinstance(r, dict) and (r.get("gaps") or r.get("error")) for r in results.values()) else 0
    if not args.entries:
        parser.error("give entry modules or --queue")
    closure = import_closure(args.root, args.entries)
    out: dict[str, object] = {"entries": args.entries, "first_party_modules": len(closure.modules),
                              "third_party": closure.third_party()}
    if Path(args.requirements).is_file():
        out["pool_gaps"] = pool_gaps(closure, provided_modules(requirement_names(args.requirements)))
    json.dump(out, sys.stdout, indent=1, sort_keys=True)
    print()
    return 1 if out.get("pool_gaps") else 0


if __name__ == "__main__":
    raise SystemExit(main())
