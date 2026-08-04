#!/usr/bin/env python3
"""Create and verify ancestry attestations for immutable Git archives.

Pool source trees deliberately omit ``.git``. Provisioning records every commit
reachable from the archived revision, then binds that record into the existing
SHA-256 source manifest. Locked runners can call :func:`require_source_ancestor`
without weakening their ancestry gate when they execute from such an archive.

New locked runners should use this helper instead of invoking ``git merge-base``
directly::

    from tools.pool.provisioning import require_source_ancestor
    require_source_ancestor(REPO_ROOT, LOCKED_ANCHOR_SHA)

The same call checks Git in a checkout and the provisioned, manifest-bound
attestation in an archive. Callers may also lock the exact execution revision
with ``expected_head=...``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


SCHEMA = "sim-source-ancestry-v1"
ATTESTATION_PATH = ".source_ancestry.json"
MANIFEST_PATH = ".source_manifest.sha256"
REVISION_PATH = ".source_revision"
_COMMIT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")


class AncestryError(ValueError):
    """Raised when source ancestry cannot be established without ambiguity."""


def _commit(value: Any, field: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise AncestryError(f"{field} must be a full lowercase Git commit ID")
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AncestryError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _run_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True,
            timeout=30, check=check,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AncestryError(f"Git command failed: git {' '.join(args)}") from exc


def _git_root(root: Path) -> bool:
    completed = _run_git(root, "rev-parse", "--show-toplevel", check=False)
    if completed.returncode != 0:
        return False
    try:
        return Path(completed.stdout.strip()).resolve(strict=True) == root
    except OSError:
        return False


def _read_regular(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AncestryError(f"cannot open {label}: {path}") from exc
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise AncestryError(f"{label} is not a regular file: {path}")
            payload = handle.read()
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise AncestryError(f"cannot read {label}: {path}") from exc
    state = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    try:
        named = path.lstat()
    except OSError as exc:
        raise AncestryError(f"{label} disappeared while reading: {path}") from exc
    if stat.S_ISLNK(named.st_mode) or state(before) != state(after) or state(after) != state(named):
        raise AncestryError(f"{label} changed while reading: {path}")
    return payload


def _revision_values(root: Path) -> dict[str, str]:
    try:
        lines = _read_regular(root / REVISION_PATH, "source revision").decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise AncestryError("source revision is not ASCII") from exc
    values: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise AncestryError(f"source revision has an invalid entry on line {number}")
        values[key] = value
    return values


def _manifest_entries(payload: bytes) -> dict[str, str]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise AncestryError("source manifest is not UTF-8") from exc
    if not lines:
        raise AncestryError("source manifest is empty")
    entries: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        digest, separator, path_text = line.partition("  ")
        path = PurePosixPath(path_text)
        if (
            not separator or _SHA256.fullmatch(digest) is None or path.is_absolute()
            or not path.name or "." in path.parts or ".." in path.parts
        ):
            raise AncestryError(f"source manifest has an invalid entry on line {number}")
        normalized = path.as_posix()
        if normalized in entries:
            raise AncestryError(f"source manifest has a duplicate entry: {normalized}")
        entries[normalized] = digest
    return entries


def _ancestry_digest(commits: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{item}\n" for item in commits).encode("ascii")).hexdigest()


def create_attestation(repo: Path, output: Path, revision: str = "HEAD") -> dict[str, Any]:
    """Write a deterministic, create-only attestation for ``revision``."""
    repo = repo.resolve(strict=True)
    if not _git_root(repo):
        raise AncestryError(f"repository root is not a Git worktree: {repo}")
    resolved = _run_git(repo, "rev-parse", f"{revision}^{{commit}}").stdout.strip().lower()
    head = _commit(resolved, "resolved revision")
    raw_commits = _run_git(repo, "rev-list", head).stdout.splitlines()
    commits = sorted({_commit(item.strip().lower(), "reachable commit") for item in raw_commits})
    if head not in commits:
        raise AncestryError("reachable commit set does not contain the archived revision")
    value = {
        "ancestor_count": len(commits),
        "ancestors": commits,
        "ancestors_sha256": _ancestry_digest(commits),
        "git_sha": head,
        "schema": SCHEMA,
    }
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")
    output = Path(os.path.abspath(output))
    if output.exists() or output.is_symlink():
        raise AncestryError(f"refusing existing ancestry attestation: {output}")
    if not output.parent.is_dir():
        raise AncestryError(f"attestation parent does not exist: {output.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(output, flags, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise AncestryError(f"cannot create ancestry attestation: {output}") from exc
    return value


def _verify_archive(root: Path, anchor: str, expected_head: str | None) -> dict[str, Any]:
    revision = _revision_values(root)
    if revision.get("source_kind") != "git_archive":
        raise AncestryError("source revision does not identify a Git archive")
    head = _commit(revision.get("git_sha"), "source revision git_sha")
    if expected_head is not None and head != _commit(expected_head, "expected head"):
        raise AncestryError(f"source revision mismatch: expected {expected_head}, found {head}")

    manifest = _read_regular(root / MANIFEST_PATH, "source manifest")
    manifest_digest = hashlib.sha256(manifest).hexdigest()
    if manifest_digest != _sha256(
        revision.get("source_manifest_sha256"), "source revision manifest digest"
    ):
        raise AncestryError("source manifest digest does not match source revision")
    entries = _manifest_entries(manifest)
    expected_attestation_digest = entries.get(ATTESTATION_PATH)
    if expected_attestation_digest is None:
        raise AncestryError("source manifest does not bind the ancestry attestation")

    attestation_payload = _read_regular(root / ATTESTATION_PATH, "ancestry attestation")
    attestation_digest = hashlib.sha256(attestation_payload).hexdigest()
    if attestation_digest != expected_attestation_digest:
        raise AncestryError("ancestry attestation digest does not match source manifest")
    if attestation_digest != _sha256(
        revision.get("source_ancestry_sha256"), "source revision ancestry digest"
    ):
        raise AncestryError("ancestry attestation digest does not match source revision")
    try:
        value = json.loads(attestation_payload.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AncestryError("ancestry attestation is not valid ASCII JSON") from exc
    expected_fields = {"ancestor_count", "ancestors", "ancestors_sha256", "git_sha", "schema"}
    if not isinstance(value, dict) or set(value) != expected_fields or value.get("schema") != SCHEMA:
        raise AncestryError("ancestry attestation schema or fields are invalid")
    if _commit(value.get("git_sha"), "attestation git_sha") != head:
        raise AncestryError("ancestry attestation revision does not match source revision")
    commits = value.get("ancestors")
    if not isinstance(commits, list) or any(not isinstance(item, str) for item in commits):
        raise AncestryError("ancestry attestation commit list is invalid")
    normalized = [_commit(item, "attested commit") for item in commits]
    if normalized != sorted(set(normalized)):
        raise AncestryError("ancestry attestation commit list is not canonical")
    if value.get("ancestor_count") != len(normalized) or head not in normalized:
        raise AncestryError("ancestry attestation count or head membership is invalid")
    if _sha256(value.get("ancestors_sha256"), "attested ancestry digest") != _ancestry_digest(normalized):
        raise AncestryError("ancestry attestation commit-set digest is invalid")
    if anchor not in normalized:
        raise AncestryError(f"locked anchor is not an ancestor of archived revision: {anchor}")
    return {
        "anchor": anchor,
        "git_sha": head,
        "is_ancestor": True,
        "kind": "git_archive",
        "source_ancestry_sha256": attestation_digest,
        "source_manifest_sha256": manifest_digest,
    }


def require_source_ancestor(
    root: Path, anchor: str, *, expected_head: str | None = None,
) -> dict[str, Any]:
    """Reusable replacement for locked runners' direct ``git merge-base`` checks.

    Fail closed unless ``anchor`` is reachable from the exact checkout or bound
    archive source. No archive fallback is accepted without all hash bindings.
    """
    root = root.resolve(strict=True)
    anchor = _commit(anchor, "locked anchor")
    if expected_head is not None:
        expected_head = _commit(expected_head, "expected head")
    if _git_root(root):
        head = _commit(_run_git(root, "rev-parse", "HEAD^{commit}").stdout.strip().lower(), "Git head")
        if expected_head is not None and head != expected_head:
            raise AncestryError(f"Git revision mismatch: expected {expected_head}, found {head}")
        completed = _run_git(root, "merge-base", "--is-ancestor", anchor, head, check=False)
        if completed.returncode == 1:
            raise AncestryError(f"locked anchor is not an ancestor of Git revision: {anchor}")
        if completed.returncode != 0:
            raise AncestryError(
                f"locked anchor is not an ancestor of Git revision or is unavailable: {anchor}"
            )
        return {"anchor": anchor, "git_sha": head, "is_ancestor": True, "kind": "git"}
    return _verify_archive(root, anchor, expected_head)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="create an attestation from a Git repository")
    create.add_argument("--repo", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--revision", default="HEAD")
    verify = commands.add_parser("verify", help="require an anchor in a checkout or bound archive")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--anchor", required=True)
    verify.add_argument("--expected-head")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            result = create_attestation(args.repo, args.output, args.revision)
        else:
            result = require_source_ancestor(
                args.root, args.anchor, expected_head=args.expected_head,
            )
    except AncestryError as exc:
        raise SystemExit(f"source ancestry verification failed: {exc}") from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
