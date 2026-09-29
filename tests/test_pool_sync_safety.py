"""tools/pool_sync.sh's rsync safety flags (2026-09-29 security re-review, M1). A pool job can be
agent-authored (pool-submit), so what a node returns is not fully trusted: the OLD pull had no `--safe-links`
(a planted symlink pointing OUTSIDE the copied tree, e.g. at `~/.ssh/id_ed25519`, would land as a real symlink
in the local repo), no `--chmod=F-x` (a planted executable file kept its executable bit), and excluded only
`*.log`/`_provenance/` -- everything else, including a `.py` file, was pulled through untouched.

Runs REAL rsync between two LOCAL scratch directories (no ssh, no real pool node -- rsync works path-to-path
locally too) using the EXACT flag list tools/pool_sync.sh now shares between both of its pull sites
(RSYNC_SAFETY_ARGS), so this is pinned to the real flags, not a hand-copied guess of them.
"""
import os
import re
import stat
import subprocess
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_POOL_SYNC = os.path.join(_ROOT, "tools", "pool_sync.sh")

# Extracted from the real script (not hand-retyped) so this test can never silently drift from what
# tools/pool_sync.sh actually runs.
_SAFETY_ARGS_RE = re.compile(r"RSYNC_SAFETY_ARGS=\((.*?)\)\n", re.S)


def _real_safety_args():
    text = open(_POOL_SYNC).read()
    m = _SAFETY_ARGS_RE.search(text)
    assert m, "could not find RSYNC_SAFETY_ARGS=(...) in tools/pool_sync.sh"
    body = m.group(1)
    # strip bash comments (a "# ..." to end of line, only outside quotes -- none of these args embed a literal
    # '#', so a simple per-line strip is safe here) and blank lines, then shell-split.
    lines = []
    for ln in body.splitlines():
        ln = ln.split("#", 1)[0].strip()
        if ln:
            lines.append(ln)
    import shlex
    return shlex.split(" ".join(lines))


def test_both_rsync_call_sites_use_the_shared_safety_args():
    text = open(_POOL_SYNC).read()
    assert text.count('"${RSYNC_SAFETY_ARGS[@]}"') == 2, (
        "both the main pull and the isolated-revision pull must use the SAME shared safety-arg array, "
        "never a re-typed copy that could drift")
    assert "--exclude='*.log'" not in text, "the old bare *.log exclude must be gone (superseded by the allow-list)"


def _run_rsync(args, src, dst):
    return subprocess.run(["rsync", "-au", "--itemize-changes"] + args + [src + "/", dst + "/"],
                           capture_output=True, text=True, timeout=15)


def test_safe_links_rejects_a_symlink_escaping_the_source_tree():
    safety = _real_safety_args()
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst, \
         tempfile.TemporaryDirectory() as secret_dir:
        secret = os.path.join(secret_dir, "id_ed25519")
        open(secret, "w").write("PRETEND-PRIVATE-KEY\n")
        os.symlink(secret, os.path.join(src, "evil.json"))   # matches the allow-list by NAME -- only --safe-links stops it
        open(os.path.join(src, "ok.json"), "w").write("{}\n")
        res = _run_rsync(safety, src, dst)
        assert res.returncode == 0, res.stdout + res.stderr
        assert not os.path.exists(os.path.join(dst, "evil.json")), (
            "a symlink escaping the source tree must never be transferred")
        assert os.path.exists(os.path.join(dst, "ok.json"))


def test_without_safe_links_the_escaping_symlink_would_have_been_pulled():
    """Failing-first proof: the OLD flag set (no --safe-links) DOES transfer it."""
    old_flags = ["--exclude=*.log", "--exclude=_provenance/"]
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst, \
         tempfile.TemporaryDirectory() as secret_dir:
        secret = os.path.join(secret_dir, "id_ed25519")
        open(secret, "w").write("PRETEND-PRIVATE-KEY\n")
        os.symlink(secret, os.path.join(src, "evil.json"))
        res = _run_rsync(old_flags, src, dst)
        assert res.returncode == 0, res.stdout + res.stderr
        assert os.path.islink(os.path.join(dst, "evil.json")), (
            "expected the OLD (no --safe-links) flags to transfer the symlink -- if this now fails, rsync's "
            "own default behaviour changed and this test's assumption needs revisiting")


def test_chmod_f_minus_x_strips_the_executable_bit():
    safety = _real_safety_args()
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        p = os.path.join(src, "result.log")
        open(p, "w").write("#!/bin/sh\necho pwned\n")
        os.chmod(p, 0o755)
        res = _run_rsync(safety, src, dst)
        assert res.returncode == 0, res.stdout + res.stderr
        dst_p = os.path.join(dst, "result.log")
        assert os.path.exists(dst_p)
        assert not (os.stat(dst_p).st_mode & stat.S_IXUSR), "the pulled file must not be executable"


def test_extension_allowlist_blocks_py_files_but_allows_result_types():
    safety = _real_safety_args()
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        names = {
            "evil.py": True,          # must be BLOCKED
            "result.json": False,
            "sidecar.prov.json": False,
            "worker.log": False,
            "table.csv": False,
            "array.npz": False,
        }
        for name in names:
            open(os.path.join(src, name), "w").write("x\n")
        os.makedirs(os.path.join(src, "_provenance"), exist_ok=True)
        open(os.path.join(src, "_provenance", "runs.jsonl"), "w").write("{}\n")

        res = _run_rsync(safety, src, dst)
        assert res.returncode == 0, res.stdout + res.stderr

        assert not os.path.exists(os.path.join(dst, "evil.py")), ".py must never be pulled"
        assert not os.path.isdir(os.path.join(dst, "_provenance")), "_provenance/ must still be excluded"
        for name, should_block in names.items():
            if not should_block:
                assert os.path.exists(os.path.join(dst, name)), "%s should have been pulled" % name
