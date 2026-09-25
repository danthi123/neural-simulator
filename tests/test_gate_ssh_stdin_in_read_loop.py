"""tests for tools/gates/ssh_stdin_in_read_loop.py (CLASS SR).

Imports and exercises the REAL gate module the pre-commit registry calls -- same convention as
tests/test_gate_finding_mechanism_on_main.py (`import tools.gates.X as X_gate`), so these tests cannot drift
from what the hook actually runs. The module's own `selftest()` is the registry's own trust mechanism (a gate
whose selftest does not fail in the failing direction is treated as BROKEN); this file adds pytest-level
coverage on top so a regression here shows up as a normal test failure, not only at commit time.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

import tools.gates.ssh_stdin_in_read_loop as sr_gate

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------------------------------------
# the module's own selftest is the registry's trust mechanism -- pin it so a regression fails a normal run
# ---------------------------------------------------------------------------------------------------------
def test_registry_selftest_passes():
    problems = sr_gate.selftest()
    assert problems == [], "gate selftest reported problems: %r" % problems


def test_registry_discovers_this_gate_with_the_expected_contract():
    from tools.gates import discover
    hits = [t for t in discover() if t[0] == sr_gate.NAME]
    assert len(hits) == 1, "gates/__init__.discover() did not find exactly one %r module" % sr_gate.NAME
    name, mod, err = hits[0]
    assert err is None, "the registry reports this gate as broken: %s" % err
    assert mod.CLASS_ID == "SR"
    assert mod.BLOCKING is True


# ---------------------------------------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------------------------------------
@pytest.fixture()
def repo(tmp_path):
    return str(tmp_path)


def _write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


# ---------------------------------------------------------------------------------------------------------
# THE FAILING DIRECTION FIRST: the actual defect this gate closes
# ---------------------------------------------------------------------------------------------------------
def test_bare_ssh_without_n_in_a_while_read_loop_is_blocked(repo):
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "while IFS= read -r cand; do\n"
              "  timeout 10 ssh \"${SSH_F[@]}\" -o BatchMode=yes \"$cand\" true 2>/dev/null\n"
              "done < <(awk '{print $2}' \"$QUEUE\")\n")
    problems = sr_gate.check([f], root=repo)
    assert len(problems) == 1
    assert "x.sh:3" in problems[0]
    assert "ssh" in problems[0]


def test_rsync_e_ssh_without_n_in_a_while_read_loop_is_blocked(repo):
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "while IFS= read -r n; do\n"
              "  timeout 30 rsync -q -e \"ssh -o BatchMode=yes\" \"$n:out/\" \"dest/\"\n"
              "done < <(printf 'pool40\\n')\n")
    problems = sr_gate.check([f], root=repo)
    assert len(problems) == 1 and "rsync" in problems[0]


def test_a_known_ssh_wrapping_script_called_unprotected_in_a_while_read_loop_is_blocked(repo):
    f = _write(repo, "research/coordination/x.sh",
              "#!/usr/bin/env bash\n"
              "while IFS= read -r line; do\n"
              "  bash tools/pool_sync.sh\n"
              "done < <(printf 'a\\n')\n")
    problems = sr_gate.check([f], root=repo)
    assert len(problems) == 1 and "pool_sync.sh" in problems[0]


def test_an_ssh_call_two_levels_of_function_indirection_from_a_read_loop_is_blocked(repo):
    """Mirrors the real tools/pool_autodispatch.sh shape: pop_job's while-read loop calls
    revision_available_cached, which calls revision_available, whose ssh call is the one that mattered."""
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "revision_available() {\n"
              "  timeout 10 ssh \"${SSH_F[@]}\" -o BatchMode=yes \"$1\" \"test -f /revisions/$2/.ok\"\n"
              "}\n"
              "revision_available_cached() {\n"
              "  revision_available \"$1\" \"$2\"\n"
              "}\n"
              "pop_job() {\n"
              "  while IFS= read -r cand; do\n"
              "    revision_available_cached \"$1\" \"$cand\"\n"
              "  done < <(awk '{print $2}' \"$QUEUE\")\n"
              "}\n")
    problems = sr_gate.check([f], root=repo)
    assert len(problems) == 1
    assert "x.sh:3" in problems[0]


# ---------------------------------------------------------------------------------------------------------
# calibration: cases that must stay silent
# ---------------------------------------------------------------------------------------------------------
def test_ssh_dash_n_in_a_while_read_loop_passes(repo):
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "while IFS= read -r cand; do\n"
              "  timeout 10 ssh -n \"${SSH_F[@]}\" -o BatchMode=yes \"$cand\" true 2>/dev/null\n"
              "done < <(awk '{print $2}' \"$QUEUE\")\n")
    assert sr_gate.check([f], root=repo) == []


def test_ssh_with_dev_null_redirect_in_a_while_read_loop_passes(repo):
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "while IFS= read -r cand; do\n"
              "  timeout 10 ssh \"${SSH_F[@]}\" \"$cand\" true </dev/null 2>/dev/null\n"
              "done < <(awk '{print $2}' \"$QUEUE\")\n")
    assert sr_gate.check([f], root=repo) == []


def test_a_multiline_ssh_call_whose_n_is_on_a_continued_line_passes(repo):
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "while IFS= read -r cand; do\n"
              "  timeout 10 ssh -n \"${SSH_F[@]}\" \\\n"
              "    -o BatchMode=yes \"$cand\" true 2>/dev/null\n"
              "done < <(awk '{print $2}' \"$QUEUE\")\n")
    assert sr_gate.check([f], root=repo) == []


def test_an_explicitly_piped_in_ssh_call_passes(repo):
    """Mirrors tools/pool_lineattractor_dispatch.sh: `printf '%s\\n' "$X" | ssh "$h" '...'` -- ssh's stdin is
    the pipe, never the enclosing loop's, so no -n is needed."""
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "while IFS= read -r cand; do\n"
              "  printf '%s\\n' \"$RUNCELL\" | ssh \"$cand\" \"cat > run_cell.sh\"\n"
              "done < <(printf 'a\\nb\\n')\n")
    assert sr_gate.check([f], root=repo) == []


def test_ssh_in_a_plain_for_loop_passes(repo):
    """Mirrors tools/pool_lineattractor_dispatch.sh / tools/pool_sync.sh: `for h in "${NODES[@]}"; do` does
    not drain stdin the way `while read` does, so an unprotected ssh there is not this bug class."""
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "for h in \"${NODES[@]}\"; do\n"
              "  ssh \"$h\" \"echo hi\"\n"
              "done\n")
    assert sr_gate.check([f], root=repo) == []


def test_ssh_in_a_while_true_condition_loop_passes(repo):
    """Mirrors tools/aws_pool_node.sh's polling loops: `while :; do` has no `read`, so it never competes with
    ssh for stdin the way a `while read` loop does."""
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "while :; do\n"
              "  ssh \"$h\" true\n"
              "  sleep 5\n"
              "done\n")
    assert sr_gate.check([f], root=repo) == []


def test_a_function_never_called_from_any_read_loop_passes(repo):
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "unused_probe() {\n"
              "  ssh \"$1\" true\n"
              "}\n"
              "for h in \"${NODES[@]}\"; do echo \"$h\"; done\n")
    assert sr_gate.check([f], root=repo) == []


def test_ssh_mentioned_only_in_a_comment_passes(repo):
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "while IFS= read -r cand; do\n"
              "  # TODO: maybe ssh \"$cand\" here later\n"
              "  echo \"$cand\"\n"
              "done < <(printf 'a\\n')\n")
    assert sr_gate.check([f], root=repo) == []


def test_a_quoted_remote_command_string_with_literal_read_loop_text_does_not_leak(repo):
    """Mirrors tools/pool_provision.sh's remote cleanup payload: a double-quoted argument containing the
    literal text 'while ... read ... do ... done' must not open a spurious LOCAL loop frame that then wrongly
    flags a later, genuinely top-level, unprotected ssh call."""
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "for h in \"${NODES[@]}\"; do\n"
              "  SSH_CMD \"$h\" \"find . | while IFS= read -r p; do rm -f -- \\\"\\$p\\\"; done\"\n"
              "done\n"
              "ssh \"$onenode\" true\n")
    assert sr_gate.check([f], root=repo) == []


def test_a_known_script_call_with_dev_null_redirect_passes(repo):
    """Mirrors the real fix in research/coordination/b2b_queue_next_wave.sh."""
    f = _write(repo, "research/coordination/x.sh",
              "#!/usr/bin/env bash\n"
              "while IFS= read -r line; do\n"
              "  bash tools/pool_sync.sh </dev/null\n"
              "done < <(printf 'a\\n')\n")
    assert sr_gate.check([f], root=repo) == []


def test_only_sh_files_are_scanned(repo):
    f = _write(repo, "tools/x.py",
              "while IFS= read -r cand:\n    ssh(cand)\n")
    assert sr_gate.check([f], root=repo) == []


def test_empty_paths_list_returns_immediately(repo):
    assert sr_gate.check([], root=repo) == []


def test_standalone_audit_mode_scans_the_whole_tracked_tree(repo):
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    f = _write(repo, "tools/x.sh",
              "#!/usr/bin/env bash\n"
              "while IFS= read -r cand; do\n"
              "  ssh \"$cand\" true\n"
              "done < <(printf 'a\\n')\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    problems = sr_gate.check(None, root=repo)
    assert any("x.sh:3" in p for p in problems)


# ---------------------------------------------------------------------------------------------------------
# the real corpus: this gate must not be BROKEN (crash) over the live tree, and must find EXACTLY the one
# known, deliberately-unfixed-here instance (tools/aws_idle_stop.sh, left for a separate lane) -- this pins
# the corpus state so a NEW unprotected ssh-in-a-read-loop landing anywhere else is caught by this test too,
# not only by the pre-commit hook on the commit that introduces it.
# ---------------------------------------------------------------------------------------------------------
def test_the_real_corpus_finds_exactly_the_known_aws_idle_stop_instance():
    problems = sr_gate.check(None, root=_ROOT)
    files_and_lines = sorted((p.split(" -- ")[0]) for p in problems)
    assert files_and_lines == [
        "tools/aws_idle_stop.sh:100",
        "tools/aws_idle_stop.sh:104",
        "tools/aws_idle_stop.sh:116",
    ], ("the corpus audit no longer matches the known, documented state -- either a NEW unprotected "
        "ssh-in-a-read-loop landed (fix it), or tools/aws_idle_stop.sh was fixed (update this pin and the "
        "research/FAILURE_LOG.md row that names it), or the gate's own detection regressed: %r" % problems)


def test_the_real_corpus_fixed_sites_stay_clean():
    """tools/pool_queue.sh (e76106fd8) and tools/pool_autodispatch.sh (096dfdae0) were fixed with `ssh -n`
    BEFORE this gate existed -- pin that they stay clean so a future edit cannot silently drop the -n."""
    for rel in ("tools/pool_queue.sh", "tools/pool_autodispatch.sh"):
        full = os.path.join(_ROOT, rel)
        if not os.path.exists(full):
            pytest.skip("%s not present on this checkout" % rel)
        assert sr_gate.check([rel], root=_ROOT) == [], "%s regressed: unprotected ssh-in-read-loop reappeared" % rel


# ---------------------------------------------------------------------------------------------------------
# an end-to-end proof that the underlying MECHANISM this gate flags is real: a genuinely stdin-draining ssh
# stub (not the `echo "$*" >> log; exit N` stub every OTHER pool test in this repo uses, which never touches
# stdin and so cannot reproduce this bug) shows an unprotected ssh call inside a while-read loop really does
# truncate the scan, and that -n really does fix it -- same technique as
# tests/test_pool_autodispatch_workflow.py::test_pop_job_does_not_let_an_unavailable_revision_probe_swallow_later_queued_candidates.
# ---------------------------------------------------------------------------------------------------------
def _write_stdin_draining_ssh_stub(bin_dir):
    stub = os.path.join(bin_dir, "ssh")
    with open(stub, "w", encoding="utf-8") as fh:
        fh.write(
            "#!/usr/bin/env bash\n"
            "has_n=0\n"
            "for a in \"$@\"; do [ \"$a\" = \"-n\" ] && has_n=1; done\n"
            "[ \"$has_n\" = 0 ] && cat >/dev/null\n"
            "exit 0\n"
        )
    os.chmod(stub, 0o755)


def test_the_flagged_pattern_really_does_drop_later_candidates_without_n(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stdin_draining_ssh_stub(str(bin_dir))
    script = tmp_path / "loop.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        "seen=()\n"
        "while IFS= read -r cand; do\n"
        "  ssh \"$cand\" true\n"
        "  seen+=(\"$cand\")\n"
        "done < <(printf 'a\\nb\\nc\\n')\n"
        "printf '%s\\n' \"${seen[@]}\"\n"
    )
    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=10).stdout
    seen = [ln for ln in out.splitlines() if ln]
    assert seen == ["a"], (
        "the unprotected-ssh-in-a-while-read-loop pattern this gate flags did NOT reproduce the drain under a "
        "real stdin-draining ssh stub -- got %r (expected only 'a': b and c's lines were consumed by ssh's own "
        "stdin read, the exact 2026-09-25 incident shape)" % seen
    )


def test_adding_n_fixes_the_same_loop(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_stdin_draining_ssh_stub(str(bin_dir))
    script = tmp_path / "loop.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        "set -u\n"
        "seen=()\n"
        "while IFS= read -r cand; do\n"
        "  ssh -n \"$cand\" true\n"
        "  seen+=(\"$cand\")\n"
        "done < <(printf 'a\\nb\\nc\\n')\n"
        "printf '%s\\n' \"${seen[@]}\"\n"
    )
    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=10).stdout
    seen = [ln for ln in out.splitlines() if ln]
    assert seen == ["a", "b", "c"], "ssh -n should let the while-read loop process every candidate: got %r" % seen
