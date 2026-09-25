"""tests for tools/gates/ssh_stdin_in_read_loop.py (CLASS SR).

Imports the REAL gate module the pre-commit registry calls (`import tools.gates.X as X_gate`), so these tests cannot
drift from what the hook runs. Three kinds of ground truth, not only fixtures written to match the code:
  * git history -- the three real incidents are replayed from the commits that introduced and fixed them;
  * bash itself -- a stub `ssh` that really reads stdin (honouring -n/-f like OpenSSH) runs each shape, and the
    gate's verdict must equal whether the loop really lost lines;
  * OpenSSH itself -- `ssh -G` (prints the resolved config, never connects) says whether an argv sets StdinNull.
Every fixture repo is a tmp_path; nothing here touches research/queue/*, .pool_ssh_config, the spend ledger or a
real host.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import warnings

import pytest

import tools.gates.ssh_stdin_in_read_loop as sr_gate

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GATE_SRC = os.path.join(_ROOT, "tools", "gates", "ssh_stdin_in_read_loop.py")

# tools/aws_idle_stop.sh's three probes, left for the lane that owns that file (fixed on
# research/aws-pool-stop-start-safety, not yet on main). The corpus test accepts any SUBSET of these.
KNOWN_UNFIXED = frozenset(("tools/aws_idle_stop.sh:100", "tools/aws_idle_stop.sh:104", "tools/aws_idle_stop.sh:116"))


def _clean_env():
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("GIT_"):
            env.pop(k)
    return env


def _git(repo, *args, check=True):
    return subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "user.email=t@example.invalid",
                           "-c", "user.name=t"] + list(args), cwd=repo, env=_clean_env(), capture_output=True,
                          text=True, check=check)


def _write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


def _flags(text, known=None):
    return sr_gate._find_unprotected(text, known)


def _loop(body):
    return "#!/usr/bin/env bash\nwhile IFS= read -r h; do\n" + body + "done < <(printf 'a\\nb\\nc\\n')\n"


# ---------------------------------------------------------------------------------------------------------------
# the registry contract
# ---------------------------------------------------------------------------------------------------------------
def test_registry_selftest_passes():
    assert sr_gate.selftest() == []


def test_registry_discovers_this_gate_with_the_expected_contract():
    from tools.gates import discover
    hits = [t for t in discover() if t[0] == sr_gate.NAME]
    assert len(hits) == 1
    _name, mod, err = hits[0]
    assert err is None, err
    assert mod.CLASS_ID == "SR" and mod.BLOCKING is True


def test_the_selftest_fails_when_detection_is_disabled(monkeypatch):
    """The registry only trusts a gate whose selftest FAILS in the failing direction."""
    monkeypatch.setattr(sr_gate, "_evaluate", lambda summary, known: ({}, False))
    sr_gate._SUMMARY_CACHE.clear()
    assert any(p.startswith("MISSED") for p in sr_gate.selftest())


def test_the_module_compiles_with_warnings_as_errors():
    """Review r2 LOW: the docstring was not raw, so an escape such as \\w was a SyntaxWarning, which `python -W
    error` turns into a SyntaxError -- the registry would mark the gate BROKEN and block every commit. Compiled
    from source (an existing .pyc would skip the warning)."""
    with open(_GATE_SRC, encoding="utf-8") as fh:
        src = fh.read()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile(src, _GATE_SRC, "exec")


# ---------------------------------------------------------------------------------------------------------------
# HIGH 1: the hook passes only ADDED files; the gate must find MODIFIED staged scripts itself, from the index
# ---------------------------------------------------------------------------------------------------------------
_CLEAN = "#!/usr/bin/env bash\nwhile IFS= read -r c; do\n  timeout 10 ssh -n \"$c\" true\ndone < q\n"
_BROKEN = _CLEAN.replace("ssh -n ", "ssh ")


@pytest.fixture()
def repo(tmp_path):
    r = str(tmp_path)
    _git(r, "init", "-q")
    return r


def test_a_staged_modification_is_scanned_although_the_hook_passes_only_added_files(repo):
    _write(repo, "tools/probe.sh", _CLEAN)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    _write(repo, "tools/probe.sh", _BROKEN)
    _git(repo, "add", "tools/probe.sh")
    assert "M\ttools/probe.sh" in _git(repo, "diff", "--cached", "--name-status").stdout
    # the hook's own list (--diff-filter=A) is EMPTY for this commit, and an unrelated added file changes nothing
    for hook_paths in ([], ["notes.md"]):
        probs = sr_gate.check(hook_paths, root=repo)
        assert [p.split(" -- ")[0] for p in probs] == ["tools/probe.sh:3"], probs


def test_the_staged_blob_is_judged_not_the_working_tree(repo):
    _write(repo, "tools/probe.sh", _BROKEN)
    _git(repo, "add", "tools/probe.sh")
    _write(repo, "tools/probe.sh", _CLEAN)          # fixed on disk, NOT staged: the commit would still ship the bug
    assert [p.split(" -- ")[0] for p in sr_gate.check([], root=repo)] == ["tools/probe.sh:3"]
    _write(repo, "tools/probe.sh", _BROKEN)
    _git(repo, "add", "tools/probe.sh")
    _write(repo, "tools/probe.sh", _CLEAN)
    _git(repo, "add", "tools/probe.sh")
    _write(repo, "tools/probe.sh", _BROKEN)         # broken on disk, clean staged: the commit is clean
    assert sr_gate.check([], root=repo) == []


def test_nothing_staged_and_no_shell_paths_returns_nothing(repo):
    _write(repo, "tools/probe.sh", _BROKEN)         # present on disk but never staged
    assert sr_gate.check([], root=repo) == []
    assert sr_gate.check(["README.md"], root=repo) == []


def test_a_staged_callee_that_starts_draining_reports_its_unstaged_callers(repo):
    """A wrapper that loses its -n turns every existing unprotected caller-in-a-read-loop into the incident."""
    _write(repo, "tools/mywrap.sh", "#!/usr/bin/env bash\nssh -n \"$1\" true\n")
    _write(repo, "tools/caller.sh", "#!/usr/bin/env bash\nwhile read -r h; do\n  bash tools/mywrap.sh \"$h\"\n"
                                    "done < q\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    assert sr_gate.check(None, root=repo) == []
    _write(repo, "tools/mywrap.sh", "#!/usr/bin/env bash\nssh \"$1\" true\n")
    _git(repo, "add", "tools/mywrap.sh")
    assert [p.split(" -- ")[0] for p in sr_gate.check([], root=repo)] == ["tools/caller.sh:3"]


# ---------------------------------------------------------------------------------------------------------------
# the three real incidents, replayed from git history with each commit's OWN tree as the corpus
# ---------------------------------------------------------------------------------------------------------------
def _replay(commit, path):
    """Problems the hook would have reported for `path` staged in `commit` (skip if history is unavailable)."""
    r = subprocess.run(["git", "ls-tree", "-r", "-z", commit], cwd=_ROOT, capture_output=True)
    if r.returncode != 0:
        pytest.skip("commit %s is not in this clone" % commit)
    shas = {}
    for rec in r.stdout.decode("utf-8", "replace").split("\0"):
        if "\t" in rec:
            meta, p = rec.split("\t", 1)
            if p.endswith(".sh") and meta.split()[1] == "blob":
                shas[p] = meta.split()[2]
    blobs = sr_gate._cat_blobs(_ROOT, list(shas.values()))
    texts = {p: blobs[s] for p, s in shas.items() if s in blobs}
    assert path in texts
    return [p.split(" -- ")[0] for p in sr_gate._report(texts, [path], True)]


def _status(commit, path):
    r = subprocess.run(["git", "diff-tree", "-r", "--no-commit-id", "--name-status", commit + "^", commit, "--",
                        path], cwd=_ROOT, capture_output=True, text=True)
    return r.stdout.split("\t")[0].strip()


def test_incident_1_b2b_queue_next_wave_calling_pool_queue_through_a_quoted_variable_is_blocked():
    """8cc766c48 added `bash "$QUEUE_TOOL" add "$line" ...` inside `while IFS= read -r line; ... done < "$JOBS"`.
    The previous gate masked the quoted variable and returned []."""
    path = "research/coordination/b2b_queue_next_wave.sh"
    assert _replay("8cc766c48", path) == [path + ":104"]
    assert _replay("6406924ee^", path) == [path + ":104"]
    assert _replay("6406924ee", path) == []         # the fix: read -r line <&3 ... done 3< "$JOBS", add </dev/null


def test_incident_2_pool_autodispatch_arrived_as_a_modification_and_is_blocked():
    path = "tools/pool_autodispatch.sh"
    assert _status("9f7d4095b", path) == "M"         # the hook's --diff-filter=A would never have shown it
    assert _replay("9f7d4095b", path) == [path + ":70"]
    assert _replay("096dfdae0", path) == []


def test_incident_3_aws_idle_stop_is_blocked_when_added_and_the_lane_a_fix_passes():
    path = "tools/aws_idle_stop.sh"
    assert _replay("ab37684784", path) == [path + ":82", path + ":86", path + ":98"]
    fix = subprocess.run(["git", "rev-parse", "--verify", "-q", "75dfb7960^{commit}"], cwd=_ROOT,
                         capture_output=True)
    if fix.returncode == 0:                             # research/aws-pool-stop-start-safety @ 75dfb7960
        assert _replay("75dfb7960", path) == []


def test_pool_queue_is_derived_as_a_stdin_draining_script_before_its_fix_and_not_after():
    """The wrapper list is computed from the corpus, not a fixed registry: pool_queue.sh drained its callers'
    stdin until e76106fd8 put -n on every probe."""
    def drains(commit):
        r = subprocess.run(["git", "show", commit + ":tools/pool_queue.sh"], cwd=_ROOT, capture_output=True,
                           text=True)
        if r.returncode != 0:
            pytest.skip("history unavailable")
        return sr_gate._evaluate(sr_gate._summarize(r.stdout), frozenset())[1]
    assert drains("e76106fd8^") is True
    assert drains("e76106fd8") is False


# ---------------------------------------------------------------------------------------------------------------
# HIGH 2: known scripts reached through variables and quoted paths; the wrapper list comes from the corpus
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("call", [
    'bash "$QUEUE_TOOL" add "$h"',
    '"$QUEUE_TOOL" add "$h"',
    'bash "${POOL_ROOT}/tools/pool_queue.sh" add "$h"',
    '"$ROOT"/tools/pool_sync.sh',
    'source "$ROOT/tools/pool_sync.sh"',
    'timeout 60 env X=1 bash "$QUEUE_TOOL" add "$h"',
])
def test_a_known_script_reached_through_a_variable_or_a_quoted_path_is_blocked(call):
    text = 'QUEUE_TOOL="$POOL_ROOT/tools/pool_queue.sh"\n' + _loop("  %s\n" % call)
    assert [p[0] for p in _flags(text)] == [4], call
    assert _flags(text.replace(call, call + " </dev/null")) == []


def test_a_script_that_drains_stdin_is_derived_from_the_corpus(tmp_path):
    root = str(tmp_path)
    _write(root, "tools/newwrap.sh", "#!/usr/bin/env bash\ntimeout 5 ssh -o BatchMode=yes \"$1\" uptime\n")
    caller = _write(root, "research/q.sh", _loop('  bash "$ROOT/tools/newwrap.sh" "$h"\n'))
    assert "newwrap.sh" not in sr_gate._FLOOR_SCRIPTS
    assert [p.split(" -- ")[0] for p in sr_gate.check([caller], root=root)] == ["research/q.sh:3"]
    _write(root, "tools/newwrap.sh", "#!/usr/bin/env bash\ntimeout 5 ssh -n -o BatchMode=yes \"$1\" uptime\n")
    assert sr_gate.check([caller], root=root) == []


def test_a_script_calling_a_draining_script_is_itself_draining(tmp_path):
    root = str(tmp_path)
    _write(root, "tools/inner.sh", "#!/usr/bin/env bash\nssh \"$1\" true\n")
    _write(root, "tools/outer.sh", "#!/usr/bin/env bash\nmain() { bash tools/inner.sh \"$1\"; }\nmain \"$@\"\n")
    caller = _write(root, "research/q.sh", _loop("  bash tools/outer.sh \"$h\"\n"))
    assert [p.split(" -- ")[0] for p in sr_gate.check([caller], root=root)] == ["research/q.sh:3"]


@pytest.mark.parametrize("prelude,call,flag", [
    ('SSH="ssh -o BatchMode=yes"\n', '$SSH "$h" true', True),
    ('SSH="ssh -n -o BatchMode=yes"\n', '$SSH "$h" true', False),
    ('SSH=(ssh -o BatchMode=yes)\n', '"${SSH[@]}" "$h" true', True),
    ('SSH_F=(-n -o BatchMode=yes)\n', 'ssh "${SSH_F[@]}" "$h" true', False),
    ('SSH_F=(-F "$CFG")\n', 'ssh "${SSH_F[@]}" "$h" true', True),
    ('SSH="${1:?usage}"\n', '$SSH "cd x && ls"', True),       # an ssh command this file cannot see
])
def test_ssh_through_variables(prelude, call, flag):
    assert bool(_flags(prelude + _loop("  %s\n" % call))) is flag


# ---------------------------------------------------------------------------------------------------------------
# MEDIUM: statement boundaries -- -n and </dev/null count only as ssh's OWN option / the call's OWN redirect
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("body", [
    'ssh "$h" "tail -n 5 log"',
    "ssh \"$h\" head -n 1 f",
    'ssh "$h" "nohup x </dev/null &"',
    "ssh \"$h\" cat f | sort -n",
    "ssh \"$h\" true && echo -n ok",
    "ssh \"$h\" true; echo x </dev/null",
    "ssh \"$h\" true || cat </dev/null",
])
def test_a_dash_n_or_redirect_that_is_not_ssh_s_own_does_not_protect_it(body):
    assert [p[0] for p in _flags(_loop("  %s\n" % body))] == [3], body


def test_a_quoted_semicolon_does_not_end_the_statement_before_its_redirect():
    assert _flags(_loop('  ssh "$h" "cd /x; ls" </dev/null\n')) == []


@pytest.mark.parametrize("body,flag", [
    ('echo "$h" | grep -q x && ssh "$h" true', True),
    ('echo "$h" | grep -q x || ssh "$h" true', True),
    ('echo "$h" | grep -q x & ssh "$h" true', True),
    ('( echo "$h" | cat ) && ssh "$h" true', True),
    ('r=$(echo "$h" | cat) ssh "$h" true', True),
    ('if echo | cat; then ssh "$h" true; fi', True),
    ('echo "$h" | ssh "$h" "cat > f"', False),
    ('echo "$h" | grep -q x && echo y | ssh "$h" "cat > f"', False),
])
def test_only_a_pipe_into_the_call_itself_protects_it(body, flag):
    assert bool(_flags(_loop("  %s\n" % body))) is flag, body


# ---------------------------------------------------------------------------------------------------------------
# MEDIUM: lexing -- `#` is a comment only at a word start; do/done count only in command position
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("noise", [
    'n=$#; for i in 1 2; do\n    :\n  done\n',
    'n=${#arr[@]}; for i in 1 2; do\n    :\n  done\n',
    'echo "${x#*/}" ${y##*.}; for i in 1 2; do\n    :\n  done\n',
    'case "$h" in done) : ;; esac\n',
    'x="$d/done"; y=$d/done; echo done\n',
    'cat <<EOF\nwhile read -r z; do\ndone\nEOF\n',
    "cat <<-'EOF'\n\tdone\n\tEOF\n",
])
def test_lexical_noise_does_not_close_the_enclosing_read_loop(noise):
    text = _loop("  " + noise + "  ssh \"$h\" true\n")
    assert len(_flags(text)) == 1, (noise, _flags(text))


def test_every_tracked_script_parses_with_no_unmatched_tokens():
    """The six files the review found with unbalanced do/done under the old masking all parse cleanly now."""
    texts = sr_gate._worktree_texts(_ROOT)
    assert len(texts) > 100
    bad = {rel: s.strays for rel, s in ((r, sr_gate._summarize(t)) for r, t in texts.items()) if s.strays}
    assert bad == {}
    for rel in ("tools/pool_provision.sh", "tools/pool_sync.sh", "tools/workflow_check.sh",
                "tools/pool_opsweep_dispatch.sh"):
        if rel in texts:
            assert sr_gate._summarize(texts[rel]).strays == 0


# ---------------------------------------------------------------------------------------------------------------
# MEDIUM: function and call shapes
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("defn", [
    'probe() { ssh "$1" true; }\n',
    'function probe {\n  ssh "$1" true\n}\n',
    'function probe() {\n  ssh "$1" true\n}\n',
    'probe()\n{\n  ssh "$1" true\n}\n',
    'probe() (\n  ssh "$1" true\n)\n',
])
@pytest.mark.parametrize("call", [
    'probe "$h"',
    'r=$(probe "$h")',
    'echo "$(probe "$h")"',
    'probe "$h" | cat',
    'probe "$h" && echo ok',
    'probe "$h"&&echo ok',
    'if probe "$h"; then :; fi',
])
def test_every_function_definition_and_call_shape_is_followed(defn, call):
    probs = _flags("#!/usr/bin/env bash\n" + defn + "while IFS= read -r h; do\n  %s\ndone < q\n" % call)
    assert len(probs) == 1 and "ssh" in probs[0][2], probs


def test_a_command_substitution_inside_double_quotes_is_walked():
    assert [p[0] for p in _flags(_loop('  x="$(ssh "$h" uptime)"\n'))] == [3]
    assert [p[0] for p in _flags(_loop('  x="${y:-$(ssh "$h" uptime)}"\n'))] == [3]
    assert [p[0] for p in _flags(_loop('  x=`ssh "$h" uptime`\n'))] == [3]


# ---------------------------------------------------------------------------------------------------------------
# MEDIUM: correct fixes must not be blocked
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "while IFS= read -r -u 3 h; do\n  ssh \"$h\" true\ndone 3< q\n",
    "while IFS= read -ru3 h; do\n  ssh \"$h\" true\ndone 3< q\n",
    "while IFS= read -r h <&3; do\n  ssh \"$h\" true\ndone 3< q\n",
    "while IFS= read -r h < /tmp/one; do\n  ssh \"$h\" true\ndone\n",
    _loop('  ssh -nT "$h" true\n'),
    _loop('  ssh -f "$h" "sleep 5"\n'),
    _loop('  ssh -fN -L 1:x:2 "$h"\n'),
    _loop('  ssh -o StdinNull=yes "$h" true\n'),
    _loop('  ssh -oStdinNull=yes "$h" true\n'),
    _loop('  ssh "$h" -n true\n'),
    _loop('  { ssh "$h" true; ssh "$h" false; } </dev/null\n'),
    _loop('  ( ssh "$h" true ) </dev/null\n'),
    _loop('  printf x |\n    ssh "$h" "cat > f"\n'),
    _loop('  ssh "$h" true <<< "payload"\n'),
    _loop('  ssh "$h" true <&-\n'),
    _loop('  ssh "$h" true 0</dev/null\n'),
    _loop('  rsync -a -e "ssh -o BatchMode=yes" "$h:x/" y/\n'),
])
def test_correct_fix_shapes_are_not_blocked(text):
    assert _flags(text) == [], text


def test_read_u_0_still_reads_fd_0():
    assert len(_flags("while IFS= read -r -u 0 h; do\n  ssh \"$h\" true\ndone < q\n")) == 1


# ---------------------------------------------------------------------------------------------------------------
# review mutation M5: loops nested in a while-read body share its stream
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("inner", [
    "for n in 1 2; do\n    ssh \"$h\" true\n  done\n",
    "for ((i=0; i<2; i++)); do\n    ssh \"$h\" true\n  done\n",
    "while :; do\n    ssh \"$h\" true; break\n  done\n",
    "until false; do\n    ssh \"$h\" true; break\n  done\n",
    "if :; then\n    ssh \"$h\" true\n  fi\n",
    "case x in *)\n    ssh \"$h\" true ;;\n  esac\n",
])
def test_a_loop_or_branch_nested_in_a_while_read_body_is_still_in_scope(inner):
    assert len(_flags(_loop("  " + inner))) == 1, inner


def test_a_for_loop_on_its_own_is_not_a_read_loop():
    assert _flags("for h in a b; do\n  ssh \"$h\" true\ndone\n") == []
    assert _flags("while :; do\n  ssh \"$h\" true\n  sleep 1\ndone\n") == []


def test_a_loop_fed_by_a_redirect_whose_body_reads_is_a_read_loop():
    text = "while true; do\n  IFS= read -r h || break\n  ssh \"$h\" true\ndone < q\n"
    assert [p[0] for p in _flags(text)] == [3]


# ---------------------------------------------------------------------------------------------------------------
# the real corpus
# ---------------------------------------------------------------------------------------------------------------
def _unexpected(problems):
    return sorted(p.split(" -- ")[0] for p in problems if p.split(" -- ")[0] not in KNOWN_UNFIXED)


def test_the_corpus_pin_tolerates_the_known_instance_being_fixed_or_not():
    """Review r2 LOW: an exact-equality pin turns red the moment lane A's aws_idle_stop.sh fix lands."""
    fake = ["tools/aws_idle_stop.sh:100 -- x", "tools/aws_idle_stop.sh:104 -- x", "tools/aws_idle_stop.sh:116 -- x"]
    assert _unexpected(fake) == [] and _unexpected([]) == [] and _unexpected(fake[:1]) == []
    assert _unexpected(fake + ["tools/other.sh:5 -- x"]) == ["tools/other.sh:5"]


def test_the_real_corpus_has_no_instance_outside_the_known_unfixed_one():
    problems = sr_gate.check(None, root=_ROOT)
    assert _unexpected(problems) == [], problems
    assert {p.split(" -- ")[0] for p in problems} <= KNOWN_UNFIXED


def test_the_real_corpus_fixed_sites_stay_clean():
    texts = sr_gate._worktree_texts(_ROOT)
    summaries = {r: sr_gate._summarize(t) for r, t in texts.items()}
    known = sr_gate._known_scripts(summaries)
    for rel in ("tools/pool_queue.sh", "tools/pool_autodispatch.sh", "research/coordination/b2b_queue_next_wave.sh"):
        if rel in summaries:
            assert sr_gate._evaluate(summaries[rel], known)[0] == {}, rel
    if "tools/pool_queue.sh" in summaries:      # its probes all carry -n now: no longer drains its callers
        assert sr_gate._evaluate(summaries["tools/pool_queue.sh"], frozenset())[1] is False


# ---------------------------------------------------------------------------------------------------------------
# ground truth 1: bash. A stub ssh that REALLY reads stdin (honouring -n/-f before the destination, as OpenSSH
# does) runs each shape; the gate must flag exactly the shapes whose loop really lost lines.
# ---------------------------------------------------------------------------------------------------------------
_STUB = r"""#!/usr/bin/env bash
null=0; skip=0
for a in "$@"; do
  if [ "$skip" = 1 ]; then skip=0; continue; fi
  case "$a" in
    -[BbcDEeFIiJLlmOoPpQRSWw]) skip=1 ;;
    -*[nf]*) null=1 ;;
    -*) ;;
    *) break ;;
  esac
done
[ "$null" = 1 ] || cat >/dev/null
exit 0
"""

_SEMANTIC_CASES = [
    ("bare", _loop('  ssh "$h" true\n  echo "$h"\n')),
    ("-n", _loop('  ssh -n "$h" true\n  echo "$h"\n')),
    ("-nT", _loop('  ssh -nT "$h" true\n  echo "$h"\n')),
    ("-f", _loop('  ssh -f "$h" true\n  echo "$h"\n')),
    ("</dev/null", _loop('  ssh "$h" true </dev/null\n  echo "$h"\n')),
    ("group redirect", _loop('  { ssh "$h" true; } </dev/null\n  echo "$h"\n')),
    ("piped in", _loop('  echo x | ssh "$h" true\n  echo "$h"\n')),
    ("pipe then &&", _loop('  echo x | cat >/dev/null && ssh "$h" true\n  echo "$h"\n')),
    ("background &", _loop('  ssh "$h" true & wait\n  echo "$h"\n')),
    ("command substitution", _loop('  r=$(ssh "$h" true)\n  echo "$h"\n')),
    ("process substitution <()", _loop('  cat <(ssh "$h" true) >/dev/null\n  echo "$h"\n')),
    ("process substitution >()", _loop('  echo x > >(ssh "$h" true); wait\n  echo "$h"\n')),
    ("bash -c", _loop('  bash -c \'ssh x true\'\n  echo "$h"\n')),
    ("for nested", _loop('  for i in 1; do ssh "$h" true; done\n  echo "$h"\n')),
    ("function 2 levels", 'a() { ssh "$1" true; }\nb() { a "$1"; }\n' + _loop('  b "$h"\n  echo "$h"\n')),
    ("read -u 3", "while IFS= read -r -u 3 h; do\n  ssh \"$h\" true\n  echo \"$h\"\ndone 3< <(printf 'a\\nb\\nc\\n')\n"),
    ("read <&3", "while IFS= read -r h <&3; do\n  ssh \"$h\" true\n  echo \"$h\"\ndone 3< <(printf 'a\\nb\\nc\\n')\n"),
    ("remote -n", _loop('  ssh "$h" head -n 1 f\n  echo "$h"\n')),
    ("draining script via variable", 'W="$BIN/wrap.sh"\n' + _loop('  bash "$W" "$h"\n  echo "$h"\n')),
    ("draining script, </dev/null", 'W="$BIN/wrap.sh"\n' + _loop('  bash "$W" "$h" </dev/null\n  echo "$h"\n')),
]


@pytest.mark.parametrize("label,script", _SEMANTIC_CASES, ids=[c[0] for c in _SEMANTIC_CASES])
def test_the_gate_verdict_matches_what_bash_actually_does(tmp_path, label, script):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ssh").write_text(_STUB)
    (bin_dir / "ssh").chmod(0o755)
    (bin_dir / "wrap.sh").write_text("#!/usr/bin/env bash\nssh \"$1\" true\n")
    (tmp_path / "s.sh").write_text(script)
    env = dict(_clean_env(), PATH=str(bin_dir) + os.pathsep + os.environ.get("PATH", ""), BIN=str(bin_dir))
    out = subprocess.run(["bash", str(tmp_path / "s.sh")], capture_output=True, text=True, env=env, timeout=20,
                         cwd=str(tmp_path)).stdout.split()
    lost = out != ["a", "b", "c"]
    flagged = bool(sr_gate._find_unprotected(script, sr_gate._FLOOR_SCRIPTS | {"wrap.sh"}))
    assert flagged == lost, "%s: gate flagged=%s but bash lost lines=%s (saw %r)" % (label, flagged, lost, out)


def test_rsync_with_a_stdin_draining_transport_leaves_the_loop_intact(tmp_path):
    """Why rsync -e ssh is not flagged: rsync gives its transport a pipe, not the loop's stdin."""
    if shutil.which("rsync") is None:
        pytest.skip("rsync not installed")
    t = tmp_path / "t"
    # drain whatever stdin it was given, but never wait more than 1 s on a pipe rsync keeps open
    t.write_text("#!/usr/bin/env bash\ntimeout 1 cat >/dev/null\nexit 0\n")
    t.chmod(0o755)
    (tmp_path / "src").mkdir()
    script = _loop('  timeout 10 rsync -q -e "%s" src/ "host:/x/" 2>/dev/null\n  echo "$h"\n' % t)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60, cwd=str(tmp_path),
                         env=_clean_env()).stdout.split()
    assert out == ["a", "b", "c"]
    assert sr_gate._find_unprotected(script) == []


# ---------------------------------------------------------------------------------------------------------------
# ground truth 2: OpenSSH. `ssh -G` prints the resolved config (StdinNull included) and never connects.
# ---------------------------------------------------------------------------------------------------------------
_SSH_ARGVS = [
    ["h"], ["-n", "h"], ["h", "-n"], ["h", "-n", "true"], ["h", "head", "-n", "1"], ["-nT", "h"],
    ["-f", "h", "true"], ["-fN", "h"], ["-o", "StdinNull=yes", "h"], ["-oStdinNull=yes", "h"],
    ["-o", "ForkAfterAuthentication=yes", "h", "true"], ["-o", "BatchMode=yes", "h", "tail", "-n", "5"],
    ["-i", "k", "-p", "22", "h", "true"], ["h", "--", "-n"], ["-t", "h", "-n"], ["-l", "u", "-n", "h"],
]


@pytest.mark.parametrize("argv", _SSH_ARGVS, ids=[" ".join(a) for a in _SSH_ARGVS])
def test_ssh_option_parsing_matches_openssh(argv):
    def resolved(args):
        r = subprocess.run(["ssh", "-F", "/dev/null", "-G"] + args, capture_output=True, text=True, timeout=10,
                           env=_clean_env())
        return dict(ln.split(" ", 1) for ln in r.stdout.splitlines() if " " in ln)
    if shutil.which("ssh") is None or "stdinnull" not in resolved(["h"]):
        pytest.skip("no OpenSSH that reports stdinnull under -G (8.7+)")
    rows = resolved(argv)
    assert "stdinnull" in rows, "ssh -G rejected %r -- fix the fixture argv" % argv
    truth = rows["stdinnull"].strip() == "yes"
    an = sr_gate._Analyzer("")
    words = [sr_gate._Word(a, 1, a) for a in argv]
    assert an._ssh_null(words, False, 0) is truth, argv
