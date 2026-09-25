"""ssh_stdin_in_read_loop (class SR, BLOCK) -- an `ssh`/`rsync -e ssh` call (or a call to a script known to ssh)
sits inside a `while ... read ...; do ... done` loop body (or a function invoked from one), without `-n` or a
stdin redirect, so the call silently drains the loop's own stdin instead of the remote's.

THE FAILURE IT CLOSES (three real hits, 2026-09-25). `ssh`, run non-interactively WITHOUT `-n` (and without a
`</dev/null`/heredoc/pipe redirect of its own), still OPENS and FORWARDS its local stdin to the remote command --
even with `BatchMode=yes`. When that call sits inside a `while IFS= read -r x; do ... ssh ...; done < <(...)`
loop, the ssh call's stdin is the SAME fd the enclosing `read` is consuming from, so the very first ssh call
drains the rest of that pipe before the remote side even answers, truncating the scan to one iteration (a
partial drain can also hand the NEXT `read` a line FRAGMENT, which then gets treated as real data):
  * `tools/pool_queue.sh`'s `add` reachability/--help probes, inside a `while read ... done < file` de-dup scan
    -- queued only the first line of a job file twice in one night (fixed `e76106fd8`: every ssh in this file
    now carries `-n`).
  * `tools/pool_autodispatch.sh`'s `revision_available()`, called from `pop_job`'s
    `while IFS= read -r cand; do ... done < <(awk ...)` scan -- pool1+pool2 starved 07:35-09:59 EDT with 74
    runnable jobs queued behind one revision-pinned line (fixed `096dfdae0`: `ssh -n`).
  * `tools/aws_idle_stop.sh`'s `while IFS= read -r iid; do ... ssh ...; done <<<"$ids"` -- only the first
    instance in a multi-instance describe-instances result was ever actually checked. Lane A fixes this one;
    this gate lists it as a known, deliberately-unfixed-here instance so `check(None)` does not silently miss it.

WHAT THIS GATE ENFORCES (staged shell files only -- BLOCKING, fast, no ssh/network access of its own). For every
staged `*.sh` file, it finds each `ssh` invocation, each `rsync ... -e '...ssh...'` invocation, and each call to
a script this repo already knows wraps ssh (`_KNOWN_SSH_SCRIPTS` below -- `tools/pool_sync.sh` named explicitly
in the incident write-up, plus its siblings that share the same fire-and-forget dispatch shape) that is LEXICALLY
inside a `while`/`for`/`until` ... `read` ... `do ... done` loop body -- or inside a function this file defines
that is CALLED (directly or transitively) from inside such a loop body -- and flags it unless the SAME statement
carries `-n`, a stdin redirect (`</dev/null`, `<<`, `<<<`, `< file`), or is fed by an explicit pipe (`... | ssh
...`, whose stdin is the pipe, never the enclosing loop's).

THE LEXICAL CASE, and what it CANNOT see (stated, not hidden):
  * Loop/function boundaries are found with a single-pass `while|for|until|do|done` keyword scan and a
    `name() { ... }` / `function name { ... }` brace-balance-free scan (a function's body ends at the next line
    that is JUST `}`, this codebase's own dominant style, checked against every real `tools/*.sh` function
    definition this gate's own corpus audit sees) -- not a real shell parser. Both are run against a MASKED copy
    of the file where single/double-quoted string contents, `#` comments, and heredoc bodies are blanked out
    first (so a REMOTE command string that itself contains the literal text "while ... read ... do ... done" --
    a real corpus shape, e.g. `tools/pool_provision.sh`'s cleanup payload -- never opens a spurious LOCAL loop
    frame). The double-quote scanner respects backslash-escaping of the closing quote but does not track
    `$( ... )` command-substitution boundaries re-opening their own quote context; a double-quoted string whose
    embedded `$(...)` carries a differently-quoted, unescaped double-quote of its own could mis-close early. Not
    observed in this repo's corpus (checked); documented as the concrete blind spot rather than silently trusted.
  * Function-call detection is a bare-word match (`(?<![\w$/.-])name(?=[ \t(;]|$)`) -- it will not see a call made
    through a variable (`$FN "$x"`), `eval`, or a function invoked only via a job dispatched by name string. It
    also will not see a call across FILES (a function defined in one script and sourced+called by another) --
    each file is scanned independently. Under-inclusive, not over-inclusive: a miss here means an in-scope ssh
    call is NOT flagged, never that a clean one is.
  * The "known ssh-wrapping script" list is a fixed registry (`_KNOWN_SSH_SCRIPTS`), not a transitive analysis of
    what a script actually does -- a NEW script that wraps ssh will not be recognised until added here.
  * `rsync`'s own stdin-forwarding behaviour to its `-e` transport is less certain than plain ssh's (rsync
    typically manages its own pipes to the transport child rather than leaving the original stdin attached), but
    the incident class this gate exists for is specifically about `ssh`'s documented behaviour, and the task that
    created this gate named `rsync -e ssh` as in-scope -- so it is flagged with the SAME criteria, conservatively
    (a false positive here costs one `-n`/`</dev/null` that was already harmless; a false negative repeats the
    incident).
"""
from __future__ import annotations

import os
import re
import subprocess

NAME = "ssh-stdin-in-read-loop"
CLASS_ID = "SR"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GIT_TIMEOUT = 30

# Scripts this repo already knows wrap ssh/rsync internally (so calling them, unprotected, from inside a
# while-read loop reproduces the identical incident -- this is exactly what happened before
# research/coordination/b2b_queue_next_wave.sh added its own `</dev/null` guard around `pool_queue.sh add`).
_KNOWN_SSH_SCRIPTS = (
    "pool_sync.sh", "pool_queue.sh", "pool_autodispatch.sh", "aws_pool_node.sh",
    "pool_provision.sh", "aws_provision.sh", "aws_cpu_provision.sh", "pool_sync_assets.sh",
    "pool_backfill_provisioned_markers.sh",
)

_KEYWORD_RE = re.compile(r"\b(while|for|until|do|done|read)\b")
_FUNC_DEF_RE = re.compile(r"^[ \t]*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(\)\s*\{?[ \t]*$", re.M)
_BRACE_OPEN_ONLY_RE = re.compile(r"^[ \t]*\{[ \t]*$", re.M)
_BRACE_CLOSE_RE = re.compile(r"^[ \t]*\}[ \t]*$", re.M)
_HEREDOC_RE = re.compile(r"<<-?\s*([\"']?)(\w+)\1")
_SSH_CMD_RE = re.compile(
    # No leading '/' in the exclusion: a script is normally invoked path-qualified (`tools/pool_sync.sh`,
    # `/usr/bin/ssh`), so "immediately after a path separator" must still count as a real command-word start.
    r"(?<![\w.-])(?:ssh|rsync|" + "|".join(re.escape(s) for s in _KNOWN_SSH_SCRIPTS) + r")(?![\w.-])"
)
_PROTECT_N_RE = re.compile(r"(?<!\S)-n(?!\S)")
_PROTECT_REDIR_RE = re.compile(r"</dev/null|<<<|<<|(?<!\S)<\s*[^\s(]")


# --- masking: blank heredoc bodies, quoted-string contents, and comments (positions/newlines preserved) -------
def _mask(text):
    n = len(text)
    out = list(text)
    for m in _HEREDOC_RE.finditer(text):
        term = m.group(2)
        body_start = text.find("\n", m.end())
        if body_start < 0:
            continue
        body_start += 1
        term_re = re.compile(r"^[ \t]*" + re.escape(term) + r"[ \t]*$", re.M)
        tm = term_re.search(text, body_start)
        end = tm.start() if tm else n
        for j in range(body_start, min(end, n)):
            if out[j] != "\n":
                out[j] = " "
    text2 = "".join(out)
    out = list(text2)
    i = 0
    while i < n:
        c = text2[i]
        if c == "#":
            j = text2.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if c == "'":
            j = text2.find("'", i + 1)
            j = n if j < 0 else j + 1
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
            continue
        if c == '"':
            j = i + 1
            while j < n:
                if text2[j] == "\\":
                    j += 2
                    continue
                if text2[j] == '"':
                    j += 1
                    break
                j += 1
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
            continue
        i += 1
    return "".join(out)


# --- loop-frame detection: single pass over while|for|until|do|done|read keyword events ------------------------
def _read_loop_frames(masked_text):
    """[(start, end, is_read)] for every do..done block, start/end = the 'do'/'done' keyword positions. A
    frame's is_read is True iff its OWN opener's condition contained `read`, OR its immediate parent frame is
    is_read (so a plain `for`/`until` nested inside a `while ... read ...` loop still counts -- the nested body
    still runs on the SAME inherited stdin unless its own commands redirect it)."""
    frames = []
    stack = []  # (start_pos, is_read)
    pending_active = False
    pending_is_read = False
    for m in _KEYWORD_RE.finditer(masked_text):
        pos, kw = m.start(), m.group(1)
        if kw in ("while", "for", "until"):
            pending_active = True
            pending_is_read = False
        elif kw == "read":
            if pending_active:
                pending_is_read = True
        elif kw == "do":
            parent_read = stack[-1][1] if stack else False
            stack.append((pos, pending_is_read or parent_read))
            pending_active = False
            pending_is_read = False
        elif kw == "done":
            if stack:
                start_pos, is_read = stack.pop()
                frames.append((start_pos, pos, is_read))
    while stack:  # malformed/truncated (e.g. a selftest fixture) -- close at EOF rather than lose the frame
        start_pos, is_read = stack.pop()
        frames.append((start_pos, len(masked_text), is_read))
    return frames


def _is_read_at(pos, frames):
    return any(s <= pos < e for s, e, r in frames if r)


# --- function bodies + who calls whom (for the "or a function called from one" half of the contract) -----------
def _find_functions(masked_text):
    funcs = {}
    for m in _FUNC_DEF_RE.finditer(masked_text):
        name = m.group(1)
        pos_after = m.end()
        if m.group(0).rstrip().endswith("{"):
            body_start = pos_after
        else:
            om = _BRACE_OPEN_ONLY_RE.search(masked_text, pos_after)
            if not om or masked_text[pos_after:om.start()].strip(" \t\n"):
                continue
            body_start = om.end()
        cm = _BRACE_CLOSE_RE.search(masked_text, body_start)
        funcs[name] = (body_start, cm.start() if cm else len(masked_text))
    return funcs


def _call_positions(masked_text, name):
    pat = re.compile(r"(?<![\w$/.-])" + re.escape(name) + r"(?=[ \t(;]|$)", re.M)
    return [m.start() for m in pat.finditer(masked_text)]


def _functions_called_from_read_loops(masked_text, funcs, frames):
    """Fixed-point over: a function is 'in scope' if it is called from a read-loop frame, OR called from the
    body of an already-in-scope function (handles the real 2-level shape: pop_job's while-read loop calls
    revision_available_cached, which calls revision_available)."""
    calls = {name: _call_positions(masked_text, name) for name in funcs}
    marked = set()
    changed = True
    while changed:
        changed = False
        for name, (fstart, fend) in funcs.items():
            if name in marked:
                continue
            for pos in calls[name]:
                if _is_read_at(pos, frames) or any(
                    other in marked and funcs[other][0] <= pos < funcs[other][1] for other in funcs
                ):
                    marked.add(name)
                    changed = True
                    break
    return marked


# --- per-occurrence protection check ----------------------------------------------------------------------
def _statement_window(text, start, max_len=400):
    """The text from an ssh/rsync/known-script token to the end of its OWN statement -- stops at the first
    unescaped ';' or a bare newline not continued by a trailing '\\' / '|' / '&' on the previous line, so a
    multi-line `ssh \\\n  -n ... \\\n  "$node" ...` call is read as one statement (the real corpus shape)."""
    end = min(len(text), start + max_len)
    i = start
    while i < end:
        c = text[i]
        if c == ";":
            return text[start:i + 1]
        if c == "\n":
            j = i - 1
            while j > start and text[j] in " \t":
                j -= 1
            if j <= start or text[j] not in "\\|&":
                return text[start:i + 1]
        i += 1
    return text[start:end]


def _piped_in(masked_text, start):
    """True iff something on the SAME (masked -- so quoted remote-command text is invisible) statement pipes
    INTO this call, e.g. `printf '%s\\n' "$X" | ssh "$h" ...` -- ssh's stdin is then the pipe, never the
    enclosing loop's, so `-n` is not needed."""
    j = start - 1
    while j >= 0 and masked_text[j] not in ";\n":
        if masked_text[j] == "|":
            if j > 0 and masked_text[j - 1] == "|":
                j -= 2
                continue
            return True
        j -= 1
    return False


def _find_unprotected(text):
    masked = _mask(text)
    frames = _read_loop_frames(masked)
    funcs = _find_functions(masked)
    in_scope_funcs = _functions_called_from_read_loops(masked, funcs, frames)
    problems = []
    for m in _SSH_CMD_RE.finditer(masked):
        pos = m.start()
        scoped = _is_read_at(pos, frames) or any(
            name in in_scope_funcs and funcs[name][0] <= pos < funcs[name][1] for name in funcs
        )
        if not scoped:
            continue
        window = _statement_window(text, pos)
        if _PROTECT_N_RE.search(window) or _PROTECT_REDIR_RE.search(window) or _piped_in(masked, pos):
            continue
        lineno = text.count("\n", 0, pos) + 1
        snippet = window.split("\n")[0].strip()
        problems.append((lineno, m.group(0), snippet[:110]))
    return problems


# --- staged-file plumbing --------------------------------------------------------------------------------------
def _git_env():
    env = dict(os.environ)
    for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
              "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES", "GIT_PREFIX"):
        env.pop(k, None)
    return env


def _all_tracked_sh(root):
    try:
        r = subprocess.run(["git", "ls-files", "*.sh"], cwd=root, env=_git_env(),
                            capture_output=True, text=True, timeout=_GIT_TIMEOUT)
        if r.returncode == 0:
            names = [ln.strip() for ln in r.stdout.split("\n") if ln.strip()]
            if names and any(os.path.exists(os.path.join(root, n)) for n in names[:20]):
                return names
    except (OSError, subprocess.SubprocessError):
        pass
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules", ".venv")]
        for fn in filenames:
            if fn.endswith(".sh"):
                out.append(os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/"))
    return out


def check(paths, root=_ROOT):
    if paths is not None and len(paths) == 0:
        return []
    cand = _all_tracked_sh(root) if paths is None else [p for p in paths if p.replace(os.sep, "/").endswith(".sh")]
    problems = []
    for rel in cand:
        full = rel if os.path.isabs(rel) else os.path.join(root, rel)
        try:
            with open(full, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        display = rel if not os.path.isabs(rel) else os.path.relpath(full, root)
        for lineno, tok, snippet in _find_unprotected(text):
            problems.append(
                "%s:%d -- `%s` runs inside a while-read loop (or a function called from one) without `-n` or a "
                "stdin redirect (</dev/null, <<, <<<, < file, or an upstream pipe); it will drain the loop's own "
                "stdin and silently truncate the scan (the 2026-09-25 pool_queue.sh / pool_autodispatch.sh "
                "incidents). Add `-n` (ssh) or `</dev/null`. Near: %s" % (display.replace(os.sep, "/"), lineno, tok, snippet)
            )
    return problems


# --- selftest ---------------------------------------------------------------------------------------------
def selftest():
    """FAILING DIRECTION FIRST: the exact incident shape must be caught; every real safe shape in this repo's
    corpus (for-loops, `while true`, already-`-n`'d calls, piped-in calls, quoted remote-command text that
    itself contains the literal words while/read/do/done) must stay silent."""
    bad = []

    # --- the failing direction: the real incident shape -----------------------------------------------------
    unprotected = (
        "#!/usr/bin/env bash\n"
        "while IFS= read -r cand; do\n"
        "  timeout 10 ssh \"${SSH_F[@]}\" -o BatchMode=yes \"$cand\" true 2>/dev/null\n"
        "done < <(awk '{print $2}' \"$QUEUE\")\n"
    )
    probs = _find_unprotected(unprotected)
    if not probs:
        bad.append("MISSED: a bare `ssh` inside a `while IFS= read; do ... done < <(...)` loop, no -n, was not caught")
    elif probs[0][0] != 3:
        bad.append("wrong line number for the missed ssh call: %r" % (probs,))

    # --- protected with -n: must stay silent -------------------------------------------------------------
    protected_n = unprotected.replace("timeout 10 ssh ", "timeout 10 ssh -n ")
    if _find_unprotected(protected_n):
        bad.append("FALSE POSITIVE: an `ssh -n` inside the same while-read loop was flagged")

    # --- protected with </dev/null: must stay silent ---------------------------------------------------------
    protected_redir = unprotected.replace('"$cand\" true 2>/dev/null', '"$cand" true </dev/null 2>/dev/null')
    if _find_unprotected(protected_redir):
        bad.append("FALSE POSITIVE: an ssh call with a trailing </dev/null was flagged")

    # --- piped-in stdin: must stay silent (ssh's stdin is the pipe, not the loop's) ---------------------------
    piped = (
        "#!/usr/bin/env bash\n"
        "while IFS= read -r cand; do\n"
        "  printf '%s\\n' \"$RUNCELL\" | ssh \"$cand\" \"cat > run_cell.sh\"\n"
        "done < <(printf 'a\\nb\\n')\n"
    )
    if _find_unprotected(piped):
        bad.append("FALSE POSITIVE: an ssh call fed by an explicit pipe was flagged")

    # --- multi-line ssh call: -n on a CONTINUED line must still be seen -------------------------------------
    multiline_n = (
        "#!/usr/bin/env bash\n"
        "while IFS= read -r cand; do\n"
        "  timeout 10 ssh -n \"${SSH_F[@]}\" \\\n"
        "    -o BatchMode=yes \"$cand\" true 2>/dev/null\n"
        "done < <(awk '{print $2}' \"$QUEUE\")\n"
    )
    if _find_unprotected(multiline_n):
        bad.append("FALSE POSITIVE: a multi-line ssh call whose -n is present was flagged")

    # --- for-loops are NOT read loops: an unprotected ssh in a for-loop must stay silent ---------------------
    for_loop = (
        "#!/usr/bin/env bash\n"
        "for h in \"${NODES[@]}\"; do\n"
        "  ssh \"$h\" \"echo hi\"\n"
        "done\n"
    )
    if _find_unprotected(for_loop):
        bad.append("FALSE POSITIVE: an unprotected ssh in a plain for-loop (no read) was flagged")

    # --- `while true; do` / `while cond; do` (no read) is NOT a read loop -------------------------------------
    while_true = (
        "#!/usr/bin/env bash\n"
        "while :; do\n"
        "  ssh \"$h\" true\n"
        "  sleep 5\n"
        "done\n"
    )
    if _find_unprotected(while_true):
        bad.append("FALSE POSITIVE: an unprotected ssh in a `while :; do` (condition loop, no read) was flagged")

    # --- a function CALLED from a while-read loop: the ssh inside it must be caught (2-level indirection) -----
    indirect = (
        "#!/usr/bin/env bash\n"
        "revision_available() {\n"
        "  local node=\"$1\" sha=\"$2\"\n"
        "  timeout 10 ssh \"${SSH_F[@]}\" -o BatchMode=yes \"$node\" \"test -f /revisions/$sha/.ok\"\n"
        "}\n"
        "revision_available_cached() {\n"
        "  revision_available \"$1\" \"$2\"\n"
        "}\n"
        "pop_job() {\n"
        "  while IFS= read -r cand; do\n"
        "    revision_available_cached \"$1\" \"$cand\"\n"
        "  done < <(awk '{print $2}' \"$QUEUE\")\n"
        "}\n"
    )
    probs = _find_unprotected(indirect)
    if not probs:
        bad.append("MISSED: an ssh call inside a function TWO levels of indirection from a while-read loop's call site was not caught")
    elif probs[0][0] != 4:
        bad.append("wrong line for the indirect-function ssh call: %r" % (probs,))

    # --- a function defined but NEVER called from a read loop: must stay silent ------------------------------
    not_called = (
        "#!/usr/bin/env bash\n"
        "unused_probe() {\n"
        "  ssh \"$1\" true\n"
        "}\n"
        "for h in \"${NODES[@]}\"; do\n"
        "  echo \"$h\"\n"
        "done\n"
    )
    if _find_unprotected(not_called):
        bad.append("FALSE POSITIVE: a function never called from any read loop was flagged")

    # --- ssh mentioned only in a comment inside a read loop: must stay silent --------------------------------
    comment_only = (
        "#!/usr/bin/env bash\n"
        "while IFS= read -r cand; do\n"
        "  # TODO: maybe ssh \"$cand\" here later\n"
        "  echo \"$cand\"\n"
        "done < <(printf 'a\\n')\n"
    )
    if _find_unprotected(comment_only):
        bad.append("FALSE POSITIVE: 'ssh' appearing only inside a # comment was flagged")

    # --- a quoted REMOTE command string containing the literal text while/read/do/done must not open a
    # spurious local loop frame that then wrongly flags a LATER, genuinely top-level, unprotected ssh call
    # (the real tools/pool_provision.sh cleanup-payload shape) -----------------------------------------------
    quoted_remote_text = (
        "#!/usr/bin/env bash\n"
        "for h in \"${NODES[@]}\"; do\n"
        "  SSH_CMD \"$h\" \"find . | while IFS= read -r p; do rm -f -- \\\"\\$p\\\"; done\"\n"
        "done\n"
        "ssh \"$onenode\" true\n"
    )
    probs = _find_unprotected(quoted_remote_text)
    if any(p[0] == 5 for p in probs):
        bad.append("FALSE POSITIVE: a top-level ssh call was flagged because a QUOTED remote command string "
                   "containing literal while/read/do/done text leaked into the local loop-frame scan: %r" % (probs,))

    # --- rsync -e ssh, unprotected, inside a while-read loop: must be caught ---------------------------------
    rsync_case = (
        "#!/usr/bin/env bash\n"
        "while IFS= read -r n; do\n"
        "  timeout 30 rsync -q -e \"ssh -o BatchMode=yes\" \"$n:out/\" \"dest/\"\n"
        "done < <(printf 'pool40\\n')\n"
    )
    probs = _find_unprotected(rsync_case)
    if not probs or probs[0][1] != "rsync":
        bad.append("MISSED: an unprotected `rsync -e ssh` inside a while-read loop was not caught: %r" % (probs,))

    # --- a KNOWN ssh-wrapping script called unprotected inside a while-read loop: must be caught -------------
    known_script = (
        "#!/usr/bin/env bash\n"
        "while IFS= read -r line; do\n"
        "  bash tools/pool_sync.sh\n"
        "done < <(printf 'a\\n')\n"
    )
    probs = _find_unprotected(known_script)
    if not probs:
        bad.append("MISSED: an unprotected call to a known ssh-wrapping script (pool_sync.sh) inside a "
                   "while-read loop was not caught")

    # --- the SAME known-script call, protected with </dev/null (the real b2b_queue_next_wave.sh fix shape) ---
    known_script_protected = known_script.replace("bash tools/pool_sync.sh\n", "bash tools/pool_sync.sh </dev/null\n")
    if _find_unprotected(known_script_protected):
        bad.append("FALSE POSITIVE: a known-script call with a trailing </dev/null was flagged")

    return bad


if __name__ == "__main__":
    hits = check(None)
    print("CLASS SR corpus audit: %d unprotected ssh/rsync-in-read-loop site(s)" % len(hits))
    for h in hits:
        print("  ⛔", h)
