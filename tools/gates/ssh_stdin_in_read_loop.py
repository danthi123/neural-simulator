r"""ssh_stdin_in_read_loop (class SR, BLOCK) -- an `ssh` call, or a call to a repo script that itself runs ssh on
its own stdin, whose stdin is the stream an enclosing `while ... read ...` loop is iterating. ssh without `-n` opens
and forwards its local stdin to the remote side even under BatchMode=yes, so the FIRST such call drains the rest of
the loop's input (a partial drain can also hand the next `read` a line FRAGMENT that is then treated as data).

THE INCIDENTS (2026-09-24/25). Each is replayed from git history by tests/test_gate_ssh_stdin_in_read_loop.py.
  1. research/coordination/b2b_queue_next_wave.sh (added 8cc766c48, fixed 6406924ee): `bash "$QUEUE_TOOL" add
     "$line" ...` inside `while IFS= read -r line; do ... done < "$JOBS"`; tools/pool_queue.sh's reachability
     probes ran `ssh` without -n and swallowed the job file (1 of N lines queued, twice).
  2. tools/pool_autodispatch.sh (MODIFIED by 9f7d4095b, fixed 096dfdae0): pop_job's `while IFS= read -r cand ...
     done < <(awk ...)` calls revision_available_cached -> revision_available -> `ssh` without -n; pool1+pool2
     starved 07:35-09:59 EDT with 74 runnable jobs queued.
  3. tools/aws_idle_stop.sh: three ssh probes in `while IFS= read -r iid; do ... done <<<"$ids"`, so only the first
     instance is ever checked (fixed on research/aws-pool-stop-start-safety; the corpus test tolerates either state).

HOW IT DECIDES (a parser, not a regex over masked text -- the previous lexical version missed incident 1 and
blocked the repo's own `<&3` fix shape). This module tokenizes each script the way bash does (quotes, `\`
escapes, `$( )`/backticks/`<( )` parsed recursively, `${ }`, `$((`, heredoc bodies skipped, `#` a comment only at
a word start, so `$#` and `${#a[@]}` are not comments) and parses it into lists, pipelines,
simple commands and compound commands (reserved words only in command position, so `done)` as a case pattern or
`$d/done` never closes a loop). It then walks the tree carrying the STATE of fd 0:
  * `loop`  -- fd 0 is the stream a `while`/`until` loop iterates: its condition runs `read` on fd 0 (a
    `read -u N` with N != 0, or a `read ... <&N`/`< file`, reads another fd and does NOT count), or its body does
    while the loop is fed by a redirect or a pipe. Everything nested inside inherits it (a `for` inside a
    while-read body still shares the stream).
  * `inherit` -- the script's (or the calling function's) own stdin.
  * `safe` -- fd 0 was redirected (`</dev/null`, `<&N`, `< f`, `<<`, `<<<`, `<&-`) on the command or on an
    enclosing compound (`{ ...; } </dev/null`), or the command is a later pipeline stage (`x | ssh ...`, also
    across a trailing `|` line continuation), or a `>( )` process substitution.
Measured, not assumed (bash 5, 2026-09-25): `cmd &` in a script does NOT get /dev/null as stdin (a backgrounded
drainer still ate the loop), `$(...)`, `<(...)` and `bash -c` inherit fd 0, `>(...)` and later pipe stages do not.
So `&` is not protection, and `x=$(ssh ...)`, `"$(probe)"` and `r=$(f)` are walked with the enclosing state.

An `ssh` is protected by its OWN options only: `-n` or `-f` (which implies -n) in any flag cluster (`-nT`, `-fN`),
or `-o StdinNull=yes` (not `-o ForkAfterAuthentication=yes`: unlike the `-f` flag it leaves StdinNull off), before
the host or in the option run straight after it (OpenSSH re-parses options once after the destination). Every one
of these rules is checked against OpenSSH itself by test_ssh_option_parsing_matches_openssh (`ssh -G` prints the
resolved StdinNull without connecting). A `-n` in the REMOTE command (`ssh "$h" head -n 1 f`) or after a `|`/`&&`
(`ssh "$h" cat f | sort -n`) is not ssh's. Commands are
resolved through wrappers (timeout, nohup, env, nice, stdbuf, setsid, flock, sudo, command, exec, ...), shell
functions (the fixed point handles any depth of calls; `function f {`, `f() { ...; }` one-liners and
`f() ( ... )` are all definitions), and variables assigned in the same file (`SSH="ssh -o ..."; $SSH ...`,
`QUEUE_TOOL="$ROOT/tools/pool_queue.sh"; bash "$QUEUE_TOOL" ...`, option arrays such as `SSH_F=(-F "$c")`).

WHICH SCRIPTS COUNT AS SSH WRAPPERS. Derived from the corpus on every run: a `*.sh` file drains its own stdin if
an unprotected ssh (or a call to another draining script) is reachable with state `inherit` from its top level,
closed under a fixed point. `_FLOOR_SCRIPTS` (the nine pool/AWS dispatch tools) stay in the set even when their
current version is clean, so a caller cannot lean on a callee's `-n` that a later edit may drop. In staged mode the
corpus is read from the INDEX (`git ls-files -s` + `git cat-file --batch`), so the gate judges what is committed.

WHAT IT SCANS. Hook mode (`check(paths)`): every staged `*.sh` that is Added, Copied, Modified or Renamed (the
hook only passes ADDED files, so the gate asks git itself, like compute_idle_persistent and discriminating_power),
plus any unstaged file that calls a staged script which now drains stdin from a read loop. Audit mode
(`check(None)` / `python -m tools.gates.ssh_stdin_in_read_loop`): every tracked `*.sh` in the working tree.

WHAT IT CANNOT SEE. Misses (a hazard goes unflagged): functions sourced from another file; commands run through
`eval`, `"$@"` or an unresolved variable; heredoc bodies (a `$(ssh ...)` in an unquoted heredoc is expanded but not
analysed); extensionless shell files; stdin drainers other than ssh (`cat`, `docker exec -i`, ...). Over-reach (a
correct call is flagged; none occurs in this corpus or its history): a function that forwards "$@" to ssh is judged
by its own body, so `f -n host` still counts -- put the -n inside the function; an earlier `exec </dev/null` is not
tracked; a floor script is treated as draining even while its current text is clean. `rsync -e ssh` is not flagged:
rsync connects its transport's stdin to its own pipe, and a stdin-draining transport stub left the loop intact
(test_rsync_with_a_stdin_draining_transport_leaves_the_loop_intact).
"""
from __future__ import annotations

import bisect
import hashlib
import os
import re
import subprocess

NAME = "ssh-stdin-in-read-loop"
CLASS_ID = "SR"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_GIT_TIMEOUT = 60

# Dispatch tools that ssh to pool/AWS nodes. Kept in the known set even when their current text is clean (see the
# module docstring); the corpus-derived set is added to this on every run.
_FLOOR_SCRIPTS = frozenset((
    "pool_sync.sh", "pool_queue.sh", "pool_autodispatch.sh", "aws_pool_node.sh",
    "pool_provision.sh", "aws_provision.sh", "aws_cpu_provision.sh", "pool_sync_assets.sh",
    "pool_backfill_provisioned_markers.sh",
))

_REDIR_RE = re.compile(r"(\d+|\{[A-Za-z_][A-Za-z0-9_]*\})?(<<<|<<-|<<|<&|<>|>>|>&|>\||<|>)")
_ASSIGN_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)(\[[^\]]*\])?(\+?)=")
_ARRAY_OPEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\[[^\]]*\])?\+?=")
_VARREF_RE = re.compile(r"\$(?:([A-Za-z_][A-Za-z0-9_]*)|\{([A-Za-z_][A-Za-z0-9_]*)(\[[@*]\])?\})")
_SSH_STDIN_NULL_OPT = re.compile(r"(?i)^\s*stdinnull\s*(=\s*|\s+)(yes|true)\s*$")
_SSH_VAR_RE = re.compile(r"(?i)(?:^|_)ssh(?:_?cmd|_?bin)?$")
_SSH_ARG_OPTS = frozenset("BbcDEeFIiJLlmOoPpQRSWw")
_READ_ARG_OPTS = frozenset("adinNptu")
_IN_OPS = frozenset(("<", "<<", "<<-", "<<<", "<&", "<>"))
_CLOSERS = frozenset(("done", "fi", "esac", "}", "then", "do", "else", "elif"))
_CASE_ENDS = frozenset((";;", ";&", ";;&"))
_SHELLS = frozenset(("bash", "sh", "dash", "zsh", "ksh"))
# wrapper -> (options that take a separate argument, positional words before the wrapped command)
_WRAPPERS = {
    "timeout": (frozenset(("-s", "-k", "--signal", "--kill-after")), 1),
    "nice": (frozenset(("-n", "--adjustment")), 0),
    "ionice": (frozenset(("-c", "-n", "-p", "-P", "-u", "--class", "--classdata")), 0),
    "stdbuf": (frozenset(("-i", "-o", "-e", "--input", "--output", "--error")), 0),
    "nohup": (frozenset(), 0),
    "setsid": (frozenset(), 0),
    "env": (frozenset(("-u", "-C", "-S", "--unset", "--chdir", "--split-string")), 0),
    "exec": (frozenset(("-a",)), 0),
    "command": (frozenset(), 0),
    "builtin": (frozenset(), 0),
    "sudo": (frozenset(("-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U", "-T", "-R")), 0),
    "flock": (frozenset(("-w", "-E", "--timeout", "--conflict-exit-code")), 1),
    "unbuffer": (frozenset(), 0),
    "time": (frozenset(("-f", "-o", "--format", "--output")), 0),
    "chronic": (frozenset(), 0),
}


# ================================================================================================================
# lexer
# ================================================================================================================
def _varref(raw):
    """(name, is_array_expansion, quoted) iff the whole word is ONE parameter expansion, else None."""
    quoted = len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"'
    inner = raw[1:-1] if quoted else raw
    m = _VARREF_RE.fullmatch(inner)
    if not m:
        return None
    return (m.group(1) or m.group(2), bool(m.group(3)), quoted)


class _Word(object):
    """One shell word. `shape` is the quote-removed text with every expansion replaced by \x00, so `lit` (the
    static value) exists only for a word with no expansion in it."""
    __slots__ = ("raw", "line", "shape", "quoted", "subs", "arr", "vref")

    def __init__(self, raw, line, shape, quoted=False, subs=None, arr=None):
        self.raw = raw
        self.line = line
        self.shape = shape
        self.quoted = quoted
        self.subs = subs if subs is not None else []   # [(kind, tokens)]: kind cmd | pin (<( )) | pout (>( ))
        self.arr = arr                                  # elements of a NAME=( ... ) compound assignment
        self.vref = _varref(raw)

    @property
    def lit(self):
        return None if "\x00" in self.shape else self.shape


class _Tok(object):
    __slots__ = ("kind", "val", "line", "fd", "word")

    def __init__(self, kind, val, line, fd=None, word=None):
        self.kind = kind        # W word, N newline, O operator, R redirection, A (( arithmetic ))
        self.val = val
        self.line = line
        self.fd = fd
        self.word = word


_EOF = _Tok("E", None, 0)


class _Lexer(object):
    def __init__(self, text, line_base=1):
        self.t = text
        self.n = len(text)
        self.line_base = line_base
        self.nl = [m.start() for m in re.finditer("\n", text)]
        self.pending = []   # heredocs waiting for the next newline: (delimiter, strip_tabs)

    def line(self, pos):
        return self.line_base + bisect.bisect_left(self.nl, pos)

    def lex(self, i, closer=False):
        """Tokens from i to EOF, or (closer=True) to the `)` that closes a `$(`/`<(`/array opened before i."""
        t, n = self.t, self.n
        toks = []
        depth = 0
        case_depth = 0
        while i < n:
            c = t[i]
            if c in " \t\r":
                i += 1
                continue
            if c == "\\" and i + 1 < n and t[i + 1] == "\n":
                i += 2
                continue
            if c == "\n":
                toks.append(_Tok("N", "\n", self.line(i)))
                i = self._skip_heredocs(i + 1)
                continue
            if c == "#":
                j = t.find("\n", i)
                i = n if j < 0 else j
                continue
            if c == ")":
                if closer:
                    if depth > 0:
                        depth -= 1
                    elif case_depth == 0:
                        return toks, i + 1
                toks.append(_Tok("O", ")", self.line(i)))
                i += 1
                continue
            if c == "(":
                if t.startswith("((", i):
                    j = self._match_paren(i)
                    toks.append(_Tok("A", t[i:j], self.line(i)))
                    i = j
                    continue
                if closer:
                    depth += 1
                toks.append(_Tok("O", "(", self.line(i)))
                i += 1
                continue
            if c == ";":
                op = (";;&" if t.startswith(";;&", i) else ";;" if t.startswith(";;", i)
                      else ";&" if t.startswith(";&", i) else ";")
                toks.append(_Tok("O", op, self.line(i)))
                i += len(op)
                continue
            if c == "|":
                op = "||" if t.startswith("||", i) else "|&" if t.startswith("|&", i) else "|"
                toks.append(_Tok("O", op, self.line(i)))
                i += len(op)
                continue
            if c == "&":
                if t.startswith("&&", i):
                    toks.append(_Tok("O", "&&", self.line(i)))
                    i += 2
                    continue
                if t.startswith("&>", i):
                    op = "&>>" if t.startswith("&>>", i) else "&>"
                    i = self._redir(toks, op, None, i + len(op), self.line(i))
                    continue
                toks.append(_Tok("O", "&", self.line(i)))
                i += 1
                continue
            if c in "<>" or c.isdigit() or c == "{":
                m = _REDIR_RE.match(t, i)
                if m and not (m.group(1) is None and m.group(2) in ("<", ">")
                              and m.end() < n and t[m.end()] == "("):
                    i = self._redir(toks, m.group(2), m.group(1), m.end(), self.line(i))
                    continue
            w, j = self._word(i)
            if j <= i:
                i += 1
                continue
            toks.append(_Tok("W", None, w.line, word=w))
            if closer and not w.quoted:
                if w.raw == "case":
                    case_depth += 1
                elif w.raw == "esac" and case_depth:
                    case_depth -= 1
            i = j
        return toks, i

    def _redir(self, toks, op, fd, i, line):
        t, n = self.t, self.n
        while i < n and t[i] in " \t":
            i += 1
        w = None
        if i < n and (t[i] not in "\n;&|()<>" or (t[i] in "<>" and i + 1 < n and t[i + 1] == "(")):
            w, i = self._word(i)
        toks.append(_Tok("R", op, line, fd=fd, word=w))
        if op in ("<<", "<<-") and w is not None:
            self.pending.append((w.shape if w.lit is not None else w.raw, op == "<<-"))
        return i

    def _skip_heredocs(self, i):
        t, n = self.t, self.n
        while self.pending:
            delim, strip = self.pending.pop(0)
            while i < n:
                j = t.find("\n", i)
                end = n if j < 0 else j
                ln = t[i:end]
                i = n if j < 0 else j + 1
                if strip:
                    ln = ln.lstrip("\t")
                if ln.rstrip(" \t\r") == delim:
                    break
        return i

    def _word(self, i):
        t, n = self.t, self.n
        start = i
        line = self.line(i)
        shape, subs = [], []
        quoted = False
        arr = None
        while i < n:
            c = t[i]
            if c in " \t\n;&|)\r":
                break
            if c == "(":
                s = "".join(shape)
                if not quoted and _ARRAY_OPEN_RE.fullmatch(s):
                    toks, j = self.lex(i + 1, closer=True)
                    arr = [tk.word for tk in toks if tk.kind == "W"]
                    for aw in arr:
                        subs.extend(aw.subs)
                    shape.append("\x00")
                    i = j
                    continue
                if s and s[-1] in "@!+*?":
                    j = self._match_paren(i)
                    shape.append(t[i:j])
                    i = j
                    continue
                break
            if c in "<>":
                if i + 1 < n and t[i + 1] == "(":
                    toks, j = self.lex(i + 2, closer=True)
                    subs.append(("pin" if c == "<" else "pout", toks))
                    shape.append("\x00")
                    i = j
                    continue
                break
            if c == "\\":
                if i + 1 < n and t[i + 1] != "\n":
                    shape.append(t[i + 1])
                    quoted = True
                i += 2
                continue
            if c == "'":
                j = t.find("'", i + 1)
                j = n if j < 0 else j
                shape.append(t[i + 1:j])
                quoted = True
                i = j + 1
                continue
            if c == '"':
                i = self._dquote(i, shape, subs)
                quoted = True
                continue
            if c == "$":
                i = self._dollar(i, shape, subs, False)
                continue
            if c == "`":
                i = self._backtick(i, shape, subs)
                continue
            shape.append(c)
            i += 1
        return _Word(t[start:min(i, n)], line, "".join(shape), quoted, subs, arr), min(i, n)

    def _dquote(self, i, shape, subs):
        t, n = self.t, self.n
        j = i + 1
        while j < n:
            c = t[j]
            if c == "\\":
                if j + 1 < n:
                    nx = t[j + 1]
                    if nx == "\n":
                        j += 2
                        continue
                    if nx in '$`"\\':
                        shape.append(nx)
                        j += 2
                        continue
                shape.append("\\")
                j += 1
                continue
            if c == '"':
                return j + 1
            if c == "$":
                j = self._dollar(j, shape, subs, True)
                continue
            if c == "`":
                j = self._backtick(j, shape, subs)
                continue
            shape.append(c)
            j += 1
        return n

    def _dollar(self, i, shape, subs, in_dq):
        t, n = self.t, self.n
        if i + 1 >= n:
            shape.append("$")
            return i + 1
        d = t[i + 1]
        if d == "(":
            if i + 2 < n and t[i + 2] == "(":
                shape.append("\x00")
                return self._match_paren(i + 1)
            toks, j = self.lex(i + 2, closer=True)
            subs.append(("cmd", toks))
            shape.append("\x00")
            return j
        if d == "{":
            shape.append("\x00")
            return self._brace(i + 1, subs, in_dq)
        if d == "'" and not in_dq:
            j = i + 2
            while j < n and t[j] != "'":
                j += 2 if t[j] == "\\" else 1
            shape.append(t[i + 2:min(j, n)])
            return j + 1
        if d == '"' and not in_dq:
            return self._dquote(i + 1, shape, subs)
        if d.isalpha() or d == "_":
            j = i + 2
            while j < n and (t[j].isalnum() or t[j] == "_"):
                j += 1
            shape.append("\x00")
            return j
        if d.isdigit() or d in "#?$!@*-":
            shape.append("\x00")
            return i + 2
        shape.append("$")
        return i + 1

    def _brace(self, i, subs, in_dq):
        t, n = self.t, self.n
        depth = 1
        j = i + 1
        scratch = []
        while j < n:
            c = t[j]
            if c == "\\":
                j += 2
                continue
            if c == "'" and not in_dq:
                k = t.find("'", j + 1)
                j = n if k < 0 else k + 1
                continue
            if c == '"':
                j = self._dquote(j, scratch, subs)
                continue
            if c == "$":
                j = self._dollar(j, scratch, subs, in_dq)
                continue
            if c == "`":
                j = self._backtick(j, scratch, subs)
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return n

    def _backtick(self, i, shape, subs):
        t, n = self.t, self.n
        j = i + 1
        buf = []
        while j < n:
            c = t[j]
            if c == "\\" and j + 1 < n and t[j + 1] in "`$\\":
                buf.append(t[j + 1])
                j += 2
                continue
            if c == "`":
                break
            buf.append(c)
            j += 1
        toks, _ = _Lexer("".join(buf), self.line(i)).lex(0)
        subs.append(("cmd", toks))
        shape.append("\x00")
        return j + 1

    def _match_paren(self, i):
        """Index just past the `)` matching the `(` at i (quotes skipped)."""
        t, n = self.t, self.n
        depth = 0
        j = i
        while j < n:
            c = t[j]
            if c == "\\":
                j += 2
                continue
            if c == "'":
                k = t.find("'", j + 1)
                j = n if k < 0 else k + 1
                continue
            if c == '"':
                j = self._dquote(j, [], [])
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    return j + 1
            j += 1
        return n


# ================================================================================================================
# parser: list = [and-or], and-or = [pipeline], pipeline = [command]; commands are dicts keyed by "t"
# ================================================================================================================
class _Parser(object):
    def __init__(self, toks):
        self.toks = toks
        self.n = len(toks)
        self.i = 0
        self.strays = 0     # tokens skipped because nothing open matched them (a diagnostic; 0 on this corpus)

    def peek(self, k=0):
        j = self.i + k
        return self.toks[j] if j < self.n else _EOF

    def at_w(self, v):
        tk = self.peek()
        return tk.kind == "W" and tk.word.raw == v

    def at_o(self, v):
        tk = self.peek()
        return tk.kind == "O" and tk.val == v

    def skip_nl(self):
        while self.i < self.n and self.toks[self.i].kind == "N":
            self.i += 1

    def program(self):
        out = []
        while self.i < self.n:
            out.extend(self.clist())
            if self.i < self.n:
                self.strays += 1
                self.i += 1     # a closer with nothing open (malformed input): skip it and carry on
        return out

    def clist(self):
        items = []
        while self.i < self.n:
            tk = self.toks[self.i]
            if tk.kind == "N" or (tk.kind == "O" and tk.val in (";", "&")):
                self.i += 1
                continue
            if tk.kind == "W" and tk.word.raw in _CLOSERS:
                break
            if tk.kind == "O" and (tk.val == ")" or tk.val in _CASE_ENDS):
                break
            start = self.i
            ao = self.andor()
            if ao:
                items.append(ao)
            if self.i == start:
                self.strays += 1
                self.i += 1     # a stray operator such as a leading `|`
        return items

    def andor(self):
        pipes = []
        p = self.pipeline()
        if p:
            pipes.append(p)
        while self.peek().kind == "O" and self.peek().val in ("&&", "||"):
            self.i += 1
            self.skip_nl()
            p = self.pipeline()
            if p:
                pipes.append(p)
        return pipes

    def pipeline(self):
        while self.peek().kind == "W" and self.peek().word.raw in ("!", "time"):
            self.i += 1
            if self.at_w("-p"):
                self.i += 1
        cmds = []
        c = self.command()
        if c is None:
            return cmds
        cmds.append(c)
        while self.peek().kind == "O" and self.peek().val in ("|", "|&"):
            self.i += 1
            self.skip_nl()
            c = self.command()
            if c is None:
                break
            cmds.append(c)
        return cmds

    def redirs(self):
        out = []
        while self.i < self.n and self.toks[self.i].kind == "R":
            tk = self.toks[self.i]
            out.append((tk.val, tk.fd, tk.word))
            self.i += 1
        return out

    def command(self):
        tk = self.peek()
        line = tk.line
        if tk.kind == "O" and tk.val == "(":
            self.i += 1
            body = self.clist()
            if self.at_o(")"):
                self.i += 1
            return {"t": "grp", "body": body, "redirs": self.redirs(), "line": line}
        if tk.kind == "A":
            self.i += 1
            return {"t": "arith", "redirs": self.redirs(), "line": line}
        if tk.kind == "W":
            r = tk.word.raw
            if r in _CLOSERS:
                return None
            if r == "{":
                self.i += 1
                body = self.clist()
                if self.at_w("}"):
                    self.i += 1
                return {"t": "grp", "body": body, "redirs": self.redirs(), "line": line}
            if r in ("while", "until"):
                self.i += 1
                cond = self.clist()
                if self.at_w("do"):
                    self.i += 1
                body = self.clist()
                if self.at_w("done"):
                    self.i += 1
                return {"t": "while", "cond": cond, "body": body, "redirs": self.redirs(), "line": line}
            if r in ("for", "select"):
                return self._for(line)
            if r == "if":
                return self._if(line)
            if r == "case":
                return self._case(line)
            if r == "[[":
                self.i += 1
                words = []
                while self.i < self.n:
                    t2 = self.toks[self.i]
                    self.i += 1
                    if t2.kind == "W" and t2.word.raw == "]]":
                        break
                    if t2.word is not None:
                        words.append(t2.word)
                return {"t": "cond", "words": words, "redirs": self.redirs(), "line": line}
            if r == "function":
                self.i += 1
                name = None
                if self.peek().kind == "W":
                    name = self.peek().word.raw
                    self.i += 1
                if self.at_o("(") and self.peek(1).kind == "O" and self.peek(1).val == ")":
                    self.i += 2
                self.skip_nl()
                return {"t": "func", "name": name, "body": self.command(), "line": line}
            if r == "coproc":
                self.i += 1
                if self.peek().kind == "W" and (self.peek(1).kind == "O" and self.peek(1).val == "("
                                                or (self.peek(1).kind == "W" and self.peek(1).word.raw == "{")):
                    self.i += 1
                return {"t": "coproc", "body": self.command(), "redirs": [], "line": line}
            nx, nx2 = self.peek(1), self.peek(2)
            if nx.kind == "O" and nx.val == "(" and nx2.kind == "O" and nx2.val == ")":
                self.i += 3
                self.skip_nl()
                return {"t": "func", "name": r, "body": self.command(), "line": line}
        if tk.kind in ("W", "R"):
            words, redirs = [], []
            while self.i < self.n:
                t2 = self.toks[self.i]
                if t2.kind == "W":
                    words.append(t2.word)
                elif t2.kind == "R":
                    redirs.append((t2.val, t2.fd, t2.word))
                else:
                    break
                self.i += 1
            return {"t": "simple", "words": words, "redirs": redirs, "line": line}
        return None

    def _for(self, line):
        self.i += 1
        words = []
        if self.peek().kind == "A":
            self.i += 1
        else:
            if self.peek().kind == "W":
                self.i += 1
            self.skip_nl()
            if self.at_w("in"):
                self.i += 1
                while self.peek().kind == "W":
                    words.append(self.peek().word)
                    self.i += 1
        while self.peek().kind == "N" or self.at_o(";"):
            self.i += 1
        body = []
        if self.at_w("do"):
            self.i += 1
            body = self.clist()
            if self.at_w("done"):
                self.i += 1
        elif self.at_w("{"):
            self.i += 1
            body = self.clist()
            if self.at_w("}"):
                self.i += 1
        return {"t": "for", "words": words, "body": body, "redirs": self.redirs(), "line": line}

    def _if(self, line):
        self.i += 1
        parts = [self.clist()]
        if self.at_w("then"):
            self.i += 1
        parts.append(self.clist())
        while self.at_w("elif"):
            self.i += 1
            parts.append(self.clist())
            if self.at_w("then"):
                self.i += 1
            parts.append(self.clist())
        if self.at_w("else"):
            self.i += 1
            parts.append(self.clist())
        if self.at_w("fi"):
            self.i += 1
        return {"t": "if", "parts": parts, "redirs": self.redirs(), "line": line}

    def _case(self, line):
        self.i += 1
        word = None
        if self.peek().kind == "W":
            word = self.peek().word
            self.i += 1
        self.skip_nl()
        if self.at_w("in"):
            self.i += 1
        bodies = []
        while True:
            self.skip_nl()
            tk = self.peek()
            if tk.kind == "E" or self.at_w("esac"):
                break
            if self.at_o("("):
                self.i += 1
            while self.i < self.n:      # the pattern list: words and `|` up to the closing `)`
                t2 = self.toks[self.i]
                if t2.kind == "N":
                    break
                self.i += 1
                if t2.kind == "O" and t2.val == ")":
                    break
            bodies.append(self.clist())
            if self.peek().kind == "O" and self.peek().val in _CASE_ENDS:
                self.i += 1
            elif not self.at_w("esac"):
                break
        if self.at_w("esac"):
            self.i += 1
        return {"t": "case", "word": word, "bodies": bodies, "redirs": self.redirs(), "line": line}


# ================================================================================================================
# analysis
# ================================================================================================================
def _basename(shape):
    b = shape.rsplit("/", 1)[-1]
    return "" if "\x00" in b else b


def _fd0_redirected(redirs):
    return any(op in _IN_OPS and (fd is None or fd == "0") for op, fd, _w in redirs)


def _is_read_fd0(words, redirs):
    """True iff this simple command is a `read` that consumes fd 0."""
    if _fd0_redirected(redirs):
        return False
    k = 0
    while k < len(words) and _ASSIGN_RE.match(words[k].raw):
        k += 1
    while k < len(words) and words[k].raw in ("builtin", "command"):
        k += 1
    if k >= len(words) or words[k].raw != "read":
        return False
    j = k + 1
    while j < len(words):
        v = words[j].lit
        if v is None or len(v) < 2 or v[0] != "-" or v == "--":
            break
        m = 1
        while m < len(v):
            ch = v[m]
            if ch in _READ_ARG_OPTS:
                arg = v[m + 1:]
                if not arg:
                    arg = words[j + 1].lit if j + 1 < len(words) else None
                    j += 1
                if ch == "u":
                    return arg == "0"
                break
            m += 1
        j += 1
    return True


def _pseudo(shape, line, vref=None):
    w = _Word(shape, line, shape)
    w.vref = vref
    return w


def _split(word, line):
    """Unquoted expansion of a scalar value: field splitting on blanks."""
    pieces = [p for p in re.split(r"[ \t\n]+", word.shape) if p]
    if len(pieces) == 1 and pieces[0] == "\x00" and word.vref is not None:
        return [_pseudo("\x00", line, (word.vref[0], word.vref[1], False))]
    return [_pseudo(p, line) for p in pieces]


def _clone(word, line):
    w = _Word(word.raw, line, word.shape, word.quoted, [], word.arr)
    return w


def _expand(alt, ref, line):
    _name, is_arr_ref, quoted_ref = ref
    kind, val = alt
    if kind == "a":
        if is_arr_ref:
            return [_clone(e, line) for e in val]
        if not val:
            return []
        return [_clone(val[0], line)] if quoted_ref else _split(val[0], line)
    return [_clone(val, line)] if quoted_ref else _split(val, line)


class _Summary(object):
    __slots__ = ("top_sites", "func_sites", "lines", "strays")

    def __init__(self, top_sites, func_sites, lines, strays=0):
        self.top_sites = top_sites      # [(state, kind, line, name)]
        self.func_sites = func_sites    # {function name: [(state, kind, line, name)]}
        self.lines = lines
        self.strays = strays            # parser tokens skipped as unmatched (0 = the whole file parsed)


class _Analyzer(object):
    def __init__(self, text):
        text = text.replace("\r\n", "\n")
        self.lines = text.split("\n")
        toks, _ = _Lexer(text).lex(0)
        parser = _Parser(toks)
        self.prog = parser.program()
        self.strays = parser.strays
        self.funcs = {}
        self.vars = {}
        self._parsed = {}
        for cmd in self._iter_list(self.prog):
            if cmd["t"] == "func" and cmd["name"]:
                self.funcs.setdefault(cmd["name"], []).append(cmd["body"])
            elif cmd["t"] == "simple":
                self._collect_assigns(cmd["words"])

    def summary(self):
        top = []
        self._walk_list(self.prog, "inherit", top)
        fsites = {}
        for name, bodies in self.funcs.items():
            sites = []
            for body in bodies:
                if body is not None:
                    self._walk_cmd(body, "inherit", False, sites)
            fsites[name] = sites
        return _Summary(top, fsites, self.lines, self.strays)

    # --- traversal ----------------------------------------------------------------------------------------
    def _sub(self, toks):
        key = id(toks)
        got = self._parsed.get(key)
        if got is None:
            got = self._parsed[key] = (toks, _Parser(toks).program())
        return got[1]

    @staticmethod
    def _cmd_words(cmd):
        words = list(cmd.get("words") or ())
        words += [w for _op, _fd, w in (cmd.get("redirs") or ()) if w is not None]
        if cmd["t"] == "case" and cmd["word"] is not None:
            words.append(cmd["word"])
        return words

    def _iter_list(self, lst):
        for ao in lst:
            for pipe in ao:
                for cmd in pipe:
                    for c in self._iter_cmd(cmd):
                        yield c

    def _iter_cmd(self, cmd):
        yield cmd
        t = cmd["t"]
        lists = []
        if t == "grp":
            lists.append(cmd["body"])
        elif t == "while":
            lists += [cmd["cond"], cmd["body"]]
        elif t == "for":
            lists.append(cmd["body"])
        elif t == "if":
            lists += cmd["parts"]
        elif t == "case":
            lists += cmd["bodies"]
        elif t in ("func", "coproc") and cmd["body"] is not None:
            lists.append([[[cmd["body"]]]])
        for w in self._cmd_words(cmd):
            for _kind, toks in w.subs:
                lists.append(self._sub(toks))
        for lst in lists:
            for c in self._iter_list(lst):
                yield c

    def _collect_assigns(self, words):
        k = 0
        while k < len(words) and _ASSIGN_RE.match(words[k].raw):
            self._add_assign(words[k])
            k += 1
        if k < len(words) and words[k].raw in ("local", "declare", "typeset", "readonly", "export"):
            for w in words[k + 1:]:
                if _ASSIGN_RE.match(w.raw):
                    self._add_assign(w)

    def _add_assign(self, w):
        m = _ASSIGN_RE.match(w.raw)
        name, append = m.group(1), m.group(3) == "+"
        plen = m.end()
        if w.arr is not None:
            alt = ("a", list(w.arr))
        else:
            vraw = w.raw[plen:]
            vshape = w.shape[plen:] if w.shape[:plen] == w.raw[:plen] else w.shape
            alt = ("s", _Word(vraw, w.line, vshape, "'" in vraw or '"' in vraw))
        alts = self.vars.setdefault(name, [])
        if append and alts:
            for idx, old in enumerate(alts):
                if old[0] == "a" or alt[0] == "a":
                    ow = old[1] if old[0] == "a" else [old[1]]
                    nw = alt[1] if alt[0] == "a" else [alt[1]]
                    alts[idx] = ("a", ow + nw)
                else:
                    o, nv = old[1], alt[1]
                    alts[idx] = ("s", _Word(o.raw + nv.raw, o.line, o.shape + nv.shape, o.quoted or nv.quoted))
        elif len(alts) < 8:
            alts.append(alt)

    # --- the stdin-state walk -----------------------------------------------------------------------------
    def _walk_list(self, lst, state, sink):
        for ao in lst:
            for pipe in ao:
                for k, cmd in enumerate(pipe):
                    self._walk_cmd(cmd, state if k == 0 else "safe", k > 0, sink)

    def _walk_subs(self, words, state, sink):
        for w in words:
            for kind, toks in w.subs:
                self._walk_list(self._sub(toks), "safe" if kind == "pout" else state, sink)

    def _walk_cmd(self, cmd, state, piped_in, sink):
        t = cmd["t"]
        if t == "func":
            return      # a definition runs nothing; its body is walked once per call context (summary())
        redirs = cmd.get("redirs") or ()
        self._walk_subs([w for _op, _fd, w in redirs if w is not None], state, sink)
        redirected = _fd0_redirected(redirs)
        post = "safe" if redirected else state
        if t == "simple":
            self._walk_simple(cmd, state, post, sink)
        elif t == "grp":
            self._walk_list(cmd["body"], post, sink)
        elif t == "while":
            is_read = self._reads_fd0(cmd["cond"]) or ((piped_in or redirected) and self._reads_fd0(cmd["body"]))
            inner = "loop" if is_read else post
            self._walk_list(cmd["cond"], inner, sink)
            self._walk_list(cmd["body"], inner, sink)
        elif t == "for":
            self._walk_subs(cmd["words"], post, sink)
            self._walk_list(cmd["body"], post, sink)
        elif t == "if":
            for part in cmd["parts"]:
                self._walk_list(part, post, sink)
        elif t == "case":
            if cmd["word"] is not None:
                self._walk_subs([cmd["word"]], post, sink)
            for body in cmd["bodies"]:
                self._walk_list(body, post, sink)
        elif t == "cond":
            self._walk_subs(cmd["words"], post, sink)
        elif t == "coproc" and cmd["body"] is not None:
            self._walk_cmd(cmd["body"], "safe", True, sink)

    def _reads_fd0(self, lst):
        for ao in lst:
            for pipe in ao:
                if pipe and self._cmd_reads_fd0(pipe[0]):
                    return True
        return False

    def _cmd_reads_fd0(self, cmd):
        if _fd0_redirected(cmd.get("redirs") or ()):
            return False
        t = cmd["t"]
        if t == "simple":
            return _is_read_fd0(cmd["words"], ())
        if t == "grp":
            return self._reads_fd0(cmd["body"])
        if t == "while":
            return self._reads_fd0(cmd["cond"]) or self._reads_fd0(cmd["body"])
        if t == "for":
            return self._reads_fd0(cmd["body"])
        if t == "if":
            return any(self._reads_fd0(p) for p in cmd["parts"])
        if t == "case":
            return any(self._reads_fd0(b) for b in cmd["bodies"])
        return False

    def _walk_simple(self, cmd, pre, post, sink):
        words = cmd["words"]
        self._walk_subs(words, pre, sink)     # expansions run before the command's own redirections apply
        if post == "safe":
            return
        k = 0
        while k < len(words) and _ASSIGN_RE.match(words[k].raw):
            k += 1
        if k >= len(words):
            return
        for kind, name, payload, cw in self._resolve(words[k:], 0, True):
            if kind == "ssh":
                if not self._ssh_null(payload, False, 0):
                    sink.append((post, "ssh", cw.line, name))
            elif kind in ("script", "call", "rsync"):
                sink.append((post, kind, cw.line, name))
            elif kind == "shc" and payload is not None:
                text = payload.shape.replace("\x00", "X")
                toks, _ = _Lexer(text, payload.line).lex(0)
                self._walk_list(_Parser(toks).program(), post, sink)

    # --- command resolution -------------------------------------------------------------------------------
    def _script_name(self, w, depth=0):
        b = _basename(w.shape)
        if b.endswith(".sh"):
            return b
        ref = w.vref
        if ref is not None and depth < 4 and ref[0] in self.vars:
            for alt in self.vars[ref[0]]:
                exp = _expand(alt, ref, w.line)
                if exp:
                    s = self._script_name(exp[0], depth + 1)
                    if s:
                        return s
        return None

    def _resolve(self, argv, depth, allow_funcs):
        """[(kind, name, payload, command word)]: kind ssh | script | call | shc (a `bash -c` string)."""
        i = 0
        while i < len(argv):
            w = argv[i]
            lit = w.lit
            if lit is None:
                ref = w.vref
                if ref is not None and depth < 4 and ref[0] in self.vars:
                    out = []
                    for alt in self.vars[ref[0]]:
                        out.extend(self._resolve(_expand(alt, ref, w.line) + argv[i + 1:], depth + 1, allow_funcs))
                    if out or not _SSH_VAR_RE.search(ref[0]):
                        return out
                if ref is not None and _SSH_VAR_RE.search(ref[0]):
                    # `$SSH ...` whose value this file cannot see (`SSH="${1:?...}"`, the environment): an ssh
                    # command by name, protected only by options at the call itself
                    return [("ssh", "$" + ref[0], argv[i + 1:], w)]
                b = _basename(w.shape)
                return [("script", b, argv[i + 1:], w)] if b.endswith(".sh") else []
            base = lit.rsplit("/", 1)[-1]
            if allow_funcs and "/" not in lit and lit in self.funcs:
                return [("call", lit, argv[i + 1:], w)]
            if lit in _WRAPPERS:
                j = _skip_wrapper(argv, i)
                if j is None:
                    return []
                if lit == "command":
                    allow_funcs = False
                i = j
                continue
            if base == "ssh":
                return [("ssh", "ssh", argv[i + 1:], w)]
            if base == "rsync":
                return [("rsync", "rsync", argv[i + 1:], w)]    # recorded, never a hazard (module docstring)
            if base in _SHELLS:
                return self._resolve_shell(argv, i)
            if lit in ("source", "."):
                s = self._script_name(argv[i + 1]) if i + 1 < len(argv) else None
                return [("script", s, argv[i + 2:], argv[i + 1])] if s else []
            if base.endswith(".sh"):
                return [("script", base, argv[i + 1:], w)]
            return []
        return []

    def _resolve_shell(self, argv, i):
        j = i + 1
        while j < len(argv):
            v = argv[j].lit
            if v is None:
                break
            if v == "--":
                j += 1
                break
            if len(v) > 1 and v[0] in "-+":
                if v in ("-o", "+o", "-O", "+O", "--rcfile", "--init-file"):
                    j += 2
                    continue
                if v[0] == "-" and not v.startswith("--") and "c" in v[1:]:
                    return [("shc", None, argv[j + 1] if j + 1 < len(argv) else None, argv[i])]
                j += 1
                continue
            break
        if j < len(argv):
            s = self._script_name(argv[j])
            if s:
                return [("script", s, argv[j + 1:], argv[j])]
        return []

    def _ssh_null(self, args, host_seen, depth):
        """True iff ssh's OWN options set stdin to /dev/null (-n, -f, -o StdinNull=yes), as `ssh -G` reports."""
        i, n = 0, len(args)
        while i < n:
            w = args[i]
            v = w.lit
            if v is None:
                ref = w.vref
                if ref is not None and depth < 4 and ref[0] in self.vars:
                    return all(self._ssh_null(_expand(a, ref, w.line) + args[i + 1:], host_seen, depth + 1)
                               for a in self.vars[ref[0]])
                if host_seen:
                    return False            # the remote command starts here
                if ref is not None and ref[1]:
                    i += 1                  # an unresolved "${ARR[@]}": an option array by this corpus's convention
                    continue
                host_seen = True
                i += 1
                continue
            if v == "--":
                return False
            if len(v) > 1 and v[0] == "-":
                took_next = False
                for k in range(1, len(v)):
                    ch = v[k]
                    if ch in "nf":
                        return True
                    if ch in _SSH_ARG_OPTS:
                        arg = v[k + 1:]
                        if not arg:
                            took_next = True
                            arg = args[i + 1].lit if i + 1 < n else None
                        if ch == "o" and arg and _SSH_STDIN_NULL_OPT.match(arg):
                            return True
                        break
                i += 2 if took_next else 1
                continue
            if host_seen:
                return False
            host_seen = True
            i += 1
        return False


def _skip_wrapper(argv, i):
    """Index of the command a wrapper (timeout, env, nice, ...) runs, or None when it runs none."""
    name = argv[i].lit
    arg_opts, npos = _WRAPPERS[name]
    j = i + 1
    while j < len(argv):
        v = argv[j].lit
        if v is None:
            break
        if v == "--":
            j += 1
            break
        if name == "env" and _ASSIGN_RE.match(v):
            j += 1
            continue
        if len(v) > 1 and v[0] == "-":
            if name == "command" and ("v" in v[1:] or "V" in v[1:]):
                return None
            if name == "flock" and v in ("-c", "--command"):
                return None
            if v.split("=", 1)[0] in arg_opts and "=" not in v:
                j += 2
                continue
            j += 1
            continue
        break
    j += npos
    return j if j < len(argv) else None


# ================================================================================================================
# evaluation: sites + the set of draining scripts -> problems
# ================================================================================================================
_SUMMARY_CACHE = {}


def _summarize(text):
    key = hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()
    got = _SUMMARY_CACHE.get(key)
    if got is None:
        if len(_SUMMARY_CACHE) > 20000:
            _SUMMARY_CACHE.clear()
        got = _SUMMARY_CACHE[key] = _Analyzer(text).summary()
    return got


def _drain_witness(kind, name, known):
    return kind == "ssh" or (kind == "script" and name in known)


def _evaluate(summary, known):
    """({(line, token): (kind, via)} for every hazard, file-drains-its-own-stdin flag)."""
    drains = {}
    for f, sites in summary.func_sites.items():
        drains[f] = {(ln, nm, kd) for st, kd, ln, nm in sites if st == "inherit" and _drain_witness(kd, nm, known)}
    changed = True
    while changed:
        changed = False
        for f, sites in summary.func_sites.items():
            for st, kd, ln, nm in sites:
                if st == "inherit" and kd == "call" and nm != f and drains.get(nm):
                    new = drains[nm] - drains[f]
                    if new:
                        drains[f] |= new
                        changed = True
    problems = {}
    all_sites = list(summary.top_sites)
    for sites in summary.func_sites.values():
        all_sites.extend(sites)
    for st, kd, ln, nm in all_sites:
        if st != "loop":
            continue
        if _drain_witness(kd, nm, known):
            problems.setdefault((ln, nm), (kd, None))
        elif kd == "call":
            for wl, wn, wk in drains.get(nm, ()):
                problems.setdefault((wl, wn), (wk, (ln, nm)))
    file_drains = any(
        st == "inherit" and (_drain_witness(kd, nm, known) or (kd == "call" and drains.get(nm)))
        for st, kd, ln, nm in summary.top_sites
    )
    return problems, file_drains


def _known_scripts(summaries):
    known = set(_FLOOR_SCRIPTS)
    while True:
        new = set(_FLOOR_SCRIPTS)
        for rel, s in summaries.items():
            if s is not None and _evaluate(s, known)[1]:
                new.add(rel.rsplit("/", 1)[-1])
        if new == known:
            return known
        known = new


def _find_unprotected(text, known=None):
    """[(line, token, snippet)] for one script against a given known-script set (default: the floor)."""
    s = _summarize(text)
    probs = _evaluate(s, _FLOOR_SCRIPTS if known is None else known)[0]
    out = []
    for (line, tok), _kv in sorted(probs.items()):
        snippet = s.lines[line - 1].strip() if 0 < line <= len(s.lines) else ""
        out.append((line, tok, snippet[:110]))
    return out


def _format(rel, line, tok, kind, via, summary):
    snippet = summary.lines[line - 1].strip()[:110] if 0 < line <= len(summary.lines) else ""
    what = "`ssh` (no -n/-f)" if kind == "ssh" else "`%s` (a script that runs ssh on its own stdin)" % tok
    where = "inside a `while read` loop" if via is None else (
        "in %s(), called from the `while read` loop at line %d" % (via[1], via[0]))
    return ("%s:%d -- %s reads fd 0 %s: no `-n`, no stdin redirect (</dev/null, <&N, < file, <<<), not fed by a "
            "pipe, so it drains the loop's input and truncates the scan (the 2026-09-25 b2b_queue_next_wave.sh / "
            "pool_autodispatch.sh / aws_idle_stop.sh incidents). Fix: `ssh -n`, or `</dev/null` on the call. "
            "Near: %s" % (rel, line, what, where, snippet))


# ================================================================================================================
# corpus plumbing
# ================================================================================================================
def _git_env(root):
    env = dict(os.environ)
    if os.path.realpath(root) != os.path.realpath(_ROOT):
        # a fixture repo (tests): never let a hook's GIT_DIR/GIT_INDEX_FILE redirect git to the real repository
        for k in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                  "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES", "GIT_PREFIX"):
            env.pop(k, None)
    return env


def _git(root, args, data=None):
    try:
        r = subprocess.run(["git"] + args, cwd=root, env=_git_env(root), input=data, capture_output=True,
                           timeout=_GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _cat_blobs(root, shas):
    uniq = list(dict.fromkeys(shas))
    if not uniq:
        return {}
    out = _git(root, ["cat-file", "--batch"], ("\n".join(uniq) + "\n").encode())
    res = {}
    if out is None:
        return res
    pos = 0
    for sha in uniq:
        nl = out.find(b"\n", pos)
        if nl < 0:
            break
        header = out[pos:nl].split()
        pos = nl + 1
        if len(header) < 3 or header[1] != b"blob":
            continue
        size = int(header[2])
        res[sha] = out[pos:pos + size].decode("utf-8", "replace")
        pos += size + 1
    return res


def _index_texts(root):
    """{path: staged text} for every *.sh in the index, or None when root is not a git work tree."""
    out = _git(root, ["ls-files", "-s", "-z", "--", "*.sh"])
    if out is None:
        return None
    entries = []
    for rec in out.decode("utf-8", "replace").split("\0"):
        if "\t" not in rec:
            continue
        meta, path = rec.split("\t", 1)
        parts = meta.split()
        if len(parts) >= 2 and parts[0] != "160000":
            entries.append((parts[1], path))
    blobs = _cat_blobs(root, [sha for sha, _p in entries])
    return {path: blobs[sha] for sha, path in entries if sha in blobs}


def _worktree_texts(root):
    names = None
    out = _git(root, ["ls-files", "-z", "--", "*.sh"])
    if out is not None:
        names = [p for p in out.decode("utf-8", "replace").split("\0") if p]
    if not names:
        names = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules", ".venv")]
            for fn in filenames:
                if fn.endswith(".sh"):
                    names.append(os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/"))
    texts = {}
    for rel in names:
        try:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
                texts[rel] = fh.read()
        except OSError:
            continue
    return texts


def _staged_sh(root):
    out = _git(root, ["diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR", "--", "*.sh"])
    if out is None:
        return []
    return [p for p in out.decode("utf-8", "replace").split("\0") if p]


def _report(texts, cands, staged_mode):
    summaries = {}
    broken = {}
    for rel, text in texts.items():
        try:
            summaries[rel] = _summarize(text)
        except Exception as e:      # the parser must never crash the registry; a candidate it cannot read is LOUD
            summaries[rel] = None
            broken[rel] = "%s: %s" % (type(e).__name__, e)
    known = _known_scripts(summaries)
    cand_set = set(cands)
    cand_bases = {c.rsplit("/", 1)[-1] for c in cands}
    out = []
    for rel in sorted(texts):
        s = summaries[rel]
        if s is None:
            if rel in cand_set:
                out.append("%s:1 -- the SR gate could not parse this file (%s); fix the gate or the file"
                           % (rel, broken[rel]))
            continue
        probs = _evaluate(s, known)[0]
        for (line, tok), (kind, via) in sorted(probs.items()):
            if rel in cand_set or (staged_mode and kind == "script" and tok in cand_bases):
                out.append(_format(rel, line, tok, kind, via, s))
    return out


def check(paths, root=_ROOT):
    if paths is None:
        texts = _worktree_texts(root)
        return _report(texts, sorted(texts), False)
    explicit = [p for p in paths if str(p).replace(os.sep, "/").endswith(".sh")]
    staged = _staged_sh(root)
    if not explicit and not staged:
        return []
    texts = _index_texts(root)
    if texts is None:
        texts = _worktree_texts(root)
    cands = []
    for p in explicit + staged:
        full = p if os.path.isabs(p) else os.path.join(root, p)
        rel = os.path.relpath(full, root).replace(os.sep, "/")
        if rel not in texts:
            try:
                with open(full, encoding="utf-8", errors="replace") as fh:
                    texts[rel] = fh.read()
            except OSError:
                continue
        if rel not in cands:
            cands.append(rel)
    return _report(texts, cands, True)


# ================================================================================================================
# selftest
# ================================================================================================================
_LOOP_HEAD = "#!/usr/bin/env bash\nwhile IFS= read -r h; do\n"
_LOOP_TAIL = "done < <(printf 'a\\nb\\n')\n"

# (label, body lines inside `while IFS= read -r h; do ... done < <(...)`, must_flag)
_SELFTEST_BODIES = (
    ("bare ssh", "  timeout 10 ssh \"${SSH_F[@]}\" -o BatchMode=yes \"$h\" true 2>/dev/null\n", True),
    ("ssh -n", "  timeout 10 ssh -n \"${SSH_F[@]}\" -o BatchMode=yes \"$h\" true\n", False),
    ("ssh -nT", "  ssh -nT \"$h\" true\n", False),
    ("ssh -f", "  ssh -f \"$h\" 'sleep 1'\n", False),
    ("ssh -o StdinNull=yes", "  ssh -o StdinNull=yes \"$h\" true\n", False),
    ("ssh </dev/null", "  ssh \"$h\" true </dev/null 2>/dev/null\n", False),
    ("ssh <&3", "  ssh \"$h\" true <&3\n", False),
    ("quoted ; before </dev/null", "  ssh \"$h\" \"cd /x; ls\" </dev/null\n", False),
    ("piped in", "  printf '%s\\n' \"$X\" | ssh \"$h\" \"cat > f\"\n", False),
    ("pipe continued on next line", "  printf x |\n    ssh \"$h\" 'cat > f'\n", False),
    ("group redirect", "  { ssh \"$h\" true; ssh \"$h\" false; } </dev/null\n", False),
    ("multi-line -n", "  timeout 10 ssh -n \"${SSH_F[@]}\" \\\n    -o BatchMode=yes \"$h\" true\n", False),
    ("remote -n in quotes", "  ssh \"$h\" \"tail -n 5 log\"\n", True),
    ("remote -n bare", "  ssh \"$h\" head -n 1 f\n", True),
    ("remote </dev/null in quotes", "  ssh \"$h\" \"nohup x </dev/null &\"\n", True),
    ("-n after a pipe", "  ssh \"$h\" cat f | sort -n\n", True),
    ("-n after &&", "  ssh \"$h\" true && echo -n ok\n", True),
    ("pipe before &&", "  echo \"$h\" | grep -q x && ssh \"$h\" true\n", True),
    ("backgrounded", "  ssh \"$h\" true &\n", True),
    ("command substitution", "  r=\"$(ssh \"$h\" uptime)\"\n", True),
    ("$# is not a comment", "  n=$#; m=${#a[@]}; for i in 1; do\n    :\n  done\n  ssh \"$h\" true\n", True),
    ("done) case pattern", "  case \"$h\" in done) : ;; esac\n  x=\"$d/done\"\n  ssh \"$h\" true\n", True),
    ("for nested in while-read", "  for n in 1 2; do\n    ssh \"$h\" true\n  done\n", True),
    ("comment only", "  # TODO: ssh \"$h\" here later\n  echo \"$h\"\n", False),
)


def selftest():
    """FAILING DIRECTION FIRST: every incident shape must be caught; every correct fix shape must stay silent."""
    bad = []

    def run(label, text, must_flag, line=None, known=None):
        try:
            probs = _find_unprotected(text, known)
        except Exception as e:
            bad.append("%s: the gate CRASHED: %s: %s" % (label, type(e).__name__, e))
            return
        if must_flag and not probs:
            bad.append("MISSED: %s" % label)
        elif not must_flag and probs:
            bad.append("FALSE POSITIVE: %s -> %r" % (label, probs))
        elif must_flag and line is not None and probs[0][0] != line:
            bad.append("wrong line for %s: %r" % (label, probs))

    for label, body, must in _SELFTEST_BODIES:
        run(label, _LOOP_HEAD + body + _LOOP_TAIL, must, 3 if must and "\n  " not in body.rstrip("\n") else None)

    run("read -u 3 loop", "while IFS= read -r -u 3 h; do\n  ssh \"$h\" true\ndone 3< f\n", False)
    run("read <&3 loop", "while IFS= read -r h <&3; do\n  ssh \"$h\" true\ndone 3< f\n", False)
    run("for loop without read", "for h in a b; do\n  ssh \"$h\" true\ndone\n", False)
    run("while : without read", "while :; do\n  ssh \"$h\" true\n  sleep 5\ndone\n", False)
    run("quoted remote loop text",
        "for h in a; do\n  SSH_CMD \"$h\" \"find . | while IFS= read -r p; do rm -f -- \\\"\\$p\\\"; done\"\n"
        "done\nssh \"$onenode\" true\n", False)
    # incident 2's shape: two levels of function calls from the read loop to the ssh
    run("2-level function indirection",
        "revision_available() {\n  timeout 10 ssh \"${SSH_F[@]}\" \"$1\" \"test -f $2\"\n}\n"
        "revision_available_cached() {\n  revision_available \"$1\" \"$2\"\n}\n"
        "pop_job() {\n  while IFS= read -r cand; do\n    revision_available_cached n \"$cand\" || continue\n"
        "  done < <(awk '{print $2}' q)\n}\n", True, 2)
    run("function never called from a loop", "probe() {\n  ssh \"$1\" true\n}\nprobe x\n", False)
    run("one-line function + `function f {` + $(f) call",
        "f() { ssh \"$1\" true; }\nfunction g {\n  f \"$1\"\n}\nwhile read -r h; do\n  r=$(g \"$h\")\ndone < q\n",
        True, 1)
    run("function called with </dev/null", "f() { ssh \"$1\" true; }\nwhile read -r h; do\n  f \"$h\" </dev/null\n"
        "done < q\n", False)
    # incident 1's shape, replayed from 6406924ee^: the ssh wrapper reached through a quoted variable
    b2b = ("QUEUE_TOOL=\"$POOL_ROOT/tools/pool_queue.sh\"\nwhile IFS= read -r line; do\n"
           "  bash \"$QUEUE_TOOL\" add \"$line\" --checked \"$R\" >/dev/null 2>\"$E\"\ndone < \"$JOBS\"\n")
    run("bash \"$QUEUE_TOOL\" add in a read loop", b2b, True, 3)
    run("the 6406924ee fix shape (read <&3, add </dev/null)",
        b2b.replace("read -r line;", "read -r line <&3;").replace("\"$R\" >", "\"$R\" </dev/null >")
        .replace("done < \"$JOBS\"", "done 3< \"$JOBS\""), False)
    run("\"$QUEUE_TOOL\" called directly", b2b.replace("bash \"$QUEUE_TOOL\"", "\"$QUEUE_TOOL\""), True, 3)
    run("a script derived as draining", "while read -r h; do\n  bash tools/mywrap.sh \"$h\"\ndone < q\n", True, 2,
        known=_FLOOR_SCRIPTS | {"mywrap.sh"})
    run("ssh through a variable", "SSH=\"ssh -o BatchMode=yes\"\nwhile read -r h; do\n  $SSH \"$h\" true\ndone < q\n",
        True, 3)
    run("ssh -n through a variable",
        "SSH=\"ssh -n -o BatchMode=yes\"\nwhile read -r h; do\n  $SSH \"$h\" true\ndone < q\n", False)
    run("-n through an option array", "SSH_F=(-n -o BatchMode=yes)\nwhile read -r h; do\n  ssh \"${SSH_F[@]}\" \"$h\" "
        "true\ndone < q\n", False)
    run("rsync -e ssh (rsync gives its transport a pipe)",
        _LOOP_HEAD + "  rsync -q -e \"ssh -o BatchMode=yes\" \"$h:out/\" dest/\n" + _LOOP_TAIL, False)
    return bad


if __name__ == "__main__":
    hits = check(None)
    print("CLASS SR corpus audit: %d ssh-drains-a-read-loop site(s)" % len(hits))
    for h in hits:
        print("  ⛔", h)
