#!/usr/bin/env python3
"""PreToolUse hook: BLOCK any delete aimed at irreplaceable locations.

WHY THIS IS A HOOK AND NOT A NOTE (2026-09-24). A code-review subagent, cleaning up its own scratch clone, ran

    rm -rf <repo>/.claude/worktrees/scratch_s06_clone /home/dant123/.claude/projects

The second path was a plain mistake that nothing checked. It destroyed every project's Claude session history and
memory notes on this machine (no /home snapshot existed; a July backup covered 62 of 94 sim notes). "Be careful with
rm -rf" is a rule; a rule cannot say no. This can.

INSTALLED AT USER LEVEL, not project level: `tools/install_delete_guard.sh` copies this file to ~/.claude/hooks/ and
registers it in ~/.claude/settings.json, so it guards every project and every subagent on the machine (a
project-level hook would not have protected the twelve other projects whose history was also lost).

WHAT IS PROTECTED
  * Nothing at or inside: ~/.claude, ~/.claude.json, ~/.claude-config, ~/.ssh, ~/.gnupg, ~/.config/Claude,
    ~/.local/share/Trash, /mnt/vault, plus any ':'-separated extra roots in $CLAUDE_GUARD_EXTRA_PROTECTED.
  * No directory that CONTAINS one of those or $HOME (/, /home, $HOME, /mnt ...).
  * No top-level folder of $HOME (~/Projects, ~/Documents ...) and no project root (~/Projects/<name>).
    Deleting INSIDE a project (a worktree, a scratch dir, an artifact) is untouched.
  * One allowance: a single named memory note (~/.claude/projects/<p>/memory/<note>.md), no -r, no wildcard, may
    be removed, or renamed within its memory folder, because the memory system asks for wrong notes to be deleted.

WHAT COUNTS AS A DELETE
  rm, unlink, shred, srm, trash, trash-put, gio trash/remove, mv (its sources), find ... -delete / -exec rm,
  rsync --delete / --remove-source-files, git worktree remove, xargs rm and `while read` loops (judged by what the
  pipe feeds them), and inline python/perl/node deletes (shutil.rmtree, os.remove, Path.unlink, rmSync ...) that
  name a protected path. `bash -c` / `sh -c` / `eval` strings, $(...) and backtick substitutions, and heredocs fed
  to a shell or an interpreter are checked recursively. rmdir is exempt: it cannot remove a non-empty directory.
  A filtered delete (find -delete, xargs rm, a while-read loop, inline code) only deletes SOME things under its
  root, so it is judged against the protected locations and their parents, not against project roots.

UNKNOWN VARIABLES. A variable the hook cannot resolve (not assigned earlier in the same command, not in the
environment) is judged as EMPTY, which is what the shell does when it is unset: "$DIR/$name" with both unset is `/`
and is blocked. Write ${DIR:?} to make the shell itself refuse an empty value; that form is allowed.

FALSE ALARMS ARE A FAILURE TOO (the lesson of block_self_matching_kill.py, which blocked its own commit): quoted
text, commit messages, echo/grep arguments and heredocs fed to cat/git/tee are data, not deletes, and are not judged.

Exit 2 blocks the call and shows the message to the model. An internal error falls back to a crude text check
(a delete verb AND a protected path both appear) instead of silently letting the command through.

    python3 .claude/hooks/test_guard_protected_delete.py
"""
import glob as _glob
import itertools
import json
import os
import re
import sys

MAX_DEPTH = 6
MAX_CANDIDATES = 64

# Characters that were QUOTED or ESCAPED in the source are carried as sentinels: literal, never expanded or globbed.
_Q = {"$": "\x01", "*": "\x02", "?": "\x03", "[": "\x04", "~": "\x05"}
_UNQ = {v: k for k, v in _Q.items()}
_GUARDED = "\x06guarded\x06"          # stands in for ${VAR:?}: the shell refuses to expand it empty
_MKTEMP = "/tmp/\x06mktemp\x06"       # stands in for VAR=$(mktemp ...)

_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_PREFIX_CMDS = {"sudo", "doas", "command", "builtin", "exec", "nohup", "nice", "ionice", "time", "setsid",
                "stdbuf", "env", "timeout", "chronic", "unbuffer", "systemd-run", "flock"}
_PREFIX_OPT_ARGS = {"sudo": {"-u", "-g", "-C", "-D", "-h", "-p", "-r", "-t", "-U"}, "doas": {"-u", "-C"},
                    "timeout": {"-k", "-s", "--kill-after", "--signal"}, "nice": {"-n"},
                    "ionice": {"-c", "-n", "-p"}, "flock": {"-w", "-E", "--timeout", "--conflict-exit-code"},
                    "systemd-run": {"-p", "-u", "--unit", "--property", "-E", "--setenv"},
                    "env": {"-u", "-C", "--unset", "--chdir"}}
_KEYWORDS = {"do", "then", "else", "elif", "if", "while", "until", "!", "{", "}", "done", "fi", "esac", "in"}
_SHELLS = {"bash", "sh", "zsh", "dash", "fish", "ksh"}
_INTERPS = {"python", "python3", "python2", "py", "perl", "node", "ruby", "deno", "bun"}
_CODE_DELETE = re.compile(
    r"rmtree|os\.remove|os\.unlink|os\.rmdir|\.unlink\s*\(|\.rmdir\s*\(|remove_tree|unlinkSync|rmSync|rmdirSync|"
    r"rimraf|File::Path|\bunlink\s*\(|fs\.rm\b|FileUtils\.rm|os\.system\([^)]*\brm\b|subprocess[^\n]*['\"]rm['\"]")
_PROTECTED_HOME_NAMES = (".claude", ".claude.json", ".claude-config", ".ssh", ".gnupg", ".config", ".local",
                         "Projects")


def _home(home=None):
    return os.path.normpath(home or os.path.expanduser("~"))


def hard_roots(home, env=None):
    env = os.environ if env is None else env
    roots = [os.path.join(home, p) for p in (".claude", ".claude.json", ".claude-config", ".ssh", ".gnupg",
                                             ".config/Claude", ".local/share/Trash")]
    roots.append("/mnt/vault")
    for extra in (env.get("CLAUDE_GUARD_EXTRA_PROTECTED") or "").split(":"):
        if extra.strip():
            roots.append(os.path.normpath(os.path.expanduser(extra.strip())))
    return roots


def _inside(p, root):
    """p equals root or lies under it."""
    root = root.rstrip("/") or "/"
    return root == "/" or p == root or p.startswith(root + "/")


def _is_memory_note(p, home):
    base = re.escape(os.path.join(home, ".claude", "projects"))
    return re.fullmatch(base + r"/[^/]+/memory/[^/]+\.md", p) is not None


def judge_path(p, home, env, recursive, filtered=False, glob_prefix=False):
    """Reason string if deleting path p is forbidden, else None. p is absolute and normalized.

    filtered: the delete removes only SOME things under p (find -delete, xargs, while-read, inline code), so p is
    judged against protected locations and their parents, not against project roots or top-level home folders.
    """
    roots = hard_roots(home, env)
    for r in roots:
        if _inside(p, r):
            if not recursive and not glob_prefix and not filtered and _is_memory_note(p, home):
                return None
            return "%s is inside the protected location %s" % (p, r)
    if p == home:
        return "%s is your whole home folder" % p
    for r in roots + [home]:
        if _inside(r, p):
            return "deleting %s would delete %s" % (p, r)
    if filtered:
        return None
    parent = os.path.dirname(p)
    big = recursive or glob_prefix or os.path.isdir(p)
    if big and parent == "/":
        return "%s is a top-level system directory" % p
    if big and parent == home:
        return "%s is a top-level folder of your home directory" % p
    if big and parent == os.path.join(home, "Projects"):
        return "%s is a whole project folder" % p
    if glob_prefix and p == os.path.join(home, "Projects"):
        return "a wildcard in %s would match whole project folders" % p
    return None


# ---------------------------------------------------------------------------------------------------- tokenizing
_HEREDOC = re.compile(r"(?<!<)<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def _strip_heredocs(text):
    """Return (shell_text, [(consumer_word, body)]). Heredoc bodies are removed from the shell text."""
    lines = text.split("\n")
    out, bodies, i = [], [], 0
    while i < len(lines):
        ln = lines[i]
        m = _HEREDOC.search(ln)
        end = None
        if m:
            term = m.group(2)
            for k in range(i + 1, len(lines)):
                if lines[k].strip() == term:
                    end = k
                    break
        if not m or end is None:                       # no terminator: not a heredoc, keep the text as shell
            out.append(ln)
            i += 1
            continue
        before = ln[:m.start()]
        consumer = ""
        for w in re.split(r"&&|\|\||[;|&(`]|\$\(", before)[-1].split():
            if _ASSIGN.match(w) or w in _PREFIX_CMDS or w in _KEYWORDS or w.startswith("-"):
                continue
            consumer = os.path.basename(w.strip("\"'"))
            break
        out.append(ln)
        bodies.append((consumer, "\n".join(lines[i + 1:end])))
        i = end + 1
    return "\n".join(out), bodies


def _substitutions(text):
    """Inner text of $(...) and `...` that the shell would EXECUTE (not inside single quotes)."""
    found, i, n, sq, dq = [], 0, len(text), False, False
    while i < n:
        c = text[i]
        if c == "\\" and not sq:
            i += 2
            continue
        if c == "'" and not dq:
            sq = not sq
        elif c == '"' and not sq:
            dq = not dq
        elif not sq and text.startswith("$(", i) and not text.startswith("$((", i):
            depth, j = 1, i + 2
            while j < n and depth:
                depth += {"(": 1, ")": -1}.get(text[j], 0)
                j += 1
            found.append(text[i + 2:j - 1])
            i = j
            continue
        elif not sq and c == "`":
            j = text.find("`", i + 1)
            j = n if j < 0 else j
            found.append(text[i + 1:j])
            i = j + 1
            continue
        i += 1
    return found


def tokenize(text):
    """Quote-aware split into ('W', word) / ('S', separator) / ('R', redirect target) tokens."""
    toks, cur, have, i, n = [], [], False, 0, len(text)
    state = {"redirect": False}

    def flush():
        nonlocal cur, have
        if have:
            toks.append(("R" if state["redirect"] else "W", "".join(cur)))
            state["redirect"] = False
        cur, have = [], False

    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            if text[i + 1] == "\n":
                i += 2
                continue
            cur.append(_Q.get(text[i + 1], text[i + 1]))
            have, i = True, i + 2
            continue
        if c == "'":
            j = text.find("'", i + 1)
            j = n if j < 0 else j
            cur.append("".join(_Q.get(ch, ch) for ch in text[i + 1:j]))
            have, i = True, j + 1
            continue
        if c == '"':
            j, buf = i + 1, []
            while j < n and text[j] != '"':
                if text[j] == "\\" and j + 1 < n and text[j + 1] in '"\\$`':
                    buf.append(_Q.get(text[j + 1], text[j + 1]))
                    j += 2
                    continue
                if text.startswith("${", j):                # parameter expansion: keep its operators raw
                    k = text.find("}", j)
                    k = n - 1 if k < 0 else k
                    buf.append(text[j:k + 1])
                    j = k + 1
                    continue
                ch = text[j]
                buf.append(ch if ch == "$" else _Q.get(ch, ch))
                j += 1
            cur.append("".join(buf))
            have, i = True, j + 1
            continue
        if c in " \t":
            flush()
            i += 1
            continue
        if c == "#" and not have:
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c in "<>":
            if have and "".join(cur).isdigit():
                cur, have = [], False
            flush()
            j = i
            while j < n and text[j] in "<>&|":
                j += 1
            if text[j - 1] == "&" and j < n and (text[j].isdigit() or text[j] == "-"):
                i = j + 1                                   # 2>&1, >&-
                continue
            state["redirect"] = True
            i = j
            continue
        two = text[i:i + 2]
        if two in ("&&", "||", "|&", ";;", "$("):
            flush()
            toks.append(("S", two))
            i += 2
            continue
        if c in ";|&()\n`":
            flush()
            toks.append(("S", c))
            i += 1
            continue
        cur.append(c)
        have, i = True, i + 1
    flush()
    return toks


def segments(toks):
    """Simple commands as (words, separator_before, separator_after)."""
    segs, words, before = [], [], None
    for kind, val in toks:
        if kind == "S":
            if words:
                segs.append([words, before, val])
            elif segs and segs[-1][2] is None:
                segs[-1][2] = val
            words, before = [], val
        elif kind == "W":
            words.append(val)
    if words:
        segs.append([words, before, None])
    return segs


# ----------------------------------------------------------------------------------------------------- expanding
_VAR = re.compile(r"\$\{([^}]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)|\$([0-9@*#?$!-])")


def expand(word, vars_, env, home):
    """Expand one word into candidate strings. Unknown variables become EMPTY (what an unset variable does)."""
    if word.startswith("~") and (len(word) == 1 or word[1] == "/"):
        word = home + word[1:]
    elif word.startswith("~") and re.match(r"^~[A-Za-z0-9_.-]+(/|$)", word):
        word = os.path.expanduser(word)
    parts, pos = [], 0
    for m in _VAR.finditer(word):
        parts.append([word[pos:m.start()]])
        pos = m.end()
        if m.group(3) is not None:
            parts.append([""])
            continue
        body = m.group(1) if m.group(1) is not None else m.group(2)
        mm = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(.*)$", body, re.S)
        if not mm:
            parts.append([""])
            continue
        name, op = mm.group(1), mm.group(2)
        known = vars_.get(name)
        if known is None and name in env:
            known = [env[name]]
        if known is None and name == "HOME":
            known = [home]
        if op == "":
            parts.append(known if known is not None else [""])
        elif op.startswith(":?") or op.startswith("?"):
            parts.append(known if known is not None else [_GUARDED])
        elif op[:2] in (":-", ":=") or op[:1] in ("-", "="):
            dflt = op[2:] if op[0] == ":" else op[1:]
            parts.append(known if known is not None else expand(dflt, vars_, env, home))
        elif op.startswith(":+") or op.startswith("+"):
            alt = op[2:] if op[0] == ":" else op[1:]
            parts.append(expand(alt, vars_, env, home) if known is not None else [""])
        else:
            parts.append([""])
    parts.append([word[pos:]])
    out = []
    for combo in itertools.product(*parts):
        out.append("".join(combo))
        if len(out) >= MAX_CANDIDATES:
            break
    return out


def _literal(s):
    return "".join(_UNQ.get(ch, ch) for ch in s)


def _has_glob(s):
    return any(ch in s for ch in "*?[")


def _path_like(w):
    lit = _literal(w)
    return (lit.startswith(("/", "~", "$", ".")) or "/" in lit) and not lit.startswith("-")


# ------------------------------------------------------------------------------------------------------ judging
class _Ctx:
    def __init__(self, home, env):
        self.home, self.env = home, env
        self.vars, self.filtered_vars = {}, set()
        self.cwds = [None]
        self.reasons = []


def _refs_filtered(word, ctx):
    return any(re.search(r"\$\{?%s\b" % re.escape(v), word) for v in ctx.filtered_vars)


def _judge_word(word, ctx, recursive, verb, filtered=False):
    filtered = filtered or _refs_filtered(word, ctx)
    for cand in expand(word, ctx.vars, ctx.env, ctx.home):
        lit = _literal(cand)
        if lit == "" or _GUARDED in lit:
            if _GUARDED in lit and lit.replace(_GUARDED, "").strip("/") == "":
                continue                                    # ${X:?} alone: the shell refuses an empty value
            if lit == "":
                continue
        globbed = _has_glob(cand)
        static = []
        for comp in cand.split("/"):
            if _has_glob(comp):
                break
            static.append(comp)
        static_lit = _literal("/".join(static)) if globbed else lit
        bases = [None] if os.path.isabs(lit) else ctx.cwds
        for cwd in bases:
            if cwd is None and not os.path.isabs(lit):
                first = lit.split("/")[0]
                if lit in (".", "..", "*", "./", "../") or lit.startswith("../") or first in _PROTECTED_HOME_NAMES \
                        or (globbed and "/" not in lit):
                    ctx.reasons.append("`%s %s` is relative to a working directory this guard cannot determine"
                                       % (verb, _literal(word)))
                continue
            full = lit if cwd is None else os.path.join(cwd, lit)
            if globbed:
                prefix = static_lit if os.path.isabs(lit) else os.path.join(cwd, static_lit)
                prefix = os.path.normpath(prefix or "/")
                r = judge_path(prefix, ctx.home, ctx.env, True, filtered=filtered, glob_prefix=True)
                if r:
                    ctx.reasons.append("`%s %s`: a wildcard under %s -- %s" % (verb, _literal(word), prefix, r))
                    continue
                for mt in _glob.glob(full)[:500]:
                    _judge_resolved(os.path.normpath(mt), ctx, recursive, verb, word, filtered)
            else:
                _judge_resolved(os.path.normpath(full), ctx, recursive, verb, word, filtered,
                                trailing_slash=lit.endswith("/"))


def _judge_resolved(p, ctx, recursive, verb, word, filtered, trailing_slash=False):
    cands = [p]
    try:
        real = os.path.realpath(p) if trailing_slash else \
            os.path.join(os.path.realpath(os.path.dirname(p)), os.path.basename(p))
        if os.path.normpath(real) != p:
            cands.append(os.path.normpath(real))
    except Exception:
        pass
    for q in cands:
        r = judge_path(q, ctx.home, ctx.env, recursive, filtered=filtered)
        if r:
            ctx.reasons.append("`%s %s` -> %s" % (verb, _literal(word), r))
            return


def _resolve_abs(word, ctx):
    out = []
    for cand in expand(word, ctx.vars, ctx.env, ctx.home):
        lit = _literal(cand)
        if os.path.isabs(lit):
            out.append(os.path.normpath(lit))
        else:
            out += [os.path.normpath(os.path.join(c, lit)) for c in ctx.cwds if c is not None]
    return out


def _analyze_code(text, ctx):
    """Inline python/perl/node: a delete call AND a protected path in the same text."""
    if not _CODE_DELETE.search(text):
        return
    lits = re.findall(r"""['"]([^'"\n]{1,400})['"]""", text)
    home_literal = False
    for s in lits:
        if s.startswith(("/", "~")):
            p = os.path.normpath(ctx.home + s[1:] if s.startswith("~") else s)
            if p in (ctx.home, ctx.home + "/"):
                home_literal = True
                continue
            r = judge_path(p, ctx.home, ctx.env, True, filtered=True)
            if r:
                ctx.reasons.append("inline code deletes: %s" % r)
                return
    if home_literal or re.search(r"Path\.home\(\)|expanduser\(\s*['\"]~|environ\[\s*['\"]HOME|environ\.get\(\s*['\"]"
                                 r"HOME|getenv\(\s*['\"]HOME|\$ENV\{HOME\}|os\.homedir\(\)", text):
        for s in lits:
            if s.strip("/").split("/")[0] in _PROTECTED_HOME_NAMES:
                ctx.reasons.append("inline code deletes something under your home folder named %r" % s)
                return


def _strip_prefixes(w):
    while w:
        if _ASSIGN.match(w[0]):
            w.pop(0)
            continue
        if w[0] in _PREFIX_CMDS:
            head = w.pop(0)
            while w and w[0].startswith("-"):
                opt = w.pop(0)
                if opt in _PREFIX_OPT_ARGS.get(head, ()) and w:
                    w.pop(0)
            if head == "timeout" and w and re.match(r"^[0-9.]+[smhd]?$", w[0]):
                w.pop(0)
            if head == "flock" and w:
                w.pop(0)                                    # the lock file or fd
            continue
        break
    return w


def _analyze(command, ctx, depth):
    if depth > MAX_DEPTH or not command:
        return
    shell_text, bodies = _strip_heredocs(command)
    for consumer, body in bodies:
        if consumer in _SHELLS or consumer in ("eval", "source", "."):
            _analyze(body, ctx, depth + 1)
        elif consumer in _INTERPS or re.match(r"^python\d", consumer or ""):
            _analyze_code(body, ctx)
    for inner in _substitutions(shell_text):
        _analyze(inner, ctx, depth + 1)

    segs = segments(tokenize(shell_text))
    pending_for, prev_words = None, None
    for idx, (words, sep_before, sep_after) in enumerate(segs):
        producer = prev_words if sep_before in ("|", "|&") else None
        prev_words = words
        w = list(words)

        # a `for x in $(...)` loop or `VAR=$(...)` assignment takes its values from the substitution segment
        if pending_for and sep_before in ("$(", "`"):
            name, pending_for = pending_for, None
            feed = [x for x in w[1:] if _path_like(x)]
            ctx.vars[name] = [_literal(c) for x in feed for c in expand(x, ctx.vars, ctx.env, ctx.home)] or [""]
            ctx.filtered_vars.add(name)
        while w and w[0] in _KEYWORDS:
            w.pop(0)
        if w and w[0] in ("export", "local", "declare", "readonly", "typeset"):
            w = [x for x in w[1:] if not x.startswith("-")]
        if w and all(_ASSIGN.match(x) for x in w):
            for x in w:
                k, v = x.split("=", 1)
                if v == "" and sep_after in ("$(", "`") and idx + 1 < len(segs):
                    nxt = _strip_prefixes(list(segs[idx + 1][0]))
                    ctx.vars[k] = [_MKTEMP] if nxt and os.path.basename(nxt[0]) == "mktemp" else [""]
                else:
                    ctx.vars[k] = expand(v, ctx.vars, ctx.env, ctx.home)
                ctx.filtered_vars.discard(k)
            continue
        w = _strip_prefixes(w)
        if not w:
            continue
        verb = os.path.basename(_literal(w[0]))
        args = w[1:]

        if verb == "for" and args:
            name = args[0]
            if "in" in args:
                vals = [_literal(c) for x in args[args.index("in") + 1:]
                        for c in expand(x, ctx.vars, ctx.env, ctx.home)]
                if vals:
                    ctx.vars[name] = vals
                    ctx.filtered_vars.discard(name)
                elif sep_after in ("$(", "`"):
                    pending_for = name
                else:
                    ctx.vars[name] = [""]
            continue
        if verb in ("read", "mapfile", "readarray"):
            names = [x for x in args if not x.startswith("-")]
            feed = [x for x in (producer or [])[1:] if _path_like(x)]
            for nme in names[:1]:
                ctx.vars[nme] = [_literal(c) for x in feed for c in expand(x, ctx.vars, ctx.env, ctx.home)] or [""]
                ctx.filtered_vars.add(nme)
            continue
        if verb in ("cd", "pushd"):
            plain = [a for a in args if not a.startswith("-")]
            if sep_after in ("$(", "`") and not plain:
                ctx.cwds = [None]                       # cd $(...): the destination is unknowable here
                continue
            tgt = plain[-1] if plain else "~"
            new = []
            for c in (_literal(x) for x in expand(tgt, ctx.vars, ctx.env, ctx.home)):
                if c == "" or _GUARDED in c or "$(" in c or "`" in c:
                    new.append(None)
                    continue
                for base in ctx.cwds:
                    if os.path.isabs(c):
                        new.append(os.path.normpath(c))
                    elif base is not None:
                        new.append(os.path.normpath(os.path.join(base, c)))
                    else:
                        new.append(None)
            # a cd that fails leaves the shell where it was: keep judging against the old directory too
            keep_old = ctx.cwds if any(x is None or not os.path.isdir(x) for x in new) else []
            ctx.cwds = list(dict.fromkeys(new + keep_old)) or [None]
            continue
        if verb == "popd":
            ctx.cwds = [None]
            continue
        if verb in _SHELLS or verb in ("eval", "source", "."):
            if verb == "eval":
                _analyze(" ".join(_literal(a) for a in args), ctx, depth + 1)
            elif "-c" in args and args.index("-c") + 1 < len(args):
                _analyze(_literal(args[args.index("-c") + 1]), ctx, depth + 1)
            continue
        if verb in _INTERPS or re.match(r"^python\d", verb):
            for flag in ("-c", "-e", "-E"):
                if flag in args and args.index(flag) + 1 < len(args):
                    # the outer shell expands $VARS inside a double-quoted -c string before the interpreter runs
                    for code in expand(args[args.index(flag) + 1], ctx.vars, ctx.env, ctx.home)[:8]:
                        _analyze_code(_literal(code), ctx)
            continue

        if verb == "xargs":
            k = 0
            while k < len(args) and args[k].startswith("-"):
                k += 2 if args[k] in ("-I", "-n", "-P", "-d", "-L", "-s", "-E") else 1
            inner = os.path.basename(_literal(args[k])) if k < len(args) else ""
            if inner in ("rm", "unlink", "shred", "trash-put", "trash", "mv") and producer:
                for x in producer[1:]:
                    if _path_like(x):
                        _judge_word(x, ctx, True, "xargs " + inner, filtered=True)
            continue
        if verb == "rm":
            if "--no-preserve-root" in args:
                ctx.reasons.append("`rm --no-preserve-root` is never allowed")
            recursive = any((re.match(r"^-[a-zA-Z]*[rR]", a) and not a.startswith("--")) or a == "--recursive"
                            for a in args)
            opts_done = False
            for a in args:
                if not opts_done and a == "--":
                    opts_done = True
                    continue
                if not opts_done and a.startswith("-") and a != "-":
                    continue
                _judge_word(a, ctx, recursive, "rm")
        elif verb in ("unlink", "shred", "srm"):
            for t in [a for a in args if not a.startswith("-")]:
                _judge_word(t, ctx, verb == "srm", verb)
        elif verb in ("trash", "trash-put"):
            for t in [a for a in args if not a.startswith("-")]:
                _judge_word(t, ctx, True, verb)
        elif verb == "gio":
            if args and args[0] in ("trash", "remove", "rm"):
                for t in [a for a in args[1:] if not a.startswith("-")]:
                    _judge_word(t, ctx, True, "gio " + args[0])
        elif verb == "mv":
            plain = [a for a in args if not a.startswith("-")]
            tdir = None
            for k, a in enumerate(args):
                if a in ("-t", "--target-directory") and k + 1 < len(args):
                    tdir = args[k + 1]
                elif a.startswith("--target-directory="):
                    tdir = a.split("=", 1)[1]
            sources = [a for a in plain if a != tdir] if tdir else plain[:-1]
            dest = tdir if tdir else (plain[-1] if plain else None)
            dests = _resolve_abs(dest, ctx) if dest else []
            for s in sources:
                srcs = _resolve_abs(s, ctx)
                if srcs and not _has_glob(s) and all(_is_memory_note(x, ctx.home) for x in srcs) and dests and \
                        all(os.path.dirname(d) == os.path.dirname(srcs[0]) for d in dests):
                    continue                            # renaming a memory note within its own folder
                _judge_word(s, ctx, True, "mv")
        elif verb == "find":
            has_delete = "-delete" in args
            for k, a in enumerate(args):
                if a in ("-exec", "-execdir", "-ok", "-okdir") and k + 1 < len(args) and \
                        os.path.basename(_literal(args[k + 1])) in ("rm", "unlink", "shred", "trash-put", "mv"):
                    has_delete = True
            if has_delete:
                roots = []
                for a in args:
                    if a.startswith("-") or a in ("(", "!", ")"):
                        break
                    roots.append(a)
                for r in roots or ["."]:
                    _judge_word(r, ctx, True, "find ... -delete", filtered=True)
        elif verb == "rsync":
            plain = [a for a in args if not a.startswith("-")]
            if plain and any(a.startswith("--delete") or a == "--del" for a in args):
                _judge_word(plain[-1], ctx, True, "rsync --delete")
            if plain and "--remove-source-files" in args:
                for s in plain[:-1]:
                    _judge_word(s, ctx, True, "rsync --remove-source-files")
        elif verb == "git":
            if "worktree" in args:
                k = args.index("worktree")
                rest = args[k + 1:]
                if rest and rest[0] == "remove":
                    tgt = [a for a in rest[1:] if not a.startswith("-")]
                    if tgt:
                        _judge_word(tgt[0], ctx, True, "git worktree remove")


def analyze(command, cwd, home=None, env=None):
    """Reasons the command must be blocked (empty list = allow)."""
    ctx = _Ctx(_home(home), dict(os.environ) if env is None else env)
    ctx.cwds = [os.path.normpath(cwd)] if cwd else [None]
    _analyze(command, ctx, 0)
    return list(dict.fromkeys(ctx.reasons))


_CRUDE_VERB = re.compile(r"(?:^|[\s;&|(`])(?:rm|unlink|shred|trash-put|rsync|find|mv)\s|rmtree|os\.remove|rmSync")


def crude(command, home=None, env=None):
    home = _home(home)
    if not _CRUDE_VERB.search(command or ""):
        return []
    needles = set()
    for r in hard_roots(home, env):
        needles.add(r)
        if r.startswith(home + "/"):
            rel = r[len(home) + 1:]
            needles.update({"~/" + rel, "$HOME/" + rel, "${HOME}/" + rel})
    hit = sorted(n for n in needles if n in command)
    return ["(fallback check) a delete command names the protected location %s" % hit[0]] if hit else []


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    if payload.get("tool_name") != "Bash":
        return 0
    cmd = (payload.get("tool_input") or {}).get("command", "") or ""
    cwd = payload.get("cwd") or os.getcwd()
    try:
        reasons = analyze(cmd, cwd)
    except Exception as exc:                                  # never fail open silently
        reasons = crude(cmd)
        if reasons:
            reasons.append("(the precise check crashed: %r)" % exc)
    if not reasons:
        return 0
    sys.stderr.write(
        "BLOCKED by the protected-delete guard:\n  - " + "\n  - ".join(reasons[:8]) + "\n\n"
        "This guard exists because on 2026-09-24 a subagent ran `rm -rf <scratch dir> "
        "/home/dant123/.claude/projects` and destroyed every project's Claude session history and memory notes.\n"
        "If a path is wrong, fix it. If an unset variable is the problem, assign it in the same command or write "
        "${VAR:?}.\nIf this deletion is genuinely intended, do NOT work around the guard: stop and ask the user to "
        "run it themselves.\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
