#!/usr/bin/env python3
"""tools/local_llm/hooks/deny_protected_edits.py -- logic for the PreToolUse hook, invoked by the .sh wrapper
in this directory (which owns the stdin/argv plumbing; see its header for the hook contract). Two independent
guards, both belt-and-braces with tools/local_llm/claude_local_settings.json's own allow/deny rules (a hook
fires regardless of settings-file drift):

1. EDIT PATH GUARD (Edit/Write/MultiEdit/NotebookEdit): denies an existing (non-_draft_) file under
   research/runners/, and tools/gates/**, sim/**, webapp/** -- see docs/plans/2026-09-29-local-model-transition-
   PLAN.md section 4 and LOCAL_LLM_RUNBOOK.md's "2026-09-28 VIOLATION" entry.

2. BASH GIT-ESCAPE GUARD (2026-09-29, loophole review): claude_local_settings.json allows
   `Bash(git -C .claude/worktrees/draft-:*)` so the local model can commit inside its own draft worktrees. That
   prefix rule matches ANY git subcommand run that way, not just the intended status/add/commit -- including
   --no-verify, reset --hard, push --force, rebase, and other ways to escape a draft worktree's sandboxing or
   bypass the commit gates. This guard blocks specific dangerous forms regardless of how they are spelled (a
   bare `git ...`, `git -C <path> ...`, an env-prefixed invocation, or a `bash -c "..."` / `sh -c "..."` wrapper)
   by tokenizing the Bash command (shlex, shell-metacharacter-separated into segments so a token in one `&&`-
   chained command never leaks into another) and recursing into any `-c "<nested command>"` argument.

3. BASH GATE-STATE WRITE GUARD (2026-09-29, same loophole review): the Edit-tool deny rules
   (research/coordination/**.jsonl, research/queue/pool.queue, research/queue/gpu.queue, tools/gates/**) only
   cover the Edit tool -- a Bash command can reach the same files through shell redirection (`>`/`>>`), `tee`,
   `sed -i`, `cp`/`mv`, or a `python -c "open(..., 'w')"` one-liner. This is exactly the failure mode
   LOCAL_LLM_RUNBOOK.md already names ("never write to a file a gate reads in order to get a commit through" --
   two such false registrations were reverted by Claude on 2026-09-27). Heuristic and deliberately narrow: it
   only looks at whether one of these mechanisms targets one of a short list of known gate-read state files, and
   it trusts the sanctioned wrappers (tools/pool_queue.sh add, tools/local_llm/gpu_add.sh,
   tools/waiver_history.py clear) when they are not ALSO paired with a raw redirect into one of those files.
"""
import json
import re
import shlex
import sys

EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
RUNNER_RE = re.compile(r"^research/runners/[^/]+\.py$")

SEGMENT_SPLIT_RE = re.compile(r"&&|\|\||;|\n|\|")
GIT_SUBCOMMANDS_BLOCKED = ("replace", "stash", "merge", "rebase")
SHORT_FLAG_CLUSTER = re.compile(r"^-[A-Za-z]+$")   # e.g. -n, -an, -na (never a long --flag)
MAX_NEST_DEPTH = 4

# gate-state write guard: files a gate reads to decide whether a commit/build/queue entry is real.
PROTECTED_WRITE_EXACT_SUFFIXES = (
    "research/queue/pool.queue",
    "research/queue/gpu.queue",
    "research/queue/.waiver_history.jsonl",
)
SANCTIONED_WRITE_MARKERS = (
    "tools/pool_queue.sh add",
    "tools/local_llm/gpu_add.sh",
    "tools/waiver_history.py clear",
)
REDIRECT_RE = re.compile(r"(>>?)\s*([^\s;&|]+)")
QUOTED_STR_RE = re.compile(r"""(['"])((?:(?!\1).)*)\1""")
PY_WRITE_MODE_RE = re.compile(r"""['"](?:w|a|wb|ab|w\+|a\+|x)['"]""")


# ------------------------------------------------------------------------------------------- edit path guard

def repo_relative(file_path, root):
    root = root.rstrip("/")
    if not file_path.startswith(root + "/"):
        return None
    rel = file_path[len(root) + 1:]
    # unwrap one level of a local-model draft worktree: .claude/worktrees/<name>/<rest> -> <rest>
    parts = rel.split("/")
    if len(parts) >= 4 and parts[0] == ".claude" and parts[1] == "worktrees":
        rel = "/".join(parts[3:])
    return rel


def _under(rel, top):
    return rel == top or rel.startswith(top + "/")


def check_edit(tool_name, tool_input, project_root):
    file_path = (tool_input or {}).get("file_path") or ""
    if not file_path:
        return None
    rel = repo_relative(file_path, project_root)
    if rel is None:
        return None   # outside the project entirely (e.g. a scratchpad path) -- nothing here to protect

    if RUNNER_RE.match(rel):
        basename = rel.rsplit("/", 1)[-1]
        if basename.startswith("_draft_"):
            return None
        return (f"{tool_name} on '{file_path}': editing an EXISTING file under research/runners/ is denied "
                 "for the local-model driver. New work goes in a NEW file whose basename starts with _draft_ "
                 "(research/coordination/LOCAL_LLM_RUNBOOK.md); never edit a landed runner in place -- this is "
                 "the 2026-09-28 violation this guard exists to stop.")

    for top, label in (("tools/gates", "tools/gates/**"), ("sim", "sim/**"), ("webapp", "webapp/**")):
        if _under(rel, top):
            return (f"{tool_name} on '{file_path}': {label} is denied for the local-model driver "
                     "(belt-and-braces with claude_local_settings.json's own deny rule).")
    return None


# ------------------------------------------------------------------------------------- bash git-escape guard

def _tokenize(segment):
    try:
        return shlex.split(segment)
    except ValueError:
        return segment.split()   # an unbalanced quote etc. -- fall back to a crude split rather than crash


def _has_flag_with_letter(tokens, letter):
    # a token that is exactly -n, or a combined short-flag cluster containing it (e.g. -an, -na) -- never a
    # long --flag, so this can never fire on something like --new or --name.
    return any(t != f"--{letter}" and SHORT_FLAG_CLUSTER.match(t) and letter in t[1:] for t in tokens)


def _is_protected_write_target(path_token):
    p = path_token.strip("'\"")
    if "research/coordination/" in p and p.endswith(".jsonl"):
        return True
    if any(p.endswith(suf) for suf in PROTECTED_WRITE_EXACT_SUFFIXES):
        return True
    if "tools/gates/" in p:
        return True
    return False


def _is_sanctioned_write_call(segment_text):
    return any(marker in segment_text for marker in SANCTIONED_WRITE_MARKERS)


def _check_write_targets(segment_text, tokens):
    # shell redirection (`>`/`>>`), spaced or glued (`cmd>file`); also incidentally catches `2>`/`&>` since the
    # regex searches for the trailing >/>> rather than anchoring on what precedes it.
    redirect_target = None
    for m in REDIRECT_RE.finditer(segment_text):
        target = m.group(2).strip("'\"")
        if _is_protected_write_target(target):
            redirect_target = target
            break

    if redirect_target is None and _is_sanctioned_write_call(segment_text):
        return None   # trust the sanctioned wrapper -- nothing here raw-redirects into a protected file

    if redirect_target:
        return f"shell redirect -> {redirect_target}"

    if "tee" in tokens:
        ti = tokens.index("tee")
        for t in tokens[ti + 1:]:
            if t.startswith("-"):
                continue
            if _is_protected_write_target(t):
                return f"tee -> {t}"

    if "sed" in tokens:
        si = tokens.index("sed")
        after = tokens[si + 1:]
        if any(t == "-i" or t.startswith("-i") for t in after):
            for t in after:
                if _is_protected_write_target(t):
                    return f"sed -i -> {t}"

    for cmdname in ("cp", "mv"):
        if cmdname in tokens:
            ci = tokens.index(cmdname)
            non_flags = [t for t in tokens[ci + 1:] if not t.startswith("-")]
            if non_flags and _is_protected_write_target(non_flags[-1]):
                return f"{cmdname} -> {non_flags[-1]}"

    if "open(" in segment_text and PY_WRITE_MODE_RE.search(segment_text):
        for q in QUOTED_STR_RE.finditer(segment_text):
            candidate = q.group(2)
            if _is_protected_write_target(candidate):
                return f"python open(..., write mode) -> {candidate}"

    return None


def _check_segment_tokens(segment_text, tokens, depth=0):
    write_hit = _check_write_targets(segment_text, tokens)
    if write_hit:
        return write_hit

    if "--no-verify" in tokens:
        return "--no-verify"
    if any(t == "GIT_NO_VERIFY" or t.startswith("GIT_NO_VERIFY=") for t in tokens):
        return "GIT_NO_VERIFY"
    if any(t == "SKIP" or t.startswith("SKIP=") for t in tokens):
        return "SKIP="
    if any("core.hookspath" in t.lower() for t in tokens):
        return "core.hooksPath"
    if "update-ref" in tokens:
        return "git update-ref"
    if "commit-tree" in tokens:
        return "git commit-tree"
    if "cherry-pick" in tokens:
        return "git cherry-pick"
    if "--force-with-lease" in tokens:
        return "--force-with-lease"

    if "hash-object" in tokens:
        hi = tokens.index("hash-object")
        after = tokens[hi + 1:]
        if "-w" in after or _has_flag_with_letter(after, "w"):
            return "git hash-object -w"

    if "git" in tokens:
        gi = tokens.index("git")
        rest = tokens[gi + 1:]

        if "commit" in rest:
            ci = rest.index("commit")
            after_commit = rest[ci + 1:]
            if "-n" in after_commit or _has_flag_with_letter(after_commit, "n"):
                return "git commit -n (--no-verify)"

        for sub in GIT_SUBCOMMANDS_BLOCKED:
            if sub in rest:
                return f"git {sub}"

        if "push" in rest:
            pi = rest.index("push")
            after_push = rest[pi + 1:]
            if any(t in ("--force", "-f", "--force-with-lease") for t in after_push):
                return "git push --force"

        if "reset" in rest and "--hard" in rest:
            return "git reset --hard"

    # recurse into a `-c "<nested command>"` argument (bash -c / sh -c / zsh -c / env ... sh -c ...)
    if depth < MAX_NEST_DEPTH:
        for i, t in enumerate(tokens):
            if t == "-c" and i + 1 < len(tokens):
                nested = _check_command_string(tokens[i + 1], depth=depth + 1)
                if nested:
                    return nested
    return None


def _check_command_string(command, depth=0):
    for segment in SEGMENT_SPLIT_RE.split(command):
        segment = segment.strip()
        if not segment:
            continue
        hit = _check_segment_tokens(segment, _tokenize(segment), depth=depth)
        if hit:
            return hit
    return None


def check_bash(tool_input):
    command = (tool_input or {}).get("command") or ""
    if not command:
        return None
    hit = _check_command_string(command)
    if hit:
        return (f"Bash command denied for the local-model driver: contains '{hit}'. This is blocked "
                "regardless of how it is spelled (git -C <worktree>, an env prefix, or a bash -c/sh -c "
                "wrapper) -- see tools/local_llm/hooks/deny_protected_edits.py's BASH GIT-ESCAPE GUARD.")
    return None


# --------------------------------------------------------------------------------------------------- entry

def decide(payload_json, project_root):
    try:
        data = json.loads(payload_json)
    except Exception:
        return None   # unparsable payload -- fail OPEN, never block on a malformed hook input

    tool_name = data.get("tool_name") or ""
    tool_input = data.get("tool_input") or {}

    if tool_name in EDIT_TOOLS:
        return check_edit(tool_name, tool_input, project_root)
    if tool_name == "Bash":
        return check_bash(tool_input)
    return None


def main():
    if len(sys.argv) != 3:
        sys.exit(0)   # wrong invocation -- fail open rather than block on our own bug
    reason = decide(sys.argv[1], sys.argv[2])
    if reason:
        print("BLOCK|" + reason)
    else:
        print("ALLOW")


if __name__ == "__main__":
    main()
