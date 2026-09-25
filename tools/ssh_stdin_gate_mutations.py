"""Mutation check for the CLASS SR gate (tools/gates/ssh_stdin_in_read_loop.py).

Each mutation reverts ONE fix from the 2026-09-25 r2 review round (or one measured design decision) and runs
tests/test_gate_ssh_stdin_in_read_loop.py. Every mutation must turn the suite red, and the restored gate must be
green again. A mutation whose pattern is not found exactly once aborts the run (the list is out of date).

    python tools/ssh_stdin_gate_mutations.py --out research/coordination/sr_gate_mutations_2026-09-25.json

It rewrites the gate file IN PLACE for each run, restores it in a `finally` and checks the restored file's hash. Do
not run it while another process is committing that file.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE = os.path.join(ROOT, "tools", "gates", "ssh_stdin_in_read_loop.py")
TESTS = "tests/test_gate_ssh_stdin_in_read_loop.py"

# (label, [(exact text in the gate, replacement)])
MUTATIONS = [
    ("HIGH1 staged ACMR query dropped", [("    staged = _staged_sh(root)\n", "    staged = []\n")]),
    ("HIGH1 index blob not read (worktree instead)",
     [("    texts = _index_texts(root)\n    if texts is None:", "    texts = None\n    if texts is None:")]),
    ("HIGH2 variable resolution of command words dropped",
     [("                if ref is not None and depth < 4 and ref[0] in self.vars:\n                    out = []",
       "                if False:\n                    out = []"),
      ("                if ref is not None and _SSH_VAR_RE.search(ref[0]):", "                if False:")]),
    ("HIGH2 corpus-derived wrapper set dropped (floor only)",
     [("                new.add(rel.rsplit(\"/\", 1)[-1])", "                pass")]),
    ("MED any later -n counts (old statement window)",
     [("            if host_seen:\n                return False\n            host_seen = True\n            i += 1\n"
       "        return False",
       "            host_seen = True\n            i += 1\n        return False")]),
    ("MED old _piped_in: an earlier pipe in the and-or list protects",
     [("        for ao in lst:\n            for pipe in ao:\n                for k, cmd in enumerate(pipe):\n"
       "                    self._walk_cmd(cmd, state if k == 0 else \"safe\", k > 0, sink)",
       "        for ao in lst:\n            seen = False\n            for pipe in ao:\n"
       "                for k, cmd in enumerate(pipe):\n"
       "                    self._walk_cmd(cmd, state if (k == 0 and not seen) else \"safe\", k > 0, sink)\n"
       "                seen = seen or len(pipe) > 1")]),
    ("MED # is a comment anywhere ($#, ${#a[@]})",
     [("        if d.isdigit() or d in \"#?$!@*-\":", "        if d.isdigit() or d in \"?$!@*-\":"),
      ("            if c == \"$\":\n                i = self._dollar(i, shape, subs, False)",
       "            if c == \"#\" and shape:\n                break\n"
       "            if c == \"$\":\n                i = self._dollar(i, shape, subs, False)"),
      ("            if c == \"$\":\n                j = self._dollar(j, scratch, subs, in_dq)",
       "            if c == \"#\":\n                return n\n"
       "            if c == \"$\":\n                j = self._dollar(j, scratch, subs, in_dq)")]),
    ("MED case pattern `done)` closes the loop",
     [("                if t2.kind == \"N\":\n                    break\n                self.i += 1",
       "                if t2.kind == \"N\" or (t2.kind == \"W\" and t2.word.raw in _CLOSERS):\n"
       "                    break\n                self.i += 1")]),
    ("MED `done` as an argument closes the loop",
     [("                if t2.kind == \"W\":\n                    words.append(t2.word)",
       "                if t2.kind == \"W\" and words and t2.word.raw in _CLOSERS:\n                    break\n"
       "                if t2.kind == \"W\":\n                    words.append(t2.word)")]),
    ("MED `function name {` not a definition", [("            if r == \"function\":", "            if False:")]),
    ("MED one-line function body not a definition",
     [("            if cmd[\"t\"] == \"func\" and cmd[\"name\"]:",
       "            if cmd[\"t\"] == \"func\" and cmd[\"name\"] and not (cmd[\"body\"] and cmd[\"body\"].get(\"body\")"
       " and cmd[\"body\"][\"body\"][0][0][0][\"line\"] == cmd[\"line\"]):")]),
    ("MED command substitutions not walked ($(probe), \"$(ssh ..)\")",
     [("            for kind, toks in w.subs:\n                self._walk_list(",
       "            for kind, toks in []:\n                self._walk_list(")]),
    ("MED FP read -u N treated as fd 0",
     [("                if ch == \"u\":\n                    return arg == \"0\"",
       "                if ch == \"u\":\n                    return True")]),
    ("MED FP read <&N treated as fd 0",
     [("        if _fd0_redirected(cmd.get(\"redirs\") or ()):\n            return False\n        t = cmd[\"t\"]\n"
       "        if t == \"simple\":",
       "        t = cmd[\"t\"]\n        if t == \"simple\":")]),
    ("MED FP only a bare -n protects (-nT, -f, -fN)",
     [("                    if ch in \"nf\":\n                        return True",
       "                    if v == \"-n\":\n                        return True")]),
    ("MED FP group redirect ignored",
     [("        elif t == \"grp\":\n            self._walk_list(cmd[\"body\"], post, sink)",
       "        elif t == \"grp\":\n            self._walk_list(cmd[\"body\"], state, sink)")]),
    ("MED FP `|` line continuation ends the pipeline",
     [("        while self.peek().kind == \"O\" and self.peek().val in (\"|\", \"|&\"):\n            self.i += 1\n"
       "            self.skip_nl()\n",
       "        while self.peek().kind == \"O\" and self.peek().val in (\"|\", \"|&\"):\n            self.i += 1\n")]),
    ("MED FP quoted ; ends the statement",
     [("            if c == '\"':\n                return j + 1\n            if c == \"$\":\n"
       "                j = self._dollar(j, shape, subs, True)",
       "            if c == '\"' or c == \";\":\n                return j + 1\n            if c == \"$\":\n"
       "                j = self._dollar(j, shape, subs, True)")]),
    ("M5 nested for loses the read-loop state",
     [("            self._walk_subs(cmd[\"words\"], post, sink)\n            self._walk_list(cmd[\"body\"], post, sink)",
       "            self._walk_subs(cmd[\"words\"], post, sink)\n"
       "            self._walk_list(cmd[\"body\"], \"inherit\" if post == \"loop\" else post, sink)")]),
    ("LOW docstring not raw",
     [("r\"\"\"ssh_stdin_in_read_loop (class SR", "\"\"\"ssh_stdin_in_read_loop (class SR")]),
    ("callee regression not reported to unstaged callers",
     [("            if rel in cand_set or (staged_mode and kind == \"script\" and tok in cand_bases):",
       "            if rel in cand_set:")]),
    ("rsync flagged (would block correct code)",
     [("    return kind == \"ssh\" or (kind == \"script\" and name in known)",
       "    return kind in (\"ssh\", \"rsync\") or (kind == \"script\" and name in known)")]),
    ("ForkAfterAuthentication=yes treated as StdinNull",
     [("re.compile(r\"(?i)^\\s*stdinnull\\s*(=\\s*|\\s+)(yes|true)\\s*$\")",
       "re.compile(r\"(?i)^\\s*(stdinnull|forkafterauthentication)\\s*(=\\s*|\\s+)(yes|true)\\s*$\")")]),
    ("unseen $SSH value not treated as ssh",
     [("                if ref is not None and _SSH_VAR_RE.search(ref[0]):", "                if False:")]),
]


def _run_tests():
    for f in glob.glob(os.path.join(ROOT, "tools", "gates", "__pycache__", "ssh_stdin_in_read_loop*")):
        os.remove(f)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TESTS, "-rf", "--no-header"],
                       cwd=ROOT, capture_output=True, text=True, env=env, timeout=900)
    failed = [ln.split("FAILED ", 1)[1].split(" - ")[0].split("::", 1)[-1] for ln in r.stdout.splitlines()
              if ln.startswith("FAILED ")]
    tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
    return r.returncode, failed, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(GATE, encoding="utf-8") as fh:
        orig = fh.read()
    orig_sha = hashlib.sha256(orig.encode()).hexdigest()
    results = []
    try:
        for label, reps in MUTATIONS:
            src = orig
            for old, new in reps:
                n = src.count(old)
                if n != 1:
                    raise SystemExit("mutation %r: pattern found %d times: %r" % (label, n, old[:80]))
                src = src.replace(old, new)
            with open(GATE, "w", encoding="utf-8") as fh:
                fh.write(src)
            rc, failed, tail = _run_tests()
            results.append({"mutation": label, "suite_red": rc != 0, "n_failed": len(failed), "failed": failed[:6],
                            "tail": tail})
            print("%-62s %s %3d failed" % (label, "RED  " if rc else "GREEN", len(failed)), flush=True)
    finally:
        with open(GATE, "w", encoding="utf-8") as fh:
            fh.write(orig)
    with open(GATE, encoding="utf-8") as fh:
        assert hashlib.sha256(fh.read().encode()).hexdigest() == orig_sha, "the gate was NOT restored"
    rc, _failed, tail = _run_tests()
    print("restored:", tail)
    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out = {
        "provenance": {"cmd": "python tools/ssh_stdin_gate_mutations.py --out " + args.out,
                       "script": "tools/ssh_stdin_gate_mutations.py", "git_sha": git_sha, "gate_sha256": orig_sha,
                       "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
        "mutations": len(results),
        "mutations_caught": sum(1 for r in results if r["suite_red"]),
        "restored_suite_green": rc == 0,
        "restored_suite": tail,
        "results": results,
    }
    with open(os.path.join(ROOT, args.out), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    return 0 if all(r["suite_red"] for r in results) and rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
