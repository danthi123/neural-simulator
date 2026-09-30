#!/usr/bin/env python3
"""Guard for the integration-wave-2 production-default batteries (tag iw2-flip-<F7> and its attribution runs).

Same contract as tools/assert_iw1_defaults.py (branch research/integration-wave1), for the wave-2 anchors. A
production-default battery passes NO BRAIN_* flag, so it measures whatever the checked-out code defaults to; a job
dispatched from the wrong revision would silently measure the wrong defaults and still look valid (the 2026-09-23
lesson, tools/assert_flipped_defaults.py). Every job line of those batteries therefore starts with

    .venv/bin/python tools/assert_iw2_defaults.py --expect-on=<comma-separated packet names> &&

which exits 1 unless
  * each wave-2 anchor named in --expect-on reads True in the SOURCE of this checkout, and every other wave-2 anchor
    reads False (read with ast, never imported: CPU-only nodes lack cupy), and
  * no BRAIN_* variable is set in the environment (a production-default battery sets none; LB_* probe flags, which
    change how the metric probes and not the brain, are allowed).
An empty --expect-on ("") is the pre-flip state. Wave-1 anchors are NOT checked here: whether wave 1's flips are in
this revision is whatever main holds at P (the shared prereg's "Interaction with wave 1").

Packets (see tools/integration_wave2_flip.py PACKETS): false-belief-chat, transitive-chat, neural-selfid.
"""
import argparse
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANCHORS = {
    "false-belief-chat": ("webapp/false_belief_chat.py", "_FALSE_BELIEF_CHAT_DEFAULT_ON"),
    "transitive-chat": ("webapp/reasoning_transitive_chat.py", "_TRANSITIVE_CHAT_DEFAULT_ON"),
    "neural-selfid": ("research/runners/brain_chat_tui.py", "_NEURAL_SELFID_DEFAULT_ON"),
}
_MISSING = object()


def source_constant(path, const):
    try:
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    except (OSError, SyntaxError):
        return _MISSING
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == const for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except ValueError:
                return _MISSING
    return _MISSING


def problems(expect_on, environ=None, root=ROOT):
    environ = os.environ if environ is None else environ
    out = []
    unknown = [p for p in expect_on if p not in ANCHORS]
    if unknown:
        out.append("unknown packet name(s) in --expect-on: %s (known: %s)" % (unknown, sorted(ANCHORS)))
    for k in sorted(environ):
        if k.startswith("BRAIN_"):
            out.append("%s is set in the environment (=%r); a production-default battery sets no BRAIN_* flag"
                       % (k, environ[k]))
    for name, (rel, const) in ANCHORS.items():
        v = source_constant(os.path.join(root, rel), const)
        want = name in expect_on
        if v is _MISSING:
            out.append("%s: constant %s not found in %s (wrong revision?)" % (name, const, rel))
        elif v is not want:
            out.append("%s: %s in %s reads %r, this battery expects %r" % (name, const, rel, v, want))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--expect-on", required=True,
                    help='comma-separated packet names whose anchor must read True ("" = none, the pre-flip state)')
    a = ap.parse_args(argv)
    expect = [p.strip() for p in a.expect_on.split(",") if p.strip()]
    probs = problems(expect)
    if probs:
        for p in probs:
            print("assert_iw2_defaults: " + p, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
