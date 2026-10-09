#!/usr/bin/env python3
"""Guard for the integration-wave-1 production-default batteries (tags iw1-base-<P7> and iw1-flip-<F7>).

Those batteries pass NO BRAIN_* flag, so they measure whatever the checked-out code defaults to. A job dispatched from
the wrong revision would silently measure the wrong defaults and still look valid (the 2026-09-23 lesson,
tools/assert_flipped_defaults.py). Every job line of those batteries therefore starts with

    .venv/bin/python tools/assert_iw1_defaults.py --expect-on "<comma-separated packet names>" &&

which exits 1 unless
  * each wave-1 anchor named in --expect-on reads True in the SOURCE of this checkout, and every other wave-1 anchor
    reads False (read with ast, never imported: CPU-only nodes lack cupy), and
  * no BRAIN_* variable is set in the environment (a production-default battery sets none; LB_* probe flags, which
    change how the metric probes and not the brain, are allowed).
An empty --expect-on ("") is the pre-flip base battery.

Packets (see tools/integration_wave1_flip.py PACKETS): d6-hebbian-write, wm-focus-bind, xedge-surprise-worldmodel.
"""
import argparse
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANCHORS = {
    "d6-hebbian-write": ("research/runners/d6_hebbian_store.py", "_D6_LEARN_DEFAULT_ON"),
    "wm-focus-bind": ("research/runners/d6_multiref_wm_production_organ.py", "_MULTIREF_FOCUS_BIND_DEFAULT_ON"),
    "xedge-surprise-worldmodel": ("research/runners/onebrain_xedge_surprise_worldmodel_production.py",
                                  "_XEDGE_SWM_DEFAULT_ON"),
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
                    help='comma-separated packet names whose anchor must read True ("" = none, the pre-flip base)')
    a = ap.parse_args(argv)
    expect = [p.strip() for p in a.expect_on.split(",") if p.strip()]
    probs = problems(expect)
    if probs:
        for p in probs:
            print("assert_iw1_defaults: " + p, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
