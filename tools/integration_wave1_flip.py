#!/usr/bin/env python3
"""Apply or revert an integration-wave-1 default flip, mechanically (no judgment needed by whoever runs it).

A flip is exactly these text edits, all marked so `revert` restores the files byte-for-byte:
  1. the packet's anchor constant in source: `<ANCHOR> = False` -> `<ANCHOR> = True`;
  2. its opt-in LBF row module (if any): the `REQUIRED_ENV = {...}` line is commented out, so the row joins the
     headline registry at the production default;
  3. docs/PRODUCTION_INTEGRATION_LEDGER.yaml: the packet's ledger row gains a `default_anchor` entry for the anchor
     (so tools/gates/production_integration.py Check A pins the source to the ledger from then on) and a
     `subflag_flip_iw1_<slug>` note. The row's own levels are not changed (each parent row is already
     on_by_default: YES; the flipped mechanism is a deeper sub-rung of it).

  python tools/integration_wave1_flip.py status
  python tools/integration_wave1_flip.py apply  d6-hebbian-write [wm-focus-bind ...] [--date YYYY-MM-DD]
  python tools/integration_wave1_flip.py revert d6-hebbian-write [...]

It refuses (exit 2) to apply a packet that is already on, or to revert one that is off, and it never commits: the
HANDOFF gives the commit command. Tests: tests/test_integration_wave1_anchors.py.
"""
import argparse
import ast
import datetime
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = "docs/PRODUCTION_INTEGRATION_LEDGER.yaml"

PACKETS = {
    "d6-hebbian-write": dict(
        slug="d6_hebbian_write", anchor="_D6_LEARN_DEFAULT_ON", anchor_file="research/runners/d6_hebbian_store.py",
        ledger_row="in-loop-learning", row_module="research/runners/lbf_rows/d6_hebbian_write.py",
        prereg="research/findings/2026-09-29-iw1-d6-hebbian-write-flip-PREREGISTRATION.md",
        flags=("BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_ENGRAM_VOCAB", "BRAIN_D6_ENGRAM_READTIME"),
        note="BRAIN_D6_HEBBIAN_STORE + BRAIN_D6_ENGRAM_VOCAB + BRAIN_D6_ENGRAM_READTIME flipped default-ON together "
             "(one anchor, integration wave 1): an in-conversation fact is written by the local Hebbian rule and "
             "every routed kb reader sees only engrams that reactivate. Explicit =0 on a flag is the escape. "
             "Validation per the prereg; LBF row d6-hebbian-write joins the headline registry."),
    "wm-focus-bind": dict(
        slug="wm_focus_bind", anchor="_MULTIREF_FOCUS_BIND_DEFAULT_ON",
        anchor_file="research/runners/d6_multiref_wm_production_organ.py", ledger_row="wm-binding-advanced",
        row_module=None, prereg="research/findings/2026-09-29-iw1-wm-focus-bind-flip-PREREGISTRATION.md",
        flags=("BRAIN_MULTIREF_FOCUS_BIND",),
        note="BRAIN_MULTIREF_FOCUS_BIND flipped default-ON (integration wave 1): an anaphor after a >=2-referent intro "
             "resolves to the referent a spiking focus-WTA retrieves from the held registers, not the positional "
             "CAND_POOLS[0]. Explicit =0 is the escape. Validation per the prereg (LB_WMB_FOCUS_PROBE, reply "
             "content)."),
    "xedge-surprise-worldmodel": dict(
        slug="xedge_surprise_worldmodel", anchor="_XEDGE_SWM_DEFAULT_ON",
        anchor_file="research/runners/onebrain_xedge_surprise_worldmodel_production.py",
        ledger_row="worldmodel-forward", row_module="research/runners/lbf_rows/xedge_surprise_worldmodel.py",
        prereg="research/findings/2026-09-29-iw1-xedge-surprise-worldmodel-flip-PREREGISTRATION.md",
        flags=("BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL",),
        note="BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL flipped default-ON (integration wave 1): on a turn that violates "
             "the affective forward model's prediction, its own D2 prediction-error unit gates one Hebbian co-fire step "
             "of the live state->pred transition toward the observed valence (cross-edge C1), so the model is no "
             "longer frozen after build. The organ is process-shared: learning accumulates across sessions of one "
             "server process. Explicit =0 is the escape. Validation per the prereg; LBF row "
             "xedge-surprise-worldmodel joins the headline registry."),
}


def _mark(faculty, end=False):
    return "# %s iw1-flip:%s" % ("<<<" if end else ">>>", faculty)


def files_touched():
    out = {LEDGER}
    for p in PACKETS.values():
        out.add(p["anchor_file"])
        if p["row_module"]:
            out.add(p["row_module"])
    return sorted(out)


def _rw(root, rel, fn):
    path = os.path.join(root, rel)
    src = open(path, encoding="utf-8").read()
    new = fn(src)
    if new != src:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(new)


def _anchor_value(root, spec):
    tree = ast.parse(open(os.path.join(root, spec["anchor_file"]), encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == spec["anchor"]
                                                for t in node.targets):
            return ast.literal_eval(node.value)
    raise SystemExit("integration_wave1_flip: anchor %s not found in %s" % (spec["anchor"], spec["anchor_file"]))


def status(root=ROOT):
    return {f: ("on" if _anchor_value(root, s) is True else "off") for f, s in PACKETS.items()}


def _set_anchor(src, anchor, value):
    pat = re.compile(r"^%s = (True|False)$" % re.escape(anchor), re.M)
    hits = pat.findall(src)
    if len(hits) != 1:
        raise SystemExit("integration_wave1_flip: expected exactly one `%s = True|False` line, found %d"
                         % (anchor, len(hits)))
    return pat.sub("%s = %s" % (anchor, value), src)


_REQ_RE = re.compile(r"^(REQUIRED_ENV = \{.*\})$", re.M)
_REQ_OFF_PREFIX = "# REQUIRED_ENV lifted by the iw1 flip (tools/integration_wave1_flip.py): "


def _lift_required_env(src):
    if len(_REQ_RE.findall(src)) != 1:
        raise SystemExit("integration_wave1_flip: expected exactly one REQUIRED_ENV line")
    return _REQ_RE.sub(lambda m: _REQ_OFF_PREFIX + m.group(1), src)


def _restore_required_env(src):
    return re.sub(r"^%s(REQUIRED_ENV = \{.*\})$" % re.escape(_REQ_OFF_PREFIX), r"\1", src, flags=re.M)


def _row_bounds(lines, key):
    start = None
    for i, ln in enumerate(lines):
        if ln.rstrip("\n") == "  - key: %s" % key:
            start = i
            break
    if start is None:
        raise SystemExit("integration_wave1_flip: ledger row %r not found" % key)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("  - key: "):
            end = j
            break
    return start, end


def _ledger_apply(src, faculty, spec, date):
    lines = src.splitlines(True)
    start, end = _row_bounds(lines, spec["ledger_row"])
    obd = [i for i in range(start, end) if lines[i].startswith("    on_by_default:")]
    if len(obd) != 1:
        raise SystemExit("integration_wave1_flip: row %r has %d on_by_default lines" % (spec["ledger_row"], len(obd)))
    note = "%s (%s; prereg %s)" % (spec["note"], date, spec["prereg"])
    note_block = ["    %s\n" % _mark(faculty),
                  "    subflag_flip_iw1_%s: %s\n" % (spec["slug"], _yaml_str(note)),
                  "    %s\n" % _mark(faculty, end=True)]
    item = ["      - file: %s\n" % spec["anchor_file"],
            "        assign: %s\n" % spec["anchor"],
            '        off_value: "False"\n',
            '        on_value: "True"\n',
            "        count: 1\n"]
    da = [i for i in range(start, end) if lines[i].rstrip("\n").startswith("    default_anchor:")]
    if len(da) > 1:
        raise SystemExit("integration_wave1_flip: row %r has %d default_anchor keys" % (spec["ledger_row"], len(da)))
    if da:
        ins = ["      %s\n" % _mark(faculty)] + item + ["      %s\n" % _mark(faculty, end=True)]
        lines[da[0] + 1:da[0] + 1] = ins
        lines[obd[0] + 1:obd[0] + 1] = note_block
    else:
        block = note_block[:2] + ["    default_anchor:\n"] + item + note_block[2:]
        lines[obd[0] + 1:obd[0] + 1] = block
    return "".join(lines)


def _ledger_revert(src, faculty):
    out, skip = [], False
    for ln in src.splitlines(True):
        s = ln.strip()
        if s == _mark(faculty):
            skip = True
            continue
        if s == _mark(faculty, end=True):
            skip = False
            continue
        if not skip:
            out.append(ln)
    if skip:
        raise SystemExit("integration_wave1_flip: unterminated iw1-flip block for %s in the ledger" % faculty)
    return "".join(out)


def _yaml_str(s):
    return '"%s"' % s.replace("\\", "\\\\").replace('"', '\\"')


def apply(root, faculties, date=None):
    date = date or datetime.date.today().isoformat()
    st = status(root)
    for f in faculties:
        if f not in PACKETS:
            raise SystemExit("integration_wave1_flip: unknown packet %r (known: %s)" % (f, sorted(PACKETS)))
        if st[f] == "on":
            raise SystemExit("integration_wave1_flip: %s is already on; refusing a double apply" % f)
    for f in faculties:
        spec = PACKETS[f]
        _rw(root, spec["anchor_file"], lambda s: _set_anchor(s, spec["anchor"], "True"))
        if spec["row_module"]:
            _rw(root, spec["row_module"], _lift_required_env)
        _rw(root, LEDGER, lambda s: _ledger_apply(s, f, spec, date))


def revert(root, faculties):
    st = status(root)
    for f in faculties:
        if f not in PACKETS:
            raise SystemExit("integration_wave1_flip: unknown packet %r" % f)
        if st[f] == "off":
            raise SystemExit("integration_wave1_flip: %s is already off; nothing to revert" % f)
    for f in faculties:
        spec = PACKETS[f]
        _rw(root, spec["anchor_file"], lambda s: _set_anchor(s, spec["anchor"], "False"))
        if spec["row_module"]:
            _rw(root, spec["row_module"], _restore_required_env)
        _rw(root, LEDGER, lambda s: _ledger_revert(s, f))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    for name in ("apply", "revert"):
        p = sub.add_parser(name)
        p.add_argument("faculties", nargs="+", choices=sorted(PACKETS))
        if name == "apply":
            p.add_argument("--date", default=None)
    ap.add_argument("--root", default=ROOT)
    a = ap.parse_args(argv)
    if a.cmd == "status":
        for f, v in status(a.root).items():
            print("%-26s %s" % (f, v))
        return 0
    try:
        if a.cmd == "apply":
            apply(a.root, a.faculties, a.date)
        else:
            revert(a.root, a.faculties)
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2
    for f, v in status(a.root).items():
        print("%-26s %s" % (f, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
