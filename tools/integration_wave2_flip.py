#!/usr/bin/env python3
"""Apply or revert an integration-wave-2 default flip, mechanically (no judgment needed by whoever runs it).

Same shape as tools/integration_wave1_flip.py (branch research/integration-wave1); separate file and separate markers
(`iw2-flip:<packet>`, `subflag_flip_iw2_<slug>`) so the two waves never edit each other's blocks. A flip is exactly
these text edits, all marked so `revert` restores the files byte-for-byte:
  1. the packet's anchor constant in source: `<ANCHOR> = False` -> `<ANCHOR> = True`;
  2. its opt-in LBF row module: the `REQUIRED_ENV = {...}` line is commented out, so the row joins the headline
     registry at the production default;
  3. docs/PRODUCTION_INTEGRATION_LEDGER.yaml: the packet's parent ledger row gains a `default_anchor` entry for the
     anchor (so tools/gates/production_integration.py Check A pins the source to the ledger from then on) and a
     `subflag_flip_iw2_<slug>` note. The row's own levels are not changed (each parent row is already
     on_by_default: YES; the flipped mechanism is recorded as a sub-rung of it, as wave 1 did).

  python tools/integration_wave2_flip.py status
  python tools/integration_wave2_flip.py apply  false-belief-chat [transitive-chat neural-selfid] [--date YYYY-MM-DD]
  python tools/integration_wave2_flip.py revert transitive-chat [...]

It refuses (exit 2) to apply a packet that is already on, or to revert one that is off, and it never commits: the
HANDOFF gives the commit command. Tests: tests/test_integration_wave2_anchors.py.
"""
import argparse
import ast
import datetime
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = "docs/PRODUCTION_INTEGRATION_LEDGER.yaml"
WAVE = "iw2"

PACKETS = {
    "false-belief-chat": dict(
        slug="false_belief_chat", anchor="_FALSE_BELIEF_CHAT_DEFAULT_ON", anchor_file="webapp/false_belief_chat.py",
        ledger_row="w5-affective-tom", row_module="research/runners/lbf_rows/tom_false_belief_chat.py",
        prereg="research/findings/2026-09-30-iw2-false-belief-chat-flip-PREREGISTRATION.md",
        flags=("BRAIN_FALSE_BELIEF_CHAT",),
        note="BRAIN_FALSE_BELIEF_CHAT flipped default-ON (integration wave 2): a narrated change-of-location scenario "
             "(Sally-Anne) is folded onto the W3 agent-keyed false-belief register and 'where will <agent> look for "
             "the <object>?' is answered off the register's witnessing-gated belief store (the stale location when the "
             "agent did not see the move). Cognitive (false-belief) theory of mind, recorded under this ToM row as a "
             "sub-rung: the ledger has no separate cognitive-ToM row. Host residuals: witnessing/presence parse and the "
             "belief-location argmax. Explicit =0 is the escape. Validation per the prereg; LBF row tom-false-belief "
             "joins the headline registry."),
    "transitive-chat": dict(
        slug="transitive_chat", anchor="_TRANSITIVE_CHAT_DEFAULT_ON", anchor_file="webapp/reasoning_transitive_chat.py",
        ledger_row="gnw-multistep-deliberation", row_module="research/runners/lbf_rows/reasoning_transitive_chat.py",
        prereg="research/findings/2026-09-30-iw2-transitive-chat-flip-PREREGISTRATION.md",
        flags=("BRAIN_TRANSITIVE_CHAT",),
        note="BRAIN_TRANSITIVE_CHAT flipped default-ON (integration wave 2): a yes/no question about a never-told "
             "non-adjacent pair ('does A precede D?') is answered by the keystone re-entrant chase this row already "
             "runs (the substrate's own n_ignited read decides each hop), with an honest abstain when the chase does "
             "not reach the target. Host residuals: the question-shape regex and the yes/no template. Explicit =0 is "
             "the escape. Validation per the prereg; LBF row reasoning-transitive-chat joins the headline registry."),
    "neural-selfid": dict(
        slug="neural_selfid", anchor="_NEURAL_SELFID_DEFAULT_ON", anchor_file="research/runners/brain_chat_tui.py",
        ledger_row="content-selection", row_module="research/runners/lbf_rows/neural_selfid.py",
        prereg="research/findings/2026-09-30-iw2-neural-selfid-flip-PREREGISTRATION.md",
        flags=("BRAIN_NEURAL_SELFID",),
        note="BRAIN_NEURAL_SELFID flipped default-ON alone (integration wave 2; BRAIN_NEURAL_ANAPHORA_ABSTAIN stays "
             "OFF): a self-referential factual question ('what do you use?') goes through the on-brain BridgeParser "
             "and the substrate recall like any other factual question, and a bare identity question is answered by "
             "the substrate recall, so the host QuestionRouter no longer owns self/identity. The noisy-anaphora "
             "fallback is still host-owned. Explicit =0 is the escape. Validation per the prereg; LBF row "
             "neural-selfid joins the headline registry."),
}


def _mark(faculty, end=False):
    return "# %s %s-flip:%s" % ("<<<" if end else ">>>", WAVE, faculty)


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
    raise SystemExit("integration_wave2_flip: anchor %s not found in %s" % (spec["anchor"], spec["anchor_file"]))


def status(root=ROOT):
    return {f: ("on" if _anchor_value(root, s) is True else "off") for f, s in PACKETS.items()}


def _set_anchor(src, anchor, value):
    pat = re.compile(r"^%s = (True|False)$" % re.escape(anchor), re.M)
    hits = pat.findall(src)
    if len(hits) != 1:
        raise SystemExit("integration_wave2_flip: expected exactly one `%s = True|False` line, found %d"
                         % (anchor, len(hits)))
    return pat.sub("%s = %s" % (anchor, value), src)


_REQ_RE = re.compile(r"^(REQUIRED_ENV = \{.*\})$", re.M)
_REQ_OFF_PREFIX = "# REQUIRED_ENV lifted by the iw2 flip (tools/integration_wave2_flip.py): "


def _lift_required_env(src):
    if len(_REQ_RE.findall(src)) != 1:
        raise SystemExit("integration_wave2_flip: expected exactly one REQUIRED_ENV line")
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
        raise SystemExit("integration_wave2_flip: ledger row %r not found" % key)
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("  - key: "):
            end = j
            break
    return start, end


def _yaml_str(s):
    return '"%s"' % s.replace("\\", "\\\\").replace('"', '\\"')


def _ledger_apply(src, faculty, spec, date):
    lines = src.splitlines(True)
    start, end = _row_bounds(lines, spec["ledger_row"])
    obd = [i for i in range(start, end) if lines[i].startswith("    on_by_default:")]
    if len(obd) != 1:
        raise SystemExit("integration_wave2_flip: row %r has %d on_by_default lines" % (spec["ledger_row"], len(obd)))
    note = "%s (%s; prereg %s)" % (spec["note"], date, spec["prereg"])
    note_block = ["    %s\n" % _mark(faculty),
                  "    subflag_flip_%s_%s: %s\n" % (WAVE, spec["slug"], _yaml_str(note)),
                  "    %s\n" % _mark(faculty, end=True)]
    item = ["      - file: %s\n" % spec["anchor_file"],
            "        assign: %s\n" % spec["anchor"],
            '        off_value: "False"\n',
            '        on_value: "True"\n',
            "        count: 1\n"]
    da = [i for i in range(start, end) if lines[i].rstrip("\n").startswith("    default_anchor:")]
    if len(da) > 1:
        raise SystemExit("integration_wave2_flip: row %r has %d default_anchor keys" % (spec["ledger_row"], len(da)))
    if da:
        if lines[da[0]].rstrip("\n") != "    default_anchor:":
            # an inline (flow-style) list cannot take a block item; the three parent rows use block lists
            raise SystemExit("integration_wave2_flip: row %r's default_anchor is not a block list"
                             % spec["ledger_row"])
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
        raise SystemExit("integration_wave2_flip: unterminated %s-flip block for %s in the ledger" % (WAVE, faculty))
    return "".join(out)


def apply(root, faculties, date=None):
    date = date or datetime.date.today().isoformat()
    st = status(root)
    for f in faculties:
        if f not in PACKETS:
            raise SystemExit("integration_wave2_flip: unknown packet %r (known: %s)" % (f, sorted(PACKETS)))
        if st[f] == "on":
            raise SystemExit("integration_wave2_flip: %s is already on; refusing a double apply" % f)
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
            raise SystemExit("integration_wave2_flip: unknown packet %r" % f)
        if st[f] == "off":
            raise SystemExit("integration_wave2_flip: %s is already off; nothing to revert" % f)
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
            print("%-20s %s" % (f, v))
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
        print("%-20s %s" % (f, v))
    return 0


if __name__ == "__main__":
    sys.exit(main())
