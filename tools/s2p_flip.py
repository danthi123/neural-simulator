#!/usr/bin/env python3
"""s2p_flip.py -- the scripted default flip for the session-2 host-decision packets (task T21, s2p-flip).

    python tools/s2p_flip.py status
    python tools/s2p_flip.py apply  fb-spiking-witness=4 transitive-spiking-reach=0 [word-role-parse=<op>]
    python tools/s2p_flip.py revert fb-spiking-witness [transitive-spiking-reach ...]

`apply <packet>=<op>` does three literal edits per packet and nothing else:
  1. the packet's DEFAULT_ON anchor False -> True (the spiking path becomes the faculty's default);
  2. its DEFAULT_OP constant -> <op>, the operating point that packet's step 1 selected (selection.json);
  3. its opt-in LBF row module: the one `REQUIRED_ENV = {...}` line is commented out (the integration-wave-2 prefix
     convention), so the capability and control rows join the headline battery.
`revert` undoes all three. Each edit must match exactly once, or the tool refuses (exit 2) and writes nothing.
None of these anchors is a docs/PRODUCTION_INTEGRATION_LEDGER.yaml default_anchor, so the production_integration
gate's sub-check A does not read them. The faculties' ledger rows (false-belief, transitive, neural-selfid) record the
host decision each flip removes; that note is Claude's at merge (HANDOFF section 5).
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKETS = {
    "fb-spiking-witness": ("webapp/false_belief_chat.py", "_FALSE_BELIEF_SPIKING_WITNESS_DEFAULT_ON",
                           "_FALSE_BELIEF_WITNESS_DEFAULT_OP", "research/runners/lbf_rows/fb_spiking_witness.py"),
    "transitive-spiking-reach": ("webapp/reasoning_transitive_chat.py", "_TRANSITIVE_SPIKING_REACH_DEFAULT_ON",
                                 "_TRANSITIVE_REACH_DEFAULT_OP", "research/runners/lbf_rows/transitive_spiking_reach.py"),
    "word-role-parse": ("research/runners/brain_chat_tui.py", "_WORD_ROLE_PARSE_DEFAULT_ON", "_WORD_ROLE_DEFAULT_OP",
                        "research/runners/lbf_rows/word_role_parse.py"),
}
REQ_PREFIX = "# REQUIRED_ENV lifted by the s2p flip (tools/s2p_flip.py): "


def _sub1(src, pattern, repl, what):
    new, n = re.subn(pattern, repl, src, flags=re.M)
    if n != 1:
        raise SystemExit("s2p_flip: expected exactly one %s, found %d" % (what, n))
    return new


def _read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def plan(action, packet, op=None):
    """Return {relpath: new_source} for one packet (nothing written)."""
    mod, anchor, opname, rows = PACKETS[packet]
    src, rsrc = _read(mod), _read(rows)
    if action == "apply":
        src = _sub1(src, r"^(%s = )False\b" % re.escape(anchor), r"\1True", anchor + " = False")
        src = _sub1(src, r"^(%s = )\d+\b" % re.escape(opname), r"\g<1>%d" % int(op), opname)
        rsrc = _sub1(rsrc, r"^(REQUIRED_ENV = \{.*\})$", REQ_PREFIX.replace("\\", "\\\\") + r"\1", "REQUIRED_ENV line")
    else:
        src = _sub1(src, r"^(%s = )True\b" % re.escape(anchor), r"\1False", anchor + " = True")
        src = _sub1(src, r"^(%s = )\d+\b" % re.escape(opname), r"\g<1>0", opname)
        rsrc = _sub1(rsrc, r"^%s(REQUIRED_ENV = \{.*\})$" % re.escape(REQ_PREFIX), r"\1", "lifted REQUIRED_ENV line")
    return {mod: src, rows: rsrc}


def status():
    out = {}
    for p, (mod, anchor, opname, rows) in PACKETS.items():
        try:
            src = _read(mod)
        except FileNotFoundError:
            out[p] = {"on": None, "op": None, "rows_lifted": None}
            continue
        m = re.search(r"^%s = (True|False)\b" % re.escape(anchor), src, re.M)
        o = re.search(r"^%s = (\d+)\b" % re.escape(opname), src, re.M)
        lifted = REQ_PREFIX in _read(rows)
        out[p] = {"on": bool(m and m.group(1) == "True"), "op": int(o.group(1)) if o else None, "rows_lifted": lifted}
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("status", "apply", "revert"):
        print(__doc__)
        return 2
    if argv[0] == "status":
        for p, st in status().items():
            print("%s: %s op=%s rows_lifted=%s" % (p, "on" if st["on"] else "off", st["op"], st["rows_lifted"]))
        return 0
    edits = {}
    try:
        for a in argv[1:]:
            if argv[0] == "apply":
                p, _, op = a.partition("=")
                if p not in PACKETS or not op.isdigit():
                    raise SystemExit("s2p_flip: apply takes <packet>=<op>, got %r" % a)
                edits.update(plan("apply", p, int(op)))
            else:
                if a not in PACKETS:
                    raise SystemExit("s2p_flip: unknown packet %r" % a)
                edits.update(plan("revert", a))
    except SystemExit as e:
        print(e, file=sys.stderr)
        return 2
    if not edits:
        print("s2p_flip: no packets named", file=sys.stderr)
        return 2
    for rel, src in edits.items():
        with open(os.path.join(ROOT, rel), "w") as f:
            f.write(src)
    for p, st in status().items():
        print("%s: %s op=%s rows_lifted=%s" % (p, "on" if st["on"] else "off", st["op"], st["rows_lifted"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
