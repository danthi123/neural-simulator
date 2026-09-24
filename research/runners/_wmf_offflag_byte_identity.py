"""Byte-identity assertion (IN DATA) for BRAIN_MULTIREF_FOCUS_BIND + LB_WMB_FOCUS_PROBE, both OFF, vs a pinned
pre-change tree. Sibling of `_wmb_offflag_byte_identity.py` (same kind of check: exact sha256 of every response the
real `webapp.server.brain_chat` returns, pinned tree vs branch tree, with the flags unset, plus the probe-roster hashes).

Coverage: every code path this change touches with the flags OFF -- the D6 MAINTAIN load + the xedge focus that
comprehension reads (`hold`/`held`), an anaphor turn over two held referents through `ChatBrain._resolve_anaphora`
and the GNW bus's anaphora branches (`bc`), the hold-query read-out (`wmb`), and one of the probe's own order-swapped
anaphor sessions (so the probe's OFF arm is shown to take the pre-change route). Messages
are given literally (not by battery label), so the pinned tree, which lacks the new labels, runs the same text.

  # collect, once per tree (cwd AND PYTHONPATH = that tree; every flag below unset):
  cd <tree> && PYTHONPATH=<tree> BRAIN_CHAT_SEED=<dev seed> SIM_BACKEND=numpy tools/memcap.sh 10 -- \\
      .venv/bin/python -u <branch>/research/runners/_wmf_offflag_byte_identity.py --collect --part 1 --out <tree_p1.json>
  # ... and again with --part 2 (two sessions per part, one process each)
  # compare (no brain build):
  .venv/bin/python -m research.runners._wmf_offflag_byte_identity --compare --pinned <p.json> --branch <b.json> \\
      --pinned-tree <T1> --branch-tree <T2> --pinned-sha <sha> --branch-sha <sha> --out <artifact.json>
Pre-registration: research/findings/2026-09-24-wm-referent-focus-bind-anaphor-probe-PREREGISTRATION.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

# (session, [messages]) -- the first message of each session resets it
SCRIPT = [
    ("bi_hold", ["the fox and the wolf walked in", "the wolf watches the owl"]),
    ("bi_bc", ["the cat and the ball walked in", "what does it eat"]),
    ("bi_wmb", ["the fox and the wolf walked in", "who are we talking about"]),
    ("bi_wmfa1", ["the dog and the cat walked in", "what does it chase"]),
]
# collected in two parts of two sessions each: every session builds its own ChatBrain + composer, and a 4-session
# process was measured past 15 GB on numpy (2026-09-24, dev seed 7); two sessions stay within one pool job budget.
PARTS = {"1": ["bi_hold", "bi_bc"], "2": ["bi_wmb", "bi_wmfa1"]}
FLAGS_UNSET = ("BRAIN_MULTIREF_FOCUS_BIND", "LB_WMB_FOCUS_PROBE", "LB_WMB_CONTENT_PROBE", "LB_WMB_HOLDQUERY_PROBE",
               "BRAIN_MULTIREF_LESION_SCOPE", "BRAIN_MULTIREF_LESION")

_ROSTER_SNIPPET = (
    "import json,hashlib;from research.runners.onebrain_regression_battery import PROBE_TURNS,FACULTY_PROBES,"
    "_TURN_BY_LABEL;h=lambda o:hashlib.sha256(json.dumps(o,sort_keys=True,default=str).encode()).hexdigest();"
    "print(json.dumps({'PROBE_TURNS':h(PROBE_TURNS),'FACULTY_PROBES':h(FACULTY_PROBES),"
    "'labels':sorted(_TURN_BY_LABEL),'turn_by_label':{k:h(v) for k,v in _TURN_BY_LABEL.items()}}))")


def _canon(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()


def collect(out, part="all"):
    # run against whichever tree is on PYTHONPATH (cwd): drop this file's own directory from sys.path first
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    for k in FLAGS_UNSET:
        if k in os.environ:
            raise SystemExit("flag %s is set -- the byte-identity check needs it UNSET" % k)
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    from webapp.server import brain_chat, BrainChatRequest
    import webapp.server as _S
    res = {"tree_webapp": os.path.abspath(_S.__file__), "seed": os.environ.get("BRAIN_CHAT_SEED"), "turns": {}}
    keep = set(PARTS[part]) if part in PARTS else {sess for sess, _m in SCRIPT}
    res["part"] = part
    for sess, msgs in SCRIPT:
        if sess not in keep:
            continue
        for i, m in enumerate(msgs):
            lab = "%s_%d" % (sess, i)
            try:
                r = brain_chat(BrainChatRequest(session=sess, message=m, brain="tiny-demo", renderer="stub",
                                                rich=False, reset=(i == 0)))
                res["turns"][lab] = {"message": m, "response": json.loads(r.body)}
            except Exception as e:
                res["turns"][lab] = {"message": m, "_error": "%s: %s" % (type(e).__name__, e)}
            print("[collect] %s %r -> %r" % (lab, m, res["turns"][lab].get("response", {}).get("answer")),
                  flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, default=str)
    return 0


def _roster(tree):
    out = subprocess.run([sys.executable, "-c", _ROSTER_SNIPPET], cwd=tree, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": tree, "SIM_NO_PROVENANCE": "1"})
    return json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {"error": out.stderr[-500:]}


def _load_parts(paths):
    merged = {"turns": {}, "seed": None, "parts": []}
    for f in paths.split(","):
        d = json.load(open(f))
        merged["turns"].update(d["turns"])
        merged["parts"].append(d.get("part"))
        if merged["seed"] is None:
            merged["seed"] = d.get("seed")
        elif merged["seed"] != d.get("seed"):
            merged["seed"] = "MISMATCH"
    return merged


def compare(a):
    p, b = _load_parts(a.pinned), _load_parts(a.branch)
    turns = {}
    for lab in sorted(set(p["turns"]) | set(b["turns"])):
        tp, tb = p["turns"].get(lab), b["turns"].get(lab)
        ok = bool(tp is not None and tb is not None and "_error" not in tp and "_error" not in tb
                  and _canon(tp) == _canon(tb))
        turns[lab] = {"identical": ok, "pinned": _canon(tp) if tp else None, "branch": _canon(tb) if tb else None,
                      "answer_pinned": (tp or {}).get("response", {}).get("answer"),
                      "answer_branch": (tb or {}).get("response", {}).get("answer")}
    rp, rb = _roster(a.pinned_tree), _roster(a.branch_tree)
    old = set(rp.get("labels") or [])
    roster = {"PROBE_TURNS_identical": rp.get("PROBE_TURNS") == rb.get("PROBE_TURNS") and "error" not in rp,
              "FACULTY_PROBES_identical": rp.get("FACULTY_PROBES") == rb.get("FACULTY_PROBES"),
              "pre_existing_labels_identical": bool(old) and all(
                  rp["turn_by_label"][k] == (rb.get("turn_by_label") or {}).get(k) for k in old),
              "added_labels": sorted(set(rb.get("labels") or []) - old),
              "removed_labels": sorted(old - set(rb.get("labels") or []))}
    same_seed = p.get("seed") == b.get("seed") and p.get("seed") not in (None, "MISMATCH")
    expected = {"%s_%d" % (sess, i) for sess, msgs in SCRIPT for i in range(len(msgs))}
    complete = set(turns) == expected
    identical = (all(t["identical"] for t in turns.values()) and roster["PROBE_TURNS_identical"]
                 and roster["FACULTY_PROBES_identical"] and roster["pre_existing_labels_identical"]
                 and not roster["removed_labels"] and same_seed and complete)
    art = {"runner": "research.runners._wmf_offflag_byte_identity", "pinned_sha": a.pinned_sha,
           "branch_sha": a.branch_sha, "seed": p.get("seed"), "same_seed": same_seed,
           "flags_unset": list(FLAGS_UNSET), "compare": "exact sha256 of each canonical brain_chat response",
           "n_turns": len(turns), "complete_script": complete, "turns": turns, "roster": roster, "byte_identical_off": identical}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(art, open(a.out, "w"), indent=2)
    print(json.dumps({"byte_identical_off": identical, "n_turns": len(turns),
                      "turns_identical": sum(t["identical"] for t in turns.values()), "roster": roster}))
    return 0 if identical else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="all", choices=["1", "2", "all"])
    ap.add_argument("--pinned", help="collect file(s), comma-separated parts")
    ap.add_argument("--branch")
    ap.add_argument("--pinned-tree")
    ap.add_argument("--branch-tree")
    ap.add_argument("--pinned-sha", default="")
    ap.add_argument("--branch-sha", default="")
    a = ap.parse_args()
    if a.collect:
        return collect(a.out, a.part)
    return compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
