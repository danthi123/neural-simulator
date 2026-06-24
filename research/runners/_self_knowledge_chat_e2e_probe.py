"""END-TO-END chat probe: the decorrelated-grounded-codes FIX through the FULL deployed path
(BrainConversationalAgent + the demo's firewall batteries), CPU, in seconds.

Confirms (1) recall >=0.8 at 52 facts through the agent's what_does (the deployed recall), and (2) the
FIREWALL still holds (0 general leaks, 0 untaught leaks) -- the decorrelation must not weaken the moat.

This imports the demo's own firewall machinery (`run_firewall`, `build_qa_agent`, `_qa_vocab`,
`_all_facts_svo`, `_load_curriculum`) so the path is byte-identical to the real demo, except the grounded
codes are decorrelated before they enter the composer.

    SIM_BACKEND=numpy python -u -m research.runners._self_knowledge_chat_e2e_probe
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._self_knowledge_demo import (  # noqa: E402
    _load_curriculum, _all_facts_svo, _qa_vocab, build_qa_agent, run_firewall,
)
from research.runners._self_knowledge_recall_probe import _decorrelate_grounded, _concept_set  # noqa: E402

GROUNDED = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_grounded_codes.json")
OUT = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_chat_e2e_probe.json")


def _load_grounded():
    with open(GROUNDED, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    return {w: np.asarray(v, dtype=float) for w, v in blob.get("grounded_codes", {}).items()}


def _agent_recall(agent, facts):
    """Deployed recall through the agent's what_does (the firewall path's recall layer). Returns acc + misses."""
    n_ok, misses = 0, []
    for a, v, p in facts:
        got = agent.what_does(a, v)
        if got == p:
            n_ok += 1
        else:
            misses.append((a, v, p, got))
    return n_ok / len(facts), misses


def run(codes_kind, seed=42):
    cur = _load_curriculum()
    facts = _all_facts_svo(cur)
    vocab = _qa_vocab(cur)
    action_words = {v for (_a, v, _p) in facts}
    grounded = _load_grounded()
    cset = _concept_set(cur)
    if codes_kind == "grounded_decorr":
        g = _decorrelate_grounded({w: ph for w, ph in grounded.items() if w in cset}, cset, seed)
    elif codes_kind == "grounded":
        g = grounded
    else:
        g = None
    agent, n_taught = build_qa_agent(cur, vocab, g, seed)
    rec_acc, misses = _agent_recall(agent, facts)
    fw = run_firewall(agent, cur, action_words)
    return {
        "codes": codes_kind, "n_taught": n_taught,
        "agent_recall_acc": round(rec_acc, 4), "agent_recall_misses": misses[:12],
        "firewall_positive_answered": fw["positive_answered"], "firewall_positive_total": fw["positive_total"],
        "firewall_general_leaks": fw["general_leaks"], "firewall_general_total": fw["general_total"],
        "firewall_untaught_leaks": fw["untaught_leaks"], "firewall_untaught_total": fw["untaught_total"],
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)
    t0 = time.time()
    res = {"runs": []}
    for codes_kind in ("grounded", "grounded_decorr"):
        print(f"\n[e2e] codes={codes_kind} ...", flush=True)
        r = run(codes_kind)
        res["runs"].append(r)
        print(f"    agent recall {r['agent_recall_acc']:.2f} | "
              f"firewall: project {r['firewall_positive_answered']}/{r['firewall_positive_total']}  "
              f"general_leaks {r['firewall_general_leaks']}  untaught_leaks {r['firewall_untaught_leaks']}",
              flush=True)
    dc = next(r for r in res["runs"] if r["codes"] == "grounded_decorr")
    res["fix_recall_ge_0.8"] = bool(dc["agent_recall_acc"] >= 0.8)
    res["fix_firewall_intact"] = bool(dc["firewall_general_leaks"] == 0 and dc["firewall_untaught_leaks"] == 0)
    res["elapsed_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\n[saved] {OUT}", flush=True)
    print(f"[VERDICT] fix_recall>=0.8={res['fix_recall_ge_0.8']}  firewall_intact={res['fix_firewall_intact']}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
