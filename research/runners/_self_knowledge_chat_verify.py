"""VERIFY the chat-ready fix end-to-end on the SAVED grounded codes (no 50-min develop loop):
  - build the firewall agent with the decorrelation RECALL fix (build_qa_agent default),
  - run the firewall (recall + 0-FA moat),
  - load the off-bridge Qwen faculty WITH the crash fix (free cupy pool first),
  - render a few self-reflective Q&As through the faculty (the fluency),
  - answer a handful of chat questions exactly as the REPL would (chat()).

This is the real deployed path (the demo's own build_qa_agent / run_firewall / run_self_reflective / chat),
just driven non-interactively on the saved codes. GPU. FOREGROUND.

    SIM_BACKEND=cupy python -u -m research.runners._self_knowledge_chat_verify
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "cupy")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import _self_knowledge_demo as D  # noqa: E402

OUT = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_chat_verify.json")
CODES = os.path.join(_REPO, "research", "findings", "raw", "_self_knowledge_grounded_codes.json")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    t0 = time.time()
    cur = D._load_curriculum()
    facts = D._all_facts_svo(cur)
    action_words = {v for (_a, v, _p) in facts}
    vocab = D._qa_vocab(cur)
    with open(CODES, "r", encoding="utf-8") as fh:
        blob = json.load(fh)
    grounded = {w: np.asarray(v, dtype=float) for w, v in blob.get("grounded_codes", {}).items()}
    print(f"[verify] loaded {len(grounded)} saved grounded codes", flush=True)

    # build the agent WITH the recall fix (decorrelate_codes default True)
    agent, n_taught = D.build_qa_agent(cur, vocab, grounded, 42)
    rec_ok = sum(1 for a, v, p in facts if agent.what_does(a, v) == p)
    rec_acc = round(rec_ok / len(facts), 4)
    fw = D.run_firewall(agent, cur, action_words)
    print(f"[verify] agent recall {rec_acc:.2f} | firewall project {fw['positive_answered']}/{fw['positive_total']} "
          f"general_leaks {fw['general_leaks']} untaught_leaks {fw['untaught_leaks']}", flush=True)

    # load the faculty WITH the crash fix (free cupy pool first)
    D._free_cupy_pool()
    faculty = None
    faculty_err = None
    try:
        from research.runners._grounded_lang_integration_derisk import SpikingQwenFaculty
        import torch
        faculty = SpikingQwenFaculty(T=16, max_new_tokens=24, seed=42,
                                     device=("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"[verify] faculty loaded in {faculty.load_seconds}s", flush=True)
    except Exception as e:
        import traceback
        faculty_err = repr(e)
        traceback.print_exc()

    # self-reflective Q&As (faculty-phrased) + a few chat turns
    qas = D.run_self_reflective(agent, cur, faculty, action_words)
    chat_qs = ["what are you", "how do you learn", "what do you use", "what prevents confabulation",
               "what consolidates memory", "what prevents forgetting", "what is the capital of France",
               "what is two plus two"]
    chat_out = [{"q": q, "a": D.chat(agent, faculty, cur, action_words, q)} for q in chat_qs]
    for c in chat_out:
        print(f"[verify]   you> {c['q']}\n[verify]   brain> {c['a']}", flush=True)

    answered = [qa for qa in qas if not qa.get("abstained")]
    verified = [qa for qa in answered if qa.get("verified")]
    res = {
        "agent_recall_acc": rec_acc,
        "firewall_project_answered": fw["positive_answered"], "firewall_project_total": fw["positive_total"],
        "firewall_general_leaks": fw["general_leaks"], "firewall_untaught_leaks": fw["untaught_leaks"],
        "faculty_loaded": faculty is not None, "faculty_error": faculty_err,
        "selfreflect_answered": len(answered), "selfreflect_verified": len(verified), "selfreflect_total": len(qas),
        "self_reflective_qas": qas,
        "chat": chat_out,
        "recall_ge_0.8": bool(rec_acc >= 0.8),
        "firewall_intact": bool(fw["general_leaks"] == 0 and fw["untaught_leaks"] == 0),
        "faculty_works": bool(faculty is not None and len(verified) >= 1),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    res["CHAT_READY"] = bool(res["recall_ge_0.8"] and res["firewall_intact"] and res["faculty_works"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"\n[saved] {OUT}", flush=True)
    print(f"[VERDICT] CHAT_READY={res['CHAT_READY']}  recall={rec_acc:.2f}  firewall_intact={res['firewall_intact']}  "
          f"faculty_works={res['faculty_works']}  (verified {len(verified)}/{len(qas)} self-Q&As)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
