# Capability: Sleep consolidation
# Record: brain/research/2026-11-15-own-cue-frame-1.5b.md (F-selection on the DEV bank)
"""Which frame makes the pretrained cortex write the USER's own question about a telling? No training, greedy, CPU.

For each pre-listed frame (OWN_FRAMES) and each DEV_FACTS36 statement, the cortex writes one question (greedy, 24
tokens, first question kept, as near_miss_cues does). A question is GOOD when it ends in "?", asks in the user's voice
(my / I / me, and no you / your), and does not contain the fact's key word. The frame with the most GOOD questions on
the DEV bank is selected (ties: list order). The weekly/probe FACTS are reported too, never used to select.

  python -m brain.experiments.own_cue_frames --out <json>
"""
import argparse
import json
import re
import socket

from brain.cortex import Cortex, resolve_model
from brain.experiments.facts_bank import DEV_FACTS36
from brain.experiments.live_brain_probe import FACTS
from brain.near_miss import OWN, _first_question

# Pre-listed before any run (brain/research/2026-11-15-own-cue-frame-1.5b.md). F0 is the near-miss guard's OWN frame.
OWN_FRAMES = {
    "F0": OWN,
    "F1": 'I told a friend: "{s}"\nThe next day, to check they remembered what I had told them about myself, I asked them: "',
    "F2": "User: {s}\n\nAssistant: Got it, I'll remember that.\n\nUser: (the next day) Quick check, do you remember? ",
    "F3": 'Something I told my friend about myself: "{s}"\nA question I could ask my friend to see if they remember it: "',
    "F4": "SELFQ",   # M3's declared self-question prompt (brain/sleep_retrieval.py SELFQ_PROMPT, after BIRTH), greedy
}
YOU = re.compile(r"\b(you|your|yours)\b", re.I)
ME = re.compile(r"\b(my|i|me|mine)\b", re.I)


def own_question(cortex, frame, statement):
    if OWN_FRAMES[frame] == "SELFQ":
        from brain.brain import BIRTH
        from brain.sleep_retrieval import self_cues
        qs = self_cues(cortex, BIRTH, statement, n_cues=1)
        return _first_question(qs[0]) if qs else ""
    lg, st = cortex.read(OWN_FRAMES[frame].format(s=statement), None)
    return _first_question(cortex.speak(lg, st, max_tokens=24, temperature=0)[0])


def grade(q, keys):
    isq = q.endswith("?")
    voice = bool(ME.search(q)) and not YOU.search(q)
    leak = any(re.search(rf"\b{re.escape(k)}\b", q, re.I) for k in keys)
    return {"q": q, "question": isq, "user_voice": voice, "leak": leak, "good": isq and voice and not leak}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = Cortex(model_path=resolve_model(a.model), device="cpu")
    out = {"model": getattr(c, "model_name", None), "host": socket.gethostname(), "frames": {}}
    for f in OWN_FRAMES:
        dev = [dict(grade(own_question(c, f, s), keys), statement=s) for s, _, _, _, keys in DEV_FACTS36]
        test = [dict(grade(own_question(c, f, s), (key,)), statement=s) for s, _, key in FACTS]
        out["frames"][f] = {"dev_good": sum(r["good"] for r in dev), "dev_n": len(dev),
                            "test_good": sum(r["good"] for r in test), "test_n": len(test), "dev": dev, "test": test}
        print(f, out["frames"][f]["dev_good"], "/", len(dev), "test", out["frames"][f]["test_good"], "/", len(test), flush=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    best = max(OWN_FRAMES, key=lambda f: (out["frames"][f]["dev_good"], -list(OWN_FRAMES).index(f)))
    out["selected"] = best
    print("selected", best)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
