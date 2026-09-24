"""A5 BYTE-IDENTITY-OFF CHECK: `BRAIN_FALSE_BELIEF_CHAT` unset vs explicitly "0" must produce IDENTICAL
`webapp.server.brain_chat` responses over a fixed 10-turn panel (9 ordinary turns + 1 full Sally-Anne
narration turn whose text WOULD trigger the new grammar if the flag were on). Mirrors the worker/spawn-arm
pattern `onebrain_regression_battery._collect_worker`/`_spawn_arm` uses, kept self-contained in this file (no
edit to that module's shared `_TURN_BY_LABEL`) since this panel is specific to this one faculty.

Usage (dev seed only):
  bash tools/mem_ok.sh 2 4 && bash tools/memcap.sh 4 -- .venv/bin/python -u \\
      -m research.runners._tom_false_belief_chat_offcheck --seed 7 \\
      --json research/findings/raw/_tom_false_belief_chat/byte_identity_off.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# (label, message, session, reset, percept, rich) -- 9 ordinary turns (mirrors onebrain_regression_battery's
# own PROBE_TURNS style) + 1 turn narrating the full Sally-Anne script this faculty's grammar recognizes.
_TURNS = [
    ("well",     "the wolf bites the apple", "well", True, None, False),
    ("question", "what does the wolf bite",  "q",    True, None, False),
    ("unknown",  "what is the capital of france", "u", True, None, False),
    ("hold",     "the fox and the wolf walked in", "d", True, None, False),
    ("held",     "the wolf watches the owl", "d", False, None, False),
    ("scalar",   "some of the dogs ran",     "s",    True, None, False),
    ("open",     "what might a dog chase",   "o",    True, None, False),
    ("emo",      "Wonderful! I am so happy and delighted, this is fantastic and amazing!", "emo", True, None, False),
    ("episodic", "did we discuss the dog",   "epi",  True, None, False),
    ("tom_fb",
     "Sally puts the marble in the basket. Sally leaves the room. Anne moves the marble to the box. "
     "Where will Sally look for the marble?",
     "tomfb", True, None, False),
]


def _collect_worker(env_json: str, out_path: str) -> int:
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    env = json.loads(env_json)
    for k, v in env.items():
        os.environ[k] = v   # explicit set, never a pop -> an OFF arm stays OFF regardless of any later default flip
    from webapp.server import brain_chat, BrainChatRequest
    responses = {}
    for label, msg, session, reset, percept, rich in _TURNS:
        try:
            kwargs = dict(session=session, message=msg, brain="tiny-demo", renderer="stub",
                         rich=bool(rich), reset=reset)
            if percept is not None:
                kwargs["percept"] = percept
            r = brain_chat(BrainChatRequest(**kwargs))
            responses[label] = json.loads(r.body)
        except Exception as e:
            responses[label] = {"_error": f"{type(e).__name__}: {e}"}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(responses, f, indent=2, default=str)
    print(f"[tom-fb-offcheck worker] env={env} -> {len(responses)} turns -> {out_path}", flush=True)
    return 0


def _spawn_arm(env: dict, out_path: str, seed: int) -> dict:
    import subprocess
    full_env = dict(os.environ)
    full_env["BRAIN_CHAT_SEED"] = str(seed)
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners._tom_false_belief_chat_offcheck",
                       "--worker", "--env", json.dumps(env), "--out", out_path], env=full_env)
    if p.returncode != 0 or not os.path.exists(out_path):
        return {}
    return json.load(open(out_path))


def _diff(a: dict, b: dict):
    diffs = []
    for label in _TURNS:
        k = label[0]
        va, vb = a.get(k), b.get(k)
        if va != vb:
            diffs.append(k)
    return diffs


def main():
    ap = argparse.ArgumentParser(description="A5 false-belief chat wire byte-identity-off check.")
    ap.add_argument("--worker", action="store_true", help="internal: build one arm + run the panel")
    ap.add_argument("--env", type=str, default="{}")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_tom_false_belief_chat/byte_identity_off.json")
    args = ap.parse_args()

    if args.worker:
        return _collect_worker(args.env, args.out)

    base = os.path.dirname(os.path.abspath(args.json))
    os.makedirs(base, exist_ok=True)
    unset_path = os.path.join(base, "arm_unset.json")
    off_path = os.path.join(base, "arm_explicit_off.json")
    on_path = os.path.join(base, "arm_on.json")

    resp_unset = _spawn_arm({}, unset_path, args.seed)
    resp_off = _spawn_arm({"BRAIN_FALSE_BELIEF_CHAT": "0"}, off_path, args.seed)
    resp_on = _spawn_arm({"BRAIN_FALSE_BELIEF_CHAT": "1"}, on_path, args.seed)

    off_diffs = _diff(resp_unset, resp_off)
    byte_identical_off = (len(resp_unset) == len(_TURNS) and len(resp_off) == len(_TURNS)
                          and not off_diffs)

    on_query_answer = None
    on_query_populated = False
    if resp_on:
        fb = (resp_on.get("tom_fb") or {}).get("false_belief_tom")
        if fb is not None:
            on_query_populated = bool(fb.get("acted"))
            on_query_answer = fb.get("answer")

    result = {
        "seed": int(args.seed), "turns": [t[0] for t in _TURNS],
        "byte_identical_off": bool(byte_identical_off),
        "off_diffs": off_diffs,
        "on_arm_query_populated": on_query_populated,
        "on_arm_answer": on_query_answer,
        "arms": {"unset": unset_path, "explicit_off": off_path, "on": on_path},
    }
    with open(args.json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[tom-fb-offcheck] byte_identical_off={byte_identical_off}  off_diffs={off_diffs}  "
          f"on_arm_answer={on_query_answer!r}  wrote {args.json}", flush=True)
    return 0 if byte_identical_off else 1


if __name__ == "__main__":
    raise SystemExit(main())
