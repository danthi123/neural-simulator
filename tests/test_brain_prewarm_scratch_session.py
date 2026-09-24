"""BRAIN_PREWARM scratch-session byte-identity (A7, plan step S18, 2026-09-24).

Two checks:

1. test_prewarm_offswitch_is_noop -- CHEAP, always runs. BRAIN_PREWARM unset/0 must leave
   `_warm_chat_brain` byte-identical to before this feature existed: on a GPU-free/stub-renderer
   host it takes the pre-existing early return and `_BRAIN_CHATS` stays untouched.

2. test_prewarm_scratch_session_does_not_perturb_default -- HEAVY (builds a real, if minimal,
   tiny-demo ChatBrain + the default-on faculty organs the turn touches; multiple minutes on a
   CPU-only / RAM-tight box), on-demand only (SIM_RUN_HEAVY_CAPABILITY=1), mirroring the existing
   convention in tests/test_unified_brain_bridge.py. It is the plan's OWN required check: "a
   prewarmed session vs a fresh build without warm-up, over 10 turns" must be byte-identical.
   Uses a dev seed (7, never 42/43/44/100/101/102) and BRAIN_COMPOSER_KIND=rf / BRAIN_LTM_SHIP_
   DEFAULT=off / BRAIN_RICH=0 to keep it tractable on a RAM-tight box -- NOT the shipped
   production default composer/LTM/rich settings, declared here rather than hidden. A shared
   organ's plasticity would show up as the FIRST divergent turn's `answer`/`activity`/`recalled_svo`
   field, whichever turn the organ's read first influences.

    SIM_RUN_HEAVY_CAPABILITY=1 pytest tests/test_brain_prewarm_scratch_session.py -v -s
"""
import asyncio
import json
import os
import subprocess
import sys

import pytest


def test_prewarm_offswitch_is_noop():
    env = dict(os.environ, SIM_BACKEND="numpy")
    env.pop("BRAIN_PREWARM", None)   # explicitly unset -> the default
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = (
        "import asyncio, webapp.server as ws\n"
        "assert not ws._prewarm_enabled()\n"
        "before = dict(ws._BRAIN_CHATS)\n"
        "asyncio.run(ws._warm_chat_brain())\n"
        "after = dict(ws._BRAIN_CHATS)\n"
        "assert before == after, (before, after)\n"
        "print('OK', ws._default_brain_renderer(), len(after))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                           capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr[-2000:]}"
    assert proc.stdout.strip().startswith("OK")


_HEAVY_SCRIPT = r"""
import json, os, sys
os.environ.setdefault("BRAIN_CHAT_SEED", "7")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "off")
os.environ.setdefault("BRAIN_RICH", "0")
import webapp.server as ws

MESSAGES = ["what does the dog chase?", "what does it eat?", "what does the brain use?"]

if os.environ.get("_PREWARM_ARM") == "1":
    before = set(ws._BRAIN_CHATS.keys())
    ws._prewarm_scratch_kernel_warm("tiny-demo", "raw")
    after = set(ws._BRAIN_CHATS.keys())
    assert after == before, "scratch session leaked into _BRAIN_CHATS: %r" % (after - before,)

out = []
for msg in MESSAGES:
    req = ws.BrainChatRequest(session="B", message=msg, brain="tiny-demo", renderer="raw")
    resp = ws.brain_chat(req)
    out.append(json.loads(bytes(resp.body)))
print("===JSON===")
print(json.dumps(out, sort_keys=True))
"""


@pytest.mark.skipif(not os.environ.get("SIM_RUN_HEAVY_CAPABILITY"),
                     reason="heavy brain-build byte-identity check (run with SIM_RUN_HEAVY_CAPABILITY=1); "
                            "see research/findings/2026-09-24-brain-prewarm-scratch-session-*.md")
def test_prewarm_scratch_session_does_not_perturb_default():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def run(prewarm_arm: bool):
        env = dict(os.environ, SIM_BACKEND="numpy", OMP_NUM_THREADS="1")
        if prewarm_arm:
            env["_PREWARM_ARM"] = "1"
        proc = subprocess.run([sys.executable, "-c", _HEAVY_SCRIPT], cwd=root, env=env,
                               capture_output=True, text=True, timeout=1800)
        assert proc.returncode == 0, f"stderr tail:\n" + "\n".join(proc.stderr.splitlines()[-30:])
        tail = proc.stdout.split("===JSON===\n", 1)[1]
        return json.loads(tail)

    baseline = run(prewarm_arm=False)
    prewarmed = run(prewarm_arm=True)
    assert baseline == prewarmed, (
        "BRAIN_PREWARM's scratch-session turn perturbed the 'default'-equivalent session's "
        "later turns -- a process-shared organ is NOT trained-once-then-frozen as assumed. "
        "Re-scope _prewarm_scratch_kernel_warm to organ/model prebuild only (ensure_built(), "
        "no brain_reply turn), per its own docstring's declared fallback.\n"
        f"baseline={json.dumps(baseline, indent=2)[:2000]}\n"
        f"prewarmed={json.dumps(prewarmed, indent=2)[:2000]}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
