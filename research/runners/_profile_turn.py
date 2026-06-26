#!/usr/bin/env python
"""Stage-0 speed root-cause: profile the per-TURN cost of the first-chat console (~13s/turn). Build the 1454
brain ONCE, then time + cProfile several discuss turns -> the real per-turn hotspot (render/VERIFY resonates?
the moat-audit's what_does? the proposer?). Separates console.respond (the DiscursiveTurn) from audit_moat
(the demo's secondary safety re-check). CPU/numpy; reuse-by-import; no sim/ or runner edits.
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
import cProfile
import io
import pstats
import sys
import time

from research.runners.first_chat_console import build_brain_on_codes, FirstChatConsole, audit_moat


def main():
    t0 = time.time()
    brain = build_brain_on_codes("bridges/firstchat/brain1454_w7000_seed42.npz",
                                 facts_json="research/findings/raw/_combined_svo_facts.json",
                                 n_facts=60, cand_cap=16, verbose=False)
    print(f"[profile] build {time.time()-t0:.1f}s", flush=True)
    console = FirstChatConsole(brain)
    console.respond("what does boy go?")   # warm (cache + JIT-ish)
    prompts = ["what is head?", "what do you think about ball?", "is head like home?",
               "what does bird fly?", "what is river?", "what do you think about dog?"]
    pr = cProfile.Profile(); pr.enable()
    for m in prompts:
        t = time.time(); para, rec = console.respond(m); t_resp = time.time() - t
        t = time.time(); ok, _ = audit_moat(brain, rec); t_audit = time.time() - t
        print(f"[turn] respond {t_resp:5.1f}s + audit {t_audit:5.1f}s  moat={'OK' if ok else 'LEAK'}  "
              f"| {m!r}", flush=True)
    pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18)
    print(s.getvalue(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
