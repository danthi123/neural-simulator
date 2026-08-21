#!/usr/bin/env python
"""Interactive knowledge chat over the brain's tiered fact store (a testable knowledge-rich chat).

Loads a developed brain + a large cortical LTM (the persisted sharded knowledge store) and answers questions
from the brain's own recall, abstaining honestly when it doesn't know (the no-confab moat). Knowledge recall +
honesty work on CPU; fluent prose (the Qwen mouth) needs SIM_BACKEND=cupy + the full server.

Usage:
  SIM_BACKEND=numpy .venv/bin/python tools/chat_with_knowledge.py \
      --brain research/findings/raw/_knowledge_bundle_wikidata_100k/chat_brain_bundle \
      --ltm   research/findings/raw/_knowledge_bundle_wikidata_100k/ltm_store_partial
Then type e.g.:  canada isa   |   gold isa   |   what is norway   |   quit
"""
import argparse, os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import logging; logging.disable(logging.INFO)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.runners.developed_brain_io import load_developed_brain

STOP = {"the", "a", "an", "of", "is", "are", "does", "do", "what", "who", "?", "'s"}

def parse(q):
    """Very small question parser: 'what is canada' / 'canada isa' -> (canada, isa). '<s> <rel>' -> (s, rel)."""
    w = [t for t in q.lower().replace("?", " ").split() if t]
    if not w:
        return None
    # 'what is X' / 'what does X <rel>'
    if w[0] in ("what", "who") and len(w) >= 3 and w[1] in ("is", "are"):
        return (" ".join(w[2:]), "isa")
    core = [t for t in w if t not in STOP]
    if len(core) >= 2:
        return (core[0], core[1])          # '<subject> <relation>'
    if len(core) == 1:
        return (core[0], "isa")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", required=True, help="a developed-brain bundle dir")
    ap.add_argument("--ltm", required=True, help="a persisted cortical LTM (sharded store) dir")
    a = ap.parse_args()
    t0 = time.time()
    agent, man = load_developed_brain(a.brain, ltm_bundle=a.ltm)
    print(f"[loaded a brain + {agent.composer.total_facts():,}-fact knowledge store in {time.time()-t0:.1f}s]")
    print("Ask e.g.  'canada isa'  ·  'gold isa'  ·  'what is norway'  ·  'ireland borders'  ·  quit\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if q.lower() in ("quit", "exit", ":q"):
            break
        p = parse(q)
        if not p:
            print("brain> (couldn't parse that — try '<subject> <relation>', e.g. 'canada isa')"); continue
        subj, rel = p
        ans = agent.what_does(subj, rel)
        if ans is None:
            print(f"brain> I don't have a fact about '{subj} {rel}'. (honest abstain — no made-up answer)")
        else:
            print(f"brain> {subj} {rel} {ans}")

if __name__ == "__main__":
    main()
