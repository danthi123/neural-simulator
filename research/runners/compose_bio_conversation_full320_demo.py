"""Biologically-grounded relational conversation across ALL 320 concepts (brain-analogue mechanism).

Uses the cached 320 hierarchical codes (_hier320_codes.npz, built by _insubstrate_hierarchical320_spiking)
so it runs fast. Stores CROSS-BANK facts (e.g. dog[noun] run[verb] big[adj], each from a different concept
bank, made distinct by the hierarchical bridge-role bind), answers wh-queries by SPIKING unbind + cleanup,
abstains on untaught facts. Relational reasoning computed by spiking neurons over the full 320-concept space.

Reuse-by-import; no protected-module change; no autograd.
Run (GPU): python -m research.runners.compose_bio_conversation_full320_demo
"""
from __future__ import annotations
import os
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend

CACHE = "research/findings/raw/_hier320_codes.npz"
D = 2000


def main():
    if not os.path.exists(CACHE):
        print(f"CANNOT-RUN: {CACHE} not found (run _insubstrate_hierarchical320_spiking first)", flush=True)
        return
    xp, backend = get_backend()
    d = np.load(CACHE)
    words = list(d["_words"]); codes = {w: d[w] for w in d.files if w != "_words"}
    print(f"=== biological relational conversation across {len(words)} concepts (backend={backend}) ===",
          flush=True)

    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    rng = np.random.default_rng(42)
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bb, bidx = P.build(42, D, xp)

    # cross-bank facts: agent (noun bank) / action (verb bank) / patient (adj or noun bank) -- all distinct
    # via the hierarchical bridge-role bind. Use words known to be in the 320 vocab.
    def pick(cands):
        for c in cands:
            if c in codes:
                return c
        return None
    facts = []
    for a, v, p in [("dog", "run", "big"), ("cat", "jump", "small"), ("bird", "fly", "tall")]:
        a, v, p = pick([a]), pick([v]), pick([p])
        if a and v and p:
            facts.append({"agent": a, "action": v, "patient": p})
    if not facts:   # fallback to any 3 distinct codes
        ks = list(codes)[:9]
        facts = [{"agent": ks[3*i], "action": ks[3*i+1], "patient": ks[3*i+2]} for i in range(3)]

    print("\n  -- teaching cross-bank facts (each a spiking role(x)filler bind) --", flush=True)
    bounds = []
    for f in facts:
        bounds.append(RM.bind_fact_spiking(bb, bidx, f, codes, roles, D, xp))
        print(f"    stored:  {f['agent']} {f['action']} {f['patient']}", flush=True)

    def ask(given, qr):
        for b in bounds:
            if all(RM.unbind_spiking(bb, bidx, b, r, roles, codes, words, D, xp) == w
                   for r, w in given.items()):
                return RM.unbind_spiking(bb, bidx, b, qr, roles, codes, words, D, xp)
        return None

    print("\n  -- asking (spiking unbind + cleanup over all 320 concepts) --", flush=True)
    ok = tot = 0
    for f in facts:
        who = ask({"action": f["action"], "patient": f["patient"]}, "agent")
        what = ask({"agent": f["agent"], "action": f["action"]}, "patient")
        ok += int(who == f["agent"]) + int(what == f["patient"]); tot += 2
        print(f"    who {f['action']} {f['patient']}?  -> {who}  ({'OK' if who==f['agent'] else 'x'})",
              flush=True)
        print(f"    what did {f['agent']} {f['action']}?  -> {what}  ({'OK' if what==f['patient'] else 'x'})",
              flush=True)

    used = set(w for f in facts for w in f.values()); spare = [w for w in words if w not in used]
    miss = ask({"action": spare[0], "patient": spare[1]}, "agent")
    print(f"\n  -- abstention (untaught) --\n    who {spare[0]} {spare[1]}?  -> "
          f"{miss if miss else '(unknown -- correctly abstains)'}", flush=True)
    print(f"\n  RESULT: {ok}/{tot} correct via the spiking bind over {len(words)} concepts; abstains = "
          f"{miss is None}. Relational reasoning by spiking neurons across the full 320-concept substrate.",
          flush=True)


if __name__ == "__main__":
    main()
