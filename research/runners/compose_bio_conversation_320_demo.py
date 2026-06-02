"""Biologically-grounded relational conversation at 64-concept scale (320-tier substrate).

The owner's goal: conversation built on the BRAIN-ANALOGUE mechanism (spiking composition), NOT static
engram-tag retrieval/ranking. This demo runs the validated in-substrate spiking bind/unbind on the REAL
deployed 320-tier bridge's 64 concept codes (captured with temporal integration), so the relational
reasoning is computed by actual spiking neurons -- scaled 4x past the 16-word relational demos.

Pipeline (all spiking): drive lang_input(word) through the trained bridge -> capture the concept code
(temporal integration) -> store SVO facts as spiking role(x)filler binds -> answer wh-queries by spiking
unbind + cleanup -> abstain on facts never stored. Readable scripted transcript.

Reuse-by-import (the validated probes + sparse builder); no protected-module change; no autograd.
Run (GPU): python -m research.runners.compose_bio_conversation_320_demo
"""
from __future__ import annotations
import os
import numpy as np

import research.findings.raw._insubstrate_real_substrate_qa_probe as Q   # capture_real_codes, N_*, SP
import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
import research.runners.concept_pool_sparse_distributed as SP
from sim.backend import get_backend

ROOT = "research/findings/raw/g11_bg"
BRIDGE = f"{ROOT}/g20_sparse_bridges_320/bridgeA_nouns_sparse64.simstate.h5"
VOCAB = f"{ROOT}/g20_bridgeA_nouns_vocab64.txt"


def main():
    if not os.path.exists(BRIDGE):
        print(f"CANNOT-RUN: {BRIDGE} not found", flush=True); return
    Q.STIM = 300          # temporal-integration readout (the validated lever)
    Q.SPARSITY = 0.007    # 320-tier
    xp, backend = get_backend()
    words = Q.load_vocab(VOCAB)
    print(f"=== biological relational conversation @ {len(words)} concepts (320-tier, backend={backend}) ===",
          flush=True)

    bridge = SP.build_sparse_pool_bridge(seed=42, n_lang_input=Q.N_LANG, n_shared_pool=Q.N_POOL,
                                         n_lang_output=Q.N_LANG, verbose=False)
    bridge.load_checkpoint(BRIDGE)
    print(f"  loaded bridge; capturing {len(words)} real concept codes (temporal integration)...", flush=True)
    codes = Q.capture_real_codes(bridge, words, 42, xp)

    D = Q.N_POOL
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    rng = np.random.default_rng(42)
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    bb, bidx = P.build(42, D, xp)

    # pick concrete in-vocab facts (agent/action/patient are 3 distinct concepts; roles are abstract)
    facts = [
        {"agent": words[0], "action": words[5], "patient": words[10]},
        {"agent": words[3], "action": words[8], "patient": words[15]},
        {"agent": words[20], "action": words[25], "patient": words[30]},
    ]
    print("\n  -- teaching facts (each stored as a spiking role(x)filler bind) --", flush=True)
    bounds = []
    for f in facts:
        b = RM.bind_fact_spiking(bb, bidx, f, codes, roles, D, xp)
        bounds.append(b)
        print(f"    stored:  agent={f['agent']}  action={f['action']}  patient={f['patient']}", flush=True)

    def ask(given, query_role):
        for b in bounds:
            if all(RM.unbind_spiking(bb, bidx, b, r, roles, codes, words, D, xp) == w
                   for r, w in given.items()):
                return RM.unbind_spiking(bb, bidx, b, query_role, roles, codes, words, D, xp)
        return None

    print("\n  -- asking (answers computed by spiking unbind + cleanup) --", flush=True)
    ok = 0; tot = 0
    for f in facts:
        who = ask({"action": f["action"], "patient": f["patient"]}, "agent")
        what = ask({"agent": f["agent"], "action": f["action"]}, "patient")
        ok += int(who == f["agent"]) + int(what == f["patient"]); tot += 2
        print(f"    who {f['action']} {f['patient']}?  -> {who}   ({'OK' if who==f['agent'] else 'x'})",
              flush=True)
        print(f"    what did {f['agent']} {f['action']}?  -> {what}   ({'OK' if what==f['patient'] else 'x'})",
              flush=True)

    # abstention: ask about a fact never stored
    used = set(w for f in facts for w in f.values())
    spare = [w for w in words if w not in used]
    miss = ask({"action": spare[0], "patient": spare[1]}, "agent")
    print(f"\n  -- abstention (a fact never taught) --", flush=True)
    print(f"    who {spare[0]} {spare[1]}?  -> {miss if miss else '(unknown -- correctly abstains)'}",
          flush=True)
    print(f"\n  RESULT: {ok}/{tot} wh-answers correct via the spiking bind; abstains on unknown = "
          f"{miss is None}. Relational reasoning computed by spiking neurons at {len(words)} concepts.",
          flush=True)


if __name__ == "__main__":
    main()
