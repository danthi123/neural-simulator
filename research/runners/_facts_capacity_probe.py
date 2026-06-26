#!/usr/bin/env python
"""Capacity de-risk (deep-knowledge build): how many REAL corpus facts can the production RFPhasorComposer
hold at recall>=0.95 + no-confab moat 0? Research-doc risk 2: a VSA composite degrades ~linearly with facts-
per-composite. Stores the top-N real facts on the 7K brain's grounded codes, measures who+what recall + the
moat (absent cues must abstain), for a sweep of N -> the Stage-1b storage budget (facts per composer before
the multi-bridge split is needed). CPU/numpy; reuse-by-import; no sim/ edit.
"""
import os
os.environ["SIM_BACKEND"] = "numpy"
import sys

import numpy as np

from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners.first_chat_console import _load_real_facts

NPZ = "bridges/firstchat/brain1454_w7000_seed42.npz"
FACTS = "research/findings/raw/_tinystories_svo_facts_full.json"


def main():
    d = np.load(NPZ, allow_pickle=True)   # our own artifact; allow_pickle only for vocab dtype=object
    vocab = [str(w) for w in d["vocab"]]; G = d["grounded"]; D = int(d["D"])
    grounded = {vocab[i]: G[i] for i in range(len(vocab))}
    print(f"[cap] brain {len(vocab)} concepts, D={D}; sweeping real-fact storage capacity", flush=True)
    for N in (60, 120, 180, 250):
        facts, absent_what, absent_who = _load_real_facts(FACTS, vocab, N, 42)
        comp = RFPhasorComposer(seed=42, D=D, vocab=sorted(set(vocab)), grounded_codes=grounded)
        for a, v, p in facts:
            comp.store(a, v, p)
        ok = tot = 0
        for a, v, p in facts:
            tot += 2
            if comp.query_patient(a, v) == p:
                ok += 1
            if comp.query_agent(v, p) == a:
                ok += 1
        fa = 0
        for a, v in absent_what:
            if comp.query_patient(a, v) is not None:
                fa += 1
        for v, p in absent_who:
            if comp.query_agent(v, p) is not None:
                fa += 1
        rec = ok / tot if tot else 0.0
        print(f"[cap] N={len(facts):3d} facts | who+what recall {ok}/{tot} = {rec:.3f} | "
              f"moat false-accepts {fa} | {'GO' if rec >= 0.95 and fa == 0 else 'degrading'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
