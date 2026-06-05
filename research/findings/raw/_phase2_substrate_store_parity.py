"""Phase 2 (cheat C) INTEGRATION gate: the composer's queries with enable_substrate_store=True (each fact's bound
composite held in per-fact substrate weights, retrieved via firing) give the SAME answers as the numpy-kb default,
multi-seed; the no-confab moat (abstention) preserved.
"""
import numpy as np
from research.runners.rf_phasor_composer import RFPhasorComposer

FACTS = [("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple")]
QUERIES = [("go", "north", "dog"), ("run", "south", "cat"), ("look", "apple", "river")]


def run(seed, D):
    cn = RFPhasorComposer(seed=seed, D=D, period=200, enable_substrate_store=False)
    cs = RFPhasorComposer(seed=seed, D=D, period=200, enable_substrate_store=True)
    for a, v, p in FACTS:
        cn.store(a, v, p); cs.store(a, v, p)
    n = match = 0
    for v, p, a in QUERIES:
        n += 1; match += int(cs.query_agent(v, p) == cn.query_agent(v, p) == a)
        n += 1; match += int(cs.query_patient(a, v) == cn.query_patient(a, v) == p)
        n += 1; match += int(cs.render_fact(a) == cn.render_fact(a))
    abstain = (cs.query_agent("go", "river") is None)        # no-confab moat under the substrate store
    return match, n, abstain


if __name__ == "__main__":
    for D in (128, 256):
        rows = []
        for seed in (42, 43, 44):
            m, n, ab = run(seed, D)
            rows.append((seed, m, n, ab))
        tot_m = sum(m for _, m, _, _ in rows); tot_n = sum(n for _, _, n, _ in rows)
        ab_ok = sum(1 for _, _, _, ab in rows if ab)
        print(f"D={D}: substrate-store==numpy answers {tot_m}/{tot_n}   abstain-preserved {ab_ok}/3   "
              + "  ".join(f"s{s}:{m}/{n}{'A' if ab else 'x'}" for s, m, n, ab in rows), flush=True)
