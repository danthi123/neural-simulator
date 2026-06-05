"""Phase 1 (cheat B) INTEGRATION gate: the composer's queries with enable_spiking_cleanup=True (the fully-on-bridge
matched filter + Izhikevich WTA) give the SAME answers as the numpy default, multi-seed. Plus the no-confab moat
(abstention) is preserved under the spiking cleanup.
"""
import numpy as np
from research.runners.rf_phasor_composer import RFPhasorComposer

FACTS = [("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple")]
QUERIES = [("go", "north"), ("run", "south"), ("look", "apple")]   # (action, patient) -> agent


def run(seed, D):
    cn = RFPhasorComposer(seed=seed, D=D, period=200, enable_spiking_cleanup=False)
    cs = RFPhasorComposer(seed=seed, D=D, period=200, enable_spiking_cleanup=True)
    for a, v, p in FACTS:
        cn.store(a, v, p); cs.store(a, v, p)
    n = match = 0
    for (v, p), (a, _, _) in zip(QUERIES, FACTS):
        an = cn.query_agent(v, p); as_ = cs.query_agent(v, p)
        n += 1; match += int(an == as_)
        pn = cn.query_patient(a, v); ps = cs.query_patient(a, v)
        n += 1; match += int(pn == ps)
        yn = cn.ask_yes_no(a, v, p); ys = cs.ask_yes_no(a, v, p)
        n += 1; match += int(yn == ys)
    # no-confab moat: an unknown cue must abstain (None) under the spiking cleanup too
    abstain = (cs.query_agent("go", "river") is None)
    return match, n, abstain


if __name__ == "__main__":
    for D in (256,):
        rows = []
        for seed in (42, 43, 44):
            m, n, ab = run(seed, D)
            rows.append((seed, m, n, ab))
        tot_m = sum(m for _, m, _, _ in rows); tot_n = sum(n for _, _, n, _ in rows)
        ab_ok = sum(1 for _, _, _, ab in rows if ab)
        print(f"D={D}: spiking==numpy answers {tot_m}/{tot_n}   abstain-preserved {ab_ok}/3   "
              + "  ".join(f"s{s}:{m}/{n}{'A' if ab else 'x'}" for s, m, n, ab in rows), flush=True)
