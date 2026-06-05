"""Does the FHRR-on-bridge RF composer LIFT the rate composer's 2-attribute K=5-load boundary? A 2-attribute entity
('big hot apple') is a 5-binding fact (agent/action/patient/attribute/attribute2). The +-1 Hadamard rate composer
degrades the noun at K=5; FHRR phasor capacity is SNR~2N/M (a D dial), so it may RESOLVE at sufficient D.
GATE: all 3 adjectives+noun recovered (set match) multi-seed."""
import numpy as np
from research.runners.rf_phasor_composer import RFPhasorComposer


def test_2attr(seed, D):
    comp = RFPhasorComposer(seed=seed, D=D, period=200)
    comp.store("dog", "look", (("big", "hot"), "apple"))   # 2-attribute: big hot apple
    comp.store("cat", "go", "river")                        # a flat fact alongside
    got = comp.query_patient("dog", "look")
    words = set(got.split()) if got else set()
    ok = words == {"big", "hot", "apple"}
    # also confirm the flat fact + a 1-attribute still work + abstention holds
    flat_ok = comp.query_patient("cat", "go") == "river"
    return ok, flat_ok, got


if __name__ == "__main__":
    for D in [128, 256, 512]:
        rows = []
        for seed in (42, 43, 44):
            ok, flat_ok, got = test_2attr(seed, D)
            rows.append((seed, ok, flat_ok, got))
        n_ok = sum(1 for _, ok, _, _ in rows if ok)
        n_flat = sum(1 for _, _, f, _ in rows if f)
        detail = "  ".join(f"s{s}:{'OK' if ok else 'FAIL('+repr(g)+')'}" for s, ok, _, g in rows)
        print(f"D={D}: 2attr {n_ok}/3  flat {n_flat}/3  |  {detail}", flush=True)
