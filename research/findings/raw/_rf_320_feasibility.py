"""(c2-scale-320) feasibility: does the RF phasor composer's correctness scale to a 320-concept vocab (the rate
composer's production tier)? The risk is the cleanup (phase-cosine argmax over 320 candidates) + the no-confab moat
(no false-match among many facts). D=512 for the 320-way cleanup separation. If GO -> the global flip path is open
(modulo a sparse-matvec speed optimization); if a boundary -> RF needs larger D / stays V=16-scoped."""
import numpy as np
from research.runners.rf_phasor_composer import RFPhasorComposer


def main():
    vocab = [f"w{i}" for i in range(320)]
    comp = RFPhasorComposer(seed=42, D=512, vocab=vocab, period=200)
    rng = np.random.default_rng(0)
    facts = []
    while len(facts) < 8:
        idx = rng.choice(320, 3, replace=False)
        f = (vocab[idx[0]], vocab[idx[1]], vocab[idx[2]])
        facts.append(f)
        comp.store(*f)

    n_who = n_what = 0
    for (a, v, p) in facts:
        if comp.query_agent(v, p) == a:
            n_who += 1
        if comp.query_patient(a, v) == p:
            n_what += 1

    # abstention: query (action, patient) pairs that are NOT stored -> must be None (no false-match among 320)
    stored_vp = {(v, p) for (a, v, p) in facts}
    n_abstain_ok = 0
    n_abstain = 0
    for _ in range(8):
        i, j = rng.choice(320, 2, replace=False)
        if (vocab[i], vocab[j]) not in stored_vp:
            n_abstain += 1
            if comp.query_agent(vocab[i], vocab[j]) is None:
                n_abstain_ok += 1

    print(f"[RF-320] who={n_who}/{len(facts)} what={n_what}/{len(facts)} "
          f"abstain={n_abstain_ok}/{n_abstain} (D=512, 320 vocab, 8 facts)", flush=True)
    verdict = "GO" if (n_who >= 7 and n_what >= 7 and n_abstain_ok == n_abstain) else "BOUNDARY"
    print(f"[RF-320 VERDICT] correctness scales to 320 -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
