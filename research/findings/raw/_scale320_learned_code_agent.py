"""Capstone scale demonstration: the unified diversity + nesting capability at 320-concept production scale,
on LEARNED codes. Trains PhasorAssociativeMemory on 320 words, builds the full nesting agent on the learned
codes, stores mixed nested facts, and reports per-kind accuracy + memory recall. Writes JSON (robust output).

  python -m research.findings.raw._scale320_learned_code_agent
"""
import json
import numpy as np

from research.runners.phasor_associative_memory import PhasorAssociativeMemory
from research.runners.nested_composition_agent import NestedCompositionAgent, Clause

N_NOUN, N_VERB, N_ADJ = 200, 60, 60          # 320 concepts
D = 2048
OUT = "research/findings/raw/_scale320_learned_code_result.json"


def run(seed, n_facts=60):
    nouns = [f"noun{i}" for i in range(N_NOUN)]
    verbs = [f"verb{i}" for i in range(N_VERB)]
    adjs = [f"adj{i}" for i in range(N_ADJ)]
    mem = PhasorAssociativeMemory(D=D, seed=seed)
    for w in nouns + verbs + adjs:
        mem.learn(w)
    recall = float(np.mean([mem.recall(w) == w for w in nouns + verbs + adjs]))
    learned = {w: mem.code(w) for w in nouns + verbs + adjs}
    ag = NestedCompositionAgent(nouns, verbs, adjs, D=D, seed=seed, external_codes=learned)
    rng = np.random.default_rng(seed + 5)
    from collections import Counter
    tot, ok = Counter(), Counter()
    for _ in range(n_facts):
        a = nouns[rng.integers(N_NOUN)]
        v = verbs[rng.integers(N_VERB)]
        k = int(rng.integers(4))
        if k == 0:
            p, kind = nouns[rng.integers(N_NOUN)], "flat"
        elif k == 1:
            p, kind = (adjs[rng.integers(N_ADJ)], nouns[rng.integers(N_NOUN)]), "1-attr"
        elif k == 2:
            a1, a2 = rng.choice(N_ADJ, 2, replace=False)
            p, kind = ((adjs[a1], adjs[a2]), nouns[rng.integers(N_NOUN)]), "2-attr"
        else:
            p = Clause(nouns[rng.integers(N_NOUN)], verbs[rng.integers(N_VERB)], nouns[rng.integers(N_NOUN)])
            kind = "clause"
        ag.learn(a, v, p)
        tot[kind] += 1
        ok[kind] += int(ag.query_patient(a, v) == ag._render_filler(p))
    return recall, dict(tot), dict(ok)


def main():
    results = []
    for seed in (42, 43):
        recall, tot, ok = run(seed)
        total_ok = sum(ok.values())
        total = sum(tot.values())
        by_kind = {k: f"{ok.get(k, 0)}/{tot[k]}" for k in tot}
        row = {"seed": seed, "memory_recall_320": round(recall, 3),
               "facts_correct": f"{total_ok}/{total}", "accuracy": round(total_ok / total, 3), "by_kind": by_kind}
        results.append(row)
        print(f"seed {seed}: 320-concept LEARNED-code agent  recall {recall:.2f}  "
              f"facts {total_ok}/{total} = {total_ok/total:.2f}  by-kind {by_kind}", flush=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
