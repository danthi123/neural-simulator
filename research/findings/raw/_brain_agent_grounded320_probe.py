"""(A) grounded-320, rung 2: the FULL BrainConversationalAgent (parser + composer) at production 320-word vocab.

Build the agent with the G.20 production sparse-distributed codes (concepts=), then run the full loop through the
agent's OWN API: hear() comprehends each 3-word SVO sentence via the Hebbian parser (position x voice -> role) and
stores it in the spiking composer; what_does/who_does query; abstain on the unknown. This validates that the
COMPLETE conversational agent -- not just the composer -- works at production vocabulary (the parser is
vocabulary-agnostic, so the same trained parser serves 320 concepts). Multi-seed.

Usage:
  python -m research.findings.raw._brain_agent_grounded320_probe --seed 42 --vocab 320 --n-facts 12
"""
import argparse
import json

import numpy as np

from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.findings.raw._core_composer_grounded320_probe import production_codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=320)
    ap.add_argument("--n-pool", type=int, default=2000)
    ap.add_argument("--pattern-size", type=int, default=100)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--n-facts", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    codes = production_codes(args.vocab, args.n_pool, args.pattern_size, args.proj_dim, args.seed)
    words = [f"c{i:03d}" for i in range(args.vocab)]
    concepts = {w: codes[i] for i, w in enumerate(words)}

    agent = BrainConversationalAgent(seed=args.seed, proj_dim=args.proj_dim, concepts=concepts)

    rng = np.random.default_rng(args.seed + 1)
    facts, cues, guard = [], set(), 0
    while len(facts) < args.n_facts and guard < 100000:
        guard += 1
        a, ac, p = (str(x) for x in rng.choice(words, size=3, replace=False))
        if (a, ac) in cues:
            continue
        cues.add((a, ac))
        facts.append((a, ac, p))

    # comprehend + store each as an active-voice SVO sentence (position 0/1/2 = agent/action/patient)
    for a, ac, p in facts:
        agent.hear(f"{a} {ac} {p}")

    okw = oka = okab = 0
    for a, ac, p in facts:
        okw += int(agent.what_does(a, ac) == p)
        oka += int(agent.who_does(ac, p) == a)
    for _ in range(args.n_facts):
        g2, a2, ac2 = 0, None, None
        while g2 < 1000:
            g2 += 1
            a2, ac2 = (str(x) for x in rng.choice(words, size=2, replace=False))
            if (a2, ac2) not in cues:
                break
        okab += int(agent.what_does(a2, ac2) is None)

    n = args.n_facts
    res = {
        "seed": args.seed, "vocab": args.vocab, "n_facts": n, "code_source": "G20_sparse_distributed_production",
        "full_agent": True, "what_correct": okw, "who_correct": oka, "abstain_correct": okab,
        "what_rate": okw / n, "who_rate": oka / n, "abstain_rate": okab / n,
    }
    print(f"[agent320] V={args.vocab} hear->comprehend->store->query  what {okw}/{n}  who {oka}/{n}  abstain {okab}/{n}")
    print("[agent320] " + json.dumps(res))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[agent320] wrote {args.out}")


if __name__ == "__main__":
    main()
