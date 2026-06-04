"""(A) capability completion: elaborate (dialogue planning) in the agent at V=320.

Build the BrainConversationalAgent with production 320 codes, store a connected set of facts, and check
elaborate(topic) returns an ON-TOPIC associate -- a concept that co-occurs with the topic in the agent's OWN stored
facts (a graph neighbor), chosen by the dlPFC spiking content-selection Control (SpikingSpreadingController, a real
2-region SimulationBridge). NOTE: elaborate spreads over the association GRAPH (built from stored-fact labels), so
its difficulty depends on the #facts, not the vocabulary size -- this confirms the dialogue-planning capability is
wired and working in the consolidated agent. Multi-seed-able.

Usage:
  python -m research.findings.raw._brain_agent_elaborate320_probe --seed 42
"""
import argparse
import json

from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.findings.raw._core_composer_grounded320_probe import production_codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vocab", type=int, default=320)
    ap.add_argument("--n-pool", type=int, default=2000)
    ap.add_argument("--pattern-size", type=int, default=100)
    ap.add_argument("--proj-dim", type=int, default=800)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    codes = production_codes(args.vocab, args.n_pool, args.pattern_size, args.proj_dim, args.seed)
    words = [f"c{i:03d}" for i in range(args.vocab)]
    concepts = {w: codes[i] for i, w in enumerate(words)}
    agent = BrainConversationalAgent(seed=args.seed, proj_dim=args.proj_dim, concepts=concepts)

    # a connected set of facts -> an association graph with clear per-topic neighbors. Each topic co-occurs
    # strongly with one concept (appears in two facts) and weakly with others.
    facts = [
        ("c000", "c010", "c020"), ("c000", "c010", "c021"),   # c000 strongly linked to c010
        ("c001", "c011", "c022"), ("c001", "c011", "c023"),   # c001 -> c011
        ("c002", "c012", "c024"), ("c002", "c012", "c025"),   # c002 -> c012
        ("c003", "c013", "c026"), ("c003", "c013", "c027"),   # c003 -> c013
    ]
    for a, ac, p in facts:
        agent.hear(f"{a} {ac} {p}")
    graph = agent._assoc_graph()

    topics = ["c000", "c001", "c002", "c003"]
    ok = 0
    results = []
    for t in topics:
        assoc = agent.elaborate(t)
        neigh = sorted(graph.get(t, {}))
        hit = assoc in graph.get(t, {})
        ok += int(hit)
        results.append({"topic": t, "elaborate": assoc, "neighbors": neigh, "on_topic": hit})

    res = {"seed": args.seed, "vocab": args.vocab, "on_topic": ok, "total": len(topics), "results": results}
    print(f"[elab320] seed {args.seed} V={args.vocab}  on-topic {ok}/{len(topics)}")
    for r in results:
        print(f"  {r['topic']} -> {r['elaborate']}  (neighbors {r['neighbors']})  {'OK' if r['on_topic'] else 'MISS'}")
    print("[elab320] " + json.dumps(res))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"[elab320] wrote {args.out}")


if __name__ == "__main__":
    main()
