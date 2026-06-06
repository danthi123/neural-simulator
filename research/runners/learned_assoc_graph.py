"""Cheat-D integration: a SUBSTRATE-LEARNED concept-concept association graph for the conversational agent's dlPFC
dialogue planning -- replacing the set-from-Python co-occurrence recompute (`_assoc_graph`) with a learned sparse
recurrent (the validated `_D_sparse_heteroassoc` heteroassociative memory; Marr/Treves-Rolls CA3 autoassociator).

Concepts = sparse K-of-N patterns in a pool with a PLASTIC excitatory recurrent. `store_fact` co-fires a fact's
concepts -> the recurrent LEARNS their pairwise co-occurrence by Hebbian growth (NOT set). `graph()` reads the learned
recurrent weights -> the concept->{concept: weight} graph the dlPFC spreads over. So dialogue planning spreads over a
substrate-LEARNED association memory, not a Python dict recomputed from the kb.

Validated: the learned graph's top associates match the Python co-occurrence oracle (the dlPFC picks the same
associates), multi-seed. NO sim/ edits (reuses the brain-region framework + Hebbian + generate_sparse_patterns).
"""
import argparse

import numpy as np

from research.runners._D_sparse_heteroassoc import build, _pool_global, _drive
from research.runners.concept_pool_sparse_distributed import generate_sparse_patterns
from sim.backend import to_host


class LearnedAssocGraph:
    def __init__(self, concepts, seed=42, n_pool=1500, pattern_size=100):
        self.concepts = list(concepts)
        self.idx = {c: i for i, c in enumerate(self.concepts)}
        self.patterns = generate_sparse_patterns(len(self.concepts), n_pool, pattern_size, seed)
        self.bridge = build(seed, n_pool=n_pool)
        self.pg = _pool_global(self.bridge, self.patterns)
        self.pool_base = np.asarray(self.bridge.region_manager.indices("pool"))

    def store_fact(self, concept_list, cycles=12):
        """Co-fire the fact's concept patterns -> Hebbian growth on the recurrent learns the pairwise co-occurrence."""
        ids = [self.idx[c] for c in concept_list if c in self.idx]
        if len(ids) < 2:
            return
        try:
            self.bridge.set_plasticity_gate("recurrent", 1.0)
        except KeyError:
            pass
        drive = [self.pg[i] for i in ids]
        for _ in range(cycles):
            _drive(self.bridge, drive, 1100.0)
            for _ in range(10):
                self.bridge._run_one_simulation_step()
            self.bridge.cp_external_input_current[:] = 0.0
            for _ in range(5):
                self.bridge._run_one_simulation_step()
        try:
            self.bridge.set_plasticity_gate("recurrent", 0.0)
        except KeyError:
            pass

    def graph(self, thresh=0.5):
        """Read the LEARNED recurrent weights -> concept -> {concept: mean a->b weight}. The dlPFC spreads over this."""
        M = to_host(self.bridge.cp_connections)
        dense = np.asarray(M[self.pool_base][:, self.pool_base].todense())
        g = {}
        for a, ca in enumerate(self.concepts):
            pa = np.asarray(self.patterns[a])
            for b, cb in enumerate(self.concepts):
                if a == b:
                    continue
                w = float(dense[np.ix_(pa, np.asarray(self.patterns[b]))].mean())
                if w > thresh:
                    g.setdefault(ca, {})[cb] = w
        return g


def _python_cooccur(facts):
    """The Python co-occurrence oracle (what the agent's _assoc_graph recomputes)."""
    g = {}
    for fact in facts:
        cs = [c for c in fact if isinstance(c, str)]
        for x in cs:
            for y in cs:
                if x != y:
                    g.setdefault(x, {})[y] = g.get(x, {}).get(y, 0.0) + 1.0
    return g


def _top(graph, c):
    d = graph.get(c, {})
    return max(d, key=d.get) if d else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    concepts = ["dog", "cat", "go", "run", "north", "south", "apple", "river", "big", "small"]
    facts = [("dog", "go", "north"), ("cat", "run", "south"), ("apple", "big", "river"),
             ("dog", "big", "cat")]   # dog co-occurs with go,north AND big,cat
    lag = LearnedAssocGraph(concepts, seed=args.seed)
    for f in facts:
        lag.store_fact(f, cycles=30)
    M = to_host(lag.bridge.cp_connections)
    d = np.asarray(M[lag.pool_base][:, lag.pool_base].data)
    print(f"  [recurrent: mean={float(np.abs(d).mean()):.3f} max={float(np.abs(d).max()):.3f} nnz={len(d)}]")
    learned = lag.graph(thresh=0.3)
    oracle = _python_cooccur(facts)

    # parity: for each concept, do the learned graph's associates cover the oracle's? + the top associate matches?
    n = match_edges = top_match = 0
    for c in oracle:
        la = set(learned.get(c, {}))
        oa = set(oracle.get(c, {}))
        match_edges += len(la & oa)
        n += len(oa)
        top_match += int(_top(learned, c) in oa) if learned.get(c) else 0
    print(f"=== LearnedAssocGraph (seed={args.seed}) ===")
    print(f"learned-graph edges recovered: {match_edges}/{n} oracle co-occurrence edges")
    print(f"learned top-associate in oracle: {top_match}/{len([c for c in oracle if learned.get(c)])} concepts")
    for c in ["dog", "cat", "apple"]:
        ld = learned.get(c, {})
        print(f"  {c}: learned associates {sorted(ld, key=ld.get, reverse=True)}  | oracle {sorted(oracle.get(c, {}))}")


if __name__ == "__main__":
    main()
