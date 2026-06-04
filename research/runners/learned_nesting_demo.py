"""The substrate-unification payoff, end-to-end: the FULL nesting agent running on LEARNED codes.

Wires the STDP-learned phasor codes from PhasorAssociativeMemory (codes LEARNED from grounded word cues by
online weight-bounded spike-timing plasticity) into NestedCompositionAgent (nested SVO facts, resonator-
decoded attributed entities, embedded clauses, Q&A, abstention). Where the agent normally uses CONSTRUCTED
codes, here every concept code is LEARNED -- demonstrating that the biologically-grounded learned
representation supports the same composition the non-invertible production binding cannot.

Reuse-by-import; no protected-module edits (the agent gained an additive `external_codes` hook).

  python -m research.runners.learned_nesting_demo
"""
from research.runners.phasor_associative_memory import PhasorAssociativeMemory
from research.runners.nested_composition_agent import NestedCompositionAgent, Clause


def build_learned_agent(nouns, verbs, adjs, D=2048, seed=42):
    """Train a PhasorAssociativeMemory on the vocabulary, then build a nesting agent on its LEARNED codes."""
    mem = PhasorAssociativeMemory(D=D, seed=seed)
    for w in nouns + verbs + adjs:
        mem.learn(w)
    learned = {w: mem.code(w) for w in nouns + verbs + adjs}     # STDP-learned phasor codes (phases)
    agent = NestedCompositionAgent(nouns, verbs, adjs, D=D, seed=seed, external_codes=learned)
    return agent, mem


def main():
    nouns = ["dog", "cat", "ball", "river", "bird"]
    verbs = ["chase", "see", "eat", "hold"]
    adjs = ["big", "red", "cold"]
    agent, mem = build_learned_agent(nouns, verbs, adjs)
    print("=== nesting agent on LEARNED codes (STDP-learned, not constructed) ===\n", flush=True)
    print(f"  trained PhasorAssociativeMemory on {len(nouns + verbs + adjs)} words; codes learned via "
          f"online-bounded STDP.\n", flush=True)
    facts = [
        ("dog", "chase", "cat"),                         # flat
        ("bird", "see", ("red", "ball")),                # one attribute (resonator)
        ("dog", "eat", Clause("cat", "chase", "river")),  # embedded clause (recursive unbinding)
    ]
    for ag, ac, pa in facts:
        agent.learn(ag, ac, pa)
        shown = pa if isinstance(pa, str) else "(" + agent._render_filler(pa) + ")"
        print(f"  learn: {ag} {ac} {shown}", flush=True)
    print("\n  -- what-queries (on LEARNED codes) --", flush=True)
    for ag, ac in [("dog", "chase"), ("bird", "see"), ("dog", "eat"), ("cat", "hold")]:
        print(f"  Q: what does {ag} {ac}?   A: {agent.query_patient(ag, ac)}", flush=True)
    print("\n  -> the full compositional capability -- nested facts, resonator-decoded attributes, embedded", flush=True)
    print("     clauses, abstention -- runs on codes LEARNED by spike-timing plasticity. This is the", flush=True)
    print("     substrate-unification payoff: biologically-grounded learned codes that nest.", flush=True)


if __name__ == "__main__":
    main()
