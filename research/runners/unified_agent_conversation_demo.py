"""(iv) A richer MULTI-TURN conversational demo on the unified agent, on SENSORY-GROUNDED concept codes.

Where the agent's own `main()` runs sectioned phases (all-learn, then all-query), this scripts a single
INTERLEAVED dialogue -- statements and questions mixed across turns, the way a real conversation flows -- and the
concept codes are GROUNDED in the real V1 Gabor bank (sim/visual_cortex.py) + ventral-hierarchy decorrelation (the
#4 recipe that matched constructed codes). One transcript exercises every capability the session validated:

  - comprehend a statement -> learn the fact (flat, one/two-attribute, or an embedded clause);
  - answer a what-query by COMPOSITION (the patient's structure auto-detected + decoded);
  - answer a who-query;
  - ABSTAIN on the unknown (no confabulation -- the no-confab moat);
  - elaborate on a topic (dialogue planning via the content-selection Control over the agent's own facts).

Honest scope: composition + the no-confab moat are validated IN SPIKES (the grounded-spiking capstone = 100% core);
the dialogue-PLANNING layer here is the numpy content-selection Control (validated separately in spikes as the
spiking content-selection arc). Concept codes are real-V1-grounded; the per-concept stimuli are synthetic distinct
textures (no natural images for abstract words). Reuse-by-import. numpy/CPU.

  python -m research.runners.unified_agent_conversation_demo
"""
from __future__ import annotations
import numpy as np

import research.runners.unified_agent_visual_grounded as uvg
from research.runners._visual_grounding_probe import _v1_matrix
from research.runners.unified_agent_visual_grounded import _v1_codes_for_tokens, _decorrelate
from research.runners.nested_composition_agent import NestedCompositionAgent, Clause

D = 2048


def _grounded_codes(tokens, seed=42):
    """Real V1 Gabor response per concept -> ventral-hierarchy decorrelation -> phase angles (the agent's
    external_codes format). The concept codes are SENSORY-grounded."""
    uvg.STIMULUS_MODE = "tiled"
    W, n_v1 = _v1_matrix()
    v1 = _decorrelate(_v1_codes_for_tokens(tokens, W))
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((D, n_v1)) + 1j * rng.standard_normal((D, n_v1))
    Z = proj @ v1.T
    return {t: np.angle(Z[:, i]) for i, t in enumerate(tokens)}


def main():
    nouns = ["dog", "cat", "ball", "bird", "river", "child", "apple", "bread"]
    verbs = ["chase", "hold", "see", "eat", "want", "give"]
    adjs = ["big", "small", "red", "cold", "fast", "soft"]
    ext = _grounded_codes(nouns + verbs + adjs)
    a = NestedCompositionAgent(nouns, verbs, adjs, D=D, seed=42, external_codes=ext)

    print("=== unified conversational agent | concept codes GROUNDED in real V1 Gabor + decorrelation ===", flush=True)
    print("    (one interleaved multi-turn dialogue; composition + no-confab validated in spikes, #4 capstone)\n", flush=True)

    def render(pa):
        return pa if isinstance(pa, str) else a._render_filler(pa)

    # A single interleaved dialogue: USER turns are statements (learn) or questions; SIM answers by composition,
    # abstains on the unknown, and elaborates on a topic.
    script = [
        ("learn", ("dog", "chase", "cat")),
        ("what",  ("dog", "chase")),                                   # flat
        ("learn", ("child", "hold", ("red", "ball"))),                 # one attribute
        ("what",  ("child", "hold")),
        ("ask_unknown", ("dog", "eat")),                               # never stated -> ABSTAIN (no confab)
        ("learn", ("cat", "want", (("big", "red"), "ball"))),          # two attributes
        ("what",  ("cat", "want")),
        ("learn", ("bird", "see", Clause("cat", "chase", ("cold", "river")))),  # embedded clause (attributed arg)
        ("what",  ("bird", "see")),
        ("learn", ("dog", "see", Clause("child", "give", "bread"))),   # embedded clause (flat args)
        ("what",  ("dog", "see")),
        ("who",   ("chase", "cat")),                                   # who-query
        ("ask_unknown", ("river", "chase")),                           # in-vocab pair never stored -> ABSTAIN
        ("topic", "dog"),                                              # dialogue planning: elaborate on a topic
    ]

    for kind, payload in script:
        if kind == "learn":
            ag, ac, pa = payload
            a.learn(ag, ac, pa)
            print(f"  user>  {ag} {ac} {render(pa)}.", flush=True)
            print(f"  sim >  ok, learned.", flush=True)
        elif kind == "what":
            ag, ac = payload
            print(f"  user>  what does {ag} {ac}?", flush=True)
            print(f"  sim >  {render(a.query_patient(ag, ac))}.", flush=True)
        elif kind == "who":
            ac, pa = payload
            print(f"  user>  who {ac} {pa}?", flush=True)
            print(f"  sim >  {a.query_agent(ac, pa)}.", flush=True)
        elif kind == "ask_unknown":
            ag, ac = payload
            got = a.query_patient(ag, ac)
            ans = "(I don't know -- you never told me.)" if got is None else render(got) + "."
            print(f"  user>  what does {ag} {ac}?", flush=True)
            print(f"  sim >  {ans}", flush=True)
        elif kind == "topic":
            a.set_topic(payload)
            print(f"  user>  tell me about {payload}.", flush=True)
            said = []
            while True:
                e = a.elaborate()
                if e is None:
                    break
                said.append(e)
            for s in said:
                print(f"  sim >  {s}.", flush=True)
            print(f"  sim >  (that's everything I know about {payload}.)", flush=True)
        print(flush=True)

    print("  -> one agent, one interleaved conversation: comprehend statements -> learn; answer who/what by", flush=True)
    print("     COMPOSITION (flat / one-or-two-attribute / embedded clause, auto-detected); ABSTAIN on the unknown;", flush=True)
    print("     elaborate on a topic -- on concept codes derived from a real biological V1 receptive-field bank.", flush=True)


if __name__ == "__main__":
    main()
