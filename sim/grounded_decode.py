"""Generator-G grounded-decode policy. The validated no-confab moat
(research.runners.abstention_gate.gate, byte-UNMODIFIED, 'gate 650')
decides answer-vs-abstain FIRST; the fluent LM is invoked ONLY on the
grounded path, conditioned on the retrieved proposition. No-confab is
preserved BY CONSTRUCTION: on the abstain path the LM object is never
touched. Pure policy; the decode delegates to a duck-typed `lm`
(TinyGPT-backed in the runner; a stand-in in tests). ASCII only."""
from __future__ import annotations


def grounded_decode(ranked, lm, tok, retrieved_text, query,
                    threshold=650.0, max_new=40, temperature=0.0):
    """ranked: list[(concept, rate, tag)] desc (validated retrieval
    output). Returns {abstained, text, retrieved}. The validated moat
    decides FIRST; the LM is touched ONLY when grounded."""
    from research.runners.abstention_gate import gate
    top = gate(ranked, threshold)
    if top is None:
        return {"abstained": True, "text": None,
                "retrieved": retrieved_text}
    prompt_ids = tok.encode(retrieved_text)
    gen_ids = lm.generate_ids(prompt_ids, int(max_new))
    return {"abstained": False,
            "text": tok.decode(gen_ids),
            "retrieved": retrieved_text}
