"""Generator-H constrained-realization policy. The validated no-confab
moat (research.runners.abstention_gate.gate, byte-UNMODIFIED, 'gate
650') decides answer-vs-abstain FIRST; on abstain the lm object is
NEVER touched (no-confab BY CONSTRUCTION -- spy-LM pure-testable). On
grounded, the realizer decodes greedily with per-step logits
HARD-MASKED to {retrieved proposition token ids} U {closed function
set}, so a NON-allowed id can never be argmax-selected (faithfulness
BY CONSTRUCTION -- a provable unit test, not just a measured bar), plus
no-repeat-ngram loop-blocking + coverage-stop. Pure stdlib; the lm is
duck-typed (`lm.logits(seq_ids)->list[float]`), reusing the validated
TinyGPT via the gate runner's adapter -- NOT reimplemented here.
Mirrors sim/grounded_decode.py's SHAPE; does NOT import or modify
grounded_decode / generator_g_core. ASCII only."""
from __future__ import annotations


def _allowed_ids(tok, retrieved_text, function_words):
    allowed = set(tok.encode(retrieved_text))
    for fw in function_words:
        allowed.update(tok.encode(fw))
    return allowed


def constrained_realize(ranked, lm, tok, retrieved_text, query,
                        function_words, threshold=650.0,
                        no_repeat_ngram=3, max_new=40):
    """ranked: list[(concept, rate, tag)] desc (validated retrieval
    output). Returns {abstained, text, retrieved}. The validated moat
    decides FIRST; the lm is touched ONLY when grounded."""
    from research.runners.abstention_gate import gate
    top = gate(ranked, threshold)
    if top is None:
        return {"abstained": True, "text": None,
                "retrieved": retrieved_text}

    allowed = _allowed_ids(tok, retrieved_text, function_words)
    fn_ids = set()
    for fw in function_words:
        fn_ids.update(tok.encode(fw))
    content_ids = set(tok.encode(retrieved_text)) - fn_ids
    allowed_sorted = sorted(allowed)
    if not allowed_sorted:
        return {"abstained": False, "text": "",
                "retrieved": retrieved_text}

    prompt_ids = tok.encode(retrieved_text)
    seq = list(prompt_ids) if prompt_ids else [allowed_sorted[0]]
    out = []
    covered = set()
    k = max(1, int(no_repeat_ngram))

    for _ in range(int(max_new)):
        logits = lm.logits(seq)
        banned = set()
        if k >= 2 and len(out) >= k - 1:
            prefix = tuple(out[-(k - 1):])
            for i in range(len(out) - (k - 1)):
                if tuple(out[i:i + k - 1]) == prefix:
                    banned.add(out[i + k - 1])
        best_id, best_v = None, None
        for cid in allowed_sorted:
            if cid in banned:
                continue
            v = logits[cid]
            if best_v is None or v > best_v:
                best_v, best_id = v, cid
        if best_id is None:                 # all allowed banned
            for cid in allowed_sorted:
                v = logits[cid]
                if best_v is None or v > best_v:
                    best_v, best_id = v, cid
        seq.append(best_id)
        out.append(best_id)
        if best_id in content_ids:
            covered.add(best_id)
        if content_ids and covered >= content_ids:
            break                            # coverage-stop

    return {"abstained": False, "text": tok.decode(out),
            "retrieved": retrieved_text}
