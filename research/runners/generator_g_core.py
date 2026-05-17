"""Pure FIXED-bar no-confab-preservation + grounding-faithfulness
verdict for Generator-G. Mirrors the song_g1_core/subword_lm_gate_core
DISCIPLINE (fixed bars, never tuned, multi-seed, fail-closed) but
holds Generator-G's OWN frozen constants -- does NOT import/modify
song_g1_core or subword_lm_gate_core. Pure numpy/stdlib;
CPU-unit-testable."""
from __future__ import annotations
from typing import Dict

_GG_UNGROUNDED_ENTITY_MAX = 0.20
_GG_MIN_GROUNDED_ANSWER_RATE = 0.5
_GG_MIN_SEEDS = 3

FUNCTION_WORDS = frozenset((
    "a an the is are was were be been to of in on at and or but it "
    "its he she they them his her their this that with for as you i "
    "we my your we had has have did do does not no yes very so then "
    "one day").split())


def ungrounded_entity_rate(response_text, retrieved_text,
                           function_words=FUNCTION_WORDS) -> float:
    """Fraction of response CONTENT words (not function words) that do
    NOT appear in the retrieved proposition. High == the LM invented
    ungrounded content (renamed entities / confabulated). Tokens are
    normalized (lowercased, non-word chars stripped) so punctuation
    cannot mask a confabulation or inflate a faithful echo."""
    import re
    def _norm(s):
        out = []
        for w in str(s).split():
            t = re.sub(r"[^\w]", "", w.lower())
            if t:
                out.append(t)
        return out
    ret = set(_norm(retrieved_text))
    content = [w for w in _norm(response_text)
               if w not in function_words]
    if not content:
        return 0.0
    ungrounded = sum(1 for w in content if w not in ret)
    return ungrounded / len(content)


def is_answered(response_text, function_words=FUNCTION_WORDS) -> bool:
    """A response counts as a real ANSWER only if it has >= 1 CONTENT
    word (non-function, word-chars). Empty / function-word-only /
    punctuation-only responses are NOT answers -- this prevents a
    vacuous responder from gaming grounded_answer_rate while scoring
    0.0 faithfulness. The Generator-G runner MUST compute
    grounded_answer_rate using this helper."""
    import re
    for w in str(response_text).split():
        t = re.sub(r"[^\w]", "", w.lower())
        if t and t not in function_words:
            return True
    return False


def gg_verdict(abstain_on_ungrounded_rate, bare_moat_abstain_rate,
               grounded_answer_rate, mean_ungrounded_entity_rate,
               has_ungrounded_control) -> Dict:
    no_confab = (bool(has_ungrounded_control)
                 and bare_moat_abstain_rate > 0.0
                 and abstain_on_ungrounded_rate
                 >= bare_moat_abstain_rate - 1e-9)
    not_trivial = grounded_answer_rate >= _GG_MIN_GROUNDED_ANSWER_RATE
    faithful = mean_ungrounded_entity_rate <= _GG_UNGROUNDED_ENTITY_MAX
    gate = bool(no_confab and not_trivial and faithful)
    return {
        "GATE": "PASS" if gate else "FAIL",
        "no_confab_preserved": bool(no_confab),
        "answers_grounded_not_trivial": bool(not_trivial),
        "grounded_faithful": bool(faithful),
        "abstain_on_ungrounded_rate":
            float(abstain_on_ungrounded_rate),
        "bare_moat_abstain_rate": float(bare_moat_abstain_rate),
        "grounded_answer_rate": float(grounded_answer_rate),
        "mean_ungrounded_entity_rate":
            float(mean_ungrounded_entity_rate),
        "bars": {"ungrounded_entity_max": _GG_UNGROUNDED_ENTITY_MAX,
                 "min_grounded_answer_rate":
                     _GG_MIN_GROUNDED_ANSWER_RATE},
    }


def gg_aggregate_multiseed(per_seed_verdicts,
                           min_seeds: int = _GG_MIN_SEEDS) -> Dict:
    n = len(per_seed_verdicts)
    eff_min = max(int(min_seeds), _GG_MIN_SEEDS)
    n_pass = sum(1 for v in per_seed_verdicts
                 if v.get("GATE") == "PASS")
    both_probes = (n > 0 and all(
        v.get("n_grounded", 0) > 0 and v.get("n_ungrounded", 0) > 0
        for v in per_seed_verdicts))
    gate = bool(n >= eff_min and n_pass == n and n > 0
                and both_probes)
    return {"GATE": "PASS" if gate else "FAIL", "n_seeds": n,
            "min_seeds": eff_min, "n_pass": n_pass,
            "all_have_both_probes": both_probes,
            "all_pass": (n > 0 and n_pass == n)}
