"""Pure FIXED-bar no-confab-preservation + faithfulness + NON-
DEGENERACY verdict for Generator-H. Mirrors the generator_g_core /
subword_lm_gate_core DISCIPLINE (fixed bars, never tuned, multi-seed,
fail-closed) but holds Generator-H's OWN frozen constants -- does NOT
import/modify generator_g_core / song_g1_core / subword_lm_gate_core /
gate_core. ungrounded_entity_rate + is_answered + FUNCTION_WORDS are
re-implemented IDENTICALLY to generator_g_core for DIRECT comparability
(NOT imported). Pure stdlib; CPU-unit-testable."""
from __future__ import annotations
from typing import Dict

_GH_UNGROUNDED_ENTITY_MAX = 0.20      # same value as Gen-G _GG (compar.)
_GH_MIN_COVERAGE = 1.0                # every stored content word appears
_GH_MAX_REPEAT = 0.50                 # loop-collapse FAIL (probe-derived)
_GH_MIN_GROUNDED_ANSWER_RATE = 0.5    # anti-trivial-always-abstain
_GH_MIN_SEEDS = 3

FUNCTION_WORDS = frozenset((
    "a an the is are was were be been to of in on at and or but it "
    "its he she they them his her their this that with for as you i "
    "we my your we had has have did do does not no yes very so then "
    "one day").split())


def _norm(s):
    import re
    out = []
    for w in str(s).split():
        t = re.sub(r"[^\w]", "", w.lower())
        if t:
            out.append(t)
    return out


def ungrounded_entity_rate(response_text, retrieved_text,
                           function_words=FUNCTION_WORDS) -> float:
    """IDENTICAL to generator_g_core.ungrounded_entity_rate (re-
    implemented for comparability, NOT imported)."""
    ret = set(_norm(retrieved_text))
    content = [w for w in _norm(response_text)
               if w not in function_words]
    if not content:
        return 0.0
    ungrounded = sum(1 for w in content if w not in ret)
    return ungrounded / len(content)


def is_answered(response_text, function_words=FUNCTION_WORDS) -> bool:
    """IDENTICAL to generator_g_core.is_answered (anti-vacuous)."""
    for w in _norm(response_text):
        if w not in function_words:
            return True
    return False


def coverage(response_text, retrieved_text,
             function_words=FUNCTION_WORDS) -> float:
    """Fraction of retrieved CONTENT words (retrieved minus function
    set) that appear at least once in the response. 1.0 == the stored
    fact is fully covered. Empty content -> 1.0 (vacuously covered;
    the no-confab/faithful/answered bars carry the load there)."""
    content = [w for w in _norm(retrieved_text)
               if w not in function_words]
    if not content:
        return 1.0
    resp = set(_norm(response_text))
    hit = sum(1 for w in set(content) if w in resp)
    return hit / len(set(content))


def max_repeat_ngram_fraction(response_text, n: int = 2) -> float:
    """Loop detector: 1 - distinct(n-grams)/total(n-grams). The probe's
    'and fast and fast and fast' scores high; clean text scores low.
    < n+1 tokens -> 0.0 (cannot loop)."""
    toks = _norm(response_text)
    if len(toks) < n + 1:
        return 0.0
    grams = list(zip(*[toks[i:] for i in range(n)]))
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def gh_verdict(abstain_on_ungrounded_rate, bare_moat_abstain_rate,
               grounded_answer_rate, mean_ungrounded_entity_rate,
               mean_coverage, mean_max_repeat,
               has_ungrounded_control) -> Dict:
    no_confab = (bool(has_ungrounded_control)
                 and bare_moat_abstain_rate > 0.0
                 and abstain_on_ungrounded_rate
                 >= bare_moat_abstain_rate - 1e-9)
    not_trivial = grounded_answer_rate >= _GH_MIN_GROUNDED_ANSWER_RATE
    faithful = mean_ungrounded_entity_rate <= _GH_UNGROUNDED_ENTITY_MAX
    covered = mean_coverage >= _GH_MIN_COVERAGE
    not_looped = mean_max_repeat <= _GH_MAX_REPEAT
    gate = bool(no_confab and not_trivial and faithful
                and covered and not_looped)
    return {
        "GATE": "PASS" if gate else "FAIL",
        "no_confab_preserved": bool(no_confab),
        "answers_grounded_not_trivial": bool(not_trivial),
        "grounded_faithful": bool(faithful),
        "grounded_covered": bool(covered),
        "not_loop_collapsed": bool(not_looped),
        "abstain_on_ungrounded_rate":
            float(abstain_on_ungrounded_rate),
        "bare_moat_abstain_rate": float(bare_moat_abstain_rate),
        "grounded_answer_rate": float(grounded_answer_rate),
        "mean_ungrounded_entity_rate":
            float(mean_ungrounded_entity_rate),
        "mean_coverage": float(mean_coverage),
        "mean_max_repeat": float(mean_max_repeat),
        "bars": {"ungrounded_entity_max": _GH_UNGROUNDED_ENTITY_MAX,
                 "min_coverage": _GH_MIN_COVERAGE,
                 "max_repeat": _GH_MAX_REPEAT,
                 "min_grounded_answer_rate":
                     _GH_MIN_GROUNDED_ANSWER_RATE},
    }


def gh_aggregate_multiseed(per_seed_verdicts,
                           min_seeds: int = _GH_MIN_SEEDS) -> Dict:
    n = len(per_seed_verdicts)
    eff_min = max(int(min_seeds), _GH_MIN_SEEDS)
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
