"""Pure pre-registered scoring/verdict core for Generator-S. Mirrors
the song_g1_core pure-verdict DISCIPLINE (fixed bars, never tuned;
control load-bearing; >=3 seeds) but holds Generator-S's OWN frozen
constants -- it does NOT import or modify song_g1_core. Pure
numpy/stdlib; CPU-unit-testable; no IO, no heavy import."""
from __future__ import annotations
import math
from typing import Dict, List

_GS_PPL_MARGIN = 0.20
_GS_GENERALIZATION_MAX = 1.5
_GS_DISTINCT_MIN = 0.5
_GS_COPY_MAX = 0.20
_GS_MIN_SEEDS = 3


def perplexity(nll_per_token: List[float]) -> float:
    if not nll_per_token:
        return float("inf")
    return float(math.exp(sum(nll_per_token) / len(nll_per_token)))


def shuffled_token_control(token_ids, rng):
    out = list(token_ids)
    rng.shuffle(out)
    if out == list(token_ids) and len(set(token_ids)) > 1:
        out.reverse()
    return out


def distinct_ngram_ratio(ids: List[int], n: int = 3) -> float:
    if len(ids) < n:
        return 0.0
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / len(grams)


def verbatim_copy_fraction(gen: List[int], train: List[int],
                            n: int = 8) -> float:
    if len(gen) < n:
        return 0.0
    tr = {tuple(train[i:i + n]) for i in range(len(train) - n + 1)}
    gg = [tuple(gen[i:i + n]) for i in range(len(gen) - n + 1)]
    if not gg:
        return 0.0
    return sum(1 for g in gg if g in tr) / len(gg)


def gs_verdict(heldout_ppl, shuffled_ppl, train_ppl, distinct,
               copy_frac, has_shuffled_control) -> Dict:
    real_structure = (has_shuffled_control and shuffled_ppl > 0
                      and heldout_ppl <= (1.0 - _GS_PPL_MARGIN)
                      * shuffled_ppl)
    generalizes = (train_ppl > 0
                   and heldout_ppl <= _GS_GENERALIZATION_MAX * train_ppl)
    non_degenerate = distinct >= _GS_DISTINCT_MIN
    not_copying = copy_frac <= _GS_COPY_MAX
    gate = bool(real_structure and generalizes and non_degenerate
                and not_copying)
    return {
        "GATE": "PASS" if gate else "FAIL",
        "real_structure_vs_shuffled": bool(real_structure),
        "generalizes_not_memorizes": bool(generalizes),
        "non_degenerate_generation": bool(non_degenerate),
        "not_verbatim_copying": bool(not_copying),
        "heldout_ppl": float(heldout_ppl),
        "shuffled_ppl": float(shuffled_ppl),
        "train_ppl": float(train_ppl),
        "distinct_trigram": float(distinct),
        "verbatim_copy_frac": float(copy_frac),
        "bars": {"ppl_margin": _GS_PPL_MARGIN,
                 "generalization_max": _GS_GENERALIZATION_MAX,
                 "distinct_min": _GS_DISTINCT_MIN,
                 "copy_max": _GS_COPY_MAX},
    }


def gs_aggregate_multiseed(per_seed_verdicts, min_seeds: int = _GS_MIN_SEEDS):
    n = len(per_seed_verdicts)
    n_pass = sum(1 for v in per_seed_verdicts if v.get("GATE") == "PASS")
    gate = bool(n >= int(min_seeds) and n_pass == n and n > 0)
    return {"GATE": "PASS" if gate else "FAIL", "n_seeds": n,
            "min_seeds": int(min_seeds), "n_pass": n_pass,
            "all_pass": (n > 0 and n_pass == n)}
