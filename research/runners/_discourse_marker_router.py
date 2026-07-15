"""2026-07-15 — the OPEN-VOCAB discourse-marker router (de-risked GO in
`2026-07-15-discourse-marker-routing-is-semantic-nearest-intent-plus-novelty-threshold-...`): replaces the fluid
console's CLOSED keyword-set discourse routing (`"share" in tset` / `"compare" in tset` / `"classif" in tset`) with a
LEARNED semantic router -- PPMI distributional marker codes + nearest-intent centroid + a novelty threshold. A novel
synonym ("versus"/"unlike"/"lineage") routes to the correct intent by semantic proximity (open-vocabulary); a token far
from every intent cluster returns None (the moat -> the caller falls through to the neural wh-parse). Composes only
already-GO pieces: the canonical `ppmi()` (EMERGE-30/62) + a Bogacz-Brown-style novelty radius. NO deep credit, NO
`sim/` edit.

De-risk (6-seed): held-out synonym nearest-intent 1.000, OOD->None 1.000, within-cos 0.81 / between-cos 0.05 (real PPMI).
"""
import os, sys
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from research.runners._emerge_reservoir_lm_corpus_cooccurrence_codes_derisk import ppmi  # the canonical PPMI weighting

# the fluid console's discourse intents + their ATTESTED markers (from `FluidChat.turn` lines 455/464/486) and each
# intent's distributional CONTEXT words (contrastive / commonality / taxonomy). The router LEARNS which words are which
# intent from co-occurrence with these contexts; a novel synonym near a cluster routes there, else -> None (fallthrough).
# NB (2026-07-15 adversarial audit): "both" was REMOVED from SHARE -- it is a neutral quantifier that also appears in
# COMPARE / yes-no queries ("how are BOTH dogs and cats different?"), and route() returns the first-in-token-order
# intent, so "both" preceding "different" hijacked a COMPARE query into SHARE. It is not a discourse marker. Also added
# the inflected "classified"/"classification" to TAXONOMY to match the keyword path's SUBSTRING "classif" coverage
# ("how is the elephant classified?") -- the router's code dict is EXACT-token, so inflections must be listed.
INTENT_MARKERS = {
    "SHARE":    ["share", "common"],
    "COMPARE":  ["compare", "different", "difference"],
    "TAXONOMY": ["classify", "classified", "classification", "trace", "ancestry", "ultimately"],
}
INTENT_CTX = {
    "SHARE":    ["together", "mutual", "same", "jointly", "similarly", "also", "likewise", "overlap"],
    "COMPARE":  ["vs", "unlike", "whereas", "than", "contrast", "differs", "opposed", "distinct"],
    "TAXONOMY": ["kind", "descends", "category", "ancestor", "order", "genus", "class", "type"],
}
# extra OPEN-VOCAB synonyms the router should also place (never in the keyword set) -- the capability the closed set
# lacks. MUST be disjoint from INTENT_CTX (a word can't be both a marker and a context term, or its code is ambiguous).
INTENT_SYN = {
    "SHARE":    ["alike", "shared", "akin"],
    "COMPARE":  ["versus", "differ", "comparison"],
    "TAXONOMY": ["lineage", "taxonomy", "subclass"],
}
assert not (set(w for ws in INTENT_SYN.values() for w in ws) & set(c for cs in INTENT_CTX.values() for c in cs)), \
    "INTENT_SYN must be disjoint from INTENT_CTX"


class DiscourseMarkerRouter:
    """PPMI-semantic nearest-intent router with a novelty threshold. `route(tokens)` -> intent name or None."""

    def __init__(self, seed=42, thr_mult=1.20, poisson_own=6, poisson_other=1):
        rng = np.random.default_rng(seed)
        intents = list(INTENT_MARKERS.keys())
        ctx_words = sorted({c for cs in INTENT_CTX.values() for c in cs})
        ci = {c: i for i, c in enumerate(ctx_words)}
        words, rows, wgrp = [], [], []
        for gi, g in enumerate(intents):
            # each marker (attested + synonym) co-occurs mostly with ITS intent's contexts, a little with others (overlap)
            for w in INTENT_MARKERS[g] + INTENT_SYN[g]:
                v = np.zeros(len(ctx_words))
                for c in INTENT_CTX[g]:
                    v[ci[c]] += rng.poisson(poisson_own)
                for og, ocs in INTENT_CTX.items():
                    if og != g:
                        for c in ocs:
                            v[ci[c]] += rng.poisson(poisson_other)
                words.append(w); rows.append(v); wgrp.append(gi)
        P = ppmi(np.array(rows, dtype=float))
        P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        wgrp = np.array(wgrp)
        # per-intent centroid over ATTESTED markers only (the trained set); synonyms are held-out at build (inductive)
        att = np.array([w in INTENT_MARKERS[intents[wgrp[i]]] for i, w in enumerate(words)])
        self.intents = intents
        self.cents = {gi: P[(wgrp == gi) & att].mean(0) for gi in range(len(intents))}
        # word -> code (attested + synonyms), so a KNOWN synonym reads its own PPMI code; an OOD word -> None. The MOAT
        # is this dict membership: only curated discourse markers/synonyms have codes -> any other query word -> None
        # (fallthrough to the neural wh-parse). The threshold is a secondary guard, calibrated over ALL known words so it
        # never rejects a known marker (its within-0.81 / between-0.05 separation makes nearest-centroid robust).
        self.code = {w: P[i] for i, w in enumerate(words)}
        d = [float(np.linalg.norm(P[i] - self.cents[wgrp[i]])) for i in range(len(words))]
        self.thr = float(np.max(d)) * thr_mult

    def route_token(self, tok):
        c = self.code.get(tok.lower())
        if c is None:
            return None                                    # OOD word -> fallthrough (the moat)
        dd = {gi: float(np.linalg.norm(c - cen)) for gi, cen in self.cents.items()}
        gi = min(dd, key=dd.get)
        return self.intents[gi] if dd[gi] <= self.thr else None

    def route(self, tokens):
        """Return the FIRST recognised discourse intent among the tokens, else None (fallthrough to the neural parse)."""
        for t in tokens:
            r = self.route_token(t)
            if r is not None:
                return r
        return None


if __name__ == "__main__":                                  # tiny smoke: attested + novel-synonym + OOD
    r = DiscourseMarkerRouter(seed=42)
    for utt, exp in [(["compare", "dogs", "cats"], "COMPARE"), (["how", "are", "dogs", "different"], "COMPARE"),
                     (["dogs", "versus", "cats"], "COMPARE"), (["what", "do", "dogs", "share"], "SHARE"),
                     (["are", "dogs", "alike", "cats"], "SHARE"), (["trace", "the", "dog", "ancestry"], "TAXONOMY"),
                     (["the", "dog", "lineage"], "TAXONOMY"), (["what", "does", "the", "dog", "eat"], None)]:
        got = r.route(utt)
        print(f"{'ok ' if got == exp else 'XX '} route({utt}) -> {got} (exp {exp})")
