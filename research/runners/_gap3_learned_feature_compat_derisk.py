"""Gap #3 residual A1 — neuralize `content_bias_target`: does the ANIMACY x VERB_SELECTS feature-compatibility
(the host lexicon that decides WHICH referent to bias for a bare pronoun) EMERGE from corpus co-occurrence, so the
bias is a LEARNED feature-compatibility rather than a host lookup? (Per the emergence bar + BRAIN-BASED-ONLY.)

Cheap-first numpy mechanism test (spiking + wire-in is the follow-on if GO): from an SVO corpus, JOINTLY learn each
concept's ANIMACY and each verb's SELECTIONAL preference by iterative co-occurrence (EM-like: a concept is animate if
it is the patient of animate-selecting verbs; a verb selects-animate if its patients are animate). Then the learned
feature-compatibility `argmax_c [ animacy(c) matches selection(verb) ]` must reproduce the host `content_bias_target`
disambiguation on held-out pronoun cases. GO: learned==host on the resolvable cases, 6-seed; anti-cheats: a
PERMUTED-corpus (shuffle patient animacy) collapses the learned map (the structure is corpus-derived, not smuggled);
NO host ANIMACY/VERB_SELECTS lexicon used by the learned path.
"""
import os, sys
import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.biased_competition_buffer import ANIMACY, VERB_SELECTS, content_bias_target  # ground truth only

# A small SVO world: animate + inanimate concepts, verbs whose THEME selects one animacy (from the host GT, used ONLY
# to GENERATE a realistic corpus + as the eval ground truth -- the LEARNED path never reads ANIMACY/VERB_SELECTS).
CONCEPTS = [c for c in ANIMACY]
ANIMATE = [c for c in CONCEPTS if ANIMACY[c] == "animate"]
INANIM = [c for c in CONCEPTS if ANIMACY[c] == "inanimate"]
VERBS = [v for v in VERB_SELECTS]


def make_corpus(seed, n=400, permute=False):
    rng = np.random.default_rng(seed)
    facts = []
    for _ in range(n):
        v = rng.choice(VERBS); want = VERB_SELECTS[v]
        agent = rng.choice(ANIMATE)                                  # agents are animate (typical)
        pool = ANIMATE if want == "animate" else INANIM
        patient = rng.choice(pool if len(pool) else CONCEPTS)
        facts.append((agent, v, patient))
    if permute:  # anti-cheat: destroy the animacy-selection structure (shuffle patients across verbs)
        pats = [f[2] for f in facts]; rng.shuffle(pats)
        facts = [(a, v, p) for (a, v, _), p in zip(facts, pats)]
    return facts


def learn_features(facts, n_iter=25):
    """JOINT co-occurrence learning (EM-like), NO host lexicon: concept animacy in [-1,1] (animate=+), verb selection
    in [-1,1]. concept <- mean selection of verbs it is a PATIENT of; verb <- mean animacy of its patients."""
    concepts = sorted({w for f in facts for w in (f[0], f[2])})
    verbs = sorted({f[1] for f in facts})
    ca = {c: 0.0 for c in concepts}; vs = {v: 0.0 for v in verbs}
    # seed the sign with a graph-Laplacian-style split so the two classes separate (unsupervised sign is arbitrary;
    # we align it to the eval at read time). init each concept by its patient-verb co-occurrence variance.
    rng = np.random.default_rng(0)
    for c in concepts:
        ca[c] = rng.standard_normal() * 0.01
    for _ in range(n_iter):
        # verb selection = mean animacy of its patients
        vnum = {v: [] for v in verbs}
        for a, v, p in facts:
            vnum[v].append(ca[p])
        for v in verbs:
            vs[v] = float(np.tanh(np.mean(vnum[v]))) if vnum[v] else 0.0
        # concept animacy = mean selection of the verbs it is a PATIENT of (+ small agent-is-animate prior)
        cnum = {c: [] for c in concepts}
        for a, v, p in facts:
            cnum[p].append(vs[v]); cnum[a].append(+0.3)             # weak "agents tend animate" prior
        for c in concepts:
            ca[c] = float(np.tanh(np.mean(cnum[c]))) if cnum[c] else 0.0
    return ca, vs


def learned_bias_target(candidates, query_verb, ca, vs, sign):
    """Learned feature-compatibility: pick the candidate whose learned animacy sign matches the verb's learned
    selection sign. Returns the concept, or None (no/ambiguous match -> the moat abstains)."""
    if query_verb not in vs:
        return None
    want = np.sign(vs[query_verb] * sign)
    scored = [(c, np.sign(ca.get(c, 0.0) * sign)) for c in candidates]
    match = [c for c, s in scored if s == want and want != 0]
    return match[0] if len(match) == 1 else None


def run_seed(seed, permute=False):
    facts = make_corpus(seed, permute=permute)
    ca, vs = learn_features(facts)
    # align the arbitrary unsupervised sign to the ground-truth animacy (a global 1-bit flip, not per-item)
    agree = np.mean([np.sign(ca.get(c, 0.0)) == (1 if ANIMACY[c] == "animate" else -1) for c in CONCEPTS if ca.get(c, 0.0) != 0])
    sign = 1.0 if agree >= 0.5 else -1.0
    # eval: for each verb, a 2-candidate case (one animate + one inanimate) -> learned must match host content_bias_target
    rng = np.random.default_rng(seed * 3 + 1)
    ok = n = 0
    for v in VERBS:
        for _ in range(6):
            a = rng.choice(ANIMATE); i = rng.choice(INANIM); cands = [a, i]; rng.shuffle(cands)
            host = content_bias_target(cands, v)
            learned = learned_bias_target(cands, v, ca, vs, sign)
            if host is not None:                                    # only score resolvable cases
                ok += int(learned == host); n += 1
    return ok / n if n else 0.0


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    seeds = (42, 43, 44, 100, 101, 102)
    acc = [run_seed(s) for s in seeds]
    perm = [run_seed(s, permute=True) for s in seeds]
    ma, mp = float(np.mean(acc)), float(np.mean(perm))
    print(f"[gap#3 A1 learned feature-compatibility] concepts={len(CONCEPTS)} verbs={len(VERBS)}")
    for s, a, p in zip(seeds, acc, perm):
        print(f"  [seed {s}] learned==host {a:.2f} | permuted-corpus {p:.2f}")
    go = ma >= 0.80 and mp <= 0.60
    print(f"  MEAN(6): learned==host {ma:.2f} | permuted-corpus {mp:.2f} (must collapse) -> {'GO' if go else 'NO'}")
    print(f"  GO: learned reproduces the host content_bias_target disambiguation >=0.80 AND permuted collapses <=0.60")


if __name__ == "__main__":
    main()
