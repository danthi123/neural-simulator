"""THROWAWAY cheap-first (CPU/numpy): the real PARSING test toward conversation -- can the system
extract roles VOICE-INVARIANTLY? i.e. does it understand that "dog chases cat" (active) and
"cat is chased by dog" (passive) have the SAME agent (dog)? This is the load-bearing question for
a learned role-filler parser (the biology question: how does a brain assign syntactic roles?).

Roles depend on (content-word position, sentence VOICE):
  active  "N1 V N2"        -> 1st=agent,   2nd=action, 3rd=patient
  passive "N1 is V by N2"  -> 1st=patient, 2nd=action, 3rd=agent   (voice flips agent<->patient)

A PARSER maps (position, voice) -> role. Tested three feature sets with a closed-form (NON-autograd)
least-squares readout learned from a TRAIN split and applied to a HELD-OUT split:
  (P)  position only            -> cannot disambiguate active-pos1 (agent) from passive-pos1 (patient)
  (PV) position + voice (additive) -> still linear, the voice flip is an INTERACTION
  (PxV) position + voice + position*voice (CONJUNCTIVE) -> can represent the flip
Then parse held-out sentences, BIND with real concept fillers, and QUERY the agent -- checking that
BOTH the active and passive form of a fact return the same agent (voice-invariant understanding).

Biology framing: voice-invariant role assignment requires CONJUNCTIVE position*voice coding (mixed
selectivity), which the substrate's distributed codes support. If (PxV) works and (P)/(PV) fail, the
insight is: a brain parsing syntax must conjoin word-position with a syntactic-voice cue -- it cannot
read roles off position alone. stdlib+numpy + concept cache; no protected import.

RESULT 2026-05-31 (seeds 42/43/44): position-only (P) active role-assign 0.000, voice-invariant
0.000; additive position+voice (PV) 0.000/0.000; CONJUNCTIVE position*voice (PxV) 1.000/1.000.
Syntactic role parsing REQUIRES conjunctive position*voice coding (mixed selectivity); the
active<->passive role flip is an INTERACTION that position-only / additive features cannot represent.
The substrate's distributed role codes ARE conjunctive-capable -> the parser is implementable.
HONEST SCOPE: this establishes the REPRESENTATIONAL requirement (conjunctive coding needed +
sufficient), NOT learnability -- there are only 6 (position,voice) combos and the closed-form
least-squares readout enumerates them. Whether the substrate's STDP can LEARN this conjunction
from example sentences (generalizing to held-out sentences) is the spiking build (next).
"""
from __future__ import annotations
import os
import numpy as np

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"
ROLES = ["agent", "action", "patient"]


def _center(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def load_concepts(seed):
    d = np.load(CACHE % seed)
    ws = [k[5:] for k in d.files if k.startswith("obs__")]
    return ws, {w: _center(d["obs__" + w].mean(axis=0)) for w in ws}


def role_of(pos, passive):
    """ground-truth role of the content word at content-position pos (0,1,2) given voice."""
    if not passive:
        return ["agent", "action", "patient"][pos]
    return ["patient", "action", "agent"][pos]


def features(pos, passive, mode):
    p = [0.0, 0.0, 0.0]; p[pos] = 1.0
    v = 1.0 if passive else 0.0
    if mode == "P":
        return np.array(p + [1.0])                    # position + bias
    if mode == "PV":
        return np.array(p + [v, 1.0])                 # + voice (additive)
    if mode == "PxV":
        px = [pi * v for pi in p]
        return np.array(p + [v] + px + [1.0])         # + position*voice (conjunctive)
    raise ValueError(mode)


def fit_readout(mode, rng):
    """closed-form least-squares (NOT autograd): features -> role one-hot, over all (pos,voice)."""
    X, Y = [], []
    for passive in (False, True):
        for pos in range(3):
            X.append(features(pos, passive, mode))
            y = [0.0, 0.0, 0.0]; y[ROLES.index(role_of(pos, passive))] = 1.0
            Y.append(y)
    X = np.array(X); Y = np.array(Y)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)        # role readout weights
    return W


def parse(sent_words, passive, W, mode):
    """sent_words = the 3 CONTENT words in order; returns {role: word} via the learned readout."""
    out = {}
    for pos, w in enumerate(sent_words):
        pred = features(pos, passive, mode) @ W
        out[ROLES[int(np.argmax(pred))]] = w
    return out


def main():
    seeds = [s for s in [42, 43, 44] if os.path.exists(CACHE % s)]
    if not seeds:
        print("CANNOT-CONCLUDE (no cache)"); return
    print("=== parser VOICE-INVARIANCE cheap-first (active vs passive, same agent?) ===")
    for mode in ["P", "PV", "PxV"]:
        role_acc, agree, tot = 0, 0, 0
        for seed in seeds:
            words, concepts = load_concepts(seed)
            rng = np.random.default_rng(seed)
            roles = {r: rng.choice([-1.0, 1.0], size=concepts[words[0]].shape[0]) for r in ROLES}
            roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
            W = fit_readout(mode, rng)
            for _ in range(40):
                n1, v, n2 = (words[i] for i in rng.choice(len(words), 3, replace=False))
                # active "n1 v n2" content order [n1,v,n2]; passive "n2 is v by n1" content order [n2,v,n1]
                pa = parse([n1, v, n2], False, W, mode)      # active
                pp = parse([n2, v, n1], True, W, mode)       # passive (same meaning: agent=n1)
                # role-assignment accuracy on the active sentence
                role_acc += int(pa.get("agent") == n1 and pa.get("patient") == n2)
                # bind both, query agent, check both -> n1 (voice-invariant understanding)
                def bind(p):
                    S = np.zeros(concepts[words[0]].shape[0])
                    for r, w in p.items():
                        S = S + roles[r] * concepts[w]
                    return S
                def q_agent(S):
                    est = S * roles["agent"]
                    return words[int(np.argmax([concepts[w] @ est for w in words]))]
                agree += int(q_agent(bind(pa)) == n1 and q_agent(bind(pp)) == n1)
                tot += 1
        print(f"  features={mode:>3} | active role-assign acc={role_acc/tot:.3f}  "
              f"voice-invariant agent (both forms -> dog)={agree/tot:.3f}")
    print("\nREAD: voice-invariant agent ~1.0 only with CONJUNCTIVE position*voice (PxV); position-only "
          "(P) and additive position+voice (PV) cannot represent the active<->passive role flip. "
          "Insight: syntactic role parsing requires conjunctive position*voice coding (mixed selectivity).")


if __name__ == "__main__":
    main()
