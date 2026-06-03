"""Cheap-first payoff DEMONSTRATION (Direction A): does the resonator decode a genuine SEMANTIC nested fact
on the phasor FHRR substrate -- where single-shot gives the documented 0.000-class nesting failure?

The structure (a real nesting a flat substrate cannot do): an SVO fact whose PATIENT is an ATTRIBUTED ENTITY
  fact = AGENT(x)dog  +  ACTION(x)chase  +  PATIENT(x)( adj (x) noun )        ("dog chase  (big cat)")
The patient slot's filler is ITSELF a bound product (adj (x) noun = "big cat"). To recover the attributed
patient you must, after unbinding PATIENT, FACTOR the product (adj (x) noun) into its (adjective, noun) -- a
factorization (search M_adj x M_noun) that single-shot decode cannot do (it returns the still-bound product,
which cleans up to garbage = the 0.000-class nesting failure). The resonator factors it.

This is on phasor FHRR (where the resonator was validated -- the real-Hadamard 320 substrate cannot do this,
per _resonator_real320_probe). numpy phasor algebra (the cheap-first level; the spiking resonator already
RESOLVED). Tests crosstalk robustness too: the patient product is recovered DESPITE the bundle crosstalk from
the agent+action bindings.

PRE-REGISTERED, FROZEN: D=1024; M_noun=M_adj=M_verb=16; n_trials=40; resonator n_iter=120.
  resonator success = recovers BOTH the patient's (adjective, noun) correctly.
  single-shot control = unbind PATIENT then clean up against the NOUN vocab (the flat-fact decode that works
    for non-nested facts) -- recovers the patient noun? (It cannot: the unbound value is adj(x)noun, not a
    noun.) Measured as its accuracy on the patient noun.
  THREE-STATE:
    RESOLVES := resonator >= 0.90 (decodes the nested attributed patient) AND single-shot < 0.50 (the flat
                decode fails on the nesting) -> the resonator unlocks nested-fact understanding on phasor FHRR.
    BOUNDARY := resonator < 0.90 (the bundle crosstalk or capacity breaks the nested decode).
    CANNOT-CONCLUDE := single-shot also succeeds (the structure wasn't genuinely nested).

  python -m research.findings.raw._resonator_nested_fact_probe
"""
import numpy as np

D = 1024
M = 16                 # vocab size per kind (nouns / adjectives / verbs)
N_TRIALS = 40
N_ITER = 120


def _unit(v):
    return v / (np.abs(v) + 1e-12)


def phasor_codebook(K, rng):
    return np.exp(1j * rng.uniform(-np.pi, np.pi, size=(D, K)))


def resonator_factor(C, cb_adj, cb_noun, n_iter):
    """Factor C ~ adj (x) noun into (adj_idx, noun_idx) by the phasor resonator."""
    ea = _unit(cb_adj.sum(1)); en = _unit(cb_noun.sum(1))
    for _ in range(n_iter):
        xa = C * np.conj(en); ea = _unit(cb_adj @ (cb_adj.conj().T @ xa))
        xn = C * np.conj(ea); en = _unit(cb_noun @ (cb_noun.conj().T @ xn))
    return int(np.argmax(np.abs(cb_adj.conj().T @ ea))), int(np.argmax(np.abs(cb_noun.conj().T @ en)))


def main():
    print(f"=== resonator on a SEMANTIC nested fact (phasor FHRR; D={D}, M={M}/kind) ===", flush=True)
    print("    fact = AGENT(x)noun + ACTION(x)verb + PATIENT(x)(adj(x)noun); decode the attributed patient.",
          flush=True)
    res_ok = ctrl_ok = 0
    for t in range(N_TRIALS):
        rng = np.random.default_rng(2024 + t)
        roles = {r: _unit(np.exp(1j * rng.uniform(-np.pi, np.pi, size=D)))
                 for r in ("AGENT", "ACTION", "PATIENT")}
        nouns = phasor_codebook(M, rng)
        adjs = phasor_codebook(M, rng)
        verbs = phasor_codebook(M, rng)
        # plant a fact
        ag = int(rng.integers(0, M)); ac = int(rng.integers(0, M))
        p_adj = int(rng.integers(0, M)); p_noun = int(rng.integers(0, M))
        patient_entity = adjs[:, p_adj] * nouns[:, p_noun]                   # NESTED: adj (x) noun
        fact = (roles["AGENT"] * nouns[:, ag]
                + roles["ACTION"] * verbs[:, ac]
                + roles["PATIENT"] * patient_entity)                        # bundle of 3 role-bindings
        # unbind the PATIENT role -> the attributed entity (a product) + bundle crosstalk
        p = fact * np.conj(roles["PATIENT"])
        # resonator: factor the attributed patient into (adjective, noun)
        a_hat, n_hat = resonator_factor(_unit(p), adjs, nouns, N_ITER)
        res_ok += int(a_hat == p_adj and n_hat == p_noun)
        # single-shot control: clean up p against the NOUN vocab (the flat-fact decode) -> the patient noun?
        ctrl_noun = int(np.argmax(np.abs(nouns.conj().T @ _unit(p))))
        ctrl_ok += int(ctrl_noun == p_noun)
    res = res_ok / N_TRIALS
    ctrl = ctrl_ok / N_TRIALS
    print(f"\n  resonator decodes the attributed patient (adj AND noun): {res:.2f}", flush=True)
    print(f"  single-shot flat decode recovers the patient noun:        {ctrl:.2f}  (chance {1/M:.3f})",
          flush=True)
    if res >= 0.90 and ctrl < 0.50:
        verdict = ("RESOLVES -- the resonator decodes the NESTED attributed fact on phasor FHRR (the patient "
                   "= adj(x)noun is recovered as BOTH its adjective and noun) where the flat single-shot "
                   "decode fails (the 0.000-class nesting failure). Nested-fact understanding is unlocked on "
                   "the phasor substrate, crosstalk-robust.")
    elif res < 0.90:
        verdict = f"BOUNDARY -- resonator {res:.2f} < 0.90; bundle crosstalk or capacity breaks the nested decode."
    else:
        verdict = "CANNOT-CONCLUDE -- single-shot also succeeds; the structure was not genuinely nested."
    print(f"\nVERDICT: {verdict}", flush=True)
    return verdict


if __name__ == "__main__":
    main()
