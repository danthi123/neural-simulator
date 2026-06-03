"""Biology-faithful capstone of Direction A: does a SEMANTIC nested fact decode on the genuine spiking
resonate-and-fire substrate (not just numpy phasor algebra)? Unifies the two validated pieces -- the spiking
resonator (validated on abstract factors, _spiking_resonator_probe) and the nested-fact capability (validated
in numpy phasor, _resonator_nested_fact_probe + the agent) -- into one test: a nested fact built and decoded
ENTIRELY with the resonate-and-fire FHRR ops (rf_bind / rf_unbind / rf_bundle), crosstalk-robust.

  fact = AGENT(x)noun + ACTION(x)verb + PATIENT(x)( adj (x) noun )      ("dog chase (big cat)")  -- in spikes
  decode the attributed patient: rf_unbind PATIENT -> the product -> spiking resonator -> (adjective, noun).

Reuse-by-import of resonate_fire_fhrr (rf_*); no protected-module change. Small scale (rf_resonate time-steps
a full cycle per call) -- tractable, still decisive.

PRE-REGISTERED, FROZEN: D=256; M=8 (per kind); n_trials=12; resonator n_iter=20.
  success = recovers BOTH the patient's (adjective, noun) in spikes.
  CONTROL = single-shot spiking flat decode (rf_unbind PATIENT, clean up vs the noun vocab) -- the patient
    noun? (cannot: the unbound value is the product adj(x)noun.)
  THREE-STATE:
    RESOLVES := spiking resonator >= 0.80 (decodes the nested attributed fact IN SPIKES) AND single-shot
                control < 0.50 -> nested-fact understanding is biology-faithful (resonate-and-fire), crosstalk-robust.
    BOUNDARY := spiking resonator < 0.80 (the spiking realization + bundle crosstalk breaks the nested decode).
    CANNOT-CONCLUDE := single-shot also succeeds.

  python -m research.findings.raw._spiking_nested_fact_probe
"""
import numpy as np

from research.runners.resonate_fire_fhrr import rf_bind, rf_unbind, rf_resonate, _to_phasor, CYCLE_STEPS
from research.runners.spiking_phasor_fhrr import phases_to_spikes

D = 256
M = 8
N_TRIALS = 12
N_ITER = 20


def rand_code(rng):
    return phases_to_spikes(rng.uniform(0.0, 1.0, size=D), CYCLE_STEPS)


def cbmat(cb):
    return np.stack([_to_phasor(c, CYCLE_STEPS) for c in cb], axis=1)


def proj_readout(spikes, S):
    z = _to_phasor(spikes, CYCLE_STEPS)
    return rf_resonate(S @ (S.conj().T @ z), CYCLE_STEPS)


def spiking_resonator_2(p_spikes, A, N, Amat, Nmat, n_iter):
    ea = rf_resonate(Amat.sum(1), CYCLE_STEPS)
    en = rf_resonate(Nmat.sum(1), CYCLE_STEPS)
    for _ in range(n_iter):
        xa = rf_unbind(p_spikes, en, CYCLE_STEPS); ea = proj_readout(xa, Amat)
        xn = rf_unbind(p_spikes, ea, CYCLE_STEPS); en = proj_readout(xn, Nmat)
    a = int(np.argmax(np.abs(Amat.conj().T @ _to_phasor(ea, CYCLE_STEPS))))
    n = int(np.argmax(np.abs(Nmat.conj().T @ _to_phasor(en, CYCLE_STEPS))))
    return a, n


def main():
    print(f"=== SPIKING nested fact (resonate-and-fire FHRR; D={D} M={M}) ===", flush=True)
    print("    fact = AGENT(x)noun + ACTION(x)verb + PATIENT(x)(adj(x)noun), decoded in spikes.", flush=True)
    res_ok = ctrl_ok = 0
    for t in range(N_TRIALS):
        rng = np.random.default_rng(500 + t)
        roles = {r: rand_code(rng) for r in ("AGENT", "ACTION", "PATIENT")}
        nouns = [rand_code(rng) for _ in range(M)]
        adjs = [rand_code(rng) for _ in range(M)]
        verbs = [rand_code(rng) for _ in range(M)]
        Nmat, Amat = cbmat(nouns), cbmat(adjs)
        ag, ac = int(rng.integers(0, M)), int(rng.integers(0, M))
        pa, pn = int(rng.integers(0, M)), int(rng.integers(0, M))
        patient = rf_bind(adjs[pa], nouns[pn], CYCLE_STEPS)             # NESTED: adj (x) noun, in spikes
        # bundle the three role-bindings (rf_bundle of the bound symbols)
        from research.runners.resonate_fire_fhrr import rf_bundle
        fact = rf_bundle([rf_bind(roles["AGENT"], nouns[ag], CYCLE_STEPS),
                          rf_bind(roles["ACTION"], verbs[ac], CYCLE_STEPS),
                          rf_bind(roles["PATIENT"], patient, CYCLE_STEPS)], CYCLE_STEPS)
        p = rf_unbind(fact, roles["PATIENT"], CYCLE_STEPS)             # attributed entity + bundle crosstalk
        a_hat, n_hat = spiking_resonator_2(p, adjs, nouns, Amat, Nmat, N_ITER)
        res_ok += int(a_hat == pa and n_hat == pn)
        ctrl_noun = int(np.argmax(np.abs(Nmat.conj().T @ _to_phasor(p, CYCLE_STEPS))))   # single-shot flat
        ctrl_ok += int(ctrl_noun == pn)
    res, ctrl = res_ok / N_TRIALS, ctrl_ok / N_TRIALS
    print(f"\n  spiking resonator decodes the attributed patient (adj AND noun): {res:.2f}", flush=True)
    print(f"  single-shot spiking flat decode recovers the patient noun:        {ctrl:.2f} (chance {1/M:.3f})",
          flush=True)
    if res >= 0.80 and ctrl < 0.50:
        verdict = ("RESOLVES -- a SEMANTIC nested fact decodes on the genuine resonate-and-fire substrate "
                   "(crosstalk-robust) where the flat decode fails. Nested-fact understanding is biology-faithful.")
    elif res < 0.80:
        verdict = f"BOUNDARY -- spiking resonator {res:.2f} < 0.80; spiking + crosstalk degrades the nested decode."
    else:
        verdict = "CANNOT-CONCLUDE -- single-shot also succeeds."
    print(f"\nVERDICT: {verdict}", flush=True)
    return verdict


if __name__ == "__main__":
    main()
