"""Biology-faithful capstone for recursive clause nesting: does an embedded clause -- "dog see
(cat chase bird)" -- decode on the GENUINE resonate-and-fire FHRR substrate (not just numpy phasor
algebra)? Companion to _spiking_nested_fact_probe (which validated an attributed patient adj⊗noun in
spikes) and _recursive_clause_probe (which validated the recursive clause in numpy phasor). This unifies
them: a clause built and decoded ENTIRELY with the resonate-and-fire ops (rf_bind / rf_unbind / rf_bundle),
crosstalk-robust, through TWO bundle levels.

  inner = rf_bundle[ AGENT(x)cat, ACTION(x)chase, PATIENT(x)bird ]            ("cat chase bird") -- in spikes
  fact  = rf_bundle[ AGENT(x)dog, ACTION(x)see,   PATIENT(x)inner ]           ("dog see (cat chase bird)")
  decode: rf_unbind outer AGENT/ACTION -> dog, see; rf_unbind outer PATIENT -> inner -> rf_unbind its
          AGENT/ACTION/PATIENT -> cat, chase, bird. All five recovered IN SPIKES.

A clause is a BUNDLE of KNOWN role-products (the roles are fixed vectors), so it needs no resonator -- only
recursive unbinding. The open question is whether the spiking realization + two levels of bundle crosstalk
preserve the signal.

Reuse-by-import of resonate_fire_fhrr (rf_*); no protected-module change. Small scale (rf_resonate
time-steps a full cycle per call) -- tractable, still decisive.

PRE-REGISTERED, FROZEN: D=256; M=8 (nouns) / 8 (verbs); n_trials=12.
  success := all FIVE fillers recovered (outer agent+action, inner agent+action+patient) IN SPIKES.
  CONTROL := single-shot spiking flat decode of the outer patient as a noun -- must FAIL (<0.50): the
             patient is a clause, not a noun.
  THREE-STATE:
    RESOLVES := full 5-filler spiking recovery >= 0.80 AND control < 0.50 -> recursive clause nesting is
                biology-faithful (resonate-and-fire), crosstalk-robust through two bundle levels.
    BOUNDARY := full spiking recovery < 0.80 (spiking + two-level crosstalk breaks the recursive decode).
    CANNOT-CONCLUDE := single-shot control also succeeds.

  python -m research.findings.raw._spiking_recursive_clause_probe
"""
import numpy as np

from research.runners.resonate_fire_fhrr import rf_bind, rf_unbind, rf_bundle, rf_resonate, _to_phasor, CYCLE_STEPS
from research.runners.spiking_phasor_fhrr import phases_to_spikes

D = 256
M = 8
N_TRIALS = 12


def rand_code(rng):
    return phases_to_spikes(rng.uniform(0.0, 1.0, size=D), CYCLE_STEPS)


def cbmat(cb):
    return np.stack([_to_phasor(c, CYCLE_STEPS) for c in cb], axis=1)


def cleanup_idx(spikes, MAT):
    return int(np.argmax(np.abs(MAT.conj().T @ _to_phasor(spikes, CYCLE_STEPS))))


def main():
    print(f"=== SPIKING recursive clause (resonate-and-fire FHRR; D={D} M={M}) ===", flush=True)
    print("    fact = AGENT(x)dog + ACTION(x)see + PATIENT(x)(AGENT(x)cat + ACTION(x)chase + PATIENT(x)bird)",
          flush=True)
    print("    -- built and decoded entirely in spikes.", flush=True)
    res_ok = 0
    ctrl_ok = 0
    outer_ok = 0
    inner_ok = 0
    for t in range(N_TRIALS):
        rng = np.random.default_rng(1100 + t)
        R = {r: rand_code(rng) for r in ("AGENT", "ACTION", "PATIENT")}
        nouns = [rand_code(rng) for _ in range(M)]
        verbs = [rand_code(rng) for _ in range(M)]
        NM, VM = cbmat(nouns), cbmat(verbs)
        oa, ov = int(rng.integers(M)), int(rng.integers(M))
        ia, iv, ip = (int(rng.integers(M)), int(rng.integers(M)), int(rng.integers(M)))

        inner = rf_bundle([rf_bind(R["AGENT"], nouns[ia], CYCLE_STEPS),
                           rf_bind(R["ACTION"], verbs[iv], CYCLE_STEPS),
                           rf_bind(R["PATIENT"], nouns[ip], CYCLE_STEPS)], CYCLE_STEPS)
        fact = rf_bundle([rf_bind(R["AGENT"], nouns[oa], CYCLE_STEPS),
                          rf_bind(R["ACTION"], verbs[ov], CYCLE_STEPS),
                          rf_bind(R["PATIENT"], inner, CYCLE_STEPS)], CYCLE_STEPS)

        d_oa = cleanup_idx(rf_unbind(fact, R["AGENT"], CYCLE_STEPS), NM)
        d_ov = cleanup_idx(rf_unbind(fact, R["ACTION"], CYCLE_STEPS), VM)
        # the inner clause in spikes -- unbind the RAW output directly; a rf_resonate cleanup here corrupts the
        # phase structure the second unbind needs (diagnosed: with resonate 0.00, without it 1.00 at all D).
        inner_hat = rf_unbind(fact, R["PATIENT"], CYCLE_STEPS)
        d_ia = cleanup_idx(rf_unbind(inner_hat, R["AGENT"], CYCLE_STEPS), NM)
        d_iv = cleanup_idx(rf_unbind(inner_hat, R["ACTION"], CYCLE_STEPS), VM)
        d_ip = cleanup_idx(rf_unbind(inner_hat, R["PATIENT"], CYCLE_STEPS), NM)

        o_ok = (d_oa == oa and d_ov == ov)
        i_ok = (d_ia == ia and d_iv == iv and d_ip == ip)
        outer_ok += int(o_ok)
        inner_ok += int(i_ok)
        res_ok += int(o_ok and i_ok)
        ctrl_ok += int(cleanup_idx(rf_unbind(fact, R["PATIENT"], CYCLE_STEPS), NM) == ip)

    res = res_ok / N_TRIALS
    ctrl = ctrl_ok / N_TRIALS
    print(f"\n  outer level recovered (agent+action):            {outer_ok / N_TRIALS:.2f}", flush=True)
    print(f"  inner clause recovered (agent+action+patient):   {inner_ok / N_TRIALS:.2f}", flush=True)
    print(f"  FULL 5-filler spiking recovery:                  {res:.2f}", flush=True)
    print(f"  control (outer patient as a flat noun):          {ctrl:.2f} (chance {1/M:.3f}; must stay low)",
          flush=True)
    if res >= 0.80 and ctrl < 0.50:
        verdict = ("RESOLVES -- a recursively nested clause decodes on the GENUINE resonate-and-fire substrate "
                   "through two bundle levels; recursive clause nesting is biology-faithful, crosstalk-robust.")
    elif res < 0.80:
        verdict = f"BOUNDARY -- spiking full recovery {res:.2f} < 0.80; spiking + two-level crosstalk degrades the decode."
    else:
        verdict = "CANNOT-CONCLUDE -- single-shot control also succeeded."
    print(f"\nVERDICT: {verdict}", flush=True)
    return verdict


if __name__ == "__main__":
    main()
