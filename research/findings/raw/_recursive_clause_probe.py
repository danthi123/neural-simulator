"""Cheap-first, pre-registered: does a RECURSIVELY NESTED fact -- a clause as an argument,
"dog see (cat chase bird)" -- decode on phasor FHRR through TWO levels of bundle crosstalk?

This is the real compositional-depth test the deep-research synthesis flagged as THE wall
(nesting / multi-hop SNR; the hierarchical-320 shortcut scored 0.000 on structured facts). Distinct
from multi-modifier ("big red ball" = a PRODUCT of unknown factors needing the resonator): a nested
clause is a BUNDLE of KNOWN role-products (AGENT/ACTION/PATIENT are fixed role vectors), so it is
decoded by RECURSIVE UNBINDING -- no resonator needed. The only open question is whether the signal
survives the accumulated crosstalk of two bundle levels.

  inner = phasor( AGENT(x)cat  + ACTION(x)chase + PATIENT(x)bird )      ("cat chase bird")
  fact  = phasor( AGENT(x)dog  + ACTION(x)see   + PATIENT(x)inner )     ("dog see (cat chase bird)")
  decode: unbind outer AGENT/ACTION -> dog, see; unbind outer PATIENT -> inner-estimate ->
          unbind its AGENT/ACTION/PATIENT -> cat, chase, bird.

PRE-REGISTERED, FROZEN: D=1024; M=8 (nouns) / 8 (verbs); n_trials=20.
  success := all FIVE fillers recovered (outer agent+action, inner agent+action+patient).
  CONTROL := treat the outer patient slot as a flat noun (single-shot cleanup vs the noun vocab) --
             it must FAIL (<0.50), because the patient is a clause, not a noun (confirms genuine nesting).
  THREE-STATE:
    RESOLVES := full 5-tuple recovery >= 0.80 AND control < 0.50 -> recursive clause nesting works on
                phasor FHRR; the SNR survives two bundle levels.
    BOUNDARY := full recovery < 0.80 -> two-level bundle crosstalk breaks recursive nesting at this
                D (report per-level recovery to localise WHERE it breaks).
    CANNOT-CONCLUDE := control also succeeds.

  python -m research.findings.raw._recursive_clause_probe
"""
import numpy as np

D = 1024
M = 8
N_TRIALS = 20


def u(v):
    return v / (np.abs(v) + 1e-12)


def code(rng):
    return np.exp(1j * rng.uniform(-np.pi, np.pi, size=D))


def main():
    print(f"=== recursive clause nesting on phasor FHRR (D={D}, M={M}) ===", flush=True)
    print("    fact = AGENT(x)dog + ACTION(x)see + PATIENT(x)(AGENT(x)cat + ACTION(x)chase + PATIENT(x)bird)",
          flush=True)
    full_ok = 0
    outer_ok = 0
    inner_ok = 0
    ctrl_ok = 0
    for t in range(N_TRIALS):
        rng = np.random.default_rng(700 + t)
        ROLES = {r: code(rng) for r in ("AGENT", "ACTION", "PATIENT")}
        nouns = [code(rng) for _ in range(M)]
        verbs = [code(rng) for _ in range(M)]
        NM = np.stack(nouns, axis=1)
        VM = np.stack(verbs, axis=1)

        oa, ov = int(rng.integers(M)), int(rng.integers(M))            # outer agent (noun), action (verb)
        ia, iv, ip = (int(rng.integers(M)), int(rng.integers(M)), int(rng.integers(M)))  # inner a/v/patient

        inner = u(ROLES["AGENT"] * nouns[ia] + ROLES["ACTION"] * verbs[iv] + ROLES["PATIENT"] * nouns[ip])
        fact = u(ROLES["AGENT"] * nouns[oa] + ROLES["ACTION"] * verbs[ov] + ROLES["PATIENT"] * inner)

        def cu(vec, MAT):
            return int(np.argmax(np.abs(MAT.conj().T @ u(vec))))

        d_oa = cu(fact * np.conj(ROLES["AGENT"]), NM)
        d_ov = cu(fact * np.conj(ROLES["ACTION"]), VM)
        inner_hat = u(fact * np.conj(ROLES["PATIENT"]))                # outer patient -> inner-clause estimate
        d_ia = cu(inner_hat * np.conj(ROLES["AGENT"]), NM)
        d_iv = cu(inner_hat * np.conj(ROLES["ACTION"]), VM)
        d_ip = cu(inner_hat * np.conj(ROLES["PATIENT"]), NM)

        o_ok = (d_oa == oa and d_ov == ov)
        i_ok = (d_ia == ia and d_iv == iv and d_ip == ip)
        outer_ok += int(o_ok)
        inner_ok += int(i_ok)
        full_ok += int(o_ok and i_ok)
        # control: is the outer patient a flat noun? (it's a clause -> must fail)
        ctrl_noun = cu(fact * np.conj(ROLES["PATIENT"]), NM)
        ctrl_ok += int(ctrl_noun == ip)   # would only "pass" by accident; the slot is not a flat noun

    full = full_ok / N_TRIALS
    outer = outer_ok / N_TRIALS
    inner = inner_ok / N_TRIALS
    ctrl = ctrl_ok / N_TRIALS
    print(f"\n  outer level recovered (agent+action):            {outer:.2f}", flush=True)
    print(f"  inner clause recovered (agent+action+patient):   {inner:.2f}", flush=True)
    print(f"  FULL 5-filler recovery:                          {full:.2f}", flush=True)
    print(f"  control (outer patient as a flat noun):          {ctrl:.2f} (chance {1/M:.3f}; must stay low)",
          flush=True)
    if full >= 0.80 and ctrl < 0.50:
        verdict = ("RESOLVES -- a recursively nested clause ('dog see (cat chase bird)') decodes on phasor "
                   "FHRR through two bundle levels; the SNR survives. Clause-as-argument recursion works.")
    elif full < 0.80:
        verdict = (f"BOUNDARY -- full recovery {full:.2f} < 0.80; two-level bundle crosstalk breaks recursive "
                   f"nesting at D={D} (outer {outer:.2f}, inner {inner:.2f}).")
    else:
        verdict = "CANNOT-CONCLUDE -- control also succeeded."
    print(f"\nVERDICT: {verdict}", flush=True)
    return verdict


if __name__ == "__main__":
    main()
