"""#4 follow-up (bounded): do Gabor-V1-grounded codes COMPOSE, not just separate?

The separability/cleanup probe (`_visual_grounding_probe`) showed real V1 Gabor features give well-separated,
cleanup-able concept codes. This closes the agent-integration question for the VISUAL subset: convert each
sensory code into a phasor (FHRR) code -- a deterministic function of the V1 features, so the phasor code is
GROUNDED -- then run the composition substrate's bind / bundle / unbind / cleanup on a 2-role fact built from two
visual-grounded concepts. If unbinding a role recovers the correct grounded concept (and still does from a
CORRUPTED sensory input), then sensory-grounded codes work end-to-end as composition codes, not just as
classifiable feature vectors.

  SIM_BACKEND=numpy python -m research.runners._visual_grounded_composition_probe
"""
import numpy as np

from research.runners._visual_grounding_probe import _v1_matrix, _stimuli, _v1_code, _corrupt

D = 2048


def _projection(n_v1, seed=42):
    """Fixed random complex projection V1(8192) -> D phases. Deterministic -> the phasor code is a fixed function
    of the SENSORY features (grounded), not a free random code."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((D, n_v1)) + 1j * rng.standard_normal((D, n_v1))).astype(np.complex128)


def _to_phasor(v1_code, proj):
    z = proj @ v1_code
    return np.exp(1j * np.angle(z))               # unit-magnitude phasor code (FHRR)


def _role(seed):
    rng = np.random.default_rng(seed)
    return np.exp(1j * rng.uniform(-np.pi, np.pi, D))


def _cleanup(query, codebook_names, codebook):
    sims = [float(np.abs(np.vdot(codebook[nm], query)) / D) for nm in codebook_names]
    i = int(np.argmax(sims)); s = np.sort(sims)[::-1]
    return codebook_names[i], float(s[0] - s[1])


def main():
    print("=== #4 follow-up: do Gabor-V1-grounded codes COMPOSE (bind/unbind/cleanup)? ===\n", flush=True)
    W, n_v1 = _v1_matrix()
    proj = _projection(n_v1)
    stim = _stimuli()
    names = list(stim)
    codebook = {nm: _to_phasor(_v1_code(W, fn()), proj) for nm, fn in stim.items()}   # GROUNDED phasor codes

    R_AGENT, R_PATIENT = _role(1), _role(2)

    # Build facts pairing concept i (agent) with concept i+1 (patient); unbind each role -> recover the grounded
    # concept; also test recovery when the sensory input is CORRUPTED (noise + translation) at query time.
    rng = np.random.default_rng(7)
    clean_ok = 0
    corrupt_ok = 0
    trials = 0
    for i in range(len(names)):
        a, b = names[i], names[(i + 1) % len(names)]
        fact = R_AGENT * codebook[a] + R_PATIENT * codebook[b]            # bind + bundle (complex sum)
        # clean unbind
        rec_a, _ = _cleanup(fact * np.conj(R_AGENT), names, codebook)
        rec_b, _ = _cleanup(fact * np.conj(R_PATIENT), names, codebook)
        clean_ok += int(rec_a == a) + int(rec_b == b)
        # corrupted-sensory unbind: rebuild the fact's agent slot from a NOISY render of concept a
        a_noisy = _to_phasor(_v1_code(W, _corrupt(stim[a], rng)), proj)
        fact_c = R_AGENT * a_noisy + R_PATIENT * codebook[b]
        rec_ac, _ = _cleanup(fact_c * np.conj(R_AGENT), names, codebook)
        corrupt_ok += int(rec_ac == a)
        trials += 1

    total_clean = 2 * trials
    print(f"  CLEAN compose (unbind agent + patient -> grounded concept): {clean_ok}/{total_clean} = "
          f"{100*clean_ok/total_clean:.0f}%", flush=True)
    print(f"  CORRUPTED-sensory compose (agent slot from noisy+shifted image): {corrupt_ok}/{trials} = "
          f"{100*corrupt_ok/trials:.0f}%", flush=True)
    ok = clean_ok >= 0.9 * total_clean and corrupt_ok >= 0.8 * trials
    print(f"\n  => {'GROUNDED CODES COMPOSE (sensory features -> bind/unbind/cleanup recovers concepts)' if ok else 'NEEDS WORK'}",
          flush=True)


if __name__ == "__main__":
    main()
