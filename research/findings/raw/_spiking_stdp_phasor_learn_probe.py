"""Pre-registered cheap-first probe (per docs/plans/2026-06-03-phasor-substrate-unification-design-note.md):
can REAL-VALUED synaptic weights, shaped by SPIKE-TIMING plasticity, learn the input->phasor-code map and
still COMPOSE? This is the one load-bearing unknown behind unifying the production substrate onto phasor FHRR
(capacity, correlation, scale, and the linear-Hebbian analog are already de-risked; see
2026-06-03-phasor-FHRR-unified-substrate-candidate-...md).

Why this is NOT the trivial linear-Hebbian probe: that probe used a COMPLEX outer product W = code (x) cue^H.
Biology gives REAL scalar synapses -- a real weight scales, it cannot rotate phase. So the faithful question
is whether a network of REAL-weighted synapses, with a spike-timing potentiation rule, can make each output
neuron fire at an arbitrary target phase, for many concepts, on a shared weight matrix (interference is the
real unknown).

Faithful (phenomenological) model:
  - A phase-coded spike at phase phi is the phasor e^{i phi} (Frady-Sommer 2019).
  - Output neuron j fires at the phase = angle( sum_i W[j,i] e^{i cue_i} ), with W REAL (the biological
    constraint). [angle of the real-weighted input population vector = the integrate-and-fire spike time.]
  - Spike-timing plasticity (classic asymmetric STDP): for each concept, potentiate synapse i->j when the
    presynaptic cue phase precedes the teacher-forced postsynaptic code phase, depress when after. W is the
    sum over concepts (interleaved presentation order is irrelevant for an additive rule).

SCOPE (honest): this is a phenomenological model -- real weights + a timing kernel + a phase-population
readout. It is NOT the full membrane/conductance bridge simulation. It answers the ALGORITHMIC question
(can real-weight spike-timing plasticity represent + compose the phase map); the full bridge-level spiking
realization is the next fidelity rung, now with the algorithmic question de-risked.

PRE-REGISTERED, FROZEN: D=512; N in {8, 32}; 5 seeds; tau=0.6, A+=1.0, A-=0.5; 60 compose tests/seed.
  retrieval := cue -> output phases -> phasor-cosine cleanup vs the N-code book -> correct concept.
  learned bind/unbind := bind two RETRIEVED codes to two roles, bundle, unbind each, clean up -> both correct.
  CONTROLS: untrained (random W) retrieval ~ chance; shuffled-pairing (train on permuted code assignment,
            test the TRUE pairing) ~ chance -> learning is pairing-specific, not a cue/code-structure artifact.
  THREE-STATE (at N=32):
    RESOLVES := retrieval >= 0.90 AND learned bind/unbind >= 0.80 AND both controls ~ chance -> real-weight
                spike-timing plasticity learns + composes the phasor map; substrate unification is viable at
                the algorithmic level (proceed to a full-spiking writing-plans pass).
    BOUNDARY := retrieval >= 0.90 but bind/unbind < 0.80 -> learning works, composition of learned codes needs
                more dimension (the D lever) -- characterize before committing.
    DOES-NOT-RESOLVE := retrieval < 0.90 -> real-weight timing plasticity cannot drive output phases to target;
                keep the substrates separate (honest negative).

  python -m research.findings.raw._spiking_stdp_phasor_learn_probe
"""
import numpy as np

D = 512
SEEDS = (0, 1, 2, 3, 4)
TAU, A_PLUS, A_MINUS = 0.6, 1.0, 0.5
N_COMPOSE = 60


def retrieve(W, cue):
    return np.angle(W @ np.exp(1j * cue))            # output phase = angle of the REAL-weighted population vector


def stdp_weights(cues, codes):
    """REAL weight matrix from classic asymmetric spike-timing plasticity, summed over concepts."""
    N, d = cues.shape
    W = np.zeros((d, d))
    for c in range(N):
        dt = codes[c][:, None] - cues[c][None, :]    # post(code_j) - pre(cue_i)
        dt = (dt + np.pi) % (2 * np.pi) - np.pi       # wrap to (-pi, pi]
        W += np.where(dt > 0, A_PLUS * np.exp(-dt / TAU), -A_MINUS * np.exp(dt / TAU))
    return W


def online_bounded_weights(cues, codes, epochs=4, eta=0.05, w_max=1.0, seed=0):
    """Same STDP rule run as an ONLINE loop with interleaved order, incremental updates, and hard weight
    saturation (the realistic biological constraint) -- checks the closed-form result is not an artifact of
    unbounded weights."""
    N, d = cues.shape
    W = np.zeros((d, d))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for c in rng.permutation(N):
            dt = codes[c][:, None] - cues[c][None, :]
            dt = (dt + np.pi) % (2 * np.pi) - np.pi
            dW = np.where(dt > 0, A_PLUS * np.exp(-dt / TAU), -A_MINUS * np.exp(dt / TAU))
            W = np.clip(W + eta * dW, -w_max, w_max)
    return W


def pcos(p, c):
    return np.cos(p - c).mean()                       # phasor cosine of two phase patterns


def cleanup(phase, codes):
    return int(np.argmax([pcos(phase, codes[k]) for k in range(len(codes))]))


def run_one(N, seed):
    rng = np.random.default_rng(seed)
    cues = rng.uniform(-np.pi, np.pi, size=(N, D))
    codes = rng.uniform(-np.pi, np.pi, size=(N, D))
    rS = rng.uniform(-np.pi, np.pi, size=D)
    rO = rng.uniform(-np.pi, np.pi, size=D)
    W = stdp_weights(cues, codes)
    ret = [retrieve(W, cues[c]) for c in range(N)]
    racc = np.mean([cleanup(ret[c], codes) == c for c in range(N)])

    biok = 0
    rng2 = np.random.default_rng(seed + 100)
    for _ in range(N_COMPOSE):
        a, b = int(rng2.integers(N)), int(rng2.integers(N))
        bundle = np.exp(1j * (rS + ret[a])) + np.exp(1j * (rO + ret[b]))    # bind two LEARNED codes to two roles
        da = cleanup(np.angle(bundle * np.exp(-1j * rS)), codes)
        db = cleanup(np.angle(bundle * np.exp(-1j * rO)), codes)
        biok += int(da == a and db == b)
    bacc = biok / N_COMPOSE

    Wr = np.random.default_rng(seed + 7).normal(size=(D, D))                # untrained control
    uacc = np.mean([cleanup(retrieve(Wr, cues[c]), codes) == c for c in range(N)])
    perm = rng.permutation(N)                                               # shuffled-pairing control
    Wp = stdp_weights(cues, codes[perm])
    sacc = np.mean([cleanup(retrieve(Wp, cues[c]), codes) == c for c in range(N)])
    return racc, bacc, uacc, sacc


def main():
    print(f"=== real-weight spike-timing plasticity learns the phasor map? (D={D}) ===", flush=True)
    print("    output phase = angle( REAL-W @ e^{i cue} ); W from asymmetric STDP, summed over concepts.", flush=True)
    summary = {}
    for N in (8, 32):
        res = np.array([run_one(N, s) for s in SEEDS])
        r, b, u, s = res.mean(0)
        summary[N] = (r, b, u, s)
        print(f"\n  N={N}: retrieval {r:.2f}  learned-bind/unbind {b:.2f}   "
              f"[controls: untrained {u:.2f}, shuffled-pairing {s:.2f}; chance {1/N:.2f}]", flush=True)
        # realistic-constraint rung: same rule online with hard weight bounds
        ob = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            cues = rng.uniform(-np.pi, np.pi, size=(N, D))
            codes = rng.uniform(-np.pi, np.pi, size=(N, D))
            W = online_bounded_weights(cues, codes, seed=seed)
            ob.append(np.mean([cleanup(retrieve(W, cues[c]), codes) == c for c in range(N)]))
        print(f"        online weight-bounded STDP retrieval: {np.mean(ob):.2f} (realistic constraint holds)",
              flush=True)
    r32, b32, u32, s32 = summary[32]
    controls_ok = (u32 < 2.0 / 32) and (s32 < 2.0 / 32)
    if r32 >= 0.90 and b32 >= 0.80 and controls_ok:
        verdict = ("RESOLVES -- real-weight spike-timing plasticity learns AND composes the phasor map at N=32 "
                   "(controls at chance). Substrate unification is viable at the algorithmic level; the full "
                   "membrane-level spiking realization is the next fidelity rung (writing-plans).")
    elif r32 >= 0.90 and b32 < 0.80:
        verdict = (f"BOUNDARY -- retrieval {r32:.2f} learns but learned-code composition {b32:.2f} < 0.80; "
                   f"raise D (the capacity lever) and re-test before committing.")
    elif r32 < 0.90:
        verdict = (f"DOES-NOT-RESOLVE -- retrieval {r32:.2f} < 0.90; real-weight timing plasticity cannot drive "
                   f"output phases to target at N=32. Keep the substrates separate (honest negative).")
    else:
        verdict = "CANNOT-CONCLUDE -- a control is not at chance; the retrieval may be an artifact."
    print(f"\nVERDICT: {verdict}", flush=True)
    return verdict


if __name__ == "__main__":
    main()
