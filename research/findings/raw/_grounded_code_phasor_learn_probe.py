"""Closes the last scientific soft spot behind phasor substrate unification: do the project's ACTUAL grounded
word codes (sparse, rate-coded, with real overlap) -- not random phasors -- still learn + compose on the
phasor substrate? Companion to _spiking_stdp_phasor_learn_probe.py (which used random-phasor cues).

The grounded codes are SPARSE RATE codes (which input neurons are active), via sim.text_embeddings.
vocab_to_drive_pattern -- fundamentally different from the PHASE codes the phasor substrate uses. The faithful
bridge: each input neuron has a fixed preferred phase; a word's cue is the population vector of its active
neurons' phases (rate-coded input -> phasor); plasticity is on co-active synapses only (the active inputs).
This is exactly the rate->phase conversion a production migration onto phasor FHRR must do, tested with the
REAL word encoder and its REAL overlap structure (~10%, the documented vocab_to_drive_pattern overlap).

PRE-REGISTERED, FROZEN: NIN=256 input neurons; D=512 concept code; N in {8, 32}; 5 seeds; sparsity=0.1;
asymmetric STDP tau=0.6, A+=1.0, A-=0.5; 60 compose tests/seed; the first N of a fixed 32-word vocabulary.
  retrieval := grounded word cue -> output phases -> phasor-cosine cleanup vs the N-code book -> correct concept.
  learned bind/unbind := bind two RETRIEVED codes to two roles, bundle, unbind, clean up -> both correct.
  THREE-STATE (at N=32):
    RESOLVES := retrieval >= 0.90 AND bind/unbind >= 0.80 -> grounded word codes learn + compose on the phasor
                substrate; the rate->phase bridge works with real overlap. No remaining scientific soft spot.
    BOUNDARY := retrieval >= 0.90 but bind/unbind < 0.80 -> learning works, composition needs more dimension.
    DOES-NOT-RESOLVE := retrieval < 0.90 -> grounded overlap breaks the map; the random-phasor results do not
                transfer to real codes (a critical honest negative -- re-examine before any migration).

  python -m research.findings.raw._grounded_code_phasor_learn_probe
"""
import numpy as np

from sim.text_embeddings import vocab_to_drive_pattern

NIN = 256
D = 512
SEEDS = (0, 1, 2, 3, 4)
TAU, A_PLUS, A_MINUS = 0.6, 1.0, 0.5
N_COMPOSE = 60
VOCAB = ["apple", "river", "dog", "cat", "big", "small", "hot", "cold", "go", "come", "stop", "look",
         "north", "east", "south", "west", "tree", "bird", "sun", "moon", "walk", "run", "eat", "sleep",
         "red", "blue", "fast", "slow", "child", "ball", "rain", "snow"]


def grounded_cue(token, phi, sparsity=0.1):
    """A word's cue phasor: its active grounded neurons (vocab_to_drive_pattern) firing at their preferred phase."""
    active = vocab_to_drive_pattern(token, n_neurons=NIN, sparsity=sparsity) > 0
    c = np.zeros(NIN, complex)
    c[active] = np.exp(1j * phi[active])
    return c, np.where(active)[0]


def pcos(p, c):
    return np.cos(p - c).mean()


def cleanup(phase, codes):
    return int(np.argmax([pcos(phase, codes[k]) for k in range(len(codes))]))


def run_one(N, seed):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(-np.pi, np.pi, size=NIN)            # fixed preferred phase per input neuron
    cues = [grounded_cue(VOCAB[t], phi) for t in range(N)]
    codes = rng.uniform(-np.pi, np.pi, size=(N, D))       # phasor concept codes
    rS = rng.uniform(-np.pi, np.pi, size=D)
    rO = rng.uniform(-np.pi, np.pi, size=D)

    cmat = np.stack([c for c, _ in cues], 1)             # grounded cue overlap (the real structure)
    G = np.abs(cmat.conj().T @ cmat)
    nrm = np.sqrt(np.diag(G).real)
    G = G / np.outer(nrm, nrm)
    np.fill_diagonal(G, 0.0)

    W = np.zeros((D, NIN))                                # STDP plastic only on co-active synapses
    for t in range(N):
        idx = cues[t][1]
        dt = codes[t][:, None] - phi[idx][None, :]
        dt = (dt + np.pi) % (2 * np.pi) - np.pi
        W[:, idx] += np.where(dt > 0, A_PLUS * np.exp(-dt / TAU), -A_MINUS * np.exp(dt / TAU))

    ret = [np.angle(W @ cues[t][0]) for t in range(N)]
    racc = np.mean([cleanup(ret[t], codes) == t for t in range(N)])
    biok = 0
    rng2 = np.random.default_rng(seed + 100)
    for _ in range(N_COMPOSE):
        a, b = int(rng2.integers(N)), int(rng2.integers(N))
        bundle = np.exp(1j * (rS + ret[a])) + np.exp(1j * (rO + ret[b]))
        da = cleanup(np.angle(bundle * np.exp(-1j * rS)), codes)
        db = cleanup(np.angle(bundle * np.exp(-1j * rO)), codes)
        biok += int(da == a and db == b)
    return racc, biok / N_COMPOSE, G.mean(), G.max()


def main():
    print(f"=== grounded word codes learn + compose on the phasor substrate? (NIN={NIN}, D={D}) ===", flush=True)
    print("    cue = vocab_to_drive_pattern (real sparse word code) -> active neurons fire at preferred phase.", flush=True)
    summary = {}
    for N in (8, 32):
        res = np.array([run_one(N, s) for s in SEEDS])
        r, b, gm, gx = res.mean(0)
        summary[N] = (r, b)
        print(f"\n  N={N}: retrieval {r:.2f}  learned-bind/unbind {b:.2f}   "
              f"[real grounded cue overlap: mean {gm:.2f}, max {res[:,3].max():.2f}]", flush=True)
    r32, b32 = summary[32]
    if r32 >= 0.90 and b32 >= 0.80:
        verdict = ("RESOLVES -- the project's ACTUAL grounded word codes learn + compose on the phasor substrate "
                   "(real overlap ~10%). The rate->phase bridge works; no remaining scientific soft spot for unification.")
    elif r32 >= 0.90:
        verdict = f"BOUNDARY -- retrieval {r32:.2f} learns but composition {b32:.2f} < 0.80; raise D and re-test."
    else:
        verdict = (f"DOES-NOT-RESOLVE -- retrieval {r32:.2f} < 0.90; grounded overlap breaks the map -- the "
                   f"random-phasor results do NOT transfer to real codes (critical honest negative).")
    print(f"\nVERDICT: {verdict}", flush=True)
    return verdict


if __name__ == "__main__":
    main()
