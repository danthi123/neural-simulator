"""Phase 3 (cheat A) tractable-increment de-risk: the RF composer works on SENSORY-GROUNDED concept codes (real V1
Gabor responses, sim/visual_cortex.py) instead of rng.uniform random phases. Each concept = a distinct visual
stimulus -> the REAL V1 Gabor bank -> an 8192-d grounded code -> a complex random projection to D -> phases
(phi = angle(P @ v1code)/2pi). GATE: the composer's who/what queries + the no-confab moat work on the grounded codes
at >= the random-code baseline, multi-seed. This proves the GROUNDING INTERFACE works on the RF phasor substrate
(the codes come from real sensory features, not random). HONEST boundary: the word->stimulus mapping is arbitrary
(no real object-image dataset), and abstract concepts (motor/verbs) have no canonical image -- the embodied-cognition
limit; this validates the interface, not full semantic grounding.
"""
import itertools
import math

import numpy as np

from research.runners._visual_grounding_probe import render_bar, render_spot, _v1_matrix, _v1_code
from research.runners.rf_phasor_composer import RFPhasorComposer

FACTS = [("dog", "go", "north"), ("cat", "run", "south"), ("river", "look", "apple")]
QUERIES = [("go", "north", "dog"), ("run", "south", "cat"), ("look", "apple", "river")]


def _distinct_stimuli(n):
    """n distinct visual stimuli: oriented bars + spots on a position grid (well-separated V1 codes)."""
    stim = [render_bar(i * math.pi / 6) for i in range(6)]                 # 6 bars
    for cx, cy in itertools.product((8, 16, 24), (8, 16, 24)):             # 9 spots (3x3 grid)
        stim.append(render_spot(cx, cy))
    stim += [render_spot(12, 12, r=5), render_spot(20, 20, r=5),           # a few more, varied size/pos
             render_spot(12, 20, r=2), render_spot(20, 12, r=4)]
    return stim[:n]


def v1_grounded_phases(words, D, seed):
    W, _ = _v1_matrix()
    rng = np.random.default_rng(seed + 7)
    stimuli = _distinct_stimuli(len(words))
    codes = np.stack([np.asarray(_v1_code(W, img), dtype=float) for img in stimuli])     # (V, 8192) grounded codes
    nfeat = codes.shape[1]
    P = (rng.standard_normal((D, nfeat)) + 1j * rng.standard_normal((D, nfeat))) / math.sqrt(nfeat)
    phases = {}
    for w, code in zip(words, codes):
        proj = P @ code
        phases[w] = (np.angle(proj) / (2.0 * np.pi)) % 1.0
    # grounded-code separability summary (cosine of the phases as phasors)
    Z = np.stack([np.exp(2j * np.pi * phases[w]) for w in words])
    G = np.abs(Z @ Z.conj().T) / D
    off = G[~np.eye(len(words), dtype=bool)]
    return phases, float(off.mean()), float(off.max())


def run(seed, D):
    cn = RFPhasorComposer(seed=seed, D=D, period=200)                       # random-code baseline
    cg = RFPhasorComposer(seed=seed, D=D, period=200)                       # V1-grounded codes
    gph, cos_mean, cos_max = v1_grounded_phases(cg.words, D, seed)
    for w in cg.words:
        cg.concepts[w] = gph[w]                                             # inject the grounded codes
    for c in (cn, cg):
        for a, v, p in FACTS:
            c.store(a, v, p)
    g_ok = r_ok = n = 0
    for v, p, a in QUERIES:
        g_ok += int(cg.query_agent(v, p) == a) + int(cg.query_patient(a, v) == p)
        r_ok += int(cn.query_agent(v, p) == a) + int(cn.query_patient(a, v) == p)
        n += 2
    g_ab = int(cg.query_agent("go", "river") is None)
    return g_ok, r_ok, n, g_ab, cos_mean, cos_max


if __name__ == "__main__":
    for D in (256,):
        for seed in (42, 43, 44):
            g, r, n, ab, cm, cx = run(seed, D)
            print(f"D={D} seed={seed}: grounded {g}/{n}  random-baseline {r}/{n}  abstain {ab}/1  "
                  f"grounded-code-cos mean={cm:.3f} max={cx:.3f}", flush=True)
