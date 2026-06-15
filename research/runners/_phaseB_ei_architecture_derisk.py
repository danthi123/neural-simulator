"""Which E/I bridge architecture recovers axis-0? (determines the next build's wiring)

The projection-sign de-risk showed a SIGNED effective projection via E/I (W_exc-W_inh, both >=0) recovers the
axis-0-centered structure (+0.29) where the excitatory-only non-neg projection collapses (+0.04). The bridge's
input-mean hub naturally fires relu(x-m) (the POSITIVE half of the centered drive; the neuron can't fire
negative). So the question for the bridge wiring: does a signed E/I projection of relu(x-m) ALONE recover it
(simplest: one excitatory hub population + an inhibitory relay/copy -> a signed projection), or is the NEGATIVE
half needed too (an OFF hub firing relu(m-x), the full ON/OFF input)? And is the cortex readout single-pop or
ON/OFF (on the projected sign)?

Compares (real corpus, axis-0 centered Xw, host +0.44):
  (a) signed W @ Xw, ON/OFF read              -- full signed input + signed proj (the +0.31 reference)
  (b) Wei @ relu(Xw), single-pop read         -- relu hub (+half only), E/I proj, single cortex pop
  (c) Wei @ relu(Xw), ON/OFF read             -- relu hub, E/I proj, ON/OFF cortex on the projected sign
  (d) [Wei @ relu(Xw)  ;  Wei @ relu(-Xw)] ON/OFF -- ON/OFF hubs (both halves) + E/I proj + ON/OFF read
  (e) exc-only @ relu(Xw) ON/OFF             -- the bridge's CURRENT arch (no E/I) = expect collapse
GATE: the simplest variant that recovers ~+0.29 is the architecture to build. (b)/(c) GO => one hub pop + E/I
projection (cheapest). Only (d) GO => need ON/OFF hubs + E/I (the full retinal x E/I cross).

Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_ei_architecture_derisk
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.dendritic_d1_learn_graded_structure_derisk import (  # noqa: E402
    _cos_sim, _pearson_vs_Strue, heldout_generalization,
)
from research.runners.learned_graded_cortex_fair_test import build_real_corpus  # noqa: E402
from research.runners.option_c_paradigmatic_host_precheck import ppmi_svd_sim, score  # noqa: E402


def poisson_spk(rate, gain, rng):
    return rng.poisson(np.maximum(rate, 0.0) * gain).astype(np.float64)


def relu(x):
    return np.maximum(x, 0.0)


def single_code(drive, gain, rng):
    return np.array([poisson_spk(np.maximum(drive[i], 0.0), gain, rng) for i in range(len(drive))])


def onoff_code(drive, gain, rng):
    on = single_code(drive, gain, rng); off = single_code(-drive, gain, rng)
    return np.concatenate([on, off], axis=1)


def run_seed(seed, n_hub=500, k=128, gain=500.0):
    C, labels, S_true = build_real_corpus(seed, n_hub)
    host_p, _, _, _ = score(ppmi_svd_sim(np.maximum(C, 0.0), svd_dim=min(50, min(C.shape) - 1),
                                         alpha=0.75), labels)
    rng = np.random.RandomState(seed)
    X = np.log1p(np.maximum(C, 0.0)); Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    Xw = Xn - Xn.mean(0, keepdims=True)            # axis-0 per-feature centering
    sc = 1.0 / np.sqrt(n_hub)
    Wsig = rng.randn(k, n_hub) * sc
    Wei = np.abs(rng.randn(k, n_hub)) * sc - np.abs(rng.randn(k, n_hub)) * sc   # E/I signed (both >=0)
    Wexc = np.abs(rng.randn(k, n_hub)) * sc                                     # excitatory-only (bridge current)

    def p(code):
        return _pearson_vs_Strue(_cos_sim(code), S_true), heldout_generalization(code, labels)[0]

    pa, ga = p(onoff_code((Wsig @ Xw.T).T, gain, rng))                                   # (a)
    pb, gb = p(single_code((Wei @ relu(Xw).T).T, gain, rng))                             # (b)
    pc, gc = p(onoff_code((Wei @ relu(Xw).T).T, gain, rng))                              # (c)
    don = (Wei @ relu(Xw).T).T; doff = (Wei @ relu(-Xw).T).T
    pd, gd = p(np.concatenate([single_code(don, gain, rng), single_code(doff, gain, rng)], axis=1))  # (d)
    pe, ge = p(onoff_code((Wexc @ relu(Xw).T).T, gain, rng))                             # (e)
    print(f"\n[E/I arch seed {seed}] {C.shape[0]}c x {n_hub}h; host={host_p:+.3f} (axis-0)\n"
          f"  (a) signed W @ Xw, ON/OFF              : {pa:+.3f} (gen {ga:.3f})  [reference]\n"
          f"  (b) Wei @ relu(Xw), single-pop         : {pb:+.3f} (gen {gb:.3f})\n"
          f"  (c) Wei @ relu(Xw), ON/OFF             : {pc:+.3f} (gen {gc:.3f})\n"
          f"  (d) Wei @ [relu(Xw);relu(-Xw)], ON/OFF : {pd:+.3f} (gen {gd:.3f})\n"
          f"  (e) Wexc @ relu(Xw), ON/OFF (no E/I)   : {pe:+.3f} (gen {ge:.3f})  [bridge current = collapse]",
          flush=True)
    return {"seed": seed, "host": host_p, "a": pa, "b": pb, "c": pc, "d": pd, "e": pe}


def main():
    seeds = [42, 43, 44]
    rows = [run_seed(s) for s in seeds]
    def m(k): return float(np.mean([r[k] for r in rows]))
    print(f"\n  MEAN ({len(seeds)}): host {m('host'):+.3f} | (a)signed {m('a'):+.3f} | (b)relu+EI-single "
          f"{m('b'):+.3f} | (c)relu+EI-onoff {m('c'):+.3f} | (d)ONOFF+EI {m('d'):+.3f} | (e)exc-only {m('e'):+.3f}",
          flush=True)
    bar = 0.27
    if m("b") >= bar or m("c") >= bar:
        best = "(b) single-pop" if m("b") >= m("c") else "(c) ON/OFF read"
        print(f"  GO simplest: a relu(x-m) hub + E/I projection recovers axis-0 ({best}, "
              f"{max(m('b'),m('c')):+.3f}) -- the build = ONE input-mean hub population + an INHIBITORY "
              f"hub->cortex pathway alongside the excitatory one. No OFF input hub needed.", flush=True)
    elif m("d") >= bar:
        print(f"  GO full: need ON/OFF input hubs + E/I projection ({m('d'):+.3f}); the relu-only variants "
              f"({m('b'):+.3f}/{m('c'):+.3f}) lose the negative half -- build hub_on+hub_off, each with an E/I "
              f"(exc+inh) projection to cortex.", flush=True)
    else:
        print(f"  NEGATIVE: no E/I variant recovers axis-0 (best {max(m('b'),m('c'),m('d')):+.3f}); the bridge "
              f"realization of the signed projection needs more than relu hubs + E/I -- inspect.", flush=True)


if __name__ == "__main__":
    main()
