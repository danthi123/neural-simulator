"""CYCLE 97 biologization sweep, piece 4 (the BINDING) — can the on-substrate +/-1 COINCIDENCE bind (the
CoreSimComposer's spiking AND-on-ON/OFF-channels) replace the idealized numpy HRR algebra, on the
stream-LEARNED codes?

THE RESIDUAL. The conversation pipeline binds role+filler with numpy HRR (Fourier holographic) -- a clean
exactly-invertible ALGEBRA, the principled idealization. The project's on-substrate spiking binds are the FHRR
phasor bind (needs UNIT-MAGNITUDE phasor codes) and the +/-1 coincidence bind (needs BINARY codes). The stream
codes are GRADED-REAL (meaning in the magnitudes) -> the spiking binds don't directly fit. So the binding looked
like the hard residual.

THE IDEA (this de-risk). Use the +/-1 coincidence bind on the SIGN of the learned codes (b = sign(code)), but
CLEAN UP against the FULL GRADED codebook. Key: graded_code . sign(code) = sum|code| (the L1 norm) is maximal
for the TRUE concept, so a binarized unbind estimate still cleans up to the right GRADED concept. If who/what
recall + the no-confab moat survive, the binding BIOLOGIZES via the +/-1 coincidence spiking bind (lossy --
binarizing discards graded magnitude -- but the cleanup recovers the graded concept from the codebook).

+/-1 VSA (the CoreSimComposer's bind, in its algebraic form): bind(role, fill) = role * fill (elementwise +/-1,
realised by coincidence AND on ON/OFF channels); a FACT = sum of role*fill (bundle); unbind(F, role) = F * role
(role*role = 1) = fill + cross-noise; cleanup = nearest GRADED concept. Roles are random +/-1 (decorrelating).

GATE (multi fact-set seeds, cached 320 stream codes): who-Q&A recall >= 0.80 AND the no-confab moat (the learned
familiarity gate, piece 1) holds (0 false-accepts) on the +/-1-bound binarized codes. Compare to the HRR
graded baseline (recall 1.00, moat 1.00). NEGATIVE => binarization breaks it => the binding stays the residual.

Reuse-by-import (the learned familiarity gate from piece 1; the cached stream codes). CPU; no GPU.
Run:  SIM_BACKEND=numpy python -u -m research.runners._phaseB_biologize_binding_pm1_streamcodes_derisk
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._phaseB_biologize_moat_streamcodes_derisk import RealAntiHebbianFamiliarity  # noqa: E402

N_FACTS = 8
HOST_GATE_COS = 0.20      # a-priori conjunctive-cue confidence floor for the +/-1 unbind (lower than HRR's 0.25:
                          # the +/-1 unbind estimate is binary+noise so the true-match cosine is lower)
NOV_GATE = 0.5            # the familiarity-gate novelty threshold (piece 1)


def _cos(a, B):
    return (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + 1e-12)


def run_factset(codes, signs, seed):
    Nc, D = codes.shape
    rng = np.random.default_rng(seed * 19 + 5)
    # random +/-1 role codes (decorrelating)
    R_a = rng.choice([-1.0, 1.0], size=D)
    R_v = rng.choice([-1.0, 1.0], size=D)
    R_o = rng.choice([-1.0, 1.0], size=D)
    facts = []
    for _ in range(N_FACTS):
        i, j, k = rng.choice(Nc, 3, replace=False)
        facts.append((int(i), int(j), int(k)))
    # +/-1 bind + bundle: F = R_a*sign(a) + R_v*sign(v) + R_o*sign(o)
    bound = np.array([R_a * signs[i] + R_v * signs[j] + R_o * signs[k] for i, j, k in facts])

    def composite(v, o):                                   # the who-Q&A partial cue (verb+object), +/-1-bound
        return R_v * signs[v] + R_o * signs[o]

    def cue_match(verb, obj):
        scores = []
        for F in bound:
            mv = _cos(F * R_v, codes)[verb]                # unbind verb -> clean up vs GRADED codebook
            mo = _cos(F * R_o, codes)[obj]
            scores.append(min(mv, mo))
        scores = np.array(scores)
        return int(np.argmax(scores)), float(scores.max())

    # who-Q&A recall on present facts
    recall_ok, within = 0, 0
    for (i, j, k) in facts:
        bf, conf = cue_match(j, k)
        if conf >= HOST_GATE_COS:
            pred = int(np.argmax(_cos(bound[bf] * R_a, codes)))   # recover agent
            recall_ok += int(pred == i)
    recall = recall_ok / N_FACTS

    # the no-confab moat: the LEARNED familiarity gate (piece 1) on the +/-1-bound composites
    gate = RealAntiHebbianFamiliarity()
    for _, v, o in facts:
        gate.imprint(composite(v, o))
    stored = {(v, o) for _, v, o in facts}
    pres_nov = [gate.novelty(composite(v, o)) for _, v, o in facts]
    fa, n_abs, abs_nov, tries = 0, 0, [], 0
    while n_abs < N_FACTS and tries < 4000:
        tries += 1
        v, o = int(rng.integers(Nc)), int(rng.integers(Nc))
        if (v, o) in stored or v == o:
            continue
        n_abs += 1
        nov = gate.novelty(composite(v, o)); abs_nov.append(nov)
        fa += int(nov < NOV_GATE)                          # gate accepted an absent cue = a confabulation
    cp, ca = float(np.mean(pres_nov)), float(np.mean(abs_nov))
    print(f"  [pm1-bind seed {seed}] who-Q&A recall {recall:.2f} | moat: present-nov {cp:+.2f} vs absent-nov "
          f"{ca:+.2f}, false-accepts {fa}/{n_abs}", flush=True)
    return {"seed": seed, "recall": recall, "moat_fa": fa, "nov_present": cp, "nov_absent": ca}


def main():
    os.environ.setdefault("SIM_BACKEND", "numpy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    signs = np.sign(codes); signs[signs == 0] = 1.0        # +/-1 binarization for the coincidence bind
    print(f"[biologize binding (+/-1 coincidence) de-risk] stream codes {codes.shape} -- does the on-substrate "
          f"+/-1 coincidence bind on the BINARIZED codes (clean up vs the GRADED codebook) keep recall + the "
          f"moat? (vs HRR graded baseline recall 1.00 / moat 1.00)", flush=True)
    rows = [run_factset(codes, signs, s) for s in (42, 43, 44, 45, 46, 47)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    recall, fa = m("recall"), sum(r["moat_fa"] for r in rows)
    margin = m("nov_absent") - m("nov_present")
    print(f"\n{'='*94}\n  MEAN (6 fact-sets): who-Q&A recall {recall:.2f} | moat false-accepts {fa} | "
          f"moat novelty margin {margin:+.2f}", flush=True)
    print(f"{'='*94}", flush=True)
    if recall >= 0.80 and fa == 0:
        print(f"  GO: the on-substrate +/-1 COINCIDENCE bind REPLACES the idealized HRR algebra on the learned "
              f"codes -- binarize for the bind, clean up vs the GRADED codebook: recall {recall:.2f}, moat holds "
              f"({fa} false-accepts, margin {margin:+.2f}). ==> the BINDING biologizes too (the +/-1 coincidence "
              f"spiking bind; lossy -- binarizes the magnitudes -- but recall survives via the graded-codebook "
              f"cleanup). The residual narrows to: a graded (non-binarizing) spiking bind.", flush=True)
    elif recall >= 0.50:
        print(f"  PARTIAL: the +/-1 bind partly works (recall {recall:.2f}, false-accepts {fa}) -- binarization "
              f"costs some recall; the magnitudes carry load-bearing structure. Tune the gate / more dims.",
              flush=True)
    else:
        print(f"  NEGATIVE: binarizing for the +/-1 bind breaks recall ({recall:.2f}) -- the graded magnitudes are "
              f"load-bearing for binding; the binding STAYS the hard residual (a graded spiking bind is needed).",
              flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"recall": recall, "moat_false_accepts": fa, "moat_margin": margin, "per_factset": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_biologize_binding_pm1.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
