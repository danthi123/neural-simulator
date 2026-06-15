"""CYCLE 97 biologization sweep, piece 2 (the CLEANUP) — replace the host argmax-to-nearest-concept with the
validated SPIKING NEF thresholded cleanup (Stewart-Tang-Eliasmith 2011, the Spaun cleanup), ON THE
STREAM-LEARNED codes, and verify it MATCHES the host argmax.

CONTEXT. The conversation pipeline cleans a noisy unbind estimate to the nearest concept via a host
`argmax(concepts . est)`. The brain-based replacement (already shipped in CoreSimComposer
enable_spiking_cleanup, validated to numpy parity at production D=2048) is the spiking NEF thresholded cleanup:
a matched-filter spiking network (on/off opponency input -> per-concept encoder neurons + a per-concept firing
threshold placed so off-target concepts emit ZERO spikes); argmax over the per-concept firing == the host
argmax. THE TRANSFER QUESTION: it was validated on the composer's own (decorrelated) codes; does it still match
the host argmax on the CORRELATED, graded-real stream-LEARNED codes? (The NEF cleanup is a matched filter over
a codebook -- code-agnostic in principle; this verifies it on the actual learned codes.)

GATE (multi fact-set seeds, cached 320 stream codes): the spiking NEF cleanup recovers the SAME concept as the
host argmax on the noisy HRR unbind estimates (agreement >= 0.90; per-concept off-target firing stays ~0).

Reuse-by-import (build_nef_bridge + cleanup from the shipped NEF probe; hrr ops; the cached stream codes).
GPU (the NEF bridge is small ~4500 neurons; light contention).
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_biologize_cleanup_streamcodes_derisk
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

from research.runners._phaseB_assembled_pipeline_ppmi_derisk import hrr_bind, hrr_unbind  # noqa: E402
from research.findings.raw._spiking_cleanup_nef import build_nef_bridge, cleanup as nef_cleanup  # noqa: E402

# the shipped NEF operating point (CoreSimComposer.NEF_CLEANUP_OP)
NEF_OP = dict(bias=-625.0, w_match=120.0, n_per=12, w_in_cfs=1.0, w_in_fs=10.0, n_in_fs=60, einh=-80.0,
              run_steps=400)
N_FACTS = 6
N_HUB = 300            # the stream code dim (= the conversation runner's n_hub)


def run_factset(codes, nef, seed):
    Nc, D = codes.shape
    bridge, idx, M, n_per = nef
    rng = np.random.default_rng(seed * 31 + 7)
    R_a = rng.standard_normal(D) / np.sqrt(D)
    R_v = rng.standard_normal(D) / np.sqrt(D)
    R_o = rng.standard_normal(D) / np.sqrt(D)
    facts = []
    for _ in range(N_FACTS):
        i, j, k = rng.choice(Nc, 3, replace=False)
        facts.append((int(i), int(j), int(k)))
    agree, n = 0, 0
    for (i, j, k) in facts:
        F = hrr_bind(R_a, codes[i]) + hrr_bind(R_v, codes[j]) + hrr_bind(R_o, codes[k])
        for role, truth in ((R_a, i), (R_v, j), (R_o, k)):       # unbind each role -> noisy est -> clean up
            est = hrr_unbind(F, role)
            host = int(np.argmax(codes @ est))                   # the host argmax cleanup
            per_concept = nef_cleanup(bridge, idx, M, n_per, est, NEF_OP["bias"], NEF_OP["run_steps"])
            spk = int(np.argmax(np.asarray(per_concept)))        # the spiking NEF cleanup
            agree += int(spk == host)
            n += 1
    print(f"  [cleanup seed {seed}] spiking-NEF == host-argmax: {agree}/{n} ({agree/n:.2f})", flush=True)
    return {"seed": seed, "agree": agree, "n": n}


def main():
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    codes_path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_stream_codes_320_seed42.npy")
    if not os.path.exists(codes_path):
        print(f"  [missing] {codes_path}", flush=True)
        return
    codes = np.load(codes_path).astype(np.float64)
    codes = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    print(f"[biologize cleanup de-risk] stream-learned codes {codes.shape} -- does the SPIKING NEF cleanup match "
          f"the host argmax on the correlated learned codes?", flush=True)
    # build ONE persistent NEF cleanup bridge from the stream codebook (Stewart-Tang-Eliasmith)
    nef = build_nef_bridge(42, codes, NEF_OP["n_per"], NEF_OP["w_match"], NEF_OP["w_in_cfs"], NEF_OP["w_in_fs"],
                           NEF_OP["n_in_fs"], NEF_OP["einh"])
    print(f"  built NEF cleanup bridge: {nef[2]} concepts x {nef[3]} neurons/concept", flush=True)
    rows = [run_factset(codes, nef, s) for s in (42, 43, 44)]
    tot_a = sum(r["agree"] for r in rows); tot_n = sum(r["n"] for r in rows)
    acc = tot_a / max(tot_n, 1)
    print(f"\n{'='*92}\n  MEAN (3 fact-sets): spiking-NEF == host-argmax {tot_a}/{tot_n} = {acc:.3f}", flush=True)
    print(f"{'='*92}", flush=True)
    if acc >= 0.90:
        print(f"  GO: the SPIKING NEF cleanup MATCHES the host argmax on the stream-learned codes ({acc:.3f}) -> "
              f"the cleanup is BIOLOGIZED (a thresholded spiking matched filter, not a host argmax) on the "
              f"correlated learned codes.", flush=True)
    elif acc >= 0.70:
        print(f"  PARTIAL: the spiking cleanup mostly matches ({acc:.3f}) -- tune the operating point (bias / "
              f"w_match / run_steps) for the learned-code regime.", flush=True)
    else:
        print(f"  NEGATIVE: the spiking cleanup diverges from the host argmax ({acc:.3f}) on the learned codes -- "
              f"the matched filter may need re-tuning for the graded/correlated regime; inspect.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"agreement": acc, "n": tot_n, "per_factset": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_biologize_cleanup_streamcodes.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
