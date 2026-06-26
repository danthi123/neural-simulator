#!/usr/bin/env python
"""Diagnose WHY the OneBrainComposer query is ~2.5s on GPU (not the A5 ~96ms) -- is the resonate megakernel
(A5 lever 3, `enable_rf_cudagraph`) actually engaging, or falling back to the per-step loop?

The megakernel gate (bridge.py:5780) requires: enable_rf_cudagraph + is_gpu_backend + cp_rf_w_re set +
NOT rf_dense_weights. This probe monkeypatches (from the OUTSIDE, no sim/ edit) the two resonate paths
to COUNT which one a single query_patient takes:
  - _rf_resonate_steps_megakernel  (the fast path, one CUDA launch/step)
  - _rf_advance_one                (the slow per-step loop, ~15-20 launches/step x 208 steps)
loop calls >> 0  => the megakernel is being bypassed (find which gate condition fails).
mega calls > 0, loop 0 => the megakernel runs and ~2.5s is its real cost (a different problem).

Standalone -- does NOT touch first_chat_console.py. GPU (cupy) only.
"""
import os
os.environ["SIM_BACKEND"] = "cupy"
import sys
import time

import numpy as np

from research.runners.one_brain_composer import OneBrainComposer
from sim.backend import is_gpu_backend

# our own artifact (this session's curriculum runner); allow_pickle only for vocab dtype=object string array
NPZ = "bridges/firstchat/brain1454_w7000_seed42.npz"


def main():
    d = np.load(NPZ, allow_pickle=True)
    vocab = list(d["vocab"]); G = d["grounded"]; D = int(d["D"])
    n = 320
    vocab = list(vocab[:n]); grounded = {vocab[i]: G[i] for i in range(n)}
    t0 = time.time()
    comp = OneBrainComposer(seed=42, D=D, vocab=sorted(set(vocab)),
                            grounded_codes=grounded, enable_rf_cudagraph=True)
    b = comp.b
    print(f"[probe] build {time.time()-t0:.1f}s | n_total={comp.n_total}", flush=True)

    cfg = b.core_config
    print(f"[gate] enable_rf_cudagraph={getattr(cfg,'enable_rf_cudagraph',None)} | "
          f"is_gpu_backend={is_gpu_backend()} | "
          f"rf_dense_weights={getattr(cfg,'rf_dense_weights',None)}", flush=True)

    # count the two resonate paths (patch from the outside -- no sim/ edit)
    counts = {"mega": 0, "loop": 0}
    orig_mega = b._rf_resonate_steps_megakernel
    orig_loop = b._rf_advance_one

    def wrap_mega(n_steps):
        counts["mega"] += 1
        return orig_mega(n_steps)

    def wrap_loop():
        counts["loop"] += 1
        return orig_loop()

    b._rf_resonate_steps_megakernel = wrap_mega
    b._rf_advance_one = wrap_loop

    w = list(vocab[:12])
    comp.store(w[0], w[1], w[2])
    print(f"[gate] cp_rf_w_re is set after store={getattr(b,'cp_rf_w_re',None) is not None} | "
          f"cp_rf_prev_im is set={getattr(b,'cp_rf_prev_im',None) is not None}", flush=True)
    counts["mega"] = 0; counts["loop"] = 0  # reset; measure ONE query
    t = time.time(); comp.query_patient(w[0], w[1]); dt = (time.time()-t)*1000
    print(f"[path] one query_patient: {dt:.0f}ms | megakernel_calls={counts['mega']} | "
          f"per_step_loop_calls={counts['loop']}", flush=True)
    verdict = ("MEGAKERNEL BYPASSED (loop running) -> a gate condition fails"
               if counts["loop"] > 0 else
               "megakernel IS running -> ~2.5s is its real cost (look elsewhere)")
    print(f"[path] VERDICT: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
