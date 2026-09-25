"""_slotbinder_step_cost_profile.py -- where one SlotBinder simulation step's time goes, on the REAL built bridge
(2026-09-25; the profile behind research/findings/2026-09-25-slotbinder-event-driven-step-bit-identical-numpy.md).

Two measurements on a real seed-S N-fact binder (the production gate's own sampler, fanout 32):
  1. the bridge's own step profiler (GPUConfig.enable_step_profiler) over a short teach and one query: the share
     of step time in the STP/NMDA-split section, the synaptic-propagation section and the plasticity section;
  2. each all-synapse operation the dense step runs, timed on the bridge's own arrays with a 20-neuron slot
     firing (the teach/read regime): the Hebbian pre/post coincidence gather, the gain-weighted decay, the
     gain-masked clip, the eligibility-trace decay, the NMDA/AMPA data split, the two transpose matvecs.
Output: one JSON. CPU/numpy by default; the per-op numbers are what the event-driven step removes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from research.runners._slotbinder_l2_sparse_derisk import _load_live_bundle, _sample_facts  # noqa: E402
from research.runners.slotbinder_composer import SlotBinderComposer  # noqa: E402
from sim.backend import get_backend, get_sparse_module  # noqa: E402


def _t(fn, reps=5):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return 1000.0 * (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-facts", type=int, nargs="+", default=[32, 128])
    ap.add_argument("--teach-facts", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    xp, backend = get_backend()
    csp = get_sparse_module()
    _, facts_full, _ = _load_live_bundle()
    out = {"seed": args.seed, "sim_backend": backend, "per_n": {}}
    for N in args.n_facts:
        sample, _ = _sample_facts(facts_full, args.seed, N)
        words = sorted({f[r] for f in sample for r in ("agent", "action", "patient") if isinstance(f.get(r), str)})
        c = SlotBinderComposer(seed=args.seed, vocab=words, max_facts=N, fanout=32, prewire_facts=list(sample),
                               sparse_step=False)
        c._ensure()
        b = c._b
        C = b.cp_connections
        nnz = int(C.nnz)
        # (1) the bridge's own section profiler, over a short teach and one query
        b.gpu_config.enable_step_profiler = True
        _OFFSET = 10 ** 9                      # keep the profiler's every-500-steps log-and-reset from firing
        b._prof_accum, b._prof_count = {}, -_OFFSET
        for f in sample[:args.teach_facts]:
            c.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
        f = sample[args.teach_facts - 1]
        c.query_patient(f["agent"], f["action"])
        acc, cnt = dict(b._prof_accum), b._prof_count + _OFFSET
        tot = sum(acc.values())
        sections = {k: {"ms_per_step": 1000.0 * v / cnt, "share": v / tot} for k, v in sorted(acc.items())}
        b.gpu_config.enable_step_profiler = False
        # (2) each all-synapse op of the dense step, on this bridge's arrays, one 20-neuron slot firing
        n = b.core_config.num_neurons
        prev = xp.zeros(n, dtype=bool)
        prev[:20] = True
        fired = prev.copy()
        coo = b._get_cached_coo()
        gain = b.cp_plasticity_rate_gain[:nnz]
        data = C.data.copy()
        mask_f = b.cp_nmda_recurrent_synapse_mask[:nnz].astype(xp.float32)
        elig = xp.zeros(nnz, dtype=xp.float32)
        x2 = xp.stack([prev.astype(xp.float32), prev.astype(xp.float32)], axis=1)
        x1 = prev.astype(xp.float32)
        ops = {
            "hebbian_pre_post_gather_where": _t(lambda: xp.where(prev[coo.row] & fired[coo.col])[0]),
            "gated_decay": _t(lambda: data.__imul__(1.0 - 1e-5 * gain)),
            "gain_masked_clip": _t(lambda: data.__setitem__(gain > 0, xp.clip(data, 0.05, 250.0)[gain > 0])),
            "eligibility_decay": _t(lambda: elig * xp.float32(0.99)),
            "nmda_ampa_split_and_csr": _t(lambda: (
                csp.csr_matrix((data * mask_f, C.indices, C.indptr), shape=C.shape),
                csp.csr_matrix((data * (1 - mask_f), C.indices, C.indptr), shape=C.shape))),
            "transpose_matvec_2col": _t(lambda: C.T @ x2),
            "transpose_matvec_1col": _t(lambda: C.T @ x1),
        }
        out["per_n"][str(N)] = {
            "n_facts": N, "nnz": nnz, "n_neurons": int(n),
            "gain_nonzero_synapses": int((gain != 0).sum()),
            "nmda_routed_synapses": int(b.cp_nmda_recurrent_synapse_mask[:nnz].sum()),
            "profiler_sections": sections, "profiler_steps": int(cnt),
            "dense_op_ms": ops, "dense_op_ms_sum": float(sum(ops.values())),
        }
        print(json.dumps(out["per_n"][str(N)], indent=1), flush=True)
        try:
            b.clear_simulation_state_and_gpu_memory()
        except Exception:
            pass
        del c, b
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
