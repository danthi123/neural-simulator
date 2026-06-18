"""Clean quiet-GPU speedup for the RF megakernel (cfg.enable_rf_cudagraph) vs the per-step loop, measured on the
PRODUCTION composer op (a bind+unbind+cleanup query) at the agent's D=128. Reports the per-op latency both ways +
the ratio. Also exercises the BrainConversationalAgent enable_rf_cudagraph pass-through (import + build). The
answer-identity gate is _phaseB_megakernel_conversation_validation.py; this is the latency number only.

Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_megakernel_clean_speedup
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim.backend import is_gpu_backend, synchronize  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402


def _time_query(composer, n_iters, n_warmup):
    """One 'query' = store a fact then query_patient (bind+bundle store, then unbind+cleanup). Times the loop."""
    composer.store("dog", "go", "north")
    composer.store("cat", "come", "south")
    composer.store("bird", "fly", "west")
    for _ in range(n_warmup):
        composer.query_patient("dog", "go")
    synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        composer.query_patient("dog", "go")
    synchronize()
    return (time.time() - t0) / n_iters * 1e3  # ms/query


def main():
    if not is_gpu_backend():
        print("NOT GPU backend -- the megakernel is GPU-only; skipping.", flush=True)
        return
    N, W = 20, 3
    VOCAB = ["dog", "cat", "bird", "go", "come", "fly", "north", "south", "west"]
    print("[megakernel clean speedup] production query (bind+bundle store amortized; unbind+cleanup timed)\n",
          flush=True)

    loop = RFPhasorComposer(seed=42, D=128, period=200, vocab=VOCAB, enable_rf_cudagraph=False)
    ms_loop = _time_query(loop, N, W)
    print(f"  loop       : {ms_loop:8.1f} ms/query", flush=True)

    mega = RFPhasorComposer(seed=42, D=128, period=200, vocab=VOCAB, enable_rf_cudagraph=True)
    ms_mega = _time_query(mega, N, W)
    print(f"  megakernel : {ms_mega:8.1f} ms/query", flush=True)

    print(f"\n  speedup    : {ms_loop / ms_mega:6.2f}x  (clean quiet-GPU)", flush=True)

    # Exercise the agent pass-through (build only -- the answer-identity gate is the conversation validation runner).
    a = BrainConversationalAgent(seed=42, concepts={w: None for w in ["dog", "go", "north"]},
                                 enable_rf_cudagraph=True)
    assert a.composer._enable_rf_cudagraph is True, "agent enable_rf_cudagraph pass-through broken"
    print("  agent enable_rf_cudagraph pass-through: OK (composer._enable_rf_cudagraph=True)", flush=True)


if __name__ == "__main__":
    main()
