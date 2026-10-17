"""preserved_global_rng -- run a block that may reseed the process-global RNGs, then put every global stream back.

WHY. Building a `SimulationBridge` runs `_initialize_simulation_data` -> `_initialize_rng(cfg.seed)`
(sim/bridge.py), which reseeds THREE process-global streams: `cp.random.seed` (the backend module: numpy on the numpy
backend, cupy on the cupy backend), `np.random.seed` and `random.seed`. The cortical LTM builds bridges lazily (one per
RF op size per shard on first use, one consolidated retrieve bridge per shard, one per fact at load), all at seed 42.
So a recall that happens to be the first on a shard reseeds the whole process to the seed-42 stream, whatever
BRAIN_CHAT_SEED is, and any later global draw (e.g. the OU noise `cp.random.randn` of a chat-path bridge) reads that
stream. With one LTM shared across conversations, whether a recall reseeds also depends on what OTHER conversations
already asked. Found in review of claude-build/shared-ltm, 2026-10-13 (research/FAILURE_LOG.md).

WHAT. Inside the block, numpy's and python's global states are saved and restored exactly; for cupy the CURRENT
global RandomState object is swapped out for a temporary one (cupy.random.seed reseeds the current object IN PLACE, so
saving a reference would not protect it) and put back afterwards, untouched. Draws the block makes from the global
streams come from the temporary / restored-over state and leave no trace outside the block.

THREADS (re-review, 2026-10-13). The global streams are process-wide, so two guarded blocks on two threads must not
overlap: A saves S0 and reseeds, B saves A's reseeded state, A restores S0, B then restores the seed-42 state. A
module-level RLock (`_GUARD_LOCK`) is held across the whole block, so guarded blocks serialize (re-entrant on one
thread, so nesting works). It is a LEAF lock: nothing inside a guarded block takes another lock of this code, so it
cannot join a lock-order cycle (the callers take their own locks -- a shared store's RLock, a key's load lock --
BEFORE this one).
RESIDUAL: a thread that draws from a global stream WITHOUT the guard while another thread is inside a guarded block
reads the block's reseeded values. The guard restores the streams; it cannot hide the window from unguarded readers.
"""
from __future__ import annotations

import contextlib
import random
import threading

_GUARD_LOCK = threading.RLock()


def _backend_name():
    try:
        from sim.backend import get_backend
        return get_backend()[1]
    except Exception:
        return "numpy"


@contextlib.contextmanager
def preserved_global_rng():
    with _GUARD_LOCK:
        with _preserved_global_rng_unlocked():
            yield


@contextlib.contextmanager
def _preserved_global_rng_unlocked():
    import numpy as np
    np_state = np.random.get_state()
    py_state = random.getstate()
    cupy_restore = None
    if _backend_name() == "cupy":
        import cupy
        try:
            from cupy.random import _generator as _g
            dev = cupy.cuda.Device().id
            states = _g._random_states
            had, orig = dev in states, states.get(dev)
            states[dev] = _g.RandomState(0)

            def cupy_restore():
                if had:
                    states[dev] = orig
                else:
                    states.pop(dev, None)
        except Exception:                      # private layout changed: the public API (creates the state if absent)
            orig = cupy.random.get_random_state()
            cupy.random.set_random_state(cupy.random.RandomState(0))

            def cupy_restore():
                cupy.random.set_random_state(orig)
    try:
        yield
    finally:
        if cupy_restore is not None:
            cupy_restore()
        random.setstate(py_state)
        np.random.set_state(np_state)
