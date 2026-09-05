"""GNW STOP-TRIGGER ACC/BG circuit -- the DEFAULT-ON (2026-09-05 production-flip) production hook for
`webapp/gnw_global_stop.py`'s `detect_trigger` (scaffold-retirement-backlog rank-12,
`research/coordination/scaffold_retirement_backlog.md`).

WHAT THIS IS. `research/runners/_gnw_acc_bg_stop_trigger_derisk.py` builds + de-risks (6/6 seeds GO) a small
feedforward ACC/BG hyperdirect circuit (two afferent relay pools -> ACC -> STN -> GPi) that reads the SAME two
conflict/mismatch afferents `gnw_global_stop.detect_trigger` already reads (`n_ignited` off `chat._last_gnw_delib`,
`mm_peak` off `chat._last_swap_drives`) DIRECTLY AS SYNAPTIC INPUT, and DECIDES the STOP trigger by spiking
integration (GPi's own late-window firing rate crossing a fixed threshold) instead of a host
`n_ignited >= 2 or swapped`. This module is the thin, additive production glue that makes that de-risked circuit
reachable from `detect_trigger` -- reuse-by-import, NO rebuild of the circuit or the afferents themselves.

CONTRACT (additive, DEFAULT-ON, byte-identical-off; mirrors the `gnw_global_stop`/`swap_drives_chat` pattern).
  * `stop_trigger_spiking_enabled()` gates the whole thing. `BRAIN_GNW_STOP_TRIGGER_SPIKING` unset (DEFAULT, since
    the 2026-09-05 production-flip verify GO) -> `detect_trigger` delegates entirely to `detect_trigger_spiking`
    below for the BOOLEAN decision only; `n_held`/`newcomer` (which content to clear / how to label it -- host
    bookkeeping, not the retired decision) are computed identically either way. An explicit falsy
    (0/false/off/no/'') -> `gnw_global_stop.detect_trigger` runs its ORIGINAL, UNCHANGED host boolean-OR -- this
    module is never imported into that code path, so the opt-out turn is provably byte-identical to the pre-flip
    logic (`git diff` shows the branch, not a rewrite; verified in the data by
    `research/runners/_gnw_stop_trigger_production_flip_verify.py`, 6/6 seeds).
  * `stop_trigger_lesion_on()` -- `BRAIN_GNW_STOP_TRIGGER_LESION` truthy zeroes BOTH afferent->ACC synapses (the
    de-risk's OWN `afferent_lesion` lever): the circuit's ACC/STN/GPi chain survives but receives no afferent drive
    at all, so it can NEVER trigger regardless of the real (n_ignited, mm_peak) read -- the load-bearing proof,
    exercisable in production exactly as the de-risk exercises it standalone.
  * The circuit runs on its OWN PRIVATE RNG timeline; the host process-global RNG (numpy + the sim backend) is
    snapshotted/restored around every read (the #77/#85/gnw-global-stop footgun, inherited pattern) -- enabling this
    module cannot perturb the other response fields even when the flag is on.

REUSE-BY-IMPORT (NO `sim/` edit, NO afferent rebuild). `build_stop_trigger_bridge` + `run_trigger_trial` come
straight from the de-risk (6/6 seeds GO); `n_ignited`/`mm_peak` are READ off `chat._last_gnw_delib` /
`chat._last_swap_drives` exactly as `gnw_global_stop.detect_trigger` already reads them -- this module never calls
`gnw_deliberation.conflict_gate` or `gnw_thought_swap.ThoughtSwapWorkspace` itself (those are the de-risk's OWN
parity-test afferent generators, standalone-only; in production the SAME turn's deliberation/swap organs have
already run and stashed their results on `chat` before `detect_trigger` is called).

HONEST RESIDUALS (named, not claimed closed; inherited from the de-risk).
  1. The scalar->current conversion is host arithmetic (the unavoidable stimulus-injection step; de-risk residual
     #1). What is retired is the COMBINATION (the OR), not this step.
  2. The GPi-rate->boolean read-out threshold is a fixed host constant, the same class of read-out every spiking
     decision in this codebase uses.
  3. The host boolean-OR is NOT deleted -- it remains as an exception-only safety fallback (`except Exception: pass`
     in `detect_trigger`, the same "never crash a turn" idiom every sibling consumer in this file family uses): on
     any turn where the circuit builds without error (the normal case) it never executes, but it is not literally
     gone from the source. Named honestly rather than claimed as a clean scaffold-retirement.
  4. DEFAULT-ON (2026-09-05): this remains a reversible, additive hook -- the explicit-falsy escape hatch is
     verified byte-identical to the pre-flip logic on every seed
     (`research/findings/2026-09-05-gnw-stop-trigger-accbg-circuit-PRODUCTION-FLIP-GO.md`).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

# reuse-by-import the de-risked ACC/BG circuit (6/6 seeds GO) -- NO sim/ edit, NO afferent rebuild.
from research.runners._gnw_acc_bg_stop_trigger_derisk import (
    build_stop_trigger_bridge, run_trigger_trial, GPI_TRIGGER_THRESH,
)

_DEFAULT_SEED = 42

_CIRCUIT_CACHE: dict = {}
_LOCK = threading.Lock()


def stop_trigger_spiking_enabled() -> bool:
    """The master flag. `BRAIN_GNW_STOP_TRIGGER_SPIKING` DEFAULT-ON (2026-09-05 production-flip, verified GO —
    `research/findings/2026-09-05-gnw-stop-trigger-accbg-circuit-PRODUCTION-FLIP-GO.md`): unset -> True (delegate
    to the de-risked spiking ACC/BG circuit for the boolean decision). An explicit falsy (0/false/off/no/'') ->
    False (`gnw_global_stop.detect_trigger` runs its original, unmodified host boolean-OR) — the escape hatch,
    verified byte-identical to the pre-flip logic on every seed. Mirrors `gnw_global_stop.stop_enabled()`'s own
    default-ON style (that flag's 2026-08-26 flip) exactly, per this module's own CONTRACT note above."""
    v = os.environ.get("BRAIN_GNW_STOP_TRIGGER_SPIKING")
    return not (v is not None and v.strip().lower() in ("0", "false", "off", "no", ""))


def stop_trigger_lesion_on() -> bool:
    """The load-bearing lesion lever (the de-risk's OWN `afferent_lesion`). `BRAIN_GNW_STOP_TRIGGER_LESION` truthy
    -> zero BOTH afferent->ACC synapses: the circuit can never trigger regardless of the real afferents."""
    return os.environ.get("BRAIN_GNW_STOP_TRIGGER_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


class _TriggerCircuit:
    """A warm ACC/BG stop-trigger circuit + its private RNG timeline (mirrors `gnw_global_stop._StopWorkspace`).
    One instance per (seed, lesion), cached in `_CIRCUIT_CACHE`."""

    def __init__(self, seed: int = _DEFAULT_SEED, lesion: bool = False):
        self.seed = int(seed)
        self.lesion = bool(lesion)
        self._built = None
        self._rng_state = None
        self._lock = threading.Lock()

    def _isolated(self, fn):
        """Run `fn()` on the circuit's PRIVATE RNG timeline, leaving the host process-global RNG (numpy + the sim
        backend) BYTE-UNTOUCHED -- the #77/#85/gnw-global-stop footgun: the circuit build/step draws OU noise off
        the SAME process-global RNG the rest of the pipeline shares unless explicitly isolated."""
        xp = None
        try:
            from sim.backend import get_backend
            xp, _ = get_backend()
        except Exception:
            xp = None
        host_np = np.random.get_state()
        host_xp = None
        if xp is not None and xp is not np:
            try:
                host_xp = xp.random.get_random_state().get_state()
            except Exception:
                host_xp = None
        if self._rng_state is None:
            np.random.seed(self.seed)
            if xp is not None and xp is not np:
                try:
                    xp.random.seed(self.seed)
                except Exception:
                    pass
        else:
            try:
                np.random.set_state(self._rng_state["np"])
            except Exception:
                pass
            if xp is not None and xp is not np and self._rng_state.get("xp") is not None:
                try:
                    xp.random.get_random_state().set_state(self._rng_state["xp"])
                except Exception:
                    pass
        try:
            return fn()
        finally:
            st = {"np": np.random.get_state(), "xp": None}
            if xp is not None and xp is not np:
                try:
                    st["xp"] = xp.random.get_random_state().get_state()
                except Exception:
                    st["xp"] = None
            self._rng_state = st
            try:
                np.random.set_state(host_np)
            except Exception:
                pass
            if host_xp is not None:
                try:
                    xp.random.get_random_state().set_state(host_xp)
                except Exception:
                    pass

    def _ensure(self):
        if self._built is None:
            self._built = build_stop_trigger_bridge(seed=self.seed, afferent_lesion=self.lesion)
        return self._built

    def decide(self, n_ignited: float, mm_peak: float) -> dict:
        def _do():
            bridge, xp, delib_dev, mm_dev, acc_dev, stn_dev, gpi_dev, snap, handles = self._ensure()
            return run_trigger_trial(bridge, xp, delib_dev, mm_dev, acc_dev, stn_dev, gpi_dev, snap,
                                     n_ignited=n_ignited, mm_peak=mm_peak)
        with self._lock:
            return self._isolated(_do)


def _get_circuit(seed: int, lesion: bool) -> _TriggerCircuit:
    key = (int(seed), bool(lesion))
    c = _CIRCUIT_CACHE.get(key)
    if c is not None:
        return c
    c = _TriggerCircuit(seed, lesion)
    with _LOCK:
        existing = _CIRCUIT_CACHE.get(key)
        if existing is None:
            _CIRCUIT_CACHE[key] = c
        else:
            c = existing
    return c


def detect_trigger_spiking(chat, *, seed: int = _DEFAULT_SEED):
    """Drop-in spiking ACC/BG replacement for `gnw_global_stop.detect_trigger`'s BOOLEAN decision -- same return
    shape `(triggered, reason, n_held, newcomer)`. Reads `n_ignited` off `chat._last_gnw_delib` and `mm_peak` off
    `chat._last_swap_drives` (the SAME already-computed afferents `detect_trigger` reads -- no rebuild), drives the
    reused-by-import circuit, and reads the trigger off its GPi rate crossing `GPI_TRIGGER_THRESH` (spiking
    integration), never a host comparison of n_ignited/mm_peak themselves. `n_held`/`newcomer` are host bookkeeping
    (which content to clear / how to label it) computed identically to `detect_trigger` -- not part of the retired
    decision. Read-only over `chat`; never raises (an unreadable chat or a circuit error -> no trigger)."""
    n_held = 2
    newcomer = None
    n_ignited = None
    mm_peak = 0.0
    try:
        delib = getattr(chat, "_last_gnw_delib", None)
        if isinstance(delib, dict):
            n_ign = delib.get("n_ignited")
            if isinstance(n_ign, (int, float)):
                n_ignited = int(n_ign)
                if n_ignited >= 2:
                    n_held = max(n_held, n_ignited)
    except Exception:
        pass
    try:
        swap = getattr(chat, "_last_swap_drives", None)
        if isinstance(swap, dict):
            mp = swap.get("mm_peak")
            if isinstance(mp, (int, float)):
                mm_peak = float(mp)
            if bool(swap.get("swapped")):
                t = swap.get("new_topic") or swap.get("held_topic")
                newcomer = str(t) if t else None
    except Exception:
        pass

    if n_ignited is None and mm_peak <= 0.0:
        return False, None, n_held, newcomer   # nothing to read -> no trigger (mirrors detect_trigger's inert path)

    try:
        circuit = _get_circuit(seed, stop_trigger_lesion_on())
        r = circuit.decide(n_ignited if n_ignited is not None else 0, mm_peak)
        triggered = bool(r["triggered"])
    except Exception:
        return False, "spiking_accbg_error", n_held, newcomer

    reason = "spiking_accbg" if triggered else None
    return triggered, reason, n_held, newcomer
