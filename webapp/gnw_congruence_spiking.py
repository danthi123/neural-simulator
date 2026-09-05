"""GNW CONGRUENCE spiking read — production glue for scaffold-retirement backlog rank-8, DEFAULT-OFF.

WHAT THIS IS. `webapp/gnw_bus_shadow.py::_organ_reads` is the LIVE production organ-combination read (installed
by default since the 2026-08-13 flip/retirement — `webapp/server.py::brain_reply` runs it on every turn). Two of
its three organs decide "does this second read CORROBORATE the first" with a bare host `==`:

    cand_B = cand_A if composer.query_patient(agent, action) == cand_A else None     # organ B: VERIFY re-check
    cand_C = cand_A if composer.query_agent(action, cand_A) == agent else None        # organ C: reverse-binding

This module supplies the retirement: `SpikingCongruenceReader` (`research/runners/
_gnw_congruence_spiking_read_derisk.py`, 6/6-seed GO) reuses the ALREADY-6/6-GO'd swap-intention circuit's
`pred_k -> mm_k` MATCH VETO — the SAME ignition-workspace populations already load-bearing on the neural
thought-swap decision — to read "does `proposed` match `held`" off spiking population dynamics instead of a host
`==`. See that module's docstring for the full mechanism + the "ADDRESSING VS DECIDING" honesty note (a content->
slot lookup is unavoidable wiring; the verdict itself is never a host comparison of the resulting indices).

CONTRACT (additive, DEFAULT-OFF, reversible, mirrors `gnw_thought_swap.py` / `gnw_two_organ_bus.py`):
  * `BRAIN_GNW_CONGRUENCE_SPIKING` unset/0/false/off/no (DEFAULT) -> `webapp.gnw_bus_shadow._organ_reads` never
    even imports this module (the check is a cheap env-var read in `gnw_bus_shadow._congruence_spiking_enabled`);
    organ B/C's congruence is decided by the ORIGINAL host `==` -> BYTE-IDENTICAL to today's production.
  * `BRAIN_GNW_CONGRUENCE_SPIKING` truthy -> `_organ_reads` routes organ B/C's raw second read through
    `spiking_congruent(held, proposed)` instead of `held == proposed`.
  * `BRAIN_GNW_CONGRUENCE_LESION` truthy -> every congruence read runs with the reused circuit's own TRIGGER-LESION
    (mm's proposal drive silenced) -> MISMATCH can no longer be discriminated from MATCH (both read "congruent")
    -> organ B/C's corroboration is no longer selective -> the bus's answer-vs-abstain behaviour on a genuine
    mismatch collapses toward "always corroborates" (the load-bearing proof, mirrors the de-risk's own lever).
  * `spiking_congruent` NEVER raises out: on any error it degrades to the ORIGINAL host `held == proposed` so a
    turn can never crash, and the one read falls back to pre-flip behaviour rather than aborting the turn.
  * RNG ISOLATION. The reused circuit reseeds `cfg.seed` at build time and steps OU noise off the SAME
    process-global RNG the rest of the pipeline shares; `_isolated` snapshots/restores the host RNG around every
    read (the SAME pattern `gnw_thought_swap.ThoughtSwapWorkspace._isolated` and `gnw_two_organ_bus`'s warm-bridge
    callers already use) so enabling this flag cannot perturb any OTHER RNG-dependent organ in the same turn.

REUSE-BY-IMPORT (NO `sim/` edit). The spiking congruence mechanism comes straight from
`research/runners/_gnw_congruence_spiking_read_derisk.py` (6/6-seed GO); this module adds only the production
glue (a process-cached warm reader, the RNG-isolation wrapper, the two flags).
"""
from __future__ import annotations

import os
import threading

import numpy as np

from research.runners._gnw_congruence_spiking_read_derisk import SpikingCongruenceReader

_DEFAULT_SEED = 42
_LOCK = threading.Lock()
_READERS: dict = {}          # seed -> warm SpikingCongruenceReader, built lazily on the first enabled turn


def congruence_spiking_enabled() -> bool:
    """The master flag. `BRAIN_GNW_CONGRUENCE_SPIKING` in {1,true,on,yes} -> organ B/C's congruence check uses the
    spiking `pred_k->mm_k` match-veto read. Default (unset/0/false/off/no) -> OFF; `_organ_reads` runs its
    ORIGINAL host `==` logic and this module is never imported by the caller."""
    return os.environ.get("BRAIN_GNW_CONGRUENCE_SPIKING", "").strip().lower() in ("1", "true", "on", "yes")


def congruence_lesion_on() -> bool:
    """`BRAIN_GNW_CONGRUENCE_LESION` truthy -> the TRIGGER-LESION (mm's proposal drive silenced, the de-risk's own
    load-bearing lever) is applied to every read this turn -> MATCH and MISMATCH become indistinguishable (both
    read "congruent"). Default OFF."""
    return os.environ.get("BRAIN_GNW_CONGRUENCE_LESION", "").strip().lower() in ("1", "true", "on", "yes")


def get_congruence_reader(seed: int = _DEFAULT_SEED) -> SpikingCongruenceReader:
    """Build (once) + cache a warm `SpikingCongruenceReader` for `seed`. Reused across every enabled turn (mirrors
    `gnw_thought_swap.get_swap_workspace` / `gnw_two_organ_bus._get_bridge`'s own process-cache pattern)."""
    r = _READERS.get(int(seed))
    if r is not None:
        return r
    with _LOCK:
        r = _READERS.get(int(seed))
        if r is None:
            r = SpikingCongruenceReader(seed=int(seed))
            _READERS[int(seed)] = r
    return r


def _isolated(reader: SpikingCongruenceReader, fn):
    """Run `fn()` on `reader`'s PRIVATE RNG timeline, leaving the host process-global RNG (numpy + the active sim
    backend) byte-untouched. See the module docstring's RNG-ISOLATION note."""
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
    state = reader._rng_state
    if state is None:
        np.random.seed(reader.seed)
        if xp is not None and xp is not np:
            try:
                xp.random.seed(reader.seed)
            except Exception:
                pass
    else:
        try:
            np.random.set_state(state["np"])
        except Exception:
            pass
        if xp is not None and xp is not np and state.get("xp") is not None:
            try:
                xp.random.get_random_state().set_state(state["xp"])
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
        reader._rng_state = st
        try:
            np.random.set_state(host_np)
        except Exception:
            pass
        if host_xp is not None:
            try:
                xp.random.get_random_state().set_state(host_xp)
            except Exception:
                pass


def spiking_congruent(held, proposed, *, seed: int = _DEFAULT_SEED, lesion: "bool | None" = None) -> bool:
    """The production entry point `webapp.gnw_bus_shadow._organ_reads` calls when `congruence_spiking_enabled()`.
    Returns a plain bool. `lesion=None` (default) reads `congruence_lesion_on()`; pass an explicit bool to override
    (used by the hook-verify harness). NEVER raises out: on any error, degrade to the ORIGINAL host `held ==
    proposed` so a turn can never crash and the one read falls back to pre-flip behaviour."""
    try:
        les = congruence_lesion_on() if lesion is None else bool(lesion)
        reader = get_congruence_reader(seed)
        r = _isolated(reader, lambda: reader.congruent(held, proposed, lesion=les))
        return bool(r.get("congruent"))
    except Exception:
        return bool(held == proposed)
