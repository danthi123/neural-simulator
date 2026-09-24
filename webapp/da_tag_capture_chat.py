"""DA-GATED TAG-AND-CAPTURE, WIRED INTO CHAT (2026-09-23, branch research/da-tag-capture-chat-wire). DEFAULT-OFF.

WHY. The v3 mechanism (`webapp/da_tag_capture.py` `SynapticTagCaptureLedger`, finding
2026-09-23-da-encoding-natural-drive-v3-synaptic-capture-6seed-GO-runner-level.md) is GO at runner level: the brain's
spiking DA, read by a spiking D1 population, drives a per-synapse tag / PRP / bistable late-phase rule that keeps a
fact told as surprising news at 24 h and lets a plainly-told one decay. It ran only inside a runner: nothing in
`/api/brain-chat` built the ledger, and the load-bearing battery had no next-day turn, so `da-gated-encoding` read
0/6 there. This module wires the SAME ledger (same constants, same a-priori gamma calibration, same spiking D1
reader) into the live chat store path and the continuous engine's idle/sleep tick.

WIRING (three hooks, all no-ops unless `BRAIN_DA_TAG_CAPTURE` is armed):
  * `observe_chat_turn(chat, seed)` -- webapp/server.py, right after the DA-encoding install (the brain's DA-mode read
    of this turn is fresh on `chat._last_da_drives`). The ledger integrates to the turn's world time (store synapses
    decay / capture up to now), then schedules this turn's D1 drive: the brain's DA level broadcast onto the spiking
    D1 (`write_gain`) population for TURN_DRIVE_H (30 s) of world time.
  * `after_store_chat(chat)` -- webapp/server.py, where the turn's response is assembled (after `chat.gate` ->
    `_maybe_acquire` -> composer.store). Registers every store block written since the last call at the turn's time
    (its increment = the composer's own DA-gated write; its baseline = the seeded pre-existing synapse strength) and
    rewrites the block as b + f * inc. Returns the ledger summary, attached to the reply as `da_tag_capture`.
  * `tick_chat(chat)` -- webapp/continuous_engine.py `tick_idle_sessions`, after the Turrigiano pass: integrate to
    world-now and rewrite the store. Night-time decay and capture therefore run on the brain's own offline tick; a
    session the tick skips is caught up on its next observed turn (the ledger is event-driven: same integral).

Blocks stored BEFORE the ledger existed (the tiny-demo build-time knowledge) are not managed (`block_offset`): they are
treated as already-consolidated knowledge. DECLARED.

WORLD CLOCK (the environment -- legitimate host code). `BRAIN_DA_TAG_CAPTURE_CLOCK`:
  "wall" (default): world time = the machine's wall clock since the ledger was built + any environment jump.
  "turn": world time = (#observed turns) x TURN_DRIVE_H + any environment jump -- a scripted conversation's clock: each
          turn lasts exactly 30 s whatever the machine's compute time, so a battery arm on a slow pool machine and one
          on a fast box see the same world. Used by the load-bearing battery probe.
`advance_world_clock_h(h)` is the environment jump (the battery's simulated night). Never called in production.

DETERMINISM. The D1 reader's first build and every D1 read run inside `_private_rng(seed, k)`: the global numpy +
python RNG state is saved, seeded from (seed, k), and restored -- so the ledger never perturbs another organ's RNG
stream and two builds at one seed read identical D1 rates (numpy backend: exact; on cupy the cupy global stream is also
reseeded and cannot be restored -- flag-ON only, declared).

HOST SHORTCUTS (declared; the v3 list plus the wiring's own):
  - the per-synapse tag / PRP / late-phase equations are host-integrated ODEs (constants pre-registered in v3);
  - the rate -> activation normalization is host arithmetic on a measured spike count;
  - TURN_DRIVE_H: one observed turn drives the D1 pool for a fixed 30 s of world time (a conversational-turn duration
    of the environment, not a measured DA time course between turns); a turn that short-circuits before the DA read
    (e.g. a content-empty BG hold) drives nothing and does not advance the turn clock;
  - `SynapticTagCaptureLedger.sync_from_store` bookkeeping for external rescales / rewrites of a managed block;
  - unmanaged build-time blocks (block_offset);
  - the world clock (environment).
The late-phase variable is a synaptic state, not a replay path: this is NOT "consolidation" in the docs/TERMS.md sense.

CONTRACT. DEFAULT-OFF. With `BRAIN_DA_TAG_CAPTURE` unset every hook returns None before touching anything, no ledger is
built, no key is added to the reply: byte-identical (asserted in data by
`research/runners/_da_tag_capture_chat_probe.py --offcheck` against the pinned pre-change SHA). No `sim/` edit.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np

from webapp.da_tag_capture import (_DA_TONIC, SpikingD1Activation, SynapticTagCaptureLedger, calibrate_gamma,
                                   tag_capture_enabled)

TURN_DRIVE_H = 30.0 / 3600.0      # == research/runners/_da_encoding_natural_drive_persistence.TURN_H (v3)
D1_READER_SEED = 42               # the production write-gain reader seed (as in the v3 runner)
_WORLD_OFFSET_H = 0.0             # environment clock jump; only the battery's scripted night moves it


def store_composer(chat):
    """The composer whose `store_conns` hold the conversation's facts. The tiny-demo production brain wraps it in a
    `TieredFactStore(buffer, ltm)` (the recent-conversation buffer + the routed LTM shard); the ledger acts on the
    BUFFER's store synapses (where `_maybe_acquire` writes), never on the LTM shard. None if absent."""
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    if comp is not None and type(comp).__name__ == "TieredFactStore":
        comp = object.__getattribute__(comp, "buffer")
    return comp


def clock_mode() -> str:
    v = os.environ.get("BRAIN_DA_TAG_CAPTURE_CLOCK", "wall").strip().lower()
    return "turn" if v == "turn" else "wall"


def advance_world_clock_h(hours: float) -> float:
    """The ENVIRONMENT's clock jump (a night passes between two conversations). Returns the new offset (hours)."""
    global _WORLD_OFFSET_H
    _WORLD_OFFSET_H += float(hours)
    return _WORLD_OFFSET_H


def world_offset_h() -> float:
    return _WORLD_OFFSET_H


class _private_rng:
    """Save the global numpy + python RNG, seed a private stream from (seed, k), restore on exit."""

    def __init__(self, seed: int, k: int):
        self.s = (int(seed) * 1000003 + int(k) * 7919 + 17) % (2 ** 32)

    def __enter__(self):
        import random as _random
        self._np = np.random.get_state()
        self._py = _random.getstate()
        np.random.seed(self.s)
        _random.seed(self.s)
        try:
            from sim.backend import get_backend
            xp, _ = get_backend()
            if xp is not np and hasattr(xp, "random"):
                xp.random.seed(self.s)
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        import random as _random
        np.random.set_state(self._np)
        _random.setstate(self._py)
        return False


class ChatTagCapture:
    """One chat session's tag-and-capture state: the v3 ledger + its world clock."""

    def __init__(self, chat, seed: int):
        comp = store_composer(chat)
        self.seed = int(seed)
        with _private_rng(self.seed, 0):
            self.d1 = SpikingD1Activation(reader_seed=D1_READER_SEED)
        self.gamma = calibrate_gamma(self.d1.a_go)          # a priori, no brain data (v3 module)
        self.ledger = SynapticTagCaptureLedger(self.seed, gamma=self.gamma, d1=self.d1,
                                               block_offset=len(comp.store_conns) // comp.D)
        self.mode = clock_mode()
        self.t0_wall = time.time()
        self.t0_offset = _WORLD_OFFSET_H
        self.n_turns = 0
        self.t_turn = 0.0

    def now_h(self) -> float:
        jump = _WORLD_OFFSET_H - self.t0_offset
        if self.mode == "turn":
            return self.n_turns * TURN_DRIVE_H + jump
        return (time.time() - self.t0_wall) / 3600.0 + jump

    def _catch_up(self, comp, t: float) -> None:
        self.ledger.sync_from_store(comp, t)
        self.ledger.on_store(comp, t)          # a block written outside an observed turn (none on the probe path)
        self.ledger.advance(comp, t)

    def observe(self, chat, da_level: float) -> dict:
        comp = store_composer(chat)
        t = max(self.now_h(), self.ledger.t)   # the ledger's clock never runs backwards
        self._catch_up(comp, t)
        with _private_rng(self.seed, self.n_turns + 1):
            a_eff = self.ledger.observe_turn(t, TURN_DRIVE_H, float(da_level))
        self.t_turn = t
        self.n_turns += 1
        return {"t_h": t, "da_level": float(da_level), "a_eff": a_eff}

    def after_store(self, chat) -> int:
        comp = store_composer(chat)
        self.ledger.sync_from_store(comp, self.t_turn)
        return self.ledger.on_store(comp, self.t_turn)

    def tick(self, chat) -> float:
        comp = store_composer(chat)
        t = max(self.now_h(), self.ledger.t)
        self._catch_up(comp, t)
        return t

    def summary(self) -> dict:
        L = self.ledger
        return {"on": True, "clock": self.mode, "world_t_h": L.t, "n_turns": self.n_turns, "gamma": self.gamma,
                "d1_a_go": self.d1.a_go, "p": L.p, "p_max": L.p_max, "block_offset": L.block_offset,
                "n_managed_blocks": len(L.blocks), "external_rescales": L.n_external_rescales,
                "external_rewrites": L.n_external_rewrites, "blocks": L.summary(),
                "last_turn": (L.turn_log[-1] if L.turn_log else None)}


def get_chat_capture(chat, seed: int) -> Optional[ChatTagCapture]:
    """The session's capture state, built lazily on the first observed turn. None when the flag is off or the chat
    has no block-structured composer store."""
    if not tag_capture_enabled():
        return None
    cap = getattr(chat, "_da_tag_capture", None)
    if cap is None:
        comp = store_composer(chat)
        if comp is None or not hasattr(comp, "store_conns") or not hasattr(comp, "D"):
            return None
        cap = ChatTagCapture(chat, seed)
        chat._da_tag_capture = cap
    return cap


def observe_chat_turn(chat, seed: int) -> Optional[dict]:
    """Server hook, right after the DA-encoding install (the brain's DA-mode read is fresh). None when off."""
    cap = get_chat_capture(chat, seed)
    if cap is None:
        return None
    da = (getattr(chat, "_last_da_drives", None) or {}).get("da_level")
    da = _DA_TONIC if da is None else float(da)
    return {"on": True, "observed": cap.observe(chat, da)}


def after_store_chat(chat) -> Optional[dict]:
    """Server hook, where the turn's response is assembled (after the gate's store). None when off."""
    cap = getattr(chat, "_da_tag_capture", None)
    if cap is None or not tag_capture_enabled():
        return None
    n_new = cap.after_store(chat)
    out = cap.summary()
    out["new_blocks_this_turn"] = n_new
    return out


def tick_chat(chat) -> Optional[dict]:
    """Continuous-engine idle-tick hook: integrate the session's ledger to world-now and rewrite its store."""
    cap = getattr(chat, "_da_tag_capture", None)
    if cap is None or not tag_capture_enabled():
        return None
    return {"t_h": cap.tick(chat), "p": cap.ledger.p}
