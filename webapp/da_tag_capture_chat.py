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

SLEEP ROUTE (branch research/sleep-replay-capture, default-OFF `BRAIN_SLEEP_REPLAY_CAPTURE`): `_catch_up` first runs any
due NREM cycle of webapp/sleep_replay_capture.py (SWR reactivation read off the store's own cleanup -> re-tag +
SWR-coupled D1 drive into the same PRP pool). Flag unset -> never entered.

AWAKE-REST ROUTE (branch research/awake-replay-capture, default-OFF `BRAIN_AWAKE_REPLAY_CAPTURE`): `tick` (the idle tick
only, never a turn) runs one quiet-rest reactivation bout of webapp/awake_replay_capture.py while the ledger's clock
says the brain is still awake (read-back -> reactivation-induced early LTP + a fresh tag; no PRP). Flag unset -> never
entered.

Blocks stored BEFORE the ledger existed (the tiny-demo build-time knowledge) are not managed (`block_offset`): they are
treated as already-consolidated knowledge. DECLARED.

WORLD CLOCK (the environment -- legitimate host code). `BRAIN_DA_TAG_CAPTURE_CLOCK`:
  "wall" (default): world time = the machine's wall clock since the ledger was built + any environment jump.
  "turn": world time = (#observed turns) x TURN_DRIVE_H + any environment jump -- a scripted conversation's clock: each
          turn lasts exactly 30 s whatever the machine's compute time, so a battery arm on a slow pool machine and one
          on a fast box see the same world. Used by the load-bearing battery probe.
`advance_world_clock_h(h)` is the environment jump (the battery's simulated night). Never called in production.
WALL-CLOCK SEAM (branch research/pair-production-path-arms, 2026-09-25; review B3 of
research/findings/2026-09-25-da-capture-sleep-replay-pair-verify-go-review.md): the "wall" clock reads `_wall_now()`,
which is `time.time()` unless the ENVIRONMENT installed another source with `set_wall_clock(fn)`. Production never
calls it (so the wall path is byte-identical: `time.time()` read at the same two sites). The battery's virtual-day world
steps install a deterministic virtual wall clock so a wall-clock arm on numpy is reproducible and machine-speed-free.

DETERMINISM. The D1 reader's first build and every D1 read run inside `_private_rng(seed, k)`: the global numpy +
python RNG state is saved, seeded from (seed, k), and restored -- so the ledger never perturbs another organ's RNG
stream and two builds at one seed read identical D1 rates (numpy backend: exact). On cupy (branch
research/pair-production-path-arms, review D3) the cupy global RandomState object of the current device is SWAPPED for a
private one seeded from (seed, k) and the original object is put back on exit, so the cupy stream every other organ
draws from is left exactly where it was (before this fix it was reseeded and never restored). The draws inside the
context are the same as before (a fresh RandomState(s) and a reseed to s give one stream).
`BRAIN_DA_TAG_CAPTURE_CUPY_NO_RESTORE=1` (default OFF, measurement-only) reproduces the pre-fix reseed, so a cupy arm can
measure the drift the fix removes. Numpy: the cupy branch is never entered. The reader itself is built in an ISOLATED cache namespace
(`SpikingD1Activation(..., isolated=True)` -> `_da_write_gain_spiking_derisk._get_isolated_reader`), not the shared
one production's `BRAIN_DA_ENCODING_SPIKING_GAIN` read populates -- see that function's docstring for the 2026-09-23
fix (review v2:dd14adaf7): sharing the production cache made gamma/d1_a_go track which arm happened to build the
entry first (intact vs lesion-pinned), not the lesioned edge. `grade_seed` below asserts gamma/d1_a_go equal across
every companion-ON arm at a seed as a standing check on this.

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
_WALL_CLOCK = None                # environment wall-clock source (seconds); None -> time.time (production)


def set_wall_clock(fn):
    """ENVIRONMENT seam: make the "wall" clock read `fn()` (seconds) instead of `time.time()`. `None` restores
    `time.time`. Returns the previous source. Never called in production."""
    global _WALL_CLOCK
    prev = _WALL_CLOCK
    _WALL_CLOCK = fn
    return prev


def _wall_now() -> float:
    return time.time() if _WALL_CLOCK is None else float(_WALL_CLOCK())


def _cupy_no_restore() -> bool:
    """`BRAIN_DA_TAG_CAPTURE_CUPY_NO_RESTORE` (default OFF, measurement-only): reproduce the pre-fix cupy reseed."""
    return os.environ.get("BRAIN_DA_TAG_CAPTURE_CUPY_NO_RESTORE", "0").strip().lower() in ("1", "true", "on", "yes")


def _replay_capture_enabled() -> bool:
    """`BRAIN_SLEEP_REPLAY_CAPTURE` (default OFF) -- read here so the flag-off path imports nothing new."""
    return os.environ.get("BRAIN_SLEEP_REPLAY_CAPTURE", "0").strip().lower() in ("1", "true", "on", "yes")


def _awake_replay_enabled() -> bool:
    """`BRAIN_AWAKE_REPLAY_CAPTURE` (default OFF) -- read here so the flag-off path imports nothing new."""
    return os.environ.get("BRAIN_AWAKE_REPLAY_CAPTURE", "0").strip().lower() in ("1", "true", "on", "yes")


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


def mark_awake(chat) -> Optional[float]:
    """The ENVIRONMENT/BODY says this session's brain stayed AWAKE up to world-now (research/sleep-replay-capture-r2:
    the battery's `awake_*` world step, a waking interval with no conversation). Without it the ledger's sleep route
    uses the engine's convention (idle >= SLEEP_IDLE_SEC = asleep); with it, sleep onset is measured from the end of
    the waking interval. Only the sleep route (BRAIN_SLEEP_REPLAY_CAPTURE) reads it. Returns the mark, or None when
    this chat has no ledger. Never called in production (the live server has no awake/asleep signal of its own)."""
    cap = getattr(chat, "_da_tag_capture", None)
    if cap is None:
        return None
    cap.awake_until_h = max(cap.now_h(), cap.ledger.t)
    return cap.awake_until_h


class _private_rng:
    """Save the global numpy + python RNG, seed a private stream from (seed, k), restore on exit. On a cupy backend the
    device's global RandomState object is swapped for a private one and put back on exit (see DETERMINISM above)."""

    def __init__(self, seed: int, k: int):
        self.s = (int(seed) * 1000003 + int(k) * 7919 + 17) % (2 ** 32)

    def __enter__(self):
        import random as _random
        self._np = np.random.get_state()
        self._py = _random.getstate()
        self._xp = None
        self._xp_rs = None
        np.random.seed(self.s)
        _random.seed(self.s)
        try:
            from sim.backend import get_backend
            xp, _ = get_backend()
            if xp is not np and hasattr(xp, "random"):
                xr = xp.random
                if (not _cupy_no_restore() and hasattr(xr, "get_random_state") and hasattr(xr, "set_random_state")
                        and hasattr(xr, "RandomState")):
                    self._xp_rs = xr.get_random_state()          # the object other organs draw from, untouched
                    xr.set_random_state(xr.RandomState(self.s))
                    self._xp = xp
                else:
                    xr.seed(self.s)                               # pre-fix behaviour (declared; measurement knob)
        except Exception:
            pass
        return self

    def __exit__(self, *exc):
        import random as _random
        np.random.set_state(self._np)
        _random.setstate(self._py)
        if self._xp is not None:
            try:
                self._xp.random.set_random_state(self._xp_rs)
            except Exception:
                pass
        return False


class ChatTagCapture:
    """One chat session's tag-and-capture state: the v3 ledger + its world clock."""

    def __init__(self, chat, seed: int):
        comp = store_composer(chat)
        self.seed = int(seed)
        with _private_rng(self.seed, 0):
            # isolated=True: build/fetch the D1 reader from a cache namespace production's
            # BRAIN_DA_ENCODING_SPIKING_GAIN path never touches, so this arm's gamma/d1_a_go cannot depend on
            # whether production happened to build (intact) or skip (lesion-pinned) the SHARED reader first in
            # this process (2026-09-23 fix, review v2:dd14adaf7 -- see `_get_isolated_reader`'s docstring).
            self.d1 = SpikingD1Activation(reader_seed=D1_READER_SEED, isolated=True,
                                          isolated_tag="da_tag_capture_chat")
        self.gamma = calibrate_gamma(self.d1.a_go)          # a priori, no brain data (v3 module)
        self.ledger = SynapticTagCaptureLedger(self.seed, gamma=self.gamma, d1=self.d1,
                                               block_offset=len(comp.store_conns) // comp.D)
        self.mode = clock_mode()
        self.t0_wall = _wall_now()
        self.t0_offset = _WORLD_OFFSET_H
        self.n_turns = 0
        self.t_turn = 0.0

    def now_h(self) -> float:
        jump = _WORLD_OFFSET_H - self.t0_offset
        if self.mode == "turn":
            return self.n_turns * TURN_DRIVE_H + jump
        return (_wall_now() - self.t0_wall) / 3600.0 + jump

    def _catch_up(self, comp, t: float) -> None:
        # SLEEP-REPLAY CAPTURE (default-OFF `BRAIN_SLEEP_REPLAY_CAPTURE`, webapp/sleep_replay_capture.py): before the
        # ledger moves to t, run every NREM cycle of the sleep episode that began a sleep-onset interval after the last
        # observed turn and is due by t. Flag unset -> this branch is never entered -> byte-identical.
        if _replay_capture_enabled() and self.n_turns > 0:
            from webapp import sleep_replay_capture as _SRC
            if getattr(self, "_src", None) is None:
                self._src = _SRC.SleepReplayCapture(self.seed, self.d1, rng_ctx=_private_rng)
            # the body was awake until the later of the last turn and any environment awake mark (r2 item 1)
            t_ref, key = self.t_turn, self.n_turns
            aw = getattr(self, "awake_until_h", None)
            if aw is not None and aw > t_ref:
                t_ref, key = aw, (self.n_turns, round(aw, 9))   # a waking interval starts its own sleep episode
            self._src.catch_up(self.ledger, comp, t_ref, key, t)
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
        # AWAKE-REST REPLAY (default-OFF `BRAIN_AWAKE_REPLAY_CAPTURE`, webapp/awake_replay_capture.py): on an idle tick
        # while the ledger's own clock says the brain is still awake, one quiet-rest reactivation bout (at most one per
        # AWAKE_BOUT_H). Only the idle tick reaches here (a turn is not rest). Flag unset -> never entered.
        if _awake_replay_enabled() and self.n_turns > 0:
            from webapp import awake_replay_capture as _ARC
            if getattr(self, "_arc", None) is None:
                self._arc = _ARC.AwakeReplayCapture(self.seed, rng_ctx=_private_rng)
            t_ref = self.t_turn
            aw = getattr(self, "awake_until_h", None)
            if aw is not None and aw > t_ref:
                t_ref = aw
            self._arc.maybe_bout(self.ledger, comp, t_ref, t)
        return t

    def summary(self) -> dict:
        L = self.ledger
        out = {"on": True, "clock": self.mode, "world_t_h": L.t, "n_turns": self.n_turns, "gamma": self.gamma,
               "d1_a_go": self.d1.a_go, "p": L.p, "p_max": L.p_max, "block_offset": L.block_offset,
               "n_managed_blocks": len(L.blocks), "external_rescales": L.n_external_rescales,
               "external_rewrites": L.n_external_rewrites, "blocks": L.summary(),
               "last_turn": (L.turn_log[-1] if L.turn_log else None)}
        if getattr(self, "_src", None) is not None:          # only ever set with BRAIN_SLEEP_REPLAY_CAPTURE armed
            out["sleep_replay_capture"] = self._src.summary()
        if getattr(self, "awake_until_h", None) is not None:  # only ever set by the battery's awake world step (r2)
            out["awake_until_h"] = self.awake_until_h
        if getattr(self, "_arc", None) is not None:          # only ever set with BRAIN_AWAKE_REPLAY_CAPTURE armed
            out["awake_replay_capture"] = self._arc.summary()
        return out


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
