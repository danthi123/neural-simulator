"""D6 LEARN-THROUGH-USE, WIRED INTO CHAT AS AN OBSERVABILITY HOOK (2026-09-24, branch research/d6-chat-wire).
DEFAULT-OFF.

WHY. The local spiking Hebbian fact-write rule (`research/runners/d6_hebbian_store.py`, `BRAIN_D6_HEBBIAN_STORE`,
default OFF) already sits ON the production write path: `OneBrainComposer._store_composite`
(research/runners/one_brain_composer.py:872) calls `d6_hebbian_store.hebbian_encode` in place of the direct
composite copy, and `ChatBrain._maybe_acquire` (research/runners/brain_chat_tui.py:1140) already wraps its writes
in `d6_hebbian_store.conversation_write` so the plasticity-freeze lesion (`BRAIN_D6_HEBBIAN_FREEZE`) targets only
in-conversation learning. The READ side is wired too: `webapp/server.py` already calls
`d6_hebbian_store.readtime_refresh` at the top of every turn and `d6_hebbian_store.visible_kb` at the episodic
content lookup (both gated on `BRAIN_D6_ENGRAM_READTIME`). This mechanism is GO 6/6 on the pre-registered K1-K7
capability gate at runner level (`research/findings/2026-09-23-d6-learn-through-use-v3-capability-gate-GO-6of6.md`,
`research/runners/d6_learn_through_use_lb.py`, calling `webapp.server.brain_chat` directly).

THE GAP THIS MODULE CLOSES. None of that produces anything a person (or a battery) reading the `/api/brain-chat`
JSON response can see: there is no `d6_hebbian` key, so nobody watching the live chat can tell whether a turn's
fact write went through the local Hebbian rule, what its encode looked like (frozen / saturated synapse count /
mean |w|), or whether the block it just wrote is currently held as an engram. This module adds that surface. It
does NOT re-implement or duplicate the write (that stays exactly in `d6_hebbian_store.hebbian_encode` /
`one_brain_composer._store_composite`) or the read (that stays in `readtime_refresh` / `visible_kb`); it only
reports, after the fact, what already happened this turn on the substrate — mirroring the read-only reporting
pattern of `webapp/da_tag_capture_chat.py`.

WIRING (one hook, a no-op unless `BRAIN_D6_HEBBIAN_STORE` is armed):
  * `after_store_d6(chat)` — called from `webapp/server.py` alongside the existing `da_tag_capture_chat.after_
    store_chat` call, after `chat.gate()` / `_maybe_acquire` has had the chance to write a new fact block this
    turn. Reads `composer._d6_last_encode` (set by `OneBrainComposer._store_composite` THIS turn via the existing
    wire; this module never writes it) and CONSUMES it (sets it back to None), so a later turn that made no new
    write reports `wrote_this_turn: False` rather than replaying a stale encode. When a block was just written, it
    also takes one genuine neural read of that block's held/engram status
    (`d6_hebbian_store.engram_held` — the same read the K4/K5 gate criteria use, not a new instrument).

CONTRACT. DEFAULT-OFF. With `BRAIN_D6_HEBBIAN_STORE` unset, `d6_hebbian_enabled()` is False and `after_store_d6`
returns None before reading or touching `chat`/`composer` state: byte-identical
(`research/runners/_d6_chat_wire_probe.py --offcheck`, against the pinned pre-change SHA — the merge-base of this
branch and origin/main, following the 2026-09-24 `da_tag_capture_chat` pinning fix so a later unrelated main merge
cannot be mistaken for this branch's own change). No `sim/` edit. No change to `research/runners/{d6_hebbian_
store,one_brain_composer,brain_chat_tui}.py` — this module only reads state those files already produce.

HOST SHORTCUTS: none beyond the ones `d6_hebbian_store.py` already declares (a)-(j) in its own docstring. This
module performs no computation between sensation and action; it reports a value already computed on the
substrate (the encode diagnostic) and takes one already-defined neural read (`engram_held`).

HONESTY. Functional read-outs only. "wrote_this_turn" / "held" mean a measured synaptic-write / neural-activity
event (established by the lesion-verified K1-K7 gate upstream), never a claim of felt experience.
"""
from __future__ import annotations

import os
from typing import Optional

_ON = ("1", "true", "yes", "on")


def d6_hebbian_enabled() -> bool:
    """`BRAIN_D6_HEBBIAN_STORE` in {1,true,yes,on} -> the composer writes fact blocks with the local Hebbian rule
    (research/runners/d6_hebbian_store.py) and this module's observability hook is active. Default OFF."""
    return os.environ.get("BRAIN_D6_HEBBIAN_STORE", "").strip().lower() in _ON


def store_composer(chat):
    """The composer whose `store_conns` / `_d6_last_encode` this hook reads. Mirrors
    `webapp.da_tag_capture_chat.store_composer`: the tiny-demo production brain wraps the real composer in a
    `TieredFactStore(buffer, ltm)`; the D6 write (and so `_d6_last_encode`) lands on the BUFFER, never the LTM
    shard. None if the chat has no such composer."""
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    if comp is not None and type(comp).__name__ == "TieredFactStore":
        comp = object.__getattribute__(comp, "buffer")
    return comp


def after_store_d6(chat) -> Optional[dict]:
    """Server hook: report this turn's D6 encode (if the composer wrote one) and a fresh engram-held read of that
    block. Returns None when the flag is off, when the chat has no block-structured composer, or on any internal
    error reading composer state (never lets a reporting failure crash a turn — the same discipline as every other
    additive chat hook in this file's siblings)."""
    if not d6_hebbian_enabled():
        return None
    try:
        comp = store_composer(chat)
        if comp is None:
            return None
        diag = getattr(comp, "_d6_last_encode", None)
        if diag is None:
            return {"on": True, "wrote_this_turn": False}
        comp._d6_last_encode = None   # consume: a later turn with no new write reports False, never a stale encode
        out = {"on": True, "wrote_this_turn": True, "encode": dict(diag)}
        try:
            from research.runners.d6_hebbian_store import engram_held
            out["held"] = engram_held(comp, int(diag["block"]))
        except Exception as e:  # the encode diagnostic is still reported even if the held-read itself fails
            out["held"] = {"error": f"{type(e).__name__}: {e}"}
        return out
    except Exception as e:
        return {"on": True, "error": f"{type(e).__name__}: {e}"}
