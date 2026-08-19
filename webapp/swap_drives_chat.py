"""The #77 GNW NEURAL THOUGHT-SWAP wired into the LIVE `/api/brain-chat` turn so the held-topic coalition is
LOAD-BEARING on what the brain SAYS -- NOT observe-only (board #85, INTEGRATION-TO-PRODUCTION).

WHAT THIS IS. Board #77 wired the GNW thought-swap onto the live chat as an OBSERVER: each turn it runs the reused
6/6-seed-GO neural swap machinery (a spiking mismatch/salience detector fires for a salient TOPIC-CHANGE proposal
that does NOT match the held coalition -> recurrence-depression evicts the incumbent -> the neural vacancy gate
admits the newcomer) and stashes a swap-vs-hold VERDICT + the held-topic coalition as response METADATA -- but it
NEVER changed the answer text (verified: topic-change swap-rate 1.00, `2026-08-19-gnw-swap-into-chat-GO.md`). An
observe-only wiring is hollow. This module makes that neural verdict CHANGE the surface: on a SWAP (a genuine topic
change) the reply LEADS with a topic-transition acknowledgment naming the newly-held coalition ("On <newtopic>,
then -- <answer>"); on a HOLD (a same-topic follow-up) it stays anchored with NO transition (the natural discourse
move -- you mark topic SHIFTS, not continuations). It is the anti-hollow-integration counterpart to the observe-only
faculty: the swap READ is neural AND it demonstrably shapes what gets said. Mirrors the board-#84 affect-DRIVES path.

THE READ (the #77 neural mechanism, reused-by-import; NO sim/ edit).
  * Each turn the user message's grounded TOPIC (the FIRST known agent/patient token -- the SAME host-comprehension
    boundary the SVO question parser occupies) is presented to the per-session held-topic swap workspace
    (`gnw_thought_swap.observe_turn`, board #77). The reused `run_intention_swap` drives that proposal into the
    spiking mismatch/salience detector + the vacancy gate: a DIFFERENT salient topic is a mismatch -> mm fires ->
    the STD boost drains the incumbent's recurrent loop below the sustain knee -> it self-evicts -> the vacancy gate
    ignites the newcomer (a SWAP); the SAME topic MATCHES -> the pred interneuron vetoes mm -> no boost -> the held
    coalition persists (NO swap). The swap-vs-hold VERDICT (`swapped`) and the identity of the winning coalition
    (`held_topic` after the decision = the neural winner slot's topic) are the substrate's, NEVER a host `if new != old`.

THE COUPLING (what makes it LOAD-BEARING, not observe-only).
  * The neural SWAP verdict -> a TOPIC-TRANSITION LEAD prepended to the answer surface. `swapped == True` ->
    `lead = "On <new_topic>, then -- "` (the transition acknowledgment; `<new_topic>` is READ from the neural winner
    coalition, not the raw message). `swapped == False` (a same-topic hold / anaphoric follow-up / first thought /
    an unmatched no-swap) -> NO lead (continuation needs no announcement). The lead is an honest EXPRESSION of the
    thought-swap the substrate just performed (a discourse-structuring "mouth", like #84's affect markers), NOT
    content: the FACT after it is the SAME gate-matched, moat-verified answer, and the VERIFY re-parse is unchanged.
    So the swap changes HOW the reply is framed (does it announce a topic shift?), never WHICH fact is true and
    never whether an unmatched cue abstains. This is the single coupling this module wires.

THE HONESTY FLOOR (preserved BY CONSTRUCTION, mirrors the #84 affect path).
  * The moat / recall / abstain verdict runs FIRST and unchanged; the swap coupling only DECORATES an
    already-matched answer surface with a transition lead. It never enters the certainty band, never manufactures a
    fact, never flips an abstain into an assert. The content fields (`abstained`, `recalled_svo`, `verified`) are
    BYTE-IDENTICAL with the coupling on or off; only the answer SURFACE (the optional lead) and the additive
    `swap_drives` trace change.

LESION (the load-bearing / brain-based proof). `BRAIN_SWAP_DRIVES_LESION=1` threads `trigger_lesion=True` into the
reused `run_intention_swap` -> the spiking mismatch/salience detector is given NO proposal drive (mm never fires ->
the STD boost stays 0), so a salient TOPIC-CHANGE can NO LONGER trigger a swap (`swapped` collapses to False) -> the
transition lead VANISHES and the surface reverts to the byte-identical no-lead (coupling-off) answer. So the surface
change RIDES the SPIKING mismatch read, not a host `if topic_changed`: silence the neural detector and the
topic-transition acknowledgment disappears even though the world input (a topic change) is unchanged. This is the
de-risk's OWN neural lesion, reused (not a host coupling cut).

CONTRACT (additive, reversible, byte-identical-off).
  * `swap_drives_enabled()` gates the whole block. When DISABLED the handler skips it: no workspace is built, no read
    runs, no `swap_drives` key is attached, and NO transition lead is prepended -> the turn is BYTE-IDENTICAL to
    pre-wiring. (The #77 observer path, if BRAIN_GNW_SWAP is separately on, still attaches its own `gnw_swap` key.)
  * The swap workspace runs on its PRIVATE RNG timeline and the host process-global RNG (numpy + the sim backend) is
    restored around every read (the #77 global-RNG footgun, inherited): enabling this module cannot perturb the
    downstream RNG-dependent organs, so the OTHER response fields stay byte-identical.
  * The workspace build (~0.8s) is lazy on the first grounded-topic turn per session and kept warm; each subsequent
    topic turn runs one ~0.3s swap decision.

REUSE-BY-IMPORT (NO `sim/` edit). The neural swap machinery, the held-topic register and the grounded-topic
extractor come STRAIGHT from `webapp/gnw_thought_swap.py` (board #77), which itself reuses the 6/6-seed-GO
`_gnw_neural_swap_intention_derisk`. This module adds only the production glue (the swap-verdict -> transition-lead
map). `git diff sim/` is empty. The lesion affordance is the de-risk's own `trigger_lesion`, threaded through #77's
`observe_turn(..., lesion=...)` (an additive kwarg, default False -> the #77 observer is byte-identical).

HONEST RESIDUALS (named, not claimed closed).
  1. The message->TOPIC extraction is host (a language-comprehension boundary, like the SVO parser). The swap DECISION
     (whether the topic change fires a swap) and the winning coalition's identity ARE the #77 neural mechanism
     (lesion-proven -- silence mm and the swap, hence the lead, collapses).
  2. The verdict->TRANSITION-STRING map is a HOST conditioned-articulation scaffold (the discourse "mouth"): the swap
     that DRIVES it is the neural mismatch/eviction/admit chain (load-bearing -- the lesion collapses the lead), but
     the surface STRING for a swap is a host template, exactly the sanctioned articulation-crutch pattern (owner:
     scaffold-ok-as-conditioned-articulation IF the faculty is load-bearing on the surface, which the lesion proves).
     A brain-native discourse-transition mouth (the marker emitted by a spiking sequencing circuit) is the named next rung.
  3. The coupling is SWAP-only (a topic change announces the shift; a hold stays silent -- the natural discourse move).
     A bidirectional continuity lead ("Still on <heldtopic> -- " on holds) that makes the persistent coalition visible
     on EVERY follow-up is a possible enrichment; it was kept swap-only so the neural lesion produces a CLEAN
     byte-identical vanish (a hold-lead would survive the mm-silencing lesion and muddy the anti-hollow check).
  4. Cross-turn CONTINUITY of the held coalition is a host label re-ignited on the substrate each turn (inherited #77
     residual): the swap-vs-hold VERDICT is neural every turn; a truly continuous cross-turn ignition is a named rung.
"""
from __future__ import annotations

import os
from typing import Optional

# reuse-by-import the board-#77 live held-topic swap workspace (the neural swap decision + the grounded-topic
# register) -- which itself reuses the 6/6-seed-GO `_gnw_neural_swap_intention_derisk`. NO sim/ edit.
from webapp import gnw_thought_swap as _GTS

_DEFAULT_SEED = 42


def swap_drives_enabled() -> bool:
    """The master flag. `BRAIN_SWAP_DRIVES` truthy (1/true/on/yes) enables; 0/false/off/no disables. The default when
    the env var is UNSET follows the production-integration anchor `_SWAP_DRIVES_DEFAULT_ON` in server.py -- this
    reads only the explicit env override (server.py combines it with the anchor, mirroring the #84 affect flags)."""
    return os.environ.get("BRAIN_SWAP_DRIVES", "0").strip().lower() in ("1", "true", "on", "yes")


def swap_drives_off() -> bool:
    """Explicit OFF (for a default-ON anchor): `BRAIN_SWAP_DRIVES` in {0,false,no,off,''}."""
    v = os.environ.get("BRAIN_SWAP_DRIVES")
    return v is not None and v.strip().lower() in ("0", "false", "no", "off", "")


def swap_drives_lesioned() -> bool:
    """`BRAIN_SWAP_DRIVES_LESION` truthy -> thread `trigger_lesion=True` into the neural swap (silence the mismatch
    detector): a topic change can no longer swap, so the transition lead VANISHES. The load-bearing proof."""
    return os.environ.get("BRAIN_SWAP_DRIVES_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def transition_lead(swapped: bool, new_topic: Optional[str]) -> str:
    """The topic-transition EXPRESSION for this turn (the conditioned-articulation scaffold; DRIVEN by the neural swap
    verdict). A neural SWAP with a named winning coalition -> a transition lead naming the new topic; a HOLD (or any
    no-swap / no-topic turn) -> '' so the surface is byte-identical. The FACT after it is unchanged (VERIFY re-parse
    intact) -- this frames WHETHER the reply announces a topic shift, never WHICH fact is true."""
    if not swapped:
        return ""
    t = (new_topic or "").strip()
    if not t:
        return ""
    return "On %s, then — " % t


def observe_turn(chat, message: str, *, seed: int = _DEFAULT_SEED) -> dict:
    """The production entry point: run ONE neural swap decision for this turn's grounded topic on the per-session
    held-topic workspace (reuse-by-import from #77, threading the lesion), then map the swap VERDICT to a
    topic-transition lead. Returns the per-turn `swap_drives` info (also stashed on `chat._last_swap_drives`; the
    underlying #77 `gnw_swap` info is stashed on `chat._last_gnw_swap` by the reused call). Never raises out (on any
    error it returns an inert no-lead info dict so a turn can never crash)."""
    try:
        info = _GTS.observe_turn(chat, message, seed=seed, lesion=swap_drives_lesioned())
        swapped = bool(info.get("swapped"))
        # the new topic is the neural winner coalition's topic (== held_topic AFTER a swap), not the raw message token.
        new_topic = info.get("held_topic") if swapped else None
        lead = transition_lead(swapped, new_topic)
        out = dict(info)
        out.update({
            "on": True,
            "lead": lead,
            "swapped": swapped,
            "new_topic": new_topic,
            "lesioned": swap_drives_lesioned(),
            "reason_lead": ("topic_transition" if lead else
                            ("lesion_collapsed" if swap_drives_lesioned() and info.get("reason") not in
                             ("first_thought", "same_topic_hold", "no_topic_hold") else
                             (info.get("reason") or "no_swap"))),
        })
    except Exception as e:   # never let the swap coupling crash / change a turn -> inert no-lead info
        out = {"on": True, "error": f"{type(e).__name__}: {e}", "lead": "", "swapped": False}
    chat._last_swap_drives = out
    return out
