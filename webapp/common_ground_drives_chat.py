"""The COMMON-GROUND LEDGER wired into the LIVE `/api/brain-chat` turn so the served reply's REFERRING EXPRESSION is
LOAD-BEARING on the conversation's accumulated common ground -- NOT observe-only (INTEGRATION-TO-PRODUCTION, DEFAULT-OFF).

WHAT THIS IS. The de-risk `research/runners/_learned_common_ground_ledger_derisk.py` (6-seed pool run; 1-seed smoke:
audience-design acc 1.000 vs chance 0.5, permute -> 0.500, lesion -> 0.500 static, substrate-read grounded 0.113 >>
ungrounded 0.000) built + validated a spiking common-ground ledger (K bistable NMDA-attractor referent stores) read by
a Namburi-Tye biased-competition reduce/introduce decision. `common_ground_ledger_production_organ` made that same
substrate PERSISTENT PER SESSION so a live conversation's common ground ACCUMULATES. This module makes that neural
audience-design verdict CHANGE the surface: when the turn's referent is ALREADY in common ground (mentioned earlier
this conversation, its NMDA store latched + self-sustained), the substrate read wins REDUCE and the reply LEADS with a
reduced / pronominal reference ("As for it -- <answer>"); a NOT-yet-grounded referent reads UNGROUNDED -> the novelty
prior wins INTRODUCE -> NO reduced lead (the reply names it in full, the natural first-mention move). The referring
expression the reply leads with therefore FOLLOWS the actual grounding history -- audience design (Clark & Brennan 1991
grounding; Clark & Marshall 1981 common ground; Duff & Brown-Schmidt 2012 hippocampal/declarative).

THE COUPLING (what makes it LOAD-BEARING, not observe-only). The neural decision -> a REDUCED-REFERENCE lead:
`decision == "reduce"` (a grounded referent) -> `lead = "As for it — "`; `decision == "introduce"` (a first mention /
un-established referent) -> NO lead. The lead is an honest EXPRESSION of the audience-design choice the substrate just
made (a discourse "mouth", like #84's affect markers / #85's swap transition), NOT content: the FACT after it is the
SAME gate-matched, moat-verified answer and the VERIFY re-parse is unchanged. So the ledger changes HOW the reply
refers (reduced vs full), never WHICH fact is true and never whether an unmatched cue abstains. This is the single
coupling this module wires. Reduce-only (mirrors the swap-only #85 pattern) so the neural lesion produces a CLEAN
byte-identical vanish.

THE HONESTY FLOOR (preserved BY CONSTRUCTION). The moat / recall / abstain verdict runs FIRST and unchanged; the
coupling only DECORATES an already-matched answer surface with a reduced-reference lead. It never enters the certainty
band, never manufactures a fact, never flips an abstain into an assert. The content fields (`abstained`,
`recalled_svo`, `verified`) are BYTE-IDENTICAL with the coupling on or off; only the answer SURFACE (the optional lead)
and the additive `common_ground_drives` trace change.

LESION (the load-bearing / brain-based proof). `BRAIN_CG_DRIVES_LESION=1` builds the ledger's referent self-loops at
weight 0 (the de-risk's own recurrence lesion) -> the ledger cannot HOLD a grounded bit -> even a re-mentioned referent
reads UNGROUNDED at speak-time -> the decision goes STATIC (always INTRODUCE) -> the reduced-reference lead VANISHES and
the surface reverts to the byte-identical no-lead answer. So the surface change RIDES the SPIKING ledger read, not a
host `if word in seen_set`: silence the ledger's persistence and the reduced reference disappears even though the world
input (a re-mentioned referent) is unchanged.

CONTRACT (additive, reversible, byte-identical-off).
  * `cg_drives_enabled()` (combined with the server anchor) gates the whole block. When DISABLED the handler skips it:
    no ledger is built, no read runs, no `common_ground_drives` key is attached, and NO lead is prepended -> the turn
    is BYTE-IDENTICAL to pre-wiring.
  * The ledger runs its substrate steps inside a numpy global-RNG save/restore (in the organ), so enabling this module
    cannot perturb the downstream RNG-dependent organs; the other response fields stay byte-identical.
  * The ledger build (~2s) is lazy on the first grounded-referent turn per session and kept warm; each subsequent
    referent turn runs one ~0.1s read + grounding act.

REUSE-BY-IMPORT (NO `sim/` edit). The ledger substrate + the ignite/hold/query primitives come from the de-risk via
`common_ground_ledger_production_organ`; the grounded-referent (topic) extractor comes from `webapp/gnw_thought_swap`
(the same host language-comprehension boundary the SVO parser / the swap-topic extractor occupy). `git diff sim/` is
empty. This module adds only the production glue (the decision -> reduced-reference-lead map).

HONEST RESIDUALS (named, not claimed closed).
  1. The message->REFERENT extraction is host (a language-comprehension boundary, like the SVO parser). The audience
     DECISION (reduce vs introduce) and the grounded-vs-ungrounded read ARE the spiking ledger (lesion-proven).
  2. The decision->LEAD-STRING map is a HOST conditioned-articulation scaffold (the discourse "mouth"); the ledger that
     DRIVES it is neural + load-bearing (the lesion collapses the lead). A brain-native referring-expression mouth is
     the named next rung.
  3. LEARNED conceptual pacts / lexical entrainment / partner-specificity are named follow-ons; this wires the
     given/new (in-common-ground vs new) audience-design axis only.
"""
from __future__ import annotations

import os
from typing import Optional

# reuse-by-import: the persistent per-session spiking ledger organ (wraps the 6-seed-GO de-risk) + the grounded-topic
# extractor (the same comprehension boundary the swap path uses). NO sim/ edit.
from research.runners import common_ground_ledger_production_organ as _CGL
from webapp import gnw_thought_swap as _GTS

_DEFAULT_SEED = 42


def cg_drives_enabled() -> bool:
    """The master flag. `BRAIN_CG_DRIVES` truthy (1/true/on/yes) enables; 0/false/off/no disables. The default when the
    env var is UNSET follows the production-integration anchor `_CG_DRIVES_DEFAULT_ON` in server.py -- this reads only
    the explicit env override (server.py combines it with the anchor, mirroring the swap/affect drive flags)."""
    return os.environ.get("BRAIN_CG_DRIVES", "0").strip().lower() in ("1", "true", "on", "yes")


def cg_drives_off() -> bool:
    """Explicit OFF (for a default-ON anchor): `BRAIN_CG_DRIVES` in {0,false,no,off,''}."""
    v = os.environ.get("BRAIN_CG_DRIVES")
    return v is not None and v.strip().lower() in ("0", "false", "no", "off", "")


def cg_drives_lesioned() -> bool:
    """`BRAIN_CG_DRIVES_LESION` truthy -> build the ledger recurrence at weight 0 (silence the ledger's hold): a
    re-mentioned referent can no longer read grounded, so the reduced-reference lead VANISHES. The load-bearing proof."""
    return os.environ.get("BRAIN_CG_DRIVES_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def audience_design_lead(decision: Optional[str], topic: Optional[str]) -> str:
    """The referring-expression EXPRESSION for this turn (the conditioned-articulation scaffold; DRIVEN by the neural
    audience-design verdict). A grounded referent (`decision == "reduce"`) -> a reduced / pronominal lead; a first
    mention / un-established referent (`decision == "introduce"`, or no decision) -> '' so the surface is byte-identical
    (the reply names it in full -- the natural first-mention move). The FACT after it is unchanged (VERIFY re-parse
    intact) -- this frames HOW the reply refers, never WHICH fact is true."""
    if decision != "reduce":
        return ""
    return "As for it — "


def observe_turn(chat, message: str, *, cache_key=None, seed: int = _DEFAULT_SEED) -> dict:
    """The production entry point: extract this turn's grounded referent, run ONE audience-design decision on the
    per-session persistent ledger (threading the lesion), then map the decision to a reduced-reference lead. Returns
    the per-turn `common_ground_drives` info (also stashed on `chat._last_cg_drives`). Never raises out (on any error
    it returns an inert no-lead info dict so a turn can never crash)."""
    try:
        composer = getattr(getattr(chat, "inner", None), "composer", None)
        topic = _GTS._extract_topic(message, composer)
        key = cache_key if cache_key is not None else id(chat)
        organ = _CGL.get_organ(key, seed=seed, lesion=cg_drives_lesioned())
        info = organ.observe_turn(topic)
        decision = info.get("decision")
        lead = audience_design_lead(decision, topic)
        out = dict(info)
        out.update({
            "on": True,
            "lead": lead,
            "lesioned": cg_drives_lesioned(),
            "reason_lead": ("reduced_reference" if lead else
                            ("lesion_collapsed" if cg_drives_lesioned() and info.get("in_common_ground") else
                             (decision or "no_referent"))),
        })
    except Exception as e:   # never let the coupling crash / change a turn -> inert no-lead info
        out = {"on": True, "error": f"{type(e).__name__}: {e}", "lead": "", "decision": None}
    try:
        chat._last_cg_drives = out
    except Exception:
        pass
    return out
