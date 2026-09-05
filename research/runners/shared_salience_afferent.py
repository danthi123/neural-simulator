"""ONE SHARED SPIKING NOVELTY/SALIENCE AFFERENT wired onto the live turn (scaffold-retirement backlog rank-4,
research/coordination/scaffold_retirement_backlog.md; 2026-09-05), DEFAULT-OFF de-risk.

WHAT THIS IS. Two already-independently-de-risked halves that have never been wired to each other:
  * HALF A -- the DA-mode SNc engagement afferent: `webapp/da_mode_drives_chat.py::engagement_of()` computes a per-
    turn engagement/novelty scalar with BARE HOST ARITHMETIC (a fraction-of-unseen-tokens + a saturating token-count,
    `_W_NOVELTY*novelty + (1-_W_NOVELTY)*richness`) and feeds it DIRECTLY (as pA) into the spiking SNc nucleus
    (`_neuromod_spiking_da_mode_derisk`, #76/#79, 6/6-seed GO). That one host scalar is the ROOT of THREE default-on
    consumers, because all three read the SAME `chat._last_da_drives["da_level"]` the SNc read produces:
    da-mode-drives-response (the engagement suffix), da-gated-encoding (`webapp/da_encoding_drives_chat.py`'s write-
    magnitude gain), da-gated-curiosity (`webapp/da_curiosity_drives_chat.py`'s crave-threshold gain). Zero neurons
    mediate the message -> SNc-afferent step today.
  * HALF B -- the curiosity organ's spiking novelty transduction: `curiosity_production_organ.CuriosityProductionOrgan`
    already drives a REAL ASK-pool population from an arbitrary `current_novelty_signal` scalar via the `from_novelty`
    neuromodulator (DR-1, `_curiosity_seek_learn_onbridge_derisk.py`, on-bridge 6-seed CPU GO; corr(gap,spiking-want)
    = +0.996 <!--derived--> reproduced 2026-08-12) and reads the wanting straight off `cp_firing_states[ask]` (Hz) --
    a genuine, already-validated spiking transducer of "how much of a gap/salience signal is present". Until this
    module, that transduction was used ONLY for the binary abstain-triggered follow-up decision (NOVEL_SIGNAL=0.95 vs
    FAMILIAR_SIGNAL=0.0), never as a general-purpose salience source.

THE INTEGRATION (not a new mechanism). `curiosity_production_organ.CuriosityProductionOrgan.salience_of(raw, lesion)`
(added alongside this module) generalizes the SAME ASK-pool read to an ARBITRARY continuous raw scalar in [0,1] and
reports a NORMALIZED salience against the organ's own familiar<->novel calibration anchors. THIS module is the shared
de-risk glue: every consumer keeps computing its OWN raw host scalar (message-novelty for DA-mode, content-token-count
for bg-action-selection, fact-recency-ratio for value-choice -- each a legitimate host sensory/environment/memory-
provenance boundary, exactly as the SVO parser and the vision percept are), but instead of using that raw scalar
DIRECTLY, it is now passed THROUGH this shared spiking ASK-pool population before reaching its consumer -- a genuine
population of neurons (excitability drive -> membrane integration -> spiking -> firing-rate read) now mediates the
signal on its way from sensory read to consumer decision, retiring the "host arithmetic straight to the consumer"
shortcut at its single root (da_mode_drives_chat.engagement_of) and at the two sibling host formulas (bg-action-
selection's regex/token-count salience, value-choice's recency-ratio engagement context) in ONE move.

CONTRACT (additive, reversible, byte-identical-off). 2026-09-05 FLIPPED DEFAULT-ON (Track-1 flip campaign, after the
6-seed rank-4 wiring gate + the rank-20 real-critic 6-seed gate + a dedicated integrated flip-soak through the REAL
`ChatBrain`/production `/api/brain-chat` path -- see `research/findings/2026-09-05-shared-salience-value-choice-
flip-soak-6seed-GO.md`): `BRAIN_SHARED_SALIENCE` UNSET now means ACTIVE -- the shared afferent is armed at all three
consumer sites simultaneously by default. `BRAIN_SHARED_SALIENCE` explicitly set to `{0,false,no,off,''}` is the
byte-identical escape back to each consumer's pre-wiring host-arithmetic code path (this module's gate is then never
entered) -- the same escape-hatch idiom `BRAIN_VALUE_CHOICE`'s 2026-08-26 flip uses.

LESION (the load-bearing proof, reused verbatim from the curiosity organ's OWN anti-cheat). `BRAIN_SHARED_SALIENCE_
LESION=1` reads `CuriosityProductionOrgan`'s drive-removed twin (`curiosity_excit_sensitivity=0`, judge()'s own
lesion): the ASK pool's want COLLAPSES to its un-driven baseline REGARDLESS of the raw input, so `normalized` loses
its dependence on the raw scalar -- every consumer's decision then reverts to a baseline that no longer tracks the
raw signal, even though the raw host scalar itself still varies turn to turn. This is DISTINCT from each consumer's
OWN pre-existing lesion (e.g. `BRAIN_DA_DRIVES_LESION` silences the SNc nucleus itself; this lesion cuts only the
shared ASK-pool afferent feeding it).

REUSE-BY-IMPORT (NO `sim/` edit). This module + `salience_of` are the only new code; every neuron/synapse/pathway
already exists (`_curiosity_seek_learn_onbridge_derisk.build_curiosity_bridge`, 6-seed GO). `git diff sim/` is empty.

FUNCTIONAL CORRELATE, NOT phenomenal. This reads + reports a spiking salience/novelty CORRELATE; it claims no
subjective wanting.
"""
from __future__ import annotations

import os
from typing import Optional

from research.runners.curiosity_production_organ import CuriosityProductionOrgan, get_organ as _get_curiosity_organ

_DEFAULT_SEED = 42

# 2026-09-05 FLIPPED DEFAULT-ON (Track-1 flip campaign, rank-4/rank-5/rank-20 GOs; verification:
# research/findings/2026-09-05-shared-salience-value-choice-flip-soak-6seed-GO.md). Mirrors the
# `_VALUE_CHOICE_DEFAULT_ON` idiom (research/runners/value_choice_production_organ.py) so an explicit
# {0,false,no,off,''} stays the byte-identical escape back to the pre-flip host arithmetic at all 3 consumer
# sites, rather than removing that escape hatch.
_SHARED_SALIENCE_DEFAULT_ON = True


def _falsy_explicit(v) -> bool:
    return v is not None and str(v).strip().lower() in ("0", "false", "no", "off", "")


def shared_salience_enabled() -> bool:
    """The master flag. DEFAULT-ON (post 2026-09-05 flip): unset -> ACTIVE -> the shared spiking afferent is armed
    at every consumer site. `BRAIN_SHARED_SALIENCE` in {0,false,no,off,''} (explicitly set) is the byte-identical
    escape back to each consumer's pre-wiring host-arithmetic path (this module's read is then never called).
    {1,true,on,yes} (explicit) stays ACTIVE, redundant now."""
    v = os.environ.get("BRAIN_SHARED_SALIENCE")
    if _SHARED_SALIENCE_DEFAULT_ON:
        return not _falsy_explicit(v)
    return str(v or "0").strip().lower() in ("1", "true", "on", "yes")


def shared_salience_lesioned() -> bool:
    """`BRAIN_SHARED_SALIENCE_LESION` truthy -> read the curiosity organ's drive-removed twin (severs the shared
    afferent's dependence on the raw host scalar at every consumer site simultaneously)."""
    return os.environ.get("BRAIN_SHARED_SALIENCE_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def get_shared_organ(seed: int = _DEFAULT_SEED) -> CuriosityProductionOrgan:
    """The process-shared curiosity organ, reused AS THE shared salience afferent (the SAME singleton
    `curiosity_production_organ.get_organ()` already installs for the follow-up-question faculty -- one substrate,
    two consumers, not a second bridge)."""
    return _get_curiosity_organ(seed=seed)


def read_salience(raw: float, *, seed: int = _DEFAULT_SEED) -> dict:
    """THE shared read every consumer site calls: raw host scalar in [0,1] -> the curiosity organ's ASK-pool spiking
    transduction (`salience_of`) -> a normalized salience + the raw want_hz, for observability. Never raises (a
    wiring failure degrades to a neutral, input-INDEPENDENT 0.5 -- the same "never crash a turn" contract every
    sibling coupling in this codebase follows)."""
    try:
        organ = get_shared_organ(seed=seed)
        info = organ.salience_of(float(raw), lesion=shared_salience_lesioned())
        info["on"] = True
        return info
    except Exception as e:  # never let a wiring failure change/crash a consumer's turn
        return {"on": True, "raw": float(raw), "normalized": 0.5, "want_hz": 0.0,
                "lesioned": shared_salience_lesioned(), "error": f"{type(e).__name__}: {e}"}
