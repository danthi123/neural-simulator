"""VISUAL OBJECT -> CATEGORY IDENTITY ("spiking HMAX") wired into the production conversational turn (2026-08-26).

This is the production consumer for the EMERGE-36 fully-spiking perception->pooler->inference GO (6 seeds):
  * research/findings/2026-07-02-emerge36-spiking-perception-pipeline-GO.md
  * research/findings/2026-07-11-EMERGENT-fully-spiking-perception-codon-drives-the-ladder-6seed.md

A percept image (the ENVIRONMENT's retinal render) -> the real `sim.visual_cortex` Gabor/V1 front end -> a spiking
Marr-Albus coincidence-column pooler on a real `SimulationBridge` (`coincidence_weighted_drive`, NO numpy kWTA) ->
the winning self-organized category column block = the recognized-object identity (the taught codon->inheritance).
The recognized concept SEEDS the answer of a 'what do you see?' live turn ('I see a <recognized-object>. It can
<property>.'). Reuse-by-import of `_emerge36_spiking_perception_pipeline_derisk.SpikingPerceptionProbe` (which itself
composes EMERGE-34 Gabor/V1 + EMERGE-35 spiking codon + EMERGE-14 on-bridge inheritance). NO `sim/` edit; additive.

BRAIN-BASED-ONLY boundary (CLAUDE.md standing standard): host code is legitimate ONLY for the ENVIRONMENT — here,
rendering the retinal image the neural retina/V1 then receive. Everything between sensation and the recognized
identity — Gabor/V1, the coincidence-column pooler, the codon->property inheritance — is neurons/synapses on the
bridge (NO numpy kWTA anywhere). Surfacing the recognized category as an object noun is a fixed label map (the
environment/body naming), the same status as the finding's CATPROP property tag.

SCOPE (honest, do NOT overclaim — a checked separate NO-GO): the invariance demonstrated is over WELL-POSED
SYNTHETIC category sets (oriented-bar shape classes with within-category visual jitter). This is NOT natural-image
translation-invariance. The recognized identity is the taught visual category (6-seed GO), read as the winning
codon->inheritance, surfaced as an object noun.

Flag: `BRAIN_VISION_IDENTITY` (default OFF — the parent flips default-on after the pool soak passes). When OFF the
production turn is BYTE-IDENTICAL to today (the wiring block reads the flag first and imports nothing).

Lesion oracles (the finding's own controls, reused here):
  * POOLER-LESION (coincidence detection OFF): the codon never charges -> empty codon -> `recognize()` returns -1
    (ABSTAIN) for every percept -> the visual answer VANISHES and the turn reverts to the flag-off/host path. This
    is the DETERMINISTIC single-turn lesion (`BRAIN_VISION_IDENTITY_LESION=1`).
  * PER-IMAGE PIXEL-SCRAMBLE: within-category visual similarity destroyed -> within-category codon overlap collapses
    -> the identity readout collapses toward chance (the finding's headline control; used at the batch/soak level).
"""
from __future__ import annotations

import os
import re

import numpy as np

# reuse-by-import: the EMERGE-36 fully-spiking perception->pooler->inference organ (6-seed GO). Importing this
# module force-sets SIM_BACKEND=numpy VIA setdefault only — in the live cupy server SIM_BACKEND is already set, so
# the setdefault is a no-op and the recognizer builds on the process backend (cupy in prod, numpy in tests).
from research.runners._emerge36_spiking_perception_pipeline_derisk import SpikingPerceptionProbe

# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The recognized-category -> surface identity maps. Category 0/1 are the two well-posed visual shape classes the
# EMERGE-36 pipeline self-organizes over; CATEGORY_PROP mirrors the finding's taught CATPROP {0:'fly', 1:'swim'}
# (the property the codon->inheritance surfaces). CATEGORY_NOUN names the object consistent with that property.
# These are the environment/body naming layer (a fixed label map), NOT part of the neural recognition.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
CATEGORY_NOUN = {0: "bird", 1: "fish"}
CATEGORY_PROP = {0: "fly", 1: "swim"}
# percept descriptor tokens the environment may hand the brain, resolved to a visual category.
_NOUN_TO_CAT = {"bird": 0, "flyer": 0, "fish": 1, "swimmer": 1}


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Flags. UNLIKE the default-ON faculties in this directory, vision-identity ships default-OFF: the parent flips it
# default-on after the 6-seed pool soak passes. `BRAIN_VISION_IDENTITY` truthy -> the wiring is live.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def vision_identity_enabled() -> bool:
    """Default-OFF. `BRAIN_VISION_IDENTITY` in {1,true,yes,on} -> the visual-identity turn class is live."""
    v = os.environ.get("BRAIN_VISION_IDENTITY")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def vision_identity_lesioned() -> bool:
    """`BRAIN_VISION_IDENTITY_LESION` in {1,true,yes,on} -> build the POOLER-LESION recognizer (coincidence OFF):
    the codon never charges -> recognize() abstains on every percept -> the visual answer VANISHES (load-bearing)."""
    v = os.environ.get("BRAIN_VISION_IDENTITY_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


# a NARROW visual-query matcher (kept tight so it never hijacks a recall turn; the block also requires a percept).
_VISUAL_RE = re.compile(
    r"\b(what (do|can) you see|what('?s| is) (this|that)|what am i (looking at|showing you)|"
    r"describe what you see|what object is (this|that)|do you (see|recogn(i[sz]e)) (this|it))\b",
    re.IGNORECASE)


def is_visual_query(text: str) -> bool:
    """True iff the message is an explicit 'what do you see?'-class visual query. Narrow by design: the wiring block
    ALSO requires a non-empty percept, so a plain recall turn (no percept) is never captured even if it matched."""
    return bool(_VISUAL_RE.search(text or ""))


def resolve_percept(percept) -> tuple[int, int] | None:
    """Resolve the environment's percept descriptor to (category, which) — which = the held-out exemplar index.

    Accepts: an object noun ('bird'/'fish'/'flyer'/'swimmer'), a bare category id ('0'/'1' or int), optionally with
    a '#<which>' exemplar suffix (e.g. 'bird#2', '1#0'). Returns None if unresolvable (-> the block falls through)."""
    if percept is None:
        return None
    s = str(percept).strip().lower()
    if not s:
        return None
    which = 0
    if "#" in s:
        s, _, w = s.partition("#")
        s = s.strip()
        try:
            which = max(0, int(w.strip()))
        except ValueError:
            which = 0
    if s in _NOUN_TO_CAT:
        return _NOUN_TO_CAT[s], which
    try:
        c = int(s)
    except ValueError:
        return None
    if c in CATEGORY_NOUN:
        return c, which
    return None


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# THE SPIKING RECOGNIZER — a trained EMERGE-36 pipeline; recognize a held-out perceived object's category identity.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class VisionIdentityRecognizer:
    """Wraps a trained `SpikingPerceptionProbe` (EMERGE-36). Built ONCE (lazily). `recognize(category, which)`
    presents a HELD-OUT perceived object of that visual category (never seen in training) through the trained
    codon->inheritance and returns the recognized category (-1 = ABSTAIN). `lesion=True` builds the POOLER-LESION
    variant (coincidence OFF -> empty codon -> abstains on every percept); `scramble=True` builds the per-image
    pixel-scramble variant (within-category similarity destroyed -> codon overlap collapses)."""

    def __init__(self, seed: int = 42, epochs: int = 40, lesion: bool = False, scramble: bool = False):
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.lesion = bool(lesion)
        self.scramble = bool(scramble)
        self._built = False
        self.probe = None

    def ensure_built(self):
        if self._built:
            return
        self.probe = SpikingPerceptionProbe(
            seed=self.seed, epochs=self.epochs, lesion=self.lesion, scramble=self.scramble)
        self._built = True

    def held_which(self, category: int) -> list:
        """The held-out exemplar indices available for a visual category (for varying `which`)."""
        self.ensure_built()
        return list(range(len(self.probe.held.get(int(category), []))))

    def recognize(self, category: int, which: int = 0) -> int:
        """Present held-out perceived object `which` of visual `category` -> recognized category (-1 = abstain)."""
        self.ensure_built()
        held = self.probe.held.get(int(category))
        if not held:
            return -1
        idx = held[int(which) % len(held)]
        return int(self.probe.infer(self.probe.OF[idx]))

    def codon(self, category: int, which: int = 0) -> set:
        """The winning self-organized column block (the recognized-object identity codon) for held-out object
        `which` of `category`. Used to measure within-category codon overlap (the scramble lesion metric)."""
        self.ensure_built()
        held = self.probe.held.get(int(category))
        if not held:
            return set()
        idx = held[int(which) % len(held)]
        return set(self.probe._codon(self.probe.OF[idx]))


# process-shared recognizer cache, keyed by (seed, lesion, scramble) so the intact + lesion + scramble variants
# coexist (the load-bearing verify + the soak build all three).
_ORGANS: dict = {}


def get_organ(seed: int = 42, lesion: bool | None = None, scramble: bool = False) -> VisionIdentityRecognizer:
    """The process-shared recognizer for (seed, lesion, scramble). `lesion=None` -> read the env lesion flag (so the
    live handler builds the POOLER-LESION recognizer when `BRAIN_VISION_IDENTITY_LESION=1`)."""
    les = vision_identity_lesioned() if lesion is None else bool(lesion)
    key = (int(seed), bool(les), bool(scramble))
    org = _ORGANS.get(key)
    if org is None:
        org = VisionIdentityRecognizer(seed=seed, lesion=les, scramble=scramble)
        _ORGANS[key] = org
    return org


def answer_percept(percept, seed: int = 42, lesion: bool | None = None) -> dict | None:
    """Recognize the environment's percept and produce the seeded answer for a 'what do you see?' turn, or None.

    Returns None when: the percept is unresolvable, OR the recognizer ABSTAINS (the pooler-lesion collapse, or an
    object the codon cannot charge) — in which case the live handler falls through to the normal/host path, so the
    lesioned/unrecognized turn is BYTE-IDENTICAL to the flag-off behavior (the load-bearing lesion-vanish).

    On success returns:
      {'answer': 'I see a bird. It can fly.', 'category': 0, 'noun': 'bird', 'prop': 'fly',
       'shown_category': 0, 'which': 0, 'recognized': True}
    The `category` is the SPIKING recognition (not the shown label) — under lesion/scramble it can differ from
    `shown_category` or collapse to abstain, which is exactly why the answer is a read of the substrate, not an echo.
    """
    r = resolve_percept(percept)
    if r is None:
        return None
    shown_cat, which = r
    org = get_organ(seed=seed, lesion=lesion)
    cat = org.recognize(shown_cat, which)
    if cat < 0 or cat not in CATEGORY_NOUN:
        return None                                  # ABSTAIN -> the handler reverts to the host/flag-off path
    noun = CATEGORY_NOUN[cat]
    prop = CATEGORY_PROP[cat]
    return {"answer": f"I see a {noun}. It can {prop}.", "category": int(cat), "noun": noun, "prop": prop,
            "shown_category": int(shown_cat), "which": int(which), "recognized": True}
