"""DA/ENGAGEMENT-GATED CURIOSITY crave-threshold wired into the LIVE chat turn (board WAVE-0, Gap-4 coupling (b)),
default-OFF. Make the brain MORE curious when it is ENGAGED.

WHAT THIS IS. A coupling of two EXISTING spiking faculties — it re-invents NEITHER.
  * CURIOSITY (`research/runners/curiosity_production_organ.py`): a genuinely-SPIKING ASK pool driven by the
    `from_novelty` neuromodulator (the DR-1 crave-drive, corr(gap,want)=+0.996). On a NOVEL-topic ABSTAIN the brain
    CRAVES a follow-up; the DECISION is a THRESHOLD on the spiking ASK-pool rate (`want >= self.threshold`, the
    threshold CALIBRATED at build from a novel-vs-familiar battery). When it craves, `webapp/server.py::brain_chat`
    APPENDS an honest FOLLOW-UP QUESTION ("crave, don't refuse") — the moat is INVERTED, never broken.
  * DA / ENGAGEMENT (`webapp/da_mode_drives_chat.py::observe_turn`): each turn the brain SELF-PRODUCES its tonic
    dopamine off the spiking SNc nucleus and stashes the level on `chat._last_da_drives["da_level"]` (tonic ~= 0.5;
    higher = more engaged / aroused). That read runs in `brain_chat` BEFORE the curiosity block, so the level is
    FRESH there.

THE COUPLING (biologically grounded: DA / arousal raises exploratory drive — Aston-Jones & Cohen LC-NE adaptive-gain,
Costa-Averbeck striatal-DA exploration bias, the tonic-DA "vigor/incentive" account). When the brain is ENGAGED
(DA > tonic) it should be MORE curious: the effective crave-threshold DROPS so the ASK pool crosses it more readily
(it asks a follow-up on a topic it would otherwise let pass). When disengaged (DA < tonic) the threshold RISES (it
lets a marginal topic pass). At tonic the coupling is a no-op (gain 1.0 -> the organ's calibrated decision, unchanged).

MECHANISM (the form the organ's decision cleanly supports). The organ decides `want >= threshold`. This scales the
ASK-pool WANT by a small DA gain — engagement AMPLIFIES the crave signal — then compares against the organ's OWN
calibrated threshold:
    da_crave_gain(DA) = clip(g_min, g_max, 1 + K_DA*(DA - tonic))        [tonic 0.5 -> gain 1.0]
    curious          = (want_hz * da_crave_gain) >= threshold
Equivalently the EFFECTIVE crave-threshold is `threshold / da_crave_gain` (reported in the trace so the read speaks
the "lower/raise the crave-threshold" language): DA > tonic -> gain > 1 -> a LOWER effective threshold (more curious);
DA < tonic -> gain < 1 -> a HIGHER effective threshold (less curious). K_DA is a small host constant.

MOAT-SAFE + ADDITIVE (preserved BY CONSTRUCTION). This NEVER manufactures a fact, flips an abstain into an assert, or
enters the certainty band. It runs ONLY inside the curiosity block (which itself runs ONLY on an ABSTAIN — there is no
answer to corrupt) and it changes ONLY WHETHER the honest follow-up QUESTION is appended. The content fields
(`abstained`, `recalled_svo`, `verified`) are byte-identical with the coupling on or off; only the optional follow-up
suffix and the additive `curiosity_da` trace change.

CONTRACT (additive, reversible, byte-identical-off).
  * `da_curiosity_enabled()` gates the whole coupling. When DISABLED the server skips it entirely: the curiosity
    organ's CALIBRATED threshold decides `curious` unchanged, and NO `curiosity_da` key is attached -> the turn is
    BYTE-IDENTICAL to pre-wiring (`BRAIN_CURIOSITY_DA` unset -> off).
  * When ENABLED, `crave_decision(chat, want_hz, base_threshold)` reads the live self-produced DA, computes the DA
    crave-gain, and returns the modulated `curious` + a trace. At tonic (gain 1.0) and on a missing DA read the
    decision is identical to the organ's own -> only a departure from tonic changes it.
  * LESION (`BRAIN_CURIOSITY_DA_LESION=1`): the DA modulation is PINNED to 0 (gain 1.0) regardless of the DA level ->
    the coupling is severed (the crave decision no longer rides the DA read even though the DA level still varies).
    DISTINCT from `BRAIN_CURIOSITY_LESION` (which removes the curiosity DRIVE pathway -> collapses the WANT itself)
    and from `BRAIN_DA_DRIVES_LESION` (which silences the SNc nucleus -> collapses the LEVEL). This lesion cuts only
    the DA->crave-threshold link.

REUSE-BY-IMPORT (NO `sim/` edit). Reads the live DA off the chat scaffold and returns a host decision; it touches
neither the curiosity organ's spiking ASK-pool read nor the SNc->DA read. `git diff sim/` is empty. Mirrors the
sibling write-side coupling `webapp/da_encoding_drives_chat.py` (board WAVE-0 Gap-4, committed d5c67f7c).

FUNCTIONAL CORRELATE, NOT phenomenal. This makes + reports an engagement->curiosity CORRELATE; it claims no
subjective wanting.
"""
from __future__ import annotations

import os
from typing import Tuple

# the live SNc-self-produced tonic DA baseline (webapp/da_mode_drives_chat: tonic ~= 0.5). gain == 1.0 here (neutral).
_DA_TONIC_BASELINE = 0.5
# the DA -> crave-gain slope (small dimensionless host constant). gain = clip(g_min, g_max, 1 + K_DA*(DA - tonic)).
# Calibrated against the curiosity organ's own battery (want_novel ~= 126.6 Hz, threshold ~= 65.9 Hz on seed 42): at a
# high-DA turn (DA ~= 1.24) gain ~= 2.1 -> want_eff ~= 267 Hz (crosses the 66 Hz threshold -> follow-up); at a low-DA
# turn (DA ~= 0.05) gain ~= 0.32 -> want_eff ~= 40 Hz (below threshold -> the follow-up is let pass). K small; the flip
# has a >=25 Hz margin on both sides (robust to the ASK-pool's ~3 Hz OU jitter).
_K_DA = 1.5
_G_MIN = 0.2
_G_MAX = 3.0


def da_curiosity_enabled() -> bool:
    """The master flag. `BRAIN_CURIOSITY_DA` truthy (1/true/on/yes) enables the DA-gated curiosity crave-threshold;
    anything else (the default UNSET) leaves it OFF -> the organ's calibrated threshold decides `curious` unchanged
    and no `curiosity_da` key is attached -> byte-identical to HEAD."""
    return os.environ.get("BRAIN_CURIOSITY_DA", "0").strip().lower() in ("1", "true", "on", "yes")


def da_curiosity_lesioned() -> bool:
    """`BRAIN_CURIOSITY_DA_LESION` truthy -> pin the DA modulation to 0 (crave-gain 1.0) regardless of the DA level
    (sever the DA->crave-threshold link). The load-bearing proof: the crave decision stops riding the DA read even
    though the DA level still varies. DISTINCT from BRAIN_CURIOSITY_LESION (collapses the WANT) and
    BRAIN_DA_DRIVES_LESION (collapses the LEVEL)."""
    return os.environ.get("BRAIN_CURIOSITY_DA_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def da_level_of(chat) -> float:
    """The live self-produced tonic DA off the DA-mode read (`chat._last_da_drives["da_level"]`, set by
    da_mode_drives_chat.observe_turn earlier in the turn). Missing (da-drives off / not yet observed) -> tonic 0.5 ->
    gain 1.0 (neutral). Never raises."""
    info = getattr(chat, "_last_da_drives", None)
    if not isinstance(info, dict):
        return _DA_TONIC_BASELINE
    try:
        lvl = info.get("da_level", _DA_TONIC_BASELINE)
        return float(lvl) if lvl is not None else _DA_TONIC_BASELINE
    except (TypeError, ValueError):
        return _DA_TONIC_BASELINE


def da_crave_gain(da: float) -> float:
    """gain = clip(g_min, g_max, 1 + K_DA*(DA - tonic)). tonic 0.5 -> gain 1.0 (the no-modulation knob); DA > tonic ->
    gain > 1 (more curious -> a lower effective threshold); DA < tonic -> gain < 1 (less curious)."""
    return float(min(_G_MAX, max(_G_MIN, 1.0 + _K_DA * (float(da) - _DA_TONIC_BASELINE))))


def crave_decision(chat, want_hz: float, base_threshold: float) -> Tuple[bool, dict]:
    """The DA-modulated crave decision. Reads the live self-produced DA, scales the ASK-pool WANT by the DA crave-gain
    (engagement amplifies the crave), and compares against the organ's OWN calibrated threshold. Returns
    (curious: bool, trace: dict). Under the lesion the gain is pinned to 1.0 (the coupling severed) -> the decision is
    the organ's own (== byte-identical to off), so the high-vs-low DA difference VANISHES. At tonic / a missing DA read
    the gain is 1.0 -> the organ's own decision. Never raises."""
    lesioned = da_curiosity_lesioned()
    da = da_level_of(chat)
    gain = 1.0 if lesioned else da_crave_gain(da)
    want_eff = float(want_hz) * gain
    # the equivalent EFFECTIVE crave-threshold (for the trace): DA-engaged -> lower; disengaged -> higher.
    eff_threshold = float(base_threshold) / gain if gain > 0 else float(base_threshold)
    curious = bool(want_eff >= float(base_threshold))
    trace = {
        "on": True, "lesioned": bool(lesioned), "da_level": float(da), "da_tonic": _DA_TONIC_BASELINE,
        "k_da": _K_DA, "da_crave_gain": float(gain), "base_threshold": float(base_threshold),
        "eff_threshold": float(eff_threshold), "want_hz": float(want_hz), "want_eff_hz": float(want_eff),
        "curious": curious,
    }
    return curious, trace
