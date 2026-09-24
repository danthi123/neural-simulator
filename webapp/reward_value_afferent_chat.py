"""A10 (midnight plan S15c, 2026-09-24) -- drive the SNc reward/context afferent that `da_mode_drives_chat`
folds into its engagement EMA from an EXISTING SPIKING READ, instead of the host `engagement_of()`
novelty+richness scalar `da_mode_drives_chat.py` names as its own HONEST RESIDUAL #1 (module docstring, line 89):
"The message->ENGAGEMENT scalar ... is host (a language/sensory-comprehension boundary) ... computing it from the
brain's own sensory stream is a SEPARATE faculty (the named next rung)." This module IS that next rung, scoped to
the two spiking reads already wired into production and already declared load-bearing on their OWN path:

  1. PRIMARY -- the surprise organ's confirm/violate verdict (`research/runners/surprise_production_organ.py`,
     the SAME 6/6-GO D2 predictive-coding mismatch circuit `webapp/server.py`'s own surprise block reads off
     `cp_firing_states[surprise]`, reused-by-import, SAME process-shared organ instance -- no duplicate build).
     A CONFIRM turn (the brain's own recall matches what was asserted) reads near-zero surprise; a
     CONTRADICT/NOVEL turn reads near its calibrated ceiling. Prediction error as the salience/reward-context
     signal (Schultz 1998 dopaminergic RPE) is a more principled "is this worth engaging with" read than a host
     `set`-membership novelty fraction, and it comes off a genuinely spiking population rate.
  2. FALLBACK -- the affect-appraisal valence ladder (`webapp/affect_drives_chat.py`, board #81's V+/V- opponent
     population read off `cp_firing_states`), read ONLY when it has already run THIS turn (no extra build/
     appraisal here) and no expectation-bearing assertion exists to drive path 1.

TRANSDUCTION (declared residual, NOT claimed closed). Both paths return a normalized scalar in [0, 1]; the map
to the SNc afferent range reuses `da_mode_drives_chat._MAX_AFFERENT_PA` (the SAME 0..1400 pA calibration
`da-mode-drives-response` already uses for its OWN afferent) via a fixed linear rescale
(`pa = normalized * _MAX_AFFERENT_PA`). This ONE host multiply is the same class of residual
`da-mode-drives-response` already declares for its e->pA map; closing it needs a spiking transduction stage
(the read driving the SNc population through a synapse, not a Python float), named as the next rung.

LESION (`BRAIN_REWARD_VALUE_LESION`). On the PRIMARY (surprise) path this reuses the organ's OWN per-call
prediction-edges-zeroed twin (`sorg.judge(..., lesion=True)`, `research/runners/_spiking_expectation_rpe_derisk.py`'s
`_install_block_diagonal(..., 0.0)` bridge) -- a genuine synaptic cut, independent of `BRAIN_SURPRISE_LESION`
(a separate per-call argument on the SAME shared organ) and of the production `surprise_info`/`surprise_prefix`
block computed later in the same turn (this module's read never mutates the shared organ's main bridge). Under
this lesion a CONFIRM and a CONTRADICT turn read the SAME (elevated, undifferentiated) rate -- the
differentiation this module's afferent depends on COLLAPSES. On the FALLBACK (affect-valence) path the lesion is
a WEAKER, declared simplification: the read magnitude is forced to 0.0 (not a synaptic cut of the #81 ladder,
which has its own independent `BRAIN_AFFECT_DRIVES_LESION`) -- excluded from any load-bearing claim on that path.

CONTRACT (additive, reversible, byte-identical-off). `reward_value_enabled()` gates everything; when
`BRAIN_REWARD_VALUE_AFFERENT` is unset, `spiking_reward_value()` is never called from
`da_mode_drives_chat.observe_turn` (see the 4-line hook there), so `afferent_override` is exactly what it was
before this module existed -> BYTE-IDENTICAL. `spiking_reward_value()` itself never raises (a caller degrades to
`None` -> the pre-existing host `engagement_of()` path, unchanged) and never mutates chat state beyond the reads
its imports already perform.

NO `sim/` edit; reuse-by-import; process backend (cupy in production, numpy in tests/dev).
See research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


def reward_value_enabled() -> bool:
    """`BRAIN_REWARD_VALUE_AFFERENT` truthy (1/true/on/yes) enables this module's SNc-afferent override. Default
    (unset) -> False -> byte-identical to before this module existed."""
    return os.environ.get("BRAIN_REWARD_VALUE_AFFERENT", "0").strip().lower() in ("1", "true", "on", "yes")


def reward_value_lesioned() -> bool:
    """`BRAIN_REWARD_VALUE_LESION` truthy -> cut the spiking read this module relies on (see module docstring for
    the per-path lesion semantics: a genuine per-call synaptic cut on the surprise path, a forced-zero on the
    weaker affect-valence fallback)."""
    return os.environ.get("BRAIN_REWARD_VALUE_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def _surprise_path(chat, message: str, seed: int, lesion: bool) -> Optional[dict]:
    """PRIMARY: an expectation-bearing assertion this turn -> the surprise organ's confirm/violate rate,
    normalized against its OWN calibrated confirm/contradict threshold. Returns None when this turn is not an
    assertion the brain already holds an expectation for (extract_assertion fails, or `chat.inner.what_does`
    recalls nothing) -- the caller then tries the affect-valence fallback."""
    try:
        import research.runners.surprise_production_organ as _SO
    except Exception:
        return None
    try:
        if not _SO.surprise_enabled():
            return None  # the parent faculty is off project-wide -- declared: never a silent back door around it
        asrt = _SO.extract_assertion(message)
        if asrt is None:
            return None
        a_s, v_s, p_asserted = asrt
        try:
            p_stored = chat.inner.what_does(a_s, v_s)   # the brain's OWN spiking recall, not a host lookup
        except Exception:
            p_stored = None
        if not p_stored:
            return None
        sorg = _SO.get_organ(seed=seed)                  # the SAME process-shared organ production reads
        sj = sorg.judge(a_s, v_s, str(p_stored), str(p_asserted), lesion=bool(lesion))
        hz = float(sj["surprise_hz"])
        threshold = float(sj["threshold"])
        denom = max(2.0 * threshold, 1e-6)
        normalized = float(np.clip(hz / denom, 0.0, 1.0))
        return {
            "on": True, "source": "surprise", "lesioned": bool(lesion),
            "agent": a_s, "action": v_s, "stored_patient": str(p_stored), "asserted_patient": str(p_asserted),
            "surprise_hz": hz, "threshold": threshold, "surprised": bool(sj["surprised"]),
            "normalized": normalized,
        }
    except Exception as e:
        return {"on": True, "source": "surprise", "error": f"{type(e).__name__}: {e}", "normalized": 0.0}


def _affect_valence_path(chat, lesion: bool) -> Optional[dict]:
    """FALLBACK: the #81 graded-affect ladder's persistent EMA valence for THIS session, read-only (no extra
    appraisal/build here -- only consulted when `affect_drives_chat.observe_turn` has ALREADY run this turn).
    Returns None when this session has no affect-drives workspace yet (the coupling never fires this turn)."""
    ws = getattr(chat, "_affect_drives_workspace", None)
    if ws is None:
        return None
    try:
        valence = 0.0 if lesion else float(getattr(ws, "ema_valence", 0.0))
        arousal = float(getattr(ws, "ema_arousal", 0.0))
        normalized = float(np.clip(abs(valence), 0.0, 1.0))
        return {
            "on": True, "source": "affect_valence", "lesioned": bool(lesion),
            "ema_valence": valence, "ema_arousal": arousal, "normalized": normalized,
        }
    except Exception as e:
        return {"on": True, "source": "affect_valence", "error": f"{type(e).__name__}: {e}", "normalized": 0.0}


def spiking_reward_value(chat, message: str, seed: int) -> Optional[dict]:
    """The production entry point `da_mode_drives_chat.observe_turn` calls when `reward_value_enabled()`. Tries
    the surprise-organ PRIMARY path, then the affect-valence FALLBACK, and returns the first that fires. Returns
    None when neither spiking source is available this turn -- the caller then falls through to the pre-existing
    host `engagement_of()` afferent, UNCHANGED (declared, not hidden -- see the PREREGISTRATION's residual #4).
    `pa` is the transduced SNc afferent current (pA), reusing `da_mode_drives_chat._MAX_AFFERENT_PA`. Never raises
    out (mirrors the sibling `*_drives_chat.observe_turn` never-crash contract)."""
    lesion = reward_value_lesioned()
    try:
        info = _surprise_path(chat, message, seed, lesion)
        if info is None:
            info = _affect_valence_path(chat, lesion)
        if info is None:
            return None
        try:
            from webapp.da_mode_drives_chat import _MAX_AFFERENT_PA
        except Exception:
            _MAX_AFFERENT_PA = 1400.0
        info["pa"] = float(np.clip(info.get("normalized", 0.0), 0.0, 1.0)) * float(_MAX_AFFERENT_PA)
        info["residual"] = "read->pA transduction is a fixed host linear rescale (declared, not claimed closed)"
        return info
    except Exception as e:
        return {"on": True, "source": "error", "error": f"{type(e).__name__}: {e}", "normalized": 0.0, "pa": 0.0}
