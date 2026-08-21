"""DA-GATED ENCODING wired into the LIVE chat store (board WAVE-0, Gap-4 coupling), default-OFF.

WHAT THIS IS. The composer's fact-ENCODING strength AT STORE TIME is scaled by the brain's OWN self-produced tonic
dopamine (Lisman-Grace hippocampal-VTA loop; Kandel D.16 — dopamine gates entry into LONG-TERM memory: a fact heard
while the SNc bursts, i.e. a salient / engaged utterance, is encoded STRONGER and stays more STABLE; a fact heard at DA
baseline is encoded at unit magnitude). The RF composer already carries the write-side hook: `encoding_gain_fn`, a
callable `() -> float` read at store time (`research/runners/rf_phasor_composer.py:933-934`,
`OneBrainComposer._write_block`). `encoding_gain_fn = None` -> g = 1.0 -> the byte-identical unit-magnitude write
(today's production default). This module installs a gain that reads the LIVE self-produced DA so the WRITE magnitude
scales with the brain's engagement — the WAVE-0 write-side Gap-4 coupling, the counterpart to the #76/#79 DA-mode
READ-side (which drives the reply's forthcomingness).

THE LIVE DA SOURCE (reused, not re-derived). `webapp/da_mode_drives_chat.observe_turn(chat, msg)` (board #79) drives
the spiking SNc nucleus from the turn's engagement, reads the SELF-PRODUCED tonic `dopamine` concentration off the
neuromodulator bus, and stashes it on `chat._last_da_drives["da_level"]` (tonic ~= 0.5). In `webapp/server.py` that
read runs BEFORE the gate/acquire, so the level is FRESH when a taught SVO is stored. This module reads that same live
level; when da-drives is off / not yet observed (`_last_da_drives` absent) the level defaults to tonic 0.5 -> g = 1.0
(neutral) -> byte-identical to the un-gated write.

THE GAIN MAP (reused verbatim — NOT a new convention). `da_to_encoding_gain(da, da_baseline, k_da, g_min, g_max)` from
the VALIDATED board I-7-b deployment de-risk (`research/runners/_burndown_I7_dopamine_encoding_deploy_derisk.py`, 3-seed
GO; the same map the consolidation-probe2 limbic write-side uses):
    g = clip(g_min, g_max, 1 + k_DA * (DA - DA_baseline))
DA at baseline (tonic 0.5) => g = 1 (the no-modulation knob = byte-identical write); a salient (high-DA) turn => g > 1
(a stronger, more-stable encoding); clamped both ways (g_min keeps a low-DA turn from erasing a fact, g_max is the
saturation ceiling). k_DA = 2.0 == the I-7-b / consolidation-probe2 default.

CONTRACT (additive, reversible, byte-identical-off).
  * `da_encoding_enabled()` gates the whole coupling. When DISABLED the server skips it entirely: `encoding_gain_fn`
    stays None (never touched) -> the store is BYTE-IDENTICAL to pre-wiring, and no `da_encoding` key is attached.
  * When ENABLED, `install_encoding_gain(chat)` sets `chat.inner.composer.encoding_gain_fn` to a fresh closure that
    reads the live DA at store time. g == 1.0 at tonic (an unengaged turn is neutral) and on a missing DA read.
  * LESION (`BRAIN_DA_ENCODING_LESION=1`): the gain is pinned to 1.0 REGARDLESS of the DA level -> the coupling is
    severed (the load-bearing proof: the write magnitude no longer rides the DA read even though the DA level still
    varies). This is DISTINCT from `BRAIN_DA_DRIVES_LESION` (which silences the SNc nucleus / collapses the level
    itself); this lesion cuts only the DA->encoding-gain link.

SUBSTRATE DEPENDENCE (honest scope). The gain scales the STORED trace only on a MAGNITUDE-carrying composer — the
production-default `OneBrainComposer` (`store_conns` complex weights) and the RF composer's substrate store
(`enable_substrate_store=True`, `_store_substrate` writes `g * zc`). On the RF numpy FAST-path recall
(`enable_substrate_store=False`, the `BRAIN_COMPOSER_KIND=rf` speed path) the stored recall is magnitude-INVARIANT
(phases only), so `encoding_gain_fn` is not read by that store -> the coupling is a write-side reserve there. The WIRING
(the live DA reaches the composer's store hook and produces a differential write gain) is what this module delivers; the
magnitude mechanism is the I-7-b / consolidation-probe2 GO.

REUSE-BY-IMPORT (NO `sim/` edit). Composer-layer callable only. `git diff sim/` is empty.
"""
from __future__ import annotations

import os
from typing import Optional

# the live SNc-self-produced tonic DA baseline (webapp/da_mode_drives_chat: tonic ~= 0.5). g == 1.0 here (neutral).
_DA_TONIC_BASELINE = 0.5
# the VALIDATED DA->gain slope (== the I-7-b / consolidation-probe2 default k_da). g = 1 + k_DA*(DA - tonic), clamped.
_K_DA = 2.0
_G_MIN = 0.5
_G_MAX = 3.0

# lazily-bound reference to the canonical gain map (imported on first ON use so the default-OFF path stays light).
_GAIN_MAP = None


def _gain_map():
    """The canonical, VALIDATED DA->encoding-gain map, reused by import from the board I-7-b deployment de-risk (do NOT
    re-invent). Lazy + cached so the default-OFF turn never pulls the heavy runner chain. Falls back to the identical
    inline formula only if the import fails (defensive; keeps a turn from crashing)."""
    global _GAIN_MAP
    if _GAIN_MAP is None:
        try:
            from research.runners._burndown_I7_dopamine_encoding_deploy_derisk import da_to_encoding_gain
            _GAIN_MAP = da_to_encoding_gain
        except Exception:
            # == da_to_encoding_gain (research/runners/_burndown_I7_dopamine_encoding_deploy_derisk.py) — identical.
            _GAIN_MAP = (lambda da, da_baseline, k_da, g_min=_G_MIN, g_max=_G_MAX:
                         float(min(g_max, max(g_min, 1.0 + k_da * (da - da_baseline)))))
    return _GAIN_MAP


def da_encoding_enabled() -> bool:
    """The master flag. `BRAIN_DA_ENCODING` truthy (1/true/on/yes) enables the DA-gated encoding coupling; anything
    else (the default UNSET) leaves it OFF -> `encoding_gain_fn` untouched -> the store is byte-identical to HEAD."""
    return os.environ.get("BRAIN_DA_ENCODING", "0").strip().lower() in ("1", "true", "on", "yes")


def da_encoding_lesioned() -> bool:
    """`BRAIN_DA_ENCODING_LESION` truthy -> pin the encoding gain to 1.0 regardless of the DA level (sever the
    DA->encoding-gain link). The load-bearing proof: the write magnitude stops riding the DA read even though the DA
    level still varies. DISTINCT from BRAIN_DA_DRIVES_LESION (which collapses the DA LEVEL itself)."""
    return os.environ.get("BRAIN_DA_ENCODING_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def da_level_of(chat) -> float:
    """The live self-produced tonic DA off the DA-mode read (`chat._last_da_drives["da_level"]`, set by
    da_mode_drives_chat.observe_turn earlier in the turn). Missing (da-drives off / not yet observed) -> tonic 0.5 ->
    g = 1.0 (neutral). Never raises."""
    info = getattr(chat, "_last_da_drives", None)
    if not isinstance(info, dict):
        return _DA_TONIC_BASELINE
    try:
        lvl = info.get("da_level", _DA_TONIC_BASELINE)
        return float(lvl) if lvl is not None else _DA_TONIC_BASELINE
    except (TypeError, ValueError):
        return _DA_TONIC_BASELINE


def encoding_gain_for(chat) -> float:
    """g = clip(g_min, g_max, 1 + k_DA*(DA - tonic)) on the LIVE self-produced DA. Under the lesion the gain is pinned
    to 1.0 (the coupling is severed). g == 1.0 at tonic (an unengaged turn is a neutral, byte-identical write)."""
    if da_encoding_lesioned():
        return 1.0
    da = da_level_of(chat)
    return float(_gain_map()(da, _DA_TONIC_BASELINE, _K_DA, _G_MIN, _G_MAX))


def install_encoding_gain(chat) -> Optional[float]:
    """Install the DA-gated `encoding_gain_fn` on the LIVE composer (read AT STORE TIME inside `chat.gate`'s
    `_maybe_acquire`). The closure reads the fresh live DA each store, so a fact taught while engaged is encoded
    stronger. Idempotent — reinstalls a fresh closure over THIS chat. Returns the current gain g (for the trace), or
    None if there is no composer. NO `sim/` edit (a composer-layer callable)."""
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    if comp is None:
        return None
    comp.encoding_gain_fn = lambda: encoding_gain_for(chat)
    return encoding_gain_for(chat)


def uninstall_encoding_gain(chat) -> None:
    """Revert the composer to the byte-identical unit-magnitude write (`encoding_gain_fn = None`)."""
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    if comp is not None:
        comp.encoding_gain_fn = None
