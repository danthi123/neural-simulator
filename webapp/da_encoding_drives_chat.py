"""DA-GATED ENCODING wired into the LIVE chat store (board WAVE-0, Gap-4 coupling), DEFAULT-ON (flipped 2026-08-25).

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

CONTRACT (additive, reversible, byte-identical-off). DEFAULT-ON: unset arms the coupling; `BRAIN_DA_ENCODING=0` is the
byte-identical escape.
  * `da_encoding_enabled()` gates the whole coupling. When DISABLED (`=0`) the server skips it entirely: `encoding_gain_fn`
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
_G_MIN = 0.5          # the RAW (pre-lever-2) recall-floor clamp (homeostasis OFF path only)
_G_MAX = 3.0

# ---------------------------------------------------------------------------
# LEVER-2 (2026-08-22): the HOMEOSTATIC companion process the bare DA gate was missing.
# ---------------------------------------------------------------------------
# The bare DA gate (g = clip(0.5, 3.0, 1 + k(DA-tonic))) writes a below-tonic-DA fact at g<1 (floored 0.5). On the
# production magnitude store that HALVES the stored |w| -> the fact's SNR drops below the RF read floor under only mild
# read stress, so recall REGRESSES at narrow DA spread AND the over-suppressed block's decode is corrupted enough to
# spuriously match an unstored cue (the "encoding-introduced" moat leak). Both are the SAME defect: the static g_min=0.5
# clamp is a PROXY for the homeostatic process real synapses run alongside potentiation -- Turrigiano 2008 homeostatic
# synaptic scaling ("The Self-Tuning Neuron", Cell 135(3):422-435): a MULTIPLICATIVE scaling of a neuron's synapses
# toward an activity SET-POINT that PRESERVES their relative strengths (so a DA-salience ORDERING survives) while
# preventing runaway weakening (no synapse driven below the level that keeps it functional).
#
# The lever-2 rule (two parts, both homeostatic):
#   (1) a recall-SAFE FLOOR at the set-point: g_floor = A* = 1.0 -- a below-tonic-DA fact is written at unit magnitude
#       (== the OFF-arm byte-identical write), never weaker. This is the "don't drive a synapse below functional" half
#       of synaptic scaling; it removes BOTH the low-sigma recall regression and the encoding-introduced moat leak
#       (which the diagnosis traced to a g=0.5 block's corrupted decode).
#   (2) a MULTIPLICATIVE population scale toward the set-point: g = clip(g_floor, g_max, s * r), r = 1 + k(DA-tonic)
#       the raw salience (UNCLAMPED), s = clip(s_min, s_max, A*/mu), mu = a running EMA of r (the neuron's tracked mean
#       "activity"). A DA distribution skewed high (mu>1) pulls s<1 -> the high-DA boost is REGULATED toward the
#       set-point (Turrigiano's total-activity homeostasis); a boring/low stretch (mu<1) pulls s>1. s is COMMON across
#       facts -> relative salience ORDER is preserved. This is the load-bearing multiplicative half (distinguishable
#       from a bare floor: it modulates the high-DA gain by the running mean).
#
# HONEST SCOPE (BRAIN-BASED-ONLY). This is HOST arithmetic (a multiply + clip + a scalar EMA) on the write gain, at the
# SAME composer layer as the DA gate it companions (`encoding_gain_fn`, a callable read in `_write_block`; NO sim/ edit).
# It is a documented PROXY for an emergent spiking homeostatic-plasticity rule on the substrate synapses, exactly as the
# DA gate itself is a host proxy for DA-gated synaptic potentiation. The biology (Turrigiano multiplicative scaling) is
# the DESIGN principle; the on-substrate synaptic-scaling realization is the tracked next target. It reads a
# brain-derived signal (the running mean of the self-produced-DA gain), so it is grounded, not a free host knob.
#
# ---------------------------------------------------------------------------
# LEVER-4 (2026-09-05, scaffold-retirement backlog rank-16): the remaining LEAF linear map, retired.
# ---------------------------------------------------------------------------
# LEVER-3 above retired the POPULATION-level homeostatic regulation onto the substrate (a genuine synaptic-scaling
# rule read from measured neural activity). What it left untouched is the PER-WRITE leaf itself: "given the live
# DA, how much gain does THIS fact get" was still `_gain_map()`'s closed-form `g = clip(g_min, g_max, 1 +
# k_DA*(DA-baseline))` -- host arithmetic on a scalar. `da_encoding_spiking_gain_enabled()` (`BRAIN_DA_ENCODING_
# SPIKING_GAIN`) swaps that ONE leaf for `research.runners._da_write_gain_spiking_derisk.
# spiking_write_gain`: a small excitatory population (IZH2007_HIPPO_PYRAMIDAL -- the SAME hippocampal cell class
# this coupling's own Lisman-Grace citation names) whose excitability is modulated by the SAME live DA through the
# neuromodulator subsystem's `excitability_drive` target (the identical target_type/scope idiom
# `_neuromod_spiking_da_mode_derisk` already uses for DA->str_D1/D2); the gain is read from that population's OWN
# firing rate, not a python formula. 6/6-seed GO (load-bearing, monotonic, lesion-collapses, parity corr>=0.99
# with the host map, deterministic): see `research/findings/2026-09-05-da-write-gain-spiking-derisk-GO.md`.
# FLIPPED DEFAULT-ON 2026-09-05 (production-flip verify GO, `research/runners/_da_write_gain_spiking_hook_verify.py`
# re-run against the new default -- OFF-arm pinned to the EXPLICIT `=0` escape per the flip_offarm_staleness
# discipline, plus a new flip-correctness arm proving unset==explicit-"1"; see
# `research/findings/2026-09-05-rank16-rank20-rank10-production-flip-GO.md`): `BRAIN_DA_ENCODING_SPIKING_GAIN` unset now
# runs the spiking population read; `=0` (or false/no/off) is the BYTE-IDENTICAL escape back to `_gain_map()`.
_A_STAR = 1.0        # the homeostatic activity set-point == the recall-safe unit-magnitude (tonic) write
_G_FLOOR_HOMEO = 1.0 # the recall-safe floor: a low-DA fact is written at unit magnitude, never below (Turrigiano floor)
_EMA_BETA = 0.25     # the homeostatic integration rate for the running mean of the raw salience (slow self-tuning)
_S_MIN, _S_MAX = 0.5, 1.5   # bound the common multiplicative scale (guards live all-low / all-high edge streams)
_MU_INIT = 1.0       # the running-mean init == the set-point (the first write is un-regulated: s=1 -> g=clip(floor,max,r))

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


# 2026-08-25 FLIP: DA-gated encoding is DEFAULT-ON (the coupling arms; a taught fact's WRITE MAGNITUDE rides the brain's
# own self-produced tonic dopamine — Lisman-Grace hippocampal-VTA loop, Kandel D.16). Gated by the 6-seed substrate-
# scaling flip-gate soak GO (research/findings 2026-08-25-da-encoding-substrate-turrigiano-scaling-FLIP) + the two prep
# rungs (the OFF-arm verifiers pinned to explicit BRAIN_DA_ENCODING=0; the substrate-homeostasis consolidation trigger
# wired into the idle tick). `BRAIN_DA_ENCODING=0` is the BYTE-IDENTICAL ESCAPE. The named constant is the ledger anchor
# (docs/PRODUCTION_INTEGRATION_LEDGER.yaml row da-gated-encoding default_anchor); flipping it back to False blocks.
_DA_ENCODING_DEFAULT_ON = True


def da_encoding_enabled() -> bool:
    """The master flag. DEFAULT-ON (flipped 2026-08-25). `BRAIN_DA_ENCODING` unset -> the default (`_DA_ENCODING_DEFAULT_ON`
    == ON): the DA-gated encoding coupling arms and a taught fact's WRITE MAGNITUDE rides the brain's own self-produced
    tonic dopamine. `BRAIN_DA_ENCODING=0` (or false/no/off) is the BYTE-IDENTICAL ESCAPE: the coupling is skipped
    entirely -> `encoding_gain_fn` untouched -> the store is byte-identical to pre-flip HEAD, and no `da_encoding` key is
    attached. Any of {1,true,on,yes} also arms it.

    FLIP GATE STATUS 2026-08-25 (CLEARED -> FLIPPED default-ON): the magnitude-store no-regression flip gate is GO
    (2026-08-25-da-encoding-substrate-turrigiano-scaling-FLIP: 6-seed soak GO with the on-substrate Turrigiano
    synaptic-scaling homeostat + target-block instrument fix -- moat_introduced=0, clean=0, genuine stress-net=0,
    cross-check byte-equal), and the two prep rungs the flip required are done: the OFF-arm verifiers
    (`_da_encoding_wired_verify`, `_wave4_composed_flip_noregression`) are pinned to explicit `BRAIN_DA_ENCODING=0`
    (they no longer rely on unset==off), and the substrate-homeostasis consolidation TRIGGER is wired into the idle tick
    (`continuous_engine.consolidate_substrate_homeostasis`, fired between turns when the store grew). Flip verification:
    the wire-in verifier + wave4 composed no-regression stay GO with the default ON, ON is load-bearing (g_high>g_low,
    lesion severs), and `=0` is byte-identical to pre-flip (the coupling install is skipped on the same code path)."""
    v = os.environ.get("BRAIN_DA_ENCODING")
    if v is None:
        return _DA_ENCODING_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "on", "yes")


def da_encoding_substrate_enabled() -> bool:
    """The LEVER-3 (2026-08-25) on-substrate Turrigiano homeostat. `BRAIN_DA_ENCODING_SUBSTRATE` unset/truthy -> the
    homeostat is the GENUINE synaptic-scaling rule `OneBrainComposer.apply_homeostatic_scaling()` (resonate-SENSE each
    engram's readout activity -> multiplicatively rescale its store synapses toward the unit set-point), run as a
    CONSOLIDATION pass (`apply_substrate_homeostasis`); the per-write path then carries the RAW DA gate (the substrate
    pass supplies the recall-safe floor + regulation). Set 0 -> fall back to the host-proxy per-write homeostat
    (`homeostatic_step`, the lever-2 ablation). This REPLACES the host arithmetic proxy with a substrate synaptic rule
    (the BRAIN-BASED-ONLY deliverable) -- the scaling factor is computed from MEASURED NEURAL ACTIVITY, not a DA EMA."""
    return os.environ.get("BRAIN_DA_ENCODING_SUBSTRATE", "1").strip().lower() in ("1", "true", "on", "yes")


def da_encoding_lesioned() -> bool:
    """`BRAIN_DA_ENCODING_LESION` truthy -> pin the encoding gain to 1.0 regardless of the DA level (sever the
    DA->encoding-gain link). The load-bearing proof: the write magnitude stops riding the DA read even though the DA
    level still varies. DISTINCT from BRAIN_DA_DRIVES_LESION (which collapses the DA LEVEL itself)."""
    return os.environ.get("BRAIN_DA_ENCODING_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


_DA_ENCODING_SPIKING_GAIN_DEFAULT_ON = True   # FLIPPED 2026-09-05 (rank-16 production-flip GO, 6/6 no-regression)


def da_encoding_spiking_gain_enabled() -> bool:
    """LEVER-4 (2026-09-05, scaffold-retirement rank-16): retire the remaining LEAF host linear map. Even with
    LEVER-3's on-substrate homeostat (`da_encoding_substrate_enabled`) doing the POPULATION-level regulation, the
    PER-WRITE computation "how strongly does THIS DA level drive the gain" was still `_gain_map()`'s closed-form
    `g = clip(g_min, g_max, 1 + k_DA*(DA - DA_baseline))` -- host arithmetic on a scalar, not a neuron or synapse.
    `BRAIN_DA_ENCODING_SPIKING_GAIN` unset -> ON (FLIPPED DEFAULT-ON 2026-09-05, `_DA_ENCODING_SPIKING_GAIN_
    DEFAULT_ON`): swaps that leaf for `research.runners._da_write_gain_spiking_derisk.spiking_write_gain`: a
    small excitatory population (IZH2007_HIPPO_PYRAMIDAL, the SAME cell class this coupling's own Lisman-Grace
    citation names) whose excitability is modulated by the SAME live DA via the neuromodulator `excitability_
    drive` target (the exact target_type/scope idiom `_neuromod_spiking_da_mode_derisk` already uses for
    str_D1/D2); the population's OWN firing rate -- not a python formula -- is what the gain is read from.
    `BRAIN_DA_ENCODING_SPIKING_GAIN` in {0,false,off,no} (explicit) is the BYTE-IDENTICAL ESCAPE back to
    `_gain_map()` (the new module is never even imported on that path). Any of {1,true,on,yes} also arms it
    (identical branch to unset, by construction). Flip verify:
    `research/findings/2026-09-05-rank16-rank20-rank10-production-flip-GO.md`."""
    v = os.environ.get("BRAIN_DA_ENCODING_SPIKING_GAIN")
    if v is None:
        return _DA_ENCODING_SPIKING_GAIN_DEFAULT_ON
    return v.strip().lower() in ("1", "true", "on", "yes")


def da_encoding_spiking_gain_lesioned() -> bool:
    """This mechanism's OWN lesion (distinct from `da_encoding_lesioned()`, which pins g=1.0 outright upstream of
    either implementation, and from `da_drives_lesioned()`, which silences the SNc nucleus that sets the DA LEVEL
    itself). `BRAIN_DA_ENCODING_SPIKING_GAIN_LESION` truthy severs JUST the excitability_drive target that lets DA
    reach the write_gain population (built with sensitivity pinned to 0.0) -- a structural severance of THIS
    mechanism's own DA->population link, proving the population's rate-derived gain (not merely its input) is
    what rides the live DA read. No effect while `da_encoding_spiking_gain_enabled()` is False."""
    return os.environ.get("BRAIN_DA_ENCODING_SPIKING_GAIN_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def _leaf_gain(da: float, g_min: float, g_max: float) -> float:
    """The single leaf computation both `encoding_gain_for()` branches below call: given the live DA and this
    branch's (g_min, g_max), return the write-magnitude gain. Dispatches to the spiking read (LEVER-4) only when
    explicitly armed; the default path is `_gain_map()`, UNCHANGED, so this function is byte-identical to a bare
    inlined `_gain_map()` call whenever the new flag is off (the import only happens on the ON branch)."""
    if da_encoding_spiking_gain_enabled():
        from research.runners._da_write_gain_spiking_derisk import spiking_write_gain
        return spiking_write_gain(da, _DA_TONIC_BASELINE, g_min=g_min, g_max=g_max,
                                  lesion=da_encoding_spiking_gain_lesioned())
    return float(_gain_map()(da, _DA_TONIC_BASELINE, _K_DA, g_min, g_max))


def da_encoding_homeostasis_enabled() -> bool:
    """The lever-2 homeostatic companion (Turrigiano multiplicative scaling + recall-safe floor). DEFAULT ON whenever
    the encoding coupling is on: `BRAIN_DA_ENCODING_HOMEOSTASIS` unset/truthy -> homeostasis ON; set to 0/false/off/no
    -> the RAW pre-lever-2 map (g = clip(0.5, 3.0, 1+k(DA-tonic))) for the ablation control. Homeostasis is what makes
    the default-ON flip safe (removes the low-sigma regression + the encoding-introduced moat leak), so it is on by
    default with the coupling; the OFF-of-homeostasis path reproduces the bare gate's UNDEFINED flip gate."""
    return os.environ.get("BRAIN_DA_ENCODING_HOMEOSTASIS", "1").strip().lower() in ("1", "true", "on", "yes")


def homeostatic_step(mu, r, a_star=_A_STAR, g_floor=_G_FLOOR_HOMEO, g_max=_G_MAX,
                     s_min=_S_MIN, s_max=_S_MAX, ema_beta=_EMA_BETA):
    """The pure lever-2 homeostatic update for ONE write (reused verbatim by the flip-gate soak so the validated logic
    IS the shipped logic). Given the running mean `mu` of the raw salience and this write's raw salience `r =
    1+k(DA-tonic)` (UNCLAMPED), return (g_effective, mu_next):
       s = clip(s_min, s_max, a_star / mu)            # Turrigiano common multiplicative scale toward the set-point
       g = clip(g_floor, g_max, s * r)                # relative order preserved; never below the recall-safe floor
       mu_next = (1-ema_beta)*mu + ema_beta*r         # slow self-tuning of the tracked mean activity
    """
    s = min(s_max, max(s_min, (a_star / mu) if mu > 0 else 1.0))
    g = float(min(g_max, max(g_floor, s * r)))
    mu_next = float((1.0 - ema_beta) * mu + ema_beta * r)
    return g, mu_next


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


def encoding_gain_for(chat, advance: bool = False) -> float:
    """The per-store write gain on the LIVE self-produced DA. Under the lesion -> pinned 1.0 (the coupling is severed).
    HOMEOSTASIS ON (lever-2 default): g = clip(1.0, 3.0, s * r), r = 1+k(DA-tonic), s = clip(0.5, 1.5, 1.0/mu), mu a
    running EMA of r held on the chat (`_da_encoding_mu`) -- Turrigiano multiplicative scaling toward the set-point with
    a recall-safe floor; a tonic (r=1) write at steady state (mu->1) stays g=1.0 (byte-identical). HOMEOSTASIS OFF
    (ablation): the RAW pre-lever-2 map g = clip(0.5, 3.0, r).
    `advance` is the STORE-vs-PEEK switch: the store hook passes advance=True so the homeostatic running mean self-tunes
    ONCE PER ACTUAL WRITE (== one homeostatic_step per store, matching the soak's per-fact fold); a trace/peek read
    (advance=False) computes g from the current mu WITHOUT advancing it, so reads never drift the set-point."""
    if da_encoding_lesioned():
        return 1.0
    da = da_level_of(chat)
    if da_encoding_substrate_enabled():
        # LEVER-3: the homeostat is the on-substrate synaptic-scaling CONSOLIDATION pass (apply_substrate_homeostasis).
        # The per-WRITE gain here carries only the RECALL-SAFE FLOOR (g >= 1.0 == the set-point): a below-tonic-DA fact
        # is never encoded below the readable floor (the "keep the synapse functional" write invariant), so the live
        # store is SAFE even between consolidation passes; a salient fact is still boosted. The substrate pass then does
        # the genuine population REGULATION (down-scaling over-strong engrams toward the set-point) -- the part that
        # needs the whole population, run offline. The NET consolidated store (low->1.0, high->regulated) equals the
        # flip-gate soak's validated state. (Do NOT also run the host per-write EMA homeostat here -> double homeostasis.)
        # LEVER-4 (default OFF, `BRAIN_DA_ENCODING_SPIKING_GAIN`): `_leaf_gain` swaps this leaf for a spiking read
        # of the SAME (da, floor, ceiling) instead of `_gain_map()`'s closed form -- see `_leaf_gain`'s docstring.
        return _leaf_gain(da, _G_FLOOR_HOMEO, _G_MAX)
    if not da_encoding_homeostasis_enabled():
        return _leaf_gain(da, _G_MIN, _G_MAX)
    r = 1.0 + _K_DA * (da - _DA_TONIC_BASELINE)                       # raw salience (UNCLAMPED); the homeostat floors it
    mu = float(getattr(chat, "_da_encoding_mu", _MU_INIT))
    g, mu_next = homeostatic_step(mu, r)
    if advance:
        try:
            chat._da_encoding_mu = mu_next                            # self-tune the running mean ONCE per real store
        except Exception:
            pass
    return g


def install_encoding_gain(chat) -> Optional[float]:
    """Install the DA-gated `encoding_gain_fn` on the LIVE composer (read AT STORE TIME inside `chat.gate`'s
    `_maybe_acquire`). The closure reads the fresh live DA each store AND advances the homeostatic running mean (once
    per real write), so a fact taught while engaged is encoded stronger AND the population is self-tuned toward the
    recall-safe set-point. Idempotent — reinstalls a fresh closure over THIS chat; the homeostatic state
    (`_da_encoding_mu`) PERSISTS across reinstalls (init to the set-point only if absent). Returns the current gain g as
    a PEEK (does NOT advance mu -- only the store closure advances it), or None if there is no composer. NO `sim/` edit
    (a composer-layer callable; the homeostat is a host proxy at that same layer)."""
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    if comp is None:
        return None
    if not hasattr(chat, "_da_encoding_mu"):
        chat._da_encoding_mu = _MU_INIT
    if da_encoding_substrate_enabled():                              # arm the on-substrate Turrigiano homeostat
        try:
            if hasattr(comp, "homeostatic_scaling"):
                comp.homeostatic_scaling = True                     # the composer's apply_homeostatic_scaling is now live
        except Exception:
            pass
    comp.encoding_gain_fn = lambda: encoding_gain_for(chat, advance=True)
    return encoding_gain_for(chat, advance=False)


def apply_substrate_homeostasis(chat):
    """Run the on-substrate Turrigiano synaptic-scaling CONSOLIDATION pass on the live composer's store: resonate-SENSE
    each stored engram's readout activity, then multiplicatively rescale its store synapses toward the unit set-point
    (`OneBrainComposer.apply_homeostatic_scaling`). This is the GENUINE synaptic homeostat that replaces the host-proxy
    per-write EMA (BRAIN-BASED-ONLY: the scale is computed from measured neural activity, actuated on the synaptic
    weight). Turrigiano scaling is biologically SLOW/OFFLINE, so this is a consolidation-time op (call at a consolidation
    event / after a batch of taught facts), NOT per turn -- re-running it repeatedly on an already-scaled store would
    pull the population toward unit. Returns the applied per-engram scale vector, or None if unavailable/disabled. NO-OP
    (returns None) when the substrate homeostat is off or the composer lacks the rule. Validated: the 6-seed flip-gate
    soak (research/runners/_da_encoding_leansoak.py --substrate-scaling) is byte-equal to a real production build."""
    if not (da_encoding_enabled() and da_encoding_substrate_enabled()) or da_encoding_lesioned():
        return None
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    if comp is None or not hasattr(comp, "apply_homeostatic_scaling"):
        return None
    try:
        comp.homeostatic_scaling = True
        return comp.apply_homeostatic_scaling()
    except Exception:
        return None


def uninstall_encoding_gain(chat) -> None:
    """Revert the composer to the byte-identical unit-magnitude write (`encoding_gain_fn = None`)."""
    comp = getattr(getattr(chat, "inner", None), "composer", None)
    if comp is not None:
        comp.encoding_gain_fn = None
        if hasattr(comp, "homeostatic_scaling"):
            comp.homeostatic_scaling = False
