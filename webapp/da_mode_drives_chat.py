"""The #76 SPIKING DA-MODE (rest/focus/arousal) wired into the LIVE `/api/brain-chat` turn so the brain's OWN
dopamine mode is LOAD-BEARING on HOW forthcoming/engaged the reply is -- NOT observe-only (board #79,
INTEGRATION-TO-PRODUCTION). Mirrors the board-#84 affect-DRIVES + board-#85 swap-DRIVES paths.

WHAT THIS IS. Board #76 landed a 6/6-seed GO (`_neuromod_spiking_da_mode_derisk`,
`2026-08-19-neuromod-spiking-da-mode-GO.md`): on the fixed-anatomy nav basal-ganglia spiking substrate the
dopamine LEVEL comes from the substrate's OWN spiking DA nucleus (the `snc` population, IZH2007_DOPAMINE) --
a reward/context afferent drives the SNc, the SNc fires, and the neuromodulator bus transduces its firing
rate into the tonic DA concentration (the Schultz-1998 signed code), which then reconfigures the striatal
DIRECT(Go)/INDIRECT(NoGo) effective circuit (the #64 reconfiguration). Silencing the SNc nucleus collapses
the level (byte-for-byte). That GO shipped as a DEFAULT-OFF de-risk RUNNER, never in the live chat. This
module wires that NEURAL read onto the live conversational brain and makes it CHANGE the response: the
self-produced DA LEVEL is binned into a MODE (rest / focus / arousal), and the mode modulates how ENGAGED /
FORTHCOMING the reply is -- a graded engagement suffix. It is the anti-hollow-integration counterpart to the
observe-only faculties: the DA-mode READ is neural AND it demonstrably shapes the surface.

WHY ENGAGEMENT/FORTHCOMINGNESS (the coupling justified from what the mode MEANS). The DA Go/NoGo mode is an
ACTION-READINESS / approach-vs-withhold switch (Albin-DeLong-Penney; Gerfen & Surmeier 2011): HIGH DA opens
the DIRECT/Go pathway (disinhibits thalamus -> the brain is primed to ACT/APPROACH), LOW DA opens the
INDIRECT/NoGo pathway (thalamus clamped -> the brain WITHHOLDS). The faithful conversational expression of
that switch is FORTHCOMINGNESS + ENERGY, on a distinct axis from the two leads already in the reply -- #84's
affect lead is VALENCE (warmth/curtness), #85's lead is TOPIC (a transition announcement); this is
ENGAGEMENT/AROUSAL (how far the brain leans in). Chosen as a SUFFIX (not a third prefix, per the board-#79
steer) so the graded length/engagement change is orthogonal and the lesion produces a CLEAN byte-identical
vanish. So:
  * REST  (DA below tonic, NoGo / disengaged)  -> withhold: NO engagement suffix (terse; the floor).
  * FOCUS (DA moderately above tonic, Go)       -> a forthcoming engagement suffix (" -- worth going ...").
  * AROUSAL (DA high above tonic)               -> an EMPHATIC engagement suffix (longer + "!"; high energy).

THE READ (the #76 neural mechanism, reused-by-import; NO sim/ edit).
  * Each turn, the message's ENGAGEMENT (novelty of its content vs the session so far + its richness -- the
    SAME host-comprehension/sensory boundary the SVO parser + #84 appraisal + #85 topic occupy) is EMA-folded
    into a persistent per-session engagement scalar e in [0,1] (a neutral/empty turn HOLDS it -> cross-turn
    persistence). This is the ENVIRONMENTAL reward/context afferent the #76 finding names as its honest
    residual ("the reward/context SCALAR is still environmental; what is closed HERE is that the DA LEVEL is
    brain-derived, set by SNc spikes").
  * e is mapped to the SNc reward/context afferent current (0..MAX_AFFERENT pA) and driven into the spiking
    SNc nucleus. The SNc FIRES, and the neuromodulator bus (`from_region_firing_signed` on ["snc"]) reads its
    rate and self-produces the tonic DA concentration -- NEVER a host `set_concentration`. The FELT DA LEVEL
    is that bus concentration (produced by SNc spikes). The MODE is that level binned into rest/focus/arousal.
  * The read reuses the #76 runner verbatim: `PM.build` (the fixed-anatomy BG substrate, cfg.seed-seeded, all
    noise/plasticity OFF), `make_manager` (the `dopamine_mode` bus config), `measure_self_driven` (the live
    loop: each step add the SNc-produced excitability drive, step, let the bus read SNc firing). The substrate
    is built ONCE per session and its full dynamic state is snapshotted so every read is a deterministic
    function of THIS turn's afferent (history-independent; the cross-turn persistence lives in the host EMA,
    exactly like #84's body-state EMA).

THE COUPLING (what makes it LOAD-BEARING, not observe-only).
  * The neural DA MODE -> a graded ENGAGEMENT SUFFIX appended to the answer surface (`lead` is a suffix here).
    REST/NEUTRAL -> "" (byte-identical). FOCUS -> a forthcoming suffix. AROUSAL -> an emphatic one. The suffix
    is an honest EXPRESSION of the brain's engagement/approach state (a discourse "mouth"), NOT content: the
    FACT before it is the SAME gate-matched, moat-verified SVO, and the VERIFY re-parse is unchanged. So the
    mode changes HOW forthcoming the reply is, never WHICH fact is true and never whether an unmatched cue
    abstains. This is the single coupling this module wires.

THE HONESTY FLOOR (preserved BY CONSTRUCTION, mirrors #84/#85).
  * The moat / recall / abstain verdict runs FIRST and unchanged; the DA coupling only DECORATES an
    already-matched answer surface with an engagement suffix. It never enters the certainty band, never
    manufactures a fact, never flips an abstain into an assert. The content fields (`abstained`,
    `recalled_svo`, `verified`) are BYTE-IDENTICAL with the coupling on or off; only the answer SURFACE (the
    optional suffix) and the additive `da_drives` trace change.

LESION (the load-bearing / brain-based proof). `BRAIN_DA_DRIVES_LESION=1` SILENCES the spiking SNc nucleus
(the #76 anti-cheat-2 lesion: clamp its input context-independently) on every read -> the reward/context can
no longer reach the DA level -> the self-produced level collapses to its sub-firing floor REGARDLESS of the
engagement afferent -> the mode is REST -> the engagement suffix VANISHES and the surface reverts to the
byte-identical (coupling-off) answer. So the surface change RIDES the SPIKING SNc read, not a host
`if engagement>x`: silence the neural DA nucleus and the mode-difference disappears even though the world
input (an engaging message) is unchanged. This is the de-risk's OWN neural lesion, reused (not a host cut).

CONTRACT (additive, reversible, byte-identical-off).
  * `da_drives_enabled()` gates the whole block. When DISABLED the handler skips it entirely: no substrate is
    built, no read runs, no `da_drives` key is attached, and NO engagement suffix is appended -> the turn is
    BYTE-IDENTICAL to pre-wiring (the #84/#85 leads, if on, are untouched -- this module is orthogonal).
  * The substrate build + read run on the workspace's PRIVATE RNG timeline and the host process-global RNG
    (numpy + the sim backend) is restored around every read (the #77 global-RNG footgun): `PM.build` reseeds
    cp.random to cfg.seed, so without this, enabling this module would perturb the downstream RNG-dependent
    organs and break byte-identity. Snapshot host RNG, run on this workspace's own timeline, restore host.
  * The substrate build (~0.9s) is lazy on the first turn per session and kept warm; each turn runs one
    ~0.05s read (restore the post-build snapshot -> fresh manager -> one live SNc->DA loop).

REUSE-BY-IMPORT (NO `sim/` edit). The BG substrate build (`PM.build`), the `dopamine_mode` bus manager
(`make_manager`), the live SNc->DA loop (`measure_self_driven`) and the operating point (`BASELINE`,
`SNC_SILENCE_CLAMP`) come STRAIGHT from `_neuromod_spiking_da_mode_derisk` (board #76, 6/6-seed GO). This
module adds only the production glue (the per-session engagement EMA + the afferent map + the level->mode
bins + the mode->suffix map). `git diff sim/` is empty.

HONEST RESIDUALS (named, not claimed closed).
  1. The message->ENGAGEMENT scalar (novelty + richness -> the SNc reward/context afferent) is host (a
     language/sensory-comprehension boundary, like the SVO parser, #84 appraisal, #85 topic). The DA LEVEL
     (set by SNc spikes off the bus) and its SNc-nucleus dependence ARE the #76 neural mechanism
     (lesion-proven -- silence the SNc and the level, hence the mode, hence the suffix, collapses). The #76
     finding itself flags the reward/context scalar's ORIGIN as the residual; computing it from the brain's
     own sensory stream is a SEPARATE faculty (the named next rung).
  2. The mode->SUFFIX-STRING map is a HOST conditioned-articulation scaffold (the discourse "mouth"): the DA
     mode that DRIVES it is the neural SNc->DA level (load-bearing -- the lesion collapses the suffix), but
     the surface STRING for a mode is a host template, exactly the sanctioned articulation-crutch pattern
     (owner: scaffold-ok-as-conditioned-articulation IF the faculty is load-bearing on the surface, which the
     lesion proves). A brain-native engagement mouth (the suffix emitted by a spiking sequencing circuit) is
     the named next rung.
  3. This module reads its OWN co-resident #76 BG substrate (built alongside the recall composer), not merged
     onto the single recall bridge (the one-brain consolidation step, shared with the #84/#85 burn-downs).
"""
from __future__ import annotations

import os
import re
import threading
from typing import Optional

import numpy as np

# reuse-by-import the board-#76 6/6-seed-GO spiking DA-mode machinery (build + the live SNc->DA read) -- NO sim/ edit.
import research.runners._neuromod_spiking_da_mode_derisk as _DA
import research.runners._perturb_and_measure_derisk as _PM
# reuse-by-import the shared spiking novelty/salience afferent (scaffold-retirement backlog rank-4, 2026-09-05,
# research/runners/shared_salience_afferent.py) -- BRAIN_SHARED_SALIENCE, default-ON since 2026-09-05 (Track-1
# flip); see engagement_of()'s docstring below for the coupling this retires at its root (da-mode-drives-response +
# da-gated-encoding + da-gated-curiosity all read the SAME chat._last_da_drives["da_level"] this module produces).
import research.runners.shared_salience_afferent as _SHARED

_DEFAULT_SEED = 42


def _is_ndarray(x) -> bool:
    """True for a numpy OR cupy ndarray (device-agnostic; used by `_ensure`'s post-build snapshot filter so a
    cupy-backed substrate's `cp_*` state is captured too, not just numpy's -- see `_ensure`'s docstring)."""
    if isinstance(x, np.ndarray):
        return True
    try:
        import cupy
        return isinstance(x, cupy.ndarray)
    except Exception:
        return False


# ── engagement -> SNc reward/context afferent. e in [0,1] -> afferent in [0, _MAX_AFFERENT] pA. Calibrated on the
#    #76 self-produced DA(afferent) curve (tonic DA=0.5): 0pA->DA~0.05 (rest), 400pA->0.52 (neutral), 800pA->0.88
#    (focus), 1300pA->1.24 (arousal). So e~0.3 lands neutral, e~0.55 focus, e~1.0 arousal.
_MAX_AFFERENT_PA = 1400.0

# ── level -> MODE bins on the SNc-self-produced DA concentration (tonic 0.5). Below tonic = NoGo/withdraw = REST;
#    a neutral band around tonic emits NO suffix (byte-identical); moderately-above = Go/engaged = FOCUS; high = AROUSAL.
_DA_REST_MAX = 0.40        # DA < 0.40  -> REST     (disengaged; the floor -> "" suffix, byte-identical to off)
_DA_NEUTRAL_MAX = 0.62     # 0.40..0.62 -> NEUTRAL  (near tonic -> "" suffix, byte-identical to off)
_DA_FOCUS_MAX = 1.00       # 0.62..1.00 -> FOCUS    (engaged);  DA >= 1.00 -> AROUSAL (high energy)

# ── EMA engagement persistence: a strong turn dominates; a neutral/empty turn (no content) HOLDS the prior scalar
#    (cross-turn persistence). Matches #84's _EMA_DECAY so the paths agree on the persistence timescale.
_EMA_DECAY = 0.4
_RICHNESS_FULL = 8         # content-word count at which richness saturates to 1.0
_W_NOVELTY = 0.6           # engagement = _W_NOVELTY*novelty + (1-_W_NOVELTY)*richness
_MIN_CONTENT_LEN = 3       # a token must be >= this many letters to count as content (drops function words)

# ── the MODE -> engagement EXPRESSION suffix (the host conditioned-articulation scaffold; DRIVEN by the neural read).
#    A SUFFIX (distinct from the #84/#85 prefixes); on the engagement/arousal axis (distinct from valence/topic).
_SUFFIX = {
    "focus": " — worth going further here.",
    "arousal": " — there's plenty more to dig into here!",
}

_STOPWORDS = frozenset((
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "and", "or", "but", "of", "to", "in",
    "on", "at", "for", "with", "as", "by", "that", "this", "these", "those", "it", "its", "do", "does", "did",
    "you", "your", "i", "me", "my", "we", "our", "he", "she", "they", "them", "his", "her", "their", "what",
    "who", "which", "when", "where", "why", "how", "not", "no", "yes", "can", "could", "would", "should", "will",
    "about", "from", "into", "than", "then", "so", "if", "have", "has", "had", "there", "here", "some", "any",
))


def da_drives_enabled() -> bool:
    """The master flag. `BRAIN_DA_DRIVES` truthy (1/true/on/yes) enables; 0/false/off/no disables. The default when
    the env var is UNSET follows the production-integration anchor `_DA_DRIVES_DEFAULT_ON` in server.py -- this reads
    only the explicit env override (server.py combines it with the anchor, mirroring the #84/#85 flags)."""
    return os.environ.get("BRAIN_DA_DRIVES", "0").strip().lower() in ("1", "true", "on", "yes")


def da_drives_off() -> bool:
    """Explicit OFF (for a default-ON anchor): `BRAIN_DA_DRIVES` in {0,false,no,off,''}."""
    v = os.environ.get("BRAIN_DA_DRIVES")
    return v is not None and v.strip().lower() in ("0", "false", "no", "off", "")


def da_drives_lesioned() -> bool:
    """`BRAIN_DA_DRIVES_LESION` truthy -> SILENCE the spiking SNc nucleus on every read (the #76 anti-cheat-2
    lesion): the self-produced DA level collapses to its floor regardless of the engagement afferent, so the mode
    is REST and the engagement suffix VANISHES. The load-bearing proof."""
    return os.environ.get("BRAIN_DA_DRIVES_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def _content_tokens(message: str) -> list:
    """The content words of a message (the host language boundary): lowercase alpha tokens >= _MIN_CONTENT_LEN
    letters, minus a small stoplist. Used for the novelty + richness engagement read."""
    toks = re.findall(r"[a-zA-Z]+", str(message or "").lower())
    return [t for t in toks if len(t) >= _MIN_CONTENT_LEN and t not in _STOPWORDS]


def engagement_of(tokens: list, seen: set) -> float:
    """The per-turn ENGAGEMENT scalar in [0,1] (the environmental reward/context signal): novelty (fraction of
    content tokens NOT seen this session -- the dopaminergic novelty/reward response) + richness (content-word
    count, saturating). This is the HOST boundary; the DA LEVEL it induces is the neural part.

    Until the shared-afferent wiring (2026-09-05, `BRAIN_SHARED_SALIENCE`) this raw scalar was fed DIRECTLY to the
    SNc afferent (zero neurons mediating message -> pA). When the flag is on, `DaModeDrivesWorkspace.observe()`
    routes this SAME raw scalar through the shared spiking ASK-pool afferent (`shared_salience_afferent.read_
    salience`) before it reaches the EMA/afferent map below -- this function's OUTPUT is unchanged (still the host
    sensory/comprehension boundary read), only what happens to it downstream changes."""
    if not tokens:
        return 0.0
    novelty = sum(1 for t in tokens if t not in seen) / float(len(tokens))
    richness = min(len(tokens) / float(_RICHNESS_FULL), 1.0)
    return float(np.clip(_W_NOVELTY * novelty + (1.0 - _W_NOVELTY) * richness, 0.0, 1.0))


def da_to_mode(da_level: float) -> str:
    """Bin the SNc-self-produced DA concentration (tonic 0.5) into the MODE (rest/neutral/focus/arousal)."""
    d = float(da_level)
    if d < _DA_REST_MAX:
        return "rest"
    if d < _DA_NEUTRAL_MAX:
        return "neutral"
    if d < _DA_FOCUS_MAX:
        return "focus"
    return "arousal"


def mode_suffix(mode: str) -> str:
    """The engagement EXPRESSION suffix for this turn's DA mode (the conditioned-articulation scaffold; DRIVEN by
    the neural DA level). REST/NEUTRAL -> "" so the surface is byte-identical; FOCUS -> a forthcoming suffix;
    AROUSAL -> an emphatic one. The FACT before it is unchanged (VERIFY re-parse intact) -- this colors HOW
    forthcoming the reply is, never WHICH fact is true."""
    return _SUFFIX.get(mode, "")


class DaModeDrivesWorkspace:
    """A per-session DA-mode workspace: a persistent #76 BG substrate (built once, full dynamic-state snapshot so
    each read is history-independent) + a persistent EMA engagement scalar + the session's seen-token set.
    `observe(message)` folds the message engagement into the EMA (a neutral turn HOLDS it), maps it to the SNc
    reward/context afferent, runs one neural SNc->DA read, and returns the self-produced DA level + the mode + the
    engagement suffix. The build + read run on the workspace's PRIVATE RNG timeline (host process-global RNG
    restored -- the #77 footgun)."""

    def __init__(self, seed: int = _DEFAULT_SEED):
        self.seed = int(seed)
        self._sb = None
        self._nbt = None
        self._snapshot = None       # full post-build cp_* dynamic state -> history-independent reads
        self._lock = threading.Lock()
        self.ema_engagement = 0.0    # persistent engagement (cross-turn); a neutral turn holds it
        self.seen = set()            # content tokens seen this session (novelty read)
        self.n_turns = 0
        self._rng_state = None       # the workspace's PRIVATE RNG timeline (host process-global RNG never advanced)

    def _isolated(self, fn):
        """Run `fn()` (the substrate build + spiking read) on the workspace's PRIVATE RNG timeline, leaving the host
        process-global RNG (numpy + the sim backend) BYTE-UNTOUCHED. `PM.build` reseeds cp.random to cfg.seed and the
        stepping shares the process-global RNG the rest of the pipeline uses -- without this, enabling this module
        would perturb the downstream RNG-dependent organs and break byte-identity. Snapshot the host RNG, swap in this
        workspace's own continuous timeline, run, capture the advanced private timeline, restore host. (Copied from
        affect_drives_chat.AffectDrivesWorkspace._isolated -- the same #77 fix.)"""
        xp = None
        try:
            from sim.backend import get_backend
            xp, _ = get_backend()
        except Exception:
            xp = None
        host_np = np.random.get_state()
        host_xp = None
        if xp is not None and xp is not np:
            try:
                host_xp = xp.random.get_random_state().get_state()
            except Exception:
                host_xp = None
        if self._rng_state is None:
            np.random.seed(self.seed)
            if xp is not None and xp is not np:
                try:
                    xp.random.seed(self.seed)
                except Exception:
                    pass
        else:
            try:
                np.random.set_state(self._rng_state["np"])
            except Exception:
                pass
            if xp is not None and xp is not np and self._rng_state.get("xp") is not None:
                try:
                    xp.random.get_random_state().set_state(self._rng_state["xp"])
                except Exception:
                    pass
        try:
            return fn()
        finally:
            st = {"np": np.random.get_state(), "xp": None}
            if xp is not None and xp is not np:
                try:
                    st["xp"] = xp.random.get_random_state().get_state()
                except Exception:
                    st["xp"] = None
            self._rng_state = st
            try:
                np.random.set_state(host_np)
            except Exception:
                pass
            if host_xp is not None:
                try:
                    xp.random.get_random_state().set_state(host_xp)
                except Exception:
                    pass

    def _ensure(self):
        """Lazy-build the #76 BG substrate once and snapshot its full post-build dynamic state (all cp_* ndarrays),
        so every read can restore to an identical starting point -> the read is a deterministic function of THIS
        turn's afferent, history-independent (the cross-turn persistence lives in the host engagement EMA).

        The snapshot filter used to be `isinstance(..., np.ndarray)`, which is FALSE for every `cp_*` attr on a
        cupy-backed substrate (`SIM_BACKEND=cupy`, the production `/api/brain-chat` path) -> the snapshot dict
        came back EMPTY and `_restore()` silently became a no-op (a second, non-crashing bug alongside the
        `.fill()` ValueError this module's cupy-interop fix addresses -- see
        `research/findings/2026-08-25-da-axis-cupy-interop-fix.md`). `_is_ndarray` recognizes cupy arrays too,
        and `.copy()` (available on both numpy and cupy ndarrays) keeps each snapshot entry on ITS OWN array's
        backend, so `_restore()`'s `getattr(self._sb, k)[:] = v` assigns same-device-to-same-device on either
        backend -- never a host numpy array into a device cupy slice. Byte-identical on numpy (`.copy()` of a
        numpy array == the old `np.array(...)` copy)."""
        if self._sb is None:
            sb, regions, _ = _PM.build(self.seed)
            self._sb = sb
            self._nbt = _PM.names_by_type(regions)
            self._snapshot = {k: getattr(sb, k).copy() for k in dir(sb)
                              if k.startswith("cp_") and _is_ndarray(getattr(sb, k, None))}

    def _restore(self):
        for k, v in self._snapshot.items():
            getattr(self._sb, k)[:] = v

    def _read_da_level(self, afferent_pa: float, lesion: bool) -> tuple:
        """One neural read: restore the post-build substrate state, make a fresh `dopamine_mode` manager (so the DA
        concentration starts at tonic), and run the #76 live SNc->DA loop under this afferent. Returns the
        self-produced DA concentration + the SNc firing fraction. `lesion` silences the SNc nucleus (#76
        anti-cheat-2) so the level collapses regardless of the afferent."""
        self._restore()
        mgr = _DA.make_manager(self._sb)
        _, conc, sncf = _DA.measure_self_driven(self._sb, mgr, self._nbt, _DA.BASELINE,
                                                float(afferent_pa), perturb=None, silence_snc=bool(lesion))
        return float(conc), float(sncf)

    def observe(self, message: str, *, lesion: bool = False,
                afferent_override: Optional[float] = None) -> dict:
        """Fold the message engagement into the persistent EMA, map it to the SNc reward/context afferent, run one
        neural SNc->DA read, and return the self-produced DA level + the mode + the engagement suffix. An empty /
        content-free turn HOLDS the prior engagement (persistence). An `afferent_override` (pA) drives the SNc
        afferent DIRECTLY (a mode INDUCTION, for the (B) proof: vary the mode with the message held fixed). `lesion`
        silences the SNc nucleus (the load-bearing lesion). Never raises out (the caller degrades to no-suffix)."""
        with self._lock:
            self.n_turns += 1
            tokens = _content_tokens(message)
            shared_info = None
            if afferent_override is not None:
                afferent = float(afferent_override)
                turn_e = None
            else:
                if tokens:
                    turn_e = engagement_of(tokens, self.seen)
                    # ── shared spiking novelty/salience afferent (rank-4, default-OFF) ─────────────────────────
                    # Route the SAME raw host engagement scalar through the shared ASK-pool spiking transduction
                    # BEFORE it folds into the EMA, instead of using the raw host arithmetic directly. `engagement_
                    # of`'s OUTPUT (the sensory/comprehension boundary read) is unchanged; only what mediates it on
                    # the way to the SNc afferent changes. OFF (unset) -> turn_e unchanged -> byte-identical.
                    if _SHARED.shared_salience_enabled():
                        shared_info = _SHARED.read_salience(turn_e, seed=self.seed)
                        turn_e = float(np.clip(shared_info["normalized"], 0.0, 1.0))
                    d = _EMA_DECAY
                    self.ema_engagement = d * self.ema_engagement + (1.0 - d) * turn_e
                else:
                    turn_e = None      # content-free turn -> HOLD the prior engagement (cross-turn persistence)
                afferent = float(np.clip(self.ema_engagement, 0.0, 1.0)) * _MAX_AFFERENT_PA
            # update the session vocabulary AFTER computing novelty (so this turn's tokens are novel this turn).
            for t in tokens:
                self.seen.add(t)

            info = {"acted": False, "turn": self.n_turns, "lesioned": bool(lesion),
                    "turn_engagement": (None if turn_e is None else float(turn_e)),
                    "ema_engagement": float(self.ema_engagement), "afferent_pA": float(afferent),
                    "da_level": 0.0, "snc_firing": 0.0, "mode": "rest", "lead": "", "reason": None,
                    "seed": self.seed}
            if shared_info is not None:      # key present ONLY when the shared afferent ran (byte-identical-off idiom)
                info["shared_salience"] = shared_info
            try:
                self._isolated(self._ensure)
                conc, sncf = self._isolated(lambda: self._read_da_level(afferent, lesion))
                mode = da_to_mode(conc)
                suffix = mode_suffix(mode)
                info.update({"acted": True, "da_level": conc, "snc_firing": sncf, "mode": mode, "lead": suffix,
                             "reason": ("lesion_collapsed" if lesion and mode in ("rest", "neutral") else
                                        ("engaged" if suffix else "low_engagement"))})
            except Exception as e:   # never let the DA read crash / change a turn
                info["reason"] = f"error:{type(e).__name__}: {e}"
            return info

    def relax_idle(self, relax: float, neutral: float = 0.0) -> dict:
        """ONE IDLE-TICK relaxation step (board #92, 2026-08-26) — the continuous-engine companion to `observe()`,
        mirroring `affect_drives_chat.AffectDrivesWorkspace.relax_idle` (board #91) on the ENGAGEMENT/AROUSAL axis
        instead of valence: decay the persistent `ema_engagement` toward the neutral set-point (the SAME homeostatic
        relaxation formula `continuous_engine.tick_session` already applies elsewhere, `v1 = NEUTRAL +
        (v0-NEUTRAL)*RELAX` — reused verbatim, not reinvented), map the decayed EMA to the SNc reward/context
        afferent, and re-run ONE #76 neural SNc->DA READ at the relaxed point. So the engagement/mode this coupling
        reports between turns is a genuine spiking read AT THE RELAXED afferent — never a host time-based formula
        computing the mode/suffix directly. Does NOT increment `n_turns` (an idle tick is not a conversational turn)
        and does NOT touch the hold/induction branching `observe()` uses — a distinct, idempotent decay step the
        idle loop calls once per tick, applied directly to the SAME `ema_engagement` state a live `observe()` turn
        reads and writes next.

        BIOLOGICAL GROUNDING (the same LC-NE / tonic-DA vigor account this module's docstring cites): without a
        salient/novel afferent, engagement is not held at its last value forever — arousal/vigor relaxes back toward
        a resting baseline over time. Today (pre-#92) `ema_engagement` was written ONLY inside a live `observe()`
        and a content-free turn merely HOLDS it — an idle session with no turns at all never touched it either, so
        telling the brain something highly engaging, waiting idle, then sending a neutral follow-up produced the
        IDENTICAL engagement suffix as zero idle time (the same observe-vs-drive gap #91 closed for #84's mood).

        Returns the same-shaped record `observe()` returns (mode/lead/da_level/...) plus `relaxed: True`, so a
        caller can log an inner-life note. Never raises out (mirrors `observe()`'s never-crash contract) — on any
        error the mode/lead stay at the inert default (rest / '')."""
        with self._lock:
            self.ema_engagement = float(neutral) + (float(self.ema_engagement) - float(neutral)) * float(relax)
            afferent = float(np.clip(self.ema_engagement, 0.0, 1.0)) * _MAX_AFFERENT_PA

            info = {"acted": False, "relaxed": True, "turn": self.n_turns, "lesioned": False,
                    "turn_engagement": None, "ema_engagement": float(self.ema_engagement),
                    "afferent_pA": float(afferent), "da_level": 0.0, "snc_firing": 0.0, "mode": "rest", "lead": "",
                    "reason": None, "seed": self.seed}
            try:
                self._isolated(self._ensure)
                conc, sncf = self._isolated(lambda: self._read_da_level(afferent, False))
                mode = da_to_mode(conc)
                suffix = mode_suffix(mode)
                info.update({"acted": True, "da_level": conc, "snc_firing": sncf, "mode": mode, "lead": suffix,
                             "reason": "idle_relax"})
            except Exception as e:   # never let the idle-relax read crash the tick loop
                info["reason"] = f"error:{type(e).__name__}: {e}"
            return info


def get_workspace(chat, *, seed: int = _DEFAULT_SEED) -> DaModeDrivesWorkspace:
    """Idempotently attach a per-session `DaModeDrivesWorkspace` to the cached ChatBrain (auto-cleared on session
    reset, which drops the ChatBrain). No `sim/` edit; the ChatBrain instance is a host scaffold."""
    ws = getattr(chat, "_da_drives_workspace", None)
    if ws is None:
        ws = DaModeDrivesWorkspace(seed=seed)
        chat._da_drives_workspace = ws
    return ws


def relax_idle(chat, relax: float, neutral: float = 0.0) -> Optional[dict]:
    """The IDLE-TICK entry point (board #92): relax THIS session's DA-mode engagement EMA toward neutral and
    re-read the #76 neural SNc->DA at the decayed point. Mirrors `affect_drives_chat.relax_idle` (board #91) on the
    ENGAGEMENT/AROUSAL axis: `continuous_engine`'s "the brain keeps feeling between turns" mechanism previously
    never reached this flagship, default-ON, most-visible engagement->forthcomingness coupling (the #76/#79 DA-mode
    suffix this module drives) — closing an observe-vs-drive gap on a SECOND axis (#91 closed it for valence/warmth;
    this closes it for engagement/energy): tell the brain something highly engaging, wait idle, then send a neutral
    follow-up — BEFORE this, the DA-mode suffix on return was IDENTICAL to zero idle time, because nothing between
    turns ever touched `ema_engagement`.

    Returns None (a clean no-op) when this session has no `_da_drives_workspace` yet — i.e. the DA-mode coupling
    was never triggered on a live turn — so a session that never had a DA-drives turn is BYTE-IDENTICAL: idling it
    does nothing new, exactly like today. Never raises (delegates to `DaModeDrivesWorkspace.relax_idle`, itself
    never-raising)."""
    ws = getattr(chat, "_da_drives_workspace", None)
    if ws is None:
        return None
    try:
        return ws.relax_idle(relax, neutral=neutral)
    except Exception as e:
        return {"acted": False, "relaxed": True, "reason": f"error:{type(e).__name__}: {e}", "lead": "", "mode": "rest"}


def observe_turn(chat, message: str, *, seed: int = _DEFAULT_SEED) -> dict:
    """The production entry point: read this turn's engagement, drive the spiking SNc nucleus, read the self-produced
    DA LEVEL off the bus, bin it to a MODE (rest/focus/arousal), and map the mode to an engagement SUFFIX. Returns
    the per-turn `da_drives` info (also stashed on `chat._last_da_drives`). Never raises out (on any error it returns
    an inert no-suffix info dict so a turn can never crash).

    MODE-INDUCTION affordance (for the (B) load-bearing proof + a live mode-set): `BRAIN_DA_DRIVES_INDUCE="<pA>"`
    drives the SNc reward/context afferent DIRECTLY (e.g. 100 -> rest, 800 -> focus, 1300 -> arousal) so the mode can
    be varied with the MESSAGE HELD FIXED (exactly the (B) design) -- the neural SNc->DA read still runs on that
    induced afferent and the lesion still collapses it."""
    try:
        afferent_override = None
        _ind = os.environ.get("BRAIN_DA_DRIVES_INDUCE")
        if _ind:
            try:
                afferent_override = float(_ind)
            except Exception:
                afferent_override = None
        ws = get_workspace(chat, seed=seed)
        info = ws.observe(message, lesion=da_drives_lesioned(), afferent_override=afferent_override)
        info["on"] = True
    except Exception as e:
        info = {"on": True, "acted": False, "reason": f"error:{type(e).__name__}: {e}", "lead": "", "mode": "rest"}
    chat._last_da_drives = info
    return info
