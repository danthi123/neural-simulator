"""The #81 GRADED-AFFECT ladder wired into the LIVE `/api/brain-chat` turn so the brain's felt valence x arousal is
LOAD-BEARING on what/how it responds -- NOT observe-only (board #84, INTEGRATION-TO-PRODUCTION).

WHAT THIS IS. Board #81 landed a 6/6-seed GO graded-affect substrate (`_graded_affect_attractor_derisk`): a
Koulakov-2002/Goldman-2003 robust-integrator LADDER of independently-latched bistable NMDA sub-pools reads the
brain's #49 interoceptive body-state as a SMOOTH valence x arousal (Pearson +0.97 / +0.95; the embodiment lesion --
cutting the interoceptive->ladder synapses -- collapses it to 0). That GO shipped as a DEFAULT-OFF de-risk RUNNER,
never in the live chat. This module wires that NEURAL read onto the live conversational brain and makes it CHANGE
the response: the felt affect state colors the AFFECTIVE EXPRESSION the reply leads with (a graded warmth/curtness
marker), and the forthcomingness the reply plans for. It is the anti-hollow-integration counterpart to the
observe-only faculties: the affect READ is neural AND it demonstrably shapes the surface.

THE READ (the #81 neural mechanism, reused-by-import; NO sim/ edit).
  * Each turn, the message's affective valence/arousal (the SAME host-comprehension boundary the SVO question parser
    and the Gate-B appraisal occupy -- `affect_production_organ.appraise_text`, DR-2 learned distributional valence)
    is EMA-folded into a persistent per-session BODY-STATE (h = comfort/homeostasis, a = bodily arousal). A neutral
    turn (no strongly-affective word) HOLDS the prior body-state -> cross-turn affect PERSISTENCE.
  * That body-state drives the #81 ladder through the interoceptive relays (`read_body`), and the FELT state is the
    ladder's OWN population read off `cp_firing_states`: mood = rate(V+ ladder) - rate(V- ladder) (graded valence),
    felt_arousal = rate(arousal ladder). NEVER a host formula. Positive body-state (h>0.5) latches more V+ sub-pools
    (mood -> +); negative (h<0.5) latches more V- (mood -> -); the set-point h=0.5 reads ~0.
  * The graded mood differential is binned into a Koulakov staircase LEVEL (-3..+3) from calibrated thresholds on
    the #81 mood scale (mood ranges ~[-0.08, +0.08]; the set-point band is neutral).

THE COUPLING (what makes it LOAD-BEARING, not observe-only).
  * VALENCE -> the affective EXPRESSION the reply leads with: a graded warmth/curtness discourse marker prepended to
    the answer surface ("Wonderful — <fact>", "Sure — <fact>", neutral -> no lead, "Honestly — <fact>",
    "Frankly — <fact>"). AROUSAL -> the marker's emphasis (high felt-arousal -> "! ", else " — "). The marker is an
    honest EXPRESSION of the read affect state (tone-of-voice / prosody the body renders), NOT content: the FACT
    after it is the SAME gate-matched, moat-verified SVO, and the VERIFY re-parse is unchanged. So affect changes
    HOW the reply sounds, never WHICH fact is true and never whether an unmatched cue abstains. (This is the single
    coupling this module wires; the Gate-B `BRAIN_AFFECT` path independently colors prose-manner + forthcomingness.)

THE HONESTY FLOOR (preserved BY CONSTRUCTION, mirrors the Gate-B affect path).
  * The moat / recall / abstain verdict runs FIRST and unchanged; the affect coupling only DECORATES an
    already-matched answer surface. It never enters the certainty band, never manufactures a fact, never flips an
    abstain into an assert. The content fields (`abstained`, `recalled_svo`, `verified`) are BYTE-IDENTICAL with the
    coupling on or off; only the answer SURFACE (tone) and the additive `affect_drives` trace change.

LESION (the load-bearing / brain-based proof). `BRAIN_AFFECT_DRIVES_LESION=1` cuts the interoceptive->ladder
synapses (`intero_out` gate=0, the #81 embodiment lesion) on every read -> the neural mood differential collapses
to ~0 -> the staircase level is 0 -> the affective lead VANISHES and the answer surface reverts to the neutral
(coupling-off) surface. So the surface change RIDES the SPIKING ladder read, not a host `if valence>0`: kill the
neural read and the tone-difference disappears.

CONTRACT (additive, reversible, byte-identical-off).
  * `affect_drives_enabled()` gates the whole block. When DISABLED the handler skips it entirely: no workspace is
    built, no read runs, no `affect_drives` key is attached, and NO affective lead is prepended -> the turn is
    BYTE-IDENTICAL to pre-wiring (the Gate-B affect path, if on, is untouched -- this module is orthogonal to it).
  * The ladder is run on the workspace's PRIVATE RNG timeline and the host process-global RNG (numpy + the sim
    backend) is restored around every read (the #77 global-RNG footgun): enabling this module cannot perturb the
    downstream RNG-dependent organs, so the OTHER response fields stay byte-identical.
  * The ladder build (~0.4s) is lazy on the first turn per session and kept warm; each turn runs one ~0.15s read.

REUSE-BY-IMPORT (NO `sim/` edit). The graded-affect ladder build (`GradedAffectBrain`), the neural body->felt read
(`read_body`) and the operating point come STRAIGHT from `_graded_affect_attractor_derisk` (board #81, 6/6-seed GO).
The appraisal comes from `affect_production_organ.appraise_text` (the Gate-B host-comprehension boundary). This
module adds only the production glue (the per-session body-state register + the level->expression map). `git diff
sim/` is empty.

HONEST RESIDUALS (named, not claimed closed).
  1. The message->valence APPRAISAL is host (a language-comprehension boundary, like the SVO parser). The felt READ
     (body-state -> graded valence x arousal off cp_firing_states) and its embodiment dependence ARE the #81 neural
     mechanism (lesion-proven). The body-state VARIABLES (h, a) are the standard body boundary.
  2. The level->EXPRESSION-MARKER map is a HOST conditioned-articulation scaffold (the "mouth"): the affect that
     DRIVES it is the neural ladder read (load-bearing -- the lesion collapses the marker), but the surface STRING
     for a given level is a host template, exactly the sanctioned articulation-crutch pattern (owner:
     scaffold-ok-as-conditioned-articulation IF the faculty is load-bearing on the tone, which the lesion proves).
     A brain-native affective mouth (the marker itself emitted by a spiking prosody circuit) is the named next rung.
  3. This module reads its OWN co-resident #81 ladder bridge, run ALONGSIDE the recall composer, not merged onto the
     single recall bridge (the one-brain consolidation step, shared with the Gate-B affect burn-down).
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

# reuse-by-import the board-#81 6/6-seed-GO graded-affect ladder (build + the neural body->felt read) -- NO sim/ edit.
from research.runners._graded_affect_attractor_derisk import (
    GradedAffectBrain as _GradedAffectBrain,
    read_body as _read_body,
    I_BODY_PA as _I_BODY_PA,
)

_DEFAULT_SEED = 42

# ── read windows (calibrated on the #81 ladder: (40,140,80) preserves the graded monotone mood(h) staircase at
#    ~0.13s/read vs the runner's (60,250,120)@0.16s -- the NMDA latches settle fast, so the shorter window is faithful).
_SETTLE_MS = 40
_ESTABLISH_MS = 140
_READ_MS = 80

# ── EMA body-state persistence: a strong induction turn dominates; a neutral turn (0 affective hits) HOLDS the prior
#    body-state (cross-turn affect persistence). Matches the Gate-B `_MOOD_EMA_DECAY` so the two affect paths agree.
_EMA_DECAY = 0.4

# ── mood -> graded Koulakov staircase LEVEL (-3..+3). Thresholds on the #81 mood scale (mood ~[-0.08,+0.08], the
#    set-point h=0.5 reads ~0). The neutral band keeps the answer surface byte-identical at a neutral mood.
_MOOD_NEUTRAL_TOL = 0.010    # |mood| below this -> level 0 (neutral: NO lead, surface unchanged)
_MOOD_L1 = 0.010
_MOOD_L2 = 0.045
_MOOD_L3 = 0.070
# ── bodily-arousal gain: the #81 arousal ladder only latches above body a~0.5 (felt ~0.04@a=0.5, ~0.065@a=0.7),
#    and appraised arousal (~0.6 for an affective word) is EMA-diluted, so map body a = clip(gain * ema_arousal) to
#    reach the felt-responsive band as affective arousal ACCUMULATES across turns (a single mild turn stays low).
_AROUSAL_GAIN = 1.5
# ── felt-arousal -> emphasis. felt_arousal ranges ~[0,0.075]; above this the affective marker is emphatic ("! ").
_AROUSAL_HIGH = 0.050

# ── the level -> affective EXPRESSION marker (the host conditioned-articulation scaffold; DRIVEN by the neural read).
_LEAD_WORD = {3: "Wonderful", 2: "Gladly", 1: "Sure", -1: "Hm", -2: "Honestly", -3: "Frankly"}


def affect_drives_enabled() -> bool:
    """The master flag. `BRAIN_AFFECT_DRIVES` truthy (1/true/on/yes) enables; 0/false/off/no disables. The default
    when the env var is UNSET follows the production-integration anchor `_AFFECT_DRIVES_DEFAULT_ON` in server.py --
    this reads only the explicit env override (server.py combines it with the anchor, mirroring the Gate-B flags)."""
    return os.environ.get("BRAIN_AFFECT_DRIVES", "0").strip().lower() in ("1", "true", "on", "yes")


def affect_drives_off() -> bool:
    """Explicit OFF (for a default-ON anchor): `BRAIN_AFFECT_DRIVES` in {0,false,no,off}."""
    v = os.environ.get("BRAIN_AFFECT_DRIVES")
    return v is not None and v.strip().lower() in ("0", "false", "no", "off", "")


def affect_drives_lesioned() -> bool:
    """`BRAIN_AFFECT_DRIVES_LESION` truthy -> cut the interoceptive->ladder synapses on every read (the #81
    embodiment lesion): the neural mood collapses to ~0, so the affective lead VANISHES. The load-bearing proof."""
    return os.environ.get("BRAIN_AFFECT_DRIVES_LESION", "0").strip().lower() in ("1", "true", "on", "yes")


def _valence_to_body(valence: float, arousal: float) -> tuple:
    """Map the appraised message affect to the #81 body-state. valence in [-1,1] -> comfort/homeostasis h in [0,1]
    (h = 0.5 + 0.5*valence: valence 0 -> the neutral set-point h=0.5; +1 -> comfort; -1 -> discomfort). arousal in
    [0,1] -> bodily arousal a. This is the body boundary; the felt READ off the ladder is the neural part."""
    v = float(np.clip(valence, -1.0, 1.0))
    a = float(np.clip(_AROUSAL_GAIN * arousal, 0.0, 1.0))
    return 0.5 + 0.5 * v, a


def mood_to_level(mood: float) -> int:
    """The graded valence LEVEL (-3..+3) from the neural ladder mood differential (the Koulakov staircase)."""
    m = float(mood)
    s = 1 if m > 0 else -1
    am = abs(m)
    if am < _MOOD_NEUTRAL_TOL:
        return 0
    if am >= _MOOD_L3:
        return 3 * s
    if am >= _MOOD_L2:
        return 2 * s
    return 1 * s


def expression_lead(level: int, high_arousal: bool) -> str:
    """The affective EXPRESSION marker for this turn's felt state (the conditioned-articulation scaffold; DRIVEN by
    the neural ladder read). Level 0 (neutral) -> '' so the surface is byte-identical. Non-neutral -> a graded
    warmth/curtness marker; high felt-arousal makes it emphatic ('! '), else measured (' — '). The FACT after it is
    unchanged (VERIFY re-parse intact) -- this colors HOW the reply sounds, never WHICH fact is true."""
    if int(level) == 0:
        return ""
    word = _LEAD_WORD.get(int(level))
    if not word:
        return ""
    return (word + "! ") if high_arousal else (word + " — ")


class AffectDrivesWorkspace:
    """A per-session graded-affect workspace: a persistent #81 ladder + an EMA body-state. `observe(valence, arousal,
    n_hits)` folds the appraisal into the body-state (a neutral turn HOLDS it), runs one neural ladder read, and
    returns the felt state + the graded level + the affective lead. The ladder build + read run on the workspace's
    PRIVATE RNG timeline (the host process-global RNG is restored around them -- the #77 footgun)."""

    def __init__(self, seed: int = _DEFAULT_SEED):
        self.seed = int(seed)
        self._brain = None
        self._lock = threading.Lock()
        self.h = 0.5           # persistent body-state: comfort/homeostasis (set-point 0.5)
        self.a = 0.0           # persistent body-state: bodily arousal
        self.ema_valence = 0.0
        self.ema_arousal = 0.0
        self.n_turns = 0
        self._rng_state = None  # the ladder's PRIVATE RNG timeline (the host process-global RNG is never advanced)

    def _isolated(self, fn):
        """Run `fn()` (the ladder build + spiking read) on the workspace's PRIVATE RNG timeline, leaving the host
        process-global RNG (numpy + the sim backend) BYTE-UNTOUCHED. The #81 build reseeds cfg.seed and its stepping
        draws OU noise off the SAME process-global RNG the rest of the pipeline shares -- without this, enabling this
        module would perturb the downstream RNG-dependent organs and break byte-identity. Snapshot the host RNG, swap
        in this workspace's own continuous timeline, run, capture the advanced private timeline, restore host. (Copied
        from gnw_thought_swap.ThoughtSwapWorkspace._isolated -- the same #77 fix.)"""
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
        if self._brain is None:
            self._brain = _GradedAffectBrain(self.seed)

    def observe(self, valence: float, arousal: float, n_hits: int, *,
                lesion: bool = False,
                valence_override: Optional[float] = None,
                arousal_override: Optional[float] = None) -> dict:
        """Fold the appraisal into the persistent body-state, run one neural ladder read, and return the felt state +
        graded level + the affective lead. n_hits==0 (a neutral turn) HOLDS the prior body-state (persistence). A
        `valence_override` / `arousal_override` sets the body-state directly (a mood INDUCTION, for the (B) proof:
        vary the affect state with the message fixed). `lesion` cuts the interoceptive->ladder synapses so the neural
        mood collapses (the load-bearing lesion). Never raises out (the caller degrades to no-lead)."""
        with self._lock:
            self.n_turns += 1
            if valence_override is not None or arousal_override is not None:
                if valence_override is not None:
                    self.ema_valence = float(valence_override)
                if arousal_override is not None:
                    self.ema_arousal = float(arousal_override)
            elif int(n_hits) > 0:
                d = _EMA_DECAY
                self.ema_valence = d * self.ema_valence + (1.0 - d) * float(valence)
                self.ema_arousal = d * self.ema_arousal + (1.0 - d) * float(arousal)
            # else: neutral turn -> HOLD the prior EMA (cross-turn persistence)
            self.h, self.a = _valence_to_body(self.ema_valence, self.ema_arousal)

            info = {"acted": False, "turn": self.n_turns, "lesioned": bool(lesion),
                    "ema_valence": float(self.ema_valence), "ema_arousal": float(self.ema_arousal),
                    "body_h": float(self.h), "body_a": float(self.a),
                    "mood": 0.0, "felt_arousal": 0.0, "level": 0, "high_arousal": False, "lead": "",
                    "reason": None, "seed": self.seed}
            try:
                self._isolated(self._ensure)
                r = self._isolated(lambda: _read_body(self._brain, self.h, self.a, _I_BODY_PA,
                                                      settle=_SETTLE_MS, establish=_ESTABLISH_MS, read=_READ_MS,
                                                      lesion_gate=bool(lesion)))
                mood = float(r["mood"])
                felt = float(r["felt_arousal"])
                level = mood_to_level(mood)
                high = bool(felt > _AROUSAL_HIGH)
                lead = expression_lead(level, high)
                info.update({"acted": True, "mood": mood, "felt_arousal": felt, "level": int(level),
                             "high_arousal": high, "lead": lead,
                             "vplus_rate": float(r.get("vplus_rate", 0.0)),
                             "vminus_rate": float(r.get("vminus_rate", 0.0)),
                             "reason": ("lesion_collapsed" if lesion else
                                        ("neutral_hold" if level == 0 else "graded_affect"))})
            except Exception as e:   # never let the affect read crash / change a turn
                info["reason"] = f"error:{type(e).__name__}: {e}"
            return info


    def relax_idle(self, relax: float, neutral: float = 0.0) -> dict:
        """ONE IDLE-TICK relaxation step (board #91, 2026-08-26) -- the continuous-engine companion to `observe()`:
        decay the persistent EMA body-state toward the neutral set-point (the SAME homeostatic relaxation formula
        `continuous_engine.tick_session` already applies to the legacy Gate-B mood, `v1 = NEUTRAL + (v0-NEUTRAL)*RELAX`
        -- reused verbatim, not reinvented), recompute the body-state (h, a), and re-run ONE neural #81 ladder READ at
        the DECAYED point. So the felt mood this coupling reports between turns is a genuine spiking read AT THE
        RELAXED body-state -- never a host time-based formula computing the level/lead directly. Does NOT increment
        `n_turns` (an idle tick is not a conversational turn) and does NOT touch the induction/hold branching
        `observe()` uses -- it is a distinct, idempotent decay step the idle loop calls once per tick, applied
        directly to the SAME `ema_valence`/`ema_arousal`/`h`/`a` state a live `observe()` turn reads and writes next.

        Returns the same-shaped record `observe()` returns (mood/felt_arousal/level/lead/...) plus `relaxed: True`,
        so a caller can log an inner-life note ("my felt warmth is fading toward neutral"). Never raises out (mirrors
        `observe()`'s never-crash contract) -- on any error the level/lead stay at the inert default (0/'')."""
        with self._lock:
            self.ema_valence = float(neutral) + (float(self.ema_valence) - float(neutral)) * float(relax)
            self.ema_arousal = float(self.ema_arousal) * float(relax)
            self.h, self.a = _valence_to_body(self.ema_valence, self.ema_arousal)

            info = {"acted": False, "relaxed": True, "turn": self.n_turns, "lesioned": False,
                    "ema_valence": float(self.ema_valence), "ema_arousal": float(self.ema_arousal),
                    "body_h": float(self.h), "body_a": float(self.a),
                    "mood": 0.0, "felt_arousal": 0.0, "level": 0, "high_arousal": False, "lead": "",
                    "reason": None, "seed": self.seed}
            try:
                self._isolated(self._ensure)
                r = self._isolated(lambda: _read_body(self._brain, self.h, self.a, _I_BODY_PA,
                                                      settle=_SETTLE_MS, establish=_ESTABLISH_MS, read=_READ_MS,
                                                      lesion_gate=False))
                mood = float(r["mood"])
                felt = float(r["felt_arousal"])
                level = mood_to_level(mood)
                high = bool(felt > _AROUSAL_HIGH)
                lead = expression_lead(level, high)
                info.update({"acted": True, "mood": mood, "felt_arousal": felt, "level": int(level),
                             "high_arousal": high, "lead": lead,
                             "vplus_rate": float(r.get("vplus_rate", 0.0)),
                             "vminus_rate": float(r.get("vminus_rate", 0.0)),
                             "reason": "idle_relax"})
            except Exception as e:   # never let the idle-relax read crash the tick loop
                info["reason"] = f"error:{type(e).__name__}: {e}"
            return info


def get_workspace(chat, *, seed: int = _DEFAULT_SEED) -> AffectDrivesWorkspace:
    """Idempotently attach a per-session `AffectDrivesWorkspace` to the cached ChatBrain (auto-cleared on session
    reset, which drops the ChatBrain). No `sim/` edit; the ChatBrain instance is a host scaffold."""
    ws = getattr(chat, "_affect_drives_workspace", None)
    if ws is None:
        ws = AffectDrivesWorkspace(seed=seed)
        chat._affect_drives_workspace = ws
    return ws


def relax_idle(chat, relax: float, neutral: float = 0.0) -> Optional[dict]:
    """The IDLE-TICK entry point (board #91): relax THIS session's #84 affect-drives EMA toward neutral and re-read
    the neural ladder at the decayed point. `continuous_engine`'s headline mechanism -- "the felt mood keeps
    evolving while idle" -- previously reached ONLY the legacy Gate-B affect path (`_SESSION_MOOD` +
    `_get_affect_organ().read_differential`); this extends the SAME idle-relax idea to the flagship, default-ON,
    most user-visible affect->tone coupling (the #84 lead marker this module drives), closing an observe-vs-drive
    gap: tell the brain something emotionally charged, wait idle, then send a neutral follow-up -- BEFORE this, the
    #84 lead on return was IDENTICAL to zero idle time, because the thing that decays (`_SESSION_MOOD`) never fed
    #84.

    Returns None (a clean no-op) when this session has no `_affect_drives_workspace` yet -- i.e. #84 was never
    triggered on a live turn -- so a session that never had an affect-drives turn is BYTE-IDENTICAL: idling it
    does nothing new, exactly like today. Never raises (delegates to `AffectDrivesWorkspace.relax_idle`, itself
    never-raising)."""
    ws = getattr(chat, "_affect_drives_workspace", None)
    if ws is None:
        return None
    try:
        return ws.relax_idle(relax, neutral=neutral)
    except Exception as e:
        return {"acted": False, "relaxed": True, "reason": f"error:{type(e).__name__}: {e}", "lead": "", "level": 0}


def observe_turn(chat, message: str, appraisal: Optional[dict] = None, *,
                 seed: int = _DEFAULT_SEED,
                 valence_override: Optional[float] = None,
                 arousal_override: Optional[float] = None) -> dict:
    """The production entry point: appraise this turn's message (reuse the Gate-B `affect_production_organ` DR-2
    appraisal unless one is passed in), fold it into the per-session body-state, run one neural #81 ladder read, and
    return the per-turn `affect_drives` info (also stashed on `chat._last_affect_drives`). Never raises out (on any
    error it returns an inert no-lead info dict so a turn can never crash).

    MOOD-INDUCTION affordance (for the (B) load-bearing proof + a live mood-set): `BRAIN_AFFECT_DRIVES_INDUCE="v"`
    or `"v,a"` sets the body-state directly (valence v in [-1,1], arousal a in [0,1]) so the affect state can be
    varied with the MESSAGE HELD FIXED (a mood induction, exactly the (B) design) -- the neural ladder read still
    runs on that induced body-state and the lesion still collapses it. An explicit override arg takes precedence."""
    try:
        if valence_override is None and arousal_override is None:
            _ind = os.environ.get("BRAIN_AFFECT_DRIVES_INDUCE")
            if _ind:
                try:
                    parts = [float(x) for x in str(_ind).split(",")]
                    valence_override = parts[0]
                    if len(parts) > 1:
                        arousal_override = parts[1]
                except Exception:
                    pass
        if appraisal is None:
            from research.runners import affect_production_organ as _AO
            appraisal = _AO.appraise_text(message)
        ws = get_workspace(chat, seed=seed)
        info = ws.observe(float(appraisal.get("valence", 0.0)), float(appraisal.get("arousal", 0.0)),
                          int(appraisal.get("n_hits", 0)), lesion=affect_drives_lesioned(),
                          valence_override=valence_override, arousal_override=arousal_override)
    except Exception as e:
        info = {"acted": False, "reason": f"error:{type(e).__name__}: {e}", "lead": "", "level": 0}
    chat._last_affect_drives = info
    return info
