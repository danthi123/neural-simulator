"""AFFECT / EMOTION wired into the PRODUCTION conversational turn (Gate-B, 2026-08-12).

This is the production-integration glue that makes the brain's MOOD genuinely color WHAT it chooses to say
(mood-congruent forthcomingness — how much it volunteers) AND HOW it says it (the fluent mouth's PROSE is warmer
/ curter), default-ON, moat-safe, lesion-load-bearing. It REUSES (does not reinvent) the adversarially-verified
Stage-A affect faculty:

  * the persistent spiking mood organ = the co-resident brain built by
    `_stageA_full_integration_derisk.build_one_brain(seed, with_faculties=True, co_resident_affect_ladder=True)`
    -- ONE SimulationBridge carrying the staggered-bistable-ladder graded-affect slice (Koulakov robust integrator)
    + the honesty relay + the 3-way arbiter, all co-resident. 6-seed GO for the valence SIGN + a graded bistable
    LADDER (2026-08-08). We READ the held mood NEURALLY as the population-rate differential
    rate(aff_pos_readout) - rate(aff_neg_readout) through the `affect_out` transmission gate.
  * the appraisal VALUE source = the DR-2 LEARNED distributional valence (2026-08-12). Each strongly-affective word's
    valence/arousal is now sourced from a cached LEAVE-ONE-OUT learned map (`_affect_learned_valence_map.json`,
    built by `_affect_distributional_tag_derisk.build_learned_valence_map` -- seed-clamped label-propagation over the
    LEARNED co-occurrence graph, every word inferred from all OTHERS so no word carries its own hand-assigned norm;
    6-seed held-out GO r~0.81). This RETIRES the hardcoded per-word value LOOKUP (the audit's #1 over-credit: the
    injection was a raw Warriner-lexicon read mis-labelled as "DR-2 learned"). Default-ON; `BRAIN_AFFECT_DR2=0`
    reverts to the raw norm value (byte-identical oracle). HONEST RESIDUALS (declared): (a) the affect-word SALIENCE
    GATE (which words move the mood) + the SEED norms are still Warriner -- DR-2 is SEEDED from them, it does NOT
    retire the lexicon; (b) the learning is numpy PPMI + label-prop, NOT spiking; (c) the fully-spiking on-bridge
    opponent V+/V- appraisal population is the named next rung. The injection into the ladder is still host; the
    READ-BACK through affect_out is the load-bearing spiking part.

THE HONESTY FLOOR is preserved BY CONSTRUCTION: affect is applied ONLY as (a) forthcomingness (how many already-
gate-matched, moat-verified facts to volunteer) and (b) prose MANNER (phrasing/warmth of a sentence whose SVO the
VERIFY re-parse still confirms). It NEVER enters the certainty band, NEVER manufactures a fact, and NEVER flips an
abstain into an assert -- the moat (query_patient / the direct gate) runs FIRST and unchanged; affect colors only a
MATCHED answer. The `affect_out` LESION (set_transmission_gate('affect_out', 0)) collapses the neural differential
to ~0, which collapses BOTH the extra content and the manner back to neutral while the matched-fact CONTENT is
byte-identical and the abstain behaviour is unchanged -- the load-bearing + moat-safe proof.

HONEST RESIDUALS (declared, ride existing burn-down items):
  * The affect organ is a co-resident affect/honesty/arbiter substrate on ITS OWN bridge, run ALONGSIDE the
    production recall composer (the tiny-demo / developed-brain onebrain composer), not merged onto the single
    recall bridge. Merging onto the ONE recall bridge is the remaining one-brain consolidation step (burn-down #1).
  * Mood-conditioning the EXTERNAL Qwen mouth is host-mediated until the mouth is brain-native (burn-down A1).
  * The appraisal injection (per-word valence -> neuromodulator concentration) is a declared host scaffold.
  * The held-value is a GRADED bistable LADDER (quantized sign + level), NOT a smooth magnitude continuum.

Backend: uses the process backend (cupy in production, numpy in tests) -- NO global-backend flip (the cupy/numpy
flip bug). NO `sim/` edit; additive; default-ON with the `BRAIN_AFFECT` env escape for the byte-identical oracle.
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

# The Warriner-approximate affect-salience NORMS (the affect-word vocabulary + the seed norms) + the stop set.
# These are the SEED/gate host scaffold; the per-word appraisal VALUE now comes from the DR-2 learned map below.
from research.runners._affect_distributional_tag_derisk import WARRINER, STOP
# The validated Stage-A co-resident affect brain + the graded-ladder constants + helpers (reuse-by-import).
from research.runners import _stageA_full_integration_derisk as SA
from research.runners._gnw_rung1_ignition_curve_derisk import _restore_state
from sim.backend import get_backend, to_host


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# APPRAISAL. The affect-word SALIENCE GATE (which words move the mood) is the Warriner norm vocabulary -- only
# STRONGLY-affective words pass, so a plain fact query reads ~neutral (mildly-valenced content/action words like
# dog/cat/sit are filtered). The per-word VALUE is now sourced from the DR-2 LEARNED map (experience-derived),
# NOT the raw hardcoded value. `BRAIN_AFFECT_DR2=0` reverts the VALUE to the raw norm (byte-identical oracle).
# WHY the value-only swap (honest, measured 2026-08-12): distributional valence genuinely bleeds affect onto
# high-frequency action words (sit/run/jump/play/cat learn positive valence >= real affect words in TinyStories),
# so a FULL drop-in (learned value AND learned gate) would color a plain "what does the cat eat" -- breaking the
# neutral-fact invariant. The norm-gate preserves neutral-default; the learned VALUE is the genuine provenance step.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
_WORD_RE = re.compile(r"[a-zA-Z']+")
_STRONG_MARGIN = 2.0   # |valence - 5| >= this to count as an affective word (filters mild content nouns)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LEARNED_MAP_PATH = os.path.join(_REPO, "research", "findings", "raw", "_affect_learned_valence_map.json")
_LEARNED_VALENCE: dict | None = None   # lazily-loaded {word: (v9, a9)} DR-2 leave-one-out learned map


def dr2_enabled() -> bool:
    """Default-ON. `BRAIN_AFFECT_DR2` in {0,false,no,off} -> the appraisal VALUE reverts to the raw Warriner norm
    (the byte-identical oracle for the pre-DR-2 hardcoded-lexicon path)."""
    v = os.environ.get("BRAIN_AFFECT_DR2")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def _get_learned_valence() -> dict:
    """The cached DR-2 LEARNED valence map (built by build_learned_valence_map). Loaded once; a missing artifact
    -> {} (appraise falls back to the raw norm, declared)."""
    global _LEARNED_VALENCE
    if _LEARNED_VALENCE is None:
        try:
            with open(_LEARNED_MAP_PATH, encoding="utf-8") as fh:
                d = json.load(fh)
            raw = d.get("map", d)
            _LEARNED_VALENCE = {str(w).lower(): (float(v[0]), float(v[1])) for w, v in raw.items()}
        except Exception:
            _LEARNED_VALENCE = {}
    return _LEARNED_VALENCE


def appraise_text(text: str) -> dict:
    """Appraise a message. Returns {'valence': [-1,1], 'arousal': [0,1], 'n_hits', 'words'}. A word MOVES the mood
    iff it is strongly affect-bearing in the Warriner norm (the salience gate); its VALUE is the DR-2 LEARNED
    valence/arousal (experience-derived; raw norm iff BRAIN_AFFECT_DR2=0 or the word is not in the learned map).
    valence = mean over gated words of (v9 - 5)/4; arousal = mean (a - 1)/8. n_hits==0 -> a neutral message (the
    caller HOLDS the prior mood, giving cross-turn persistence)."""
    toks = [w.lower() for w in _WORD_RE.findall(text or "")]
    use_learned = dr2_enabled()
    learned = _get_learned_valence() if use_learned else {}
    vals, ars, hits = [], [], []
    for w in toks:
        if w in STOP or w not in WARRINER:
            continue
        v9_norm, a9_norm = WARRINER[w]
        if abs(v9_norm - 5.0) < _STRONG_MARGIN:   # not strongly affective -> ignore (keeps neutral queries neutral)
            continue
        v9, a9 = learned.get(w, (v9_norm, a9_norm)) if use_learned else (v9_norm, a9_norm)
        vals.append((v9 - 5.0) / 4.0)             # signed valence in [-1, 1]
        ars.append((a9 - 1.0) / 8.0)              # arousal in [0, 1]
        hits.append(w)
    if not vals:
        return {"valence": 0.0, "arousal": 0.0, "n_hits": 0, "words": []}
    return {"valence": float(np.mean(vals)), "arousal": float(np.mean(ars)),
            "n_hits": len(vals), "words": hits}


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# THE SPIKING AFFECT ORGAN -- the co-resident graded-affect ladder; read NEURALLY (sign-aware).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class AffectProductionOrgan:
    """A process-shared spiking mood organ. Built ONCE (lazily) via build_one_brain(co_resident_affect_ladder=True);
    each read RAMPS the appraisal into the ladder (POSITIVE appraisal drives the V+ rungs, NEGATIVE drives the V-
    rungs), holds via the within-pool NMDA latches, and reads the held differential rate(pos_readout)-rate(neg_readout)
    through the `affect_out` gate. Snapshot/restore-isolated (a read leaves the bridge unchanged)."""

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self._built = False
        self.bridge = self.comp = self.idx = self.snap = self.xp = None

    def ensure_built(self):
        if self._built:
            return
        self.bridge, self.comp, self.idx, self.snap = SA.build_one_brain(
            self.seed, with_faculties=True, co_resident_affect_ladder=True)
        self.xp, _ = get_backend()
        self._built = True

    @property
    def n_regions(self) -> int:
        try:
            return len(self.bridge.core_config.brain_regions)
        except Exception:
            return -1

    def read_differential(self, appraisal: float, lesion: bool = False,
                          ramp_ms: int = SA.LAD_RAMP_MS, drive_off_ms: int = SA.LAD_DRIVE_OFF_MS,
                          read_ms: int = SA.LAD_READ_MS) -> dict:
        """SIGN-AWARE neural ladder read. Mirrors SA.read_affect_ladder but drives the V+ ladder for a positive
        appraisal and the V- ladder for a negative one, so the held differential is genuinely signed. `lesion`
        clamps affect_out=0 -> the readout collapses (the load-bearing proof). Returns the differential + rates.

        BOARD-#84 ADAPTATION (2026-09-05, PRODUCTION-DEFAULT-ON as of the same-day flip; scaffold-retirement
        backlog rank 5): `appraisal_interoceptive_enabled()` routes this call through `research.runners.
        _appraisal_interoceptive_ladder_derisk.AppraisalInteroceptiveLadder` -- the SAME ladder spec (reused by
        import) driven by a real interoceptive-relay CURRENT afferent (the board #49/#81 pattern) instead of the
        `nm.set_concentration(...)` write below. Default (env var unset) -> NOW takes this branch (the new
        production default). The ESCAPE HATCH (`BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE=0`) -> this method's body
        below runs, COMPLETELY UNCHANGED -- byte-identical-to-pre-flip by construction (a distinct code path,
        not a shared one with new parameters), re-verified post-flip against the pre-recorded 6-seed host
        reference. See `research/findings/2026-09-05-gateB-appraisal-interoceptive-afferent-derisk-GO.md` (the de-risk)
        and `research/findings/2026-09-05-gateB-appraisal-interoceptive-production-flip-GO.md` (the flip
        verification) for the verdicts."""
        self.ensure_built()
        if appraisal_interoceptive_enabled():
            from research.runners._appraisal_interoceptive_ladder_derisk import get_ladder
            return get_ladder(self.seed).read_differential(
                appraisal, lesion=lesion, intero_lesion=appraisal_interoceptive_lesioned(),
                ramp_ms=ramp_ms, drive_off_ms=drive_off_ms, read_ms=read_ms)
        b, xp, idx, snap = self.bridge, self.xp, self.idx, self.snap
        lad = idx["ladder"]
        _restore_state(b, snap)
        b.cp_external_input_current[:] = 0.0
        SA._reset_modulators(b)
        b.set_transmission_gate("affect_out", 0.0 if lesion else 1.0)
        b.core_config.enable_ou_process = True
        nm = b.neuromodulator_manager
        m_abs = abs(float(appraisal))
        pos_sign = float(appraisal) >= 0.0

        def _set(m):
            nm.set_concentration("appraisal_lad_vplus", float(m) if pos_sign else 0.0)
            nm.set_concentration("appraisal_lad_vminus", 0.0 if pos_sign else float(m))
            nm.set_concentration("appraisal_lad_arousal", float(m))

        for _ in range(40):                                       # settle
            _set(0.0); b.cp_external_input_current[:] = 0.0; b._run_one_simulation_step()
        for s in range(int(ramp_ms)):                             # graded ramp 0 -> |appraisal|
            _set(m_abs * (s + 1) / ramp_ms); b.cp_external_input_current[:] = 0.0; b._run_one_simulation_step()
        for _ in range(int(drive_off_ms)):                        # DRIVE-OFF: persistence via the NMDA latches
            _set(0.0); b.cp_external_input_current[:] = 0.0; b._run_one_simulation_step()
        pos = neg = 0.0
        for _ in range(int(read_ms)):                             # read the held differential
            _set(0.0); b.cp_external_input_current[:] = 0.0; b._run_one_simulation_step()
            fs = to_host(b.cp_firing_states)
            pos += float(np.asarray(fs)[lad["pos_readout"]].sum())
            neg += float(np.asarray(fs)[lad["neg_readout"]].sum())
        b.core_config.enable_ou_process = False
        b.set_transmission_gate("affect_out", 1.0)
        _restore_state(b, snap)
        b.cp_external_input_current[:] = 0.0
        denom = float(SA.LAD_N_RO * max(1, read_ms))
        pr, nr = pos / denom, neg / denom
        return {"differential": float(pr - nr), "pos_rate": float(pr), "neg_rate": float(nr),
                "appraisal": float(appraisal), "lesioned": bool(lesion)}


_ORGAN: AffectProductionOrgan | None = None


def get_organ(seed: int = 42) -> AffectProductionOrgan:
    """The process-shared affect organ (built once on first use)."""
    global _ORGAN
    if _ORGAN is None:
        _ORGAN = AffectProductionOrgan(seed=seed)
    return _ORGAN


def affect_enabled() -> bool:
    """Default-ON. `BRAIN_AFFECT` in {0,false,no,off} -> the byte-identical oracle (affect fully disabled)."""
    v = os.environ.get("BRAIN_AFFECT")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def affect_lesioned() -> bool:
    """`BRAIN_AFFECT_LESION` in {1,true,yes,on} -> clamp affect_out=0 for the load-bearing lesion verify."""
    v = os.environ.get("BRAIN_AFFECT_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


def appraisal_interoceptive_enabled() -> bool:
    """PRODUCTION-DEFAULT-ON (2026-09-05 flip; was a default-OFF de-risk flag, scaffold-retirement backlog rank
    5). Unset -> ON: the Gate-B appraisal routes through the interoceptive-relay CURRENT afferent
    (`research.runners._appraisal_interoceptive_ladder_derisk.AppraisalInteroceptiveLadder`) instead of the
    direct host `nm.set_concentration(...)` write in `AffectProductionOrgan.read_differential` — matching the
    board #49/#81 pattern already running in production for the same substrate. `BRAIN_AFFECT_APPRAISAL_
    INTEROCEPTIVE` in {0,false,no,off} -> the ESCAPE HATCH: reverts to the ORIGINAL, byte-unchanged host-write
    path (the pre-flip production default), for instant rollback. See `research/findings/
    2026-09-05-gateB-appraisal-interoceptive-production-flip-GO.md` for the flip verification (6-seed
    mechanism-level + integrated handler-level, no-regression + load-bearing + anti-hollow)."""
    v = os.environ.get("BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


def appraisal_interoceptive_lesioned() -> bool:
    """`BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE_LESION` truthy -> (only meaningful when the flag above is on) cut the
    relay->ladder SYNAPSES (the `appraisal_intero_out` transmission gate = 0) so the appraisal can no longer reach
    the ladder even though the relay pools still fire -- the load-bearing dissociation proof for THIS adaptation,
    distinct from `lesion=True` (which cuts the ladder's OWN `affect_out` readout gate, identical semantics to the
    pre-existing host-write path)."""
    v = os.environ.get("BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE_LESION")
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The mood -> (content plan, prose manner) MAPS. tone_level is the Koulakov staircase LEVEL (-3..+3); NEUTRAL keeps
# the production fluent-multi-sentence default (no regression), POSITIVE expands + warms, NEGATIVE terse + curt.
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
NEUTRAL_SENTENCES = 4      # the current production default (RichAnswerComposer max_sentences) -- preserved at neutral
NEUTRAL_ELAB = 2           # the current production default (max_elaborations)


def tone_level(differential: float) -> int:
    """The graded valence LEVEL from the neural ladder differential (reuses the Stage-A Koulakov staircase)."""
    return int(SA._graded_tone_level(float(differential)))


def content_plan(level: int) -> dict:
    """FORTHCOMINGNESS (the genuine WHAT): how many gate-matched facts to volunteer + elaboration depth, from the
    mood level. Positive -> expansive; neutral -> the production default (no regression); negative -> terse. The
    facts are ALWAYS gate-sourced + moat-verified downstream, so this only changes HOW MUCH is volunteered, never
    WHICH fact is true and never whether an unmatched cue abstains."""
    if level >= 1:                                    # positive mood: forthcoming / expansive
        return {"max_sentences": 4, "max_elaborations": 3}
    if level <= -2:                                   # strongly negative: curt, direct fact only
        return {"max_sentences": 1, "max_elaborations": 0}
    if level == -1:                                   # mildly negative: brief
        return {"max_sentences": 2, "max_elaborations": 1}
    return {"max_sentences": NEUTRAL_SENTENCES, "max_elaborations": NEUTRAL_ELAB}   # neutral -> production default


# Manner clauses reinforce keeping the exact content words + the verb (so the VERIFY re-parse still recovers the
# gated SVO) while steering warmth. Capped at ±2 strength -- the reliably-verifying band (a stronger clause makes
# the mouth drift, which VERIFY safely DROPS -> a neutral render, never a moat breach). {a}/{v}/{p} are filled.
_MANNER_WARM = ("Keep the words {a}, {v}, {p} and the verb '{v}'. Write ONE short, warm, friendly sentence.")
_MANNER_CURT = ("Keep the words {a}, {v}, {p} and the verb '{v}'. Write ONE short, blunt, matter-of-fact sentence.")


def manner_template_for(level: int) -> str:
    """The prose-MANNER instruction TEMPLATE (with {a}/{v}/{p} placeholders) for THIS turn's mood level, filled
    per-fact by the MoodConditionedRenderer (each rich-path sentence is a different SVO). Positive -> warm/friendly
    phrasing; negative -> blunt/curt phrasing; neutral -> '' (byte-default render, no manner)."""
    if level >= 1:
        return _MANNER_WARM
    if level <= -1:
        return _MANNER_CURT
    return ""


def manner_for(level: int, a: str, v: str, p: str) -> str:
    """The prose-MANNER instruction filled for one specific SVO (for the single-fact path + debug display)."""
    t = manner_template_for(level)
    return t.format(a=a, v=v, p=p) if t else ""


def feel_readout(differential: float, valence: float, arousal: float) -> str:
    """Wire-2 -- the HONEST inner-state read-out for 'how do you feel'. A FUNCTIONAL read of the live valence
    differential (never a phenomenal claim). The differential is the neural ladder read; valence/arousal are the
    current appraised drive."""
    d = float(differential)
    if d > SA.LADDER_NEUTRAL_TOL:
        mood = "positive"
    elif d < -SA.LADDER_NEUTRAL_TOL:
        mood = "negative"
    else:
        mood = "neutral"
    aro = "high" if float(arousal) > 0.55 else ("low" if float(arousal) < 0.4 else "moderate")
    return (f"My affect monitor reads {mood} right now (valence differential {d:+.3f}, "
            f"{aro} arousal) -- that's a functional read of my mood state, not a felt experience.")


# match an explicit 'how do you feel' style inner-state query (kept narrow so it never hijacks a recall turn).
_FEEL_RE = re.compile(
    r"\b(how (are|do) you feel(ing)?|how('?s| is) your mood|what('?s| is) your mood|"
    r"are you (happy|sad|okay|ok|angry|upset)|how are you (doing|feeling))\b", re.IGNORECASE)


def is_feel_query(text: str) -> bool:
    return bool(_FEEL_RE.search(text or ""))


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The MANNER-CONDITIONED renderer wrapper. Delegates the actual generation to the base renderer's spiking faculty
# but injects the per-turn manner clause into the constrain prompt so the PROSE ITSELF is colored. The VERIFY
# re-parse (in ChatBrain / RichAnswerComposer) is unchanged -> a manner render that drifts is DROPPED (moat-safe).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class MoodConditionedRenderer:
    """Wraps a QwenRenderer (or StubRenderer). Per-turn `.manner` (an SVO-templated clause) is injected into the
    Qwen constrain prompt so the fluent mouth phrases the SAME fact warmer/curter. For a base without a spiking
    `_fac._generate` + CONSTRAIN_TEMPLATE (the GPU-free StubRenderer), the manner is a NO-OP (delegates unchanged)
    -- manner-coloring is genuine only on the fluent mouth; that is the declared boundary."""

    def __init__(self, base):
        self.base = base
        self.name = getattr(base, "name", "renderer")
        self.manner = ""     # set per-turn by the endpoint from manner_for(level, a, v, p)

    def _fac(self):
        fac = getattr(self.base, "_fac", None)
        if fac is not None and hasattr(fac, "_generate") and hasattr(fac, "CONSTRAIN_TEMPLATE"):
            return fac
        return None

    def render_svo(self, a, v, p):
        fac = self._fac()
        if self.manner and fac is not None:
            manner = self.manner.format(a=a, v=v, p=p) if "{" in self.manner else self.manner
            prompt = fac.CONSTRAIN_TEMPLATE.format(a=a, v=v, p=p) + " " + manner
            surface, _full, _s = fac._generate(prompt)
            return surface, None
        return self.base.render_svo(a, v, p)

    def render_svo_regen(self, a, v, p):
        # the tighter regenerate (after a VERIFY reject) forces the exact words -> NO manner (recovery, not color).
        if hasattr(self.base, "render_svo_regen"):
            return self.base.render_svo_regen(a, v, p)
        return self.base.render_svo(a, v, p)
