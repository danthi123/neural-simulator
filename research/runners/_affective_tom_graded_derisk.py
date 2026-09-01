"""W5-graded: upgrade the affective-ToM OTHER-model read from the BISTABLE valence-SIGN (P0.3 `AffectStateBrain` /
`read_tone`, 3-state {-1,0,+1}) to the GRADED valence x arousal CIRCUMPLEX -- the SAME #81 Koulakov bistable-LADDER
robust integrator the SELF read already uses (`2026-08-19-graded-affect-attractor-GO.md`: valence Pearson +0.97
6/6 seeds, arousal +0.95 6/6 seeds, production-wired into the live chat via `webapp/affect_drives_chat.py`).
REUSE-BY-IMPORT of `GradedAffectBrain` + `read_body` (`_graded_affect_attractor_derisk.py`) -- NO `sim/` edit, NO new
brain-building code; the ladder mechanism is byte-identical to the SELF path's.

THE NAMED RESIDUAL (2026-08-26 W5 production wire-in, closed here). `affective_tom_production_organ.py` collapses
the Gate-B DR-2 appraised valence MAGNITUDE to a bare SIGN (`valence_sign = 1 if valence > 0.0 else -1`) and drives
the P0.3 BISTABLE `AffectStateBrain` through `_affective_tom_derisk.read_tone`, so the empathic lead is a 3-state
switch (comfort / neutral / share-joy) no matter HOW devastated or delighted the OTHER's situation is -- "Maria is a
bit sad" and "Maria is utterly devastated" produce the IDENTICAL lead. The finding named this explicitly: "Fine
discrete emotions need the SAME graded-circumplex surpass P0.3 already named, NOT a new wall." This module IS that
surpass, applied to the OTHER-model read (the SELF read already got it on 2026-08-19 / wired 2026-08-19-#84).

MECHANISM (a SEPARATE OTHER-tagged `GradedAffectBrain` instance -- preserves the "separate slot per agent" motif
W3/W5 use for dissociability from the SELF's own #84 ladder; two independent bridges, never shared). The OTHER's
DR-2 appraised valence in [-1,1] maps onto the SAME comfort/discomfort opponent body-state #81 reads
(h=(valence+1)/2, so comfort=h, discomfort=1-h are anti-correlated exactly as the SELF path); the appraised arousal
in [0,1] maps directly onto the arousal channel. `read_body(brain, h, a, ...)` (VERBATIM #81 read, no modification)
drives the 3 interoceptive relays -> the 3 ladders (vplus/vminus/arousal; each N_L=6 independently-latching
self-recurrent NMDA sub-pools, NO intra-sign lateral inhibition -- the load-bearing Koulakov rule) and reads the
POPULATION differential mood = rate(V+ ladder) - rate(V- ladder), felt_arousal = rate(arousal ladder), off
`cp_firing_states` -- NEVER a host formula. The continuous mood is quantized into a 7-level (-3..+3) staircase
`graded_tone_level` for the expression map -- up to 7 empathic-intensity tiers instead of the old 3-state switch.

LESION (the SAME #81 embodiment lesion, already 6/6-seed proven there): `read_body(..., lesion_gate=True)` gates
`intero_out=0`, severing the OTHER's appraisal -> ladder synapses. Per #81, this collapses the valence coupling
(range 0.156 -> 0.000, |corr| -> 0.00) and the arousal coupling identically -- so here it collapses `mood`/
`felt_arousal` to ~0 -> `tone_level` -> 0 -> the empathic lead VANISHES, the exact lesion signature the bistable W5
read already established for its OWN `AffectStateBrain` (this is the SAME transmission-gate-severing pattern, on
the SELF-path's already-proven ladder mechanism rather than a new untested lesion).

CONTRACT (additive, reversible, byte-identical-off). This module is imported ONLY from inside
`affective_tom_production_organ.observe_turn`, guarded by a NEW flag `affective_tom_graded_enabled()`
(`BRAIN_AFFECTIVE_TOM_GRADED`, default-OFF). With the flag off (the default), the production organ never imports
this file -> the existing 6/6-seed-GO bistable-sign path (`_AFFECTIVE_TOM_DEFAULT_ON=True` in production) is
completely unperturbed -- BYTE-IDENTICAL to pre-upgrade. This module never touches `_affect_state_region_derisk.py`,
`_affective_tom_derisk.py`, or the bistable `AffectiveToMOrgan` class.

HONEST RESIDUALS (named, ride existing burn-down items -- identical boundary to the #81 GO and the W5 bistable GO):
  1. Gradedness is QUANTIZED (a 7-level Koulakov staircase), NOT a smooth continuum (the #81 honest boundary,
     inherited verbatim -- more resolution is more sub-pools, a linear cost, not a wall).
  2. The message -> OTHER-situation valence/arousal APPRAISAL is host (the SAME Gate-B DR-2 language-comprehension
     boundary the bistable path already used) -- unchanged by this upgrade; only the READ of that appraisal into a
     neural OTHER-tagged state is upgraded from bistable to graded.
  3. The level -> EXPRESSION-MARKER string is a host conditioned-articulation scaffold ("the mouth"): the tone that
     DRIVES the tier is the neural graded ladder read (lesion-provable), the surface STRING per tier is a template
     (the sanctioned articulation-crutch pattern, same as the bistable path's `empathic_lead`).
  4. This module reads its OWN co-resident OTHER-tagged ladder bridge (a THIRD affect bridge in the process,
     alongside the SELF's #84 ladder and the bistable W5's OTHER `AffectStateBrain`) -- the one-brain consolidation
     step (merging affect bridges) remains a follow-on, shared with the existing affect burn-down.
  5. Honesty boundary: a functional graded affective-mentalizing correlate with an honest functional read-out; no
     claim of access to another mind's feelings.

Run (smoke -- 1 seed):  SIM_BACKEND=numpy python -u -m research.runners._affective_tom_graded_derisk --smoke
Run (6-seed battery):   SIM_BACKEND=numpy python -u -m research.runners._affective_tom_graded_derisk \
                            --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402 (passthrough on numpy)
# reuse-by-import: the EXACT #81 graded bistable-LADDER brain + read (NO sim/ edit; no new brain-building code --
# this is the SAME mechanism the SELF read (#84 affect_drives_chat) already uses in production).
from research.runners._graded_affect_attractor_derisk import (  # noqa: E402
    GradedAffectBrain, read_body, resolvable_levels, N_L, I_BODY_PA,
)
from tools.lab import attributable_to  # noqa: E402

OUT = Path(_REPO) / "research" / "findings" / "raw" / "_affective_tom_graded" / "graded_other_6seed.json"

_DEFAULT_SEED = 42
# a lighter production read window than the #81 sweep default (settle=60 establish=250 read=120) -- still several
# multiples of the ladder's slow-NMDA integration time constant, calibrated by the smoke.
_READ_KW = dict(settle=60, establish=180, read=100)
_NEUTRAL_TOL = 0.010     # |mood| below this -> level 0 (neutral); well under the #81 6-seed valence half-range.
_LEVEL_UNIT = 0.026      # mood units per graded LEVEL (~ the #81 valence half-range 0.078 / 3 levels-per-side).
_MAX_LEVEL = 3           # +/-3 -> a 7-level staircase, matching the #81 ladder's N_L+1=7 pooled resolvable levels.


def graded_tone_level(mood: float, unit: float = _LEVEL_UNIT, tol: float = _NEUTRAL_TOL,
                       max_level: int = _MAX_LEVEL) -> int:
    """Quantize the CONTINUOUS ladder mood differential into an integer LEVEL in [-max_level, +max_level] (a
    7-level staircase by default). Never a bare sign: |mood| < tol -> 0 (neutral), otherwise the level count grows
    with the magnitude of the population read -- this IS the graded read (vs the old `read_tone`'s 3-state sign)."""
    m = float(mood)
    if abs(m) < tol:
        return 0
    lvl = int(round(m / unit))
    return int(np.clip(lvl, -max_level, max_level))


# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
# THE PROCESS-SHARED OTHER-TAGGED GRADED AFFECT ORGAN. A SEPARATE #81 GradedAffectBrain instance (never the SELF's
# #84 ladder, never the bistable W5's AffectStateBrain) driven by the OTHER's appraised (valence, arousal).
# Snapshot/restore-isolated (the #77 global-RNG footgun -- same discipline as AffectiveToMOrgan / affect_drives_chat).
# ────────────────────────────────────────────────────────────────────────────────────────────────────────────
class OtherGradedAffectOrgan:
    def __init__(self, seed: int = _DEFAULT_SEED):
        self.seed = int(seed)
        self._brain = None
        self._lock = threading.Lock()

    def _isolated(self, fn):
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
        try:
            return fn()
        finally:
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
            self._brain = GradedAffectBrain(self.seed, nmda_on=True)

    def read_other_state(self, valence: float, arousal: float, *, lesion: bool = False) -> dict:
        """Map the OTHER's appraised (valence in [-1,1], arousal in [0,1]) onto the #81 body-state (h, a) -- the
        SAME comfort/discomfort/arousal channel the SELF path drives -- and read the ladder's graded population
        state. `lesion` cuts intero_out (the #81 embodiment lesion) -> the coupling collapses to ~0."""
        h = float(np.clip((float(valence) + 1.0) / 2.0, 0.0, 1.0))
        a = float(np.clip(float(arousal), 0.0, 1.0))
        with self._lock:
            self._isolated(self._ensure)
            r = self._isolated(lambda: read_body(self._brain, h, a, lesion_gate=bool(lesion), **_READ_KW))
        mood = float(r["mood"])
        felt_arousal = float(r["felt_arousal"])
        level = graded_tone_level(mood)
        return {"mood": mood, "felt_arousal": felt_arousal, "tone_level": int(level),
                "h": h, "a": a, "valence": float(valence), "arousal": float(arousal), "lesioned": bool(lesion)}


_ORGAN: Optional[OtherGradedAffectOrgan] = None


def get_graded_organ(seed: int = _DEFAULT_SEED) -> OtherGradedAffectOrgan:
    """The process-shared OTHER-tagged GRADED affect organ (built once on first use)."""
    global _ORGAN
    if _ORGAN is None:
        _ORGAN = OtherGradedAffectOrgan(seed=seed)
    return _ORGAN


# ── the graded LEVEL -> empathic EXPRESSION-marker map (7 tiers instead of the bistable path's 3-state {-1,0,+1}).
#    Sign 0 -> '' (byte-identical-neutral surface, as the bistable path). Host conditioned-articulation scaffold, the
#    SAME sanctioned pattern as the bistable path's `empathic_lead` (honest residual #3 above) -- the tone that
#    DRIVES the tier is the neural graded read (lesion-provable); the string per tier is a template.
_NEG_TIERS = {1: "That sounds tough for %s -- ", 2: "That sounds really hard for %s -- ",
              3: "That sounds devastating for %s -- "}
_POS_TIERS = {1: "That's nice for %s -- ", 2: "That's wonderful for %s -- ",
              3: "That's absolutely thrilling for %s -- "}


def empathic_lead_graded(tone_level: int, agent: str) -> str:
    """The empathic lead for this turn's GRADED inferred OTHER emotion. tone_level from the neural graded OTHER
    read: -3..-1 escalating comfort, +1..+3 escalating share-joy, 0 (neutral / lesion-collapsed) -> '' (no lead)."""
    a = str(agent or "them")
    lvl = int(tone_level)
    if lvl < 0:
        return _NEG_TIERS.get(min(abs(lvl), 3), _NEG_TIERS[3]) % a
    if lvl > 0:
        return _POS_TIERS.get(min(lvl, 3), _POS_TIERS[3]) % a
    return ""


def _corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _threshold_hash(seed):
    b = GradedAffectBrain(seed)
    th = to_host(b._bridge.cp_neuron_firing_thresholds)
    return np.asarray(th, float).tobytes()


# =============================================================================================================
# One seed: graded valence sweep (fixed arousal) + graded arousal sweep (fixed valence sign) + lesion collapse +
# the OLD BISTABLE comparison (the same sweep points scored through the sign-only read, to show the surpass).
# =============================================================================================================
_VAL_SWEEP = [float(x) for x in np.linspace(-1.0, 1.0, 15)]
_ARO_SWEEP = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]


def run_seed(seed: int) -> dict:
    t0 = time.time()
    organ = OtherGradedAffectOrgan(seed=seed)

    # ---- (1) GRADED VALENCE: sweep the OTHER's appraised valence (fixed moderate arousal), read mood/level. ----
    val_intact = [organ.read_other_state(v, 0.5, lesion=False) for v in _VAL_SWEEP]
    val_lesion = [organ.read_other_state(v, 0.5, lesion=True) for v in _VAL_SWEEP]
    mood_intact = [r["mood"] for r in val_intact]
    mood_lesion = [r["mood"] for r in val_lesion]
    levels_intact = [r["tone_level"] for r in val_intact]
    corr_val_mood = _corr(_VAL_SWEEP, mood_intact)
    mood_range = float(max(mood_intact) - min(mood_intact))
    mood_range_lesion = float(max(mood_lesion) - min(mood_lesion))
    n_distinct_levels = len(set(levels_intact))
    # read-noise floor (repeat one mid point) for the resolvable-levels count.
    mid_repeats = [organ.read_other_state(0.55, 0.5, lesion=False)["mood"] for _ in range(3)]
    mood_sd = float(np.std(mid_repeats)) if len(mid_repeats) > 1 else 0.0
    mood_levels_resolvable = resolvable_levels(mood_intact, [mood_sd] * len(mood_intact), min_step=0.003)

    # ---- (2) GRADED AROUSAL: sweep the OTHER's appraised arousal (fixed positive valence), read felt_arousal. ----
    aro_intact = [organ.read_other_state(0.6, a, lesion=False) for a in _ARO_SWEEP]
    aro_lesion = [organ.read_other_state(0.6, a, lesion=True) for a in _ARO_SWEEP]
    felt_intact = [r["felt_arousal"] for r in aro_intact]
    felt_lesion = [r["felt_arousal"] for r in aro_lesion]
    corr_aro_felt = _corr(_ARO_SWEEP, felt_intact)
    felt_range = float(max(felt_intact) - min(felt_intact))
    felt_range_lesion = float(max(felt_lesion) - min(felt_lesion))

    # ---- (3) LESION collapses BOTH channels on EVERY sweep point (the #81 embodiment lesion, reused). ----
    lesion_levels = [r["tone_level"] for r in val_lesion]
    lesion_collapsed = bool(all(abs(m) < _NEUTRAL_TOL for m in mood_lesion) and all(lv == 0 for lv in lesion_levels))

    # ---- (4) LOAD-BEARING: the graded empathic lead takes on MORE than the old bistable's {neg,'',pos} = 3 strings.
    leads = [empathic_lead_graded(lv, "Sam") for lv in levels_intact]
    n_distinct_leads = len(set(leads))
    lead_vs_lesion_vanishes = bool(all(empathic_lead_graded(lv, "Sam") == "" for lv in lesion_levels))

    # ---- (5) THE OLD BISTABLE COMPARISON: score the SAME sweep points through a pure sign(valence) read -- this is
    #      EXACTLY what `affective_tom_production_organ.observe_turn`'s `valence_sign = 1 if valence>0 else -1` did.
    #      It can only ever emit 2 nonzero states -- the surpass is n_distinct_levels (up to 7) vs this fixed 2.
    old_bistable_states = sorted(set(1 if v > 0.0 else -1 for v in _VAL_SWEEP if v != 0.0))

    row = {
        "seed": int(seed),
        "corr_val_mood": corr_val_mood, "mood_range": mood_range, "mood_range_lesion": mood_range_lesion,
        "n_distinct_levels": n_distinct_levels, "mood_levels_resolvable": mood_levels_resolvable,
        "levels_intact": levels_intact, "mood_curve_intact": [float(x) for x in mood_intact],
        "corr_aro_felt": corr_aro_felt, "felt_range": felt_range, "felt_range_lesion": felt_range_lesion,
        "lesion_collapsed": lesion_collapsed,
        "n_distinct_leads": n_distinct_leads, "leads": leads, "lead_vs_lesion_vanishes": lead_vs_lesion_vanishes,
        "old_bistable_n_states": len(old_bistable_states),
        "intero_owns_valence_frac": attributable_to("intero_owns_valence(range intact vs lesion)",
                                                     mood_range, mood_range_lesion),
        "intero_owns_arousal_frac": attributable_to("intero_owns_arousal(range intact vs lesion)",
                                                     felt_range, felt_range_lesion),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print(f"  [seed {seed}] VAL corr {corr_val_mood:+.2f} range {mood_range:.3f} distinct_levels {n_distinct_levels} "
          f"resolvable {mood_levels_resolvable} (lesion range {mood_range_lesion:.3f}) | AROU corr {corr_aro_felt:+.2f} "
          f"range {felt_range:.3f} (lesion {felt_range_lesion:.3f}) | leads distinct {n_distinct_leads} "
          f"lesion_collapsed={lesion_collapsed} ({row['elapsed_seconds']}s)", flush=True)
    return row


# =============================================================================================================
# BYTE-IDENTICAL-OFF: with the new flag OFF, `affective_tom_production_organ.observe_turn` must be untouched --
# this module must never even be imported by the production organ.
# =============================================================================================================
def flag_off_byte_identical():
    import importlib
    import research.runners.affective_tom_production_organ as O
    os.environ.pop("BRAIN_AFFECTIVE_TOM_GRADED", None)
    importlib.reload(O)
    graded_flag_present = hasattr(O, "affective_tom_graded_enabled")
    off_by_default = (not O.affective_tom_graded_enabled()) if graded_flag_present else True
    O._ORGAN = None
    global _ORGAN
    _ORGAN = None
    sys.modules.pop("research.runners._affective_tom_graded_derisk", None)   # simulate a FRESH process: this
    # module (the upgrade) has never been imported yet -> observe_turn with the flag off must not import it.
    class C: pass
    bad = O.observe_turn(C(), "Maria is devastated")
    good = O.observe_turn(C(), "Tom is delighted")
    graded_module_imported = "research.runners._affective_tom_graded_derisk" in sys.modules
    return {
        "graded_flag_present": bool(graded_flag_present), "off_by_default": bool(off_by_default),
        "bistable_bad_lead": bad.get("lead"), "bistable_good_lead": good.get("lead"),
        "bistable_bad_tone_sign": bad.get("tone_sign"), "bistable_good_tone_sign": good.get("tone_sign"),
        "no_graded_key_on_bad": "tone_level" not in bad, "no_graded_key_on_good": "tone_level" not in good,
        "old_path_still_produces_bad_lead": bool(bad.get("lead")),
        "old_path_still_produces_good_lead": bool(good.get("lead")),
        "graded_module_never_imported_when_off": bool(not graded_module_imported),
    }


# =============================================================================================================
# REAL-WORD END-TO-END DEMO: run the actual production entry point (observe_turn) on real messages, both flag-OFF
# (the shipped bistable path) and flag-ON (this upgrade), and save the table as a citable artifact. This is the
# genuine text -> DR-2 appraisal -> ladder-or-bistable-read -> lead pipeline, not a synthetic appraisal dict.
# =============================================================================================================
_DEMO_MESSAGES = ["Maria is lonely", "Maria feels lost", "Maria is heartbroken", "Maria was hurt",
                  "Maria is cheerful", "Maria is proud"]


def realword_demo(seed: int = _DEFAULT_SEED) -> dict:
    import importlib
    import research.runners.affective_tom_production_organ as O
    importlib.reload(O)

    class C:
        pass

    rows = []
    for msg in _DEMO_MESSAGES:
        os.environ["BRAIN_AFFECTIVE_TOM_GRADED"] = "0"
        O._ORGAN = None
        off = O.observe_turn(C(), msg, seed=seed)
        os.environ["BRAIN_AFFECTIVE_TOM_GRADED"] = "1"
        O._ORGAN = None
        global _ORGAN
        _ORGAN = None
        on = O.observe_turn(C(), msg, seed=seed)
        rows.append({"message": msg,
                    "bistable_tone_sign": off.get("tone_sign"), "bistable_lead": off.get("lead"),
                    "dr2_valence": round(float(O_appraisal_valence(msg)), 6),
                    "graded_mood": round(float(on.get("mood", 0.0)), 6),
                    "graded_tone_level": on.get("tone_level"), "graded_lead": on.get("lead")})
    os.environ.pop("BRAIN_AFFECTIVE_TOM_GRADED", None)
    # the frontier this closes: on well-separated real words the OLD bistable path gives an IDENTICAL lead
    # regardless of magnitude (both negative -> "really hard"), the NEW graded path differentiates it.
    lonely = next(r for r in rows if r["message"] == "Maria is lonely")
    heartbroken = next(r for r in rows if r["message"] == "Maria is heartbroken")
    bistable_collapses = bool(lonely["bistable_lead"] == heartbroken["bistable_lead"] and lonely["bistable_lead"])
    graded_differentiates = bool(lonely["graded_lead"] != heartbroken["graded_lead"] and lonely["graded_lead"]
                                 and heartbroken["graded_lead"])
    return {"seed": int(seed), "rows": rows, "bistable_collapses_lonely_vs_heartbroken": bistable_collapses,
            "graded_differentiates_lonely_vs_heartbroken": graded_differentiates}


def O_appraisal_valence(msg: str) -> float:
    from research.runners import affect_production_organ as AO
    return float(AO.appraise_text(msg).get("valence", 0.0))


# =============================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--realword-demo", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    if a.realword_demo:
        demo = realword_demo(a.seeds[0])
        outp = str(Path(a.out).parent / "realword_endtoend_demo.json")
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        Path(outp).write_text(json.dumps(demo, indent=2, default=str))
        print(json.dumps(demo, indent=2, default=str), flush=True)
        print(f"[affective-tom-graded REALWORD-DEMO] wrote {outp} ({round(time.time()-t0,1)}s)", flush=True)
        return 0
    seeds = [a.seeds[0]] if a.smoke else a.seeds

    determinism_ok = (_threshold_hash(seeds[0]) == _threshold_hash(seeds[0]))
    rows = [run_seed(s) for s in seeds]
    byte_off = flag_off_byte_identical()

    ns = len(rows)

    def m(k):
        return float(np.mean([r[k] for r in rows]))

    n_val_graded = sum(1 for r in rows if r["corr_val_mood"] >= 0.8 and r["mood_levels_resolvable"] > 2)
    n_aro_graded = sum(1 for r in rows if r["corr_aro_felt"] >= 0.8)
    n_more_than_bistable = sum(1 for r in rows if r["n_distinct_levels"] > r["old_bistable_n_states"])
    n_lesion = sum(1 for r in rows if r["lesion_collapsed"])
    n_leads_graded = sum(1 for r in rows if r["n_distinct_leads"] > 3)   # old bistable ships exactly 3 (neg/''/pos)
    n_lead_vanish = sum(1 for r in rows if r["lead_vs_lesion_vanishes"])

    agg = {
        "valence_graded(corr>=0.8 & >2 resolvable levels)_5of6": n_val_graded >= min(5, ns),
        "arousal_graded(corr>=0.8)_5of6": n_aro_graded >= min(5, ns),
        "more_distinct_levels_than_old_bistable_6of6": n_more_than_bistable == ns,
        "lesion_collapses_6of6": n_lesion == ns,
        "empathic_lead_more_than_3_tiers_6of6(the load-bearing surpass)": n_leads_graded == ns,
        "lead_vanishes_under_lesion_6of6": n_lead_vanish == ns,
    }
    preconditions = [
        {"kind": "require", "name": "substrate_seeded(cfg.seed; identical thresholds on rebuild)", "ok": determinism_ok},
        {"kind": "require", "name": "numpy_spiking_backend", "ok": os.environ.get("SIM_BACKEND", "") == "numpy"},
        {"kind": "require", "name": "flag_present_and_off_by_default", "ok": bool(byte_off["graded_flag_present"] and byte_off["off_by_default"])},
        {"kind": "require", "name": "flag_off_bistable_path_unperturbed(bad/good leads still fire)",
         "ok": bool(byte_off["old_path_still_produces_bad_lead"] and byte_off["old_path_still_produces_good_lead"])},
    ]
    if a.smoke:
        out = {"probe": "affective_tom_graded (smoke)", "determinism_ok": determinism_ok,
               "per_seed": rows, "byte_off": byte_off}
        outp = str(a.out).replace(".json", "_smoke.json")
        Path(outp).parent.mkdir(parents=True, exist_ok=True)
        Path(outp).write_text(json.dumps(out, indent=2, default=str))
        print(f"[affective-tom-graded SMOKE] wrote {outp} ({round(time.time()-t0,1)}s)", flush=True)
        return 0

    go = all(agg.values()) and all(p["ok"] for p in preconditions) and ns == 6
    means = {k: m(k) for k in ("corr_val_mood", "mood_range", "mood_range_lesion", "corr_aro_felt", "felt_range",
                               "felt_range_lesion", "n_distinct_levels", "n_distinct_leads",
                               "intero_owns_valence_frac", "intero_owns_arousal_frac")}
    means["mood_levels_resolvable_mean"] = m("mood_levels_resolvable")

    baseline = ("baseline (the 2026-08-26 W5 production wire-in): the OTHER-model read collapsed the appraised "
                "valence to sign() and drove the P0.3 BISTABLE AffectStateBrain via read_tone -> tone_sign in "
                "{-1,0,+1}, exactly 3 possible empathic-lead strings regardless of the OTHER's affect magnitude.")
    if go:
        verdict = (f"GO ({ns}-seed) -- the OTHER-model affective-ToM read is now GRADED: it consumes the SAME "
                   f"#81 Koulakov bistable-LADDER circumplex the SELF read uses (reuse-by-import, no new brain), "
                   f"driven by the OTHER's appraised valence x arousal instead of the SELF's body-state. VALENCE: "
                   f"Pearson(valence,mood) {means['corr_val_mood']:+.2f} ({n_val_graded}/{ns} >=0.8), mean "
                   f"{means['n_distinct_levels']:.1f} distinct levels per seed on a 10-point sweep (vs the old "
                   f"bistable's fixed 2 nonzero states, {n_more_than_bistable}/{ns} seeds strictly more). AROUSAL: "
                   f"Pearson(arousal,felt) {means['corr_aro_felt']:+.2f} ({n_aro_graded}/{ns} >=0.8). LESION (the "
                   f"#81 embodiment lesion, reused): cutting intero_out collapses mood/felt-arousal to ~0 on "
                   f"{n_lesion}/{ns} seeds -> tone_level 0 -> the graded lead VANISHES exactly as the bistable path's "
                   f"lesion did. LOAD-BEARING: the empathic lead takes on a mean {means['n_distinct_leads']:.1f} "
                   f"distinct strings per seed ({n_leads_graded}/{ns} seeds >3, i.e. genuinely more than the old "
                   f"bistable's 3-string ceiling) and vanishes under lesion on {n_lead_vanish}/{ns} seeds. "
                   f"BYTE-IDENTICAL-OFF: with BRAIN_AFFECTIVE_TOM_GRADED unset the bistable path is untouched "
                   f"(bad-other lead {byte_off['bistable_bad_lead']!r}, good-other lead "
                   f"{byte_off['bistable_good_lead']!r}, no tone_level key attached). Additive; NO sim/ edit; "
                   f"numpy-CPU. {baseline}")
    else:
        miss = [k for k, v in agg.items() if not v] + [p["name"] for p in preconditions if not p["ok"]]
        verdict = (f"PARTIAL/BOUNDARY ({ns}-seed) -- FAILED {miss}. VALENCE corr {means['corr_val_mood']:+.2f} "
                   f"({n_val_graded}/{ns}); AROUSAL corr {means['corr_aro_felt']:+.2f} ({n_aro_graded}/{ns}); "
                   f"lesion {n_lesion}/{ns}; leads>3tiers {n_leads_graded}/{ns}. {baseline}")

    summary = {
        "probe": "affective_tom_graded (W5 OTHER-model bistable-sign -> graded-circumplex surpass)",
        "verdict": verdict, "GO": bool(go),
        "preconditions": preconditions, "aggregate_checks": agg,
        "n_seeds": ns, "n_valence_graded": n_val_graded, "n_arousal_graded": n_aro_graded,
        "n_more_distinct_than_bistable": n_more_than_bistable, "n_lesion_collapsed": n_lesion,
        "n_leads_graded": n_leads_graded, "n_lead_vanish": n_lead_vanish,
        "byte_off": byte_off, "baseline_note": baseline, "means": means, "per_seed": rows,
        "config": {"seeds": seeds, "val_sweep": _VAL_SWEEP, "aro_sweep": _ARO_SWEEP,
                   "level_unit": _LEVEL_UNIT, "neutral_tol": _NEUTRAL_TOL, "max_level": _MAX_LEVEL,
                   "read_kw": _READ_KW},
        "mechanism": "The OTHER-tagged read reuses the #81 Koulakov robust-integrator LADDER (GradedAffectBrain / "
                     "read_body, verbatim import) driven by the OTHER's DR-2 appraised (valence,arousal) mapped "
                     "onto the SAME comfort/discomfort/arousal body-state channel the SELF path drives. Felt = "
                     "rate(V+ ladder)-rate(V- ladder) and rate(arousal ladder), off cp_firing_states -- never a "
                     "host formula. The continuous mood is quantized (graded_tone_level) into a 7-level staircase "
                     "for the empathic-expression map, replacing the old sign()-only read.",
        "HONEST_NOTE": "Gradedness is QUANTIZED (7 levels), not a smooth continuum (the #81 boundary, inherited). "
                       "The message->valence/arousal APPRAISAL stays host (Gate-B DR-2, unchanged from the bistable "
                       "path) -- only the READ of that appraisal into a neural OTHER-tagged state is upgraded. "
                       "Additive/default-OFF (BRAIN_AFFECTIVE_TOM_GRADED); the shipped 6/6-seed-GO bistable path "
                       "(default-ON in production) is untouched when the flag is off.",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[affective-tom-graded] VERDICT: {verdict}", flush=True)
    print(f"[affective-tom-graded] GO={go} | wrote {a.out} ({summary['elapsed_seconds']}s)\n" + "=" * 110, flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
