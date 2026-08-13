"""PRODUCTION VERIFY for the SURPRISE-organ per-block HOMEOSTATIC PREDICTION-GAIN equalizer wired into the production
`SurpriseProductionOrgan.ensure_built` (Gate-B/D2 precision companion, 2026-08-13).

The de-risk (`_surprise_organ_homeostat_derisk`, GO 6/6, `2026-08-13-surprise-organ-homeostat-GO.md`) closed the
surprise organ's single-read confirm-precision residual (`het_vote_rate` 0.9375 -> 1.0) with a per-block homeostatic
prediction-gain equalizer. This wiring adds that equalizer to the SHIPPED organ's build (default-ON, escape
`BRAIN_SURPRISE_HOMEOSTAT=0`). This verify asserts the closure on the PRODUCTION organ + no regression through the
REAL brain_chat D2 handler:

  (A) ORGAN-LEVEL PRECISION CLOSURE (the deliverable), 6 seeds 42/43/44/100/101/102, numpy-CPU. With the homeostat ON
      (default): every FAMILIAR (confirm) edge reads BELOW threshold -> `vote_rate` (familiar recognized) == 1.0 (8/8)
      on EVERY seed, AND surprise SPECIFICITY holds (novel + contradict still register surprise, rate >= threshold, on
      every edge). With the homeostat OFF: the residual is REAL (>=1 seed drops below 1.0) -> the equalizer is
      load-bearing, not cosmetic. `calib.homeostat` is True iff the equalizer ran.
  (B) REAL-HANDLER NO-REGRESSION (Gate-B/D2 turn). Teach 'wolf hunt deer'; a CONFIRM re-statement ('wolf hunt deer')
      is NOT surprised (no notice); a CONTRADICT ('wolf hunt rock') IS surprised (notice prepended); the D2 organ the
      handler uses carries the equalizer by default (`surprise.calib.homeostat` True). LESION (`BRAIN_SURPRISE_LESION=1`)
      makes the SAME confirm FIRE (surprised) -> the prediction inhibition is load-bearing. The `BRAIN_SURPRISE_HOMEOSTAT=0`
      escape gives the SAME confirm/contradict verdicts on this (non-marginal) fact -> additive, no behavior regression.

Run (numpy-CPU, fast rf recall path):
  SIM_BACKEND=numpy BRAIN_COMPOSER_KIND=rf python -u -m research.runners._surprise_organ_homeostat_production_verify
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_COMPOSER_KIND", "rf")

import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

import numpy as np  # noqa: E402

from research.runners.surprise_production_organ import SurpriseProductionOrgan  # noqa: E402
from research.runners._spiking_expectation_rpe_derisk import measure_conditions  # noqa: E402

SEEDS = [42, 43, 44, 100, 101, 102]


def _organ_metrics(seed, homeo):
    """Build the PRODUCTION organ at `seed` with the homeostat ON/OFF and read its per-block confirm/contradict/novel
    rates. `vote_rate` = fraction of stored (familiar) blocks whose confirm reads BELOW threshold (recognized, not
    falsely surprised); specificity = fraction of contradict/novel edges that still register surprise (>= threshold)."""
    os.environ["BRAIN_SURPRISE_HOMEOSTAT"] = "1" if homeo else "0"
    try:
        o = SurpriseProductionOrgan(seed=seed)
        o.ensure_built()
        res = measure_conditions(o.bridge, o.cfg, o.idx_map, o.meta, o.xp)
    finally:
        os.environ.pop("BRAIN_SURPRISE_HOMEOSTAT", None)
    thr = float(o.threshold)
    conf = np.asarray(res["confirm_per"]); contra = np.asarray(res["contradict_per"]); nov = np.asarray(res["novel_per"])
    return {
        "seed": seed, "homeostat_flag": bool(homeo), "calib_homeostat": bool(o.calib.get("homeostat", False)),
        "threshold": round(thr, 3), "confirm_max": round(float(conf.max()), 3),
        "vote_rate": round(float(np.mean(conf < thr)), 4),
        "novel_registers": round(float(np.mean(nov >= thr)), 4),
        "contradict_registers": round(float(np.mean(contra >= thr)), 4),
        "pred_gain": (None if o.pred_gains is None else [round(float(o.pred_gains.min()), 2),
                                                         round(float(o.pred_gains.max()), 2)]),
    }


def _setup_session(session):
    import webapp.server as S
    chat, source = S._build_chat_brain("tiny-demo", "stub")
    cache_key = (session, "tiny-demo", "stub")
    chat._brain_chat_source = source
    S._BRAIN_CHATS[cache_key] = chat
    return cache_key


def _turn(session, message):
    from webapp.server import brain_chat, BrainChatRequest as Req
    r = brain_chat(Req(session=session, message=message, brain="tiny-demo", renderer="stub", rich=False))
    return json.loads(r.body.decode("utf-8"))


def main():
    rows = {}

    # ── (A) ORGAN-LEVEL PRECISION CLOSURE — 6 seeds, homeostat ON vs OFF ─────────────────────────────────────────
    on = [_organ_metrics(s, True) for s in SEEDS]
    off = [_organ_metrics(s, False) for s in SEEDS]
    on_vote_all = all(abs(r["vote_rate"] - 1.0) < 1e-9 for r in on)
    on_spec_all = all(r["novel_registers"] >= 0.999 and r["contradict_registers"] >= 0.999 for r in on)
    on_calib_all = all(r["calib_homeostat"] for r in on)
    off_calib_none = all((not r["calib_homeostat"]) for r in off)
    # the residual is REAL: with the homeostat OFF at least one seed fails to recognize a familiar edge (vote<1.0).
    off_residual_real = any(r["vote_rate"] < 1.0 - 1e-9 for r in off)
    a_ok = bool(on_vote_all and on_spec_all and on_calib_all and off_calib_none and off_residual_real)
    rows["A_organ_precision"] = {"on": on, "off": off, "on_vote_rate_all_1": on_vote_all,
                                 "on_specificity_all": on_spec_all, "on_calib_homeostat_all": on_calib_all,
                                 "off_calib_homeostat_none": off_calib_none, "off_residual_real": off_residual_real,
                                 "ok": a_ok}

    # ── (B) REAL-HANDLER NO-REGRESSION — Gate-B/D2 turn, homeostat default-ON ────────────────────────────────────
    # Isolate the surprise organ: keep the belief-rewriting organs off so the taught fact is stable across the reads.
    for k in ("BRAIN_RECONSOLIDATION", "BRAIN_NONCONTRADICTION_GATE"):
        os.environ[k] = "0"
    import research.runners.surprise_production_organ as _SO  # the process-shared organ singleton (reset to rebuild)

    def _run_handler_panel(tag):
        _setup_session(tag)
        _turn(tag, "wolf hunt deer")                        # TEACH -> the brain now holds (wolf,hunt)->deer
        t_conf = _turn(tag, "wolf hunt deer")               # CONFIRM (stored==asserted) -> not surprised
        t_contra = _turn(tag, "wolf hunt rock")             # CONTRADICT (rock != deer) -> surprised
        return t_conf, t_contra

    _SO._ORGAN = None                                       # clean process singleton -> first D2 turn builds default-on
    t_conf, t_contra = _run_handler_panel("sh")
    sc = t_conf.get("surprise") or {}
    sx = t_contra.get("surprise") or {}
    confirm_not_surprised = bool(sc.get("surprised") is False)
    contradict_surprised = bool(sx.get("surprised") is True and t_contra["answer"].startswith("That surprises me"))
    handler_homeostat_on = bool((sc.get("calib") or {}).get("homeostat") is True)
    b_core_ok = bool(confirm_not_surprised and contradict_surprised and handler_homeostat_on)

    # LESION: the SAME confirm fires (the prediction inhibition is load-bearing). Reuses the homeostatted singleton +
    # its on-demand lesioned twin (edges zeroed) — the deployed default organ under lesion.
    os.environ["BRAIN_SURPRISE_LESION"] = "1"
    try:
        _setup_session("sh_les")
        _turn("sh_les", "wolf hunt deer")
        t_conf_les = _turn("sh_les", "wolf hunt deer")
    finally:
        os.environ.pop("BRAIN_SURPRISE_LESION", None)
    scl = t_conf_les.get("surprise") or {}
    lesion_confirm_fires = bool(scl.get("surprised") is True)

    # ESCAPE (homeostat OFF) end-to-end: RESET the process singleton so it rebuilds honoring the flag, then the same
    # confirm/contradict verdicts hold on this (non-marginal) fact -> additive, no behavior regression.
    _SO._ORGAN = None
    os.environ["BRAIN_SURPRISE_HOMEOSTAT"] = "0"
    try:
        t_conf_off, t_contra_off = _run_handler_panel("sh_off")
    finally:
        os.environ.pop("BRAIN_SURPRISE_HOMEOSTAT", None)
        _SO._ORGAN = None                                   # restore: next use rebuilds default-on
    off_conf_ns = bool((t_conf_off.get("surprise") or {}).get("surprised") is False)
    off_contra_s = bool((t_contra_off.get("surprise") or {}).get("surprised") is True)
    off_calib_off = bool(((t_conf_off.get("surprise") or {}).get("calib") or {}).get("homeostat") is None)
    escape_ok = bool(off_conf_ns and off_contra_s and off_calib_off)

    for k in ("BRAIN_RECONSOLIDATION", "BRAIN_NONCONTRADICTION_GATE"):
        os.environ.pop(k, None)

    b_ok = bool(b_core_ok and lesion_confirm_fires and escape_ok)
    rows["B_real_handler"] = {
        "confirm_not_surprised": confirm_not_surprised, "confirm_hz": sc.get("surprise_hz"),
        "contradict_surprised": contradict_surprised, "contradict_hz": sx.get("surprise_hz"),
        "handler_homeostat_on": handler_homeostat_on, "lesion_confirm_fires": lesion_confirm_fires,
        "lesion_confirm_hz": scl.get("surprise_hz"),
        "escape_off_confirm_not_surprised": off_conf_ns, "escape_off_contradict_surprised": off_contra_s,
        "escape_off_calib_homeostat_absent": off_calib_off, "ok": b_ok}

    go = bool(a_ok and b_ok)

    from tools.verdict import Verdict
    v = Verdict("surprise organ homeostat production wiring")
    v.require("ORGAN PRECISION: homeostat-ON vote_rate==1.0 (8/8) on all 6 seeds", on_vote_all, expect=True)
    v.require("SURPRISE SPECIFICITY: novel + contradict still register on all 6 seeds (ON)", on_spec_all, expect=True)
    v.require("the equalizer is LOAD-BEARING: >=1 seed drops below 1.0 with the homeostat OFF", off_residual_real, expect=True)
    v.require("REAL HANDLER: CONFIRM not-surprised + CONTRADICT surprised + D2 organ homeostat-on by default", b_core_ok, expect=True)
    v.require("LESION-LOAD-BEARING: the SAME confirm FIRES under BRAIN_SURPRISE_LESION=1", lesion_confirm_fires, expect=True)
    v.require("ESCAPE BRAIN_SURPRISE_HOMEOSTAT=0: same confirm/contradict verdicts (no regression)", escape_ok, expect=True)
    v.disabled("online spiking homeostatic-plasticity rule",
               why="the equalizer is a BUILD-TIME host-orchestrated calibration loop (like the organ's existing "
               "threshold + Hebbian train_expectation loops); an ONLINE spiking inhibitory/homeostatic-plasticity "
               "rule (Vogels 2011) is the further step. The which-patient MAPPING is still a topographic prior "
               "(a fully-learned all-to-all CA3 recall is the unchanged separate rung).")
    decided = v.decide(go=go, verbose=False)
    go = bool(decided["go"])

    print("\n" + "=" * 104, flush=True)
    print("  SURPRISE-ORGAN HOMEOSTAT — PRODUCTION VERIFY (production organ + real brain_chat D2 handler, numpy-CPU)", flush=True)
    print("=" * 104, flush=True)
    print(f"  (A) ORGAN PRECISION CLOSURE ok={a_ok}", flush=True)
    for r_on, r_off in zip(on, off):
        print(f"        seed {r_on['seed']:>3}  ON vote={r_on['vote_rate']:.3f} confirm_max={r_on['confirm_max']:.2f} "
              f"thr={r_on['threshold']:.2f} spec(nov/con)={r_on['novel_registers']:.2f}/{r_on['contradict_registers']:.2f} "
              f"gain={r_on['pred_gain']} | OFF vote={r_off['vote_rate']:.3f} confirm_max={r_off['confirm_max']:.2f}", flush=True)
    print(f"        ON vote_all=1.0 {on_vote_all}; ON spec_all {on_spec_all}; OFF residual real {off_residual_real}", flush=True)
    print(f"  (B) REAL HANDLER ok={b_ok}", flush=True)
    print(f"        CONFIRM not-surprised={confirm_not_surprised} (hz={sc.get('surprise_hz')}); "
          f"CONTRADICT surprised={contradict_surprised} (hz={sx.get('surprise_hz')}); homeostat-on={handler_homeostat_on}", flush=True)
    print(f"        LESION confirm fires={lesion_confirm_fires} (hz={scl.get('surprise_hz')}); escape-off ok={escape_ok}", flush=True)
    verdict = "GO" if go else "NO-GO"
    print(f"\n  VERDICT: {verdict}\n" + "=" * 104, flush=True)

    out = {"runner": "_surprise_organ_homeostat_production_verify", "go": go, "status": decided["status"],
           "a_organ_precision_ok": a_ok, "b_real_handler_ok": b_ok, "rows": rows,
           "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
           "undefined_reasons": decided["undefined_reasons"]}
    op = "research/findings/raw/_surprise_organ_homeostat/production_verify.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
