"""D5 AFFECT-CONDITIONED GENERATION on the Qwen articulation mouth — the literature-backed method after two
DECODE-POINT NO-GOs, measured with the SAME 6-seed directional independent-lexicon open-output gate.

CONTEXT. research/findings/2026-09-22-affect-tone-open-output-directional-6seed-positive-asymmetric-NOGO.md
(additive top-margin decode bias) and research/findings/2026-09-23-affect-tone-neural-coupling-6seed-subdelta-
NOGO.md (neuromodulator nudge at the spiking decode race) both intervene at the DECODE. Their external-research
section names the proven class: condition the GENERATION REPRESENTATION on a continuous valence/arousal signal
(Affect-LM, Ghosh et al. ACL 2017; VAD-conditioning, Guo/Xu/Chua arXiv:2111.04730). The Qwen mouth is the
owner-ratified permanent conditioned-articulation scaffold (2026-09-19), already conditioned by the brain through
`build_prompt` (KNOWLEDGE + MOOD lines). This runner measures `webapp/affect_conditioned_mouth.py`
(`BRAIN_OPEN_ENDED_AFFECT_CONDITIONED=prompt|resid`, default-OFF) with the base runner's gate, unchanged.

WHAT IS REUSED BY IMPORT (not copied): prompts, primings, the independent lexicon + VADER-style scorer, the
attribution-control injection magnitude/RNG seed, and `score_and_gate` (called with mouth="qwen", which swaps
only the two WKV-specific preconditions for "every tone reply was written by generator==qwen").

EXTRA INSTRUMENT PRECONDITIONS this runner adds (each unmet -> UNDEFINED, never a negative):
  (L) conditioning LEVER moved: every pos/neg/ctrl tone row carries c != 0 and every lesion row c == 0; in
      resid mode the hook was registered and called on every conditioned row.
  (O) the ORGAN read is the NO-GO runs' read: each arm's priming differential equals the committed
      research/findings/raw/_affect_tone_open_output/arm_s<seed>_<arm>.json value (same brain state as the two
      NO-GOs -> the three verdicts are comparable; only the mouth + its conditioning differ).

MODES:
  --magnitude-audit   pure, no brain: quantify the live valence magnitude problem from the committed NO-GO arms
                      (held differential, valence, organ tone_level, what `_mood_phrase` rendered). Writes
                      research/findings/raw/_affect_conditioned_mouth/magnitude_audit.json.
  --calibrate-organ   builds the affect organ (full numpy brain, memcap 8): the held differential over an appraisal
                      sweep -> the organ's FULL-SCALE valence VALENCE_FS (the normalization of c). Per seed.
  --worker / --controller / --score-only   the 36-arm run (6 seeds x pos/neg/lesion/lesion_rep/ctrl_pos/ctrl_neg).
  --smoke             one arm, two prompts, CPU Qwen: footprint + wiring check.
  --selftest          pure logic (no brain, no model).

GO-GATE (PREREGISTERED 2026-09-23, before any conditioned-mouth tone result existed): the base gate verbatim —
(1) DIRECTIONAL pos_gap > +delta AND neg_gap < -delta per seed, 6/6 or 5/6 with the 6th null, delta = the pooled
std of the lesion arm's per-prompt tone; (2) ATTRIBUTION |ctrl_pos - ctrl_neg| < delta every seed; (3) content-
identity + moat; (4) fluency salad <= 0.16; (5) lesion == lesion_rep byte-identical — plus (L) and (O) above.
Operating point fixed in webapp/affect_conditioned_mouth.py (RESID_LAYER=12, RESID_K=4.0, c = valence/VALENCE_FS
with VALENCE_FS measured by --calibrate-organ). Not tuned on tone. The literal scoring command:
  .venv/bin/python -m research.runners._lbf_affect_conditioned_mouth_derisk --score-only --mode <prompt|resid>

COMPUTE: the brain runs numpy (SIM_BACKEND=numpy — the same organ numerics as the NO-GOs); Qwen runs on CUDA when
visible (gpu_queue) or float32 CPU. brain_chat + Qwen ~ memcap 16 per arm -> run under tools/gpu_queue.sh, one
seed per queue line (arms sequential inside, so one brain at a time).
"""
import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

import research.runners._lbf_affect_tone_open_output_derisk as _BASE  # noqa: E402

SEEDS = list(_BASE.SEEDS)
ARMS = list(_BASE.ARMS)
MODES = ("prompt", "resid")
OUT_ROOT = "research/findings/raw/_affect_conditioned_mouth"
NOGO_ARMS_DIR = "research/findings/raw/_affect_tone_open_output"
CALIB_APPRAISALS = [-1.0, -0.75, -0.475, -0.25, 0.0, 0.25, 0.475, 0.75, 1.0]

# the Qwen-mouth harness env: identical to the NO-GO harness EXCEPT the mouth is forced to Qwen (WKV mouth off)
# and every alternative generator that could pre-empt the Qwen one-shot is off, so the tone prompts reach Qwen.
_ENV_QWEN = {
    "SIM_BACKEND": "numpy",
    "BRAIN_OPEN_ENDED": "1",
    "BRAIN_OPEN_ENDED_WKV_MOUTH": "0",
    "BRAIN_OPEN_ENDED_NP_ENTAILMENT": "0",
    "BRAIN_OPEN_ENDED_GEN_TIME_HONESTY": "0",
    "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK": "0",
    "BRAIN_OPEN_ENDED_NO_QWEN_FALLBACK": "0",
    "BRAIN_LTM_SHIP_DEFAULT": "1",
}


def out_dir_for(mode):
    return os.path.join(OUT_ROOT, mode)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  MAGNITUDE AUDIT (pure; reads committed NO-GO arm data)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def magnitude_audit(write=True):
    from research.runners._open_ended_state_driven_generation_derisk import _mood_phrase, _valence_from_differential
    from research.runners._stageA_full_integration_derisk import LADDER_NEUTRAL_TOL
    rows = []
    for s in SEEDS:
        for a in ("pos", "neg"):
            p = os.path.join(_REPO, NOGO_ARMS_DIR, "arm_s%d_%s.json" % (s, a))
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            pa = d.get("priming_affect") or {}
            diff = float(pa.get("differential", 0.0))
            val = _valence_from_differential(diff)
            phrase = _mood_phrase(val, float(pa.get("appraisal_arousal", 0.3)))
            rows.append({"seed": s, "arm": a, "appraisal_valence": pa.get("appraisal_valence"),
                         "differential": diff, "valence": val, "organ_tone_level": pa.get("tone_level"),
                         "above_organ_neutral_tol": abs(diff) > LADDER_NEUTRAL_TOL,
                         "mood_phrase_rendered": phrase,
                         "mood_phrase_dead_zoned": phrase.startswith("even and steady")})
    out = {
        "what": "live valence magnitude problem, quantified from the committed NO-GO arm data (no re-run)",
        "source_dir": NOGO_ARMS_DIR, "n_rows": len(rows), "rows": rows,
        "mood_phrase_threshold": 0.25, "valence_map": "clip(4 * differential, -1, 1)",
        "organ_neutral_tol": LADDER_NEUTRAL_TOL,
        "n_dead_zoned": sum(r["mood_phrase_dead_zoned"] for r in rows),
        "pos_valence_range": [min((r["valence"] for r in rows if r["arm"] == "pos"), default=None),
                              max((r["valence"] for r in rows if r["arm"] == "pos"), default=None)],
        "neg_valence_range": [min((r["valence"] for r in rows if r["arm"] == "neg"), default=None),
                              max((r["valence"] for r in rows if r["arm"] == "neg"), default=None)],
        "n_neg_below_organ_tol": sum(1 for r in rows if r["arm"] == "neg" and not r["above_organ_neutral_tol"]),
        "n_pos_below_organ_tol": sum(1 for r in rows if r["arm"] == "pos" and not r["above_organ_neutral_tol"]),
    }
    if write:
        os.makedirs(os.path.join(_REPO, OUT_ROOT), exist_ok=True)
        ap = os.path.join(_REPO, OUT_ROOT, "magnitude_audit.json")
        json.dump(out, open(ap, "w"), indent=2)
        print("[audit] wrote %s" % ap)
    print("[audit] dead-zoned MOOD lines: %d/%d  pos valence %s  neg valence %s  neg below organ tol: %d/6"
          % (out["n_dead_zoned"], out["n_rows"], out["pos_valence_range"], out["neg_valence_range"],
             out["n_neg_below_organ_tol"]))
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  ORGAN CALIBRATION (builds the affect organ; memcap 8)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def calib_path(seed):
    return os.path.join(_REPO, OUT_ROOT, "calibration", "organ_calib_s%d.json" % seed)


def calibrate_organ(seed):
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from research.runners import affect_production_organ as AO
    t0 = time.time()
    organ = AO.get_organ(seed=seed)
    sweep = []
    for a in CALIB_APPRAISALS:
        r = organ.read_differential(float(a))
        sweep.append({"appraisal": a, "differential": float(r["differential"]),
                      "pos_rate": float(r.get("pos_rate", 0.0)), "neg_rate": float(r.get("neg_rate", 0.0)),
                      "tone_level": int(AO.tone_level(float(r["differential"])))})
    full_pos = next(x["differential"] for x in sweep if x["appraisal"] == 1.0)
    full_neg = next(x["differential"] for x in sweep if x["appraisal"] == -1.0)
    # ONE full-scale for both signs (the mean full-scale magnitude) -> the organ's sign asymmetry is preserved.
    fs_diff = (abs(full_pos) + abs(full_neg)) / 2.0
    out = {"seed": seed, "sweep": sweep, "full_scale_pos_diff": full_pos, "full_scale_neg_diff": full_neg,
           "full_scale_diff_mean": fs_diff, "valence_fs": 4.0 * fs_diff,
           "appraisal_interoceptive": AO.appraisal_interoceptive_enabled(),
           "backend": os.environ.get("SIM_BACKEND"), "wall_seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(calib_path(seed)), exist_ok=True)
    json.dump(out, open(calib_path(seed), "w"), indent=2)
    print("[calib] seed=%d valence_fs=%.4f (full-scale diff +%.4f / %.4f) -> %s"
          % (seed, out["valence_fs"], full_pos, full_neg, calib_path(seed)), flush=True)
    return out


def valence_fs_for(seed):
    p = calib_path(seed)
    if os.path.exists(p):
        return float(json.load(open(p))["valence_fs"])
    return None


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  WORKER (one arm, one fresh process)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def run_worker(seed, arm, mode, out_path, prompts=None, known_prompts=None):
    import random
    assert arm in ARMS and mode in MODES
    lesion = "1" if arm in ("lesion", "lesion_rep") else "0"
    priming = _BASE.PRIMING_NEG if arm in ("neg", "ctrl_neg") else _BASE.PRIMING_POS
    control = arm in ("ctrl_pos", "ctrl_neg")
    for k, v in _ENV_QWEN.items():
        os.environ[k] = v
    os.environ["BRAIN_CHAT_SEED"] = str(seed)
    os.environ["BRAIN_AFFECT_LESION"] = lesion
    os.environ["BRAIN_OPEN_ENDED_AFFECT_CONDITIONED"] = mode
    fs = valence_fs_for(seed)
    if fs is None:
        raise SystemExit("no organ calibration for seed %d (%s) -- run --calibrate-organ first; the "
                         "normalization must be MEASURED, not defaulted" % (seed, calib_path(seed)))
    os.environ["BRAIN_AFFECT_COND_VALENCE_FS"] = repr(fs)

    sys.path.insert(0, _REPO)
    t0 = time.time()
    import webapp.server as S
    from webapp import open_ended_chat as _OE
    from webapp import affect_conditioned_mouth as _ACM

    ctrl_state = {"active": False, "idx": 0, "vals": []}
    if control:
        rng = random.Random(_BASE._CTRL_RNG_SEED)
        n = len(_BASE.TONE_PROMPTS) + len(_BASE.KNOWN_PROMPTS) + 4
        ctrl_state["vals"] = [rng.choice([_BASE._CTRL_MAG, -_BASE._CTRL_MAG]) for _ in range(n)]
        _orig_vfa = _OE.valence_from_affect

        def _patched_valence(differential):
            if ctrl_state["active"]:
                v = ctrl_state["vals"][ctrl_state["idx"] % len(ctrl_state["vals"])]
                ctrl_state["idx"] += 1
                return float(v)
            return _orig_vfa(differential)
        _OE.valence_from_affect = _patched_valence

    def chat(msg, reset):
        _ACM.LAST_TRACE.clear()
        resp = S.brain_chat(S.BrainChatRequest(session="lbf_affect_cond_%d_%s_%s" % (seed, arm, mode),
                                               message=msg, brain="tiny-demo", reset=reset,
                                               rich=True, renderer="stub"))
        return json.loads(bytes(resp.body)), dict(_ACM.LAST_TRACE)

    d0, _ = chat(priming, True)
    affect0 = (d0.get("affect") or {})
    ctrl_state["active"] = True
    ctrl_state["idx"] = 0
    lex, _ = _BASE.load_indep_lexicon()

    def measure(ps):
        rows = []
        for p in ps:
            d, tr = chat(p, False)
            oe = d.get("open_ended") or {}
            raw = oe.get("raw") or ""
            comp, hits = _BASE.tone_compound(raw, lex)
            rows.append({"prompt": p, "raw": raw, "known": bool(oe.get("known")), "facts": oe.get("facts") or [],
                         "generator": oe.get("generator"), "wkv_mouth_used": bool(oe.get("wkv_mouth_used")),
                         "tone_compound": comp, "tone_hits": hits, "salad_frac": _BASE.salad_frac(raw),
                         "affect_differential": (d.get("affect") or {}).get("differential"),
                         "state_valence": (oe.get("state") or {}).get("valence"),
                         "gen_seconds": oe.get("gen_seconds"), "conditioning": tr})
        return rows

    tone_rows = measure(prompts or _BASE.TONE_PROMPTS)
    known_rows = measure(known_prompts if known_prompts is not None else _BASE.KNOWN_PROMPTS)
    try:
        import resource
        maxrss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        maxrss_gb = None
    out = {"runner": "_lbf_affect_conditioned_mouth_derisk (worker)", "seed": seed, "arm": arm, "mode": mode,
           "lesion": lesion, "control": control, "priming": priming, "valence_fs": fs,
           "backend": os.environ.get("SIM_BACKEND"),
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "resolved_ckpt": None, "ckpt_is_per_seed": True,   # Qwen: no per-seed ckpt (precondition swapped)
           "priming_affect": affect0, "tone_rows": tone_rows, "known_rows": known_rows,
           "maxrss_gb": maxrss_gb, "wall_seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print("[worker] seed=%d arm=%s mode=%s wrote %s (%.1fs, maxrss %.1f GB)"
          % (seed, arm, mode, out_path, out["wall_seconds"], maxrss_gb or -1), flush=True)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  CONTROLLER (sequential arms: one brain at a time; one queue line per seed)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def run_controller(mode, seeds, memcap_gb=16):
    od = os.path.join(_REPO, out_dir_for(mode))
    os.makedirs(od, exist_ok=True)
    memcap = os.path.join(_REPO, "tools", "memcap.sh")
    use_memcap = memcap_gb and os.path.exists(memcap) and _BASE._memcap_ok()
    for s in seeds:
        if valence_fs_for(s) is None:
            cmd = [sys.executable, "-u", "-m", "research.runners._lbf_affect_conditioned_mouth_derisk",
                   "--calibrate-organ", "--seed", str(s)]
            if use_memcap:
                cmd = ["bash", memcap, "8", "--"] + cmd
            env = dict(os.environ, SIM_BACKEND="numpy", CUDA_VISIBLE_DEVICES="")
            print("[controller] calibrating organ seed=%d" % s, flush=True)
            subprocess.run(cmd, cwd=_REPO, env=env, check=False)
        for a in ARMS:
            op = _BASE._worker_out(od, s, a)
            if os.path.exists(op):
                try:
                    json.load(open(op))
                    continue
                except Exception:
                    pass
            cmd = [sys.executable, "-u", "-m", "research.runners._lbf_affect_conditioned_mouth_derisk",
                   "--worker", "--seed", str(s), "--arm", a, "--mode", mode, "--out", op]
            if use_memcap:
                cmd = ["bash", memcap, str(memcap_gb), "--"] + cmd
            env = dict(os.environ)
            env.update(_ENV_QWEN)
            t0 = time.time()
            with open(os.path.join(od, "arm_s%d_%s.log" % (s, a)), "w") as log:
                rc = subprocess.run(cmd, cwd=_REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            print("[controller] seed=%d arm=%s rc=%d wrote=%s (%.0fs)"
                  % (s, a, rc, os.path.exists(op), time.time() - t0), flush=True)


def _lever_and_organ_checks(od, mode):
    """(L) conditioning lever moved / held at zero where it must; (O) organ read == the NO-GO runs' read."""
    lever_ok, organ_ok = True, True
    lever_bad, organ_bad = [], []
    for s in SEEDS:
        for a in ARMS:
            p = _BASE._worker_out(od, s, a)
            if not os.path.exists(p):
                lever_ok = organ_ok = False
                continue
            d = json.load(open(p))
            for r in d["tone_rows"]:
                tr = r.get("conditioning") or {}
                c = tr.get("c")
                if a in ("lesion", "lesion_rep"):
                    good = (c == 0.0)
                else:
                    good = (c is not None and c != 0.0)
                    if good and mode == "resid":
                        good = bool(tr.get("hook_registered")) and int(tr.get("hook_calls", 0)) > 0
                    if good and mode == "prompt":
                        good = "even and steady" not in (tr.get("mood_line") or "even and steady")
                if not good:
                    lever_ok = False
                    lever_bad.append((s, a, r["prompt"], c))
            ref = os.path.join(_REPO, NOGO_ARMS_DIR, "arm_s%d_%s.json" % (s, a))
            if os.path.exists(ref):
                want = (json.load(open(ref)).get("priming_affect") or {}).get("differential")
                got = (d.get("priming_affect") or {}).get("differential")
                if want is None or got is None or abs(float(want) - float(got)) > 1e-12:
                    organ_ok = False
                    organ_bad.append((s, a, want, got))
            else:
                organ_ok = False
                organ_bad.append((s, a, "no-ref", None))
    # tools.lab.lever: the manipulation must MOVE the conditioning scalar between the lesion and the affect arms
    # (a lever that never moved would make the whole A/B void). Non-raising here: the precondition above decides.
    from tools.lab import lever
    c_by_arm = {}
    for s in SEEDS:
        for a in ("lesion", "pos", "neg"):
            p = _BASE._worker_out(od, s, a)
            if os.path.exists(p):
                c_by_arm.setdefault(a, []).extend(
                    (r.get("conditioning") or {}).get("c") for r in json.load(open(p))["tone_rows"])
    if c_by_arm.get("lesion") and c_by_arm.get("pos"):
        lever("affect-conditioning scalar c: lesion -> pos", sorted(set(map(str, c_by_arm["lesion"]))),
              sorted(set(map(str, c_by_arm["pos"]))), required=False)
    return [("(L) conditioning lever: c!=0 on pos/neg/ctrl rows (hook fired / graded line), c==0 on lesion",
             lever_ok, "bad=%s" % (lever_bad[:6] or "none")),
            ("(O) organ read == the NO-GO runs' priming differential (comparable brain state)",
             organ_ok, "bad=%s" % (organ_bad[:6] or "none"))]


def score(mode):
    od = os.path.join(_REPO, out_dir_for(mode))
    extra = _lever_and_organ_checks(od, mode)
    art = _BASE.score_and_gate(od, write_artifact=False, mouth="qwen", extra_require=extra)
    art["runner"] = "_lbf_affect_conditioned_mouth_derisk (reuses _lbf_affect_tone_open_output_derisk's scorer/gate)"
    art["what"] = ("affect->tone over the OPEN Qwen-mouth reply with the generation CONDITIONED on the spiking "
                   "affect organ's valence (mode=%s); 6-seed directional independent-lexicon lesion probe" % mode)
    art["coupling_mechanism"] = {"prompt": "graded dead-zone-free MOOD-line conditioning (existing channel)",
                                 "resid": "residual-stream conditioning c*K*u at layer 12 (CAA-style)"}[mode]
    # tools.lab.attributable_to: WHOSE is the directional tone gap -- the real spiking mood, or "any active
    # conditioning"? (the base gate computes it too; re-asserted here so this runner's own verdict carries it.)
    from tools.lab import attributable_to
    art["attribution_fraction_recheck"] = attributable_to(
        "affect-conditioned mouth (%s): real mood vs mood-decoupled control" % mode,
        art.get("attribution_treatment_mean", 0.0), art.get("attribution_control_mean", 0.0))
    art["valence_fs_per_seed"] = {s: valence_fs_for(s) for s in SEEDS}
    art["shortcut_status"] = ("c = valence / VALENCE_FS is a host readout-gain normalization (named shortcut on the "
                              "scaffold boundary); the conditioning SIGNAL is the spiking organ's held differential.")
    ap = os.path.join(od, "affect_conditioned_mouth_%s_verdict.json" % mode)
    json.dump(art, open(ap, "w"), indent=2, default=str)
    print("[score] wrote %s  GO=%s" % (ap, art.get("GO")), flush=True)
    return art


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  SELFTEST (pure; no brain, no model)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def selftest():
    import re
    from webapp import affect_conditioned_mouth as ACM
    from research.runners._affect_distributional_tag_derisk import WARRINER, STOP
    ok = True

    def check(name, cond):
        nonlocal ok
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))
        ok = ok and bool(cond)

    lex, _ = _BASE.load_indep_lexicon()
    toks = set(re.findall(r"[a-z']+", " ".join(ACM.CONTRAST_POS + ACM.CONTRAST_NEG).lower()))
    check("steering contrast set shares NO word with the independent scoring lexicon", not (toks & set(lex)))
    desc = set(re.findall(r"[a-z']+", " ".join(ACM.graded_mood_phrase(c, 0.3) for c in
                                                (-1.0, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 0.7, 1.0)).lower()))
    check("every prompt-mode MOOD descriptor word is OUTSIDE the scoring lexicon (no echo-scoring)",
          not (desc & set(lex)))
    content = {t for t in toks if t not in STOP and len(t) > 2}
    affect_words = content & set(WARRINER)
    check("steering contrast set carries WARRINER affect words (>=12)", len(affect_words) >= 12)
    # c normalization
    os.environ["BRAIN_AFFECT_COND_VALENCE_FS"] = "0.4"
    check("c(0) == 0 exactly (lesion is an exact no-op)", ACM.conditioning_scalar(0.0) == 0.0)
    check("c monotone + clipped", ACM.conditioning_scalar(0.1) < ACM.conditioning_scalar(0.2)
          and ACM.conditioning_scalar(5.0) == 1.0 and ACM.conditioning_scalar(-5.0) == -1.0)
    # graded phrase: NO dead zone at the live magnitude, sign-correct, neutral only at 0
    from research.runners._open_ended_state_driven_generation_derisk import _mood_phrase
    check("production _mood_phrase IS dead-zoned at live valence +0.16 (the defect)",
          _mood_phrase(0.16, 0.4).startswith("even and steady"))
    c = ACM.conditioning_scalar(0.16)
    check("graded phrase at live +0.16 is NOT neutral and is positive",
          "warm" in ACM.graded_mood_phrase(c, 0.4))
    check("graded phrase at live -0.08 is NOT neutral and is negative",
          "subdued" in ACM.graded_mood_phrase(ACM.conditioning_scalar(-0.08), 0.3))
    check("graded phrase at c=0 is the production neutral wording",
          ACM.graded_mood_phrase(0.0, 0.3).startswith("even and steady"))
    # condition_prompt touches ONLY the MOOD line
    sysp = "A\nSELF: x.\nKNOWLEDGE: y\nMOOD: you feel even and steady (valence +0.16, arousal 0.40).\nFAMILIARITY: z"
    new = ACM.condition_prompt(sysp, 0.16, 0.4)
    diff_lines = [(a, b) for a, b in zip(sysp.split("\n"), new.split("\n")) if a != b]
    check("condition_prompt rewrites exactly one line, the MOOD line",
          len(diff_lines) == 1 and diff_lines[0][0].startswith("MOOD: "))
    # default-off: answer_turn only imports the module behind the env read
    src = open(os.path.join(_REPO, "webapp", "open_ended_chat.py")).read()
    i_imp = src.find("from webapp import affect_conditioned_mouth")
    i_gate = src.find('os.environ.get("BRAIN_OPEN_ENDED_AFFECT_CONDITIONED"')
    check("open_ended_chat imports the module ONLY inside the env-gated branch", 0 < i_gate < i_imp)
    os.environ.pop("BRAIN_OPEN_ENDED_AFFECT_CONDITIONED", None)
    check("mode unset -> off", ACM.affect_conditioned_mode() == "")
    os.environ["BRAIN_OPEN_ENDED_AFFECT_CONDITIONED"] = "bogus"
    check("unknown mode -> off", ACM.affect_conditioned_mode() == "")
    os.environ.pop("BRAIN_OPEN_ENDED_AFFECT_CONDITIONED", None)
    os.environ.pop("BRAIN_AFFECT_COND_VALENCE_FS", None)
    # base gate signature default preserved
    import inspect
    sig = inspect.signature(_BASE.score_and_gate)
    check("base score_and_gate default mouth is 'wkv' (NO-GO runs re-score identically)",
          sig.parameters["mouth"].default == "wkv" and tuple(sig.parameters["extra_require"].default) == ())
    check("shared base selftest still passes", _BASE.selftest())
    # lever check must FAIL on a fabricated arm set where the lesion carries a nonzero c
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for s in SEEDS:
            for a in ARMS:
                cval = 0.0 if a in ("lesion", "lesion_rep") else 0.3
                if s == 42 and a == "lesion":
                    cval = 0.3          # the planted failure
                row = {"prompt": "p", "conditioning": {"c": cval, "mood_line": "MOOD: slightly warm"}}
                json.dump({"tone_rows": [row], "priming_affect": {"differential": 0.0}},
                          open(_BASE._worker_out(td, s, a), "w"))
        checks = _lever_and_organ_checks(td, "prompt")
        check("lever precondition FAILS when a lesion row carries c!=0 (can fail in its failing direction)",
              checks[0][1] is False)
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--magnitude-audit", action="store_true")
    ap.add_argument("--calibrate-organ", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--controller", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one arm (pos), 2 tone prompts, no known prompt")
    ap.add_argument("--mode", choices=MODES, default="resid")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--arm", type=str, default="pos")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--memcap-gb", type=int, default=16)
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    elif args.magnitude_audit:
        magnitude_audit()
    elif args.calibrate_organ:
        calibrate_organ(args.seed)
    elif args.smoke:
        op = args.out or os.path.join(_REPO, OUT_ROOT, "smoke", "smoke_s%d_%s_%s.json" % (args.seed, args.arm, args.mode))
        run_worker(args.seed, args.arm, args.mode, op, prompts=_BASE.TONE_PROMPTS[:2], known_prompts=[])
    elif args.worker:
        run_worker(args.seed, args.arm, args.mode,
                   args.out or _BASE._worker_out(os.path.join(_REPO, out_dir_for(args.mode)), args.seed, args.arm))
    elif args.score_only:
        score(args.mode)
    else:
        run_controller(args.mode, args.seeds, memcap_gb=args.memcap_gb)
