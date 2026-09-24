"""D5 CONTENT-PRESERVING TONE: the brain's affect SELECTS among content-locked restyles of ONE draft — 6-seed
directional independent-lexicon lesion probe, with the content held fixed AND measured fixed.

CONTEXT. research/findings/2026-09-23-affect-conditioned-mouth-live-valence-diagnosis-and-staged-derisk.md (D5,
amend1): conditioning the Qwen mouth's generation (prompt MOOD line / residual steering) scored UNDEFINED in both
modes because the reply's CONTENT moved with the conditioning — (3a-reply) fact-word recall vs the lesion reply was 0
on most seeds, in the treatment AND the shuffled-valence control ('Frank Lincoln Wright' -> Franklin D. Roosevelt);
resid mode also broke fluency. Mechanism: `webapp/affect_tone_selection.py` (`BRAIN_OPEN_ENDED_AFFECT_TONE_SELECT=1`,
default-OFF): draft once affect-free, restyle that one draft in 4 fixed tones, content-lock, let the spiking affect
organ evaluate each admissible candidate, release the one closest to the organ's held valence (HOST argmin —
declared shortcut; not credited as spiking selection).

ARMS (7 per seed, one fresh process each, sequential): pos / neg (primed, organ active), lesion / lesion_rep
(BRAIN_AFFECT_LESION=1: held valence 0 -> the most-neutral-reading candidate), ctrl_pos / ctrl_neg (the held valence
fed to answer_turn is replaced by the base runner's mood-decoupled random-sign ±0.16 sequence, identical in both ctrl
arms), pos_rep (determinism of the conditioned path).

PREREGISTERED GATE (see the finding's gate block; written before any run of this runner, including the smoke):
 base gate verbatim via `_lbf_affect_tone_open_output_derisk.score_and_gate(mouth="qwen")`:
  (1) DIRECTIONAL pos_gap > +delta AND neg_gap < -delta, 6/6 or 5/6 with the 6th null, delta = pooled std of
      the lesion arm's per-prompt tone on the independent 356-word lexicon (disjoint from WARRINER);
  (2) ATTRIBUTION |ctrl_pos - ctrl_neg| < delta every seed; (3a) base retrieval identity (INTEGRITY SMOKE here);
  (3b) moat; (4) fluency salad <= 0.16; (5) lesion == lesion_rep byte-identical.
 plus, this method:
  (P)  seed-level exact sign-flip permutation null over the 6 per-seed directional gaps D_s = tone_pos - tone_neg:
       GO needs mean(D) > 0 and p <= 0.05; the control's D_ctrl must NOT reach it (require).
  (3a-reply) D5's content identity over the GENERATED known replies (fact-word recall vs lesion >= 0.75, every
       conditioned arm, every seed); measurability (>= 2 lesion fact words per seed) is a require.
  (C2) general content-word recall vs the lesion reply (content words = len>=4, not stop, not WARRINER, not the
       tone lexicon; pooled over all tone + known prompts) >= 0.60 per conditioned arm per seed — NOT enforced by the
       lock (the lock checks numbers / names / fact words only); measurability (>= 10 lesion content words) is a
       require.
 instrument requires: (L) held valence != 0 on every pos/neg/ctrl row and == 0 on every lesion row, and every
  row ran the selection with >= 1 evaluated candidate; (O) priming differential == the committed NO-GO arm's;
  (C1) the DRAFT is identical across all 7 arms of a seed for every prompt (integrity: the draft must not see the
  affect state); (R) 6 distinct lesion realizations; (5b) pos == pos_rep byte-identical.
 Final: UNDEFINED if any require is unmet; else GO iff (1) and (P) and (3a-reply) and (C2); else NO-GO.

Literal scoring command:
  .venv/bin/python -m research.runners._lbf_affect_tone_selection_derisk --score-only
COMPUTE: brain numpy (SIM_BACKEND=numpy, the NO-GO organ numerics); Qwen on CUDA via tools/gpu_queue.sh, one seed
per queue line, arms sequential, each arm under tools/memcap.sh (D5 arms peaked at 8.8 GB maxrss -> cap 12).
"""
import argparse
import itertools
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

import research.runners._lbf_affect_tone_open_output_derisk as _BASE  # noqa: E402
import research.runners._lbf_affect_conditioned_mouth_derisk as _D5  # noqa: E402

SEEDS = list(_BASE.SEEDS)
ARMS = list(_BASE.ARMS) + ["pos_rep"]
CONDITIONED = ("pos", "neg", "ctrl_pos", "ctrl_neg", "pos_rep")
TONE_PROMPTS = list(_BASE.TONE_PROMPTS)
# D5's single known prompt left (3a-reply) unmeasurable on 3/6 seeds (<2 lesion fact words). Four more topics the
# shipped LTM (wikidata_100k) holds facts for are added so the content check is defined on every seed.
KNOWN_PROMPTS = list(_BASE.KNOWN_PROMPTS) + [
    "Tell me about wolfgang_amadeus_mozart", "Tell me about enrico_fermi", "Tell me about john_adams",
    "Tell me about john_von_neumann"]
OUT_ROOT = "research/findings/raw/_affect_tone_selection"
RUN_TAG = "run1"
C2_MIN = 0.60
C2_MIN_WORDS = 10
PERM_ALPHA = 0.05

_ENV = dict(_D5._ENV_QWEN)
_ENV["BRAIN_OPEN_ENDED_AFFECT_TONE_SELECT"] = "1"


def out_dir_for(tag=RUN_TAG):
    return os.path.join(OUT_ROOT, tag)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  WORKER (one arm, one fresh process)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def run_worker(seed, arm, out_path, prompts=None, known_prompts=None, decode_seed=None):
    import random
    import functools
    assert arm in ARMS
    decode_seed = int(seed if decode_seed is None else decode_seed)
    lesion = "1" if arm in ("lesion", "lesion_rep") else "0"
    priming = _BASE.PRIMING_NEG if arm in ("neg", "ctrl_neg") else _BASE.PRIMING_POS
    control = arm in ("ctrl_pos", "ctrl_neg")
    for k, v in _ENV.items():
        os.environ[k] = v
    os.environ.pop("BRAIN_OPEN_ENDED_AFFECT_CONDITIONED", None)
    os.environ["BRAIN_CHAT_SEED"] = str(seed)
    os.environ["BRAIN_AFFECT_LESION"] = lesion
    sys.path.insert(0, _REPO)
    t0 = time.time()
    import webapp.server as S
    from webapp import open_ended_chat as _OE
    from webapp import affect_tone_selection as _ATS

    _orig_answer_turn = _OE.answer_turn

    @functools.wraps(_orig_answer_turn)
    def _answer_turn_seeded(*a, **k):
        k["seed"] = decode_seed
        return _orig_answer_turn(*a, **k)
    _OE.answer_turn = _answer_turn_seeded

    ctrl_state = {"active": False, "idx": 0, "vals": []}
    if control:
        rng = random.Random(_BASE._CTRL_RNG_SEED)
        n = len(TONE_PROMPTS) + len(KNOWN_PROMPTS) + 4
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
        _ATS.LAST_TRACE.clear()
        resp = S.brain_chat(S.BrainChatRequest(session="lbf_affect_sel_%d_%s" % (seed, arm), message=msg,
                                               brain="tiny-demo", reset=reset, rich=True, renderer="stub"))
        return json.loads(bytes(resp.body)), json.loads(json.dumps(_ATS.LAST_TRACE))

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
                         "gen_seconds": oe.get("gen_seconds"), "selection": tr})
        return rows

    tone_rows = measure(prompts if prompts is not None else TONE_PROMPTS)
    known_rows = measure(known_prompts if known_prompts is not None else KNOWN_PROMPTS)
    try:
        import resource
        maxrss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        maxrss_gb = None
    out = {"runner": "_lbf_affect_tone_selection_derisk (worker)", "seed": seed, "arm": arm, "lesion": lesion,
           "control": control, "priming": priming, "decode_seed": decode_seed, "run_tag": RUN_TAG,
           "backend": os.environ.get("SIM_BACKEND"), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "resolved_ckpt": None, "ckpt_is_per_seed": True,
           "priming_affect": affect0, "tone_rows": tone_rows, "known_rows": known_rows,
           "maxrss_gb": maxrss_gb, "wall_seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print("[worker] seed=%d arm=%s wrote %s (%.1fs, maxrss %.1f GB)"
          % (seed, arm, out_path, out["wall_seconds"], maxrss_gb or -1), flush=True)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  CONTROLLER (sequential arms: one brain at a time)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def run_controller(seeds, memcap_gb=12, arms=None, tag=RUN_TAG, n_tone=None, n_known=None, allow_uncapped=False):
    od = os.path.join(_REPO, out_dir_for(tag))
    os.makedirs(od, exist_ok=True)
    memcap = os.path.join(_REPO, "tools", "memcap.sh")
    if not os.environ.get("XDG_RUNTIME_DIR") and os.path.isdir("/run/user/%d" % os.getuid()):
        os.environ["XDG_RUNTIME_DIR"] = "/run/user/%d" % os.getuid()
    use_memcap = bool(memcap_gb and os.path.exists(memcap) and _BASE._memcap_ok())
    print("[controller] memcap engaged=%s (%s GB)" % (use_memcap, memcap_gb), flush=True)
    if memcap_gb and not use_memcap and not allow_uncapped:
        raise SystemExit("[controller] memcap requested but systemd-run --user is unavailable -- refusing to run "
                         "uncapped (pass --allow-uncapped to override)")
    for s in seeds:
        for a in (arms or ARMS):
            op = _BASE._worker_out(od, s, a)
            if os.path.exists(op):
                try:
                    json.load(open(op))
                    continue
                except Exception:
                    pass
            cmd = [sys.executable, "-u", "-m", "research.runners._lbf_affect_tone_selection_derisk",
                   "--worker", "--seed", str(s), "--arm", a, "--out", op]
            if n_tone is not None:
                cmd += ["--n-tone", str(n_tone)]
            if n_known is not None:
                cmd += ["--n-known", str(n_known)]
            if use_memcap:
                cmd = ["bash", memcap, str(memcap_gb), "--"] + cmd
            env = dict(os.environ)
            env.update(_ENV)
            env.pop("BRAIN_OPEN_ENDED_AFFECT_CONDITIONED", None)
            t0 = time.time()
            with open(os.path.join(od, "arm_s%d_%s.log" % (s, a)), "w") as log:
                rc = subprocess.run(cmd, cwd=_REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            print("[controller] seed=%d arm=%s rc=%d wrote=%s (%.0fs)"
                  % (s, a, rc, os.path.exists(op), time.time() - t0), flush=True)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  CHECKS (pure; each has a planted-failure selftest)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_arms(od):
    A = {}
    for s in SEEDS:
        for a in ARMS:
            p = _BASE._worker_out(od, s, a)
            if os.path.exists(p):
                A[(s, a)] = json.load(open(p))
    return A


def _rows(d):
    return d["tone_rows"] + d["known_rows"]


def lever_check(A):
    """(L) held valence nonzero on conditioned rows, exactly 0 on lesion rows; every row ran the selection."""
    bad = []
    for (s, a), d in A.items():
        for r in _rows(d):
            tr = r.get("selection") or {}
            v = tr.get("v_held")
            ran = tr.get("mode") == "select" and int(tr.get("n_admissible", 0)) >= 1 and all(
                (tr["candidates"][i].get("eval") or {}).get("valence") is not None for i in tr.get("admissible", []))
            want = (v == 0.0) if a in ("lesion", "lesion_rep") else (v is not None and v != 0.0)
            if not (ran and want):
                bad.append((s, a, r["prompt"], v, ran))
    return (not bad and bool(A)), bad


def organ_check(A):
    """(O) the priming differential equals the committed NO-GO arm's (same brain state as the two NO-GOs + D5)."""
    bad = []
    for (s, a), d in A.items():
        ref = os.path.join(_REPO, _D5.NOGO_ARMS_DIR, "arm_s%d_%s.json" % (s, "pos" if a == "pos_rep" else a))
        if not os.path.exists(ref):
            bad.append((s, a, "no-ref"))
            continue
        want = (json.load(open(ref)).get("priming_affect") or {}).get("differential")
        got = (d.get("priming_affect") or {}).get("differential")
        if want is None or got is None or abs(float(want) - float(got)) > 1e-12:
            bad.append((s, a, want, got))
    return (not bad and bool(A)), bad


def draft_identity_check(A):
    """(C1) INTEGRITY: per seed and prompt, the affect-free DRAFT is byte-identical (sha) across all arms."""
    bad = []
    for s in SEEDS:
        shas = {}
        for a in ARMS:
            d = A.get((s, a))
            if d is None:
                bad.append((s, a, "missing"))
                continue
            for r in _rows(d):
                shas.setdefault(r["prompt"], set()).add((r.get("selection") or {}).get("draft_sha"))
        for p, ss in shas.items():
            if len(ss) != 1 or None in ss:
                bad.append((s, p, sorted(str(x) for x in ss)))
    return (not bad), bad


_EXCLUDE = None


def content_words(text):
    """Affect-neutral content words: len >= 4, not stop, not WARRINER (the appraisal lexicon), not the tone lexicon."""
    global _EXCLUDE
    if _EXCLUDE is None:
        from research.runners._affect_distributional_tag_derisk import WARRINER, STOP
        lex, _ = _BASE.load_indep_lexicon()
        _EXCLUDE = set(STOP) | set(WARRINER) | set(lex)
    return {w for w in _D5._words(text) if len(w) >= 4 and w not in _EXCLUDE}


def c2_check(A):
    """(C2) general content-word recall of every conditioned arm's replies vs the lesion replies, pooled per seed.
    Returns (measurable, ok, detail)."""
    measurable, ok, det = True, True, []
    for s in SEEDS:
        les = A.get((s, "lesion"))
        if les is None:
            measurable = False
            det.append({"seed": s, "error": "no lesion arm"})
            continue
        W = {r["prompt"]: content_words(r["raw"]) for r in _rows(les)}
        n_w = sum(len(v) for v in W.values())
        if n_w < C2_MIN_WORDS:
            measurable = False
            det.append({"seed": s, "n_lesion_content_words": n_w, "error": "< %d lesion content words" % C2_MIN_WORDS})
            continue
        for a in CONDITIONED:
            d = A.get((s, a))
            if d is None:
                measurable = False
                det.append({"seed": s, "arm": a, "error": "missing"})
                continue
            hit = sum(len(W.get(r["prompt"], set()) & content_words(r["raw"])) for r in _rows(d))
            rec = hit / n_w
            good = rec >= C2_MIN
            ok = ok and good
            det.append({"seed": s, "arm": a, "n_lesion_content_words": n_w, "recall": round(rec, 4), "ok": good})
    return measurable, (ok and measurable), det


def reply_content(A):
    """(3a-reply) D5's check verbatim, split into measurability (require) and the recall gate."""
    ok, det = _D5.reply_content_check(A)
    measurable = not any("error" in x for x in det)
    passed = ok and measurable
    return measurable, passed, det


def signflip_p(ds):
    """Exact one-sided sign-flip p over all 2^n sign assignments: P(mean(eps*d) >= mean(d))."""
    ds = [float(x) for x in ds]
    if not ds:
        return None
    obs = sum(ds) / len(ds)
    n_ge = 0
    tot = 0
    for eps in itertools.product((1.0, -1.0), repeat=len(ds)):
        tot += 1
        if sum(e * x for e, x in zip(eps, ds)) / len(ds) >= obs - 1e-12:
            n_ge += 1
    return n_ge / tot


def perm_check(per_seed):
    real = [per_seed[s]["real_directional_gap"] for s in SEEDS if per_seed.get(s, {}).get("complete")]
    ctrl = [per_seed[s]["ctrl_directional_gap"] for s in SEEDS if per_seed.get(s, {}).get("complete")]
    if len(real) != len(SEEDS):
        return None
    p_real, p_ctrl = signflip_p(real), signflip_p(ctrl)
    m_real, m_ctrl = sum(real) / len(real), sum(ctrl) / len(ctrl)
    return {"real_gaps": real, "mean_real": m_real, "p_real": p_real,
            "real_ok": bool(m_real > 0 and p_real <= PERM_ALPHA),
            "ctrl_gaps": ctrl, "mean_ctrl": m_ctrl, "p_ctrl": p_ctrl,
            "ctrl_clean": not (m_ctrl > 0 and p_ctrl <= PERM_ALPHA),
            "n_sign_assignments": 2 ** len(real)}


def diagnostics(A):
    """Reported, NOT gated: which candidate each arm released, lock pass rate per style, evoked-valence spread."""
    from collections import Counter
    styles, lock_pass, lock_n, vals = Counter(), Counter(), Counter(), {}
    held = {}
    for (s, a), d in A.items():
        held["%d|%s" % (s, a)] = sorted({r.get("affect_differential") for r in _rows(d)}, key=str)
        for r in _rows(d):
            tr = r.get("selection") or {}
            styles[(a, tr.get("selected_style"))] += 1
            for c in tr.get("candidates", []):
                lock_n[c["style"]] += 1
                lock_pass[c["style"]] += int(bool(c.get("lock_ok")))
                if c.get("eval"):
                    vals.setdefault(c["style"], []).append(c["eval"]["valence"])
    return {"held_differential_values_per_arm": held,
            "released_style_counts":{"%s|%s" % k: v for k, v in sorted(styles.items(), key=str)},
            "lock_pass_rate": {k: round(lock_pass[k] / lock_n[k], 3) for k in lock_n},
            "evoked_valence_mean": {k: round(sum(v) / len(v), 4) for k, v in vals.items() if v},
            "evoked_valence_nonzero_frac": {k: round(sum(1 for x in v if x != 0) / len(v), 3)
                                            for k, v in vals.items() if v}}


def score(tag=RUN_TAG, write=True):
    from tools.verdict import Verdict
    od = os.path.join(_REPO, out_dir_for(tag))
    A = load_arms(od)
    l_ok, l_bad = lever_check(A)
    o_ok, o_bad = organ_check(A)
    c1_ok, c1_bad = draft_identity_check(A)
    r_ok, r_det = _D5.mouth_replication_check(A)
    d_ok, d_det = _D5.conditioned_determinism_check(A)
    base = _BASE.score_and_gate(od, write_artifact=False, mouth="qwen")
    pm = perm_check(base["per_seed"])
    rc_meas, rc_ok, rc_det = reply_content(A)
    c2_meas, c2_ok, c2_det = c2_check(A)

    v = Verdict("affect SELECTS reply tone among content-locked restyles (Qwen mouth, 6-seed)")
    for p in base["preconditions"]:
        v.require(p["name"], p["ok"], True, p.get("note", ""))
    v.require("pos_rep arm present on every seed", all((s, "pos_rep") in A for s in SEEDS), True)
    v.require("(L) held valence !=0 on conditioned rows, ==0 on lesion rows; selection ran on every row", l_ok, True,
              "bad=%s" % (l_bad[:6] or "none"))
    v.require("(O) priming differential == committed NO-GO arm's", o_ok, True, "bad=%s" % (o_bad[:6] or "none"))
    v.require("(C1) INTEGRITY: affect-free draft identical across all arms, every seed/prompt", c1_ok, True,
              "bad=%s" % (c1_bad[:6] or "none"))
    v.require("(R) 6 distinct lesion realizations, decode_seed == brain seed", r_ok, True, json.dumps(r_det))
    v.require("(5b) pos == pos_rep byte-identical/seed", d_ok, True,
              "bad=%s" % ([x for x in d_det if not x["byte_identical"]] or "none"))
    v.require("(P-ctrl) the control's seed-level directional gap is NOT significant (sign-flip)",
              None if pm is None else pm["ctrl_clean"], True, json.dumps(pm) if pm else "incomplete")
    v.require("(3a-reply) measurable: >=2 lesion fact words per seed", rc_meas, True)
    v.require("(C2) measurable: >=%d lesion content words per seed" % C2_MIN_WORDS, c2_meas, True)
    gate = {"directional": bool(base["directional"]["go"]),
            "perm_real": bool(pm and pm["real_ok"]),
            "reply_content_3a": bool(rc_ok),
            "content_words_c2": bool(c2_ok)}
    decided = v.decide(all(gate.values()))
    art = dict(base)
    # tools.lab.attributable_to: whose is the directional gap -- the organ's held mood, or anything the ctrl arms
    # (mood-decoupled held valence, same priming) also carry? Reported beside the (P)/(P-ctrl) gates.
    from tools.lab import attributable_to
    art["attribution_fraction_recheck"] = attributable_to(
        "affect tone selection: real held mood vs mood-decoupled control",
        base.get("attribution_treatment_mean", 0.0), base.get("attribution_control_mean", 0.0))
    art.update({
        "runner": "_lbf_affect_tone_selection_derisk", "run_tag": tag, "arms": ARMS,
        "what": "affect selects the OPEN Qwen reply's tone among content-locked restyles of one affect-free draft; "
                "6-seed directional independent-lexicon lesion probe with content measured fixed",
        "known_prompts": KNOWN_PROMPTS, "gate_components": gate, "permutation": pm,
        "reply_content_3a": rc_det, "content_words_c2": c2_det, "c2_min": C2_MIN,
        "draft_identity_bad": c1_bad[:20], "lever_bad": l_bad[:20], "organ_bad": o_bad[:20],
        "mouth_replication": r_det, "conditioned_determinism": d_det, "diagnostics": diagnostics(A),
        "base_verdict": base["verdict"], "verdict": decided, "preconditions": decided["preconditions"],
        "GO": bool(decided["go"]),
        "claim_scope": "A GO shows: with the reply's content held fixed by an affect-free draft + a content lock (and "
                       "measured fixed by (3a-reply) and (C2)), the spiking organ's held valence selects a reply "
                       "whose tone moves directionally past the lesion null on an independent lexicon, and a "
                       "mood-decoupled control does not. It does NOT credit a spiking SELECTION (the comparator + "
                       "argmin are host), and it does not separate the organ from the host appraisal lexicon that "
                       "feeds it both the held mood and each candidate.",
        "shortcut_status": "host: appraisal lexicon (upstream), content lock, comparator+argmin selection, fixed "
                           "STYLES; brain: held mood + per-candidate evoked differential (spiking affect organ).",
        "honesty_boundary": "functional read-out only; never a felt/phenomenal claim."})
    if write:
        ap = os.path.join(od, "affect_tone_selection_verdict.json")
        json.dump(art, open(ap, "w"), indent=2, default=str)
        print("[score] wrote %s  status=%s gate=%s" % (ap, decided["status"], gate), flush=True)
    return art


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  SELFTEST (pure; no brain, no model) — every new check must fail in its failing direction
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def selftest():
    import re
    import tempfile
    from webapp import affect_tone_selection as ATS
    ok = True

    def check(name, cond):
        nonlocal ok
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))
        ok = ok and bool(cond)

    lex, _ = _BASE.load_indep_lexicon()
    words = set(re.findall(r"[a-z']+", " ".join([d for _, d in ATS.STYLES] + [ATS.RESTYLE_SYSTEM,
                                                                              ATS.NEUTRAL_MOOD_LINE]).lower()))
    check("style descriptors + restyle instruction + neutral MOOD line share NO word with the tone lexicon",
          not (words & set(lex)))
    check("flag unset -> off", (os.environ.pop("BRAIN_OPEN_ENDED_AFFECT_TONE_SELECT", None) or True)
          and not ATS.tone_select_enabled())
    src = open(os.path.join(_REPO, "webapp", "open_ended_chat.py")).read()
    i_gate = src.find('os.environ.get("BRAIN_OPEN_ENDED_AFFECT_TONE_SELECT"')
    i_imp = src.find("from webapp import affect_tone_selection")
    check("open_ended_chat imports the module ONLY inside the env-gated branch", 0 < i_gate < i_imp)
    check("signflip p: all-positive 6 -> 1/64", abs(signflip_p([1, 2, 3, 1, 2, 3]) - 1 / 64) < 1e-12)
    check("signflip p: all-zero -> 1.0 (a zero control is clean)", signflip_p([0.0] * 6) == 1.0)
    check("signflip p: mixed signs -> large", signflip_p([1, -1, 1, -1, 1, -1]) > 0.3)

    def fab(variant):
        A = {}
        for s in SEEDS:
            for a in ARMS:
                les = a in ("lesion", "lesion_rep")
                v = 0.0 if les else 0.15
                if variant == "lesion_nonzero" and s == 43 and a == "lesion":
                    v = 0.15
                dsha = "d%d" % s
                if variant == "draft_differs" and s == 44 and a == "neg":
                    dsha = "other"
                tone_raw = "The ocean waves roll across the sandy coastline under morning light near harbour towns."
                if variant == "paraphrase" and s == 100 and a == "neg":
                    tone_raw = "Water. Blue. Big. Wet stuff everywhere, honestly."
                tr = {"mode": "select", "v_held": v, "draft_sha": dsha, "n_admissible": 1, "admissible": [0],
                      "candidates": [{"style": "draft", "lock_ok": True, "eval": {"valence": 0.0}}]}
                A[(s, a)] = {"decode_seed": s, "priming_affect": {"differential": 0.0},
                             "tone_rows": [{"prompt": "t", "raw": tone_raw, "selection": tr}],
                             "known_rows": [{"prompt": "k", "raw": "He studied at the University of Wisconsin in "
                                             "Madison and worked with architects.",
                                             "facts": [["x", "educated_at", "university_of_wisconsin_madison"]],
                                             "selection": tr}]}
        return A
    clean = fab("clean")
    check("(L) passes clean", lever_check(clean)[0] is True)
    check("(L) FAILS when a lesion row carries a nonzero held valence", lever_check(fab("lesion_nonzero"))[0] is False)
    check("(C1) passes clean", draft_identity_check(clean)[0] is True)
    check("(C1) FAILS when one arm's draft differs", draft_identity_check(fab("draft_differs"))[0] is False)
    m, g, _ = c2_check(clean)
    check("(C2) measurable + passes clean", m and g)
    m, g, det = c2_check(fab("paraphrase"))
    check("(C2) FAILS when a conditioned reply paraphrases the content away", m and not g)
    check("(3a-reply) passes clean (shared D5 check)", reply_content(clean)[1] is True)
    pm = perm_check({s: {"complete": True, "real_directional_gap": 0.3, "ctrl_directional_gap": 0.0} for s in SEEDS})
    check("(P) real passes + ctrl clean on a consistent real effect and a zero control", pm["real_ok"] and pm["ctrl_clean"])
    pm = perm_check({s: {"complete": True, "real_directional_gap": 0.3, "ctrl_directional_gap": 0.3} for s in SEEDS})
    check("(P-ctrl) FAILS when the control shows the same consistent effect", pm["ctrl_clean"] is False)
    pm = perm_check({s: {"complete": True, "real_directional_gap": g, "ctrl_directional_gap": 0.0}
                     for s, g in zip(SEEDS, [0.3, -0.3, 0.3, -0.3, 0.3, -0.2])})
    check("(P) real FAILS on sign-inconsistent gaps", pm["real_ok"] is False)
    with tempfile.TemporaryDirectory() as td:
        check("(L)/(O) fail on an empty run (UNDEFINED, never a pass)",
              lever_check(load_arms(td))[0] is False and organ_check(load_arms(td))[0] is False)
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--controller", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--arm", type=str, default="pos")
    ap.add_argument("--arms", type=str, nargs="+", default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--tag", type=str, default=RUN_TAG)
    ap.add_argument("--n-tone", type=int, default=None)
    ap.add_argument("--n-known", type=int, default=None)
    ap.add_argument("--memcap-gb", type=int, default=12)
    ap.add_argument("--allow-uncapped", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    elif args.worker:
        run_worker(args.seed, args.arm,
                   args.out or _BASE._worker_out(os.path.join(_REPO, out_dir_for(args.tag)), args.seed, args.arm),
                   prompts=(TONE_PROMPTS[:args.n_tone] if args.n_tone is not None else None),
                   known_prompts=(KNOWN_PROMPTS[:args.n_known] if args.n_known is not None else None))
    elif args.score_only:
        score(args.tag)
    else:
        run_controller(args.seeds, memcap_gb=args.memcap_gb, arms=args.arms, tag=args.tag, n_tone=args.n_tone,
                       n_known=args.n_known, allow_uncapped=args.allow_uncapped)
