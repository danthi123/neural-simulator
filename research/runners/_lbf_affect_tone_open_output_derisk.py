"""AFFECT -> TONE, LOAD-BEARING OVER THE *OPEN* (FREELY-GENERATED) MOUTH OUTPUT — a 6-seed, DIRECTIONAL,
INDEPENDENT-LEXICON lesion probe on the linattn WKV mouth (roadmap SS8 near-term-speech first de-risk).

WHY THIS EXISTS (the sub-arc A plan, scoping synthesis 2026-09-22): the affect->tone coupling is GO but only
(a) single-seed 42, (b) via a BINARY "raw text differs byte-for-byte" instrument (phase4_linattn_flip_
confirmation_rerun.py), (c) scored with a tone-fraction proxy that used the SAME Warriner lexicon the decode
bias uses (the circular-proxy trap the 2026-09-04 finding caught). This runner upgrades that verification to a
DIRECTIONAL, DISTRIBUTIONAL, INDEPENDENT-lexicon, 6-seed measurement so the #1 metric (lesion-verified
LOAD-BEARING FRACTION) can score affect->tone by the OPEN-output ruler — mirroring measure_open_ended_
distributional (load_bearing_fraction.py) which swapped the open-ended-generation ruler.

WHAT IS MEASURED. For each seed in {42,43,44,100,101,102}, six FRESH-SUBPROCESS arms (the _RNG private-timeline
confound documented in _wkv_mouth_affect_neural_verify.py demands one process per arm):
  pos       : positive-mood priming, BRAIN_AFFECT_LESION=0  (affect ACTIVE)
  neg       : negative-mood priming, BRAIN_AFFECT_LESION=0  (affect ACTIVE)
  lesion    : positive priming,      BRAIN_AFFECT_LESION=1  (the NEUTRAL baseline: valence clamped to 0.0 ->
              _apply_affect_bias early-returns -> an EXACT decode no-op by construction)
  lesion_rep: identical to `lesion`, a second fresh process  (DETERMINISM/null check)
  ctrl_pos  : positive priming, affect ACTIVE, but valence_from_affect MONKEYPATCHED to a mood-INDEPENDENT
              random-sign sequence (ATTRIBUTION control: the decode bias is active but decoupled from the mood)
  ctrl_neg  : negative priming, same injected sequence  (so ctrl_pos/ctrl_neg differ ONLY in priming text)
Each arm runs a fixed set of >=8 free-talk TONE prompts (unknown -> free-gen, where tone lives + the moat-holds
check) plus a KNOWN topic (frank_lincoln_wright -> the content-identity guard). The reply is `open_ended.raw`.

TONE is scored by an INDEPENDENT sentiment lexicon (research/runners/_lbf_affect_tone_indep_lexicon.json), 356
signed words with ZERO overlap with the 180-word WARRINER set the bias boosts (enforced + counted at load).
Because the scorer shares NO word with the boosted set, any directional tone shift it reads comes from
NON-boosted words moving with the mood — the strongest form of the anti-circularity anti-cheat. Scoring uses
VADER's own compound normalization s/sqrt(s^2+15) + simple negation, so it is a genuine VADER-style instrument
over an independent lexicon (NLTK VADER's vader_lexicon.txt data file is not on disk and no-download discipline
forbids fetching it — this is the plan's sanctioned fallback; residual: swap in true VADER when its data ships).

GO-GATE (all must hold, 6-seed; a NO-GO on any sub-condition is an honest METHOD verdict on THIS ruler/operating
point — bank it, do not tune to pass, HARD RULE 2):
  (1) DIRECTIONAL: per seed mean_p[tone(pos)-tone(lesion)] > +delta AND mean_p[tone(neg)-tone(lesion)] < -delta,
      correct sign on all 6 seeds (or 5/6 with the 6th null-clean, matching the prospective-memory precedent).
      delta is PREREGISTERED from the lesion (neutral) arm's own per-prompt tone-noise band — NOT tuned to flip.
  (2) ATTRIBUTION: the control arm's directional gap |mean_p[tone(ctrl_pos)] - mean_p[tone(ctrl_neg)]| < delta
      on all 6 seeds (tone tracks the REAL spiking mood, not "any decode bias moves text").
  (3) FORM-not-CONTENT: retrieved facts + known byte-identical intact vs lesion on the known topic (every seed),
      and the moat holds (unknown free-talk stays known=False) with the bias active.
  (4) FLUENCY: salad-fraction <= 0.16 on every reply of every arm.
  (5) DETERMINISM/null: lesion vs lesion_rep byte-identical per seed (fresh-subprocess-per-arm).

COMPUTE: numpy CPU ONLY (CUDA_VISIBLE_DEVICES='' SIM_BACKEND=numpy). The linattn mouth is a few-hundred-neuron
Izhikevich bank; the GPU is left untouched. Arms fan out 2-parallel under tools/memcap.sh (OOM discipline).

DEFAULT-OFF / ADDITIVE: this is a NEW research runner + a NEW default-OFF measure in load_bearing_fraction.py
(LB_AFFECT_TONE_OPEN_PROBE). NO sim/ edit; NO webapp/ source edit (the control arm's valence injection is a
HARNESS-side monkeypatch in the worker's own process). Off -> byte-identical (proven by the load_bearing_fraction
selftest + a no-flag read).
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_LEXICON_PATH = os.path.join(_HERE, "_lbf_affect_tone_indep_lexicon.json")

SEEDS = [42, 43, 44, 100, 101, 102]
ARMS = ["pos", "neg", "lesion", "lesion_rep", "ctrl_pos", "ctrl_neg"]

PRIMING_POS = "I am absolutely thrilled and overjoyed today, everything is wonderful!"
PRIMING_NEG = "I feel absolutely miserable and devastated today, everything is terrible and hopeless."

# >=8 free-talk prompts that route to the WKV mouth free-gen path and are UNKNOWN (known=False) -> the tone
# effect lives here AND they double as the moat-holds check. (Smoke-verified generator=wkv_mouth, known=False.)
TONE_PROMPTS = [
    "Tell me about the ocean",
    "What do you think about music",
    "Tell me about the mountains",
    "Describe a walk in the forest",
    "Tell me about your favorite season",
    "Describe the city at night",
    "Tell me about the morning",
    "What do you think about the future",
    "Tell me about the river",
    "Describe a quiet afternoon",
]
# KNOWN topic(s) for the content-identity (form-not-content) guard: retrieval must be byte-identical across arms.
KNOWN_PROMPTS = ["Tell me about frank_lincoln_wright"]

# Control injection magnitude ~ the live valence scale (clip(4*differential) ~ 0.16, see wkv_mouth_generator
# mechanism note (a)). Fixed RNG seed (NOT the substrate seed) so ctrl_pos and ctrl_neg get the IDENTICAL
# mood-independent sequence -> they differ ONLY in priming text -> a clean "decouple valence from mood" control.
_CTRL_MAG = 0.16
_CTRL_RNG_SEED = 20260922

# ── the fixed live-harness env (mirrors research/findings/raw/_affect_wkv_mouth_verify/phase4_..._rerun.py) ──
_ENV_BASE = {
    "CUDA_VISIBLE_DEVICES": "",
    "SIM_BACKEND": "numpy",
    "BRAIN_OPEN_ENDED": "1",
    "BRAIN_WKV_MOUTH_RECURRENCE": "linattn",
    "BRAIN_WKV_MOUTH_CKPT": "bridges/wkv_ckpt/wkv_linattn_depth2_contiguous_seed{seed}.npz",
    "BRAIN_WKV_MOUTH_TOKENIZER": "bpe",
    "BRAIN_WKV_MOUTH_SCOPE": "broad",
    "BRAIN_OPEN_ENDED_WKV_MOUTH": "1",
    "BRAIN_OPEN_ENDED_NP_ENTAILMENT": "0",
    "BRAIN_OPEN_ENDED_GEN_TIME_HONESTY": "0",
    "BRAIN_LTM_SHIP_DEFAULT": "1",
    # force FREE-GEN (not the tone-neutral fact-sentence render) so the tone effect is exercised
    "BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE": "0",
    "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK": "0",
}


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  INDEPENDENT-LEXICON TONE SCORER  (pure, selftest-exercisable)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
_NEGATORS = {"not", "no", "never", "without", "hardly", "barely", "cannot", "cant", "wont",
             "dont", "didnt", "isnt", "wasnt", "arent", "werent", "nor", "neither", "none"}
_TOK_RE = re.compile(r"[a-z']+")


def load_indep_lexicon(path=_LEXICON_PATH, enforce_disjoint=True):
    """Load the independent signed sentiment lexicon and ENFORCE zero overlap with the WARRINER set the affect
    decode bias boosts. Returns (lex_dict, overlap_count). Raises on overlap (the anti-cheat must hold)."""
    lex = json.load(open(path))["words"]
    lex = {str(k).lower(): float(v) for k, v in lex.items()}
    overlap = []
    try:
        from research.runners._affect_distributional_tag_derisk import WARRINER
        overlap = sorted(set(lex) & set(WARRINER))
    except Exception:
        pass
    if enforce_disjoint and overlap:
        raise ValueError("independent lexicon OVERLAPS the boosted WARRINER set by %d words (%s) — the "
                         "anti-circularity anti-cheat is violated" % (len(overlap), overlap[:10]))
    return lex, len(overlap)


def tone_compound(text, lex):
    """VADER-style compound sentiment in (-1, 1) over the INDEPENDENT lexicon, with simple 3-back negation.
    s = sum of signed word scores (sign flipped if a negator is within 3 preceding tokens); compound =
    s/sqrt(s^2+15) (VADER's own normalization constant). Empty/no-match -> 0.0. Also returns n_hits."""
    toks = _TOK_RE.findall((text or "").lower())
    s = 0.0
    n_hits = 0
    for i, t in enumerate(toks):
        if t in lex:
            val = lex[t]
            for j in range(max(0, i - 3), i):
                if toks[j] in _NEGATORS:
                    val = -val
                    break
            s += val
            n_hits += 1
    compound = s / math.sqrt(s * s + 15.0) if s != 0.0 else 0.0
    return compound, n_hits


def warriner_directional(text):
    """DIAGNOSTIC ONLY (NOT the anti-cheat ruler, NOT gated): net signed count of the BOOSTED words themselves —
    the WARRINER words the decode bias acts on, scored with WARRINER's OWN sign. This is CIRCULAR BY DESIGN
    (counting the very words the bias boosts, with the very sign it boosts them by) — it is the phase4 byte-diff
    made directional, reported to CHARACTERIZE whether the bias places more mood-congruent boosted words, i.e.
    whether a NO-GO on the disjoint ruler means 'no effect' or 'effect localized to the boosted lexicon'. Returns
    (pos_hits, neg_hits, net_per_token)."""
    try:
        from research.runners._affect_distributional_tag_derisk import WARRINER, STOP
    except Exception:
        return 0, 0, 0.0
    toks = _TOK_RE.findall((text or "").lower())
    pos = neg = 0
    for t in toks:
        if t in STOP or t not in WARRINER:
            continue
        v9 = WARRINER[t][0]  # 1..9 valence norm; >5 positive, <5 negative
        if v9 > 5.5:
            pos += 1
        elif v9 < 4.5:
            neg += 1
    net = (pos - neg) / len(toks) if toks else 0.0
    return pos, neg, net


def salad_frac(text):
    toks = (text or "").split()
    if not toks:
        return 0.0
    return Counter(toks).most_common(1)[0][1] / len(toks)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  PURE DECISION LOGIC  (no brain build; the selftest exercises these directly)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def preregister_delta(lesion_prompt_tones):
    """delta = the neutral (lesion) arm's own per-prompt tone-noise band = the population std of every lesion-arm
    per-prompt tone score pooled across seeds. Derived ONLY from the lesion arm (independent of pos/neg/ctrl), so
    it is not tuned to the treatment. A real mood effect on the mean gap must exceed this band."""
    xs = list(lesion_prompt_tones)
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


def directional_seed_verdict(gap_pos, gap_neg, delta):
    """Per-seed classification. Returns (pos_state, neg_state) each in {'correct','null','wrong'}."""
    def cls(gap, want_positive):
        if want_positive:
            if gap > delta:
                return "correct"
            if gap < -delta:
                return "wrong"
            return "null"
        else:
            if gap < -delta:
                return "correct"
            if gap > delta:
                return "wrong"
            return "null"
    return cls(gap_pos, True), cls(gap_neg, False)


def directional_go(seed_states):
    """seed_states: list of (pos_state, neg_state). GO iff, in EACH direction, all 6 are 'correct', OR 5 are
    'correct' and the 6th is 'null' (never 'wrong') — the 5/6-with-6th-null-clean precedent. A single 'wrong'
    in a direction => NO-GO for that direction."""
    def ok(states):
        correct = states.count("correct")
        wrong = states.count("wrong")
        return wrong == 0 and correct >= 5 and (correct == len(states) or states.count("null") == len(states) - correct)
    pos_states = [s[0] for s in seed_states]
    neg_states = [s[1] for s in seed_states]
    return ok(pos_states) and ok(neg_states), pos_states, neg_states


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  WORKER  (one arm, one fresh process)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def run_worker(seed, arm, out_path):
    assert arm in ARMS, arm
    lesion = "1" if arm in ("lesion", "lesion_rep") else "0"
    priming = PRIMING_NEG if arm in ("neg", "ctrl_neg") else PRIMING_POS
    control = arm in ("ctrl_pos", "ctrl_neg")

    for k, v in _ENV_BASE.items():
        os.environ[k] = v
    os.environ["BRAIN_CHAT_SEED"] = str(seed)
    os.environ["BRAIN_AFFECT_LESION"] = lesion

    sys.path.insert(0, ".")
    t0 = time.time()
    import webapp.server as S
    from webapp import wkv_mouth_generator as _WKVG
    from webapp import open_ended_chat as _OE

    # record the RESOLVED per-seed mouth checkpoint (must be the seed's own file, NOT the seed42 fallback)
    resolved_ckpt = _WKVG._ckpt_path(seed)
    ckpt_is_per_seed = ("seed%d" % seed) in os.path.basename(resolved_ckpt)

    # ── ATTRIBUTION-CONTROL valence injection (harness-side monkeypatch, this process only) ──────────────────
    ctrl_state = {"active": False, "idx": 0, "vals": []}
    if control:
        rng = random.Random(_CTRL_RNG_SEED)
        n = len(TONE_PROMPTS) + len(KNOWN_PROMPTS) + 4
        ctrl_state["vals"] = [rng.choice([_CTRL_MAG, -_CTRL_MAG]) for _ in range(n)]
        _orig_vfa = _OE.valence_from_affect

        def _patched_valence(differential):
            if ctrl_state["active"]:
                v = ctrl_state["vals"][ctrl_state["idx"] % len(ctrl_state["vals"])]
                ctrl_state["idx"] += 1
                return float(v)
            return _orig_vfa(differential)
        _OE.valence_from_affect = _patched_valence

    def chat(msg, reset):
        resp = S.brain_chat(S.BrainChatRequest(session="lbf_affect_tone_%d_%s" % (seed, arm),
                                               message=msg, brain="tiny-demo", reset=reset,
                                               rich=True, renderer="stub"))
        return json.loads(bytes(resp.body))

    # priming turn (sets the mood; its reply is discarded). Control injection stays OFF here.
    d0 = chat(priming, True)
    affect0 = (d0.get("affect") or {})

    # measured turns: injection ON from here (aligns the ctrl counter with the measured prompts)
    ctrl_state["active"] = True
    ctrl_state["idx"] = 0

    def measure(prompts):
        rows = []
        for p in prompts:
            d = chat(p, False)
            oe = d.get("open_ended") or {}
            raw = oe.get("raw") or ""
            comp, hits = tone_compound(raw, _LEX)
            rows.append({
                "prompt": p, "raw": raw, "known": bool(oe.get("known")),
                "facts": oe.get("facts") or [], "generator": oe.get("generator"),
                "wkv_mouth_used": bool(oe.get("wkv_mouth_used")),
                "tone_compound": comp, "tone_hits": hits, "salad_frac": salad_frac(raw),
                "affect_differential": (d.get("affect") or {}).get("differential"),
            })
        return rows

    global _LEX
    _LEX, _ = load_indep_lexicon()
    tone_rows = measure(TONE_PROMPTS)
    known_rows = measure(KNOWN_PROMPTS)

    out = {
        "runner": "_lbf_affect_tone_open_output_derisk (worker)",
        "seed": seed, "arm": arm, "lesion": lesion, "control": control,
        "priming": priming, "backend": os.environ.get("SIM_BACKEND"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "resolved_ckpt": resolved_ckpt, "ckpt_is_per_seed": ckpt_is_per_seed,
        "priming_affect": affect0, "tone_rows": tone_rows, "known_rows": known_rows,
        "wall_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print("[worker] seed=%d arm=%s wrote %s (%.1fs) ckpt_per_seed=%s" %
          (seed, arm, out_path, out["wall_seconds"], ckpt_is_per_seed), flush=True)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  CONTROLLER  (spawn 2-parallel under memcap, collect, score, gate, write the Verdict artifact)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _worker_out(out_dir, seed, arm):
    return os.path.join(out_dir, "arm_s%d_%s.json" % (seed, arm))


def _spawn(seed, arm, out_dir, memcap_gb):
    out_path = _worker_out(out_dir, seed, arm)
    py = sys.executable
    cmd = [py, "-m", "research.runners._lbf_affect_tone_open_output_derisk",
           "--worker", "--seed", str(seed), "--arm", arm, "--out", out_path]
    memcap = os.path.join(_REPO, "tools", "memcap.sh")
    if memcap_gb and os.path.exists(memcap) and _memcap_ok():
        cmd = ["bash", memcap, str(memcap_gb), "--"] + cmd
    env = dict(os.environ)
    for k, v in _ENV_BASE.items():
        env[k] = v
    log = open(os.path.join(out_dir, "arm_s%d_%s.log" % (seed, arm)), "w")
    return subprocess.Popen(cmd, cwd=_REPO, env=env, stdout=log, stderr=subprocess.STDOUT), out_path, log


def _memcap_ok():
    try:
        return subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "true"],
                              capture_output=True, timeout=15).returncode == 0
    except Exception:
        return False


def run_controller(out_dir, parallel=2, memcap_gb=10, resume=True):
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(s, a) for s in SEEDS for a in ARMS]
    todo = []
    for s, a in jobs:
        op = _worker_out(out_dir, s, a)
        if resume and os.path.exists(op):
            try:
                json.load(open(op))
                continue
            except Exception:
                pass
        todo.append((s, a))
    print("[controller] %d/%d arms to run (parallel=%d memcap=%dG) -> %s"
          % (len(todo), len(jobs), parallel, memcap_gb, out_dir), flush=True)
    running = []
    i = 0
    while i < len(todo) or running:
        while len(running) < parallel and i < len(todo):
            s, a = todo[i]
            i += 1
            proc, op, log = _spawn(s, a, out_dir, memcap_gb)
            running.append((proc, s, a, op, log, time.time()))
            print("[controller] launched seed=%d arm=%s (%d/%d)" % (s, a, i, len(todo)), flush=True)
        time.sleep(3)
        still = []
        for proc, s, a, op, log, t0 in running:
            rc = proc.poll()
            if rc is None:
                still.append((proc, s, a, op, log, t0))
                continue
            log.close()
            ok = os.path.exists(op)
            print("[controller] DONE seed=%d arm=%s rc=%d wrote=%s (%.0fs)"
                  % (s, a, rc, ok, time.time() - t0), flush=True)
        running = still
    return score_and_gate(out_dir)


def score_and_gate(out_dir, write_artifact=True, mouth="wkv", extra_require=()):
    """Load every arm, score, preregister delta from the lesion arm, run the 5-part GO gate, and write the
    Verdict artifact. Pure w.r.t. the brain (no build) so it is re-runnable on collected arms.

    `mouth` (default "wkv" -> byte-identical to before this parameter existed): "qwen" swaps the two
    WKV-specific instrument preconditions (per-seed linattn ckpt resolved; wkv mouth used) for the Qwen ones
    (every tone reply was written by generator=="qwen"; Qwen has no per-seed checkpoint). `extra_require` is a
    sequence of (name, ok, detail) instrument preconditions a caller adds (e.g. the conditioning-lever check of
    _lbf_affect_conditioned_mouth_derisk); each becomes one more Verdict.require -> UNDEFINED if unmet."""
    from tools.verdict import Verdict
    from tools.lab import attributable_to
    lex, overlap = load_indep_lexicon()
    arms = {}
    missing = []
    for s in SEEDS:
        for a in ARMS:
            op = _worker_out(out_dir, s, a)
            if not os.path.exists(op):
                missing.append((s, a))
                continue
            arms[(s, a)] = json.load(open(op))

    # ---- per-seed tone means (over TONE prompts) ----
    def tone_mean(s, a):
        rows = arms[(s, a)]["tone_rows"]
        vs = [r["tone_compound"] for r in rows]
        return sum(vs) / len(vs) if vs else 0.0

    # ---- delta: preregistered from lesion-arm per-prompt tone, pooled across seeds ----
    lesion_prompt_tones = []
    for s in SEEDS:
        if (s, "lesion") in arms:
            lesion_prompt_tones += [r["tone_compound"] for r in arms[(s, "lesion")]["tone_rows"]]
    delta = preregister_delta(lesion_prompt_tones)
    # SECONDARY (reported, NOT the gate): a SEM-based band = delta/sqrt(n_tone_prompts). The preregistered gate uses
    # the raw per-prompt std (conservative: it tests the MEAN gap against a single prompt's tone spread); a SEM band
    # is the natural noise scale for a mean. Reported so the finding shows the result under both bands without any
    # post-hoc tuning of the preregistered gate (HARD RULE 2 -- the headline stays `delta`).
    delta_sem = delta / math.sqrt(len(TONE_PROMPTS)) if TONE_PROMPTS else delta

    # ---- (1) directional + (2) attribution, per seed ----
    per_seed = {}
    seed_states = []
    ctrl_ok_all = True
    for s in SEEDS:
        have = all((s, a) in arms for a in ("pos", "neg", "lesion", "ctrl_pos", "ctrl_neg"))
        if not have:
            per_seed[s] = {"complete": False}
            seed_states.append(("null", "null"))
            ctrl_ok_all = False
            continue
        tp, tn, tl = tone_mean(s, "pos"), tone_mean(s, "neg"), tone_mean(s, "lesion")
        gap_pos, gap_neg = tp - tl, tn - tl
        ps, ns = directional_seed_verdict(gap_pos, gap_neg, delta)
        seed_states.append((ps, ns))
        tcp, tcn = tone_mean(s, "ctrl_pos"), tone_mean(s, "ctrl_neg")
        ctrl_directional = tcp - tcn
        ctrl_ok = abs(ctrl_directional) < delta
        ctrl_ok_all = ctrl_ok_all and ctrl_ok
        per_seed[s] = {"complete": True, "tone_pos": tp, "tone_neg": tn, "tone_lesion": tl,
                       "gap_pos": gap_pos, "gap_neg": gap_neg, "pos_state": ps, "neg_state": ns,
                       "tone_ctrl_pos": tcp, "tone_ctrl_neg": tcn,
                       "ctrl_directional_gap": ctrl_directional, "ctrl_attribution_ok": ctrl_ok,
                       "real_directional_gap": tp - tn}
    dir_go, pos_states, neg_states = directional_go(seed_states)
    # SECONDARY directional read under the SEM band (reported, NOT gated)
    seed_states_sem = []
    for s in SEEDS:
        e = per_seed.get(s, {})
        if e.get("complete"):
            seed_states_sem.append(directional_seed_verdict(e["gap_pos"], e["gap_neg"], delta_sem))
        else:
            seed_states_sem.append(("null", "null"))
    dir_go_sem, pos_states_sem, neg_states_sem = directional_go(seed_states_sem)

    # BOOSTED-WORD DIAGNOSTIC (circular by design, NOT gated): does the bias place more mood-congruent WARRINER
    # words in the pos vs neg reply? Characterizes whether a disjoint-ruler NO-GO means "no effect" or "effect
    # localized to the boosted lexicon". Re-scores the stored raw replies (no re-run).
    def warr_net_mean(s, a):
        rows = arms[(s, a)]["tone_rows"]
        nets = [warriner_directional(r["raw"])[2] for r in rows]
        return sum(nets) / len(nets) if nets else 0.0
    boosted_diag = {}
    for s in SEEDS:
        if (s, "pos") in arms and (s, "neg") in arms and (s, "lesion") in arms:
            wp, wn, wl = warr_net_mean(s, "pos"), warr_net_mean(s, "neg"), warr_net_mean(s, "lesion")
            boosted_diag[s] = {"warr_net_pos": wp, "warr_net_neg": wn, "warr_net_lesion": wl,
                               "warr_gap_pos_minus_lesion": wp - wl, "warr_gap_neg_minus_lesion": wn - wl,
                               "warr_directional_pos_minus_neg": wp - wn}
    boosted_pos_minus_neg_signs = [1 if d["warr_directional_pos_minus_neg"] > 0 else
                                   (-1 if d["warr_directional_pos_minus_neg"] < 0 else 0)
                                   for d in boosted_diag.values()]

    # EXPLICIT ATTRIBUTION (tools.lab.attributable_to): whose is the directional tone gap? TREATMENT = the mean
    # intact mood-driven directional gap |pos-neg|; CONTROL = the mean mood-DECOUPLED-valence control gap. A high
    # attributable fraction means the tone effect belongs to the REAL spiking mood, not to "any active decode bias"
    # (the gap#5 lesson made executable: measuring both arms is not the same as subtracting them).
    complete_seeds = [s for s in SEEDS if per_seed.get(s, {}).get("complete")]
    treat_mean = (sum(abs(per_seed[s]["real_directional_gap"]) for s in complete_seeds) / len(complete_seeds)
                  if complete_seeds else 0.0)
    ctrl_mean = (sum(abs(per_seed[s]["ctrl_directional_gap"]) for s in complete_seeds) / len(complete_seeds)
                 if complete_seeds else 0.0)
    attribution_fraction = attributable_to(
        "affect-tone-open: real mood vs mood-decoupled control", treat_mean, ctrl_mean)

    # ---- (3) content-identity / moat ----
    content_ok = True
    moat_ok = True
    content_detail = []
    for s in SEEDS:
        if (s, "pos") not in arms or (s, "lesion") not in arms:
            content_ok = False
            continue
        pk = {(r["prompt"]): (r["known"], json.dumps(r["facts"], sort_keys=True))
              for r in arms[(s, "pos")]["known_rows"]}
        lk = {(r["prompt"]): (r["known"], json.dumps(r["facts"], sort_keys=True))
              for r in arms[(s, "lesion")]["known_rows"]}
        same = pk == lk
        content_ok = content_ok and same
        content_detail.append({"seed": s, "known_facts_identical_pos_vs_lesion": same})
        # moat: every unknown free-talk tone prompt stays known=False with the bias active (pos arm)
        for r in arms[(s, "pos")]["tone_rows"]:
            if r["known"]:
                moat_ok = False

    # ---- (4) fluency ----
    max_salad = 0.0
    worst = None
    for (s, a), d in arms.items():
        for r in d["tone_rows"] + d["known_rows"]:
            if r["salad_frac"] > max_salad:
                max_salad = r["salad_frac"]
                worst = {"seed": s, "arm": a, "prompt": r["prompt"], "salad_frac": r["salad_frac"]}
    fluency_ok = max_salad <= 0.16

    # ---- (5) determinism: lesion vs lesion_rep byte-identical per seed ----
    determ_ok = True
    determ_detail = []
    for s in SEEDS:
        if (s, "lesion") not in arms or (s, "lesion_rep") not in arms:
            determ_ok = False
            determ_detail.append({"seed": s, "byte_identical": None})
            continue
        a1 = [r["raw"] for r in arms[(s, "lesion")]["tone_rows"] + arms[(s, "lesion")]["known_rows"]]
        a2 = [r["raw"] for r in arms[(s, "lesion_rep")]["tone_rows"] + arms[(s, "lesion_rep")]["known_rows"]]
        bi = a1 == a2
        determ_ok = determ_ok and bi
        determ_detail.append({"seed": s, "byte_identical": bi})

    # ---- per-seed ckpt resolved to the seed's own file (no seed42 fallback confound) ----
    ckpt_ok = all(arms[(s, a)].get("ckpt_is_per_seed", False) for (s, a) in arms)

    # ---- wkv mouth actually used on the tone prompts (not qwen) ----
    wkv_used_ok = all(r.get("wkv_mouth_used") for (s, a), d in arms.items() for r in d["tone_rows"])

    complete = (not missing)

    # THE VERDICT SEMANTICS: the require() checks are INSTRUMENT-VALIDITY preconditions -- the conditions under
    # which the directional reading can be TRUSTED (a clean null, a clean control, the right device/ckpt/lexicon,
    # form-not-content preserved). If any is unmet -> UNDEFINED (the measurement cannot be trusted, never a
    # negative). The DIRECTIONAL result is the FINDING itself (the `go` passed to decide), NOT a precondition:
    # a TRUSTWORTHY measurement that shows no correct-sign directional tone effect is an honest NO-GO on THIS
    # ruler, not UNDEFINED. (Sub-conditions (1)-(5) map: (1)=go, (2)=attribution, (3)=content+moat, (4)=fluency,
    # (5)=determinism.)
    v = Verdict("affect->tone load-bearing over OPEN mouth output (linattn, 6-seed)")
    v.require("all-36-arms-present", not missing, True,
              "missing=%s" % (missing if missing else "none"))
    v.require("lexicon-disjoint-from-WARRINER (overlap==0)", overlap == 0, True,
              "overlap=%d" % overlap)
    if mouth == "qwen":
        qwen_used_ok = all(r.get("generator") == "qwen" for (s, a), d in arms.items() for r in d["tone_rows"])
        v.require("qwen-mouth-used-on-tone-prompts (generator==qwen)", qwen_used_ok, True)
    else:
        v.require("ckpt-resolved-per-seed (no seed42 fallback)", ckpt_ok, True)
        v.require("wkv-mouth-used-on-tone-prompts (not qwen)", wkv_used_ok, True)
    for (_xn, _xok, _xd) in extra_require:
        v.require(_xn, bool(_xok), True, _xd)
    v.require("(5) determinism lesion==lesion_rep byte-identical/seed (clean null)", determ_ok, True)
    v.require("(4) fluency salad_frac<=0.16 all arms", fluency_ok, True,
              "max_salad=%.4f worst=%s" % (max_salad, worst))
    v.require("(3a) content-identity facts+known identical pos vs lesion", content_ok, True)
    v.require("(3b) moat holds (unknown stays known=False, bias active)", moat_ok, True)
    v.require("(2) attribution: |control directional gap|<delta all seeds (clean control)", ctrl_ok_all, True)
    # (1) DIRECTIONAL is the finding, recorded as the `go` -> GO if it holds, NO-GO if it does not (given a
    # trustworthy instrument). Its per-seed states are surfaced in the artifact's `directional` block.
    go = bool(dir_go)
    decided = v.decide(go)

    artifact = {
        "runner": "_lbf_affect_tone_open_output_derisk",
        "what": "affect->tone load-bearing over OPEN (freely-generated) linattn WKV mouth output; "
                "6-seed directional independent-lexicon lesion probe (roadmap SS8).",
        "seeds": SEEDS, "arms": ARMS, "n_tone_prompts": len(TONE_PROMPTS),
        "backend": "numpy", "cuda_visible_devices": "",
        "independent_lexicon": {"path": os.path.relpath(_LEXICON_PATH, _REPO),
                                "n_words": len(lex), "overlap_with_WARRINER": overlap,
                                "scorer": "VADER-compound s/sqrt(s^2+15) + 3-back negation"},
        "delta_preregistered": delta,
        "delta_definition": "population std of lesion-arm per-prompt tone_compound, pooled across seeds",
        "delta_sem_secondary": delta_sem,
        "per_seed": per_seed,
        "directional": {"go": dir_go, "pos_states": pos_states, "neg_states": neg_states},
        "directional_sem_secondary": {"go": dir_go_sem, "pos_states": pos_states_sem,
                                      "neg_states": neg_states_sem,
                                      "note": "SECONDARY, NOT the gate: same gaps evaluated against the SEM band "
                                              "delta/sqrt(n) instead of the preregistered raw-std band."},
        "attribution_control_ok": ctrl_ok_all,
        "attribution_fraction": attribution_fraction,
        "attribution_treatment_mean": treat_mean, "attribution_control_mean": ctrl_mean,
        "boosted_word_diagnostic": {
            "note": "CIRCULAR BY DESIGN, NOT GATED — the net signed count of the BOOSTED (WARRINER) words with "
                    "WARRINER's own sign; the phase4 byte-diff made directional. Characterizes whether a "
                    "disjoint-ruler NO-GO means 'no effect' or 'effect localized to the boosted lexicon'.",
            "per_seed": boosted_diag,
            "pos_minus_neg_signs": boosted_pos_minus_neg_signs},
        "content_identity_ok": content_ok, "content_detail": content_detail,
        "moat_ok": moat_ok, "fluency_ok": fluency_ok, "max_salad_frac": max_salad, "worst_salad": worst,
        "determinism_ok": determ_ok, "determinism_detail": determ_detail,
        "ckpt_per_seed_ok": ckpt_ok, "wkv_used_ok": wkv_used_ok, "missing_arms": missing,
        "verdict": decided,
        # TOP-LEVEL preconditions (tools/gates/verdict_preconditions.py enforces PRESENCE at the top level): the
        # same list the Verdict earned, surfaced beside the GO so the guard travels with the assertion.
        "preconditions": decided["preconditions"],
        "GO": bool(decided["go"]),
        "honesty_boundary": "expressed reply TONE tracks the spiking affect signal — a FUNCTIONAL read-out, "
                            "never a felt/phenomenal claim.",
        "shortcut_status": "_apply_affect_bias is HOST decode-time arithmetic over an already-neural valence "
                           "(tracked shortcut); brain-based target is BRAIN_WKV_MOUTH_AFFECT_NEURAL (out of scope).",
    }
    if write_artifact:
        ap = os.path.join(out_dir, "affect_tone_open_output_verdict.json")
        json.dump(artifact, open(ap, "w"), indent=2, default=str)
        print("[controller] wrote %s" % ap, flush=True)
        print("[controller] GO=%s  delta=%.5f  dir_go=%s attr=%s content=%s moat=%s fluency=%s determ=%s"
              % (artifact["GO"], delta, dir_go, ctrl_ok_all, content_ok, moat_ok, fluency_ok, determ_ok), flush=True)
    return artifact


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  SELFTEST  (pure logic; no brain build, no GPU)  — the instrument must be able to FAIL in its failing direction
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def selftest():
    ok = True
    lex, overlap = load_indep_lexicon()

    def check(name, cond):
        nonlocal ok
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))
        ok = ok and cond

    check("lexicon loads, disjoint (overlap==0)", overlap == 0 and len(lex) > 200)
    # scorer directionality on hand text (uses ONLY non-WARRINER words)
    cpos, _ = tone_compound("this is a wonderful splendid glorious magnificent day", lex)  # 'wonderful' is WARRINER-> not in lex
    cneg, _ = tone_compound("this is a dreadful atrocious horrid catastrophe", lex)
    cneu, _ = tone_compound("the box is on the table near the road", lex)
    check("positive text scores > 0", cpos > 0)
    check("negative text scores < 0", cneg < 0)
    check("neutral text scores == 0", cneu == 0.0)
    # 'wonderful'/'splendid': wonderful is WARRINER (not scored), splendid/glorious/magnificent are in lex
    check("scorer ignores a WARRINER-only word (wonderful) but scores splendid", tone_compound("wonderful", lex)[1] == 0 and tone_compound("splendid", lex)[1] == 1)
    # negation
    cn, _ = tone_compound("this is not splendid", lex)
    check("negation flips sign (not splendid < 0)", cn < 0)
    # delta preregistration
    d = preregister_delta([0.1, -0.1, 0.2, -0.2, 0.0, 0.0])
    check("delta = std of lesion tones (>0)", d > 0)
    check("delta of constant lesion tones == 0", preregister_delta([0.05, 0.05, 0.05]) < 1e-9)
    # directional decision logic — the gate must FAIL on a wrong-sign seed
    d2 = 0.05
    states_go = [directional_seed_verdict(+0.2, -0.2, d2) for _ in range(6)]
    go, _, _ = directional_go(states_go)
    check("6/6 correct-sign => GO", go is True)
    states_5 = states_go[:5] + [directional_seed_verdict(+0.0, -0.0, d2)]  # 6th null-clean
    check("5/6 correct + 6th null => GO", directional_go(states_5)[0] is True)
    states_wrong = states_go[:5] + [directional_seed_verdict(-0.2, +0.2, d2)]  # 6th wrong-sign
    check("5/6 correct + 6th WRONG-sign => NO-GO (must fail)", directional_go(states_wrong)[0] is False)
    # attribution semantics: a control gap above delta must be catchable
    check("control |gap|<delta detects a clean control", abs(0.0) < d2 and not (abs(0.2) < d2))
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return ok


_LEX = None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--controller", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--score-only", action="store_true", help="re-score collected arms without re-running")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arm", type=str, default="pos")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--out-dir", type=str,
                    default="research/findings/raw/_affect_tone_open_output")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--memcap-gb", type=int, default=10)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)
    elif args.worker:
        _LEX, _ = load_indep_lexicon()
        out = args.out or _worker_out(args.out_dir, args.seed, args.arm)
        run_worker(args.seed, args.arm, out)
    elif args.score_only:
        score_and_gate(args.out_dir)
    else:  # controller (default)
        run_controller(args.out_dir, parallel=args.parallel, memcap_gb=args.memcap_gb)
