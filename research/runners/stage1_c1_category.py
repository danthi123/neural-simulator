"""STAGE-1 C1 (category held-out inheritance) with the default-OFF category-inheritance wiring ON: the Stage-1 exit
gate's own C1 group (`cat_disc`, turns + scorer imported from research/runners/stage1_exit_gate.py, unchanged), run
through the live `webapp.server.brain_chat` handler with `BRAIN_CATEGORY_INHERIT=1`, so the same criterion can read
PASS once the wiring is on. Prereg (frozen bars, arms, predictions):
research/findings/2026-09-30-stage1-c1-category-inheritance-PREREGISTRATION.md. Execution packet:
research/packets/stage1-c1-category/HANDOFF.md.

WHAT IS MEASURED (the Stage-1 prereg section 4.1 criterion, reused byte-for-byte through imports):
  correct   = `stage1_exit_gate.score_cat(intact_a)["inherits"]`: either inheritance probe names `worm`
  lesion    = the responsible faculty cut: `BRAIN_CATEGORY_INHERIT_LESION=1` (the organ's learned pathways silenced
              at read time); effect = a primary reply field differs from intact_a on a C1 probe; reproduced = the
              lesion arm equals its own `.rep0` on every primary field of every turn
  permuted  = the canary streamed with the FISH context; tracks = `score_cat(permuted, permuted=True)["inherits"]`
              (names `bug`), defined only when intact_a is correct (the Stage-1 aggregate's own rule)
  null      = intact_a == intact_b on every primary field of every turn (else the seed is UNDEFINED)
The Stage-1 prereg's own C1 lesion (`lesion_draw`, the generative DRAW) is kept as an INFORMATIONAL arm here: the
wiring answers on the strict route, which never calls the DRAW, so that arm is predicted to leave the reply unchanged
(that prediction is what the Stage-1 re-attempt's amendment rests on: HANDOFF NEEDS-CLAUDE).

INFORMATIONAL arms (never counted; reported beside the verdict):
  off                  the production default (`BRAIN_CATEGORY_INHERIT=0`, forced explicitly): the Stage-1 FAIL shape;
                       also the SPECIFICITY check -- ON vs OFF may differ only on the C1 direct probe
  lesion_lateral_learn the frame->frame association never forms (plasticity blockade): is it NECESSARY?
  lesion_lateral_read  frame->frame silenced at read only: does the read need it, or was the link written at encoding?
  lesion_draw          the Stage-1 prereg's frozen C1 lesion (`BRAIN_SPIKING_DRAW_LESION=1`) with the wiring ON
FLIP-READINESS (reported beside the C1 verdict, never part of it): on every seed the intact arm keeps the identity
controls (robin -> worm, trout -> bug) and the moat (`what does the canary swim` abstains), and the ON-vs-OFF reply
differs on no turn except `p_canary_eat`.

HARNESS: `stage1_exit_gate.BASE_ENV` exactly (numpy/CPU, one BLAS thread, stub renderer, no LLM, tiny-demo + shipped
LTM tier, PYTHONHASHSEED=0, `rich` omitted), `BRAIN_CHAT_SEED=<seed>`, one fresh process + fresh ChatBrain per arm.

  .venv/bin/python -m research.runners.stage1_c1_category --selftest
  bash tools/memcap.sh 9 -- .venv/bin/python -m research.runners.stage1_c1_category --shard --seed 7 --arm intact_a \
      --out research/findings/raw/_stage1_c1_category/dev/s7/cat_disc__intact_a.json
  .venv/bin/python -m research.runners.stage1_c1_category --aggregate research/findings/raw/_stage1_c1_category/gate \
      --seeds 42,43,44,100,101,102
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import stage1_exit_gate as SG          # noqa: E402  (the Stage-1 C1 group + scorer, reused)
from tools.lab import attributable_to                        # noqa: E402  (gates/attribution_required)

GATE_SEEDS = SG.GATE_SEEDS
DEV_SEEDS = SG.DEV_SEEDS
GROUP = "cat_disc"
EXPECTED_SOURCE = SG.EXPECTED_SOURCE
GATE_OUT_DIR = "research/findings/raw/_stage1_c1_category/gate"
PREREG = "research/findings/2026-09-30-stage1-c1-category-inheritance-PREREGISTRATION.md"
FLAG = "BRAIN_CATEGORY_INHERIT"
LESION = "BRAIN_CATEGORY_INHERIT_LESION"
PROBES = ["p_canary_eat", "p_canary_might"]                  # the Stage-1 C1 scored labels (SG._scored_labels)
DIRECT_PROBE = "p_canary_eat"
POOL_MAX_AGE_H = 120                                         # the queue was ~440 deep (3-4 days) at packet time

# (env, turns key, kind). The flag is set EXPLICITLY in every arm, including the OFF arm ("0", never unset).
ARMS = {
    "intact_a": ({FLAG: "1"}, "turns", "intact"),
    "intact_b": ({FLAG: "1"}, "turns", "intact"),
    "lesion_inherit": ({FLAG: "1", LESION: "1"}, "turns", "lesion"),
    "lesion_inherit.rep0": ({FLAG: "1", LESION: "1"}, "turns", "lesion"),
    "permuted": ({FLAG: "1"}, "permuted_turns", "permuted"),
    "off": ({FLAG: "0"}, "turns", "informational"),
    "lesion_lateral_learn": ({FLAG: "1", LESION: "lateral_learn"}, "turns", "informational"),
    "lesion_lateral_read": ({FLAG: "1", LESION: "lateral"}, "turns", "informational"),
    "lesion_draw": ({FLAG: "1", "BRAIN_SPIKING_DRAW_LESION": "1"}, "turns", "informational"),
}
COUNTED_ARMS = [a for a, (_e, _t, k) in ARMS.items() if k != "informational"]
INFO_ARMS = [a for a, (_e, _t, k) in ARMS.items() if k == "informational"]


def turns_for(arm):
    return list(SG.GROUPS[GROUP][ARMS[arm][1]])


def labels():
    return [t[0] for t in SG.GROUPS[GROUP]["turns"]]


# ══════════════════════════════════════════════ WORKER (inside the arm subprocess) ═════════════════════════════════
def run_worker(seed, arm, out_path):
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    env, _tk, kind = ARMS[arm]
    for k, v in env.items():
        os.environ[k] = v
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    hash_ok = os.environ.get("PYTHONHASHSEED") == "0"
    corpus_present = os.path.exists(os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    t0 = time.time()
    from webapp.server import brain_chat, BrainChatRequest
    import_s = time.time() - t0
    turns = turns_for(arm)
    session = "s1c1_%s_%d" % (arm.replace(".", "_"), int(seed))
    responses, timings = {}, {}
    for i, (label, msg, _role) in enumerate(turns):
        t1 = time.time()
        try:
            r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo", renderer="stub",
                                            reset=(i == 0)))      # `rich` OMITTED = the production default path
            responses[label] = json.loads(r.body)
        except Exception as e:
            responses[label] = {"_error": "%s: %s" % (type(e).__name__, e)}
        timings[label] = round(time.time() - t1, 1)
    out = {"runner": "research/runners/stage1_c1_category.py", "prereg": PREREG, "stage1_group": GROUP,
           "seed": int(seed), "arm": arm, "kind": kind, "env": dict(env), "pythonhashseed_ok": hash_ok,
           "corpus_present": corpus_present, "session": session,
           "turns": [{"label": l, "message": m, "role": r} for l, m, r in turns],
           "responses": responses, "timings_s": timings, "import_s": round(import_s, 1),
           "wall_s": round(time.time() - t0, 1), "elapsed_seconds": round(time.time() - t0, 1),
           "expected_source": EXPECTED_SOURCE,
           "sources": sorted({str(r.get("source")) for r in responses.values() if isinstance(r, dict)})}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("[stage1-c1 worker] seed=%d %s -> %d turns, %.0fs -> %s" % (seed, arm, len(responses), out["wall_s"],
                                                                     out_path), flush=True)
    return 0


def run_shard(seed, arm, out_path, backend="numpy"):
    """One (seed, arm) in a FRESH subprocess with the Stage-1 harness env (PYTHONHASHSEED must precede the
    interpreter)."""
    env = dict(os.environ)
    env.update(SG.BASE_ENV)
    env["SIM_BACKEND"] = backend
    for k in (FLAG, LESION, "BRAIN_SPIKING_DRAW_LESION"):
        env.pop(k, None)                                       # the arm's worker sets its own (explicit) values
    part = out_path + ".part.json"
    cmd = [sys.executable, "-u", "-m", "research.runners.stage1_c1_category", "--worker", "--seed", str(int(seed)),
           "--arm", arm, "--out", part]
    p = subprocess.run(cmd, env=env, cwd=_REPO)
    if p.returncode != 0 or not os.path.exists(part):
        print("[stage1-c1 shard] worker FAILED rc=%s for seed=%s %s" % (p.returncode, seed, arm), flush=True)
        return 1
    with open(part) as f:
        d = json.load(f)
    d["backend"] = backend
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(d, f, indent=1, default=str)
    for suffix in ("", ".prov.json"):
        try:
            os.remove(part + suffix)
        except OSError:
            pass
    return 0


# ══════════════════════════════════════════════ SCORING (pure functions over shard dicts) ══════════════════════════
def shard_path(out_dir, seed, arm):
    return os.path.join(out_dir, "s%d" % int(seed), "%s__%s.json" % (GROUP, arm))


def _trace(resp):
    return (resp or {}).get("category_inherit") if isinstance(resp, dict) else None


def wiring_applied(shard, arm):
    """The flag reached the handler: every flag-ON arm carries the `category_inherit` trace on every turn, and the
    direct probe's organ read reports the arm's lesion mode; the OFF arm carries it on no turn."""
    resp = shard.get("responses") or {}
    on = ARMS[arm][0].get(FLAG) == "1"
    has = [_trace(r) is not None for r in resp.values()]
    if not on:
        return not any(has)
    if not (has and all(has)):
        return False
    want = {None: None, "1": "all", "lateral": "lateral", "lateral_learn": "lateral_learn"}[ARMS[arm][0].get(LESION)]
    rd = (_trace(resp.get(DIRECT_PROBE)) or {}).get("read") or {}
    if want in ("all", "lateral"):                  # the read-time cut was in force AND still held (docs/TERMS.md)
        return rd.get("lesion") == want and rd.get("lesion_held") is True
    return True


def safety(resp):
    s = SG.score_cat(resp)
    return {"identity_robin": s["identity_robin"], "identity_trout": s["identity_trout"],
            "moat_canary_swim": s["moat_canary_swim"],
            "all": bool(s["identity_robin"] and s["identity_trout"] and s["moat_canary_swim"])}


def _load(root, seeds):
    out = {}
    for seed in seeds:
        for arm in ARMS:
            p = shard_path(root, seed, arm)
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        out[(int(seed), arm)] = json.load(f)
                except Exception as e:
                    out[(int(seed), arm)] = {"_load_error": repr(e)}
    return out


def aggregate(root, seeds=GATE_SEEDS):
    shards = _load(root, seeds)
    seeds = [int(s) for s in seeds]
    labs = labels()
    rows, defined, missing = {}, [], []
    for seed in seeds:
        have = {a: shards.get((seed, a)) for a in ARMS}
        for a in COUNTED_ARMS:
            if have[a] is None:
                missing.append(shard_path(root, seed, a))
        row = {"seed": seed, "undefined_reasons": []}
        readable = {a: (have[a] is not None and "responses" in have[a]) for a in ARMS}
        for a in COUNTED_ARMS:
            if not readable[a]:
                row["undefined_reasons"].append("%s arm missing or unreadable" % a)
        if readable["intact_a"] and readable["intact_b"]:
            ia, ib = have["intact_a"], have["intact_b"]
            ok, diffs = SG.null_clean(ia["responses"], ib["responses"], labs)
            row["null_clean"], row["null_diffs"] = ok, diffs[:8]
            if not ok:
                row["undefined_reasons"].append("null failed (intact_a != intact_b) on %d turn(s)" % len(diffs))
        for a in COUNTED_ARMS:
            if not readable[a]:
                continue
            sh = have[a]
            if set(sh.get("sources") or []) - {EXPECTED_SOURCE}:
                row["undefined_reasons"].append("%s: unexpected brain source %s" % (a, sh.get("sources")))
            if not sh.get("pythonhashseed_ok"):
                row["undefined_reasons"].append("%s: PYTHONHASHSEED not 0" % a)
            if not sh.get("corpus_present"):
                row["undefined_reasons"].append("%s: corpus absent (corpus-learned organs degraded)" % a)
            if any("_error" in (v or {}) for v in (sh.get("responses") or {}).values()):
                row["undefined_reasons"].append("%s: an errored turn" % a)
            if not wiring_applied(sh, a):
                row["undefined_reasons"].append("%s: wiring/lesion not applied (category_inherit trace)" % a)
        if readable["intact_a"]:
            ra = have["intact_a"]["responses"]
            row["intact"] = SG.score_cat(ra)
            row["correct"] = bool(row["intact"]["inherits"])
            row["safety"] = safety(ra)
            rd = (_trace(ra.get(DIRECT_PROBE)) or {}).get("read") or {}
            row["organ_read_intact"] = {k: rd.get(k) for k in ("patient", "reason", "admitted")}
        # lesion: the reply-level effect must be present AND reproduced
        L, R = have.get("lesion_inherit"), have.get("lesion_inherit.rep0")
        rec = {"present": L is not None, "repeat_present": R is not None}
        if readable["intact_a"] and readable["intact_b"] and readable["lesion_inherit"]:
            ra, rb, rl = have["intact_a"]["responses"], have["intact_b"]["responses"], L["responses"]
            d = [SG.reply_differs(ra, rl, l) for l in PROBES]
            rec["differs"] = dict(zip(PROBES, d))
            rec["effect"] = any(x is True for x in d)
            rec["attributable"] = attributable_to(
                "C1 lesion_inherit seed %d (primary reply fields on %s)" % (seed, ",".join(PROBES)),
                float(SG.n_field_diffs(ra, rl, PROBES)), float(SG.n_field_diffs(ra, rb, PROBES)))
            rec["reproduced"] = (SG.null_clean(rl, R["responses"], labs)[0] if readable["lesion_inherit.rep0"]
                                 else None)
            rdl = (_trace(rl.get(DIRECT_PROBE)) or {}).get("read") or {}
            rec["organ_read"] = {k: rdl.get(k) for k in ("patient", "reason", "admitted", "lesion")}
        else:
            rec["effect"] = None
            rec["reproduced"] = None
        row["lesion_inherit"] = rec
        # permuted exposure: the reply must follow the exposure (defined only when intact is correct)
        if readable["permuted"]:
            sp = SG.score_cat(have["permuted"]["responses"], permuted=True)
            row["permuted"] = sp
            row["tracks"] = sp["inherits"] if row.get("correct") else None
        else:
            row["tracks"] = None
        # informational arms + flip-readiness
        info = {}
        if readable["intact_a"]:
            ra = have["intact_a"]["responses"]
            for a in INFO_ARMS:
                if not readable[a]:
                    info[a] = None
                    continue
                ri = have[a]["responses"]
                sc = SG.score_cat(ri)
                info[a] = {"inherits": sc["inherits"], "direct_patient": sc["direct_patient"],
                           "generative_patient": sc["generative_patient"],
                           "differs_from_intact_on": [l for l in labs if SG.reply_differs(ra, ri, l)],
                           "wiring_applied": wiring_applied(have[a], a)}
        row["informational"] = info
        off = info.get("off")
        row["specificity"] = (None if off is None else set(off["differs_from_intact_on"]) <= {DIRECT_PROBE})
        rows[seed] = row
        if not row["undefined_reasons"] and "correct" in row:
            defined.append(seed)
    n_def = len(defined)
    n_correct = sum(1 for s in defined if rows[s]["correct"])
    n_eff = sum(1 for s in defined if rows[s]["lesion_inherit"].get("effect") is True)
    n_rep = sum(1 for s in defined if rows[s]["lesion_inherit"].get("reproduced") is True)
    n_tr_def = sum(1 for s in defined if rows[s].get("tracks") is not None)
    n_tracks = sum(1 for s in defined if rows[s].get("tracks") is True)
    if n_def < len(seeds):
        verdict, why = "INCOMPLETE", ["%d of %d seeds defined" % (n_def, len(seeds))]
    elif n_correct == n_def and n_eff == n_def and n_rep == n_def and n_tr_def == n_def and n_tracks == n_def:
        verdict, why = "PASS", []
    else:
        verdict, why = "FAIL", []
        if n_correct < n_def:
            why.append("capability absent on %d of %d seeds" % (n_def - n_correct, n_def))
        else:
            if n_eff < n_def:
                why.append("lesion_inherit: reply-level effect on %d of %d seeds (not load-bearing)" % (n_eff, n_def))
            if n_rep < n_def:
                why.append("lesion_inherit: reproduced on %d of %d seeds" % (n_rep, n_def))
            if n_tracks < n_tr_def:
                why.append("permuted exposure tracked on %d of %d seeds (not exposure-driven)" % (n_tracks, n_tr_def))
            elif n_tr_def < n_def:
                why.append("permuted tracking undefined on %d of %d seeds" % (n_def - n_tr_def, n_def))
    n_safe = sum(1 for s in defined if (rows[s].get("safety") or {}).get("all"))
    n_spec_def = sum(1 for s in defined if rows[s].get("specificity") is not None)
    n_spec = sum(1 for s in defined if rows[s].get("specificity") is True)
    flip_ready = (verdict == "PASS" and n_safe == n_def and n_spec_def == n_def and n_spec == n_def)
    info_counts = {a: {"inherits": sum(1 for s in defined if ((rows[s]["informational"].get(a) or {}).get("inherits"))),
                       "differs_on_a_probe": sum(1 for s in defined if set(
                           ((rows[s]["informational"].get(a) or {}).get("differs_from_intact_on") or [])) & set(PROBES)),
                       "present": sum(1 for s in defined if rows[s]["informational"].get(a) is not None)}
                   for a in INFO_ARMS}
    from tools.verdict import Verdict
    v = Verdict("Stage-1 C1 (category held-out inheritance) with the category-inheritance wiring ON")
    v.require("no counted shard missing", len(missing), expect=0)
    v.require("null clean (intact_a == intact_b) on every seed",
              sum(1 for s in seeds if rows[s].get("null_clean") is not True), expect=0)
    v.require("every counted arm from the expected brain with the corpus and PYTHONHASHSEED=0",
              sum(1 for s in seeds for r in rows[s]["undefined_reasons"]
                  if ("source" in r or "PYTHONHASHSEED" in r or "corpus" in r)), expect=0)
    v.require("the wiring and the lesion applied in every counted arm (category_inherit trace)",
              sum(1 for s in seeds for r in rows[s]["undefined_reasons"] if "not applied" in r), expect=0)
    v.require("six seeds defined", len(seeds) - n_def, expect=0)
    earned = v.decide(go=(verdict == "PASS"), verbose=False)
    if earned["status"] == "UNDEFINED":
        final = "UNDEFINED (INCOMPLETE): " + "; ".join(earned["undefined_reasons"])[:400]
    else:
        final = verdict
    return {"criterion": "C1 (Stage-1 cat_disc) with BRAIN_CATEGORY_INHERIT=1", "verdict": final,
            "group_verdict": verdict, "why": why, "GO": bool(earned["go"]),
            "preconditions": earned["preconditions"], "undefined_reasons": earned["undefined_reasons"],
            "n_defined": n_def, "n_correct": n_correct, "n_lesion_effect": n_eff, "n_reproduced": n_rep,
            "n_permuted_tracks": n_tracks, "n_permuted_defined": n_tr_def,
            "flip_readiness": {"ready": flip_ready, "n_safety": n_safe, "n_specificity": n_spec,
                               "n_specificity_defined": n_spec_def},
            "informational": info_counts, "seeds": seeds, "rows": rows, "missing_shards": missing,
            "runner": "research/runners/stage1_c1_category.py", "prereg": PREREG, "expected_source": EXPECTED_SOURCE}


def report(agg):
    n = agg["n_defined"]
    fr = agg["flip_readiness"]
    lines = ["**C1 with the category-inheritance wiring ON: %s**" % agg["verdict"], "",
             "| verdict | correct | lesion effect | reproduced | permuted tracks | why |", "|---|---|---|---|---|---|",
             "| %s | %d/%d | %d/%d | %d/%d | %d/%d | %s |" % (
                 agg["group_verdict"], agg["n_correct"], n, agg["n_lesion_effect"], n, agg["n_reproduced"], n,
                 agg["n_permuted_tracks"], agg["n_permuted_defined"], "; ".join(agg["why"]) or "-"),
             "", "Flip-readiness (not part of the C1 verdict): %s (safety %d/%d, specificity %d/%d)" % (
                 "READY" if fr["ready"] else "NOT READY", fr["n_safety"], n, fr["n_specificity"],
                 fr["n_specificity_defined"]),
             "", "Informational arms (defined seeds): " + "; ".join(
                 "%s inherits %d/%d, changes a C1 probe %d/%d" % (a, c["inherits"], c["present"],
                                                                 c["differs_on_a_probe"], c["present"])
                 for a, c in agg["informational"].items()), "", "Per seed:"]
    for s, row in agg["rows"].items():
        it = row.get("intact") or {}
        lines.append("- seed %s: direct=%s generative=%s; lesion direct=%s; permuted direct=%s; organ %s; %s" % (
            s, it.get("direct_patient"), it.get("generative_patient"),
            ((row.get("lesion_inherit") or {}).get("organ_read") or {}).get("patient"),
            (row.get("permuted") or {}).get("direct_patient"),
            (row.get("organ_read_intact") or {}).get("admitted"),
            "; ".join(row.get("undefined_reasons") or []) or "defined"))
    if agg["missing_shards"]:
        lines += ["", "Missing shards: %d" % len(agg["missing_shards"])]
    return "\n".join(lines)


# ══════════════════════════════════════════════ POOL JOBS ═══════════════════════════════════════════════════════════
def pool_job_lines(sha, seeds=GATE_SEEDS, out_dir=GATE_OUT_DIR, max_age_h=POOL_MAX_AGE_H):
    """One pool job per (seed, arm). A tiny-demo+LTM build is ~6.3-7.7 GB RSS -> mem_gb=7, cap 9 GB (Stage-1 figure)."""
    out = []
    for seed in seeds:
        for arm in ARMS:
            p = shard_path(out_dir, seed, arm)
            cmd = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh 7 1 && mkdir -p %s && "
                   "bash tools/memcap.sh 9 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                   "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m "
                   "research.runners.stage1_c1_category --shard --seed %d --arm %s --allow-gate-seed "
                   "--prereg %s --out %s" % (sha, os.path.dirname(p), seed, arm, PREREG, p))
            why = ("prereg %s (%s); Stage-1 C1 with the category-inheritance wiring ON, seed %d arm %s; record: "
                   "Stage-1 gate dev seed 7 read both C1 routes empty, rung-3c learned member->context pathway "
                   "chosen (prereg section 2); mem_gb=7; max_age_h=%d" % (PREREG, sha[:9], seed, arm, int(max_age_h)))
            out.append((cmd, why))
    return out


# ══════════════════════════════════════════════ SELFTEST (no brain) ════════════════════════════════════════════════
def _fake(answer, abstained=False, svo=None, hyp=None, trace=True, lesion=None, patient=None):
    d = {"answer": answer, "abstained": abstained, "recalled_svo": svo, "source": EXPECTED_SOURCE}
    if hyp:
        d["hypothesis"] = True
        d["hypothesis_svo"] = hyp
    if trace:
        d["category_inherit"] = {"observe": None, "read": {"patient": patient, "lesion": lesion, "reason": "x",
                                                            "admitted": [], "lesion_held": (
                                                                True if lesion in ("all", "lateral") else None)}}
    return d


def _synthetic(kind, arm):
    """Synthetic responses for one arm. kind: pass | absent | not_lb | no_track."""
    env = ARMS[arm][0]
    on = env.get(FLAG) == "1"
    lesion = {None: None, "1": "all", "lateral": "lateral", "lateral_learn": "lateral_learn"}[env.get(LESION)]
    permuted = ARMS[arm][1] == "permuted_turns"
    r = {}
    for l, _m, _role in turns_for(arm):
        r[l] = _fake("ok " + l, svo=["x", "y", l], trace=on, lesion=lesion)
    r["c_robin_eat"] = _fake("the robin eats the worm", svo=["robin", "eat", "worm"], trace=on, lesion=lesion)
    r["c_trout_eat"] = _fake("the trout eats the bug", svo=["trout", "eat", "bug"], trace=on, lesion=lesion)
    r["c_canary_swim"] = _fake("I don't know", abstained=True, trace=on, lesion=lesion)
    r["p_canary_might"] = _fake("I don't know", abstained=True, trace=on, lesion=lesion)
    want = "bug" if permuted else "worm"
    reaches = on and lesion in (None, "lateral") and kind != "absent"
    if kind == "not_lb" and lesion == "all":
        reaches = True
    if kind == "no_track" and permuted:
        want = "worm"
    if reaches:
        r["p_canary_eat"] = _fake("Maybe the canary eats the %s" % want, hyp=["canary", "eat", want], trace=on,
                                  lesion=lesion, patient=want)
    else:
        r["p_canary_eat"] = _fake("I don't know", abstained=True, trace=on, lesion=lesion)
    return {"responses": r, "sources": [EXPECTED_SOURCE], "pythonhashseed_ok": True, "corpus_present": True}


def selftest(verbose=True):
    import tempfile
    checks = {}
    for kind, expect in (("pass", "PASS"), ("absent", "FAIL"), ("not_lb", "FAIL"), ("no_track", "FAIL")):
        with tempfile.TemporaryDirectory() as td:
            for s in GATE_SEEDS:
                for arm in ARMS:
                    p = shard_path(td, s, arm)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    with open(p, "w") as f:
                        json.dump(_synthetic(kind, arm), f)
            with contextlib.redirect_stdout(sys.stderr):
                agg = aggregate(td)
            checks["%s -> %s" % (kind, expect)] = agg["verdict"] == expect
            if kind == "pass":
                checks["pass is flip-ready"] = agg["flip_readiness"]["ready"] is True
                checks["lesion_draw informational leaves the reply"] = agg["informational"]["lesion_draw"][
                    "differs_on_a_probe"] == 0
            if kind == "absent":
                checks["absent names the clause"] = any("capability absent" in w for w in agg["why"])
            if kind == "not_lb":
                checks["not_lb names the clause"] = any("not load-bearing" in w for w in agg["why"])
            if kind == "no_track":
                checks["no_track names the clause"] = any("not exposure-driven" in w for w in agg["why"])
    with tempfile.TemporaryDirectory() as td:          # a missing counted shard -> UNDEFINED, never FAIL
        for s in GATE_SEEDS:
            for arm in ARMS:
                if s == 44 and arm == "lesion_inherit.rep0":
                    continue
                p = shard_path(td, s, arm)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as f:
                    json.dump(_synthetic("pass", arm), f)
        with contextlib.redirect_stdout(sys.stderr):
            agg = aggregate(td)
        checks["missing shard -> UNDEFINED"] = agg["verdict"].startswith("UNDEFINED")
    with tempfile.TemporaryDirectory() as td:          # the flag never reached the handler -> UNDEFINED
        for s in GATE_SEEDS:
            for arm in ARMS:
                d = _synthetic("pass", arm)
                if arm == "intact_a" and s == 42:
                    for v in d["responses"].values():
                        v.pop("category_inherit", None)
                p = shard_path(td, s, arm)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as f:
                    json.dump(d, f)
        with contextlib.redirect_stdout(sys.stderr):
            agg = aggregate(td)
        checks["wiring not applied -> UNDEFINED"] = agg["verdict"].startswith("UNDEFINED")
    lines = pool_job_lines("f" * 40)
    checks["pool lines: one per seed x arm, pinned, prereg-tagged, max_age_h"] = (
        len(lines) == len(GATE_SEEDS) * len(ARMS)
        and all(("revisions/%s" % ("f" * 40)) in c and "--prereg %s" % PREREG in c and "memcap.sh 9" in c
                and "max_age_h=%d" % POOL_MAX_AGE_H in w and "mem_gb=7" in w for c, w in lines))
    checks["the OFF arm forces the flag to 0 explicitly"] = ARMS["off"][0].get(FLAG) == "0"
    checks["C1 turns are the Stage-1 group's own"] = turns_for("intact_a") == list(SG.GROUPS[GROUP]["turns"])
    ok = all(checks.values())
    if verbose:
        for k, v in checks.items():
            print("  [%s] %s" % ("ok" if v else "FAIL", k))
        print("SELFTEST OK" if ok else "SELFTEST FAILED")
    return ok, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker", action="store_true", help="internal: run one arm in THIS process")
    ap.add_argument("--shard", action="store_true", help="run one (seed, arm) in a fresh subprocess")
    ap.add_argument("--aggregate", default=None, help="shard directory to aggregate")
    ap.add_argument("--report", default=None, metavar="AGGREGATE_JSON", help="print the fixed Markdown result block")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-pool-jobs", default=None, metavar="SHA", help="print the gate's pool job lines (TSV)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds for --aggregate / --emit-pool-jobs")
    ap.add_argument("--arm", default=None, choices=sorted(ARMS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", default="numpy", choices=("numpy", "cupy"))
    ap.add_argument("--max-age-h", type=int, default=POOL_MAX_AGE_H, help="pool line max_age_h (cap 240)")
    ap.add_argument("--prereg", default=None, help="the committed preregistration a GATE-seed run is governed by")
    ap.add_argument("--allow-gate-seed", action="store_true",
                    help="required to run a GATE seed (42/43/44/100/101/102); dev seeds need nothing")
    a = ap.parse_args()
    if a.selftest:
        ok, _ = selftest()
        sys.exit(0 if ok else 1)
    if a.emit_pool_jobs:
        seeds = tuple(int(s) for s in a.seeds.split(",")) if a.seeds else GATE_SEEDS
        for cmd, why in pool_job_lines(a.emit_pool_jobs, seeds=seeds, max_age_h=min(240, a.max_age_h)):
            print("%s\t%s" % (cmd, why))
        return
    if a.aggregate:
        seeds = tuple(int(s) for s in a.seeds.split(",")) if a.seeds else GATE_SEEDS
        with contextlib.redirect_stdout(sys.stderr):
            agg = aggregate(a.aggregate, seeds=seeds)
        out = a.out or os.path.join(a.aggregate, "aggregate.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(agg, f, indent=1, default=str)
        print(report(agg))
        print("\n-> %s" % out)
        return
    if a.report:
        with open(a.report) as f:
            print(report(json.load(f)))
        return
    if a.worker or a.shard:
        if a.seed is None or not a.arm or not a.out:
            ap.error("--seed, --arm and --out are required")
        if a.seed in GATE_SEEDS and not a.allow_gate_seed:
            sys.stderr.write("GATE seed %d refused: a gate seed runs only from a pool job emitted against the committed "
                             "prereg (%s); pass --allow-gate-seed --prereg <path> from such a job. Dev seeds: 1-12.\n"
                             % (a.seed, PREREG))
            sys.exit(2)
        if a.worker:
            sys.exit(run_worker(a.seed, a.arm, a.out))
        sys.exit(run_shard(a.seed, a.arm, a.out, backend=a.backend))
    ap.error("choose one of --worker/--shard/--aggregate/--report/--selftest/--emit-pool-jobs")


if __name__ == "__main__":
    main()
