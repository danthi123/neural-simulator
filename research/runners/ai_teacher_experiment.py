"""AI-TEACHER LEARNING EXPERIMENT — does the brain learn, through CHAT ONLY, what a teacher tells it, and is the later
recall (teacher absent) carried by the brain's own synaptic write?

Pre-registration: research/findings/2026-09-24-ai-teacher-environment-PREREGISTRATION.md (committed on its own, before
any evaluation run). The teacher is research/runners/ai_teacher.py; its only channel is `webapp.server.brain_chat`.

PER ARM (a FRESH brain in its own subprocess, BRAIN_CHAT_SEED=seed, numpy/CPU, stub renderer, no LLM, LTM OFF, the D6
capability configuration BRAIN_D6_HEBBIAN_STORE=1 + BRAIN_D6_ENGRAM_VOCAB=1 + BRAIN_D6_ENGRAM_READTIME=1):
  warmup  experimenter: "what does the cat eat" (builds the brain; a build-time fact; read path baseline)
  lesson  TEACHER: for each of K facts -- ask the question; answer the brain's curiosity ask, or tell the fact
  quiz    TEACHER: ask each question again; restate the fact when the reply text is wrong
  sleep   ENVIRONMENT: the session's clock is advanced to sleep depth (>= SLEEP_IDLE_SEC) and the server's own idle
          tick runs `--sleep-ticks` times (whatever offline processes the brain has run there; recorded)
  lesion  experimenter (ZERO arm only): zero every taught block's synapses
  test    experimenter, TEACHER ABSENT: ask each of the K questions + 2 build-time control questions
ARMS
  TEACH     the teacher teaches the vetted facts                         (learning)
  NOTEACH   no lesson, no quiz (teacher lesion)                         (no teaching -> no recall)
  FREEZE    TEACH + BRAIN_D6_HEBBIAN_FREEZE=1 (eta=0 for in-conversation writes; same encode activity)
  ZERO      TEACH, then every taught block zeroed after sleep           (the recall must ride those synapses)
  PERM      the teacher believes a per-seed derangement of the objects  (recall follows what was TOLD)
  ERR       the teacher is wrong on half the facts (distractor objects) (teacher-error propagation vs ground truth)
  TEACH_REP identical to TEACH, rebuilt (determinism null; seed-7 dev smoke only)

Run (pool node; one worker per arm):
  SIM_BACKEND=numpy OMP_NUM_THREADS=1 .venv/bin/python -u -m research.runners.ai_teacher_experiment \
      --seeds 7 --K 4 --arms TEACH NOTEACH FREEZE ZERO PERM ERR TEACH_REP --k-sweep 2 8 --jobs 3 \
      --arm-dir research/findings/raw/_ai_teacher/seed7
Score only:   ... --score-only --seeds 7 --K 4 --arm-dir research/findings/raw/_ai_teacher/seed7 --json <out>
Self-test:    .venv/bin/python -m research.runners.ai_teacher_experiment --selftest
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

SESSION = "aiteach"
BRAIN = "tiny-demo"
RENDERER = "stub"
WARMUP = "what does the cat eat"
CONTROL_PROBES = [("cat", "eat", "fish"), ("dog", "chase", "cat")]
BASE_ENV = {"SIM_BACKEND": "numpy", "BRAIN_CHAT_RENDERER": "stub", "SIM_DISABLE_LLM": "1",
            "BRAIN_LTM_BUNDLE": "off", "BRAIN_AI_TEACHER": "1",
            "BRAIN_D6_HEBBIAN_STORE": "1", "BRAIN_D6_ENGRAM_VOCAB": "1", "BRAIN_D6_ENGRAM_READTIME": "1"}
ARMS = {
    "TEACH":     {"teacher": True,  "freeze": False, "belief": "vetted",    "post_sleep": None},
    "TEACH_REP": {"teacher": True,  "freeze": False, "belief": "vetted",    "post_sleep": None},
    "NOTEACH":   {"teacher": False, "freeze": False, "belief": "vetted",    "post_sleep": None},
    "FREEZE":    {"teacher": True,  "freeze": True,  "belief": "vetted",    "post_sleep": None},
    "ZERO":      {"teacher": True,  "freeze": False, "belief": "vetted",    "post_sleep": "ablate_taught"},
    "PERM":      {"teacher": True,  "freeze": False, "belief": "permuted",  "post_sleep": None},
    "ERR":       {"teacher": True,  "freeze": False, "belief": "corrupted", "post_sleep": None},
}
GATED_ARMS = ("TEACH", "NOTEACH", "FREEZE", "ZERO", "PERM", "ERR")
SEEDS6 = [42, 43, 44, 100, 101, 102]
# pre-registered thresholds (see the PREREGISTRATION; do not edit after results exist)
T1_MIN_RECALL = 0.75
T3_TEACH_LEVER_MIN = 0.5
T5_MIN_RECALL = 0.75
T6_CLEAN_MIN_RECALL = 0.5
T8_MIN_PATCHED = 10
T8_REQUIRED = ("OneBrainComposer._write_block", "ChatBrain._maybe_acquire", "d6_hebbian_store.hebbian_encode")


def corrupted_positions(K):
    return [i for i in range(K) if i % 2 == 1]


def arm_file(arm_dir, seed, arm, K):
    return os.path.join(arm_dir, "s%d_%s_K%d.json" % (seed, arm, K))


# ── worker ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def _worker(arm, K, seed, out_path, sleep_ticks):
    cfg = ARMS[arm]
    for k, v in BASE_ENV.items():
        os.environ[k] = v
    os.environ["BRAIN_D6_HEBBIAN_FREEZE"] = "1" if cfg["freeze"] else "0"
    os.environ["BRAIN_CHAT_SEED"] = str(seed)
    t_start = time.time()
    from tools.lab import assert_backend
    assert_backend("numpy", note="ai_teacher arm %s: the registered protocol is the numpy/CPU tiny-demo" % arm)
    # the isolation guard goes in BEFORE webapp.server is imported, so a by-name import of a writer gets the wrapper
    from research.runners.ai_teacher_guard import TeacherIsolationGuard
    guard = TeacherIsolationGuard(mode="attribute").install()
    from research.runners import ai_teacher as AT
    from webapp import server as S
    from webapp.server import brain_chat, BrainChatRequest
    from webapp import continuous_engine as CE
    from research.runners.d6_hebbian_store import ablate_block

    cur = AT.load_curriculum()
    vetted = AT.curriculum_facts(cur)[:K]
    if cfg["belief"] == "permuted":
        know = AT.TeacherKnowledge(vetted).permuted(seed)
    elif cfg["belief"] == "corrupted":
        know = AT.TeacherKnowledge(vetted).corrupted(corrupted_positions(K), cur["distractors"])
    else:
        know = AT.TeacherKnowledge(vetted)
    key = (SESSION, BRAIN, RENDERER)
    st = {"first": True, "phase": "warmup", "turns": [], "writes": {}, "homeo": {}}
    out = {"arm": arm, "K": K, "seed": seed, "cfg": cfg, "backend": os.environ.get("SIM_BACKEND"),
           "env": {k: os.environ.get(k) for k in
           list(BASE_ENV) + ["BRAIN_D6_HEBBIAN_FREEZE", "BRAIN_CHAT_SEED"]},
           "vetted": [f.__dict__ for f in vetted], "belief": [f.__dict__ for f in know.facts],
           "belief_label": know.label, "counter_installed": False, "errors": []}

    def brain_channel(text):
        """THE chat boundary: one /api/brain-chat turn. The experimenter keeps the full reply; the teacher gets TEXT."""
        t0 = time.time()
        r = brain_chat(BrainChatRequest(session=SESSION, message=text, brain=BRAIN, renderer=RENDERER, rich=False,
                                        reset=st["first"]))
        st["first"] = False
        d = json.loads(r.body)
        st["turns"].append({"phase": st["phase"], "message": text, "answer": d.get("answer"),
                            "abstained": d.get("abstained"), "recalled_svo": d.get("recalled_svo"),
                            "curiosity_topic": (d.get("curiosity") or {}).get("topic") if isinstance(
                                d.get("curiosity"), dict) else None,
                            "t_s": round(time.time() - t0, 2)})
        print("[ai_teacher turn] %s s%d %s #%d %.0fs | %s -> %s" % (arm, seed, st["phase"], len(st["turns"]),
                                                                 time.time() - t0, text, (d.get("answer") or "")[:90]),
              flush=True)
        return d.get("answer") or ""

    def teacher_channel(text):          # text in, text out -- nothing else crosses
        return str(brain_channel(text))

    def composer():
        chat = S._BRAIN_CHATS.get(key)
        return getattr(getattr(chat, "inner", None), "composer", None)

    def taught_blocks(comp, subjects):
        rows = []
        D = comp.D
        for j, (f, _h) in enumerate(getattr(comp, "kb", []) or []):
            if str(f.get("agent", "")).lower() in subjects:
                ws = [complex(w) for (_p, _q, w) in comp.store_conns[j * D:(j + 1) * D]]
                rows.append({"block": j, "agent": f.get("agent"), "action": f.get("action"),
                             "patient": f.get("patient"), "mean_abs_w": round(sum(abs(w) for w in ws) / max(len(ws), 1),
                                                                                6)})
        return rows

    try:
        brain_channel(WARMUP)                                   # builds the brain (reset=True)
        comp = composer()
        out["n_build_kb"] = len(getattr(comp, "kb", []) or []) if comp is not None else None
        out["composer_class"] = type(comp).__name__ if comp is not None else None
        if comp is not None and hasattr(comp, "_write_block"):
            _orig = comp._write_block

            def _counted(bi, zc, _o=_orig):
                st["writes"][st["phase"]] = st["writes"].get(st["phase"], 0) + 1
                return _o(bi, zc)
            comp._write_block = _counted
            if hasattr(comp, "apply_homeostatic_scaling"):
                _oh = comp.apply_homeostatic_scaling

                def _hcounted(*a, _o=_oh, **k):
                    st["homeo"][st["phase"]] = st["homeo"].get(st["phase"], 0) + 1
                    return _o(*a, **k)
                comp.apply_homeostatic_scaling = _hcounted
            out["counter_installed"] = True
        subjects = {f.subject for f in vetted}
        if cfg["teacher"]:
            teacher = AT.AITeacher(teacher_channel, know)
            st["phase"] = "lesson"
            lesson = teacher.lesson(vetted)
            st["phase"] = "quiz"
            quiz = teacher.quiz(vetted)
            out["teacher"] = {"lesson": lesson, "quiz": quiz, "told": dict(teacher.told),
                              "utterances": [u.__dict__ for u in teacher.log]}
        else:
            out["teacher"] = None
        out["taught_blocks_after_teaching"] = taught_blocks(comp, subjects) if comp is not None else None
        # sleep: the environment's clock passes; the server's own idle tick runs at sleep depth
        st["phase"] = "sleep"
        sleep = {"ticks": [], "in_mood": key in S._SESSION_MOOD, "inner_life_before": len(CE.inner_life(key))}
        for t in range(int(sleep_ticks)):
            last = CE._LAST_REQUEST.get(key, time.time())
            now = last + CE.SLEEP_IDLE_SEC + 1.0 + t
            t0 = time.time()
            n = CE.tick_idle_sessions(S._SESSION_MOOD, S._get_affect_organ, now=now,
                                      selfinit_getter=S._get_selfinit_organ,
                                      episodic_getter=S._get_episodic_organ_existing,
                                      chat_getter=S._get_chat_existing)
            sleep["ticks"].append({"n_sessions": n, "t_s": round(time.time() - t0, 2)})
        sleep["inner_life_new"] = [
            {k: v for k, v in rec.items() if k in ("note", "substrate_homeostasis", "trigger", "wandered")}
            for rec in CE.inner_life(key)[sleep["inner_life_before"]:]]
        out["sleep"] = sleep
        st["phase"] = "lesion"
        if cfg["post_sleep"] == "ablate_taught":
            abl = []
            for row in taught_blocks(comp, subjects):
                abl.append(ablate_block(comp, row["block"]))
            out["ablation"] = abl
        out["taught_blocks_at_test"] = taught_blocks(comp, subjects) if comp is not None else None
        st["phase"] = "test"                                    # TEACHER ABSENT from here on
        for f in vetted:
            brain_channel(AT.render_ask(f))
        for (s, v, _o) in CONTROL_PROBES:
            brain_channel("what does the %s %s" % (s, v))
        out["kb_final"] = [{"i": j, "agent": f.get("agent"), "action": f.get("action"), "patient": f.get("patient")}
                           for j, (f, _h) in enumerate(getattr(comp, "kb", []) or [])]
    except Exception as e:
        import traceback
        out["errors"].append("%s: %s" % (type(e).__name__, e))
        out["traceback"] = traceback.format_exc()
    finally:
        guard.uninstall()
    out["turns"] = st["turns"]
    out["writes_by_phase"] = dict(st["writes"])
    out["homeostatic_calls_by_phase"] = dict(st["homeo"])
    out["guard"] = guard.report()
    out["elapsed_s"] = round(time.time() - t_start, 1)
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("[ai_teacher worker] %s K=%d seed=%d -> %s (%.0fs, errors=%d)" % (arm, K, seed, out_path, out["elapsed_s"],
                                                                         len(out["errors"])), flush=True)
    return 0 if not out["errors"] else 3


def _spawn(arm, K, seed, arm_dir, sleep_ticks):
    path = arm_file(arm_dir, seed, arm, K)
    env = dict(os.environ)
    env["BRAIN_CHAT_SEED"] = str(seed)
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners.ai_teacher_experiment", "--worker",
                        "--arm", arm, "--K", str(K), "--seed", str(seed), "--out", path,
                        "--sleep-ticks", str(sleep_ticks)], env=env)
    return arm, K, seed, p.returncode, os.path.exists(path)


# ── scoring ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def _test_turn(arm_json, subject, verb):
    msg = "what does the %s %s" % (subject, verb)
    rows = [t for t in (arm_json or {}).get("turns", []) if t.get("phase") == "test" and t.get("message") == msg]
    return rows[-1] if rows else None


def recalls(arm_json, subject, verb, obj):
    t = _test_turn(arm_json, subject, verb)
    if t is None:
        return None                                             # UNDEFINED: the probe never ran
    svo = [str(w).lower() for w in (t.get("recalled_svo") or [])]
    return bool(t.get("abstained") is False and subject in svo and obj.lower() in svo)


def recalled_object(arm_json, subject, verb):
    t = _test_turn(arm_json, subject, verb)
    if t is None or t.get("abstained") is not False:
        return None
    svo = [str(w).lower() for w in (t.get("recalled_svo") or [])]
    return svo[2] if len(svo) == 3 and svo[0] == subject else None


def _frac(vals):
    vals = [v for v in vals]
    if not vals or any(v is None for v in vals):
        return None
    return sum(1 for v in vals if v) / len(vals)


def _void_reason(a):
    if a is None:
        return "missing"
    if a.get("errors"):
        return "errors: %s" % a["errors"][:1]
    if not a.get("counter_installed"):
        return "write counter not installed"
    return None


def score_seed(arms, K):
    """The pre-registered gate for one seed. `arms` = {name: arm_json | None}."""
    res = {"K": K, "void": {}, "criteria": {}, "secondary": {}}
    for name in GATED_ARMS:
        r = _void_reason(arms.get(name))
        if r:
            res["void"][name] = r
    if res["void"]:
        res["defined"] = False
        res["go"] = None
        return res
    T, N, F, Z, P, E = (arms[n] for n in GATED_ARMS)
    vet = [(f["subject"], f["verb"], f["obj"]) for f in T["vetted"]]
    c = res["criteria"]
    # T1 learns
    t1 = _frac([recalls(T, s, v, o) for s, v, o in vet])
    c["T1_learns"] = {"recall": t1, "pass": t1 is not None and t1 >= T1_MIN_RECALL}
    # T2 teacher lesion
    n_rec = [recalls(N, s, v, o) for s, v, o in vet]
    c["T2_teacher_lesion"] = {"n_recalled": None if None in n_rec else sum(n_rec),
                              "pass": None not in n_rec and sum(n_rec) == 0}
    # T3 freeze (void if the lever did not hold: any taught block nonzero in FREEZE, or none written)
    f_rec = [recalls(F, s, v, o) for s, v, o in vet]
    f_blocks = F.get("taught_blocks_after_teaching") or []
    t_blocks = T.get("taught_blocks_after_teaching") or []
    f_subj = {b["agent"] for b in f_blocks}
    # the freeze arm must have run the SAME write episodes TEACH ran (a block for every subject TEACH wrote), all at 0
    lever_ok = (len(f_blocks) >= 1 and all(b["mean_abs_w"] == 0 for b in f_blocks)
                and {b["agent"] for b in t_blocks} <= f_subj
                and len(t_blocks) >= 1 and all(b["mean_abs_w"] > T3_TEACH_LEVER_MIN for b in t_blocks))
    c["T3_freeze"] = {"n_recalled": None if None in f_rec else sum(f_rec), "lever_ok": lever_ok,
                      "freeze_block_w": [b["mean_abs_w"] for b in f_blocks],
                      "teach_block_w": [b["mean_abs_w"] for b in t_blocks],
                      "pass": (None not in f_rec and sum(f_rec) == 0) if lever_ok else None}
    # T4 zero
    z_rec = [recalls(Z, s, v, o) for s, v, o in vet]
    abl = Z.get("ablation") or []
    abl_ok = (len(abl) >= K and all(a.get("mean_abs_w_after") == 0 for a in abl)
              and all(b["mean_abs_w"] == 0 for b in (Z.get("taught_blocks_at_test") or [{"mean_abs_w": 1}])))
    c["T4_zero"] = {"n_recalled": None if None in z_rec else sum(z_rec), "ablation_ok": abl_ok,
                    "n_ablated": len(abl),
                    "pass": (None not in z_rec and sum(z_rec) == 0) if abl_ok else None}
    # T5 permuted
    perm = [(b["subject"], b["verb"], b["obj"]) for b in P["belief"]]
    p_taught = _frac([recalls(P, s, v, o) for s, v, o in perm])
    p_canon = [recalls(P, s, v, o) for s, v, o in vet]
    c["T5_permuted"] = {"recall_taught": p_taught, "n_recalled_canonical": None if None in p_canon else sum(p_canon),
                        "pass": (p_taught is not None and p_taught >= T5_MIN_RECALL
                                 and None not in p_canon and sum(p_canon) == 0)}
    # T6 teacher error
    cpos = corrupted_positions(K)
    eb = E["belief"]
    corrupt = [(eb[i]["subject"], eb[i]["verb"], eb[i]["obj"], vet[i][2]) for i in cpos]
    clean = [vet[i] for i in range(K) if i not in cpos]
    gt_leak = [recalls(E, s, v, truth) for s, v, _w, truth in corrupt]
    propagated = [recalls(E, s, v, w) for s, v, w, _t in corrupt]
    clean_rec = _frac([recalls(E, s, v, o) for s, v, o in clean])
    c["T6_teacher_error"] = {"n_corrupted": len(corrupt), "ground_truth_recalled": None if None in gt_leak
                             else sum(gt_leak), "propagation_rate": _frac(propagated), "clean_recall": clean_rec,
                             "pass": (None not in gt_leak and sum(gt_leak) == 0 and clean_rec is not None
                                      and clean_rec >= T6_CLEAN_MIN_RECALL)}
    # T7 controls intact in every gated arm
    ctl = {}
    for name in GATED_ARMS:
        ctl[name] = [recalls(arms[name], s, v, o) for s, v, o in CONTROL_PROBES]
    c["T7_controls"] = {"by_arm": ctl, "pass": all(all(x is True for x in v) for v in ctl.values())}
    # T8 isolation
    g = {}
    for name in GATED_ARMS:
        rep = arms[name].get("guard") or {}
        g[name] = {"violations": len(rep.get("violations") or []), "n_patched": rep.get("n_patched"),
                   "required_present": all(any(req.split(".")[-1] == pn.split(".")[-1] and
                                               req.split(".")[0] in pn for pn in (rep.get("patched_names") or []))
                                           for req in T8_REQUIRED),
                   "teacher_calls": (rep.get("calls_by_attribution") or {}).get("teacher", 0)}
    c["T8_isolation"] = {"by_arm": g, "pass": all(v["violations"] == 0 and v["teacher_calls"] == 0 and
                                                  (v["n_patched"] or 0) >= T8_MIN_PATCHED and v["required_present"]
                                                  for v in g.values())}
    # T9 no writes during the test phase
    tw = {name: (arms[name].get("writes_by_phase") or {}).get("test", 0) for name in GATED_ARMS}
    c["T9_no_test_writes"] = {"by_arm": tw, "pass": all(v == 0 for v in tw.values())}
    passes = [v["pass"] for v in c.values()]
    res["defined"] = all(p is not None for p in passes)
    res["go"] = all(p is True for p in passes) if res["defined"] else None
    # secondary (reported, never gating)
    content = []
    for s, v, o in vet + [tuple(x) for x in CONTROL_PROBES]:
        ot = recalled_object(T, s, v)
        if ot is None:
            continue
        learned = recalled_object(Z, s, v) != ot and recalled_object(N, s, v) != ot
        content.append({"probe": [s, v], "teach_object": ot, "learned": learned})
    res["secondary"]["learned_content_fraction_test"] = (
        None if not content else sum(1 for x in content if x["learned"]) / len(content))
    res["secondary"]["test_content"] = content
    res["secondary"]["delivery"] = _delivery(T)
    # attribution: what fraction of TEACH's recall is absent in each control (1.0 = all of it rides the manipulation)
    from tools.lab import attributable_to
    f_rec_frac = _frac(f_rec)
    z_rec_frac = _frac(z_rec)
    n_rec_frac = _frac(n_rec)
    res["secondary"]["attributable_to"] = {
        "write (TEACH vs FREEZE)": attributable_to("ai_teacher recall: TEACH vs FREEZE", t1, f_rec_frac)
        if None not in (t1, f_rec_frac) else None,
        "taught synapses (TEACH vs ZERO)": attributable_to("ai_teacher recall: TEACH vs ZERO", t1, z_rec_frac)
        if None not in (t1, z_rec_frac) else None,
        "teaching (TEACH vs NOTEACH)": attributable_to("ai_teacher recall: TEACH vs NOTEACH", t1, n_rec_frac)
        if None not in (t1, n_rec_frac) else None,
    }
    return res


def _delivery(arm):
    t = (arm or {}).get("teacher") or {}
    les = t.get("lesson") or []
    qz = t.get("quiz") or []
    return {"curiosity_answer": sum(1 for r in les if r.get("delivered_by") == "curiosity_answer"),
            "tell": sum(1 for r in les if r.get("delivered_by") == "tell"),
            "pre_known": sum(1 for r in les if r.get("pre_known")),
            "quiz_right": sum(1 for r in qz if r.get("answered_right")),
            "quiz_corrections": sum(1 for r in qz if r.get("corrected"))}


def session_lcf(arm):
    """Learned-content fraction over the WHOLE session's question replies (warmup, lesson/quiz asks, test probes):
    the fraction of content-bearing recalls whose subject is a taught (in-session learned) subject."""
    if not arm:
        return None
    taught = {f["subject"] for f in arm.get("vetted", [])}
    rows = [t for t in arm.get("turns", []) if str(t.get("message", "")).startswith("what does")
            and t.get("abstained") is False and t.get("recalled_svo")]
    if not rows:
        return None
    return sum(1 for t in rows if str(t["recalled_svo"][0]).lower() in taught) / len(rows)


def retention_curve(arm_dir, seed, Ks):
    out = {}
    for K in Ks:
        a = _load(arm_file(arm_dir, seed, "TEACH", K))
        if a is None or _void_reason(a):
            out[str(K)] = None
            continue
        out[str(K)] = _frac([recalls(a, f["subject"], f["verb"], f["obj"]) for f in a["vetted"]])
    return out


def null_check(T, R):
    if T is None or R is None:
        return None
    key = lambda t: (t.get("phase"), t.get("message"), t.get("abstained"), json.dumps(t.get("recalled_svo")),
                     t.get("answer"))
    a = [key(t) for t in T.get("turns", [])]
    b = [key(t) for t in R.get("turns", [])]
    return {"n_turns": len(a), "n_diffs": sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))}


def score(arm_dir, seeds, K, k_sweep):
    per = {}
    for s in seeds:
        arms = {n: _load(arm_file(arm_dir, s, n, K)) for n in ARMS}
        r = score_seed(arms, K)
        r["retention_vs_K"] = retention_curve(arm_dir, s, sorted(set([K] + list(k_sweep))))
        r["session_lcf_teach"] = session_lcf(arms.get("TEACH"))
        r["null_TEACH_vs_REP"] = null_check(arms.get("TEACH"), arms.get("TEACH_REP"))
        r["costs"] = {n: {"elapsed_s": (a or {}).get("elapsed_s"), "peak_rss_mb": (a or {}).get("peak_rss_mb"),
                          "n_turns": len((a or {}).get("turns", []))} for n, a in arms.items() if a}
        per[str(s)] = r
    n_def = sum(1 for r in per.values() if r["defined"])
    n_go = sum(1 for r in per.values() if r["go"] is True)
    backends = sorted({str(r.get("backend")) for s in seeds for n in ARMS
                       for r in [_load(arm_file(arm_dir, s, n, K))] if r})
    return {"K": K, "seeds": seeds, "backend": ",".join(backends) or None, "per_seed": per,
            "n_defined": n_def, "n_go": n_go,
            "GO": bool(n_def == len(seeds) and n_go == len(seeds))}


# ── self-test: the gate must PASS the capability case and FAIL in every failing direction ─────────────────────────
def _synthetic(K=4):
    facts = [{"subject": "s%d" % i, "verb": "v", "obj": "o%d" % i, "id": "f%d" % i, "tier": "novel"} for i in range(K)]

    def arm(recall_map, name, belief=None, blocks_w=1.0, extra=None):
        turns = []
        for f in facts:
            o = recall_map.get(f["subject"])
            turns.append({"phase": "test", "message": "what does the %s v" % f["subject"],
                          "abstained": o is None, "recalled_svo": None if o is None else [f["subject"], "v", o]})
        for s, v, o in CONTROL_PROBES:
            turns.append({"phase": "test", "message": "what does the %s %s" % (s, v), "abstained": False,
                          "recalled_svo": [s, v, o]})
        a = {"arm": name, "vetted": copy.deepcopy(facts), "belief": copy.deepcopy(belief or facts), "turns": turns,
             "counter_installed": True, "errors": [], "writes_by_phase": {"lesson": K},
             "taught_blocks_after_teaching": [{"agent": f["subject"], "mean_abs_w": blocks_w} for f in facts],
             "taught_blocks_at_test": [{"agent": f["subject"], "mean_abs_w": blocks_w} for f in facts],
             "guard": {"violations": [], "n_patched": 40, "calls_by_attribution": {"brain": K},
                       "patched_names": ["OneBrainComposer._write_block", "ChatBrain._maybe_acquire",
                                         "d6_hebbian_store.hebbian_encode"]}}
        a.update(extra or {})
        return a
    perm_belief = [dict(f, obj=facts[(i + 1) % K]["obj"]) for i, f in enumerate(facts)]
    err_belief = [dict(f, obj=("d%d" % i) if i in corrupted_positions(K) else f["obj"]) for i, f in enumerate(facts)]
    good = {
        "TEACH": arm({f["subject"]: f["obj"] for f in facts}, "TEACH"),
        "NOTEACH": arm({}, "NOTEACH", blocks_w=0.0, extra={"taught_blocks_after_teaching": [],
                                                          "taught_blocks_at_test": []}),
        "FREEZE": arm({}, "FREEZE", blocks_w=0.0),
        "ZERO": arm({}, "ZERO", extra={"ablation": [{"mean_abs_w_after": 0} for _ in facts],
                                       "taught_blocks_at_test": [{"agent": f["subject"], "mean_abs_w": 0}
                                                                 for f in facts]}),
        "PERM": arm({b["subject"]: b["obj"] for b in perm_belief}, "PERM", belief=perm_belief),
        "ERR": arm({b["subject"]: b["obj"] for b in err_belief}, "ERR", belief=err_belief),
    }
    return good, facts


def selftest():
    K = 4
    good, facts = _synthetic(K)
    checks = {}
    r = score_seed(good, K)
    checks["capability_case_GO"] = r["go"] is True
    checks["lcf_is_K_over_K_plus_controls"] = abs(r["secondary"]["learned_content_fraction_test"] - K / (K + 2)) < 1e-9

    def mutate(fn):
        g = copy.deepcopy(good)
        fn(g)
        return score_seed(g, K)
    s0 = facts[0]["subject"]

    def _set(g, arm, subj, obj):
        for t in g[arm]["turns"]:
            if t["message"] == "what does the %s v" % subj:
                t["abstained"], t["recalled_svo"] = (obj is None), (None if obj is None else [subj, "v", obj])
    # T1: the brain does not learn
    checks["T1_fails_no_learning"] = mutate(lambda g: [_set(g, "TEACH", f["subject"], None) for f in facts])["go"] is False
    # T2: recall without teaching (a leak)
    checks["T2_fails_leak"] = mutate(lambda g: _set(g, "NOTEACH", s0, facts[0]["obj"]))["go"] is False
    # T3: recall survives the freeze
    checks["T3_fails_recall_under_freeze"] = mutate(lambda g: _set(g, "FREEZE", s0, facts[0]["obj"]))["go"] is False
    # T3 void: a freeze block is nonzero (e.g. a direct-copy write bypassed the freeze) -> UNDEFINED, never a pass
    def _unfrozen(g):
        g["FREEZE"]["taught_blocks_after_teaching"][0]["mean_abs_w"] = 1.0
    checks["T3_undefined_when_lever_broken"] = mutate(_unfrozen)["go"] is None

    def _skipped(g):
        g["FREEZE"]["taught_blocks_after_teaching"] = g["FREEZE"]["taught_blocks_after_teaching"][1:]
    checks["T3_undefined_when_freeze_skipped_a_write"] = mutate(_skipped)["go"] is None
    # T4: recall survives zeroing
    checks["T4_fails_recall_after_zero"] = mutate(lambda g: _set(g, "ZERO", s0, facts[0]["obj"]))["go"] is False
    def _noabl(g):
        g["ZERO"]["ablation"] = []
    checks["T4_undefined_without_ablation"] = mutate(_noabl)["go"] is None
    # T5: the permuted-teacher brain recalls the CANONICAL objects (it read the source, not the teacher)
    checks["T5_fails_canonical_recall"] = mutate(
        lambda g: [_set(g, "PERM", f["subject"], f["obj"]) for f in facts])["go"] is False
    # T6: the corrupted-teacher brain recalls the ground truth it was never told
    def _gt(g):
        i = corrupted_positions(K)[0]
        _set(g, "ERR", facts[i]["subject"], facts[i]["obj"])
    checks["T6_fails_ground_truth_leak"] = mutate(_gt)["go"] is False
    # T7: a control fact lost
    def _ctl(g):
        for t in g["ZERO"]["turns"]:
            if t["message"] == "what does the cat eat":
                t["abstained"], t["recalled_svo"] = True, None
    checks["T7_fails_control_lost"] = mutate(_ctl)["go"] is False
    # T8: a teacher-attributed store call
    def _viol(g):
        g["PERM"]["guard"]["calls_by_attribution"]["teacher"] = 1
    checks["T8_fails_teacher_write"] = mutate(_viol)["go"] is False
    def _unguarded(g):
        g["TEACH"]["guard"]["patched_names"] = ["ChatBrain._maybe_acquire"]
    checks["T8_fails_required_entry_unpatched"] = mutate(_unguarded)["go"] is False
    # T9: a write during the test phase
    def _tw(g):
        g["TEACH"]["writes_by_phase"]["test"] = 1
    checks["T9_fails_test_write"] = mutate(_tw)["go"] is False
    # void arm -> UNDEFINED, never GO/NO-GO
    def _void(g):
        g["ERR"] = None
    checks["missing_arm_is_undefined"] = mutate(_void)["defined"] is False
    ok = all(checks.values())
    print(json.dumps(checks, indent=2))
    print("SELFTEST %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--seeds", type=int, nargs="*", default=[7])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--k-sweep", type=int, nargs="*", default=[])
    ap.add_argument("--arms", nargs="*", default=list(GATED_ARMS))
    ap.add_argument("--arm-dir", default="research/findings/raw/_ai_teacher")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--sleep-ticks", type=int, default=1)
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.worker:
        return _worker(a.arm, a.K, a.seed, a.out, a.sleep_ticks)
    if not a.score_only:
        jobs = [(arm, a.K, s) for s in a.seeds for arm in a.arms]
        jobs += [("TEACH", k, s) for s in a.seeds for k in a.k_sweep if k != a.K]
        jobs = [j for j in jobs if not os.path.exists(arm_file(a.arm_dir, j[2], j[0], j[1]))]
        print("[ai_teacher] %d arm job(s), jobs=%d" % (len(jobs), a.jobs), flush=True)
        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            for res in ex.map(lambda j: _spawn(j[0], j[1], j[2], a.arm_dir, a.sleep_ticks), jobs):
                print("[ai_teacher] done %s" % (res,), flush=True)
    verdict = score(a.arm_dir, a.seeds, a.K, a.k_sweep)
    print(json.dumps({k: verdict[k] for k in ("K", "seeds", "n_defined", "n_go", "GO")}))
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(verdict, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
