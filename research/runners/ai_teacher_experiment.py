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
  SHAM      TEACH, then after sleep the ablation's write path is run on every taught block with its OWN weights (no
            change), and every block that is neither taught nor a control is zeroed (AMENDMENT 2, T10: the T4 cut's
            loss is specific to the taught synapses, not to the procedure or to losing weight elsewhere)
  TEACH_REP identical to TEACH, rebuilt (determinism null; seed-7 dev smoke only)

AMENDMENT 2 (see the PREREGISTRATION's amendment log): three-valued criteria and a Kleene seed rule (a MEASURED fail
is NO-GO whatever any other criterion reads; a T1 fail reads the TEACH arm alone), T4 defined when every block ZERO
held was cut and the cut covers every subject TEACH wrote (not ">= K blocks"), the SHAM arm and T10, and per-arm
provenance preconditions (sidecar present and naming this arm, clean, manifest verified, revision governed by the
pre-registration, one revision across a seed's T1-T9 arms).

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
    "SHAM":      {"teacher": True,  "freeze": False, "belief": "vetted",    "post_sleep": "sham_taught_cut_offtarget"},
}
CORE_ARMS = ("TEACH", "NOTEACH", "FREEZE", "ZERO", "PERM", "ERR")     # the T1-T9 arms (one revision per seed)
GATED_ARMS = CORE_ARMS + ("SHAM",)                                      # amendment 2 adds SHAM (T10)
CONTROL_SUBJECTS = frozenset(s for s, _v, _o in CONTROL_PROBES)
SEEDS6 = [42, 43, 44, 100, 101, 102]
# the pre-registration commit: an arm is evidence only if it was produced by this commit or a descendant of it
PREREG_SHA = "c12c0d47e07ba301bd8fa14b6bd4030c8edc6b29"
PRE_LESION_PHASES = ("warmup", "lesson", "quiz")
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


def sham_rewrite_block(comp, block_idx):
    """EXPERIMENTER SHAM (arm SHAM, amendment 2). Runs `d6_hebbian_store.ablate_block`'s exact path -- encoding gain
    off, `comp._write_block`, the store-CSR / CSR-cache / fact-shard invalidation -- on block `block_idx`, but writes
    the block's OWN current weights back instead of zeros. Not a brain mechanism: the surgery without the cut. The
    returned `max_abs_dw` is the lever read (it must be exactly 0)."""
    import numpy as np
    D = comp.D
    cur = np.array([complex(t[2]) for t in comp.store_conns[block_idx * D:(block_idx + 1) * D]], dtype=np.complex128)
    before = float(np.mean(np.abs(cur)))
    g_fn = getattr(comp, "encoding_gain_fn", None)
    comp.encoding_gain_fn = None                      # g=1: the rewrite is the block's own weights, exactly
    try:
        comp._write_block(block_idx, cur.copy())
    finally:
        comp.encoding_gain_fn = g_fn
    comp._store_csr = None
    if getattr(comp, "_csr_cache", None) is not None:
        comp._csr_cache = {}
    comp._fact_shard = None
    comp._fact_shard_built_K = -1
    new = np.array([complex(t[2]) for t in comp.store_conns[block_idx * D:(block_idx + 1) * D]], dtype=np.complex128)
    return {"block": int(block_idx), "mean_abs_w_before": before, "mean_abs_w_after": float(np.mean(np.abs(new))),
            "max_abs_dw": float(np.max(np.abs(new - cur))) if len(new) == len(cur) and len(cur) else None}


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

    def taught_blocks(comp, subjects, offtarget=False):
        """The blocks whose agent is a taught subject; with `offtarget`, those whose agent is NEITHER a taught subject
        NOR a control-probe subject (the SHAM arm's off-target cut)."""
        rows = []
        D = comp.D
        for j, (f, _h) in enumerate(getattr(comp, "kb", []) or []):
            ag = str(f.get("agent", "")).lower()
            if (ag not in subjects and ag not in CONTROL_SUBJECTS) if offtarget else (ag in subjects):
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
        elif cfg["post_sleep"] == "sham_taught_cut_offtarget":
            # (amendment 2) the surgery without the cut on every taught block, then a real cut of every block that is
            # neither taught nor a control (at K=4: the build-time 'brain' self-facts)
            out["sham"] = [sham_rewrite_block(comp, row["block"]) for row in taught_blocks(comp, subjects)]
            off = []
            for row in taught_blocks(comp, subjects, offtarget=True):
                rec = ablate_block(comp, row["block"])
                rec.update({"agent": row["agent"], "action": row["action"], "patient": row["patient"]})
                off.append(rec)
            out["offtarget_ablation"] = off
        out["taught_blocks_at_test"] = taught_blocks(comp, subjects) if comp is not None else None
        out["offtarget_blocks_at_test"] = taught_blocks(comp, subjects, offtarget=True) if comp is not None else None
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


# ── provenance (amendment 2): an arm is evidence only if its sidecar says what produced it, and that is clean ──────
_GOVERNED_CACHE = {}


def _governed(sha):
    """True iff `sha` is the pre-registration commit or a descendant of it (so the registration governs the code that
    produced the arm). None when git cannot answer (no repository, object missing): unknown is not a pass."""
    if not sha:
        return None
    if sha not in _GOVERNED_CACHE:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        try:
            r = subprocess.run(["git", "merge-base", "--is-ancestor", PREREG_SHA, sha], cwd=root,
                               capture_output=True, text=True, timeout=30)
            _GOVERNED_CACHE[sha] = True if r.returncode == 0 else (False if r.returncode == 1 else None)
        except Exception:
            _GOVERNED_CACHE[sha] = None
    return _GOVERNED_CACHE[sha]


def _argv_value(argv, flag):
    argv = [str(x) for x in (argv or [])]
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def arm_provenance(path, arm, seed, K):
    """(reason or None, git_sha). P1 the `.prov.json` sidecar exists and names THIS arm (artifact basename, and the
    worker argv's --arm/--seed/--K); P2 git_dirty is False and, for a git-archive revision, the source manifest was
    verified at start AND at exit; P3 the revision is the pre-registration commit or a descendant of it."""
    p = _load(path + ".prov.json")
    if not isinstance(p, dict):
        return "no provenance sidecar", None
    sha = p.get("git_sha")
    if os.path.basename(str(p.get("artifact") or "")) != os.path.basename(path):
        return "sidecar names another artifact (%s)" % p.get("artifact"), sha
    argv = p.get("argv") or []
    got = (_argv_value(argv, "--arm"), _argv_value(argv, "--seed"), _argv_value(argv, "--K"))
    if got != (arm, str(seed), str(K)):
        return "sidecar argv ran arm/seed/K %s, not %s" % (got, (arm, seed, K)), sha
    if p.get("git_dirty") is not False:
        return "git_dirty=%s" % p.get("git_dirty"), sha
    if p.get("source_kind") == "git_archive" and not (p.get("source_manifest_verified_at_start") is True
                                                      and p.get("source_manifest_verified_at_exit") is True):
        return "source manifest not verified at start and at exit", sha
    gov = _governed(sha)
    if gov is not True:
        return ("revision %s is not the pre-registration commit or a descendant of it" % (sha,) if gov is False
                else "cannot establish that revision %s descends from the pre-registration commit" % (sha,)), sha
    return None, sha


def seed_provenance(arm_dir, seed, K):
    """{arm: reason or None} for the gated arms that have a file, plus P4: every T1-T9 arm of one seed ran at ONE
    revision (else none of them can be singled out as the leftover, and all are void). SHAM, added by amendment 2 and
    run later, is exempt from P4; T10's same-session read checks it against TEACH turn for turn instead."""
    prov, shas = {}, {}
    for name in GATED_ARMS:
        path = arm_file(arm_dir, seed, name, K)
        if not os.path.exists(path):
            continue
        prov[name], shas[name] = arm_provenance(path, name, seed, K)
    core = {n: shas.get(n) for n in CORE_ARMS if n in prov and prov[n] is None}
    if len(set(core.values())) > 1:
        why = "the T1-T9 arms of this seed ran at different revisions: %s" % {n: str(s)[:9] for n, s in core.items()}
        for n in core:
            prov[n] = why
    return prov, shas


# ── the gate ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _tri_all(vals):
    """Kleene AND over True / False / None: False if any is False, else None if any is None, else True."""
    vals = list(vals)
    if any(v is False for v in vals):
        return False
    if any(v is None for v in vals):
        return None
    return True


def _vet(a):
    return [(f["subject"], f["verb"], f["obj"]) for f in (a or {}).get("vetted", [])]


def _turn_key(t):
    return (t.get("phase"), t.get("message"), t.get("abstained"), json.dumps(t.get("recalled_svo")), t.get("answer"))


def score_seed(arms, K, prov=None):
    """The gate for one seed (pre-registered; amendment 2's three-valued form). `arms` = {name: arm_json | None};
    `prov` = {name: provenance reason | None} (None: provenance not checked -- the aggregate always checks it).

    Every criterion reads a declared set of arms and is True / False / None. None (UNDEFINED) when an arm it reads is
    void (missing, errored, no write counter, provenance failed), a probe it reads never ran, or its lever did not
    hold. The seed is NO-GO if ANY criterion is False -- a measured failure is never hidden by a criterion elsewhere
    that could not be measured (review 2026-09-24, issue 1: a refused or weak write made T3/T4 UNDEFINED and hid a
    T1 fail) -- GO if all are True, else UNDEFINED."""
    res = {"K": K, "void": {}, "criteria": {}, "secondary": {}, "provenance_checked": prov is not None}
    for name in GATED_ARMS:
        r = _void_reason(arms.get(name)) or (prov or {}).get(name)
        if r:
            res["void"][name] = r
    c = res["criteria"]

    def crit(key, reads, fn):
        bad = {n: res["void"][n] for n in reads if n in res["void"]}
        if bad:
            c[key] = {"reads": list(reads), "pass": None, "undefined": "void arm(s): %s" % json.dumps(bad)}
            return
        d = fn()
        d["reads"] = list(reads)
        if d["pass"] is None and not d.get("undefined"):
            d["undefined"] = "a probe it reads never ran"
        c[key] = d

    T, N, F, Z, P, E, S = (arms.get(n) for n in GATED_ARMS)
    vet = _vet(T) if T else _vet(next((a for a in arms.values() if a), None))

    def rec(a, facts):
        return [recalls(a, s, v, o) for s, v, o in facts]

    # T1 learns (reads TEACH alone: a T1 fail is NO-GO whatever any other arm or lever reads)
    def t1():
        r = _frac(rec(T, vet))
        return {"recall": r, "pass": None if r is None else r >= T1_MIN_RECALL}
    crit("T1_learns", ("TEACH",), t1)

    # T2 teacher lesion
    def t2():
        n = rec(N, vet)
        return {"n_recalled": None if None in n else sum(n), "pass": None if None in n else sum(n) == 0}
    crit("T2_teacher_lesion", ("NOTEACH",), t2)

    # T3 freeze (defined only if the lever held: every FREEZE taught block 0, a FREEZE block for every subject TEACH
    # wrote, and every TEACH taught block > T3_TEACH_LEVER_MIN)
    def t3():
        f_rec = rec(F, vet)
        f_blocks = F.get("taught_blocks_after_teaching") or []
        t_blocks = T.get("taught_blocks_after_teaching") or []
        lever_ok = (len(f_blocks) >= 1 and all(b["mean_abs_w"] == 0 for b in f_blocks)
                    and {b["agent"] for b in t_blocks} <= {b["agent"] for b in f_blocks}
                    and len(t_blocks) >= 1 and all(b["mean_abs_w"] > T3_TEACH_LEVER_MIN for b in t_blocks))
        d = {"n_recalled": None if None in f_rec else sum(f_rec), "lever_ok": lever_ok,
             "freeze_block_w": [b["mean_abs_w"] for b in f_blocks], "teach_block_w": [b["mean_abs_w"] for b in t_blocks],
             "pass": ((None if None in f_rec else sum(f_rec) == 0) if lever_ok else None)}
        if not lever_ok:
            d["undefined"] = "the freeze lever did not hold"
        return d
    crit("T3_freeze", ("FREEZE", "TEACH"), t3)

    # T4 zero (amendment 2: defined when EVERY taught block ZERO held was cut and reads 0 at test, and the cut covers
    # every subject TEACH wrote -- not "at least K blocks", which read UNDEFINED whenever one told fact was refused)
    def t4():
        z_rec = rec(Z, vet)
        abl = Z.get("ablation") or []
        abl_ids = {a.get("block") for a in abl}
        at_test = Z.get("taught_blocks_at_test")
        t_subj = {b["agent"] for b in (T.get("taught_blocks_after_teaching") or [])}
        abl_ok = bool(at_test is not None
                      and all(a.get("mean_abs_w_after") == 0 for a in abl)
                      and all(b.get("block") in abl_ids and b["mean_abs_w"] == 0 for b in at_test)
                      and t_subj <= {b["agent"] for b in at_test if b.get("block") in abl_ids})
        d = {"n_recalled": None if None in z_rec else sum(z_rec), "ablation_ok": abl_ok, "n_ablated": len(abl),
             "n_teach_blocks": len(t_subj),
             "pass": ((None if None in z_rec else sum(z_rec) == 0) if abl_ok else None)}
        if not abl_ok:
            d["undefined"] = "the ablation did not cover every taught block"
        return d
    crit("T4_zero", ("ZERO", "TEACH"), t4)

    # T5 permuted
    def t5():
        perm = [(b["subject"], b["verb"], b["obj"]) for b in P["belief"]]
        p_taught = _frac(rec(P, perm))
        p_canon = rec(P, vet)
        ok = p_taught is not None and None not in p_canon
        return {"recall_taught": p_taught, "n_recalled_canonical": None if None in p_canon else sum(p_canon),
                "pass": (p_taught >= T5_MIN_RECALL and sum(p_canon) == 0) if ok else None}
    crit("T5_permuted", ("PERM",), t5)

    # T6 teacher error
    def t6():
        cpos = corrupted_positions(K)
        eb = E["belief"]
        corrupt = [(eb[i]["subject"], eb[i]["verb"], eb[i]["obj"], vet[i][2]) for i in cpos]
        clean = [vet[i] for i in range(K) if i not in cpos]
        gt_leak = [recalls(E, s, v, truth) for s, v, _w, truth in corrupt]
        propagated = [recalls(E, s, v, w) for s, v, w, _t in corrupt]
        clean_rec = _frac(rec(E, clean))
        ok = None not in gt_leak and clean_rec is not None
        return {"n_corrupted": len(corrupt), "ground_truth_recalled": None if None in gt_leak else sum(gt_leak),
                "propagation_rate": _frac(propagated), "clean_recall": clean_rec,
                "pass": (sum(gt_leak) == 0 and clean_rec >= T6_CLEAN_MIN_RECALL) if ok else None}
    crit("T6_teacher_error", ("ERR",), t6)

    # T7 controls intact / T8 isolation / T9 no test-phase writes -- per arm, Kleene across arms (a void arm is None
    # for itself, never a pass and never a fail; a measured fail in any valid arm fails the criterion)
    ctl, g, tw = {}, {}, {}
    for name in GATED_ARMS:
        a = arms.get(name)
        if name in res["void"]:
            ctl[name] = g[name] = tw[name] = None
            continue
        ctl[name] = [recalls(a, s, v, o) for s, v, o in CONTROL_PROBES]
        rep = a.get("guard") or {}
        g[name] = {"violations": len(rep.get("violations") or []), "n_patched": rep.get("n_patched"),
                   "required_present": all(any(req.split(".")[-1] == pn.split(".")[-1] and
                                               req.split(".")[0] in pn for pn in (rep.get("patched_names") or []))
                                           for req in T8_REQUIRED),
                   "teacher_calls": (rep.get("calls_by_attribution") or {}).get("teacher", 0)}
        tw[name] = (a.get("writes_by_phase") or {}).get("test", 0)
    c["T7_controls"] = {"reads": list(GATED_ARMS), "by_arm": ctl, "pass": _tri_all(
        None if v is None else _tri_all(v) for v in ctl.values())}
    c["T8_isolation"] = {"reads": list(GATED_ARMS), "by_arm": g, "pass": _tri_all(
        None if v is None else (v["violations"] == 0 and v["teacher_calls"] == 0 and
                                (v["n_patched"] or 0) >= T8_MIN_PATCHED and v["required_present"])
        for v in g.values())}
    c["T9_no_test_writes"] = {"reads": list(GATED_ARMS), "by_arm": tw,
                              "pass": _tri_all(None if v is None else v == 0 for v in tw.values())}
    for k in ("T7_controls", "T8_isolation", "T9_no_test_writes"):
        if c[k]["pass"] is None:
            c[k]["undefined"] = "void arm(s) or a probe that never ran: %s" % json.dumps(
                {n: res["void"][n] for n in GATED_ARMS if n in res["void"]})

    # T10 sham (amendment 2): the cut's loss is specific to the taught synapses. SHAM loses NONE of the taught facts
    # TEACH recalls. Defined only if SHAM and TEACH are the same session up to the manipulation (warmup, lesson and quiz
    # turns identical), the sham rewrite ran on every taught block SHAM held (covering every subject TEACH wrote) and
    # moved no weight, every taught block still reads > 0 at test, and at least one off-target block was cut, every
    # off-target block SHAM held was cut, and each reads 0 at test.
    def t10():
        s_rec, t_rec = rec(S, vet), rec(T, vet)
        pre_t = [_turn_key(t) for t in T.get("turns", []) if t.get("phase") in PRE_LESION_PHASES]
        pre_s = [_turn_key(t) for t in S.get("turns", []) if t.get("phase") in PRE_LESION_PHASES]
        same_session = bool(pre_t) and pre_t == pre_s
        sham = S.get("sham") or []
        sham_ids = {x.get("block") for x in sham}
        s_at_test = S.get("taught_blocks_at_test")
        t_subj = {b["agent"] for b in (T.get("taught_blocks_after_teaching") or [])}
        sham_ok = bool(s_at_test is not None
                       and all(b.get("block") in sham_ids and b["mean_abs_w"] > 0 for b in s_at_test)
                       and t_subj <= {b["agent"] for b in s_at_test if b.get("block") in sham_ids}
                       and all(x.get("max_abs_dw") == 0 for x in sham))
        off = S.get("offtarget_ablation") or []
        off_ids = {a.get("block") for a in off}
        off_at_test = S.get("offtarget_blocks_at_test")
        off_ok = bool(len(off) >= 1 and off_at_test is not None
                      and all(a.get("mean_abs_w_after") == 0 for a in off)
                      and all(b.get("block") in off_ids and b["mean_abs_w"] == 0 for b in off_at_test))
        lever_ok = same_session and sham_ok and off_ok
        measured = None not in s_rec and None not in t_rec
        n_lost = sum(1 for a, b in zip(t_rec, s_rec) if a and not b) if measured else None
        d = {"n_teach_recalled": sum(1 for x in t_rec if x) if measured else None,
             "n_sham_recalled": sum(1 for x in s_rec if x) if measured else None, "n_lost": n_lost,
             "same_session_pre_lesion": same_session, "n_pre_lesion_turns": len(pre_t), "sham_ok": sham_ok,
             "n_sham_blocks": len(sham), "sham_max_abs_dw": [x.get("max_abs_dw") for x in sham],
             "offtarget_ok": off_ok, "n_offtarget_cut": len(off),
             "offtarget_cut": ["%s %s %s" % (a.get("agent"), a.get("action"), a.get("patient")) for a in off],
             "pass": (n_lost == 0 if measured else None) if lever_ok else None}
        if not lever_ok:
            d["undefined"] = "sham lever did not hold (same_session=%s sham_ok=%s offtarget_ok=%s)" % (
                same_session, sham_ok, off_ok)
        return d
    crit("T10_sham", ("SHAM", "TEACH"), t10)

    passes = {k: v["pass"] for k, v in c.items()}
    go = _tri_all(passes.values())
    res["go"] = go
    res["defined"] = go is not None
    res["status"] = "GO" if go is True else ("NO-GO" if go is False else "UNDEFINED")
    res["failed"] = [k for k, p in passes.items() if p is False]
    res["undefined"] = [k for k, p in passes.items() if p is None]

    # secondary (reported, never gating)
    sec = res["secondary"]
    ok = lambda n: n not in res["void"]                                  # noqa: E731
    content = []
    if ok("TEACH") and ok("ZERO") and ok("NOTEACH"):
        for s, v, o in vet + [tuple(x) for x in CONTROL_PROBES]:
            ot = recalled_object(T, s, v)
            if ot is None:
                continue
            learned = recalled_object(Z, s, v) != ot and recalled_object(N, s, v) != ot
            content.append({"probe": [s, v], "teach_object": ot, "learned": learned})
    sec["learned_content_fraction_test"] = (
        None if not content else sum(1 for x in content if x["learned"]) / len(content))
    sec["test_content"] = content
    sec["delivery"] = _delivery(T) if ok("TEACH") else None
    # exposure per arm (review issue 4: FREEZE is NOT the same input as TEACH -- its wrong quiz replies draw one
    # correction per fact, so it gets more turns and more write episodes; reported so the finding states it)
    sec["exposure_by_arm"] = {n: {"n_turns": len((arms.get(n) or {}).get("turns", [])),
                                  "writes_by_phase": (arms.get(n) or {}).get("writes_by_phase"),
                                  "quiz_corrections": (_delivery(arms.get(n)) or {}).get("quiz_corrections")}
                              for n in GATED_ARMS if ok(n)}
    tw_ = [b["mean_abs_w"] for b in ((T or {}).get("taught_blocks_after_teaching") or [])] if ok("TEACH") else []
    sec["teach_taught_block_w"] = {"min": min(tw_) if tw_ else None, "max": max(tw_) if tw_ else None, "n": len(tw_)}
    # attribution: what fraction of TEACH's recall is absent in each control (1.0 = all of it rides the manipulation)
    from tools.lab import attributable_to
    t1v = c["T1_learns"].get("recall")
    fr = {n: (_frac(rec(arms[n], vet)) if ok(n) else None) for n in ("FREEZE", "ZERO", "NOTEACH")}
    sec["attributable_to"] = {
        "write (TEACH vs FREEZE)": attributable_to("ai_teacher recall: TEACH vs FREEZE", t1v, fr["FREEZE"])
        if None not in (t1v, fr["FREEZE"]) else None,
        "taught synapses (TEACH vs ZERO)": attributable_to("ai_teacher recall: TEACH vs ZERO", t1v, fr["ZERO"])
        if None not in (t1v, fr["ZERO"]) else None,
        "teaching (TEACH vs NOTEACH)": attributable_to("ai_teacher recall: TEACH vs NOTEACH", t1v, fr["NOTEACH"])
        if None not in (t1v, fr["NOTEACH"]) else None,
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
    a = [_turn_key(t) for t in T.get("turns", [])]
    b = [_turn_key(t) for t in R.get("turns", [])]
    return {"n_turns": len(a), "n_diffs": sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))}


def score(arm_dir, seeds, K, k_sweep):
    per = {}
    for s in seeds:
        arms = {n: _load(arm_file(arm_dir, s, n, K)) for n in ARMS}
        prov, shas = seed_provenance(arm_dir, s, K)
        r = score_seed(arms, K, prov=prov)
        r["provenance"] = {n: {"ok": prov[n] is None, "reason": prov[n], "git_sha": shas.get(n)} for n in prov}
        r["retention_vs_K"] = retention_curve(arm_dir, s, sorted(set([K] + list(k_sweep))))
        r["session_lcf_teach"] = session_lcf(arms.get("TEACH"))
        r["null_TEACH_vs_REP"] = null_check(arms.get("TEACH"), arms.get("TEACH_REP"))
        r["costs"] = {n: {"elapsed_s": (a or {}).get("elapsed_s"), "peak_rss_mb": (a or {}).get("peak_rss_mb"),
                          "n_turns": len((a or {}).get("turns", []))} for n, a in arms.items() if a}
        per[str(s)] = r
    n_def = sum(1 for r in per.values() if r["defined"])
    n_go = sum(1 for r in per.values() if r["go"] is True)
    n_nogo = sum(1 for r in per.values() if r["go"] is False)
    backends = sorted({str(r.get("backend")) for s in seeds for n in ARMS
                       for r in [_load(arm_file(arm_dir, s, n, K))] if r})
    earned = earn_verdict(arm_dir, seeds, K, per)
    return {"K": K, "seeds": seeds, "backend": ",".join(backends) or None, "per_seed": per,
            "n_defined": n_def, "n_go": n_go, "n_nogo": n_nogo,
            "GO": earned["status"] == "GO", "status": earned["status"], "preconditions": earned["preconditions"],
            "verdict": earned}


def _probes_unrun(arm_json, K):
    """Every test probe the gate reads for this arm, and which of them has no test-phase turn (never ran)."""
    probes = [(f["subject"], f["verb"]) for f in (arm_json or {}).get("vetted", [])[:K]]
    probes += [(b["subject"], b["verb"]) for b in (arm_json or {}).get("belief", [])[:K]]
    probes += [(s, v) for s, v, _o in CONTROL_PROBES]
    return sorted({"%s %s" % p for p in probes if _test_turn(arm_json, *p) is None})


def earn_verdict(arm_dir, seeds, K, per):
    """The aggregate verdict carries its preconditions (amendment 1) under amendment 2's rule. A seed is GO / NO-GO /
    UNDEFINED by `score_seed`. Aggregate NO-GO iff some seed is NO-GO: it is registered with exactly what that NO-GO
    rests on (the arms each failing criterion reads are valid with clean provenance, and the criterion was measured),
    so a missing arm on another seed cannot hide it. Aggregate GO iff every seed is GO, with every seed's every arm and
    criterion registered. Otherwise UNDEFINED, with the same full registration (some of it unmet)."""
    from tools.verdict import Verdict
    v = Verdict("ai_teacher K=%d: T1-T10 on every registered seed %s (amendment 2)" % (K, list(seeds)))
    nogo = [s for s in seeds if per[str(s)]["status"] == "NO-GO"]

    def reg_arm(s, name):
        r = per[str(s)]
        a_path = arm_file(arm_dir, s, name, K)
        unrun = _probes_unrun(_load(a_path), K) if os.path.exists(a_path) else None
        v.require("seed %s: arm %s present, error-free, write counter on, provenance clean" % (s, name),
                  name not in r["void"], expect=True,
                  note=json.dumps({"void": r["void"].get(name), "probes_unrun": unrun,
                                   "git_sha": (r.get("provenance", {}).get(name) or {}).get("git_sha")}))

    def reg_crit(s, key):
        cr = per[str(s)]["criteria"][key]
        v.require("seed %s: %s measured" % (s, key), cr["pass"] is not None, expect=True,
                  note="pass=%s %s" % (cr["pass"], cr.get("undefined") or ""))

    if nogo:
        for s in nogo:
            r = per[str(s)]
            for key in r["failed"]:
                for name in r["criteria"][key]["reads"]:
                    if name in r["void"]:
                        continue            # a per-arm criterion (T7-T9) failed on the valid arms; void ones are None
                    reg_arm(s, name)
                reg_crit(s, key)
        return v.decide(go=False, verbose=False)
    for s in seeds:
        for name in GATED_ARMS:
            reg_arm(s, name)
        for key in per[str(s)]["criteria"]:
            reg_crit(s, key)
    return v.decide(go=all(per[str(s)]["status"] == "GO" for s in seeds), verbose=False)


# ── self-test: the gate must PASS the capability case and FAIL in every failing direction ─────────────────────────
def _synthetic(K=4):
    facts = [{"subject": "s%d" % i, "verb": "v", "obj": "o%d" % i, "id": "f%d" % i, "tier": "novel"} for i in range(K)]
    warm = {"phase": "warmup", "message": WARMUP, "abstained": False, "recalled_svo": ["cat", "eat", "fish"],
            "answer": "the cat eats the fish"}

    def blocks(w, subj=None):
        return [{"block": 5 + i, "agent": f["subject"], "mean_abs_w": w} for i, f in enumerate(facts)
                if subj is None or f["subject"] in subj]

    def arm(recall_map, name, belief=None, blocks_w=1.0, extra=None):
        turns = [dict(warm)]
        for f in facts:
            o = recall_map.get(f["subject"])
            turns.append({"phase": "test", "message": "what does the %s v" % f["subject"],
                          "abstained": o is None, "recalled_svo": None if o is None else [f["subject"], "v", o]})
        for s, v, o in CONTROL_PROBES:
            turns.append({"phase": "test", "message": "what does the %s %s" % (s, v), "abstained": False,
                          "recalled_svo": [s, v, o]})
        a = {"arm": name, "vetted": copy.deepcopy(facts), "belief": copy.deepcopy(belief or facts), "turns": turns,
             "counter_installed": True, "errors": [], "writes_by_phase": {"lesson": K},
             "taught_blocks_after_teaching": blocks(blocks_w), "taught_blocks_at_test": blocks(blocks_w),
             "guard": {"violations": [], "n_patched": 40, "calls_by_attribution": {"brain": K},
                       "patched_names": ["OneBrainComposer._write_block", "ChatBrain._maybe_acquire",
                                         "d6_hebbian_store.hebbian_encode"]}}
        a.update(extra or {})
        return a
    perm_belief = [dict(f, obj=facts[(i + 1) % K]["obj"]) for i, f in enumerate(facts)]
    err_belief = [dict(f, obj=("d%d" % i) if i in corrupted_positions(K) else f["obj"]) for i, f in enumerate(facts)]
    off = [{"block": j, "agent": "brain", "action": "a%d" % j, "patient": "p%d" % j, "mean_abs_w_before": 1.0,
            "mean_abs_w_after": 0} for j in range(3)]
    good = {
        "TEACH": arm({f["subject"]: f["obj"] for f in facts}, "TEACH"),
        "NOTEACH": arm({}, "NOTEACH", blocks_w=0.0, extra={"taught_blocks_after_teaching": [],
                                                          "taught_blocks_at_test": []}),
        "FREEZE": arm({}, "FREEZE", blocks_w=0.0),
        "ZERO": arm({}, "ZERO", extra={"ablation": [{"block": b["block"], "mean_abs_w_after": 0} for b in blocks(1.0)],
                                       "taught_blocks_at_test": blocks(0)}),
        "PERM": arm({b["subject"]: b["obj"] for b in perm_belief}, "PERM", belief=perm_belief),
        "ERR": arm({b["subject"]: b["obj"] for b in err_belief}, "ERR", belief=err_belief),
        "SHAM": arm({f["subject"]: f["obj"] for f in facts}, "SHAM", extra={
            "sham": [{"block": b["block"], "mean_abs_w_before": 1.0, "mean_abs_w_after": 1.0, "max_abs_dw": 0.0}
                     for b in blocks(1.0)],
            "offtarget_ablation": off,
            "offtarget_blocks_at_test": [{"block": a["block"], "agent": "brain", "mean_abs_w": 0} for a in off]}),
    }
    return good, facts


def _write_synth(d, seed, arms, K, sha=PREREG_SHA, sidecar=None):
    """Write synthetic arms + provenance sidecars into `d` (seed `seed`). `sidecar` = {arm: dict overrides | None to
    omit the sidecar}."""
    sidecar = sidecar or {}
    for n in ARMS:
        path = arm_file(d, seed, n, K)
        for p in (path, path + ".prov.json"):
            if os.path.exists(p):
                os.remove(p)
    for n, a in arms.items():
        if a is None:
            continue
        path = arm_file(d, seed, n, K)
        with open(path, "w") as fh:
            json.dump(a, fh)
        if n in sidecar and sidecar[n] is None:
            continue
        p = {"artifact": path, "argv": ["x", "--worker", "--arm", n, "--K", str(K), "--seed", str(seed)],
             "git_sha": sha, "git_dirty": False, "source_kind": "git_archive",
             "source_manifest_verified_at_start": True, "source_manifest_verified_at_exit": True}
        p.update(sidecar.get(n) or {})
        with open(path + ".prov.json", "w") as fh:
            json.dump(p, fh)


def selftest():
    K = 4
    good, facts = _synthetic(K)
    checks = {}
    r = score_seed(good, K)
    checks["capability_case_GO"] = r["go"] is True and r["status"] == "GO"
    checks["lcf_is_K_over_K_plus_controls"] = abs(r["secondary"]["learned_content_fraction_test"] - K / (K + 2)) < 1e-9

    def mutate(fn):
        g = copy.deepcopy(good)
        fn(g)
        return score_seed(g, K)
    s0 = facts[0]["subject"]

    def _set(g, arm, subj, obj):
        for t in g[arm]["turns"]:
            if t["phase"] == "test" and t["message"] == "what does the %s v" % subj:
                t["abstained"], t["recalled_svo"] = (obj is None), (None if obj is None else [subj, "v", obj])

    def _keep_blocks(g, subj):
        """the told sentences outside `subj` were refused: no block for them in any arm that ran the lesson"""
        for n in ("TEACH", "FREEZE", "ZERO", "PERM", "ERR", "SHAM"):
            for k in ("taught_blocks_after_teaching", "taught_blocks_at_test"):
                g[n][k] = [b for b in g[n][k] if b["agent"] in subj]
        keep = {b["block"] for b in g["TEACH"]["taught_blocks_after_teaching"]}
        g["ZERO"]["ablation"] = [a for a in g["ZERO"]["ablation"] if a["block"] in keep]
        g["SHAM"]["sham"] = [a for a in g["SHAM"]["sham"] if a["block"] in keep]
    # T1: the brain does not learn
    checks["T1_fails_no_learning"] = mutate(lambda g: [_set(g, "TEACH", f["subject"], None) for f in facts])["go"] is False
    # REVIEW 2026-09-24 issue 1 (a): no write in ANY arm -> T3 UNDEFINED, but the T1 fail must still read NO-GO
    def _no_write(g):
        for f in facts:
            for n in ("TEACH", "PERM", "ERR", "SHAM"):
                _set(g, n, f["subject"], None)
        _keep_blocks(g, set())
    rr = mutate(_no_write)
    checks["review1a_no_write_anywhere_NO-GO"] = (rr["go"] is False and rr["criteria"]["T1_learns"]["pass"] is False
                                                  and rr["criteria"]["T3_freeze"]["pass"] is None)
    # (b) half the told facts refused (T1 = 0.5) -> NO-GO, and T4 is now DEFINED (the cut covered every block written)
    def _half(g):
        keep = {facts[0]["subject"], facts[1]["subject"]}
        for f in facts[2:]:
            for n in ("TEACH", "SHAM"):
                _set(g, n, f["subject"], None)
        _keep_blocks(g, keep)
    rr = mutate(_half)
    checks["review1b_half_refused_NO-GO_T4_defined"] = (rr["go"] is False and rr["criteria"]["T1_learns"]["pass"] is False
                                                       and rr["criteria"]["T4_zero"]["pass"] is True)
    # (c) weak TEACH writes (|w| 0.3) and 0 recall -> T3 UNDEFINED (lever), T1 fail -> NO-GO
    def _weak(g):
        for k in ("taught_blocks_after_teaching", "taught_blocks_at_test"):
            for b in g["TEACH"][k]:
                b["mean_abs_w"] = 0.3
        for f in facts:
            _set(g, "TEACH", f["subject"], None)
    rr = mutate(_weak)
    checks["review1c_weak_write_no_recall_NO-GO"] = (rr["go"] is False and rr["criteria"]["T3_freeze"]["pass"] is None)
    # 3 of 4 learned, the 4th refused in every arm: T1 = 0.75 passes and T4 is DEFINED (it was UNDEFINED under >= K)
    def _three(g):
        keep = {f["subject"] for f in facts[1:]}
        for n in ("TEACH", "SHAM", "PERM", "ERR"):
            _set(g, n, facts[0]["subject"], None)
        _keep_blocks(g, keep)
    rr = mutate(_three)
    checks["review1_three_of_four_T4_defined_GO"] = (rr["go"] is True and rr["criteria"]["T4_zero"]["pass"] is True)
    # Kleene: a measured fail elsewhere is not hidden by an UNDEFINED lever (T2 leak + broken T3 lever -> NO-GO)
    def _leak_and_lever(g):
        _set(g, "NOTEACH", s0, facts[0]["obj"])
        g["FREEZE"]["taught_blocks_after_teaching"][0]["mean_abs_w"] = 1.0
    rr = mutate(_leak_and_lever)
    checks["kleene_T2_fail_with_T3_undefined_NO-GO"] = rr["go"] is False and rr["criteria"]["T3_freeze"]["pass"] is None
    # a T1 fail reads TEACH alone: NO-GO even with another arm missing
    def _t1_fail_err_missing(g):
        [_set(g, "TEACH", f["subject"], None) for f in facts]
        g["ERR"] = None
    checks["T1_fail_with_other_arm_missing_NO-GO"] = mutate(_t1_fail_err_missing)["go"] is False
    # a probe that never ran is UNDEFINED, never a fail (the old scorer failed T1 on it)
    def _unrun_teach(g):
        g["TEACH"]["turns"] = [t for t in g["TEACH"]["turns"] if t["message"] != "what does the %s v" % s0]
    rr = mutate(_unrun_teach)
    checks["unrun_probe_is_UNDEFINED_not_fail"] = rr["go"] is None and rr["criteria"]["T1_learns"]["pass"] is None
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

    def _partial_abl(g):
        g["ZERO"]["ablation"] = g["ZERO"]["ablation"][1:]
    checks["T4_undefined_when_a_held_block_was_not_cut"] = mutate(_partial_abl)["go"] is None

    def _zero_missed_subject(g):
        g["ZERO"]["taught_blocks_at_test"] = g["ZERO"]["taught_blocks_at_test"][1:]
        g["ZERO"]["ablation"] = g["ZERO"]["ablation"][1:]
    checks["T4_undefined_when_cut_misses_a_subject_TEACH_wrote"] = mutate(_zero_missed_subject)["go"] is None
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
            if t["phase"] == "test" and t["message"] == "what does the cat eat":
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
    # T10 (amendment 2): the sham loses a taught fact TEACH recalls -> NO-GO
    checks["T10_fails_sham_loses_a_fact"] = mutate(lambda g: _set(g, "SHAM", s0, None))["go"] is False

    def _sham_moved(g):
        g["SHAM"]["sham"][0]["max_abs_dw"] = 0.01
    checks["T10_undefined_when_sham_moved_a_weight"] = mutate(_sham_moved)["go"] is None

    def _sham_diverged(g):
        g["SHAM"]["turns"][0]["answer"] = "something else"
    checks["T10_undefined_when_sessions_diverged_before_the_lesion"] = mutate(_sham_diverged)["go"] is None

    def _no_offtarget(g):
        g["SHAM"]["offtarget_ablation"] = []
    checks["T10_undefined_without_an_offtarget_cut"] = mutate(_no_offtarget)["go"] is None

    def _offtarget_uncut(g):
        g["SHAM"]["offtarget_blocks_at_test"][0]["mean_abs_w"] = 0.9
    checks["T10_undefined_when_an_offtarget_block_survived"] = mutate(_offtarget_uncut)["go"] is None
    # void arm -> UNDEFINED, never GO/NO-GO

    def _void(g):
        g["ERR"] = None
    checks["missing_arm_is_undefined"] = mutate(_void)["defined"] is False

    def _void_sham(g):
        g["SHAM"] = None
    checks["missing_sham_is_undefined"] = mutate(_void_sham)["go"] is None
    # the aggregate verdict carries its preconditions (amendment 1) and provenance (amendment 2): GO when earned,
    # UNDEFINED when an arm is missing / a probe never ran / provenance fails, NO-GO when a criterion failed on arms
    # whose own preconditions held -- even while another seed is incomplete
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        def agg(arms, seeds=(1,), sidecar=None, sha=PREREG_SHA, per_seed=None):
            for s in seeds:
                _write_synth(d, s, (per_seed or {}).get(s, arms), K, sha=sha, sidecar=sidecar)
            return score(d, list(seeds), K, [])
        g = agg(copy.deepcopy(good))
        checks["aggregate_GO_carries_preconditions"] = g["status"] == "GO" and g["GO"] is True and bool(
            g["preconditions"]) and all(p["ok"] is True for p in g["preconditions"])
        bad = copy.deepcopy(good)
        bad["ERR"] = None
        checks["aggregate_missing_arm_UNDEFINED"] = agg(bad)["status"] == "UNDEFINED"
        bad = copy.deepcopy(good)
        bad["NOTEACH"]["turns"] = [t for t in bad["NOTEACH"]["turns"]
                                   if not (t["phase"] == "test" and t["message"] == "what does the cat eat")]
        checks["aggregate_unrun_probe_UNDEFINED"] = agg(bad)["status"] == "UNDEFINED"
        bad = copy.deepcopy(good)
        _set(bad, "TEACH", facts[0]["subject"], None)
        _set(bad, "TEACH", facts[1]["subject"], None)
        n = agg(bad)
        checks["aggregate_criterion_fail_NO-GO"] = n["status"] == "NO-GO" and n["GO"] is False
        # review issue 1 at the aggregate: a refused/weak write (T3/T4 levers down) with a T1 fail reads NO-GO
        bad = copy.deepcopy(good)
        _weak(bad)
        checks["aggregate_review1c_NO-GO"] = agg(bad)["status"] == "NO-GO"
        # one seed NO-GO, another seed with every arm missing -> NO-GO (the incomplete seed cannot hide it)
        bad = copy.deepcopy(good)
        _set(bad, "TEACH", facts[0]["subject"], None)
        _set(bad, "TEACH", facts[1]["subject"], None)
        n = agg(None, seeds=(1, 2), per_seed={1: bad, 2: {}})
        checks["aggregate_NO-GO_not_hidden_by_missing_seed"] = n["status"] == "NO-GO" and n["n_nogo"] == 1
        # provenance (review issue 8)
        checks["prov_missing_sidecar_UNDEFINED"] = agg(copy.deepcopy(good), sidecar={"ZERO": None})["status"] == "UNDEFINED"
        checks["prov_dirty_UNDEFINED"] = agg(copy.deepcopy(good), sidecar={"PERM": {"git_dirty": True}})[
            "status"] == "UNDEFINED"
        checks["prov_manifest_unverified_UNDEFINED"] = agg(copy.deepcopy(good), sidecar={
            "FREEZE": {"source_manifest_verified_at_exit": False}})["status"] == "UNDEFINED"
        checks["prov_wrong_argv_UNDEFINED"] = agg(copy.deepcopy(good), sidecar={
            "TEACH": {"argv": ["x", "--arm", "ZERO", "--K", "4", "--seed", "1"]}})["status"] == "UNDEFINED"
        # 153af81bf predates the pre-registration (the dev-seed-3 plumbing code): not governed
        checks["prov_pre_registration_revision_UNDEFINED"] = agg(copy.deepcopy(good), sidecar={
            "NOTEACH": {"git_sha": "153af81bf35436a6d4faf97fbe0162604f4fd10f"}})["status"] == "UNDEFINED"
        # a T1-T9 arm at another (governed) revision than its seed-mates -> the seed's T1-T9 arms are void
        checks["prov_mixed_core_revisions_UNDEFINED"] = agg(copy.deepcopy(good), sidecar={
            "ERR": {"git_sha": "490a13cc3848bde373a90ac884672b451c0812ca"}})["status"] == "UNDEFINED"
        # SHAM (run later, by amendment 2) may carry a later governed revision
        rr = agg(copy.deepcopy(good), sidecar={"SHAM": {"git_sha": "490a13cc3848bde373a90ac884672b451c0812ca"}})
        checks["prov_sham_at_later_governed_revision_GO"] = rr["status"] == "GO"
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
        for j in jobs:                          # an arm already on disk is SKIPPED: say what produced it (review issue 8;
            path = arm_file(a.arm_dir, j[2], j[0], j[1])    # the scorer voids it unless its provenance is clean)
            if os.path.exists(path):
                why, sha = arm_provenance(path, j[0], j[2], j[1])
                print("[ai_teacher] skip existing %s (git_sha=%s provenance=%s)" % (path, sha, why or "clean"),
                      flush=True)
        jobs = [j for j in jobs if not os.path.exists(arm_file(a.arm_dir, j[2], j[0], j[1]))]
        print("[ai_teacher] %d arm job(s), jobs=%d" % (len(jobs), a.jobs), flush=True)
        with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
            for res in ex.map(lambda j: _spawn(j[0], j[1], j[2], a.arm_dir, a.sleep_ticks), jobs):
                print("[ai_teacher] done %s" % (res,), flush=True)
    verdict = score(a.arm_dir, a.seeds, a.K, a.k_sweep)
    print(json.dumps({k: verdict[k] for k in ("K", "seeds", "n_defined", "n_go", "n_nogo", "GO", "status")}))
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(verdict, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
