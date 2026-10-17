"""ONE-BRAIN PROBE (M0 of docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md): does a fact taught in one
conversation reach another conversation, and does it survive a restart?

THE TEST. A fact is taught in conversation A (session id A) and asked in conversation B (session id B), through the
REAL `webapp.server.brain_chat` handler called directly (as the regression-battery worker does), never a runner-level
shortcut. Every arm is a FRESH process (one subprocess per phase), seed `--seed`, numpy, the GPU-free stub renderer,
`rich` omitted (the production default fluent path), the production LTM default (wikidata_100k attached when the data
lake is present). Persistence writes go ONLY to the arm's own directory (BRAIN_PERSIST_DIR), never bridges/persist,
and are deleted after the arm unless --keep-persist.

  A_teach      session A  "the wug eats the dax"        a nonce fact (no prior, not in the build KB, not in the LTM)
  A_ask        session A  "what does the wug eat"       same-conversation sanity: learning works inside A
  B_ask        session B  "what does the wug eat"       THE READ: does B know what A was told?
  A_ask_again  session A  "what does the wug eat"       restart arms only, after the restart: same identity as A
  B_ltm        session B  "Tell me about angora_turkey."  CONTROL: a fact in the imported LTM (wikidata_100k)
  B_build      session B  "what does the dog chase"     CONTROL: a build-time fact of the tiny-demo KB (dog chase cat)

A turn is read from the reply's SVO fields (recalled_svo, supporting_facts, derived_svo), never from the text:
  recall   = not abstained and an SVO matches the expected role (patient 'dax' / agent 'angora_turkey' / patient 'cat')
  abstain  = abstained and no SVO matches
  other    = anything else (an answer that names no expected fact, or an SVO match on an abstained turn)

ARMS. "needs" = the code the arm requires; an arm whose need is absent from webapp/ is reported UNDEFINED (with the
missing need), never run on a substitute. At M0 the server had per-session brains only, so the M2 arms were UNDEFINED;
M2 (webapp/one_brain.py, `BRAIN_ONE_BRAIN`, default off) provides every need. Dev seeds 7 and 8 at M2:
research/findings/raw/_one_brain_probe/m2_s7.json, m2_s7_instruments.json, m2_s8.json (docs/drafts/one-brain-m2/
WORKLOG.md section 5).

  intact                 today's per-session brains                  phases [A_teach A_ask B_ask B_ltm B_build]
  restart_persist_off    BRAIN_PERSIST_LEARNING=0                    [A_teach A_ask @flush] | [B_ask A_ask_again B_ltm
                                                                     B_build]   (fresh process after '|')
  restart_persist_on     BRAIN_PERSIST_LEARNING=1                    same phases as restart_persist_off
  one_brain              needs BRAIN_ONE_BRAIN (M2), set to 1        same turns as intact
  freeze_learning        needs BRAIN_ONE_BRAIN + BRAIN_ONE_BRAIN_FREEZE_LEARNING (the shared store's plasticity gate
                         shut: the teach turn is heard, nothing is written)      same turns as intact
  zero_learned_store     needs BRAIN_ONE_BRAIN + `def one_brain_zero_learned_store(` in webapp/ (the experimenter
                         zeroes every block learned since build, build KB + LTM prior untouched)
                                                                     [A_teach A_ask @zero B_ask B_ltm B_build]
  restart_persist_on_one_brain   needs BRAIN_ONE_BRAIN; BRAIN_ONE_BRAIN=1 + BRAIN_PERSIST_LEARNING=1, restart phases

`@flush` runs the server's real shutdown hook (`_persist_learning_on_shutdown`, a no-op when persistence is off);
`@zero` calls `webapp.server.one_brain_zero_learned_store()` (the M2 build provides it, or the arm stays UNDEFINED).

PREDICTIONS (written 2026-10-13, BEFORE any run of this file; the runner scores each arm against PREDICTIONS below):
  intact                A_ask recall, B_ask ABSTAIN (B has its own brain; nothing A was told reaches it),
                        B_ltm recall, B_build recall.
  restart_persist_off   A_ask recall; after the restart A_ask_again ABSTAIN and B_ask ABSTAIN (nothing was saved:
                        0 files in the persist dir); controls recall.
  restart_persist_on    A_ask recall; after the restart A_ask_again RECALL (the same session identity reloads its
                        saved facts) and B_ask ABSTAIN (persistence is keyed per session today); controls recall.
  one_brain (M2-ON)     A_ask recall, B_ask RECALL (one learned store for every conversation); controls recall.
  freeze_learning       A_ask ABSTAIN and B_ask ABSTAIN (nothing written); controls recall (the lesion is specific).
  zero_learned_store    A_ask recall, then after the zero B_ask ABSTAIN; controls recall (the prior is untouched).
  restart_persist_on_one_brain   A_ask recall; after the restart B_ask RECALL and A_ask_again RECALL.
The LTM control (B_ltm) carries the M0 instrument check: if it does not recall, the instrument cannot tell "B does
not know A's fact" from "B recalls nothing", and the arm's B_ask read is reported but not interpretable.

HONEST SCOPE. M0 documents today's baseline at one DEV seed (7); it is not a generalization claim (six seeds,
42/43/44/100/101/102, are for the M2 flip and need a committed prereg: --prereg PATH). Host code here is
experimenter instrumentation only (the turn script, the SVO read, the flush/zero hooks); none of it feeds a reply.

Self-test (no brain):  .venv/bin/python -m research.runners.one_brain_probe --selftest
Dev run (seed 7; brain lock + memcap, numpy CPU):
  flock -w 7200 /home/dant123/Projects/sim-rev/brain.lock bash -c 'bash tools/mem_ok.sh 14 && bash tools/memcap.sh 14 \
    -- env SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 .venv/bin/python -u -m \
    research.runners.one_brain_probe --seed 7 --arms intact restart_persist_off restart_persist_on \
    --out research/findings/raw/_one_brain_probe/m0_s7.json'
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIX_SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEED = 7
BRAIN = "tiny-demo"
RENDERER = "stub"
SESSION_A = "one_brain_probe_A"
SESSION_B = "one_brain_probe_B"

TEACH = "the wug eats the dax"
ASK = "what does the wug eat"
LTM_Q = "Tell me about angora_turkey."
BUILD_Q = "what does the dog chase"

# label -> (session, message, expected SVO role match (index, value)): 0 agent, 2 patient
TURNS = {
    "A_teach": (SESSION_A, TEACH, None),
    "A_ask": (SESSION_A, ASK, (2, "dax")),
    "B_ask": (SESSION_B, ASK, (2, "dax")),
    "A_ask_again": (SESSION_A, ASK, (2, "dax")),
    "B_ltm": (SESSION_B, LTM_Q, (0, "angora_turkey")),
    "B_build": (SESSION_B, BUILD_Q, (2, "cat")),
}
_INTACT = [["A_teach", "A_ask", "B_ask", "B_ltm", "B_build"]]
_RESTART = [["A_teach", "A_ask", "@flush"], ["B_ask", "A_ask_again", "B_ltm", "B_build"]]

_NEED_ONE_BRAIN = ("env", "BRAIN_ONE_BRAIN")
ARMS = {
    "intact": {"needs": [], "env": {}, "phases": _INTACT},
    "restart_persist_off": {"needs": [], "env": {"BRAIN_PERSIST_LEARNING": "0"}, "phases": _RESTART},
    "restart_persist_on": {"needs": [], "env": {"BRAIN_PERSIST_LEARNING": "1"}, "phases": _RESTART},
    "one_brain": {"needs": [_NEED_ONE_BRAIN], "env": {"BRAIN_ONE_BRAIN": "1"}, "phases": _INTACT},
    "freeze_learning": {"needs": [_NEED_ONE_BRAIN, ("env", "BRAIN_ONE_BRAIN_FREEZE_LEARNING")],
                        "env": {"BRAIN_ONE_BRAIN": "1", "BRAIN_ONE_BRAIN_FREEZE_LEARNING": "1"}, "phases": _INTACT},
    "zero_learned_store": {"needs": [_NEED_ONE_BRAIN, ("def", "one_brain_zero_learned_store")],
                           "env": {"BRAIN_ONE_BRAIN": "1"},
                           "phases": [["A_teach", "A_ask", "@zero", "B_ask", "B_ltm", "B_build"]]},
    "restart_persist_on_one_brain": {"needs": [_NEED_ONE_BRAIN],
                                     "env": {"BRAIN_ONE_BRAIN": "1", "BRAIN_PERSIST_LEARNING": "1"},
                                     "phases": _RESTART},
}
_KNOWN = {"B_ltm": "recall", "B_build": "recall"}
PREDICTIONS = {
    "intact": dict(_KNOWN, A_ask="recall", B_ask="abstain"),
    "restart_persist_off": dict(_KNOWN, A_ask="recall", B_ask="abstain", A_ask_again="abstain"),
    "restart_persist_on": dict(_KNOWN, A_ask="recall", B_ask="abstain", A_ask_again="recall"),
    "one_brain": dict(_KNOWN, A_ask="recall", B_ask="recall"),
    "freeze_learning": dict(_KNOWN, A_ask="abstain", B_ask="abstain"),
    "zero_learned_store": dict(_KNOWN, A_ask="recall", B_ask="abstain"),
    "restart_persist_on_one_brain": dict(_KNOWN, A_ask="recall", B_ask="recall", A_ask_again="recall"),
}
# Recorded per turn (None when absent from the reply).
_FIELDS = ("answer", "abstained", "recalled_svo", "supporting_facts", "derived", "derived_svo", "derived_from",
           "verified", "rich", "renderer", "source", "provenance", "episodic", "followup", "n_sentences")


# ── pure helpers (selftested) ───────────────────────────────────────────────────────────────────────────────────────

def need_present(need, webapp_dir=None, texts=None) -> bool:
    """Is the code an arm needs present in webapp/? ('env', NAME): a quoted string literal NAME (an env read);
    ('def', NAME): a `def NAME(`. `texts` (a list of source strings) replaces the scan for the selftest."""
    kind, name = need
    pat = (re.compile(r"""["']%s["']""" % re.escape(name)) if kind == "env"
           else re.compile(r"\bdef %s\(" % re.escape(name)))
    if texts is None:
        texts = [p.read_text(errors="replace") for p in sorted(Path(webapp_dir or REPO / "webapp").rglob("*.py"))]
    return any(pat.search(t) for t in texts)


def _svos(resp):
    out = []
    for k in ("recalled_svo", "derived_svo"):
        v = resp.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            out.append([str(x).lower() for x in v[:3]])
    for v in resp.get("supporting_facts") or []:
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            out.append([str(x).lower() for x in v[:3]])
    return out


def read_turn(resp, expect):
    """recall / abstain / other (see the module docstring), or 'error' for a failed turn, None for a teach turn."""
    if expect is None:
        return None
    if not isinstance(resp, dict) or "_error" in resp:
        return "error"
    idx, val = expect
    hit = any(s[idx] == val for s in _svos(resp))
    ab = bool(resp.get("abstained"))
    if hit and not ab:
        return "recall"
    if ab and not hit:
        return "abstain"
    return "other"


def score_arm(arm, reads):
    """Compare an arm's reads to PREDICTIONS[arm]: {label: {predicted, observed, match}} + all_match. A control that
    does not recall marks the arm's B_ask read not interpretable."""
    pred = PREDICTIONS[arm]
    rows = {k: {"predicted": v, "observed": reads.get(k), "match": reads.get(k) == v} for k, v in pred.items()}
    known_ok = all(reads.get(k) == "recall" for k in _KNOWN)
    return {"rows": rows, "all_match": all(r["match"] for r in rows.values()), "controls_recall": known_ok,
            "b_ask_interpretable": known_ok}


def seed_allowed(seed, prereg):
    """Dev seeds run freely; a validation seed (42/43/44/100/101/102) needs a committed prereg path."""
    if seed not in SIX_SEEDS:
        return True, ""
    if not prereg:
        return False, "seed %d is a validation seed: pass --prereg <committed prereg path>" % seed
    p = Path(prereg)
    if not p.is_file():
        return False, "prereg %s does not exist" % prereg
    r = subprocess.run(["git", "ls-files", "--error-unmatch", str(p)], cwd=str(REPO), capture_output=True)
    if r.returncode != 0:
        return False, "prereg %s is not committed (git ls-files)" % prereg
    return True, ""


def _tree_size(d):
    files = [p for p in Path(d).rglob("*") if p.is_file()] if Path(d).exists() else []
    return {"n_files": len(files), "bytes": int(sum(p.stat().st_size for p in files)),
            "files": sorted(str(p.relative_to(d)) for p in files)[:200]}


# ── the worker (one fresh process per phase) ─────────────────────────────────────────────────────────────────────────

def run_worker(spec_path, out_path):
    spec = json.loads(Path(spec_path).read_text())
    for k, v in spec["env"].items():
        os.environ[k] = v          # explicit values only (an OFF arm says "0", never relies on a pop)
    t0 = time.time()
    from webapp import server as S
    from webapp import brain_turn_lock as L
    L.reset_stats()
    turns = []
    for step in spec["steps"]:
        ts = time.time()
        if step == "@flush":
            import asyncio
            try:
                asyncio.run(S._persist_learning_on_shutdown())
                turns.append({"label": step, "ok": True})
            except Exception as e:  # noqa: BLE001 -- recorded, never swallowed
                turns.append({"label": step, "_error": "%s: %s" % (type(e).__name__, e)})
            continue
        if step == "@zero":
            try:
                turns.append({"label": step, "result": S.one_brain_zero_learned_store()})
            except Exception as e:  # noqa: BLE001
                turns.append({"label": step, "_error": "%s: %s" % (type(e).__name__, e)})
            continue
        session, msg, expect = TURNS[step]
        rec = {"label": step, "session": session, "message": msg}
        try:
            r = S.brain_chat(S.BrainChatRequest(session=session, message=msg, brain=BRAIN, renderer=RENDERER))
            body = json.loads(bytes(r.body))
            for f in _FIELDS:
                rec[f] = body.get(f)
            rec["composer"] = (body.get("activity") or {}).get("composer") if isinstance(body.get("activity"),
                                                                                        dict) else None
        except Exception as e:  # noqa: BLE001
            rec["_error"] = "%s: %s" % (type(e).__name__, e)
        rec["read"] = read_turn(rec, expect)
        rec["seconds"] = round(time.time() - ts, 2)
        turns.append(rec)
        print("[one_brain_probe] %s %s -> %s (%.1fs)" % (spec["arm"], step, rec["read"], rec["seconds"]),
              flush=True)
    try:
        ltm = S._resolve_ltm_bundle()
    except Exception as e:  # noqa: BLE001
        ltm = "unresolved: %s" % e
    out = {"arm": spec["arm"], "phase": spec["phase"], "turns": turns, "seconds": round(time.time() - t0, 1),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
           "turn_lock": L.stats(), "ltm_bundle": ltm,
           "env": {k: os.environ.get(k) for k in sorted(set(spec["env"]) | {"SIM_BACKEND", "BRAIN_CHAT_SEED",
                                                                            "BRAIN_TURN_LOCK"})}}
    Path(out_path).write_text(json.dumps(out, indent=1, default=str))
    return 0


# ── arm contrasts: which manipulation owns a difference in a read ───────────────────────────────────────────────────

# (name, treatment arm, comparison arm, read label). A read is 1.0 for 'recall', else 0.0.
CONTRASTS = [
    ("persistence flag: A recalls after a restart", "restart_persist_on", "restart_persist_off", "A_ask_again"),
    ("persistence flag: B recalls after a restart", "restart_persist_on", "restart_persist_off", "B_ask"),
    ("one brain: B recalls what A was told", "one_brain", "intact", "B_ask"),
    ("plasticity gate: B recall under one brain", "one_brain", "freeze_learning", "B_ask"),
    ("learned store: B recall under one brain", "one_brain", "zero_learned_store", "B_ask"),
    ("one brain: B recalls after a restart", "restart_persist_on_one_brain", "restart_persist_on", "B_ask"),
]


def contrasts(arms_rec):
    """Per contrast: did the read move between the two arms (tools.lab.lever, never required), and what fraction of
    the treatment arm's read is absent from the comparison arm (tools.lab.attributable_to; None = no effect in
    either arm). An arm that did not run makes the contrast UNDEFINED. Plus the persistence manipulation check:
    checkpoint files written after phase 0, ON vs OFF (a flag that wrote nothing would void the restart contrast)."""
    from tools.lab import attributable_to, lever
    out = []
    for name, t, c, label in CONTRASTS:
        rt, rc = arms_rec.get(t) or {}, arms_rec.get(c) or {}
        if rt.get("status") != "RAN" or rc.get("status") != "RAN":
            out.append({"name": name, "status": "UNDEFINED", "treatment": t, "comparison": c, "read": label,
                        "reason": "arm not run: %s" % ", ".join(a for a, r in ((t, rt), (c, rc))
                                                                 if r.get("status") != "RAN")})
            continue
        vt = 1.0 if rt.get("reads", {}).get(label) == "recall" else 0.0
        vc = 1.0 if rc.get("reads", {}).get(label) == "recall" else 0.0
        out.append({"name": name, "status": "RAN", "treatment": t, "comparison": c, "read": label,
                    "treatment_read": rt.get("reads", {}).get(label), "comparison_read": rc.get("reads", {}).get(label),
                    "moved": lever(name, vc, vt, required=False), "attributable_fraction": attributable_to(name, vt, vc)})
    on, off = arms_rec.get("restart_persist_on") or {}, arms_rec.get("restart_persist_off") or {}
    if on.get("status") == "RAN" and off.get("status") == "RAN":
        n_on = ((on.get("phases") or [{}])[0].get("persist_dir_after") or {}).get("n_files")
        n_off = ((off.get("phases") or [{}])[0].get("persist_dir_after") or {}).get("n_files")
        out.append({"name": "persistence flag: checkpoint files written", "status": "RAN", "treatment_read": n_on,
                    "comparison_read": n_off,
                    "moved": lever("persistence flag: checkpoint files written", n_off, n_on, required=False)})
    return out


# ── the orchestrator ────────────────────────────────────────────────────────────────────────────────────────────────

def run_arm(arm, seed, out_dir, keep_persist=False):
    cfg = ARMS[arm]
    missing = [n for n in cfg["needs"] if not need_present(n)]
    if missing:
        return {"arm": arm, "status": "UNDEFINED",
                "reason": "needs %s" % ", ".join("%s %s" % (k, v) for k, v in missing),
                "prediction": PREDICTIONS[arm]}
    persist = Path(out_dir) / "_persist" / ("%s_s%d" % (arm, seed))
    if persist.exists():
        shutil.rmtree(persist)
    persist.mkdir(parents=True)
    env = {"SIM_BACKEND": os.environ.get("SIM_BACKEND", "numpy"), "BRAIN_CHAT_RENDERER": RENDERER,
           "SIM_DISABLE_LLM": "1", "BRAIN_CHAT_SEED": str(seed), "BRAIN_PERSIST_DIR": str(persist),
           "BRAIN_PERSIST_LEARNING": "0"}
    env.update(cfg["env"])
    phases, reads = [], {}
    for i, steps in enumerate(cfg["phases"]):
        spec = {"arm": arm, "phase": i, "steps": steps, "env": env}
        sp = Path(out_dir) / ("_spec_%s_p%d_s%d.json" % (arm, i, seed))
        po = Path(out_dir) / ("_phase_%s_p%d_s%d.json" % (arm, i, seed))
        sp.write_text(json.dumps(spec))
        cmd = [sys.executable, "-u", "-m", "research.runners.one_brain_probe", "--worker", str(sp), "--out", str(po)]
        t0 = time.time()
        # the orchestrator's own --out carries the run's provenance; the transient phase files are folded into it
        rc = subprocess.call(cmd, cwd=str(REPO), env=dict(os.environ, SIM_NO_PROVENANCE="1", **env))
        ph = json.loads(po.read_text()) if (rc == 0 and po.exists()) else {"_error": "worker rc=%d" % rc}
        ph["rc"], ph["wall_seconds"] = rc, round(time.time() - t0, 1)
        ph["persist_dir_after"] = _tree_size(persist)
        phases.append(ph)
        for t in ph.get("turns", []):
            if t.get("read") is not None:
                reads[t["label"]] = t["read"]
        sp.unlink(missing_ok=True)
        po.unlink(missing_ok=True)
        if rc != 0:
            break
    if not keep_persist:
        shutil.rmtree(persist, ignore_errors=True)
    return {"arm": arm, "status": "RAN", "env": env, "phases": phases, "reads": reads,
            "prediction": PREDICTIONS[arm], "score": score_arm(arm, reads),
            "peak_rss_mb": max([p.get("peak_rss_mb") or 0 for p in phases] or [0])}


def _git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True,
                              text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def selftest():
    # the read: recall / abstain / other / error, from SVO fields only
    assert read_turn({"abstained": False, "recalled_svo": ["wug", "eat", "dax"]}, (2, "dax")) == "recall"
    assert read_turn({"abstained": False, "supporting_facts": [["angora_turkey", "country", "x"]]},
                     (0, "angora_turkey")) == "recall"
    assert read_turn({"abstained": True, "recalled_svo": None, "answer": "the wug eats the dax?"}, (2, "dax")) \
        == "abstain"                                       # the text is never read
    assert read_turn({"abstained": False, "recalled_svo": ["wug", "eat", "moon"]}, (2, "dax")) == "other"
    assert read_turn({"abstained": True, "recalled_svo": ["wug", "eat", "dax"]}, (2, "dax")) == "other"
    assert read_turn({"_error": "boom"}, (2, "dax")) == "error"
    assert read_turn({"answer": "ok"}, None) is None
    # the scorer FAILS in its failing direction: B recalling on the intact arm is a mismatch
    good = {"A_ask": "recall", "B_ask": "abstain", "B_ltm": "recall", "B_build": "recall"}
    assert score_arm("intact", good)["all_match"]
    assert not score_arm("intact", dict(good, B_ask="recall"))["all_match"]
    no_ltm = score_arm("intact", dict(good, B_ltm="abstain"))
    assert not no_ltm["all_match"] and not no_ltm["b_ask_interpretable"]
    # the arm contrasts: a moved read is attributed, an arm that did not run is UNDEFINED (never a 0)
    fake = {"restart_persist_on": {"status": "RAN", "reads": {"A_ask_again": "recall", "B_ask": "abstain"},
                                   "phases": [{"persist_dir_after": {"n_files": 5}}]},
            "restart_persist_off": {"status": "RAN", "reads": {"A_ask_again": "abstain", "B_ask": "abstain"},
                                    "phases": [{"persist_dir_after": {"n_files": 0}}]},
            "intact": {"status": "RAN", "reads": {"B_ask": "abstain"}}, "one_brain": {"status": "UNDEFINED"}}
    cs = {c["name"]: c for c in contrasts(fake)}
    assert cs["persistence flag: A recalls after a restart"]["moved"] is True
    assert cs["persistence flag: A recalls after a restart"]["attributable_fraction"] == 1.0
    assert cs["persistence flag: B recalls after a restart"]["moved"] is False
    assert cs["one brain: B recalls what A was told"]["status"] == "UNDEFINED"
    assert cs["persistence flag: checkpoint files written"]["moved"] is True
    # every arm has a prediction for every read turn it runs
    for arm, cfg in ARMS.items():
        labels = {s for ph in cfg["phases"] for s in ph if not s.startswith("@") and TURNS[s][2] is not None}
        assert labels == set(PREDICTIONS[arm]), (arm, labels, set(PREDICTIONS[arm]))
    # needs: a quoted env literal or a def; the existing BRAIN_ONEBRAIN_SINGLE_POOL flag is NOT the M2 flag
    assert need_present(("env", "BRAIN_ONE_BRAIN"), texts=['os.environ.get("BRAIN_ONE_BRAIN")'])
    assert not need_present(("env", "BRAIN_ONE_BRAIN"), texts=['os.environ.get("BRAIN_ONEBRAIN_SINGLE_POOL")'])
    assert not need_present(("env", "BRAIN_ONE_BRAIN"), texts=['os.environ.get("BRAIN_ONE_BRAIN_FREEZE_X")'])
    assert need_present(("def", "one_brain_zero_learned_store"), texts=["def one_brain_zero_learned_store():"])
    # seeds: dev runs freely, validation seeds need a committed prereg
    assert seed_allowed(7, None)[0] and not seed_allowed(42, None)[0]
    assert not seed_allowed(100, "/nonexistent/prereg.md")[0]
    print("one_brain_probe selftest: OK (today: BRAIN_ONE_BRAIN present in webapp/ = %s)"
          % need_present(_NEED_ONE_BRAIN))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=DEV_SEED)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--prereg", default=None, help="committed prereg path (required for a validation seed)")
    ap.add_argument("--keep-persist", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.worker:
        return run_worker(a.worker, a.out)
    ok, why = seed_allowed(a.seed, a.prereg)
    if not ok:
        print("REFUSED: " + why, file=sys.stderr)
        return 2
    if not a.out:
        ap.error("--out is required")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"runner": "one_brain_probe", "seed": a.seed, "prereg": a.prereg, "git_sha": _git_sha(),
           "design": "docs/plans/2026-10-13-one-brain-shared-memory-DESIGN.md (M0)",
           "backend": os.environ.get("SIM_BACKEND"), "arms": {}}
    for arm in a.arms:
        print("[one_brain_probe] arm %s" % arm, flush=True)
        rec["arms"][arm] = run_arm(arm, a.seed, out.parent, keep_persist=a.keep_persist)
        out.write_text(json.dumps(rec, indent=1, default=str))   # incremental: a later crash keeps earlier arms
    rec["contrasts"] = contrasts(rec["arms"])
    rec["peak_rss_mb_children"] = round(resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0, 1)
    out.write_text(json.dumps(rec, indent=1, default=str))
    for arm, r in rec["arms"].items():
        if r["status"] == "UNDEFINED":
            print("  %-30s UNDEFINED (%s)" % (arm, r["reason"]))
        else:
            print("  %-30s reads=%s all_match=%s controls_recall=%s peak_rss_mb=%s"
                  % (arm, r["reads"], r["score"]["all_match"], r["score"]["controls_recall"], r["peak_rss_mb"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
