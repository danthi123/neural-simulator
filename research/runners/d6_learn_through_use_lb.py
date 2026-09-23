"""D6 — LEARN-THROUGH-USE, lesion-verified: does the brain CHANGE FROM USE, and is the change CARRIED BY ITS OWN
PLASTICITY? A 2x2 (use x plasticity) probe through the REAL /api/brain-chat handler (webapp.server.brain_chat).

CHARTER D6 (docs/plans/2026-09-23-autonomous-charter.md): "Continuous learning (learn-through-use / novelty /
surprise-driven) is lesion-verified load-bearing + on-by-default -- the brain measurably changes from use, 6-seed."

WHY A NEW INSTRUMENT (not the load-bearing battery). The D1 battery (research/runners/load_bearing_fraction.py) cuts a
faculty's READ and asks whether the reply changes. D6 asks a different question: does something taught in turn N
change a LATER reply, and does FREEZING THE WRITE (plasticity), with the read intact, remove that change? The battery
cannot ask it: `in-loop-learning` is `kind="in-process"` there (not covered), and the in-process LEARN check it defers
to (research/runners/_production_lesion_probe.py) has a check that cannot fail -- its lesion verdict tests for the
word "bird" in an answer about a "deer" (FAILURE_LOG 2026-09-23). And the production write itself was a host copy of
the composite into the weights, not a plasticity rule; research/runners/d6_hebbian_store.py makes it a LOCAL
Hebbian rule (default-OFF BRAIN_D6_HEBBIAN_STORE) with a write-only freeze lesion (BRAIN_D6_HEBBIAN_FREEZE).

THE DESIGN (per seed; every arm is a FRESH brain built at BRAIN_CHAT_SEED=seed in its own subprocess, numpy/CPU, stub
renderer, no LLM -- the same worker discipline as research/runners/onebrain_regression_battery.py):
  session 'd6u' turns, in order:
    teach    USE: "the wolf hunts the deer"   SHUF: "the fox eats the berry"   (in-conversation acquisition -> write)
    d1       "what does the cat eat"      (intervening turn; a question -> no write)
    d2       "what does the dog chase"    (intervening turn + the READ-INTACT check: a build-time fact, never frozen)
    probe    "what does the wolf hunt"    (depends on what was learned at `teach`)
    xprobe   "what does the fox eat"      (the SHUF arm's own learned fact -> the double dissociation)
  arms:
    USE_H     BRAIN_D6_HEBBIAN_STORE=1, teach=USE                         (learns via the local Hebbian rule)
    USE_H_REP identical to USE_H, rebuilt                                  (NULL / determinism control)
    SHUF_H    BRAIN_D6_HEBBIAN_STORE=1, teach=SHUF                        (shuffled-content control: learning happened,
                                                                            but not of the probed fact)
    FREEZE_H  BRAIN_D6_HEBBIAN_STORE=1 + BRAIN_D6_HEBBIAN_FREEZE=1, teach=USE (the D6 LESION: same encode activity,
                                                                            eta=0 for in-conversation writes)
    USE_D     BRAIN_D6_HEBBIAN_STORE=0, teach=USE                         (the production direct-copy write: no-regression)

PRE-REGISTERED GO GATE (written 2026-09-23 BEFORE any result existed; do not edit after results land). Decision
fields per turn: `abstained`, `recalled_svo`, `answer` (categorical; the stub renderer is deterministic). Per seed,
ALL of:
  C1 LEARNS         USE_H.probe recalls 'deer' (not abstained, 'deer' in recalled_svo).
  C2 USE CHANGES IT USE_H.probe decision != SHUF_H.probe decision, and SHUF_H.probe does NOT recall 'deer'.
  C3 FREEZE REMOVES FREEZE_H.probe does NOT recall 'deer' and FREEZE_H.probe decision == SHUF_H.probe decision
                    (the use-driven difference vanishes when only the plasticity is frozen).
  C4 WRITE-ONLY     FREEZE_H.teach decision == USE_H.teach decision (the encode turn still parsed + acknowledged) AND
                    FREEZE_H.d2 decision == USE_H.d2 decision (the read path is intact: build-time recall unchanged)
                    AND the lever moved: learned |w| of the taught block > 0.5 in USE_H and == 0 in FREEZE_H.
  C5 SPECIFIC       SHUF_H.xprobe recalls 'berry' AND USE_H.xprobe does not (double dissociation: each arm learned
                    its own fact and only its own fact).
  C6 DETERMINISTIC  USE_H == USE_H_REP on every turn's decision fields (null control clean -> the C2/C3 differences
                    are attributable, attributable_to == 1.0, not run-to-run noise).
  C7 NO-REGRESSION  USE_D decisions == USE_H decisions on teach/d2/probe/xprobe (the Hebbian write is decision-
                    identical to the host copy it replaces, on this protocol).
AGGREGATE GO = C1..C7 hold on all 6 seeds (42 43 44 100 101 102). An arm that fails to build/run is VOID (reported,
never scored as a failure or a pass); a seed with any VOID arm is UNDEFINED, not NO-GO.
The literal scoring command:  .venv/bin/python -m research.runners.d6_learn_through_use_lb --score-only \
    --arm-dir research/findings/raw/_d6_learn_through_use --seeds 42 43 44 100 101 102 \
    --json research/findings/raw/_d6_learn_through_use/d6_ltu_6seed_verdict.json

HONEST SCOPE. This measures ONE learning pathway (in-conversation declarative fact acquisition, on the tiny-demo
brain). It is not a claim that the brain learns open-endedly, and the instructive pattern the rule stores is still
produced by the composer's FHRR bind/bundle (the standing composer idealization). The rule is local and its output
IS the weight; which trigger cell a new fact takes is host bookkeeping (declared in d6_hebbian_store.py).

Run one seed (full brain, ~10-25 min numpy; ALWAYS memcapped):
  bash tools/mem_ok.sh 12 && SIM_BACKEND=numpy bash tools/memcap.sh 12 -- .venv/bin/python -u \
      -m research.runners.d6_learn_through_use_lb --seeds 42 --arm-dir research/findings/raw/_d6_learn_through_use \
      --json research/findings/raw/_d6_learn_through_use/d6_ltu_s42.json
Self-test (no brain): .venv/bin/python -m research.runners.d6_learn_through_use_lb --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

SESSION = "d6u"
TEACH_USE = "the wolf hunts the deer"
TEACH_SHUF = "the fox eats the berry"
TURNS = [  # (label, message)  -- the teach message is filled per arm
    ("teach", None),
    ("d1", "what does the cat eat"),
    ("d2", "what does the dog chase"),
    ("probe", "what does the wolf hunt"),
    ("xprobe", "what does the fox eat"),
]
DECISION_FIELDS = ("abstained", "recalled_svo", "answer")

ARMS = {
    "USE_H":     ({"BRAIN_D6_HEBBIAN_STORE": "1", "BRAIN_D6_HEBBIAN_FREEZE": "0"}, TEACH_USE),
    "USE_H_REP": ({"BRAIN_D6_HEBBIAN_STORE": "1", "BRAIN_D6_HEBBIAN_FREEZE": "0"}, TEACH_USE),
    "SHUF_H":    ({"BRAIN_D6_HEBBIAN_STORE": "1", "BRAIN_D6_HEBBIAN_FREEZE": "0"}, TEACH_SHUF),
    "FREEZE_H":  ({"BRAIN_D6_HEBBIAN_STORE": "1", "BRAIN_D6_HEBBIAN_FREEZE": "1"}, TEACH_USE),
    "USE_D":     ({"BRAIN_D6_HEBBIAN_STORE": "0", "BRAIN_D6_HEBBIAN_FREEZE": "0"}, TEACH_USE),
}
SEEDS6 = [42, 43, 44, 100, 101, 102]

# VARIANT "engram" (pre-registered 2026-09-23 AFTER the v1 seed-42 smoke returned NO-GO on C3 and BEFORE any engram-
# variant result existed). v1 s42: C1,C2,C4-C7 held; C3 failed ONLY on the `answer` text -- the frozen arm abstained
# exactly like the shuffled control (abstained/recalled_svo identical) but the reply still treated the taught word as
# FAMILIAR (curiosity novelty 0.0 vs 0.97, grounded common-ground topic, thread swap), because the host `kb` list
# records every heard fact whether or not a synapse changed and `ChatBrain._refresh_facts` derives the known-word
# sets from it. The engram variant adds BRAIN_D6_ENGRAM_VOCAB=1 to every HEBBIAN arm (the known-fact/word sets are
# read off the engrams that reactivate on the substrate). USE_D stays the pure production path (all D6 flags 0), so
# C7 now also checks that the full D6 configuration is decision-identical to production on teach/d2/probe/xprobe.
# The GO gate is C1..C7 UNCHANGED (same thresholds, same fields, same 6 seeds). Arm dir: _d6_learn_through_use_engram.
VARIANTS = {"base": {}, "engram": {"BRAIN_D6_ENGRAM_VOCAB": "1"}}


def arms_for(variant):
    extra = VARIANTS[variant]
    out = {}
    for name, (env, teach) in ARMS.items():
        e = dict(env)
        if name != "USE_D":
            e.update(extra)
        else:
            e.update({k: "0" for k in extra})               # explicit OFF, never a pop
        out[name] = (e, teach)
    return out


# ── worker: ONE fresh brain, the session's turns through the real handler, + the taught block's learned |w| ───────
def _worker(env_json, teach, out_path):
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    for k, v in json.loads(env_json).items():
        os.environ[k] = v                                   # explicit values both directions (never a pop)
    from webapp import server as S
    from webapp.server import brain_chat, BrainChatRequest
    t0 = time.time()
    out = {"env": json.loads(env_json), "teach": teach, "seed": os.environ.get("BRAIN_CHAT_SEED"),
           "backend": os.environ.get("SIM_BACKEND"), "turns": {}, "taught_block": None}
    for i, (label, msg) in enumerate(TURNS):
        m = teach if label == "teach" else msg
        try:
            r = brain_chat(BrainChatRequest(session=SESSION, message=m, brain="tiny-demo", renderer="stub",
                                            rich=False, reset=(i == 0)))
            out["turns"][label] = json.loads(r.body)
        except Exception as e:
            out["turns"][label] = {"_error": "%s: %s" % (type(e).__name__, e)}
        if label == "teach":
            out["taught_block"] = _taught_block_record(S, teach)
    out["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2, default=str)
    print("[d6 worker] env=%s teach=%r -> %s (%.0fs)" % (out["env"], teach, out_path, out["elapsed_s"]), flush=True)
    return 0


def _taught_block_record(S, teach):
    """Read the just-taught fact's block off the composer: its index, learned mean |w| over the D trigger->readout
    synapses (the LEVER: ~1 plastic, 0 frozen), and the rule's own encode diag. Read-only."""
    try:
        chat = S._BRAIN_CHATS.get((SESSION, "tiny-demo", "stub"))
        comp = getattr(getattr(chat, "inner", None), "composer", None)
        agent = teach.split()[1]
        idx = None
        for j, (f, _h) in enumerate(getattr(comp, "kb", []) or []):
            if str(f.get("agent", "")).lower() == agent:
                idx = j
        if idx is None:
            return {"found": False, "agent": agent}
        D = comp.D
        ws = [complex(w) for (_p, _q, w) in comp.store_conns[idx * D:(idx + 1) * D]]
        mean_abs = sum(abs(w) for w in ws) / max(len(ws), 1)
        return {"found": True, "agent": agent, "block": idx, "mean_abs_w": round(mean_abs, 6),
                "d6_last_encode": getattr(comp, "_d6_last_encode", None), "n_kb": len(comp.kb)}
    except Exception as e:
        return {"found": False, "error": "%s: %s" % (type(e).__name__, e)}


def _spawn(env, teach, out_path, seed):
    penv = dict(os.environ); penv["BRAIN_CHAT_SEED"] = str(seed)
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners.d6_learn_through_use_lb", "--worker",
                        "--env", json.dumps(env), "--teach", teach, "--out", out_path], env=penv)
    if p.returncode != 0 or not os.path.exists(out_path):
        return None
    return json.load(open(out_path))


def _load_arm(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


# ── scoring ─────────────────────────────────────────────────────────────────────────────────────────────────────
def _dec(arm, label):
    t = ((arm or {}).get("turns") or {}).get(label) or {}
    if "_error" in t:
        return {"_error": t["_error"]}
    return {k: t.get(k) for k in DECISION_FIELDS}


def _recalls(arm, label, word):
    d = _dec(arm, label)
    svo = d.get("recalled_svo") or []
    return (d.get("abstained") is False) and (word in [str(w).lower() for w in svo])


def score_seed(arms):
    """Apply the pre-registered C1..C7 to one seed's arms dict {name: arm_json|None}. Returns the per-seed record."""
    from tools.lab import attributable_to, undefined_if_empty, void_if
    rec = {"criteria": {}, "void_arms": [], "go": None}
    for name in ARMS:
        a = arms.get(name)
        bad = a is None or any("_error" in (a.get("turns") or {}).get(lbl, {"_error": "missing"})
                               for lbl, _ in TURNS)
        if void_if(bad, "arm %s missing / errored" % name):
            rec["void_arms"].append(name)
    if rec["void_arms"]:
        rec["go"] = None
        rec["verdict"] = "UNDEFINED (void arms: %s)" % ",".join(rec["void_arms"])
        undefined_if_empty("d6 seed", 0, None, 7)
        return rec
    U, R, SH, F, DR = (arms[k] for k in ("USE_H", "USE_H_REP", "SHUF_H", "FREEZE_H", "USE_D"))
    c = rec["criteria"]
    c["C1_learns"] = _recalls(U, "probe", "deer")
    c["C2_use_changes_reply"] = (_dec(U, "probe") != _dec(SH, "probe")) and not _recalls(SH, "probe", "deer")
    c["C3_freeze_removes"] = (not _recalls(F, "probe", "deer")) and (_dec(F, "probe") == _dec(SH, "probe"))
    tbU, tbF = (U.get("taught_block") or {}), (F.get("taught_block") or {})
    wU, wF = tbU.get("mean_abs_w"), tbF.get("mean_abs_w")
    lever_moved = (wU is not None and wF is not None and wU > 0.5 and wF == 0.0)
    c["C4_write_only"] = ((_dec(F, "teach") == _dec(U, "teach")) and (_dec(F, "d2") == _dec(U, "d2"))
                          and lever_moved)
    c["C5_specific"] = _recalls(SH, "xprobe", "berry") and not _recalls(U, "xprobe", "berry")
    null_diffs = sum(_dec(U, lbl) != _dec(R, lbl) for lbl, _ in TURNS)
    c["C6_deterministic"] = (null_diffs == 0)
    c["C7_no_regression"] = all(_dec(DR, lbl) == _dec(U, lbl) for lbl in ("teach", "d2", "probe", "xprobe"))
    # attribution: treatment = the use->probe change (1 if USE vs SHUF probe differ); control = the null (rebuild).
    treat = 1.0 if _dec(U, "probe") != _dec(SH, "probe") else 0.0
    ctrl = 1.0 if _dec(U, "probe") != _dec(R, "probe") else 0.0
    rec["attributable_to_use"] = attributable_to("d6 use->probe change vs null rebuild", treat, ctrl)
    rec["lever"] = {"learned_mean_abs_w_plastic": wU, "learned_mean_abs_w_frozen": wF, "moved": lever_moved,
                    "encode_plastic": tbU.get("d6_last_encode"), "encode_frozen": tbF.get("d6_last_encode")}
    rec["null_diffs"] = null_diffs
    rec["probe_decisions"] = {k: _dec(arms[k], "probe") for k in ARMS}
    rec["xprobe_decisions"] = {k: _dec(arms[k], "xprobe") for k in ARMS}
    rec["go"] = all(c.values())
    rec["verdict"] = "GO" if rec["go"] else "NO-GO (failed: %s)" % ",".join(k for k, v in c.items() if not v)
    return rec


def aggregate(per_seed):
    defined = {s: r for s, r in per_seed.items() if r.get("go") is not None}
    n_go = sum(1 for r in defined.values() if r["go"])
    agg = {"n_seeds": len(per_seed), "n_defined": len(defined), "n_go": n_go,
           "undefined_seeds": [s for s, r in per_seed.items() if r.get("go") is None]}
    agg["GO"] = (len(per_seed) == 6 and len(defined) == 6 and n_go == 6)
    agg["verdict"] = ("GO 6/6" if agg["GO"] else
                      ("UNDEFINED (%d/%d seeds defined)" % (len(defined), len(per_seed)) if len(defined) < len(per_seed)
                       else "NO-GO %d/%d" % (n_go, len(defined))))
    return agg


def _arm_path(arm_dir, seed, name):
    return os.path.join(arm_dir, "s%d_%s.json" % (int(seed), name))


def run(seeds, arm_dir, resume=True, score_only=False, variant="base"):
    per = {}
    for s in seeds:
        arms = {}
        for name, (env, teach) in arms_for(variant).items():
            path = _arm_path(arm_dir, s, name)
            a = _load_arm(path) if (resume or score_only) and os.path.exists(path) else None
            if a is None and not score_only:
                print("[d6] seed %s arm %s ..." % (s, name), flush=True)
                a = _spawn(env, teach, path, s)
            arms[name] = a
        per[str(s)] = score_seed(arms)
        print("[d6] seed %s -> %s" % (s, per[str(s)]["verdict"]), flush=True)
    return {"runner": "research.runners.d6_learn_through_use_lb", "variant": variant, "seeds": list(seeds), "arm_dir": arm_dir,
            "per_seed": per, "aggregate": aggregate(per)}


# ── self-test: the verdict must FAIL in each failing direction (a gate that cannot fail measures nothing) ────────
def _synthetic(probe_use="deer", probe_frozen=None, rep_same=True, shuf_berry=True, w_frozen=0.0, direct_same=True):
    def arm(teach_word, probe_word, xprobe_word, w, d2="cat"):
        def t(word, ab=None):
            if word is None:
                return {"abstained": True, "recalled_svo": None, "answer": "I don't know."}
            return {"abstained": False, "recalled_svo": ["x", "y", word], "answer": "x y %s." % word}
        return {"turns": {"teach": {"abstained": False, "recalled_svo": ["a", "b", teach_word], "answer": "Got it."},
                          "d1": t("fish"), "d2": t(d2), "probe": t(probe_word), "xprobe": t(xprobe_word)},
                "taught_block": {"mean_abs_w": w}}
    U = arm("deer", probe_use, None, 1.0)
    return {"USE_H": U,
            "USE_H_REP": U if rep_same else arm("deer", None, None, 1.0),
            "SHUF_H": arm("berry", None, "berry" if shuf_berry else None, 1.0),
            "FREEZE_H": arm("deer", probe_frozen, None, w_frozen),
            "USE_D": U if direct_same else arm("deer", None, None, 1.0)}


def selftest():
    ok = score_seed(_synthetic())["go"] is True
    fails = {
        "no_learning": score_seed(_synthetic(probe_use=None))["go"] is False,
        "freeze_does_not_remove": score_seed(_synthetic(probe_frozen="deer"))["go"] is False,
        "lever_did_not_move": score_seed(_synthetic(w_frozen=1.0))["go"] is False,
        "null_dirty": score_seed(_synthetic(rep_same=False))["go"] is False,
        "not_specific": score_seed(_synthetic(shuf_berry=False))["go"] is False,
        "direct_regression": score_seed(_synthetic(direct_same=False))["go"] is False,
    }
    void = score_seed(dict(_synthetic(), FREEZE_H=None))
    fails["void_is_undefined"] = void["go"] is None
    agg_undef = aggregate({"42": {"go": True}, "43": {"go": None}})["GO"] is False
    res = {"go_case_passes": ok, "fails_in_failing_direction": fails, "partial_seed_set_not_go": agg_undef}
    passed = ok and all(fails.values()) and agg_undef
    print(json.dumps(res, indent=2))
    print("SELFTEST", "PASS" if passed else "FAIL")
    return 0 if passed else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--env", default="{}")
    ap.add_argument("--teach", default=TEACH_USE)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--arm-dir", default="research/findings/raw/_d6_learn_through_use")
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="base",
                    help="base = the v1 pre-registered arms; engram = + BRAIN_D6_ENGRAM_VOCAB=1 on the Hebbian arms")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.worker:
        return _worker(a.env, a.teach, a.out)
    res = run(a.seeds, a.arm_dir, resume=not a.no_resume, score_only=a.score_only, variant=a.variant)
    print(json.dumps(res["aggregate"], indent=2))
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(res, open(a.json, "w"), indent=2, default=str)
        print("[d6] wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
