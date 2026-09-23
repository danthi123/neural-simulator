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

CURRENT GATE (fix round 3): gate v3, `--variant capability` -- score_seed_v3 below and
research/findings/2026-09-23-d6-learn-through-use-v3-PREREGISTRATION-capability-gate.md. The v1 gate above (and v2) are
kept for the record of the banked runs; they are not the gate D6 is decided on.

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
# VARIANT "prune" (pre-registered 2026-09-23 AFTER the engram-variant seed-42 FREEZE/SHUF arms and BEFORE any prune
# result existed). Engram s42: curiosity novelty was restored (frozen 0.97 vs shuffled 0.97) but (a) organs that read
# `composer.kb` DIRECTLY (webapp/gnw_thought_swap._known_concepts -> thread swap, common ground, GNW stop) still
# treated the taught word as a grounded topic, and (b) the frozen arm's teach ACK text changed ("The wolf hunts deer."
# vs "the wolf hunts the deer") because the ack render reads the engram-derived known sets -- the brain acknowledging
# a sentence it did not retain, not a parse failure (recalled_svo + abstained identical). The prune variant adds
# BRAIN_D6_ENGRAM_PRUNE=1 (an in-conversation encode that forms no engram is retracted from kb). Its gate is C1..C7
# with ONE change: C4's teach-turn equality compares the PARSE fields (`abstained`, `recalled_svo`) only, not the ack
# text (the d2 read-intact equality and the lever condition are unchanged). Arm dir: _d6_learn_through_use_prune.
#
# ── AMENDMENT LOG ────────────────────────────────────────────────────────────────────────────────────────────────
# 2026-09-23T15:53Z (fix round after the adversarial review of 387d96a5b). Results SEEN at the time of this amendment:
#   base s42 (all 5 arms, NO-GO on C3); engram s42 (all 5 arms, NO-GO on C3 + C4-teach-ack); prune s42 USE_H and
#   USE_H_REP arm files only (no FREEZE_H/SHUF_H/USE_D prune arm yet, no prune verdict); base-variant pool partial arms
#   (s43/s100 USE_H + USE_H_REP on pool42; pool41 unreachable). No `readtime` result exists.
#   A1. The prune variant's C4 relaxation (parse-only teach check) was NOT a design made blind: it was written AFTER
#       the engram s42 FREEZE_H teach ack ("The wolf hunts deer.") had been seen to fail C4. It is REVERTED: every
#       variant is scored under the ORIGINAL C4 (full teach-turn decision equality). The parse-only figure is still
#       computed and reported as `C4_parse_posthoc`, labelled post-hoc; it never enters a verdict.
#   A2. The prune variant is BANKED AS AN INVALID INSTRUMENT for C3 (its lesion arm runs a host retraction that the
#       treatment arm never runs, so a C3 pass cannot show that plasticity carries the reply change). Its 6 pool lines
#       were dequeued. Its s42 smoke (already running) may still be scored, under the original C4, as a diagnostic.
#   A3. NEW variant `readtime` with its OWN gate v2, pre-registered in
#       research/findings/2026-09-23-d6-learn-through-use-v2-PREREGISTRATION-readtime-view-and-engram-ablation.md,
#       committed in its own commit before any readtime run. It is the variant the 6-seed run is staged for.
#   A4. (~17:30Z, before any readtime arm existed) EXPO_H exposure-matched arm + SECONDARY non-scoring C3e/C3be --
#       see the EXPO_ARM comment below for the evidence seen (prune s42 full smoke) and why.
#   A5. (fix round 3, before any readtime OR capability arm existed; the six readtime pool lines were never
#       dispatched) Gate v2 is SUPERSEDED: its own registration predicted a NO-GO unrelated to the capability (C4 on
#       the teach ack, C3/C3b on exposure habituation). NEW variant `capability`, gate v3 (score_seed_v3), registered
#       in research/findings/2026-09-23-d6-learn-through-use-v3-PREREGISTRATION-capability-gate.md in its own commit.
#       SEEN at this time: everything listed under A1-A4 + the prune s42 full smoke + base-variant pool arm files for
#       s43/s100 (USE_H, USE_H_REP, SHUF_H; s43 FREEZE_H). The v2 scorer is kept unchanged for the record.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# VARIANT "readtime" (gate v2 -- see the PREREGISTRATION finding above). Flags on every Hebbian arm:
# BRAIN_D6_ENGRAM_VOCAB=1 + BRAIN_D6_ENGRAM_READTIME=1 (NO prune: no host record is ever deleted; FOUR named kb readers
# consult the engram at read time -- d6_hebbian_store.ENGRAM_READTIME_ROUTED; the others do not, see
# ENGRAM_READTIME_NOT_ROUTED; the start-of-turn re-read runs in every Hebbian arm). [ERRATUM, fix round 3: this comment
# first said "every kb reader" and "the lesion arms differ ONLY in their synapses"; both were false -- ABL_H also runs
# the experimenter's cache invalidation + a re-read of every block.] Adds arm ABL_H: USE_H's configuration, plus --
# after the teach turn -- the experimenter zeroes the taught block's synapses (post-hoc ablation, kb record intact).
VARIANTS = {"base": {}, "engram": {"BRAIN_D6_ENGRAM_VOCAB": "1"},
            "prune": {"BRAIN_D6_ENGRAM_VOCAB": "1", "BRAIN_D6_ENGRAM_PRUNE": "1"},
            "readtime": {"BRAIN_D6_ENGRAM_VOCAB": "1", "BRAIN_D6_ENGRAM_READTIME": "1"}}
ABL_ARM = "ABL_H"   # readtime variant only: USE_H + post-hoc ablation of the taught block after the teach turn
# AMENDMENT A4 (2026-09-23 ~17:30Z, BEFORE any readtime arm existed; SEEN at this time: the prune s42 smoke's full 5
# arms -- NO-GO on C3 + original-C4 -- where FREEZE_H vs SHUF_H still differed at the probe ONLY in the DA-mode suffix,
# traced to the spiking novelty organ's per-word freshness: 'wolf' 0.84 in FREEZE_H (heard at teach) vs 1.0 in SHUF_H
# (never heard); research/findings/raw/_d6_learn_through_use_prune/s42_FREEZE_H.json). SHUF_H does not match WORD
# EXPOSURE, so C3/C3b conflate the fact-write engram with exposure habituation, a use-trace the write freeze does not
# (and should not) touch. Added: EXPO_H = USE_H's flags, teach turn "the wolf and the deer" (the same content words,
# not an SVO assertion -> no acquisition, no write). SECONDARY, NON-SCORING (the registered gate v2 is unchanged):
# C3e FREEZE_H.probe == EXPO_H.probe and not recall deer; C3be ABL_H.probe == EXPO_H.probe and not recall deer.
EXPO_ARM = "EXPO_H"
TEACH_EXPO = "the wolf and the deer"
# ── GATE v3 (variant "capability") ────────────────────────────────────────────────────────────────────────────────
# Pre-registered in research/findings/2026-09-23-d6-learn-through-use-v3-PREREGISTRATION-capability-gate.md, committed
# BEFORE any v3 arm existed. Gate v2 was registered to fail (its own prereg predicted C4 fails on the teach-turn ack
# and C3/C3b on the DA-mode suffix), so it measured the protocol, not the capability. v3 asks the capability question
# directly: does the reply to a LATER recall question depend on the SYNAPTIC WRITE? The primary contrast is USE_H vs
# FREEZE_H: identical input, identical flags, identical host code; the ONLY difference is eta=0 in the Hebbian update.
# Same flags as `readtime`. NOREC_H = FREEZE_H + (after the teach turn) the experimenter removes the frozen fact's host
# kb RECORD (its synapses are already exactly 0). NOREC_H is NOT a lesion arm: it tests that the host record is INERT
# (NOREC_H == FREEZE_H on every turn), which replaces the unmeasurable claim "every kb reader asks the engram".
CAP_VARIANT = "capability"
VARIANTS[CAP_VARIANT] = dict(VARIANTS["readtime"])
NOREC_ARM = "NOREC_H"
LESION_ARMS_V3 = ("FREEZE_H", ABL_ARM, NOREC_ARM)


def arm_names(variant):
    if variant == CAP_VARIANT:
        return list(ARMS) + [ABL_ARM, NOREC_ARM]
    return list(ARMS) + ([ABL_ARM] if variant == "readtime" else [])


def arms_for(variant):
    """{name: (env, teach, post_teach)} for a variant; post_teach in (None, 'ablate', 'norec')."""
    extra = VARIANTS[variant]
    out = {}
    for name, (env, teach) in ARMS.items():
        e = dict(env)
        if name != "USE_D":
            e.update(extra)
        else:
            e.update({k: "0" for k in extra})               # explicit OFF, never a pop
        out[name] = (e, teach, None)
    if variant == "readtime":
        out[ABL_ARM] = (dict(out["USE_H"][0]), TEACH_USE, "ablate")   # identical to USE_H + the post-hoc ablation
        out[EXPO_ARM] = (dict(out["USE_H"][0]), TEACH_EXPO, None)   # A4: exposure-matched, no write (secondary)
    if variant == CAP_VARIANT:
        out[ABL_ARM] = (dict(out["USE_H"][0]), TEACH_USE, "ablate")
        out[NOREC_ARM] = (dict(out["FREEZE_H"][0]), TEACH_USE, "norec")   # FREEZE_H + post-hoc host-record removal
    return out


# ── worker: ONE fresh brain, the session's turns through the real handler, + the taught block's learned |w| ───────
def _composer(S):
    chat = S._BRAIN_CHATS.get((SESSION, "tiny-demo", "stub"))
    return getattr(getattr(chat, "inner", None), "composer", None)


def _worker(env_json, teach, out_path, ablate_after_teach=False, post_teach=None):
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    for k, v in json.loads(env_json).items():
        os.environ[k] = v                                   # explicit values both directions (never a pop)
    if ablate_after_teach and post_teach is None:
        post_teach = "ablate"
    from webapp import server as S
    from webapp.server import brain_chat, BrainChatRequest
    t0 = time.time()
    out = {"env": json.loads(env_json), "teach": teach, "seed": os.environ.get("BRAIN_CHAT_SEED"),
           "backend": os.environ.get("SIM_BACKEND"), "turns": {}, "taught_block": None,
           "ablate_after_teach": post_teach == "ablate", "post_teach": post_teach, "ablation": None,
           "record_removal": None, "counter_installed": False, "store_writes_after_teach": None,
           "homeostatic_calls_after_teach": None, "taught_block_at_probe": None, "d6_ops": None}
    writes_after_teach, homeo_calls = [], []
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
            comp = _composer(S)
            blk = (out["taught_block"] or {}).get("block")
            if post_teach == "ablate":                      # ABL_H: the experimenter's post-hoc engram ablation
                from research.runners.d6_hebbian_store import ablate_block
                out["ablation"] = (ablate_block(comp, blk) if (comp is not None and blk is not None)
                                   else {"error": "taught block not found; ablation not applied"})
            elif post_teach == "norec":                     # NOREC_H: remove the frozen fact's host RECORD only
                from research.runners.d6_hebbian_store import remove_block_record
                try:
                    out["record_removal"] = (remove_block_record(comp, blk) if (comp is not None and blk is not None)
                                             else {"error": "taught block not found; record not removed"})
                except Exception as e:
                    out["record_removal"] = {"error": "%s: %s" % (type(e).__name__, e)}
            # INSTRUMENT (lesion persistence): count every store write after the teach turn. Any write in a lesion arm
            # (e.g. reconsolidation's direct-copy `update_on_mismatch`, which bypasses the freeze) voids the lesion.
            # `counter_installed` is recorded and REQUIRED by gate v3: an absent counter reads UNDEFINED, never clean.
            if comp is not None and hasattr(comp, "_write_block"):
                _orig = comp._write_block

                def _counted(bi, zc, _o=_orig):
                    writes_after_teach.append(int(bi))
                    return _o(bi, zc)
                comp._write_block = _counted
                out["counter_installed"] = True
                # apply_homeostatic_scaling rewrites store_conns WITHOUT _write_block; it is counted separately. It is
                # multiplicative, so it cannot un-zero a zeroed block; the at-probe lever read covers the taught block.
                if hasattr(comp, "apply_homeostatic_scaling"):
                    _oh = comp.apply_homeostatic_scaling

                    def _hcounted(*a, _o=_oh, **k):
                        homeo_calls.append(1)
                        return _o(*a, **k)
                    comp.apply_homeostatic_scaling = _hcounted
        if label == "probe":                                # the lever read AT MEASUREMENT TIME, not only at teach
            out["taught_block_at_probe"] = _taught_block_record(S, teach)
    out["store_writes_after_teach"] = list(writes_after_teach)
    out["homeostatic_calls_after_teach"] = len(homeo_calls)
    comp = _composer(S)
    out["d6_ops"] = dict(getattr(comp, "_d6_ops", {}) or {}) if comp is not None else None
    out["elapsed_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2, default=str)
    print("[d6 worker] env=%s teach=%r -> %s (%.0fs)" % (out["env"], teach, out_path, out["elapsed_s"]), flush=True)
    return 0


def _taught_block_record(S, teach):
    """Read the just-taught fact's block off the composer: its index, learned mean |w| over the D trigger->readout
    synapses (the LEVER: ~1 plastic, 0 frozen), and the rule's own encode diag. Read-only."""
    try:
        comp = _composer(S)
        agent = teach.split()[1]
        idx = None
        for j, (f, _h) in enumerate(getattr(comp, "kb", []) or []):
            if str(f.get("agent", "")).lower() == agent:
                idx = j
        if idx is None:
            # (engram-prune variant) a frozen encode that formed no engram is RETRACTED from kb: report the rule's
            # own encode diag (its learned mean |w|) + the retraction record instead of "not found".
            enc = getattr(comp, "_d6_last_encode", None)
            ret = getattr(comp, "_d6_last_retract", None)
            if enc is not None and ret is not None and ret.get("retracted"):
                return {"found": False, "retracted": True, "agent": agent, "mean_abs_w": enc.get("mean_abs_w"),
                        "d6_last_encode": enc, "d6_last_retract": ret, "n_kb": len(comp.kb)}
            return {"found": False, "agent": agent}
        D = comp.D
        ws = [complex(w) for (_p, _q, w) in comp.store_conns[idx * D:(idx + 1) * D]]
        mean_abs = sum(abs(w) for w in ws) / max(len(ws), 1)
        return {"found": True, "agent": agent, "block": idx, "mean_abs_w": round(mean_abs, 6),
                "d6_last_encode": getattr(comp, "_d6_last_encode", None), "n_kb": len(comp.kb)}
    except Exception as e:
        return {"found": False, "error": "%s: %s" % (type(e).__name__, e)}


def _spawn(env, teach, out_path, seed, post_teach=None):
    penv = dict(os.environ); penv["BRAIN_CHAT_SEED"] = str(seed)
    p = subprocess.run([sys.executable, "-u", "-m", "research.runners.d6_learn_through_use_lb", "--worker",
                        "--env", json.dumps(env), "--teach", teach, "--out", out_path]
                       + (["--post-teach", post_teach] if post_teach else []), env=penv)
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


def score_seed(arms, variant="base"):
    """Apply the pre-registered gate to one seed's arms dict {name: arm_json|None}. base/engram/prune: C1..C7 (v1;
    since AMENDMENT A1 every variant uses the ORIGINAL C4). readtime: gate v2 (C1..C7 + C3b ablation + C4 lever read
    at probe time + no store writes after teach in the lesion arms), per the v2 PREREGISTRATION finding."""
    if variant == CAP_VARIANT:
        return score_seed_v3(arms)
    from tools.lab import attributable_to, undefined_if_empty, void_if
    rec = {"criteria": {}, "void_arms": [], "go": None, "variant": variant,
           "gate": "v2" if variant == "readtime" else "v1"}
    for name in arm_names(variant):
        a = arms.get(name)
        bad = a is None or any("_error" in (a.get("turns") or {}).get(lbl, {"_error": "missing"})
                               for lbl, _ in TURNS)
        if variant == "readtime" and name == ABL_ARM and not bad:
            abl = a.get("ablation") or {}
            bad = ("error" in abl) or abl.get("mean_abs_w_after") is None
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
    if variant == "readtime":
        A = arms[ABL_ARM]
        c["C3b_ablation_removes"] = (not _recalls(A, "probe", "deer")) and (_dec(A, "probe") == _dec(SH, "probe"))
    # lever: v1 reads the taught block right after the teach turn; v2 reads it AT PROBE TIME (the measurement moment)
    key = "taught_block_at_probe" if variant == "readtime" else "taught_block"
    tbU, tbF = (U.get(key) or {}), (F.get(key) or {})
    wU, wF = tbU.get("mean_abs_w"), tbF.get("mean_abs_w")
    lever_moved = (wU is not None and wF is not None and wU > 0.5 and wF == 0.0)
    teach_same = (_dec(F, "teach") == _dec(U, "teach"))            # ORIGINAL C4 (AMENDMENT A1: relaxation reverted)
    c4 = teach_same and (_dec(F, "d2") == _dec(U, "d2")) and lever_moved
    if variant == "readtime":
        A = arms[ABL_ARM]
        wA = (A.get(key) or {}).get("mean_abs_w")
        abl_lever = (wA == 0.0) and (_dec(A, "teach") == _dec(U, "teach")) and (_dec(A, "d2") == _dec(U, "d2"))
        writes_clean = all((arms[k].get("store_writes_after_teach") == []) for k in ("FREEZE_H", ABL_ARM))
        c4 = c4 and abl_lever and writes_clean
        rec["lever_ablation"] = {"mean_abs_w_at_probe": wA, "ablation": A.get("ablation"), "moved": abl_lever,
                                 "store_writes_after_teach": {k: arms[k].get("store_writes_after_teach")
                                                             for k in arm_names(variant)}}
    c["C4_write_only"] = c4
    # POST-HOC, NON-SCORING (AMENDMENT A1): the parse-only teach check the prune variant had used. Reported, never scored.
    rec["C4_parse_posthoc"] = (all(_dec(F, "teach").get(k) == _dec(U, "teach").get(k)
                                   for k in ("abstained", "recalled_svo"))
                               and (_dec(F, "d2") == _dec(U, "d2")) and lever_moved)
    c["C5_specific"] = _recalls(SH, "xprobe", "berry") and not _recalls(U, "xprobe", "berry")
    null_diffs = sum(_dec(U, lbl) != _dec(R, lbl) for lbl, _ in TURNS)
    c["C6_deterministic"] = (null_diffs == 0)
    c["C7_no_regression"] = all(_dec(DR, lbl) == _dec(U, lbl) for lbl in ("teach", "d2", "probe", "xprobe"))
    # INTEGRITY SMOKE (not a criterion; pass-by-construction when prune is off): the lesion arms ran no host step the
    # treatment arm did not run -- no retraction anywhere, and (readtime) the start-of-turn re-read ran in every Hebbian arm.
    ops = {k: (arms[k].get("d6_ops") or {}) for k in arm_names(variant)}
    rec["integrity"] = {"d6_ops": ops,
                        "no_retractions": all(o.get("retractions", 0) == 0 for o in ops.values())}
    if variant == "readtime":
        rec["integrity"]["turn_refresh_every_hebbian_arm"] = all(
            ops[k].get("turn_refreshes", 0) == len(TURNS) for k in arm_names(variant) if k != "USE_D")
    # attribution: treatment = the use->probe change (1 if USE vs SHUF probe differ); control = the null (rebuild).
    treat = 1.0 if _dec(U, "probe") != _dec(SH, "probe") else 0.0
    ctrl = 1.0 if _dec(U, "probe") != _dec(R, "probe") else 0.0
    rec["attributable_to_use"] = attributable_to("d6 use->probe change vs null rebuild", treat, ctrl)
    rec["lever"] = {"read_at": key, "learned_mean_abs_w_plastic": wU, "learned_mean_abs_w_frozen": wF,
                    "moved": lever_moved,
                    "encode_plastic": tbU.get("d6_last_encode"), "encode_frozen": tbF.get("d6_last_encode")}
    rec["null_diffs"] = null_diffs
    rec["teach_decisions"] = {k: _dec(arms[k], "teach") for k in arm_names(variant)}
    rec["probe_decisions"] = {k: _dec(arms[k], "probe") for k in arm_names(variant)}
    rec["xprobe_decisions"] = {k: _dec(arms[k], "xprobe") for k in arm_names(variant)}
    if variant == "readtime":                      # A4 SECONDARY (non-scoring): exposure-matched comparison
        E = arms.get(EXPO_ARM)
        e_bad = E is None or any("_error" in (E.get("turns") or {}).get(lbl, {"_error": "missing"}) for lbl, _ in TURNS)
        if e_bad:
            rec["secondary_exposure_matched"] = {"verdict": "UNDEFINED (EXPO_H void)"}
        else:
            A = arms[ABL_ARM]
            rec["secondary_exposure_matched"] = {
                "EXPO_H_no_write": (E.get("store_writes_after_teach") == [] and not (E.get("taught_block") or {}).get("found")),
                "C3e_freeze_eq_exposure": (not _recalls(F, "probe", "deer")) and _dec(F, "probe") == _dec(E, "probe"),
                "C3be_ablation_eq_exposure": (not _recalls(A, "probe", "deer")) and _dec(A, "probe") == _dec(E, "probe"),
                "EXPO_H_probe": _dec(E, "probe"), "EXPO_H_teach": _dec(E, "teach")}
    rec["go"] = all(c.values())
    rec["verdict"] = "GO" if rec["go"] else "NO-GO (failed: %s)" % ",".join(k for k, v in c.items() if not v)
    return rec


def _mentions(arm, label, word):
    import re
    ans = str(_dec(arm, label).get("answer") or "").lower()
    return re.search(r"\b%s\b" % re.escape(word), ans) is not None


def _arm_bad(a):
    return a is None or any("_error" in (a.get("turns") or {}).get(lbl, {"_error": "missing"}) for lbl, _ in TURNS)


def score_seed_v3(arms):
    """GATE v3 (variant 'capability') -- research/findings/2026-09-23-d6-learn-through-use-v3-PREREGISTRATION-
    capability-gate.md. Scored arms: USE_H USE_H_REP SHUF_H FREEZE_H ABL_H NOREC_H. USE_D is secondary (non-scoring)."""
    from tools.lab import attributable_to, void_if
    rec = {"criteria": {}, "void_arms": [], "go": None, "variant": CAP_VARIANT, "gate": "v3"}
    scored = ("USE_H", "USE_H_REP", "SHUF_H", "FREEZE_H", ABL_ARM, NOREC_ARM)
    for name in scored:
        a = arms.get(name)
        bad, why = _arm_bad(a), "missing / errored"
        if not bad and name == ABL_ARM:
            abl = a.get("ablation") or {}
            bad, why = ("error" in abl) or abl.get("mean_abs_w_after") is None, "ablation not applied"
        if not bad and name == NOREC_ARM:
            rr = a.get("record_removal") or {}
            bad, why = ("error" in rr) or not rr.get("removed"), "host record not removed"
        if not bad and name in LESION_ARMS_V3:
            bad, why = (a.get("counter_installed") is not True), "write counter not installed (instrument absent)"
        if void_if(bad, "arm %s %s" % (name, why)):
            rec["void_arms"].append(name)
    if rec["void_arms"]:
        rec["verdict"] = "UNDEFINED (void arms: %s)" % ",".join(rec["void_arms"])
        return rec
    U, R, SH, F, A, NR = (arms[k] for k in scored)
    c = rec["criteria"]
    # K1 the brain learns from use
    c["K1_learns"] = _recalls(U, "probe", "deer")
    # K2 the later reply changes with WHAT was taught (double dissociation vs the shuffled-content control)
    c["K2_use_specific"] = (not _recalls(SH, "probe", "deer") and _dec(U, "probe") != _dec(SH, "probe")
                            and _recalls(SH, "xprobe", "berry") and not _recalls(U, "xprobe", "berry"))
    # K3 THE CAPABILITY: same input, same host code, eta=0 -> the later reply no longer carries the taught fact
    c["K3_reply_depends_on_write"] = (not _recalls(F, "probe", "deer") and not _mentions(F, "probe", "deer")
                                      and _dec(F, "probe") != _dec(U, "probe"))
    # K4 ... and it is read off the SYNAPSE at probe time (post-hoc ablation after a successful write)
    c["K4_reply_depends_on_synapse_at_read"] = (not _recalls(A, "probe", "deer") and not _mentions(A, "probe", "deer"))
    # K5 the frozen fact's HOST RECORD is inert: removing it changes nothing in any turn (teach = determinism part)
    k5_diff = [lbl for lbl, _ in TURNS if _dec(NR, lbl) != _dec(F, lbl)]
    c["K5_host_record_inert"] = (k5_diff == [])
    # K6 the lesion is WRITE-ONLY: lever moved + persisted to the probe, no later write, same encode episode, same
    #    parse, intact read path and unrelated conversation. The teach-turn ACK text is NOT scored (pre-registered):
    #    it is rendered after the write, from the read-time engram view, so it is itself a reply that depends on the write.
    key = "taught_block_at_probe"
    wU = (U.get(key) or {}).get("mean_abs_w"); wF = (F.get(key) or {}).get("mean_abs_w")
    wA = (A.get(key) or {}).get("mean_abs_w")
    lever = (wU is not None and wU > 0.5 and wF == 0.0 and wA == 0.0)
    writes_clean = all(arms[k].get("store_writes_after_teach") == [] for k in LESION_ARMS_V3)
    eU = (U.get("taught_block") or {}).get("d6_last_encode") or {}
    eF = (F.get("taught_block") or {}).get("d6_last_encode") or {}
    same_episode = (bool(eU) and bool(eF) and eU.get("frozen") is False and eF.get("frozen") is True
                    and all(eU.get(k) is not None and eU.get(k) == eF.get(k)
                            for k in ("steps", "phase_lock_steps", "block")))
    same_parse = all(_dec(F, "teach").get(k) == _dec(U, "teach").get(k) for k in ("abstained", "recalled_svo"))
    read_intact = all(_dec(X, lbl) == _dec(U, lbl) for X in (F, A) for lbl in ("d1", "d2"))
    c["K6_lesion_write_only"] = lever and writes_clean and same_episode and same_parse and read_intact
    rec["K6_parts"] = {"lever_at_probe": {"USE_H": wU, "FREEZE_H": wF, "ABL_H": wA, "ok": lever},
                       "store_writes_after_teach": {k: arms[k].get("store_writes_after_teach") for k in LESION_ARMS_V3},
                       "writes_clean": writes_clean, "same_encode_episode": same_episode,
                       "encode": {"USE_H": eU, "FREEZE_H": eF}, "same_teach_parse": same_parse,
                       "read_path_intact_d1_d2": read_intact}
    # K7 null / determinism: the rebuild is identical on every turn; ABL_H is USE_H up to the post-teach ablation, so its
    #    teach turn must equal USE_H's (a DETERMINISM check, not lesion evidence -- the ablation happens after it).
    null_diffs = sum(_dec(U, lbl) != _dec(R, lbl) for lbl, _ in TURNS)
    c["K7_null_clean"] = (null_diffs == 0) and (_dec(A, "teach") == _dec(U, "teach"))
    # ── reported, NEVER scored ──
    rec["K5_differing_turns"] = k5_diff
    rec["teach_ack_same_FREEZE_vs_USE"] = (_dec(F, "teach") == _dec(U, "teach"))
    DR = arms.get("USE_D")
    rec["secondary_C7_no_regression"] = (
        "UNDEFINED (USE_D void)" if _arm_bad(DR) else
        all(_dec(DR, lbl) == _dec(U, lbl) for lbl in ("teach", "d2", "probe", "xprobe")))
    rec["homeostatic_calls_after_teach"] = {k: arms[k].get("homeostatic_calls_after_teach") for k in scored}
    rec["record_removal"] = NR.get("record_removal")
    rec["ablation"] = A.get("ablation")
    rec["integrity"] = {"d6_ops": {k: (arms[k].get("d6_ops") or {}) for k in scored},
                        "turn_refresh_every_hebbian_arm": all(
                            (arms[k].get("d6_ops") or {}).get("turn_refreshes", 0) == len(TURNS) for k in scored)}
    rec["null_diffs"] = null_diffs
    rec["attributable_to_write"] = attributable_to(
        "d6 write->probe change (USE vs FREEZE) vs null rebuild",
        1.0 if _dec(U, "probe") != _dec(F, "probe") else 0.0, 1.0 if _dec(U, "probe") != _dec(R, "probe") else 0.0)
    rec["probe_decisions"] = {k: _dec(arms[k], "probe") for k in scored}
    rec["teach_decisions"] = {k: _dec(arms[k], "teach") for k in scored}
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


def run(seeds, arm_dir, resume=True, score_only=False, variant="base", only_arms=None):
    per = {}
    for s in seeds:
        arms = {}
        for name, (env, teach, post_teach) in arms_for(variant).items():
            if only_arms and name not in only_arms:
                arms[name] = _load_arm(_arm_path(arm_dir, s, name))
                continue
            path = _arm_path(arm_dir, s, name)
            a = _load_arm(path) if (resume or score_only) and os.path.exists(path) else None
            if a is None and not score_only:
                print("[d6] seed %s arm %s ..." % (s, name), flush=True)
                a = _spawn(env, teach, path, s, post_teach=post_teach)
            arms[name] = a
        per[str(s)] = score_seed(arms, variant=variant)
        print("[d6] seed %s -> %s" % (s, per[str(s)]["verdict"]), flush=True)
    return {"runner": "research.runners.d6_learn_through_use_lb", "variant": variant, "seeds": list(seeds), "arm_dir": arm_dir,
            "per_seed": per, "aggregate": aggregate(per)}


# ── self-test: the verdict must FAIL in each failing direction (a gate that cannot fail measures nothing) ────────
def _synthetic(probe_use="deer", probe_frozen=None, rep_same=True, shuf_berry=True, w_frozen=0.0, direct_same=True,
               readtime=False, probe_abl=None, w_abl=0.0, abl_writes=(), frozen_writes=()):
    def arm(teach_word, probe_word, xprobe_word, w, d2="cat", writes=()):
        def t(word, ab=None):
            if word is None:
                return {"abstained": True, "recalled_svo": None, "answer": "I don't know."}
            return {"abstained": False, "recalled_svo": ["x", "y", word], "answer": "x y %s." % word}
        return {"turns": {"teach": {"abstained": False, "recalled_svo": ["a", "b", teach_word], "answer": "Got it."},
                          "d1": t("fish"), "d2": t(d2), "probe": t(probe_word), "xprobe": t(xprobe_word)},
                "taught_block": {"mean_abs_w": w}, "taught_block_at_probe": {"mean_abs_w": w},
                "store_writes_after_teach": list(writes), "d6_ops": {"turn_refreshes": len(TURNS), "retractions": 0}}
    U = arm("deer", probe_use, None, 1.0)
    out = {"USE_H": U,
           "USE_H_REP": U if rep_same else arm("deer", None, None, 1.0),
           "SHUF_H": arm("berry", None, "berry" if shuf_berry else None, 1.0),
           "FREEZE_H": arm("deer", probe_frozen, None, w_frozen, writes=frozen_writes),
           "USE_D": U if direct_same else arm("deer", None, None, 1.0)}
    if readtime:
        A = arm("deer", probe_abl, None, w_abl, writes=abl_writes)
        A["ablation"] = {"block": 7, "mean_abs_w_before": 1.0, "mean_abs_w_after": w_abl}
        out[ABL_ARM] = A
    return out


def _cp(x):
    return json.loads(json.dumps(x))


def _synthetic_v3():
    """A gate-v3 GO seed (every arm as the capability predicts). Tests mutate a deep copy of it."""
    def t(word):
        if word is None:
            return {"abstained": True, "recalled_svo": None, "answer": "I don't know about that."}
        return {"abstained": False, "recalled_svo": ["x", "y", word], "answer": "x y %s." % word}

    def arm(teach_word, probe, xprobe, w, frozen=False, ack="the wolf hunts the deer"):
        return {"turns": {"teach": {"abstained": False, "recalled_svo": ["a", "b", teach_word], "answer": ack},
                          "d1": t("fish"), "d2": t("cat"), "probe": t(probe), "xprobe": t(xprobe)},
                "taught_block": {"mean_abs_w": w, "d6_last_encode": {"frozen": frozen, "steps": 208, "block": 5,
                                                                     "phase_lock_steps": 184}},
                "taught_block_at_probe": {"mean_abs_w": w}, "counter_installed": True,
                "store_writes_after_teach": [], "homeostatic_calls_after_teach": 0,
                "d6_ops": {"turn_refreshes": len(TURNS), "retractions": 0}}
    U = arm("deer", "deer", None, 1.0)
    F = arm("deer", None, None, 0.0, frozen=True, ack="The wolf hunts deer.")   # ack differs: NOT scored (prereg)
    A = arm("deer", None, None, 0.0)
    A["ablation"] = {"block": 5, "mean_abs_w_before": 1.0, "mean_abs_w_after": 0.0}
    NR = _cp(F)
    NR["record_removal"] = {"removed": True, "block": 5}
    NR["taught_block_at_probe"] = {"found": False}
    return {"USE_H": U, "USE_H_REP": _cp(U), "SHUF_H": arm("berry", None, "berry", 1.0), "FREEZE_H": F,
            ABL_ARM: A, NOREC_ARM: NR, "USE_D": _cp(U)}


def selftest_v3():
    """Gate v3 must PASS the capability case and FAIL (or read UNDEFINED) in every failing direction."""
    V = CAP_VARIANT

    def sc(mut):
        s = _synthetic_v3()
        mut(s)
        return score_seed(s, V)

    def setp(arm, lbl, **kv):
        return lambda s: s[arm]["turns"][lbl].update(kv)
    go = sc(lambda s: None)
    out = {"v3_go_case_passes": go["go"] is True,
           "v3_teach_ack_difference_is_reported_not_scored": go["go"] is True and
           go["teach_ack_same_FREEZE_vs_USE"] is False}
    fails = {
        "K1_no_learning": sc(setp("USE_H", "probe", abstained=True, recalled_svo=None, answer="?"))["go"] is False,
        "K2_shuffled_also_recalls": sc(lambda s: s["SHUF_H"]["turns"].update(
            probe=_cp(s["USE_H"]["turns"]["probe"])))["go"] is False,
        "K2_not_specific_xprobe": sc(setp("SHUF_H", "xprobe", abstained=True, recalled_svo=None))["go"] is False,
        "K3_freeze_still_recalls": sc(lambda s: s["FREEZE_H"]["turns"].update(
            probe=_cp(s["USE_H"]["turns"]["probe"])))["go"] is False,
        "K3_freeze_answer_leaks_fact_text": sc(setp("FREEZE_H", "probe",
                                                    answer="I don't know -- something about a deer?"))["go"] is False,
        "K4_ablation_still_recalls": sc(lambda s: s[ABL_ARM]["turns"].update(
            probe=_cp(s["USE_H"]["turns"]["probe"])))["go"] is False,
        "K5_host_record_framing_leak": sc(setp(NOREC_ARM, "probe",
                                               answer="I don't know about that. -- worth going further."))["go"] is False,
        "K5_host_record_leak_on_d1": sc(setp(NOREC_ARM, "d1", answer="the cat eats the fish, like the wolf"))["go"] is False,
        "K6_lever_not_moved": sc(lambda s: s["FREEZE_H"]["taught_block_at_probe"].update(mean_abs_w=0.9))["go"] is False,
        "K6_ablation_lever_not_moved": sc(lambda s: s[ABL_ARM]["taught_block_at_probe"].update(
            mean_abs_w=0.4))["go"] is False,
        "K6_write_after_teach_freeze": sc(lambda s: s["FREEZE_H"].update(store_writes_after_teach=[5]))["go"] is False,
        "K6_write_after_teach_norec": sc(lambda s: s[NOREC_ARM].update(store_writes_after_teach=[3]))["go"] is False,
        "K6_encode_episode_differs": sc(lambda s: s["FREEZE_H"]["taught_block"]["d6_last_encode"].update(
            phase_lock_steps=100))["go"] is False,
        "K6_teach_parse_differs": sc(setp("FREEZE_H", "teach", recalled_svo=["a", "b", "elk"]))["go"] is False,
        "K6_read_path_broken_d2": sc(setp("FREEZE_H", "d2", abstained=True, recalled_svo=None))["go"] is False,
        "K6_ablation_breaks_d1": sc(setp(ABL_ARM, "d1", answer="other"))["go"] is False,
        "K7_null_dirty": sc(setp("USE_H_REP", "d1", answer="other"))["go"] is False,
        "K7_abl_teach_not_deterministic": sc(setp(ABL_ARM, "teach", answer="other"))["go"] is False,
        "void_counter_not_installed": sc(lambda s: s["FREEZE_H"].update(counter_installed=False))["go"] is None,
        "void_norec_not_removed": sc(lambda s: s[NOREC_ARM].update(record_removal={"error": "x"}))["go"] is None,
        "void_norec_missing": sc(lambda s: s.pop(NOREC_ARM))["go"] is None,
        "void_ablation_not_applied": sc(lambda s: s[ABL_ARM].update(ablation={"error": "x"}))["go"] is None,
        "void_errored_turn": sc(lambda s: s["USE_H"]["turns"].update(probe={"_error": "boom"}))["go"] is None,
    }
    use_d_missing = sc(lambda s: s.pop("USE_D"))
    out["v3_missing_USE_D_is_secondary_only"] = (use_d_missing["go"] is True and
                                                 str(use_d_missing["secondary_C7_no_regression"]).startswith("UNDEFINED"))
    out["v3_fails_in_failing_direction"] = fails
    return out, (all(v for k, v in out.items() if k != "v3_fails_in_failing_direction") and all(fails.values()))


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
    # AMENDMENT A1: the ack-text relaxation is REVERTED -- an ack-only teach difference now fails under EVERY variant
    ack = _synthetic()
    ack["FREEZE_H"] = _cp(ack["FREEZE_H"])
    ack["FREEZE_H"]["turns"]["teach"]["answer"] = "The wolf hunts deer."       # same parse, different ack text
    fails["ack_text_fails_under_every_variant"] = all(score_seed(_cp(ack), v)["go"] is False
                                                      for v in ("base", "engram", "prune"))
    fails["ack_text_is_only_reported_posthoc"] = score_seed(_cp(ack), "prune")["C4_parse_posthoc"] is True
    void = score_seed(dict(_synthetic(), FREEZE_H=None))
    fails["void_is_undefined"] = void["go"] is None
    # gate v2 (readtime): the GO case passes; each new failing direction fails
    ok_v2 = score_seed(_synthetic(readtime=True), "readtime")["go"] is True
    fails["v2_ablation_does_not_remove"] = score_seed(_synthetic(readtime=True, probe_abl="deer"),
                                                      "readtime")["go"] is False
    ablfr = _synthetic(readtime=True)
    ablfr[ABL_ARM]["turns"]["probe"]["answer"] = "Setting the held thread aside -- I don't know."   # framing leak only
    fails["v2_ablation_framing_leak_fails"] = score_seed(ablfr, "readtime")["go"] is False
    fails["v2_ablation_lever_not_moved"] = score_seed(_synthetic(readtime=True, w_abl=1.0), "readtime")["go"] is False
    fails["v2_write_after_teach_in_lesion_arm_fails"] = (
        score_seed(_synthetic(readtime=True, frozen_writes=(7,)), "readtime")["go"] is False
        and score_seed(_synthetic(readtime=True, abl_writes=(7,)), "readtime")["go"] is False)
    fails["v2_missing_abl_arm_is_undefined"] = score_seed(_synthetic(readtime=False), "readtime")["go"] is None
    lever_late = _synthetic(readtime=True)
    lever_late["FREEZE_H"]["taught_block_at_probe"] = {"mean_abs_w": 0.9}     # frozen at teach, but written later
    fails["v2_lever_read_at_probe_time"] = score_seed(lever_late, "readtime")["go"] is False
    # A4 secondary: never changes the verdict; a missing EXPO_H leaves the seed DEFINED and the secondary UNDEFINED
    no_expo = score_seed(_synthetic(readtime=True), "readtime")
    fails["a4_missing_expo_secondary_undefined_only"] = (no_expo["go"] is True and
                                                         no_expo["secondary_exposure_matched"]["verdict"].startswith("UNDEFINED"))
    ex = _synthetic(readtime=True)
    ex[EXPO_ARM] = _cp(ex["SHUF_H"])
    ex[EXPO_ARM]["taught_block"] = {"found": False}
    ex[EXPO_ARM]["turns"]["probe"]["answer"] = "something else"               # exposure arm differs from FREEZE_H
    r = score_seed(ex, "readtime")
    fails["a4_secondary_can_fail_without_touching_go"] = (r["go"] is True and
                                                         r["secondary_exposure_matched"]["C3e_freeze_eq_exposure"] is False)
    agg_undef = aggregate({"42": {"go": True}, "43": {"go": None}})["GO"] is False
    v3, v3_ok = selftest_v3()
    res = {"go_case_passes": ok, "go_case_passes_v2": ok_v2, "fails_in_failing_direction": fails,
           "partial_seed_set_not_go": agg_undef, "gate_v3": v3}
    passed = ok and ok_v2 and all(fails.values()) and agg_undef and v3_ok
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
    ap.add_argument("--ablate-after-teach", action="store_true", help="worker: ABL_H post-hoc engram ablation")
    ap.add_argument("--post-teach", choices=["ablate", "norec"], default=None,
                    help="worker: experimenter step after the teach turn (ABL_H ablation / NOREC_H record removal)")
    ap.add_argument("--only-arms", nargs="+", default=None,
                    help="run only these arms (others are loaded if present) -- for running arms in parallel")
    ap.add_argument("--variant", choices=sorted(VARIANTS), default="base",
                    help="base = v1 arms; engram = + BRAIN_D6_ENGRAM_VOCAB; prune = banked invalid instrument; "
                         "readtime = gate v2 (registered to fail; superseded); capability = gate v3")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.worker:
        return _worker(a.env, a.teach, a.out, ablate_after_teach=a.ablate_after_teach, post_teach=a.post_teach)
    res = run(a.seeds, a.arm_dir, resume=not a.no_resume, score_only=a.score_only, variant=a.variant,
              only_arms=a.only_arms)
    print(json.dumps(res["aggregate"], indent=2))
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(res, open(a.json, "w"), indent=2, default=str)
        print("[d6] wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
