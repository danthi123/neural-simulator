"""C1 GENERALIZATION PROBES -- dev seeds only: do the category organ's flagged one-hop resemblance guesses behave
like LEARNING (specific evidence wins; the guess grows with the evidence) rather than TEMPLATING?

WHY (RULING c1-moat-ruled-1013, Claude, owner-delegated; research/coordination/LOCAL_LLM_RUNBOOK.md). C1 category
inheritance is default-ON (webapp/category_inherit_chat.py + research/runners/category_inherit_organ.py) and makes
flagged one-hop guesses: a bat that eats the bug guesses that it swims in the lake, because the fish eat the bug too
(`c_bat_swim` in research/runners/c1_flip_dev_check.py). The ruling KEEPS the guesses and owes two dev-seed probes
before the 11-01 flips; if either fails, file NEEDS. Plan and bars: docs/drafts/stage1-c1-inheritance/PROBES-PLAN.md
(the bars below are copied there; both were written and committed before any probe run).

SESSIONS (one fresh process and one fresh ChatBrain per (seed, probe), through the real handler; production defaults:
BRAIN_CATEGORY_INHERIT and BRAIN_CATEGORY_INHERIT_HOP are UNSET so the flipped defaults apply, the lesion env is
removed, and the effective flag values are recorded):
  override  the cat_bridge teach turns (C1 birds and fish, the bat eats the bug and flies in the sky, the canary flies
            in the sky); `what does the bat swim` (the guess, lake); `the bat swims in the pond` (the specific told
            fact: frame `swim pond`, admitted under the same verb as `swim lake`); `what does the bat swim` again;
            three reported collateral reads.
  graded    a fish category (trout, salmon: eat bug, bite hook, hide weed, swim lake); three targets with THREE told
            facts each, k of them fish traits (eel k=3, toad k=2, newt k=1; the rest frames no one else carries),
            taught in DESCENDING k with shared traits first (the order favors the low-k targets: against the
            hypothesis); `what does the <target> swim` for each; then `the eel swims in the river` and `what does the
            eel swim` (the override against the strongest guess).
DIRECT READS: after the turns in DIRECT_READS the worker calls `infer(agent, verb)` on the session's own organ
(chat._category_inherit_organ; read-only, it restores state and weights): the organ's decision in its own frame
space, even when the strict route answers the reply. After a turn the organ answered, it must reproduce the trace.

THE BARS (written before any run; PROBES-PLAN.md section 3).
  OVERRIDE, per dev seed:
    (O1) organ: after `the bat swims in the pond`, the direct organ read of (bat, swim) DECIDES pond (reason
         `reached`: pond clears the floor and beats lake by the organ's own 1.25x margin);
    (O2) reply: the reply to the second `what does the bat swim` names pond (recalled or flagged) and names lake
         nowhere (neither SVO field, no word of the answer);
    (O3, O4) the same for the eel and river, when the eel's pre-override guess reached lake; otherwise the eel case
         is VACUOUS (reported, decides nothing).
    Seed precondition: the first `what does the bat swim` was an organ read that reached lake (a guess existed to
    override); without it the seed is UNDEFINED.
  GRADED, per dev seed: the lake rate recorded in the organ trace of `what does the <target> swim` rises STRICTLY with
    the shared traits, newt (1) < toad (2) < eel (3); a tie fails (the rate is recorded whether or not it clears the
    floor).
  Each probe: PASS iff all 8 dev seeds are DEFINED and pass; UNDEFINED if any seed is undefined (re-run that shard
  once); otherwise FAIL naming each seed and clause.
  DEFINED (tools.verdict preconditions, data quality only; the bar outcome is never a precondition): valid shards (no
  errored turn, source tiny-demo +LTM, PYTHONHASHSEED=0, corpus present); effective BRAIN_CATEGORY_INHERIT and
  BRAIN_CATEGORY_INHERIT_HOP both true; the organ ENGAGED (every teach turn observed with its expected SVO; every
  probe in ENGAGE_LABELS carries an organ query read); every organ read (trace and direct) records `hop: true` and no
  lesion; no swallowed organ error; every direct read present and reproducing its organ-answered turn's trace.

HOST-DECISION DEBT (PROBES-PLAN.md section 5): candidate admission by the asked VERB STRING and the floor + tie-margin
compare are host-side (as are slot recruitment, SVO extraction + lemma, the read window, and the strict-route-first
precedence, which is why the organ clauses, not the reply clauses, are the evidence).

LANE: one pool job per (seed, probe), numpy, mem_gb=9, max_age_h=72. Dev seeds only: gate seeds are refused.
No sim/ or webapp/ edit.

    python -m research.runners.c1_generalization_probes --emit-pool-jobs <SHA> [--seeds 1,...,8]   # TSV cmd<TAB>why
    python -m research.runners.c1_generalization_probes --shard --seed 7 --probe override --out <path>
    python -m research.runners.c1_generalization_probes --score <root> [--seeds 1,...,8] [--out <json>]
    python -m research.runners.c1_generalization_probes --organ-feed --seeds 1,...,8 [--out <json>]  # diagnostic
    python -m research.runners.c1_generalization_probes --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import stage1_exit_gate as SG  # noqa: E402
from research.runners import c1_flip_dev_check as CF  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

PLAN = "docs/drafts/stage1-c1-inheritance/PROBES-PLAN.md"
RULING = "c1-moat-ruled-1013"
DEV_SEEDS = tuple(range(1, 9))
GATE_SEEDS = SG.GATE_SEEDS
OUT_DIR = "research/findings/raw/_stage1_c1_generalization"
PROBES = ("override", "graded")
POOL_MEM_GB = 9
POOL_MAX_AGE_H = 72
C1_ENVS_REMOVED = ("BRAIN_CATEGORY_INHERIT", "BRAIN_CATEGORY_INHERIT_HOP", "BRAIN_CATEGORY_INHERIT_LESION")

# ── the turn tables (label, message, role) and each teach turn's expected organ SVO (agent, lemma, patient) ─────────
_VERB_LEMMA = {"eats": "eat", "flies": "fly", "swims": "swim", "bites": "bite", "hides": "hide", "finds": "find",
               "sees": "see", "likes": "like"}
_FUNCTION = {"the", "in", "a"}


def expected_svo(message):
    """The (agent, lemma, patient) a teach sentence should reach the organ as (pinned against the production B3
    extractor + lemma_verb by a CPU test)."""
    toks = [t for t in message.lower().split() if t not in _FUNCTION]
    if len(toks) != 3 or toks[1] not in _VERB_LEMMA:
        raise ValueError("not a probe teach sentence: %r" % message)
    return toks[0], _VERB_LEMMA[toks[1]], toks[2]


OVERRIDE_TURNS = [t for t in CF.CAT_BRIDGE_TURNS if t[2] == "teach"] + [
    ("p_bat_swim_pre", "what does the bat swim", "probe"),          # the inherited guess (lake) before the told fact
    ("t_bat_swim_told", "the bat swims in the pond", "teach"),      # the specific told fact: frame `swim pond`
    ("p_bat_swim_post", "what does the bat swim", "probe"),         # O2 (reply); O1 reads the organ directly after it
    ("c_bat_eat", "what does the bat eat", "control"),              # reported: the bat's told eat (bug)
    ("c_canary_eat", "what does the canary eat", "control"),        # reported: the C1 inheritance after the told fact
    ("c_canary_swim", "what does the canary swim", "control"),      # reported collateral: canary shares fly sky w/ bat
]
_FISH = [("t_trout_eat", "the trout eats the bug"), ("t_salmon_eat", "the salmon eats the bug"),
         ("t_trout_bite", "the trout bites the hook"), ("t_salmon_bite", "the salmon bites the hook"),
         ("t_trout_hide", "the trout hides in the weed"), ("t_salmon_hide", "the salmon hides in the weed"),
         ("t_trout_swim", "the trout swims in the lake"), ("t_salmon_swim", "the salmon swims in the lake")]
_TARGETS = [  # DESCENDING k, shared traits first (the conservative order: see the module doc)
    ("t_eel_eat", "the eel eats the bug"), ("t_eel_bite", "the eel bites the hook"),
    ("t_eel_hide", "the eel hides in the weed"),
    ("t_toad_eat", "the toad eats the bug"), ("t_toad_bite", "the toad bites the hook"),
    ("t_toad_find", "the toad finds the stone"),
    ("t_newt_eat", "the newt eats the bug"), ("t_newt_see", "the newt sees the moon"),
    ("t_newt_like", "the newt likes the rain")]
SHARED_TRAITS = {"eel": 3, "toad": 2, "newt": 1}
GRADED_PROBE = {"eel": "p_eel_swim", "toad": "p_toad_swim", "newt": "p_newt_swim"}
GRADED_TURNS = ([(l, m, "teach") for l, m in _FISH + _TARGETS]
                + [(GRADED_PROBE[a], "what does the %s swim" % a, "probe") for a in ("eel", "toad", "newt")]
                + [("t_eel_swim_told", "the eel swims in the river", "teach"),
                   ("p_eel_swim_post", "what does the eel swim", "probe")])
TURNS = {"override": OVERRIDE_TURNS, "graded": GRADED_TURNS}
DIRECT_READS = {"override": {"p_bat_swim_pre": [("bat", "swim")], "p_bat_swim_post": [("bat", "swim")]},
                "graded": {"p_eel_swim": [("eel", "swim")], "p_toad_swim": [("toad", "swim")],
                           "p_newt_swim": [("newt", "swim")], "p_eel_swim_post": [("eel", "swim")]}}
ENGAGE_LABELS = {"override": ("p_bat_swim_pre",), "graded": ("p_eel_swim", "p_toad_swim", "p_newt_swim")}
GUESS, TOLD = "lake", {"bat": "pond", "eel": "river"}
OVERRIDE_CASES = {  # agent -> (shard, pre-override probe, post-override probe)
    "bat": ("override", "p_bat_swim_pre", "p_bat_swim_post"),
    "eel": ("graded", "p_eel_swim", "p_eel_swim_post"),
}


def shard_path(root, seed, probe):
    return os.path.join(root, "s%d" % int(seed), "%s.json" % probe)


def _refuse_gate_seed(seed):
    if int(seed) in GATE_SEEDS:
        raise SystemExit("c1_generalization_probes: seed %d is a GATE seed; this module runs dev seeds only" % int(seed))


# ══════════════════════════════════════════════ RUN (one fresh process per (seed, probe)) ═══════════════════════════
def _direct(org, pairs):
    if org is None:
        return None
    out = {}
    for a, v in pairs:
        try:
            out["%s/%s" % (a, v)] = org.infer(a, v)
        except Exception as e:                       # recorded, never raised: a failed read is UNDEFINED, not a fail
            out["%s/%s" % (a, v)] = {"error": "%s: %s" % (type(e).__name__, e)}
    return out


def run_worker(seed, probe, out_path):
    _refuse_gate_seed(seed)
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))
    hash_ok = os.environ.get("PYTHONHASHSEED") == "0"
    corpus_present = os.path.exists(os.path.join(_REPO, "data", "corpus", "tinystories.txt"))
    t0 = time.time()
    from webapp.server import brain_chat, BrainChatRequest, _BRAIN_CHATS
    c1_flags = SG.effective_c1_flags()
    session = "c1gp_%s_%d" % (probe, int(seed))
    key = (session, "tiny-demo", "stub")
    responses, direct, timings = {}, {}, {}
    for i, (label, msg, _role) in enumerate(TURNS[probe]):
        t1 = time.time()
        try:
            r = brain_chat(BrainChatRequest(session=session, message=msg, brain="tiny-demo", renderer="stub",
                                            reset=(i == 0)))     # `rich` omitted = the production default path
            responses[label] = json.loads(r.body)
        except Exception as e:
            responses[label] = {"_error": "%s: %s" % (type(e).__name__, e)}
        if label in DIRECT_READS[probe]:
            org = getattr(_BRAIN_CHATS.get(key), "_category_inherit_organ", None)
            direct[label] = _direct(org, DIRECT_READS[probe][label])
        timings[label] = round(time.time() - t1, 1)
    org = getattr(_BRAIN_CHATS.get(key), "_category_inherit_organ", None)
    try:
        audit = org.weight_audit() if org is not None else None
    except Exception as e:
        audit = {"error": "%s: %s" % (type(e).__name__, e)}
    out = {"runner": "research/runners/c1_generalization_probes.py", "plan": PLAN, "ruling": RULING,
           "seed": int(seed), "probe": probe, "pythonhashseed_ok": hash_ok, "corpus_present": corpus_present,
           "c1_flags": c1_flags, "c1_envs_seen": {k: os.environ.get(k) for k in C1_ENVS_REMOVED},
           "session": session, "turns": [{"label": l, "message": m, "role": r} for l, m, r in TURNS[probe]],
           "responses": responses, "direct_reads": direct, "weight_audit": audit, "timings_s": timings,
           "wall_s": round(time.time() - t0, 1), "expected_source": SG.EXPECTED_SOURCE,
           "peak_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,   # pool mem_gb sizing
           "sources": sorted({str(r.get("source")) for r in responses.values() if isinstance(r, dict)})}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("[c1 probes worker] seed=%d %s -> %d turns, %.0fs -> %s" % (seed, probe, len(responses), out["wall_s"],
                                                                     out_path), flush=True)
    return 0


def run_shard(seed, probe, out_path):
    """One (seed, probe) session in a FRESH subprocess with the Stage-1 frozen env (PYTHONHASHSEED must be set before
    the interpreter starts) and the C1 envs REMOVED, so the flipped production defaults apply."""
    _refuse_gate_seed(seed)
    env = dict(os.environ)
    env.update(SG.BASE_ENV)
    for k in C1_ENVS_REMOVED:
        env.pop(k, None)
    part = out_path + ".part.json"
    cmd = [sys.executable, "-u", "-m", "research.runners.c1_generalization_probes", "--worker", "--seed",
           str(int(seed)), "--probe", probe, "--out", part]
    p = subprocess.run(cmd, env=env, cwd=_REPO)
    if p.returncode != 0 or not os.path.exists(part):
        print("[c1 probes shard] worker FAILED rc=%s seed=%s %s" % (p.returncode, seed, probe), flush=True)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    os.replace(part, out_path)
    try:
        os.remove(part + ".prov.json")
    except OSError:
        pass
    return 0


def pool_job_lines(sha, seeds=DEV_SEEDS, out_dir=OUT_DIR, probes=PROBES):
    out = []
    for seed in seeds:
        _refuse_gate_seed(seed)
        for probe in probes:
            p = shard_path(out_dir, seed, probe)
            cmd = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh %d 1 && mkdir -p %s && "
                   "bash tools/memcap.sh %d -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                   "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m "
                   "research.runners.c1_generalization_probes --shard --seed %d --probe %s --out %s"
                   % (sha, POOL_MEM_GB, os.path.dirname(p), POOL_MEM_GB, seed, probe, p))
            why = ("C1 generalization probe %s (RULING %s), DEV seed %d at %s; production C1 defaults (flags unset, "
                   "effective values recorded); bars before data in %s; record: flip dev rerun 2c2d55952 bat-swim "
                   "lake guess 8/8 dev, override/graded unmeasured; mem_gb=%d; max_age_h=%d"
                   % (probe, RULING, seed, sha[:9], PLAN, POOL_MEM_GB, POOL_MAX_AGE_H))
            out.append((cmd, why))
    return out


# ══════════════════════════════════════════════ SCORING (pure functions over shard dicts) ══════════════════════════
def _trace(resp):
    return resp.get("category_inherit") if isinstance(resp, dict) else None


def _query_read(resp):
    """The organ QUERY record of a turn (it carries `lesion`), or None (no trace, or the organ_not_built decline)."""
    rd = (_trace(resp) or {}).get("read") if isinstance(_trace(resp), dict) else None
    return rd if isinstance(rd, dict) and "lesion" in rd else None


def rate_of(read, obj):
    """The rate the read's `admitted` list records for `obj` (None if the read or the frame is absent)."""
    if not isinstance(read, dict):
        return None
    for o, r in read.get("admitted") or []:
        if str(o).lower() == obj:
            return float(r)
    return None


def decides(read, obj):
    return isinstance(read, dict) and read.get("patient") == obj and read.get("reason") == "reached"


def reply_names(resp, want, forbid):
    """True iff the reply is an answer naming `want` (recalled or flagged) and names `forbid` nowhere."""
    p = SG._primary(resp)
    if p is None:
        return None
    named = {str(s[2]).lower() for s in (p["recalled_svo"], p["hypothesis_svo"]) if s and len(s) == 3}
    words = set(re.findall(r"[a-z]+", p["answer"].lower()))
    return bool(p["abstained"] is False and want in named and forbid not in named and forbid not in words)


def _direct_read(shard, label, agent, verb="swim"):
    return ((shard.get("direct_reads") or {}).get(label) or {}).get("%s/%s" % (agent, verb))


def shard_problems(shard, probe):
    """Validity + engagement problems of one shard (empty list = DEFINED for that shard)."""
    if shard is None:
        return ["missing"]
    if not isinstance(shard, dict) or "_load_error" in shard:
        return ["unreadable"]
    probs = []
    resp = shard.get("responses") or {}
    if shard.get("probe") != probe:
        probs.append("wrong_probe")
    if [t[0] for t in TURNS[probe]] != list(resp):
        probs.append("turn_labels_differ")
    if any(isinstance(r, dict) and "_error" in r for r in resp.values()):
        probs.append("errored_turn")
    if shard.get("sources") != [SG.EXPECTED_SOURCE]:
        probs.append("source:%s" % shard.get("sources"))
    if shard.get("pythonhashseed_ok") is not True:
        probs.append("pythonhashseed")
    if shard.get("corpus_present") is not True:
        probs.append("corpus_absent")
    fl = shard.get("c1_flags") or {}
    if fl.get("BRAIN_CATEGORY_INHERIT") is not True or fl.get("BRAIN_CATEGORY_INHERIT_HOP") is not True:
        probs.append("flag_not_effective")
    for label, msg, role in TURNS[probe]:          # engaged (1): every teach turn reached the organ as intended
        if role != "teach":
            continue
        ob = (_trace(resp.get(label)) or {}).get("observe") if isinstance(_trace(resp.get(label)), dict) else None
        if not (isinstance(ob, dict) and ob.get("observed") is True
                and list(ob.get("svo") or []) == list(expected_svo(msg))):
            probs.append("teach_not_observed:%s" % label)
    for label in ENGAGE_LABELS[probe]:              # engaged (2): the pre-override probes are organ reads
        if _query_read(resp.get(label)) is None:
            probs.append("organ_not_engaged:%s" % label)
    reads = [_query_read(r) for r in resp.values()]
    for label, pairs in DIRECT_READS[probe].items():
        for a, v in pairs:
            d = _direct_read(shard, label, a, v)
            if not isinstance(d, dict) or "lesion" not in d:
                probs.append("direct_read_missing:%s" % label)
                continue
            reads.append(d)
            q = _query_read(resp.get(label))
            if q is not None and (q.get("admitted") != d.get("admitted") or q.get("patient") != d.get("patient")):
                probs.append("direct_read_differs:%s" % label)
    reads = [r for r in reads if r is not None]
    if any(r.get("hop") is not True for r in reads):
        probs.append("hop_not_recorded")
    if any(r.get("lesion") is not None for r in reads):
        probs.append("lesion_active")
    for label, r in resp.items():
        tr = _trace(r)
        if isinstance(tr, dict):
            ob, rd = tr.get("observe"), tr.get("read")
            if tr.get("error") or (isinstance(ob, dict) and ob.get("error")) or (
                    isinstance(rd, dict) and str(rd.get("reason") or "").startswith("error")):
                probs.append("organ_error:%s" % label)
    return probs


def override_case(pre_read, post_direct, post_reply, told):
    """One override case (agent asked `swim`): was there a guess, does the organ decide the told object after the
    told fact (organ clause), does the reply name it (reply clause). post_reply None = not measured (organ feed)."""
    out = {"guess_before": decides(pre_read, GUESS), "lake_rate_before": rate_of(pre_read, GUESS),
           "lake_rate_after": rate_of(post_direct, GUESS), "told_rate_after": rate_of(post_direct, told),
           "organ_decides_told": decides(post_direct, told), "post_reason": (post_direct or {}).get("reason"),
           "post_patient": (post_direct or {}).get("patient")}
    la, ta = out["lake_rate_after"], out["told_rate_after"]
    out["told_over_lake_after"] = (round(ta / la, 3) if (ta is not None and la) else None)
    if post_reply is not None:
        out["reply_names_told"] = reply_names(post_reply, told, GUESS)
        p = SG._primary(post_reply) or {}
        out["reply_route"] = ("recalled" if p.get("recalled_svo") else
                              "flagged_guess" if p.get("hypothesis_svo") else "abstained")
    return out


def override_seed(ov, gr):
    """ov / gr: the seed's override / graded shard dicts. Returns the per-seed OVERRIDE record (module doc bars)."""
    undefined = ["override/%s" % p for p in shard_problems(ov, "override")]
    undefined += ["graded/%s" % p for p in shard_problems(gr, "graded")]
    if undefined:
        return {"defined": False, "pass": None, "undefined": undefined}
    cases = {}
    for agent, (probe, pre, post) in OVERRIDE_CASES.items():
        sh = ov if probe == "override" else gr
        cases[agent] = override_case(_query_read(sh["responses"].get(pre)), _direct_read(sh, post, agent),
                                     sh["responses"].get(post), TOLD[agent])
    if not cases["bat"]["guess_before"]:
        return {"defined": False, "pass": None, "undefined": ["no_guess_to_override:bat"], "cases": cases}
    why = []
    if not cases["bat"]["organ_decides_told"]:
        why.append("O1 bat: organ read decides %s (%s), not pond" % (cases["bat"]["post_patient"],
                                                                      cases["bat"]["post_reason"]))
    if not cases["bat"]["reply_names_told"]:
        why.append("O2 bat: the reply does not name pond alone")
    eel_vacuous = not cases["eel"]["guess_before"]
    if not eel_vacuous:
        if not cases["eel"]["organ_decides_told"]:
            why.append("O3 eel: organ read decides %s (%s), not river" % (cases["eel"]["post_patient"],
                                                                           cases["eel"]["post_reason"]))
        if not cases["eel"]["reply_names_told"]:
            why.append("O4 eel: the reply does not name river alone")
    collateral = {l: {"patient": SG._patient(ov["responses"].get(l)), "abstained": SG._abstained(ov["responses"].get(l)),
                      "read": {k: (_query_read(ov["responses"].get(l)) or {}).get(k) for k in ("patient", "admitted")}}
                  for l in ("c_bat_eat", "c_canary_eat", "c_canary_swim")}
    return {"defined": True, "pass": not why, "why": why, "eel_vacuous": eel_vacuous, "cases": cases,
            "collateral": collateral}


def graded_core(rates):
    """rates: {agent: lake rate}. PASS iff strictly rising with the shared traits (newt 1 < toad 2 < eel 3)."""
    order = sorted(SHARED_TRAITS, key=SHARED_TRAITS.get)
    vals = [rates.get(a) for a in order]
    if any(v is None for v in vals):
        return {"measured": False, "rates": rates}
    rising = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    return {"measured": True, "rising": rising, "rates": {a: rates[a] for a in order},
            "steps": [round(vals[i + 1] - vals[i], 4) for i in range(len(vals) - 1)]}


def graded_seed(gr):
    undefined = ["graded/%s" % p for p in shard_problems(gr, "graded")]
    if undefined:
        return {"defined": False, "pass": None, "undefined": undefined}
    reads = {a: _query_read(gr["responses"].get(GRADED_PROBE[a])) for a in SHARED_TRAITS}
    core = graded_core({a: rate_of(r, GUESS) for a, r in reads.items()})
    if not core["measured"]:
        return {"defined": False, "pass": None, "undefined": ["graded/lake_rate_unrecorded"], "core": core}
    detail = {a: {"k": SHARED_TRAITS[a], "lake_rate": rate_of(r, GUESS), "reason": r.get("reason"),
                  "patient": r.get("patient"), "reply_patient": SG._patient(gr["responses"].get(GRADED_PROBE[a]))}
              for a, r in reads.items()}
    why = [] if core["rising"] else ["lake rate not strictly rising newt<toad<eel: %s" % core["rates"]]
    return {"defined": True, "pass": core["rising"], "why": why, "core": core, "per_target": detail}


def decide(per_seed, seeds=DEV_SEEDS):
    """PASS iff every listed seed is defined and passes; UNDEFINED if any is undefined; else FAIL naming each."""
    recs = [per_seed.get(int(s)) for s in seeds]
    bad = [int(s) for s, r in zip(seeds, recs) if r is None or not r.get("defined")]
    if bad:
        return {"verdict": "UNDEFINED", "undefined_seeds": bad, "n_pass": None}
    n_pass = sum(1 for r in recs if r["pass"])
    why = ["seed %d: %s" % (int(s), w) for s, r in zip(seeds, recs) for w in (r.get("why") or [])]
    return {"verdict": "PASS" if n_pass == len(recs) else "FAIL", "n_pass": n_pass, "n_seeds": len(recs), "why": why}


def _earn(res, label):
    """Attach the tools.verdict preconditions block (data quality only; decide() owns PASS/FAIL). A precondition that
    fails under an asserted PASS/FAIL downgrades the verdict to UNDEFINED."""
    per = res.get("per_seed") or {}
    reasons = [str(u) for r in per.values() for u in (r.get("undefined") or [])]

    def _none_of(*subs):
        return not any(s in x for x in reasons for s in subs)

    v = Verdict(label)
    v.require("every seed defined", bool(per) and all(bool(r.get("defined")) for r in per.values()), expect=True)
    v.require("C1 flags effective (BRAIN_CATEGORY_INHERIT and _HOP true)", _none_of("flag_not_effective"), expect=True)
    v.require("organ engaged (teach turns observed with the expected SVO; pre-override probes are organ reads)",
              _none_of("teach_not_observed", "organ_not_engaged"), expect=True)
    v.require("every organ read records hop:true and no lesion", _none_of("hop_not_recorded", "lesion_active"),
              expect=True)
    v.require("direct reads present and reproducing the turn's trace",
              _none_of("direct_read_missing", "direct_read_differs"), expect=True)
    v.require("no organ errors, no errored turns, valid shards",
              _none_of("organ_error", "errored_turn", "missing", "unreadable", "source:", "pythonhashseed",
                       "corpus_absent", "turn_labels_differ", "wrong_probe"), expect=True)
    d = v.decide(go=(res.get("verdict") == "PASS"), verbose=False)
    if res.get("verdict") in ("PASS", "FAIL") and d["status"] == "UNDEFINED":
        res["verdict_unearned"] = res["verdict"]
        res["verdict"] = "UNDEFINED"
    res["preconditions"] = d["preconditions"]
    res["undefined_reasons"] = d["undefined_reasons"]
    return res


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": repr(e)}


def score_shards(shards, seeds=DEV_SEEDS):
    """shards: {seed: {"override": dict|None, "graded": dict|None}} -> both probe verdicts."""
    out = {"runner": "research/runners/c1_generalization_probes.py", "plan": PLAN, "ruling": RULING,
           "seeds": [int(s) for s in seeds]}
    ov = {int(s): override_seed(shards[s].get("override"), shards[s].get("graded")) for s in seeds}
    gr = {int(s): graded_seed(shards[s].get("graded")) for s in seeds}
    out["override"] = _earn(dict(decide(ov, seeds), per_seed=ov), "C1 probe OVERRIDE (dev, %s)" % RULING)
    out["graded"] = _earn(dict(decide(gr, seeds), per_seed=gr), "C1 probe GRADED (dev, %s)" % RULING)
    return out


def score(root, seeds=DEV_SEEDS):
    shards = {s: {p: _load(shard_path(root, s, p)) for p in PROBES} for s in seeds}
    out = score_shards(shards, seeds)
    out.update(root=root, scored_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    return out


# ══════════════════════════════════════════════ ORGAN FEED (diagnostic, not the bar; no brain) ════════════════════
def organ_feed(seed, probe):
    """The same teach facts fed straight into a standalone organ (seed = the chat's BRAIN_CHAT_SEED), read at the
    DIRECT_READS points. Equal to the chat organ's direct reads iff the chat fed exactly these facts in this order."""
    _refuse_gate_seed(seed)
    from research.runners.category_inherit_organ import CategoryInheritOrgan
    o = CategoryInheritOrgan(seed=int(seed), lesion=None, hop=True)
    reads = {}
    for label, msg, role in TURNS[probe]:
        if role == "teach":
            o.observe(*expected_svo(msg))
        if label in DIRECT_READS[probe]:
            reads[label] = _direct(o, DIRECT_READS[probe][label])
    return json.loads(json.dumps(reads, default=str))


def organ_feed_seed(seed):
    ov, gr = organ_feed(seed, "override"), organ_feed(seed, "graded")
    cases = {}
    for agent, (probe, pre, post) in OVERRIDE_CASES.items():
        r = ov if probe == "override" else gr
        cases[agent] = override_case(r[pre]["%s/swim" % agent], r[post]["%s/swim" % agent], None, TOLD[agent])
    core = graded_core({a: rate_of(gr[GRADED_PROBE[a]]["%s/swim" % a], GUESS) for a in SHARED_TRAITS})
    return {"seed": int(seed), "override_cases": cases, "graded": core, "reads": {"override": ov, "graded": gr}}


# ══════════════════════════════════════════════ SELFTEST (no brain) ════════════════════════════════════════════════
def _read(admitted, lesion=None, hop=True):
    """A synthetic organ read in infer()'s format, decided by the organ's own floor/margin rule."""
    from research.runners.category_inherit_organ import FLOOR_RATE, TIE_MARGIN
    adm = sorted(([o, round(float(r), 4)] for o, r in admitted.items()), key=lambda t: (-t[1], t[0]))
    if not adm or adm[0][1] < FLOOR_RATE:
        pat, why = None, "below_floor"
    elif len(adm) > 1 and adm[1][1] >= FLOOR_RATE and adm[0][1] < TIE_MARGIN * adm[1][1]:
        pat, why = None, "tie"
    else:
        pat, why = adm[0][0], "reached"
    return {"patient": pat, "reason": why, "admitted": adm, "lesion": lesion, "lesion_held": None, "hop": hop,
            "rates": {}}


def _resp(abstained=True, svo=None, hyp=None, trace=None, answer=None):
    d = {"answer": answer or ("I don't know about that." if abstained else "x"), "abstained": abstained,
         "recalled_svo": svo, "hypothesis_svo": hyp, "source": SG.EXPECTED_SOURCE}
    if trace is not None:
        d["category_inherit"] = trace
    return d


def synthetic_shard(probe, kind="perfect", rng=None):
    """A valid shard in the worker's format for a brain of `kind` (selftest + bar-sanity). `rng` (numpy Generator)
    jitters the rates inside the kind's profile."""
    import numpy as np
    rng = rng or np.random.default_rng(0)
    u = lambda lo, hi: float(rng.uniform(lo, hi))  # noqa: E731
    responses, direct = {}, {}
    lake_bat = u(0.03, 0.08)
    graded = {"newt": u(0.015, 0.03)}          # the perfect k=3 guess then clears the floor (eel >= 0.023)
    graded["toad"] = graded["newt"] + u(0.004, 0.03)
    graded["eel"] = graded["toad"] + u(0.004, 0.03)
    if kind == "graded_flat":
        graded = dict.fromkeys(graded, round(u(0.02, 0.08), 4))
    elif kind == "graded_inverted":
        graded = {"newt": graded["eel"], "toad": graded["toad"], "eel": graded["newt"]}
    elif kind == "graded_top_tie":
        graded["eel"] = graded["toad"]
    elif kind == "graded_count_only":     # strength follows the NUMBER of facts (constant here): one level, jittered
        base = u(0.02, 0.08)
        graded = {a: round(base + u(-0.002, 0.002), 4) for a in graded}
    for label, msg, role in TURNS[probe]:
        if role == "teach":
            tr = {"observe": {"observed": True, "member_slot": 0, "context_slot": 0, "svo": list(expected_svo(msg))},
                  "read": None}
            if kind == "acquire_fail" and label in ("t_bat_swim_told", "t_eel_swim_told"):
                tr = None
            responses[label] = _resp(abstained=False, svo=list(expected_svo(msg)), trace=tr)
        else:
            responses[label] = _resp()
    def _q(label, admitted, reply_patient=None, hyp=True, agent=None, **kw):  # noqa: E306
        rd = _read(admitted, **kw)
        tr = {"observe": None, "read": dict(rd, agent=agent, action="swim")}
        if rd["patient"] is not None and hyp:
            responses[label] = _resp(abstained=False, hyp=[agent, "swim", rd["patient"]], trace=tr)
        else:
            responses[label] = _resp(trace=tr)
        return rd
    if probe == "override":
        pre_lake = 0.0 if kind == "no_guess" else lake_bat
        pre = _q("p_bat_swim_pre", {"lake": pre_lake}, agent="bat",
                 hop=(kind != "hop_missing"))
        direct["p_bat_swim_pre"] = {"bat/swim": {k: v for k, v in pre.items()}}
        if kind == "not_engaged":
            responses["p_bat_swim_pre"] = _resp()
        told = u(0.15, 0.4)
        post_adm = {"lake": lake_bat + u(0.0, 0.02), "pond": told}
        if kind in ("template", "reply_lake"):
            post_adm = {"lake": lake_bat + 0.05, "pond": u(0.0, 0.01)} if kind == "template" else post_adm
        if kind == "override_tie":
            post_adm = {"lake": 0.10, "pond": 0.11}
        direct["p_bat_swim_post"] = {"bat/swim": _read(post_adm)}
        if kind == "reply_lake":
            responses["p_bat_swim_post"] = _resp(abstained=False, svo=["bat", "swim", "pond"],
                                                 answer="The bat swims in the pond, or the lake.")
        elif kind == "template":
            responses["p_bat_swim_post"] = _resp(abstained=False, hyp=["bat", "swim", "lake"],
                                                 trace={"observe": None, "read": direct["p_bat_swim_post"]["bat/swim"]})
        else:
            responses["p_bat_swim_post"] = _resp(abstained=False, svo=["bat", "swim", "pond"],
                                                 answer="The bat swims the pond.")
    else:
        for a in SHARED_TRAITS:
            rd = _q(GRADED_PROBE[a], {"lake": graded[a]}, agent=a)
            direct[GRADED_PROBE[a]] = {"%s/swim" % a: dict(rd)}
        if kind == "eel_vacuous":
            rd = _q("p_eel_swim", {"lake": 0.01}, agent="eel")
            direct["p_eel_swim"] = {"eel/swim": dict(rd)}
        river = u(0.15, 0.4)
        post = {"lake": graded["eel"], "river": river} if kind != "eel_template" else {"lake": 0.3, "river": 0.0}
        direct["p_eel_swim_post"] = {"eel/swim": _read(post)}
        responses["p_eel_swim_post"] = _resp(abstained=False, svo=["eel", "swim", "river"],
                                             answer="The eel swims the river.")
        if kind == "direct_differs":
            direct["p_toad_swim"]["toad/swim"]["admitted"] = [["lake", 0.5]]
    return {"runner": "research/runners/c1_generalization_probes.py", "probe": probe, "responses": responses,
            "direct_reads": direct, "sources": [SG.EXPECTED_SOURCE], "pythonhashseed_ok": True,
            "corpus_present": True, "c1_flags": {"BRAIN_CATEGORY_INHERIT": kind != "flag_off",
                                                 "BRAIN_CATEGORY_INHERIT_HOP": True}}


def synthetic_seed(kind="perfect", rng=None):
    """{"override": shard, "graded": shard} for one seed of a brain of `kind`."""
    return {p: synthetic_shard(p, kind, rng) for p in PROBES}


def selftest(verbose=True):
    checks = []

    def chk(name, cond):
        checks.append((name, bool(cond)))
        if verbose:
            print("  [%s] %s" % ("ok" if cond else "FAIL", name))

    def ov(kind):
        s = synthetic_seed(kind)
        return override_seed(s["override"], s["graded"])

    def gr(kind):
        return graded_seed(synthetic_seed(kind)["graded"])

    r = ov("perfect")
    chk("perfect: override defined and passes (bat + eel)", r["defined"] and r["pass"] and not r["eel_vacuous"])
    chk("perfect: graded rises and passes", gr("perfect")["pass"] is True)
    chk("template (lake still wins after the told pond) fails O1", ov("template")["pass"] is False
        and any(w.startswith("O1") for w in ov("template")["why"]))
    chk("a pond/lake tie (organ abstains) fails O1", ov("override_tie")["pass"] is False)
    chk("a reply that also names lake fails O2", ov("reply_lake")["pass"] is False
        and any(w.startswith("O2") for w in ov("reply_lake")["why"]))
    chk("no guess before the told fact is UNDEFINED", ov("no_guess")["defined"] is False
        and "no_guess_to_override:bat" in ov("no_guess")["undefined"])
    chk("pre-override probe without an organ read is UNDEFINED", ov("not_engaged")["defined"] is False)
    chk("a read without hop is UNDEFINED", ov("hop_missing")["defined"] is False)
    chk("a told fact the organ never observed is UNDEFINED", ov("acquire_fail")["defined"] is False)
    chk("C1 flag not effective is UNDEFINED", ov("flag_off")["defined"] is False)
    chk("the eel still guessing lake after the told river fails O3", ov("eel_template")["pass"] is False
        and any(w.startswith("O3") for w in ov("eel_template")["why"]))
    r = ov("eel_vacuous")
    chk("an eel with no guess is VACUOUS (bat decides)", r["defined"] and r["pass"] and r["eel_vacuous"])
    chk("flat graded fails", gr("graded_flat")["pass"] is False)
    chk("inverted graded fails", gr("graded_inverted")["pass"] is False)
    chk("a tie between k=2 and k=3 fails", gr("graded_top_tie")["pass"] is False)
    chk("a direct read that does not reproduce the trace is UNDEFINED", gr("direct_differs")["defined"] is False)
    per = {s: ov("perfect") for s in DEV_SEEDS}
    chk("8/8 -> PASS", decide(per)["verdict"] == "PASS")
    per[3] = ov("template")
    chk("7/8 -> FAIL naming seed 3", decide(per)["verdict"] == "FAIL" and any("seed 3" in w for w in decide(per)["why"]))
    per[5] = {"defined": False}
    chk("an undefined seed -> UNDEFINED", decide(per)["verdict"] == "UNDEFINED")
    shards = {s: synthetic_seed("perfect") for s in DEV_SEEDS}
    out = score_shards(shards)
    chk("score_shards: both probes PASS with preconditions attached",
        out["override"]["verdict"] == "PASS" and out["graded"]["verdict"] == "PASS"
        and out["override"]["preconditions"] and not out["graded"]["undefined_reasons"])
    shards[2] = synthetic_seed("graded_flat")
    chk("score_shards: one flat graded seed FAILS graded, override still PASS",
        score_shards(shards)["graded"]["verdict"] == "FAIL" and score_shards(shards)["override"]["verdict"] == "PASS")
    try:
        pool_job_lines("0" * 40, seeds=(42,))
        chk("gate seeds refused", False)
    except SystemExit:
        chk("gate seeds refused", True)
    lines = pool_job_lines("a" * 40)
    chk("16 dev lines (8 seeds x 2 probes), mem_gb=9, max_age_h=72",
        len(lines) == 16 and all("mem_gb=9" in w and "max_age_h=72" in w for _c, w in lines)
        and all("memcap.sh 9 --" in c for c, _w in lines))
    ok = all(c for _n, c in checks)
    print("SELFTEST %s" % ("OK" if ok else "FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help="internal: run one session in THIS process")
    ap.add_argument("--shard", action="store_true", help="run one (seed, probe) session in a fresh subprocess")
    ap.add_argument("--score", default=None, help="shard root to score")
    ap.add_argument("--organ-feed", action="store_true", help="diagnostic: the same facts into a standalone organ")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-pool-jobs", default=None, metavar="SHA", help="print the dev pool job lines (TSV)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated dev seeds")
    ap.add_argument("--probe", default=None, choices=PROBES)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = tuple(int(s) for s in a.seeds.split(",")) if a.seeds else DEV_SEEDS
    for s in seeds:
        _refuse_gate_seed(s)
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.emit_pool_jobs:
        for cmd, why in pool_job_lines(a.emit_pool_jobs, seeds=seeds):
            print("%s\t%s" % (cmd, why))
        return
    if a.score:
        out = score(a.score, seeds)
        dest = a.out or os.path.join(a.score, "probes_score.json")
        with open(dest, "w") as f:
            json.dump(out, f, indent=1, default=str)
        for p in PROBES:
            r = out[p]
            print("C1 probe %s: %s (%s/%s seeds pass)" % (p.upper(), r["verdict"], r.get("n_pass"), len(seeds)))
            for w in (r.get("why") or []) + (r.get("undefined_reasons") or []):
                print("  - " + w)
        print("-> " + dest)
        return
    if a.organ_feed:
        rows = [organ_feed_seed(s) for s in seeds]
        for r in rows:
            b, e, g = r["override_cases"]["bat"], r["override_cases"]["eel"], r["graded"]
            print("seed %d | bat: guess %s lake %.4f -> told %s (pond %.4f lake %.4f) | eel: guess %s -> %s | "
                  "graded %s rising=%s" % (r["seed"], b["guess_before"], b["lake_rate_before"] or 0,
                                           b["organ_decides_told"], b["told_rate_after"] or 0,
                                           b["lake_rate_after"] or 0, e["guess_before"], e["organ_decides_told"],
                                           g.get("rates"), g.get("rising")), flush=True)
        if a.out:
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            with open(a.out, "w") as f:
                json.dump({"runner": "research/runners/c1_generalization_probes.py --organ-feed", "plan": PLAN,
                           "note": "DIAGNOSTIC, not the bar: standalone organ fed the probe facts, no brain",
                           "rows": rows}, f, indent=1, default=str)
        return
    if a.worker or a.shard:
        if a.seed is None or a.probe is None or a.out is None:
            ap.error("--worker/--shard need --seed --probe --out")
        sys.exit((run_worker if a.worker else run_shard)(a.seed, a.probe, a.out))
    ap.error("choose one of --worker/--shard/--score/--organ-feed/--selftest/--emit-pool-jobs")


if __name__ == "__main__":
    main()
