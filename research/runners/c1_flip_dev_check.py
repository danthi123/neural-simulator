"""C1 FLIP DEV CHECK -- dev seeds only: does turning the category-inheritance wiring ON (BRAIN_CATEGORY_INHERIT=1)
change any Stage-1 reply other than the one it is for, and how far does it over-generalize?

WHY (Claude review 2026-10-06, docs/drafts/stage1-c1-inheritance/PLAN.md). The wiring passed its own C1 gate
(research/findings/2026-10-06-stage1-c1-category-inheritance-RESULT.md), but that gate measured only the C1 script.
A default-ON flip acts on EVERY strict recall miss of every conversation, so before the flip we must see, through
the real handler, (1) that on the other Stage-1 groups (C2 `word`, C3 `lived`, the C4 diagnostic `val_novel`, the C1
diagnostic `cat_told`) flag-on changes no reply CONTENT (abstained / recalled_svo / hypothesis_svo) except the C1
inheritance probe, so it never turns an answer into an abstention and never volunteers a guess on a moat; and (2) how
it behaves when a member BRIDGES two categories (organ dev, prereg 2026-09-30 section 12: a bat that eats the bug and
flies in the sky made `what does the canary swim` reach `lake` on 8 of 8 dev seeds at organ level).

WITH THE HOP FIX (RULING c1-hop-first, Claude 2026-10-06): every cell, both sides, runs with
BRAIN_CATEGORY_INHERIT_HOP=1 (the organ's hop attenuation; inert when the wiring is off), and the bridge group
`cat_bridge` is part of the bar: `what does the canary swim` and `what does the robin swim` (two category hops through
the bat) must not change, while `what does the bat swim` may become a FLAGGED guess `lake` (one hop: the bat's own
told `eat bug` is shared with the fish, the same inference C1 measures; reported in `bridge`, decided by the owner).

AMENDMENT B-1 (registered in PLAN.md 2026-10-11, RULING tree-c1-rescore): the DEFINED precondition is the wiring
APPLIED, not every turn traced. A flag-on cell must record the envs (`flip_dev.flag_value` == "1", `flip_dev.hop`
== "1"), carry at least ONE `category_inherit` trace (zero traces = flag-on, wiring never consulted =
UNDEFINED), swallow no organ error, and keep every organ read `hop: true`. A flag-on turn WITHOUT the trace is a
bypass turn: the seed stays DEFINED (the direct route answered without consulting the organ), the scorer REPORTS
those turns per group (count and labels, never silently passed), and the turn is still compared flag-on vs
flag-off: a content change there FAILS the seed (why `untraced_turn_changed`). A QUERY record (carries
`lesion`) must record `hop: true`; the `organ_not_built` DECLINE record never queried the organ and is not
a read to check. Flag-off is unchanged: no turn carries the trace. The bar below ((a), (b), (c), the hop clauses, PASS / UNDEFINED / FAIL) is unchanged.

THE DEV BAR (bar-sanity record: docs/drafts/stage1-c1-inheritance/bar_sanity_record_flipdev.json). Per dev seed, both
sides (flag on / flag off) of every bar group are present and valid (no errored turn, source `tiny-demo +LTM`,
PYTHONHASHSEED=0, corpus present), the flag engaged per Amendment B-1 (>= 1 flag-on turn carries the
`category_inherit` trace -- untraced bypass turns are reported -- with no swallowed organ error; no flag-off turn
carries it). The seed PASSES iff
  (a) correct: cat_disc flag-on names `worm` on the direct probe (stage1_exit_gate.score_cat),
  (b) specific: on every turn of every bar group, flag-on and flag-off reply CONTENT are equal, except the ALLOWED
      turns (cat_disc / cat_told / cat_bridge `p_canary_eat` -> `worm`; cat_bridge `c_bat_swim` -> `lake`), where
      flag-on may only add a FLAGGED guess (hypothesis_svo, never recalled_svo) naming that patient, never turn an
      answer into an abstention;
  (c) the fix keeps the inheritance: cat_bridge flag-on `p_canary_eat` IS the flagged worm guess (a tie fails).
Every cell must record the hop env (`flip_dev.hop` == "1"), every flag-on organ read must carry `hop: true`, and the
flag-on cat_disc / cat_bridge cells must hold that read on the canary probe.
Dev bar PASS iff all 8 dev seeds (1-8) are defined and pass; any undefined seed -> UNDEFINED (re-run that cell once);
otherwise FAIL naming each violation. Deterministic per seed (OU noise off), so one violating seed is a defect.

LANE: one pool job per (seed, side, group), numpy, mem_gb 7 / memcap 9 (the Stage-1 figure). Dev seeds only: this
module refuses every gate seed (42/43/44/100/101/102). No sim/ or webapp/ edit.

    python -m research.runners.c1_flip_dev_check --emit-pool-jobs <SHA> [--seeds 1,...,8]      # TSV: cmd<TAB>why
    python -m research.runners.c1_flip_dev_check --shard --seed 3 --side on --group word --out <path>
    python -m research.runners.c1_flip_dev_check --score <root> [--seeds 1,...,8] [--out <json>]
    python -m research.runners.c1_flip_dev_check --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import stage1_exit_gate as SG  # noqa: E402
from tools.verdict import Verdict  # noqa: E402 -- earned preconditions carried into the artifact

DEV_SEEDS = tuple(range(1, 9))
GATE_SEEDS = SG.GATE_SEEDS
OUT_DIR = "research/findings/raw/_stage1_c1_flipdev"
FLAG = "BRAIN_CATEGORY_INHERIT"
SIDES = {"on": "1", "off": "0"}
HOP_FLAG = "BRAIN_CATEGORY_INHERIT_HOP"            # the hop fix (RULING c1-hop-first): "1" on BOTH sides of every cell
HOP_VALUE = "1"
HOP_READ_GROUPS = ("cat_disc", "cat_bridge")       # flag-on cells whose canary probe must record an organ read with hop
BAR_GROUPS = ("cat_disc", "cat_told", "word", "lived", "val_novel", "cat_bridge")
REPORT_GROUPS = ()
# (group, label) -> the patients a flagged flag-on guess may name there (flag-off must abstain)
ALLOWED = {("cat_disc", "p_canary_eat"): {"worm"}, ("cat_told", "p_canary_eat"): {"worm"},
           ("cat_bridge", "p_canary_eat"): {"worm"}, ("cat_bridge", "c_bat_swim"): {"lake"}}
CONTENT_FIELDS = ("abstained", "recalled_svo", "hypothesis_svo")
POOL_MAX_AGE_H = 120

# The bridging-member probe (in the bar since the hop fix; also reported in `bridge`). The C1 context plus a bat that eats the bug and flies in the sky: the
# canary's only context (`fly sky`) is now shared by three birds and one bug-eater. Over-generalization reads:
# `what does the canary swim` / `what does the bat swim` / `what does the robin swim` should abstain (never taught;
# reaching `lake` needs a chain through the bat and a fish), and `what does the canary eat` is the ambiguity read.
CAT_BRIDGE_TURNS = list(SG._CAT_CONTEXT) + [
    ("t_bat_eat", "the bat eats the bug", "teach"),
    ("t_bat_fly", "the bat flies in the sky", "teach"),
    ("t_canary_fly", "the canary flies in the sky", "teach"),
    ("p_canary_eat", "what does the canary eat", "probe"),
    ("c_canary_swim", "what does the canary swim", "control"),
    ("c_bat_swim", "what does the bat swim", "control"),
    ("c_robin_swim", "what does the robin swim", "control"),
]
BRIDGE_MOATS = ("c_canary_swim", "c_bat_swim", "c_robin_swim")


def _register_bridge_group():
    """Adds the bridge group to the gate runner's table. Called only inside a worker process (never at import,
    never in tests), so the Stage-1 runner's own group table is unchanged everywhere else."""
    if "cat_bridge" not in SG.GROUPS:
        SG.GROUPS["cat_bridge"] = {"criterion": "C1-report", "primary": False, "env": {},
                                   "turns": CAT_BRIDGE_TURNS, "permuted_turns": list(CAT_BRIDGE_TURNS),
                                   "lesions": {}, "informational": {}}


def shard_path(root, seed, side, group):
    return os.path.join(root, side, "s%d" % int(seed), "%s__intact_a.json" % group)


# ══════════════════════════════════════════════ RUN (one fresh process per cell) ═══════════════════════════════════
def _refuse_gate_seed(seed):
    if int(seed) in GATE_SEEDS:
        raise SystemExit("c1_flip_dev_check: seed %d is a GATE seed; this module runs dev seeds only" % int(seed))


def run_worker(seed, side, group, out_path):
    _refuse_gate_seed(seed)
    _register_bridge_group()
    os.environ[FLAG] = SIDES[side]
    os.environ[HOP_FLAG] = HOP_VALUE
    rc = SG.run_worker(seed, group, "intact_a", out_path)
    with open(out_path) as f:
        d = json.load(f)
    d["runner"] = "research/runners/c1_flip_dev_check.py"
    d["flip_dev"] = {"side": side, "flag": FLAG, "flag_value": os.environ.get(FLAG), "group": group,
                     "hop": os.environ.get(HOP_FLAG)}
    with open(out_path, "w") as f:
        json.dump(d, f, indent=1, default=str)
    return rc


def run_shard(seed, side, group, out_path):
    _refuse_gate_seed(seed)
    env = dict(os.environ)
    env.update(SG.BASE_ENV)
    env[FLAG] = SIDES[side]
    env[HOP_FLAG] = HOP_VALUE
    part = out_path + ".part.json"
    cmd = [sys.executable, "-u", "-m", "research.runners.c1_flip_dev_check", "--worker", "--seed", str(int(seed)),
           "--side", side, "--group", group, "--out", part]
    p = subprocess.run(cmd, env=env, cwd=_REPO)
    if p.returncode != 0 or not os.path.exists(part):
        print("[c1 flipdev shard] worker FAILED rc=%s seed=%s %s/%s" % (p.returncode, seed, side, group), flush=True)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    os.replace(part, out_path)
    try:
        os.remove(part + ".prov.json")
    except OSError:
        pass
    return 0


def pool_job_lines(sha, seeds=DEV_SEEDS, out_dir=OUT_DIR, groups=BAR_GROUPS):
    out = []
    for seed in seeds:
        _refuse_gate_seed(seed)
        for group in groups:
            for side in SIDES:
                p = shard_path(out_dir, seed, side, group)
                cmd = ("cd ~/derisk-pool/revisions/%s && bash tools/mem_ok.sh 7 1 && mkdir -p %s && "
                       "bash tools/memcap.sh 9 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
                       "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m "
                       "research.runners.c1_flip_dev_check --shard --seed %d --side %s --group %s --out %s"
                       % (sha, os.path.dirname(p), seed, side, group, p))
                why = ("C1 flip dev check WITH the hop fix (%s), DEV seed %d, %s flag %s, %s=1, group %s; record: C1 "
                       "wiring gate PASS 2026-10-06, hop fix organ dev 8/8 (RULING c1-hop-first), flip specificity "
                       "across the Stage-1 groups unmeasured; mem_gb=7; max_age_h=%d"
                       % (sha[:9], seed, FLAG, side, HOP_FLAG, group, POOL_MAX_AGE_H))
                out.append((cmd, why))
    return out


# ══════════════════════════════════════════════ SCORING (pure functions over shard dicts) ══════════════════════════
def _trace(resp):
    return resp.get("category_inherit") if isinstance(resp, dict) else None


def _organ_errors(shard):
    out = []
    for label, r in (shard.get("responses") or {}).items():
        tr = _trace(r)
        if not isinstance(tr, dict):
            continue
        if tr.get("error"):
            out.append((label, "trace: %s" % tr["error"]))
        ob, rd = tr.get("observe"), tr.get("read")
        if isinstance(ob, dict) and ob.get("error"):
            out.append((label, "observe: %s" % ob["error"]))
        if isinstance(rd, dict) and str(rd.get("reason") or "").startswith("error"):
            out.append((label, "read: %s" % rd.get("reason")))
    return out


def _content(resp):
    p = SG._primary(resp)
    return None if p is None else {k: p[k] for k in CONTENT_FIELDS}


def _cell_problems(shard, side):
    """Validity + engagement problems of one cell (empty list = valid and engaged)."""
    if shard is None:
        return ["missing"]
    if not isinstance(shard, dict) or "_load_error" in shard:
        return ["unreadable"]
    probs = []
    resp = shard.get("responses") or {}
    if not resp:
        probs.append("no_responses")
    if any(isinstance(r, dict) and "_error" in r for r in resp.values()):
        probs.append("errored_turn")
    if shard.get("sources") != [SG.EXPECTED_SOURCE]:
        probs.append("source:%s" % shard.get("sources"))
    if shard.get("pythonhashseed_ok") is not True:
        probs.append("pythonhashseed")
    if shard.get("corpus_present") is not True:
        probs.append("corpus_absent")
    if (shard.get("flip_dev") or {}).get("hop") != HOP_VALUE:
        probs.append("hop_not_set")
    has = [isinstance(_trace(r), dict) for r in resp.values()]
    if side == "on":
        if not any(has):                                # B-1: >= 1 traced turn; zero traces = never consulted
            probs.append("flag_on_not_engaged")
        if (shard.get("flip_dev") or {}).get("flag_value") != SIDES["on"]:   # B-1 precondition (1): flag recorded
            probs.append("flag_not_recorded")
        if _organ_errors(shard):
            probs.append("organ_error")
        reads = [(_trace(r) or {}).get("read") for r in resp.values()]
        # a QUERY record carries `lesion` and must record hop; the `organ_not_built` DECLINE record
        # (no lesion) never queried the organ, so it is not a read to check.
        if any(isinstance(rd, dict) and "lesion" in rd and rd.get("hop") is not True for rd in reads):
            probs.append("hop_not_engaged")             # B-1: every organ read must record the hop
        elif (shard.get("flip_dev") or {}).get("group") in HOP_READ_GROUPS:
            rd = (_trace(resp.get("p_canary_eat")) or {}).get("read")
            if not (isinstance(rd, dict) and rd.get("hop") is True):
                probs.append("hop_not_engaged")         # the canary probe must leave an organ read that records hop
    elif any(has):
        probs.append("flag_off_carries_trace")
    return probs


def seed_verdict(on, off, groups=BAR_GROUPS):
    """on / off: {group: shard dict or None}. Returns the per-seed record (see module doc for the rule)."""
    undefined = []
    for g in groups:
        for side, cells in (("on", on), ("off", off)):
            for pr in _cell_problems((cells or {}).get(g), side):
                undefined.append("%s/%s:%s" % (side, g, pr))
    # B-1: report the flag-on bypass turns per group (count and labels) -- never silently passed.
    untraced = {}
    for g in groups:
        labs = [l for l, r in (((on.get(g) or {}).get("responses")) or {}).items()
                if not isinstance(_trace(r), dict)]
        if labs:
            untraced[g] = labs
    if undefined:
        return {"defined": False, "pass": None, "undefined": undefined, "untraced": untraced}
    violations, allowed_changes = [], []
    for g in groups:
        ra, rb = on[g]["responses"], off[g]["responses"]
        if list(ra) != list(rb):
            return {"defined": False, "pass": None, "undefined": ["%s:turn_labels_differ" % g],
                    "untraced": untraced}
        for label in ra:
            ca, cb = _content(ra[label]), _content(rb[label])
            if ca == cb:
                continue
            traced = isinstance(_trace(ra[label]), dict)
            rec = {"group": g, "label": label, "on": ca, "off": cb, "traced": traced}
            if not traced:                              # B-1: a bypass turn's reply must not change at all
                rec["why"] = "untraced_turn_changed"
                violations.append(rec)
            elif (g, label) in ALLOWED:
                hyp = ca.get("hypothesis_svo")
                ok = (cb.get("abstained") is True and ca.get("abstained") is False and ca.get("recalled_svo") is None
                      and isinstance(hyp, list) and len(hyp) == 3 and str(hyp[2]).lower() in ALLOWED[(g, label)])
                (allowed_changes if ok else violations).append(rec)
            else:
                rec["why"] = "answer_lost" if (ca.get("abstained") and not cb.get("abstained")) else "content_changed"
                violations.append(rec)
    sc = SG.score_cat(on["cat_disc"]["responses"])
    sc_off = SG.score_cat(off["cat_disc"]["responses"])
    correct = bool(sc["direct_correct"])
    bridge_correct = None
    if "cat_bridge" in groups:                      # the fix keeps the inheritance in the bridge session too
        pb = SG._primary(on["cat_bridge"]["responses"].get("p_canary_eat")) or {}
        hyp = pb.get("hypothesis_svo")
        bridge_correct = bool(pb.get("abstained") is False and pb.get("recalled_svo") is None and isinstance(hyp, list)
                              and len(hyp) == 3 and str(hyp[2]).lower() == "worm")
    ok = correct and bridge_correct is not False and not violations
    return {"defined": True, "pass": bool(ok), "untraced": untraced, "correct": correct,
            "bridge_correct": bridge_correct,
            "off_inherits": bool(sc_off["inherits"]), "violations": violations, "allowed_changes": allowed_changes,
            "cat_disc_on": sc}


def bridge_report(on_shard, off_shard):
    """REPORTED (the bar reads the same cells through seed_verdict): what the bridging member does."""
    if _cell_problems(on_shard, "on") or _cell_problems(off_shard, "off"):
        return {"defined": False, "on_problems": _cell_problems(on_shard, "on"),
                "off_problems": _cell_problems(off_shard, "off")}
    ra, rb = on_shard["responses"], off_shard["responses"]
    out = {"defined": True, "canary_eat_on": SG._patient(ra.get("p_canary_eat")),
           "canary_eat_off": SG._patient(rb.get("p_canary_eat")), "moat_guesses_on": {}, "moat_guesses_off": {},
           "bat_facts_observed": []}
    for m in BRIDGE_MOATS:
        out["moat_guesses_on"][m] = None if SG._abstained(ra.get(m)) else SG._patient(ra.get(m))
        out["moat_guesses_off"][m] = None if SG._abstained(rb.get(m)) else SG._patient(rb.get(m))
    for lab in ("t_bat_eat", "t_bat_fly"):
        tr = _trace(ra.get(lab)) or {}
        out["bat_facts_observed"].append(bool(tr.get("observe")))
    out["n_moat_guesses_on"] = sum(1 for v in out["moat_guesses_on"].values() if v)
    return out


def decide_dev(per_seed, seeds=DEV_SEEDS):
    """The dev bar over the per-seed records: PASS iff every listed seed is defined and passes."""
    recs = [per_seed.get(int(s)) for s in seeds]
    if any(r is None or not r.get("defined") for r in recs):
        bad = [int(s) for s, r in zip(seeds, recs) if r is None or not r.get("defined")]
        return {"verdict": "UNDEFINED", "undefined_seeds": bad, "n_pass": None}
    n_pass = sum(1 for r in recs if r["pass"])
    v = "PASS" if n_pass == len(recs) else "FAIL"
    why = []
    for s, r in zip(seeds, recs):
        if not r["correct"]:
            why.append("seed %d: canary does not inherit worm" % int(s))
        if r.get("bridge_correct") is False:
            why.append("seed %d: cat_bridge canary eat is not the flagged worm guess" % int(s))
        for x in r["violations"]:
            why.append("seed %d: %s/%s %s" % (int(s), x["group"], x["label"], x.get("why", "allowed_turn_bad")))
    return {"verdict": v, "n_pass": n_pass, "n_seeds": len(recs), "why": why}


def _load(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"_load_error": repr(e)}


def _earn(res):
    """Attach the `preconditions` block that earned `res["verdict"]` (tools.verdict.Verdict; gate
    verdict-preconditions, ported from tools/iw2_score.py's `_earn`). The step's own logic decides
    PASS/FAIL/UNDEFINED as registered; the block only makes the artifact carry what earned it. A check that
    fails under an asserted PASS/FAIL downgrades the verdict to UNDEFINED - a verdict is never asserted over
    a failed precondition (the affect-eviction case)."""
    per = res.get("per_seed") or {}
    reasons = [str(u) for r in per.values() for u in (r.get("undefined") or [])]

    def _none_of(*subs):
        return not any(s in x for x in reasons for s in subs)

    defined = [bool(r.get("defined")) for r in per.values()]
    checks = [
        ("every seed defined (both sides valid, no errored turn, sources/pythonhashseed/corpus)",
         bool(defined) and all(defined), True),
        ("flag recorded and engaged on every ON cell (flag_value=1, >=1 traced turn)",
         _none_of("flag_not_recorded", "flag_on_not_engaged"), True),
        ("every organ query read records hop:true", _none_of("hop_not_engaged", "hop_not_set"), True),
        ("flag-off cells carry no trace", _none_of("flag_off_carries_trace"), True),
        ("no organ errors and no errored turns", _none_of("organ_error", "errored_turn"), True),
        # NB (2026-10-11, driver fix on research/c1-flip): the bar OUTCOME is deliberately NOT a precondition --
        # a defined seed failing the bar is a FAIL (B-1 registered), and making "every seed passes" a require()
        # check was circular: it downgraded FAIL to UNDEFINED (broke test_one_moat_leak_on_one_seed_fails, which
        # already failed on main at 32db95f88). Preconditions are data-quality only; decide_dev owns PASS/FAIL.
    ]
    v = Verdict("c1 flip-dev dev bar (Amendment B-1)")
    for name, measured, expect in checks:
        v.require(name, measured, expect=expect)
    d = v.decide(go=(res.get("verdict") == "PASS"), verbose=False)
    if res.get("verdict") in ("PASS", "FAIL") and d["status"] == "UNDEFINED":
        res["verdict_unearned"] = res["verdict"]
        res["verdict"] = "UNDEFINED"
    res["preconditions"] = d["preconditions"]
    res["undefined_reasons"] = d["undefined_reasons"]
    return res


def score(root, seeds=DEV_SEEDS):
    per_seed, bridge = {}, {}
    for s in seeds:
        on = {g: _load(shard_path(root, s, "on", g)) for g in BAR_GROUPS}
        off = {g: _load(shard_path(root, s, "off", g)) for g in BAR_GROUPS}
        per_seed[int(s)] = seed_verdict(on, off)
        bridge[int(s)] = bridge_report(_load(shard_path(root, s, "on", "cat_bridge")),
                                       _load(shard_path(root, s, "off", "cat_bridge")))
    out = decide_dev(per_seed, seeds)
    nb = [b for b in bridge.values() if b.get("defined")]
    out["bridge_summary"] = {"n_defined": len(nb), "n_seeds_with_moat_guess_on": sum(1 for b in nb
                                                                                      if b["n_moat_guesses_on"]),
                             "canary_eat_on": [b["canary_eat_on"] for b in nb]}
    out.update({"runner": "research/runners/c1_flip_dev_check.py", "root": root, "seeds": [int(s) for s in seeds],
                "per_seed": per_seed, "bridge": bridge, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    _earn(out)
    return out


# ══════════════════════════════════════════════ SELFTEST (no brain) ════════════════════════════════════════════════
def _fake(abstained=True, svo=None, hyp=None, trace=True):
    d = {"answer": "x", "abstained": abstained, "recalled_svo": svo, "hypothesis_svo": hyp,
         "source": SG.EXPECTED_SOURCE}
    if trace:
        d["category_inherit"] = {"observe": None, "read": None}
    return d


def synthetic_cell(group, side, overrides=None):
    """A valid cell in the runner's output format: every turn abstains except teach turns (stored) and the C1
    identity controls; `overrides` maps label -> response dict."""
    turns = CAT_BRIDGE_TURNS if group == "cat_bridge" else (SG.GROUPS[group]["turns"] or [])
    labels = [l for (l, _m, _r) in turns]
    resp = {}
    for l in labels:
        if l.startswith("t_"):
            resp[l] = _fake(abstained=False, svo=["a", "b", "c"], trace=(side == "on"))
        elif l == "c_robin_eat":
            resp[l] = _fake(abstained=False, svo=["robin", "eat", "worm"], trace=(side == "on"))
        elif l == "c_trout_eat":
            resp[l] = _fake(abstained=False, svo=["trout", "eat", "bug"], trace=(side == "on"))
        else:
            resp[l] = _fake(trace=(side == "on"))
    for l, r in (overrides or {}).items():
        r = dict(r)
        if side == "on":
            r.setdefault("category_inherit", {"observe": None, "read": None})
        else:
            r.pop("category_inherit", None)
        resp[l] = r
    if side == "on" and group in HOP_READ_GROUPS and "p_canary_eat" in resp:
        tr = resp["p_canary_eat"].setdefault("category_inherit", {"observe": None, "read": None})
        if tr.get("read") is None:                  # the organ was consulted on the canary probe (abstained)
            tr["read"] = {"patient": None, "reason": "below_floor", "lesion": None, "hop": True}
    return {"responses": resp, "sources": [SG.EXPECTED_SOURCE], "pythonhashseed_ok": True, "corpus_present": True,
            "flip_dev": {"side": side, "flag": FLAG, "flag_value": SIDES[side], "group": group,
                         "hop": HOP_VALUE}}


def synthetic_seed(kind="perfect"):
    """(on, off) cells for one seed of a brain of the given kind (used by the selftest and the bar-sanity spec)."""
    inherit = {"p_canary_eat": dict(_fake(abstained=False, hyp=["canary", "eat", "worm"]), category_inherit={
        "observe": None, "read": {"patient": "worm", "reason": "reached", "lesion": None, "hop": True}})}
    off = {g: synthetic_cell(g, "off") for g in BAR_GROUPS}
    on = {g: synthetic_cell(g, "on") for g in BAR_GROUPS}
    if kind in ("perfect", "perfect_told", "leak_moat", "answer_lost", "as_fact", "generative_too", "perfect_bat_lake",
                "bridge_lake", "bridge_robin_lake", "hop_off", "hop_not_engaged", "bridge_tie",
                "bypass_untraced", "bypass_changed"):
        on["cat_disc"] = synthetic_cell("cat_disc", "on", inherit)
        on["cat_bridge"] = synthetic_cell("cat_bridge", "on", inherit)
    if kind == "perfect_bat_lake":
        on["cat_bridge"] = synthetic_cell("cat_bridge", "on", dict(inherit, c_bat_swim=_fake(
            abstained=False, hyp=["bat", "swim", "lake"])))
    if kind == "bridge_lake":
        on["cat_bridge"] = synthetic_cell("cat_bridge", "on", dict(inherit, c_canary_swim=_fake(
            abstained=False, hyp=["canary", "swim", "lake"])))
    if kind == "bridge_robin_lake":
        on["cat_bridge"] = synthetic_cell("cat_bridge", "on", dict(inherit, c_robin_swim=_fake(
            abstained=False, hyp=["robin", "swim", "lake"])))
    if kind == "hop_off":
        for cells in (on, off):
            for c in cells.values():
                c["flip_dev"]["hop"] = None
    if kind == "hop_not_engaged":
        on["cat_disc"]["responses"]["p_canary_eat"]["category_inherit"]["read"]["hop"] = None
    if kind == "bridge_tie":                        # the bridge session's canary eat abstains (tie), as flag-off
        on["cat_bridge"] = synthetic_cell("cat_bridge", "on")
        on["cat_bridge"]["responses"]["p_canary_eat"]["category_inherit"]["read"] = {
            "patient": None, "reason": "tie", "lesion": None, "hop": True}
    if kind in ("bypass_untraced", "bypass_changed"):   # B-1: the direct route answers bypass turns untraced
        for g in ("word", "lived", "val_novel"):
            resp = on[g]["responses"]
            labs = list(resp)
            for l in labs[1:]:
                resp[l].pop("category_inherit", None)   # every turn but the first loses the trace (>= 1 stays)
    if kind == "bypass_changed":                        # B-1: an untraced turn whose reply changes
        resp = on["word"]["responses"]
        labs = [l for l in resp if "category_inherit" not in resp[l]]
        resp[labs[0]] = _fake(abstained=False, hyp=["otter", "chase", "clam"], trace=False)
    if kind == "perfect_told":
        on["cat_told"] = synthetic_cell("cat_told", "on", inherit)
    if kind == "wrong_category":
        on["cat_disc"] = synthetic_cell("cat_disc", "on",
                                        {"p_canary_eat": _fake(abstained=False, hyp=["canary", "eat", "bug"])})
    if kind == "leak_moat":
        on["word"] = synthetic_cell("word", "on",
                                    {"c_otter_chase": _fake(abstained=False, hyp=["otter", "chase", "clam"])})
    if kind == "answer_lost":
        off["lived"] = synthetic_cell("lived", "off",
                                      {"p_wolf_bite": _fake(abstained=False, svo=["wolf", "bite", "apple"])})
    if kind == "as_fact":
        on["cat_disc"] = synthetic_cell("cat_disc", "on",
                                        {"p_canary_eat": _fake(abstained=False, svo=["canary", "eat", "worm"])})
    if kind == "generative_too":
        on["cat_disc"] = synthetic_cell("cat_disc", "on", dict(inherit, p_canary_might=_fake(
            abstained=False, hyp=["canary", "eat", "worm"])))
    if kind == "not_engaged":
        on = {g: synthetic_cell(g, "off") for g in BAR_GROUPS}
    return on, off


def selftest(verbose=True):
    checks = []

    def chk(name, cond):
        checks.append((name, bool(cond)))
        if verbose:
            print("  [%s] %s" % ("ok" if cond else "FAIL", name))

    v = seed_verdict(*synthetic_seed("perfect"))
    chk("perfect seed passes (canary guess on cat_disc and cat_bridge)",
        v["defined"] and v["pass"] and len(v["allowed_changes"]) == 2)
    chk("a flagged worm guess on cat_told is allowed", seed_verdict(*synthetic_seed("perfect_told"))["pass"])
    chk("no inheritance fails (correctness)", seed_verdict(*synthetic_seed("null"))["pass"] is False)
    chk("bug for the canary fails", seed_verdict(*synthetic_seed("wrong_category"))["pass"] is False)
    chk("a guess on a C2 moat fails", seed_verdict(*synthetic_seed("leak_moat"))["pass"] is False)
    chk("an answer turned into an abstention fails", seed_verdict(*synthetic_seed("answer_lost"))["pass"] is False)
    chk("worm stated as a recalled FACT fails", seed_verdict(*synthetic_seed("as_fact"))["pass"] is False)
    chk("a changed generative probe fails", seed_verdict(*synthetic_seed("generative_too"))["pass"] is False)
    chk("flag-on with no trace is UNDEFINED", seed_verdict(*synthetic_seed("not_engaged"))["defined"] is False)
    chk("a flagged lake guess for the bat (one hop) is allowed", seed_verdict(*synthetic_seed("perfect_bat_lake"))["pass"])
    chk("lake for the canary through the bat fails", seed_verdict(*synthetic_seed("bridge_lake"))["pass"] is False)
    chk("lake for the robin through the bat fails", seed_verdict(*synthetic_seed("bridge_robin_lake"))["pass"] is False)
    chk("a cell without the hop env is UNDEFINED", seed_verdict(*synthetic_seed("hop_off"))["defined"] is False)
    chk("an organ read without hop is UNDEFINED", seed_verdict(*synthetic_seed("hop_not_engaged"))["defined"] is False)
    chk("a bridge-session canary eat that ties fails", seed_verdict(*synthetic_seed("bridge_tie"))["pass"] is False)
    v = seed_verdict(*synthetic_seed("bypass_untraced"))
    chk("B-1: untraced bypass turns with >=1 trace are DEFINED", v["defined"])
    chk("B-1: the untraced turns are reported per group (count and labels)",
        set(v.get("untraced") or {}) == {"word", "lived", "val_novel"}
        and all(v["untraced"].values()))
    v = seed_verdict(*synthetic_seed("bypass_changed"))
    chk("B-1: an untraced turn whose reply changes FAILS the seed",
        v["defined"] and v["pass"] is False
        and any(x.get("why") == "untraced_turn_changed" for x in v["violations"]))
    on, off = synthetic_seed("perfect")
    on["word"]["responses"]["p_bear_chase"]["category_inherit"]["read"] = {"patient": None,
        "reason": "bypass", "lesion": None, "admitted": []}   # query-shaped, no hop key
    chk("B-1: an organ read without hop is UNDEFINED", seed_verdict(on, off)["defined"] is False)
    on, off = synthetic_seed("perfect")
    on["word"]["flip_dev"]["flag_value"] = "0"
    chk("B-1: a cell without the recorded flag value is UNDEFINED", seed_verdict(on, off)["defined"] is False)
    on, off = synthetic_seed("perfect")
    on["word"] = None
    chk("a missing cell is UNDEFINED", seed_verdict(on, off)["defined"] is False)
    per = {s: seed_verdict(*synthetic_seed("perfect")) for s in DEV_SEEDS}
    chk("8/8 perfect -> PASS", decide_dev(per)["verdict"] == "PASS")
    per[3] = seed_verdict(*synthetic_seed("leak_moat"))
    chk("7/8 -> FAIL", decide_dev(per)["verdict"] == "FAIL")
    per[5] = {"defined": False}
    chk("an undefined seed -> UNDEFINED", decide_dev(per)["verdict"] == "UNDEFINED")
    try:
        pool_job_lines("0" * 40, seeds=(42,))
        chk("gate seeds refused", False)
    except SystemExit:
        chk("gate seeds refused", True)
    lines = pool_job_lines("a" * 40)
    chk("96 dev lines (6 bar groups incl. the bridge, 2 sides, 8 seeds)", len(lines) == 96)
    chk("every line names its side's flag value via --side", all("--side on" in c or "--side off" in c
                                                                 for c, _w in lines))
    ok = all(c for _n, c in checks)
    print("SELFTEST %s" % ("OK" if ok else "FAILED"))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help="internal: run one cell in THIS process")
    ap.add_argument("--shard", action="store_true", help="run one (seed, side, group) cell in a fresh subprocess")
    ap.add_argument("--score", default=None, help="shard root to score")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-pool-jobs", default=None, metavar="SHA", help="print the dev pool job lines (TSV)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated dev seeds")
    ap.add_argument("--side", default=None, choices=sorted(SIDES))
    ap.add_argument("--group", default=None, choices=list(BAR_GROUPS + REPORT_GROUPS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = tuple(int(s) for s in a.seeds.split(",")) if a.seeds else DEV_SEEDS
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.emit_pool_jobs:
        for cmd, why in pool_job_lines(a.emit_pool_jobs, seeds=seeds):
            print("%s\t%s" % (cmd, why))
        return
    if a.score:
        out = score(a.score, seeds)
        dest = a.out or os.path.join(a.score, "flipdev_score.json")
        with open(dest, "w") as f:
            json.dump(out, f, indent=1, default=str)
        print("C1 flip dev check: %s (%s/%s seeds pass); bridge moat guesses on %s of %s defined seeds -> %s"
              % (out["verdict"], out.get("n_pass"), len(seeds), out["bridge_summary"]["n_seeds_with_moat_guess_on"],
                 out["bridge_summary"]["n_defined"], dest))
        for w in out.get("why") or []:
            print("  - " + w)
        return
    if a.worker or a.shard:
        if a.seed is None or a.side is None or a.group is None or a.out is None:
            ap.error("--worker/--shard need --seed --side --group --out")
        fn = run_worker if a.worker else run_shard
        sys.exit(fn(a.seed, a.side, a.group, a.out))
    ap.error("choose one of --worker/--shard/--score/--selftest/--emit-pool-jobs")


if __name__ == "__main__":
    main()
