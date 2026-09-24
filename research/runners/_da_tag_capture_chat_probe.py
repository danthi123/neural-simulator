"""DA TAG-AND-CAPTURE IN CHAT: does lesioning the DA gate change the brain's NEXT-DAY answer? (default-off wiring)

QUESTION. The v3 synaptic tag-and-capture ledger (webapp/da_tag_capture.py, GO at runner level, finding
2026-09-23-da-encoding-natural-drive-v3-synaptic-capture-6seed-GO-runner-level.md) is now wired into the live
`/api/brain-chat` store path and the continuous engine's idle/sleep tick behind BRAIN_DA_TAG_CAPTURE
(webapp/da_tag_capture_chat.py). Through the REAL handler (webapp.server.brain_chat, tiny-demo brain, numpy, stub
renderer, no LLM): a fact told inside surprising news -> a night passes through the brain's own idle/sleep tick -> the
fact is asked for. Does the reply depend on the DA->encoding edge (BRAIN_DA_ENCODING_LESION), selectively for the
salient telling, only when the companion is armed, and without touching immediate recall?

ARMS (each a fresh brain build in its own subprocess, `load_bearing_fraction._spawn_arm`, seed via BRAIN_CHAT_SEED;
turn groups in onebrain_regression_battery: datc / datn / datci / datni):
  sal_night_intact_a / _b  salient telling -> night -> recall, companion ON (b = the null-control rebuild)
  sal_night_lesion         the same + BRAIN_DA_ENCODING_LESION=1 (the lesion the battery already uses for this row)
  neu_night_intact / _lesion  plain telling (words introduced first) -> night -> recall, companion ON
  sal_imm_intact / _lesion    salient telling -> recall at once (no night), companion ON
  neu_imm_intact              plain telling -> recall at once, companion ON
  sal_night_off_intact / _lesion  salient -> night -> recall, companion OFF (today's production default)
ON = {BRAIN_DA_TAG_CAPTURE=1, BRAIN_DA_TAG_CAPTURE_CLOCK=turn}; OFF = {BRAIN_DA_TAG_CAPTURE=0, ...CLOCK=turn}.

The gates are pre-registered in research/findings/2026-09-23-da-tag-capture-chat-wire-PREREGISTRATION.md (its own
commit, before any gate-seed run). `grade_seed` below implements them verbatim.

Run one seed (numpy CPU, 10 brain builds, `--workers N` concurrent subprocesses; ALWAYS memcap, gate with mem_ok):
  bash tools/mem_ok.sh 12 && BRAIN_CHAT_SEED unset is fine (the runner sets it per seed):
  tools/memcap.sh 12 -- .venv/bin/python -u -m research.runners._da_tag_capture_chat_probe --seed 42 \
      --out research/findings/raw/_da_tag_capture_chat
Selftest (no brain):   ... --selftest
Aggregate:             ... --aggregate research/findings/raw/_da_tag_capture_chat
Byte-identical off vs the pinned pre-change SHA (two tiny-demo builds, exact sha256):
  tools/memcap.sh 12 -- .venv/bin/python -u -m research.runners._da_tag_capture_chat_probe --offcheck \
      --pinned-sha 36a175534 --out research/findings/raw/_da_tag_capture_chat/offcheck.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))

SEEDS = [42, 43, 44, 100, 101, 102]
FACT = ["cat", "chase", "ball"]
PINNED_SHA = "36a175534"            # merge-base of this branch and origin/main (2026-09-24 Amendment 2, review
                                    #  v2:2a37f2493): was f35196e66, a SHA that PREDATES this branch's own two
                                    #  origin/main merges, so webapp/server.py (+55/-1 between the two SHAs) could
                                    #  make branch-OFF differ from the pinned tree for reasons that are not this
                                    #  branch's change; the merge-base is the correct byte-identical-off reference
ON = {"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_DA_TAG_CAPTURE_CLOCK": "turn"}
OFF = {"BRAIN_DA_TAG_CAPTURE": "0", "BRAIN_DA_TAG_CAPTURE_CLOCK": "turn"}
LES = {"BRAIN_DA_ENCODING_LESION": "1"}
# (arm name, recall label, env)
ARMS = [
    ("sal_night_intact_a", "datc_recall", dict(ON)),
    ("sal_night_intact_b", "datc_recall", dict(ON)),
    ("sal_night_lesion", "datc_recall", {**ON, **LES}),
    ("neu_night_intact", "datn_recall", dict(ON)),
    ("neu_night_lesion", "datn_recall", {**ON, **LES}),
    ("sal_imm_intact", "datci_recall", dict(ON)),
    ("sal_imm_lesion", "datci_recall", {**ON, **LES}),
    ("neu_imm_intact", "datni_recall", dict(ON)),
    ("sal_night_off_intact", "datc_recall", dict(OFF)),
    ("sal_night_off_lesion", "datc_recall", {**OFF, **LES}),
]
LESION_HELD_MAX_RATIO = 0.25        # G6: lesion PRP p_max must stay below 25 % of the intact arm's (the D1 pool's
                                    #  tonic-rate noise floor gives a ~0.1 per turn at DA=0.5; see the prereg)


def outcome(resp):
    """correct / abstain / confab / undefined for one recall-turn response."""
    if not isinstance(resp, dict) or resp.get("_error"):
        return "undefined"
    svo = resp.get("recalled_svo")
    if svo is not None and list(svo) == FACT:
        return "correct"
    if svo is None and resp.get("abstained"):
        return "abstain"
    if svo is not None:
        return "confab"
    return "undefined"


def _tc(resp):
    return (resp or {}).get("da_tag_capture") or {}


def _fact_block(resp):
    """The managed store block holding the fact (the ledger's first managed block on these groups), or None."""
    blocks = _tc(resp).get("blocks") or []
    return blocks[0] if blocks else None


# ── arms ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def run_seed(seed, out_dir, ltm="off", workers=1):
    sys.path.insert(0, _REPO)
    os.environ["BRAIN_CHAT_SEED"] = str(int(seed))       # every arm's worker inherits it (the battery's seed thread)
    if ltm == "off":                                     # the LTM tier is a separate routed store the ledger never
        os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "0"       #  touches; its load does not fit the 15 GB pool nodes (declared)
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners.load_bearing_fraction import _spawn_arm, turn_group
    sdir = os.path.join(out_dir, "seed%d" % int(seed))
    os.makedirs(sdir, exist_ok=True)
    arms = {}

    def _one(spec):
        name, label, env = spec
        grp = turn_group(label)
        r = _spawn_arm(dict(env), grp, os.path.join(sdir, "%s.json" % name))   # a fresh subprocess per arm
        print("[seed %d] arm %s -> %s" % (seed, name, outcome((r or {}).get(label))), flush=True)
        return name, {"label": label, "env": env, "turns": grp, "responses": r}

    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        for name, rec in ex.map(_one, ARMS):
            arms[name] = rec
    res = {"seed": int(seed), "fact": FACT, "arms": {}, "pinned_sha": PINNED_SHA, "ltm": ltm,
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "argv": list(sys.argv), "workers": int(workers)}
    for name, a in arms.items():
        r = a["responses"] or {}
        rec = r.get(a["label"])
        res["arms"][name] = {
            "label": a["label"], "env": a["env"], "turns": a["turns"], "artifact": os.path.join(sdir, name + ".json"),
            "recall_outcome": outcome(rec),
            "recalled_svo": (rec or {}).get("recalled_svo"), "abstained": (rec or {}).get("abstained"),
            "tag_capture_at_recall": {k: _tc(rec).get(k) for k in ("world_t_h", "n_turns", "p", "p_max",
                                                                    "n_managed_blocks", "external_rescales",
                                                                    "external_rewrites", "gamma", "d1_a_go")},
            "fact_block_at_recall": _fact_block(rec),
            "turn_da": [((r.get(t) or {}).get("da_drives") or {}).get("da_level") for t in a["turns"]],
            "turn_a_eff": [(((r.get(t) or {}).get("da_tag_capture") or {}).get("observe") or {}).get("observed", {})
                           .get("a_eff") for t in a["turns"]],
            "errors": [str(v.get("_error"))[:300] for v in r.values() if isinstance(v, dict) and v.get("_error")],
        }
    res["gates"] = grade_seed(res)
    json.dump(res, open(os.path.join(out_dir, "seed%d.json" % int(seed)), "w"), indent=2, default=str)
    print(json.dumps(res["gates"], indent=2, default=str), flush=True)
    return res


# ── gates (pre-registered; see the PREREGISTRATION finding) ─────────────────────────────────────────────────────
def grade_seed(res):
    A = res["arms"]
    o = {k: v["recall_outcome"] for k, v in A.items()}
    g = {}
    # G0 null control: the intact rebuild reproduces the recall decision AND the ledger's own state exactly
    a, b = A["sal_night_intact_a"], A["sal_night_intact_b"]
    g["G0_null_clean"] = bool(o["sal_night_intact_a"] == o["sal_night_intact_b"]
                              and a["recalled_svo"] == b["recalled_svo"] and a["abstained"] == b["abstained"]
                              and a["tag_capture_at_recall"] == b["tag_capture_at_recall"]
                              and a["fact_block_at_recall"] == b["fact_block_at_recall"])
    # P1 precondition: the fact is stored and immediately recallable in BOTH tellings (else the next-day contrast is
    # about encoding, not persistence) -> UNDEFINED, never a pass or a fail
    g["P1_immediate_precondition"] = bool(o["sal_imm_intact"] == "correct" and o["neu_imm_intact"] == "correct")
    # G1 the load-bearing claim: salient told, next day -> intact answers, DA-gate lesion does not
    g["G1_lesion_changes_next_day_reply"] = bool(o["sal_night_intact_a"] == "correct"
                                                 and o["sal_night_lesion"] == "abstain")
    # G2 the lesion spares immediate recall (it acts on persistence, Bethus 2010)
    g["G2_lesion_spares_immediate"] = bool(o["sal_imm_lesion"] == "correct")
    # G3 selectivity: the plainly-told fact is NOT kept overnight (capture follows the salient DA drive)
    g["G3_neutral_not_kept"] = bool(o["neu_night_intact"] == "abstain")
    # G4 companion-off contrast: without the companion the lesion does NOT change the next-day reply
    g["G4_off_lesion_no_change"] = bool(o["sal_night_off_intact"] == o["sal_night_off_lesion"]
                                        and A["sal_night_off_intact"]["recalled_svo"]
                                        == A["sal_night_off_lesion"]["recalled_svo"])
    # G5 no confabulation anywhere
    g["G5_no_confab"] = bool(all(v != "confab" for v in o.values()))
    # G6 the lesion held at measurement: lesion-arm PRP p_max < 25 % of intact (UNDEFINED if intact p_max is 0/None)
    pi = A["sal_night_intact_a"]["tag_capture_at_recall"].get("p_max")
    pl = A["sal_night_lesion"]["tag_capture_at_recall"].get("p_max")
    g["G6_lesion_held"] = (None if not pi else bool(pl is not None and pl < LESION_HELD_MAX_RATIO * pi))
    # ATTRIBUTION (reported; tools.lab): the lesion's next-day reply change vs (i) the null rebuild and (ii) the same
    # lesion with the companion OFF -- what fraction of the change is the lesion, and what fraction needs the companion.
    from tools.lab import attributable_to

    def _ndiff(x, y):
        return int(x["recalled_svo"] != y["recalled_svo"]) + int(x["abstained"] != y["abstained"])
    treat = _ndiff(A["sal_night_intact_a"], A["sal_night_lesion"])
    g["attribution"] = {
        "lesion_vs_null": attributable_to("da-tag-capture lesion vs null rebuild", treat,
                                          _ndiff(A["sal_night_intact_a"], A["sal_night_intact_b"])),
        "needs_companion": attributable_to("da-tag-capture lesion effect ON vs companion OFF", treat,
                                           _ndiff(A["sal_night_off_intact"], A["sal_night_off_lesion"])),
        "treatment_diffs": treat}
    # ISOLATION: gamma / d1_a_go must be identical across every companion-ON arm at this seed, independent of
    # BRAIN_DA_ENCODING_LESION (2026-09-23 fix, review v2:dd14adaf7 -- `ChatTagCapture`'s D1 reader used to share
    # the process-level cache with the production spiking write gain, so a lesion arm's ledger got a DIFFERENT
    # gamma/d1_a_go than an intact arm's purely from build order, not from the lesioned edge). Only checked when
    # gamma data is actually present: a record with no "env" / no "gamma" (a hand-built selftest arm, or an older
    # artifact from before this field existed) is exempt, never penalized for a field it never had.
    on_gammas, on_d1s = [], []
    for name, rec in A.items():
        if (rec.get("env") or {}).get("BRAIN_DA_TAG_CAPTURE") != "1":
            continue
        tc = rec.get("tag_capture_at_recall") or {}
        if tc.get("gamma") is not None:
            on_gammas.append(tc["gamma"])
        if tc.get("d1_a_go") is not None:
            on_d1s.append(tc["d1_a_go"])
    g["G_isolation_gamma_consistent"] = bool((not on_gammas or all(abs(v - on_gammas[0]) < 1e-6 for v in on_gammas))
                                             and (not on_d1s or all(abs(v - on_d1s[0]) < 1e-6 for v in on_d1s)))
    errs = sum(len(v["errors"]) for v in A.values())
    undefined = (not g["G0_null_clean"]) or (not g["P1_immediate_precondition"]) or g["G6_lesion_held"] is None \
        or errs > 0 or any(v == "undefined" for v in o.values()) or not g["G_isolation_gamma_consistent"]
    core = all(g[k] for k in ("G1_lesion_changes_next_day_reply", "G2_lesion_spares_immediate",
                              "G3_neutral_not_kept", "G4_off_lesion_no_change", "G5_no_confab", "G6_lesion_held"))
    g["outcomes"] = o
    g["n_arm_errors"] = errs
    g["seed_verdict"] = "UNDEFINED" if undefined else ("GO" if core else "NO-GO")
    return g


def seed_signflip_p(diffs):
    """One-sided exact sign-flip p over seeds for mean(diff) > 0 (the seed is the unit of replication)."""
    obs = sum(diffs)
    n = len(diffs)
    hits = sum(1 for signs in itertools.product((1, -1), repeat=n) if sum(s * d for s, d in zip(signs, diffs)) >= obs)
    return hits / float(2 ** n)


def aggregate(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "seed*.json"))):
        try:
            rows.append(json.load(open(p)))
        except Exception:
            continue
    # RE-GRADE every row with the CURRENT grade_seed rather than trusting the stored gates.seed_verdict (2026-09-24,
    # review v2:2a37f2493 of commit 5d3810f2d): a seed*.json written before a grading-logic fix lands (e.g. the
    # confounded research/findings/raw/_da_tag_capture_chat/seed42.json, run at c4c62d066 before the D1-reader-
    # isolation fix daa4b382d, whose stored gates read seed_verdict=GO) must not silently count toward n_go / the
    # 6-seed verdict on its stale grade. grade_seed is a pure function of res["arms"], so re-calling it here is
    # idempotent for a row already graded under the current code and corrective for one that is not.
    for r in rows:
        r["gates"] = grade_seed(r)
    verdicts = {r["seed"]: r["gates"]["seed_verdict"] for r in rows}
    n_go = sum(1 for v in verdicts.values() if v == "GO")
    diffs = [int(r["gates"]["outcomes"]["sal_night_intact_a"] == "correct")
             - int(r["gates"]["outcomes"]["sal_night_lesion"] == "correct") for r in rows]
    sal_vs_neu = [int(r["gates"]["outcomes"]["sal_night_intact_a"] == "correct")
                  - int(r["gates"]["outcomes"]["neu_night_intact"] == "correct") for r in rows]
    complete = sorted(verdicts) == sorted(SEEDS)
    if not complete:
        verdict = "INCOMPLETE"
    elif n_go == 6:
        verdict = "GO"
    elif n_go >= 4 and not any(v == "NO-GO" for v in verdicts.values()):
        verdict = "PARTIAL"
    else:
        verdict = "NO-GO"
    out = {"seeds": sorted(verdicts), "seed_verdicts": verdicts, "n_go": n_go, "verdict": verdict,
           "signflip_p_intact_vs_lesion": (seed_signflip_p(diffs) if rows else None),
           "signflip_p_salient_vs_neutral": (seed_signflip_p(sal_vs_neu) if rows else None),
           "diffs_intact_minus_lesion": diffs, "diffs_salient_minus_neutral": sal_vs_neu}
    json.dump(out, open(os.path.join(d, "aggregate.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))
    return out


# ── byte-identical OFF vs the pinned pre-change SHA ─────────────────────────────────────────────────────────────
_OFFCHECK_TURNS = [  # the salient next-day group, verbatim (texts duplicated here so a pinned tree can run it)
    "Guess what, something unbelievable happened at the circus today!",
    "You will never believe this crazy story, it is absolutely amazing!",
    "the cat chases the ball",
    "Everyone in the audience was screaming and laughing in total shock!",
    "Honestly it was the most astonishing spectacle anybody has ever witnessed!",
    "__NIGHT__",
    "what does the cat chase",
]


def offcheck_worker(repo, out):
    """Run the salient next-day group IN-PROCESS against the tree at `repo` with BRAIN_DA_TAG_CAPTURE unset; hash every
    reply and the composer's store synapses. The night is run by calling continuous_engine.tick_idle_sessions at
    now+24h directly (the pinned tree has no world-step label)."""
    import time as _time
    os.chdir(repo)
    sys.path.insert(0, repo)
    for k in ("BRAIN_DA_TAG_CAPTURE", "BRAIN_DA_TAG_CAPTURE_CLOCK", "BRAIN_DA_ENCODING_LESION"):
        os.environ.pop(k, None)
    os.environ.setdefault("SIM_BACKEND", "numpy")
    os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")
    os.environ.setdefault("SIM_DISABLE_LLM", "1")
    from webapp import server as S
    from webapp import continuous_engine as CE
    replies = []
    for i, msg in enumerate(_OFFCHECK_TURNS):
        if msg == "__NIGHT__":
            n = CE.tick_idle_sessions(S._SESSION_MOOD, S._get_affect_organ, now=_time.time() + 24 * 3600.0,
                                      selfinit_getter=S._get_selfinit_organ,
                                      episodic_getter=S._get_episodic_organ_existing,
                                      chat_getter=S._get_chat_existing)
            replies.append({"night_ticked": int(n or 0)})
            continue
        r = S.brain_chat(S.BrainChatRequest(session="offchk", message=msg, brain="tiny-demo", renderer="stub",
                                            rich=False, reset=(i == 0)))
        replies.append(json.loads(r.body))
    chat = S._BRAIN_CHATS.get(("offchk", "tiny-demo", "stub"))
    comp = chat.inner.composer
    store = [(int(p), int(q), complex(w).real, complex(w).imag) for (p, q, w) in comp.store_conns]
    rj = json.dumps(replies, sort_keys=True, default=str)
    sj = json.dumps(store)
    json.dump({"repo": repo, "replies_sha256": hashlib.sha256(rj.encode()).hexdigest(),
               "store_sha256": hashlib.sha256(sj.encode()).hexdigest(), "n_store_conns": len(store),
               "replies": replies}, open(out, "w"), indent=1, default=str)


def _ledger_scenario_hash(repo):
    """The SynapticTagCaptureLedger edit (block_offset / last_w / sync_from_store) must leave the v3 runner's use
    byte-identical: run one deterministic synthetic scenario (a_override drives, no brain) on the tree at `repo`."""
    code = r'''
import sys, json, hashlib, numpy as np
sys.path.insert(0, sys.argv[1])
from webapp.da_tag_capture import SynapticTagCaptureLedger
class C:
    D = 16
    def __init__(self): self.store_conns = []
rng = np.random.default_rng(3)
comp = C(); L = SynapticTagCaptureLedger(5, gamma=30.0)
for i, t in enumerate([0.0, 0.01, 0.02, 0.5]):
    L.observe_turn(t, 30/3600., 0.9, a_override=[0.6, 0.0, 0.8, 0.3][i])
    for k in range(C.D):
        comp.store_conns.append((k + 1, 100 + k, complex(*rng.standard_normal(2))))
    L.on_store(comp, t)
L.advance(comp, 1.0); L.advance(comp, 24.0)
print(hashlib.sha256(json.dumps([(p, q, w.real, w.imag) for (p, q, w) in comp.store_conns]).encode()).hexdigest())
'''
    r = subprocess.run([sys.executable, "-c", code, repo], capture_output=True, text=True, cwd=repo)
    return r.stdout.strip() or ("ERR " + r.stderr[-500:])


def offcheck(pinned_sha, out, ltm="off"):
    if ltm == "off":
        os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "0"     # both trees; the workers inherit it (declared, as run_seed)
    with tempfile.TemporaryDirectory() as td:
        pin = os.path.join(td, "pinned")
        os.makedirs(pin)
        arc = subprocess.run(["git", "-C", _REPO, "archive", pinned_sha], capture_output=True)
        if arc.returncode != 0:
            raise RuntimeError("git archive failed: %s" % arc.stderr[-500:])
        subprocess.run(["tar", "-x", "-C", pin], input=arc.stdout, check=True)
        os.makedirs(os.path.join(pin, "data"), exist_ok=True)
        os.symlink(os.path.realpath(os.path.join(_REPO, "data", "corpus")), os.path.join(pin, "data", "corpus"))
        res = {"pinned_sha": pinned_sha, "branch_repo": _REPO, "ltm": ltm}
        for tag, repo in (("pinned", pin), ("branch", _REPO)):
            o = os.path.join(td, tag + ".json")
            r = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--offcheck-worker", repo, o],
                               capture_output=True, text=True)
            if r.returncode != 0 or not os.path.exists(o):
                raise RuntimeError("offcheck worker %s failed: %s" % (tag, r.stderr[-2000:]))
            res[tag] = json.load(open(o))
            res[tag].pop("replies", None)
            res[tag + "_ledger_scenario_sha256"] = _ledger_scenario_hash(repo)
        res["replies_identical"] = res["pinned"]["replies_sha256"] == res["branch"]["replies_sha256"]
        res["store_identical"] = res["pinned"]["store_sha256"] == res["branch"]["store_sha256"]
        res["v3_ledger_scenario_identical"] = (res["pinned_ledger_scenario_sha256"]
                                              == res["branch_ledger_scenario_sha256"])
        res["byte_identical_off"] = bool(res["replies_identical"] and res["store_identical"]
                                         and res["v3_ledger_scenario_identical"])
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))
    return res


# ── selftest (no brain) ─────────────────────────────────────────────────────────────────────────────────────────
def selftest():
    sys.path.insert(0, _REPO)
    import numpy as np
    from webapp import da_tag_capture_chat as W
    from webapp.da_tag_capture import SynapticTagCaptureLedger
    checks = {}

    class C:
        D = 8

        def __init__(self, n):
            self.store_conns = [(k % 8 + 1, 50 + k % 8, complex(1.0, 0.0)) for k in range(8 * n)]
    # block_offset: pre-existing blocks untouched, a new block managed
    comp = C(2)
    before = list(comp.store_conns)
    L = SynapticTagCaptureLedger(1, gamma=30.0, block_offset=2)
    comp.store_conns += [(k + 1, 60 + k, complex(0.0, 1.0)) for k in range(8)]
    L.on_store(comp, 0.0)
    L.advance(comp, 24.0)
    checks["block_offset leaves build-time blocks untouched"] = comp.store_conns[:16] == before
    checks["new block is managed"] = len(L.blocks) == 1
    # uncaptured block decays to its baseline at 24 h (no drive)
    w = np.array([c[2] for c in comp.store_conns[16:24]])
    checks["uncaptured block ~ baseline at 24 h"] = bool(np.allclose(w, L.blocks[0]["base"], atol=1e-6))
    # sync_from_store: a pure rescale is absorbed, a rewrite re-tags
    comp.store_conns[16:24] = [(p, q, 2.0 * x) for (p, q, x) in comp.store_conns[16:24]]
    s = L.sync_from_store(comp, 24.0)
    checks["external rescale detected"] = s == {"rescaled": 1, "rewritten": 0}
    comp.store_conns[16:24] = [(p, q, complex(3.0, float(i))) for i, (p, q, _x) in enumerate(comp.store_conns[16:24])]
    s = L.sync_from_store(comp, 25.0)
    checks["external rewrite re-tagged"] = s == {"rescaled": 0, "rewritten": 1} and L.blocks[0]["t_w"] == 25.0
    # flag OFF -> every hook is inert
    os.environ.pop("BRAIN_DA_TAG_CAPTURE", None)

    class Chat:
        pass
    ch = Chat()
    checks["off: observe_chat_turn is None"] = W.observe_chat_turn(ch, 42) is None
    checks["off: after_store_chat is None"] = W.after_store_chat(ch) is None
    checks["off: tick_chat is None"] = W.tick_chat(ch) is None
    checks["off: no ledger attached"] = not hasattr(ch, "_da_tag_capture")
    # clock modes
    os.environ["BRAIN_DA_TAG_CAPTURE_CLOCK"] = "turn"
    checks["clock turn mode parses"] = W.clock_mode() == "turn"
    os.environ.pop("BRAIN_DA_TAG_CAPTURE_CLOCK", None)
    checks["clock default is wall"] = W.clock_mode() == "wall"
    # private rng restores the global stream
    st = np.random.get_state()[1].copy()
    with W._private_rng(42, 3):
        np.random.random(10)
    checks["private rng restores global numpy state"] = bool(np.array_equal(st, np.random.get_state()[1]))
    # grading logic can fail: a synthetic seed where the lesion does not change the reply must NOT be GO
    base = {"recalled_svo": FACT, "abstained": False, "tag_capture_at_recall": {"p_max": 0.05},
            "fact_block_at_recall": None, "errors": []}
    arms = {n: dict(base, recall_outcome="correct") for n, _l, _e in ARMS}
    arms["neu_night_intact"] = dict(base, recall_outcome="abstain", recalled_svo=None, abstained=True)
    arms["sal_night_lesion"] = dict(base, recall_outcome="abstain", recalled_svo=None, abstained=True,
                                    tag_capture_at_recall={"p_max": 0.001})
    checks["grade: designed-GO pattern -> GO"] = grade_seed({"arms": arms})["seed_verdict"] == "GO"
    arms2 = dict(arms)
    arms2["sal_night_lesion"] = dict(base, recall_outcome="correct", tag_capture_at_recall={"p_max": 0.001})
    checks["grade: lesion no change -> NO-GO"] = grade_seed({"arms": arms2})["seed_verdict"] == "NO-GO"
    arms3 = dict(arms)
    arms3["neu_night_intact"] = dict(base, recall_outcome="correct")
    checks["grade: neutral kept -> NO-GO"] = grade_seed({"arms": arms3})["seed_verdict"] == "NO-GO"
    arms4 = dict(arms)
    arms4["sal_imm_intact"] = dict(base, recall_outcome="abstain", recalled_svo=None, abstained=True)
    checks["grade: no immediate recall -> UNDEFINED"] = grade_seed({"arms": arms4})["seed_verdict"] == "UNDEFINED"
    arms5 = dict(arms)
    arms5["sal_night_off_lesion"] = dict(base, recall_outcome="abstain", recalled_svo=None, abstained=True)
    checks["grade: off arm changes under lesion -> NO-GO"] = grade_seed({"arms": arms5})["seed_verdict"] == "NO-GO"
    arms6 = dict(arms)
    arms6["sal_night_intact_a"] = dict(base, recall_outcome="correct", tag_capture_at_recall={"p_max": 0.0})
    checks["grade: intact p_max 0 -> UNDEFINED"] = grade_seed({"arms": arms6})["seed_verdict"] == "UNDEFINED"
    checks["signflip 6/6 -> 1/64"] = abs(seed_signflip_p([1] * 6) - 1 / 64.0) < 1e-12
    # G_isolation_gamma_consistent: a record with no env/gamma at all (the synthetic `arms` pattern above) must not
    # be penalized for a field it never had -- the designed-GO pattern must still read GO.
    checks["grade: no gamma data anywhere -> gate exempt, still GO"] = \
        grade_seed({"arms": arms})["G_isolation_gamma_consistent"] is True
    # consistent gamma/d1_a_go across every companion-ON arm, companion-OFF arms excluded (they carry no D1 read) ->
    # the gate passes and the seed reads its ordinary verdict (GO, from the designed-GO base pattern).
    arms_iso_ok = {}
    for name, label, env in ARMS:
        rec = dict(arms[name])
        rec["env"] = dict(env)
        if env.get("BRAIN_DA_TAG_CAPTURE") == "1":
            rec["tag_capture_at_recall"] = dict(rec["tag_capture_at_recall"], gamma=40.0, d1_a_go=0.15)
        arms_iso_ok[name] = rec
    g_iso_ok = grade_seed({"arms": arms_iso_ok})
    checks["grade: isolation gamma consistent across ON arms -> gate passes, GO"] = \
        g_iso_ok["G_isolation_gamma_consistent"] is True and g_iso_ok["seed_verdict"] == "GO"
    # THE FAILING DIRECTION (the gate must be able to fail): one companion-ON arm reads a different gamma than the
    # rest -- the exact confound this fix closes (a lesion arm calibrating differently from an intact arm) -- and
    # the seed must go UNDEFINED even though every G1-G6 outcome is otherwise the designed-GO pattern.
    arms_iso_bad = {k: dict(v) for k, v in arms_iso_ok.items()}
    arms_iso_bad["sal_night_lesion"] = dict(arms_iso_bad["sal_night_lesion"],
                                            tag_capture_at_recall=dict(
                                                arms_iso_bad["sal_night_lesion"]["tag_capture_at_recall"],
                                                gamma=32.8, d1_a_go=0.187))
    g_bad = grade_seed({"arms": arms_iso_bad})
    checks["grade: isolation gamma MISMATCH -> gate fails"] = g_bad["G_isolation_gamma_consistent"] is False
    checks["grade: isolation gamma mismatch -> seed UNDEFINED"] = g_bad["seed_verdict"] == "UNDEFINED"
    # aggregate() must RE-GRADE every seed.json with the CURRENT grade_seed, never trust a stored gates.seed_verdict
    # (2026-09-24, review v2:2a37f2493: the confounded research/findings/raw/_da_tag_capture_chat/seed42.json was
    # written under pre-fix code and its stored gates read seed_verdict=GO, but re-grading its own arms under the
    # isolation-consistency gate above reads UNDEFINED). Build one seed*.json on disk whose STORED gates say GO
    # (as if written by old code) but whose "arms" data is the isolation-mismatch pattern above, and confirm
    # aggregate() reports it as UNDEFINED, not GO.
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _td:
        stale = {"seed": 900, "arms": arms_iso_bad,
                 "gates": {"seed_verdict": "GO", "outcomes": g_iso_ok["outcomes"]}}   # stale, pre-fix-shaped record
        json.dump(stale, open(os.path.join(_td, "seed900.json"), "w"))
        agg = aggregate(_td)
        checks["aggregate: re-grades a stale stored-GO row to the current UNDEFINED verdict"] = \
            agg["seed_verdicts"][900] == "UNDEFINED" and agg["seed_verdicts"][900] != stale["gates"]["seed_verdict"]
    for k, v in checks.items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", k))
    ok = all(checks.values())
    print("VERDICT:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--ltm", choices=["off", "on"], default="off")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="research/findings/raw/_da_tag_capture_chat")
    ap.add_argument("--aggregate")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--offcheck", action="store_true")
    ap.add_argument("--pinned-sha", default=PINNED_SHA)
    ap.add_argument("--offcheck-worker", nargs=2)
    a = ap.parse_args()
    if a.offcheck_worker:
        return offcheck_worker(*a.offcheck_worker)
    if a.selftest:
        return 0 if selftest() else 1
    if a.offcheck:
        return 0 if offcheck(a.pinned_sha, a.out, ltm=a.ltm)["byte_identical_off"] else 1
    if a.aggregate:
        aggregate(a.aggregate)
        return 0
    if a.seed is None:
        ap.error("--seed required")
    run_seed(a.seed, a.out, ltm=a.ltm, workers=a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
