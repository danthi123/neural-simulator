"""bar-sanity spec: the C3 topic-stop GATE decision (G1-G10 and DEC of
research/findings/2026-10-28-c3-topic-stop-gate-PREREGISTRATION.md) under the real scorer,
`research.runners.c3_topic_stop_score.score(..., mode="gate")`, over synthetic six-gate-seed cell sets.

PERFECT brains are built from the REAL dev-battery shards (research/findings/raw/_c3_topic_stop/dev, 79c6f5128): each gate
seed takes the sixteen cells of a counted dev seed drawn at random (the realistic seed-to-seed variation: u_toad_2 asked
or not, Lpinned / pconst vectors), relabelled to the gate seed, with sidecars rewritten as clean gate-seed measurements of
one synthetic pin. `perfect_goat_acquired` adds the case the dev data never showed (the goat teach acquired, so the moat
rate rests on three moats). NULL brains each remove one thing the gate claims, on every seed or on one seed only (the
6-of-6 need), or break one validity rule (a missing cell, a sidecar off the pin, a missing engagement trace, a flag-off
arm that is not off, a sidecar without the prereg): DEC must not read PASS for any of them. `random` asks at random.
Copy-on-write: a generator replaces only the shard dicts and responses it changes; the templates are never mutated."""
import copy
import json
import os
import sys

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from research.runners import c3_topic_stop_score as S  # noqa: E402

NAME = "c3-topic-stop gate DEC"
DESCRIPTION = ("C3 topic-stop gate: DEC PASS over six gate seeds (G1-G10, need 6/6 each, the D4 attribution, the "
               "UNDEFINED rules); perfect = the real dev shards relabelled (and with the goat teach acquired); nulls = "
               "each capability removed on every seed or on one seed, each validity rule broken; random asks.")
SCORER_FILES = ["research/runners/c3_topic_stop_score.py", "research/runners/stage2_exit_gate.py",
                "tools/answer_rate.py", "docs/drafts/c3-topic-stop-gate/bar_sanity_spec_c3ts_gate.py"]
PIN = "f" * 40
SEEDS = S.GATE_SEEDS
_AB = "I don't know about that."
_TPL = {}


def _templates():
    if not _TPL:
        root = os.path.join(_ROOT, S.DEV_DIR)
        for s in S.DEV_COUNTED:
            for tag in S.TAGS:
                with open(S.cell_path(root, s, tag)) as f:
                    _TPL[(s, tag)] = json.load(f)
    return _TPL


def _prov(gs, tag):
    arm = S.TAGS[tag][0]
    out = "%s/s%d/ask__%s.json" % (S.GATE_DIR, gs, tag)
    env = {"SIM_BACKEND": "numpy"}
    env.update(S.env_of(tag))
    return {"git_sha": PIN, "git_dirty": False, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
            "source_manifest_verified_at_exit": True, "env": env,
            "argv": ["research/runners/stage2_exit_gate.py", "--shard", "--seed", str(gs), "--group", "ask", "--arm", arm,
                     "--allow-gate-seed", "--prereg", S.PREREG, "--out", out]}


def _base(rng):
    """{(gate seed, tag): cell} from a random counted dev seed per gate seed (shallow: shard dicts are new, the
    responses dict objects are the templates' until a generator replaces them)."""
    tpl = _templates()
    cells = {}
    for gs in SEEDS:
        ds = S.DEV_COUNTED[int(rng.integers(len(S.DEV_COUNTED)))]
        for tag in S.TAGS:
            sh = dict(tpl[(ds, tag)])
            sh["seed"] = gs
            cells[(gs, tag)] = {"shard": sh, "prov": _prov(gs, tag), "missing": None, "load_error": None}
    return cells


def _set(cells, gs, tag, label, **fields):
    """Replace one response (copy-on-write) with `fields` merged over it."""
    sh = dict(cells[(gs, tag)]["shard"])
    resp = dict(sh["responses"])
    r = dict(resp[label])
    r.update(fields)
    resp[label] = r
    sh["responses"] = resp
    cells[(gs, tag)] = dict(cells[(gs, tag)], shard=sh)


def _resp(cells, gs, tag, label):
    return cells[(gs, tag)]["shard"]["responses"][label]


def _ask(cells, gs, tag, label, on):
    topic = S.G.ASK_PROBE_TOPIC.get(label, "it")
    text = (_AB + " My curiosity is piqued — I haven't learned about %s yet: what can you tell me about %s?"
            % (topic, topic)) if on else _AB
    _set(cells, gs, tag, label, answer=text, abstained=True, recalled_svo=None)


def _set_ts(cells, gs, tag, label, fn):
    """Edit one response's curiosity_topic_stop trace (deep-copied first)."""
    r = copy.deepcopy(_resp(cells, gs, tag, label))
    fn(r["curiosity"]["curiosity_topic_stop"])
    _set(cells, gs, tag, label, curiosity=r["curiosity"])


def decide(data):
    res = S.score(data, SEEDS, "gate", PIN)
    return res["DEC"]["verdict"] == "PASS"


# ── perfect ───────────────────────────────────────────────────────────────────────────────────────────────────────
def perfect(rng):
    return _base(rng)


def perfect_goat_acquired(rng):
    """The goat teach acquired in every registered-order arm (the same reply in all, so D7/AR are unaffected), and the
    taught goat not asked: the moat rate rests on three moats instead of two."""
    cells = _base(rng)
    for gs in SEEDS:
        for tag, (arm, _e, _k) in S.TAGS.items():
            if arm == "permuted":
                continue
            _set(cells, gs, tag, "t_goat", answer="the goat eats the hay", abstained=False,
                 recalled_svo=["goat", "eat", "hay"], supporting_facts=[["goat", "eat", "hay"]])
            if S.G.asked(_resp(cells, gs, tag, "m_goat")):
                _ask(cells, gs, tag, "m_goat", False)
    return cells


# ── nulls: a capability removed on every seed ─────────────────────────────────────────────────────────────────────
def null_no_stop(rng):
    """The unlearnable gap is asked on every presentation (no per-topic stop) in every C session."""
    cells = _base(rng)
    for gs in SEEDS:
        for tag in ("C_intact_a", "C_intact_b", "C_reversed"):
            for t in S.TOADS:
                _ask(cells, gs, tag, t, True)
    return cells


def null_stop_order_carried(rng):
    """The stop holds in the registered order only."""
    cells = _base(rng)
    for gs in SEEDS:
        for t in S.TOADS:
            _ask(cells, gs, "C_reversed", t, True)
    return cells


def null_learnable_not_asked(rng):
    cells = _base(rng)
    for gs in SEEDS:
        for tag in ("C_intact_a", "C_intact_b", "C_reversed"):
            _ask(cells, gs, tag, "lp_ibex_1", False)
    return cells


def null_moats_asked(rng):
    """Asks every topic it cannot answer, taught or not (no discrimination)."""
    cells = _base(rng)
    for gs in SEEDS:
        for tag in ("C_intact_a", "C_intact_b", "C_reversed"):
            for l in ("m_yak", "m_moose", "m_goat"):
                _ask(cells, gs, tag, l, True)
    return cells


def null_nondeterministic(rng):
    """Two same-seed builds disagree on an ask."""
    cells = _base(rng)
    for gs in SEEDS:
        l = ("u_toad_2", "g_crab", "m_yak")[int(rng.integers(3))]
        _ask(cells, gs, "C_intact_b", l, not S.G.asked(_resp(cells, gs, "C_intact_b", l)))
    return cells


def null_permuted_inert(rng):
    """The asks ignore the exposure: under the permuted teaching the brain still asks the registered gaps."""
    cells = _base(rng)
    pairs = (("g_eel", "m_yak"), ("g_crab", "m_moose"), ("g_newt", "m_goat"))
    for gs in SEEDS:
        orig = {l: dict(_resp(cells, gs, "C_permuted", l)) for pr in pairs for l in pr}
        for g, m in pairs:
            _set(cells, gs, "C_permuted", g, **orig[m])
            _set(cells, gs, "C_permuted", m, **orig[g])
    return cells


def null_lesionL_inert(rng):
    """lesion_L leaves the toad asks as they were (the stop is not the topic slices)."""
    cells = _base(rng)
    for gs in SEEDS:
        for order in ("intact_a", "reversed"):
            for t in S.TOADS:
                _ask(cells, gs, "lesionL_" + order, t, bool(S.G.asked(_resp(cells, gs, "C_" + order, t))))
    return cells


def null_lesionL_one_order(rng):
    cells = _base(rng)
    for gs in SEEDS:
        for t in S.TOADS:
            _ask(cells, gs, "lesionL_reversed", t, bool(S.G.asked(_resp(cells, gs, "C_reversed", t))))
    return cells


def null_lesionP_inert(rng):
    """lesion_P changes nothing: the learnable and the unlearnable gap are still asked at first."""
    cells = _base(rng)
    for gs in SEEDS:
        _copy_asks(cells, gs, "C_intact_a", "lesionP_intact_a")
    return cells


def _copy_asks(cells, gs, src, dst):
    """dst asks exactly where src asks, on every probe both arms abstained on (only the follow-up question moves)."""
    for l in S.LAB12:
        if S.G._abstained(_resp(cells, gs, dst, l)) is True and S.G._abstained(_resp(cells, gs, src, l)) is True:
            _ask(cells, gs, dst, l, bool(S.G.asked(_resp(cells, gs, src, l))))


def null_lesion_curiosity_asks(rng):
    cells = _base(rng)
    for gs in SEEDS:
        _ask(cells, gs, "C_lesion_curiosity", "g_eel", True)
    return cells


def null_answer_lost(rng):
    """Flag on, a teach turn the flag-off brain answered now abstains (an answer lost; C3 itself unaffected)."""
    cells = _base(rng)
    for gs in SEEDS:
        _set(cells, gs, "C_reversed", "t_moose", answer=_AB, abstained=True, recalled_svo=None, supporting_facts=None)
    return cells


def null_recalled_differs(rng):
    """Flag on, the recalled fact differs from the flag-off arm's on one answered turn (D7's recalled_svo half)."""
    cells = _base(rng)
    for gs in SEEDS:
        _set(cells, gs, "C_permuted", "c_dog_2", recalled_svo=["dog", "chase", "fish"])
    return cells


def null_lesionP_not_held(rng):
    """lesion_P read back as not held on one probe (reward_us fired)."""
    cells = _base(rng)
    for gs in SEEDS:
        def f(ts):
            ts["phasic_da"]["reward_us_spikes"] = 5
        _set_ts(cells, gs, "lesionP_reversed", "g_eel", f)
    return cells


def null_lesionL_not_held(rng):
    """lesion_L read back as not held on one probe (tau_d back at 12 s)."""
    cells = _base(rng)
    for gs in SEEDS:
        def f(ts):
            ts["held"]["stp_tau_d_ms"] = 12000.0
        _set_ts(cells, gs, "lesionL_intact_a", "u_toad_3", f)
    return cells


# ── nulls: one seed only (the 6-of-6 need) ─────────────────────────────────────────────────────────────────────────
def _one(rng):
    return SEEDS[int(rng.integers(len(SEEDS)))]


def null_one_seed_no_stop(rng):
    cells = _base(rng)
    gs = _one(rng)
    for tag in ("C_intact_a", "C_intact_b", "C_reversed"):
        for t in S.TOADS:
            _ask(cells, gs, tag, t, True)
    return cells


def null_one_seed_lesionP_inert(rng):
    """G5 at 5/6: the dev bar's >= 9/11 threshold would admit this rate (5/6 = 0.83 >= 9/11); the gate's 6/6 does not."""
    cells = _base(rng)
    _copy_asks(cells, _one(rng), "C_intact_a", "lesionP_intact_a")
    return cells


def null_one_seed_answer_lost(rng):
    cells = _base(rng)
    gs = _one(rng)
    _set(cells, gs, "C_intact_a", "t_yak", answer=_AB, abstained=True, recalled_svo=None, supporting_facts=None)
    return cells


# ── nulls: validity (UNDEFINED is never PASS) ─────────────────────────────────────────────────────────────────────
_BARRED = tuple(t for t, (_a, _e, k) in S.TAGS.items() if k != "informational")   # the cells the bars read


def _cell(rng):
    return _one(rng), _BARRED[int(rng.integers(len(_BARRED)))]


def null_missing_cell(rng):
    cells = _base(rng)
    gs, tag = _cell(rng)
    cells[(gs, tag)] = {"shard": None, "prov": None, "missing": "shard missing", "load_error": None}
    return cells


def null_sidecar_off_pin(rng):
    cells = _base(rng)
    gs, tag = _cell(rng)
    cells[(gs, tag)] = dict(cells[(gs, tag)], prov=dict(cells[(gs, tag)]["prov"], git_sha="e" * 40))
    return cells


def null_sidecar_no_prereg(rng):
    cells = _base(rng)
    gs, tag = _cell(rng)
    pv = dict(cells[(gs, tag)]["prov"])
    pv["argv"] = [a for a in pv["argv"] if a not in ("--allow-gate-seed", "--prereg", S.PREREG)]
    cells[(gs, tag)] = dict(cells[(gs, tag)], prov=pv)
    return cells


def null_engagement_missing(rng):
    """An abstained probe of a flag-on arm carries no applied topic-stop trace."""
    cells = _base(rng)
    gs = _one(rng)
    tag = ("C_intact_a", "C_reversed", "C_permuted", "lesionL_intact_a", "lesionP_reversed")[int(rng.integers(5))]

    def f(ts):
        ts["applied"] = False
    _set_ts(cells, gs, tag, "u_toad_1", f)
    return cells


def null_flag_off_not_off(rng):
    """A flag-off arm carries a topic-stop trace (it was not run with the flags off)."""
    cells = _base(rng)
    gs = _one(rng)
    tag = ("off_intact_a", "off_reversed", "off_permuted")[int(rng.integers(3))]
    src = _resp(cells, gs, "C_intact_a", "g_eel")["curiosity"]
    r = copy.deepcopy(_resp(cells, gs, tag, "g_eel"))
    r.setdefault("curiosity", {})["curiosity_topic_stop"] = copy.deepcopy(src["curiosity_topic_stop"])
    _set(cells, gs, tag, "g_eel", curiosity=r["curiosity"])
    return cells


def null_sidecar_env_wrong(rng):
    """A lesion arm's sidecar shows the lesion flag absent (the arm ran as C)."""
    cells = _base(rng)
    gs = _one(rng)
    tag = ("lesionL_intact_a", "lesionP_intact_a", "lesionL_reversed", "lesionP_reversed")[int(rng.integers(4))]
    pv = dict(cells[(gs, tag)]["prov"])
    pv["env"] = {k: v for k, v in pv["env"].items() if not k.endswith("_LESION")}
    cells[(gs, tag)] = dict(cells[(gs, tag)], prov=pv)
    return cells


# ── info: reported, never thresholded ─────────────────────────────────────────────────────────────────────────────
def info_informational_cell_missing(rng):
    """An informational arm (Lpinned / pconst) missing: no bar reads it, so DEC is unaffected (expected 1.0); the
    prereg re-runs it once and reports it, it does not hold the gate."""
    cells = _base(rng)
    info = tuple(t for t, (_a, _e, k) in S.TAGS.items() if k == "informational")
    gs, tag = _one(rng), info[int(rng.integers(len(info)))]
    cells[(gs, tag)] = {"shard": None, "prov": None, "missing": "shard missing", "load_error": None}
    return cells


# ── random ────────────────────────────────────────────────────────────────────────────────────────────────────────
def random(rng):
    """Every abstained probe of every counted arm asks with probability 1/2, independently."""
    cells = _base(rng)
    for gs in SEEDS:
        for tag, (_a, _e, kind) in S.TAGS.items():
            if kind != "counted":
                continue
            sh = cells[(gs, tag)]["shard"]
            for t in sh["turns"]:
                if t["role"] == "probe" and S.G._abstained(sh["responses"].get(t["label"])) is True:
                    _ask(cells, gs, tag, t["label"], bool(rng.integers(2)))
    return cells
