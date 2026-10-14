"""Hop attenuation in the category-inheritance organ (BRAIN_CATEGORY_INHERIT_HOP; default off; 2026-10-06).

The defect: a bridging member (`bat` eats the bug and flies in the sky) made `what does the canary swim` reach `lake`
on 8/8 dev seeds (research/findings/2026-10-06-c1-hop-attenuation-bridging-member.md). The option attenuates the
frame -> frame pathway (HOP_LATERAL_GAIN, at encoding and at the read) and reads one retrieval cycle (HOP_QUERY_STEPS).

CPU (numpy), one DEV seed. (a) the bat case: the canary's swim abstains in both fact orders; (b) the canary's eat still
reaches worm (the handler flags it as a guess: tests/test_category_inherit_organ.py covers the flag); (c) the organ's
controls still behave with the option on (permuted, read lesion, learning lesion, moat, determinism); (d) option off
reproduces the recorded 09-30 organ reads exactly and builds no transmission gate."""
import json
import os

import pytest

os.environ.setdefault("SIM_NO_PROVENANCE", "1")
os.environ.setdefault("SIM_BACKEND", "numpy")

C1_FACTS = [("robin", "eat", "worm"), ("trout", "eat", "bug"), ("robin", "fly", "sky"), ("trout", "swim", "lake"),
            ("sparrow", "eat", "worm"), ("salmon", "eat", "bug"), ("sparrow", "fly", "sky"),
            ("salmon", "swim", "lake"), ("finch", "eat", "worm"), ("finch", "fly", "sky")]
BAT = [("bat", "eat", "bug"), ("bat", "fly", "sky")]
CANARY = [("canary", "fly", "sky")]
DEV_SEED = 3
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_0930 = os.path.join(_ROOT, "research/findings/raw/_stage1_c1_category/dev/organ_dev_seeds1-8.json")


def _organ(facts, hop, lesion=None, seed=DEV_SEED):
    from research.runners.category_inherit_organ import CategoryInheritOrgan
    o = CategoryInheritOrgan(seed=seed, lesion=lesion, hop=hop)
    for f in facts:
        assert o.observe(*f)["observed"]
    return o


def test_env_parse(monkeypatch):
    from research.runners.category_inherit_organ import category_inherit_hop_enabled
    # flip (RULING c1-flip-go): UNSET is the new default -- on. Every EXPLICIT value keeps the 2026-10-06
    # semantics, so `BRAIN_CATEGORY_INHERIT_HOP=0` restores the 09-30 organ byte-identically.
    monkeypatch.delenv("BRAIN_CATEGORY_INHERIT_HOP", raising=False)
    assert category_inherit_hop_enabled() is True
    for v in ("0", "false", "off", "", "no"):
        monkeypatch.setenv("BRAIN_CATEGORY_INHERIT_HOP", v)
        assert category_inherit_hop_enabled() is False, v
    for v in ("1", "true", "on", "YES"):
        monkeypatch.setenv("BRAIN_CATEGORY_INHERIT_HOP", v)
        assert category_inherit_hop_enabled() is True, v


@pytest.fixture(scope="module")
def hop_on():
    return {"c1": _organ(C1_FACTS + CANARY, True), "c1_again": _organ(C1_FACTS + CANARY, True),
            "permuted": _organ(C1_FACTS + [("canary", "swim", "lake")], True),
            "lateral_learn": _organ(C1_FACTS + CANARY, True, lesion="lateral_learn"),
            "bat_first": _organ(C1_FACTS + BAT + CANARY, True),
            "canary_first": _organ(C1_FACTS + CANARY + BAT, True),
            "moat": _organ([("otter", "eat", "clam"), ("bear", "chase", "otter"), ("wolf", "bite", "apple")], True)}


# ── (a) the bat case ────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("order", ["bat_first", "canary_first"])
def test_bat_case_canary_swim_abstains(hop_on, order):
    r = hop_on[order].infer("canary", "swim")
    assert r["patient"] is None and r["reason"] == "below_floor", r
    from research.runners.category_inherit_organ import HOP_QUERY_STEPS
    assert r["hop"] is True and r["query_steps"] == HOP_QUERY_STEPS == 30, r


# ── (b) the inheritance survives ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("order", ["c1", "bat_first", "canary_first"])
def test_canary_eat_still_reaches_worm(hop_on, order):
    r = hop_on[order].infer("canary", "eat")
    assert r["patient"] == "worm" and r["reason"] == "reached", r
    assert r["attribution"]["learned_pathways"] == 1.0, r["attribution"]


# ── (c) the controls with the option on ─────────────────────────────────────────────────────────────────────────
def test_permuted_exposure_is_followed(hop_on):
    assert hop_on["permuted"].infer("canary", "eat")["patient"] == "bug"


def test_read_lesion_abstains_and_restores(hop_on):
    o = hop_on["c1"]
    o.lesion = "all"
    try:
        r = o.infer("canary", "eat")
        assert r["patient"] is None and r["lesion"] == "all" and r["lesion_held"] is True, r
        assert all(v == 0.0 for _p, v in r["admitted"]), r
    finally:
        o.lesion = None
    assert o.infer("canary", "eat")["patient"] == "worm"


def test_learning_lesion_abstains(hop_on):
    assert hop_on["lateral_learn"].infer("canary", "eat")["patient"] is None


def test_moat_shapes_abstain(hop_on):
    o = hop_on["moat"]
    for a, v in (("otter", "chase"), ("clam", "eat"), ("wolf", "chase"), ("bear", "eat")):
        assert o.infer(a, v)["patient"] is None, (a, v)


def test_deterministic(hop_on):
    assert hop_on["c1"].read("canary") == hop_on["c1_again"].read("canary")


def test_the_transmission_gate_holds_the_gain(hop_on):
    from research.runners.category_inherit_organ import TX_LAT, HOP_LATERAL_GAIN
    b = hop_on["c1"].bridge
    assert b._transmission_gate_values[TX_LAT] == HOP_LATERAL_GAIN
    n_lat = hop_on["c1"].n_learn_synapses["lateral"]
    assert len(b._transmission_gate_to_synapses[TX_LAT]) == n_lat     # exactly the frame -> frame synapses


# ── (d) option off = the 09-30 organ ────────────────────────────────────────────────────────────────────────────
def test_off_builds_no_gate_and_reproduces_the_recorded_0930_reads(monkeypatch):
    # flip (RULING c1-flip-go): the hop default is now ON when unset, so pin the pre-flip OFF build explicitly.
    monkeypatch.setenv("BRAIN_CATEGORY_INHERIT_HOP", "0")
    rec = {r["seed"]: r for r in json.load(open(DEV_0930))["rows"]}[DEV_SEED]
    o = _organ(C1_FACTS + CANARY, "env")
    assert o.hop is False and o.query_steps == 60
    assert not getattr(o.bridge, "_transmission_gate_to_synapses", {})
    r = o.infer("canary", "eat")
    assert "hop" not in r and r["admitted"] == rec["intact"]["canary/eat"]["admitted"], (r, rec["intact"])
    b = _organ(C1_FACTS + BAT + CANARY, False)
    for q in ("canary/eat", "canary/swim", "trout/fly"):
        a, v = q.split("/")
        assert b.infer(a, v)["admitted"] == rec["bridge_bat"][q]["admitted"], q


# ── the dev bar's decision (research/runners/category_inherit_organ_dev.hop_dev_verdict) ────────────────────────
def _row(seed, **over):
    def q(p):
        return {"patient": p}
    row = {"seed": seed, "hop": True, "intact": {"canary/eat": q("worm")}, "permuted": {"canary/eat": q("bug")},
           "read_lesion_all": {"canary/eat": {"patient": None, "lesion": "all", "lesion_held": True}},
           "stage1_c2_c3_shapes": {"otter/chase": q(None)},
           "bridge_bat": {"canary/eat": q("worm"), "canary/swim": q(None)},
           "bridge_bat_canary_first": {"canary/eat": q("worm"), "canary/swim": q(None)}}
    row.update(over)
    return row


def test_hop_dev_verdict():
    from research.runners.category_inherit_organ_dev import hop_dev_verdict
    rows = [_row(s) for s in range(1, 9)]
    assert hop_dev_verdict(rows, seeds=tuple(range(1, 9)))["verdict"] == "PASS"
    assert hop_dev_verdict(rows)["verdict"] == "UNDEFINED"            # the default bar is dev 1-16
    lake = rows[:7] + [_row(8, bridge_bat={"canary/eat": {"patient": "worm"}, "canary/swim": {"patient": "lake"}})]
    v = hop_dev_verdict(lake, seeds=tuple(range(1, 9)))
    assert v["verdict"] == "FAIL" and v["n_pass"] == 7 and 8 in v["why"], v
    inert = rows[:7] + [_row(8, read_lesion_all={"canary/eat": {"patient": "worm", "lesion": "all",
                                                                "lesion_held": True}})]
    s8 = tuple(range(1, 9))
    assert hop_dev_verdict(inert, seeds=s8)["verdict"] == "FAIL"
    assert hop_dev_verdict(rows[:7], seeds=s8)["verdict"] == "UNDEFINED"
    assert hop_dev_verdict(rows[:7] + [_row(8, hop=False)], seeds=s8)["verdict"] == "UNDEFINED"


# ── the C1 gate re-measure with the option on (stage1_c1_category --hop; prereg section 13, 2026-10-06) ────────────
def _write_gate(td, hop_env, read_hop):
    from research.runners import stage1_c1_category as R
    for s in R.GATE_SEEDS:
        for arm in R.ARMS:
            d = R._synthetic("pass", arm)
            d["hop_env"] = hop_env
            rd = (d["responses"][R.DIRECT_PROBE].get("category_inherit") or {}).get("read")
            if isinstance(rd, dict) and read_hop:
                rd["hop"] = True
            p = R.shard_path(td, s, arm)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump(d, f)


@pytest.mark.parametrize("hop_env,read_hop,want", [("1", True, "PASS"), (None, True, "UNDEFINED"),
                                                    ("1", False, "UNDEFINED")])
def test_gate_aggregate_requires_the_hop_when_asked(tmp_path, hop_env, read_hop, want):
    from research.runners import stage1_c1_category as R
    _write_gate(str(tmp_path), hop_env, read_hop)
    agg = R.aggregate(str(tmp_path), require_hop=True)
    assert agg["verdict"].startswith(want), (agg["verdict"], agg["why"])
    assert R.aggregate(str(tmp_path))["verdict"] == "PASS"          # without --hop the 09-30 scoring is unchanged


def test_gate_hop_pool_lines():
    from research.runners import stage1_c1_category as R
    plain, hop = R.pool_job_lines("f" * 40), R.pool_job_lines("f" * 40, hop=True)
    assert len(plain) == len(hop) == 54
    assert not any("BRAIN_CATEGORY_INHERIT_HOP" in c for c, _w in plain)
    assert all("BRAIN_CATEGORY_INHERIT_HOP=1 " in c and R.GATE_HOP_OUT_DIR in c and "section 13" in w
               for c, w in hop)
