"""Amendment B to the 2026-10-12 validation-battery prereg (research/findings/2026-10-12-validation-battery-prereg.md):
the scorer and harness changes it registers, on synthetic data. No brain build except the small idle-replay circuits
(a few seconds each, CPU), the same ones tests/test_idle_replay_organ.py builds.

  (1) C5: the exposure sessions are label-only battery turns (the default roster is unchanged); the C5 scorer reads
      every clause (C5.1-C5.3, B-C5a, B-C5b, pooled B-C5c); the commit lesion env reaches the organ and is
      byte-identical when unset.
  (2) C5 seed provenance: the scorer reads BRAIN_CHAT_SEED from each sidecar; a missing or wrong seed is UNDEFINED;
      seed 42 is reported NON-BLIND and still counted.
  (3) LBF: tom-false-belief is a host-input row in b2d_score (keyed by faculty name, whatever lb.json says), the row
      module, the harness aggregate and lb_shard: denominator only, never the numerator.
  (4) transitive-chat engagement: a cell without the chase trace, or with an error, is UNDEFINED.
"""
import argparse
import json
import os
import sys

import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# never put tools/ itself on sys.path: tools/experiment.py would shadow the `experiment` package sim/bridge.py imports
from tools import b2d_score as bd  # noqa: E402
from tools import c5_battery_score as C  # noqa: E402
from tools import lb_shard  # noqa: E402

SPEC_PATH = os.path.join(_ROOT, "docs", "drafts", "idle-replay", "bar_sanity_spec_c5_battery_amendB.py")
RECORD_PATH = os.path.join(_ROOT, "docs", "drafts", "idle-replay", "bar_sanity_record_c5_battery_amendB.json")


def _spec():
    from tools.bar_sanity import load_spec
    return load_spec(SPEC_PATH)


def _rng(seed=0):
    import numpy as np
    return np.random.default_rng(seed)


def _score_draw(S, data, sessions=None):
    per = {s: C.score_seed(s, data[s], sessions=sessions or S.SESSIONS, pin=S.PIN, rb_pin=S.RB_PIN) for s in S.SEEDS}
    return per, C.aggregate(per, S.SEEDS)


# ── (1) the battery turns ────────────────────────────────────────────────────────────────────────────────────────
def test_exposure_sessions_are_label_only_and_the_default_roster_is_unchanged():
    from research.runners import onebrain_regression_battery as B
    from research.runners.self_initiated_production_organ import is_selfinit_trigger
    default = {t[0] for t in B.PROBE_TURNS}
    assert len(B.PROBE_TURNS) == 26
    for name, sess in B.C5B_SESSIONS.items():
        labels = sess["labels"]
        assert not set(labels) & default, name
        turns = [B._TURN_BY_LABEL[lab] for lab in labels]
        assert len({t[2] for t in turns}) == 1 and turns[0][3] is True and not any(t[3] for t in turns[1:])
        assert labels[-1] == sess["idle_label"] and is_selfinit_trigger(turns[-1][1])
        assert not any(is_selfinit_trigger(t[1]) for t in turns[:-1])
        assert len(sess["taught"]) == 4
    st, pm = B.C5B_SESSIONS["stagger"]["taught"], B.C5B_SESSIONS["rotated"]["taught"]
    assert sorted(st) == sorted(pm) and st != pm                     # the same facts, another order
    assert not ({tuple(f) for f in st[-2:]} & {tuple(f) for f in pm[-2:]})   # disjoint two-newest sets
    assert B.C5B_SESSIONS == C.c5b_sessions()                        # the scorer reads the runner's own registry


def test_neutral_idle_line_matches_the_organ():
    from research.runners.self_initiated_production_organ import idle_fallback_text
    assert C.NEUTRAL_IDLE_LINE == idle_fallback_text()


# ── (1) the commit lesion env: reaches the organ, byte-identical when unset ─────────────────────────────────────
def _organ_conversation():
    from research.runners import idle_replay_battery as IB
    from research.runners import idle_replay_organ as IR
    sys.path.insert(0, os.path.join(_ROOT, "tests"))
    from test_idle_replay_organ import _conversation
    return _conversation(IB.CHAINS["intact"]), IR


def test_commit_lesion_env_unset_is_byte_identical_to_explicit_off(monkeypatch):
    """Two fresh organs (the salience read is cached per organ, so one organ's second call reads `cached`)."""
    (org, chat), IR = _organ_conversation()
    (org2, chat2), _ = _organ_conversation()
    monkeypatch.delenv("BRAIN_IDLE_REPLAY_COMMIT_LESION", raising=False)
    assert IR.commit_lesioned() is False
    a = json.dumps(org.speak(chat, lesion=False, tag_lesion=False), sort_keys=True, default=str)
    b = json.dumps(org2.speak(chat2, lesion=False, tag_lesion=False, commit_lesion=False), sort_keys=True, default=str)
    assert a == b and json.loads(a)["n_utt"] == 1
    monkeypatch.setenv("BRAIN_IDLE_REPLAY_COMMIT_LESION", "0")
    assert IR.commit_lesioned() is False


def test_commit_lesion_env_reaches_the_reply_level_organ_call(monkeypatch):
    (org, chat), IR = _organ_conversation()
    monkeypatch.setenv("BRAIN_IDLE_REPLAY_COMMIT_LESION", "1")
    r = org.speak(chat, lesion=False, tag_lesion=False)
    assert r["n_utt"] == 0 and r["utterance"] is None
    assert r["idle_replay"]["lesions"]["commit"] is True and r["idle_replay"]["winner"] is None
    # the server's idle branch calls the module-level speak with no lesion kwargs, so the env decides
    src = open(os.path.join(_ROOT, "webapp", "server.py"), encoding="utf-8").read()
    assert "si = _IR.speak(cache_key, chat, seed=_brain_chat_seed())" in src
    assert "answer = _SI.self_initiated_text(si) if spoke else _SI.idle_fallback_text()" in src


# ── (1)/(2) the C5 scorer ────────────────────────────────────────────────────────────────────────────────────────
def test_perfect_brain_stands_and_every_clause_is_read():
    S = _spec()
    per, agg = _score_draw(S, S.perfect(_rng(1)))
    assert agg["verdict"] == "C5 STANDS (PROVISIONAL)", agg
    for s in S.SEEDS:
        assert per[s]["state"] == "DEFINED"
        assert set(per[s]["clauses"]) == {"C5.2", "C5.3", "B-C5a", "B-C5b"}
    assert agg["pooled_B-C5c"]["n_sessions"] == 12


def test_seed_42_is_non_blind_and_still_counted():
    S = _spec()
    per, agg = _score_draw(S, S.perfect(_rng(2)))
    assert set(agg["non_blind"]) == {42} and agg["n_seeds"] == 6 and agg["n_defined"] == 6
    assert per[42]["non_blind"] and per[43]["non_blind"] is None


@pytest.mark.parametrize("mutate,reason", [
    (lambda env: env.pop("BRAIN_CHAT_SEED"), "seed-missing:on"),
    (lambda env: env.__setitem__("BRAIN_CHAT_SEED", "42"), "seed-mismatch:on=42"),
    (lambda env: env.__setitem__("BRAIN_IDLE_REPLAY", "1"), "arm-env-extra:on:BRAIN_IDLE_REPLAY"),
    (lambda env: env.__setitem__("BRAIN_IDLE_REPLAY_TAG_LESION", "1"), "arm-env-extra:on:BRAIN_IDLE_REPLAY_TAG_LESION"),
])
def test_seed_provenance_is_read_from_each_sidecar(mutate, reason):
    S = _spec()
    data = S.perfect(_rng(3))
    mutate(data[43]["on"]["prov"]["env"])
    per, agg = _score_draw(S, data)
    assert per[43]["state"] == "UNDEFINED" and reason in per[43]["undefined_reasons"]
    assert agg["verdict"].startswith("C5 UNDEFINED")


def test_wrong_pin_and_missing_sidecar_are_undefined():
    S = _spec()
    data = S.perfect(_rng(4))
    data[44]["stagger"]["prov"]["git_sha"] = "0" * 40
    data[100]["commitlesion"]["prov"] = None
    per, _ = _score_draw(S, data)
    assert any(r.startswith("pin:stagger") for r in per[44]["undefined_reasons"])
    assert "prov-missing:commitlesion" in per[100]["undefined_reasons"]
    assert per[44]["clauses"]["B-C5b"] is None                   # an invalid arm is never read by a clause


def test_rb_arms_are_held_to_the_rb_pin_and_p_arms_to_the_pin():
    S = _spec()
    data = S.perfect(_rng(13))
    per = {s: C.score_seed(s, data[s], sessions=S.SESSIONS, pin=S.PIN) for s in S.SEEDS}   # no rb_pin given
    assert all(any(r.startswith("pin:on_rb") for r in per[s]["undefined_reasons"]) for s in S.SEEDS)
    per = {s: C.score_seed(s, data[s], sessions=S.SESSIONS, pin=S.PIN, rb_pin=S.RB_PIN) for s in S.SEEDS}
    assert C.aggregate(per, S.SEEDS)["verdict"] == "C5 STANDS (PROVISIONAL)"
    data[42]["commitlesion"]["prov"]["git_sha"] = S.RB_PIN           # the commit lesion runs AT THE PIN
    assert "pin:commitlesion=%s" % S.RB_PIN in C.score_seed(42, data[42], sessions=S.SESSIONS, pin=S.PIN,
                                                            rb_pin=S.RB_PIN)["undefined_reasons"]


def test_e3_rb_pair_must_reproduce_the_pin_pair_whole_record():
    S = _spec()
    per, agg = _score_draw(S, S.null_rb_drift(_rng(14)))
    for s in S.SEEDS:
        assert "E3:on_rb-differs" in per[s]["undefined_reasons"] and per[s]["clauses"]["B-C5b"] is None
        assert all(per[s]["sessions"][n]["newest_hit"] is None for n in C.SESSION_ARMS)
    assert agg["verdict"].startswith("C5 UNDEFINED") and agg["pooled_B-C5c"]["undecided"] == 12
    data = S.perfect(_rng(15))
    data[101]["off_rb"]["resp"] = dict(data[101]["off_rb"]["resp"], emo={"answer": "drifted"})
    per, _ = _score_draw(S, data)
    assert "E3:off_rb-differs" in per[101]["undefined_reasons"] and per[102]["detail"]["E3"] is True
    data[102]["on_rb"]["resp"] = None
    per, _ = _score_draw(S, data)
    assert "E3:unavailable" in per[102]["undefined_reasons"]


def test_release_mode_must_be_off_for_every_c5_arm():
    S = _spec()
    per, agg = _score_draw(S, S.null_release_mode_on(_rng(16)))
    assert all("release-mode:on=1" in per[s]["undefined_reasons"] for s in S.SEEDS)
    assert per[42]["clauses"]["C5.3"] is None and agg["verdict"].startswith("C5 UNDEFINED")
    data = S.perfect(_rng(17))
    data[44]["stagger"]["prov"]["env"]["LB_ARM_RENDERER"] = "stub"   # any LB_* key is foreign to a C5 arm
    per, _ = _score_draw(S, data)
    assert "arm-env-extra:stagger:LB_ARM_RENDERER" in per[44]["undefined_reasons"]


def test_replay_not_fired_is_undefined_never_a_pass():
    S = _spec()
    data = S.null_replay_never_fires(_rng(5))
    per, agg = _score_draw(S, data)
    assert all("replay-not-fired:on" in per[s]["undefined_reasons"] for s in S.SEEDS)
    assert agg["verdict"].startswith("C5 UNDEFINED")


def test_lesion_blind_host_fails_the_commit_lesion_clause():
    S = _spec()
    per, agg = _score_draw(S, S.null_host_random_kb(_rng(6)))
    assert agg["verdict"].startswith("C5 REVERT-CANDIDATE")
    assert any(per[s]["clauses"]["B-C5a"] is False for s in S.SEEDS)


def test_neutral_on_reply_passes_c52_but_fails_amendment_b():
    """The hole B closes: replay fires, nothing latches, the neutral line differs from the OFF demo lexicon, so
    the registered C5.2 holds; the commit lesion then changes nothing and the sessions voice nothing."""
    S = _spec()
    per, agg = _score_draw(S, S.null_neutral_on(_rng(7)))
    assert all(per[s]["clauses"]["C5.2"] is True for s in S.SEEDS)
    assert all(per[s]["clauses"]["B-C5a"] is False and per[s]["clauses"]["B-C5b"] is False for s in S.SEEDS)
    assert agg["verdict"].startswith("C5 REVERT-CANDIDATE")


def test_primacy_host_passes_b_c5b_but_fails_pooled_b_c5c():
    S = _spec()
    per, agg = _score_draw(S, S.null_primacy(_rng(8)))
    assert all(per[s]["clauses"]["B-C5b"] is True for s in S.SEEDS)
    assert agg["pooled_B-C5c"]["ok"] is False and "pooled clause B-C5c" in agg["verdict"]


def test_pooled_clause_tolerates_two_misses_not_three():
    S = _spec()
    data = S.perfect(_rng(9))

    def miss(seed, name):
        body = data[seed][name]["resp"][S.SESSIONS[name]["idle_label"]]
        cands = body["self_initiated"]["idle_replay"]["candidates"]
        data[seed][name]["resp"][S.SESSIONS[name]["idle_label"]] = S._idle(cands, len(cands) - 2)
    for s in S.SEEDS:                                            # make every session hit first
        for name in C.SESSION_ARMS:
            body = data[s][name]["resp"][S.SESSIONS[name]["idle_label"]]
            cands = body["self_initiated"]["idle_replay"]["candidates"]
            data[s][name]["resp"][S.SESSIONS[name]["idle_label"]] = S._idle(cands, len(cands) - 1)
    miss(42, "stagger")
    miss(43, "rotated")
    _, agg = _score_draw(S, data)
    assert agg["pooled_B-C5c"]["misses"] == 2 and agg["verdict"] == "C5 STANDS (PROVISIONAL)"
    miss(44, "stagger")
    _, agg = _score_draw(S, data)
    assert agg["pooled_B-C5c"]["ok"] is False and agg["verdict"].startswith("C5 REVERT-CANDIDATE")


def test_untaught_or_out_of_order_encoding_is_undefined():
    S = _spec()
    data = S.perfect(_rng(10))
    body = data[101]["rotated"]["resp"]["c5rt_idle"]
    ir = body["self_initiated"]["idle_replay"]
    ir["candidates"] = [c for c in ir["candidates"] if c["fact"] != ["lynx", "chase", "vole"]]
    per, _ = _score_draw(S, data)
    assert "rotated:not-encoded:lynx-chase-vole" in per[101]["undefined_reasons"]
    data = S.perfect(_rng(10))
    cands = data[102]["stagger"]["resp"]["c5st_idle"]["self_initiated"]["idle_replay"]["candidates"]
    cands[-1]["t_enc"], cands[-2]["t_enc"] = cands[-2]["t_enc"], cands[-1]["t_enc"]
    per, _ = _score_draw(S, data)
    assert any(r.startswith("stagger:t_enc-not-increasing") for r in per[102]["undefined_reasons"])


def test_memory_bar_is_two_gigabytes_decimal():
    assert C.MEM_BAR_KIB == 2 * 10 ** 9 // 1024
    S = _spec()
    per, agg = _score_draw(S, S.null_memory_hog(_rng(11)))
    assert all(per[s]["clauses"]["C5.3"] is False for s in S.SEEDS)


def test_score_dir_reads_the_registered_file_layout(tmp_path):
    S = _spec()
    data = S.perfect(_rng(12))
    for s, arms in data.items():
        for arm, a in arms.items():
            p = C.arm_paths(str(tmp_path), s, arm)
            json.dump(a["resp"], open(p["resp"], "w"))
            json.dump(a["prov"], open(p["prov"], "w"))
            if a["rss"] is not None:
                with open(p["rss"], "w") as fh:
                    fh.write("\n".join(json.dumps(r) for r in a["rss"]) + "\n")
    res = C.score_dir(str(tmp_path), S.SEEDS, pin=S.PIN, sessions=S.SESSIONS, rb_pin=S.RB_PIN)
    assert res["verdict"] == "C5 STANDS (PROVISIONAL)", res["per_seed"]
    os.remove(C.arm_paths(str(tmp_path), 43, "commitlesion")["resp"])
    res = C.score_dir(str(tmp_path), S.SEEDS, pin=S.PIN, sessions=S.SESSIONS, rb_pin=S.RB_PIN)
    assert "arm-missing:commitlesion" in res["per_seed"]["43"]["undefined_reasons"]


def test_bar_sanity_record_is_ok_and_current():
    from tools.bar_sanity import sha256_file
    rec = json.load(open(RECORD_PATH))
    assert rec["verdict"] == "OK" and rec["n"] >= 1000
    for rel, sha in rec["scorer_files"].items():
        assert sha256_file(os.path.join(_ROOT, rel)) == sha, rel


# ── (3) LBF host-input rows ──────────────────────────────────────────────────────────────────────────────────────
PIN = "f" * 40


def _prov(extra_env=None):
    env = {"SIM_BACKEND": "numpy"}
    env.update(extra_env or {})
    return {"git_sha": PIN, "source_kind": "git_archive", "source_manifest_verified_at_start": True,
            "source_manifest_verified_at_exit": True, "env": env, "host": "node-x"}


def _cell(sdir, fac, turn, ia, le, kind="neural-lesion", verdict="regressed"):
    d = os.path.join(sdir, fac)
    os.makedirs(d, exist_ok=True)
    stem = fac.replace("-", "_")
    for name, body in (("intact_a_%s.json" % stem, ia), ("intact_b_%s.json" % stem, ia),
                       ("lesion_%s.json" % stem, le), ("lesion_%s.json.rep0" % stem, le)):
        json.dump(body, open(os.path.join(d, name), "w"))
        json.dump(_prov({"BRAIN_CHAT_SEED": "42"}), open(os.path.join(d, name + ".prov.json"), "w"))
    rep = {"per_faculty": [{"faculty": fac, "kind": kind, "verdict": verdict, "load_bearing": verdict == "regressed",
                            "turn": turn, "diffs": [], "null_control_clean": True, "lesion_reproduced": True}]}
    json.dump(rep, open(os.path.join(d, "lb.json"), "w"))
    json.dump(_prov(), open(os.path.join(d, "lb.json.prov.json"), "w"))
    return d


def _reply(answer, extra=None):
    t = {"answer": answer, "abstained": False, "recalled_svo": None}
    t.update(extra or {})
    return t


def _battery(tmp_path, seeds=(42, 43), tom_kind="neural-lesion", tom_lb=True, trans=None, trans_verdict="regressed"):
    base = tmp_path / "shards"
    for s in seeds:
        sdir = str(base / "tag" / ("s%d" % s))
        _cell(sdir, "row-a", "well", {"well": _reply("yes")}, {"well": _reply("no")})
        _cell(sdir, "row-b", "well", {"well": _reply("same")}, {"well": _reply("same")}, verdict="pass")
        _cell(sdir, "tom-false-belief", "tom_fb", {"tom_fb": _reply("basket")},
              {"tom_fb": _reply("box" if tom_lb else "basket")}, kind=tom_kind,
              verdict="regressed" if tom_lb else "pass")
        if trans is not None:
            ia, le = trans(s)
            _cell(sdir, "reasoning-transitive-chat", "transitive_nonadjacent", ia, le, verdict=trans_verdict)
    return str(base)


def test_host_input_row_is_out_of_the_numerator_in_the_denominator(tmp_path):
    base = _battery(tmp_path)
    res = bd.score(base, "tag", PIN, [42, 43])
    for lvl in ("s0", "s1", "s2"):
        ps = res[lvl]["per_seed"]["42"]
        if lvl == "s1":                                          # no fields file: S1 is UNDEFINED, nothing exercised
            continue
        assert ps["n_exercised"] == 3 and ps["n_load_bearing"] == 1, (lvl, ps)
        assert ps["n_host_input_load_bearing_excluded"] == 1
        assert "tom-false-belief" not in res[lvl]["robust_core"] + res[lvl]["union"]
        assert "tom-false-belief" not in res[lvl]["per_faculty"]
        assert res[lvl]["host_input"]["tom-false-belief"]["n_load_bearing"] == 2
    assert res["manifest"]["s42/tom-false-belief"]["host_input"] is True
    assert res["counts"]["n_coverable_faculties"] == 2 and res["counts"]["n_host_input_cells"] == 2


def test_host_input_registry_never_raises_a_fraction(tmp_path, monkeypatch):
    for tom_lb in (True, False):
        base = _battery(tmp_path / str(tom_lb), tom_lb=tom_lb)
        with_b = bd.score(base, "tag", PIN, [42, 43])["s2"]["mean_fraction"]
        monkeypatch.setattr(bd, "HOST_INPUT_ROWS", {})
        without_b = bd.score(base, "tag", PIN, [42, 43])["s2"]["mean_fraction"]
        monkeypatch.undo()
        assert with_b <= without_b, (tom_lb, with_b, without_b)


def test_retagged_lb_json_kind_is_scored_the_same_way(tmp_path, monkeypatch):
    monkeypatch.setattr(bd, "HOST_INPUT_ROWS", {})
    res = bd.score(_battery(tmp_path, tom_kind="host-input"), "tag", PIN, [42, 43])
    assert res["s2"]["per_seed"]["42"]["n_exercised"] == 3 and res["s2"]["per_seed"]["42"]["n_load_bearing"] == 1
    assert res["s2"]["host_input"]["tom-false-belief"]["n_load_bearing"] == 2
    assert res["verdict"] == "COMPLETE"


def test_row_module_is_retagged_host_input():
    from research.runners import lbf_rows
    from research.runners.lbf_rows import tom_false_belief_chat as m
    assert m.EXTRA_LESIONS["tom-false-belief"]["kind"] == "host-input" and m.selftest()["ok"]
    lesions, probes = {}, []
    lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[m])
    assert lesions["tom-false-belief"]["kind"] == "host-input"
    assert "host-input" in lb_shard.MEASURABLE_KINDS and "host-input" not in lb_shard.COVERABLE_KINDS


def test_harness_aggregate_keeps_host_input_in_the_denominator_only(monkeypatch, tmp_path):
    from research.runners import load_bearing_fraction as lbf
    rows = {"a": ("neural-lesion", "regressed", True), "b": ("neural-lesion", "pass", False),
            "tom-false-belief": ("host-input", "regressed", True)}

    def fake_measure(k, out_dir, repeats=1, intact_cache=None, seed=42):
        kind, verdict, lb = rows[k]
        return {"faculty": k, "kind": kind, "verdict": verdict, "load_bearing": lb, "control_diffs": [],
                "null_control_clean": True}
    monkeypatch.setattr(lbf, "measure_faculty", fake_measure)
    rep = lbf.run(out_dir=str(tmp_path), only=list(rows))
    assert rep["counts"]["n_load_bearing"] == 1 and rep["counts"]["n_host_input_load_bearing"] == 1
    assert rep["load_bearing_fraction"] == pytest.approx(1 / 3)
    assert rep["load_bearing_faculties"] == ["a"] and "tom-false-belief" in rep["host_input_faculties"]
    rows.pop("tom-false-belief")
    rep = lbf.run(out_dir=str(tmp_path), only=list(rows))
    assert rep["load_bearing_fraction"] == pytest.approx(1 / 2) and "n_host_input" not in rep["counts"]


def test_lb_shard_aggregate_keeps_host_input_in_the_denominator_only(tmp_path):
    def cell(fac, kind, lb):
        d = tmp_path / "t" / "s42" / fac
        d.mkdir(parents=True)
        (d / "lb.json").write_text(json.dumps({"per_faculty": [{"faculty": fac, "kind": kind, "verdict": "regressed"
                                                                if lb else "pass", "load_bearing": lb,
                                                                "null_control_clean": True}], "UNRELIABLE": False}))
    cell("a", "neural-lesion", True)
    cell("b", "neural-lesion", False)
    cell("tom-false-belief", "host-input", True)
    out = str(tmp_path / "agg.json")
    lb_shard.cmd_aggregate(argparse.Namespace(tag="t", seeds=[42], pin=None, base=str(tmp_path), out=out))
    agg = json.load(open(out))
    ps = agg["per_seed"]["42"]
    assert ps["n_load_bearing"] == 1 and ps["n_host_input_exercised"] == 1
    assert ps["load_bearing_fraction"] == pytest.approx(1 / 3)
    assert "tom-false-belief" not in agg["per_faculty"] and "tom-false-belief" in agg["host_input_faculties"]


# ── (4) transitive-chat engagement ───────────────────────────────────────────────────────────────────────────────
_HOP = [{"action": "ADVANCE", "committed": "ball", "n_ignited": 1}]


def _ttrace(lesioned, route="chase", error=None, chase=None):
    return {"route": route, "lesioned": lesioned, "error": error, "chase": _HOP if chase is None else chase,
            "path_len": 2}


def _trans(ia_trace, le_trace):
    def make(seed):
        ia = {"transitive_nonadjacent": _reply("yes, cat precedes river", {"transitive_trace": ia_trace})}
        le = {"transitive_nonadjacent": _reply("I don't know", {"transitive_trace": le_trace})}
        if ia_trace is None:
            ia["transitive_nonadjacent"].pop("transitive_trace")
        if le_trace is None:
            le["transitive_nonadjacent"].pop("transitive_trace")
        return ia, le
    return make


def _fields(tmp_path):
    rows = {f: {"kept": ["answer"], "probe_turn": t} for f, t in (
        ("row-a", "well"), ("row-b", "well"), ("tom-false-belief", "tom_fb"),
        ("reasoning-transitive-chat", "transitive_nonadjacent"))}
    p = str(tmp_path / "fields.json")
    json.dump({"rows": rows, "reply_fields": ["answer", "abstained", "recalled_svo"]}, open(p, "w"))
    return p


def test_engaged_transitive_cells_are_defined(tmp_path):
    res = bd.score(_battery(tmp_path, trans=_trans(_ttrace(False), _ttrace(True))), "tag", PIN, [42, 43],
                   fields_path=_fields(tmp_path))
    assert res["engagement"]["s42/reasoning-transitive-chat"] == "engaged"
    # stop-gate row (ruling 2026-10-13): gating + denominator only, never a numerator / robust_core / union
    sg = res["s2"]["stop_gate"]["reasoning-transitive-chat"]
    assert sg["gate"] == "not blocked" and sg["n_not_blocked"] == 2 and sg["per_seed"]["42"] == "not blocked"
    for lvl in ("s0", "s1", "s2"):                                # S1 needs the fields file (kept: answer)
        ps = res[lvl]["per_seed"]["42"]
        assert ps["n_exercised"] == 4 and ps["n_load_bearing"] == 1 and ps["n_stop_gate_load_bearing_excluded"] == 1, \
            (lvl, ps)
        assert res[lvl]["stop_gate"]["reasoning-transitive-chat"]["gate"] == "not blocked"
        assert "reasoning-transitive-chat" not in res[lvl]["per_faculty"]
        assert "reasoning-transitive-chat" not in res[lvl]["robust_core"] + res[lvl]["union"]
    res = bd.score(_battery(tmp_path / "nofields", trans=_trans(_ttrace(False), _ttrace(True))), "tag", PIN, [42, 43])
    assert res["counts"]["n_stop_gate_cells"] == 2 and res["verdict"] == "COMPLETE"
    assert res["stop_gate"]["reasoning-transitive-chat"] == {
        "governing_level": "s0", "gate": "not blocked", "per_seed": {"42": "not blocked", "43": "not blocked"}}


@pytest.mark.parametrize("ia,le,reason", [
    (_ttrace(False), None, "no-record:lesion_"),
    (_ttrace(False), _ttrace(True, chase=[]), "no-hops:lesion_"),
    (_ttrace(False), _ttrace(False), "lesion-flag:lesion_"),
    (_ttrace(True), _ttrace(True), "lesion-flag:intact_"),
    (_ttrace(False), _ttrace(True, route="spiking_reach"), "route-spiking_reach:lesion_"),
    (None, _ttrace(True), "no-record:intact_"),
])
def test_unengaged_transitive_cells_are_undefined(tmp_path, ia, le, reason):
    res = bd.score(_battery(tmp_path, trans=_trans(ia, le)), "tag", PIN, [42, 43])
    st = res["not_defined"]["s42/reasoning-transitive-chat"]["state"]
    assert st.startswith("UNDEFINED:engagement-") and reason in st, st
    sg = res["s2"]["stop_gate"]["reasoning-transitive-chat"]
    assert sg["gate"] == "BLOCKED" and sg["per_seed"]["42"] == "blocked:" + st
    assert res["verdict"] == "INCOMPLETE; STOP-GATE BLOCKED at S0: reasoning-transitive-chat"


@pytest.mark.parametrize("ia,le", [(_ttrace(False), _ttrace(True, error="RuntimeError: build")),
                                   (_ttrace(False, error="ValueError"), _ttrace(True))])
def test_errored_transitive_cells_never_count(tmp_path, ia, le):
    """An error in the trace is caught first by rule 7 (class C); the engagement rule would read error:<arm> too."""
    res = bd.score(_battery(tmp_path, trans=_trans(ia, le)), "tag", PIN, [42, 43])
    assert res["not_defined"]["s42/reasoning-transitive-chat"]["state"] == "C-UNCONFIRMED"
    canon = bd.judge_dir(os.path.join(tmp_path, "shards", "tag", "s42", "reasoning-transitive-chat"),
                         "reasoning-transitive-chat", PIN)
    assert bd.engagement_reason(canon, bd.ENGAGEMENT_ROWS["reasoning-transitive-chat"]).startswith("error:")
    assert res["s2"]["stop_gate"]["reasoning-transitive-chat"]["per_seed"]["42"] == "blocked:C-UNCONFIRMED"


def test_engagement_registry_touches_no_other_row(tmp_path):
    res = bd.score(_battery(tmp_path), "tag", PIN, [42, 43])
    assert "engagement" not in res and res["verdict"] == "COMPLETE"


def test_failing_transitive_cell_blocks_and_never_validates(tmp_path):
    same = _trans(_ttrace(False), _ttrace(True))

    def no_change(seed):
        ia, le = same(seed)
        le["transitive_nonadjacent"]["answer"] = ia["transitive_nonadjacent"]["answer"]
        return ia, le
    base = _battery(tmp_path, trans=no_change, trans_verdict="pass")
    res = bd.score(base, "tag", PIN, [42, 43])
    st = res["s2"]["stop_gate"]["reasoning-transitive-chat"]
    assert st["gate"] == "BLOCKED" and "validated" not in json.dumps(st)
    assert res["verdict"].endswith("STOP-GATE BLOCKED at S0: reasoning-transitive-chat")


def test_stop_gate_registry_never_raises_a_fraction(tmp_path, monkeypatch):
    base = _battery(tmp_path, trans=_trans(_ttrace(False), _ttrace(True)))
    with_b = bd.score(base, "tag", PIN, [42, 43])["s0"]["mean_fraction"]
    monkeypatch.setattr(bd, "ENGAGEMENT_ROWS", {})
    without_b = bd.score(base, "tag", PIN, [42, 43])["s0"]["mean_fraction"]
    assert with_b <= without_b


def test_defined_count_over_all_scored_cells_is_reported(tmp_path):
    res = bd.score(_battery(tmp_path), "tag", PIN, [42, 43])
    assert res["counts"]["n_defined"] == 4                       # neural (coverable) cells: row-a, row-b x 2 seeds
    assert res["counts"]["n_defined_all_scored"] == 6 and res["counts"]["n_scored_cells"] == 6   # + host-input


def test_pre_b_scorer_maps_a_retagged_host_input_row_back_by_name(tmp_path):
    base = _battery(tmp_path, tom_kind="host-input")
    res = bd.score(base, "tag", PIN, [42, 43], amendment_b=False)
    assert "tom-false-belief" in res["s2"]["per_faculty"] and "host_input" not in res["s2"]
    assert res["s2"]["per_seed"]["42"]["n_load_bearing"] == 2
    assert "n_defined_all_scored" not in res["counts"] and "stop_gate" not in res["s2"]


def test_neural_selfid_is_reported_host_substitutable_only(tmp_path):
    base = _battery(tmp_path)
    for s in (42, 43):
        _cell(str(tmp_path / "shards" / "tag" / ("s%d" % s)), "neural-selfid", "iw2si_use",
              {"iw2si_use": _reply("I use spikes")}, {"iw2si_use": _reply("I don't know")})
    res = bd.score(base, "tag", PIN, [42, 43])
    assert res["host_substitutable"]["neural-selfid"]["per_seed"] == {"42": "DEFINED", "43": "DEFINED"}
    assert "neural-selfid" in res["s2"]["per_faculty"]                     # report-only: no count moves
    assert "host_substitutable" not in bd.score(base, "tag", PIN, [42, 43], amendment_b=False)


def test_lb_shard_detects_host_input_rows_by_faculty_name(tmp_path):
    def cell(fac, kind, lb):
        d = tmp_path / "t" / "s42" / fac
        d.mkdir(parents=True)
        (d / "lb.json").write_text(json.dumps({"per_faculty": [{"faculty": fac, "kind": kind, "verdict": "regressed"
                                                                if lb else "pass", "load_bearing": lb,
                                                                "null_control_clean": True}], "UNRELIABLE": False}))
    cell("a", "neural-lesion", True)
    cell("b", "neural-lesion", False)
    cell("tom-false-belief", "neural-lesion", True)              # a shard made at the pin, before the retag
    out = str(tmp_path / "agg.json")
    lb_shard.cmd_aggregate(argparse.Namespace(tag="t", seeds=[42], pin=None, base=str(tmp_path), out=out))
    agg = json.load(open(out))
    assert agg["per_seed"]["42"]["n_load_bearing"] == 1 and agg["per_seed"]["42"]["n_host_input_exercised"] == 1
    assert "tom-false-belief" in agg["host_input_faculties"] and "tom-false-belief" not in agg["per_faculty"]


# ── review fixes 2026-10-13: commit-gate silence, tag lesion, re-tell, the LIFO host ─────────────────────────────
def test_commit_lesion_with_a_gate_spike_is_undefined():
    S = _spec()
    data = S.perfect(_rng(21))
    body = data[44]["commitlesion"]["resp"]["selfinit"]
    body["self_initiated"]["idle_replay"]["selection"]["gate_counts"][0] = 3
    per, agg = _score_draw(S, data)
    assert any(r.startswith("commit-lesion-gate-fired") for r in per[44]["undefined_reasons"])
    assert per[44]["clauses"]["B-C5a"] is None and agg["verdict"].startswith("C5 UNDEFINED")


def test_tag_lesion_must_move_the_voiced_fact():
    S = _spec()
    per, agg = _score_draw(S, S.null_tag_lesion_inert(_rng(22)))
    assert agg["pooled_B-C5d"]["ok"] is False and "B-C5d" in agg["verdict"]
    per, agg = _score_draw(S, S.null_host_newest_lifo(_rng(23)))
    assert agg["pooled_B-C5d"]["ok"] is False and agg["verdict"].startswith("C5 REVERT-CANDIDATE")
    # sessions pass by construction for the LIFO host: the review's confound, now caught only by the tag lesion
    assert all(per[s]["clauses"]["B-C5b"] is True for s in S.SEEDS) and agg["pooled_B-C5c"]["misses"] == 0


def test_tag_lesion_validity():
    S = _spec()
    data = S.perfect(_rng(24))
    data[100]["taglesion"]["resp"]["c5st_idle"]["self_initiated"]["idle_replay"]["selection"]["x_applied"][0] = 9.0
    data[101]["taglesion"]["prov"]["env"].pop("BRAIN_IDLE_REPLAY_TAG_LESION")
    per, _ = _score_draw(S, data)
    assert "taglesion:x-not-equal" in per[100]["undefined_reasons"]
    assert "arm-env:taglesion:BRAIN_IDLE_REPLAY_TAG_LESION=None" in per[101]["undefined_reasons"]


def _inplace_retell(S, data):
    for s in S.SEEDS:                                            # the shape a counted re-tell needs (newest tag early)
        c = S._retell_cands(inplace=True)
        data[s]["retell"]["resp"]["c5re_idle"] = S._idle(c, 5)


@pytest.mark.parametrize("counted", [True, False])
def test_retell_counts_only_when_registered_counted(monkeypatch, counted):
    S = _spec()
    monkeypatch.setattr(C, "RETELL_COUNTED", counted)
    data = S.perfect(_rng(25))
    _inplace_retell(S, data)
    for s in S.SEEDS:                                            # the re-tell idle turn voices the LAST slot everywhere
        cands = data[s]["retell"]["resp"]["c5re_idle"]["self_initiated"]["idle_replay"]["candidates"]
        data[s]["retell"]["resp"]["c5re_idle"] = S._idle(cands, len(cands) - 1)
    per, agg = _score_draw(S, data)
    assert agg["pooled_B-C5e"]["n_bad"] == 6 and agg["pooled_B-C5e"]["counted"] is counted
    assert agg["verdict"].startswith("C5 REVERT-CANDIDATE") is counted
    if not counted:
        assert agg["verdict"] == "C5 STANDS (PROVISIONAL)" and "UNDETERMINED" in agg["reading"]
        assert "re-tell arm is informational" in agg["reading"]


def test_retell_as_the_smoke_left_it_is_undecided(monkeypatch):
    """The dev-7 smoke: both re-tells appended a slot, so the newest tag is in the last slot. Counted, every seed would
    be UNDEFINED; registered informational (RETELL_COUNTED False), it never enters the verdict."""
    S = _spec()
    assert C.RETELL_COUNTED is False
    per, agg = _score_draw(S, S.perfect(_rng(26)))
    assert all(per[s]["sessions"]["retell"]["reasons"] == ["retell:newest-in-last-slot"] for s in S.SEEDS)
    assert agg["verdict"] == "C5 STANDS (PROVISIONAL)" and agg["pooled_B-C5e"]["counted"] is False
    monkeypatch.setattr(C, "RETELL_COUNTED", True)
    per, agg = _score_draw(S, S.perfect(_rng(26)))
    assert "retell:newest-in-last-slot" in per[42]["undefined_reasons"] and agg["verdict"].startswith("C5 UNDEFINED")
    data = S.perfect(_rng(26))
    _inplace_retell(S, data)
    per, agg = _score_draw(S, data)
    assert agg["verdict"] == "C5 STANDS (PROVISIONAL)" and "counted re-tell separates" in agg["reading"]
    assert "UNDETERMINED" in agg["reading"]                      # even a counted re-tell never settles ownership


def test_dev7_retell_smoke_appended_both_slots():
    d = json.load(open(os.path.join(_ROOT, "research/findings/raw/_regression_battery/c5val_1012_dev/c5_s7_retell.json")))
    ir = d["c5re_idle"]["self_initiated"]["idle_replay"]
    why, hit, det = C.retell_read(d["c5re_idle"], C.c5b_sessions()["retell"])
    assert why == ["newest-in-last-slot"] and det["newest_tagged"] == ["lynx", "chase", "frog"]
    assert {tuple(c["fact"]): c["kb_idx"] for c in ir["candidates"]}[("lynx", "chase", "vole")] == 5


def test_lifo_host_fails_amendment_b_overall():
    S = _spec()
    per, agg = _score_draw(S, S.null_host_newest_lifo(_rng(27)))
    assert agg["verdict"].startswith("C5 REVERT-CANDIDATE")
    lifo_aware = getattr(S, "null_host_newest_lifo_lesion_aware", None) or S.info_host_newest_lifo_lesion_aware
    per, agg = _score_draw(S, lifo_aware(_rng(28)))
    assert agg["verdict"].startswith("C5 REVERT-CANDIDATE") is bool(C.RETELL_COUNTED)
    assert not hasattr(S, "info_host_fifo_lesion_aware")          # renamed: it was newest/LIFO, not FIFO


def test_lb_shard_keeps_stop_gate_rows_out_of_the_numerator(tmp_path):
    def cell(fac, lb):
        d = tmp_path / "t" / "s42" / fac
        d.mkdir(parents=True)
        (d / "lb.json").write_text(json.dumps({"per_faculty": [{"faculty": fac, "kind": "neural-lesion", "verdict":
                                                                "regressed" if lb else "pass", "load_bearing": lb,
                                                                "null_control_clean": True}], "UNRELIABLE": False}))
    cell("a", True)
    cell("b", False)
    cell("reasoning-transitive-chat", True)
    out = str(tmp_path / "agg.json")
    lb_shard.cmd_aggregate(argparse.Namespace(tag="t", seeds=[42], pin=None, base=str(tmp_path), out=out))
    agg = json.load(open(out))
    assert agg["per_seed"]["42"]["n_load_bearing"] == 1 and agg["per_seed"]["42"]["n_exercised"] == 3
    assert agg["per_seed"]["42"]["load_bearing_fraction"] == pytest.approx(1 / 3)
    assert "reasoning-transitive-chat" not in agg["per_faculty"] and "reasoning-transitive-chat" not in agg["robust_core"]
    assert agg["union_n"] == 1 and agg["stop_gate_faculties"]["reasoning-transitive-chat"]["n_load_bearing"] == 1


def test_silent_tag_lesion_is_never_moved_and_decode_failure_is_undefined():
    S = _spec()
    data = S.perfect(_rng(31))
    for s_ in S.SEEDS:                                           # every tag-lesion arm latches nothing
        c = data[s_]["taglesion"]["resp"]["c5st_idle"]["self_initiated"]["idle_replay"]["candidates"]
        data[s_]["taglesion"]["resp"]["c5st_idle"] = S._idle(c, None, lesion_tag=True)
    per, agg = _score_draw(S, data)
    assert all(per[s_]["sessions"]["taglesion"]["moved"] is None for s_ in S.SEEDS)
    assert agg["pooled_B-C5d"]["ok"] is None and agg["verdict"].startswith("C5 UNDEFINED")
    assert agg["pooled_B-C5d"]["n_moved"] == 0
    data = S.perfect(_rng(32))
    ir = data[43]["taglesion"]["resp"]["c5st_idle"]["self_initiated"]["idle_replay"]
    ir["decode_matches_winner"] = False                          # a winner that does not decode
    per, _ = _score_draw(S, data)
    assert "taglesion:decode-failed" in per[43]["undefined_reasons"]


def test_host_tag_argmax_passes_every_counted_clause_so_ownership_is_undetermined():
    S = _spec()
    per, agg = _score_draw(S, S.info_host_tag_argmax(_rng(33)))
    assert agg["verdict"] == "C5 STANDS (PROVISIONAL)"
    assert "UNDETERMINED" in agg["reading"] and "race-steer" in agg["reading"]
    assert "owns content selection" not in agg["reading"].replace("Ownership of content selection", "")
    per, agg = _score_draw(S, S.null_tag_lesion_silent(_rng(34)))
    assert not agg["verdict"].startswith("C5 STANDS")
