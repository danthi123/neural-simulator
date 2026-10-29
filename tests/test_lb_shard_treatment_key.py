"""tools/lb_shard.py treatment-key allowlist and cell completeness under a pin (prereg Amendment 2, 2026-10-28).

THE FAILURE (research/findings/2026-10-27-emo-bounded-flip-PREREGISTRATION.md, Amendment 2; research/FAILURE_LOG.md
2026-10-28). The emo-bounded-flip gate's flag-ON row lines inject BRAIN_EMOTION_CATEGORY_BOUNDED=1 (the treatment the
prereg itself requires), and `cell_prov_fails` read every BRAIN_* key outside {BRAIN_CHAT_SEED} as stray, so all six
ON cells failed exactly five sidecars and B8 was unmeasurable under --pin by construction. A second gap: the pin
rule only looked at files that were PRESENT, so a cell missing an arm file (and its sidecar) still passed.

The fixtures mirror a real gate cell (seed 43, --repeats 2): lb.json, intact_a / intact_b of the emocat turn group, the
lesion arm and its .rep0, each with its own sidecar; the ON sidecars carry the key, the OFF ones do not. Both
directions for each rule: the right cell passes, the wrong one is UNDEFINED and says why."""
from __future__ import annotations

import argparse
import json
import os

import pytest

from tools import lb_shard
from research.runners import emo_flip_score as S

PIN = "c" * 40
KEY = S.TREATMENT_KEY
GROUP = "emocat_s1_emocat_s2_emocat_feel"


def _w(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh)


def _prov(env_extra=None):
    env = {"SIM_BACKEND": "numpy", "OMP_NUM_THREADS": "1"}
    env.update(env_extra or {})
    return {"git_sha": PIN, "git_dirty": False, "source_kind": "git_archive",
            "source_manifest_verified_at_start": True, "source_manifest_verified_at_exit": True, "env": env}


def _arm_names(seed):
    sfx = "" if seed == 42 else "_s%d" % seed
    return ["intact_a_%s%s.json" % (GROUP, sfx), "intact_b_%s%s.json" % (GROUP, sfx),
            "lesion_emotion_category%s.json" % sfx, "lesion_emotion_category%s.json.rep0" % sfx]


def build_cell(base, tag, seed, on=True, lb=True, drop=(), drop_sidecar=(), env_override=None):
    """One emotion-category row cell. `on` puts KEY=1 in every sidecar (as the gate's ON lines do). `drop` removes
    arm files (and, with drop_sidecar, their sidecars too); env_override = {filename: {k: v or None}} edits one
    sidecar's env (None deletes the key)."""
    cell = os.path.join(str(base), tag, "s%d" % seed, "emotion-category")
    treat = {KEY: "1"} if on else {}
    rep = {"seed": seed, "repeats": 2, "per_faculty": [{
        "faculty": "emotion-category", "turn": "emocat_feel", "kind": "neural-lesion",
        "flag": "BRAIN_EMOTION_CATEGORY_COMMIT_LESION", "load_bearing": lb, "verdict": "regressed" if lb else "pass",
        "null_control_clean": True, "lesion_reproduced": True, "treatment_diffs": 2 if lb else 0, "control_diffs": 0}]}
    _w(os.path.join(cell, "lb.json"), rep)
    _w(os.path.join(cell, "lb.json.prov.json"), _prov(treat))
    for fn in _arm_names(seed):
        ec = {"category": "joy", "commit_lesioned": fn.startswith("lesion_"), "bounded_flag": on,
              "bounded_env": "1" if on else None, "bounded_readback": on, "oor_count": 0, "oor_count_session": 0}
        if fn not in drop:
            _w(os.path.join(cell, fn), {"emocat_feel": {"answer": "joy.", "emotion_category": ec}})
        if fn not in drop or fn not in drop_sidecar:
            _w(os.path.join(cell, fn + ".prov.json"), _prov(dict(treat, BRAIN_CHAT_SEED=str(seed))))
    for fn, edits in (env_override or {}).items():
        p = os.path.join(cell, fn + ".prov.json" if not fn.endswith(".prov.json") else fn)
        pj = json.load(open(p))
        for k, v in edits.items():
            if v is None:
                pj["env"].pop(k, None)
            else:
                pj["env"][k] = v
        _w(p, pj)
    return cell


def _fails(cell, treatment):
    rep = json.load(open(os.path.join(cell, "lb.json")))
    return lb_shard.cell_prov_fails(cell, PIN, rep["per_faculty"], treatment=treatment)[0]


# ── the failure this pins: the default rule rejects a treatment cell on exactly the treatment key ──────────────────
def test_default_rule_rejects_the_on_cell_on_five_sidecars_for_the_treatment_key_only(tmp_path):
    cell = build_cell(tmp_path, "on", 43, on=True)
    fails = _fails(cell, None)
    assert len(fails) == 5, fails
    assert all(v == ["stray BRAIN_* env key(s): %s" % KEY] for v in fails.values()), fails
    # and the same cell without the key passes the default rule (the OFF cells of the gate)
    assert _fails(build_cell(tmp_path, "off", 43, on=False), None) == {}


# ── (1) ON cell with the key: passes ─────────────────────────────────────────────────────────────────────────────
def test_on_cell_with_the_treatment_key_passes(tmp_path):
    assert _fails(build_cell(tmp_path, "on", 43, on=True), S.TREATMENT_ON) == {}
    assert _fails(build_cell(tmp_path, "on42", 42, on=True), S.TREATMENT_ON) == {}   # seed 42: unsuffixed names


# ── (2) ON cell missing the key, or with another value: UNDEFINED ─────────────────────────────────────────────────
@pytest.mark.parametrize("where", ["lb.json", "lesion_emotion_category_s43.json.rep0", "intact_a_%s_s43.json" % GROUP])
def test_on_cell_missing_the_key_in_any_sidecar_fails(tmp_path, where):
    cell = build_cell(tmp_path, "on", 43, on=True, env_override={where: {KEY: None}})
    fails = _fails(cell, S.TREATMENT_ON)
    assert list(fails) == [where + ".prov.json"]
    assert fails[where + ".prov.json"] == ["treatment key %s absent (want %s='1')" % (KEY, KEY)]


def test_on_cell_with_another_value_fails(tmp_path):
    cell = build_cell(tmp_path, "on", 43, on=True, env_override={"lb.json": {KEY: "0"}})
    assert _fails(cell, S.TREATMENT_ON) == {"lb.json.prov.json": ["treatment key %s='0' (want '1')" % KEY]}
    cell = build_cell(tmp_path, "on_all", 43, on=False)   # the key absent everywhere: every sidecar says so
    fails = _fails(cell, S.TREATMENT_ON)
    assert len(fails) == 5 and all("absent" in v[0] for v in fails.values())


# ── (3) ON cell with a second stray key: UNDEFINED ─────────────────────────────────────────────────────────────────
def test_on_cell_with_a_second_stray_brain_key_fails(tmp_path):
    cell = build_cell(tmp_path, "on", 43, on=True,
                      env_override={"lesion_emotion_category_s43.json": {"BRAIN_AFFECT_LESION": "1"}})
    assert _fails(cell, S.TREATMENT_ON) == {
        "lesion_emotion_category_s43.json.prov.json": ["stray BRAIN_* env key(s): BRAIN_AFFECT_LESION"]}
    # lb.json's own env still allows NONE but the treatment (BRAIN_CHAT_SEED is an arm-only allowance)
    cell = build_cell(tmp_path, "on2", 43, on=True, env_override={"lb.json": {"BRAIN_CHAT_SEED": "43"}})
    assert _fails(cell, S.TREATMENT_ON) == {"lb.json.prov.json": ["stray BRAIN_* env key(s): BRAIN_CHAT_SEED"]}


# ── (4) OFF cell with the key: UNDEFINED; without it: passes ─────────────────────────────────────────────────────
def test_off_cell_with_the_key_fails_and_without_it_passes(tmp_path):
    assert _fails(build_cell(tmp_path, "off", 43, on=False), S.TREATMENT_OFF) == {}
    cell = build_cell(tmp_path, "off_leak", 43, on=False, env_override={"intact_b_%s_s43.json" % GROUP: {KEY: "1"}})
    assert _fails(cell, S.TREATMENT_OFF) == {"intact_b_%s_s43.json.prov.json" % GROUP: [
        "treatment key %s='1' present (this tag must not carry it)" % KEY]}
    cell = build_cell(tmp_path, "off_zero", 43, on=False, env_override={"lb.json": {KEY: "0"}})   # "0" is not "unset"
    assert "present" in _fails(cell, S.TREATMENT_OFF)["lb.json.prov.json"][0]


# ── (5) a missing arm file: UNDEFINED, named ───────────────────────────────────────────────────────────────────────
def test_a_missing_arm_file_fails_completeness_and_is_named(tmp_path):
    rep0 = "lesion_emotion_category_s43.json.rep0"
    ia = "intact_a_%s_s43.json" % GROUP
    les = "lesion_emotion_category_s43.json"
    # the arm file gone, its sidecar left (the shape the landed gate cells on main have)
    cell = build_cell(tmp_path, "a", 43, on=True, drop=(rep0,))
    assert _fails(cell, S.TREATMENT_ON) == {"completeness": ["missing arm file %s (its sidecar is present)" % rep0]}
    # the arm file AND its sidecar gone: only the row's own record can say it is owed
    cell = build_cell(tmp_path, "b", 43, on=True, drop=(ia, les), drop_sidecar=(ia, les))
    assert _fails(cell, S.TREATMENT_ON) == {"completeness": ["missing arm file intact_a_*_s43.json",
                                                             "missing arm file %s" % les]}
    # before Amendment 2 that cell passed the pin: only present files were checked
    assert not any(k != "completeness" for k in _fails(cell, S.TREATMENT_ON))


def test_completeness_follows_the_record_not_a_fixed_list(tmp_path):
    # a lesion that did not reproduce stops the repeat loop: no .rep owed (the cell is not load-bearing anyway)
    cell = build_cell(tmp_path, "norep", 43, on=False, drop=("lesion_emotion_category_s43.json.rep0",),
                      drop_sidecar=("lesion_emotion_category_s43.json.rep0",))
    rep = json.load(open(os.path.join(cell, "lb.json")))
    rep["per_faculty"][0]["lesion_reproduced"] = False
    _w(os.path.join(cell, "lb.json"), rep)
    assert _fails(cell, None) == {}
    # --repeats 3 owes .rep0 and .rep1
    rep["per_faculty"][0]["lesion_reproduced"] = True
    rep["repeats"] = 3
    _w(os.path.join(cell, "lb.json"), rep)
    assert _fails(cell, None) == {"completeness": ["missing arm file lesion_emotion_category_s43.json.rep0",
                                                   "missing arm file lesion_emotion_category_s43.json.rep1"]}
    # an early-return ruler names its own files: the standard arm set is not asked of it
    assert lb_shard._standard_arm_patterns({"faculty": "affect-coloring", "verdict": "regressed",
                                            "lesion_reproduced": True, "measurement_ruler": "x"}, 43, 2) == ()
    assert lb_shard._standard_arm_patterns({"faculty": "x", "verdict": "not-covered:thin",
                                            "lesion_reproduced": None}, 43, 2) == ()


# ── the scorer: only the ON dir passes the key, the OFF dir forbids it, every cell must be complete ──────────────
def _tree(base, on_kw=None, off_kw=None):
    for s in S.GATE_SEEDS:
        build_cell(base, S.LB_ON_TAG, s, on=True, **(on_kw or {}).get(s, {}))
        build_cell(base, S.LB_OFF_TAG, s, on=False, **(off_kw or {}).get(s, {}))
    return str(base)


def test_score_b8_under_pin_reads_the_treatment_tag(tmp_path):
    ok = S.score_b8(_tree(tmp_path / "ok"), pin=PIN)
    assert ok["status"] == S.PASS and ok["n_load_bearing_on"] == 6 and ok["n_load_bearing_off"] == 6
    # an ON cell without the key: UNDEFINED, never a pass
    r = S.score_b8(_tree(tmp_path / "nokey", on_kw={44: dict(env_override={"lb.json": {KEY: None}})}), pin=PIN)
    assert r["status"] == S.UNDEFINED and "treatment key %s absent" % KEY in r["on_cells"]["s44"]["why"]
    # an ON cell with a second stray key: UNDEFINED
    r = S.score_b8(_tree(tmp_path / "stray", on_kw={100: dict(
        env_override={"intact_a_%s_s100.json" % GROUP: {"BRAIN_AFFECT_LESION": "1"}})}), pin=PIN)
    assert r["status"] == S.UNDEFINED and "BRAIN_AFFECT_LESION" in r["on_cells"]["s100"]["why"]
    # an ON cell missing its lesion repeat: UNDEFINED, the file named in full
    r = S.score_b8(_tree(tmp_path / "inc", on_kw={101: dict(drop=("lesion_emotion_category_s101.json.rep0",))}), pin=PIN)
    assert r["status"] == S.UNDEFINED
    assert "incomplete cell: missing arm file lesion_emotion_category_s101.json.rep0" in r["on_cells"]["s101"]["why"]
    # an OFF cell carrying the key is UNDEFINED beside the bar; it does not move the ON bar
    r = S.score_b8(_tree(tmp_path / "offleak", off_kw={42: dict(env_override={"lb.json": {KEY: "1"}})}), pin=PIN)
    assert r["status"] == S.PASS and r["off_cells"]["s42"]["status"] == "UNDEFINED"
    assert "must not carry it" in r["off_cells"]["s42"]["why"]
    # no pin: the pin rule (and with it the treatment rule and completeness) is not applied, as before
    assert S.score_b8(_tree(tmp_path / "nopin", on_kw={44: dict(env_override={"lb.json": {KEY: None}})}))["status"] == S.PASS


# ── (6) the arm's argv --env (applied in-process by the battery worker) is read under a treatment ──────────────────
LESION_KNOB = "BRAIN_EMOTION_CATEGORY_COMMIT_LESION"
BATTERY = "/r/research/runners/onebrain_regression_battery.py"


def _set_argv_env(cell, fn, env_json):
    """Give `fn`'s sidecar the battery worker's argv, `--env <env_json>` (a str: written verbatim)."""
    p = os.path.join(cell, fn + ".prov.json")
    pj = json.load(open(p))
    pj["argv"] = [BATTERY, "--worker", "--env", env_json, "--turns", "emocat_s1,emocat_s2,emocat_feel",
                  "--out", os.path.join(cell, fn)]
    _w(p, pj)


def _real_argv(cell, seed, lesion_env=None):
    """The gate cells' shape: intact arms --env {}, the lesion arm and its .rep0 --env {lesion knob: 1}."""
    for fn in _arm_names(seed):
        _set_argv_env(cell, fn, json.dumps(lesion_env if fn.startswith("lesion_") and lesion_env is not None
                                           else ({LESION_KNOB: "1"} if fn.startswith("lesion_") else {})))
    return cell


def test_argv_env_of_the_real_cell_shape_passes_on_and_off(tmp_path):
    assert _fails(_real_argv(build_cell(tmp_path, "on", 43, on=True), 43), S.TREATMENT_ON) == {}
    assert _fails(_real_argv(build_cell(tmp_path, "off", 43, on=False), 43), S.TREATMENT_OFF) == {}
    assert lb_shard._argv_env({"argv": ["x", "--env={\"A\": \"1\"}"]}) == {"A": "1"}
    assert lb_shard._argv_env({"argv": ["x", "--only", "y"]}) == {}
    assert lb_shard._argv_env({"argv": ["x", "--env", "[1]"]}) is None


def test_a_treatment_key_overridden_in_argv_env_fails(tmp_path):
    """verify-go 2026-10-28: the sidecar's env carries KEY=1 but the lesion arms' argv --env set it to 0, so they ran
    OFF; the old rule read only pj['env'] and passed the cell."""
    cell = _real_argv(build_cell(tmp_path, "on", 43, on=True), 43, lesion_env={LESION_KNOB: "1", KEY: "0"})
    fails = _fails(cell, S.TREATMENT_ON)
    want = ["treatment key %s='0' set by argv --env (want '1')" % KEY]
    assert fails == {"lesion_emotion_category_s43.json.prov.json": want,
                     "lesion_emotion_category_s43.json.rep0.prov.json": want}
    # the same value as the treatment is not an override
    cell = _real_argv(build_cell(tmp_path, "on1", 43, on=True), 43, lesion_env={LESION_KNOB: "1", KEY: "1"})
    assert _fails(cell, S.TREATMENT_ON) == {}
    # OFF: the key must not be set there at any value
    cell = build_cell(tmp_path, "off", 43, on=False)
    _real_argv(cell, 43)
    _set_argv_env(cell, "intact_a_%s_s43.json" % GROUP, json.dumps({KEY: "1"}))
    assert _fails(cell, S.TREATMENT_OFF) == {"intact_a_%s_s43.json.prov.json" % GROUP: [
        "treatment key %s='1' set by argv --env (this tag must not carry it)" % KEY]}


def test_a_stray_key_in_argv_env_fails_and_the_lesion_knob_only_on_a_lesion_arm(tmp_path):
    cell = build_cell(tmp_path, "on", 43, on=True)
    _real_argv(cell, 43)
    _set_argv_env(cell, "intact_b_%s_s43.json" % GROUP, json.dumps({LESION_KNOB: "1"}))   # the lesion in the null arm
    _set_argv_env(cell, "lesion_emotion_category_s43.json", json.dumps({LESION_KNOB: "1", "BRAIN_AFFECT_LESION": "1"}))
    assert _fails(cell, S.TREATMENT_ON) == {
        "intact_b_%s_s43.json.prov.json" % GROUP: ["stray BRAIN_* key(s) in argv --env: %s" % LESION_KNOB],
        "lesion_emotion_category_s43.json.prov.json": ["stray BRAIN_* key(s) in argv --env: BRAIN_AFFECT_LESION"]}
    # a non-BRAIN key there is not read (the same scope as the env rule)
    cell = _real_argv(build_cell(tmp_path, "nb", 43, on=True), 43)
    _set_argv_env(cell, "intact_a_%s_s43.json" % GROUP, json.dumps({"LB_SOMETHING": "1"}))
    assert _fails(cell, S.TREATMENT_ON) == {}


def test_an_unparseable_argv_env_fails_and_the_default_rule_is_unchanged(tmp_path):
    cell = _real_argv(build_cell(tmp_path, "on", 43, on=True), 43)
    _set_argv_env(cell, "intact_a_%s_s43.json" % GROUP, "{not json")
    assert _fails(cell, S.TREATMENT_ON) == {"intact_a_%s_s43.json.prov.json" % GROUP: [
        "argv --env unparseable (want a JSON object)"]}
    # without a treatment the argv is not read (every other caller keeps the old rule)
    cell = _real_argv(build_cell(tmp_path, "off", 43, on=False), 43)
    _set_argv_env(cell, "intact_a_%s_s43.json" % GROUP, json.dumps({KEY: "1", "BRAIN_X": "1"}))
    assert _fails(cell, None) == {}


def test_score_b8_under_pin_reads_argv_env(tmp_path):
    """The panel's mutation through the scorer: the ON s42 lesion arms' argv --env set KEY to 0 -> UNDEFINED."""
    base = _tree(tmp_path / "t")
    for s in S.GATE_SEEDS:
        for tag in (S.LB_ON_TAG, S.LB_OFF_TAG):
            _real_argv(os.path.join(base, tag, "s%d" % s, "emotion-category"), s)
    assert S.score_b8(base, pin=PIN)["status"] == S.PASS
    _real_argv(os.path.join(base, S.LB_ON_TAG, "s42", "emotion-category"), 42, lesion_env={LESION_KNOB: "1", KEY: "0"})
    r = S.score_b8(base, pin=PIN)
    assert r["status"] == S.UNDEFINED and "set by argv --env" in r["on_cells"]["s42"]["why"]


# ── lb_shard aggregate: --treatment is opt-in; the default aggregate is unchanged ────────────────────────────────
def _agg(tmp_path, tag, treatment=None):
    out = str(tmp_path / ("agg_%s_%s.json" % (tag, bool(treatment))))
    a = argparse.Namespace(tag=tag, seeds=[43], pin=PIN, base=str(tmp_path), out=out, treatment=treatment)
    lb_shard.cmd_aggregate(a)
    return json.load(open(out))


def test_aggregate_treatment_flag(tmp_path):
    build_cell(tmp_path, "t_on", 43, on=True)
    build_cell(tmp_path, "t_off", 43, on=False)
    assert _agg(tmp_path, "t_on")["provenance"]["n_invalid"] == 1                       # default rule: stray key
    on = _agg(tmp_path, "t_on", ["%s=1" % KEY])
    assert on["provenance"]["n_invalid"] == 0 and on["provenance"]["treatment"] == {KEY: "1"}
    assert on["per_faculty"]["emotion-category"]["n_load_bearing"] == 1
    off = _agg(tmp_path, "t_off")
    assert off["provenance"]["n_invalid"] == 0 and "treatment" not in off["provenance"]
    assert _agg(tmp_path, "t_off", ["%s=" % KEY])["provenance"]["n_invalid"] == 0      # forbidden and absent: fine
    assert _agg(tmp_path, "t_on", ["%s=" % KEY])["provenance"]["n_invalid"] == 1       # forbidden and present


def test_parse_treatment():
    assert lb_shard.parse_treatment(None) == {}
    assert lb_shard.parse_treatment(["A=1", "B="]) == {"A": "1", "B": None}
    with pytest.raises(SystemExit):
        lb_shard.parse_treatment(["A"])
