"""Light unit tests for the D3 affect -> one-brain-pool migration (no full brain; the 690-neuron ladder only).

The heavy answer-preservation / cross-edge gate is `research/runners/_onebrain_affect_pool_verify.py` (pool nodes).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np
import pytest


def test_flags_default_off(monkeypatch):
    from research.runners import onebrain_affect_pool_flags as F
    monkeypatch.delenv("BRAIN_ONEBRAIN_AFFECT_POOL", raising=False)
    monkeypatch.delenv("BRAIN_ONEBRAIN_AFFECT_XEDGE", raising=False)
    assert F.affect_pool_enabled() is False
    assert F.affect_xedge_enabled() is False
    monkeypatch.setenv("BRAIN_ONEBRAIN_AFFECT_POOL", "1")
    assert F.affect_pool_enabled() is True
    monkeypatch.setenv("BRAIN_ONEBRAIN_AFFECT_POOL", "0")
    assert F.affect_pool_enabled() is False


def test_flags_module_is_import_light():
    """The production call sites import ONLY the flags module when the flags are off -> nothing heavy loads."""
    import ast
    import inspect
    from research.runners import onebrain_affect_pool_flags as F
    tree = ast.parse(inspect.getsource(F))
    mods = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= {"os", "__future__"}


class _RM:
    def __init__(self, m):
        self._m = m

    def indices(self, name):
        return self._m[name]


class _B:
    def __init__(self, m):
        self.region_manager = _RM(m)


def test_cross_edge_transmission_gate_is_additive():
    """A CrossEdge WITHOUT a transmission_gate produces exactly the pre-existing dense dict (no new key);
    with one, the key is added and nothing else changes."""
    from research.runners.onebrain_merge_framework import CrossEdge, _cross_edge_dense
    b = _B({"a": [0, 1], "b": [5, 6, 7]})
    ce = CrossEdge(key="k", source_key="x", source_region="a", target_key="y", target_region="b",
                   init_weight=0.3, plastic=False, freeze_rest=False)
    d0 = _cross_edge_dense(b, ce)
    assert set(d0) == {"pre_indices", "post_indices", "initial_weights", "plastic", "conn_type", "count"}
    import dataclasses
    d1 = _cross_edge_dense(b, dataclasses.replace(ce, transmission_gate="g"))
    assert d1.pop("transmission_gate") == "g"
    assert set(d1) == set(d0)
    for k in d0:
        assert np.array_equal(np.asarray(d0[k]), np.asarray(d1[k]))


def test_descriptor_spec_matches_production_ladder():
    """The pool descriptor carries EXACTLY the production ladder's regions (names, sizes, intrinsic offsets,
    internal density) and pathways (endpoints, weights, gates) — reuse, not a re-derivation."""
    from research.runners.onebrain_affect_pool import _spec_affect_ladder, AFFECT_DESCRIPTOR
    from research.runners._appraisal_interoceptive_ladder_derisk import AppraisalInteroceptiveLadder
    regions, pathways, meta = _spec_affect_ladder(42)
    lad = AppraisalInteroceptiveLadder(seed=42)
    cfg = lad._bridge.core_config
    key_r = lambda r: (r.name, r.n_neurons, r.exc_fraction, r.internal_density, r.exc_weight_mean,
                       r.intrinsic_current_pA, r.izh_neuron_type, bool(r.enable_nmda))
    key_p = lambda p: (p.from_region, p.to_region, p.density, p.weight_mean, p.plastic, p.transmission_gate)
    assert [key_r(r) for r in regions] == [key_r(r) for r in cfg.brain_regions]
    assert [key_p(p) for p in pathways] == [key_p(p) for p in cfg.region_pathways]
    assert all(not p.plastic for p in pathways)
    assert tuple(r.name for r in regions) == AFFECT_DESCRIPTOR.regions
    assert all(n.startswith(("aff_", "appr_intero_")) for n in AFFECT_DESCRIPTOR.regions)


def test_affect_alone_on_pool_reads_signed_and_deterministic():
    """Affect alone on its own merge-framework pool (its OWN config only — cheap): signed, neutral, deterministic,
    and the production tone levels match the standalone production ladder at |a|=1."""
    from research.runners.onebrain_merge_framework import merge_organs
    from research.runners.onebrain_affect_pool import AFFECT_DESCRIPTOR, PoolAffectLadder
    from research.runners.affect_production_organ import tone_level
    pool = merge_organs([AFFECT_DESCRIPTOR], 42, wire=True)
    lad = PoolAffectLadder(42, shared=pool)
    d_pos = lad.read_differential(1.0)["differential"]
    d_neg = lad.read_differential(-1.0)["differential"]
    d_0 = lad.read_differential(0.0)["differential"]
    assert d_pos > 0 > d_neg and d_0 == 0.0
    assert lad.read_differential(1.0)["differential"] == d_pos
    assert lad.read_differential(0.7, lesion=True)["differential"] == 0.0
    std = PoolAffectLadder(42, shared=None)
    assert tone_level(d_pos) == tone_level(std.read_differential(1.0)["differential"])
    assert tone_level(d_neg) == tone_level(std.read_differential(-1.0)["differential"])


def test_local_ou_restores_prior_ou_state_and_scope_mask():
    """Fix round 2026-09-23: local_ou RESTORES a pre-existing OU state (it used to None it unconditionally), and
    scope='affect' installs the engine's cp_ou_neuron_mask only inside the window. On an affect-only pool every
    neuron is an affect neuron, so the scoped read must equal the unscoped one exactly."""
    from research.runners.onebrain_merge_framework import merge_organs
    from research.runners.onebrain_affect_pool import AFFECT_DESCRIPTOR, PoolAffectLadder
    pool = merge_organs([AFFECT_DESCRIPTOR], 42, wire=True)
    lad = PoolAffectLadder(42, shared=pool)
    lad.ensure_built()
    b = pool.bridge
    n = int(b.cp_membrane_potential_v.shape[0])
    sentinel = pool.xp.full(n, 3.25, dtype=pool.xp.float32)
    b.cp_ou_current = sentinel
    b._ou_pn_step = 17
    with lad.local_ou(scope="affect"):
        assert b.cp_ou_neuron_mask is not None and bool(np.asarray(b.cp_ou_neuron_mask).all())
        assert b.cp_ou_current is not sentinel
    assert b.cp_ou_current is sentinel and b._ou_pn_step == 17 and b.cp_ou_neuron_mask is None
    b.cp_ou_current = None
    b._ou_pn_step = 0
    d_all = lad.read_differential(1.0)["differential"]
    d_aff = lad.read_differential(1.0, ou_scope="affect")["differential"]
    assert d_all == d_aff and d_all > 0
    with pytest.raises(ValueError):
        with lad.local_ou(scope="nope"):
            pass


def _cond(v, hz=None):
    return {"surprised": list(v), "hz": hz or [float(x) for x in v], "frac": float(np.mean(v)),
            "n_surprised": int(sum(v)), "mean_hz": 0.0}


def test_marginal_strength_selection_and_flip_counting():
    """S* is the LARGEST grid strength whose a=0 flagged fraction is <= MARGINAL_FRAC; none -> None (UNDEFINED)."""
    from research.runners import _onebrain_affect_pool_verify as V
    all_on = [True] * 8
    half = [True] * 4 + [False] * 4
    base = {"contradict": {f"{S:g}": _cond(all_on if S >= 400 else half) for S in V.ASSERT_GRID}}
    assert V.select_marginal_strength(base) == 375.0
    sat = {"contradict": {f"{S:g}": _cond(all_on) for S in V.ASSERT_GRID}}
    assert V.select_marginal_strength(sat) is None
    b0 = _cond([False, False, True, True])
    assert V._newly(b0, _cond([True, False, True, False])) == (1, 1)
    assert V._n_flip_required(8) == 2 and V._n_flip_required(12) == 3


# ── gate v3 (fix round 3): synthetic raw batteries, so the SCORING RULE is tested without a 7.7k-neuron pool ──
_SSTAR = 375.0
_STAGED_V2_REV = "c6fdf7be7673b264316888615be64883f23f48cf"


def _bat(pos_flips=0, arousal=0.0, conf_fa=0, lost_at_prod=0):
    """A battery dict shaped like `arousal_surprise_battery`'s: a=0 flags every contradiction at >=400 pA and half
    below (so S* = 375); `pos_flips` of the not-flagged S* trials are newly flagged; confirm at 600 pA is silent
    except `conf_fa` false alarms; `lost_at_prod` 600 pA contradict detections are lost."""
    from research.runners import _onebrain_affect_pool_verify as V
    contra = {}
    for S in V.ASSERT_GRID:
        v = [True] * 8 if S >= 400 else [True] * 4 + [False] * 4
        if S == _SSTAR:
            v = [True] * 4 + [True] * pos_flips + [False] * (4 - pos_flips)
        if S == V.PROD_ASSERT_PA and lost_at_prod:
            v = [False] * lost_at_prod + [True] * (8 - lost_at_prod)
        contra[f"{S:g}"] = _cond(v)
    conf = {f"{V.PROD_ASSERT_PA:g}": _cond([True] * conf_fa + [False] * (8 - conf_fa))}
    return {"n_trained": 8, "threshold": 1.0, "arousal_rung_hz": float(arousal), "contradict": contra,
            "confirm": conf}


def _raw(pos_flips=2, neg_flips=2, conf_fa_pos=0, lesion_flips=0, inull_flips=0, lost_at_prod=0):
    return {"production_path": _bat(), "base": _bat(),
            "pos": _bat(pos_flips, 5.0, conf_fa_pos, lost_at_prod), "neg": _bat(neg_flips, 5.0),
            "pos_again": _bat(pos_flips, 5.0, conf_fa_pos, lost_at_prod),
            "lesion": _bat(lesion_flips, 5.0), "intero_null": _bat(inull_flips), "noedge_base": _bat(),
            "noedge_pos": _bat(0, 5.0), "ladder_scope_invariance": {"+1.0": [0.3, 0.3], "-1.0": [-0.3, -0.3]},
            "x5_reads": {"affect": {"byte_identical": True, "same_answer": True}}}


def test_x1_counts_a_plus1_once_and_a_minus1_is_reported_not_counted():
    """Re-review issue 2: a=-1 is a construction duplicate of a=+1 (the arousal relay is driven by |appraisal|), so
    X1 is ONE test. a=+1 at the threshold passes with a=-1 at zero; a=-1 cannot rescue a sub-threshold a=+1."""
    from research.runners import _onebrain_affect_pool_verify as V
    chk, det = V.score_x_arm(_raw(pos_flips=2, neg_flips=0))
    assert det["s_star"] == _SSTAR and det["n_flip_required"] == 2
    assert chk["X1_functional_verdict_flip_at_marginal_strength"] is True
    assert det["flips_at_Sstar"]["neg_reported"] == 0
    chk, _ = V.score_x_arm(_raw(pos_flips=1, neg_flips=4))
    assert chk["X1_functional_verdict_flip_at_marginal_strength"] is False
    _, det = V.score_x_arm(_raw(pos_flips=2, neg_flips=2))
    assert det["neg_is_construction_duplicate_of_pos_at_Sstar"] is True


def test_x2_is_relabelled_integrity_I8_and_can_still_fail():
    """Re-review issue 3: v2's X2 was near-guaranteed at w=0.05 -> it is I8 (safety), not evidence. The evidential
    set is exactly {X0, X1}; I8 still fails on a production-strength false alarm or a lost detection."""
    from research.runners import _onebrain_affect_pool_verify as V
    chk, _ = V.score_x_arm(_raw())
    assert {k.split("_")[0] for k in chk if k[0] == "X"} == {"X0", "X1"}
    assert {k.split("_")[0] for k in chk if k[0] == "I"} == {f"I{i}" for i in range(1, 9)}
    assert all(chk.values())
    chk, _ = V.score_x_arm(_raw(conf_fa_pos=1))
    assert chk["I8_production_strength_safety"] is False
    assert chk["X1_functional_verdict_flip_at_marginal_strength"] is True
    chk, _ = V.score_x_arm(_raw(lost_at_prod=1))
    assert chk["I8_production_strength_safety"] is False
    assert "X2" not in V._REQUIRED and "I8" in V._REQUIRED and "X1" in V._REQUIRED


def test_attribution_is_computed_on_the_x1_flips():
    """Re-review issue 1 (behaviour): the lever + attributable_to calls run on the X1 flips vs each control."""
    from research.runners import _onebrain_affect_pool_verify as V
    chk, det = V.score_x_arm(_raw(pos_flips=4))
    a = det["attribution"]
    assert a["lever_moved"] is True and a["edge_lesion"] == 1.0 and a["intero_null"] == 1.0
    assert a["no_edge_pool"] == 1.0
    chk, det = V.score_x_arm(_raw(pos_flips=4, lesion_flips=4, inull_flips=4))
    assert det["attribution"]["lever_moved"] is False and det["attribution"]["edge_lesion"] == 0.0
    assert chk["I1_edge_lesion_verdicts_equal_baseline"] is False and chk["I3_intero_null_collapses"] is False


def test_runner_passes_the_attribution_required_gate():
    """Re-review issue 1 (the BLOCK): the runner computes lesion/null controls, so it must make a tools.lab
    attribution call. The v2 rewrite dropped v1's `lever`/`attributable_to` and this gate went red."""
    from tools.gates import attribution_required as g
    assert g.check(["research/runners/_onebrain_affect_pool_verify.py"]) == []


def _staged_v2_source():
    import subprocess
    try:
        return subprocess.run(["git", "show", f"{_STAGED_V2_REV}:research/runners/_onebrain_affect_pool_verify.py"],
                              capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("staged revision not in this clone")


def test_x_instrument_code_unchanged_since_the_staged_v2_revision():
    """The v2 arm-X pool jobs run at revision c6fdf7be7 and are RE-SCORED by gate v3; that is valid only if the
    MEASUREMENT code there is identical to HEAD's. Pinned at AST level (docstrings excluded)."""
    import ast
    import inspect
    from research.runners import _onebrain_affect_pool_verify as V
    old = _staged_v2_source()
    new = inspect.getsource(V)

    def fns(src):
        out = {}
        for n in ast.parse(src).body:
            if isinstance(n, ast.FunctionDef):
                if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant):
                    n.body = n.body[1:]
                out[n.name] = ast.dump(n)
        return out

    o, h = fns(old), fns(new)
    for name in ("arousal_surprise_battery", "production_path_battery", "_surprise_trial", "_snap_held",
                 "select_marginal_strength", "_newly", "_cond_summary", "_surprise_and_ladder",
                 "_n_flip_required", "_verdicts_equal"):
        assert o[name] == h[name], name

    def calls(src):   # the battery CALLS inside verify_seed (pool build .. the scope read)
        a = src.index("xpool = build_affect_pool(seed, xedge=True)")
        return src[a:src.index("for a in (1.0, -1.0)}", a)]
    assert calls(old) == calls(new)
    for c in ("ASSERT_GRID", "MARGINAL_FRAC", "FLIP_FRAC", "MIN_FLIPS", "LESION_RATIO", "PRE_STEPS", "HOLD",
              "PROD_ASSERT_PA", "CUE_PA", "X_INSTRUMENT"):
        assert old.split(f"\n{c} = ")[1].split("\n")[0] == new.split(f"\n{c} = ")[1].split("\n")[0], c


def test_aggregate_rescores_current_instrument_x_from_raw_and_drops_v1(tmp_path):
    """The stored checks of a current-instrument arm-X record are IGNORED and re-scored from its raw batteries
    (gate v3): all-True stored checks over a failing raw read are NOT GO; all-False stored checks over a passing raw
    read ARE GO; a record missing a raw battery is UNSCORABLE (not a pass); v1 records' X checks are dropped."""
    import json
    from research.runners import _onebrain_affect_pool_verify as V
    seeds = list(V.SEEDS)
    M = str(tmp_path / "M.json")
    (tmp_path / "M.json").write_text(json.dumps({"mode": "verify", "per_seed": [
        {"seed": s, "checks": {f"M{i}_x": True for i in range(1, 8)}} for s in seeds]}))
    liar_true = {**{f"X{i}_x": True for i in range(0, 3)}, **{f"I{i}_x": True for i in range(1, 9)}}
    liar_false = {k: False for k in liar_true}

    def xfile(name, raw, stored):
        (tmp_path / name).write_text(json.dumps({"mode": "verify", "per_seed": [
            {"seed": s, "checks": stored, "x_instrument": V.X_INSTRUMENT, "X": raw} for s in seeds]}))
        return str(tmp_path / name)

    assert V.aggregate([M, xfile("good.json", _raw(), liar_false)]) is True
    assert V.aggregate([M, xfile("thin.json", _raw(pos_flips=1, neg_flips=4), liar_true)]) is False
    partial = {k: v for k, v in _raw().items() if k != "intero_null"}
    assert V.aggregate([M, xfile("partial.json", partial, liar_true)]) is False
    (tmp_path / "v1.json").write_text(json.dumps({"mode": "verify", "per_seed": [
        {"seed": s, "checks": {f"X{i}_x": True for i in range(1, 8)}} for s in seeds]}))
    assert V.aggregate([M, str(tmp_path / "v1.json")]) is False
    assert V.aggregate([M]) is False


def _full_pass_record(seed):
    """A single seed's per-seed dict that passes every _REQUIRED check (M and current-instrument X)."""
    from research.runners import _onebrain_affect_pool_verify as V
    return {"seed": seed, "checks": {f"M{i}_x": True for i in range(1, 8)},
            "x_instrument": V.X_INSTRUMENT, "X": _raw()}


def _write(tmp_path, name, records):
    import json
    (tmp_path / name).write_text(json.dumps({"mode": "verify", "per_seed": records}))
    return str(tmp_path / name)


def test_aggregate_go_count_is_restricted_to_exactly_the_registered_gate_seeds(tmp_path):
    """Re-review issue (SCORER LOOPHOLE): a GO on a NON-gate seed (e.g. the diagnostic seed 7, whose verify-mode X
    file matches the harvest glob) must never stand in for a failing GATE seed. Here seed 42 (a gate seed) fails
    M1, but the non-gate seed 7 fully passes -- before the fix, n_go counted every seed in by_seed (7 seeds; 6 of
    them GO: 43,44,100,101,102,7), so ALL-GO read True at exactly 6/6. After the fix, only the 6 registered gate
    seeds are ever counted (5 of them GO, 42 fails) -> ALL-GO must be False."""
    from research.runners import _onebrain_affect_pool_verify as V
    recs = [_full_pass_record(s) for s in V.SEEDS]
    recs[0]["checks"]["M1_x"] = False       # seed 42 (the first gate seed) fails
    recs.append(_full_pass_record(7))       # non-gate diagnostic seed, fully passing
    path = _write(tmp_path, "mixed.json", recs)
    assert V.aggregate([path]) is False


def test_aggregate_duplicate_record_for_one_seed_reads_undefined_not_last_wins(tmp_path):
    """Re-review issue (duplicate records): more than one record contributing the SAME arm's checks for one seed
    must never resolve via silent dict.update last-wins (an order-dependent selection lever a rerun/retry file
    could exploit). Seed 42 gets TWO current-instrument X records -- a failing one, then (later in the merge
    order) a fully passing one. Before the fix, the later file's all-True checks silently overwrite the failing
    one and seed 42 reads GO. After the fix, a seed with >1 contributing record is DUPLICATE-RECORDS -> UNDEFINED,
    never GO, regardless of file order."""
    from research.runners import _onebrain_affect_pool_verify as V
    good = [_full_pass_record(s) for s in V.SEEDS]
    bad_42 = _full_pass_record(42)
    bad_42["X"] = _raw(pos_flips=1, neg_flips=4)   # fails X1 outright
    # Two records for seed 42 across two files; the LAST one processed is the fully-passing one.
    path_a = _write(tmp_path, "a.json", [bad_42])
    path_b = _write(tmp_path, "b.json", good)
    assert V.aggregate([path_a, path_b]) is False
    # Order reversed: still must not GO (order-independence is the point of the fix).
    assert V.aggregate([path_b, path_a]) is False
