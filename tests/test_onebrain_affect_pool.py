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


def test_aggregate_ignores_superseded_v1_x_checks_and_requires_v2(tmp_path):
    """A seed whose only arm-X record is the superseded v1 instrument is NOT GO (its X checks are dropped, so the
    v2 checks are MISSING); the same seed with a v2 record passing everything is GO. Fails in the failing direction."""
    import json
    from research.runners import _onebrain_affect_pool_verify as V
    m = {f"M{i}_x": True for i in range(1, 8)}
    v1 = {f"X{i}_x": True for i in range(1, 8)}
    v2 = {**{f"X{i}_x": True for i in range(0, 3)}, **{f"I{i}_x": True for i in range(1, 8)}}
    seeds = list(V.SEEDS)
    (tmp_path / "M.json").write_text(json.dumps({"mode": "verify", "per_seed": [
        {"seed": s, "checks": m} for s in seeds]}))
    (tmp_path / "X1.json").write_text(json.dumps({"mode": "verify", "per_seed": [
        {"seed": s, "checks": v1} for s in seeds]}))
    assert V.aggregate([str(tmp_path / "M.json"), str(tmp_path / "X1.json")]) is False
    (tmp_path / "X2.json").write_text(json.dumps({"mode": "verify", "per_seed": [
        {"seed": s, "checks": v2, "x_instrument": V.X_INSTRUMENT} for s in seeds]}))
    assert V.aggregate([str(tmp_path / "M.json"), str(tmp_path / "X2.json")]) is True
    bad = dict(v2, X1_x=False)
    (tmp_path / "X3.json").write_text(json.dumps({"mode": "verify", "per_seed": [
        {"seed": s, "checks": bad, "x_instrument": V.X_INSTRUMENT} for s in seeds]}))
    assert V.aggregate([str(tmp_path / "M.json"), str(tmp_path / "X3.json")]) is False
