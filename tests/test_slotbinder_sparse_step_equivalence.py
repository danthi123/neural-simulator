"""EQUIVALENCE GATE for the SlotBinder's opt-in event-driven step (2026-09-25).

`SlotBinderComposer(sparse_step=True)` -> `build_binder_bridge(sparse_step=True)` -> `cfg.sparse_activity_step`:
synaptic propagation and Hebbian bookkeeping touch only the synapses of neurons that fired plus the gain!=0
synapses, instead of all nnz every step (plus `enable_reward_modulation=False`, inert on this private bridge).
On numpy it must be BIT-IDENTICAL to the unchanged path: same stored weights, same state, same answers.

Pinned here:
  1. per-step lockstep bit-identity of every state array + the weights, through a teach window and a read window;
  2. the composer contract on synthetic facts (dense AND fanout wiring): weights after teach / after queries /
     after the zeroed-synapse ablation, every answer (patient, agent, yes/no, attribute, describe), final state;
  3. seed 7, N=8 REAL day_33 facts, the production gate's own slotbinder-arm protocol (teach, every query, moat,
     mismatch, ablation re-query) -- skipped only when the machine-local live bundle is absent;
  4. seed 7, N=32 real facts, same protocol -- RUN_SLOW_TESTS=1 (the unchanged path takes ~30 min on numpy;
     the committed artifact research/findings/raw/_slotbinder_sparse_step/equivalence_seed7_n8_n32.json is
     that run);
  5. the comparison CAN FAIL: a sabotaged event-driven decay is detected;
  6. flag-off is the default, and the dispatch guard refuses configurations it was not verified in.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from research.runners.slotbinder_composer import SlotBinderComposer  # noqa: E402
from research.runners._keystone2_spiking_slot_binder_derisk import build_binder_bridge, _idx  # noqa: E402
from sim.backend import to_host, from_host  # noqa: E402

_STATE = ("cp_membrane_potential_v", "cp_recovery_variable_u", "cp_firing_states", "cp_prev_firing_states",
          "cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise",
          "cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise")
_VOCAB = ["dog", "cat", "fish", "bird", "chase", "eat", "see", "hear", "big", "red", "river"]


def _bits(a):
    return np.ascontiguousarray(np.asarray(to_host(a))).view(np.uint8).tobytes()


def _same_state(b1, b2):
    diffs = [a for a in _STATE if getattr(b1, a, None) is not None and _bits(getattr(b1, a)) != _bits(getattr(b2, a))]
    if _bits(b1.cp_connections.data) != _bits(b2.cp_connections.data):
        diffs.append("cp_connections.data")
    return diffs


def test_flag_off_is_the_default():
    from sim.config import CoreSimConfig
    assert CoreSimConfig().sparse_activity_step is False
    saved = os.environ.pop("BRAIN_SLOTBINDER_SPARSE_STEP", None)
    try:
        assert SlotBinderComposer(vocab=list(_VOCAB)).sparse_step is False
    finally:
        if saved is not None:
            os.environ["BRAIN_SLOTBINDER_SPARSE_STEP"] = saved
    b = build_binder_bridge(3, K=4, KF=6)
    assert b.core_config.sparse_activity_step is False
    assert b._sparse_activity_step_can_dispatch(b.core_config) is False


def test_dispatch_guard_refuses_unverified_features():
    b = build_binder_bridge(3, K=4, KF=6, sparse_step=True)
    cfg = b.core_config
    assert b._sparse_activity_step_can_dispatch(cfg) is True
    for attr, val in (("hebbian_symmetric", True), ("enable_branchless_plasticity", True),
                      ("enable_short_term_plasticity", True), ("deterministic_transpose_matvec", True),
                      ("enable_coincidence_detection", True), ("hebbian_rate_window", True)):
        old = getattr(cfg, attr)
        setattr(cfg, attr, val)
        try:
            assert b._sparse_activity_step_can_dispatch(cfg) is False, attr
        finally:
            setattr(cfg, attr, old)


def test_lockstep_per_step_bit_identity_teach_and_read():
    """Two bridges, same seed, flag off/on, driven by the SAME external currents: every state array and every
    weight must match bit for bit after EVERY step, through a teach window (one gate open) and a read window."""
    K, KF, n_steps = 6, 8, 40
    b_off = build_binder_bridge(11, K=K, KF=KF, fanout=4, required_fillers={0: [2], 1: [5]})
    b_on = build_binder_bridge(11, K=K, KF=KF, fanout=4, required_fillers={0: [2], 1: [5]}, sparse_step=True)
    assert b_on._sparse_activity_step_can_dispatch(b_on.core_config)
    assert not b_off._sparse_activity_step_can_dispatch(b_off.core_config)
    assert _bits(b_off.cp_neuron_firing_thresholds) == _bits(b_on.cp_neuron_firing_thresholds), "seed not in control"
    n = b_off.core_config.num_neurons
    fired_total = 0
    for phase, (slot, filler, gate) in enumerate(((0, 2, "slot0_to_filler"), (1, 5, "slot1_to_filler"),
                                                  (0, None, None))):
        cur = np.zeros(n)
        cur[_idx(b_off, f"w{slot}")] = 400.0
        if filler is not None:
            cur[_idx(b_off, f"f{filler}")] = 400.0
        for b in (b_off, b_on):
            if gate:
                b.set_plasticity_gate(gate, 1.0)
        dev_off, dev_on = from_host(cur.astype(np.float64)), from_host(cur.astype(np.float64))
        for t in range(n_steps):
            b_off.cp_external_input_current[:] = dev_off
            b_off._run_one_simulation_step()
            b_on.cp_external_input_current[:] = dev_on
            b_on._run_one_simulation_step()
            d = _same_state(b_off, b_on)
            assert not d, f"phase {phase} step {t}: differs in {d}"
            fired_total += int(np.asarray(to_host(b_off.cp_firing_states)).sum())
        for b in (b_off, b_on):
            if gate:
                b.set_plasticity_gate(gate, 0.0)
    assert fired_total > 0, "no spikes -- the lockstep comparison would be vacuous"
    w = np.asarray(to_host(b_off.cp_connections.data))
    assert np.count_nonzero(w > 1.0) > 0


def _protocol(c, facts):
    """Teach, then exercise every read-side contract call, then the zeroed-synapse ablation. Returns
    (answers, weights snapshots)."""
    for f in facts:
        assert c.store(*f[:3], polarity=f[3], attribute=f[4]) is True
    snaps = [np.array(to_host(c._b.cp_connections.data), copy=True)]
    ans = []
    for f in facts:
        ans.append(("patient", c.query_patient(f[0], f[1])))
        ans.append(("agent", c.query_agent(f[1], f[2])))
        ans.append(("yn", c.ask_yes_no(f[0], f[1], f[2])))
        ans.append(("attr", c.query_attribute(f[0], f[1])))
    ans.append(("moat", c.query_patient("river", "hear")))
    ans.append(("mismatch", c.query_patient(facts[0][0], facts[1][1])))
    ans.append(("render", c.render_fact(facts[0][0])))
    snaps.append(np.array(to_host(c._b.cp_connections.data), copy=True))
    c._b.cp_connections.data[:] = 0
    ans.extend(("ablated", c.query_patient(f[0], f[1])) for f in facts)
    snaps.append(np.array(to_host(c._b.cp_connections.data), copy=True))
    return ans, snaps


_FACTS = [("dog", "chase", "cat", None, None), ("cat", "eat", "fish", "NEGATE", None),
          ("bird", "see", "dog", None, "big"), ("fish", "hear", "bird", None, "red")]


@pytest.mark.parametrize("fanout", [None, 6])
def test_composer_contract_bit_identical(fanout):
    kw = dict(seed=5, vocab=list(_VOCAB), max_facts=5, fanout=fanout,
              prewire_facts=(list(_FACTS) if fanout else None))
    c_off = SlotBinderComposer(sparse_step=False, **kw)
    c_on = SlotBinderComposer(sparse_step=True, **kw)
    a_off, w_off = _protocol(c_off, _FACTS)
    a_on, w_on = _protocol(c_on, _FACTS)
    assert c_on._b._sparse_activity_step_can_dispatch(c_on._b.core_config)
    assert a_off == a_on
    for label, x, y in zip(("after teach", "after queries", "after ablation"), w_off, w_on):
        assert x.view(np.uint8).tobytes() == y.view(np.uint8).tobytes(), f"weights differ {label}"
    assert not _same_state(c_off._b, c_on._b)
    # not vacuous: the intact binder answers, the ablated one does not
    assert ("patient", "cat") in a_off and ("yn", "no") in a_off
    assert all(v is None for k, v in a_off if k == "ablated")


def test_the_comparison_can_fail():
    """Sabotage the event-driven decay (empty gain!=0 set): the stored weights must then DIFFER -- proving the
    bit comparison used above is able to fail in its failing direction."""
    kw = dict(seed=5, vocab=list(_VOCAB), max_facts=3)
    c_off = SlotBinderComposer(sparse_step=False, **kw)
    c_bad = SlotBinderComposer(sparse_step=True, **kw)
    c_bad._ensure()
    empty = np.zeros(0, dtype=np.int64)
    c_bad._b._sparse_gain_index_sets = lambda: (empty, empty)
    for c in (c_off, c_bad):
        c.store("dog", "chase", "cat")
    w_off = np.asarray(to_host(c_off._b.cp_connections.data)).view(np.uint8).tobytes()
    w_bad = np.asarray(to_host(c_bad._b.cp_connections.data)).view(np.uint8).tobytes()
    assert w_off != w_bad


def _live_bundle_or_skip():
    try:
        from research.runners._slotbinder_l2_sparse_derisk import _load_live_bundle
        return _load_live_bundle()
    except FileNotFoundError as e:
        pytest.skip(f"machine-local live bundle absent: {e}")


def _real_facts_equivalence(n_facts):
    from research.runners._slotbinder_sparse_step_equivalence import run_path, compare
    from research.runners._slotbinder_l2_sparse_derisk import _sample_facts
    _, facts_full, _ = _live_bundle_or_skip()
    sample, _ = _sample_facts(facts_full, 7, n_facts)
    off, w_off = run_path(sample, 7, 32, False, n_facts, n_facts, True, lambda m: None)
    on, w_on = run_path(sample, 7, 32, True, n_facts, n_facts, True, lambda m: None)
    eq = compare(off, on, w_off, w_on)
    assert on["sparse_activity_step_dispatches"] and not off["sparse_activity_step_dispatches"]
    assert eq["all"], eq
    hits = sum(a == f["patient"] for a, f in zip(off["intact"]["answers"], sample))
    assert hits >= n_facts // 2, "intact recall collapsed -- the comparison would be vacuous"


def test_real_facts_seed7_n8_production_protocol_bit_identical():
    _real_facts_equivalence(8)


@pytest.mark.skipif(not os.environ.get("RUN_SLOW_TESTS"),
                    reason="~35 min on numpy (the unchanged path). Set RUN_SLOW_TESTS=1 to enable.")
def test_real_facts_seed7_n32_production_protocol_bit_identical():
    _real_facts_equivalence(32)
