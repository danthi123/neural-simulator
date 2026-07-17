"""Pin the "this is inert / this is a no-op" claims that the plasticity machinery makes about itself.

WHY: on 2026-07-16 THREE separate unasserted inertness claims were found FALSE in one session, each failing
SILENTLY and each corrupting results for an unknown span of prior work:

  1. `sim/bridge.py`'s Hebbian weight-decay used the RAW cp_plasticity_rate_gain. Structural plasticity (ON by
     default) FORMS synapses, growing cp_connections.nnz without growing the gate arrays -> "operands could not be
     broadcast" EVERY step -> silently caught -> the Hebbian decay stopped applying entirely. Known bug class
     (_ensure_gate_capacity guards 7 other sites); the Hebbian block was missed. FIXED -- test 3 pins the fix.
  2. `_onbridge_eprop_port_derisk.py` claims "we pass lr=0.0 so the committed BDSP kernel is byte-INERT
     (w_new = w + 0*...)". FALSE: fused_bdsp_update ENDS in `return cp.clip(w_new, w_min, w_max)` -- at eta=0 the
     ADD term vanishes, the CLIP does not. Tests 1+2 pin the real semantics and the CORRECT lever.
  3. A runner silently no-op'd forever (swallowed init -> is_initialized False -> every step an early return).

THE LESSON THESE TESTS ENCODE: a claim of the form "X is inert / byte-identical / a no-op" is a HYPOTHESIS. It
belongs in an ASSERTION, not a comment. A comment cannot fail; it just rots, and the result silently rots with it.

CPU/numpy, no GPU needed.
"""
import numpy as np
import pytest

from sim.backend import get_backend

xp, _backend = get_backend()


# ---------------------------------------------------------------------------------------------------------------
# 1+2: the BDSP kernel's inertness -- lr=0 is NOT the lever; enable_bdsp=False is.
# ---------------------------------------------------------------------------------------------------------------

def test_bdsp_kernel_at_eta0_is_NOT_inert_it_still_clips():
    """eta=0 kills the ADD term but NOT the clip. This is the exact false claim that silently crushed ~47% of the
    e-prop port's feedforward weights (ff_w_init=2000, --w-clip 4000, inherited bdsp_w_max=6.0) on EVERY forward
    while its docstring asserted the kernel was 'byte-INERT'."""
    from sim.kernels import fused_bdsp_update

    w = xp.asarray([-4000.0, -7.0, -6.0, 0.0, 2000.0], dtype=xp.float32)   # e-prop-scale weights
    z = xp.zeros_like(w)
    # positional: cupy's @fuse() kernels reject keyword args, so kwargs here made the test pass on numpy
    # and TypeError on cupy -- a test that only works on one backend is a broken instrument.
    out = fused_bdsp_update(w, z, z, z, z, 0.0, -6.0, 6.0)
    out_h = np.asarray(out.get() if hasattr(out, "get") else out)

    # The ADD term IS inert at eta=0 (weights already inside the band are untouched)...
    assert out_h[2] == pytest.approx(-6.0)
    assert out_h[3] == pytest.approx(0.0)
    # ...but the CLIP is NOT: anything outside [w_min, w_max] is silently crushed even though eta == 0.
    assert out_h[0] == pytest.approx(-6.0), "eta=0 did NOT protect a -4000 weight: the clip is unconditional"
    assert out_h[4] == pytest.approx(6.0), "eta=0 did NOT protect a +2000 weight: the clip is unconditional"
    assert out_h[1] == pytest.approx(-6.0)
    # => lr/eta=0 is NOT an inertness lever for weights that live outside [w_min, w_max].


def test_bdsp_kernel_at_eta0_IS_inert_inside_the_clip_band():
    """The kernel is genuinely a no-op at eta=0 for weights already within [w_min, w_max] -- which is exactly why
    the false claim survived unnoticed: it is true in the regime people usually test, and false in the regime the
    e-prop port actually runs in (2000-scale weights vs a +-6 band)."""
    from sim.kernels import fused_bdsp_update

    rng = np.random.RandomState(0)
    w0 = rng.uniform(-5.0, 5.0, size=64).astype(np.float32)
    w = xp.asarray(w0)
    z = xp.zeros_like(w)
    # positional: cupy's @fuse() kernels reject keyword args, so kwargs here made the test pass on numpy
    # and TypeError on cupy -- a test that only works on one backend is a broken instrument.
    out = fused_bdsp_update(w, z, z, z, z, 0.0, -6.0, 6.0)
    out_h = np.asarray(out.get() if hasattr(out, "get") else out)
    np.testing.assert_allclose(out_h, w0, rtol=0, atol=0)


def test_bdsp_kernel_docstring_lever_is_enable_bdsp_not_lr():
    """The kernel's OWN docstring names the correct lever: 'byte-inert when enable_bdsp is False (the block is
    unreached and this kernel is never invoked)'. Pin that wording so a future edit cannot quietly re-assert the
    lr=0 story that was false."""
    from sim.kernels import fused_bdsp_update

    doc = (fused_bdsp_update.__doc__ or "")
    assert "enable_bdsp" in doc, "the kernel must document WHICH flag makes it inert"
    assert "clip" in doc.lower() or "clamp" in doc.lower(), "the kernel must document that it clamps"


# ---------------------------------------------------------------------------------------------------------------
# 3: the Hebbian decay must survive the gate array going stale (the bug fixed 2026-07-16).
# ---------------------------------------------------------------------------------------------------------------

def test_ensure_gate_capacity_grows_a_stale_gate_array_to_nnz():
    """The regression for the fixed bug. Structural plasticity grows cp_connections.nnz WITHOUT growing the gate
    arrays; the Hebbian decay then did `hebbian_weight_decay * cp_plasticity_rate_gain` (raw) and raised
    'operands could not be broadcast' EVERY step -- silently caught, so the decay just stopped applying.
    _ensure_gate_capacity must grow it, defaulting NEW entries to 1.0 (an open gate = a no-op multiplier)."""
    from sim.bridge import SimulationBridge

    br = SimulationBridge.__new__(SimulationBridge)          # no full init: exercise the helper in isolation
    br.cp_plasticity_rate_gain = xp.full(100, 0.5, dtype=xp.float32)

    grown = br._ensure_gate_capacity("cp_plasticity_rate_gain", 137)
    assert grown is not None
    assert grown.shape[0] >= 137, "the gate array was not grown to the requested nnz"
    g = np.asarray(grown.get() if hasattr(grown, "get") else grown)
    np.testing.assert_allclose(g[:100], 0.5, rtol=0, atol=0)   # existing entries preserved verbatim
    np.testing.assert_allclose(g[100:137], 1.0, rtol=0, atol=0)  # new entries = open gate (no-op multiplier)


def test_hebbian_decay_uses_ensure_gate_capacity_not_the_raw_array():
    """Source-level guard. The Hebbian block must route the gate through _ensure_gate_capacity like the other 7
    sites -- if someone reverts to the raw array, any bridge with a plasticity_gate + structural growth silently
    stops decaying weights again, and NOTHING else in the suite would catch it (the failure is swallowed)."""
    import inspect
    from sim import bridge as bridge_mod

    src = inspect.getsource(bridge_mod.SimulationBridge._run_one_simulation_step)
    i = src.find("hebbian_weight_decay *")
    assert i != -1, "could not locate the Hebbian weight-decay site"
    window = src[i - 400:i + 200]
    assert "_ensure_gate_capacity" in window, (
        "the Hebbian weight-decay must use _ensure_gate_capacity for cp_plasticity_rate_gain; using the raw array "
        "breaks (silently, via a swallowed broadcast error) as soon as structural plasticity grows nnz"
    )


# ---------------------------------------------------------------------------------------------------------------
# 4: the deep-credit GO gate must include a RESERVOIR control (added 2026-07-16).
# ---------------------------------------------------------------------------------------------------------------

def test_deep_credit_gate_includes_a_reservoir_control():
    """The banked headline "feedforward spiking deep credit is ALREADY GO (K=8 0.877)" turned out to be ~80% a
    FIXED RANDOM SPIKING RESERVOIR + a trained linear readout (measured 2026-07-16: FULL 0.889 vs FROZEN 0.778 vs
    chance 0.333). It passed its gate because `trains_the_task` compared against chance / permuted / shuffle-DFA --
    and NOT ONE of those is a frozen-hidden baseline, so a reservoir result passed UNCHANGED. Worse, the isolation
    hook (`train_layers`, documented in-file as "None => update all FF pathways; a set => update only those
    (isolation)") had been written FOR EXACTLY THIS and was never once invoked.

    Source-level guard: if someone drops the frozen-hidden arm from the gate, a random projection + logistic
    regression can be reported as deep credit again, and NOTHING else in the suite would catch it -- the failure is
    a silently-passing GO, not an error."""
    import inspect
    from research.runners import _onbridge_eprop_port_derisk as m

    src = inspect.getsource(m.run_seed)
    assert "reservoir_control" in src, "run_seed must run a frozen-hidden RESERVOIR control"
    assert "train_layers" in src, "the reservoir control must use the train_layers isolation hook"
    i = src.find("trains = bool(")
    assert i != -1, "could not locate the GO gate"
    gate = src[i:i + 700]
    assert "froz_inh" in gate, (
        "the GO gate must compare against the frozen-hidden baseline. Without it, a fixed random reservoir + a "
        "linear readout passes as 'deep credit' -- exactly how the banked K=8 0.877 headline (~80% reservoir) "
        "passed for months."
    )
    sig = inspect.signature(m.run_seed)
    assert sig.parameters["reservoir_control"].default is True, (
        "the reservoir control must default ON -- a gate that CAN pass without it is the bug itself"
    )


# ---------------------------------------------------------------------------------------------------------------
# 5: PROVENANCE -- every knob that changes the experiment must reach the output artifact (added 2026-07-16).
# ---------------------------------------------------------------------------------------------------------------

def test_output_config_records_pool_k_and_freeze_hidden():
    """`--pool-k` DEFAULTS TO 1 while this whole arc runs at 8, and `pool_k` was NOT written into the output config
    -- so a result file that does not mention pool_k is INDISTINGUISHABLE FROM A pool_k=1 RUN, and the only
    provenance for the arc's most load-bearing knob was the string "k8" in a FILENAME. Recovering it for
    `_eprop_banked_{FULL,FROZEN}.json` required forensics off the bridge's synapse count (1408 @ k=1, 22528 @ k=4,
    90112 @ k=8 -- exact k^2 scaling).

    The general rule this pins: AN ABSENT FLAG MEANS *DEFAULT*, NOT *OFF*. If a config must be reconstructed from a
    filename or a log's byte-count, the record is already broken."""
    import inspect
    from research.runners import _onbridge_eprop_port_derisk as m

    src = inspect.getsource(m.main)
    # main() builds TWO "config" dicts (the positive-control summary and the real one); select the one
    # carrying the real run's knobs, not whichever comes first.
    i = src.find('"train_subsample": a.train_subsample')
    assert i != -1, "could not locate the main output config dict"
    cfg = src[max(0, i - 800):i + 800]
    for knob in ('"pool_k"', '"freeze_hidden"'):
        assert knob in cfg, (
            f"{knob} must be recorded in the output config. It changes the experiment: pool_k sizes the network "
            f"(population coding; the parent ctor takes it) and freeze_hidden decides whether the hidden layers "
            f"learn at all. Unrecorded => the artifact cannot be told apart from a default run."
        )


def test_per_seed_records_the_linear_floor():
    """`stage0_depth_genuineness` COMPUTES `linear_inherit_heldout` (the no-hidden linear floor) and run_seed used to
    record only `l1`, so the linear floor never reached any output file -- leaving "is the task just linearly
    decodable?" unanswerable from the artifacts. It is the cheapest rebuttal available (measured 0.265, BELOW chance
    0.333 at this config). Ladder: chance -> l0 -> l1 -> frozen_hidden -> inherit."""
    import inspect
    from research.runners import _onbridge_eprop_port_derisk as m

    src = inspect.getsource(m.run_seed)
    assert "stage0_l0_linear" in src, "run_seed must record the LINEAR floor (stage0's linear_inherit_heldout)"
    assert "linear_inherit_heldout" in src, "the linear floor must come from stage0's own computed value"


# ---------------------------------------------------------------------------------------------------------------
# 6: REPRODUCIBILITY -- the seed must actually control the SUBSTRATE, not just the weights (found 2026-07-17).
# ---------------------------------------------------------------------------------------------------------------

def test_onbridge_net_seeds_cfg_seed_not_only_actual_seed_used():
    """THE BUG THIS PINS (found 2026-07-17, after it silently confounded an entire arc):

    The builder set `cfg.actual_seed_used = seed` -- a REPORTING field the bridge never reads. The bridge seeds
    heterogeneity from `cfg.seed` (bridge.py:2136):
        het_seed = cfg.heterogeneity_seed if cfg.heterogeneity_seed >= 0 else cfg.seed
        if het_seed >= 0: cp.random.seed(het_seed)
    Both defaulted to -1, so the guard was FALSE and `cp.random.seed()` was NEVER CALLED -- leaving the per-neuron
    firing thresholds (bridge.py:1508 `cp.random.uniform(...)`) drawn from the UNSEEDED GLOBAL RNG.

    MEASURED CONSEQUENCE: `--seeds 42` did not control the substrate. Two FRESH processes at seed 42 produced
    DIFFERENT thresholds (md5 55c612a7 vs cc6815c6), and four nets built back-to-back in ONE process differed by up
    to 18.4 mV (each _mk() advanced the global RNG). So every FULL-vs-FROZEN comparison in the arc compared nets with
    DIFFERENT NEURONS, and the confound was LARGER than the effect (deep_credit_share swung +0.33 / 0.00 / -0.33 on
    the same seed 42).

    Source-level guard: if `cfg.seed` is dropped again, the arc silently returns to measuring noise, and NOTHING
    else in the suite would catch it -- the failure is a plausible number, not an error."""
    import inspect
    from research.runners import _semantic_inheritance_onbridge_spiking_derisk as m

    src = inspect.getsource(m.OnBridgeBDSPNet.__init__)
    assert "cfg.seed = int(seed)" in src, (
        "OnBridgeBDSPNet must set cfg.seed -- that is the field the bridge reads for heterogeneity "
        "(bridge.py:2136). cfg.actual_seed_used is a REPORTING field only; setting it alone leaves the "
        "per-neuron firing thresholds on the UNSEEDED global RNG, which confounds every same-seed comparison."
    )
    i = src.find("cfg.actual_seed_used")
    j = src.find("cfg.seed = int(seed)")
    assert i != -1 and j > i, "cfg.seed must be set alongside/after actual_seed_used in the config block"


def test_bridge_heterogeneity_guard_still_requires_a_nonnegative_seed():
    """Pins WHY the bug existed, so a future reader cannot mistake it for paranoia: the bridge's heterogeneity
    seeding is GUARDED by `het_seed >= 0`, and both source fields default to -1. The guard is correct; the builder
    was simply not setting the field it reads. If this guard ever becomes unconditional, this test should be
    revisited -- but until then, NOT setting cfg.seed means NOT seeding."""
    import inspect
    from sim import bridge as bridge_mod

    src = inspect.getsource(bridge_mod)
    assert "cfg.heterogeneity_seed if cfg.heterogeneity_seed >= 0 else cfg.seed" in src, (
        "the bridge's het_seed derivation changed -- re-verify what the builder must set"
    )
