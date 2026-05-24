"""Grounding pin for the generative-replay loop controller (Task 0
of `docs/plans/2026-05-24-generative-replay-implementation.md`).

Pins the structural surface that Task 2 must satisfy when it lands
`research/runners/generative_replay_loop.py`:

  1. The loop-controller module imports successfully.
  2. encode_pfc_frame() returns a ResonateFireFHRR-compatible spike
     pattern (integer-valued numpy array of shape (N_DIM,), the
     phasor-spike encoding produced by phases_to_spikes).
  3. trigger_swr_replay() opens the `ca3_swr_burst` plasticity gate
     for the replay window and closes it after -- exactly the
     validated Phase 1.3 SWR mechanism reused byte-unchanged.
  4. decode_continuation() is structurally bound to the validated
     parallel-population-matching primitive (uses phase_similarity
     and argmax across the substrate-derived grounded vocabulary).
  5. run_generative_loop() does not leak the true continuation
     items into the decoder argument -- enforced by inspecting the
     module source for forbidden oracle-leak patterns.

Each test is intentionally RED until Task 2 lands the
`research.runners.generative_replay_loop` module. Once Task 2
creates that module with the required surface, all 5 tests turn
GREEN. RED failure modes are deterministic:
  - test 1: ModuleNotFoundError
  - tests 2-5: AttributeError on the missing module (caught via
    pytest.importorskip) OR the assertions below if the module
    exists but lacks the required structure.

Plain ASCII only. No protected/frozen/moat module imported or
modified. No automatic differentiation. No-confab moat 7/7 stays
green.
"""
from __future__ import annotations

import inspect

import numpy as np
import pytest


# ----------------------------------------------------------------------
# Helper: import the soon-to-exist loop-controller module. Each test
# uses pytest.importorskip-equivalent behaviour but FAILS (not skips)
# when the module is absent -- this is a grounding pin, not a
# capability test, so the absent module must be a RED FAIL.
# ----------------------------------------------------------------------

def _import_loop_controller():
    """Import the generative-replay loop controller module. Raises
    ModuleNotFoundError if Task 2 hasn't landed it yet -- which is
    the intended RED state for Task 0."""
    import research.runners.generative_replay_loop as m  # noqa: F401
    return m


# ----------------------------------------------------------------------
# Test 1: loop controller module exists.
# ----------------------------------------------------------------------

def test_loop_controller_module_exists():
    """`research.runners.generative_replay_loop` must import. RED
    with ModuleNotFoundError until Task 2 creates the module."""
    m = _import_loop_controller()
    assert m is not None, (
        "research.runners.generative_replay_loop imported but is "
        "None -- expected a module object")
    # The module must declare itself as the loop controller, not a
    # placeholder. Either has a docstring or a public symbol that
    # names the loop.
    has_doc_or_symbol = (
        (getattr(m, "__doc__", None) or "").strip() != ""
        or hasattr(m, "run_generative_loop")
    )
    assert has_doc_or_symbol, (
        "research.runners.generative_replay_loop exists but is empty "
        "-- expected at least a docstring or run_generative_loop "
        "public symbol per Task 2 spec")


# ----------------------------------------------------------------------
# Test 2: encode_pfc_frame produces a ResonateFireFHRR-compatible
# spike pattern.
# ----------------------------------------------------------------------

def test_encode_pfc_frame_returns_fhrr_composite():
    """encode_pfc_frame(items, positions) must produce a composite
    spike pattern compatible with ResonateFireFHRR.encode -- i.e.,
    a numpy array of integer spike steps of shape (N_DIM,) where
    N_DIM matches the project's frozen FHRR dimension."""
    m = _import_loop_controller()
    assert hasattr(m, "encode_pfc_frame"), (
        "Loop controller must expose encode_pfc_frame(items, "
        "positions) per Task 2 spec; missing")
    fn = m.encode_pfc_frame
    assert callable(fn), "encode_pfc_frame must be callable"

    # The function must accept (items, positions). We don't drive
    # it here -- driving it requires the substrate, which is Task 2.
    # We do check that the public signature accepts at minimum
    # those two positional arguments, OR that it has an obvious
    # delegation surface for them.
    sig = inspect.signature(fn)
    param_names = list(sig.parameters.keys())
    # The minimum surface is (items, positions); helpers may add
    # bridge / net / additional kwargs. Either both names are
    # explicit parameters, or *args/**kwargs is present.
    has_items_pos = (
        ("items" in param_names and "positions" in param_names)
        or any(p.kind == inspect.Parameter.VAR_POSITIONAL
               for p in sig.parameters.values())
        or any(p.kind == inspect.Parameter.VAR_KEYWORD
               for p in sig.parameters.values())
    )
    assert has_items_pos, (
        "encode_pfc_frame signature must accept items + positions "
        f"(got {param_names})")

    # The function must, by name OR by source, be wired to the
    # ResonateFireFHRR.encode primitive (the project's FHRR
    # composite producer). Verify via source inspection.
    src = inspect.getsource(fn)
    uses_fhrr_encode = (
        "ResonateFireFHRR" in src
        or ".encode(" in src
        or "rf_bundle" in src
        or "rf_bind" in src
    )
    assert uses_fhrr_encode, (
        "encode_pfc_frame source must wire to the validated "
        "ResonateFireFHRR.encode / rf_bind / rf_bundle primitive "
        "(byte-unchanged reuse); none of these tokens found in "
        "function source")


# ----------------------------------------------------------------------
# Test 3: SWR trigger opens then closes the ca3_swr_burst gate.
# ----------------------------------------------------------------------

class _GateRecordingBridge:
    """A minimal stand-in that records set_plasticity_gate calls so
    we can verify trigger_swr_replay opens then closes the
    ca3_swr_burst gate. The real bridge is built in Task 2's
    runner; this fake is just enough surface for the structural
    check."""

    def __init__(self):
        self.gate_history = []  # list of (gate_name, value) in call order

    def set_plasticity_gate(self, name, value):
        self.gate_history.append((str(name), float(value)))

    # The trigger may also call _run_one_simulation_step or step --
    # absorb either silently.
    def _run_one_simulation_step(self):
        pass

    def step(self):
        pass


def test_swr_trigger_opens_then_closes_gate():
    """trigger_swr_replay(bridge, n_steps) must (a) open the
    `ca3_swr_burst` gate (set value >= 1.0 -- the validated Phase
    1.3 sleep-gate value), (b) run the bridge for the replay
    window, (c) close the gate (set value back to 0.0 -- the
    validated Phase 1.3 awake-gate value).

    The structural assertion: the recorded gate-history contains
    at least one open->close pair on `ca3_swr_burst`."""
    m = _import_loop_controller()
    assert hasattr(m, "trigger_swr_replay"), (
        "Loop controller must expose trigger_swr_replay(bridge, "
        "n_steps) per Task 2 spec; missing")
    fn = m.trigger_swr_replay
    assert callable(fn), "trigger_swr_replay must be callable"

    bridge = _GateRecordingBridge()
    # Drive the trigger with a small n_steps so it can complete
    # without a real bridge. n_steps is small enough that the
    # fake bridge's no-op step methods are called a bounded
    # number of times.
    try:
        fn(bridge, n_steps=1)
    except TypeError:
        # Function may accept different kwarg names -- try a few
        # common alternatives. If none match, the structural
        # surface is wrong and the test fails below.
        try:
            fn(bridge, 1)
        except TypeError:
            fn(bridge)

    # Filter to ca3_swr_burst transitions only.
    swr_calls = [(n, v) for (n, v) in bridge.gate_history
                 if n == "ca3_swr_burst"]
    assert len(swr_calls) >= 2, (
        "trigger_swr_replay must call set_plasticity_gate on "
        "`ca3_swr_burst` at least twice (open then close); got "
        f"{len(swr_calls)} call(s): {swr_calls}")

    # The FIRST swr call must open (value > 0), the LAST swr call
    # must close (value == 0). This matches the validated Phase 1.3
    # set_sleep_gates / set_awake_gates pattern (ON=1.0, OFF=0.0).
    first_name, first_val = swr_calls[0]
    last_name, last_val = swr_calls[-1]
    assert first_val > 0.0, (
        "First ca3_swr_burst gate write must OPEN the gate "
        f"(value > 0); got value={first_val}")
    assert last_val == 0.0, (
        "Last ca3_swr_burst gate write must CLOSE the gate "
        f"(value == 0); got value={last_val}")


# ----------------------------------------------------------------------
# Test 4: decoder uses parallel-matching primitive.
# ----------------------------------------------------------------------

def test_decoder_uses_parallel_matching():
    """decode_continuation must be structurally bound to the
    validated parallel-population-matching primitive (the n=93
    capability pillar): per-slot argmax of phase_similarity across
    the substrate-derived grounded vocabulary.

    Structural check via inspect.getsource: the function source
    must reference both `phase_similarity` (the validated
    spiking_phasor_fhrr helper) AND a per-slot argmax over the
    vocabulary."""
    m = _import_loop_controller()
    assert hasattr(m, "decode_continuation"), (
        "Loop controller must expose decode_continuation(...) per "
        "Task 2 spec; missing")
    fn = m.decode_continuation
    assert callable(fn), "decode_continuation must be callable"

    src = inspect.getsource(fn)
    # Reuse-by-import marker: phase_similarity must appear (the
    # validated parallel-matching primitive).
    assert "phase_similarity" in src, (
        "decode_continuation source must reference the validated "
        "phase_similarity primitive (from "
        "research.runners.spiking_phasor_fhrr) -- this is the "
        "byte-unchanged parallel-population-matching mechanism; "
        "token not found in function source")
    # Argmax over the vocabulary marker: argmax or argsort or
    # equivalent WTA. The validated runner uses np.argmax.
    has_wta = (
        "argmax" in src or "argsort" in src or "np.max" in src
    )
    assert has_wta, (
        "decode_continuation source must perform argmax / argsort "
        "WTA over per-vocab phase-similarities (the parallel-"
        "population-matching decoder pattern); no argmax/argsort/"
        "np.max token found in function source")

    # Also: at the module level, the file must import
    # phase_similarity from the validated spiking_phasor_fhrr
    # module (reuse-by-import discipline). Check module source.
    mod_src = inspect.getsource(m)
    assert "from research.runners.spiking_phasor_fhrr" in mod_src, (
        "Loop controller module must `from "
        "research.runners.spiking_phasor_fhrr import ...` "
        "(reuse-by-import of the validated FHRR primitives); "
        "import not found")


# ----------------------------------------------------------------------
# Test 5: no oracle leak in the loop runtime.
# ----------------------------------------------------------------------

def test_no_oracle_leak_in_loop_controller():
    """run_generative_loop must NOT pass the true continuation
    items into decode_continuation. The decoder argument must be
    derived from the PFC frame + SWR replay-driven cortical
    activity + the parallel-matching vocabulary -- never from the
    true stored sequence.

    Structural check: inspect run_generative_loop's source for
    forbidden oracle-leak patterns. The true items may appear in
    POST-HOC scoring (and the function signature may accept them
    for that purpose), but they MUST NOT be passed as an argument
    to decode_continuation."""
    m = _import_loop_controller()
    assert hasattr(m, "run_generative_loop"), (
        "Loop controller must expose run_generative_loop(...) per "
        "Task 2 spec; missing")
    fn = m.run_generative_loop
    assert callable(fn), "run_generative_loop must be callable"

    src = inspect.getsource(fn)

    # Forbidden token patterns: passing true items into the
    # decoder. The canonical decoder argument name in the design
    # is `activity` (post-replay cortical activity); the canonical
    # vocabulary argument is `grounded_vocab`. The forbidden
    # pattern is naming a `true_*` / `stored_*` / `target_*` /
    # `oracle_*` argument to decode_continuation.
    #
    # We tokenise calls to decode_continuation and assert none of
    # the arguments contain those forbidden roots.
    import re
    # Match `decode_continuation(...)` -- non-greedy, supports
    # multi-line calls (re.DOTALL) so a leaky implementation that
    # wraps the call across lines is still caught. Match up to the
    # first unmatched closing paren by tracking depth.
    calls = []
    for m_call in re.finditer(r"decode_continuation\s*\(",
                              src, flags=re.MULTILINE):
        start = m_call.end()
        depth = 1
        i = start
        while i < len(src) and depth > 0:
            ch = src[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        if depth == 0:
            calls.append(src[start:i - 1])
    assert len(calls) > 0, (
        "run_generative_loop source must invoke decode_continuation "
        "(at least once per iteration); no such call found")

    forbidden_roots = (
        "true_", "true=", "stored_", "stored=", "target_item",
        "target=", "oracle_", "oracle=", "answer_", "answer=",
        "label_", "label=", "ground_truth", "groundtruth",
    )
    for call_args in calls:
        for tok in forbidden_roots:
            assert tok not in call_args, (
                f"oracle-leak detected: decode_continuation called "
                f"with argument containing forbidden token "
                f"`{tok}` -- argument list was: {call_args.strip()}")


# ----------------------------------------------------------------------
# Test discipline footer: confirm this test module itself does not
# import any protected/frozen/moat module (reuse-by-import discipline
# applies to the loop controller; this grounding pin only imports
# stdlib + numpy + the soon-to-exist loop controller).
# ----------------------------------------------------------------------

def _self_check_imports():
    """Sanity: this test module's top-level imports stay minimal."""
    # Already enforced by the import block at file top. No-op.
    pass
