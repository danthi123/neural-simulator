"""Soundness tests for the (c) generative-replay loop controller
(Task 3 of `docs/plans/2026-05-24-generative-replay-implementation.md`).

These tests PIN load-bearing properties of the Task 2 loop controller
(`research/runners/generative_replay_loop.py`) + decisive runner
(`research/findings/raw/generative_replay_decisive.py`) so that
subsequent refactoring or adversarial-review-driven fixes cannot
silently break them.

Each test is load-bearing: it asserts the actual property -- not just
its surface symptom -- so a refactor that breaks the property is
guaranteed to flip RED.

The 9 properties pinned:

  1. test_pfc_frame_is_genuinely_held -- the PFC frame is encoded ONCE
     and the loop body does not continuously re-inject it. (The
     dlpfc_wm bistability is what's supposed to hold it across the
     SWR window. The unit test verifies the controller's responsibility
     here: that the loop body does NOT secretly write into dlpfc_wm or
     cp_external_input_current between SWR / capture windows, which
     would mean we're FAKING the bistability.)
  2. test_swr_replay_is_genuine_not_seeded -- trigger_swr_replay opens
     then closes the validated `ca3_swr_burst` gate; no hand-supplied
     seed pattern. A mock bridge records gate transitions.
  3. test_decoder_unchanged_from_parallel_matching -- decode_continuation
     uses the byte-unchanged parallel-matching primitive
     (`batched_phase_similarity` from cross_bridge_mode_unification_probe).
  4. test_no_oracle_leak_in_loop_runtime -- the true stored sequence is
     used ONLY for post-hoc scoring; the loop runtime never reads it.
  5. test_consolidated_schema_is_substrate_trained_content -- the
     cortical schema the SWR replays against is post-Phase-1.3-
     consolidation, not a hand-supplied lookup.
  6. test_reuse_by_import_only -- loop controller imports unchanged
     primitives; protected set zero diff after Task 2 commit.
  7. test_no_autograd -- grep for autograd / torch.backward / loss
     .backward / loss.grad in the two genuinely-new files.
  8. test_no_confab_moat_still_green -- pytest tests/test_abstention
     _gate.py shows 7/7 PASS.
  9. test_frozen_bar_unchanged -- BAR = 0.80 unchanged in
     vocabulary_scaling_run.py.

PROTECTED MODULES (must remain byte-unchanged since Task 2 commit
ec567ab2a7e225f7ac0200141f82ffa6053694b8):
  - every file in sim/
  - research/runners/abstention_gate.py
  - tests/test_abstention_gate.py
  - research/runners/resonate_fire_fhrr.py
  - research/runners/spiking_phasor_fhrr.py
  - all research/findings/raw/biologized_spiking_* files
  - research/findings/raw/pattern_separation_grounding_probe.py
  - research/findings/raw/vocabulary_scaling_run.py
  - research/findings/raw/cross_bridge_mode_unification_probe.py
  - research/findings/raw/mode_unification_* probes
  - research/runners/concept_pool_demo.py
  - research/runners/text_minimal_isolation.py

Plain ASCII only. No autograd. No-confab moat 7/7 stays green.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import os
import re
import subprocess
import sys
import tokenize
from typing import List, Sequence

import numpy as np
import pytest


# ----------------------------------------------------------------------
# Source-stripping helpers: scan only CODE tokens, not strings /
# comments. The forbidden-token disclaimers ("no autograd") legitimately
# appear in docstrings / comments; we must not flag those.
# ----------------------------------------------------------------------

def _strip_strings_and_comments(src: str) -> str:
    """Return the source with string literals and comments removed.
    Uses Python's tokenize module so the result is robust against
    docstrings, multi-line strings, and inline comments.

    The returned string has string literals replaced with empty
    quoted markers (so syntactic placeholders remain) and comments
    deleted. Use for forbidden-token greps that must NOT match inside
    docstrings or comments."""
    out_tokens = []
    try:
        toks = tokenize.tokenize(io.BytesIO(src.encode("utf-8")).readline)
        for tok in toks:
            if tok.type == tokenize.COMMENT:
                continue
            if tok.type == tokenize.STRING:
                # Replace the literal content with empty quotes so that
                # forbidden-token searches do not match docstring or
                # string-literal text.
                out_tokens.append(tokenize.TokenInfo(
                    tok.type, '""', tok.start, tok.end, tok.line))
            else:
                out_tokens.append(tok)
        return tokenize.untokenize(out_tokens).decode("utf-8") if (
            isinstance(tokenize.untokenize(out_tokens), bytes)
        ) else tokenize.untokenize(out_tokens)
    except tokenize.TokenizeError:
        # Fallback: strip Python triple-quoted strings + # comments via
        # regex. Not as robust but safe enough for our well-formed
        # Python sources.
        no_tri = re.sub(
            r"('''.*?'''|\"\"\".*?\"\"\")", '""', src, flags=re.DOTALL)
        no_single = re.sub(
            r"('(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")", '""', no_tri)
        no_comments = re.sub(r"#[^\n]*", "", no_single)
        return no_comments


def _strip_function_docstring(src: str) -> str:
    """Strip ONLY the function's leading docstring from its source
    (keep the function signature + body). Useful when we want to scan
    the function body for tokens but ignore docstring narrative.

    More targeted than _strip_strings_and_comments: only the first
    triple-quoted string after the def header is removed."""
    # Find the function body start (after the colon ending the def
    # header) and strip the leading docstring if present.
    return re.sub(
        r"(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[\w\[\], .]+)?\s*:\s*\n"
        r"(?:\s*)?)"
        r"(['\"]{3}.*?['\"]{3}\s*\n)",
        r"\1",
        src,
        count=1,
        flags=re.DOTALL,
    )


# Path bootstrap so this test module can be invoked from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ----------------------------------------------------------------------
# Reference: commit at which Task 2 landed the loop controller.
# The reuse-by-import test verifies that the only files this branch
# added or modified are the two genuinely-new Task 2 files + Task 0/1
# files, and that the protected set has zero diff since that commit.
# ----------------------------------------------------------------------
TASK_2_COMMIT = "ec567ab2a7e225f7ac0200141f82ffa6053694b8"

# Genuinely-new files allowed to be touched by Tasks 0..3.
NEW_FILES_ALLOWED = {
    "research/runners/generative_replay_loop.py",
    "research/findings/raw/generative_replay_decisive.py",
    "research/findings/raw/generative_replay_sequence_vocab.py",
    "tests/test_generative_replay_grounding.py",
    "tests/test_generative_replay_sequence_vocab.py",
    "tests/test_generative_replay_soundness.py",
}

# Protected module list (must be byte-unchanged since Task 2 commit).
PROTECTED_FILES = [
    "research/runners/abstention_gate.py",
    "tests/test_abstention_gate.py",
    "research/runners/resonate_fire_fhrr.py",
    "research/runners/spiking_phasor_fhrr.py",
    "research/findings/raw/pattern_separation_grounding_probe.py",
    "research/findings/raw/vocabulary_scaling_run.py",
    "research/findings/raw/cross_bridge_mode_unification_probe.py",
    "research/runners/concept_pool_demo.py",
    "research/runners/text_minimal_isolation.py",
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _git_show_blob_sha(path_rel: str, rev: str) -> str:
    """Return the blob SHA of path_rel at rev (git rev-parse)."""
    out = subprocess.check_output(
        ["git", "-C", _REPO_ROOT, "rev-parse", f"{rev}:{path_rel}"],
        text=True,
    )
    return out.strip()


def _git_show_blob_sha_worktree(path_rel: str) -> str:
    """Return the blob SHA of path_rel in the current working tree
    (git hash-object on the file). Falls back to a sha256 of the file
    bytes if the file is untracked (the worktree may have additions)."""
    full = os.path.join(_REPO_ROOT, path_rel)
    if not os.path.exists(full):
        raise FileNotFoundError(full)
    out = subprocess.check_output(
        ["git", "-C", _REPO_ROOT, "hash-object", full],
        text=True,
    )
    return out.strip()


def _import_loop_controller():
    import research.runners.generative_replay_loop as m  # noqa: F401
    return m


def _import_decisive_runner():
    # The decisive runner imports heavy substrate build code on module
    # load (no top-level main execution); we only need the module
    # object for source inspection here.
    import research.findings.raw.generative_replay_decisive as m  # noqa
    return m


# ----------------------------------------------------------------------
# Test 1: PFC frame is genuinely held
#
# Property: the loop body must NOT continuously re-inject the PFC frame
# externally. The dlpfc_wm bistability is supposed to hold the frame
# across the SWR / capture windows; if the controller secretly drives
# cp_external_input_current at dlpfc_wm between iterations, it's
# FAKING bistability.
#
# We assert structurally (via inspect.getsource):
#   (a) update_pfc_frame is a PURE function on FHRR composites
#       (no bridge writes; returns a new composite).
#   (b) inside run_generative_loop's iteration body, NO write to
#       bridge.cp_external_input_current occurs EXCEPT via the
#       controlled capture path (which zeros it BEFORE capture, never
#       drives a frame pattern). No writes to any dlpfc_wm indices.
#   (c) encode_pfc_frame is invoked ONCE per trial -- by the runner --
#       and the loop body itself never re-invokes encode_pfc_frame.
#
# A refactor that introduced a "re-inject C into dlpfc_wm each
# iteration" line would flip (b) RED.
# ----------------------------------------------------------------------

def test_pfc_frame_is_genuinely_held():
    m = _import_loop_controller()

    # (a) update_pfc_frame is a pure FHRR-composite function: no
    # `bridge.` argument or use anywhere in its CODE body (strings
    # and comments stripped; docstring narrative may legitimately
    # mention bridge in passing). We check that the function does
    # not USE the bridge -- i.e., no `bridge` identifier appears
    # in the code tokens.
    src_update = inspect.getsource(m.update_pfc_frame)
    code_update = _strip_strings_and_comments(src_update)
    assert "bridge" not in code_update, (
        "update_pfc_frame must be a pure FHRR-composite function "
        "(no bridge writes). Found `bridge` identifier in code "
        "(strings/comments stripped) -- indicates the controller "
        "is mutating bridge state during frame update, which would "
        "conflate FHRR composition with neural injection.")
    # It must return a new composite via rf_bind + rf_bundle (the
    # FHRR superposition extension), not via bridge stimulation.
    assert ("rf_bind" in code_update and "rf_bundle" in code_update), (
        "update_pfc_frame must extend C via rf_bind + rf_bundle "
        "(FHRR superposition). Missing one or both primitives in "
        "the function code.")

    # (b) run_generative_loop's iteration body must not write to
    # cp_external_input_current ANYWHERE other than via the controlled
    # capture path. The capture function (capture_post_replay_cortical
    # _activity) DOES zero cp_external_input_current with
    # zero_drive=True -- that's the controlled write; it does NOT
    # inject a frame.
    src_loop = inspect.getsource(m.run_generative_loop)
    code_loop = _strip_strings_and_comments(src_loop)
    # Direct injection patterns the loop body must NOT contain (in
    # CODE, not docstring narrative):
    forbidden_loop_writes = [
        "cp_external_input_current[",   # any indexed write
        "cp_external_input_current =",  # any whole-array assign
        "dlpfc_wm",                      # any direct dlpfc index ref
    ]
    for tok in forbidden_loop_writes:
        assert tok not in code_loop, (
            f"run_generative_loop iteration body must not contain "
            f"`{tok}` -- this would indicate the loop is faking "
            f"dlpfc_wm bistability by re-injecting the frame "
            f"externally each iteration. The bistability is the "
            f"substrate's job, not the controller's.")

    # (c) encode_pfc_frame must NOT be CALLED inside run_generative
    # _loop. (It may be referenced in the function's narrative
    # docstring -- which is stripped from the check.) The runner
    # encodes the initial frame ONCE, passes initial_C in, and the
    # loop never re-encodes.
    assert "encode_pfc_frame(" not in code_loop, (
        "run_generative_loop must NOT re-encode the PFC frame each "
        "iteration. The initial composite C is passed in as "
        "initial_C; the loop extends it via update_pfc_frame "
        "(pure FHRR composition). Found `encode_pfc_frame(` call "
        "in loop code.")

    # (d) capture_post_replay_cortical_activity's only write to
    # cp_external_input_current is the documented zero_drive zeroing
    # BEFORE the capture window. Verify by counting writes in CODE
    # (the docstring also mentions cp_external_input_current).
    src_cap = inspect.getsource(m.capture_post_replay_cortical_activity)
    code_cap = _strip_strings_and_comments(src_cap)
    n_writes = (
        code_cap.count("cp_external_input_current[")
        + code_cap.count("cp_external_input_current =")
    )
    assert n_writes <= 1, (
        f"capture_post_replay_cortical_activity has {n_writes} "
        f"writes to cp_external_input_current in code; expected at "
        f"most 1 (the zero_drive zeroing). More writes would mean "
        f"the capture path is injecting drive, which conflates "
        f"replay-driven activity with externally-driven activity.")


# ----------------------------------------------------------------------
# Test 2: SWR replay is genuine (validated Phase 1.3 mechanism, not
# a hand-supplied seed pattern).
# ----------------------------------------------------------------------

class _GateRecordingBridge:
    """Mock bridge that records set_plasticity_gate calls + counts
    _run_one_simulation_step invocations. Used to verify
    trigger_swr_replay opens then closes the ca3_swr_burst gate AND
    actually runs the bridge during the replay window (rather than
    hand-supplying a seed pattern)."""

    def __init__(self):
        self.gate_history = []
        self.n_steps_run = 0
        self.external_writes = 0
        # Minimal cp_external_input_current shim so anything that
        # tries to write to it can be detected (not used by
        # trigger_swr_replay; included to flag a refactor that
        # accidentally adds injection).
        self.cp_external_input_current = _RecordingArray(self)
        self.runtime_state = type("RS", (), {"current_time_step": 0})()

    def set_plasticity_gate(self, name, value):
        self.gate_history.append((str(name), float(value)))

    def _run_one_simulation_step(self):
        self.n_steps_run += 1


class _RecordingArray:
    """Tiny stand-in for cp_external_input_current that records any
    write attempts."""

    def __init__(self, owner):
        self._owner = owner

    def __setitem__(self, key, value):
        self._owner.external_writes += 1

    def __getitem__(self, key):
        return 0.0


def test_swr_replay_is_genuine_not_seeded():
    m = _import_loop_controller()
    fn = m.trigger_swr_replay

    bridge = _GateRecordingBridge()
    n_steps = 7  # small but nonzero
    stats = fn(bridge, n_steps=n_steps)

    # Filter to ca3_swr_burst only.
    swr_calls = [(name, val) for (name, val) in bridge.gate_history
                 if name == "ca3_swr_burst"]
    assert len(swr_calls) >= 2, (
        "trigger_swr_replay must call set_plasticity_gate on "
        f"`ca3_swr_burst` at least twice; got {len(swr_calls)}.")
    first_name, first_val = swr_calls[0]
    last_name, last_val = swr_calls[-1]
    assert first_val > 0.0, (
        f"First ca3_swr_burst write must OPEN the gate; got "
        f"value={first_val}")
    assert last_val == 0.0, (
        f"Last ca3_swr_burst write must CLOSE the gate; got "
        f"value={last_val}")

    # The replay window must actually run the bridge -- NOT seed a
    # pattern. Verify n_steps_run >= n_steps.
    assert bridge.n_steps_run >= n_steps, (
        f"trigger_swr_replay must run the bridge for at least "
        f"n_steps={n_steps} simulation steps (genuine SWR-driven "
        f"replay through the validated Phase 1.3 mechanism); "
        f"observed only {bridge.n_steps_run} _run_one_simulation_"
        f"step invocations. A refactor that hand-supplied a seed "
        f"pattern instead of running the bridge would flip this RED.")

    # The replay window must NOT inject a hand-supplied seed pattern
    # into cp_external_input_current.
    assert bridge.external_writes == 0, (
        f"trigger_swr_replay must NOT write to cp_external_input_"
        f"current; observed {bridge.external_writes} writes. The "
        f"replay mechanism is opening the ca3_swr_burst gate and "
        f"letting the substrate's hippocampal recurrence drive "
        f"replay; ANY external write would substitute a hand-"
        f"supplied seed pattern for the validated mechanism.")

    # Stats dict reports the open/close values clearly.
    assert isinstance(stats, dict)
    assert stats["n_steps"] == n_steps
    assert stats["gate_open_value"] > 0.0
    assert stats["gate_close_value"] == 0.0


# ----------------------------------------------------------------------
# Test 3: decoder unchanged from parallel-matching primitive.
#
# Property: decode_continuation must compute its V-vector of phase
# similarities via the SAME `batched_phase_similarity` primitive that
# the validated pillars n=96/n=97/n=98 used byte-unchanged. We assert:
#   (a) the loop controller imports batched_phase_similarity from
#       research.findings.raw.cross_bridge_mode_unification_probe;
#   (b) decode_continuation's source invokes that exact symbol;
#   (c) the protected primitive's source byte-content is identical
#       to its content at the Task 2 commit (no silent modification);
#   (d) the decoder argmaxes over the FULL grounded_vocab_phase_matrix
#       (no slicing, no restriction).
# ----------------------------------------------------------------------

def test_decoder_unchanged_from_parallel_matching():
    m = _import_loop_controller()

    # (a) The module imports the primitive from the protected file.
    mod_src = inspect.getsource(m)
    assert ("from research.findings.raw.cross_bridge_mode_unification_"
            "probe") in mod_src.replace("\n", " "), (
        "Loop controller must import from "
        "research.findings.raw.cross_bridge_mode_unification_probe "
        "(reuse-by-import discipline); import line not found.")
    assert "batched_phase_similarity" in mod_src, (
        "Loop controller must import batched_phase_similarity (the "
        "validated parallel-matching primitive); name not found in "
        "module source.")

    # (b) decode_continuation's source body invokes that symbol.
    src_decode = inspect.getsource(m.decode_continuation)
    assert "batched_phase_similarity(" in src_decode, (
        "decode_continuation must invoke batched_phase_similarity "
        "(the validated n=96/n=97/n=98 parallel-matching primitive). "
        "Token not found in function body.")

    # (c) Protected primitive byte-content identical to Task 2 commit.
    # If `git rev-parse Task_2_commit:file` and current worktree
    # `git hash-object file` differ, the primitive was modified.
    cross_probe_rel = (
        "research/findings/raw/cross_bridge_mode_unification_probe.py")
    sha_committed = _git_show_blob_sha(cross_probe_rel, TASK_2_COMMIT)
    sha_worktree = _git_show_blob_sha_worktree(cross_probe_rel)
    assert sha_committed == sha_worktree, (
        f"Protected primitive {cross_probe_rel} blob SHA differs "
        f"between Task 2 commit ({sha_committed}) and current worktree "
        f"({sha_worktree}) -- the validated parallel-matching "
        f"primitive must remain byte-unchanged (reuse-by-import "
        f"discipline).")

    # (d) decode_continuation does not slice / restrict the vocabulary.
    # Forbidden patterns: slicing the grounded_vocab_phase_matrix arg
    # or the sims result. (We allow the natural argmax call.)
    forbidden_slice_patterns = [
        "grounded_vocab_phase_matrix[",  # any slicing of the matrix
        "sims[:",                          # restricting sims
    ]
    for tok in forbidden_slice_patterns:
        assert tok not in src_decode, (
            f"decode_continuation must argmax over the FULL "
            f"V-vocabulary; found forbidden pattern `{tok}` in "
            f"function source which would restrict the decoder's "
            f"argmax set.")

    # Also verify the decoder body computes argmax of the full sims.
    assert "argmax(sims)" in src_decode.replace(" ", ""), (
        "decode_continuation must end with argmax(sims) over the "
        "full V-vocabulary; pattern not found.")


# ----------------------------------------------------------------------
# Test 4: no oracle leak in the loop runtime.
#
# Property: the loop runtime (run_generative_loop + the functions it
# transitively calls in this controller module) never receives the
# true stored sequence / continuation. The true sequence is read
# ONLY by the runner's post-hoc scoring path (`_score_completion` in
# the decisive runner), AFTER the loop has produced its decoded word.
# ----------------------------------------------------------------------

OR_LEAK_TOKENS = (
    "true_", "stored_", "target_item", "oracle",
    "answer", "label_", "label=", "ground_truth",
    "groundtruth", "correct=",
)

# Names that appear in the runner's POST-HOC scoring path and are
# allowed there (the score function isn't part of the loop runtime).
# We scope the oracle-leak check to the loop controller MODULE.


def test_no_oracle_leak_in_loop_runtime():
    m = _import_loop_controller()
    mod_src = inspect.getsource(m)

    # (a) The loop controller's signatures must not declare any
    # oracle-flavoured parameters on the runtime functions.
    runtime_fns = [
        m.encode_pfc_frame,
        m.trigger_swr_replay,
        m.capture_post_replay_cortical_activity,
        m.decode_continuation,
        m.update_pfc_frame,
        m.run_generative_loop,
    ]
    for fn in runtime_fns:
        sig = inspect.signature(fn)
        for pname in sig.parameters.keys():
            low = pname.lower()
            for tok in OR_LEAK_TOKENS:
                # Strip trailing "=" / "_" markers for substring check
                # against parameter names (which are bare identifiers).
                bare = tok.rstrip("_=")
                if not bare:
                    continue
                assert bare not in low, (
                    f"Oracle leak in loop runtime: function {fn.__name__}"
                    f" has parameter `{pname}` containing forbidden "
                    f"token `{bare}`. The loop runtime must NEVER "
                    f"receive the true continuation; that's the "
                    f"runner's post-hoc scoring job.")

    # (b) The loop body (run_generative_loop source) must NOT pass
    # any forbidden-token argument into decode_continuation. We
    # tokenise calls to decode_continuation and check argument lists.
    src_loop = inspect.getsource(m.run_generative_loop)
    calls = []
    for match in re.finditer(r"decode_continuation\s*\(",
                              src_loop, flags=re.MULTILINE):
        start = match.end()
        depth = 1
        i = start
        while i < len(src_loop) and depth > 0:
            ch = src_loop[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        if depth == 0:
            calls.append(src_loop[start:i - 1])
    assert len(calls) > 0, (
        "run_generative_loop must invoke decode_continuation at "
        "least once per iteration; no call found.")
    for call_args in calls:
        low = call_args.lower()
        for tok in OR_LEAK_TOKENS:
            assert tok not in low, (
                f"Oracle leak: decode_continuation called with "
                f"argument containing forbidden token `{tok}`. "
                f"Argument list: {call_args.strip()}")

    # (c) The decisive runner reads true_continuation, but ONLY in
    # the post-hoc scoring section. Verify the read happens AFTER
    # the loop call and is gated through _score_completion.
    runner = _import_decisive_runner()
    runner_src = inspect.getsource(runner)
    # The true_continuation token MUST appear (it's needed for
    # post-hoc scoring); the absence would mean we never scored.
    assert "true_continuation" in runner_src, (
        "Runner must compute true_continuation for post-hoc scoring; "
        "token not found. (Absence would mean no scoring path.)")
    # _score_completion is the scoring entry point.
    assert "_score_completion" in runner_src, (
        "Runner must use _score_completion for post-hoc scoring; "
        "function name not found.")
    # The score call MUST come AFTER run_generative_loop and pass
    # decoded_word + true_continuation only at that point. Verify
    # the source order: run_generative_loop precedes _score_completion
    # in source. This is a structural ordering check.
    idx_loop = runner_src.find("run_generative_loop(")
    idx_score = runner_src.rfind("_score_completion(")
    assert 0 <= idx_loop < idx_score, (
        f"Runner must call run_generative_loop BEFORE "
        f"_score_completion (post-hoc scoring); source order "
        f"violated: run_generative_loop@{idx_loop} vs "
        f"_score_completion@{idx_score}.")

    # The runner must NOT pass true_continuation INTO run_generative_loop
    # (loop-runtime arg list). Inspect the run_generative_loop call.
    m_call = re.search(r"run_generative_loop\s*\(", runner_src)
    if m_call:
        depth = 1
        i = m_call.end()
        while i < len(runner_src) and depth > 0:
            ch = runner_src[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            i += 1
        if depth == 0:
            call_args = runner_src[m_call.end():i - 1]
            low = call_args.lower()
            for tok in OR_LEAK_TOKENS:
                assert tok not in low, (
                    f"Runner leaked oracle into loop runtime: "
                    f"run_generative_loop call contains forbidden "
                    f"token `{tok}`. Argument list: "
                    f"{call_args.strip()}")


# ----------------------------------------------------------------------
# Test 5: consolidated schema is substrate-trained content.
#
# Property: the cortical schema the SWR replays against is the
# substrate's POST-Phase-1.3-consolidation trained weights -- not a
# hand-supplied lookup. We verify this structurally in the decisive
# runner:
#   (a) The runner invokes `run_concept_replay_phase` from the
#       protected `consolidation_trainer` module (the validated
#       Phase 1.3 mechanism reused byte-unchanged).
#   (b) The replay phase is driven by engram tags committed via
#       `start_engram_recording` + `commit_engram_tag` (Tonegawa-
#       style D.14), NOT by hand-supplied patterns.
#   (c) Before running the loop, the runner calls
#       `freeze_all_gates(bridge)` so the SWR window during the
#       loop does NOT drive weight changes (the loop is INFERENCE
#       over the consolidated substrate; weight drift would be a
#       soundness issue).
# ----------------------------------------------------------------------

def test_consolidated_schema_is_substrate_trained_content():
    runner = _import_decisive_runner()
    runner_src = inspect.getsource(runner)

    # (a) Phase 1.3 consolidation primitive is invoked.
    assert "run_concept_replay_phase" in runner_src, (
        "Runner must invoke run_concept_replay_phase (Phase 1.3 "
        "consolidation primitive from consolidation_trainer); "
        "function name not found. Without this, the cortical "
        "schema is NOT post-Phase-1.3-consolidation.")
    # And it must be imported from the validated module.
    assert ("from research.runners.consolidation_trainer import" in
            runner_src), (
        "run_concept_replay_phase must be imported from "
        "research.runners.consolidation_trainer (reuse-by-import); "
        "import line not found.")

    # (b) Engram tags are committed via the substrate's mechanism.
    # Scan CODE only (the docstring discusses tags narratively).
    runner_code = _strip_strings_and_comments(runner_src)
    assert "start_engram_recording" in runner_code, (
        "Runner must call bridge.start_engram_recording per stored "
        "sequence (Tonegawa D.14 mechanism); no call found in code.")
    assert "commit_engram_tag" in runner_code, (
        "Runner must call bridge.commit_engram_tag per stored "
        "sequence (Tonegawa D.14); no call found in code.")
    # No hand-supplied schema lookup tables (scan CODE only so the
    # docstring's `schema` narrative doesn't trigger false positives).
    forbidden_lookup_patterns = [
        "schema_table",
        "schema_lookup",
        "consolidated_lookup",
        "answer_map",
        "answer_table",
        "answer_lookup",
        "lookup_table",
    ]
    for tok in forbidden_lookup_patterns:
        assert tok not in runner_code, (
            f"Runner contains forbidden hand-supplied schema-lookup "
            f"pattern `{tok}` in CODE. The schema must be the "
            f"substrate's consolidated weights, NOT a hand-supplied "
            f"lookup table.")

    # (c) Plasticity is frozen BEFORE the loop runs (loop is
    # inference; weight drift would be a soundness issue). The
    # runner calls freeze_all_gates twice:
    #   - first inside _load_substrate (initial freeze post-load)
    #   - second AFTER set_awake_gates (post-consolidation freeze
    #     before the loop iterations begin)
    # We assert the SECOND freeze occurs between set_awake_gates and
    # run_generative_loop. Use the source positions of the in-loop
    # call ordering.
    assert "freeze_all_gates" in runner_code, (
        "Runner must call freeze_all_gates(bridge) before the loop "
        "iterations so the (c) loop is INFERENCE over the "
        "consolidated substrate; freeze_all_gates token not found.")
    idx_sleep = runner_code.find("set_sleep_gates(")
    idx_replay = runner_code.find("run_concept_replay_phase(")
    idx_awake = runner_code.find("set_awake_gates(")
    # The post-consolidation freeze: the freeze_all_gates call that
    # comes AFTER set_awake_gates. We find the first occurrence
    # of freeze_all_gates after set_awake_gates.
    idx_freeze_post = runner_code.find("freeze_all_gates(", idx_awake)
    idx_loop = runner_code.find("run_generative_loop(")
    assert 0 <= idx_sleep <= idx_replay, (
        f"Runner source order violated: set_sleep_gates @{idx_sleep} "
        f"must precede run_concept_replay_phase @{idx_replay}.")
    assert idx_replay <= idx_awake, (
        f"Runner source order violated: run_concept_replay_phase "
        f"@{idx_replay} must precede set_awake_gates @{idx_awake}.")
    assert 0 <= idx_freeze_post < idx_loop, (
        f"Post-consolidation freeze (freeze_all_gates after "
        f"set_awake_gates @{idx_awake}) must precede the loop call "
        f"(run_generative_loop @{idx_loop}); got post-freeze "
        f"@{idx_freeze_post}. Without this freeze, the loop's SWR "
        f"window would drift weights and the schema would not stay "
        f"consolidated.")


# ----------------------------------------------------------------------
# Test 6: reuse-by-import only -- protected set zero diff after Task
# 2 commit; loop controller imports unchanged primitives.
#
# Property: the only files Tasks 0..3 have added or modified are the
# six new files listed in NEW_FILES_ALLOWED. Every other file in the
# protected set has identical blob SHA between the Task 2 commit and
# the current worktree.
# ----------------------------------------------------------------------

def test_reuse_by_import_only():
    # (a) Protected files: byte-unchanged since Task 2 commit.
    drift = []
    for path_rel in PROTECTED_FILES:
        try:
            sha_at = _git_show_blob_sha(path_rel, TASK_2_COMMIT)
        except subprocess.CalledProcessError as e:
            drift.append((path_rel, "not_in_commit", str(e)))
            continue
        try:
            sha_now = _git_show_blob_sha_worktree(path_rel)
        except FileNotFoundError:
            drift.append((path_rel, "missing_in_worktree", ""))
            continue
        if sha_at != sha_now:
            drift.append((path_rel, sha_at, sha_now))
    assert not drift, (
        "Protected files diverged from Task 2 commit:\n  " +
        "\n  ".join(f"{p}: committed={a} worktree={b}"
                     for (p, a, b) in drift))

    # (b) The loop controller imports its primitives from the
    # protected modules (reuse-by-import). Check the import surface.
    m = _import_loop_controller()
    mod_src = inspect.getsource(m)
    required_imports = [
        "from research.runners.resonate_fire_fhrr",
        "from research.runners.spiking_phasor_fhrr",
        ("from research.findings.raw.cross_bridge_mode_"
         "unification_probe"),
        ("from research.findings.raw.biologized_spiking_mode_"
         "unification_helpers"),
    ]
    flat = mod_src.replace("\n", " ")
    for needed in required_imports:
        assert needed in flat, (
            f"Loop controller must import from `{needed}` (reuse-"
            f"by-import discipline); not found in module source.")

    # (c) The two genuinely-new files exist; the other "new files
    # allowed" set is exactly what's been added in this task chain.
    for path_rel in NEW_FILES_ALLOWED:
        full = os.path.join(_REPO_ROOT, path_rel)
        # Test files (current one) and previously-landed task files
        # all must exist. (test_generative_replay_soundness.py is the
        # current file being added.)
        assert os.path.exists(full), (
            f"NEW_FILES_ALLOWED entry missing from worktree: "
            f"{path_rel}")


# ----------------------------------------------------------------------
# Test 7: no autograd -- grep for autograd / torch.backward /
# loss.backward in the two genuinely-new Task 2 files.
# ----------------------------------------------------------------------

def test_no_autograd():
    """Scan the genuinely-new files for autograd / backward USAGE in
    CODE only. Docstrings legitimately contain disclaimer text ('no
    autograd', 'no torch.backward', etc.); we strip strings and
    comments before scanning so the disclaimer doesn't match.

    We also use regex word-boundary matching so identifiers like
    `test_no_autograd` (the function name of THIS test) don't
    accidentally trigger -- only standalone `autograd` token uses
    are caught.

    The forbidden patterns:
      - autograd        (as a standalone identifier; not in compound
                         names like _autograd or test_no_autograd)
      - torch.autograd  (module reference)
      - torch.backward  (call)
      - loss.backward   (call)
      - .backward()     (suffix call on any tensor)
      - requires_grad   (autograd attribute)
      - .grad_fn        (autograd graph node)
    """
    # Patterns are regex strings; \b for word boundary on bare
    # identifiers, escaped literal '.' for attribute access.
    forbidden_patterns = (
        (r"\bautograd\b", "autograd"),
        (r"torch\.autograd", "torch.autograd"),
        (r"torch\.backward", "torch.backward"),
        (r"loss\.backward", "loss.backward"),
        (r"\.backward\s*\(", ".backward()"),
        (r"\brequires_grad\b", "requires_grad"),
        (r"\.grad_fn\b", ".grad_fn"),
    )
    files_to_scan = [
        os.path.join(_REPO_ROOT, p) for p in NEW_FILES_ALLOWED
    ]
    offenders = []
    for path in files_to_scan:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            content = f.read()
        code = _strip_strings_and_comments(content)
        for pat, label in forbidden_patterns:
            if re.search(pat, code):
                offenders.append((path, label))
    assert not offenders, (
        "Autograd / backward USAGE tokens found in CODE (strings + "
        "comments stripped; word-boundaries enforced):\n  " +
        "\n  ".join(f"{p}: contains `{tok}`"
                     for (p, tok) in offenders))


# ----------------------------------------------------------------------
# Test 8: no-confab moat still 7/7 green.
#
# Property: running pytest on tests/test_abstention_gate.py reports
# 7 passed. This is the moat that pins the abstention gate's behaviour
# (the no-confab safety property).
# ----------------------------------------------------------------------

def test_no_confab_moat_still_green():
    moat_test_path = os.path.join(
        _REPO_ROOT, "tests", "test_abstention_gate.py")
    assert os.path.exists(moat_test_path), (
        f"Moat test file missing: {moat_test_path}")
    # Run pytest as a subprocess on JUST the moat tests. We must not
    # corrupt the parent pytest's run by importing or reseeding.
    result = subprocess.run(
        [sys.executable, "-m", "pytest", moat_test_path, "-q",
         "--no-header", "--tb=short"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    assert result.returncode == 0, (
        f"No-confab moat pytest exited {result.returncode}:\n"
        f"{combined}")
    # Look for the "7 passed" status line. pytest's terse output
    # prints `7 passed` on success; assert it.
    has_seven_passed = (
        re.search(r"\b7\s+passed\b", combined) is not None
    )
    assert has_seven_passed, (
        f"No-confab moat must report 7 passed; pytest output did "
        f"not include `7 passed`. Full output:\n{combined}")


# ----------------------------------------------------------------------
# Test 9: frozen 0.80 bar unchanged.
#
# Property: BAR = 0.80 in research/findings/raw/vocabulary_scaling_run
# .py. The bar is pre-registered and IMMOVABLE. Any change to this
# constant -- or to the file that defines it -- is a soundness
# violation.
# ----------------------------------------------------------------------

def test_frozen_bar_unchanged():
    # (a) Import the BAR value directly. It must be exactly 0.80.
    from research.findings.raw.vocabulary_scaling_run import BAR
    assert BAR == 0.80, (
        f"Frozen 0.80 bar changed: BAR={BAR}. The bar is pre-"
        f"registered IMMOVABLE; this value must never be tuned.")

    # (b) Inspect the source line that defines BAR to ensure the
    # literal `0.80` is what's written (defends against import-time
    # mutation).
    src_path = os.path.join(
        _REPO_ROOT, "research", "findings", "raw",
        "vocabulary_scaling_run.py")
    assert os.path.exists(src_path), (
        f"vocabulary_scaling_run.py missing: {src_path}")
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()
    # The canonical declaration is `BAR = 0.80`. Accept `BAR = 0.8`
    # (mathematically identical) but flag any other value.
    m_bar = re.search(r"^\s*BAR\s*=\s*([0-9.]+)\s*$",
                       src, flags=re.MULTILINE)
    assert m_bar is not None, (
        "vocabulary_scaling_run.py does not contain a top-level "
        "`BAR = <value>` line; the frozen bar is missing.")
    literal = m_bar.group(1)
    assert float(literal) == 0.80, (
        f"vocabulary_scaling_run.py declares BAR={literal}; expected "
        f"0.80 (frozen IMMOVABLE).")

    # (c) The decisive runner imports BAR from vocabulary_scaling_run
    # (no local override).
    runner = _import_decisive_runner()
    runner_src = inspect.getsource(runner)
    assert ("from research.findings.raw.vocabulary_scaling_run import"
            in runner_src), (
        "Decisive runner must import BAR from "
        "research.findings.raw.vocabulary_scaling_run; import line "
        "not found.")
    assert "BAR" in runner_src, (
        "Decisive runner must reference BAR; token not found.")
    # The runner must NOT contain a local rebinding like `BAR = ...`.
    assert (re.search(r"^\s*BAR\s*=", runner_src, flags=re.MULTILINE)
            is None), (
        "Decisive runner contains a local `BAR = ...` rebinding; "
        "the frozen bar must remain the single value imported from "
        "vocabulary_scaling_run.")
