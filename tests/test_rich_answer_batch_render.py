"""Byte-identical-equivalence + gating tests for the batched sentence-render path (2026-09-01,
`research/batch-sentence-rendering`, board frontier "make each chat reply snappier by batching the sentence
rendering").

WHAT THIS COVERS (CPU-only, no GPU, no model download -- a FAKE renderer stands in for the real off-bridge
`QwenRenderer` so this runs fast in CI). The real Qwen-backed batched-vs-sequential TEXT equivalence + the
measured latency are covered separately by
`research.runners._batch_sentence_rendering_derisk` (GPU, real model, writes a JSON verdict artifact under
research/findings/raw/_batch_sentence_rendering/) -- that file is the load-bearing latency/equivalence
measurement; THIS file is the fast, deterministic proof of the WIRING itself:
  1. Flag OFF (`BRAIN_RICH_BATCH_RENDER` unset) -> `RichAnswerComposer.render_paragraph` NEVER calls
     `renderer.render_svo_batch`, even when the renderer exposes one -- proved by an invocation-count
     assertion (`render_svo_batch` call count == 0), not just output equality.
  2. Flag ON + >1 gathered fact + a renderer exposing `render_svo_batch` -> the batched path IS taken (exactly
     ONE `render_svo_batch` call, ZERO per-item `render_svo` calls for the facts that batch cleanly) and its
     output is BYTE-IDENTICAL to the sequential (flag-OFF) path's output for the SAME facts.
  3. A renderer WITHOUT `render_svo_batch` (e.g. the pre-existing `StubRenderer`) -> byte-identical no-op
     regardless of the flag (the `hasattr` guard).
  4. A single-fact turn -> never batches even with the flag ON (nothing to batch).
  5. A batched candidate that fails VERIFY falls back to the pre-existing single-item path (regen included),
     so the moat is never weaker than before this feature existed.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SIM_BACKEND", "numpy")

BATCH_FLAG = "BRAIN_RICH_BATCH_RENDER"
# `BRAIN_SPIKING_MOUTH_RECALL` is production DEFAULT-ON (2026-08-26 flip): `spiking_recall_surface` (checked
# FIRST by BOTH the sequential and the batched path, identically -- see `_render_one_verified` /
# `_render_paragraph_batched`) would otherwise render a bounded-transitive-SVO fact on the spiking Broca
# WITHOUT ever reaching this test's fake renderer, leaving nothing to batch. This file isolates the RENDER-
# BATCHING wiring under test from that (already separately verified, orthogonal) production feature by forcing
# it off for the duration of these tests only -- the SAME per-fact call this test's fake renderer would
# otherwise never see exercised.
_ENV_OVERRIDES = {BATCH_FLAG: None, "BRAIN_SPIKING_MOUTH_RECALL": "0"}


@pytest.fixture(autouse=True)
def _clean_flag():
    """Force a known env state for every test in this file, and never let it leak into another test."""
    prev = {k: os.environ.get(k) for k in _ENV_OVERRIDES}
    for k, v in _ENV_OVERRIDES.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    yield
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _CountingBatchRenderer:
    """A deterministic, GPU-free stand-in for `QwenRenderer` that ALSO exposes `render_svo_batch`, wrapping the
    same `TemplateStubFaculty` the pre-existing `StubRenderer` uses (so single-item output is byte-for-byte the
    project's own established deterministic renderer) -- this isolates the WIRING under test (call routing,
    fallback, order) from any question about a real generative model's own determinism."""

    name = "counting-batch-stub (test-only)"

    def __init__(self):
        from research.runners._grounded_lang_p3_derisk import TemplateStubFaculty
        self._fac = TemplateStubFaculty()
        self.render_svo_calls = []
        self.render_svo_batch_calls = []

    def render_svo(self, a, v, p):
        self.render_svo_calls.append((a, v, p))
        return self._fac.render_svo(a, v, p)

    def render_svo_batch(self, triples):
        self.render_svo_batch_calls.append(list(triples))
        out = [self._fac.render_svo(a, v, p) for (a, v, p) in triples]
        return out, 0.0


def _build_rich(seed=42):
    from research.runners.rich_answer_composer import _build_smoke_chat, RichAnswerComposer
    chat = _build_smoke_chat(seed, use_multiturn=True)
    renderer = _CountingBatchRenderer()
    chat.renderer = renderer
    rich = RichAnswerComposer(chat, max_chain_hops=5, max_elaborations=2, max_sentences=6)
    return chat, rich, renderer


def _gather_multi_fact(rich):
    topic, facts = rich.gather("what are you", followup=False)
    assert len(facts) > 1, f"test fixture must gather >1 fact to exercise batching, got {facts!r}"
    return topic, facts


def test_flag_off_never_calls_render_svo_batch():
    """BYTE-IDENTICAL-WHEN-OFF, by invocation count, not just output: with the flag unset, `render_paragraph`
    must call `render_svo` once per fact and `render_svo_batch` ZERO times, even though the active renderer
    exposes it."""
    os.environ.pop(BATCH_FLAG, None)
    _chat, rich, renderer = _build_rich()
    _topic, facts = _gather_multi_fact(rich)
    paragraph, kept, dropped = rich.render_paragraph(facts)
    assert renderer.render_svo_batch_calls == []
    assert len(renderer.render_svo_calls) == len(facts)
    assert kept == facts
    assert dropped == []
    assert paragraph


def test_flag_on_batches_and_matches_sequential_output_byte_identical():
    """Flag ON + a renderer with `render_svo_batch` + >1 fact -> the batched path is taken (one
    `render_svo_batch` call, no per-item `render_svo` calls for the facts that batch cleanly), and its
    (paragraph, kept, dropped) is BYTE-IDENTICAL to the sequential (flag-OFF) run on the SAME facts."""
    os.environ.pop(BATCH_FLAG, None)
    chat_off, rich_off, renderer_off = _build_rich()
    _topic, facts = _gather_multi_fact(rich_off)
    off_result = rich_off.render_paragraph(facts)

    os.environ[BATCH_FLAG] = "1"
    chat_on, rich_on, renderer_on = _build_rich()
    on_result = rich_on.render_paragraph(facts)

    assert len(renderer_on.render_svo_batch_calls) == 1
    assert renderer_on.render_svo_calls == []          # nothing fell back to the per-item path
    assert renderer_on.render_svo_batch_calls[0] == [tuple(f) for f in facts]

    assert on_result == off_result, f"batched {on_result!r} != sequential {off_result!r}"


def test_stub_renderer_without_batch_support_is_unaffected_by_the_flag():
    """A renderer that does NOT expose `render_svo_batch` (the pre-existing `StubRenderer`, and every renderer
    that predates this feature) is a byte-identical no-op under the flag -- the `hasattr` guard, not a
    try/except, so this is a STATIC property, not a runtime fallback."""
    from research.runners.rich_answer_composer import _build_smoke_chat, RichAnswerComposer
    from research.runners.brain_chat_tui import StubRenderer

    os.environ.pop(BATCH_FLAG, None)
    chat = _build_smoke_chat(42, use_multiturn=True)
    chat.renderer = StubRenderer()
    rich = RichAnswerComposer(chat, max_chain_hops=5, max_elaborations=2, max_sentences=6)
    _topic, facts = _gather_multi_fact(rich)
    off_result = rich.render_paragraph(facts)

    os.environ[BATCH_FLAG] = "1"
    chat2 = _build_smoke_chat(42, use_multiturn=True)
    chat2.renderer = StubRenderer()
    rich2 = RichAnswerComposer(chat2, max_chain_hops=5, max_elaborations=2, max_sentences=6)
    on_result = rich2.render_paragraph(facts)

    assert off_result == on_result


def test_single_fact_turn_never_batches_even_with_flag_on():
    """Nothing to batch with exactly one fact -- the `len(facts) > 1` guard must keep this on the sequential
    path even with the flag on (asserted by call count, not inference)."""
    os.environ[BATCH_FLAG] = "1"
    _chat, rich, renderer = _build_rich()
    _topic, facts = _gather_multi_fact(rich)
    one = facts[:1]
    rich.render_paragraph(one)
    assert renderer.render_svo_batch_calls == []
    assert len(renderer.render_svo_calls) == 1


def test_batched_candidate_failing_verify_falls_back_to_single_item_path():
    """A batched render that returns a WRONG (unverifiable) surface for one fact must not corrupt the turn:
    that one fact falls back to the pre-existing single-item `_render_one_verified` (regen included) rather
    than being silently kept unverified or silently dropped without a retry -- the moat stays exactly as
    strict as the sequential path's."""
    os.environ[BATCH_FLAG] = "1"
    chat, rich, renderer = _build_rich()
    _topic, facts = _gather_multi_fact(rich)

    real_batch = renderer.render_svo_batch

    def _poisoned_batch(triples):
        out, secs = real_batch(triples)
        # corrupt the FIRST candidate's surface so it can never re-parse to its own SVO
        if out:
            out[0] = ("zzz completely unrelated nonsense zzz", ["zzz", "zzz", "zzz"])
        return out, secs

    renderer.render_svo_batch = _poisoned_batch
    paragraph, kept, dropped = rich.render_paragraph(facts)

    # the corrupted fact must have gone through the single-item fallback (an extra render_svo call logged),
    # and every KEPT fact must still be one the brain actually verified (kept is always a subset of gathered
    # facts, by construction -- the moat is never weaker for a batched-but-unverifiable candidate).
    assert len(renderer.render_svo_calls) >= 1
    assert all(f in facts for f in kept)
    assert all(f in facts for f in dropped)
    assert set(tuple(f) for f in kept) | set(tuple(f) for f in dropped) == set(tuple(f) for f in facts)
