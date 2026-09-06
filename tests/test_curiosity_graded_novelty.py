"""Unit tests for `TopicNoveltyGate` + the graded-novelty flags (scaffold-retirement backlog rank-10,
2026-09-05) in `research/runners/curiosity_production_organ.py`.

RANK-10's gap: the production curiosity call (`webapp/server.py::_curiosity_followup`) always fed the ASK-pool
judge a HOST CONSTANT (`NOVEL_SIGNAL = 0.95`) on every abstain, whatever the topic. `TopicNoveltyGate` retires
that constant with a genuine Bogacz-Brown familiarity/mismatch read (reuse-by-import of `AntiHebbianFamiliarity`
+ the genuine spike-phasor bind `phase_sum_neuron`) of the SPECIFIC topic word.

These tests pin, at unit level (fast, deterministic, no on-bridge substrate build):
  (1) GRADED, not binary: an imprinted (known) word reads near-zero novelty; an unrelated word reads near the
      ceiling; a noisy/partial draw of a known word reads strictly in between, monotonically with noise level.
  (2) LESION load-bearing: a never-imprinted gate reads the SAME ceiling novelty for every word (the gradation
      collapses) -- and an explicit post-imprint `.lesion()` reverts an already-familiar word to that same
      ceiling.
  (3) The IMPRINT<->QUERY correspondence, not word-string shape, drives a low reading (imprinting a DISJOINT
      vocabulary leaves the original "known" words reading at the ceiling too).
  (4) Determinism: the SAME (seed, word) always renders the SAME phase code / novelty (repeated reads agree).
  (5) FLIPPED DEFAULT-ON (2026-09-05, rank-10 production-flip GO): `graded_novelty_enabled()` now defaults True
      (env unset); `graded_novelty_lesioned()` stays default False (unaffected by the flip). The explicit
      `BRAIN_CURIOSITY_GRADED_NOVELTY=0` (or false/off/no) is the BYTE-IDENTICAL ESCAPE, and `topic_novelty()`
      with `topic=None` (or falsy) always falls back to the EXACT `NOVEL_SIGNAL` constant regardless of the flag
      -- the escape hatch a caller relies on to stay byte-identical.

The full 6-seed scientific validation (graded novelty IS load-bearing on the real on-bridge ASK-pool `want_hz` /
`curious` decision, discriminating the old always-curious constant) lives in the dedicated research runner
`research/runners/_curiosity_graded_novelty_derisk.py` (GO 6/6, reproduced across 3 independent runs) -- building
a `SimulationBridge` per case here would make this file slow; these tests cover the deterministic, CPU-cheap
half of the mechanism (the novelty READ itself), not the on-bridge spiking consumption.
"""
from __future__ import annotations

import os

import pytest

from research.runners.curiosity_production_organ import (
    TopicNoveltyGate,
    NOVEL_SIGNAL,
    graded_novelty_enabled,
    graded_novelty_lesioned,
    topic_novelty,
    get_topic_gate,
)

_TINY_D = 32   # a small render dimension keeps these tests fast; the gate's math is dimension-independent


def _gate(seed=42):
    return TopicNoveltyGate(seed=seed, D=_TINY_D)


# ── (1) graded, not binary ──────────────────────────────────────────────────────────────────────────────────
def test_imprinted_word_reads_near_zero_novelty():
    g = _gate()
    g.imprint("apple")
    assert g.novelty("apple") == pytest.approx(0.0, abs=1e-6)


def test_unrelated_word_reads_near_ceiling_novelty():
    g = _gate()
    g.imprint("apple")
    assert g.novelty("wombat") > 0.9


def test_noisy_partial_cue_of_a_known_word_reads_strictly_between():
    g = _gate()
    g.imprint("apple")
    known = g.novelty("apple")
    lo = g.novelty("apple", noise=0.03)
    hi = g.novelty("apple", noise=0.25)
    novel = g.novelty("wombat")
    assert known < lo < hi < novel, (known, lo, hi, novel)


def test_gradation_is_monotonic_in_noise_level():
    g = _gate()
    g.imprint("apple")
    reads = [g.novelty("apple", noise=n) for n in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)]
    assert reads == sorted(reads), reads


# ── (2) lesion load-bearing ──────────────────────────────────────────────────────────────────────────────────
def test_never_imprinted_gate_gives_uniform_ceiling_novelty():
    g = _gate()   # nothing imprinted -- the production `lesion=True` semantics
    known_words_never_imprinted = [g.novelty(w) for w in ("apple", "banana", "cherry")]
    unrelated = [g.novelty(w) for w in ("wombat", "xylophone", "quasar")]
    allvals = known_words_never_imprinted + unrelated
    assert max(allvals) - min(allvals) < 1e-6, allvals   # uniform -- no discrimination without imprinting
    assert min(allvals) > 0.99   # the ceiling energy (unit-normalized render, W==0)


def test_explicit_lesion_after_imprint_reverts_to_ceiling():
    g = _gate()
    g.imprint("apple")
    before = g.novelty("apple")
    assert before == pytest.approx(0.0, abs=1e-6)
    g.lesion()
    after = g.novelty("apple")
    assert after > 0.99, "lesion() must clear the learned weights -> the familiar word reverts to the ceiling"


def test_lesion_then_reimprint_rebuilds_the_projector():
    """`lesion()` must fully reset (fresh `AntiHebbianFamiliarity`), not merely zero W in place, so a later
    re-imprint of the SAME word actually rebuilds it (the `SpikingConjunctiveFamiliarityGate.lesion()` discipline
    this mirrors) -- otherwise a word already 'in span' before the lesion would silently stay unlearnable after."""
    g = _gate()
    g.imprint("apple")
    g.lesion()
    g.imprint("apple")
    assert g.novelty("apple") == pytest.approx(0.0, abs=1e-6)


# ── (3) the imprint<->query correspondence, not word-string shape, drives a low reading ────────────────────────
def test_disjoint_imprint_vocabulary_leaves_original_words_at_ceiling():
    g_real = _gate()
    g_real.imprint_vocab(["apple", "banana"])
    real_reading = g_real.novelty("apple")

    g_permuted = _gate()
    g_permuted.imprint_vocab(["decoy1", "decoy2"])   # a DISJOINT vocabulary -- "apple" never imprinted here
    permuted_reading = g_permuted.novelty("apple")

    assert real_reading < 1e-6
    assert permuted_reading > 0.9
    assert permuted_reading > real_reading + 0.5


def test_imprint_vocab_returns_count_of_newly_imprinted_words():
    g = _gate()
    n1 = g.imprint_vocab(["apple", "banana", "apple"])   # a duplicate within the same call
    assert n1 == 2
    n2 = g.imprint_vocab(["apple", "cherry"])            # "apple" already imprinted -- idempotent
    assert n2 == 1


# ── (4) determinism ─────────────────────────────────────────────────────────────────────────────────────────
def test_same_seed_and_word_render_identically_across_instances():
    g1 = _gate(seed=42)
    g2 = _gate(seed=42)
    g1.imprint("apple")
    g2.imprint("apple")
    assert g1.novelty("banana") == pytest.approx(g2.novelty("banana"), abs=1e-9)


def test_different_seeds_still_agree_on_the_gradation_shape():
    for seed in (42, 43, 44, 100, 101, 102):
        g = _gate(seed=seed)
        g.imprint("apple")
        assert g.novelty("apple") < g.novelty("apple", noise=0.2) < g.novelty("wombat")


# ── (5) FLIPPED DEFAULT-ON 2026-09-05 (rank-10 production-flip GO) -- unset now means ON; explicit "0"/false/
#     off/no is the BYTE-IDENTICAL escape back to the pre-flip constant. ─────────────────────────────────────
def test_graded_novelty_enabled_defaults_on(monkeypatch):
    monkeypatch.delenv("BRAIN_CURIOSITY_GRADED_NOVELTY", raising=False)
    assert graded_novelty_enabled() is True


def test_graded_novelty_enabled_explicit_off_is_the_byte_identical_escape(monkeypatch):
    monkeypatch.setenv("BRAIN_CURIOSITY_GRADED_NOVELTY", "0")
    assert graded_novelty_enabled() is False


def test_graded_novelty_lesioned_defaults_off(monkeypatch):
    monkeypatch.delenv("BRAIN_CURIOSITY_GRADED_NOVELTY_LESION", raising=False)
    assert graded_novelty_lesioned() is False


def test_graded_novelty_enabled_truthy_values(monkeypatch):
    for v in ("1", "true", "on", "yes", "TRUE", "On"):
        monkeypatch.setenv("BRAIN_CURIOSITY_GRADED_NOVELTY", v)
        assert graded_novelty_enabled() is True, v
    for v in ("0", "false", "off", "no", ""):
        monkeypatch.setenv("BRAIN_CURIOSITY_GRADED_NOVELTY", v)
        assert graded_novelty_enabled() is False, v


def test_topic_novelty_falls_back_to_constant_on_falsy_topic():
    assert topic_novelty(None, known_vocab=["apple"]) == NOVEL_SIGNAL
    assert topic_novelty("", known_vocab=["apple"]) == NOVEL_SIGNAL


def test_topic_novelty_never_raises_on_bad_vocab():
    # `known_vocab=None` is falsy-safe (imprint_vocab's own `words or ()` guard) -- no exception, a normal read.
    seed = 4246
    val_none = topic_novelty("apple", known_vocab=None, seed=seed)
    assert isinstance(val_none, float)
    # a non-iterable, TRUTHY known_vocab (e.g. `object()`) must degrade to the constant, never crash a turn or
    # corrupt the moat -- `topic_novelty`'s own try/except is the safety net.
    val_bad = topic_novelty("apple", known_vocab=object(), seed=seed + 1)
    assert val_bad == NOVEL_SIGNAL


def test_topic_novelty_lesioned_never_imprints_and_stays_at_ceiling():
    seed = 4242   # an unused seed key so this test does not collide with the module-level cache
    known = ["apple", "banana", "cherry"]
    lo = topic_novelty("apple", known_vocab=known, seed=seed, lesion=True)
    hi = topic_novelty("wombat", known_vocab=known, seed=seed, lesion=True)
    assert lo > 0.9 and hi > 0.9, (lo, hi)   # the lesioned twin is NEVER imprinted -- everything reads the ceiling


def test_topic_novelty_real_arm_discriminates_known_from_unrelated():
    seed = 4243   # a fresh, unused seed key
    known = ["apple", "banana", "cherry"]
    lo = topic_novelty("apple", known_vocab=known, seed=seed, lesion=False)
    hi = topic_novelty("wombat", known_vocab=known, seed=seed, lesion=False)
    assert lo < 1e-6
    assert hi > 0.9


def test_get_topic_gate_is_process_shared_per_seed_and_lesion_arm():
    seed = 4244
    g1 = get_topic_gate(seed=seed, lesion=False)
    g2 = get_topic_gate(seed=seed, lesion=False)
    assert g1 is g2, "the non-lesioned singleton must be shared across calls at the SAME seed"
    g_les = get_topic_gate(seed=seed, lesion=True)
    assert g_les is not g1, "the lesion arm must be a SEPARATE instance, never the real gate"
