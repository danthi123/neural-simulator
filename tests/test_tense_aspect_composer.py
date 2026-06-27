"""CI GUARD (Tier 2.5): TENSE/ASPECT as a bound fact-tag -- a PAST/PRESENT/FUTURE role tag bound onto each fact
must keep reading back faithfully AND driving the surface verb form (went / goes / will go), with the full
anti-cheat bar holding. This pins the de-risked GO (research/findings/2026-06-27-tier2.5-tense-aspect-GO.md) so it
does not silently bit-rot.

The anti-cheats (tag read-fidelity >> chance 1/3; the tag DRIVES the correct surface form; permuted-tag collapses
the rendered tense; lesion collapses to present-only; the no-confab moat is 0-FA on a fabricated tense AND a
fabricated tensed sentence) are asserted as the guard -- a regression that broke the tag's necessity, the render
integration, or the moat would flip these.

Composes the PROVEN polarity/negation + common-ground tag mechanism (a bound role tag cleaned only against a small
codebook), reuse-by-import on ArgStructureComposer (the frame renderer with the TENSE unit) -- NO `sim/` edit, NO
existing-composer edit (SUBCLASS). CPU-safe (the parent's RF ops run on the NumPy backend = the == test-oracle
path); a GPU-gated test confirms parity on the real RF substrate.

D=128 is the validated GO point: at D=64 a 4-bound-role composite occasionally mis-cleans an UNRELATED cue role,
making read_tense's whole-fact cue match abstain (never a WRONG tense; the tag reads faithfully). Raising D is the
standard VSA bundle-SNR lever (production composers run D=2048). See the finding.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.tense_aspect_composer import TenseAspectComposer, inflect, TENSE_ROLE  # noqa: E402
from sim.backend import is_gpu_backend  # noqa: E402

_D = 128

# A balanced PAST/PRESENT/FUTURE corpus (3 each), GOAL-frame verbs so the frame renders 'the <a> <V> to the <g>'.
_FACTS = [
    ({"agent": "boy", "action": "go", "GOAL": "park"}, "PAST"),
    ({"agent": "cat", "action": "run", "GOAL": "home"}, "PRESENT"),
    ({"agent": "dog", "action": "come", "GOAL": "home"}, "FUTURE"),
    ({"agent": "girl", "action": "walk", "GOAL": "school"}, "PAST"),
    ({"agent": "dog", "action": "run", "GOAL": "park"}, "PRESENT"),
    ({"agent": "boy", "action": "come", "GOAL": "school"}, "FUTURE"),
    ({"agent": "cat", "action": "go", "GOAL": "river"}, "PAST"),
    ({"agent": "girl", "action": "run", "GOAL": "river"}, "PRESENT"),
    ({"agent": "dog", "action": "walk", "GOAL": "home"}, "FUTURE"),
]


def _vocab(seed):
    base = set(TenseAspectComposer(seed=seed, D=_D).words)
    for fct, _ in _FACTS:
        base |= {fct["agent"], fct["action"], fct["GOAL"]}
    return sorted(base)


def _build(seed):
    c = TenseAspectComposer(seed=seed, D=_D, vocab=_vocab(seed))
    for fct, tn in _FACTS:
        c.store_tensed(fct, tense=tn)
    return c


def _surf_tense(rendered, verb):
    """Classify the surface tense of a render by the inflected verb form present (PAST/PRESENT/FUTURE or None)."""
    if rendered is None:
        return None
    toks = rendered.split()
    if "will" in toks:
        return "FUTURE"
    if inflect(verb, "PAST") in toks:
        return "PAST"
    if inflect(verb, "PRESENT") in toks:
        return "PRESENT"
    return None


def test_inflection_table_drives_the_three_forms():
    """The inflection helper produces the three surface forms the task names: went / goes / will go."""
    assert inflect("go", "PAST") == "went"
    assert inflect("go", "PRESENT") == "goes"
    assert inflect("go", "FUTURE") == "will go"
    assert inflect("come", "PAST") == "came"          # irregular past
    assert inflect("walk", "PAST") == "walked"        # regular -ed
    assert inflect("jump", "PAST") == "jumped"        # unknown verb -> regular rule (tag still drives the form)


def test_tag_read_fidelity():
    """Each bound PAST/PRESENT/FUTURE tag reads back faithfully (the polarity-read mechanism on the TENSE role)."""
    c = _build(42)
    for fct, tn in _FACTS:
        assert c.read_tense(fct) == tn


def test_render_driven_by_bound_tag():
    """The bound tense tag DRIVES the surface verb form: an actual tensed render per class."""
    c = _build(42)
    assert c.render_tensed({"agent": "boy", "action": "go", "GOAL": "park"}) == "the boy went to the park"
    assert c.render_tensed({"agent": "dog", "action": "come", "GOAL": "home"}) == "the dog will come to the home"
    # every fact's rendered surface tense matches its stored tag
    for fct, tn in _FACTS:
        assert _surf_tense(c.render_tensed(fct), fct["action"]) == tn


def test_permuted_tag_collapses_rendered_tense():
    """Storing the facts with a SHUFFLED tag assignment makes the rendered tense track the permuted tag, not the
    true tense -> agreement with the true tense collapses to ~chance (1/3)."""
    seed = 42
    rng = np.random.default_rng(seed + 991)
    scores = []
    for _ in range(8):
        p = rng.permutation(len(_FACTS))
        cp = TenseAspectComposer(seed=seed, D=_D, vocab=_vocab(seed))
        for i, (fct, _tn) in enumerate(_FACTS):
            cp.store_tensed(fct, tense=_FACTS[p[i]][1])
        ok = sum(_surf_tense(cp.render_tensed(fct), fct["action"]) == tn for fct, tn in _FACTS)
        scores.append(ok / len(_FACTS))
    assert np.mean(scores) <= 0.45        # collapses toward chance (1/3)


def test_lesion_tense_collapses_to_present():
    """Severing the tense read -> render defaults to PRESENT regardless of the stored tag -> agreement with the
    true tense collapses to the present-only fraction (1/3 by construction)."""
    c = _build(42)
    ok = sum(_surf_tense(c.render_tensed(fct, lesion_tense=True), fct["action"]) == tn for fct, tn in _FACTS)
    assert ok / len(_FACTS) <= 0.45
    # and every lesioned render is in fact present-tense (the form collapsed)
    for fct, _tn in _FACTS:
        assert _surf_tense(c.render_tensed(fct, lesion_tense=True), fct["action"]) == "PRESENT"


def test_moat_no_fabricated_tense():
    """A fact whose cue roles were never stored reads tense=None -- no fabricated tense (the no-confab moat).
    Includes a right-(agent,action)/wrong-GOAL case ('boy go school' vs stored 'boy go park')."""
    c = _build(42)
    for fct in [{"agent": "boy", "action": "stop", "GOAL": "river"},
                {"agent": "dog", "action": "look", "GOAL": "park"},
                {"agent": "boy", "action": "go", "GOAL": "school"}]:
        assert c.read_tense(fct) is None


def test_moat_no_invented_tensed_sentence():
    """A render over an unknown subject -> None (no invented tensed sentence)."""
    c = _build(42)
    assert c.render_tensed({"agent": "horse", "action": "go", "GOAL": "park"}) is None


def test_underlying_argstructure_api_unregressed():
    """Storing with a tense tag does NOT regress the parent's typed-role recall + the moat (the tense is an EXTRA
    bound role; the argument-structure content is unchanged)."""
    c = _build(42)
    assert c.query_role("GOAL", agent="boy", action="go") == "park"     # who/what still works with the tag bound
    assert c.query_role("agent", action="come", GOAL="home") == "dog"
    assert c.query_role("GOAL", agent="boy", action="fly") is None      # moat (unstored action)


def test_default_tense_is_present():
    """store_tensed with no tense defaults to PRESENT (byte-compatible with a tense-free fact)."""
    c = TenseAspectComposer(seed=42, D=_D, vocab=_vocab(42))
    c.store_tensed({"agent": "boy", "action": "go", "GOAL": "park"})
    assert c.read_tense({"agent": "boy", "action": "go", "GOAL": "park"}) == "PRESENT"
    assert c.render_tensed({"agent": "boy", "action": "go", "GOAL": "park"}) == "the boy goes to the park"


def test_parent_codes_byte_identical():
    """Adding the TENSE role + tag codebook from a DISJOINT rng stream leaves the parent's concept/role codes
    byte-identical (so the tense extension does not perturb the validated argstructure/RF composer)."""
    from research.runners.argstructure_composer import ArgStructureComposer
    vocab = _vocab(42)
    base = ArgStructureComposer(seed=42, D=_D, vocab=vocab)
    ext = TenseAspectComposer(seed=42, D=_D, vocab=vocab)
    for w in base.words:
        assert np.array_equal(base.concepts[w], ext.concepts[w])
    for r in base.roles:
        assert np.array_equal(base.roles[r], ext.roles[r])
    assert TENSE_ROLE not in base.roles            # the tense role is new, not colliding with a parent role


def test_six_seed_robustness():
    """6-seed: tag fidelity 1.0 + render fidelity 1.0; permuted+lesion collapse; moat 0-FA -- every seed."""
    for seed in (42, 43, 44, 100, 101, 102):
        c = _build(seed)
        assert all(c.read_tense(fct) == tn for fct, tn in _FACTS)
        assert all(_surf_tense(c.render_tensed(fct), fct["action"]) == tn for fct, tn in _FACTS)
        # lesion collapses
        ok = sum(_surf_tense(c.render_tensed(fct, lesion_tense=True), fct["action"]) == tn for fct, tn in _FACTS)
        assert ok / len(_FACTS) <= 0.45
        # moat
        assert c.read_tense({"agent": "boy", "action": "go", "GOAL": "school"}) is None
        assert c.render_tensed({"agent": "horse", "action": "go", "GOAL": "park"}) is None


@pytest.mark.skipif(not is_gpu_backend(), reason="GPU-only: confirms the tag runs through the REAL RF spiking bind")
def test_spiking_substrate_parity():
    """On the CuPy backend the tense tag bind/unbind run through the real resonate-and-fire complex-synapse ops;
    the read-fidelity, the tensed render, and the moat are identical to the NumPy oracle (the brain-based claim)."""
    c = _build(42)
    for fct, tn in _FACTS:
        assert c.read_tense(fct) == tn
        assert _surf_tense(c.render_tensed(fct), fct["action"]) == tn
    assert c.read_tense({"agent": "boy", "action": "go", "GOAL": "school"}) is None
    assert c.render_tensed({"agent": "horse", "action": "go", "GOAL": "park"}) is None
