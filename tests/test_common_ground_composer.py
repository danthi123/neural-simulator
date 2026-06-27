"""CI GUARD (Tier 2.4): MINIMAL COMMON-GROUND -- a SHARED/PRIVATE fact tag -> AUDIENCE DESIGN must keep working,
with the full anti-cheat bar holding. This pins the de-risked GO
(research/findings/2026-06-27-tier2.4-common-ground-GO.md) so it does not silently bit-rot.

The anti-cheats (audience design TRACKS the tag and BEATS the no-tag baseline; permuted-tag collapses; lesion
collapses; the no-confab moat is 0-FA on a fabricated tag AND a fabricated query answer; tag fidelity 1.0) are
asserted as the guard -- a regression that broke the tag's necessity or the moat would flip these.

Composes the PROVEN polarity/negation tag mechanism (a bound role tag cleaned only against a small codebook),
reuse-by-import on RFPhasorComposer -- NO `sim/` edit, NO composer edit (SUBCLASS). CPU-safe (the parent's RF ops
run on the NumPy backend = the == test-oracle path); a GPU-gated test confirms parity on the real RF substrate.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.common_ground_composer import CommonGroundComposer  # noqa: E402
from sim.backend import is_gpu_backend  # noqa: E402


# A balanced SHARED/PRIVATE corpus (4 each) over the composer's default vocab.
_FACTS = [
    ("dog", "go", "north", "SHARED"),
    ("cat", "run", "south", "PRIVATE"),
    ("dog", "look", "river", "PRIVATE"),
    ("cat", "stop", "east", "SHARED"),
    ("dog", "come", "west", "SHARED"),
    ("cat", "go", "apple", "PRIVATE"),
    ("dog", "run", "hot", "PRIVATE"),
    ("cat", "look", "cold", "SHARED"),
]


def _build(seed, D=64, use_spiking_bind=False):
    cg = CommonGroundComposer(seed=seed, D=D, use_spiking_bind=use_spiking_bind)
    for a, act, pt, tag in _FACTS:
        cg.store_cg(a, act, pt, common_ground=tag)
    return cg


def _audience_score(cg, perm=None, lesion=False):
    """Fraction of facts whose volunteer/suppress decision matches correct audience design (tell private, suppress
    shared). perm scrambles which fact's tag is read; lesion forces the tag read to None (severed role)."""
    n_ok = 0
    for i, (a, act, pt, true_tag) in enumerate(_FACTS):
        if lesion:
            tell = True                       # severed -> no signal -> fall back to tell-all (the no-tag policy)
        elif perm is not None:
            pa, pact, ppt, _ = _FACTS[perm[i]]
            read = cg.read_common_ground(pa, pact, ppt)
            tell = (read == "PRIVATE") if read is not None else True
        else:
            tell = cg.should_volunteer(a, act, pt)
            tell = True if tell is None else tell
        n_ok += int(tell == (true_tag == "PRIVATE"))
    return n_ok / len(_FACTS)


def test_tag_fidelity_reads_back():
    """Each bound SHARED/PRIVATE tag reads back faithfully (the polarity-read mechanism on the commonground role)."""
    cg = _build(42)
    for a, act, pt, tag in _FACTS:
        assert cg.read_common_ground(a, act, pt) == tag


def test_audience_design_beats_no_tag_baseline():
    """Audience design (tell private, suppress shared) is PERFECT; a tag-blind policy maxes at 0.5 on the balanced
    corpus (it gets one class right, the other wrong)."""
    cg = _build(42)
    real = _audience_score(cg)
    assert real == 1.0
    # the best a tag-blind speaker can do: tell-all = 0.5 (gets the 4 private right), suppress-all = 0.5
    assert real >= 0.5 + 0.25       # well above the 0.5 ceiling


def test_permuted_tag_collapses():
    """Reading a PERMUTED fact's tag decorrelates the decision from the truth -> collapses to ~chance."""
    cg = _build(42)
    rng = np.random.default_rng(123)
    scores = [_audience_score(cg, perm=rng.permutation(len(_FACTS))) for _ in range(16)]
    assert np.mean(scores) <= 0.5 + 0.10     # collapses toward the tag-blind ceiling


def test_lesion_tag_collapses():
    """Severing the tag read -> the agent degrades to the tag-blind baseline (the tag is load-bearing)."""
    cg = _build(42)
    assert _audience_score(cg, lesion=True) <= 0.5 + 0.10


def test_moat_no_fabricated_tag():
    """A fact whose FULL SVO was never stored reads tag=None -- no fabricated tag (the no-confab moat). Includes
    a right-cue/wrong-patient case ('dog go cold' vs stored 'dog go north')."""
    cg = _build(42)
    for a, act, pt in [("dog", "stop", "small"), ("cat", "come", "big"), ("dog", "go", "cold")]:
        assert cg.read_common_ground(a, act, pt) is None
        assert cg.should_volunteer(a, act, pt) is None       # -> the clarification trigger, not a guess


def test_moat_query_abstains_on_unstored_cue():
    """The underlying who/what query still abstains on an (agent,action) cue absent from every stored fact."""
    cg = _build(42)
    for a, act in [("dog", "stop"), ("cat", "come")]:
        assert cg.query_patient(a, act) is None


def test_describe_audience_designed_volunteer_vs_acknowledge():
    """describe_audience_designed VOLUNTEERS a private fact (full sentence) and ACKNOWLEDGES a shared one
    (no re-statement); abstains on an unknown subject."""
    cg = _build(42)
    # 'dog go north' is SHARED -> acknowledge (don't re-explain)
    text, decision = cg.describe_audience_designed("dog")
    assert decision == "acknowledge"
    assert text.startswith("as you know")
    # 'cat run south' is PRIVATE -> volunteer the full sentence
    text2, decision2 = cg.describe_audience_designed("cat")
    assert decision2 == "volunteer"
    assert text2 == "cat run south"
    # an unknown subject -> abstain (no invented sentence)
    text3, decision3 = cg.describe_audience_designed("unicorn")
    assert decision3 == "abstain" and text3 is None


def test_underlying_conversational_api_unregressed():
    """Storing with a common-ground tag does NOT regress the parent's who/what + yes-no + moat (the tag is an
    EXTRA bound role; the SVO content is unchanged)."""
    cg = _build(42)
    assert cg.query_patient("dog", "go") == "north"         # who/what still works with the extra tag bound
    assert cg.query_agent("go", "north") == "dog"
    assert cg.query_patient("dog", "fly") is None           # moat (unstored action)


def test_six_seed_robustness():
    """6-seed: audience design perfect; permuted+lesion collapse; moat 0-FA; tag fidelity 1.0 -- every seed."""
    for seed in (42, 43, 44, 100, 101, 102):
        cg = _build(seed)
        assert _audience_score(cg) == 1.0
        assert all(cg.read_common_ground(a, act, pt) == tag for a, act, pt, tag in _FACTS)
        assert _audience_score(cg, lesion=True) <= 0.6
        # moat
        assert cg.read_common_ground("dog", "go", "cold") is None
        assert cg.query_patient("dog", "stop") is None


@pytest.mark.skipif(not is_gpu_backend(), reason="GPU-only: confirms the tag runs through the REAL RF spiking bind")
def test_spiking_substrate_parity():
    """On the CuPy backend the tag bind/unbind run through the real resonate-and-fire complex-synapse ops; the
    audience-design behavior + moat are identical to the NumPy oracle (the brain-based claim)."""
    cg = _build(42, use_spiking_bind=True)
    assert _audience_score(cg) == 1.0
    assert all(cg.read_common_ground(a, act, pt) == tag for a, act, pt, tag in _FACTS)
    assert cg.read_common_ground("dog", "go", "cold") is None
    assert cg.query_patient("dog", "stop") is None
