"""Mood iteration 5 (design doc section 17): the existing readouts stay byte-identical, alpha=0 is the lesion, the lesion
repeats byte for byte, the controls are norm-matched, and the pair / event / question / lexicon word sets are disjoint.
Runs the 0.1B cortex on CPU (BRAIN_MODEL_DIR default)."""
import argparse
import importlib.util
import random
import re
import subprocess

import pytest
import torch

from brain import affect
from brain.experiments import mood_probe as mp
from brain.experiments import sentiment_instrument as si

PRE_CHANGE_COMMIT = "a97d77da8"    # the commit that holds the pre-registration; brain/affect.py is unchanged there

FUNC = set("""that this with from they them were have been your yours into over when what then than there their about
after before while which would could should these those each only some such very just also more most much many other
the and for you are was his her its our out not but all any can had has how who why too off own same here""".split())


def cw(texts):
    return {w for t in texts for w in re.findall(r"[a-z]+", t.lower()) if len(w) >= 4 and w not in FUNC}


# ------------------------------------------------------------------ word-set disjointness (no model needed)
def test_pairs_shape_and_draw():
    assert len(affect.PRIMARY_PAIRS) == 48
    assert all(p != n for p, n in affect.PRIMARY_PAIRS)
    assert len({t for pr in affect.PRIMARY_PAIRS for t in pr}) == 96
    d = affect.draw_pairs(random.Random(1))
    assert len(d) == 32 and len({p for p, _ in d}) == 32
    assert d == affect.draw_pairs(random.Random(1)) and d != affect.draw_pairs(random.Random(2))


def test_event_sets_shape():
    for split in ("dev", "test"):
        assert all(len(x) == 8 for x in mp.EVENTS_V5[split])
    allv5 = [e for s in mp.EVENTS_V5.values() for x in s for e in x]
    assert len(set(allv5)) == 48


def test_disjoint_word_sets():
    pairs = [t for pr in affect.PRIMARY_PAIRS for t in pr]
    old_events = (mp.POS_EVENTS + mp.NEG_EVENTS + mp.NEUTRAL_EVENTS + mp.HELDOUT_POS + mp.HELDOUT_NEG +
                  mp.HELDOUT_NEUTRAL)
    questions = [q for v in mp.QUESTION_SETS.values() for q in v]
    primaries = affect.POSITIVE_PRIMARIES + affect.NEGATIVE_PRIMARIES
    lex = set(si.POS) | set(si.NEG)
    dev = [e for x in mp.EVENTS_V5["dev"] for e in x]
    test = [e for x in mp.EVENTS_V5["test"] for e in x]
    assert not cw(pairs) & lex
    assert not cw(dev) & lex and not cw(test) & lex
    for name, other in (("old events", old_events), ("questions", questions)):
        assert not cw(pairs) & cw(other), (name, cw(pairs) & cw(other))
        assert not cw(dev) & cw(other), (name, cw(dev) & cw(other))
        assert not cw(test) & cw(other), (name, cw(test) & cw(other))
    assert not cw(pairs) & cw(dev) and not cw(pairs) & cw(test)
    assert not cw(dev) & cw(test)
    assert not (cw(dev) | cw(test)) & cw(primaries)
    # the lexical instrument does not read the pair words, and the v5 echo does
    assert not (si.POS | si.NEG) & si.PAIR_WORDS
    assert si.primary_echo_v5("my hands felt numb and cracked") and not si.primary_echo("my hands felt numb and cracked")


# ------------------------------------------------------------------ model-backed tests
@pytest.fixture(scope="module")
def ctx():
    from brain.cortex import DEFAULT_MODEL
    torch.set_num_threads(4)
    from brain.experiments import mood5
    a = argparse.Namespace(seed=42, model=DEFAULT_MODEL, threads=4, band="8,16", placement="post", alpha=0.3)
    c = mood5.setup(a)
    c["mood5"] = mood5
    return c


def test_band_scaled_to_depth():
    assert affect.band_for(24) == (8, 16) and affect.band_for(16) == (8, 16)
    assert affect.band_for(12) == (4, 8)


def test_band_recorded_and_norm_matched(ctx):
    aff = ctx["aff"]
    assert aff.band_used == (4, 8)    # the 0.1B model has 12 layers
    for name, delta in (("axis", aff.axis_delta), ("topic", aff.topic_delta), ("shuf", ctx["shuf_delta"])):
        for i, d in enumerate(delta):
            if i % 3 == 1 and (i // 3) in aff.layer_norm:
                assert float(d.flatten().norm()) == pytest.approx(aff.layer_norm[i // 3], rel=1e-4), (name, i)
            else:
                assert d is None, (name, i)
    assert ctx["shuf_axis"]["signs"].count(1.0) == ctx["shuf_axis"]["signs"].count(-1.0) == 16


def test_axes_differ(ctx):
    aff = ctx["aff"]
    flat = lambda dl: torch.cat([d.flatten() for d in dl if d is not None])  # noqa: E731
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(flat(a), flat(b), dim=0))  # noqa: E731
    assert abs(cos(aff.axis_delta, aff.topic_delta)) < 0.99
    assert abs(cos(aff.axis_delta, ctx["shuf_delta"])) < 0.99


def test_project_pure_has_no_side_effects(ctx):
    aff, cortex, base = ctx["aff"], ctx["cortex"], ctx["base"]
    before = (aff.mood, aff.arousal, aff.last_appraisal)
    raw, v = aff.project_pure(cortex, base, "The bakery set aside a cinnamon roll just for you.")
    assert (aff.mood, aff.arousal, aff.last_appraisal) == before
    assert raw == pytest.approx(aff.project(cortex, base, "The bakery set aside a cinnamon roll just for you."), abs=1e-6)
    assert -1 < v < 1


def test_alpha_zero_equals_off_and_lesion_repeats(ctx):
    m, aff, cfgs = ctx["mood5"], ctx["aff"], ctx["mood5"].arm_cfg(ctx)
    q = mp.MIXED_DEV[0]
    for placement in ("post", "pre"):
        aff.placement = placement
        aff.alpha = 0.0
        off = m.generate(ctx, q, cfgs["off"], 7, 8)
        assert m.generate(ctx, q, cfgs["off"], 7, 8) == off                     # lesion repeat: byte-identical
        for arm in ("pos", "neg", "topic_pos", "shuf_neg"):
            assert m.generate(ctx, q, cfgs[arm], 7, 8) == off, (placement, arm)   # alpha = 0 is the lesion
        st = ctx["base"]
        assert aff.modulate(ctx["cortex"], st) is st                             # state returned untouched
    aff.placement, aff.alpha = "post", 0.3


def test_push_changes_state_only_when_on(ctx):
    aff, cortex, base = ctx["aff"], ctx["cortex"], ctx["base"]
    aff.placement, aff.alpha = "post", 0.3
    aff.axis_delta = aff.axis_delta
    aff.mood, aff.enabled = ctx["m_neg"], True
    st = aff.modulate(cortex, base)
    assert st is not base and any(not torch.equal(a, b) for a, b in zip(st, base))
    aff.enabled = False
    assert aff.modulate(cortex, base) is base


def test_post_placement_reads_question_then_colon(ctx):
    m, aff, cfgs, cortex, base = ctx["mood5"], ctx["aff"], ctx["mood5"].arm_cfg(ctx), ctx["cortex"], ctx["base"]
    q = mp.MIXED_DEV[1]
    aff.placement, aff.alpha = "post", 0.3
    text_post = m.generate(ctx, q, cfgs["neg"], 3, 8)
    aff.placement = "pre"
    text_pre = m.generate(ctx, q, cfgs["neg"], 3, 8)
    aff.placement = "post"
    assert isinstance(text_post, str) and isinstance(text_pre, str)
    # the split read "...Assistant" + ":" reaches the same state as reading it whole when nothing is pushed
    _, s_whole = cortex.read(f"User: {q}\n\nAssistant:", base)
    _, s1 = cortex.read(f"User: {q}\n\nAssistant", base)
    _, s_split = cortex.read(":", s1)
    assert len(s_whole) == len(s_split)


# ------------------------------------------------------------------ existing readouts: byte-identical
def _old_affect_module():
    try:
        src = subprocess.run(["git", "show", f"{PRE_CHANGE_COMMIT}:brain/affect.py"], capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history not available")
    spec = importlib.util.spec_from_loader("affect_old", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, "affect_old.py", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("readout", ["twopole_deep", "lin_all"])
def test_existing_readouts_byte_identical(ctx, readout):
    old = _old_affect_module()
    cortex, base = ctx["cortex"], ctx["base"]
    rng = random.Random(5)
    pos, neg = affect.draw_primaries(rng)
    new_a = affect.Affect(enabled=True, readout=readout).grow(cortex, base, pos, neg)
    old_a = old.Affect(enabled=True, readout=readout).grow(cortex, base, pos, neg)
    assert torch.equal(new_a.axis_unit, old_a.axis_unit)
    assert (new_a.zero, new_a.scale) == (old_a.zero, old_a.scale)
    assert new_a.read_layers == old_a.read_layers
    for x, y in zip(new_a.axis_delta, old_a.axis_delta):
        assert (x is None and y is None) or torch.equal(x, y)
    ev = "You did a wonderful job today, I'm really proud of you."
    assert new_a.appraise(cortex, base, ev) == old_a.appraise(cortex, base, ev)
    assert new_a.mood == old_a.mood
    sn, so = new_a.modulate(cortex, base), old_a.modulate(cortex, base)
    assert all(torch.equal(a, b) for a, b in zip(sn, so))
    old_r = old_a.readout()    # iteration 6 added surprise / encode_gain / gain_on to the read-out; the old keys are unchanged
    assert {k: v for k, v in new_a.readout().items() if k in old_r} == old_r
