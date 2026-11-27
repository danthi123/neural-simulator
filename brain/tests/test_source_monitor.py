# Capability: Self and identity
# Record: brain/research/2026-11-27-source-monitor-port-prereg-DRAFT.md
# Expected-fail: none (plumbing on a tiny random cortex; no model weights)
"""Source monitor: the efference mask tags what the brain produced (registered own texts, fed-back tokens, Assistant turns)
and never a heard telling; no tag or a zero tag leaves the plastic cortex unchanged; sleep learns the vector only when an
own-tagged text is in the night; the voice_tag channel swap restores itself."""
import json

import pytest
import torch

from brain import source_monitor as SM
from brain import source_tag as ST
from brain.experiments import source_monitor_screen as SMS
from brain.experiments import voice_tag as VT
from brain.sleep import sleep
from brain.tests.test_brain_key_switch import _Pipe
from brain.tests.test_source_tag import _plastic

PIPE = _Pipe()
TOLD = "User: my cat is Pixel\n\n"
OWN = "User: sampled by the cortex\n\n"
CHAT = "Hi there\n\nUser: my cat\n\nAssistant: ok\n\n"


def _tagged(text, prov):
    m = prov.mask(PIPE, PIPE.encode(text))
    return "".join(c for c, t in zip(text, m) if t)


def test_mask_rules():
    prov = SM.Provenance("own")
    assert prov.register(PIPE, [OWN, OWN]) == 1
    assert _tagged(TOLD, prov) == ""                       # a heard telling carries nothing
    assert _tagged(OWN, prov) == OWN                       # a registered own text is tagged whole, same surface form
    assert _tagged(CHAT, prov) == "Assistant: ok\n\n"      # an Assistant turn is own, the User turn is heard
    assert _tagged("User: q\n\nAssistant:", prov) == "Assistant:"   # the turn-open label at the moment of speaking
    assert prov.mask(PIPE, [65]) == [True]                 # a token fed back after sampling
    assert prov.stats["registered_hits"] == 1 and prov.stats["calls"] == 5
    allp = SM.Provenance("all")
    assert all(allp.mask(PIPE, PIPE.encode(TOLD)))
    with pytest.raises(ValueError):
        SM.Provenance("heard")


def test_off_zero_and_lesion_leave_output_unchanged_and_heard_text_is_untouched():
    p = _plastic()
    assert p.tag_mask_fn is None                           # default: the user-turn mask path, nothing attached
    told, own = PIPE.encode(TOLD), PIPE.encode(OWN)
    prov = SM.Provenance("own")
    prov.register(PIPE, [OWN])
    with torch.no_grad():
        ref_t, ref_o = p(told)[0], p(own)[0]
        SM.attach(p, torch.zeros(8), 0, False, prov)
        assert torch.equal(p(told)[0], ref_t) and torch.equal(p(own)[0], ref_o)       # zero tag: no change
        SM.attach(p, torch.ones(8), 0, False, prov)
        assert torch.equal(p(told)[0], ref_t)              # a heard telling is bit-identical with the tag on
        assert not torch.allclose(p(own)[0], ref_o)        # the own text is changed
        SM.detach(p)
        assert p.tag_mask_fn is None and torch.equal(p(own)[0], ref_o)                 # lesion = off


def test_sleep_learns_the_vector_only_with_an_own_tagged_text():
    eps, gen = [(TOLD, 1.0)], [OWN]
    base = None
    ref = sleep(_plastic(), base, eps, gen, steps=4, seed=3)
    zero = _plastic()
    SM.attach(zero, torch.zeros(8), 1, False, SM.Provenance("own"))
    assert sleep(zero, base, eps, gen, steps=4, seed=3) == ref          # a zero fixed tag: the night is unchanged
    heard_only = _plastic()                                            # keep text NOT registered as own: nothing tagged
    t0 = SM.attach(heard_only, torch.zeros(8), 1, True, SM.Provenance("own"))
    sleep(heard_only, base, eps, gen, steps=4, seed=3)
    assert float(t0.detach().abs().sum()) == 0.0
    contrast = _plastic()
    prov = SM.Provenance("own")
    prov.register(PIPE, [OWN])
    t1 = SM.attach(contrast, torch.zeros(8), 1, True, prov)
    sleep(contrast, base, eps, gen, steps=4, seed=3)
    assert float(t1.detach().abs().sum()) > 0                          # the own-tagged keep text gives the vector a gradient
    assert 0 < prov.stats["tagged"] < prov.stats["tokens"]             # contrast: some tokens tagged, some not
    fixed = _plastic()
    v = torch.full((8,), 0.5)
    SM.attach(fixed, v, 1, False, prov)
    sleep(fixed, base, eps, gen, steps=4, seed=3)
    assert torch.equal(fixed.source_tag, v)


def test_channel_swaps_voice_tag_and_restores_it():
    orig = VT.ST
    p = _plastic()
    prov = SM.Provenance("own")
    with SM.channel(VT, prov):
        assert VT.ST is not orig
        VT.ST.attach(p, torch.zeros(8), 1, learn=True)
        assert p.tag_mask_fn == prov.mask and isinstance(p.source_tag, torch.nn.Parameter)
        VT.ST.detach(p)
        assert p.source_tag is None and p.tag_mask_fn is None
    assert VT.ST is orig
    with pytest.raises(RuntimeError):
        with SM.channel(VT, prov):
            raise RuntimeError("x")
    assert VT.ST is orig


def test_bank_texts_needs_the_keep_bank(tmp_path):
    with pytest.raises(FileNotFoundError):
        SMS.bank_texts(str(tmp_path), 3)
    d = tmp_path / "n2"
    d.mkdir()
    (d / "keep_bank.json").write_text(json.dumps({"frame": "User:", "texts": ["User: a\n\n"]}))
    assert SMS.bank_texts(str(tmp_path), 3) == (["User: a\n\n"], "User:")


def _row(item, brain, onset):
    return {"kind": "H", "item": item, "you": f"What is {item}?", "brain": brain, "onset": onset}


def test_table_adds_all_arm_and_flags_a_degenerate_mask():
    led = {"pixel": {"telling": "My cat is called Pixel.", "keys": ["Pixel"], "told": True}}

    def arm(brain, onset):
        return {"hoff": {"rows": [_row("pixel", brain, onset)]}, "intact": {"rows": [_row("pixel", brain, None)]}}
    outs = [{"seed": s, "reproduction": {"rel_diff": 0.1, "cosine": 0.99}, "engagement": {"TAG": [0.5], "SHAM": [0.01]},
             "engagement_ALL": [0.2], "masked": {"TAG": {"tagged": 30, "tokens": 100}},
             "batteries": {"A0snap": {"hoff": {"rows": [_row("pixel", "My cat is called Pixel.", -1.0)]}},
                           "A0": arm("My cat is called Pixel.", -1.0), "TAG": arm("Your cat is called Pixel.", 1.0),
                           "LESION": arm("My cat is called Pixel.", -0.9), "SHAM": arm("My cat is Pixel.", -1.0),
                           "ALL": arm("My cat is Pixel.", -0.5)}} for s in (1, 2)]
    t = SMS.table(outs, {1: led, 2: led})
    assert t["verdict"] in ("GO", "NEGATIVE", "MIXED", "NO-ENGAGEMENT", "UNDEFINED")
    assert t["onset_TAG-ALL"]["n"] == 2 and t["masked_share_TAG"] == {1: 0.3, 2: 0.3}
    outs[0]["masked"]["TAG_degenerate"] = True
    assert SMS.table(outs, {1: led, 2: led})["verdict"] == "UNDEFINED"
