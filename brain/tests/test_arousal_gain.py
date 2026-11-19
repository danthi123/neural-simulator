"""Brain(arousal_gain=...) (mood iteration 6, switch default OFF) and the arousal_gain harness.

Off is byte-identical (hear and commit_exchange); LESION (pinned) is byte-identical to the split OFF path; the split
read matches the joint read (S1; the 1e-5 criterion is measured and reported, the harness falls back to the split path
for OFF when it fails); the banks are disjoint from every earlier bank in the repo."""
import glob
import os
import random
import re

import numpy as np
import pytest
import torch

from brain.affect import Affect, calibrate_surprise, top2_mean
from brain.brain import BIRTH, Brain
from brain.cortex import Cortex, resolve_model
from brain.experiments import arousal_gain as ag
from brain.tests.test_brain_key_switch import KW, StubCortex

torch.set_num_threads(4)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAL = (8.0, 1.5)


class NllStub(StubCortex):
    """The stand-in cortex with a per-token NLL (the real one needs the model package's full output)."""

    def read_nll(self, text, state=None):
        ids = self.pipe.encode(text)
        lg, st = self.forward(ids, state)
        nll = torch.tensor([float(abs(torch.sin(torch.tensor(float(t))) * 9.0)) for t in ids[1:]])
        return lg, st, nll


@pytest.fixture(scope="module")
def cortex():
    return Cortex(resolve_model("0.1b"))


def _seed(n=1):
    random.seed(n); np.random.seed(n); torch.manual_seed(n)


def test_switch_off_is_byte_identical(tmp_path):
    c = StubCortex()
    b = Brain(str(tmp_path / "a"), cortex=c, **KW)
    assert b.arousal_gain is None and b.arousal is None
    st = [s.clone() for s in b.state]
    joint = b.cortex.read("User: Hello there.\n\nAssistant: Hi.\n\n", st)[1]
    assert all(torch.equal(x, y) for x, y in zip(b.commit_exchange("Hello there.", "Hi.", st), joint))
    b.own_gain = 0.5
    su = b.cortex.read("User: Hello there.\n\nAssistant:", st)[1]
    sp = b.cortex.read(" Hi.\n\n", su)[1]
    out = b.commit_exchange("Hello there.", "Hi.", st)
    assert all(torch.equal(out[i], (su[i] + 0.5 * (sp[i] - su[i])) if i % 3 == 1 else sp[i]) for i in range(len(sp)))
    _seed(); r0, i0 = b.hear("I went for a long walk by the river.")
    assert "arousal" not in i0


def test_off_hear_reads_the_joint_prompt_only(tmp_path):
    """Off: hear reads "User: ...\n\nAssistant:" in one call (as before) and never touches read_nll."""
    class Spy(NllStub):
        reads = []

        def read(self, text, state=None):
            Spy.reads.append(text)
            return super().read(text, state)

        def read_nll(self, text, state=None):
            raise AssertionError("read_nll used with the switch off")
    b = Brain(str(tmp_path / "a"), cortex=Spy(), **KW)
    Spy.reads.clear()
    _seed(); b.hear("My cat is called Pixel.")
    assert "User: My cat is called Pixel.\n\nAssistant:" in Spy.reads


def test_gain_math_on_the_stub(tmp_path):
    c = NllStub()
    b = Brain(str(tmp_path / "a"), cortex=c, **KW, arousal_gain=1.0, arousal_calibration=CAL)
    st = [s.clone() for s in b.state]
    _, su, _ = c.read_nll("User: Hello there.\n\n", st)
    g = 1.5
    out = b._write_gain(st, su, g)
    for i in range(len(st)):
        if i % 3 == 1:
            assert torch.allclose(out[i], st[i] + g * (su[i] - st[i]))
        else:
            assert torch.equal(out[i], su[i])
    assert b._write_gain(st, su, 1.0) is su
    b._turn_gain = 1.5
    com = b.commit_exchange("Hello there.", "Hi.", st)
    sp = c.read("Assistant: Hi.\n\n", out)[1]
    assert all(torch.equal(x, y) for x, y in zip(com, sp))
    assert b._turn_gain == 1.0                                  # consumed
    _seed(); _, info = b.hear("Hello there.")
    assert info["arousal"]["gain_on"] and 1.0 <= info["arousal"]["encode_gain"] <= 2.0


def test_affect_arousal_drive_and_gain():
    a = Affect(enabled=False, surprise_mu=8.0, surprise_sd=2.0)
    a.drive_arousal(8.0)
    assert a.arousal == 0.0 and a.encode_gain(1.0) == 1.0           # at the mean: no gain, never below 1
    a.drive_arousal(20.0)                                            # clipped at 3 SD: s = 1
    assert a.arousal == pytest.approx(1 / 1.5)
    assert a.encode_gain(1.0) == pytest.approx(1 + 1 / 1.5)
    assert a.encode_gain(5.0) == 2.0                                 # ceiling
    assert a.encode_gain(5.0, pinned=True) == 1.0                    # lesion
    assert a.mood == 0.0                                             # does not touch mood
    r = a.readout()
    assert {"surprise", "encode_gain", "gain_on"} <= set(r)
    assert top2_mean([1.0, 5.0, 3.0, 4.0]) == 4.5 and top2_mean([]) == 0.0


def test_switch_on_needs_a_calibration(tmp_path):
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "a"), cortex=StubCortex(), **KW, arousal_gain=1.0)


def test_switch_on_raises_when_consolidated_cortex_is_active(cortex, tmp_path):
    b = Brain(str(tmp_path / "a"), cortex=cortex, arousal_gain=1.0, arousal_calibration=CAL)
    b.cortex._changed = True                                          # declared scope: brains that have not slept
    with pytest.raises(RuntimeError):
        b.cortex.read_nll("User: My cat is called Pixel.\n\n", b.state)     # the call hear makes when the switch is on


def test_read_nll_matches_read_and_nll_is_the_models(cortex):
    _, base = cortex.read(BIRTH)
    t = "User: My neighbour keeps a plimberry in her garage.\n\n"
    lg, st = cortex.read(t, base)
    lg2, st2, nll = cortex.read_nll(t, base)
    assert (lg - lg2).abs().max() < 1e-4 and len(nll) == len(cortex.pipe.encode(t)) - 1
    assert all((x - y).abs().max() < 1e-4 for x, y in zip(st, st2))
    assert float(nll.min()) >= 0.0


def test_split_read_vs_joint_read_s1(cortex, tmp_path):
    """S1: split vs joint logits (the 1e-5 criterion is reported by `arousal_gain smoke`; here 1e-3) and 20 greedy turns."""
    calib = ag.calibration(cortex, cortex.read(BIRTH)[1])
    _, base = cortex.read(BIRTH)
    s1 = ag.s1_check(cortex, base, calib, turns=6, max_tokens=16, scratch=str(tmp_path))
    assert s1["max_logit_diff"] < 1e-3 and s1["replies_identical"] and s1["lesion_equals_split_off"]
    assert s1["off_split"] == (not s1["S1_pass"])


def test_lesion_is_byte_identical_to_the_split_off_path(cortex, tmp_path):
    """S2 (the fallback form: OFF on the split path). Greedy 6 turns, states bit-equal."""
    calib = {"mu": 9.0, "sd": 1.5}
    a = ag.make_brain(str(tmp_path / "a"), cortex, calib, "OFF", 1.0, True, 20, greedy=True)
    b = ag.make_brain(str(tmp_path / "b"), cortex, calib, "LESION", 2.0, False, 20, greedy=True)
    for t in ag.smoke_turns()[:6]:
        ra, _ = a.hear(t)
        rb, ib = b.hear(t)
        assert ra == rb and all(torch.equal(x, y) for x, y in zip(a.state, b.state))
    assert ib["arousal"]["encode_gain"] == 1.0 and ib["arousal"]["surprise"] > 0       # computed, not applied


def test_on_changes_the_state_on_a_surprising_turn(cortex, tmp_path):
    _, base = cortex.read(BIRTH)
    calib = ag.calibration(cortex, base)
    on = ag.make_brain(str(tmp_path / "a"), cortex, calib, "ON", 2.0, False, 12, greedy=True)
    les = ag.make_brain(str(tmp_path / "b"), cortex, calib, "LESION", 2.0, False, 12, greedy=True)
    _, i1 = on.hear("Yesterday my aunt gave me a snudgeloom for my birthday.")
    les.hear("Yesterday my aunt gave me a snudgeloom for my birthday.")
    assert i1["arousal"]["encode_gain"] > 1.2
    assert not all(torch.equal(x, y) for x, y in zip(on.state, les.state))


def test_replayed_gains_and_current_arm(cortex, tmp_path):
    calib = {"mu": 9.0, "sd": 1.5}
    b = ag.make_brain(str(tmp_path / "a"), cortex, calib, "SHUFFLED", 1.0, False, 8, greedy=True)
    b.arousal_gains = [1.7, 1.2]
    _, i1 = b.hear("My cat is called Pixel.")
    _, i2 = b.hear("I need to go to the shop later.")
    assert i1["arousal"]["encode_gain"] == 1.7 and i2["arousal"]["encode_gain"] == 1.2
    cur = ag.make_brain(str(tmp_path / "c"), cortex, calib, "CURRENT", 1.0, False, 8)
    assert cur.arousal.TAU_AROUSAL == 1.0


# ---------------------------------------------------------------- banks
def _words_of(text):
    return set(re.findall(r"[a-z']+", text.lower()))


def test_banks_shape():
    assert len(ag.BANKS["dev"]) == 12 and len(ag.BANKS["test"]) == 24
    for name, frames in (("dev", ag.DEV_FRAMES), ("test", ag.TEST_FRAMES)):
        bank = ag.BANKS[name]
        for i in range(len(frames)):
            pair = [c for c in bank if c["pair"] == i]
            assert sorted(c["cls"] for c in pair) == ["M", "S"]
            assert pair[0]["turns"][1:] == pair[1]["turns"][1:] == ag.FILLERS + [ag.T4]
            assert len({c["turns"][0].replace(c["item"], "{}") for c in pair}) == 1       # matched frame
    assert ag.T4 == "How's your day going?"


def test_banks_are_disjoint_from_every_earlier_bank():
    mine = ROOT + "/brain/experiments/arousal_gain.py"
    older = [p for p in glob.glob(ROOT + "/brain/**/*.py", recursive=True) + glob.glob(ROOT + "/brain/**/*.json", recursive=True)
             if os.path.abspath(p) not in (os.path.abspath(mine), os.path.abspath(__file__)) and "/results/" not in p]
    corpus = ""
    for p in older:
        corpus += open(p, errors="ignore").read().lower() + "\n"
    cw = _words_of(corpus)
    coined = [f[2] for f in ag.DEV_FRAMES + ag.TEST_FRAMES]
    assert len(set(coined)) == len(coined)
    for w in coined:
        assert w not in cw, w
    # the dev and test banks share no coined word, frame or filler-free T1 sentence
    dev_t1 = {c["turns"][0] for c in ag.BANKS["dev"]}
    test_t1 = {c["turns"][0] for c in ag.BANKS["test"]}
    assert not (dev_t1 & test_t1)
    assert not ({f[0] for f in ag.DEV_FRAMES} & {f[0] for f in ag.TEST_FRAMES})
    # the calibration sentences are not in either bank nor in the stage-0 banks / V5 events
    from brain.experiments.novelty_stage0 import DEV_FRAMES, FRAMES
    stage0 = {f[0].format(x).lower() for f in FRAMES + DEV_FRAMES for x in f[1:]}
    banked = {t.lower() for c in ag.BANKS["dev"] + ag.BANKS["test"] for t in c["turns"]}
    for s in ag.CAL_SENTENCES:
        assert s.lower() not in stage0 | banked and s not in ag.FILLERS
    assert len(ag.CAL_SENTENCES) == 12
    from brain.experiments.mood_probe import NEU_EVENTS_V5_DEV
    assert not set(ag.CAL_SENTENCES) & set(NEU_EVENTS_V5_DEV)


def test_carry_measure():
    assert ag.carries("Oh, a snudgeloom! What is it?", "snudgeloom") and not ag.carries("Nice hat", "scarf")
    assert ag.carries("I like the cookies.", "cookies") and ag.carries("A cookie!", "cookies")
    assert not ag.carries("How lovely.", "scarf")


# ---------------------------------------------------------------- table
def _fake_rows():
    rows = []
    rng = random.Random(0)
    for arm in ("ON", "LESION", "SHUFFLED", "CONSTANT", "CURRENT"):
        for seed in (42, 43):
            for conv in ag.BANKS["dev"]:
                for t in (1, 2, 3, 4):
                    s = conv["cls"] == "S"
                    reply = f"reply {arm if arm == 'ON' else 'x'} {conv['id']} {t} {rng.random():.3f}" if arm == "ON" and s else f"other {conv['id']} {t}"
                    rows.append({"arm": arm, "seed": seed, "sample": 0, "conv": conv["id"], "cls": conv["cls"], "pair": conv["pair"],
                                 "turn": t, "t1": conv["turns"][0], "text": "x", "reply": reply, "item": conv["item"],
                                 "gain": (1.8 if t == 1 and s else 1.0) if arm != "LESION" else 1.0,
                                 "arousal": 0.8 if (t == 1 and s) else 0.05, "surprise": 1.0, "carry": False, "copy": False,
                                 "q": False, "leak": False, "nll": 3.0})
    return rows


def test_table_runs_and_reads_labels(tmp_path):
    rows = _fake_rows()
    labels = {}
    for r in rows:
        if r["turn"] in (2, 4):
            on = r["arm"] == "ON" and r["cls"] == "S"
            labels[ag.item_id(r["t1"], r["reply"])] = [{"a": on, "b": False, "c": False}, {"a": on, "b": False, "c": False}]
    res = ag.table(rows, labels, overlap_min=1)
    assert res["C0"]["auroc"] == 1.0 and res["C0"]["pass"]
    assert res["C1"]["ON_minus_LESION_a"]["mean"] == pytest.approx(1.0) and res["C1"]["pass"]
    assert res["C3"]["pass"] and res["C5"]["pass"]
    assert res["reader_agreement"]["kappa_a"] is not None
    assert "branch" in res and res["name"]["claim"]
