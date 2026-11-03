"""Retrieval-practice sleep (M3, brain/sleep_retrieval.py) on the tiny random RWKV-7-shaped mock (no weights)."""
import torch

from brain.plastic_cortex import PlasticCortex
from brain.sleep_retrieval import encode_pair, practice_loss, selfq_text, sleep_retrieval, teacher_answers
from tests.test_brain_sleep import cortex  # noqa: F401  (fixture)


def _pc(c):
    torch.manual_seed(0)
    return PlasticCortex(c, rank=2)


def test_selfq_text_reattaches_onset():
    assert selfq_text("is my cat called?\nmore") == "What is my cat called?"
    assert selfq_text("'s my dog?") == "What's my dog?"


def test_answer_span_is_only_the_answer(cortex):
    ids, start, n = encode_pair(cortex.pipe, "Q?", "Pixel.")
    assert cortex.pipe.decode(ids[start:start + n]) == " Pixel."
    assert cortex.pipe.decode(ids[start + n:]) == "\n\n"
    assert cortex.pipe.decode(ids[:start]).endswith("Assistant:")


def test_teacher_uses_pretrained_cortex_and_restores_switch(cortex):
    pc = _pc(cortex)
    base = [s.float() for s in pc.zero_state()]
    pairs = teacher_answers(pc, base, "My cat is Pixel.", ["What is it?"], max_tokens=4)
    assert pc.enabled
    assert all(isinstance(a, str) and a for _, a in pairs)


def test_failed_flag_matches_greedy_divergence(cortex):
    pc = _pc(cortex)
    base = [s.float() for s in pc.zero_state()]
    ids, start, n = encode_pair(cortex.pipe, "Q?", "ab")
    with torch.no_grad():
        logits, _ = pc(ids, base)
    greedy = logits[start - 1:start - 1 + n].argmax(-1).tolist()
    want = greedy != ids[start:start + n]
    _, failed = practice_loss(pc, base, "Q?", "ab")
    assert failed == want
    # an answer equal to the cortex's own greedy continuation is a success
    own = cortex.pipe.decode(greedy)[1:] if cortex.pipe.decode(greedy).startswith(" ") else None
    if own is not None:
        _, failed2 = practice_loss(pc, base, "Q?", own)
        assert failed2 is False


def test_sleep_retrieval_lowers_answer_loss_and_only_moves_adapters(cortex):
    pc = _pc(cortex)
    base = [s.float() for s in pc.zero_state()]
    frozen = {k: v.clone() for k, v in pc.z.items()}
    prac = [[("Q?", "xy")], [("R?", "zw")]]
    before = sum(float(practice_loss(pc, base, q, a)[0]) for p in prac for q, a in p)
    hist = sleep_retrieval(pc, base, prac, ["User: hi\n\n"], steps=60, lr=3e-2, kl_weight=0.1, seed=1)
    after = sum(float(practice_loss(pc, base, q, a)[0]) for p in prac for q, a in p)
    assert len(hist) == 60 and after < before
    assert all(torch.equal(v, frozen[k]) for k, v in pc.z.items())
    assert pc.adapter_norm() > 0
