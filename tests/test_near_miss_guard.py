"""Near-miss guard (brain/near_miss.py, brain/sleep.py guard term) on the tiny random RWKV-7-shaped mock (no weights)."""
import torch

from brain.experiments.mixed_day_probe import names_eval_topic
from brain.near_miss import _first_question
from brain.plastic_cortex import PlasticCortex
from brain.sleep import sleep
from tests.test_brain_sleep import cortex  # noqa: F401  (fixture)


def _pc(c):
    torch.manual_seed(0)
    return PlasticCortex(c, rank=2)


def test_no_guard_is_the_old_loop(cortex):
    a, b = _pc(cortex), _pc(cortex)
    base = [s.float() for s in a.zero_state()]
    eps, gen = [("User: My cat is Pixel.\n\n", 1.0), ("User: I was born in Lyon.\n\n", 2.0)], ["User: hi\n\n"]
    la = sleep(a, base, eps, gen, steps=12, lr=1e-2, seed=3)
    lb = sleep(b, base, eps, gen, steps=12, lr=1e-2, seed=3, guard=())
    assert la == lb


def test_guard_keeps_the_replay_sequence_and_holds_guarded_text(cortex):
    pc_off, pc_on = _pc(cortex), _pc(cortex)
    base = [s.float() for s in pc_off.zero_state()]
    eps, gen = [("User: My cat is Pixel.\n\n", 1.0)], ["User: hi\n\n"]
    g = ["User: What is my horse called?\n\nAssistant: ok\n\n"]
    l_off = sleep(pc_off, base, eps, gen, steps=40, lr=3e-2, seed=1)
    l_on = sleep(pc_on, base, eps, gen, steps=40, lr=3e-2, seed=1, guard=g, guard_weight=5.0)
    assert l_off[0][0] == l_on[0][0]   # the first replay draw and loss are the same

    def drift(pc):
        ids = cortex.pipe.encode(g[0])
        with torch.no_grad():
            pc.enabled = False
            ref = torch.log_softmax(pc(ids, base)[0], -1)
            pc.enabled = True
            on = torch.log_softmax(pc(ids, base)[0], -1)
        return float(torch.nn.functional.kl_div(on, ref, log_target=True, reduction="batchmean"))
    assert drift(pc_on) < drift(pc_off)


def test_first_question_and_eval_hygiene():
    assert _first_question('"Where does my aunt live?" And more') == "Where does my aunt live?"
    assert names_eval_topic("Where does my aunt live?") and names_eval_topic("What's my father's job?")
    assert not names_eval_topic("Where does my grandfather live?") and not names_eval_topic("What is my cat called?")
