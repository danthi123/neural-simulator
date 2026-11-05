"""The mood3 exploratory re-analysis: the fixed refusal instrument, and the recorded pairing — the re-analysis must
reproduce the stored pos_null / neg_null lex diffs to 1e-6 on all 12 JSONs (it uses the same (q, k) pairing as
mood_probe.main). Reads the recorded JSONs only; no model, no weights, nothing written."""
import glob
import json

from brain.experiments import mood_v3
from brain.experiments.sentiment_instrument import is_refusal

D = "brain/experiments/results/2026-11-03-mood3"


def test_is_refusal_cases():
    assert is_refusal("I'm sorry, but I can't help with that.")
    assert is_refusal("I\u2019m sorry, but I cannot assist.")          # curly apostrophe
    assert is_refusal("AS AN AI, I must decline.")
    assert is_refusal("I am unable to do that.")
    assert is_refusal("i cannot help with that")
    assert is_refusal("I'm unable to help.")
    assert not is_refusal("I love this, it makes me smile.")
    assert not is_refusal("It rained all day, sorry.")                  # the bare word is not a refusal pattern
    assert not is_refusal("My favourite colour is teal.")
    assert not is_refusal("The answer is no, but here is why.")


def test_pairing_reproduces_the_stored_null_lex_diffs():
    files = sorted(glob.glob(f"{D}/mood3_*_s*.json"))
    assert len(files) == 12                     # six seeds at 0.1B and 0.4B
    for f in files:
        j = json.load(open(f))
        for sg in ("pos", "neg"):
            d = mood_v3.lex_null_diffs(j["rows"], sg)
            stored = j["contrasts"][f"{sg}_null"]["lex_diff"]
            assert abs(sum(d) / len(d) - stored) < 1e-6, (f, sg, stored)