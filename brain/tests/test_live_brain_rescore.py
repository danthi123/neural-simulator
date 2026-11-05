"""The live-brain 12-fact rescore: correct_answer recomputed per told/para row with key and told sentence from FACTS
by position (echo and tautology void, "I believe you drive a Volvo" counts, wrong-subject voids), false recall
unchanged, and the old table call still prints the committed table.txt byte for byte. No weights, no model.
"""
import json
from pathlib import Path

from brain.experiments import live_brain_rescore as lr
from brain.experiments import live_brain_table as lt

DIR = Path(__file__).resolve().parents[2] / "brain/experiments/results/2026-11-03-live-brain-12facts"


def _fixture():
    """One seed, one arm; told/para rows in FACTS order (positions 0-5 / 0-2), one never row."""
    told = [("What is my cat called?", "Pixel is called Pixel."),                       # tautology: void
            ("What is my favourite colour?", "My horse is called Teal."),               # wrong subject: void
            ("Where was I born?", "Lyon."),                                             # bare key: counts
            ("What is my sister's name?", "My sister's name is Marguerite."),           # echo of the told line: void
            ("What is my dog's name?", "Biscuit."),                                     # bare key: counts
            ("What car do I drive?", "I believe you drive a Volvo.")]                   # first-person word alone: counts
    para = [("What's the name of my kitty?", "Pixel."),                                  # counts
            ("Which colour do I like best?", "Teal is called Teal."),                   # tautology: void
            ("What's my hometown?", "Lyon.")]                                           # counts
    rows = [{"q": q, "kind": "told", "reply": r, "recalled": None, "lenient": True, "hit": True, "echo": False}
            for q, r in told]
    rows += [{"q": q, "kind": "para", "reply": r, "recalled": None, "lenient": True, "hit": True, "echo": False}
             for q, r in para]
    rows += [{"q": "What is my horse called?", "kind": "never", "reply": "Your horse is called Biscuit.",
              "recalled": None, "false_recall": True, "false_lenient": True}]
    arm = {"recall": 6, "paraphrase": 3, "false_recall": 1, "echo": 2, "recall_lenient": 6,
           "paraphrase_lenient": 3, "false_lenient": 1, "rows": rows}
    return {"seed": 42, "arms": {a: dict(arm) for a in lt.ARMS}}   # the table reads all five arms


def test_rescore_counts_echo_tautology_wrong_subject_and_counts_the_believe_reply():
    out = lr.rescore_seed(_fixture())["arms"]["a_intact_sleep"]
    assert out["recall_correct"] == 3            # Lyon. / Biscuit. / I believe you drive a Volvo.
    assert out["paraphrase_correct"] == 2        # Pixel. / Lyon.
    flags = [(r["kind"], r["reply"], r.get("correct")) for r in out["rows"]]
    assert ("told", "Pixel is called Pixel.", False) in flags                     # tautology void
    assert ("told", "My horse is called Teal.", False) in flags                   # wrong subject void
    assert ("told", "My sister's name is Marguerite.", False) in flags            # echo void
    assert ("told", "I believe you drive a Volvo.", True) in flags                # counted
    assert all("correct" not in r for r in out["rows"] if r["kind"] == "never")   # never rows untouched


def test_false_recall_and_stored_metrics_are_unchanged():
    out = lr.rescore_seed(_fixture())["arms"]["a_intact_sleep"]
    assert out["false_recall"] == 1 and out["false_lenient"] == 1
    assert out["echo"] == 2 and out["recall_lenient"] == 6 and out["paraphrase_lenient"] == 3


def test_old_table_call_prints_the_committed_table_txt_byte_for_byte(capsys):
    lt.main(str(DIR))
    assert capsys.readouterr().out == (DIR / "table.txt").read_text()


def test_rescore_writes_only_new_files(tmp_path, capsys):
    seed = tmp_path / "seed42.json"
    seed.write_text(json.dumps(_fixture()))
    (tmp_path / "table.txt").write_text("do not touch\n")
    before = seed.read_bytes()
    lr.main(str(tmp_path))
    assert seed.read_bytes() == before                       # the seed JSON is never modified
    assert (tmp_path / "table.txt").read_text() == "do not touch\n"
    rescored = json.loads((tmp_path / "rescored.json").read_text())
    assert rescored[0]["arms"]["a_intact_sleep"]["recall_correct"] == 3
    table = (tmp_path / "table_correct.txt").read_text()
    assert table.startswith("rescored with strict_scorer.correct_answer")
    assert "recall_correct" not in table and "P1 " in table  # the P-checks are printed for Claude to rule on
    assert "seeds [42]" in table