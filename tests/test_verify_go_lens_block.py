"""The land-result skill carries the REQUIRED refuter-lens block (Claude review 2026-10-06; moved there from the retired
verify-go skill 2026-11-02). Hermes pastes it verbatim into
every refuter prompt, so the block must exist between its markers, hold the three lenses, and name the tools that make
them mechanical (tools/answer_rate.py, tools/where_ran.py). A reworded or deleted lens fails here."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = os.path.join(ROOT, ".claude", "skills", "land-result", "SKILL.md")


def _block():
    text = open(SKILL, encoding="utf-8").read()
    m = re.search(r"<!-- REFUTER-LENS-BEGIN -->\n(.*?)\n<!-- REFUTER-LENS-END -->", text, re.S)
    assert m, "the refuter-lens block markers are missing from the land-result skill"
    return m.group(1)


def test_block_has_the_three_lenses_in_order():
    b = _block()
    pos = [b.index(x) for x in ("(a) ENGAGEMENT", "(b) ANSWER-RATE", "(c) WHERE-RAN")]
    assert pos == sorted(pos)
    assert "never a PASS or a FAIL" in b and "UNDEFINED" in b                       # (a): no record is UNDEFINED
    assert "FLIP-HURTS-ANSWERS" in b and "Claude ruling" in b                       # (b): the consequence is stated
    assert "tools/where_ran.py" in b and "tools/answer_rate.py" in b                # (c) and (b): the tools exist
    assert "I don't know about that." in b


def test_tools_the_block_names_exist():
    for rel in ("tools/where_ran.py", "tools/answer_rate.py"):
        assert os.path.exists(os.path.join(ROOT, rel)), rel


def test_block_is_required_verbatim_and_appears_once():
    text = open(SKILL, encoding="utf-8").read()
    assert text.count("<!-- REFUTER-LENS-BEGIN -->") == 1 and text.count("<!-- REFUTER-LENS-END -->") == 1
    assert "VERBATIM" in text.split("<!-- REFUTER-LENS-BEGIN -->")[0].rsplit("###", 1)[-1]
