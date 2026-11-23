# Capability: Lifelong learning
# Record: brain/research/2026-11-23-stage2-test2-prereg.md
# Expected-fail: none (host-only byte-identity check on synthetic lives)
"""The existing --emit output of brain/eval/life_read.py is byte-identical to the output generated BEFORE the G9
cross-arm mode was added (2026-11-23): sha256 of every emitted file for fixed synthetic inputs (the lives of
test_life_read.py). A change to those lives or to the frozen read spec changes these hashes on purpose."""
import hashlib
import os

import pytest

from brain.eval import life_read as R
from brain.tests import test_life_read as T

# generated with the pre-G9 life_read.py (sha256 of each file)
FROZEN = {
    "syn": {
        "answer_prompts.jsonl": "6aad7af0e5504ee32b5a9ed774ff8011733d2fc610ad8c48ed0553b29f1fd3af",
        "blind_key.json": "08ff0144c4f7a306f944b7b136d724d7cf074e8afd73a76d22afb67665171186",
        "pairs.jsonl": "cae16aea753fe1739c832e589d14a024e27bf4199daa11410cc963bf2d2a32c0",
        "reader_prompts.jsonl": "40bb6755cef46288c259f80ac36d56557d4354db23dd62a0f82c15dfa8291662",
        "units.jsonl": "b89e813b7b546106e1a86334e8fedad0276aaf58cacfe00552e3e24ca5ee4a0c",
    },
    "syn_intact_w3": {
        "answer_prompts.jsonl": "42bbc50f51cbb80e1f80e01ef7cfb5aa70094e97757f59dfe73c185e3ec51bbc",
        "blind_key.json": "0f4feb6e772eb65744859843717ad75e46023587a0e540d897781758c3869349",
        "pairs.jsonl": "4e85d9102adcf41b10f13e6a1c0c0b93a84be29ecc07a4fd904f982c93451804",
        "reader_prompts.jsonl": "73b0742e57e56d2895ff02743197f56c58d3f4208cbd0f79b300bfaff7af2cc1",
        "units.jsonl": "a89d88a9ff03c7afe2a13f06d140e3646f9f266ce52b91d8f234f553039c6e69",
    },
    "syn_short": {
        "answer_prompts.jsonl": "9e8c6df75893708022a6235f2228652397fec419ac8903eb11e92d5b6b33732b",
        "blind_key.json": "ee0db95db5ab4c68925cd674302023731a1fd7ab0670f40d3f3d677d7998dbc8",
        "pairs.jsonl": "c941e3510c3fd01dadc47212bf9ffb94174cc56062030d9ff9eb71dfb1748662",
        "reader_prompts.jsonl": "334d57a1657242f6a45535c4b9e2e735c647877091ef0fcd935c228e25088664",
        "units.jsonl": "f19be130de515b3f6add1e75c3d254d37a063be9e2c9f275fadecc8f8fe1c673",
    },
}
CASES = {"syn": (lambda: T.LIVES, ["--weeks", "1,3"]),
         "syn_intact_w3": (lambda: T.LIVES, ["--weeks", "3", "--lived-context", "intact", "--rng-seed", "5"]),
         "syn_short": (lambda: [T.life(7, 7), T.life(8, 7)], [])}


@pytest.mark.parametrize("name", sorted(FROZEN))
def test_emit_byte_identical(tmp_path, name):
    import json
    lives, args = CASES[name]
    paths = []
    for i, lf in enumerate(lives()):
        p = tmp_path / f"life{i}.json"
        p.write_text(json.dumps(lf))
        paths.append(str(p))
    d = tmp_path / "out"
    R.main(["--emit", str(d), "--lives", *paths, *args])     # the CLI path, as the orchestrator calls it
    got = {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in sorted(os.listdir(d))}
    assert got == FROZEN[name]
