"""mood_probe --add-embedding (embedding scored off-node) must reproduce a run scored on the node."""
import json

import pytest

SRC = "brain/experiments/results/2026-11-03-mood3/mood3_0.1b_s42.json"


def test_add_embedding_matches_on_node_scoring(tmp_path):
    pytest.importorskip("transformers")
    from brain.experiments import mood_probe
    try:
        from brain.experiments.sentiment_instrument import Embedding
        Embedding()
    except OSError:
        pytest.skip("MiniLM not in the local cache")
    ref = json.load(open(SRC))
    j = json.load(open(SRC))
    for arm in j["rows"].values():
        for r in arm:
            r.pop("emb", None)
    p = tmp_path / "s42.json"
    p.write_text(json.dumps(j))
    mood_probe.add_embedding(str(p))
    got = json.load(open(p))
    for k, c in ref["contrasts"].items():
        for kk, v in c.items():
            assert got["contrasts"][k][kk] == pytest.approx(v, abs=1e-6), (k, kk)
