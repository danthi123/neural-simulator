"""Regression pin for the 2026-09-29 crash in `research.runners.host_share_bg_seam.score()`: it called
`host_share.classify_seam(F, S, D)` (three pooled scalars) -- a function that does not exist on that module, and
even on `host_share_schema.classify_seam` takes one `stats` dict built by `seam_stats(events)`. No brain build:
`score()` reads plain-dict shard files from disk, so this fabricates a minimal one-seed shard tree and asserts
`score()` runs to completion and returns a valid classification, instead of raising `AttributeError`."""
import json
import os
import tempfile

os.environ.setdefault("SIM_BACKEND", "numpy")

import research.runners.host_share_bg_seam as M
from research.runners import host_share_schema as SCHEMA


def _resp(held):
    return {"answer": "x", "abstained": False, "recalled_svo": ["dog", "chase", "ball"],
            "bg_select": {"action": "STAY_SILENT" if held else "SPEAK", "on": held},
            "source": "tiny-demo +LTM"}


def _write_fake_shard(shard_dir, seed, flip_intervene=True):
    sdir = os.path.join(shard_dir, "s%d" % seed)
    os.makedirs(sdir, exist_ok=True)
    labels = [t[0] for t in M.TURNS]
    for arm in M.ARMS:
        responses = {}
        for i, lab in enumerate(labels):
            held = (i % 2 == 0)
            if arm == "prod_intervene" and flip_intervene:
                held = not held
            responses[lab] = _resp(held)
        json.dump({"responses": responses, "events": {}}, open(os.path.join(sdir, arm + ".json"), "w"))


def test_score_does_not_raise_and_classifies():
    with tempfile.TemporaryDirectory() as d:
        _write_fake_shard(d, 7)
        out = M.score(d, [7])
        assert out["production_seam_class"] in SCHEMA.CLASSES
        assert set(("F", "S", "D")) <= set(out["pooled"])


def test_production_seam_class_uses_schema_not_a_missing_attribute():
    """The bug's exact shape: `host_share` (the module `lane_c()` returns) has no `classify_seam` attribute."""
    import research.runners.host_share as host_share
    assert not hasattr(host_share, "classify_seam"), (
        "host_share now defines classify_seam directly -- if this is intentional, update score() accordingly "
        "and this test may be relaxed, but the fix must keep using a real, existing callable.")
    with tempfile.TemporaryDirectory() as d:
        _write_fake_shard(d, 7)
        # Must not raise AttributeError (the original crash).
        M.score(d, [7])


def test_too_few_seeds_is_undefined_not_a_crash():
    with tempfile.TemporaryDirectory() as d:
        _write_fake_shard(d, 7)
        out = M.score(d, [7])
        assert out["production_seam_class"] == "UNDEFINED"  # 1 seed < SCHEMA.MIN_LIVE_SEEDS_GATE
