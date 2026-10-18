"""Pins for the claim-entailment verifier ("the moat"), research/runners/_moat_claim_entailment_derisk.py.

PART 1 -- BYTE-IDENTITY (this commit, before any change to the verifier): with no slug-norm option set the verifier's
output (accepted, reject_reason and a hash of the FULL result dict, trace included) is the same as the unmodified
verifier's on 30 toy-world suite prose + 178 real-Qwen sentences (the 2026-10-18 memo corpus, each with its turn's
gathered set) + 20 template-stub literal-slug sentences + 120 smuggling mutations of those. The golden file was captured
from the UNMODIFIED verifier (see its `generated_from`); regenerate it only from an unmodified verifier.

Run: CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m pytest tests/test_moat_slug_norm.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners import _moat_claim_entailment_derisk as M
from research.runners.brain_conversational_agent import BridgeParser

pytestmark = pytest.mark.filterwarnings("ignore")

GOLDEN = json.loads((Path(__file__).parent / "fixtures" / "moat_claim_entailment_golden.json").read_text())


@pytest.fixture(scope="module")
def parser():
    return BridgeParser(seed=GOLDEN["seed"])


def _digest(res):
    return hashlib.sha256(json.dumps(res, sort_keys=True, default=str).encode()).hexdigest()[:20]


def _build_like_production(parser, gated, **kw):
    """Mirror of ChatBrain._build_claim_verifier's construction (None on a role-permutation collision)."""
    g = [list(f) for f in {tuple(f) for f in gated}]
    nouns = {t for f in g for t in (f[0], f[2])}
    verbs = {f[1] for f in g}
    try:
        return M.ClaimEntailmentVerifier(parser, g, nouns, verbs, M.VERB_SYNONYMS,
                                         M._build_inflection_map(sorted(verbs)), **kw)
    except AssertionError:
        return None


def test_golden_is_nontrivial():
    """Anti-vacuity: the golden holds both accepts and rejects, and the unmodified-verifier provenance."""
    assert GOLDEN["generated_from"]
    assert sum(1 for t in GOLDEN["toy"] if t["a"]) >= 10 and sum(1 for t in GOLDEN["toy"] if not t["a"]) >= 10
    w = GOLDEN["world"]
    assert len(w) >= 300
    assert sum(1 for e in w if e["a"]) >= 50 and sum(1 for e in w if e["a"] is False) >= 100


@pytest.mark.parametrize("kw", [{}], ids=["default"])
def test_toy_suite_matches_unmodified_verifier(parser, kw):
    ver = M.ClaimEntailmentVerifier(parser, M.GATED, M.NOUNS, M.VERBS, M.VERB_SYNONYMS, M.INFLECT, **kw)
    bad = []
    for t in GOLDEN["toy"]:
        r = ver.verify(t["p"])
        if (r["accepted"], r["reject_reason"], _digest(r)) != (t["a"], t["r"], t["h"]):
            bad.append((t["p"], r["accepted"], r["reject_reason"]))
    assert not bad, bad[:3]


@pytest.mark.parametrize("kw", [{}], ids=["default"])
def test_real_world_sentences_match_unmodified_verifier(parser, kw):
    bad = []
    for e in GOLDEN["world"]:
        ver = _build_like_production(parser, GOLDEN["gated_sets"][e["g"]], **kw)
        if ver is None:
            assert e["a"] is None, e
            continue
        r = ver.verify(e["s"])
        if (r["accepted"], r["reject_reason"], _digest(r)) != (e["a"], e["r"], e["h"]):
            bad.append((e["src"], e["s"], r["accepted"], r["reject_reason"]))
    assert not bad, bad[:3]
