"""Verify the 2026-09-04 generator-trace-mislabel fix through the REAL `webapp.open_ended_chat.answer_turn` /
`webapp.wkv_mouth_generator.generate` entry points -- NOT a parallel harness.

THE BUG (found 2026-09-03 during the linattn live verification, see research/findings/2026-09-04-generator-
trace-mislabel-fix.md for the full root-cause writeup). `answer_turn` labelled a reply `generator="wkv_mouth"`
purely because it was produced from inside its own WKV-mouth try-block -- never checking that
`wkv_mouth_generator.generate()` can ITSELF dispatch to `render_fact_sentence` (the SAME SpikingClauseProducer
mechanism the SEPARATE fact-clause-fallback branch wires in) when `sentence_facts` names a lexicon-covered
relation. Under `BRAIN_WKV_MOUTH_SCOPE=broad` (where `in_vocab_scope` admits every prompt) this fired on nearly
every known-topic reply, corrupting the per-touchpoint Qwen-vs-substrate provenance the one-brain roadmap's
de-risk #2 depends on (research/findings/2026-09-03-one-brain-mouth-integration-ROADMAP.md SS3).

THE FIX. `generate()` gained an additive `trace: dict | None = None` out-parameter recording which of its own
two branches produced the text; `answer_turn` reads it back and labels `generator`/`fact_clause_used`/
`wkv_mouth_used` from the ACTUAL producer, gated on a new `wkv_attempted` flag (not the old `wkv_used`) so the
separate fact-clause-fallback block never double-renders the same fact.

WHAT THIS RUNNER MEASURES, end-to-end, no mocked mechanism (the cheap unit-level pin lives in
tests/test_generator_trace_matches_producer.py -- this runner is the JSON-artifact companion, mirroring the
SAME scenarios, for provenance):
  1. The bug scenario itself, 6-seed: BRAIN_WKV_MOUTH_SCOPE=broad + a known topic whose relation
     RELATION_LEXICON covers traces "spiking_clause", not "wkv_mouth".
  2. The SAME bug under the DEFAULT scope='vocab', for a message that also passes the narrow in_vocab_scope
     gate (the ~3% real-traffic residual `wkv_fact_sentence_enabled` names).
  3. The pre-existing-correct cell (regression guard): scope='vocab' + an out-of-vocab known topic, which never
     enters the WKV try-block at all.
  4. A genuine free-gen turn (known topic, uncovered relation) still traces "wkv_mouth" under both scope
     settings -- the fix must not over-correct.
  5. An out-of-vocab + unknown-topic turn still degrades to "qwen" (a deterministic fake-Qwen stub replaces the
     real off-bridge Qwen-0.5B load).
  6. Byte-identical reply content: the SAME known+covered-relation turn under both scope settings returns
     IDENTICAL `raw`/`answer` text -- only the internal route and the (now correct, shared) label differ.

Run: SIM_BACKEND=numpy .venv/bin/python -m research.runners._generator_trace_mislabel_fix_verify
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

from webapp import open_ended_chat as OEC  # noqa: E402
from webapp import wkv_mouth_generator as WKV  # noqa: E402
from research.runners._wkv_fact_to_sentence_lexicon_lever import RELATION_LEXICON, expected_surface  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
SEEDS = (42, 43, 44, 100, 101, 102)

_COVERED_ACTION = "employer"
_UNCOVERED_ACTION = "zzz_totally_uncovered_relation_xyz"
_OOV_TOPIC = "isaac_asimov"
_OOV_KNOWN_MSG = f"tell me about {_OOV_TOPIC}"
_IN_VOCAB_TOPIC = "dog and the cat"
_IN_VOCAB_KNOWN_MSG = "tell me about the dog and the cat"
_UNKNOWN_IN_VOCAB_MSG = "the dog and the cat had a story"
_OOV_UNKNOWN_MSG = "what do you think about quantum chromodynamics superconductivity"


class _FakeQwenGenerator:
    """Drop-in replacement for `OpenEndedGenerator` -- avoids loading the real off-bridge Qwen-0.5B, mirroring
    `research/runners/_open_ended_qwen_fact_clause_fallback_verify.py::_FakeQwenGenerator`."""

    def __init__(self):
        self.calls = 0

    def generate(self, system, user, seed=42, max_new_tokens=None):
        self.calls += 1
        return "FAKE QWEN REPLY", 0.01


def _make_bundle(tmp_dir: Path, facts: list) -> str:
    d = tmp_dir
    d.mkdir(parents=True, exist_ok=True)
    (d / "facts.json").write_text(json.dumps({"schema_version": 1, "facts": facts}), encoding="utf-8")
    return str(d)


def _scoped_env(**kv):
    old = {k: os.environ.get(k) for k in kv}

    class _Ctx:
        def __enter__(self):
            for k, v in kv.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            return self

        def __exit__(self, *a):
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    return _Ctx()


def main() -> dict:
    assert _COVERED_ACTION in RELATION_LEXICON
    assert _UNCOVERED_ACTION not in RELATION_LEXICON
    tmp_root = REPO_ROOT / "research" / "findings" / "raw" / "_generator_trace_mislabel_fix_verify_bundles"

    out = {"runner": "_generator_trace_mislabel_fix_verify", "seeds": list(SEEDS)}
    rows = []

    # 1. THE BUG ITSELF, 6-seed: scope=broad + known + covered relation -> "spiking_clause".
    bundle1 = _make_bundle(tmp_root / "case1",
                           [{"agent": _OOV_TOPIC, "action": _COVERED_ACTION,
                             "patient": "university_of_boston", "polarity": "AFFIRM"}])
    case1_rows = []
    with _scoped_env(BRAIN_WKV_MOUTH_SCOPE="broad"):
        for seed in SEEDS:
            res = OEC.answer_turn(_OOV_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle1,
                                  brain_bundle=None, seed=seed, max_new_tokens=25)
            exp, covered, _n = expected_surface(_OOV_TOPIC, _COVERED_ACTION, "university_of_boston")
            case1_rows.append({
                "seed": seed, "generator": res["generator"], "fact_clause_used": res["fact_clause_used"],
                "wkv_mouth_used": res["wkv_mouth_used"], "raw": res["raw"], "expected": exp,
                "raw_matches_expected": res["raw"] == exp,
                "ok": (res["generator"] == "spiking_clause" and res["fact_clause_used"] is True
                      and res["wkv_mouth_used"] is False and res["raw"] == exp),
            })
    rows.append({"case": "1_broad_scope_covered_relation_6seed", "rows": case1_rows,
                "all_ok": all(r["ok"] for r in case1_rows)})

    # 2. Same bug, default scope='vocab', in-vocab message.
    bundle2 = _make_bundle(tmp_root / "case2",
                           [{"agent": _IN_VOCAB_TOPIC, "action": _COVERED_ACTION,
                             "patient": "university_of_boston", "polarity": "AFFIRM"}])
    with _scoped_env(BRAIN_WKV_MOUTH_SCOPE=None):
        in_vocab = WKV.in_vocab_scope(_IN_VOCAB_KNOWN_MSG, seed=42)
        res2 = OEC.answer_turn(_IN_VOCAB_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle2,
                               brain_bundle=None, seed=42, max_new_tokens=25)
    case2 = {"case": "2_vocab_scope_in_vocab_covered_relation", "in_vocab_scope": in_vocab,
            "generator": res2["generator"], "fact_clause_used": res2["fact_clause_used"],
            "wkv_mouth_used": res2["wkv_mouth_used"],
            "ok": bool(in_vocab and res2["generator"] == "spiking_clause" and res2["fact_clause_used"] is True
                      and res2["wkv_mouth_used"] is False)}
    rows.append(case2)

    # 3. Pre-existing-correct cell: scope='vocab' + out-of-vocab known topic -> outer fallback, "spiking_clause".
    with _scoped_env(BRAIN_WKV_MOUTH_SCOPE=None):
        oov_scope = WKV.in_vocab_scope(_OOV_KNOWN_MSG, seed=42)
        res3 = OEC.answer_turn(_OOV_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle1,
                               brain_bundle=None, seed=42, max_new_tokens=25)
    case3 = {"case": "3_vocab_scope_out_of_vocab_covered_relation_preexisting_correct",
            "in_vocab_scope": oov_scope, "generator": res3["generator"],
            "fact_clause_used": res3["fact_clause_used"], "wkv_mouth_used": res3["wkv_mouth_used"],
            "ok": bool((not oov_scope) and res3["generator"] == "spiking_clause"
                      and res3["fact_clause_used"] is True and res3["wkv_mouth_used"] is False)}
    rows.append(case3)

    # 4. Genuine free-gen (uncovered relation), both scope settings -> "wkv_mouth", not an over-correction.
    bundle4 = _make_bundle(tmp_root / "case4",
                           [{"agent": _IN_VOCAB_TOPIC, "action": _UNCOVERED_ACTION,
                             "patient": "something", "polarity": "AFFIRM"}])
    case4_rows = []
    for scope in ("broad", "vocab"):
        with _scoped_env(BRAIN_WKV_MOUTH_SCOPE=(scope if scope == "broad" else None)):
            res4 = OEC.answer_turn(_IN_VOCAB_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle4,
                                   brain_bundle=None, seed=42, max_new_tokens=25)
        case4_rows.append({"scope": scope, "generator": res4["generator"],
                           "wkv_mouth_used": res4["wkv_mouth_used"], "fact_clause_used": res4["fact_clause_used"],
                           "ok": bool(res4["generator"] == "wkv_mouth" and res4["wkv_mouth_used"] is True
                                     and res4["fact_clause_used"] is False)})
    rows.append({"case": "4_uncovered_relation_genuine_free_gen", "rows": case4_rows,
                "all_ok": all(r["ok"] for r in case4_rows)})

    # 5. unknown topic, in-vocab prompt -> "wkv_mouth" (no facts at all -> sentence_facts never even passed).
    res5 = OEC.answer_turn(_UNKNOWN_IN_VOCAB_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=None,
                           brain_bundle=None, seed=42, max_new_tokens=25)
    case5 = {"case": "5_unknown_topic_in_vocab", "known": res5["known"], "generator": res5["generator"],
            "ok": bool(res5["known"] is False and res5["generator"] == "wkv_mouth"
                      and res5["wkv_mouth_used"] is True and res5["fact_clause_used"] is False)}
    rows.append(case5)

    # 6. out-of-vocab + unknown -> qwen fallback (deterministic stub), unaffected by this fix.
    fake = _FakeQwenGenerator()
    old_get_gen = OEC.get_generator
    OEC.get_generator = lambda warm_faculty: fake
    try:
        oov_unk_scope = WKV.in_vocab_scope(_OOV_UNKNOWN_MSG, seed=42)
        res6 = OEC.answer_turn(_OOV_UNKNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=None,
                               brain_bundle=None, seed=42, max_new_tokens=25)
    finally:
        OEC.get_generator = old_get_gen
    case6 = {"case": "6_out_of_vocab_unknown_qwen_fallback", "in_vocab_scope": oov_unk_scope,
            "known": res6["known"], "generator": res6["generator"], "qwen_stub_fired": fake.calls,
            "ok": bool((not oov_unk_scope) and res6["known"] is False and res6["generator"] == "qwen"
                      and res6["wkv_mouth_used"] is False and res6["fact_clause_used"] is False
                      and fake.calls == 1)}
    rows.append(case6)

    # 7. Byte-identical reply content: same fact, both scope settings -> IDENTICAL raw/answer text.
    with _scoped_env(BRAIN_WKV_MOUTH_SCOPE=None):
        res_vocab = OEC.answer_turn(_OOV_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle1,
                                    brain_bundle=None, seed=42, max_new_tokens=25)
    with _scoped_env(BRAIN_WKV_MOUTH_SCOPE="broad"):
        res_broad = OEC.answer_turn(_OOV_KNOWN_MSG, None, valence=0.0, arousal=0.5, ltm_bundle=bundle1,
                                    brain_bundle=None, seed=42, max_new_tokens=25)
    case7 = {"case": "7_byte_identical_reply_content_across_scope", "raw_vocab": res_vocab["raw"],
            "raw_broad": res_broad["raw"], "answer_vocab": res_vocab["answer"], "answer_broad": res_broad["answer"],
            "generator_vocab": res_vocab["generator"], "generator_broad": res_broad["generator"],
            "ok": bool(res_vocab["raw"] == res_broad["raw"] and res_vocab["answer"] == res_broad["answer"]
                      and res_vocab["generator"] == res_broad["generator"] == "spiking_clause")}
    rows.append(case7)

    out["rows"] = rows
    all_ok = {r["case"]: r.get("all_ok", r.get("ok")) for r in rows}
    out["all_ok_by_case"] = all_ok

    v = Verdict("generator-trace-mislabel fix: label follows the actual producer, reply content unchanged")
    v.require("case 1 -- broad-scope covered-relation traces spiking_clause, 6/6 seeds",
              all_ok["1_broad_scope_covered_relation_6seed"], expect=True)
    v.require("case 2 -- vocab-scope in-vocab covered-relation ALSO traces spiking_clause (the ~3% residual)",
              all_ok["2_vocab_scope_in_vocab_covered_relation"], expect=True)
    v.require("case 3 -- pre-existing-correct cell unaffected (vocab-scope out-of-vocab fallback)",
              all_ok["3_vocab_scope_out_of_vocab_covered_relation_preexisting_correct"], expect=True)
    v.require("case 4 -- uncovered-relation genuine free-gen still traces wkv_mouth (no over-correction)",
              all_ok["4_uncovered_relation_genuine_free_gen"], expect=True)
    v.require("case 5 -- unknown-topic in-vocab traces wkv_mouth", all_ok["5_unknown_topic_in_vocab"], expect=True)
    v.require("case 6 -- out-of-vocab+unknown degrades to qwen, unaffected",
              all_ok["6_out_of_vocab_unknown_qwen_fallback"], expect=True)
    v.require("case 7 -- reply content byte-identical across scope for the same fact",
              all_ok["7_byte_identical_reply_content_across_scope"], expect=True)
    go = all(all_ok.values())
    verdict = v.decide(go=go)
    out["verdict"] = verdict
    out["verdict_preconditions"] = v.to_dict()
    print(json.dumps({k: v_ for k, v_ in all_ok.items()}, indent=2))
    print(f"\nVERDICT: {verdict}")
    return out


if __name__ == "__main__":
    result = main()
    out_path = REPO_ROOT / "research/findings/raw/_generator_trace_mislabel_fix_verify.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
    sys.exit(0 if result["verdict"] == "GO" else 1)
