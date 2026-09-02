"""Board #112 residual verify (2026-09-02, branch `research/open-ended-oov-grounding-fix`): 6-seed, end-to-end
verification of `BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK` through the REAL `webapp.open_ended_chat.answer_turn()`
entry point -- NOT a parallel harness.

THE RESIDUAL THIS CLOSES. The 2026-09-01 wire-in (`BRAIN_OPEN_ENDED_WKV_MOUTH_FACT_SENTENCE`,
research/findings/2026-09-01-wkv-mouth-fact-sentence-wirein.md) made `render_fact_sentence` reachable from the
WKV mouth's own `generate()`, but ONLY for the ~3% of real known topics that also pass the WKV checkpoint's
`in_vocab_scope` free-gen gate -- that finding's own SS5 named, and explicitly left untouched, "the much larger
Qwen-routed (out-of-vocab) known-topic grounding regression"
(research/findings/2026-09-01-open-ended-bundle-moat-safety-soak-fabrication-delta.md). That soak found: the
retrieved facts ARE assembled into Qwen's prompt (`build_prompt`'s KNOWLEDGE block + "use ONLY the facts"
instruction) -- diagnosis (a)/(b) (never retrieved / never injected) are both FALSE -- but a pretrained Qwen does
not reliably obey the instruction: it supplements with confident wrong parametric detail (`castleford_f_c` ->
"a professional football club" when the store's only sport fact is `rugby_leauge`) or ignores the facts
entirely -- diagnosis (c), retrieved+injected-but-overridden. The existing post-hoc moat (base `post_filter`,
`NP_ENTAILMENT`, `GEN_TIME_HONESTY`) can only SUBTRACT a sentence it catches as wrong, and on real Qwen prose
(copula/participial/pronoun-heavy) often cannot even PARSE the wrong clause to catch it (that soak's own
measurement: `NP_ENTAILMENT` changed ZERO of 12 real known-topic replies).

THE FIX measured here: `webapp/open_ended_chat.py::answer_turn` now tries the SAME already-6-seed-GO
`render_fact_sentence` (`webapp.wkv_mouth_generator`, unmodified) on ANY known topic the WKV mouth did not
already handle -- independent of `in_vocab_scope` (that gate only scopes the WKV checkpoint's OWN word-level
free-gen decode; the clause render uses its own closed-class `RELATION_LEXICON`/`slug_to_np` + the
SpikingClauseProducer, a structurally different, vocabulary-independent mechanism). A hit means Qwen is
bypassed for the turn entirely: the reply is GROUNDED (attributable to the actual retrieved fact) rather than
merely not-yet-caught-as-wrong.

METHOD. Samples real `wikidata_core_15k` agents that (a) are known to the store, (b) FAIL `in_vocab_scope` for
the WKV mouth (the real-traffic Qwen-routed slice this residual is about -- ~97% of real topics per the 2026-09-01
scan), and (c) have >=1 fact whose relation the lexicon covers. For each case, `answer_turn` is called through
TWO scoped env-var arms (flag on/off) with the server's `get_generator` monkey-patched to a deterministic FAKE
Qwen stub (records whether it fired, returns a canned confident-but-unsupported sentence mimicking the soak's own
observed fabrication pattern) -- this isolates the ROUTING/mechanism change from Qwen's actual weights/latency,
the SAME isolation strategy `_wkv_mouth_open_ended_wiring_verify.py` used for the sibling WKV-mouth wiring. A
poison-pill on `wkv_mouth_generator.render_fact_sentence` additionally proves the new path is NEVER imported when
the flag is off. Scored with the SAME independent parser (`expected_surface`/`parse_and_score`) the lexicon
lever's own GO used -- ground-truth reuse, not producer-internal trust.

Run: SIM_BACKEND=numpy .venv/bin/python -m research.runners._open_ended_qwen_fact_clause_fallback_verify
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from webapp import open_ended_chat as OEC  # noqa: E402
from webapp import wkv_mouth_generator as WKV  # noqa: E402
from research.runners._wkv_fact_svo_clause_first_lever import _bundle_dir  # noqa: E402
from research.runners._wkv_fact_to_sentence_lexicon_lever import (  # noqa: E402
    expected_surface, parse_and_score,
)
from tools.verdict import Verdict  # noqa: E402
from tools.lab import attributable_to  # noqa: E402

REPO_ROOT = _REPO
SEEDS = (42, 43, 44, 100, 101, 102)
_WORD_RE = WKV._WORD_RE
_FUNCTION_WORDS = WKV._FUNCTION_WORDS
FLAG = "BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK"

# a deterministic, canned "Qwen fabrication" -- mirrors the soak's own observed failure mode (a confident,
# specific, wrong supplement) without needing the real off-bridge Qwen-0.5B (heavy: torch/CUDA model load).
# Never contains any real store content word, so `specificity`-style overlap checks below read cleanly as 0.
_FAKE_QWEN_TEXT = ("This is a professional football club with a long and proud history in the top division, "
                   "known for its passionate supporters and a modern stadium on the edge of town.")


class _FakeQwenGenerator:
    """Drop-in replacement for `OpenEndedGenerator` -- records every `.generate()` call so a test can assert
    whether the Qwen path fired, without loading the real off-bridge Qwen-0.5B (heavy torch/CUDA model)."""

    def __init__(self, call_log: list):
        self._log = call_log

    def generate(self, system, user, seed=42, max_new_tokens=None):
        self._log.append({"seed": seed, "max_new_tokens": max_new_tokens})
        return _FAKE_QWEN_TEXT, 0.01


class _PoisonPillRenderFactSentence:
    """Raises if called -- substituted for `wkv_mouth_generator.render_fact_sentence` to PROVE the new fallback
    branch never invokes it when the flag is off (the same poison-pill discipline
    `_wkv_mouth_open_ended_wiring_verify.py` used for the sibling WKV-mouth wire-in)."""

    def __call__(self, *a, **kw):
        raise AssertionError("render_fact_sentence must not be called when BRAIN_OPEN_ENDED_FACT_CLAUSE_FALLBACK "
                             "is off")


def _sample_known_oov_covered(by_agent: dict, seed: int, n: int = 8, scan: int = 2000) -> list:
    """Real store agents that (a) are known, (b) FAIL `in_vocab_scope` (the real-traffic Qwen-routed slice this
    residual is about), and (c) have >=1 fact whose relation the lexicon covers -- the mirror-image sampling
    condition of the sibling wire-in verify's `_sample_known_invocab_covered`."""
    agents = list(by_agent.keys())
    rng = random.Random(seed)
    rng.shuffle(agents)
    picked = []
    for a in agents[:scan]:
        msg = f"tell me about {a}"
        if WKV.in_vocab_scope(msg, seed=seed):
            continue                                    # would route through the WKV mouth -- not this residual
        facts = OEC.retrieve(by_agent, OEC.extract_topic(msg))
        triple = WKV.pick_covered_fact(facts)
        if triple is None:
            continue
        picked.append({"agent": a, "msg": msg, "facts": facts, "triple": triple})
        if len(picked) >= n:
            break
    return picked


def _content_words(triple) -> set:
    agent, action, patient = triple
    words = set()
    for field in (agent, action, patient):
        for w in _WORD_RE.findall(str(field)):
            wl = w.lower()
            if wl not in _FUNCTION_WORDS:
                words.add(wl)
    return words


def _content_hit_frac(text: str, words: set):
    if not words:
        return None
    toks = set(w.lower() for w in _WORD_RE.findall(text or ""))
    hits = sum(1 for w in words if w in toks)
    return hits / len(words)


def _run_case(msg: str, seed: int, flag_value: str, bundle: str, poison: bool = False) -> dict:
    """Call the REAL `answer_turn` with (1) a scoped env-var override for the new flag (restored in `finally`)
    and (2) `webapp.open_ended_chat.get_generator` monkey-patched to the deterministic fake-Qwen stub (also
    restored). `poison=True` additionally substitutes a poison-pill for `render_fact_sentence` on
    `webapp.wkv_mouth_generator` -- used only on flag-OFF cases, to prove the new branch never imports/calls it."""
    old_flag = os.environ.get(FLAG)
    call_log: list = []
    old_get_gen = OEC.get_generator
    old_render = WKV.render_fact_sentence
    OEC.get_generator = lambda warm_faculty: _FakeQwenGenerator(call_log)
    if poison:
        WKV.render_fact_sentence = _PoisonPillRenderFactSentence()
    try:
        os.environ[FLAG] = flag_value
        res = OEC.answer_turn(msg, None, valence=0.0, arousal=0.5, ltm_bundle=bundle, brain_bundle=None, seed=seed)
        res["_qwen_stub_fired"] = bool(call_log)
        return res
    finally:
        OEC.get_generator = old_get_gen
        WKV.render_fact_sentence = old_render
        if old_flag is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = old_flag


def main() -> dict:
    out: dict = {"runner": "_open_ended_qwen_fact_clause_fallback_verify", "seeds": list(SEEDS)}
    bundle = _bundle_dir()
    out["bundle_dir"] = bundle
    if bundle is None:
        out["skipped"] = "no data lake (sim-data/knowledge_bundles/wikidata_core_15k not found)"
        print(json.dumps(out, indent=2))
        return out
    by_agent = OEC.build_index(bundle, None)
    out["n_agents_indexed"] = len(by_agent)

    per_seed = []
    for seed in SEEDS:
        cases = _sample_known_oov_covered(by_agent, seed, n=8)
        rows = []
        for c in cases:
            agent, action, patient = c["triple"]
            exp, covered, _n = expected_surface(agent, action, patient)

            res_on = _run_case(c["msg"], seed, "1", bundle)
            res_off = _run_case(c["msg"], seed, "0", bundle, poison=True)

            score_raw = parse_and_score(res_on["raw"], agent, action, patient)
            score_answer = parse_and_score(res_on["answer"], agent, action, patient)
            answer_is_safe_degrade = bool(
                res_on["answer"] == res_on["raw"]
                or res_on["answer"] == OEC._empty_known_fallback(OEC.extract_topic(c["msg"])))

            cwords = _content_words(c["triple"])
            rows.append({
                "agent": agent, "action": action, "patient": patient,
                "expected_surface": exp, "covered": covered,
                # flag ON -- the new mechanism's own claim (raw) + the end-to-end post-filtered claim:
                "on_raw": res_on["raw"], "on_answer": res_on["answer"],
                "on_generator": res_on["generator"], "on_fact_clause_used": bool(res_on["fact_clause_used"]),
                "on_wkv_used": bool(res_on["wkv_mouth_used"]), "on_known": bool(res_on["known"]),
                "on_qwen_stub_fired": bool(res_on["_qwen_stub_fired"]),
                "raw_readable": score_raw["readable"], "raw_faithful": score_raw["faithful"],
                "raw_moat_safe": score_raw["moat_safe"],
                "answer_readable": score_answer["readable"], "answer_faithful": score_answer["faithful"],
                "answer_moat_safe": score_answer["moat_safe"],
                "answer_is_safe_degrade": answer_is_safe_degrade,
                # flag OFF -- byte-identical-off: must reproduce the PRE-EXISTING Qwen-routed path exactly.
                "off_generator": res_off["generator"], "off_fact_clause_used": bool(res_off["fact_clause_used"]),
                "off_qwen_stub_fired": bool(res_off["_qwen_stub_fired"]),
                "off_answer": res_off["answer"], "off_raw_is_fake_qwen": res_off["raw"] == _FAKE_QWEN_TEXT,
                "off_content_hit_frac": _content_hit_frac(res_off["answer"], cwords),
                "off_is_clause": res_off["answer"] == exp,
            })

        # unknown-topic honesty check (moat intact): a made-up topic, not a store agent -- must still route to
        # (the fake) Qwen + hedge/abstain via post_filter, unaffected by the new flag (known=False short-circuits
        # the new branch regardless of its state) -- run with the flag ON to prove no regression on this class.
        unk_msg = "tell me about the zorvennian quiblex artifact"
        res_unk = _run_case(unk_msg, seed, "1", bundle)
        unk_abstained = any(p in res_unk["answer"].lower() for p in
                            ("not sure", "don't have", "guessing", "i don't", "unsure", "no information"))

        n = len(rows)

        def frac(key):
            vals = [r[key] for r in rows if isinstance(r.get(key), bool)]
            return float(np.mean(vals)) if vals else None

        row = {
            "seed": seed, "n_cases": n,
            "raw_readable_frac": frac("raw_readable"), "raw_faithful_frac": frac("raw_faithful"),
            "raw_moat_safe_frac": frac("raw_moat_safe"),
            "answer_moat_safe_frac": frac("answer_moat_safe"),
            "answer_is_safe_degrade_frac": frac("answer_is_safe_degrade"),
            "on_fact_clause_used_frac": frac("on_fact_clause_used"),
            "on_qwen_stub_fired_frac": frac("on_qwen_stub_fired"),
            "on_known_frac": frac("on_known"),
            "off_fact_clause_used_frac": frac("off_fact_clause_used"),
            "off_qwen_stub_fired_frac": frac("off_qwen_stub_fired"),
            "off_is_clause_frac": frac("off_is_clause"),
            "off_content_hit_frac_mean": (float(np.mean([r["off_content_hit_frac"] for r in rows
                                                          if r["off_content_hit_frac"] is not None]))
                                          if rows else None),
            "unknown_topic_known": bool(res_unk["known"]),
            "unknown_topic_answer": res_unk["answer"],
            "unknown_topic_fact_clause_used": bool(res_unk["fact_clause_used"]),
            "unknown_topic_qwen_stub_fired": bool(res_unk["_qwen_stub_fired"]),
            "unknown_topic_abstained": bool(unk_abstained),
            "rows": rows,
        }
        per_seed.append(row)
        print(f"[seed {seed}] n={n} raw_readable={row['raw_readable_frac']} raw_faithful={row['raw_faithful_frac']} "
              f"raw_moat_safe={row['raw_moat_safe_frac']} | on_fact_clause_used={row['on_fact_clause_used_frac']} "
              f"on_qwen_fired={row['on_qwen_stub_fired_frac']} | off_qwen_fired={row['off_qwen_stub_fired_frac']} "
              f"off_fact_clause_used={row['off_fact_clause_used_frac']} | "
              f"unknown_abstained={row['unknown_topic_abstained']}")
        for r in rows[:3]:
            print(f"    {r['agent']:35s} {r['action']:25s} on_raw={r['on_raw']!r}")
            print(f"    {'':35s} {'':25s} off_answer={r['off_answer']!r}")

    out["per_seed"] = per_seed

    def agg(key):
        vals = [s[key] for s in per_seed if s[key] is not None]
        return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4)} if vals else None

    out["aggregate"] = {
        "n_seeds": len(per_seed), "n_cases_total": sum(s["n_cases"] for s in per_seed),
        "raw_readable": agg("raw_readable_frac"), "raw_faithful": agg("raw_faithful_frac"),
        "raw_moat_safe": agg("raw_moat_safe_frac"), "answer_moat_safe": agg("answer_moat_safe_frac"),
        "answer_is_safe_degrade": agg("answer_is_safe_degrade_frac"),
        "on_fact_clause_used": agg("on_fact_clause_used_frac"), "on_qwen_stub_fired": agg("on_qwen_stub_fired_frac"),
        "on_known": agg("on_known_frac"),
        "off_fact_clause_used": agg("off_fact_clause_used_frac"),
        "off_qwen_stub_fired": agg("off_qwen_stub_fired_frac"),
        "off_is_clause": agg("off_is_clause_frac"), "off_content_hit_frac_mean": agg("off_content_hit_frac_mean"),
        "all_seeds_min_1_case": bool(all(s["n_cases"] >= 1 for s in per_seed)),
        "all_unknown_abstained": bool(all(s["unknown_topic_abstained"] for s in per_seed)),
        "all_unknown_qwen_fired": bool(all(s["unknown_topic_qwen_stub_fired"] for s in per_seed)),
        "all_unknown_fact_clause_never_used": bool(all(not s["unknown_topic_fact_clause_used"]
                                                        for s in per_seed)),
    }

    a = out["aggregate"]
    v = Verdict("Qwen-routed known-topic fact-clause fallback (board #112 residual): 6-seed end-to-end verify "
               "through the real answer_turn path")
    v.require("every seed found >=1 real known+out-of-vocab+covered-relation case",
              a["all_seeds_min_1_case"], expect=True)
    v.require("flag ON reaches the fact-clause fallback on every found case (Qwen-routed, previously fabricating)",
              a["on_fact_clause_used"]["min"] if a["on_fact_clause_used"] else None,
              expect=lambda x: x is not None and x >= 0.999)
    v.require("flag ON: the fake-Qwen stub NEVER fires when the fact-clause fallback handled the turn",
              a["on_qwen_stub_fired"]["mean"] if a["on_qwen_stub_fired"] else None,
              expect=lambda x: x is not None and x <= 0.001)
    v.require("RAW readable (coherent-clause) rate >= 0.95 on every seed",
              a["raw_readable"]["min"] if a["raw_readable"] else None, expect=lambda x: x is not None and x >= 0.95)
    v.require("RAW faithful rate >= 0.95 on every seed",
              a["raw_faithful"]["min"] if a["raw_faithful"] else None, expect=lambda x: x is not None and x >= 0.95)
    v.require("RAW moat-safe on every seed",
              a["raw_moat_safe"]["min"] if a["raw_moat_safe"] else None,
              expect=lambda x: x is not None and x >= 0.999)
    v.require("post-filtered answer is always either the raw clause or the honest fallback (never corrupted)",
              a["answer_is_safe_degrade"]["min"] if a["answer_is_safe_degrade"] else None,
              expect=lambda x: x is not None and x >= 0.999)
    v.require("flag OFF: the fact-clause fallback NEVER fires (poison-pill on render_fact_sentence never trips, "
              "generator stays 'qwen', fact_clause_used stays False on every case)",
              a["off_fact_clause_used"]["mean"] if a["off_fact_clause_used"] else None,
              expect=lambda x: x is not None and x <= 0.001)
    v.require("flag OFF: the fake-Qwen stub fires on EVERY case (byte-identical-off routing -- the pre-existing "
              "Qwen-routed path runs exactly as before this change)",
              a["off_qwen_stub_fired"]["min"] if a["off_qwen_stub_fired"] else None,
              expect=lambda x: x is not None and x >= 0.999)
    v.require("unknown-topic honesty (hedge/abstain) intact on every seed with the flag ON (no regression)",
              a["all_unknown_abstained"], expect=True)
    v.require("unknown-topic turn still routes to Qwen (known=False short-circuits the new branch regardless "
              "of flag state)", a["all_unknown_qwen_fired"], expect=True)
    v.require("unknown-topic turn never uses the fact-clause fallback (no over-reach past known topics)",
              a["all_unknown_fact_clause_never_used"], expect=True)
    v.control("does the reply preserve the fact -- exact-clause rate, fact-clause-fallback (raw) vs the "
              "pre-existing flags-off Qwen-routed path's own exact-clause rate",
              treatment=a["raw_faithful"]["mean"] if a["raw_faithful"] else None,
              control=a["off_is_clause"]["mean"] if a["off_is_clause"] else None)
    frac_attributable = attributable_to(
        "exact-clause fidelity, fact-clause-fallback (raw) vs the pre-existing flags-off Qwen-routed free-gen path",
        treatment_value=a["raw_faithful"]["mean"] if a["raw_faithful"] else None,
        control_value=a["off_is_clause"]["mean"] if a["off_is_clause"] else None)
    a["exact_clause_gain_fraction_attributable_to_fix"] = frac_attributable

    go = bool(
        a["all_seeds_min_1_case"]
        and (a["on_fact_clause_used"]["min"] if a["on_fact_clause_used"] else 0) >= 0.999
        and (a["on_qwen_stub_fired"]["mean"] if a["on_qwen_stub_fired"] else 1) <= 0.001
        and (a["raw_readable"]["min"] if a["raw_readable"] else 0) >= 0.95
        and (a["raw_faithful"]["min"] if a["raw_faithful"] else 0) >= 0.95
        and (a["raw_moat_safe"]["min"] if a["raw_moat_safe"] else 0) >= 0.999
        and (a["answer_is_safe_degrade"]["min"] if a["answer_is_safe_degrade"] else 0) >= 0.999
        and (a["off_fact_clause_used"]["mean"] if a["off_fact_clause_used"] else 1) <= 0.001
        and (a["off_qwen_stub_fired"]["min"] if a["off_qwen_stub_fired"] else 0) >= 0.999
        and a["all_unknown_abstained"]
        and a["all_unknown_qwen_fired"]
        and a["all_unknown_fact_clause_never_used"]
    )
    verdict = v.decide(go=go)
    out["verdict"] = verdict
    out["verdict_preconditions"] = v.to_dict()
    print(f"\nVERDICT: {verdict}")
    return out


if __name__ == "__main__":
    result = main()
    out_path = REPO_ROOT / "research/findings/raw/_open_ended_qwen_fact_clause_fallback_verify.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {out_path}")
