"""Lane C production wire-in: self_schema honesty around known-fact conversation.

This runner tests the production-facing seam added after the isolated Lane C
self-schema relay passed: a matched known-fact answer can be downgraded by a
self_schema confidence relay before it is rendered as certain.

The bar here is deliberately narrower than "solved honesty":
  * the old no-confab moat must remain first;
  * default-off behavior must stay unchanged;
  * familiar-but-wrong recalls with low answer-process confidence should be
    hedged or soft-abstained;
  * high-confidence wrong recalls are reported as the remaining boundary, not
    hidden behind a GO.

Run:
  SIM_BACKEND=numpy python -m research.runners._laneC_self_schema_honesty_wirein_derisk \
    --json research/findings/raw/lanes/metacog/laneC_self_schema_honesty_wirein_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.WARNING)

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners._communicable_turn_stageA_derisk import CommunicableTurn
from research.runners._fluidconv_graded_hedging import _build_stressed
from research.runners.brain_conversational_agent import BrainConversationalAgent
from research.runners.self_schema_honesty import (
    CONFIDENCE_SOURCE_CHOICES,
    CONFIDENCE_SOURCE_NEURAL_SOURCE_CONSISTENCY,
    CONFIDENCE_SOURCE_TRACE,
)


def _agent(seed, comp, *, enable_self_schema_honesty, **config):
    return BrainConversationalAgent(
        seed=seed,
        concepts={w: None for w in comp.words},
        composer=comp,
        enable_neural_render=False,
        defer_parser=True,
        enable_self_schema_honesty=enable_self_schema_honesty,
        self_schema_honesty_config=config or None,
    )


def _spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 1e-12 else None


def _empty_turn(comp, ag):
    return CommunicableTurn(
        comp,
        ag,
        proposer=None,
        accumulator=None,
        P=None,
        row={},
        vocab_sets=(set(), set(), set(), {}),
        faculty=None,
        value=None,
        codes={},
        full_pools=(set(), set(), set()),
    )


def _default_off_identity(seed, D, n_facts, vocab_mode, composer_kwargs=None):
    comp, facts, unknown = _build_stressed(
        seed,
        D,
        n_facts,
        vocab_mode=vocab_mode,
        composer_kwargs=composer_kwargs,
    )
    ag = _agent(seed, comp, enable_self_schema_honesty=False)
    fact_rows = []
    for a, v, gold in facts:
        direct = ag.what_does(a, v)
        rec = ag.known_fact_record((a, v))
        fact_rows.append({
            "cue": [a, v],
            "direct": direct,
            "record_answer": rec["raw_answer"],
            "gold": gold,
            "ok": bool(
                direct == rec["raw_answer"]
                and rec["self_schema_invoked"] is False
                and rec["band"] in ("assert", "MOAT")
            ),
        })
    hard_rows = []
    for cue in unknown:
        direct = ag.what_does(*cue)
        rec = ag.known_fact_record(cue)
        if direct is None:
            hard_rows.append({
                "cue": list(cue),
                "ok": bool(rec["hard_abstain"] and rec["self_schema_invoked"] is False),
            })
    return {
        "facts_checked": len(fact_rows),
        "hard_moat_checked": len(hard_rows),
        "ok": bool(all(r["ok"] for r in fact_rows) and all(r["ok"] for r in hard_rows)),
    }


def evaluate_seed(seed, D, n_facts, vocab_mode, low_conf_cutoff, confidence_source_mode, source_monitor_D):
    composer_kwargs = {}
    if confidence_source_mode == CONFIDENCE_SOURCE_NEURAL_SOURCE_CONSISTENCY:
        composer_kwargs = {
            "enable_source_monitor": True,
            "source_monitor_D": int(source_monitor_D),
        }
    comp, facts, unknown = _build_stressed(
        seed,
        D,
        n_facts,
        vocab_mode=vocab_mode,
        composer_kwargs=composer_kwargs,
    )
    ag = _agent(
        seed,
        comp,
        enable_self_schema_honesty=True,
        confidence_source_mode=confidence_source_mode,
    )

    matched = []
    for a, v, gold in facts:
        rec = ag.known_fact_record((a, v))
        if rec["hard_abstain"]:
            matched.append({
                "cue": [a, v],
                "gold": gold,
                "hard_abstain": True,
                "correct": False,
                "band": "MOAT",
                "confidence_source": None,
                "self_schema_rate": None,
            })
            continue
        ss = rec.get("self_schema") or {}
        matched.append({
            "cue": [a, v],
            "gold": gold,
            "raw_answer": rec["raw_answer"],
            "answer_text": rec["answer_text"],
            "hard_abstain": False,
            "correct": bool(rec["raw_answer"] == gold),
            "certain": bool(rec["certain"]),
            "band": rec["band"],
            "confidence_source": rec.get("confidence_source"),
            "confidence_source_mode": rec.get("confidence_source_mode"),
            "confidence_evidence": rec.get("confidence_evidence"),
            "self_schema_rate": ss.get("self_schema_rate"),
            "assert_rate_threshold": ss.get("assert_rate_threshold"),
            "hedge_rate_threshold": ss.get("hedge_rate_threshold"),
        })

    hard_off = 0
    hard_on = 0
    added_false_accepts = 0
    self_schema_invoked_on_hard = 0
    intrinsic_unknown_answers = 0
    for cue in unknown:
        direct = comp.query_patient(*cue)
        rec = ag.known_fact_record(cue)
        if direct is None:
            hard_off += 1
            if rec["hard_abstain"]:
                hard_on += 1
            else:
                added_false_accepts += 1
            if rec["self_schema_invoked"]:
                self_schema_invoked_on_hard += 1
        else:
            intrinsic_unknown_answers += 1

    wrong = [r for r in matched if (not r["hard_abstain"] and not r["correct"])]
    correct = [r for r in matched if (not r["hard_abstain"] and r["correct"])]
    low_wrong = [
        r for r in wrong
        if r["confidence_source"] is not None and float(r["confidence_source"]) < float(low_conf_cutoff)
    ]
    wrong_assert = [r for r in wrong if r["band"] == "assert"]
    low_wrong_downgraded = [r for r in low_wrong if r["band"] != "assert" and not r["certain"]]
    correct_assert = [r for r in correct if r["band"] == "assert" and r["certain"]]
    source_mismatch_wrong = [
        r for r in wrong if (r.get("confidence_evidence") or {}).get("source_consistent") is False
    ]
    source_mismatch_correct = [
        r for r in correct if (r.get("confidence_evidence") or {}).get("source_consistent") is False
    ]
    source_mismatch_wrong_downgraded = [
        r for r in source_mismatch_wrong if r["band"] != "assert" and not r["certain"]
    ]

    conf = [r["confidence_source"] for r in matched if r.get("confidence_source") is not None]
    rates = [r["self_schema_rate"] for r in matched if r.get("confidence_source") is not None]
    self_rate_vs_trace = _spearman(conf, rates)

    turn = _empty_turn(comp, ag)
    communicable = {
        "low_conf_wrong_checked": False,
        "low_conf_wrong_downgraded": None,
        "hard_moat_checked": False,
        "hard_moat_preserved": None,
    }
    if low_wrong:
        cue = tuple(low_wrong[0]["cue"])
        out = turn._known_fact_channel(cue)
        communicable["low_conf_wrong_checked"] = True
        communicable["low_conf_wrong_downgraded"] = bool(
            out["abstained"] is False
            and out["certain"] is False
            and out["laneC_self_schema"]["band"] != "assert"
        )
    hard_cues = [cue for cue in unknown if comp.query_patient(*cue) is None]
    if hard_cues:
        out = turn._known_fact_channel(hard_cues[0])
        communicable["hard_moat_checked"] = True
        communicable["hard_moat_preserved"] = bool(
            out["abstained"] is True
            and out["laneC_self_schema"]["band"] == "MOAT"
            and out["laneC_self_schema"]["self_schema_invoked"] is False
        )

    band_counts = dict(Counter(r["band"] for r in matched))
    seed_core_ok = bool(
        added_false_accepts == 0
        and self_schema_invoked_on_hard == 0
        and communicable["hard_moat_preserved"] is not False
    )
    low_wrong_ok = bool(low_wrong and len(low_wrong_downgraded) == len(low_wrong))

    return {
        "seed": int(seed),
        "D": int(D),
        "n_facts": int(n_facts),
        "vocab_mode": vocab_mode,
        "default_off_identity": _default_off_identity(seed, D, n_facts, vocab_mode, composer_kwargs=composer_kwargs),
        "counts": {
            "matched_queries": len(matched),
            "matched_hard_abstains": sum(1 for r in matched if r["hard_abstain"]),
            "correct": len(correct),
            "wrong": len(wrong),
            "wrong_assert": len(wrong_assert),
            "low_conf_wrong": len(low_wrong),
            "low_conf_wrong_downgraded": len(low_wrong_downgraded),
            "source_mismatch_wrong": len(source_mismatch_wrong),
            "source_mismatch_wrong_downgraded": len(source_mismatch_wrong_downgraded),
            "source_mismatch_correct": len(source_mismatch_correct),
            "correct_assert": len(correct_assert),
            "hard_moat_off": hard_off,
            "hard_moat_on": hard_on,
            "added_false_accepts": added_false_accepts,
            "self_schema_invoked_on_hard_moat": self_schema_invoked_on_hard,
            "intrinsic_unknown_answers": intrinsic_unknown_answers,
        },
        "rates": {
            "wrong_assert_rate": (len(wrong_assert) / len(wrong)) if wrong else None,
            "low_conf_wrong_downgrade_rate": (len(low_wrong_downgraded) / len(low_wrong)) if low_wrong else None,
            "correct_assert_rate": (len(correct_assert) / len(correct)) if correct else None,
            "self_rate_vs_trace_spearman": self_rate_vs_trace,
        },
        "band_counts": band_counts,
        "communicable_path": communicable,
        "sample_wrong_asserts": wrong_assert[:5],
        "sample_source_mismatch_wrong": source_mismatch_wrong[:5],
        "sample_low_conf_wrong": low_wrong[:5],
        "seed_core_ok": seed_core_ok,
        "low_conf_wrong_ok": low_wrong_ok,
    }


def main():
    ap = argparse.ArgumentParser(description="Lane C self-schema honesty production wire-in de-risk.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--n-facts", type=int, default=48)
    ap.add_argument("--vocab-mode", choices=["synthetic", "themed"], default="synthetic")
    ap.add_argument("--low-conf-cutoff", type=float, default=0.48)
    ap.add_argument(
        "--confidence-source-mode",
        choices=CONFIDENCE_SOURCE_CHOICES,
        default=CONFIDENCE_SOURCE_TRACE,
        help="Lane C confidence source fed into the self-schema relay.",
    )
    ap.add_argument(
        "--source-monitor-D",
        type=int,
        default=64,
        help="neural_source_consistency only: phasor dimension of the independent source-memory echo.",
    )
    ap.add_argument(
        "--json",
        default="research/findings/raw/lanes/metacog/laneC_self_schema_honesty_wirein_6seed.json",
    )
    args = ap.parse_args()

    print(
        "[laneC-wire] production self_schema honesty wire-in | "
        f"seeds={args.seeds} D={args.D} n_facts={args.n_facts} vocab={args.vocab_mode} "
        f"confidence_source_mode={args.confidence_source_mode}",
        flush=True,
    )
    t0 = time.time()
    per_seed = [
        evaluate_seed(
            s,
            args.D,
            args.n_facts,
            args.vocab_mode,
            args.low_conf_cutoff,
            args.confidence_source_mode,
            args.source_monitor_D,
        )
        for s in args.seeds
    ]

    totals = Counter()
    for r in per_seed:
        totals.update(r["counts"])
    core_ok = all(
        r["default_off_identity"]["ok"]
        and r["seed_core_ok"]
        and r["communicable_path"]["hard_moat_preserved"] is not False
        for r in per_seed
    )
    low_conf_measured = totals["low_conf_wrong"] > 0
    low_conf_all_downgraded = bool(
        low_conf_measured and totals["low_conf_wrong_downgraded"] == totals["low_conf_wrong"]
    )
    high_conf_errors_remain = totals["wrong_assert"] > 0
    source_mismatch_measured = totals["source_mismatch_wrong"] > 0
    source_mismatch_all_downgraded = bool(
        source_mismatch_measured
        and totals["source_mismatch_wrong_downgraded"] == totals["source_mismatch_wrong"]
        and totals["source_mismatch_correct"] == 0
    )

    low_conf_some_downgraded = totals["low_conf_wrong_downgraded"] > 0

    if core_ok and low_conf_all_downgraded and not high_conf_errors_remain:
        verdict = "GO"
    elif core_ok and low_conf_measured and low_conf_some_downgraded:
        verdict = "PARTIAL"
    else:
        verdict = "NEGATIVE"

    success_components = {
        "default_off_identity_preserved": all(r["default_off_identity"]["ok"] for r in per_seed),
        "hard_moat_added_false_accepts_zero": totals["added_false_accepts"] == 0,
        "self_schema_not_invoked_on_hard_moat": totals["self_schema_invoked_on_hard_moat"] == 0,
        "low_conf_familiar_wrong_measured": low_conf_measured,
        "low_conf_familiar_wrong_some_downgraded": low_conf_some_downgraded,
        "low_conf_familiar_wrong_all_downgraded": low_conf_all_downgraded,
        "source_mismatch_wrong_measured": source_mismatch_measured,
        "source_mismatch_wrong_all_downgraded": source_mismatch_all_downgraded,
        "source_mismatch_correct_false_positive_zero": totals["source_mismatch_correct"] == 0,
        "wrong_asserts_absent_for_full_go": not high_conf_errors_remain,
    }
    preconditions = [
        {
            "name": "production_and_default_off_paths_measured",
            "ok": all("default_off_identity" in r and "counts" in r for r in per_seed),
        },
        {
            "name": "moat_and_matched_fact_counts_recorded",
            "ok": all(
                r["counts"]["matched_queries"] > 0
                and "added_false_accepts" in r["counts"]
                and "self_schema_invoked_on_hard_moat" in r["counts"]
                for r in per_seed
            ),
        },
        {
            "name": "verdict_derived_from_recorded_counts",
            "ok": verdict == (
                "GO" if core_ok and low_conf_all_downgraded and not high_conf_errors_remain
                else ("PARTIAL" if core_ok and low_conf_measured and low_conf_some_downgraded else "NEGATIVE")
            ),
        },
    ]

    aggregate = {
        "core_ok": core_ok,
        "low_conf_measured": low_conf_measured,
        "low_conf_all_downgraded": low_conf_all_downgraded,
        "source_mismatch_measured": source_mismatch_measured,
        "source_mismatch_all_downgraded": source_mismatch_all_downgraded,
        "high_conf_errors_remain": high_conf_errors_remain,
        "wrong_assert_rate": (totals["wrong_assert"] / totals["wrong"]) if totals["wrong"] else None,
        "source_mismatch_wrong_downgrade_rate": (
            totals["source_mismatch_wrong_downgraded"] / totals["source_mismatch_wrong"]
        ) if totals["source_mismatch_wrong"] else None,
        "low_conf_wrong_downgrade_rate": (
            totals["low_conf_wrong_downgraded"] / totals["low_conf_wrong"]
        ) if totals["low_conf_wrong"] else None,
        "correct_assert_rate": (totals["correct_assert"] / totals["correct"]) if totals["correct"] else None,
        "total_counts": dict(totals),
    }

    out = {
        "runner": "research/runners/_laneC_self_schema_honesty_wirein_derisk.py",
        "faculty": "Lane C metacognition / self-schema honesty in production known-fact conversation",
        "theory": "answer-process confidence routed through a fixed meta_schema -> self_schema spiking relay",
        "backend": os.environ.get("SIM_BACKEND", "(unset)"),
        "seeds": list(args.seeds),
        "D": int(args.D),
        "n_facts": int(args.n_facts),
        "vocab_mode": args.vocab_mode,
        "low_conf_cutoff": float(args.low_conf_cutoff),
        "confidence_source_mode": args.confidence_source_mode,
        "source_monitor_D": int(args.source_monitor_D),
        "verdict": verdict,
        "aggregate": aggregate,
        "success_components": success_components,
        "per_seed": per_seed,
        "preconditions": preconditions,
        "honest_scope": (
            "Production wire-in is built and default-off. The moat remains first, and low-confidence familiar-wrong "
            "recalls are downgraded. High-confidence wrong recalls still assert, so this is not a solved honesty "
            "mechanism; the next step needs a learned/calibrated correctness confidence signal rather than trace "
            "confidence alone. The source_consistency_floor mode is a named scaffold over composer source metadata, "
            "not an end-state biological correctness mechanism. The neural_source_consistency mode uses a separate "
            "RF source-memory echo, not the exact trace source dict; it remains a bounded source-monitor burn-down step."
        ),
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    out_path = Path(args.json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    print(
        f"[laneC-wire] === VERDICT: {verdict} === "
        f"low_conf_wrong={totals['low_conf_wrong']} downgraded={totals['low_conf_wrong_downgraded']} "
        f"wrong_assert={totals['wrong_assert']} added_FA={totals['added_false_accepts']} "
        f"elapsed={out['elapsed_seconds']}s",
        flush=True,
    )
    print(f"[laneC-wire] wrote {out_path}", flush=True)
    return 0 if verdict in ("GO", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
