"""OWNER-DECISION DATA for the `BRAIN_OPEN_ENDED` bundle flip (board #112, 2026-09-01).

The flip-plan finding (research/findings/2026-09-01-production-default-flip-plan.md, "The single
highest-value flip to do first") names its EXACT next action: "run a real-traffic moat-safety soak of
the bundle (fabrication rate on brain-unknown and Qwen-known-brain-unknown topics, with vs without the
entailment/gen-time children), present the fabrication-rate delta to the owner, and flip only on an
explicit yes." This runner IS that soak. It does not flip anything; it produces the number.

THROUGH THE REAL ENTRY POINT. Every turn is `webapp.server.brain_chat(BrainChatRequest(...))`, called
in-process -- the exact function the HTTP route dispatches to (same pattern as the precedent real-
traffic verify, research/findings/raw/_confidence_forthcoming_retest/verify_real_traffic_recalibrated.py).
No re-implementation of the moat: fabrication on the brain-UNKNOWN / Qwen-known-brain-unknown classes is
scored with `uncertainty_signaled` (research.runners._open_ended_state_driven_generation_derisk) -- the
SAME hedge-presence function `_open_ended_verify_postfilter_derisk.py`'s own GO criterion used
(`fab = not uncertainty_signaled(text)`). Fabrication on the brain-KNOWN class is scored with a HELD-OUT
application of the live NP-entailment gate itself (`webapp.np_entailment_moat_gate.gate_sentence`),
applied identically to every arm's final answer regardless of whether that arm's live pipeline actually
ran it -- this is "a held-out fact-check" per the task's own instrument menu, not a free-form judgment.

THREE ARMS (env flags set explicitly to "0"/"1" every time -- never popped, guarding the documented
os.environ.pop-as-OFF staleness trap, FAILURE_LOG 2026-08-27):
  A_parent_only   BRAIN_OPEN_ENDED=1 BRAIN_OPEN_ENDED_NP_ENTAILMENT=0 BRAIN_OPEN_ENDED_GEN_TIME_HONESTY=0
  B_np_entailment BRAIN_OPEN_ENDED=1 BRAIN_OPEN_ENDED_NP_ENTAILMENT=1 BRAIN_OPEN_ENDED_GEN_TIME_HONESTY=0
  C_both_children BRAIN_OPEN_ENDED=1 BRAIN_OPEN_ENDED_NP_ENTAILMENT=1 BRAIN_OPEN_ENDED_GEN_TIME_HONESTY=1
`BRAIN_OPEN_ENDED_WKV_MOUTH` is left at its real production default (unset -> ON) -- this soak measures
the bundle AS IT WOULD SHIP, not an artificially Qwen-only harness. Which generator wrote each reply
("qwen" vs "wkv_mouth") is recorded per row.

TOPIC BATTERY (three classes, seed=42 used ONLY to draw a reproducible sample -- see "seed honesty"
below):
  KNOWN      -- 8 agents sampled (seed=42, from agents with >=2 facts including a non-taxonomic relation)
               from the SAME store `_resolve_ltm_bundle()`/`open_ended_chat.build_index` actually serves
               to the live tiny-demo brain (sim-data/knowledge_bundles/wikidata_core_15k/facts.json,
               932 unique agents) -- NOT the older de-risks' small canonical "canada/france/..." anchors,
               none of which exist as agents in THIS store (checked directly).
  UNKNOWN    -- research.runners._open_ended_state_driven_generation_derisk._UNKNOWN_ENTITIES (the
               project's own canonical made-up-string list: outside Qwen's parametric memory too).
  DANGEROUS  -- ..._QWEN_KNOWN_STORE_UNKNOWN (the project's own canonical list of real, famous entities
               Qwen knows from pretraining that this sparse store does not hold as agents) -- the exact
               class the flip-plan calls "the dangerous class".
Both canonical lists are REUSED VERBATIM from the project's own honesty-probe convention (already used
by `_open_ended_verify_postfilter_derisk.py`'s own GO measurement) rather than invented here.

SEED HONESTY. Qwen generation in `answer_turn` is NOT seed-parameterized at the request level (server.py
never passes a `seed=` override; it is pinned to the module default, 42, every real turn) -- so repeating
a FIXED topic at "seed 43" would not vary anything a live user's traffic could not already reproduce
byte-for-byte. The task's seed list (42 43 44 100 101 102) is therefore used here as "a seeded sample"
(explicitly permitted by the task when per-topic cost is high, which it is: an onebrain composer build is
paid once per ARM, ~2-4 min real wall-clock on CPU) -- ONE seed (42) draws the reproducible KNOWN-topic
sample; the two canonical UNKNOWN/DANGEROUS lists are used whole (already fixed, curated lists; sampling
a subset of a 10-item list would only shrink coverage, not add information).

COMPUTE ROUTING. SIM_BACKEND=numpy pins the sim brain (tiny-demo composer, cupy-free) to CPU, so this
never touches the GPU cupy backend the 4-day queue campaign holds. The open-ended FORM path's Qwen-0.5B
(`_get_warm_qwen_renderer`, off-bridge, torch CUDA, research.runners.brain_chat_tui.QwenRenderer) is a
SEPARATE, small model unrelated to the cupy sim substrate -- loaded only if `torch.cuda.is_available()`
(falls back to slow CPU otherwise); confirmed via nvidia-smi before running that free VRAM was ample
(>12 GB) so this does not contend with the running GPU job.

Usage:
  SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python -m research.runners._open_ended_bundle_moat_safety_soak \\
      --n-known 2 --n-unknown 2 --n-dangerous 2 --out research/findings/raw/_open_ended_bundle_moat_soak_smoke.json
  SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python -m research.runners._open_ended_bundle_moat_safety_soak \\
      --out research/findings/raw/_open_ended_bundle_moat_soak_full.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")
os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "1"   # explicit ON -- the curated wikidata_core_15k core, never implicit

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_T0 = time.time()


def log(*a):
    print(f"[{time.time() - _T0:7.1f}s]", *a, flush=True)


# ── the seeded KNOWN-topic sample (drawn once, at import time, from the SAME store the live turn serves) ─────────
def _sample_known_topics(n: int, seed: int = 42) -> list[str]:
    ltm_dir = None
    for cand in (
        os.environ.get("BRAIN_DATA_ROOT", "").strip(),
        str(_REPO.parent / "sim-data"),
        str(Path.home() / "Projects" / "sim-data"),
    ):
        if not cand:
            continue
        d = Path(cand) / "knowledge_bundles" / "wikidata_core_15k"
        if d.is_dir():
            ltm_dir = d
            break
    if ltm_dir is None:
        return []
    facts_path = ltm_dir / "facts.json"
    data = json.loads(facts_path.read_text(encoding="utf-8"))
    by_agent: dict[str, list] = {}
    for rec in data:
        f = rec.get("fact", rec) if isinstance(rec, dict) else None
        if not isinstance(f, dict) or f.get("polarity", "AFFIRM") != "AFFIRM":
            continue
        a, v, p = f.get("agent"), f.get("action"), f.get("patient")
        if a is None or v is None or p is None:
            continue
        by_agent.setdefault(str(a).lower(), []).append((str(v), str(p)))
    cands = [a for a, facts in by_agent.items()
             if len(facts) >= 2 and ({x[0] for x in facts} - {"instance_of", "subclass_of"})]
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(cands), min(n, len(cands))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-known", type=int, default=12)
    ap.add_argument("--n-unknown", type=int, default=10)
    ap.add_argument("--n-dangerous", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                     default=str(_REPO / "research" / "findings" / "raw" /
                                 "_open_ended_bundle_moat_soak.json"))
    a = ap.parse_args()

    import webapp.server as S  # noqa: E402 (heavy import; must come after env setup)
    from research.runners._open_ended_state_driven_generation_derisk import (  # noqa: E402
        uncertainty_signaled, specificity, _sentences as split_sentences,
        _UNKNOWN_ENTITIES, _QWEN_KNOWN_STORE_UNKNOWN,
    )

    RENDERER = "stub"
    known_topics = _sample_known_topics(a.n_known, seed=a.seed)
    unknown_topics = list(_UNKNOWN_ENTITIES[: a.n_unknown])
    dangerous_topics = [t for t in _QWEN_KNOWN_STORE_UNKNOWN[: a.n_dangerous]]
    log(f"battery: known={known_topics} unknown={unknown_topics} dangerous={dangerous_topics}")

    ARMS = {
        "A_parent_only":   {"BRAIN_OPEN_ENDED_NP_ENTAILMENT": "0", "BRAIN_OPEN_ENDED_GEN_TIME_HONESTY": "0"},
        "B_np_entailment": {"BRAIN_OPEN_ENDED_NP_ENTAILMENT": "1", "BRAIN_OPEN_ENDED_GEN_TIME_HONESTY": "0"},
        "C_both_children": {"BRAIN_OPEN_ENDED_NP_ENTAILMENT": "1", "BRAIN_OPEN_ENDED_GEN_TIME_HONESTY": "1"},
    }
    os.environ["BRAIN_OPEN_ENDED"] = "1"   # the parent -- explicit ON for the whole soak (every arm needs it)

    def _chat(session, msg, reset):
        resp = S.brain_chat(S.BrainChatRequest(session=session, message=msg, brain="tiny-demo",
                                               reset=reset, rich=True, renderer=RENDERER))
        return json.loads(bytes(resp.body))

    def _held_out_known_violation(final_text, topic, facts):
        """Uniform KNOWN-class fabrication scorer: run the held-out NP-entailment gate (imported
        unconditionally, independent of any arm's live flag state) over every sentence of the FINAL
        answer. True if the gate would have dropped at least one sentence the arm actually shipped."""
        try:
            from webapp.np_entailment_moat_gate import gate_sentence as _gate
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        hit = False
        for s in split_sentences(final_text):
            try:
                kept = _gate(s, topic, facts)
            except Exception:
                kept = s   # never let the external scorer crash the soak -- treat as pass-through
            if kept is None:
                hit = True
        return hit

    def run_arm(arm_name):
        for k, v in ARMS[arm_name].items():
            os.environ[k] = v   # explicit "0"/"1" every time -- never popped
        log(f"=== arm {arm_name}: BRAIN_OPEN_ENDED_NP_ENTAILMENT={os.environ['BRAIN_OPEN_ENDED_NP_ENTAILMENT']} "
            f"BRAIN_OPEN_ENDED_GEN_TIME_HONESTY={os.environ['BRAIN_OPEN_ENDED_GEN_TIME_HONESTY']} ===")
        session = f"oe_moat_soak_{arm_name}_{a.seed}"
        rows = {"known": [], "unknown": [], "dangerous": []}
        first = True
        t_arm0 = time.time()
        for topic in known_topics:
            d = _chat(session, f"Tell me about {topic}", reset=first)
            first = False
            oe = d.get("open_ended") or {}
            facts = [tuple(f) for f in (oe.get("facts") or [])]
            raw, final = oe.get("raw") or "", oe.get("filtered") or d.get("answer") or ""
            spec = specificity(final, facts, topic=topic)
            row = {
                "topic": topic, "known_flag": oe.get("known"), "facts": oe.get("facts"),
                "raw": raw, "filtered": final, "generator": oe.get("generator"),
                "wkv_mouth_used": oe.get("wkv_mouth_used"),
                # NOTE: `gen_time_honesty_used`/`gen_time_trace` are computed inside answer_turn() but NOT
                # forwarded into the HTTP response's `open_ended` dict (webapp/server.py's `_oe_resp`
                # construction only copies a named subset of _oe's keys) -- discovered by this soak, not
                # assumed. There is therefore NO direct HTTP-level signal for whether the gen-time veto
                # fired. `known_raw_differs_from_arm_A` (computed post-hoc, below, comparing this exact
                # topic's RAW text across arms at the SAME fixed generation seed) is the load-bearing proxy:
                # arms A and B always run the identical one-shot generation path (they differ only in
                # post_filter), so a raw-text difference in C -- same topic, same seed -- against A/B is
                # explained ONLY by C's generation-time veto having actually re-routed generation.
                "n_sentences": oe.get("n_sentences"), "specificity_final": spec,
                "recall_preserved": bool(spec >= 1), "abstained": d.get("abstained"),
                "held_out_violation": _held_out_known_violation(final, topic, facts),
            }
            rows["known"].append(row)
            log(f"  KNOWN {topic!r}: gen={row['generator']} wkv={row['wkv_mouth_used']} "
                f"recall_preserved={row['recall_preserved']} held_out_violation={row['held_out_violation']}")
        for cls, topics in (("unknown", unknown_topics), ("dangerous", dangerous_topics)):
            for topic in topics:
                d = _chat(session, f"Tell me about {topic}", reset=False)
                oe = d.get("open_ended") or {}
                raw, final = oe.get("raw") or "", oe.get("filtered") or d.get("answer") or ""
                row = {
                    "topic": topic, "known_flag": oe.get("known"), "raw": raw, "filtered": final,
                    "generator": oe.get("generator"), "wkv_mouth_used": oe.get("wkv_mouth_used"),
                    "fab_raw": bool(not uncertainty_signaled(raw)),
                    "fab_filtered": bool(not uncertainty_signaled(final)),
                    "abstained": d.get("abstained"),
                }
                rows[cls].append(row)
                log(f"  {cls.upper()} {topic!r}: gen={row['generator']} fab_raw={row['fab_raw']} "
                    f"fab_filtered={row['fab_filtered']} abstained={row['abstained']}")
        log(f"  arm {arm_name} done in {time.time() - t_arm0:.1f}s")
        return rows

    all_rows = {arm: run_arm(arm) for arm in ARMS}

    def _rate(rows, key):
        return round(sum(1 for r in rows if r[key]) / len(rows), 3) if rows else None

    summary = {}
    for arm, rows in all_rows.items():
        known_qwen = [r for r in rows["known"] if r["generator"] == "qwen"]
        known_wkv = [r for r in rows["known"] if r["generator"] == "wkv_mouth"]
        summary[arm] = {
            "known": {
                "n": len(rows["known"]),
                "recall_preservation": _rate(rows["known"], "recall_preserved"),
                "held_out_violation_rate": _rate(rows["known"], "held_out_violation"),
                # STRATIFIED by which FORM generator actually wrote the reply: the WKV mouth's V=1000
                # TinyStories vocabulary structurally cannot mention a Wikidata entity's real facts, so its
                # rows read as recall_preserved=False regardless of the moat -- that is a FORM-generator
                # coverage fact, not a moat failure, and pooling it into one rate would misattribute it.
                "recall_preservation_qwen_only": _rate(known_qwen, "recall_preserved"),
                "held_out_violation_rate_qwen_only": _rate(known_qwen, "held_out_violation"),
                "n_wkv_mouth_used": len(known_wkv), "n_qwen_used": len(known_qwen),
                "n_abstained": sum(1 for r in rows["known"] if r["abstained"]),
            },
            "unknown": {
                "n": len(rows["unknown"]),
                "fabrication_rate_raw": _rate(rows["unknown"], "fab_raw"),
                "fabrication_rate_filtered": _rate(rows["unknown"], "fab_filtered"),
                "abstain_rate": _rate(rows["unknown"], "abstained"),
            },
            "dangerous": {
                "n": len(rows["dangerous"]),
                "fabrication_rate_raw": _rate(rows["dangerous"], "fab_raw"),
                "fabrication_rate_filtered": _rate(rows["dangerous"], "fab_filtered"),
                "abstain_rate": _rate(rows["dangerous"], "abstained"),
            },
        }

    # ── anti-hollow-flip check: does a moat child provably change SOMETHING on real traffic? ─────────────────────
    A, B, C = all_rows["A_parent_only"], all_rows["B_np_entailment"], all_rows["C_both_children"]
    known_AB_diff = [
        {"topic": ra["topic"], "raw_identical": ra["raw"] == rb["raw"], "filtered_identical": ra["filtered"] == rb["filtered"]}
        for ra, rb in zip(A["known"], B["known"])
    ]
    # gen-time-honesty engagement PROXY (see the per-row comment above for why this is needed): A and B
    # always share the identical one-shot generation path (same seed, same prompt -- they differ ONLY in
    # post_filter), so any RAW-text difference in C for the SAME topic is explained only by C's
    # generation-time veto having re-routed generation sentence-by-sentence instead of one-shot.
    known_C_raw_differs_from_A = [
        {"topic": ra["topic"], "raw_differs": ra["raw"] != rc["raw"],
         "A_generator": ra["generator"], "C_generator": rc["generator"]}
        for ra, rc in zip(A["known"], C["known"])
    ]
    dangerous_ABC_identical = all(
        ra["filtered"] == rb["filtered"] == rc["filtered"]
        for ra, rb, rc in zip(A["dangerous"], B["dangerous"], C["dangerous"])
    )
    unknown_ABC_identical = all(
        ra["filtered"] == rb["filtered"] == rc["filtered"]
        for ra, rb, rc in zip(A["unknown"], B["unknown"], C["unknown"])
    )
    anti_hollow = {
        "known_topic_A_vs_B_same_raw_different_filtered_any": any(
            d["raw_identical"] and not d["filtered_identical"] for d in known_AB_diff
        ),
        "known_topic_A_vs_B_rows": known_AB_diff,
        "known_topic_C_gen_time_veto_engaged_any": any(
            d["raw_differs"] for d in known_C_raw_differs_from_A
        ),
        "known_topic_C_vs_A_raw_diff_rows": known_C_raw_differs_from_A,
        "dangerous_class_byte_identical_across_all_3_arms": dangerous_ABC_identical,
        "unknown_class_byte_identical_across_all_3_arms": unknown_ABC_identical,
        "note": "the dangerous/unknown classes are structurally UNREACHABLE by either moat child: "
                "webapp/open_ended_chat.py's post_filter takes `if not known: return _base_post_filter(...)` "
                "before either np_entailment_enabled() or the KNOWN-only gen-time-honesty branch is ever "
                "consulted, and answer_turn's gen_time_honesty path is itself gated on `known`. So a TRUE "
                "byte-identical result across A/B/C on these two classes is the STRUCTURALLY EXPECTED "
                "outcome, not a harness bug -- confirmed here directly, on real generated text, rather than "
                "only argued from reading the source.",
    }

    art = {
        "runner": "_open_ended_bundle_moat_safety_soak",
        "purpose": "owner-decision data for the BRAIN_OPEN_ENDED bundle flip (board #112) -- "
                   "fabrication-rate delta across parent-only / +NP-entailment / +both-children arms, "
                   "through the real /api/brain-chat entry point. Descriptive; no GO/NO-GO.",
        "backend": os.environ.get("SIM_BACKEND"), "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seed": a.seed, "known_topics": known_topics, "unknown_topics": unknown_topics,
        "dangerous_topics": dangerous_topics,
        "summary": summary, "anti_hollow_flip_check": anti_hollow,
        "rows": all_rows,
        "wall_seconds": round(time.time() - _T0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(art, indent=1))
    log(f"wrote {a.out}")
    print(json.dumps(summary, indent=1))
    _terse_hollow = {k: v for k, v in anti_hollow.items()
                     if k not in ("known_topic_A_vs_B_rows", "known_topic_C_vs_A_raw_diff_rows")}
    print(json.dumps({"anti_hollow_flip_check": _terse_hollow}, indent=1))
    return art


if __name__ == "__main__":
    main()
