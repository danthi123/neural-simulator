"""FULL-BRAIN INTEGRATION SMOKE for `BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE` (2026-09-02
follow-on to `research/findings/2026-09-01-np-entailment-copula-coverage-widening.md`, whose own
verify is PARSING-level only, no 15k-LTM brain).

WHAT THIS ADDS over that parsing-level verify. The widening was proven against hand-built
(sentence, topic, facts) inputs called directly against `gate_sentence`. This runner proves the
SAME widening actually FIRES through the REAL production entry point
(`webapp.server.brain_chat`, in-process, the exact function the HTTP route dispatches to -- same
pattern as `_open_ended_bundle_moat_safety_soak.py`, which this is a small follow-on to), against
the REAL shipped 15k-LTM store (`BRAIN_LTM_SHIP_DEFAULT=1`), on the REAL Qwen-generated reply for
`castleford_f_c` (not a hand-typed sentence) -- i.e. that the wiring from `post_filter` down to
this new code path is load-bearing end to end, not just correct in isolation.

TWO ARMS, one session each (fresh composer build per arm, mirroring the parent soak's own
resilience pattern):
  B        BRAIN_OPEN_ENDED=1 BRAIN_OPEN_ENDED_NP_ENTAILMENT=1 BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE=0
  B_wide   BRAIN_OPEN_ENDED=1 BRAIN_OPEN_ENDED_NP_ENTAILMENT=1 BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE=1
Same fixed internal Qwen generation seed both arms (server.py never takes a seed override -- see
the parent soak's own "SEED HONESTY" note), so arms B and B_wide share the identical RAW
generation; any FILTERED-text difference is attributable ONLY to the new post-filter code path
firing, not to a different generation.

TOPICS default to `castleford_f_c` (the concrete measured miss) -- override with `--topics` for a
broader real-traffic check (comma-separated agent slugs from the wikidata_core_15k store).

COMPUTE ROUTING (why this is a POOL job, not a local run). Building the tiny-demo onebrain
composer + the 15k-LTM store per arm costs real wall-clock (~2-4 min/arm on CPU, per the parent
soak's own measurement) and non-trivial resident RAM; the 2026-08-26 OOM history + this session's
RAM-safety scope say route a 15k-LTM brain build to the pool rather than build it locally without
first checking `free -m` for >=5000MB available. `SIM_BACKEND=numpy` pins to CPU (no GPU init,
matches the parent soak's own routing, does not touch the running GPU queue campaign).

Usage (pool node, from ~/derisk-pool/sim):
  SIM_BACKEND=numpy PYTHONPATH=. .venv/bin/python -m \\
      research.runners._np_entailment_copula_coverage_integration_smoke \\
      --out research/findings/raw/_np_entailment_copula_coverage_integration_smoke.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
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


ARMS = {
    "B":      {"BRAIN_OPEN_ENDED_NP_ENTAILMENT": "1", "BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE": "0",
               "BRAIN_OPEN_ENDED_GEN_TIME_HONESTY": "0"},
    "B_wide": {"BRAIN_OPEN_ENDED_NP_ENTAILMENT": "1", "BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE": "1",
               "BRAIN_OPEN_ENDED_GEN_TIME_HONESTY": "0"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topics", type=str, default="castleford_f_c",
                     help="comma-separated wikidata_core_15k agent slugs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str,
                     default=str(_REPO / "research" / "findings" / "raw" /
                                 "_np_entailment_copula_coverage_integration_smoke.json"))
    a = ap.parse_args()
    topics = [t.strip() for t in a.topics.split(",") if t.strip()]

    import webapp.server as S  # noqa: E402 (heavy import; must come after env setup)

    RENDERER = "stub"
    os.environ["BRAIN_OPEN_ENDED"] = "1"   # the parent -- explicit ON, both arms need it

    def _chat(session, msg, reset):
        resp = S.brain_chat(S.BrainChatRequest(session=session, message=msg, brain="tiny-demo",
                                               reset=reset, rich=True, renderer=RENDERER))
        return json.loads(bytes(resp.body))

    def _free_session(session):
        """Mirrors `_open_ended_bundle_moat_safety_soak._free_session` -- pop this arm's ChatBrain +
        every per-session cache before the next arm builds its own, bounding residency to ONE composer
        at a time."""
        cache_key = (session, "tiny-demo", RENDERER)
        S._BRAIN_CHATS.pop(cache_key, None)
        S._BRAIN_RICH.pop(cache_key, None)
        try:
            from webapp import continuous_engine as _CE
            _CE.forget_session(cache_key)
        except Exception:
            pass
        for _dname in ("_SESSION_MOOD", "_SESSION_WORLDVIEW", "_SESSION_MULTIREF", "_SESSION_SILENT_WM",
                       "_SESSION_SELFINIT", "_SESSION_DISCOURSE", "_SESSION_PMEM"):
            try:
                getattr(S, _dname).pop(cache_key, None)
            except Exception:
                pass
        try:
            import research.runners.d5_episodic_production_organ as _EP
            _EP.reset_episodic_organ(cache_key)
        except Exception:
            pass
        try:
            import research.runners.causal_whatif_production_organ as _CA
            _CA.reset_organ(cache_key)
        except Exception:
            pass
        gc.collect()

    def run_arm(arm_name):
        for k, v in ARMS[arm_name].items():
            os.environ[k] = v   # explicit "0"/"1" every time -- never popped (staleness trap guard)
        log(f"=== arm {arm_name}: COPULA_COVERAGE={os.environ['BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE']} ===")
        session = f"oe_copula_smoke_{arm_name}_{a.seed}"
        rows = []
        first = True
        for topic in topics:
            d = _chat(session, f"Tell me about {topic}", reset=first)
            first = False
            oe = d.get("open_ended") or {}
            row = {"topic": topic, "known_flag": oe.get("known"), "facts": oe.get("facts"),
                   "raw": oe.get("raw") or "", "filtered": oe.get("filtered") or d.get("answer") or "",
                   "generator": oe.get("generator"), "wkv_mouth_used": oe.get("wkv_mouth_used")}
            rows.append(row)
            log(f"  {topic!r}: gen={row['generator']} filtered={row['filtered'][:160]!r}")
        _free_session(session)
        return rows

    t0 = time.time()
    rows_B = run_arm("B")
    rows_Bwide = run_arm("B_wide")
    elapsed = time.time() - t0

    by_topic_B = {r["topic"]: r for r in rows_B}
    by_topic_Bwide = {r["topic"]: r for r in rows_Bwide}
    diffs = []
    for t in topics:
        rb, rw = by_topic_B[t], by_topic_Bwide[t]
        same_raw = (rb["raw"] == rw["raw"])
        filtered_differs = (rb["filtered"] != rw["filtered"])
        diffs.append({
            "topic": t,
            "same_raw_generation": same_raw,          # sanity: B/B_wide must share generation (differ only in filter)
            "filtered_differs": filtered_differs,       # the load-bearing signal: did the widening actually fire?
            "B_filtered": rb["filtered"], "B_wide_filtered": rw["filtered"],
        })

    n_fired = sum(1 for d in diffs if d["filtered_differs"])
    all_same_raw = all(d["same_raw_generation"] for d in diffs)

    art = {
        "runner": "np_entailment_copula_coverage_integration_smoke",
        "purpose": "prove BRAIN_OPEN_ENDED_NP_ENTAILMENT_COPULA_COVERAGE fires through the real "
                   "webapp.server.brain_chat path against the real 15k-LTM store (not just the "
                   "parsing-level verify's hand-built inputs)",
        "topics": topics, "seed": a.seed,
        "rows_B": rows_B, "rows_B_wide": rows_Bwide,
        "diffs": diffs,
        "n_topics": len(topics), "n_filtered_differs": n_fired,
        "all_arms_shared_raw_generation": all_same_raw,
        "wall_seconds": round(elapsed, 1),
    }

    print("\n=== DIFFS (B vs B_wide, same raw generation expected) ===")
    print(json.dumps(diffs, indent=1))
    print(f"\nn_topics={len(topics)} n_filtered_differs={n_fired} all_arms_shared_raw_generation={all_same_raw}")

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(art, indent=1))
    print(f"\nwrote {out_path}")
    return art


if __name__ == "__main__":
    main()
