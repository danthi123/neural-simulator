"""HANDLER-LEVEL no-regression soak for the W5 GRADED-CIRCUMPLEX upgrade (`BRAIN_AFFECTIVE_TOM_GRADED`), through the
REAL production `webapp.server.brain_chat` handler (stub renderer). Mirrors `_affective_tom_flip_soak.py` (the
original bistable-path flip gate) but toggles the NEW graded flag while the BASE W5 flag stays ON (production
default), across two other-agent situations of DIFFERENT REAL-WORD intensity ("Maria is lonely" vs "Maria is
heartbroken") so the check exercises genuine end-to-end text -> DR-2 appraisal -> graded-ladder -> lead behaviour,
not a synthetic appraisal dict.

BAR:
  NO-REGRESSION: every ORDINARY turn (recall / abstain) is BYTE-IDENTICAL bistable-lead-path vs graded-path (content
  fields + answer-with-lead-stripped); no crash from the new organ coexisting with the tiny-demo recall bridge.
  LOAD-BEARING / SURPASS: on the SAME two other-agent messages, the graded path's lead DIFFERS between the mild
  ("lonely") and strong ("heartbroken") situation, while the bistable path gives the IDENTICAL lead for both
  (the frontier this upgrade closes, demonstrated end-to-end through the real handler).
  LESION: BRAIN_AFFECTIVE_TOM_LESION=1 collapses the graded lead to '' on every trigger, reverting the answer
  byte-identically to the flag-off-equivalent bare surface.

  Run (1-seed confirm): SIM_BACKEND=numpy python -m research.runners._affective_tom_graded_flip_soak --seeds 42
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
os.environ.setdefault("BRAIN_CHAT_RENDERER", "stub")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

OUT = _REPO / "research" / "findings" / "raw" / "_affective_tom_graded" / "flip_soak_seed42.json"

CONV = [
    ("what does the dog chase?", "ordinary"),
    ("what is the capital of france?", "ordinary"),
    ("Maria is lonely", "trigger_mild_bad"),        # a WEAK-magnitude other-situation (real DR-2 valence ~-0.25)
    ("Maria is heartbroken", "trigger_strong_bad"), # a STRONG-magnitude other-situation (real DR-2 valence ~-0.83)
    ("what does the dog chase?", "ordinary"),
]

_SILENCE = ["BRAIN_AFFECT", "BRAIN_AFFECT_DRIVES", "BRAIN_SWAP_DRIVES", "BRAIN_DA_DRIVES", "BRAIN_DA_ENCODING",
            "BRAIN_CONTINUOUS", "BRAIN_CONTINUOUS_DRIVES", "BRAIN_CONTINUOUS_IDEATE", "BRAIN_PMEM",
            "BRAIN_GNW_SWAP", "BRAIN_GNW_BUS"]
_KEEP = ("answer", "abstained", "recalled_svo", "verified")


def _slim(resp: dict) -> dict:
    d = {k: resp.get(k) for k in _KEEP}
    d["has_tom_key"] = ("affective_tom" in resp)
    tom = resp.get("affective_tom") or {}
    d["tom_lead"] = tom.get("lead", "") if "affective_tom" in resp else ""
    d["tom_level_or_sign"] = tom.get("tone_level", tom.get("tone_sign")) if "affective_tom" in resp else None
    return d


def _fresh_chat(S, session):
    chat, source = S._build_chat_brain("tiny-demo", "stub")
    ck = (session, "tiny-demo", "stub")
    chat._brain_chat_source = source
    S._BRAIN_CHATS[ck] = chat
    return ck


def _run_conversation(S, session, *, graded, lesion, seed):
    from webapp.server import brain_chat, BrainChatRequest as Req
    import research.runners.affective_tom_production_organ as _ATM
    os.environ["BRAIN_AFFECTIVE_TOM"] = "1"                          # base W5 stays ON (production default)
    os.environ["BRAIN_AFFECTIVE_TOM_GRADED"] = "1" if graded else "0"
    os.environ["BRAIN_AFFECTIVE_TOM_LESION"] = "1" if lesion else "0"
    _ATM._ORGAN = None
    try:
        import research.runners._affective_tom_graded_derisk as _G
        _G._ORGAN = None
    except Exception:
        pass
    _fresh_chat(S, session)
    rows = []
    for msg, kind in CONV:
        r = brain_chat(Req(session=session, message=msg, brain="tiny-demo", renderer="stub", rich=False))
        rows.append({"msg": msg, "kind": kind, "slim": _slim(json.loads(r.body.decode("utf-8")))})
    return rows


def run_one(seed, backend):
    t0 = time.time()
    print(f"[tom-graded-soak] seed={seed} backend={backend}", flush=True)
    import webapp.server as S
    for f in _SILENCE:
        os.environ[f] = "0"
    result = {"seed": seed, "backend": backend}
    try:
        bistable = _run_conversation(S, f"tom-gsoak-bi-{seed}", graded=False, lesion=False, seed=seed)
        graded = _run_conversation(S, f"tom-gsoak-gr-{seed}", graded=True, lesion=False, seed=seed)
        les = _run_conversation(S, f"tom-gsoak-les-{seed}", graded=True, lesion=True, seed=seed)

        ordinary_identical = True
        triggered_content_identical = True
        per_turn = []
        mild_lead = strong_lead = None
        lesion_collapsed = True
        for b, g, l in zip(bistable, graded, les):
            kind = b["kind"]
            rec = {"msg": b["msg"], "kind": kind, "bistable": b["slim"], "graded": g["slim"], "lesion": l["slim"]}
            if kind == "ordinary":
                same = (b["slim"] == g["slim"])
                rec["ordinary_identical"] = bool(same)
                ordinary_identical = ordinary_identical and same
            else:
                content_same = all(b["slim"][k] == g["slim"][k] for k in ("abstained", "recalled_svo", "verified"))
                lead = g["slim"]["tom_lead"]
                answer_is_lead_plus_bistable_content = bool(lead) and \
                    (g["slim"]["answer"] == lead + b["slim"]["answer"][len(b["slim"]["tom_lead"]):])
                rec["content_same"] = bool(content_same)
                rec["graded_lead"] = lead
                triggered_content_identical = triggered_content_identical and content_same
                if kind == "trigger_mild_bad":
                    mild_lead = lead
                elif kind == "trigger_strong_bad":
                    strong_lead = lead
                les_lead = l["slim"]["tom_lead"]
                les_ok = (les_lead == "")
                rec["lesion_lead"] = les_lead
                rec["lesion_collapsed"] = bool(les_ok)
                lesion_collapsed = lesion_collapsed and les_ok
            per_turn.append(rec)

        # THE SURPASS, end-to-end through the real handler: mild vs strong OTHER-situation give DIFFERENT graded
        # leads (bistable would give the IDENTICAL lead for both -- both are negative-sign).
        bistable_mild = next(r["bistable"]["tom_lead"] for r in per_turn if r["kind"] == "trigger_mild_bad")
        bistable_strong = next(r["bistable"]["tom_lead"] for r in per_turn if r["kind"] == "trigger_strong_bad")
        bistable_collapses_magnitude = bool(bistable_mild == bistable_strong and bistable_mild != "")
        graded_differentiates_magnitude = bool(mild_lead and strong_lead and mild_lead != strong_lead)

        GO = bool(ordinary_identical and triggered_content_identical and lesion_collapsed
                  and bistable_collapses_magnitude and graded_differentiates_magnitude)
        result.update(dict(
            GO=GO, ordinary_identical=ordinary_identical, triggered_content_identical=triggered_content_identical,
            lesion_collapsed=lesion_collapsed, bistable_mild_lead=bistable_mild, bistable_strong_lead=bistable_strong,
            bistable_collapses_magnitude=bistable_collapses_magnitude, graded_mild_lead=mild_lead,
            graded_strong_lead=strong_lead, graded_differentiates_magnitude=graded_differentiates_magnitude,
            per_turn=per_turn))
        print(f"[tom-graded-soak] ordinary_identical={ordinary_identical} triggered_content_identical="
              f"{triggered_content_identical} lesion_collapsed={lesion_collapsed}", flush=True)
        print(f"[tom-graded-soak] BISTABLE mild={bistable_mild!r} strong={bistable_strong!r} "
              f"(collapses_magnitude={bistable_collapses_magnitude}) | GRADED mild={mild_lead!r} "
              f"strong={strong_lead!r} (differentiates={graded_differentiates_magnitude})", flush=True)
        print(f"[tom-graded-soak] seed={seed} => {'GO' if GO else 'NO-GO'}", flush=True)
    except Exception as e:  # noqa: BLE001
        result["error"] = repr(e); result["GO"] = False; traceback.print_exc()
    finally:
        os.environ["BRAIN_AFFECTIVE_TOM_GRADED"] = "0"
        os.environ["BRAIN_AFFECTIVE_TOM_LESION"] = "0"
    result["elapsed_s"] = round(time.time() - t0, 1)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    from sim.backend import get_backend
    _, backend = get_backend()
    results = {}; go = []
    for seed in a.seeds:
        r = run_one(seed, backend)
        results[seed] = r; go.append(bool(r.get("GO")))
    out_path = Path(a.out)
    if len(a.seeds) > 1:
        out_path = out_path.parent / f"flip_soak_summary_{len(a.seeds)}seed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"seeds": a.seeds, "n_go": int(sum(go)), "go": go, "backend": backend,
                                    "results": {str(s): results[s] for s in a.seeds}}, indent=2, default=str))
    print(f"[tom-graded-soak] wrote {out_path}")
    return 0 if (go and all(go)) else 1


if __name__ == "__main__":
    sys.exit(main())
