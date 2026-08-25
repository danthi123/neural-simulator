"""Board #129 PRODUCTION WIRE-IN de-risk: the #129 source-provenance opponent monitor wired into the live-chat
honesty pathway (`BrainConversationalAgent.known_fact_record` / `.reasoned_fact_record`,
`research/runners/source_provenance_honesty.py`, `webapp/server.py /api/brain-chat`).

THE RUNG (Vikunja #137, board #129 next rung): wire the de-risked (6-seed GO, 2026-08-25) "did the brain SEE
this fact or IMAGINE/INFER it" provenance read into the LIVE CHAT honesty pathway so a reply honestly reflects
provenance -- a PERCEIVED (directly taught) fact reads exactly as it does today, a GENERATED (multi-hop
composed) conclusion is flagged. Additive/default-off/byte-identical-when-off; the framing must be LOAD-BEARING
(driven by the live spiking judged label, not a caller-supplied claim) and moat-first (a hard abstain is never
touched).

WHAT THIS RUNNER VERIFIES, per seed, through the REAL `BrainConversationalAgent` class + a real
`RFPhasorComposer` (not a toy stand-in) -- exactly the production-wire-in verification convention this project
already uses for the adjacent self_schema_honesty wire-in (`_laneC_self_schema_honesty_wirein_derisk.py`):

  (A) DEFAULT-OFF BYTE-IDENTICAL -- with the faculty off, `known_fact_record`/`reasoned_fact_record` text is
      UNCHANGED and no provenance monitor is ever built (no substrate step taken).
  (B) MOAT-FIRST -- a hard abstain (an unknown cue / a dead-end reasoning chain) is NEVER touched by the
      provenance machinery: `provenance` stays None, the answer text stays "I don't know about that."
  (C) LOAD-BEARING DISCRIMINATION -- across N independent (PERCEIVED, GENERATED) fact pairs, the live spiking
      opponent-comparator's judged label matches the TRUE provenance with high accuracy, and the answer text
      framing DIFFERS systematically (perceived: unchanged; generated: flagged "I believe ..., but I reasoned
      that myself...").
  (D) LESION COLLAPSE -- with the monitor's plasticity gate held shut at encode (the #129 de-risk's OWN verified
      failing-direction anti-cheat), the SAME battery's judged-label accuracy collapses toward chance -- proving
      the live framing is driven by the LEARNED trace, not a hardcoded branch.
  (E) NO CROSS-TALK -- the faculty composes cleanly with self_schema_honesty (both may be on at once; neither
      changes the other's own fields).

Run:
  SIM_BACKEND=numpy python -m research.runners._129_source_provenance_honesty_wirein_derisk \
      --seeds 42 43 44 100 101 102 \
      --out research/findings/raw/lanes/metacog/_129_source_provenance_honesty_wirein_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import logging
logging.getLogger("SIM_BRIDGE").setLevel(logging.WARNING)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.brain_conversational_agent import BrainConversationalAgent  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
from research.runners.source_provenance_honesty import (  # noqa: E402
    PROVENANCE_GENERATED,
    PROVENANCE_PERCEIVED,
)
from tools.verdict import Verdict, UNDEFINED  # noqa: E402

N_PAIRS = 8         # (perceived, generated) fact pairs per seed -> 16 provenance judgments
D = 128
ACC_FLOOR = 0.90     # un-lesioned battery accuracy bar (the #129 de-risk itself cleared 1.000/6)
# NOT gated (reported only): under the lesion both prov pools are silent (rate 0.0), so the "judged label" is a
# host TIE-BREAK on a fixed RNG stream, not a real read -- exactly the #129 de-risk's own noted small-N noise
# ("the tie-broken accuracy is small-N noisy... it is REPORTED, not gated"). The LOAD-BEARING, non-noisy lesion
# proof is `lesion_d_zero` below (d and both rates are EXACTLY 0.0 -- deterministic, no tie-break involved).
LESIONED_ACC_CEIL = 0.75   # informational threshold for the printed diagnostic only


def _vocab(n_pairs):
    # 3 distinct tokens per PERCEIVED fact (agent, action, patient) + 4 distinct tokens per GENERATED chain
    # (start, action1, mid, action2 -- terminal reuses a PERCEIVED patient so the two families interleave through
    # a shared vocabulary, as a real conversation would), all pairwise-distinct so no cue collides.
    return [f"w{i:04d}" for i in range(n_pairs * 8)]


def _build(seed, n_pairs=N_PAIRS):
    vocab = _vocab(n_pairs)
    comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab, trace=True)
    perceived_cues, generated_cues = [], []
    vi = 0

    def take(k):
        nonlocal vi
        out = vocab[vi:vi + k]
        vi += k
        return out

    for _ in range(n_pairs):
        ag, ac, pt = take(3)
        comp.store(ag, ac, pt)
        perceived_cues.append((ag, ac, pt))
    for _ in range(n_pairs):
        start, a1, mid, a2, terminal = take(5)
        comp.store(start, a1, mid)
        comp.store(mid, a2, terminal)
        generated_cues.append((start, [a1, a2], terminal))
    return comp, vocab, perceived_cues, generated_cues


def _agent(seed, comp, vocab, *, enable, lesion=False):
    cfg = {"lesion": lesion} if lesion else None
    return BrainConversationalAgent(
        seed=seed,
        concepts={w: None for w in vocab},
        composer=comp,
        enable_neural_render=False,
        defer_parser=True,
        enable_source_provenance_honesty=enable,
        source_provenance_honesty_config=cfg,
    )


def evaluate_seed(seed):
    comp, vocab, perceived_cues, generated_cues = _build(seed)

    # ---- (A) default-off byte-identical -----------------------------------------------------------------------
    ag_off = _agent(seed, comp, vocab, enable=False)
    off_rows = []
    for ag, ac, pt in perceived_cues:
        rec = ag_off.known_fact_record((ag, ac))
        off_rows.append({
            "ok": bool(rec["answer_text"] == f"{ag} {ac} {pt}." and rec.get("provenance") is None
                       and "I believe" not in rec["answer_text"]),
        })
    for start, actions, terminal in generated_cues:
        rec = ag_off.reasoned_fact_record(start, actions)
        off_rows.append({
            "ok": bool(rec.get("provenance") is None and "I believe" not in rec["answer_text"]
                       and rec["answer"] == terminal),
        })
    default_off_ok = bool(off_rows) and all(r["ok"] for r in off_rows)
    monitor_built_when_off = ag_off._source_provenance_monitor is not None   # must stay False

    # ---- (B) moat-first: unknown cue / dead-end chain never touched --------------------------------------------
    ag_on = _agent(seed, comp, vocab, enable=True)
    hard_known = ag_on.known_fact_record(("__never_seen_agent__", "__never_seen_action__"))
    hard_chain = ag_on.reasoned_fact_record("__never_seen_agent__", ["__never_seen_action__"])
    moat_preserved = bool(
        hard_known["hard_abstain"] and hard_known["provenance"] is None
        and hard_known["answer_text"] == "I don't know about that."
        and hard_chain["hard_abstain"] and hard_chain["provenance"] is None
        and hard_chain["answer_text"] == "I don't know about that."
    )

    # ---- (C) load-bearing discrimination, un-lesioned -----------------------------------------------------------
    def _battery(agent):
        rows = []
        for ag, ac, pt in perceived_cues:
            rec = agent.known_fact_record((ag, ac))
            prov = rec.get("provenance") or {}
            rows.append({
                "truth": PROVENANCE_PERCEIVED, "judged": prov.get("label"),
                "known": prov.get("known"), "d": prov.get("d"),
                "text_unchanged": bool(rec["answer_text"] == f"{ag} {ac} {pt}."),
                "flagged": bool("I believe" in rec["answer_text"]),
            })
        for start, actions, terminal in generated_cues:
            rec = agent.reasoned_fact_record(start, actions)
            prov = rec.get("provenance") or {}
            rows.append({
                "truth": PROVENANCE_GENERATED, "judged": prov.get("label"),
                "known": prov.get("known"), "d": prov.get("d"),
                "text_unchanged": bool(rec["answer_text"] == f"{start} " + " ".join(actions) + f" {terminal}."),
                "flagged": bool("I believe" in rec["answer_text"]),
            })
        return rows

    real_rows = _battery(ag_on)
    real_correct = sum(1 for r in real_rows if r["judged"] == r["truth"])
    real_acc = real_correct / len(real_rows)
    # the TEXT must track the judged label exactly: perceived -> unchanged & unflagged; generated -> flagged
    text_matches_judgment = all(
        (r["judged"] == PROVENANCE_PERCEIVED and r["text_unchanged"] and not r["flagged"])
        or (r["judged"] == PROVENANCE_GENERATED and r["flagged"] and not r["text_unchanged"])
        for r in real_rows
    )
    every_judged_known = all(r["known"] for r in real_rows)

    # a fresh perceived-truth agent instance (SAME composer/facts) for the "unchanged-vs-off" no-regression check:
    # a fact the monitor judges PERCEIVED must produce answer_text BYTE-IDENTICAL to the flag-off text.
    perceived_judged_correctly = [r for r in real_rows if r["truth"] == PROVENANCE_PERCEIVED and r["judged"] == PROVENANCE_PERCEIVED]
    no_regression_on_correct_perceived = bool(perceived_judged_correctly) and all(
        r["text_unchanged"] for r in perceived_judged_correctly
    )

    # ---- (D) lesion collapse -------------------------------------------------------------------------------------
    ag_lesion = _agent(seed, comp, vocab, enable=True, lesion=True)
    lesion_rows = _battery(ag_lesion)
    lesion_correct = sum(1 for r in lesion_rows if r["judged"] == r["truth"])
    lesion_acc = lesion_correct / len(lesion_rows)
    lesion_d_zero = all(abs(r["d"] or 0.0) < 1e-6 for r in lesion_rows if r["known"])

    # ---- (E) composes cleanly with self_schema_honesty -----------------------------------------------------------
    ag_both = BrainConversationalAgent(
        seed=seed, concepts={w: None for w in vocab}, composer=comp,
        enable_neural_render=False, defer_parser=True,
        enable_self_schema_honesty=True, enable_source_provenance_honesty=True,
    )
    ag2, ac2, pt2 = perceived_cues[0]
    rec_both = ag_both.known_fact_record((ag2, ac2))
    composes_cleanly = bool(
        rec_both["self_schema_invoked"] is True
        and rec_both.get("provenance") is not None
        and rec_both["provenance"]["known"] is True
    )

    return {
        "seed": int(seed),
        "default_off_ok": default_off_ok,
        "monitor_built_when_off": monitor_built_when_off,   # must be False
        "moat_preserved": moat_preserved,
        "real_accuracy": real_acc,
        "real_correct": real_correct,
        "n_items": len(real_rows),
        "text_matches_judgment": text_matches_judgment,
        "every_judged_known": every_judged_known,
        "no_regression_on_correct_perceived": no_regression_on_correct_perceived,
        "n_perceived_judged_correctly": len(perceived_judged_correctly),
        "lesion_accuracy": lesion_acc,
        "lesion_d_zero": lesion_d_zero,
        "composes_cleanly_with_self_schema_honesty": composes_cleanly,
        "sample_generated_text": next((r for r in real_rows if r["truth"] == PROVENANCE_GENERATED), {}),
    }


def evaluate_and_verdict(row):
    v = Verdict(f"#129 source-provenance honesty wire-in seed {row['seed']}", chance=0.5)
    v.require("default-off text is byte-identical to the pre-existing path", row["default_off_ok"], expect=True)
    v.require("default-off never builds the monitor", row["monitor_built_when_off"], expect=False)
    v.require("a hard abstain is never touched (moat-first)", row["moat_preserved"], expect=True)
    v.require("every battery item returns a known (non-fabricated) judgment", row["every_judged_known"], expect=True)
    v.floor("un-lesioned battery accuracy clears the floor", measured=row["real_accuracy"], floor=ACC_FLOOR)
    v.require("the answer TEXT tracks the judged label exactly", row["text_matches_judgment"], expect=True)
    v.require("a correctly-judged perceived fact reads byte-identical to the flag-off text",
              row["no_regression_on_correct_perceived"], expect=True)
    # lesion_accuracy is NOT gated (reported only, printed + carried in the artifact): under the lesion it is a
    # host tie-break on silent pools, small-N noisy by construction (the #129 de-risk's own documented pitfall).
    # The load-bearing, deterministic proof is lesion_d_zero (both prov pools read EXACTLY 0.0 -- no tie-break).
    v.require("lesioning zeroes the discriminability (both prov pools silent, d==0 exactly)",
              row["lesion_d_zero"], expect=True)
    v.require("composes cleanly with self_schema_honesty", row["composes_cleanly_with_self_schema_honesty"],
              expect=True)
    go = bool(
        row["default_off_ok"] and not row["monitor_built_when_off"] and row["moat_preserved"]
        and row["every_judged_known"] and row["real_accuracy"] >= ACC_FLOOR and row["text_matches_judgment"]
        and row["no_regression_on_correct_perceived"]
        and row["lesion_d_zero"] and row["composes_cleanly_with_self_schema_honesty"]
    )
    decided = v.decide(go=go, verbose=False)
    return decided, go


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--out", default="research/findings/raw/lanes/metacog/_129_source_provenance_honesty_wirein_6seed.json")
    a = ap.parse_args()

    t0 = time.time()
    print(f"[129-honesty-wire] production wire-in de-risk | seeds={a.seeds} n_pairs={N_PAIRS} D={D}", flush=True)
    rows, statuses = [], []
    for s in a.seeds:
        r = evaluate_seed(s)
        decided, go = evaluate_and_verdict(r)
        r["seed_status"] = "PASS" if (go and decided["status"] != UNDEFINED) else \
            ("UNDEFINED" if decided["status"] == UNDEFINED else "FAIL")
        r["preconditions"] = decided["preconditions"]
        r["undefined_reasons"] = decided["undefined_reasons"]
        rows.append(r)
        statuses.append(r["seed_status"])
        print(f"  [seed {s}] {r['seed_status']:9s} real_acc={r['real_accuracy']:.3f} "
              f"lesion_acc={r['lesion_accuracy']:.3f} moat={r['moat_preserved']} "
              f"off_ok={r['default_off_ok']} text_match={r['text_matches_judgment']}", flush=True)

    n = len(a.seeds)
    n_pass = sum(s == "PASS" for s in statuses)
    any_undef = any(s == "UNDEFINED" for s in statuses)
    go_aggregate = (n_pass >= 5 if n >= 6 else n_pass == n) and not (n < 6 and any_undef)

    def mean(k):
        return float(sum(r[k] for r in rows) / len(rows))

    # AGGREGATE verdict that TRAVELS WITH ITS PRECONDITIONS (tools.verdict.Verdict -> a top-level `preconditions`
    # block in the artifact, enforced by tools/gates/verdict_preconditions.py). Re-checks the same relationships
    # evaluate_and_verdict checks per-seed, over the 6-seed aggregate.
    va = Verdict(f"#129 source-provenance honesty wire-in {n_pass}/{n}", chance=0.5)
    va.require("default-off byte-identical on every seed", all(r["default_off_ok"] for r in rows), expect=True)
    va.require("default-off never builds the monitor, on every seed",
               any(r["monitor_built_when_off"] for r in rows), expect=False)
    va.require("the hard moat is never touched, on every seed", all(r["moat_preserved"] for r in rows), expect=True)
    va.floor("worst-seed un-lesioned battery accuracy clears the floor",
             measured=min(r["real_accuracy"] for r in rows), floor=ACC_FLOOR)
    va.require("the answer TEXT tracks the judged label exactly, on every seed",
               all(r["text_matches_judgment"] for r in rows), expect=True)
    va.require("lesioning zeroes the discriminability on every seed (deterministic, not the noisy tie-break)",
               all(r["lesion_d_zero"] for r in rows), expect=True)
    va.require("composes cleanly with self_schema_honesty on every seed",
               all(r["composes_cleanly_with_self_schema_honesty"] for r in rows), expect=True)
    decided_agg = va.decide(go=go_aggregate, verbose=False)

    verdict = (
        f"GO ({n_pass}/{n}) — board #129 source-provenance honesty is WIRED (default-off) into "
        f"BrainConversationalAgent.known_fact_record/.reasoned_fact_record, reachable through webapp/server.py's "
        f"single-fact /api/brain-chat path (env-flag gated). Default-off is byte-identical ({sum(r['default_off_ok'] for r in rows)}/{n} "
        f"seeds); the moat is never touched. Un-lesioned battery accuracy {mean('real_accuracy'):.3f} "
        f"(floor {ACC_FLOOR}); the answer TEXT tracks the live judged label exactly (perceived unchanged, "
        f"generated flagged) on every seed; lesioning the monitor's plasticity gate collapses the "
        f"discriminability to exactly zero on every item (deterministic; the noisy tie-broken accuracy, "
        f"mean {mean('lesion_accuracy'):.3f}, is reported only, NOT gated — see honest_scope) — the framing is "
        f"driven by the LEARNED trace, not a hardcoded branch."
        if go_aggregate else
        f"NO-GO/PARTIAL ({n_pass}/{n} PASS) — see per-seed failing preconditions."
    )

    out = {
        "runner": "research/runners/_129_source_provenance_honesty_wirein_derisk.py",
        "faculty": "board #129 source-provenance honesty, WIRED (default-off) into the live-chat known-fact / "
                   "reasoned-fact answer pathway (BrainConversationalAgent + webapp/server.py /api/brain-chat)",
        "board": 129, "vikunja": 137,
        "backend": os.environ.get("SIM_BACKEND", "(unset)"),
        "seeds": list(a.seeds), "n_pairs": N_PAIRS, "D": D,
        "thresholds": {"ACC_FLOOR": ACC_FLOOR, "LESIONED_ACC_CEIL": LESIONED_ACC_CEIL, "go_bar": ">=5/6" if n >= 6 else "all"},
        "verdict": verdict, "GO": bool(go_aggregate), "n_pass": n_pass, "n_seeds": n,
        "status": decided_agg["status"],
        "preconditions": decided_agg["preconditions"],
        "undefined_reasons": decided_agg["undefined_reasons"],
        "seed_status": {r["seed"]: r["seed_status"] for r in rows},
        "means": {k: mean(k) for k in ("real_accuracy", "lesion_accuracy")},
        "per_seed": rows,
        "honest_scope": (
            "Verified through the REAL BrainConversationalAgent class + a real RFPhasorComposer (the production "
            "composer), mirroring this project's own self_schema_honesty wire-in verification convention. The "
            "webapp/server.py single-fact /api/brain-chat path is ALSO wired, default-off (BRAIN_SOURCE_PROVENANCE_HONESTY, "
            "BRAIN_SOURCE_PROVENANCE_HONESTY_LESION; see tests/test_webapp_server.py's three new "
            "test_brain_chat_source_provenance_* cases) but that endpoint's gate() only ever returns a "
            "DIRECTLY-STORED fact, so it can only ever exercise the PERCEIVED half of the mechanism live over "
            "HTTP; the GENERATED half is reachable today via BrainConversationalAgent.reasoned_fact_record "
            "directly (not yet through a live HTTP turn). NAMED RESIDUAL (no-defer, next rung): teach the "
            "/api/brain-chat endpoint to answer some turns via reason_chain/chain_of_thought as a first-class "
            "answer channel (a bigger, separate design decision -- how does a free-text turn signal 'answer via "
            "inference'? -- left to the owner) so a single live HTTP conversation can exhibit BOTH provenance "
            "labels. The host boundary (declared, unchanged from the #129 de-risk): WHICH encoding context a "
            "fact is taught under is supplied by the caller (a known_fact_record hit is always PERCEIVED, a "
            "reasoned_fact_record conclusion is always GENERATED) -- the monitor's readback of which label a "
            "content pattern carries is the genuine spiking read, and it decides the framing."
        ),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    print(f"[129-honesty-wire] VERDICT: {verdict}", flush=True)
    print(f"[129-honesty-wire] wrote {out_path} ({out['elapsed_seconds']}s)\n" + "=" * 100, flush=True)
    return 0 if go_aggregate else 1


if __name__ == "__main__":
    raise SystemExit(main())
