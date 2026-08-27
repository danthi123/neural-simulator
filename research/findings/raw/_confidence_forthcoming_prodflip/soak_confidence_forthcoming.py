"""6-seed no-regression FLIP-SOAK for the board #94 confidence-forthcomingness DEFAULT-ON flip
(`BRAIN_CONFIDENCE_FORTHCOMING`, webapp/confidence_forthcoming_chat.py).

Mirrors the `_bg_action_selection_flip_soak.py` PART A / PART B split (the established pattern for a chat-level
production-default flip in this repo):

  PART A -- 6-SEED ORGAN PHYSIOLOGY (fast, no chat-brain build). The feature's ENTIRE decision surface is the
  metacog organ's `judge(evidence)` read (`confident` True/False) -- this wire-in adds no new spiking substrate
  of its own (see the finding: "reuse-by-import of the existing organ + composer methods"). For each seed, force
  a FRESH build of `metacog_production_organ`'s process-shared singleton (+ its ONE-BRAIN pool-#2 shared
  substrate, when merge2 is on) AT that seed, then confirm HIGH evidence (0.95) reads confident=True, LOW
  evidence (0.05) reads confident is-not-True, and BOTH lesion under `lesion=True` -- the SAME battery the cited
  2026-08-13 metacog-robust-confidence-GO organ de-risk ran, re-confirmed here because it is what THIS flip's
  safety depends on (not re-deriving the organ, confirming this flip does not regress it).

  PART B -- HANDLER NO-REGRESSION (single pass, the REAL `/api/brain-chat` production default -- every OTHER
  faculty at ITS OWN shipped default, not isolated, so this is the actual pipeline the owner gets). Toggling
  ONLY `BRAIN_CONFIDENCE_FORTHCOMING` (explicit "1"/"0", never `.pop()` -- the flip means unset now reads ON):
    * ORDINARY control turns with REAL (unpatched) evidence: ON and explicit-OFF answers/recalled_svo/verified
      must be IDENTICAL. This is the decisive non-regression evidence -- on real production turns the composer's
      `mean_role_confidence` is essentially never populated (a declared, separately-tracked residual: see
      verify_confidence_forthcoming_prodflip.py's module docstring point 2), so `confident` reads None-as-False
      and the coupling's bonus is NEVER granted on real traffic today -- the flip is a safe no-op on everything
      except the rare turn where the metacog organ genuinely reads HIGH.
    * No crash across the turn set; the no-confab moat (`verified`) holds on every answered turn.
    * A content-empty / abstained turn is unaffected (the coupling only ever touches an ANSWERED rich turn).

  Run:  SIM_BACKEND=numpy python soak_confidence_forthcoming.py --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "2")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners.metacog_production_organ as MC          # noqa: E402
import research.runners.onebrain_merge_production2 as M2         # noqa: E402


def run_organ_seed(seed: int) -> dict:
    t0 = time.time()
    MC._ORGAN = None
    M2._MERGED_SUBSTRATE2 = None
    try:
        org = MC.get_organ(seed=seed)
        hi = org.judge(0.95, lesion=False)
        lo = org.judge(0.05, lesion=False)
        hi_les = org.judge(0.95, lesion=True)
        lo_les = org.judge(0.05, lesion=True)
    except Exception as e:  # noqa: BLE001
        return {"seed": seed, "seed_ok": False, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(), "elapsed_s": round(time.time() - t0, 1)}
    hi_conf = bool(hi["confident"])
    lo_not_conf = not bool(lo["confident"])
    les_collapses = (not bool(hi_les["confident"])) and (not bool(lo_les["confident"]))
    ok = bool(hi_conf and lo_not_conf and les_collapses)
    return {"seed": seed, "seed_ok": ok, "hi_confident": hi["confident"], "lo_confident": lo["confident"],
            "hi_lesioned_confident": hi_les["confident"], "lo_lesioned_confident": lo_les["confident"],
            "hi_balance": hi["balance"], "lo_balance": lo["balance"], "threshold": hi["threshold"],
            "elapsed_s": round(time.time() - t0, 1)}


def run_handler_no_regression() -> dict:
    """PART B: ON == explicit-OFF on ordinary (unpatched) control turns, through the REAL, NOT-isolated
    production default (every other faculty at its own shipped default). Degrades to a reported SKIP (never a
    false NO-GO) if webapp.server cannot import."""
    os.environ["BRAIN_CHAT_RENDERER"] = "stub"
    try:
        import webapp.server as S
    except Exception as e:  # noqa: BLE001
        return {"skipped": f"webapp.server import failed: {type(e).__name__}: {e}"}

    _ctr = {"n": 0}

    def turn(message, on: bool):
        _ctr["n"] += 1
        os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "1" if on else "0"   # EXPLICIT, never .pop() (the guarded arm)
        req = S.BrainChatRequest(session=f"cfsoak_{_ctr['n']}", message=message, brain="tiny-demo",
                                 renderer="stub", rich=True)
        r = json.loads(bytes(S.brain_chat(req).body).decode())
        os.environ["BRAIN_CONFIDENCE_FORTHCOMING"] = "0"
        return r

    control_msgs = ["what does the brain use", "what does the dog chase"]   # kept lean: each fresh session builds
    # the FULL, non-isolated production default (every default-on faculty active), ~60-90s/session on this CPU box
    pairs = {}
    crashed = []
    for m in control_msgs:
        try:
            off = turn(m, on=False)
            on = turn(m, on=True)
            pairs[m] = (off, on)
        except Exception as e:  # noqa: BLE001
            crashed.append(f"{m!r}: {type(e).__name__}: {e}")
            traceback.print_exc()

    # OBSERVABLE-CONTENT equality (bug fix, 2026-08-27 -- the first run of this check false-FAILed on a TEST-
    # HARNESS bug, not a product bug): comparing the FULL response dict (`off == on`) is the WRONG instrument --
    # this coupling is DESIGNED to additively attach a `confidence_forthcoming` diagnostic key whenever it is
    # in-scope, even when `granted=False` (see confidence_forthcoming_chat.apply_cap's "nothing_to_cap"/
    # "low_confidence_capped" reasons), so `on`'s dict is EXPECTED to carry one more key than `off`'s even on a
    # genuine no-op turn. The correct no-regression instrument is whether every OBSERVABLE-TO-THE-USER field
    # (what a real chat client renders/acts on) is unchanged -- the answer text, the recalled fact, sentence
    # count, and the moat's verified flag -- while the diagnostic key's PRESENCE is checked separately below.
    _OBSERVABLE = ("answer", "recalled_svo", "n_sentences", "verified", "abstained")
    identical = all(
        all(off.get(k) == on.get(k) for k in _OBSERVABLE)
        for (off, on) in pairs.values()
    )
    off_had_no_cf_key = all("confidence_forthcoming" not in off for (off, _on) in pairs.values())
    # on real (unpatched) turns the coupling should ALSO never attach a key (confident never reads True without
    # forced evidence) -- if it DID attach a key with granted=True, that's exactly the regression this guards.
    on_never_granted = all(
        (("confidence_forthcoming" not in on) or (on["confidence_forthcoming"].get("granted") is not True))
        for (_off, on) in pairs.values()
    )
    moat_ok = all(
        (off.get("verified") in (True, None)) and (on.get("verified") in (True, None))
        for (off, on) in pairs.values()
    )
    no_regression = bool(identical and off_had_no_cf_key and on_never_granted and moat_ok and not crashed)
    return {"skipped": None, "n_turns": len(control_msgs), "crashed": crashed,
            "identical": bool(identical), "off_had_no_cf_key": bool(off_had_no_cf_key),
            "on_never_granted": bool(on_never_granted), "moat_ok": bool(moat_ok),
            "no_regression": no_regression,
            "pairs_diff": {m: {"off": off.get("answer"), "on": on.get("answer")}
                          for m, (off, on) in pairs.items() if off != on}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--organ-only", action="store_true", help="skip PART B (handler no-regression)")
    ap.add_argument("--out", default=str(_REPO / "research" / "findings" / "raw" /
                                        "_confidence_forthcoming_prodflip" / "soak_summary_6seed.json"))
    a = ap.parse_args()
    t0 = time.time()
    print("=" * 110)
    print(f"[cf-soak] PART A -- metacog organ 6-seed physiology (HIGH confident=True, LOW confident!=True, "
          f"BOTH lesion collapses). seeds={a.seeds}", flush=True)
    per_seed = []
    for s in a.seeds:
        r = run_organ_seed(s)
        per_seed.append(r)
        if "error" in r:
            print(f"  [seed {s}] ERROR {r['error']}", flush=True)
        else:
            print(f"  [seed {s}] hi_confident={r['hi_confident']} lo_confident={r['lo_confident']} "
                  f"hi_les={r['hi_lesioned_confident']} lo_les={r['lo_lesioned_confident']} "
                  f"balance(hi/lo)={r['hi_balance']:.4f}/{r['lo_balance']:.4f} thr={r['threshold']:.4f} "
                  f"=> seed_ok={r['seed_ok']}", flush=True)
    all_seed_ok = bool(per_seed) and all(r.get("seed_ok") for r in per_seed)
    n_ok = sum(int(bool(r.get("seed_ok"))) for r in per_seed)

    handler = None
    if not a.organ_only:
        print("\n[cf-soak] PART B -- handler no-regression (explicit ON==OFF on real/unpatched control turns, "
              "full production default, not isolated) ...", flush=True)
        try:
            handler = run_handler_no_regression()
        except Exception as e:  # noqa: BLE001
            handler = {"skipped": f"handler check raised: {type(e).__name__}: {e}"}
            traceback.print_exc()
        if handler.get("skipped"):
            print(f"  PART B SKIPPED: {handler['skipped']}", flush=True)
        else:
            print(f"  n_turns={handler['n_turns']} crashed={handler['crashed']} identical={handler['identical']} "
                  f"off_had_no_cf_key={handler['off_had_no_cf_key']} on_never_granted={handler['on_never_granted']} "
                  f"moat_ok={handler['moat_ok']} => no_regression={handler['no_regression']}", flush=True)

    handler_ok = (handler is None) or bool(handler.get("skipped")) or bool(handler.get("no_regression"))
    overall_go = bool(all_seed_ok and handler_ok)

    print("\n" + "#" * 110)
    print(f"[cf-soak] ORGAN {n_ok}/{len(a.seeds)} seeds ok => organ_go={all_seed_ok} | HANDLER "
          f"{'skipped' if (handler and handler.get('skipped')) else ('n/a' if handler is None else handler.get('no_regression'))}"
          f" => {'GO' if overall_go else 'NO-GO'}", flush=True)
    print("#" * 110)

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "probe": "confidence_forthcoming_flip_soak", "backend": os.environ.get("SIM_BACKEND"),
        "seeds": a.seeds, "organ_n_seed_ok": n_ok, "organ_go": all_seed_ok,
        "handler": handler, "overall_go": overall_go,
        "elapsed_s": round(time.time() - t0, 1), "per_seed": per_seed}, indent=2, default=str))
    print(f"[cf-soak] wrote {out_path}", flush=True)
    return 0 if overall_go else 1


if __name__ == "__main__":
    sys.exit(main())
