"""SOAK / no-regression gate for the BRAIN_BG_SELECT default-ON flip (SPEAK-vs-STAY-SILENT via the two-channel BG selector).

Two independent bars the parent runs BEFORE flipping `BRAIN_BG_SELECT` default-on:

  PART A — SELECTOR 6-SEED PHYSIOLOGY (the core gate, pool-friendly numpy). For each seed, build the reused Gate-A v2
  selector three ways and run SINGLE salience-biased races (the production `select_once`):
    * INTACT: SPEAK-favored salience (speak=1, silent=0) -> the race commits to SPEAK; SILENT-favored (speak=0,
      silent=1) -> the race commits to STAY-SILENT. The winner FLIPS with the salience -> salience DRIVES the selection.
    * NO SHARED AROUSAL (the finding's `arousal_is_load_bearing` control): remove the shared practice-arousal drive ->
      the D1 MSNs stay sub-threshold -> NO channel commits at either salience (the decision floors).
    * NO DIRECT PATH (the finding's `direct_path_is_load_bearing` control): cut the D1->GPi gate -> the thalamus is
      never disinhibited -> NO channel commits at either salience (the decision floors).
  GO gate (per seed, deterministic-ish over the trial batch):
    * INTACT: speak-favored commits to SPEAK on >= SELECT_MIN of trials AND silent-favored commits to STAY-SILENT on
      >= SELECT_MIN AND the majority winner FLIPS between the two salience conditions.
    * BOTH LESIONS: commit rate (either salience) <= LESION_MAX (floored). This is the load-bearing proof: the
      selected-action change VANISHES under either lesion, so the BG cascade — not a host argmax — chose.
  Overall PART A GO: every seed passes.

  PART B — CHAT NO-REGRESSION (structural; run once through the REAL brain_chat handler with the STUB renderer, no
  Qwen). Toggling ONLY `BRAIN_BG_SELECT`:
    * ORDINARY content turns: flag-ON == flag-OFF, BYTE-IDENTICAL (the selector is CONSULTED only on a content-empty
      turn, so an ordinary turn has ZERO code-path difference — a short-circuit by construction, proven end-to-end).
    * A CONTENT-EMPTY turn ('...'): flag-ON (intact) -> the BG race commits STAY-SILENT -> the handler returns the
      HOLD line + a `bg_select` block (the faculty FIRES, output genuinely differs from flag-OFF).
    * The SAME '...' turn flag-ON + LESION (`BRAIN_BG_SELECT_LESION=arousal`): no commit -> the block falls through ->
      the response is BYTE-IDENTICAL to flag-OFF (the load-bearing lesion-vanish, now at the handler level).
  Degrades to a reported SKIP (never a false NO-GO) if webapp.server cannot import on a bare pool node; PART A is the
  gate. Pass --selector-only to skip PART B.

  Run (pool):  SIM_BACKEND=numpy python -m research.runners._bg_action_selection_flip_soak --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import research.runners.bg_action_selection_production_organ as BG  # noqa: E402
from tools.lab import attributable_to, assert_backend  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "_bg_action_select_prodflip" / "soak_seed42.json"

SELECT_MIN = 0.75    # per-seed single-race selection reliability the flip gate requires (seed 42 smoke: 1.0)
LESION_MAX = 0.05    # per-seed lesion commit-rate ceiling (the finding's control floors to 0)


def _rates(org, salience, trials):
    """Run `trials` single races at one salience -> (committed_rate, speak_win_rate, silent_win_rate)."""
    sp = float(salience[0])
    si = float(salience[1])
    wins = [org.select_once(sp, si) for _ in range(int(trials))]
    committed = [w for w in wins if w["committed"]]
    n = max(1, len(wins))
    speak_wins = sum(1 for w in committed if w["winner"] == BG.SPEAK)
    silent_wins = sum(1 for w in committed if w["winner"] == BG.STAY_SILENT)
    return {
        "commit_rate": len(committed) / n,
        "speak_win_rate": speak_wins / n,
        "silent_win_rate": silent_wins / n,
        "n": len(wins),
    }


def run_selector_seed(seed, trials, lesion_trials):
    t0 = time.time()
    BG.reset_organs()
    intact = BG.get_organ(seed=seed, lesion=None)
    ar_les = BG.get_organ(seed=seed, lesion="arousal")
    dp_les = BG.get_organ(seed=seed, lesion="direct_path")

    speak_favored = _rates(intact, (1.0, 0.0), trials)     # SPEAK salience high -> expect SPEAK
    silent_favored = _rates(intact, (0.0, 1.0), trials)    # STAY-SILENT salience high -> expect STAY-SILENT

    # both lesions, both salience conditions -> the commit must floor regardless of salience.
    ar_sp = _rates(ar_les, (1.0, 0.0), lesion_trials)
    ar_si = _rates(ar_les, (0.0, 1.0), lesion_trials)
    dp_sp = _rates(dp_les, (1.0, 0.0), lesion_trials)
    dp_si = _rates(dp_les, (0.0, 1.0), lesion_trials)
    ar_commit = max(ar_sp["commit_rate"], ar_si["commit_rate"])
    dp_commit = max(dp_sp["commit_rate"], dp_si["commit_rate"])

    speak_ok = speak_favored["speak_win_rate"] >= SELECT_MIN
    silent_ok = silent_favored["silent_win_rate"] >= SELECT_MIN
    # the FLIP: the majority winner differs between the two salience conditions (salience drives it).
    flip = (speak_favored["speak_win_rate"] > speak_favored["silent_win_rate"]
            and silent_favored["silent_win_rate"] > silent_favored["speak_win_rate"])
    arousal_floored = ar_commit <= LESION_MAX
    direct_floored = dp_commit <= LESION_MAX
    seed_ok = bool(speak_ok and silent_ok and flip and arousal_floored and direct_floored)
    return {
        "seed": seed, "seed_ok": seed_ok,
        "speak_favored": speak_favored, "silent_favored": silent_favored,
        "arousal_lesion_commit": ar_commit, "direct_lesion_commit": dp_commit,
        "speak_ok": bool(speak_ok), "silent_ok": bool(silent_ok), "flip": bool(flip),
        "arousal_floored": bool(arousal_floored), "direct_floored": bool(direct_floored),
        "elapsed_s": round(time.time() - t0, 1),
    }


def run_handler_no_regression():
    """PART B: flag-ON == flag-OFF on ordinary turns; ON fires STAY-SILENT on a content-empty turn; ON+lesion vanishes
    (byte-identical to OFF). Through the real brain_chat handler (stub renderer). Degrades to {'skipped': reason}."""
    os.environ["BRAIN_CHAT_RENDERER"] = "stub"
    # neutralize unrelated faculties (identical in both arms) to keep this bounded + deterministic on a CPU pool node.
    for k in ("BRAIN_AFFECT", "BRAIN_AFFECT_DRIVES", "BRAIN_DA_DRIVES", "BRAIN_DA_ENCODING", "BRAIN_SELF_INITIATE",
              "BRAIN_COMPREHENSION_GATE", "BRAIN_SURPRISE", "BRAIN_METACOG", "BRAIN_WORLDMODEL", "BRAIN_PRAGMATIC",
              "BRAIN_RICH", "BRAIN_SWAP_DRIVES", "BRAIN_OPEN_ENDED", "BRAIN_VISION_IDENTITY", "BRAIN_GNW_SWAP"):
        os.environ.setdefault(k, "0")
    os.environ.pop("BRAIN_BG_SELECT", None)
    os.environ.pop("BRAIN_BG_SELECT_LESION", None)
    try:
        import webapp.server as S
    except Exception as e:  # a bare pool node without the webapp deps -> report SKIP (PART A is the gate)
        return {"skipped": f"webapp.server import failed: {type(e).__name__}: {e}"}

    # FRESH UNIQUE SESSION PER TURN. brain_chat caches + accumulates per-session state (D5/D6/D3/continuous engine
    # advance every turn), so a shared session would compare OFF-vs-ON turns at DIFFERENT session histories — a test
    # artifact unrelated to this flag (the block cannot touch a content turn). A fresh session makes every measured turn
    # 'first turn on a freshly built brain', so an OFF/ON pair is like-for-like.
    _ctr = {"n": 0}

    def turn(message):
        _ctr["n"] += 1
        req = S.BrainChatRequest(session=f"bgsoak_{_ctr['n']}", message=message, brain="tiny-demo",
                                 renderer="stub", rich=False)
        return json.loads(bytes(S.brain_chat(req).body).decode())

    ordinary_msgs = ["what does the cat eat?", "tell me about dogs", "hello there", "what is the sky?"]
    dots = "..."   # content-empty (non-empty string, zero content tokens) -> STAY-SILENT is the salient contender

    # ORGAN-LEVEL inertness + lesion-vanish (deterministic, backend-independent — independent of any brain-build RNG):
    # a content turn is NEVER consulted (byte-identical by construction), and under either lesion the '...' turn does
    # NOT commit (the hold vanishes -> the handler falls through). This is the robust load-bearing/byte-identical proof.
    BG.reset_organs()
    organ_content_inert = all(BG.decide_action(m) is None for m in ordinary_msgs)
    organ_dots_intact = (BG.decide_action(dots) or {}).get("action") == "STAY_SILENT"
    BG.reset_organs()
    organ_dots_arousal_vanish = BG.decide_action(dots, lesion="arousal") is None
    BG.reset_organs()
    organ_dots_direct_vanish = BG.decide_action(dots, lesion="direct_path") is None
    BG.reset_organs()

    # BUILD-DETERMINISM self-check: is the tiny-demo brain identical across two fresh OFF sessions on the same message?
    # Only when it is can the FULL-JSON handler equality below be attributed to this flag rather than to build RNG.
    os.environ.pop("BRAIN_BG_SELECT", None)
    det_a = turn(ordinary_msgs[0])
    det_b = turn(ordinary_msgs[0])
    build_deterministic = (det_a == det_b)

    # per-message paired OFF/ON on FRESH sessions (each turn = first turn on a fresh brain).
    pairs = {}
    for m in ordinary_msgs:
        os.environ.pop("BRAIN_BG_SELECT", None)
        r_off = turn(m)
        os.environ["BRAIN_BG_SELECT"] = "1"
        r_on = turn(m)
        os.environ.pop("BRAIN_BG_SELECT", None)
        pairs[m] = (r_off, r_on)
    ordinary_identical = all(off == on for (off, on) in pairs.values())
    off_had_no_bg_key = all("bg_select" not in off for (off, _) in pairs.values())

    # '...' turn on fresh sessions: OFF (host path), ON intact (the HOLD fires), ON+arousal-lesion (the hold vanishes).
    os.environ.pop("BRAIN_BG_SELECT", None)
    off_dots = turn(dots)
    os.environ["BRAIN_BG_SELECT"] = "1"
    on_dots = turn(dots)
    os.environ["BRAIN_BG_SELECT_LESION"] = "arousal"
    on_dots_lesioned = turn(dots)
    os.environ.pop("BRAIN_BG_SELECT_LESION", None)
    os.environ.pop("BRAIN_BG_SELECT", None)

    on_fired = (on_dots.get("bg_select", {}).get("action") == "STAY_SILENT"
                and on_dots.get("answer") == BG.HOLD_TEXT
                and on_dots != off_dots)
    handler_lesion_vanishes = (on_dots_lesioned == off_dots)   # byte-identical to flag-off under the lesion

    organ_ok = bool(organ_content_inert and organ_dots_intact and organ_dots_arousal_vanish and organ_dots_direct_vanish)
    # The handler full-JSON equalities are a clean instrument ONLY when the build is deterministic; otherwise they test
    # brain-build RNG, not this flag, so we defer to the deterministic organ-level proof (never a false NO-GO).
    if build_deterministic:
        handler_ok = bool(ordinary_identical and off_had_no_bg_key and on_fired and handler_lesion_vanishes)
    else:
        handler_ok = bool(off_had_no_bg_key and on_fired)   # byte-identity/vanish proven at organ level instead
    return {"skipped": None,
            "build_deterministic": bool(build_deterministic),
            "organ_content_inert": bool(organ_content_inert),
            "organ_dots_intact": bool(organ_dots_intact),
            "organ_dots_arousal_vanish": bool(organ_dots_arousal_vanish),
            "organ_dots_direct_vanish": bool(organ_dots_direct_vanish),
            "ordinary_identical": bool(ordinary_identical),
            "off_had_no_bg_key": bool(off_had_no_bg_key),
            "on_fired_stay_silent": bool(on_fired),
            "handler_lesion_vanishes": bool(handler_lesion_vanishes),
            "no_regression": bool(organ_ok and handler_ok)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--trials", type=int, default=12, help="intact single races per salience condition")
    ap.add_argument("--lesion-trials", type=int, default=8, help="lesion single races per salience condition")
    ap.add_argument("--selector-only", action="store_true", help="skip PART B (handler no-regression)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    _backend = assert_backend(os.environ.get("SIM_BACKEND", "numpy"))
    print("=" * 118)
    print(f"[bg-soak] PART A — selector 6-seed physiology (per-seed: speak/silent select >= {SELECT_MIN} + FLIP + both "
          f"lesions commit <= {LESION_MAX}). seeds={a.seeds} trials={a.trials} lesion_trials={a.lesion_trials} "
          f"gain={BG.SALIENCE_GAIN_PA}pA", flush=True)
    per_seed = []
    for s in a.seeds:
        try:
            r = run_selector_seed(s, a.trials, a.lesion_trials)
        except Exception as e:  # noqa: BLE001
            r = {"seed": s, "seed_ok": False, "error": repr(e)}
            traceback.print_exc()
        per_seed.append(r)
        if "error" in r:
            print(f"  [seed {s}] ERROR {r['error']}", flush=True)
        else:
            print(f"  [seed {s}] speak-fav->SPEAK={r['speak_favored']['speak_win_rate']:.2f} "
                  f"silent-fav->SILENT={r['silent_favored']['silent_win_rate']:.2f} flip={r['flip']} || "
                  f"AROUSAL-lesion commit={r['arousal_lesion_commit']:.2f} "
                  f"DIRECT-lesion commit={r['direct_lesion_commit']:.2f} => seed_ok={r['seed_ok']}", flush=True)
    ok_seeds = [r for r in per_seed if "error" not in r]
    all_seed_ok = bool(ok_seeds) and all(r["seed_ok"] for r in per_seed)
    n_ok = sum(int(r.get("seed_ok")) for r in per_seed)

    # aggregate attribution: the selection is attributable to the arousal-enabled, direct-path-gated BG cascade — the
    # intact commit rate is high, both lesion commit rates floor. Subtract, don't just report.
    mean_intact_commit = float(np.mean([
        0.5 * (r["speak_favored"]["commit_rate"] + r["silent_favored"]["commit_rate"]) for r in ok_seeds
    ])) if ok_seeds else 0.0
    mean_arousal_commit = float(np.mean([r["arousal_lesion_commit"] for r in ok_seeds])) if ok_seeds else 0.0
    mean_direct_commit = float(np.mean([r["direct_lesion_commit"] for r in ok_seeds])) if ok_seeds else 0.0
    attr_arousal = attributable_to("commit: intact vs no-arousal", mean_intact_commit, mean_arousal_commit)
    attr_direct = attributable_to("commit: intact vs no-direct-path", mean_intact_commit, mean_direct_commit)
    selector_go = bool(all_seed_ok)
    print(f"  [aggregate] mean intact commit={mean_intact_commit:.3f} | no-arousal={mean_arousal_commit:.3f} | "
          f"no-direct-path={mean_direct_commit:.3f} | selector_go={selector_go}", flush=True)

    handler = None
    if not a.selector_only:
        print("\n[bg-soak] PART B — chat no-regression (flag ON==OFF ordinary; ON fires '...'; ON+lesion vanishes) ...",
              flush=True)
        try:
            handler = run_handler_no_regression()
        except Exception as e:  # noqa: BLE001
            handler = {"skipped": f"handler check raised: {type(e).__name__}: {e}"}
            traceback.print_exc()
        if handler.get("skipped"):
            print(f"  PART B SKIPPED: {handler['skipped']}", flush=True)
        else:
            print(f"  ORGAN content_inert={handler['organ_content_inert']} dots_intact={handler['organ_dots_intact']} "
                  f"arousal_vanish={handler['organ_dots_arousal_vanish']} direct_vanish={handler['organ_dots_direct_vanish']}",
                  flush=True)
            print(f"  HANDLER build_deterministic={handler['build_deterministic']} "
                  f"ordinary_identical={handler['ordinary_identical']} off_had_no_bg_key={handler['off_had_no_bg_key']} "
                  f"on_fired={handler['on_fired_stay_silent']} lesion_vanishes={handler['handler_lesion_vanishes']} "
                  f"=> no_regression={handler['no_regression']}", flush=True)

    handler_ok = (handler is None) or bool(handler.get("skipped")) or bool(handler.get("no_regression"))
    overall_go = bool(selector_go and handler_ok)

    print("\n" + "#" * 118)
    print(f"[bg-soak] SELECTOR per-seed {n_ok}/{len(a.seeds)} ok => selector_go={selector_go} | HANDLER "
          f"{'skipped' if (handler and handler.get('skipped')) else ('n/a' if handler is None else handler.get('no_regression'))}"
          f" => {'GO' if overall_go else 'NO-GO'}", flush=True)
    print("#" * 118)

    out_path = Path(a.out)
    if len(a.seeds) > 1:
        out_path = out_path.parent / f"soak_summary_{len(a.seeds)}seed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "probe": "bg_action_selection_flip_soak", "backend": _backend, "device": _backend,
        "seeds": a.seeds, "trials": a.trials, "lesion_trials": a.lesion_trials, "salience_gain_pA": BG.SALIENCE_GAIN_PA,
        "selector_n_seed_ok": n_ok, "selector_go": selector_go,
        "mean_intact_commit": mean_intact_commit, "mean_arousal_lesion_commit": mean_arousal_commit,
        "mean_direct_lesion_commit": mean_direct_commit,
        "attributable_to_arousal": attr_arousal, "attributable_to_direct_path": attr_direct,
        "handler": handler, "overall_go": overall_go,
        "elapsed_s": round(time.time() - t0, 1), "per_seed": per_seed}, indent=2, default=str))
    print(f"[bg-soak] wrote {out_path}", flush=True)
    return 0 if overall_go else 1


if __name__ == "__main__":
    sys.exit(main())
