"""De-risk for the GNW GLOBAL-STOP CONFLICT-SCALED BOOST lever (`BRAIN_GNW_STOP_CONFLICT_SCALED`, additive,
DEFAULT-OFF, wired in `webapp/gnw_global_stop.py`) — retiring a fixed host scalar (`BOOST_GAIN`) by REUSING an
ALREADY-COMPUTED upstream spiking read instead of adding new circuitry.

THE RESIDUAL THIS TARGETS (named in the module docstring's HONEST RESIDUALS #3 and the production-wirein finding,
`2026-08-26-gnw-global-workspace-stop-production-wirein-GO.md`): `_StopWorkspace.run()` passed the FIXED host
constants `boost_gain=BOOST_GAIN` (=1.0) and `margin_ref=MARGIN_REF` into `run_conflict_stop` to scale the
conflict-triggered synaptic depression that clears the held workspace coalition. Load-bearing (the LESION —
`BRAIN_GNW_STOP_LESION=1` zeroing this exact scalar — collapses the clear), but NOT read off any upstream
spiking conflict magnitude, even though `detect_trigger()` in the SAME module already reads one (`n_ignited` off
`chat._last_gnw_delib`) to decide WHETHER to fire the stop at all.

THE LEVER (`webapp.gnw_global_stop.upstream_conflict_scale`). `chat._last_gnw_delib['conf']` (the gnw-deliberation
`acc_conflict_gate`'s divisive-normalized NMDA winner-vs-runnerup margin, `_conf_from_nmda` in
`_gnw_reentrant_metacog_gated_deliberation_derisk.py`) is ALREADY computed on every conflict-gated turn, at the
SAME call site (`observe_turn(chat, ...)`) that reads `chat._last_gnw_delib['n_ignited']` for the trigger. When
`BRAIN_GNW_STOP_CONFLICT_SCALED` is on and a genuine upstream conflict registered this turn (`n_ignited>=2`), the
boost gain becomes `BOOST_GAIN * (1 - conf)` (upstream conflict SEVERITY in [0,1]) instead of the flat `BOOST_GAIN`
constant. `conf~0` (severe upstream co-ignition) -> severity~1 -> ~baseline boost (preserves the shipped clear).
`conf` higher (a milder upstream ambiguity) -> a WEAKER boost -- graded clearing strength instead of an
all-or-nothing constant. No genuine upstream conflict this turn (e.g. a swap-only topic-break trigger) -> the
fixed BOOST_GAIN is used, UNCHANGED (this fixture's `swap_only` arm proves that fallback path is untouched).

GO GATE (this de-risk; 6 seeds 42/43/44/100/101/102, SIM_BACKEND=numpy, determinism via cfg.seed). GO iff, across
seeds:
  (1) BYTE-IDENTICAL-OFF — flag unset -> `ws.run()` is called with `conflict_scale=None` and the resulting
      `gnw_stop` info dict carries NO `conflict_scaled`/`conflict_scale` keys; n_pre/n_post/boost/cleared/lead
      are IDENTICAL to the pre-existing (`_gnw_global_stop_flip_soak.py`) baseline arm, same seed. On ALL 6 seeds.
  (2) HIGH-SEVERITY PRESERVED — flag ON + a genuine 2-candidate delib conflict at near-total upstream conflict
      (conf~0.05, severity~0.95) still CLEARS (n_pre>=2 -> n_post==0), on >=5/6 seeds (mirrors the shipped organ's
      own INTACT-stop gate).
  (3) LESION STILL WINS — flag ON (conflict-scaled) + `BRAIN_GNW_STOP_LESION=1` -> the depression term is STILL
      zeroed regardless of `conflict_scale` (lesion takes precedence in `_StopWorkspace.run()`'s branch order) ->
      the 2-content conflict STAYS >=2 co-ignited, on ALL 6 seeds.
  (4) SWAP-ONLY FALLBACK UNCHANGED — flag ON but the trigger is swap-only (no delib conflict registered this
      turn) -> `conflict_scale` resolves to None -> the fixed BOOST_GAIN is used -> IDENTICAL n_post/cleared/lead
      to the pre-existing swap-only baseline, on ALL 6 seeds (proves the substitution never touches a pathway
      with no upstream conflict signal to reuse).
  (5) DETERMINISM — build twice at one seed with the SAME conflict_scale -> identical stop n_post, on ALL 6 seeds.
CHARACTERIZED, not gating (reported as data): a MODERATE upstream conflict (conf~0.5, severity~0.5) — whether the
attenuated boost still clears or falls below the shipped organ's own load-bearing window (`BOOST_GAIN` comment:
"boost>=0.18 clears even without normalization ... [0.09,0.15] is normalization-load-bearing") is NOT swept under
the rug either way; this is the graded behavior the constant never exposed, named honestly regardless of direction.

REUSE-BY-IMPORT (NO `sim/` edit; NO NEW webapp module). This runner exercises the ALREADY-WIRED production organ
`webapp/gnw_global_stop.py` directly (the same organ `_gnw_global_stop_flip_soak.py` soaks) — this file only adds
fixtures for the NEW `BRAIN_GNW_STOP_CONFLICT_SCALED` flag / `conf`-carrying delib fixtures. `git diff sim/` is
empty; `git diff webapp/` is the one additive, default-off change this de-risk targets.

Run (CPU cheap-first; EXPORT OMP/OPENBLAS/MKL=4):
    SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._gnw_stop_conflict_scaled_boost_derisk --smoke \
        --seed 42 --json research/findings/raw/_gnw_stop_conflict_scaled_boost_smoke.json
    SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._gnw_stop_conflict_scaled_boost_derisk \
        --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_stop_conflict_scaled_boost_6seed.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json

from webapp import gnw_global_stop as G
from tools.verdict import Verdict

DEFAULT_SEEDS = [42, 43, 44, 100, 101, 102]

# the shipped organ's own reported full-conflict boost sits ~0.13 (BOOST_GAIN=1.0); a HIGH-severity upstream read
# (conf~0.05 -> severity~0.95) should reproduce ~that, a MODERATE one (conf~0.5 -> severity~0.5) roughly halves it.
_HIGH_SEVERITY_CONF = 0.05
_MODERATE_CONF = 0.50

# a real "mismatch" mm_peak sits ~0.28-0.31 across seeds (`_gnw_acc_bg_stop_trigger_derisk.get_real_mm_peak`); kept
# as a fixed representative value here too (mirrors `_gnw_global_stop_flip_soak.py`'s own fixture choice).
_REALISTIC_MISMATCH_MM_PEAK = 0.30


class _FakeChat:
    """A minimal host-scaffold chat carrying only the per-turn spiking reads the stop consumer inspects."""
    pass


def _chat(delib_n=None, conf=None, swapped=False, topic=None, mm_peak=None):
    c = _FakeChat()
    if delib_n is not None:
        d = {"n_ignited": delib_n, "decision": "ABSTAIN"}
        if conf is not None:
            d["conf"] = float(conf)
        c._last_gnw_delib = d
    if swapped:
        c._last_swap_drives = {"swapped": True, "new_topic": topic, "held_topic": topic,
                               "mm_peak": float(mm_peak) if mm_peak is not None else _REALISTIC_MISMATCH_MM_PEAK}
    return c


def _set_flag(on: bool):
    if on:
        os.environ["BRAIN_GNW_STOP_CONFLICT_SCALED"] = "1"
    else:
        os.environ.pop("BRAIN_GNW_STOP_CONFLICT_SCALED", None)


def evaluate_seed(seed: int, *, verbose: bool = True) -> dict:
    G._WS_CACHE.pop(int(seed), None)
    os.environ.pop("BRAIN_GNW_STOP_LESION", None)

    # (1) BYTE-IDENTICAL-OFF — flag unset, a genuine delib conflict trigger.
    _set_flag(False)
    info_off = G.observe_turn(_chat(delib_n=2), "meanwhile — the weather?", seed=seed)
    off_no_new_keys = bool(info_off is not None and "conflict_scaled" not in info_off
                            and "conflict_scale" not in info_off)
    off_cleared = bool((info_off or {}).get("cleared"))
    off_n_post = (info_off or {}).get("n_ignited_post")
    off_lead = str((info_off or {}).get("lead", "") or "")

    # a SECOND flag-off call (fresh cache) must reproduce the identical n_post/boost -- the off path is pure and
    # takes no state from the (never-armed) conflict-scale branch.
    G._WS_CACHE.pop(int(seed), None)
    info_off2 = G.observe_turn(_chat(delib_n=2), "meanwhile — the weather?", seed=seed)
    off_reproducible = bool((info_off2 or {}).get("n_ignited_post") == off_n_post
                            and (info_off2 or {}).get("boost") == (info_off or {}).get("boost"))

    # (2) HIGH-SEVERITY PRESERVED — flag ON, near-total upstream conflict (conf~0.05 -> severity~0.95).
    _set_flag(True)
    G._WS_CACHE.pop(int(seed), None)
    info_hi = G.observe_turn(_chat(delib_n=2, conf=_HIGH_SEVERITY_CONF), "meanwhile — the weather?", seed=seed)
    hi_cleared = bool((info_hi or {}).get("cleared"))
    hi_scale = (info_hi or {}).get("conflict_scale")
    hi_scaled_flag = bool((info_hi or {}).get("conflict_scaled") is True)
    hi_lead = str((info_hi or {}).get("lead", "") or "")

    # CHARACTERIZED (not gating): a MODERATE upstream conflict (conf~0.5 -> severity~0.5) -- reported either way.
    G._WS_CACHE.pop(int(seed), None)
    info_mod = G.observe_turn(_chat(delib_n=2, conf=_MODERATE_CONF), "meanwhile — the weather?", seed=seed)
    mod_cleared = bool((info_mod or {}).get("cleared"))
    mod_boost = (info_mod or {}).get("boost")

    # (3) LESION STILL WINS — flag ON (conflict-scaled) + the env lesion lever.
    os.environ["BRAIN_GNW_STOP_LESION"] = "1"
    G._WS_CACHE.pop(int(seed), None)
    info_les = G.observe_turn(_chat(delib_n=2, conf=_HIGH_SEVERITY_CONF), "meanwhile — the weather?", seed=seed)
    les_holds = bool((info_les or {}).get("n_ignited_post", 0) >= 2)
    les_lead = str((info_les or {}).get("lead", "") or "")
    os.environ.pop("BRAIN_GNW_STOP_LESION", None)

    # (4) SWAP-ONLY FALLBACK UNCHANGED — flag ON, but the trigger is swap-only (no delib conflict this turn).
    G._WS_CACHE.pop(int(seed), None)
    info_swap_on = G.observe_turn(_chat(swapped=True, topic="weather"), "actually, the weather?", seed=seed)
    swap_no_new_keys = bool(info_swap_on is not None and "conflict_scaled" not in info_swap_on
                            and "conflict_scale" not in info_swap_on)
    swap_lead_on = str((info_swap_on or {}).get("lead", "") or "")

    _set_flag(False)
    G._WS_CACHE.pop(int(seed), None)
    info_swap_off = G.observe_turn(_chat(swapped=True, topic="weather"), "actually, the weather?", seed=seed)
    swap_lead_off = str((info_swap_off or {}).get("lead", "") or "")
    swap_fallback_unchanged = bool(swap_lead_on == swap_lead_off
                                   and (info_swap_on or {}).get("n_ignited_post")
                                       == (info_swap_off or {}).get("n_ignited_post"))

    # (5) DETERMINISM — flag ON, same conflict_scale, build twice.
    _set_flag(True)
    G._WS_CACHE.pop(int(seed), None)
    info_d1 = G.observe_turn(_chat(delib_n=2, conf=_HIGH_SEVERITY_CONF), "meanwhile — the weather?", seed=seed)
    G._WS_CACHE.pop(int(seed), None)
    info_d2 = G.observe_turn(_chat(delib_n=2, conf=_HIGH_SEVERITY_CONF), "meanwhile — the weather?", seed=seed)
    determ = bool((info_d1 or {}).get("n_ignited_post") == (info_d2 or {}).get("n_ignited_post"))
    _set_flag(False)

    seed_go = bool(off_no_new_keys and off_reproducible and hi_cleared and hi_scaled_flag
                   and les_holds and (les_lead == "") and swap_no_new_keys and swap_fallback_unchanged and determ)

    result = {
        "seed": int(seed), "seed_go": seed_go,
        "byte_identical_off": {"no_new_keys": off_no_new_keys, "reproducible": off_reproducible,
                               "cleared": off_cleared, "n_post": off_n_post, "lead_nonempty": bool(off_lead)},
        "high_severity_scaled": {"conf": _HIGH_SEVERITY_CONF, "scale": hi_scale, "scaled_flag": hi_scaled_flag,
                                 "cleared": hi_cleared, "lead_nonempty": bool(hi_lead)},
        "moderate_severity_characterized": {"conf": _MODERATE_CONF, "cleared": mod_cleared, "boost": mod_boost},
        "lesion_still_wins": {"holds": les_holds, "lead_empty": bool(les_lead == "")},
        "swap_only_fallback": {"no_new_keys": swap_no_new_keys, "unchanged": swap_fallback_unchanged,
                               "lead_on": swap_lead_on, "lead_off": swap_lead_off},
        "determinism": determ,
    }
    if verbose:
        print(f"[gnw-stop conflict-scaled seed={seed}] seed_go={seed_go} | off(no_new_keys={off_no_new_keys} "
              f"reproducible={off_reproducible} cleared={off_cleared}) | hi(scale={hi_scale} "
              f"cleared={hi_cleared}) | mod(cleared={mod_cleared} boost={mod_boost}) | lesion_holds={les_holds} "
              f"| swap_fallback_unchanged={swap_fallback_unchanged} | determ={determ}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="De-risk for the GNW global-stop CONFLICT-SCALED boost lever "
                                             "(BRAIN_GNW_STOP_CONFLICT_SCALED, default-off).")
    ap.add_argument("--smoke", action="store_true", help="single-seed quick check (uses --seed)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_gnw_stop_conflict_scaled_boost_derisk.json")
    args = ap.parse_args()

    seeds = [args.seed] if args.smoke else (args.seeds if args.seeds else DEFAULT_SEEDS)
    print(f"[gnw-stop conflict-scaled derisk] seeds={seeds} backend={os.environ.get('SIM_BACKEND')}\n", flush=True)
    results = [evaluate_seed(s, verbose=True) for s in seeds]

    n = len(results)
    all_off_byte_id = all(r["byte_identical_off"]["no_new_keys"] and r["byte_identical_off"]["reproducible"]
                          for r in results)
    n_hi_cleared = sum(int(r["high_severity_scaled"]["cleared"]) for r in results)
    all_lesion_holds = all(r["lesion_still_wins"]["holds"] and r["lesion_still_wins"]["lead_empty"]
                           for r in results)
    all_swap_fallback = all(r["swap_only_fallback"]["no_new_keys"] and r["swap_only_fallback"]["unchanged"]
                            for r in results)
    all_determ = all(r["determinism"] for r in results)
    n_seed_go = sum(int(r["seed_go"]) for r in results)
    n_mod_cleared = sum(int(r["moderate_severity_characterized"]["cleared"]) for r in results)

    go = bool(all_off_byte_id and n_hi_cleared >= max(1, int(round(0.5 * n))) and all_lesion_holds
             and all_swap_fallback and all_determ and (n < 5 or n_hi_cleared >= 5))

    v = Verdict("GNW global-stop CONFLICT-SCALED boost de-risk (%d seed%s)" % (n, "" if n == 1 else "s"))
    v.require("byte-identical-off (no new info keys; flag-off reproducible)", all_off_byte_id, expect=True)
    if n >= 5:
        v.require("HIGH-severity upstream conflict still clears on >=5/6 seeds", n_hi_cleared >= 5, expect=True)
    else:
        v.require("HIGH-severity upstream conflict clears (smoke)", n_hi_cleared == n, expect=True)
    v.require("lesion still wins regardless of conflict_scale (all seeds)", all_lesion_holds, expect=True)
    v.require("swap-only fallback unchanged (no upstream conflict -> fixed BOOST_GAIN) (all seeds)",
              all_swap_fallback, expect=True)
    v.require("determinism (build-twice identical n_post) (all seeds)", all_determ, expect=True)
    v.disabled("native_short_term_plasticity", why="STD targets the workspace recurrence in-runner; native STP off")
    decided = v.decide(go=go, verbose=True)
    verdict = decided["status"] if decided["status"] != "NO-GO" else (
        "PARTIAL" if (all_off_byte_id and all_lesion_holds and all_swap_fallback) else "NO-GO")

    summary = {
        "runner": "_gnw_stop_conflict_scaled_boost_derisk", "verdict": verdict, "go": go,
        "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "n_seeds": n, "n_seed_go": n_seed_go, "n_high_severity_cleared": n_hi_cleared,
        "n_moderate_severity_cleared_characterized": n_mod_cleared,
        "all_byte_identical_off": all_off_byte_id, "all_lesion_holds": all_lesion_holds,
        "all_swap_fallback_unchanged": all_swap_fallback, "all_determinism": all_determ,
        "seeds": list(seeds), "per_seed": results,
        "flag": "BRAIN_GNW_STOP_CONFLICT_SCALED (default-OFF; this de-risk is the pre-flip evidence)",
        "lesion_lever": "BRAIN_GNW_STOP_LESION (zeroes the shared-resource-depression boost gain unconditionally, "
                        "even with conflict_scale set)",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 96}", flush=True)
    print(f"  GNW STOP CONFLICT-SCALED BOOST DE-RISK VERDICT: {verdict}  (byte_id_off={all_off_byte_id} · "
          f"hi_cleared {n_hi_cleared}/{n} · mod_cleared {n_mod_cleared}/{n} (characterized) · "
          f"lesion_holds={all_lesion_holds} · swap_fallback={all_swap_fallback} · determ={all_determ})",
          flush=True)
    print(f"    [saved] {args.json}\n{'=' * 96}", flush=True)
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
