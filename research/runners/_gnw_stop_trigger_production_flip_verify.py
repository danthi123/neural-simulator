"""PRODUCTION-FLIP verification for the rank-12 GNW STOP-trigger spiking ACC/BG circuit — is it SAFE + GENUINELY
LOAD-BEARING to run `BRAIN_GNW_STOP_TRIGGER_SPIKING` DEFAULT-ON (Track 1, ship-the-validated-wins campaign)?

This is NOT a re-derivation of the circuit's own GO gate (`_gnw_acc_bg_stop_trigger_derisk.py`, 6/6 seeds GO) or the
production-dispatch hook-verify (`_gnw_acc_bg_stop_trigger_hook_verify.py`, 6/6 seeds GO) — both already committed
2026-09-05 and reused-by-import below, never re-derived. It answers the FLIP-SPECIFIC questions those two did not
ask: is flipping the module's DEFAULT (not merely setting the env var to "1") safe for the rest of the live turn,
and does the load-bearing proof survive all the way to the OBSERVABLE conversation surface (whether the `gnw_stop`
trace key attaches to a real `/api/brain-chat` response) — not just the internal trigger boolean the hook-verify
stopped at?

ARM 1 — NO REGRESSION (per seed, 42/43/44/100/101/102):
  (a) the explicit opt-out escape hatch (`BRAIN_GNW_STOP_TRIGGER_SPIKING="0"`) still reproduces the ORIGINAL frozen
      host boolean-OR byte-for-byte on 3 fixtures — never rely on unset==off once a default is flipped
      (`gates/flip_offarm_staleness`'s own lesson, applied here to this soak's OWN off arm, not just the sibling
      hook-verify file this commit also fixes).
  (b) with the flag genuinely UNSET (the actual shipped default after this commit), `detect_trigger` on real
      organ-sourced afferents (3 turn-classes) is byte-identical to explicit `="1"` — the new default really is the
      audited ON code path, not a different one.
  (c) the SIBLING production organ this trigger feeds, `webapp.gnw_global_stop`'s OWN flip-soak
      (`_gnw_global_stop_flip_soak.evaluate_seed`, reused verbatim — no re-derivation), still returns full GO with
      this trigger's default now flipped underneath it: the already-shipped (2026-08-26) STOP-*clear* mechanism is
      unharmed by swapping its trigger source.
  (d) [once, not per-seed — see main()] a cross-faculty REGRESSION BATTERY
      (`onebrain_regression_battery.run_regression_battery`, reused verbatim) drives ~38 OTHER default-ON faculties
      through the REAL `webapp.server.brain_chat` handler (`brain="tiny-demo"`, GPU-free) and asserts every one
      still DECIDES identically with this flag ON vs OFF — "every other faculty stays alive" on the actual
      /api/brain-chat path, not an isolated stub.

ARM 2 — LOAD-BEARING, NOT HOLLOW (the crux; per seed). Real, organ-sourced n_ignited (1..4) and mm_peak
  (match->mismatch) afferents are driven through the REAL production entry point `gnw_global_stop.observe_turn`
  (exactly what `webapp/server.py` calls, twice, on every turn) at the shipped default, and the read-out is
  `chat._last_gnw_stop`'s `acted` field — the SAME boolean that gates whether `resp["gnw_stop"]` is attached to a
  real JSON response — not just the internal `detect_trigger` boolean the existing hook-verify stopped at, and
  deliberately NOT the downstream `cleared`/clearing-lead string (that sub-field is owned by the SIBLING, already-
  accepted <6/6-bar STOP-*clear* depression mechanism this finding does not touch — gating on it would conflate two
  independent mechanisms' load-bearing claims). VARYING either afferent alone must flip `acted` OFF->ON; zeroing
  the ACC/BG circuit's OWN afferent->ACC synapses (`BRAIN_GNW_STOP_TRIGGER_LESION=1`) must make the SAME variation
  produce a byte-identical (always-`acted=False`) surface — internally-driven-yet-invisible is impossible here by
  construction (there is nothing to be invisible: the read-out IS the key-attachment), so the bar is `n_hollow==0`
  measured directly, not inferred.

ARM 3 — THE FLIP IS REAL. `stop_trigger_spiking_enabled()` read with the flag genuinely unset (no env var, no
  monkeypatch) returns True on this branch — asserted in the data (folded into ARM 1a's own per-seed check).

GO iff all arms hold on all 6 seeds (+ the one-shot battery).

Run (CPU-only; tiny-demo is GPU-free, no brain-load to queue):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_stop_trigger_production_flip_verify \
      --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_gnw_stop_trigger_production_flip_verify.json
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from webapp import gnw_global_stop as G
from webapp import gnw_acc_bg_stop_trigger as ACCBG
from research.runners._gnw_acc_bg_stop_trigger_derisk import get_real_n_ignited, get_real_mm_peak
from research.runners._gnw_acc_bg_stop_trigger_hook_verify import _original_host_boolean_or, _chat
from research.runners import _gnw_global_stop_flip_soak as SIBLING_SOAK
from tools.verdict import Verdict
from tools.lab import attributable_to

DEFAULT_SEEDS = [42, 43, 44, 100, 101, 102]

_FLAG = "BRAIN_GNW_STOP_TRIGGER_SPIKING"
_LESION_FLAG = "BRAIN_GNW_STOP_TRIGGER_LESION"


def _clear_caches():
    ACCBG._CIRCUIT_CACHE.clear()
    G._WS_CACHE.clear()


# ── ARM 1: the flip is real, and nothing else regresses ───────────────────────────────────────────────────────────
def arm1_flip_and_no_regression(seed: int, *, verbose: bool = True) -> dict:
    fixtures = {
        "delib_conflict": _chat(delib_n=2, mm_peak=0.02),
        "swap_only": _chat(delib_n=1, swapped=True, mm_peak=0.3, topic="weather"),
        "no_trigger": _chat(delib_n=1, swapped=False, mm_peak=0.02),
    }
    n_ign_solo = get_real_n_ignited(seed, 1)
    n_ign_conflict = get_real_n_ignited(seed, 2)
    match = get_real_mm_peak(seed, "match")
    mismatch = get_real_mm_peak(seed, "mismatch")
    real_fixtures = {
        "delib_conflict": _chat(delib_n=n_ign_conflict, mm_peak=match["mm_peak"]),
        "swap_only": _chat(delib_n=n_ign_solo, swapped=mismatch["swapped"], mm_peak=mismatch["mm_peak"], topic="cat"),
        "no_trigger": _chat(delib_n=n_ign_solo, swapped=match["swapped"], mm_peak=match["mm_peak"]),
    }

    # (ARM 3, folded in here) the flip is REAL: bare-unset resolves True, asserted in the data, no monkeypatch.
    os.environ.pop(_FLAG, None)
    os.environ.pop(_LESION_FLAG, None)
    flip_is_on_when_unset = bool(ACCBG.stop_trigger_spiking_enabled() is True)
    _clear_caches()
    unset_results = {name: G.detect_trigger(c) for name, c in real_fixtures.items()}

    # (a) explicit OPT-OUT still reproduces the frozen original host boolean-OR byte-for-byte (never rely on
    #     unset==off again — flip_offarm_staleness's own lesson, applied to THIS runner's own off arm).
    os.environ[_FLAG] = "0"
    _clear_caches()
    off_matches = {name: bool(G.detect_trigger(c) == _original_host_boolean_or(c)) for name, c in fixtures.items()}
    off_arm_byte_identical = bool(all(off_matches.values()))

    # (b) explicit ON reproduces the SAME results as bare-unset — the new default really is the audited ON path.
    os.environ[_FLAG] = "1"
    _clear_caches()
    on_results = {name: G.detect_trigger(c) for name, c in real_fixtures.items()}
    default_matches_explicit_on = {name: bool(unset_results[name] == on_results[name]) for name in real_fixtures}
    default_is_audited_on = bool(all(default_matches_explicit_on.values()))

    # (c) the SIBLING gnw-global-stop production organ's OWN flip-soak, reused verbatim, unaffected by this flip's
    #     default now flipped underneath it (the STOP *clear* mechanism, already default-ON since 2026-08-26).
    os.environ.pop(_FLAG, None)   # back to the literal shipped default for the sibling soak
    _clear_caches()
    sibling = SIBLING_SOAK.evaluate_seed(seed, verbose=False)
    sibling_ok = bool(sibling["seed_go"])

    os.environ.pop(_FLAG, None)
    os.environ.pop(_LESION_FLAG, None)
    _clear_caches()

    ok = bool(flip_is_on_when_unset and off_arm_byte_identical and default_is_audited_on and sibling_ok)
    result = {
        "seed": int(seed), "arm1_ok": ok,
        "flip_is_on_when_unset": flip_is_on_when_unset,
        "off_arm_byte_identical": off_arm_byte_identical, "off_matches": off_matches,
        "default_matches_explicit_on": default_matches_explicit_on, "default_is_audited_on": default_is_audited_on,
        "sibling_gnw_global_stop_flip_soak_go": sibling_ok, "sibling_gates": sibling.get("gates"),
    }
    if verbose:
        print(f"[flip-verify seed={seed}] ARM1 ok={ok} flip_on_unset={flip_is_on_when_unset} "
              f"off_byte_id={off_arm_byte_identical} default==explicit_on={default_is_audited_on} "
              f"sibling_stop_soak_go={sibling_ok}", flush=True)
    return result


# ── ARM 2: load-bearing, not hollow — all the way to observe_turn's OBSERVABLE key-attachment ──────────────────────
def arm2_load_bearing(seed: int, *, verbose: bool = True) -> dict:
    match = get_real_mm_peak(seed, "match")
    mm_match = match["mm_peak"]
    mismatch = get_real_mm_peak(seed, "mismatch")
    mm_mismatch = mismatch["mm_peak"]
    n_solo = get_real_n_ignited(seed, 1)

    os.environ.pop(_FLAG, None)            # the shipped default (unset -> on)
    os.environ.pop(_LESION_FLAG, None)     # INTACT afferents
    os.environ.pop("BRAIN_GNW_STOP_LESION", None)   # the SIBLING (unrelated) STOP-clear lesion stays off throughout
    _clear_caches()

    def _observe(n_ign, mm_pk, tag):
        c = _chat(delib_n=n_ign, swapped=False, mm_peak=mm_pk)
        info = G.observe_turn(c, f"turn-{tag}", seed=seed)
        acted = bool(info is not None and info.get("acted"))
        return {"n_ignited": n_ign, "mm_peak": float(mm_pk), "acted": acted,
                "reason": (info or {}).get("reason"), "lead": (info or {}).get("lead", "")}

    delib_levels = [1, 2, 3, 4]
    intact_delib_sweep = [_observe(n, mm_match, f"delib{n}") for n in delib_levels]
    mm_levels = [mm_match, (mm_match + mm_mismatch) / 2.0, mm_mismatch]
    intact_mm_sweep = [_observe(n_solo, mm, f"mm{i}") for i, mm in enumerate(mm_levels)]

    delib_varies = bool((not intact_delib_sweep[0]["acted"]) and any(s["acted"] for s in intact_delib_sweep[1:]))
    mm_varies = bool((not intact_mm_sweep[0]["acted"]) and any(s["acted"] for s in intact_mm_sweep[1:]))

    # LESION: the ACC/BG circuit's OWN afferent->ACC synapses zeroed. The SAME varying inputs must now produce a
    # byte-identical NEVER-`acted` surface — the anti-hollow bar.
    os.environ[_LESION_FLAG] = "1"
    _clear_caches()
    lesioned_delib_sweep = [_observe(n, mm_match, f"lesdelib{n}") for n in delib_levels]
    lesioned_mm_sweep = [_observe(n_solo, mm, f"lesmm{i}") for i, mm in enumerate(mm_levels)]
    os.environ.pop(_LESION_FLAG, None)
    _clear_caches()

    lesion_kills_delib_variation = bool(not any(s["acted"] for s in lesioned_delib_sweep))
    lesion_kills_mm_variation = bool(not any(s["acted"] for s in lesioned_mm_sweep))
    n_hollow = int(sum(1 for s in lesioned_delib_sweep + lesioned_mm_sweep if s["acted"]))   # must be 0

    n_acted_intact = sum(1 for s in intact_delib_sweep + intact_mm_sweep if s["acted"])
    n_total = len(intact_delib_sweep) + len(intact_mm_sweep)
    n_acted_lesioned = sum(1 for s in lesioned_delib_sweep + lesioned_mm_sweep if s["acted"])
    attrib = attributable_to("gnw_stop key-attachment rate via the afferent->ACC pathway (seed %d)" % seed,
                             n_acted_intact / n_total, n_acted_lesioned / n_total, warn_below=0.5)

    seed_ok = bool(delib_varies and mm_varies and lesion_kills_delib_variation and lesion_kills_mm_variation
                  and n_hollow == 0)
    result = {
        "seed": int(seed), "arm2_ok": seed_ok,
        "intact_delib_sweep": intact_delib_sweep, "intact_mm_sweep": intact_mm_sweep,
        "lesioned_delib_sweep": lesioned_delib_sweep, "lesioned_mm_sweep": lesioned_mm_sweep,
        "delib_varies": delib_varies, "mm_varies": mm_varies,
        "lesion_kills_delib_variation": lesion_kills_delib_variation,
        "lesion_kills_mm_variation": lesion_kills_mm_variation,
        "n_hollow": n_hollow, "attribution": (None if attrib is None else float(attrib)),
    }
    if verbose:
        print(f"[flip-verify seed={seed}] ARM2 ok={seed_ok} delib_varies={delib_varies} mm_varies={mm_varies} "
              f"lesion_kills_delib={lesion_kills_delib_variation} lesion_kills_mm={lesion_kills_mm_variation} "
              f"n_hollow={n_hollow} attrib={attrib}", flush=True)
    return result


def run_battery_once(out_dir: str) -> dict:
    from research.runners.onebrain_regression_battery import run_regression_battery
    return run_regression_battery(flag=_FLAG, out_dir=out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Production-flip verification for BRAIN_GNW_STOP_TRIGGER_SPIKING (rank-12).")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--no-battery", action="store_true",
                    help="skip the one-shot cross-faculty regression battery (mechanics-only smoke)")
    ap.add_argument("--json", type=str,
                    default="research/findings/raw/_gnw_stop_trigger_production_flip_verify.json")
    args = ap.parse_args()

    print(f"[flip-verify] seeds={args.seeds} backend={os.environ.get('SIM_BACKEND')} flag={_FLAG}\n", flush=True)
    arm1 = [arm1_flip_and_no_regression(s) for s in args.seeds]
    arm2 = [arm2_load_bearing(s) for s in args.seeds]

    battery = None
    if not args.no_battery:
        raw_dir = os.path.join(os.path.dirname(os.path.abspath(args.json)), "_gnw_stop_trigger_flip_battery")
        battery = run_battery_once(raw_dir)

    all_arm1 = all(r["arm1_ok"] for r in arm1)
    all_arm2 = all(r["arm2_ok"] for r in arm2)
    battery_ok = bool(battery is None or battery.get("all_pass"))

    flip_go = bool(all_arm1 and all_arm2 and battery_ok)

    v = Verdict("GNW STOP-trigger ACC/BG circuit PRODUCTION-FLIP verify (%d seeds)" % len(args.seeds))
    v.require("the flag's default genuinely resolves ON when unset (asserted in the data), all seeds",
              all(r["flip_is_on_when_unset"] for r in arm1), expect=True)
    v.require("explicit opt-out (=0) still reproduces the frozen original host boolean-OR byte-for-byte, all seeds",
              all(r["off_arm_byte_identical"] for r in arm1), expect=True)
    v.require("bare-unset default byte-identical to explicit ON on real afferents, all seeds",
              all(r["default_is_audited_on"] for r in arm1), expect=True)
    v.require("the SIBLING gnw-global-stop STOP-clear flip-soak (default-ON since 2026-08-26) stays GO underneath "
              "this flip, all seeds", all(r["sibling_gnw_global_stop_flip_soak_go"] for r in arm1), expect=True)
    v.require("n_ignited afferent ALONE varies the observable gnw_stop key-attachment (OFF->ON), all seeds",
              all(r["delib_varies"] for r in arm2), expect=True)
    v.require("mm_peak afferent ALONE varies the observable gnw_stop key-attachment (OFF->ON), all seeds",
              all(r["mm_varies"] for r in arm2), expect=True)
    v.require("afferent->ACC LESION makes n_hollow==0: the SAME input variation NEVER attaches gnw_stop, all seeds",
              all(r["n_hollow"] == 0 for r in arm2), expect=True)
    if battery is not None:
        v.require("the cross-faculty regression battery (~38 other default-ON faculties) reports all_pass",
                  battery_ok, expect=True)
    else:
        v.disabled("regression_battery", why="--no-battery smoke run: the one-shot cross-faculty check was skipped")
    vd = v.decide(go=flip_go, verbose=True)

    summary = {
        "runner": "_gnw_stop_trigger_production_flip_verify", "verdict": vd["status"], "flip_go": flip_go,
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
        "seeds": list(args.seeds), "flag": _FLAG, "lesion_flag": _LESION_FLAG,
        "arm1_per_seed": arm1, "arm2_per_seed": arm2,
        "regression_battery": battery,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 100}", flush=True)
    print(f"  PRODUCTION-FLIP VERDICT ({_FLAG}): {vd['status']}  (arm1={all_arm1} arm2={all_arm2} "
          f"battery_all_pass={battery_ok})", flush=True)
    print(f"    [saved] {args.json}\n{'=' * 100}", flush=True)
    return 0 if flip_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
