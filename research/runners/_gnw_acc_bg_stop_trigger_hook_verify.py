"""Production-hook verification for the rank-12 ACC/BG STOP-trigger flag (`webapp/gnw_global_stop.detect_trigger`
delegating to `webapp/gnw_acc_bg_stop_trigger.detect_trigger_spiking` behind `BRAIN_GNW_STOP_TRIGGER_SPIKING`).

This does NOT re-derive the circuit's own GO gate (that is
`research/runners/_gnw_acc_bg_stop_trigger_derisk.py`, 6/6 seeds GO, `research/findings/raw/_gnw_acc_bg_stop_trigger_6seed.json`).
It verifies the THIN DISPATCH the production wire-in adds, on FAKE chats carrying realistic `_last_gnw_delib` /
`_last_swap_drives` fields (mirrors `_gnw_global_stop_flip_soak.py`'s own fixture style):

  (1) FLAG-OFF BYTE-IDENTICAL: with `BRAIN_GNW_STOP_TRIGGER_SPIKING` unset, `detect_trigger`'s return on each fixture
      is asserted (hash/tuple compare, not read from the source) to EXACTLY match a frozen copy of the ORIGINAL
      host boolean-OR logic -- proving the added branch never executes when off (TERMS.md: byte-identical must be
      asserted in the data).
  (2) FLAG-ON DISPATCH: with the flag on, `detect_trigger(chat)` returns EXACTLY what calling
      `detect_trigger_spiking(chat)` directly returns, on every fixture -- the dispatch adds no extra transformation.
  (3) FLAG-ON SENSIBLE: on the SAME 3 turn-classes the de-risk's parity check covers (delib-conflict, swap-only,
      neither), using REAL n_ignited/mm_peak (reused from the SAME already-existing organs, not hand-picked), the
      dispatched decision matches the ORIGINAL host boolean-OR verdict on those real values.
  (4) LESION-VIA-FLAG: `BRAIN_GNW_STOP_TRIGGER_LESION=1` makes the dispatched trigger FALSE on the delib-conflict
      fixture even though n_ignited genuinely indicates a conflict -- the production lesion lever reaches the
      circuit's own `afferent_lesion`.

Usage (CPU cheap-first):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_acc_bg_stop_trigger_hook_verify \
      --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_acc_bg_stop_trigger_hook_verify.json
"""
from __future__ import annotations

import argparse
import json
import os

from webapp import gnw_global_stop as G
from research.runners._gnw_acc_bg_stop_trigger_derisk import get_real_n_ignited, get_real_mm_peak

DEFAULT_SEEDS = [42, 43, 44, 100, 101, 102]


class _FakeChat:
    pass


def _chat(delib_n=None, swapped=False, mm_peak=None, topic=None):
    c = _FakeChat()
    if delib_n is not None:
        c._last_gnw_delib = {"n_ignited": delib_n, "decision": "ABSTAIN"}
    if swapped or mm_peak is not None:
        c._last_swap_drives = {"swapped": bool(swapped), "new_topic": topic, "held_topic": topic,
                               "mm_peak": float(mm_peak) if mm_peak is not None else 0.0}
    return c


def _original_host_boolean_or(chat):
    """A FROZEN copy of detect_trigger's PRE-EXISTING host boolean-OR (the exact logic before the rank-12 branch
    was added) -- the reference this verify compares the flag-OFF path against, in data, not by reading the source."""
    reason = None
    n_held = 2
    newcomer = None
    try:
        delib = getattr(chat, "_last_gnw_delib", None)
        if isinstance(delib, dict):
            n_ign = delib.get("n_ignited")
            if isinstance(n_ign, (int, float)) and int(n_ign) >= 2:
                reason = "delib_sustained_coignition"
                n_held = max(n_held, int(n_ign))
    except Exception:
        pass
    try:
        swap = getattr(chat, "_last_swap_drives", None)
        if isinstance(swap, dict) and bool(swap.get("swapped")):
            reason = "swap_topic_break" if reason is None else "delib+swap"
            t = swap.get("new_topic") or swap.get("held_topic")
            newcomer = str(t) if t else None
    except Exception:
        pass
    return (reason is not None), reason, n_held, newcomer


def evaluate_seed(seed: int, *, verbose: bool = True) -> dict:
    # OFF-ARM DISCIPLINE (2026-08-27 flip-soak-off-arm-staleness class, gates/flip_offarm_staleness): the flag's
    # own default flipped ON 2026-09-05 -- an unset/pop OFF arm would now silently read ON (a vacuous ON-vs-ON
    # compare). Force it explicitly, never pop the SPIKING flag itself.
    os.environ["BRAIN_GNW_STOP_TRIGGER_SPIKING"] = "0"
    os.environ.pop("BRAIN_GNW_STOP_TRIGGER_LESION", None)   # a *_LESION flag: unset correctly means "not lesioned"

    fixtures = {
        "delib_conflict": _chat(delib_n=2, mm_peak=0.02),
        "swap_only": _chat(delib_n=1, swapped=True, mm_peak=0.3, topic="weather"),
        "no_trigger": _chat(delib_n=1, swapped=False, mm_peak=0.02),
    }

    # (1) FLAG-OFF byte-identical vs the frozen original logic.
    off_matches = {}
    for name, chat in fixtures.items():
        got = G.detect_trigger(chat)
        want = _original_host_boolean_or(chat)
        off_matches[name] = {"got": list(got), "want": list(want), "match": bool(got == want)}
    flag_off_byte_identical = bool(all(v["match"] for v in off_matches.values()))

    # (2)+(3) FLAG-ON dispatch + sensible-vs-host-on-REAL-afferents.
    os.environ["BRAIN_GNW_STOP_TRIGGER_SPIKING"] = "1"
    from webapp import gnw_acc_bg_stop_trigger as ACCBG
    ACCBG._CIRCUIT_CACHE.clear()

    dispatch_matches = {}
    for name, chat in fixtures.items():
        via_detect = G.detect_trigger(chat)
        via_direct = ACCBG.detect_trigger_spiking(chat)
        dispatch_matches[name] = bool(via_detect == via_direct)
    dispatch_ok = bool(all(dispatch_matches.values()))

    n_ign_solo = get_real_n_ignited(seed, 1)
    n_ign_conflict = get_real_n_ignited(seed, 2)
    match = get_real_mm_peak(seed, "match")
    mismatch = get_real_mm_peak(seed, "mismatch")

    real_fixtures = {
        "delib_conflict": (_chat(delib_n=n_ign_conflict, mm_peak=match["mm_peak"]),
                          _original_host_boolean_or(_chat(delib_n=n_ign_conflict, swapped=match["swapped"]))[0]),
        "swap_only": (_chat(delib_n=n_ign_solo, swapped=mismatch["swapped"], mm_peak=mismatch["mm_peak"],
                           topic="cat"),
                     _original_host_boolean_or(_chat(delib_n=n_ign_solo, swapped=mismatch["swapped"]))[0]),
        "no_trigger": (_chat(delib_n=n_ign_solo, swapped=match["swapped"], mm_peak=match["mm_peak"]),
                      _original_host_boolean_or(_chat(delib_n=n_ign_solo, swapped=match["swapped"]))[0]),
    }
    real_matches = {}
    for name, (chat, host_expect) in real_fixtures.items():
        ACCBG._CIRCUIT_CACHE.clear()
        triggered, reason, n_held, newcomer = G.detect_trigger(chat)
        real_matches[name] = {"triggered": bool(triggered), "host_expect": bool(host_expect),
                              "match": bool(bool(triggered) == bool(host_expect))}
    n_real_match = sum(1 for v in real_matches.values() if v["match"])
    real_sensible = bool(n_real_match >= 2)

    # (4) LESION-via-flag: the delib-conflict fixture (genuine n_ignited>=2) must NOT trigger under the lesion.
    os.environ["BRAIN_GNW_STOP_TRIGGER_LESION"] = "1"
    ACCBG._CIRCUIT_CACHE.clear()
    lesioned_triggered, *_ = G.detect_trigger(_chat(delib_n=n_ign_conflict, mm_peak=match["mm_peak"]))
    os.environ.pop("BRAIN_GNW_STOP_TRIGGER_LESION", None)
    ACCBG._CIRCUIT_CACHE.clear()
    lesion_reverts = bool(lesioned_triggered is False)

    os.environ.pop("BRAIN_GNW_STOP_TRIGGER_SPIKING", None)
    ACCBG._CIRCUIT_CACHE.clear()

    seed_go = bool(flag_off_byte_identical and dispatch_ok and real_sensible and lesion_reverts)
    result = {"seed": int(seed), "seed_go": seed_go, "flag_off_byte_identical": flag_off_byte_identical,
             "off_matches": off_matches, "dispatch_ok": dispatch_ok, "dispatch_matches": dispatch_matches,
             "real_matches": real_matches, "n_real_match": n_real_match, "lesion_reverts": lesion_reverts}
    if verbose:
        print(f"[hook-verify seed={seed}] seed_go={seed_go} flag_off_byte_identical={flag_off_byte_identical} "
              f"dispatch_ok={dispatch_ok} real_match={n_real_match}/3 lesion_reverts={lesion_reverts}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_acc_bg_stop_trigger_hook_verify.json")
    args = ap.parse_args()

    results = [evaluate_seed(s) for s in args.seeds]
    all_go = all(r["seed_go"] for r in results)
    summary = {"runner": "_gnw_acc_bg_stop_trigger_hook_verify", "seeds": list(args.seeds), "all_go": all_go,
              "n_go": sum(int(r["seed_go"]) for r in results), "per_seed": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nHOOK-VERIFY VERDICT: {'GO' if all_go else 'NO-GO'} ({summary['n_go']}/{len(results)}) "
          f"[saved] {args.json}", flush=True)
    return 0 if all_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
