"""6-SEED FLIP-SOAK for the GNW GLOBAL-WORKSPACE STOP production organ (`webapp/gnw_global_stop.py`) — the
default-ON flip gate the parent runs before flipping `_GNW_STOP_DEFAULT_ON`/`BRAIN_GNW_STOP` on.

WHAT THIS GATES. The organ wires the distributed-overwrite GLOBAL STOP (de-risk global-stop capability 6/6 GO,
`2026-08-18-gnw-distributed-overwrite-workspace-PARTIAL.md`) into the live turn: on a strong interrupt / hard
topic-break the held P1.2 coalition (a stale incumbent + the newcomer == a 2-content conflict) is driven into a
divisively-normalized distributed workspace and a conflict-triggered depression of the SHARED recurrence CLEARS it to
n_ignited=0 BEFORE the newcomer ignites (a clean single-content workspace, no stale bleed). A CLEAN neural stop
prepends a clearing lead to the reply; the LESION (the shared-resource-depression term zeroed) leaves the workspace
>=2 co-ignited so the lead VANISHES. This soak proves that stop/lesion dissociation is STABLE across 6 seeds and that
the flag-off / no-interrupt turn is byte-identical (no lead, no key).

For each seed it exercises the SAME organ the handler calls (`webapp.gnw_global_stop`):
  * INTACT STOP     — a 2-content conflict driven to n_ignited=0 (the global stop; `run(..., lesion=False)`);
  * LESION          — the shared-resource-depression term zeroed (`run(..., lesion=True)`): the 2-content conflict
                      STAYS >=2 co-ignited (the localist boundary the divisive-norm+STD stop surpasses);
  * COUPLING        — `observe_turn` on a fake chat with a delib `n_ignited>=2` trigger and with a #85 swap
                      `swapped=True` trigger: a CLEAN stop -> the clearing lead is present; under the env lesion the
                      lead vanishes (byte-identical to the no-lead baseline);
  * NO-TRIGGER      — `observe_turn` on an ordinary turn (delib n_ignited=1 hold, no swap) -> None (no key, no lead);
  * DETERMINISM     — build the workspace twice at the seed -> identical stop n_post.

  FLIP-GATE (default-ON) passes iff, across the 6 seeds:
    * INTACT clean stop (n_pre>=2 -> n_post==0)      on >= 5/6 seeds   (the global-stop capability)
    * LESION holds (n_post >= 2)                      on ALL 6 seeds    (load-bearing: zero the depression -> stale
                                                                         content persists, the lead cannot fire)
    * COUPLING lead present-on-stop / vanish-on-lesion on ALL 6 seeds   (the surface change RIDES the neural stop)
    * NO-TRIGGER byte-identical (observe -> None)     on ALL 6 seeds    (an ordinary turn is untouched)
    * DETERMINISM (build-twice identical n_post)       on ALL 6 seeds

This is the NO-REGRESSION / stability soak, the load-bearing evidence (not a headline): the lesion arm proves the
clearing is carried by the SPIKING depression of the shared recurrence at every seed, not a re-hidden host `if`.

FIXTURE FIX (2026-09-05, found by the rank-12 ACC/BG STOP-trigger production-flip verify). The `swap_only` COUPLING
fixture originally set `swapped=True` with NO `mm_peak` at all -- harmless while `detect_trigger` was a host
`n_ignited>=2 or swapped` (mm_peak was never read), but STALE once `BRAIN_GNW_STOP_TRIGGER_SPIKING` flipped
default-ON: the spiking circuit reads `mm_peak` as its OWN synaptic afferent, and a `swap_drives` dict with `swapped`
set but `mm_peak` absent reads as `mm_peak=0.0` -> `detect_trigger_spiking`'s "nothing to read" bail-out fires ->
no trigger. Traced end-to-end (`webapp/gnw_thought_swap.py::ThoughtSwapWorkspace.observe` / `run_intention_swap`'s
own `swapped` computation, `webapp/swap_drives_chat.py::observe_turn`): a genuine `swapped=True` is CAUSALLY
downstream of an elevated `mm_peak` (the boost driving the eviction IS `boost_gain * mm_rate`) and is ALWAYS
returned alongside a real `mm_peak` float in production -- `swapped=True` with `mm_peak` absent is unreachable from
`/api/brain-chat`, confirmed by code trace, not merely by not having seen it. The fixture now carries a realistic
mismatch-level `mm_peak` (matching the de-risk's own real "mismatch" scenario magnitude, `_gnw_acc_bg_stop_trigger_derisk.get_real_mm_peak`), so this soak exercises the SAME afferent SHAPE production actually produces.

Run (CPU cheap-first; EXPORT OMP/OPENBLAS/MKL=4):
    SIM_BACKEND=numpy python -u -m research.runners._gnw_global_stop_flip_soak \
        --seeds 42 43 44 100 101 102 --json research/findings/raw/_gnw_global_stop_flip_soak.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import json

from webapp import gnw_global_stop as G
from tools.lab import attributable_to
from tools.verdict import Verdict

DEFAULT_SEEDS = [42, 43, 44, 100, 101, 102]

# a real "mismatch" mm_peak sits ~0.28-0.31 across seeds (research/runners/_gnw_acc_bg_stop_trigger_derisk.py's own
# get_real_mm_peak); this fixture uses a fixed representative value rather than reuse-by-import so this file's own
# COUPLING fixture stays the minimal, self-contained hand-built chat it always was (no new cross-module dependency).
_REALISTIC_MISMATCH_MM_PEAK = 0.30


class _FakeChat:
    """A minimal host-scaffold chat carrying only the per-turn spiking reads the stop consumer inspects."""
    pass


def _chat(delib_n=None, swapped=False, topic=None, mm_peak=None):
    c = _FakeChat()
    if delib_n is not None:
        c._last_gnw_delib = {"n_ignited": delib_n, "decision": "ABSTAIN"}
    if swapped:
        # a genuine swapped=True is CAUSALLY downstream of an elevated mm_peak in production (see the fixture-fix
        # note above) -- default to a realistic mismatch-level value so this fixture matches production's shape
        # instead of a `swapped` boolean production never actually delivers unaccompanied.
        c._last_swap_drives = {"swapped": True, "new_topic": topic, "held_topic": topic,
                               "mm_peak": float(mm_peak) if mm_peak is not None else _REALISTIC_MISMATCH_MM_PEAK}
    return c


def evaluate_seed(seed: int, *, verbose: bool = True) -> dict:
    # fresh per-seed workspace (bypass the module cache so a soak seed never reuses another seed's warm build).
    ws = G._StopWorkspace(seed)
    n_pre_i, n_post_i, boost_i, cleared_i = ws.run(2, lesion=False)     # INTACT stop
    n_pre_l, n_post_l, boost_l, cleared_l = ws.run(2, lesion=True)      # LESION (depression term zeroed)
    # DETERMINISM: a second fresh workspace at the same seed -> identical stop n_post.
    ws2 = G._StopWorkspace(seed)
    _p2, n_post_i2, _b2, _c2 = ws2.run(2, lesion=False)
    determ = bool(n_post_i2 == n_post_i)

    # COUPLING through observe_turn (the exact production entry point). Use a private module cache slot per seed.
    G._WS_CACHE.pop(int(seed), None)
    os.environ.pop("BRAIN_GNW_STOP_LESION", None)
    info_delib = G.observe_turn(_chat(delib_n=2), "meanwhile — the weather?", seed=seed)   # delib-conflict trigger
    lead_stop = str((info_delib or {}).get("lead", "") or "")
    info_swap = G.observe_turn(_chat(swapped=True, topic="weather"), "actually, the weather?", seed=seed)  # swap
    lead_swap = str((info_swap or {}).get("lead", "") or "")
    os.environ["BRAIN_GNW_STOP_LESION"] = "1"
    info_les = G.observe_turn(_chat(delib_n=2), "meanwhile — the weather?", seed=seed)      # lesion -> no lead
    lead_les = str((info_les or {}).get("lead", "") or "")
    os.environ.pop("BRAIN_GNW_STOP_LESION", None)
    info_none = G.observe_turn(_chat(delib_n=1, swapped=False), "who does the dog chase", seed=seed)  # no trigger

    # ATTRIBUTION: what fraction of the CLEARING is owned by the shared-recurrence depression? The intact arm clears
    # (n_pre_i - n_post_i); the LESION control (depression term zeroed) clears (n_pre_l - n_post_l). attributable_to
    # subtracts them -> ~1.0 means the depression owns the whole clear (the lesion clears ~nothing = the stale bleed).
    intact_cleared = float(n_pre_i - n_post_i)
    lesion_cleared = float(n_pre_l - n_post_l)
    clearing_attrib = attributable_to("global-stop clearing via shared-recurrence depression",
                                      intact_cleared, lesion_cleared, warn_below=0.8)

    clean_stop = bool(cleared_i and n_pre_i >= 2 and n_post_i == 0)
    lesion_holds = bool(n_post_l >= 2)
    lead_present_on_stop = bool(lead_stop != "" and lead_swap != "")
    lead_vanishes_on_lesion = bool(lead_les == "")
    no_trigger_byte_id = bool(info_none is None)
    coupling_ok = bool(lead_present_on_stop and lead_vanishes_on_lesion)

    seed_go = bool(clean_stop and lesion_holds and coupling_ok and no_trigger_byte_id and determ)
    result = {
        "seed": int(seed), "seed_go": seed_go,
        "clearing_attribution": (None if clearing_attrib is None else float(clearing_attrib)),
        "intact_stop": {"n_pre": n_pre_i, "n_post": n_post_i, "boost": boost_i, "cleared": cleared_i},
        "lesion_stop": {"n_pre": n_pre_l, "n_post": n_post_l, "boost": boost_l, "cleared": cleared_l},
        "coupling": {"lead_delib": lead_stop, "lead_swap": lead_swap, "lead_lesion": lead_les,
                     "no_trigger_info": info_none},
        "gates": {"clean_stop": clean_stop, "lesion_holds": lesion_holds,
                  "lead_present_on_stop": lead_present_on_stop, "lead_vanishes_on_lesion": lead_vanishes_on_lesion,
                  "no_trigger_byte_identical": no_trigger_byte_id, "determinism": determ},
    }
    if verbose:
        print(f"[gnw-stop soak seed={seed}] seed_go={seed_go} | INTACT n {n_pre_i}->{n_post_i} (clean={clean_stop}) "
              f"| LESION n {n_pre_l}->{n_post_l} (holds={lesion_holds}) | lead_stop={lead_present_on_stop} "
              f"lead_vanish={lead_vanishes_on_lesion} no_trig={no_trigger_byte_id} determ={determ}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description="6-seed flip-soak for the GNW global-workspace STOP production organ.")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_global_stop_flip_soak.json")
    args = ap.parse_args()

    print(f"[gnw-stop flip-soak] seeds={args.seeds} backend={os.environ.get('SIM_BACKEND')}\n", flush=True)
    results = [evaluate_seed(s, verbose=True) for s in args.seeds]

    n = len(results)
    n_clean_stop = sum(int(r["gates"]["clean_stop"]) for r in results)
    all_lesion_holds = all(r["gates"]["lesion_holds"] for r in results)
    all_coupling = all(r["gates"]["lead_present_on_stop"] and r["gates"]["lead_vanishes_on_lesion"] for r in results)
    all_no_trigger = all(r["gates"]["no_trigger_byte_identical"] for r in results)
    all_determ = all(r["gates"]["determinism"] for r in results)
    n_seed_go = sum(int(r["seed_go"]) for r in results)
    attribs = [r["clearing_attribution"] for r in results if r["clearing_attribution"] is not None]
    min_attrib = min(attribs) if len(attribs) == n else None   # None if any seed's attribution was UNDEFINED
    all_attrib_ok = bool(min_attrib is not None and min_attrib >= 0.8)

    flip_go = bool(n_clean_stop >= 5 and all_lesion_holds and all_attrib_ok and all_coupling
                   and all_no_trigger and all_determ)

    # A verdict you EARN: the preconditions travel with the artifact (tools.verdict.Verdict; verdict_preconditions gate).
    v = Verdict("GNW global-stop production organ flip-soak (%d seeds)" % n)
    v.require("clean global stop on >=5/6 (co-ignited n>=2 -> n_ignited=0)", n_clean_stop >= 5, expect=True)
    v.require("LESION holds (depression term zeroed -> n_post>=2, stale persists) on all seeds",
              all_lesion_holds, expect=True)
    v.require("clearing attributable to the shared-recurrence depression (>=0.8) on all seeds",
              all_attrib_ok, expect=True)
    v.require("coupling: clearing lead present-on-stop AND vanish-on-lesion on all seeds", all_coupling, expect=True)
    v.require("no-trigger turn byte-identical (observe_turn -> None) on all seeds", all_no_trigger, expect=True)
    v.require("determinism: build-twice identical stop n_post on all seeds", all_determ, expect=True)
    v.disabled("native_short_term_plasticity", why="STD targets the workspace recurrence in-runner; native STP off")
    decided = v.decide(go=flip_go, verbose=True)
    verdict = decided["status"] if decided["status"] != "NO-GO" else (
        "PARTIAL" if (n_clean_stop >= 5 and all_lesion_holds) else "NO-GO")

    summary = {
        "runner": "_gnw_global_stop_flip_soak", "verdict": verdict, "flip_go": flip_go,
        "preconditions": decided["preconditions"], "disabled_processes": decided["disabled_processes"],
        "undefined_reasons": decided["undefined_reasons"],
        "n_seeds": n, "n_clean_stop": n_clean_stop, "n_seed_go": n_seed_go, "min_clearing_attribution": min_attrib,
        "all_lesion_holds": all_lesion_holds, "all_coupling": all_coupling, "all_clearing_attributed": all_attrib_ok,
        "all_no_trigger_byte_identical": all_no_trigger, "all_determinism": all_determ,
        "seeds": list(args.seeds), "per_seed": results,
        "flag": "BRAIN_GNW_STOP (default-OFF; parent flips _GNW_STOP_DEFAULT_ON after this soak)",
        "lesion_lever": "BRAIN_GNW_STOP_LESION (zeroes the shared-resource-depression boost gain)",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 96}", flush=True)
    print(f"  GNW GLOBAL-STOP FLIP-SOAK VERDICT: {verdict}  (clean-stop {n_clean_stop}/{n} · seed_go {n_seed_go}/{n} · "
          f"lesion_holds={all_lesion_holds} coupling={all_coupling} no_trigger={all_no_trigger} determ={all_determ})",
          flush=True)
    print(f"    [saved] {args.json}\n{'=' * 96}", flush=True)
    return 0 if flip_go else 1


if __name__ == "__main__":
    raise SystemExit(main())
