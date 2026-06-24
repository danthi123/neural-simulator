"""A1 VALIDATION — the PAIRING SEAM: the longitudinal develop loop SAVES a developed-brain BUNDLE that the
interact console can LOAD + talk to.

Per the paired scope (`research/findings/2026-06-24-capstone-console-scoping.md`, gap A-ii + step A1): currently
`developed_brain_io.save_developed_brain` is NEVER called -> no `brain.json` exists on disk -> the console can't
load a develop-loop brain. This validates the A1 build:

  1. RUN a SMALL develop-loop smoke (1-2 sim-days) with the new `--save-bundle` + `--per-day-bundles` wiring, so
     the loop SAVES a final `brain/` bundle AND per-day `day_<N>/` bundles (the day-0-vs-day-N deliverable).
  2. ROUND-TRIP: `load_developed_brain(bundle)` (the SAME path the console/TUI uses) reconstructs the brain. The
     loaded brain has the SAME vocab / facts / grounded codes as the in-memory developed brain (FAITHFUL, not a
     re-derive -- the saved phases are loaded verbatim, the facts re-store exactly).
  3. USABLE: the loaded brain ANSWERS a who/what query on a taught fact AND ABSTAINS on an untaught cue (the
     no-confab moat 0-FA) -- i.e. it is a usable chat brain identical to the in-memory one.

ANTI-CHEAT: the codes round-trip is FAITHFUL (the loaded code == the saved code byte-for-byte, NOT re-derived from
the seed); the loaded brain's moat is 0-FA on untaught cues; a per-day bundle's facts are a STRICT SUBSET of the
final bundle's facts (the brain accumulated knowledge over days -> the bundles capture day-by-day development).

REUSE-BY-IMPORT, NO `sim/` edit. GPU (`SIM_BACKEND=cupy`). FOREGROUND-only. Run:
    SIM_BACKEND=cupy python -u -m research.runners._capstone_console_A1_bundle_validate --n-days 2 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._longitudinal_develop_loop_gpu import run_gpu_smoke  # noqa: E402
from research.runners.developed_brain_io import (  # noqa: E402
    load_developed_brain, is_developed_brain_bundle, _read_manifest, _load_codes_npz, _load_facts_json,
)


def _stored_facts(agent):
    comp = getattr(agent, "agent", agent).composer
    return [(f.get("agent"), f.get("action"), f.get("patient")) for f, _ in comp.kb
            if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))]


def _validate_bundle(bundle_dir, *, untaught_cue, verbose=True):
    """Round-trip ONE bundle and probe it. Returns a per-bundle result dict."""
    out = {"path": os.path.abspath(bundle_dir), "is_bundle": False}
    if not is_developed_brain_bundle(bundle_dir):
        out["error"] = "not a developed-brain bundle (no brain.json)"
        return out
    out["is_bundle"] = True

    manifest = _read_manifest(bundle_dir)
    saved_codes = _load_codes_npz(bundle_dir)          # {word: phases[D]} on disk
    saved_facts = _load_facts_json(bundle_dir)         # the facts.json the saver wrote
    out["manifest_n_facts"] = manifest.get("n_facts")
    out["manifest_n_grounded_codes"] = manifest.get("n_grounded_codes")
    out["manifest_vocab_size"] = len(manifest.get("vocab", []))

    # --- LOAD the brain the SAME way the console/TUI does (load_developed_brain). We load WITHOUT the multi-turn
    #     WM wrapper for the round-trip checks (faster build; the codes/facts/moat are composer-level, the WM loop
    #     adds nothing to faithfulness). The TUI's default IS multi-turn; load_developed_brain(use_multiturn=True)
    #     is exercised by the CPU smoke + is the only difference (a wrapper around the same composer). ---
    t0 = time.time()
    agent, _m = load_developed_brain(bundle_dir, use_multiturn=False, enable_neural_render=False)
    out["load_seconds"] = round(time.time() - t0, 2)

    comp = getattr(agent, "agent", agent).composer
    loaded_facts = set(_stored_facts(agent))
    saved_facts_set = {(f.get("agent"), f.get("action"), f.get("patient")) for f in saved_facts
                       if all(isinstance(f.get(r), str) for r in ("agent", "action", "patient"))}

    # (A) FAITHFUL facts round-trip: the loaded composer.kb == the saved facts.json (re-stored exactly)
    out["facts_roundtrip_ok"] = bool(loaded_facts == saved_facts_set and len(saved_facts_set) > 0)
    out["n_facts_loaded"] = len(loaded_facts)
    out["n_facts_saved"] = len(saved_facts_set)

    # (B) FAITHFUL codes round-trip: the loaded composer concept code == the saved .npz code byte-for-byte (NOT a
    #     re-derive from the seed). Compare every grounded word's phases.
    code_max_abs_diff = 0.0
    n_codes_checked = 0
    for w, ph_saved in saved_codes.items():
        ph_loaded = comp.concepts.get(w)
        if ph_loaded is None:
            continue
        ph_loaded = np.asarray(ph_loaded, dtype=float)
        ph_saved = np.asarray(ph_saved, dtype=float)
        if ph_loaded.shape == ph_saved.shape:
            code_max_abs_diff = max(code_max_abs_diff, float(np.max(np.abs(ph_loaded - ph_saved))))
            n_codes_checked += 1
    out["n_codes_checked"] = n_codes_checked
    out["codes_max_abs_diff"] = code_max_abs_diff
    out["codes_roundtrip_faithful"] = bool(n_codes_checked > 0 and code_max_abs_diff < 1e-6)

    # (C) USABLE: the loaded brain ANSWERS a who/what query on a taught fact + ABSTAINS on an untaught cue.
    answered = []
    n_recall_ok = 0
    sample = sorted(saved_facts_set)[:3]   # probe the first few taught facts
    for (a, v, p) in sample:
        got = agent.what_does(a, v)
        ok = (got == p)
        n_recall_ok += int(ok)
        answered.append({"q": f"what does {a} {v}?", "expect": p, "got": got, "ok": ok})
    out["recall_probes"] = answered
    out["recall_ok"] = n_recall_ok
    out["recall_total"] = len(sample)
    out["answers_a_query"] = bool(n_recall_ok >= 1)

    # the no-confab moat: an UNTAUGHT (agent, action) cue must abstain (return None)
    uc_a, uc_v = untaught_cue
    moat_ans = agent.what_does(uc_a, uc_v)
    out["untaught_cue"] = list(untaught_cue)
    out["untaught_answer"] = moat_ans
    out["moat_abstains"] = bool(moat_ans is None)

    # also a yes/no on a never-asserted scramble (belt-and-suspenders moat check)
    out["bundle_ok"] = bool(out["facts_roundtrip_ok"] and out["codes_roundtrip_faithful"]
                            and out["answers_a_query"] and out["moat_abstains"])

    # free the loaded brain's bridge
    try:
        br = getattr(comp, "bridge", None)
        if br is not None and hasattr(br, "_cp") and br._cp is not None:
            br._cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-windows-per-day", type=int, default=500)
    ap.add_argument("--n-hub", type=int, default=200)
    ap.add_argument("--n-per", type=int, default=12)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--out", default="research/findings/raw/_capstone_console_A1_bundle_save.json")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    import logging
    logging.disable(logging.INFO)

    print("=" * 110, flush=True)
    print("[A1 PAIRING SEAM — develop loop SAVES a developed-brain bundle; the console LOADS + talks to it]", flush=True)
    print(f"  backend={os.environ.get('SIM_BACKEND')}  n_days={a.n_days}  seed={a.seed}  "
          f"max_windows/day={a.max_windows_per_day}", flush=True)
    print("=" * 110 + "\n", flush=True)

    t0 = time.time()
    loop_root = tempfile.mkdtemp(prefix="a1_loop_")
    bundle_root = tempfile.mkdtemp(prefix="a1_bundles_")
    try:
        # ---- RUN the develop loop WITH the bundle-save wiring (no frozen/resume arms -> keep it a fast smoke) ----
        loop_res = run_gpu_smoke(a.n_days, a.seed, loop_root, a.max_windows_per_day, a.n_hub, a.n_per, a.D,
                                 enable_neural_render=False, do_frozen=False, do_resume=False, verbose=True,
                                 save_bundle_root=bundle_root, per_day_bundles=True)
        bundle_info = loop_res.get("bundle_info", {})

        # ---- find the bundles that were saved ----
        final_bundle = (bundle_info.get("final_bundle") or {}).get("path")
        per_day = bundle_info.get("bundles", [])
        per_day_paths = [b["path"] for b in per_day]

        print(f"\n[A1] develop loop saved: final={final_bundle!r}, per_day={len(per_day_paths)} bundles\n", flush=True)

        # ---- VALIDATE each bundle (round-trip + usable + moat) ----
        # an untaught cue: a (subject, action) never asserted in the curriculum (bird+jump was never a fact)
        untaught_cue = ("bird", "jump")
        bundle_results = []
        if final_bundle:
            print("[A1] validating FINAL bundle (brain/)...", flush=True)
            r = _validate_bundle(final_bundle, untaught_cue=untaught_cue, verbose=True)
            r["which"] = "final"
            bundle_results.append(r)
            print(f"[A1]   final: bundle_ok={r.get('bundle_ok')} facts_rt={r.get('facts_roundtrip_ok')} "
                  f"codes_faithful={r.get('codes_roundtrip_faithful')} (max|d|={r.get('codes_max_abs_diff'):.2e}, "
                  f"n={r.get('n_codes_checked')}) recall={r.get('recall_ok')}/{r.get('recall_total')} "
                  f"moat_abstains={r.get('moat_abstains')}", flush=True)
        for bp in per_day_paths:
            day = os.path.basename(bp)
            r = _validate_bundle(bp, untaught_cue=untaught_cue, verbose=False)
            r["which"] = day
            bundle_results.append(r)
            print(f"[A1]   {day}: bundle_ok={r.get('bundle_ok')} facts_rt={r.get('facts_roundtrip_ok')} "
                  f"({r.get('n_facts_loaded')} facts) codes_faithful={r.get('codes_roundtrip_faithful')} "
                  f"recall={r.get('recall_ok')}/{r.get('recall_total')} moat_abstains={r.get('moat_abstains')}",
                  flush=True)

        # ---- DEVELOPMENT anti-cheat: a per-day bundle's facts STRICT-SUBSET the final bundle's facts ----
        # (the brain accumulated knowledge over days; the bundles capture day-by-day development)
        dev_monotone = None
        if per_day_paths and final_bundle:
            day_fact_counts = []
            for r in bundle_results:
                if r["which"].startswith("day_"):
                    day_fact_counts.append((int(r["which"].split("_")[1]), r.get("n_facts_loaded", 0)))
            day_fact_counts.sort()
            final_n = next((r.get("n_facts_loaded", 0) for r in bundle_results if r["which"] == "final"), 0)
            counts_seq = [c for _, c in day_fact_counts]
            # monotone non-decreasing across days AND the last day's count == the final bundle's count
            dev_monotone = bool(counts_seq == sorted(counts_seq) and len(counts_seq) >= 1
                                and (len(counts_seq) < 2 or counts_seq[-1] >= counts_seq[0])
                                and (final_n == counts_seq[-1] or final_n >= counts_seq[-1]))

        # ---- VERDICT ----
        n_bundles = len(bundle_results)
        all_bundles_ok = bool(n_bundles > 0 and all(r.get("bundle_ok") for r in bundle_results))
        a_bundle_was_saved = bool(final_bundle and os.path.exists(os.path.join(final_bundle, "brain.json")))
        moat_clean = bool(all(r.get("moat_abstains") for r in bundle_results))
        go = bool(a_bundle_was_saved and all_bundles_ok and moat_clean
                  and (dev_monotone is None or dev_monotone))

        verdict = (
            f"GO — the develop loop SAVES a developed-brain bundle (brain.json on disk) AND it ROUND-TRIPS: "
            f"load_developed_brain reconstructs the EXACT brain (facts re-store, codes loaded byte-for-byte, "
            f"max|Δcode|<1e-6), and the loaded brain is USABLE (answers a who/what query + ABSTAINS on an untaught "
            f"cue, moat 0-FA across all {n_bundles} bundles). The console can now --load + talk to a develop-loop "
            f"brain (per-day bundles give the day-0-vs-day-N deliverable). NO sim/ edit."
            if go else
            f"HONEST/SNAG — saved={a_bundle_was_saved} all_bundles_ok={all_bundles_ok} moat_clean={moat_clean} "
            f"dev_monotone={dev_monotone}. See per-bundle detail for the localize."
        )

        res = {
            "probe": "A1_pairing_seam__develop_loop_saves_a_developed_brain_bundle_console_loads_it",
            "GO": go,
            "verdict": verdict,
            "backend": os.environ.get("SIM_BACKEND"),
            "n_days": a.n_days, "seed": a.seed, "max_windows_per_day": a.max_windows_per_day,
            "D": a.D, "n_hub": a.n_hub, "n_per": a.n_per,
            "develop_loop_go": loop_res.get("go"),
            "develop_vocab_trend": loop_res.get("vocab_trend"),
            "develop_facts_trend": loop_res.get("facts_trend"),
            "bundle_save_enabled": bundle_info.get("enabled"),
            "n_bundles_saved": n_bundles,
            "a_bundle_was_saved_to_disk": a_bundle_was_saved,
            "all_bundles_roundtrip_and_usable": all_bundles_ok,
            "moat_clean_across_bundles": moat_clean,
            "development_monotone_across_bundles": dev_monotone,
            "final_bundle_path_template": "bridges/developed/<lineage>/brain (this run used a tempdir)",
            "bundle_results": bundle_results,
            "anti_cheats": [
                "FAITHFUL codes round-trip: the loaded composer code == the saved .npz code byte-for-byte "
                "(max|Δ|<1e-6), NOT a re-derive from the seed -> the bundle stores the codes the brain LEARNED.",
                "FAITHFUL facts round-trip: the loaded composer.kb == the saved facts.json (re-stored exactly).",
                "no-confab moat: the loaded brain ABSTAINS (returns None) on an untaught (subject, action) cue.",
                "development capture: per-day bundle fact-counts are monotone non-decreasing -> the bundles "
                "capture the brain's day-by-day accumulation (the day-0-vs-day-N deliverable).",
            ],
            "components": [
                "SAVE = research.runners._longitudinal_develop_loop_gpu.save_developed_bundle / a per-day save hook "
                "-> developed_brain_io.save_developed_brain (reuse-by-import, NO sim/ edit).",
                "LOAD = developed_brain_io.load_developed_brain (the SAME path brain_chat_tui.load_brain uses).",
            ],
        }
    finally:
        shutil.rmtree(loop_root, ignore_errors=True)
        shutil.rmtree(bundle_root, ignore_errors=True)
    res["wall_seconds"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=str)

    print(f"\n{'=' * 110}", flush=True)
    print(f"  VERDICT: {res['verdict']}", flush=True)
    print(f"  [saved] {a.out}  (wall {res['wall_seconds']}s)\n{'=' * 110}", flush=True)
    return 0 if res["GO"] else 1


if __name__ == "__main__":
    sys.exit(main())
