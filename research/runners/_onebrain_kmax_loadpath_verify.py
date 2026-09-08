"""Smoke verification for the onebrain_k_max LOAD-PATH thread (rank-1 scaffold-retirement backlog item,
2026-09-08): `load_developed_brain` previously had NO `onebrain_k_max` parameter at all (and `MultiTurnAgent`,
the ONLY path `load_developed_brain(use_multiturn=True)` and webapp/server.py's `_build_chat_brain` ever use,
dropped it entirely too), so a >32-fact `composer_kind='onebrain'` bundle crashed on reload with
"OneBrainComposer store full: k_max=32 reached" -- the composer's own hardcoded default was the only value that
ever reached it. This is the named blocker in
research/findings/2026-09-08-rank1-composer-rebuild-rf-to-onebrain-real-bundle-parity-GO.md.

Confirms, on the REAL 404-fact rebuilt onebrain bundle (bridges/developed/rebuilt_scale787_onebrain, the literal
rf->onebrain rebuild of the deployed scale787/day_33 brain from `_rank1_composer_rebuild_onebrain_verify.py`):
  (1) AFTER-FIX: the default reload (`onebrain_k_max=None`) auto-sizes k_max from the bundle's own fact count
      (`len(facts)+16`) and restores ALL 404 facts (composer.kb length == 404), not truncated/crashed at 32.
  (2) BEFORE-FIX REPRO: forcing `onebrain_k_max=32` (the prior hardcoded ceiling) on the SAME bundle still
      crashes at fact #33 -- proving the fix is load-bearing (the bundle genuinely needs >32), not a no-op.
  (3) NO REGRESSION: the 'rf' PRODUCTION bundle (scale787/day_33, composer_kind stays 'rf' -- the untouched
      production default) reloads to the identical composer class, fact count, and has no k_max concept at all
      (RFPhasorComposer never reads onebrain_k_max), confirming this change is inert on the production path.

Cost-routing: numpy/CPU (matching the rank-1 runner this smoke re-verifies against); no GPU needed.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_FACT_SHARD_RETRIEVAL", "1")
os.environ.setdefault("BRAIN_COMPOSER_MERGE", "0")

try:
    from tools.verdict import Verdict
except Exception:
    Verdict = None

from research.runners.developed_brain_io import load_developed_brain, _read_manifest, _load_facts_json

DEFAULT_ONEBRAIN_BUNDLE = "/home/dant123/Projects/sim/bridges/developed/rebuilt_scale787_onebrain"
DEFAULT_RF_BUNDLE = "/home/dant123/Projects/sim/bridges/developed/scale787/day_33"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onebrain-bundle", default=DEFAULT_ONEBRAIN_BUNDLE)
    ap.add_argument("--rf-bundle", default=DEFAULT_RF_BUNDLE)
    ap.add_argument("--out", default="research/findings/raw/_onebrain_kmax_loadpath_thread/verify_smoke.json")
    args = ap.parse_args()

    result = {"runner": "_onebrain_kmax_loadpath_verify", "sim_backend": os.environ.get("SIM_BACKEND")}

    # ---- (0) ground truth: how many facts are actually in the bundle? ----
    manifest_ob = _read_manifest(args.onebrain_bundle)
    facts_ob = _load_facts_json(args.onebrain_bundle)
    n_facts_stored = len(facts_ob)
    result["bundle"] = args.onebrain_bundle
    result["manifest_composer_kind"] = manifest_ob.get("composer_kind")
    result["manifest_n_facts"] = manifest_ob.get("n_facts")
    result["actual_facts_json_len"] = n_facts_stored
    print(f"[smoke] onebrain bundle: manifest composer_kind={manifest_ob.get('composer_kind')!r} "
          f"manifest_n_facts={manifest_ob.get('n_facts')} actual facts.json len={n_facts_stored}", flush=True)

    # ---- (1) AFTER-FIX: default load (onebrain_k_max=None -> auto-sized to n_facts+16) ----
    t0 = time.time()
    agent_after, manifest_after = load_developed_brain(args.onebrain_bundle, use_multiturn=False,
                                                        enable_neural_render=False, defer_parser=True)
    after_s = time.time() - t0
    kb_after = getattr(agent_after.composer, "kb", None)
    kmax_after = getattr(agent_after.composer, "k_max", None)
    n_recalled_after = len(kb_after) if kb_after is not None else None
    result["after_fix"] = {
        "load_s": after_s, "composer_class": type(agent_after.composer).__name__,
        "k_max": kmax_after, "n_recalled": n_recalled_after,
    }
    print(f"[smoke] AFTER-FIX (onebrain_k_max=None, auto-sized): k_max={kmax_after} "
          f"n_recalled={n_recalled_after} (stored={n_facts_stored}) load_s={after_s:.1f}", flush=True)

    after_fix_full_recall = (n_recalled_after == n_facts_stored) and (n_facts_stored > 32)

    # ---- (2) BEFORE-FIX repro: force onebrain_k_max=32 (the prior hardcoded default) -> must crash ----
    before_fix_crashed = False
    try:
        t0 = time.time()
        agent_before, _m = load_developed_brain(args.onebrain_bundle, use_multiturn=False,
                                                 enable_neural_render=False, defer_parser=True,
                                                 onebrain_k_max=32)
        before_s = time.time() - t0
        kb_before = getattr(agent_before.composer, "kb", None)
        n_recalled_before = len(kb_before) if kb_before is not None else None
        result["before_fix_forced_32"] = {"load_s": before_s, "n_recalled": n_recalled_before, "crashed": False}
        print(f"[smoke] BEFORE-FIX repro (onebrain_k_max=32 forced): n_recalled={n_recalled_before} "
              f"(NO crash -- unexpected if >32 facts)", flush=True)
    except RuntimeError as e:
        before_fix_crashed = True
        result["before_fix_forced_32"] = {"crashed": True, "error": repr(e)}
        print(f"[smoke] BEFORE-FIX repro (onebrain_k_max=32 forced): CRASHED as expected: {e}", flush=True)

    # ---- (3) 'rf' bundle (production default) reload -- confirm UNCHANGED ----
    t0 = time.time()
    agent_rf, manifest_rf = load_developed_brain(args.rf_bundle, use_multiturn=False, enable_neural_render=False)
    rf_s = time.time() - t0
    kb_rf = getattr(agent_rf.composer, "kb", None)
    n_recalled_rf = len(kb_rf) if kb_rf is not None else None
    result["rf_bundle"] = {
        "bundle": args.rf_bundle, "manifest_composer_kind": manifest_rf.get("composer_kind"),
        "composer_class": type(agent_rf.composer).__name__,
        "manifest_n_facts": manifest_rf.get("n_facts"), "n_recalled": n_recalled_rf, "load_s": rf_s,
        "has_k_max_attr": hasattr(agent_rf.composer, "k_max"),
    }
    print(f"[smoke] rf bundle (production default): composer={type(agent_rf.composer).__name__} "
          f"manifest_n_facts={manifest_rf.get('n_facts')} n_recalled={n_recalled_rf} "
          f"has_k_max_attr={hasattr(agent_rf.composer, 'k_max')} load_s={rf_s:.1f}", flush=True)

    rf_unaffected = (
        manifest_rf.get("composer_kind") == "rf"
        and type(agent_rf.composer).__name__ == "RFPhasorComposer"
        and n_recalled_rf == manifest_rf.get("n_facts")
        and not hasattr(agent_rf.composer, "k_max")   # RFPhasorComposer has no k_max concept at all
    )

    go = bool(after_fix_full_recall and before_fix_crashed and rf_unaffected)
    result["go_flags"] = {
        "after_fix_full_recall_no_truncation": after_fix_full_recall,
        "before_fix_repro_crashed_at_kmax32": before_fix_crashed,
        "rf_bundle_unaffected": rf_unaffected,
    }

    verdict_block = None
    if Verdict is not None:
        v = Verdict("onebrain_k_max load-path thread: no truncation, prior default still crashes, rf untouched")
        v.require("after-fix reload restores ALL stored facts (no truncation)", after_fix_full_recall, expect=True)
        v.require("before-fix repro (k_max=32 forced) still crashes on the SAME bundle (fix is load-bearing)",
                  before_fix_crashed, expect=True)
        v.require("'rf' production bundle unaffected (same composer/class family, full recall, no k_max)",
                  rf_unaffected, expect=True)
        try:
            _ = v.decide(go=go)
            verdict_block = v.to_dict()
        except Exception as e:
            verdict_block = {"error": repr(e)}

    result["verdict"] = (verdict_block or {}).get("status", "GO" if go else "NO-GO")
    result["verdict_block"] = verdict_block
    # surface preconditions at top level (gates/verdict_preconditions reads this key, mirroring
    # _rank1_composer_rebuild_onebrain_verify.py's own convention).
    result["preconditions"] = (verdict_block.get("preconditions") if isinstance(verdict_block, dict) else []) or []
    print(f"\n[smoke] VERDICT = {result['verdict']}  flags={result['go_flags']}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(f"[smoke] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
