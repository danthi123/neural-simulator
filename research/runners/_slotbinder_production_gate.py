"""_slotbinder_production_gate.py -- the PRODUCTION-SCALE gate for BRAIN_COMPOSER_KIND=slotbinder through the REAL
chat path (webapp.server._build_chat_brain -> research.runners.developed_brain_io.load_developed_brain ->
MultiTurnAgent -> BrainConversationalAgent -> a SlotBinderComposer), per
research/findings/2026-09-24-slotbinder-production-composer-gate-PREREG.md.

WHY THIS DIFFERS FROM THE L3 LATENCY DE-RISK RUNNER (_slotbinder_l3_latency_derisk.py, 2026-09-05/2026-09-24):
that runner constructs `SlotBinderComposer(...)` DIRECTLY -- it never calls `load_developed_brain` or
`webapp.server._build_chat_brain`, so it never exercises `developed_brain_io._restore_facts`'s REAL production
fact-restore semantics (which, for composer_kind='slotbinder', re-`store()`s -- re-teaches -- EVERY fact in the
bundle unconditionally, since SlotBinderComposer has no `.kb` composite fast-path -- see that module's own
docstring). This runner routes through the ACTUAL webapp entry point instead, so the measured build/teach cost is
the real wire-in's own cost, not a hand-picked partial-teach shortcut.

FACT SCALE: the REAL day_33 bundle is K=2020/KF=1195 (404 facts, 788-word vocab) regardless of how many of its
facts are actually taught, because `load_developed_brain` always sizes `slotbinder_max_facts=len(facts)` and
`slotbinder_prewire_facts=list(facts)` from the BUNDLE it is given (see developed_brain_io.py's own
`_slotbinder_kwargs` construction) -- there is no argument to make it teach a partial sample of a 404-fact bundle.
So "the largest fact count that fits" is made a genuine, measurable variable by constructing a SMALLER, REAL
bundle: a bona fide `save_developed_brain` bundle containing a genuine sample of N real day_33 facts (the SAME
`_sample_facts` seeded sampler L2/L3 use, unmodified) restricted to the REAL grounded codes those facts' own
words carry (not fresh random codes) -- a real sub-corpus, not a synthetic one. `load_developed_brain` then
re-teaches ALL N of ITS facts exactly as it would for the full 404-fact bundle -- the identical code path, at a
scale chosen by measured GPU wall-clock budget (VRAM is not the binding constraint at this topology -- see the
prereg's own sizing note: fanout=32 pins effective KF at 32 for any real vocab size, so nnz -- and VRAM -- scales
with K=5N alone, ~14,160 synapses/fact, trivial against 24 GB at any N this corpus could support).

For each seed this script:
 1. samples N_FACTS real facts from the live day_33 corpus + builds/saves a standalone developed-brain bundle
    (an 'rf'-taught staging composer, matching the batch-consolidation scenario slotbinder_composer.py's own
    docstring names) under a scratch directory (never under the repo).
 2. loads that SAME bundle TWICE through `webapp.server._build_chat_brain(bundle, "stub")` -- once with
    BRAIN_COMPOSER_KIND=slotbinder (the arm under test), once with BRAIN_COMPOSER_KIND=rf (the FHRR reference
    arm) -- sequentially in ONE process on the SAME GPU, so the latency comparison is apples-to-apples and only
    one brain-loading GPU process is ever resident at a time.
 3. per arm: times the build (for slotbinder this IS the real per-fact re-teach `_restore_facts` performs),
    times each per-fact `chat.inner.what_does(agent, action)` query, a moat probe (a never-stored (agent,action)
    pair must abstain), and a mismatch probe (cross fact i's agent with fact j's action must not leak fact i's
    patient).
 4. PARITY: slotbinder's answer must equal the FHRR arm's answer, question-for-question.
 5. ABLATION (slotbinder only, the falsifiability check this gate must be able to FAIL): zeroes every synapse in
    the built SlotBinder bridge (`composer._b.cp_connections.data[:] = 0`) and re-runs the SAME per-fact
    queries -- recall must COLLAPSE. A composer answering via a host shortcut (not the taught synapses) would be
    unaffected by this, so a collapse is the falsifiable, positive evidence the recall is genuinely load-bearing
    on the substrate.
 6. optionally (--check-flagoff, cheap, run once for the dev seed, not the 6-seed battery): builds the SAME
    bundle a third time with BRAIN_COMPOSER_KIND deliberately UNSET, confirming it resolves to
    webapp.server._DEVELOPED_COMPOSER_KIND_DEFAULT_OVERRIDE ('onebrain' today) exactly as an unmodified `main`
    checkout would (this script adds no new file to webapp/ or research/runners/ wiring modules, so byte-identity
    of the flag-off path holds by construction -- `git diff main` is the authoritative check; this is an
    empirical smoke on top of that).

CPU/numpy is REFUSED by default (this is a GPU production gate) -- pass --allow-numpy-debug for a
correctness-only dry run of this SCRIPT's own logic (never a valid gate result; the output JSON is stamped
`sim_backend` either way so a numpy debug run can never be mistaken for a real gate result).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# DISCOVERED INTERACTION BUG (this gate's own dev-seed sizing pass, 2026-09-24; see the prereg's "residuals"
# section + research/FAILURE_LOG.md): webapp.server._build_chat_brain's developed-bundle branch attaches the
# SHIPPED curated-LTM core (board #133, default-ON) by wrapping the composer in a TieredFactStore
# (research/runners/tiered_fact_store.py) whenever `_resolve_ltm_bundle()` finds one -- and TieredFactStore's
# `__getattr__('kb')` proxies straight to `self.buffer.kb`, which SlotBinderComposer does not have (facts live in
# `.facts`, taught into per-slot synapses -- see slotbinder_composer.py's own docstring). ChatBrain.__init__ ->
# `_refresh_facts()` reads `comp.kb` unconditionally, so composer_kind='slotbinder' + the LTM ship-default
# BOTH on crashes with `AttributeError: 'SlotBinderComposer' object has no attribute 'kb'` -- a genuine,
# previously-undiscovered cross-feature incompatibility (neither the L1/L2/L3 SlotBinder findings nor the LTM
# ship-default's own GO ever exercised BOTH together: the SlotBinder findings called `load_developed_brain`/the
# composer directly, never `ChatBrain`; the LTM findings never set BRAIN_COMPOSER_KIND=slotbinder). This gate
# is about the SlotBinder wire-in specifically, not the (separately shipped, separately gated) LTM tier, so it
# uses the LTM tier's OWN documented, already-shipped escape (`BRAIN_LTM_SHIP_DEFAULT=off` -- webapp/server.py's
# `_resolve_ltm_bundle` docstring) to isolate the composer-level question -- NOT a new code change, and NOT a
# fix for the crash (which is reported as an honest residual, not silently routed around).
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "off")

import numpy as np  # noqa: E402

from research.runners._slotbinder_l2_sparse_derisk import (  # noqa: E402
    _load_live_bundle, _sample_facts, _find_live_bundle_dir,
)
from research.runners.developed_brain_io import _load_codes_npz  # noqa: E402


# ============================================================================================================
# sample-bundle construction (a REAL sub-corpus: real facts, real learned codes, N chosen by the caller)
# ============================================================================================================

def _sample_vocab(sample_facts):
    words = set()
    for f in sample_facts:
        for role in ("agent", "action", "patient"):
            w = f.get(role)
            if isinstance(w, str):
                words.add(w)
    return sorted(words)


def _default_scratch_dir():
    d = os.environ.get("SLOTBINDER_GATE_SCRATCH") or os.path.join(
        os.path.expanduser("~"), ".cache", "slotbinder_production_gate")
    os.makedirs(d, exist_ok=True)
    return d


def build_sample_bundle(seed, n_facts, out_dir):
    """Sample N real day_33 facts, teach them into a fresh 'rf' staging composer (using day_33's OWN real
    grounded codes for the words they touch -- a genuine sub-corpus, not synthetic content), and persist via
    `save_developed_brain` -- a bundle `webapp.server._build_chat_brain` can load exactly like day_33 itself.
    Returns (bundle_dir, sample_facts, sampled_corpus_indices, words_missing_a_real_code)."""
    from research.runners.brain_conversational_agent import BrainConversationalAgent
    from research.runners.developed_brain_io import save_developed_brain

    vocab_full, facts_full, brain_full = _load_live_bundle()
    sample, idx = _sample_facts(facts_full, seed, n_facts)
    vocab = _sample_vocab(sample)
    live_dir = _find_live_bundle_dir()
    real_codes = _load_codes_npz(live_dir)
    codes = {w: real_codes[w] for w in vocab if w in real_codes}
    missing = [w for w in vocab if w not in real_codes]

    concepts = {w: None for w in vocab}
    staging = BrainConversationalAgent(seed=seed, concepts=concepts, grounded_codes=codes, composer_kind="rf")
    # RFPhasorComposer.store() has no return value (unlike SlotBinderComposer.store(), which returns True/False)
    # -- a rejected fact raises instead. Catch it explicitly so a bad sampled fact is reported with context.
    for f in sample:
        try:
            staging.composer.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
        except Exception as e:
            raise RuntimeError(f"could not teach a REAL sampled day_33 fact into the rf staging composer: "
                               f"{f} ({type(e).__name__}: {e})") from e

    bundle_dir = os.path.join(out_dir, f"seed{seed}_n{n_facts}")
    save_developed_brain(staging, bundle_dir, seed=seed, composer_kind="rf",
                         extra_metadata={"provenance": "slotbinder_production_gate sample bundle",
                                         "source_live_bundle": live_dir,
                                         "sampled_corpus_indices": idx, "n_facts_requested": n_facts})
    return bundle_dir, sample, idx, missing


# ============================================================================================================
# GPU memory snapshot (cupy pool + nvidia-smi ground truth; either may be absent, both are best-effort)
# ============================================================================================================

def _gpu_mem_snapshot():
    snap = {}
    try:
        import cupy
        pool = cupy.get_default_memory_pool()
        snap["cupy_used_bytes"] = int(pool.used_bytes())
        snap["cupy_total_bytes"] = int(pool.total_bytes())
    except Exception as e:
        snap["cupy_error"] = f"{type(e).__name__}: {e}"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        if out.returncode == 0 and out.stdout.strip():
            used, total = out.stdout.strip().splitlines()[0].split(",")
            snap["nvidia_smi_used_mib"] = int(used.strip())
            snap["nvidia_smi_total_mib"] = int(total.strip())
    except Exception as e:
        snap["nvidia_smi_error"] = f"{type(e).__name__}: {e}"
    return snap


def _free_gpu():
    gc.collect()
    try:
        import cupy
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ============================================================================================================
# one arm: build (real teach for slotbinder) + recall/moat/mismatch queries (+ ablation for slotbinder)
# ============================================================================================================

def run_arm(bundle_dir, composer_kind, fanout, sample, seed, renderer="stub", run_ablation=True):
    """composer_kind in {'slotbinder', 'rf'}, or None to mean "leave BRAIN_COMPOSER_KIND UNSET" (the flag-off
    check). Routes through webapp.server._build_chat_brain -- the SAME function /api/brain-chat calls."""
    if composer_kind is None:
        os.environ.pop("BRAIN_COMPOSER_KIND", None)
    else:
        os.environ["BRAIN_COMPOSER_KIND"] = composer_kind
    if composer_kind == "slotbinder":
        os.environ["BRAIN_SLOTBINDER_FANOUT"] = str(fanout)

    from webapp.server import _build_chat_brain

    t0 = time.time()
    chat, source = _build_chat_brain(bundle_dir, renderer)
    build_s = time.time() - t0
    inner = chat.inner
    comp = inner.composer
    resolved_class = type(comp).__name__

    per_fact = []
    for f in sample:
        a, v, p = f["agent"], f["action"], f["patient"]
        t0 = time.time()
        got = inner.what_does(a, v)
        dt = time.time() - t0
        per_fact.append({"agent": a, "action": v, "expected_patient": p, "got_patient": got,
                         "hit": got == p, "query_latency_s": dt})

    stored_pairs = {(f["agent"], f["action"]) for f in sample}
    words = sorted({w for f in sample for w in (f["agent"], f["action"], f["patient"])})
    rng = np.random.default_rng(seed * 97 + 3)
    moat = None
    for _try in range(300):
        a, v = words[rng.integers(len(words))], words[rng.integers(len(words))]
        if (a, v) in stored_pairs:
            continue
        t0 = time.time()
        got = inner.what_does(a, v)
        dt = time.time() - t0
        moat = {"agent": a, "action": v, "abstained": got is None, "query_latency_s": dt}
        break

    mismatch = None
    if len(sample) >= 2:
        a, v = sample[0]["agent"], sample[1]["action"]
        if (a, v) not in stored_pairs:
            t0 = time.time()
            got = inner.what_does(a, v)
            dt = time.time() - t0
            mismatch = {"agent": a, "action_from_other_fact": v,
                       "did_not_leak_fact0_patient": got != sample[0]["patient"], "query_latency_s": dt}

    result = {
        "composer_kind_requested": composer_kind, "composer_class_resolved": resolved_class, "source": source,
        "build_seconds": build_s,
        "per_fact": per_fact,
        "moat_probe": moat, "mismatch_probe": mismatch,
        "recall_accuracy": (sum(r["hit"] for r in per_fact) / len(per_fact)) if per_fact else None,
        "moat_pass": bool(moat and moat["abstained"]),
        "mismatch_pass": bool(mismatch is None or mismatch["did_not_leak_fact0_patient"]),
        "gpu_mem_after_build_and_queries": _gpu_mem_snapshot(),
    }

    # ABLATION (falsifiability check): zero the built SlotBinder bridge's OWN synapses (both the plastic
    # slot->filler associative weights AND the fixed hold/competition weights -- a coarse but unambiguous
    # lesion), then re-run the identical per-fact queries. A genuine spiking recall MUST collapse; a composer
    # that secretly answered from a host cache (not the taught synapses) would be unaffected by this.
    if run_ablation and composer_kind == "slotbinder":
        b = getattr(comp, "_b", None)
        conn = getattr(b, "cp_connections", None) if b is not None else None
        if conn is not None and hasattr(conn, "data"):
            nnz_before = int(conn.nnz)
            nonzero_before = int((conn.data != 0).sum()) if hasattr(conn.data, "sum") else None
            conn.data[:] = 0
            ablated = []
            for f in sample:
                a, v, p = f["agent"], f["action"], f["patient"]
                got = inner.what_does(a, v)
                ablated.append({"agent": a, "action": v, "expected_patient": p, "got_patient": got, "hit": got == p})
            result["ablation_zeroed_synapses"] = {
                "nnz": nnz_before, "nonzero_weights_before_zeroing": nonzero_before,
                "recall_accuracy": (sum(r["hit"] for r in ablated) / len(ablated)) if ablated else None,
                "per_fact": ablated,
            }
        else:
            result["ablation_zeroed_synapses"] = {"skipped_reason": "composer built no cp_connections bridge handle"}

    return chat, result


# ============================================================================================================
# comparison across arms
# ============================================================================================================

def _parity(slot_result, rf_result):
    rows = []
    for sb, rf in zip(slot_result["per_fact"], rf_result["per_fact"]):
        assert sb["agent"] == rf["agent"] and sb["action"] == rf["action"], "per_fact rows must be co-indexed"
        rows.append({"agent": sb["agent"], "action": sb["action"], "expected_patient": sb["expected_patient"],
                    "slotbinder_answer": sb["got_patient"], "fhrr_answer": rf["got_patient"],
                    "match": sb["got_patient"] == rf["got_patient"]})
    return {"rows": rows, "parity_rate": (sum(r["match"] for r in rows) / len(rows)) if rows else None}


def _pctiles(latencies):
    if not latencies:
        return {"p50": None, "p95": None, "mean": None, "max": None, "min": None, "n": 0}
    arr = np.asarray(latencies, dtype=float)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)),
           "mean": float(arr.mean()), "max": float(arr.max()), "min": float(arr.min()), "n": int(arr.size)}


# ============================================================================================================
# main
# ============================================================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-facts", type=int, default=40)
    ap.add_argument("--fanout", type=int, default=32)
    ap.add_argument("--renderer", type=str, default="stub")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--scratch-dir", type=str, default=None)
    ap.add_argument("--check-flagoff", action="store_true",
                    help="also build a THIRD arm with BRAIN_COMPOSER_KIND unset -- cheap, run once (the dev "
                         "seed), not for every seed of the 6-seed battery")
    ap.add_argument("--no-ablation", action="store_true", help="skip the zeroed-synapses falsifiability check")
    ap.add_argument("--allow-numpy-debug", action="store_true",
                    help="run on SIM_BACKEND=numpy for a correctness-only dry run of THIS SCRIPT -- never a "
                         "valid production-gate result (the output JSON's sim_backend field says so)")
    args = ap.parse_args()

    backend = os.environ.get("SIM_BACKEND", "")
    if backend != "cupy" and not args.allow_numpy_debug:
        print(f"REFUSING: SIM_BACKEND={backend!r} -- this is a GPU production gate, set SIM_BACKEND=cupy "
              f"(or pass --allow-numpy-debug for a correctness-only dry run that is NEVER a valid gate result)",
              file=sys.stderr)
        return 2

    scratch = args.scratch_dir or _default_scratch_dir()
    t_bundle0 = time.time()
    bundle_dir, sample, idx, missing_codes = build_sample_bundle(args.seed, args.n_facts, scratch)
    bundle_build_s = time.time() - t_bundle0
    print(f"[seed {args.seed}] sample bundle: {bundle_dir} n_facts={len(sample)} corpus_idx={idx} "
          f"missing_real_codes_for={missing_codes} (staging build {bundle_build_s:.1f}s)", flush=True)

    arm_results = {}
    for kind in ("slotbinder", "rf"):
        print(f"[seed {args.seed}] running arm={kind} ...", flush=True)
        t0 = time.time()
        chat, res = run_arm(bundle_dir, kind, args.fanout, sample, args.seed, renderer=args.renderer,
                            run_ablation=(not args.no_ablation))
        res["wall_clock_s"] = time.time() - t0
        arm_results[kind] = res
        print(f"[seed {args.seed}] arm={kind} resolved={res['composer_class_resolved']} "
              f"build={res['build_seconds']:.2f}s recall={res['recall_accuracy']} "
              f"moat_pass={res['moat_pass']} mismatch_pass={res['mismatch_pass']} "
              f"wall={res['wall_clock_s']:.1f}s", flush=True)
        del chat, res
        _free_gpu()

    flagoff = None
    if args.check_flagoff:
        print(f"[seed {args.seed}] running arm=<flag-off, BRAIN_COMPOSER_KIND unset> ...", flush=True)
        t0 = time.time()
        chat, flagoff = run_arm(bundle_dir, None, args.fanout, sample, args.seed, renderer=args.renderer,
                                run_ablation=False)
        flagoff["wall_clock_s"] = time.time() - t0
        print(f"[seed {args.seed}] flag-off resolved={flagoff['composer_class_resolved']} "
              f"(expect the production default -- webapp.server._DEVELOPED_COMPOSER_KIND_DEFAULT_OVERRIDE) "
              f"recall={flagoff['recall_accuracy']} moat_pass={flagoff['moat_pass']}", flush=True)
        del chat
        _free_gpu()

    parity = _parity(arm_results["slotbinder"], arm_results["rf"])
    lat_slot = _pctiles([r["query_latency_s"] for r in arm_results["slotbinder"]["per_fact"]])
    lat_rf = _pctiles([r["query_latency_s"] for r in arm_results["rf"]["per_fact"]])

    ablation = arm_results["slotbinder"].get("ablation_zeroed_synapses")
    ablation_collapses = None
    if ablation is not None and "recall_accuracy" in ablation:
        sb_intact = arm_results["slotbinder"]["recall_accuracy"]
        ablation_collapses = (ablation["recall_accuracy"] is not None and sb_intact is not None
                              and ablation["recall_accuracy"] < sb_intact)

    verdict_criteria = {
        "recall_ge_fhrr": bool((arm_results["slotbinder"]["recall_accuracy"] or 0)
                              >= (arm_results["rf"]["recall_accuracy"] or 0)),
        "parity_1_0": parity["parity_rate"] == 1.0,
        "slotbinder_moat_pass": arm_results["slotbinder"]["moat_pass"],
        "slotbinder_mismatch_pass": arm_results["slotbinder"]["mismatch_pass"],
        "fhrr_moat_pass": arm_results["rf"]["moat_pass"],
        "fhrr_mismatch_pass": arm_results["rf"]["mismatch_pass"],
        "ablation_falsifies_intact_pass": ablation_collapses,
    }

    summary = {
        "seed": args.seed, "n_facts": args.n_facts, "fanout": args.fanout,
        "sim_backend": os.environ.get("SIM_BACKEND", ""),
        "sampled_corpus_indices": idx, "missing_real_codes_for": missing_codes,
        "bundle_dir": bundle_dir, "bundle_staging_build_seconds": bundle_build_s,
        "arms": arm_results, "flagoff_check": flagoff,
        "parity": parity,
        "latency_slotbinder_per_fact_query_s": lat_slot,
        "latency_fhrr_per_fact_query_s": lat_rf,
        "ablation_collapses_recall": ablation_collapses,
        "verdict_criteria": verdict_criteria,
        "verdict": "GO" if all(v is True for v in verdict_criteria.values()) else "NOT-YET",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\n[seed {args.seed}] -> {args.out}")
    print(json.dumps(verdict_criteria, indent=2))
    print(f"[seed {args.seed}] verdict (this seed only, not a multi-seed GO): {summary['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
