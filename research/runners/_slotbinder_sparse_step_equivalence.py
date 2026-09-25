"""_slotbinder_sparse_step_equivalence.py -- EQUIVALENCE GATE + per-N timing for SlotBinderComposer's opt-in
event-driven step (`sparse_step=True` -> `cfg.sparse_activity_step`, 2026-09-25).

WHAT IT CHECKS. For a real seed-S sample of N day_33 facts (the SAME `_sample_facts` sampler the production gate
uses), it builds the binder twice -- sparse_step OFF (the unchanged path) and ON -- and runs the production gate's
own slotbinder-arm protocol on each: teach every fact (`store`), ask every fact's `query_patient(agent, action)`,
the moat probe and the mismatch probe (same RNG draw as `_slotbinder_production_gate.run_arm`), then the ablation
(zero every synapse, re-ask every fact). It then compares, bit for bit (sha256 of the raw bytes):
  - the per-neuron firing thresholds right after build (the seed actually controls the substrate in both);
  - every synapse weight after the teach (the stored binding state);
  - every answer (intact, moat, mismatch, ablated);
  - every synapse weight after the queries and after the ablation re-queries (reads drive Hebbian changes in the
    un-gated scaffold, so this is a second, stricter weight check);
  - the final neuron state (v, u, firing, every conductance).
A tolerance is NOT used: on numpy the event-driven path must be bit-identical, and any difference fails the gate.

TIMING. `--teach-facts K` / `--queries K` / `--no-ablation` shrink the protocol for the large-N timing points
(N=128/404 on numpy, where the OFF path costs hours); equality is still checked on whatever subset was run.

Output: one JSON with, per N, `equal` (all comparisons), each comparison's boolean, and per path the per-fact teach
seconds, per-read seconds and read counts. Any mismatched weight comparison (`weights_after_teach/queries/
ablation`) additionally gets a `..._first_diff` entry (`n_diff`, `idx`, `max_abs`) -- on cupy, where bit-identity
is NOT expected (see below), this is what a caller checks against an explicit numeric criterion instead of the
plain boolean. CPU/numpy by default (this is the equivalence instrument); cupy is allowed for timing but its
transpose matvec is cuSPARSE's atomic scatter in BOTH paths, so bit-identity is not expected there (the verdict
field says which backend ran) -- see
research/findings/2026-09-24-slotbinder-production-composer-gate-PREREG.md AMENDMENT 3 item (5) for the pass
criterion this instrument's cupy output is read against before a GPU production-gate run trusts the flag.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402

from research.runners._slotbinder_l2_sparse_derisk import _load_live_bundle, _sample_facts  # noqa: E402
from research.runners.slotbinder_composer import SlotBinderComposer  # noqa: E402
from sim.backend import to_host  # noqa: E402

_STATE_ATTRS = ("cp_membrane_potential_v", "cp_recovery_variable_u", "cp_firing_states", "cp_prev_firing_states",
                "cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_nmda", "cp_conductance_g_nmda_rise",
                "cp_conductance_g_nmda_recurrent", "cp_conductance_g_nmda_recurrent_rise")


def _h(arr):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(to_host(arr))).tobytes()).hexdigest()


def _words(sample):
    return sorted({f[r] for f in sample for r in ("agent", "action", "patient") if isinstance(f.get(r), str)})


def _moat_pair(sample, seed):
    """The production gate's own moat draw (run_arm: rng = default_rng(seed*97+3), first never-stored pair)."""
    stored = {(f["agent"], f["action"]) for f in sample}
    words = sorted({w for f in sample for w in (f["agent"], f["action"], f["patient"])})
    rng = np.random.default_rng(seed * 97 + 3)
    for _ in range(300):
        a, v = words[rng.integers(len(words))], words[rng.integers(len(words))]
        if (a, v) not in stored:
            return a, v
    return None


def run_path(sample, seed, fanout, sparse_step, n_teach, n_query, ablation, log):
    N = len(sample)
    c = SlotBinderComposer(seed=seed, vocab=_words(sample), max_facts=N, fanout=fanout,
                           prewire_facts=list(sample), sparse_step=sparse_step)
    t0 = time.time()
    c._ensure()
    b = c._b
    out = {"sparse_step": bool(sparse_step), "build_s": time.time() - t0, "nnz": int(b.cp_connections.nnz),
           "n_neurons": int(b.core_config.num_neurons),
           "sparse_activity_step_dispatches": bool(b._sparse_activity_step_can_dispatch(b.core_config)),
           "thresholds_sha256": _h(b.cp_neuron_firing_thresholds)}
    n_reads = [0]
    orig_read = c._read_slot

    def counted(slot):
        n_reads[0] += 1
        return orig_read(slot)

    c._read_slot = counted
    teach_t = []
    for f in sample[:n_teach]:
        t = time.time()
        ok = c.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
        teach_t.append(time.time() - t)
        if ok is not True:
            raise RuntimeError(f"store() rejected a real sampled fact: {f}")
        log(f"  [{'on' if sparse_step else 'off'} N={N}] teach {len(teach_t)}/{n_teach} {teach_t[-1]:.3f}s")
    out["teach_seconds"] = teach_t
    out["teach_s_per_fact_mean"] = float(np.mean(teach_t)) if teach_t else None
    out["weights_after_teach_sha256"] = _h(b.cp_connections.data)
    w_teach = np.array(to_host(b.cp_connections.data), copy=True)

    def ask(pairs, label):
        rows, r0, t0 = [], n_reads[0], time.time()
        for a, v in pairs:
            t = time.time()
            got = c.query_patient(a, v)
            rows.append({"agent": a, "action": v, "got": got, "s": time.time() - t})
        return rows, n_reads[0] - r0, time.time() - t0

    qpairs = [(f["agent"], f["action"]) for f in sample[:n_query]]
    rows, reads, secs = ask(qpairs, "intact")
    out["intact"] = {"answers": [r["got"] for r in rows], "reads": reads, "seconds": secs,
                     "s_per_read": secs / reads if reads else None}
    log(f"  [{'on' if sparse_step else 'off'} N={N}] {len(rows)} queries, {reads} reads, {secs:.1f}s")
    probes = []
    mp = _moat_pair(sample[:n_teach], seed)
    if mp is not None:
        probes.append(mp)
    if n_teach >= 2 and (sample[0]["agent"], sample[1]["action"]) not in {(f["agent"], f["action"])
                                                                          for f in sample[:n_teach]}:
        probes.append((sample[0]["agent"], sample[1]["action"]))
    prow, preads, psecs = ask(probes, "probes")
    out["probes"] = {"pairs": probes, "answers": [r["got"] for r in prow], "reads": preads, "seconds": psecs}
    out["weights_after_queries_sha256"] = _h(b.cp_connections.data)
    w_query = np.array(to_host(b.cp_connections.data), copy=True)
    w_abl = None
    if ablation:
        b.cp_connections.data[:] = 0
        arow, areads, asecs = ask(qpairs, "ablation")
        out["ablation"] = {"answers": [r["got"] for r in arow], "reads": areads, "seconds": asecs,
                           "s_per_read": asecs / areads if areads else None}
        out["weights_after_ablation_sha256"] = _h(b.cp_connections.data)
        w_abl = np.array(to_host(b.cp_connections.data), copy=True)
        log(f"  [{'on' if sparse_step else 'off'} N={N}] ablation {len(arow)} queries, {areads} reads, {asecs:.1f}s")
    out["final_state_sha256"] = {a: (_h(getattr(b, a)) if getattr(b, a, None) is not None else None)
                                 for a in _STATE_ATTRS}
    total_steps = (5 * c.teach_steps * len(teach_t)
                   + c.retr_steps * (out["intact"]["reads"] + out["probes"]["reads"]
                                     + (out["ablation"]["reads"] if ablation else 0)))
    total_secs = sum(teach_t) + out["intact"]["seconds"] + out["probes"]["seconds"] + (
        out["ablation"]["seconds"] if ablation else 0.0)
    out["total_sim_steps"] = int(total_steps)
    out["total_seconds"] = float(total_secs)
    out["ms_per_step"] = 1000.0 * total_secs / total_steps if total_steps else None
    out["ms_per_step_teach"] = (1000.0 * sum(teach_t) / (5 * c.teach_steps * len(teach_t))) if teach_t else None
    out["ms_per_step_read"] = (1000.0 * out["intact"]["seconds"] / (c.retr_steps * out["intact"]["reads"])
                               if out["intact"]["reads"] else None)
    try:
        b.clear_simulation_state_and_gpu_memory()
    except Exception:
        pass
    del c, b
    gc.collect()
    return out, (w_teach, w_query, w_abl)


def _first_diff(a, b):
    """n_diff/idx/max_abs over a mismatched weight-array pair -- the diagnostic a cupy run (where bit-identity is
    NOT expected; see the AMENDMENT 3 item-5 pass criterion in
    research/findings/2026-09-24-slotbinder-production-composer-gate-PREREG.md) is read against, without hand
    re-deriving it from the raw arrays. Applied uniformly to every weight comparison (teach/queries/ablation),
    not just teach, so the same criterion is machine-checkable regardless of which snapshot mismatches."""
    d = np.flatnonzero(a != b)
    return {"n_diff": int(d.size), "idx": int(d[0]) if d.size else None,
            "max_abs": float(np.max(np.abs(a - b))) if d.size else 0.0}


def compare(off, on, w_off, w_on):
    eq = {
        "thresholds": off["thresholds_sha256"] == on["thresholds_sha256"],
        "weights_after_teach": bool(np.array_equal(w_off[0].view(np.uint8), w_on[0].view(np.uint8))),
        "intact_answers": off["intact"]["answers"] == on["intact"]["answers"],
        "intact_reads": off["intact"]["reads"] == on["intact"]["reads"],
        "probe_answers": off["probes"]["answers"] == on["probes"]["answers"],
        "weights_after_queries": bool(np.array_equal(w_off[1].view(np.uint8), w_on[1].view(np.uint8))),
        "final_state": off["final_state_sha256"] == on["final_state_sha256"],
    }
    if "ablation" in off:
        eq["ablation_answers"] = off["ablation"]["answers"] == on["ablation"]["answers"]
        eq["weights_after_ablation"] = bool(np.array_equal(w_off[2].view(np.uint8), w_on[2].view(np.uint8)))
    if not eq["weights_after_teach"]:
        eq["weights_after_teach_first_diff"] = _first_diff(w_off[0], w_on[0])
    if not eq["weights_after_queries"]:
        eq["weights_after_queries_first_diff"] = _first_diff(w_off[1], w_on[1])
    if "weights_after_ablation" in eq and not eq["weights_after_ablation"]:
        eq["weights_after_ablation_first_diff"] = _first_diff(w_off[2], w_on[2])
    eq["all"] = all(v for k, v in eq.items() if isinstance(v, bool))
    return eq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-facts", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--fanout", type=int, default=32)
    ap.add_argument("--teach-facts", type=int, default=None, help="teach only the first K facts (timing points)")
    ap.add_argument("--queries", type=int, default=None, help="ask only the first K facts (timing points)")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--paths", default="off,on", help="comma list of off,on")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    backend = os.environ.get("SIM_BACKEND", "")
    _, facts_full, _ = _load_live_bundle()
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    result = {"seed": args.seed, "fanout": args.fanout, "sim_backend": backend, "per_n": {}}

    def log(msg):
        print(msg, flush=True)

    for N in args.n_facts:
        sample, idx = _sample_facts(facts_full, args.seed, N)
        n_teach = min(N, args.teach_facts or N)
        n_query = min(n_teach, args.queries or n_teach)
        entry = {"n_facts": N, "sampled_corpus_indices": idx, "n_teach": n_teach, "n_query": n_query,
                 "ablation": not args.no_ablation, "paths": {}}
        ws = {}
        for p in paths:
            log(f"[seed {args.seed}] N={N} path={p} teach={n_teach} query={n_query} ...")
            res, w = run_path(sample, args.seed, args.fanout, p == "on", n_teach, n_query,
                              not args.no_ablation, log)
            entry["paths"][p] = res
            ws[p] = w
            log(f"[seed {args.seed}] N={N} path={p}: teach {res['teach_s_per_fact_mean']:.3f}s/fact, "
                f"{res['ms_per_step']:.2f} ms/step overall, dispatch={res['sparse_activity_step_dispatches']}")
        if "off" in ws and "on" in ws:
            entry["equal"] = compare(entry["paths"]["off"], entry["paths"]["on"], ws["off"], ws["on"])
            off, on = entry["paths"]["off"], entry["paths"]["on"]
            entry["speedup_teach"] = off["teach_s_per_fact_mean"] / on["teach_s_per_fact_mean"]
            entry["speedup_overall_per_step"] = off["ms_per_step"] / on["ms_per_step"]
            log(f"[seed {args.seed}] N={N} EQUAL={entry['equal']['all']} {json.dumps(entry['equal'])} "
                f"speedup teach x{entry['speedup_teach']:.2f} overall x{entry['speedup_overall_per_step']:.2f}")
        ws.clear()
        gc.collect()
        result["per_n"][str(N)] = entry
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
    eqs = [e["equal"]["all"] for e in result["per_n"].values() if "equal" in e]
    result["verdict"] = ("EQUIVALENT (bit-identical)" if eqs and all(eqs) else
                         ("NOT EQUIVALENT" if eqs else "TIMING ONLY (one path)"))
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    log(f"-> {args.out}  verdict: {result['verdict']}")
    return 0 if (not eqs or all(eqs)) else 1


if __name__ == "__main__":
    sys.exit(main())
