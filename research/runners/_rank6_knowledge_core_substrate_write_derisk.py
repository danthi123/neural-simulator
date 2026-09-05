"""RANK-6 de-risk (scaffold_retirement_backlog.md #6, 2026-09-05): characterize + probe a more brain-faithful
WRITE for the ~78k-fact knowledge core, vs the closed-form host persistence the shipped LTM currently uses.

PREMISE CHECK (read the substrate before theorizing; verified against the current code, not assumed):

  1. THE 78k-FACT CORE IS REAL AND IDENTIFIED. `sim-data/knowledge_bundles/wikidata_100k/curation_report.json`
     records `n_facts: 78857` (the actual number of qualifying (subject,relation) candidates the curator found
     under `--top-entities 25000 --top-relations 60`, despite the directory's "100k" name) -- this is the
     backlog's "~78k-fact knowledge core". It is a FOLLOW-ON scale bundle, not the currently-SHIPPED default
     (`wikidata_core_15k`, 15,000 facts, is what `webapp/server.py` loads by default per the 2026-08-26
     knowledge-core-ship finding) -- but both are built by the identical mechanism this file probes, so the
     characterization below applies to either, and to the still-larger 500k/1M bundles already on disk.

  2. THE PER-FACT BIND IS ALREADY GENUINELY NEURAL, NOT CLOSED-FORM. `research/runners/_knowledge_core_curate.py`
     built wikidata_100k with `fast=False` (confirmed: `curation_report.json`'s `"fast": false`, `build_seconds:
     1913.5` for 78,857 facts = ~24ms/fact, consistent with the module's own "~52-63ms/fact" resonate-bind
     estimate) -- so `RFPhasorComposer.store()` -> `_encode()` -> `_bind()`/`_bundle()` -> `_resonate()` steps a
     REAL `SimulationBridge` of RESONATE_AND_FIRE neurons for every fact (`_build_rf_bridge` + `rf_resonate_steps`,
     not `tiered_fact_store.encode_fast()`'s closed-form `np.exp(2j*pi*(...))` shortcut, which IS used for the
     500k/1M "_fast" bundles -- those two, unlike wikidata_100k, are genuinely closed-form end to end and are
     correctly out of scope for "ship_ready").  The composer's own SPIKING COMPUTATION (bind/unbind/bundle/
     cleanup) was already established fully-on-substrate by 2026-07-20 (composer-factstore-host-persistence-is-
     the-VSA-idealization-scoping.md) for the SMALL conversational buffer; this file re-confirms the SAME holds
     at real knowledge-core scale (a different question -- the 2026-07-20 finding never built past buffer-sized
     stores) and moves on to what that finding named as the ACTUAL residual.

  3. THE ACTUAL HOST RESIDUAL, per 2026-07-20, is DATA PERSISTENCE, not the bind computation: the composite that
     `_encode()` produces is committed to `self.kb` -- a bare host Python list holding a numpy array (the
     "numpy-kb fast path"), not a synaptic weight. `RFPhasorComposer` ALREADY has an additive, validated,
     default-off alternative for this (`enable_substrate_store=True` -> `_store_substrate`: the composite lives
     in a PERSISTENT (1+D)-neuron RF bridge's complex synaptic weights, "the Crawford-Eliasmith weight-store" --
     Phase-2 GO at small N, 2026-06-05-phase2-substrate-store-derisk-GO.md). But `sharded_phasor_store.py`
     (the class the LTM/knowledge-core actually uses) says outright, in its own `save()` docstring: "Numpy fast
     path only (enable_substrate_store=False, the LTM default)" -- i.e. the ALREADY-VALIDATED synaptic write is
     explicitly EXCLUDED from ever reaching the knowledge core. This file asks, empirically, WHY that exclusion
     is there and WHAT it would cost to lift it -- rather than assuming the small-N Phase-2 validation transfers.

  4. THE CURATION SELECTION is a THIRD, separate, and deeper residual this file does NOT attempt to close: WHICH
     ~78,857 of wikidata5m's 5M triples make the cut is a closed-form host frequency/degree ranking
     (`_knowledge_core_curate.curate()`), run ONCE, outside any conversational/experiential context -- no
     salience, curiosity, or reward signal the brain itself produces has any say in what gets encoded. Closing
     THIS would need an autonomous reading/attention loop over the corpus, a materially larger mechanism than a
     store-persistence swap; it is named here as the honest next-deeper rung, not attempted.

THIS FILE (additive, default-off by construction -- it is a NEW standalone research runner that touches NO
production file and no `sim/` code; production's `enable_substrate_store=False` default is never adjusted):

  (a) PARITY: build the REAL curated wikidata_100k facts (a genuine sample, not synthetic) into a
      `ShardedPhasorStore` twice -- once exactly as production does (`enable_substrate_store=False`) and once
      with the already-validated candidate (`=True`) -- and check every stored fact's recall AND every held-out
      (unseen-agent) moat probe answer AGAINST EACH OTHER, 6 seeds (42/43/44/100/101/102).
  (b) COST: measure (not estimate -- `tools.lab.project_cost`'s own standing lesson) the marginal memory + time
      the synaptic-substrate write actually costs per fact, at three real scales, subprocess-isolated so each
      reading is a clean peak-RSS delta, then PROJECT to 78,857 (and to the 500k/1M bundles already on disk).
  (c) STRUCTURAL PROBE: attempt a real `.save()`/`.load()` round-trip under `enable_substrate_store=True` --
      `ShardedPhasorStore.save()` was written assuming `handle` is a numpy array (`comps.append(np.asarray(handle))`)
      and never exercised against a bridge handle; confirm empirically what actually happens instead of assuming.
  (d) PROVENANCE CHARACTERIZATION (the "at minimum, honest" fallback the task names): confirm
      `TieredFactStore.query_patient_source()` (shipped 2026-08-27 for `BRAIN_GNW_ORGANB_LTM_EXEMPT`) ALREADY
      exposes a genuine, zero-new-mechanism tier-of-origin signal (`"buffer"` == conversationally taught this
      session, vs `"ltm"` == bulk-curated background knowledge) for every LTM-era recall, and that this signal is
      currently UNCONSUMED by the shipped provenance-honesty framing (`known_fact_record` labels every hit
      `PROVENANCE_PERCEIVED` regardless of tier -- board #129/#140's PERCEIVED/GENERATED axis discriminates
      single-fact-recall vs multi-hop-inference, a different question from write-origin). This file does NOT
      extend that monitor's judged vocabulary (a 3-way discrimination would need its OWN accuracy validation,
      exactly like the existing 2-way one earned) -- it characterizes that the DATA dependency for a future
      rung already exists in production, unused, lowering that rung from "build a mechanism" to "wire a signal".

Run (headless, CPU/numpy -- matches how the bundles themselves were built):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners._rank6_knowledge_core_substrate_write_derisk \
      --bundle /home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k \
      --seeds 42 43 44 100 101 102 \
      --out research/findings/raw/_rank6_knowledge_core_substrate_write_derisk.json
  # --smoke : tiny N (10 facts, 2 seeds) for a fast end-to-end sanity pass
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")   # matches how the bundles themselves were built (headless CPU)

from tools.lab import Verdict, project_cost, assert_backend  # noqa: E402

DEFAULT_BUNDLE = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_100k"
SEEDS = [42, 43, 44, 100, 101, 102]
D = 128


# ------------------------------------------------------------------------------------------------------------
# real-fact loading (NOT synthetic -- the whole point is to probe the mechanism against what actually shipped)
# ------------------------------------------------------------------------------------------------------------
def load_real_facts(bundle_dir):
    path = os.path.join(bundle_dir, "facts.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as fh:
        recs = json.load(fh)
    return [r["fact"] for r in recs]


def vocab_of(facts):
    vs = set()
    for f in facts:
        for role in ("agent", "action", "patient"):
            w = f.get(role)
            if isinstance(w, str):
                vs.add(w)
    return sorted(vs)


def pick_moat_probes(all_facts, stored_facts, n_probe):
    """Facts whose AGENT never appears among `stored_facts` -- both stores must abstain on these (routing sends
    them to some shard, which then genuinely holds no matching fact; NOT the same as an out-of-vocab word)."""
    stored_agents = {f["agent"] for f in stored_facts}
    out = []
    for f in all_facts:
        if f["agent"] not in stored_agents and f["agent"] not in {p["agent"] for p in out}:
            out.append(f)
        if len(out) >= n_probe:
            break
    return out


# ------------------------------------------------------------------------------------------------------------
# store construction (uses ONLY the already-existing, already-validated `enable_substrate_store` kwarg --
# no production file is touched; both arms are exercised from this standalone script alone)
# ------------------------------------------------------------------------------------------------------------
def build_store(facts, vocab, seed, substrate):
    from research.runners.sharded_phasor_store import ShardedPhasorStore
    from research.runners.tiered_fact_store import auto_n_shards
    n_shards = auto_n_shards(len(facts))
    store = ShardedPhasorStore(n_shards=n_shards, seed=seed, D=D, vocab=list(vocab), share_codebook=True,
                               enable_substrate_store=substrate)
    t0 = time.time()
    for f in facts:
        store.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
    return store, time.time() - t0


def query_all(store, facts):
    """{(agent,action): patient_or_None} for every fact's cue."""
    return {(f["agent"], f["action"]): store.query_patient(f["agent"], f["action"]) for f in facts}


def _substrate_flag_reached():
    """LEVER check (tools.lab convention): does `enable_substrate_store` actually change what `.kb` holds, or
    could the whole A/B above be silently comparing two identical configurations? True iff the numpy-kb arm
    stores a plain array and the substrate arm stores something else (a bridge handle) for the SAME fact."""
    import numpy as np
    from research.runners.rf_phasor_composer import RFPhasorComposer
    vocab = ["probe_agent", "probe_action", "probe_patient"]
    c_off = RFPhasorComposer(seed=1, D=16, vocab=vocab, enable_substrate_store=False)
    c_on = RFPhasorComposer(seed=1, D=16, vocab=vocab, enable_substrate_store=True)
    c_off.store("probe_agent", "probe_action", "probe_patient", polarity="AFFIRM")
    c_on.store("probe_agent", "probe_action", "probe_patient", polarity="AFFIRM")
    handle_off = c_off.kb[0][1]
    handle_on = c_on.kb[0][1]
    return isinstance(handle_off, np.ndarray) and not isinstance(handle_on, np.ndarray)


# ------------------------------------------------------------------------------------------------------------
# (a) PARITY -- 6 seeds
# ------------------------------------------------------------------------------------------------------------
def run_parity(facts_all, n_stored, n_moat, seeds, verbose=True):
    stored = facts_all[:n_stored]
    moat = pick_moat_probes(facts_all, stored, n_moat)
    vocab = vocab_of(stored + moat)
    per_seed = []
    for seed in seeds:
        base, base_s = build_store(stored, vocab, seed, substrate=False)
        cand, cand_s = build_store(stored, vocab, seed, substrate=True)

        base_ans = query_all(base, stored)
        cand_ans = query_all(cand, stored)
        n_base_recalled = sum(1 for v in base_ans.values() if v is not None)
        n_agree = sum(1 for k in base_ans if base_ans[k] == cand_ans[k])

        base_moat = query_all(base, moat)
        cand_moat = query_all(cand, moat)
        base_moat_abstained = sum(1 for v in base_moat.values() if v is None)
        moat_agree = sum(1 for k in base_moat if base_moat[k] == cand_moat[k])

        row = dict(seed=seed, n_stored=len(stored), n_moat=len(moat),
                   base_build_s=round(base_s, 3), cand_build_s=round(cand_s, 3),
                   n_base_recalled=n_base_recalled, n_agree=n_agree,
                   base_moat_abstained=base_moat_abstained, moat_agree=moat_agree)
        if verbose:
            print("  seed=%-4d recalled=%d/%d agree=%d/%d moat_abstain=%d/%d moat_agree=%d/%d "
                  "build(base=%.2fs cand=%.2fs)"
                  % (seed, n_base_recalled, len(stored), n_agree, len(stored),
                     base_moat_abstained, len(moat), moat_agree, len(moat), base_s, cand_s))
        per_seed.append(row)
    return dict(n_stored=len(stored), n_moat=len(moat), per_seed=per_seed)


# ------------------------------------------------------------------------------------------------------------
# (b) COST. EARNED MID-RUN (this file's own smoke test): a naive "build N facts in a fresh subprocess, read
# ru_maxrss, divide by N" is swamped by ~300MB of FIXED cost (python/numpy/SimulationBridge import + the
# codebook allocation, which scales with VOCAB not fact count) at any N small enough to stay fast -- the smoke
# run's own numbers (N=5 -> 310216 KB, N=20 -> 310332 KB) divided out to a nonsense "1166 GB at 78,857 facts"
# projection. Exactly the proxy-dominates shape `tools.lab` warns about, caught by looking at the actual numbers
# instead of trusting the formula. FIX: pay the fixed cost ONCE per subprocess (imports + codebook), take a
# BASELINE ru_maxrss reading there, then grow ONE store continuously to the largest N, snapshotting the
# high-water mark at each checkpoint -- ru_maxrss only grows and nothing is ever freed here, so each checkpoint
# delta is a clean marginal reading uncontaminated by the fixed cost, without needing N large enough to swamp
# ~300MB on its own. Substrate=True and substrate=False still run in SEPARATE processes (the high-water mark
# would otherwise carry over from whichever variant ran first in the same process).
# ------------------------------------------------------------------------------------------------------------
def _cost_worker(facts_path, vocab, checkpoints, substrate, seed, q):
    import resource
    os.environ.setdefault("SIM_BACKEND", "numpy")
    from research.runners.sharded_phasor_store import ShardedPhasorStore
    from research.runners.tiered_fact_store import auto_n_shards

    with open(facts_path) as fh:
        facts = [r["fact"] for r in json.load(fh)][: max(checkpoints)]

    n_shards = auto_n_shards(len(facts))
    store = ShardedPhasorStore(n_shards=n_shards, seed=seed, D=D, vocab=list(vocab), share_codebook=True,
                               enable_substrate_store=substrate)
    rss0_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # AFTER imports + codebook, BEFORE any fact
    rows = []
    done = 0
    t_start = time.time()
    for cp in sorted(checkpoints):
        while done < cp:
            f = facts[done]
            store.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
            done += 1
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rows.append(dict(n=cp, substrate=substrate, elapsed_s=round(time.time() - t_start, 3),
                          rss0_kb=int(rss0_kb), rss_kb=int(rss_kb), marginal_kb=int(rss_kb - rss0_kb)))
    q.put(rows)


def run_cost(bundle_dir, checkpoints, seed=42, verbose=True):
    facts_path = os.path.join(bundle_dir, "facts.json")
    all_facts = load_real_facts(bundle_dir)
    vocab = vocab_of(all_facts[: max(checkpoints)])
    all_rows = []
    for substrate in (False, True):
        q = mp.Queue()
        p = mp.Process(target=_cost_worker, args=(facts_path, vocab, checkpoints, substrate, seed, q))
        p.start()
        rows = q.get()
        p.join()
        all_rows.extend(rows)
        for row in rows:
            if verbose:
                print("  N=%-6d substrate=%-5s elapsed=%.2fs  rss0=%.1fMB rss=%.1fMB  marginal=%.1fMB"
                      % (row["n"], substrate, row["elapsed_s"], row["rss0_kb"] / 1024.0,
                         row["rss_kb"] / 1024.0, row["marginal_kb"] / 1024.0))
    return all_rows


def project_from_cost(rows, target_n, label):
    """Per-fact marginal cost from the LARGEST-vs-SMALLEST checkpoint of each variant (both already exclude the
    fixed import+codebook cost via the rss0 baseline in `_cost_worker`) -- an honest incremental slope, not a
    total-divided-by-N ratio contaminated by fixed overhead (see the module note above this function)."""
    def slope(sub_rows):
        sub_rows = sorted(sub_rows, key=lambda r: r["n"])
        if len(sub_rows) < 2:
            return None
        lo, hi = sub_rows[0], sub_rows[-1]
        dn = hi["n"] - lo["n"]
        if dn <= 0:
            return None
        return dict(per_fact_kb=(hi["marginal_kb"] - lo["marginal_kb"]) / dn,
                    per_fact_s=(hi["elapsed_s"] - lo["elapsed_s"]) / dn,
                    n_lo=lo["n"], n_hi=hi["n"])

    sub_rows = [r for r in rows if r["substrate"]]
    base_rows = [r for r in rows if not r["substrate"]]
    sub_slope = slope(sub_rows)
    base_slope = slope(base_rows)
    if sub_slope is None:
        print("  PROJECT %s: fewer than 2 substrate checkpoints -- cannot slope, UNDEFINED" % label)
        return None
    proj_rss_gb = sub_slope["per_fact_kb"] * target_n / (1024.0 * 1024.0)
    proj_s = sub_slope["per_fact_s"] * target_n
    print("  PROJECT %-28s substrate slope (N=%d->%d): %.4f KB/fact, %.5f s/fact"
          % (label, sub_slope["n_lo"], sub_slope["n_hi"], sub_slope["per_fact_kb"], sub_slope["per_fact_s"]))
    project_cost("%s substrate-store build time" % label, 1, 1, proj_s, warn_hours=1.0)
    print("  => PROJECTED marginal peak RSS at N=%d: %.2f GB (substrate-store path, ABOVE the fixed "
          "import+codebook floor)" % (target_n, proj_rss_gb))
    out = dict(label=label, target_n=target_n, substrate_slope=sub_slope, proj_rss_gb=round(proj_rss_gb, 4),
               proj_build_s=round(proj_s, 1))
    if base_slope is not None:
        base_proj_gb = base_slope["per_fact_kb"] * target_n / (1024.0 * 1024.0)
        print("  => for comparison, the CURRENT numpy-kb path's marginal slope (N=%d->%d): %.4f KB/fact "
              "-> %.3f GB at N=%d" % (base_slope["n_lo"], base_slope["n_hi"], base_slope["per_fact_kb"],
                                       base_proj_gb, target_n))
        out["numpy_kb_slope"] = base_slope
        out["numpy_kb_proj_gb"] = round(base_proj_gb, 4)
        if base_proj_gb > 0:
            out["substrate_overhead_multiple"] = round(proj_rss_gb / base_proj_gb, 2)
            print("  => substrate-store costs %.1fx the numpy-kb path's marginal memory per fact"
                  % out["substrate_overhead_multiple"])
    return out


# ------------------------------------------------------------------------------------------------------------
# (c) STRUCTURAL PROBE -- does save()/load() even survive enable_substrate_store=True?
# ------------------------------------------------------------------------------------------------------------
def run_save_reload_probe(facts_all, out_dir, n=8, seed=42, verbose=True):
    from research.runners.sharded_phasor_store import ShardedPhasorStore
    stored = facts_all[:n]
    vocab = vocab_of(stored)
    store, _ = build_store(stored, vocab, seed, substrate=True)
    path = os.path.join(out_dir, "_rank6_save_reload_probe_bundle")
    result = dict(attempted=True, save_ok=None, save_error=None, load_ok=None, load_error=None,
                  reloaded_answers_match=None)
    try:
        store.save(path)
        result["save_ok"] = True
    except Exception as e:
        result["save_ok"] = False
        result["save_error"] = "%s: %s" % (type(e).__name__, e)
        if verbose:
            print("  save() under enable_substrate_store=True FAILED: %s" % result["save_error"])
        return result
    if verbose:
        print("  save() under enable_substrate_store=True: no exception raised")
    try:
        reloaded = ShardedPhasorStore.load(path)
        result["load_ok"] = True
        pre = query_all(store, stored)
        post = query_all(reloaded, stored)
        result["reloaded_answers_match"] = (pre == post)
        if verbose:
            print("  load() succeeded; reloaded answers %s the pre-save answers"
                  % ("MATCH" if result["reloaded_answers_match"] else "DO NOT MATCH"))
    except Exception as e:
        result["load_ok"] = False
        result["load_error"] = "%s: %s" % (type(e).__name__, e)
        if verbose:
            print("  load() FAILED: %s" % result["load_error"])
    return result


# ------------------------------------------------------------------------------------------------------------
# (d) PROVENANCE CHARACTERIZATION -- confirm the tier signal already exists in production, unused for framing
# ------------------------------------------------------------------------------------------------------------
def run_provenance_characterization(facts_all, n_stored=20, seed=42, verbose=True):
    from research.runners.tiered_fact_store import TieredFactStore, auto_n_shards
    from research.runners.rf_phasor_composer import RFPhasorComposer
    from research.runners.sharded_phasor_store import ShardedPhasorStore

    ltm_facts = facts_all[:n_stored]
    vocab = vocab_of(ltm_facts + [{"agent": "buffer_taught_thing", "action": "is", "patient": "new"}])
    buffer = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    ltm = ShardedPhasorStore(n_shards=auto_n_shards(len(ltm_facts)), seed=seed, D=D, vocab=vocab,
                             share_codebook=True)
    for f in ltm_facts:
        ltm.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
    tiered = TieredFactStore(buffer, ltm)

    # a fact taught THIS session (buffer) vs a bulk-curated background fact (ltm) -- same API, different origin
    buffer.store("buffer_taught_thing", "is", "new", polarity="AFFIRM")
    probe_ltm_fact = ltm_facts[0]

    ans_buffer, tier_buffer = tiered.query_patient_source("buffer_taught_thing", "is")
    ans_ltm, tier_ltm = tiered.query_patient_source(probe_ltm_fact["agent"], probe_ltm_fact["action"])

    result = dict(
        tier_signal_exists=True,
        buffer_sourced_example=dict(cue=["buffer_taught_thing", "is"], answer=ans_buffer, tier=tier_buffer),
        ltm_sourced_example=dict(cue=[probe_ltm_fact["agent"], probe_ltm_fact["action"]], answer=ans_ltm,
                                  tier=tier_ltm),
        tier_correctly_distinguishes_origin=(tier_buffer == "buffer" and tier_ltm == "ltm"),
    )
    if verbose:
        print("  query_patient_source(buffer-taught fact) -> tier=%r (expect 'buffer')" % tier_buffer)
        print("  query_patient_source(bulk-curated LTM fact) -> tier=%r (expect 'ltm')" % tier_ltm)
        print("  => the tier-of-origin signal %s distinguish conversational-write from bulk-curated-write"
              % ("DOES ALREADY" if result["tier_correctly_distinguishes_origin"] else "DOES NOT"))
    return result


# ------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=DEFAULT_BUNDLE)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--n-stored", type=int, default=240)
    ap.add_argument("--n-moat", type=int, default=40)
    ap.add_argument("--cost-points", type=int, nargs="+", default=[500, 2000, 8000])
    ap.add_argument("--target-n", type=int, default=78857)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-cost", action="store_true", help="skip the (slower) memory/time scaling probe")
    ap.add_argument("--out", default="research/findings/raw/_rank6_knowledge_core_substrate_write_derisk.json")
    a = ap.parse_args()

    if a.smoke:
        a.seeds = a.seeds[:2]
        a.n_stored, a.n_moat = 10, 5
        a.cost_points = [5, 15, 30]
        a.target_n = 78857

    assert_backend("numpy", note="(matches how the bundles were originally built)")

    if not os.path.isdir(a.bundle):
        print("SKIPPED: bundle not found at %r -- this de-risk needs the local wikidata_100k bundle (not "
              "checked into git; box-local data lake). Nothing to characterize on this machine." % a.bundle)
        out = dict(verdict="UNDEFINED", reason="bundle_not_found", bundle=a.bundle)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)
        return 0

    facts_all = load_real_facts(a.bundle)
    if not facts_all:
        print("SKIPPED: %r/facts.json missing or empty" % a.bundle)
        return 0
    print("[rank6] loaded %d REAL curated facts from %s" % (len(facts_all), a.bundle))

    print("\n=== (a) PARITY: numpy-kb (current) vs synaptic-substrate-store (candidate), %d seeds ==="
          % len(a.seeds))
    parity = run_parity(facts_all, a.n_stored, a.n_moat, a.seeds)

    cost = None
    projection = None
    if not a.skip_cost:
        print("\n=== (b) COST: subprocess-isolated peak-RSS + build time, substrate vs numpy-kb ===")
        cost = run_cost(a.bundle, a.cost_points)
        projection = project_from_cost(cost, a.target_n, "wikidata_100k(%d facts)" % a.target_n)

    print("\n=== (c) STRUCTURAL PROBE: does save()/load() survive enable_substrate_store=True? ===")
    save_probe = run_save_reload_probe(facts_all, os.path.dirname(a.out) or ".", n=min(8, a.n_stored))

    print("\n=== (d) PROVENANCE CHARACTERIZATION: does a tier-of-origin signal already exist, unused? ===")
    prov = run_provenance_characterization(facts_all)

    # ---- verdict over the PARITY claim (the only claim this file makes a GO/NO-GO call on) ----
    # PRECONDITIONS (is the comparison even meaningful?) go through floor/knob; the ACTUAL claim (does the
    # candidate agree with the baseline, 6/6 seeds) is computed directly and passed as `go=` -- per Verdict's
    # own design, folding the claim itself into require() would report a genuine disagreement as UNDEFINED
    # rather than NO-GO (require/floor/knob are for validating the INSTRUMENT, not the hypothesis under test).
    v = Verdict("rank6 knowledge-core substrate-store write: recall+moat parity vs the shipped numpy-kb path")
    all_agree = True
    for row in parity["per_seed"]:
        v.floor("seed %d: baseline actually recalls most stored facts" % row["seed"],
                row["n_base_recalled"] / max(1, row["n_stored"]), 0.9)
        stored_ok = row["n_agree"] == row["n_stored"]
        moat_ok = row["moat_agree"] == row["n_moat"]
        print("  CLAIM seed=%d stored_agree=%s(%d/%d) moat_agree=%s(%d/%d)"
              % (row["seed"], stored_ok, row["n_agree"], row["n_stored"], moat_ok,
                 row["moat_agree"], row["n_moat"]))
        all_agree = all_agree and stored_ok and moat_ok
    v.knob("enable_substrate_store reaches the base shard's composer (a real lever, not a no-op)",
           requested=1.0, applied=1.0 if _substrate_flag_reached() else 0.0)
    v.disabled("curation selection (which facts are worth learning)",
               "closed-form host frequency ranking, unchanged by this probe -- see module docstring point 4")
    decided = v.decide(go=all_agree)

    out = dict(
        verdict=decided,
        bundle=a.bundle, n_facts_in_bundle=len(facts_all), target_n=a.target_n,
        parity=parity, cost=cost, projection=projection, save_reload_probe=save_probe,
        provenance_characterization=prov,
    )
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[rank6] wrote %s  verdict=%s" % (a.out, decided["status"]))
    return 0 if decided["status"] != "NO-GO" else 1


if __name__ == "__main__":
    sys.exit(main())
