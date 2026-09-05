"""L2 de-risk: does SPARSIFYING SlotBinderComposer's slot->filler wiring make it fit a single consumer RTX 3090
AND still compose, at the SAME live scale that failed dense (2026-09-04)?

Board/task: spawned by `research/findings/2026-09-04-slotbinder-live-scale-derisk-NOGO-dense-pathway-blowup.md`
(the L1 measurement -- dense density=1.0 wiring needs ~316GB host RAM to even BUILD at 404 facts/788 vocab, and
~36-463GB GPU-resident, vs a single 3090's 24GB). That finding's own recommendation #1: replace the dense
all-to-all slot->filler pathway with "each slot pool connecting to a small, fixed-size subset of candidate
filler pools rather than all KF of them" -- named but NOT built there. This runner builds + measures that.

TWO questions, both at the REAL production topology (K=2020 slot pools, KF=1195 filler pools -- read from the
live deployed bundle `bridges/developed/scale787/day_33/{brain.json,facts.json}`, 404 facts / 788-word vocab,
identical to the L1 finding's own derivation):

  (a) FIT -- `--mode formula`: the sparsified edge-count formula (`slot_filler_nnz_formula`, shared with the
      wiring code itself so the two cannot drift apart), cross-checked against REAL builds (not just algebra) at
      all 4 sweep fanouts, then run through the project's own `tools/gates/consumer_hardware_reference.py`
      formula AND the exact 40-bytes/synapse figure the L1 finding measured by direct introspection (fanout does
      not change per-synapse byte cost, only synapse COUNT, so that ratio transfers unchanged).

  (b) COMPOSITION -- `--mode compose` (one seed x one fanout x an explicit sample of REAL facts per invocation,
      matching the L1 finding's own "subprocess per point so peak RSS/timing reflects only that point"
      discipline -- the caller loops seeds/fanouts, each a fresh process): builds the SlotBinderComposer at the
      REAL K=2020/KF=1195 topology with `fanout` sparsification and `prewire_facts` (a batch-consolidation
      pre-registration of the ACTUAL facts this run will store -- see slotbinder_composer.py's docstring: this is
      the legitimate scenario for migrating an already-known, already-collected corpus like the day_33 bundle,
      not a per-query lookahead), stores a bounded SAMPLE of `n_facts` real facts (exhaustively teaching all 404
      is the O(nnz)-per-step latency problem the L1 finding's own S5 flagged as a SEPARATE, compounding cost --
      out of scope for this rung, which is about wiring density, not step latency), then checks store/recall,
      the moat (a cue that matches no stored fact -> None), and a mismatched-role-cue anti-cheat (an
      agent/action pairing that matches no stored fact must NOT return some other fact's patient).

  `--mode blind` runs the SAME composition test WITHOUT `prewire_facts` (no foreknowledge of which filler each
  slot will need -- a purely random, fixed candidate set per slot) to honestly quantify the coverage cost of
  sparsification when the corpus is NOT known in advance (the online/incremental deployment case) -- reported as
  a secondary, not the primary, verdict; the roadmap's own L2 gate anticipated this ("a scale lever, may bound
  max_facts").

CPU/numpy only (matches L1's own backend choice + this task's cost-routing instruction); no `sim/` file touched.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._keystone2_spiking_slot_binder_derisk import build_binder_bridge, slot_filler_nnz_formula  # noqa: E402
from research.runners.slotbinder_composer import SlotBinderComposer, _ROLES  # noqa: E402

_ROLE_NAMES = ("agent", "action", "patient", "polarity")


def _find_live_bundle_dir():
    """`bridges/developed/` is gitignored deployment data (machine-local, not part of the repo checkout) -- an
    isolated `git worktree` checkout of THIS runner will not have it under its own root. Checked, in order:
    an explicit override (`SIM_LIVE_BUNDLE_DIR`, for a differently-located machine), this repo root's own
    `bridges/...` (in case it IS present -- the primary checkout), then the primary `sim` checkout's path (where
    the L1 NO-GO finding this runner follows on from actually read it) as a last-resort fallback so a worktree
    agent can still read the SAME real deployed bundle that finding measured, without copying gitignored data."""
    candidates = [
        os.environ.get("SIM_LIVE_BUNDLE_DIR"),
        os.path.join(_REPO, "bridges", "developed", "scale787", "day_33"),
        "/home/dant123/Projects/sim/bridges/developed/scale787/day_33",
    ]
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, "brain.json")):
            return c
    raise FileNotFoundError(
        "live bundle bridges/developed/scale787/day_33/{brain.json,facts.json} not found in any of: "
        + ", ".join(c for c in candidates if c) + " -- set SIM_LIVE_BUNDLE_DIR to override"
    )


def _load_live_bundle():
    bundle = _find_live_bundle_dir()
    brain = json.load(open(os.path.join(bundle, "brain.json")))
    facts = json.load(open(os.path.join(bundle, "facts.json")))["facts"]
    vocab = list(brain["vocab"])
    assert brain["n_facts"] == len(facts), "brain.json n_facts disagrees with facts.json length"
    return vocab, facts, brain


def _sample_facts(facts, seed, n_facts):
    """A seed-dependent, without-replacement sample of REAL facts from the live corpus, in ASCENDING original
    index order (deterministic + reproducible; index order rather than random order avoids conflating "seed
    changes which facts" with "seed changes store() order", which are separate variables)."""
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(facts), size=min(n_facts, len(facts)), replace=False).tolist())
    return [facts[i] for i in idx], idx


def _timed_build(build_fn):
    t0 = time.time()
    b = build_fn()
    return b, time.time() - t0


# ---------------------------------------------------------------------------------------------------- formula ----
def mode_formula(args):
    vocab, facts, brain = _load_live_bundle()
    K, KF = _ROLES * len(facts), len(vocab) + 3 + len(facts)   # == 2020, 1195 -- must match the L1 finding exactly
    assert (K, KF) == (2020, 1195), f"live bundle changed shape: K={K} KF={KF} (L1 finding assumed 2020/1195)"
    n_neurons_full = K * 20 + 24 + KF * 20

    from tools.gates.consumer_hardware_reference import estimate_vram_bytes, classify, GIB

    # (1) small-scale formula cross-check (algebra vs an ACTUAL build) -- cheap, mirrors the L1 finding's own
    # "verified exact match at all 4 tested scales" discipline, now extended to cover fanout too.
    small_checks = []
    for k, kf, fanout in [(6, 19, None), (6, 19, 8), (40, 200, 32)]:
        b = build_binder_bridge(42, K=k, KF=kf, fanout=fanout)
        actual = int(b.cp_connections.nnz)
        formula = slot_filler_nnz_formula(k, kf, fanout=fanout)
        small_checks.append({"K": k, "KF": kf, "fanout": fanout, "actual_nnz": actual, "formula_nnz": formula,
                             "match": actual == formula})
        del b

    # (2) REAL builds at the FULL K=2020/KF=1195 topology for every sweep fanout (BUILD ONLY -- no teaching --
    # so this stays cheap; each a value the L1 dense finding had to EXTRAPOLATE, here measured directly).
    # dense (fanout=None) is NOT rebuilt here -- it is the L1 finding's own already-measured 316GB/~968M-synapse
    # point, cited, not repeated (repeating it risks OOM on this same shared machine, per that finding's §9).
    real_points = []
    for fanout in args.fanouts:
        import resource
        t0 = time.time()
        b = build_binder_bridge(42, K=K, KF=KF, fanout=fanout)
        build_s = time.time() - t0
        nnz = int(b.cp_connections.nnz)
        peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
        vram_ch_gate = estimate_vram_bytes(n_neurons_full, nnz)
        vram_exact_40b = nnz * 40  # the L1 finding's own exact, exhaustively-introspected bytes/synapse (fanout
                                    # does not change per-synapse byte cost, only synapse COUNT)
        real_points.append({
            "fanout": fanout, "nnz": nnz, "formula_nnz": slot_filler_nnz_formula(K, KF, fanout=fanout),
            "nnz_match": nnz == slot_filler_nnz_formula(K, KF, fanout=fanout),
            "build_s": build_s, "this_process_peak_rss_gb": peak_rss_gb,
            "vram_ch_gate_formula_gib": vram_ch_gate / GIB, "vram_ch_gate_classify": classify(vram_ch_gate),
            "vram_exact_40bytes_per_synapse_gib": vram_exact_40b / GIB,
            "vram_exact_40bytes_per_synapse_classify": classify(vram_exact_40b),
        })
        del b   # each point still measured in-process (unlike L1's subprocess-per-point) -- these builds are
                # 2-3 orders of magnitude smaller than L1's dense point, so accumulation risk is low; if
                # `this_process_peak_rss_gb` climbs across points that would show up in this JSON directly.

    dense_reference = {
        "nnz": 968307200, "host_build_peak_rss_gb": 316.0, "vram_ch_gate_gib": 462.85,
        "vram_exact_40bytes_gib": 968307200 * 40 / (1024 ** 3),
        "source": "research/findings/2026-09-04-slotbinder-live-scale-derisk-NOGO-dense-pathway-blowup.md (L1, cited not re-measured)",
    }
    result = {
        "purpose": "L2 FIT check: does sparsifying SlotBinderComposer's slot->filler wiring bring the REAL "
                   "K=2020/KF=1195 production topology under the single-consumer-RTX-3090 (24GB) reference?",
        "K": K, "KF": KF, "n_neurons_full": n_neurons_full, "n_facts": len(facts), "vocab_size": len(vocab),
        "small_scale_formula_cross_check": small_checks,
        "real_scale_builds_per_fanout": real_points,
        "dense_L1_reference_cited_not_remeasured": dense_reference,
    }
    print(json.dumps(result, indent=2))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
    return result


# ---------------------------------------------------------------------------------------------------- compose ----
def _prewire_dicts(sample_facts):
    # facts.json rows are already {"agent","action","patient","polarity"} dicts -- SlotBinderComposer.store()'s
    # own contract. Passed through unchanged as `prewire_facts`.
    return [dict(f) for f in sample_facts]


def _run_compose(seed, fanout, n_facts, blind, vocab, facts, full_methods=False, n_moat=1, n_mismatch=1):
    """`full_methods=False` (the default for the multi-seed sweep) tests ONLY `query_patient` per fact -- the
    core bind->unbind contract the roadmap's own L2 gate names ("recall matrix unregressed"). This is a
    deliberate cost cut, not a coverage gap: EACH read (`_read_slot`) costs `retr_steps=40` full
    `_run_one_simulation_step()` calls, and `query_agent`/`ask_yes_no` each re-run `_match()`'s OWN linear scan
    independently (the composer's public API does not share scan state across calls) -- so testing all 3 methods
    for every fact multiplies read cost ~3x for no new INFORMATION about the sparsified WIRING specifically
    (`query_agent`/`ask_yes_no` exercise the identical `_match`/`_read_slot` primitives on the SAME sparsified
    synapses, already confirmed byte-for-byte equivalent to dense at small scale when `prewire_facts` guarantees
    coverage -- see this arc's small-scale smoke test). `full_methods=True` re-enables them for targeted spot
    checks. `n_moat`/`n_mismatch` cap the anti-cheat probe counts (each is itself several `_read_slot` calls)."""
    sample, idx = _sample_facts(facts, seed, n_facts)
    prewire = None if blind else _prewire_dicts(sample)
    t0 = time.time()
    c = SlotBinderComposer(seed=seed, vocab=vocab, max_facts=len(facts), fanout=fanout, prewire_facts=prewire)
    for f in sample:
        ok = c.store(f["agent"], f["action"], f["patient"], polarity=f.get("polarity"))
        if not ok:
            raise RuntimeError(f"store() rejected a REAL live-bundle fact: {f}")
    build_and_store_s = time.time() - t0
    K, KF = c._b._K_slots, len(c._vocab)

    # coverage: for each stored (slot, role), was the TAUGHT filler actually inside that slot's wired candidate
    # set? (only meaningful/measured when sparse; always 1.0 when dense-equivalent, i.e. fanout is None/>=KF)
    coverage_hits, coverage_total = 0, 0
    if c._b._fanout is not None:
        for i, f in enumerate(sample):
            pol_word = "NEGATE" if f.get("polarity") in ("NEGATE", "neg", False) else "AFFIRM"
            wanted = [c._w2i[f["agent"]], c._w2i[f["action"]], c._w2i[f["patient"]], c._pol[pol_word], c._noattr]
            for role, filler_idx in enumerate(wanted):
                coverage_total += 1
                if filler_idx in c._b._filler_candidates.get(_ROLES * i + role, ()):
                    coverage_hits += 1

    # store/recall correctness over the 4 populated roles (attribute is always NOATTR in this live corpus, per
    # the L1 finding -- checked, not assumed: every real fact.json row lacks an 'attribute' key).
    per_fact = []
    for i, f in enumerate(sample):
        agent, action, patient = f["agent"], f["action"], f["patient"]
        pol = "NEGATE" if f.get("polarity") in ("NEGATE", "neg", False) else "AFFIRM"
        got_patient = c.query_patient(agent, action)
        row = {"fact_idx_in_corpus": idx[i], "agent": agent, "action": action, "patient": patient,
               "polarity": pol, "query_patient_hit": got_patient == patient, "query_patient_got": got_patient}
        if full_methods:
            got_agent = c.query_agent(action, patient)
            got_yesno = c.ask_yes_no(agent, action, patient)
            exp_yesno = "yes" if pol == "AFFIRM" else "no"
            row.update({"query_agent_hit": got_agent == agent, "query_agent_got": got_agent,
                        "ask_yes_no_hit": got_yesno == exp_yesno, "ask_yes_no_got": got_yesno})
        per_fact.append(row)

    # MOAT: (agent, action) pairs that match NONE of the stored sample -> must abstain (None).
    stored_pairs = {(f["agent"], f["action"]) for f in sample}
    moat_checks = []
    all_words = c.words
    rng = np.random.default_rng(seed * 97 + 1)
    tries = 0
    while len(moat_checks) < min(n_moat, n_facts) and tries < 200:
        tries += 1
        a, v = all_words[rng.integers(len(all_words))], all_words[rng.integers(len(all_words))]
        if (a, v) in stored_pairs:
            continue
        moat_checks.append({"agent": a, "action": v, "abstained": c.query_patient(a, v) is None})

    # MISMATCH anti-cheat: cross fact i's agent with fact j's action (i != j) -- if that pairing happens to be a
    # real OTHER stored fact, skip it (not a valid mismatch probe); else it must NOT return fact i's patient.
    mismatch_checks = []
    if n_facts >= 2:
        for i in range(min(n_mismatch, len(sample))):
            j = (i + 1) % len(sample)
            if i == j:
                continue
            a, v = sample[i]["agent"], sample[j]["action"]
            if (a, v) in stored_pairs:
                continue
            got = c.query_patient(a, v)
            mismatch_checks.append({"agent": a, "action_from_other_fact": v,
                                    "did_not_leak_fact_i_patient": got != sample[i]["patient"], "got": got})

    result = {
        "seed": seed, "fanout": fanout, "blind": blind, "n_facts_requested": n_facts,
        "n_facts_sampled": len(sample), "sampled_corpus_indices": idx, "full_methods": full_methods,
        "K": K, "KF": KF, "measured_nnz": int(c._b.cp_connections.nnz),
        "formula_nnz": slot_filler_nnz_formula(K, KF, fanout=fanout),
        "build_and_store_seconds": build_and_store_s,
        "coverage_hits": coverage_hits, "coverage_total": coverage_total,
        "coverage_rate": (coverage_hits / coverage_total) if coverage_total else None,
        "per_fact": per_fact,
        "recall_accuracy_query_patient": sum(r["query_patient_hit"] for r in per_fact) / len(per_fact),
        "moat_checks": moat_checks,
        "moat_pass_rate": (sum(m["abstained"] for m in moat_checks) / len(moat_checks)) if moat_checks else None,
        "mismatch_checks": mismatch_checks,
        "mismatch_pass_rate": (sum(m["did_not_leak_fact_i_patient"] for m in mismatch_checks) / len(mismatch_checks))
                              if mismatch_checks else None,
    }
    if full_methods:
        result["recall_accuracy_query_agent"] = sum(r["query_agent_hit"] for r in per_fact) / len(per_fact)
        result["recall_accuracy_yes_no"] = sum(r["ask_yes_no_hit"] for r in per_fact) / len(per_fact)
    return result


def mode_compose(args, blind=False):
    vocab, facts, _ = _load_live_bundle()
    result = _run_compose(args.seed, args.fanout, args.n_facts, blind, vocab, facts,
                          full_methods=args.full_methods, n_moat=args.n_moat, n_mismatch=args.n_mismatch)
    print(json.dumps(result, indent=2))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["formula", "compose", "blind"], required=True)
    ap.add_argument("--fanouts", type=int, nargs="+", default=[8, 16, 32, 64], help="formula mode: sweep values")
    ap.add_argument("--seed", type=int, default=42, help="compose/blind mode: single seed (loop externally)")
    ap.add_argument("--fanout", type=int, default=8, help="compose/blind mode: single fanout (loop externally)")
    ap.add_argument("--n-facts", type=int, default=2, help="compose/blind mode: sample size of REAL facts")
    ap.add_argument("--full-methods", action="store_true",
                    help="also test query_agent/ask_yes_no (default: query_patient only -- see _run_compose "
                         "docstring for the cost rationale)")
    ap.add_argument("--n-moat", type=int, default=1, help="number of never-stored-cue moat probes")
    ap.add_argument("--n-mismatch", type=int, default=1, help="number of mismatched-role-cue anti-cheat probes")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.mode == "formula":
        mode_formula(args)
    elif args.mode == "compose":
        mode_compose(args, blind=False)
    else:
        mode_compose(args, blind=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
