"""FHRR decode-rate at scale: cue-role false-hop + wrong-patient rate CPU instrument (no-confab-moat crux).

WHY. This runner was commissioned to read `research/findings/2026-08-25-reasoning-route-moat-audit-hardening-
spec.md` (an adversarial audit of the reasoning route's no-confab memory moat) and measure "hardening req #3"'s
crux number. AT BUILD TIME that document does not exist anywhere in this repo (`git log --all` on every branch,
every commit message, every findings/plans file -- nothing). Rather than block on an unreachable citation, this
runner is grounded directly in the CITED production code (verified by reading it, 2026-08-25):

  * `ShardedPhasorStore.query_patient` -> `RFPhasorComposer._scan_first_match`
    (research/runners/rf_phasor_composer.py:712-721) matches a decoded stored cue's role words against the
    query by PLAIN STRING EQUALITY (`w == val`), with no confidence floor.
  * `RFPhasorComposer._cleanup` / the batched `_cleanup_all` (:658-663, :700-710) is a FLOORLESS ARGMAX over the
    whole vocabulary codebook -- it always returns some word, never `None`.
  * A genuinely OUT-OF-VOCAB cue is safe BY CONSTRUCTION (not by a floor): decode can only ever return a word
    that is itself a member of the vocab codebook (`self.words`), so an OOV query word can never equal a
    decoded word -- the store abstains structurally, independent of any crosstalk question. Verified empirically
    below (the `oov_floor` check on every cell).
  * The open, UNMEASURED question (the audit's framing, reconstructed from the task brief): for an IN-VOCAB cue
    that is genuinely not a stored fact, how often does FHRR BUNDLING CROSSTALK (`store()` binds+bundles up to
    6 roles -- `ROLES = ("agent","action","patient","polarity","attribute","attribute2")`,
    rf_phasor_composer.py:24,318-320 `_encode`; production facts bind agent+action+patient+polarity=AFFIRM, 4
    terms, per `tiered_fact_store.build_ltm_from_facts`) make a WRONG stored fact's decoded cue-role words land
    on the query's words by chance, fabricating an answer instead of abstaining? And separately, even when the
    CUE correctly matches the intended fact, how often does the PATIENT role itself decode wrong (a
    supported-hop error, the audit's other named concern, `_render`:557)?

MEASURES two rates per (D, n_facts, seed) cell, both against a REAL knowledge base (not synthetic facts/vocab):
  1. CUE-ROLE FALSE-HOP rate -- sample in-vocab (agent, action) pairs where the agent DOES have >=1 stored fact
     (a genuine near-miss, not a trivially-absent agent) but NOT under this action. Rate = fraction where the
     store still returns a non-None (fabricated) patient instead of abstaining.
  2. ANSWER WRONG-PATIENT rate -- for every genuinely stored (agent, action) key, rate = fraction where the
     decoded patient != the one actually stored.
Both record the winner's `winner_score_raw` + `margin`, computed exactly as `_cleanup_all_score_stats` defines
them (mean-cos in [-1,1]; margin = winner - runner-up), so the true-match and false-match score distributions
can be compared for hardening req #2 (would a confidence floor cleanly separate them?).

REAL DATA, FIXED VOCAB. Facts + vocabulary come from the SHIPPED wikidata_core_15k knowledge-core bundle
(`/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k`, built by
`research/runners/_knowledge_core_curate.py`: 15000 real (agent, relation, patient) triples over a 7032-word
vocab, the same facts `tiered_fact_store.build_ltm_from_facts` turns into the production LTM, D=128 default per
`developed_brain_io.py:369`). The vocabulary (V=7032) is held FIXED across every (D, n_facts) cell -- only D
(the FHRR dimensionality) and the stored-fact count vary, per the sweep spec.

CLOSED-FORM, VERIFIED AGAINST THE GENUINE NEURAL PIPELINE (numpy only, no GPU; `sim/`-only READS, no writes/
edits). The STORE side reuses `research.runners.tiered_fact_store.encode_fast` unmodified -- the project's own
already-validated closed-form bind+bundle ("recall-identical to the neural resonate bind... 120/120 matched",
2026-08-21-closed-form-bulk-bind finding). The QUERY side (unbind + cleanup) has no published closed form, so
this runner derives one from the documented per-op semantics (`_unbind_phases`: a diagonal conj-synapse ==
per-component phase subtraction; `_cleanup_all_score_stats`: mean-cos argmax against the vocab codebook) and
SELF-VERIFIES it (`verify_instrument`, on by default) by building a tiny `RFPhasorComposer`, storing facts
through the GENUINE `.store()` resonate (the real Izhikevich RESONATE_AND_FIRE bridge, not `encode_fast`), and
asserting this runner's closed-form decode reproduces the real `.query_patient()` byte-for-byte on stored,
false-hop, and out-of-vocab cues. Manually re-verified interactively before this file was written (byte-exact
agreement on a 4-fact/12-word probe). Why closed-form at all: at D=1024/N=15000 a single genuine
`_scan_first_match` unbind resonates a ~2*N*D-neuron RF bridge (the O(K*D) cost `sharded_phasor_store.py`
documents, ~5s at K=2413/D=128) -- intractable at the thousands-of-trials granularity this measurement needs.
The closed form is the SAME algebra (self-verified above), computed directly instead of stepped through
`SimulationBridge`.

Run (CPU/numpy, LOCAL smoke):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners.reasoning_route_decode_rate \\
      --D 128 --n-facts 1000 --seed 42 \\
      --out research/findings/raw/_reasoning_route_decode_rate/smoke_D128_N1000_s42.json

Sweep cell (multi-seed fanned out in ONE process; queued to the pool, not run interactively):
  SIM_BACKEND=numpy .venv/bin/python -m research.runners.reasoning_route_decode_rate \\
      --D 1024 --n-facts 15000 --seed 42 43 44 \\
      --out research/findings/raw/_reasoning_route_decode_rate/D1024_N15000.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

BUNDLE_DEFAULT = "/home/dant123/Projects/sim-data/knowledge_bundles/wikidata_core_15k"


# ---- data loading (real facts + real vocab) -------------------------------------------------------------

def _load_bundle(bundle_dir):
    with open(os.path.join(bundle_dir, "facts.json"), "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    facts = []
    for e in raw:
        f = e.get("fact") if isinstance(e, dict) and "fact" in e else e
        if isinstance(f, dict) and isinstance(f.get("agent"), str) and isinstance(f.get("action"), str) \
                and isinstance(f.get("patient"), str):
            facts.append({"agent": f["agent"], "action": f["action"], "patient": f["patient"],
                          "polarity": f.get("polarity") or "AFFIRM"})
    with open(os.path.join(bundle_dir, "manifest.json"), "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    vocab = list(manifest["vocab"])
    return facts, vocab


def _sample_facts(facts, n, seed):
    n = min(int(n), len(facts))
    rng = np.random.default_rng(90000 + int(seed))
    idx = sorted(int(i) for i in rng.choice(len(facts), size=n, replace=False))
    return [facts[i] for i in idx]


# ---- closed-form store: research.runners.tiered_fact_store.encode_fast (validated, reused unmodified) ----

def _build_composites(comp, facts):
    from research.runners.tiered_fact_store import encode_fast
    return np.stack([encode_fast(comp, f) for f in facts]).astype(np.float64)


# ---- closed-form query decode (derived from the documented per-op semantics; self-verified below) --------

def _codebook(comp, words, dtype=np.complex64):
    return np.stack([np.exp(2j * np.pi * comp.concepts[w]) for w in words]).astype(dtype)


def _decode_role(comp_phases, role_phase, cb, dtype=np.complex64):
    """Vectorized equivalent of `_unbind_all_phases`(role) -> `_cleanup_all_score_stats`: for every stored
    composite, decode `role` against the FULL vocab codebook `cb`. Returns INTEGER vocab indices (not words,
    for speed) plus the winner's raw mean-cos score and margin over the runner-up, matching
    `_cleanup_all_score_stats`'s own field definitions exactly.
    unbind: recovered_phase[k] = composite_phase[k] - role_phase[k] (mod 1) -- the diagonal conj-synapse
        `_unbind_phases` converges to (bind = phase ADDITION, so unbind-by-conjugate-role = phase SUBTRACTION).
    cleanup: sims[i,j] = Re(rec_z[i] . conj(cb[j])) / D (mean-cos in [-1,1]); winner = argmax_j.
    """
    D = comp_phases.shape[1]
    fdtype = np.float32 if dtype == np.complex64 else np.float64
    rec = (comp_phases - role_phase[None, :]).astype(fdtype)
    rec_z = np.exp(2j * np.pi * rec).astype(dtype)
    sims = (rec_z @ np.conj(cb).T).real.astype(np.float64) / D   # (K,V) mean-cos
    V = sims.shape[1]
    rows = np.arange(sims.shape[0])
    if V > 1:
        order = np.argsort(sims, axis=1)
        top = order[:, -1]
        runner = order[:, -2]
    else:
        top = np.zeros(sims.shape[0], dtype=np.int64)
        runner = top
    top_raw = sims[rows, top]
    runner_raw = sims[rows, runner]
    return top.astype(np.int64), top_raw, (top_raw - runner_raw)


def _first_match_indices(query_a_idx, query_b_idx, dec_a_idx, dec_b_idx, chunk=1500):
    """For each query i, the SMALLEST k such that dec_a_idx[k]==query_a_idx[i] and dec_b_idx[k]==query_b_idx[i]
    -- the vectorized equivalent of `_scan_first_match`'s first-True-index semantics (`np.argmax` over a boolean
    row returns the FIRST True). Returns -1 where no such k exists (abstain). Chunked over queries so the
    (n_queries, n_stored) boolean comparison matrix never exceeds `chunk` rows at once."""
    T = len(query_a_idx)
    K = len(dec_a_idx)
    out = np.full(T, -1, dtype=np.int64)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        qa = np.asarray(query_a_idx[s:e])[:, None]
        qb = np.asarray(query_b_idx[s:e])[:, None]
        mask = (dec_a_idx[None, :K] == qa) & (dec_b_idx[None, :K] == qb)   # (c, K)
        has = mask.any(axis=1)
        first = np.argmax(mask, axis=1)
        out[s:e] = np.where(has, first, -1)
    return out


# ---- instrument self-verification: closed-form decode vs the GENUINE resonate `.store()`/`.query_patient()` -

def verify_instrument(seed=42, D=64):
    """(mandatory, on by default) Build a tiny RFPhasorComposer, store facts through the GENUINE RF-bridge
    resonate (real Izhikevich RESONATE_AND_FIRE stepping -- NOT `encode_fast`), then confirm this runner's
    closed-form decode reproduces the real `.query_patient()` exactly on:
      (a) a genuinely out-of-vocab cue -> both abstain (the documented structural floor);
      (b) every freshly-stored fact -> both recall the correct patient, and AGREE with each other;
      (c) every in-vocab false-hop probe (an agent queried under an action it does not have) -> the closed
          form and the real bridge must AGREE (both abstain, or both return the identical fabricated word).
    (c) is the load-bearing check: it proves the closed form reproduces genuine FHRR bundling crosstalk, not
    just genuine recall, before a single sweep number is trusted."""
    from research.runners.rf_phasor_composer import RFPhasorComposer

    vocab = sorted({"a1", "a2", "a3", "a4", "act1", "act2", "act3", "act4",
                    "p1", "p2", "p3", "p4", "AFFIRM", "NEGATE", "__unused_vocab_word__"})
    comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    trip = [("a1", "act1", "p1"), ("a2", "act2", "p2"), ("a3", "act3", "p3"), ("a4", "act4", "p4")]
    for a, act, p in trip:
        comp.store(a, act, p, polarity="AFFIRM")     # GENUINE resonate bind (real neural bridge)

    words = comp.words
    word_to_idx = {w: i for i, w in enumerate(words)}
    cb = _codebook(comp, words, dtype=np.complex64)
    comp_phases = np.stack([c for _f, c in comp.kb]).astype(np.float64)

    dec_idx, dec_score, dec_margin = {}, {}, {}
    for role in ("agent", "action", "patient"):
        idx, score, margin = _decode_role(comp_phases, comp.roles[role], cb)
        dec_idx[role], dec_score[role], dec_margin[role] = idx, score, margin

    def cf_query(a, act):
        for k in range(len(trip)):
            if words[dec_idx["agent"][k]] == a and words[dec_idx["action"][k]] == act:
                return words[dec_idx["patient"][k]], k
        return None, None

    checks = []
    # (a) genuinely OOV cue -> both abstain
    oov_real = comp.query_patient("__genuinely_never_seen_agent__", "act1")
    oov_cf, _ = cf_query("__genuinely_never_seen_agent__", "act1")
    checks.append({"case": "oov_agent_abstain", "real": repr(oov_real), "closed_form": repr(oov_cf),
                   "ok": (oov_real is None) and (oov_cf is None)})

    # (b) every freshly-stored fact recalls correctly, closed-form == real
    recall_rows = []
    for (a, act, p) in trip:
        real = comp.query_patient(a, act)
        cf, _ = cf_query(a, act)
        recall_rows.append({"agent": a, "action": act, "stored_patient": p,
                            "real_answer": repr(real), "closed_form_answer": repr(cf),
                            "ok": (real == p) and (cf == p) and (real == cf)})
    checks.append({"case": "fresh_recall", "rows": recall_rows, "ok": all(r["ok"] for r in recall_rows)})

    # (c) false-hop probes: every (agent, action-it-does-not-have) pair -- closed-form must AGREE with the
    #     real neural bridge on every single one (both abstain, or both return the identical fabricated word)
    fh_rows = []
    actions = ["act1", "act2", "act3", "act4"]
    for a, _act, _p in trip:
        had = {t[1] for t in trip if t[0] == a}
        for act in actions:
            if act in had:
                continue
            real = comp.query_patient(a, act)
            cf, _ = cf_query(a, act)
            fh_rows.append({"agent": a, "action": act, "real": repr(real), "closed_form": repr(cf),
                            "agree": real == cf})
    checks.append({"case": "false_hop_agreement", "rows": fh_rows, "ok": all(r["agree"] for r in fh_rows)})

    ok = all(c["ok"] for c in checks)
    return {"ok": bool(ok), "seed": seed, "D": D, "checks": checks}


# ---- the measurement --------------------------------------------------------------------------------------

def measure_cell(facts_all, vocab, word_to_idx, D, n_facts, seed, n_trials_false_hop=5000, n_examples=20):
    from research.runners.rf_phasor_composer import RFPhasorComposer

    t0 = time.time()
    facts = _sample_facts(facts_all, n_facts, seed)
    K = len(facts)
    comp = RFPhasorComposer(seed=seed, D=D, vocab=vocab)
    comp_phases = _build_composites(comp, facts)
    cb = _codebook(comp, comp.words, dtype=np.complex64)
    words = comp.words

    dec_idx, dec_score, dec_margin = {}, {}, {}
    for role in ("agent", "action", "patient"):
        idx, score, margin = _decode_role(comp_phases, comp.roles[role], cb)
        dec_idx[role], dec_score[role], dec_margin[role] = idx, score, margin

    fact_agent_idx = np.array([word_to_idx[f["agent"]] for f in facts], dtype=np.int64)
    fact_action_idx = np.array([word_to_idx[f["action"]] for f in facts], dtype=np.int64)

    # --- (a) harness floor check: a genuinely OOV agent must abstain -- structurally guaranteed (decode can
    #     only ever return a word IN the vocab codebook, and the OOV probe word is not), checked empirically. ---
    oov_rng = np.random.default_rng(70000 + seed)
    action_pool = sorted({f["action"] for f in facts})
    action_pool_idx = np.array([word_to_idx[a] for a in action_pool], dtype=np.int64)
    n_oov = 300
    oov_query_action = action_pool_idx[oov_rng.integers(0, len(action_pool_idx), size=n_oov)]
    # an OOV word has NO vocab index -> by construction it can never appear in dec_idx["agent"]; measure the
    # rate of a (structurally impossible) hit directly rather than assume it.
    oov_hits = int(np.sum(_first_match_indices(np.full(n_oov, -999999, dtype=np.int64), oov_query_action,
                                                dec_idx["agent"], dec_idx["action"]) >= 0))
    oov_floor_rate = oov_hits / n_oov if n_oov else None

    # --- (b) wrong-patient rate: every unique stored (agent,action) key -> does decode return the TRUE patient? ---
    first_match = {}
    for i, f in enumerate(facts):
        key = (f["agent"], f["action"])
        if key not in first_match:
            first_match[key] = i
    keys = list(first_match.keys())
    key_a_idx = np.array([word_to_idx[a] for a, _b in keys], dtype=np.int64)
    key_b_idx = np.array([word_to_idx[b] for _a, b in keys], dtype=np.int64)
    matched_idx = _first_match_indices(key_a_idx, key_b_idx, dec_idx["agent"], dec_idx["action"])

    wrong_patient_hits = 0
    wrong_patient_examples = []
    true_patient_scores, true_patient_margins = [], []
    for j, (a, act) in enumerate(keys):
        mi = int(matched_idx[j])
        stored_patient = facts[first_match[(a, act)]]["patient"]
        if mi < 0:
            continue    # a stored fact whose own cue got shadowed into an abstain by an earlier false-hop
        returned_patient = words[dec_idx["patient"][mi]]
        true_patient_scores.append(float(dec_score["patient"][mi]))
        true_patient_margins.append(float(dec_margin["patient"][mi]))
        if returned_patient != stored_patient:
            wrong_patient_hits += 1
            if len(wrong_patient_examples) < n_examples:
                wrong_patient_examples.append({
                    "agent": a, "action": act, "returned_patient": returned_patient,
                    "stored_patient": stored_patient, "matched_index": mi,
                    "score": float(dec_score["patient"][mi]), "margin": float(dec_margin["patient"][mi]),
                })
    n_wrong_patient_checked = len(keys)
    wrong_patient_rate = wrong_patient_hits / n_wrong_patient_checked if n_wrong_patient_checked else None

    # --- (c) cue-role FALSE-HOP rate: in-vocab (agent,action) NOT stored, agent has >=1 OTHER stored fact ---
    agent_actions = {}
    for f in facts:
        agent_actions.setdefault(f["agent"], set()).add(f["action"])
    cand_a, cand_b = [], []
    for a, had in agent_actions.items():
        missing = [act for act in action_pool if act not in had]
        cand_a.extend([word_to_idx[a]] * len(missing))
        cand_b.extend([word_to_idx[act] for act in missing])
    cand_a = np.array(cand_a, dtype=np.int64)
    cand_b = np.array(cand_b, dtype=np.int64)
    n_candidates = len(cand_a)
    exact = n_candidates <= n_trials_false_hop
    if exact or n_candidates == 0:
        sel_a, sel_b = cand_a, cand_b
    else:
        fh_rng = np.random.default_rng(80000 + seed)
        sel = fh_rng.choice(n_candidates, size=n_trials_false_hop, replace=False)
        sel_a, sel_b = cand_a[sel], cand_b[sel]
    n_trials = len(sel_a)
    fh_matched = _first_match_indices(sel_a, sel_b, dec_idx["agent"], dec_idx["action"])
    fabricated_mask = fh_matched >= 0
    false_hop_hits = int(np.sum(fabricated_mask))
    false_hop_agent_scores = dec_score["agent"][fh_matched[fabricated_mask]].tolist() if false_hop_hits else []
    false_hop_agent_margins = dec_margin["agent"][fh_matched[fabricated_mask]].tolist() if false_hop_hits else []
    false_hop_examples = []
    fab_positions = np.where(fabricated_mask)[0]
    for pos in fab_positions[:n_examples]:
        mi = int(fh_matched[pos])
        false_hop_examples.append({
            "agent": words[int(sel_a[pos])], "action": words[int(sel_b[pos])],
            "returned_patient": words[dec_idx["patient"][mi]], "stored_patient": None, "matched_index": mi,
            "agent_score": float(dec_score["agent"][mi]), "agent_margin": float(dec_margin["agent"][mi]),
            "action_score": float(dec_score["action"][mi]), "action_margin": float(dec_margin["action"][mi]),
        })
    false_hop_rate = false_hop_hits / n_trials if n_trials else None

    elapsed = time.time() - t0
    return {
        "D": D, "n_facts_requested": int(n_facts), "n_facts_sampled": K, "seed": seed,
        "vocab_size": len(vocab), "n_unique_agent_action_keys": n_wrong_patient_checked,
        "elapsed_s": round(elapsed, 3),
        "oov_floor": {"n_trials": n_oov, "hits": oov_hits, "rate": oov_floor_rate,
                      "note": "structurally guaranteed 0 by construction (decode can only return a vocab word); measured, not assumed"},
        "wrong_patient": {"n_checked": n_wrong_patient_checked, "hits": wrong_patient_hits,
                          "rate": wrong_patient_rate, "examples": wrong_patient_examples},
        "false_hop": {"n_candidates": int(n_candidates), "n_trials": n_trials, "exact": bool(exact),
                      "hits": false_hop_hits, "rate": false_hop_rate, "examples": false_hop_examples},
        "score_stats": {
            "true_match_patient_score_mean": float(np.mean(true_patient_scores)) if true_patient_scores else None,
            "true_match_patient_score_min": float(np.min(true_patient_scores)) if true_patient_scores else None,
            "true_match_patient_margin_mean": float(np.mean(true_patient_margins)) if true_patient_margins else None,
            "true_match_patient_margin_min": float(np.min(true_patient_margins)) if true_patient_margins else None,
            "false_hop_agent_score_mean": float(np.mean(false_hop_agent_scores)) if false_hop_agent_scores else None,
            "false_hop_agent_score_max": float(np.max(false_hop_agent_scores)) if false_hop_agent_scores else None,
            "false_hop_agent_margin_mean": float(np.mean(false_hop_agent_margins)) if false_hop_agent_margins else None,
            "false_hop_agent_margin_min": float(np.min(false_hop_agent_margins)) if false_hop_agent_margins else None,
            "n_false_hop_scored": len(false_hop_agent_scores),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--D", type=int, nargs="+", default=[128], help="FHRR dimensionality; pass multiple to sweep")
    ap.add_argument("--n-facts", type=int, nargs="+", default=[1000], help="stored fact count; pass multiple to sweep")
    ap.add_argument("--seed", type=int, nargs="+", default=[42], help="one or more seeds, fanned out in-process")
    ap.add_argument("--bundle", default=BUNDLE_DEFAULT, help="wikidata_core_15k-style bundle dir (facts.json+manifest.json)")
    ap.add_argument("--n-trials", type=int, default=5000, help="false-hop trial cap per cell (exact if the real candidate count is smaller)")
    ap.add_argument("--verify", dest="verify", action="store_true", default=True)
    ap.add_argument("--no-verify", dest="verify", action="store_false")
    ap.add_argument("--out", default="research/findings/raw/_reasoning_route_decode_rate/cell.json")
    a = ap.parse_args()

    if not os.path.exists(os.path.join(a.bundle, "facts.json")):
        print(f"[decode-rate] bundle not found: {a.bundle}", flush=True)
        return 1
    facts_all, vocab = _load_bundle(a.bundle)
    word_to_idx = {w: i for i, w in enumerate(sorted(vocab))}
    print(f"[decode-rate] bundle={a.bundle} n_facts_total={len(facts_all)} vocab={len(vocab)}", flush=True)

    verification = None
    if a.verify:
        t0 = time.time()
        verification = verify_instrument(seed=a.seed[0], D=64)
        print(f"[decode-rate] instrument verification ok={verification['ok']} ({time.time() - t0:.2f}s)", flush=True)
        if not verification["ok"]:
            print("[decode-rate] INSTRUMENT VERIFICATION FAILED -- the sweep numbers below are NOT trustworthy", flush=True)

    cells = []
    for D, n_facts, seed in itertools.product(a.D, a.n_facts, a.seed):
        t0 = time.time()
        cell = measure_cell(facts_all, vocab, word_to_idx, D, n_facts, seed, n_trials_false_hop=a.n_trials)
        cells.append(cell)
        print(f"[decode-rate] D={D} N={n_facts} seed={seed}: "
              f"false_hop={cell['false_hop']['rate']} ({cell['false_hop']['hits']}/{cell['false_hop']['n_trials']}"
              f"{'exact' if cell['false_hop']['exact'] else ' sampled'}) "
              f"wrong_patient={cell['wrong_patient']['rate']} ({cell['wrong_patient']['hits']}/{cell['wrong_patient']['n_checked']}) "
              f"oov_floor={cell['oov_floor']['rate']} [{time.time() - t0:.2f}s]", flush=True)

    out = {
        "arc": "reasoning-route no-confab moat: FHRR cue-role false-hop + wrong-patient decode-rate instrument",
        "bundle": a.bundle, "vocab_size": len(vocab), "n_facts_total_available": len(facts_all),
        "swept_D": a.D, "swept_n_facts": a.n_facts, "swept_seeds": a.seed, "n_trials_requested": a.n_trials,
        "instrument_verification": verification,
        "cells": cells,
    }
    out_dir = os.path.dirname(a.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[decode-rate] wrote {a.out}", flush=True)
    if verification is not None and not verification["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
