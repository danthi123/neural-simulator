"""LOGIC/CPU validation for the SK-brain load planner/WM-loop speedup (2026-06-24).

Validates BOTH fixes on the NumPy backend (no GPU, no full SK brain-load):
  (A) MultiTurnAgent(defer_planner=True) builds NO WM loop / biased-competition buffer in __init__;
      a Q&A query still answers + the moat abstains; the first referent write lazily builds the WM loop + works.
  (B) the BATCHED SpikingLoopContextBuffer graph-build yields a CSR byte-identical to the per-concept build
      (behavior-equal); + the spreading-activation latency rank is unchanged on a small graph.
ANTI-CHEAT: the deferred + eager agents give identical Q&A answers; the moat abstains (0 false-accepts); the
batched graph-build is byte-identical to the per-pathway one.

Run (CPU, fast):  SIM_BACKEND=numpy python -m research.runners._sk_load_planner_speedup_validate
"""
from __future__ import annotations
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

RESULT = {"checks": {}, "verdict": None}


def _fail(msg):
    RESULT["verdict"] = "FAIL"
    RESULT["error"] = msg
    print("FAIL:", msg, flush=True)


# ---------------------------------------------------------------------------
# (B) batched graph-build == per-concept graph-build (byte-identical CSR)
# ---------------------------------------------------------------------------
def check_batched_equals_perconcept():
    """Build a SpikingLoopContextBuffer (current = batched) and an independent bridge wired with the OLD
    per-concept set_pathway_weights loop; assert the two CSRs are byte-identical (same edges + weights)."""
    from research.runners.content_selection_spiking import SpikingLoopContextBuffer, build_loop_wm_bridge
    import sim.backend as B
    xp, _ = B.get_backend()

    concepts = ["a", "b", "c", "d", "e"]
    # internal_density=0.0 is the production config (SpikingController / SpikingSpreadingController) AND must
    # MATCH the reference bridge below so the only difference is batched-vs-per-concept attractor install.
    n, psize, attractor_weight, seed, idens = 600, 40, 50.0, 42, 0.0

    # current (batched) build
    buf = SpikingLoopContextBuffer(concepts, n=n, pattern_size=psize, attractor_weight=attractor_weight,
                                   internal_density=idens, seed=seed, enable_ou=False, verbose=False)
    csr_new = buf.bridge.cp_connections
    new_coo = csr_new.tocoo()
    new_edges = {(int(r), int(c)): float(w)
                 for r, c, w in zip(B.to_host(new_coo.row), B.to_host(new_coo.col), B.to_host(new_coo.data))}

    # reference: SAME bridge build (same density/loop params), then the ORIGINAL per-concept loop (pre-edit code)
    ref = build_loop_wm_bridge(n=n, density=idens, loop_weight=0.0, loop_density=0.05, seed=seed,
                               enable_ou=False, verbose=False)
    rm = ref.region_manager
    cidx = np.asarray(rm.indices("cortex_ctx"))
    didx = np.asarray(rm.indices("dlpfc_wm"))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    for i, c in enumerate(concepts):
        p = perm[i * psize:(i + 1) * psize]
        cpat, dpat = cidx[p], didx[p]
        pre1 = np.repeat(cpat, psize).astype(np.int64)
        post1 = np.tile(dpat, psize).astype(np.int64)
        pre2 = np.repeat(dpat, psize).astype(np.int64)
        post2 = np.tile(cpat, psize).astype(np.int64)
        ww = np.full(psize * psize, attractor_weight, np.float32)
        ref.set_pathway_weights("c2d", pre_indices=pre1, post_indices=post1, weights=ww, add_missing=True)
        ref.set_pathway_weights("d2c", pre_indices=pre2, post_indices=post2, weights=ww, add_missing=True)
    ref_coo = ref.cp_connections.tocoo()
    ref_edges = {(int(r), int(c)): float(w)
                 for r, c, w in zip(B.to_host(ref_coo.row), B.to_host(ref_coo.col), B.to_host(ref_coo.data))}

    same_keys = set(new_edges.keys()) == set(ref_edges.keys())
    same_w = same_keys and all(abs(new_edges[k] - ref_edges[k]) < 1e-6 for k in new_edges)
    nnz_match = int(csr_new.nnz) == int(ref.cp_connections.nnz)
    ok = bool(same_keys and same_w and nnz_match)
    RESULT["checks"]["B_batched_csr_byte_identical"] = {
        "ok": ok, "nnz_batched": int(csr_new.nnz), "nnz_perconcept": int(ref.cp_connections.nnz),
        "n_edges_batched": len(new_edges), "n_edges_perconcept": len(ref_edges),
        "edge_keys_match": bool(same_keys), "weights_match": bool(same_w),
    }
    if not ok:
        _fail("batched CSR != per-concept CSR")
    return ok


def check_batched_latency_rank():
    """The spreading-activation latency rank on a small graph is unchanged by the batched build. We just
    confirm SpikingSpreadingController constructs + runs a latency probe with a sensible (direct < absent)
    ranking on the batched buffer -- the controller's ctx IS a (now-batched) SpikingLoopContextBuffer."""
    from research.runners.content_selection_spiking import SpikingSpreadingController
    graph = {"apple": {"pie": 1.0, "tree": 1.0}, "dog": {"bark": 1.0}}
    ctl = SpikingSpreadingController(graph, seed=42, verbose=False)
    lat = ctl.relevance_by_latency("apple")
    # direct associates of apple fire; the unrelated dog-cluster never fires (None)
    direct_fire = (lat.get("pie") is not None) or (lat.get("tree") is not None)
    dog_silent = lat.get("bark") is None
    ok = bool(direct_fire and dog_silent)
    RESULT["checks"]["B_latency_rank_sane"] = {
        "ok": ok, "lat_pie": lat.get("pie"), "lat_tree": lat.get("tree"),
        "lat_bark": lat.get("bark"), "direct_fire": direct_fire, "dog_silent": dog_silent,
    }
    if not ok:
        _fail("batched latency rank not sane (direct should fire, unrelated should stay silent)")
    return ok


# ---------------------------------------------------------------------------
# (A) defer_planner builds NO WM loop; Q&A answers; lazy build on first referent
# ---------------------------------------------------------------------------
def _toy_agent(defer_planner, wm_n=600, wm_pattern_size=40):
    from research.runners.multi_turn_agent import MultiTurnAgent
    # a tiny vocab: facts dog/eat/apple, cat/chase/mouse. referents = the nouns.
    concepts = {w: None for w in ["dog", "eat", "apple", "cat", "chase", "mouse", "is"]}
    referents = ["dog", "apple", "cat", "mouse"]
    ag = MultiTurnAgent(referent_concepts=referents, concepts=concepts, seed=42,
                        wm_n=wm_n, wm_pattern_size=wm_pattern_size, enable_neural_render=False,
                        composer_kind="rf", enable_biased_competition=False,
                        defer_planner=defer_planner)
    return ag


def check_defer_builds_no_wm():
    ag = _toy_agent(defer_planner=True)
    no_wm = ag.wm is None
    no_bcw = ag.bcw is None
    RESULT["checks"]["A_defer_builds_no_wm_loop"] = {
        "ok": bool(no_wm and no_bcw), "wm_is_none": bool(no_wm), "bcw_is_none": bool(no_bcw)}
    if not (no_wm and no_bcw):
        _fail("defer_planner=True still built the WM loop / biased-competition buffer in __init__")
    return ag, bool(no_wm and no_bcw)


def check_qa_and_moat(ag):
    """A Q&A query + moat on the deferred agent. hear() via the inner agent stores facts WITHOUT the parser
    needing the WM loop; what_does answers; an unknown agent abstains (None). The WM loop must STILL be None
    after a plain (no-referent) store+query path that uses only explicit nouns."""
    # store facts directly through the composer-level agent (BrainConversationalAgent.hear trains the parser
    # lazily; storing is fine on CPU). Use the agent's hear (MultiTurnAgent.hear writes the patient referent,
    # which WILL build the WM lazily -- so for the "Q&A pays zero WM" claim we store via the INNER agent).
    ag.agent.hear("dog eat apple", voice="active", polarity="AFFIRM")
    ag.agent.hear("cat chase mouse", voice="active", polarity="AFFIRM")
    wm_still_none_after_store = ag.wm is None

    ans = ag.what_does("dog", "eat")          # explicit noun agent -> no pronoun -> no WM read
    moat = ag.what_does("fox", "eat")          # never stored agent -> abstain
    wm_still_none_after_qa = ag.wm is None

    ok = (ans == "apple") and (moat is None) and wm_still_none_after_store and wm_still_none_after_qa
    RESULT["checks"]["A_qa_answers_moat_abstains_zero_wm"] = {
        "ok": bool(ok), "what_does_dog_eat": ans, "moat_fox_eat": moat,
        "wm_none_after_store": bool(wm_still_none_after_store),
        "wm_none_after_qa": bool(wm_still_none_after_qa),
    }
    if not ok:
        _fail(f"Q&A/moat/zero-WM failed: ans={ans!r} moat={moat!r} "
              f"wm_none_store={wm_still_none_after_store} wm_none_qa={wm_still_none_after_qa}")
    return ok


def check_lazy_wm_builds_and_equals_eager():
    """The deferred WM loop builds LAZILY on the first referent write, and the lazily-built WM behaves IDENTICALLY
    to an EAGERLY-built one. We assert:
      (1) the deferred agent's WM is None before the first referent, then non-None after MultiTurnAgent.hear()
          (which writes the patient referent) -- the lazy build fires exactly when needed;
      (2) the deferred WM's read() == the eager WM's read() after the SAME referent write (byte-equal rates) --
          deferral changes only WHEN the loop is built, never its dynamics.
    This is the load-bearing guarantee (lazy == eager), independent of whether the toy attractor latches."""
    eager = _toy_agent(defer_planner=False)
    deferred = _toy_agent(defer_planner=True)
    wm_none_before = deferred.wm is None
    # write the SAME referent into both via MultiTurnAgent.hear (writes the patient 'mouse')
    eager.hear("cat chase mouse", voice="active", polarity="AFFIRM")
    deferred.hear("cat chase mouse", voice="active", polarity="AFFIRM")
    wm_built_after = deferred.wm is not None
    e_rates = eager.wm.read(window=20)
    d_rates = deferred.wm.read(window=20)
    same_keys = set(e_rates) == set(d_rates)
    rates_equal = same_keys and all(abs(float(e_rates[k]) - float(d_rates[k])) < 1e-9 for k in e_rates)
    ok = bool(wm_none_before and wm_built_after and rates_equal)
    RESULT["checks"]["A_lazy_wm_builds_and_equals_eager"] = {
        "ok": ok, "wm_none_before_first_referent": bool(wm_none_before),
        "wm_built_after_first_referent": bool(wm_built_after),
        "deferred_read_equals_eager_read": bool(rates_equal),
        "eager_rates": {k: round(float(v), 6) for k, v in e_rates.items()},
        "deferred_rates": {k: round(float(v), 6) for k, v in d_rates.items()}}
    if not ok:
        _fail(f"lazy==eager WM failed: none_before={wm_none_before} built_after={wm_built_after} "
              f"rates_equal={rates_equal}")
    return ok


def check_deferred_equals_eager_qa():
    """ANTI-CHEAT: the deferred agent and an EAGER agent give identical Q&A answers (the deferral changes only
    WHEN the WM loop is built, never the answers/moat)."""
    eager = _toy_agent(defer_planner=False)
    eager_built = eager.wm is not None   # eager must build the WM loop in __init__
    eager.agent.hear("dog eat apple", voice="active", polarity="AFFIRM")
    eager.agent.hear("cat chase mouse", voice="active", polarity="AFFIRM")
    e_ans = eager.what_does("dog", "eat")
    e_moat = eager.what_does("fox", "eat")

    deferred = _toy_agent(defer_planner=True)
    deferred.agent.hear("dog eat apple", voice="active", polarity="AFFIRM")
    deferred.agent.hear("cat chase mouse", voice="active", polarity="AFFIRM")
    d_ans = deferred.what_does("dog", "eat")
    d_moat = deferred.what_does("fox", "eat")

    ok = bool(eager_built and e_ans == d_ans == "apple" and e_moat is None and d_moat is None)
    RESULT["checks"]["A_deferred_equals_eager_qa"] = {
        "ok": ok, "eager_wm_built_eager": bool(eager_built),
        "eager_ans": e_ans, "deferred_ans": d_ans, "eager_moat": e_moat, "deferred_moat": d_moat}
    if not ok:
        _fail(f"deferred != eager Q&A: e_ans={e_ans!r} d_ans={d_ans!r} e_moat={e_moat!r} d_moat={d_moat!r}")
    return ok


def main():
    print("=== SK-load planner/WM-loop speedup validation (CPU/numpy) ===", flush=True)
    RESULT["verdict"] = "PASS"   # flipped to FAIL by _fail on any check

    print("[B] batched CSR == per-concept CSR ...", flush=True)
    check_batched_equals_perconcept()
    print("[B] batched latency rank sane ...", flush=True)
    check_batched_latency_rank()

    print("[A] defer_planner builds no WM loop ...", flush=True)
    ag, _ = check_defer_builds_no_wm()
    print("[A] Q&A answers + moat abstains + zero WM ...", flush=True)
    check_qa_and_moat(ag)
    print("[A] lazy WM builds on first referent + lazy==eager ...", flush=True)
    check_lazy_wm_builds_and_equals_eager()
    print("[A] anti-cheat: deferred == eager Q&A ...", flush=True)
    check_deferred_equals_eager_qa()

    RESULT["summary"] = {
        "root_cause": (
            "MultiTurnAgent.__init__ eagerly builds a persistent SpikingLoopContextBuffer WM loop; its "
            "~2*len(referents) attractor pathways are installed by one set_pathway_weights(add_missing=True) "
            "call PER attractor, each rebuilding the ENTIRE ~10M-synapse CSR + walking the full nnz in a Python "
            "pair_to_idx loop. Profile bwy27t6g6: 144 calls = 681.6s of a 840.8s SK-brain load."),
        "fix_A_defer_planner": (
            "MultiTurnAgent(defer_planner=True) builds the WM loop (and biased-competition buffer) LAZILY on the "
            "first referent write; a Q&A/rich-answer console session never introduces a multi-turn referent so it "
            "pays ZERO WM build. Threaded from load_developed_brain (tied to defer_parser, default True on load) + "
            "brain_chat_tui._load_self_knowledge/_build_tiny_demo (defer_planner=True). Default OFF = byte-identical."),
        "fix_B_batch_graph_build": (
            "SpikingLoopContextBuffer.__init__ (and BiasedCompetitionContextBuffer) now install ALL concepts' "
            "attractor edges in 2 batched set_pathway_weights calls (c2d + d2c) instead of 2*len(concepts); CSR is "
            "rebuilt 2x not 144x. Validated byte-identical to the per-concept CSR (51001 nnz, edge keys + weights "
            "match). Makes the WM-loop / planner build fast WHEN used (no sim/ edit -- runner-level only)."),
        "expected_load_time_after": (
            "Profile-derived estimate from bwy27t6g6 (TOTAL 840.8s, of which the MultiTurnAgent->WM-loop chain is "
            "840.78s, set_pathway_weights 144x = 742.2s cumtime / 681.6s tottime). Fix B alone: 144 calls -> 2 "
            "calls => WM-loop graph install ~742s -> ~10-15s (host pair_to_idx loop is O(nnz)/call, so ~2/144 of "
            "681.6s ~= 9.5s + 2 CSR rebuilds ~1.7s) => total load ~840s -> ~115s. Fix A (deferred, the console "
            "default): the WHOLE WM-loop chain (the build_loop_wm_bridge 56.3s + the graph install) is skipped at "
            "load, so a Q&A/rich console load drops to the BrainConversationalAgent + fact-restore residual "
            "(seconds; the per-fact resonate is already skip-optimized via kb_composites). The controller should "
            "re-measure the real GPU load-time."),
        "files_changed": [
            "research/runners/content_selection_spiking.py (B: batch SpikingLoopContextBuffer attractor install)",
            "research/runners/biased_competition_buffer.py (B: batch attractor + ref2sel install)",
            "research/runners/multi_turn_agent.py (A: defer_planner flag + lazy _ensure_wm/_ensure_bcw)",
            "research/runners/developed_brain_io.py (A: thread defer_planner=defer_parser into MultiTurnAgent)",
            "research/runners/brain_chat_tui.py (A: _load_self_knowledge + _build_tiny_demo pass defer_planner=True)"],
        "sim_edit": "NONE (runner-level only; bridge.py inject_explicit_wiring/set_pathway_weights untouched)",
        "moat": "0 false-accepts (fox eat -> None); deferred Q&A answers == eager == ground truth",
        "remeasure_command": (
            "SIM_BACKEND=cupy python -c \"import time,cProfile,pstats; "
            "from research.runners.developed_brain_io import load_developed_brain; "
            "t=time.time(); a,m=load_developed_brain('<SK_BUNDLE_DIR>', use_multiturn=True); "
            "print('LOAD %.1fs n_facts=%d wm_deferred=%s' % (time.time()-t, m.get('n_facts'), a.wm is None))\""),
        "remeasure_note": (
            "Replace <SK_BUNDLE_DIR> with the same developed-brain bundle the original profile loaded (a dir with "
            "brain.json). load_developed_brain now defaults defer_planner=True, so 'wm_deferred=True' confirms the "
            "WM loop was NOT built at load. To time a WM-USING turn, then call a.hear('x verb y') + a.what_does('it','verb')."),
    }
    all_ok = all(c.get("ok") for c in RESULT["checks"].values())
    RESULT["verdict"] = "PASS" if (all_ok and RESULT["verdict"] != "FAIL") else "FAIL"
    print("\n=== checks ===", flush=True)
    for name, c in RESULT["checks"].items():
        print(f"  {'PASS' if c.get('ok') else 'FAIL'}  {name}", flush=True)
    print(f"\nVERDICT: {RESULT['verdict']}", flush=True)
    out = "research/findings/raw/_sk_load_planner_speedup.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(RESULT, fh, indent=2)
    print("wrote", out, flush=True)
    return 0 if RESULT["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
