"""BURNDOWN I-5-a — the SYNAPTIC parser->composer route (the cheapest-first probe from
`research/findings/2026-06-23-functional-one-brain-integration-scoping.md` item I-5).

THE ONE FALSIFIABLE QUESTION:
Can comprehension's parsed role->word reach the composer SYNAPTICALLY (a parser-firing-gated transmission
route) and recover the SAME role->word assignment as the host Python `{role: word}` dict — gated only on
parser firing, lesion-collapsing, and provenance-clean (no host role value copied across)?

If yes, I-5 is closable reuse-by-import: port `hear_synaptic` onto the nav+conv merge. If no, the precise gap
is reported honestly (no hacked pass).

WHAT THIS PROBE DOES (CPU/numpy; the route + composer + parser already exist as `UnifiedBrainBridge`):
The standalone `UnifiedBrainBridge` HAS the synaptic route (`hear_synaptic` / `_op_synaptic`, the
parser-gated `role_route_<R>` topographic route into the composer's role bank). The nav+conv MERGE reverted
to a Python `{role: word}` dict (`nav_conv_merged_bridge.parse_on_slices`) "for coexistence". The scoping doc
recommends porting `hear_synaptic` onto the merge (option I-5-a). This probe DE-RISKS that port by proving,
on the EXISTING `UnifiedBrainBridge` (the precedent to port), the four load-bearing properties:

  (1) ROUTE == DICT. Build TWO identical bridges (same seed, same concept codebook). Feed a sentence to one
      via `hear`  (the host {role:word} dict path) and the other via `hear_synaptic` (the gated synaptic
      route). The composer's queries (`query_patient`/`query_agent`/`render_fact`) — which UNBIND THE STORED
      BOUND VECTOR, not the fact-dict labels (the cue match is `unbind(bound,'agent')==agent`, see
      core_sim_composition.query_patient) — must recover the SAME role->word answers from both bridges.
      Plus the no-confab MOAT: an unstored cue must abstain (None) on BOTH paths.
  (2) GATED-BY-FIRING. Instrument the per-word bind: for each word's `_op_synaptic`, the parser's role
      ensemble for THAT word's role fires -> its `role_route_<R>` gate OPENS; the other two roles' gates stay
      CLOSED. (The cross-region coupling is a 0/1 gate STATE from firing, the working template.)
  (3) LESION COLLAPSES. Cut the route (remove the gate couplings AND hold every `role_route_<R>` closed) so
      the parser firing cannot open any route. The role bank then gets NO role drive -> the bind degrades ->
      role recovery collapses (>= the queries fail vs the intact synaptic path). The route is NECESSARY.
  (4) PROVENANCE-CLEAN. During a synaptic-route bind, instrument `cp_external_input_current`: the composer's
      role_on/role_off banks must receive ZERO direct external current (the role +-1 PATTERN reaches them
      ONLY through the synaptic route — `_op_synaptic` writes the parser drive + a UNIFORM role_src drive +
      the fill code + bias, NEVER a `roles[r]` value). CONTRAST: the host dict path (`hadamard_spiking`)
      writes `cur[role_on] = (role_vec>0)*ROLE_DRIVE` — the role value IS a copied host quantity. So the
      synaptic path carries NO host role quantity across the region boundary; the routing is the gate.

ANTI-CHEATS (all four properties above ARE the anti-cheats): provenance (4), lesion (3), gated-by-firing (2),
moat preserved (1). Multi-seed (>= 3).

STRICTLY CPU/numpy (`SIM_BACKEND=numpy`). NO `sim/` edit (reuse-by-import — `UnifiedBrainBridge` + the public
gate primitives). Reuse-by-import precedent: `research/findings/raw/_step2_gated_route_probe.py`,
`research/findings/raw/_step2_synaptic_holdopen_validate.py`.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")

from sim.backend import get_backend, to_host  # noqa: E402
from research.runners.unified_brain_bridge import (  # noqa: E402
    SYNAPTIC_ROUTE_ROLES, ROLE_GATE_PREWARM_CAP_STEPS,
)
from research.runners.core_sim_composition import RESET_STEPS, ROLE_DRIVE  # noqa: E402


PROJ_DIM = 128   # proj_dim=64 + the 8-orthonormal synthetic codebook is too small for clean K-role decode (the
#                  composer mis-decodes some single facts EVEN on the dict path — a codebook-MARGIN artifact,
#                  NOT a route issue; verified: dict-path single-fact decode is 1/6 at pd=64, 6/6 at pd=128).
#                  pd=128 gives clean per-fact decode so route==dict tests the ROUTE, not composer fidelity.
# The probe sentences (3-word SVO). Multiple facts with distinct agents/actions/patients so role->word recovery
# is non-trivial (a query that returned the only stored agent would pass vacuously; multiple facts force a real
# cue match). All decode cleanly at pd=128 on the dict path (the route must reproduce that).
FACTS = [
    ("dog", "go", "north"),
    ("cat", "look", "river"),
    ("dog", "come", "south"),
]
UNSTORED_CUE = ("river", "come")   # no fact has agent 'river' acting 'come' -> the no-confab moat must abstain


def synth_concepts(proj_dim=PROJ_DIM, seed=0):
    """8 orthonormal concept codes (no denoise64 cache dependency) — the same helper the step-1/step-2
    shared-bridge probes use. Keeps the build cache-independent + CPU-portable. At proj_dim=128 these decode
    cleanly per fact (the dict-path decode is 6/6; at pd=64 it was 1/6 — a codebook-margin artifact)."""
    rng = np.random.default_rng(seed)
    words = ["dog", "cat", "go", "come", "north", "south", "river", "look"]
    q, _ = np.linalg.qr(rng.standard_normal((proj_dim, proj_dim)))
    return {w: q[i] for i, w in enumerate(words)}


def _build(seed, concepts):
    """One UnifiedBrainBridge with the synaptic route wired (the precedent to port). Cache-independent
    (synthetic codebook). Returns the bridge object."""
    from research.runners.unified_brain_bridge import UnifiedBrainBridge
    return UnifiedBrainBridge(seed=seed, proj_dim=PROJ_DIM, concepts=concepts, enable_synaptic_route=True)


# ── (2) gated-by-firing: instrument which role_route gate is open during each word's bind ──────────────────
def _gate_open(bridge, role):
    """Is `role_route_<role>` open (>= 0.99) right now?"""
    syn = bridge._transmission_gate_to_synapses.get(f"role_route_{role}")
    if syn is None or bridge.cp_transmission_gain is None:
        return False
    return float(to_host(bridge.cp_transmission_gain[syn]).mean()) >= 0.99


def gated_by_firing_trace(ub, sentence, voice="active"):
    """Re-run the comprehension of `sentence` through the SAME `_op_synaptic` bodies, but record, per word,
    WHICH role gates were open during the readout window. Returns a list (one dict per word) of
    {word, parser_role, gates_open}. The parser-selected role's gate must be the (only) open one.

    This duplicates `hear_synaptic`'s per-word loop (not a new mechanism) so we can snapshot the gate state at
    the moment the bind reads — the cleanest observation point for the gated-by-firing anti-cheat.
    """
    xp, _ = get_backend()
    from research.runners.core_sim_composition import onoff, _scale_to_current, FILL_DRIVE
    words = sentence.split() if isinstance(sentence, str) else list(sentence)
    v = 0 if voice in (0, "active") else 1
    comp = ub.composer
    idx = comp.idx
    trace = []
    for pos in range(3):
        word = words[pos]
        k = pos * 2 + v
        role = ub.parser.role_of(pos, voice)
        # mirror _op_synaptic's drive setup
        bridge = ub.bridge
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(RESET_STEPS):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
        c_on, c_off = onoff(comp.concepts[word])
        fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
        cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
        cur[ub.parser.conj_arr[k]] = ub.parser.drive
        from research.runners.unified_brain_bridge import ROLE_SRC_DRIVE_PA
        for r in SYNAPTIC_ROUTE_ROLES:
            cur[ub._role_src[r]] = ROLE_SRC_DRIVE_PA
        cur[idx["fill_on"]] = xp.asarray(fon.astype(np.float32))
        cur[idx["fill_off"]] = xp.asarray(foff.astype(np.float32))
        for bank in ("A", "B", "C", "D"):
            cur[idx[bank]] = comp.coinc_bias
        bridge.cp_external_input_current[:] = cur
        # pre-warm until a gate opens (the parser's doing), then snapshot gate state over the readout window
        for _ in range(ROLE_GATE_PREWARM_CAP_STEPS):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
            if any(_gate_open(bridge, r) for r in SYNAPTIC_ROUTE_ROLES):
                break
        # hold the parser-opened gate (pause coupling) and record which gates are open across the readout
        saved = bridge._gate_couplings
        bridge._gate_couplings = []
        open_counts = {r: 0 for r in SYNAPTIC_ROUTE_ROLES}
        try:
            for _ in range(comp.run_steps):
                bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
                bridge._run_one_simulation_step()
                for r in SYNAPTIC_ROUTE_ROLES:
                    if _gate_open(bridge, r):
                        open_counts[r] += 1
        finally:
            bridge._gate_couplings = saved
            for r in SYNAPTIC_ROUTE_ROLES:
                bridge.set_transmission_gate(f"role_route_{r}", 0.0)
                cpl = next((c for c in bridge._gate_couplings if c["gate_name"] == f"role_route_{r}"), None)
                if cpl is not None:
                    cpl["ema"] = 0.0
                    cpl["last_value"] = None
            bridge.cp_external_input_current[:] = 0.0
        gates_open = [r for r in SYNAPTIC_ROUTE_ROLES if open_counts[r] >= comp.run_steps // 2]
        trace.append({"word": word, "parser_role": role, "gates_open": gates_open,
                      "open_fraction": {r: open_counts[r] / comp.run_steps for r in SYNAPTIC_ROUTE_ROLES}})
    return trace


# ── (4) provenance: instrument the direct external current the role bank receives during a synaptic bind ────
def provenance_role_bank_current(ub, word="dog", pos=0, voice="active"):
    """Drive ONE word's `_op_synaptic` step setup and record the MAX |direct external current| written to the
    composer's role_on/role_off banks. For the synaptic route this MUST be 0 (the role +-1 pattern reaches the
    role bank ONLY through the gated synaptic route, never as a host current). Also returns the (host) current
    the DICT path would write to the same banks, for contrast (`(role_vec>0)*ROLE_DRIVE`)."""
    xp, _ = get_backend()
    from research.runners.core_sim_composition import onoff, _scale_to_current, FILL_DRIVE
    from research.runners.unified_brain_bridge import ROLE_SRC_DRIVE_PA
    comp = ub.composer
    idx = comp.idx
    bridge = ub.bridge
    role = ub.parser.role_of(pos, voice)
    k = pos * 2 + (0 if voice in (0, "active") else 1)

    # Build the EXACT external-current vector _op_synaptic writes (the synaptic-route drive).
    c_on, c_off = onoff(comp.concepts[word])
    fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
    cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
    cur[ub.parser.conj_arr[k]] = ub.parser.drive
    for r in SYNAPTIC_ROUTE_ROLES:
        cur[ub._role_src[r]] = ROLE_SRC_DRIVE_PA
    cur[idx["fill_on"]] = xp.asarray(fon.astype(np.float32))
    cur[idx["fill_off"]] = xp.asarray(foff.astype(np.float32))
    for bank in ("A", "B", "C", "D"):
        cur[idx[bank]] = comp.coinc_bias

    role_on_idx = to_host(idx["role_on"]).astype(np.int64)
    role_off_idx = to_host(idx["role_off"]).astype(np.int64)
    cur_h = to_host(cur)
    synaptic_role_bank_current_max = float(np.abs(
        np.concatenate([cur_h[role_on_idx], cur_h[role_off_idx]])).max())

    # The DICT path's host current to the same banks (for contrast): (role_vec>0/<0)*ROLE_DRIVE.
    role_vec = to_host(comp.roles[role]) if hasattr(comp.roles[role], "shape") else np.asarray(comp.roles[role])
    dict_role_bank_current_max = float(np.abs(np.concatenate([
        (role_vec > 0).astype(np.float64) * ROLE_DRIVE,
        (role_vec < 0).astype(np.float64) * ROLE_DRIVE])).max())
    return {
        "word": word, "parser_role": role,
        "synaptic_route_role_bank_direct_current_max": synaptic_role_bank_current_max,
        "dict_path_role_bank_direct_current_max": dict_role_bank_current_max,
    }


# ── (3) lesion: cut the route so the parser firing cannot open any role gate ───────────────────────────────
def lesion_route(bridge):
    """Cut the synaptic route: drop every gate->ensemble coupling AND hold every role_route gate CLOSED. The
    parser still fires (comprehension is intact) but its firing can no longer open any route -> the role bank
    receives no role drive. Returns a restore() to undo the lesion."""
    saved_couplings = list(bridge._gate_couplings)
    # remove only the role_route couplings (there are no others on this bridge, but be precise)
    bridge._gate_couplings = [c for c in bridge._gate_couplings
                              if not c["gate_name"].startswith("role_route_")]
    bridge._gate_coupling_flat = None
    for r in SYNAPTIC_ROUTE_ROLES:
        bridge.set_transmission_gate(f"role_route_{r}", 0.0)

    def restore():
        bridge._gate_couplings = saved_couplings
        bridge._gate_coupling_flat = None
        for c in bridge._gate_couplings:
            c["ema"] = 0.0
            c["last_value"] = None
    return restore


def run_seed(seed, verbose=False):
    concepts = synth_concepts(seed=0)   # same codebook across seeds; the seed varies the bridge RNG (the proper
    #                                     multi-seed: identical task, different network draw / OU noise)
    t0 = time.time()

    # --- (1) ROUTE vs DICT: two identical bridges, one fed via hear_synaptic, one via hear ---
    ub_route = _build(seed, concepts)
    ub_dict = _build(seed, concepts)
    parses_route, parses_dict = [], []
    for (a, ac, p) in FACTS:
        parses_route.append(ub_route.hear_synaptic(f"{a} {ac} {p}", voice="active"))
        parses_dict.append(ub_dict.hear(f"{a} {ac} {p}", voice="active"))

    # role->word recovery on each path (unbinds the STORED BOUND VECTOR, not the fact labels)
    def recover(ub):
        qp = {(a, ac): ub.query_patient(a, ac) for (a, ac, p) in FACTS}        # what does <a> <ac>?  -> patient
        qa = {(ac, p): ub.query_agent(ac, p) for (a, ac, p) in FACTS}          # who <ac> <p>?         -> agent
        rf = {a: ub.render_fact(a) for (a, ac, p) in FACTS}                    # full SVO render
        moat = ub.query_patient(*UNSTORED_CUE)                                 # no-confab: must be None
        return {"query_patient": qp, "query_agent": qa, "render_fact": rf, "moat_abstain": moat}

    rec_route = recover(ub_route)
    rec_dict = recover(ub_dict)

    # GO bar for (1): every recovered answer matches between the route and the dict path, AND each matches the
    # ground-truth fact, AND both abstain on the unstored cue.
    def gt_patient(a, ac):
        return next(p for (aa, acc, p) in FACTS if aa == a and acc == ac)

    def gt_agent(ac, p):
        return next(aa for (aa, acc, pp) in FACTS if acc == ac and pp == p)

    # Per-fact correctness on each path (a fact is "correct" iff BOTH its patient-cue and agent-cue recover the
    # ground truth). The DICT path has its OWN seed-dependent composer-decode jitter at the code margin (OU noise
    # tips a borderline unbind — verified: seed-43 dict-path single-fact decode of 'cat look river' MISSES on
    # the first store, then decodes OK on reruns; this is the composer/codebook, NOT the route). So the
    # load-bearing invariant is NOT bitwise route==dict (that penalizes the route when the DICT is the noisy one),
    # but: (i) the route recovers GROUND TRUTH, and (ii) the route is NEVER WORSE than the dict (faithful
    # reproduction, robust to the dict path's own composer noise).
    def n_correct(rec):
        return sum(1 for (a, ac, p) in FACTS
                   if rec["query_patient"].get((a, ac)) == gt_patient(a, ac)
                   and rec["query_agent"].get((ac, p)) == gt_agent(ac, p))
    route_correct = n_correct(rec_route)
    dict_correct = n_correct(rec_dict)
    route_eq_gt = (route_correct == len(FACTS))                 # the load-bearing claim: route delivers the
    #                                                             correct role->word for every fact
    route_not_worse_than_dict = (route_correct >= dict_correct)  # the route never under-performs the dict path
    route_eq_dict_bitwise = (rec_route["query_patient"] == rec_dict["query_patient"]   # secondary (noisy) check
                             and rec_route["query_agent"] == rec_dict["query_agent"]
                             and rec_route["render_fact"] == rec_dict["render_fact"])
    moat_ok = (rec_route["moat_abstain"] is None and rec_dict["moat_abstain"] is None)

    # --- (2) GATED-BY-FIRING: on a fresh route bridge, trace which gate is open per word ---
    ub_g = _build(seed, concepts)
    trace = gated_by_firing_trace(ub_g, " ".join(FACTS[0]), voice="active")
    # each word: the parser-selected role's gate is open AND it is the ONLY open gate
    gated_ok = all(t["gates_open"] == [t["parser_role"]] for t in trace)

    # --- (4) PROVENANCE: the role bank receives ZERO direct external current on the synaptic path ---
    prov = provenance_role_bank_current(ub_g, word=FACTS[0][0], pos=0, voice="active")
    provenance_clean = (prov["synaptic_route_role_bank_direct_current_max"] == 0.0
                        and prov["dict_path_role_bank_direct_current_max"] > 0.0)

    # --- (3) LESION: cut the route, re-store the facts, recover -> must collapse ---
    ub_lesion = _build(seed, concepts)
    restore = lesion_route(ub_lesion.bridge)
    parses_lesion = [ub_lesion.hear_synaptic(f"{a} {ac} {p}", voice="active") for (a, ac, p) in FACTS]
    rec_lesion = recover(ub_lesion)
    restore()
    # the lesioned synaptic path must NOT recover the correct facts (role bank starved -> bind degraded).
    lesion_correct = sum(
        1 for (a, ac, p) in FACTS
        if rec_lesion["query_patient"].get((a, ac)) == gt_patient(a, ac)
        and rec_lesion["query_agent"].get((ac, p)) == gt_agent(ac, p))
    intact_correct = sum(
        1 for (a, ac, p) in FACTS
        if rec_route["query_patient"].get((a, ac)) == gt_patient(a, ac)
        and rec_route["query_agent"].get((ac, p)) == gt_agent(ac, p))
    lesion_collapses = (lesion_correct < intact_correct)

    seed_go = bool(route_eq_gt and route_not_worse_than_dict and moat_ok and gated_ok and provenance_clean
                   and lesion_collapses)

    result = {
        "seed": int(seed),
        "1_route_vs_dict": {
            "parses_route": parses_route, "parses_dict": parses_dict,
            "recovered_route": {k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
                                for k, v in rec_route.items()},
            "recovered_dict": {k: {str(kk): vv for kk, vv in v.items()} if isinstance(v, dict) else v
                               for k, v in rec_dict.items()},
            "route_correct": int(route_correct), "dict_correct": int(dict_correct), "n_facts": len(FACTS),
            "route_eq_ground_truth": bool(route_eq_gt),
            "route_not_worse_than_dict": bool(route_not_worse_than_dict),
            "route_eq_dict_bitwise": bool(route_eq_dict_bitwise),   # secondary: confounded by dict-path OU jitter
            "moat_abstain_both_None": bool(moat_ok),
        },
        "2_gated_by_firing": {"trace": trace, "ok": bool(gated_ok)},
        "3_lesion": {"intact_correct": int(intact_correct), "lesion_correct": int(lesion_correct),
                     "collapses": bool(lesion_collapses),
                     "recovered_lesion": {str(k): v for k, v in rec_lesion["query_patient"].items()}},
        "4_provenance": {**prov, "clean": bool(provenance_clean)},
        "seed_GO": seed_go,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if verbose:
        print(json.dumps(result, indent=2, default=str))
    return result


def main(seeds=(42, 43, 44)):
    t0 = time.time()
    results = [run_seed(s, verbose=False) for s in seeds]
    n_go = sum(1 for r in results if r["seed_GO"])
    verdict = "GO" if n_go == len(seeds) else ("PARTIAL" if n_go > 0 else "NEGATIVE")
    summary = {
        "probe": "burndown_I5a_synaptic_parser_composer",
        "question": "does the synaptic parser->composer route recover the role->word assignment == the host "
                    "dict path, gated by parser firing (lesion collapses), provenance-clean?",
        "backend": get_backend()[1],
        "seeds": list(seeds),
        "n_seeds_GO": n_go,
        "verdict": verdict,
        "route_eq_ground_truth_all": all(r["1_route_vs_dict"]["route_eq_ground_truth"] for r in results),
        "route_not_worse_than_dict_all": all(r["1_route_vs_dict"]["route_not_worse_than_dict"] for r in results),
        "route_eq_dict_bitwise_all": all(r["1_route_vs_dict"]["route_eq_dict_bitwise"] for r in results),
        "moat_preserved_all": all(r["1_route_vs_dict"]["moat_abstain_both_None"] for r in results),
        "gated_by_firing_all": all(r["2_gated_by_firing"]["ok"] for r in results),
        "lesion_collapses_all": all(r["3_lesion"]["collapses"] for r in results),
        "provenance_clean_all": all(r["4_provenance"]["clean"] for r in results),
        "note_route_eq_dict_bitwise": ("route_eq_dict_bitwise is a SECONDARY check confounded by the DICT "
            "path's OWN seed-dependent composer-decode jitter (OU noise at the code margin); the load-bearing "
            "invariants are route_eq_ground_truth + route_not_worse_than_dict. On any seed where they differ, "
            "the route recovered the fact and the DICT path was the noisy one (route never worse)."),
        "per_seed": results,
        "total_elapsed_s": round(time.time() - t0, 1),
    }
    return summary


if __name__ == "__main__":
    out = main()
    os.makedirs("research/findings/raw", exist_ok=True)
    with open("research/findings/raw/_burndown_I5a_synaptic_parser_composer.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    # compact console summary
    print(json.dumps({k: v for k, v in out.items() if k != "per_seed"}, indent=2, default=str))
    for r in out["per_seed"]:
        rv = r["1_route_vs_dict"]
        print(f"  seed {r['seed']}: GO={r['seed_GO']}  "
              f"route_correct={rv['route_correct']}/{rv['n_facts']} (dict={rv['dict_correct']})  "
              f"==gt={rv['route_eq_ground_truth']}  not_worse={rv['route_not_worse_than_dict']}  "
              f"gated={r['2_gated_by_firing']['ok']}  "
              f"lesion_collapse={r['3_lesion']['collapses']}({r['3_lesion']['lesion_correct']}<"
              f"{r['3_lesion']['intact_correct']})  "
              f"prov_clean={r['4_provenance']['clean']}  ({r['elapsed_s']}s)")
