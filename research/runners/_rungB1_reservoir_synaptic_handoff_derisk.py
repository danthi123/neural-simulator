"""RUNG B-1 (the FUNCTIONAL one-brain, cheapest-first) -- the reservoir's LEARNED role output drives the composer's
bind through the I5a SYNAPTIC `role_route_<R>` gates, REPLACING the host {role:word} dict.

CONTEXT. The EMERGE-92..95 ladder consolidated the conversational turn onto ONE spiking bridge (the SUBSTRATE bar),
but the comprehension->composition hand-off is still a host Python dict: the reservoir parses roles in host, and
those roles are handed to the composer as a `{role: word}` dict. RUNG B is the FUNCTIONAL bar -- make that hand-off
SYNAPTIC. The synaptic parser->composer route already exists + is validated (`_burndown_I5a_...`, all 4 anti-cheats
GO): a parser-firing-gated `role_route_<R>` topographic route carries role R's +-1 pattern into the composer's role
bank, provenance-clean. In I5a the role per word comes from the HAND parser's positional rule (`parser.role_of(pos)`).

THE ONE NEW PIECE (this runner): the role per word comes from the RESERVOIR's LEARNED form->role map
(`argmax(f @ Ws[k])`, EMERGE-78/88) instead of the parser's positional rule. Because `_op_synaptic(k)` fires the
role ensemble selected by conjunction index `k`, we map the reservoir's chosen role -> the conjunction that fires it
(`role2k`), and reuse the ENTIRE I5a route + all four anti-cheats UNCHANGED. NO edit to any shared file (reuse-by-
import: the reservoir comprehender from EMERGE-88, the synaptic route + instruments from I5a).

  reservoir final state f --(Ws[k], the learned read-out)--> role r per content word
      --> fire conj[role2k[r]] --> gate role_route_<r> opens --> composer role bank gets role r's +-1 pattern
      --> the word binds with role r, provenance-clean (the role pattern crosses the region boundary only via
          the gated synapses, never as a host {role:word} current).

ANTI-CHEATS (multi-seed):
  (1) ROUTE RECOVERS THE FACT. Comprehend -> synaptic-bind -> store -> query_patient/query_agent recover the
      ground-truth (agent, action, patient). Compared to the host-dict store of the SAME reservoir-parsed fact:
      the route is NEVER WORSE than the dict (robust to the composer's own OU decode jitter, per I5a).
  (2) GATED-BY-FIRING. Per content word, ONLY the reservoir-selected role's gate opens (the other two closed).
  (3) ROUTE-LESION COLLAPSES. Cut the synaptic route (I5a `lesion_route`) -> the gate cannot open -> the role
      bank is starved -> recall collapses. The synaptic route is NECESSARY for the hand-off.
  (4) PROVENANCE-CLEAN. The composer role bank receives ZERO direct external current on a synaptic bind (I5a
      `provenance_role_bank_current`); the dict path writes a host role current (contrast). No host role quantity
      crosses the region boundary.
  (5) RESERVOIR-NECESSITY (comprehension-lesion). Lesion the reservoir's closed-class identity (EMERGE-88
      `lesion=True`) -> the roles collapse -> wrong facts bound -> recall collapses. Proves the RESERVOIR's learned
      comprehension (not the parser's positional rule) is what selects the roles routed synaptically.
  (6) MOAT. An (agent, action) never stored -> the composer abstains (None) -> no confabulation.

STRICTLY CPU/numpy (SIM_BACKEND=numpy). NO `sim/` edit. Multi-seed: ONE shared corpus/task (built once), per-seed
reservoir + bridge RNG (I5a's multi-seed pattern -- identical task, different network draw / OU noise).

Run:  SIM_BACKEND=numpy python -m research.runners._rungB1_reservoir_synaptic_handoff_derisk \
          --seeds 42 43 44 --json research/findings/raw/_rungB1_reservoir_synaptic_handoff.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

from sim.backend import get_backend, to_host  # noqa: E402
import research.runners._emerge62_discover_function_words_derisk as m62  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    _content_pools, _ROLES, _gen, _TRAIN_KINDS, _N_TRAIN_PER_CONSTRUCTION,
)
from research.runners._emerge88_reservoir_comprehends_composer_answers_derisk import (  # noqa: E402
    ReservoirComprehender, _ROLE2FIELD, _build_test_facts,
)
from research.runners.unified_brain_bridge import UnifiedBrainBridge, SYNAPTIC_ROUTE_ROLES  # noqa: E402
from research.runners.core_sim_composition import onoff, _scale_to_current, FILL_DRIVE  # noqa: E402
# reuse I5a's synaptic-route anti-cheat instruments UNCHANGED
from research.runners._burndown_I5a_synaptic_parser_composer import (  # noqa: E402
    _gate_open, lesion_route, provenance_role_bank_current,
)

PROJ_DIM = 128    # I5a: pd=128 gives clean per-fact composer decode (pd=64 is a codebook-margin artifact)
N_TEST = 6        # small test set -> small composer codebook (clean decode) + small bridge (weak-CPU tractable)


def _orthonormal_concepts(vocab, proj_dim, seed=0):
    """Orthonormal concept codes for exactly the test vocab (cache-independent; clean per-fact decode at pd=128)."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((proj_dim, proj_dim)))
    return {w: q[i] for i, w in enumerate(sorted(vocab))}


def _reservoir_roles(comp, tokens, lesion=False):
    """Per content word (left-to-right): the reservoir's composer-role (agent/action/patient) + the surface word,
    from the whole-sentence final state (Dominey-Hinaut). `lesion=True` collapses the closed-class identity (the
    reservoir-necessity control) -> the role read-out degrades."""
    f = np.concatenate([comp.res.final_state(comp.enc.encode(tokens, lesion=lesion)), [1.0]])
    content = [t for t, w in enumerate(tokens) if w not in comp.closed]
    pairs = []
    for k, t in enumerate(content):
        if comp.Ws is None or k not in comp.Ws:
            continue
        role = _ROLES[int(np.argmax(f @ comp.Ws[k]))]
        field = _ROLE2FIELD.get(role)     # AGENT->agent, PREDICATE->action, THEME->patient (GOAL/LOCATION->None)
        if field is not None:
            pairs.append((tokens[t], field))
    return pairs


def _bind_reservoir_fact(ub, role2k, pairs):
    """Bind each (word, role) via the SYNAPTIC route: fire conj[role2k[role]] so gate role_route_<role> opens and
    routes that role's +-1 pattern into the composer role bank. Role SELECTION is the reservoir's; hand-off is
    synaptic. Appends the fact + the synaptically-bound vector to the composer kb (as `hear_synaptic` does)."""
    comp = ub.composer
    bound_on = np.zeros(comp.D); bound_off = np.zeros(comp.D)
    fact = {}
    for word, role in pairs:
        if role not in role2k or role in fact:      # first-wins per role (a well-formed SVO fills each once)
            continue
        c_on, c_off = onoff(comp.concepts[word])
        fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
        bon, boff = ub._op_synaptic(role2k[role], fon, foff)
        bound_on += bon; bound_off += boff
        fact[role] = word
    # Only a COMPLETE comprehension is storable: query_patient reads fact["patient"], so a fact missing a role
    # (a collapsed/mislabeled comprehension, e.g. under the reservoir-lesion control) is dropped -> it counts as a
    # recall miss (the honest collapse) rather than a malformed kb entry.
    if {"agent", "action", "patient"} <= set(fact):
        comp.kb.append((fact, onoff(bound_on - bound_off)))
        return fact
    return None


def _recall(ub, test):
    """who/what recall over the composer kb: fraction of query_patient + query_agent that return the ground truth."""
    hp = sum(int(ub.query_patient(s, v3) == o) for _t, s, v3, o in test)
    ha = sum(int(ub.query_agent(v3, o) == s) for _t, s, v3, o in test)
    return hp, ha


def _gated_by_reservoir_trace(ub, comp, role2k, tokens):
    """For each content word, drive the RESERVOIR-selected role's conjunction and record which role gate(s) open
    over the readout window -> the reservoir-selected role's gate must be the (only) one open."""
    xp, _ = get_backend()
    from research.runners.core_sim_composition import RESET_STEPS
    from research.runners.unified_brain_bridge import ROLE_SRC_DRIVE_PA, ROLE_GATE_PREWARM_CAP_STEPS
    bridge = ub.bridge; comp_ = ub.composer; idx = comp_.idx
    pairs = _reservoir_roles(comp, tokens)
    trace = []
    for word, role in pairs:
        if role not in role2k:
            continue
        k = role2k[role]
        c_on, c_off = onoff(comp_.concepts[word]); fon, foff = _scale_to_current(c_on, c_off, FILL_DRIVE)
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(RESET_STEPS):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
        cur = xp.zeros(bridge.core_config.num_neurons, dtype=xp.float32)
        cur[ub.parser.conj_arr[k]] = ub.parser.drive
        for r in SYNAPTIC_ROUTE_ROLES:
            cur[ub._role_src[r]] = ROLE_SRC_DRIVE_PA
        cur[idx["fill_on"]] = xp.asarray(fon.astype(np.float32))
        cur[idx["fill_off"]] = xp.asarray(foff.astype(np.float32))
        for bank in ("A", "B", "C", "D"):
            cur[idx[bank]] = comp_.coinc_bias
        bridge.cp_external_input_current[:] = cur
        for _ in range(ROLE_GATE_PREWARM_CAP_STEPS):
            bridge.runtime_state.current_time_ms += bridge.core_config.dt_ms
            bridge._run_one_simulation_step()
            if any(_gate_open(bridge, r) for r in SYNAPTIC_ROUTE_ROLES):
                break
        saved = bridge._gate_couplings; bridge._gate_couplings = []
        open_counts = {r: 0 for r in SYNAPTIC_ROUTE_ROLES}
        try:
            for _ in range(comp_.run_steps):
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
                    cpl["ema"] = 0.0; cpl["last_value"] = None
            bridge.cp_external_input_current[:] = 0.0
        gates_open = [r for r in SYNAPTIC_ROUTE_ROLES if open_counts[r] >= comp_.run_steps // 2]
        trace.append({"word": word, "reservoir_role": role, "gates_open": gates_open})
    return trace


def setup_corpus(seed=42):
    """Build the shared corpus/task ONCE: discover the closed class, content pools, the fixed test facts, and the
    fixed composer codebook. Reused across seeds (the multi-seed varies only the reservoir + bridge RNG)."""
    stream = m62.build_stream(seed, n_sentences=6000)
    words, freq, cover, _c = m62.compute_stats(stream)
    discovered, *_ = m62.discover_closed_class(words, freq, cover)
    subj, verb, obj = _content_pools(discovered)
    test, _seen, _trng = _build_test_facts(seed, subj, verb, obj, n=N_TEST)
    vocab = sorted({w for _toks, s, v3, o in test for w in (s, v3, o)})
    concepts = _orthonormal_concepts(vocab, PROJ_DIM, seed=0)
    return {"discovered": discovered, "subj": subj, "verb": verb, "obj": obj,
            "test": test, "vocab": vocab, "concepts": concepts}


def run_seed(seed, corpus):
    t0 = time.time()
    discovered, subj, verb, obj = corpus["discovered"], corpus["subj"], corpus["verb"], corpus["obj"]
    test, concepts = corpus["test"], corpus["concepts"]
    rng = np.random.default_rng(seed * 101 + 5)

    # the reservoir comprehender (per-seed network + read-out fit)
    comp = ReservoirComprehender(seed, discovered)
    comp.fit(_gen(_TRAIN_KINDS, _N_TRAIN_PER_CONSTRUCTION, rng, subj, verb, obj))

    # PARSE: the reservoir's role assignment for each transitive sentence
    parse_hit = 0
    for toks, s, v3, o in test:
        d = {r: w for w, r in _reservoir_roles(comp, toks)}
        parse_hit += int(d.get("agent") == s and d.get("action") == v3 and d.get("patient") == o)
    parse_acc = parse_hit / len(test)

    def new_bridge():
        return UnifiedBrainBridge(seed=seed, proj_dim=PROJ_DIM, concepts=concepts, enable_synaptic_route=True)

    # (1) ROUTE: reservoir roles -> synaptic bind -> store -> recall
    ub = new_bridge()
    role2k = {ub.parser.role_of(pos, "active"): pos * 2 for pos in range(3)}
    for toks, s, v3, o in test:
        _bind_reservoir_fact(ub, role2k, _reservoir_roles(comp, toks))
    hp, ha = _recall(ub, test)
    route_correct = hp + ha
    # no-confab MOAT: (agent, action) never stored -> abstain
    stored = {(s, v3) for _t, s, v3, _o in test}
    fa = tot = 0; mg = 0
    trng = np.random.default_rng(seed * 733 + 999)
    while tot < 30 and mg < 3000:
        mg += 1
        s = str(trng.choice(subj)); v3q = str(trng.choice(verb)) + "s"
        if (s, v3q) in stored:
            continue
        tot += 1; fa += int(ub.query_patient(s, v3q) is not None)
    moat_fa = fa / max(1, tot)

    # (1b) DICT path (same reservoir-parsed facts, host {role:word} store) -> route must be NOT WORSE
    ub_d = new_bridge()
    for toks, s, v3, o in test:
        fact = {r: w for w, r in _reservoir_roles(comp, toks)}
        if {"agent", "action", "patient"} <= set(fact):
            ub_d.store(fact["agent"], fact["action"], fact["patient"])
    dp, da = _recall(ub_d, test)
    dict_correct = dp + da

    # (2) GATED-BY-FIRING
    ub_g = new_bridge()
    role2k_g = {ub_g.parser.role_of(pos, "active"): pos * 2 for pos in range(3)}
    trace = _gated_by_reservoir_trace(ub_g, comp, role2k_g, test[0][0])
    gated_ok = all(t["gates_open"] == [t["reservoir_role"]] for t in trace)

    # (4) PROVENANCE-CLEAN (reuse I5a instrument on a content word)
    prov = provenance_role_bank_current(ub_g, word=corpus["vocab"][0], pos=0, voice="active")
    provenance_clean = (prov["synaptic_route_role_bank_direct_current_max"] == 0.0
                        and prov["dict_path_role_bank_direct_current_max"] > 0.0)

    # (3) ROUTE-LESION: cut the synaptic route -> recall collapses
    ub_l = new_bridge()
    role2k_l = {ub_l.parser.role_of(pos, "active"): pos * 2 for pos in range(3)}
    restore = lesion_route(ub_l.bridge)
    for toks, s, v3, o in test:
        _bind_reservoir_fact(ub_l, role2k_l, _reservoir_roles(comp, toks))
    lp, la = _recall(ub_l, test)
    route_lesion_correct = lp + la
    restore()
    route_lesion_collapses = route_lesion_correct < route_correct

    # (5) RESERVOIR-NECESSITY: lesion the reservoir's comprehension -> wrong roles -> recall collapses
    ub_r = new_bridge()
    role2k_r = {ub_r.parser.role_of(pos, "active"): pos * 2 for pos in range(3)}
    for toks, s, v3, o in test:
        _bind_reservoir_fact(ub_r, role2k_r, _reservoir_roles(comp, toks, lesion=True))
    rp, ra = _recall(ub_r, test)
    res_lesion_correct = rp + ra
    res_lesion_collapses = res_lesion_correct < route_correct

    n_q = 2 * len(test)
    seed_go = bool(
        route_correct >= 0.80 * n_q
        and route_correct >= dict_correct               # route never worse than the host-dict path
        and moat_fa <= 0.05
        and gated_ok and provenance_clean
        and route_lesion_collapses and res_lesion_collapses
    )
    return {
        "seed": int(seed), "parse_acc": parse_acc,
        "route_correct": int(route_correct), "dict_correct": int(dict_correct), "n_queries": n_q,
        "route_recall": route_correct / n_q,
        "route_not_worse_than_dict": bool(route_correct >= dict_correct),
        "moat_false_accept": moat_fa,
        "gated_by_firing": bool(gated_ok), "gate_trace": trace,
        "provenance": {**prov, "clean": bool(provenance_clean)},
        "route_lesion_correct": int(route_lesion_correct), "route_lesion_collapses": bool(route_lesion_collapses),
        "res_lesion_correct": int(res_lesion_correct), "res_lesion_collapses": bool(res_lesion_collapses),
        "seed_GO": seed_go, "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    t0 = time.time()
    corpus = setup_corpus(seed=42)
    print(f"[rungB1] corpus: {len(corpus['test'])} facts, vocab {len(corpus['vocab'])}", flush=True)
    rows = []
    for s in args.seeds:
        d = run_seed(s, corpus)
        rows.append(d)
        print(f"[seed {s}] GO={d['seed_GO']} parse {d['parse_acc']:.2f} | route {d['route_correct']}/{d['n_queries']}"
              f" (dict {d['dict_correct']}) | moat-FA {d['moat_false_accept']:.2f} | gated {d['gated_by_firing']}"
              f" | prov {d['provenance']['clean']} | route-lesion {d['route_lesion_correct']}<{d['route_correct']}"
              f"={d['route_lesion_collapses']} | res-lesion {d['res_lesion_correct']}<{d['route_correct']}"
              f"={d['res_lesion_collapses']} ({d['elapsed_s']}s)", flush=True)

    n_go = sum(r["seed_GO"] for r in rows)
    agg = {
        "n_seeds": len(rows), "n_seeds_GO": int(n_go),
        "verdict": "GO" if n_go == len(rows) else ("PARTIAL" if n_go else "NO-GO"),
        "route_not_worse_than_dict_all": all(r["route_not_worse_than_dict"] for r in rows),
        "gated_all": all(r["gated_by_firing"] for r in rows),
        "provenance_clean_all": all(r["provenance"]["clean"] for r in rows),
        "route_lesion_collapses_all": all(r["route_lesion_collapses"] for r in rows),
        "res_lesion_collapses_all": all(r["res_lesion_collapses"] for r in rows),
        "moat_clean_all": all(r["moat_false_accept"] <= 0.05 for r in rows),
        "mean_route_recall": float(np.mean([r["route_recall"] for r in rows])),
        "total_elapsed_s": round(time.time() - t0, 1),
    }
    print(f"\n[rungB1] VERDICT: {agg['verdict']} ({n_go}/{len(rows)}) -- the reservoir's LEARNED role output drives "
          f"the composer's bind SYNAPTICALLY (mean route recall {agg['mean_route_recall']:.3f}; route not worse than "
          f"dict {agg['route_not_worse_than_dict_all']}; gated {agg['gated_all']}; provenance-clean "
          f"{agg['provenance_clean_all']}; route-lesion collapses {agg['route_lesion_collapses_all']}; "
          f"reservoir-lesion collapses {agg['res_lesion_collapses_all']}; moat {agg['moat_clean_all']}).", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg}, fh, indent=2, default=str)
        print(f"[rungB1] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
