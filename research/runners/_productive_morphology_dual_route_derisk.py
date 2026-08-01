"""SEPARATE-ROUTE dual-route past tense ON A SPIKING SUBSTRATE (roadmap Stage-2 [CPU] Language de-risk) -- the
architectural FIX for the single-pool NO-GO in `_productive_morphology_construction_derisk.py`.

WHAT THE SINGLE POOL COULD NOT DO. The single-pool runner put BOTH routes -- the PROCEDURAL rule (PAST -> "-ed")
and the DECLARATIVE store (stem -> whole-form) -- in ONE shared D-sparse WTA pool. The store worked (irregular
blocking irr_acc 0.857, lesion -> over-regularization 0.952, permuted-collapse 0.024) but the rule did NOT
generalize (reg_acc 0.188): novel stems (wug/blick/dax) got CAPTURED by the entrenched irregular whole-form
attractors instead of taking "-ed". An operating-point sweep proved it ARCHITECTURAL -- raising rule strength
drives reg_acc -> 1.0 but COLLAPSES blocking (irr_acc -> 0.43); no single op-point does both, because rule and
store COMPETE in the same WTA. Pinker-Ullman words-and-rules: the rule and the exceptions live in SEPARATE
systems, and blocking is a CROSS-system suppression, not a within-pool race.

THE FIX (all neurons/synapses -- NO host `if verb in irregular_dict`; the routing is SYNAPTIC).
  * ROUTE 1 -- PROCEDURAL (dedicated, STRONG): PAST -> AFFIX excitatory recurrent trained so PAST fires the "-ed"
    affix STEM-INDEPENDENTLY and RELIABLY (co-activate PAST+AFFIX only, no stem in the conjunction). Because it
    is NOT racing the whole-forms in a shared WTA, strengthening it does NOT harm blocking -- the wall the single
    pool hit is gone.
  * ROUTE 2 -- DECLARATIVE (the exception store): stem -> whole-form associations for the irregulars, ENTRENCHED
    (identical to the single-pool store; NO PAST in the conjunction, so PAST never drives a whole-form).
  * BLOCKING = whole-form -> AFFIX INHIBITION (the new cross-route mechanism). The whole-form neurons send
    INHIBITORY synapses onto the AFFIX neurons. When an entrenched whole-form is strongly retrieved (an irregular
    stem cue), it SUPPRESSES the affix -> the whole-form wins ("went", not "goed"). For a NOVEL stem no whole-form
    is retrieved -> no inhibition -> ROUTE 1 (PAST->AFFIX) wins by default ("wug" -> "wug-ed"). This is
    Marcus/Pinker blocking as retrieval-strength-GATED cross-route inhibition: strong store association => strong
    inhibition; weak/absent => the rule surfaces. The inhibition is presynaptic-firing-gated FOR FREE -- a
    fixed-magnitude inhibitory synapse delivers inhibition PROPORTIONAL to how strongly the whole-form fires.

REALIZATION on the D-sparse primitives (`_D_sparse_heteroassoc.build`: one excitatory pool + FS-WTA + a plastic
pool->pool recurrent). The blocking inhibition is written as NEGATIVE weights on EXISTING recurrent edges from the
whole-form neurons to the affix neurons (`wire_wf_to_affix_inhibition`), modifying only `.data` in place -- the
same safe pattern the store LESION uses. Layout is cp_connections[i,j] = weight of i->j (row=pre, col=post,
bridge.py:2702); the step routes g_increase[post] = sum_pre W[pre,post]*fired[pre], so W[wf, affix] < 0 subtracts
from the affix's excitatory drive whenever the whole-form fires. These edges never co-fire (whole-forms train
with stems, affix with PAST), so they sit at ~0 and are free to repurpose as the blocking pathway.

  >> DOCUMENTED SHORTCUT (brain-based-only ledger): the engine enforces DALE'S LAW -- E/I sign comes from the
     PREsynaptic neuron's trait, and pool neurons are excitatory. So this realizes the inhibition as a
     SIGN-INVERTED excitatory synapse (a negative g_e contribution), NOT a Dale-compliant GABAergic interneuron.
     The FAITHFUL biologization is DI-SYNAPTIC feedforward inhibition: whole-form(exc) -> dedicated inhibitory
     interneuron pool -> affix, so the sign is carried by a real inhibitory cell. That is the next step if this
     GOes; it does not change the computation being tested here (retrieval-gated cross-route suppression), only
     which cell carries the minus sign.

GO gate (the thing the single pool COULD NOT do -- BOTH SIMULTANEOUSLY, 6-seed): novel-stem/held-out regular
inflection reg_acc >= 0.90 (the rule GENERALIZES) AND irregular blocking irr_acc >= 0.85 (the store still BLOCKS),
on the SAME brain, on >=5/6 seeds. Anti-cheats (all mandatory, reused from the single-pool runner): (1) UNSEEN
pseudo-stems inflect by RULE; (2) LESION the declarative store -> OVER-REGULARIZATION (irregulars -> "-ed"
specifically); (3) PERMUTED stem->whole-form binding COLLAPSES irregular retrieval to chance; (4) cfg.seed
substrate-hash -- the D-sparse build must actually seed the neurons from cfg.seed (the 2026-07-17 trap).

numpy, POOL-PORTABLE (no bridge checkpoint), seeded by cfg.seed. Reuse-by-import from the single-pool runner +
_D_sparse_heteroassoc; NO `sim/` edit, NO edit to any shared runner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os

import numpy as np

from research.runners._D_sparse_heteroassoc import build, _pool_global
from research.runners.concept_pool_sparse_distributed import generate_sparse_patterns
# reuse the vocabulary, item map, encode/readout/lesion primitives VERBATIM (no shared-runner edit)
from research.runners._productive_morphology_construction_derisk import (
    IRREGULARS, REG_TRAIN, REG_HELDOUT, NOVEL_STEMS, PAST, AFFIX,
    _make_items, co_activate, complete, lesion_neurons, _argmax_form,
)
from sim.backend import to_host


# ---- the NEW mechanism: whole-form -> affix inhibition (cross-route blocking) ------------------------------
def wire_wf_to_affix_inhibition(bridge, wf_globals, affix_globals, inhib):
    """Write INHIBITORY synapses whole-form -> affix into the live recurrent CSR, in place (structure/nnz
    unchanged -- exactly the `lesion_neurons` pattern, so no CSR rebuild). Sets every EXISTING recurrent edge
    whose PREsynaptic source is a whole-form neuron and whose POSTsynaptic target is an affix neuron to
    -inhib.

    Orientation (bridge.py:2702): cp_connections[i,j] = weight of i->j, so row i = presynaptic (FROM), col j =
    postsynaptic (TO). A whole-form(pre) -> affix(post) edge is W[wf_row, affix_col]. The per-step propagation
    is g_increase[post] = sum_pre W[pre,post]*fired[pre] (W.T @ fired), so a NEGATIVE W[wf, affix] SUBTRACTS
    from the affix's excitatory conductance whenever the whole-form fires -> retrieval-strength-gated
    suppression of the "-ed" affix (BLOCKING). These edges never co-fire during training (whole-forms with
    stems, affix with PAST), so they sit at ~0 and are free to repurpose.

    Returns the number of edges written (blocking-pathway coverage; expect ~pattern_size^2 * rec_density)."""
    W = bridge.cp_connections
    affix = np.asarray(affix_globals, dtype=np.int64)
    written = 0
    for r in np.asarray(wf_globals, dtype=np.int64):        # r = presynaptic whole-form neuron (source row)
        lo, hi = int(W.indptr[r]), int(W.indptr[r + 1])
        cols = np.asarray(W.indices[lo:hi])                 # postsynaptic targets of this whole-form neuron
        mask = np.isin(cols, affix)                         # ... that land on an affix neuron
        if mask.any():
            seg = np.asarray(W.data[lo:hi])
            seg[mask] = -abs(float(inhib))                  # SIGN-INVERTED excitatory synapse = inhibition (Dale shortcut)
            W.data[lo:hi] = seg
            written += int(mask.sum())
    try:
        bridge._invalidate_coo_cache()
    except Exception:
        pass
    return written


# ---- anti-cheat (4): the D-sparse build MUST seed the substrate from cfg.seed ------------------------------
def _threshold_hash(bridge):
    th = getattr(bridge, "cp_neuron_firing_thresholds", None)
    if th is None:
        return None
    arr = np.ascontiguousarray(np.asarray(to_host(th)))
    return hashlib.md5(arr.tobytes()).hexdigest()


def verify_seeded(seed, n_pool):
    """Build twice at the SAME seed and once at a DIFFERENT seed; same-seed firing thresholds identical,
    cross-seed different => cfg.seed actually controls the neurons (the 2026-07-17 'seed never controlled the
    substrate' trap -- `actual_seed_used` seeds NOTHING; `_D_sparse_heteroassoc.build` sets `cfg.seed`)."""
    h1 = _threshold_hash(build(seed, n_pool=n_pool))
    h2 = _threshold_hash(build(seed, n_pool=n_pool))
    h3 = _threshold_hash(build(seed + 9973, n_pool=n_pool))
    same = (h1 is not None and h1 == h2)
    diff = (h1 is not None and h1 != h3)
    return {"same_seed_identical": bool(same), "cross_seed_differs": bool(diff),
            "seeds_substrate": bool(same and diff), "threshold_hash": h1}


# ---- one seed: build the two routes + blocking inhibition, run every probe --------------------------------
def run(seed, n_pool=2000, pattern_size=90, cyc_rule=40, cyc_irr=48, inhib_strength=6.0, verbose=True):
    items, item2idx = _make_items()
    idx2name = {i: w for w, i in item2idx.items()}
    n_items = len(items)

    # the readout decides among the default affix + every stored whole-form
    competitors = [item2idx[AFFIX]] + [item2idx[wf] for wf in IRREGULARS.values()]

    def build_brain(permute_binding=False):
        b = build(seed, n_pool=n_pool)
        pats = generate_sparse_patterns(n_items, n_pool, pattern_size, seed)
        pg = _pool_global(b, pats)
        # --- ROUTE 1 -- PROCEDURAL: PAST -> AFFIX, DEDICATED + STRONG, stem-independent. One entrenched
        #     conjunction of PAST+AFFIX ONLY (no stem) => PAST reliably drives "-ed" for ANY stem. This is the
        #     key change from the single pool (where the rule was DELIBERATELY WEAK so the WTA could block it);
        #     here blocking is separate inhibition, so a strong rule costs nothing. ---
        co_activate(b, pg, [item2idx[PAST], item2idx[AFFIX]], cycles=cyc_rule)
        # --- ROUTE 2 -- DECLARATIVE store: stem -> whole-form per irregular, ENTRENCHED (no PAST in the
        #     conjunction, identical to the single-pool store). ---
        wf_names = list(IRREGULARS.values())
        if permute_binding:                                  # ANTI-CHEAT (3): derange which stem takes which form
            rng = np.random.RandomState(seed * 101 + 7)
            perm = rng.permutation(len(wf_names))
            while np.any(perm == np.arange(len(wf_names))):  # a true derangement (no fixed point)
                perm = rng.permutation(len(wf_names))
            wf_for_stem = {stem: wf_names[perm[i]] for i, stem in enumerate(IRREGULARS.keys())}
        else:
            wf_for_stem = dict(IRREGULARS)
        for stem, wf in wf_for_stem.items():
            co_activate(b, pg, [item2idx[stem], item2idx[wf]], cycles=cyc_irr)
        # --- BLOCKING -- whole-form -> AFFIX INHIBITION (the new cross-route mechanism). Wire AFTER both routes
        #     are trained so the negative edges are not re-grown by Hebbian. ---
        affix_glob = np.asarray(pg[item2idx[AFFIX]])
        wf_glob = np.concatenate([np.asarray(pg[item2idx[wf]]) for wf in IRREGULARS.values()])
        n_inh = wire_wf_to_affix_inhibition(b, wf_glob, affix_glob, inhib_strength)
        return b, pats, pg, n_inh

    # ============================ MAIN BRAIN (correct binding) ============================
    b, pats, pg, n_inh = build_brain(permute_binding=False)

    # (1) REGULAR PRODUCTIVITY -- held-out + novel stems must take "-ed" by RULE (route 1 wins: no whole-form
    #     retrieved => no inhibition => PAST->AFFIX surfaces).
    reg_probe = REG_HELDOUT + NOVEL_STEMS
    reg_ok = 0
    reg_detail = {}
    for stem in reg_probe:
        cue = [item2idx[stem], item2idx[PAST]]
        form, sc = _argmax_form(complete(b, pats, pg, cue, competitors, n_pool), idx2name)
        ok = (form == AFFIX)
        reg_ok += int(ok)
        reg_detail[stem] = form
        if verbose:
            print(f"  REGULAR {stem:6s}+PAST -> '{stem}{form}' {'[rule -ed OK]' if ok else '[got '+form+']'}", flush=True)
    reg_acc = reg_ok / len(reg_probe)

    # (2) IRREGULAR RETRIEVAL / BLOCKING -- the entrenched whole-form is retrieved and INHIBITS the affix, so it
    #     WINS over the default "-ed" (cross-route blocking).
    irr_ok = 0
    irr_edrate_pre = 0
    irr_detail = {}
    for stem, wf in IRREGULARS.items():
        cue = [item2idx[stem], item2idx[PAST]]
        form, sc = _argmax_form(complete(b, pats, pg, cue, competitors, n_pool), idx2name)
        ok = (form == wf)
        irr_ok += int(ok)
        irr_edrate_pre += int(form == AFFIX)
        irr_detail[stem] = form
        if verbose:
            print(f"  IRREG   {stem:6s}+PAST -> '{form}' (target '{wf}') "
                  f"{'[stored, BLOCKS -ed]' if ok else '[got '+form+']'}", flush=True)
    irr_acc = irr_ok / len(IRREGULARS)
    irr_edrate_pre /= len(IRREGULARS)

    # (3) LESION the declarative store -> OVER-REGULARIZATION. Zeroing every edge incident to the whole-form
    #     neurons removes BOTH the store's stem->whole-form edges AND the whole-form->affix inhibitory edges
    #     (both are edges from/to whole-forms), so no whole-form is retrieved AND no inhibition remains ->
    #     PAST->AFFIX surfaces for the irregulars -> "-ed" specifically.
    wf_globals = np.concatenate([np.asarray(pg[item2idx[wf]]) for wf in IRREGULARS.values()])
    lesion_neurons(b, wf_globals)
    overreg = 0
    overreg_other = 0
    les_detail = {}
    for stem, wf in IRREGULARS.items():
        cue = [item2idx[stem], item2idx[PAST]]
        form, sc = _argmax_form(complete(b, pats, pg, cue, competitors, n_pool), idx2name)
        les_detail[stem] = form
        if form == AFFIX:
            overreg += 1
        elif form != wf:
            overreg_other += 1
        if verbose:
            print(f"  LESION  {stem:6s}+PAST -> '{stem}{form if form==AFFIX else '/'+form}' "
                  f"{'[OVER-REGULARIZED]' if form==AFFIX else ''}", flush=True)
    overreg_rate = overreg / len(IRREGULARS)
    # ATTRIBUTION: is the over-regularization ATTRIBUTABLE to LESIONING the declarative store, or was the affix
    # already surfacing pre-lesion? (attribution-required gate + the honest "whose is the difference".)
    from tools.lab import attributable_to
    attributable_to("irregular over-regularization to -ed: post-store-lesion vs pre-lesion baseline",
                    overreg_rate, irr_edrate_pre)

    # ============================ PERMUTED-BINDING BRAIN (anti-cheat 3) ============================
    bp, patsp, pgp, _ = build_brain(permute_binding=True)
    perm_ok = 0
    for stem, wf in IRREGULARS.items():
        cue = [item2idx[stem], item2idx[PAST]]
        form, sc = _argmax_form(complete(bp, patsp, pgp, cue, competitors, n_pool), idx2name)
        perm_ok += int(form == wf)                           # correct only by chance now (binding deranged)
    perm_acc = perm_ok / len(IRREGULARS)

    # ============================ PER-SEED VERDICT ============================
    # BOTH GATES SIMULTANEOUSLY (the single pool could not) + the store/lesion/permuted anti-cheats.
    both_gates = (reg_acc >= 0.90 and irr_acc >= 0.85)
    go = (both_gates and overreg_rate >= 0.85
          and overreg_rate - irr_edrate_pre >= 0.5 and perm_acc <= 0.30)
    result = {
        "seed": int(seed), "n_pool": n_pool, "pattern_size": pattern_size,
        "cyc_rule": cyc_rule, "cyc_irr": cyc_irr, "inhib_strength": inhib_strength,
        "n_inhib_edges": int(n_inh),
        "reg_acc": reg_acc, "irr_acc": irr_acc, "both_gates": bool(both_gates),
        "irr_edrate_pre_lesion": irr_edrate_pre,
        "overreg_rate_lesion": overreg_rate, "lesion_other_error_rate": overreg_other / len(IRREGULARS),
        "permuted_binding_irr_acc": perm_acc,
        "reg_detail": reg_detail, "irr_detail": irr_detail, "lesion_detail": les_detail,
        "GO": bool(go),
    }
    if verbose:
        print(f"\n  seed {seed}: reg(rule) {reg_acc:.2f} & irr(blocking) {irr_acc:.2f} "
              f"{'[BOTH]' if both_gates else '[NOT both]'} | lesion->over-reg {overreg_rate:.2f} "
              f"(pre {irr_edrate_pre:.2f}) | permuted {perm_acc:.2f} | inhib_edges {n_inh} "
              f"| {'GO' if go else 'NO-GO'}", flush=True)
    return result


# ---- 6-seed aggregation (GO = both gates + anti-cheats on >=5/6 seeds) -------------------------------------
def summarize(base_seed, n_seeds=6, n_pool=2000, pattern_size=90, cyc_rule=40, cyc_irr=48,
              inhib_strength=6.0, verbose=True):
    seeds = [base_seed + i for i in range(n_seeds)]
    seed_check = verify_seeded(base_seed, n_pool)
    if verbose:
        print(f"[seed anti-cheat] cfg.seed controls substrate: {seed_check['seeds_substrate']} "
              f"(same-seed identical={seed_check['same_seed_identical']}, "
              f"cross-seed differs={seed_check['cross_seed_differs']})", flush=True)

    results = [run(s, n_pool=n_pool, pattern_size=pattern_size, cyc_rule=cyc_rule, cyc_irr=cyc_irr,
                   inhib_strength=inhib_strength, verbose=verbose) for s in seeds]

    n_reg = sum(1 for r in results if r["reg_acc"] >= 0.90)
    n_irr = sum(1 for r in results if r["irr_acc"] >= 0.85)
    n_both = sum(1 for r in results if r["both_gates"])       # reg AND blocking on the SAME seed
    n_go = sum(1 for r in results if r["GO"])                 # both gates + all anti-cheats

    go = (n_go >= 5) and seed_check["seeds_substrate"]
    if go:
        verdict = f"DUAL-ROUTE GO -- reg AND blocking simultaneously ({n_both}/{n_seeds} both-gates, {n_go}/{n_seeds} full-GO)"
    elif not seed_check["seeds_substrate"]:
        verdict = "NEGATIVE -- substrate NOT seeded by cfg.seed (anti-cheat 4 failed; results confounded)"
    elif n_both < 5 and n_reg < 5 and n_irr >= 5:
        verdict = f"NEGATIVE -- the RULE fails to generalize (reg_acc>=0.90 on only {n_reg}/{n_seeds})"
    elif n_both < 5 and n_irr < 5 and n_reg >= 5:
        verdict = f"NEGATIVE -- BLOCKING fails (irr_acc>=0.85 on only {n_irr}/{n_seeds})"
    elif n_both < 5:
        verdict = (f"NEGATIVE -- no single op-point does BOTH (both-gates {n_both}/{n_seeds}; "
                   f"reg {n_reg}/{n_seeds}, irr {n_irr}/{n_seeds}) -- sweep --inhib-strength / --cyc-rule")
    else:
        verdict = (f"NEGATIVE -- anti-cheats fail on >1 seed (both-gates {n_both}/{n_seeds} but "
                   f"full-GO only {n_go}/{n_seeds}: check lesion->over-reg / permuted-collapse)")

    summary = {
        "probe": "productive_morphology_dual_route",
        "config": {
            "n_seeds": n_seeds, "base_seed": base_seed, "n_pool": n_pool, "pattern_size": pattern_size,
            "cyc_rule": cyc_rule, "cyc_irr": cyc_irr, "inhib_strength": inhib_strength,
        },
        "per_seed": [
            {"seed": r["seed"], "reg_acc": r["reg_acc"], "irr_acc": r["irr_acc"],
             "both_gates": r["both_gates"], "overreg_rate_lesion": r["overreg_rate_lesion"],
             "permuted_binding_irr_acc": r["permuted_binding_irr_acc"],
             "n_inhib_edges": r["n_inhib_edges"], "GO": r["GO"]}
            for r in results
        ],
        "n_reg_ge_0.90": n_reg, "n_irr_ge_0.85": n_irr,
        "n_both_gates": n_both, "n_full_go": n_go,
        "seed_check": seed_check,
        "GO": bool(go),
        "verdict": verdict,
    }
    if verbose:
        print(f"\n=== dual-route morphology summary ({n_seeds} seeds) ===", flush=True)
        print(f"  reg_acc>=0.90: {n_reg}/{n_seeds} | irr_acc>=0.85: {n_irr}/{n_seeds} "
              f"| BOTH simultaneously: {n_both}/{n_seeds} | full-GO: {n_go}/{n_seeds}", flush=True)
        print(f"  VERDICT: {verdict}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42, help="base seed; the 6-seed sweep uses seed..seed+n_seeds-1")
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--n-pool", type=int, default=2000)
    ap.add_argument("--pattern-size", type=int, default=90)
    ap.add_argument("--cyc-rule", type=int, default=40, help="ROUTE 1 (PAST->AFFIX) strength -- dedicated, strong")
    ap.add_argument("--cyc-irr", type=int, default=48, help="ROUTE 2 (stem->whole-form) entrenchment")
    ap.add_argument("--inhib-strength", type=float, default=6.0,
                    help="whole-form->affix inhibition weight -- THE key new knob (cross-route blocking)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print(f"[dual-route morphology] SEPARATE routes + whole-form->affix inhibition | base seed={a.seed} "
          f"n_seeds={a.n_seeds} inhib={a.inhib_strength}", flush=True)
    s = summarize(a.seed, n_seeds=a.n_seeds, n_pool=a.n_pool, pattern_size=a.pattern_size,
                  cyc_rule=a.cyc_rule, cyc_irr=a.cyc_irr, inhib_strength=a.inhib_strength)
    print(f"\n  {'GO' if s['GO'] else 'NO-GO'} -- {s['verdict']}", flush=True)
    if a.out:
        if os.path.dirname(a.out):
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(s, open(a.out, "w"), indent=1)
        print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
