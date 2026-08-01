"""DUAL-ROUTE productive morphology ON A SPIKING SUBSTRATE (roadmap Stage-2 [CPU] Language de-risk): the
past-tense of a verb is produced by TWO competing neural routes -- a PROCEDURAL rule (PAST -> default "-ed"
affix, stem-independent, applies to ANY stem incl. never-seen pseudo-stems) and a DECLARATIVE store (an
entrenched stem -> whole-form association for irregulars, go->went / run->ran). The two routes compete in a
shared spiking pool via FS/WTA lateral inhibition; the entrenched stored form OUTCOMPETES the default rule
(Marcus/Pinker BLOCKING: "went", not "goed"). LESION the declarative store -> the default rule surfaces for
irregulars -> OVER-REGULARIZATION ("goed"/"runned"), the load-bearing dual-route dissociation.

Pinker-Ullman words-and-rules / Marcus et al. 1992 blocking + over-regularization. The substrate is the D
sparse heteroassociative spiking memory (`_D_sparse_heteroassoc.build`): a shared spiking pool with a
PLASTIC excitatory recurrent (the associations, grown by Hebbian co-firing) + FS inhibition (the WTA that
implements route competition / blocking). All neurons/synapses -- no host `if verb in irregular_dict`; which
verbs block is stored in the recurrent weights, and the block is removed by a SYNAPTIC lesion. numpy,
POOL-PORTABLE (no bridge checkpoint), seeded by `cfg.seed`. NO `sim/` edit.

GO gate (roadmap): novel-stem regular inflection >=0.90 (rule not lookup); irregulars -> stored path
(blocking); 6-seed. Anti-cheats: (1) UNSEEN-STEM inflects by RULE -- never-encoded pseudo-stems still take
"-ed"; (2) LESION -> OVER-REGULARIZATION -- zero the declarative store, irregulars regularize (to "-ed"
specifically, not random); (3) PERMUTED morph-binding COLLAPSES -- deranging the stem->whole-form binding at
encode collapses irregular retrieval to chance (the learned synaptic binding is load-bearing).
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np

from research.runners._D_sparse_heteroassoc import build, _pool_global, _drive
from research.runners.concept_pool_sparse_distributed import generate_sparse_patterns
from sim.backend import to_host

# ---- the morphological vocabulary --------------------------------------------------------------------------
# irregular verbs: STEM -> stored whole-form PAST (the declarative exceptions that must BLOCK the rule)
IRREGULARS = {"go": "went", "run": "ran", "eat": "ate", "come": "came",
              "take": "took", "give": "gave", "sleep": "slept"}
# regular verbs used to TEACH the PAST->-ed rule (co-activated with PAST + the affix, stem-independent)
REG_TRAIN = ["walk", "jump", "play", "look", "want", "help", "call", "wait"]
# HELD-OUT regular stems -- never co-encoded with the affix (tests the rule GENERALIZES, not looks up)
REG_HELDOUT = ["kick", "pull", "cook", "talk"]
# genuinely NOVEL pseudo-stems -- never in ANY encoding (the strongest rule-not-lookup probe; "wug" test)
NOVEL_STEMS = ["wug", "blick", "dax", "gorp"]

PAST = "<PAST>"   # the tense feature that triggers past-tense production (drives the default affix)
AFFIX = "-ed"     # the regular past bound morpheme (the default rule's output)


def _make_items():
    """Ordered item list -> each gets its own sparse pattern in the pool."""
    stems = list(IRREGULARS.keys()) + REG_TRAIN + REG_HELDOUT + NOVEL_STEMS
    wholeforms = list(IRREGULARS.values())
    items = stems + wholeforms + [PAST, AFFIX]
    assert len(items) == len(set(items)), "item names must be unique"
    return items, {w: i for i, w in enumerate(items)}


def co_activate(bridge, pg, idxs, cycles, pA=1100.0, on=10, off=5):
    """Co-drive the given item patterns together for `cycles` episodes -> Hebbian grows the all-to-all recurrent
    among the co-active neurons (plasticity gated ON only here; readout is strictly read-only)."""
    try:
        bridge.set_plasticity_gate("recurrent", 1.0)
    except KeyError:
        pass
    for _ in range(cycles):
        _drive(bridge, [pg[i] for i in idxs], pA)
        for _ in range(on):
            bridge._run_one_simulation_step()
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(off):
            bridge._run_one_simulation_step()
    try:
        bridge.set_plasticity_gate("recurrent", 0.0)
    except KeyError:
        pass


def complete(bridge, patterns, pg, cue_idxs, competitor_idxs, n_pool, window=40, pA=1100.0):
    """READ-ONLY (plasticity gate 0): drive the cue items -> accumulate pool firing -> EXCLUDE the directly-driven
    cue neurons -> cosine of the RECURRENT completion to each competitor pattern. Returns {item_idx: score}."""
    pool_base = np.asarray(bridge.region_manager.indices("pool"))
    _drive(bridge, [pg[i] for i in cue_idxs], pA)
    firing = np.zeros(n_pool)
    for _ in range(window):
        bridge._run_one_simulation_step()
        fs = np.asarray(to_host(bridge.cp_firing_states)).astype(float)
        firing += fs[pool_base]
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(15):
        bridge._run_one_simulation_step()
    for i in cue_idxs:                                   # remove the directly-driven cue -> pure recurrent output
        firing[np.asarray(patterns[i])] = 0.0
    nf = float(np.linalg.norm(firing))
    scores = {}
    for j in competitor_idxs:
        v = np.zeros(n_pool)
        v[np.asarray(patterns[j])] = 1.0
        scores[j] = float(firing @ v / (nf * np.linalg.norm(v))) if nf > 0 else 0.0
    return scores


def lesion_neurons(bridge, global_idx):
    """SYNAPTIC lesion of the declarative store: zero every recurrent edge INCIDENT to the whole-form neurons
    (in place, structure/nnz unchanged -> the live CSR .data the step reads is updated). Removes the stored
    irregular representations while leaving PAST->-ed (the rule, onto the affix neurons) intact."""
    W = bridge.cp_connections
    wf = np.asarray(global_idx, dtype=np.int64)
    col_mask = np.isin(W.indices, wf)                   # edges FROM whole-form neurons
    W.data[col_mask] = 0.0
    for r in wf:                                         # edges TO whole-form neurons
        W.data[W.indptr[r]:W.indptr[r + 1]] = 0.0
    try:
        bridge._invalidate_coo_cache()
    except Exception:
        pass


def _argmax_form(scores, idx2name):
    j = max(scores, key=lambda k: scores[k])
    return idx2name[j], scores


def run(seed, n_pool=2000, pattern_size=90, cyc_rule=25, cyc_irr=48, verbose=True):
    items, item2idx = _make_items()
    idx2name = {i: w for w, i in item2idx.items()}
    n_items = len(items)

    # competitor set the readout decides among: the default affix + every stored whole-form
    competitors = [item2idx[AFFIX]] + [item2idx[wf] for wf in IRREGULARS.values()]

    def build_brain(permute_binding=False):
        b = build(seed, n_pool=n_pool)
        pats = generate_sparse_patterns(n_items, n_pool, pattern_size, seed)
        pg = _pool_global(b, pats)
        # --- PROCEDURAL RULE: PAST -> "-ed" (stem-independent default). Fewer episodes => weaker than an
        #     entrenched irregular, so the specific stored form BLOCKS it (mature dual-route ordering). ---
        for _ in REG_TRAIN:
            co_activate(b, pg, [item2idx[PAST], item2idx[AFFIX]], cycles=max(1, cyc_rule // len(REG_TRAIN)))
        # --- DECLARATIVE STORE: stem -> whole-form for each irregular, ENTRENCHED (more episodes). NO PAST in
        #     the conjunction, so PAST never drives whole-forms -> a regular verb never spuriously retrieves one. ---
        wf_names = list(IRREGULARS.values())
        if permute_binding:                              # ANTI-CHEAT: derange which stem takes which stored form
            rng = np.random.RandomState(seed * 101 + 7)
            perm = rng.permutation(len(wf_names))
            while np.any(perm == np.arange(len(wf_names))):   # a true derangement (no fixed point)
                perm = rng.permutation(len(wf_names))
            wf_for_stem = {stem: wf_names[perm[i]] for i, stem in enumerate(IRREGULARS.keys())}
        else:
            wf_for_stem = dict(IRREGULARS)
        for stem, wf in wf_for_stem.items():
            co_activate(b, pg, [item2idx[stem], item2idx[wf]], cycles=cyc_irr)
        return b, pats, pg

    # ============================ MAIN BRAIN (correct binding) ============================
    b, pats, pg = build_brain(permute_binding=False)

    # (1) REGULAR PRODUCTIVITY -- held-out + novel stems must take "-ed" by RULE (not lookup)
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

    # (2) IRREGULAR RETRIEVAL / BLOCKING -- the stored whole-form must WIN over the default "-ed"
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

    # (3) LESION the declarative store -> OVER-REGULARIZATION (irregulars now take "-ed")
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

    # ============================ PERMUTED-BINDING BRAIN (anti-cheat) ============================
    bp, patsp, pgp = build_brain(permute_binding=True)
    perm_ok = 0
    for stem, wf in IRREGULARS.items():
        cue = [item2idx[stem], item2idx[PAST]]
        form, sc = _argmax_form(complete(bp, patsp, pgp, cue, competitors, n_pool), idx2name)
        perm_ok += int(form == wf)                       # correct only by chance now (binding deranged)
    perm_acc = perm_ok / len(IRREGULARS)

    # ============================ VERDICT ============================
    go = (reg_acc >= 0.90 and irr_acc >= 0.85 and overreg_rate >= 0.85
          and overreg_rate - irr_edrate_pre >= 0.5 and perm_acc <= 0.30)
    result = {
        "seed": int(seed), "n_pool": n_pool, "pattern_size": pattern_size,
        "cyc_rule": cyc_rule, "cyc_irr": cyc_irr,
        "reg_acc": reg_acc, "irr_acc": irr_acc,
        "irr_edrate_pre_lesion": irr_edrate_pre,
        "overreg_rate_lesion": overreg_rate, "lesion_other_error_rate": overreg_other / len(IRREGULARS),
        "permuted_binding_irr_acc": perm_acc,
        "reg_detail": reg_detail, "irr_detail": irr_detail, "lesion_detail": les_detail,
        "GO": bool(go),
    }
    if verbose:
        print(f"\n  seed {seed}: reg(rule) {reg_acc:.2f} | irr(blocking) {irr_acc:.2f} "
              f"| lesion->over-reg {overreg_rate:.2f} (pre {irr_edrate_pre:.2f}) "
              f"| permuted-binding {perm_acc:.2f} | {'GO' if go else 'NO-GO'}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-pool", type=int, default=2000)
    ap.add_argument("--pattern-size", type=int, default=90)
    ap.add_argument("--cyc-rule", type=int, default=25)
    ap.add_argument("--cyc-irr", type=int, default=48)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print(f"[dual-route morphology] spiking pool, WTA blocking, synaptic lesion | seed={a.seed}", flush=True)
    r = run(a.seed, n_pool=a.n_pool, pattern_size=a.pattern_size, cyc_rule=a.cyc_rule, cyc_irr=a.cyc_irr)
    print(f"\n  VERDICT: {'GO' if r['GO'] else 'NO-GO'} -- dual-route past tense ON SPIKES: regular rule "
          f"generalizes to unseen stems ({r['reg_acc']:.2f}), entrenched irregulars BLOCK the rule "
          f"({r['irr_acc']:.2f}), lesioning the declarative store -> OVER-REGULARIZATION "
          f"({r['overreg_rate_lesion']:.2f}), permuted binding collapses ({r['permuted_binding_irr_acc']:.2f}).",
          flush=True)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True) if os.path.dirname(a.out) else None
        json.dump(r, open(a.out, "w"), indent=1)
        print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
