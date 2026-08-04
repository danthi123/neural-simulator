"""DISTINCT-POPULATIONS dual-route past tense ON A SPIKING SUBSTRATE (roadmap Stage-2 [CPU] Language de-risk) --
the ARCHITECTURAL fix the two prior morphology negatives BOTH named as the untested next lever.

WHAT THE RECORD ALREADY ESTABLISHED (do not re-derive):
  * SINGLE SHARED POOL, both routes (`_productive_morphology_construction_derisk.py`): the declarative store works
    6/6 (irr blocking 0.857, lesion->over-reg 0.952, permuted-collapse 0.024) but the RULE does NOT generalize
    (reg_acc 0.188). Op-point sweep proved it ARCHITECTURAL: raising rule strength -> reg_acc 1.0 but COLLAPSES
    blocking (irr_acc 0.43). NO single op-point does both -- rule and store COMPETE in the same WTA.
  * SINGLE POOL + whole-form->affix INHIBITION (`_productive_morphology_dual_route_derisk.py`, despite its name it
    is still ONE `build(n_pool=2000)`): inhib sweep 0.5..6.0 caps reg_acc at 0.25 regardless of inhibition. The
    routes are CO-LOCATED, so a novel stem's pattern spuriously activates whole-form neurons via pattern overlap in
    the SHARED recurrent, and those win the readout -- inhibition adds blocking but cannot make the affix
    competitive for novel stems.

Both negatives named the SAME next lever verbatim: "genuinely separate pools -- the PAST->affix rule in its OWN
dedicated pool, isolated from the whole-form store, so a novel stem cannot spuriously retrieve a whole-form, and the
affix wins by default; the declarative store in a SECOND pool; and a clean inter-pool inhibitory projection
(whole-form pool -> affix neurons) for blocking." This runner builds exactly that and tests it.

THE HYPOTHESIS (single variable vs the single-pool control): does STRUCTURAL SEPARATION -- two structurally-isolated
spiking pools with their OWN recurrents and OWN FS-WTA, plus a cross-pool blocking projection -- let the procedural
route generalize the -ed rule to novel stems (wug->wugged) where the single pool (reg_acc<=0.25) could not? The
prediction: yes, because the PROC pool contains NO whole-form attractors, so a novel stem cue cannot pull it off the
default affix; and for an irregular the LEX pool retrieves the whole-form which inhibits the PROC affix (blocking).

ARCHITECTURE (all neurons/synapses in ONE brain -- no host `if verb in irregular_dict`):
  * LEX pool (declarative store): STEM + WHOLE-FORM patterns; recurrent trained stem->whole-form for irregulars
    (ENTRENCHED). Its OWN FS-WTA (fs_lex). A novel/regular stem retrieves NO whole-form.
  * PROC pool (procedural rule): PAST + AFFIX patterns; recurrent trained PAST->AFFIX (STRONG, stem-independent).
    Its OWN FS-WTA (fs_proc). NO stems and NO whole-forms live here -> PAST fires "-ed" for ANY stem by default,
    with nothing to capture it. This is the structural change vs both single-pool runners.
  * BLOCKING = LEX whole-form -> PROC affix cross-pool inhibition. A dedicated lex->proc pathway (weight 0 at build)
    whose whole-form(pre)->affix(post) edges are set NEGATIVE after training (`wire_wf_to_affix_inhibition`, reused
    verbatim from the dual_route runner). When an irregular whole-form is strongly retrieved in LEX it suppresses
    the affix in PROC (retrieval-strength-gated blocking -> "went"). For a novel stem no whole-form fires -> no
    inhibition -> PROC's PAST->AFFIX wins by default -> "wug-ed".

DOCUMENTED HOST-SHORTCUTS (brain-based-only ledger -- to burn down if this GOes):
  (S1) DALE'S LAW: the engine derives E/I sign from the PREsynaptic trait and LEX neurons are excitatory, so the
       whole-form->affix blocking is realized as a SIGN-INVERTED excitatory synapse (negative g_e contribution),
       NOT a Dale-compliant GABAergic interneuron. FAITHFUL version = di-synaptic feedforward inhibition
       (whole-form(exc) -> dedicated inhibitory interneuron pool -> affix). Same computation, only which cell carries
       the minus sign. (Inherited from the dual_route runner; unchanged here.)
  (S2) HAND-WIRED ROUTING: which pool each item projects to (stems+whole-forms -> LEX, PAST+AFFIX -> PROC) is
       assigned in host code, not learned. In a real cortex the tense/stem features would project to both routes and
       the routing would SELF-ORGANIZE (procedural BG vs declarative MTL). This is an architecture shortcut: it
       supplies the very separation being tested. The honest next step is a developmental version where the split
       emerges; here we test whether the separation, GIVEN, unlocks generalization the single pool could not.

GO gate (same as the prior runners, both SIMULTANEOUSLY, 6-seed): novel/held-out reg_acc >= 0.90 (the rule
GENERALIZES) AND irregular blocking irr_acc >= 0.85, on the SAME brain, on >=5/6 seeds. Anti-cheats (mandatory):
(1) UNSEEN pseudo-stems inflect by RULE; (2) LESION the LEX store -> OVER-REGULARIZATION (irregulars -> "-ed");
(3) PERMUTED stem->whole-form binding COLLAPSES irregular retrieval to chance; (4) cfg.seed substrate-hash (the
2026-07-17 trap). Held-out/novel stems are GENUINELY disjoint from REG_TRAIN (imported vocabulary, unchanged).

numpy, POOL-PORTABLE (no bridge checkpoint), seeded by cfg.seed. Reuse-by-import of the vocabulary + primitives; NO
`sim/` edit, NO edit to any shared runner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os

import numpy as np

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
from sim.regions import BrainRegion, RegionPathway
from sim.bridge import SimulationBridge
from sim.backend import get_backend, to_host

from research.runners.concept_pool_sparse_distributed import generate_sparse_patterns
from research.runners._D_sparse_heteroassoc import _drive
# vocabulary + primitives reused VERBATIM (no shared-runner edit)
from research.runners._productive_morphology_construction_derisk import (
    IRREGULARS, REG_TRAIN, REG_HELDOUT, NOVEL_STEMS, PAST, AFFIX,
    _make_items, co_activate, lesion_neurons, _argmax_form,
)
from research.runners._productive_morphology_dual_route_derisk import wire_wf_to_affix_inhibition


# Recorded runs closed this mechanism. A successor needs a new preregistration
# and fresh partitions before this runner may execute again.
OPEN_PHASES: tuple[str, ...] = ()
CONSUMED_SEEDS = frozenset((*range(42, 48), *range(100, 103)))


def validate_execution_request(phase: str, base_seed: int, n_seeds: int) -> tuple[int, ...]:
    """Fail closed without constructing a bridge or consuming another seed."""
    if int(n_seeds) < 1:
        raise ValueError("n-seeds must be positive")
    seeds = tuple(range(int(base_seed), int(base_seed) + int(n_seeds)))
    consumed = sorted(set(seeds) & CONSUMED_SEEDS)
    if consumed:
        raise ValueError(
            f"morphology seeds {consumed} are consumed by existing evidence and closed"
        )
    raise ValueError(
        f"morphology phase {phase!r} is not open; no scientific partition is "
        "preregistered for this retired mechanism"
    )


# ---- TWO structurally-isolated pools (the architectural change) --------------------------------------------
def build_two_pool(seed, n_lex=2000, n_proc=800, n_fs_lex=300, n_fs_proc=150,
                   rec_density=0.6, fs_inh=1.2, block_density=0.6,
                   di_synaptic=False, n_inh_block=150, inh_block_density=0.6):
    """One brain, TWO isolated excitatory pools (lex, proc), each with its OWN FS-WTA, plus a lex->proc pathway
    (weight 0 at build) reserved for the whole-form->affix blocking inhibition. NO lex<->proc recurrent for the
    associative dynamics and NO shared FS -> the two routes cannot spuriously activate each other via pattern
    overlap (the single-pool failure mode). cfg.seed seeds the substrate (2026-07-17 trap).

    di_synaptic=False -> BYTE-IDENTICAL to the sign-inverted path (regions/pathways below unchanged; blocking via
    the lex->proc sign-inverted excitatory synapse, the S1 Dale shortcut). di_synaptic=True -> ADD a dedicated
    INHIBITORY interneuron region `inh_block` (exc_fraction=0.0) plus lex->inh_block (excitatory drive) and
    inh_block->proc (inhibitory output) pathways, so the FAITHFUL two-hop feedforward inhibition can be wired
    (whole-form(exc) -> interneuron(inh) -> affix). The extra region/pathways exist ONLY when the flag is on."""
    regions = [
        BrainRegion(name="lex", n_neurons=n_lex, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="proc", n_neurons=n_proc, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="fs_lex", n_neurons=n_fs_lex, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
        BrainRegion(name="fs_proc", n_neurons=n_fs_proc, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False),
    ]
    pathways = [
        # PLASTIC recurrents -- the heteroassociative weights within each isolated pool (grown by co-fire Hebbian)
        RegionPathway(from_region="lex", to_region="lex", density=rec_density, weight_mean=0.0,
                      weight_jitter=0.0, plastic=True, plasticity_gate="recurrent"),
        RegionPathway(from_region="proc", to_region="proc", density=rec_density, weight_mean=0.0,
                      weight_jitter=0.0, plastic=True, plasticity_gate="recurrent"),
        # per-pool FS-WTA (SEPARATE fs pools -> no cross-pool inhibition; isolation preserved)
        RegionPathway(from_region="lex", to_region="fs_lex", density=0.3, weight_mean=1.0, weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="fs_lex", to_region="lex", density=0.3, weight_mean=fs_inh, weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="proc", to_region="fs_proc", density=0.3, weight_mean=1.0, weight_jitter=0.2, plastic=False),
        RegionPathway(from_region="fs_proc", to_region="proc", density=0.3, weight_mean=fs_inh, weight_jitter=0.2, plastic=False),
        # BLOCKING substrate: lex->proc edges at weight 0 (excitatory by Dale), repurposed to whole-form->affix
        # NEGATIVE after training. weight_mean tiny-positive so the edges definitely instantiate in the CSR.
        RegionPathway(from_region="lex", to_region="proc", density=block_density, weight_mean=1e-4,
                      weight_jitter=0.0, plastic=False),
    ]
    if di_synaptic:
        # FAITHFUL di-synaptic feedforward inhibition substrate (burns down the S1 sign-inverted shortcut):
        # a dedicated GABAergic interneuron pool driven by lex whole-forms, inhibiting the proc affix. The
        # engine derives E/I sign from the PREsynaptic cell (bridge.py:6363), so an exc_fraction=0.0 region's
        # POSITIVE output weights route to g_i -> Dale-compliant inhibition carried by the cell, not the weight.
        regions.append(
            BrainRegion(name="inh_block", n_neurons=n_inh_block, exc_fraction=0.0, internal_density=0.0,
                        exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False))
        pathways.append(  # HOP 1: lex whole-form (exc) -> interneuron. Tiny-positive so edges instantiate in CSR.
            RegionPathway(from_region="lex", to_region="inh_block", density=inh_block_density,
                          weight_mean=1e-4, weight_jitter=0.0, plastic=False))
        pathways.append(  # HOP 2: interneuron (inh) -> proc affix. Tiny-positive; wired to inhib_strength after training.
            RegionPathway(from_region="inh_block", to_region="proc", density=inh_block_density,
                          weight_mean=1e-4, weight_jitter=0.0, plastic=False))
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = int(seed)
    cfg.enable_nmda = False
    cfg.enable_stdp = False
    cfg.stdp_w_max = 8.0
    cfg.enable_hebbian_learning = True
    cfg.enable_reward_modulation = False
    cfg.hebbian_max_weight = 45.0
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_learning_rate = 0.004
    cfg.enable_structural_plasticity = False
    cfg.enable_short_term_plasticity = False
    cfg.ou_std_current_pA = 0.0
    cfg.fast_spike_reset = True
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                         runtime_state=RuntimeState(), gpu_config=GPUConfig())
    b._initialize_simulation_data(called_from_playback_init=False)
    return b


def wire_disynaptic_inhibition(bridge, wf_globals, inh_globals, affix_globals, inh_drive, inhib_out):
    """FAITHFUL Dale-compliant di-synaptic feedforward inhibition (the S1 burn-down of the sign-inverted shortcut).

    TWO hops, each a REAL synapse whose sign is carried by the PREsynaptic CELL -- the engine routes each
    presynaptic spike into g_e or g_i by that neuron's inhibitory trait (bridge.py:6363, the same mechanism the
    FS-WTA pools use), NOT by the weight sign:
      HOP 1  lex whole-form (EXCITATORY) -> inh_block interneuron:  positive weight -> g_e drive on the interneuron.
             Written ONLY on whole-form presynaptic rows, so a novel/regular stem (which retrieves NO whole-form,
             so no whole-form neuron fires) drives the interneuron NOT AT ALL. This is what gates suppression on
             ACTUAL retrieval -- the property a single sign-inverted weight could not have.
      HOP 2  inh_block interneuron (INHIBITORY, exc_fraction=0) -> proc affix:  positive weight -> g_i on the affix
             (Dale-compliant; the minus sign lives in the CELL). Magnitude = inhib_out (the swept 'strength').

    Net: affix suppression SCALES with whole-form retrieval strength -- a strongly-retrieved irregular whole-form
    fires the interneuron hard -> strong affix suppression (blocking -> 'went'); a novel regular fires no
    whole-form -> interneuron silent -> affix wins by default (rule generalizes -> 'wug-ed'). Both edge sets are
    set in place on the live CSR (structure/nnz unchanged -- the lesion_neurons pattern, no CSR rebuild).
    Returns (n_drive_edges, n_out_edges)."""
    W = bridge.cp_connections
    inh = np.asarray(inh_globals, dtype=np.int64)
    affix = np.asarray(affix_globals, dtype=np.int64)
    # HOP 1: whole-form(pre) -> interneuron(post) excitatory drive (positive; lex is excitatory -> routes to g_e)
    n_drive = 0
    for r in np.asarray(wf_globals, dtype=np.int64):
        lo, hi = int(W.indptr[r]), int(W.indptr[r + 1])
        cols = np.asarray(W.indices[lo:hi])
        mask = np.isin(cols, inh)
        if mask.any():
            seg = np.asarray(W.data[lo:hi]); seg[mask] = abs(float(inh_drive)); W.data[lo:hi] = seg
            n_drive += int(mask.sum())
    # HOP 2: interneuron(pre) -> affix(post) inhibitory output (positive; inh_block is inhibitory -> routes to g_i)
    n_out = 0
    for r in inh:
        lo, hi = int(W.indptr[r]), int(W.indptr[r + 1])
        cols = np.asarray(W.indices[lo:hi])
        mask = np.isin(cols, affix)
        if mask.any():
            seg = np.asarray(W.data[lo:hi]); seg[mask] = abs(float(inhib_out)); W.data[lo:hi] = seg
            n_out += int(mask.sum())
    try:
        bridge._invalidate_coo_cache()
    except Exception:
        pass
    return n_drive, n_out


def _assign_patterns(bridge, seed, n_lex, n_proc, pattern_size, item2idx):
    """Give every item a GLOBAL-index sparse pattern in ITS pool: stems+whole-forms in LEX, PAST+AFFIX in PROC.
    Returns pg (item_idx -> global neuron indices)."""
    lex_items = list(IRREGULARS.keys()) + REG_TRAIN + REG_HELDOUT + NOVEL_STEMS + list(IRREGULARS.values())
    proc_items = [PAST, AFFIX]
    lex_base = np.asarray(bridge.region_manager.indices("lex"))
    proc_base = np.asarray(bridge.region_manager.indices("proc"))
    lex_pats = generate_sparse_patterns(len(lex_items), n_lex, pattern_size, seed)
    proc_pats = generate_sparse_patterns(len(proc_items), n_proc, pattern_size, seed + 777)
    n_items = len(item2idx)
    pg = [None] * n_items
    for k, name in enumerate(lex_items):
        pg[item2idx[name]] = lex_base[np.asarray(lex_pats[k])]
    for k, name in enumerate(proc_items):
        pg[item2idx[name]] = proc_base[np.asarray(proc_pats[k])]
    return pg


def complete_two_pool(bridge, pg, cue_idxs, competitor_idxs, window=40, pA=1100.0):
    """READ-ONLY (plasticity gate 0): drive the cue items -> accumulate GLOBAL pool firing (both pools) -> EXCLUDE
    the directly-driven cue neurons -> cosine of the recurrent+cross-pool completion to each competitor's global
    pattern. Returns {item_idx: score}."""
    _drive(bridge, [pg[i] for i in cue_idxs], pA)
    n = bridge.cp_firing_states.shape[0]
    firing = np.zeros(n)
    for _ in range(window):
        bridge._run_one_simulation_step()
        firing += np.asarray(to_host(bridge.cp_firing_states)).astype(float)
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(15):
        bridge._run_one_simulation_step()
    for i in cue_idxs:                                   # remove directly-driven cue -> pure completion
        firing[np.asarray(pg[i])] = 0.0
    nf = float(np.linalg.norm(firing))
    scores = {}
    for j in competitor_idxs:
        v = np.zeros(n)
        v[np.asarray(pg[j])] = 1.0
        scores[j] = float(firing @ v / (nf * np.linalg.norm(v))) if nf > 0 else 0.0
    return scores


# ---- anti-cheat (4): the two-pool build MUST seed the substrate from cfg.seed ------------------------------
def _threshold_hash(bridge):
    th = getattr(bridge, "cp_neuron_firing_thresholds", None)
    if th is None:
        return None
    arr = np.ascontiguousarray(np.asarray(to_host(th)))
    return hashlib.md5(arr.tobytes()).hexdigest()


def verify_seeded(seed, n_lex, n_proc):
    h1 = _threshold_hash(build_two_pool(seed, n_lex=n_lex, n_proc=n_proc))
    h2 = _threshold_hash(build_two_pool(seed, n_lex=n_lex, n_proc=n_proc))
    h3 = _threshold_hash(build_two_pool(seed + 9973, n_lex=n_lex, n_proc=n_proc))
    same = (h1 is not None and h1 == h2)
    diff = (h1 is not None and h1 != h3)
    return {"same_seed_identical": bool(same), "cross_seed_differs": bool(diff),
            "seeds_substrate": bool(same and diff), "threshold_hash": h1}


def run(seed, n_lex=2000, n_proc=800, pattern_size=90, cyc_rule=40, cyc_irr=48,
        inhib_strength=6.0, di_synaptic=False, n_inh_block=150, inh_drive=3.0, verbose=True):
    validate_execution_request("direct-run", seed, 1)
    items, item2idx = _make_items()
    idx2name = {i: w for w, i in item2idx.items()}
    competitors = [item2idx[AFFIX]] + [item2idx[wf] for wf in IRREGULARS.values()]

    def build_brain(permute_binding=False):
        b = build_two_pool(seed, n_lex=n_lex, n_proc=n_proc,
                            di_synaptic=di_synaptic, n_inh_block=n_inh_block)
        pg = _assign_patterns(b, seed, n_lex, n_proc, pattern_size, item2idx)
        # ROUTE 1 -- PROCEDURAL (PROC pool): PAST -> AFFIX, dedicated + strong, stem-independent.
        co_activate(b, pg, [item2idx[PAST], item2idx[AFFIX]], cycles=cyc_rule)
        # ROUTE 2 -- DECLARATIVE (LEX pool): stem -> whole-form per irregular, entrenched.
        wf_names = list(IRREGULARS.values())
        if permute_binding:                                  # ANTI-CHEAT (3): derange stem->whole-form binding
            rng = np.random.RandomState(seed * 101 + 7)
            perm = rng.permutation(len(wf_names))
            while np.any(perm == np.arange(len(wf_names))):
                perm = rng.permutation(len(wf_names))
            wf_for_stem = {stem: wf_names[perm[i]] for i, stem in enumerate(IRREGULARS.keys())}
        else:
            wf_for_stem = dict(IRREGULARS)
        for stem, wf in wf_for_stem.items():
            co_activate(b, pg, [item2idx[stem], item2idx[wf]], cycles=cyc_irr)
        # BLOCKING -- LEX whole-form -> PROC affix suppression (wire AFTER training).
        affix_glob = np.asarray(pg[item2idx[AFFIX]])
        wf_glob = np.concatenate([np.asarray(pg[item2idx[wf]]) for wf in IRREGULARS.values()])
        if di_synaptic:
            # FAITHFUL: whole-form(exc) -> dedicated GABAergic interneuron(inh) -> affix. Suppression scales with
            # retrieval; the minus sign is carried by the interneuron cell (Dale-compliant), not a flipped weight.
            inh_glob = np.asarray(b.region_manager.indices("inh_block"))
            n_drive, n_inh = wire_disynaptic_inhibition(b, wf_glob, inh_glob, affix_glob, inh_drive, inhib_strength)
        else:
            # S1 SHORTCUT: sign-inverted excitatory synapse whole-form -> affix (single scalar weight, byte-identical path).
            n_inh = wire_wf_to_affix_inhibition(b, wf_glob, affix_glob, inhib_strength)
        return b, pg, n_inh

    # ============================ MAIN BRAIN ============================
    b, pg, n_inh = build_brain(permute_binding=False)

    # (1) REGULAR PRODUCTIVITY -- held-out + novel stems must take "-ed" by RULE
    reg_probe = REG_HELDOUT + NOVEL_STEMS
    reg_ok = 0
    reg_detail = {}
    for stem in reg_probe:
        cue = [item2idx[stem], item2idx[PAST]]
        form, _ = _argmax_form(complete_two_pool(b, pg, cue, competitors), idx2name)
        ok = (form == AFFIX)
        reg_ok += int(ok)
        reg_detail[stem] = form
        if verbose:
            print(f"  REGULAR {stem:6s}+PAST -> '{stem}{form}' {'[rule -ed OK]' if ok else '[got '+form+']'}", flush=True)
    reg_acc = reg_ok / len(reg_probe)

    # (2) IRREGULAR BLOCKING -- the stored whole-form inhibits the affix and WINS
    irr_ok = 0
    irr_edrate_pre = 0
    irr_detail = {}
    for stem, wf in IRREGULARS.items():
        cue = [item2idx[stem], item2idx[PAST]]
        form, _ = _argmax_form(complete_two_pool(b, pg, cue, competitors), idx2name)
        ok = (form == wf)
        irr_ok += int(ok)
        irr_edrate_pre += int(form == AFFIX)
        irr_detail[stem] = form
        if verbose:
            print(f"  IRREG   {stem:6s}+PAST -> '{form}' (target '{wf}') "
                  f"{'[stored, BLOCKS -ed]' if ok else '[got '+form+']'}", flush=True)
    irr_acc = irr_ok / len(IRREGULARS)
    irr_edrate_pre /= len(IRREGULARS)

    # (3) LESION the LEX store -> OVER-REGULARIZATION (removes stem->wf retrieval AND wf->affix inhibition)
    wf_globals = np.concatenate([np.asarray(pg[item2idx[wf]]) for wf in IRREGULARS.values()])
    lesion_neurons(b, wf_globals)
    overreg = 0
    overreg_other = 0
    les_detail = {}
    for stem, wf in IRREGULARS.items():
        cue = [item2idx[stem], item2idx[PAST]]
        form, _ = _argmax_form(complete_two_pool(b, pg, cue, competitors), idx2name)
        les_detail[stem] = form
        if form == AFFIX:
            overreg += 1
        elif form != wf:
            overreg_other += 1
        if verbose:
            print(f"  LESION  {stem:6s}+PAST -> '{stem}{form if form==AFFIX else '/'+form}' "
                  f"{'[OVER-REGULARIZED]' if form==AFFIX else ''}", flush=True)
    overreg_rate = overreg / len(IRREGULARS)
    # ATTRIBUTION: the irregular blocking is OWNED by the lex whole-form store — lesioning it flips
    # irregulars to over-regularization (treatment) vs the intact -ed rate (control); the gap is the blocking.
    from tools.lab import attributable_to
    attributable_to("irr_blocking_owned_by_lex_store", treatment_value=overreg_rate, control_value=irr_edrate_pre)

    # ============================ PERMUTED-BINDING BRAIN (anti-cheat 3) ============================
    bp, pgp, _ = build_brain(permute_binding=True)
    perm_ok = 0
    for stem, wf in IRREGULARS.items():
        cue = [item2idx[stem], item2idx[PAST]]
        form, _ = _argmax_form(complete_two_pool(bp, pgp, cue, competitors), idx2name)
        perm_ok += int(form == wf)
    perm_acc = perm_ok / len(IRREGULARS)

    both_gates = (reg_acc >= 0.90 and irr_acc >= 0.85)
    go = (both_gates and overreg_rate >= 0.85
          and overreg_rate - irr_edrate_pre >= 0.5 and perm_acc <= 0.30)
    result = {
        "seed": int(seed), "n_lex": n_lex, "n_proc": n_proc, "pattern_size": pattern_size,
        "cyc_rule": cyc_rule, "cyc_irr": cyc_irr, "inhib_strength": inhib_strength,
        "di_synaptic": bool(di_synaptic), "n_inh_block": int(n_inh_block) if di_synaptic else 0,
        "inh_drive": float(inh_drive) if di_synaptic else None,
        "n_inhib_edges": int(n_inh),
        "reg_acc": reg_acc, "irr_acc": irr_acc, "both_gates": bool(both_gates),
        "irr_edrate_pre_lesion": irr_edrate_pre,
        "overreg_rate_lesion": overreg_rate, "lesion_other_error_rate": overreg_other / len(IRREGULARS),
        "permuted_binding_irr_acc": perm_acc,
        "reg_detail": reg_detail, "irr_detail": irr_detail, "lesion_detail": les_detail,
        "GO": bool(go),
    }
    if verbose:
        mech = f"di-synaptic(inh_block={n_inh_block},drive={inh_drive})" if di_synaptic else "sign-inverted"
        print(f"\n  seed {seed} [{mech}]: reg(rule) {reg_acc:.2f} & irr(blocking) {irr_acc:.2f} "
              f"{'[BOTH]' if both_gates else '[NOT both]'} | lesion->over-reg {overreg_rate:.2f} "
              f"(pre {irr_edrate_pre:.2f}) | permuted {perm_acc:.2f} | inhib_edges {n_inh} "
              f"| {'GO' if go else 'NO-GO'}", flush=True)
    return result


def summarize(base_seed, n_seeds=6, n_lex=2000, n_proc=800, pattern_size=90, cyc_rule=40, cyc_irr=48,
              inhib_strength=6.0, di_synaptic=False, n_inh_block=150, inh_drive=3.0, verbose=True):
    validate_execution_request("direct-summary", base_seed, n_seeds)
    seeds = [base_seed + i for i in range(n_seeds)]
    seed_check = verify_seeded(base_seed, n_lex, n_proc)
    if verbose:
        print(f"[seed anti-cheat] cfg.seed controls substrate: {seed_check['seeds_substrate']} "
              f"(same-seed identical={seed_check['same_seed_identical']}, "
              f"cross-seed differs={seed_check['cross_seed_differs']})", flush=True)
    results = [run(s, n_lex=n_lex, n_proc=n_proc, pattern_size=pattern_size, cyc_rule=cyc_rule,
                   cyc_irr=cyc_irr, inhib_strength=inhib_strength, di_synaptic=di_synaptic,
                   n_inh_block=n_inh_block, inh_drive=inh_drive, verbose=verbose) for s in seeds]

    n_reg = sum(1 for r in results if r["reg_acc"] >= 0.90)
    n_irr = sum(1 for r in results if r["irr_acc"] >= 0.85)
    n_both = sum(1 for r in results if r["both_gates"])
    n_go = sum(1 for r in results if r["GO"])
    go = (n_go >= 5) and seed_check["seeds_substrate"]
    if go:
        verdict = f"TWO-POOL DUAL-ROUTE GO -- reg AND blocking simultaneously ({n_both}/{n_seeds} both, {n_go}/{n_seeds} full-GO)"
    elif not seed_check["seeds_substrate"]:
        verdict = "NEGATIVE -- substrate NOT seeded by cfg.seed (anti-cheat 4 failed; confounded)"
    elif n_both < 5 and n_reg < 5 and n_irr >= 5:
        verdict = f"NEGATIVE -- the RULE still fails to generalize (reg_acc>=0.90 on only {n_reg}/{n_seeds})"
    elif n_both < 5 and n_irr < 5 and n_reg >= 5:
        verdict = f"NEGATIVE -- BLOCKING fails (irr_acc>=0.85 on only {n_irr}/{n_seeds})"
    elif n_both < 5:
        verdict = (f"NEGATIVE -- no op-point does BOTH (both {n_both}/{n_seeds}; reg {n_reg}, irr {n_irr}) "
                   f"-- sweep --inhib-strength / --cyc-rule")
    else:
        verdict = (f"NEGATIVE -- anti-cheats fail on >1 seed (both {n_both}/{n_seeds} but full-GO {n_go}/{n_seeds})")

    summary = {
        "probe": "productive_morphology_two_pool" + ("_disynaptic" if di_synaptic else ""),
        "config": {"n_seeds": n_seeds, "base_seed": base_seed, "n_lex": n_lex, "n_proc": n_proc,
                   "pattern_size": pattern_size, "cyc_rule": cyc_rule, "cyc_irr": cyc_irr,
                   "inhib_strength": inhib_strength, "di_synaptic": bool(di_synaptic),
                   "n_inh_block": n_inh_block, "inh_drive": inh_drive},
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
        print(f"\n=== two-pool morphology summary ({n_seeds} seeds) ===", flush=True)
        print(f"  reg_acc>=0.90: {n_reg}/{n_seeds} | irr_acc>=0.85: {n_irr}/{n_seeds} "
              f"| BOTH: {n_both}/{n_seeds} | full-GO: {n_go}/{n_seeds}", flush=True)
        print(f"  VERDICT: {verdict}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Retired two-pool morphology runner. No scientific phase is open; "
            "historical seeds are consumed."
        )
    )
    ap.add_argument(
        "--phase",
        required=True,
        help="required execution phase; currently no scientific phase is open",
    )
    ap.add_argument("--seed", type=int, default=42, help="base seed; the sweep uses seed..seed+n_seeds-1")
    ap.add_argument("--n-seeds", type=int, default=6)
    ap.add_argument("--n-lex", type=int, default=2000)
    ap.add_argument("--n-proc", type=int, default=800)
    ap.add_argument("--pattern-size", type=int, default=90)
    ap.add_argument("--cyc-rule", type=int, default=40, help="ROUTE 1 (PAST->AFFIX) strength")
    ap.add_argument("--cyc-irr", type=int, default=48, help="ROUTE 2 (stem->whole-form) entrenchment")
    ap.add_argument("--inhib-strength", type=float, default=6.0,
                    help="blocking strength: sign-inverted whole-form->affix weight, OR (di-synaptic) the "
                         "interneuron->affix inhibitory output weight")
    ap.add_argument("--di-synaptic", action="store_true",
                    help="FAITHFUL Dale-compliant blocking: whole-form(exc) -> GABAergic interneuron(inh) -> affix "
                         "(default OFF = byte-identical sign-inverted excitatory synapse, the S1 shortcut)")
    ap.add_argument("--n-inh-block", type=int, default=150, help="di-synaptic: interneuron pool size")
    ap.add_argument("--inh-drive", type=float, default=3.0,
                    help="di-synaptic: lex whole-form -> interneuron excitatory drive weight (fires the interneuron)")
    ap.add_argument("--out", default=None)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    validate_execution_request(a.phase, a.seed, a.n_seeds)
    mech = (f"DI-SYNAPTIC interneuron (n={a.n_inh_block}, drive={a.inh_drive})" if a.di_synaptic
            else "sign-inverted excitatory (S1 shortcut)")
    print(f"[two-pool morphology] DISTINCT populations (lex/proc) + {mech} blocking | base seed={a.seed} "
          f"n_seeds={a.n_seeds} inhib={a.inhib_strength}", flush=True)
    s = summarize(a.seed, n_seeds=a.n_seeds, n_lex=a.n_lex, n_proc=a.n_proc, pattern_size=a.pattern_size,
                  cyc_rule=a.cyc_rule, cyc_irr=a.cyc_irr, inhib_strength=a.inhib_strength,
                  di_synaptic=a.di_synaptic, n_inh_block=a.n_inh_block, inh_drive=a.inh_drive)
    print(f"\n  {'GO' if s['GO'] else 'NO-GO'} -- {s['verdict']}", flush=True)
    if a.out:
        if os.path.dirname(a.out):
            os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(s, open(a.out, "w"), indent=1)
        print(f"  wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
