"""KNOWLEDGE-half of breadth, rung 2 (SPIKING): property inheritance over REAL-corpus-discovered
categories, realized on the spiking substrate via the EMERGE-42 competitive pooler + the committed
HTM coincidence kernel -- NOT the rung-1 rate associative-memory read.

Rung 1 (2026-07-08, GO) showed the real-corpus RATE codes support held-out inheritance. This rung
realizes it ON SPIKES: the real-corpus co-occurrence codes (SDR-ified to their top-T active hubs)
drive the EMERGE-42 competitive self-organizing pooler (discovers shared category columns), a CLASS
property is taught on each category's NON-HELD member codons via the committed
`sim.kernels.fused_htm_permanence_update` three-term rule on a real SimulationBridge, and a HELD-OUT
member's category is read from the spiking apical drive (`cp_v_apical`) -- inheriting the class
property it was NEVER directly taught, via the shared pooler codon.

Reuse-by-import: EMERGE-42's spiking machinery (`apply_kernel_update`, `_prime_from_winners`, `_host`)
+ the breadth discovery (`discover_vocab`/`learn_stream_codes`/`build_probe`). NO `sim/` edit.

Anti-cheats (per seed):
  * DERANGED-labels (primary): reassign the SAME word SDRs to RANDOM categories -> the pooler codons no
    longer align with real categories -> held-out inheritance collapses to chance.
  * PERMUTED-features: replace each word's real SDR with a random SDR -> the pooler cannot discover
    categories -> collapse (isolates the pooler's role, EMERGE-42's own control).
  * LESION: coincidence detection off -> the kernel can't bind -> inheritance collapses.
GO = real held-out inheritance CLEARLY exceeds chance AND both controls, multi-seed.
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners
from research.runners._emergent_vocab_breadth_scale_derisk import (
    discover_vocab, learn_stream_codes, build_probe, TAXONOMY_8x8,
    STOPLIST, MIN_WORD_LEN, N_HUB, WINDOW,
)
from research.runners.corpus_stream import load_token_stream_multi

NCOL = 200
K_WIN = 8
POOL_EPOCHS = 400
POOL_LP = 0.05
POOL_LD = 0.02
N_ID_PER = 3
SDR_T = 50          # active hubs per word SDR (top-T of the code); 50 = the swept operating point
                    # (T=12 too sparse -> within-category SDR overlap too low for the pooler; T=150 over-dense)
PROP_K = 2          # cells per category property (population coding); 2 = the original operating point.
                    # Larger K averages more independent noisy apical readers of the same codon -> lower
                    # read variance -> sharper argmax (the on-substrate read-out-variance lever).
FLOOR = -40.0


def _sdr(cells):
    return set(int(c) for c in cells)


def _codes_to_sdr(codes, probe_rows, sdr_t=SDR_T):
    """Each probe word -> the set of its top-`sdr_t` most-active hub features (a sparse binary SDR)."""
    out = {}
    for r in probe_rows.tolist():
        v = codes[r]
        out[r] = set(int(h) for h in np.argsort(-v)[:sdr_t])
    return out


class RealCorpusPoolerProbe:
    """EMERGE-42's spiking pooler-inference, fed REAL-corpus SDR features + REAL categories."""

    def __init__(self, seed, sdr_by_row, row_to_cat, cat_ids, epochs=40, learn=True,
                 permute_features=False, lesion=False, prop_k=PROP_K, k_win=K_WIN,
                 diverse_readers=False, reader_frac=0.5):
        self.k_win = k_win                     # codon width (top-k pooler columns); the codon-side read-variance lever
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion
        from sim.enums import NeuronModel, NeuronType
        rng = np.random.default_rng(seed)

        NF = N_HUB
        rows = list(sdr_by_row)
        self.rows = rows
        NMEM = len(rows)
        NMEM_CELLS = NMEM * N_ID_PER
        self.cat_ids = list(cat_ids)
        NCAT = len(self.cat_ids)
        NCLASSPROP = NCAT
        NPROPUNITS = NCLASSPROP * prop_k       # prop_k cells per category property (population coding)
        FEAT0 = 0
        ID0 = NF
        COL0 = NF + NMEM_CELLS
        PROP0 = NF + NMEM_CELLS + NCOL
        M = PROP0 + NPROPUNITS
        self.NF, self.ID0, self.COL0, self.PROP0 = NF, ID0, COL0, PROP0
        self.ridx = {r: i for i, r in enumerate(rows)}

        # feature SDRs (real, or permuted-random for the control)
        if permute_features:
            self.feats = {r: set(int(x) for x in rng.choice(NF, SDR_T, replace=False)) for r in rows}
        else:
            self.feats = {r: set(sdr_by_row[r]) for r in rows}
        self.row2cat = dict(row_to_cat)

        regions = [BrainRegion(name="cells", n_neurons=M, exc_fraction=1.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                               plastic_internal=False,
                               izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)]
        cfg = CoreSimConfig()
        cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)
        cfg.dt_ms = 1.0
        cfg.num_traits = 1
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.connections_per_neuron = 0
        cfg.enable_brain_region_framework = True
        cfg.brain_regions = list(regions)
        cfg.region_pathways = []
        cfg.enable_stdp = False
        cfg.enable_hebbian_learning = False
        cfg.enable_nmda = False
        cfg.stdp_w_max = 1.0
        cfg.fast_spike_reset = True
        for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
                  "enable_conductance_noise", "enable_parameter_heterogeneity",
                  "enable_structural_plasticity"):
            setattr(cfg, f, False)
        cfg.enable_coincidence_detection = (not lesion)
        cfg.coincidence_weighted_drive = True
        cfg.coincidence_k_threshold = 1.5
        cfg.coincidence_plateau_strength = 160.0
        cfg.enable_two_compartment_dap = True
        cfg.apical_g_couple = 2.0
        b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                             runtime_state=RuntimeState(), gpu_config=GPUConfig())
        b.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        b.runtime_state.actual_seed_used = seed
        b._initialize_simulation_data(called_from_playback_init=False)
        ci = np.asarray(b.region_manager.indices("cells"), int)

        pre, post, w = [], [], []
        wrng = np.random.default_rng(seed * 7919 + 13)
        n_sub = max(1, int(round(NCOL * reader_frac)))
        for pc in range(NPROPUNITS):
            # DIVERSE READERS (true population coding): each property cell wires to a RANDOM SUBSET of columns,
            # so the prop_k cells per property have INDEPENDENT (not identical) reads of the codon -> averaging
            # the apical drive genuinely reduces variance (vs the CYCLE-958 no-op where all readers were identical).
            cols = wrng.choice(NCOL, n_sub, replace=False) if diverse_readers else range(NCOL)
            for c in cols:                                         # pooler columns -> property (class inheritance)
                pre.append(int(ci[COL0 + int(c)])); post.append(int(ci[PROP0 + pc])); w.append(0.0)
            for idx in range(NMEM_CELLS):                          # member-identity -> property (member-specific)
                pre.append(int(ci[ID0 + idx])); post.append(int(ci[PROP0 + pc])); w.append(0.0)
        b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                         "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
        coo = b._get_cached_coo()
        self.b, self.ci = b, ci
        self.row, self.col = np.asarray(_host(coo.row)), np.asarray(_host(coo.col))
        self.z = np.zeros(len(ci))

        # competitive self-organizing pooler (EMERGE-38) over the REAL SDR features
        self.Wp = rng.uniform(0.30, 0.55, (NCOL, NF))
        if learn:
            duty = np.zeros(NCOL); boost = np.ones(NCOL); order = list(rows)
            for e in range(POOL_EPOCHS):
                rng.shuffle(order)
                for r in order:
                    x = np.zeros(NF); x[list(self.feats[r])] = 1.0
                    win = np.argsort(-(((self.Wp > 0.5) @ x) * boost))[:self.k_win]
                    self.Wp[win] += POOL_LP * x - POOL_LD * (1 - x)
                    self.Wp[win] = np.clip(self.Wp[win], 0, 1); duty[win] += 1
                boost = np.exp(2.0 * (self.k_win / NCOL - duty / ((e + 1) * len(rows))))

        self.CLASS = {k: [PROP0 + prop_k * i + j for j in range(prop_k)]
                      for i, k in enumerate(self.cat_ids)}
        # held-out = last 2 members per category, EXCLUDED from CLASS teaching (genuine generalization)
        by_cat = {k: [r for r in rows if self.row2cat[r] == k] for k in self.cat_ids}
        self.held = {k: v[-2:] for k, v in by_cat.items() if len(v) >= 4}
        held_set = {r for v in self.held.values() for r in v}
        for _ in range(epochs):
            for k in self.cat_ids:
                for r in [rr for rr in by_cat[k] if rr not in held_set]:
                    apply_kernel_update(self.b, self.row, self.col, self.ci, self._codon(self.feats[r]),
                                        _sdr(self.CLASS[k]), self.z, 0.14, 0.02, 1.0)

    def _codon(self, feats):
        x = np.zeros(self.NF); x[list(feats)] = 1.0
        return set(self.COL0 + int(c) for c in np.argsort(-((self.Wp > 0.5) @ x))[:self.k_win])

    def _drive_to(self, row, targets):
        resp = self._codon(self.feats[row])
        if not resp:
            return None
        ab = np.zeros(len(self.ci), bool)
        for c in resp:
            ab[c] = True
        for j in range(N_ID_PER):
            ab[self.ID0 + self.ridx[row] * N_ID_PER + j] = True
        _prime_from_winners(self.b, self.ci, ab)
        vap = getattr(self.b, "cp_v_apical", None)
        if vap is None or np.asarray(_host(vap)).ndim == 0:
            return None
        vap = _host(vap)[self.ci]
        return {t: float(np.mean([vap[x] for x in u])) for t, u in targets.items()}

    def query(self, row):
        targets = {f"C{k}": self.CLASS[k] for k in self.cat_ids}
        dr = self._drive_to(row, targets)
        if dr is None:
            return None
        best = max(dr, key=dr.get)
        return best if dr[best] > FLOOR else None

    def inheritance_acc(self):
        checks = [self.query(r) == f"C{k}" for k in self.held for r in self.held[k]]
        return float(np.mean(checks)) if checks else float("nan")

    def n_categories(self):
        return len(self.held)


def build_inputs(corpus_path, K, seed, sdr_t=SDR_T):
    stories = load_token_stream_multi(corpus_path, max_stories=None)
    vocab, gfreq = discover_vocab(stories, K)
    target_set = set(vocab)
    hubs = []
    for w, _ in gfreq.most_common():
        if w in STOPLIST or w in target_set or len(w) < MIN_WORD_LEN:
            continue
        hubs.append(w)
        if len(hubs) >= N_HUB:
            break
    probe_rows, probe_labels, probe_words, n_cat, per_cat = build_probe(vocab, TAXONOMY_8x8)
    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
    sdr_by_row = _codes_to_sdr(codes, probe_rows, sdr_t=sdr_t)
    row_to_cat = {int(r): int(lab) for r, lab in zip(probe_rows.tolist(), probe_labels.tolist())}
    # usable categories: >=4 members
    from collections import Counter
    cnt = Counter(row_to_cat.values())
    cat_ids = sorted([k for k, v in cnt.items() if v >= 4])
    # restrict to usable-category rows
    keep = [r for r in sdr_by_row if row_to_cat[r] in cat_ids]
    sdr_by_row = {r: sdr_by_row[r] for r in keep}
    row_to_cat = {r: row_to_cat[r] for r in keep}
    return stories, sdr_by_row, row_to_cat, cat_ids, per_cat


def run_seed(seed, sdr_by_row, row_to_cat, cat_ids, epochs, rng, prop_k=PROP_K, k_win=K_WIN, diverse_readers=False):
    chance = 1.0 / len(cat_ids)
    main = RealCorpusPoolerProbe(seed, sdr_by_row, row_to_cat, cat_ids, epochs=epochs, prop_k=prop_k, k_win=k_win, diverse_readers=diverse_readers)
    ho = main.inheritance_acc()
    # DERANGED: shuffle category labels across the same word SDRs
    rows = list(sdr_by_row)
    labs = [row_to_cat[r] for r in rows]
    dl = list(labs); rng.shuffle(dl)
    deranged_map = {r: dl[i] for i, r in enumerate(rows)}
    der = RealCorpusPoolerProbe(seed, sdr_by_row, deranged_map, cat_ids, epochs=epochs, prop_k=prop_k, k_win=k_win, diverse_readers=diverse_readers).inheritance_acc()
    # PERMUTED-features: random SDRs -> pooler can't discover categories
    perm = RealCorpusPoolerProbe(seed, sdr_by_row, row_to_cat, cat_ids, epochs=epochs,
                                 permute_features=True, prop_k=prop_k, k_win=k_win, diverse_readers=diverse_readers).inheritance_acc()
    # LESION: coincidence off
    les = RealCorpusPoolerProbe(seed, sdr_by_row, row_to_cat, cat_ids, epochs=epochs,
                                lesion=True, prop_k=prop_k, k_win=k_win, diverse_readers=diverse_readers).inheritance_acc()
    return {"seed": seed, "n_categories": main.n_categories(), "chance": chance,
            "heldout_inherit_acc": ho, "deranged_acc": der, "permuted_feat_acc": perm,
            "lesion_acc": les}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--prop-k", type=int, default=PROP_K, help="cells per category property (population coding)")
    ap.add_argument("--k-win", type=int, default=K_WIN, help="codon width (top-k pooler columns) -- the codon-side lever")
    ap.add_argument("--diverse-readers", action="store_true", help="true population coding: each prop cell reads a random column subset")
    ap.add_argument("--sdr-t", type=int, default=SDR_T, help="active hubs per word SDR (top-T of the code)")
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    # inputs are seed-shuffled internally; build per-seed (codes depend on seed only via story order -> ~deterministic)
    recs = []
    for s in seeds:
        _, sdr_by_row, row_to_cat, cat_ids, per_cat = build_inputs(a.corpus_path, a.K, s, sdr_t=a.sdr_t)
        rng = np.random.default_rng(s)
        r = run_seed(s, sdr_by_row, row_to_cat, cat_ids, a.epochs, rng, prop_k=a.prop_k, k_win=a.k_win, diverse_readers=a.diverse_readers)
        recs.append(r)
        print(f"  [seed {s}] SPIKING held-out inherit={r['heldout_inherit_acc']:.3f} | "
              f"deranged={r['deranged_acc']:.3f} | permuted-feat={r['permuted_feat_acc']:.3f} | "
              f"lesion={r['lesion_acc']:.3f} | chance={r['chance']:.3f} (cats={r['n_categories']})", flush=True)

    def m(k): return float(np.nanmean([r[k] for r in recs]))
    ho, der, perm, les, ch = (m("heldout_inherit_acc"), m("deranged_acc"),
                              m("permuted_feat_acc"), m("lesion_acc"), m("chance"))
    beats_chance = all(r["heldout_inherit_acc"] - r["chance"] > a.margin for r in recs)
    beats_der = all(r["heldout_inherit_acc"] - r["deranged_acc"] > a.margin for r in recs)
    beats_perm = all(r["heldout_inherit_acc"] - r["permuted_feat_acc"] > a.margin for r in recs)
    go = beats_chance and beats_der and beats_perm
    verdict = "GO" if go else "NEGATIVE"
    print(f"\n  AGGREGATE ({len(recs)} seeds): SPIKING held-out inherit={ho:.3f} | deranged={der:.3f} | "
          f"permuted-feat={perm:.3f} | lesion={les:.3f} | chance={ch:.3f}", flush=True)
    print(f"  beats_chance={beats_chance} | beats_deranged={beats_der} | beats_permuted_feat={beats_perm}", flush=True)
    print(f"  VERDICT: {verdict} -- a HELD-OUT member of a REAL-corpus-DISCOVERED category "
          f"{'INHERITS its class property ON SPIKES' if go else 'does NOT clearly inherit on spikes'} "
          f"(EMERGE-42 pooler + committed HTM kernel; read from cp_v_apical), "
          f"{'above chance, label-derangement, and permuted-features -> the KNOWLEDGE half rides real-corpus '
             'breadth ON THE SPIKING SUBSTRATE' if go else 'within control range'}.", flush=True)
    if a.out:
        json.dump({"verdict": verdict, "K": a.K,
                   "aggregate": {"heldout": ho, "deranged": der, "permuted_feat": perm, "lesion": les, "chance": ch},
                   "beats_chance": beats_chance, "beats_deranged": beats_der, "beats_permuted": beats_perm,
                   "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
