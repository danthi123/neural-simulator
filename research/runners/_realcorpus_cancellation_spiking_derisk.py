"""KNOWLEDGE-half of breadth, CANCELLATION ON SPIKES: a member's OWN property overrides its category's
inherited one, realized on the spiking substrate (NOT the rung-4 rate associative memory).

The rate cancellation (`_realcorpus_cancellation_derisk`, 6-seed GO) rode the numpy associative memory.
The mission's non-negotiable is FULLY SPIKING on one brain, so this realizes cancellation on the same
spiking substrate as rung-2 inheritance (EMERGE-42 competitive pooler + the committed HTM coincidence
kernel + apical read from `cp_v_apical`):

  * INHERIT (rung-2): a held-out member's category is read from the codon->class-property apical drive.
  * CANCEL (this rung, EMERGE-54 apical competition): the exception member's IDENTITY ensemble is bound
    to a DEDICATED exception property with a stronger apical drive (regulated to override), so priming
    the exception member drives its OWN property ABOVE the codon-driven inherited class -> the apical
    argmax flips to the exception property. Other held-out members (no identity->exc binding) still
    inherit. The member-identity->property wiring the rung-2 substrate ALREADY builds carries it.

Reuse-by-import: rung-2's `apply_kernel_update`/`_prime_from_winners`/`_host` + the breadth discovery.
NO `sim/` edit.

Gates (per seed):
  * INHERIT: held-out class members (not the exception) -> their class property (rung-2 intact).
  * CANCEL: the exception member -> the EXCEPTION property (its own), NOT the inherited class.
  * CONTROL (load-bearing): BEFORE teaching the exception, the SAME member inherits its class ->
    proves the exception binding flips it (not a code artifact).
  * LESION: coincidence detection off -> the exception can't bind -> no override (falls back to inherit/none).
"""
from __future__ import annotations
import argparse
import json
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners
from research.runners._realcorpus_inheritance_rung2_spiking_derisk import (
    build_inputs, _codes_to_sdr, _sdr, NCOL, K_WIN, POOL_EPOCHS, POOL_LP, POOL_LD, N_ID_PER, PROP_K, FLOOR, SDR_T,
)
from research.runners._emergent_vocab_breadth_scale_derisk import (
    N_HUB, discover_vocab, learn_stream_codes, STOPLIST, MIN_WORD_LEN, WINDOW,
)
from research.runners._realcorpus_inheritance_emergent_clusters_derisk import _kmeans
from research.runners.corpus_stream import load_token_stream_multi


def emergent_inputs(corpus_path, K, seed, n_clusters):
    """Like build_inputs, but DISCOVER the categories by clustering the codes (NO taxonomy labels) ->
    fully-EMERGENT categories fed to the spiking probe. Removes the last hand-labeled scaffold on spikes."""
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
    codes, _ = learn_stream_codes(seed, stories, vocab, hubs, window=WINDOW)
    labels = _kmeans(codes, n_clusters, seed)
    from collections import Counter
    cnt = Counter(labels.tolist())
    cat_ids = sorted([c for c in cnt if cnt[c] >= 4])
    rows = [i for i in range(len(vocab)) if labels[i] in cat_ids]
    sdr_by_row = _codes_to_sdr(codes, np.asarray(rows))
    row_to_cat = {int(r): int(labels[r]) for r in rows}
    return sdr_by_row, row_to_cat, cat_ids


class CancellingPoolerProbe:
    """rung-2 spiking pooler-inference + a DEDICATED exception property bound to an exception member's
    identity ensemble (EMERGE-54 apical competition) -> member-specific cancellation ON SPIKES."""

    def __init__(self, seed, sdr_by_row, row_to_cat, cat_ids, epochs=40, lesion=False, prop_k=PROP_K, k_win=K_WIN):
        from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
        from sim.bridge import SimulationBridge
        from sim.regions import BrainRegion
        from sim.enums import NeuronModel, NeuronType
        rng = np.random.default_rng(seed)
        self.k_win = k_win
        NF = N_HUB
        rows = list(sdr_by_row); self.rows = rows
        NMEM = len(rows); NMEM_CELLS = NMEM * N_ID_PER
        self.cat_ids = list(cat_ids); NCAT = len(self.cat_ids)
        NPROPUNITS = NCAT * prop_k
        NEXC = 1                                    # one dedicated exception property (prop_k cells)
        NEXCUNITS = NEXC * prop_k
        FEAT0, ID0 = 0, NF
        COL0 = NF + NMEM_CELLS
        PROP0 = COL0 + NCOL
        EXC0 = PROP0 + NPROPUNITS
        M = EXC0 + NEXCUNITS
        self.NF, self.ID0, self.COL0, self.PROP0, self.EXC0 = NF, ID0, COL0, PROP0, EXC0
        self.prop_k = prop_k
        self.ridx = {r: i for i, r in enumerate(rows)}
        self.feats = {r: set(sdr_by_row[r]) for r in rows}
        self.row2cat = dict(row_to_cat)

        regions = [BrainRegion(name="cells", n_neurons=M, exc_fraction=1.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                               izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)]
        cfg = CoreSimConfig()
        cfg.seed = cfg.heterogeneity_seed = cfg.ou_seed = int(seed)
        cfg.dt_ms = 1.0; cfg.num_traits = 1
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"; cfg.connections_per_neuron = 0
        cfg.enable_brain_region_framework = True
        cfg.brain_regions = list(regions); cfg.region_pathways = []
        cfg.enable_stdp = False; cfg.enable_hebbian_learning = False; cfg.enable_nmda = False
        cfg.stdp_w_max = 1.0; cfg.fast_spike_reset = True
        for f in ("enable_homeostasis", "enable_short_term_plasticity", "enable_ou_process",
                  "enable_conductance_noise", "enable_parameter_heterogeneity", "enable_structural_plasticity"):
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
        for pc in range(NPROPUNITS):                          # pooler columns + identities -> CLASS property
            for c in range(NCOL):
                pre.append(int(ci[COL0 + c])); post.append(int(ci[PROP0 + pc])); w.append(0.0)
            for idx in range(NMEM_CELLS):
                pre.append(int(ci[ID0 + idx])); post.append(int(ci[PROP0 + pc])); w.append(0.0)
        for ec in range(NEXCUNITS):                           # identities -> EXCEPTION property (member-specific)
            for idx in range(NMEM_CELLS):
                pre.append(int(ci[ID0 + idx])); post.append(int(ci[EXC0 + ec])); w.append(0.0)
        b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                         "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
        coo = b._get_cached_coo()
        self.b, self.ci = b, ci
        self.row, self.col = np.asarray(_host(coo.row)), np.asarray(_host(coo.col))
        self.z = np.zeros(len(ci))

        # competitive self-organizing pooler (EMERGE-38) over the real SDR features
        self.Wp = rng.uniform(0.30, 0.55, (NCOL, NF))
        duty = np.zeros(NCOL); boost = np.ones(NCOL); order = list(rows)
        for e in range(POOL_EPOCHS):
            rng.shuffle(order)
            for r in order:
                x = self._x(r)
                win = np.argsort(-(((self.Wp > 0.5) @ x) * boost))[:self.k_win]
                self.Wp[win] += POOL_LP * x - POOL_LD * (1 - x)
                self.Wp[win] = np.clip(self.Wp[win], 0, 1); duty[win] += 1
            boost = np.exp(2.0 * (self.k_win / NCOL - duty / ((e + 1) * len(rows))))

        self.CLASS = {k: [PROP0 + prop_k * i + j for j in range(prop_k)] for i, k in enumerate(self.cat_ids)}
        self.EXC = [EXC0 + j for j in range(prop_k)]
        by_cat = {k: [r for r in rows if self.row2cat[r] == k] for k in self.cat_ids}
        self.held = {k: v[-2:] for k, v in by_cat.items() if len(v) >= 4}
        held_set = {r for v in self.held.values() for r in v}
        for _ in range(epochs):                               # teach CLASS properties (codon -> class), held-out excluded
            for k in self.cat_ids:
                for r in [rr for rr in by_cat[k] if rr not in held_set]:
                    apply_kernel_update(self.b, self.row, self.col, self.ci, self._codon(r),
                                        _sdr(self.CLASS[k]), self.z, 0.14, 0.02, 1.0)

    def _x(self, row):
        x = np.zeros(self.NF); x[list(self.feats[row])] = 1.0; return x

    def _codon(self, row):
        return set(self.COL0 + int(c) for c in np.argsort(-((self.Wp > 0.5) @ self._x(row)))[:self.k_win])

    def _identity(self, row):
        return set(self.ID0 + self.ridx[row] * N_ID_PER + j for j in range(N_ID_PER))

    def teach_exception(self, row, passes):
        """Bind the exception member's IDENTITY ensemble -> the dedicated exception property (EMERGE-54
        apical competition); `passes` = the regulated drive strength (more passes -> stronger override)."""
        for _ in range(passes):
            apply_kernel_update(self.b, self.row, self.col, self.ci, self._identity(row),
                                _sdr(self.EXC), self.z, 0.14, 0.02, 1.0)

    def _apical(self, row, targets):
        resp = self._codon(row)
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

    def query(self, row, include_exc=True):
        """apical argmax over class properties (+ the exception property if include_exc) -> label or None."""
        targets = {f"C{k}": self.CLASS[k] for k in self.cat_ids}
        if include_exc:
            targets["EXC"] = self.EXC
        dr = self._apical(row, targets)
        if dr is None:
            return None
        best = max(dr, key=dr.get)
        return best if dr[best] > FLOOR else None


def _adaptive_teach(con, exc_row, pos_label, max_passes=12):
    """Regulated graded drive on spikes: add exception-teaching passes until the exception member's apical
    argmax flips to EXC (its own property), capped. Returns the passes used (0 if already/never flips)."""
    for p in range(1, max_passes + 1):
        con.teach_exception(exc_row, 1)                       # +1 pass
        if con.query(exc_row) == "EXC":
            return p
    return max_passes


def run_seed(seed, sdr_by_row, row_to_cat, cat_ids, epochs, prop_k, k_win, max_passes=12):
    con = CancellingPoolerProbe(seed, sdr_by_row, row_to_cat, cat_ids, epochs=epochs, prop_k=prop_k, k_win=k_win)
    if not con.held:
        return None
    # pos = the category with a held-out member that INHERITS (so the exception flip is meaningful).
    pos, exc_row = None, None
    for k in con.held:
        for r in con.held[k]:
            if con.query(r, include_exc=False) == f"C{k}":     # inherits its class before any exception
                pos, exc_row = k, r; break
        if pos is not None:
            break
    if pos is None:
        return None
    inherit_before = con.query(exc_row) == f"C{pos}"          # expect True (inherits)
    # specificity over ALL held-out members (every category), not just pos's -> a strong collateral test.
    others = [r for k in con.held for r in con.held[k] if r != exc_row]
    before_others = {r: con.query(r) for r in others}

    passes = _adaptive_teach(con, exc_row, pos, max_passes=max_passes)               # regulated graded drive on spikes
    cancel = con.query(exc_row) == "EXC"                      # exception overrides -> own property
    not_class = con.query(exc_row) != f"C{pos}"               # and NOT the inherited class
    after_others = {r: con.query(r) for r in others}
    n_collateral = sum(1 for r in others if before_others[r] != after_others[r])

    # LESION: coincidence detection off -> the identity->exc binding cannot drive the apical -> no override.
    con_l = CancellingPoolerProbe(seed, sdr_by_row, row_to_cat, cat_ids, epochs=epochs, prop_k=prop_k, k_win=k_win)
    con_l.b.core_config.enable_coincidence_detection = False   # ablate coincidence on the SAME bridge state
    lesion_passes = _adaptive_teach(con_l, exc_row, pos, max_passes=max_passes)
    lesion_override = con_l.query(exc_row) == "EXC"            # expect False (can't bind without coincidence)

    return {"seed": seed, "pos": int(pos), "passes": int(passes),
            "inherit_before": bool(inherit_before), "cancel": bool(cancel), "not_class": bool(not_class),
            "n_others": len(others), "n_collateral": int(n_collateral),
            "lesion_override": bool(lesion_override)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", default="data/corpus/tinystories.txt")
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--prop-k", type=int, default=PROP_K)
    ap.add_argument("--k-win", type=int, default=K_WIN)
    ap.add_argument("--emergent", action="store_true", help="DISCOVER categories by clustering (no taxonomy labels)")
    ap.add_argument("--n-clusters", type=int, default=10)
    ap.add_argument("--max-passes", type=int, default=12, help="regulated-drive ceiling (noisier emergent codes need more)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    print(f"[cancellation ON SPIKES] corpus={a.corpus_path} K={a.K} prop_k={a.prop_k} "
          f"{'EMERGENT-clusters' if a.emergent else 'taxonomy-labels'}", flush=True)

    recs = []
    for s in seeds:
        if a.emergent:
            sdr_by_row, row_to_cat, cat_ids = emergent_inputs(a.corpus_path, a.K, s, a.n_clusters)
        else:
            _, sdr_by_row, row_to_cat, cat_ids, per_cat, _ = build_inputs(a.corpus_path, a.K, s)
        r = run_seed(s, sdr_by_row, row_to_cat, cat_ids, a.epochs, a.prop_k, a.k_win, max_passes=a.max_passes)
        if r is None:
            print(f"  [seed {s}] not evaluable (no inheriting held-out member)", flush=True); continue
        recs.append(r)
        print(f"  [seed {s}] pos={r['pos']} passes={r['passes']} | inherit_before={r['inherit_before']} -> "
              f"CANCEL={r['cancel']} not_class={r['not_class']} | collateral={r['n_collateral']}/{r['n_others']} | "
              f"lesion_override={r['lesion_override']}", flush=True)

    if not recs:
        print("  VERDICT: NOT-EVALUABLE"); return
    cancel_ok = all(r["inherit_before"] and r["cancel"] and r["not_class"] for r in recs)
    no_collat = all(r["n_collateral"] == 0 for r in recs)
    lesion_ok = all(not r["lesion_override"] for r in recs)             # coincidence off -> no override
    go = cancel_ok and no_collat and lesion_ok
    print(f"\n  AGGREGATE ({len(recs)} seeds): CANCEL(inherit->EXC) all={cancel_ok} | no-collateral all={no_collat} | "
          f"lesion-no-override all={lesion_ok}", flush=True)
    print(f"  VERDICT: {'GO' if go else 'NEGATIVE'} -- a member's OWN property OVERRIDES its inherited class "
          f"{'ON SPIKES (EMERGE-54 apical competition; identity->exc drive beats codon->class, read from cp_v_apical), '
             'other members still inherit' if go else 'does NOT cleanly cancel on spikes'}.", flush=True)
    if a.out:
        json.dump({"verdict": "GO" if go else "NEGATIVE", "K": a.K, "per_seed": recs}, open(a.out, "w"), indent=2)
        print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
