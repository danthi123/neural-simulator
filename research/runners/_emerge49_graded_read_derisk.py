"""EMERGE-49 / toward-semantics — SURPASS the EMERGE-46 fully-spiking-stacked-pooler BOUNDARY via a GRADED DRIVE/READ on
the on-substrate L2 pooler (rung b of the EMERGE-48 boundary's identified next rungs; the CHEAPEST). EMERGE-46/47/48
PRECISELY ISOLATED the residual: the fully-spiking STACKED pooler fails held-out generalization because the on-substrate
L2 pooler has NO graded soft-pooling WINDOW — it jumps from over-selective (super-acc 0.03) straight to indiscriminate
collision (super-acc 0.53 ~= chance) as the winner-inactive depression drops, SKIPPING numpy's clean window
(super-acc 0.06 -> 1.00 across ld=0.15 -> 0.005). The SUSPECTED cause (EMERGE-48 §"genuine residual"): the on-substrate
drive/winner-read uses a HARD `perm > 0.5` connected-threshold (a sharp connected/not-connected split), whereas numpy
`(W>0.5)@x` COMBINED WITH its update trajectory keeps a graded pooling window. The EMERGE-47/48 isolation diagnostic
confirmed the residual is the on-substrate pooler's LEARNED-representation + READ (feeding the SAME good numpy L1 codons,
numpy L2 recovers to held-within 0.483 while on-substrate L2 stays 0.012), NOT the L1 codon quality.

THE MECHANISM TO DE-RISK (rung b): give the on-substrate L2 pooler a GRADED drive read so a soft-pooling window exists.
Concretely, three single-variable variants:
  (1) GRADED WINNER-SELECTION DRIVE ("graded_drive"): the L2 winner-selection drive is the GRADED raw-permanence-weighted
      sum of active-input synapses — `sum(perm * x)` rather than the thresholded connected-count `sum((perm>0.5) * x)`.
      Because the drive steers BOTH training winner-selection AND the codon read, partially-connected shared columns
      compete AND contribute -> graded competition, a possible soft window.
  (2) GRADED CODON READ ("graded_read"): TRAIN with the vanilla hard-threshold drive (== EMERGE-46), but READ the final
      L2 codon by ranking columns via the GRADED permanence-weighted drive. So partially-connected shared columns still
      contribute to held-out routing even if training was hard-thresholded.
  (3) BOTH ("graded_both"): graded drive during training AND graded read.

FIRST DIAGNOSE (the load-bearing honesty check, EMERGE-48 §next-rung): is the on-substrate LEARNED L2 permanence set
GRADED (a spread of values in [0,1]) or BIMODAL (clustered at 0 and 1)? Print the permanence histogram after L2 learning.
IF graded-under-the-threshold -> a graded read should reproduce numpy's soft window and reach GO. IF genuinely bimodal ->
the graded read cannot help (report that honestly; the Foldiak trace rule = rung a is then the next mechanism).

ANTI-CHEATS (mirror EMERGE-44/46/47/48 exactly; all must still hold): held out ENTIRE SUB-CATEGORIES {2,5} (a held-out
member can inherit ONLY via the L2-DISCOVERED grouping); PERMUTED-co-occurrence collapses (no superordinate structure);
dAP-LESION collapses (coincidence read is load-bearing); l2lesion is REPORTED-not-gated (a fixed-random control, per the
anti-cheat control-validity methodology). CRITICAL (the shortcut guard): the graded read must NOT just raise cross-super
overlap equally — WITHIN-super held-out overlap must rise while CROSS-super stays LOW (else it is indiscriminate collision,
not generalization). GATE: super-acc >= 0.80 AND >= permuted + 0.25 AND >= dAP-lesion + 0.30 AND held-within > held-cross
by a clear margin (>= 0.05).

Reuse-by-import (`_emerge44` task constants + numpy pooler; `_emerge46` `OnSubstratePooler` + bridge; `_emerge47`
`compute_idf_weights`; `_emerge14`/`_emerge12` kernels); NO NEW `sim/` edit (the graded drive/read is a HOST-side read of
`cp_connections.data`, using the raw permanence value instead of the `>0.5` threshold — the learning kernels are byte-
unchanged); CPU numpy-backend; 3-seed (42/43/44). `--demo`, `--diag` (histogram + numpy-parity, fast), `--onsubstrate`
(the decisive port; slow bridge builds).
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import apply_kernel_update, _host
from research.runners._emerge12_stageB2_bridge_tm_derisk import _prime_from_winners
from research.runners._emerge44_stacked_pooler_derisk import (
    SUBCATS, SUPER, NSUPER, POOLS, NCOL1, NCOL2, K1, K2,
    POOL_EPOCHS, N_PER, HELD_SUB, FLOOR, NPROPUNITS,
    _sdr,
)
from research.runners._emerge47_l2_input_normalization_derisk import compute_idf_weights

OUT = Path("research/findings/raw/_emerge49_graded_read.json")

# The GRADED-READ variants. "graded_drive" = graded during training AND read; "graded_read" = hard train, graded read;
# "graded_both" == "graded_drive"; "hard" == EMERGE-46 exactly (the boundary control).
GRADED_MODES = ("hard", "graded_read", "graded_drive")
# The soft/union L2 winner-inactive depression rate to pair with the graded read (a graded read gives a soft window only
# if the depression is also soft enough not to over-sparsify; the EMERGE-48 numpy sweet spot was ld~0.005-0.02).
L2_LD_DEFAULT = 0.005


# =====================================================================================================================
# ON-SUBSTRATE PORT — EMERGE-46 OnSubstratePooler with a GRADED (raw-permanence) drive/read
# =====================================================================================================================
def _build_onsubstrate_probe():
    """Lazy import of EMERGE-46's on-substrate pooler (slow bridge builds); returns (Probe, GradedPooler) where the L2
    pooler reads the raw permanence value instead of the `>0.5` connected-threshold (graded drive/read)."""
    from research.runners._emerge46_spiking_stacked_pooler_derisk import (
        OnSubstratePooler, _build_cells_bridge, M_INHERIT, POOL_LD_WI,
        NCOL1 as E46_NCOL1, NCOL2 as E46_NCOL2, NF as E46_NF, K1 as E46_K1, K2 as E46_K2,
        POOL_EPOCHS as E46_POOL_EPOCHS, L2_EPOCHS as E46_L2_EPOCHS,
    )

    class GradedOnSubstratePooler(OnSubstratePooler):
        """EMERGE-46's on-substrate pooler with (1) an optional per-input-column LOCAL-NORMALIZATION vector applied to
        the drive (EMERGE-47; in_weights=None disables it), (2) the SOFT winner-inactive depression rate `ld_wi` from
        the constructor, and (3) the GRADED-READ variant: `graded_drive` uses the raw permanence-weighted sum for BOTH
        the training winner-selection drive AND the codon read; `graded_read` uses the vanilla hard-threshold drive for
        TRAINING but the graded drive for the final codon read; `hard` == EMERGE-46 exactly.

        The learning kernels are BYTE-UNCHANGED — the graded read only changes HOW `cp_connections.data` is READ into a
        column drive (raw perm vs `perm>0.5`); the potentiation/depression still use the binary active mask. So this is
        NO NEW `sim/` edit (a host-side read change). `graded='hard' + in_weights=None + ld_wi=POOL_LD_WI` == EMERGE-46.
        """

        def __init__(self, *args, in_weights=None, graded="hard", **kwargs):
            super().__init__(*args, **kwargs)
            self.in_weights = None if in_weights is None else np.asarray(in_weights, float)
            self.graded = graded

        def _drive_common(self, feats, boost, graded_now):
            """Shared drive: `graded_now=False` -> the vanilla hard-threshold `(perm>0.5)` connected-count drive
            (== EMERGE-46); `graded_now=True` -> the raw permanence-weighted sum `perm * x` (graded soft-pooling)."""
            active = np.zeros(self.n_in); active[list(feats)] = 1.0
            if self.in_weights is not None:
                active = active * self.in_weights                 # <-- L2-INPUT LOCAL NORMALIZATION (EMERGE-47), optional
            data = _host(self.b.cp_connections.data)
            perm = data[self.ff_pos]
            gate = perm if graded_now else (perm > 0.5).astype(float)
            contrib = active[self.ff_feat] * gate
            drive = np.zeros(self.n_col); np.add.at(drive, self.ff_col, contrib)
            return drive * boost if boost is not None else drive

        def _drive(self, feats, boost=None):
            # TRAINING winner-selection drive: graded iff the variant grades the DRIVE (graded_drive / graded_both).
            return self._drive_common(feats, boost, graded_now=(self.graded == "graded_drive"))

        def codon(self, feats):
            # FINAL codon read: graded for BOTH graded_read and graded_drive; hard only for 'hard'.
            graded_now = self.graded in ("graded_read", "graded_drive")
            return set(int(c) for c in np.argsort(-self._drive_common(feats, None, graded_now))[:self.k_win])

    class GradedSpikingStackedPoolerProbe:
        def __init__(self, seed=42, epochs=40, lesion=False, permute=False, l2_lesion=False,
                     normalize=False, permute_stats=False, graded="graded_read", l2_ld=L2_LD_DEFAULT):
            self.graded = graded; self.l2_ld = l2_ld
            self.mem = {f"{k}_{i}": k for k in SUBCATS for i in range(N_PER)}
            self.feats = {}
            for i, (m, k) in enumerate(self.mem.items()):
                r = np.random.default_rng(seed * 100 + i)
                self.feats[m] = set(r.choice(POOLS[k], 4, replace=False))
            # L1: on-substrate pooler at the NORMAL discriminative ld_wi + HARD read (L1 discrimination is fine; only L2
            # needs the graded soft-pooling for invariance) -> sub-category codons
            l1 = OnSubstratePooler(seed=seed, n_in=E46_NF, n_col=E46_NCOL1, k_win=E46_K1)
            l1.train([self.feats[m] for m in self.mem], E46_POOL_EPOCHS, seed)
            self.l1codon = {m: l1.codon(self.feats[m]) for m in self.mem}
            members = list(self.mem)
            cooc = []
            rr = np.random.default_rng(seed * 3 + 7)
            for _ in range(240):
                if permute:
                    a, b = rr.choice(members, 2, replace=False)
                else:
                    sup = int(rr.integers(NSUPER))
                    pool = [m for m in members if SUPER[self.mem[m]] == sup]
                    a, b = rr.choice(pool, 2, replace=False)
                cooc.append(self.l1codon[a] | self.l1codon[b])
            in_w = None
            if normalize:
                ss = (seed * 17 + 5) if permute_stats else None
                in_w = compute_idf_weights(cooc, E46_NCOL1, shuffle_stats_seed=ss)
            # L2: on-substrate pooler with the GRADED drive/read (+ soft ld_wi + optional normalization)
            self.l2 = GradedOnSubstratePooler(seed=seed + 1, n_in=E46_NCOL1, n_col=E46_NCOL2, k_win=E46_K2,
                                              ld_wi=l2_ld, in_weights=in_w, graded=graded)
            if not l2_lesion:
                self.l2.train(cooc, E46_L2_EPOCHS, seed + 1)
            self.l2codon = {m: self.l2.codon(self.l1codon[m]) for m in self.mem}
            self._build_inherit_bridge(seed, lesion)
            self.SPROP = {s: [E46_NCOL2 + 2 * s, E46_NCOL2 + 2 * s + 1] for s in range(NSUPER)}
            self.held = {s: [] for s in range(NSUPER)}
            train = {s: [] for s in range(NSUPER)}
            for k in SUBCATS:
                ms = [m for m in self.mem if self.mem[m] == k]
                tgt = self.held if k in HELD_SUB else train
                for m in ms:
                    tgt[SUPER[k]].append(m)
            for _ in range(epochs):
                for s in range(NSUPER):
                    for m in train[s]:
                        apply_kernel_update(self.b, self.row, self.col, self.ci,
                                            _sdr(self.l2codon[m]), _sdr(self.SPROP[s]), self.z, 0.14, 0.02, 1.0)

        def _build_inherit_bridge(self, seed, lesion):
            b, ci = _build_cells_bridge(seed, M_INHERIT, coincidence=(not lesion))
            pre, post, w = [], [], []
            for pc in range(NPROPUNITS):
                for c in range(E46_NCOL2):
                    pre.append(int(ci[c])); post.append(int(ci[E46_NCOL2 + pc])); w.append(0.0)
            b.inject_explicit_wiring({"ff": {"pre_indices": pre, "post_indices": post, "initial_weights": w,
                                             "plastic": False, "coincidence_detector": True, "conn_type": "ff"}})
            coo = b._get_cached_coo()
            self.b, self.ci, self.row, self.col = b, ci, np.asarray(_host(coo.row)), np.asarray(_host(coo.col))
            self.z = np.zeros(len(ci))

        def infer_super(self, member):
            codon = self.l2codon[member]
            if not codon:
                return -1
            ab = np.zeros(len(self.ci), bool)
            for c in codon:
                ab[c] = True
            _prime_from_winners(self.b, self.ci, ab)
            vap = getattr(self.b, "cp_v_apical", None)
            if vap is None or np.asarray(_host(vap)).ndim == 0:
                return -1
            vap = _host(vap)[self.ci]
            dr = {s: float(np.mean([vap[x] for x in u])) for s, u in self.SPROP.items()}
            bs = max(dr, key=dr.get)
            return bs if dr[bs] > FLOOR else -1

        def held_out_super_acc(self):
            return np.mean([self.infer_super(m) == s for s in range(NSUPER) for m in self.held[s]])

        def held_out_within_cross_overlap(self):
            train_ms = {s: [m for m in self.mem if self.mem[m] not in HELD_SUB and SUPER[self.mem[m]] == s]
                        for s in range(NSUPER)}
            within, cross = [], []
            for s in range(NSUPER):
                for hm in self.held[s]:
                    for tm in train_ms[s]:
                        within.append(len(self.l2codon[hm] & self.l2codon[tm]) / E46_K2)
                    for so in range(NSUPER):
                        if so == s:
                            continue
                        for tm in train_ms[so]:
                            cross.append(len(self.l2codon[hm] & self.l2codon[tm]) / E46_K2)
            return float(np.mean(within)) if within else 0.0, float(np.mean(cross)) if cross else 0.0

        def l2_grouping(self):
            within, cross = [], []
            ms = list(self.mem)
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    ov = len(self.l2codon[ms[i]] & self.l2codon[ms[j]]) / E46_K2
                    (within if SUPER[self.mem[ms[i]]] == SUPER[self.mem[ms[j]]] else cross).append(ov)
            return float(np.mean(within) - np.mean(cross))

        def l2_permanence_hist(self, bins=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
            """The load-bearing DIAGNOSTIC: histogram of the LEARNED L2 feat->col permanences (ONLY the ff synapses),
            to distinguish GRADED (a spread across [0,1]) from BIMODAL (clustered at 0 and 1). A high fraction in the
            middle bins (0.2-0.8) => graded => a graded read can work; a high fraction only at the extremes => bimodal
            => the graded read cannot rescue it (report honestly)."""
            perm = _host(self.l2.b.cp_connections.data)[self.l2.ff_pos]
            counts, _ = np.histogram(perm, bins=list(bins))
            total = max(int(counts.sum()), 1)
            frac = (counts / total).tolist()
            mid = float(np.mean((perm >= 0.2) & (perm <= 0.8)))   # fraction in the graded middle band
            near0 = float(np.mean(perm < 0.05)); near1 = float(np.mean(perm > 0.95))
            return {"bin_edges": list(bins), "frac": frac, "mid_frac": mid, "near0_frac": near0,
                    "near1_frac": near1, "mean": float(np.mean(perm)), "std": float(np.std(perm)),
                    "bimodal": bool((near0 + near1) > 0.60 and mid < 0.25)}

    return GradedSpikingStackedPoolerProbe, GradedOnSubstratePooler


def _onsubstrate_run(seeds=(42, 43, 44), epochs=40, l2_ld=L2_LD_DEFAULT, normalize=False,
                     graded="graded_read", verbose=True):
    """Port to the on-substrate pooler: the GRADED drive/read (+ optional normalization) with the full anti-cheat arms,
    3-seed. Compare super-acc vs EMERGE-46's 0.03. Also returns the permanence histogram for the stacked arm."""
    Probe, _ = _build_onsubstrate_probe()
    arms = {
        "stacked_graded": dict(),
        "permuted_cooc": dict(permute=True),
        "dap_lesion": dict(lesion=True),
        "l2lesion": dict(l2_lesion=True),                                        # reported-not-gated
    }
    if normalize:
        arms["permuted_stats"] = dict(permute_stats=True)
    rows = {}
    hist = None
    for name, kw in arms.items():
        wi, cr, acc, grp = [], [], [], []
        for s in seeds:
            p = Probe(seed=s, epochs=epochs, graded=graded, l2_ld=l2_ld, normalize=normalize, **kw)
            w, c = p.held_out_within_cross_overlap()
            wi.append(w); cr.append(c); acc.append(float(p.held_out_super_acc())); grp.append(p.l2_grouping())
            if name == "stacked_graded" and hist is None:
                hist = p.l2_permanence_hist()                                    # histogram from the first stacked seed
            if verbose:
                print(f"    [{name} seed {s}] within {w:.3f} cross {c:.3f} super-acc {acc[-1]:.2f} L2-group {grp[-1]:+.2f}",
                      flush=True)
        rows[name] = {"held_within": float(np.mean(wi)), "held_cross": float(np.mean(cr)),
                      "super_acc": float(np.mean(acc)), "l2_group": float(np.mean(grp)),
                      "super_acc_per_seed": acc, "held_within_per_seed": wi, "held_cross_per_seed": cr}
    rows["_l2_perm_hist"] = hist
    return rows


# =====================================================================================================================
def _diag(seed=42, epochs=40, l2_ld=L2_LD_DEFAULT):
    """FAST diagnostic (single seed): print the L2 permanence histogram (graded vs bimodal) for the HARD-trained pooler,
    and the held-out within/cross overlap for each graded mode, so the graded-read hypothesis is visible before the full
    3-seed port. This is the load-bearing honesty check."""
    Probe, _ = _build_onsubstrate_probe()
    print(f"\n=== EMERGE-49 DIAG (seed {seed}, L2 ld_wi={l2_ld}) ===", flush=True)
    for mode in GRADED_MODES:
        p = Probe(seed=seed, epochs=epochs, graded=mode, l2_ld=l2_ld)
        w, c = p.held_out_within_cross_overlap(); acc = float(p.held_out_super_acc()); grp = p.l2_grouping()
        h = p.l2_permanence_hist()
        print(f"  [{mode:<12}] held-within {w:.3f}  held-cross {c:.3f}  super-acc {acc:.2f}  L2-group {grp:+.2f}", flush=True)
        if mode == "hard":
            print(f"     L2 permanence hist (learned, {mode}): mean {h['mean']:.3f} std {h['std']:.3f}  "
                  f"mid-band(0.2-0.8) {h['mid_frac']:.2f}  near0 {h['near0_frac']:.2f}  near1 {h['near1_frac']:.2f}  "
                  f"bimodal={h['bimodal']}", flush=True)
            print(f"     bins {['%.1f'%b for b in h['bin_edges']]}", flush=True)
            print(f"     frac {['%.2f'%f for f in h['frac']]}", flush=True)
    print()


def _verdict(onsub, l2_ld, graded, normalize):
    """Compose the GO/BOUNDARY verdict. The load-bearing test is the ON-SUBSTRATE stacked_graded arm vs 0.80 + controls +
    the within>cross discrimination guard, informed by the permanence histogram."""
    st = onsub["stacked_graded"]; perm = onsub["permuted_cooc"]; dap = onsub["dap_lesion"]
    l2l = onsub["l2lesion"]["super_acc"]; hist = onsub.get("_l2_perm_hist") or {}
    pstats = onsub.get("permuted_stats", {"super_acc": 0.0})
    acc = st["super_acc"]; grp = st["l2_group"]
    disc = bool(st["held_within"] - st["held_cross"] >= 0.05)
    gate_go = bool(acc >= 0.80 and acc >= perm["super_acc"] + 0.25 and acc >= dap["super_acc"] + 0.30)
    if normalize:
        gate_go = gate_go and bool(acc >= pstats["super_acc"] + 0.20)
    onsub_go = bool(gate_go and disc)

    hist_txt = ""
    if hist:
        hist_txt = (f" L2-permanence histogram (learned): mean {hist.get('mean', 0):.3f}, mid-band(0.2-0.8) "
                    f"{hist.get('mid_frac', 0):.2f}, near0 {hist.get('near0_frac', 0):.2f}, near1 "
                    f"{hist.get('near1_frac', 0):.2f} -> {'BIMODAL' if hist.get('bimodal') else 'GRADED'}.")

    pstats_txt = f"; permuted-stats {pstats['super_acc']:.2f}" if normalize else ""
    if onsub_go:
        verdict = (f"GO -- a GRADED DRIVE/READ on the on-substrate L2 pooler ({graded}, ld_wi={l2_ld}"
                   f"{' + L2-input normalization' if normalize else ''}) SURPASSES the EMERGE-46 fully-spiking-stacked-pooler "
                   f"BOUNDARY. Reading the raw permanence-weighted drive (instead of the hard `perm>0.5` connected-threshold) "
                   f"reproduces numpy's SOFT-POOLING WINDOW on-substrate: partially-connected shared columns contribute to a "
                   f"held-out sub-category's L2 code, so it SHARES its same-superordinate columns and inherits -- on-substrate "
                   f"super-acc {acc:.2f} (vs EMERGE-46's 0.03, chance {1/NSUPER:.2f}), held-out within-super L2 overlap "
                   f"{st['held_within']:.3f} > cross-super {st['held_cross']:.3f} (GENERALIZATION, not indiscriminate "
                   f"collision), L2-group {grp:+.2f}.{hist_txt} GATED CONTROLS: PERMUTED-co-occurrence {perm['super_acc']:.2f} "
                   f"(input-destruction collapses); dAP-LESION {dap['super_acc']:.2f} (coincidence read load-bearing)"
                   f"{pstats_txt}. REPORTED-secondary: L1->L2 lesion {l2l:.2f}. => the EMERGE-46 residual was the HARD "
                   f"connected-threshold READ (no graded soft-pooling window), fixed by a graded permanence read -- NOT an "
                   f"irreducible point-neuron limit. NO NEW sim/ edit (a host-side read of cp_connections.data using the raw "
                   f"permanence instead of >0.5; the learning kernels are byte-unchanged). 3-seed (42/43/44); 6-seed is a "
                   f"cheap confirmation follow-on.")
    else:
        miss = []
        if acc < 0.80: miss.append(f"super-acc {acc:.2f} < 0.80")
        if acc < perm["super_acc"] + 0.25: miss.append(f"permuted didn't collapse ({acc:.2f} vs {perm['super_acc']:.2f})")
        if acc < dap["super_acc"] + 0.30: miss.append(f"dAP-lesion didn't collapse ({acc:.2f} vs {dap['super_acc']:.2f})")
        if not disc: miss.append(f"no within>cross discrimination (within {st['held_within']:.3f} vs cross "
                                 f"{st['held_cross']:.3f} = collision, not generalization)")
        bimodal_note = ""
        if hist.get("bimodal"):
            bimodal_note = (" ROOT CAUSE (the histogram): the learned on-substrate L2 permanences are BIMODAL (clustered near "
                            "0/1), so a graded read cannot recover a soft-pooling window -- the potentiation drives winner "
                            "permanences to a sharp connected/not-connected split, exactly as EMERGE-48 predicted. The graded "
                            "read is NOT the fix; the next rung is the Foldiak (1991) trace / temporal-continuity rule (rung a).")
        else:
            bimodal_note = (" The learned permanences are GRADED (a spread across [0,1]) yet the graded read still does not "
                            "route held-out inheritance cleanly -- the residual is deeper than the read threshold (the "
                            "learned column TUNING, not merely its readout). Next rung: the Foldiak trace rule (rung a).")
        verdict = (f"BOUNDARY (on-substrate) -- a GRADED DRIVE/READ ({graded}, ld_wi={l2_ld}"
                   f"{' + normalization' if normalize else ''}) does NOT reach the on-substrate GO: " + "; ".join(miss) +
                   f". on-substrate stacked super-acc {acc:.2f} (EMERGE-46 was 0.03), within {st['held_within']:.3f} "
                   f"cross {st['held_cross']:.3f}; permuted-cooc {perm['super_acc']:.2f}, dAP-lesion {dap['super_acc']:.2f}."
                   f"{hist_txt}{bimodal_note} NO NEW sim/ edit.")
    flags = {"acc": acc, "grp": grp, "perm": perm["super_acc"], "dap": dap["super_acc"], "disc": disc,
             "within": st["held_within"], "cross": st["held_cross"], "onsub_go": onsub_go,
             "l2_perm_bimodal": hist.get("bimodal"), "l2_perm_mid_frac": hist.get("mid_frac"),
             "permuted_stats": pstats["super_acc"] if normalize else None}
    return verdict, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--l2-ld", type=float, default=L2_LD_DEFAULT, help="L2 winner-inactive depression rate (paired w/ graded read)")
    ap.add_argument("--graded", choices=list(GRADED_MODES), default="graded_read", help="graded drive/read variant")
    ap.add_argument("--normalize", action="store_true", help="also apply EMERGE-47 L2-input normalization")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--diag", action="store_true", help="fast single-seed diagnostic: permanence histogram + per-mode overlaps")
    ap.add_argument("--onsubstrate", action="store_true", help="run the 3-seed on-substrate port (slow, DECISIVE)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.demo or a.diag:
        _diag(a.seeds[0], a.epochs, a.l2_ld); return 0
    if len(a.seeds) < 3:
        print("NOT-RUNNABLE: need >=3 seeds"); return 2

    t0 = time.time(); err = None; onsub = None
    try:
        print(f"EMERGE-49: GRADED DRIVE/READ (graded={a.graded}, ld_wi={a.l2_ld}, normalize={a.normalize}) to surpass the "
              f"EMERGE-46 stacked-pooler boundary (rung b)", flush=True)
        print("  porting to the on-substrate pooler (slow bridge builds)...", flush=True)
        onsub = _onsubstrate_run(seeds=tuple(a.seeds), epochs=a.epochs, l2_ld=a.l2_ld, normalize=a.normalize,
                                 graded=a.graded, verbose=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    if err is None:
        verdict, flags = _verdict(onsub, a.l2_ld, a.graded, a.normalize)
    else:
        verdict, flags = f"ERROR -- {err}", {}

    summary = {"probe": "emerge49_graded_read", "verdict": verdict, "flags": flags,
               "mechanism": "GRADED DRIVE/READ: the on-substrate L2 pooler reads the RAW permanence-weighted drive "
                            "(sum(perm * x)) instead of the HARD `perm>0.5` connected-count (sum((perm>0.5) * x)). "
                            "'graded_drive' grades BOTH the training winner-selection AND the codon read; 'graded_read' "
                            "trains hard but reads graded. Partially-connected shared columns then contribute to a held-out "
                            "sub-category's L2 code, reproducing numpy's soft-pooling window on-substrate. The learning "
                            "kernels are byte-unchanged (a host-side read of cp_connections.data). Paired with a soft L2 "
                            "ld_wi + optional EMERGE-47 L2-input normalization.",
               "task": "EMERGE-44/46 6-sub-cat -> 2-superordinate; hold out ENTIRE sub-categories {2,5}; L2 on-substrate "
                       "pooler with a graded drive/read; measure held-out within/cross overlap + super-acc + the L2 "
                       "permanence histogram (graded vs bimodal); anti-cheats permuted-cooc + dAP-lesion (+ permuted-stats "
                       "if normalize), l2lesion reported-not-gated",
               "seeds": a.seeds, "config": {"epochs": a.epochs, "l2_ld": a.l2_ld, "graded": a.graded,
                                            "normalize": a.normalize, "n_col1": NCOL1, "n_col2": NCOL2, "k1": K1,
                                            "k2": K2, "n_super": NSUPER},
               "onsubstrate": onsub, "elapsed_seconds": round(time.time() - t0, 1),
               "sim_edit": "NONE (NO NEW sim/ edit) -- the graded drive/read is a HOST-side read of cp_connections.data "
                           "(the raw permanence value instead of the >0.5 connected-threshold); the committed learning "
                           "kernels (fused_htm_permanence_update, fused_htm_winner_inactive_depression) are byte-unchanged; "
                           "reuse-by-import of EMERGE-44/46/47 poolers",
               "HONEST_NOTE": "EMERGE-48 ISOLATED the residual: the on-substrate L2 pooler has NO soft-pooling window (jumps "
                              "over-selective -> collision), SUSPECTED cause the hard perm>0.5 read. This de-risk (rung b) gives "
                              "the L2 pooler a GRADED read so a soft window can exist. FIRST the permanence histogram diagnoses "
                              "graded vs bimodal: if bimodal, the graded read cannot help (the Foldiak trace rule = rung a is "
                              "then next); if graded and the read routes held-out inheritance with within>cross, the boundary is "
                              "surpassed. The shortcut guard: within-super held-out overlap must EXCEED cross-super, and permuted "
                              "must still collapse. Winner SELECTION is a host top-k over the on-substrate drive. 3 seeds "
                              "(42/43/44); 6-seed is a cheap confirmation follow-on."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge49] VERDICT: {verdict}", flush=True)
    print(f"[emerge49] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
