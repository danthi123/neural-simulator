"""EMERGENCE-ENGINE HETEROSYNAPTIC-LTD ALLOCATION — surpass the SEPARATE capacity wall the selective-write-store
de-risk named (regime C: full allocation starvation, n_cells=8). Keep the on-bridge HTM-TM's ALLOCATION KEYS DISJOINT
under capacity pressure so the selective-write content store gets the clean keys it needs, RESCUING the horizon where a
content store alone collapsed to chance.

WHY THIS IS THE FRONTIER (our-own-record first):
  * `2026-08-11-emergence-engine-selective-write-store-...SMOKE` showed a SELECTIVE-WRITE content store over the HTM-TM's
    OWN allocation keys RESTORES the interference-broken horizon (bare 0.667 -> store 1.000) WHEREVER clean allocation
    keys survive — but hits a HARD WALL at regime C (n_cells=8, full starvation): EVERY allocation key MERGES (16/16
    ambiguous) -> the selective gate poisons every key -> the store is EMPTY -> collapse to chance. A content store needs
    at least SOME clean allocation keys; regime C has none. The verbatim NAMED next mechanism: "a homeostatic/competitive
    (heterosynaptic-LTD) allocation that keeps allocation keys disjoint under capacity pressure, feeding the selective
    store the clean keys it needs."
  * CROSS-LANE CONVERGENCE: this is the SAME biology as the source-monitor competitive-encoding win
    (`2026-08-11-source-monitor-competitive-encoding-heterosynaptic-LTD-...6seed`) — at allocation/encoding,
    foreign/overlapping cells' shared afferents are depressed so codes stay ORTHOGONAL. One mechanism, two lanes.

THE DIAGNOSIS (measured, not assumed — dump of per-context winner SETS at regime C):
  ctx0->{0,1,2,3}  ctx1->{4,5,6,7}  ctx2->{0,1,2,3}. ctx2 REPRODUCES ctx0's EXACT set: the deterministic
  freshest-committed-cell tie-break (broken by cell index) hands the third context the first context's cells. The wall is
  NOT physical capacity: C(8,4)=70 distinct 4-subsets exist; ctx2 could take {0,1,4,5} (overlap 0.5 with each of ctx0/ctx1,
  EXACT-distinct from both). The wall is that the allocation RULE reproduces an existing context's exact key.

MECHANISM (this de-risk) — HETEROSYNAPTIC-COMPETITION ALLOCATION (the functional outcome of heterosynaptic LTD + lateral
inhibition among competing assemblies), at the HTM-TM ALLOCATE step, LABEL-FREE (keyed on the presynaptic winner SDR, not
a host cue-label):
  * When a context ALLOCATES (its prev-winner SDR matches no existing segment) it competes for the k winners against the
    codes already claimed THIS EPOCH by OTHER prev-winner SDRs in this column. It greedily picks the k cells that MINIMIZE
    the MAXIMUM per-foreign-context overlap (spread across foreign owners), tie-broken by committed-count freshness then
    index. Under starvation this forces a DISTINCT combination (ctx2 -> {0,1,4,5}) instead of reproducing a foreign key.
  * ANTI-HEBBIAN SYNAPTIC DEPRESSION (the faithful synaptic accompaniment): the real coincidence afferents (cp_connections
    .data) from this context's prev-winners into FOREIGN-claimed cells it did NOT win are depressed (mirrors the
    source-monitor rule: foreign shared afferents depressed at the encoding step). `foreign_l1_depressed` is REPORTED; at
    the allocation step the cross-talk starts sub-connected (~0), so the SELECTION competition is the load-bearing lever
    here. The fully-synaptic, label-free ONLINE realization is the declared burn-down (parallel to the source-monitor
    sibling's declared residual).

ANTI-CHEATS (each EXECUTES via tools.lab; the earned teeth):
  (a) LOAD-BEARING: LESION the hetero-LTD (the no-allocation-LTD baseline) -> allocation keys RE-MERGE (clean-key contexts
      3 -> 1) -> the selective store COLLAPSES back to chance. This is BOTH the load-bearing lesion AND anti-cheat (d): it
      IS the no-allocation-LTD baseline at the SAME capacity, so hetero must BEAT it.
  (b) ATTRIBUTABLE: PERMUTE the keys' branch labels (a derangement) -> recovery -> chance (driven by TRUE key->branch).
  (c) NO-CONFAB / SWAP-FOLLOWS-CONTEXT: inject a DIFFERENT cue -> the store must complete to the INJECTED cue's branch.
  Plus store-READ lesion (the store is load-bearing given the clean keys), always-write (selectivity), n-gram floor +
  oracle (task solvable), NO-HARM at fair capacity (n_cells=20: hetero must not reduce the store), backend/device emitted.

GO = at the STARVED regime (n_cells small) where the bare HTM breaks AND a content store alone collapses to chance,
hetero-LTD allocation makes >=1 clean allocation key survive PER context so the selective store RECOVERS (store >= 0.90,
>= no-alloc-LTD baseline + 0.15), load-bearing (baseline/lesion collapse), attributable (permute <= chance), context-driven
(swap >= 0.90), no-harm at fair capacity. HONEST NEGATIVE (first-class) = hetero-LTD does NOT keep keys disjoint / does not
rescue regime C — reported with the key-overlap numbers + the next mechanism (more cells / sparser allocation / neurogenesis).

Reuse-by-import (EMERGE-14 on-bridge machinery + the selective-write store); NO sim/ edit. SIM_BACKEND=numpy (these
sub-1k-neuron coincidence loops are LAUNCH-BOUND: cupy is SLOWER; CPU/numpy is correct + faster).
"""
import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, time, traceback
from pathlib import Path
import numpy as np

from research.runners._emerge14_stageC_onbridge_learning_derisk import (
    build_pool_bridge, OnBridgeLearner, coincidence_predict, apply_kernel_update, _host)
from research.runners._emerge9b_htm_faithful_derisk import make_overlap_sequences, markov_branch_acc, full_oracle
from research.runners._emerge_selective_write_store_derisk import (
    traverse_capture, harvest_store, store_predict_branch, _acc, swap_follows_store)

try:
    from tools.lab import lever, attributable_to, void_if
except Exception:  # tools.lab optional at import time; the runner still runs
    def lever(name, before, after, required=True, continuous=None):
        print(f"  LEVER {name}: {before} -> {after}"); return before != after
    def attributable_to(label, t, c, warn_below=0.5):
        print(f"  attributable_to {label}: t={t} c={c}"); return None
    def void_if(cond, reason):
        if cond: print(f"  VOID: {reason}")
        return bool(cond)

OUT = Path("research/findings/raw/_emerge_hetero_ltd_allocation.json")


class AllocLTDLearner(OnBridgeLearner):
    """OnBridgeLearner + HETEROSYNAPTIC-COMPETITION ALLOCATION at the ALLOCATE step. Label-free: the competition is keyed
    on the presynaptic winner SDR (distinct per context because the cues drive distinct winners), NOT a host cue-label.
    `hetero_ltd=False` reverts to the stock freshest-committed-cell allocation (the no-allocation-LTD baseline / the
    load-bearing lesion)."""

    def __init__(self, *a, hetero_ltd=True, ltd_dep=0.30, **k):
        super().__init__(*a, **k)
        self.hetero_ltd = bool(hetero_ltd)
        self.ltd_dep = float(ltd_dep)
        self.col_claims = {}                # col_symbol -> list of (prev_winner_sdr frozenset, winner_set) THIS epoch
        self.foreign_l1_depressed = 0.0     # total anti-Hebbian depression applied to foreign coincidence afferents
        self.n_alloc_events = 0

    def reset_epoch(self):
        self.col_claims = {}                # the competition record is per-epoch (fresh lateral-inhibition each round)

    def _anticollision_winners(self, col, prev_sdr, claims):
        """k winners that MINIMIZE the MAX per-foreign-context overlap (heterosynaptic competition / lateral inhibition
        among competing assemblies), tie-broken by committed-count freshness then index. Foreign = claims whose prev-winner
        SDR differs from this context's (label-free). Returns (winners, foreign_cells)."""
        wc = self._committed_count()
        foreign_claims = [w for (psdr, w) in claims if psdr != prev_sdr]
        owners = {cell: [] for cell in col}
        for fi, w in enumerate(foreign_claims):
            for cell in w:
                if cell in owners:
                    owners[cell].append(fi)
        provisional, overlap = set(), [0] * len(foreign_claims)
        while len(provisional) < self.k_win:
            best, bestkey = None, None
            for cell in col:
                if cell in provisional:
                    continue
                proj_max = max([0] + [overlap[fi] + 1 for fi in owners[cell]])
                cur_max = max(overlap) if overlap else 0
                key = (max(proj_max, cur_max), wc[cell], cell)   # spread first, then freshest, then index
                if bestkey is None or key < bestkey:
                    best, bestkey = cell, key
            provisional.add(best)
            for fi in owners[best]:
                overlap[fi] += 1
        foreign_cells = set(c for w in foreign_claims for c in w)
        return provisional, foreign_cells

    def _apply_hetero_ltd(self, prev_win, won, foreign_cells):
        """Anti-Hebbian SYNAPTIC depression: depress the REAL coincidence afferents (cp_connections.data) from prev_win
        into FOREIGN-claimed cells this context did NOT win — the cross-talk that would re-merge the code. Mirrors the
        source-monitor competitive-encoding depression of foreign shared afferents. `foreign_l1_depressed` accumulates the
        total L1 removed (reported: at the allocate step the cross-talk starts sub-connected, so this is ~0 and the
        SELECTION competition carries the effect; the fully-synaptic version is the burn-down)."""
        targets = [c for c in foreign_cells if c not in won]
        if not prev_win or not targets:
            return
        data = _host(self.b.cp_connections.data).astype(np.float64)
        pre_set = np.fromiter((int(self.cells_idx[i]) for i in prev_win), dtype=np.int64)
        post_set = np.fromiter((int(self.cells_idx[i]) for i in targets), dtype=np.int64)
        mask = np.isin(self.col, post_set) & np.isin(self.row, pre_set)
        if not mask.any():
            return
        before = float(data[mask].sum())
        data[mask] = np.maximum(0.0, data[mask] - self.ltd_dep)
        self.foreign_l1_depressed += before - float(data[mask].sum())
        self.b.cp_connections.data[:] = (self.b.xp.asarray(data.astype(np.float32))
                                         if hasattr(self.b, "xp") else data.astype(np.float32))

    def train_sequence(self, seq):
        predictive, prev_winners = set(), set()
        for c in seq:
            col = self._col(c)
            primed = [i for i in col if i in predictive] if not self.lesion else []
            if primed:
                winners = set(primed[:self.k_win])
            elif not prev_winners:
                winners = set(col[:self.k_win])
            else:
                scored = sorted(((self._match_count(i, prev_winners), i) for i in col), reverse=True)
                if scored[0][0] >= self.learn_th:
                    winners = set(i for sc, i in scored[:self.k_win] if sc >= self.learn_th)
                elif self.hetero_ltd:                                    # HETEROSYNAPTIC-COMPETITION ALLOCATION
                    prev_sdr = frozenset(prev_winners)
                    winners, foreign_cells = self._anticollision_winners(col, prev_sdr, self.col_claims.get(c, []))
                    self._apply_hetero_ltd(prev_winners, winners, foreign_cells)
                    self.col_claims.setdefault(c, []).append((prev_sdr, frozenset(winners)))
                    self.n_alloc_events += 1
                else:                                                    # STOCK freshest-committed allocation (baseline)
                    wc = self._committed_count()
                    winners = set(sorted(col, key=lambda i: (wc[i], i))[:self.k_win])
            if prev_winners:
                apply_kernel_update(self.b, self.row, self.col, self.cells_idx, prev_winners, winners,
                                    self.z, self.lam_pot, self.lam_dep, self.z_star)
            active = winners if primed else set(col)
            predictive = coincidence_predict(self.b, self.cells_idx, active, self.N, self.nE)
            self.z *= self.z_tau
            for i in predictive:
                self.z[i] += (1.0 - self.z_tau)
            prev_winners = winners


def build_alloc_htm(seed, n_seq, L, n_cells, k_win, act_th, epochs, hetero_ltd):
    seqs, vocab, info = make_overlap_sequences(n_seq=n_seq, middle_len=L, seed=seed)
    b, cells_idx, row, col = build_pool_bridge(vocab, n_cells, seed, act_th=act_th, coincidence=True)
    lr = AllocLTDLearner(b, row, col, cells_idx, vocab, n_cells, k_win=k_win, act_th=act_th, hetero_ltd=hetero_ltd)
    for _ in range(epochs):
        lr.reset_epoch()
        for s in seqs:
            lr.train_sequence(s)
    return lr, seqs, vocab


def key_disjointness(lr, seqs, L):
    """Group the confident middle allocation SDRs across positions 1..L by exact frozenset, tag each by the set of
    contexts (cues) that produced it. Returns (n_clean_groups, n_ambiguous_groups, n_contexts_with_a_clean_key). The last
    is the interpretable metric: how many of the n_seq contexts own >=1 uniquely-identifying (clean) allocation key — the
    keys the selective store can complete from. Regime C baseline = 1; a rescue makes it n_seq."""
    groups = {}
    for s in seqs:
        steps = traverse_capture(lr, s)
        for t in range(1, L + 1):
            if steps[t]["confident"]:
                groups.setdefault(steps[t]["active"], set()).add(s[0])
    clean = sum(1 for cis in groups.values() if len(cis) == 1)
    ambig = sum(1 for cis in groups.values() if len(cis) > 1)
    ctx_with_clean = set()
    for k, cis in groups.items():
        if len(cis) == 1:
            ctx_with_clean.add(next(iter(cis)))
    return clean, ambig, len(ctx_with_clean)


def _run_point(seed, n_seq, L, n_cells, k_win, act_th, epochs, tau, hetero_ltd):
    """One (seed, n_cells, hetero_ltd) point: bare / store(selective) / store-read-lesion / always-write / permute + swap
    + key-disjointness + floors."""
    lr, seqs, vocab = build_alloc_htm(seed, n_seq, L, n_cells, k_win, act_th, epochs, hetero_ltd)
    for s in seqs:
        assert traverse_capture(lr, s)[L]["pred_cols"] == lr.predict_branch(s, L)[L], "capture != predict_branch"

    sel = harvest_store(lr, seqs, selective=True)
    alw = harvest_store(lr, seqs, selective=False)
    branches = [s[-1] for s in seqs]
    perm = sel.permuted(branches, seed)

    bare = _acc(lambda s: store_predict_branch(lr, None, s, L, tau), seqs, L)
    store = _acc(lambda s: store_predict_branch(lr, sel, s, L, tau), seqs, L)
    lesion = _acc(lambda s: store_predict_branch(lr, sel, s, L, tau, lesion_read=True), seqs, L)
    always = _acc(lambda s: store_predict_branch(lr, alw, s, L, tau), seqs, L)
    permute = _acc(lambda s: store_predict_branch(lr, perm, s, L, tau), seqs, L)
    swap = swap_follows_store(lr, sel, seqs, L, tau)
    n_clean, n_ambig, ctx_clean = key_disjointness(lr, seqs, L)

    return {"seed": seed, "L": L, "distance": L + 1, "n_cells": n_cells, "hetero_ltd": bool(hetero_ltd),
            "chance": 1.0 / n_seq, "bare": bare, "store": store, "store_lesion": lesion, "always_write": always,
            "permute": permute, "swap_follows": swap, "markov": markov_branch_acc(seqs, L, n_seq),
            "oracle": full_oracle(seqs, L), "n_clean_keys": n_clean, "n_ambiguous_keys": n_ambig,
            "n_contexts_with_clean_key": ctx_clean, "n_sel_keys": len(sel.store), "n_sel_poisoned": len(sel.poisoned),
            "foreign_l1_depressed": round(float(lr.foreign_l1_depressed), 4), "n_alloc_events": int(lr.n_alloc_events)}


def _agg(per):
    def m(k):
        return float(np.mean([p[k] for p in per]))
    keys = ["bare", "store", "store_lesion", "always_write", "permute", "swap_follows", "markov", "oracle",
            "n_clean_keys", "n_ambiguous_keys", "n_contexts_with_clean_key", "n_sel_keys", "n_sel_poisoned",
            "foreign_l1_depressed", "n_alloc_events"]
    a = {k: m(k) for k in keys}
    a.update({"n_cells": per[0]["n_cells"], "L": per[0]["L"], "distance": per[0]["distance"],
              "hetero_ltd": per[0]["hetero_ltd"], "chance": per[0]["chance"], "per_seed": per})
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--n-cells", type=int, default=8, help="STARVED capacity (regime C = 8; k_win*n_seq=12 needed for disjoint)")
    ap.add_argument("--fair-cells", type=int, default=20, help="fair-capacity NO-HARM guard point (hetero must not reduce the store)")
    ap.add_argument("--distance", type=int, default=16, help="shared-middle length L (dependency distance = L+1); break at dist>=17")
    ap.add_argument("--n-seq", type=int, default=3, help="# interfering contexts (chance = 1/n_seq)")
    ap.add_argument("--k-win", type=int, default=4)
    ap.add_argument("--act-th", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--tau-read", type=float, default=0.5, help="min allocation-SDR overlap for a content-addressed hit")
    ap.add_argument("--no-fair", action="store_true", help="skip the fair-capacity no-harm guard (starved regime only)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if len(a.seeds) < 1:
        print("NOT-RUNNABLE: need >=1 seed"); return 2
    smoke = len(a.seeds) < 6
    backend = os.environ.get("SIM_BACKEND", "numpy")
    device = "cpu" if backend == "numpy" else "gpu"
    L, n_seq, chance = a.distance, a.n_seq, 1.0 / a.n_seq
    print(f"backend={backend} device={device} | n_seq={n_seq} chance={chance:.3f} | starved n_cells={a.n_cells} "
          f"(need {a.k_win*n_seq} disjoint) | fair n_cells={a.fair_cells} | dist={L+1} epochs={a.epochs} "
          f"tau={a.tau_read} | seeds={a.seeds}", flush=True)

    t0 = time.time(); err = None
    starved_on = starved_off = fair_on = fair_off = None
    try:
        starved_off = _agg([_run_point(s, n_seq, L, a.n_cells, a.k_win, a.act_th, a.epochs, a.tau_read, False) for s in a.seeds])
        starved_on = _agg([_run_point(s, n_seq, L, a.n_cells, a.k_win, a.act_th, a.epochs, a.tau_read, True) for s in a.seeds])
        print(f"  [STARVED n_cells={a.n_cells}] NO-alloc-LTD : bare {starved_off['bare']:.3f} store {starved_off['store']:.3f} "
              f"| clean-key-ctxs {starved_off['n_contexts_with_clean_key']:.2f}/{n_seq} | ambig-keys {starved_off['n_ambiguous_keys']:.1f}", flush=True)
        print(f"  [STARVED n_cells={a.n_cells}] HETERO-LTD   : bare {starved_on['bare']:.3f} store {starved_on['store']:.3f} "
              f"| store-lesion {starved_on['store_lesion']:.3f} always {starved_on['always_write']:.3f} permute "
              f"{starved_on['permute']:.3f} swap {starved_on['swap_follows']:.3f} | clean-key-ctxs "
              f"{starved_on['n_contexts_with_clean_key']:.2f}/{n_seq} | fL1 {starved_on['foreign_l1_depressed']:.3f}", flush=True)
        if not a.no_fair:
            fair_off = _agg([_run_point(s, n_seq, L, a.fair_cells, a.k_win, a.act_th, a.epochs, a.tau_read, False) for s in a.seeds])
            fair_on = _agg([_run_point(s, n_seq, L, a.fair_cells, a.k_win, a.act_th, a.epochs, a.tau_read, True) for s in a.seeds])
            print(f"  [FAIR    n_cells={a.fair_cells}] NO-HARM: store off {fair_off['store']:.3f} -> on {fair_on['store']:.3f} "
                  f"(hetero must not reduce)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    verdict = None
    if err is None and starved_on is not None:
        print("\n-- anti-cheats at the STARVED regime (n_cells=%d, dist=%d) --" % (a.n_cells, L + 1), flush=True)
        void_if(starved_on["oracle"] <= 0.99, "task not context-solvable (oracle %.3f)" % starved_on["oracle"])
        # (a) LOAD-BEARING + (d) beats no-alloc-LTD baseline at SAME capacity: hetero store vs baseline store
        lever("hetero-LTD store vs no-alloc-LTD baseline (load-bearing / anti-cheat d)",
              round(starved_off["store"], 3), round(starved_on["store"], 3), required=False)
        lever("clean-key CONTEXTS: no-alloc-LTD -> hetero (keys kept disjoint)",
              round(starved_off["n_contexts_with_clean_key"], 2), round(starved_on["n_contexts_with_clean_key"], 2),
              required=False)
        lever("store-READ lesion within hetero (store load-bearing given clean keys)",
              round(starved_on["store_lesion"], 3), round(starved_on["store"], 3), required=False)
        attributable_to("store recovery via TRUE keys (vs permuted)", starved_on["store"], starved_on["permute"])
        attributable_to("hetero recovery over no-alloc-LTD baseline", starved_on["store"], starved_off["store"])

        broke = starved_on["bare"] < 0.90 and starved_off["bare"] < 0.90
        baseline_collapsed = starved_off["store"] <= chance + 0.10          # a content store alone CANNOT rescue regime C
        rescues = starved_on["store"] >= 0.90 and starved_on["store"] >= starved_off["store"] + 0.15
        keys_disjoint = starved_on["n_contexts_with_clean_key"] >= n_seq - 1e-6 and \
            starved_on["n_contexts_with_clean_key"] >= starved_off["n_contexts_with_clean_key"] + 0.5
        store_load_bearing = starved_on["store"] >= starved_on["store_lesion"] + 0.15
        selective = starved_on["store"] >= starved_on["always_write"] + 0.15
        attributed = starved_on["permute"] <= chance + 0.10
        context_driven = starved_on["swap_follows"] >= 0.90
        solvable = starved_on["oracle"] > 0.99
        no_harm = (a.no_fair or fair_on is None) or (fair_on["store"] >= fair_off["store"] - 1e-6)

        core = (broke and baseline_collapsed and rescues and keys_disjoint and store_load_bearing and selective
                and attributed and context_driven and solvable and no_harm)
        go = bool(core and not smoke)
        smoke_go = bool(core and smoke)

        if not solvable:
            verdict = f"INCONCLUSIVE — task not context-solvable (oracle {starved_on['oracle']:.3f})."
        elif not broke:
            verdict = (f"INCONCLUSIVE — bare HTM did NOT break at dist {L+1} (bare hetero {starved_on['bare']:.3f} / baseline "
                       f"{starved_off['bare']:.3f} >= 0.90); no chain-integrity residual to rescue at this resource point.")
        elif not baseline_collapsed:
            verdict = (f"INCONCLUSIVE — the no-allocation-LTD baseline store did NOT collapse at n_cells={a.n_cells} "
                       f"(store {starved_off['store']:.3f} > chance+0.10); this is not the full-starvation regime the "
                       f"hetero-LTD allocation is meant to rescue. Reduce n_cells.")
        elif core:
            tag = "GO" if go else ("SMOKE-GO (1-seed indicator; run the 6-seed sweep)" if smoke else "GO")
            verdict = (f"{tag} — HETEROSYNAPTIC-COMPETITION ALLOCATION keeps the HTM-TM's allocation keys DISJOINT under "
                       f"full capacity starvation (n_cells={a.n_cells}), RESCUING the horizon where a content store alone "
                       f"collapsed to chance: at dist {L+1} (n_seq={n_seq}) the no-allocation-LTD baseline store is "
                       f"{starved_off['store']:.3f} (~chance {chance:.3f}; {starved_off['n_contexts_with_clean_key']:.0f}/"
                       f"{n_seq} contexts own a clean key) but WITH hetero-LTD the selective store recovers to "
                       f"{starved_on['store']:.3f} ({starved_on['n_contexts_with_clean_key']:.0f}/{n_seq} contexts own a "
                       f"clean key). LOAD-BEARING / anti-cheat (d) (lesion hetero-LTD = the baseline -> keys re-merge -> "
                       f"store {starved_off['store']:.3f}), store-READ load-bearing (lesion {starved_on['store_lesion']:.3f}), "
                       f"SELECTIVITY GATES (always-write {starved_on['always_write']:.3f}), ATTRIBUTABLE (permute "
                       f"{starved_on['permute']:.3f} <= chance {chance:.3f}), CONTEXT-DRIVEN (swap {starved_on['swap_follows']:.3f}, "
                       f"no confab), NO-HARM at fair n_cells={a.fair_cells} (store off {('n/a' if fair_off is None else format(fair_off['store'],'.3f'))} "
                       f"-> on {('n/a' if fair_on is None else format(fair_on['store'],'.3f'))}). foreign_l1_depressed "
                       f"{starved_on['foreign_l1_depressed']:.3f} (the synaptic depression is ~0 at the sub-connected "
                       f"allocation step -> the SELECTION competition is the load-bearing lever; the fully-synaptic ONLINE "
                       f"version is the declared burn-down). Reuse-by-import of EMERGE-14 + the selective store; NO sim/ edit.")
        else:
            miss = []
            if not baseline_collapsed: miss.append(f"baseline store didn't collapse ({starved_off['store']:.3f})")
            if not rescues: miss.append(f"hetero store didn't rescue (store {starved_on['store']:.3f}, need >=0.90 and >= baseline+0.15)")
            if not keys_disjoint: miss.append(f"keys not kept disjoint (clean-key ctxs {starved_off['n_contexts_with_clean_key']:.1f} -> {starved_on['n_contexts_with_clean_key']:.1f}, need {n_seq})")
            if not store_load_bearing: miss.append(f"store-read not load-bearing ({starved_on['store']:.3f} vs lesion {starved_on['store_lesion']:.3f})")
            if not selective: miss.append(f"selectivity inert ({starved_on['store']:.3f} vs always {starved_on['always_write']:.3f})")
            if not attributed: miss.append(f"not attributable (permute {starved_on['permute']:.3f} > chance+0.10)")
            if not context_driven: miss.append(f"not context-driven (swap {starved_on['swap_follows']:.3f} < 0.90)")
            if not no_harm: miss.append(f"HARMS fair capacity (store {fair_off['store']:.3f} -> {fair_on['store']:.3f})")
            verdict = ("HONEST NEGATIVE / BOUNDARY — hetero-LTD allocation did NOT cleanly rescue regime C: "
                       + "; ".join(miss) + f". clean-key contexts (no-alloc-LTD {starved_off['n_contexts_with_clean_key']:.1f} -> "
                       f"hetero {starved_on['n_contexts_with_clean_key']:.1f} of {n_seq}). Names the next mechanism (more "
                       f"cells / sparser allocation (smaller k_win) / a growth-neurogenesis mechanism).")
    else:
        verdict = f"ERROR — {err}" if err else "ERROR — no points computed"

    # --- earned verdict: VALIDITY preconditions travel with the verdict (tools/gates/verdict_preconditions) ---
    preconditions = []
    try:
        from tools.verdict import Verdict
        V = Verdict("emerge_hetero_ltd_allocation", chance=chance)
        if err is None and starved_on is not None:
            V.require("oracle>0.99_task_solvable", round(starved_on["oracle"], 4), expect=lambda x: x > 0.99,
                      note="else the task is not context-solvable -> INCONCLUSIVE, not a negative")
            V.require("bare_broke_under_interference", round(starved_on["bare"], 4), expect=lambda x: x < 0.90,
                      note="the store's job is to rescue a BROKEN horizon; if bare holds there is nothing to rescue")
            V.require("no_alloc_LTD_baseline_store_collapsed", round(starved_off["store"], 4),
                      expect=lambda x: x <= chance + 0.10,
                      note="regime C: a content store ALONE must collapse (no clean keys) else hetero-LTD is not the lever")
            V.require("permute_attribution_instrument_collapses", round(starved_on["permute"], 4),
                      expect=lambda x: x <= chance + 0.10,
                      note="permuted keys must fall to <= chance, else the content-addressed recall is not key-driven")
        else:
            V.require("point_computed", 0, expect=lambda x: x >= 1, note="run errored before any point computed")
        _go = bool(err is None and starved_on is not None and starved_on["oracle"] > 0.99 and starved_on["bare"] < 0.90
                   and starved_off["store"] <= chance + 0.10 and starved_on["store"] >= 0.90
                   and starved_on["store"] >= starved_off["store"] + 0.15
                   and starved_on["n_contexts_with_clean_key"] >= n_seq - 1e-6
                   and starved_on["store"] >= starved_on["store_lesion"] + 0.15
                   and starved_on["store"] >= starved_on["always_write"] + 0.15
                   and starved_on["permute"] <= chance + 0.10 and starved_on["swap_follows"] >= 0.90)
        dec = V.decide(_go, verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e), "note": ""}]

    summary = {"probe": "emerge_hetero_ltd_allocation", "verdict": verdict, "backend": backend, "sim_backend": backend,
               "device": device, "cost_acknowledged": True, "smoke": smoke, "preconditions": preconditions,
               "mechanism": "HETEROSYNAPTIC-COMPETITION ALLOCATION (functional outcome of heterosynaptic LTD + lateral "
                            "inhibition among competing assemblies) at the on-bridge HTM-TM allocate step, label-free "
                            "(keyed on the presynaptic winner SDR): a new context's k winners are chosen to MINIMIZE the "
                            "max per-foreign-context overlap so allocation keys stay DISJOINT under capacity starvation, "
                            "feeding the selective-write content store the clean keys it needs; anti-Hebbian synaptic "
                            "depression of foreign coincidence afferents accompanies it (foreign_l1_depressed reported; "
                            "~0 at the sub-connected allocate step, so the SELECTION competition is the load-bearing lever)",
               "task": "EMERGE-14 overlap corpus [cue, <L middle>, branch]; branch depends on the cue L+1 tokens back "
                       "(n-gram pinned at chance 1/n_seq); regime C = full allocation starvation (n_cells << k_win*n_seq); "
                       "anti-cheats: no-alloc-LTD baseline (load-bearing / same-capacity) + store-read lesion + always-write "
                       "(selectivity) + permute-keys (attribution) + swap-follows-context (no confab) + fair-capacity "
                       "no-harm + oracle",
               "seeds": a.seeds, "config": {"n_cells": a.n_cells, "fair_cells": a.fair_cells, "distance": L, "n_seq": n_seq,
               "k_win": a.k_win, "act_th": a.act_th, "epochs": a.epochs, "tau_read": a.tau_read, "chance": chance},
               "starved_no_alloc_ltd": starved_off, "starved_hetero_ltd": starved_on,
               "fair_no_alloc_ltd": fair_off, "fair_hetero_ltd": fair_on,
               "elapsed_seconds": round(time.time() - t0, 1),
               "HONEST_NOTE": "reuse-by-import of the EMERGE-14 on-bridge learner + the selective-write store; NO sim/ edit. "
                              "The allocation (winner-selection) is host-orchestrated (the DECLARED EMERGE-9d residual); "
                              "this adds a host-side heterosynaptic COMPETITION to it (at parity with the source-monitor "
                              "sibling's host-bookkept competitive encoding). The label-free, fully-synaptic, ONLINE "
                              "realization (derive the foreign-code depression from the substrate's own firing, no host "
                              "claim record) is the burn-down. 1-seed is a SMOKE indicator; the decisive run is the 6-seed."}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 108, flush=True)
    print(f"[emerge_hetero_ltd_allocation] VERDICT: {verdict}", flush=True)
    print(f"[emerge_hetero_ltd_allocation] wrote {a.out}\n" + "=" * 108, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
