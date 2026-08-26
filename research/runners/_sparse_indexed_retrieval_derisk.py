"""DG-indexed SUBLINEAR fact retrieval de-risk (KNOWLEDGE SCALE, board #66).

THE WALL (verified in `one_brain_composer.py`). The production knowledge/fact store recalls by a CLEANUP: it
matches a recovered query phasor against the codebook of ALL V concept codes -- a V x D matched-filter matvec --
then `argmax` (`self.words[int(np.argmax(scores))]`, the `_select`/`_block_role_scores` cleanup path; abstain via
the `_margin` confidence gate). That is O(V * D), LINEAR in vocabulary: ~1.1 s / recall at ~37 K vocab, and
INTRACTABLE (~15-30 s) at small-LLM scale (500 K - 1 M concepts). Loading facts is easy; RETRIEVING them at scale
is the single biggest blocker to LLM-scale knowledge.

THE BRAIN-FAITHFUL TARGET. Real brains do NOT linearly scan memory. The hippocampal trisynaptic loop does
DENTATE-GYRUS sparse PATTERN SEPARATION (a cue -> a sparse, decorrelated granule-cell code) then CA3
auto-associative COMPLETION restricted to the routed ensemble -- i.e. a cue CONTENT-ADDRESSABLY routes to a SMALL
candidate set and completes there, O(shard) not O(V). This de-risk builds retrieval as exactly that: a DG-like
sparse index routes a query cue to its small candidate SHARD, then the EXISTING cleanup runs only within the shard.

MECHANISM (the sparse index = DG expansion + hard k-WTA + CA3-conjunction routing).
  * DG expansion + sparsification: each cue's real feature x = [cos(phi), sin(phi)] in R^(2D) is projected through
    L fixed random SPARSE granule-cell bands (each granule cell samples only c input dims -- DG afferents are
    sparse), then a hard WINNER-TAKE-ALL per band (k=1) yields a band code (one active granule cell of m) -- the
    sparsest possible pattern-separated code. Derived PURELY from the CUE CONTENT; never from the answer id.
  * CA3 conjunction routing: the L bands are partitioned into G groups of g bands. A group's bucket key is the
    g-tuple of its band winners (a conjunctive code -- a memory is the CO-activation of a specific granule
    ensemble). At store time each fact drops its id into its G group-buckets; at query time the candidate SHARD is
    the union of the query's G group-buckets. Setting m ~ V^(1/g) keeps bucket occupancy O(1), so the shard stays
    ~CONSTANT as V grows while the linear cleanup runs over the shard only -> SUBLINEAR retrieval. G>1 groups
    (OR-of-conjunctions, multi-probe LSH amplification) give noise-robust recall.

  This is the standard banded-LSH sublinearity result, re-derived as the DG->CA3 sparse-conjunctive code.

GO CRITERIA (6-seed 42 43 44 100 101 102; wired into the printed VERDICT):
  1. ACCURACY PARITY: at V in {10k, 50k, 200k} synthetic FHRR concepts at production D, the DG-indexed retrieval
     returns the SAME top-1 as the full linear cleanup (top-1 agreement >= 0.98). Sharding must not lose recall.
  2. SUBLINEAR COST: measured cost (candidate-set size / matvec rows touched AND wall-clock) grows SUBLINEARLY
     with V for the indexed path while the linear path grows O(V). Report the speedup at each V (>= 10x at 200k).
  3. MOAT PRESERVED: a genuinely out-of-store cue ABSTAINS under BOTH paths (the sharding introduces no new
     confabulation). PROVABLE: shard is a SUBSET of the codebook, so the shard's peak score <= the full peak; if
     the full scan abstains (peak < familiarity floor tau*D), the DG shard's peak is <= that and also abstains.

ANTI-CHEATS (wired in):
  (a) CONTENT-ADDRESSABLE routing: the index key is computed from the CUE VECTOR via the DG sparse projection,
      NEVER a host dict keyed on the ground-truth answer id. Asserted structurally (the encoder only ever sees
      feature vectors, never ids).
  (b) ACCURACY PARITY vs the full scan is a hard gate -- a fast-but-wrong index is a NO-GO (criterion 1).
  (c) SCRAMBLE control: permuting the query's band-winner tuple (routing decorrelated from content) collapses
      accuracy to ~chance -- proves the routing is load-bearing, not luck.

BRAIN-BASED NOTE (honest). The matched-filter cleanup INSIDE a shard is the production on-substrate op (this is the
mechanism the composer already runs, just over fewer rows). The DG sparse projection here is a RATE/host shortcut
(a fixed random sparse projection + hard WTA), DECLARED as such. Its named biological burn-down is the SPIKING DG
granule-cell layer already validated in the trisynaptic-loop probes: `_riii_ca3_completion_specificity_derisk.py`
(CA3 partial-cue completion specificity), `cortex_dg_ca3_cleanup_probe.py`, `_gap5_emergent_dg_selection_derisk.py`
(emergent DG k-WTA selection). The routing here is content-addressable and sparse exactly as the granule layer is;
the burn-down replaces the host argmax-WTA with the spiking granule competition (the same NEF-WTA the cleanup
Stage-2 already uses). NO `sim/` edit.

Determinism: every RNG is seeded from the --seeds value (cfg.seed discipline; there is no sim substrate to reseed
here -- this is a host-rate de-risk of the INDEX, the spiking substrate is the burn-down).
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np


# ------------------------------- FHRR codebook (production-shaped) -------------------------------
def gen_fhrr_phases(V: int, D: int, rng: np.random.Generator) -> np.ndarray:
    """V decorrelated unit-modulus FHRR concept codes, as phase angles in [0, 2pi). The production concept/word
    codebook is exactly this: random unit phasors, cleaned up by a matched filter (conj-correlation)."""
    return rng.uniform(0.0, 2.0 * np.pi, size=(V, D)).astype(np.float32)


def phases_to_complex(phases: np.ndarray) -> np.ndarray:
    return np.exp(1j * phases).astype(np.complex64)


def make_query_phases(code_phases: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """A noisy cue = a stored code + per-component Gaussian PHASE jitter (a partial/degraded recall cue). sigma=0
    reproduces the exact stored code. Expected self matched-filter score = D * exp(-sigma^2/2)."""
    return (code_phases + rng.normal(0.0, sigma, size=code_phases.shape)).astype(np.float32)


# ------------------------------- full linear cleanup (the production path) -----------------------
def linear_scores(Zc: np.ndarray, q_phases: np.ndarray) -> np.ndarray:
    """The matched filter: score_j = Re(<conj(z_j), q>) for all V codes = the V x D complex matvec the composer
    runs. Returns real scores (V,). A stored match scores ~D; a random code scores ~sqrt(D/2)."""
    q = np.exp(1j * q_phases).astype(np.complex64)
    return (Zc.conj() @ q).real


def decide(scores_subset: np.ndarray, ids_subset: np.ndarray, D: int, tau: float):
    """Shared decision rule for BOTH paths: top-1 over the given score subset, ABSTAIN if peak < tau*D
    (familiarity floor) -- the no-confab moat. Returns (answer_id or None, peak_score)."""
    if scores_subset.size == 0:
        return None, 0.0
    k = int(np.argmax(scores_subset))
    peak = float(scores_subset[k])
    if peak < tau * D:
        return None, peak
    return int(ids_subset[k]), peak


# ------------------------------- DG-like sparse index -------------------------------------------
class DGSparseIndex:
    """DG expansion + hard k-WTA + CA3-conjunction routing. Content-addressable: encode() only ever sees feature
    vectors, NEVER answer ids (anti-cheat a). See module docstring."""

    def __init__(self, D: int, m: int, g: int, G: int, c: int, seed: int):
        self.D = int(D)
        self.m = int(m)        # granule cells per band (WTA width) ~ V^(1/g)
        self.g = int(g)        # conjunction order (bands per group)
        self.G = int(G)        # number of groups / probes (OR amplification)
        self.c = int(c)        # DG afferent fan-in per granule cell (sparse dendrite)
        self.L = self.g * self.G                                   # total bands
        rng = np.random.default_rng(seed * 2654435761 % (2 ** 32))
        in_dim = 2 * self.D
        n_cells = self.L * self.m
        # sparse granule-cell afferents: each cell samples c input dims with +-1 weights (DG sparse connectivity)
        self._idx = rng.integers(0, in_dim, size=(n_cells, self.c)).astype(np.int64)
        self._w = (rng.integers(0, 2, size=(n_cells, self.c)) * 2 - 1).astype(np.float32)
        # groups = disjoint partitions of the L bands
        self._groups = [list(range(gi * self.g, (gi + 1) * self.g)) for gi in range(self.G)]
        self._buckets = None                                       # list[G] of dict{tuple->list[int]}

    @staticmethod
    def _features(phases: np.ndarray) -> np.ndarray:
        return np.concatenate([np.cos(phases), np.sin(phases)], axis=-1).astype(np.float32)

    def _winners(self, phases: np.ndarray, chunk: int = 20000) -> np.ndarray:
        """Band winners (N, L): for each band, activation of its m granule cells (sparse gather-sum) then hard WTA
        (argmax). Chunked over codes to bound memory. This is the only op that touches code content -> the routing
        is content-addressable by construction."""
        phases = np.atleast_2d(phases)
        N = phases.shape[0]
        out = np.empty((N, self.L), dtype=np.int32)
        for s in range(0, N, chunk):
            X = self._features(phases[s:s + chunk])                # (n, 2D)
            n = X.shape[0]
            for b in range(self.L):
                sl = slice(b * self.m, (b + 1) * self.m)
                idx = self._idx[sl]                                # (m, c)
                w = self._w[sl]                                    # (m, c)
                act = np.einsum('nmc,mc->nm', X[:, idx], w)        # (n, m) granule activations
                out[s:s + n, b] = np.argmax(act, axis=1)
        return out

    def build(self, code_phases: np.ndarray):
        """Store: each fact drops its id into its G group-buckets (keyed by the conjunctive band-winner tuple)."""
        W = self._winners(code_phases)                             # (V, L)
        self._buckets = [dict() for _ in range(self.G)]
        V = W.shape[0]
        for gi, bands in enumerate(self._groups):
            bk = self._buckets[gi]
            wg = W[:, bands]
            for fid in range(V):
                bk.setdefault(tuple(wg[fid].tolist()), []).append(fid)
        return W

    def query(self, q_phases: np.ndarray, scramble_rng: np.random.Generator = None) -> np.ndarray:
        """Route a cue to its candidate SHARD = union of its G group-buckets. scramble_rng (anti-cheat c) replaces
        the content-derived winners with RANDOM winners -> routing decorrelated from content."""
        w = self._winners(q_phases)[0]                             # (L,)
        if scramble_rng is not None:
            w = scramble_rng.integers(0, self.m, size=self.L).astype(np.int32)
        cand = set()
        for gi, bands in enumerate(self._groups):
            key = tuple(w[bands].tolist())
            hit = self._buckets[gi].get(key)
            if hit:
                cand.update(hit)
        return np.fromiter(cand, dtype=np.int64, count=len(cand))


# ------------------------------- per-scale evaluation -------------------------------------------
def eval_scale(V: int, D: int, seed: int, g: int, G: int, c: int, sigma: float, tau: float,
               n_query: int, oos_query: int):
    rng = np.random.default_rng(seed * 1000003 + V)
    phases = gen_fhrr_phases(V, D, rng)
    Zc = phases_to_complex(phases)
    m = max(2, int(np.ceil(V ** (1.0 / g))))                       # granule width -> occupancy O(1)

    idx = DGSparseIndex(D=D, m=m, g=g, G=G, c=c, seed=seed)
    t0 = time.perf_counter()
    idx.build(phases)
    build_s = time.perf_counter() - t0

    q_ids = rng.integers(0, V, size=n_query)
    q_phases = np.stack([make_query_phases(phases[i], sigma, rng) for i in q_ids])

    # ----- FULL linear path (production) -----
    full_ans, full_peak, full_rows, full_wall = [], [], [], []
    for j in range(n_query):
        t = time.perf_counter()
        sc = linear_scores(Zc, q_phases[j])
        ans, pk = decide(sc, np.arange(V), D, tau)
        full_wall.append(time.perf_counter() - t)
        full_ans.append(ans); full_peak.append(pk); full_rows.append(V)

    # ----- DG-indexed path -----
    dg_ans, dg_rows, dg_wall = [], [], []
    scr_ans = []
    scr_rng = np.random.default_rng(seed * 777 + V)
    for j in range(n_query):
        t = time.perf_counter()
        shard = idx.query(q_phases[j])
        if shard.size:
            sc = linear_scores(Zc[shard], q_phases[j])
            ans, _ = decide(sc, shard, D, tau)
        else:
            ans = None
        dg_wall.append(time.perf_counter() - t)
        dg_ans.append(ans); dg_rows.append(int(shard.size))
        # scramble control (anti-cheat c)
        sshard = idx.query(q_phases[j], scramble_rng=scr_rng)
        if sshard.size:
            ssc = linear_scores(Zc[sshard], q_phases[j])
            sa, _ = decide(ssc, sshard, D, tau)
        else:
            sa = None
        scr_ans.append(sa)

    # ----- accuracy parity (criterion 1) : DG agrees with FULL, over queries FULL did not abstain -----
    valid = [j for j in range(n_query) if full_ans[j] is not None]
    agree = np.mean([dg_ans[j] == full_ans[j] for j in valid]) if valid else 0.0
    truth_acc = np.mean([dg_ans[j] == int(q_ids[j]) for j in valid]) if valid else 0.0
    scr_agree = np.mean([scr_ans[j] == full_ans[j] for j in valid]) if valid else 0.0
    chance = 1.0 / V

    # ----- moat (criterion 3): out-of-store cues must abstain under BOTH -----
    oos_phases = gen_fhrr_phases(oos_query, D, np.random.default_rng(seed * 31337 + V))  # codes NOT in the store
    full_abstain, dg_abstain, dg_confab = 0, 0, 0
    for j in range(oos_query):
        sc = linear_scores(Zc, oos_phases[j])
        fa, _ = decide(sc, np.arange(V), D, tau)
        shard = idx.query(oos_phases[j])
        if shard.size:
            dsc = linear_scores(Zc[shard], oos_phases[j])
            da, _ = decide(dsc, shard, D, tau)
        else:
            da = None
        full_abstain += (fa is None)
        dg_abstain += (da is None)
        # new confab = DG returns a match where FULL abstained
        dg_confab += (fa is None and da is not None)

    return {
        "V": V, "D": D, "m": m, "g": g, "G": G, "shard_bands": idx.L,
        "parity_agree": float(agree), "truth_acc": float(truth_acc),
        "scramble_agree": float(scr_agree), "chance": float(chance),
        "rows_full": float(np.mean(full_rows)), "rows_dg": float(np.mean(dg_rows)),
        "rows_speedup": float(np.mean(full_rows) / max(1e-9, np.mean(dg_rows))),
        "wall_full_ms": float(np.mean(full_wall) * 1e3), "wall_dg_ms": float(np.mean(dg_wall) * 1e3),
        "wall_speedup": float(np.mean(full_wall) / max(1e-12, np.mean(dg_wall))),
        "build_s": float(build_s),
        "oos_full_abstain": float(full_abstain / max(1, oos_query)),
        "oos_dg_abstain": float(dg_abstain / max(1, oos_query)),
        "dg_new_confab": int(dg_confab),
    }


def run_seed(seed: int, scales, D, g, G, c, sigma, tau, n_query, oos_query, verbose=True):
    rows = []
    for V in scales:
        r = eval_scale(V, D, seed, g, G, c, sigma, tau, n_query, oos_query)
        rows.append(r)
        if verbose:
            print(f"  [seed {seed}] V={V:>7d} m={r['m']:>3d} | parity(DG==FULL)={r['parity_agree']:.3f} "
                  f"truth={r['truth_acc']:.3f} scramble={r['scramble_agree']:.4f}(chance={r['chance']:.1e}) | "
                  f"rows FULL={r['rows_full']:.0f} DG={r['rows_dg']:.1f} ({r['rows_speedup']:.0f}x) | "
                  f"wall FULL={r['wall_full_ms']:.2f}ms DG={r['wall_dg_ms']:.3f}ms ({r['wall_speedup']:.1f}x) | "
                  f"moat oos-abstain FULL={r['oos_full_abstain']:.2f} DG={r['oos_dg_abstain']:.2f} "
                  f"new-confab={r['dg_new_confab']}", flush=True)
    return rows


def verdict(all_rows, scales):
    """GO logic. Aggregates the per-seed per-scale rows."""
    by_V = {V: [r for rows in all_rows for r in rows if r["V"] == V] for V in scales}
    Vmax = max(scales)
    # 1. accuracy parity >= 0.98 at every scale
    parity_ok = all(np.mean([r["parity_agree"] for r in by_V[V]]) >= 0.98 for V in scales)
    # 2. sublinear cost: the operational definition -- the index's ROWS-SPEEDUP WIDENS with V (cost grows strictly
    #    slower than the O(V) linear path) AND the shard grows strictly slower than V; plus >=10x rows- AND
    #    wall-speedup at Vmax.
    Vmin = min(scales)
    rows_small = np.mean([r["rows_dg"] for r in by_V[Vmin]])
    rows_large = np.mean([r["rows_dg"] for r in by_V[Vmax]])
    v_growth = Vmax / Vmin
    shard_growth = rows_large / max(1e-9, rows_small)
    speedup_small = np.mean([r["rows_speedup"] for r in by_V[Vmin]])
    speedup_large = np.mean([r["rows_speedup"] for r in by_V[Vmax]])
    sublinear = (speedup_large > speedup_small) and (shard_growth < v_growth)  # advantage widens; shard << V
    rows_speedup_ok = speedup_large >= 10.0
    wall_speedup_ok = np.mean([r["wall_speedup"] for r in by_V[Vmax]]) >= 10.0
    cost_ok = sublinear and rows_speedup_ok and wall_speedup_ok
    # 3. moat: out-of-store abstains under BOTH, zero DG new-confab
    moat_ok = all(np.mean([r["oos_dg_abstain"] for r in by_V[V]]) >= 0.999
                  and np.mean([r["oos_full_abstain"] for r in by_V[V]]) >= 0.999
                  and sum(r["dg_new_confab"] for r in by_V[V]) == 0 for V in scales)
    # anti-cheat c: scramble collapses to ~chance
    scramble_ok = all(np.mean([r["scramble_agree"] for r in by_V[V]]) <= 0.05 for V in scales)
    go = parity_ok and cost_ok and moat_ok and scramble_ok
    return {
        "go": bool(go), "parity_ok": bool(parity_ok), "cost_ok": bool(cost_ok),
        "sublinear": bool(sublinear), "rows_speedup_ok": bool(rows_speedup_ok),
        "wall_speedup_ok": bool(wall_speedup_ok), "moat_ok": bool(moat_ok),
        "scramble_ok": bool(scramble_ok), "v_growth": float(v_growth), "shard_growth": float(shard_growth),
        "rows_speedup_Vmax": float(np.mean([r["rows_speedup"] for r in by_V[Vmax]])),
        "wall_speedup_Vmax": float(np.mean([r["wall_speedup"] for r in by_V[Vmax]])),
    }


def main():
    ap = argparse.ArgumentParser(description="DG-indexed sublinear fact retrieval de-risk (#66 knowledge scale)")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--scales", default="10000,50000,200000", help="V values (facts) to sweep")
    ap.add_argument("--D", type=int, default=256, help="FHRR dimension (production console op-point)")
    ap.add_argument("--g", type=int, default=3, help="DG->CA3 conjunction order (bands per group)")
    ap.add_argument("--G", type=int, default=16, help="probe groups (OR amplification / noise robustness)")
    ap.add_argument("--c", type=int, default=8, help="DG afferent fan-in per granule cell")
    ap.add_argument("--sigma", type=float, default=0.30, help="query phase-noise (partial/degraded cue)")
    ap.add_argument("--tau", type=float, default=0.5, help="familiarity floor (fraction of D) for abstain")
    ap.add_argument("--n-query", type=int, default=120, help="in-store cue queries per scale")
    ap.add_argument("--oos-query", type=int, default=120, help="out-of-store cue queries per scale (moat)")
    ap.add_argument("--json", default=None, help="write per-seed rows JSON (fan-out aggregation)")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",")]
    scales = [int(x) for x in a.scales.split(",")]
    print(f"[DG-indexed sublinear retrieval] seeds={seeds} scales={scales} D={a.D} g={a.g} G={a.G} c={a.c} "
          f"sigma={a.sigma} tau={a.tau}", flush=True)
    print("  mechanism: DG sparse expansion + hard k-WTA + CA3 conjunction routing -> cleanup within the shard "
          "(host-rate DG projection = declared shortcut; spiking-DG granule layer = named burn-down).", flush=True)

    all_rows = []
    for s in seeds:
        t0 = time.time()
        rows = run_seed(s, scales, a.D, a.g, a.G, a.c, a.sigma, a.tau, a.n_query, a.oos_query)
        all_rows.append(rows)
        print(f"  [seed {s}] done ({time.time()-t0:.0f}s)", flush=True)

    if a.json:
        json.dump({"seeds": seeds, "scales": scales, "rows": all_rows}, open(a.json, "w"), indent=1)

    vd = verdict(all_rows, scales)
    print("\n  ===== AGGREGATE =====", flush=True)
    for V in scales:
        rs = [r for rows in all_rows for r in rows if r["V"] == V]
        print(f"  V={V:>7d}: parity={np.mean([r['parity_agree'] for r in rs]):.3f} "
              f"shard={np.mean([r['rows_dg'] for r in rs]):.1f} rows-speedup={np.mean([r['rows_speedup'] for r in rs]):.0f}x "
              f"wall-speedup={np.mean([r['wall_speedup'] for r in rs]):.1f}x "
              f"scramble={np.mean([r['scramble_agree'] for r in rs]):.4f} "
              f"moat(oos-abstain DG)={np.mean([r['oos_dg_abstain'] for r in rs]):.3f}", flush=True)
    print(f"\n  sublinearity: V grew {vd['v_growth']:.0f}x while shard grew {vd['shard_growth']:.2f}x "
          f"(sublinear={vd['sublinear']}); rows-speedup@Vmax={vd['rows_speedup_Vmax']:.0f}x "
          f"wall-speedup@Vmax={vd['wall_speedup_Vmax']:.1f}x", flush=True)
    print(f"  checks: parity>=0.98={vd['parity_ok']} cost(sublinear+>=10x)={vd['cost_ok']} "
          f"moat(oos-abstain both, 0 new-confab)={vd['moat_ok']} scramble->chance={vd['scramble_ok']}", flush=True)
    print(f"\n  VERDICT: {'GO' if vd['go'] else 'NO-GO/PARTIAL'} -- a DG-like sparse index makes fact retrieval "
          f"{'SUBLINEAR (O(shard) not O(V)) with parity-preserved recall + intact no-confab moat; the cleanup runs over the routed shard exactly as the composer already does, just over ~constant rows instead of all V. Burn-down: replace the host DG projection with the spiking granule-cell WTA (trisynaptic-loop probes).' if vd['go'] else 'DID NOT clear all gates -- read the per-scale block; a fast-but-wrong index is a NO-GO. Iterate m/g/G/sigma or the abstain floor.'}", flush=True)
    return 0 if vd["go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
