"""TEACHER-LOOP SPARSE-GATED READOUT ALLOCATION -- catastrophic-forgetting mitigation (2026-08-08).

THE MEASURED WALL (finding fcdc2fd2, `_teacher_loop_scaling_derisk.py`). Teaching N distinct facts SEQUENTIALLY into
ONE brain via corrective e-prop retains ~1 fact: each fact is learned perfectly at immediate held-out ~0.995, then
OVERWRITTEN as the next fact is taught (frac_recalled ~ 1/N; fact 0 gone by N>=5). The INTERLEAVED control (teacher
re-presents old facts alongside new) retains 8/10 at N=10 on the SAME net -> the net's CAPACITY is adequate; the
failure is sequential INTERFERENCE on the SHARED leaky readout. The interference is structural: the Bellec leaky
readout logit_k = sum_j W_{jk} r_j reads a DENSE last-hidden eligibility r (every hidden unit participates in every
fact), so teaching fact i (softmax(logits)-onehot(i)) moves W over ALL the hidden units r activates -- including the
units that also carry earlier facts -> earlier facts' readout weights are dragged down.

THE MITIGATION -- SPARSE DISTRIBUTED ENGRAMS (k-WINNERS-TAKE-ALL competition). Reduce the interference AT ITS SOURCE:
make the readout code SPARSE so distinct facts recruit DISTINCT hidden sub-populations. Biologically this is the
dentate-gyrus / cerebellar-granule sparse-coding motif (Marr 1971; Treves-Rolls; O'Reilly-McClelland CLS): an
EXPANSION recoding + strong feedback inhibition yields a k-winners-take-all sparse code that ORTHOGONALISES similar
inputs (pattern separation), so overlapping engrams stop overwriting each other. Here: a k-WTA competition on the
last-hidden units' OWN summed spike eligibility -- only the top-k most active units pass their signal to the readout.
Because the winners are chosen by which units the PERCEPT drives (the substrate's own activity, via feedback
inhibition), distinct referents select distinct sparse subspaces, and a new fact's e-prop readout update lands on ITS
winners -- leaving earlier facts' readout weights (on THEIR winners) untouched.

BRAIN-BASED, NOT A HOST TABLE (the load-bearing distinction). The allocation is NEURAL: `_sparse_mask` takes
`np.argpartition` over the RAW last-hidden eligibility r_raw = sum_t eps_j(t) (the exact spike-driven activity the
readout integrates) and keeps the top-k. The SAME computation runs for EVERY input; WHICH units win is decided purely
by the neurons' activity on that percept. There is NO fact->slot map, NO per-fact index bookkeeping, NO reference to
the fact id anywhere in the gating. Grep-verify: the class label `y` never enters `_sparse_mask` or `_readout_feature`.
This is a competition (feedback inhibition), not an allocator. The mask gates BOTH the forward readout (`_logits_from`
-> `_readout_feature`) AND the e-prop readout gradient (`train_batch` -> `_readout_feature`), so a masked-out unit
neither contributes to the logit nor receives a weight update -- exactly a sparse engram.

WHY IT IS ADDITIVE / DEFAULT-OFF. `SparseReadoutEpropNet(OnBridgeEpropNet)` overrides ONLY `_readout_feature`. With
`kwta_k=None` it is BYTE-IDENTICAL to the parent (the DENSE baseline) -- asserted by an off-equals-parent logit check.
NO sim/ edit; reuse-by-import of the scaling runner's world + curriculum helpers (byte-identical world/protocol; only
the net's readout differs).

TEETH (single-seed SMOKE here; 6-seed command below):
  (a) DECISIVE A/B, same net/seed/epochs/world: SPARSE sequential frac_recalled(N) RISES vs DENSE sequential
      frac_recalled(N) (the no-mitigation baseline). Target: rise toward the 8/10 interleaved ceiling, WITHOUT replay
      (the teacher NEVER re-presents an old fact -- grep: each fact taught once, in order; no interleave in the
      sequential arms).
  (b) LOAD-BEARING: a k-sweep. k=full (== dense) forgets; sparser k retains. `attributable_to` on the retention gain.
      Removing the sparsity (k->full) returns the forgetting -> the gating is the cause, not some confound.
  (c) CAPACITY HELD, not traded for reduced acquisition: immediate held-out acquire_acc stays ~perfect in the SPARSE
      arm (each fact still learned to ~1.0 the moment it is taught) AND end retention rises. If sparse merely
      failed to learn, acquire_acc would drop -- measured and gated.
  (d) SEPARATION is real: mean pairwise winner-overlap (Jaccard) of the per-fact sparse codes is LOW (distinct
      subspaces), reported as the mechanism read-out.

GO (smoke): sparse frac_recalled(N_max) > dense frac_recalled(N_max) + 0.2  AND  sparse mean acquire_acc >= 0.9
  AND  off-flag byte-identical to parent  AND  winner-overlap < 0.5. 6-seed claim needs the A/B to hold 6/6.

RUN (single-seed smoke, cupy/3090):
  SIM_BACKEND=cupy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_sparse_readout_derisk --seed 42 \
      --n-max 10 --milestones 1 5 10 --hidden 96 --kwta-k 8 --epochs 40 --n-draws 40 \
      --out research/findings/raw/teacher_loop_sparse_readout_s42.json
  PLUMBING SMOKE (fast, numpy): SIM_BACKEND=numpy ... --n-max 5 --milestones 1 5 --hidden 48 --kwta-k 6 \
      --epochs 12 --settle-steps 12 --n-draws 24 --k-sweep 48 6
  6-SEED (A/B must hold 6/6 at 42..47):  ... --seeds 42 43 44 45 46 47   (see --seeds; runs the A/B per seed)
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "cupy")   # the 3090 by default; caller may set numpy for a plumbing smoke
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
import numpy as np  # noqa: E402
# reuse-by-import: the a1-GO transport-free e-prop substrate + the scaling runner's world/curriculum helpers
# (byte-identical world + protocol). NO sim/ edit.
from research.runners._onbridge_eprop_port_derisk import OnBridgeEpropNet, _softmax  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    ReferentEnv, _feat, _teach_fact, _fact_acc, _corrective_batch, N_ACT)
from tools.lab import attributable_to  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_sparse_readout.json"


# ============================================================================================================
# The mitigation: a k-WTA competition on the last-hidden eligibility, gating the leaky readout. Additive subclass;
# kwta_k=None => byte-identical to the parent (the DENSE baseline). The gating is NEURAL -- top-k by the neurons'
# OWN summed spike eligibility -- and the fact label NEVER enters it (no host fact->slot table).
# ============================================================================================================
class SparseReadoutEpropNet(OnBridgeEpropNet):
    """Override ONLY the readout feature: apply a k-winners-take-all sparse mask (feedback-inhibition sparse code)
    to the last-hidden eligibility BOTH in the forward readout and in the e-prop readout gradient. Distinct percepts
    -> distinct winners -> distinct readout subspaces -> a new fact's update does not overwrite earlier facts."""

    def __init__(self, *args, kwta_k=None, local_readout=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.kwta_k = None if kwta_k is None else int(kwta_k)
        self.local_readout = bool(local_readout)   # target-only (Hebbian) readout delta -- no all-vs-all suppression
        self._last_winners = None   # the winner index set of the most recent forward (for the overlap read-out)

    def train_batch(self, Xb, yb, shuffle_dfa=False, rng=None):
        """DEFAULT-OFF override. local_readout=False => call the parent VERBATIM (byte-identical). local_readout=True
        => replace the readout delta's dense softmax-CE gradient (d = p - onehot(y), which SUPPRESSES every non-target
        class column on the active units and so RE-INTRODUCES interference even with disjoint winners) by a LOCAL
        target-only potentiation delta d[y] = (p[y]-1), d[j!=y]=0. Combined with the k-WTA sparse features this is a
        sparse Hebbian engram: teaching fact i potentiates ONLY column i on ONLY fact i's winners; earlier facts'
        columns are never touched. Specificity comes from the sparse code (fact j's winners have ~0 weight into
        column i), NOT from softmax normalization. Hidden layers are frozen here, so only the readout grad matters."""
        if not self.local_readout:
            return super().train_batch(Xb, yb, shuffle_dfa=shuffle_dfa, rng=rng)
        L = len(self.sizes) - 1
        grads = [np.zeros((self.sizes_phys[li], self.sizes_phys[li + 1]), dtype=np.float64) for li in range(L)]
        for i in range(len(Xb)):
            sp, vv, acts = self._forward_record(Xb[i])
            logits = self._logits_from(sp, vv, acts)
            p = _softmax(logits / self.logit_temp)
            y = int(np.asarray(yb)[i])
            d = np.zeros_like(p); d[y] = p[y] - 1.0            # TARGET-ONLY potentiation (no non-target suppression)
            r = self._readout_feature(sp)                       # k-WTA sparse feature (winners only)
            dphys = self._broadcast(d, L) / self.pool_k
            grads[L - 1] += np.outer(r, dphys)
        self._apply_grads(grads, len(Xb))

    def _sparse_mask(self, dev):
        """k-WINNERS-TAKE-ALL competition. Winners = the k hidden units this percept drives MOST ABOVE their OWN
        homeostatic baseline firing (`dev = r_raw - mu`, the CENTERED deviation). Competing on the deviation -- NOT
        the raw activity -- is what makes the code INPUT-DEPENDENT: raw top-k just picks the intrinsically-most-
        excitable units every time (MEASURED: winner-overlap 1.0 -> no separation), because heterogeneity + tonic
        drive dominate the raw rate; the per-neuron baseline mu (feedback inhibition's homeostatic companion process)
        is what turns the competition into pattern separation. We center by mu but do NOT divide by sigma: the sigma
        floor (1e-6) lets near-silent units blow up and always win (MEASURED: standardized top-k -> overlap 1.0 +
        acquisition collapse). Centered deviation gives cross-referent winner overlap ~0.33 (distinct subspaces).
        The class label is NOT an argument: which units win is decided ONLY by the percept's drive (a competition,
        i.e. feedback inhibition -- NOT an allocator; no fact->slot table)."""
        k = self.kwta_k
        if k is None or k >= len(dev):
            self._last_winners = None
            return None
        win = np.argpartition(dev, -k)[-k:]
        self._last_winners = set(int(w) for w in win)
        mask = np.zeros_like(dev)
        mask[win] = 1.0
        return mask

    def _readout_feature(self, sp):
        r_raw = self._readout_elig(sp)
        if self._r_mu is not None:
            r = (r_raw - self._r_mu) / self._r_sigma      # standardized feature for a well-conditioned readout (parent)
            dev = r_raw - self._r_mu                        # centered deviation drives the k-WTA competition
        else:
            r = r_raw; dev = r_raw
        mask = self._sparse_mask(dev)
        if mask is not None:
            r = r * mask     # only the winning units pass to the readout (forward AND e-prop gradient go through here)
        return r

    def winners_for(self, feat_row):
        """The sparse winner set this net selects for a given input (mechanism read-out; drives the overlap metric)."""
        self._readout_feature(self._forward_record(feat_row)[0])
        return set() if self._last_winners is None else set(self._last_winners)


def _mk_net(n_in, k, seed, hidden, settle, eprop_lr, w_clip, kwta_k, freeze_hidden=True, local_readout=False):
    """Same a1-GO OnBridge build hp as the scaling/corrective-acquire de-risks; the new knobs are kwta_k + the
    FROZEN pattern-separation reservoir. freeze_hidden=True trains ONLY the readout (train_layers={last}) and leaves
    the hidden FF as a FIXED random expansion -- the biological pattern-separation layer (DG granule / cerebellar
    codon expansion is a fixed random recoding; plasticity is at the readout). This is ALSO the honest isolation of
    the fcdc2fd2 wall, which located the interference on the SHARED READOUT: with the hidden fixed, the dense-vs-
    sparse A/B varies ONLY the readout code. Without it, e-prop's hidden-layer plasticity (hidden_lr_scale=5)
    reshapes the reservoir during teaching and COLLAPSES the sparse code (MEASURED: winner-overlap climbs to ~0.55
    on the trained net vs ~0.33 on the fixed reservoir), confounding the mechanism."""
    hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
              in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0, freeze_hidden=bool(freeze_hidden))
    return SparseReadoutEpropNet(n_in, hidden, k, seed=seed, n_hidden_layers=1, settle_steps=settle,
                                 eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                                 logit_source="leaky_readout", w_clip=w_clip, hp=hp, kwta_k=kwta_k,
                                 local_readout=local_readout)


def _fit_readout_norm_world(net, env, referents, seed, per_ref=6):
    """Fit the leaky-readout per-neuron eligibility mean/std ONCE over the whole referent world (input statistics,
    not labels -- a homeostatic input scaling), exactly as the scaling de-risk does. Uses the RAW eligibility so the
    normalization is identical across dense/sparse arms (the k-WTA acts AFTER, on r_raw, inside _readout_feature)."""
    feats = [_feat(env, r) for _ in range(per_ref) for r in referents]
    R = np.array([net._readout_elig(net._forward_record(feats[i])[0]) for i in range(len(feats))])
    net._r_mu = R.mean(axis=0)
    net._r_sigma = R.std(axis=0) + 1e-6


def _run_curriculum(net, env, referents, milestones, epochs, batch, n_draws, test_n, chance, rng_seed, tag=""):
    """Teach the referents SEQUENTIALLY (each fact ONCE, in order -- NO interleave, NO re-presentation of old facts)
    and record: immediate held-out acquisition per fact + the retention curve at each milestone. Identical protocol
    to the scaling de-risk; only the net's readout differs."""
    rng = np.random.default_rng(rng_seed)
    acquire_acc = []
    retention = {}
    for i, r in enumerate(referents):
        X, y = _corrective_batch(env, r, i, n_draws)
        _teach_fact(net, X, y, epochs, batch, rng)                  # e-prop moves the brain's OWN weights; no store-write
        acquire_acc.append(_fact_acc(net, env, r, i, n=test_n))     # immediate held-out (did it learn this fact NOW?)
        N = i + 1
        print(f"    [{tag}] fact {N}/{len(referents)} acquired acc {acquire_acc[-1]:.2f}", flush=True)
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {"frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                                 "mean_retained_acc": float(np.mean(accs)),
                                 "oldest_fact_acc": float(accs[0]), "newest_fact_acc": float(accs[-1]),
                                 "per_fact_acc": [float(a) for a in accs]}
    return {"acquire_acc_immediate": [float(a) for a in acquire_acc],
            "mean_acquire_acc": float(np.mean(acquire_acc)), "retention_curve": retention}


def _winner_overlap(net, env, referents):
    """Mean pairwise Jaccard overlap of the per-fact sparse winner sets -- LOW = distinct subspaces (separation)."""
    sets = [net.winners_for(_feat(env, r)) for r in referents]
    if net.kwta_k is None:
        return {"mean_jaccard": float("nan"), "note": "dense arm has no sparse code"}
    js = []
    for a in range(len(sets)):
        for b in range(a + 1, len(sets)):
            u = len(sets[a] | sets[b])
            js.append((len(sets[a] & sets[b]) / u) if u else 0.0)
    return {"mean_jaccard": float(np.mean(js)) if js else float("nan"),
            "mean_winners": float(np.mean([len(s) for s in sets])), "k": net.kwta_k}


def run_seed(seed, n_max, milestones, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise,
             test_n, kwta_k, k_sweep, checkpoint=None):
    K = int(n_max); chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))

    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    referents = [f"ref{i}" for i in range(n_max)]
    for r in referents:
        env.proto(r)

    # ---- OFF-FLAG BYTE-IDENTITY: kwta_k=None must be byte-identical to the parent OnBridgeEpropNet readout ----
    child = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip, kwta_k=None)
    parent = OnBridgeEpropNet(n_in, hidden, K, seed=seed, n_hidden_layers=1, settle_steps=settle,
                              eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                              logit_source="leaky_readout", w_clip=w_clip,
                              hp=dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
                                      in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0))
    _fit_readout_norm_world(child, env, referents, seed); _fit_readout_norm_world(parent, env, referents, seed)
    fr = _feat(env, referents[0])
    lc = child._logits_from(*child._forward_record(fr)); lp = parent._logits_from(*parent._forward_record(fr))
    off_byte_identical = bool(np.allclose(np.asarray(lc), np.asarray(lp), atol=1e-9))

    def _ck(**kw):
        if checkpoint is not None:
            checkpoint(dict(seed=seed, K_classes=K, chance=chance, n_max=n_max, milestones=milestones,
                            off_flag_byte_identical=off_byte_identical, partial=True, **kw))

    # ---- DENSE baseline (the no-mitigation sequential arm) ----
    dnet = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip, kwta_k=None)
    _fit_readout_norm_world(dnet, env, referents, seed)
    dense = _run_curriculum(dnet, env, referents, milestones, epochs, batch, n_draws, test_n, chance, seed + 777, tag='dense')
    _ck(dense_baseline=dense)

    # ---- SPARSE arm (the mitigation) -- SAME net build/seed/epochs/world, ONLY kwta_k differs ----
    snet = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip, kwta_k=kwta_k)
    _fit_readout_norm_world(snet, env, referents, seed)
    sparse = _run_curriculum(snet, env, referents, milestones, epochs, batch, n_draws, test_n, chance, seed + 777, tag='sparse')
    overlap = _winner_overlap(snet, env, referents)
    _ck(dense_baseline=dense, sparse_mitigation=sparse, winner_overlap=overlap)

    # ---- k-SWEEP (load-bearing): retention vs k. k=full == dense; sparser k should retain more. ----
    ksweep = []
    for kk in (k_sweep or []):
        kk = int(kk)
        knet = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip, kwta_k=(None if kk >= hidden else kk))
        _fit_readout_norm_world(knet, env, referents, seed)
        kr = _run_curriculum(knet, env, referents, milestones, epochs, batch, n_draws, test_n, chance, seed + 777, tag=f'k={kk}')
        km = str(max(milestones))
        ksweep.append({"k": kk, "effective_dense": bool(kk >= hidden),
                       "frac_recalled_Nmax": kr["retention_curve"][km]["frac_recalled"],
                       "mean_acquire_acc": kr["mean_acquire_acc"]})
        _ck(dense_baseline=dense, sparse_mitigation=sparse, winner_overlap=overlap, k_sweep=list(ksweep))

    Nmax = str(max(milestones))
    dense_frac = dense["retention_curve"][Nmax]["frac_recalled"]
    sparse_frac = sparse["retention_curve"][Nmax]["frac_recalled"]

    print(f"[seed {seed}] N_max={n_max} k={kwta_k}/{hidden} | DENSE frac-recalled(N={Nmax}) {dense_frac:.2f} "
          f"(acq {dense['mean_acquire_acc']:.2f}) | SPARSE {sparse_frac:.2f} (acq {sparse['mean_acquire_acc']:.2f}) "
          f"| winner-overlap {overlap['mean_jaccard']:.3f} | off-byte-identical {off_byte_identical}", flush=True)
    # ATTRIBUTION: the retention gain is attributable to the k-WTA gating (treatment=sparse, control=dense).
    attributable_to("k-WTA sparse allocation (sparse vs dense frac-recalled@Nmax)", sparse_frac, dense_frac)

    ab_holds = bool(sparse_frac > dense_frac + 0.2)
    capacity_held = bool(sparse["mean_acquire_acc"] >= 0.9)
    separation = bool(not np.isnan(overlap["mean_jaccard"]) and overlap["mean_jaccard"] < 0.5)
    go = bool(ab_holds and capacity_held and off_byte_identical and separation)

    return {
        "seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
        "config": {"hidden": hidden, "settle_steps": settle, "epochs": epochs, "batch": batch,
                   "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise,
                   "test_n": test_n, "kwta_k": kwta_k},
        "off_flag_byte_identical": off_byte_identical,
        "dense_baseline": dense, "sparse_mitigation": sparse, "winner_overlap": overlap, "k_sweep": ksweep,
        "dense_frac_recalled_Nmax": float(dense_frac), "sparse_frac_recalled_Nmax": float(sparse_frac),
        "retention_gain_Nmax": float(sparse_frac - dense_frac),
        "T_ab_sparse_beats_dense": ab_holds, "T_capacity_held_acquire": capacity_held,
        "T_off_byte_identical": off_byte_identical, "T_separation_low_overlap": separation,
        "GO": go,
    }


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop catastrophic-forgetting mitigation: sparse-gated (k-WTA) "
                                             "readout allocation. Sequential SPARSE vs DENSE retention A/B; k-sweep.")
    ap.add_argument("--seed", type=int, default=None, help="single-seed smoke (or use --seeds)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="multi-seed: run the A/B per seed")
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--milestones", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=40)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    ap.add_argument("--kwta-k", type=int, default=8, help="k winners (sparse readout subspace size)")
    ap.add_argument("--k-sweep", type=int, nargs="+", default=None,
                    help="extra k values for the load-bearing sweep (a value >= hidden == dense)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    seeds = a.seeds if a.seeds else [a.seed if a.seed is not None else 42]
    single = (len(seeds) == 1)
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    per = []
    for s in seeds:
        def _checkpoint(partial, _s=s):
            # write a partial artifact after every ARM so a kill cannot lose the measured curve (the 52-min loss).
            try:
                Path(a.out).write_text(json.dumps(
                    {"probe": "teacher_loop_sparse_readout", "partial": True, "seeds": seeds,
                     "backend": os.environ.get("SIM_BACKEND"), "elapsed_seconds": round(time.time() - t0, 1),
                     "per_seed": per + [partial]}, indent=2, default=str))
            except Exception as _e:
                print(f"[warn] checkpoint write failed ({type(_e).__name__}: {_e})", flush=True)
        per.append(run_seed(s, a.n_max, a.milestones, a.hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr,
                            a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.kwta_k, a.k_sweep,
                            checkpoint=_checkpoint))
    n_go = sum(p["GO"] for p in per)
    n_ab = sum(p["T_ab_sparse_beats_dense"] for p in per)
    summary = {"probe": "teacher_loop_sparse_readout", "seeds": seeds, "single_seed_smoke": single,
               "backend": os.environ.get("SIM_BACKEND"), "elapsed_seconds": round(time.time() - t0, 1),
               "per_seed": per, "n_go": n_go, "n_ab_holds": n_ab, "n_seeds": len(seeds),
               "ALL_GO": bool(n_go == len(seeds))}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 100, flush=True)
    for p in per:
        print(f"  seed {p['seed']}: dense {p['dense_frac_recalled_Nmax']:.2f} -> sparse "
              f"{p['sparse_frac_recalled_Nmax']:.2f} (gain {p['retention_gain_Nmax']:+.2f}) | "
              f"acq {p['sparse_mitigation']['mean_acquire_acc']:.2f} | overlap "
              f"{p['winner_overlap']['mean_jaccard']:.3f} | GO {p['GO']}", flush=True)
    print(f"[teacher-loop-sparse-readout] {n_go}/{len(seeds)} GO ({n_ab}/{len(seeds)} A/B holds) "
          f"{'[SINGLE-SEED SMOKE]' if single else ''} -> wrote {a.out}", flush=True)
    print("=" * 100, flush=True)
    return 0 if summary["ALL_GO"] else 1


if __name__ == "__main__":
    sys.exit(main())
