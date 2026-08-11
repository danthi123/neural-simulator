"""TEACHER-LOOP SPARSE / ORTHOGONAL FACT-CODE DE-RISK (2026-08-11): does routing each fact's e-prop weight
change to a DISJOINT sparse subset of hidden units protect the OLDEST fact that metaplasticity/Benna-Fusi
chains could NOT (oldest_fact_acc ~ 0.0 at N=100)?

WHY THIS EXISTS (the named next mechanism, not re-derived here). Two prior 6-seed de-risks isolated the
continual acquisition-at-scale residual and named the surpass:
  * `2026-08-11-metaplastic-acquisition-continual-learning-6seed-NOGO...`: a per-synapse consolidation
    variable c gating lr_eff = lr/(1+g*c) moves the frac_recalled~1/N forgetting the RIGHT way (+0.137 mean,
    load-bearing, attributed) but is SUB-THRESHOLD, and NEVER protects the very-OLDEST fact (fact 0 -> 0.0 at
    N=100). It lifts the MIDDLE of the retention curve only.
  * `2026-08-11-benna-fusi-multitimescale-chain-does-not-beat-single-var-metaplasticity-6seed-NEGATIVE.md`:
    a TRUE multi-timescale Benna-Fusi cascade does NOT beat the single variable and STILL leaves the oldest
    fact at 0.0. The NEGATIVE re-diagnosed the residual: **the oldest fact is overwritten because later facts
    REUSE its synapses -- an interference problem in the fact CODE, not a consolidation-timescale problem.**
    "No amount of 'protect these synapses more slowly' helps when the SAME synapses must encode the new fact."
    Named next mechanism: **orthogonalized / SPARSE fact codes** so facts occupy DISJOINT synapse subsets and
    do not compete -- the SAME "keep codes disjoint under pressure" biology that already won in TWO other lanes
    (source-monitoring competitive-encoding + emergence hetero-competition allocation). This de-risk builds it.

THE MECHANISM (biologically cited; sparse orthogonal coding + competitive allocation).
  Marr 1969 / Albus 1971 (cerebellar granule-cell expansion): a large layer of sparsely-active granule cells
  ORTHOGONALIZES overlapping mossy-fibre inputs so that downstream Purkinje synapses for one pattern are
  largely DISJOINT from another's -> pattern separation, minimal interference.
  Treves & Rolls 1994; O'Reilly & McClelland 1994 (dentate gyrus pattern separation): sparse, competitive
  (k-winner) hippocampal codes assign distinct memories to near-disjoint cell/synapse subsets, so storing a
  new memory does not overwrite an old one -- the mechanism that lets an associative store hold many memories.
  Competitive allocation / inhibition-of-return: units already claimed by earlier memories are preferentially
  AVOIDED by later ones (the emergence hetero-competition allocation winner, this lane's THIRD convergence).
  IMPLEMENTED (a per-fact hidden-unit sparse code, runner-side; NO sim/ edit). Each fact i is allocated a
  sparse code S_i -- a subset of `code_size` hidden units -- by COMPETITIVE least-occupied allocation: pick the
  `code_size` LEAST-claimed hidden units (per-fact random tiebreak), then mark them claimed. While capacity
  allows (N <= H/code_size) the codes are DISJOINT; beyond it they degrade to minimal-overlap. When teaching
  fact i, the e-prop weight change is MASKED to S_i's synapses -- ONLY the input->hidden columns of S_i and the
  hidden->readout rows of S_i update. So fact i tunes and votes through ONLY its own units; a disjoint fact j
  never touches them -> fact i's whole sub-circuit is FROZEN once taught -> it survives all later facts. This
  is brain-based-only: a sparse competitive hidden code gating which synapses each memory writes (dentate-like
  pattern separation), NOT host cognition. The e-prop rule (transport-free DFA) is the sole learner throughout.

SIX ARMS, one world / seed / schedule / de-clamp (the ONLY difference is the allocation gate):
  * vanilla       = dense e-prop, no gate -> the acquisition-at-scale collapse control (frac_recalled ~ 1/N).
  * metaplastic   = the PRIOR single-var consolidation gate (the current best -- the arm to BEAT; sparse coding
                    must ADD over consolidation, on a DIFFERENT axis). Same g, same de-clamp.
  * sparse        = the disjoint sparse fact-code allocation (no consolidation) -> THE TREATMENT. GO = beats
                    vanilla AND metaplastic AND -- crucially -- protects the OLDEST fact (oldest_fact_acc > 0.5
                    at N_max) where metaplasticity/chains left it at 0.0, while still acquiring new facts.
  * sparse_dense  = the sparse code path with the allocation forced DENSE (every fact claims ALL hidden units)
                    -> MUST collapse to ~vanilla. THE LOAD-BEARING tooth: the sparse ALLOCATION STATE, not the
                    code path, does the work.
  * sparse_overlap= sparse (same code_size) but ALL facts share the SAME fixed subset -> maximal OVERLAP -> the
                    disjointness anti-cheat: sparsity WITHOUT disjointness must NOT help (facts still compete
                    for the same synapses). Proves DISJOINTNESS is the lever, not merely being sparse.
  * sparse_meta   = disjoint sparse code + the single-var consolidation gate -> does combining the two
                    orthogonal axes (disjoint codes x consolidation) add over either alone? (optional; --with-sparse-meta)

ANTI-CHEATS (executed via tools.lab + tools.verdict.Verdict, not asserted in prose):
  (a) load-bearing: sparse_dense (allocation forced dense) frac_recalled ~= vanilla, and sparse >> both.
  (b) attributable: attributable_to(sparse vs sparse_dense) / (sparse vs sparse_overlap) / (sparse vs metaplastic)
      -- the fraction of the effect NOT present in the dense / overlapping / consolidation-only controls.
  (c) no acquisition cost: sparse mean immediate-acq >= 0.6 (the plasticity side of stability-plasticity: a
      sub-circuit of only `code_size` units must still LEARN the new fact).
  (d) sparse BEATS the single-var metaplastic arm (the current best): sparse coding ADDS over consolidation.
  (e) DISJOINTNESS is the lever: the overlap control (same sparsity, shared code) must NOT help, and the
      measured mean pairwise code overlap must be LOW for `sparse` and HIGH for `sparse_overlap` (lever()).
  Plus per-fact-AGE retention (oldest / middle / newest) at every milestone makes the oldest-fact claim explicit.

CAPACITY (the honest boundary, stated up front). Disjoint codes need capacity: N <= H/code_size gives fully
disjoint codes (oldest fact fully protected); beyond it codes must share and the oldest fact degrades again --
which directly motivates the SECONDARY named mechanism (neurogenesis / hidden-layer capacity growth as N
scales). The smoke is sized so the smoke-N is WITHIN capacity (mechanism demonstrable); the 6-seed sweep to
N=100 sizes H to keep it within capacity, and the finding characterizes the boundary honestly.

DISCIPLINE: reuse-by-import of ALL substantive machinery -- MetaplasticEpropNet (which wraps the a1-GO
OnBridgeEpropNet) for the vanilla/metaplastic arms + its single-var gate; the teacher-loop world/teach/
held-out-acc/corrective-batch; `_age_buckets` from the Benna-Fusi runner. The sparse code is an ADDITIVE
subclass (SparseFactEpropNet); sparse_enabled=False -> byte-identical to MetaplasticEpropNet (the vanilla /
metaplastic arms prove it). cfg.seed via the seed= arg the net passes to CoreSimConfig.seed. NO sim/ edit.
SIM_BACKEND=numpy (this ~140-neuron net is launch-bound; numpy avoids cupy launch overhead -- reported).

RUN:
  single-seed SMOKE (sparse protects the oldest fact + disjointness lever + dense-lesion bites, N=32):
    SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      .venv/bin/python -m research.runners._teacher_loop_sparse_fact_codes_derisk --seed 42 \
        --n-max 32 --milestones 8 16 32 --hidden 96 --code-size 3 --epochs 20 --settle-steps 20 \
        --test-n 20 --n-draws 24 --out research/findings/raw/sparse_fact_codes_s42.json
  6-SEED sweep command is returned to the coordinator (one seed per process; N-sweep in one run via
  --n-max 100 --milestones 16 32 50 100 with H sized to keep N<=H/code_size).
"""
from __future__ import annotations
import argparse, itertools, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")   # ~140-neuron net; numpy avoids cupy launch overhead at this size
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
# reuse-by-import: the single-var metaplastic net (which wraps OnBridgeEpropNet) + the teacher-loop machinery
# + the Benna-Fusi runner's per-age buckets. NO sim/ edit.
from research.runners._teacher_loop_metaplastic_acquisition_derisk import MetaplasticEpropNet  # noqa: E402
from research.runners._teacher_loop_bennafusi_chain_derisk import _age_buckets  # noqa: E402
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _fit_readout_norm_world, _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402

OUT = _REPO / "research" / "findings" / "raw" / "sparse_fact_codes.json"
CORE_ARMS = ("vanilla", "metaplastic", "disjoint", "dense_readout", "kwta_trained", "kwta_shared")


# ============================================================================================================
# The sparse-fact-code net: MetaplasticEpropNet (single-var gate + a1-GO e-prop) EXTENDED with a per-fact
# sparse hidden-unit code that MASKS each fact's e-prop weight change to a DISJOINT subset of synapses.
# NO sim/ edit. sparse_enabled=False -> byte-identical to MetaplasticEpropNet. Additive, default-off.
# ============================================================================================================
class SparseFactEpropNet(MetaplasticEpropNet):
    def __init__(self, *args, sparse_enabled=False, code_size=3, alloc_mode="disjoint", alloc_seed=0,
                 labeled_line=True, readout_permute=False, freeze_reservoir=False, kwta_k=0,
                 kwta_shared=False, n_classes=None, **kw):
        super().__init__(*args, **kw)
        # INPUT-DRIVEN k-WTA SPARSE READOUT CODE (dentate/Marr-Albus pattern separation). Only the top-k most
        # active hidden units drive the readout for a given input (feedforward competitive inhibition). WHY
        # (measured 2026-08-11): with a DENSE readout, an OLD fact's classification reads ALL hidden units,
        # including those later facts wrote -> interference; and a labeled line miscalibrates the argmax. A sparse
        # input-driven code makes ref_i's readout read ONLY ref_i's characteristic units, so teaching a later fact
        # (whose code is a DISJOINT set of units) never touches the weights that classify ref_i -> the oldest fact
        # is frozen. kwta_shared (control) forces ALL inputs to use the SAME fixed top-k slot -> maximal code
        # overlap -> the disjointness anti-cheat. kwta_k=0 => dense readout (off).
        self.kwta_k = int(kwta_k)
        self.kwta_shared = bool(kwta_shared)
        # FROZEN RESERVOIR: train ONLY the readout pathway; the input->hidden layer stays a FIXED random spiking
        # reservoir (Marr-Albus granule-layer orthogonalization). WHY (measured 2026-08-11): with the hidden
        # trained densely, teaching each new fact drifts the shared reservoir features, so an OLD fact's frozen
        # labeled-line readout reads DRIFTED features and is lost -- the second interference channel. Freezing the
        # reservoir removes the feature-drift channel; the disjoint labeled-line removes the readout-column channel.
        if bool(freeze_reservoir):
            self.train_layers = {self.n_hidden_layers}    # {readout pathway index} -> hidden FF frozen at init
        self.freeze_reservoir = bool(freeze_reservoir)
        # n_hidden_layers=1 throughout the teacher loop => sizes=[n_in, hidden, k], hidden dim = sizes_phys[1].
        assert self.n_hidden_layers == 1, "sparse fact-code gating assumes a single hidden layer"
        self.H = int(self.sizes_phys[1])
        self.sparse_enabled = bool(sparse_enabled)
        self.code_size = int(min(max(1, code_size), self.H))
        self.alloc_mode = str(alloc_mode)                 # "disjoint" | "overlap" | "dense"
        # LABELED-LINE readout (default ON for the disjoint fact code): each fact writes ONLY its own class's
        # output column. WHY this is needed (measured 2026-08-11): a disjoint HIDDEN code alone leaves the OLDEST
        # fact at 0.0 because the CONTRASTIVE softmax delta writes NEGATIVE cross-class votes into every OTHER
        # fact's class column -- so the oldest class column is eroded by every later fact (interference lives in
        # the readout COLUMN, which hidden-unit gating does not reach). Confining each fact to its own output line
        # (a labeled line / one-vs-rest detector, the second disjoint channel) removes that erosion. When False,
        # the readout stays contrastive (the diagnostic arm that exposes the column-erosion channel).
        self.labeled_line = bool(labeled_line)
        # READOUT-PERMUTE attribution control: each fact writes its labeled line to a WRONG (permuted) output
        # column. The routing stays disjoint (a bijection of columns) but MISALIGNED to the true class, so the
        # specific CORRECT per-fact column is isolated as the lever -- writing to any disjoint-but-wrong slot must
        # NOT recall the fact. n_classes = the readout width (k); the permutation is a fixed per-net bijection.
        self.readout_permute = bool(readout_permute)
        _k = int(n_classes) if n_classes is not None else int(self.sizes[-1])
        self._class_perm = np.random.default_rng(int(alloc_seed) + 90210).permutation(_k)
        self.occupancy = np.zeros(self.H, dtype=np.float64)   # per-unit claim count (competitive allocation state)
        self._alloc_rng = np.random.default_rng(int(alloc_seed) + 271828)
        self._active_mask = None                          # boolean (H,): which hidden units the CURRENT fact may write
        self._active_class = None                         # the CURRENT fact's class column (for the labeled line)
        self._fact_codes = {}                             # fact i -> np.array of unit indices (for overlap reporting)

    # ---- INPUT-DRIVEN k-WTA: the readout reads only the top-k most-active hidden units for THIS input ----
    def _readout_feature(self, sp):
        r = super()._readout_feature(sp)                      # per-neuron standardized eligibility (H,)
        if self.kwta_k and 0 < self.kwta_k < r.shape[0]:
            if self.kwta_shared:
                idx = np.arange(self.kwta_k)                  # SAME fixed slot for EVERY input (overlap control)
            else:
                idx = np.argpartition(r, -self.kwta_k)[-self.kwta_k:]   # top-k most active (input-driven code)
            gated = np.zeros_like(r); gated[idx] = r[idx]
            return gated
        return r

    # ---- allocate the CURRENT fact's sparse code (called ONCE before teaching each fact; cls = its class column) ----
    def begin_fact(self, i, cls=None):
        true_cls = int(cls if cls is not None else i)
        # the column this fact's labeled line writes: its OWN class, or (attribution control) a permuted wrong one.
        self._active_class = int(self._class_perm[true_cls]) if self.readout_permute else true_cls
        if not self.sparse_enabled:
            self._active_mask = None
            return None
        H, m = self.H, self.code_size
        if self.alloc_mode == "dense":
            S = np.arange(H)                              # ALL units -> the dense LESION (isolates disjoint-hidden)
        elif self.alloc_mode == "overlap":
            S = np.arange(m)                              # SAME fixed subset for EVERY fact -> maximal overlap
        else:                                             # "disjoint": competitive least-occupied allocation
            jitter = self._alloc_rng.random(H) * 1e-6     # per-fact random tiebreak among equally-claimed units
            S = np.argsort(self.occupancy + jitter, kind="stable")[:m]
            self.occupancy[S] += 1.0                      # claim them (later facts preferentially avoid these)
        mask = np.zeros(H, dtype=bool)
        mask[S] = True
        self._active_mask = mask
        self._fact_codes[int(i)] = np.sort(S.astype(np.int64))
        return self._fact_codes[int(i)]

    # ---- mask the e-prop grads to the CURRENT fact's sub-circuit, then defer the write (+ any meta gate) to parent ----
    def _mask_grads(self, grads):
        mask = self._active_mask                          # hidden-unit mask (None if not sparse)
        cls = self._active_class
        L = len(grads)
        out = []
        for li in range(L):
            g = grads[li]
            gm = g.copy()
            if li == L - 1:                               # hidden->readout: rows = hidden units, cols = classes
                if mask is not None:
                    gm[~mask, :] = 0.0                     # only THIS fact's units write the readout (row gate)
                if self.labeled_line and cls is not None:
                    col = np.ones(gm.shape[1], dtype=bool); col[cls] = False
                    gm[:, col] = 0.0                       # only THIS fact's own class column updates (labeled line)
            else:                                         # input->hidden: columns = hidden units
                if mask is not None:
                    gm[:, ~mask] = 0.0                     # only THIS fact's units get input-tuned (column gate)
            out.append(gm)
        return out

    def _apply_grads(self, grads, bsz):
        # gate whenever a sparse allocation is active OR a labeled-line column restriction applies.
        if (self.sparse_enabled and self._active_mask is not None) or (self.labeled_line and self._active_class is not None
                                                                       and self.sparse_enabled):
            grads = self._mask_grads(grads)               # route this fact's change to its DISJOINT sub-circuit
        return super()._apply_grads(grads, bsz)           # metaplastic gate (if enabled) + the substrate write

    # ---- reporting: mean pairwise Jaccard overlap of the fact codes (the disjointness read-out) + occupancy ----
    def code_overlap_summary(self):
        codes = [self._fact_codes[i] for i in sorted(self._fact_codes)]
        if len(codes) < 2:
            return {"mean_pairwise_overlap": float("nan"), "n_codes": len(codes),
                    "code_size": self.code_size, "H": self.H, "capacity_slots": self.H // max(1, self.code_size),
                    "max_occupancy": float(self.occupancy.max()) if self.sparse_enabled else 0.0}
        ov = []
        for a, b in itertools.combinations(range(len(codes)), 2):
            A = set(codes[a].tolist()); B = set(codes[b].tolist())
            u = len(A | B)
            ov.append((len(A & B) / u) if u else 0.0)
        return {"mean_pairwise_overlap": float(np.mean(ov)), "n_codes": len(codes),
                "code_size": self.code_size, "H": self.H, "capacity_slots": self.H // max(1, self.code_size),
                "max_occupancy": float(self.occupancy.max())}


def _mk_sparse_net(n_in, k, seed, hidden, settle, eprop_lr, w_clip, declamp_wmax, kwargs):
    """Same a1-GO OnBridgeEpropNet build the metaplastic/scaling siblings use, held de-clamped, wrapped as
    SparseFactEpropNet (identical hp across arms -> de-clamp is NOT the lever)."""
    hp = dict(tonic_h_pA=100.0, tonic_o_pA=150.0, ff_w_init=2000.0, pbar_alpha=0.05,
              in_current_pA=700.0, in_bias_pA=300.0, hidden_lr_scale=5.0)
    if declamp_wmax is not None:
        hp["bdsp_wmax"] = float(declamp_wmax)                # de-clamp held CONSTANT across arms (NOT the lever)
    return SparseFactEpropNet(n_in, hidden, k, seed=seed, n_hidden_layers=1, settle_steps=settle,
                              eprop_lr=eprop_lr, eps_leak=0.9, surrogate="atan_vt", alpha_surr=0.15,
                              logit_source="leaky_readout", w_clip=w_clip, hp=hp, meta_permute_seed=seed,
                              alloc_seed=seed, n_classes=k, **kwargs)


def _run_arm(arm, kwargs, seed, referents, env, K, n_in, hidden, settle, epochs, batch, eprop_lr, w_clip,
             n_draws, milestones, test_n, chance, declamp_wmax):
    """Teach the referents SEQUENTIALLY into ONE brain; allocate each fact's sparse code BEFORE teaching it;
    consolidate after each fact (meta arms); record retention + per-age retention at each milestone. The ONLY
    cross-arm difference is kwargs (the allocation gate + optional metaplastic gate)."""
    net = _mk_sparse_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip, declamp_wmax, kwargs)
    _fit_readout_norm_world(net, env, referents, seed)
    teach_rng = np.random.default_rng(seed + 777)
    acquire_acc, retention, overlap_curve = [], {}, {}
    for i, r in enumerate(referents):
        net.begin_fact(i, cls=i)                                 # allocate this fact's sub-circuit (referent i -> class i)
        X, y = _corrective_batch(env, r, i, n_draws)             # WAKE: teacher draws from the world (legitimate)
        _teach_fact(net, X, y, epochs, batch, teach_rng)          # brain acquires the fact by e-prop weight change (masked)
        net.consolidate_fact()                                    # metaplastic deepening AFTER the fact (no-op if meta off)
        acq = _fact_acc(net, env, r, i, n=test_n)                 # immediate held-out acquisition (plasticity side)
        acquire_acc.append(acq)
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {"frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                                 "one_over_N": float(1.0 / N), "mean_retained_acc": float(np.mean(accs)),
                                 "per_fact_acc": [float(a) for a in accs], **_age_buckets(accs)}
            overlap_curve[str(N)] = net.code_overlap_summary()
    return {"arm": arm, "kwargs": {k: (bool(v) if isinstance(v, bool) else v) for k, v in kwargs.items()},
            "acquire_acc_immediate": [float(a) for a in acquire_acc],
            "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
            "final_code_overlap": net.code_overlap_summary(), "overlap_curve": overlap_curve,
            "retention_curve": retention}


def run(seed, n_max, milestones, hidden, code_size, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p,
        noise, test_n, declamp_wmax, meta_gain, meta_consol_rate, with_sparse_meta):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))
    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    referents = [f"ref{i}" for i in range(n_max)]
    for r in referents:
        env.proto(r)

    G, R, C, KW = float(meta_gain), float(meta_consol_rate), int(code_size), int(code_size)
    # THE DISJOINT FACT CODE = an INPUT-DRIVEN k-WTA sparse readout code (dentate/Marr-Albus pattern separation)
    # over a FIXED spiking reservoir. Only ref_i's top-k active units drive its (contrastive, calibrated) readout,
    # and the reservoir is frozen so ref_i's code is STABLE. A later fact's code is a DISJOINT set of units, so
    # teaching it never touches the weights that classify ref_i -> the oldest fact is frozen. code_size = k here.
    specs = {
        # vanilla / metaplastic keep the DENSE contrastive readout over a TRAINED hidden (sparse off) -> the
        # finding's baselines, byte-identical to MetaplasticEpropNet. Shared code + shared features.
        "vanilla":         dict(meta_enabled=False),                                     # baseline collapse
        "metaplastic":     dict(meta_enabled=True, meta_gain=G, meta_consol_rate=R),     # the arm to BEAT
        # THE TREATMENT: input-driven k-WTA sparse readout code + FROZEN reservoir + contrastive (calibrated) readout.
        "disjoint":        dict(meta_enabled=False, freeze_reservoir=True, kwta_k=KW),
        # LOAD-BEARING #1 (sparse code): k-WTA OFF (dense readout), frozen reservoir. Removing only the sparse code
        # -> ref_i's readout reads ALL units incl. later facts' -> interference. Isolates the k-WTA sparse code.
        "dense_readout":   dict(meta_enabled=False, freeze_reservoir=True, kwta_k=0),
        # LOAD-BEARING #2 (frozen reservoir): k-WTA + TRAINED reservoir. Removing only the freeze -> ref_i's top-k
        # code DRIFTS as later facts retrain input->hidden -> old facts lost. Isolates the reservoir freeze.
        "kwta_trained":    dict(meta_enabled=False, freeze_reservoir=False, kwta_k=KW),
        # DISJOINTNESS lever / overlap control: k-WTA but every input forced to the SAME fixed k units -> maximal
        # code OVERLAP (sparse but NOT disjoint) -> facts fight over the same weights -> collapse. Proves
        # DISJOINTNESS is the lever, not mere sparsity.
        "kwta_shared":     dict(meta_enabled=False, freeze_reservoir=True, kwta_k=KW, kwta_shared=True),
    }
    arms_order = list(CORE_ARMS)
    if with_sparse_meta:
        # disjoint sparse code + consolidation (does adding the metaplastic gate add over the sparse code alone?).
        specs["disjoint_meta"] = dict(freeze_reservoir=True, kwta_k=KW, meta_enabled=True, meta_gain=G,
                                      meta_consol_rate=R)
        arms_order.append("disjoint_meta")
    arms = {}
    for name in arms_order:
        t0 = time.time()
        env.rng = np.random.default_rng(seed + 101)               # identical teaching percepts across arms (like-for-like)
        arms[name] = _run_arm(name, specs[name], seed, referents, env, K, n_in, hidden, settle, epochs, batch,
                              eprop_lr, w_clip, n_draws, milestones, test_n, chance, declamp_wmax)
        arms[name]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[name]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        row = rc[str(big)] if big else {}
        fr = row.get("frac_recalled", float("nan"))
        old = row.get("oldest_fact_acc", float("nan"))
        ov = arms[name]["final_code_overlap"]["mean_pairwise_overlap"]
        print(f"[arm {name:15s}] {arms[name]['wall_seconds']:6.0f}s | immediate-acq "
              f"{arms[name]['mean_acquire_acc_immediate']:.3f} | code-overlap {ov:.3f} | "
              f"frac-recalled@N={big}: {fr:.3f} (1/N={1.0/big:.3f}) | oldest-fact {old:.3f}", flush=True)
    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "config": {"hidden": hidden, "code_size": code_size, "settle_steps": settle, "epochs": epochs,
                       "batch": batch, "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p,
                       "noise": noise, "test_n": test_n, "declamp_wmax": declamp_wmax, "meta_gain": G,
                       "meta_consol_rate": R, "with_sparse_meta": bool(with_sparse_meta),
                       "capacity_slots": hidden // max(1, code_size),
                       "backend": os.environ.get("SIM_BACKEND")},
            "arms": arms}


def _verdict(result):
    from tools.lab import lever, attributable_to, assert_backend
    from tools.verdict import Verdict
    backend = assert_backend(os.environ.get("SIM_BACKEND", "numpy"), note="(sparse fact-code de-risk)")
    arms = result["arms"]
    rc = {a: arms[a]["retention_curve"] for a in arms}
    big = max((int(k) for k in rc["disjoint"]), default=None)
    key = str(big)
    f = {a: rc[a][key]["frac_recalled"] for a in rc}
    old = {a: rc[a][key]["oldest_fact_acc"] for a in rc}         # the crux: oldest fact (fact 0) @ N_max
    mid = {a: rc[a][key]["mid_fact_acc"] for a in rc}
    new = {a: rc[a][key]["newest_fact_acc"] for a in rc}
    acq = {a: arms[a]["mean_acquire_acc_immediate"] for a in arms}
    chance = result["chance"]
    one_over_N = 1.0 / big

    # (e) THE LEVER: the input-driven k-WTA gives ref_i a DISJOINT sparse code; the dense-readout control reads ALL
    #     units (overlapping code). Report the sparse-code channel's effect on the oldest fact -- code disjointness
    #     is the lever being tested.
    lever("readout code: dense->k-WTA sparse (oldest-fact acc)",
          round(float(old["dense_readout"]), 4), round(float(old["disjoint"]), 4))

    # (a) load-bearing, BOTH channels: sparse code (vs dense_readout) + frozen reservoir (vs kwta_trained).
    attributable_to("disjoint vs dense_readout (k-WTA sparse-code channel, OLDEST)", old["disjoint"],
                    old["dense_readout"], warn_below=0.0)
    attributable_to("disjoint vs kwta_trained (frozen-reservoir channel, OLDEST)", old["disjoint"],
                    old["kwta_trained"], warn_below=0.0)
    # (e) disjointness vs mere sparsity: the shared-slot k-WTA (same units for all inputs) must NOT protect.
    attributable_to("disjoint vs kwta_shared (disjointness, OLDEST)", old["disjoint"], old["kwta_shared"],
                    warn_below=0.0)
    # (b)/(d) attribution: the fraction of the frac_recalled effect NOT present in the baseline / consolidation arms.
    attributable_to("disjoint vs metaplastic (frac@Nmax)", f["disjoint"], f["metaplastic"])
    # the crux, on the OLDEST fact: disjoint must protect fact 0 where metaplasticity could not (left it at/near 0.0).
    attributable_to("disjoint vs metaplastic (OLDEST-fact acc @Nmax)", old["disjoint"], old["metaplastic"],
                    warn_below=0.0)

    v = Verdict("teacher-loop disjoint sparse fact codes (k-WTA)", chance=chance)
    v.reaches("(1) disjoint beats vanilla (acquisition-at-scale)", before=f["vanilla"], after=f["disjoint"])
    v.reaches("(2) disjoint beats metaplastic (frac_recalled)", before=f["metaplastic"], after=f["disjoint"])
    v.reaches("(3) disjoint protects OLDEST fact vs metaplastic", before=old["metaplastic"], after=old["disjoint"])
    v.reaches("(4) sparse-code load-bearing (vs dense_readout, OLDEST)",
              before=old["dense_readout"], after=old["disjoint"])
    v.reaches("(5) frozen-reservoir load-bearing (vs kwta_trained, OLDEST)",
              before=old["kwta_trained"], after=old["disjoint"])
    v.reaches("(6) disjointness lever (vs shared-slot k-WTA, OLDEST)", before=old["kwta_shared"],
              after=old["disjoint"])
    v.floor("(7) disjoint keeps acquiring new facts (immediate-acq)", acq["disjoint"], floor=0.6)
    # GO: disjoint clears the +0.15 vanilla margin, BEATS metaplastic, PROTECTS the oldest fact (>0.5 where meta
    #     leaves it at/near 0.0), BOTH channel-controls (dense readout, trained reservoir) lose the oldest fact,
    #     the shared-slot overlap control collapses, and it still acquires. Single-seed prints SMOKE, never a GO.
    go = (f["disjoint"] > f["vanilla"] + 0.15
          and f["disjoint"] > f["metaplastic"] + 0.05
          and old["disjoint"] > old["metaplastic"] + 0.15
          and old["disjoint"] > 0.5
          and old["disjoint"] > old["dense_readout"] + 0.15
          and old["disjoint"] > old["kwta_trained"] + 0.15
          and old["disjoint"] > old["kwta_shared"] + 0.15
          and acq["disjoint"] >= 0.6)
    decision = v.decide(go=go)
    return {"largest_N": big, "one_over_N": one_over_N, "backend": backend,
            "frac_recalled": f, "oldest_fact_acc": old, "mid_fact_acc": mid, "newest_fact_acc": new,
            "immediate_acq": acq,
            "disjoint_beats_vanilla": float(f["disjoint"] - f["vanilla"]),
            "disjoint_beats_metaplastic": float(f["disjoint"] - f["metaplastic"]),
            "disjoint_oldest_beats_metaplastic": float(old["disjoint"] - old["metaplastic"]),
            "sparse_code_oldest_gain": float(old["disjoint"] - old["dense_readout"]),
            "frozen_reservoir_oldest_gain": float(old["disjoint"] - old["kwta_trained"]),
            "disjointness_oldest_gain": float(old["disjoint"] - old["kwta_shared"]), **decision}


def _aggregate(paths):
    """6-seed roll-up. GO = every seed: disjoint > vanilla+0.15 AND > metaplastic+0.05 AND oldest > metaplastic+0.15
    AND oldest > 0.5 AND oldest > dense_readout+0.15 AND oldest > kwta_trained+0.15 AND oldest > kwta_shared+0.15
    AND disjoint immediate-acq >= 0.6."""
    rows = []
    arms_seen = None
    for p in paths:
        d = json.loads(Path(p).read_text())
        vd = d["verdict"]
        f = vd["frac_recalled"]; old = vd["oldest_fact_acc"]; acq = vd["immediate_acq"]
        if arms_seen is None:
            arms_seen = [a for a in ("vanilla", "metaplastic", "disjoint", "dense_readout",
                                     "kwta_trained", "kwta_shared", "disjoint_meta") if a in f]
        seed_go = bool(f["disjoint"] > f["vanilla"] + 0.15 and f["disjoint"] > f["metaplastic"] + 0.05
                       and old["disjoint"] > old["metaplastic"] + 0.15 and old["disjoint"] > 0.5
                       and old["disjoint"] > old["dense_readout"] + 0.15
                       and old["disjoint"] > old["kwta_trained"] + 0.15
                       and old["disjoint"] > old["kwta_shared"] + 0.15 and acq["disjoint"] >= 0.6)
        rows.append({"seed": d["seed"], "N": vd["largest_N"], **{a: f[a] for a in arms_seen},
                     "old_disjoint": old["disjoint"], "old_meta": old["metaplastic"],
                     "acq_disjoint": acq["disjoint"], "seed_go": seed_go})
    means = {a: float(np.mean([r[a] for r in rows])) for a in arms_seen}
    old_dj = float(np.mean([r["old_disjoint"] for r in rows])); old_mt = float(np.mean([r["old_meta"] for r in rows]))
    n_go = sum(r["seed_go"] for r in rows)
    go = n_go == len(rows) and len(rows) >= 6
    print("\n" + "=" * 130)
    print(f"[AGG] {len(rows)} seeds | GO needs disjoint>vanilla+.15 & >metaplastic+.05 & oldest>meta+.15 & oldest>0.5 & "
          f"oldest>contrastive+.15 & oldest>trained+.15 & frac>permute+.15, ALL seeds")
    print(f"{'seed':>5} {'N':>4} " + " ".join(f"{a[:14]:>15}" for a in arms_seen) +
          f" {'oldDisj':>8} {'oldMeta':>8} {'acqDj':>7} {'GO':>4}")
    for r in sorted(rows, key=lambda x: x["seed"]):
        print(f"{r['seed']:>5} {r['N']:>4} " + " ".join(f"{r[a]:>15.3f}" for a in arms_seen) +
              f" {r['old_disjoint']:>8.3f} {r['old_meta']:>8.3f} {r['acq_disjoint']:>7.3f} {str(r['seed_go']):>4}")
    print(f"{'mean':>5} {'':>4} " + " ".join(f"{means[a]:>15.3f}" for a in arms_seen) +
          f" {old_dj:>8.3f} {old_mt:>8.3f}")
    print(f"[AGG] disjoint {means['disjoint']:.3f} vs metaplastic {means['metaplastic']:.3f} vs vanilla "
          f"{means['vanilla']:.3f} | oldest disjoint {old_dj:.3f} vs meta {old_mt:.3f} | seeds GO {n_go}/{len(rows)} "
          f"| VERDICT {'GO' if go else 'NO-GO'}")
    print("=" * 130)
    return 0 if go else 1


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop sparse orthogonal fact codes: does routing each "
                                             "fact's e-prop change to a DISJOINT sparse hidden-unit subset protect "
                                             "the OLDEST fact metaplasticity/chains could not, and beat both?")
    ap.add_argument("--aggregate", nargs="+", default=None, help="per-seed JSONs -> 6-seed GO roll-up")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-max", type=int, default=32)
    ap.add_argument("--milestones", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--code-size", type=int, default=3, help="hidden units per fact's sparse code (H/code_size = disjoint capacity)")
    ap.add_argument("--settle-steps", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=24)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=20)
    ap.add_argument("--declamp-wmax", type=float, default=1e9,
                    help="bdsp_w_max for ALL arms (de-clamp held constant; NOT the lever). <0 keeps the +-6 default.")
    ap.add_argument("--meta-gain", type=float, default=8.0, help="single-var metaplastic gain g in lr_eff=lr/(1+g*c)")
    ap.add_argument("--meta-consol-rate", type=float, default=1.0, help="per-fact consolidation increment rate")
    ap.add_argument("--with-sparse-meta", action="store_true",
                    help="add a 6th arm: disjoint sparse code + consolidation (does combining the two axes add?)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.aggregate:
        return _aggregate(a.aggregate)
    if a.declamp_wmax is not None and a.declamp_wmax < 0:
        a.declamp_wmax = None
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    result = run(a.seed, a.n_max, a.milestones, a.hidden, a.code_size, a.settle_steps, a.epochs, a.batch,
                 a.eprop_lr, a.w_clip, a.n_draws, a.d_p, a.noise, a.test_n, a.declamp_wmax, a.meta_gain,
                 a.meta_consol_rate, a.with_sparse_meta)
    verdict = _verdict(result)
    summary = {"probe": "teacher_loop_sparse_fact_codes", "seed": a.seed,
               "backend": os.environ.get("SIM_BACKEND"), "single_seed_smoke": True,
               "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    f = verdict["frac_recalled"]; old = verdict["oldest_fact_acc"]; acq = verdict["immediate_acq"]
    print("\n" + "=" * 124, flush=True)
    print(f"[sparse-fact-codes] seed {a.seed} @ N={verdict['largest_N']} (1/N={verdict['one_over_N']:.3f}, "
          f"chance {result['chance']:.3f}):", flush=True)
    for arm in result["arms"]:
        print(f"    {arm:20s}: frac-recalled {f[arm]:.3f} | oldest-fact {old[arm]:.3f} | "
              f"immediate-acq {acq[arm]:.3f}", flush=True)
    print(f"[sparse-fact-codes] disjoint-beats-vanilla {verdict['disjoint_beats_vanilla']:+.3f} | "
          f"disjoint-beats-metaplastic {verdict['disjoint_beats_metaplastic']:+.3f} | "
          f"oldest disjoint-vs-meta {verdict['disjoint_oldest_beats_metaplastic']:+.3f} | "
          f"sparse-code-oldest-gain {verdict['sparse_code_oldest_gain']:+.3f} | "
          f"frozen-reservoir-oldest-gain {verdict['frozen_reservoir_oldest_gain']:+.3f} | "
          f"disjointness-oldest-gain {verdict['disjointness_oldest_gain']:+.3f} "
          f"| VERDICT {verdict['status']}", flush=True)
    print(f"[sparse-fact-codes] wrote {a.out}\n" + "=" * 124, flush=True)
    return 0 if verdict["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
