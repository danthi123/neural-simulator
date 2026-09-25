"""CA3 SUPERPOSED FACT ATTRACTOR -- a DISTRIBUTED fact store whose capacity CAN be measured (default-off research runner).

WHY (owner question 2026-09-23: "does the fact learning we're proving scale to the levels needed to go head to head with
even a tiny LLM?"). The D6 store (`research/runners/d6_hebbian_store.py`) gives every fact its OWN block of synapses,
so recall cannot degrade with the number of facts N and a capacity-curve instrument on it cannot fail: the storage
shared between facts is fixed at ZERO. `research/biology/semantic-store-cortical-capacity.md` states that premise in
writing ("per-fact decode integrity ... is independent of how many other facts are stored"). The real hippocampus
does the opposite: every memory is written into the SAME recurrent CA3 matrix (Marr; Kandel 6e "The CA3 Region Is
Important for Pattern Completion"), so interference is intrinsic and capacity is set by synapses per cell and code
sparseness (Rolls 2013, p_max ~ k C/(a ln(1/a)); Tsodyks & Feigel'man 1988). This runner builds that store and
measures its capacity law, with the catastrophic-interference cliff EXPECTED past capacity.

THE MECHANISM (research/biology/ca3-superposed-fact-attractor.md):
  EC-in    three role sub-populations (agent | relation | patient), n_ec_role cells each; a filler is a sparse
           random code of k_ec cells (the G.20 `generate_sparse_patterns` codes, reused by import).
  DG       FIXED random sparse EC->DG expansion (n_dg = 5 x EC-in); granules fire under gamma-cycle inhibition
           (sparseness a_dg). A fact's DG code is a conjunction of its three role inputs.
  CA3      n_ca3 cells, sparseness a_ca3. Mossy fibres: a FIXED sparse DG->CA3 "detonator" (c_mf = 46 per cell, Rolls
           2013) that selects which CA3 cells fire during ENCODING. The stored CA3 pattern is whatever fires; the host
           neither chooses nor copies it. Recurrent collaterals: a FIXED random diluted topology (c_rec per cell) with
           PLASTIC values. Perforant path EC-in -> CA3: fixed diluted topology (c_pp per cell), plastic values; it
           carries the RECALL cue.
  EC-out   the patient sub-population of deep EC, reached by a plastic CA3 -> EC-out projection (c_out per cell):
           the Teyler-DiScenna index reinstating the cortical pattern.

  WRITE (one presentation, local rule, ALL facts into the SAME matrices): the fact's EC-in pattern drives DG and the
  mossy detonator; the CA3 assembly that fires, the EC-in pattern and the EC-out pattern then change every plastic
  synapse by the COVARIANCE rule with each cell's OWN running mean rate as its sliding threshold. Implemented exactly
  via Welford's online covariance: dw_ij = (x_i - m_i_old)(x_j - m_j_new) summed over writes equals
  H_ij - n_i n_j / P, where H is the co-activity count over the fixed topology and n the per-cell spike counts.
  Both quantities are local to the synapse and its two cells; the decomposition is an arithmetic identity used for
  speed, not a different rule (pinned by tests/test_ca3_superposed_fact_attractor.py).
  BOUNDED arm (the palimpsest companion): the same post-gated increments on integer synapses clipped to [-B, B], LTP
  +1 when pre and post fire together, heterosynaptic LTD -1 with probability = the pre cell's own running rate when
  the post cell fires without it (Amit & Fusi 1994; Rolls 2013 heterosynaptic LTD).

  READ (pattern completion from a PARTIAL cue): agent + relation at EC-in, patient sub-population EMPTY. The
  perforant path gives a partial CA3 pattern; T gamma cycles of recurrent drive + the persisting perforant drive,
  each followed by gamma-cycle inhibition, settle the assembly; CA3 -> EC-out reinstates the patient code.
  Per-query cost is O(T * a n_ca3 * c_rec): no scan over facts, no host lookup by key.

DECLARED ABSTRACTIONS AND HOST SHORTCUTS (named, not hidden):
  (h1) GAMMA-CYCLE BINARY DISCRETIZATION. Each cell emits one spike or none per gamma cycle. Feedback inhibition
       within a cycle is abstracted as k-WTA: the k most-driven cells fire (host `argpartition` over the summed
       synaptic drive). This is the fixed-sparseness idealization of the E%-max rule (de Almeida, Idiart & Lisman
       2009): rank-based, hence reproducible. It is NOT cross-checked against the full Izhikevich bridge here; that
       small-N cross-check is the named next rung.
  (h2) EC-OUT TEACHING RELAY. During encoding the EC-out patient layer carries the same cortical patient code that
       drives EC-in (the heard word; the superficial -> deep EC relay abstracted as an identity mapping). The readout
       synapses learn by the local rule, but this relay's one-to-one topology is designed.
  (h3) THE NAMING STEP IS THE INSTRUMENT. The EC-out spiking pattern is the brain's output. The experimenter scores it
       by overlap against every entity's code (correct iff the true patient's overlap strictly exceeds all others).
       No brain decision depends on it.
  (h4) Which facts are presented, in which order, and the partial cue are the ENVIRONMENT (the teacher's utterances).
  (h5) Fixed topologies (EC->DG, mossy, all diluted fan-ins) are developmental wiring drawn once per network from
       the seed; they are NOT per-fact and not self-organized (self-organized mossy topology = named next rung).
  (h6) Row-centering: each cell's drive subtracts its own mean synaptic weight x its active-input count (a
       subtractive homeostatic normalization local to the cell), applied identically in every arm.

DEFAULT-OFF: a standalone research runner. Nothing in sim/ or webapp/ imports it; no production wiring.

Usage:
  python -m research.runners.ca3_superposed_fact_attractor --arm sparse_dg --seed 42 \
      --out research/findings/raw/_ca3_superposed_fact_attractor/grid
  python -m research.runners.ca3_superposed_fact_attractor --aggregate research/findings/raw/_ca3_superposed_fact_attractor/grid
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import scipy.sparse as sp

from research.runners.concept_pool_sparse_distributed import generate_sparse_patterns

SEEDS = (42, 43, 44, 100, 101, 102)
DEV_SEED = 7                      # development / design-viability seed; never an evaluation seed
P_GRID = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)

# biology-bound defaults (research/biology/ca3-superposed-fact-attractor.md constraints_config)
CA3_SPARSENESS = 0.01             # sparse CA3 code (rat CA3 a ~ 0.02; Rolls 2013); capacity ~ 1/(a ln(1/a))
MOSSY_FANIN = 46                  # mossy-fibre synapses per CA3 cell (Rolls 2013): the sparse strong detonator

N_REL_HUB = 128                   # relation-hub regime: few relations, each shared by ~P/128 stored facts

# Two fact regimes (split BEFORE registration, from the dev seed; see the PREREGISTRATION's dev-seed section).
# UNIFORM: n_rel = n_ent = 2000, so a partial cue (agent, relation) is shared with few other stored facts and
#   capacity is limited by synaptic crosstalk -- the regime that tests the capacity LAW.
# HUB: n_rel = 128 (a knowledge base has few relations, each used by many facts), so every cue half-matches
#   ~P/128 other facts -- structured cue ambiguity, the regime where DG pattern separation has work to do.
ARMS = {
    # ---- UNIFORM regime ----
    "sparse_dg": {},                                                       # DG sparse coding (the default)
    "sparse_dg_c2": dict(c_rec=4000, c_pp=1200, c_out=4000, p_grid=P_GRID + (100000,)),   # every fan-in x2
    # c_rec ALONE doubled (c_pp, c_out held fixed): isolates the recurrent edge so k can be attributed to it.
    # sparse_dg_c2 doubles all three fan-ins at once and cannot do this (2026-09-24 review: "a coincidental
    # match of a confounded normalization" -- the dev rec_zero lesion shows the recurrent edge owns only ~25%
    # of capacity linearly, so a k fit off sparse_dg_c2 is not a recurrent-synapse measurement).
    "sparse_dg_recx2": dict(c_rec=4000),
    "dense_nodg": dict(use_dg=False, a_ca3=0.05),                          # no DG, dense code (baseline)
    # ---- HUB regime ----
    "sparse_dg_hub": dict(n_rel=N_REL_HUB),
    "dense_nodg_hub": dict(n_rel=N_REL_HUB, use_dg=False, a_ca3=0.05),     # the no-companion baseline
    "sparse_nodg_hub": dict(n_rel=N_REL_HUB, use_dg=False),                # dissociation: sparse without DG
    "sparse_dg_c2_hub": dict(n_rel=N_REL_HUB, c_rec=4000, c_pp=1200, c_out=4000),
    "sparse_dg_bounded_hub": dict(n_rel=N_REL_HUB, plasticity="bounded"),  # palimpsest companion
}
UNBOUNDED_ARMS = tuple(a for a in ARMS if "bounded" not in a)


@dataclass
class Cfg:
    seed: int
    arm: str = "sparse_dg"
    n_ec_role: int = 1000
    k_ec: int = 20
    n_ent: int = 2000
    n_rel: int = 2000
    n_ca3: int = 10000
    a_ca3: float = CA3_SPARSENESS
    c_rec: int = 2000
    c_pp: int = 600
    c_out: int = 2000
    use_dg: bool = True
    n_dg: int = 15000
    a_dg: float = 0.005
    c_ecdg: int = 200
    c_mf: int = MOSSY_FANIN
    c_fix: int = 300
    plasticity: str = "covariance"
    bound: int = 1
    T: int = 8
    n_query: int = 200
    n_recent: int = 100
    n_novel: int = 100
    n_xtalk: int = 50
    p_grid: tuple = P_GRID
    encode_batch: int = 500

    @property
    def k_ca3(self):
        return int(round(self.a_ca3 * self.n_ca3))

    @property
    def k_dg(self):
        return int(round(self.a_dg * self.n_dg))

    @property
    def n_ec(self):
        return 3 * self.n_ec_role


def make_cfg(arm: str, seed: int, **over) -> Cfg:
    if arm not in ARMS:
        raise SystemExit("unknown arm %r (have %s)" % (arm, sorted(ARMS)))
    kw = dict(ARMS[arm])
    kw.update(over)
    return Cfg(seed=int(seed), arm=arm, **kw)


# --------------------------------------------------------------------------------------------------------------
# substrate pieces
# --------------------------------------------------------------------------------------------------------------

def random_fanin(n_post: int, n_pre: int, c: int, rng: np.random.Generator, no_self: bool = False) -> np.ndarray:
    """Fixed random fan-in topology: row i lists the c distinct presynaptic cells of postsynaptic cell i."""
    if c > n_pre - (1 if no_self else 0):
        raise ValueError("fan-in %d exceeds presynaptic pool %d" % (c, n_pre))
    idx = np.empty((n_post, c), dtype=np.int32)
    for i in range(n_post):
        if no_self:
            r = rng.choice(n_pre - 1, size=c, replace=False)
            r = r + (r >= i)
        else:
            r = rng.choice(n_pre, size=c, replace=False)
        idx[i] = r
    return idx


def kwta(h: np.ndarray, k: int) -> np.ndarray:
    """Gamma-cycle inhibition (declared abstraction h1): the k most-driven cells of each column fire."""
    n, q = h.shape
    win = np.argpartition(-h, k - 1, axis=0)[:k]
    out = np.zeros((n, q), dtype=bool)
    out[win, np.arange(q)[None, :]] = True
    return out


class Projection:
    """A diluted projection with FIXED topology and (optionally) PLASTIC values shared by every stored fact.

    mode 'fixed'       developmental weights ~ U(0.5, 1.5), never change (EC->DG, mossy, EC->CA3 encoding drive)
    mode 'covariance'  Welford online covariance with each cell's running mean rate (exact H - n n^T / P form)
    mode 'bounded'     post-gated integer synapses in [-B, B]: LTP +1 (pre & post), heterosynaptic LTD -1 with
                       probability = the pre cell's running rate (post fires, pre silent)
    """

    def __init__(self, n_post, n_pre, c, rng, mode, bound=1, no_self=False):
        self.n_post, self.n_pre, self.c, self.mode, self.bound = n_post, n_pre, c, mode, int(bound)
        self.idx = random_fanin(n_post, n_pre, c, rng, no_self=no_self)
        self.P = 0
        self.n_post_spk = np.zeros(n_post, dtype=np.int64)
        self.n_pre_spk = np.zeros(n_pre, dtype=np.int64)
        if mode == "fixed":
            self.W = rng.uniform(0.5, 1.5, size=(n_post, c)).astype(np.float32)
        elif mode == "covariance":
            self.H = np.zeros((n_post, c), dtype=np.float32)
        elif mode == "bounded":
            self.W = np.zeros((n_post, c), dtype=np.int8)
        else:
            raise ValueError(mode)
        self._csr_cache = None

    # ---- write (one fact) ----
    def write(self, post: np.ndarray, pre: np.ndarray, rng: np.random.Generator | None = None):
        a = np.flatnonzero(post)
        if self.mode == "covariance":
            if a.size:
                self.H[a] += pre[self.idx[a]]
        elif self.mode == "bounded":
            if a.size:
                pj = pre[self.idx[a]]
                rate = self.n_pre_spk[self.idx[a]] / max(self.P, 1)
                ltd = (~pj) & (rng.random(pj.shape) < rate)
                w = self.W[a].astype(np.int16) + pj.astype(np.int16) - ltd.astype(np.int16)
                self.W[a] = np.clip(w, -self.bound, self.bound).astype(np.int8)
        else:
            raise RuntimeError("fixed projection is not plastic")
        self.n_post_spk += post
        self.n_pre_spk += pre
        self.P += 1
        self._csr_cache = None

    # ---- materialize the synaptic values the read uses ----
    def values(self) -> np.ndarray:
        if self.mode == "fixed":
            return self.W
        if self.mode == "covariance":
            if self.P == 0:
                return np.zeros_like(self.H)
            W = self.H - (self.n_post_spk[:, None].astype(np.float32)
                          * self.n_pre_spk[self.idx].astype(np.float32)) / np.float32(self.P)
            return W
        return self.W.astype(np.float32)

    def matrix(self, lesion: str = "intact", rng: np.random.Generator | None = None) -> sp.csc_matrix:
        """The synapses as an (n_post x n_pre) matrix stored COLUMN-compressed, i.e. grouped by presynaptic cell
        (the axonal view): a product with a sparse activity vector touches only the active cells' out-synapses,
        so a read costs O(active x fan-out), not O(all synapses)."""
        if lesion == "intact" and self._csr_cache is not None:
            return self._csr_cache
        W = self.values()
        if self.mode != "fixed":
            W = W - W.mean(axis=1, keepdims=True)          # h6: subtractive homeostatic normalization
        if lesion == "zero":
            W = np.zeros_like(W)
        elif lesion == "shuffle":
            # re-attach each row's values to OTHER presynaptic cells (row marginal preserved exactly)
            off = rng.integers(1, self.c, size=self.n_post)
            cols = (np.arange(self.c)[None, :] + off[:, None]) % self.c
            W = np.take_along_axis(W, cols, axis=1)
        elif lesion != "intact":
            raise ValueError(lesion)
        indptr = np.arange(0, self.n_post * self.c + 1, self.c, dtype=np.int64)
        m = sp.csr_matrix((W.ravel().astype(np.float32), self.idx.ravel(), indptr),
                          shape=(self.n_post, self.n_pre)).tocsc()
        if lesion == "intact":
            self._csr_cache = m
        return m

    def nbytes(self) -> int:
        n = self.idx.nbytes + self.n_post_spk.nbytes + self.n_pre_spk.nbytes
        n += self.H.nbytes if self.mode == "covariance" else self.W.nbytes
        return int(n)


def _col_sparse(x: np.ndarray) -> sp.csc_matrix:
    return sp.csc_matrix(x.astype(np.float32))


class Network:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        base = np.random.SeedSequence([cfg.seed, 0xCA3])
        s_topo, s_facts, s_plast, s_eval = base.spawn(4)
        self.rng_topo = np.random.default_rng(s_topo)
        self.rng_facts = np.random.default_rng(s_facts)
        self.rng_plast = np.random.default_rng(s_plast)
        self.rng_eval = np.random.default_rng(s_eval)
        r = self.rng_topo
        mode = cfg.plasticity
        if cfg.use_dg:
            self.ec_dg = Projection(cfg.n_dg, cfg.n_ec, cfg.c_ecdg, r, "fixed")
            self.mossy = Projection(cfg.n_ca3, cfg.n_dg, cfg.c_mf, r, "fixed")
            self.ec_ca3_fix = None
        else:
            self.ec_dg = self.mossy = None
            self.ec_ca3_fix = Projection(cfg.n_ca3, cfg.n_ec, cfg.c_fix, r, "fixed")
        self.rec = Projection(cfg.n_ca3, cfg.n_ca3, cfg.c_rec, r, mode, cfg.bound, no_self=True)
        self.pp = Projection(cfg.n_ca3, cfg.n_ec, cfg.c_pp, r, mode, cfg.bound)
        self.out = Projection(cfg.n_ec_role, cfg.n_ca3, cfg.c_out, r, mode, cfg.bound)
        # filler codes: G.20 sparse-distributed codes (reused by import), seeded from cfg.seed
        self.ent_codes = [np.asarray(p) for p in generate_sparse_patterns(cfg.n_ent, cfg.n_ec_role, cfg.k_ec,
                                                                          seed=cfg.seed)]
        self.rel_codes = [np.asarray(p) for p in generate_sparse_patterns(cfg.n_rel, cfg.n_ec_role, cfg.k_ec,
                                                                          seed=cfg.seed + 7919)]
        E = np.zeros((cfg.n_ent, cfg.n_ec_role), dtype=np.float32)
        for i, p in enumerate(self.ent_codes):
            E[i, p] = 1.0
        self.ent_matrix = E

    # ---- EC patterns ----
    def ec_in(self, facts: np.ndarray, with_patient: bool) -> np.ndarray:
        """facts (Q,3) -> EC-in activity (n_ec, Q) bool."""
        cfg = self.cfg
        X = np.zeros((cfg.n_ec, len(facts)), dtype=bool)
        for q, (a, r, p) in enumerate(facts):
            X[self.ent_codes[a], q] = True
            X[cfg.n_ec_role + self.rel_codes[r], q] = True
            if with_patient:
                X[2 * cfg.n_ec_role + self.ent_codes[p], q] = True
        return X

    def ec_out_patient(self, facts: np.ndarray) -> np.ndarray:
        Y = np.zeros((self.cfg.n_ec_role, len(facts)), dtype=bool)
        for q, f in enumerate(facts):
            Y[self.ent_codes[f[2]], q] = True
        return Y

    # ---- encoding: which CA3 cells fire for a fact (fixed detonator; no plasticity involved) ----
    def encode_ca3(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        cfg = self.cfg
        Xs = _col_sparse(X)
        if cfg.use_dg:
            dg = kwta((self.ec_dg.matrix() @ Xs).toarray(), cfg.k_dg)
            ca3 = kwta((self.mossy.matrix() @ _col_sparse(dg)).toarray(), cfg.k_ca3)
            return ca3, dg
        ca3 = kwta((self.ec_ca3_fix.matrix() @ Xs).toarray(), cfg.k_ca3)
        return ca3, None

    def write_fact(self, x_ca3: np.ndarray, x_ec: np.ndarray, y_out: np.ndarray):
        self.rec.write(x_ca3, x_ca3, self.rng_plast)
        self.pp.write(x_ca3, x_ec, self.rng_plast)
        self.out.write(y_out, x_ca3, self.rng_plast)

    def synapse_bytes(self) -> int:
        return self.rec.nbytes() + self.pp.nbytes() + self.out.nbytes()

    # ---- recall: pattern completion from agent + relation ----
    def recall(self, cues: np.ndarray, lesion: str = "intact", rng=None, clamp: np.ndarray | None = None):
        cfg = self.cfg
        k = cfg.k_ca3
        rec_lesion = {"intact": "intact", "rec_zero": "zero", "rec_shuffle": "shuffle"}[lesion]
        Wpp = self.pp.matrix()
        Wrec = self.rec.matrix(rec_lesion, rng)
        Wout = self.out.matrix()
        h_pp = (Wpp @ _col_sparse(self.ec_in(cues, with_patient=False))).toarray()
        s = kwta(h_pp, k)
        s0 = s.copy()
        conv = None
        for _ in range(cfg.T):
            h = (Wrec @ _col_sparse(s)).toarray() + h_pp
            s_new = kwta(h, k)
            conv = (s_new & s).sum(axis=0) / k
            s = s_new
        y = kwta((Wout @ _col_sparse(s)).toarray(), cfg.k_ec)
        return s0, s, y, conv

    def name(self, y: np.ndarray, true_patient: np.ndarray) -> np.ndarray:
        """INSTRUMENT (h3): correct iff the true patient's code overlap strictly exceeds every other entity's."""
        ov = self.ent_matrix @ y.astype(np.float32)                       # (n_ent, Q)
        q = np.arange(y.shape[1])
        true_ov = ov[true_patient, q].copy()
        ov[true_patient, q] = -1.0
        return true_ov > ov.max(axis=0)


# --------------------------------------------------------------------------------------------------------------
# the experiment
# --------------------------------------------------------------------------------------------------------------

def draw_facts(net: Network, p_max: int, n_novel: int):
    cfg = net.cfg
    r = net.rng_facts
    n_keys = cfg.n_ent * cfg.n_rel
    if p_max + n_novel > n_keys:
        raise ValueError("not enough distinct (agent, relation) keys")
    keys = r.choice(n_keys, size=p_max + n_novel, replace=False)
    agents, rels = keys // cfg.n_rel, keys % cfg.n_rel
    patients = r.integers(0, cfg.n_ent, size=p_max + n_novel)
    F = np.stack([agents, rels, patients], axis=1).astype(np.int64)
    return F[:p_max], F[p_max:]


def _log_interp_cross(ps, ys, level):
    """P at which recall first falls below `level`, log-interpolated; None if it never does (censored)."""
    for i in range(1, len(ps)):
        if ys[i - 1] >= level > ys[i]:
            x0, x1 = math.log(ps[i - 1]), math.log(ps[i])
            t = (ys[i - 1] - level) / (ys[i - 1] - ys[i])
            return float(math.exp(x0 + t * (x1 - x0)))
    if ys and ys[0] < level:
        return float(ps[0])
    return None


def crosstalk_dprime(net: Network, facts_x: np.ndarray) -> float:
    """d' of recurrent drive for in-pattern vs out-of-pattern cells when a stored CA3 pattern is clamped."""
    h = (net.rec.matrix() @ _col_sparse(facts_x)).toarray()
    ds = []
    for q in range(facts_x.shape[1]):
        m = facts_x[:, q]
        out = h[~m, q]
        sd = out.std()
        if sd > 0:
            ds.append((h[m, q].mean() - out.mean()) / sd)
    return float(np.median(ds)) if ds else float("nan")


def run(cfg: Cfg, out_path: str | None, p_max_override: int | None = None, log=print) -> dict:
    t_build = time.time()
    net = Network(cfg)
    grid = [p for p in cfg.p_grid if p_max_override is None or p <= p_max_override]
    p_max = grid[-1]
    facts, novel = draw_facts(net, p_max, cfg.n_novel)
    build_s = time.time() - t_build
    log("[%s s%d] built in %.1fs; synapse state %.1f MB; grid %s"
        % (cfg.arm, cfg.seed, build_s, net.synapse_bytes() / 1e6, grid))

    # the stored CA3 pattern of each fact is whatever the fixed detonator fires (kept only for the INSTRUMENT:
    # completion overlap and the crosstalk d'; the read never consults it)
    ca3_idx = np.zeros((p_max, cfg.k_ca3), dtype=np.int32)      # stored pattern of each fact, as cell indices
    # `complete` (2026-09-25, research/FAILURE_LOG.md): each checkpoint's `_dump()` below writes the file to disk
    # PROGRESSIVELY, so a structurally-valid, ostensibly-complete-looking JSON exists on disk at every P in the
    # grid, not just the last one -- a scorer read `rerun_s101_s102/sparse_dg_c2_s101.json` while its job was
    # still running and mistook that in-progress checkpoint for the finished run (the missing P=100000 point was
    # not obviously missing from the file's shape). `complete` stays False until the FINAL `_dump()` call after
    # the loop; a reader/aggregator should refuse a file where this is not `true`.
    rec = dict(config=asdict(cfg), backend="numpy+scipy.sparse (CPU)", runner="research.runners.ca3_superposed_fact_attractor",
               build_s=build_s, checkpoints=[], sha_facts=hashlib.sha256(facts.tobytes()).hexdigest(), complete=False)
    written = 0
    write_s = 0.0
    freeze_done = False
    for P in grid:
        t0 = time.time()
        while written < P:
            b = min(cfg.encode_batch, P - written)
            F = facts[written:written + b]
            X = net.ec_in(F, with_patient=True)
            Y = net.ec_out_patient(F)
            ca3, _ = net.encode_ca3(X)
            for q in range(b):
                net.write_fact(ca3[:, q], X[:, q], Y[:, q])
            ca3_idx[written:written + b] = np.sort(np.argpartition(~ca3, cfg.k_ca3 - 1, axis=0)[:cfg.k_ca3].T, axis=1)
            written += b
        dt_w = time.time() - t0
        write_s += dt_w

        ev = net.rng_eval
        q_idx = np.sort(ev.choice(P, size=min(cfg.n_query, P), replace=False))
        r_idx = np.arange(max(0, P - cfg.n_recent), P)
        cp = dict(P=P, write_s_interval=dt_w, write_ms_per_fact=1000.0 * write_s / P,
                  synapse_bytes=net.synapse_bytes())
        # materialize once (the live synapses; not a per-query cost)
        t_m = time.time()
        net.rec.matrix(); net.pp.matrix(); net.out.matrix()
        cp["materialize_s"] = time.time() - t_m
        for lesion in ("intact", "rec_zero", "rec_shuffle"):
            t_q = time.time()
            s0, s, y, conv = net.recall(facts[q_idx], lesion=lesion, rng=ev)
            dt_q = time.time() - t_q
            ok = net.name(y, facts[q_idx, 2])
            tgt = _pat_matrix(ca3_idx, q_idx, cfg.n_ca3)
            d = dict(recall=float(ok.mean()), n=int(len(q_idx)),
                     ca3_overlap_init=float(((s0 & tgt).sum(0) / cfg.k_ca3).mean()),
                     ca3_overlap_final=float(((s & tgt).sum(0) / cfg.k_ca3).mean()),
                     settle_final=float(conv.mean()), per_query_ms_batched=1000.0 * dt_q / len(q_idx))
            if lesion == "intact":
                _, _, y_r, _ = net.recall(facts[r_idx], lesion="intact")
                d["recall_recent"] = float(net.name(y_r, facts[r_idx, 2]).mean())
                _, _, y_n, conv_n = net.recall(novel, lesion="intact")
                d["novel_settle_final"] = float(conv_n.mean())
                d["stored_settle_final"] = float(conv.mean())
                # single-query latency (the per-turn cost a chat read would pay)
                lat = []
                for j in q_idx[:5]:
                    t1 = time.time()
                    net.recall(facts[j:j + 1], lesion="intact")
                    lat.append(time.time() - t1)
                d["per_query_ms_single"] = 1000.0 * float(np.median(lat))
            cp[lesion] = d
        xs = _pat_matrix(ca3_idx, np.sort(ev.choice(P, size=min(cfg.n_xtalk, P), replace=False)), cfg.n_ca3)
        cp["crosstalk_dprime"] = crosstalk_dprime(net, xs)
        if cfg.plasticity == "covariance":
            H = net.rec.H
            touched = H >= 1
            cp["rec_synapses_touched_frac"] = float(touched.mean())
            cp["rec_touched_shared_by_ge2_facts_frac"] = float((H >= 2).sum() / max(touched.sum(), 1))
        cp["ca3_pattern_mean_pairwise_overlap"] = float(_mean_pairwise_overlap(ca3_idx[:P], ev, cfg))
        if not freeze_done:
            cp["freeze_all_recall"] = float(_freeze_all_recall(cfg, facts[q_idx]))
            freeze_done = True
        rec["checkpoints"].append(cp)
        log("[%s s%d] P=%6d recall %.3f (rec0 %.3f, shuf %.3f) recent %.3f  ca3ov %.2f->%.2f  d'=%.2f  "
            "q=%.1fms  write %.2fms/fact"
            % (cfg.arm, cfg.seed, P, cp["intact"]["recall"], cp["rec_zero"]["recall"], cp["rec_shuffle"]["recall"],
               cp["intact"]["recall_recent"], cp["intact"]["ca3_overlap_init"], cp["intact"]["ca3_overlap_final"],
               cp["crosstalk_dprime"], cp["intact"]["per_query_ms_single"], cp["write_ms_per_fact"]))
        rec["summary"] = summarize(rec)
        if out_path:
            _dump(rec, out_path)
    rec["complete"] = True             # every P in `grid` finished -- see the `complete=False` comment above
    if out_path:
        _dump(rec, out_path)
    return rec


def _pat_matrix(ca3_idx: np.ndarray, cols, n: int) -> np.ndarray:
    cols = np.asarray(cols)
    X = np.zeros((n, len(cols)), dtype=bool)
    X[ca3_idx[cols].T, np.arange(len(cols))[None, :]] = True
    return X


def _mean_pairwise_overlap(ca3_idx, rng, cfg, n_pairs=2000):
    P = ca3_idx.shape[0]
    if P < 2:
        return float("nan")
    i = rng.integers(0, P, n_pairs)
    j = rng.integers(0, P, n_pairs)
    m = i != j
    A = _pat_matrix(ca3_idx, i[m], cfg.n_ca3)
    B = _pat_matrix(ca3_idx, j[m], cfg.n_ca3)
    return float(((A & B).sum(0) / cfg.k_ca3).mean())


def _freeze_all_recall(cfg: Cfg, cues: np.ndarray) -> float:
    """INTEGRITY SMOKE (not evidence): a network whose plastic synapses never changed must name at chance."""
    net = Network(cfg)
    _, _, y, _ = net.recall(cues)
    return float(net.name(y, cues[:, 2]).mean())


def summarize(rec: dict) -> dict:
    cps = rec["checkpoints"]
    ps = [c["P"] for c in cps]
    ys = [c["intact"]["recall"] for c in cps]
    yr = [c["intact"]["recall_recent"] for c in cps]
    out = dict(P=ps, recall=ys, recall_recent=yr,
               recall_rec_zero=[c["rec_zero"]["recall"] for c in cps],
               recall_rec_shuffle=[c["rec_shuffle"]["recall"] for c in cps],
               dprime=[c["crosstalk_dprime"] for c in cps],
               per_query_ms_single=[c["intact"]["per_query_ms_single"] for c in cps],
               synapse_bytes=[c["synapse_bytes"] for c in cps],
               P50=_log_interp_cross(ps, ys, 0.5), P90=_log_interp_cross(ps, ys, 0.9))
    return out


def _dump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------------------------------
# pre-registered gates (docs: research/findings/2026-09-23-ca3-superposed-fact-attractor-capacity-PREREGISTRATION.md)
# --------------------------------------------------------------------------------------------------------------

G1_LEVEL = 0.9
G2_CEILING = 0.2
G3_RATIO = 1.5
G4_BAND = (1.4, 3.0)
G5_RATIO = 1.2
G7_RECENT_BOUNDED = 0.5
G7_RECENT_UNBOUNDED = 0.2
G8_SLOPE_BAND = (-0.8, -0.2)
MIN_SEEDS = 5        # an effect gate passes on >= 5 of 6 seeds; an UNDEFINED seed counts as a FAIL


def _at(summary, P, key="recall"):
    try:
        return summary[key][summary["P"].index(P)]
    except (ValueError, KeyError):
        return None


def _dprime_slope(s):
    """log-log slope of d' vs P over checkpoints with finite, positive d' (theory: -0.5)."""
    pts = [(math.log(p), math.log(d)) for p, d in zip(s["P"], s["dprime"])
           if d is not None and np.isfinite(d) and d > 0 and p >= 200]
    if len(pts) < 3:
        return None
    x, y = np.array(pts).T
    return float(np.polyfit(x, y, 1)[0])


def _p50_ratio(S, num, den):
    if num not in S or den not in S:
        return None
    a, b = S[num]["P50"], S[den]["P50"]
    return None if (a is None or b is None) else a / b


def gates_for_seed(S: dict) -> dict:
    """S: arm -> summary for ONE seed. Returns gate -> (pass: bool|None, value). None = UNDEFINED (a FAIL)."""
    g = {}
    sd = S.get("sparse_dg")
    # G1 learns + completes below capacity (uniform regime)
    if sd:
        v = [_at(sd, 50), _at(sd, 200)]
        g["G1_learns"] = (None if None in v else all(x >= G1_LEVEL for x in v), v)
    # G2 the cliff is visible for every UNBOUNDED arm (instrument validity: otherwise VOID)
    vals = {arm: S[arm]["recall"][-1] for arm in UNBOUNDED_ARMS if arm in S}
    g["G2_cliff"] = (None if len(vals) < len(UNBOUNDED_ARMS) else all(v <= G2_CEILING for v in vals.values()),
                     vals)
    # G3 the companion (DG sparse coding) pushes capacity up where facts are correlated (hub regime)
    r = _p50_ratio(S, "sparse_dg_hub", "dense_nodg_hub")
    g["G3_companion_capacity"] = (None if r is None else r >= G3_RATIO, r)
    # G4 capacity law (uniform regime): doubling synapses per cell roughly doubles capacity
    r = _p50_ratio(S, "sparse_dg_c2", "sparse_dg")
    g["G4_capacity_law"] = (None if r is None else G4_BAND[0] <= r <= G4_BAND[1], r)
    # G5 the recurrent completion is load-bearing FOR CAPACITY: removing (or scrambling) only the recurrent edge
    # must pull the capacity down; otherwise the store is a perforant -> readout heteroassociator and the
    # attractor is decorative
    if sd:
        p_i = sd["P50"]
        p_z = _log_interp_cross(sd["P"], sd["recall_rec_zero"], 0.5)
        p_s = _log_interp_cross(sd["P"], sd["recall_rec_shuffle"], 0.5)
        ok = None if None in (p_i, p_z, p_s) else (p_z <= p_i / G5_RATIO and p_s <= p_i / G5_RATIO)
        attributable_linear_frac = None
        if p_i is not None and p_z is not None:
            from tools.lab import attributable_to       # local import: aggregation runs on the main box
            # LINEAR P50, not log(P50): log has an arbitrary zero (one fact), so a fraction taken over
            # log(P50) is not a meaningful quantity (2026-09-24 review). attributable_to(t, c) = (t-c)/t,
            # i.e. exactly the linear fraction of P50 lost when the recurrent edge alone is removed.
            attributable_linear_frac = attributable_to("recurrent edge -> P50 (linear)", p_i, p_z)
        g["G5_recurrent_loadbearing"] = (ok, dict(P50_intact=p_i, P50_rec_zero=p_z, P50_rec_shuffle=p_s,
                                                  attributable_linear_frac=attributable_linear_frac))
    # G6 cost law (INTEGRITY SMOKE, not evidence): per-query time flat in P, synapse memory constant in P
    if sd:
        t = sd["per_query_ms_single"]
        flat = t[-1] / t[0] if t and t[0] > 0 else None
        const = len(set(sd["synapse_bytes"])) == 1
        g["G6_cost_flat_INTEGRITY"] = (None if flat is None else (flat <= 2.0 and const),
                                       dict(ratio=flat, mem_const=const))
    # G7 palimpsest (hub regime): bounded synapses keep RECENT facts past capacity where the unbounded store
    # has collapsed
    if "sparse_dg_hub" in S and "sparse_dg_bounded_hub" in S:
        b, u = S["sparse_dg_bounded_hub"]["recall_recent"][-1], S["sparse_dg_hub"]["recall_recent"][-1]
        g["G7_palimpsest"] = (b >= G7_RECENT_BOUNDED and u <= G7_RECENT_UNBOUNDED, dict(bounded=b, unbounded=u))
    # G8 crosstalk grows with P (storage is SHARED): d' falls ~ P^-1/2
    if sd:
        sl = _dprime_slope(sd)
        g["G8_shared_crosstalk"] = (None if sl is None else G8_SLOPE_BAND[0] <= sl <= G8_SLOPE_BAND[1], sl)
    # G9 in the hub regime capacity is limited by cue ambiguity, which more synapses per cell do NOT cure
    r = _p50_ratio(S, "sparse_dg_c2_hub", "sparse_dg_hub")
    g["G9_hub_limit_not_synaptic"] = (None if r is None else r < G4_BAND[0], r)
    return g


def aggregate(grid_dir: str, seeds=SEEDS) -> dict:
    per_seed = {}
    missing = []
    for s in seeds:
        S = {}
        for arm in ARMS:
            p = os.path.join(grid_dir, "%s_s%d.json" % (arm, s))
            if not os.path.exists(p):
                missing.append(p)
                continue
            with open(p) as f:
                S[arm] = json.load(f)["summary"]
        per_seed[s] = S
    gate_rows = {s: gates_for_seed(S) for s, S in per_seed.items()}
    names = sorted({k for r in gate_rows.values() for k in r})
    verdict = {}
    for n in names:
        res = [gate_rows[s].get(n, (None, None))[0] for s in seeds]
        n_pass = sum(1 for x in res if x is True)
        n_undef = sum(1 for x in res if x is None)
        need = len(seeds) if n.endswith("INTEGRITY") else MIN_SEEDS
        verdict[n] = dict(n_pass=n_pass, n_undefined=n_undef, need=need, passed=n_pass >= need,
                          per_seed={str(s): gate_rows[s].get(n, (None, None)) for s in seeds})
    # capacity-law fit, ATTRIBUTED TO THE RECURRENT EDGE ALONE. 2026-09-24 SECOND review (fix-required): fitting
    # k per arm as k = P50 * a * ln(1/a) / c_rec and taking the MEDIAN OVER (sparse_dg, sparse_dg_recx2) does NOT
    # attribute capacity to the recurrent synapses -- the sparse_dg half of that median is the identical
    # all-fan-in-confounded quantity ("P50 driven by all fan-ins, divided by c_rec") the FIRST review rejected
    # off sparse_dg_c2. sparse_dg_recx2 was computed but never used as a MARGINAL (a difference between the two
    # arms); the median just diluted one confounded number with a second, less-confounded one and still labelled
    # the result "ATTRIBUTED TO THE RECURRENT EDGE ALONE".
    #
    # The fix: fit the MARGINAL capacity per recurrent synapse from the CONTRAST between the two arms (c_pp and
    # c_out held fixed; only c_rec differs), PER SEED:
    #   k_rec = (P50_hi - P50_lo) * a * ln(1/a) / (c_rec_hi - c_rec_lo)
    # k_fit is the median of this per-seed marginal -- never a per-arm value, never a median taken across arms.
    # A seed whose marginal is <= 0 (doubling c_rec did not raise P50, e.g. because a bottleneck elsewhere caps
    # both arms) is kept AS MEASURED, never clipped to zero and never dropped: that is a real, falsifying outcome
    # of the capacity-law prediction, not noise to be smoothed away (docs/BUILD_LANE_CHECKLIST.md "UNDEFINED is
    # never a pass" -- a negative marginal is DEFINED and failing, not undefined).
    #
    # The per-arm all-fan-in k (P50 * a ln(1/a) / c_rec for sparse_dg or sparse_dg_recx2 ALONE) is still reported
    # per seed, but ONLY as a descriptive quantity: it is never fed into k_fit and never compared to Rolls' range.
    K_FIT_LO_ARM, K_FIT_HI_ARM = "sparse_dg", "sparse_dg_recx2"
    K_FIT_ARM_PAIR = (K_FIT_LO_ARM, K_FIT_HI_ARM)
    k_marginal_per_seed = {}
    k_allfanin_per_arm_DESCRIPTIVE_ONLY = []
    for s, S in per_seed.items():
        lo, hi = S.get(K_FIT_LO_ARM), S.get(K_FIT_HI_ARM)
        c_lo, c_hi = make_cfg(K_FIT_LO_ARM, s), make_cfg(K_FIT_HI_ARM, s)
        for arm, summ, c in ((K_FIT_LO_ARM, lo, c_lo), (K_FIT_HI_ARM, hi, c_hi)):
            if summ and summ["P50"]:
                k_allfanin_per_arm_DESCRIPTIVE_ONLY.append(
                    dict(seed=s, arm=arm, k_allfanin=summ["P50"] * c.a_ca3 * math.log(1 / c.a_ca3) / c.c_rec))
        if lo and hi and lo["P50"] and hi["P50"]:
            # the marginal fit assumes a_ca3 (and therefore a ln(1/a)) is unchanged between the two arms; only
            # c_rec differs by construction of sparse_dg_recx2 -- this would catch an ARMS-table edit that broke
            # that assumption silently.
            assert c_lo.a_ca3 == c_hi.a_ca3, "K_FIT_ARM_PAIR must vary c_rec alone (a_ca3 differs between arms)"
            d_c_rec = c_hi.c_rec - c_lo.c_rec
            k_marginal_per_seed[str(s)] = ((hi["P50"] - lo["P50"]) * c_lo.a_ca3 * math.log(1 / c_lo.a_ca3)
                                            / d_c_rec)
    ks = list(k_marginal_per_seed.values())
    k_fit = float(np.median(ks)) if ks else None
    extrap = None
    if k_fit is not None and k_fit > 0:
        a, C = 0.005, 10000
        pmax = k_fit * C / (a * math.log(1 / a))
        extrap = dict(note="EXTRAPOLATION from the fitted MARGINAL k (recurrent-edge marginal capacity law, fit "
                           "from the per-seed CONTRAST between %s), not a measurement; the a-scaling from 0.01 "
                           "to 0.005 is itself UNTESTED -- no arm varies a with the DG held fixed in the uniform "
                           "regime" % (K_FIT_ARM_PAIR,),
                      n_ca3=100000, c_rec=C, a=a, predicted_P50_facts=pmax, recurrent_synapses=100000 * C,
                      bits_per_recurrent_synapse=pmax * math.log2(2000) / (100000 * C))
    elif k_fit is not None:
        extrap = dict(note="k_fit <= 0: the median per-seed MARGINAL says doubling the recurrent edge alone did "
                           "NOT raise P50. No extrapolation is defined. This is the realistic failing outcome "
                           "the prereg's capacity-law prediction names, not an error.", k_fit=k_fit)
    out = dict(grid_dir=grid_dir, seeds=list(seeds), missing=missing, verdict=verdict, k_fit=k_fit,
               k_fit_arm_pair=K_FIT_ARM_PAIR, k_marginal_per_seed=k_marginal_per_seed,
               k_allfanin_per_arm_DESCRIPTIVE_ONLY=k_allfanin_per_arm_DESCRIPTIVE_ONLY,
               gpu_point_extrapolation=extrap)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", default="sparse_dg", choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="directory; writes <arm>_s<seed>.json")
    ap.add_argument("--p-max", type=int, default=None, help="truncate the P grid (smoke)")
    ap.add_argument("--aggregate", default=None, help="grid directory to aggregate into gate verdicts")
    ap.add_argument("--allow-dev-seed", action="store_true")
    a = ap.parse_args(argv)
    if a.aggregate:
        res = aggregate(a.aggregate)
        _dump(res, os.path.join(a.aggregate, "aggregate.json"))
        for n, v in res["verdict"].items():
            print("%-32s %s  %d/%d pass (%d undefined)" % (n, "PASS" if v["passed"] else "FAIL", v["n_pass"],
                                                         len(res["seeds"]), v["n_undefined"]))
        print("k_fit", res["k_fit"], "fit_from", res.get("k_fit_arm_pair"), "missing", len(res["missing"]))
        g9, g4 = res["verdict"].get("G9_hub_limit_not_synaptic"), res["verdict"].get("G4_capacity_law")
        if g9 and g9["passed"] and not (g4 and g4["passed"]):
            print("NOTE: G9 passed but G4 did not on the same seeds -- G9 is an absence-of-effect gate that a "
                  "broken sparse_dg_c2_hub arm would also pass; do not headline G9 without G4 (2026-09-24 review).")
        return 0
    if a.seed is None:
        ap.error("--seed is required")
    if a.seed not in SEEDS and not (a.allow_dev_seed and a.seed == DEV_SEED):
        ap.error("seed %d is neither an evaluation seed %s nor the declared dev seed %d (--allow-dev-seed)"
                 % (a.seed, SEEDS, DEV_SEED))
    cfg = make_cfg(a.arm, a.seed)
    out = os.path.join(a.out, "%s_s%d.json" % (a.arm, a.seed)) if a.out else None
    rec = run(cfg, out, p_max_override=a.p_max, log=lambda m: print(m, flush=True))
    print(json.dumps({k: v for k, v in rec["summary"].items() if k in ("P50", "P90")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
