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
P_GRID = (50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)

# biology-bound defaults (research/biology/ca3-superposed-fact-attractor.md constraints_config)
CA3_SPARSENESS = 0.01             # sparse CA3 code (rat CA3 a ~ 0.02; Rolls 2013); capacity ~ 1/(a ln(1/a))
MOSSY_FANIN = 46                  # mossy-fibre synapses per CA3 cell (Rolls 2013): the sparse strong detonator

ARMS = {
    # the companion arm: DG pattern separation supplies sparse CA3 codes
    "sparse_dg": {},
    # capacity law: every diluted fan-in doubled (recurrent, perforant, readout)
    "sparse_dg_c2": dict(c_rec=4000, c_pp=600, c_out=4000, p_grid=P_GRID + (40000,)),
    # the no-companion baseline: no DG, CA3 selected by a fixed EC->CA3 projection at a dense code
    "dense_nodg": dict(use_dg=False, a_ca3=0.05),
    # dissociation arm: sparse code WITHOUT the DG (is the gain sparseness or separation?)
    "sparse_nodg": dict(use_dg=False),
    # palimpsest companion: bounded integer synapses, small steps, stochastic heterosynaptic LTD
    "sparse_dg_bounded": dict(plasticity="bounded"),
}


@dataclass
class Cfg:
    seed: int
    arm: str = "sparse_dg"
    n_ec_role: int = 1000
    k_ec: int = 20
    n_ent: int = 2000
    n_rel: int = 64
    n_ca3: int = 10000
    a_ca3: float = CA3_SPARSENESS
    c_rec: int = 2000
    c_pp: int = 300
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

    def csr(self, lesion: str = "intact", rng: np.random.Generator | None = None) -> sp.csr_matrix:
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
        m = sp.csr_matrix((W.ravel().astype(np.float32), self.idx.ravel(), indptr), shape=(self.n_post, self.n_pre))
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
            dg = kwta((self.ec_dg.csr() @ Xs).toarray(), cfg.k_dg)
            ca3 = kwta((self.mossy.csr() @ _col_sparse(dg)).toarray(), cfg.k_ca3)
            return ca3, dg
        ca3 = kwta((self.ec_ca3_fix.csr() @ Xs).toarray(), cfg.k_ca3)
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
        Wpp = self.pp.csr()
        Wrec = self.rec.csr(rec_lesion, rng)
        Wout = self.out.csr()
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
    h = (net.rec.csr() @ _col_sparse(facts_x)).toarray()
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
    ca3_patterns = np.zeros((cfg.n_ca3, p_max), dtype=bool)
    rec = dict(config=asdict(cfg), build_s=build_s, checkpoints=[], sha_facts=hashlib.sha256(facts.tobytes()).hexdigest())
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
            ca3_patterns[:, written:written + b] = ca3
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
        net.rec.csr(); net.pp.csr(); net.out.csr()
        cp["materialize_s"] = time.time() - t_m
        for lesion in ("intact", "rec_zero", "rec_shuffle"):
            t_q = time.time()
            s0, s, y, conv = net.recall(facts[q_idx], lesion=lesion, rng=ev)
            dt_q = time.time() - t_q
            ok = net.name(y, facts[q_idx, 2])
            tgt = ca3_patterns[:, q_idx]
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
        xs = ca3_patterns[:, np.sort(ev.choice(P, size=min(cfg.n_xtalk, P), replace=False))]
        cp["crosstalk_dprime"] = crosstalk_dprime(net, xs)
        if cfg.plasticity == "covariance":
            H = net.rec.H
            touched = H >= 1
            cp["rec_synapses_touched_frac"] = float(touched.mean())
            cp["rec_touched_shared_by_ge2_facts_frac"] = float((H >= 2).sum() / max(touched.sum(), 1))
        cp["ca3_pattern_mean_pairwise_overlap"] = float(_mean_pairwise_overlap(ca3_patterns[:, :P], ev, cfg.k_ca3))
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
    return rec


def _mean_pairwise_overlap(X, rng, k, n_pairs=2000):
    P = X.shape[1]
    if P < 2:
        return float("nan")
    i = rng.integers(0, P, n_pairs)
    j = rng.integers(0, P, n_pairs)
    m = i != j
    return float(((X[:, i[m]] & X[:, j[m]]).sum(0) / k).mean())


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

G5_P = 1000          # sub-capacity operating point for the recurrent-lesion gate (fixed from the dev seed)
G1_LEVEL = 0.9
G2_CEILING = 0.2
G3_RATIO = 1.5
G4_BAND = (1.4, 3.0)
G5_DROP = 0.2
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


def gates_for_seed(S: dict) -> dict:
    """S: arm -> summary for ONE seed. Returns gate -> (pass: bool|None, value)."""
    g = {}
    sd = S.get("sparse_dg")
    # G1 learns + completes below capacity
    if sd:
        v = [_at(sd, 50), _at(sd, 200)]
        g["G1_learns"] = (None if None in v else all(x >= G1_LEVEL for x in v), v)
    # G2 the cliff is visible for every UNBOUNDED arm (instrument validity: otherwise VOID)
    vals = {}
    for arm in ("sparse_dg", "sparse_dg_c2", "dense_nodg", "sparse_nodg"):
        if arm in S:
            vals[arm] = S[arm]["recall"][-1]
    g["G2_cliff"] = (None if len(vals) < 4 else all(v <= G2_CEILING for v in vals.values()), vals)
    # G3 the companion (sparse DG coding) pushes capacity up
    if sd and "dense_nodg" in S:
        a, b = sd["P50"], S["dense_nodg"]["P50"]
        r = None if (a is None or b is None) else a / b
        g["G3_companion_capacity"] = (None if r is None else r >= G3_RATIO, r)
    # G4 capacity law: doubling synapses per cell roughly doubles capacity
    if sd and "sparse_dg_c2" in S:
        a, b = S["sparse_dg_c2"]["P50"], sd["P50"]
        r = None if (a is None or b is None) else a / b
        g["G4_capacity_law"] = (None if r is None else G4_BAND[0] <= r <= G4_BAND[1], r)
    # G5 the recurrent completion is load-bearing at the pre-registered sub-capacity point
    if sd:
        i, z, sh = _at(sd, G5_P), _at(sd, G5_P, "recall_rec_zero"), _at(sd, G5_P, "recall_rec_shuffle")
        ok = None if None in (i, z, sh) else (z <= i - G5_DROP and sh <= i - G5_DROP)
        attr = None
        if None not in (i, z):
            from tools.lab import attributable_to       # local import: aggregation runs on the main box
            chance = 1.0 / Cfg(seed=0).n_ent
            # fraction of above-chance recall that is NOT present when only the recurrent edge is removed
            attr = attributable_to("recurrent edge @P=%d" % G5_P, i - chance, z - chance)
        g["G5_recurrent_loadbearing"] = (ok, dict(intact=i, rec_zero=z, rec_shuffle=sh, attributable=attr))
    # G6 cost law (INTEGRITY SMOKE, not evidence): per-query time flat in P, synapse memory constant in P
    if sd:
        t = sd["per_query_ms_single"]
        flat = t[-1] / t[0] if t and t[0] > 0 else None
        const = len(set(sd["synapse_bytes"])) == 1
        g["G6_cost_flat_INTEGRITY"] = (None if flat is None else (flat <= 2.0 and const), dict(ratio=flat, mem_const=const))
    # G7 palimpsest: bounded synapses keep RECENT facts past capacity where the unbounded store has collapsed
    if sd and "sparse_dg_bounded" in S:
        b, u = S["sparse_dg_bounded"]["recall_recent"][-1], sd["recall_recent"][-1]
        g["G7_palimpsest"] = (b >= G7_RECENT_BOUNDED and u <= G7_RECENT_UNBOUNDED, dict(bounded=b, unbounded=u))
    # G8 crosstalk grows with P (storage is SHARED): d' falls ~ P^-1/2
    if sd:
        sl = _dprime_slope(sd)
        g["G8_shared_crosstalk"] = (None if sl is None else G8_SLOPE_BAND[0] <= sl <= G8_SLOPE_BAND[1], sl)
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
    # capacity-law fit: P50 = k * C / (a ln(1/a))  (reported; extrapolation is labelled as such)
    ks = []
    for s, S in per_seed.items():
        for arm in ("sparse_dg", "sparse_dg_c2"):
            if arm in S and S[arm]["P50"]:
                c = make_cfg(arm, s)
                ks.append(S[arm]["P50"] * c.a_ca3 * math.log(1 / c.a_ca3) / c.c_rec)
    k_fit = float(np.median(ks)) if ks else None
    extrap = None
    if k_fit:
        a, C = 0.005, 10000
        pmax = k_fit * C / (a * math.log(1 / a))
        extrap = dict(note="EXTRAPOLATION from the fitted k, not a measurement", n_ca3=100000, c_rec=C, a=a,
                      predicted_P50_facts=pmax, recurrent_synapses=100000 * C,
                      bits_per_recurrent_synapse=pmax * math.log2(2000) / (100000 * C))
    out = dict(grid_dir=grid_dir, seeds=list(seeds), missing=missing, verdict=verdict, k_fit=k_fit,
               k_per_seed_arm=ks, gpu_point_extrapolation=extrap)
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
        print("k_fit", res["k_fit"], "missing", len(res["missing"]))
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
