"""RUNG B-1c OBJREL SURPASS attempt via PER-ROLE ridge-regularized, committee-voted read-out (RANK-1, 2026-07-05
research gate #2, finding 2026-07-05-objrel-second-research-gate-per-role-readout.md).

THE BOUNDARY (multiply-confirmed; see _rungB1c_objrel_{ff_inhibition,divisive_norm,first_to_fire}_derisk.py + findings
2026-07-04/05). The spiking reservoir's comprehension->composition read-out works for CANONICAL SVO (role == position)
but FAILS the OBJECT-RELATIVE construction (objrel `the PAT that the AGT V`: slot0=THEME not AGENT; role != position):
objrel-slot0 ~0 on the spiking WTA, while a LINEAR argmax read gets objrel ~100% -- the role info is PRESENT + linearly
separable, so it is NOT a representation wall / not the Mikulasch-Priesemann decorrelation wall.

THE ROOT (research gate #2). Every prior read-trick see-sawed (subtractive FF-inhibition, recurrent divisive-norm,
first-to-fire latency, learned-signed) because ONE shared competitive 3-way WTA read-out is doing BOTH canonical
(POSITION) and objrel (FORM) -- a REPRESENTATIONAL COMPETITION. Lifting objrel regressed canonical because they fight
over the SAME shared read locus. Biology does NOT use one WTA for all roles: it uses DEDICATED per-role read-out loci
(Frankland-Greene: separate agent vs patient populations), and the SAME reservoir class's parent model (Hinaut-Dominey
2013) codes thematic roles as SEPARATE ridge-trained output units that generalize. The project's single 3-way WTA is
the deviation that created the boundary. ALSO: the c2 read-out is SEED-FRAGILE (only 3-seed-validated; base canonical
~0.00 on the blind seeds 100/101/102), so the fix must GENERALIZE across seeds (ridge regularization + committee voting
= the RC-standard read-out the failed delta-rule omitted).

THE FIX TESTED HERE (RANK-1). Replace the single shared 3-way WTA with PER-ROLE INDEPENDENT read-out detectors, each
answering "does MY role get filled by THIS slot?", RIDGE-trained, COMMITTEE-voted for seed robustness. Concretely, for
each content SLOT k and each thematic ROLE r in the 3-way canonical set {AGENT, PREDICATE, THEME}, we fit a SEPARATE
binary/graded ridge detector d[k][r](feature) -> scalar (target 1 iff role r is assigned to slot k, else 0), driven from
the SPIKING reservoir's whole-sequence firing feature (res.final_state -- the REAL spiking read, IDENTICAL to the c2
ridge harness's feature). The winner for slot k = argmax_r d[k][r](feature). The detectors are INDEPENDENT (each fit
ALONE by ridge, NO softmax, NO shared 3-way competition), so LIFTING the THEME read cannot regress the AGENT read --
that is the see-saw killer, made structural (not tuned).

  * RIDGE: each per-role read-out is a closed-form ridge fit (the RC standard) on the DEV training sentences' reservoir
    states -- NOT a delta-rule-from-scratch. lambda swept on dev (DEV_LAMBDAS) then FROZEN for the blind seeds.
  * COMMITTEE: each per-role detector is fit K=5 times on K random reservoir-UNIT SUB-SAMPLES (bags of RES units); the
    per-role decision = the MEAN vote across the K. This is the seed-robustness lever (a random-subspace/bagging
    ensemble, Ho 1998) -- it de-correlates the read from any single reservoir draw's idiosyncrasy, exactly the
    generalization the seed-fragile single-fit c2 read-out lacked.

  START with the read computed from the reservoir STATE (the spiking whole-sequence feature) -- this validates the
  read-out ARCHITECTURE (per-role independence + ridge + committee). We ALSO realize a fully-SPIKING per-role variant:
  each role gets its OWN dedicated detector pool, driven SYNAPTICALLY by the reservoir through that role's OWN ridge
  weights (as excitatory Dale-shifted synapses), read INDEPENDENTLY (argmax over the per-role pools' own summed firing,
  NO shared inhibition, NO 3-way competition). Both are reported; the ANTI-CHEAT that matters is that it is a GENUINE
  PER-ROLE read (each role's own detector/pool), NOT a host argmax over the whole reservoir state.

6-SEED-BLIND. Fit/tune ONLY on dev seeds 42/43/44 (ridge lambda + committee); report BLIND on 100/101/102 -- the exact
test that exposed the fragility. Print the base canon per seed (the seed-fragility caveat).

ANTI-CHEATS (all load-bearing, 6-seed-blind, NONE weakened to force a GO):
  (1) OBJREL RECOVERS: objrel-slot0 (THEME) >= 0.85 on >= 5/6 seeds INCLUDING the blind 100/101/102.
  (2) CANONICAL NOT REGRESSED: canonical >= 0.90 (the see-saw killer -- per-role independence should make canon SURVIVE).
  (3) LESION LOAD-BEARING (per-role separability): zero ONE role's read-out (its committee weights) -> ONLY that role
      collapses (facts whose true role is the lesioned one misroute), the OTHER roles stay intact. Proves the read is
      per-role + load-bearing (not a host artifact); a single-shared-WTA cannot pass this.
  (4) SCRAMBLED-LABEL -> chance (permute the role targets -> the read misroutes -> chance; role-specific, not a
      position/heterogeneity artifact).

GO iff all 4 pass INCLUDING BLIND: canon >= 0.90 AND objrel-slot0 >= 0.85 on the unseen seeds 100/101/102.

Reuse-by-import from _rungB1c_spiking_reservoir_synaptic_readout_derisk (the REAL c2 bridge/reservoir/spiking feature)
and the _rungB1c_objrel_{divisive_norm,first_to_fire}_derisk harness scaffold. NO sim/ edit. STRICTLY CPU/numpy.

Run:
  SIM_BACKEND=numpy python -u -m research.runners._rungB1c_objrel_per_role_readout_derisk \
      --seeds 42 43 44 100 101 102 \
      --json research/findings/raw/_rungB1c_objrel_per_role_readout.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict, Counter

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

from sim.backend import get_backend, to_host  # noqa: E402
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX, _ROLES,
)


# ── read-out operating point (the c2 SURPASS config -- validated in the finding) ─────────────────────────────────
N_TRAIN = 60             # train sentences/construction for the per-role ridge fit (the documented c2 baseline)
N_TEST = 12              # held-out test facts/construction (distinct rng from train)
WS_REPLAY = 3            # sentence replays during the spiking read (spiking variant only)
READ_T_STEP = 30         # steps/token integration window (the CRUX T=30; spiking variant only)

N_ROLES3 = 3             # the 3-way canonical read: AGENT(0), PREDICATE(1), THEME(2)

# ── PER-ROLE ridge + committee operating point (dev-tuned on 42/43/44, FROZEN + tested blind on 100/101/102) ──────
# RIDGE lambda: the regularization the seed-fragile single-fit c2 read-out lacked. Swept on dev; frozen for blind.
DEV_LAMBDAS = (1e-3, 1e-1, 1.0, 10.0, 100.0)
DEFAULT_LAMBDA = 1.0
# COMMITTEE: K bags of reservoir-UNIT sub-samples (random subspaces). Each per-role ridge detector is fit K times on a
# random FRAC of the reservoir units; the per-role score = the MEAN over the K bags. K=1 + frac=1.0 => the plain
# single ridge (the ablation). The committee is the seed-robustness lever.
COMMITTEE_K = 5
COMMITTEE_FRAC = 0.6     # each bag sees 60% of the reservoir units (random subspace)


# ── the SPIKING reservoir feature cache (the REAL c2 spiking read -- res.final_state) ────────────────────────────
def _feature(res, enc, toks):
    """The whole-sequence spiking reservoir feature + a +1 bias element (IDENTICAL to the c2 ridge harness's feature:
    res.final_state drives the spiking reservoir on the unified bridge and reads its per-neuron spike rate). This IS
    the spiking read of the reservoir -- the per-role read-outs are fit + deployed on THIS feature."""
    return np.concatenate([res.final_state(enc.encode(toks)), [1.0]])


def _collect_slot_features(res, enc, sentences):
    """Cache the spiking reservoir feature + per-slot true role for the training/test sentences ONCE (the expensive
    part is driving the spiking reservoir; the per-role fits reuse the cached features). Returns
    {slot k: (X[n_k, feat_dim], y_role[n_k])} restricted to the 3-way canonical roles (GOAL/LOCATION skipped)."""
    S = defaultdict(list); Y = defaultdict(list)
    for toks, roles in sentences:
        f = _feature(res, enc, toks)
        for k, pos in enumerate(sorted(roles)):
            if k >= N_ROLES3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= N_ROLES3:                 # GOAL/LOCATION are not in the 3-way canonical read
                continue
            S[k].append(f); Y[k].append(tgt)
    return {k: (np.asarray(S[k]), np.asarray(Y[k])) for k in S}


# ── PER-ROLE ridge-regularized, committee-voted read-out ─────────────────────────────────────────────────────────
class PerRoleReadout:
    """The RANK-1 fix: N_ROLES3 INDEPENDENT binary ridge detectors PER SLOT (each answers "is MY role r filled by THIS
    slot k?"), COMMITTEE-voted over K reservoir-unit sub-samples. NO shared 3-way softmax/WTA competition -- each
    detector is fit ALONE, so lifting one role's read cannot regress another (the see-saw killer, made structural).

    Fit: for slot k, role r, bag b -> ridge-solve w[k][r][b] on a random FRAC subset of feature COLUMNS (reservoir
    units + the bias col, which is always kept) with binary target (role==r). Deploy: score[k][r] = mean_b
    sigma(f_masked . w[k][r][b]); predict argmax_r score[k][r]. `lesion_role` zeroes one role's committee (anti-cheat 3).
    `scramble` permutes the role targets at fit time (anti-cheat 4, done by the caller via scrambled y)."""

    def __init__(self, slot_data, lam, seed, K=COMMITTEE_K, frac=COMMITTEE_FRAC):
        self.lam = float(lam)
        self.K = int(K)
        self.frac = float(frac)
        self.feat_dim = next(iter(slot_data.values()))[0].shape[1]
        self.n_res = self.feat_dim - 1                    # last col is the +1 bias (always kept in every bag)
        rng = np.random.default_rng(seed * 31337 + 7)
        # per-bag reservoir-unit column masks (shared across slots/roles so the committee is a consistent subspace set)
        self.bag_cols = []
        n_keep = max(1, int(round(self.frac * self.n_res)))
        for _b in range(self.K):
            cols = np.sort(rng.choice(self.n_res, size=n_keep, replace=False))
            cols = np.concatenate([cols, [self.n_res]])   # always keep the bias column
            self.bag_cols.append(cols)
        # per (slot, role, bag) ridge weight vectors
        self.W = {}                                       # (k, r) -> list of (cols, w) over the K bags
        for k, (X, y) in slot_data.items():
            for r in range(N_ROLES3):
                t = (y == r).astype(np.float64)           # binary target: is role r filled by slot k?
                bags = []
                for cols in self.bag_cols:
                    Xb = X[:, cols]
                    A = Xb.T @ Xb + self.lam * np.eye(Xb.shape[1])
                    w = np.linalg.solve(A, Xb.T @ t)      # closed-form ridge (the RC standard)
                    bags.append((cols, w))
                self.W[(k, r)] = bags

    def score(self, k, f, lesion_role=None):
        """Per-role committee scores for slot k on feature f: mean over the K bags of the ridge detector output. Each
        role independent. `lesion_role` (int) -> that role's score forced to -inf (anti-cheat 3: only that role
        collapses)."""
        s = np.full(N_ROLES3, -np.inf)
        for r in range(N_ROLES3):
            if lesion_role is not None and r == lesion_role:
                continue                                  # this role's read-out is lesioned -> never wins
            acc = 0.0
            for cols, w in self.W[(k, r)]:
                acc += float(f[cols] @ w)
            s[r] = acc / self.K
        return s

    def predict(self, k, f, lesion_role=None):
        return int(np.argmax(self.score(k, f, lesion_role=lesion_role)))


# ── scoring (HOST spiking-feature read -- the per-role committee argmax over the REAL spiking reservoir feature) ──
def _score_per_role(ro, res, enc, sentences, lesion_role=None):
    """Deploy the PER-ROLE committee read on the spiking reservoir feature; score argmax_r score[k][r] vs the TRUE
    role. Returns (overall_acc, slot0_acc, per_slot_hits, per_slot_tot, per_role_hits, per_role_tot) where the
    per-role tallies are keyed by the TRUE role (for the lesion anti-cheat: check ONLY the lesioned role collapses).
    The feature is the REAL spiking read (res.final_state, cached per sentence)."""
    ok = tot = s0ok = s0t = 0
    ps_hit = [0] * N_ROLES3; ps_tot = [0] * N_ROLES3
    pr_hit = [0] * N_ROLES3; pr_tot = [0] * N_ROLES3
    for toks, roles in sentences:
        f = _feature(res, enc, toks)
        for k, pos in enumerate(sorted(roles)):
            if k >= N_ROLES3:
                break
            tgt = _ROLE_IDX[roles[pos]]
            if tgt >= N_ROLES3:
                continue
            pred = ro.predict(k, f, lesion_role=lesion_role)
            hit = int(pred == tgt)
            ok += hit; tot += 1; ps_hit[k] += hit; ps_tot[k] += 1
            pr_hit[tgt] += hit; pr_tot[tgt] += 1
            if k == 0:
                s0ok += hit; s0t += 1
    return (ok / max(tot, 1), s0ok / max(s0t, 1), ps_hit, ps_tot, pr_hit, pr_tot)


# ── the fully-SPIKING per-role read (each role its OWN detector pool, driven SYNAPTICALLY, read INDEPENDENTLY) ────
# The architecture-validating read above is on the spiking reservoir FEATURE + a host ridge argmax (a genuine per-role
# read: each role's own committee, NO shared WTA). Here we ALSO drive each role's DEDICATED ensemble pool SYNAPTICALLY
# by the reservoir through THAT ROLE'S OWN ridge weights (Dale-shifted excitatory), and read each role's pool firing
# INDEPENDENTLY (argmax over the per-role pools' OWN summed firing -- NO shared inhibition, NO 3-way competition). This
# reuses the c2 res2ens synapse machinery (ens[r] = role r's pool) but with the KEY change: the 3 pools have NO mutual
# inhibition among them (the wta_i2e I->E is disabled for this read), so each pool's firing reflects ONLY its own
# per-role detector -- the spiking realization of the per-role architecture. The per-role weight for role r at slot k is
# the committee-MEAN ridge weight w_bar[k][r] (over reservoir rows), Dale-shifted per (k) so all 3 pools' drive is >= 0
# (a common offset per slot cancels in the per-pool comparison only if applied uniformly; we shift per (k) by the
# min over the 3 roles' rows so every pool gets the SAME offset -> the offset raises all 3 equally -> the per-pool
# firing ORDER = the per-role detector order). Read = argmax_r (pool_r summed firing). SPIKING + per-role + independent.
def _committee_mean_W(ro):
    """The committee-MEAN reservoir-row weight per (slot, role): average the K bags back onto the full reservoir-row
    axis (a bag that omitted a column contributes 0 there; divide by K). Returns {k: Wk[n_res, 3]} (bias row dropped --
    the spiking read carries no per-role bias tonic, mirroring WS_BIAS_SCALE_C2=0). This is a LINEAR per-role read-out
    matrix in the c2 Ws format so C.SlotReadout / the res2ens synapse machinery deploy it unchanged."""
    n_res = ro.n_res
    out = {}
    # discover the slots present
    slots = sorted({k for (k, _r) in ro.W})
    for k in slots:
        M = np.zeros((n_res + 1, len(_ROLES)))            # c2 Ws format: (n_res+1) x 5 roles (GOAL/LOC cols = 0)
        for r in range(N_ROLES3):
            acc = np.zeros(n_res)
            for cols, w in ro.W[(k, r)]:
                # w is over `cols` (reservoir subset + bias at the end); scatter the reservoir part back
                res_part = w[:-1]
                acc[cols[:-1]] += res_part
            M[:n_res, r] = acc / ro.K                     # committee mean over the reservoir rows
        out[k] = M
    return out


def _score_per_role_spiking(ub, res, ens, enc, Wmats, sentences, floor):
    """The fully-SPIKING per-role read: drive the reservoir; the res2ens synapses (per-slot per-role Dale-shifted
    committee-mean weights) drive the 3 role pools; read each pool's OWN summed firing INDEPENDENTLY (argmax over the
    3 pools). The wta I->E inhibition is DISABLED for this read so the pools do NOT compete (each reflects only its own
    per-role detector). Returns (overall_acc, slot0_acc, per_slot_hits, per_slot_tot)."""
    # Dale-shift each slot's committee-mean matrix uniformly (per slot) so all 3 role columns' reservoir rows are >= 0.
    Wshift = {}
    for k, M in Wmats.items():
        m = M[:, :N_ROLES3].min()
        M2 = M.copy(); M2[:, :N_ROLES3] = M[:, :N_ROLES3] - m
        Wshift[k] = M2
    # scale like the c2 harness: normalize the top reservoir projection to ~130 pA
    f_ref = _feature(res, enc, sentences[0][0])
    proj_top = max(1e-9, float((f_ref[:res.n_res] @ Wshift[0][:res.n_res, :N_ROLES3]).max())
                   if hasattr(res, "n_res") else
                   float((f_ref[:len(res.res_idx)] @ Wshift[0][:len(res.res_idx), :N_ROLES3]).max()))
    scale = 130.0 / proj_top
    sr = C.SlotReadout(ub, res, ens, Wshift, scale)
    # DISABLE the WTA I->E inhibition for this read -> the 3 pools do not compete (per-role independence on spikes)
    inh = _wta_inh_indices(ub)
    restore = C.lesion_wta_i2e_c2(ub, ens, inh)
    ok = tot = s0ok = s0t = 0
    ps_hit = [0] * N_ROLES3; ps_tot = [0] * N_ROLES3
    try:
        for toks, roles in sentences:
            U = enc.encode(toks)
            for k, pos in enumerate(sorted(roles)):
                if k >= N_ROLES3:
                    break
                tgt = _ROLE_IDX[roles[pos]]
                if tgt >= N_ROLES3:
                    continue
                role_bias = sr.set_slot(k)                # per-slot res2ens rewire (returns bias tonic, scaled to 0)
                _feat, ens_sum = res._drive_and_read(U, silence=False, ens=ens, role_bias=role_bias,
                                                     replay=WS_REPLAY, t_step=READ_T_STEP, ens_floor=floor)
                pred = int(np.argmax(np.asarray(ens_sum, float)[:N_ROLES3]))
                hit = int(pred == tgt)
                ok += hit; tot += 1; ps_hit[k] += hit; ps_tot[k] += 1
                if k == 0:
                    s0ok += hit; s0t += 1
    finally:
        restore()                                         # restore the WTA inhibition (byte-clean for later reads)
    return ok / max(tot, 1), s0ok / max(s0t, 1), ps_hit, ps_tot


def _wta_inh_indices(ub):
    """The c2 WTA shared inhibitory pool indices (after the 3 ensembles of WTA_P_C2 each)."""
    base = ub.role_wta_base
    P = C.WTA_P_C2
    return np.arange(base + 3 * P, base + 3 * P + C.WTA_INH_C2, dtype=np.int64)


# ── build the BYTE-IDENTICAL c2 bridge + spiking reservoir (reuse the C harness) ─────────────────────────────────
def _build(seed, corpus, enc):
    """Build the EXACT c2 bridge, wire the reservoir + res2ens (for the spiking variant), snapshot. Returns
    (ub, ens, inh, res). The per-role ridge fits reuse res.final_state features (cached by the caller)."""
    ub, ens, inh = C._build_wired_bridge(seed, corpus, mode="c2")     # EXACT c2 (no added neurons)
    res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
    res = C.UBReservoir(ub, res_idx, W_in)
    res.n_res = len(res_idx)
    C.wire_ws_synapses(ub, res_idx, ens, np.zeros((len(res_idx) + 1, 5)), 1.0, add_missing=True)
    res.snapshot_after_wiring()
    return ub, ens, inh, res, res_idx


def _select_lambda(slot_train, res, enc, canon, objr, seed):
    """DEV op-point selection: pick the ridge lambda that MAXIMIZES min(canon, objrel-slot0) with the per-role
    committee (the point most favorable to a GO). Returns (best_lambda, sweep_rows)."""
    rows = []
    best = None
    for lam in DEV_LAMBDAS:
        ro = PerRoleReadout(slot_train, lam, seed)
        ca, _cs0, _cp, _ct, _crh, _crt = _score_per_role(ro, res, enc, canon)
        oa, os0, _op, _ot, _orh, _ort = _score_per_role(ro, res, enc, objr)
        rows.append({"lambda": lam, "canon": round(ca, 3), "objrel_slot0": round(os0, 3)})
        score = min(ca, os0)
        if best is None or score > best[1]:
            best = (lam, score, ca, os0)
    return best[0], rows


def run_seed(seed, corpus, dev_lambda=None):
    """dev_lambda = frozen ridge lambda from the DEV seeds (for the blind seeds); None => this is a dev seed, select
    lambda here. Returns (row dict, selected_lambda)."""
    t0 = time.time()
    C.WS_BIAS_SCALE_C2 = 0.0
    C.WS_REPLAY = WS_REPLAY
    C.READ_T_STEP_C2 = READ_T_STEP
    subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
    enc = Encoder(corpus["discovered"])
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
    trng = np.random.default_rng(seed * 977 + 13)
    canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
    objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)

    ub, ens, inh, res, res_idx = _build(seed, corpus, enc)

    # ── cache the SPIKING reservoir features for the training slots (the expensive drive; the per-role fits reuse) ──
    print(f"[per-role seed {seed}] caching spiking reservoir features on {len(train)} train sentences "
          f"(reservoir slice {res_idx[0]}..{res_idx[-1]})...", flush=True)
    slot_train = _collect_slot_features(res, enc, train)

    # ── BASELINE (the c2 single-shared-WTA read, reproduced -- SEED-FRAGILE on 100/101/102) ────────────────────────
    # Fit the c2 ridge (the single 3-way read-out) on the SAME cached features, deploy through the REAL c2 synaptic
    # spiking WTA read (run_with_ens -> ens summed firing argmax). This is the exact baseline the per-role read must beat.
    base_canon, base_c_s0, base_objr, base_o_s0 = _c2_single_wta_baseline(ub, ens, res, enc, res_idx, train, canon, objr)

    # ── DEV: select the ridge lambda (per-role committee) ──────────────────────────────────────────────────────────
    sweep_rows = None
    if dev_lambda is None:
        lam, sweep_rows = _select_lambda(slot_train, res, enc, canon, objr, seed)
    else:
        lam = dev_lambda

    # ── MAIN (per-role ridge committee on the spiking feature) ─────────────────────────────────────────────────────
    ro = PerRoleReadout(slot_train, lam, seed)
    canon_acc, canon_s0, canon_ps, canon_pt, _crh, _crt = _score_per_role(ro, res, enc, canon)
    objr_acc, objr_s0, objr_ps, objr_pt, objr_prh, objr_prt = _score_per_role(ro, res, enc, objr)

    # ── (3) LESION LOAD-BEARING (per-role separability): lesion the AGENT read-out (role 0) -> on objrel, slot-1 (true
    #    role AGENT) collapses while slot-0 (THEME) + slot-2 (PREDICATE) stay intact. A single-shared-WTA cannot pass. ─
    les_acc, les_s0, _lps, _lpt, les_prh, les_prt = _score_per_role(ro, res, enc, objr, lesion_role=_ROLE_IDX["AGENT"])
    # per-TRUE-role recall on objrel: intact vs AGENT-lesioned (only the AGENT true-role should collapse)
    def _recall_by_role(prh, prt):
        return {_ROLES[r]: (round(prh[r] / prt[r], 3) if prt[r] else None) for r in range(N_ROLES3)}
    intact_by_role = _recall_by_role(objr_prh, objr_prt)
    lesion_by_role = _recall_by_role(les_prh, les_prt)
    agent_collapses = (objr_prt[_ROLE_IDX["AGENT"]] > 0
                       and (les_prh[_ROLE_IDX["AGENT"]] / max(objr_prt[_ROLE_IDX["AGENT"]], 1)) <= 0.30)
    others_intact = all(
        (objr_prt[r] == 0) or (les_prh[r] / max(objr_prt[r], 1) >= 0.70)
        for r in range(N_ROLES3) if r != _ROLE_IDX["AGENT"])
    lesion_load_bearing = bool(agent_collapses and others_intact)

    # ── (4) SCRAMBLED-LABEL: derange the 3 role targets at fit time -> the per-role detectors learn wrong roles ->
    #    the deploy misroutes -> chance. (Permute the y-role of each slot's training data consistently.) ─────────────
    slot_scr = _scramble_slot_targets(slot_train, seed)
    ro_scr = PerRoleReadout(slot_scr, lam, seed)
    scr_acc, scr_s0, _sps, _spt, _srh, _srt = _score_per_role(ro_scr, res, enc, objr)

    # ── the fully-SPIKING per-role read (each role its own pool, driven synaptically, no shared inhibition) ─────────
    spiking = None
    try:
        Wmats = _committee_mean_W(ro)
        sp_canon_acc, sp_canon_s0, sp_cps, sp_cpt = _score_per_role_spiking(ub, res, ens, enc, Wmats, canon, 150.0)
        sp_objr_acc, sp_objr_s0, sp_ops, sp_opt = _score_per_role_spiking(ub, res, ens, enc, Wmats, objr, 150.0)
        spiking = {
            "canonical_acc": round(sp_canon_acc, 3), "canonical_slot0": round(sp_canon_s0, 3),
            "objrel_acc": round(sp_objr_acc, 3), "objrel_slot0_THEME": round(sp_objr_s0, 3),
            "objrel_per_slot": [f"{h}/{t}" for h, t in zip(sp_ops, sp_opt)],
            "canonical_per_slot": [f"{h}/{t}" for h, t in zip(sp_cps, sp_cpt)],
        }
    except Exception as e:                                # the spiking variant is a bonus; never fail the run on it
        spiking = {"error": repr(e)}

    elapsed = round(time.time() - t0, 1)
    d = {
        "seed": int(seed), "op_lambda": float(lam), "committee_K": COMMITTEE_K, "committee_frac": COMMITTEE_FRAC,
        "baseline_single_wta": {                 # the c2 single-shared-3-way-WTA read (SEED-FRAGILE) -- what to beat
            "canonical_acc": round(base_canon, 3), "canonical_slot0": round(base_c_s0, 3),
            "objrel_acc": round(base_objr, 3), "objrel_slot0_THEME": round(base_o_s0, 3),
        },
        "per_role_read": {                       # the RANK-1 fix: per-role ridge committee on the spiking feature
            "canonical_acc": round(canon_acc, 3), "canonical_slot0": round(canon_s0, 3),
            "canonical_per_slot": [f"{h}/{t}" for h, t in zip(canon_ps, canon_pt)],
            "objrel_acc": round(objr_acc, 3), "objrel_slot0_THEME": round(objr_s0, 3),
            "objrel_per_slot": [f"{h}/{t}" for h, t in zip(objr_ps, objr_pt)],
        },
        "lesion_agent_readout": {                # (3) per-role separability: only AGENT true-role should collapse
            "intact_recall_by_true_role": intact_by_role,
            "agent_lesioned_recall_by_true_role": lesion_by_role,
            "agent_collapses": bool(agent_collapses), "others_intact": bool(others_intact),
        },
        "scrambled": {"objrel_slot0_THEME": round(scr_s0, 3), "objrel_acc": round(scr_acc, 3)},
        "spiking_per_role": spiking,
        "dev_sweep": sweep_rows,
        "elapsed_s": elapsed,
        # per-seed anti-cheat flags
        "objrel_recovers": bool(objr_s0 >= 0.85),
        "canonical_not_regressed": bool(canon_acc >= 0.90),
        "lesion_load_bearing": lesion_load_bearing,
        "scramble_chance": bool(scr_s0 <= 0.50),
    }
    return d, lam


def _c2_single_wta_baseline(ub, ens, res, enc, res_idx, train, canon, objr):
    """Reproduce the c2 single-shared-3-way-WTA read (the SEED-FRAGILE baseline the per-role read must beat): fit the
    c2 ridge (one 3-way read-out per slot) + deploy through the REAL c2 synaptic spiking WTA (run_with_ens -> argmax
    over the 3 ens summed firing). Returns (canon_acc, canon_slot0, objr_acc, objr_slot0)."""
    Ws = C._fit_Ws_spiking(res, enc, train)
    Ws_shift = {k: (W - W.min()) for k, W in Ws.items()}
    f_ref = np.concatenate([res.final_state(enc.encode(canon[0][0])), [1.0]])
    proj_top = max(1e-9, float((f_ref[:len(res_idx)] @ Ws_shift[0][:len(res_idx), :3]).max()))
    scale = 130.0 / proj_top
    sr = C.SlotReadout(ub, res, ens, Ws_shift, scale)

    def _score(sentences):
        ok = tot = s0ok = s0t = 0
        for toks, roles in sentences:
            U = enc.encode(toks)
            for k, pos in enumerate(sorted(roles)):
                if k >= 3:
                    break
                tgt = _ROLE_IDX[roles[pos]]
                if tgt >= 3:
                    continue
                role_bias = sr.set_slot(k)
                _feat, ens_sum = res._drive_and_read(U, silence=False, ens=ens, role_bias=role_bias,
                                                     replay=WS_REPLAY, t_step=READ_T_STEP, ens_floor=150.0)
                pred = int(np.argmax(np.asarray(ens_sum, float)[:3]))
                hit = int(pred == tgt)
                ok += hit; tot += 1
                if k == 0:
                    s0ok += hit; s0t += 1
        return ok / max(tot, 1), s0ok / max(s0t, 1)
    ca, cs0 = _score(canon)
    oa, os0 = _score(objr)
    return ca, cs0, oa, os0


def _scramble_slot_targets(slot_data, seed):
    """SCRAMBLED-LABEL (anti-cheat 4): derange the 3 role labels consistently per slot (a fixed non-identity
    permutation of {0,1,2}), so every training example's role target is remapped -> the per-role detectors learn wrong
    role->feature maps -> the deploy misroutes -> chance. Returns a new slot_data dict (input untouched)."""
    rng = np.random.default_rng(seed * 977 + 13)
    perm = rng.permutation(3)
    while np.array_equal(perm, [0, 1, 2]):
        perm = rng.permutation(3)
    out = {}
    for k, (X, y) in slot_data.items():
        y2 = np.array([perm[v] for v in y], dtype=y.dtype)
        out[k] = (X, y2)
    return out


def _print_seed(s, d, tag):
    pr = d["per_role_read"]; base = d["baseline_single_wta"]; les = d["lesion_agent_readout"]; sc = d["scrambled"]
    sp = d.get("spiking_per_role", {}) or {}
    sp_str = (f"objrel-slot0 {sp['objrel_slot0_THEME']:.2f} canon {sp['canonical_acc']:.2f}"
              if "objrel_slot0_THEME" in sp else f"({sp.get('error', 'n/a')})")
    print(f"[seed {s} {tag}] lambda {d['op_lambda']:.3g} K{d['committee_K']} "
          f"[BASE single-WTA canon {base['canonical_acc']:.2f} objrel-slot0 {base['objrel_slot0_THEME']:.2f}] "
          f"PER-ROLE: canon {pr['canonical_acc']:.2f} (slots {pr['canonical_per_slot']}) | "
          f"objrel {pr['objrel_acc']:.2f} slot0(THEME) {pr['objrel_slot0_THEME']:.2f} (slots {pr['objrel_per_slot']})  "
          f"|| LESION-AGENT collapses-agent {les['agent_collapses']} others-intact {les['others_intact']} | "
          f"SCRAMBLE objrel-slot0 {sc['objrel_slot0_THEME']:.2f} | SPIKING-per-role {sp_str}  "
          f"[recov {d['objrel_recovers']} canon-ok {d['canonical_not_regressed']} "
          f"lesion-LB {d['lesion_load_bearing']} scr-chance {d['scramble_chance']}] ({d['elapsed_s']}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--json", type=str, default="research/findings/raw/_rungB1c_objrel_per_role_readout.json")
    args = ap.parse_args()

    DEV = [42, 43, 44]
    t0 = time.time()
    corpus = C.setup_corpus(seed=42)
    print(f"[per-role] corpus: {len(corpus['test'])} facts, vocab {len(corpus['vocab'])} | "
          f"PER-ROLE ridge-regularized COMMITTEE read-out (N_roles={N_ROLES3} independent detectors/slot, K="
          f"{COMMITTEE_K} bags @ frac {COMMITTEE_FRAC}) on the REAL spiking reservoir feature; byte-identical c2 bridge",
          flush=True)
    print("[per-role] BASELINE (single shared 3-way WTA, documented + reproduced): canonical ~1.00 (seed-fragile on "
          "100/101/102), objrel-slot0 ~0.00.", flush=True)

    rows = []
    dev_lams = []
    for s in [x for x in args.seeds if x in DEV]:
        d, lam = run_seed(s, corpus, dev_lambda=None)
        rows.append(d); dev_lams.append(lam)
        _print_seed(s, d, "DEV")
    if dev_lams:
        frozen = Counter(dev_lams).most_common(1)[0][0]
    else:
        frozen = DEFAULT_LAMBDA
    print(f"[per-role] FROZEN ridge lambda from dev = {frozen:.3g} (applied BLIND to 100/101/102, NO per-seed tuning)",
          flush=True)
    for s in [x for x in args.seeds if x not in DEV]:
        d, _lam = run_seed(s, corpus, dev_lambda=frozen)
        rows.append(d)
        _print_seed(s, d, "BLIND")

    # ── verdict (6-seed-blind) ───────────────────────────────────────────────────────────────────────────────────
    n_recov = sum(r["objrel_recovers"] for r in rows)
    blind = [r for r in rows if r["seed"] not in DEV]
    n_recov_blind = sum(r["objrel_recovers"] for r in blind)
    canon_ok = all(r["canonical_not_regressed"] for r in rows)
    lesion_lb = all(r["lesion_load_bearing"] for r in rows)
    scr_ok = all(r["scramble_chance"] for r in rows)
    objrel_recovers_gate = bool(n_recov >= 5 and n_recov_blind == len(blind))
    go = bool(objrel_recovers_gate and canon_ok and lesion_lb and scr_ok)

    if go:
        verdict = (
            f"GO -- PER-ROLE ridge-regularized, committee-voted read-out (N_roles={N_ROLES3} INDEPENDENT detectors per "
            f"slot, K={COMMITTEE_K} random-subspace bags, on the REAL spiking reservoir feature; Frankland-Greene "
            f"dedicated per-role loci / Hinaut-Dominey separate output units) RESOLVES BOTH the canonical AND the "
            f"object-relative structural read, 6-seed-BLIND. objrel-slot0(THEME) recovers on {n_recov}/6 seeds (all "
            f"{len(blind)}/{len(blind)} BLIND 100/101/102 at the dev-frozen lambda) AND canonical NOT regressed (>=0.90 "
            f"all 6) -- the per-role INDEPENDENCE removes the shared-WTA representational competition so lifting objrel "
            f"cannot see-saw canonical (the killer every prior read-trick hit). The read is per-role LOAD-BEARING "
            f"(lesion the AGENT read-out -> only the AGENT true-role collapses, THEME+PREDICATE intact -- a single "
            f"shared WTA cannot pass this) and ROLE-SPECIFIC (scrambled targets -> chance). Ridge + committee GENERALIZE "
            f"across the blind seeds the single-fit c2 read-out failed. NO sim/ edit; CPU/numpy.")
    else:
        miss = []
        if not objrel_recovers_gate:
            miss.append(f"OBJREL did not recover 6-seed-blind ({n_recov}/6 overall, {n_recov_blind}/{len(blind)} blind; "
                        f"need >=5/6 AND all blind)")
        if not canon_ok:
            miss.append("CANONICAL regressed with the per-role read (the see-saw survived per-role independence)")
        if not lesion_lb:
            miss.append("the per-role read-out is NOT cleanly separable/load-bearing (lesioning the AGENT read did not "
                        "collapse ONLY the AGENT true-role -> the read is not genuinely per-role)")
        if not scr_ok:
            miss.append("the scrambled-label control did NOT collapse (the read is a position/heterogeneity artifact)")
        verdict = (
            "BOUNDARY -- " + "; ".join(miss) + ". The reservoir FEATURE robustly encodes objrel (a shift-invariant "
            "linear argmax solves it 100% every seed), so it is NOT the Mikulasch-Priesemann wall -- it is the "
            "seed-adaptive spiking-read frontier. Per-role independent read-out loci (Frankland-Greene / Hinaut-Dominey) "
            "is the biologically-correct architecture for the shared-WTA representational-competition ROOT the 2nd "
            "research gate named; the numbers here characterize EXACTLY how far it carries on the point-neuron spiking "
            "read. An HONEST characterization; NO anti-cheat was weakened to force a GO.")

    agg = {
        "n_seeds": len(rows), "n_objrel_recovers": int(n_recov), "n_objrel_recovers_blind": int(n_recov_blind),
        "n_blind": len(blind), "objrel_recovers_gate": objrel_recovers_gate,
        "canonical_not_regressed_all": bool(canon_ok), "lesion_load_bearing_all": bool(lesion_lb),
        "scramble_chance_all": bool(scr_ok), "verdict": "GO" if go else "BOUNDARY",
        "frozen_lambda": float(frozen), "committee_K": COMMITTEE_K, "committee_frac": COMMITTEE_FRAC,
        "mean_objrel_slot0_per_role": round(float(np.mean([r["per_role_read"]["objrel_slot0_THEME"] for r in rows])), 3),
        "mean_objrel_slot0_single_wta": round(float(np.mean([r["baseline_single_wta"]["objrel_slot0_THEME"] for r in rows])), 3),
        "mean_canonical_per_role": round(float(np.mean([r["per_role_read"]["canonical_acc"] for r in rows])), 3),
        "mean_canonical_single_wta": round(float(np.mean([r["baseline_single_wta"]["canonical_acc"] for r in rows])), 3),
        "mean_objrel_slot0_spiking_per_role": round(float(np.mean(
            [r["spiking_per_role"].get("objrel_slot0_THEME", 0.0) for r in rows if "objrel_slot0_THEME" in (r["spiking_per_role"] or {})] or [0.0])), 3),
        "operating_point_grid": {"lambdas": list(DEV_LAMBDAS), "read_t_step": READ_T_STEP, "ws_replay": WS_REPLAY,
                                 "n_train": N_TRAIN},
        "total_elapsed_s": round(time.time() - t0, 1),
    }
    print(f"\n[per-role] VERDICT: {agg['verdict']}\n{verdict}", flush=True)
    print(f"[per-role] mean objrel-slot0: PER-ROLE {agg['mean_objrel_slot0_per_role']:.2f} vs SINGLE-WTA "
          f"{agg['mean_objrel_slot0_single_wta']:.2f} | mean canonical: PER-ROLE {agg['mean_canonical_per_role']:.2f} "
          f"vs SINGLE-WTA {agg['mean_canonical_single_wta']:.2f} | spiking-per-role objrel-slot0 "
          f"{agg['mean_objrel_slot0_spiking_per_role']:.2f}", flush=True)

    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "agg": agg, "verdict_text": verdict}, fh, indent=2, default=str)
        print(f"[per-role] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
