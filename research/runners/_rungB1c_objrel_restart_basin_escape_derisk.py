"""RUNG B-1c OBJREL EMERGENT-LEARNING closure -- BASIN-ESCAPE via K RANDOM RESTARTS per role, selected by the REWARD
CRITIC (2026-07-06). The de-risk of ONE biological lever on top of the VERIFIED-GENUINE dopamine-plasticity closure.

WHAT IS ALREADY ESTABLISHED (do NOT re-derive; see 2026-07-06-objrel-dopamine-plasticity-emergent-REACHES-most-seeds-
BOUNDARY-reachability.md, adversarially verified): the per-role Dale-legal SPIKING reward-modulated three-factor
plasticity in _rungB1c_objrel_dopamine_plasticity_derisk (BinaryRoleDetector + DopaminePlasticReadout) GENUINELY learns
objrel EMERGENTLY (spiking, Dale-legal, held-out, reward-load-bearing, NOT warm-started), reaching objrel-slot0 9/10 with
salience (BPTT-from-scratch was 0/6). It is correctly a BOUNDARY: NO single config is 6/6. THE RESIDUAL is a per-seed
STOCHASTIC INIT-BASIN REACHABILITY BERNOULLI -- the random Dale-init lands in the majority-AGENT basin ~10-40% of the
time, so the THEME detector never carves the minority direction on those seeds. This is NOT a representational wall (the
analytic Dale reference reads objrel 1.00 on ALL seeds; a host linear argmax generalizes objrel held-out ~100%).

THE LEVER DE-RISKED HERE -- multiple random restarts per role, selected by the REWARD CRITIC (biological basin-escape).
Since the failure is a per-seed Bernoulli (bad init basin), K independent random restarts per role detector, with the
REWARD CRITIC selecting the best restart, drives the miss-rate to ~(miss-rate)^K. This is biological exploration + reward-
critic selection (Miconi 2017 node-perturbation; Legenstein 2010 reward-modulated exploration -- the RL exploration the
emergent-learning research gate emphasized), NOT an ML trick: an animal EXPLORES multiple candidate policies and the DA
reward-prediction-error SELECTS the one that maximizes reward.

THE SELECTION HONESTY (the crux anti-cheat). Each restart is trained from a DIFFERENT random Dale-init. The restart is
SELECTED by the REWARD CRITIC = the accumulated TRAINING reward (the salience-weighted training-signal FIT) -- NOT the
test objrel accuracy (that would be cheating; the reward critic can only see training sentences' reward). Concretely the
critic score for a restart is the NEGATIVE salience-weighted training squared error -sum_i sw[i]*(tgt[i]-act_i)^2 -- the
SAME objective the graded-DA delta rule ascends (reward-weighted regression / graded-RPE three-factor), evaluated on the
TRAIN features only. A basin-missed restart fits the majority + misses the minority -> large salience-weighted training
error on the minority positives -> LOW training reward -> REJECTED; a basin-escaped restart fits the minority -> HIGH
training reward -> SELECTED. We ALSO report a "select-by-test-accuracy" ORACLE column (what K-restart WOULD achieve if it
could cheat) -- if train-reward-selection ~= test-oracle, the training reward is a valid critic; if train-reward-selection
<< test-oracle, the reward critic cannot distinguish basins (an honest negative -> the next lever). SALIENCE is KEPT (the
verified finding shows salience HELPS 9/10 vs 6/10 -- do not drop it).

PRE-REGISTERED (BEFORE the run, declared here, NO post-hoc partition selection):
  * FIXED 10-seed dev/blind split:  DEV = {42, 43, 44, 45, 46}   BLIND = {100, 101, 102, 103, 104}.
  * K-sweep:  K in {1, 3, 5, 8}.  Report per-config recovery on ALL 10 seeds.
  * GENUINELY-EMERGENT gate: a seed is COUNTED only if its PRE-learning (K=0 random Dale-init) objrel-slot0 < 0.85 (an
    init-lucky seed, pre >= 0.85 like the known seed 100, is EXCLUDED from the GO tally -- an init-lucky seed is not
    evidence the plasticity/restart works). The counted-set is FIXED by the K=0 pre-learning read (independent of K).
  * K=1 is the single-init baseline and MUST reproduce the current ~single-init recovery (so K>1 improvement is
    attributable to the restarts, not to any code change).

ANTI-CHEATS (each seed; NONE weakened to force a GO):
  (#0) GENUINELY SPIKING + LIKE-FOR-LIKE: the read is argmax over the OUTPUT-LIF SUMMED SPIKE COUNT (the D._score path);
       a no-output-spike lesion -> chance. Baseline = the FIXED SPIKING WTA (~0.5), never a host ridge argmax.
  (#1) DALE-LEGAL: every deployed detector's weights sign-constrained (W_e>=0, W_fi>=0, W_io<=0); asserted per seed.
  (#2) EMERGENT / NOT WARM-STARTED: the deployed restart is LEARNED from a random Dale-init by the reward-modulated rule;
       PRE-learning (K=0) objrel-slot0 ~chance on the counted seeds and RISES via learning + restart selection.
  (#3) REWARD LOAD-BEARING: no-reward (DA==0) / shuffled-reward -> objrel does NOT recover (per restart; the critic then
       has nothing to select). SALIENCE kept ON (load-bearing per the verified finding).
  (#4) HELD-OUT: TEST facts from a DISTINCT rng (0 train/test objrel overlap; the subject pool is SHARED across THEME-
       slot0 and AGENT-slot1 so per-word memorization is impossible). SCRAMBLE (deranged role targets) -> chance.
  (#crux) SELECTION HONESTY: the deployed restart is chosen by the TRAINING reward critic ONLY. The test-oracle column is
       reported for comparison, NEVER deployed.

GO iff: K-restart with reward-critic selection recovers objrel-slot0 on the GENUINELY-EMERGENT counted seeds at a
robustly higher rate than K=1 (target: >= 9/10 or 10/10 genuinely-emergent, ALL-blind, MONOTONE-improving in K), AND
train-reward-selection ~= test-oracle (the critic is valid), AND all anti-cheats hold. Else HONEST BOUNDARY with the
numbers (e.g. the reward critic cannot distinguish basins -> the miss persists; or a miss-rate floor). A clean BOUNDARY
is a valid result; report the K-sweep recovery curve.

Reuse-by-import (NO sim/ edit; CPU/numpy): DP (_rungB1c_objrel_dopamine_plasticity_derisk: BinaryRoleDetector,
DopaminePlasticReadout, EPOCHS/LR/SALIENCE_POWER, _train_dopamine), D (_rungB1c_objrel_dann_readout_derisk: _score,
_cache_slot_features, N_ROLES3, H_INH, N_TRAIN, N_TEST), PR (_build, _c2_single_wta_baseline, _feature), C (the c2
bridge/reservoir), the _emerge78 corpus/encoder scaffold. The restart-selection is a THIN wrapper over the VERIFIED
BinaryRoleDetector.fit_reward (no change to the detector or the update rule).

Run:
  SIM_BACKEND=numpy python -u -m research.runners._rungB1c_objrel_restart_basin_escape_derisk \
      --seeds 42 43 44 45 46 100 101 102 103 104 --ks 1 3 5 8 \
      --json research/findings/raw/_rungB1c_objrel_restart_basin_escape.json \
      2>&1 | tee research/findings/raw/_rungB1c_objrel_restart_basin_escape.log
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np  # noqa: E402

import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C  # noqa: E402
import research.runners._rungB1c_objrel_per_role_readout_derisk as PR  # noqa: E402
import research.runners._rungB1c_objrel_dann_readout_derisk as D  # noqa: E402
import research.runners._rungB1c_objrel_dopamine_plasticity_derisk as DP  # noqa: E402
from research.runners._emerge78_reservoir_form_to_role_derisk import (  # noqa: E402
    Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX, _ROLES,
)

BinaryRoleDetector = DP.BinaryRoleDetector
N_ROLES3 = D.N_ROLES3
N_TRAIN = D.N_TRAIN
N_TEST = D.N_TEST
H_INH = DP.H_INH
READ_T = DP.READ_T
EPOCHS = DP.EPOCHS
LR = DP.LR
SALIENCE_POWER = DP.SALIENCE_POWER

# PRE-REGISTERED split (declared here, BEFORE the run; no post-hoc partition selection)
DEV = [42, 43, 44, 45, 46]
BLIND = [100, 101, 102, 103, 104]
GENUINE_PRE_THRESH = 0.85       # a seed is COUNTED only if its K=0 pre-learning objrel-slot0 < this (exclude init-lucky)


def _critic_score(det, X, is_pos, salience_pos):
    """The REWARD CRITIC = the accumulated TRAINING reward for a trained restart = the salience-weighted training-signal
    FIT (the SAME objective the graded-DA delta rule ascends: DA = salience*(target - act), a reward-weighted regression).
    Uses ONLY the TRAINING sentences' reward -- the test objrel is NEVER seen. Returns the NEGATIVE salience-weighted
    training squared error -sum_i sw[i]*(tgt[i] - act_i)^2 (higher = better fit = higher accumulated reward). A basin-
    missed restart fits the majority + misses the minority positives -> large salience-weighted error -> LOW score ->
    rejected; a basin-escaped restart fits the minority -> HIGH score -> selected."""
    act = X @ det.w + det.b                                  # the graded post-synaptic activation on TRAIN (no test read)
    tgt = np.where(is_pos.astype(bool), 1.0, -1.0)
    sw = np.where(is_pos.astype(bool), salience_pos, 1.0)    # salience: the rare positive counts heavier (kept ON)
    return float(-np.sum(sw * (tgt - act) ** 2))


class RestartRoleDetector:
    """A per-ROLE detector that trains K independent RESTARTS of the VERIFIED BinaryRoleDetector (each from a DISTINCT
    random Dale-init -> a distinct init BASIN) and SELECTS the deployed restart by the REWARD CRITIC = the accumulated
    TRAINING reward (salience-weighted training-signal fit). The selected restart's Dale-legal spiking detector is
    deployed (read = output-LIF spike-count argmax). ALSO records the test-oracle selection (best restart by held-out
    objrel) for the honesty comparison -- NEVER deployed."""

    def __init__(self, feat_dim, k, role_r, base_seed, h_inh=H_INH):
        self.feat_dim = int(feat_dim)
        self.k = int(k)
        self.role_r = int(role_r)
        self.base_seed = int(base_seed)
        self.h_inh = int(h_inh)
        self.restarts = []                  # list of trained BinaryRoleDetector
        self.critic_scores = []             # per-restart training-reward critic score
        self.selected = None                # the reward-critic-selected detector (DEPLOYED)
        self.selected_idx = None

    def fit(self, X, is_pos, salience_pos, epochs=EPOCHS, lr=LR, reward_on=True):
        """Train K restarts; select the deployed one by the training-reward critic. Each restart uses a DISTINCT restart-
        seed feeding BinaryRoleDetector.__init__ (which sets both the learned direction w and the Dale-init W_e/W_fi/W_io
        rng) AND fit_reward's example-permutation rng -> a genuinely different init basin + training trajectory. The
        update rule + the salience are UNCHANGED from the verified BinaryRoleDetector.fit_reward."""
        self.restarts = []
        self.critic_scores = []
        for j in range(self.k):
            restart_seed = self.base_seed * 100003 + self.role_r * 9973 + j * 101 + 7   # distinct per (seed, role, restart)
            det = BinaryRoleDetector(self.feat_dim, h_inh=self.h_inh, seed=restart_seed)
            det.fit_reward(X, is_pos, seed=restart_seed, epochs=epochs, lr=lr,
                           salience_pos=salience_pos, reward_on=reward_on)
            self.restarts.append(det)
            self.critic_scores.append(_critic_score(det, X, is_pos, salience_pos))
        self.selected_idx = int(np.argmax(self.critic_scores))       # REWARD CRITIC selects (training reward only)
        self.selected = self.restarts[self.selected_idx]
        return self

    # ── deploy interface (delegates to the reward-critic-SELECTED restart; the D._score path) ──
    def _forward1(self, f, silence_inh=False, no_spike_lesion=False):
        return self.selected._forward1(f, silence_inh=silence_inh, no_spike_lesion=no_spike_lesion)

    def dale_legal(self):
        return self.selected.dale_legal()


class RestartPlasticReadout:
    """A per-SLOT read-out = N_ROLES3 INDEPENDENT RestartRoleDetectors (the see-saw killer, unchanged from the verified
    DopaminePlasticReadout). Each role's detector trains K restarts + reward-critic-selects. The per-slot decision
    argmaxes over the SELECTED restarts' genuinely-spiking OUTPUT SPIKE COUNTS. Exposes the D._score interface
    (predict_spikes + dale_legal). Also exposes an ORACLE variant that picks the test-best restart (reported only)."""

    def __init__(self, feat_dim, k, seed, h_inh=H_INH):
        self.feat_dim = int(feat_dim)
        self.k = int(k)
        self.h_inh = int(h_inh)
        self.det = [RestartRoleDetector(feat_dim, k, r, seed, h_inh=h_inh) for r in range(N_ROLES3)]

    def fit_reward(self, X, y, epochs=EPOCHS, lr=LR, seed=0, salience=True, reward_on=True, shuffle_reward=False):
        rng = np.random.default_rng(seed * 251 + 17)
        y_for_reward = rng.permutation(y) if shuffle_reward else y.copy()
        cnt = np.bincount(y, minlength=N_ROLES3).astype(np.float64)
        cnt[cnt == 0] = 1.0
        inv_freq = (cnt.sum() / (N_ROLES3 * cnt))
        for r in range(N_ROLES3):
            is_pos = (y_for_reward == r)
            sal_pos = float(inv_freq[r] ** SALIENCE_POWER) if salience else 1.0
            self.det[r].fit(X, is_pos, sal_pos, epochs=epochs, lr=lr, reward_on=reward_on)
        return self

    def predict_spikes(self, f, silence_inh=False, no_spike_lesion=False):
        counts = np.zeros(N_ROLES3, dtype=np.float64)
        inh_total = 0.0
        for r in range(N_ROLES3):
            st = self.det[r]._forward1(f, silence_inh=silence_inh, no_spike_lesion=no_spike_lesion)
            counts[r] = st["count"]
            inh_total += st["inh"]
        return int(np.argmax(counts)), counts, inh_total / N_ROLES3

    def dale_legal(self):
        dd = [d.dale_legal() for d in self.det]
        return {
            "W_e_min": float(min(x["W_e_min"] for x in dd)),
            "W_fi_min": float(min(x["W_fi_min"] for x in dd)),
            "W_io_max": float(max(x["W_io_max"] for x in dd)),
            "legal": bool(all(x["legal"] for x in dd)),
        }

    def select_by_oracle(self, res, enc, objr_test):
        """(HONESTY column, NOT deployed via the critic) For each role, pick the restart whose DEPLOYED detector maximizes
        the held-out objrel-slot0 accuracy -- i.e. what K-restart WOULD reach if it could cheat by looking at the test.
        Returns a NEW RestartPlasticReadout-like readout wrapping the test-best restarts. Compare its objrel-slot0 to the
        training-reward-selected read: ~= => the training reward is a valid critic; >> => the critic can't tell basins
        apart (an honest negative)."""
        oracle = _OracleWrap(self.feat_dim, self.h_inh)
        # For each role, evaluate every restart's slot0 objrel accuracy (role 0 is the THEME-carrying slot0 detector; the
        # per-slot decision at slot0 argmaxes over roles, so we approximate the oracle by the SLOT-LEVEL best combination:
        # pick, per role, the restart that -- HOLDING the other roles at their training-reward-selected restart -- gives
        # the highest slot0 objrel accuracy. This is a generous oracle upper bound.)
        base = [d.selected for d in self.det]                        # training-reward-selected per role
        chosen = list(base)
        for r in range(N_ROLES3):
            best_acc = -1.0
            best_det = base[r]
            for cand in self.det[r].restarts:
                trial = list(chosen)
                trial[r] = cand
                acc = _slot0_objrel_acc(trial, res, enc, objr_test)
                if acc > best_acc:
                    best_acc = acc
                    best_det = cand
            chosen[r] = best_det
        oracle.dets = chosen
        return oracle


class _OracleWrap:
    """A thin read-out that deploys a fixed list of per-role BinaryRoleDetectors (the test-oracle picks). Reported ONLY;
    never selected by the critic / never the GO deploy path."""

    def __init__(self, feat_dim, h_inh):
        self.feat_dim = int(feat_dim)
        self.h_inh = int(h_inh)
        self.dets = []

    def predict_spikes(self, f, silence_inh=False, no_spike_lesion=False):
        counts = np.zeros(N_ROLES3, dtype=np.float64)
        inh_total = 0.0
        for r in range(N_ROLES3):
            st = self.dets[r]._forward1(f, silence_inh=silence_inh, no_spike_lesion=no_spike_lesion)
            counts[r] = st["count"]
            inh_total += st["inh"]
        return int(np.argmax(counts)), counts, inh_total / N_ROLES3

    def dale_legal(self):
        dd = [d.dale_legal() for d in self.dets]
        return {"W_e_min": float(min(x["W_e_min"] for x in dd)), "W_fi_min": float(min(x["W_fi_min"] for x in dd)),
                "W_io_max": float(max(x["W_io_max"] for x in dd)), "legal": bool(all(x["legal"] for x in dd))}


def _slot0_objrel_acc(dets, res, enc, objr):
    """Slot-0 objrel accuracy of a per-role detector list (argmax over the roles' output spike counts at slot 0)."""
    ok = tot = 0
    for toks, roles in objr:
        f = PR._feature(res, enc, toks)
        pos0 = sorted(roles)[0]
        tgt = _ROLE_IDX[roles[pos0]]
        if tgt >= N_ROLES3:
            continue
        counts = np.array([d._forward1(f)["count"] for d in dets], dtype=np.float64)
        ok += int(int(np.argmax(counts)) == tgt)
        tot += 1
    return ok / max(tot, 1)


def _train_restart(slot_train, feat_dim, seed, k, epochs=EPOCHS, salience=True, reward_on=True, shuffle_reward=False,
                   scramble=False):
    """Train one RestartPlasticReadout per slot (K restarts + reward-critic selection). `scramble` deranges the role
    targets at fit time; `k=0` = the PRE-learning random Dale-init read (no restarts trained -> deploy a raw random init;
    should be ~chance on the counted seeds). Returns {slot k_slot: readout}."""
    perm = None
    if scramble:
        srng = np.random.default_rng(seed * 977 + 13)
        perm = srng.permutation(3)
        while np.array_equal(perm, [0, 1, 2]):
            perm = srng.permutation(3)
    ros = {}
    for kslot, (X, y) in slot_train.items():
        yk = np.array([perm[v] for v in y], dtype=y.dtype) if perm is not None else y
        if k <= 0:
            # PRE-learning: a raw random Dale-init read (reuse the verified DopaminePlasticReadout with 0 epochs).
            ro = DP.DopaminePlasticReadout(feat_dim, seed=seed * 100 + kslot)   # random init, no fit -> ~chance
        else:
            ro = RestartPlasticReadout(feat_dim, k, seed=seed * 100 + kslot)
            ro.fit_reward(X, yk, epochs=epochs, seed=seed * 100 + kslot, salience=salience,
                          reward_on=reward_on, shuffle_reward=shuffle_reward)
        ros[kslot] = ro
    return ros


def _oracle_readouts(ros_main, res, enc, objr):
    """Build the per-slot test-ORACLE read-outs from the trained restart read-outs (reported honesty column)."""
    out = {}
    for kslot, ro in ros_main.items():
        out[kslot] = ro.select_by_oracle(res, enc, objr)
    return out


def run_seed_ksweep(seed, corpus, ks):
    """Build the byte-identical c2 reservoir (FROZEN) ONCE, cache the spiking feature ONCE, reproduce the FIXED SPIKING
    WTA baseline, compute the K=0 PRE-learning read (the genuinely-emergent gate), then for EACH K in `ks` train the
    restart read + the anti-cheats + the test-oracle column. Returns the per-seed row dict (with a per-K sub-table)."""
    t0 = time.time()
    C.WS_BIAS_SCALE_C2 = 0.0
    C.WS_REPLAY = PR.WS_REPLAY
    C.READ_T_STEP_C2 = PR.READ_T_STEP
    subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
    enc = Encoder(corpus["discovered"])
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
    trng = np.random.default_rng(seed * 977 + 13)          # DISTINCT rng => test facts held out (no leakage)
    canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
    objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)

    ub, ens, inh, res, res_idx = PR._build(seed, corpus, enc)
    print(f"[restart seed {seed}] caching spiking reservoir features on {len(train)} train sentences "
          f"(reservoir slice {res_idx[0]}..{res_idx[-1]})...", flush=True)
    slot_train = D._cache_slot_features(res, enc, train)
    feat_dim = next(iter(slot_train.values()))[0].shape[1]
    slot0_counts = np.bincount(slot_train[0][1], minlength=N_ROLES3).tolist() if 0 in slot_train else []

    # ── BASELINE: the FIXED SPIKING WTA (like-for-like comparator, #0) ──
    base_canon, base_c_s0, base_objr, base_o_s0 = PR._c2_single_wta_baseline(
        ub, ens, res, enc, res_idx, train, canon, objr)

    # ── (#2) PRE-LEARNING (K=0 random Dale-init) -> the GENUINELY-EMERGENT gate + the emergent control ──
    ros0 = _train_restart(slot_train, feat_dim, seed, k=0)
    zc_acc, zc_s0, _zcp, _zct, _zcspk, _zcih = D._score(ros0, res, enc, canon)
    zo_acc, zo_s0, _zop, _zot, _zospk, _zoih = D._score(ros0, res, enc, objr)
    genuinely_emergent = bool(zo_s0 < GENUINE_PRE_THRESH)     # counted only if pre-learning objrel-slot0 < 0.85

    # ── ANALYTIC Dale reference (reported; the target exists in weight space) ──
    ros_an = D._analytic_dale_readout(slot_train, feat_dim, seed)
    an_canon_acc, an_canon_s0, *_ = D._score(ros_an, res, enc, canon)
    an_objr_acc, an_objr_s0, *_ = D._score(ros_an, res, enc, objr)

    per_k = {}
    for k in ks:
        tk = time.time()
        print(f"[restart seed {seed}] K={k}: training {k} restart(s)/role, reward-critic selection...", flush=True)
        ros = _train_restart(slot_train, feat_dim, seed, k=k, salience=True, reward_on=True)
        canon_acc, canon_s0, canon_ps, canon_pt, canon_spk, canon_inh = D._score(ros, res, enc, canon)
        objr_acc, objr_s0, objr_ps, objr_pt, objr_spk, objr_inh = D._score(ros, res, enc, objr)

        # test-ORACLE column (reported only; what K-restart WOULD reach if it could cheat by selecting on the test)
        ros_oracle = _oracle_readouts(ros, res, enc, objr)
        or_acc, or_s0, _orp, _ort, _orspk, _orih = D._score(ros_oracle, res, enc, objr)

        # Dale-legal + no-spike lesion on the DEPLOYED (critic-selected) read
        dale = [ro.dale_legal() for ro in ros.values()]
        dale_legal_all = all(dd["legal"] for dd in dale)
        les_acc, les_s0, _lps, _lpt, les_spk, _lih = D._score(ros, res, enc, objr, no_spike_lesion=True)

        # reward-load-bearing controls (per K; the restart selection has nothing valid to pick if reward is off/shuffled)
        ros_nr = _train_restart(slot_train, feat_dim, seed, k=k, salience=True, reward_on=False)
        nr_acc, nr_s0, *_ = D._score(ros_nr, res, enc, objr)
        ros_sr = _train_restart(slot_train, feat_dim, seed, k=k, salience=True, reward_on=True, shuffle_reward=True)
        sr_acc, sr_s0, *_ = D._score(ros_sr, res, enc, objr)

        # scramble control (per K)
        ros_scr = _train_restart(slot_train, feat_dim, seed, k=k, salience=True, reward_on=True, scramble=True)
        scr_acc, scr_s0, *_ = D._score(ros_scr, res, enc, objr)

        # collect per-role selected restart indices + critic scores (transparency)
        sel_idx = {r: [int(ros[kslot].det[r].selected_idx) for kslot in sorted(ros)] for r in range(N_ROLES3)}

        per_k[str(k)] = {
            "objrel_slot0_THEME": round(objr_s0, 3), "objrel_acc": round(objr_acc, 3),
            "objrel_per_slot": [f"{h}/{t}" for h, t in zip(objr_ps, objr_pt)],
            "canonical_acc": round(canon_acc, 3), "canonical_slot0": round(canon_s0, 3),
            "mean_out_spikes_canon": round(canon_spk, 3), "mean_out_spikes_objr": round(objr_spk, 3),
            "test_oracle_objrel_slot0": round(or_s0, 3), "test_oracle_objrel_acc": round(or_acc, 3),
            "no_reward_objrel_slot0": round(nr_s0, 3), "shuffled_reward_objrel_slot0": round(sr_s0, 3),
            "scramble_objrel_slot0": round(scr_s0, 3),
            "no_spike_objrel_slot0": round(les_s0, 3), "no_spike_out_spikes": round(les_spk, 3),
            "dale_legal": bool(dale_legal_all),
            "selected_restart_idx": sel_idx,
            # per-K per-seed anti-cheat flags
            "genuinely_spiking": bool(objr_spk > 0.0 and canon_spk > 0.0),
            "no_spike_collapses": bool(les_s0 <= 0.50),
            "reward_load_bearing": bool(objr_s0 - max(nr_s0, sr_s0) >= 0.15),
            "canonical_not_regressed": bool(canon_acc >= 0.90),
            "scramble_chance": bool(scr_s0 <= 0.50),
            "objrel_recovers": bool(objr_s0 >= 0.85),
            "train_reward_matches_oracle": bool(abs(objr_s0 - or_s0) <= 0.15),
            "elapsed_s": round(time.time() - tk, 1),
        }
        print(f"[restart seed {seed}] K={k}: objrel-slot0 {objr_s0:.2f} (per-slot {per_k[str(k)]['objrel_per_slot']}) "
              f"canon {canon_acc:.2f} | test-ORACLE {or_s0:.2f} | no-reward {nr_s0:.2f} shuf {sr_s0:.2f} | "
              f"scramble {scr_s0:.2f} | no-spike {les_s0:.2f} | dale {dale_legal_all} "
              f"[recov {per_k[str(k)]['objrel_recovers']} critic~oracle {per_k[str(k)]['train_reward_matches_oracle']}] "
              f"({per_k[str(k)]['elapsed_s']}s)", flush=True)

    d = {
        "seed": int(seed), "h_inh": H_INH, "read_t": READ_T, "epochs": EPOCHS, "lr": LR,
        "salience_power": SALIENCE_POWER, "slot0_class_counts": slot0_counts,
        "baseline_fixed_spiking_wta": {
            "canonical_acc": round(base_canon, 3), "objrel_slot0_THEME": round(base_o_s0, 3)},
        "pre_learning_k0": {
            "objrel_slot0_THEME": round(zo_s0, 3), "objrel_acc": round(zo_acc, 3),
            "canonical_acc": round(zc_acc, 3), "canonical_slot0": round(zc_s0, 3)},
        "genuinely_emergent": genuinely_emergent,       # counted in the GO tally iff True (pre < 0.85)
        "analytic_dale_reference": {
            "canonical_acc": round(an_canon_acc, 3), "objrel_slot0_THEME": round(an_objr_s0, 3)},
        "per_k": per_k,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return d


def _agg_ksweep(rows, ks):
    """Aggregate the K-sweep recovery over the genuinely-emergent counted seeds (dev + blind reported separately)."""
    counted = [r for r in rows if r["genuinely_emergent"]]
    counted_blind = [r for r in counted if r["seed"] in BLIND]
    excluded = [r["seed"] for r in rows if not r["genuinely_emergent"]]

    recovery = {}       # K -> {counted recovery over counted seeds, all-blind recovered?, per-seed}
    for k in ks:
        ks_ = str(k)
        n_rec = sum(r["per_k"][ks_]["objrel_recovers"] for r in counted)
        n_rec_blind = sum(r["per_k"][ks_]["objrel_recovers"] for r in counted_blind)
        # anti-cheats over ALL seeds at this K
        spk = all(r["per_k"][ks_]["genuinely_spiking"] for r in rows)
        nospk = all(r["per_k"][ks_]["no_spike_collapses"] for r in rows)
        dale = all(r["per_k"][ks_]["dale_legal"] for r in rows)
        rlb = all(r["per_k"][ks_]["reward_load_bearing"] for r in counted)   # reward LB where learning matters
        canon = all(r["per_k"][ks_]["canonical_not_regressed"] for r in rows)
        scr = all(r["per_k"][ks_]["scramble_chance"] for r in rows)
        critic_ok = all(r["per_k"][ks_]["train_reward_matches_oracle"] for r in counted)
        mean_objr = float(np.mean([r["per_k"][ks_]["objrel_slot0_THEME"] for r in counted])) if counted else 0.0
        mean_oracle = float(np.mean([r["per_k"][ks_]["test_oracle_objrel_slot0"] for r in counted])) if counted else 0.0
        recovery[ks_] = {
            "n_counted": len(counted), "n_recovered": int(n_rec),
            "n_counted_blind": len(counted_blind), "n_recovered_blind": int(n_rec_blind),
            "all_blind_recovered": bool(n_rec_blind == len(counted_blind) and len(counted_blind) > 0),
            "mean_objrel_slot0": round(mean_objr, 3), "mean_test_oracle_objrel_slot0": round(mean_oracle, 3),
            "genuinely_spiking_all": bool(spk), "no_spike_collapses_all": bool(nospk), "dale_legal_all": bool(dale),
            "reward_load_bearing_counted": bool(rlb), "canonical_not_regressed_all": bool(canon),
            "scramble_chance_all": bool(scr), "train_reward_matches_oracle_counted": bool(critic_ok),
            "per_seed_recovered": {str(r["seed"]): bool(r["per_k"][ks_]["objrel_recovers"]) for r in rows},
        }
    # monotone-improving in K over counted recovery?
    rec_curve = [recovery[str(k)]["n_recovered"] for k in ks]
    monotone = all(rec_curve[i] <= rec_curve[i + 1] for i in range(len(rec_curve) - 1))
    kmax = str(ks[-1])
    n_ct = recovery[kmax]["n_counted"]
    return {
        "counted_seeds": [r["seed"] for r in counted],
        "excluded_init_lucky_seeds": excluded,
        "recovery_by_k": recovery,
        "recovery_curve_counted": rec_curve,
        "monotone_in_k": bool(monotone),
        "k1_baseline_recovered": recovery[str(ks[0])]["n_recovered"],
        "kmax": int(ks[-1]),
        "kmax_recovered": recovery[kmax]["n_recovered"],
        "kmax_all_blind_recovered": recovery[kmax]["all_blind_recovered"],
        "kmax_counted": n_ct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEV + BLIND)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 8])
    ap.add_argument("--json", type=str, default="research/findings/raw/_rungB1c_objrel_restart_basin_escape.json")
    args = ap.parse_args()
    ks = sorted(set(args.ks))

    t0 = time.time()
    corpus = C.setup_corpus(seed=42)
    print(f"[restart] BASIN-ESCAPE via K random restarts + REWARD-CRITIC selection. PRE-REGISTERED split "
          f"DEV={DEV} BLIND={BLIND}; K-sweep {ks}; genuinely-emergent gate pre(K=0)<{GENUINE_PRE_THRESH}. "
          f"Selection = TRAINING-reward critic (salience-weighted training-signal fit) ONLY; the test-oracle column is "
          f"reported, NEVER deployed. Salience KEPT ON. NO sim/ edit; CPU/numpy.", flush=True)

    rows = []
    for s in args.seeds:
        tag = "DEV" if s in DEV else "BLIND"
        d = run_seed_ksweep(s, corpus, ks)
        rows.append(d)
        print(f"[seed {s} {tag}] pre-learn(K=0) objrel-slot0 {d['pre_learning_k0']['objrel_slot0_THEME']:.2f} "
              f"(genuinely-emergent {d['genuinely_emergent']}) | base-WTA {d['baseline_fixed_spiking_wta']['objrel_slot0_THEME']:.2f} "
              f"| analytic {d['analytic_dale_reference']['objrel_slot0_THEME']:.2f} | "
              + " ".join(f"K{k}={d['per_k'][str(k)]['objrel_slot0_THEME']:.2f}" for k in ks)
              + f"  ({d['elapsed_s']}s)", flush=True)

    agg = _agg_ksweep(rows, ks)
    kmax = str(ks[-1])
    rk = agg["recovery_by_k"][kmax]

    # GO condition (honest either way) -- monotone in K, kmax recovers >= 9/10 genuinely-emergent all-blind, critic valid.
    go = bool(
        agg["monotone_in_k"]
        and rk["n_recovered"] >= max(1, int(np.ceil(0.9 * rk["n_counted"])))
        and rk["all_blind_recovered"]
        and rk["n_recovered"] > agg["k1_baseline_recovered"]      # restarts must IMPROVE over the single-init baseline
        and rk["train_reward_matches_oracle_counted"]
        and rk["genuinely_spiking_all"] and rk["no_spike_collapses_all"] and rk["dale_legal_all"]
        and rk["reward_load_bearing_counted"] and rk["canonical_not_regressed_all"] and rk["scramble_chance_all"]
    )

    print(f"\n[restart] K-sweep recovery (genuinely-emergent counted={agg['counted_seeds']}, "
          f"excluded-init-lucky={agg['excluded_init_lucky_seeds']}):", flush=True)
    for k in ks:
        r = agg["recovery_by_k"][str(k)]
        print(f"  K={k}: recovered {r['n_recovered']}/{r['n_counted']} counted "
              f"({r['n_recovered_blind']}/{r['n_counted_blind']} blind, all-blind {r['all_blind_recovered']}) | "
              f"mean objrel-slot0 {r['mean_objrel_slot0']:.2f} (test-oracle {r['mean_test_oracle_objrel_slot0']:.2f}, "
              f"critic~oracle {r['train_reward_matches_oracle_counted']}) | spiking {r['genuinely_spiking_all']} "
              f"dale {r['dale_legal_all']} reward-LB {r['reward_load_bearing_counted']} canon {r['canonical_not_regressed_all']} "
              f"scramble {r['scramble_chance_all']} no-spk {r['no_spike_collapses_all']}", flush=True)
    print(f"[restart] recovery-vs-K curve (counted): {agg['recovery_curve_counted']} monotone={agg['monotone_in_k']} "
          f"| K1-baseline {agg['k1_baseline_recovered']} -> K{ks[-1]} {agg['kmax_recovered']}/{agg['kmax_counted']} "
          f"| VERDICT {'GO' if go else 'BOUNDARY'}", flush=True)

    out = {"rows": rows, "agg": agg, "verdict": "GO" if go else "BOUNDARY",
           "pre_registered": {"DEV": DEV, "BLIND": BLIND, "ks": ks, "genuine_pre_thresh": GENUINE_PRE_THRESH},
           "total_elapsed_s": round(time.time() - t0, 1)}
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"[restart] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
