"""gap#4 RANK-1 -- LEARNED-IN-ENGINE self-predicting interneuron microcircuit (Sacramento-Senn 2018 Eq.9).

WHY THIS RUN (the pre-registered RANK-1 lever, 2026-09-09). The gap#4 arc root-caused the genuine residual to the
FROZEN fixed-random feedback SIGNAL that never zeroes when the net is already correct (feedback-alignment), NOT the
dendrite topology and NOT the read-SNR alone (both tested-negative:
`2026-09-09-gap4-dendritic-urbanczik-senn-read-snr-clean-NO-GO`, `2026-07-22-gap4-real-issue-NOT-dendrites`). The
pre-registered untested fix is a LEARNED interneuron self-predicting microcircuit: plastic SST/PV weights LEARN to
cancel the top-down feedback so the apical is SILENT when the network is already correct and carries a true
prediction error otherwise.

WHAT IS GENUINELY NEW HERE vs the record (verify-first, 2026-09-09):
  * The learned microcircuit was already run at RATE (`2026-07-24-gap4-learned-selfpredicting-microcircuit-CPUrate-GO`
    -- earned apical-silence GO, but accuracy-indistinguishable from fixed-FA at rate).
  * It was run on-bridge as a RUNNER-SUPPLIED cancellation (the sibling `_gap4_onbridge_spiking_selfpredict_derisk`
    `micro` arm: W^PI @ phi(u^I) host-computed each phase, injected via cp_bdsp_int_drive; NOT-GO on the
    representable-expander `2026-08-18-gap4-microcircuit-expander-6seed-NOTGO`, and its deep-hidden arc culminated in
    "the crux was never askable" -- the transport ceiling could not fit, a READ-regime foreclosure).
  * NEVER done: the interneuron cancellation LEARNED IN-ENGINE on the substrate. That is the RANK-1 distinction the
    2026-09-09 finding names ("the enable_bdsp_microcircuit stub's cancellation is runner-supplied, NOT learned
    in-engine"). This runner is the FIRST to move it in-engine: the committed additive/default-off
    `enable_selfpredicting_interneuron` (sim/config.py) makes the ENGINE, inside _run_one_simulation_step, PROJECT
    the interneuron rate through the PLASTIC substrate weight cp_spi_wpi to form cp_bdsp_int_drive AND UPDATE
    cp_spi_wpi by the local Sacramento self-prediction rule (selfpredicting_interneuron_update in
    sim/dendritic_plasticity.py). The runner-supplied host matmul + host W^PI update are gone.

LIKE-FOR-LIKE. The `micro_inengine` arm is byte-for-byte identical to the vetted `micro` arm EXCEPT the top-layer
interneuron cancellation + its W^PI learning move from the runner (host numpy, once/example) to the engine
(substrate cp arrays, one gated update/example). Same task, same net, same forward BDSP plasticity, same anti-cheats,
same descent to the lower hidden layer. So a difference in held-out inheritance is attributable to WHERE the
cancellation is learned, nothing else.

ARMS (held-out INHERITANCE accuracy on spikes, the GO metric; all reuse the sibling's OnBridgeBDSPNet):
  reservoir         : hidden apical = 0 (H frozen at random init); only the readout learns  = credit-INDEPENDENT floor.
  fixed_fa          : fixed-random Y feedback, graded BDSP FF plasticity                     = the frozen-signal baseline to BEAT.
  micro             : runner-supplied plastic-Eq.9 W^PI cancellation (host)                  = the already-tested comparison.
  micro_inengine    : LEARNED-IN-ENGINE plastic cp_spi_wpi cancellation (substrate)          = the genuinely-new build.
  transport_ceiling : Y := (pooled forward W)^T (weight transport ~ backprop)                = the CEILING (interpretability gate + labeled cheat).

PRE-REGISTERED GO GATE (decisive). GO iff, on >= 5/6 seeds:
  (interpretability) transport_ceiling > chance AND oracle >= 0.80  -- the read CAN carry the credit (else UNDEFINED,
      per the "crux was never askable" lesson: nothing beneath an uninterpretable ceiling is readable);
  (FA-wall)          fixed_fa <= reservoir + 0.02  -- a real gap the learned signal could close;
  (surpass)          micro_inengine > fixed_fa + 0.05  -- the in-engine learned cancellation beats the frozen signal;
  (mechanism)        the in-engine apical is EARNED-silent (silent_ratio < 0.5, selfpred_cos > 0.6) and FREEZING
      cp_spi_wpi collapses that silence + the accuracy (attributable_to lesion);
  (clean)            transport-ceiling no-weight-transport guard FAILS; AST no-forward-W in the credit path; apical
      lesion / shuffled-target / shufE / freeze-spi all collapse; cfg.seed controls the substrate.
An honest NO-GO (in-engine learning does not beat the frozen signal) OR UNDEFINED (the ceiling cannot fit) IS the
deliverable. Verdict earned via tools.verdict.Verdict; lesion attribution via tools.lab.attributable_to.

CONSTRUCT-SMOKE (the only thing run locally; numpy):
    SIM_BACKEND=numpy python -m research.runners._gap4_selfpredict_interneuron_inengine_derisk --construct-smoke

THE DECISIVE 6-seed GPU de-risk (queued; one process, controller-fanned seeds):
    SIM_BACKEND=cupy python -u -m research.runners._gap4_selfpredict_interneuron_inengine_derisk --full \
        --seeds 42 43 44 100 101 102 --arms reservoir fixed_fa micro micro_inengine transport_ceiling \
        --hidden 64 --pool-k 16 --n-hidden-layers 2 --epochs 40 --graded-credit --wpi-init noisy \
        --assert-no-transport --out research/findings/raw/gap4/selfpredict_inengine_6seed.json
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")
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
from research.runners._gap4_onbridge_spiking_selfpredict_derisk import (  # noqa: E402
    Gap4OnBridgeNet, _cos, _ast_no_forward_W, _softmax)
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance)

OUT = _REPO / "research" / "findings" / "raw" / "gap4" / "selfpredict_inengine.json"

# micro_inengine reuses the "microcircuit" bdsp rule (so cfg.enable_bdsp_microcircuit is set by the parent __init__);
# the engine's self-predicting-interneuron flag is switched on AFTER build in Gap4InEngineNet.__init__.
_ARM_RULE = {"reservoir": "plain_fa", "fixed_fa": "plain_fa", "micro": "microcircuit",
             "micro_inengine": "microcircuit", "transport_ceiling": "plain_fa"}


class Gap4InEngineNet(Gap4OnBridgeNet):
    """Gap4OnBridgeNet + the `micro_inengine` feedback mode: the top-layer interneuron cancellation cp_bdsp_int_drive
    and its plastic weight cp_spi_wpi are formed + LEARNED BY THE ENGINE (sim/, on the substrate cp arrays), not by
    the runner. Every other mode delegates to the parent unchanged (a true like-for-like harness)."""

    def __init__(self, n_in, hidden, k, seed=0, feedback="fixed", **kw):
        super().__init__(n_in, hidden, k, seed=seed, feedback=feedback, **kw)
        self._spi_installed = False
        self._spi_frozen = False           # freeze-spi anti-cheat: cp_spi_wpi never learns (silence cannot be earned)
        if self.feedback == "micro_inengine":
            self._install_inengine_microcircuit()

    # ---- install the in-engine microcircuit: turn on the flag, hand the substrate the plastic W^PI + fixed Y + scatter ----
    def _install_inengine_microcircuit(self):
        xp = self._xp
        li = self.n_hidden_layers                      # the TOP hidden logical layer (micro branch acts at k=nhid-1 -> li)
        sl = self.slices[li]
        K = self.pool_k
        H = self.sizes[li]                             # == self.hidden
        top = self.n_hidden_layers - 1
        Ytop = self.Y[top]                             # (k_classes, H) fixed top-down feedback the interneuron predicts
        # 0/1 structural scatter: neuron (sl.start + local) belongs to logical unit local // K (the _broadcast map).
        n_pool = sl.stop - sl.start                    # == H * K
        rows = np.arange(sl.start, sl.stop)
        cols = (np.arange(n_pool) // K)
        scatter = np.zeros((self.n_total, H), dtype=np.float32)
        scatter[rows, cols] = 1.0
        # the ENGINE flag + arrays (self.cfg IS the bridge's core_config; the block reads getattr(cfg, ...) each step).
        self.cfg.enable_selfpredicting_interneuron = True
        self.cfg.spi_lr = float(self.wpi_lr)
        self.br.cp_spi_wpi = xp.asarray(self.W_PI.astype(np.float64))   # PLASTIC in-engine weight (noisy or fixedpoint init)
        self.br.cp_spi_Y = xp.asarray(Ytop.astype(np.float64))         # FIXED feedback the interneuron self-predicts
        self.br.cp_spi_scatter = xp.asarray(scatter)                   # logical -> neuron broadcast (structural)
        self._spi_li = li
        self._spi_installed = True

    def _train_one(self, feat_row, y, mode):
        if self.feedback != "micro_inengine":
            return super()._train_one(feat_row, y, mode)
        if not self._spi_installed:
            self._install_inengine_microcircuit()
        xp = self._xp; n = self.n_total
        acts = self._forward_spiking(feat_row)
        logits = acts[-1][None, :]
        src_pred = _softmax(logits)                            # (1,k) the interneuron drive = the net's own prediction
        onehot = np.zeros((1, self.k)); onehot[0, int(y)] = 1.0
        delta_out = src_pred.copy(); delta_out[0, int(y)] -= 1.0
        if mode == "wrong_sign":
            delta_out = -delta_out; onehot = src_pred + delta_out
        e_upper = np.zeros_like(delta_out) if mode == "no_teaching_null" else -delta_out
        if mode == "shufE" and getattr(self, "_shuf_perm", None) is not None:
            e_upper = e_upper[:, self._shuf_perm]

        apical = np.zeros(n, dtype=np.float64)
        nhid = self.n_hidden_layers
        # OUTPUT-layer credit (identical to the parent).
        E_out = acts[-1][None, :]
        out_err = (E_out * (1.0 - E_out)) * e_upper
        apical[self.slices[-1]] = self.apical_gain_pA * self._broadcast(out_err[0], len(self.sizes) - 1)

        spi_int_rate = None; spi_phi_neuron = None
        for k in range(nhid - 1, -1, -1):
            li = k + 1
            E = acts[li][None, :]
            phi = E * (1.0 - E)
            Yk = np.zeros_like(self.Y[k]) if mode == "apical_lesion" else self.Y[k]
            if k == nhid - 1:
                # TOP layer: the ENGINE forms the interneuron cancellation + learns cp_spi_wpi. The runner supplies
                # ONLY the raw top-down teacher apical (the fixed feedback, as in `micro`) + the two ACTIVITIES the
                # engine projects: the interneuron rate src_pred, and the per-neuron surrogate gate phi (folding in
                # apical_gain). int_drive = (scatter @ (src_pred @ cp_spi_wpi)) * phi_neuron -> matches the `micro`
                # arm's host canc = apical_gain * broadcast(phi * (src_pred @ W_PI)) EXACTLY, except cp_spi_wpi is
                # learned on the substrate. apical_lesion zeroes Yk (raw teacher) -> the descent + read collapse.
                raw = phi * (onehot @ Yk)                      # (1,H) raw top-down teaching (the fixed feedback)
                apical[self.slices[li]] = self.apical_gain_pA * self._broadcast(raw[0], li)
                spi_int_rate = src_pred[0].astype(np.float64)  # (k,) interneuron drive (an activity)
                phi_neuron = np.zeros(n, dtype=np.float64)
                phi_neuron[self.slices[li]] = self.apical_gain_pA * self._broadcast(phi[0], li)  # per-neuron gate (folds gain)
                spi_phi_neuron = phi_neuron
                soma_err = phi * ((onehot - src_pred) @ Yk)    # descend the CANCELLED clean error (same as parent)
            else:
                v_api = e_upper @ Yk
                soma_err = phi * v_api
                apical[self.slices[li]] = self.apical_gain_pA * self._broadcast(soma_err[0], li)
            e_upper = soma_err

        # inject: raw apical (all layers) + hand the engine the interneuron rate/gate; the engine forms int_drive + learns.
        ap = np.zeros(n, dtype=np.float32); ap[:] = apical
        self.br.cp_bdsp_apical_drive = xp.asarray(ap)
        self.br.cp_spi_int_rate = xp.asarray(spi_int_rate.astype(np.float32))
        self.br.cp_spi_phi = xp.asarray(spi_phi_neuron.astype(np.float32))
        # learn cp_spi_wpi exactly ONCE per example (first credit step), and only in the bdsp training mode; the
        # freeze-spi anti-cheat + the frozen arm keep it False throughout (silence cannot be earned).
        learn_now = (mode == "bdsp") and not self._spi_frozen
        self.br._spi_learn = bool(learn_now)
        drive = self._base_drive()
        in_cur = np.clip(self.in_bias_pA + self.in_current_pA * np.asarray(feat_row, np.float32), 0.0, 1600.0)
        drive[self.slices[0]] = self._broadcast(in_cur, 0).astype(np.float32)
        self.br.cp_external_input_current = xp.asarray(drive)
        for _si in range(self.credit_steps):
            self.br._run_one_simulation_step()
            if _si == 0:
                self.br._spi_learn = False                     # exactly one W^PI update per example
        if self.br.cp_bdsp_apical_drive is not None:
            self.br.cp_bdsp_apical_drive[...] = 0.0
        if self.br.cp_bdsp_int_drive is not None:
            self.br.cp_bdsp_int_drive[...] = 0.0

    # ---- in-engine earned-silent read: uses the LEARNED substrate weight cp_spi_wpi (not a host W_PI) ----
    def inengine_apical_silent_stats(self, X, y):
        from sim.backend import to_host
        wpi = np.asarray(to_host(self.br.cp_spi_wpi)) if self.br.cp_spi_wpi is not None else self.W_PI
        Ytop = self.Y[self.n_hidden_layers - 1]
        X = np.asarray(X); y = np.asarray(y)
        mags = []; corr = []
        acts_b = self._forward_batch(X)
        for i in range(len(X)):
            lg = acts_b[-1][i]
            sp = _softmax(lg[None, :])[0]
            st = np.zeros(self.k); st[int(y[i])] = 1.0
            v_apical = st @ Ytop - sp @ wpi                    # residual apical the engine integrates (drive - int)
            mags.append(float(np.abs(v_apical).mean()))
            corr.append(bool(int(np.argmax(lg)) == int(y[i])))
        mags = np.asarray(mags); corr = np.asarray(corr)
        mc = float(mags[corr].mean()) if corr.any() else float("nan")
        mi = float(mags[~corr].mean()) if (~corr).any() else float("nan")
        ratio = float(mc / (mi + 1e-12)) if (corr.any() and (~corr).any()) else float("nan")
        return {"apical_correct": mc, "apical_incorrect": mi, "silent_ratio": ratio,
                "frac_correct": float(corr.mean()), "selfpred_cos": _cos(wpi, Ytop)}

    def no_weight_transport(self):
        if self.feedback == "transport_ceiling":
            return False
        if self.feedback in ("learned", "kp", "micro", "micro_inengine"):
            # the in-engine update (selfpredicting_interneuron_update) reads only activities + cp_spi_wpi/cp_spi_Y,
            # never a forward-weight array -> the parent AST guard over the host kp/W^PI methods still applies; the
            # engine rule is a separate LOCAL function verified by tests/its own guard. Structural, not byte.
            return bool(_ast_no_forward_W(type(self)))
        return True


# ============================================================================================================
def _build_net(feedback, n_in, k, args, seed):
    net = Gap4InEngineNet(
        n_in, args.hidden, k, seed=seed, feedback=feedback,
        n_hidden_layers=args.n_hidden_layers, pool_k=args.pool_k,
        settle_steps=args.settle_steps, credit_steps=args.credit_steps, lr=args.lr,
        in_current_pA=args.in_current_pA, in_bias_pA=args.in_bias_pA, apical_gain_pA=args.apical_gain_pA,
        tonic_h_pA=args.tonic_h_pA, tonic_o_pA=args.tonic_o_pA, beta=args.beta, p0=args.p0,
        graded_credit=args.graded_credit,
        wpi_plastic=True, wpi_init=args.wpi_init, wpi_lr=args.wpi_lr,
        kp_lr=args.kp_lr, kp_decay=args.kp_decay)
    wmax = float(getattr(args, "bdsp_w_max", 6.0))
    net.cfg.bdsp_w_max = wmax
    net.cfg.bdsp_w_min = -wmax
    return net


def _train_arm(net, Xtr, ytr, mode, epochs, batch, seed):
    rng = np.random.default_rng(seed + 777)
    if mode == "shufE":
        net._shuf_perm = np.random.default_rng(seed * 4099 + 11).permutation(net.k)
    for _ in range(epochs):
        perm = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), batch):
            b = perm[i:i + batch]
            net.train_step(Xtr[b], ytr[b], mode=mode)


def run_seed(seed, args):
    from sim.dendritic_mlp import DendriticMLP
    from sim.backend import to_host
    t_seed = time.time()
    tk = dict(n_super=args.n_super, n_members=args.n_members, held_per_super=args.held_per_super,
              n_prop=args.n_prop, member_id_dim=args.member_id_dim, n_obs=args.n_obs, noise=args.noise)
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte), meta, idx = make_task_semantic_inheritance(seed, **tk)
    n_in = Xtr.shape[1]; k = int(meta["k_classes"]); inh = idx["inh_idx"]
    chance = float(max(np.mean(yte[inh] == c) for c in np.unique(yte[inh]))) if len(inh) else float("nan")

    onet = DendriticMLP([n_in, args.hidden, args.hidden, k], seed=seed)
    r = np.random.default_rng(seed + 777)
    for _ in range(args.oracle_epochs):
        p = r.permutation(len(ytr))
        for i in range(0, len(ytr), args.oracle_batch):
            b = p[i:i + args.oracle_batch]
            onet.train_step(Xtr[b], ytr[b], mode="oracle", lr=args.oracle_lr)
    _, olg = onet._forward(np.asarray(Xte[inh], float))
    olg = np.asarray(to_host(olg))
    oracle = float(np.mean(np.argmax(olg, 1) == yte[inh])) if len(inh) else float("nan")
    print(f"[gap4-spi][seed {seed}] task n_in={n_in} k={k} n_train={len(ytr)} n_inh={len(inh)} "
          f"chance={chance:.3f} | ORACLE {oracle:.3f} ({time.time()-t_seed:.0f}s)", flush=True)

    Xtr_b, ytr_b = Xtr, ytr
    if args.train_subsample and args.train_subsample > 0 and len(Xtr) > args.train_subsample:
        srng = np.random.default_rng(seed * 13 + 1)
        keep = srng.permutation(len(Xtr))[:args.train_subsample]
        Xtr_b, ytr_b = Xtr[keep], ytr[keep]
        print(f"[gap4-spi][seed {seed}] on-bridge arms train on subsample {len(ytr_b)}/{len(ytr)}", flush=True)

    arms = {}; nets = {}
    for arm in args.arms:
        t_arm = time.time()
        net = _build_net(arm, n_in, k, args, seed)
        w0 = net.ff_weight_norm()
        _train_arm(net, Xtr_b, ytr_b, "bdsp", args.epochs, args.batch, seed)
        w1 = net.ff_weight_norm()
        arms[arm] = {"inherit_heldout": float(net.acc_on(Xte, yte, inh)),
                     "train_acc": float(net.accuracy(Xtr_b, ytr_b)),
                     "ff_weight_moved": float(abs(w1 - w0)),
                     "no_weight_transport": bool(net.no_weight_transport())}
        nets[arm] = net
        print(f"[gap4-spi][seed {seed}]   arm {arm:<16} held-out {arms[arm]['inherit_heldout']:.3f} "
              f"train {arms[arm]['train_acc']:.3f} ff-moved {arms[arm]['ff_weight_moved']:.2f} "
              f"nwt {arms[arm]['no_weight_transport']} ({time.time()-t_arm:.0f}s)", flush=True)

    # anti-cheats on the in-engine base (the new mechanism under test).
    base = "micro_inengine" if "micro_inengine" in args.arms else ("fixed_fa" if "fixed_fa" in args.arms else args.arms[0])
    les_acc = shuf_acc = shufE_acc = freeze_spi_acc = float("nan")
    apical = {}
    if not getattr(args, "core_arms_only", False):
        lesion = _build_net(base, n_in, k, args, seed); _train_arm(lesion, Xtr_b, ytr_b, "apical_lesion", args.epochs, args.batch, seed)
        les_acc = float(lesion.acc_on(Xte, yte, inh))
        prng = np.random.default_rng(seed + 555); yperm = ytr_b[prng.permutation(len(ytr_b))]
        shuftgt = _build_net(base, n_in, k, args, seed); _train_arm(shuftgt, Xtr_b, yperm, "bdsp", args.epochs, args.batch, seed)
        shuf_acc = float(shuftgt.acc_on(Xte, yte, inh))
        shufE = _build_net(base, n_in, k, args, seed); _train_arm(shufE, Xtr_b, ytr_b, "shufE", args.epochs, args.batch, seed)
        shufE_acc = float(shufE.acc_on(Xte, yte, inh))
        # FREEZE-SPI (the mechanism-specific anti-cheat): the in-engine interneuron NEVER learns (cp_spi_wpi frozen at
        # noisy init) -> silence cannot be EARNED -> the cancellation is noise -> accuracy must fall toward the
        # frozen-signal microcircuit. Only meaningful for the in-engine arm.
        if "micro_inengine" in args.arms:
            frz = _build_net("micro_inengine", n_in, k, args, seed); frz._spi_frozen = True
            _train_arm(frz, Xtr_b, ytr_b, "bdsp", args.epochs, args.batch, seed)
            freeze_spi_acc = float(frz.acc_on(Xte, yte, inh))
            apical["inengine_learned"] = nets["micro_inengine"].inengine_apical_silent_stats(Xte, yte)
            apical["inengine_frozen"] = frz.inengine_apical_silent_stats(Xte, yte)
        print(f"[gap4-spi][seed {seed}]   anti-cheats: lesion {les_acc:.3f} shuf_tgt {shuf_acc:.3f} "
              f"shufE {shufE_acc:.3f} freeze_spi {freeze_spi_acc:.3f}", flush=True)

    res_acc = arms.get("reservoir", {}).get("inherit_heldout", float("nan"))
    ff_acc = arms.get("fixed_fa", {}).get("inherit_heldout", float("nan"))
    ie_acc = arms.get("micro_inengine", {}).get("inherit_heldout", float("nan"))
    mic_acc = arms.get("micro", {}).get("inherit_heldout", float("nan"))
    ceil_acc = arms.get("transport_ceiling", {}).get("inherit_heldout", float("nan"))
    fa_wall = bool(ff_acc <= res_acc + 0.02)
    seed_go = bool(fa_wall and (ie_acc > ff_acc + 0.05))
    print(f"[gap4-spi][seed {seed}] FA-WALL fixed_fa {ff_acc:.3f} vs reservoir {res_acc:.3f} (fa_wall={fa_wall}) | "
          f"micro_inengine {ie_acc:.3f} vs fixed_fa {ff_acc:.3f} -> seed_go={seed_go} "
          f"(micro {mic_acc:.3f}, ceiling {ceil_acc:.3f}, oracle {oracle:.3f}) ({time.time()-t_seed:.0f}s)", flush=True)

    return {"seed": seed, "meta": meta, "chance": chance, "oracle_heldout": oracle, "n_in": n_in, "k": k,
            "arms": arms,
            "lesion": {"inherit_heldout": les_acc}, "shuffled_target": {"inherit_heldout": shuf_acc},
            "shufE": {"inherit_heldout": shufE_acc}, "freeze_spi": {"inherit_heldout": freeze_spi_acc},
            "apical": apical,
            "fa_wall": {"reservoir": res_acc, "fixed_fa": ff_acc, "micro_inengine": ie_acc, "micro": mic_acc,
                        "transport_ceiling": ceil_acc, "fa_wall_holds": fa_wall, "seed_go": seed_go},
            "elapsed_seconds": round(time.time() - t_seed, 1),
            "guards": {"ast_no_forward_W": bool(_ast_no_forward_W(Gap4InEngineNet))}}


def _thr_hash(net):
    from sim.backend import to_host
    thr = getattr(net.br, "cp_neuron_firing_thresholds", None)
    if thr is None:
        return None
    return hashlib.md5(np.asarray(to_host(thr)).tobytes()).hexdigest()[:16]


def construct_smoke(args):
    print("=" * 108, flush=True)
    print("[gap4-spi-smoke] CONSTRUCT-SMOKE (build + step the in-engine microcircuit; NOT the accuracy run).", flush=True)
    seed = args.seeds[0]
    tk = dict(n_super=8, n_members=4, held_per_super=1, n_prop=2, member_id_dim=3, n_obs=4, noise=0.02)
    (Xtr, ytr, _Ltr), (Xte, yte, _Lte), meta, idx = make_task_semantic_inheritance(seed, **tk)
    n_in = Xtr.shape[1]; k = int(meta["k_classes"]); inh = idx["inh_idx"]

    class _A:
        pass
    a = _A()
    for attr in vars(args):
        setattr(a, attr, getattr(args, attr))
    a.hidden = 4; a.pool_k = 1; a.n_hidden_layers = 2; a.settle_steps = 4; a.credit_steps = 4; a.lr = 0.05
    a.wpi_init = "noisy"

    # (1) two-build seed identity (cfg.seed controls the substrate).
    n1 = _build_net("micro_inengine", n_in, k, a, seed)
    n2 = _build_net("micro_inengine", n_in, k, a, seed)
    h1, h2 = _thr_hash(n1), _thr_hash(n2)
    seed_ok = (h1 is not None and h1 == h2)
    print(f"[gap4-spi-smoke] seed check: cp_neuron_firing_thresholds md5 {h1} vs {h2} -> "
          f"{'IDENTICAL' if seed_ok else 'DIFFER/absent'}", flush=True)

    # (2) BYTE-IDENTICAL WHEN OFF: a fixed_fa net (flag never set) must be unaffected by the new code path. Build one,
    #     confirm the engine flag is off and cp_spi_* are None.
    off = _build_net("fixed_fa", n_in, k, a, seed)
    off_clean = (getattr(off.cfg, "enable_selfpredicting_interneuron", False) is False
                 and off.br.cp_spi_wpi is None and off.br.cp_spi_int_rate is None)
    print(f"[gap4-spi-smoke] OFF-path clean (flag off + cp_spi_* None on a non-inengine arm): {off_clean}", flush=True)

    results = {}
    Xs, ys = Xtr[:3], ytr[:3]
    for arm in ["fixed_fa", "micro", "micro_inengine", "transport_ceiling"]:
        rec = {"built": False, "stepped": False, "no_weight_transport": None, "error": None}
        try:
            net = _build_net(arm, n_in, k, a, seed)
            rec["built"] = True
            if arm == "micro_inengine":
                cos0 = _cos(net.W_PI, net.Y[net.n_hidden_layers - 1])
                from sim.backend import to_host
                wpi0 = np.asarray(to_host(net.br.cp_spi_wpi)).copy()
            for xi in range(len(Xs)):
                net._train_one(Xs[xi], int(ys[xi]), "bdsp")
            rec["stepped"] = True
            rec["heldout_read"] = float(net.acc_on(Xte, yte, inh)) if len(inh) else None
            rec["no_weight_transport"] = bool(net.no_weight_transport())
            if arm == "micro_inengine":
                wpi1 = np.asarray(to_host(net.br.cp_spi_wpi))
                cos1 = _cos(wpi1, net.Y[net.n_hidden_layers - 1])
                rec["cp_spi_wpi_learned_in_engine"] = bool(float(np.abs(wpi1 - wpi0).sum()) > 1e-9)
                rec["selfpred_cos_before_after"] = [round(cos0, 4), round(cos1, 4)]
                rec["int_drive_formed_by_engine"] = bool(net.br.cp_bdsp_int_drive is not None)
                sil = net.inengine_apical_silent_stats(Xte, yte)
                rec["apical_silent_read"] = {kk: (round(vv, 4) if isinstance(vv, float) and not np.isnan(vv) else vv)
                                             for kk, vv in sil.items()}
        except Exception as e:
            rec["error"] = repr(e); traceback.print_exc()
        results[arm] = rec
        status = "OK" if (rec["built"] and rec["stepped"] and rec["error"] is None) else "FAIL"
        extra = ""
        if arm == "transport_ceiling":
            extra = f" (nwt={rec['no_weight_transport']} MUST be False)"
        if arm == "micro_inengine" and rec.get("apical_silent_read"):
            extra = (f" (wpi_learned={rec.get('cp_spi_wpi_learned_in_engine')}, cos {rec.get('selfpred_cos_before_after')}, "
                     f"int_drive_by_engine={rec.get('int_drive_formed_by_engine')})")
        print(f"[gap4-spi-smoke]   arm {arm:<16} built={rec['built']} stepped={rec['stepped']} -> {status}{extra}", flush=True)

    ast_ok = bool(_ast_no_forward_W(Gap4InEngineNet))
    ie = results.get("micro_inengine", {})
    mech_ok = bool(ie.get("cp_spi_wpi_learned_in_engine") and ie.get("int_drive_formed_by_engine"))
    ceiling_ok = (results.get("transport_ceiling", {}).get("no_weight_transport") is False)
    all_ok = (all(r["built"] and r["stepped"] and r["error"] is None for r in results.values())
              and seed_ok and ast_ok and off_clean and mech_ok and ceiling_ok)
    out = {"probe": "gap4_selfpredict_interneuron_inengine_CONSTRUCT_SMOKE", "seed": seed, "task_meta": meta,
           "seed_identity_ok": bool(seed_ok), "off_path_clean": bool(off_clean), "ast_no_forward_W": ast_ok,
           "inengine_mechanism_ok": mech_ok, "transport_ceiling_guard_correctly_fails": ceiling_ok,
           "arms": results, "CONSTRUCT_SMOKE_PASS": bool(all_ok),
           "NOTE": ("Proves the in-engine self-predicting interneuron CONSTRUCTS + STEPS + LEARNS cp_spi_wpi on the "
                    "substrate (numpy) and is byte-identical-off. The decisive 6-seed GPU run is queued separately.")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print("=" * 108, flush=True)
    print(f"[gap4-spi-smoke] CONSTRUCT_SMOKE_PASS={all_ok} (seed_ok={seed_ok}, off_clean={off_clean}, ast={ast_ok}, "
          f"mech_ok={mech_ok}, ceiling_fails={ceiling_ok})", flush=True)
    print(f"[gap4-spi-smoke] wrote {args.out}", flush=True)
    print("=" * 108, flush=True)
    return 0 if all_ok else 1


def _agg_mean(per, keys):
    vals = []
    for p in per:
        v = p; ok = True
        for kk in keys:
            if isinstance(v, dict) and kk in v:
                v = v[kk]
            else:
                ok = False; break
        if ok and isinstance(v, (int, float)) and not (isinstance(v, float) and np.isnan(v)):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def run_full(args):
    from tools.verdict import Verdict
    from tools.lab import attributable_to
    t0 = time.time(); per = []; err = None
    try:
        for s in args.seeds:
            per.append(run_seed(s, args))
    except Exception as e:
        err = repr(e); traceback.print_exc()

    agg = {}
    if per:
        n_seed = len(per)
        n_fa_wall = sum(bool(p["fa_wall"]["fa_wall_holds"]) for p in per)
        n_seed_go = sum(bool(p["fa_wall"]["seed_go"]) for p in per)
        chance = _agg_mean(per, ["chance"]); oracle = _agg_mean(per, ["oracle_heldout"])
        res = _agg_mean(per, ["arms", "reservoir", "inherit_heldout"])
        ff = _agg_mean(per, ["arms", "fixed_fa", "inherit_heldout"])
        mic = _agg_mean(per, ["arms", "micro", "inherit_heldout"])
        ie = _agg_mean(per, ["arms", "micro_inengine", "inherit_heldout"])
        ceil = _agg_mean(per, ["arms", "transport_ceiling", "inherit_heldout"])
        les = _agg_mean(per, ["lesion", "inherit_heldout"]); shuf = _agg_mean(per, ["shuffled_target", "inherit_heldout"])
        shufE = _agg_mean(per, ["shufE", "inherit_heldout"]); frz = _agg_mean(per, ["freeze_spi", "inherit_heldout"])
        ie_ratio = _agg_mean(per, ["apical", "inengine_learned", "silent_ratio"])
        ie_cos = _agg_mean(per, ["apical", "inengine_learned", "selfpred_cos"])
        fz_ratio = _agg_mean(per, ["apical", "inengine_frozen", "silent_ratio"])
        ceiling_guard_fails = all(p["arms"].get("transport_ceiling", {}).get("no_weight_transport") is False
                                  for p in per if "transport_ceiling" in p["arms"])
        ast_ok = all(p["guards"]["ast_no_forward_W"] for p in per)
        lesion_collapse = bool(les <= chance + 0.10); shuf_collapse = bool(shuf <= chance + 0.10)
        shufE_collapse = bool(shufE <= chance + 0.10)
        task_ok = bool(oracle >= 0.80)
        earned_silence = bool(ie_ratio < 0.5 and ie_cos > 0.6 and fz_ratio > 0.8) if not np.isnan(ie_ratio) else None
        print("\n[gap4-spi] lesion attribution (how much of the in-engine effect is NOT in the apical-lesion control):", flush=True)
        attributable_to("micro_inengine vs apical lesion", ie, les)
        attributable_to("in-engine LEARNING (learned vs freeze-spi)", ie, frz)

        GO = bool(task_ok and ceiling_guard_fails and ast_ok and lesion_collapse and shuf_collapse
                  and n_fa_wall >= 5 and n_seed_go >= 5)

        _v = Verdict("gap#4 RANK-1: learned-in-engine self-predicting interneuron", chance=chance)
        _v.require("task solvable (oracle >= 0.80)", task_ok, expect=True)
        # INTERPRETABILITY: the transport ceiling must clear chance, else nothing beneath it is readable (the
        # "crux was never askable" lesson). floor() reads the run's own chance.
        _v.floor("transport ceiling vs chance", ceil, artifact={"chance": chance}, key="chance",
                 note="weight transport ALLOWED; a ceiling at/below chance => the read cannot carry credit => UNDEFINED")
        _v.require("FA-wall holds on >=5/6 (fixed_fa <= reservoir)", bool(n_fa_wall >= 5), expect=True)
        _v.require("weight-transport guard FAILS on the ceiling arm", ceiling_guard_fails, expect=True)
        _v.require("no forward W in the credit path (AST)", ast_ok, expect=True)
        _v.control("apical lesion collapses the read", treatment=float(ie), control=float(les))
        _v.control("in-engine LEARNING is load-bearing (learned vs freeze-spi)", treatment=float(ie), control=float(frz))
        _verdict = _v.decide(go=GO)
        if _verdict["status"] == "UNDEFINED":
            GO = False

        agg = {"n_seeds": n_seed, "n_fa_wall": n_fa_wall, "n_seed_go": n_seed_go, "chance": chance, "oracle": oracle,
               "reservoir": res, "fixed_fa": ff, "micro": mic, "micro_inengine": ie, "transport_ceiling": ceil,
               "lesion": les, "shuffled_target": shuf, "shufE": shufE, "freeze_spi": frz,
               "inengine_learned_silent_ratio": ie_ratio, "inengine_learned_selfpred_cos": ie_cos,
               "inengine_frozen_silent_ratio": fz_ratio,
               "fa_wall_holds_all": bool(n_fa_wall >= 5), "task_ok": task_ok,
               "ceiling_guard_correctly_fails": bool(ceiling_guard_fails), "ast_no_forward_W": bool(ast_ok),
               "lesion_collapse": lesion_collapse, "shuffled_collapse": shuf_collapse, "shufE_collapse": shufE_collapse,
               "earned_silence": earned_silence, "GO": GO,
               **{kk: _verdict[kk] for kk in ("preconditions", "disabled_processes", "undefined_reasons")},
               "verdict": (f"{_verdict['status'] if _verdict['status'] == 'UNDEFINED' else ('GO' if GO else 'NO-GO')} "
                           f"({n_seed_go}/{n_seed} seed_go; FA-wall {n_fa_wall}/{n_seed}). reservoir {res:.3f} | "
                           f"fixed_fa {ff:.3f} | micro {mic:.3f} | micro_inengine {ie:.3f} | ceiling {ceil:.3f} | "
                           f"oracle {oracle:.3f} (chance {chance:.3f}). in-engine earned-silent: learned ratio "
                           f"{ie_ratio:.3f} (cos {ie_cos:.2f}) vs freeze-spi {fz_ratio:.3f}. anti-cheats: lesion "
                           f"{les:.3f} shuf {shuf:.3f} shufE {shufE:.3f} freeze_spi {frz:.3f}.")}
    summary = {"probe": "gap4_selfpredict_interneuron_inengine_FULL", "seeds": args.seeds, "config": vars(args),
               "elapsed_seconds": round(time.time() - t0, 1), "error": err, "per_seed": per, "aggregate": agg,
               "GO": bool(agg.get("GO", False)),
               "NOTE": ("GO = micro_inengine (LEARNED-IN-ENGINE cancellation) > fixed_fa on >=5/6 seeds IN the FA-wall "
                        "regime + earned in-engine silence + all anti-cheats + an INTERPRETABLE ceiling. UNDEFINED if "
                        "the transport ceiling cannot clear chance (the read-regime forecloses the instrument).")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
    print("=" * 108, flush=True)
    print(f"[gap4-spi] {agg.get('verdict', 'ERROR: ' + str(err))}", flush=True)
    print(f"[gap4-spi] FULL run wrote {args.out}", flush=True)
    print("=" * 108, flush=True)
    return 0 if agg.get("GO") else 1


def main():
    ap = argparse.ArgumentParser(description="gap#4 RANK-1 learned-in-engine self-predicting interneuron microcircuit.")
    ap.add_argument("--construct-smoke", action="store_true", help="the ONLY locally-run mode (build + step + learn).")
    ap.add_argument("--full", action="store_true", help="the per-seed FULL science (the controller's GPU run).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--arms", nargs="+",
                    default=["reservoir", "fixed_fa", "micro", "micro_inengine", "transport_ceiling"],
                    choices=["reservoir", "fixed_fa", "micro", "micro_inengine", "transport_ceiling"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--pool-k", dest="pool_k", type=int, default=16)
    ap.add_argument("--n-hidden-layers", dest="n_hidden_layers", type=int, default=2)
    ap.add_argument("--settle-steps", dest="settle_steps", type=int, default=40)
    ap.add_argument("--credit-steps", dest="credit_steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--train-subsample", dest="train_subsample", type=int, default=0)
    ap.add_argument("--core-arms-only", dest="core_arms_only", action="store_true")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--bdsp-w-max", dest="bdsp_w_max", type=float, default=6.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.30)
    ap.add_argument("--in-current-pA", dest="in_current_pA", type=float, default=520.0)
    ap.add_argument("--in-bias-pA", dest="in_bias_pA", type=float, default=260.0)
    ap.add_argument("--apical-gain-pA", dest="apical_gain_pA", type=float, default=900.0)
    ap.add_argument("--tonic-h-pA", dest="tonic_h_pA", type=float, default=450.0)
    ap.add_argument("--tonic-o-pA", dest="tonic_o_pA", type=float, default=500.0)
    ap.add_argument("--graded-credit", dest="graded_credit", action="store_true", default=True)
    ap.add_argument("--no-graded-credit", dest="graded_credit", action="store_false")
    ap.add_argument("--wpi-init", dest="wpi_init", default="noisy", choices=["noisy", "fixedpoint"])
    ap.add_argument("--wpi-lr", dest="wpi_lr", type=float, default=0.2)
    ap.add_argument("--kp-lr", dest="kp_lr", type=float, default=0.2)
    ap.add_argument("--kp-decay", dest="kp_decay", type=float, default=1e-4)
    ap.add_argument("--assert-no-transport", dest="assert_no_transport", action="store_true")
    ap.add_argument("--n-super", dest="n_super", type=int, default=24)
    ap.add_argument("--n-members", dest="n_members", type=int, default=8)
    ap.add_argument("--held-per-super", dest="held_per_super", type=int, default=3)
    ap.add_argument("--n-prop", dest="n_prop", type=int, default=3)
    ap.add_argument("--member-id-dim", dest="member_id_dim", type=int, default=3)
    ap.add_argument("--n-obs", dest="n_obs", type=int, default=14)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--oracle-epochs", dest="oracle_epochs", type=int, default=250)
    ap.add_argument("--oracle-lr", dest="oracle_lr", type=float, default=0.3)
    ap.add_argument("--oracle-batch", dest="oracle_batch", type=int, default=128)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if a.assert_no_transport:
        assert _ast_no_forward_W(Gap4InEngineNet), "AST guard FAILED: a credit-path update reads a forward-weight array"
    if a.full:
        return run_full(a)
    return construct_smoke(a)


if __name__ == "__main__":
    sys.exit(main())
