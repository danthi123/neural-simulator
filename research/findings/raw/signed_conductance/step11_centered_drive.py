"""RED-HERRING de-risk (research verdict: objrel is a rank-1 COMMON-MODE removal, NOT dendritic). The spiking WTA loses
objrel because it reads the ens firing driven by (logit_r + a role-INDEPENDENT pedestal C = WS_ENS_FLOOR + Dale-shift);
the f-I on the large C swamps the sub-1% structural margin. FIX (never tried cleanly): subtract the common-mode BEFORE the
read = drive the SAME WTA with the MEAN-SUBTRACTED logit (centered_r = logit_r - mean_r logit) + a small offset. Test on the
SAME on-bridge WTA, seeds 42/44/100, per-slot vs TRUE roles.
  PEDESTAL control (logit + FLOOR, the current deploy): must FAIL objrel (reproduce the 0.00 boundary).
  CENTERED (mean-subtracted): if it RECOVERS objrel robustly multi-seed -> the somatic feedforward common-mode subtraction
    is the fix (a shared summing inhibitory pool), NO dendrites, NO sim/ edit for the concept. Then build the biological FF
    inhibition. NOTE: the logit here is host-computed (f·Ws) = a DIAGNOSTIC to isolate the drive geometry; the biological
    version delivers the same via a shared inh pool tracking the mean ens drive."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
from sim.backend import get_backend, to_host
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS, _ROLE_IDX
from research.runners._emerge61_spiking_broca_order_robustness_derisk import _restore_state
from research.runners.core_sim_composition import RESET_STEPS

seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42", "44", "100"])]
OFFSET = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0        # small uniform offset so the ens can fire (NOT the big FLOOR)
DSCALE = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0        # logit -> pA scale
N_TRAIN, N_TEST = 60, 12
C.WS_BIAS_SCALE_C2 = 0.0; C.WS_REPLAY = 3; C.READ_T_STEP_C2 = 30
corpus = C.setup_corpus(seed=42)
subj, verb, obj = corpus["subj"], corpus["verb"], corpus["obj"]
enc = Encoder(corpus["discovered"])
xp, _ = get_backend()


def drive_wta(ub, ens, res, drive3):
    """Drive the 3 role ens with drive3 (pA) under the WTA competition; return argmax over ens summed firing."""
    b = ub.bridge
    _restore_state(b, res._snap)
    prev_ou, prev_heb = b.core_config.enable_ou_process, b.core_config.enable_hebbian_learning
    b.core_config.enable_ou_process = False; b.core_config.enable_hebbian_learning = False
    ef = np.zeros(3)
    try:
        for _ in range(RESET_STEPS):
            b.runtime_state.current_time_ms += b.core_config.dt_ms; b._run_one_simulation_step()
        for _rep in range(C.WS_REPLAY):
            cur = np.zeros(b.core_config.num_neurons, np.float32)
            for r in range(3):
                cur[ens[r]] = np.float32(drive3[r])
            b.cp_external_input_current[:] = xp.asarray(cur)
            for _ in range(C.READ_T_STEP_C2):
                b.runtime_state.current_time_ms += b.core_config.dt_ms; b._run_one_simulation_step()
                fs = np.asarray(to_host(b.cp_firing_states))
                for r in range(3):
                    ef[r] += fs[ens[r]].sum()
    finally:
        b.cp_external_input_current[:] = 0.0
        b.core_config.enable_ou_process = prev_ou; b.core_config.enable_hebbian_learning = prev_heb
    return int(np.argmax(ef))


for seed in seeds:
    t0 = time.time()
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, N_TRAIN, rng, subj, verb, obj)
    ub, ens, inh = C._build_wired_bridge(seed, corpus, mode="c2")
    res_idx, W_in = C.wire_reservoir(ub, enc.dim, seed)
    res = C.UBReservoir(ub, res_idx, W_in); res.snapshot_after_wiring()
    Ws = C._fit_Ws_spiking(res, enc, train)                       # ridge; host logit = f·Ws (DIAGNOSTIC drive geometry)
    n_res = len(res_idx)
    trng = np.random.default_rng(seed * 977 + 13)
    canon = _gen(["transitive"], N_TEST, trng, subj, verb, obj)
    objr = _gen(["objrel"], N_TEST, trng, subj, verb, obj)

    def score(facts, mode):
        ok = tot = s0ok = s0t = 0
        for toks, roles in facts:
            f = np.asarray(res.final_state(enc.encode(toks)), float)[:n_res]
            for k, pos in enumerate(sorted(roles)):
                if k >= 3:
                    break
                tgt = _ROLE_IDX[roles[pos]]
                if tgt >= 3:
                    continue
                logit = np.array([f @ Ws[k][:n_res, r] for r in range(3)])
                if mode == "pedestal":
                    drive3 = (logit - logit.min()) * DSCALE + C.WS_ENS_FLOOR_C2   # the current deploy geometry (common-mode)
                elif mode == "centered":
                    drive3 = (logit - logit.mean()) * DSCALE + OFFSET             # mean-subtracted (rank-1 common-mode gone)
                else:  # linear ceiling
                    pred = int(np.argmax(logit)); ok += int(pred == tgt); tot += 1
                    if k == 0:
                        s0ok += int(pred == tgt); s0t += 1
                    continue
                pred = drive_wta(ub, ens, res, drive3)
                ok += int(pred == tgt); tot += 1
                if k == 0:
                    s0ok += int(pred == tgt); s0t += 1
        return ok / max(tot, 1), s0ok / max(s0t, 1)

    lo, ls0 = score(objr, "linear")
    po, ps0 = score(objr, "pedestal"); pc, _ = score(canon, "pedestal")
    co, cs0 = score(objr, "centered"); cc, _ = score(canon, "centered")
    print(f"seed {seed}: LINEAR objrel-slot0 {ls0:.2f} | PEDESTAL-WTA canon {pc:.2f}/objrel-slot0 {ps0:.2f} | "
          f"CENTERED-WTA canon {cc:.2f}/objrel-slot0 {cs0:.2f}  [{time.time()-t0:.0f}s]", flush=True)
