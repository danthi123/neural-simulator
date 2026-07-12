"""SPIKING learn-W_in on the CORRECT instrument: next-token PREDICTION + distributional GENERALIZATION on ONE bridge.

The cue-classification task was the wrong instrument (Johnson-Lindenstrauss: a fixed random W_in already separates
distinct cues, so no headroom -- see 2026-07-12 findings). The R3 learn-W_in benefit is a PREDICTION + GENERALIZATION
phenomenon, and on a class-structured next-token task with a class-irrelevant identity confound it has a strong 6-seed
rate headroom (held-out learn 0.900 vs fixed 0.656 at sf=3/idn=20/id_pool=60/n=60). This runs the SPIKING version of
that task on ONE SimulationBridge: a fixed spiking reservoir + a PLASTIC input->reservoir W_in learned by the committed
`enable_bdsp` rule (apical = k*(Y@delta), fixed-random Y, NO weight transport). Reuses `WinLearnReservoir` +
`_run_arm` + the rate task builders; the ONLY new code is a multi-hot code drive (GenReservoir._drive_for_token) and
the task glue. NO `sim/` edit, NO edit to the existing runner.

TASK (per example): present a token's multi-hot code (sf shared class dims + idn identity-confound dims) for one step,
then a QUERY step; the read-out predicts the token's NEXT CLASS (G-way). Train W_in + fit the ridge on COMMON synonyms;
the metric is HELD-OUT next-class accuracy on the RARE (held-out) synonym of each class. learn_win should hold the
Markov ceiling; fixed_win should collapse (the identity confound swamps a fixed random projection's generalization).

GATE (learn_win beats fixed_win on HELD-OUT generalization, anti-cheats intact):
  learn_heldout - fixed_heldout >= 0.10; input_lesion ~ chance; label-scramble ~ chance; dw_rec == 0 (frozen recurrence);
  no_weight_transport True. Do NOT force a positive; a spiking collapse of the rate margin is an honest boundary.

Run (1-seed CPU smoke):
  SIM_BACKEND=numpy python -u -m research.runners._reslm_onbridge_generalize_derisk \
      --seeds 42 --n-pool 60 --G 6 --syn 5 --sf 3 --idn 20 --id-pool 60 --n-seq 12 --epochs 3 --in-hi 120 --smoke
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

import research.runners._reslm_onbridge_learn_win_derisk as _base
from research.runners._reslm_onbridge_learn_win_derisk import (
    WinLearnReservoir, _run_arm, _fit_ridge, _decode_acc,
)
from research.runners._reslm_generalize_rate_check import build_codes, build_stream


# ---------------------------------------------------------------------------------------------------------------------
class GenReservoir(WinLearnReservoir):
    """Same on-bridge fixed reservoir + BDSP-learned W_in, but each presentation step drives a MULTI-HOT code (a list of
    active input dims) instead of a single token sub-pop -- so tokens can share code structure (the class feature) and
    carry a class-irrelevant identity confound. Everything else (forward/train_arm/read/BDSP credit) is inherited.

    DENDRITIC per-compartment gain (dend_gain, the D1/D1.5 mechanism): each input dim = its own compartment with a LOCAL
    divisive gain g_d adapted to that dim's own activation FREQUENCY; the drive is normalized in_hi/(sigma + g_d). Common
    dims (the identity confound, shared across classes -> high frequency) get a large g_d -> down-weighted; category-
    specific class dims (lower frequency) keep a small g_d -> emphasized. This is the per-input normalization a single-
    soma point neuron provably cannot deliver (2026-06-14 D1 GO; survives the spike read D1.5) -- the mapped escape for
    the point-neuron confound-suppression boundary. dend_gain=None => byte-identical to the plain multi-hot drive."""
    dend_gain = None
    dend_sigma = 0.05
    dend_scale = 1.0        # mean(sigma+g) -> keeps the MEAN drive == in_hi so the re-weighting is scale-consistent
                           # across seeds (fixes the operating-point fragility of the bare in_hi/(sigma+g) form)

    def _drive_for_token(self, active, silence):
        cur = np.zeros(self._num, np.float32)
        if not silence:
            for d in np.atleast_1d(active):
                d = int(d)
                if self.dend_gain is None:
                    _hi = self.in_hi
                else:
                    # mean-normalized per-compartment divisive gain: common dims (high g) -> <1x, rare -> >1x, mean ~1x
                    _hi = self.in_hi * self.dend_scale / (self.dend_sigma + float(self.dend_gain[d]))
                cur[self.tok_idx[d]] = _hi
        cur[self.res_idx] += self.res_bias
        return cur


def _make_sents(codes, stream, m):
    """(cur_token, next_class) -> (toks=[code_active_dims], next_class). READ AT the code step (predict next class from
    the current token's reservoir response) -- no separate delayed QUERY step: fast synaptic decay wipes the code's
    residual before a delayed read, so the delayed read was silent (the spiking reservoir has no slow leaky integrator
    like the rate reference). Reading while the code drives gives a rich, code-dependent read = the clean input-projection
    test. (A distal-cue version needing persistence is a separate NMDA/adaptation rung.)"""
    return [([np.where(codes[tok] > 0)[0].tolist()], int(nc)) for tok, nc in stream]


def _derisk_one(seed, args):
    codes, V_tok, m = build_codes(seed, args.G, args.syn, args.sf, args.idn, id_pool=args.id_pool)
    train_raw, held_raw = build_stream(seed, args.G, args.syn, args.n_seq)
    train = _make_sents(codes, train_raw, m)
    held = _make_sents(codes, held_raw, m)
    n_classes = args.G
    chance = 1.0 / n_classes
    V = m                                                        # read at the code step; no separate QUERY dim

    res = GenReservoir(V, args.n_pool, args.in_pop, seed, args.soma_g, args.bdsp_lr, args.bdsp_p0,
                       args.bdsp_beta, args.bdsp_w_min, args.bdsp_w_max, args.in_hi, args.res_bias,
                       args.k_apical, args.fwd_wmean, args.fwd_wjit, args.fwd_density)
    # graded clean-error credit (the M2.6 lever): read the apical credit as the graded E*P expectation instead of the
    # noisy measured burst B (additive sim/ flag, default-off byte-identical). The kernel reads cfg at runtime.
    res.cfg.enable_bdsp_graded_credit = bool(args.bdsp_graded)
    # DENDRITIC per-compartment gain (D1/D1.5): per-input-dim divisive gain = the dim's activation FREQUENCY over the
    # training stream (common identity dims -> high freq -> down-weighted; class dims -> lower -> emphasized). The
    # per-input normalization a single-soma point neuron cannot deliver -- the mapped escape for this boundary.
    if args.dend_gain:
        toks_seen = np.array([t for (t, _) in train_raw], dtype=int)
        g = codes[toks_seen].mean(axis=0).astype(np.float64)              # per-dim activation frequency
        if args.dend_permute:
            # ANTI-CHEAT: shuffle which dim gets which gain -> breaks the frequency<->dim correspondence. If the lift
            # survives this, it's a drive-scale artifact, not the per-compartment (frequency-matched) normalization.
            np.random.default_rng(seed * 613 + 5).shuffle(g)
        res.dend_gain = g
        res.dend_sigma = float(args.dend_sigma)
        active = g > 0                                                    # dims that appear in some code (the driven ones)
        res.dend_scale = float(np.mean(args.dend_sigma + g[active])) if active.any() else 1.0
    res._seed_for_Y = seed + 9973
    res.set_n_classes(n_classes)
    res._w_init = res._weights().copy()
    args.seed_base = seed

    mean_spk = res.mean_res_spikes(train[0][0])
    coupling = res.apical_coupling_diag(train[0][0])

    arms = {}
    _salt = {"fixed_win": 1, "learn_win": 2, "apical_lesion": 3, "wrong_sign": 4}
    for mode in args.arms:
        # _run_arm fits the ridge on `train` (COMMON synonyms) and decodes `held` (HELD-OUT rare synonyms) = the
        # generalization metric; for learn_win it also runs the input-lesion + label-scramble collapse controls.
        arms[mode] = _run_arm(res, mode, train, held, n_classes,
                              args, np.random.default_rng(seed * 211 + _salt.get(mode, 9) * 101))
    nwt = res.no_weight_transport()

    def marm(mode, key):
        return arms.get(mode, {}).get(key)

    return {"seed": seed, "G": args.G, "syn": args.syn, "sf": args.sf, "idn": args.idn, "id_pool": args.id_pool,
            "m": int(m), "V": V, "n_pool": res.n, "chance": round(chance, 4),
            "res_spikes_per_step": round(float(mean_spk), 4), "coupling": coupling,
            "no_weight_transport": bool(nwt),
            "fixed_heldout": marm("fixed_win", "decode_acc"), "learn_heldout": marm("learn_win", "decode_acc"),
            "learn_dw_win": marm("learn_win", "dw_win"), "learn_dw_rec": marm("learn_win", "dw_rec"),
            "learn_input_lesion": marm("learn_win", "input_lesion_acc"),
            "learn_scramble": marm("learn_win", "scramble_acc"),
            "apical_lesion_heldout": marm("apical_lesion", "decode_acc"),
            "wrong_sign_heldout": marm("wrong_sign", "decode_acc")}


def _print_seed(d):
    print(f"  [seed {d['seed']}] G={d['G']} syn={d['syn']} sf={d['sf']} idn={d['idn']} id_pool={d['id_pool']} "
          f"m={d['m']} n_pool={d['n_pool']} chance={d['chance']:.3f} | res spikes/step {d['res_spikes_per_step']:.4f}",
          flush=True)
    c = d["coupling"]
    print(f"    B_rises: B_rest {c['B_rest']:.4f} -> B_apical {c['B_apical']:.4f} (B_rises {c['B_rises']})", flush=True)
    fx, ln = d["fixed_heldout"], d["learn_heldout"]
    margin = (ln - fx) if (fx is not None and ln is not None) else None
    print(f"    HELD-OUT generalization: fixed {fx} | learn {ln} | margin {margin}", flush=True)
    print(f"    learn dw_win {d['learn_dw_win']} dw_rec {d['learn_dw_rec']} | input_lesion {d['learn_input_lesion']} "
          f"scramble {d['learn_scramble']} | no_weight_transport {d['no_weight_transport']}", flush=True)
    if d.get("apical_lesion_heldout") is not None:
        print(f"    apical_lesion held-out {d['apical_lesion_heldout']} | wrong_sign held-out {d['wrong_sign_heldout']}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--syn", type=int, default=5)
    ap.add_argument("--sf", type=int, default=3)
    ap.add_argument("--idn", type=int, default=20)
    ap.add_argument("--id-pool", type=int, default=60)
    ap.add_argument("--n-seq", type=int, default=12)
    ap.add_argument("--n-pool", type=int, default=60)
    ap.add_argument("--in-pop", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-out", type=float, default=0.01)
    ap.add_argument("--ridge-lam", type=float, default=1.0)
    ap.add_argument("--bdsp-lr", type=float, default=0.02)
    ap.add_argument("--bdsp-p0", type=float, default=0.30)
    ap.add_argument("--bdsp-beta", type=float, default=1.0)
    ap.add_argument("--bdsp-w-min", type=float, default=0.0)
    ap.add_argument("--bdsp-w-max", type=float, default=160.0)
    ap.add_argument("--soma-g", type=float, default=120.0)
    ap.add_argument("--k-apical", type=float, default=150.0)
    ap.add_argument("--in-hi", type=float, default=120.0,
                    help="active input sub-pop drive (pA); LOWER than the 1-hot task since many dims fire at once")
    ap.add_argument("--res-bias", type=float, default=55.0)
    ap.add_argument("--fwd-wmean", type=float, default=32.0)
    ap.add_argument("--fwd-wjit", type=float, default=0.5)
    ap.add_argument("--fwd-density", type=float, default=1.0)
    ap.add_argument("--t-step", type=int, default=0,
                    help="override bridge steps per token (read window); 0 = base default (8). Longer window = more "
                         "spikes averaged into the read = higher read fidelity (a free read-window lever, no sim/ edit).")
    ap.add_argument("--exc-w", type=float, default=0.0,
                    help="override reservoir recurrent EXC weight (0 = EMERGE-82 default 6.0). Raise the E/I ratio so a "
                         "broad multi-hot code drives the reservoir to spike during the CLEAN read (the E/I-balanced "
                         "default suppresses broad inputs -> silent read -> chance).")
    ap.add_argument("--inh-w", type=float, default=-1.0,
                    help="override reservoir recurrent INH weight (<0 = EMERGE-82 default 8.0). Lower it to reduce the "
                         "recurrent inhibition that clamps the read.")
    ap.add_argument("--bdsp-graded", action="store_true",
                    help="use the GRADED clean-error credit (E*P) instead of the noisy measured burst B "
                         "(enable_bdsp_graded_credit; the M2.6 lever for the credit-coarseness boundary)")
    ap.add_argument("--dend-gain", action="store_true",
                    help="DENDRITIC per-compartment divisive gain on the input drive (D1/D1.5): down-weight common "
                         "(identity-confound) dims, emphasize category-specific class dims -- the point-neuron-can't "
                         "per-input normalization that is the mapped escape for the confound-suppression boundary")
    ap.add_argument("--dend-sigma", type=float, default=0.05)
    ap.add_argument("--dend-permute", action="store_true",
                    help="anti-cheat: shuffle the per-dim dendritic gains (break frequency<->dim) -> the lift must collapse")
    ap.add_argument("--arms", type=str, nargs="+", default=["fixed_win", "learn_win"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", "--out", dest="json", type=str, default="raw/_reslm_gen_spk.json")
    args = ap.parse_args()

    if args.t_step and args.t_step > 0:
        _base._T_STEP = int(args.t_step)                        # widen the read window (module global forward() reads)
        print(f"[reslm-gen] read window _T_STEP = {_base._T_STEP} (overridden)", flush=True)
    if args.exc_w and args.exc_w > 0:
        _base._EXC_W = float(args.exc_w)                        # reservoir region reads these module globals at build
        print(f"[reslm-gen] reservoir _EXC_W = {_base._EXC_W} (overridden)", flush=True)
    if args.inh_w >= 0:
        _base._INH_W = float(args.inh_w)
        print(f"[reslm-gen] reservoir _INH_W = {_base._INH_W} (overridden)", flush=True)

    print(f"RES-LM ON-BRIDGE GENERALIZE: spiking learn-W_in (committed enable_bdsp) on a class-structured next-token "
          f"PREDICTION task; HELD-OUT synonym generalization, fixed vs learned W_in; seeds={args.seeds} arms={args.arms}",
          flush=True)
    per = [_derisk_one(s, args) for s in args.seeds]
    for d in per:
        _print_seed(d)

    fx = [p["fixed_heldout"] for p in per if p["fixed_heldout"] is not None]
    ln = [p["learn_heldout"] for p in per if p["learn_heldout"] is not None]
    summary = {}
    if fx and ln:
        fxm, lnm = float(np.mean(fx)), float(np.mean(ln))
        margin = lnm - fxm
        lesion = [p["learn_input_lesion"] for p in per if p["learn_input_lesion"] is not None]
        scr = [p["learn_scramble"] for p in per if p["learn_scramble"] is not None]
        dwr = [p["learn_dw_rec"] for p in per if p["learn_dw_rec"] is not None]
        chance = per[0]["chance"]
        anti_ok = (all(l <= chance + 0.08 for l in lesion) and all(s <= chance + 0.08 for s in scr)
                   and all(abs(r) < 1e-9 for r in dwr) and all(p["no_weight_transport"] for p in per))
        go = margin >= 0.10 and lnm > fxm and anti_ok
        summary = {"fixed_heldout_mean": round(fxm, 4), "learn_heldout_mean": round(lnm, 4),
                   "margin_mean": round(margin, 4), "anti_cheats_ok": bool(anti_ok), "chance": chance,
                   "verdict": "GO" if go else "BOUNDARY"}
        print("\n" + "=" * 110, flush=True)
        if go:
            print(f"[reslm-gen] VERDICT: GO -- spiking learn-W_in (committed enable_bdsp) HOLDS held-out generalization "
                  f"{lnm:.3f} while fixed-W_in collapses to {fxm:.3f} (margin {margin:+.3f}); anti-cheats intact "
                  f"(input-lesion+scramble ~ chance {chance:.3f}, dw_rec==0, no weight transport). The R3 learn-W_in "
                  f"generalization mechanism realizes ON SPIKES. Escalate 6-seed + adversarially verify before committing.",
                  flush=True)
        else:
            print(f"[reslm-gen] VERDICT: BOUNDARY -- learn {lnm:.3f} vs fixed {fxm:.3f} (margin {margin:+.3f}); "
                  f"anti_cheats_ok={anti_ok}. Machinery ran; the rate headroom did not clear on spikes at this "
                  f"operating point/scale. Name the residual (in_hi/activity, epochs, n_seq, confound) as the next "
                  f"single-variable de-risk. Do NOT force a positive.", flush=True)
        print("=" * 110, flush=True)

    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    json.dump({"args": vars(args), "per_seed": per, "summary": summary}, open(args.json, "w"), indent=2)
    print(f"[reslm-gen] wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
