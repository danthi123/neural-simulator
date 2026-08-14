"""gap#1 / A1 — the FULL mouth next-word path on the substrate END-TO-END: the WKV recurrent STATE (slow-NMDA
conductance) -> the OUTPUT PROJECTION (graded read) -> the READ-OUT + head_b (graded read + tonic bias pop) ->
next-word logits. NO host state, NO host matmul on the margin/state. Composes two separately-validated GO pieces.

WHERE THIS SITS (per-token `tid`, WKV leaky state `ap`/`an`):
    (1) v      = Wv @ LN(emb[tid])                       # input projection    (host, BPTT weights — DECLARED residual)
    (2) ap,an  = decay*ap+relu(v), decay*an+relu(-v)     # WKV leaky STATE     <<< [WK] SUBSTRATE slow-NMDA conductance
    (3) r_h    = sigmoid(Wr @ LN(emb[tid]))              # receptance gate     (host — DECLARED residual)
    (4) h      = r_h * (Wo_sp @ [ap,an])                 # OUTPUT PROJECTION   <<< [CE] SUBSTRATE graded read
    (5) logits = head_w @ h + head_b                     # read-out            <<< [CE] SUBSTRATE graded read + bias pop

Two pieces were each validated in ISOLATION and are here CHAINED (wiring, NO new mechanism):
  [WK] `_wkv_graded_recurrent_state_derisk.GradedRecurrentState` — step (2) held on the real Izhikevich bridge as the
       slow recurrent-NMDA conductance `cp_conductance_g_nmda_recurrent` (a clean dual-exp leaky integral of a graded
       carrier drive), state_corr 0.793 6/6, reproduces the host next-word decision 0.797 THROUGH A HOST READ (finding
       2026-08-13-fluid-mouth-wkv-state-graded-conductance-integrator-GO). Its named next rung #2: "compose the whole
       host-free chain end-to-end — feed the substrate state into the graded Wo_sp@state into the graded head_w@h".
  [CE] `_wkv_mouth_endtoend_substrate_read_derisk.ComposedEndToEndRead` (composed_biaspop arm) — steps (4)+(5) as ONE
       substrate signed-graded pipeline (every matmul a cp_conductance_g_e/g_i read) + head_b as a tonic bias-input
       population, recov_argmax 0.9495 6/6 — but it took the STATE HOST-SIDE (finding 2026-08-13-fluid-mouth-endtoend-
       substrate-read-GO, residual #4: "the WKV recurrent STATE ... is host").

THIS LANE feeds [WK]'s SUBSTRATE state (calibrated `cst = scale*g_nmda_recurrent + off`, on the host-state scale) into
[CE]'s substrate read chain in place of the host `[ap,an]`. So the ENTIRE next-word path — state integration ->
projection -> read-out -> logits — is substrate end-to-end. Both calibrations are the ARCS' OWN fixed seed-42 values
(no new tuning): [WK]'s per-channel affine (scale/off, fit once on seed 42) maps g -> host-state scale; [CE]'s
proj_out_scale=0.30 + bias_scale=0.14 (fixed on seed 42) map the projection margin -> feature scale and head_b -> bias.

THE A/B DECOMPOSITION (3 arms per position, same eval set -> the deltas isolate WHERE composition costs):
    A  fullsub          : SUBSTRATE state -> SUBSTRATE reads       (THE deliverable — state->logits fully on substrate)
    B  hoststate_subread: HOST state      -> SUBSTRATE reads       (== [CE] composed_biaspop; isolates the READ chain)
    C  substate_hostread: SUBSTRATE state -> HOST read (Wo_sp/head) (== [WK] downstream; isolates the STATE fidelity)
   ref host state -> host read = the full host mouth (the CEILING; recov=1 by construction).
Roughly recov(A) ~ recov(B) x recov(C): does chaining the SUBSTRATE state into the SUBSTRATE reads hold near that
product (~0.75-0.85), or collapse? Headline = arm A recov_argmax + argmax_agree vs the full host mouth.

ANTI-CHEATS (arm A; each MUST collapse — brain-based, negatives load-bearing):
  * LESION THE STATE (zero the WKV carrier input -> state decays to ~0)   -> downstream chance (state drives the chain)
  * MEMORYLESS (reset the NMDA conductance every token)                   -> degrades (recurrence load-bearing)
  * LESION THE PROJECTION read stage (zero_state at the proj input)       -> chance
  * LESION THE READ-OUT read stage (zero_feat at the read-out input)      -> chance
  * SCRAMBLE (post-hoc pool->word relabel)                                -> chance
  * SIGNED shadow: signed argmax_agree > positive-only (inhibitory Wn load-bearing)
  * PROVENANCE: state read off cp_conductance_g_nmda_recurrent; winner off cp_conductance_g_e/g_i; head_b via a spiking
    synapse; host_rng_draws_on_read_path == 0; 0 host matmul on the margin/state.
  * 6 seeds 42/43/44/100/101/102 (smoke first); single fixed operating point (both arcs' seed-42 calibrations reused).

HONEST SCOPE: still host = the input projection Wv (BPTT weights; the LEARNING rule is the separate 2026-08-12 diagonal
e-prop GO), the r_h gate, the LN, the trained decay/Wo_sp/head weights, and two fixed unit-scalars (scale/off, proj_out_
scale, bias_scale). This lane moves the STATE INTEGRATION + PROJECTION + READ-OUT + head_b onto neurons/synapses/graded
conductances end-to-end. NOT "fully spiking" / NOT production-wired. Runner-only, default-off, NO `sim/` edit.

Run (smoke):   SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_full_substrate_pipeline_derisk \
                 --smoke --seeds 42
Run (6-seed):  SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_full_substrate_pipeline_derisk \
                 --seeds 42,43,44,100,101,102 \
                 --json research/findings/raw/_wkv_full_substrate_pipeline_6seed.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.runners._wkv_fewspike_read_derisk import (  # noqa: E402
    WKVReadout, _softmax, _native, _load_eval,
)
from research.runners._wkv_graded_recurrent_state_derisk import (  # noqa: E402
    GradedRecurrentState, _rates, _ref_advance, _fit_calib, _cal_state, _core_probe,
)
from research.runners._wkv_mouth_endtoend_substrate_read_derisk import (  # noqa: E402
    ComposedEndToEndRead, _build_proj, _build_read, ARMS,
)
from tools.lab import lever, void_if  # noqa: E402


def _host_read(ro, cst, tid):
    """Arm C: the HOST read of the SUBSTRATE state (== [WK] downstream). r_h + Wo_sp + head are host."""
    r_h = 1.0 / (1.0 + np.exp(-(ro.Wr @ ro._ln(ro.emb[tid]))))
    lg = ro.head_w @ (r_h * (ro.Wo_sp @ cst)) + ro.head_b
    if ro.unk_idx >= 0:
        lg = lg.copy(); lg[ro.unk_idx] = -1e30
    return lg


# ====================================================================================================================
# Eval — teacher-forced over held-out positions; reference = the FULL host mouth read ro.logits(ap_host,an_host,tid).
# ====================================================================================================================
def _eval(seed, ro, s_wk, scale, off, s_read, ev_ids, warmup, n_eval_pos, deep_lo, n_ac):
    D = ro.D; V = ro.V; chance = 1.0 / V
    # per-arm accumulators
    def _z():
        return dict(n=0, agree=0.0, agree_pos=0.0, mass_read=0.0, mass_ax=0.0,
                    deep_n=0, deep_agree=0.0, deep_nll_read=0.0, deep_nll_host=0.0)
    A = _z(); B = _z(); C = _z()
    pool_sp = 0.0; bias_sp = 0.0
    # bounded read-boundary anti-cheats on arm A (clean cst, read-flag lesions -> collapse)
    ac = dict(n=0, scr=0.0, zstate=0.0, zfeat=0.0)

    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        s_wk.reset_state(); ap_h = np.zeros(D); an_h = np.zeros(D)
        for t in range(len(ids) - 1):
            rate = _rates(ro, ids[t])
            g = s_wk.advance(rate)                                   # SUBSTRATE WKV state (persistent recurrence)
            cst = _cal_state(g, scale, off)                          # [2D] = [ap_sub, an_sub], host-state scale
            ap_s, an_s = cst[:D], cst[D:]
            ap_h, an_h = _ref_advance(ro, ap_h, an_h, ids[t])        # host reference state
            if t < warmup:
                continue
            tid = ids[t]; tgt = ids[t + 1]
            lg_h = ro.logits(ap_h, an_h, tid).copy()                 # FULL HOST MOUTH (the reference)
            if ro.unk_idx >= 0:
                lg_h[ro.unk_idx] = -1e30
            host_am = int(np.argmax(lg_h)); pfull = _softmax(lg_h)
            deep = (t >= warmup + deep_lo)

            # -- arm A: SUBSTRATE state -> SUBSTRATE reads (the deliverable) --
            rA = s_read.read_endtoend(ap_s, an_s, tid)
            winA = rA["win"]; winA_pos = rA["win_pos"]
            pool_sp += rA["pool_sp"]; bias_sp += rA["bias_sp"]
            # -- arm B: HOST state -> SUBSTRATE reads (== [CE]) --
            rB = s_read.read_endtoend(ap_h, an_h, tid)
            winB = rB["win"]; winB_pos = rB["win_pos"]
            # -- arm C: SUBSTRATE state -> HOST read (== [WK] downstream) --
            lg_c = _host_read(ro, cst, tid); winC = int(np.argmax(lg_c))

            for arm, win, win_pos in ((A, winA, winA_pos), (B, winB, winB_pos), (C, winC, winC)):
                arm["n"] += 1
                arm["agree"] += float(win == host_am)
                arm["agree_pos"] += float(win_pos == host_am)
                arm["mass_read"] += (pfull[win] if win >= 0 else 0.0)
                arm["mass_ax"] += pfull[host_am]
                if deep:
                    arm["deep_n"] += 1
                    arm["deep_agree"] += float(win == host_am)
                    arm["deep_nll_read"] += -math.log(max(pfull[win] if win >= 0 else 1e-12, 1e-12))
                    arm["deep_nll_host"] += -math.log(max(pfull[host_am], 1e-12))

            # bounded read-boundary anti-cheats on arm A (clean cst; lesion at the read boundary)
            if ac["n"] < n_ac:
                scr_perm = np.random.default_rng(seed * 83 + 3 + positions).permutation(V)
                ws = s_read.read_endtoend(ap_s, an_s, tid, scramble_perm=scr_perm)["win"]
                wzs = s_read.read_endtoend(ap_s, an_s, tid, zero_state=True)["win"]
                wzf = s_read.read_endtoend(ap_s, an_s, tid, zero_feat=True)["win"]
                ac["scr"] += float(ws == host_am)
                ac["zstate"] += float(wzs == host_am)
                ac["zfeat"] += float(wzf == host_am)
                ac["n"] += 1

            positions += 1
            if positions >= n_eval_pos:
                break
        if positions >= n_eval_pos:
            break

    void_if(A["n"] == 0, "no evaluable positions")

    # -- state-corruption anti-cheats on arm A (fresh substrate advance; zero-input + memoryless) --
    zi = ml = ac2_n = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        s_wk.reset_state(); ap2 = np.zeros(D); an2 = np.zeros(D)
        for t in range(min(len(ids) - 1, warmup + 25)):
            rate = _rates(ro, ids[t])
            g_zi = s_wk.advance(rate, zero_input=True)               # state input lesioned -> decays to ~0
            g_ml = s_wk.advance(rate, memoryless=True)               # reset conductance each token -> no persistence
            ap2, an2 = _ref_advance(ro, ap2, an2, ids[t])
            if t < warmup:
                continue
            tid = ids[t]
            lg_h = ro.logits(ap2, an2, tid).copy()
            if ro.unk_idx >= 0:
                lg_h[ro.unk_idx] = -1e30
            host_am = int(np.argmax(lg_h))
            cst_zi = _cal_state(g_zi, scale, off); cst_ml = _cal_state(g_ml, scale, off)
            wzi = s_read.read_endtoend(cst_zi[:D], cst_zi[D:], tid)["win"]
            wml = s_read.read_endtoend(cst_ml[:D], cst_ml[D:], tid)["win"]
            zi += int(wzi == host_am); ml += int(wml == host_am); ac2_n += 1
            if ac2_n >= n_ac:
                break
        if ac2_n >= n_ac:
            break

    def _fin(arm):
        n = max(1, arm["n"]); dn = max(1, arm["deep_n"])
        return dict(
            n=arm["n"],
            argmax_agree=round(arm["agree"] / n, 4),
            argmax_agree_positive_only=round(arm["agree_pos"] / n, 4),
            mass_read=round(arm["mass_read"] / n, 4),
            mass_argmax_ceiling=round(arm["mass_ax"] / n, 4),
            recov_argmax=round((arm["mass_read"] / n) / max(1e-9, arm["mass_ax"] / n), 4),
            deep_n=arm["deep_n"],
            deep_argmax_agree=round(arm["deep_agree"] / dn, 4),
            deep_nll_read=round(arm["deep_nll_read"] / dn, 4),
            deep_nll_host=round(arm["deep_nll_host"] / dn, 4),
            deep_nll_gap_to_host=round((arm["deep_nll_read"] - arm["deep_nll_host"]) / dn, 4),
        )

    mA = _fin(A); mB = _fin(B); mC = _fin(C)
    nac = max(1, ac["n"]); nac2 = max(1, ac2_n)
    m = {
        "seed": seed, "V": V, "D": D, "F": s_wk.F, "chance_1_over_v": round(chance, 6),
        "t_step": s_wk.t_step, "carrier_pop": s_wk.Cp,
        "hid_pop": s_read.Hp, "pop": s_read.P, "bias_scale": s_read.bias_scale,
        "proj_out_scale": s_read.proj_out_scale,
        "n_positions": A["n"],
        "mean_pool_spikes": round(pool_sp / max(1, A["n"]), 3),
        "mean_bias_spikes": round(bias_sp / max(1, A["n"]), 3),
        "host_rng_draws_on_read_path": int(s_read.n_host_rng_draws),
        # arm A (the deliverable)
        "fullsub": mA,
        # arm B (== [CE] read chain) + arm C (== [WK] state)
        "hoststate_subread": mB,
        "substate_hostread": mC,
        # arm-A anti-cheats
        "argmax_agree_scramble": round(ac["scr"] / nac, 4),
        "argmax_agree_zerostate": round(ac["zstate"] / nac, 4),
        "argmax_agree_zerofeat": round(ac["zfeat"] / nac, 4),
        "argmax_agree_zeroinput": round(zi / nac2, 4),
        "argmax_agree_memoryless": round(ml / nac2, 4),
        "n_anticheat": ac["n"], "n_anticheat_state": ac2_n,
    }
    lever("fullsub_argmax_vs_zeroinput", before=m["argmax_agree_zeroinput"],
          after=mA["argmax_agree"], required=False)
    lever("fullsub_argmax_vs_memoryless", before=m["argmax_agree_memoryless"],
          after=mA["argmax_agree"], required=False)
    lever("fullsub_recov_vs_hoststate_read", before=mB["recov_argmax"],
          after=mA["recov_argmax"], required=False)
    return m


def _scramble_at_chance(agree_scramble, chance, n):
    sigma = math.sqrt(max(chance * (1.0 - chance), 1e-12) / max(1, n))
    return agree_scramble <= chance + 3.0 * sigma


def _verdict(m):
    mA = m["fullsub"]; mB = m["hoststate_subread"]
    chance = m["chance_1_over_v"]; n = m["n_positions"]; nac = max(1, m["n_anticheat"])
    aa = mA["argmax_agree"]
    checks = {
        # the full substrate pipeline reproduces the host next-word mass NEAR the two-stage composition (not collapsed).
        "recov_argmax_ge_0.70": mA["recov_argmax"] >= 0.70,
        # composition penalty vs the host-state substrate read (== [CE]) bounded — the STATE substitution is the cost.
        "within_tol_of_hoststate": mA["recov_argmax"] >= mB["recov_argmax"] - 0.15,
        "argmax_agree_gt_10x_chance": aa > 10 * chance,
        # the SUBSTRATE state drives the whole chain (zero its input -> collapse; cache-immune).
        "state_drives_chain": aa - m["argmax_agree_zeroinput"] > 0.30,
        # recurrence load-bearing (reset the conductance every token -> degrade).
        "recurrence_load_bearing": aa - m["argmax_agree_memoryless"] > 0.15,
        # lesion either read stage -> collapse to <=1/3 of intact.
        "read_input_collapses": max(m["argmax_agree_zerostate"], m["argmax_agree_zerofeat"]) <= 0.34 * aa,
        "scramble_at_chance": _scramble_at_chance(m["argmax_agree_scramble"], chance, nac),
        # the inhibitory shadow (Wn) is load-bearing.
        "signed_beats_positive_only": aa > mA["argmax_agree_positive_only"],
        # provenance: 0 host draws on the read path; head_b via a spiking synapse.
        "provenance_no_host_draw": m["host_rng_draws_on_read_path"] == 0,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    m["checks"] = checks
    m["GO"] = bool(all(checks.values()))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--n-eval-pos", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--deep-lo", type=int, default=8)
    ap.add_argument("--n-anticheat", type=int, default=120)
    # [WK] substrate-state operating point (its seed-42-calibrated GO values)
    ap.add_argument("--t-step", type=int, default=40)
    ap.add_argument("--carrier-pop", type=int, default=24)
    ap.add_argument("--wk-drive-gain", type=float, default=40.0)
    ap.add_argument("--wk-drive-bias", type=float, default=40.0)
    ap.add_argument("--wk-syn-w", type=float, default=2.0)
    ap.add_argument("--wk-ou-std", type=float, default=60.0)
    # [CE] substrate read-chain operating point (its seed-42-calibrated GO values)
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--read-window", type=int, default=150)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--hb-k", type=float, default=0.5)
    ap.add_argument("--settle-frac", type=float, default=0.2)
    ap.add_argument("--proj-drive-gain", type=float, default=120.0)
    ap.add_argument("--proj-syn-scale", type=float, default=12.0)
    ap.add_argument("--proj-ratio", type=float, default=0.5)
    ap.add_argument("--proj-out-scale", type=float, default=0.30)
    ap.add_argument("--bias-scale", type=float, default=0.14)
    ap.add_argument("--n-bias", type=int, default=16)
    ap.add_argument("--bias-drive-pA", type=float, default=160.0)
    ap.add_argument("--probe", action="store_true", help="[WK] core state-realization probe only")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_full_substrate_pipeline.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_sentences = 2000; args.n_eval_pos = min(args.n_eval_pos, 60); args.n_anticheat = min(args.n_anticheat, 40)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    results = []
    calib = None
    t_all = time.time()
    for seed in seeds:
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        ro = WKVReadout(ckpt)
        s_wk = GradedRecurrentState(ro.D, seed, t_step=args.t_step, carrier_pop=args.carrier_pop,
                                    ou_std=args.wk_ou_std, drive_gain=args.wk_drive_gain,
                                    drive_bias=args.wk_drive_bias, syn_w=args.wk_syn_w, ssm_decay=ro.decay)
        if args.probe:
            mean_c, med_c, lin = _core_probe(s_wk, ro.decay, n_tokens=150, seed=seed)
            print(f"[probe seed {seed}] core state_corr mean={mean_c:.4f} med={med_c:.4f} input_lin={lin:.4f}",
                  flush=True)
            results.append({"seed": seed, "arm": "core_probe", "state_corr": round(mean_c, 4),
                            "input_lin_corr": round(lin, 4)})
            continue
        ev_ids, vocab = _load_eval(ro, args.corpus, args.n_sentences, seed, max(64, args.n_eval_pos // 6))
        # [WK] state calibration: fit the per-channel affine ONCE on the first seed, FIX for the rest (unseen-seed test)
        if calib is None:
            calib = _fit_calib(s_wk, ro, ev_ids, args.warmup, min(600, args.n_eval_pos), ro.decay)
            print(f"[calib-state on seed {seed}] scale.mean={calib[0].mean():.4f}", flush=True)
        scale, off = calib
        # [CE] substrate read chain (composed_biaspop: substrate proj feat + tonic-bias head_b), fixed seed-42 calib
        proj = _build_proj(ro, seed, args)
        s_read = _build_read(ro, seed, "composed_biaspop", args, proj)
        s_read._arm = "composed_biaspop"

        t0 = time.time()
        m = _verdict(_eval(seed, ro, s_wk, scale, off, s_read, ev_ids, args.warmup, args.n_eval_pos,
                           args.deep_lo, args.n_anticheat))
        m["secs"] = round(time.time() - t0, 1)
        results.append(m)
        mA = m["fullsub"]; mB = m["hoststate_subread"]; mC = m["substate_hostread"]
        print(f"[seed {seed}] fullsub recov={mA['recov_argmax']:.4f} agree={mA['argmax_agree']:.4f}"
              f">pos{mA['argmax_agree_positive_only']:.3f} deep_agree={mA['deep_argmax_agree']:.4f}"
              f"(n={mA['deep_n']}) | B[host-state]={mB['recov_argmax']:.4f} C[host-read]={mC['recov_argmax']:.4f}"
              f" | zin={m['argmax_agree_zeroinput']:.3f} mless={m['argmax_agree_memoryless']:.3f}"
              f" zstate={m['argmax_agree_zerostate']:.3f} zfeat={m['argmax_agree_zerofeat']:.3f}"
              f" scr={m['argmax_agree_scramble']:.3f} pool_spk={m['mean_pool_spikes']:.2f}"
              f" GO={m['GO']} ({sum(m['checks'].values())}/{len(m['checks'])}) ({m['secs']}s)", flush=True)
        if not m["GO"]:
            print(f"    checks: {json.dumps(m['checks'])}", flush=True)

    # summary
    rows = [r for r in results if "fullsub" in r]
    summary = {}
    if rows:
        summary = {
            "n_seeds": len(rows),
            "go_count": int(sum(1 for r in rows if r.get("GO"))),
            "fullsub_recov_mean": round(float(np.mean([r["fullsub"]["recov_argmax"] for r in rows])), 4),
            "fullsub_recov_min": round(float(np.min([r["fullsub"]["recov_argmax"] for r in rows])), 4),
            "fullsub_argmax_agree_mean": round(float(np.mean([r["fullsub"]["argmax_agree"] for r in rows])), 4),
            "fullsub_deep_agree_mean": round(float(np.mean([r["fullsub"]["deep_argmax_agree"] for r in rows])), 4),
            "hoststate_subread_recov_mean": round(float(np.mean([r["hoststate_subread"]["recov_argmax"] for r in rows])), 4),
            "substate_hostread_recov_mean": round(float(np.mean([r["substate_hostread"]["recov_argmax"] for r in rows])), 4),
            "zeroinput_mean": round(float(np.mean([r["argmax_agree_zeroinput"] for r in rows])), 4),
            "memoryless_mean": round(float(np.mean([r["argmax_agree_memoryless"] for r in rows])), 4),
            "zerostate_mean": round(float(np.mean([r["argmax_agree_zerostate"] for r in rows])), 4),
            "zerofeat_mean": round(float(np.mean([r["argmax_agree_zerofeat"] for r in rows])), 4),
            "scramble_mean": round(float(np.mean([r["argmax_agree_scramble"] for r in rows])), 4),
        }
    out = {"results": _native(results), "summary": _native(summary), "seeds": seeds,
           "n_eval_pos": args.n_eval_pos, "plasticity_off": True,
           "backend": os.environ.get("SIM_BACKEND", "numpy"), "elapsed_s": round(time.time() - t_all, 1),
           "argv": sys.argv}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))
    if summary:
        print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} rows -> {args.json} ({time.time()-t_all:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
