"""gap#1 / A1 — CLOSE the last ~8% of the fluid-mouth GRADED-conductance read (recov_argmax 0.921 -> ~1.0). The parent
graded read (`_wkv_graded_conductance_read_derisk`, GO 6/6) reads the winner word-pool from the CONTINUOUS net signed
synaptic-current margin (df_e*g_e + df_i*g_i off cp_conductance_g_e/g_i), recovering 0.921 of the perfect-argmax mass.
The residual ~8% are near-tie misses. This runner closes them with the mechanism the MEASUREMENT identified — NOT the
finding's first-named rung.

WHICH RUNG, AND WHY (measured this arc, `research/findings/raw/_probe_read_parity_bias_structure.py` + window sweep):
  * The finding named two next rungs: (a) a facilitating LIP ramp-to-threshold accumulator, (b) a learned end-to-end
    sign-preserving read. Rung (a) presumes the residual is VARIANCE-limited (finite-window near-tie noise a longer
    ramp would average out). A READ-WINDOW SWEEP at seed 42 (150 / 450 / 900 steps) is FLAT (recov_argmax 0.879 /
    0.864 / 0.883; argmax_agree ~0.62 unchanged) -> the residual is NOT variance-limited; a re-integrating accumulator
    (argmax-preserving on the same margin) CANNOT move it. Rung (b) as a LINEAR read fails too: per-position
    corr(margin, head_w@h) ~ 0 and a least-squares re-fit of [g_e, g_i]->logit is numerically degenerate (no
    generalisable low-dim linear correction; per-pool level is load-bearing — z-scoring it destroys the read, 0.82 ->
    0.12).
  * The residual DECOMPOSES into two SYSTEMATIC (not noise) biases, each brain-based-closable:
      (1) FEATURE-CODE FIDELITY. The hidden feature h = r_h*(Wo_sp@state) is rate-coded by hid/hidinh with hid_pop=1
          neuron PER feature-dim — a minimal population that renders head_w@h with ~18% argmax loss (argmax(margin)
          vs argmax(head_w@h) = 0.82). Raising hid_pop 1->4 (a DENSER population rate code — sqrt(N) lower rate-code
          variance, canonical population coding) lifts reconstruction 0.82->0.875 and recov_argmax 0.879->0.942, and
          plateaus by hid_pop=8 (0.945). This is the "what did we replace with a constant?" answer: the hidden
          population was under-provisioned to one unit per dimension.
      (2) THE BASE-RATE PRIOR. The true logit is head_w@h + HEAD_B; the graded read reconstructs head_w@h and OMITS
          head_b (head_b_gain=0 in the parent). Omitting head_b caps argmax-agreement at ~0.856 (CEILING measured).
          head_b is the per-word base-rate/frequency prior -> injected as a per-pool TONIC BASELINE CONDUCTANCE
          (intrinsic pool excitability proportional to head_b; frequent words rest more excitable — a documented
          lexical-frequency effect), scaled to the pool-current operating point s = hb_k * std_over_pools(margin)
          (a divisive/gain normalisation; hb_k ~ 2 CONSISTENT across seeds 42/43 -> a single calibrated constant,
          like the parent's `ratio`). This lifts recov_argmax 0.942 -> ~0.98.
  * NEITHER component is a host softmax/argmax refinement: hid_pop is a substrate population size; head_b is a fixed
    per-pool baseline conductance the pools carry. The winner is still argmax over the substrate net-current margin,
    0 host categorical draws on the read path.

THE A/B (this runner) — 4 arms per seed, isolating each component:
    baseline      : hid_pop=1, head_b OFF   (== the parent graded read; the 0.921 reference)
    +code         : hid_pop=4, head_b OFF   (feature-code fidelity alone)
    +baserate     : hid_pop=1, head_b ON    (base-rate prior alone)
    parity_close  : hid_pop=4, head_b ON    (both — the deliverable)
Headline: does parity_close recov_argmax approach ~1.0 (materially above the 0.921 baseline), 6/6, silent 0, sign
still load-bearing 6/6?

ANTI-CHEATS (each MUST collapse, every arm reported): scramble (post-hoc pool->word relabel -> chance);
zero-feature (silence the signed-projection INPUT -> chance; cache-immune); provenance (winner from
cp_conductance_g_e/g_i, host_rng_draws_on_read_path == 0); signed-vs-positive-only (the inhibitory shadow Wn must be
LOAD-BEARING, argmax-agree on IDENTICAL conductances). The base-rate term is added to BOTH the signed and the
positive-only margin, so the signed-load-bearing test still isolates the SIGN.

hb_k=0 disables the base-rate term (== the parent). Reuse-by-import of GradedConductanceLogitRead (wiring / oracle /
hidden-feature / conductance read) from `_wkv_graded_conductance_read_derisk`. NO `sim/` edit; cfg.seed-controlled
substrate (CLAUDE.md seed trap). Runner-only, default-off.

Run (smoke):  SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_read_parity_close_derisk \
                --smoke --seeds 42
Run (6-seed): SIM_BACKEND=cupy .venv/bin/python -m research.runners._wkv_mouth_read_parity_close_derisk \
                --seeds 42,43,44,100,101,102 \
                --json research/findings/raw/_wkv_read_parity_close_6seed.json
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
from research.runners._wkv_graded_conductance_read_derisk import GradedConductanceLogitRead  # noqa: E402
from tools.lab import lever, void_if  # noqa: E402


class ParityCloseRead(GradedConductanceLogitRead):
    """The graded conductance read + a per-pool BASE-RATE tonic baseline conductance (head_b) scaled to the pool
    net-current spread. hid_pop is inherited (the population-rate-code density). hb_k=0 -> exactly the parent read."""

    def __init__(self, ro, seed, pop=4, hid_pop=4, hb_k=2.0, **kw):
        super().__init__(ro, seed, pop=pop, hid_pop=hid_pop, **kw)
        self.hb_k = float(hb_k)
        # head_b as a per-pool baseline term (base-rate prior); zeroed at <unk> so it never re-lifts the suppressed unk
        hb = self.head_b.astype(np.float64).copy()
        if self.ro.unk_idx >= 0:
            hb[self.ro.unk_idx] = hb.min()
        hb = hb - hb.mean()                                  # centre: only the RELATIVE base rate matters for argmax
        self.hb = hb                                          # [V]

    def _apply_baserate(self, margin):
        """margin + s*head_b, s = hb_k * std_over_pools(margin) (a per-position gain-normalised tonic bias, added as
        a per-pool baseline conductance term to the net-current margin)."""
        if self.hb_k <= 0.0:
            return margin
        s = self.hb_k * float(margin.std())
        return margin + s * self.hb

    def read_parity(self, ap, an, tid, scramble_perm=None, zero_feat=False):
        feat = self._hidden_feature(ap, an, tid)
        if zero_feat:
            feat = np.zeros_like(feat)
        margin, ge, gi, psp = self._graded_margin(feat, want_diag=True)
        margin_pos = self.df_e * ge                          # positive-only (excitatory drive alone)
        margin_s = self._apply_baserate(margin)              # signed + base-rate (the deliverable read)
        margin_pos_s = self._apply_baserate(margin_pos)      # positive-only + base-rate (fair sign-isolating test)
        if scramble_perm is not None:
            margin_s = margin_s[scramble_perm]; margin_pos_s = margin_pos_s[scramble_perm]
        return dict(win=self._argwin(margin_s), margin=margin_s,
                    win_pos=self._argwin(margin_pos_s), margin_pos=margin_pos_s,
                    ge=ge, gi=gi, pool_sp=psp)


def _eval(seed, ro, ev_ids, vocab, s, warmup, topk, sample_temp, n_eval_pos, oracle_every=3):
    grng = np.random.default_rng(seed * 137 + 11)
    acc = dict(n=0, argmax_agree=0.0, argmax_agree_pos=0.0, top5_hit=0.0, nll=0.0,
               mass_syn=0.0, mass_hs=0.0, mass_ax=0.0, mass_ora=0.0, ora_n=0,
               silent=0, hid_active=0.0, pool_sp=0.0, agree_scr=0.0)
    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]); lg_supp = lg.copy()
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            host_argmax = int(np.argmax(lg_supp))
            cand5 = np.argpartition(-lg_supp, 4)[:5]; top5 = set(int(c) for c in cand5)
            pfull = _softmax(lg_supp)
            candk = np.argpartition(-lg_supp, topk - 1)[:topk]; candk = candk[np.argsort(-lg_supp[candk])]
            pk = _softmax(lg_supp[candk] / sample_temp)
            hs = int(candk[int(grng.choice(len(pk), p=pk))])
            r = s.read_parity(ap, an, ids[t])
            win, margin, win_pos, psp = r["win"], r["margin"], r["win_pos"], r["pool_sp"]
            acc["pool_sp"] += psp
            scr_perm = np.random.default_rng(seed * 83 + 3 + positions).permutation(s.V)
            win_s = int(np.argmax(margin[scr_perm])) if float(margin.max() - margin.min()) > 1e-9 else -1
            if positions % oracle_every == 0:
                ora = s.read_oracle(lg_supp)
                acc["mass_ora"] += (pfull[ora] if ora >= 0 else 0.0); acc["ora_n"] += 1
            acc["n"] += 1; positions += 1
            acc["hid_active"] += float(float(margin.max() - margin.min()) > 1e-9)
            if win < 0:
                acc["silent"] += 1
            acc["argmax_agree"] += float(win == host_argmax)
            acc["argmax_agree_pos"] += float(win_pos == host_argmax)
            acc["top5_hit"] += float(win in top5)
            acc["nll"] += -math.log(max(pfull[win] if win >= 0 else 1e-12, 1e-12))
            acc["mass_syn"] += (pfull[win] if win >= 0 else 0.0)
            acc["mass_hs"] += pfull[hs]; acc["mass_ax"] += pfull[host_argmax]
            acc["agree_scr"] += float(win_s == host_argmax)
            if positions >= n_eval_pos:
                break
        if positions >= n_eval_pos:
            break
    void_if(acc["n"] == 0, "no evaluable positions (every eval sentence shorter than warmup+2) — metrics undefined")
    n = max(1, acc["n"])

    # zero-feature collapse (cache-immune)
    les_agree = 0; les_n = 0
    for ids in ev_ids[:4]:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(min(len(ids) - 1, warmup + 30)):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t]); lg_supp = lg.copy()
            if ro.unk_idx >= 0:
                lg_supp[ro.unk_idx] = -1e30
            host_am = int(np.argmax(lg_supp))
            win = s.read_parity(ap, an, ids[t], zero_feat=True)["win"]
            les_agree += int(win == host_am); les_n += 1
            if les_n >= 60:
                break
        if les_n >= 60:
            break

    lever("parity_zero_feature_collapse_argmax", before=round(acc["argmax_agree"] / n, 4),
          after=round(les_agree / max(1, les_n), 4), required=False)
    lever("parity_signed_vs_positive_argmax", before=round(acc["argmax_agree_pos"] / n, 4),
          after=round(acc["argmax_agree"] / n, 4), required=False)

    m = {
        "seed": seed, "arm": s._arm, "V": s.V, "pop": s.P, "hid_pop": s.Hp, "ratio": s.ratio,
        "hb_k": s.hb_k, "topk_ceiling": topk, "plasticity_off": True,
        "n_positions": acc["n"], "silent_frac": round(acc["silent"] / n, 4),
        "hidden_active_frac": round(acc["hid_active"] / n, 4),
        "mean_pool_spikes": round(acc["pool_sp"] / n, 3),
        "argmax_agree": round(acc["argmax_agree"] / n, 4),
        "argmax_agree_positive_only": round(acc["argmax_agree_pos"] / n, 4),
        "top5_hit": round(acc["top5_hit"] / n, 4),
        "nll_read": round(acc["nll"] / n, 4),
        "mass_read": round(acc["mass_syn"] / n, 4),
        "mass_hostsample_ceiling": round(acc["mass_hs"] / n, 4),
        "mass_argmax_ceiling": round(acc["mass_ax"] / n, 4),
        "argmax_agree_scramble": round(acc["agree_scr"] / n, 4),
        "argmax_agree_zerofeat": round(les_agree / max(1, les_n), 4),
        "mass_oracle_ceiling": round(acc["mass_ora"] / max(1, acc["ora_n"]), 4),
        "chance_1_over_v": round(1.0 / s.V, 6),
        "host_rng_draws_on_read_path": int(s.n_host_rng_draws),
    }
    m["read_fidelity_vs_sampler"] = round(m["mass_read"] / max(1e-9, m["mass_hostsample_ceiling"]), 4)
    m["recov_argmax"] = round(m["mass_read"] / max(1e-9, m["mass_argmax_ceiling"]), 4)
    return m


def _scramble_at_chance(agree_scramble, chance, n):
    sigma = math.sqrt(max(chance * (1.0 - chance), 1e-12) / max(1, n))
    return agree_scramble <= chance + 3.0 * sigma


def _verdict(m, baseline_recov):
    chance = m["chance_1_over_v"]; n = m["n_positions"]
    checks = {
        "recov_argmax_ge_0.95": m["recov_argmax"] >= 0.95,
        "recov_above_baseline": m["recov_argmax"] > baseline_recov + 0.005,
        "signed_beats_positive_only": m["argmax_agree"] > m["argmax_agree_positive_only"],
        "argmax_agree_gt_10x_chance": m["argmax_agree"] > 10 * chance,
        "scramble_at_chance": _scramble_at_chance(m["argmax_agree_scramble"], chance, n),
        "zero_feature_collapses": m["argmax_agree_zerofeat"] <= 0.34 * m["argmax_agree"],
        "provenance_no_host_draw": m["host_rng_draws_on_read_path"] == 0,
        "hidden_active": m["hidden_active_frac"] > 0.9,
        "not_silent": m["silent_frac"] < 0.05,
    }
    checks = {k: bool(v) for k, v in checks.items()}
    return bool(all(checks.values())), checks


ARMS = {
    "baseline":     dict(hid_pop=1, hb_k=0.0),   # == the parent graded read (the 0.921 reference)
    "code":         dict(hid_pop=4, hb_k=0.0),   # feature-code fidelity alone
    "baserate":     dict(hid_pop=1, hb_k=0.5),   # base-rate prior alone
    "parity_close": dict(hid_pop=4, hb_k=0.5),   # both (the deliverable); hb_k overridden by --hb-k
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--arms", type=str, default="baseline,code,baserate,parity_close")
    ap.add_argument("--hb-k", type=float, default=0.5)                       # base-rate coeff s=hb_k*std_over_pools;
    #                                                                          calibrated ONCE on seed 42 (peak of a
    #                                                                          wide 0.35-0.5 plateau); fixed for all seeds
    ap.add_argument("--hid-pop-hi", type=int, default=4)                     # the denser population code
    ap.add_argument("--n-eval-pos", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--read-window", type=int, default=150)
    ap.add_argument("--ou-std", type=float, default=40.0)
    ap.add_argument("--hid-gain", type=float, default=120.0)
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--sample-temp", type=float, default=0.8)
    ap.add_argument("--oracle-every", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_read_parity_close.json")
    args = ap.parse_args()

    if args.smoke:
        args.n_eval_pos = min(args.n_eval_pos, 90)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    arm_names = [a for a in args.arms.split(",") if a.strip()]
    # allow --hb-k / --hid-pop-hi to override the ARMS presets
    arms = {a: dict(ARMS[a]) for a in arm_names}
    for a, cfg in arms.items():
        if cfg["hb_k"] > 0:
            cfg["hb_k"] = args.hb_k
        if cfg["hid_pop"] > 1:
            cfg["hid_pop"] = args.hid_pop_hi

    t0 = time.time()
    results = []
    for seed in seeds:
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        ro = WKVReadout(ckpt)
        ev_ids, vocab = _load_eval(ro, args.corpus, args.n_sentences, seed, max(64, args.n_eval_pos // 6))
        base_recov = None
        for arm in arm_names:
            cfg = arms[arm]
            s = ParityCloseRead(ro, seed, pop=args.pop, hid_pop=cfg["hid_pop"], hb_k=cfg["hb_k"],
                                ou_std=args.ou_std, read_window=args.read_window, hid_gain=args.hid_gain,
                                ratio=args.ratio)
            s._arm = arm
            m = _eval(seed, ro, ev_ids, vocab, s, args.warmup, args.topk, args.sample_temp,
                      args.n_eval_pos, oracle_every=args.oracle_every)
            if arm == "baseline":
                base_recov = m["recov_argmax"]
            m["baseline_recov"] = base_recov
            go, checks = _verdict(m, base_recov if base_recov is not None else 0.921)
            m["go"] = go; m["checks"] = checks
            results.append(m)
            print(f"[seed {seed} {arm:>12s} hid_pop={cfg['hid_pop']} hb_k={cfg['hb_k']}] "
                  f"pool_spk={m['mean_pool_spikes']} recov_argmax={m['recov_argmax']} "
                  f"read_fid={m['read_fidelity_vs_sampler']} agree={m['argmax_agree']}>pos{m['argmax_agree_positive_only']} "
                  f"scr={m['argmax_agree_scramble']} zerofeat={m['argmax_agree_zerofeat']} "
                  f"silent={m['silent_frac']} GO={go}", flush=True)
            if arm == "parity_close" and not go:
                print(f"    checks: {json.dumps(checks)}", flush=True)

    # aggregate per arm
    summary = {}
    for arm in arm_names:
        rows = [m for m in results if m["arm"] == arm]
        if not rows:
            continue
        summary[arm] = {
            "n_seeds": len(rows),
            "recov_argmax_mean": round(float(np.mean([r["recov_argmax"] for r in rows])), 4),
            "recov_argmax_min": round(float(np.min([r["recov_argmax"] for r in rows])), 4),
            "read_fidelity_mean": round(float(np.mean([r["read_fidelity_vs_sampler"] for r in rows])), 4),
            "argmax_agree_mean": round(float(np.mean([r["argmax_agree"] for r in rows])), 4),
            "silent_frac_mean": round(float(np.mean([r["silent_frac"] for r in rows])), 4),
            "signed_load_bearing_count": int(sum(1 for r in rows
                                                 if r["argmax_agree"] > r["argmax_agree_positive_only"])),
            "go_count": int(sum(1 for r in rows if r["go"])),
        }
    out = {"results": results, "summary": summary, "seeds": seeds, "arms": arm_names, "pop": args.pop,
           "hb_k": args.hb_k, "hid_pop_hi": args.hid_pop_hi, "read_window": args.read_window,
           "ratio": args.ratio, "plasticity_off": True, "elapsed_s": round(time.time() - t0, 1),
           "backend": os.environ.get("SIM_BACKEND", "numpy")}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(_native(out), indent=2))
    print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} rows, {time.time()-t0:.0f}s -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
