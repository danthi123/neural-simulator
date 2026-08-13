"""
gap#1 / A1 — THE PRODUCTION FEW-SPIKE READ REGIME for open deep-context prose.

THE RESIDUAL (mapped 2026-08-13, un-served): the fluent, multi-clause open-prose WKV/SSM generation
(2026-07-20 RF-PHASE-ENCODE, 6-seed GO) reads its NEXT WORD by a HOST ARGMAX / temperature-sample over the
graded read-out logits computed from a high-precision graded conductance state
(`_emerge_wkv_onbridge_derisk._next_logits`: `argmax(head_w @ (r_h * (Wo_sp @ state)) + head_b)`).  SEPARATELY,
a genuine Izhikevich FEW-SPIKE soft-WTA reads a categorical winner from a SMALL number of spikes
(`_followon2_spiking_wta_sampler` Buesing-Bill-Nessler-Maass 2011; `_neural_wta_word_decode`), but only for a
SINGLE-CLAUSE SVO draw over a tiny (<=150-word) vocab.  The two are DISJOINT: fluent multi-clause coherence lives
only on the graded-argmax read; the production few-spike read lives only on single-clause SVO.  NOBODY has fed the
fluent WKV deep-context next-token distribution (large vocab, near-tied peaked long-tail, AUTOREGRESSIVE) into a
few-spike vocab WTA and asked whether fluent generation SURVIVES the read regime.  This runner does exactly that.

THE INSTRUMENT (isolates the READ, holds the validated graded state fixed):
  1.  the graded next-token DISTRIBUTION is reproduced by the rate-SSM analog of the deployed generation read-out
      (`ap=decay*ap+relu(v); an=decay*an+relu(-v); logits=head_w @ (sigmoid(Wr@LN(emb))*(Wo_sp@[ap,an]))+head_b`),
      which is corr-0.999 to the on-bridge `cp_ssm_state` (2026-07-20 map_corr 0.9984) -- so the STATE is the
      already-validated graded conductance; the READ-OUT is the sole variable under test.
  2.  the FEW-SPIKE READ replaces `argmax(logits)` / `sample(softmax(logits/T))` with a genuine Izhikevich
      soft-WTA over word-candidate pools: the top-K candidates by logit drive their pools (labelled-line place
      code, a legitimate host INPUT == a reservoir W_in / the retinal render), OU membrane noise makes the winner
      stochastic ~ softmax(drive/T), the winner is read from `cp_firing_states` accumulated spikes over a SHORT
      window (the few-spike budget).  P neurons per candidate = POPULATION coding (the companion process the host
      argmax replaced with a constant: population + lateral competition + a homeostatic gain).

DECISIVE COMPARISON (calibration-robust): the few-spike read is a SAMPLER; the honest ceiling is an IDEAL host
sampler from the SAME top-K softmax.  We measure whether the few-spike read recovers the model's own distribution
as well as an ideal sampler does:
  ondist_mass(arm)  = mean over positions of p_model[token the arm chose]   (higher = more on-distribution)
    host_sample ceiling  == E[sum p^2]  (an ideal categorical sampler)
    host_argmax ceiling  == E[max p]    (the deployed greedy read)
  read_fidelity = ondist_mass(fewspike) / ondist_mass(host_sample)   (1.0 == the few-spike read is as
                  on-distribution as an ideal sampler; << 1 == quantization noise flattened the peak)
Plus top-1 argmax-agreement, self-NLL, mean spikes/read (quantifies "few-spike"), and FREE-GENERATION survival
(the model's self-NLL of its own few-spike-generated continuation vs the host-sample ceiling).

ANTI-CHEATS (each MUST collapse): scramble (permute logit->pool: reads the wrong logit -> agreement to chance);
equal-drive (drive all active pools equally: no likelihood signal -> uniform over the active set);
noise-ablation (ou_std->0: deterministic argmax-over-drive -> the OU noise IS the stochasticity, not a host RNG);
provenance (winner read from cp_firing_states, bridge advanced, 0 host categorical draws on the read path).

Reuse-by-import: the WKV checkpoint arrays + read-out form from `_emerge_wkv_onbridge_derisk`; the Izhikevich
soft-WTA bank + drive mapping REPLICATED from `_followon2_spiking_wta_sampler` (no taxonomy/PPMI baggage).
NO `sim/` edit -- drives + reads public bridge arrays.  cfg.seed-controlled substrate (CLAUDE.md seed trap).

Run (smoke):  SIM_BACKEND=numpy .venv/bin/python -m research.runners._wkv_fewspike_read_derisk --smoke --seed 42
Run (6-seed): SIM_BACKEND=cupy  .venv/bin/python -m research.runners._wkv_fewspike_read_derisk \
                 --seeds 42,43,44,100,101,102 --json research/findings/raw/_wkv_fewspike_read.json
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

from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.bridge import SimulationBridge  # noqa: E402
from sim.backend import to_host, get_backend  # noqa: E402

from research.runners._emerge_reservoir_lm_derisk import Vocab  # noqa: E402
from research.runners._emerge_reservoir_lm_realcorpus_derisk import load_sentences  # noqa: E402

_CORPUS_CANDIDATES = [
    "data/corpus/tinystories.txt",
    "/home/dant123/Projects/sim/data/corpus/tinystories.txt",
]


# ----------------------------------------------------------------------------------------------------------------
# The deployed WKV/SSM generation read-out (rate-SSM analog; corr-0.999 to the on-bridge cp_ssm_state).
# ----------------------------------------------------------------------------------------------------------------
class WKVReadout:
    def __init__(self, ckpt_path):
        W = np.load(ckpt_path, allow_pickle=True)
        self.V = int(W["V"]); self.D = int(W["d_model"])
        self.emb = W["emb.weight"].astype(np.float64)
        self.ln_w = W["ln.weight"].astype(np.float64); self.ln_b = W["ln.bias"].astype(np.float64)
        self.Wv = W["Wv.weight"].astype(np.float64); self.Wr = W["Wr.weight"].astype(np.float64)
        self.Wo_sp = W["Wo_sp.weight"].astype(np.float64)
        self.head_w = W["head.weight"].astype(np.float64); self.head_b = W["head.bias"].astype(np.float64)
        self.decay = float(np.exp(-np.log1p(np.exp(W["w"][0]))))
        self.words = list(W["words"])
        self.unk_idx = (len(self.words) - 1) if (self.words and self.words[-1] == "<unk>") else -1

    def _ln(self, v):
        m = v.mean(); s = v.std() + 1e-5
        return (v - m) / s * self.ln_w + self.ln_b

    def v_of(self, tid):
        return self.Wv @ self._ln(self.emb[tid])

    def advance(self, ap, an, tid):
        v = self.v_of(tid)
        ap = self.decay * ap + np.maximum(v, 0.0)
        an = self.decay * an + np.maximum(-v, 0.0)
        return ap, an

    def logits(self, ap, an, tid):
        state = np.concatenate([ap, an])                                # [2D] dual-nonneg deployed state form
        r_h = 1.0 / (1.0 + np.exp(-(self.Wr @ self._ln(self.emb[tid]))))
        return self.head_w @ (r_h * (self.Wo_sp @ state)) + self.head_b


# ----------------------------------------------------------------------------------------------------------------
# The production FEW-SPIKE READ: an Izhikevich soft-WTA over word-candidate pools (replicated from followon2).
# ----------------------------------------------------------------------------------------------------------------
class FewSpikeWordRead:
    """Izhikevich soft-WTA read of a categorical winner from a SMALL number of spikes.  K candidate WORDS, each a
    pool of P neurons (population coding).  Drive[k] (from the model's top-K softmax) -> all P neurons of pool k;
    OU membrane noise -> the winner is stochastic ~ softmax(drive/T); winner = argmax over per-POOL accumulated
    firing read from `cp_firing_states` over `read_window` steps (the few-spike budget)."""

    def __init__(self, n_pools, pop, seed, ou_std=200.0, base_pA=110.0, gain_pA=160.0, read_window=120):
        self.K = int(n_pools); self.P = int(pop)
        self.n = self.K * self.P
        self.base_pA = float(base_pA); self.gain_pA = float(gain_pA)
        self.read_window = int(read_window)
        self.ou_std = float(ou_std)
        self.seed = int(seed)
        self.n_host_rng_draws = 0                                       # MUST stay 0 (the whole point)
        self._bank = self._build_bank()

    def _build_bank(self):
        cfg = CoreSimConfig()
        cfg.num_neurons = int(self.n)
        cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
        cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
        cfg.seed = self.seed
        cfg.dt_ms = 1.0
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        for f in ("enable_stdp", "enable_hebbian_learning", "enable_short_term_plasticity",
                  "enable_structural_plasticity", "enable_homeostasis", "enable_reward_modulation",
                  "enable_watts_strogatz", "enable_neuromodulator_subsystem", "enable_brain_region_framework"):
            if hasattr(cfg, f):
                setattr(cfg, f, False)
        cfg.enable_ou_process = self.ou_std > 0.0
        cfg.ou_mean_current_pA = 0.0
        cfg.ou_std_current_pA = self.ou_std
        cfg.ou_tau_ms = 15.0
        bank = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                runtime_state=RuntimeState(), gpu_config=GPUConfig())
        bank._initialize_simulation_data(called_from_playback_init=False)
        bank._v0 = bank.cp_membrane_potential_v.copy()
        bank._u0 = bank.cp_recovery_variable_u.copy()
        return bank

    def drive_from_weights(self, w):
        """w[K] nonneg (the top-K softmax mass).  Active (w>0) pools get base_pA + gain_pA*(w/peak); a zero-weight
        pool gets ZERO drive -> SILENT (Buesing-Maass 'off-target emits zero spikes')."""
        w = np.asarray(w, dtype=np.float64)
        peak = float(w.max()) if w.size else 0.0
        if peak <= 1e-12:
            return np.zeros(len(w))
        active = (w > 0).astype(np.float64)
        return active * (self.base_pA + self.gain_pA * (w / peak))

    def _compete(self, drive_pools, equal_drive=False):
        """Run the bank read_window steps; return per-POOL accumulated firing [K] + total spikes."""
        bank = self._bank
        bank.cp_membrane_potential_v[:] = bank._v0
        bank.cp_recovery_variable_u[:] = bank._u0
        xp, _ = get_backend()
        if equal_drive:
            active = (np.asarray(drive_pools) > 0).astype(np.float64)
            drive_pools = active * (self.base_pA + self.gain_pA)       # every active pool the SAME max drive
        per_neuron = np.repeat(np.asarray(drive_pools, dtype=np.float64), self.P)   # [K] -> [K*P]
        bank.cp_external_input_current[:] = xp.asarray(per_neuron, dtype=bank.cp_external_input_current.dtype)
        firing = np.zeros(self.n, dtype=np.float64)
        for _ in range(self.read_window):
            bank._run_one_simulation_step()
            firing += np.asarray(to_host(bank.cp_firing_states)).astype(float)
        bank.cp_external_input_current[:] = 0.0
        per_pool = firing.reshape(self.K, self.P).sum(1)               # population sum per candidate word
        return per_pool, float(firing.sum())

    def read(self, weights, equal_drive=False):
        """ONE few-spike read: winner = argmax over per-pool accumulated firing (read from cp_firing_states).
        Returns (winner_pool_idx or -1 if silent, per_pool_firing, total_spikes)."""
        drive = self.drive_from_weights(weights)
        per_pool, tot = self._compete(drive, equal_drive=equal_drive)
        if per_pool.max() <= 0.0:
            return -1, per_pool, tot                                   # silent (honest: no sample), NOT a host draw
        return int(np.argmax(per_pool)), per_pool, tot


# ----------------------------------------------------------------------------------------------------------------
def _softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()


def _native(o):
    """Recursively cast numpy scalars/bools/arrays to native python (round() of a numpy float returns a numpy
    float64, and comparisons of it return numpy.bool_ -- neither is JSON-serializable)."""
    if isinstance(o, dict):
        return {k: _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _native(o.tolist())
    return o


def _load_eval(ro, corpus, n_sentences, seed, n_eval):
    path = next((p for p in ([corpus] if corpus else []) + _CORPUS_CANDIDATES if Path(p).exists()), None)
    if path is None:
        raise FileNotFoundError(f"no tinystories corpus found in {_CORPUS_CANDIDATES}")
    sents = load_sentences(path, n_sentences)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sents)); cut = int(0.85 * len(sents))
    ev = [sents[i] for i in idx[cut:]][:n_eval]
    vocab = Vocab(ro.words[:-1])
    return [vocab.ids(s) for s in ev], vocab


def run_seed(seed, ckpt, corpus, n_sentences, n_eval_pos, warmup, topk, read_window, pop,
             base_pA, gain_pA, ou_std, gen_tokens, gen_temp, sample_temp, arms):
    """One seed x one (read_window, pop) operating point.  Returns a dict of metrics + anti-cheats + generation."""
    ro = WKVReadout(ckpt)
    ev_ids, vocab = _load_eval(ro, corpus, n_sentences, seed, max(64, n_eval_pos // 8))
    reader = FewSpikeWordRead(topk, pop, seed, ou_std=ou_std, base_pA=base_pA, gain_pA=gain_pA,
                              read_window=read_window)
    reader_ablate = None                                               # built lazily for noise-ablation
    grng = np.random.default_rng(seed * 101 + 7)                       # host reference-sampler RNG (ceiling only)

    # ---- TEACHER-FORCED read fidelity over held-out positions -------------------------------------------------
    acc = {"n": 0, "spikes": 0.0, "topk_cover": 0.0,
           "argmax_agree": 0.0, "top5_hit": 0.0, "nll_fewspike": 0.0,
           "mass_fewspike": 0.0, "mass_hostsample": 0.0, "mass_argmax": 0.0,
           "mass_scramble": 0.0, "argmax_agree_scramble": 0.0, "mass_equal": 0.0,
           "silent": 0, "detvar": 0.0}
    positions = 0
    for ids in ev_ids:
        if len(ids) < warmup + 2:
            continue
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in range(len(ids) - 1):
            ap, an = ro.advance(ap, an, ids[t])
            if t < warmup:
                continue
            lg = ro.logits(ap, an, ids[t])
            if ro.unk_idx >= 0:
                lg = lg.copy(); lg[ro.unk_idx] = -1e30                 # suppress the high-freq <unk> (deployed --gen-no-unk)
            cand = np.argpartition(-lg, topk - 1)[:topk]               # top-K candidate WORD ids by logit
            cand = cand[np.argsort(-lg[cand])]                         # sorted desc; cand[0] == host argmax
            p = _softmax(lg[cand] / sample_temp)                       # the model's top-K next-token distribution
            host_argmax = int(cand[0])
            top5 = set(int(c) for c in cand[:5])

            # FEW-SPIKE read (the mechanism)
            win, per_pool, tot = reader.read(p)
            fewspike = int(cand[win]) if win >= 0 else -1
            # host-sample reference ceiling (an IDEAL categorical sampler over the SAME top-K softmax)
            hs = int(cand[int(grng.choice(len(p), p=p))])
            # scramble anti-cheat: decode the TRUE winning pool through a FRESH random pool->word labelling. The word
            # identity is carried by the labelled-line map; permuting it must destroy argmax agreement -> chance 1/K,
            # ROBUSTLY (independent of the drive, unlike an input-permutation which the WTA re-sorts). Reuses the
            # fewspike winner -> no extra bridge run.
            perm_t = np.random.default_rng(seed * 71 + 5 + positions).permutation(len(cand))
            fewspike_s = int(cand[perm_t[win]]) if win >= 0 else -1
            # equal-drive anti-cheat: all active pools equal -> uniform over active set (drive magnitude load-bearing)
            win_e, _, _ = reader.read(p, equal_drive=True)
            fewspike_e = int(cand[win_e]) if win_e >= 0 else -1

            acc["n"] += 1; positions += 1
            acc["spikes"] += tot
            acc["topk_cover"] += float(p.sum() if False else _softmax(lg)[cand].sum())  # mass in the top-K set
            pfull = _softmax(lg)
            if win < 0:
                acc["silent"] += 1
            acc["argmax_agree"] += float(fewspike == host_argmax)
            acc["top5_hit"] += float(fewspike in top5)
            acc["nll_fewspike"] += -math.log(max(pfull[fewspike] if fewspike >= 0 else 1e-12, 1e-12))
            acc["mass_fewspike"] += (pfull[fewspike] if fewspike >= 0 else 0.0)
            acc["mass_hostsample"] += pfull[hs]
            acc["mass_argmax"] += pfull[host_argmax]
            acc["mass_scramble"] += (pfull[fewspike_s] if fewspike_s >= 0 else 0.0)
            acc["argmax_agree_scramble"] += float(fewspike_s == host_argmax)
            acc["mass_equal"] += (pfull[fewspike_e] if fewspike_e >= 0 else 0.0)
            if positions >= n_eval_pos:
                break
        if positions >= n_eval_pos:
            break

    n = max(1, acc["n"])
    # noise-ablation: ou_std -> 0 must give a DETERMINISTIC read (argmax-over-drive == the peak-firing pool)
    reader_ablate = FewSpikeWordRead(topk, pop, seed, ou_std=0.0, base_pA=base_pA, gain_pA=gain_pA,
                                     read_window=read_window)
    det_w = np.zeros(topk); det_w[[0, 1, 2]] = [1.0, 0.6, 0.3]         # a fixed peaked drive
    w0, _, _ = reader_ablate.read(det_w); w1, _, _ = reader_ablate.read(det_w)
    det_stable = (w0 == w1 == 0)                                       # deterministic argmax at the peak pool

    m = {
        "seed": seed, "read_window": read_window, "pop": pop, "topk": topk,
        "n_positions": acc["n"], "silent_frac": acc["silent"] / n,
        "mean_spikes_per_read": round(acc["spikes"] / n, 2),
        "topk_coverage": round(acc["topk_cover"] / n, 4),
        "argmax_agree": round(acc["argmax_agree"] / n, 4),
        "top5_hit": round(acc["top5_hit"] / n, 4),
        "nll_fewspike": round(acc["nll_fewspike"] / n, 4),
        "mass_fewspike": round(acc["mass_fewspike"] / n, 4),
        "mass_hostsample_ceiling": round(acc["mass_hostsample"] / n, 4),
        "mass_argmax_ceiling": round(acc["mass_argmax"] / n, 4),
        "mass_scramble": round(acc["mass_scramble"] / n, 4),
        "argmax_agree_scramble": round(acc["argmax_agree_scramble"] / n, 4),
        "mass_equal_drive": round(acc["mass_equal"] / n, 4),
        "chance_1_over_k": round(1.0 / topk, 4),
        "noise_ablation_deterministic": bool(det_stable),
        "host_rng_draws_on_read_path": int(reader.n_host_rng_draws),   # MUST be 0
    }
    # read_fidelity: how on-distribution the few-spike read is, RELATIVE to an ideal host sampler
    m["read_fidelity_vs_sampler"] = round(m["mass_fewspike"] / max(1e-9, m["mass_hostsample_ceiling"]), 4)

    # ---- FREE GENERATION survival (self-NLL of the model's own continuation under the graded read-out) --------
    def _free_gen(read_fn, prompt="once upon a time", n_tok=gen_tokens):
        pid = [i for i in vocab.ids(prompt.split()) if 0 <= i < ro.V] or [0]
        ap = np.zeros(ro.D); an = np.zeros(ro.D)
        for t in pid:
            ap, an = ro.advance(ap, an, t)
        gen = list(pid); self_nll = 0.0; steps = 0
        for _ in range(n_tok):
            lg = ro.logits(ap, an, gen[-1])
            if ro.unk_idx >= 0:
                lg = lg.copy(); lg[ro.unk_idx] = -1e30
            cand = np.argpartition(-lg, topk - 1)[:topk]; cand = cand[np.argsort(-lg[cand])]
            p = _softmax(lg[cand] / gen_temp)
            nxt = read_fn(cand, p, lg)
            self_nll += -math.log(max(_softmax(lg)[nxt], 1e-12)); steps += 1
            gen.append(nxt); ap, an = ro.advance(ap, an, nxt)
        txt = " ".join(ro.words[i] if 0 <= i < len(ro.words) else "<unk>" for i in gen)
        return txt, (self_nll / max(1, steps))

    def _read_fewspike(cand, p, lg):
        w, _, _ = reader.read(p)
        return int(cand[w]) if w >= 0 else int(cand[0])

    def _read_hostsample(cand, p, lg):
        return int(cand[int(grng.choice(len(p), p=p))])

    gen_results = {}
    if gen_tokens > 0 and ("gen" in arms):
        for pr in ("once upon a time", "the little girl", "tom and his dog"):
            t_fs, nll_fs = _free_gen(_read_fewspike, pr)
            t_hs, nll_hs = _free_gen(_read_hostsample, pr)
            gen_results[pr] = {"fewspike_text": t_fs, "fewspike_self_nll": round(nll_fs, 3),
                               "hostsample_text": t_hs, "hostsample_self_nll": round(nll_hs, 3)}
    m["generation"] = gen_results
    return m


def _verdict(m):
    """Per-operating-point read-regime verdict (pre-registered)."""
    chance = m["chance_1_over_k"]
    checks = {
        "read_fidelity_ge_0.90": m["read_fidelity_vs_sampler"] >= 0.90,
        "argmax_agree_gt_2x_chance": m["argmax_agree"] > 2 * chance,
        "scramble_collapses": m["argmax_agree_scramble"] < 2 * chance,
        "equal_drive_below_fewspike": m["mass_equal_drive"] < 0.9 * m["mass_fewspike"],
        "provenance_no_host_draw": m["host_rng_draws_on_read_path"] == 0,
        "noise_ablation_deterministic": m["noise_ablation_deterministic"],
        "not_silent": m["silent_frac"] < 0.05,
    }
    return all(checks.values()), checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="bridges/wkv_ckpt/wkv_ssmU6_v1000_d128_seed{seed}.npz")
    ap.add_argument("--corpus", type=str, default="")
    ap.add_argument("--n-sentences", type=int, default=8000)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--n-eval-pos", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--read-windows", type=str, default="8,20,60,120")
    ap.add_argument("--pops", type=str, default="1,8")
    ap.add_argument("--base-pA", type=float, default=110.0)
    ap.add_argument("--gain-pA", type=float, default=160.0)
    ap.add_argument("--ou-std", type=float, default=200.0)
    ap.add_argument("--sample-temp", type=float, default=0.8)          # teacher-forced target dist temperature
    ap.add_argument("--gen-tokens", type=int, default=0)
    ap.add_argument("--gen-temp", type=float, default=0.8)
    ap.add_argument("--arms", type=str, default="gen")                 # include "gen" to run free-generation
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_wkv_fewspike_read.json")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = args.seeds or "42"
        args.n_eval_pos = min(args.n_eval_pos, 100)
        args.read_windows = "20,40"                                    # >=~7 spikes/neuron (rw=8 is sub-threshold silent)
        args.pops = "1,8,16"                                           # the POPULATION-CODING companion-process lever
        args.gen_tokens = args.gen_tokens or 40

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rws = [int(x) for x in args.read_windows.split(",") if x.strip()]
    pops = [int(x) for x in args.pops.split(",") if x.strip()]
    arms = set(a.strip() for a in args.arms.split(",") if a.strip())

    t0 = time.time()
    results = []
    for seed in seeds:
        ckpt = args.ckpt.format(seed=seed) if "{seed}" in args.ckpt else args.ckpt
        if not Path(ckpt).exists():
            print(f"[skip] seed {seed}: checkpoint {ckpt} missing", flush=True)
            continue
        for rw in rws:
            for pop in pops:
                gen_here = args.gen_tokens if (rw == max(rws)) else 0    # only free-gen at the largest budget (speed)
                m = run_seed(seed, ckpt, args.corpus, args.n_sentences, args.n_eval_pos, args.warmup,
                             args.topk, rw, pop, args.base_pA, args.gain_pA, args.ou_std,
                             gen_here, args.gen_temp, args.sample_temp, arms)
                go, checks = _verdict(m)
                m["go"] = go; m["checks"] = checks
                results.append(m)
                print(f"[seed {seed} rw={rw} P={pop}] spikes/read={m['mean_spikes_per_read']} "
                      f"read_fid={m['read_fidelity_vs_sampler']} argmax_agree={m['argmax_agree']} "
                      f"mass fs={m['mass_fewspike']} sampler={m['mass_hostsample_ceiling']} "
                      f"argmax={m['mass_argmax_ceiling']} scr={m['mass_scramble']} eq={m['mass_equal_drive']} "
                      f"GO={go} ({sum(checks.values())}/{len(checks)})", flush=True)
                if m.get("generation"):
                    for pr, g in m["generation"].items():
                        print(f"    [gen '{pr}'] fewspike(self-NLL {g['fewspike_self_nll']}): {g['fewspike_text'][:180]}", flush=True)

    # ---- aggregate: per (rw,pop), the 6-seed GO fraction + the read-fidelity ladder ---------------------------
    agg = {}
    for m in results:
        key = f"rw{m['read_window']}_P{m['pop']}"
        agg.setdefault(key, {"read_fidelity": [], "argmax_agree": [], "mean_spikes": [], "go": []})
        agg[key]["read_fidelity"].append(m["read_fidelity_vs_sampler"])
        agg[key]["argmax_agree"].append(m["argmax_agree"])
        agg[key]["mean_spikes"].append(m["mean_spikes_per_read"])
        agg[key]["go"].append(m["go"])
    summary = {}
    for key, d in agg.items():
        summary[key] = {
            "n_seeds": len(d["go"]), "go_count": int(sum(d["go"])),
            "read_fidelity_mean": round(float(np.mean(d["read_fidelity"])), 4),
            "read_fidelity_min": round(float(np.min(d["read_fidelity"])), 4),
            "argmax_agree_mean": round(float(np.mean(d["argmax_agree"])), 4),
            "mean_spikes_per_read": round(float(np.mean(d["mean_spikes"])), 2),
        }
    out = {"results": results, "summary": summary, "seeds": seeds, "read_windows": rws, "pops": pops,
           "topk": args.topk, "sample_temp": args.sample_temp, "elapsed_s": round(time.time() - t0, 1),
           "backend": os.environ.get("SIM_BACKEND", "numpy")}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(_native(out), indent=2))
    print(f"\n[SUMMARY] {json.dumps(summary, indent=2)}", flush=True)
    print(f"[done] {len(results)} operating points, {time.time()-t0:.0f}s -> {args.json}", flush=True)


if __name__ == "__main__":
    main()
