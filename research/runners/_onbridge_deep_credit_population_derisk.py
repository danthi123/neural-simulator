"""ON-BRIDGE deep-credit POPULATION-CODING K-SWEEP de-risk -- the DECISIVE test of the read-SNR-wall fix.

WHY THIS RUN (the boundary + the researched fix): the on-bridge SPIKING deep-credit net
(`_semantic_inheritance_onbridge_spiking_derisk.OnBridgeBDSPNet`) does NOT train the depth-2 compositional-inheritance
task at CPU-smoke scale (6-seed, all arms below chance; the fenced-backprop RATE oracle reaches ~1.0 -> the task IS
learnable). DIAGNOSIS (`2026-07-07-onbridge-spiking-deep-credit-training-research-gate.md`, sec 1): each logical unit is
read as a SINGLE-NEURON low-pass event rate `cp_bdsp_E`; at the operating event rate (~0.05-0.10) with the ~10-step
low-pass window the single-neuron E has CV~1.0-1.4 -> the forward activation, the softmax logits, AND the descending
clean-error credit are all noise-dominated per example -> weights move HARD but in NOISE directions -> no learning.

THE FIX (research gate sec 2/3, mechanism #1 -- catalog E.03/H.17 population coding; Payeur-2021 Burstprop + Greedy-Costa
BurstCCN were DEFINED at the ENSEMBLE level, BurstCCN used 500 neurons/unit for spiking XOR): POPULATION CODING -- each
logical unit = K neurons; read the POOLED (block-mean) event rate over the K neurons, which cuts CV by ~sqrt(K)
(K=8 -> CV~0.5, K=16 -> CV~0.35). This is a runner-side READ change, NOT a `sim/` edit.

WHAT THIS RUNNER IS: a THIN K-SWEEP DRIVER. The population-coding mechanism ITSELF already lives in `OnBridgeBDSPNet`
(the `--pool-k` knob: `_pool`/`_broadcast` block-mean read + broadcast credit; `read_snr_corr` diagnostic; K=1 ==
byte-identical single-neuron runner = the causal control). This runner REUSES-BY-IMPORT `_run_arm` (the arm trainer),
`make_task_semantic_inheritance` + `stage0_depth_genuineness` (the task + depth gate), and `DendriticMLP`/`_train_oracle`
(the rate oracle ceiling), and sweeps K in {1, 8, 16, ...}. The ONE new variable = K (population size). NO net/task/arm
re-implementation; NO `sim/` edit.

THE K-SCALING GATE (the decisive reads):
  * trains_at_all(K) = best on-bridge arm held-out-inheritance > 1-layer floor + margin AND > chance + margin.
  * ANTI-CHEAT k1_reproduces_boundary: K=1 must NOT train (== the committed negative) -- the causal control.
  * population_lifts_training: some K>1 trains where K=1 does not -> the lift is the pooling.
  * read-SNR diagnostic corr(pooled E, soma_rate) per hidden layer must RISE with K -> the direct fingerprint that
    population lifts the read SNR (a read-VARIANCE residual population fixes) vs stay FLAT (a credit-STRUCTURE residual
    it does NOT fix, and the microcircuit clean channel would be needed instead).
  * oracle ceiling (~1.0, K-independent) confirms the task is learnable; permuted->chance + apical-lesion->collapse
    (at the largest K) guard leakage / that the top-down credit is load-bearing.
  GO-candidate (for the controller's multi-seed fan-out) = population_lifts_training AND corr rises with K AND the
  anti-cheats hold. Honest-negative (a first-class deliverable) = K=1 reproduces the boundary + the corr/accuracy TREND,
  mapping whether the residual is read-variance (population is the fix, scale K) or credit-structure.

HONEST SCOPE / BUILDER vs CONTROLLER: BUILDER 1-seed CPU (numpy) SMOKE -- small H / few epochs / subsampled train so the
K-sweep finishes in minutes. The multi-seed GPU K-sweep + adversarial-verify is the CONTROLLER's. Held-out sets are NEVER
subsampled. NO `sim/` edit (all reuse-by-import).

Run (1-seed CPU smoke -- the K=1 causal control vs the K=8/16 population fix):
    SIM_BACKEND=numpy OMP_NUM_THREADS=1 python -m research.runners._onbridge_deep_credit_population_derisk \
        --seed 42 --k-list 1 8 16

The CONTROLLER's multi-seed GPU K-sweep (one process per seed; aggregate the per-seed JSONs):
    for s in 42 43 44 100 101 102; do SIM_BACKEND=cupy python -m \
        research.runners._onbridge_deep_credit_population_derisk --seed $s --k-list 1 8 16 32 \
        --hidden 64 --epochs 40 --settle-steps 40 --credit-steps 25 --train-subsample 0 \
        --out research/findings/raw/_onbridge_pop_K_seed$s.json & done; wait
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
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

# reuse-by-import: the ARM trainer (builds OnBridgeBDSPNet with pool_k, trains it, reads inherit/memctrl + read_snr).
from research.runners._semantic_inheritance_onbridge_spiking_derisk import _run_arm  # noqa: E402
# reuse-by-import: the TASK builder + the rate-oracle depth gate (K-independent, pure numpy).
from research.runners._semantic_inheritance_deep_credit_derisk import (  # noqa: E402
    make_task_semantic_inheritance, stage0_depth_genuineness, _train_oracle, _acc_on)
from sim.dendritic_mlp import DendriticMLP  # noqa: E402 -- the fenced backprop oracle ceiling

OUT = _REPO / "research" / "findings" / "raw" / "_onbridge_deep_credit_population.json"

TRAIN_MARGIN = 0.03      # best-arm must clear the floor AND chance by this to count as "trains"
CORR_RISE_MARGIN = 0.05  # corr(Kmax) must exceed corr(K1) by this to count as "read-SNR rises with K"


def _mean_corr(read_snr_corr):
    """Mean of the per-hidden-layer corr(pooled E, soma_rate) list (NaN-safe)."""
    if not read_snr_corr:
        return float("nan")
    return float(np.nanmean(read_snr_corr))


def sweep_seed(seed, k_list, hidden, epochs, batch, settle_steps, credit_steps, lr, subsample,
               hp, task_kwargs, n_hidden_layers=2):
    """Build the task ONCE; run the K-independent depth gate + rate oracle ONCE; sweep pool_k over `k_list` on the
    plain-FA on-bridge arm (the clean-error-descended credit the research gate predicts trains with population coding)
    + a pooled 1-layer floor per K; run the permuted / apical-lesion anti-cheats at the LARGEST K."""
    task_full = make_task_semantic_inheritance(seed, **task_kwargs)
    (Xtr, ytr, Ltr), (Xte, yte, Lte), meta, idx = task_full
    k = meta["k_classes"]; n_in = Xtr.shape[1]
    inh_idx = idx["inh_idx"]

    # STAGE 0 -- the depth gate (rate oracle on the FULL train set; validates the TASK genuinely requires depth). This
    # is K-INDEPENDENT (a numpy DendriticMLP, not the bridge) so it runs ONCE. Oracle settings match the rate reference
    # (hidden=96, epochs=250) so a light oracle doesn't spuriously fail the gate.
    full_task = ((Xtr, ytr, Ltr), (Xte, yte, Lte))
    s0 = stage0_depth_genuineness(full_task, idx, k, hidden=96, epochs=250, lr=0.3, batch=128, seed=seed)

    # chance on the inheritance held-out subset (the composition targets).
    if len(inh_idx):
        yv = yte[inh_idx]; chance = float(max(np.mean(yv == c) for c in np.unique(yv)))
    else:
        chance = float("nan")

    # rate oracle ceiling on the depth-`n_hidden_layers` net (task sanity; K-independent). FULL train set.
    onet = DendriticMLP([n_in] + [96] * n_hidden_layers + [k], seed=seed)
    _train_oracle(onet, Xtr, ytr, 250, 0.3, 128, seed)
    oracle = {"inherit_heldout": _acc_on(onet, Xte, yte, inh_idx),
              "memctrl_heldout": _acc_on(onet, Xte, yte, idx["memctrl_idx"]),
              "train": float(onet.accuracy(Xtr, ytr))}

    # SMOKE speed: subsample the TRAIN set for the on-bridge (per-example spiking) arms. Held-out NEVER subsampled.
    if subsample is not None and len(Xtr) > subsample:
        srng = np.random.default_rng(seed + 13)
        keep = srng.permutation(len(Xtr))[:subsample]
        Xtr_b, ytr_b, Ltr_b = Xtr[keep], ytr[keep], Ltr[keep]
    else:
        Xtr_b, ytr_b, Ltr_b = Xtr, ytr, Ltr
    task_b = ((Xtr_b, ytr_b, Ltr_b), (Xte, yte, Lte))

    # ---- THE K SWEEP: plain-FA deep arm (want_read_snr) + a pooled 1-layer floor, per K ----
    per_k = []
    for K in k_list:
        deep = _run_arm("plain_fa", task_b, idx, n_in, hidden, k, epochs, batch, seed,
                        n_hidden_layers=n_hidden_layers, settle_steps=settle_steps,
                        credit_steps=credit_steps, lr=lr, hp=hp, pool_k=K, want_read_snr=True)
        floor = _run_arm("plain_fa", task_b, idx, n_in, hidden, k, epochs, batch, seed,
                         n_hidden_layers=1, settle_steps=settle_steps, credit_steps=credit_steps,
                         lr=lr, hp=hp, pool_k=K)
        d_inh = deep["inherit_heldout"]; f_inh = floor["inherit_heldout"]
        trains = bool((not np.isnan(d_inh)) and d_inh > f_inh + TRAIN_MARGIN and d_inh > chance + TRAIN_MARGIN)
        per_k.append({
            "pool_k": int(K),
            "deep_inherit_heldout": d_inh, "deep_memctrl_heldout": deep["memctrl_heldout"],
            "deep_train": deep["train"], "ff_weight_moved": deep["ff_weight_moved"],
            "floor_inherit_heldout": f_inh, "floor_train": floor["train"],
            "read_snr_corr": deep.get("read_snr_corr"), "read_snr_corr_mean": _mean_corr(deep.get("read_snr_corr")),
            "trains_at_all": trains})

    # ---- anti-cheats at the LARGEST K (the one that should train if the population fix works) ----
    Kmax = int(max(k_list))
    prng = np.random.default_rng(seed + 555)
    yperm = ytr_b[prng.permutation(len(ytr_b))]
    permuted = _run_arm("plain_fa", ((Xtr_b, yperm, Ltr_b), (Xte, yte, Lte)), idx, n_in, hidden, k,
                        epochs, batch, seed, n_hidden_layers=n_hidden_layers, settle_steps=settle_steps,
                        credit_steps=credit_steps, lr=lr, mode="bdsp", hp=hp, pool_k=Kmax)
    lesion = _run_arm("plain_fa", task_b, idx, n_in, hidden, k, epochs, batch, seed,
                      n_hidden_layers=n_hidden_layers, settle_steps=settle_steps,
                      credit_steps=credit_steps, lr=lr, mode="apical_lesion", hp=hp, pool_k=Kmax)

    return {"seed": seed, "meta": meta, "chance": chance, "k_list": [int(x) for x in k_list],
            "stage0_depth_genuineness": s0, "oracle": oracle,
            "per_k": per_k, "anticheat_k": Kmax,
            "permuted": {"inherit_heldout": permuted["inherit_heldout"], "pool_k": Kmax},
            "apical_lesion": {"inherit_heldout": lesion["inherit_heldout"], "pool_k": Kmax}}


def _aggregate(per):
    """Aggregate the per-seed sweeps into the K-scaling verdict (mean over seeds at each K)."""
    seeds = [p["seed"] for p in per]
    k_list = per[0]["k_list"]
    ch = float(np.nanmean([p["chance"] for p in per]))
    s0_sep = all(p["stage0_depth_genuineness"]["depth_separating"] for p in per)
    oracle = float(np.nanmean([p["oracle"]["inherit_heldout"] for p in per]))
    oracle_mem = float(np.nanmean([p["oracle"]["memctrl_heldout"] for p in per]))

    per_k = []
    for ki, K in enumerate(k_list):
        di = float(np.nanmean([p["per_k"][ki]["deep_inherit_heldout"] for p in per]))
        fi = float(np.nanmean([p["per_k"][ki]["floor_inherit_heldout"] for p in per]))
        corr = float(np.nanmean([p["per_k"][ki]["read_snr_corr_mean"] for p in per]))
        ffm = float(np.nanmean([p["per_k"][ki]["ff_weight_moved"] for p in per]))
        dtr = float(np.nanmean([p["per_k"][ki]["deep_train"] for p in per]))
        trains = bool(di > fi + TRAIN_MARGIN and di > ch + TRAIN_MARGIN)
        per_k.append({"pool_k": int(K), "deep_inherit_heldout": di, "floor_inherit_heldout": fi,
                      "read_snr_corr_mean": corr, "ff_weight_moved": ffm, "deep_train": dtr,
                      "trains_at_all": trains})

    k1 = per_k[0]; kmax = per_k[-1]
    perm = float(np.nanmean([p["permuted"]["inherit_heldout"] for p in per]))
    les = float(np.nanmean([p["apical_lesion"]["inherit_heldout"] for p in per]))

    # the decisive reads
    k1_reproduces_boundary = bool(not k1["trains_at_all"])
    population_lifts_training = bool((not k1["trains_at_all"]) and any(pk["trains_at_all"] for pk in per_k[1:]))
    corr_k1 = k1["read_snr_corr_mean"]; corr_kmax = kmax["read_snr_corr_mean"]
    corr_rises_with_K = bool((not np.isnan(corr_k1)) and (not np.isnan(corr_kmax))
                             and corr_kmax > corr_k1 + CORR_RISE_MARGIN)
    corr_direction = float(corr_kmax - corr_k1) if not (np.isnan(corr_k1) or np.isnan(corr_kmax)) else float("nan")
    permuted_chance = bool(np.isnan(perm) or perm <= ch + 0.10)
    lesion_collapses = bool(np.isnan(les) or les <= max(k1["floor_inherit_heldout"], ch) + 0.08)
    oracle_ok = bool(oracle >= 0.80)
    memctrl_holds = bool(np.isnan(oracle_mem) or oracle_mem <= ch + 0.15)

    go_candidate = bool(s0_sep and oracle_ok and population_lifts_training and corr_rises_with_K
                        and k1_reproduces_boundary and permuted_chance and lesion_collapses and memctrl_holds)

    return {"seeds": seeds, "k_list": k_list, "chance": ch, "stage0_depth_separating": s0_sep,
            "oracle_inherit": oracle, "oracle_memctrl": oracle_mem, "per_k": per_k,
            "permuted_inherit": perm, "apical_lesion_inherit": les,
            "k1_reproduces_boundary": k1_reproduces_boundary,
            "population_lifts_training": population_lifts_training,
            "corr_k1": corr_k1, "corr_kmax": corr_kmax, "corr_direction": corr_direction,
            "corr_rises_with_K": corr_rises_with_K, "permuted_chance": permuted_chance,
            "lesion_collapses": lesion_collapses, "oracle_ok": oracle_ok, "memctrl_holds": memctrl_holds,
            "GO_CANDIDATE": go_candidate}


def _verdict(agg):
    ch = agg["chance"]; oracle = agg["oracle_inherit"]
    ktab = " | ".join(f"K={pk['pool_k']}: inh {pk['deep_inherit_heldout']:.3f} (floor {pk['floor_inherit_heldout']:.3f}, "
                      f"corr {pk['read_snr_corr_mean']:.3f}, trains {pk['trains_at_all']})" for pk in agg["per_k"])
    if not agg["stage0_depth_separating"]:
        return (f"STAGE-0 not depth-separating -- fix the task config before reading the K-sweep. ({ktab})")
    if not agg["oracle_ok"]:
        return (f"INCONCLUSIVE -- the rate oracle only reached {oracle:.3f} held-out inheritance; the task/oracle "
                f"need tuning before the K-sweep is readable. ({ktab})")
    if not agg["k1_reproduces_boundary"]:
        return (f"UNEXPECTED -- K=1 already TRAINS at this smoke scale (does NOT reproduce the committed does-not-train "
                f"boundary); the causal control fails, so the K-lift is not isolatable here. Re-check the smoke config "
                f"(smaller/cleaner may have made K=1 trainable). ({ktab})")
    if agg["GO_CANDIDATE"]:
        return (f"GO-CANDIDATE (1-seed smoke) -- POPULATION CODING lifts on-bridge deep-credit training: K=1 does NOT "
                f"train (reproduces the boundary), a larger K DOES, AND the read-SNR corr(pooled E, soma_rate) RISES "
                f"with K ({agg['corr_k1']:.3f}->{agg['corr_kmax']:.3f}, +{agg['corr_direction']:.3f}). Anti-cheats hold "
                f"(permuted {agg['permuted_inherit']:.3f}~chance {ch:.3f}, lesion {agg['apical_lesion_inherit']:.3f} "
                f"collapses, oracle {oracle:.3f}, memctrl {agg['oracle_memctrl']:.3f}). ({ktab}) => CONTROLLER: run the "
                f"6-seed GPU K-sweep + adversarial-verify.")
    # honest-negative branches -- map the residual by the corr TREND (read-variance vs credit-structure).
    lifts = agg["population_lifts_training"]; rises = agg["corr_rises_with_K"]
    if (not lifts) and rises:
        return (f"HONEST NEGATIVE (1-seed smoke) -- the read-SNR corr(pooled E, soma_rate) DOES rise with K "
                f"({agg['corr_k1']:.3f}->{agg['corr_kmax']:.3f}, +{agg['corr_direction']:.3f}) [the read gets cleaner, "
                f"the diagnosed mechanism], but no K in {agg['k_list']} yet crosses the training bar at this smoke scale "
                f"(K=1 correctly at boundary). READ: the residual is READ-VARIANCE (population is the right lever); "
                f"SCALE K (BurstCCN's own working K is 500/unit) + hidden/epochs/settle (GPU) before concluding. ({ktab})")
    if lifts and not rises:
        return (f"MIXED (1-seed smoke) -- a larger K TRAINS where K=1 does not, but the read-SNR corr did NOT clearly "
                f"rise (+{agg['corr_direction']:.3f}); the lift may not be the clean read-variance mechanism (or the corr "
                f"diagnostic is noisy at 1 seed). CONTROLLER: 6-seed + the temporal-averaging matched control (spatial "
                f"pooling vs equal-total-spike temporal averaging) to confirm the mechanism. ({ktab})")
    return (f"HONEST NEGATIVE (1-seed smoke) -- population coding did NOT lift training at K in {agg['k_list']} AND the "
            f"read-SNR corr did NOT clearly rise (+{agg['corr_direction']:.3f}). This maps toward a CREDIT-STRUCTURE "
            f"residual (population would NOT fix it; the microcircuit clean-error channel is the lever) OR the smoke is "
            f"too small/short to move either. K=1 reproduces the boundary ({agg['k1_reproduces_boundary']}). CONTROLLER: "
            f"scale + run the microcircuit arm K-sweep + the temporal-matched control. ({ktab})")


def main():
    ap = argparse.ArgumentParser(description="On-bridge deep-credit POPULATION-CODING K-sweep de-risk (read-SNR fix).")
    ap.add_argument("--seed", type=int, default=None, help="single seed (smoke); overrides --seeds if given")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--k-list", type=int, nargs="+", default=[1, 8, 16],
                    help="population sizes K (neurons per logical unit) to sweep. K=1 == the single-neuron causal "
                         "control (must reproduce the does-not-train boundary); K=8/16/... = the population fix.")
    # SMOKE defaults: small H / few epochs / subsampled train so the multi-K sweep finishes in minutes on CPU.
    ap.add_argument("--hidden", type=int, default=24, help="hidden LOGICAL units per layer (CPU smoke 24; GPU 64+)")
    ap.add_argument("--epochs", type=int, default=14, help="on-bridge epochs (per-example online)")
    ap.add_argument("--batch", type=int, default=120, help="(informational: training is per-example online)")
    ap.add_argument("--lr", type=float, default=0.25, help="bdsp_learning_rate")
    ap.add_argument("--settle-steps", type=int, default=25, help="spiking forward-settle steps per example")
    ap.add_argument("--credit-steps", type=int, default=15, help="credit-injection steps per example")
    ap.add_argument("--n-hidden-layers", type=int, default=2)
    ap.add_argument("--train-subsample", type=int, default=80,
                    help="CPU-smoke train subsample (held-out NEVER subsampled); set 0 for full (GPU).")
    # drive/credit hyperparameters (match the on-bridge runner's smoke defaults so K=1 == its causal control).
    ap.add_argument("--tonic-h-pA", type=float, default=560.0)
    ap.add_argument("--tonic-o-pA", type=float, default=620.0)
    ap.add_argument("--apical-gain-pA", type=float, default=2000.0)
    ap.add_argument("--pbar-alpha", type=float, default=0.05)
    ap.add_argument("--ff-w-init", type=float, default=4.5)
    # task knobs -- MATCH the on-bridge runner's CPU-smoke defaults (n_prop=2 = the 5-class trainable-on-spikes config)
    # so the K=1 arm reproduces THAT runner's committed does-not-train negative.
    ap.add_argument("--n-super", type=int, default=12)
    ap.add_argument("--n-members", type=int, default=8)
    ap.add_argument("--held-per-super", type=int, default=3)
    ap.add_argument("--n-prop", type=int, default=2)
    ap.add_argument("--member-id-dim", type=int, default=3)
    ap.add_argument("--n-obs", type=int, default=16)
    ap.add_argument("--noise", type=float, default=0.02)
    ap.add_argument("--feature-seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    seeds = [a.seed] if a.seed is not None else list(a.seeds)
    task_kwargs = dict(n_super=a.n_super, n_members=a.n_members, held_per_super=a.held_per_super,
                       n_prop=a.n_prop, member_id_dim=a.member_id_dim, n_obs=a.n_obs, noise=a.noise,
                       feature_seed=a.feature_seed)
    subsample = None if a.train_subsample == 0 else a.train_subsample
    hp = dict(tonic_h_pA=a.tonic_h_pA, tonic_o_pA=a.tonic_o_pA, apical_gain_pA=a.apical_gain_pA,
              ff_w_init=a.ff_w_init, pbar_alpha=a.pbar_alpha)

    t0 = time.time(); err = None; per = []
    try:
        for s in seeds:
            r = sweep_seed(s, a.k_list, a.hidden, a.epochs, a.batch, a.settle_steps, a.credit_steps,
                           a.lr, subsample, hp, task_kwargs, n_hidden_layers=a.n_hidden_layers)
            per.append(r)
            s0 = r["stage0_depth_genuineness"]; ch = r["chance"]
            print("-" * 112, flush=True)
            print(f"[seed {s}] chance {ch:.3f} | STAGE0 depth-sep (rate oracle): 1-layer "
                  f"{s0['l1_inherit_heldout']:.3f} vs deep-best {s0['deep_best_inherit_heldout']:.3f} "
                  f"(gap {s0['depth_gap']:+.3f}) => DEPTH-SEPARATING {s0['depth_separating']} | oracle-ceiling "
                  f"{r['oracle']['inherit_heldout']:.3f}", flush=True)
            print(f"  ON-BRIDGE SPIKING plain-FA held-out INHERITANCE as a function of POPULATION K:", flush=True)
            for pk in r["per_k"]:
                snr = pk.get("read_snr_corr")
                snrs = ("[" + ", ".join("%.3f" % c for c in snr) + "]") if snr else "n/a"
                print(f"    K={pk['pool_k']:<3d} inherit {pk['deep_inherit_heldout']:.3f} | floor "
                      f"{pk['floor_inherit_heldout']:.3f} | train {pk['deep_train']:.3f} | ff-moved "
                      f"{pk['ff_weight_moved']:.1f} | read-SNR corr {snrs} (mean {pk['read_snr_corr_mean']:.3f}) "
                      f"| TRAINS {pk['trains_at_all']}", flush=True)
            print(f"    [anti-cheat @K={r['anticheat_k']}] permuted {r['permuted']['inherit_heldout']:.3f} (~chance "
                  f"{ch:.3f}) | apical-lesion {r['apical_lesion']['inherit_heldout']:.3f} (must collapse)", flush=True)
    except Exception as e:
        err = repr(e); traceback.print_exc()

    summary = {"probe": "onbridge_deep_credit_population_ksweep", "seeds": seeds, "k_list": a.k_list,
               "config": {"hidden": a.hidden, "epochs": a.epochs, "lr": a.lr, "settle_steps": a.settle_steps,
                          "credit_steps": a.credit_steps, "n_hidden_layers": a.n_hidden_layers,
                          "train_subsample": subsample, "task": task_kwargs, "hp": hp,
                          "backend": os.environ.get("SIM_BACKEND")},
               "elapsed_seconds": round(time.time() - t0, 1), "per_seed": per}
    if err is None and per:
        agg = _aggregate(per)
        summary["aggregate"] = agg
        summary["GO_CANDIDATE"] = agg["GO_CANDIDATE"]
        summary["verdict"] = _verdict(agg)
    else:
        summary["GO_CANDIDATE"] = False
        summary["verdict"] = f"ERROR -- {err}" if err else "no seeds ran"

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 112, flush=True)
    print(f"[onbridge-deep-credit-population] {summary['verdict']}", flush=True)
    print(f"[onbridge-deep-credit-population] wrote {a.out}\n" + "=" * 112, flush=True)
    return 0 if summary.get("GO_CANDIDATE") else 1


if __name__ == "__main__":
    sys.exit(main())
