"""CANDIDATE B: is the replay window winner set by SLOT EXCITABILITY HETEROGENEITY? (2026-07-28)

THE QUESTION
------------
`coactivation_replay` drives fact i's dedicated slot `comp_attr_i` with 1400 pA, yet the driven slot
wins its own 30-step window only 15/27 times (chance 9/27), and competition inside a window is
near-exclusive (winner 400-1100 spikes, others 0-12). Something OTHER than the 1400 pA cue is picking
the winner. Candidate B says that something is intrinsic per-slot excitability.

WHY THIS IS PLAUSIBLE FROM THE CODE (read, not assumed)
  * `sim/bridge.py:1629`: cp_neuron_firing_thresholds = uniform(homeostasis_threshold_min,
    homeostasis_threshold_max) = uniform(-55, -30) -> a **25 mV** per-neuron spread.
  * `cfg.enable_homeostasis` defaults **True** (`sim/config.py:567`) and this runner never sets it,
    so `bridge.py:7397` takes the FIRST branch: the spike threshold IS cp_neuron_firing_thresholds
    (not cp_izh_vpeak). The array is load-bearing, not decorative.
  * `_initialize_rng(cfg.seed)` runs at `bridge.py:1263`, BEFORE the draw at 1629, and this runner DOES
    set `cfg.seed` (line 350) -> the thresholds are seeded and seed-dependent. A seed-dependent winner
    (42->4/9, 43->4/9, 44->7/9) is exactly the signature this predicts.
  * Homeostasis is ACTIVE every step (no experiment_engine -> `_homeostasis_gated` stays True,
    `bridge.py:8656`), so thresholds ALSO DRIFT during encode+replay. That is a second, distinct
    mechanism living in the same array -- hence the FREEZE arm below, which separates them.

WHAT IS MEASURED
  1. Per-slot threshold statistics at build / after encode / after replay (+ per-slot Izhikevich
     heterogeneity: C, a, b, d are jittered by _apply_parameter_heterogeneity; vt/vr/k are not).
  2. Per-slot afferent/efferent WEIGHT sums (inhibitory from comp_attr_inh, recurrent, pool->slot,
     ca1->slot) -- a wiring-level excitability bias would masquerade as an intrinsic one, and it is
     nearly free to measure here. (Informs candidates A/D.)
  3. The replay windows: per-slot spikes per 30-step burst, which fact was driven, who won.
  4. THE DECISIVE CORRELATION: is the window winner the LOWEST-mean-threshold slot?

CAUSAL ARMS (the null-has-three-explanations rule: a correlation alone can be confounded)
  --arm baseline : untouched (reproduces the 15/27 measurement).
  --arm freeze   : each slot neuron's threshold is re-pinned to its OWN pre-replay value every step.
                   Removes homeostatic DRIFT during replay, PRESERVES heterogeneity.
  --arm equalize : every slot neuron's threshold is pinned to the COMMON mean over all slot neurons
                   every step. Removes heterogeneity AND drift.
  baseline == freeze != equalize  => heterogeneity is the cause (candidate B CONFIRMED).
  baseline != freeze              => within-replay homeostatic drift is the cause (candidate C).
  baseline == freeze == equalize  => candidate B EXCLUDED; the thresholds are not what picks the winner.

THE LEVER IS VERIFIED, NOT ASSUMED: every arm prints the per-slot threshold spread measured from the
LIVE array at the last replay step, so an inert clamp is visible as a non-zero spread in `equalize`.

  SIM_BACKEND=cupy .venv/bin/python -m research.runners._consol_slot_excitability_probe --seed 42 --arm baseline
"""
import os, sys, json, argparse
os.environ.setdefault("SIM_BACKEND", "cupy")
for _tv in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")
sys.path.insert(0, "/home/dant123/Projects/sim")
import hashlib
import numpy as np
from types import SimpleNamespace
from pathlib import Path
from research.runners.nmda_compositional_consolidation import (
    build_substrate, encode_facts_with_reinstatement, coactivation_replay,
    CONSOLIDATED_FACTS)
from research.runners._consol_direct_weight_probe import BASE
from sim.backend import get_backend, to_host

cp, BACKEND = get_backend()
N = len(CONSOLIDATED_FACTS)
BURST = 30


def _slot_stats(b, slot, label):
    """Per-slot intrinsic-excitability statistics read from the LIVE arrays."""
    thr = to_host(b.cp_neuron_firing_thresholds).astype(np.float64)
    out = {}
    for j in sorted(slot):
        t = thr[slot[j]]
        row = dict(thr_mean=float(t.mean()), thr_std=float(t.std()), thr_min=float(t.min()),
                   thr_p10=float(np.percentile(t, 10)), thr_p25=float(np.percentile(t, 25)))
        for nm, arr in (("C", getattr(b, "cp_izh_C", None)), ("a", getattr(b, "cp_izh_a", None)),
                        ("b", getattr(b, "cp_izh_b", None)), ("d", getattr(b, "cp_izh_d_increment", None)),
                        ("k", getattr(b, "cp_izh_k", None)), ("vt", getattr(b, "cp_izh_vt", None)),
                        ("vr", getattr(b, "cp_izh_vr", None))):
            if arr is not None:
                row[f"izh_{nm}_mean"] = float(to_host(arr).astype(np.float64)[slot[j]].mean())
        out[j] = row
    print(f"  [{label}] per-slot firing threshold (spike threshold: homeostasis ON => THIS array is used):")
    for j in sorted(out):
        r = out[j]
        print(f"     slot {j}: mean={r['thr_mean']:8.4f}  std={r['thr_std']:6.4f}  min={r['thr_min']:8.4f}  "
              f"p10={r['thr_p10']:8.4f}  izh_C={r.get('izh_C_mean', float('nan')):7.3f} "
              f"izh_a={r.get('izh_a_mean', float('nan')):7.4f} izh_d={r.get('izh_d_mean', float('nan')):7.3f}")
    means = [out[j]["thr_mean"] for j in sorted(out)]
    print(f"     => threshold-mean SPREAD across slots = {max(means) - min(means):.4f} mV "
          f"(most excitable = LOWEST mean = slot {int(np.argmin(means))})")
    return out


def _wiring_sums(b, rm, slot, label):
    """Per-slot afferent weight sums by presynaptic source. A wiring bias would look like an
    intrinsic excitability bias in the window winner, so measure it alongside (candidates A/D)."""
    c = b.cp_connections
    nz = int(c.nnz)
    post = to_host(c.indices).astype(np.int64)[:nz]
    ip = to_host(c.indptr).astype(np.int64)
    pre = np.repeat(np.arange(len(ip) - 1), np.diff(ip))[:nz]
    w = to_host(c.data).astype(np.float64)[:nz]
    src = {}
    for nm in ("comp_attr_inh", "ca1"):
        try:
            src[nm] = np.asarray(sorted(rm.indices(nm)), dtype=np.int64)
        except Exception:
            pass
    out = {}
    print(f"  [{label}] per-slot AFFERENT weight sums (a wiring bias mimics an excitability bias):")
    for j in sorted(slot):
        m_post = np.isin(post, slot[j])
        row = {}
        for nm, idx in src.items():
            m = m_post & np.isin(pre, idx)
            row[f"w_from_{nm}"] = float(w[m].sum())
            row[f"n_from_{nm}"] = int(m.sum())
        m_self = m_post & np.isin(pre, slot[j])
        row["w_recurrent"] = float(w[m_self].sum())
        row["n_recurrent"] = int(m_self.sum())
        # everything else afferent (pool->slot broadcast etc.)
        m_other = m_post & ~np.isin(pre, slot[j])
        for idx in src.values():
            m_other &= ~np.isin(pre, idx)
        row["w_from_other"] = float(w[m_other].sum())
        row["n_from_other"] = int(m_other.sum())
        out[j] = row
        print(f"     slot {j}: inh_from_comp_attr_inh={row.get('w_from_comp_attr_inh', 0.0):9.2f} "
              f"(n={row.get('n_from_comp_attr_inh', 0)})  recurrent={row['w_recurrent']:9.2f} "
              f"(n={row['n_recurrent']})  ca1={row.get('w_from_ca1', 0.0):9.3f}  "
              f"other(pools)={row['w_from_other']:9.2f} (n={row['n_from_other']})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--arm", choices=["baseline", "freeze", "equalize"], default="baseline")
    ap.add_argument("--hebb-max", type=float, default=2.5,
                    help="matches _consol_replay_apical_probe's default so the 15/27 baseline is like-for-like")
    ap.add_argument("--out", default="research/findings/raw/cortical_store")
    args = ap.parse_args()

    a = dict(BASE)
    a.update(comp_dendritic=True, comp_wta_weight=5.0, comp_k_thresh=2.0, comp_self_regen=0.15,
             comp_kir_g=3.0, comp_v_hold=-50.0, comp_apical_R=0.15, comp_gc_read=0.5,
             comp_btsp=True, comp_btsp_lr=0.0005, comp_btsp_wmax=2000.0, comp_btsp_elig_tau=30.0,
             comp_no_pool_slot=False, comp_pool_slot_weight=1.5, comp_attractor_slots=N,
             comp_per_slot_fs=False, enable_hebbian=True)
    b = build_substrate(args.seed, SimpleNamespace(**a))
    b.core_config.hebbian_max_weight = float(args.hebb_max)
    b.core_config.enable_stdp = False

    # --- PROVENANCE + the load-bearing structural facts, MEASURED not assumed.
    thr_hash = hashlib.md5(to_host(b.cp_neuron_firing_thresholds).tobytes()).hexdigest()[:12]
    print(f"[seed {args.seed}] arm={args.arm} backend={BACKEND} thr_hash={thr_hash}")
    print(f"  cfg.seed={b.core_config.seed}  enable_homeostasis={b.core_config.enable_homeostasis} "
          f"(True => cp_neuron_firing_thresholds IS the spike threshold, bridge.py:7397)")
    print(f"  homeostasis_threshold_[min,max]=[{b.core_config.homeostasis_threshold_min}, "
          f"{b.core_config.homeostasis_threshold_max}]  adapt_rate={b.core_config.homeostasis_threshold_adapt_rate} "
          f"ema_alpha={b.core_config.homeostasis_ema_alpha}  target_rate={b.core_config.homeostasis_target_rate}")
    print(f"  hebbian_max_weight={b.core_config.hebbian_max_weight}  enable_stdp={b.core_config.enable_stdp}")

    rm = b.region_manager
    slot = {i: np.asarray(sorted(rm.indices(f"comp_attr_{i}")), dtype=np.int64) for i in range(N)}
    st_build = _slot_stats(b, slot, "AT BUILD")

    tags, _dims = encode_facts_with_reinstatement(b, CONSOLIDATED_FACTS)
    st_pre = _slot_stats(b, slot, "AFTER ENCODE / PRE-REPLAY")
    wir = _wiring_sums(b, rm, slot, "PRE-REPLAY")

    # --- the clamp for the causal arms. Applied BEFORE each step so it is the threshold the spike
    #     comparison at bridge.py:7397 actually uses (homeostasis rewrites the array at the END of a step).
    slot_all = np.concatenate([slot[j] for j in sorted(slot)])
    slot_all_gpu = cp.asarray(slot_all, dtype=cp.int64)
    thr_pre = to_host(b.cp_neuron_firing_thresholds).astype(np.float64)
    pin_vals = None
    if args.arm == "freeze":
        pin_vals = cp.asarray(thr_pre[slot_all], dtype=b.cp_neuron_firing_thresholds.dtype)
        print(f"  LEVER[freeze]: pinning {slot_all.size} slot thresholds to their own pre-replay values "
              f"(spread across slot MEANS preserved = {max(st_pre[j]['thr_mean'] for j in st_pre) - min(st_pre[j]['thr_mean'] for j in st_pre):.4f} mV)")
    elif args.arm == "equalize":
        common = float(thr_pre[slot_all].mean())
        pin_vals = cp.full(slot_all.size, common, dtype=b.cp_neuron_firing_thresholds.dtype)
        print(f"  LEVER[equalize]: pinning ALL {slot_all.size} slot thresholds to the common mean {common:.4f} mV "
              f"(pre-replay spread across slot means was "
              f"{max(st_pre[j]['thr_mean'] for j in st_pre) - min(st_pre[j]['thr_mean'] for j in st_pre):.4f} mV)")

    fire = {j: [] for j in sorted(slot)}
    orig_step = b._run_one_simulation_step

    def sampling_step(*a_, **k_):
        if pin_vals is not None:
            b.cp_neuron_firing_thresholds[slot_all_gpu] = pin_vals
        r = orig_step(*a_, **k_)
        fs = to_host(b.cp_firing_states)
        for j in sorted(slot):
            fire[j].append(float(fs[slot[j]].sum()))
        return r

    b._run_one_simulation_step = sampling_step
    try:
        coactivation_replay(b, CONSOLIDATED_FACTS, tags, int(args.cycles), args.seed,
                            coactivate=True, attractor_on=True)
    finally:
        b._run_one_simulation_step = orig_step

    # --- VERIFY THE LEVER MOVED: read the LIVE array right after replay.
    thr_post = to_host(b.cp_neuron_firing_thresholds).astype(np.float64)
    post_means = [float(thr_post[slot[j]].mean()) for j in sorted(slot)]
    post_spread = max(post_means) - min(post_means)
    within = float(np.mean([thr_post[slot[j]].std() for j in sorted(slot)]))
    print(f"  LEVER CHECK (live array, post-replay): per-slot means {[round(v, 4) for v in post_means]}  "
          f"spread={post_spread:.6f} mV  mean within-slot std={within:.6f}")
    if args.arm == "equalize" and (post_spread > 1e-4 or within > 1e-4):
        print("     ⛔ THE CLAMP DID NOT HOLD -- this arm is UNINTERPRETABLE, do not read a null off it.")
    if args.arm == "baseline":
        drift = float(np.abs(thr_post[slot_all] - thr_pre[slot_all]).mean())
        print(f"     baseline homeostatic DRIFT during replay: mean |Δthreshold| = {drift:.6f} mV "
              f"(if ~0, within-replay drift cannot be the mechanism and 'freeze' must equal 'baseline')")
    st_post = _slot_stats(b, slot, "POST-REPLAY")

    # --- windows: reconstruct the driven fact exactly as coactivation_replay shuffles them.
    rng = np.random.default_rng(int(args.seed) + 777)
    order = []
    for _c in range(int(args.cycles)):
        o = list(range(N)); rng.shuffle(o); order.extend(o)
    nw = min(len(order), len(fire[0]) // BURST)
    # excitability rank from the PRE-REPLAY thresholds (what the competition actually saw)
    pre_means = [st_pre[j]["thr_mean"] for j in sorted(slot)]
    pre_p10 = [st_pre[j]["thr_p10"] for j in sorted(slot)]
    lowest_thr = int(np.argmin(pre_means))
    lowest_p10 = int(np.argmin(pre_p10))
    print(f"  PREDICTION under candidate B: slot {lowest_thr} (lowest mean threshold {min(pre_means):.4f}) "
          f"should win regardless of which fact is driven. (lowest p10 = slot {lowest_p10})")

    rows, dom, thr_hit, wins = [], 0, 0, [0] * N
    tot_spikes = [0.0] * N
    shares = []
    for w in range(nw):
        sl = slice(w * BURST, (w + 1) * BURST)
        tot = [float(np.asarray(fire[j][sl]).sum()) for j in sorted(slot)]
        drv = order[w]
        win = int(np.argmax(tot))
        dom += (win == drv)
        thr_hit += (win == lowest_thr)
        wins[win] += 1
        for j in range(N):
            tot_spikes[j] += tot[j]
        if sum(tot) > 0:
            shares.append(tot[drv] / sum(tot))
        rows.append(dict(window=w, driven=drv, spikes=[int(t) for t in tot], winner=win))
        print(f"     window {w:2d} [fact {drv}]: spikes={[int(t) for t in tot]}  winner=slot {win}  "
              f"{'driven' if win == drv else ('lowest-thr' if win == lowest_thr else 'other')}")
    share = float(np.mean(shares)) if shares else float("nan")
    print(f"  => DRIVEN slot won {dom}/{nw} windows (chance {nw // N}/{nw}); "
          f"continuous driven-share = {share:.4f} (chance {1.0 / N:.4f})")
    print(f"  => LOWEST-THRESHOLD slot ({lowest_thr}) won {thr_hit}/{nw} windows (chance {nw // N}/{nw}). "
          f"per-slot win counts {wins}; per-slot total spikes {[int(t) for t in tot_spikes]}")
    # rank agreement over the 3 slots: threshold rank (ascending = most excitable first) vs spike rank
    thr_rank = list(np.argsort(np.argsort(pre_means)))          # 0 = lowest threshold
    spk_rank = list(np.argsort(np.argsort([-t for t in tot_spikes])))  # 0 = most spikes
    print(f"  => rank(threshold asc) {thr_rank}  vs  rank(total spikes desc) {spk_rank}  "
          f"{'MATCH' if thr_rank == spk_rank else 'MISMATCH'}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    res = dict(seed=args.seed, arm=args.arm, backend=BACKEND, thr_hash=thr_hash,
               argv=sys.argv[1:], cycles=int(args.cycles), burst=BURST,
               knobs=dict(hebbian_max_weight=float(b.core_config.hebbian_max_weight),
                          enable_homeostasis=bool(b.core_config.enable_homeostasis),
                          enable_stdp=bool(b.core_config.enable_stdp),
                          homeostasis_threshold_adapt_rate=float(b.core_config.homeostasis_threshold_adapt_rate),
                          comp_per_slot_fs=False, comp_wta_weight=5.0, comp_pool_slot_weight=1.5),
               thr_build={str(k): v for k, v in st_build.items()},
               thr_pre={str(k): v for k, v in st_pre.items()},
               thr_post={str(k): v for k, v in st_post.items()},
               wiring={str(k): v for k, v in wir.items()},
               post_slot_means=post_means, post_spread=post_spread, within_slot_std=within,
               windows=rows, driven_wins=int(dom), n_windows=int(nw), driven_share=share,
               lowest_thr_slot=lowest_thr, lowest_thr_wins=int(thr_hit),
               per_slot_wins=wins, per_slot_spikes=tot_spikes,
               thr_rank=[int(x) for x in thr_rank], spk_rank=[int(x) for x in spk_rank])
    # A FILENAME IS NOT PROVENANCE, but a filename that OMITS a knob silently OVERWRITES a different
    # experiment: the first run of this probe wrote seed42/baseline at hebb_max=2.5 and the hebb_max=20
    # arm then clobbered it under the same name. `knobs` inside the JSON is the authority; the hebb tag
    # here just stops two distinct experiments from sharing one path.
    Path(f"{args.out}/slot_excit_seed{args.seed}_{args.arm}_hebb{args.hebb_max:g}.json").write_text(
        json.dumps(res, indent=2))
    print("SLOT-EXCITABILITY-PROBE DONE")


if __name__ == "__main__":
    main()
