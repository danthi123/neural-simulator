"""TEACHER-LOOP NOISE-DRIVEN UNSUPERVISED-STDP SLEEP DE-RISK (2026-08-09): attack the BREADTH crux -- catastrophic
forgetting in the sequential teacher-loop -- with Bazhenov's SLEEP-REPLAY-CONSOLIDATION mechanism, which is
MECHANISTICALLY DIFFERENT from our supervised self-replay baseline.

THE HYPOTHESIS (why this could beat self-replay in the overlapping N=20 regime). Our best mechanism so far is
IN-RUN SELF-REPLAY: the brain regenerates its OWN stored engram patterns and re-consolidates them into the shared
leaky-readout by SUPERVISED e-prop (the taught class label rides along with each replayed pattern). It reaches
6-seed retention ~0.85 @ N=10 but DEGRADES to ~0.45 @ N=20 -- as facts overlap, the supervised targets fight over
the same readout weights. Bazhenov's SNN sleep works DIFFERENTLY: after sequential acquisition it SILENCES the cue
input, drives the network with POISSON NOISE, and switches the readout plasticity from supervised e-prop to
UNSUPERVISED (Hebbian) spike-timing STDP -- NO class label. The claim (Golden, Delanois, Sanda, Bazhenov 2022,
PLOS Comput Biol 18(11):e1010628) is that noise-driven reactivation + unsupervised STDP moves the synapses to the
INTERSECTION of the task manifolds (a 'joint synaptic weight representation') that satisfies ALL facts at once,
rather than the most-recent one -- and that the noise specificity is NOT critical (broadband Poisson suffices).
Their forgetting floor was 0.52 (chance), single-task 0.70, sleep-recovered 0.70/0.68.

BRAIN-BASED SLEEP (the load-bearing distinction from supervised self-replay). The sleep phase here is NEURAL and
UNSUPERVISED end-to-end:
  * SILENCE THE CUE (teacher + world ABSENT): the input slice carries NO percept -- the structured sensory cue is
    silenced. Instead the network is driven by broadband POISSON NOISE current (a brain-owned noise RNG), which
    propagates through the ALREADY-TRAINED input->hidden feedforward weights (which carry each fact's structure
    from acquisition) and reactivates the readout-upstream hidden units. Optionally the noise is injected directly
    into the hidden slice (`--noise-site hidden`, the literal "drive the readout-upstream units with noise" reading).
  * UNSUPERVISED STDP (no label): the hidden units SPIKE under noise; those spikes propagate through the readout
    synapses (H_last->out in cp_connections) and the output neurons SPIKE; PAIR-BASED spike-timing STDP
    (LTP for pre-before-post, LTD for post-before-pre) then updates the readout synapse weights DIRECTLY in
    cp_connections.data -- noise -> spikes -> STDP on synapses, NO class label, NO target, NO gradient, NO host
    weight-average. `_noise_stdp_sleep_consolidate` has NO label/target/class parameter (grep-verifiable).

FOUR ARMS, same net build / seed / per-fact teaching budget (the ONLY difference is the consolidation phase):
  * NOSLEEP    = the scaling baseline: teach each fact, NO consolidation -> frac_recalled ~ 1/N (the wall, lesion).
  * SELFREPLAY = the MEASURED supervised self-replay baseline (imported VERBATIM from the sleep-replay derisk:
                 regenerate stored engrams + re-consolidate by SUPERVISED e-prop). The thing to beat.
  * NOISESTDP  = teach each fact, THEN the noise-driven UNSUPERVISED-STDP sleep phase (this arm).
  * SCRAMBLE-ACQ (optional, --with-scramble) = NOISESTDP but ACQUISITION labels shuffled: probes whether the sleep
                 recovers anything the readout never encoded (it should not; it is a sanity floor, not the label
                 control). The LABEL-INDEPENDENCE of the sleep step itself is proven by the byte-identical-delta
                 check (`--label-check`), not by this arm.

TEETH:
  (a) BEATS SELF-REPLAY: NOISESTDP frac_recalled >= SELFREPLAY frac_recalled at BOTH N=10 and N=20 (the decisive
      comparison; the crux regime is N=20 where self-replay degrades to ~0.45). GO requires the win at N=20.
  (b) LOAD-BEARING: NOSLEEP (== the sleep phase lesioned) forgets -> the sleep phase is what carries any rise.
  (c) IMMEDIATE ACQUISITION STAYS PERFECT: acquire_acc measured right after teaching each fact (BEFORE the sleep
      phase), in the NOISESTDP arm, stays high -- the sleep phase must not destroy the just-taught fact.
  (d) NEURAL + UNSUPERVISED: the sleep phase drives real substrate spikes and updates cp_connections by STDP
      (mean output-spike rate during sleep + total STDP weight moved are REPORTED, so a starved/inert sleep is
      visible); the sleep function takes NO label (grep + the byte-identical-delta `--label-check`: scrambling any
      label array in scope leaves the sleep weight update byte-identical, because the step cannot read a label).
  grep-verify TEACHER/WORLD/LABEL ABSENT during the sleep step:
      grep -n 'def _noise_stdp_sleep_consolidate' research/runners/_teacher_loop_noise_stdp_sleep_derisk.py
      -> the signature has no env / y / cls / label / target; grep those tokens inside its body -> empty.

HONEST NEGATIVE WITH TEETH is a first-class deliverable: if unsupervised noise-STDP does NOT beat supervised
self-replay (likely, given the substrate's readout is a NON-spiking leaky integrator whose weak readout synapses
starve the output of the post-spikes STDP needs), the exact residual is reported (does it beat NOSLEEP at all? is
the output spiking during sleep? how much weight did STDP move?) -- that MAPS what unsupervised STDP can/can't do
on this substrate, which is the science.

DISCIPLINE: reuse-by-import (the scaling machinery + ReferentEnv + the sleep-replay Hippocampus/self-replay). NO
sim/ edit (assert `git diff main -- sim/` empty). cfg.seed via the seed= the net passes to CoreSimConfig.seed
(NOT actual_seed_used). SIM_BACKEND=numpy (the teacher-loop substrate is numpy). tools.lab attribution + a Verdict
preconditions block.

RUN (one seed per process; --n-max 20 --milestones 10 20 measures BOTH N=10 and N=20 in ONE run):
  SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
    python -m research.runners._teacher_loop_noise_stdp_sleep_derisk --seed 42 \
      --n-max 20 --milestones 10 20 --out research/findings/raw/teacher_loop_noise_stdp_sleep_s42.json
  6-SEED (GO needs NOISESTDP >= SELFREPLAY at N=20, 6/6 at 42..47):
    for s in 42 43 44 45 46 47; do SIM_BACKEND=numpy PYTHONPATH=$PWD OPENBLAS_NUM_THREADS=1 \
      python -m research.runners._teacher_loop_noise_stdp_sleep_derisk --seed $s \
      --n-max 20 --milestones 10 20 \
      --out research/findings/raw/teacher_loop_noise_stdp_sleep_s$s.json & done; wait
  PLUMBING SMOKE: ... --n-max 3 --milestones 3 --epochs 8 --replay-epochs 6 --n-draws 12 --sleep-cycles 8
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("SIM_BACKEND", "numpy")   # the teacher-loop substrate is numpy
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
# reuse-by-import: the teacher-loop SCALING machinery + the world + the SUPERVISED self-replay baseline. NO sim/ edit.
from research.runners._teacher_loop_scaling_derisk import (  # noqa: E402
    _mk_net, _feat, _fit_readout_norm_world, _teach_fact, _fact_acc, _corrective_batch, N_ACT,
)
from research.runners._teacher_loop_corrective_acquire_derisk import ReferentEnv  # noqa: E402
from research.runners._teacher_loop_sleep_replay_consolidation_derisk import (  # noqa: E402
    Hippocampus, _self_replay_consolidate,
)

OUT = _REPO / "research" / "findings" / "raw" / "teacher_loop_noise_stdp_sleep.json"


# =============================== NEURAL NOISE-DRIVEN sleep forward + UNSUPERVISED STDP ===============================
def _sleep_forward_record(net, brain_rng, noise_rate, noise_amp, settle, site, sleep_o_tonic):
    """One SLEEP presentation. Teacher + world ABSENT: the cue input is SILENCED (no percept). The network is driven
    by broadband POISSON NOISE current (brain-owned RNG); the hidden/output slices SPIKE and we RECORD every neuron's
    spike per settle step. Returns sp (T, n_total). This is the substrate's own forward (_run_one_simulation_step),
    NOT a host computation -- the only host role is silencing the cue and injecting the noise current (world/body)."""
    from sim.backend import to_host
    xp = net._xp
    n = net.n_total
    # per-presentation WASH-OUT (the substrate's own EMERGE-61 precedent) so each noise presentation is a fresh draw.
    if net.reset_state:
        for name, arr0 in net._state0.items():
            getattr(net.br, name)[...] = arr0
    if net.br.cp_bdsp_E is not None:
        net.br.cp_bdsp_E[...] = 0.0
        net.br.cp_bdsp_B[...] = 0.0
        net.br.cp_bdsp_last_spike_step = xp.full(n, -1000000, dtype=xp.int64)
    base = net._base_drive()                  # tonic on hidden+output, 0 on the input slice (cue silenced)
    base = np.asarray(base, dtype=np.float32).copy()
    if sleep_o_tonic is not None:
        base[net.slices[-1]] = np.float32(sleep_o_tonic)   # bring output neurons near threshold so noise-reactivated
        #                                                    hidden drive can tip them into spikes (else the weak
        #                                                    leaky-readout synapses give ~0 post-spikes -> STDP starves)
    noise_sl = net.slices[0] if site == "input" else net.slices[-2]
    m = noise_sl.stop - noise_sl.start
    T = int(settle)
    sp = np.zeros((T, n), dtype=np.float32)
    for t in range(T):
        drive = base.copy()
        counts = brain_rng.poisson(noise_rate, size=m).astype(np.float32)   # independent Poisson noise per neuron/step
        drive[noise_sl] = drive[noise_sl] + noise_amp * counts
        net.br.cp_external_input_current = xp.asarray(drive)
        net.br._run_one_simulation_step()
        sp[t] = np.asarray(to_host(net.br.cp_firing_states), dtype=np.float32)
    return sp


def _stdp_readout_delta(net, sp, a_plus, a_minus, tau_stdp):
    """PAIR-BASED spike-timing STDP weight-change for the readout synapses (H_last pre -> output post), computed from
    the RECORDED substrate spikes ONLY -- NO label, NO target, NO error. LTP for pre-before-post, LTD for
    post-before-pre (Bi & Poo pair rule). Returns dw (n_pre_phys, n_post_phys), row(pre)-major (aligns with
    net._data_idx_flat[-1])."""
    pre = sp[:, net.slices[-2]].astype(np.float64)          # (T, H_phys)
    post = sp[:, net.slices[-1]].astype(np.float64)         # (T, K_phys)
    decay = float(np.exp(-1.0 / max(1e-6, tau_stdp)))
    H = pre.shape[1]
    K = post.shape[1]
    x = np.zeros(H, dtype=np.float64)                       # pre eligibility trace (spikes strictly BEFORE now)
    y = np.zeros(K, dtype=np.float64)                       # post eligibility trace (spikes strictly BEFORE now)
    dltp = np.zeros((H, K), dtype=np.float64)
    dltd = np.zeros((H, K), dtype=np.float64)
    for t in range(pre.shape[0]):
        dltp += np.outer(x, post[t])                        # a post spike now, potentiated by earlier pre (pre->post)
        dltd += np.outer(pre[t], y)                         # a pre spike now, depressed by earlier post (post->pre)
        x = x * decay + pre[t]
        y = y * decay + post[t]
    return a_plus * dltp - a_minus * dltd                   # (H, K)


def _apply_readout_dw(net, dw, lr_stdp):
    """Write the STDP weight-change into the readout synapses in cp_connections.data (the SAME position map + plastic
    mask + rate gain the e-prop path uses). Signed-clamped to [-w_clip, w_clip]. NEURAL: synapse weights move.
    Returns the total |actual weight change| (post- vs pre-write, after mask/gain/clip)."""
    from sim.backend import to_host
    xp = net._xp
    data = net.br.cp_connections.data
    idx = net._data_idx_flat[-1]
    delta = xp.asarray((lr_stdp * dw).astype(np.float32).ravel())
    cur = data[idx]
    new = xp.clip(cur + delta, -net.w_clip, net.w_clip)
    if net.br.cp_synapse_plastic_mask is not None:
        pm = net.br.cp_synapse_plastic_mask[idx]
        new = xp.where(pm, new, cur)
    if net.br.cp_plasticity_rate_gain is not None:
        gain = net.br.cp_plasticity_rate_gain[idx]
        new = cur + (new - cur) * gain
    moved = float(np.sum(np.abs(np.asarray(to_host(new)) - np.asarray(to_host(cur)))))
    data[idx] = new
    return moved


def _noise_stdp_sleep_consolidate(net, brain_rng, cycles, noise_rate, noise_amp, settle, site, sleep_o_tonic,
                                  a_plus, a_minus, tau_stdp, lr_stdp):
    """OFFLINE SLEEP consolidation -- NOISE-DRIVEN, UNSUPERVISED. NO env, NO teacher, NO label/target/class parameter
    (grep-verify: no `env`, `y`, `cls`, `label`, `target` here). For each of `cycles` noise presentations: drive the
    network with broadband Poisson noise (cue silenced), the substrate SPIKES, and PAIR-STDP updates the readout
    synapses in cp_connections from those spikes ALONE. Returns (mean_output_spikes_per_cycle, total_abs_weight_moved,
    mean_hidden_spikes_per_cycle)."""
    out_spk = 0.0
    hid_spk = 0.0
    moved = 0.0
    for _c in range(int(cycles)):
        sp = _sleep_forward_record(net, brain_rng, noise_rate, noise_amp, settle, site, sleep_o_tonic)
        out_spk += float(sp[:, net.slices[-1]].sum())
        hid_spk += float(sp[:, net.slices[-2]].sum())
        dw = _stdp_readout_delta(net, sp, a_plus, a_minus, tau_stdp)
        moved += _apply_readout_dw(net, dw, lr_stdp)
    c = max(1, int(cycles))
    return out_spk / c, moved, hid_spk / c


# ================================= one arm of the sequential curriculum =================================
def _run_arm(arm, seed, referents, env, K, n_in, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws,
             milestones, test_n, chance, sleep_cfg, replay_cfg, scramble_acq=False):
    """Teach the referents SEQUENTIALLY into ONE brain. arm in {nosleep, selfreplay, noisestdp}. For noisestdp, after
    teaching each fact run the noise-driven unsupervised-STDP sleep phase; for selfreplay, the SUPERVISED self-replay."""
    net = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    _fit_readout_norm_world(net, env, referents, seed)
    teach_rng = np.random.default_rng(seed + 777)
    brain_rng = np.random.default_rng(seed + 313)            # brain-owned RNG (noise + replay shuffling)
    acq_label_rng = np.random.default_rng(seed + 2027)       # only for the scramble-acq sanity floor
    hippo = Hippocampus(seed, replay_noise=replay_cfg["replay_noise"]) if arm == "selfreplay" else None

    acquire_acc = []
    retention = {}
    sleep_diag = {"mean_out_spikes_per_cycle": [], "mean_hidden_spikes_per_cycle": [], "abs_weight_moved": []}
    n_ref = len(referents)
    for i, r in enumerate(referents):
        # --- WAKE: teacher teaches fact i from the world (env draws are legitimate: the sensory environment) ---
        X, y = _corrective_batch(env, r, i, n_draws)
        if scramble_acq:                                     # sanity floor only: teach with a shuffled target class
            y = np.full(n_draws, int(acq_label_rng.integers(0, K)), dtype=np.int64)
        _teach_fact(net, X, y, epochs, batch, teach_rng)
        acq = _fact_acc(net, env, r, i, n=test_n)
        acquire_acc.append(acq)
        # --- SLEEP / REPLAY: offline consolidation (teacher + world ABSENT); nosleep SKIPS this ---
        if arm == "selfreplay":
            hippo.encode(X, i)
            _self_replay_consolidate(net, hippo, replay_cfg["replay_epochs"], batch, brain_rng,
                                     replay_cfg["replay_per_fact"], scramble=False)
        elif arm == "noisestdp":
            o, mv, h = _noise_stdp_sleep_consolidate(
                net, brain_rng, sleep_cfg["cycles"], sleep_cfg["noise_rate"], sleep_cfg["noise_amp"],
                settle, sleep_cfg["site"], sleep_cfg["sleep_o_tonic"], sleep_cfg["a_plus"], sleep_cfg["a_minus"],
                sleep_cfg["tau_stdp"], sleep_cfg["lr_stdp"])
            sleep_diag["mean_out_spikes_per_cycle"].append(o)
            sleep_diag["mean_hidden_spikes_per_cycle"].append(h)
            sleep_diag["abs_weight_moved"].append(mv)
        N = i + 1
        if N in milestones:
            accs = [_fact_acc(net, env, referents[j], j, n=test_n) for j in range(N)]
            n_recalled = int(sum(a >= max(0.5, chance + 0.15) for a in accs))
            retention[str(N)] = {
                "frac_recalled": float(n_recalled / N), "n_recalled": n_recalled, "N": N,
                "mean_retained_acc": float(np.mean(accs)),
                "oldest_fact_acc": float(accs[0]), "most_recent_fact_acc": float(accs[-1]),
                "per_fact_acc": [float(a) for a in accs],
            }
    out = {
        "arm": arm,
        "acquire_acc_immediate": [float(a) for a in acquire_acc],
        "mean_acquire_acc_immediate": float(np.mean(acquire_acc)) if acquire_acc else float("nan"),
        "retention_curve": retention,
    }
    if arm == "noisestdp":
        out["sleep_diagnostics"] = {
            "mean_out_spikes_per_cycle": float(np.mean(sleep_diag["mean_out_spikes_per_cycle"])) if sleep_diag["mean_out_spikes_per_cycle"] else 0.0,
            "mean_hidden_spikes_per_cycle": float(np.mean(sleep_diag["mean_hidden_spikes_per_cycle"])) if sleep_diag["mean_hidden_spikes_per_cycle"] else 0.0,
            "total_abs_weight_moved": float(np.sum(sleep_diag["abs_weight_moved"])),
        }
    return out


def _acquire_and_sleep(seed, referents, env_seed, d_p, noise, K, n_in, hidden, settle, epochs, batch, eprop_lr,
                       w_clip, n_draws, sleep_cfg, n_facts, make_scrambled_labels):
    """Build a FRESH net (the build re-seeds the substrate deterministically from cfg.seed), acquire n_facts, then run
    ONE noise-STDP sleep consolidation, and return the final readout-synapse weights. If make_scrambled_labels, a
    scrambled label array is constructed (from an ISOLATED RNG so global state is untouched) and left in scope -- the
    sleep step has no label parameter and cannot read it. Two calls that differ ONLY in make_scrambled_labels must
    return byte-identical readout weights => the label cannot enter the sleep update."""
    env = ReferentEnv(env_seed, d_p=d_p, noise=noise)
    for r in referents:
        env.proto(r)
    env.rng = np.random.default_rng(env_seed + 101)
    net = _mk_net(n_in, K, seed, hidden, settle, eprop_lr, w_clip)
    _fit_readout_norm_world(net, env, referents, seed)
    teach_rng = np.random.default_rng(seed + 777)
    for i in range(int(n_facts)):
        X, y = _corrective_batch(env, referents[i], i, n_draws)
        _teach_fact(net, X, y, epochs, batch, teach_rng)
    if make_scrambled_labels:
        _scrambled = np.random.default_rng(seed + 999).permutation(np.arange(int(n_facts)))  # noqa: F841 -- sleep ignores it
    br = np.random.default_rng(seed + 313)
    _noise_stdp_sleep_consolidate(net, br, sleep_cfg["cycles"], sleep_cfg["noise_rate"], sleep_cfg["noise_amp"],
                                  settle, sleep_cfg["site"], sleep_cfg["sleep_o_tonic"], sleep_cfg["a_plus"],
                                  sleep_cfg["a_minus"], sleep_cfg["tau_stdp"], sleep_cfg["lr_stdp"])
    from sim.backend import to_host
    return np.asarray(to_host(net.br.cp_connections.data[net._data_idx_flat[-1]])).copy()


def _label_independence_check(seed, referents, env_seed, d_p, noise, K, n_in, hidden, settle, epochs, batch,
                              eprop_lr, w_clip, n_draws, sleep_cfg, n_facts=5):
    """ANTI-CHEAT (d): the sleep STDP step is UNSUPERVISED. Two FRESH, identically-built + identically-acquired nets
    run the IDENTICAL sleep consolidation (same brain RNG); the ONLY difference is that one has a scrambled label
    array in scope. Because the sleep step has no label parameter, the resulting readout weights are byte-identical.
    Returns the max abs difference (0.0 == a label provably cannot enter the sleep update)."""
    w_a = _acquire_and_sleep(seed, referents, env_seed, d_p, noise, K, n_in, hidden, settle, epochs, batch, eprop_lr,
                             w_clip, n_draws, sleep_cfg, n_facts, make_scrambled_labels=False)
    w_b = _acquire_and_sleep(seed, referents, env_seed, d_p, noise, K, n_in, hidden, settle, epochs, batch, eprop_lr,
                             w_clip, n_draws, sleep_cfg, n_facts, make_scrambled_labels=True)
    return float(np.max(np.abs(w_a - w_b)))


def run(seed, n_max, milestones, hidden, settle, epochs, batch, eprop_lr, w_clip, n_draws, d_p, noise, test_n,
        replay_cfg, sleep_cfg, with_scramble, do_label_check):
    K = int(n_max)
    chance = 1.0 / K
    n_in = d_p + N_ACT
    milestones = sorted(set(int(m) for m in milestones if 1 <= int(m) <= n_max))

    env = ReferentEnv(seed, d_p=d_p, noise=noise)
    referents = [f"ref{i}" for i in range(n_max)]
    for r in referents:
        env.proto(r)

    arms = {}
    arm_names = ["nosleep", "selfreplay", "noisestdp"]
    if with_scramble:
        arm_names.append("scramble_acq")
    for arm in arm_names:
        t0 = time.time()
        env.rng = np.random.default_rng(seed + 101)          # SAME teaching percepts per arm (reset the env stream)
        arms[arm] = _run_arm(arm if arm != "scramble_acq" else "noisestdp", seed, referents, env, K, n_in, hidden,
                             settle, epochs, batch, eprop_lr, w_clip, n_draws, milestones, test_n, chance,
                             sleep_cfg, replay_cfg, scramble_acq=(arm == "scramble_acq"))
        arms[arm]["arm"] = arm
        arms[arm]["wall_seconds"] = round(time.time() - t0, 1)
        rc = arms[arm]["retention_curve"]
        big = max((int(k) for k in rc), default=None)
        fr = rc[str(big)]["frac_recalled"] if big else float("nan")
        print(f"[arm {arm}] done in {arms[arm]['wall_seconds']:.0f}s | immediate-acq "
              f"{arms[arm]['mean_acquire_acc_immediate']:.3f} | frac-recalled@N={big}: {fr:.2f}", flush=True)

    label_indep_maxdiff = None
    if do_label_check:
        label_indep_maxdiff = _label_independence_check(seed, referents, seed, d_p, noise, K, n_in, hidden, settle,
                                                        epochs, batch, eprop_lr, w_clip, n_draws, sleep_cfg,
                                                        n_facts=min(5, n_max))
        print(f"[label-check] sleep-update byte-diff with scrambled labels in scope: {label_indep_maxdiff:.2e} "
              f"(0.0 == label cannot enter the sleep step)", flush=True)

    return {"seed": seed, "K_classes": K, "chance": chance, "n_max": n_max, "milestones": milestones,
            "config": {"hidden": hidden, "settle_steps": settle, "epochs": epochs, "batch": batch,
                       "eprop_lr": eprop_lr, "w_clip": w_clip, "n_draws": n_draws, "d_p": d_p, "noise": noise,
                       "test_n": test_n, "replay_cfg": replay_cfg, "sleep_cfg": sleep_cfg},
            "arms": arms, "sleep_label_independent_maxdiff": label_indep_maxdiff}


def _verdict(result):
    """Verdict preconditions + GO. TEETH:
      (a) BEATS SELF-REPLAY at N=20 (crux) AND N=10: noisestdp frac >= selfreplay frac (GO gates on N=20).
      (b) LOAD-BEARING: nosleep forgets (< 0.5) at the largest N.
      (c) immediate acquisition stays high in noisestdp (>= 0.9).
      (d) sleep is neural + unsupervised: output spikes during sleep > 0, weight moved > 0, label byte-diff == 0."""
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    rc = {a: result["arms"][a]["retention_curve"] for a in result["arms"]}
    chance = result["chance"]
    Ns = sorted(int(k) for k in rc["noisestdp"])
    big = Ns[-1] if Ns else None
    key = str(big)

    def fr(a, k):
        return rc[a][str(k)]["frac_recalled"] if str(k) in rc[a] else float("nan")

    nosleep_big = fr("nosleep", big)
    selfreplay_big = fr("selfreplay", big)
    noisestdp_big = fr("noisestdp", big)
    noise_acq = result["arms"]["noisestdp"]["mean_acquire_acc_immediate"]
    diag = result["arms"]["noisestdp"].get("sleep_diagnostics", {})
    out_spk = diag.get("mean_out_spikes_per_cycle", 0.0)
    wt_moved = diag.get("total_abs_weight_moved", 0.0)
    label_diff = result.get("sleep_label_independent_maxdiff", None)

    attributable_to("noise-STDP sleep vs supervised self-replay", noisestdp_big, selfreplay_big)
    attributable_to("noise-STDP sleep phase (vs no-sleep)", noisestdp_big, nosleep_big)

    # PRECONDITIONS are VALIDITY checks (they make the comparison MEANINGFUL); the "beats self-replay" HYPOTHESIS is
    # the `go` boolean below, NOT a precondition -- a legitimate negative (mechanism ran fine, underperformed) must
    # report as NO-GO, never as UNDEFINED (which is reserved for an instrument failure).
    v = Verdict("teacher-loop noise-driven unsupervised-STDP sleep", chance=chance)
    v.require("(b) no-sleep forgets (baseline valid, sleep load-bearing)", nosleep_big < 0.5, expect=True,
              note=f"nosleep frac_recalled@N={big} = {nosleep_big:.2f}")
    v.floor("(c) immediate acquisition stays perfect (NOISESTDP)", noise_acq, floor=0.9)
    v.require("(d1) sleep drives output spikes (neural, not inert)", out_spk > 0.0, expect=True,
              note=f"mean output spikes/cycle = {out_spk:.2f}")
    v.require("(d2) sleep moves readout synapses (STDP applied)", wt_moved > 0.0, expect=True,
              note=f"total abs weight moved = {wt_moved:.2f}")
    if label_diff is not None:
        v.require("(d3) sleep step is label-independent (byte-identical)", label_diff == 0.0, expect=True,
                  note=f"max byte-diff with scrambled labels = {label_diff:.2e}")

    # THE HYPOTHESIS (go): noise-STDP sleep beats supervised self-replay at the CRUX N (largest) AND at N=10.
    beats_at_10 = True
    if 10 in Ns and big != 10:
        beats_at_10 = fr("noisestdp", 10) >= fr("selfreplay", 10)
    go = (noisestdp_big >= selfreplay_big and beats_at_10 and nosleep_big < 0.5 and noise_acq >= 0.9
          and out_spk > 0.0 and wt_moved > 0.0 and (label_diff is None or label_diff == 0.0))
    decision = v.decide(go=go)
    return {
        "largest_N": big,
        "nosleep_frac_recalled": nosleep_big, "selfreplay_frac_recalled": selfreplay_big,
        "noisestdp_frac_recalled": noisestdp_big, "noisestdp_immediate_acq": noise_acq,
        "beats_selfreplay_at_largest_N": bool(noisestdp_big >= selfreplay_big),
        "beats_selfreplay_at_N10": bool(beats_at_10),
        "noisestdp_minus_selfreplay": float(noisestdp_big - selfreplay_big),
        "noisestdp_minus_nosleep": float(noisestdp_big - nosleep_big),
        "sleep_mean_output_spikes_per_cycle": out_spk, "sleep_total_abs_weight_moved": wt_moved,
        "sleep_label_independent_maxdiff": label_diff,
        **decision,
    }


def main():
    ap = argparse.ArgumentParser(description="Teacher-loop NOISE-DRIVEN UNSUPERVISED-STDP sleep (Bazhenov 2022): "
                                             "after sequential acquisition, silence the cue, drive with Poisson noise, "
                                             "switch the readout to unsupervised spike-timing STDP; beat catastrophic "
                                             "forgetting WITHOUT the supervised self-replay label.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-max", type=int, default=20)
    ap.add_argument("--milestones", type=int, nargs="+", default=[10, 20])
    ap.add_argument("--hidden", type=int, default=24)
    ap.add_argument("--settle-steps", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=40, help="per-fact WAKE teaching epochs")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--eprop-lr", type=float, default=0.5)
    ap.add_argument("--w-clip", type=float, default=4000.0)
    ap.add_argument("--n-draws", type=int, default=32)
    ap.add_argument("--d-p", type=int, default=12)
    ap.add_argument("--noise", type=float, default=0.12)
    ap.add_argument("--test-n", type=int, default=40)
    # SUPERVISED self-replay baseline (imported) -- matched budget to the banked sleep-replay derisk.
    ap.add_argument("--replay-epochs", type=int, default=24)
    ap.add_argument("--replay-per-fact", type=int, default=16)
    ap.add_argument("--replay-noise", type=float, default=0.10)
    # NOISE-DRIVEN UNSUPERVISED-STDP sleep knobs.
    ap.add_argument("--sleep-cycles", type=int, default=48, help="Poisson-noise presentations per consolidation")
    ap.add_argument("--noise-rate", type=float, default=0.5, help="Poisson lambda per noise-driven neuron per step")
    ap.add_argument("--noise-amp", type=float, default=1200.0, help="pA per Poisson event (drive scale)")
    ap.add_argument("--noise-site", choices=["input", "hidden"], default="input",
                    help="inject Poisson noise at the input slice (propagates through TRAINED input->hidden weights, "
                         "reactivating readout-upstream units -- the Bazhenov reading) or directly at the hidden slice")
    ap.add_argument("--sleep-o-tonic", type=float, default=480.0,
                    help="output-slice tonic during sleep (near threshold) so noise-reactivated hidden drive can tip "
                         "the weak leaky-readout synapses into post-spikes STDP needs; None uses the base tonic_o")
    ap.add_argument("--a-plus", type=float, default=1.0, help="STDP LTP amplitude (pre-before-post)")
    ap.add_argument("--a-minus", type=float, default=1.0, help="STDP LTD amplitude (post-before-pre)")
    ap.add_argument("--tau-stdp", type=float, default=4.0, help="STDP trace time constant (settle steps)")
    ap.add_argument("--lr-stdp", type=float, default=2.0, help="STDP weight-update learning rate")
    ap.add_argument("--with-scramble", action="store_true", help="add the scramble-acq sanity-floor arm")
    ap.add_argument("--no-label-check", action="store_true", help="skip the byte-identical label-independence check")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    t0 = time.time()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    replay_cfg = {"replay_epochs": a.replay_epochs, "replay_per_fact": a.replay_per_fact, "replay_noise": a.replay_noise}
    sleep_cfg = {"cycles": a.sleep_cycles, "noise_rate": a.noise_rate, "noise_amp": a.noise_amp, "site": a.noise_site,
                 "sleep_o_tonic": (None if a.sleep_o_tonic < 0 else a.sleep_o_tonic), "a_plus": a.a_plus,
                 "a_minus": a.a_minus, "tau_stdp": a.tau_stdp, "lr_stdp": a.lr_stdp}

    result = run(a.seed, a.n_max, a.milestones, a.hidden, a.settle_steps, a.epochs, a.batch, a.eprop_lr, a.w_clip,
                 a.n_draws, a.d_p, a.noise, a.test_n, replay_cfg, sleep_cfg, a.with_scramble, not a.no_label_check)
    verdict = _verdict(result)
    summary = {"probe": "teacher_loop_noise_stdp_sleep", "seed": a.seed, "backend": os.environ.get("SIM_BACKEND"),
               "elapsed_seconds": round(time.time() - t0, 1), "result": result, "verdict": verdict}
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 100, flush=True)
    print(f"[noise-stdp-sleep] seed {a.seed} @ N={verdict['largest_N']}: "
          f"NOSLEEP {verdict['nosleep_frac_recalled']:.2f} | SELFREPLAY {verdict['selfreplay_frac_recalled']:.2f} | "
          f"NOISESTDP {verdict['noisestdp_frac_recalled']:.2f} (chance {result['chance']:.2f})", flush=True)
    print(f"[noise-stdp-sleep] noisestdp-selfreplay {verdict['noisestdp_minus_selfreplay']:+.2f} | "
          f"noisestdp-nosleep {verdict['noisestdp_minus_nosleep']:+.2f} | acq {verdict['noisestdp_immediate_acq']:.3f} | "
          f"sleep out-spk/cyc {verdict['sleep_mean_output_spikes_per_cycle']:.1f} | wt-moved "
          f"{verdict['sleep_total_abs_weight_moved']:.0f} | label-diff {verdict['sleep_label_independent_maxdiff']} | "
          f"VERDICT {verdict['status']}", flush=True)
    for arm in result["arms"]:
        rc = result["arms"][arm]["retention_curve"]
        line = " ".join(f"N={k}:{rc[k]['n_recalled']}/{k}({rc[k]['frac_recalled']:.2f})" for k in sorted(rc, key=int))
        print(f"    {arm:12s}: {line}", flush=True)
    print(f"[noise-stdp-sleep] wrote {a.out}\n" + "=" * 100, flush=True)
    return 0 if verdict["status"] == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
