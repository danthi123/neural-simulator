"""JOINT ATTENTION (theory-of-mind Stage-0): an OTHER-ATTENTION-SCHEMA that ALIGNS the agent's attentional
spotlight to the target a PARTNER is INFERRED to be attending to, from the partner's gaze/biological-motion cue.

BIOLOGY (READ, binding: research/biology/joint-attention-gaze-following.md). Kandel, Principles of Neural Science,
Ch. 62, Fig. 62-4 "The mentalizing system of the brain": the second of the four mentalizing components, "in the
temporoparietal region of the superior temporal lobe, is known to be activated by EYE GAZE and BIOLOGICAL MOTION"
-- the STS-temporoparietal junction, the perceptual front end that reads another agent's DIRECTION of attention.
Developmentally, "mutual attention normally appears toward the end of the first year when signs of mentalizing are
still sparse": joint/mutual attention is a DISSOCIABLE, EARLIER precursor to full false-belief mentalizing (which is
W3 / the Sally-Anne register, mPFC). This de-risk builds that Stage-0 precursor: gaze DIRECTION in -> inferred
attended object out -> the agent's spotlight aligns to it.

MECHANISM (fully spiking; reuse-by-import the project's spiking one-of-K WTA organ; NO `sim/` edit):
  1. GAZE CODE (STS eye-gaze / biological-motion input). A ring of N_DIR direction-tuned Izhikevich neurons
     (preferred angles evenly on the circle). The partner looks at object t*; its gaze is that object's angular
     direction + biological-motion noise. Each ring neuron i is driven by cos-tuned current about the gaze angle
     and FIRES -> spike rates r[i] are the partner's attention-direction population code. This is the retinal read
     of the partner's eyes/body (legit body/world -> neural input); the identity t* is NEVER transmitted.
  2. OTHER-ATTENTION-SCHEMA (STS-TPJ object cells). K object-selective cells; cell k has direction-tuned SYNAPSES
     W_obj[k,i] = relu(cos(theta_i - phi_k)) onto the gaze ring, where phi_k is object k's CURRENT angular position
     (the layout, randomized per trial). Its drive s[k] = W_obj[k,:] @ r is a dendritic sum of gaze-ring SPIKES:
     high for the object lying along the gaze direction. This is the INFERENCE -- a conjunction of gaze-with-layout,
     not a copy of a coordinate.
  3. ATTENTIONAL SPOTLIGHT (reused GO organ). s[K] drives build_fswta_score_bridge / fswta_drive (K Izhikevich
     attractor pools + shared inhibitory FS lateral inhibition -> a clean one-of-K spiking winner). The winner pool
     that FIRES = the agent's inferred attended object. align = (winner == t*).

GO GATE (6-seed 42/43/44/100/101/102, CPU numpy; chance = 1/K):
  (a) align_acc            >= 0.85  -- the spotlight tracks the partner's actual attended object (>> 1/K).
  (b) LESION other-schema (STS-TPJ read severed: W_obj := 0 -> spotlight gets no gaze-derived drive):
        align_acc_lesion   <= 1/K + 0.10  -- the schema OUTPUT is load-bearing; without it, chance.
  (c) SCRAMBLE partner gaze (gaze angles permuted across trials, scored vs the TRUE t*):
        align_acc_scramble <= 1/K + 0.10  -- the alignment rides the ACTUAL partner gaze, not a fixed response.
  (d) NOT-A-COPY (layout-blind baseline: decode by a FIXED angular bin, ignoring the per-trial object layout):
        align_acc_blind    <= 1/K + 0.10  -- gaze alone cannot name the object; the answer needs gaze x layout,
                                             so the intact success is an INFERENCE, not a copy of a visible index.

DISCIPLINE: SIM_BACKEND=numpy (CPU lane), reuse-by-import, NO `sim/` edit. cfg.seed set per-seed inside every
bridge (build_fswta_score_bridge sets cfg.seed; the gaze ring sets cfg.seed too) -- NOT actual_seed_used
(CLAUDE.md substrate-seeding gotcha). The task RNG is seeded per-seed as well.

Run (smoke):  SIM_BACKEND=numpy python -u -m research.runners._joint_attention_derisk --smoke --seed 42 \
                  --json research/findings/raw/_joint_attention_smoke.json
Run (6-seed): SIM_BACKEND=numpy python -u -m research.runners._joint_attention_derisk \
                  --seeds 42 43 44 100 101 102 --n-trials 60 \
                  --json research/findings/raw/_joint_attention/summary_6seed.json
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse-by-import: the project's spiking one-of-K attentional-spotlight organ (K Izhikevich attractor pools +
# shared inhibitory FS pool with lateral inhibition -> a clean one-of-K spiking winner).
from research.runners._d3_spiking_attractor_derisk import (  # noqa: E402
    build_fswta_score_bridge, fswta_drive,
)


# ---------------------------------------------------------------------------------------------------------------
# The spiking gaze-direction ring (STS eye-gaze / biological-motion code).
# ---------------------------------------------------------------------------------------------------------------
def build_gaze_ring_bridge(seed, n_dir=48):
    """A single region of N_DIR Izhikevich neurons = the STS gaze-direction population. Preferred angles are laid
    out evenly on the circle by neuron index; we drive each with cos-tuned current about the partner's gaze angle
    and read its firing. No internal pathways -- the ring is a pure direction-encoder; the tuning lives in the
    DRIVE (the partner's eyes fall on the retina) and in the downstream STS-TPJ synapses (W_obj). NO `sim/` edit."""
    from sim.config import CoreSimConfig, VisualizationConfig, RuntimeState, GPUConfig
    from sim.bridge import SimulationBridge
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel
    cfg = CoreSimConfig(); cfg.num_neurons = 0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name; cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.dt_ms = 1.0; cfg.seed = int(seed); cfg.enable_brain_region_framework = True; cfg.ou_std_current_pA = 0.0
    for flag in ("enable_short_term_plasticity", "enable_hebbian_learning", "enable_homeostasis",
                 "enable_structural_plasticity", "enable_reward_modulation", "enable_stdp",
                 "enable_input_divisive_norm"):
        setattr(cfg, flag, False)
    # A functionally-INERT self-pathway (weight_mean=0.0) so the bridge builds real synapses and the neurons
    # integrate external current. Without any synapse the region hits a no-synapse init path and the membrane
    # never responds to the drive (v stuck at rest); the zero weight means the ring stays a pure direction-encoder.
    cfg.brain_regions = [BrainRegion(name="ring", n_neurons=int(n_dir), exc_fraction=1.0, internal_density=0.05,
                                     exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0,
                                     plastic_internal=False)]
    cfg.region_pathways = [RegionPathway(from_region="ring", to_region="ring", density=0.05,
                                         weight_mean=0.0, weight_jitter=0.0, plastic=False)]
    sb = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=RuntimeState(),
                          gpu_config=GPUConfig())
    sb.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    sb._initialize_simulation_data(called_from_playback_init=False)
    ridx = np.asarray(list(sb.region_manager.indices("ring")), dtype=int)
    pref = np.linspace(0.0, 2.0 * np.pi, num=int(n_dir), endpoint=False)   # preferred direction per ring neuron
    return sb, ridx, pref


def gaze_ring_rates(sb, ridx, pref, gaze_angle, gain=900.0, sharpness=2.0, settle=25):
    """Drive the ring with cos-tuned current about `gaze_angle` and return per-neuron spike rates (the partner's
    attention-direction population code, read from SPIKES)."""
    from sim.backend import to_host, from_host
    if getattr(sb, "cp_izh_c_reset", None) is not None:
        sb.cp_membrane_potential_v[:] = sb.cp_izh_c_reset
    else:
        sb.cp_membrane_potential_v[:] = -65.0
    sb.cp_recovery_variable_u[:] = 0.0
    if getattr(sb, "cp_firing_states", None) is not None:
        sb.cp_firing_states[:] = False
    # cos tuning, rectified and sharpened -> a bump of drive centred on the gaze direction
    tuning = np.maximum(np.cos(pref - gaze_angle), 0.0) ** sharpness
    cur = np.zeros(sb.core_config.num_neurons, dtype=np.float64)
    cur[ridx] = gain * tuning
    cur_dev = from_host(cur)
    rates = np.zeros(len(ridx), dtype=np.float64)
    for _ in range(settle):
        sb.cp_external_input_current[:] = cur_dev
        sb._run_one_simulation_step()
        fir = np.asarray(to_host(sb.cp_firing_states)).astype(float)
        rates += fir[ridx]
    sb.cp_external_input_current[:] = 0.0
    return rates / float(settle)


# ---------------------------------------------------------------------------------------------------------------
# The task: a partner attends one of K objects; only its gaze DIRECTION (+ noise) is observable.
# ---------------------------------------------------------------------------------------------------------------
def make_layout(rng, K, min_sep):
    """Place K objects at random angles on the circle with a minimum angular separation (so the target is
    resolvable). Returns object angles phi[K] indexed by OBJECT IDENTITY. The angle->identity assignment is
    RANDOMLY PERMUTED, so an object's INDEX carries NO angular information (a layout-blind angular decode is at
    chance) -- the identity of the attended object exists only as the conjunction of gaze WITH this layout."""
    for _ in range(200):
        ang = np.sort(rng.uniform(0.0, 2.0 * np.pi, size=K))
        d = np.diff(np.concatenate([ang, [ang[0] + 2.0 * np.pi]]))
        if d.min() >= min_sep:
            return ang[rng.permutation(K)]        # decorrelate object index from angular rank
    return rng.uniform(0.0, 2.0 * np.pi, size=K)   # fallback (rare)


def sts_scores(rates, phi, sharpness=2.0):
    """OTHER-ATTENTION-SCHEMA (STS-TPJ object cells): s[k] = sum_i W_obj[k,i] * rates[i], with the direction-tuned
    synapse W_obj[k,i] = relu(cos(theta_i - phi_k))**sharpness onto object k's CURRENT angular position phi_k.
    A dendritic sum of gaze-ring SPIKES -> high for the object along the gaze direction. n_dir inferred from rates."""
    n_dir = len(rates)
    theta = np.linspace(0.0, 2.0 * np.pi, num=n_dir, endpoint=False)
    W = np.maximum(np.cos(theta[None, :] - phi[:, None]), 0.0) ** sharpness   # [K, n_dir] direction-tuned synapses
    return W @ rates


def run_seed(seed, n_trials=60, K=6, n_dir=48, gaze_noise=0.10, min_sep_frac=0.55,
             gain=1400.0, ring_settle=25, fs_inh=9.0, fs_settle=25, input_gain=1200.0):
    rng = np.random.RandomState(seed)
    min_sep = (2.0 * np.pi / K) * min_sep_frac
    chance = 1.0 / K

    sb_ring, ridx, pref = build_gaze_ring_bridge(seed=seed, n_dir=n_dir)
    sb_spot = build_fswta_score_bridge(seed=seed, K=K, fs_to_exc=fs_inh)   # reused spiking one-of-K spotlight

    # pre-draw a per-trial random permutation for the SCRAMBLE control (partner gaze shuffled across trials)
    trials = []
    for _ in range(n_trials):
        phi = make_layout(rng, K, min_sep)
        t_star = int(rng.randint(K))
        gaze = float(phi[t_star] + rng.normal(0.0, gaze_noise))       # partner gaze = target dir + bio-motion noise
        trials.append({"phi": phi, "t_star": t_star, "gaze": gaze})
    perm = rng.permutation(n_trials)                                   # scramble: reuse OTHER trials' gaze angles

    def spotlight_winner(scores):
        _, acc = fswta_drive(sb_spot, K, scores, input_gain=input_gain, settle=fs_settle)
        return int(np.argmax(acc)) if acc.max() > 0 else -1

    ok = ok_les = ok_scr = ok_blind = 0
    for n, tr in enumerate(trials):
        phi, t_star, gaze = tr["phi"], tr["t_star"], tr["gaze"]

        # (a) INTACT joint attention
        rates = gaze_ring_rates(sb_ring, ridx, pref, gaze, gain=gain, settle=ring_settle)
        s = sts_scores(rates, phi)
        ok += int(spotlight_winner(s) == t_star)

        # (b) LESION the other-attention-schema OUTPUT (STS-TPJ read severed) -> uniform drive -> chance
        s_les = np.zeros(K)
        ok_les += int(spotlight_winner(s_les) == t_star)

        # (c) SCRAMBLE the partner gaze (use another trial's gaze angle), score vs THIS trial's true t_star
        gaze_scr = trials[perm[n]]["gaze"]
        rates_scr = gaze_ring_rates(sb_ring, ridx, pref, gaze_scr, gain=gain, settle=ring_settle)
        s_scr = sts_scores(rates_scr, phi)
        ok_scr += int(spotlight_winner(s_scr) == t_star)

        # (d) NOT-A-COPY: layout-blind decode -- pick the object index by a FIXED angular bin of the gaze
        #     (ignores this trial's phi layout). Correct only by chance since phi is randomized per trial.
        blind_idx = int(np.floor((gaze % (2.0 * np.pi)) / (2.0 * np.pi / K)))
        ok_blind += int(blind_idx == t_star)

    return {
        "seed": int(seed), "K": int(K), "n_trials": int(n_trials), "chance": round(chance, 3),
        "align_acc": round(ok / n_trials, 3),
        "align_acc_lesion": round(ok_les / n_trials, 3),
        "align_acc_scramble": round(ok_scr / n_trials, 3),
        "align_acc_blind": round(ok_blind / n_trials, 3),
    }


def _verdict(rows):
    K = rows[0]["K"]; chance = 1.0 / K; band = chance + 0.10
    agg = {k: float(np.mean([r[k] for r in rows])) for k in
           ("align_acc", "align_acc_lesion", "align_acc_scramble", "align_acc_blind")}
    a = agg["align_acc"] >= 0.85
    b = agg["align_acc_lesion"] <= band
    c = agg["align_acc_scramble"] <= band
    d = agg["align_acc_blind"] <= band
    go = a and b and c and d
    return go, agg, band, {"a_align": a, "b_lesion": b, "c_scramble": c, "d_notacopy": d}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--n-trials", type=int, default=60)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--n-dir", type=int, default=48)
    ap.add_argument("--smoke", action="store_true", help="1-seed reduced trials (fast numpy foreground smoke)")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    seeds = a.seeds if a.seeds is not None else [a.seed]
    # smoke = 1 seed, numpy, foreground. 60 trials (not fewer): the scramble/lesion anti-cheats compare against
    # chance (1/K), which is dominated by sampling noise below ~50 trials -- a valid single-seed VERDICT needs n.
    n_trials = 60 if a.smoke else a.n_trials
    if a.smoke:
        seeds = [seeds[0]]

    print(f"[JOINT ATTENTION] STS-TPJ other-attention-schema -> spiking one-of-K spotlight (reused FS-WTA organ) | "
          f"K={a.K} n_dir={a.n_dir} n_trials={n_trials} seeds={seeds}", flush=True)
    t0 = time.time()
    rows = []
    for s in seeds:
        r = run_seed(s, n_trials=n_trials, K=a.K, n_dir=a.n_dir)
        rows.append(r)
        print(f"  seed {s}: align={r['align_acc']:.3f}  lesion={r['align_acc_lesion']:.3f}  "
              f"scramble={r['align_acc_scramble']:.3f}  blind={r['align_acc_blind']:.3f}  (chance={r['chance']:.3f})",
              flush=True)

    go, agg, band, checks = _verdict(rows)
    print(f"  ---- aggregate ({len(rows)} seed{'s' if len(rows) != 1 else ''}) ----", flush=True)
    print(f"  align_acc         = {agg['align_acc']:.3f}  (GO >= 0.85)", flush=True)
    print(f"  align_acc_lesion  = {agg['align_acc_lesion']:.3f}  (anti-cheat <= {band:.3f})", flush=True)
    print(f"  align_acc_scramble= {agg['align_acc_scramble']:.3f}  (anti-cheat <= {band:.3f})", flush=True)
    print(f"  align_acc_blind   = {agg['align_acc_blind']:.3f}  (not-a-copy  <= {band:.3f})", flush=True)
    tag = "GO" if go else "PARTIAL/NO-GO"
    if a.smoke:
        tag = ("SMOKE-GO" if go else "SMOKE-PARTIAL") + " (1-seed numpy; real gate is 6-seed)"
    print(f"  VERDICT: {tag}  checks={checks}", flush=True)
    print(f"  ({time.time() - t0:.1f}s)  BIOLOGY: Kandel Ch.62 Fig.62-4 STS-TPJ gaze/biological-motion; "
          f"binding research/biology/joint-attention-gaze-following.md. NO sim/ edit.", flush=True)

    if a.json:
        out = {"rows": rows, "aggregate": agg, "band": band, "go": bool(go), "checks": checks,
               "K": a.K, "n_dir": a.n_dir, "n_trials": n_trials, "seeds": seeds}
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"  wrote {a.json}", flush=True)


if __name__ == "__main__":
    main()
