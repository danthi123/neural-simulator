"""_fork_pcs_emergence_derisk — the AGI-fork FIRST-MOVE decisive battery (design section c+d+g#4).

Live ONE predictive-continual substrate (sim.pcs_substrate) online in the grounded egocentric-crop
world (research.runners.fork_pcs_world), driven by ITS OWN curiosity policy (drive-reduction +
learning-progress). Then, with weights FROZEN, ask whether faculties the strict path hand-builds as
separate organs have EMERGED — each read off the SAME population h_t, each shown LOAD-BEARING ON
BEHAVIOR (not merely decodable), against three anti-cheat floors.

FACULTIES (each needs a PRESENCE decode AND a behavioral-dependency lesion)
  place        ridge-decode abs (x,y) from h_t         | lesion place-units -> foraging/approach degrades ≫ random
  object       RSA of h_t vs the 4 object types         | lesion object-units -> object-contingent pred-error rises ≫ random
  permanence   decode food (x,y) when OFF the crop      | lesion -> off-view approach-to-food degrades ≫ random
  value        decode discounted future drive-reduction | lesion value-units -> reward-rate / approach degrades ≫ random

FLOORS (a presence number must beat ALL three)
  untrained-core   fresh random substrate replayed on the SAME input sequence (kills "the recurrence trivially echoes it")
  raw-V1-of-crop   decode from the raw Gabor-V1 of the current crop (kills "it's trivially in the frame")
  temporal-shuffle h rows permuted vs labels (kills "any decoder fits")

STRUCTURAL CONTROLS
  recurrent-core lesion   zero W_h -> every integration faculty must collapse to ~floor (proves they live in the recurrence)
  curiosity vs random     distinct-cell coverage of the curiosity policy vs a uniform-random policy (invariant #5)
  rate maps               per-unit mean activation binned by (x,y) (the visually-convincing bonus)

⭐ PRE-REGISTERED GO GATE (design section d — fixed BEFORE the multi-seed run) ⭐
  EMERGENCE GO (rate arm) iff, on >= 5/6 seeds:
    (1) >= 3 of {place, object, permanence, value} clear their PRESENCE bar
        (place R^2 >= 0.6 ; object RSA rho >= 0.4 ; permanence R^2 >= 0.4 ; value R^2 >= 0.4)
    (2) each of those cleared faculties BEATS ALL THREE FLOORS (by >= 0.05), AND
    (3) each of those cleared faculties PASSES its behavioral-dependency lesion
        (faculty-unit lesion degrades its behavioral metric by >= 1.5x the equal-size RANDOM-unit lesion),
  AND the recurrent-core lesion collapses all faculties to ~floor (mean cleared-faculty presence drops >= 50%),
  AND curiosity coverage >= 1.5x random coverage on >= 5/6 seeds.
  The single hardest-to-fake line: abs-position R^2 high while raw-V1-of-crop sits at chance
  (position was INTEGRATED, not seen) AND lesioning it demonstrably breaks navigation.

  FORK-THESIS GO (Day-5, separate): rerun --units spike; GO if rate reaches >= N load-bearing faculties
  in a fraction of the spike arm's GPU-hours (or a breadth spike can't reach in-window). A spike-matches-rate
  result is the bankable negative that all-spiking was NOT the bottleneck (a valid fork outcome).

HONESTY: every read-out is a FUNCTIONAL instrument reading; nothing here asserts felt/phenomenal experience.

Run:
  # runner self-test (numpy, tiny, 1 seed) — confirms the battery runs end-to-end + emits JSON
  SIM_BACKEND=numpy python -m research.runners._fork_pcs_emergence_derisk --smoke --out /tmp/fork_smoke.json
  # full 6-seed rate arm (GPU — queue it; 0 agent tokens):
  SIM_BACKEND=cupy python -m research.runners._fork_pcs_emergence_derisk \
      --seeds 42 43 44 100 101 102 --units rate --out research/findings/raw/_fork_pcs_emergence_rate_6seed.json
  # DECISIVE behavioral-load-bearing arm (5th move): shaping-FREE homing via the reverse curriculum + the
  # aux-loc/value heads + the held-out novel-start SHORTCUT probe (NO --nav-shaping — the leaked cue is gone):
  SIM_BACKEND=cupy python -m research.runners._fork_pcs_emergence_derisk \
      --seeds 42 43 44 100 101 102 --units rate --nav-required --nav-curriculum \
      --value-weight 1.0 --aux-loc-weight 1.0 --lesion-mode both \
      --out research/findings/raw/_fork_pcs_emergence_navcurriculum_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.pcs_substrate import PredictiveContinualSubstrate, PCSConfig
from research.runners.fork_pcs_world import (
    WorldConfig, ForkPCSWorld, N_ACTIONS, K_OBJECTS, MOVES, _ridge_r2,
)
# the anti-hollow attribution: (treatment - control)/treatment — how much of a lesion's behavioral
# degradation is due to the FACULTY units vs an equal RANDOM-unit lesion (measuring both arms is not
# the same as attributing the difference; gap#5 banked both arms one key apart for weeks).
from tools.lab import attributable_to

# ── pre-registered presence bars + gate thresholds ──────────────────────────
PRESENCE_BAR = {"place": 0.60, "object": 0.40, "permanence": 0.40, "value": 0.40}
FLOOR_MARGIN = 0.05        # a presence number must beat each floor by this
BEHAV_LESION_RATIO = 1.5   # faculty-lesion must degrade its metric >= this x the random-unit lesion
MIN_FACULTIES = 3          # >= 3 of 4 must clear
CORE_LESION_COLLAPSE = 0.50  # core-lesion must drop mean cleared-faculty presence by >= this fraction
CURIOSITY_RATIO = 1.5      # curiosity coverage / random coverage
SEEDS_REQUIRED_FRAC = 5.0 / 6.0


# ─────────────────────────────────────────────────────────────────────────────
# small decode helpers
# ─────────────────────────────────────────────────────────────────────────────
def _rank(a):
    order = np.argsort(a, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def _spearman(x, y):
    rx, ry = _rank(x), _rank(y)
    rx = rx - rx.mean(); ry = ry - ry.mean()
    denom = (np.sqrt((rx * rx).sum()) * np.sqrt((ry * ry).sum())) + 1e-12
    return float((rx * ry).sum() / denom)


def _rsa(H, labels, max_n=600):
    """RSA: Spearman corr between h-similarity and same-object-type indicator (upper triangle)."""
    if len(H) < 8 or len(np.unique(labels)) < 2:
        return float("nan")
    if len(H) > max_n:
        idx = np.random.default_rng(0).choice(len(H), max_n, replace=False)
        H, labels = H[idx], labels[idx]
    Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)
    S = Hn @ Hn.T
    M = (labels[:, None] == labels[None, :]).astype(np.float64)
    iu = np.triu_indices(len(H), 1)
    return _spearman(S[iu], M[iu])


def _ridge_weights(X, Y, lam=10.0):
    """DECODING-importance: fit ridge (standardized) and return per-input-unit importance = sum|weight|
    over targets. This selects the most DECODABLE units (the current battery's only lesion-selection rule)."""
    mu = X.mean(0, keepdims=True); sd = np.maximum(X.std(0, keepdims=True), 1e-2)
    Xs = (X - mu) / sd
    d = Xs.shape[1]
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(d, dtype=np.float64), Xs.T @ Y)
    return np.abs(W).sum(axis=1)   # (d,) importance per unit


def _host(a):
    """Marshal a backend (cupy/numpy) array to a host float64 ndarray."""
    return np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.float64)


def _behavioral_saliency(sub, H_probe, metric):
    """BEHAVIORAL-importance ranking of hidden units — the causal read-head SALIENCY, per faculty metric.

    WHY THIS METHOD (the fix's core choice). Our decoding-importance ranking (_ridge_weights) selects the
    most DECODABLE units. The literature (Schøyen 2023; Schaeffer 2022 "No Free Lunch") shows decodability
    and causal LOAD-BEARING can DISSOCIATE: high-decodability spatial units were found causally DISPENSABLE
    while a DIFFERENT population carried path integration. So a lesion aimed by decodability can miss the
    behaviorally load-bearing units. This ranking aims instead at the units the BEHAVIOR-PRODUCING read-head
    actually reads, so the two arms can be compared and the dissociation made visible.

    A unit is behaviorally load-bearing iff the head that PRODUCES the faculty's behavioral metric reads it
    strongly AND it varies (a strongly-read but constant unit cannot change behavior). So rank unit u by
        saliency(u) = |head weight on u|  *  std(h_u over the frozen probe rollout).
      * approach_in / approach_off / reward_rate  -> the ACTION/VALUE channel: sum_a |W_pi[a,u]| (policy
        head over all actions) + |w_v[u]| when the value head is present (actor-critic). These metrics are
        produced by action selection and, with a value head, the value estimate — the units those heads read.
      * pred_err (object)                         -> the JEPA PREDICTOR head sum_e |W_pred[e,u]|, the head
        whose loss IS the object faculty's metric.
    This is the standard connection-strength x activity saliency. It is CHEAP (reads the trained read-head
    weights + one std over the ALREADY-collected probe H — no extra rollouts), and FAITHFUL to the causal
    path (it ranks by the head that generates the metric, not by decodability). It is a first-order
    (linear-readout) proxy for SELECTING the candidate set; the behavioral LESION that follows is the actual
    causal test, exactly as ridge-importance only SELECTS the decoding set before its lesion.
    """
    std_h = H_probe.std(axis=0)                                   # (H,) activation variability over the probe
    if metric == "pred_err":
        head = np.abs(_host(sub.P["W_pred"])).sum(axis=0)         # (H,) predictor head reads h -> next latent
    else:
        head = np.abs(_host(sub.W_pi)).sum(axis=0)               # (H,) policy head, summed over actions
        if "w_v" in sub.P:
            head = head + np.abs(_host(sub.P["w_v"]))            # + value head (actor-critic) when present
    return head * std_h                                          # (H,) behavioral importance per unit


def _jaccard(mask_a, mask_b):
    """Jaccard overlap of two boolean top-k unit masks (|A∩B| / |A∪B|). The Schøyen dissociation measure:
    low overlap between the decoding-selected and behavioral-selected sets + only-behavioral-lesion-degrades
    means decodability is misleading (the decodable units are not the load-bearing ones)."""
    inter = int(np.logical_and(mask_a, mask_b).sum())
    union = int(np.logical_or(mask_a, mask_b).sum())
    return (inter / union) if union > 0 else float("nan")


def _r2_with_floors(H, H_un, RAW, labels, seed):
    """Return dict: trained R^2 + the three floors (untrained / raw-V1 / temporal-shuffle)."""
    n = len(labels)
    perm = np.random.default_rng(seed + 5).permutation(n)
    cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    r2 = _ridge_r2(H[tr], labels[tr], H[te], labels[te])
    r2_un = _ridge_r2(H_un[tr], labels[tr], H_un[te], labels[te])
    r2_raw = _ridge_r2(RAW[tr], labels[tr], RAW[te], labels[te])
    # temporal-shuffle: break the h<->label correspondence
    sh = np.random.default_rng(seed + 9).permutation(n)
    r2_sh = _ridge_r2(H[tr], labels[sh][tr], H[te], labels[sh][te])
    return {"r2": r2, "floor_untrained": r2_un, "floor_rawv1": r2_raw, "floor_shuffle": r2_sh}


def _beats_floors(d, bar):
    return (d["r2"] >= bar
            and d["r2"] >= d["floor_untrained"] + FLOOR_MARGIN
            and d["r2"] >= d["floor_rawv1"] + FLOOR_MARGIN
            and d["r2"] >= d["floor_shuffle"] + FLOOR_MARGIN)


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ─────────────────────────────────────────────────────────────────────────────
# rollout: run world+substrate online, collect traces
# ─────────────────────────────────────────────────────────────────────────────
def rollout(world, sub, n_steps, train, explore_eps, collect=False, lesion_mask=None,
            a_prev_start=-1, log_loss=False, aux_loc=False):
    """Drive the substrate against the world. Returns traces (+ leaves world/sub advanced).

    train=False freezes the predictive update (probe/behavioral rollouts). lesion_mask sets a
    hidden-unit lesion for the whole rollout. collect=True records per-step h/labels/rawv1.
    log_loss=True records a downsampled training-loss curve (train-stability visibility).
    aux_loc=True supplies the true allocentric position (normalized by grid_size) as the per-step
    stop-grad target of the substrate's aux self-localization loss (5th move) — a fork-accepted mild
    host scaffold, only meaningful when the substrate was built with aux_loc_weight>0. Passed only on
    the online TRAINING rollout; OFF -> observe() gets no target and the tape is byte-identical.
    """
    gsz = float(world.cfg.grid_size)
    if train:
        sub.unfreeze()
    else:
        sub.freeze()
    sub.set_lesion_mask(lesion_mask)
    loss_curve = []
    log_every = max(1, n_steps // 40)
    H, POS, FOOD, FIC, OBJ, RAW, REW, INSEQ = [], [], [], [], [], [], [], []
    approach_in, approach_off = [], []
    a_prev = a_prev_start
    total_reward = 0.0
    curriculum = train and getattr(world.cfg, "nav_curriculum", False)
    for t in range(n_steps):
        # REVERSE-CURRICULUM (5th move): during ONLINE TRAINING only, advance the post-eat respawn distance
        # from NEAR the larder to the full nav_dmin over the schedule (fraction of training). Eval rollouts
        # (train=False) never enter here, so the frozen policy is always probed at full difficulty (the caller
        # pins progress to 1.0). Inert unless the world has nav_curriculum ON.
        if curriculum:
            world.set_curriculum_progress(t / max(1, n_steps))
        # capture ALL pre-step sensory info aligned to the observation the substrate acts on
        d = world.drive_afferent()
        v1 = world.crop_v1feat()
        v1_host = np.asarray(v1.get() if hasattr(v1, "get") else v1, dtype=np.float32)
        pos_before = world.agent
        food_before = world.food
        fic = world.food_in_crop
        obj_in = world.objects_in_crop()
        # AUX-LOC (5th move): the true allocentric position when the substrate makes THIS observation is
        # pos_before (captured pre-step), normalized to ~[0,1) by the grid size -> the stop-grad target of the
        # supervised self-localization loss. Supplied only on the training rollout with aux_loc=True.
        pos_target = (np.asarray(pos_before, dtype=np.float32) / gsz) if aux_loc else None
        h = sub.observe(v1, a_prev, d, pos_target=pos_target)
        a = sub.act(h, explore_eps=explore_eps)
        r, info = world.step(a)
        sub.learn(r)
        total_reward += r
        # behavioral approach metric (measurement only — host ground truth, NOT a reward term)
        if food_before is not None:
            delta = _manhattan(pos_before, food_before) - _manhattan(world.agent, food_before)
            (approach_in if fic else approach_off).append(1.0 if delta > 0 else 0.0)
        if collect:
            H.append(np.asarray(h.get() if hasattr(h, "get") else h, dtype=np.float32))
            POS.append(np.asarray(pos_before, dtype=np.float32))
            FOOD.append(np.asarray(food_before if food_before is not None else (-1, -1), dtype=np.float32))
            FIC.append(1.0 if fic else 0.0)
            OBJ.append(obj_in)
            RAW.append(v1_host)                       # raw V1 of the CURRENT crop (the raw-V1 floor feature)
            INSEQ.append((v1_host, a_prev, np.asarray(d, np.float32)))   # exact input for the untrained replay
        REW.append(r)
        if log_loss and (t % log_every == 0) and sub.last_pred_loss is not None:
            loss_curve.append((t, round(float(sub.last_pred_loss), 4)))
        a_prev = a
    sub.set_lesion_mask(None)
    out = {
        "reward_rate": total_reward / max(1, n_steps),
        "approach_in": float(np.mean(approach_in)) if approach_in else float("nan"),
        "approach_off": float(np.mean(approach_off)) if approach_off else float("nan"),
        "eats": world.n_eats, "a_prev_end": a_prev, "loss_curve": loss_curve,
    }
    if collect:
        out.update({"H": np.asarray(H), "POS": np.asarray(POS), "FOOD": np.asarray(FOOD),
                    "FIC": np.asarray(FIC), "OBJ": OBJ, "RAW": np.asarray(RAW), "REW": np.asarray(REW),
                    "INPUT_SEQ": INSEQ})
    return out


def replay_untrained(world_cfg, seed, units, encoder, feat_dim, n_latent, n_hidden, input_seq):
    """Replay a fixed input sequence through a FRESH UNTRAINED core (the untrained floor)."""
    sub = PredictiveContinualSubstrate(PCSConfig(
        n_hidden=n_hidden, feat_dim=feat_dim, n_latent=n_latent, n_actions=N_ACTIONS,
        n_drive=4, units=units, encoder=encoder, seed=seed + 777))
    sub.freeze()
    H = []
    for (v1, ap, d) in input_seq:
        h = sub.observe(v1, ap, d)
        H.append(np.asarray(h.get() if hasattr(h, "get") else h, dtype=np.float32))
    return np.asarray(H)


# ─────────────────────────────────────────────────────────────────────────────
# per-seed battery
# ─────────────────────────────────────────────────────────────────────────────
def run_seed(seed, units="rate", encoder="learned_ema", n_hidden=512, n_latent=64,
             n_train=200_000, n_probe=8000, n_behav=4000, n_cov=8000, gamma=0.95,
             lesion_frac=0.10, n_random_lesions=3, consolidation=False,
             grad_clip=1.0, grad_skip_factor=8.0, ema_momentum=0.9999, ema_warmup=0,
             pred_horizon=1, nav_required=False, nav_dmin=6, value_weight=0.0, sr_weight=0.0,
             aux_loc_weight=0.0, nav_shaping=0.0, nav_curriculum=False, nav_curriculum_frac=0.5,
             nav_curriculum_dmin_start=1, si=False, lesion_mode="decoding", verbose=True):
    t0 = time.time()
    wcfg = WorldConfig(seed=seed, nav_required=nav_required, nav_dmin=nav_dmin, nav_shaping=nav_shaping,
                       nav_curriculum=nav_curriculum, nav_curriculum_frac=nav_curriculum_frac,
                       nav_curriculum_dmin_start=nav_curriculum_dmin_start)
    world = ForkPCSWorld(wcfg)
    scfg = PCSConfig(n_hidden=n_hidden, feat_dim=wcfg.n_v1, n_latent=n_latent, n_actions=N_ACTIONS,
                     n_drive=4, tbptt_T=18, units=units, encoder=encoder, seed=seed,
                     consolidation=consolidation, grad_clip=grad_clip, grad_skip_factor=grad_skip_factor,
                     ema_rate=ema_momentum, ema_warmup_updates=ema_warmup, pred_horizon=pred_horizon,
                     value_weight=value_weight, sr_weight=sr_weight, aux_loc_weight=aux_loc_weight, si=si)
    sub = PredictiveContinualSubstrate(scfg)

    # ---- 1. TRAIN online with the curiosity policy (small explore for early coverage) ----
    # aux_loc>0 -> supply the true per-step position target to the substrate's self-localization loss (5th move).
    train_out = rollout(world, sub, n_train, train=True, explore_eps=0.2, log_loss=True,
                        aux_loc=(aux_loc_weight > 0))
    # Curriculum: pin progress to 1.0 so EVERY subsequent frozen eval (probe/behav/coverage/shortcut) runs at
    # FULL difficulty (far respawns). Capture the set of TRAINING respawn start cells NOW, before
    # _core_lesion_presence resets the world to a different layout (which would clear it). The shortcut probe
    # rebuilds its own world from wcfg, so the fixed larder is recovered there (same seed layout).
    if nav_curriculum:
        world.set_curriculum_progress(1.0)
    train_respawn_cells = set(world._respawn_cells)

    # ---- 2. PROBE rollout (frozen) collecting h/labels/rawv1 + the input seq. Higher explore_eps here
    #         gives the FROZEN core coverage of the whole grid so the decode is measured over varied
    #         positions (exploration does not affect the frozen weights, only the probe trajectory). ----
    pr = rollout(world, sub, n_probe, train=False, explore_eps=0.4, collect=True)
    H, POS, FOOD, FIC, OBJ, RAW = pr["H"], pr["POS"], pr["FOOD"], pr["FIC"], pr["OBJ"], pr["RAW"]
    input_seq = pr["INPUT_SEQ"]      # exact (v1_host, a_prev, d) — replayed through the untrained core

    # place / permanence / value labels
    place_lab = POS
    # value: discounted future reward (Monte-Carlo return) from the probe reward trace
    REW = pr["REW"]
    G = np.zeros(len(REW), dtype=np.float32)
    acc = 0.0
    for i in range(len(REW) - 1, -1, -1):
        acc = REW[i] + gamma * acc
        G[i] = acc
    value_lab = G[:, None]
    # permanence: steps where food is OFF the crop and known (food != -1)
    off = (FIC < 0.5) & (FOOD[:, 0] >= 0)
    # object: steps where exactly one object type is in the crop
    obj_single_idx = [i for i, o in enumerate(OBJ) if len(o) == 1]
    obj_lab = np.asarray([OBJ[i][0] for i in obj_single_idx], dtype=np.int64) if obj_single_idx else np.zeros(0, np.int64)

    # untrained-core + raw-V1 traces aligned to H
    H_un = replay_untrained(wcfg, seed, units, encoder, wcfg.n_v1, n_latent, n_hidden, input_seq)

    # ---- 3. PRESENCE decodes vs 3 floors ----
    presence = {}
    presence["place"] = _r2_with_floors(H, H_un, RAW, place_lab, seed)
    presence["value"] = _r2_with_floors(H, H_un, RAW, value_lab, seed)
    if off.sum() >= 20:
        presence["permanence"] = _r2_with_floors(H[off], H_un[off], RAW[off], FOOD[off], seed)
    else:
        presence["permanence"] = {"r2": float("nan"), "floor_untrained": float("nan"),
                                  "floor_rawv1": float("nan"), "floor_shuffle": float("nan"),
                                  "note": f"only {int(off.sum())} off-view-food steps"}
    # object via RSA (+ floors)
    if len(obj_single_idx) >= 8:
        Ho = H[obj_single_idx]; Huno = H_un[obj_single_idx]; Rawo = RAW[obj_single_idx]
        presence["object"] = {"r2": _rsa(Ho, obj_lab), "floor_untrained": _rsa(Huno, obj_lab),
                              "floor_rawv1": _rsa(Rawo, obj_lab),
                              "floor_shuffle": _rsa(Ho, np.random.default_rng(seed + 9).permutation(obj_lab))}
    else:
        presence["object"] = {"r2": float("nan"), "floor_untrained": float("nan"),
                              "floor_rawv1": float("nan"), "floor_shuffle": float("nan"),
                              "note": f"only {len(obj_single_idx)} single-object steps"}

    cleared = {f: (not np.isnan(presence[f]["r2"]) and _beats_floors(presence[f], PRESENCE_BAR[f]))
               for f in PRESENCE_BAR}

    # ---- 4. BEHAVIORAL-DEPENDENCY LESIONS (the anti-hollow gate) ----
    # TWO instruments select the units to lesion, compared side-by-side when --lesion-mode both:
    #   DECODING importance  = ridge weights of the target on h (_ridge_weights) — the MOST DECODABLE units.
    #   BEHAVIORAL importance = policy/value(-or-predictor)-head SALIENCY x activation std
    #                           (_behavioral_saliency) — the units the BEHAVIOR-PRODUCING read-head reads.
    # Motivation (Schøyen 2023 / Schaeffer 2022): the decodable units and the causally load-bearing units can
    # DISSOCIATE, so a lesion aimed by decodability alone can mis-report a faculty's true behavioral
    # dependence. The Jaccard overlap of the two top-k sets (reported in `both`) IS the dissociation measure.
    # DEFAULT lesion_mode="decoding" runs ONLY the original decoding block below -> byte-identical output.
    k = max(8, int(lesion_frac * n_hidden))
    imp = {}
    imp["place"] = _ridge_weights(H, place_lab)
    imp["value"] = _ridge_weights(H, value_lab)
    imp["permanence"] = _ridge_weights(H[off], FOOD[off]) if off.sum() >= 20 else None
    imp["object"] = _ridge_weights(H[obj_single_idx],
                                   np.eye(K_OBJECTS, dtype=np.float32)[obj_lab]) if len(obj_single_idx) >= 8 else None

    def _mask_from_imp(importance):
        m = np.zeros(n_hidden, dtype=bool)
        m[np.argsort(importance)[::-1][:k]] = True
        return m

    # intact behavioral baseline
    intact = rollout(world, sub, n_behav, train=False, explore_eps=0.1)
    # random-unit lesion baseline (mean over draws) — SHARED by both lesion arms (the equal-size control)
    rng = np.random.default_rng(seed + 3)
    rand_metrics = {"reward_rate": [], "approach_in": [], "approach_off": [], "pred_err": []}
    eval_seq = _capture_eval_seq(world, sub, 400)
    for _ in range(n_random_lesions):
        m = np.zeros(n_hidden, dtype=bool); m[rng.choice(n_hidden, k, replace=False)] = True
        rl = rollout(world, sub, n_behav, train=False, explore_eps=0.1, lesion_mask=m)
        sub.set_lesion_mask(m)
        pe = sub.eval_predictive_loss(eval_seq, respect_lesion=True)
        sub.set_lesion_mask(None)
        rand_metrics["reward_rate"].append(rl["reward_rate"]); rand_metrics["approach_in"].append(rl["approach_in"])
        rand_metrics["approach_off"].append(rl["approach_off"]); rand_metrics["pred_err"].append(pe)
    rand = {kk: float(np.nanmean(vv)) for kk, vv in rand_metrics.items()}
    intact_pe = _pred_err(sub, eval_seq)

    metric_for = {"place": "approach_in", "permanence": "approach_off", "value": "reward_rate", "object": "pred_err"}

    def _lesion_dependency(f, mask, label):
        """Lesion `mask` (a top-k unit set) and measure the attributable_to degradation on faculty f's
        behavioral metric vs the SHARED random-unit lesion baseline. Instrument-agnostic (the decoding and
        behavioral arms call this with their own mask), so the two are measured identically."""
        met = metric_for[f]
        fl = rollout(world, sub, n_behav, train=False, explore_eps=0.1, lesion_mask=mask)
        sub.set_lesion_mask(mask); fpe = sub.eval_predictive_loss(eval_seq, respect_lesion=True); sub.set_lesion_mask(None)
        if met == "pred_err":
            # degradation = pred-error RISE; faculty must raise it >= ratio x the random rise
            fac_deg = fpe - intact_pe
            rnd_deg = rand["pred_err"] - intact_pe
        else:
            # degradation = metric DROP; faculty must drop it >= ratio x the random drop
            base = {"approach_in": intact["approach_in"], "approach_off": intact["approach_off"],
                    "reward_rate": intact["reward_rate"]}[met]
            fac_val = {"approach_in": fl["approach_in"], "approach_off": fl["approach_off"],
                       "reward_rate": fl["reward_rate"]}[met]
            fac_deg = base - fac_val
            rnd_deg = base - rand[met]
        # ATTRIBUTION (not just measurement): fraction of the degradation due to the FACULTY units, not
        # what an equal random-unit lesion does. frac >= (1 - 1/ratio) is equivalent to fac_deg >= ratio*rnd_deg.
        frac = attributable_to(f"{label} {f}", fac_deg, rnd_deg)
        load_bearing = (frac is not None) and (fac_deg > 0) and (frac >= (1.0 - 1.0 / BEHAV_LESION_RATIO))
        return {"metric": met, "faculty_degradation": _f(fac_deg), "random_degradation": _f(rnd_deg),
                "attributable_fraction": _f(frac), "load_bearing": bool(load_bearing)}

    # DECODING arm (the current instrument; UNCHANGED code path -> byte-identical when mode=decoding).
    behav_decoding = {}
    if lesion_mode in ("decoding", "both"):
        for f in PRESENCE_BAR:
            if imp[f] is None:
                behav_decoding[f] = {"note": "insufficient data for lesion", "load_bearing": False}
                continue
            behav_decoding[f] = _lesion_dependency(f, _mask_from_imp(imp[f]), "behav-lesion")

    # BEHAVIORAL arm (the new instrument): rank by causal read-head saliency, lesion its top-k, same metric.
    behav_behavioral = {}
    overlap = {}
    if lesion_mode in ("behavioral", "both"):
        for f in PRESENCE_BAR:
            # parity with the decoding arm's data-availability guard: if a faculty had no decoding importance
            # (insufficient single-object / off-view steps), its behavioral lesion is equally undefined.
            if imp[f] is None:
                behav_behavioral[f] = {"note": "insufficient data for lesion", "load_bearing": False}
                continue
            bmask = _mask_from_imp(_behavioral_saliency(sub, H, metric_for[f]))
            d = _lesion_dependency(f, bmask, "behav-lesion-BEHAV")
            d["n_behav_units"] = int(bmask.sum())
            behav_behavioral[f] = d
            if lesion_mode == "both":     # Jaccard overlap of decoding vs behavioral top-k (dissociation)
                overlap[f] = _f(_jaccard(_mask_from_imp(imp[f]), bmask))

    # The GATE-facing `behav` is the DECODING instrument (the pre-registered gate) in decoding/both modes;
    # in behavioral-only mode it is the behavioral instrument (the only arm run). Both/behavioral ADD the
    # behavioral results (and, in both, the Jaccard overlap) as separate keys below.
    behav = behav_behavioral if lesion_mode == "behavioral" else behav_decoding

    # ---- 5. CORE LESION: zero W_h -> integration faculties collapse ----
    core = _core_lesion_presence(sub, H_un, RAW, POS, value_lab, FOOD, off, obj_single_idx, obj_lab,
                                 world, wcfg, seed, units, encoder, n_hidden, n_latent, n_probe)

    # ---- 6. CURIOSITY vs RANDOM coverage (EXPLORATION EFFICIENCY, not saturation) ----
    # Measured over a SHORT budget where uniform-random does NOT saturate the grid, else "1.5x random"
    # is structurally impossible (random covers ~all cells given enough steps). Averaged over windows.
    cov_budget = max(40, 4 * wcfg.grid_size)
    cur_list = [_coverage(world, sub, cov_budget, curiosity=True, restart_seed=seed + 200 + i) for i in range(5)]
    rnd_list = [_coverage(world, sub, cov_budget, curiosity=False, restart_seed=seed + 200 + i) for i in range(5)]
    cov_cur = float(np.mean(cur_list)); cov_rnd = float(np.mean(rnd_list))
    coverage = {"curiosity_cells": cov_cur, "random_cells": cov_rnd, "budget": cov_budget,
                "ratio": cov_cur / max(1.0, cov_rnd), "pass": (cov_cur / max(1.0, cov_rnd)) >= CURIOSITY_RATIO}

    # ---- 7. RATE MAPS (per-unit mean activation binned by (x,y)); store a compact summary ----
    ratemap = _rate_map_summary(H, POS, wcfg.grid_size)

    # ---- 7b. SPATIAL-INFORMATION place-cell metric (FLOOR-INDEPENDENT; ADDITIVE diagnostic only) ----
    # Reported ALONGSIDE the linear-decode place metric (presence["place"]) so the two can be compared on
    # the aux-loc runs; the pre-registered GO gate is UNCHANGED. Computed on the SAME probe trajectory for
    # the TRAINED core (H) and the UNTRAINED reservoir (H_un — the inflated-floor control, replayed on the
    # identical input sequence so its rows align 1:1 with POS). The demonstration: the untrained reservoir
    # has HIGH linear-decode R^2 (presence["place"]["floor_untrained"]) but LOW SI / few significant place
    # cells, whereas a genuine emergent place code raises SI >> its shuffle null. This RE-MEASURES place
    # floor-independently without touching the gate.
    place_si_trained = _place_cell_metrics(H, POS, wcfg.grid_size, seed)
    place_si_untrained = _place_cell_metrics(H_un, POS, wcfg.grid_size, seed)

    # ---- 7c. OBJECT / PERMANENCE / VALUE floor-independent diagnostics (ADDITIVE; SAME SI formalism) ----
    # Computed for TRAINED (H) and UNTRAINED-reservoir-replay (H_un) on the identical aligned input sequence,
    # exactly as the place SI pair above, so the trained-vs-reservoir separation is visible per faculty. See
    # the banner above `_categorical_tuning_metrics` for why each is the floor-independent read for its
    # faculty. Guarded by the SAME data-availability checks as their presence[...] counterparts.
    if len(obj_single_idx) >= 8:
        object_si_trained = _object_selectivity_metrics(H[obj_single_idx], obj_lab, seed)
        object_si_untrained = _object_selectivity_metrics(H_un[obj_single_idx], obj_lab, seed)
    else:
        note = {"note": f"only {len(obj_single_idx)} single-object steps"}
        object_si_trained = dict(note); object_si_untrained = dict(note)
    if off.sum() >= 20:
        permanence_si_trained = _permanence_food_si_metrics(H[off], FOOD[off], wcfg.grid_size, seed)
        permanence_si_untrained = _permanence_food_si_metrics(H_un[off], FOOD[off], wcfg.grid_size, seed)
    else:
        note = {"note": f"only {int(off.sum())} off-view-food steps"}
        permanence_si_trained = dict(note); permanence_si_untrained = dict(note)
    value_si_trained = _value_tuning_metrics(H, value_lab, seed)
    value_si_untrained = _value_tuning_metrics(H_un, value_lab, seed)

    # ---- seed verdict ----
    n_cleared_lb = sum(1 for f in PRESENCE_BAR if cleared[f] and behav.get(f, {}).get("load_bearing", False))
    cleared_names = [f for f in PRESENCE_BAR if cleared[f]]
    cleared_pres = [presence[f]["r2"] for f in cleared_names if not np.isnan(presence[f]["r2"])]
    core_pres = [core.get(f, float("nan")) for f in cleared_names]
    core_collapses = (len(cleared_pres) > 0 and np.nanmean(core_pres) <= (1 - CORE_LESION_COLLAPSE) * np.nanmean(cleared_pres))
    seed_go = (n_cleared_lb >= MIN_FACULTIES) and core_collapses and coverage["pass"]

    result = {
        "seed": seed, "units": units, "encoder": encoder, "n_hidden": n_hidden, "n_train": n_train,
        "consolidation": consolidation, "n_replay_updates": int(sub.n_replay_updates),
        "grad_clip": grad_clip, "grad_skip_factor": grad_skip_factor, "pred_horizon": pred_horizon,
        "nav_required": nav_required, "nav_dmin": nav_dmin, "value_weight": value_weight,
        "sr_weight": sr_weight, "aux_loc_weight": aux_loc_weight,
        "nav_shaping": nav_shaping,
        "ema_momentum": ema_momentum, "ema_warmup": ema_warmup,
        "max_grad_norm": round(float(sub.max_grad_norm), 3), "n_grad_skipped": int(sub.n_skipped),
        "train_max_online_loss": round(float(max([v for _, v in (train_out.get("loss_curve") or [(0, 0.0)])])), 3),
        "presence": {f: {kk: _f(vv) for kk, vv in presence[f].items() if isinstance(vv, (int, float))}
                     | ({"note": presence[f]["note"]} if "note" in presence[f] else {}) for f in presence},
        "cleared_presence_and_floors": cleared,
        "lesion_mode": lesion_mode,
        "behavioral_dependency": behav,
        "n_faculties_load_bearing": n_cleared_lb,
        "core_lesion_presence": {k2: _f(v2) for k2, v2 in core.items()},
        "core_lesion_collapses_all": bool(core_collapses),
        "coverage": {k2: _f(v2) if isinstance(v2, (int, float)) else v2 for k2, v2 in coverage.items()},
        "rate_map_summary": ratemap,
        # FLOOR-INDEPENDENT place metric (additive diagnostic) — trained core vs the untrained reservoir,
        # side-by-side with the linear-decode place metric in presence["place"] (r2 vs floor_untrained).
        "place_cell_si": place_si_trained,
        "place_cell_si_untrained": place_si_untrained,
        # FLOOR-INDEPENDENT object/permanence/value diagnostics (additive) — trained core vs the untrained
        # reservoir, side-by-side with their linear-decode/RSA presence[...] counterparts (unchanged above).
        "object_selectivity_si": object_si_trained,
        "object_selectivity_si_untrained": object_si_untrained,
        "permanence_food_si": permanence_si_trained,
        "permanence_food_si_untrained": permanence_si_untrained,
        "value_tuning_si": value_si_trained,
        "value_tuning_si_untrained": value_si_untrained,
        "train_loss_curve": train_out.get("loss_curve", []),
        "intact_behavior": {k2: _f(v2) for k2, v2 in intact.items() if isinstance(v2, (int, float))},
        "SEED_GO": bool(seed_go),
        "elapsed_s": round(time.time() - t0, 1),
    }
    # CURRICULUM echo (only when ON -> non-curriculum output stays byte-identical, no new keys).
    if nav_curriculum:
        result["nav_curriculum"] = True
        result["nav_curriculum_frac"] = nav_curriculum_frac
        result["nav_curriculum_dmin_start"] = nav_curriculum_dmin_start
    # NOVEL-START SHORTCUT PROBE — only meaningful with a fixed larder (nav_required). Runs LAST on a fresh
    # world with the FROZEN trained core, so it never perturbs any measurement above. Uses the same place-unit
    # importance (imp["place"]) that selects the main place lesion, over the TRAINING respawn cells captured
    # pre-core-lesion. Absent from the output entirely on the non-nav path (byte-identical).
    if nav_required:
        result["shortcut_probe"] = _shortcut_probe(
            sub, wcfg, seed, imp["place"], n_hidden, lesion_frac=lesion_frac,
            n_random_lesions=n_random_lesions, exclude_cells=train_respawn_cells, verbose=verbose)
    # BEHAVIORAL-arm results (only when run) — reported ALONGSIDE the decoding arm so the Schøyen
    # dissociation is visible. In `both` mode, `behavioral_dependency` above is the DECODING arm (the
    # pre-registered gate, byte-identical), and these keys add the behavioral arm + the Jaccard overlap.
    if lesion_mode in ("behavioral", "both"):
        n_lb_behav = sum(1 for f in PRESENCE_BAR
                         if cleared[f] and behav_behavioral.get(f, {}).get("load_bearing", False))
        result["behavioral_dependency_behavioral"] = behav_behavioral
        result["n_faculties_load_bearing_behavioral"] = n_lb_behav
    if lesion_mode == "both":
        result["lesion_overlap_jaccard"] = overlap
    if verbose:
        print(f"[seed {seed} units={units} k={pred_horizon}] cleared+LB={n_cleared_lb}/4 "
              f"core_collapse={core_collapses} coverage_ratio={coverage['ratio']:.2f} SEED_GO={seed_go} "
              f"lesion_mode={lesion_mode} ({result['elapsed_s']}s)")
        for f in PRESENCE_BAR:
            p = presence[f]
            line = (f"    {f:11s} r2/rho={_f(p['r2'])}  floors(un/raw/sh)="
                    f"{_f(p['floor_untrained'])}/{_f(p['floor_rawv1'])}/{_f(p['floor_shuffle'])}  "
                    f"cleared={cleared[f]}  LB[dec]={behav_decoding.get(f, {}).get('load_bearing')}")
            if lesion_mode in ("behavioral", "both"):
                bb = behav_behavioral.get(f, {})
                line += (f"  LB[beh]={bb.get('load_bearing')}"
                         f"  facΔ[beh]={bb.get('faculty_degradation')} rndΔ={bb.get('random_degradation')}")
                if lesion_mode == "both":
                    line += f"  jaccard={overlap.get(f)}"
            print(line)
        # FLOOR-INDEPENDENT place metric — trained vs untrained reservoir, side by side (the dissociation)
        pt, pu = place_si_trained, place_si_untrained
        print(f"    place[SI]  trained: decodeR2={_f(presence['place']['r2'])} "
              f"SI={pt.get('mean_si')} (shuf={pt.get('mean_si_shuffle')}, x{pt.get('si_real_over_shuffle_ratio')}) "
              f"place_cells={pt.get('n_place_cells')}/{pt.get('n_units')} stab={pt.get('mean_stability')}")
        print(f"    place[SI]  UNTRAINED: decodeR2={_f(presence['place']['floor_untrained'])} "
              f"SI={pu.get('mean_si')} (shuf={pu.get('mean_si_shuffle')}, x{pu.get('si_real_over_shuffle_ratio')}) "
              f"place_cells={pu.get('n_place_cells')}/{pu.get('n_units')} stab={pu.get('mean_stability')}  "
              f"<- HIGH decode, LOW SI = the inflated floor")
        # FLOOR-INDEPENDENT object/permanence/value — same trained-vs-untrained side-by-side print
        ot, ou = object_si_trained, object_si_untrained
        print(f"    object[SI] trained:   RSA={_f(presence['object']['r2'])} "
              f"SI={ot.get('mean_si')} (shuf={ot.get('mean_si_shuffle')}, x{ot.get('si_real_over_shuffle_ratio')}) "
              f"sel_units={ot.get('n_object_selective_units')}/{ot.get('n_units')}")
        print(f"    object[SI] UNTRAINED: RSA={_f(presence['object']['floor_untrained'])} "
              f"SI={ou.get('mean_si')} (shuf={ou.get('mean_si_shuffle')}, x{ou.get('si_real_over_shuffle_ratio')}) "
              f"sel_units={ou.get('n_object_selective_units')}/{ou.get('n_units')}")
        pmt, pmu = permanence_si_trained, permanence_si_untrained
        print(f"    perman[SI] trained:   decodeR2={_f(presence['permanence']['r2'])} "
              f"SI={pmt.get('mean_si')} (shuf={pmt.get('mean_si_shuffle')}, x{pmt.get('si_real_over_shuffle_ratio')}) "
              f"cells={pmt.get('n_permanence_cells')}/{pmt.get('n_units')}")
        print(f"    perman[SI] UNTRAINED: decodeR2={_f(presence['permanence']['floor_untrained'])} "
              f"SI={pmu.get('mean_si')} (shuf={pmu.get('mean_si_shuffle')}, x{pmu.get('si_real_over_shuffle_ratio')}) "
              f"cells={pmu.get('n_permanence_cells')}/{pmu.get('n_units')}")
        vt, vu = value_si_trained, value_si_untrained
        print(f"    value[SI]  trained:   decodeR2={_f(presence['value']['r2'])} "
              f"SI={vt.get('mean_si')} (shuf={vt.get('mean_si_shuffle')}, x{vt.get('si_real_over_shuffle_ratio')}) "
              f"tuned_units={vt.get('n_value_tuned_units')}/{vt.get('n_units')}")
        print(f"    value[SI]  UNTRAINED: decodeR2={_f(presence['value']['floor_untrained'])} "
              f"SI={vu.get('mean_si')} (shuf={vu.get('mean_si_shuffle')}, x{vu.get('si_real_over_shuffle_ratio')}) "
              f"tuned_units={vu.get('n_value_tuned_units')}/{vu.get('n_units')}")
    return result


def _f(x):
    try:
        return round(float(x), 4)
    except Exception:
        return None


# ── input-sequence capture (exact untrained replay) ─────────────────────────
def _capture_eval_seq(world, sub, n):
    """A short fixed (v1, a_prev, d, reward) sequence for held-out pred-error under lesion. Does not train."""
    seq = []
    was_frozen = sub._frozen
    sub.freeze()
    a_prev = -1
    for _ in range(n):
        d = world.drive_afferent(); v1 = world.crop_v1feat()
        v1h = np.asarray(v1.get() if hasattr(v1, "get") else v1, dtype=np.float32)
        h = sub.observe(v1, a_prev, d); a = sub.act(h, explore_eps=0.1)
        r, _ = world.step(a); sub.learn(r)
        seq.append((v1h, a_prev, np.asarray(d, np.float32), float(r)))
        a_prev = a
    if not was_frozen:
        sub.unfreeze()
    return seq


def _pred_err(sub, eval_seq):
    sub.set_lesion_mask(None)
    return sub.eval_predictive_loss(eval_seq)


def _coverage(world, sub, n_steps, curiosity, restart_seed=12345):
    """Distinct cells visited in `n_steps` by the curiosity policy vs a uniform-random policy,
    from a randomized start (exploration efficiency in a non-saturating window)."""
    sub.freeze()
    sub.set_lesion_mask(None)
    G = world.cfg.grid_size
    rng = np.random.default_rng(restart_seed)
    world.agent = (int(rng.integers(G)), int(rng.integers(G)))
    sub.reset_state()
    seen = set()
    a_prev = -1
    for _ in range(n_steps):
        d = world.drive_afferent(); v1 = world.crop_v1feat()
        h = sub.observe(v1, a_prev, d)
        if curiosity:
            a = sub.act(h, explore_eps=0.0)      # the substrate's OWN policy, no forced exploration
        else:
            a = int(rng.integers(N_ACTIONS))
        world.step(a); seen.add(tuple(world.agent)); a_prev = a
    return len(seen)


def _core_lesion_presence(sub, H_un, RAW, POS, value_lab, FOOD, off, obj_idx, obj_lab,
                          world, wcfg, seed, units, encoder, n_hidden, n_latent, n_probe):
    """Zero W_h, re-run a frozen probe, decode faculties. All integration faculties must collapse."""
    xp = sub.xp
    saved = sub.P["W_h"]
    sub.P["W_h"] = xp.zeros_like(saved)
    sub.freeze()
    world.reset(seed + 55)
    if getattr(world.cfg, "nav_curriculum", False):
        world.set_curriculum_progress(1.0)     # reset zeroed progress; probe at full difficulty
    pr = rollout(world, sub, n_probe, train=False, explore_eps=0.1, collect=True)
    sub.P["W_h"] = saved
    H = pr["H"]; POS2 = pr["POS"]; FOOD2 = pr["FOOD"]; FIC2 = pr["FIC"]; OBJ2 = pr["OBJ"]; REW2 = pr["REW"]
    G = np.zeros(len(REW2), np.float32); acc = 0.0
    for i in range(len(REW2) - 1, -1, -1):
        acc = REW2[i] + 0.95 * acc; G[i] = acc
    off2 = (FIC2 < 0.5) & (FOOD2[:, 0] >= 0)
    oidx = [i for i, o in enumerate(OBJ2) if len(o) == 1]
    out = {}
    n = len(POS2); perm = np.random.default_rng(seed + 5).permutation(n); cut = int(0.7 * n); tr, te = perm[:cut], perm[cut:]
    out["place"] = _ridge_r2(H[tr], POS2[tr], H[te], POS2[te])
    out["value"] = _ridge_r2(H[tr], G[tr, None], H[te], G[te, None])
    if off2.sum() >= 20:
        m = off2; out["permanence"] = _ridge_r2(H[m][: int(0.7 * m.sum())], FOOD2[m][: int(0.7 * m.sum())],
                                                H[m][int(0.7 * m.sum()):], FOOD2[m][int(0.7 * m.sum()):])
    else:
        out["permanence"] = float("nan")
    out["object"] = _rsa(H[oidx], np.asarray([OBJ2[i][0] for i in oidx], np.int64)) if len(oidx) >= 8 else float("nan")
    return out


def _rate_map_summary(H, POS, grid_size):
    """Compact rate-map summary: for the 8 most position-selective units, their spatial selectivity
    (max-bin / mean) — the visually-convincing bonus, stored compactly (not full maps)."""
    if len(H) < 50:
        return {"note": "too few probe steps"}
    xs = POS[:, 0].astype(int); ys = POS[:, 1].astype(int)
    sel = []
    for u in range(H.shape[1]):
        grid = np.zeros((grid_size, grid_size)); cnt = np.zeros((grid_size, grid_size))
        np.add.at(grid, (xs, ys), H[:, u]); np.add.at(cnt, (xs, ys), 1.0)
        occ = cnt > 0
        if occ.sum() < 5:
            continue
        m = grid[occ] / cnt[occ]
        sel.append((float(m.max() - m.mean()) / (abs(m.mean()) + 1e-6), u))
    sel.sort(reverse=True)
    return {"n_units_scored": len(sel),
            "top8_selectivity": [round(s, 3) for s, _ in sel[:8]],
            "top8_units": [int(u) for _, u in sel[:8]]}


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL-INFORMATION place-cell metric (FLOOR-INDEPENDENT, additive diagnostic)
# ─────────────────────────────────────────────────────────────────────────────
PLACE_SI_SHUFFLES = 100           # shuffle-null draws for the SI significance test
PLACE_SI_STABILITY_THRESH = 0.30  # split-half rate-map correlation a place cell must exceed


def _place_cell_metrics(H, POS, grid_size, seed, n_shuffle=PLACE_SI_SHUFFLES,
                        stab_thresh=PLACE_SI_STABILITY_THRESH):
    """FLOOR-INDEPENDENT place metric: Skaggs spatial information + rate-map stability + a shuffle-null
    significance test, computed per hidden unit from H (states) and POS (true (x,y)).

    WHY THIS EXISTS (the floor-scaling finding). The pre-registered place metric is LINEAR-DECODE R^2 of
    (x,y) from h_t, and that metric is a CAPACITY ARTIFACT: a big UNTRAINED random reservoir linearly-
    decodes position well (its floor R^2 rises with n_hidden, ~0.49@128 -> ~0.72@2048) WITHOUT any single
    unit being place-tuned, because a linear head over many random features approximates position
    (Schoyen 2023: decodability != tuning). At n_hidden=512 a genuine emergent place code (decode ~0.6-0.7)
    is masked because it sits at/below the inflated floor. Skaggs SI + stability are SINGLE-UNIT tuning
    measures — a random reservoir scores near its OWN shuffle null on them even while its population decode
    is high — so this metric credits a genuine place code and does NOT credit the reservoir. It is reported
    ALONGSIDE (never replacing) the linear-decode place metric + the pre-registered GO gate.

    RATE PROXY. Firing rates are non-negative, so the unit "rate" is the RECTIFIED activation max(0, h):
    rate units are tanh in [-1,1] (the negative lobe is treated as no firing, as a real cell's sub-threshold
    drive is); spike units carry a non-negative low-pass trace, so rectification is ~inert there. The SAME
    transform is applied to the trained core and the untrained reservoir, so the comparison is fair.

    SKAGGS bits-per-activation:   SI_u = Sum_i p_i (lam_i / lam) log2(lam_i / lam)
      p_i   = occupancy of spatial bin i (fraction of steps whose (x,y) fell in bin i)
      lam_i = mean rectified activation of unit u over the steps in bin i
      lam   = overall mean rectified activation of unit u  ( = Sum_i p_i lam_i )
    Bins with lam_i = 0 contribute 0 (0*log0 := 0). This is a KL divergence KL(q||p), q_i = p_i lam_i/lam,
    so SI >= 0 always. It is MAGNITUDE-NORMALIZED (bits per activation): a unit that merely fires MORE is
    not rewarded — only spatial CONCENTRATION of firing raises SI. (This is the property the linear decode
    lacks and why the metric is floor-independent.)

    STABILITY. Split the rollout into first/second temporal halves, build a rate map for each, and
    Pearson-correlate the two per unit over bins occupied in BOTH halves. Place cells are stable; a random
    unit's two half-maps are uncorrelated.

    SIGNIFICANCE. Shuffle the POS<->state correspondence n_shuffle times (occupancy p_i and overall lam are
    permutation-invariant; only lam_i changes), recompute SI -> a per-unit null distribution. A unit is a
    'place cell' iff its real SI exceeds its OWN shuffle 95th percentile AND its stability > stab_thresh.

    Returns population summaries (mean/median/max SI, #/frac significant place cells, mean stability) plus
    the pooled shuffle-null level and the real/shuffle SI ratio, so trained vs untrained read side-by-side.
    HONESTY: this is a functional tuning read-out; it asserts nothing about felt spatial experience.
    """
    H = np.asarray(H, dtype=np.float64)
    POS = np.asarray(POS, dtype=np.float64)
    if H.ndim != 2 or len(H) < 50:
        return {"note": "too few probe steps", "n_steps": int(len(H))}
    T, U = H.shape
    R = np.maximum(0.0, H)                         # non-negative rate proxy (rectified activation)
    xs = np.clip(POS[:, 0].astype(int), 0, grid_size - 1)
    ys = np.clip(POS[:, 1].astype(int), 0, grid_size - 1)
    bin_idx = xs * grid_size + ys                 # (T,) flat spatial bin per step
    n_bins = grid_size * grid_size
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)   # (n_bins,) occupancy count
    occ = counts > 0
    n_occ = int(occ.sum())
    if n_occ < 3:
        return {"note": "too few occupied bins", "n_occupied_bins": n_occ, "n_steps": int(T)}
    p_i = counts / counts.sum()                   # (n_bins,) occupancy probability

    # one-hot bin matrix B (n_bins x T): sum_R = B @ R gives per-bin summed rate (n_bins x U). Permuting R's
    # rows realizes the shuffle null cheaply via BLAS (occupancy/overall-mean stay invariant, only lam_i moves).
    B = np.zeros((n_bins, T), dtype=np.float64)
    B[bin_idx, np.arange(T)] = 1.0
    lam = R.mean(axis=0)                           # (U,) overall mean rate ( = Sum_i p_i lam_i )
    live = lam > 1e-9                              # dead (all-nonpositive) units -> SI := 0

    def _si_from_sumR(sum_R):
        lam_i = np.zeros_like(sum_R)               # (n_bins x U)
        lam_i[occ] = sum_R[occ] / counts[occ, None]
        lam_safe = np.where(live, lam, 1.0)        # avoid /0 for dead units (masked out below)
        ratio = lam_i / lam_safe[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            contrib = p_i[:, None] * ratio * np.log2(ratio)
        contrib[~np.isfinite(contrib)] = 0.0       # lam_i=0 bins (0*log0) + unoccupied bins -> 0
        si = contrib.sum(axis=0)
        si[~live] = 0.0
        return si

    si_real = _si_from_sumR(B @ R)                 # (U,) real spatial information per unit

    rng = np.random.default_rng(seed + 4242)
    si_shuf = np.empty((n_shuffle, U), dtype=np.float64)
    for s in range(n_shuffle):
        si_shuf[s] = _si_from_sumR(B @ R[rng.permutation(T)])
    thresh_u = np.percentile(si_shuf, 95, axis=0)  # (U,) per-unit 95th-pct shuffle null

    # split-half rate-map stability per unit over co-occupied bins
    def _ratemap(sl):
        bi = bin_idx[sl]
        c = np.bincount(bi, minlength=n_bins).astype(np.float64)
        Bh = np.zeros((n_bins, len(bi))); Bh[bi, np.arange(len(bi))] = 1.0
        o = c > 0
        rm = np.full((n_bins, U), np.nan)
        rm[o] = (Bh @ R[sl])[o] / c[o, None]
        return rm, o

    half = T // 2
    rm1, o1 = _ratemap(slice(0, half))
    rm2, o2 = _ratemap(slice(half, T))
    both = o1 & o2
    stability = np.full(U, np.nan)
    if int(both.sum()) >= 3:
        a = rm1[both]; b = rm2[both]               # (n_both x U)
        am = a - a.mean(0); bm = b - b.mean(0)
        denom = np.sqrt((am * am).sum(0) * (bm * bm).sum(0)) + 1e-12
        stability = (am * bm).sum(0) / denom
        stability[~live] = np.nan

    is_place = live & (si_real > thresh_u) & (np.nan_to_num(stability, nan=-1.0) > stab_thresh)
    n_place = int(is_place.sum())
    live_any = bool(live.any())
    mean_si_live = float(np.nanmean(si_real[live])) if live_any else float("nan")
    return {
        "n_steps": int(T), "n_units": int(U), "n_live_units": int(live.sum()),
        "n_occupied_bins": n_occ, "n_shuffle": int(n_shuffle), "stability_thresh": stab_thresh,
        "mean_si": _f(mean_si_live) if live_any else None,
        "median_si": _f(np.nanmedian(si_real[live])) if live_any else None,
        "max_si": _f(np.nanmax(si_real[live])) if live_any else None,
        "mean_si_shuffle": _f(float(si_shuf.mean())),
        "si_95_shuffle_pooled": _f(float(np.percentile(si_shuf, 95))),
        "si_real_over_shuffle_ratio": _f(mean_si_live / (float(si_shuf.mean()) + 1e-9)) if live_any else None,
        "mean_stability": _f(float(np.nanmean(stability))) if np.isfinite(stability).any() else None,
        "n_place_cells": n_place,
        "frac_place_cells": _f(n_place / U),
        "mean_si_place_cells": _f(float(np.nanmean(si_real[is_place]))) if n_place > 0 else None,
        "mean_stability_place_cells": _f(float(np.nanmean(stability[is_place]))) if n_place > 0 else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FLOOR-INDEPENDENT diagnostics for OBJECT / PERMANENCE / VALUE (ADDITIVE, additive-only)
#
# WHY (research-pass motivation). _r2_with_floors is the IDENTICAL ridge-linear-decode-vs-3-floors function
# used for place/value/permanence, and object's RSA is a population-similarity metric with the same failure
# mode: a big random reservoir can score well on either instrument WITHOUT any single unit being genuinely
# TUNED, because (a) linear decode over many random features approximates a smooth label (the place-metric
# finding: floor R^2 rises with n_hidden), and (b) RSA over h can track INPUT similarity (the raw V1 already
# differs by object type) rather than a learned category code. Both are magnitude/capacity-sensitive, not
# tuning-sensitive. The place fix's floor-independent answer was Skaggs spatial information: a magnitude-
# NORMALIZED (bits per activation) measure of whether a unit's firing CONCENTRATES on specific bins of a
# label, tested against the unit's OWN shuffle null. That formalism generalizes beyond spatial (x,y) bins to
# ANY discrete binning of a label -- so the SAME math (below, factored out of _place_cell_metrics as
# `_categorical_tuning_metrics`) is reused for:
#   OBJECT      bins = the 4 discrete object types already used by the RSA presence metric (obj_lab).
#   PERMANENCE  bins = (x,y) of the REMEMBERED off-crop food location -- this IS the place-SI formalism,
#               just keyed on the food's location instead of the agent's own, so it delegates to
#               `_place_cell_metrics` directly rather than re-deriving the identical math.
#   VALUE       bins = quantiles of the discounted-return label (a continuous target discretized the same
#               way (x,y) is already discretized into grid cells for place) -- "does firing concentrate by
#               value level" is a floor-independent stand-in for "TD-consistency" that needs no extra
#               rollouts or a trained critic to evaluate.
# Each is computed for BOTH the trained core (H) and the untrained-reservoir replay (H_un) on the identical
# input sequence, exactly as place_cell_si / place_cell_si_untrained are, so the trained-vs-reservoir
# separation is visible per faculty. ADDITIVE ONLY: presence[...], _beats_floors, and the pre-registered GO
# gate are untouched; these are new per-seed + aggregate diagnostic fields.
# ─────────────────────────────────────────────────────────────────────────────
VALUE_SI_N_BINS = 5   # target # of discounted-return quantile bins for the value tuning-info metric


def _categorical_tuning_metrics(H, bin_idx, n_bins, seed, n_shuffle=PLACE_SI_SHUFFLES,
                                stab_thresh=PLACE_SI_STABILITY_THRESH, cell_noun="tuned"):
    """Generic FLOOR-INDEPENDENT tuning-information metric: the SAME Skaggs-style bits-per-activation math as
    `_place_cell_metrics`, generalized from spatial (x,y) bins to ANY discrete per-step label `bin_idx`
    (0..n_bins-1) -- object type, a value-return quantile, etc. A unit's rectified rate must CONCENTRATE on
    specific bins (KL(q||p) so SI>=0, and >0 only under non-uniform firing across bins) to clear its OWN
    shuffle null; a reservoir unit whose firing is driven near-uniformly by input magnitude does not clear
    this even when a population linear-decode/RSA of the same label is high. See the section banner above
    for why this generalization is the correct fix for object/permanence/value, not just place.

    Not used for place itself (`_place_cell_metrics` is left byte-identical/untouched); permanence delegates
    to `_place_cell_metrics` directly since its label is already an (x,y) position. This function backs the
    object (categorical bins = object type) and value (quantile bins = discretized return) diagnostics.

    Returns a dict shaped like `_place_cell_metrics`'s (mean/median/max SI, shuffle-null level, real/shuffle
    ratio, mean split-half stability, #/frac significant units) but with the "place_cells" naming replaced by
    `cell_noun` so the JSON is unambiguous about which faculty it's reporting. HONESTY: a functional tuning
    read-out; asserts nothing about felt/represented content.
    """
    H = np.asarray(H, dtype=np.float64)
    bin_idx = np.asarray(bin_idx, dtype=np.int64)
    if H.ndim != 2 or len(H) < 50 or n_bins < 2:
        return {"note": "too few probe steps or bins", "n_steps": int(len(H)) if H.ndim == 2 else 0,
                "n_bins": int(n_bins)}
    T, U = H.shape
    R = np.maximum(0.0, H)                          # non-negative rate proxy (rectified activation)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    occ = counts > 0
    n_occ = int(occ.sum())
    if n_occ < 2:
        return {"note": "too few occupied bins", "n_occupied_bins": n_occ, "n_steps": int(T)}
    p_i = counts / counts.sum()

    B = np.zeros((n_bins, T), dtype=np.float64)
    B[bin_idx, np.arange(T)] = 1.0
    lam = R.mean(axis=0)
    live = lam > 1e-9

    def _si_from_sumR(sum_R):
        lam_i = np.zeros_like(sum_R)
        lam_i[occ] = sum_R[occ] / counts[occ, None]
        lam_safe = np.where(live, lam, 1.0)
        ratio = lam_i / lam_safe[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            contrib = p_i[:, None] * ratio * np.log2(ratio)
        contrib[~np.isfinite(contrib)] = 0.0
        si = contrib.sum(axis=0)
        si[~live] = 0.0
        return si

    si_real = _si_from_sumR(B @ R)
    rng = np.random.default_rng(seed + 6060)
    si_shuf = np.empty((n_shuffle, U), dtype=np.float64)
    for s in range(n_shuffle):
        si_shuf[s] = _si_from_sumR(B @ R[rng.permutation(T)])
    thresh_u = np.percentile(si_shuf, 95, axis=0)

    def _binmeans(sl):
        bi = bin_idx[sl]
        c = np.bincount(bi, minlength=n_bins).astype(np.float64)
        Bh = np.zeros((n_bins, len(bi))); Bh[bi, np.arange(len(bi))] = 1.0
        o = c > 0
        bm = np.full((n_bins, U), np.nan)
        bm[o] = (Bh @ R[sl])[o] / c[o, None]
        return bm, o

    half = T // 2
    bm1, o1 = _binmeans(slice(0, half))
    bm2, o2 = _binmeans(slice(half, T))
    both = o1 & o2
    stability = np.full(U, np.nan)
    if int(both.sum()) >= 3:
        a = bm1[both]; b = bm2[both]
        am = a - a.mean(0); bmn = b - b.mean(0)
        denom = np.sqrt((am * am).sum(0) * (bmn * bmn).sum(0)) + 1e-12
        stability = (am * bmn).sum(0) / denom
        stability[~live] = np.nan

    is_tuned = live & (si_real > thresh_u) & (np.nan_to_num(stability, nan=-1.0) > stab_thresh)
    n_tuned = int(is_tuned.sum())
    live_any = bool(live.any())
    mean_si_live = float(np.nanmean(si_real[live])) if live_any else float("nan")
    return {
        "n_steps": int(T), "n_units": int(U), "n_live_units": int(live.sum()),
        "n_bins": int(n_bins), "n_occupied_bins": n_occ, "n_shuffle": int(n_shuffle),
        "stability_thresh": stab_thresh,
        "mean_si": _f(mean_si_live) if live_any else None,
        "median_si": _f(np.nanmedian(si_real[live])) if live_any else None,
        "max_si": _f(np.nanmax(si_real[live])) if live_any else None,
        "mean_si_shuffle": _f(float(si_shuf.mean())),
        "si_95_shuffle_pooled": _f(float(np.percentile(si_shuf, 95))),
        "si_real_over_shuffle_ratio": _f(mean_si_live / (float(si_shuf.mean()) + 1e-9)) if live_any else None,
        "mean_stability": _f(float(np.nanmean(stability))) if np.isfinite(stability).any() else None,
        f"n_{cell_noun}_units": n_tuned,
        f"frac_{cell_noun}_units": _f(n_tuned / U),
        f"mean_si_{cell_noun}_units": _f(float(np.nanmean(si_real[is_tuned]))) if n_tuned > 0 else None,
        f"mean_stability_{cell_noun}_units": _f(float(np.nanmean(stability[is_tuned]))) if n_tuned > 0 else None,
    }


def _rename_cell_keys(d, noun):
    """Rename `_place_cell_metrics`'s generic 'place_cells' terminology to `noun` (e.g. 'permanence') for
    output clarity, WITHOUT touching `_place_cell_metrics`'s tested code path (it stays byte-identical)."""
    mapping = {
        "n_place_cells": f"n_{noun}_cells",
        "frac_place_cells": f"frac_{noun}_cells",
        "mean_si_place_cells": f"mean_si_{noun}_cells",
        "mean_stability_place_cells": f"mean_stability_{noun}_cells",
    }
    return {mapping.get(k, k): v for k, v in d.items()}


def _object_selectivity_metrics(H, obj_lab, seed, n_shuffle=PLACE_SI_SHUFFLES,
                                stab_thresh=PLACE_SI_STABILITY_THRESH):
    """FLOOR-INDEPENDENT object metric: per-unit object-TYPE selectivity index (Skaggs SI over the 4 discrete
    object-type bins already used by the presence RSA metric), trained vs shuffle-null. WHY floor-independent
    vs RSA: RSA(h, same-type-indicator) can be high because the raw V1 input already differs by object type
    (distinct oriented bars -> distinct Gabor responses) and a random reservoir linearly mixes that input
    similarity into h's similarity structure WITHOUT any unit being object-TUNED (magnitude/capacity, not
    tuning). SI is magnitude-normalized (bits per activation) and asks a different question per unit -- does
    ITS firing concentrate on one object type -- so a reservoir whose per-unit rate is roughly type-invariant
    (even while its POPULATION geometry echoes the input) scores near its own shuffle null. Restricted to the
    same single-object-in-crop steps the RSA presence metric uses, for an apples-to-apples population."""
    return _categorical_tuning_metrics(H, obj_lab, K_OBJECTS, seed, n_shuffle=n_shuffle,
                                       stab_thresh=stab_thresh, cell_noun="object_selective")


def _permanence_food_si_metrics(H, FOOD, grid_size, seed, n_shuffle=PLACE_SI_SHUFFLES,
                                stab_thresh=PLACE_SI_STABILITY_THRESH):
    """FLOOR-INDEPENDENT permanence metric: Skaggs spatial information of the REMEMBERED food location,
    restricted to steps where food is OFF the current crop (the permanence-relevant regime) -- IDENTICAL
    formalism to the place SI metric, just keyed on the food's (x,y) instead of the agent's own, since a
    genuine allocentric-memory ("object permanence") code should concentrate off-view firing by WHERE the
    food is remembered to be, exactly as a place code concentrates firing by where the agent itself is. Why
    floor-independent vs the linear-decode presence metric: the same capacity artifact applies (a large
    random reservoir's rate still correlates with recent input/position history enough for a linear head to
    partially reconstruct the food's location without any unit being permanence-tuned); SI is magnitude-
    normalized and tests each unit against its own shuffle null, so it does not credit the reservoir for
    incidental linear reconstructability. Delegates to `_place_cell_metrics` (identical math; not touched)."""
    raw = _place_cell_metrics(H, FOOD, grid_size, seed, n_shuffle=n_shuffle, stab_thresh=stab_thresh)
    return _rename_cell_keys(raw, "permanence")


def _quantile_bin_edges(x, n_bins):
    """De-duplicated quantile bin edges for `x` -> (edges, effective_n_bins). Ties (e.g. many exact-0 returns
    from a sparse reward signal) collapse adjacent edges, so effective_n_bins can be < n_bins; the caller
    reports that honestly (UNDEFINED, not a faked bin count) rather than forcing degenerate bins."""
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    return edges, max(0, len(edges) - 1)


def _value_tuning_metrics(H, G, seed, n_bins=VALUE_SI_N_BINS, n_shuffle=PLACE_SI_SHUFFLES,
                          stab_thresh=PLACE_SI_STABILITY_THRESH):
    """FLOOR-INDEPENDENT value metric: Skaggs tuning-information of firing w.r.t. QUANTILE bins of the
    discounted-return label G -- a floor-independent stand-in for "TD-consistency"/value-tuning that needs no
    extra rollouts or a trained critic (only a re-binning of the label already used by the presence R^2
    decode). WHY floor-independent vs linear decode: G correlates with recent drive/position/action history,
    all of which are DIRECT inputs to the reservoir, so a big random projection can linearly reconstruct G
    reasonably well without any unit being value-tuned (the same capacity story as place/permanence). SI asks
    whether a unit's firing CONCENTRATES by value level (magnitude-normalized, shuffle-null-tested per unit),
    which a reservoir echoing its inputs does not do merely by being reconstructable in aggregate. Quantile
    (not raw-value) binning keeps occupancy p_i comparable across bins the way place's spatial bins are."""
    G = np.asarray(G, dtype=np.float64).reshape(-1)
    if len(G) < 50:
        return {"note": "too few probe steps", "n_steps": int(len(G))}
    edges, eff_bins = _quantile_bin_edges(G, n_bins)
    if eff_bins < 2:
        return {"note": "insufficient value spread for quantile binning", "n_steps": int(len(G)),
                "n_unique_values": int(len(np.unique(G)))}
    bin_idx = np.clip(np.digitize(G, edges[1:-1], right=False), 0, eff_bins - 1)
    out = _categorical_tuning_metrics(H, bin_idx, eff_bins, seed, n_shuffle=n_shuffle,
                                      stab_thresh=stab_thresh, cell_noun="value_tuned")
    out["n_value_bins_requested"] = int(n_bins)
    out["n_value_bins_effective"] = int(eff_bins)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# NOVEL-START SHORTCUT / DETOUR PROBE (Banino 2018) — the sharp behavioral-load-bearing test
# ─────────────────────────────────────────────────────────────────────────────
def _shortcut_probe(sub, wcfg, seed, place_importance, n_hidden, lesion_frac=0.10,
                    n_random_lesions=3, exclude_cells=frozenset(), n_starts=24, verbose=False):
    """HELD-OUT novel-start homing probe — the decisive load-bearing test the curriculum unlocks.

    Evaluate the FROZEN trained policy homing to the fixed, OUT-OF-VIEW larder from a set of far START cells
    that were NOT used as training respawns (drawn from an independent RNG = an out-of-sample eval set). A
    reactive/memoryless policy CANNOT home from a novel far start: the larder is invisible from afar and — with
    shaping off — nothing in the observation points to it, so there is no local gradient to hill-climb. Only a
    policy routing through a PERSISTENT, path-integrated place code reaches it. So success on this set, and its
    DROP under a place-unit lesion vs an equal RANDOM-unit lesion, is a low-noise causal read of whether the
    emergent place code is behaviorally load-bearing (Banino 2018's shortcut/detour test). HONESTY: functional
    read-out only. Requires nav_required (a fixed larder); returns None-ish note if no far start cells exist.
    """
    world = ForkPCSWorld(wcfg)
    larder = world.larder
    G = wcfg.grid_size
    dist_min = max(wcfg.crop_radius + 1, wcfg.nav_dmin)     # out of view AND >= the trial-reset distance
    occupied = set(world.objects) | set(world.landmarks) | {larder}
    all_far = [(x, y) for x in range(G) for y in range(G)
               if (x, y) not in occupied and _manhattan((x, y), larder) >= dist_min]
    heldout = [c for c in all_far if c not in exclude_cells]
    # prefer truly-held-out far cells; if too few exist (training covered the far region), fall back to all far
    # cells (reported), so the probe still yields a number — the lesion contrast is still valid either way.
    pool = heldout if len(heldout) >= max(6, n_starts // 2) else all_far
    if len(pool) == 0:
        return {"note": "no far start cells available", "n_starts": 0, "shortcut_probe_success": None,
                "place_load_bearing": False}
    rng = np.random.default_rng(seed + 99991)
    idx = rng.choice(len(pool), size=int(min(n_starts, len(pool))), replace=False)
    starts = [pool[int(i)] for i in idx]
    n_truly_heldout = int(sum(1 for c in starts if c not in exclude_cells))
    max_steps = int(max(2 * (G - 1), 3 * G))               # slack for the farthest start; bounded so a random
                                                           # walk rarely reaches a specific far cell in-window

    def _run_starts(mask):
        sub.set_lesion_mask(mask)
        succ = 0; steps_to = []
        for s in starts:
            world.reset(wcfg.seed)                          # fresh drive + fixed layout (larder unchanged)
            world.agent = s
            world.food = larder
            world.energy = 0.5 * wcfg.set_point             # hungry -> a real interoceptive drive to home
            world._prime_drive()
            a_prev = -1; reached = False; used = max_steps
            for step_i in range(max_steps):
                d = world.drive_afferent(); v1 = world.crop_v1feat()
                h = sub.observe(v1, a_prev, d)
                a = sub.act(h, explore_eps=0.0)             # the pure frozen policy (no forced exploration)
                r, info = world.step(a)
                a_prev = a
                if info.get("ate"):                         # stepped onto the larder = homed successfully
                    reached = True; used = step_i + 1; break
            if reached:
                succ += 1; steps_to.append(used)
        sub.set_lesion_mask(None)
        return succ / len(starts), (float(np.mean(steps_to)) if steps_to else float("nan"))

    k = max(8, int(lesion_frac * n_hidden))
    place_mask = np.zeros(n_hidden, dtype=bool)
    place_mask[np.argsort(place_importance)[::-1][:k]] = True

    intact_succ, intact_steps = _run_starts(None)
    place_succ, _ = _run_starts(place_mask)
    rng2 = np.random.default_rng(seed + 131)
    rand_succs = []
    for _ in range(n_random_lesions):
        m = np.zeros(n_hidden, dtype=bool); m[rng2.choice(n_hidden, k, replace=False)] = True
        rs, _ = _run_starts(m); rand_succs.append(rs)
    rand_succ = float(np.mean(rand_succs)) if rand_succs else float("nan")
    place_deg = intact_succ - place_succ
    rand_deg = intact_succ - rand_succ
    frac = attributable_to("shortcut place-lesion", place_deg, rand_deg)
    load_bearing = (frac is not None) and (place_deg > 0) and (frac >= (1.0 - 1.0 / BEHAV_LESION_RATIO))
    out = {
        "n_starts": len(starts), "n_truly_heldout": n_truly_heldout,
        "n_train_respawn_cells": int(len(exclude_cells)), "dist_min": int(dist_min),
        "max_steps": max_steps, "k_units": int(k),
        "shortcut_probe_success": _f(intact_succ),
        "success_place_lesion": _f(place_succ), "success_random_lesion": _f(rand_succ),
        "shortcut_probe_place_lesion_degradation": _f(place_deg),
        "random_lesion_degradation": _f(rand_deg),
        "attributable_fraction": _f(frac),
        "shortcut_probe_place_lesion_degradation_vs_random": _f((place_deg / rand_deg) if abs(rand_deg) > 1e-9
                                                                 else float("inf") if place_deg > 1e-9 else float("nan")),
        "place_load_bearing": bool(load_bearing),
        "intact_mean_steps_to_larder": _f(intact_steps),
    }
    if verbose:
        print(f"    shortcut-probe: success={out['shortcut_probe_success']} "
              f"(place-lesion {out['success_place_lesion']}, random-lesion {out['success_random_lesion']})  "
              f"place_deg={out['shortcut_probe_place_lesion_degradation']} vs rand_deg={out['random_lesion_degradation']}  "
              f"LOAD_BEARING={out['place_load_bearing']}  "
              f"(n_starts={out['n_starts']}, truly_heldout={out['n_truly_heldout']}, dist_min={out['dist_min']})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# aggregate + main
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(per_seed):
    n = len(per_seed)
    n_go = sum(1 for r in per_seed if r["SEED_GO"])
    n_cov = sum(1 for r in per_seed if r["coverage"].get("pass"))
    faculty_lb_counts = {f: sum(1 for r in per_seed if r["behavioral_dependency"].get(f, {}).get("load_bearing"))
                         for f in PRESENCE_BAR}
    emergence_go = (n_go >= int(np.ceil(SEEDS_REQUIRED_FRAC * n))) and (n_cov >= int(np.ceil(SEEDS_REQUIRED_FRAC * n)))

    # FLOOR-INDEPENDENT place metric summary (additive diagnostic; NOT part of the GO gate). Mean across
    # seeds of the linear-decode place metric vs the spatial-information metric, trained core vs untrained
    # reservoir — the side-by-side that shows the untrained floor is high-decode-but-low-SI.
    def _col(getter):
        vals = [getter(r) for r in per_seed]
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        return _f(float(np.mean(vals))) if vals else None
    place_cell_si_summary = {
        "trained_place_decode_r2": _col(lambda r: r["presence"]["place"].get("r2")),
        "untrained_place_decode_r2": _col(lambda r: r["presence"]["place"].get("floor_untrained")),
        "trained_mean_si": _col(lambda r: r["place_cell_si"].get("mean_si")),
        "untrained_mean_si": _col(lambda r: r["place_cell_si_untrained"].get("mean_si")),
        "trained_si_over_shuffle_ratio": _col(lambda r: r["place_cell_si"].get("si_real_over_shuffle_ratio")),
        "untrained_si_over_shuffle_ratio": _col(lambda r: r["place_cell_si_untrained"].get("si_real_over_shuffle_ratio")),
        "trained_frac_place_cells": _col(lambda r: r["place_cell_si"].get("frac_place_cells")),
        "untrained_frac_place_cells": _col(lambda r: r["place_cell_si_untrained"].get("frac_place_cells")),
        "trained_mean_stability": _col(lambda r: r["place_cell_si"].get("mean_stability")),
        "untrained_mean_stability": _col(lambda r: r["place_cell_si_untrained"].get("mean_stability")),
    }
    # FLOOR-INDEPENDENT object/permanence/value summaries (additive; NOT part of the GO gate) — same
    # trained-vs-untrained-reservoir side-by-side as place_cell_si_summary, plus their linear-decode/RSA
    # presence[...] counterparts for direct comparison (does THIS faculty show the same capacity artifact?).
    object_selectivity_si_summary = {
        "trained_rsa": _col(lambda r: r["presence"]["object"].get("r2")),
        "untrained_rsa": _col(lambda r: r["presence"]["object"].get("floor_untrained")),
        "trained_mean_si": _col(lambda r: r["object_selectivity_si"].get("mean_si")),
        "untrained_mean_si": _col(lambda r: r["object_selectivity_si_untrained"].get("mean_si")),
        "trained_si_over_shuffle_ratio": _col(lambda r: r["object_selectivity_si"].get("si_real_over_shuffle_ratio")),
        "untrained_si_over_shuffle_ratio": _col(lambda r: r["object_selectivity_si_untrained"].get("si_real_over_shuffle_ratio")),
        "trained_frac_object_selective_units": _col(lambda r: r["object_selectivity_si"].get("frac_object_selective_units")),
        "untrained_frac_object_selective_units": _col(lambda r: r["object_selectivity_si_untrained"].get("frac_object_selective_units")),
    }
    permanence_food_si_summary = {
        "trained_decode_r2": _col(lambda r: r["presence"]["permanence"].get("r2")),
        "untrained_decode_r2": _col(lambda r: r["presence"]["permanence"].get("floor_untrained")),
        "trained_mean_si": _col(lambda r: r["permanence_food_si"].get("mean_si")),
        "untrained_mean_si": _col(lambda r: r["permanence_food_si_untrained"].get("mean_si")),
        "trained_si_over_shuffle_ratio": _col(lambda r: r["permanence_food_si"].get("si_real_over_shuffle_ratio")),
        "untrained_si_over_shuffle_ratio": _col(lambda r: r["permanence_food_si_untrained"].get("si_real_over_shuffle_ratio")),
        "trained_frac_permanence_cells": _col(lambda r: r["permanence_food_si"].get("frac_permanence_cells")),
        "untrained_frac_permanence_cells": _col(lambda r: r["permanence_food_si_untrained"].get("frac_permanence_cells")),
    }
    value_tuning_si_summary = {
        "trained_decode_r2": _col(lambda r: r["presence"]["value"].get("r2")),
        "untrained_decode_r2": _col(lambda r: r["presence"]["value"].get("floor_untrained")),
        "trained_mean_si": _col(lambda r: r["value_tuning_si"].get("mean_si")),
        "untrained_mean_si": _col(lambda r: r["value_tuning_si_untrained"].get("mean_si")),
        "trained_si_over_shuffle_ratio": _col(lambda r: r["value_tuning_si"].get("si_real_over_shuffle_ratio")),
        "untrained_si_over_shuffle_ratio": _col(lambda r: r["value_tuning_si_untrained"].get("si_real_over_shuffle_ratio")),
        "trained_frac_value_tuned_units": _col(lambda r: r["value_tuning_si"].get("frac_value_tuned_units")),
        "untrained_frac_value_tuned_units": _col(lambda r: r["value_tuning_si_untrained"].get("frac_value_tuned_units")),
    }
    # NOVEL-START shortcut-probe summary (additive; only present when the nav runs carried it). The decisive
    # behavioral-load-bearing read: mean held-out homing success + how much a place-unit lesion degrades it
    # vs a random-unit lesion, and on how many seeds place was load-bearing on THIS probe.
    sc = [r["shortcut_probe"] for r in per_seed if isinstance(r.get("shortcut_probe"), dict)
          and r["shortcut_probe"].get("shortcut_probe_success") is not None]
    shortcut_summary = None
    if sc:
        def _mean_key(key):
            vals = [s[key] for s in sc if isinstance(s.get(key), (int, float))]
            return _f(float(np.mean(vals))) if vals else None
        shortcut_summary = {
            "n_seeds_with_probe": len(sc),
            "mean_success": _mean_key("shortcut_probe_success"),
            "mean_success_place_lesion": _mean_key("success_place_lesion"),
            "mean_success_random_lesion": _mean_key("success_random_lesion"),
            "mean_place_lesion_degradation": _mean_key("shortcut_probe_place_lesion_degradation"),
            "mean_random_lesion_degradation": _mean_key("random_lesion_degradation"),
            "n_seeds_place_load_bearing": int(sum(1 for s in sc if s.get("place_load_bearing"))),
        }

    out = {"n_seeds": n, "n_seed_go": n_go, "n_coverage_pass": n_cov,
           "faculty_load_bearing_counts": faculty_lb_counts,
           "seeds_required": int(np.ceil(SEEDS_REQUIRED_FRAC * n)),
           "place_cell_si_summary": place_cell_si_summary,
           "object_selectivity_si_summary": object_selectivity_si_summary,
           "permanence_food_si_summary": permanence_food_si_summary,
           "value_tuning_si_summary": value_tuning_si_summary,
           "EMERGENCE_GO": bool(emergence_go)}
    if shortcut_summary is not None:
        out["shortcut_probe_summary"] = shortcut_summary
    return out


def main():
    ap = argparse.ArgumentParser(description="AGI-fork first-move emergence battery")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--units", choices=["rate", "spike"], default="rate")
    ap.add_argument("--encoder", choices=["learned_ema", "fixed"], default="learned_ema")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--n-hidden", type=int, default=512)
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--consolidation", action="store_true",
                    help="enable the hippocampal-replay consolidation companion (default OFF = the baseline arm)")
    ap.add_argument("--si", action="store_true",
                    help="enable Synaptic Intelligence weight-anchoring (Zenke 2017) anti-forgetting on the online path (default OFF)")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="global grad-norm clip (default 1.0 = stable). Set 5.0 to reproduce the unstable control; 0 disables.")
    ap.add_argument("--grad-skip-factor", type=float, default=8.0,
                    help="relative spike-skip guard (default 8.0). Set 0 to reproduce the unstable control.")
    ap.add_argument("--ema-momentum", type=float, default=0.9999,
                    help="JEPA target EMA decay (default 0.9999 = slow/stable target; 0.999 only delays the "
                         "climb). Set 0.99 to reproduce the drifting-target control; --encoder fixed is flattest.")
    ap.add_argument("--ema-warmup", type=int, default=0,
                    help="hold the JEPA target frozen for the first N updates (0=off)")
    ap.add_argument("--pred-horizon", type=int, default=1,
                    help="JEPA prediction horizon k (default 1 = the 1-step control, byte-identical to the "
                         "first-move code + its 5 artifacts). k>1 (the 2nd move; use 4 and 8) ADDS an "
                         "h-step-ahead term (predict view_{t+k} from h_t + the summed efference of the k "
                         "intervening actions), forcing path-integration so the place code is load-bearing.")
    ap.add_argument("--nav-required", action="store_true",
                    help="3rd move — TASK-REQUIRED position. Switch the world to the fixed-larder homing task "
                         "(food at a remembered, out-of-view larder; agent displaced on each eat), so reaching "
                         "food REQUIRES a persistent path-integrated place code. Default OFF = the random-respawn "
                         "control (byte-identical world). Pair with --value-weight>0 so reward/value shapes the core.")
    ap.add_argument("--nav-shaping", type=float, default=0.0,
                    help="4th move: potential-based approach-shaping coefficient (0=OFF). Makes homing LEARNABLE "
                         "but LEAKS a per-step position-derived goal gradient the agent can hill-climb reactively "
                         "-> the place code goes behaviorally INERT. PREFER --nav-curriculum (shaping-free).")
    ap.add_argument("--nav-curriculum", action="store_true",
                    help="5th move: REVERSE/START-DISTANCE CURRICULUM — the shaping-FREE learnability mechanism "
                         "(Florensa 2017; Andrychowicz 2017). Start the agent NEAR the larder (homing learnable "
                         "from the SPARSE TERMINAL drive-reduction reward alone) and ramp the post-eat respawn "
                         "distance out to the full nav_dmin over the first --nav-curriculum-frac of training. NO "
                         "leaked goal gradient (reward stays pure drive-reduction) -> the place code stays "
                         "load-bearing. Use INSTEAD of --nav-shaping, with --nav-required. Default OFF "
                         "(byte-identical).")
    ap.add_argument("--nav-curriculum-frac", type=float, default=0.5,
                    help="fraction of training over which the reverse-curriculum respawn distance ramps to nav_dmin")
    ap.add_argument("--nav-curriculum-dmin-start", type=int, default=1,
                    help="reverse-curriculum starting respawn distance (near the larder) at progress 0")
    ap.add_argument("--nav-dmin", type=int, default=6,
                    help="min post-eat agent-respawn Manhattan distance from the larder (nav-required only)")
    ap.add_argument("--value-weight", type=float, default=0.0,
                    help="weight of the value(return)-prediction head (default 0.0 = OFF, byte-identical: no "
                         "w_v/b_v params). >0 adds a value head whose gradient flows into the shared core "
                         "(value shapes cortex) + serves as the actor-critic baseline. Use 1.0 with --nav-required.")
    ap.add_argument("--sr-weight", type=float, default=0.0,
                    help="weight of the successor-representation head (default 0.0 = OFF, byte-identical: no "
                         "W_sr/b_sr params). >0 adds an SR head predicting gamma-discounted future latent "
                         "occupancy (Stachenfeld 2017: place cells ARE an SR); its gradient flows into the "
                         "shared core, making position load-bearing on the self-supervised objective (no host "
                         "position label).")
    ap.add_argument("--aux-loc-weight", type=float, default=0.0,
                    help="weight of the supervised self-localization auxiliary loss (5th move; default 0.0 = "
                         "OFF, byte-identical: no W_loc/b_loc params, no position target read). >0 adds a linear "
                         "readout of the true allocentric (x,y) from h_t at every step (Cueva & Wei 2018; Banino "
                         "2018), whose gradient flows into the shared core -> position load-bearing on the "
                         "objective DIRECTLY. The host-supplied position target is a fork-accepted mild scaffold "
                         "(same category as reward shaping), supplied per-step on the training rollout.")
    ap.add_argument("--lesion-mode", choices=["decoding", "behavioral", "both"], default="decoding",
                    help="which instrument SELECTS the units to lesion for the behavioral-dependency gate. "
                         "'decoding' (default, BYTE-IDENTICAL to the prior code) = ridge-decoding importance "
                         "(most decodable units). 'behavioral' = policy/value(-or-predictor)-head saliency x "
                         "activation std (the units the behavior-producing read-head reads). 'both' = run BOTH "
                         "per faculty and also record their top-k Jaccard overlap — the Schøyen dissociation "
                         "measure (low overlap + only-behavioral-lesion-degrades => decodability is misleading, "
                         "e.g. Schøyen 2023: the high-decodability spatial units were causally DISPENSABLE).")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end self-test (small core, short, 1 seed)")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = args.seeds[:1]
        kw = dict(n_hidden=128, n_train=6000, n_probe=2500, n_behav=1200, n_cov=2500)
    else:
        kw = dict(n_hidden=args.n_hidden, n_train=args.n_train, n_probe=8000, n_behav=4000, n_cov=8000)

    per_seed = [run_seed(s, units=args.units, encoder=args.encoder, consolidation=args.consolidation,
                         grad_clip=args.grad_clip, grad_skip_factor=args.grad_skip_factor,
                         ema_momentum=args.ema_momentum, ema_warmup=args.ema_warmup,
                         pred_horizon=args.pred_horizon, nav_required=args.nav_required,
                         nav_dmin=args.nav_dmin, value_weight=args.value_weight, sr_weight=args.sr_weight,
                         aux_loc_weight=args.aux_loc_weight,
                         nav_shaping=args.nav_shaping, nav_curriculum=args.nav_curriculum,
                         nav_curriculum_frac=args.nav_curriculum_frac,
                         nav_curriculum_dmin_start=args.nav_curriculum_dmin_start,
                         si=args.si,
                         lesion_mode=args.lesion_mode, **kw)
                for s in args.seeds]
    agg = aggregate(per_seed)
    payload = {"battery": "fork_pcs_emergence", "units": args.units, "encoder": args.encoder,
               "consolidation": args.consolidation, "si": args.si, "pred_horizon": args.pred_horizon,
               "nav_required": args.nav_required, "nav_dmin": args.nav_dmin, "value_weight": args.value_weight,
               "sr_weight": args.sr_weight, "aux_loc_weight": args.aux_loc_weight,
               "nav_shaping": args.nav_shaping,
               "lesion_mode": args.lesion_mode,
               **({"nav_curriculum": True, "nav_curriculum_frac": args.nav_curriculum_frac,
                   "nav_curriculum_dmin_start": args.nav_curriculum_dmin_start} if args.nav_curriculum else {}),
               "grad_clip": args.grad_clip, "grad_skip_factor": args.grad_skip_factor,
               "ema_momentum": args.ema_momentum, "ema_warmup": args.ema_warmup,
               "pre_registered_gate": {
                   "presence_bars": PRESENCE_BAR, "floor_margin": FLOOR_MARGIN,
                   "behav_lesion_ratio": BEHAV_LESION_RATIO, "min_faculties": MIN_FACULTIES,
                   "core_lesion_collapse_frac": CORE_LESION_COLLAPSE, "curiosity_ratio": CURIOSITY_RATIO,
                   "seeds_required_frac": SEEDS_REQUIRED_FRAC},
               "per_seed": per_seed, "aggregate": agg}
    print("\n=== AGGREGATE ===")
    print(json.dumps(agg, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
