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
            a_prev_start=-1, log_loss=False):
    """Drive the substrate against the world. Returns traces (+ leaves world/sub advanced).

    train=False freezes the predictive update (probe/behavioral rollouts). lesion_mask sets a
    hidden-unit lesion for the whole rollout. collect=True records per-step h/labels/rawv1.
    log_loss=True records a downsampled training-loss curve (train-stability visibility).
    """
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
    for t in range(n_steps):
        # capture ALL pre-step sensory info aligned to the observation the substrate acts on
        d = world.drive_afferent()
        v1 = world.crop_v1feat()
        v1_host = np.asarray(v1.get() if hasattr(v1, "get") else v1, dtype=np.float32)
        pos_before = world.agent
        food_before = world.food
        fic = world.food_in_crop
        obj_in = world.objects_in_crop()
        h = sub.observe(v1, a_prev, d)
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
             nav_shaping=0.0, lesion_mode="decoding", verbose=True):
    t0 = time.time()
    wcfg = WorldConfig(seed=seed, nav_required=nav_required, nav_dmin=nav_dmin, nav_shaping=nav_shaping)
    world = ForkPCSWorld(wcfg)
    scfg = PCSConfig(n_hidden=n_hidden, feat_dim=wcfg.n_v1, n_latent=n_latent, n_actions=N_ACTIONS,
                     n_drive=4, tbptt_T=18, units=units, encoder=encoder, seed=seed,
                     consolidation=consolidation, grad_clip=grad_clip, grad_skip_factor=grad_skip_factor,
                     ema_rate=ema_momentum, ema_warmup_updates=ema_warmup, pred_horizon=pred_horizon,
                     value_weight=value_weight, sr_weight=sr_weight)
    sub = PredictiveContinualSubstrate(scfg)

    # ---- 1. TRAIN online with the curiosity policy (small explore for early coverage) ----
    train_out = rollout(world, sub, n_train, train=True, explore_eps=0.2, log_loss=True)

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
        "sr_weight": sr_weight,
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
        "train_loss_curve": train_out.get("loss_curve", []),
        "intact_behavior": {k2: _f(v2) for k2, v2 in intact.items() if isinstance(v2, (int, float))},
        "SEED_GO": bool(seed_go),
        "elapsed_s": round(time.time() - t0, 1),
    }
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
# aggregate + main
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(per_seed):
    n = len(per_seed)
    n_go = sum(1 for r in per_seed if r["SEED_GO"])
    n_cov = sum(1 for r in per_seed if r["coverage"].get("pass"))
    faculty_lb_counts = {f: sum(1 for r in per_seed if r["behavioral_dependency"].get(f, {}).get("load_bearing"))
                         for f in PRESENCE_BAR}
    emergence_go = (n_go >= int(np.ceil(SEEDS_REQUIRED_FRAC * n))) and (n_cov >= int(np.ceil(SEEDS_REQUIRED_FRAC * n)))
    return {"n_seeds": n, "n_seed_go": n_go, "n_coverage_pass": n_cov,
            "faculty_load_bearing_counts": faculty_lb_counts,
            "seeds_required": int(np.ceil(SEEDS_REQUIRED_FRAC * n)),
            "EMERGENCE_GO": bool(emergence_go)}


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
                         "so the task-required place code can actually bind (PBS is policy-invariant).")
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
                         nav_shaping=args.nav_shaping, lesion_mode=args.lesion_mode, **kw)
                for s in args.seeds]
    agg = aggregate(per_seed)
    payload = {"battery": "fork_pcs_emergence", "units": args.units, "encoder": args.encoder,
               "consolidation": args.consolidation, "pred_horizon": args.pred_horizon,
               "nav_required": args.nav_required, "nav_dmin": args.nav_dmin, "value_weight": args.value_weight,
               "sr_weight": args.sr_weight,
               "nav_shaping": args.nav_shaping,
               "lesion_mode": args.lesion_mode,
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
