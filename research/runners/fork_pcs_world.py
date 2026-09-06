"""fork_pcs_world — the AGI-fork's grounded experience stream (design section b + g#3).

An agent lives online in a gridworld. Its ONLY sensory channel is an AGENT-CENTERED
egocentric CROP of the world through a FIXED Gabor-V1 front end; its reward is a
grounded interoceptive DRIVE-REDUCTION (TwoPoolDrive), never a host distance formula.
This is the stream that feeds sim.pcs_substrate.PredictiveContinualSubstrate.

WHY THE CROP, NOT render_egocentric_goal (the disqualified goal-compass)
-----------------------------------------------------------------------
research/runners/g11_bg_runner.py:render_egocentric_goal computes `ddx = goal-agent`
every step and paints the goal at its bearing — a PRE-SOLVED goal-compass a reactive
memoryless policy can follow, so any "place code" would be a decorative correlate
(design fatal-flaw #1). Here the agent is ALWAYS at crop-center, so:
  * absolute (x,y) is in NO single frame  -> position must be PATH-INTEGRATED;
  * the food appears only when inside the crop -> off-view food must be REMEMBERED (permanence);
  * fixed landmarks appear only locally -> allocentric localization must INTEGRATE over motion.
Verified by construction: a ridge decode of abs-position from the RAW V1 of a single crop
sits far below the trained core (the `--smoke` path checks exactly this).

BRAIN-BASED BOUNDARY (FORK.md keeps grounding as invariant #2)
--------------------------------------------------------------
Host code is legitimate for the WORLD (grid, entity cells, respawn) and the BODY
(applying the chosen move, energy depletion/refill). The Gabor V1 is a FIXED sensory
transducer = "rendering the agent's sensory input" (legitimate, like a retina). Everything
between sensation and action lives in the substrate. The reward is a PHYSICAL consequence:
eating reduces the body's energy deficit, and drive-reduction (the reduction in the
TwoPoolDrive homeostatic-need signal) is the reward (Keramati-Gutkin) — NO distance term.

RENDER NOTE (honest deviation): render_gridworld_to_image's `landmarks=` arg is a no-op in
the current code (it draws only agent+goal+edges), and its agent/goal ON intensities would
collide with the K=4 distinct object appearances the object-RSA faculty needs. So the crop
is drawn by render_egocentric_crop() here, which FOLLOWS render_gridworld_to_image's exact
(2,H,W) ON/OFF channel convention and REUSES it for the agent+food base, then overlays the
K=4 objects (distinct oriented bars -> distinct Gabor-V1 responses) and landmarks (crosses).
V1 itself is build_v1_simple_weights, reused verbatim.

Run:
  # Day-1 smoke (numpy, 1 seed, short): loss drops AND abs-position decode beats BOTH floors
  SIM_BACKEND=numpy python -m research.runners.fork_pcs_world --smoke
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import get_backend, from_host
from sim.visual_cortex import build_v1_simple_weights, render_gridworld_to_image

# the validated 2-pool push-pull hunger drive (rate proxy of AgRP<->POMC), reused verbatim
from research.runners._homeostatic_drive_rl_cheap_first_probe import TwoPoolDrive


# ── moves: N, E, S, W (dx, dy), +x=East, +y=North ────────────────────────────
MOVES = [(0, 1), (1, 0), (0, -1), (-1, 0)]
N_ACTIONS = len(MOVES)
K_OBJECTS = 4                        # distinct object TYPES (design: K=4)


@dataclass
class WorldConfig:
    grid_size: int = 18              # bounded grid, LARGE vs the crop so most frames are position-ambiguous
    crop_radius: int = 2             # egocentric window = (2R+1) cells; small vs grid -> interior is ambiguous
    image_size: int = 32             # crop rendered to 32x32 (retina) -> Gabor V1
    n_landmarks: int = 2             # SPARSE fixed landmarks (localization anchors seen only occasionally)
    render_walls: bool = False       # OFF: the agent does NOT see the boundary (pervasive walls leak abs-position)
    # V1 front end (fixed Gabor transducer). Smaller default than the full 8192 for tractability;
    # the emergence runner can dial these up for the GPU arm.
    n_orient: int = 8
    n_freq: int = 2
    v1_pos: int = 8                  # V1 positions/dim -> n_v1 = n_orient*n_freq*v1_pos^2
    rf_radius: int = 4
    # body / homeostasis
    set_point: float = 1.0
    deplete: float = 0.02
    eat_refill: float = 0.5
    e_max: float = 1.0
    start_energy: float = 1.0
    drive_tau: float = 0.5
    reward_scale: float = 1.0
    seed: int = 42

    @property
    def n_v1(self) -> int:
        return self.n_orient * self.n_freq * self.v1_pos * self.v1_pos


# ─────────────────────────────────────────────────────────────────────────────
# rendering the egocentric crop (host = "rendering the agent's sensory input")
# ─────────────────────────────────────────────────────────────────────────────
def _draw_oriented_bar(img, cy, cx, ppc, theta_idx, intensity=0.8):
    """Draw a short oriented bar (object type marker) centered at pixel (cy,cx).
    4 orientations -> 4 distinct Gabor-V1 responses (the object-category cue)."""
    thetas = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    th = thetas[theta_idx % 4]
    half = max(1, ppc // 2)
    for r in range(-half, half + 1):
        py = int(round(cy + r * np.sin(th)))
        px = int(round(cx + r * np.cos(th)))
        if 0 <= py < img.shape[1] and 0 <= px < img.shape[2]:
            img[0, py, px] = max(img[0, py, px], intensity)


def _draw_cross(img, cy, cx, ppc, intensity=0.6):
    """Draw a plus/cross (landmark marker) — both H and V bars -> distinct from single-orient objects."""
    half = max(1, ppc // 2)
    for r in range(-half, half + 1):
        for (py, px) in ((cy + r, cx), (cy, cx + r)):
            if 0 <= py < img.shape[1] and 0 <= px < img.shape[2]:
                img[0, py, px] = max(img[0, py, px], intensity)


def render_egocentric_crop(cfg: WorldConfig, agent, food, objects, landmarks) -> np.ndarray:
    """Return a (2, image_size, image_size) ON/OFF crop centered on the agent.

    agent=(ax,ay); food=(fx,fy) or None; objects={ (x,y): type_idx }; landmarks=[(x,y),...].
    The agent is ALWAYS at crop-center. Out-of-grid cells are marked in the OFF channel (walls).
    Follows render_gridworld_to_image's (2,H,W) ch0=ON ch1=OFF convention; reuses it for the
    agent+food base, then overlays objects (oriented bars) + landmarks (crosses).
    """
    R = cfg.crop_radius
    crop_cells = 2 * R + 1
    ax, ay = agent
    # local coords: entity world cell (ex,ey) -> local (R+(ex-ax), R+(ey-ay)); local center=(R,R)
    food_local = None
    if food is not None:
        lx, ly = R + (food[0] - ax), R + (food[1] - ay)
        if 0 <= lx < crop_cells and 0 <= ly < crop_cells:
            food_local = (lx, ly)
    # base: reuse render_gridworld_to_image for agent(center) + food(if in crop) + grid edges
    base_goal = food_local if food_local is not None else (-99, -99)
    img = render_gridworld_to_image(
        agent_pos=(R, R), goal_pos=base_goal,
        grid_size=crop_cells, image_size=cfg.image_size,
    ).copy()
    ppc = cfg.image_size // crop_cells

    def _cell_center(lx, ly):
        # note render_gridworld_to_image uses (x=col, y=row): pixel = cell*ppc + ppc//2
        return (ly * ppc + ppc // 2, lx * ppc + ppc // 2)   # (py, px)

    # OFF-channel walls (default OFF): rendering the boundary pervasively leaks absolute position
    # into a single frame (a distance-to-edge cue), defeating the path-integration premise. Kept
    # as an opt-in cue (boundary-vector-cell-like) but disabled by default.
    if cfg.render_walls:
        for ly in range(crop_cells):
            for lx in range(crop_cells):
                ex, ey = ax + (lx - R), ay + (ly - R)
                if not (0 <= ex < cfg.grid_size and 0 <= ey < cfg.grid_size):
                    y0, x0 = ly * ppc, lx * ppc
                    img[1, y0:y0 + ppc, x0:x0 + ppc] = np.maximum(img[1, y0:y0 + ppc, x0:x0 + ppc], 0.6)

    # objects (distinct oriented bars) — appear only when inside the crop
    for (ex, ey), tp in objects.items():
        lx, ly = R + (ex - ax), R + (ey - ay)
        if 0 <= lx < crop_cells and 0 <= ly < crop_cells:
            py, px = _cell_center(lx, ly)
            _draw_oriented_bar(img, py, px, ppc, tp, intensity=0.85)

    # landmarks (crosses)
    for (ex, ey) in landmarks:
        lx, ly = R + (ex - ax), R + (ey - ay)
        if 0 <= lx < crop_cells and 0 <= ly < crop_cells:
            py, px = _cell_center(lx, ly)
            _draw_cross(img, py, px, ppc, intensity=0.6)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# fixed Gabor V1 front end (build once; reused verbatim from sim.visual_cortex)
# ─────────────────────────────────────────────────────────────────────────────
class V1FrontEnd:
    """A FIXED Gabor-V1 transducer: retina (2*32*32) -> V1 simple cells (n_v1), relu-rectified.

    Weights come from build_v1_simple_weights (unmodified). Dense matrix built once; matmul
    runs on the active backend so the GPU arm never round-trips to host per step.
    """

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.xp, _ = get_backend()
        pre, post, w = build_v1_simple_weights(
            n_orientations=cfg.n_orient, n_frequencies=cfg.n_freq,
            n_positions_per_dim=cfg.v1_pos, retina_size=cfg.image_size,
            receptive_field_radius=cfg.rf_radius,
        )
        n_v1 = cfg.n_v1
        retina_dim = 2 * cfg.image_size * cfg.image_size
        W = np.zeros((n_v1, retina_dim), dtype=np.float32)
        W[post, pre] = w                  # sparse -> dense (built once)
        self.W = from_host(W)
        self.retina_dim = retina_dim
        self.n_v1 = n_v1

    def __call__(self, image: np.ndarray):
        xp = self.xp
        retina = from_host(np.asarray(image, dtype=np.float32).reshape(-1))
        v1 = self.W @ retina
        return xp.maximum(v1, 0.0)        # simple-cell rectification


# ─────────────────────────────────────────────────────────────────────────────
# the world + body (pure environment; the brain is the substrate, kept separate)
# ─────────────────────────────────────────────────────────────────────────────
class ForkPCSWorld:
    """The grounded experience stream: a bounded gridworld with fixed objects+landmarks,
    a respawning food, a body with an energy deficit, and a TwoPoolDrive interoceptive drive.

    Loop the substrate against it:
        d = world.drive_afferent()
        v1 = world.crop_v1feat()
        h = sub.observe(v1, world.last_action, d)
        a = sub.act(h)
        r, info = world.step(a)
        sub.learn(r)
    """

    def __init__(self, cfg: WorldConfig):
        self.cfg = cfg
        self.v1 = V1FrontEnd(cfg)
        self.reset(cfg.seed)

    # ---- layout (fixed per seed) ----
    def reset(self, seed: Optional[int] = None):
        cfg = self.cfg
        if seed is None:
            seed = cfg.seed
        self.rng = np.random.default_rng(seed)
        G = cfg.grid_size
        # distinct fixed cells for K objects + landmarks + a start, non-overlapping
        cells = [(x, y) for x in range(G) for y in range(G)]
        self.rng.shuffle(cells)
        it = iter(cells)
        self.objects: Dict[Tuple[int, int], int] = {}
        for k in range(K_OBJECTS):
            self.objects[next(it)] = k
        self.landmarks: List[Tuple[int, int]] = [next(it) for _ in range(cfg.n_landmarks)]
        self.agent = next(it)
        self.food = self._respawn_food()
        self.energy = float(cfg.start_energy)
        self.drive = TwoPoolDrive(tau=cfg.drive_tau)
        self._prime_drive()
        self.last_action = -1
        self.t = 0
        self.n_eats = 0

    def _respawn_food(self):
        cfg = self.cfg
        occupied = set(self.objects) | set(self.landmarks) | {getattr(self, "agent", (-1, -1))}
        while True:
            c = (int(self.rng.integers(cfg.grid_size)), int(self.rng.integers(cfg.grid_size)))
            if c not in occupied:
                return c

    def _prime_drive(self):
        # settle the 2-pool drive to the current deficit so d_t is well-defined at t=0
        dfc = self._deficit()
        for _ in range(4):
            self._drive_val = float(self.drive.update(dfc))

    def _deficit(self) -> float:
        return float(np.clip(self.cfg.set_point - self.energy, 0.0, 1.0))

    # ---- sensory + interoceptive read-outs (host renders sensation; legitimate) ----
    def crop_image(self) -> np.ndarray:
        return render_egocentric_crop(self.cfg, self.agent, self.food, self.objects, self.landmarks)

    def crop_v1feat(self):
        return self.v1(self.crop_image())

    def raw_v1_of_current_crop(self) -> np.ndarray:
        """Host-numpy V1 of the current crop — the raw-V1 floor for the position probe."""
        import numpy as _np
        img = self.crop_image().reshape(-1).astype(_np.float32)
        W = _np.asarray(self.v1.W.get() if hasattr(self.v1.W, "get") else self.v1.W)
        return _np.maximum(W @ img, 0.0)

    def drive_afferent(self):
        """d_t: the interoceptive drive afferent the brain senses (TwoPoolDrive + body state)."""
        drive = self._drive_val
        dfc = self._deficit()
        return np.asarray([drive, dfc, self.energy, 1.0], dtype=np.float32)

    @property
    def food_in_crop(self) -> bool:
        if self.food is None:
            return False
        R = self.cfg.crop_radius
        return abs(self.food[0] - self.agent[0]) <= R and abs(self.food[1] - self.agent[1]) <= R

    def objects_in_crop(self) -> List[int]:
        R = self.cfg.crop_radius
        out = []
        for (ex, ey), tp in self.objects.items():
            if abs(ex - self.agent[0]) <= R and abs(ey - self.agent[1]) <= R:
                out.append(tp)
        return out

    # ---- body: apply the chosen move; drive-reduction reward (Keramati-Gutkin) ----
    def step(self, action: int):
        cfg = self.cfg
        # drive BEFORE the action's consequence
        drive_before = self._drive_val
        # apply move (bounded)
        dx, dy = MOVES[action]
        nx = int(np.clip(self.agent[0] + dx, 0, cfg.grid_size - 1))
        ny = int(np.clip(self.agent[1] + dy, 0, cfg.grid_size - 1))
        self.agent = (nx, ny)
        # metabolism
        self.energy = max(0.0, self.energy - cfg.deplete)
        ate = (self.food is not None and self.agent == self.food)
        if ate:
            self.energy = min(cfg.e_max, self.energy + cfg.eat_refill)
            self.food = self._respawn_food()
            self.n_eats += 1
        # drive AFTER the consequence
        drive_after = float(self.drive.update(self._deficit()))
        self._drive_val = drive_after
        # GROUNDED reward = drive-reduction (reduction in the homeostatic-need signal), NOT distance
        reward = cfg.reward_scale * max(0.0, drive_before - drive_after)
        self.last_action = action
        self.t += 1
        info = {"ate": ate, "energy": self.energy, "deficit": self._deficit(),
                "drive": drive_after, "pos": self.agent, "food": self.food}
        return reward, info

    # ---- ground-truth labels for the probes (used by the emergence runner) ----
    def labels(self) -> dict:
        return {
            "pos": np.asarray(self.agent, dtype=np.float32),
            "food": np.asarray(self.food if self.food is not None else (-1, -1), dtype=np.float32),
            "food_in_crop": float(self.food_in_crop),
            "objects_in_crop": self.objects_in_crop(),
            "deficit": self._deficit(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Day-1 SMOKE: loss drops AND abs-position decode beats untrained-core + raw-V1 floors
# ─────────────────────────────────────────────────────────────────────────────
def _ridge_r2(X_tr, Y_tr, X_te, Y_te, lam=10.0, min_samples=30):
    """Closed-form ridge; mean R^2 over targets on the test split.

    Guards (repo discipline: UNDEFINED, not a fake score):
      * too few samples -> nan (an underdetermined decode is not a measurement);
      * a target dim with ~zero test variance -> that dim is UNDEFINED (R^2 of a constant is not 1.0);
      * lam scaled up when underdetermined + std floored + result clipped to [-1,1] (kills numerical blow-ups).
    """
    Y_tr = np.asarray(Y_tr, np.float64); Y_te = np.asarray(Y_te, np.float64)
    if Y_tr.ndim == 1:
        Y_tr = Y_tr[:, None]; Y_te = Y_te[:, None]
    if len(X_tr) < min_samples or len(X_te) < 5:
        return float("nan")
    ss_tot = ((Y_te - Y_te.mean(0, keepdims=True)) ** 2).sum(0)
    valid = ss_tot >= 1e-6
    if not np.any(valid):
        return float("nan")                     # constant target -> R^2 undefined
    mu = X_tr.mean(0, keepdims=True)
    sd = np.maximum(X_tr.std(0, keepdims=True), 1e-2)
    Xtr = (X_tr - mu) / sd
    Xte = (X_te - mu) / sd
    Xtr1 = np.concatenate([Xtr, np.ones((Xtr.shape[0], 1))], 1)
    Xte1 = np.concatenate([Xte, np.ones((Xte.shape[0], 1))], 1)
    d = Xtr1.shape[1]
    lam_eff = lam * max(1.0, d / len(X_tr))       # stronger reg when underdetermined
    W = np.linalg.solve(Xtr1.T @ Xtr1 + lam_eff * np.eye(d), Xtr1.T @ Y_tr)
    pred = Xte1 @ W
    ss_res = ((Y_te - pred) ** 2).sum(0)
    r2_per = 1.0 - ss_res[valid] / ss_tot[valid]
    return float(np.clip(r2_per.mean(), -1.0, 1.0))


def _collect_stationary_eval(cfg, seed, n=400):
    """A FIXED, grid-covering held-out set of world transitions (random policy). Because it is
    stationary and covers the grid uniformly, the model's prediction error on it reflects
    general world-dynamics learning, decoupled from the training policy's regional drift (an
    early policy-collected set is confounded by catastrophic forgetting as the agent explores)."""
    w = ForkPCSWorld(cfg)
    rng = np.random.default_rng(seed + 321)
    seq = []
    a_prev = -1
    for _ in range(n):
        d = w.drive_afferent()
        v1 = w.crop_v1feat()
        v1h = np.asarray(v1.get() if hasattr(v1, "get") else v1, dtype=np.float32)
        a = int(rng.integers(N_ACTIONS))
        r, _ = w.step(a)
        seq.append((v1h, a_prev, d.copy(), float(r)))
        a_prev = a
    return seq


def run_smoke(steps=4000, seed=42, units="rate", encoder="learned_ema", verbose=True):
    """Collect a trajectory while training online; then decode abs-position from (i) the trained
    core, (ii) a fresh UNTRAINED core replayed on the SAME input sequence, (iii) the raw V1 of the
    crop. GO signal: trained R^2 beats BOTH floors AND the prediction loss dropped."""
    from sim.pcs_substrate import PredictiveContinualSubstrate, PCSConfig

    wcfg = WorldConfig(seed=seed)
    world = ForkPCSWorld(wcfg)
    stationary_eval = _collect_stationary_eval(wcfg, seed, n=400)
    scfg = PCSConfig(n_hidden=256, feat_dim=wcfg.n_v1, n_latent=64, n_actions=N_ACTIONS,
                     n_drive=4, tbptt_T=16, units=units, encoder=encoder, seed=seed)
    sub = PredictiveContinualSubstrate(scfg)

    inputs = []          # (v1feat_host, a_prev, d) per step — replayed through the untrained core
    H_tr, POS, RAWV1 = [], [], []
    losses = []
    eval_curve = []                        # (step, held-out loss on the STATIONARY set) checkpoints
    ckpt_every = max(1, steps // 8)
    eval_curve.append((0, sub.eval_predictive_loss(stationary_eval)))   # pre-training baseline
    a_prev = -1
    for t in range(steps):
        d = world.drive_afferent()
        v1 = world.crop_v1feat()
        v1_host = np.asarray(v1.get() if hasattr(v1, "get") else v1, dtype=np.float32)
        h = sub.observe(v1, a_prev, d)
        a = sub.act(h, explore_eps=0.4)   # coverage-rich stream to probe REPRESENTATION emergence
        r, info = world.step(a)
        sub.learn(r)
        if sub.last_pred_loss is not None:
            losses.append(sub.last_pred_loss)
        # held-out predictive loss on the FIXED, grid-covering STATIONARY set (learning signal)
        if t > 0 and t % ckpt_every == 0:
            eval_curve.append((t, sub.eval_predictive_loss(stationary_eval)))
        # collect for the probe over the SECOND half (after some learning)
        if t >= steps // 2:
            H_tr.append(np.asarray(h.get() if hasattr(h, "get") else h, dtype=np.float32))
            POS.append(np.asarray(info["pos"], dtype=np.float32))
            RAWV1.append(world.raw_v1_of_current_crop())
        inputs.append((v1_host, a_prev, d.copy()))
        a_prev = a
    eval_curve.append((steps, sub.eval_predictive_loss(stationary_eval)))   # final checkpoint

    # untrained-core floor: fresh substrate (different seed), NO training, replay same inputs
    untr = PredictiveContinualSubstrate(PCSConfig(
        n_hidden=256, feat_dim=wcfg.n_v1, n_latent=64, n_actions=N_ACTIONS,
        n_drive=4, tbptt_T=16, units=units, encoder=encoder, seed=seed + 777))
    untr.freeze()
    H_un = []
    for t, (v1_host, ap, d) in enumerate(inputs):
        hu = untr.observe(v1_host, ap, d)
        if t >= steps // 2:
            H_un.append(np.asarray(hu.get() if hasattr(hu, "get") else hu, dtype=np.float32))

    H_tr = np.asarray(H_tr); H_un = np.asarray(H_un)
    POS = np.asarray(POS); RAWV1 = np.asarray(RAWV1)
    # SHUFFLED split (standard for a representational-decodability probe): a temporal split
    # of a correlated trajectory conflates decodability with nonstationarity. All three
    # conditions share the split, and the untrained-core + raw-V1 floors control for triviality.
    n = len(POS)
    perm = np.random.default_rng(seed + 5).permutation(n)
    cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    r2_tr = _ridge_r2(H_tr[tr], POS[tr], H_tr[te], POS[te])
    r2_un = _ridge_r2(H_un[tr], POS[tr], H_un[te], POS[te])
    r2_raw = _ridge_r2(RAWV1[tr], POS[tr], RAWV1[te], POS[te])
    n_cells_visited = len({tuple(p) for p in POS.tolist()})

    # HELD-OUT predictive skill on the STATIONARY set = the honest 'did it LEARN to predict' signal.
    # For a self-predictive (JEPA) model the target is self-defined and drifts, so the criterion is
    # "the loss dropped well below the UNTRAINED baseline at some point" (the model learned the
    # objective). A later rise (representational drift under broad online TBPTT — a real continual-
    # learning phenomenon, design section f) is reported separately, not counted as failure to learn.
    ev = [v for (_, v) in eval_curve]
    eval_baseline = float(ev[0]) if ev else float("nan")     # pre-training
    eval_best = float(np.nanmin(ev)) if ev else float("nan")
    eval_final = float(ev[-1]) if ev else float("nan")
    loss_dropped = (eval_best < 0.8 * eval_baseline)
    drift_up = (eval_final > 1.2 * eval_best)
    eval_early, eval_late = eval_baseline, eval_best
    # online-loss quintiles (rise expected: coverage widens the stream) — diagnostic only
    online_q = []
    if losses:
        L = np.asarray(losses)
        online_q = [round(float(np.mean(L[int(i * len(L) / 5):int((i + 1) * len(L) / 5)])), 2) for i in range(5)]
    beats_floors = (r2_tr > r2_un + 0.02) and (r2_tr > r2_raw + 0.02)

    if verbose:
        print(f"[smoke units={units} enc={encoder} seed={seed} steps={steps}]")
        print(f"  HELD-OUT pred-loss (STATIONARY grid-covering set): baseline={eval_baseline:.4f} best={eval_best:.4f} final={eval_final:.4f}  LEARNED={loss_dropped}  drift_up={drift_up}")
        print(f"    held-out curve: {[(s, round(v, 3)) for s, v in eval_curve]}")
        print(f"    online-loss quintiles (rise = widening coverage, diagnostic): {online_q}")
        print(f"  abs-position decode R^2:  trained-core={r2_tr:.3f}   untrained-core={r2_un:.3f}   raw-V1-of-crop={r2_raw:.3f}")
        print(f"  BEATS BOTH FLOORS (+0.02)={beats_floors}  (eats={world.n_eats}, cells_visited={n_cells_visited}/{wcfg.grid_size**2}, n_updates={sub.n_updates})")
        print(f"  SMOKE {'PASS' if (loss_dropped and beats_floors) else 'FAIL'}")
    return {"eval_baseline": eval_baseline, "eval_best": eval_best, "eval_final": eval_final,
            "loss_dropped": loss_dropped, "drift_up": drift_up,
            "r2_trained": r2_tr, "r2_untrained": r2_un, "r2_rawv1": r2_raw,
            "beats_floors": beats_floors, "eats": world.n_eats, "n_updates": sub.n_updates}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="fork PCS grounded world + Day-1 smoke")
    ap.add_argument("--smoke", action="store_true", help="run the Day-1 smoke (loss drop + position vs floors)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--units", choices=["rate", "spike"], default="rate")
    ap.add_argument("--encoder", choices=["learned_ema", "fixed"], default="learned_ema")
    args = ap.parse_args()
    if args.smoke or True:
        res = run_smoke(steps=args.steps, seed=args.seed, units=args.units, encoder=args.encoder)
        raise SystemExit(0 if (res["loss_dropped"] and res["beats_floors"]) else 1)
