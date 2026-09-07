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
    # ── NAV-REQUIRED homing task (the fork's 3rd move; default OFF -> byte-identical to the first/second-move
    # world: food respawns at a RANDOM cell each eat, so the goal is findable reactively/by wandering and a
    # persistent place code is not REQUIRED — the 1st-move finding). When ON the food sits at a FIXED larder
    # cell (per seed), and each time it is eaten the AGENT is displaced to a random cell >= nav_dmin Manhattan
    # from the larder (a trial reset) while the food REGROWS at the same larder. So the goal is a stable,
    # REMEMBERED, out-of-view location the agent must NAVIGATE BACK to from a novel start — a reactive/
    # memoryless policy cannot (the larder is invisible from afar; only landmarks localize), so reaching it
    # (and thus reducing the interoceptive drive = reward) REQUIRES a persistent, path-integrated place code.
    # This makes position LOAD-BEARING ON REWARD. Displacing after each eat forbids the degenerate "sit on the
    # larder" solution (which needs no position). Teleport = a world/trial event (legit host: world state),
    # standard in animal navigation trials; relocalization-after-displacement is itself a real place-cell
    # function. Pair with sim.pcs_substrate value_weight>0 so the reward/value gradient shapes the shared core.
    nav_required: bool = False
    nav_dmin: int = 6                # min Manhattan distance the post-eat agent-respawn keeps from the larder
    # ── 4th move: POTENTIAL-BASED APPROACH SHAPING (Ng-Harada-Russell 1999) to make homing LEARNABLE.
    # The 3rd move failed not because task-required-position is wrong but because the reward was too SPARSE
    # (larder reached ~0.1-0.5% of steps -> REINFORCE never learned to home -> the requirement never bound).
    # PBS densifies the reward with r += nav_shaping*(gamma*Phi(s') - Phi(s)), Phi(s) = -dist(agent,larder)/dnorm.
    # PBS is provably POLICY-INVARIANT (it changes learning SPEED, not the optimal policy), so it does NOT
    # hand the agent the goal direction: the scalar shaping reward is a post-hoc CONSEQUENCE (never a policy
    # input), so to exploit it the policy must still infer which action approaches the out-of-view larder from
    # its OWN state -> the place code stays load-bearing; shaping only makes the credit-assignment tractable.
    # Applied on the pre-teleport post-move position so an EAT step (which teleports the agent far) scores the
    # approach it EARNED, not the trial-reset displacement. Default 0.0 -> byte-identical to the 3rd-move world.
    nav_shaping: float = 0.0         # PBS coefficient (0 = OFF); only active when nav_required
    nav_shaping_gamma: float = 0.99  # PBS discount (telescoping term; keeps total shaping policy-invariant)
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
        # NAV-REQUIRED (3rd move): a FIXED larder cell (per seed) the food always returns to; the agent must
        # navigate back to this remembered, out-of-view location. OFF path unchanged (larder=None, random food).
        if cfg.nav_required:
            self.larder: Optional[Tuple[int, int]] = next(it)      # extra draw ONLY in nav mode -> OFF unaffected
            self.food = self.larder
        else:
            self.larder = None
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

    def _respawn_agent_far(self, target, dmin):
        """NAV-REQUIRED trial reset: place the agent at a random cell >= dmin Manhattan from `target`
        (the larder), not on an object/landmark — so the remembered larder is out of view and the agent
        must re-localize + navigate back. Falls back to the farthest sampled cell if dmin is unreachable."""
        cfg = self.cfg
        occupied = set(self.objects) | set(self.landmarks)
        best = None; best_d = -1
        for _ in range(200):
            c = (int(self.rng.integers(cfg.grid_size)), int(self.rng.integers(cfg.grid_size)))
            if c in occupied or c == target:
                continue
            d = abs(c[0] - target[0]) + abs(c[1] - target[1])
            if d >= dmin:
                return c
            if d > best_d:
                best_d, best = d, c
        return best if best is not None else self.agent

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
        px, py = self.agent                       # pre-move position (for PBS shaping)
        dx, dy = MOVES[action]
        nx = int(np.clip(self.agent[0] + dx, 0, cfg.grid_size - 1))
        ny = int(np.clip(self.agent[1] + dy, 0, cfg.grid_size - 1))
        self.agent = (nx, ny)
        # 4th-move POTENTIAL-BASED approach shaping (Phi=-dist/dnorm), computed on the PRE-teleport post-move
        # position so an eat step scores the approach it EARNED, not the trial-reset displacement. 0 when OFF ->
        # the reward stays the pure drive-reduction of the 3rd-move world (byte-identical).
        shaping = 0.0
        if cfg.nav_required and cfg.nav_shaping > 0.0 and self.larder is not None:
            dnorm = 2.0 * (cfg.grid_size - 1)
            lx, ly = self.larder
            phi_pre = -(abs(px - lx) + abs(py - ly)) / dnorm
            phi_post = -(abs(nx - lx) + abs(ny - ly)) / dnorm
            shaping = cfg.nav_shaping * (cfg.nav_shaping_gamma * phi_post - phi_pre)
        # metabolism
        self.energy = max(0.0, self.energy - cfg.deplete)
        ate = (self.food is not None and self.agent == self.food)
        if ate:
            self.energy = min(cfg.e_max, self.energy + cfg.eat_refill)
            if cfg.nav_required:
                # food REGROWS at the fixed larder; the agent is displaced far -> must navigate back (trial reset)
                self.food = self.larder
                self.agent = self._respawn_agent_far(self.larder, cfg.nav_dmin)
            else:
                self.food = self._respawn_food()
            self.n_eats += 1
        # drive AFTER the consequence
        drive_after = float(self.drive.update(self._deficit()))
        self._drive_val = drive_after
        # GROUNDED reward = drive-reduction (reduction in the homeostatic-need signal), NOT distance;
        # + the PBS approach-shaping term (0 unless nav_shaping>0). PBS is policy-invariant, so it only
        # densifies the LEARNING signal (it does not change which policy is optimal, i.e. no goal-compass cheat).
        reward = cfg.reward_scale * max(0.0, drive_before - drive_after) + shaping
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


def run_smoke(steps=4000, seed=42, units="rate", encoder="learned_ema", consolidation=False, verbose=True):
    """Collect a trajectory while training online; then decode abs-position from (i) the trained
    core, (ii) a fresh UNTRAINED core replayed on the SAME input sequence, (iii) the raw V1 of the
    crop. GO signal: trained R^2 beats BOTH floors AND the prediction loss dropped."""
    from sim.pcs_substrate import PredictiveContinualSubstrate, PCSConfig

    wcfg = WorldConfig(seed=seed)
    world = ForkPCSWorld(wcfg)
    stationary_eval = _collect_stationary_eval(wcfg, seed, n=400)
    scfg = PCSConfig(n_hidden=256, feat_dim=wcfg.n_v1, n_latent=64, n_actions=N_ACTIONS,
                     n_drive=4, tbptt_T=16, units=units, encoder=encoder, seed=seed,
                     consolidation=consolidation)
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


def _probe_place_decode(sub, world, seed, n_probe, n_hidden, n_latent, units, encoder, feat_dim):
    """Freeze `sub`, run an exploratory probe, and decode abs-position from (i) the trained core,
    (ii) a fresh UNTRAINED core replayed on the SAME inputs, (iii) raw V1. Returns the three R^2.
    Advances world/sub state (sub is frozen -> no weight change; used symmetrically across arms)."""
    from sim.pcs_substrate import PredictiveContinualSubstrate, PCSConfig
    sub.freeze()
    H, POS, RAW, INSEQ = [], [], [], []
    a_prev = -1
    for _ in range(n_probe):
        d = world.drive_afferent(); v1 = world.crop_v1feat()
        v1h = np.asarray(v1.get() if hasattr(v1, "get") else v1, dtype=np.float32)
        pos = world.agent
        h = sub.observe(v1, a_prev, d)
        a = sub.act(h, explore_eps=0.4)
        world.step(a)
        H.append(np.asarray(h.get() if hasattr(h, "get") else h, dtype=np.float32))
        POS.append(np.asarray(pos, dtype=np.float32)); RAW.append(v1h)
        INSEQ.append((v1h, a_prev, d.copy())); a_prev = a
    untr = PredictiveContinualSubstrate(PCSConfig(
        n_hidden=n_hidden, feat_dim=feat_dim, n_latent=n_latent, n_actions=N_ACTIONS,
        n_drive=4, units=units, encoder=encoder, seed=seed + 777))
    untr.freeze()
    H_un = []
    for (v1h, ap, d) in INSEQ:
        hu = untr.observe(v1h, ap, d)
        H_un.append(np.asarray(hu.get() if hasattr(hu, "get") else hu, dtype=np.float32))
    H = np.asarray(H); H_un = np.asarray(H_un); POS = np.asarray(POS); RAW = np.asarray(RAW)
    n = len(POS); perm = np.random.default_rng(seed + 5).permutation(n); cut = int(0.7 * n)
    tr, te = perm[:cut], perm[cut:]
    sub.unfreeze()
    return {"trained": _ridge_r2(H[tr], POS[tr], H[te], POS[te]),
            "untrained": _ridge_r2(H_un[tr], POS[tr], H_un[te], POS[te]),
            "rawv1": _ridge_r2(RAW[tr], POS[tr], RAW[te], POS[te])}


def run_consolidation_ab(seed=42, steps=60000, n_hidden=128, n_latent=64, n_probe=1500,
                         units="rate", encoder="learned_ema", verbose=True):
    """OFF-vs-ON demonstration at a scaled-down LONG horizon: online TBPTT alone (OFF) OVERWRITES the
    place code over the horizon (falls toward/below the untrained floor); the consolidation companion
    (ON) RETAINS it (stays clearly above untrained). Reports place-decode-vs-untrained at an early and a
    late checkpoint for both arms, plus held-out predictive-loss drift. Honest: if ON does not retain,
    it says so."""
    from sim.pcs_substrate import PredictiveContinualSubstrate, PCSConfig
    checkpoints = sorted(set([max(1, steps // 5), steps // 2, steps]))   # early peak, mid, late
    arms = {}
    for consol in (False, True):
        wcfg = WorldConfig(seed=seed)
        world = ForkPCSWorld(wcfg)
        stat_eval = _collect_stationary_eval(wcfg, seed, n=400)
        scfg = PCSConfig(n_hidden=n_hidden, feat_dim=wcfg.n_v1, n_latent=n_latent, n_actions=N_ACTIONS,
                         n_drive=4, tbptt_T=16, units=units, encoder=encoder, seed=seed,
                         consolidation=consol)
        sub = PredictiveContinualSubstrate(scfg)
        traj = []
        done = 0
        a_prev = -1
        for ckpt in checkpoints:
            for _ in range(ckpt - done):
                d = world.drive_afferent(); v1 = world.crop_v1feat()
                h = sub.observe(v1, a_prev, d)
                a = sub.act(h, explore_eps=0.4)
                r, _ = world.step(a); sub.learn(r); a_prev = a
            done = ckpt
            place = _probe_place_decode(sub, world, seed, n_probe, n_hidden, n_latent, units, encoder, wcfg.n_v1)
            held = sub.eval_predictive_loss(stat_eval)
            traj.append({"step": ckpt, **place, "held_out": round(float(held), 4),
                         "n_replay": int(sub.n_replay_updates)})
        arms["ON" if consol else "OFF"] = traj

    off_last, on_last = arms["OFF"][-1], arms["ON"][-1]
    retention_ok = on_last["trained"] > on_last["untrained"] + 0.05
    drift_shown = off_last["trained"] < on_last["trained"] - 0.05
    off_degraded = off_last["trained"] <= off_last["untrained"] + 0.05
    passed = retention_ok and drift_shown
    if verbose:
        print(f"[consolidation A/B  seed={seed} units={units} steps={steps} n_hidden={n_hidden}]")
        for arm in ("OFF", "ON"):
            print(f"  {arm}:")
            for c in arms[arm]:
                print(f"    step={c['step']:>7}  place(trained)={c['trained']:+.3f}  untrained={c['untrained']:+.3f}"
                      f"  rawV1={c['rawv1']:+.3f}  held_out={c['held_out']:.3f}  n_replay={c['n_replay']}")
        print(f"  --> OFF place@end={off_last['trained']:+.3f} (untrained {off_last['untrained']:+.3f}); "
              f"ON place@end={on_last['trained']:+.3f} (untrained {on_last['untrained']:+.3f})")
        print(f"  retention_ok(ON above untrained)={retention_ok}  drift_shown(OFF below ON)={drift_shown}  "
              f"OFF_degraded_to_floor={off_degraded}")
        print(f"  CONSOLIDATION A/B {'PASS' if passed else 'INCONCLUSIVE'}")
    return {"OFF": arms["OFF"], "ON": arms["ON"], "retention_ok": retention_ok,
            "drift_shown": drift_shown, "off_degraded": off_degraded, "passed": passed}


def _train_and_track(sub, world, stationary_eval, steps, hb_every):
    """Train `steps` online, tracking spike diagnostics: max online (training) loss, max held-out loss
    on the stationary set, max pre-clip grad-norm, and #updates skipped by the spike guard."""
    max_online = 0.0
    heldout = []
    a_prev = -1
    for t in range(steps):
        d = world.drive_afferent(); v1 = world.crop_v1feat()
        h = sub.observe(v1, a_prev, d)
        a = sub.act(h, explore_eps=0.4)
        r, _ = world.step(a); sub.learn(r); a_prev = a
        if sub.last_pred_loss is not None and sub.last_pred_loss > max_online:
            max_online = float(sub.last_pred_loss)
        if t > 0 and t % hb_every == 0:
            heldout.append(round(float(sub.eval_predictive_loss(stationary_eval)), 3))
    heldout.append(round(float(sub.eval_predictive_loss(stationary_eval)), 3))
    return {"max_online_loss": round(max_online, 3),
            "max_heldout_loss": round(float(np.max(heldout)), 3),
            "final_heldout_loss": heldout[-1],
            "max_grad_norm": round(float(sub.max_grad_norm), 3),
            "n_skipped": int(sub.n_skipped), "heldout_curve": heldout}


def run_stability_ab(seed=42, steps=40000, n_hidden=256, n_latent=64, units="rate",
                     encoder="learned_ema", verbose=True):
    """Stabilization A/B: UNSTABLE (grad_clip=5.0, skip=0 — the current control that produced the
    existing 6-seed artifacts) vs STABLE (grad_clip=1.0 + spike-skip=20 — the new default). Reports
    the max online/held-out loss and max grad-norm for each, plus end place-decode-vs-untrained.
    PASS if the STABLE arm's loss spikes are gone (max loss much lower) AND it retains the place code."""
    from sim.pcs_substrate import PredictiveContinualSubstrate, PCSConfig
    configs = {"UNSTABLE(clip5,noskip)": dict(grad_clip=5.0, grad_skip_factor=0.0),
               "STABLE(clip1,skipx8)": dict(grad_clip=1.0, grad_skip_factor=8.0)}
    out = {}
    for name, gk in configs.items():
        wcfg = WorldConfig(seed=seed)
        world = ForkPCSWorld(wcfg)
        stat_eval = _collect_stationary_eval(wcfg, seed, n=400)
        scfg = PCSConfig(n_hidden=n_hidden, feat_dim=wcfg.n_v1, n_latent=n_latent, n_actions=N_ACTIONS,
                         n_drive=4, tbptt_T=16, units=units, encoder=encoder, seed=seed, **gk)
        sub = PredictiveContinualSubstrate(scfg)
        tr = _train_and_track(sub, world, stat_eval, steps, hb_every=max(1, steps // 20))
        place = _probe_place_decode(sub, world, seed, 1500, n_hidden, n_latent, units, encoder, wcfg.n_v1)
        out[name] = {**tr, "place_trained": round(place["trained"], 3),
                     "place_untrained": round(place["untrained"], 3)}
    u = out["UNSTABLE(clip5,noskip)"]; s = out["STABLE(clip1,skipx8)"]
    spikes_tamed = s["max_heldout_loss"] < 0.5 * u["max_heldout_loss"]
    retains = s["place_trained"] > s["place_untrained"] + 0.05
    passed = spikes_tamed and retains
    if verbose:
        print(f"[stability A/B  seed={seed} units={units} steps={steps} n_hidden={n_hidden}]")
        for name in configs:
            c = out[name]
            print(f"  {name:22}  max_online={c['max_online_loss']:>9}  max_heldout={c['max_heldout_loss']:>9}"
                  f"  final_heldout={c['final_heldout_loss']:>8}  max_grad_norm={c['max_grad_norm']:>9}"
                  f"  n_skipped={c['n_skipped']:>4}  place={c['place_trained']:+.3f}(untr {c['place_untrained']:+.3f})")
        print(f"    UNSTABLE heldout curve: {u['heldout_curve']}")
        print(f"    STABLE   heldout curve: {s['heldout_curve']}")
        print(f"  spikes_tamed(STABLE max_heldout < 0.5x UNSTABLE)={spikes_tamed}  retains(place>untrained)={retains}")
        print(f"  STABILIZATION A/B {'PASS' if passed else 'INCONCLUSIVE'}")
    return {"unstable": u, "stable": s, "spikes_tamed": spikes_tamed, "retains": retains, "passed": passed}


def run_ema_ab(seed=42, steps=50000, n_hidden=256, n_latent=64, units="rate",
               ema_values=(0.99, 0.999, 0.9999), new_default=0.9999, verbose=True):
    """EMA-target A/B: the 200k failure was TARGET-driven (grad clip/skip did nothing). Slower EMA =
    a target that can't jump. Reports max online/held-out loss + place-decode for each EMA momentum.
    PASS if the new default (0.9999) tames the held-out spikes (< 0.5x the old 0.99 control)."""
    from sim.pcs_substrate import PredictiveContinualSubstrate, PCSConfig
    out = {}
    for ema in ema_values:
        wcfg = WorldConfig(seed=seed); world = ForkPCSWorld(wcfg)
        stat = _collect_stationary_eval(wcfg, seed, n=400)
        scfg = PCSConfig(n_hidden=n_hidden, feat_dim=wcfg.n_v1, n_latent=n_latent, n_actions=N_ACTIONS,
                         n_drive=4, tbptt_T=16, units=units, encoder="learned_ema", seed=seed,
                         ema_rate=ema, grad_clip=1.0, grad_skip_factor=8.0)
        sub = PredictiveContinualSubstrate(scfg)
        tr = _train_and_track(sub, world, stat, steps, hb_every=max(1, steps // 20))
        place = _probe_place_decode(sub, world, seed, 1500, n_hidden, n_latent, units, "learned_ema", wcfg.n_v1)
        out[ema] = {**tr, "place_trained": round(place["trained"], 3),
                    "place_untrained": round(place["untrained"], 3)}
    old = out[ema_values[0]]
    new = out[new_default] if new_default in out else out[ema_values[-1]]
    tamed = new["max_heldout_loss"] < 0.5 * old["max_heldout_loss"]
    if verbose:
        print(f"[EMA A/B seed={seed} units={units} steps={steps} n_hidden={n_hidden}]")
        for ema in ema_values:
            c = out[ema]
            print(f"  ema={ema:<8} max_online={c['max_online_loss']:>9} max_heldout={c['max_heldout_loss']:>9}"
                  f" final_heldout={c['final_heldout_loss']:>8} place={c['place_trained']:+.3f}"
                  f"(untr {c['place_untrained']:+.3f}) delta={c['place_trained']-c['place_untrained']:+.3f}")
            print(f"      heldout curve: {c['heldout_curve']}")
        print(f"  spikes_tamed(new-default {new_default} max_heldout < 0.5x old 0.99)={tamed}")
        print(f"  EMA A/B {'PASS' if tamed else 'INCONCLUSIVE'}")
    return {"arms": out, "tamed": tamed}


def _local_ridge_importance(X, Y, lam=10.0):
    """Per-hidden-unit importance for decoding Y from X = sum|ridge weight| over targets (standardized)."""
    Y = np.asarray(Y, np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    mu = X.mean(0, keepdims=True); sd = np.maximum(X.std(0, keepdims=True), 1e-2)
    Xs = (X - mu) / sd
    d = Xs.shape[1]
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(d, dtype=np.float64), Xs.T @ Y)
    return np.abs(W).sum(axis=1)


def _behav_reward_rate(sub, world, n=1500, explore_eps=0.05):
    """Frozen behavioral rollout (the substrate's OWN policy, no learning): mean reward per step + #eats.
    Respects any lesion mask the caller has set on `sub` (set it before, clear it after)."""
    was_frozen = sub._frozen
    sub.freeze()
    e0 = world.n_eats
    tot = 0.0
    a_prev = world.last_action
    for _ in range(n):
        d = world.drive_afferent(); v1 = world.crop_v1feat()
        h = sub.observe(v1, a_prev, d)
        a = sub.act(h, explore_eps=explore_eps)
        r, _ = world.step(a); tot += r; a_prev = a
    if not was_frozen:
        sub.unfreeze()
    return tot / max(1, n), int(world.n_eats - e0)


def _place_lesion_reward(sub, world, seed, n_hidden, n_probe=1500, n_behav=3000, lesion_frac=0.10,
                         n_random=4):
    """Is the place code LOAD-BEARING on reward now? Find the place-carrying units (ridge importance for
    abs-position), lesion them, and measure the reward-rate DROP vs an equal RANDOM-unit lesion."""
    sub.freeze()
    sub.set_lesion_mask(None)
    H, POS = [], []
    a_prev = world.last_action
    for _ in range(n_probe):
        d = world.drive_afferent(); v1 = world.crop_v1feat()
        pos = world.agent
        h = sub.observe(v1, a_prev, d)
        a = sub.act(h, explore_eps=0.3)
        world.step(a)
        H.append(np.asarray(h.get() if hasattr(h, "get") else h, dtype=np.float32))
        POS.append(np.asarray(pos, dtype=np.float32)); a_prev = a
    H = np.asarray(H); POS = np.asarray(POS)
    k = max(8, int(lesion_frac * n_hidden))
    imp = _local_ridge_importance(H, POS)
    place_mask = np.zeros(n_hidden, dtype=bool); place_mask[np.argsort(imp)[::-1][:k]] = True

    rr_intact, _ = _behav_reward_rate(sub, world, n_behav)
    sub.set_lesion_mask(place_mask); rr_place, eats_place = _behav_reward_rate(sub, world, n_behav)
    sub.set_lesion_mask(None)
    rng = np.random.default_rng(seed + 71)
    rand_rrs = []
    for _ in range(n_random):
        m = np.zeros(n_hidden, dtype=bool); m[rng.choice(n_hidden, k, replace=False)] = True
        sub.set_lesion_mask(m); rr_r, _ = _behav_reward_rate(sub, world, n_behav); rand_rrs.append(rr_r)
    sub.set_lesion_mask(None)
    rr_rand = float(np.mean(rand_rrs))
    place_drop = rr_intact - rr_place
    rand_drop = rr_intact - rr_rand
    ratio = (place_drop / rand_drop) if rand_drop > 1e-9 else float("inf") if place_drop > 1e-9 else float("nan")
    load_bearing = (place_drop > 0) and (rand_drop <= 0 or place_drop >= 1.5 * rand_drop)
    return {"rr_intact": round(rr_intact, 5), "rr_place_lesion": round(rr_place, 5),
            "rr_random_lesion": round(rr_rand, 5), "place_drop": round(place_drop, 5),
            "random_drop": round(rand_drop, 5), "ratio": (round(ratio, 3) if np.isfinite(ratio) else ratio),
            "k_units": k, "place_load_bearing": bool(load_bearing)}


def run_nav_ab(seed=42, steps=60000, n_hidden=128, n_latent=64, n_probe=1500, units="rate",
               encoder="learned_ema", value_weight=1.0, nav_dmin=6, pred_horizon=1,
               grid_size=18, verbose=True):
    """3rd-move A/B — TASK-REQUIRED position. OFF = the predictive-only control (random-respawn food, value
    head off) — the regime the 1st-move finding showed LOSES the place code by the long horizon (drifts to /
    below the untrained-reservoir floor). ON = the nav-required homing world (fixed remembered larder + trial
    reset on eat) + a value head whose gradient shapes the shared core. Reports, at early/mid/late checkpoints
    for both arms: place-decode vs the untrained-reservoir floor (does ON PERSIST where OFF FADES?), the
    held-out predictive loss, and the behavioral reward-rate (does ON LEARN to home?). Then an END place-unit
    lesion vs a random-unit lesion (is place LOAD-BEARING on reward now?). Honest: if ON does not persist or
    is not load-bearing, the printout says so."""
    from sim.pcs_substrate import PredictiveContinualSubstrate, PCSConfig
    checkpoints = sorted(set([max(1, steps // 5), steps // 2, steps]))
    arms = {}
    for nav in (False, True):
        wcfg = WorldConfig(seed=seed, nav_required=nav, nav_dmin=nav_dmin, grid_size=grid_size)
        world = ForkPCSWorld(wcfg)
        stat_eval = _collect_stationary_eval(wcfg, seed, n=400)
        vw = value_weight if nav else 0.0
        scfg = PCSConfig(n_hidden=n_hidden, feat_dim=wcfg.n_v1, n_latent=n_latent, n_actions=N_ACTIONS,
                         n_drive=4, tbptt_T=16, units=units, encoder=encoder, seed=seed,
                         value_weight=vw, pred_horizon=pred_horizon)
        sub = PredictiveContinualSubstrate(scfg)
        traj = []; done = 0; a_prev = -1
        for ckpt in checkpoints:
            for _ in range(ckpt - done):
                d = world.drive_afferent(); v1 = world.crop_v1feat()
                h = sub.observe(v1, a_prev, d); a = sub.act(h, explore_eps=0.3)
                r, _ = world.step(a); sub.learn(r); a_prev = a
            done = ckpt
            place = _probe_place_decode(sub, world, seed, n_probe, n_hidden, n_latent, units, encoder, wcfg.n_v1)
            held = sub.eval_predictive_loss(stat_eval)
            rr, eats = _behav_reward_rate(sub, world, 1500)
            traj.append({"step": ckpt, **place, "held_out": round(float(held), 4),
                         "reward_rate": round(rr, 5), "eats": eats})
        lesion = _place_lesion_reward(sub, world, seed, n_hidden, n_probe)
        arms["ON" if nav else "OFF"] = {"traj": traj, "lesion": lesion, "value_weight": vw}

    off, on = arms["OFF"]["traj"][-1], arms["ON"]["traj"][-1]
    # PERSIST: ON keeps place clearly above the untrained floor at the long horizon
    on_persists = on["trained"] > on["untrained"] + 0.05
    off_faded = off["trained"] <= off["untrained"] + 0.05
    on_beats_off = on["trained"] > off["trained"] + 0.05
    lb = arms["ON"]["lesion"]["place_load_bearing"]
    learned = on["reward_rate"] > off["reward_rate"] * 1.05 or arms["ON"]["traj"][-1]["reward_rate"] > arms["ON"]["traj"][0]["reward_rate"]
    passed = on_persists and lb
    if verbose:
        print(f"[NAV A/B seed={seed} units={units} steps={steps} n_hidden={n_hidden} vw={value_weight} dmin={nav_dmin}]")
        for arm in ("OFF", "ON"):
            print(f"  {arm} (value_weight={arms[arm]['value_weight']}):")
            for c in arms[arm]["traj"]:
                print(f"    step={c['step']:>7}  place(trained)={c['trained']:+.3f}  untrained={c['untrained']:+.3f}"
                      f"  rawV1={c['rawv1']:+.3f}  Δ(tr-un)={c['trained']-c['untrained']:+.3f}"
                      f"  held={c['held_out']:.3f}  reward_rate={c['reward_rate']:.5f}  eats={c['eats']}")
            L = arms[arm]["lesion"]
            print(f"    place-lesion: rr_intact={L['rr_intact']:.5f} rr_place={L['rr_place_lesion']:.5f} "
                  f"rr_random={L['rr_random_lesion']:.5f}  place_drop={L['place_drop']:+.5f} "
                  f"random_drop={L['random_drop']:+.5f} ratio={L['ratio']} LOAD_BEARING={L['place_load_bearing']}")
        print(f"  --> ON place@end={on['trained']:+.3f}(untr {on['untrained']:+.3f})  "
              f"OFF place@end={off['trained']:+.3f}(untr {off['untrained']:+.3f})")
        print(f"  ON_persists(above untrained)={on_persists}  OFF_faded(at/below untrained)={off_faded}  "
              f"ON_beats_OFF={on_beats_off}  ON_place_load_bearing={lb}  ON_learned(reward)={learned}")
        print(f"  NAV A/B {'PASS' if passed else 'INCONCLUSIVE'}  (PASS = ON persists above floor AND place load-bearing)")
    return {"OFF": arms["OFF"], "ON": arms["ON"], "on_persists": on_persists, "off_faded": off_faded,
            "on_beats_off": on_beats_off, "place_load_bearing": lb, "learned": learned, "passed": passed}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="fork PCS grounded world + Day-1 smoke")
    ap.add_argument("--smoke", action="store_true", help="run the Day-1 smoke (loss drop + position vs floors)")
    ap.add_argument("--consolidation-ab", action="store_true",
                    help="OFF-vs-ON long-horizon demonstration of consolidation (drift vs retention)")
    ap.add_argument("--stability-ab", action="store_true",
                    help="UNSTABLE-vs-STABLE demonstration (grad-clip/spike-skip taming loss spikes)")
    ap.add_argument("--ema-ab", action="store_true",
                    help="EMA-target A/B (0.99 vs 0.999 vs 0.9999) — taming the target-driven loss spikes")
    ap.add_argument("--nav-ab", action="store_true",
                    help="3rd-move A/B — TASK-REQUIRED position: predictive-only OFF vs nav-required homing + "
                         "value-head ON (place persistence + reward-rate + place-lesion load-bearing)")
    ap.add_argument("--consolidation", action="store_true", help="single-arm smoke with consolidation ON")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--ab-steps", type=int, default=60000)
    ap.add_argument("--n-hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--value-weight", type=float, default=1.0,
                    help="value-head weight for the ON arm of --nav-ab (0 disables the value channel)")
    ap.add_argument("--nav-dmin", type=int, default=6, help="min post-eat agent-respawn distance from the larder")
    ap.add_argument("--grid-size", type=int, default=18, help="world grid size for --nav-ab (smoke: smaller = more eating)")
    ap.add_argument("--pred-horizon", type=int, default=1, help="JEPA horizon k for --nav-ab arms (isolate: 1)")
    ap.add_argument("--units", choices=["rate", "spike"], default="rate")
    ap.add_argument("--encoder", choices=["learned_ema", "fixed"], default="learned_ema")
    args = ap.parse_args()
    if args.nav_ab:
        res = run_nav_ab(seed=args.seed, steps=args.ab_steps, n_hidden=args.n_hidden, units=args.units,
                         encoder=args.encoder, value_weight=args.value_weight, nav_dmin=args.nav_dmin,
                         pred_horizon=args.pred_horizon, grid_size=args.grid_size)
        raise SystemExit(0 if res["passed"] else 1)
    if args.ema_ab:
        res = run_ema_ab(seed=args.seed, steps=args.ab_steps, n_hidden=args.n_hidden, units=args.units)
        raise SystemExit(0 if res["tamed"] else 1)
    if args.stability_ab:
        res = run_stability_ab(seed=args.seed, steps=args.ab_steps, n_hidden=args.n_hidden,
                               units=args.units, encoder=args.encoder)
        raise SystemExit(0 if res["passed"] else 1)
    if args.consolidation_ab:
        res = run_consolidation_ab(seed=args.seed, steps=args.ab_steps, n_hidden=args.n_hidden,
                                   units=args.units, encoder=args.encoder)
        raise SystemExit(0 if res["passed"] else 1)
    res = run_smoke(steps=args.steps, seed=args.seed, units=args.units, encoder=args.encoder,
                    consolidation=args.consolidation)
    raise SystemExit(0 if (res["loss_dropped"] and res["beats_floors"]) else 1)
