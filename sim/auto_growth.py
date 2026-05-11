"""Auto-Growth — TierPromoter for vocabulary tier promotion via checkpoint reload.

Per user (2026-05-10): "is there a way to run the sim in a fashion where
it starts off smaller, then grows (scales) autonomously as it learns/
grows to allow us to start small and migrate to more powerful hardware
as it grows?"

This module implements Phase A of the auto-growth roadmap (design doc
docs/plans/2026-05-10-auto-growth-design.md): the sim trains at a
small vocabulary tier, hits a mastery threshold for N consecutive
evals, and then auto-promotes to the next tier with weights copied
from the trained smaller bridge.

Two-class scaffold:

- `TierLadder` — pure-data: tier sizes, per-tier arch (lang/motor sizes),
  and tier-to-tier mapping. Easy to unit test.

- `TierPromoter` — orchestration: monitors eval accuracy, triggers
  promotions, records growth events to a lineage. Holds a reference to
  a bridge_loader callable (so it can build the next-tier bridge with
  matching arch).

The weight-transfer logic is split into:
- `compute_transfer_plan(old_arch, new_arch, pathway_names)` — pure
  numpy/python; describes which (pre, post) edges map 1:1 vs random-init.
- `bridge.set_pathway_weights(...)` — GPU integration; not exercised
  in unit tests.

Design doc: docs/plans/2026-05-10-auto-growth-design.md (Phase A)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# Vocabulary tier ladder, matching the encoding-axis discovery + Tier 2.1
# BREAKTHROUGH arch. Each tier has a target arch (n_lang_input, n_motor,
# n_motor_fs). Tiers above 64 are speculative / cloud-anchored per the
# Phase 1 strategic addendum (and the 96-word XL NEGATIVE finding
# 2026-05-11).
TIER_LADDER = {
    4:   {"n_lang": 2048,  "n_motor": 500,  "n_motor_fs": 60},
    8:   {"n_lang": 4096,  "n_motor": 1000, "n_motor_fs": 120},
    12:  {"n_lang": 4096,  "n_motor": 2000, "n_motor_fs": 240},
    16:  {"n_lang": 4096,  "n_motor": 2000, "n_motor_fs": 240},
    24:  {"n_lang": 8192,  "n_motor": 2000, "n_motor_fs": 240},
    32:  {"n_lang": 8192,  "n_motor": 2000, "n_motor_fs": 240},
    48:  {"n_lang": 8192,  "n_motor": 2000, "n_motor_fs": 240},
    64:  {"n_lang": 8192,  "n_motor": 2000, "n_motor_fs": 240},
    # Above is the 64-word ceiling validated for local 3090. Cloud-only:
    96:  {"n_lang": 16384, "n_motor": 4000, "n_motor_fs": 480},  # speculative; 96 NEGATIVE at n_motor=2000
    128: {"n_lang": 16384, "n_motor": 4000, "n_motor_fs": 480},
    256: {"n_lang": 32768, "n_motor": 8000, "n_motor_fs": 960},
}

# Default mastery threshold: 90% W->A accuracy over the synonym vocabulary
DEFAULT_THRESHOLD = 0.90

# Default consecutive passes required before promotion (debounce)
DEFAULT_CONSECUTIVE = 3

# Pathways that need weight transfer on promotion (per Phase 1.4 BRANCH A
# arch). For each pathway, both ends may grow if the involved regions grow.
GROWING_PATHWAYS = (
    "language_input_to_motor_N",
    "language_input_to_motor_E",
    "language_input_to_motor_S",
    "language_input_to_motor_W",
    "motor_N_to_language_output",
    "motor_E_to_language_output",
    "motor_S_to_language_output",
    "motor_W_to_language_output",
    "motor_N_to_motor_FS_N",
    "motor_E_to_motor_FS_E",
    "motor_S_to_motor_FS_S",
    "motor_W_to_motor_FS_W",
    "motor_FS_N_to_motor_N",
    "motor_FS_E_to_motor_E",
    "motor_FS_S_to_motor_S",
    "motor_FS_W_to_motor_W",
)


@dataclass
class TierLadder:
    """Immutable list of tiers + helpers for next/prev/arch lookup."""
    tiers: tuple[int, ...] = tuple(sorted(TIER_LADDER.keys()))

    def next_tier(self, current_tier: int) -> Optional[int]:
        """Return the tier immediately above current_tier, or None at top."""
        try:
            idx = self.tiers.index(current_tier)
        except ValueError:
            # current_tier not on the ladder; return the smallest tier above it
            for t in self.tiers:
                if t > current_tier:
                    return t
            return None
        if idx + 1 >= len(self.tiers):
            return None
        return self.tiers[idx + 1]

    def prev_tier(self, current_tier: int) -> Optional[int]:
        """Return the tier immediately below current_tier, or None at bottom."""
        try:
            idx = self.tiers.index(current_tier)
        except ValueError:
            return None
        if idx == 0:
            return None
        return self.tiers[idx - 1]

    def arch_for_tier(self, tier: int) -> dict:
        """Return {n_lang, n_motor, n_motor_fs} for the given tier."""
        if tier not in TIER_LADDER:
            raise ValueError(
                f"Unknown tier {tier}; valid tiers: {sorted(TIER_LADDER.keys())}"
            )
        return dict(TIER_LADDER[tier])


@dataclass
class PromotionPlan:
    """Plan describing a single tier promotion.

    Captures everything needed to actually execute the promotion (build
    next-tier bridge + transfer weights). Constructed by TierPromoter.step()
    when it decides a promotion is warranted, then passed back to the
    caller which holds the actual bridge reference.
    """
    from_tier: int
    to_tier: int
    from_arch: dict
    to_arch: dict
    triggered_by: str = "eval_threshold"
    pathways: tuple[str, ...] = GROWING_PATHWAYS


class TierPromoter:
    """Monitors eval accuracy + triggers tier promotions.

    Usage:
        promoter = TierPromoter(initial_tier=4, threshold=0.90)
        for ep in range(n_epochs):
            train_one_epoch(bridge)
            acc = evaluate_w_to_a(bridge)["accuracy"]
            plan = promoter.step(acc)
            if plan is not None:
                # Caller executes the promotion (it owns the bridge)
                new_bridge = build_bridge(plan.to_arch)
                transfer_weights(bridge, new_bridge, plan.pathways)
                bridge = new_bridge
                promoter.confirm_promotion(plan, lineage=my_lineage)

    The orchestration is intentionally split so that the heavy GPU work
    (build_bridge + transfer_weights) sits in the caller, not in this
    class. TierPromoter itself is pure-Python and unit-testable
    without GPU.
    """

    def __init__(self,
                 initial_tier: int = 4,
                 threshold: float = DEFAULT_THRESHOLD,
                 consecutive_required: int = DEFAULT_CONSECUTIVE,
                 ladder: Optional[TierLadder] = None):
        self.ladder = ladder or TierLadder()
        if initial_tier not in self.ladder.tiers:
            raise ValueError(
                f"initial_tier {initial_tier} not on ladder "
                f"({self.ladder.tiers})"
            )
        self.current_tier = initial_tier
        self.threshold = float(threshold)
        self.consecutive_required = int(consecutive_required)
        self.consecutive_passes = 0
        # History of (tier, accuracy, decision) for debugging
        self.eval_history: list[tuple[int, float, str]] = []

    def step(self, eval_accuracy: float) -> Optional[PromotionPlan]:
        """Called after each evaluation. Returns a PromotionPlan if a
        promotion should fire NOW, else None.

        Args:
            eval_accuracy: W->A accuracy at the current tier (0.0-1.0).

        Returns:
            PromotionPlan describing the next tier + arch + pathways
            to transfer, OR None if not promoting yet.
        """
        if eval_accuracy >= self.threshold:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
        if self.consecutive_passes >= self.consecutive_required:
            next_t = self.ladder.next_tier(self.current_tier)
            if next_t is None:
                # At top of ladder
                self.eval_history.append(
                    (self.current_tier, eval_accuracy, "at_top")
                )
                return None
            plan = PromotionPlan(
                from_tier=self.current_tier,
                to_tier=next_t,
                from_arch=self.ladder.arch_for_tier(self.current_tier),
                to_arch=self.ladder.arch_for_tier(next_t),
                triggered_by=(
                    f"{self.consecutive_passes} consecutive passes "
                    f">= {self.threshold:.0%}"
                ),
            )
            self.eval_history.append(
                (self.current_tier, eval_accuracy, "PROMOTE")
            )
            return plan
        self.eval_history.append(
            (self.current_tier, eval_accuracy, "wait")
        )
        return None

    def confirm_promotion(self, plan: PromotionPlan,
                            lineage=None) -> None:
        """Called by the caller after the bridge is rebuilt + weights
        transferred. Updates current_tier and records a growth event
        on the lineage (if provided).
        """
        if plan.from_tier != self.current_tier:
            raise ValueError(
                f"confirm_promotion: plan.from_tier {plan.from_tier} != "
                f"current_tier {self.current_tier}"
            )
        self.current_tier = plan.to_tier
        self.consecutive_passes = 0  # reset counter for next tier
        if lineage is not None:
            meta = lineage.read_metadata()
            meta.current_tier = f"{plan.to_tier}-word"
            meta.arch.update({
                "n_lang_input": plan.to_arch["n_lang"],
                "n_motor_per_action": plan.to_arch["n_motor"],
                "n_motor_fs_per_action": plan.to_arch["n_motor_fs"],
            })
            meta.add_growth_event(
                kind="tier_promotion",
                description=(
                    f"Promoted {plan.from_tier}-word -> {plan.to_tier}-word "
                    f"({plan.triggered_by})"
                ),
                from_tier=plan.from_tier,
                to_tier=plan.to_tier,
                from_arch=plan.from_arch,
                to_arch=plan.to_arch,
            )
            lineage.write_metadata(meta)


# ── Weight-transfer logic (pure-Python; tested without GPU) ─────────────


@dataclass
class TransferPlan:
    """Plan describing how weights map from old -> new pool for one pathway.

    Used by both unit tests and the real GPU implementation. The pure-
    Python version operates on integer index arrays; the GPU version
    additionally fetches and writes weights via bridge.set_pathway_weights.
    """
    pathway_name: str
    pre_old_size: int
    post_old_size: int
    pre_new_size: int
    post_new_size: int
    # Number of 1:1 mapped edges expected (upper-left block)
    expected_mapped_edges: int
    # Number of new edges added (random init)
    expected_new_edges: int


def compute_transfer_plan(pathway_name: str,
                            pre_old_size: int,
                            post_old_size: int,
                            pre_new_size: int,
                            post_new_size: int,
                            density: float = 1.0) -> TransferPlan:
    """Pure-data plan for one pathway. No GPU access.

    Used to predict mapped + new edge counts before executing on GPU.
    Useful for sanity-checking arch deltas.

    For an old pool of size (n_pre_old, n_post_old) -> new
    (n_pre_new, n_post_new), the 1:1 mapped block is the upper-left
    n_post_old x n_pre_old slice. Random-init edges fill the rest.

    Args:
        density: connection density per pathway (default 1.0 =
            fully-connected; real Phase 1.4 pathways use ~0.1).
    """
    if (pre_new_size < pre_old_size or post_new_size < post_old_size):
        raise ValueError(
            f"compute_transfer_plan({pathway_name}): new pool smaller "
            f"than old (pre: {pre_old_size}->{pre_new_size}, post: "
            f"{post_old_size}->{post_new_size}). Tier promotion only "
            f"grows; demotion not supported."
        )
    full_old = int(pre_old_size * post_old_size * density)
    full_new = int(pre_new_size * post_new_size * density)
    expected_mapped = full_old
    expected_new = full_new - full_old
    return TransferPlan(
        pathway_name=pathway_name,
        pre_old_size=pre_old_size,
        post_old_size=post_old_size,
        pre_new_size=pre_new_size,
        post_new_size=post_new_size,
        expected_mapped_edges=expected_mapped,
        expected_new_edges=expected_new,
    )


def transfer_weights_dense(old_W,
                              new_pre_size: int,
                              new_post_size: int,
                              rng=None,
                              new_weight_mean: float = 0.0,
                              new_weight_jitter: float = 0.0):
    """Pure-numpy weight transfer for a dense block.

    Args:
        old_W: (post_old, pre_old) numpy array of trained weights.
        new_pre_size, new_post_size: target dimensions.
        rng: numpy.random.Generator (for random init); None -> seeded(0).
        new_weight_mean, new_weight_jitter: distribution for the
            random-init quadrants (matches original arch's pathway prior).

    Returns:
        new_W: (post_new, pre_new) numpy array. Upper-left
            (post_old, pre_old) block is copied from old_W; remainder
            is sampled from N(new_weight_mean, new_weight_jitter).
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng(0)
    post_old, pre_old = old_W.shape
    if new_pre_size < pre_old or new_post_size < post_old:
        raise ValueError(
            f"transfer_weights_dense: new pool ({new_post_size}, "
            f"{new_pre_size}) smaller than old ({post_old}, {pre_old})"
        )
    # Allocate new W with random init from the prior
    new_W = rng.normal(loc=new_weight_mean, scale=new_weight_jitter,
                         size=(new_post_size, new_pre_size)).astype(
        old_W.dtype, copy=False
    )
    # Overwrite the upper-left block with trained weights
    new_W[:post_old, :pre_old] = old_W
    return new_W
