"""Auto-grow chat demo — Strategy B of Phase A2.

Demonstrates the TierPromoter orchestration loop end-to-end without
committing to GPU bridge integration. Uses pluggable train_fn /
transfer_fn so tests can supply mocks and the real version can wire
to bio_three_factor + bridge.set_pathway_weights later.

Design doc: docs/plans/2026-05-11-phase-a2-chat-repl-auto-grow-design.md

Usage:
    # Toy demo with synthetic accuracy ramp
    python -m research.runners.auto_grow_chat \
        --initial-tier 4 --max-promotions 3

    # With lineage tracking (growth events recorded)
    python -m research.runners.auto_grow_chat \
        --initial-tier 4 --max-promotions 3 \
        --lineage auto_grow_demo

The toy `train_fn` returns synthetic accuracy that climbs over epochs.
The toy `transfer_fn` returns a stub bridge object with the new tier
label. Both will be replaced in Phase A2 Strategy A (next session)
with real `bio_three_factor.run_three_factor` + actual weight transfer
via `bridge.set_pathway_weights`.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional


# ── Mock training / transfer (for the toy demo + tests) ─────────────────


@dataclass
class _MockBridge:
    """Tiny bridge stand-in carrying just enough state for the demo.

    A real bridge from bio_three_factor would replace this; the
    orchestration loop doesn't care about the actual contents.
    """
    tier: int
    arch: dict = field(default_factory=dict)
    n_training_epochs: int = 0


def synthetic_train_fn(tier: int, arch: dict,
                        bridge: Optional[_MockBridge] = None,
                        epoch: int = 0) -> tuple[_MockBridge, float]:
    """Toy training: accuracy ramps up over epochs at each tier.

    Pattern:
    - First 2 epochs at a new tier: low accuracy (0.4-0.6) — model still adapting
    - Epochs 3-5: climbs (0.7-0.85) — partial mastery
    - Epoch 6+: high (0.91-0.95) — mastery, triggers promotion

    Args:
        tier: current vocab tier
        arch: tier's arch dict (n_lang, n_motor, n_motor_fs)
        bridge: existing bridge or None (if None, build new)
        epoch: training epoch counter (resets on promotion)

    Returns:
        (bridge, accuracy)
    """
    if bridge is None:
        bridge = _MockBridge(tier=tier, arch=dict(arch))
    bridge.n_training_epochs = epoch
    if epoch < 2:
        acc = 0.45 + 0.05 * epoch
    elif epoch < 6:
        acc = 0.70 + 0.04 * (epoch - 2)
    else:
        # Stable high accuracy that triggers promotion
        acc = 0.92 + 0.005 * (epoch - 6)
    return bridge, acc


def synthetic_transfer_fn(from_tier: int, to_tier: int,
                            old_bridge: _MockBridge,
                            new_arch: dict) -> _MockBridge:
    """Toy weight transfer: just builds a new MockBridge at new_tier.

    A real implementation would call bridge.extract_per_pathway_csrs +
    transfer_weights_dense + new_bridge.set_pathway_weights per pathway.
    """
    return _MockBridge(tier=to_tier, arch=dict(new_arch),
                         n_training_epochs=0)


# ── The orchestration loop ──────────────────────────────────────────────


@dataclass
class AutoGrowResult:
    """Outcome of an auto-grow run."""
    initial_tier: int
    final_tier: int
    promotions_executed: int
    epochs_total: int
    epochs_at_each_tier: dict[int, int] = field(default_factory=dict)
    tier_history: list[tuple[int, float, int]] = field(default_factory=list)
    # (tier, acc, epoch_global) at each step
    growth_event_count: int = 0
    bridge: Any = None  # final bridge

    def summary(self) -> dict:
        return {
            "initial_tier": self.initial_tier,
            "final_tier": self.final_tier,
            "promotions_executed": self.promotions_executed,
            "epochs_total": self.epochs_total,
            "epochs_at_each_tier": dict(self.epochs_at_each_tier),
            "growth_event_count": self.growth_event_count,
        }


def run_auto_grow_demo(
    initial_tier: int = 4,
    threshold: float = 0.90,
    consecutive_required: int = 3,
    max_promotions: int = 3,
    max_epochs_per_tier: int = 50,
    train_fn: Callable = synthetic_train_fn,
    transfer_fn: Callable = synthetic_transfer_fn,
    lineage_name: Optional[str] = None,
    lineage_root: Optional[Path] = None,
    verbose: bool = True,
) -> AutoGrowResult:
    """Run the auto-grow orchestration loop.

    Args:
        initial_tier: starting vocab tier (4, 8, 12, ...)
        threshold: accuracy needed to count as a "pass" toward promotion
        consecutive_required: how many consecutive passes trigger promote
        max_promotions: stop after this many promotions
        max_epochs_per_tier: safety cap per tier (avoid infinite training)
        train_fn(tier, arch, bridge, epoch) -> (bridge, acc):
            One training epoch + eval. Toy default uses synthetic ramp.
        transfer_fn(from_tier, to_tier, old_bridge, new_arch) -> new_bridge:
            Build new-tier bridge with weights transferred. Toy default
            returns a stub MockBridge.
        lineage_name: if given, record promotion growth events to lineage
        lineage_root: optional Path for lineage storage (default: project's
            bridges/lineage/)
        verbose: print progress to stdout

    Returns:
        AutoGrowResult with the full history.
    """
    from sim.auto_growth import TierPromoter

    lineage = None
    if lineage_name is not None:
        from sim.lineage import BridgeLineage
        lineage = BridgeLineage(lineage_name,
                                 root=lineage_root)
        if not lineage.exists():
            # Seed metadata so growth events have a place to land
            meta = lineage.read_metadata()
            meta.current_tier = f"{initial_tier}-word"
            lineage.write_metadata(meta)

    promoter = TierPromoter(
        initial_tier=initial_tier,
        threshold=threshold,
        consecutive_required=consecutive_required,
    )

    # Build initial bridge
    initial_arch = promoter.ladder.arch_for_tier(initial_tier)
    bridge, acc = train_fn(initial_tier, initial_arch, bridge=None, epoch=0)

    result = AutoGrowResult(
        initial_tier=initial_tier,
        final_tier=initial_tier,
        promotions_executed=0,
        epochs_total=0,
    )

    if verbose:
        target_tier = _peek_n_promotions_ahead(
            promoter.ladder, initial_tier, max_promotions
        )
        print(f"[AUTO-GROW] Starting at tier {initial_tier} "
              f"(target: tier {target_tier} in {max_promotions} promotions)",
              flush=True)

    epoch_global = 0
    epoch_at_tier = 0
    for _ in range(max_promotions + 1):  # +1 = final tier completion
        # Inner training loop at current tier
        for epoch_at_tier in range(max_epochs_per_tier):
            bridge, acc = train_fn(
                tier=promoter.current_tier,
                arch=promoter.ladder.arch_for_tier(promoter.current_tier),
                bridge=bridge,
                epoch=epoch_at_tier,
            )
            epoch_global += 1
            result.tier_history.append(
                (promoter.current_tier, acc, epoch_global)
            )
            if verbose:
                pass_str = ""
                if acc >= threshold:
                    pass_str = (f" ({promoter.consecutive_passes + 1}/"
                                f"{consecutive_required} pass)")
                print(f"[EPOCH {epoch_global}] tier={promoter.current_tier} "
                      f"acc={acc:.3f}{pass_str}", flush=True)
            # Check promotion
            plan = promoter.step(acc)
            if plan is not None:
                if result.promotions_executed >= max_promotions:
                    # Promotion budget exhausted — note the plan but
                    # don't execute. Loop ends below.
                    if verbose:
                        print(f"[AUTO-GROW] Promotion candidate "
                              f"{plan.from_tier}->{plan.to_tier} "
                              f"reached, but max_promotions="
                              f"{max_promotions} budget exhausted; "
                              f"stopping.", flush=True)
                    result.epochs_at_each_tier[promoter.current_tier] = epoch_at_tier + 1
                    break
                # Promote!
                if verbose:
                    print(f"[AUTO-GROW] -> PROMOTING {plan.from_tier} -> "
                          f"{plan.to_tier}", flush=True)
                bridge = transfer_fn(
                    from_tier=plan.from_tier,
                    to_tier=plan.to_tier,
                    old_bridge=bridge,
                    new_arch=plan.to_arch,
                )
                promoter.confirm_promotion(plan, lineage=lineage)
                if lineage is not None:
                    result.growth_event_count += 1
                result.epochs_at_each_tier[plan.from_tier] = epoch_at_tier + 1
                result.promotions_executed += 1
                epoch_at_tier = 0
                break  # exit inner loop; outer loop starts new tier
        else:
            # Inner loop hit max_epochs_per_tier without promotion;
            # break outer loop too
            if verbose:
                print(f"[AUTO-GROW] Reached max_epochs_per_tier="
                      f"{max_epochs_per_tier} at tier "
                      f"{promoter.current_tier} without promotion. "
                      f"Stopping.", flush=True)
            result.epochs_at_each_tier[promoter.current_tier] = max_epochs_per_tier
            break
        if result.promotions_executed >= max_promotions:
            # Promotion budget exhausted
            break

    result.final_tier = promoter.current_tier
    result.epochs_total = epoch_global
    result.bridge = bridge
    if verbose:
        print(f"[AUTO-GROW] Complete. Final tier: {result.final_tier} "
              f"in {result.promotions_executed} promotions "
              f"({result.epochs_total} epochs total).", flush=True)
    return result


def _peek_n_promotions_ahead(ladder, current: int, n: int) -> int:
    """What tier is `n` promotions ahead of `current`?"""
    t = current
    for _ in range(n):
        nxt = ladder.next_tier(t)
        if nxt is None:
            return t
        t = nxt
    return t


# ── CLI entry ──────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--initial-tier", type=int, default=4,
                    help="Starting vocab tier (default 4)")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="Accuracy threshold for promotion (default 0.90)")
    ap.add_argument("--consecutive-required", type=int, default=3,
                    help="Consecutive passes needed (default 3)")
    ap.add_argument("--max-promotions", type=int, default=3,
                    help="Stop after this many promotions (default 3)")
    ap.add_argument("--max-epochs-per-tier", type=int, default=50,
                    help="Safety cap per tier (default 50)")
    ap.add_argument("--lineage", type=str, default=None,
                    help="Optional lineage name to record growth events")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional JSON output of result summary")
    args = ap.parse_args()

    result = run_auto_grow_demo(
        initial_tier=args.initial_tier,
        threshold=args.threshold,
        consecutive_required=args.consecutive_required,
        max_promotions=args.max_promotions,
        max_epochs_per_tier=args.max_epochs_per_tier,
        lineage_name=args.lineage,
        verbose=True,
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(result.summary(), indent=2),
            encoding="utf-8",
        )
        print(f"\n[SUMMARY] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
