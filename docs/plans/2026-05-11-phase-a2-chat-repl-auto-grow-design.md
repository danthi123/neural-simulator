---
type: plan
status: live
date: 2026-05-11
---

# Phase A2 — chat_repl --auto-grow integration design

**Date:** 2026-05-11 04:55 EDT
**Status:** DESIGN — wiring the already-shipped TierPromoter (commit
`aef098a`) into chat_repl so the sim auto-promotes vocabulary tiers
as it masters each one.
**Prereqs:**
- `sim/auto_growth.py` (✅ shipped, 25 tests)
- `sim/lineage.py` with growth-event recording (✅ shipped)
- `chat_repl` with lineage integration (✅ shipped)

---

## The integration gap

Today (post-Phase A):

- `TierPromoter` is standalone: tracks consecutive passes, generates
  `PromotionPlan` objects, records growth events on lineage when
  `confirm_promotion()` is called.
- `transfer_weights_dense()` provides the pure-Python weight-copy
  logic (upper-left block preserved, new neurons random-init).
- **But nothing calls these.** chat_repl trains a fixed tier and stops.

The integration:
1. User invokes `chat_repl --auto-grow --max-promotions 3`
2. chat_repl runs an outer loop: train at tier T → eval → promote → repeat
3. At each promotion: build new tier bridge, transfer weights, save to lineage
4. Outer loop exits at top tier or after max_promotions

## Three integration strategies (ranked by risk)

### Strategy A — Full GPU bridge swap (highest payoff, highest risk)

**Scope:** ~3-5 days of focused work + 1 day of GPU validation.

The flow:
```python
def run_auto_grow(initial_tier=4, threshold=0.90, max_promotions=3, ...):
    promoter = TierPromoter(initial_tier=initial_tier, threshold=threshold)
    lineage = BridgeLineage("main")
    bridge = build_bridge_for_tier(initial_tier)

    for promotion_count in range(max_promotions + 1):
        # Train at current tier
        bio_three_factor.run_three_factor(bridge=bridge, ...)
        # Eval
        acc = evaluate_w_to_a(bridge)
        # Promotion check
        plan = promoter.step(acc)
        if plan is None:
            # Not enough consecutive passes yet; train more
            continue
        # Promotion! Build new tier + transfer
        old_bridge = bridge
        bridge = build_bridge_for_tier(plan.to_tier)
        for pathway_name in plan.pathways:
            _transfer_pathway_weights(old_bridge, bridge, pathway_name)
        # Save to lineage with growth event
        lineage.save(bridge, tier=f"{plan.to_tier}-word",
                     arch=plan.to_arch)
        promoter.confirm_promotion(plan, lineage=lineage)
        print(f"🎉 Promoted to {plan.to_tier}-word!")

    return bridge
```

**Pros:**
- Real auto-growth working end-to-end
- Lineage records each promotion as a growth event (visible in webapp)
- User can resume from any tier via lineage load

**Cons:**
- Bridge swap is invasive — need to deallocate old bridge cleanly
- Weight transfer at production scale (n_lang=2048→4096, n_motor=500→1000)
  needs careful per-pathway slicing
- Training cost: each tier needs significant training (~minutes to hours)
  to actually master before promotion. Multi-tier auto-grow could take
  many hours.
- GPU-only test path; CPU NumPy backend is too slow for production tiers

### Strategy B — CPU-toy auto-grow demo (low risk, low payoff)

**Scope:** ~1 day of focused work.

A toy-scale auto-grow demo that runs on the NumPy backend in seconds.
Uses tiny tiers (n_lang=32 → 64 → 128) with synthetic eval accuracies
that pass the threshold on schedule. Demonstrates the orchestration
logic without committing to GPU training infrastructure.

**Pros:**
- Pure-Python; ships fast
- Validates the orchestration loop end-to-end
- Reference implementation for Strategy A

**Cons:**
- Demo only; not a production auto-grow
- Mock eval doesn't exercise real chat REPL training
- Sets a "this is a research artifact" framing rather than a product

### Strategy C — CLI flag + scaffold only (minimum viable)

**Scope:** ~half-day.

Add `--auto-grow` to chat_repl + a scaffold function in
`research/runners/auto_grow_chat.py` that prints "auto-grow not yet
integrated; see design doc". Tests verify the CLI flag is wired.

**Pros:**
- Trivial to ship
- Reserves the namespace without committing to implementation
- Forces the design conversation before the bigger Strategy A scope

**Cons:**
- No user-facing capability
- Vaporware risk if Strategy A is never built

## Recommended approach — Strategy B for this autonomous arc, Strategy A for next

Strategy B is the **shippable** unit:
- Pure CPU; no GPU dependency
- Validates the orchestration end-to-end
- Provides the reference implementation that Strategy A will productionize

Strategy A waits for:
- The user's strategic Path 1/2/3 decision (auto-grow only makes sense
  on biology-grounded scale-up paths)
- A GPU session to validate weight transfer at production tier sizes
- Decision on whether to ship "auto-grow" as a top-level chat_repl feature
  or keep it as a separate batch runner

## Strategy B implementation plan

### New module: `research/runners/auto_grow_chat.py`

```python
def run_auto_grow_demo(
    initial_tier: int = 4,
    threshold: float = 0.90,
    consecutive_required: int = 3,
    max_promotions: int = 3,
    train_fn: Callable[[int, dict], Tuple[Any, float]] = None,
    transfer_fn: Callable[[int, int, Any], Any] = None,
    lineage_name: str = "auto_grow_demo",
) -> dict:
    """Demonstrate TierPromoter orchestration without bridge integration.

    Args:
        train_fn(tier, arch) -> (bridge, accuracy)
            Mock or real training function. Toy demo uses a function
            that returns synthetic accuracy that climbs over epochs.
        transfer_fn(from_tier, to_tier, old_bridge) -> new_bridge
            Mock or real weight transfer. Toy demo: returns a stub
            "bridge" object with the new tier label.

    Returns:
        Dict with:
        - promotions_executed: int
        - final_tier: int
        - tier_history: list[(tier, acc, epoch)]
        - growth_events: list[dict]
    """
    promoter = TierPromoter(
        initial_tier=initial_tier,
        threshold=threshold,
        consecutive_required=consecutive_required,
    )
    # ... orchestration loop ...
```

### Tests

`tests/test_auto_grow_chat.py`:
- Synthetic accuracy climbs steadily → 3 promotions executed
- Synthetic accuracy stays low → 0 promotions
- Synthetic accuracy oscillates → correct consecutive-pass tracking
- At-top-of-ladder behavior (no further promotion)
- Lineage receives growth events
- Mock weight transfer is called per promotion

### CLI

```bash
python -m research.runners.auto_grow_chat \
    --initial-tier 4 --max-promotions 3 \
    --threshold 0.90 --consecutive-required 3 \
    --lineage auto_grow_demo
```

Output:
```
[AUTO-GROW] Starting at tier 4 (target: tier 32 in 3 promotions)
[EPOCH 1] tier=4 acc=0.45 (1/3 needed for promotion)
[EPOCH 2] tier=4 acc=0.62
[EPOCH 3] tier=4 acc=0.91 (1 consecutive pass)
[EPOCH 4] tier=4 acc=0.93 (2 consecutive)
[EPOCH 5] tier=4 acc=0.94 (3 consecutive) -> PROMOTING to tier 8
[AUTO-GROW] Promoted 4 -> 8 (mock weight transfer)
...
[AUTO-GROW] Complete. Final tier: 32 in 3 promotions.
```

### Lineage integration

Each promotion writes a growth event:
```json
{
  "kind": "tier_promotion",
  "description": "Promoted 4-word -> 8-word (3 consecutive passes >= 90%)",
  "metadata": {"from_tier": 4, "to_tier": 8, "epoch": 5}
}
```

Visible in:
- `bridge_lineage show auto_grow_demo`
- `bridge_lineage growth-log auto_grow_demo`
- Webapp Lineages tab detail view

## What Strategy A would add (deferred to next session)

1. **Real `train_fn`:** `bio_three_factor.run_three_factor` with periodic
   eval callbacks. Returns (bridge, accuracy) each epoch.
2. **Real `transfer_fn`:** uses `bridge.set_pathway_weights` to copy
   from old → new bridge across all 16 pathways in GROWING_PATHWAYS.
3. **chat_repl CLI flag:** `--auto-grow` invokes the Strategy A loop
   instead of single-tier training.
4. **GPU validation:** 6-seed Tier 1 → Tier 2.1 auto-promotion
   with embodied-Hebbian binding still aligned post-promotion.

## Provenance

- This doc: `docs/plans/2026-05-11-phase-a2-chat-repl-auto-grow-design.md`
- TierPromoter: `sim/auto_growth.py` (✅ shipped)
- Auto-growth design: `docs/plans/2026-05-10-auto-growth-design.md`
- Lineage growth events: `sim/lineage.py::LineageMetadata.add_growth_event`

Next autonomous-arc unit: Strategy B implementation + tests.
