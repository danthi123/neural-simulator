# Tomorrow's plan — based on tonight's results

**Generated:** 2026-05-03 ~07:55 EDT (autonomous, mid-investigation)
**For:** Daniel waking up

---

## What you should look at first

```bash
python -m research.runners.swr_aggregate
```

Single-line summary table of all 9 conditions (baseline / v2+SWR / H1 /
H4 / 5 arch variants). Tells you everything at a glance.

Then:
```bash
python -m research.runners.swr_per_seed
```
Per-seed cross-condition comparison.

If the auto-followup ran, also look at:
```bash
ls research/findings/raw/g11_bg/text_eval_arch_*_seed*.json
```

---

## Decision matrix

| If H1 mean | If arch sweep best | Action |
|---|---|---|
| ≥ 27% (rescues) | any | Ship balanced replay as default. Done. |
| 24-26% (no rescue) | < 32% on seed 42 | More architecture investigation needed. |
| 24-26% (no rescue) | ≥ 32% on seed 42 | Auto-followup ran 6-seed validation. Check results. |
| ≥ 32% with mean ≥ 32% on 6 seeds | --- | Ship that variant + return to higher tasks |

---

## Likely scenarios

### Scenario A: H1 fixes regression (W->A ~28%)

The 4pp regression in v2+SWR was caused by buffer-composition bias.
Fix: replace default replay with `--phase3-balanced-directions` as
the default. Update text_train_curriculum.py defaults. Rerun a quick
sanity check at 1 seed to confirm. Move on to next investigation.

### Scenario B: H1 doesn't fix it (W->A ~24%)

The regression is more fundamental — soft-bound STDP saturation,
broad collateral plasticity during replay, or pre-existing weight
amplification. Skip SWR replay entirely; v2 baseline is the
operational ceiling for SWR. Pivot fully to architectural changes.

### Scenario C: arch sweep finds a winner (e.g. motor50 = 35%)

Auto-followup ran 6 seeds. If motor50 6-seed mean ≥ 32%, that's
**the new operational best**. Update the flagship recommendation
in CLAUDE.md.

### Scenario D: arch sweep finds nothing (all ≤ 32%)

Architecture-limited at fundamental level. Next directions:
1. **Bigger across the board** (lang1024 + motor100, sparsity 0.025)
2. **Hebbian re-enable with fixed decay** — 2026-05-02 disabled
   Hebbian to fix decay, but Hebbian co-firing IS biology. The
   right fix is a non-decaying Hebbian rule, not disabling Hebbian.
3. **Phase 1 visuomotor pre-training back on** — current v2 uses
   `--phase1-episodes 0`. With Phase 1, the cascade could acquire
   useful structure that helps Phase 2 word-action discrimination.
4. **Different motor readout** — population vector decoding via
   distributed motor pop (already implemented; tested standalone in
   variant D of the arch sweep, called dpop, but not included in
   tonight's run because it's a different beast — schedule for
   future runs).

---

## Health check

If anything looks wrong:

```bash
python -m research.runners.swr_status
```

PIDs to check:
- 9768 — H4-then-H1 super-orchestrator
- 28684 — wait_h1_then_arch_sweep waiter
- 49476 — wait_arch_then_followup waiter

Master logs:
- `research/findings/raw/g11_bg/run_h4.master.log` — H4 batch
- `research/findings/raw/g11_bg/run_h1.master.log` — H1 batch
- `research/findings/raw/g11_bg/run_arch_sweep_seed42.master.log` — arch sweep
- `research/findings/raw/g11_bg/auto_followup_arch.log` — followup

---

## What I deferred

* **Hebbian decay fix** — significant code change (sim/bridge.py
  Hebbian update kernel). Worth a careful PR rather than midnight hack.
* **Bigger architecture variants (lang1024, motor100)** — too long
  per-run for tonight (probably 90 min each). Schedule as a tier-2
  sweep later.
* **Investigate seed 42 H4 confusion identical north/east rows** —
  noted in the H4 results doc as a measurement artifact, but a real
  bug investigation would trace the random state through the
  bridge to see if there's an aliasing issue.

## My honest assessment

The v2 ceiling at 28.5% is essentially **chance + small bias**. We're
not doing real word-action learning yet — we're surfing the cascade's
default biases. The buffer-composition mechanism doc (committed
2026-05-03 ~07:55) explains why SWR makes it worse rather than
better.

The arch sweep tests known structural hypotheses but if all 5 stay
≤ 30%, the right path forward is probably a more biology-grounded
architectural rebuild:
- Cell-assembly Wernicke (recurrent within-region)
- Cortico-cortical Wernicke→Broca pathway with developmentally
  pruned topography
- Larger motor pools with population-vector decoding

This is a multi-day rebuild, not an overnight tweak.
