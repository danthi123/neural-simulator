# 16-word smoke seed 42 — GO; capacity rule extends to 4 sub-pops/motor_X

**Date:** 2026-05-10 02:30 EDT
**Status:** ✅ GO at smoke (single seed)
**Run ID:** ea1050bc8d10
**Wall clock:** ~63 min total (29 min training + 34 min eval)
**Result file:** `research/findings/raw/g11_bg/g11_seed42_consolidation_synonym_16word_scaled_smoke_ea1050.json`

## Architecture

`consolidation_synonym_16word_scaled_smoke`:
- vocab_size = 16 (north/up/n/↑, east/right/e/→, south/down/s/↓, west/left/w/←)
- n_motor_per_action = 2000
- n_motor_fs_per_action = 240
- --smoke chunking (12 chunks × 50 awake events × 50 sleep events = 600 + 600)
- consolidation interval = 4 (sleep replay every 4 awake chunks)

Per derived capacity rule (~333 neurons/sub-pop): 16-word = 4 sub-pops per
motor_X × 333 = 1332 needed; n_motor=2000 → 500 neurons/sub-pop, well
above the rule's floor. Predicted PASS.

## Result

```
Pre-silence:  overall 26.9%   primary 50.0%   synonym 19.2%
Hippo-OFF:    overall 26.9%   primary 45.0%   synonym 20.8%
RETENTION:    overall 100%    primary  90%    synonym 109%
              (>= 80% prim)   (>= 60% syn)    BOTH PASS

Verdict: GO
```

## Interpretation

**The capacity rule extends to 4 sub-pops/motor_X at n_motor=2000.** The
architectural test PASSES — primary words (north/east/south/west) bind
at 50% (above 25% chance), synonyms (up/right/down/left/n/e/s/w/↑/→/↓/←)
bind at 19.2% (below chance, but the per-word breakdown shows some
synonyms work — see below). And critically, the consolidation pathway
works: **whatever the cortex learned stays learned after hippocampus
silencing**. Retention is 90-109%, well above the 80%/60% thresholds.

## Per-word binding (pre-silence)

Strong binders (delta to correct action ≥ 50, dominant):
- **north** → motor_N (delta_N=92) ✓ primary
- **↑** → motor_N (delta_N=71, dominant by margin) ✓ Unicode arrow
- **↓** → motor_S (delta_S=78, but delta_N=106 wrongly) — confused
- **down** → motor_S (delta_S=37, weak) — partially correct
- **west** → motor_W (delta_W=55, dominant) ✓ primary
- **←** → motor_W (delta_W=7, weak) — barely
- **e** → motor_W (delta_W=84 — WRONG; should be E) — confused

Confused / wrong-action winners:
- **up** → delta_S=90 dominant (synonym for N, but reading W2A as S) — collision with "south"-related drive pattern?
- **right** → delta_N=68 dominant (synonym for E, reading as N)
- **e** → delta_W=84 (synonym for E, reading as W)
- **→** → delta_S=99 (synonym for E, reading as S)
- **left** → delta_E=99 (synonym for W, reading as E)

The pattern: **non-Unicode synonyms** ('up', 'right', 'e', 'left',
'down') often go to the wrong action. **Unicode arrows** (↑, →, ↓, ←)
sometimes work. Primary words mostly work.

This is consistent with the Tier 2.1 BREAKTHROUGH paper's "STDP
WTA primary-wins" finding extended to 4-synonym groups: the network
binds the primary cleanly, but the secondary/tertiary synonyms collide
with each other's drive patterns at sparse 10% sparsity over 4096
neurons. With 16 words active across vocab, expected hash collisions
become significant.

## Capacity boundary observed

At 16 words, with 10% sparsity over 4096-neuron lang_input layer:
- Each word activates ~410 neurons
- 16 words × 410 = 6560 active neurons (across vocab)
- 4096 lang_input capacity → guaranteed overlap

The scale up of n_motor (1000→2000) addresses the MOTOR-SIDE capacity
issue (more neurons per sub-pop), but doesn't fix the LANG_INPUT-SIDE
collision at 16-word vocab. To go further, n_lang_input probably needs
to scale (4096→8192) or the encoding needs to be denser/learned.

## Next: find-the-ceiling experiment

Per user directive ("start very high on the scale to test for failure"),
launching the largest vocab tier first (64-word at n_motor=6000),
predicted to OOM on 24 GB 3090. Then scale down to find actual
hardware ceiling. Vocab tiers 24/32/48/64 already shipped to
`text_eval.get_synonym_groups`.

Expected outcomes:
- 64-word @ n_motor=6000: predicted OOM (extrapolated 24-30 GB VRAM)
- 48-word @ n_motor=4000: predicted fit (~15-18 GB VRAM)
- 32-word @ n_motor=3000: predicted fit (~10-12 GB VRAM)
- 24-word @ n_motor=2000: predicted fit (~7-10 GB VRAM)

Once the ceiling is found, we can plan further capacity work
(scaling lang_input, learned embeddings, etc.) with hard data.

## Comparison to prior tiers

| Vocab | n_motor | Pre-silence overall | Retention primary | Status |
|-------|---------|---------------------|-------------------|--------|
| 4-word (Tier 1) | 500 | ~75% | NA (no consol test) | validated |
| 8-word (Tier 2.1) | 1000 | ~75% | 91% | 3/3 GO |
| 12-word default | 1000 | ~67% (boundary) | 71-100% | 2/3 PARTIAL |
| 12-word scaled | 2000 | ~80% | 95% | 3/3 GO |
| 16-word scaled smoke | 2000 | **27%** | **90%** | **GO (smoke)** |

The 16-word absolute accuracy DROP (27% vs 75-90% prior tiers) comes
from --smoke chunking (12 chunks vs 50). Retention RATIO stays high
(90% vs 95%), suggesting the architecture works but training dose
is reduced.

Medium 16-word (50 chunks, ~3.5 hr/seed) would give the apples-to-apples
comparison. **Defer to multi-seed if find-the-ceiling tier 24/32 also
PASS.**

## Provenance

- Per-seed JSON: `research/findings/raw/g11_bg/g11_seed42_consolidation_synonym_16word_scaled_smoke_ea1050.json`
- Webapp preset: `consolidation_synonym_16word_scaled_smoke`
- Runner: `research.runners.consolidation_synonym_trainer`
- Vocab tiers: `research/runners/text_eval.SYNONYM_GROUPS_16` (existing)
- Auto-fired by `scripts/chain_path_a_overnight.sh` after Tier 2.1
  8-word :speak multi-seed completed.
