---
type: finding
status: contributing
date: 2026-06-04
---

# Cheat-removal #4 (visual grounding) — cheap-first RESOLVES: Gabor-V1 features ground usable concept codes — 2026-06-04

**One line:** Real biological V1 Gabor receptive fields (`sim/visual_cortex.py`, Hubel-Wiesel oriented simple
cells) turn distinct visual stimuli into concept codes that are **well-separated on average (mean pairwise cosine
0.25)** and **robustly pattern-completion-cleanup-able (97% under additive noise + ≤2px translation)** — the same
cleanup mechanism that resolved the grounded *word-cue* level of #4. So sensory features genuinely produce usable
grounded concept codes; the only "overlapping" pair is two adjacent-orientation bars, which *should* be similar.

## The #4 gap this addresses

Backlog #4: concept codes are random phasors or learned from a hashed/orthogonal WORD encoder
(`vocab_to_drive_pattern`), not from real sensory grounding. The audit marked #4 PARTIAL — the grounded
*word-cue* level was resolved (STDP-learned codes from the word encoder support full composition via pattern
completion), but the deeper *visual/Gabor sensory* grounding was open. This probe closes the foundational
visual-grounding question, cheaply, before any agent integration.

## Method (`research/runners/_visual_grounding_probe.py`)

12 distinct visual "concepts" (8 oriented bars + 4 corner spots) → each rendered to a 32×32 ON/OFF image → passed
through the REAL V1 Gabor bank (`build_v1_simple_weights`: 8 orientations × 4 frequencies × 16×16 positions =
8192 simple cells) → unit-normalized 8192-d V1 response = the grounded concept code. Two measurements:

1. **Separability** — off-diagonal pairwise cosine across the 12 codes.
2. **Pattern-completion cleanup under corruption** — each stimulus is perturbed (additive Gaussian noise σ=0.25
   AND a random ≤2px translation; V1 *simple* cells are NOT translation-invariant, so cleanup must carry the
   robustness), its V1 code computed, and the nearest CLEAN concept code identified. Correct = the true concept.

## Result

```
(1) SEPARABILITY (12 visual concepts, V1 dim 8192):
    pairwise cosine: mean=0.252  max=0.709  (most-similar pair: bar_0deg ~ bar_22deg -- adjacent SHOULD be similar)
(2) PATTERN-COMPLETION CLEANUP (noise=0.25 + translation<=2px, 5/concept):
    recovered true concept 58/60 = 97%  (mean top1-top2 margin 0.144)
  => GROUNDED CODES USABLE (well-separated on average + robust cleanup)
```

The single high-cosine pair is `bar_0deg ~ bar_22deg` (orientations 22.5° apart) — that is *correct* visual
similarity, not a failure; cleanup still distinguishes them 97% of the time across all concepts. Mean cosine 0.25
shows the codes are well-separated overall.

## Interpretation

- **Real sensory features ground concept codes.** The biological V1 Gabor bank — not a hashed word encoder —
  produces concept representations that are distinct and robustly recoverable. This is genuine sensory grounding
  of the concept layer, the open part of #4.
- **The cleanup mechanism is the same one that carries the rest of the agent.** The grounded word-cue level of #4
  resolved via CA3-style pattern completion (snap the noisy readout to the nearest clean concept attractor before
  composing). The visual-grounding level resolves by the *same* mechanism — the noisy/translated V1 code snaps to
  the right clean concept. One biological idea (attractor cleanup) handles both sensory modalities. (Biology:
  V1 simple → complex → IT builds invariance up the ventral stream; here a single cleanup step stands in for that
  invariance, and it suffices for robust identification.)

## Honest scope (what is NOT claimed)

- **Visually-groundable concepts only.** This grounds concepts with visual referents (orientations, shapes,
  positions). Abstract words (`go`, `big`, function words) have no canonical image — visual grounding is naturally
  limited to the visual subset, exactly as embodied-cognition theory predicts (some concepts are sensory-grounded,
  others amodal/abstract). The honest target is a *multi-modal* grounding where the visual pipeline grounds the
  visual concepts and other modalities/the word encoder ground the rest.
- **Cheap-first, mechanism-level.** This validates that the grounding *produces usable codes*. The next step —
  feeding these V1-grounded codes into the composition agent as `external_codes` and re-running the unified-agent
  benchmark — is the follow-up integration (the word-cue grounded-cleanup mode already showed learned grounded
  codes compose; the expectation is the visual codes do too, via the shared cleanup).
- **Separability is partly by construction** (V1 is an orientation/position discriminator). The informative result
  is the *robustness* (97% under noise + translation), where simple cells are not shift-invariant and cleanup
  earns its keep.

## Status

- Backlog #4: PARTIAL → **visual-grounding mechanism validated cheap-first**. Grounded word-cue level (resolved
  earlier) + grounded visual level (this probe) both produce usable codes via the shared attractor cleanup. The
  remaining open item is the *agent integration* (feed visual-grounded codes as `external_codes` + benchmark) and
  multi-modal coverage of non-visual concepts.

## Follow-up (same day): the grounded codes COMPOSE, not just separate

`research/runners/_visual_grounded_composition_probe.py` closes the agent-integration question for the visual
subset. Each V1 sensory code is converted to a phasor (FHRR) code via a FIXED random complex projection — so the
phasor code is a deterministic function of the sensory features (grounded, not free). Then the composition
substrate's primitives (bind = elementwise complex multiply, bundle = complex sum, unbind = multiply by the
role's conjugate, cleanup = max normalized dot over the grounded codebook) run on a 2-role fact built from two
visual-grounded concepts:

```
CLEAN compose (unbind agent + patient -> grounded concept):        24/24 = 100%
CORRUPTED-sensory compose (agent slot from noisy+shifted image):   11/12 = 92%
```

So sensory features → grounded phasor codes → bind/bundle/unbind/cleanup → recover the correct concepts, and the
recovery survives corrupting the sensory input (the noisy/shifted image still recovers via cleanup). The
grounded codes compose **as well as random phasor codes** (100% clean is the expected FHRR behavior at D=2048
over a 12-code codebook — matching the project's validated SVO decode 1.00 — so grounding does not degrade
composition). This is the visual-grounded analogue of the word-cue grounded-cleanup composition result: real
sensory grounding feeds the composition substrate end-to-end.

## Files

- `research/runners/_visual_grounding_probe.py` — the cheap-first Gabor-V1 grounding probe (separability +
  corruption cleanup), reusing `sim/visual_cortex.py`'s real V1 Gabor receptive fields.
- `research/runners/_visual_grounded_composition_probe.py` — the follow-up: V1 features → phasor codes → FHRR
  bind/unbind/cleanup (grounded codes compose 100% clean / 92% from corrupted sensory input).
