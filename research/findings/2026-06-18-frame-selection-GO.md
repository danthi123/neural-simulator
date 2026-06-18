# Frame selection — richer-syntax #2 complete on the substrate, GO 6/6 (2026-06-18, CYCLE 204)

## Headline

The missing half of multi-frame comprehension — **frame SELECTION** — is GO 6/6: a neural
verb-position→frame map auto-selects the word-order frame from a structural cue, and end-to-end
(auto-select then comprehend) the agent reads the correct roles for a sentence in an *unknown*
frame. selection **1.000**, end-to-end **1.000**, with the permuted-selection-map control collapsing
to **0.222 ≈ chance** and the lesion to **0.333 ≈ chance** — the cue→frame map is load-bearing and
neural. With CYCLE 203 (per-frame comprehension GO), **richer-syntax #2 (productive multi-frame
comprehension) is complete on the spiking substrate.**

## Mechanism

CYCLE 203 proved the parser comprehends N frames (SVO/VSO/OSV) *given* the frame. The open piece:
how does the agent know which frame? The structural cue is the **verb's position** — it uniquely
identifies the frame (verb-at-0 → VSO, verb-at-1 → SVO, verb-at-2 → OSV). The agent computes the
verb-position by knowing its vocabulary's verbs (a legitimate lexical/morphology lookup — the host
role is the dictionary, not the cognition); a neural `FrameSelector` (3 verb-position cue units → 3
frame ensembles, Hebbian co-firing, the v16 rule) then maps verb-position → frame **in spikes**. The
selected frame indexes the `MultiFrameParser`'s comprehension. So the selection is neural; only
"which word is the verb" is a lexical lookup.

## Results (6 seeds: 42/43/44/100/101/102, GPU)

| metric | mean | note |
|--------|------|------|
| frame selection (verb-pos cue → correct frame) | **1.000** | perfect every seed |
| end-to-end (auto-select → comprehend → roles) | **1.000** | the agent comprehends an unknown-frame sentence |
| permuted-selection control | **0.222** (chance 0.333) | a scrambled cue→frame map → wrong frame → wrong roles (collapses) |
| lesion control (zero cue→frame weights) | **0.333** | collapses to chance — the learned map is load-bearing |
| no-confab moat | True (6/6) | |
| seeds GO (selection ≥0.90 ∧ end-to-end ≥0.90) | **6/6** | frozen gate met |

## Debugging note (honest)

The first run showed selection 1.000 but end-to-end 0.000 — a **test-harness bug, not a substrate
failure**: `MultiFrameParser.role_of` returns `(role, margin)`, and the end-to-end loop compared the
whole tuple to a role string (the `[0]` was missing). The neural frame-selection was perfect from
the first run; fixing the harness gave end-to-end 1.000. Systematic-debugging caught it (selection
1.000 + e2e 0.000 is logically inconsistent with a working selector → look at the hand-off, not the
brain).

## Anti-cheat controls (all passed)

- **Permuted-selection collapses (0.222 ≈ chance):** scrambling the verb-position→frame teacher makes
  the selector pick the wrong frame → wrong roles. So the cue→frame mapping is genuinely learned.
- **Lesion collapses (0.333 ≈ chance):** zeroing the cue→frame weights drops selection to chance —
  the spiking map is load-bearing, not a hard-coded rule.
- **The selection is neural:** the frame is the max-firing frame ensemble on the bridge; only the
  verb-detection is a host lexical lookup (the dictionary, not the cognition).

## Honest scope + next

- Frames: SVO + VSO + OSV. Real wh-questions, datives, imperatives are bounded extensions (more
  frames + more cues; the verb-position cue generalizes to any frame distinguished by verb position).
- NEXT (the production integration, mirroring richer-syntax #1's `enable_attributed`): wire a
  `FrameParser` (FrameSelector + MultiFrameParser) into `BrainConversationalAgent` as an opt-in, so
  the production agent comprehends multiple frames end-to-end. NO `sim/` edit (reuse the
  BridgeParser pattern). Then richer-syntax #3 (embedded-clause PARSING from nested input).

## Reproduce

```bash
SIM_BACKEND=cupy python -u -m research.runners._phaseB_frame_selection_derisk \
    --seeds 42,43,44,100,101,102
```
Runner: `research/runners/_phaseB_frame_selection_derisk.py`. Prior: `2026-06-18-multiframe-comprehension-GO.md`.
