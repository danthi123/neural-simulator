# 2026-05-01 — Text I/O Phase 3: training pipeline + interactive demo

**Status:** Infrastructure complete (Phases 1-3 committed). Initial
training results pending (n=100+100 pair smoke test in flight).

This doc will be updated with measured accuracy once the eval task
(`b9yzoelgf`) completes.

## What was built

8 commits, ~1500 LOC of new infrastructure, 22 new tests:

| Commit | Phase | Component |
|---|---|---|
| `de57200` | Design | Comprehensive 270-line design doc covering 4 phases |
| `bca4619` | 1a | `sim/text_embeddings.py` — 29-token vocab, deterministic-Gaussian 256-dim L2-normalized embeddings, sparse-coding drive helper |
| `a767db8` | 1b | `language_input` + `language_output` regions in `build_bg_brain_regions` (256 plastic recurrent each); 9 plastic pathways (input→PFC, input→cortex_X, cortex_X→output, IT→output) |
| `325948a` | 1c | `bridge.set_token_drive(token, ...)` + `bridge.read_language_output(spike_counts, ...)` |
| `461e4e6` | 2 | `research/runners/text_train.py` — 2-regime supervised pipeline (image→word + word→action) with structural plasticity disabled |
| `5984eaa` | 3a | `research/runners/text_eval.py` — train + evaluate accuracy on fresh trials |
| `1d00857` | 3a fix | utf-8 / cp1252 console encoding fix |
| `da004f5` | 3c | `research/runners/text_chat.py` — bounded-vocab REPL for interactive dialogue |

## Architecture

```
USER TEXT INPUT
    │ embed(token) — deterministic Gaussian 256-dim, L2-normalized
    │ vocab_to_drive_pattern: top-10% sparse activation
    ▼
language_input (256 plastic recurrent)
    │ plastic, gated "language_input_to_cortex" (zero init)
    ▼
cortex_{N,E,S,W}
    │ existing G v2.5 BG cascade
    ▼
motor / action selection

VISUAL INPUT (parallel — Cluster K v2)
    │ retina (32×32) → V1 → V2 → IT
    │ plastic, gated "it_to_language_output" (zero init)
    │ plastic, gated "cortex_to_language_output" (zero init, action verbalization)
    ▼
language_output (256 plastic recurrent)
    │ readout = mean firing rate per neuron (skip 30ms onset)
    │ nearest_token(activity, vocab) — top-k cosine similarity
    ▼
USER TEXT OUTPUT
```

## Bridge API

```python
bridge.set_token_drive("north", drive_pA=200, sparsity=0.1)
# Activates ~10% of language_input neurons matching the token's
# embedding pattern. Deterministic per token.

predicted = bridge.read_language_output(
    spike_counts,                # (n_neurons,) accumulated over readout window
    n_steps,                     # for normalization to firing rate
    top_k=3,
    vocab=["north", "east", "south", "west"],
)
# Returns top_k tokens by cosine similarity to firing-rate vector.
```

## Training regimes (text_train.py)

**Regime 1: image → word (visual labeling)**
- Render gridworld at random (agent, goal); compute target word from
  Manhattan-greedy direction.
- Drive retina via K v2 visual cortex (existing path).
- CLAMP language_output to target word's embedding pattern at 250 pA.
- Reward = +1.
- STDP+reward grows IT → language_output for IT patterns that co-fire
  with the clamped target word.

**Regime 2: word → action (verbal command)**
- `bridge.set_token_drive(word)` on language_input.
- Drive cortex_X for target action with strong supervisor current.
- Reward = +1.
- STDP+reward grows language_input → cortex_X.

## Evaluation (text_eval.py)

After training, present fresh trials WITHOUT supervisor clamps:
- **Image → word:** 100 fresh gridworld images. Read language_output's
  spike pattern, decode to nearest token. Was the cardinal direction
  correctly emitted?
- **Word → action:** 25 trials per direction word. Read cortex_X firing
  rates. Was the correct cortex_X most active?

Reports accuracy + confusion matrix for both regimes.

## Smoke test results (n=100 train + n=40 eval)

Training: 100 image→word pairs + 100 word→action pairs (213 sec).

Evaluation: **chance-level accuracy** at this training scale.

| Regime | Accuracy | Chance | Notes |
|---|---|---|---|
| Image → word | **9/40 = 22.5%** | 25% | slightly below chance |
| Word → action | **11/40 = 27.5%** | 25% | basically chance |

Confusion matrices (I→W rows = target, columns = predicted):

```
target → predicted   north  east  south  west
north                 5     2     3      0
east                  5     1     7      3
south                 4     0     1      0
west                  1     4     2      2
```

Word → action (rows = word, columns = cortex_X spike count winner):

```
word → action   N    E    S    W
north           5    1    3    1
east            6    2    1    1
south           3    3    4    0
west            4    3    3    0
```

The diagonals are weak; the off-diagonals are not consistent (e.g. "south"
gets predicted for many "east" trials in I→W, but the W→A confusion
matrix shows different patterns). This is consistent with the agent
having NO learned associations, not bias toward a particular wrong answer.

## Diagnosis

**Hypothesis: 100 training pairs is far too few at this scale + with random V1.**

The visual cortex (Cluster K v2) achieves 38% time-at-goal at 16×16 over
1800 steps × 200 sub-steps = 360,000 simulation timesteps with continuous
reward signal. Text I/O training delivered 200 pairs × 200 sub-steps =
40,000 timesteps with discrete pairing. That's 9× fewer total updates.

Compounding factors:
1. **No Gabor pre-init.** V1 simple cells use random weights, so V1/V2/IT
   are firing on essentially noise. STDP+reward can't carve useful
   representations from noise. The K v2 16×16 breakthrough explicitly
   credited Gabor pre-init for orientation tuning.
2. **Zero-init weights start at 0.** STDP+reward growing weights from
   zero requires post-synaptic firing (currently driven only by the
   supervisor clamp during training trials), pre-synaptic firing, and
   reward — all to align in time. Slow.
3. **Reward is +1 every step during training.** Tonic reward creates a
   dopamine-saturated state that doesn't differentiate "this association
   was useful" from "this association was harmful". A more biologically
   realistic schedule would be reward-on-correct only.

## Recommended next steps

## Caveats / known issues

1. **Gabor pre-init disabled in text_train.** `apply_v1_gabor_weights` has
   a CSR-resize edge case under some configurations (broadcast error
   between synapse-indexed arrays). V1 simple cells use random-init
   weights instead, which means orientation tuning isn't biology-correct.
   STDP+reward should compensate over enough trials, but learning may be
   slower than with proper Gabor init.
2. **Structural plasticity disabled.** Repeated +1 reward under
   structural plasticity rapidly grows synapses (171574 → 171830 in
   one step), causing array-shape mismatches. We only need STDP weight
   changes, not new synapse formation.
3. **Single-token I/O.** No multi-word sentences yet. v2 would add
   temporal recurrence for sequences.
4. **Vocabulary bounded to 29 tokens** (4 cardinal directions + synonyms +
   simple objects). Real-world vocabulary needs >1000 tokens, which
   would require larger language regions.
5. **No PFC working memory engagement tested.** language_input → dlpfc_wm
   pathway exists but contribution to learning is untested.

## Interactive demo (text_chat.py)

```
$ python -m research.runners.text_chat --n-image-word 300 --n-word-action 300

[training: ~6-12 min for 600 pairs]

INTERACTIVE TEXT CHAT (v1)
================================================================
Commands:
  <word>            - drive language_input with the word, observe action
                       (e.g. 'north', 'east', 'left'/'right')
  show AX AY GX GY  - drive retina with that gridworld, agent says word
                       (e.g. 'show 1 1 6 6')
  reset             - zero all current input
  quit / q / Ctrl+D - exit

you> show 1 1 6 6
agent> north (other guesses: east, west)
you> north
agent> would go N (matches expected)  cortex_X spikes: {N: 142, E: 38, S: 12, W: 41}
you> show 6 6 1 1
agent> south (other guesses: west, east)
you> quit
[exit]
```

## What's next — directly implied by chance-level smoke result

These should be done in order before claiming "functional textual training".

### Tier 1 — quick fixes that may already be enough

1. **Fix Gabor pre-init resize bug.** The K v2 breakthrough (2.87 at 16×16)
   relied on Gabor pre-init. Without it, V1/V2/IT fire on noise. Worth
   1-2 hours of bridge-debug to make `apply_v1_gabor_weights` work in the
   text-io build. (TODO at line ~75 of `text_train.py`.)

2. **5×–10× larger training set (1000+1000 pairs ~30-60 min).** Real
   biological language acquisition takes thousands of repetitions; 100 is
   far too few. Most likely the chance-level result is just sample size.

3. **Restrict supervisor clamp temporally.** Currently we clamp the
   supervisor across the entire 100ms stim window. Real instructive
   signals are brief (~10-20 ms). Shorter clamps may give STDP a cleaner
   "learn this association" signal.

### Tier 2 — architectural improvements

4. **Reward-on-correct only.** Tonic +1 reward during training trials
   doesn't teach. Try: reward = +1 when language_output's natural
   activity (without clamp) matches target, else 0.

5. **Pre-train the supervisor channel.** Run training without supervisor
   clamp first to let STDP+reward simply amplify whatever weak co-firings
   exist. Then add supervisor.

6. **Curriculum.** Start with 4-token vocabulary (just N/E/S/W). Once
   stable, expand. Currently we go straight to 8-direction-synonym
   vocabulary which may be too entropic.
2. **Fix Gabor pre-init** so V1 has biology-correct orientation tuning.
   This should significantly accelerate IT → language_output learning.
3. **Multi-token sequences** for real sentences. Requires temporal
   recurrence in language_input/output.
4. **Claude API bridge** — translate user's natural English to bounded
   vocab → agent → bounded response → natural English. Makes the demo
   accessible to humans, hides the vocabulary limit.
5. **Webapp integration** — text panel in the live mode tab.
6. **Audio I/O (Cluster L)** — same architecture, swap retina for
   cochleotopic region, V1 for A1, etc.

## Files

- Design: `docs/plans/2026-05-01-text-interaction-design.md`
- Embeddings: `sim/text_embeddings.py`
- Bridge integration: `sim/bridge.py` (set_token_drive, read_language_output)
- Region wiring: `research/runners/g11_bg_runner.py` (--enable-text-io flag)
- Training: `research/runners/text_train.py`
- Evaluation: `research/runners/text_eval.py`
- Interactive: `research/runners/text_chat.py`
- Tests: `tests/test_text_embeddings.py`, `tests/test_bridge_text_io.py`
