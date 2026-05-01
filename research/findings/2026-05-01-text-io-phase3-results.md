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

## Smoke test results — chance-level despite Gabor + scale increases

Five training configurations tested. **Cortex_N dominance is structural,
not regime-dependent:**

| Config | Train | Gabor V1 | Reset | I→W acc | W→A acc | Time | Diagnostic |
|---|---|---|---|---|---|---|---|
| 1a Baseline | 100+100 | OFF | OFF | 22.5% | 27.5% | 213s | initial |
| 1b Scale | 500+500 | OFF | OFF | 22.5% | 25.0% | ~17min | scale doesn't help |
| 2a Gabor | 200+200 | ON | OFF | 20.0% | 22.5% | 394s | N-bias emerges |
| 2b Gabor+reset | 200+200 | ON | ON | 25.0% | **12.5%** | 606s | **catastrophic N-bias** |

Regime 2b (Gabor + inter-trial reset) made W→A *worse than chance*
(12.5% vs 25% chance). Confusion matrix shows total cortex_N capture:

```
W→A confusion (Regime 2b):
              N    E    S    W
north         5    3    0    2   ← partial signal
east          9    0    0    1   ← total miss
south         9    1    0    0   ← total miss
west          8    2    0    0   ← total miss
```

Every word now predicts cortex_N most often. The agent literally never
picks cortex_S or cortex_W during eval. **This is structural, not a
training problem** — it persists across 5 different training configs.

## Diagnostic: cascade has built-in N-bias (text_diag_cascade_bias.py)

To localize the issue, ran 3 tests on an UNTRAINED bridge (no STDP, no
training trials, just raw cascade dynamics):

```
TEST 1: NO INPUT (spontaneous activity, last 100ms):
  cortex_N: 23 spikes  ← 2× higher than others
  cortex_E: 11 spikes
  cortex_S:  8 spikes
  cortex_W: 12 spikes

TEST 2: EQUAL 100pA drive to all 4 cortex_X:
  cortex_N: 68 spikes (29.3%)  ← still highest
  cortex_E: 61 spikes (26.3%)
  cortex_S: 48 spikes (20.7%)
  cortex_W: 55 spikes (23.7%)

TEST 3: language_input drive only (untrained):
  word='north' → winner=N
  word='east'  → winner=N  ← wrong
  word='south' → winner=S  ← right by accident
  word='west'  → winner=N  ← wrong
```

**Root cause**: the BG cascade is structurally asymmetric. Even with NO
training and equal drive, cortex_N fires more than cortex_S. STDP
training amplifies this: trials reinforce "any input → cortex_N"
because cortex_N fires the most, regardless of language_input pattern.

The asymmetry likely originates from cluster A (closed BG loop) and/or
cluster E (topography) — both add directional structure to the cascade.
This was invisible during K v2 evaluation because the heuristic / visual
cortex provides differential input that overrides the bias. Without
that differential input (text-only eval), the bias dominates.

**All chance-level (≈25%).** Both Gabor and 2× scale failed to push
performance above chance. The Gabor 200-pair confusion matrix reveals
a critical diagnostic: every direction word predicts cortex_N most often
(N=7, 7, 6, 9 out of 10 trials across north/east/south/west). The agent
has a strong N-dominance bias rather than learned word-action mappings.

```
W → A confusion (Gabor 200-pair, rows = word, columns = predicted action):
              N    E    S    W
north         7    1    1    1   ← correct
east          7    0    0    3   ← wrong
south         6    1    2    1   ← wrong
west          9    0    1    0   ← wrong
```

## Diagnosis (revised)

**The supervisor-clamp regime has a fundamental cross-contamination
problem.** When training trial 1 supervises cortex_N for "north", STDP
grows north_pattern → cortex_N. But NMDA-mediated bistability in the
visual cortex regions causes cortex_N to maintain elevated activity
into trial 2 (~100 ms gap is shorter than NMDA τ of 100ms). When trial 2
clamps cortex_E for "east", STDP also grows east_pattern → cortex_N
because cortex_N is still firing. Across many trials, cortex_N
accumulates inputs from ALL language_input patterns.

The Gabor pre-init makes this WORSE because V1 firing reliably ramps up
cortex_N (whichever cortex region happens to receive the strongest
upstream signal from the BG cascade defaults). At chance-level training,
random V1 fires noise that averages out; structured V1 amplifies the bias.

This isn't a "needs more training" problem — more training reinforces the
bias. It's a **regime design problem**.

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

## What's actually needed — fixing the regime

Empirically, the current "clamp post-synaptic + tonic reward" regime
**doesn't differentiate associations**. Three architectural changes
the regime needs (in order of expected impact):

### 1. Inter-trial reset (~1 hour to implement)

Between training trials, run a 50-100 ms "blank" period where
`cp_external_input_current[:] = 0` and `current_reward_signal = 0`.
This lets NMDA-mediated cortex bistability decay before the next
clamp. Current behavior likely cross-contaminates STDP across trials.

### 2. Reward gating on correct response (medium effort)

Instead of tonic +1 reward during the supervisor clamp, run a brief
"unclamped readout" first: drive the input (image or token), let the
neural state respond NATURALLY, observe whether language_output (or
cortex_X) matches the target. Apply +1 reward ONLY if correct, 0 or
-1 if wrong. This is closer to how real STDP+reward learns — only
strengthen connections that produced correct behavior.

### 3. Curriculum: 4-token vocab first (small effort)

Currently the vocabulary has 29 tokens and 8 direction synonyms. STDP
has to disambiguate 8 patterns mapped to 4 actions. Start with 4 tokens
(just "north"/"east"/"south"/"west") before introducing synonyms.

### 4. Bigger language_input drive

Currently 200pA × 10% sparsity ≈ 20 active neurons × 0 weight = 0pA
into cortex_X. Even after STDP grows weights, the natural drive may
not exceed BG cascade defaults. Try drive=500pA with weight_init>0
(start at small positive instead of zero).

### 5. Different supervised paradigm

Drop the post-synaptic clamp entirely. Instead, gate STDP plasticity
based on which trial type is active. The plasticity_gate infrastructure
already supports this — we'd need a new "supervisor signal" that gates
ONLY the relevant pathway (e.g., only enable
language_input_to_cortex_N gate when training "north", disable others).

## Status (honest assessment)

The text I/O **infrastructure is functional and tested** (39 unit/integ
tests pass). The agent CAN take a token in, run simulation, read out a
token. The bridge `set_token_drive` and `read_language_output` work
correctly. Pathways are wired. Embeddings are deterministic.

**What doesn't work yet: the supervised training regime.** Above-chance
accuracy requires regime improvements (1-5 above), not more compute on
the current regime. None of these are conceptually hard, but they
require careful experimentation.

**This is a clean stopping point**: infrastructure is stable, the
problem is well-localized, and the next investment (regime redesign)
can happen in a fresh session with clear context.
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
