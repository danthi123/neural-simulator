---
type: plan
status: live
date: 2026-05-01
---

# Text interaction infrastructure: design

**Status:** Design (2026-05-01)
**Goal:** Add text input/output to the spiking neural simulator so a user
(or LLM like Claude) can train the agent on word-to-action / image-to-word
mappings, then have a bounded-vocabulary text dialogue with it.

**Predecessor:** Cluster K v2 visual cortex breakthrough (commit 7de3c84,
2.97 ± 0.12 at 16×16 perception-only). The same hierarchical-region pattern
that worked for retina → V1 → V2 → IT generalizes to language: token
embeddings → language_input region → cortex association → language_output
readout.

## Scope (what's in v1)

✅ Two new brain regions: `language_input`, `language_output`.
✅ Training pipeline: supervised (image, word) pairs and (word_in, word_out) pairs.
✅ STDP+reward as the learning signal — same as visual cortex.
✅ Bounded vocabulary: ~16-32 tokens initially (action words + simple
   spatial/object words).
✅ Single-token I/O — no sequences yet.
✅ Demonstrable test: agent learns to emit "north" when shown an image of
   the goal to its north.

## Out of scope (deferred to v2+)

❌ Multi-token sequences (would need temporal recurrence)
❌ Full LLM-grade vocabulary (~50K tokens)
❌ Compositional grammar (needs syntactic structure)
❌ Audio I/O (separate cochlea-like infrastructure, deferred to Cluster L)
❌ Embodied conversation across long contexts (would need episodic memory)

## Success metric (v1)

After training on ~500 (image, action_word) pairs:
- Agent emits the correct cardinal direction word ≥70% of the time when
  presented a fresh gridworld image.
- Agent correctly executes the cardinal action when presented a word
  embedding for that direction (no image — pure word-to-action).

This is a "talking parrot" benchmark — not deep linguistic understanding,
but a real bidirectional language ↔ action loop running on biology-grounded
spiking neurons.

## Architecture

```
                        TEXT INPUT
                            │
                  embed(token)  (use pre-trained word2vec / GloVe / Claude API)
                            │
                  embedding (256-dim float)
                            │
                            ▼ project (256 → 256 sparse drive)
              language_input  (256 plastic recurrent)  
                            │ plastic, gated "language_input_to_pfc"
                            ▼
                       PFC / cortex_X
                          (existing infrastructure)
                            │ plastic
                            ▼
                       motor / cortex_it (action selection or visual)


                       VISUAL INPUT (existing)
                            │
                          retina → V1 → V2 → IT (Cluster K v2)
                            │ plastic, gated "it_to_language"
                            ▼
              language_output  (256 plastic recurrent)
                            │
                  readout: top-k → token distribution
                            │
                        TEXT OUTPUT
```

## Components

### 1. Pre-trained embedding wrapper

`sim/text_embeddings.py`:
- Load a 256-dim subset of GloVe embeddings (~50K most common English
  words). One-time load, cached on disk.
- `embed(token: str) -> np.ndarray[256]`: returns embedding.
- `nearest_token(activity: np.ndarray[256], k=1) -> List[str]`: cosine-
  match the agent's output activity vector to nearest known token(s).

Alternative: Claude API embeddings (`voyage-3` or similar). Simpler
first cut — just use a frozen pre-trained model. Doesn't matter what we
choose; it's a fixed encoder.

### 2. Language regions in build_bg_brain_regions

```python
if enable_text_io:
    regions.append(BrainRegion(
        name="language_input", n_neurons=256,
        exc_fraction=0.8, internal_density=0.05,
        exc_weight_mean=2.0, inh_weight_mean=4.0,
        weight_jitter=0.2, plastic_internal=True,
        izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
    ))
    regions.append(BrainRegion(
        name="language_output", n_neurons=256,
        exc_fraction=0.8, internal_density=0.10,
        exc_weight_mean=2.0, inh_weight_mean=4.0,
        weight_jitter=0.2, plastic_internal=True,
        izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
    ))

    # language_input -> PFC and cortex (plastic, learns word-to-action)
    pathways.append(RegionPathway(
        from_region="language_input", to_region="dlpfc_wm",
        density=0.20, weight_mean=2.0, weight_jitter=0.5,
        plastic=True, plasticity_gate="language_input_to_pfc",
    ))
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region="language_input", to_region=f"cortex_{action}",
            density=0.20, weight_mean=0.0, weight_jitter=0.0,
            plastic=True, plasticity_gate="language_input_to_cortex",
        ))

    # IT -> language_output (plastic, learns image-to-word)
    if enable_visual_cortex:
        pathways.append(RegionPathway(
            from_region="cortex_it", to_region="language_output",
            density=0.20, weight_mean=0.0, weight_jitter=0.0,
            plastic=True, plasticity_gate="it_to_language",
        ))
```

### 3. Bridge integration

Two helpers needed:
- `bridge.set_token_drive(token: str, drive_pA: float = 200.0)`:
  embeds the token via the pre-trained encoder, projects 256-dim
  embedding onto language_input neurons via a fixed projection matrix
  (e.g., identity if vocab=256, or random projection for vocab<256).
- `bridge.read_language_output(top_k: int = 1) -> List[str]`:
  reads cp_firing_states for language_output, computes mean firing
  rate per neuron over a readout window, projects back through the
  pre-trained encoder via cosine similarity to find nearest tokens.

### 4. Training pipeline

A new runner: `research/runners/text_train.py` — analogous to
`g11_bg_trajectory_train.py` but for text training.

**Three supervised regimes:**

**A. Image → word (visual labeling):**
For each (image, target_word) pair:
1. Render image, drive retina (existing K v2 path).
2. Clamp language_output to target_word's embedding pattern at high rate
   (~50 Hz) via cp_external_input_current.
3. Run stim window (100 ms).
4. Reward = +1 (the supervisor approves).

STDP grows IT → language_output weights for IT patterns that co-activate
with the target word. After training, IT → language_output is the
visual classifier.

**B. Word → action (verbal command following):**
For each (word, target_action) pair:
1. set_token_drive(word) — drives language_input.
2. Drive cortex_{target_action} with strong heuristic-like current.
3. Run stim window.
4. Reward = +1.

STDP grows language_input → cortex_{target_action} for word-action pairs.

**C. Action → word (action verbalization):**
Inverse of B. Drive cortex_{action}, clamp language_output to corresponding
word, reward. Agent learns to "say what it just did".

### 5. Interactive demo

A new mode in g11_bg_runner: `--interactive-text-mode`.

Loop:
1. Read user input from stdin (or interactive_control_file's "text_input"
   field).
2. Embed → set_token_drive on language_input.
3. Run a few hundred sub-steps.
4. Sample agent's response: read language_output, pick top-k tokens, print.
5. If user typed an action word ("north"/"east"/etc), agent should
   take that action in the gridworld.

Optional: route through Claude API. User types arbitrary English; Claude
maps to the agent's known vocabulary; agent responds in known vocabulary;
Claude explains the response in natural English.

## Phases

### Phase 1: scaffolding (1 day) ⭐ start here

- [ ] `sim/text_embeddings.py` with GloVe-subset loader
- [ ] `enable_text_io` flag in `build_bg_brain_regions`
- [ ] language_input + language_output regions added
- [ ] basic pathways (language_input→PFC, IT→language_output)
- [ ] `bridge.set_token_drive()` + `bridge.read_language_output()`
- [ ] Unit tests: token drives correct neurons; readout returns
  correct token after fully clamped firing pattern

### Phase 2: training pipeline (1-2 days)

- [ ] `research/runners/text_train.py`
- [ ] Image → word supervision with STDP+reward
- [ ] Word → action supervision
- [ ] Action → word supervision
- [ ] Validate: after 500 image-word pairs, classification accuracy
  on held-out images >70% on cardinal directions

### Phase 3: interactive demo (1 day)

- [ ] `--interactive-text-mode` in g11_bg_runner
- [ ] stdin/stdout loop with token drive + readout
- [ ] Webapp integration: text I/O in live mode panel
- [ ] Optional: Claude API wrapper for natural-English bridge

### Phase 4: beyond v1 (deferred)

- [ ] Multi-token sequences via temporal recurrence
- [ ] Larger vocabulary (>1000 tokens)
- [ ] Compositional grammar emergence
- [ ] Episodic dialogue across multiple exchanges

## Risk register

| Risk | Mitigation |
|---|---|
| GloVe embeddings too high-dim for 256 neurons | Random projection 300→256, or use language model with 256-dim outputs |
| Single-token output too limiting for interesting demo | Phase 1 stays single-token; phase 4 adds sequences |
| STDP+reward too slow for word learning | Use higher reward magnitude, longer training (e.g., 5K supervised pairs) |
| Disrupting K v2 visual cortex flagship | All language regions are opt-in via `--enable-text-io`; default OFF |
| Catastrophic interference between word-action and image-word training | Train in separate phases (curriculum); use plasticity gates |

## Naming

Following existing conventions (Kandel 6e, Felleman & Van Essen):
- `language_input` (analogous to Wernicke's area / superior temporal gyrus)
- `language_output` (analogous to Broca's area / inferior frontal gyrus)

These are oversimplifications of real cortical language regions but
match the level of biological abstraction we use elsewhere (e.g.,
`cortex_v2` is a single region representing what's actually V2, V3, V4,
TEO, etc.).

## What success looks like

A 30-second interaction demo:

```
USER: north
[agent moves agent up one cell]
USER: what do you see?
AGENT: goal
USER: how do you reach it?
AGENT: north
[agent moves up]
AGENT: goal
[goal reached]
```

The agent's vocabulary is bounded; the responses are biologically-grounded
neural firing patterns decoded to nearest token. The understanding is
shallow but real — it's not a hardcoded chatbot; it's STDP+reward-trained
visuomotor associations that happen to use language as one channel.

This unlocks downstream Cluster L (audio), multimodal grounding, and
eventually compositional language. The architecture is the same one we
just validated with K v2 — the visual cortex template generalizes.
