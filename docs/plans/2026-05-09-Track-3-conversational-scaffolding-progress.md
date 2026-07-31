---
type: plan
status: live
date: 2026-05-09
---

# 2026-05-09 — Track 3 conversational scaffolding: progress + roadmap

Per the master plan, Track 3 is the biology-only conversational artifact —
independent of Phase 2 outcome. Builds on the validated Phase 1.4
BRANCH A + Phase 1.3 consolidation + Tier 2.1 8/12-word vocab
foundation.

## What's shipped (2026-05-09)

| Layer | Commit | Description |
|---|---|---|
| 1 | `f6c919c` | `chat_repl --learn` primitive — online word↔action binding via embodied-Hebbian co-firing on the trained bridge |
| 2 | `20ec1ce` | `chat_learn_demo` runner + webapp launcher surface + `chat_demo_aggregate` branch handles its multi-seed output |
| 3 | (just shipped) | Dialog state commands `:again` / `:opposite` / `:history` / `:forget` for multi-turn context |

Every layer is testable + tested:
- 5 parser tests for `_parse_learn_command`
- 6 parser tests for `_parse_dialog_command` + inverse-action table
- 3 aggregator tests for `chat_learn_demo` JSON shape

Total Track 3 test count: 14. All passing.

## What's NOT yet shipped

### Layer 4: generative decoder (action → word)

**The biological direction:** the existing `chat_inference` reads
language_input → motor (the W→A direction validated 5/6+6/6 aligned at
Tier 2.1). The reverse direction (A→W: action → word) was also
validated at 6/6 aligned with mean accuracy 63.7% in the Tier 2.1
BREAKTHROUGH. But chat_repl currently doesn't expose A→W as an
inference primitive.

**Design sketch:**

```python
def generative_inference(bridge, target_action, top_k=4, vocab=None):
    """Drive motor_<target_action>, read language_output, decode to word.

    Inverse of chat_inference. Tests whether the motor→language pathway
    has internalized the binding.

    Returns:
        list of (word, similarity_score) tuples, sorted descending by
        similarity. The top-1 is the network's "spoken" word for the
        given action; the full list lets us see runner-up activations.
    """
    rm = bridge.region_manager
    motor_arr = cp.asarray(list(rm.indices(f"motor_{target_action}")), dtype=cp.int64)
    lang_out_arr = cp.asarray(list(rm.indices("language_output")), dtype=cp.int64)
    n_lang_out = len(lang_out_arr)

    # Phase A: baseline
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    baseline = cp.zeros(n_lang_out, dtype=cp.int32)
    for _ in range(100):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        baseline += bridge.cp_firing_states[lang_out_arr].astype(cp.int32)

    # Phase B: motor-driven
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(50):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.cp_external_input_current[motor_arr] = MOTOR_DRIVE_pA  # ~1500
    drive = cp.zeros(n_lang_out, dtype=cp.int32)
    for _ in range(100):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        drive += bridge.cp_firing_states[lang_out_arr].astype(cp.int32)

    # Delta = action-driven minus baseline
    delta = (drive - baseline).get().astype(np.float32)

    # Decode: compute cosine similarity to each known word's drive pattern
    if vocab is None:
        vocab = ["north", "east", "south", "west"]  # default Tier 1
    rankings = []
    for word in vocab:
        word_pattern = vocab_to_drive_pattern(
            word, n_neurons=n_lang_out,
            drive_max_pA=200.0, sparsity=0.1,
        )
        sim = _cosine_similarity(delta, word_pattern)
        rankings.append((word, sim))
    rankings.sort(key=lambda x: -x[1])
    return rankings[:top_k]
```

**REPL integration:** new dialog command `:speak <action>` (or just
type an action letter prefixed with `>` like `>N`):

```
> :speak N
  [SPEAK] motor_N driven, language_output produces:
    1. north  (sim=0.81)
    2. up     (sim=0.74)
    3. east   (sim=0.12)
    4. south  (sim=-0.03)
```

This makes the chat genuinely BIDIRECTIONAL: the user can ask the
network to predict an action from a word, OR ask the network to
generate a word from an action. Both are validated at Tier 2.1.

**Testable parts (CPU-only, no GPU):**
- `_cosine_similarity(a, b)` — pure numpy
- `_rank_words_by_similarity(spike_pattern, vocab_patterns)` — pure
  ranking logic
- `_parse_speak_command(line)` — parses `:speak <action>` syntax

**GPU-bound smoke test:** run via `chat_repl --mode tier1
--scripted-words ":speak N,:speak E,north,:speak S"` to verify the
A→W direction works end-to-end against a freshly-trained bridge.

### Layer 5: bidirectional conversation (future)

True conversation: a turn alternates between user-says-word →
bridge-predicts-action AND bridge-says-word → user-confirms-action.
Builds on layer 4. The user could "drive" the conversation with
input words, and the bridge could "respond" with generated words
based on its internal motor activity.

### Layer 6: dialog context conditioning (future)

The brittle case: word "again" without context is meaningless. With
conversation history, "again" should re-execute the prior word's
action. Layer 3 (dialog state `:again`) handles this for explicit
commands; layer 6 would extend it to implicit context.

## Why layer 4 was deferred this iteration

- Phase 1.5 multi-seed (~7 hr remaining) is still validating the
  architecture foundation that layer 4 builds on. If interference
  benchmark fails at multi-seed, the architecture choice changes
  and layer 4 needs different decoding.
- Layer 4 needs a real bridge for end-to-end smoke test — wall clock
  ~10 min to train a Tier 1 bridge + run smoke. With the GPU busy
  on Phase 1.5 + the v400 chain queued, this iteration's wait window
  isn't ideal.
- The current iteration shipped 8 substantive commits (frontend audit,
  Track 3 layers 1-3, inflight + brain3d bug fixes, structured
  progress future-proofing). Diminishing returns on shipping more
  big features in this stretch.

Layer 4 is the natural next item for the post-Phase-1.5 iteration.

## Track 3 success criterion

Per master plan: "biology-only conversational artifact independent of
Phase 2 outcome." A conversation looks like:

```
[user]   north
[chat]   motor_N (delta N+15 E-2 S-1 W+1, x4.2)
[user]   :speak N
[chat]   "north" (sim=0.81)
[user]   :again
[chat]   motor_N (delta N+13 E-1 S+0 W+0, x3.8)
[user]   :opposite
[chat]   asking for opposite via 'south' → motor_S (delta N-1 E+0 S+11 W-1, x2.9)
[user]   learn ahead N
[chat]   [LEARN] 50 events; test: motor_N (bound OK)
[user]   ahead
[chat]   motor_N (delta N+10 E-2 S+1 W+1, x3.5)
[user]   :speak N
[chat]   "ahead" (sim=0.65)  ← network "remembers" the new word!
```

Layer 4 (`:speak`) is the missing piece for this script. After it
lands, Track 3 v1 is feature-complete: the network reads, writes,
remembers, and learns new words during the conversation, all via
biology-grounded mechanisms (STDP, embodied-Hebbian co-firing, no
backprop).

## Related

- Master plan: `docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md`
  ("Track 3 conversational scaffolding")
- Phase 2 resumption plan (parallel track):
  `docs/plans/2026-05-09-Phase-2-resumption-plan.md`
- Tier 2.1 BREAKTHROUGH (the validated A→W mean 63.7%):
  `research/findings/2026-05-06-Tier2.1-BREAKTHROUGH-synonym-binding-via-scale.md`
- chat_repl primitive:
  `research/runners/chat_repl.py` (`learn_word_pairing`,
  `_parse_learn_command`, `_parse_dialog_command`)
- chat_learn_demo runner: `research/runners/chat_learn_demo.py`
