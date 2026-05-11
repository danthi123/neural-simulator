# P3 — SWR sequential replay design

**Date:** 2026-05-11
**Phase:** P3 of realigned plan v3
**Roadmap entry:** T1.B (Month 2, 2-3 weeks)
**Catalog entries:** D.19 (SWRs), N.04 (ripple-coupled replay), J.07
(LTP-mediated consolidation transfer)

## Goal

Augment the existing SWR sleep-replay infrastructure
(`run_swr_replay_phase` in `consolidation_trainer.py`) to consolidate
**engram-tagged concepts** from hippocampus to cortex during NREM
windows.

The existing replay drives RANDOM sparse CA3 patterns (~15% of CA3
per burst, ~150Hz). That's enough for the existing Phase 1.3
consolidation (which transfers word→motor bindings to cortex) but
gives only random consolidation signal — no specific replay of the
DAY'S learning.

P3 adds:
1. **Concept replay** (cheap, uses P2 engram tags) — drive specific
   tagged CA3 ensembles repeatedly during NREM. Each engram gets
   K replay events. Consolidates that specific concept to cortex.
2. **Sequence replay** (deferred; needs sequence tracking) —
   time-compressed playback of waking spike sequences. Per roadmap:
   10-20× compression vs waking.

P3.1 (concept replay) ships first. P3.2 (sequence replay) deferred
until we have actual sequences worth replaying (P4 episodic encoder).

## What's already there

`run_swr_replay_phase(bridge, n_swr_events, swr_drive_pA, ...)`:
- Picks random 15% of CA3 per burst
- Drives at 100 pA for 100 ms burst (~150 Hz population firing)
- 50 ms inter-burst quiet
- `set_sleep_gates` enables ca3_swr_burst plasticity, freezes
  language_input_to_motor

This drives RANDOM sparse activity. The hippo→cortex consolidation
pathway STDPs against whatever fires together.

## P3.1: Concept replay (concrete plan)

### Design

Add `run_concept_replay_phase(bridge, tag_names, ...)` to
`consolidation_trainer.py`:

```python
def run_concept_replay_phase(
    bridge,
    tag_names: list[str],
    n_replays_per_tag: int = 20,
    burst_duration_ms: int = 100,
    inter_burst_ms: int = 50,
    drive_pA: float = 100.0,
    randomize_order: bool = True,
    rng=None,
):
    """During NREM, drive each engram-tagged ensemble in turn.

    Each replay: drive the tag's neurons (via bridge.stimulate_tag)
    at drive_pA for burst_duration_ms (~150 Hz population firing),
    then quiet for inter_burst_ms.

    STDP at downstream pathways (ca1->motor, ca1->lang_output,
    ca1->cortex_X) auto-consolidates that specific concept's
    hippocampal trace into cortex.

    Compared to run_swr_replay_phase (random sparse CA3 drives):
    - Random replay consolidates whatever has been recently learned
      generically.
    - Concept replay consolidates specific tagged concepts.

    Use case: after binding "apple" via P1+P2 (hippo encoding + engram
    tag), run concept replay on "apple" for N cycles to consolidate
    to cortex. Subsequent recall ("what's an apple?") works from
    cortex without needing hippocampus.
    """
    if rng is None:
        rng = np.random.default_rng()
    rep_count = 0
    order = list(tag_names) * n_replays_per_tag
    if randomize_order:
        rng.shuffle(order)
    for tag_name in order:
        bridge.clear_tag_drive()  # zero everything first
        bridge.stimulate_tag(tag_name, drive_pA=drive_pA)
        for _ in range(burst_duration_ms):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.clear_tag_drive()
        for _ in range(inter_burst_ms):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        rep_count += 1
    return {"n_replays": rep_count, "tags_replayed": list(tag_names)}
```

### Validation

Test plan in `tests/test_concept_replay.py`:

1. **Test consolidation transfers concept to cortex.**
   - Build hippo bridge.
   - Encode 2 concepts via lang_input drive while recording CA3
     ensembles via `start_engram_recording`. Commit as "apple" and "river".
   - Snapshot ca1→motor pathway weights as `weights_pre`.
   - Run `run_concept_replay_phase(["apple", "river"], n_replays=20)`.
   - Snapshot ca1→motor pathway weights as `weights_post`.
   - PASS: weights_post differs from weights_pre by > some threshold
     (consolidation actually changed cortex).

2. **Test selective consolidation.**
   - Encode "apple" and "river"; tag both.
   - Run concept replay on "apple" only (10 cycles).
   - Measure cortical recall for both. PASS: apple recall stronger than
     river (since only apple was replayed).

3. **Test concept replay doesn't disrupt unrelated learning.**
   - Encode "apple"; tag.
   - Bind a separate word "banana" via direct lang→motor (no hippo).
   - Run concept replay on "apple".
   - Recall "banana". PASS: banana recall unaffected.

### Wall-clock budget

~5-10 min per validation run on RTX 3090 (encoding + sleep + recall
tests). Multi-seed: ~30-60 min for 6 seeds.

## P3.2: Sequence replay (deferred)

Per roadmap, requires:
- Place-cell-like sequences recorded during waking trajectories
- 10-20× time compression during NREM replay
- Phase-locked to slow oscillation surrogate

Substrate exists (`set_sleep_gates` + `run_swr_replay_phase`);
what's missing is the SEQUENCE recording. The natural primitive:

```python
bridge.start_sequence_recording("trajectory_1", regions=["ca1"])
# Drive sequential inputs (e.g. word_1, word_2, word_3 over time)
for word in ["hello", "world", "today"]:
    bridge.set_token_drive(word)
    for _ in range(100):
        bridge._run_one_simulation_step()
recording = bridge.commit_sequence("trajectory_1")
# recording.spike_times: dict of neuron_idx -> list of step_idx
# recording.window_ms: total duration

# Later, during NREM:
bridge.replay_sequence("trajectory_1", compression=15.0)
# Compresses the recorded waking sequence by 15x for replay
```

Deferred until P4 (episodic encoder) produces sequences worth
replaying.

## Open questions

1. **How long should `n_replays_per_tag` be?** Real biology: hundreds
   of ripples per NREM cycle. Empirical tuning.

2. **Should concept replay be paced by slow oscillation surrogate?**
   Roadmap says yes for sequence replay; concept replay may not need
   it. Defer to empirical.

3. **Plasticity gates during concept replay?**
   - `ca3_swr_burst`: ON (CA3 recurrents learn from co-activation
     during burst)
   - `ca1_to_motor`: ON (cortical consolidation)
   - `language_input_to_motor`: OFF (don't disturb awake-learned
     direct bindings)
   - The existing `set_sleep_gates` does this; just call it.

4. **Engram tag lifetime?** Real biology: hippocampal engrams can
   become independent of hippo after consolidation (Tonegawa 2014).
   In our sim: after enough concept replay cycles, the cortical
   pathway should be strong enough that "apple" recall works even
   with CA3 silenced. This is a quantitative test, not architectural.

## Sequencing

```
P1 multi-seed PASS (in flight)
    ↓
P3.1 concept replay (1 week implementation + validation)
    ↓
Liu-2012-style causal recall test using engram tags
(roadmap T1.C validation criterion — currently pending)
    ↓
P4 episodic encoder (gates on relational binding)
    ↓
P3.2 sequence replay (if/when P4 produces sequences)
```

P3.1 estimate: 2-3 days implementation + 1 week validation.

## Why this matters

Without consolidation, every concept lives only in hippocampus. Real
biology: consolidated memories survive hippocampal damage; new
memories don't (Patient HM). Our sim needs the same property for
"continual learning across sessions" — concepts encoded today should
be retrievable tomorrow from cortex, even if the hippo state has
shifted toward newer bindings.

P3 is the mechanism that makes the engram-tag substrate (P2) actually
*durable*.
