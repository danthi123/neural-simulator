---
type: plan
status: live
date: 2026-05-02
---

# 2026-05-02 — SWR consolidation integration with text I/O

After distributed-motor-pop architecture is validated/falsified, the next
biology-grounded direction is integrating sharp-wave-ripple (SWR) replay
with text I/O training. Cluster D v2 SWR infrastructure exists for
navigation (commit at design doc 2026-04-30-cluster-d-v2-swr-design.md);
this doc designs how to integrate it for text I/O specifically.

## Biology source

Wilson & McNaughton (1994) demonstrated hippocampal replay during sleep
reactivates waking trajectory patterns. Buzsáki (1986+) showed sharp-wave-
ripples are the substrate. O'Reilly's complementary learning systems
(1995) frames replay as cortex-hippocampus consolidation: hippocampus
captures fast/episodic; replay during sleep transfers to slow cortex.

For language: Diekelmann & Born (2010) show sleep-dependent consolidation
of declarative memory. Word learning specifically is consolidated during
sleep (Tononi & Cirelli "synaptic homeostasis" hypothesis).

## Current state

We have Cluster D v2 (`--enable-cluster-d-v2-swr`):
- DG, CA3, CA1 regions wired in trisynaptic loop
- CA3 internal recurrent for autoassociator
- SWR burst detection in CA3 firing rate
- Plasticity gate `ca3_swr_burst` opens during burst windows

This works for NAVIGATION (CA1 → place_cells → cortex_X → motor_X) but
NOT integrated with text I/O training.

## What text I/O integration requires

Goal: during sleep windows, replay recent (token, action, reward) tuples
to consolidate language→motor mappings via STDP+reward.

Requires:

1. **Recording**: capture (target_word, motor_output, reward) during waking
   text training. Stored as a circular buffer of last N events.

2. **Replay drive**: during sleep, randomly select a recent (token, action)
   pair and drive language_input + motor pool patterns to recreate the
   waking-state firing.

3. **SWR-gated plasticity**: language pathway plasticity gates
   (language_input_to_motor, cortex_to_language_output, etc.) are OPEN
   during SWR bursts only. Outside SWR, gates are at low gain (0.1).

4. **Compression**: replay 10-20× faster than waking (real biology).
   Implementation: shorter stim windows during replay (50ms vs 100ms).

5. **Multiple replays per sleep**: each sleep window does 5-10 replay
   events to consolidate multiple recent experiences.

## Scope of changes

### text_train_curriculum.py (extend)

Add an optional Phase 3 between Phase 2 (text training) and final eval:

```
Phase 1 (visuomotor): 200 ep — cascade competence
Phase 2 (text I/O):   100 ep — language pathway formation
Phase 3 (sleep replay): 50 sleep cycles — SWR consolidation [NEW]
Eval                  — same as v2
```

Phase 3 dynamics:
- Each "sleep cycle" = 1 episode-worth of replay activity (30 events)
- Each event = randomly sampled (token, action) from last 100 episodes
- Drive: language_input pattern for token, motor pool pattern for action
- Stim window: 100ms (compressed from waking 200ms)
- SWR burst trigger: occasionally inject high CA3 drive to open plasticity gate
- Reward: positive for the original action selected (replay reinforces learned)

### research/runners/text_train_curriculum.py

```python
def _run_swr_replay_phase(
    bridge, cp, rng,
    n_sleep_cycles: int,
    events_per_cycle: int,
    waking_buffer: list[tuple[str, str]],  # (token, action) tuples
    ...
):
    # For each sleep cycle:
    for cycle in range(n_sleep_cycles):
        for event in range(events_per_cycle):
            # 1. Random sample from recent waking experience
            token, action = rng.choice(waking_buffer)
            # 2. Trigger SWR burst (high CA3 drive)
            bridge.cp_external_input_current[ca3_idx] = 200.0  # burst trigger
            # 3. Drive language + motor to recreate waking pattern
            bridge.set_token_drive(token, drive_pA=200.0, sparsity=0.1)
            # motor pool drive: small amount in target pool
            # 4. Run shorter stim window (compressed time)
            # 5. Reward = the stored reward (or +1 for re-entering)
            bridge.core_config.current_reward_signal = 1.0
            # 6. Run reward window for STDP application
```

### Integration with distributed-motor-pop

If distributed-pop architecture is validated, replay should drive
sub-pools matching the action's preferred direction (cosine-tuned drive
pattern). For target action=N, drive motor_pop_N strongly + motor_pop_NE
+ motor_pop_NW with cos-weighted amplitudes.

## Hypothesized outcomes

If SWR consolidation works:
- Recent (token, action) co-occurrences strengthened more than rare ones
- Per-direction variance should DECREASE (multi-seed std drops from ~5pp to ~2pp)
- Mean accuracy increases by 3-5pp via consolidation of correct pairings
- "Lucky direction" issue partially mitigated (consolidation strengthens
  whatever direction got randomly successful in waking)

If SWR doesn't help:
- Cluster D infrastructure may not produce the right replay patterns
- May need explicit pretraining of the lang→motor mapping rather than
  hoping SWR captures it correctly

## Risk assessment

- **Cluster D infrastructure overhead**: enables hippocampus regions
  (DG, CA3, CA1, place_cells) which add ~500 neurons. Slows training.
- **SWR burst trigger**: needs tuning to actually elicit bursts at our
  scale. Cluster D v2 has burst detection already.
- **Replay buffer**: needs ~100 entries × small size. Easy.

## Test plan

1. Smoke (5 ep × 10 steps + 3 sleep cycles) — verify no crash
2. Single seed=42 full run — compare to v2 baseline
3. If positive, multi-seed validation
4. If negative, document and explore other directions

## Effort estimate

Implementation: 3-5 hours
- 1 hr: extend text_train_curriculum with phase 3 + replay buffer
- 1 hr: SWR-aware drive scheme during replay
- 30 min: integration with cluster D infrastructure (existing flags)
- 30 min: smoke test + debugging
- 1 hr: full single-seed run + analysis

## Files to modify

- `research/runners/text_train_curriculum.py` — add phase 3 replay
- `research/runners/text_eval.py` — already supports cluster D regions
- `research/runners/text_eval_curriculum.py` (if needed) — CLI for new phase
- Possibly: instrument SWR burst counter in runner

## What we're NOT doing in v1

- Slow-oscillation phase locking (Option 3 of cluster D v2 design)
- REM-specific replay (just NREM-like for now)
- Multi-day consolidation cycles (single sleep phase)

These are deferrable refinements. v1 focus: does basic SWR-gated replay
of recent (token, action) tuples meaningfully improve text I/O accuracy
or stability across seeds?

## Decision gate

If distributed-motor-pop validates (>30% W→A), SWR replay is a COMPOSE.
Run on top of distributed-pop architecture.

If distributed-motor-pop fails, SWR replay tested on v2 baseline.
Different mechanism (consolidation, not coding), independent of dpop result.

Either way, SWR is the next experiment after distributed-pop result is in.
