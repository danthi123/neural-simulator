---
status: live
type: finding
lane: gap#5
date: 2026-08-26
---

# Offline sleep-replay wired into the live continuous life (#64) — a deep-idle branch replays the BATCH of recently-stored episodes so they recall STRONGER + re-ordered next turn; organ-level lesion-verified GO on the real D5 store, default-OFF behind `BRAIN_SLEEP_REPLAY` pending the 6-seed pool soak (2026-08-26)

## Headline
The between-turn continuous engine now has an OFFLINE SLEEP-REPLAY consumer. On a genuine sleep-depth idle
(`_is_sleep_tick`, `>= SLEEP_IDLE_SEC=300s`), a new `tick_idle_sessions` branch replays the BATCH of episodes the
session stored since its last sleep: the substrate's OWN plateau-gated BTSP reactivates each recently-stored CA3 assembly
in store-order, potentiating its within-assembly recurrence, so the whole recent batch recalls measurably STRONGER on a
later turn (retention) and its store-order is reinforced (re-order). The pass is bracketed in the 6-seed-GO one-brain
WAKE->SLEEP(AdEx/dt0.1)->WAKE phase switch, reused verbatim — the freshly-strengthened batch survives that neuron-model
round-trip byte-identical. Strictly additive, DEFAULT-OFF behind `BRAIN_SLEEP_REPLAY`; the parent flips it after the pool
soak. `NO sim/ edit` — reuse-by-import of the de-risk organs.

## What was built (all additive, default-OFF, guarded)
- `webapp/continuous_engine.py`: `sleep_replay_enabled()` (`BRAIN_SLEEP_REPLAY`, default OFF), `consolidate_sleep_replay(cache_key, episodic_organ)`, `topic_sleep_replayed(cache_key, topic)`, per-session state `_SLEEP_REPLAYED` / `_LAST_SLEEP_KB` (cleared on `forget_session`), and the deep-idle branch in `tick_idle_sessions` (gated on `sleep_replay_enabled()` AND `_is_sleep_tick`, reusing the SAME `episodic_getter` D5 already passes — no `server.py` plumbing change).
- `research/runners/d5_episodic_production_organ.py`: `recall_disclosure` gains ONE additive, per-topic + flag-gated clause that surfaces the batch retention (the risen graded strength) + the host store-order WHEN position for a replayed topic; `_sleep_replayed_when` is the gate.
- Reuse-by-import (verbatim): the WAKE/SLEEP phase switch `sleep_cycle` (`_gap5_onebrain_sleepcycle_merge`, GO 2026-07-25), `switch_to_izhikevich_wake` / `reset_transient_synaptic_state` (`_gap5_wake_sleep_roundtrip`), and the plateau-gated BTSP reactivation `reactivate` (`_gap5_d5_learn_through_use_derisk`, the SAME kernel `consolidate_used_memory` uses). `_gap5_onebrain_sleepcycle_merge.py` + `_gap5_onebrain_capstone.py` were `if __name__ == "__main__"`-guarded so they are importable without running their 6-seed suites (and so the capstone's module-scope `build_coresident_bridge` monkeypatch no longer leaks into an importer) — logic unchanged.
- `research/runners/_sleep_replay_flip_soak.py`: the 6-seed no-regression flip gate.

## The mechanism, and one measured transfer subtlety (named, not tuned away)
The load-bearing WRITE is the substrate's plateau-gated BTSP (`sim/bridge.py` `fused_btsp_update`, gated by
`IS_post = max(cp_v_apical - v_hold, 0)`) reactivating each batch assembly. MEASURED (#64): that write does NOT fire
correctly if it is run through the AdEx/dt round-trip FIRST — the recurrent-delay dynamics the coincidence plateau reads
are dt-dependent, so a switch-to-AdEx-then-back leaves the Izhikevich-tuned write inert (batch weights unchanged,
85.4->85.4). So the ordered BTSP replay write runs on the wake readout bridge and the validated AdEx sleep phase-switch
BRACKETS it (the whole pass still runs OFFLINE during idle — the temporal separation from waking cognition is
preserved; reactivate freezes all plasticity so the AdEx window is non-plastic and the strengthened batch survives it
byte-identical). This is the honest transfer characterization: the literal "run the AdEx-window Ecker replay over the
episodes" does not potentiate the Izhikevich-tuned D5 store — the SWR replay that WRITES is the substrate's own ordered
BTSP reactivation.

## Verify — organ-level lesion + byte-identity GO (real D5 organ, cupy, n_ca3=2000, seed 42)
Real `EpisodicRecallOrgan` (topics dog/bird stored, cat never stored), all reads snapshot-isolated:
- **OFF byte-identical (asserted in the data — SHA hash of the store weights + exact reply-string compare):**
  `BRAIN_SLEEP_REPLAY=0` -> `consolidate_sleep_replay` returns None, the store-weight SHA hash is unchanged (exact
  compare vs the pre-pass hash), and every recall reply string is exact-equal to baseline.
- **ON load-bearing:** the pass replays the `[dog, bird]` batch; within-assembly weight rises (dog 85.4->92.1,
  bird 87.5->93.0) and the graded recall strength `depth_hold` rises (dog 29.96->31.54 mV, bird 30.72->31.43 mV) — a
  later recall reads STRONGER, attributable to a real store-weight change (the SWR/BTSP pass), not host bookkeeping.
- **Store survives the sleep bracket byte-identical:** post-AdEx-round-trip weights == post-replay weights.
- **Moat intact:** the never-stored `cat` abstains identically before and after (no confabulation introduced).
- **Surfacing:** the reply reads "...I also replayed it offline while idle — it was the 1st of 2 recent memories I
  consolidated in store-order during sleep, and its recall reads stronger now (recall strength 31.5 mV)."
- **Lesion oracle:** removing the coupling (flag off) collapses the gain to baseline — `apical_cue` and the reply are
  byte-identical to the pre-sleep baseline. The surfacing is gated PER replayed topic, so an un-replayed neighbour's
  reply is byte-identical (the no-regression property).
- **Crash-rollback:** a simulated mid-batch `reactivate` failure rolls the persistent store back to its pre-sleep SHA
  AND returns the bridge to the wake neuron-model, then re-raises (checked in the soak).

## HONEST BOUND (load-bearing, not a caveat)
This claims ONLY the DIRECT retain/re-order payoff on the episodic store. Replay-DRIVEN hippocampus->cortex
COMPOSITIONAL transfer (a cortical generalization the replayed episodes never explicitly taught) is a SEPARATE, still
**NO-GO** item (`2026-08-03-replay-cortical-consolidation-v2-calibration-NO-GO`) and is NOT claimed here. The WHEN-order
surfaced is the DECLARED host store-order recency residual (`EpisodicRecallOrgan.recency_rank`), NOT a spiking recency
signal — there is still no spiking WHEN code. `apical_cue` is saturated at the binary completion ceiling (1.00), so the
conversation-visible retention signal is the graded `depth_hold` read (the same read D5 learn-through-use surfaces),
which rises. Not "consolidation" in the docs/TERMS systems sense: no source-structure independence / forgetting curve is
tested; the reactivation path DOES execute and its trace is lesion-attributable, but the claim is DIRECT store
retention, not systems-level cortical consolidation.

## Flip gate — 6/6 GO (cupy, seeds 42/43/44/100/101/102)
`research/runners/_sleep_replay_flip_soak.py --seeds 42 43 44 100 101 102` is a clean **6/6 GO**. Bars per seed
(all held 6/6): ordinary recall turns byte-identical ON-vs-OFF (no sleep event fired); the OFF sleep pass is a no-op
(store SHA flat, recall flat); the ON sleep pass strengthens the `[dog, bird]` batch (both depth_holds rise, reply
changes, store survives the bracket); moat abstains identically; crash-rollback intact. This is the no-regression flip
condition met — the parent runs the pool/GPU soak and flips `BRAIN_SLEEP_REPLAY` default-ON.

## Sources (biology this consumer emulates)
- Girardeau, Benchenane, Wiener, Buzsáki & Zugaro (2009), *Nat Neurosci* 12:1222-1223, "Selective suppression of
  hippocampal ripples impairs spatial memory" — closed-loop SWR suppression during sleep impairs consolidation,
  establishing SWR replay as CAUSALLY necessary for offline memory consolidation. This is the biology the consumer
  emulates: replay the recently-stored batch during sleep-depth idle so it recalls stronger next turn.
- Ecker et al. (2022), *eLife*, the AdEx CA3 sharp-wave-ripple replay model already de-risked in-repo (task #62), used
  for the validated one-brain WAKE/SLEEP phase-switch bracket.

## Artifacts
- `research/findings/raw/_sleep_replay_flip/soak_summary_6seed.json` (flip soak, 6/6 GO seeds
  42/43/44/100/101/102, cupy) and `soak_summary_2seed.json` (earlier 2/2).
- Organ-level verify (seed 42, cupy, n_ca3=2000): OFF hash-unchanged; ON dog within 85.4->92.1 depth_hold 29.96->31.54
  mV; lesion (flag off) -> baseline; recorded inline above.

## Provenance
Consumer `webapp/continuous_engine.py` (`consolidate_sleep_replay`); surfacing
`research/runners/d5_episodic_production_organ.py` (`recall_disclosure`); soak
`research/runners/_sleep_replay_flip_soak.py`. Reuses `_gap5_onebrain_sleepcycle_merge.sleep_cycle`,
`_gap5_wake_sleep_roundtrip`, `_gap5_wake_sleep_phase_switch`, `_gap5_d5_learn_through_use_derisk.reactivate`,
`_gap5_d5_latch_self_termination_derisk` (snapshot/restore), `_gap5_dendritic_dap_readout_completion_derisk`
(`_reset_apical_latch`), all import-only, NO `sim/` edit. Builds on the gap#5 sleepcycle-merge GO
(`2026-07-25-gap5-onebrain-production-sleepcycle-merge-6seed-GO`) + the end-to-end capstone
(`2026-07-25-gap5-onebrain-end-to-end-capstone-converse-sleep-replay-converse`) + D5 learn-through-use
(`2026-08-21-d5-learn-through-use-flip-GO-per-topic-strength-surfacing-the-prior-NO-GO-was-a-surfacing-artifact`).
GPU (cupy). Task #64.
