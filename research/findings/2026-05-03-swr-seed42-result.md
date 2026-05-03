# SWR Phase 3 replay — seed 42 result + ongoing seed 43

**Date:** 2026-05-03
**Status:** preliminary, n=1 (seed 43 in flight)
**Run:** `text_eval_v2_swr500_seed42.json`
**Config:** v2 baseline (Hebbian off, stdp_w_max=5, readout init=0.5) + curriculum: phase1=0, phase2=100ep, phase3=500 SWR replay events, replay_correct_only=True

---

## Headline

| Direction | v2 baseline (seed 42) | v2 + SWR (seed 42) | Δ |
|---|---|---|---|
| **I→W** | 33.0% | **39.0%** | **+6.0 pp** |
| **W→A** | 27.0% | **22.0%** | **−5.0 pp** |

For comparison the **6-seed v2 baseline** gave I→W 25.3% / W→A 28.5% (p=0.027).

So SWR replay at seed 42:
- Boosted the long-cascade `image → V1 → V2 → IT → language_output` pathway by 6 pp
- Hurt the direct `language_input → motor` PFC-bypass pathway by 5 pp

This is the **opposite** of the prior expectation (that SWR would consolidate
correct (token, action) pairs and improve W→A). With n=1 the result is too
noisy to call a real effect, but the directionality is interesting.

## Why the directional split might be real

Two competing hypotheses for why SWR would help I→W but hurt W→A:

1. **Long cascade benefits from re-presentation.** I→W requires the agent
   to recognize the gridworld scene (retina → V1 → V2 → IT) and emit the
   correct word at language_output. The multi-stage perceptual cascade has
   plenty of room for STDP to refine intermediate representations. Replay
   events re-present `(image, correct_word)` pairs which let cortex_X →
   language_output and IT → language_output continue learning even though
   no real-world reward arrives during the SWR phase.

2. **Direct pathway gets disrupted by replay noise.** W→A relies primarily
   on the `language_input → motor_X` PFC-bypass pathway (mimicking
   Wernicke→arcuate→Broca→M1 anatomy). This pathway is *short* — the
   correct mapping is essentially "word X → motor X" with one hop. Replay
   that is biased toward correct moves still carries the cascade's
   underlying north-bias (correct moves are mostly N because the cascade
   selects N more often), so over 500 replay events the language→motor
   weights are pushed toward N rather than the balanced 4-way mapping.

Hypothesis 2 makes a testable prediction: **the W→A regression should
be most visible in the north column of the confusion matrix**, with
non-north words being mis-classified as N more often than in baseline.

### Per-direction breakdown (seed 42, post-SWR)

From the new "per-direction accuracy" bars in the Language tab:

**I→W per-direction**:
- north: 27% (close to chance — baseline was likely above)
- east: 18% (below chance)
- south: 44% (above chance)
- west: 58% (well above chance)

**W→A per-direction:** (need to read confusion matrix)

If hypothesis 2 holds, we'd expect W→A's per-direction breakdown to show
N stealing from other directions. Pending re-eval.

## What's next

1. **Seed 43 SWR run is in flight** (Phase 2 ep 60/100 as of 01:14 EDT).
   Should finish ~01:50.
2. **If seed 43 also shows I→W↑ / W→A↓**, the asymmetry is real and we
   should investigate the replay distribution to see if it's biased
   toward already-frequent actions (which would amplify N).
3. **If seed 43 is mixed**, n=1 noise dominates. Need 6 seeds to
   conclude anything.
4. **Possible mitigation**: balance the replay distribution by per-token
   instead of per-event — sample 125 replay events per direction word
   instead of 500 weighted by training distribution.

## Configuration archaeology

The seed 42 config was launched via:

```bash
python -m research.runners.text_train_curriculum \
    --seed 42 \
    --phase1-episodes 0 \
    --phase2-episodes 100 \
    --phase3-replays 500 \
    --stim-steps-per-step 200 \
    --reset-steps 100 \
    --out-stats research/findings/raw/g11_bg/text_eval_v2_swr500_seed42.json
```

Phase 2 elapsed: 3328.8s = 55.5 min
Phase 3 elapsed: 976.1s = 16.3 min
Total: 71.8 min

(Seed 43 launched with identical config except `--seed 43`.)

## Webapp surfaces (2026-05-03 work)

This finding can now be browsed via:

- http://localhost:8765/#tab=language&run=text_eval_v2_swr500_seed42.json
  — full detail with confusion matrices + per-direction bars
- http://localhost:8765/#tab=language — aggregate across all 31 text I/O runs
  (best W→A 35% from R5_delta seed 42 still leads; seed 42 SWR is at 39%
  I→W which leads that direction)

The Brain tab now has:
- Camera presets (BG cascade view shows the 4-action lanes clearly)
- Pathway type toggles (excitatory/inhibitory/dopamine)
- Hover tooltip + click pinned info panel for any region
- Multi-live-run picker if more than one detached run is in flight
- Live scrubber that walks past progress samples while a run is running

Multi-seed launcher in Lab makes it trivial to dispatch the 6-seed
validation set:
```
Seed(s): 42,43,44,100,101,102
```
