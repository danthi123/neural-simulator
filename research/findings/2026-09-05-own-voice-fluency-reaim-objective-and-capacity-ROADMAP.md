---
type: finding
status: design
claim_check: synthesis
date: 2026-09-05
mechanism: ROADMAP — re-aim the own-voice-fluency arc after the content-addressing direction is exhausted: the next levers are the training OBJECTIVE (predictive-coding auxiliary, --pred-aux-weight, already built) and CAPACITY, NOT another key/attention variant
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: >
  ROADMAP / re-aim (no new measurement). Three content-addressable attention families now fail on the deployable
  simplewiki protocol (assoc -0.347, assoc_t -0.147, hippokey -0.284 deep, 6/6 NO-GO), with the richest key
  (multi-timescale HiPPO) the WORST — so "a better content-addressable memory" is banked exhausted as the next
  lever. The single deployable mouth that CROSSES a fair trigram is linattn (+0.0505 6/6 simplewiki), but it FALLS
  BELOW the trigram on the BROAD wikitext103 domain — the actual capability that retires Qwen. The record's own
  strongest same-budget external evidence points at the training OBJECTIVE, not the memory mechanism: at a matched
  10M-word budget a causal+masked hybrid objective hits BLiMP 0.794 vs a tuned n-gram 0.633 and a plain causal
  LSTM 0.661 (recurrence/attention alone barely ties the n-gram — this arc's exact failure mode). The
  causal-compatible port ALREADY EXISTS (--pred-aux-weight). Re-aimed ladder: (1) linattn + predictive-objective,
  (2) capacity (depth/width) on linattn, both on simplewiki then the broad wt103 domain; NOT another
  content-addressing arm. The own-voice mouth stays the #1 goal-blocker (~48/64 one-brain ledger rows).
lane_wall: brain-native open-ended generation (own-voice mouth) — roadmap Wall #7 / R4
external: >
  Rao & Ballard 1999 ("Predictive coding in the visual cortex", Nat Neurosci 2:79-87); Friston hierarchical
  predictive processing — cortex continuously predicts upcoming input at multiple horizons, the bio anchor for the
  multi-horizon auxiliary objective. The dominant-lever datapoint (causal+masked hybrid BLiMP 0.794 vs n-gram
  0.633 vs causal LSTM 0.661 at a matched 10M-word budget) is from the ordered-attention bound-investigation's
  external round (2026-09-03). Hoffmann et al. 2022 (Chinchilla, arXiv:2203.15556) + the token-supply lever
  (2026-09-01) ground the capacity x tokens axis. Same external round that grounded this lane 2026-09-01/03.
artifacts:
  - research/runners/_emerge_wkv_lm_derisk.py
  - research/findings/raw/_emerge_wkv_lm_hippokey_depth2_contiguous_6seed.json
  - research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json
  - research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json
  - research/findings/2026-09-05-hippokey-hippo-ssm-content-addressable-attention-NO-GO.md
  - research/findings/2026-09-03-ordered-attention-at-shared-fluency-bound-investigation-verdict.md
  - research/findings/2026-09-01-generative-cortex-token-supply-lever-broad-domain-plateau-is-starvation-not-capacity-wall.md
runner: research/runners/_emerge_wkv_lm_derisk.py
---

# Own-voice fluency re-aim: after content-addressing is exhausted, the levers are OBJECTIVE and CAPACITY (not another attention variant)

**Context:** the own-voice mouth is the #1 goal-blocker (it blocks ~48/64 one-brain ledger rows: retiring the
Qwen scaffold needs the brain's own mouth fluent about arbitrary topics). This note re-anchors the arc after the
hippokey NO-GO (`research/findings/2026-09-05-hippokey-hippo-ssm-content-addressable-attention-NO-GO.md`) closed
the content-addressing direction.

## 1. What is now banked (the direction that is exhausted)

Three content-addressable attention families have been run on the identical deployable simplewiki depth-2
protocol and all lose to a fair trigram at deep context: `assoc` (-0.347), `assoc_t` (-0.147), and now
`hippokey` (-0.284 deep, 6/6). The richest key — a multi-timescale HiPPO context code, the July "give it a better
key" prescription taken literally — was the WORST, because next-token prediction needs sharp token-identity
matching and the HiPPO code low-passes exactly that. ⇒ "a better content-addressable memory / a richer key" is
banked as NOT the missing piece. The deep-bucket numbers backing this live in
`research/findings/raw/_emerge_wkv_lm_hippokey_depth2_contiguous_6seed.json` (hippokey) and
`research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json` (the linattn crossing).

## 2. The current best + the real target

The one deployable mouth that CROSSES a fair trigram is `linattn` (normalized Hebbian fast-weight linear
attention, +0.0505 6/6 on simplewiki). But on the BROAD wikitext103 domain it FALLS BELOW the trigram
(`research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json`, margin -0.29..-0.57 at depth>=2). Broad-
domain fluency IS the capability that retires Qwen, so linattn — not a new attention arm — is the substrate to
push, on the axis that actually moves it.

## 3. The re-aimed ladder (named next mechanisms, in order)

1. **Predictive-coding OBJECTIVE — ALREADY BUILT (`--pred-aux-weight`), the cheapest decisive next run.** The
   ordered-attention bound-investigation's strongest same-budget external datapoint says the training OBJECTIVE,
   not the memory mechanism, is the dominant lever below ~20M tokens (causal+masked hybrid BLiMP 0.794 vs n-gram
   0.633 vs causal LSTM 0.661). The causal-compatible port is in the runner: multi-horizon further-ahead
   auxiliary read-out heads on the shared hidden state (Rao & Ballard 1999 predictive coding), strictly causal,
   discarded at generation. **Next run:** `linattn --pred-aux-weight <w>` 6-seed on the simplewiki protocol
   (does a richer objective push the already-crossing mouth further?), then on wt103 (does it lift the mouth off
   the broad-domain sub-trigram bound?). It composes with linattn unchanged and is cheap.
2. **CAPACITY (depth then width) on linattn.** The token-supply lever (2026-09-01) showed the substrate scales
   with capacity-matched tokens (deep-context NLL still descending at 4.5 tok/param, 6/6); a small from-scratch
   mouth on ~13.5M BPE tokens is capacity-limited. Sweep d_model / n_layers on linattn at simplewiki then wt103
   to locate where it crosses on the broad domain (Chinchilla, arXiv:2203.15556).
3. **NOT another content-addressing / key / attention variant** — banked exhausted by the hippokey NO-GO (§1).

## 4. No-defer note

The capability (brain-native arbitrary prose) is not deferred; a wall defers a METHOD. The content-addressing
method is banked, and the next methods (objective, then capacity) are named, cheap-first, and grounded in the
record's own external evidence. The own-voice mouth remains the #1 goal-blocker and the highest-leverage arc.
