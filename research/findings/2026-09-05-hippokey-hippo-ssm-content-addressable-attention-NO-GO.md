---
type: finding
status: negative
claim_check: measured
date: 2026-09-05
mechanism: hippokey — STRUCTURED HiPPO SSM -> CONTENT-ADDRESSABLE LEARNED-KEY ATTENTION (a FIXED HiPPO multi-timescale diagonal SSM produces a per-position multi-timescale context code x_s; a causal softmax read forms Q/K over x_s and V over the token content z), the literal owner steer for the next own-voice-fluency mechanism class
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [42, 43, 44, 100, 101, 102]
verdict: >
  NO-GO, 6/6. On the exact deployable simplewiki depth-2 contiguous protocol (only --recurrence linattn ->
  hippokey), the HiPPO-keyed content-addressable read LOSES to a fair interpolated trigram at every context
  depth >= 2 on all 6 seeds (deep bucket 10-99 negative on every seed; MORE negative at shallower buckets). It
  beats the trigram ONLY at the trivial depth-1 bucket. This is far below linattn's 6/6 crossing and WORSE than
  the plain token-keyed attention arms it was meant to surpass (assoc, assoc_t) — the exact numbers are in the
  derived-marked body table, traced to the cited artifact. Anti-cheats hold (memoryless and permute both collapse
  vs the model at deep bucket) — it genuinely uses context and order, it just is not better than a trigram at
  depth. THE VERDICT IS ON THE METHOD (a smoothed multi-timescale HiPPO code as the attention KEY), NOT on the
  capability: the "better key" hypothesis is FALSIFIED — a deeper/multi-timescale context key is a WORSE
  next-token key than the token-local one, so the content-addressing DIRECTION is exhausted (three attention
  families now fail, the fanciest key the worst). Re-aim (banked, named): the training OBJECTIVE
  (predictive-coding / richer targets, --pred-aux-weight, already built) and CAPACITY, NOT another
  content-addressing variant. The mouth stays the #1 goal-blocker.
lane_wall: brain-native open-ended generation (own-voice mouth) — roadmap Wall #7 / R4
external: >
  Gu, Dao, Ermon, Rudra, Re 2020, "HiPPO: Recurrent Memory with Optimal Polynomial Projections" (NeurIPS) — the
  multi-timescale diagonal state family. MacDonald et al. 2011 (Neuron 71:737-749) hippocampal time cells; Howard
  & Kahana 2002 (J Math Psychol 46:269-299) Temporal Context Model — the entorhinal multi-timescale context code.
  Ramsauer et al. 2020 ("Hopfield Networks is All You Need", ICLR 2021) — modern-Hopfield <-> softmax-attention.
  Marr 1971; Treves & Rolls 1994 — CA3 autoassociation. The NO-GO's re-aim rests on the ordered-attention
  bound-investigation's strongest same-budget external datapoint: at a matched 10M-word budget a causal+masked
  hybrid OBJECTIVE far outperforms a tuned n-gram and a plain causal LSTM on BLiMP (recurrence/attention alone
  barely ties the n-gram — exactly this arc's failure mode), pointing at OBJECTIVE, not memory-mechanism.
artifacts:
  - research/findings/raw/_emerge_wkv_lm_hippokey_depth2_contiguous_6seed.json
  - research/runners/_emerge_wkv_lm_derisk.py
  - research/findings/raw/_emerge_wkv_lm_linattn_depth2_contiguous_6seed.json
  - research/findings/raw/_emerge_wkv_lm_assoc_temporal_depth2_contiguous_6seed.json
  - research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json
  - research/findings/2026-09-03-OPEN-FLUENCY-BREAKTHROUGH-linattn-deployable-spiking-mouth-beats-trigram-6of6.md
  - research/findings/2026-09-03-ordered-attention-at-shared-fluency-bound-investigation-verdict.md
  - research/findings/2026-07-11-content-addressable-retrieval-needs-LEARNED-keys-the-arc-converges-on-deep-credit-learned-representations.md
runner: research/runners/_emerge_wkv_lm_derisk.py
---

# hippokey (structured HiPPO SSM -> content-addressable learned-key attention): NO-GO 6/6 — a better key was not the missing piece

**Artifact:** `research/findings/raw/_emerge_wkv_lm_hippokey_depth2_contiguous_6seed.json` (6-seed simplewiki
contiguous depth-2, provenance-stamped). This is the measured verdict on the mechanism scoped + built + queued
earlier this cycle (the doc was a ROADMAP note until the GPU run cleared; the run is now in).

## 0. Headline — NO-GO, and what it rules out

The literal owner steer for the next own-voice-fluency mechanism class was "a structured HiPPO-style SSM ->
content-addressable learned-key attention." Built as `--recurrence hippokey` (a FIXED HiPPO multi-timescale SSM
whose per-position state x_s KEYS a causal softmax read; V = token content z) and run 6-seed on the exact
deployable simplewiki protocol. **Result: NO-GO on all 6 seeds — it loses to a fair trigram at every context
depth >= 2, and is WORSE than both linattn and the plain token-keyed attention arms.** Per THE LAW this is a
verdict on THIS METHOD (a multi-timescale HiPPO code as the attention key), not on the capability: it **falsifies
the "better key" hypothesis** and, together with the two prior attention arms, **exhausts the content-addressing
direction** — the next lever is the OBJECTIVE / CAPACITY, not another key/attention variant.

## 1. Result (6 seeds, simplewiki, BPE V=8001, depth-2 contiguous, d_model=192)

(Per-seed values are rounded from the cited hippokey artifact's `per_seed[*].by_depth[*].margin_vs_trigram`; the mean is derived.)
<!--derived-->
Deep bucket (positions 10-99), per-seed `margin_vs_trigram` (trigram_nll - wkv_nll; positive = the mouth wins):
**-0.323, -0.286, -0.340, -0.210, -0.283, -0.260** (seeds 42/43/44/100/101/102) -> **mean -0.284, all 6 negative.**

margin_vs_trigram by context depth (mean over 6 seeds; the ONLY bucket the mouth wins is the trivial depth-1):

<!--derived-->
| bucket | 1 | 2 | 3 | 4-5 | 6-9 | 10-99 |
|---|---|---|---|---|---|---|
| mean margin_vs_trigram | +0.735 | -0.660 | -0.547 | -0.420 | -0.338 | -0.284 |

<!--derived-->
Deep-bucket absolutes (mean over seeds): hippokey NLL ~4.70, trigram ~4.42, bigram ~5.43; memoryless ~5.90
(collapse ~+1.2 vs the model), permute ~8.20 (collapse ~+3.5). So the anti-cheats HOLD — the read genuinely uses
content (memoryless is much worse) and order (permute is much worse) — the mechanism is real; it is simply not
better than a trigram at any depth that matters.

## 2. Where hippokey lands vs the family (it is WORSE, not better)

(linattn/assoc rows are the deep-bucket means in the cited artifacts / the breakthrough table; all rows derived.)
<!--derived-->
| deployable-family (simplewiki, depth-2, contiguous, 6-seed) | mean margin_vs_trigram (deep 10-99) |
|---|---|
| bag content-addressable attention (`assoc`) | -0.347 |
| ordered attention (`assoc_t`, +time-cell) | -0.147 |
| spiking SSM dual-nonneg (`ssm`) | -0.125 |
| **hippokey (HiPPO-keyed attention, THIS arm)** | **-0.284 (NO-GO, 6/6)** |
| linattn (the current deployable mouth) | +0.0505 (6/6 cross) |

The linattn/assoc numbers are the deep-bucket means in `research/findings/raw/_emerge_wkv_lm_linattn_depth2_
contiguous_6seed.json` and `research/findings/raw/_emerge_wkv_lm_assoc_temporal_depth2_contiguous_6seed.json`
(ssm/bag from the breakthrough finding's table). hippokey is **~0.33 below linattn** and **below even the
token-keyed attention arms** it was designed to surpass — the HiPPO front-end HURT.

## 3. The honest read — the "better key" hypothesis is FALSIFIED (a method verdict, not a wall)

The design bet (§4-5 below) was the July diagnosis: assoc's ceiling was "a BAD KEY" (a shallow token-local
representation), so a RICHER, deeper, multi-timescale HiPPO context key should make content-addressable recall
load-bearing at long range. The measurement says the opposite: **a smoothed multi-timescale HiPPO code is a
WORSE next-token key than the token-local representation.** The likely reason is now legible: next-token
prediction needs SHARP token-identity matching for the retrieval competition, and the HiPPO state is a
low-pass, multi-timescale SUMMARY that blurs exactly that identity — so the softmax matches on a fuzzier signal
and retrieves a fuzzier value. The depth profile confirms it (see the §1 table): hippokey is most negative at
depth-2, where sharp recent-token identity matters most and the trigram is strongest, and least negative at
deep context, where a smoothed context code helps a little — but it never crosses.

⇒ This is a verdict on the METHOD (HiPPO-multi-timescale-key content-addressing), and it composes with the two
prior attention arms into a decisive DIRECTION result: **three content-addressable attention families now fail on
this protocol (assoc, assoc_t, hippokey), with the fanciest key the WORST.** "More/better content-addressing
memory" is exhausted as the next lever. The capability (brain-native open fluency) is NOT deferred — its next
method is re-aimed (§6), exactly as this de-risk's own GO-bar pre-committed.

## 4. The mechanism that was tested (context — still the literal steer, just falsified as the lever)

`--recurrence hippokey`, per position (causal, s<=t): a FIXED HiPPO multi-timescale diagonal SSM
(`x_{t+1}=A x_t + B u_t`, A a fixed log-spaced fast->slow decay grid, B a fixed random projection — both
register_buffers, no learned recurrent credit) produces a per-position multi-timescale context code x_s; then
`q_t=Wq(x_t)`, `k_s=Wk(x_s)`, `v_s=Wv(z_s)`, `read_t = sum_{s<=t} softmax(q_t.k_s/sqrt(D)) v_s`, `delta_t =
Wo(read_t)`. The only change from `assoc` is that Q/K key off the HiPPO state x rather than the token-local z.
Bio anchor (still valid): entorhinal multi-timescale context (time cells / TCM / grid modules ~ the diagonal
HiPPO-LegS spread, Gu 2020) -> CA3 content-addressable pattern completion (Ramsauer 2020's Hopfield<->attention).
The biology was sound; it simply is not the axis that moves this metric.

## 5. What the project had substituted, and the implementation (banked, default-off)

`--recurrence learnkey` (2026-09-04) had been tagged as this steer but is a FIXED codebook with NO HiPPO SSM;
hippokey built the literal composition. The arm is additive in `research/runners/_emerge_wkv_lm_derisk.py`
(new `HippoAssocLayer`, byte-identical when off by construction — guarded empty ModuleList, zero init-RNG draws;
no sim/ edit, no production edit) and stays **default-off / banked** (nothing in production changes). The read is
exact-softmax O(T^2) (T<=40 here), a ceiling instrument; there is no reason to spend the spike-port rung on it
given the NO-GO. It remains in the runner as a reproducible, adjudicated member of the attention family.

## 6. The re-aim (named next mechanisms — NOT another content-addressing variant)

Detailed in the companion ROADMAP finding (`2026-09-05-own-voice-fluency-reaim-objective-and-capacity-ROADMAP.md`).
In order:

1. **Predictive-coding OBJECTIVE (the strongest same-budget external lever, ALREADY BUILT).** The
   ordered-attention bound-investigation's best external datapoint says the training OBJECTIVE, not the
   memory mechanism, is the dominant lever below ~20M tokens (a causal+masked hybrid far outperforms a tuned
   n-gram and a causal LSTM on BLiMP; recurrence/attention alone barely ties the n-gram — this arc's exact
   failure mode). The
   causal-compatible port already exists as `--pred-aux-weight` (multi-horizon further-ahead auxiliary heads,
   Rao & Ballard 1999 predictive coding). Cheapest decisive next run: `linattn --pred-aux-weight` 6-seed on the
   same protocol — does a richer objective push the ALREADY-CROSSING deployable mouth further, and does it lift
   the family off the bound on the BROAD (wt103) domain where linattn fell below the trigram?
2. **CAPACITY** (depth then width) on linattn, to escape the ~6.4 tok/param reversal band — the token-supply
   finding (2026-09-01) showed the substrate scales with capacity-matched tokens.
3. NOT another key/attention variant — that direction is banked exhausted by this NO-GO.

## 7. No-defer note

A wall defers a METHOD, never the capability. hippokey banks the "richer content-addressable key" method as
NO-GO (and, with assoc/assoc_t, the content-addressing direction), and hands the next method (objective, then
capacity). The own-voice mouth remains the #1 goal-blocker (it blocks ~48/64 one-brain ledger rows); its critical
path is re-aimed onto the axis the record's own external evidence points to.

## Reproduce

```bash
SIM_BACKEND=cupy .venv/bin/python -u -m research.runners._emerge_wkv_lm_derisk \
    --recurrence hippokey --n-layers 2 --uniform-decay --d-model 192 --batch 128 --tokenizer bpe \
    --corpus data/corpus/simplewiki.txt --contiguous --max-len 40 \
    --n-sentences 1200000 --max-train-sents 1000000 --max-eval-sents 4000 --epochs 5 \
    --seeds 42 43 44 100 101 102 --tok-cache \
    --json _emerge_wkv_lm_hippokey_depth2_contiguous_6seed.json   # writes into the raw findings dir
```
