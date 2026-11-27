# Route to a spiking cortex without losing fluency (memo, 2026-11-27)

Status: plan only, nothing run. Tags: **[M]** measured in our record, **[C]** cited, **[E]** my estimate.
Record check run 11-27.

## 1. Recommendation

**Take route C (convert the RWKV-7 cortex into a spiking twin, initialised from its own weights, distilled from it),
and keep route B only as a 0.1B control arm.** Both routes are surrogate-gradient training, so both fit the declared
trade (`brain/ARCHITECTURE.md` rule 4). C starts from a fluent network and keeps the RWKV-7 state layout that the
hippocampus, sleep and every result since 11-02 depend on. B starts from nothing, needs a corpus we do not have, and
our own record shows it failing three times at small scale.

## 2. What "fully spiking" can honestly mean

No fluent spiking language model, published or ours, holds its recurrent state in spikes.

- SpikeGPT keeps a real-valued numerator/denominator state; only block outputs are binarised [C, our
  `2026-09-03-spiking-content-addressable-read-DESIGN.md`, a secondary reading; UNCHECKED against the SpikeGPT paper, arXiv 2302.13939].
- SpikingBrain-7B trains with integer spike counts in one collapsed time step and expands them to {-1,0,1} trains
  only at inference [C, arXiv:2509.05276].
- SpikingMamba puts signed-integer spikes (range ±4, one token step) before the input and output projections [C,
  arXiv:2510.04595].

So three levels, to be declared in `brain/SCAFFOLDS.md`:

| Level | Meaning | Reachable |
|---|---|---|
| L1 | Every projection input is a signed spike count; state (matrix memory, token shift) stays graded, as synaptic variables | yes, route C |
| L2 | Counts expanded to binary/ternary spike trains over micro-steps, parity-tested against L1 | yes; our run4 phasor port reached ppl ratio 0.9999999974 [M] |
| L3 | State itself carried by spikes | no existence proof [C]; our record calls it the wrong target [M, GAP_CLOSURE_MISSION.md:7343] |

Target: L1 with an L2 parity test. L3 is a research programme, not a port.

## 3. The record that decides between B and C

- Direct LIF language model trained by surrogate-gradient BPTT, 25M parameters: held-out ppl 203,753, token soup [M,
  2026-06-02]. Distillation from a trigram teacher into a LIF student: ppl 804, negative [M, 2026-05-17].
- Best deployable from-scratch spiking mouth (linattn, d192): +0.05 nats over a trigram on SimpleWiki, 6/6 seeds; -0.29
  to -0.57 on WikiText-103, one seed [M]. Token scaling then saturated at d96/d192: the limit is parameters [M, 09-17].
- 267M WKV trained by BPTT on 7.0B tokens: val ppl 45.66 [M]; SpikeGPT-216M 39.75 on WikiText-103 [C, pivot memo].
- SpikingMamba, distilled from a same-size Mamba2 and initialised from its weights: WikiText-103 ppl 26.32 vs 19.56
  at 130M; WikiText-2 15.17 vs 10.42 at 1.3B [C]. That is a ratio of 1.35 and 1.46, or 0.30 and 0.38 nats.
- SpikingBrain-7B (from Qwen2.5-7B, ~150B tokens): 69.98 vs 68.75 are both SpikingBrain variants (W8ASpike vs another
  SpikingBrain variant) [C]. The paper gives no Qwen comparison on this pair, so 1.2 points is NOT a measure of
  conversion loss from the teacher; no conversion-loss figure from SpikingBrain is used in this memo.

Reading: from scratch at our budget gives ppl in the 40s at best [M]; same-architecture conversion gives a 0.3 to
0.4 nat gap to a same-size teacher, from the SpikingMamba numbers only [C]; SpikingBrain adds no gap evidence. The 05-17 negative was another method (LIF student, trigram teacher, random init).

## 4. What is realistic per size

Throughput assumptions [E], one A10G or 3090, bf16, context 512: student training with teacher forward about
25k tok/s at 0.1B, 7k at 0.4B, 2k at 1.5B. The 1.5B figure agrees with SpikingMamba-1.3B: 7B tokens in 42 h on 8
A100-class accelerators [C], about 900 A10G-hours [E]. g5.xlarge about $1/h on demand [E, list price from memory].

| Size | Route C tokens | A10G-hours | Cost | Expected gap to teacher | Route B from scratch |
|---|---|---|---|---|---|
| 0.1B | 0.3B screen, 1-3B full [E] | 4, then 11-33 [E] | $4, then $11-33 [E] | 0.3 nats [C, 130M Mamba] | 3B tokens, about 30 h [E]; quality near our 267M WKV (ppl 40s) [M], far below G1 |
| 0.4B | 1-3B [E] | 40-120 [E] | $40-120 [E] | 0.3-0.4 nats [E] | 8B+ tokens, 300+ h [E]; not worth it |
| 1.5B | 1B staged, 7B full [C for 7B] | 140, then about 970 [E] | $140, then about $1,000; 3 weeks at the cap [E] | 0.38 nats [C, 1.3B Mamba] | 30B tokens at 20 tokens/parameter [C, Chinchilla], 3,000+ h [E]; out of budget |

Corpus on disk: `fineweb_edu.txt`, 4.3 GB, about 1B tokens [E]. The RWKV G1 corpus is not available, so chat-format
text must be sampled from the teacher and mixed with fineweb.

**Energy and latency on a GPU: little or no gain.** SpikingBrain states that GPUs cannot FULLY exploit spike
sparsity (not that they exploit none); its speedups come from linear attention. Its 97.7% energy saving is against
FP16 MACs (85.2% against INT8), for the spiking operations only, a 45 nm estimate for asynchronous hardware, not a
whole-model figure [C]. SpikingMamba's 4.76x energy figure: UNCHECKED here (I assumed, not read, that it is an
operation-count estimate) [C, unverified]. On the 3090, count mode (L1) should cost about
1.0-1.2x the teacher per token [E]; expanded trains (L2) up to 4x [E]. The value is architectural (the 2027 Q1
shared substrate; spike counts for local rules), not speed.

## 5. Cheapest decisive first experiment (E1)

**Distil RWKV-7 0.1B into a spiking twin and test it inside the brain.**

- Build: extend `brain/plastic_cortex.py` (already a differentiable RWKV-7 forward [M]) with a signed-integer spike
  function (range ±4, straight-through gradient) on the input of every linear projection. All weights trainable,
  initialised from the teacher. Loss: token KL to the teacher plus per-layer hidden-state match. Training loop from RWKV-LM, not hand-built.
- Data: SpikeOPD (abstract only, UNCHECKED beyond that) reports its own method avoids the on-policy collapse and
  gains 0.8-2.9 points at 0.125B-1.3B, and argues for training on self-generated prefixes. So the twin is not
  trained on teacher/fineweb text alone: a share of each batch is prefixes sampled from the twin itself, with the
  teacher's token KL as the target on them. Follow SpikeOPD's recipe as published, not a hand-built variant.
- Arms: (a) twin from teacher weights, teacher-text data; (a2) as (a) plus self-generated prefixes (SpikeOPD-style);
  (b) same twin, random init, same tokens (route B control); (c) teacher. (a2) adds about 2.5 A10G-hours [E].
- Budget (corrected 11-27: four arms, of which three train; re-corrected 11-27 after the build review, the prereg's
  `estimate` is now the one source). Per seed at 25k tok/s: (a) 3.8 h, (a2) 3.8 h + 2.5 h of self-generated prefixes
  (2,289 refills x about 4 s, a fixed cost that does not scale with training throughput) = 6.4 h, (b) 3.8 h, (c) 0.25 h
  (its sleep unit only; its ppl and ladder do not depend on the seed and run once): 14.3 A10G-hours, about $14 [E].
  Seed 7 plus six seeds: about 100 h, about $100 [E], not the $13-20 this memo first gave (that was one seed of two
  arms). The 25k tok/s of section 4 assumes RWKV-LM's training kernel. The in-repo training forward (a per-token WKV
  loop in Python, no training kernel on disk; the rwkv package's rwkv7.cu is inference-only, no backward) is reviewed
  at 0.5-1.5k tok/s [E] even at micro-batch 16: 1,200-1,800 h, which E1 cannot spend. E1 therefore needs an
  off-the-shelf training kernel (RWKV-LM wkv7 or flash-linear-attention rwkv7, a download) before it can run as
  registered; the launch refuses prep and smoke until that decision is written down, and gates the run on a measured
  10k tok/s. Device: the workstation 3090 (BRAIN_ALLOW_W3090=1, via tools/gpu_queue.sh, $0) or AWS g5.
- Read-outs:
  1. Held-out ppl ratio to the teacher on WikiText-103 validation and on chat-format text.
  2. `brain/experiments/sleep_probe.py`: recall with the hippocampus off after sleep. Teacher reference 3.83/12
     in the 11-27 six-seed gradient arm [M].
  3. Scale-ladder rows: in-context recall (teacher 6/12 [M]), reinstatement at gain 4.0 (11/12 [M]), same-words
     selection (12/12 [M]). This is the question no paper answers: do state-delta episodes still store and reinstate
     through a spike-coded cortex.
  4. Spikes per channel and the share of silent channels (SpikingBrain: 1.13 and 18.4% [C]).
  5. Weekly 20 prompts read whole, twin beside teacher.
- Proposed bars, to be registered before data, applied to (a) and to (a2) separately: ppl ratio at most 1.5;
  sleep recall at least half the teacher's; reinstatement at least 8/12; the arm better than arm (b) by at least 1 nat.
  Decision (a2 added 11-27): (a) is the primary method. GO on (a) if (a) passes; (a2) replaces (a) as the method carried
  to E2 only if (a2) also passes and its chat-text gap is at least 0.05 nats smaller. GO on (a2) if only (a2) passes.
  If both fail, that is two failed methods: the stop rule below ends the arc until a new method is named. Seed 7 for
  dev, then six seeds. Exact rule and controls: `brain/research/2026-11-27-spiking-twin-e1-prereg.md`.
- Expected-fail note: 0.1B is weak in the brain (in-context 6/12 [M]). A fluency miss there is a scale result; a
  reinstatement miss is a method result.

## 6. Risks

1. **Meaning of "spiking".** L1 is integer-coded communication with graded state. If that is not acceptable, stop
   before spending.
2. **The gap lands on thresholds.** 0.3-0.4 nats may break recall and identity, which sit near their limits. Gains
   must be refitted per model (operating points did not survive a scale change [M, PREREQUISITES]).
3. **Hippocampus interface.** Spike-coded keys and values change the matrix-memory deltas; old brain homes will not
   load into a twin (a home refuses another cortex [M]).
4. **Data distribution.** Without the G1 corpus the twin may lose chat ability. Vanilla on-policy distillation
   on the student's own text may collapse in SpikeOPD's stress test, but its own method fixes this and argues for
   self-generated prefixes, which E1 now uses in arm (a2) [C, SpikeOPD arXiv:2608.27857, abstract only].
5. **1.5B cost.** About $1,000 and three weeks at the full cap, in competition with the 12-14 and 12-31 lanes.
6. **Size does not close the gap.** Mamba ratio 1.35 at 130M, 1.46 at 1.3B [C].

## 7. Dated plan, after test2

| Date | Step | Cost |
|---|---|---|
| 11-28 to 12-03 | Record brief in `brain/research/`, prereg for E1, sonnet pre-data review; smoke the twin forward on pool CPU (parity with spikes off) | $0 |
| 12-04 to 12-08 | Training-kernel decision (download, parity test) before prep; E1 seed 7 on the 3090 or one g5, then six seeds; read with a fresh skeptic. The dates assume 25k tok/s; on the in-repo loop they do not hold | $14 + $86 on g5 at 25k tok/s, $0 on the 3090 [E] |
| 12-10 | test2 read done. Decision gate on E1: GO to 0.4B, or bank the method | |
| 12-11 to 12-20 | E2: 0.4B twin, 1B tokens, same read-outs plus weekly transcript | $40-60 [E] |
| 12-31 | Milestone unchanged (local sleep rule, 1.5B). Report E1/E2 beside it; try the local sleep rule on the twin, where pre and post spike counts exist | |
| 01-04 to 01-11 | 1.5B stage 1: 1B tokens, read the ppl curve, project the rest | $140 [E] |
| 01-12 to 02-05 | 1.5B full, only if stage 1 projects a ratio at most 1.5; about $40/day | about $850 [E] |
| 2027 Q1 | L2 parity test (expanded spike trains) at 0.1B; adopt the twin as cortex only if the weekly transcript holds | |

Stop rule: two failed methods on E1 end the arc until a new method is named.

## Sources

Ours: the six 2026-06-02 to 2026-09-17 spiking-LM findings in `research/findings/` named above, `2026-08-02-gap1-wkv-width-ladder-...`,
the 11-02 pivot memo, `brain/experiments/results/2026-11-27-local-sleep-01b-read.md`, `2026-11-02-scale-ladder.txt`.
External: arXiv 2302.13939, 2509.05276, 2510.04595, 2608.27857 (abstract only). Not verified: g5 price, SpikeGPT state
reading (from our own design note), SpikingMamba 4.76x energy basis, SpikeOPD beyond its abstract.
