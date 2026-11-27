# E1 spiking twin prereg DRAFT (2026-11-27): distil the 0.1B RWKV-7 cortex into an L1 spiking twin, test it in the brain

STATUS: DRAFT, waiting for a fresh sonnet pre-data review. It is not registered until that review passes and
`brain/experiments/results/2026-11-27-spiking-twin-e1/run/REGISTERED` exists. The launch script refuses six seeds
without that file. No data has been taken. Route memo (record): `docs/plans/2026-11-27-spiking-cortex-route-memo.md`,
route C. Its budget arithmetic and its decision rule (arm a2) were corrected on 11-27, before this prereg. A build
review on 11-27 (7 defects) was applied before data; section 9 lists each fix.

## 1. Record brief
- **What we knew.** From-scratch spiking language models in our record failed at small scale. A LIF model trained by
  BPTT reached ppl 203,753 (06-02). A trigram teacher distilled into a randomly initialised LIF student reached ppl 804
  (05-17, a different method: LIF student, trigram teacher, random init). Spiking mouths saturated on parameters (09-17).
  The 267M WKV trained by BPTT reached val ppl 45.66. A same-architecture conversion (SpikingMamba-130M, initialised
  from the teacher and distilled) is cited at a 0.30-nat gap to the teacher [C]. Teacher references at 0.1B, from our
  own harnesses: in-context recall 6/12, reinstatement at gain 4.0 11/12, same-words selection 12/12 (scale ladder
  11-02). Gradient sleep with the hippocampus off gives 3.83/12 over six seeds (11-27 local-sleep read).
- **What is new.** A spiking twin of OUR cortex. It is initialised from the teacher's weights, so it keeps the RWKV-7
  state layout that the hippocampus and sleep depend on. It is run through the brain's own memory read-outs. No paper
  answers whether state-delta episodes still store and reinstate through a cortex coded in spike counts.
- **Prerequisites (brain/PREREQUISITES.md, Language).** Met: a pretrained cortex, used as the declared pretraining
  trade. A differentiable v7 forward exists (PlasticCortex), and so do the sleep_probe and scale-ladder harnesses.
  Missing: spikes in the state (L3). That is out of scope: no published fluent model has it, see the memo, section 2.
  Missing: the G1 chat corpus. Teacher-written chat stands in for it. Missing: RWKV-LM's WKV7 CUDA training kernel. It
  is not on disk and E1 downloads nothing. See section 7.
- **Expected.** A fluency gap of about 0.3 nats [C]. My estimate of the outcome: 40% GO, 30% FLUENT_ONLY (memory breaks
  near its thresholds, memo risk 2), 20% NEGATIVE, 10% VOID. A fluency miss at 0.1B is a scale result. A memory miss is
  a verdict on this method, not on spiking cortices.
- **Host decision.** None. The only host text is the chat marker `User:`, from which the teacher writes whole
  dialogues. The probes reuse the existing fact banks (`facts.py`, `sleep_probe.py`, `weekly.py` turns).

## 2. What the twin is (level L1)
`brain/spiking_twin.py`, `SpikingTwin(PlasticCortex)`. Every projection input carries a signed integer spike count
`c = clamp(round(x/theta), -4, 4)`, so the projection sees `c * theta`, with one learnable theta per input channel.
This covers 15 sites per layer (att r, k, v, output; w1/w2, a1/a2, g1/g2, v1/v2; ffn key, value) plus the readout
head. The state (matrix memory and token shifts) stays graded. The gradient is a clipped straight-through estimator
with an LSQ-style theta gradient. Theta is initialised at the 0.99 quantile of |x| divided by 4, read on 16
calibration sequences with spikes off. Spikes off with teacher weights equals PlasticCortex (test).

## 3. Arms (all at 0.1B, teacher = rwkv7-g1d-0.1b-20260129-ctx8192)
| arm | init | data | role |
|---|---|---|---|
| a | teacher weights | fineweb_edu (90%) + teacher-written chat (10%) | primary method |
| a2 | teacher weights | as a, but 1 of every 4 sequences is the twin's own continuation (448 tokens) of a 64-token prompt; the teacher's KL is the target on it | SpikeOPD-style second method |
| b | RWKV-LM x070-style random init (seeded; recipe in `SpikingTwin.random_init`), same distillation | same tokens as a | route-B control (no-init contrast) |
| c | teacher, spikes off | no training | reference and instrument check |

The registered constants are `CFG` in `brain/experiments/spiking_twin_e1.py`. Each trained arm gets 0.3B tokens:
seq 512, 16 sequences per step (micro-batch 16 x accumulation 1), 36,622 steps. AdamW, wd 0.01, betas (0.9, 0.99),
clip 1.0. The learning rate is 1e-4 for a and a2 and 6e-4 for b, with 500 warmup steps and a cosine decay to 10%. Loss =
token KL(teacher || twin) + 0.5 x the per-layer relative residual MSE. Training runs in bf16 autocast, with the WKV
state in fp32. The slow-store adapters stay at zero during distillation; they belong to sleep. With one seed, arms a,
a2 and b draw the same sequences in the same order. A chat draw is one whole pool row (512 tokens starting at the
marker), so no training window straddles two dialogues, the same alignment as the eval windows. a2 replaces 4 of the
16 sequences of each step with self-generated ones; it takes its prompts from a separate stream and refills its
buffer (64 sequences) every 16 steps. The sampler state and the unconsumed buffer are checkpointed, so a resumed a2 run
equals an uninterrupted one (test).

Arm b's init is RWKV-LM's x070 recipe as recalled from its model.py (RWKV-LM is not on disk, so the recipe is not
checked against the source [E]): token-shift mixes as layer-ratio ramps, a per-channel decay-speed ramp for w0, zero
LoRA downs with orthogonal 0.1 ups, zero attention output and FFN value, small uniform r/k/v, ln_x scaled by layer,
orthogonal head. This package's decay `exp(-e^-0.5 sigmoid(w0 + lora))` equals RWKV-LM's parameterisation, so w0 maps
one to one. Keys the recipe does not name fall back to moment-matched noise and are listed in the train record
(`init_fallback`). A channel whose calibration input is exactly zero (the zero LoRA downs in b) gets theta 1/K
instead of the 1e-4 floor. b's learning rate stays 6e-4, RWKV-LM's usual value at this size, with no sweep; see
section 7 item 4.

Data on disk; E1 downloads nothing. Training text is the first 0.3B tokens of
`/home/dant123/Projects/sim/data/corpus/fineweb_edu.txt` (4.3 GB), plus a 10M-token chat pool written by the teacher
(generator seeds 0..). Held-out text is the WikiText-103 validation split
(`/home/dant123/Projects/Lift-Gate-Glue/data/wikitext103/wikitext-103-raw-v1/validation-00000-of-00001.parquet`, about
0.25M tokens) and 0.1M tokens of teacher-written chat (generator seeds 100000..). Both pools are disjoint from the
training seeds.

Seeds: 7 for dev (one reading, not a claim), then 42/43/44/100/101/102. Nothing is tuned on seed 7. It checks the
instrument, the throughput and VOID only.

## 4. Measures (per arm and seed; one atomic JSON per unit, with host, commit, device and precision)
1. **ppl.** Mean next-token NLL on WT103 val and on held-out chat, in non-overlapping 512-token windows. The ratio to
   the teacher is exp(mean NLL gap). Also the in-sample unigram entropy of the WT103 targets (a property of the data,
   read from arm c), the floor that decides whether b trained at all.
2. **Lesion.** The same trained twin with spikes off (graded inputs): NLL on both sets.
3. **Spikes.** Mean |count| per channel per token, and the share of channels that never fire on WT103 val.
4. **sleep.** The existing `sleep_probe.evaluate` with the twin as the plastic cortex. Arms are no_sleep (no-contrast),
   sleep_true, and sleep_shuffled (sham: the names are swapped). Settings: 240 steps, lr 5e-4, KL 1.0, hippocampus off.
   Recall (of 12) and false recall (of the 4 never-told questions) are both scored.
5. **ladder.** The 11-02 scale-ladder rows on the twin, through `TwinCortex`: in-context recall, reinstatement at gains
   1.5/2.5/4.0, same-words and paraphrase selection. A new sham is added: reinstatement at 4.0 with the WRONG memory
   (fact j+1's delta for question j).
6. **transcript.** The weekly test's 16 turns as one conversation with no hippocampus, greedy, twin beside teacher.
   It is read whole and has no bar.

Arm c (the teacher) is deterministic on ppl, ladder and transcript: greedy decoding, fixed fact banks, no seed. Those
units are measured once (`eval_c_shared_<unit>.json`) and shared by every seed, so the six-seed C1 and C4 means are one
measurement, not six. c's sleep unit is seeded (episodes, adapter init) and runs per seed.

## 5. Pass rule (written before data; `verdict()` in the harness, exact tests in `brain/tests/test_spiking_twin_e1.py`)
All values are six-seed means. X is a or a2.
- **F1 fluency.** The ppl ratio to c is at most 1.5 on WT103 val AND on chat.
- **F2 init matters.** NLL(b) - NLL(X) on WT103 val is at least 1.0 nat.
- **S sleep.** All four: recall(X, sleep_true) is at least 0.5 x recall(c, sleep_true); recall(X, sleep_true) minus
  recall(X, sleep_shuffled) is at least 1.0; recall(X, sleep_shuffled) is at most 1.5; false recall(X, sleep_true) is
  at most false recall(c, sleep_true) + 0.5 (of 4). The teacher's own false recall is already about 3/4 at 0.1B
  (11-03 record), so the bound is relative: the twin may not confabulate more than the teacher.
- **R reinstatement.** Reinstatement(X, gain 4.0) is at least 8/12 AND the twin's own wrong-memory sham at gain 4.0
  is at most 3/12. Without the sham bound a twin that answers every question with any memory would pass.
- X passes if F1, S and R pass and F2 is not failed (F2 passes, or is unassessable, below).
- **b unassessable.** If any b unit is non-finite, or b's six-seed WT103 NLL is not at least 1.0 nat below the unigram
  entropy, b's recipe failed (it diverged or learned only token frequencies). Then F2 is unassessable (None), b is
  left out of C3 and C5, and the verdict carries `f2_assessable: false` with the reason. a and a2 are still judged on
  F1, S and R. A GO with F2 unassessable is a GO on the twin method with no route-B claim (init mattering is not
  shown), and is reported that way. A b that was never run (missing units) is still VOID (C6).

Controls. Any failure makes the verdict VOID, and the harness names the clause.
- **C1.** The teacher reproduces the ladder: in-context in [5, 7], reinstatement at 4.0 in [10, 12], same-words at
  least 11.
- **C2.** Teacher sleep_true recall is at least 2, and teacher sleep_shuffled is at most 1.5.
- **C3.** No-contrast: no_sleep recall is at most 1 for every arm.
- **C4.** Sham: the teacher's wrong-memory reinstatement at 4.0 is at most 3.
- **C5.** Spikes are engaged in a, a2 and b: |NLL(spikes off) - NLL(spikes on)| is at least 0.01 nats, the silent share
  is below 0.99, and the mean count is above 0.
- **C6.** Every arm x seed x unit (ppl, sleep, ladder) is present, and finite for a, a2 and c (b's finiteness decides
  "b unassessable" instead).

Verdict.
- **GO (carry a)** if a passes. a2 replaces a as the carried method only if a2 also passes AND its chat gap is at least
  0.05 nats smaller.
- **GO (carry a2)** if only a2 passes.
- **FLUENT_ONLY** if neither passes but some X passes F1 and F2. This is a method result on memory through spike
  counts.
- **NEGATIVE** otherwise.
- FLUENT_ONLY and NEGATIVE both mean two failed methods (a and a2). The memo's stop rule then ends the arc until a new
  method is named.
- The seed-7 reading is reported with claim = false.

## 6. Compute estimate (harness `estimate`; per seed = a + a2 + b training + evals)
The a2 sampling is a fixed cost: 2,289 refills x the time of one refill (64 prompts x 448 sampled tokens, batched,
one token per forward). It does not scale with training throughput. Default 4 s per refill [E]; the smoke measures it
and the gate passes the measured value to `estimate`. Evals: 0.5 h per trained arm, 0.25 h for c's per-seed sleep,
plus 0.25 h once for c's shared units [E].

| tok/s (training) | a / a2 / b / c h | per seed | seed 7 + six seeds | g5 cost at $1/h [E] |
|---|---|---|---|---|
| 25,000 (memo [E], needs a training kernel) | 3.8 / 6.4 / 3.8 / 0.25 | 14.3 h | 100 h | about $100 |
| 10,000 (launch gate) | 8.8 / 11.4 / 8.8 / 0.25 | 29.3 h | 205 h | about $205 |
| 5,000 | 17.2 / 19.7 / 17.2 / 0.25 | 54.3 h | 380 h | about $380 |
| 1,000-1,500 (review [E] for the in-repo loop at micro-batch 16) | 56-84 / 59-86 / 56-84 / 0.25 | 171-254 h | 1,200-1,800 h | not run |

Prep: tokenising 0.3B fineweb tokens with the pure-Python tokenizer takes about 1-2 h of CPU [E], and it holds a GPU
queue slot while it does. Writing the 10M-token chat pool takes about 0.3-1 h of GPU [E]. Disk: about 15-20 GB [E]
(21 fp32 twins plus one live resume checkpoint per training job, deleted when its twin is final); 579 GB free.
On the 3090 the cost is $0 but the card time is long. The gate (section 7) stops the launch below 10k tok/s.

## 7. Deviations from the memo and items for the pre-data review
1. **No RWKV-LM training loop or kernel.** The memo asks for RWKV-LM's training loop, not a hand-built one. RWKV-LM is
   not on disk, and E1 downloads nothing. The harness therefore uses an in-repo per-token WKV loop with per-block
   activation checkpointing. It is likely 3-5x slower than the memo's 25k tok/s. The launch is gated on a measured
   200-step smoke: below 10k tok/s, STOP. The build review estimates the loop at 0.5-1.5k tok/s even after moving to
   micro-batch 16 x accumulation 1 (the loop cost barely depends on B, so 4 x 4 was the worst split; 16 x 1 is about
   4x and costs about 6 GB of block-recompute memory [E]). The rwkv package's `rwkv7.cu` is an inference op with no
   backward. **So E1 cannot run as registered without a training kernel.** The off-the-shelf fix is RWKV-LM's wkv7
   CUDA op or flash-linear-attention's rwkv7 chunked kernels, both a download, and that decision belongs to the owner
   or reviewer. The launch script refuses prep and smoke until the decision is written to `run/KERNEL`, so no GPU
   time is spent first. "in-repo" there means measuring the loop only, and the gate is then expected to stop. A
   chosen kernel needs a parity test against `SpikingTwin` (spikes off, and spikes on through the same count
   function) and its own registered addendum before prep.
2. **SpikeOPD is known from its abstract only.** a2 is the generic self-generated-prefix variant, not the published
   recipe.
3. **The chat data is the teacher's own text.** This flatters the teacher on chat ppl. WT103 val is the clean measure,
   and F1 needs both.
4. **Arm b learning rate.** b's learning rate (6e-4, a from-scratch value) differs from a's (1e-4, a fine-tune value).
   Both are fixed here, with no sweep. A b that is undertrained by its learning rate makes F2 easier. Two guards now
   limit this: b uses RWKV-LM's own init recipe (moment-matched noise was not a trainable RWKV init: decay, gates and
   output were all noise), and a b that is not at least 1 nat below unigram makes F2 unassessable instead of trivially
   true. A b that trains but stays undertrained by its learning rate still flatters F2; the reviewer should judge
   whether 1 nat is still meaningful. A one-seed b learning-rate sweep would cost 3 x 3.8 h at 25k tok/s; it is not
   in E1.
5. **The C1/C2 tolerances are guesses.** They assume the GPU fp32 twin forward reproduces the 11-02 CPU ladder to
   within one item. If seed 7 shows a systematic offset, that is VOID and gets a new prereg, not a moved band.
6. **Spikes-off parity is to 1e-5, not byte-identical.** The spikes-off twin matches PlasticCortex to 1e-5 (float
   order of the batched math). C1's bands rest on greedy text surviving that difference (with item 5).
7. **SCAFFOLDS.md.** L1 and L2 are not yet declared there. They are declared with the result (land-result), not before.
8. **L2 parity.** The L2 parity test (expanded spike trains) belongs to 2027 Q1. It is not part of E1.

## 8. Run
`brain/experiments/results/2026-11-27-spiking-twin-e1/launch_3090.sh` runs these stages in order: (write `run/KERNEL`),
prep, smoke, gate, score7, then six (only after REGISTERED exists and seed 7 is not VOID), then score. The script
changes to its own checkout and sets PYTHONPATH there, so the harness it scores with is the one beside it, from any
cwd. All GPU work goes through tools/gpu_queue.sh under tools/memcap.sh, with BRAIN_ALLOW_W3090=1.

## 9. Build review (11-27, before data) and what changed
1. R passed without its sham: the twin's own `sham_4.0` was recorded and never read. Fixed: R needs sham at most 3
   (section 5); test with 12/12 sham gives FLUENT_ONLY, not GO.
2. S had no margin and ignored false recall. Fixed: margin 1.0, twin shuffled at most 1.5, false recall at most the
   teacher's + 0.5; each part tested failing alone and at its boundary.
3. The launch script ran the module without its checkout on the path. Fixed: `cd "$REPO"; export PYTHONPATH="$REPO"`
   (score from `/` now runs).
4. a2 resume differed from an uninterrupted run (sampler and buffer not checkpointed). Fixed and tested with a
   non-empty buffer at the checkpoint (resumed weights now equal, not just close).
5. Throughput: accepted (section 7 item 1). micro-batch 16 x 1, and prep and smoke refuse until the kernel decision.
6. The smoke did not time a2's sampling. Fixed: the smoke times one refill; `estimate` adds refills x time as a
   constant.
7. b could void the whole run. Fixed: RWKV-LM-style init, and a diverged or untrained b makes F2 unassessable
   (section 5) rather than VOID.
Minor: c's seed-free units run once (section 4); chat draws are whole rows; resume checkpoints are deleted once a twin
is final and the disk figure is corrected; the memo now quotes this section's numbers.
Record check (`tools/before_you_build.sh`, run 11-27 after the review): top hits are the 05-17 subword spiking LM
NEGATIVE, the 09-03 spiking-mouth fluency NO-GO and the 06-02 spiking word-recognition read. All three are from-scratch
or different-architecture spiking models, already in the brief (section 1); none is a teacher-initialised conversion of
our cortex, so E1 does not repeat a killed lever.
