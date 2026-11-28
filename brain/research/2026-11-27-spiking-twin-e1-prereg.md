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

   **Decision (2026-11-27, owner-approved): kernel = flash-linear-attention fla-core 0.6.0, `fla.ops.rwkv7.chunk_rwkv7`**
   (installed `--no-deps` in `_scratch/pylib_fla`, on PYTHONPATH for the twin's jobs only, imported lazily). Switch
   `SpikingTwin(wkv="fla")` / `train --wkv fla`, default `loop` (the per-token loop; byte-identical to the pre-switch
   code on CPU, outputs, states and gradients, spikes off and on). Scope: training only, on CUDA, for the student and
   the teacher's distillation targets inside `train()` (its theta calibration too). Eval, prep and single-token sampling
   steps (T = 1) run the loop, so every registered read-out is measured on the reference math. Mapping onto fla's DPLR
   form `S' <- S' exp(gk) + b (a^T S') + k v^T, o = scale S'^T q` (`brain.spiking_twin.wkv_fla_args`): S' = our state
   transposed (ours [H, V, K], fla [H, K, V]), q = r, gk = log w = -0.606531 sigmoid(w0 + lora) (log space), a = -kk,
   b = kk * a_gate, scale = 1, layout [B, T, H, N], fp32 state in and out. Kernel input dtype: fp32 (default;
   `--wkv-dtype bf16` only after its parity case passes). `lower_bound` (fla's 64-chunk scheme) is off.
   **Parity bar** (`brain/tests/test_spiking_twin_fla.py`; relative L2 error to the fp32 loop): CPU, the mapping equals
   fla's kernel recurrence and fla's own `dplr_recurrence` reference to fp32 rounding (< 1e-6, outputs, final state,
   gradients). GPU, tiny random cortex and the real 0.1B teacher, spikes off / on, carried state, nonzero adapters:
   fp32 spikes off logits <= 2e-3, state <= 2e-3, adapter and weight/theta gradients <= 1e-2, mean token KL <= 1e-4;
   fp32 spikes on logits / state <= 1e-2, gradients <= 5e-2, KL <= 1e-3 (count flips); bf16 AMP: <= max(the fp32 bar,
   2 x the bf16 loop's own error). Gradient errors are per parameter (the worst adapter, the worst other weight/theta,
   and the kernel-only att decay / a / kk group on its own), and each case also runs at T = 512 with checkpointing and
   at a ragged T = 50 (tiny cortex, spikes on); every fla case asserts the kernel actually ran. The bars are set before
   data [E]: they allow TF32 inside fla's Triton dots. A
   failure is not met by moving a bar; it is a mapping bug or a new addendum. `launch_3090.sh kernel` writes
   `run/KERNEL` with this decision and queues the parity test (a skip counts as a failure); prep and smoke refuse until
   `run/PARITY_OK` exists for the current code (hash of the twin, plastic_cortex, the E1 harness, the test and fla's
   installed RECORD), and the gate refuses a smoke whose recorded kernel differs from `run/KERNEL`. **The throughput gate is unchanged** (200-step smoke,
   >= 10k tok/s, else STOP).

   **Amendment A1 (2026-11-27, pre-data: no E1 data exists; prep, smoke and training have not run).** The first GPU
   parity run (`run/parity/log.txt`, `tiny_off.json`, `teacher_off.json`, `teacher_on.json`; 8 passed, 5 failed) showed
   three things the bar above did not anticipate. It is amended once, for these reasons, before any data:
   - *Spikes on, per logit, is not a meetable bar.* Spikes off at fp32 the kernel matches the loop on the teacher
     (logits 3.3e-4, KL 1.6e-6, gradients <= 0.008), so the mapping is right. Spikes on, fla vs the loop is KL 1.49,
     and the loop against itself (bf16 AMP vs fp32) is KL 1.59: integer counts are discontinuous, any numeric
     difference flips some count and the flip cascades. No implementation, the loop included, meets a per-logit bar
     there. **Replaced** (spikes off per logit unchanged) by a statistical criterion on the 0.1B teacher, each fla read
     compared with the loop at its own precision, with floor F = the loop's own bf16 AMP vs fp32 difference and bar
     max(1.5 F, an absolute epsilon): S1, per-token NLL on held-out WT103 val (8 windows x 512 tokens): rms <=
     max(1.5 F, 1e-3) and |mean| <= max(1.5 F, 0.01 nats); S2, spike counts per channel over every site (mean |count|
     per token and firing share): mean |diff| <= max(1.5 F, 1e-4); S3, 50 optimizer steps of the real `train()` (arm
     a, seed 7, seq 256, micro-batch 4, warmup 10, teacher targets and student through the kernel, checkpointing on):
     fla bf16 AMP vs the loop bf16 AMP, mean |d loss| per step and |d mean loss over the last 10 steps| <= max(1.5 F,
     1e-5), F from the loop's own bf16 vs fp32 runs. The tiny-cortex spikes-on cases (T = 96, ragged 50, 512 with
     checkpointing) now assert that the kernel ran (call count, and the graded final state not bit-equal to the loop);
     when no count flips (fp32 logits bit-equal) the spikes-off fp32 state and gradient bars apply, since the kernel is
     then the only difference.
   - *The "kernel not used" failures were not a bypass.* On the tiny spikes-on, T50 and T512-checkpointed cases the
     call counter passed (fla ran 4 / 4 / 8 times per pass, as expected) and only the "logits not bit-equal" check
     failed. Every path (spikes on, ragged T, checkpoint recompute, lower_bound, T > 1) reaches `_wkv_fla`; the
     kernel's ~1e-5 difference is below the rounding margin of the counts at the `o` site and the head, so the counts
     and hence the logits are unchanged while the graded state differs. Reproduced on CPU with a kernel-sized
     perturbation of the loop (`test_spike_counts_absorb_small_kernel_differences`). The check is now on the state.
   - *bf16 AMP with fp32 kernel inputs failed the gradient bar* on the teacher, spikes off: worst weight gradient 0.67
     (blocks.4.att.a2) > 2 x the loop's own bf16 floor (0.27). The gradient bar was already relative to the loop's
     floor (2 x, every metric, as the logits); it is kept and not moved. bf16 kernel inputs under AMP passed every
     bar on the teacher (weight gradients 0.21 <= 0.27, logits 7.3e-3 <= 1.5e-2, KL 5.3e-4 <= 1.1e-3). **So E1
     trains under bf16 AMP with bf16 kernel inputs only** (`--wkv-dtype bf16`, now the default; `train()` refuses fla
     + bf16 AMP + fp32 inputs); fp32 kernel inputs are allowed only with `--precision fp32`, where they passed. The
     AMP fp32-input case is still measured and recorded, not asserted. Its cause is not diagnosed (fla itself warns
     that its chunk kernel's fp32 path is unsupported on some platforms).
   Throughput gate, scope (training only; eval, prep and T = 1 on the loop) and the mapping are unchanged.

   **Amendment A2 (2026-11-27, pre-data for E1: no E1 science data exists; prep, smoke and training have not run.
   This is an instrument gate, the kernel parity test, not an E1 measure).** A2 is written **after seeing a failing
   S1**. The parity run under A1 (IEEE fix not yet in; now `run/parity/run1_tf32/`) passed 17 of 18 tests and failed S1
   |mean dNLL|: fla fp32 vs loop fp32 0.0586, fla bf16 AMP vs loop AMP 0.0134, bar max(1.5 x floor 0.0005, 0.01) =
   0.01. That run stays on record **as failed**; A2 does not reinterpret it. The diagnosis
   (`run/parity/bias_diagnosis.md`) found two things:
   - *A real kernel defect, fixed.* fla's fp32 `tl.dot` ran as TF32 (Triton's default). That shrank fla's fp32 output
     toward zero in all 12 layers (63% of the o-site count flips lowered |count|). With IEEE fp32 dots the error is
     fp32 rounding (3e-7 to 2e-6, the size of an fp64 recurrence against the loop). The bf16-input path is unbiased.
   - *The A1 mean bar could not discriminate.* The spikes-on twin is chaotic (one count flip cascades), so per-token
     NLLs of two correct implementations are close to independent draws. On 8 windows x 512 tokens the mean dNLL has a
     window SE of 0.03-0.06 nats against a 0.01 bar: a correct kernel passes 18-50% of the time. IEEE fla read 0.0117
     (SE 0.015) and an fp64 loop, more accurate than the reference, read 0.0054 (SE 0.012); a sign-free perturbation of
     the TF32 error's size missed 0.01 in 5 of 6 seeds. The floor's 0.0005 was a lucky draw (SE 0.043).

   **Changes** (implemented in `brain/tests/test_spiking_twin_fla.py`, CPU-tested there):
   1. *Sample.* S1 and S2 read 64 windows x 512 tokens (32,768 tokens; 8x A1) of held-out WT103 val. The windows are
      drawn once by a fixed seed (1127, `numpy.random.default_rng(...).choice`, no replacement, sorted) from the
      non-overlapping 513-token grid of the whole val file (489 windows with the rwkv vocab), excluding grid windows
      0-1, which stay the calibration windows (as in A1). The same 64 windows are read by every arm (loop fp32, loop
      bf16 AMP, fla fp32, fla bf16 AMP), so every comparison is paired. Window ids are written to the JSON. The seed
      was picked before the windows were drawn; one of the 64 (window 9) was among A1's 8.
   2. *Statistic.* The paired mean dNLL with an item-clustered SE: the window is the item (each starts from a zero
      state), SE = sd of the 64 per-window mean dNLLs / 8; 95% CI = mean +- t(0.975, df 63) x SE (t = 1.998).
      Projected SE at 64 windows [E, from the 8-window SEs / sqrt 8]: fp32 about 0.005, bf16 AMP 0.015-0.019.
      **This is an assumption**, not a measurement: it takes the windows as independent draws from one distribution
      and the 8-window SD as that distribution's SD. A CPU bootstrap of the existing per-window means (pooled from
      the correct-kernel comparisons in `bias_diag_*.json`: fp32 IEEE chunk, fused_recurrent, fp64 loop and the
      reversed IEEE error, 32 window means; bf16 chunk, fused_recurrent, reversed bf16 error and the floor, 32;
      centred, resampled 64 at a time) gives SE 0.0044 (fp32) and 0.016 (bf16) [E]. If the true SD sits at the upper
      90% bound of an 8-window estimate (df 7), the bf16 SE is 0.026-0.030 and a correct bf16 kernel passes the rule
      only 0.68-0.74 of the time [E]; the repeat (item 8) is the recourse for that.
   3. *Mean rule.* Each fla comparison passes S1-mean only if the 95% CI includes 0 **and** |mean| <= 0.03 nats.
      Why 0.03: E1 measures a teacher-twin gap of about 0.3-0.4 nats (the memo
      `docs/plans/2026-11-27-spiking-cortex-route-memo.md`: SpikingMamba, distilled from a same-size Mamba2, WT103 ppl
      26.32 vs 19.56 at 130M = 0.30 nats, WikiText-2 15.17 vs 10.42 at 1.3B = 0.38 nats [C]). A kernel bias is
      tolerable only if it is at most 10% of the smallest of these, 0.03 nats, so the kernel cannot move E1's main
      read-out by more than a tenth. The CI part catches any bias the sample can resolve; the cap bounds what the
      sample cannot. A correct kernel passes with probability about 0.95 (fp32) and 0.89-0.95 (bf16, where the cap
      binds) [E]; a 0.05-nat bias passes the bf16 comparison 10-15% of the time and the fp32 one almost never [E].
      Bootstrap (item 2's assumption): a correct kernel passes 0.954 (fp32), 0.936 (bf16), both 0.892; a 0.05-nat
      bias passes bf16 0.098 and fp32 0.000; a 0.03-nat bias passes bf16 0.48 [E]. **The mean rule is a weak TF32
      detector.** The diagnosis never showed that the TF32 kernel bias moved mean NLL at all: the TF32 run read
      -0.0586, the same error with its sign reversed read +0.005, and a random-sign error of the same size spread
      -0.068..+0.013. So "a 0.06-nat TF32-kind bias" is a hypothetical NLL effect, not a measured one. What guards
      against a TF32-type defect is the `ieee` assertion (item 5) and the per-layer statistic (item 7), not this CI.
      A1's mean bar max(1.5 F, 0.01) is retired; the floor's mean and CI are still computed and recorded.
   4. *Unchanged.* The floor-based bar max(1.5 F, absolute epsilon) stays for S1 rms (1e-3), S2 counts and firing
      share (1e-4) and S3 training (1e-5), with F now measured on the same 64 windows for S1/S2. Spikes-off bars,
      tiny-cortex cases, the throughput gate, scope and mapping are unchanged. S1 now costs about 8x its A1 GPU time [E].
   5. *Kernel config.* IEEE fp32 dots are part of the kernel: `brain.spiking_twin.fla_import` sets
      `TRITON_F32_DEFAULT=ieee` before importing fla and refuses any other explicit setting; `run/KERNEL` records
      `f32_dot=ieee`; every parity JSON records `triton_f32_default`, and S1 asserts it is `ieee`. The setting also
      reaches fla's internal fp32 dot on the bf16-input path, so every parity case is rerun, not only S1. The fp32
      bars were set allowing TF32 and are not tightened.
   6. *Record.* `run/parity/run1_tf32/` (the TF32 run, 17/18, S1 failed) stays as a failed run. The rerun overwrites
      the top-level `run/parity/*.json` and `log.txt`.
   7. *Per-layer kernel-bias check (asserted, added after the skeptic).* A new parity case
      (`test_gpu_kernel_bias_per_layer_teacher`, JSON `teacher_on_kernel_bias.json`) reads the first 8 S1 windows
      with the loop fp32 and hands each layer's WKV inputs to fla on the SAME inputs (`DiagTwin._iso_layer` from
      `brain/experiments/spiking_twin_fla_bias_diag.py`, fp32 IEEE and bf16 inputs). fla fp32: |slope of
      (y_fla - y_loop) on y_loop| <= 1e-5 in every layer (IEEE read 1.5e-6 at most and the fp64 loop 1.4e-6; TF32
      read 2.9e-4 at least). Both paths: when there are at least 1000 o-site count flips over all layers, the share
      of flips that lower |count| lies in 0.47-0.53 (bf16 inputs 0.4987-0.5002; TF32 0.63). With fewer flips
      (IEEE fp32 had 10), the share is recorded, not asserted. bf16 slopes are recorded only (up to +4e-4 with no
      sign pattern; the diagnosis found that path unbiased by its flips). These bars were set after seeing the
      diagnostic numbers on A1's 8 windows. They test a new property (a sign-consistent shrink) that was not run
      before, and they fail the retired TF32 config by a factor of 29 (slope) and 0.10 (share).
   8. *One pre-registered repeat (the only recourse, added after the skeptic).* Two comparisons, each with a 5%
      false fail from the CI and the bf16 cap binding, give a correct kernel about an 11% chance (bootstrap) to fail
      S1-mean. Up to 30% if the SE is at its upper bound. Under "no further amendment", that would send E1 to training
      without fla. So: if the A2 rerun fails **only** the S1 mean rule (every other parity case and S1/S2 bar
      passes, `ieee` asserted, item 7 passes), S1 runs **once** more on a second disjoint draw: 64 windows by seed
      **1128** (fixed now), drawn from the same grid minus the calibration windows and the first draw's 64. Every arm
      reads it, as in item 1. S1-mean then passes only if each fla comparison passes the A2 rule **on draw 2 and on
      the pooled 128 windows** (window SE over 128, t df 127 = 1.9788). The other S1/S2 bars stay on draw 1, and
      draw 1's mean verdict is recorded. Run by `S1_REPEAT=1 WKV_DTYPE=bf16 bash launch_3090.sh kernel`. The script
      refuses unless the last parity run was A2 and failed the mean rule alone. It keeps that run in
      `run/parity/rerun_draw1/`, refuses a second repeat, and the test writes `teacher_on_stat_repeat.json`.
      Bootstrap [E]: with the repeat, a correct kernel passes 0.958 (was 0.892). A 0.05-nat bf16 bias passes 0.095
      (was 0.093), because the pooled rule blocks it. A 0.03- or 0.05-nat fp32 bias never passes. The alternative was
      to drop the CI clause for bf16 and keep the cap alone. It was not taken: at SE 0.015-0.019 the CI and the cap
      bind at nearly the same |mean|, so dropping the clause cuts little of the false fail. Owner may swap it before
      the rerun.

   **Why this is not a rescue.** The A1 bar failed a run, but the diagnosis shows it could not tell a correct kernel
   from a biased one: at 4096 tokens a pass or fail was close to a coin flip, so the failure carried no information
   about the kernel beyond what the per-layer statistics showed (and that defect, TF32, is fixed, not waved through).
   A2 does not widen the bar on the old sample. It raises power (8x tokens, a paired clustered test) and fixes the
   bias ceiling now, from a quantity outside this test (the gap E1 measures), before the rerun. Read against run1's
   numbers A2 would fail the fp32 TF32 comparison (0.0586 > 0.03) and pass bf16 (0.0134, SE 0.048); neither counts,
   because run1 used the old windows and the retired TF32 config. **The rerun is the test.** If it fails S1 under A2,
   that is a kernel bias (diagnosed per layer, as in `bias_diagnosis.md`) or E1 trains without fla; no further
   amendment of the S1 bar. The single repeat in item 8 is part of A2, with its seed fixed before the rerun, and it is
   the only recourse. A failed repeat is final.

   **A2 skeptic verdict (fresh-context review, 2026-11-27, before the rerun): APPROVE-WITH-CHANGES.** What held:
   - The noise estimate. Run1's fp32 miss was t = -1.05; the fp64 loop read 0.0054; the sign-free injection missed in
     5 of 6 seeds.
   - It is not a rescue. The ceiling was fixed before the data, run1 stays failed, and TF32 was fixed rather than
     passed.
   - The 0.03 cap is derived from outside this test (10% of the 0.30-0.38 nat E1 gap). It is somewhat arbitrary,
     and the text says so.

   Problems and what was done:
   1. The mean test is a weak TF32 detector. Stated in item 3; the per-layer slope and flip-asymmetry check is added
      and asserted (item 7).
   2. The power numbers were projections. Labelled as an assumption and bootstrapped from the per-window data
      (items 2-3). The bootstrap still assumes independent, identically distributed windows.
   3. The false-fail rate is about 10-20%. One fixed-seed repeat on a disjoint draw, with a pooled rule, was added
      (item 8). The skeptic left the choice between this and the cap-only alternative to the owner; the repeat was
      chosen (reason in item 8).
   4. Window 9 overlaps A1's 8 windows. Already disclosed; minor; unchanged.
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
