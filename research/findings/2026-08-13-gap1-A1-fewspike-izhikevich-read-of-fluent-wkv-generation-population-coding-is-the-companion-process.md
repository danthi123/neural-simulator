---
type: finding
status: qualified
verdict: 6-seed GO at P>=8 (the POPULATION-coded few-spike Izhikevich read reaches ideal-sampler parity on the fluent WKV deep-context generation); the NAIVE single-neuron few-spike read is 0/6 (loses ~half the distribution + degrades fluency). NOT "fully spiking" / NOT "retires the mouth": the graded state->logits step is still a host matmul over the graded conductance + a BPTT-trained store; the top-K candidate set + the labelled-line drive are host inputs; runner-only, default-off. This is the FIRST time the fluent open-prose WKV generation is read onto the PRODUCTION few-spike Izhikevich read regime.
lane: gap#1 / A1 (brain-native open-prose generation — the production few-spike READ regime)
date: 2026-08-13
mechanism: few-spike Izhikevich POPULATION soft-WTA next-word read (Buesing-Bill-Nessler-Maass 2011 noise-driven WTA + population coding) replacing the host argmax/temperature-sample over the graded WKV read-out logits
artifacts:
  - research/runners/_wkv_fewspike_read_derisk.py
  - research/findings/raw/_wkv_fewspike_smoke.json
  - research/findings/raw/_wkv_fewspike_6seed.json
  - research/findings/raw/_wkv_fewspike_6seed.log
---

# gap#1 / A1 — reading the FLUENT WKV open-prose generation onto the PRODUCTION few-spike Izhikevich read regime: POPULATION coding is the load-bearing companion process (6-seed GO at P>=8)

## The precise residual this attacks (mapped from the record, un-served before this)

Deep research (RAG + the A1/A1a corpus, `before_you_build`) established that the A1 "open arbitrary prose" residual is
NOT a deep-context CREDIT wall — the 2026-08-12 correction (`2026-08-12-gap1-A1-deep-context-...`) showed that on the
DIAGONAL WKV store a transport-free local rule TIES the BPTT ceiling at adequate capacity, so there is no rate-level
credit-quality wall. The genuine residual named by the mission is the **production Izhikevich FEW-SPIKE READ regime**:
generating open arbitrary prose that survives reading the next word **from a small number of spikes**, at scale.

Tracing the record pinned the residual to a DISJOINTNESS between the two brain-native generation tracks:

| track | what it does | how it READS the word | limit |
|---|---|---|---|
| **WKV/SSM (RF-phase, 2026-07-20 GO)** | generates FLUENT, coherent, multi-clause open TinyStories prose on the spiking substrate (fully-synaptic RF-phase input into a graded `cp_ssm_state`) | **HOST argmax / temperature-sample** over the graded read-out logits `head_w @ (r_h*(Wo_sp@state))+head_b`, `state` = a high-precision graded conductance | the word read is NOT from spikes |
| **Buesing-Maass soft-WTA (followon2/neural-wta-word-decode, GO)** | reads a categorical winner from a FEW Izhikevich spikes (`cp_firing_states`, one-of-K FS-WTA) — the production few-spike read regime | genuine few-spike Izhikevich WTA | single-clause SVO, vocab <=150 |

So fluent multi-clause coherence lived ONLY on the host-argmax-over-graded read; the production few-spike Izhikevich
read lived ONLY on single-clause SVO. **Nobody had fed the fluent WKV deep-context next-token distribution (large
vocab, near-tied peaked long-tail, AUTOREGRESSIVE) into a few-spike vocab WTA and asked whether fluent generation
SURVIVES the read regime.** This de-risk does exactly that.

## The instrument (isolates the READ; holds the validated graded state fixed)

`research/runners/_wkv_fewspike_read_derisk.py` (NO `sim/` edit; drives + reads public bridge arrays; cfg.seed-
controlled per the seed trap). It reproduces the DEPLOYED generation read-out via the rate-SSM analog
(`ap=decay*ap+relu(v); an=decay*an+relu(-v); logits=head_w @ (sigmoid(Wr@LN(emb))*(Wo_sp@[ap,an]))+head_b`), which the
2026-07-20 RF-PHASE finding validated as near-perfectly correlated with the on-bridge `cp_ssm_state` (that finding's
map_corr) — so the STATE is the already-validated graded conductance and the **word READ-OUT is the sole variable
under test**. It then replaces `argmax(logits)` /
`sample(softmax(logits/T))` with a genuine Izhikevich few-spike soft-WTA over word-candidate pools (the followon2 bank,
replicated without the taxonomy/PPMI baggage): the top-K candidates by logit drive their pools (labelled-line place
code — a legitimate host INPUT, same status as a reservoir `W_in` / the retinal render), OU membrane noise makes the
winner stochastic ~ softmax(drive/T), the winner is read from `cp_firing_states` accumulated spikes over a SHORT window
(the few-spike budget). **P neurons per candidate = POPULATION coding.**

**Decisive, calibration-robust metric.** The few-spike read is a SAMPLER; the honest ceiling is an IDEAL host sampler
over the SAME top-K softmax. `ondist_mass(arm)` = mean_positions of `p_model[token the arm chose]` (higher = more
on-distribution); the ceilings are `host_sample` = E[sum p^2] and `host_argmax` = E[max p].
`read_fidelity = ondist_mass(fewspike) / ondist_mass(host_sample)` — 1.0 means the few-spike read is AS on-distribution
as an ideal sampler; << 1 means quantization noise flattened the peak. Plus top-1 argmax-agreement, mean spikes/read
(quantifies "few-spike"), and FREE-GENERATION survival (the model's self-NLL of its OWN few-spike-generated
continuation under the graded read-out — the "stays on the fluent manifold" test — with the prose itself saved).

**Anti-cheats** (each MUST collapse, all pre-registered, checked in-runner): equal-drive (all active pools driven
equally -> uniform over the active set: the drive MAGNITUDE is load-bearing); scramble (decode the true winning pool
through a fresh random pool->word labelling -> argmax agreement to chance 1/K: the labelled-line map is load-bearing);
noise-ablation (ou_std->0 -> deterministic argmax-over-drive: the OU noise IS the stochasticity, not a host RNG);
provenance (winner read from `cp_firing_states`, bridge advanced, 0 host categorical draws on the read path).

## RESULT — the population-coding lever (V=1000 6-seed WKV set, seeds 42/43/44/100/101/102)

<!--derived-->

Aggregate across 6 seeds (`_wkv_fewspike_6seed.json` summary; base_pA=60, top-K=64, sample-temp 0.8; SIM_BACKEND=numpy,
whole run 31s — the pools are <=1024 neurons, CPU beats GPU launch overhead here; matching job to bottleneck):

| operating point | mean spikes / read | read_fidelity mean (min) | argmax_agree | mechanism GO (read_fid>=0.90) | full 7-check gate |
|---|---|---|---|---|---|
| rw20 **P=1** (naive single-neuron) | ~7 | **0.557 (0.422)** | 0.333 | **0 / 6** | 0 / 6 |
| rw20 **P=8** (population) | ~56 | **1.033 (0.936)** | 0.725 | **6 / 6** | 5 / 6 * |
| rw20 **P=16** (population) | ~115 | **1.218 (1.140)** | 0.853 | **6 / 6** | **6 / 6** |
| rw40 **P=1** (naive single-neuron) | ~29 | **0.459 (0.326)** | 0.289 | **0 / 6** | 0 / 6 |
| rw40 **P=8** (population) | ~236 | **1.207 (1.123)** | 0.854 | **6 / 6** | **6 / 6** |
| rw40 **P=16** (population) | ~475 | **1.285 (1.223)** | 0.915 | **6 / 6** | **6 / 6** |

(* rw20-P8 full-gate 5/6: seed 43 is a razor-edge miss on ONLY the scramble control's tight `< 2x chance` threshold —
argmax_agree_scramble 0.032 vs threshold 0.031 (8 vs ~4 expected chance hits over n=250, a ~2sigma Poisson fluctuation);
its read_fidelity is 0.999 and every OTHER control passes. The scramble control is at chance within Poisson noise on
all 6 seeds (0.008-0.032 vs chance 0.0156); the tight threshold, not the mechanism, produces the single miss.)

**The headline.** The NAIVE single-neuron few-spike read (P=1) is **0/6 GO** — it recovers only ~46-56% of an ideal
sampler's on-distribution mass; the Poisson quantization of ~7 spikes over a large-vocab near-tied distribution flattens
the peak. Adding POPULATION coding (P>=8, ~7 spikes/neuron x 8 neurons) lifts read_fidelity to **>=0.936 on every seed**
(mean 1.03-1.29): the population-coded few-spike Izhikevich read is **at-or-above ideal-sampler parity**. read_fidelity
slightly EXCEEDS 1.0 because population averaging lowers the effective sampling temperature (a slightly-sharpened
sampler), and it stays BOUNDED BY the greedy-argmax ceiling (`mass_fewspike <= mass_argmax` on every seed — an internal
consistency check the instrument passes), i.e. it interpolates between the unbiased-sampler and greedy ceilings exactly
as a soft-WTA with more population averaging should.

**Population coding IS the companion process the host argmax replaced with a constant.** The wall-discipline question —
*"what does biology run ALONGSIDE the few-spike read that we replaced with a constant?"* — answers cleanly: the host
argmax replaced a POPULATION of competing word-assembly neurons + a homeostatic gain with a single exact max over graded
logits. Restore the population and the read regime is no longer lossy.

### Free-generation SURVIVAL (the prose itself, seed 42, `--gen-no-unk`, temp 0.8)

<!--derived-->

The metric is confirmed by the prose. Reading each next word from genuine Izhikevich spikes:

- **P=1 (naive, self-NLL 2.6-3.7 — DEGRADED):** *"once upon a time in a garden sure what a little crayon was inside for
  dinner next it to her room in many friends and the birds laughed liked her new things..."* — the quantization noise
  pushes generation off the fluent manifold.
- **P>=8 (population, self-NLL 0.8-1.7 — FLUENT):** *"once upon a time there was a little boy named tim was very excited
  to play with his friends and they both laughed and played together every day and had lots of fun in the park they saw
  a big tree..."* / *"tom and his dog were happy to help him feel better and happy again and they played together every
  day and had lots of fun in the park..."* — fluent, coherent, multi-clause TinyStories prose, every word read from a
  POPULATION of Izhikevich neurons emitting a few spikes each. self-NLL is at-or-below the full-temperature host-sample
  ceiling (2.278), so the population few-spike read is not just on-distribution but avoids BOTH the argmax mode-collapse
  (to `<unk>`/function-words) AND the P=1 noise degradation.

Anti-cheats (all 6 seeds, P>=8): equal_drive collapses to ~0.01-0.04 (vs mass_fewspike ~0.27-0.39) — the drive
magnitude is load-bearing; scramble -> argmax agreement 0.008-0.032 == chance 1/64 within Poisson noise — the
labelled-line map is load-bearing; noise-ablation deterministic (True/6); provenance 0 host draws, bridge advanced,
winner from `cp_firing_states`; silent_frac 0.0.

## VERDICT (honest, non-overclaimed)

**6-seed GO at P>=8: the population-coded few-spike Izhikevich read carries the fluent WKV deep-context generation onto
the production few-spike read regime at ideal-sampler parity, and free generation stays fluent.** The FIRST time the
fluent open-prose WKV generation is read from a small number of spikes rather than a host argmax over the graded state.
The load-bearing finding is the companion process: **the naive single-neuron few-spike read (0/6) does NOT preserve the
distribution or the fluency — POPULATION coding does.**

## Honest scope / declared residuals (the biologization targets, NOT hidden)

- **NOT "fully spiking" (TERMS.md).** The de-risked piece is the next-word READ (host-argmax -> few-spike Izhikevich
  population soft-WTA). The state->logits step upstream is STILL a host matmul over the graded conductance
  (`head_w @ (r_h*(Wo_sp@state))`), and the WKV store weights are BPTT-trained (a tracked scaffold — though the credit
  is now shown tractable by a local diagonal rule, 2026-08-12). Routing the graded logit projection itself through
  read-out neurons (so the DRIVE is a synaptic current, not a host matmul read) is the next rung.
- **The top-K candidate set is a host argpartition** (biologically: only above-threshold word-assemblies are active,
  but the SELECTION of the K is host); the **labelled-line logit->current drive is host-DESIGNED**, not learned
  (`feedback_spiking_structure_must_self_organize`).
- **No lateral inhibition yet.** The pool is independent Izhikevich + OU noise + argmax-over-firing; the production
  few-spike word-decode (`_neural_wta_word_decode`, `fswta_drive`) adds a shared inhibitory FS pool that SHARPENS the
  winner — the obvious next polish (likely lets P drop below 8 for the same fidelity, cutting the spike budget).
- **V=1000 checkpoint** (the 6-seed matched set). The fluent V=4000 checkpoint is single-seed; confirming the read
  regime holds at V=4000 (where the near-tied top is denser) is the scale follow-on.
- **NOT wired / default-off / runner-only** — this is a de-risk, not a production integration (TERMS.md `wired`).
- speed is secondary (per the mission): the population read spends ~57-475 spikes/read; the FS-inhibition polish is the
  lever to reduce it. The point here is FIDELITY of the read regime, not its cost.

## The single most promising next lever

Add the shared-inhibitory FS-WTA (`build_fswta_score_bridge`/`fswta_drive`, the production few-spike word-decode
mechanism) to SHARPEN the winner, and re-map the read_fidelity-vs-(population P, window) surface — the hypothesis is
that lateral inhibition buys the same fidelity at a much SMALLER spike budget (P<8), because it removes the runner-up
noise the population averaging currently has to out-vote. Then route the graded logit projection through read-out
neurons so the DRIVE is a synaptic current (retiring the host matmul on the read path), and confirm at V=4000.

## Files
- Runner: `research/runners/_wkv_fewspike_read_derisk.py`
- Raw: `research/findings/raw/_wkv_fewspike_6seed.json` (+ `.log`), `research/findings/raw/_wkv_fewspike_smoke.json`
- Builds on / reframes: `2026-07-20-gap1-RF-PHASE-ENCODE-...` (the fluent WKV generation),
  `2026-08-12-gap1-A1-deep-context-...` (no credit wall), `2026-08-12-vocab-agnostic-spiking-openended-generation-...`
  + `_followon2_spiking_wta_sampler` (the few-spike categorical read at small vocab),
  `2026-08-10-neural-wta-word-decode-...` (the production few-spike word-decode),
  `2026-08-11-PRODUCTION-chat-pipeline-is-largely-HOST-...` (the read-out is host argmax over graded values).
