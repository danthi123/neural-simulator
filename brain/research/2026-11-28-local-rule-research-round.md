# Local rule research round (2026-11-28): why four methods failed, what to try next, is 12-31 reachable

STATUS: research note, not a prereg. Nothing here has run. Inputs: the four 11-27 reads (`brain/experiments/results/
2026-11-27-local-sleep-01b-read.md`, `-local-sleep-shy-read.md`, `-local-rule-v2-read.md`, `-local-rule-v3-read.md`),
the v3 DESIGN, a record sweep, and two literature sweeps (search snippets only; no full paper read; 2026 items unverified).

## 1. Why all four failed (testable claim)

**Claim.** None of the four rules ever moved the adapters along the direction that lowers the output error at the
answer word. The update that gives recall is small and specific (a few layers, the positions that predict the name,
one direction in the residual), and each rule's error signal had near-zero projection onto it.
- v1, SHY, v2: the output error travelled over fixed or copied feedback into a pretrained network that was never
  aligned with that feedback. Evidence: local equals shuffled on every seed; replay loss 2.27 against gradient 0.17.
  The rule did not fit even its training text. Drift without learning is what a misdirected update looks like.
- v3: no error travelled, but the target was every layer's whole copy-mode activity at every position, unweighted.
  The fit (error down 70%) says nothing about the output-relevant part, which can sit in the 30% left over.

**Test T0 (diagnostic, about 1 pool-day at 0.1B, seed 7; backprop used as an instrument only, never as the rule).**
1. Per layer, cosine between each rule's accumulated update (v1, v3) and the gradient-sleep update.
2. Gradient sleep restricted to one layer at a time, and to answer-predicting positions only: where does recall live?
3. v3's unrun `grad_distill` arm (backprop to the same teacher's logits): bad target or bad rule?

**Prediction.** Cosine under 0.1 in the layers that carry recall; recall concentrated in a minority of layers.
**Falsified if** cosine is above 0.3 with no recall. Then direction is fine and the limit is step size or drift budget,
which is the 08-11 per-arm learning-rate artifact; the next step would be a per-layer lr sweep on v1, not a new rule.

**Second cause, already measured.** The instrument has two facts of range. Gradient recall is 3.83/12 at 0.1B and 2.33
at 0.4B over six seeds (dev seed 7 showed 5), and the bar is 2. Dev picks rest on one seed. Fix in section 4.

## 2. Candidates, ranked

### 1. Reward-gated perturbation on the adapter's rank units (three-factor rule; MeZO family)
- **What.** Inject noise into the 8 rank-unit activities per adapter during replay. Two forward passes give one scalar:
  did the replay loss (plus the keep KL) go down? Update = scalar x noise x pre-synaptic activity. No feedback path.
- **Evidence: measured at scale, secondhand.** MeZO fine-tunes pretrained LLMs up to OPT-13B, including LoRA, about 5%
  behind backprop on RoBERTa-large in the prompted setting, in thousands of steps
  ([MeZO](https://arxiv.org/pdf/2305.17333), [MeZO-SVRG](https://arxiv.org/pdf/2404.08080)). Forward gradients degrade
  above about 1.15M trainable parameters ([Panchal 2024](https://people.cs.umass.edu/~brun/pubs/pubs/Panchal24neurips.pdf));
  rank-8 adapters at 0.1B are below that.
- **Why it addresses the claim.** It optimises the same objective as gradient sleep, so its expected update is the true
  gradient direction. Nothing can misalign.
- **Record.** 07-13 retired node perturbation for "no edge over KP" on toys. Here KP itself failed, so that verdict does
  not transfer. Variance is the open question; perturbing rank units (not weights) keeps the dimension small.
- **Biology.** Synaptic noise plus a global neuromodulator. It is not synapse-local: the global scalar is a third factor.
- **Screen.** 0.1B, seed 7, forward-only, same episodes and keep bank as the gradient arm, budget 30x the gradient
  arm's forward passes. Kill: replay loss not under 1.0, or cosine to the true gradient under 0.1. Pass: recall at the
  floor under the drift caps, then six seeds, then 0.4B. Arms: shuffled scalar, wrong-sign scalar, no_sleep, gradient.

### 2. Key-value write: perturbation-found value, closed-form Hebbian storage (ROME/MEMIT form)
- **What.** Treat channel mix as a key-value memory. Key = pre-synaptic activity at the position that must predict the
  name. Value = a single residual vector at one layer, found forward-only by candidate 1's search (hundreds of
  dimensions, not hundreds of thousands). Store with a decorrelated outer product: `dW = (v* - Wk)(C^-1 k)^T / (k^T C^-1 k)`.
- **Evidence: storage formula measured at scale** on GPT-2 XL and GPT-J, thousands of facts
  ([ROME](https://arxiv.org/pdf/2202.05262v2), [MEMIT](https://arxiv.org/pdf/2210.07229v2.pdf)). **Caveats:** the
  original finds `v*` by backprop; a forward-only `v*` is theory. No RWKV result. A 2026 critique (unverified) says
  edits partly work through over-attention.
- **Why it addresses the claim.** It writes at the one place T0 is expected to point to, and `C^-1` protects ordinary
  activity by decorrelation instead of v2's blanket projection (which stopped learning). The covariance is a declared
  stand-in for an anti-Hebbian lateral circuit, as in v2's scaffold row.
- **Risk.** With a copy-mode value it collapses into v3 at one position. The value must be chosen by its output effect.
  Twelve facts exceed rank 8 at one layer: spread across layers or declare a rank change.
- **Screen.** After T0. One fact first: does a single write at the T0 layer raise the answer's log-probability at the
  question with hippocampus off? Then 12 facts, 0.1B and 0.4B, seed 7, same arms as above.

### 3. v3 with question-form targets (the design's named follow-up)
- **What.** Same code, same rule. The clamped phase is the cortex answering a cue with the telling in context; targets
  only at answer positions.
- **Evidence: theory plus our own G0** (teacher correct 12/12 with the telling in context). Task-shaped local losses
  are the ones that work ([arXiv 2601.21683](https://arxiv.org/html/2601.21683v1), unverified). Sleep replay is
  cue-driven completion ([Spens and Burgess 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10963272/)).
- **Caution.** M3 (11-03): retrieval practice lost to restudy under gradient descent (-0.5/12 at 0.1B, -2.67 at 0.4B).
  Layerwise fitting may again leave the output-relevant part unfitted. Cues must be the brain's own (self-generated or
  stored user questions), never host templates.
- **Screen.** One engine run on the existing v3 spec at 0.1B and 0.4B, seed 7, with `top_only` and `grad_distill`.
  Cheapest of the three; run in parallel with candidate 1.

Not ranked: a meta-learned adapter writer ([Doc-to-LoRA](https://arxiv.org/abs/2602.15902), unverified) is the strongest
long-horizon option but needs offline gradient meta-training. It is a 2027 scaffold question, not a December method.

## 3. Is 12-31 reachable at 1.5B? What to tell the owner

**Not as written.** Estimates, mine, not measured:
- A forward-only three-factor rule meeting the bar on six seeds at 0.1B and 0.4B by 12-31: about 35%.
- The same at 1.5B on six seeds by 12-31: about 15%. Perturbation needs tens of times more forward passes, and the
  1.5B operating point is not derived.
- A strictly synapse-local rule with no global signal: under 5%. No published rule of that kind has a measured
  fine-tuning result on a pretrained network of 100M parameters or more. Ours were the first tests at that scale.

**Say to the owner.**
1. Four methods failed for one probable reason (misdirected credit), and one diagnostic day confirms or refutes it.
2. The best remaining route uses a global "did it get better" signal. That is three-factor learning, biology-style in
   the dopamine sense, not Hebbian-only. It will be declared in `brain/SCAFFOLDS.md`; work proceeds on that basis.
3. Re-date honestly: 12-12 decision gate (a forward-only rule at half of gradient recall, six seeds, 0.1B and 0.4B).
   If met, 1.5B runs on the AWS lane and lands in early January. If not, the milestone is recorded as missed with the
   method verdicts, and gradient sleep stays the declared scaffold. The weekly transcript does not depend on this.

## 4. Prerequisites missing (PREREQUISITES.md style)

- **Credit map.** Missing: which layers and positions carry gradient sleep's recall (T0). Without it every local rule
  is aimed blind. Readiness: NOT-READY.
- **Measure with resolution.** Have: recall out of 12 with a bar of 2 and a baseline of 2.3 to 3.8. Missing: a graded
  read-out (answer log-probability and rank at the question) and a six-seed gradient operating point at 0.4B (dev 5,
  six-seed 2.33, unreconciled). Readiness: PARTIAL; a dev result of 0-2 against a floor of 2 is weakly informative.
- **Global third factor.** `modulators.py` is not built. The outcome scalar would be host-computed: declare it.
- **Replay cue form.** Questions are not stored; the hippocampus has no partial-cue self-generation. Candidate 3 needs
  it. Host-templated cues are forbidden.
- **Decorrelated keys.** Code-substitute: v2's SVD of keep-bank activity. Needed by candidate 2.
- **1.5B reference.** Not found in the four reads: a six-seed gradient-sleep reference and a forward-pass cost per
  night at 1.5B on the 3090. Confirm before any 1.5B claim.
- **Controls the record requires.** Shuffled and wrong-sign arms, per-arm learning rate (08-11), fresh seeds beyond
  the dev seed (seed-42 retractions), frozen readout (05-17).
- **Literature.** Snippets only. Read MeZO, MeZO-SVRG and MEMIT in full before the candidate 1 prereg.
