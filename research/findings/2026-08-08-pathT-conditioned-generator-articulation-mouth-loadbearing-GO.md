---
type: finding
status: contributing
date: 2026-08-08
mechanism: pathT-conditioned-generator
lane: conversation/generation (Step 4 grounded free-gen, path-T-explicit)
runner: research/runners/_pathT_conditioned_generator_derisk.py
artifacts:
  - research/findings/raw/lanes/pathT/pathT_conditioned_generator_s42.json
  - research/findings/raw/lanes/pathT/pathT_conditioned_generator_s42.json.prov.json
---

# Path-T generator: the spiking-LLM as a CONDITIONED, GATED articulation mouth — faculties LOAD-BEARING, single-seed GO

**Scope.** Step 4 (grounded free-gen), path-T-explicit. Wave-1's E de-risked the condition+post-hoc-verify
LOOP but rendered only a host STUB ('gladly apple big cat'). This replaces the stub with REAL conditioned
MULTI-SENTENCE generation from the spiking Qwen forward, and — per the owner steer (2026-08-08,
`feedback_scaffold_ok_as_conditioned_articulation_if_faculties_load_bearing`) — proves the acceptance test that
matters: **lesion a faculty and the conversation changes** (real + matched SHAM), not "is there a transformer".

## What runs (one turn about a topic entity)

1. **RETRIEVE the knowledge-neighbourhood — BRAIN-BASED.** The brain's own spiking recall
   (`agent.what_does(topic, action)` = a VSA unbind of the RF-phasor store, abstaining where no fact is bound)
   enumerates the topic's grounded SVO neighbourhood; dlPFC spiking spreading (`agent.elaborate`) orders it.
2. **CONDITION the generator — the Broca-like MOUTH (SCAFFOLD).** The real spiking Qwen faculty
   (`SpikingQwenFaculty`; spiking ops installed + enabled — the run records `spiking_ops_enabled: true`, 49
   RMSNorm + 24 SiLU ops at T=16) is prompted with the retrieved neighbourhood and writes a coherent short
   multi-sentence reply.
3. **POST-HOC no-confab MOAT, per PROPOSITION.** Each sentence is re-parsed to an SVO and CHECKED against the
   store by the spiking moat read (`agent.is_it_true` = `ask_yes_no` unbind). Any proposition that does not
   read `yes` is a CONFABULATION and is DROPPED. Labelled **SCAFFOLD + POST-HOC-VERIFY — NOT "moat GO"**.

## Result — single-seed smoke (seed 42), GO

The mouth is genuinely conditioned. Intact, it renders the retrieved facts faithfully; e.g. dog →
*"A dog chases a cat because it enjoys chasing its prey. A dog eats a meat because it needs protein to survive.
A dog likes a bone because it finds bones edible and tasty."* — 3 propositions, all verify against the store.

<!--derived-->

| check | intact | REAL lesion | SHAM lesion | verdict |
|---|---|---|---|---|
| **A. world-model/memory → content** (fidelity, 3 parseable turns) | 1.00 | 0.00 | 1.00 | load-bearing |
| **B. honesty (post-hoc moat) → confabs emitted** (real-lesion text) | — | 0 (moat ON) | — | load-bearing |
| B, moat OFF on the same confab-laden text | — | 8 | 0 (sham) | load-bearing |
| matched-sham teeth: `txt_sham != txt_intact` (all parseable turns) | | | true | verified |
| seed seeds substrate (hash cp_neuron_firing_thresholds ×2) | identical | | | verified |

**A — world-model is load-bearing (content).** REAL lesion = corrupt the CONTENT the brain supplies: scramble
the topic's own retrieved neighbourhood before conditioning (each true patient → a patient from an UNRELATED
entity, matched size + same SVO structure). The mouth then renders the SCRAMBLED content verbatim — e.g.
*"A fox chased a cat… ate a shade."* — so it still emits the SAME NUMBER of re-parseable propositions
(candidates > 0) but NONE verify against the true store → fidelity 1.00 → 0.00. **TEETH (non-tautological):**
the real lesion does NOT silence the mouth or zero the metric it tests — the fidelity drop is WRONG content, not
NO content. This is the exact trap Wave-1's B was refuted for.

**HARDENED matched SHAM (this revision) — the vacuous A-sham is fixed.** Wave-2a fed the SAME true prompt, so
`txt_sham == txt_intact`: the sham arm literally re-ran intact and proved nothing. The hardened sham is a
**SURFACE-axis** perturbation — the SAME true facts and the SAME "state one of the facts" content-lock, but the
mouth is additionally required to **NUMBER each sentence**. The generation genuinely differs
(*"1. The dog is chasing a cat. 2. The dog enjoys eating meat. 3. The dog likes to chew on a bone."*,
`txt_sham != txt_intact` on every parseable turn = **teeth**) yet fidelity **holds at 1.00**, because the digits
are not `[a-z]+` content tokens so the SVO re-parse is unchanged. This is the thesis's dissociation made into a
control: the **brain supplies content, the mouth supplies surface** — corrupting content collapses fidelity;
perturbing surface does not. **Two looser shams were tried and REJECTED as confounded** on the 0.5B mouth
(honest-negatives): a past-tense/own-words **paraphrase** lets the weak mouth DROP the specific patient
(*"A dog chased its prey"*) → content drift (fidelity 0.0, not a surface-only change); **scrambled-distractor
injection** BLEEDS into topic propositions (*"The dog chases a rabbit"*) → a diluted real lesion (~0.72), not
off-target. Numbering keeps the content-lock verbatim, so it is the clean off-target control.

**B — honesty (post-hoc moat) is load-bearing.** On the confab-laden real-lesion text: moat ON drops all 8
confabulations (0 reach the user); the LESION (verify OFF) emits all 8; the matched SHAM (same operation —
verify OFF — applied to the true-conditioned INTACT text) manufactures 0 confab. Real flips, sham does not. Each
count is computed by applying the emission policy to the props; none is hardcoded.

## Attribution (what is the brain vs a declared host shortcut)

- **Brain-based:** CONTENT retrieval is `agent.what_does` (spiking VSA unbind of the RF-phasor store) +
  `agent.elaborate` (dlPFC spiking spread); the MOAT accept/reject decision is `agent.is_it_true` (`ask_yes_no`
  unbind) — the neural half of the post-hoc verify.
- **Declared host shortcuts:** the GENERATOR (converted spiking-Qwen 0.5B forward, spiking-ops installed,
  ppl~1.0) is a SCAFFOLD mouth to biologize later; brain→generator conditioning is a HOST TEXT INTERFACE
  (facts rendered to a prompt string), not synaptic drive; sentence-split + SVO re-parse is host parsing.

## Preconditions (the verdict is conditional on these)

cfg.seed actually seeds the substrate (hash-verified, `fe3475ff29771f0c` ×2); the mouth is the converted
spiking forward (`spiking_ops_enabled: true`), not a vanilla fp16 model; the matched sham genuinely perturbs the
prompt (teeth, `txt_sham != txt_intact`); ≥2 instrument-visible (re-parseable) turns exist to score content
fidelity on (3 of 5 here).

## Honest-negatives (first-class)

- **Generator fluency itself = the UNCHANGED field wall.** The mouth is a converted 0.5B transformer
  (spiking-ops forward, ppl~1.0), NOT an emergent-from-a-learning-substrate producer. This de-risks the
  CONDITION+GATE loop around a real generator; it does NOT close the generative-fluency wall.
- **Brain→generator conditioning is a HOST TEXT INTERFACE.** The retrieved neighbourhood is rendered to a
  prompt string (host glue); the neurons do not synaptically drive the generator context. Same characterized
  boundary as all grounded-language work — declared a shortcut, not sold as neural drive.
- **Sentence-split + re-parse is HOST parsing.** The accept/reject DECISION, however, is the brain's spiking
  `is_it_true` (ask_yes_no unbind) — that half is neural.
- **Re-parse instrument BLIND on 2/5 turns.** When the mouth free-paraphrases (*"A bird enjoys munching on
  worms"*) no clean SVO is recovered, so the content-fidelity metric cannot see those turns — the world-model
  claim is scored ONLY on the 3 instrument-visible turns. The moat is conservative there: it DROPS every
  unverifiable proposition (never emits it). This is the generator-fluency/instrument wall, not a mechanism
  failure.
- **Mouth surface/content coupling (matched-sham design).** The 0.5B mouth couples surface and content, so two
  matched-sham designs are confounded (see A above): a paraphrase drops the patient; scrambled-distractor
  injection bleeds into topic props. The clean matched sham is therefore the surface-only NUMBERING perturbation.
  A characterized property of the scaffold mouth, not a mechanism failure.
- **Single-seed SMOKE** → a verdict in one foreground process. Multi-seed sweep (6 seeds) — the parent runs:

```
for s in 42 43 44 100 101 102; do PYTHONPATH=$PWD SIM_BACKEND=numpy \
  /home/dant123/Projects/sim/.venv/bin/python -m research.runners._pathT_conditioned_generator_derisk \
  --seed $s --T 16 --max-new-tokens 64 \
  --out research/findings/raw/lanes/pathT/pathT_conditioned_generator_s$s.json; done
```

## Discipline

Reuse-by-import; NO `sim/` edit (a new additive runner). Brain half numpy-CPU (`SIM_BACKEND=numpy`); the mouth
forward is its own torch-CUDA device. `cfg.seed` seeds the substrate — verified (build ×2 @ seed 42, identical
`cp_neuron_firing_thresholds` hash `fe3475ff29771f0c`). Run:

```
PYTHONPATH=$PWD SIM_BACKEND=numpy python -m research.runners._pathT_conditioned_generator_derisk \
  --seed 42 --T 16 --max-new-tokens 64 \
  --out research/findings/raw/lanes/pathT/pathT_conditioned_generator_s42.json
```
