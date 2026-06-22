# Learned generative-sequence substrate → a small spiking LM, pretrained then CONSOLIDATED onto the one brain — frontier scoping (2026-06-22)

> **Status:** READ-ONLY deep-research + code/findings/literature scoping for an OWNER-APPROVED new frontier. NO `sim/`
> edits, NO experiments, NO GPU training. Single deliverable = this doc. Every load-bearing project claim re-verified
> against the repo (file:line cited); SOTA bounded by a fresh June-2026 literature pass. **This is a scoping/decision
> doc, NOT a brain-based result and NOT a commitment to build.** The controller should trust-but-verify the claims
> flagged **[VERIFY]**, then push + present before building.
>
> **What this closes.** The just-measured gap (`2026-06-22-generation-novelty-categorical-gap-MEASURED.md`): the
> production composer generates **0 novel sentences** (ratio 1.0, novel-composition 0.0, 16/16 held-out triples
> abstained) — the categorical "no free generation" wall, now MEASURED, not asserted. The §6 of the parent scoping
> (`2026-06-22-conversational-scaling-vs-dendritic-scoping.md`) named item **§5.4 — "a LEARNED GENERATIVE SEQUENCE
> MODEL on the substrate (the benched surrogate-grad SNN cortex)"** — as "the ONLY path to free generation … the
> actual unlock for the generative gap, NOT the dendrite." This doc turns that one-line item into a concrete,
> loop-structured plan, under the owner's three constraints (C1/C2/C3) and the validation loop (train → generate →
> grow → no-forget).

---

## 0. The one-paragraph answer (the rest is the evidence)

**The frontier is feasible, but its single load-bearing bet is one the project has never tested, and the framing must
change accordingly.** Every prior generative-spiking-LM attempt FAILED, but for one reason — *training a spiking LM
from scratch (or distilling into one) does not generalize on this substrate; it overfits to token-soup* — and **none
of those negatives ever reached the consolidation step.** They all died at the GENERATOR, before any transfer to the
bridge. The new frontier inverts the recipe to the one the SOTA actually uses: **pretrain a COMPETENT model first
(the project already proved a ~6M-param non-spiking Transformer reaches held-out perplexity ~6.1 and coherent
TinyStories English — `2026-05-17-generator-F`), THEN convert/consolidate it to spikes on the bridge** (the 2025 SOTA
"ANN-to-SNN conversion" path: converted GPT-2-small loses only ~5–12% cosine / ~10% perplexity; LAS/MBE conversions
are near-lossless). So the minimum-viable scale is **a 5–10M-param TinyStories-class generator (trains on the 3090 in
≤30 h; <10M params already produce fluent multi-paragraph coherent English)** — *big enough to demonstrate the full
loop, deliberately not LLM-competitive.* **C1 (one-bridge consolidation)** is feasible via `inject_explicit_wiring` but
has a genuine, named hard part: **the bridge has NO LIF neuron model** (only Izhikevich/HH/AdEx/RF), so the trained
*weights* transfer but the *LIF dynamics* do not — closing that mismatch is the core engineering risk (an additive
guarded LIF/AdEx-as-LIF path, or surrogate-grad finetuning post-conversion). **C2 (continue learning + no
catastrophic forgetting)** is the DEEPEST risk but is *exactly* where the project is strongest and the 2025 literature
agrees: SNN continual learning converges on **latent-replay + a NREM-analogous sleep phase + noise** (the project's
SWR/concept replay + sleep gates + OU noise — already validated 5/6 retention ≥80%, Phase 1.4 BRANCH A), and
brain-inspired replay/metaplasticity beats EWC at ~1% the compute. **The honest ranked plan is structured as the
owner's loop, cheapest-first: (1) a no-train ceiling-confirming feasibility probe — does the SHIPPED Phase-2.2 SNN
slice load onto the bridge and SPIKE? (hours); (2) pretrain the competent generator (the non-spiking Transformer is
the safe baseline; a small spiking-RWKV/SpikeGPT-style net is the biology-faithful target) and test REAL novel
generation; (3) convert + consolidate to the one bridge (C1); (4) the continual-learning loop + the catastrophic-
forgetting confirmation (C2).** Cloud becomes logical at exactly ONE point: when the **spiking** generator's
BPTT/conversion training (≫ ANN cost per param) is scaled past the 3090's 24 GB / ≤30 h envelope to reach the
fluency the non-spiking baseline already shows — flagged precisely in §5.

---

## 1. THE PRIOR — what is built, and the decisive correction the framing must inherit

### 1.1 Generative groundwork (verified on `main`, despite "path-f-hybrid only" headers)

- `sim/bptt_snn.py` (numpy reference) + `sim/bptt_snn_gpu.py` (CuPy/numpy) — surrogate-grad BPTT through stacked
  **LIF** layers (`LIFLayer`: leak 0.95, threshold 1.0, hard reset — `bptt_snn.py:46-86`), ATan surrogate
  (`atan_surrogate_np`, `bptt_snn.py:38-43`), full backward unroll with the reset chain-rule (`bptt_snn.py:180-256`).
- `sim/surrogate_grad.py` (ATan + fast_sigmoid CuPy surrogates).
- `sim/char_tokenizer.py` (66-char Tiny-Shakespeare vocab, one-hot, `make_seq_dataset` next-char). `sim/bpe_tokenizer.py`
  exists (a subword tokenizer is available — used by the generator-S/F arc).
- `research/runners/cortex_pretraining.py` — the numpy/GPU BPTT trainer. `train_shakespeare` builds an N-layer LIF SNN
  `V → [128,128] → V` (default 4-layer, `cortex_pretraining.py:208-231`), produces a **per-layer weight artifact**
  `<path>.npz` (`W_layer_0`, `W_layer_1`, … + `layer_sizes`/`thresholds`/`leaks`) + a sidecar `.metadata.json` vocab
  (`save_checkpoint`, `cortex_pretraining.py:328-365`). **Critical limitation to fix:** it trains on the
  **last-position target only** (`target_batch = y_shuf[:, -1]`, `cortex_pretraining.py:259`) — i.e. it is NOT a full
  autoregressive next-token LM (one supervised target per window, not one per position). A proper generative LM needs
  the all-positions loss; this is a known, cheap fix but a real gap in the current artifact.

### 1.2 The measured outcomes of the OLD Phase-2 arc (the corpus of negatives — all re-verified)

| Attempt (finding) | Mechanism | Scale | Verdict | Why |
|---|---|---|---|---|
| Phase 2.1 (master plan) | BPTT ABC toy | 3→64→3 | PASS | mechanism/grad-check only |
| Phase 2.2 (master plan) | 4-layer LIF SNN, Tiny-Shakespeare | 66→128³→66 | "PASS" (loss 14.1→2.24, ppl ~9.4) | **train loss only**; never held-out-tested for generation |
| **Phase 2.3a** (`2026-05-07`) | pretrained SNN as feature extractor for word→action | 134K | NEGATIVE | **wrong objective** (char features don't transfer to binding); cosine 0.72 |
| **Phase 2.3b 50M** (`2026-05-09`) | scale the SNN, re-test transfer | 50M (66→4096³→66) | **REFUTES scale thesis** | 375× params → inter-word cosine 0.72→**0.85 WORSE**; transfer is wrong-objective, not under-param |
| Gen-S (`2026-05-17`) | subword spiking LM, real TinyStories | 512 vocab, 256² | NEGATIVE | held-out ppl 117k–388k (**230–758× WORSE than random**) = token-soup; a relative-only gate false-PASSED it |
| Gen-D (`2026-05-17`) | **distill** a competent trigram teacher into the SNN | 256², T=32 | NEGATIVE 0/3 | closed ~99.3% of the abs-ppl gap (804) yet still > random floor 512 → **the wall is the surrogate-grad-LIF substrate's learnability**, not signal/teacher |
| Gen-inc3 (`2026-05-16`) | max-capacity char SNN | 3×512, ~600K | NEGATIVE | MEMORIZATION (held-out 5× worse than chance; REAL only 6.75% below PERMUTED) |
| Gen-P (`2026-05-16`) | predictive-coding next-concept, self-comprehension | — | TERMINAL NEGATIVE | learned order doesn't survive realization through the write-only substrate read back by its own comprehension |
| Gen-G1 (`2026-05-16`) | **HVC songbird** sequencer (`sim/song_hvc.py`), self-comprehension teacher | — | NEGATIVE 0/2 | HVC is sound; the substrate's **order-blind self-comprehension JUDGE gives 0 gradient** (mean_reward 0.0000) |
| **Gen-25M/50M ceiling** (`2026-06-02`) | scale-up spiking LM, TinyStories | 25M / 50M | NEGATIVE | held-out ppl **203,753** (≈200× worse than random); **OVERFIT, not size** — scaling 100× made held-out WORSE |
| **Gen-F** (`2026-05-17`) | **small NON-spiking Transformer**, from scratch, TinyStories | **~6M (4-layer d=256)** | **PASS 3/3** | held-out ppl **~6.1**, coherent story-shaped English; cleared the gate 9 spiking/order-blind mechanisms failed |
| Gen-E (`2026-05-17`) | classical trigram n-gram | 512 | PASS bounded | ppl 14.75; coherent only as local fragments |

### 1.3 The two corrections this scoping MUST inherit (so we don't re-tread)

**(A) The Phase-2 "DEAD-END" verdict was about word→action TRANSFER, NOT about generation.** The 50M net DID train
(loss 9.75→3.22, `2026-05-09-Phase-2.3b-50M-cosine-REFUTED.md`); it was declared dead because its char-level features
didn't transfer to the *binding* task — "Bigger models memorize regularities better but pack similar-looking features
tighter, RAISING inter-word cosine. Wrong objective for word-action transfer." That is a verdict on *using a generator
as a feature extractor for binding*, **not** on the generator generating. The new frontier keeps the generator AS a
generator — a category the old arc abandoned.

**(B) Every spiking-generator negative died at the GENERATOR, before any consolidation.** Verified across S/D/inc3/P/G1
and the 25–50M ceiling: each failed at *train-from-scratch / distill generalization* — none ever installed a trained
net on the bridge. **The consolidation/transfer hypothesis is therefore UNTESTED**, and it is exactly the field's
documented remedy (next section). This is the single most important reframe: *the prior negatives do not bound the new
frontier; they bound only the from-scratch-spiking-generator sub-step the new frontier replaces with a pretrained-then-
converted model.* **[VERIFY — most load-bearing: read the headers of `2026-06-02-generative-ceiling-...`, `Gen-D`, and
`Gen-S`; confirm (i) the failure is always overfit/learnability of the from-scratch spiking generator, and (ii) no
prior attempt reached a bridge-consolidation step.]**

**Two gate caveats the new frontier must carry up front:** (i) the generation gate MUST keep an **absolute-competence
floor** (held-out ppl < vocab) — a relative-only gate false-PASSED Gen-S on token-soup; (ii) the project's standing
BRAIN-BASED-ONLY bar classifies a backprop-pretrained generator as a host shortcut **until** it is genuinely realized
as spikes on the bridge — which is precisely the C1 step where all prior compute stopped short. C3 (below) is what
licenses the backprop pretraining as legitimate.

---

## 2. C3 — is "backprop+corpus pretraining" a legitimate developmental stand-in?

**Yes, and the project already pre-authorized exactly this framing.** `docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md`
(lines 14-27) defines the `path-f-hybrid` line: "allows surrogate-gradient backprop ONCE for 'developmental' cortex
pretraining (mirroring real cortex's slow maturational learning). **All POST-pretraining learning still uses
biology-grounded mechanisms (continual learning preserved).**" That is verbatim the owner's C3 (backprop = the long
developmental period) gated on C1+C2 (ends fully-spiking-on-bridge; keeps learning). The Complementary-Learning-Systems
mapping the owner draws — **backprop = slow neocortical maturation; hippocampal SWR consolidation + no-forgetting = the
continual back half** — is the CLS theory the project already validates (McClelland 1995; Phase 1.3/1.4). C3 holds *iff*
C1 and C2 hold; the rest of this doc is about whether they do.

**The honest sharpening:** the SOTA (§3b) is blunt that *every* capable spiking LM borrows its capability from a
backprop-trained (or converted) ANN — the capability lives in "architecture + corpus + backprop," not the neuron model.
So C3 is not just *permitted*, it is the *only* route the field knows to a competent spiking generator. The biology
claim the project can honestly make is narrower than "the brain learns language by backprop": it is **"a competent
generative cortex is installed by a developmental process (here, backprop on a corpus), and thereafter the agent
learns + grows by biology-grounded local mechanisms on the spiking substrate, without forgetting."** That is a
defensible CLS-shaped claim, and an honest negative (the consolidated net degrades, or can't keep learning) is itself
the deliverable.

---

## 3. The minimum-viable SCALE (and the precise CLOUD trigger)

**Target capability = the TinyStories ceiling, deliberately.** Eldan & Li 2023 (TinyStories): models **<10M params**
trained on the synthetic TinyStories corpus "produce fluent and consistent stories with several paragraphs … almost
perfect grammar … and demonstrate reasoning." All 1M–35M models train **on a single V100 (≈3090) within ≤30 h**
(reported: a 3070/8 GB did it in 10 h). Eval loss ~1.6–2.3 for 8–20M-param models. **This is exactly "big enough to
demonstrate the full loop (train → coherent novel generation → grow → no-forget), and explicitly NOT LLM-competitive."**

| Component | Min-viable scale | 3090 cost (local ceiling) | When CLOUD is logical |
|---|---|---|---|
| **Tokenizer/corpus** | BPE ~2k–8k vocab, TinyStories (~1 GB synthetic) — `sim/bpe_tokenizer.py` exists | trivial | never (corpus is small) |
| **The GENERATOR (non-spiking baseline)** | ~6M Transformer (Gen-F: ppl 6.1, coherent) | ✅ trains in hours, fits easily | never (already done) |
| **The GENERATOR (spiking, the biology target)** | 10–46M spiking-RWKV / SpikeGPT-style (SpikeGPT-46M reaches usable WikiText-2 ppl) | ⚠️ marginal: surrogate-grad BPTT is ≫ ANN cost/param; 30 h at ~10M is plausible, 46M is the edge of the 30 h / 24 GB envelope | **HERE — the precise trigger**: if the *spiking* generator must exceed ~46M params, or BPTT wall-clock exceeds ~30 h, to reach the coherence the non-spiking baseline shows, provision cloud (1× A100/H100, ~$50–200 for a few days). Flag the moment the spiking-BPTT loss curve plateaus *above* the non-spiking baseline at the 3090 ceiling. |
| **CONVERSION (ANN→SNN, C1)** | inference-side, no training | ✅ cheap | never (conversion is calibration, not training) |
| **CONSOLIDATION + continual loop (C2)** | the existing bridge + SWR replay | ✅ the validated path runs on 3090 | never (this is the project's home turf) |

**The single cloud trigger, stated precisely:** *only the spiking-generator's BPTT/conversion-finetune training step
(§5 step 2b/3) justifies cloud, and only once its 3090 ceiling is empirically hit (loss plateau above the non-spiking
baseline OR > ~30 h / > 24 GB).* Everything else — the non-spiking baseline, the conversion calibration, the
consolidation, the continual-learning loop, and ALL four validation-loop measurements — runs locally. **Distinguish
"enough to demonstrate the loop" (10M, TinyStories, coherent multi-sentence English) from "LLM-competitive" (~360M+
params + trillions of tokens, the documented hard wall — `2026-06-17` scoping, SpikeGPT overfit-at-scale): the latter
is explicitly OUT of scope and is NOT the goal.**

---

## 3b. SOTA — what a spiking LM can reach, and the recipe everyone uses (fresh June-2026 pass)

| System | Reaches | Recipe / what it requires |
|---|---|---|
| **SpikeGPT** (arXiv 2302.13939) | 46M & 216M; 216M → **WikiText-2 ppl 18–19 (surpasses GPT-2-small after finetune)**, WikiText-103 ppl ~40 | RWKV-style block + **surrogate-grad BPTT** + GPU pretraining. Spikes are an *activation* substitution on a near-transformer architecture. **Note vs the project's 25–50M token-soup:** SpikeGPT did NOT collapse at 46M — the difference is **architecture (RWKV recurrence) + corpus + training recipe**, not raw params. This is a hint the project's from-scratch stacked-LIF + last-position loss + TinyStories harness was the wrong recipe, not proof that ~10–46M spiking LMs can't generate. |
| **ANN-to-SNN conversion** (2025: LAS arXiv 2505.09659; training-free MBE arXiv 2508.07710; GPT-2 conversion benchmarks) | **near-lossless**: converted GPT-2-small → ~5–12% cosine loss, **~10% ppl change** (WikiText-2 5.68→6.844 naïve; LAS/MBE close most of that) | Convert a **pretrained** transformer's activations to spikes (threshold-balancing / multi-basis decay neurons). **This is the C1 path** — and it is *calibration*, not training. |
| **SpikingBrain-7B** (arXiv 2509.05276) | Transformer-comparable, 100× TTFT speedup | ANN→SNN conversion of a hybrid-attention transformer + **~150B-token** continual pretraining. Capability = transformer + corpus; spikes = efficiency. |
| **SpikeLLM** (arXiv 2407.04752) | LLAMA-7B-class accuracy | saliency-based spiking *quantization* of a pretrained LLAMA. Compresses an LLM; doesn't learn language from biology. |

**The hard external bound:** *every* spiking system that reaches LM-class capability **trains or converts a standard
(near-)transformer with backprop on a corpus, then spikes the activations.** A point-neuron, local-rule, no-backprop
substrate is categorically NOT on the SOTA spiking-LM path. **⇒ The frontier's only viable recipe is the SOTA recipe:
pretrain (or convert) a competent net, then realize it as spikes — which is precisely C1+C3, and precisely the step the
project's from-scratch attempts never took.** The biology-faithful degree of freedom is *how spiking the final
generator is and how it consolidates*, NOT whether backprop is in the developmental loop.

---

## 4. C1 — consolidating the trained generator onto the ONE bridge

### 4.1 The load-bearing facts (verified, file:line)

- **Install path = `inject_explicit_wiring(wiring_plan, …)`** — `sim/bridge.py:2393`. Wiring-plan = dict
  `population_name → {pre_indices, post_indices, initial_weights, plastic(bool), plasticity_gate, transmission_gate,
  receptor, exc_receptor, coincidence_detector, graded, …}` (`bridge.py:2400-2475`); builds `cp_connections` directly
  via COO→CSR (`bridge.py:2491-2492`). Must run AFTER `_initialize_simulation_data()`. **This is the mechanism** —
  emit one group per trained weight matrix, with `plastic=False` to freeze it.
- **Region-framework path** — the trained net's layers declare as `BrainRegion`s + `RegionPathway`s; `RegionManager.build_wiring_plan`
  (`sim/regions.py:597`) yields the same dict, and the bridge auto-calls `inject_explicit_wiring` on it (`bridge.py:1706-1718`).
  The generator becomes a set of disjoint index slices co-resident with the conversational/nav brain — the *exact*
  one-brain pattern roadmap-step-2 used.
- **Post-build weight write** — `set_pathway_weights(name, pre, post, weights, add_missing=…)` (`bridge.py:2986`)
  overwrites/append edges (the Gabor-V1 pre-init precedent). `add_missing=True` rebuilds the CSR (re-sorts data, stales
  gate-index maps — the CLAUDE.md nav-merge caveat applies; handle by index).
- **Persistence of the installed weights is reliable** — `save_checkpoint` persists `cp_connections` data/indices/indptr
  (`bridge.py:7602-7610`) + engram tags (`bridge.py:7716-7733`); `BridgeLineage` (`sim/lineage.py:190`) gives atomic
  `current.simstate.h5` + history + growth-log + per-pathway shard export (`lineage.py:392`). **This is the C2
  persistence layer** — the consolidated generator lives across sessions.

### 4.2 The genuine HARD PART — there is NO LIF model on the bridge

**`NeuronModel` (`sim/enums.py:7-15`) = {IZHIKEVICH, HODGKIN_HUXLEY, ADEX, RESONATE_AND_FIRE}. There is no
LEAKY_INTEGRATE_AND_FIRE.** The BPTT generator's layers are plain LIF (leak 0.95, threshold 1.0, hard reset). So:

- **What transfers cleanly:** the *weights* (as static synapses). The bridge will run them — but driving Izhikevich/AdEx/RF
  post-synaptic dynamics, **not** the LIF dynamics they were trained under. This is a real dynamics mismatch, not a
  config detail. A net trained at LIF threshold 1.0 / leak 0.95 will, naïvely installed, produce a different spike
  pattern → degraded (or broken) generation.
- **Three options to close it (ranked cheap-first):**
  1. **AdEx-as-LIF** (cheapest): AdEx with adaptation `a=b=0` and tuned `tau_m`/`v_thresh`/`v_reset` reduces to a leaky
     IF. Install the weights between AdEx populations configured to match the training-time leak/threshold. **Reuse-only,
     no `sim/` edit** (AdEx params are per-region, `regions.py:89`). The residual is whether AdEx-zero-adaptation tracks
     the trained LIF closely enough — measure in the §6 probe.
  2. **ANN→SNN conversion calibration** (the SOTA path): rather than 1:1 weight copy, run the 2025 threshold-balancing /
     activation-matching conversion so the spiking net's *firing rates* reproduce the trained activations on a held-out
     calibration set (the LAS/MBE recipe; ~10% ppl cost). This is the principled way to absorb the dynamics gap; it is
     *calibration*, not retraining, and runs locally.
  3. **Post-conversion surrogate-grad finetune ON the bridge dynamics** (most robust, a `sim/` edit): a few epochs of
     surrogate-grad BPTT with the bridge's actual neuron model in the forward pass, so the weights adapt to
     Izhikevich/AdEx/RF. This is the cleanest fidelity fix and the place a small additive guarded `sim/` edit (a LIF or
     "AdEx-LIF" forward consistent with `bptt_snn_gpu`) is justified — owner-reviewed, default-off, byte-identical when
     unused, mirroring the transmission-gate / dendritic-gain precedents.
- **Isolation of the generator slice:** the bridge's core `_run_one_simulation_step` (Izhikevich/HH/AdEx) has **no
  neuron-mask** to step a sub-population in isolation (the masked-step path is RF-specific: `rf_kick(neuron_mask=…)`
  `bridge.py:5646`, `_rf_advance_one` `bridge.py:5710`, megakernel mask `bridge.py:5788`). So a *non-RF* generative
  slice isolates the way nav+conv already do — **a disjoint index range with zero cross-synapses to the rest of the
  brain** (then wire deliberate read/write routes). This is proven (roadmap step 2 co-residence). **An RF/phasor
  generator** *could* use the masked-step path — relevant to Option B below.

**C1 verdict:** *feasible, with one named hard part (the LIF↔Izhikevich/AdEx/RF dynamics gap).* The cheap path
(AdEx-as-LIF + conversion calibration, no `sim/` edit) is the first thing to de-risk; the robust path (a small guarded
LIF/AdEx-LIF forward for post-conversion finetune) is the fallback if calibration alone loses too much.

---

## 5. C2 — continue learning + grow, WITHOUT catastrophic forgetting (the deepest risk)

### 5.1 What the project already has (verified)

- **Hippocampal consolidation, awake/sleep-gated** — `set_awake_gates`/`set_sleep_gates`/`freeze_all_gates`
  (`text_minimal_isolation.py:1664/1704/1744`): awake = encoding on; sleep = encoding off, `ca3_swr_burst=1`,
  `ca1_to_motor=1`, `ca1_to_lang_out=1`. `run_swr_replay_phase` (`consolidation_trainer.py:154`) bursts random ~15%-sparse
  CA3 subsets at SWR rate; `run_concept_replay_phase` (`consolidation_trainer.py:43`) drives engram-tagged CA3 ensembles
  selectively → STDP at `ca3→ca1→cortex` transfers the pattern (McClelland CLS). **Validated:** Phase 1.3 consolidation
  3/3 strict anti-cheat; Phase 1.4 BRANCH A no-forgetting **5/6 retention ≥80%, mean 103%** (master plan 2026-05-07);
  the moat held throughout.
- **Engram tagging** (`bridge.py` start/commit/stimulate, persisted) — tag a new concept's ensemble, replay it during
  sleep → consolidate without a gradient. **This is the C2 "grow" primitive.**
- **Plasticity gates / transmission gates / neuromodulators** — freeze the pretrained generator's weights
  (`cp_plasticity_rate_gain=0`, the masked-clip paths at `bridge.py:6673/6990/7253` now keep frozen weights byte-exact),
  while a *new* plastic layer learns on top. The selection/context-gating + metaplasticity primitives the SNN-CL
  literature favors are all present.

### 5.2 What the 2025 literature says (fresh pass) — the project's approach IS the field's

- **Latent-replay + a NREM-analogous SLEEP phase + noise** (arXiv 2507.02901, 2025): "a sleep phase where the network
  rehearses latent representations without external input … analogous to NREM sleep-based memory consolidation," + noise
  injection for robustness — *exactly* the project's SWR/concept replay + sleep gates + OU noise. The field independently
  arrived at the project's mechanism.
- **Replay/metaplasticity ≫ EWC at low cost**: NACA (Science Advances 2023, PMC10456855) — neuromodulation-assisted
  metaplasticity — beats EWC by up to ~50 pts at **~1% the FLOPs** (continuous MNIST: NACA 60% vs EWC 10%). The project
  HAS a neuromodulator subsystem, so this is a second, independent biology-grounded CL lever if replay alone is
  insufficient. Context-gating (arXiv 2406.01883) is a third.
- **The honest risk:** these CL results are validated on *classification* task-sequences, not on *generative* sequence
  models. **Whether SWR consolidation preserves a learned GENERATIVE cortex while it acquires new facts is UNTESTED** —
  it is the genuine open question of this frontier. The de-risk: treat the consolidated generator's weights as the
  "remote/cortical" store (frozen, plus slow replay-driven updates) and route NEW learning through the hippocampal
  fast-store + a thin plastic adapter, then run the catastrophic-forgetting battery (§6 step 4) — the same protocol that
  validated Phase 1.4, now with "generation quality on the original corpus" as the retention metric.

### 5.3 C2 verdict

*The deepest risk, but the best-supported.* The project's validated SWR consolidation + no-forgetting is the same
mechanism the 2025 SNN-CL literature converges on, and there are two independent biological fallbacks (metaplasticity/NACA,
context-gating) that beat EWC cheaply. **The one untested claim is "consolidation preserves a generative cortex while it
keeps learning,"** and §6 step 4 measures it directly. An honest negative here (generation degrades as new facts are
added) is a *publishable biology-translatable finding* about CLS in generative substrates — i.e. a real deliverable
either way.

---

## 6. The RANKED, cheapest-first BUILD PLAN — structured as the owner's LOOP

> **Loop:** train → test GENERATION (coherent novel text) → test GROWTH (learns new things) → CONFIRM NO CATASTROPHIC
> FORGETTING. Each step has scale, cost, what it proves, a GO/NO-GO, and the explicit cloud point. Cheapest-first; every
> step is a decisive gate before the next.

| # | Step | Scale / cost | What it PROVES | GO / NO-GO |
|---|---|---|---|---|
| **0** | **Cheap feasibility PROBE (§6.1)** — load the SHIPPED Phase-2.2 SNN `.npz` slice onto the bridge and check it SPIKES + a no-train generation sanity on the existing Transformer (Gen-F) | **hours**, CPU/1×3090, NO training | (a) the C1 install path works end-to-end on a real artifact; (b) the LIF↔bridge-neuron gap is *measured* (does an AdEx-as-LIF slice reproduce the trained net's spikes?); (c) re-confirms Gen-F generates coherent novel text | **GO** if the slice installs + spikes (even degraded) AND Gen-F still generates novel coherent text → proceed. **NO-GO** if the install path is broken → fix tooling first (cheap). |
| **1 (train)** | **Pretrain the COMPETENT generator** — (a) the **non-spiking Transformer baseline** (~6M, Gen-F recipe, the SAFE path) AND (b) scope/start the **spiking generator** (10–46M, spiking-RWKV/SpikeGPT-style, all-positions autoregressive loss — fixing the `cortex_pretraining.py:259` last-position limitation) | (a) **hours**, 3090; (b) **≤30 h**, 3090 — *the cloud trigger lives here* | a competent generator exists, with a REAL held-out generation gate (abs-ppl floor + word-shuffle + verbatim-copy controls, per Gen-F) | **GO** if held-out ppl < vocab AND coherent multi-sentence English beats the word-shuffle control (Gen-F cleared this). **NO-GO (spiking only)** if the spiking net plateaus at token-soup at the 3090 ceiling → **CLOUD decision point** (scale the spiking BPTT) OR fall back to the conversion path (step 3 on the non-spiking baseline). |
| **2 (generate)** | **Measure REAL novel generation** — the gap-MEASURED probe re-run on the trained generator: novel-composition score, distinct-generated/stored ratio > 1.0, held-out-triple production > 0 | **hours**, no train | the categorical "0 novel" wall is BROKEN — the generator produces sentences it was never told | **GO** if novel-composition > 0 AND coherent (not token-soup) AND the no-confab posture is preserved where the agent should abstain (the generator generates; the *retrieval/moat* layer still abstains on unknown facts). |
| **3 (consolidate, C1)** | **Convert + install onto the one bridge** — ANN→SNN conversion calibration (LAS/MBE-style) → `inject_explicit_wiring` as a disjoint slice co-resident with conv/nav; verify spiking generation on the bridge == off-bridge within the conversion tolerance | **hours–days**, 3090 (+ a small guarded LIF/AdEx-LIF `sim/` edit IFF calibration loses too much) | C1: the generator runs **as spikes on the ONE bridge**, co-resident, generating | **GO** if on-bridge generation matches off-bridge within ~10% ppl (the SOTA conversion tolerance) AND the conversational no-confab moat is byte-intact (regression: `test_nav_conv_step2b_coresident`, the moat asserts). **NO-GO** if the dynamics gap kills generation → escalate to the surrogate-grad-on-bridge finetune (the guarded edit). |
| **4 (grow + no-forget, C2)** | **The continual-learning loop + the catastrophic-forgetting battery** — freeze the consolidated cortex; teach NEW content via hippocampal fast-store + SWR/concept replay consolidation; re-measure generation on the ORIGINAL corpus (retention) + on the NEW content (acquisition) | **days**, 3090 — the project's home turf | C2: the agent **keeps learning + growing** post-training, and the original generative capability is **NOT catastrophically forgotten** (the Phase-1.4 protocol, retention metric = generation quality) | **GO** if new content is acquired AND original-corpus generation retention ≥80% (the Phase-1.4 BRANCH-A bar) AND the moat holds. **HONEST NEGATIVE** (generation degrades as facts are added) is itself the deliverable — a CLS-in-generative-substrates finding. |

**The decision the plan encodes:** *Step 0 now* (hours, settles whether the whole frontier is even tooling-feasible).
*Step 1a (non-spiking baseline) is the safe spine* — it is already proven (Gen-F) and de-risks steps 2–4 independently
of the spiking-generator risk. *Step 1b (spiking generator) is the biology-faithful target and the ONLY cloud trigger.*
*Steps 3–4 are where C1/C2 — the owner's actual constraints — are decided, on the project's strongest machinery.*

**A note on the biology-faithful vs safe spine:** there are two honest routings, and the owner should pick (it's a
genuine fork, surfaced not chosen):
- **Spine A (safe, faster to the loop):** pretrain the NON-spiking Transformer (Gen-F, proven) → ANN→SNN convert →
  consolidate. This demonstrates the FULL loop soonest and is the SOTA recipe; the generator is non-spiking *during
  pretraining* but fully spiking *on the bridge* after conversion (satisfies C1). The biology claim is the CLS one (§2).
- **Spine B (biology-faithful, higher-variance):** pretrain a SPIKING generator (SpikeGPT-style, surrogate-grad) →
  consolidate. Stays spiking throughout. Higher risk (the project's 25–50M from-scratch token-soup is a warning, though
  SpikeGPT-46M's success says it's a recipe problem, not a wall) and the cloud trigger lives here.
*Recommendation: run Spine A and Spine B's pretraining in PARALLEL after step 0 — A guarantees the loop completes and
de-risks 2–4; B chases the biology-faithful version with the cloud option held in reserve. They share steps 0, 2's
harness, 3's install path, and 4 entirely.*

### 6.1 The cheap feasibility PROBE (step 0) — design

**Goal:** the cheapest possible de-risk of the whole frontier — *can a trained slice load onto the bridge and spike,
and does the existing competent generator still generate?* — with NO big training run.

- **Probe A — install + spike (C1 tooling de-risk):** load the **already-shipped** Phase-2.2 SNN `.npz` (or train a
  60-second tiny one via `cortex_pretraining.train_shakespeare` at 1×64 hidden, the validated Gen-inc1 smoke) → build a
  small bridge → declare its layers as AdEx-as-LIF regions (`a=b=0`, tuned threshold/leak) → `inject_explicit_wiring`
  the weight matrices (`plastic=False`) → drive the input layer with a one-hot char and run steps → **assert the output
  layer SPIKES and the argmax-over-output-spikes is non-random** (matches the off-bridge `forward_unroll` prediction
  above chance). *Metric:* on-bridge vs off-bridge next-char agreement. *GO:* > chance and the slice spikes → the C1
  install path + the AdEx-as-LIF approximation are viable. *NO-GO:* slice silent or random → the dynamics gap needs the
  conversion-calibration or the guarded-finetune path *before* investing in a big generator (cheap to learn now).
- **Probe B — novel-generation sanity (the loop's "generate" gate, on the proven generator):** re-run the
  `generation_novelty_probe.py` harness against the **Gen-F Transformer** checkpoint (non-spiking, already trained):
  confirm distinct-generated/stored > 1.0, novel-composition > 0, coherent (beats word-shuffle), abs-ppl floor held.
  *GO:* Gen-F generates novel coherent text (it did, 2026-05-17) → the "generate" gate is real and the only open
  question is making it spiking + consolidated (steps 1b/3). This re-anchors the loop on a known-good generator so
  steps 2–4 are testing *consolidation*, not *whether a generator can be trained at all.*

**Anti-cheats (carry the harness battery):** abs-competence floor (ppl < vocab — the Gen-S false-PASS lesson);
word-shuffle + verbatim-copy controls (Gen-F's gate); the no-confab moat asserted intact at every step (the generator
generates; the retrieval layer still abstains); ≥6 seeds for any variable claim (`feedback_6seed_validation`); CuPy for
the decisive runs, numpy only for the tiny probe (`feedback_gpu_not_numpy`); **frozen gates pre-registered before
seeing held-out data.**

**Why this is the right cheap-first:** Probe A settles the single genuinely-novel C1 question (does the trained net
spike on the bridge through a real-neuron model) in *hours* on a *shipped* artifact, and its NO-GO routes directly to
the conversion/finetune fidelity work *before* any expensive pretraining — while Probe B re-confirms the loop's
"generate" gate on a generator already proven to clear it. Together they convert "is the whole frontier feasible?" from
a months-scale bet into an afternoon's measurement, exactly per the cheapest-first gate discipline.

---

## 7. Trust-but-verify (load-bearing claims; verified vs flagged)

**Verified directly this pass (file:line / finding read in full):**
- The generation gap is MEASURED (0 novel, ratio 1.0, 16/16 abstained) — `2026-06-22-generation-novelty-categorical-gap-MEASURED.md`, read in full.
- The parent scoping names §5.4 (learned generative-sequence substrate) as "the ONLY path to free generation, NOT the
  dendrite" — `2026-06-22-conversational-scaling-vs-dendritic-scoping.md` §3.5/§5, read in full.
- Generative groundwork exists on `main`: `sim/bptt_snn.py` (LIF + ATan + BPTT), `bptt_snn_gpu.py`, `surrogate_grad.py`,
  `char_tokenizer.py`, `bpe_tokenizer.py`, `cortex_pretraining.py` (`.npz` artifact `save_checkpoint:328`; **last-position
  loss `:259`** — the autoregressive-gap), all read.
- **The C1 facts:** `inject_explicit_wiring` (`bridge.py:2393`, plan format `:2400-2475`), `set_pathway_weights`
  (`:2986`), `RegionManager.build_wiring_plan` (`regions.py:597`), region-framework auto-wire (`bridge.py:1706-1718`),
  `save_checkpoint`/`cp_connections` (`bridge.py:7602-7610`), `BridgeLineage.save`/`export_shards` (`lineage.py:190/392`)
  — all verified via the read-only sub-investigation, anchors confirmed against the file.
- **The hard part:** `NeuronModel` has NO LIF (`enums.py:7-15`) — the weights transfer, the LIF dynamics do not.
  Masked-step isolation is RF-specific (`bridge.py:5646/5710/5788`); non-RF slices isolate by disjoint index range.
- **The C2 machinery:** awake/sleep gates (`text_minimal_isolation.py:1664/1704/1744`), `run_swr_replay_phase`
  (`consolidation_trainer.py:154`), `run_concept_replay_phase` (`:43`); Phase 1.3/1.4 validation (master plan, read).
- **C3 pre-authorization** — `MASTER-PLAN-main-then-pathF.md:14-27` (backprop = developmental pretraining; biology-grounded
  continual learning after), read in full.
- **The prior negatives are about the GENERATOR, not consolidation** — Gen-S/D/inc3/P/G1 + 25–50M ceiling + Gen-F PASS,
  synthesized via the read-only sub-investigation, each verdict/number quoted from its finding.
- **SOTA** (fresh June-2026 web pass, abstracts read): SpikeGPT (216M → WikiText-2 ppl 18–19, surpasses GPT-2-small);
  ANN→SNN conversion near-lossless (GPT-2-small ~10% ppl, LAS arXiv 2505.09659, MBE arXiv 2508.07710); SpikingBrain-7B
  (ANN→SNN + 150B tokens); TinyStories (<10M params → coherent multi-paragraph English, ≤30 h on a V100/3090-class GPU);
  SNN-CL = latent-replay + sleep + noise (arXiv 2507.02901); NACA metaplasticity ≫ EWC at ~1% FLOPs (PMC10456855).

**Could NOT fully verify (flagged honestly):**
1. **[VERIFY — most load-bearing]** That *no* prior spiking-generator attempt reached a bridge-consolidation step (the
   reframe rests on this). The sub-investigation found each died at train/distill generalization; the controller should
   spot-confirm by reading the Gen-S/D/25–50M finding *conclusions* — if any DID attempt a bridge install, the "untested
   consolidation" claim weakens.
2. **[VERIFY — feasibility]** That AdEx-with-zero-adaptation tracks a trained LIF closely enough to preserve generation
   (the cheap C1 path). This is *exactly* what step-0 Probe A measures — it is a hypothesis, not a verified result.
3. **[VERIFY — the open scientific risk]** That SWR consolidation preserves a *generative* cortex while it keeps learning
   (C2). Validated for *classification/fact-binding* (Phase 1.4), UNTESTED for generation. §6 step 4 measures it; honest
   negative is the deliverable.
4. **[VERIFY — scope]** Whether the owner's "demonstrate the loop" target accepts **Spine A** (non-spiking pretraining →
   ANN→SNN convert → spiking-on-bridge) as satisfying C1+C3, or requires **Spine B** (spiking throughout). The recommend
   is to run both in parallel, but this is a genuine fork the owner should confirm (it sets the cloud-budget exposure).
5. **The exact SpikeGPT-vs-project recipe difference** (RWKV recurrence + corpus vs stacked-LIF + last-position +
   TinyStories) is inferred from abstracts + the `cortex_pretraining.py:259` code, not from a re-run — it is the strongest
   reason to believe a 10–46M spiking generator *can* generate (vs the project's token-soup), but it is a hypothesis.

---

## Sources

### Project record (re-verified this pass, file:line cited)
- `research/findings/2026-06-22-generation-novelty-categorical-gap-MEASURED.md` (the 0-novel gap this closes).
- `research/findings/2026-06-22-conversational-scaling-vs-dendritic-scoping.md` (§5.4 names the generative-sequence substrate as the unlock).
- `research/findings/2026-06-17-conversational-architecture-to-basic-LLM-scoping.md` (the "free generation = hard wall, Option 4" framing this frontier now attacks directly).
- `sim/bptt_snn.py`, `sim/bptt_snn_gpu.py`, `sim/surrogate_grad.py`, `sim/char_tokenizer.py`, `sim/bpe_tokenizer.py`, `research/runners/cortex_pretraining.py` (the generative groundwork).
- `sim/bridge.py` (`inject_explicit_wiring:2393`, `set_pathway_weights:2986`, RF masked ops `:5646/5710/5788`, `save_checkpoint:7573`, `cp_connections:7602`, engram `:7716`), `sim/regions.py` (`build_wiring_plan:597`, `BrainRegion:32`, `RegionPathway:251`), `sim/enums.py` (`NeuronModel:7-15` — no LIF), `sim/lineage.py` (`save:190`, `export_shards:392`).
- `research/runners/consolidation_trainer.py` (`run_swr_replay_phase:154`, `run_concept_replay_phase:43`), `research/runners/text_minimal_isolation.py` (gate fns `:1664/1704/1744`).
- `docs/plans/2026-05-06-MASTER-PLAN-main-then-pathF.md` (C3 pre-authorization; Phase 2 dead-end-as-TRANSFER; Phase 1.3/1.4 validation).
- `research/findings/2026-05-09-Phase-2.3b-50M-cosine-REFUTED.md` (the dead-end was TRANSFER, the 50M net trained fine).
- Generator arc: `2026-05-17-generator-{F-small-transformer-LM-PASS, E-ngram, D-distillation-NEGATIVE, S-subword-spiking-LM-NEGATIVE}.md`; `2026-05-16-generator-{P-predictive-coding, increment1-foundation, increment3-capacity-scan, G1-songbird}-*.md`; `2026-06-02-generative-ceiling-spiking-LM-NEGATIVE-overfit-not-size.md`; `2026-06-03-{pre-compute-review-the-tiny-LLM-gap-is-ALREADY-MEASURED, deep-research-how-the-field-gets-past-our-generative-conversation-wall}.md`.
- `sim/song_hvc.py` (HVC sequencer — sound, but the order-blind self-comprehension judge gives 0 gradient).
- Catalog: `sim-catalog/references/feature-catalog.md` G.02 (active dendrites MISSING), G.07/H.19 (medial-premotor sequence generation), N.15 (theta-gamma multiplexed buffer).

### Current literature (June 2026 pass)
- **SpikeGPT** — Zhu et al., arXiv 2302.13939 (46M/216M generative SNN; 216M → WikiText-2 ppl 18–19, surpasses GPT-2-small; surrogate-grad BPTT + RWKV block).
- **TinyStories** — Eldan & Li, arXiv 2305.07759 (<10M params → fluent multi-paragraph coherent English; 1–35M trains ≤30 h on a V100/3090-class GPU). The minimum-viable-scale anchor.
- **ANN→SNN conversion (C1 path)** — LAS arXiv 2505.09659 (loss-less spike-driven LLM conversion); training-free MBE arXiv 2508.07710; GPT-2 conversion benchmarks (~5–12% cosine, ~10% ppl).
- **SpikingBrain-7B** — arXiv 2509.05276 (ANN→SNN + ~150B-token continual pretraining; Transformer-comparable).
- **SpikeLLM** — arXiv 2407.04752 (saliency-based spiking quantization of a pretrained LLAMA).
- **SNN continual learning (C2 path)** — latent-replay + NREM-analogous sleep + noise: arXiv 2507.02901 (2025); NACA metaplasticity ≫ EWC at ~1% FLOPs: PMC10456855 / Science Advances 2023; context-gating arXiv 2406.01883.
