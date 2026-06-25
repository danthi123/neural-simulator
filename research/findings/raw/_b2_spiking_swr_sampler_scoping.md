# Scoping — make the b2 generative-replay SAMPLING a SPIKING mechanism (the generative ACT is the brain's), 2026-06-24

**Read-only deep-research scoping. NO edits / runs / webapp.** Per the standing deep-research-first practice
(a new mechanism *class*: a spiking generative SAMPLER) and the SURPASS sharpening (the prior comfortable
verdict is the START of the research, not the end). The trigger is the biology-fidelity-audit class-(b) #2
residual: in the b2 generative-replay proposer the LIKELIHOOD/GROUNDING is the brain's (the learned PPMI
co-occurrence cortex; lesion/shuffle-proven load-bearing), but the generative ACT — drawing one role-filler
from that distribution — is a host `numpy.random.choice`. The full-capacity close-out is to make the SAMPLING
itself a spiking mechanism so the generative act IS the brain's.

**Headline up front: a first cheap-first attempt at exactly this already EXISTS and returned HONEST_NEGATIVE
(`research/runners/_followon1_spiking_generative_sampler.py`, raw `_followon1_spiking_generative_sampler.json`,
3-seed: spiking plausible-frac 0.027 ≈ the random floor 0.021, advantage 1.3× vs the host's 17×, ~3 novel
props vs the host's ~25, spiking/host quality 0.074). So the gate fires on a CONFIRMED-BOUNDARY (audit
condition (a)) + a KNOWN-FAMILY wall (rate-code/pattern-completion, condition (b)). This doc's job is the
SURPASS round: ISOLATE the exact residual, REFRAME via how the brain actually samples, RANK cheap-first
mechanisms that DIDN'T get tried, and give the verdict — surpassable-and-how-cheaply vs needs-a-deeper-
substrate.** Files cross-checked at file:line where load-bearing.

---

## MOVE 1 — DIAGNOSIS: what the host sampler does + ISOLATE the exact residual bytes

### 1.1 What the host sampler does (the genuine residual, pinned to the line)
The proposer is `GenerativeReplayProposer` in `research/runners/_genfrontier_b2_generative_replay_derisk.py`.
One generative sample (`propose`, lines 211–239) is a host loop that, per attempt:
1. picks a seed agent uniformly — `self.agents[int(self.rng.integers(...))]` (line 219);
2. `_weight_partner((a,), self.actions)` (lines 191–202) — reads the PPMI matrix `P` and sums
   `P[row[x], row[c]]` over the partial triple's fillers → a weight vector `w` over candidate actions;
3. `_sample_weighted(self.actions, w)` (lines 204–209) — normalizes `p = w/w.sum()` and draws
   **`self.rng.choice(len(candidates), p=p)`**;
4. repeats step 2–3 for the patient, cued by `{agent, action}`.

The accept gates (`_plausible` lines 167–174, `_contradicts` line 189, novelty line 223) read the brain's
PPMI graph + the RF composer's no-confab moat (`ask_yes_no`). Those are the brain's and are lesion/shuffle/
moat-proven (the b2 GO: shuffled-graph collapses 0.328→0.018; 0 moat leaks; 6 seeds).

**ISOLATED residual = exactly TWO host primitives, nothing else:**
- **(R1) the weight read** `_weight_partner` — a host matvec `P @ onehot(partial)` over the PPMI matrix.
  This is host arithmetic but it reads the brain-derived signal directly; it is the "matched-filter against
  the likelihood" and is the *lighter* part of the residual.
- **(R2) the DRAW** `_sample_weighted` → `np.random.choice(p=softmax-free-normalized weights)`. **This is THE
  generative act** — the stochastic selection of one filler from the likelihood. It is the single byte the
  owner's residual statement names ("the generative ACT itself"). Everything load-bearing around it (the PPMI
  likelihood `P`, the fact store, the moat) is already the brain's.

So the genuine residual is NOT "the whole proposer" — it is one normalize-and-draw (R2), plus the host matvec
that feeds it (R1). The b2 finding itself names the close-out: "A fully-spiking generative-replay sampler
(resampling on the substrate via SWR-gated CA3 + the engram/replay machinery the project already has) is the
natural follow-on" (finding §Scope).

### 1.2 What the prior spiking attempt ACTUALLY did, and WHY it failed (the precise root cause)
`_followon1_spiking_generative_sampler.py` (`SpikingSWRCa3Sampler`) replaced R1+R2 with a single-pool
spiking **CA3 pattern-completion** mechanism (reusing `_D_sparse_heteroassoc`):
1. each word = a sparse K-of-N pattern in ONE shared CA3 pool with a plastic excitatory recurrent;
2. it ENCODES the brain's PPMI graph by Hebbian co-firing EVERY PPMI-related (agent,action) AND (action,patient)
   pair into that ONE recurrent (`_encode_graph`, lines 163–183);
3. RESAMPLE: drive a seed agent → the recurrent pattern-completes → read the firing profile → **temperature-
   softmax over the spiking firing profile + `rng.choice`** (`_sample_from_profile`, lines 212–223) → action;
   repeat cued by (agent, action) → patient.

It returned HONEST_NEGATIVE. The data localize the failure precisely — the lesion + moat HELD (the design's
controls work), but the SIGNAL collapsed: spiking plausible-frac **0.027 ≈ random floor 0.021**, i.e. the
completed firing profile carries essentially NO selectional structure. Three concrete, diagnosable causes
(none is "the substrate can't sample" — they are mechanism-construction faults):

- **(F1) ONE pool conflated two distinct relations.** Agent→action and action→patient associations were
  Hebbian-encoded into the SAME recurrent. So when you cue an action to complete a patient, the recurrent
  ALSO pulls back agents (which co-fired that action) and other actions — the completion is a blurred
  superposition, not a clean action→patient read. The role structure was destroyed at encode time.
- **(F2) Pattern-completion ≠ a likelihood sampler (the wrong mechanism).** A Hopfield/Treves-Rolls
  autoassociator *converges to the single nearest stored attractor* — it is a denoiser, not a sampler over a
  graded distribution. b2's likelihood is broad and multi-modal (many actions are weakly PPMI-related to an
  agent); a completion network either locks onto ONE basin or produces mush, neither of which reproduces the
  graded `p ∝ PPMI` draw. This is the SAME family as the documented `_D_sparse_heteroassoc` cue-recall arc and
  the rate-code/whitening wall — the substrate is being asked to read a graded multi-modal distribution off a
  point-neuron firing-rate vector.
- **(F3) The stochasticity was STILL host.** Even after the spiking completion, the actual DRAW was
  `_sample_from_profile` → `rng.choice` over the firing profile (line 223). So the attempt moved the
  *likelihood read* onto the substrate (badly, F1+F2) but **left the generative ACT (R2) as a host
  `rng.choice` anyway** — it did not actually close the residual it set out to close; it only relocated R1.

**The decisive reframe this sets up:** the prior attempt tried to make the likelihood EMERGE from attractor
completion (and failed F1/F2) while leaving the draw host (F3). The correct target is the inverse — keep the
brain's *already-validated* likelihood signal, and make the **DRAW** (R2) the spiking event. That is what the
neural-sampling biology actually says (Move 2), and it reuses machinery the prior attempt ignored (Move 3).

---

## MOVE 2 — REFRAME via real biology: how does the brain actually GENERATE a sample?

Two complementary, well-established biology strands — and the prior attempt used NEITHER cleanly.

### 2.1 The neural-SAMPLING hypothesis (the load-bearing reframe)
**Buesing, Bill, Nessler & Maass 2011, "Neural Dynamics as Sampling"** (PLoS Comput Biol): networks of
stochastically-spiking neurons sample from a probability distribution — *the spike pattern at a moment IS a
sample from the network's stationary distribution* (a Boltzmann/Gibbs distribution for symmetric weights).
Critically: **"clamping" a subset of neurons makes the rest sample from the CONDITIONAL distribution given the
observed values.** This is EXACTLY b2's structure: clamp the seed agent (the SWR cue) → the action assembly
samples `p(action | agent)`; clamp (agent, action) → the patient assembly samples `p(patient | agent, action)`.
The draw is then NOT a host `rng.choice` — it is *which assembly wins the noisy spiking competition this
event*. The stochasticity is the substrate's intrinsic membrane/synaptic noise (the OU background the bridge
already has, `ou_std_current_pA`), not `numpy`.

Corollary (Buesing/Nessler/Maass and the WTA-sampling line, e.g. Nessler-Pfeiffer-Buesing-Maass): a **soft
winner-take-all** network of spiking neurons, where each unit's drive is `log p(candidate)` (here the PPMI
relatedness), fires the winner with probability proportional to `softmax(drive/T)` — i.e. a WTA over
log-likelihood-driven units IS a categorical sampler. The temperature is the noise/inhibition level. **This is
the canonical biology for "a spiking network draws a sample," and the project already owns soft-WTA spiking
machinery** (Move 3).

### 2.2 SWR-gated generative replay (the *when/what* gate — already named, partly built)
- **D.19 / N.07 / N.16 / N.17** (catalog): the CA3 recurrent intrinsically generates sharp-wave-ripple bursts
  (N.16: self-organized, not EC-driven; N.17: awake replay at choice points = online deliberation). A ripple
  is the *event window* in which a reactivation/recombination is emitted. The catalog repeatedly flags **"replay
  CONTENT is the named bottleneck, not replay quantity"** (D.19, N.07, N.50 sim-status) — precisely the
  `_followon1` failure (it produced ripples, but content at chance).
- **G.09 imagination / future simulation as constructive memory** (catalog, currently "missing"; prereqs D.01/
  D.02 HC microcircuit + N.* replay): the DMN recombines stored elements into never-experienced configurations.
  "Replay assembles elements into compounds, each a hypothesis about a possible configuration." This is the
  exact b2 framing and the exact owner residual.
- **Stoianov-Maisto-Pezzulo 2022** (hippocampus as a hierarchical generative model; generative replay resamples
  FICTIVE sequences) + **Barry/Love 2023** (Nat Hum Behav, generative memory construction): replay SAMPLES from
  the learned generative model's likelihood — confirming the *draw-from-a-likelihood* framing, not pattern-
  completion-to-one-attractor.
- **Current SNN literature (web cross-check, June 2026):** "Coherent noise enables probabilistic sequence
  replay in spiking neuronal networks" (arXiv 2206.10538) and "Leakage and Second-Order Dynamics Improve
  Hippocampal RNN Replay" (arXiv 2602.18401) — **noise drives the stochastic/probabilistic selection in
  spiking replay**; metastable-attractor replay models (PMC9822116) use noise + short-term depression to hop
  between assemblies. All three say the SAMPLE comes from noise-driven spiking dynamics, not a host RNG.

### 2.3 The synthesis (the testable hypothesis)
The brain's generative sample = **a noise-driven soft-WTA competition among candidate-filler assemblies, each
driven by its learned likelihood (the PPMI relatedness to the clamped seed), within an SWR event window.** The
winner each event IS the draw; intrinsic spiking noise IS the stochasticity. The likelihood stays the brain's
PPMI cortex (already validated); the GENERATIVE ACT becomes the spiking WTA's stochastic winner. This is
mechanism-correct (Buesing-Maass sampling + Buzsáki SWR + Stoianov-Pezzulo generative replay) AND it reuses
proven project machinery — unlike the `_followon1` pattern-completion approach, which is the wrong primitive.

---

## MOVE 3 — RANK cheap-first SPIKING-sampler mechanisms (the ones NOT yet tried)

Design constraint (the SURPASS lesson): **keep the brain's validated PPMI likelihood; make the DRAW (R2) the
spiking event.** Do NOT try to re-learn the likelihood from attractor completion (that is F1/F2, already
NEGATIVE). All options are reuse-by-import, NO `sim/` edit, CPU (`SIM_BACKEND=numpy`).

### Option A (RANK 1 — cheapest, highest-probability) — Noise-driven spiking soft-WTA sampler over the likelihood
**What.** Per generative event: build a small spiking **soft-WTA** assembly with one pool per candidate filler
(actions for stage 1; patients for stage 2). Drive each pool with a current proportional to its **log-PPMI
relatedness** to the clamped seed (the brain's likelihood, read as drive — biologically a learned cortico-
cortical projection strength). Add the bridge's intrinsic OU noise. Run a short window; the pool that wins the
noisy competition (first/most to cross a commit threshold) IS the sampled filler. Repeat for the patient cued
by (agent + winning action). The draw is `which assembly the noisy spiking competition selected` — **no
`rng.choice`.**
- **Reuse (the machinery `_followon1` ignored):** the project's spiking WTA / biased-competition is built and
  validated three ways — the **NEF thresholded WTA cleanup** in `one_brain_composer.py` (Izhikevich WTA ==
  argmax, the shipped cleanup-selection), `biased_competition_buffer.py` (biased-competition referent
  selection), `content_selection_spiking.py` (the dlPFC spiking content-selection). Any of these is a
  drop-in "pick-one-from-graded-drives" spiking primitive; adding OU noise + reading the stochastic winner
  across repeats converts argmax→sample.
- **Why it beats the prior attempt:** it does NOT conflate relations in one recurrent (each event clamps the
  seed and competes only the candidate role-pool — F1 gone); it is a sampler by construction (noisy WTA over
  log-likelihood = categorical sampling, Buesing-Maass — F2 gone, it's the RIGHT primitive); and the draw is
  the spiking winner, not host `rng.choice` (F3 gone — this actually closes R2).
- **Cost / risk.** Cheapest (small per-event WTA pool, ~the cleanup cost; the b2 universe is ~45 triples,
  ~200–500 events suffice). Risk: tuning the noise/temperature so the winner distribution matches `softmax(log
  PPMI / T)` rather than collapsing to argmax (too little noise) or going uniform (too much) — but this is a
  one-knob calibration with a direct GO check (the realized winner histogram vs the target softmax), and the
  WTA primitives are already validated. **Recommended de-risk target.**

### Option B (RANK 2 — strongest "the distribution itself is spiking") — Shipped spiking-softmax → spiking categorical draw
**What.** The normalize-and-draw (R2) is literally `sample ~ softmax(weights/T)`. The project ALREADY has a
**validated, shipped spiking softmax on the bridge**: `_genseq_spiking_softmax_derisk.py` computes
`softmax = spiking_exp (calibrated rectified-basis read on a live Izhikevich pool) + the shipped divisive-norm
circuit (`enable_input_divisive_norm`, bridge.py:6190) for the sum-normalization` — full-block fidelity GO
(corr ~1.0; the predicted exp rate-code wall did NOT bite because max-subtract bounds the exponential). Feed
the PPMI weight vector through this spiking softmax to get a spiking-computed probability vector, then realize
the categorical DRAW as a spiking event (a noisy WTA over that vector — Option A's selector, or a spiking
"first-to-threshold under noise" read).
- **Reuse:** `_genseq_spiking_softmax_derisk` (the exp+divnorm softmax) + Option A's WTA for the draw.
- **Why.** Makes BOTH the distribution-shaping (softmax) AND the draw spiking — the most complete close of R2
  AND part of R1's normalization. It directly answers "is the probability vector computed by neurons?" yes.
- **Cost / risk.** Slightly heavier than A (two spiking stages: softmax read + WTA draw). The exp read is
  already GO so the risk is low; the residual is the same WTA calibration as A. **Use if A's draw alone is
  judged not to make the *distribution* spiking enough** (A drives the WTA with log-PPMI directly, which is
  arguably already the distribution; B makes the softmax explicit + on-substrate).

### Option C (RANK 3 — most biology-faithful "SWR event" framing, but heaviest) — SWR-gated CA3 reactivation as the event window, with the Option-A sampler INSIDE the ripple
**What.** Keep the failed `_followon1`'s *event* idea (an SWR ripple gates the generative event) but replace
its *content* mechanism (pattern-completion) with Option A's noise-driven soft-WTA. The CA3 recurrent (N.16
intrinsic SWR) provides the ripple window + a seed reactivation; INSIDE the window the candidate-filler soft-
WTA (driven by likelihood + noise) selects the filler. This is the fullest catalog-faithful realization
(D.19 + N.16/N.17 + G.09).
- **Reuse:** `_D_sparse_heteroassoc` CA3 (for the ripple/seed only, NOT for the filler read), engram/replay
  infra, + Option A's WTA.
- **Why.** Most biology-complete; the ripple-gating is the literal "when," the WTA is the literal "what/draw."
- **Cost / risk.** Heaviest (two spiking subsystems + their interface). The `_followon1` build cost was ~82 s
  per CA3 build, ~1250 s total 3-seed; this is more. The ripple-gating adds biological completeness but NOT
  sampling correctness (that's Option A) — so it is the right *eventual* form but the WRONG first de-risk
  (it re-incurs the prior attempt's cost without first proving the sampler works in isolation). **Defer to a
  follow-on AFTER A/B prove the draw is surpassable.**

### Explicitly NOT recommended (the prior-attempt trap)
- **Pattern-completion-as-sampler** (the `_followon1` mechanism) — F1/F2 root cause; the wrong primitive.
  A denoiser-to-one-attractor cannot reproduce a graded multi-modal likelihood draw. Do not retry it.
- **Learning the likelihood graph INTO the recurrent** (F1) — conflates roles; also it duplicates the
  already-validated PPMI cortex for no benefit. Keep the PPMI likelihood as the brain's signal; sample from it.

---

## MOVE 4 — ANTI-CHEATS + the recommended cheap-first de-risk + GO bars + the verdict

### 4.1 The recommended cheap-first de-risk (Option A, CPU, ~hours)
Build a new probe (e.g. `_followon2_spiking_wta_sampler_derisk.py`), reuse-by-import, NO `sim/` edit. Reuse:
`_genfrontier_b2_generative_replay_derisk` (PPMI `build_plausibility`, `build_stored_facts`, the gates
`_plausible`/`_contradicts`, `random_recombination`, `shuffle_graph`), the RF composer / agent (store + moat),
and a spiking soft-WTA selector imported from `one_brain_composer` (NEF WTA) / `biased_competition_buffer` /
`content_selection_spiking`. Per generative event: clamp seed → drive the candidate role-pool WTA with
log-PPMI relatedness + OU noise → read the spiking winner as the draw (stage 1 action, stage 2 patient) →
apply the brain's UNCHANGED gates. Compare head-to-head against the HOST `rng.choice` sampler (the b2 GO) on
the SAME world, SAME gates, SAME seeds. Run 3 seeds first (mechanism check), promote to **6 seeds** before any
GO (`feedback_6seed_validation`; novel-comp + advantage are variable effects).

### 4.2 Anti-cheats (the b2/`_followon1` battery + the NEW provenance check the residual demands)
1. **Provenance: the SAMPLING is spiking, not host `rng.choice` (THE load-bearing new control).** Assert at
   runtime that NO `np.random.choice`/`rng.choice` is on the draw path — the winner is read from
   `bridge.cp_firing_states` of the WTA pools. A "spiking-source" assertion (the draw index == the argmax/
   first-to-threshold of the spiking pools, not a host categorical draw). This is the exact analogue of the
   project's standing "no Python value-copy" provenance checks (Route A/B). Without this the whole exercise is
   moot (it's the residual we're closing).
2. **Lesion (the WTA/likelihood is load-bearing).** Ablate the likelihood drive (drive all candidate pools
   equally) → the sampler must collapse to the uniform/random-recombination floor (the likelihood, read as
   spiking drive, is causally responsible). Ablate the noise (OU→0) → the draw must collapse to deterministic
   argmax (no longer a sampler — proves the noise IS the stochasticity, not a hidden host RNG). Both are the
   spiking analogue of b2's shuffled-graph + the `_followon1` CA3-lesion (which DID work — keep it).
3. **Shuffled-graph (structure, not template).** Shuffle the off-diagonal PPMI (b2's `shuffle_graph`) → the
   spiking sampler's TRUE-graph plausibility must collapse toward the random floor (the learned co-occurrence
   statistics, not the SVO template, drive the proposals). Carried verbatim from b2.
4. **No-confab moat (owner-sanctioned trade preserved).** 0 generated-proposition → known-fact leaks (a
   hypothesis NEVER passes `what_does`/`is_it_true`); 0 explicitly-negated facts re-proposed (the non-
   contradiction gate); the agent's standing untaught-cue abstention unregressed (the documented RF code-
   fidelity tail tolerance, NOT a sampler gate). Carried verbatim from b2/`_followon1` (both held it at 0).
5. **Random-recombination floor + host parity.** vs the uniform floor (advantage ratio) AND vs the HOST
   `rng.choice` sampler — the spiking sampler must MATCH the host's quality (the point is provenance, not a
   quality regression). Carried from `_followon1`.

### 4.3 GO bars (pre-registered; mirror the b2/`_followon1` bars so it's apples-to-apples)
GO iff, across all 6 seeds:
- **(provenance)** the draw is sourced from spiking firing (assertion 1 passes) — HARD gate, the whole point;
- **(novel)** ≥ `min_novel` (≥3) distinct novel-plausible triples generated, novel-comp > 0, disjoint from
  store, retrieval abstains on every one;
- **(plausible + parity)** spiking plausible-frac advantage ≥ 3× the random floor AND spiking/host quality ≥
  0.7 (the `_followon1` `host_match_frac` bar — the bar `_followon1` FAILED at 0.074; surpassing it to ≥0.7 is
  the precise quantitative target);
- **(lesion)** likelihood-ablation collapses to the floor AND noise-ablation collapses to deterministic argmax;
- **(shuffle)** shuffled-graph collapses TRUE plausibility to ≤0.5× the real;
- **(moat)** 0 leaks + 0 negated re-proposed.
HONEST_NEGATIVE otherwise, with the precise residual named (per move 1.2's F1/F2/F3 taxonomy: which one still
bites).

### 4.4 VERDICT — surpassable-and-how-cheaply vs needs-a-deeper-substrate
**SURPASSABLE, and cheaply — with HIGH confidence the right mechanism (Option A) has NOT yet been tried.** The
residual is precisely TWO host primitives (R1 weight-read, R2 the draw), and the genuine target is R2 (the
draw). The prior NEGATIVE (`_followon1`) is NOT evidence the substrate can't sample — it is evidence that
*pattern-completion is the wrong primitive* (F1 role-conflation + F2 denoiser-not-sampler) AND that it didn't
even close R2 (F3 the draw stayed host `rng.choice`). The neural-sampling biology (Buesing-Maass: a noisy
soft-WTA over log-likelihood-driven assemblies IS a categorical sampler; SWR/Stoianov-Pezzulo: replay samples
the likelihood) points at a DIFFERENT mechanism that (a) keeps the brain's already-validated PPMI likelihood,
(b) makes the DRAW the spiking WTA winner under intrinsic noise, and (c) reuses three ALREADY-VALIDATED spiking
primitives the prior attempt ignored — the NEF WTA cleanup (`one_brain_composer`), biased-competition
(`biased_competition_buffer`), and the shipped spiking-softmax (`_genseq_spiking_softmax_derisk`, GO). The
cheap-first de-risk is CPU, hours, reuse-by-import, NO `sim/` edit, and has a sharp GO check (the realized
spiking-winner histogram vs the target softmax + the provenance assertion).

**Confidence + honest caveat.** The mechanism is sound and the machinery exists, so the de-risk is cheap and
high-value; but the ONE genuine risk is the noise/temperature calibration making the spiking-WTA winner
distribution actually MATCH `softmax(log PPMI/T)` across the broad multi-modal b2 likelihood (the same family
as every graded-read-on-point-neurons concern). If that calibration cannot reach the host's 17×/quality-0.7
parity, the residual would be the WTA's ability to faithfully sample a *broad graded* (not peaked)
distribution — which would then be the precisely-isolated, characterized point-neuron boundary (an honest
negative IS the deliverable under BRAIN-BASED-ONLY), distinct from the prior attempt's wrong-primitive
failure. Either way the comfortable verdict is NOT accepted without this round. **Recommendation: run the
Option-A de-risk (3-seed → 6-seed) before accepting any boundary; it is the cheapest move that either closes
the last generative-act host residual or precisely characterizes why it can't be closed on this substrate.**

---

## Files / evidence
- Residual code: `research/runners/_genfrontier_b2_generative_replay_derisk.py` (`GenerativeReplayProposer`,
  `_weight_partner` L191, `_sample_weighted` L204 → `rng.choice` L209). b2 GO finding:
  `research/findings/2026-06-23-genfrontier-b2-generative-replay-derisk.md`; raw
  `research/findings/raw/_genfrontier_b2_generative_replay_derisk.json` (novel-comp 0.752, 17×, 0 leaks).
- The prior NEGATIVE attempt: `research/runners/_followon1_spiking_generative_sampler.py`
  (`SpikingSWRCa3Sampler`); raw `research/findings/raw/_followon1_spiking_generative_sampler.json`
  (HONEST_NEGATIVE: spiking-frac 0.027 ≈ floor, quality 0.074, lesion+moat held).
- Reusable spiking primitives (the un-tried machinery): `one_brain_composer.py` (NEF WTA == argmax),
  `biased_competition_buffer.py`, `content_selection_spiking.py`, `_genseq_spiking_softmax_derisk.py`
  (shipped spiking softmax: rectified-basis exp + `enable_input_divisive_norm`, bridge.py:6190, GO).
  CA3 (event-window only): `research/runners/_D_sparse_heteroassoc.py`.
- The 3E integration (where the proposer is wired into a turn):
  `research/runners/_burndown_3E_brain_owns_generation.py`.
- Audit (the trigger): `research/findings/raw/_biology_fidelity_audit_2026-06-24.md` (M15 + class-(b) #2).
- Prior generative-replay-loop NEGATIVE (the "replay content is the bottleneck" precedent):
  `research/findings/2026-05-24-c-generative-replay-decisive-NEGATIVE-...md`.
- Catalog: D.19 (SWR replay), N.07/N.16/N.17 (SWR / intrinsic ripple / awake replay), G.09 (constructive
  imagination, "missing"; prereqs D.01/D.02 + N.*), D.50 (DMN recombination, "missing")
  — `E:\Documents\Projects\sim-catalog\references\feature-catalog.md`. Kandel 6e Ch 52 (G.09 pp 1300–1302),
  Ch 54 (SWR pp 1365–1366).
- Literature: Buesing-Bill-Nessler-Maass 2011 "Neural Dynamics as Sampling" (PLoS Comput Biol — WTA/spiking =
  categorical sampler, clamping = conditional sampling); Stoianov-Maisto-Pezzulo 2022 (hippocampal generative
  model / generative replay); Barry/Love 2023 (Nat Hum Behav generative memory construction); "Coherent noise
  enables probabilistic sequence replay in spiking neuronal networks" (arXiv 2206.10538); "Leakage and
  Second-Order Dynamics Improve Hippocampal RNN Replay" (arXiv 2602.18401); metastable spiking replay
  (PMC9822116).
