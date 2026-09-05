# Changelog

All notable changes to the GPU-Accelerated Neural Network Simulator will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This is a research codebase; entries are organised chronologically rather than by release tag. The freshest dated section is the working tip.

## [Unreleased] — 2026-09-05 — One shared neural pool for core cognition; the fact-recall speed wall solved; more host shortcuts retired; own-voice fluency re-aimed

### One-brain substrate consolidation and a production speed fix

- **Surprise, world-model, self-monitoring and phrasing now share one literal neural pool by default (2026-09-05).** These four organs — previously co-resident but separate networks — were merged onto a single shared spiking substrate and verified safe before the flip; this is the "one-brain" step, not just organs sitting side by side. Finding: `2026-09-05-onebrain-single-pool-production-flip-derisk-GO.md`.
- **The meaning-composer's fact-recall speed wall is solved and shipped on by default (2026-09-05).** A hippocampus-style sharded lookup (DG-CA3-inspired sublinear spiking retrieval) replaces the prior full-scan recall, running roughly 400x faster with no loss of recall accuracy and no weakening of the no-confabulation guarantee. Finding: `2026-09-05-onebrain-fact-shard-dg-ca3-sublinear-spiking-retrieval-derisk-GO.md`.
- **Four more host shortcuts flipped on by default after lesion verification (2026-09-05):** the salience/attention signal, the value-driven action-choice circuit, body-state-driven affective appraisal, and the "halt a shaky thought" stop mechanism are now load-bearing, default-on spiking circuits in the live chat rather than host computation standing in for them.

### Own-voice fluency: two candidate mechanisms tested and banked, not shipped

- **The brain's own from-scratch spiking "mouth" remains the default generation path (flipped 2026-09-04)** — it beats a simple word-pair baseline on simple text at a deployable size, but is honestly not yet fluent enough on broad, arbitrary-topic text to retire the Qwen2.5-0.5B language-model scaffold, which remains the tracked #1 blocker.
- **Two candidate memory-write mechanisms for that fluency wall were built and tested this week, and both fell short of the bar.** An erase-before-write "delta-rule" memory write showed only a modest lift on the decisive wikitext-103 test, below the bar for shipping; a HiPPO/content-addressable-attention "hippokey" memory was a clear NO-GO. Both are banked (recorded, not deployed) rather than discarded, and a ranked next-step plan was produced via deep external research rather than another ad hoc lever.

### Same-cycle: full knowledge base defaulted on, and a measurement bug caught and fixed (2026-09-02)

- **The brain's full ~79,000-fact knowledge base is now the default it talks from**, replacing the earlier 15,000-fact core, with the prior blockers to that scale-up cleared.
- **An organ-merge safety-checker was built** to guard future one-brain consolidation steps like the 2026-09-05 pool merge above.
- **A widespread measurement bug (a stale reset between test reads) was found and fixed across 14 similar tests.** Correcting it flipped one on-by-default capability — curiosity steering working memory — from a passing verdict to failing, and it was switched off as a result; the earlier "pass" is superseded by this corrected read.
- **The first real cross-organ influence landed on the shared substrate** (surprise updates the world-model), wired into the live chat but off by default pending further verification.

### Honest state of the whole (do not read the wins as more than they are)

As of the production-integration ledger head (`docs/PRODUCTION_INTEGRATION_LEDGER.yaml`, 2026-09-05): 64 faculties are tracked in total, 29 run as genuinely load-bearing, lesion-verified spiking circuits in the default production chat turn, and only 1 is fully scaffold-retired onto the one-brain substrate — the ledger's own note states that most of the live chat's load-bearing cognition is still host, NumPy, or off-bridge.

A mechanical production-integration gate (`tools/gates/production_integration.py`) now cross-checks every ledger row's claimed state against live source-code anchors before a doc can claim more than the code does. Affect, self-model and metacognitive read-outs remain **functional read-outs only** ("my familiarity monitor reads this as novel, so I'm uncertain") — never a claim of feeling or phenomenal experience.

## [Unreleased] — 2026-08-19 — The brain's own signals now steer the live conversation; a production correctness fix; and an honest observe-vs-drive audit

### The frontier moved from "wire it in" to "make it load-bearing"

The integration arc's next question is stricter than "is the faculty wired and on by default?" — it is **does the brain's own internal state actually change what it says?** A neural verdict stashed as metadata beside an unchanged reply is a hollow checkbox, not integration. So the anti-hollow test used throughout this wave is: vary the internal signal, require the reply to differ, and require that difference to **vanish when the coupling is lesioned**. As before, affect and self-model read-outs are stated as **functional read-outs only** ("my decision-margin reads this as low-confidence") — never as feeling or phenomenal experience — and "GO" refers only to the exact six-seed test that passed, never a whole-faculty or consciousness claim.

### The brain's internal signals now drive the conversation (default-on, lesion-load-bearing)

- **Three internal signals now shape the actual reply (default-on, load-bearing).** The brain's own live spiking mood now colours **how** it phrases the answer (#84 affect→tone); its neural thought-swap decision now steers **which** topic the turn engages (#85 swap→topic); and its self-selected spiking dopamine mode now sets **how engaged** the reply is (#79 DA-mode→engagement). Each is verified by the anti-hollow test — vary the signal and the reply changes; lesion the coupling and the difference vanishes (ledger rows `swap-drives-response`, `da-mode-drives-response`, plus the affect drive; landed 2026-08-19).
- **The mood signal is now caused by a simulated body-state (6-seed GO).** The affect signal that colours replies is now driven by a body-state read through dedicated spiking **interoceptive** neurons wiring into the mood system; cutting those synapses makes the feeling stop tracking the body entirely. This remains a functional interoceptive read-out, not a claim of felt sensation. Finding: `2026-08-19-embodied-affect-interoception-GO.md`.

### Deliberation, metacognition, and self-initiated speech became more the brain's own

- **The re-entrant deliberation loop's cycle count now emerges from the substrate (2026-08-18).** How many rounds of re-entrant deliberation the workspace runs is now read from the substrate's own spiking ignition/conflict signal rather than a host-fixed counter — the first time a spiking conflict signal *controls* deliberation depth. Honest scope: this is a GO with a documented caveat, not an unqualified close. Finding: `2026-08-18-gnw-reentrant-metacog-gated-deliberation-GO-caveat.md`.
- **Self-organized metacognition (6-seed GO, 2026-08-18).** The confidence→correctness mapping behind the "my decision-margin reads this as low-confidence" hedge is now learned by a local reward-gated three-factor Hebbian rule instead of a host logistic fit (mean type-2 AUC 0.825, meta-d' 2.49).
- **Self-initiated utterance wired into production (GO, 2026-08-18).** On an idle/empty turn the brain now selects a stored concept itself and speaks it as an unprompted remark or question — moat-safe, and byte-identical on every reactive turn (ledger row `self-initiated-utterance`, default-on).
- **The GNW ignition bus became the default organ-combination (2026-08-13).** The spiking Global-Neuronal-Workspace ignition bus now authors the combine-and-decide step host Python used to do, on every live turn — byte-identical to the prior behaviour with a one-flag revert. It also gates deliberation that **abstains** when the workspace reads a genuine multi-answer conflict.

### Rigour: a shipped-broken fix, an observe-vs-drive audit, an independent oracle, and a mapped boundary

- **A "verified" faculty had been shipping broken on the GPU — found and fixed.** The production GPU chat had been silently returning a 400 error on every request (a onebrain parser handed GPU data to a CPU-only routine); every earlier check had exercised only the CPU path where the bug does not occur, so a faculty that read as "wired, on-by-default, verified" had in fact been broken on the GPU. Fixed and guarded by a GPU-free regression test. Finding: `2026-08-19-production-gpu-chat-was-400-crashing-onebrain-parser-tocoo-cupy.md`.
- **An observe-vs-drive audit found zero dead observers.** All 31 default-on faculties were lesioned through the real chat handler to hunt "wired but inert" drift: 23 genuinely change the reply, 2 are shared-substrate plumbing (answer-preserving by design), 6 are not exercisable on the CPU/tiny-demo config, and **0 dead observers** were found. Finding: `2026-08-19-observe-vs-drive-faculty-audit.md`.
- **An independent second simulator was added as a correctness oracle.** A Brian2-based simulator (sharing no code with our engine) is now a test-only oracle for the *vanilla* spiking core — Izhikevich/AdEx neurons, COBA synapses and pair-STDP agree to ~1e-11 mV. Honest scope: this validates the vanilla core only, not the project's custom mechanisms.
- **Learning the "mouth" read-out from scratch is a mapped NO-GO boundary.** Learning the thought→words read-out from scratch plateaus at ~0.34 while a copied read reaches ~0.97; at 5× the data the spiking-read learner stays at ~0.34 (a host-arithmetic learner reaches ~0.86 on the same data), so data/coverage is excluded as the cause and the residual is the noisy few-spike **read** (the deep-credit "gap#4" limit). The decision is to keep the external language-model mouth for now. Finding: `2026-08-19-mouth-substrate-forward-40k-coverage-EXCLUDED-real-credit-limit.md`.

### Honest state of the whole (do not read the wins as more than they are)

Against the project's own "done" bar — a faculty is done only when it is wired **and** on-by-default **and** its host scaffold is retired — the honest state is **~one integrated spiking family plus a bench of validated-but-unwired de-risks**. `scaffold_retired` is **0** across the whole ledger (the single scoped exception is the GNW-bus organ-combination, for its covered class). It is still **co-residency, not one true substrate**: organ routing is host Python and cross-region synaptic interaction is proven for one pathway.

The open-prose "mouth" is still the external Qwen2.5-0.5B transformer scaffold. Many organ internals remain host/NumPy (the VSA recall algebra, hand-assigned assemblies, the appraisal seeding, the plausibility gate). The three properties the north-star still lacks are **fluid open-ended conversation, one true substrate, and emergent-not-hand-wired** structure. Honest negatives under strict biology are recorded as deliverables, not failures. The live per-cycle resume point remains `GAP_CLOSURE_MISSION.md`; `ROADMAP.md` is the plain-language skim.

## [Unreleased] — 2026-08-10 — The integration pivot: the validated faculties are wired into a LIVE conversation; and episodic memory is closed on a dendritic read-out

### The pivot — from "prove each mechanism in isolation" to continuous integration

The owner steered the project from validating each brain mechanism on its own toward **continuous integration**: wiring the parts that already pass their multi-seed tests into the **live conversational loop** and judging each by one question — *did the conversation actually get better?* Running the real 14-turn chat is what exposes a mechanism that looked finished in isolation but was mis-scoped for a real exchange.

The evaluation is a deliberately **toy world** (two agents, three actions, a small fact set). Of the fourteen turns only the in-domain ones genuinely engage; the roughly eight-to-ten out-of-domain turns **correctly abstain** — that silence is the "won't make things up" safeguard working, not a gap. The reply-phrasing "mouth" is the declared temporary language-model scaffold (a small spiking generator, modelling Broca's area) and is kept off or on CPU through most of this work; the grounded *content* of every answer is the brain's own read.

Each integration below is additive and runner-side, with no core-engine edit and byte-identical defaults unless noted, on the CPU/NumPy backend with the seed controlling the substrate. "GO" refers only to the exact six-seed test (42/43/44/100/101/102) that passed — never a whole-faculty or consciousness claim. Affect and self-model read-outs are stated as **functional read-outs only** ("my familiarity monitor reads this as novel") — never as feeling or phenomenal experience.

### Integrations landed on the live chat

- **The no-fabrication safeguard now checks every clause, not just the headline (6/6 seeds).** The mouth's replies are verified clause-by-clause against the brain's own neural fact store — each subordinate clause as well as the leading subject-verb-object — so an invented, ungrounded causal clause ("the dog went east *because* it was looking for water") is dropped while the grounded motion facts, the affect tone, the curiosity question, and the honest abstains all survive.
  On the real chat, confabulations drop **3→0**; all six seeds show zero ungrounded subordinate clauses. Honest scope: one seed's leftover count is a *separate* surface-detector tokenizer edge (a contraction missing from the stopword set), not a safeguard failure — flagged for a one-token fix. Finding: `2026-08-10-INTEGRATION1-subclausal-moat-live-chat-confab-6seed.md`.
- **Episodic dialogue memory — the brain recalls the conversation instead of going silent (single-seed behavioural).** A question that refers back to an earlier topic ("you mentioned a cat — what was it doing?") is no longer a silent abstain but an honest, grounded recall of the *actual* prior topic. When the premise is false (only the dog was ever discussed), the brain does **not** fabricate a cat memory — it recalls the dog it did discuss. Three symmetric cases hold: genuine recall, honest false-premise, and silence on an empty store. Finding: `2026-08-10-integration2-episodic-dialogue-memory-turn7.md`.
- **That recall now runs on spikes on the CPU backend — a corrected claim (retracted-and-superseded).** An earlier write-up of the spiking recall path claimed it could only fire on the GPU backend and read zero on the CPU/NumPy substrate (a supposed forward-Euler limit). A direct six-seed test in fresh isolated builds **falsified that**: the module had been mis-verified at a firing threshold that fires on *neither* backend.
  The apical dendritic read has a narrow per-assembly threshold window (set it too high and the smallest ~13-cell emergent assemblies fall silent; too low and some self-ignite). At the corrected threshold it fires **cue-specifically 6/6 on both backends**, including on the live CPU substrate (seed 42: cat 0.929 / dog 0.909; the smallest 13-cell assemblies fire; permuted-cue, no-cue and lesion controls all exactly 0.000). So the recall *gate* is now genuinely spiking with no GPU needed — the CPU non-firing was the *wrong operating point*, not a substrate wall.
  Watched residuals: build-to-build variation in emergent membership at the firing threshold, and a small cross-topic specificity margin — both safety-netted by the eval's self-consistency guard so chat honesty holds regardless. The fact *content* buffer remains a named host scaffold (the next conversion). The old "CPU backend-blocked" text is **superseded — do not re-cite it.** Finding: `2026-08-10-episodic-dialogue-recall-wired-to-spiking-dAP-readout-numpy-backend-honest-negative.md` (status: corrected; registered in `docs/RETRACTED.md`).
- **Honest inner-state read-outs, with a robust certainty band (6/6 on the certainty test).** Asked "how do you feel?", the brain gives a **functional** affect self-report derived from its spiking valence differential (a small positive valence read as a low warmth level), explicitly framed as "an honest functional read-out, not a feeling" — never a phenomenal claim.
  Asked "are you a simulated brain?", it gives an honest *structural* self-affirmation ("I am a simulated spiking substrate … not a person") plus a **graded certainty band** from the self-model's confidence relay. This last piece first landed as an honest *negative* (the pooled relay read total drive, not confidence-versus-tie); it was root-caused (a continuous clamp stopped the winner-take-all from resolving) and fixed runner-side by letting the competition settle before reading, then hardened with an **opponent comparator** that reads the *asymmetry* of the settled competition rather than its magnitude — so a partial tie can no longer inflate it.
  Result: confident-versus-tie separation clears the "meaningful" bar on **all six seeds** with >2.5× headroom. Honest scope: the read is still a host subtraction over two neural population rates; a fully-neural opponent read-out pair is the named (deferred, not walled) next step. Composed live chat (seed 42): 6 honest replies, 8 honest silences, **0 confabulations**. Findings: `2026-08-10-INTEGRATION-3c-certainty-band-opponent-margin-robust-turn13-all6-clear-002.md` (and the #3 / #3b predecessors).
- **An honest causal-query disclaimer instead of an evasion (6/6 seeds).** "Why did the dog go east?" previously deflected to other stored motion facts, which reads as evasion — the brain has no causal faculty. It now **confirms the stored fact** through the no-fabrication safeguard (a spiking unbind of the who-did-what-to-what fact) and **honestly discloses the faculty's absence** ("I have learned associations, not causes … I will not invent a reason") — a functional read-out of a missing capability, not a phenomenal claim.
  Six seeds, zero confabulations, and a per-turn exact comparison shows **only this one turn changes** (byte-identical elsewhere). The disclaimer template and its trigger are declared host scaffolds; the fact-confirmation and the suppression of an invented reason are mechanism. Named follow-on: a truly-emergent answer would compose stored facts into a grounded causal chain via a learned relational faculty (the toy substrate stores flat associations with no causal graph). Finding: `2026-08-10-INTEGRATION-5-honest-causal-query-disclaimer-turn4-6seed.md`.
- **Grounded breadth learned from the corpus the brain "heard" (6/6 GO).** The chat could previously only talk grounded-ly about two subjects (dog and cat) from six hand-taught facts. It now stores relational who-did-what facts **mined from the corpus the brain heard** (a children's-story corpus), wired in through a single additive vocabulary argument. Grounded-subject breadth rises **2→9**; grounded replies rise **4→9** against the matched six-fact baseline; six seeds, zero confabulations; the safeguard holds by construction (zero false accepts, every invented proposition dropped) and out-of-domain turns still abstain.
  Anti-cheats: a permuted-corpus control drops the overlap to ~0 (the knowledge is word-order-derived), and the key **empty-knowledge-base control** (same expanded vocabulary, zero stored facts → breadth stays 2) shows the competence is in the *facts*, not the vocabulary; capacity holds well under the single-network binding bound. Declared scaffolds, named not hidden: host fact-mining (the "linguistic environment" boundary), the host fact-write, and host frame-rendering; the named emergent successor learns co-occurrence in *synapses* rather than a host mine-and-store. Finding: `2026-08-10-INTEGRATION-6-corpus-learned-facts-into-live-chat-6seed.md`.
- **Facts learned by synaptic plasticity — the emergence-bar burn-down of the previous item (6/6 GO).** Where the corpus-breadth item wrote facts in through a host store, this replaces that at demo scale: the brain is **taught three facts by corrective interaction**, so each fact becomes a genuine **weight change on a spiking read-out**, and the chat answers about them behind a **learned familiarity gate** as the no-fabrication safeguard.
  Six seeds: taught-recall goes 0→3/3 while a *frozen* read-out recalls 0 (the content rode the weight change, not a host path); the facts are absent from the host fact store; safeguard false-accepts are 0 at chat scale for both untaught cues and out-of-domain turns; **lesioning the learned gate collapses the novelty margin to 0.00 and confabulation returns** (it is load-bearing); a mis-paired teacher yields near-zero recall against 0.94–0.99 when taught. So the *acquisition* is now synaptic and the *safeguard* is learned.
  Honest ceiling: this is a **small, jointly-taught demo (three facts)** standing beside the host-stored breadth above — multi-fact continual acquisition without forgetting is an **open arc** (recall currently scales roughly as one-over-the-count; sleep-replay partial; a sparse-gated read-out negative; neurogenesis partial), and scaling this path up to the corpus-breadth item's size is explicitly gated on that continual-learning arc reaching GO. Finding: `2026-08-10-INTEGRATION-7-plasticity-learned-facts-into-live-chat-6seed.md`.
- **The learned safeguard is now fully spiking too (6/6 GO).** The learned familiarity gate above was still a host (NumPy) projector. This swaps in the project's standing **spiking** familiarity gate — the same conjunctive-novelty projector, but read through a real resonate-and-fire phasor conjunction — behind an additive flag. Six-seed GO: the whole gate holds with the abstain decided **on spikes**; lesioning the spiking pool collapses the novelty margin to 0.00 and confabulation returns; byte-identical when off.
  So **both** the plasticity-learned fact's acquisition **and** its no-fabrication safeguard are now the brain's own. Remaining burn-downs, named per the project's no-defer law: (1) merge the two co-resident spiking bridges into **one** network — the "one brain" step, and the *next* arc; (2) replace the argmax fact read-out with a neural motor read-out; (3) the small-count continual-breadth scale-up. Finding: `2026-08-10-INTEGRATION-7-burndown2-spiking-familiarity-gate-moat-fully-spiking-6seed.md`.

### gap#5 — episodic memory closed via a per-cell dendritic read-out

The episodic-composition seam (gap#5) is **closed at the read-out level** (6/6 GO). The emergent loop now composes end-to-end: an emergent dentate layer **selects** the assembly, a one-shot burst-timing rule **forms** the recurrent attractor, and an intrinsic **per-cell dendritic plateau read-out completes** it cue-specifically. The prior recurrent-attractor completion path had hit a self-drive-versus-cue wall because the emergently-selected assemblies are *small* (~14–33 cells) — too small for a recurrent bistable attractor at any inhibition — and that same work **corrected an earlier "assembly-too-small" diagnosis** (a random 72-cell assembly failed identically, so size was ruled out; the real cause was self-drive-versus-cue coupling at composition scale).

The fix reads memory from a **size-independent per-cell apical dendritic bistable latch** (a coincidence plateau with self-regeneration and a down-state, weakly coupled to the soma so the plateau *holds* the memory without re-driving the loop).
Six-seed GO on the apical read: held-out members reach the up-state from a partial cue while permuted-cue, silent-rest, no-encoding, and recurrence-zeroed controls are all exactly 0.000; a linear coincidence-off control fails on all six seeds (the plateau is the completer, not the weights alone); lesioning the mossy input collapses membership completely (the dentate selection is load-bearing); the one-shot formation is genuine.
Honest scope: this is a de-risk GO at one density, read *during* the cue; the *shipped default* completion path is still the recurrent read, so wiring the dendritic read-out in as the default is the integration step — which the episodic-recall item above did for the live chat. Named residuals, each a next method and none a wall: a homeostatic per-cell plateau threshold that self-adjusts (replacing the global constant with the companion process the animal runs), robustness across the full density range, and a hold-after-cue-offset test for a true standalone attractor.
This is explicitly **distinct** from the tested-negative dendritic deep-credit rule — the plateau *reads* a burst-formed weight; there is no hidden-credit learning (gap#4 deep-credit-on-spikes remains a mapped, deprioritized boundary). Finding: `2026-08-10-gap5-lever-B-dendritic-dAP-readout-completes-emergent-small-assembly-6seed-GO.md`.

### Same-cycle supporting results

- **Learn-to-speak learning wall fixed** with a per-context state-value critic (modelling the songbird Area-X/VTA pathway): the collapsing per-action value estimate is replaced by a signed advantage that compounds (contingency 5/6).
- **Reward-misspecification distinctiveness term: partial**, re-diagnosed as a substrate coincidence-detector artifact rather than a belief-representation gap.
- **A vigilance (noradrenergic gain) result is an honest negative on the real substrate (3/6):** the idealized positive does not robustly transfer through the real neuromodulator manager — it is seed-fragile once the heterogeneity, noise and threshold variation the probe had replaced with constants are restored.
- **A cross-arc winner-take-all reframe:** several "silent / latched" neural WTA negatives were largely an over-strong-inhibition operating-point artifact (separable-assembly WTA is weight-controllable, verified), so recall runs on the associative afferent directly and the WTA competition is inert.
- **A neural word-decode de-risk** burns the host cosine-argmax in the production speaker path (parity 1.000 with the host, 6/6; the decision is now made on spikes via lateral inhibition) — wiring it into the speaker is the named follow-on.
- **A composer "arity-capacity break" was retracted** as a read-out DC-offset artifact and corrected: neural superposition composes through arity 6 with no break in the tested range (registered in `docs/RETRACTED.md`).

The live per-cycle resume point remains `GAP_CLOSURE_MISSION.md`; `ROADMAP.md` is the plain-language skim; **the integration arc is the current frontier** — wire more validated faculties into the chat, dependency-ordered, and gate on whether the conversation improves.

## [Unreleased] — 2026-07-23 — Closing the last capability gaps on one spiking brain; a rigorous self-audit of the record; faster engine and a move to Linux

### The research direction: five open frontiers, and real progress on them

The project is now driving toward a single, explicit goal: make **every** cognitive step of the navigating-and-conversing brain happen in spiking neurons on **one** network, and close the handful of capabilities that still lean on ordinary program code. Five frontiers are being worked in parallel:

1. **Open-ended fluent generation** — moving past a bounded set of sentence shapes toward free conversation produced by the brain's own circuitry. As groundwork, a from-scratch **language-model training pipeline was validated on real web text** (an educational-web corpus) and made to **resume itself automatically** after an interruption — so the brain can eventually learn language from a large, natural corpus rather than a toy one.
2. **Learned concept binding** — replacing the hand-designed scheme that combines concepts into facts (today a fixed mathematical binding rule) with one the brain *learns*. A **spiking, learned binder now covers the full set of fact shapes the system actually uses** — plain who-did-what facts, facts with a described object, and facts that point at an embedded clause — validated across multiple random seeds, with the no-fabrication safeguard intact. The fixed algebra can now be retired on that path.
3. **Resolving ambiguous references** — deciding which of several remembered things a bare pronoun ("it") refers to, via competition between the candidate memories (biased-competition inhibition). Validated across multiple seeds.
4. **Dendrite-based credit assignment** — a biological, local learning rule (how a neuron decides which of its inputs to strengthen) that does not use backpropagation. The rule is shown to build genuine multi-layer accuracy in rate-based tests; the open piece is realizing it faithfully on the sparse *spiking* substrate, now characterized down to a specific operating-point question rather than a vague wall.
5. **Memory replay and imagination** — the brain internally replaying and recombining stored sequences the way the hippocampus does during rest. Hippocampal **pattern completion** (recalling a whole memory from a partial cue) was closed on the spiking substrate; ordered sequence **replay** is progressing through candidate biological mechanisms (rhythm-gated inhibition, intrinsic spike-frequency adaptation, theta/phase organization), with each dead-end recorded as a ruled-out *method* rather than a closed *capability*.

A related milestone: **grounded conversation now runs reachably on one shared network** — the conversational binding system, a learned language layer, and the fact store are co-resident on a single simulation network, so a full grounded exchange happens in one place rather than across separate brains.

### An honest self-audit corrected the record

An internal audit of the project's own historical notes found and **corrected several overstated headline claims** — the kind of self-checking this project depends on. Several **navigation** benchmark figures in particular were walked back:

- A widely-copied "navigates with no heuristic / all shortcuts closed" description was wrong. That
  configuration still had the goal-direction heuristic switched on by default.
- At least one "X% better than baseline" navigation improvement was a favourable-seed selection. It
  does not survive fresh blind seeds.
- A "the larger grid is 13.3% better than the smaller one" comparison was mixing two different metrics. Those claims are retracted in the record. The lesson is restated as standing practice: **prefer qualitative capability descriptions to fragile single-number headlines**, and scrutinize a nominal win at least as hard as a failure.

### A subtle reproducibility bug — found and fixed

A latent bug was found in how random seeds control the simulated substrate: a reporting-only field was being set in some runners in the mistaken belief that it seeded the network, when in fact the per-neuron parameters were coming from an *unseeded* global generator. The effect was that two "same-seed" runs could quietly build **different** neurons — a confound large enough to have muddied one learning-rule research arc (comparisons that looked controlled were not). The fix is simple (set the real seed field), it is now covered by a determinism test that builds the substrate twice at one seed and checks the neurons are byte-identical, and the affected runners were corrected. The engine itself was never wrong — it seeds correctly the moment the right field is set.

### Engine: faster inference by default, and a move to Linux with portability confirmed

- **~3× faster inference in the read-only regime, by default and bit-for-bit safe.** Two performance paths are now on by default and verified byte-identical to the previous behaviour: a read-only fast-step path that removes two per-step CPU↔GPU synchronizations, and a fused "megakernel" that folds the per-neuron update chain into a single GPU launch. Both are guaranteed inert whenever learning or an experiment is active, so scientific results are unchanged. The learning path also gained a branchless (compaction-free) spike-timing-dependent-plasticity update, and the Adaptive-Exponential neuron model gained the same fast spike-reset branch the Izhikevich model already had.
- **Moved to Linux, portability confirmed.** The development environment migrated to Linux; the engine was re-verified end-to-end on the GPU, and the CuPy (GPU) / NumPy (CPU) backend split continues to let it run with or without a GPU. New crash-recovery tooling and an auto-resume-on-boot service were added after a GPU hardware fault, so a long training run picks itself back up (bit-exact) after a reboot.

## [Unreleased] — 2026-07-12 — The long-range-language frontier, reframed: the brain can learn it with biological rules — and *not* the way we expected

### Added / Changed (research — all rate-level + on-substrate probes, no `sim/` edit)
- **The decisive reframe (`research/findings/2026-07-11-R3-REFRAME-*.md`, 6-seed).** The long-range-language question — can the brain learn structure that depends on distant words using only biologically-legal learning (local, no backprop-through-time, no weight transport)? — was run to ground. Finding: long-range capture is **input-representation bound, not recurrent-credit bound.** A network with **fixed, random recurrent wiring** (a reservoir) that learns *only its input representation* **beats full backprop-through-time** that trains everything (training the recurrent wiring is counterproductive). So the hard-looking "deep recurrent credit" rewrite is *not needed* for long-range language.
- **The biology-legal version works, multi-seed (~78%).** Local input-learning (an eligibility-trace rule) + **learned local feedback** (Kolen-Pollack weight-mirror, replacing the impossible weight-transport) + a local read-out reaches **~78%** of the full-backprop reference (plain +0.176±0.001, +learned-feedback +0.351±0.003 across seeds). One-step-local, no BPTT, no weight transport, no `sim/` edit. The residual to full backprop is a characterized feedback-alignment cost with a known next lever.
- **An honest self-correction (the discipline working).** A promising "dual-timescale" breakthrough was caught by an adversarial-verify workflow to be a hidden learning-rate artifact (plain e-prop at 5× learning-rate reproduced it exactly) and **retracted before it was recorded** — a would-be shortcut prevented.
- **On-substrate realization scoped + verified (`2026-07-12-spiking-realization-scoping-*.md`).** The mechanism realizes on a real `SimulationBridge` with the **already-committed `enable_bdsp` dendritic burst-learning rule** on a plastic input→reservoir pathway — verified against `sim/bridge.py` to need **no engine edit**. Cheapest-first spiking de-risk in build.
- **Process (`feedback_run_ceiling_early_and_keep_gpu_busy`):** the whole arc was recontextualized by running a reference transformer *ceiling* early — the earlier small-scale long-range negatives were partly scale-confounded (at genuine scale, TinyStories/WikiText-103, long-range signal is real; a transformer's advantage grows with context). Lesson saved: run the ceiling first, keep idle GPU busy.
- **Docs:** `ROADMAP.md` §9.1 + the two-gaps summary and the `docs/diagrams/` flowcharts updated to reflect the reframe.

> Note: this CHANGELOG's previous tip was 2026-06-30; the intervening July work (the EMERGE-56..85 spiking-Broca/self-organized-grammar/reservoir-comprehension arcs and the D3 discourse-event register) is recorded in `CLAUDE.md` + `research/findings/` but is a pending CHANGELOG backfill.

## [Unreleased] — 2026-07-10 — Discourse memory: the brain tracks who is doing what across a multi-turn conversation

### A spiking, updatable memory of "who" across a conversation

- **A two-gate discourse register.** A running memory of the current event (who is doing what) can now be *held* and *resumed* across a conversation. When a sentence introduces a new named subject after a connective ("… but Mary …"), a **push** gate copies the running event into a held slot; when the conversation returns with a pronoun ("… and she …"), a **pop** gate reads that held slot back — so the brain can answer both "who is doing it *now*?" and "who was doing it *before*?" across the turns. Both memories are held as self-sustaining loops of neural activity on a single shared network, and the read that resumes an earlier subject does so *without erasing* what it reads. Validated across multiple random seeds; no core-engine change.
- **Replay stands in for backpropagation-through-time.** A striking result: the only reason the held slot has value is for a *future* question — nothing in the present rewards holding it — so an ordinary training signal that reaches back through time cannot teach it. Replacing that with a **hippocampus-style replay** target (after an event ends, the brain replays it and predicts its last observed detail from what the slot now holds) fully recovers the ability using only a **local, one-step learning rule plus replay** — exactly the pair a real brain has. Scrambled-replay and stateless controls collapse, confirming the mechanism; the cross-clause transition it depends on is likewise learned by a biological local rule (feedback alignment, with no weight transport). Honest scope: learning the whole register to full accuracy purely on the network, and wiring the finished capability into a console a person talks to, are the reported open follow-ons.
- **A next-token language layer from a fixed-random reservoir.** Related work showed that a fixed-random recurrent network (a reservoir) whose *input representation* alone is learned can pick up genuine next-token language dynamics across multiple seeds — a cheap, biologically-plausible path to sequence prediction that does not require training the recurrent wiring at all.

## [Unreleased] — 2026-07-03 — The brain speaks in its own neurons: self-organized grammar and spiking speech production; comprehension without templates

### The speech-production system (models Broca's area) — built, then self-organized, then put fully on spikes

- **The brain produces its grounded answers as spikes, not as text templates.** The emergent-language system's replies used to be assembled by ordinary string code. They are now produced by the brain's own **speech-production circuitry** (modelling Broca's area): each reply is an ordered set of typed slots (function-word slots like "the / can / does / not", and content slots carrying the right inflection), the *order* is produced by a spiking competitive read-out, and every word — content words *and* function words — is spelled by driving the word's concept population on a real network and decoding it from the language-output neurons. The no-fabrication safeguard is gate-first: when the brain chooses to abstain, the speech system is never invoked.
- **Its entire grammatical structure is discovered from a text stream, not hand-written.** The three things that used to be hand-specified — which words are function words, the order of slots, and which slots each construction uses — are now **discovered from corpus statistics**: function words from their high-frequency, distribution-flat, phrase-edge signature (with later cues for morphology and attributive position); slot order from corpus word-order statistics; slot inventory mined from the corpus. A permuted-corpus control collapses the multi-slot constructions, proving the structure is corpus-derived rather than smuggled in. The set of sentence shapes it can speak was broadened from three to seven, including transitive ("the wolf chases the ball") and ditransitive ("the dog gives the cat a bone"), each rendered exactly on spikes.
- **One brain, one process.** Structure-discovery, reasoning, and fully-spiking speech now **co-execute in a single process on one backend** — the whole flagship exchange ("can a penguin breathe?" → reason "yes" → speak "the penguin can breathe") runs in one place. The *only* core-engine change in this entire arc is a single small additive backend accessor; everything else is reuse of existing machinery.

### Comprehension without hand-written templates

- **Form→role mapping is learned, not branched by hand.** The last hand-written piece on the *understanding* side — a rule per sentence shape that assigns who-is-the-agent — is replaced by a **fixed-random recurrent network (a reservoir) with a trained read-out** that learns the mapping from the discovered function-word pattern, and can resolve a non-local dependency (a relative clause) that no fixed window can. Realized on the project's spiking neurons on a real network, its graded-memory advantage survives on spikes. An adversarial review caught and forced a rebuild of an over-optimistic first version (its "held-out" test was trivially local) before it was recorded.
- **Bounded recursion via a working-memory buffer.** The reservoir handles one level of centre-embedding perfectly and then degrades (fading memory, not a stack). Adding a rhythm-multiplexed **working-memory buffer** that holds the number-markers in ordered slots pushes reliable nested subject-verb matching to depth three before hitting the biologically-faithful human limit; scrambling the buffer slots collapses it, confirming the ordered slots are doing the work.

All of the above is validated across multiple seeds with collapse controls; no core-engine change beyond the single additive accessor noted.

## [Unreleased] — 2026-07-02 — Learning categories, taxonomies, and inference from experience — then conversing about what it discovered

### The brain discovers structure from experience, unsupervised

- **Categories and taxonomies discovered on its own.** By observing a stream of experience — word co-occurrences, or objects *seen* through the real retina → V1 edge-detector front end — the brain **discovers categories** (that several things are a kind of "bird") and even multi-level **taxonomies**, with no labels and no dictionary. A competitive, self-organizing pooling layer learns the shared representation; control conditions (permuted input, scrambled images, a lesioned pooler) collapse the effect, and image-provenance checks confirm the structure is genuinely visual rather than injected.
- **Inference that goes beyond what it was told.** On the discovered structure the brain **inherits** properties down the taxonomy (a never-taught robin "can fly" because a bird can), **cancels** them for exceptions (a penguin walks, yet still breathes — cancellation is per-property), inherits across **multiple levels** at once, and makes **transitive** inferences (from only adjacent premises A > B … D > E it infers B > D). None of this uses a separate inference engine — it emerges from the overlapping learned codes plus the brain's own next-state predictor. Each ability is checked with held-out items and collapse controls, across multiple seeds.
- **Fully on spikes, end to end.** The discover→categorize→infer pipeline runs with no ordinary-code shortcut in the competition step: pixels → real Gabor/V1 filters → a spiking sparse-expansion pooling layer → on-network inference.

### Conversing about what it discovered — grounded and fluent

- **A conversational inference console.** You can teach the brain a taxonomy and properties in plain sentences ("a robin is a bird", "a bird can fly"), ask it questions it was never directly told ("can a robin breathe?" → "yes", inherited two levels up), and get an honest "I don't know what a zzz is" for the unknown — the no-fabrication safeguard, grounded in the brain's own (even *perceived*) categories.
- **Wired to fluent speech, safely.** The grounded reasoning was connected to the small local language generator so the answers come out as fluent English ("no, the penguin walks") — gate-first, so the generator is never invoked on an abstain and cannot fabricate. A small format-only fine-tune of that generator on the grounded sentence forms reached faithful rendering with no catastrophic forgetting of its original abilities.

All arcs are reuse of existing machinery (with a single small additive engine kernel in one place), validated across multiple seeds with collapse controls, and preserve the no-fabrication safeguard throughout. An exhaustive adversarial audit of this arc found and fixed a systematic class of measurement/control defects (held-out sets that weren't truly held out; over-reliance on a weak control), and every surviving result passed its corrected test — the same audit discipline that has repeatedly caught the project's own overclaims.

## [Unreleased] — 2026-07-01 — Fluid, LLM-like conversation; and a living brain that develops with a body and remembers

### Talking to the brain like a chat assistant — while minimizing the language model

- **The "minimize-the-transformer" thesis, validated end-to-end.** The aim is to talk to the brain about almost any topic, grounded in its own knowledge and experience and the conversation so far, and have it grow through the exchange — while shrinking the ordinary language model as far as possible. The approach: a **small, locally-trained generator (~21M parameters, roughly 15–25× smaller than a small external language model)** supplies fluent English *phrasing only*; the **brain supplies all the cognition** — comprehension, knowledge, grounding, and the decision of whether to answer at all. The generator is never invoked when the brain abstains, so the no-fabrication safeguard holds by construction. This was built and validated across a full stack: fluent grounded answering (not rambling), multi-turn pronoun resolution, learning new facts mid-conversation and generalizing the *format* to unseen entities, and real encyclopedic breadth fetched on demand from a public knowledge base.
- **An interactive console ties it together** — ask what/who/yes-no/describe questions, resolve pronouns across turns, teach a fact live, get connected multi-sentence answers, compare two things, and **remember across sessions** (a learned concept's representation is deterministic, so the learned facts can be saved and re-instated on the next run). Honest scope: the generator still runs as an ordinary network (a validated, temporary scaffold; spiking conversion deferred), and fluent single-pass *synthesis* over many facts still confabulates on a model this small — so the system lists/groups facts rather than fusing them — which, along with fully open-domain conversation, is named as the genuine remaining frontier rather than papered over.

### A living brain: develops with a body, remembers across restarts, one shared drive

- **The one brain lives a simulated life** — not a battery of demos. A single unified brain forages under a hunger drive, **perceives and grounds the objects it encounters during its own behaviour**, can be **asked about what it lived**, and **persists across a reset** (its body state, the facts it lived, and the grounded representations all resume). Validated across multiple seeds, with every control decisive: the hunger drive is load-bearing for survival, corrupting the grounded percepts collapses recall, and the brain abstains on anything it never encountered.
- **It develops over "days" without forgetting.** Across successive days it forages a progressively richer world, so its lived knowledge **grows day over day from perception** (not a scripted curriculum), while older lived facts are **retained** — and the developed brain persists across a reset. Validated across multiple seeds.
- **One shared drive touches both halves.** The *same* hunger-raised dopamine that shapes the acting half also tightens the *conversational* no-fabrication safeguard — a hungry brain abstains more and closes off low-confidence false answers — demonstrating a single emotional/value core modulating both action and speech. Honest scope: the learned spatial policy still uses a validated rate-based stand-in, and persistence is a structured re-instate rather than a raw synapse snapshot — both named as deliberate scaffolds; no core-engine change (one off-by-default switch).

## [Unreleased] — 2026-06-30 — The largest spiking generative model yet, and "one brain" delivered end-to-end: the whole conversational turn runs in neurons, driven by a shared emotional core

### Added

- **The biggest brain-runnable generative model to date — ~88.6M parameters, running faithfully in spikes.** A ~88.6M-parameter text generator was trained on a 143 MB simplified-Wikipedia corpus and then run on the spiking substrate: its spiking output matches the ordinary-computer version essentially exactly (perplexity ratio 1.0000) — the largest generative model the project has shown running faithfully as spikes. Alongside it, a long-standing "it stops working above a certain size" belief was overturned: it was two ordinary bugs (a mis-set fine-tuning rate + an over-trained base), not a real wall. At this size the model *learns a new topic while keeping the old one* (92.9% retained), and the "sleep-replay" mechanism that prevents forgetting helps more the more it's used. Honest limit recorded: at ~88.6M parameters the model saturates a 41M-word corpus and stops improving — going further needs *more data*, not more compute. Finding: `2026-06-30-100M-C2-scaleup-C1-GO-C2-nuanced.md`.

### Changed

- **"One brain" delivered end-to-end: the whole conversational turn runs in neurons, driven by the same emotional/value core that drives movement.** Two closures. First, the shared dopamine signal that already guides navigation now also shapes *conversation* — it sharpens the "don't make things up" threshold, strengthens how strongly a salient fact is stored, and (newly, six seeds, zero fabrications) biases *which* remembered fact wins when a question is asked. One shared reward/value core now drives both action and speech — the "one self" the roadmap aimed for. Second, the last places where a question's internal steps briefly dropped out of neurons into ordinary code were closed: the whole "who did what?" turn now runs on one brain, in spikes, start to finish (validated at the full 2048-dimension / 320-word production scale, the "won't make things up" guarantee bit-for-bit intact). No engine edits anywhere — every piece was an off-by-default switch or a reuse of existing machinery. Findings: `2026-06-30-tier2-6-limbic-to-composer-scoping.md`, `-integrated-spiking-loop-scoping.md`, `-R4-perception-compose-grounding-onbridge-CLOSE.md`.
- **A couple of conversational polish skills folded into the shipping agents.** Two abilities previously only proven in test scripts are now in the production agents behind off-by-default switches: resolving *which* of several remembered things a pronoun refers to using a graded (rather than fixed) nudge, and understanding sentences with an embedded relative clause ("the dog that chased the cat ran"). Defaults unchanged; the "won't make things up" guarantee holds throughout.

## [Unreleased] — 2026-06-19 — One brain, fully spiking by default: the navigation decision moves into neurons; the conversational engine gets 10–20× faster and a little richer

### Added

- **The navigation move-decision is now made in spikes — by default (6-seed GO).** On the one shared brain, the choice of which way to step now *emerges* from a race between competing neural populations: an accumulator integrates the evidence (the way a working-memory circuit holds and builds up a signal) and the race ends when the winner fires an all-or-none committing burst. The old off-brain "pick the best option" step (a plain `argmax` in ordinary code) is retired, kept only as an optional baseline for benchmarks. Validated across six random seeds on a 32×32 grid: the decision terminates on the neural burst **100% of the time** (it never silently falls back to the shortcut), at a cost of about **16% more steps** than the shortcut — a genuine, reported cost of doing the decision in neurons rather than a number hidden away. The conversational half is untouched (its neurons stay exactly unchanged, so the "won't make things up" guarantee holds by construction). Two sibling navigation steps stayed honest open boundaries rather than being forced: a self-organizing place code and a moved-goal re-learning step each hit a well-localized substrate limit and are reported as such. No engine edit — a runner-side default flip. Finding: `2026-06-19-spiking-decision-default-on-GO.md`.

### Changed

- **The conversational engine is 10–20× faster.** Answering a question used to rebuild a large internal table from scratch on *every* query; that table actually never changes between queries (it depends only on the vocabulary and layout, not on the facts stored), so it is now built once and reused. A question now returns in **under a tenth of a second** on a desktop GPU — a flat ~9.5 ms regardless of how many facts are stored — and the speed-up *grows* with the size of the knowledge base (about 19.5× at the 32-fact production scale). The answers are identical to before, and the "won't make things up" guarantee is bit-for-bit unchanged. Finding: `2026-06-19-latency-csr-cache-GO.md`.
- **A few richer conversational skills folded into the one production agent (3-seed GO at 320 concepts).** Capabilities that had been validated only in standalone test scripts are now part of the single production agent, behind off-by-default switches so nothing existing changes: understanding a **described object** ("the dog ate the **big apple**" → the patient is "big apple"), and **auto-selecting the sentence frame** so word orders beyond plain subject-verb-object are understood. Validated on the codes the system learned from conversation, at the 320-concept scale, with **zero fabrications** across every seed and read-out. One honest boundary is deliberately *not* shipped: an object with **two** attributes ("big red ball") is still unreliable on the learned codes (the codes' natural similarity defeats the disentangling step) — a documented limit, with the specific fixes it would need written down. Finding: `2026-06-19-consolidation-attr-multiframe.md`.

## [Unreleased] — 2026-06-17 — Closing the conversational loop: the agent talks with the words it learned by listening

### The production agent converses using the meanings it learned from conversation

- **Consolidation (3-of-3 seeds, plus the fully-brain-based read-out).** Two halves of the system had never been run together. One half is a "cortex" that learns what 320 everyday words mean purely by listening to a stream of sentences (it strengthens connections between words that occur together — no dictionary, no labels). The other half is the production conversational agent that parses sentences, remembers who-did-what facts, answers questions, says "I don't know" when it has no fact, handles yes/no and negation, generates a spoken sentence (word order produced by spiking neurons), and decides what to bring up next. This release feeds the learned word-meanings straight into the production agent and runs a full multi-turn conversation on them: it recalls every fact perfectly, refuses to invent answers it was never told (zero fabrications, every seed), and gets yes/no, generation, and topic-association right — on the codes it learned by listening, not on hand-made ones. It is an assembly of already-validated parts rather than a brand-new ability, but it is the first time the loop closes end-to-end on one agent. Finding: `2026-06-17-consolidated-320-production-conversation-GO.md`.

### Reasoning across several facts, and remembering across turns

- **Multi-hop reasoning (now a built-in ability, 3 seeds × 3 sizes).** The agent can now answer questions that require *chaining* stored facts: told separately that "dog eats cat" and "cat eats mouse", it answers "what does the thing the dog eats, eat?" → "mouse" — following the relation from fact to fact, cleaning up the in-between concept at each step. It chains correctly through *four* hops, and it can mix relations along the way ("dog eats cat, cat plays with ball" → follow *eat* then *play* → "ball"). Critically, this is checked against the exact trap that produced a retracted result months ago: a "lazy" baseline that just walks word-co-occurrence is given the same questions and **fails** (the agent beats it by a wide margin), scrambling which facts connect collapses the agent to chance, and at every step it still refuses to invent an answer it wasn't given. Findings: `2026-06-17-multihop-query-chain-GO.md` (the validation), with the ability shipped as `query_chain` / `reason_chain`.

- **Multi-turn memory — pronouns across turns (3 seeds).** The agent now keeps a *working memory* of what was just talked about, held as a self-sustaining loop of neural activity (the way the brain keeps something "in mind" for a few seconds). So in a two-turn exchange — "dog chase cat." / "what does **it** eat?" — the word "it" is resolved to *cat* by reading what the loop is currently holding, and the agent answers correctly. Wiping the loop between turns, or cutting the feedback that lets it sustain activity, both break the resolution — proving the persistence is doing the work — and a pronoun with nothing to refer to is answered with an honest "I don't know" rather than a guess. The two abilities are then combined into one `MultiTurnAgent` where a multi-step reasoning chain is itself held in that same working-memory loop. Finding: `2026-06-17-multiturn-anaphora-derisk-GO.md`.

### An honest negative — and a same-day self-correction

- **"Does the agent confuse similar words?" — No (and the first explanation was wrong).** A natural question: because the learned word-codes carry meaning (dog and cat are near each other), when the agent's read-out is made noisy and it slips up, does it confuse *similar* words (dog↔cat) rather than random ones? Across three seeds the answer is no — its mistakes are essentially random, not semantically sensible (about 2.3× chance, far below what a "confuses-similar-things" story predicts). The first write-up explained this by claiming the similarity structure simply isn't present in the raw codes. A direct check run the same night **falsified that claim**: the structure *is* there (a word's nearest neighbour is the same category about 8× more often than chance), it is just a *thin* margin that the noise needed to cause a mistake easily swamps. The negative result stands; the explanation was corrected in the same session rather than left to stand. This is the kind of self-check the project relies on (the same discipline that once caught a year of misleading "successes"). Findings: `2026-06-17-within-category-error-signature-NEGATIVE.md` (with the correction inline).

## [Unreleased] — 2026-06-16 — The unified embodied agent: one brain navigates, perceives, composes, and converses; cross-region interaction; generalization across similar concepts

### One brain that does it all at once

- **The unified embodied agent (6-seed GO on the integration).** A *single* `SimulationBridge` (one network, one update loop) in which navigation, perception, fact-composition, and conversation all run as distinct, non-overlapping groups of neurons. In one continuous live episode the agent navigates a grid from simulated vision (the basal-ganglia action cascade selecting each move in spikes), perceives the objects it passes, **binds a perceived object into a new who-did-what fact**, answers a who/what question about it, and **abstains** ("I don't know") on anything it never saw. Validated first on a single random seed, then across six (42/43/44/100/101/102): the integration, the no-fabrication guarantee, navigation, fact-composition, conversation, and sentence-parsing all hold on every seed — four are a full pass; the two misses (100/101) are a per-seed fidelity wobble in the *generalization* step only, diagnosed as a side-effect of co-locating the circuits (one tempting fix — adding more training examples per category — was rejected because it widened a gap in the no-fabrication gate, and that bar is never loosened to manufacture a pass; a fix that leaves the gate untouched is under test), and is the one open edge, not a lapse in the no-fabrication guarantee. Reuse-by-import; no engine edits. Findings: `2026-06-16-unified-embodied-agent-stage2-GO.md`, `2026-06-16-navigate-to-compose-then-answer.md`.

### The two brains genuinely interact through synapses (the real "one brain")

- **Language → action (6-seed GO).** A *spoken command* steers the navigating body: the conversational parser's firing opens a synaptic gate that lets the learned word→action route bias the navigation cascade, with the command as the only goal signal. Every control is decisive (cutting the route collapses command-following to chance even with the parser still firing; a scrambled word makes the body track the actual word spoken). Finding: `2026-06-10-spoken-instruction-nav-GO.md`.
- **Perception → memory (6-seed GO).** The agent navigates a corridor, perceives objects live in-episode, tags them as engram ensembles, and afterward **recalls** what it saw via neural reactivation through a trained perception→language route. Coupled recall is perfect on every seed; lesioning the route collapses it; with no body to encounter objects there is nothing to recall. Finding: `2026-06-16-navigate-to-see-then-answer.md`.

### Generalization across similar concepts — the dendritic rewrite was NOT required

- **Vision → spiking concept (GO).** A *novel* object, perceived through the project's real retina → V1 edge-detector front end, makes its concept-neurons fire in the correct *semantic category* (about 3× chance, 3 seeds), with a flat-distinct baseline at chance and a category-derangement control collapsing — so the structure rides the learned vision-category ↔ concept-category correspondence, and the no-fabrication guarantee survives. The mechanism was de-risked four independent ways; crucially the heavier biological machinery people assumed was needed (modelling neuronal **dendrites**) turned out **not** to be required — plain point-neurons with local learning suffice. Biology: convergence-zone / hub-and-spoke semantic memory (Patterson & Lambon Ralph; spiking precedent Garagnani & Pulvermüller 2018). Finding: `2026-06-16-generalization-capstone-vision-to-concept.md`.
- **Verbalize the generalization (hybrid path, 0.92 at 3 seeds).** The spiking concept-category keys the validated fact-binding composer to recall a fact about the matched known category. The *fully*-spiking fact-recall + spiking abstention gate is an honestly reported boundary (at chance, and the runner refused to weaken the no-fabrication gate to force a pass); the validated-recall hybrid reaches 0.92. Finding: `2026-06-16-generalization-capstone-verbalize.md`.

### The conversational pipeline is now (almost) all neurons

- **Biologization sweep.** The four core cognitive steps of the conversational pipeline — the no-fabrication gate, memory cleanup, concept-binding, and output normalization — each now have a validated spiking/neural mechanism rather than an off-brain shortcut. Notably the neural no-fabrication gate (a learned familiarity gate) is *cleaner* than the host check it replaces (zero false-accepts on the same codes). What stays idealized is precisely localized: the binding *algebra* (a learned cortical bind = "step 3") and open-ended sentence generation. Finding: `2026-06-16-biologization-sweep-conversational-pipeline.md`.
- **A generalizing cortex learned from a conversation stream (mechanism GO).** The "step 3" learned cortex — one whose internal codes carry meaning-similarity — is now real on the spiking network at ~64 concepts, **learned from a conversation stream** (the network hears words in context with no pre-processing of the text; a plain local learning rule learns their co-occurrence). A blocker once feared here — having to strip the natural correlations out of the concept codes — turned out to be a *red herring*; the fix was simple local normalization plus online learning. Scaling to the 320-concept production tier is the remaining build. Finding: `2026-06-15-on-bridge-hebbian-co-occurrence-learning-mechanism-GO.md`.

## [Unreleased] — 2026-06-13 — Step 3 (the learned cortex): scaling de-risked to ~2,048 concepts; production build in progress

### Replacing the fixed binding algebra with a learned, meaning-structured cortex

With navigation and conversation consolidated onto one network (step 2, below), the remaining frontier is **step 3**: replacing the fact-binding "composer" — which today uses a fixed, exact mathematical binding rule (a vector-binding scheme) as a stand-in for cortex — with a *learned* model cortex whose internal codes carry meaning-similarity (so similar concepts sit close together). That is what would let the system answer about a never-seen concept by analogy to a similar known one — something the fixed algebra cannot do. This arc was de-risked cheapest-first; the production build is now in progress. **No `sim/` edits in this arc — it is reuse + wiring + validation.**

- **The cortex question forks (2026-06-11).** Running the "learned cortex" idea to ground revealed that the brain's *own* concept codes are correlated (they carry meaning-similarity), and four mechanistically distinct brain-based attempts to *remove* that correlation on a simple point-neuron substrate all failed — converging on a documented limit (decorrelation is an analog, pre-spike computation a point-neuron substrate fundamentally cannot do). But on *decorrelated* codes everything works: a learned binder even generalizes systematically to never-seen role-filler combinations (held-out = train = 1.000, 3 seeds). This forks the cortex into **(A)** a meaning-*flat* cortex achievable now (already passes the full conversational matrix at 320 concepts; its "won't make things up" guarantee now has a validated *neural* version that matches the prior host check with zero breaches) and **(B)** a meaning-*structured* cortex that generalizes, needing a deeper dendritic-substrate rewrite (a deliberate, deferred owner decision). Finding: `research/findings/2026-06-11-familiarity-gate-v320-GO.md`.
- **The learned cortex answers about similar concepts, in conversation (2026-06-12).** A learned, graded cortex wired into the conversational pipeline delivers generalization-in-conversation (answer about a held-out concept via a similar known one — what the fixed algebra could not), with the "won't make things up" guarantee intact: conversational matrix 6/6, generalization 4× chance, zero false accepts, every anti-cheat control collapses. Validated on a 3-bridge ensemble. Findings: `2026-06-12-cortex-conversation-capability-GO.md`, `-3bridge-ensemble-GO.md`.
- **Production architecture decided — route A (2026-06-13).** The production design uses **per-network composers** (each small cortex binds within its own ~64-concept slice, so the per-operation cost does not grow with total vocabulary) plus a cross-network identity layer for facts that span slices. Validated GO at 8 networks / 512 concepts: every within-network matrix passes, generalization 0.99–1.00 on all 8, cross-network composition recalls the true target far above the noise floor while the permuted anti-cheat collapses, moat intact. Finding: `2026-06-13-phase1-composer-architecture-routeA-GO.md`.
- **The deepest scaling risk retired — 32 networks / ~2,048 concepts (2026-06-13).** The cross-network fact-binding and the "won't make things up" guarantee both hold at a 32-network fan-out (4× the validated 8) — true-target recall at 20.95× signal-over-noise across a 2,048-concept floor (no degradation vs the 8-network case), the permuted anti-cheat collapses, zero moat breaches. Multi-seed (42/43/44). Finding: `2026-06-13-phase1-32bridge-fanout-derisk-GO.md`.
- **Corpus-source decision + one inconclusive follow-up.** The production cortex will learn meaning-similarity from a *curated* semantic scheme (validated). A side de-risk testing whether the network could instead learn similarity from *raw* text (a children's-story corpus) came back **inconclusive** — the gold-standard host control also fell short, because that corpus's word co-occurrence reflects scene structure more than substitutability. Logged as a genuine follow-on, not a blocker. Findings: `2026-06-13-option-c-real-cooccurrence-derisk-INCONCLUSIVE.md`; build plan `docs/plans/2026-06-13-phase1-production-build-plan.md`.
- **Status: mechanism + architecture + vocabulary + corpus-source are all in place; the production system (the 32-network cortex + the full conversational matrix at ~2,048 concepts) is now being built and validated — not finished.** Reported as in-progress.

## [Unreleased] — 2026-06-10 — Navigation + Conversational merged onto ONE bridge (roadmap step 2, STEP 2a)

### Navigation fully biologized → consolidating the two brains onto one substrate

- **Navigation + Conversational on ONE `SimulationBridge` (roadmap step 2, STEP 2a — conversational gate green, navigation gate in flight)** — after navigation was fully biologized (every cognitive step between sensation and action is a validated neural mechanism), the navigation cascade, the conversational parser, the dlPFC dialogue planner, and the resonate-and-fire (RF) composer are being consolidated onto ONE bridge as disjoint neuron-index slices (the owner's "one brain" directive). Capability-equivalent to the separate brains, single substrate. Builder `research/runners/nav_conv_merged_bridge.py` (`build_merged_nav_conv_bridge` + `MergedNavConvAgent`). De-risked cheapest-first **before** any protected edit:
  - **De-risk 5a (plasticity isolation) — PASS + one characterized gap.** The per-synapse plasticity gate (`cp_plasticity_rate_gain=0`) isolates weight UPDATES against the full navigation stressor (reward-modulated STDP + the global dopamine `scope="all"` + Hebbian) — a frozen conversational slice stays byte-identical (max\|Δw\|=0), the controls change, a conversational read is unchanged across a navigation burst. The ONE gap: the two global weight CLIPS (`bridge.py:6200` Hebbian, `:6505` reward) are UNGATED, so a frozen weight outside the active rule's clip bounds is moved by the clip; mitigated by raising `stdp_w_max`+`hebbian_max_weight` above the ~300 frozen conversational weight (the RF composer's complex binding weights are array-disjoint from `cp_connections` → immune). Realistic-scale confirmed at weight 300. Findings `2026-06-10-unification-5a-plasticity-isolation-PASS-with-clip-caveat.md`.
  - **De-risk 5b (RF vs Izhikevich) — KILL → the minimal protected edit, owner-approved.** RF stores its complex phasor in the same `v`/`u` arrays Izhikevich uses; one Izhikevich step destroys a phasor (|z| 1.0 → 16.3). But the composer is stateless-per-op (re-kicks each op) and stores memory in complex synapses, so the minimal edit slices the RF ops — `rf_kick(..., neuron_mask=)` + `_rf_advance_one` mask all `v`/`u` writes to the RF slice. **Default `None` = byte-identical** (18/18 conversational tests verbatim incl. the no-confab moat; the 5b reference unchanged); validated co-residence (an RF op on a masked slice == a standalone RF bridge exactly, the Izhikevich slice byte-isolated). `tests/test_rf_neuron_mask_coexistence.py`. **Owner byte-reviewed + approved.**
  - **STEP 2a merge.** The brain-region framework path IS a wrapper around `inject_explicit_wiring` (`bridge.py:1514-1526`), so the parser + dlPFC append as framework regions (decision A). The merged bridge holds nav+parser+dlPFC (2904 neurons; both gates frozen). The conversational gate (b) passes VERBATIM — `tests/test_nav_conv_merged_agent.py` 8/8 incl. the three `is None` no-confab assertions (`what_does`/`elaborate`/`describe`). The navigation gate (a) uses a HYBRID `run_moving_goal_episode` integration (4 additive no-op-default params + an index-based `finalize_conv_for_nav_gate` hook that runs AFTER the V1/SC post-init `set_pathway_weights(add_missing=True)` CSR rebuild — which re-sorts data + stales gate-index maps + the Hebbian decay would erode the fixed perception weights; the hook masks BY INDEX, not gate name, and gain-masks the train pass). The **nav-on-merged smoke PASSES**: the merged bridge (47 regions, 6808 neurons) navigates AND the 720 parser synapses stay byte-identical pre/post the episode + gains 0 (the 5a isolation in vivo). A `stdp_w_max=400` cheap-check confirmed the navigation score is byte-identical to 150 (the actor is ceiling-bound not soft-bound — over-grows to 311 — but inert because the spiking WTA readout saturates). The full 6-seed navigation gate (a) (12 runs, flagship recipe, conv-inertness) is the final statistical rigor (in flight).
  - **STEP 2b** (RF composer co-resident on the one bridge via the owner-approved masked ops) and **step 3** (replace the composer's exact-inverse VSA algebra with a learned spiking cortex, a deferred later arc) follow.
  - All runner-side + one additive default-off byte-identical `sim/` edit (the masked RF ops). Architecture: `docs/ARCHITECTURE_nav_conv_merge.md`. Designs: `docs/plans/2026-06-10-nav-conv-merge-implementation-design.md`, `-nav-episode-integration-design.md`, `-step2b-rf-coresident-implementation.md`. Honest negatives are the deliverable throughout.

## [Unreleased] — 2026-05-30 — Pillar n=110 promoted VALIDATED BOUNDARY (D7 V=320: capacity envelope begins to bend)

### Pillar n=110 promoted (Direction 7 dedicated-pool V=320 cross-bridge composition — VALIDATED BOUNDARY)

- **Pillar n=110 VALIDATED BOUNDARY (D7 V=320 dedicated-pool cross-bridge composition; first measured capacity-envelope bend)** — Direction 7 production decisive multi-seed (seeds 42/43/44, loads {2,3,5}; V=320 = 5 bridges × V=64; vocab byte-identical to the Direction M G.20 sparse 320-concept production deliverable, making this the biology-faithful counterpart of the user-facing G.20 chat capability at identical vocabulary). Returns DIRECTION_7_PASS by the pre-registered frozen mean-rule (verdict module committed 3 days before the result; tests multi-seed MEAN ≥ 0.80): OB perfect 1.000 every load; OI 1.000 / 0.993 / **0.830** at L=2/3/5. **But honestly a BOUNDARY, not a clean PASS:** per-seed L=5 OI = [0.925, **0.700**, 0.865] — seed 43 sits clearly below the 0.80 bar; the mean clears only because seeds 42/44 carry it; the mean margin collapsed from D6 V=160's +0.187 to +0.030. **First vocab tier where the capacity envelope bends:** L=5 OI trajectory 0.977 (D4 V=80, seed-robust) → 0.987 (D6 V=160, seed-robust, FHRR prediction shattered) → 0.830 (D7 V=320, seed-sensitive, one below bar). The whole D7 seed distribution dropped below the D4/D6 range (D7's best seed 0.925 < D4's worst 0.965); seed spread widened 22× (0.025 → 0.010 → 0.225). The dedicated-pool grounded-symbol geometry that shattered the FHRR prediction at V=160 (near-orthogonal, seed-robust, abs cos ~0.04) becomes seed-sensitive at V=320 (seeds 42/44 ~0.06, seed 43 ~0.085) — the orthogonality advantage has a vocabulary ceiling that begins to bind at or below V=320 (n=3 supports the trend, not a pinned ceiling). **Mid-run client crash + KILL-SAFE recovery** absorbed gracefully (12 of 15 cells survived in per-cell cache; E_functional re-trained). **Crash-retrain confound ruled out** by a per-(seed,bridge) geometry diagnostic: seed 43 uniformly ~40% less orthogonal across all 5 bridges, and seeds 42/44 are clean on the same post-relaunch bridges where seed 43 degrades — the degradation tracks the seed, not the run phase. The mandatory smell-test (scrutinize a nominal PASS harder than a FAIL) caught the below-bar seed and ruled out the confound before promotion. Adversarial reviewer ran all 9 scrutiny items + independently re-ran the diagnostic, returned CLEAR with 3 documentation corrections (crash-confound argument strengthened, ceiling-location claim softened to trend, BOUNDARY headline preserved); commit 06af100. Same VALIDATED BOUNDARY pattern as pillar n=106 (D5 hybrid), and **stronger** — D7 clears the frozen mean bar (0.830) where n=106 sat below it (0.790). Wall 880.9 min training compute (~57 hr effective incl crash recovery) + 118.5s probe. SIXTH pillar in the autonomous arc (n=105 through n=110). Commits: 3d83ae0 (result + diagnostic) + 06af100 (reviewer CLEAR + doc corrections) + capability_status promotion. Establishes 'scrutinize a nominal PASS harder than a FAIL; surface below-bar seeds + rule out crash confounds before promotion; report BOUNDARY honestly rather than overclaim a clean PASS' as standing discipline. No bar change; reuse-only; protected set byte-empty; no-confab moat 7/7 green.

## [Unreleased] — 2026-05-27 — Pillar n=109 promoted (D6 V=160 SHATTERS FHRR capacity prediction); D7 V=320 infrastructure shipped

### Pillar n=109 promoted (D6 dedicated-pool V=160 cross-bridge composition)

- **Pillar n=109 VALIDATED (D6 dedicated-pool bio_brain_regions V=160 cross-bridge composition; SHATTERS FHRR algebra capacity-ratio prediction)** — Direction 6 production decisive multi-seed (3 seeds × 5 bridges × loads {2,3,5}; V=32 per bridge × 5 bridges = 160 unique cross-bridge concepts on pure dedicated-pool bio_brain_regions; doubled vocab per bridge vs the pillar n=108 D4 reference). Result: OB perfect every load (1.000 / 1.000 / 1.000); OI 1.000 / 1.000 / **0.987** at L=2 / 3 / 5 (margin > 0.18 every seed). BEATS BOTH pillar n=108 D4 V=80 (OI L=5 = 0.977) AND pillar n=95 G.20 sparse V=160 (OI L=5 = 0.790) at the L=5 OI cell. Production BETTER than smoke (0.987 vs 0.972 = +0.015pp at scale). The FHRR algebra capacity-ratio prediction (capacity ∝ N_dim/V; doubling V should drop boundary ~2 rungs from L=6/L=7 to L=3/L=4) was DECISIVELY SHATTERED at production scale — boundary did not drop, slightly improved with more vocabulary. Wall 817.4 min (~13.6 hr) training + 139.5s probe on CuPy/RTX 3090. Adversarial reviewer 9/9 CLEAR (commit e739543), including D6 > D4 surprise verification across 4 sub-checks (no duplicates; distractor pool genuinely V=160; per-bridge mean-centring at V=32 yields sharper common-mode; anti-cheat primitives byte-unchanged since cd30fc6). Commit c1fca54 (decisive findings); e739543 (reviewer); 43c443d (capability_status). Biology-translatable insight: cortical column-style dedicated representation produces substantially CLEANER FHRR-substrate geometry than distributed sparse coding OR uniform random codes (near-orthogonal because each concept fires its own dedicated pool with other pools quiet); this is a measurable architectural advantage for cross-bridge compositional capability.

### Direction 7 V=320 infrastructure shipped (next-tier vocab scaling on biology-faithful substrate)

- **Direction 7 V=320 infrastructure shipped (commit 72e8964)** — 5 categories × V=64 = 320 unique concepts; vocab byte-identical to the Direction M G.20 sparse production deliverable (g20_bridge{A..E}_*_vocab64.txt). Makes D7 the biology-faithful counterpart of the user-facing G.20 sparse 320-concept chat capability at identical vocabulary. Components:
  - `research/findings/raw/direction_7_vocab_spec.py` — 5 frozen 64-word lists; 320 unique concepts; cross-validated byte-identical to G.20 production vocab via direct file comparison
  - `research/findings/raw/direction_7_bridge_builder.py` — 5 per-bridge builders with `_DIRECTION_7_BRIDGE_LABEL_SEED_OFFSETS` (100k stride); reuses `build_biological_brain_regions` byte-unchanged
  - `research/findings/raw/direction_7_verdict.py` — frozen 0.80 multi-seed bar; PASS / PARTIAL / NEGATIVE / VOID_MALFORMED tags; 14/14 adversarial cases pass
  - `research/findings/raw/direction_7_5bridge_runner.py` — GPU controller with KILL-SAFE caches; scale params preserve D6 per-cue n_active=61 footprint exactly (FULL n_lang=4096 sparsity=0.015; SMOKE n_lang=2048 sparsity=0.01)
  - `research/findings/raw/direction_7_cross_bridge_probe.py` — CPU-only V_total=320 union probe; reuses pillar n=95 + D4/D6 byte-unchanged primitives
  - `tests/test_direction_7_grounding.py` — 11 grounding pin tests all PASS (vocab-uniqueness, V=64 per category, frozen-threshold tampering, bridge-builder seed-offset, protected-builder-byte-unchanged)
- **Pre-staged adversarial reviewer prompt (commit 9303a99)** — 9 scrutiny items including D7 V=320 vs D6 V=160 surprise verification, G.20 sparse 320-tier comparison (same vocab; biology-faithful vs sparse), protected-set byte-empty diff vs reviewer commit e739543.

### Cumulative autonomous arc state

5 pillars promoted in the bug-discovery / FHRR-prediction-shatter arc (2026-05-25 through 2026-05-27): n=105 / n=106 / n=107 / n=108 / n=109. D7 V=320 is the pillar n=110 candidate; smoke in flight at this CHANGELOG entry's commit timestamp (ETA 2-3 hr GPU). The "bug-discovery first when architecture returns chance results" discipline + "FHRR algebra predictions derived for uniform-random codes overstate boundaries on dedicated-pool grounded geometry" discipline are now both empirically established standing principles for this codebase.

## [Unreleased] — 2026-05-26 — Four pillars in one autonomous arc + bug-discovery pattern

### Four pillars promoted (n=105 / n=106 / n=107 / n=108)

- **Pillar n=108 VALIDATED (D4 dedicated-pool cross-bridge composition)** — Direction 4 production decisive multi-seed: OB 1.000 every load through L=7 (3600 trials, zero errors); OI 1.000 / 1.000 / 0.977 at L=2/3/5; capacity envelope passes through L=6 (0.813) before collapse at L=7 (0.608). DRAMATICALLY beats both D5 hybrid (0.790) and pillar n=95 G.20 sparse (0.790) at L=5 OI — pure dedicated-pool is the cleanest cross-bridge substrate the project has produced. The D5 hybrid's shared sparse pool was a workaround for the cross-bridge uniformity bug, not a necessary architectural component. Adversarial reviewer 9/9 CLEAR. Commit 0f7dfd9 (capability_status); 9acadb4 (decisive); 79c1dd3 (reviewer); efbad3d (bug fix). Biology-translatable: 6 simultaneous compositional bindings per theta cycle matches natural utterance complexity (Lisman 2005 theta-gamma 5-9).

- **Pillar n=107 VALIDATED (Wang 2002 cortical NMDA bistability at substrate scale via NMDA:AMPA ratio fix)** — Direction Q-tertiary NMDA-AMPA ratio sweep at n=1000 dlpfc_wm density=0.20 inh=2.0: PASS at nmda_ratio=0.6 and 0.8 (3.00s sustained activity all 3 seeds; rate_ratio 753-897); nmda_ratio=0.4 (default) still PARTIAL. NMDA-off control silent (ratio 0.92-1.06) confirms NMDA-mediated. Closes Direction Q across 4 prior PARTIAL axes (density 0.10/0.20, scale 1000/2000, E/I 2.0/3.0/4.0). Falsifies "Izhikevich isn't biological enough" alternative hypothesis — the substrate genuinely supports Wang 2002 bistability once the conductance ratio is correctly tuned. Reframes Direction I bound (60-neuron PFC bistability failure): works at 1000+ neurons with NMDA-dominant conductance ratio. Adversarial reviewer 12/12 CLEAR. Commit a328d00 (capability_status); e94017e (sweep); c23b7c6 (reviewer).

- **Pillar n=106 VALIDATED (BOUNDARY) (D5 hybrid sparse-distributed bio_brain_regions cross-bridge)** — Direction 5 hybrid production decisive: OB 1.000 every load; OI 1.000 / 0.998 / 0.790 at L=2/3/5; L=5 OI EXACTLY mirrors pillar n=95 G.20 sparse cross-bridge boundary at 0.790. First architecture unifying biology-faithful dedicated pools (n=98/n=105) with sparse-distributed cross-bridge composition (n=95) on a single substrate. Adversarial reviewer 9/9 CLEAR. Commit 8737d41 (capability_status); 7ba8e8d (decisive); 1c7e51a (reviewer); c4e18f2 (bug fix).

- **Pillar n=105 VALIDATED (bio_brain_regions V=16 → V=32 single-substrate vocab scaling)** — Direction 3 V=32 production decisive multi-seed: 18/18 cells PASS at 0.80 bar; L=5 OI = 0.993. Doubles single-substrate vocab from V=16 (pillars n=96/n=97/n=98) to V=32 cleanly without precision loss. Adversarial reviewer 7/7 CLEAR. Commit 068bf1a (capability_status); 3ffae15 (decisive); 7a65e53 (reviewer).

### Bug-discovery discipline lesson

Four reversals in 24 hours where prior NEGATIVE findings turned out to be specific seeding/parameter bugs rather than fundamental architectural limits:

1. **D5 NEGATIVE → bug fix → BOUNDARY**: K-of-N sparse patterns 100% identical across all 5 bridges (seeded only by base_seed); cross-bridge discrimination mathematically impossible. Fix: `_BRIDGE_LABEL_SEED_OFFSETS` map at 100k spacing.
2. **D4 NEGATIVE → bug fix → PASS**: activity vectors byte-identical across bridges at every word position (same RNG seed → identical orthogonal codes + weight matrices). Fix: analog of D5's seed-offset.
3. **Q PARTIAL across 4 axes → conductance fix → PASS**: NMDA-AMPA ratio 0.4 (AMPA-dominant default) drained recurrent loop before NMDA could rebuild it; inverting to 0.6 (NMDA-dominant) latched the attractor.
4. **Direction I bound (60-neuron PFC bistability) → reframed by n=107**: works at 1000+ neurons with correct conductance ratio.

Standing discipline going forward: when architecture returns essentially-chance results, suspect a systematic bug BEFORE declaring architectural failure. Verify cross-bridge inputs are genuinely distinct; verify conductance/synapse parameters are in biological range; only then conclude the architecture itself is bounded.

## [Unreleased] — 2026-05-16 — Trustworthy continual memory + self-contained generator (Increment 1)

### Distributed-sparse cortical memory (G.20), validated + characterized
- 320-concept (5 cortices × 64) and 160-concept (5 × 32) distributed
  sparse-ensemble memory. Cross-cue recall ~87% (320) / 92.7%
  multi-seed (160); subject→(verb, object) sentence recall ≈80%; all
  measured with pre/post **anti-cheat controls**. Scientific basis:
  Pulvermüller distributed cortical word-webs, Kanerva sparse
  distributed memory, Tonegawa engram tagging.
- **Trustworthiness:** clean confidence separation between known and
  unknown — the system abstains instead of confabulating.
- **No catastrophic forgetting** (continual learning) — multi-seed;
  complementary-learning-systems consolidation (McClelland 1995).
- Failure mode falsification-chained to "dynamical under-recall"; an
  artifact-safe capture-quality fix + a query-time aggregation lever
  shipped (modest, honestly bounded). Several seed-favourable /
  bug-artifact results **retracted forthrightly** when anti-cheat
  controls failed — corrections are part of the record.

### Self-contained generator — Increment 1 (foundation)
- Ported the project's own surrogate-gradient backprop-through-time
  spiking net to `main` (28/28 ported tests pass). On a zero-download
  local English corpus it provably learns **real** sequential
  structure (loss −70.4%, 22% below a shuffled-text control). Honest:
  foundation only, **not yet fluent**, not conversational.
- Design constraint (user): the system must be **entirely
  self-contained at runtime** — no external/local LLM as speaker or
  interpreter, no hand-written response templates in the UX. A local
  model may be used **only** as a training-time distillation teacher.
- Stage-1 template verbalizer reframed honestly as **test scaffold**,
  not a conversational deliverable; its genuine value (grounded
  recall + abstention) carries forward.

## [Unreleased] — 2026-05-11 — Path 3 Phase 3.2 SHIPPED: LLM tool-use stack end-to-end

### Path 3 Phase 3.2 — LLM orchestrator + chat surface
- **LLMMemoryOrchestrator** (`sim/llm_memory_orchestrator.py`, 347 lines)
  drives a tool-use loop between an LLM and BridgeMemory. Three
  OpenAI-compatible tool schemas: `memory_store`, `memory_recall`,
  `memory_speak`. MockLLM ships as the default callable for zero
  external-dependency demos; real LLM swap-in (Phi-3 / Llama 3.2 /
  Qwen2.5) is a one-line constructor change.
- **End-to-end demo runner** (`research/runners/llm_memory_demo.py`)
  proves the full stack: MockLLM → orchestrator → BridgeMemory →
  SimulationBridge → BridgeLineage (atomic persisted state). Scripted
  5-turn chat, JSON output, validated under SIM_BACKEND=numpy.
- **Webapp endpoints** (`webapp/server.py`):
  - `POST /api/llm-chat` — one chat turn, dispatched against a cached
    orchestrator per (lineage, mode) tuple
  - `GET /api/llm-chat/{name}/transcript?mode=...` — conversation log
  - `POST /api/llm-chat/{name}/reset?mode=...` — clear cached state
- **Frontend chat panel** in the Lineages tab: mode selector
  (tier1 / synonym / synonym12 / synonym16), color-coded message log,
  send/reset/Enter, transcript auto-load.
- **BridgeMemory.forget() real-ops** (Phase 3.1 stub → Phase 3.2 real):
  multiplicative weight decay on synapses originating from the key's
  language_input neurons. Decay rate 0.0 = full erase, 0.5 = halve,
  1.0 = no-op. Returns full schema (n_active_neurons, n_synapses_decayed,
  mean_weight_pre/post, estimated_retention). Backend-aware (CuPy +
  NumPy via sim.backend.get_backend).
- **32 new tests** across the LLM stack:
  - 14 in `tests/test_llm_memory_orchestrator.py` (tool schema, MockLLM
    patterns, orchestrator end-to-end, max-iter cap, error propagation)
  - 2 in `tests/test_llm_memory_demo.py` (single-turn + multi-turn
    smoke against a real bridge; SIM_BACKEND=numpy CI-portable)
  - 5 in `tests/test_webapp_server.py` (404 + reset idempotent +
    validation + frontend asset)
  - 5 new in `tests/test_bridge_memory.py` (real-ops decay assertions)
  - 6 updated MockBridge to support region_manager + cp_connections
- **Phase 3.3 design doc** (`docs/plans/2026-05-11-path3-phase3.3-real-llm-design.md`)
  details LLM candidate comparison, ollama adapter sketch, validation
  plan, risk matrix. Estimated ~4 hours implementation work once LLM
  is chosen.
- **Bootstrap `main` synonym lineage** auto-trained in background
  (12,672 neurons, ~12.9M synapses; 1600 events at ~4s/event) for the
  user to chat against on next session.

### Commits in this arc
- `343ea94` feat(path3): Phase 3.2 LLM-memory orchestrator scaffold + MockLLM
- `dbf037f` feat(path3): llm_memory_demo runner — end-to-end Phase 3.2 stack
- `bd1eb13` feat(webapp): POST /api/llm-chat endpoint + transcript / reset siblings
- `3d63dbb` feat(webapp): chat-with-lineage panel in Lineages tab (Phase 3.2 UI)
- `4aab01f` docs: Phase 3.2 LLM stack — CLAUDE.md + findings + user guide
- `3f19a85` docs(path3): Phase 3.3 design — real LLM integration plan
- `31d4a3c` feat(path3): BridgeMemory.forget() real-ops — multiplicative weight decay

Findings: `research/findings/2026-05-11-path3-phase3.2-llm-stack-shipped.md`.

### Continuation (same arc, post-user-checkin)
- **continual-autonomous-work skill** (`0244ee7`) — project-scoped
  addendum to autonomous-runs codifying 5 hard rules against
  self-imposed clean break points; `.claude/skills/` now version-
  controlled via .gitignore exception.
- **BridgeMemory.consolidate() real-ops** (`7867bbf`) — replaces the
  Phase 3.1 stub with SWR sleep replay on hippocampus-enabled
  bridges; detects 'ca3' region, returns degenerate result otherwise.
- **bootstrap_hippo_lineage runner** (`055aa86`) — wraps
  consolidation_trainer to produce a hippocampus-enabled lineage
  (`main_hippo`) usable by consolidate() real-ops.
- **MockLLM forget + consolidate patterns** (`fa1d8b1`) —
  TOOL_SCHEMAS 3 → 5 (memory_forget + memory_consolidate added);
  natural-language patterns recognized ("forget my X", "fully forget",
  "consolidate", "sleep on it", "for N cycles"); dispatch routing.
- **Chat UI chips for forget + consolidate** (`3d69f82`) — example
  chips below input expose the new patterns.
- **Phase 3.4 design** (`9ef3cd0`) — multi-session continuity test
  plan: capacity / retention / interference / forget-rebind across a
  7-calendar-day arc.

Total this arc (post-checkin extension): 8 commits, 4 new tool schemas
exposed, 1 new skill, 1 new design doc, 24/24 tests still pass.

## [Unreleased] — 2026-05-11 — P1-P6 catalog-grounded path to conversational sim

Major realignment after 3 user clarifications:
  1. No external LLM, ever (sim does language itself)
  2. Biology-first workflow (state capability → consult catalog →
     copy biology → test)
  3. Use the research catalog (Kandel 6e PDFs + feature-catalog.md
     at sim-catalog/references/) instead of citing biology from
     memory

Per the realigned plan v3 (commit c075be5), six catalog-grounded
phases (P1-P6) shipped or designed:

### P1 — Hippocampal trisynaptic loop (catalog D.03 + D.12 + D.13)

- `validate_trisynaptic_loop.py` runner shipped. Tests pattern
  separation (DG sparsifies overlapping inputs) and pattern
  completion (CA3 attractor reconstructs from partial cue).
- D.12 separation: 3/3 PASS multi-seed (DG cosine 0.218 from input
  0.800; 58pp orthogonalization).
- D.13 completion (absolute cos > 0.7): 1/3 strict (seed 42 = 0.748;
  seeds 43, 44 ~0.68).
- **Two-concept discrimination test (biology-faithful Marr 1971
  criterion): 3/3 PASS**. Cross-concept tag overlap 0.000-0.120
  (target < 0.3), discrimination margin 0.215-0.432 (target > 0.2).
  Architecture confirmed to support "concepts as distinguishable
  CA3 ensembles."

### P2 — Engram-tagging API (catalog D.14 / roadmap T1.C)

- `bridge.start_engram_recording / commit_engram_tag / stimulate_tag`
  + companions (`clear_tag_drive`, `list_engram_tags`,
  `get_engram_tag_indices`, `delete_engram_tag`).
- Auto-tick in `_run_one_simulation_step` (zero overhead when no
  active recordings).
- Persistence through `save_checkpoint`/`load_checkpoint`.
- 12 unit tests pass + 2 persistence tests skipped pending fuller
  test bridge.

### P3.1 — Concept replay during NREM (catalog D.19 / roadmap T1.B)

- `run_concept_replay_phase(bridge, tag_names, ...)` in
  `consolidation_trainer.py`.
- Selective consolidation: drive each engram-tagged ensemble during
  sleep. STDP at ca3 → ca1 → cortex pathways auto-consolidates.
  Distinct from existing random-CA3-pattern `run_swr_replay_phase`.
- 5 unit tests pass.

### P4.1 — Positional context for episodic binding (catalog D.01 + D.02 + D.11)

- `positional_drive_pattern(position, ...)` in `text_embeddings.py`.
  Deterministic sparse code per position (max 16 positions at
  sparsity=0.1, n_neurons=200).
- `enable_episodic_context` flag adds `ec_context` region + plastic
  `ec_context → dg` pathway. DG receives combined (word, position)
  drive → distinct CA3 ensembles per (word, position) tuple.
- `validate_positional_binding.py` runner shipped. Tests
  (apple, pos_0) vs (apple, pos_2) cosine, etc. Multi-seed
  validation in flight.

### P5 — Ventral semantic stream substrate (catalog G.11 + G.13)

- `enable_ventral_semantic` flag adds:
  - `wernicke` region (200 neurons, lang↔semantic bridge)
  - `semantic_cortex` region (1000 neurons, ATL analog, recurrent)
- 5 plastic pathways: lang_input → wernicke → semantic_cortex
  (comprehension); semantic_cortex → wernicke → language_output
  (production); ca1 → semantic_cortex (consolidation).
- Designs at `docs/plans/2026-05-11-P5-ventral-semantic-stream-design.md`.
- Validation runner pending (next phase of work).

### P6 — Broca's compositional syntax (catalog G.12)

- Design at `docs/plans/2026-05-11-P6-brocas-grammar-design.md`.
- Replaces failed Tier 2.3 PFC verb pool (2026-05-07 PARTIAL).
- Implementation pending.

### Liu 2012-style causal recall test (P2 behavioral validation)

- `validate_causal_recall.py` runner shipped. Encode word→motor via
  hippo, tag CA3 ensemble, then test: stimulating ONLY the CA3 tag
  (no lang_input) reproduces the conditioned motor response.
- Multi-seed validation pending.

### Methodology

- `continual-autonomous-work` skill — Rule 8 codifies the catalog-
  first workflow. Two worked drift examples captured (engineering-
  variants without biology citations; semantic_hub invention without
  checking the catalog).

### Commits in this arc (~30 total)

All catalog-cited. ~1500 LOC new code (validators, runners, tests,
engram API). 5 new design docs (~1100 lines). All committed +
pushed to both remotes.

After this arc, the architecture for "concepts as tagged ensembles
→ consolidated to cortex → composed into sentences" is either
shipped (P1+P2+P3.1+P4.1+P5 substrate) or designed (P5 validation,
P6 full).

## [Unreleased] — 2026-05-03/04 — Permuted-label control debunks 28.5% W→A + autonomous-arc tooling

### Critical correction
- **Permuted-label control test — NEGATIVE.** Across 25 prior text I/O
  eval files (baseline / v2+SWR / H4 / curriculum / dpop / BigLang /
  BigMotor / NoLTD / NoT1 / xcouple / multidec / 200ep), 0/25 had the
  TRUE labeled mapping ranked best of 24 permutations. Best permutations
  consistently score 30-37% (8pp above chance), but the structure is
  randomly oriented per-seed, NOT aligned with task labels. The
  previously-documented 28.5% W→A "validated" result is structure above
  chance, not aligned word→action learning. The W→A binomial p=0.027
  is technically correct but doesn't measure aligned learning — only
  the presence of *some* above-chance structure. Real learning requires
  aligned ratio ≥ 4/6 across seeds. Finding:
  `research/findings/2026-05-03-permuted-label-control-NEGATIVE.md`.
- **Cascade-as-cause hypothesis FALSIFIED.** Minimal-isolation test
  (`text_minimal_isolation.py`, 2026-05-04): minimal architecture
  (`language_input → motor_X` with NO cascade) gives mean 16.7% (BELOW
  chance) at 3 seeds. Cascade was a weak DAMPENER on seed-dependent
  random structure, not its source. Finding:
  `research/findings/2026-05-04-minimal-isolation-INVERSION.md`.

### Added — autonomous-arc tooling
- **`research/runners/permuted_label_check.py`** — definitive
  learning-vs-noise tool: ranks the true labeled mapping against all 24
  symbol permutations. Use as the gate for any text I/O claim.
- **`research/runners/eval_sanity_check.py`** — eval methodology
  validation via hand-built perfect weights. Tests whether the eval
  methodology itself works before debating architectural changes.
- **`research/runners/morning_briefing.py`** — summarizes overnight
  background runs into a single status report.
- **`research/runners/text_minimal_isolation.py`** — minimal architecture
  + biology helpers (`apply_topographic_bias`, `enable_motor_fs`,
  `freeze_stdp`) for testing biology fixes in isolation.
- **`research/runners/unaligned_pattern_analysis.py`** — cross-condition
  structural-bias analyzer; surfaces seed-dependent +3pp motor_E
  cascade bias.
- **`sim/progress.py`** — universal `[PROGRESS] {json}` event format
  consumed by experiment runner and webapp.
- **`research/experiment_runner.py`** — YAML-driven sweep orchestrator
  with built-in configs (biology, minimum_biology, sanity_check,
  b2_sparse_codes, b4_long_training).
- **`research/result_aggregator.py`** — cross-condition aggregation +
  verdict line.
- **Pre-staged decision chain**
  (`research/findings/raw/g11_bg/wait_biology_then_decide.ps1`) —
  auto-launches outcome-A vs outcome-B follow-ups based on biology
  sweep alignment ratio.
- **7-8× speedup stack shipped.** dt=1.0 + parallel-3 GPU sharing +
  `cfg.fast_spike_reset` (cp.where masked-update). Brings 6-seed
  batches from ~6h down to ~45-55 min. Finding:
  `research/findings/2026-05-04-perf-speedup-stack.md`.
- **Autonomous-runs skill** — guides multi-hour autonomous overnight
  arcs with quick-win prioritization, internal trade-off debate, and
  continuation through eval cycles.

### In-flight (2026-05-04)
- Biology-grounded sweep testing topographic prior (Pulvermüller
  2001-2003), PV-FS lateral inhibition between motor pools (Vogels 2011
  / Hofer 2011), and combined. Anti-cheat control runs first with STDP
  frozen — if alignment occurs without learning, prior is too strong.
  Tier-2 fallbacks (`b2_sparse_codes.yaml`, `b4_long_training.yaml`)
  pre-staged.

## [Unreleased] — 2026-04-29 — Catalog remediation pass + Clusters A/C/D scaffolding

### Added — Clusters A (closed BG loop) + C (tonic DA) + D (hippocampus trisynaptic loop)

After the catalog-driven remediation pass (R items below), three opt-in clusters
were scaffolded as the cheat-5 closure attempts continue:

- **Cluster A — Closed BG loop (`2d8be00`).** New `--enable-cluster-a-closed-loop`
  flag adds (a) cortex_X → stn hyperdirect (Nambu 2002, sparse 0.10, weight 3.0)
  and (b) thal_X → cortex_X feedback (action-specific only, density 0.50, weight
  5.0). Both static. Provides the post-synaptic activity / "teaching signal"
  that's been flagged as missing for cross-projection learning. Plan:
  `docs/plans/2026-04-29-cluster-a-closed-bg-loop-design.md`. 4 tests.
  Cheat-5 multi-goal eval n=3 in progress.
- **Cluster C v1 — Tonic DA (`01fddf4`).** New `--enable-tonic-da` flag registers
  a `dopamine` neuromodulator (tonic baseline 0.5, decay_tau 200ms, plasticity_rate
  sensitivity +1.0, from_reward production rule). Bridge reward-modulation block
  switched to use (DA conc - DA baseline) as plasticity signal when DA registered;
  legacy path kept when off. Unlocks B.3 ACh window-gating which was a no-op
  without tonic DA-driven plasticity. Plan: `docs/plans/2026-04-29-cluster-c-tonic-da-design.md`.
  3 tests + smoke pass.
- **Cluster D v1 — Hippocampus trisynaptic loop (`3204c3e`).** New
  `--enable-cluster-d-hippocampus` flag adds 5 regions (ec, dg, dg_fs, ca3, ca1)
  with the canonical EC→DG→CA3→CA1 + EC→CA1 + CA3→CA3 wiring. DG sparsity via
  feedforward inhibition (dg_fs). CA3 plastic recurrent autoassociator
  (internal_density=0.30, plastic_internal=True). Mossy-fiber, perforant-path,
  Schaffer-collateral pathways all wired. CA1 → place_cells provides additional
  drive into existing perception arc when --hippocampus is also on. SWR generator
  (v2) and engram tagging (v3) deferred. Plan:
  `docs/plans/2026-04-29-cluster-d-hippocampus-design.md`. 6 tests + smoke pass
  (50 regions, 1454 neurons, 82,270 synapses).

### Eval status

Three chained background evals (each 6 runs at 1800 steps multi-goal, seeds 42/43/44):
1. Cluster A: baseline (no-A) vs +A (running)
2. Combo: A+C+B.3 vs C-only (queued)
3. Cluster D: D-only vs A+D (queued)

Total 18 data points across 6 conditions. Findings docs to follow.



### Added — Catalog-driven remediation pass (Kandel 6e + supplemental texts)

The textbook catalog session surfaced ~13 sim-level corrections. Plan: `docs/plans/2026-04-29-catalog-remediation-pass.md`.

- **R1.1 per-region E_inh override (`82b3d0d`).** PBR-160 ch 6/11: striatal MSNs use ~−60 mV (depolarizing-but-shunting GABA_A near rest); SNc DA neurons use ~−55 mV (lacks KCC2 chloride exporter). Global default −75 mV is wrong for these regions. Added `BrainRegion.syn_reversal_potential_i_override` field; bridge allocates per-neuron `cp_syn_reversal_potential_i_per_neuron` array; fused conductance kernel accepts scalar OR per-neuron array. Runner overrides on str_D1_*, str_D2_*, dopamine. 8 new tests.
- **R1.2 FSI cross-action wiring (`a1765b0`).** TK-2017 / Tepper-2018: paired recordings show MSN→MSN collaterals are weak (<0.5 mV IPSPs, ~14-25% conn prob); FSI→MSN feedforward is the dominant cross-pool WTA substrate. Rewired build_bg_brain_regions FSI loop: FS_X → MSN_Y for X≠Y only (24 paths, was 32 broadcast). Probe and tests updated.
- **R3.5 sparse + decorrelated cortex→MSN (`1521a9b`).** Bolam-2000 / Kincaid 1998: density 1.0 was anatomically dense. Same-action density 1.0 → 0.20; cross-action density 1.0 → 0.10. Synapse-count smoke drop: 39172 → 31113.
- **R3.10 SNr→SNc disinhibition (`dfa9d15`).** PBR-160 ch 11 Tepper & Lee: SNr collaterals onto SNc; the major in-vivo DA-burst driver. Adds 4 gpi_X → dopamine pathways (density 0.3, weight 2.0).
- **R3.7 GPe PV+/PV− split (`b359bb1`).** Mallet 2008 / Kita 2007: GPe is heterogeneous. New gpe_arky_X (4-neuron arkypallidal subpool); D2 drives both subpools; arky → all str_FS_Y when FSIs enabled (Mallet 2012 stop-signal; 16 broadcast pathways). 3 new tests.
- **R3.11 striosome (patch) / matrix split (`0e041e3`).** PBR-160 ch 9 Deniau: striosomes project to BOTH SNc and SNr; patch/matrix aligns with SNc/SNr at output level. New str_patch_X regions (8 D1-MSN-class neurons each, E_inh −60 mV via R1.1); cortex_X → patch (limbic placeholder); patch → dopamine (canonical SNc); patch → gpi (R3.11 SNr arm). 3 new tests.
- **R3.8 GPi/SNr pacemaker channels (`35f1908`).** PBR-160 ch 9 Deniau: SNr 40-80 Hz tonic pacemaker rests on NaP + SK (firing precision; we use M-current as AHP analogue) + Ih (slow Ca spikes). Tuned HH_GPI_OUTPUT preset: g_NaP_max 0.12 → 0.4; g_h_max 0.05 → 0.15; g_M_max 0.4 → 1.0. Affects HH-mode users; runner default unchanged.
- **R2.3 striatal interneuron taxonomy doc + R3.12 CA3 SWR framing (`8461a03`).** R2.3 (TK-2017 / Tepper-2018): clarified in CLAUDE.md that --enable-striatal-fsis models PV-FSI specifically, one of 8 distinct striatal GABAergic interneuron classes (non-isomorphic to cortex). R3.12 (Bz Cycle 12 / Leinekugel 2002): forward-looking design note — future SWR / replay must place generator inside CA3 intrinsic dynamics with NREM as passive gate, not a sleep-stage scheduler.
- **R3.6 D1/D2 neuropeptide arms (`bdb6452`).** PBR-160 ch 16 McGinty: D1 → dynorphin (KOR plasticity-rate brake) + substance P (NK-1 ACh boost); D2 → enkephalin (DOR plasticity-rate boost). Added new ProductionRule type "from_region_firing" (reads firing-fraction EMA across source_regions) + 3 default neuropeptide configs + `--enable-bg-neuropeptides` CLI flag. 39 neuromod tests pass.
- **R2.4 asymmetric aversive reward magnitude (`23b38fc`).** Schultz98 / Schultz16 / Fiorillo 2013: phasic DA aversive "activations" reflect physical-impact artifacts; underlying valence response is a depression below tonic, smaller magnitude than appetitive activations. Added `cfg.reward_aversive_scale: float = 0.5` (default 0.5 reflects observed ~50% magnitude); negative reward_prediction_error scaled by this factor before applying. Tunable per-experiment. 7 determinism tests pass.
- **R3.9 MSN KIR2/Kv2 (`befc1d0`, design-doc deferral).** PBR-160 ch 6 Wilson: biological MSN bistability rests on KIR2 + Kv-1.2/Kv-2.1 dual currents producing 6× input-resistance peak at -60 mV. Existing Izh `b=-20` approximates KIR2 but doesn't capture the IR-peak feature. Single largest deferral in pass — requires new GPU kernel work (~1-2 days). Catalog ref + design sketch + integration plan documented in `2026-04-29-catalog-remediation-pass.md`.

Final region/synapse counts (flagship smoke + Cluster B): 42 regions (was 30), 758 neurons (was 710), ~32K synapses (was ~41K — sparser cortex per Bolam). All 340 tests pass post-remediation.

## [Unreleased] — 2026-04-28 — Cheat #5 ON HOLD pending biology buildout (reframed) + throughput investigation + webapp polish

### Added
- **Cheat #5 v3 (`--bg-lateral-inhibition`) — GO, permanent default.** MSN cross-pool lateral inhibition: 24 GABAergic pathways between striatal action pools (D1↔D1' and D2↔D2' for X≠Y), `plastic=False`. The missing winner-take-all biology of the BG cascade. 6-seed sum 4.26 ± 0.50 (no regression vs flagship 4.08); P1 (1.91) actually beats P0 (2.35) — readaptation improved. Added to recommended flagship config in CLAUDE.md, README.md, QUICKSTART.md, SCIENCE_ROADMAP.md. Finding: `research/findings/2026-04-28-cheat5-v3-results.md`.
- **Cheat #5 v3.1 (cross-projections layered on v3 lateral inhibition) — NO-GO.** 6-seed sum 8.92 ± 2.44; P1=6.35 (2.5× P0). Phase-2 readaptation breaks even with proper lateral inhibition.
- **Cheat #5 v4 (`--developmental-pretraining`) — NO-GO under single-goal eval.** 3-seed Tier 2 sum 11.34 ± 1.85, P0 4.88, P1 6.46. Originally framed as "closed by design"; reframed later same day after the multi-goal eval correction + option 1 NO-GO + patch-matrix HIGH-VARIANCE PARTIAL signal. Finding: `research/findings/2026-04-28-cheat5-v4-results.md`.
- **Cheat #5 multi-goal eval correction (afternoon).** All prior cheat-5 NO-GO calls used single-goal scenario (one transition at step 300 + 1500 stable steps) — a "static adult" test. Multi-goal (`--goal-schedule multi`, 4 phases × 450 steps, 3 transitions) is the proper test for cross-action coordination. Re-baselined under multi-goal: v3 baseline 7.08 ± 0.12 (n=3).
- **Cheat #5 option 1 (`--enable-structural-pruning`) under multi-goal — NO-GO.** Catastrophic 22.46 ± 4.84 (n=2; seed 42 hung). Structural pruning didn't reshape topology meaningfully.
- **Cheat #5 option 2 (`--cross-projection-density 0.25`) under multi-goal — HIGH-VARIANCE PARTIAL.** 3-seed mean 8.76 ± 2.54, seed 44 actually beat baseline at 5.88. Phase 2 (1,6→1,1) shows topology-luck signal — std 2.09 across seeds vs 0.22-0.46 on others. Sparse cross-projections aren't fundamentally broken, just under-constrained without surrounding biology to consistently select useful pairs.
- **Cheat #5 reframe: ON HOLD pending biology buildout, not closed by design.** Cross-projections need a complete striatal microcircuit (D1/D2 asymmetry, FSIs, TANs), a closed BG loop (thalamo-cortical feedback, hyperdirect pathway), and a properly-structured DA system to behaviorally pay off. Building these out is a multi-month research program organized cluster-by-cluster. Finding: `research/findings/2026-04-28-cheat5-post-v4-reframe.md`. Strategy: `docs/plans/2026-04-28-cheat5-real-options-survey.md`.
- **Reference textbook directory + gitignore.** New `references/textbooks/` directory for source material (Kandel 6e PNS at `references/textbooks/kandel-pns-6e/`). PDFs gitignored (large binaries; only extracted catalogs are committed). Parallel session will build a comprehensive feature catalog from the textbook to drive cluster prioritization.
- **Cluster B.1 (`--enable-d1-d2-asymmetry`) — PARTIAL SIGNAL.** First piece of empirical evidence supporting the cluster-buildout strategy. D1/D2 plasticity asymmetry implemented (per-synapse `cp_d1_d2_sign` array, +1 default, -1 for D2-targeting; multiplicative on the reward-modulated weight update at sim/bridge.py:4309). Biology probe PASS (D1↑/D2↓ under +reward; inverted under -reward; magnitudes match closed-form expectation). Cheat-5 multi-goal: patch-matrix + B.1 = 7.62 ± 1.23 (n=3) vs patch-matrix alone 8.76 ± 2.54. Variance halved, Phase 2 catastrophe eliminated (P2 mean 3.36 → 1.92, std 2.09 → 0.77). Still 7% above v3 baseline 7.08; cheat-5 not fully closed by B.1 alone. Continuing to Cluster B.2 (striatal FSIs) + B.3 (cholinergic interneurons / TANs). 6 unit tests + 1 runner kwarg test + biology probe + 2 cheat-5 batches (n=3 each) shipped. Finding: `research/findings/2026-04-28-cluster-b1-d1d2-asymmetry-results.md`.
- **`--enable-d1-d2-asymmetry` flag, `cp_d1_d2_sign` GPU array, `enable_d1_d2_asymmetry` cfg field.** New plumbing for Cluster B.1.
- **`research/probes/` directory + first probe (D1/D2 asymmetry).** Standalone validation scripts for biological correctness, distinct from pytest-based behavior tests. The probe is reproducible (`python -m research.probes.d1_d2_asymmetry_probe`) and writes JSON output for downstream tooling.
- **Cluster B.2 (`--enable-striatal-fsis`) — MIXED.** Adds 4 `str_FS_{N,E,S,W}` regions (5 PV-positive FSIs each, IZH2007_FS_CORTICAL_INTERNEURON preset, exc_fraction=0.0) + 4 cortex→FS excitatory pathways (weight 30) + 32 FS→MSN broadcast inhibitory pathways (weight 2.0 retuned from 8.0 after over-suppression diagnosis). Plastic=False on all FS pathways (static gating). Biology probe PASS (FSI broadcast inhibition observed: str_D1_N peak rate 36.4 → 23.6 Hz with FSIs engaged at 16 Hz). Cheat-5 multi-goal: patch-matrix+B.1+B.2 = 8.44 ± 0.62 (n=3) — variance keeps halving (1.23 → 0.62) and P1+P2+P3 (4.72) beats v3 baseline (4.89), but Phase 0 broken (3.72 vs 1.83) because FSIs broadcast inhibition before agent commits to winner. Phase-0 issue is architectural — real FSIs have tonic baseline + burst dynamics + high-pass filtering our model lacks. Proceeding to B.3 per unit-cluster strategy. Finding: `research/findings/2026-04-28-cluster-b2-striatal-fsis-results.md`. Probe: `research/probes/striatal_fsi_probe.py`.
- **`--enable-striatal-fsis` flag, `enable_striatal_fsis` kwarg, `cortex_to_str_fs_weight=30`, `str_fs_to_msn_weight=2.0`, `n_striatal_fs_per_action=5` defaults.** New plumbing for Cluster B.2.
- **Cluster B.3 (`--enable-tans`) — NULL on cheat-5, infrastructure shipped (2026-04-28 evening).** Cholinergic TANs implementation: `pause_on_reward` neuromodulator production rule (drives ACh concentration DOWN by `sensitivity*(|reward|-threshold)`); `plasticity_window_gate` target type (`clip(1-conc/baseline, 0, 1)`, multiplicative aggregation); `_default_acetylcholine_config()` helper (baseline=1.0, decay_tau_ms=500, sensitivity=-2.0); `--enable-tans` CLI flag; biology probe at `research/probes/tan_ach_probe.py` (PASS). 47 unit tests pass (15 in `tests/test_tans.py` including step-order regression, 39 in `tests/test_neuromodulators.py`, 6 in `tests/test_d1_d2_asymmetry.py`, 38 in `tests/test_g11_bg_runner_flags.py`). Cheat-5 multi-goal eval at n=3: TAN-on vs TAN-off statistically neutral (B.1+B.2 alone 18.02 vs +TANs 18.59; patch-matrix variants 15.18 vs 14.83). Why no-op: gate fires inside reward-modulation block which is skipped when reward=0 (between rewards); at reward steps `pause_on_reward` drops ACh → gate ≈ 1 (no suppression). Real TAN function requires tonic DA-driven plasticity for ACh to gate; our model has only phasic DA. Real win: bridge step-order bug fix — `manager.step()` now runs BEFORE reward modulation, correcting one-step lag for fast-dynamics modulators (regression test `test_single_pulse_reward_fires_plasticity_within_step` catches the bug pattern). Finding: `research/findings/2026-04-28-cluster-b3-tans-results.md`.
- **Bridge step-order fix (`59dc1fc`).** `self.neuromodulator_manager.step(self)` and `compute_plasticity_gate_values()` propagation moved from after the reward-modulation block to before it. Pre-fix: gate was read from previous step's NM concentration → one-step lag. Post-fix: this step's reward signal drives this step's NM concentration changes (e.g., `pause_on_reward` → ACh pause → gate opens) AT THE SAME STEP. Required for fast-dynamics gates; harmless for slow-dynamics modulators (DA tonic, NE) which read at synaptic-conductance/excitability sections (still one-step-lagged). New regression test catches the pre-fix bug pattern; existing test_bridge_blocks_reward_weight_updates_when_ach_at_baseline updated to use a no-production-rule ACh config so the gate mechanic is tested in isolation.
- **Methodology finding: multi-goal benchmark regressed at seed 42** (2026-04-28 evening). v3 baseline 7.08 → 12.05 in current code; B.1+B.2 9.50 → 22.03; patch-matrix+B.1+B.2 8.44 → 18.87. Phase 3 (after 3 transitions) shows the dominant regression. Bisect at pre-B.3 commit `714bc29` reproduces 21.22 for B.1+B.2 — predates B.3 changes. Future cluster work should use fresh current-code baselines, not historical numbers.
- **`--developmental-pretraining`, `--pretraining-n-goals`, `--pretraining-steps-per-goal` flags + `_run_pretraining_phase` helper.** v4 implementation (8 commits, TDD, 30/30 pretraining/regression tests pass). Kept opt-in for future experiments (e.g., pretraining other pathways with structural plasticity).
- **GPU throughput investigation** — concurrency sweep, MPS daemon ruled out (Linux-only on RTX 3090/Windows host), motor-counting code fix REVERTED (no measurable improvement, n=1 showed -15%). Concurrency knee at 4-6 (4× hits 76% of 10× aggregate at 1.7× per-run speed). Finding: `research/findings/2026-04-28-throughput-investigation.md`.
- **Webapp polish (UX pass)** — live mode toggle, top-bar layout, font consistency via `--font-sans`/`--font-mono` CSS vars, collapsible HUDs, runs page no-flicker, goal-change dots on live chart, no-cache static asset serving, Windows DETACHED_PROCESS subprocess detach so closing dashboard doesn't kill running launcher subprocesses.

### Changed
- **Webapp default `--progress-print-interval`** changed from `1` (always-on for live-viz) to `20` for non-interactive presets. `interactive_*` presets keep `1` for live attach. Reduces stdout pressure during background batches.
- **Recommended flagship config now includes `--bg-lateral-inhibition`** by default. Backward compatible — flag is opt-in, off by default at the runner level, but the recommended/documented recipe ships with it on.

### Negative results (kept opt-in)
- **Cheat #5 v1 (curriculum-staged cross-projections)** — 3-seed mean 10.87. Plasticity gate freezes weight updates but not synaptic transmission; non-zero cross-projection weights disrupted BG disinhibition from step 0. Finding: `research/findings/2026-04-28-cheat5-v1-NEGATIVE.md`.
- **Cheat #5 v2 (zero-init cross-projections)** — 3-seed mean 7.89. Zero-init fixed the structural-damage failure mode (P0=2.49, intact), but exposed a learning-dynamics failure mode (P1=5.40): thaw-time STDP corrupts the converged policy. Diagnosis pointed at missing MSN lateral inhibition → led to v3. Finding: `research/findings/2026-04-28-cheat5-v2-NEGATIVE.md`.

## [Unreleased] — 2026-04-27/28 — NEW BEST: 4 of 5 cheats closed + Phase C + Item 1

### Added
- **🎉🎉🎉🎉 NEW BEST CONFIGURATION (2026-04-27/28 overnight) — 4 of 5 cheats closed, biology-grounded BEATS cheats-allowed**
  - **Sensed reward** (cheat #4 closed): reward computed from beacon-intensity gradient instead of ground-truth Manhattan distance. Real animals don't have access to ground-truth distances — they sense whether a cue is getting stronger or weaker. `--sensed-reward` flag.
  - **Result: sum 4.08, 6/6 seeds beat baseline 5.88, p=0.00045, 30.6% improvement.** Biology-grounded version (4.08) is *better* than cheats-allowed (4.41) — closing perception/reward cheats actually *helps* learning quality.
  - **Cheat #5 (BG cross-projections) tested — NEGATIVE.** Learnable cortex_X → str_D1_Y all-to-all broke phase-1 readaptation (3-seed avg 8.40, much worse). Phase-0 cortex_N/E activations reinforce cross-projections to all D1 pools, locking in motor bias the agent can't unlearn. Kept opt-in (`--bg-cross-projections`) for future experiments.
  - **Final cheats inventory:** 4 of 5 perception/reward cheats now closed. Only structural BG connectivity remains, plus minor items (discrete N/E/S/W actions, discrete time steps).
  - Finding: `research/findings/2026-04-27-NEW-BEST-4cheats-closed.md`
  - Recipe: `g11_bg_runner.py --moving-goal --hippocampus --learned-perception --pfc --beacon-perception --beacon-replaces-goal --cue-reflex --cue-reflex-replaces-heuristic --landmarks --landmarks-replace-place --sensed-reward --adaptive-da --adaptive-da-ema-decay-negative 0.7 --curriculum --curriculum-warmup-steps 600 --seed N --n-steps 1800`

- **🎉🎉🎉 Item 1 PERCEPTION ARC COMPLETE (2026-04-27 night)** — agent navigates from PERCEIVED sensory information; ALL major coordinate cheats closed
  - **Stage 1: Goal-beacon perception** — replaces direct (gx, gy) goal cell access with 8 directional sensors detecting beacon strength × cosine alignment. Plastic beacon → goal_cells pathway (curriculum-gated). 6-seed: 5/6 beat baseline (5.36 vs 5.88, p=0.34).
  - **Stage 3: Cue-following reflex** — replaces the heuristic with non-plastic reflex computing cortex drive from direction-normalized beacon sensor pattern. Models innate phototaxis-like wiring. Combined with Stage 1: 6/6 seeds beat baseline (4.77 vs 5.88, **p=0.00188**, 18.9% improvement).
  - **Stage 2: Landmark-based place cell self-organization** — fixed-position landmark (default at grid center) with 8 directional sensors + plastic landmark → place_cells pathway. Replaces direct (x, y) place cell access. Combined with Stage 1+3: **6/6 seeds beat baseline (4.56 vs 5.88, p=0.00819, 22.4% improvement)**.
  - **Final state**: agent has NO direct (gx, gy) AND NO direct (x, y) AND NO heuristic. Only 3% behind cheats-allowed best (4.41) — closing all coordinate cheats costs almost nothing.
  - Findings: `research/findings/2026-04-27-FULL-PERCEPTION-ARC-COMPLETE.md`, `2026-04-27-stage3-full-perception-BREAKTHROUGH.md`, `2026-04-27-stage1-beacon-perception.md`, `2026-04-27-perception-cheats-investigation.md`
  - Plan: `docs/plans/2026-04-27-perception-arc-plan.md` (executed in single session)

- **PFC working memory region (Item 3, 2026-04-27)** — recurrent prefrontal cortex for working memory dynamics. 60 neurons, internal_density=0.2, plastic recurrent. Pathways: `goal_cells → PFC → cortex_{N,E,S,W}`. 6-seed: 5/6 beat baseline (4.41 vs 5.88, p=0.018, 25% improvement).

- **Per-pathway plasticity gating (Phase C, 2026-04-27)** — biologically-grounded staged plasticity
  - `RegionPathway.plasticity_gate: str | None` field tags pathways
  - `cp_plasticity_gain` per-synapse array gates STDP/eligibility/Hebbian/synaptic-scaling
  - Bridge methods: `set_plasticity_gate(name, value)`, `get_plasticity_gate_value()`, `list_plasticity_gates()`
  - **NM-driven gates**: `target_type="plasticity_gate"` with `scope="gate:<name>"` lets NM concentrations drive gates
  - 8 unit tests for gating semantics; 1 test for NM-driven gates
  - Closed the 7-NEGATIVE plastic-input-layer arc that ran 2026-04-26

- **Real curriculum learning** — phase 1 corticostriatal plastic + input layers frozen; phase 2 cortex frozen + input layers plastic. Configurable warmup steps, smooth ramping, partial-freeze gain. (Gate renamed from `cortex_to_d1` to `corticostriatal` 2026-04-29; old name aliased with deprecation warning for one release cycle.)

- **Sleep-replay infrastructure** — NREM trajectory replay (logged successful (place, goal) tuples) + REM random replay alternation. Mechanism works; current task structure doesn't reward consolidation.

- **Spatial scaling** — `--grid-size`, `--n-hippocampus-per-layer` for arbitrary grid sizes. Architecture scales to 16×16; recipe needs re-tuning for larger grids.

- **g11_bg_runner CLI growth** — many opt-in flags: `--curriculum`, `--curriculum-warmup-steps`, `--curriculum-ramp-steps`, `--curriculum-phase2-cortex-gain`, `--pfc`, `--n-pfc`, `--beacon-perception`, `--beacon-replaces-goal`, `--cue-reflex`, `--cue-reflex-replaces-heuristic`, `--landmarks`, `--landmarks-replace-place`, `--sleep-replay-after-step`, `--sleep-nrem-rem-alternate`, `--goal-silence-after-step` (PFC delayed-response test), `--heuristic-decay-after-step` (heuristic-off validation)

- **TROUBLESHOOTING doc** (`research/runners/TROUBLESHOOTING.md`) — gotchas accumulated across sessions

### Changed
- **Recommended flagship config (current best, biology-grounded BEATS cheats-allowed)**:
  - Flagship: `--hippocampus --learned-perception --pfc --beacon-perception --beacon-replaces-goal --cue-reflex --cue-reflex-replaces-heuristic --landmarks --landmarks-replace-place --sensed-reward --adaptive-da --adaptive-da-ema-decay-negative 0.7 --curriculum --curriculum-warmup-steps 600` (**4.08, p=0.00045, 30.6% over baseline**)
  - Cheats-allowed (older): same minus beacon/reflex/landmarks/sensed-reward flags (4.41, p=0.018)
- QUICKSTART.md added — 60-second getting-started for new users
- CLAUDE.md, SCIENCE_ROADMAP.md, INDEX.md, README.md all reflect the new state

## [Unreleased] — 2026-04-25 — Phase A presets + Phase B BG action selection

### Added
- **Phase B: BG-style action selection cascade** — silent-motor trap resolved
  - `research/runners/g11_bg_runner.py` builds 30-region cascade: cortex → str_D1/str_D2 → GPi/GPe → STN → thalamus → motor with disinhibition gating
  - 3-seed acid test: phase 1 finalQ 1.76 avg vs G9 baseline 6.74 (74% improvement, agent stays at Manhattan distance ~1.7 from goal vs random walk's ~5.5)
  - Per-action populations replace shared reservoir + argmax — eliminates the dominant-motor bias that defeated 7 prior runner-side variants (V1–V7)
  - Findings: `research/findings/2026-04-25-phase-b-acid-test-real-win.md`, `2026-04-25-phase-b-cascade-stability-fix.md`, `2026-04-25-phase-b-honest-correction.md`
- **Phase A: comprehensive preset audit + retuning** (HH + Izh + AdEx — 30 working biological presets)
  - 4 new HH BG cell types: `HH_STRIATAL_MSN_D1`, `HH_STRIATAL_MSN_D2`, `HH_STRIATAL_TAN`, `HH_GPI_OUTPUT`
  - 8 new IZH2007 brain-region presets: `IZH2007_STRIATAL_MSN`, `IZH2007_STRIATAL_MSN_D1/D2`, `IZH2007_STRIATAL_TAN`, `IZH2007_GPE_PACEMAKER`, `IZH2007_GPI_OUTPUT`, `IZH2007_STN_BURST`, `IZH2007_THALAMIC_RELAY`, `IZH2007_THALAMIC_RETICULAR`, `IZH2007_HIPPO_PYRAMIDAL`, `IZH2007_DOPAMINE`
  - Full AdEx preset library (`DefaultAdExParamsManager`): RS, FS, IB, CH, LTS, MSN, DOPAMINE — all 7 fire at 37°C with biological rates
  - Per-region neuron type override on `BrainRegion`: `izh_neuron_type`, `hh_neuron_type`, `adex_neuron_type` (independent of global default)
  - Findings: `2026-04-25-hh-preset-audit.md`, `2026-04-25-izh-preset-audit.md`, `2026-04-25-hh-presets-after-q10-fix.md`
- **Per-gate Q10 temperature scaling** for Hodgkin–Huxley (`hh_q10_m=3.0`, `hh_q10_h=hh_q10_n=1.5`)
  - Replaces uniform Q10=3 that over-compressed gating dynamics at 37°C
  - HH model now produces action potentials at body temperature
  - Finding: `2026-04-25-hh-temperature-bug.md`

### Fixed
- **STDP soft-bound w_max collapse** — when synapse `weight_mean > stdp_w_max`, every "LTP" event is strongly negative (`Δw = A_plus * (w_max - w) * exp(...)`), collapsing weights to w_max within milliseconds. Set `cfg.stdp_w_max` above design weights (e.g. cortex→D1 `weight_mean=25` needs `stdp_w_max=30`). Documented in CLAUDE.md and Phase B findings; runners now set it explicitly.
- **n_cortex saturation in BG cascade** — over-driving D1 above ~150 Hz puts MSNs into refractory dominance and breaks D1→GPi inhibition. Probes must use the same `n_cortex` value as deployment. (`research/runners/g11_bg_runner.py` now uses `n_cortex=100` matching the static probe.)
- **Izhikevich preset wasn't applied** — bridge always trait-split via `traits % num_variants`; now opt-in only when `cfg.num_traits > 1`. `cfg.default_neuron_type_izh` now respected when single-type is intended.
- **AdEx presets all behaved identically** — bridge wasn't loading preset params into `cfg.adex_*` fields. Now overlays preset onto config before initialization.
- **GPE/STN didn't fire** — `g_NaP=0.8` was 5–10× too high for these cell types. Retuned in `HH_GPE_PACEMAKER` and `HH_STN_BURST` (commit `9f4c3f7`).

## [Unreleased] — 2026-04-24 — Brain-region framework + neuromodulator subsystem + Route C performance

### Added
- **Brain-region framework** (Session E.2, opt-in)
  - `sim/regions.py` — `BrainRegion`, `RegionPathway`, `RegionManager` dataclasses
  - Declarative multi-region simulations (PFC + Motor + Hippocampus + Striatum on one bridge)
  - Each region owns a contiguous neuron-index slice with its own internal connectivity
  - Cross-region pathways declared with density, weight_mean, plasticity flag, and optional neuromodulator gates
  - `cfg.enable_brain_region_framework=True` opt-in; default OFF for backward compatibility
  - Bridge integration: regions allocated before neuron arrays (auto-sets `num_neurons`); wiring fed through `inject_explicit_wiring()` replacing legacy motif/WS/spatial paths
  - Composes with neuromodulator subsystem — regions auto-register as NM groups
  - Plan: `docs/plans/2026-04-24-brain-region-framework.md`; tests: `tests/test_regions.py`
- **Neuromodulator subsystem** (Session E.1, opt-in)
  - `sim/neuromodulators.py` — `NeuromodulatorConfig`, `ModulatorTarget`, `ProductionRule`, `NeuromodulatorManager`
  - Declarative concentration dynamics for DA / NE / 5-HT / etc.
  - Built-in target types: `synaptic_gain`, `plasticity_rate`, `excitability_drive`
  - Built-in production rules: `manual`, `from_reward`, `from_error_persistence`
  - Replaces ad-hoc `current_reward_signal` and shelved `cp_synaptic_gain_modulator`
  - `cfg.enable_neuromodulator_subsystem=True` opt-in; default OFF
  - Plan: `docs/plans/2026-04-24-neuromodulator-subsystem.md`; tests: `tests/test_neuromodulators.py`
  - Finding: `research/findings/2026-04-24-session-e1-neuromodulator-subsystem.md` (framework GO, NE params NO-GO on silent-motor)
- **Route C: 101× synapse-update throughput** at 1.2× wall time (bigger networks performance)
  - Finding: `research/findings/2026-04-24-route-b-profile.md`
- **Module split** — extracted `sim/`, `viz/`, `ui/`, `experiment/` packages from monolithic `neural-simulator.py`
  - `sim/__init__.py` exposes public API (`SimulationBridge`, configs, `NeuronModel`, `NeuronType`)
  - `neural-simulator.py` reduced from ~12K lines to ~2.2K (now just GUI host)

## [Unreleased] — 2026-04-20/21 — Research-gate runner framework (G1–G6)

### Added
- **Research-gate runner framework** (`research/runners/`)
  - 16 headless runners (g1..g11) each invocable via `python -m research.runners.gN_runner`
  - Each writes raw data to `research/findings/raw/gN/` and a markdown finding to `research/findings/`
  - Negative results documented as findings, not failures
- **G1: encoder-decoder roundtrip** — GO (v3, 71.3% test acc, 3 seeds, threshold 55%)
  - `research/datasets/tiny_patterns.py` — K=4 Poisson-rate synthetic dataset
  - `RATE_VECTOR_POISSON` stimulus pattern type
  - `SimulationBridge.inject_explicit_wiring()` — injectable explicit CSR connectivity
  - Three runner variants explored; v3 (264-neuron reservoir + external LogReg) passes
  - Finding: `2026-04-20-g1.md`
- **G2: STDP local learning** — NO-GO (no epoch-over-epoch improvement on target task) — `2026-04-20-g2.md`
- **G3: persistence/checkpointing** — GO — `2026-04-20-g3.md`
- **G5: sensorimotor signed perceptron** — GO (v3 with LR decay, 3/3 seeds pass) — `2026-04-21-g5v3.md`, `2026-04-21-signed-eligibility-branch.md`
- **G6: 2D gridworld** — PARTIAL (gate metric needs redesign — agent converges too fast) — `2026-04-21-g6.md`, `2026-04-21-g7.md` (proposed metric replacements)

## [Unreleased] — Earlier (2026-04-06 baseline)

### Added
- **G1 pipeline GO** - First end-to-end dataset → encoder → sim → decoder → loss round-trip
  - `research/datasets/tiny_patterns.py` + canonical `.npz` - K=4 Poisson-rate synthetic dataset
  - `RATE_VECTOR_POISSON` stimulus pattern - per-neuron Poisson rate encoding
  - `SimulationBridge.inject_explicit_wiring()` - injectable explicit CSR connectivity for research networks
  - Three runners explored: v1 teacher-forced STDP (NO-GO), v2 external perceptron (NO-GO), v3 reservoir + external LogReg (**GO** - mean 71.3% test acc across 3 seeds, threshold 55%)
  - v1/v2 post-mortem: sim's default `propagation_strength=0.05` is calibrated for ~1000 converging synapses per neuron; the 68-neuron tiny architecture needs non-default params. v3 uses a 264-neuron reservoir in the sim's calibrated regime.
  - Full findings in `research/findings/2026-04-20-g1.md`

- **Profile System** - Biologically accurate brain region presets and UI integration
  - 9 brain region profile JSONs with realistic neuron models and connectivity: Cortex L2/3, Cortex L4, Hippocampus CA1, Hippocampus CA3, Thalamus TC-TRN, Basal Ganglia Striatum, Basal Ganglia STN-GPe, Cerebellar Cortex, Spinal Cord
  - Quick Demo profile for rapid testing
  - Full profile dropdown menu in UI that auto-populates from `simulation_profiles/*.json` files
  - Refresh button to reload profiles from disk without restarting

- **Plasticity Parameters in UI** - STDP, reward modulation, and structural plasticity
  - 28 new fields added to SimulationConfiguration for complete plasticity roundtrip (save/load)
  - Support for STDP timing windows, reward modulation learning rates, and structural synapse thresholds
  - Full persistence in simulation profiles and checkpoint files

- **Per-Connection-Type STP** - Biologically realistic short-term plasticity heterogeneity
  - New `enable_per_type_stp` parameter and per-type arrays `stp_U_per_type`, `stp_tau_d_per_type`, `stp_tau_f_per_type`
  - Each is indexed by connection type [E->E, E->I, I->E, I->I]
  - Different brain regions now use experimentally validated STP profiles per connection type
  - UI table exposes all 12 parameters (4 connection types × 3 STP variables)

- **Activity-Dependent Structural Synaptogenesis** - Cline & Haas 2008 model
  - New `struct_plast_activity_bias` parameter (0.0-1.0, default 0.5)
  - Biases new synapse formation toward co-active neuron pairs using activity EMA
  - 0 = random synapse formation; 1 = fully activity-driven (Hebbian structuring)

- **COO Cache Invalidation** - Fixed stale data handling in GPU memory
  - Cache invalidation in `clear_simulation_state_and_gpu_memory()` prevents stale sparse matrix data across reinitializations

### Fixed
- **STP/Connection Shape Mismatch at Scale** - CSR matrix deduplication bug
  - Fixed shape mismatch occurring at 100K+ neurons caused by CSR matrix addition deduplicating overlapping (pre,post) pairs
  - Now uses `cp_connections.nnz` as authoritative size instead of stale shape values
  - Structural plasticity synapse count now properly synced after CSR addition

- **Synaptic Scaling Crash** - Stale COO cache surviving reinitialization
  - COO cache no longer persists across simulation reinitializations, preventing crashes when synaptic scaling is active

- **Unicode Handling on Windows** - JSON I/O encoding issue
  - UnicodeDecodeError on Windows (cp1252) when loading profile JSONs with Unicode characters (em dashes, etc.)
  - All JSON I/O now uses UTF-8 encoding explicitly

- **Em Dash Rendering** - DearPyGui font limitation
  - Em dashes rendered as question marks in DearPyGUI default font
  - Replaced with regular hyphens in all UI text and profile names

### Changed
- **Hodgkin-Huxley Numerical Stability** - Automatic time step adjustment
  - dt automatically reduces to 0.05ms when switching to HH model for improved numerical stability
  - dt automatically restores to 0.5ms when switching away from HH model
  - Prevents instabilities in voltage-gated kinetics at larger time steps

- **Homeostatic Plasticity Timescale** - Biologically realistic adaptation
  - EMA alpha reduced from 0.01 (tau ~100ms) to 0.0002 (tau ~5s at dt=1ms)
  - Threshold adapt rate reduced from 0.015 to 0.0005
  - Homeostatic mechanisms now operate on seconds-to-minutes timescale, matching experimental observations

- **Inhibitory Reversal Potential** - Corrected Nernst equilibrium
  - E_inh changed from -70mV to -75mV (matches Cl- Nernst potential at 37°C)
  - Inhibitory propagation strength scaled by 0.7 to compensate for increased driving force
  - Improves accuracy of GABAergic synaptic transmission

- **.gitignore** - Profile tracking and auto-tuning separation
  - Now tracks `simulation_profiles/*.json` to include biologically accurate presets in repository
  - Excludes `auto_tuned_overrides.json` to prevent auto-tuned parameters overwriting checked-in profiles

- **Profile Files** - Superseded files removed
  - Removed 7 old profile files replaced by new standardized brain region profiles

- **System Logs Panel** - Comprehensive log viewing and management
  - Real-time display of all console output within the GUI
  - Auto-scroll functionality using DearPyGUI's `tracked` parameter
  - Search functionality with previous/next navigation through matches
  - Export logs to timestamped text files
  - Clear logs functionality
  - Thread-safe `LogCapture` class for zero-overhead console mirroring
  
- **Performance Test Controls**
  - Stop button for halting running benchmarks and auto-tuning mid-execution
  - Proper state tracking to preserve existing result files
  - Informative logging showing which test type was stopped
  - Located above "Reload Auto-Tuned Overrides" button in GUI

### Changed
- **VRAM Utilization for Initialization** - Increased chunking from 40% to 70% of free VRAM
  - ~2x faster initialization for networks with 50K+ neurons
  - Example: With 18GB free VRAM, now uses 12.6GB instead of 7.2GB for chunking
  - Maintains 30% safety margin for stability
  
- **GUI Layout Improvements**
  - Auto-tuning button now stretches to fill available width (width=-80)
  - Better space utilization when window is resized wider
  - "Quick" checkbox properly positioned at right edge

### Fixed
- Auto-scroll in System Logs now works correctly using DPG best practices
  - Replaced manual scroll manipulation with `tracked=True` and `track_offset=1.0`
  - Dynamic height adjustment based on text size for proper scrolling
  - Toggle auto-scroll on/off via checkbox callback
  
- Performance test stop functionality prevents corrupted result files
  - Benchmark and auto-tuning only save results at completion
  - Stopping mid-run preserves any previously existing result files

### Technical Details
- Log capture uses thread-safe deque with 5000-line rolling buffer
- System logs display widget uses `child_window` with `input_text` for proper scrolling
- Auto-scroll implementation follows official DearPyGUI documentation patterns
- Stop flags (`performance_test_stop_flag`, `performance_test_running_type`) properly managed in try/finally blocks

## [Previous Versions]

See git commit history for details on earlier changes.
