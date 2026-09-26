# Neural Simulator

Neural Simulator is an open research project attempting to grow an artificial
brain from spiking neurons and synapses. The long-term goal is one integrated
brain that starts small, develops through embodied and social experience, and
eventually holds a genuinely self-directed conversation: reasoning to its own
conclusions from its own perceptions, memories, needs, functional affect, and
uncertainty, rather than retrieving stored text or predicting plausible-sounding
text. The project's working hypothesis, sometimes called the emergentist bet, is
that consciousness-relevant function emerges once a human brain's faculties and
behavior are emulated completely and faithfully enough. The project does not
claim to have reached that bar, or to have produced anything conscious, felt, or
sentient; it is testing the hypothesis, not asserting its outcome (see
[Boundaries And License](#boundaries-and-license) below).

This is not a finished mind or a chatbot wrapped in neuroscience terminology.
It is a simulator, a collection of tested brain mechanisms, and a research
program working to join those mechanisms into a continuously developing whole.

## What The Project Is Building

The target system has five defining properties:

- **Fully spiking:** cognition is carried by simulated neurons, synapses, and
  their changing activity.
- **One shared substrate:** specialized regions are welcome, but perception,
  memory, value, emotion, thought, and language must interact within one brain.
- **Grounded development:** the brain learns by sensing a world, acting in it,
  experiencing consequences, and interacting with people. Text can be part of
  that experience, but isolated text prediction is not the goal.
- **Causal inner state:** needs, emotions, curiosity, confidence, and memory must
  change what the brain attends to, learns, says, and does. They are not labels
  added to an answer after it has been generated.
- **Individual ownership:** early brains should run locally and grow only as
  learning requires it. The engineering target is high-end consumer hardware,
  with sparse and local computation that could later suit neuromorphic hardware.

The intended developmental loop is:

```text
world and people -> sensation -> internal state -> action or speech
                 -> consequence -> learning -> changed future behavior
```

Ordinary host code may provide the outside world, body interface, storage,
visualization, and scientific instruments. It must not remain a substitute for
a cognitive function that the simulated brain is meant to perform.

Read the [project charter](docs/PROJECT-CHARTER.md) for the commitments behind
this work and the [technical overview](docs/TECHNICAL-OVERVIEW.md) for the
architecture, current evidence, and major gaps.

## Current State

The simulation foundation is substantial. It supports central processing unit
and NVIDIA graphics processing unit execution, sparse spiking networks,
multiple neuron models, declared brain regions and pathways, several forms of
local plasticity, neuromodulation, homeostasis, structural change,
checkpointing, experiment runners, and visualization.

The cognitive system is much earlier:

| Area | Honest status |
|---|---|
| Shared brain | Multiple regions and pathways can occupy one simulation and update together. The recall composer now runs on the spiking one-brain store by default, with a hippocampus-style sharded lookup (2026-09-05) that keeps that same recall roughly 400x faster with no loss of recall or no-confabulation guarantee. As of 2026-09-05 four core thinking organs — surprise, the forward world-model, self-monitoring, and phrasing — share one literal neural pool by default (the "one-brain" merge), validated answer-preserving across 6 seeds — a genuine one-substrate step, though still co-residency (their organ routing is host Python), not itself a scaffold retirement. As of the 2026-09-16 ledger head the tracked production-integration ledger lists 69 faculties in total, 30 of them genuinely spiking and on by default, and 5 with their host scaffold fully retired (`scaffold_retired: 5`: one-brain-substrate, spiking-novelty-habituation, spiking-anaphor-detection, spiking-qroute-selection, and selective-attention-biased-competition, each verified byte-identical against main) — the ledger's own note states that most of the live chat's remaining load-bearing cognition is still host, Python, or NumPy, not the shared spiking substrate. As of 2026-09-19 the project's headline integration metric is no longer "percent of scaffold retired" but a lesion-verified **load-bearing fraction** of the full production conversation, always reported as a pair (owner standard), never a single cherry-picked number: a 2026-09-25 re-scored 6-seed battery reads mean 0.603 (robust core 15 of 26, union 16) with the thin measurement probes the shipped default actually uses, and mean 0.949 (robust core 24 of 26, union 25, SD 0.018 for both) with adequate measurement-only probe fixes not yet shipped — see [`research/findings/2026-09-25-production-default-battery-B2a-rescored-PASS.md`](research/findings/2026-09-25-production-default-battery-B2a-rescored-PASS.md). |
| Grounded action | Navigation, action selection, reward learning, perception, memory, and replay have working research results, usually in constrained tasks. |
| Grounded communication | A six-seed result joins learned visual association, hunger, a request-or-silence choice, and a world consequence. A newer six-seed result learns a tiny external vocal convention with two intents and two referents and succeeds on both untrained cross-combinations. It still uses injected motor exploration, fixed neural channels, and a host listener; it is preverbal learning, not natural language. |
| Conversational integration (current frontier) | Validated faculties are wired into the live chat one dependency at a time and kept only when a causal lesion changes the reply. Default-on and spiking: a claim-level no-confabulation moat that drops ungrounded content (an honest "I don't know" rather than a made-up answer); comprehension monitoring with targeted repair ("I didn't follow that", naming the unresolved role); a why / what-if causal forward model; scalar-implicature pragmatics ("some" reads as "some but not all"); a non-contradiction belief gate with in-place reconsolidation on a prediction error; self-initiated remarks on idle turns; open-ended generation that volunteers a novel, grounded, plausibility-gated, moat-checked proposition flagged as a guess; and a spiking global-workspace ignition bus that now authors the combine-and-decide step host Python used to do, abstains on a genuine multi-answer conflict, and runs re-entrant deliberation whose depth the substrate discovers itself. Honest inner-state read-outs (affect, a structural self-model, a graded certainty band) are functional statements, never felt states. As of 2026-08-19 three of the brain's own signals were made load-bearing on the conversation itself: its mood colors phrasing, its thought-swap decision steers which topic the turn engages, and its dopamine mode sets how engaged the reply is. A 2026-08-26 batch flipped ten more validated faculties to on-by-default, synced to `main`: empathy for another person's situation, an honest self-authorship marker on the brain's own guesses, attention as a competitive race between held discourse referents, a spiking vision-identity pipeline, a dopamine-trained value-driven choice between tied recalls, offline sleep-replay consolidation, a bounded-recall spiking mouth (a spiking Broca-like renderer that speaks single-role transitive-SVO recall sentences directly, still narrow in scope — see the separate, broader own-voice fluency arc under Language), a neurally-mediated global-workspace "stop" on topic change, silent (no ongoing firing) working memory, and a basal-ganglia-style speak-or-stay-silent selector. None of these ten retired a host scaffold by itself; as of 2026-09-16, 5 of the 69 tracked faculties overall have had their host scaffold fully retired (see the Shared brain row above). Most named host scaffolds remain, and multi-fact continual learning is still open. |
| Language | The repository contains simple question answering and bounded spiking language experiments, plus a live conversational loop that composes honest grounded replies and honest silences. On an open-ended prompt the brain now volunteers a novel, grounded, plausibility-gated, moat-verified proposition flagged as a guess, and multi-sentence replies are planned by a neural discourse planner. As of 2026-09-04, the brain's own from-scratch spiking "mouth" beat a simple word-pair baseline on simple text at a deployable size, but was not fluent enough on broad, arbitrary-topic conversation. As of 2026-09-19 that "retire the transformer mouth" arc was closed as falsified and the external Qwen2.5-0.5B transformer is now treated as a **permanent** articulation mouth rather than a scaffold to be retired; the tracked number-one question is instead whether the spiking brain, not Qwen, is provably load-bearing on what gets said (see the load-bearing fraction under Shared Brain above). Fluent, open-ended conversation grounded in ongoing life is not yet achieved. |
| Emotion and drives | Reward, value, neuromodulator signals, and curiosity mechanisms exist and several now act in the live chat. The brain reads its own live spiking mood; it shapes forthcomingness (how many gate-matched facts the turn volunteers) and sets the manner instruction given to the mouth, rather than inserting a feeling word into the reply — a prepended marker word (e.g. "Wonderful!") was retired from default output on 2026-09-25 per owner directive that affect should shape *how* the brain speaks, not insert marker words; mood is still computed and recorded, and the forthcomingness/manner paths are unchanged. A six-seed result makes that mood caused by a simulated body-state read by dedicated interoceptive neurons, and cutting those synapses makes the feeling stop tracking the body. Affect is load-bearing on the surface (the difference vanishes when the embodiment pathway is lesioned), the self-selected dopamine mode sets how engaged the reply is, and a curiosity mechanism appends an honest follow-up question on an abstain about a novel topic. A rich emotional system that develops broadly through experience remains open. |
| Memory and self-monitoring | Several memory and self-monitoring faculties now run on by default and spiking in the live chat: episodic recall of the current conversation (an honest "I recall / I don't recall discussing X" from a hippocampal pattern-completion); prospective memory that holds a deferred "remind me to X when Y" across turns and fires it on the cue; working memory holding two or more discourse referents on a bump attractor, plus an anaphora store and a discourse-event register ("who was doing it before?"); a metacognition confidence hedge whose confidence-to-correctness mapping is now learned by a local three-factor Hebbian rule (six-seed GO, type-2 AUC 0.825); an expectation-violation surprise notice; and a queryable forward world-model. The earlier episodic composition seam stays closed at readout (emergent assembly select, one-shot attractor, dendritic-plateau completion). As of 2026-09-02 the brain's full roughly 79,000-fact knowledge base, not an earlier 15,000-fact core, is the default it talks from. Some fact content and self-report wording remain host scaffolds. |
| Growth | Structural plasticity and capacity-growth infrastructure exist. A brain-native policy that safely grows a whole developing brain is not yet complete. |

The central problem is integration. Passing a small test does not show that a
mechanism can serve its role in a living brain, so the current frontier is a
continuous-integration arc: validated faculties are wired into a live
conversational loop one dependency at a time and kept only when the conversation
itself measurably improves under causal controls and independent seeds. Running
the real conversation, rather than an isolated probe, is what exposes a
mechanism that was mis-scoped in isolation. A 2026-08-19 audit lesioned the
thirty-one faculties then on by default and found twenty-three genuinely change
the reply and none inert, with roughly fifteen genuinely spiking and
lesion-load-bearing. As of the 2026-09-16 ledger head the tracked
production-integration ledger counts 69 faculties in total, 30 of them
genuinely spiking and on by default, and 5 with their host scaffold fully
retired (`scaffold_retired: 5`), so this remains, by the ledger's own note,
co-residency of many faculties in one loop rather than one true substrate for
most of the live chat's load-bearing cognition, which still runs in host
Python or NumPy. Since 2026-09-19 the headline metric for that problem is a
lesion-verified load-bearing fraction of the whole production conversation
rather than a percent-retired count. A 2026-09-25 re-scored 6-seed battery
reports it as the owner-mandated pair: shipped-default (thin) probes mean
0.603, robust core 15/26 (union 16); adequate measurement-only probes mean
0.949, robust core 24/26 (union 25), SD 0.018 for both — see
[`research/findings/2026-09-25-production-default-battery-B2a-rescored-PASS.md`](research/findings/2026-09-25-production-default-battery-B2a-rescored-PASS.md).
The open problems are a single true substrate for the remaining faculties,
raising that load-bearing fraction under the permanent Qwen mouth, and
mechanisms that emerge rather than being hand-wired.

For a dated status report, see [Current State](docs/CURRENT-STATE.md). For the
planned build order, see the [Roadmap](ROADMAP.md).

## Research Discipline

Temporary scaffolds are allowed to make a question testable, but every scaffold
must remain visible, have a brain-native replacement, and have a condition for
removal. Examples include host-written parsers, fixed concept codes, hand-set
pathways, conventional language training, and external teaching systems. See
the [Scaffold Ledger](docs/SCAFFOLD-LEDGER.md).

Capability claims are expected to survive causal controls, independent random
seeds when practical, and comparison with simpler explanations. Findings cite
raw artifacts and provenance. Negative, corrected, and superseded results stay
in the record because they constrain future work. The chronological evidence is
under [`research/findings/`](research/findings/).

## Run The Project

Start with [QUICKSTART.md](QUICKSTART.md) for installation and backend setup.

Run a small central-processing-unit test:

```bash
SIM_BACKEND=numpy python -m pytest tests/test_strict_step_errors.py -q
```

Launch the interactive simulator on a configured NVIDIA system:

```bash
python neural-simulator.py
```

Most scientific experiments are headless modules in `research/runners/`. They
are research instruments rather than polished product demos.

## Repository Guide

| Path | Purpose |
|---|---|
| `sim/` | Simulation engine, neuron and synapse state, learning, brain regions, neuromodulation, growth, and backends. |
| `experiment/` | Reusable stimulus, training, and readout support. |
| `research/runners/` | Executable experiments and diagnostic runs. |
| `research/findings/` | Dated interpretations of results, including failures and corrections. |
| `research/findings/raw/` | Raw measurements and run provenance. |
| `tools/gates/` | Automated checks for unsupported or stale research claims. |
| `ui/`, `viz/`, `webapp/` | Interactive control and visualization surfaces. |
| `docs/` | Current status, architecture, research standards, and historical plans. |

## Boundaries And License

The project studies functional mechanisms associated with cognition, affect,
self-monitoring, and communication. Every self-report the brain gives is
designed as an honest functional read-out of an internal spiking signal (for
example, "my familiarity monitor reads this as novel"), never a claim of
phenomenal experience; building and measuring those correlates is treated as a
deliverable, not a caveat. It does not claim consciousness, sentience, felt
emotion, human equivalence, or reliable general intelligence. Outputs from
research demos are experimental and must not be treated as authoritative or as
a safety-critical decision system.

The code is released under the [MIT License](LICENSE) and is provided without
warranty. Contributions should preserve the distinction between measured
behavior, interpretation, and speculation.
