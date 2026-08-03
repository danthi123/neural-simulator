# Roadmap

This roadmap is organized around abilities the whole simulated brain must show.
A working component is useful evidence, but it is not a completed capability
until it changes the behavior of the same continuously running brain.

## Destination

Build a developing, fully spiking brain that can learn from a body, a world, and
other people. It should form memories, needs, emotions, beliefs, and language as
parts of one system; speak because of its own state; express uncertainty when
its evidence is weak; and keep learning without requiring a datacenter.

The target is not a text model wrapped in brain terminology. Computation that a
biological brain would perform must ultimately be carried by neurons, synapses,
local signals, and activity on the shared simulation substrate. The host
computer may provide the body, world, sensory input, motor effects, storage,
and measurement.

## Design Constraints

- **One brain:** perception, memory, motivation, action, and language must
  interact through the same evolving spiking system.
- **Grounded learning:** words and concepts must be learned through perception,
  action, internal state, memory, and social consequences.
- **Local mechanisms:** final learning and decision pathways must not depend on
  an external answer key, symbolic controller, or host-side cognitive rule.
- **Development:** the system should start small, add useful structure and
  capacity as it learns, and consolidate experience without erasing older
  learning.
- **Ownable compute:** the practical target is high-end consumer hardware.
  Sparse, event-driven, and local mechanisms should remain compatible with
  future neuromorphic hardware.
- **Visible scaffolding:** temporary shortcuts are allowed for research only
  when their purpose, replacement, and removal test are recorded in the
  [scaffold ledger](docs/SCAFFOLD-LEDGER.md).

## Capability Roadmap

### 1. Communicate For A Grounded Reason

**Outcome:** the brain chooses whether to communicate, what kind of message to
send, and what it refers to because of its perception, body state, memory, and
expected social result.

Build toward multiple needs, objects, and message types in a continuous world.
The brain must learn the useful vocal action from contingent interaction rather
than receive a desired output pattern from the host. A listener's response must
change later neural activity and behavior.

**Evidence required:** novel combinations of learned intent and referent work;
removing the relevant need, perception, learned pathway, dopamine-like teaching
signal, or social consequence removes the behavior; shuffled or unrelated
rewards do not teach it; the result repeats across random seeds.

This is the highest-priority capability. The single food-request result closes
the smallest loop. A newer six-seed experiment learns two intent channels and
two referent channels from contingent listener responses and composes the two
combinations withheld from training. The next gates are intrinsic neural
exploration, same-brain adaptation when a listener changes convention, broader
meanings and contexts, and removal of the fixed raw-channel decoder.

### 2. Turn Internal Messages Into Natural Speech

**Outcome:** once the brain has decided to speak, a neural production pathway
turns its message, certainty, affect, and conversational context into variable,
coherent language.

Near-term work can use a conventionally trained language circuit as a recorded
scaffold downstream of the brain's message decision. In parallel, grounded
word, sequence, and speech-motor learning must replace fixed concept labels,
grammar frames, and host rendering.

**Evidence required:** phrasing varies while meaning remains grounded; speech
changes appropriately with memory, affect, and uncertainty; new words and
constructions can be learned through interaction; removing message or grounding
inputs changes or prevents the utterance.

### 3. Know The Source And Strength Of Its Knowledge

**Outcome:** confidence and honesty arise from the brain's own memory and
decision state. It can answer, hedge, ask, or remain silent without consulting a
host fact table or expected answer.

Source memory must distinguish experienced, heard, self-generated, inferred,
imagined, and uncertain content. Confidence and conflict signals must directly
influence speech selection through the shared spiking network.

**Evidence required:** familiar but incorrect recalls are downgraded; correct
recalls remain usable; unknowns do not become assertions; source swaps change
the report in the predicted direction; lesions to source or confidence pathways
remove the effect; no symbolic source lookup is used during retrieval.

### 4. Build A Lived, Reconstructive Memory

**Outcome:** episodes are stored with context and source, later reconstructed,
used to predict, and gradually consolidated into distributed knowledge.

The brain needs pattern completion, replay, correction of old memories, and a
controlled way to combine prior experience without turning memory into fixed
database slots. Replay must train useful cortical pathways while preserving
older learning.

**Evidence required:** partial cues recover appropriate episodes; changed facts
update rather than duplicate indefinitely; replay causally improves later
behavior; interference tests show retention; novel combinations are inferred
without an answer table.

### 5. Develop Emotion, Motivation, And Curiosity

**Outcome:** changing internal states continuously influence attention, memory,
learning, speech, and action. Emotion should be learned and graded, not a label
or binary mood switch.

Develop interoceptive body signals, appraisal of events relative to needs and
relationships, persistent valence and arousal, and curiosity based on learning
progress. These signals must participate in the same action and communication
loops as perception and memory.

**Evidence required:** internal-state changes alter several faculties; matched
lesions remove the predicted effects; the brain distinguishes learnable novelty
from noise; affect develops from history and can recover or change when
conditions change.

### 6. Learn Continually And Grow

**Outcome:** natural interaction produces durable learning throughout operation,
old abilities survive new experience, and capacity expands only when needed.

This requires local credit assignment over useful time spans, replay-based
consolidation, homeostatic stability, activity-dependent wiring, and explicit
growth and pruning. External teaching should fade from structured caregiver
interaction toward ordinary human interaction.

**Evidence required:** learning continues after initial training; delayed
consequences credit the right pathways; new learning does not erase established
skills; added neurons or connections improve a measured capacity limit; the
same behavior survives removal of oracle-like teaching.

Deep credit assignment on real spikes is still an open research problem in this
repository and must be treated as such.

### 7. Scale Without Changing The Scientific Claim

**Outcome:** richer worlds, memories, and language run efficiently on accessible
hardware while preserving the same neural mechanisms.

Use the local GPU for large coupled simulations and the available CPU pool for
independent trials and parameter searches. Improve sparse kernels, memory use,
checkpointing, and workload scheduling before increasing scale.

**Evidence required:** report wall time, memory, neuron and synapse counts, and
energy-relevant activity; optimized and reference implementations agree; larger
systems retain causal controls; normal development remains practical on
consumer hardware.

## Near-Term Success Target

The next meaningful prototype is a small continuously running brain in a simple
world that can:

- perceive several objects and body conditions;
- learn which outcomes satisfy which needs;
- learn at least two communicative intents and several referents;
- choose speech or silence from internal state;
- remember the source of a small set of experiences;
- let uncertainty, affect, and curiosity alter what it does;
- change future behavior after interaction with a person or caregiver;
- retain earlier learning across continued experience.

Its speech may be simple. Success is an integrated, causal, developing system,
not surface fluency.

## How Progress Is Accepted

A capability claim should include a reproducible artifact, multiple random
seeds when practical, matched controls, and a test showing that the named neural
mechanism is necessary. Results that depend on fixed labels, host decisions, or
external training remain valuable experiments, but stay marked as partial until
their scaffolds pass the removal conditions in the ledger.
