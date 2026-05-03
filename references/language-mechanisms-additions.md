# Language mechanisms — catalog additions (2026-05-02)

This document adds language-specific neuroscience entries to extend the
project's biology catalog. Currently the main catalog (`catalog-build`
branch, `references/feature-catalog.md`) covers basal ganglia, hippocampus,
cerebellum, and general cortical mechanisms thoroughly but underweights
language acquisition / processing literature.

These entries inform the 2026-05-02 text I/O work where 6-seed v2 baseline
plateaued at W→A 28.5% (p=0.027). Eight architectural variations all NEGATIVE.
The remaining biology-grounded directions come from language-specific
neuroscience not yet represented in the catalog.

---

## Cluster G — Language additions (post-2026-05-02)

### G.20 Pulvermüller's neuronal action-word ensembles (embodied semantics)

**System:** Distributed cortical assemblies spanning Wernicke's, Broca's,
and motor/somatosensory cortex. Same neurons participate in BOTH perceiving
the word AND executing the action.

**Biological role:** Hebbian learning binds word patterns (auditory) to
action patterns (motor) through repeated co-activation during embodied
language acquisition. Reading "kick" activates leg motor cortex; reading
"grasp" activates hand motor cortex. fMRI evidence shows somatotopic
mapping of action verbs onto motor cortex.

**Sim status:** PARTIALLY MISSING. Project has `language_input → motor_X`
PFC-bypass pathway (right idea per Geschwind dorsal stream) but uses
SEPARATE 10-neuron motor pools per direction. Pulvermüller's framework
requires OVERLAPPING ensembles — same neurons fire for word + action.

**Cluster:** G primary (cortical integration), H secondary (motor),
J (Hebbian plasticity).

**Prerequisites:** Distributed motor coding substrate (preferred-direction
tuning per neuron in shared M1 pool, instead of 4 separate pools).

**Citations:**
- Pulvermüller F. (1999). "Words in the brain's language." *Behav Brain Sci* 22(2):253-336.
- Pulvermüller F. (2005). "Brain mechanisms linking language and action." *Nat Rev Neurosci* 6(7):576-582.
- Hauk O, Johnsrude I, Pulvermüller F. (2004). "Somatotopic representation of action words in human motor and premotor cortex." *Neuron* 41(2):301-307.

**Behavioral validation:** Word-induced motor cortex activation (TMS,
fMRI) selectively for action words matching effector. Lesion in motor
cortex impairs action-word comprehension specifically.

**Implementation hypothesis (project-specific):** The W→A 28.5% ceiling
is a property of the 4-separate-pool architecture. With distributed
preferred-direction tuning (cosine tuning across shared 40-neuron pool),
STDP can sculpt smoother direction selectivity, and population vector
decoding (Georgopoulos 1986) extracts cleaner direction signal.

---

### G.21 Hagoort's Memory-Unification-Control (MUC) framework

**System:** Three-component architecture for language processing:
- **Memory** (lexical store): posterior temporal cortex, MTG/STG.
  Stored word-form-meaning pairs.
- **Unification** (combinatorial): inferior frontal gyrus / Broca's.
  Combining lexical entries into syntactic / semantic structure.
- **Control** (selection/attention): dorsolateral PFC.
  Selecting among alternatives, maintaining task set.

**Biological role:** All three components active during normal language
use. Memory provides candidates; Unification combines; Control selects.
Damage to any component produces dissociable deficits.

**Sim status:** PARTIAL. Project has:
- Memory: `language_input` region with token embeddings (substrate for lexical store)
- Control: `dlpfc_wm` region (substrate for selection)
- Unification: MISSING — no compositional / syntactic processing

**Cluster:** G primary, O secondary (DA-modulated control gating).

**Prerequisites:** Multi-token (compositional) input rather than single-token.
Compositional structure substrate.

**Citation:** Hagoort P. (2014). "Nodes and networks in the neural
architecture for language: Broca's region and beyond." *Curr Opin Neurobiol*
28:136-141.

**Behavioral validation:** Selective deficits per component damage
(comprehension OK + production OK + binding fails = unification deficit).

---

### G.22 Tomasello shared intentionality / joint attention

**System:** Anterior cingulate cortex, mPFC, TPJ, mirror system; behavioral
mechanism in human infants ~9-14 months ("9-month revolution").

**Biological role:** Language acquisition fundamentally requires joint
attention with caregiver. Word-meaning binding occurs in moments of shared
reference, not from passive auditory exposure. Cross-cultural and
developmental evidence is robust: deaf children of hearing parents who
get late language input still acquire language IF social pragmatics are
preserved.

**Sim status:** MISSING. Current text I/O has action-contingent reward
(closest analog) but no attention mechanism that gates language plasticity.
A shared-attention modulator would amplify word-action binding when both
agent AND "speaker" are attending to the same goal.

**Cluster:** G primary, O secondary (motivation/attention), C tertiary (NM).

**Prerequisites:** Attention modulator (could be added via existing
NeuromodulatorConfig framework as "joint_attention_DA"). Reward modulation
of plasticity already exists; just need attention-dependent gain.

**Citations:**
- Tomasello M. (2003). *Constructing a Language: A Usage-Based Theory of Language Acquisition*. Harvard.
- Tomasello M. (2008). *Origins of Human Communication*. MIT Press.
- Tomasello M. (1999). *The Cultural Origins of Human Cognition*. Harvard.

**Behavioral validation:** Word-learning rate scales with joint-attention
density (Tomasello 2003 reviews). Children with autism (joint attention
deficits) have correspondingly impaired vocabulary acquisition.

**Implementation hypothesis:** Add `attention_dopamine` neuromodulator
that's high when goal_log shows agent has reached goal recently (mimics
"shared reference" of joint attention). Gate `language_input_to_motor`
plasticity by this modulator. Predicts: text I/O accuracy improves when
training only counts trials where agent successfully reached previous
goals (high attention) and discounts post-failure trials.

---

### G.23 Indefrey neurochronometry of word production

**System:** Time-resolved language production sequence in cortex (MEG/EEG):
1. **Conceptual preparation** (0-150 ms): semantic activation
2. **Lexical selection** (150-275 ms): word-form retrieval, MTG
3. **Phonological encoding** (275-455 ms): phoneme assembly, STG
4. **Phonetic encoding** (455-600 ms): articulatory programming, IFG
5. **Articulation** (600+ ms): motor cortex output

**Biological role:** Word production unfolds sequentially via cascaded
activation. Each stage has characteristic latency. Errors at different
stages produce different aphasia subtypes.

**Sim status:** MISSING. Current text I/O has no temporal structure;
language_output drive and readout happen in single 100-200 ms windows
without staged processing.

**Cluster:** G primary, H secondary.

**Prerequisites:** Multi-stage temporal architecture (likely requires
multi-region pipeline with staged synaptic delays).

**Citations:**
- Indefrey P. (2011). "The spatial and temporal signatures of word production
  components: a critical update." *Front Psychol* 2:255.
- Indefrey P, Levelt WJ. (2004). "The spatial and temporal signatures of
  word production components." *Cognition* 92(1-2):101-144.

**Behavioral validation:** EEG component latencies match Indefrey timeline
(N400 for semantic mismatch ~400ms; phonological mismatch earlier).

**Implementation hypothesis:** For our 4-direction task, neurochronometry
may be overkill — but for multi-token (compositional) language, staged
production would be necessary.

---

### G.24 Hickok-Poeppel dual-stream language model (extended)

**System:** Two parallel pathways from auditory cortex:
- **Dorsal stream** (sensorimotor): pSTG → arcuate fasciculus → IFG/Broca's.
  Maps sound to articulation. Damage → conduction aphasia (repetition deficit).
- **Ventral stream** (semantic): mid/inf temporal → ATL/temporal pole.
  Maps sound to meaning. Damage → semantic aphasia.

**Biological role:** Language comprehension requires BOTH streams. Speaking
words requires DORSAL (sound→motor); understanding words requires VENTRAL
(sound→meaning).

**Sim status:** PARTIAL. Project has:
- Dorsal-like: `language_input → motor_X` PFC-bypass (sound-to-motor) ✓
- Ventral-like: `IT → language_output`, `cortex_X → language_output`
  (vision-to-word, action-to-word) — wrong direction (output, not input)
- MISSING: sound → semantic / meaning representation

**Cluster:** G primary, K secondary (audition).

**Prerequisites:** Auditory input pathway (we have it via language_input
text drive); semantic representation substrate (none — our embeddings are
just direction-keyed Gaussian random).

**Citations:**
- Hickok G, Poeppel D. (2007). "The cortical organization of speech
  processing." *Nat Rev Neurosci* 8(5):393-402.
- Saur D et al. (2008). "Ventral and dorsal pathways for language."
  *PNAS* 105(46):18035-18040.

**Behavioral validation:** Dissociation of comprehension vs production
deficits in stroke patients. DTI confirms anatomically distinct AF
(dorsal) and IFOF (ventral) tracts.

---

### G.25 Friederici language network anatomy (multiple parallel pathways)

**System:** White matter tracts critical for language:
- **Arcuate fasciculus (AF)**: dorsal stream — pSTG ↔ Broca's
- **Superior longitudinal fasciculus (SLF)**: parietal ↔ frontal
- **Inferior fronto-occipital fasciculus (IFOF)**: occipital ↔ frontal,
  semantic ventral
- **Uncinate fasciculus (UF)**: anterior temporal ↔ orbital frontal
- **Inferior longitudinal fasciculus (ILF)**: occipital ↔ temporal,
  ventral form-to-meaning

**Biological role:** Language is NOT a single pathway. ~5+ white-matter
tracts each carry specialized information. AF is what most "language
arc" diagrams show, but the others are equally critical for fluent
communication.

**Sim status:** ONE pathway only. `language_input → motor_X` is
analog of AF dorsal. Other pathways missing entirely.

**Cluster:** G primary.

**Prerequisites:** Multi-pathway architecture between language regions
and other cortex.

**Citation:** Friederici AD. (2017). *Language in Our Brain: The Origins
of a Uniquely Human Capacity*. MIT Press.

**Behavioral validation:** Specific lesion patterns produce specific
language deficits (apraxia of speech vs comprehension deficit etc.).

---

## Recommended additions sequence

For pushing text I/O past 28.5% W→A ceiling:

1. **G.20 Pulvermüller distributed coding** — IMMEDIATE. Architectural
   change to motor pools. Tests the hypothesis that 4-separate-pools is
   the bottleneck.
2. **G.22 Tomasello joint attention** — NEAR-TERM. Add attention NM that
   gates plasticity.
3. **G.21 Hagoort MUC + G.23 Indefrey** — LONGER-TERM. Compositional
   language requires multi-token + temporal structure.
4. **G.25 Friederici multi-pathway** — LONGER-TERM. Adds ventral semantic
   stream alongside existing dorsal sensorimotor.

These additions should eventually be merged into `references/feature-catalog.md`
on the `catalog-build` branch under "Cluster G — additions" section.

## Implementation cross-references

- 2026-05-02 W→A 28.5% baseline — `research/findings/2026-05-02-text-io-BREAKTHROUGH-v2.md`
- Negative followups (8 variations) — see `2026-05-02-FINAL-overnight-summary.md`
- Curriculum-NEGATIVE diagnosis — `2026-05-02-curriculum-NEGATIVE-but-INFORMATIVE.md`
- Distributed motor coding design (forthcoming) — TBD
