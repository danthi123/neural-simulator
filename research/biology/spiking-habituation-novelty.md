---
type: biology
id: spiking-habituation-novelty
mechanism: A per-word "have I heard this before" NOVELTY judgment is realized as short-term synaptic DEPRESSION
  (Tsodyks-Markram `cp_stp_x`, already implemented in `sim/bridge.py`) at a dedicated presynaptic-input -> readout
  pathway, not a permanent host `set`/`dict` membership check. Repeated presentation of the same word depresses its
  own synapse (a smaller postsynaptic response each time, "despite no change in the presynaptic action potential");
  the depression RECOVERS over a multi-second time constant (`stp_tau_d`) during silence, so an old topic revisited
  after a long gap reads as freshly novel again -- a property a permanent Python `set` structurally cannot have.
status: proposed
last_verified: 2026-09-09
current_finding: research/findings/raw/_spiking_habituation_novelty/decisive_6seed.json
current_status: "Focused mechanism de-risk BUILT (research/runners/_spiking_habituation_novelty_derisk.py, no
  sim/ edit): a block-diagonal per-word input->readout circuit realizes graded, monotonic, recoverable habituation
  on a real SimulationBridge. A smoke run (1 seed, numpy) passed the pre-registered gate shape before the decisive
  6-seed battery was queued (research/queue/pool.queue) -- see the finding for the seed-by-seed verdict. NOT wired
  to production this session: `webapp/da_mode_drives_chat.py::engagement_of()` still computes novelty via the host
  `set` (`sum(1 for t in tokens if t not in seen) / len(tokens)`); wiring an open-vocabulary (recruit-on-demand,
  matching `VocabAgnosticSpikingSampler`'s existing pattern) version of this circuit into that function is the
  named next rung."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "decreases despite no change in the presynaptic action potential"
    note: "Kandel, Principles of Neural Science 6e, Ch 53 'Cellular Mechanisms of Implicit Memory Storage',
      Fig 53-2 caption (line ~73726), Aplysia gill-withdrawal habituation. The source file is ISO-8859-encoded
      with a two-column PDF-extraction interleave -- resolve manually with `grep -a`, not plain `grep` (which
      silently treats the file as binary and returns nothing). Repeated stimulation of the siphon sensory
      neuron progressively depresses the motor-neuron EPSP while the sensory neuron's OWN presynaptic action
      potential is unchanged -- the memory trace lives at the SYNAPSE (transmitter release), not in whether the
      upstream neuron fires. This is the direct biological warrant for modeling per-word novelty as a
      synaptic-depression variable (`cp_stp_x`) rather than any change to whether/how a word's own detector
      neuron spikes."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "Habituation involves a decrease in"
    note: "Same figure caption (line ~73730, immediately following the anchor above in the source's own
      two-column interleave). Continues (confirmed by direct reading, not independently grep-anchorable due to
      the interleave): '...transmitter release at many synaptic sites throughout the reflex circuit.' Also on
      this same figure (verified by direct reading): 'One hour after repetitive stimulation, both the EPSP and
      gill withdrawal have recovered' -- habituation is a RECOVERABLE, time-bounded synaptic state, the
      load-bearing property this mechanism's G3 gate (recovery-surpass) measures and the host `set` shortcut
      categorically lacks (a Python `set` never un-learns membership)."
  - path: research/coordination/scaffold_retirement_backlog.md
    anchor: "rank-4"
    note: "The project's OWN prior classification of this exact residual: `research/runners/shared_salience_
      afferent.py`'s docstring calls the raw host scalar computation (message-novelty for DA-mode, content-token-
      count for bg-action-selection, fact-recency-ratio for value-choice) 'a legitimate host sensory/environment/
      memory-provenance boundary, exactly as the SVO parser and the vision percept are' and does not attempt to
      retire it. This binding takes the opposite position for the NOVELTY term specifically: 'has this exact word
      been said before in this conversation' is not a passive percept (unlike a pixel value or an SVO parse of the
      CURRENT sentence) -- it is a judgment over the brain's own history, and CLAUDE.md's brain-based-only standard
      places any such judgment on the neurons/synapses side of the sensation-to-action boundary. The de-risk exists
      to test that position empirically (does synaptic depression actually behave better, not just spikier) before
      committing to override the recorded consensus.
constraints_config:
  enable_short_term_plasticity: true   # THE mechanism; disabling it is this de-risk's own pre-registered LESION
  stp_tau_d: ">= 500.0 (ms)"   # recovery must be on a multi-second, "does the topic recur within one conversation"
                               # scale, not the engine's ~200ms depression default -- the de-risk pins 800.0.
  stp_tau_f: "<= 20.0 (ms)"    # facilitation must decay fast enough to NOT dominate over depression -- this is the
                               # opposite STP regime from this repo's Mongillo working-memory de-risks (which use a
                               # LONG tau_f + LOW U); a habituation reading needs the reverse (short tau_f, high U).
  stp_U: ">= 0.25"             # each presentation must release enough of the readily-releasable pool that a
                               # handful of repeats produces visible depression within the conversational timescale
                               # this mechanism targets (a handful of turns, not hundreds).
implemented_by:
  - research/runners/_spiking_habituation_novelty_derisk.py
findings:
  - research/findings/raw/_spiking_habituation_novelty/decisive_6seed.json
---

# Habituation (synaptic depression) as a spiking, recoverable novelty read

**What the biology says.** Kandel's classic Aplysia gill-withdrawal experiments (Pinsker et al. 1970;
Castellucci & Kandel 1974; Kandel PNS 6e Fig 53-2) showed that repeated stimulation of a single siphon
sensory neuron produces progressively weaker gill withdrawal, and traced the cause directly to the
sensory-to-motor synapse: the postsynaptic EPSP "gradually decreases despite no change in the presynaptic
action potential" — the sensory neuron keeps firing identically on every trial; what changes is how much
transmitter its terminal releases. Critically, the effect is **not permanent**: "one hour after repetitive
stimulation, both the EPSP and gill withdrawal have recovered." Habituation is a graded, synapse-local,
time-bounded memory trace, not a one-bit flag.

**Why this matters for `engagement_of()`.** The novelty term consumed by three production faculties
(da-mode-drives-response, da-gated-encoding, da-gated-curiosity-threshold, all reading it through the
rank-4 shared spiking salience afferent) is computed as `sum(1 for t in tokens if t not in seen) /
len(tokens)`, where `seen` is a Python `set` that only ever grows for the life of a session. Structurally
this is the same class of shortcut CLAUDE.md's brain-based-only standard already names explicitly ("a
reward computed by a distance formula ... IS a shortcut, because the brain is not doing them") — a
cognitive judgment (is this familiar?) computed by host bookkeeping instead of neurons. It also gets the
biology backwards on the one property that actually matters in a real conversation: a topic mentioned once
early on and never revisited should, biologically, regain some freshness — Kandel's own "recovered" gill
withdrawal — but the host `set` marks it "seen" forever, so if it ever comes back up the brain would (once
this scalar reaches the DA/curiosity/encoding gates) treat it as maximally stale regardless of how long ago
it was actually mentioned.

**The mechanism this de-risk builds.** Each candidate word gets a dedicated presynaptic input population
projecting, through dense fixed-weight synapses running in the Tsodyks-Markram **depression-dominant**
regime (high `stp_U`, long `stp_tau_d`, negligible `stp_tau_f` — the mirror image of the
facilitation-dominant regime this repo's Mongillo working-memory de-risks use), onto its own readout
population. Presenting the word both habituates it (drives depression on its own synapse only) and reads
its response in the same step — the readout pool's firing rate during that presentation is the
biologically-grounded EPSP-amplitude analogue, and the sole quantity handed downstream. No `set`, no
`dict`, no host membership test anywhere on the score path (checked structurally, G6).

**What would make this a genuine surpass, not a parity rebadge.** The recovery property: after an
identical habituation history, a **long** silent gap should let the depressed synapse recover substantially
more than a **short** gap (G3), while a **never-touched sibling channel gives an unaffected fresh
reading throughout** (G4, ruling out generic drift or cross-talk as the source of any recovery-looking
signal). If the mechanism only reproduces "seen once -> flat 0 forever," it has merely added spikes to the
existing shortcut; if it shows graded, time-dependent dishabituation instead, it has replaced a permanent
lookup table with something closer to what the cited biology actually does.

**Honest scope.** This is a bounded-vocabulary mechanism de-risk (3 channels), not a wired, open-vocabulary
replacement — `engagement_of()`'s own per-word set membership is untouched this session. The named next
rung is recruiting channels on demand for an arbitrary runtime vocabulary, the same pattern
`VocabAgnosticSpikingSampler` already uses elsewhere in this codebase for open-ended generation, so no new
architectural idea is required — only the wiring.
