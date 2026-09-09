---
type: biology
id: spiking-closed-class-pattern-completion
mechanism: Recognizing a candidate word as a known CLOSED-CLASS item (here, an anaphoric pronoun) is realized
  as CA3-style autoassociative PATTERN COMPLETION over a stored attractor assembly, not a permanent host
  `set`/tuple membership check (`word.lower() in {"it","that","they","them","this"}`). Each closed-class word
  gets its own Hebbian outer-product-installed neuron assembly (Marr's "changes in connections between active
  [cell]s"); a candidate word is turned into a CUE -- possibly a NOISY/PARTIAL rendering of a stored assembly,
  or a pattern drawn from neurons no assembly ever claimed -- and the network's own recurrent excitation
  decides, by ignition above a firing-rate threshold, whether that cue completes to a known assembly. A
  degraded/corrupted cue that shares even a modest fraction of a stored assembly's neurons still completes;
  an unrelated (open-class) word's cue, sharing none, does not ignite anything.
status: proposed
last_verified: 2026-09-09
current_finding: research/findings/raw/_spiking_anaphor_detection/decisive_6seed.json
current_status: "Focused mechanism de-risk BUILT (research/runners/_spiking_anaphor_detection_derisk.py, no
  sim/ edit, reuse-by-import of research.runners.content_selection_spiking.SpikingLoopContextBuffer): a
  cortico-PFC NMDA-bistable attractor loop with one Hebbian-installed assembly per anaphor recognizes clean
  AND heavily-corrupted (80% wrong/missing neurons) cues as their true stored word, does not spuriously
  ignite on content-word cues drawn from the unused neuron pool, and collapses to floor when the attractor
  weights are removed (untrained lesion). See the finding for the seed-by-seed verdict. NOT wired to
  production this session: `research/runners/brain_chat_tui.py::ChatBrain._resolve_anaphora` and
  `research/runners/multi_turn_agent.py::MultiTurnAgent._resolve` still gate on the host Python set test;
  wiring an open-vocabulary (recruit-on-demand, matching `VocabAgnosticSpikingSampler`'s existing pattern)
  version of this circuit into those two call sites is the named next rung."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "the recurrent excitatory connections of CA3"
    note: "Kandel, Principles of Neural Science 6e, section 'The CA3 Region Is Important for Pattern
      Completion' (confirmed by direct reading around this anchor, offset ~6,920,000 chars into the plain-text
      extraction). 'A key feature of explicit memory is that a few cues are often sufficient to retrieve a
      complex stored memory. Marr suggested ... that the recurrent excitatory connections of CA3 pyramidal
      cells might underlie this phenomenon. He proposed that when a memory is encoded, neuronal activity
      patterns are stored as changes in connections between active CA3 cells.' This is the direct biological
      warrant for installing each closed-class word as a Hebbian outer-product assembly (this project's
      established convention, e.g. `research/biology/dg-ca3-sparse-index.md`'s identical anchor) rather than
      as a Python dict/set entry."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "referred to as pattern completion."
    note: "Same section, immediately following the anchor above: 'During subsequent retrieval of the memory,
      the reactivation of a subset of this stored cell assembly would be sufficient to activate the entire
      original neural ensemble that encodes the memory because of the strong recurrent connections between the
      cells of the ensemble. This restoration is referred to as pattern completion.' The direct biological
      warrant for the G2 gate (a cue containing only a SUBSET -- here as little as 20% -- of a stored
      assembly's neurons still completes to the full assembly), and for WHY this is a genuinely different,
      more fault-tolerant claim than the host set test it replaces (`x in {...}` requires the FULL, EXACT
      token; there is no such thing as a 'subset' of a Python string matching)."
implemented_by:
  - research/runners/_spiking_anaphor_detection_derisk.py
  - research/runners/content_selection_spiking.py
findings:
  - research/findings/raw/_spiking_anaphor_detection/decisive_6seed.json
---

# Closed-class (pronoun) recognition as CA3-style pattern completion, not a host set-membership test

**The claim the code must respect.** `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s "anaphora-wm" row names the
residual verbatim: the referent RESOLUTION (`held_referent()` / the WTA biased-competition read in
`multi_turn_agent.py::_resolve_biased`) is already spiking and default-on, but the DETECTION step that gates
it -- "is the current token one of my known pronouns at all?" -- is a bare Python set test in two call sites
(`brain_chat_tui.py::_resolve_anaphora`'s `anaphors = {"it","that","they","them","this"}` and
`multi_turn_agent.py::_resolve`'s `word.lower() in _ANAPHORS`). Per CLAUDE.md's brain-based-only standard,
this is exactly the shortcut class named there: "an argmax over spike counts" is a shortcut even when
biologically-shaped, and a bare host membership test is the same failure mode one level earlier -- it is not
even shaped, it is pure host bookkeeping deciding whether the spiking substrate gets consulted at all.

**Why pattern completion, specifically.** A closed-class word is a small, stable, memorized set -- exactly the
class of "a few cues are sufficient to retrieve a complex stored memory" Kandel describes for CA3. Marr's
1971 proposal -- "when a memory is encoded, neuronal activity patterns are stored as changes in connections
between active CA3 cells," and "the reactivation of a subset of this stored cell assembly would be sufficient
to activate the entire original neural ensemble ... referred to as pattern completion" -- gives a direct,
already-project-validated recipe (the identical citation anchors this project's `dg-ca3-sparse-index.md`):
install each word as a Hebbian outer-product assembly, and let recurrent excitation complete a degraded cue.
The resulting property (graceful degradation under corruption) is not a re-badging of the host set's exact-
match behavior -- it is a **strictly stronger** capability the host set structurally cannot have at any
"badness" level: `x in {"it", ...}` has ONE bit of information (equal or not); pattern completion recovers
the correct classification from a cue sharing as little as ~20% of the true assembly's neurons (the mechanism
de-risk's own calibration sweep found even far sparser cues -- as few as 1-5 of 50 neurons -- often still
complete, at higher seed-to-seed variance; 20% was chosen as the pre-registered, comfortably-robust operating
point, not the mechanism's actual floor).

## What is established, and where the shortcut still stands

**Established (mechanism de-risk, `_spiking_anaphor_detection_derisk.py`):** on a real
`SimulationBridge`-backed cortico-PFC NMDA-bistable loop (`content_selection_spiking.
SpikingLoopContextBuffer`, reused by import, no `sim/` edit), driving a clean full pattern for a known anaphor
ignites that anaphor's own assembly above threshold and no other's; driving a heavily corrupted cue (80% of
the neurons replaced by neurons no assembly claims) still completes correctly; driving a cue drawn entirely
from the unused-neuron pool (the open-class/"unfamiliar word" condition) never spuriously ignites any stored
assembly; and removing the attractor weights entirely (an untrained network, mirroring
`_riii_ca3_completion_specificity_derisk.py`'s own NO-TRAIN control) collapses completion accuracy to floor.
See the finding for the exact seed-by-seed numbers and the pre-registered `tools.verdict.Verdict` gate.

**Declared shortcut, and the named burn-down.** This de-risk validates the DETECTION mechanism in isolation,
on the CURRENT literal 5-word host set (`{"it","that","they","them","this"}`) reproduced as the trained
vocabulary -- it does not yet wire into `_resolve_anaphora`/`_resolve`, and it does not yet generalize to an
OPEN vocabulary (a pronoun the current host set is itself missing, e.g. "he"/"she"/"him"/"her", is out of
scope for this de-risk; the buffer would need to be built with those words as additional trained concepts,
exactly the same "recruit new concepts on demand" step `VocabAgnosticSpikingSampler` already performs
elsewhere in this codebase for an open vocabulary -- the named next rung). The candidate-word -> cue-pattern
ENCODING itself (mapping an arbitrary token string to a specific set of neuron indices) is host bookkeeping
in this de-risk (a permutation-derived assignment, analogous to the DG-CA3 sparse index's own declared "host
rate shortcut" for its projection step) -- the CLASSIFICATION DECISION is what moved onto the substrate, not
yet the string-to-neuron-index encoding a live deployment would need.

## What this entry cannot catch

No `constraints_config`. As with `dg-ca3-sparse-index.md`, the properties that matter here are inequalities
and call-graph shapes (an assembly must ignite from a partial cue; a foreign cue must not), not a scalar
config equality `biology_check --config` could pin. Both live as RUNNER anti-cheats: G2's noisy-cue accuracy
floor and G3's false-positive ceiling are hard gates, and G4's untrained-lesion collapse plus
`tools.lab.attributable_to` prove the recurrent attractor connections -- not some incidental property of the
drive current or the read window -- are what is load-bearing.
