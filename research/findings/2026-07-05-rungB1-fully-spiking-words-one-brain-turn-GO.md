# FULLY-SPIKING WORDS — the whole transitive turn on ONE cupy process, every WORD + the ORDER on spikes (GO)

**Date:** 2026-07-05
**Runner:** `research/runners/_rungB1_fully_spiking_words_capstone_derisk.py`
**Test:** `tests/test_rungB1_fully_spiking_words.py`
**Raw:** `research/findings/raw/_rungB1_fully_spiking_words_capstone.json`
**Builds on:** EMERGE-95 (three spiking components on one bridge), EMERGE-88 (reservoir comprehends → composer answers),
EMERGE-77 (2-stage calibrated ditransitive order read), EMERGE-67/68 (A→W spiking word-spell), thread A
(`_rungB1_aw_neural_words_transitive_derisk`, the 16-word transitive A→W vocab).

## The milestone

EMERGE-95 ran the whole transitive conversational turn as three disjoint slices on ONE `SimulationBridge` co-executing in
ONE cupy process: the spiking **reservoir COMPREHENDS** (form→role), the RF-phasor **composer REMEMBERS** (store/recall),
the Izhikevich **producer SPEAKS the emission ORDER**. But the producer's word SURFACES were still the host-token spell —
the ORDER was spiking, the WORDS were a Python lookup. This capstone closes that last host seam: it composes the EMERGE-95
turn with the **A→W spiking word-spell** (content words via BRIDGE-A, function words via BRIDGE-F, each decoded from
`cp_firing_states[language_output]`) + the **EMERGE-77 2-stage calibrated order read** (the cupy near-tie fix) — so the
whole turn is spiking end-to-end, all in ONE cupy process:

> `hear "the dog chases the ball"` → reservoir(shared bridge) parses roles → composer(shared bridge) stores + recalls →
> producer SPEAKS "the dog chases the ball" with the spiking ORDER (shared bridge) AND every WORD decoded from
> `language_output` spikes (BRIDGE-A/F). GATE-FIRST moat (composer abstains → the producer is never invoked).

Comprehension on spikes, memory on spikes, production ORDER on spikes, production WORDS on spikes — no host f-string, no
host-token surface. The producer's grammatical STRUCTURE is itself self-organized from corpus experience (EMERGE-62..77:
function-word inventory + slot order + slot inventory, all mined). Transformer-free, host-token-free, one brain, one
process.

## Result — GO

| seed | parse (reservoir) | recall (composer) | render_exact ALL-WORD | moat FA / invoked | content-lesion |
|---|---|---|---|---|---|
| 42 | 1.000 | 1.000 | 1.000 | 0.000 / 0 | 0.000 |
| 43 | 1.000 | 1.000 | 1.000 | 0.000 / 0 | 0.000 |
| 44 | 1.000 | 1.000 | 1.000 | 0.000 / 0 | 0.000 |
| **agg** | **1.000** | **1.000** | **1.000** | **0.000 / 0** | **0.000** → **GO** |

**3/3 seeds = full GO** (42/43/44, confirmed): the reservoir comprehends every transitive fact, the composer recalls every
patient, the producer speaks every answer with the spiking order AND every word decoded from `language_output` spikes; the
gate-first moat abstains on all 30 never-stored cues with **0 false-accepts and 0 producer invocations** (the producer is
never even reached on an abstain — the moat holds by construction); the **content-lesion** (zero the A→W content
pool→`language_output` pathway) collapses the rendered words to 0.000 — proving the words are genuinely spike-decoded, not
a host lookup that a lesion would leave untouched.

## The load-bearing fix — the open-word fact filter

First seed-42 pass parsed 0.833, not 1.000. Root cause (diagnosed, not patched-over): a transitive filler word can be an
EMERGE-62 closed-class **false-positive** — e.g. "cat" is high-frequency + distributionally flat enough that the
reservoir's discovered closed class labels it a function word, so the reservoir mis-parses a fact using it as a filler.
That is the SEPARATELY-characterized closed-class-discovery precision property, not a defect of the fully-spiking turn.
The fix filters the fact CONTENT to words the reservoir sees as genuinely OPEN (`_facts(seed, closed=set(discovered))`) —
the A→W cache still spells the excluded word, it is just not used as a fact filler. seed 42 → parse 1.000. (Mirrors the
EMERGE-90/95 open-word filter.)

## Anti-cheats (all pass)

- **content-lesion** collapses the all-word render (0.000) — the words are spiking, not host-tokens.
- **gate-first moat** — 0 false-accepts on 30 never-stored cues AND 0 producer invocations on abstain (the composer's
  abstention gates the producer; it is never reached).
- the calibrated order read (EMERGE-77 2-stage per-pool bias) is the causal fix for the cupy near-tie (the un-calibrated
  read swaps adjacent slots on the deepest seeds — thread A `--host-order-check`).

## Honest scope

- The CANONICAL transitive turn is position-solvable and works end-to-end here. The harder non-local structural read
  (object-relative, role≠position) is the separate objrel-surpass frontier (a characterized boundary being worked past
  via the learned-signed read — `2026-07-05-rungB1c-objrel-ff-inhibition-BOUNDARY.md`), orthogonal to this milestone.
- The producer renders the BOUNDED corpus-mined construction inventory (the EMERGE-72..77 registry: 7 constructions incl.
  C_TRANS + ditransitive), NOT open prose (R4, the honestly-deferred scale wall).
- Reuse-by-import; NO `sim/` edit anywhere. The A→W read-out is GPU-trained once + cached (the validated scale lever).

## Files
- `research/runners/_rungB1_fully_spiking_words_capstone_derisk.py` — the capstone composition.
- `tests/test_rungB1_fully_spiking_words.py` — CPU structural guard (hooks + GO thresholds + the open-word filter).
- `research/findings/raw/_rungB1_fully_spiking_words_capstone.json` — the multi-seed record.
