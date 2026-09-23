---
type: finding
status: partial
lane: load-bearing
date: 2026-09-23
mechanism: DA-gated synaptic tagging-and-capture (webapp/da_tag_capture.py, BRAIN_DA_TAG_CAPTURE default OFF) under a natural surprising-vs-expected conversational drive, read as next-day recall
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_da_encoding_natural_drive/pilot_seed7.json
---

# PRE-REGISTRATION — DA-gated encoding under a natural drive, measured as 24 h recall (2026-09-23)

This file fixes the gates BEFORE the 6-seed run (outputs will land in the `_da_encoding_natural_drive/` raw directory, one file per seed plus an aggregate). It is committed on its own. The run it governs is
`research/runners/_da_encoding_natural_drive_persistence.py` at the commit that adds this file. The result will be a
separate finding that cites this one. Terms follow `docs/TERMS.md`: nothing here is called *consolidation* (no replay
executes); the process is synaptic tagging and capture.

## What came before (read, not re-derived)

- `da-gated-encoding` read hollow on the load-bearing battery (0/6 on the adequate battery).
- The first fix (branch `research/hollow-da-gated-encoding-drive`) was REJECTED as tuned: it only read load-bearing under a swept `BRAIN_ONEBRAIN_RETRIEVE_DAMAGE_SIGMA` knee plus induced arousal.
- The natural probe (branch `research/gap-da-gated-encoding-v2`, finding `2026-09-20-da-gated-encoding-not-load-bearing-on-a-natural-probe-honest-negative`) showed the DA write gain is invisible to a clean immediate read. It named next-day persistence as the right probe (Bethus, Tse & Morris 2010, J Neurosci 30:1610: hippocampal D1/D5 blockade changed memory persistence over time, not encoding or immediate recall).

## The wall question, answered before building

"What does the real system run alongside this that we replaced with a constant?" Two things, both measured:

1. **Persistence was infinite.** A stored block never decays. In CA1, early-phase LTP returns to baseline within about 1-3 h unless plasticity-related proteins (PRPs) capture it into late-phase LTP. D1/D5 dopamine drives PRP synthesis. A novel event supplies PRPs to synapses tagged up to about 1 h before or after it (Frey & Morris 1997, Nature 385:533; Redondo & Morris 2011, Nat Rev Neurosci 12:17; Moncada & Viola 2007, J Neurosci 27:7476; Wang, Redondo & Morris 2010, PNAS 107:19537).
2. **The synaptic baseline was zero.** A block holds only the fact's increment, so the read has infinite SNR. Pilot (2026-09-23, seed 7, scratch run): scaling a stored block by 1e-6 still decoded the fact exactly; only an exact 0 abstained. Real synapses carry a pre-existing strength; a CA1 E-LTP of about +100% makes the increment about the size of the baseline.

The companion `webapp/da_tag_capture.py` restores both. It is DEFAULT-OFF. No `sim/` edit and no edit to `one_brain_composer.py`.

## The natural drive (no induced arousal, no damage knob)

Two conversations over the SAME four facts (zebra swallow violin; otter steal lantern; goat eat passport; parrot hide key), 16 turns each, 30 s per turn:

- **NEUTRAL**: each fact's words are introduced in short plain turns ("the zebra is here", "the violin is here", "the zebra saw the violin"), then the fact is told plainly ("the zebra swallowed the violin").
- **SALIENT**: three content-free turns ("ok", "oh", "hm"), then the fact told as surprising news with fresh content ("Guess what, at the circus today the zebra swallowed the violin completely whole!").

Every turn goes through the production DA path (`DaModeDrivesWorkspace.observe`: spiking habituation novelty, shared spiking salience afferent, spiking SNc). The DA level at a fact turn sets the production write gain (spiking gain population) and is what the capture machinery reads. The PRP threshold is the brain's existing Go boundary `da_mode_drives_chat._DA_NEUTRAL_MAX` = 0.62, imported.

## Pre-registered constants (primary point, then the robustness band)

| constant | primary | band | source |
|---|---|---|---|
| E-LTP decay `tau_e` | 1.5 h | 1.0 h, 3.0 h | Frey & Morris 1997 (E-LTP gone within ~3 h) |
| baseline/increment `beta` | 1.0 | 0.67, 2.0 | CA1 E-LTP of +50% to +150% of baseline |
| tag window after write | 1.5 h | fixed | Frey & Morris 1997; Redondo & Morris 2011 |
| PRP window before write | 1.0 h | fixed | Moncada & Viola 2007 |
| L-LTP decay `tau_late` | 30 d | fixed | Abraham 2003 (months in vivo) |
| delay to recall | 24 h | fixed | Bethus 2010; Wang 2010 |
| PRP threshold | 0.62 (existing Go boundary) | fixed | `da_mode_drives_chat._DA_NEUTRAL_MAX` |

The band is not a search for a knee. The verdict must hold at EVERY band point where immediate recall holds.

## Arms (each a fresh composer at the same seed)

`intact`; `lesion_da_encoding` (`BRAIN_DA_ENCODING_LESION=1`, the lesion the battery uses: write gain pinned to 1 and the PRP read pinned to tonic); `lesion_capture` (`BRAIN_DA_CAPTURE_LESION=1`, only the capture sub-edge); `companion_off` (no ledger, the production default today); `lesion_novelty` (DA trace recomputed with `BRAIN_SPIKING_NOVELTY_LESION=1`, exploratory, not gated). The production idle-tick Turrigiano pass runs once after the conversation. The reply is `query_patient(agent, action)`: the patient word, or None (abstain).

## Gates (per seed; the aggregate needs GO on all six seeds 42/43/44/100/101/102)

- **G1 natural drive** (precondition): the smallest salient fact-turn DA >= 0.72 (threshold + 0.10) AND every neutral-conversation turn DA < 0.62. If not, the seed is UNDEFINED: the natural contrast was not constructed. A neutral turn at or above 0.62 is a PRP event that would legitimately capture the neutral facts (behavioural tagging), so it makes the condition non-neutral rather than the mechanism wrong.
- **G1b** (precondition): the salient DA trace is identical on a repeat build at the seed.
- **G2 immediate recall** (precondition): 4/4 correct at +1 min in every arm and both conditions. If it fails, the companion broke encoding, which contradicts the biology it models, so the seed is UNDEFINED.
- **Reach** (precondition): the lesion changes the salient mean write gain and the salient PRP-event count.
- **G3 load-bearing**: SALIENT at 24 h, intact >= 3/4 correct AND lesion_da_encoding <= 1/4 AND the difference >= 3. The exact permutation p over the 8 fact outcomes is reported.
- **G4 specific to surprise**: NEUTRAL at 24 h, intact <= 1/4 AND salient-intact minus neutral-intact >= 3.
- **G5 capture sub-edge**: SALIENT at 24 h, lesion_capture <= 1/4 (the magnitude gain alone does not carry the effect).
- **G6 production-default null**: companion_off recalls 4/4 at 24 h in BOTH conditions (it reproduces the 2026-09-20 honest negative).
- **G8 robustness**: at every band point where immediate recall holds, salient intact >= 3, salient lesion <= 1, neutral intact <= 1.
- **Reported, not gated**: confabulations (a wrong non-None answer) across all arms; the novelty-lesion arm; the attribution `attributable_to(salient effect, neutral effect)`.
- **Aggregate**: all six seeds present, no seed UNDEFINED, and the pooled intact-minus-lesion effect above the 95th percentile of a 10 000-draw seed-wise label-permutation null.

## Pilot disclosure

<!--derived-->

Seed 7 (not a gate seed) was used to design the stimuli and check the instrument before this file: salient fact DA 0.90-1.24, neutral max 0.554; salient intact 4/4 at 24 h, lesion_da_encoding 0/4, lesion_capture 0/4, companion_off 4/4; immediate 4/4 in all four arms (`research/findings/raw/_da_encoding_natural_drive/pilot_seed7.json`). The selftest (`--selftest`) checks bit-exact pass-through at beta 0, the flag gate, that FORGOTTEN and REMEMBERED are both detectable, and that the lesions pin the PRP read. G1's thresholds (0.10 above, strict below) were set after seeing the seed-7 trace. The neutral margin there was 0.066.

## Declared host shortcuts

The runner is the parse boundary (it calls `comp.store` for fact turns). The world clock is host (environment). The decay, the capture decision and the threshold compare are host arithmetic on the store synapses, at the same layer as the existing on-store homeostat. The next rung is an on-substrate late-phase synaptic variable (Clopath et al. 2008, PLoS Comput Biol 4:e1000248) and wiring the ledger into the live chat store with a world-clock turn, so the battery can read it.
