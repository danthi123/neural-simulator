---
type: finding
status: go
date: 2026-08-11
mechanism: spiking FS-WTA (lateral-inhibition) patient read-out replacing the host argmax/cosine in the live chat
lane: E-language / INTEGRATION (brain-based-only burn-down)
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/lanes/stageA/corpus_facts_live_chat_neural_readout_6seed.json
runner: research/runners/_corpus_facts_into_live_chat_derisk.py (--neural-patient-readout)
instrument: INTEGRATION #6's live-chat de-risk with an additive `--neural-patient-readout` flag routing the patient-word decision through a spiking FS-WTA read-out (from `_neural_wta_word_decode_derisk`, a standing parity-1.0 GO) instead of the host argmax/cosine; SIM_BACKEND=numpy; cfg.seed-controlled.
---

# INTEGRATION burn-down — the patient-word decision in the live chat is now on SPIKES (6/6 GO)

The live chat (#6/#7) read the patient word via a HOST argmax/cosine over the composer's `query_patient` — a declared
shortcut (the neural-motor-read-out target, project-wide). This burn-down routes that decision through a SPIKING
FS-WTA read-out (winner-take-all via lateral inhibition — the standing `_neural_wta_word_decode_derisk` GO, parity
1.000 with host argmax on fresh bridges), behind an additive `--neural-patient-readout` flag (default OFF ⇒ #6
byte-identical).

## Result (`research/findings/raw/lanes/stageA/corpus_facts_live_chat_neural_readout_6seed.json`)

<!--derived-->

The runner's own GO gate — which requires the spiking read-out to MATCH the host argmax on the taught facts (parity),
the no-confab moat to hold (0 false-accepts), and the decision to be attributable to the true drive (permute-drive →
chance) — returns **`verdict: GO`, `n_go: 6/6`, `neural_patient_readout_go: true`** across seeds 42/43/44/100/101/102.
So the chat's grounded replies are preserved while the patient decision is now decided on SPIKES, with the moat intact.

## Scope / honesty

<!--derived-->

- The patient-word DECISION is now a spiking FS-WTA (lateral inhibition), not a host `max()` — a brain-based-only
  burn-down of a project-wide shortcut.
- Default OFF ⇒ byte-identical to #6. Additive, runner-side, NO `sim/` edit.
- Remaining scaffolds (unchanged, named): the composer cue codebook + the fact CONTENT store (host VSA), the AI-teacher
  presentation, the generator mouth. This closes the READ-OUT shortcut only.
- The build agent deferred before reporting per-seed parity numbers; the coordinator merged on the artifact's own GO
  verdict (the gate encodes the parity/moat/attribution anti-cheats). Per-seed detail lives in the cited artifact.
