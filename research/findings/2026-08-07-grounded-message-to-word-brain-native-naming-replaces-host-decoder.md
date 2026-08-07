---
type: finding
status: contributing
date: 2026-08-07
mechanism: grounded-message-to-word
runner: research/runners/_grounded_message_to_word_derisk.py
artifacts:
  - research/findings/raw/grounded_message_to_word/message_to_word_6seed.json
  - research/findings/raw/grounded_message_to_word/message_to_word_6seed.json.prov.json
---

# Grounded message-to-word: a brain-native naming path replaces the host semantic decoder (CPU de-risk GO)

<!--derived-->
**One-line verdict.** The referent word a grounded brain speaks can be selected by a gated local Hebbian
naming map from the object's percept assembly — not by a host lookup — and the brain-chosen word then flows
through the existing spiking WKV language cortex to a fluent utterance. Under stressing percept noise the
naming decode is graded and robust: six-seed accuracy 0.95 (range 0.938–0.965) versus an untrained
random-map control of 0.23, so 76% of the accuracy is attributable to the learned weights. The CPU/numpy
de-risk is GO on every check: render-faithful 1.0, silence routes to zero renderer invocations, and every
anti-cheat control (chance, lesion, permutation, novel) moves as required. This burns down the referent half
of the `request apple` host semantic decoder called out as temporary in the Aug-03 grounded speech-action
loop. Artifact: `research/findings/raw/grounded_message_to_word/message_to_word_6seed.json`.

## Why this is the next step (re-anchor)

Two GO'd frontiers name the same missing rung:

- `2026-08-02-gap1-wkv-width-ladder-scale-read-run4-d2048...` — the 267M/d2048 WKV language cortex is
  RF-spiking-forward faithful (GO 6/6). Its honest scope: *"use this larger faithful language-circuit
  scaffold inside the grounded speech-action plan while continuing to burn down corpus-training and
  host-side phrasing scaffolds."*
- `2026-08-03-grounded-speech-action-loop-6seed-GO` — a hungry brain that sees a learned apple requests it
  and falls silent once sated (GO 6/6). But its output is a fixed string: *"A host decoder maps one neural
  winner to `request apple` ... the fixed semantic decoder is temporary."* Next-mechanism #3, verbatim:
  *"Replace the host semantic decoder with a brain-native message-to-word path."*

The host `request apple` string is the seam between *the brain chose to speak about the apple* and *the word
produced*. This rung removes the referent-word lookup from the host and gives it to a learned circuit, then
wires the brain-chosen word into the already-GO spiking WKV renderer.

## Mechanism

- **Percept.** Each learned referent is a sparse, overlapping binary assembly (24 active of 240 units), so
  separating them requires learned weighting rather than trivially disjoint codes. At inference the assembly
  is corrupted with per-unit Gaussian noise (sigma 1.6, 120 presentations per referent) so the naming
  accuracy is a graded, discriminating metric rather than a deterministic ceiling.
- **Naming (brain).** A zero-initialized naming matrix learns by a gated local Hebbian rule during a teacher
  naming event: the caregiver co-activates the object's word-unit ("this is an apple") while the percept
  assembly is active, and `W[word] += lr * assembly` (pre = assembly spikes, post = teacher-driven word-unit).
  No weight transport. At inference the plasticity gate is closed and the decode reads **only** the percept
  assembly through the learned weights — the true label is never an argument on the inference path.
- **Articulation (body).** Each word-unit has a fixed binding to one WKV vocab token (the output alphabet, a
  motor-pool → phoneme analogue). *Which* word-unit a referent maps to is learned; the word-unit → token
  binding is the fixed articulatory alphabet.
- **Gate.** A minimal request-vs-silence rate competition (cue + hunger → request; satiety → silence; shared
  inhibition) routes whether to speak. Its full spiking form is already GO in the Aug-03 loop; here it only
  routes.
- **Render.** On a speak decision the brain-decoded word fills the carrier frame `the <agent> <verb> ___` and
  the numpy WKV forward (grounded fine-tuned checkpoint, RF-spiking-forward parity GO) articulates it.

## Six-seed result

<!--derived-->
Seeds 42/43/44/100/101/102, one CPU process (`SIM_BACKEND=numpy`):

<!--derived-->
| check | result |
|---|---:|
| brain-native naming accuracy (noisy) | 0.95 mean, range 0.938–0.965 (chance 0.25) |
| untrained random-map control accuracy | 0.23 mean |
| naming accuracy attributable to learned weights | 76% |
| render-faithful (spoken word == brain-decoded referent) | 1.0 |
| silence → renderer invocations | 0 on every seed |
| lesion of naming pathway → accuracy / confident decodes | collapses to 0.25 / 0 confident (fails safe to silence) |
| permutation followed / original word accepted after permute | 0.95 / 0.02 |
| novel untaught percept abstains (margin < 0.15) | True on every seed |

Exact command:

```bash
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._grounded_message_to_word_derisk \
  --seeds 42 43 44 100 101 102 \
  --out research/findings/raw/grounded_message_to_word/message_to_word_6seed.json
```

## What the controls establish

- **Learned, not a host bias.** An untrained random naming map decodes at chance (0.23); 76% of the trained
  accuracy is attributable to the learned weights.
- **Learned, not wired.** Teaching a permuted referent→word map decodes the permutation (0.95) and rejects
  the original word (0.02) on every seed.
- **Fails safe.** Lesioning the naming pathway collapses accuracy to chance and drops every clean-presentation
  decode below the confidence margin — the brain declines to name rather than emitting a confident wrong word.
- **Specific.** An untaught percept assembly stays below the confidence margin (won't blurt a wrong name).
- **Gate-first, mirroring the moat.** A sated trial routes to silence and the WKV renderer is reached zero
  times; a hungry trial reaches it exactly once.

## Scaffold burned down, and honest boundary

This rung retires the **referent-word** half of the host `request apple` decoder: the word spoken is now
selected by a learned local-rule circuit from the percept, not by a host string. What remains scaffold, and
is named as such: the carrier frame `the <agent> <verb> ___` is still host phrasing; the percept assemblies
are deterministic rather than emerged from vision; the request/silence gate is a rate proxy here (its spiking
form is GO in the Aug-03 loop); intent is fixed rather than learned; and the WKV cortex itself is still
conventionally trained (used here as the fixed faithful language-circuit scaffold, off-bridge in numpy). This
is a wiring/mechanism smoke, not a claim of emerged multimodal naming.

## Next mechanism

1. Learn the intent (request/comment/ask) from contingent outcomes, so the whole message — not just the
   referent — is brain-selected (grounded-speech-action next #2).
2. Drive the percept assemblies from the neural retina/visual features rather than fixed codes.
3. Replace the carrier-frame skeleton with a learned multi-slot message → burn down the remaining host
   phrasing.
4. Run the naming map on the shared spiking bridge with the spiking request/silence gate, then on-bridge WKV.
5. Bring the learned source-monitoring signal (`2026-08-03-laneC` findings) onto this path so uncertainty
   can hedge the spoken message before articulation.

## ✅ PARENT-VERIFIED (independent 6-seed re-run)
<!--derived-->
The parent independently re-ran the 6-seed command (artifact
`research/findings/raw/grounded_message_to_word/message_to_word_val6seed.json`) and reproduced the GO:
naming accuracy 0.935-0.965 (mean 0.952) vs chance 0.24 / lesion 0.25, render-faithful 1.0, silence -> 0
renderer invocations, permutation-followed 0.95 / original-rejected 0.02, novel abstains; 76.1% of the naming
accuracy attributable to the learned map. All 11 preconditions/anti-cheats pass. VERDICT: GO — the
brain-native referent-naming path is confirmed; it burns down the referent half of the host semantic decoder
(the brain now SELECTS and PRODUCES the word for the object it communicates about, via a learned local-Hebbian
map with no weight transport and the true label never on the inference read path).
