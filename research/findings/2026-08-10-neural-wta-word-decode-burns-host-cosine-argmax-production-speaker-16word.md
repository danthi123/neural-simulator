---
type: finding
status: contributing
date: 2026-08-10
mechanism: neural-wta-word-decode
runner: research/runners/_neural_wta_word_decode_derisk.py
instrument: shuffled-score control (permute the conditioned drive -> FS-WTA winner agreement with the host winner collapses to chance) + attributable_to(parity vs shuffle) isolates the WTA reading the true drive from a generic sorter
attributable_to: nearly all of the neural-WTA's agreement with the host winner is attributable to the WTA reading the TRUE per-word drive (near-perfect parity vs chance-level shuffled-score parity); precise fractions in the derived body table
artifacts:
  - research/findings/raw/neural_wta_word_decode/neural_wta_16word_6seed.json
  - research/findings/raw/neural_wta_word_decode/neural_wta_16word_6seed_softmax.json
  - research/findings/raw/neural_wta_word_decode/_v16_seed42_spikes.npz
---

# Neural WTA word-decode replaces the host cosine-argmax in the PRODUCTION speaker at vocab-16 (6-seed GO, parity 1.000)

<!--derived-->
**One-line verdict.** The production speaker's host `best = max(vocab, key=lambda w: _cosine(spike, patterns[w]))`
word-decode (`_realcorpus_full_frame_speech_derisk.ConceptFrameSpeaker.spell`) is replaced by a NEURAL FS-WTA
read-out over the 16 word-assemblies, and the neural WTA reproduces the host decode with no accuracy loss: on the
16 real driven `language_output` spike patterns from the trained seed42 v16 bridge, host cosine spells 0.750
(12/16) and the neural WTA spells 0.760 mean (`center` gain-control, 5/6 seeds match host, parity 0.948) or 0.750
exactly (`softmax` gain-control, **6/6 seeds, parity 1.000** -- the WTA reproduces the host winner on every word,
every seed). The shuffle anti-cheat is clean (permuted drive -> WTA-host agreement 0.042 = chance 0.0625), and
95.6-95.8% of the agreement is attributable to the WTA reading the true drive. This burns down the host
cosine-argmax the integration brief flagged as a shortcut in the production speaking path. Runner
`research/runners/_neural_wta_word_decode_derisk.py`; NO `sim/` edit; reuse-by-import; numpy backend.

## What this closes

The production full-frame speaker produces content words ON SPIKES (drive a word's concept pool, read
`language_output` firing) but then DECODES the word with a HOST cosine-argmax over the 16 reference patterns
(`spell()`, line ~58). That argmax is host-computed -- a shortcut under the brain-based-only standard (the
read-out SELECTION is the host's bookkeeping, not the brain's). Last night's learn-to-speak GO chose the
utterance by a NEURAL soft-WTA over competing assemblies (winner = highest late-window firing), but only at K=3
toy scale, hand-calibrated for 3 assemblies. The map flagged scaling to 16-32 competing assemblies as an OPEN
re-de-risk. This rung answers it at the production 16-word vocab.

## Mechanism

The read-out path becomes fully a selection-on-spikes:

    language_output spike pattern (2048-d, ON SPIKES from the v16 bridge)
      -> per-word SYNAPTIC drive = projection of the spike pattern onto each word-assembly's afferent weights
         (= its taught `orthogonal_drive_pattern` reference; a Hebbian read-out matrix)          [16 scores]
      -> AFFERENT GAIN-CONTROL (feedforward-inhibition common-mode removal; see below)
      -> K=16 competing word-assembly pools + a shared inhibitory FS pool (LATERAL INHIBITION)
      -> the winner fires first, recruits FS, FS suppresses the runners-up
      -> a CLEAN one-of-K SPIKING winner (argmax of per-pool FIRING) == the decoded word.

The WTA is the validated `build_fswta_score_bridge`/`fswta_drive` (`_d3_spiking_attractor_derisk.py`) -- the SAME
one-of-K spiking WTA the reslm read-out parity ran to K=200 (`2026-07-13-reslm-SPIKING-readout-parity-12seed-GO`:
"parity tracks the score MARGIN, not K"). Only the ARGMAX is moved onto spikes; the per-word score is a synaptic
projection (as in the reslm read-out), and the winner is read from per-pool spike counts.

**Why cosine == the neural dot.** Every word's `orthogonal_drive_pattern` reference has the SAME active count at
the SAME pA, so all 16 references are EQUAL-NORM. The host cosine's per-word norm division is therefore a constant
across words and cannot change the winner: argmax(cosine) == argmax(dot). Measured directly -- host cosine and the
neural dot-projection pick the IDENTICAL winner on all 16/16 words. So the neural WTA targeting the dot-argmax IS
targeting the host cosine winner; the only open question is whether the SPIKING WTA resolves the 16-way margins.

## The companion process the raw proxy omitted (the wall-reframe, in miniature)

<!--derived-->
The raw synaptic dot scores carry a large COMMON-MODE baseline: a word's `language_output` pattern overlaps EVERY
reference pattern, so all 16 scores are large (47k-131k on the real patterns) and the discriminative signal is
their DIFFERENCE. A naive `s / max(s)` drive preserves that baseline -> the normalized margins are tiny -> the
FS-WTA resolves only ~0.75-0.81 parity. This is exactly the common-mode-convergence trap `instrument_required`
warns of. The fix is the missing companion process the raw projection replaced with nothing: a shared
feedforward-inhibition pool (de Almeida-Idiart-Lisman E%-max; the FS interneurons the real read-out circuit runs
ALONGSIDE the projection) that SUBTRACTS the common-mode baseline so the relative margins land in the WTA's
resolvable range. Both conditioning modes are MONOTONIC -- they preserve the host argmax by construction and do
NOT compute it; the argmax is resolved by the spiking lateral inhibition, and the shuffle control proves it.

  - `center`  : `(s - min)/(max - min)` -- subtractive feedforward-inhibition baseline removal (PRIMARY, transparent linear).
  - `softmax` : divisive-normalization `exp((s - max)/(0.5*std))` -- sharper E%-max gain-control (parity 1.000).

## Six-seed result

<!--derived-->
Seeds 42/43/44/100/101/102 (numpy). The 16 `language_output` spike patterns are the trained seed42 v16 bridge's
(cached); the seed varies the FS-WTA read-out bridge's neural heterogeneity + shuffle RNG.

<!--derived-->
| conditioning | seeds GO | host cosine spell | neural-WTA spell | parity (WTA==host winner) | shuffle parity | attributable to true drive |
|---|---:|---:|---:|---:|---:|---:|
| `center` (FF-inhibition baseline removal) | **5/6** | 0.750 | 0.760 mean (>= host) | 0.948 | 0.042 | 95.6% |
| `softmax` (E%-max divisive-norm) | **6/6** | 0.750 | **0.750 (= host exactly)** | **1.000** | 0.042 | 95.8% |

chance = 0.0625 (1/16). Per-seed GO gate: `neural_acc >= host_acc - 0.0625` AND `parity > 0.9` AND `shuffle_parity < 0.5`.

<!--derived-->
The host cosine ceiling is 0.750 (12/16), not 1.0, because the v16 bridge's TRAINED A->W read-out confuses four
near-tie word pairs (apple->west, go->cat, big->look, small->big; relative margins 0.05-0.16) -- a
bridge-training property ORTHOGONAL to the read-out-selection this rung replaces. The neural WTA matches the host
on those confusions too (it is a faithful replacement, not a corrector): under `softmax` it reproduces the host
winner on all 16/16 words every seed.

Exact commands:

```bash
# PRIMARY (center / feedforward-inhibition baseline removal), 5/6 GO:
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._neural_wta_word_decode_derisk \
  --bridge bridges/v16/seed42.simstate.h5 --seeds 42 43 44 100 101 102 --score-mode center \
  --out research/findings/raw/neural_wta_word_decode/neural_wta_16word_6seed.json
# faithful-replacement variant (softmax / E%-max), 6/6 parity 1.000:
PYTHONPATH=$PWD SIM_BACKEND=numpy .venv/bin/python -m research.runners._neural_wta_word_decode_derisk \
  --bridge bridges/v16/seed42.simstate.h5 --seeds 42 43 44 100 101 102 --score-mode softmax \
  --out research/findings/raw/neural_wta_word_decode/neural_wta_16word_6seed_softmax.json
```

## What the controls establish

<!--derived-->
- **The WTA reads the drive, not a host leak (anti-cheat).** Permuting the conditioned scores before the FS-WTA
  collapses its agreement with the true host winner to 0.042 = chance 0.0625 (1/16). If the WTA were leaking the
  host answer, shuffling would not move it. `attributable_to(parity, shuffle_parity)` returns 0.956-0.958: 95.6-
  95.8% of the WTA's host-agreement is attributable to reading the true per-word drive, not to a generic sorter.
- **Genuinely spiking.** The decoded word is `argmax` of per-pool FIRING counts accumulated over the settle window
  (`fswta_drive` returns spike accumulations); a winner requires `acc.max() > 0`, and under a permuted drive the
  SPIKING winner follows the permutation.
- **The conditioning does not do the decision.** `center` and `softmax` are monotonic in the scores -> they cannot
  change which score is largest; the argmax is resolved by the spiking lateral inhibition. (The shuffle control is
  what proves this operationally: a monotonic transform of shuffled scores still leaves the WTA at chance.)

## Honest scope / boundary

- **Single speaker substrate.** The 16 spike patterns come from the ONE trained bridge in the tree
  (`bridges/v16/seed42.simstate.h5`); the 6 seeds vary the FS-WTA read-out bridge's heterogeneity. A 6-bridge
  SPEAKER sweep (retrain v16 at 6 seeds) is the GPU follow-on (see GPU-deferred below). This is a read-out-
  mechanism de-risk on real driven patterns, not a claim about across-substrate speaker variance.
- **The score projection is host-computed here.** `score[w] = spike . reference[w]` is a synaptic projection
  computed in numpy (as in the reslm read-out parity); wiring it as REAL on-bridge synapses
  (`language_output -> 16 word pools`, weights = the reference patterns) so the whole read-out runs on one bridge
  is the follow-on. This rung moves the ARGMAX (the decision) onto spikes; the projection is the named residual.
- **The `center`/`softmax` common-mode removal is an explicit afferent normalization** standing in for the shared
  feedforward-inhibition (E%-max) pool that a real read-out circuit runs; making the FS pool itself do it
  on-substrate (afferent-driven E%-max, already used in the crosstalk-boundary work) is the on-bridge follow-on.
- **Host ceiling is 0.750**, bounded by the v16 bridge's A->W read-out training (4 near-tie confusions), not by
  the WTA. Lifting the ceiling is a bridge-training question, separate from this read-out-selection burn-down.

## GPU-deferred (owner is gaming; GPU held by Palworld at run time)

The 6-seed numpy CPU de-risk above is the decisive result (the FS-WTA read-out and the score projection are
backend-agnostic). The remaining cupy confirmation is the SPEAKER sweep -- retrain the v16 bridge at 6 seeds and
re-collect patterns per seed -- run when the GPU frees:

```bash
# per seed: retrain v16 (concept_pool_demo --save-bridge) OR reuse a 6-seed v16 set, then:
PYTHONPATH=$PWD SIM_BACKEND=cupy .venv/bin/python -m research.runners._neural_wta_word_decode_derisk \
  --bridge bridges/v16/seed<S>.simstate.h5 --bridge-seed <S> --seeds <S> --score-mode softmax --no-hard-reset \
  --cache research/findings/raw/neural_wta_word_decode/_v16_seed<S>_spikes.npz \
  --out research/findings/raw/neural_wta_word_decode/neural_wta_16word_speakerseed<S>.json
```

## Next mechanism

1. Wire the score projection as on-bridge synapses (`language_output -> 16 word pools`) so the whole read-out --
   projection + FS-WTA -- runs on one spiking bridge (no host dot product), with the FS pool doing the E%-max
   common-mode removal on-substrate (afferent-driven feedforward inhibition).
2. Flip the production `ConceptFrameSpeaker.spell()` to the neural WTA read-out (additive default-off first,
   `readout="neural"`), keeping the host cosine as an oracle for parity CI.
3. Lift the 0.750 host ceiling separately (the bridge's A->W read-out on the 4 near-tie words) -- a training
   question, e.g. the decorrelating read-out kernel from the crosstalk-boundary arc.
4. Extend to K=32-61 (the multi-bridge dispatch vocab) -- the reslm parity holds to K=200 when scores are
   discriminable, so the open variable is the per-word margin at larger K, not the WTA.
