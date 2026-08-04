---
type: finding
status: live
date: 2026-08-04
mechanism: auditory-cochlear-transduction
---

# Microphone to cochlea to tonotopic A1: construction package

## What was built

The project previously had text- or concept-driven `language_input` pathways,
but no actual microphone, cochlear hair-cell, auditory-nerve, or primary
auditory-cortex front end. The new construction runner accepts normalized mono
PCM from a WAV file, an array, or the local ALSA microphone. It converts that
waveform into a place-coded auditory-nerve spike raster. A declarative builder
adds channel-specific auditory-nerve, excitatory A1, and PV-like inhibitory A1
regions and pathways to the existing shared `SimulationBridge` before bridge
initialization. A strict adapter can then drive only the auditory-nerve input
regions from the peripheral spike raster.

This is an input pathway, not speech understanding. It performs no automatic
gain normalization, transcription, phoneme labeling, word segmentation,
classification, or semantic lookup. A1 receives only normal synaptic activity
from auditory-nerve bridge regions or other A1 regions. There is no standalone
NumPy A1 dynamics loop.

## Why this mechanism

The local reference catalog was searched before implementation. Kandel chapter
26 describes the basilar membrane as an approximately logarithmic place map,
inner hair cells as the main afferent transducers, active cochlear compression,
adaptation, and tonotopically organized auditory-nerve output. Kandel chapters
27 and 28 describe phase-limited nerve timing and preservation of frequency
organization through the ascending pathway and auditory cortex. The project's
own glossary and coverage audit explicitly recorded hair cells as missing and
the existing generic topographic support as only partial.

Missing implementation detail was filled from primary or direct technical
sources:

- Greenwood's measured cochlear frequency-position relation supports a
  monotonic near-exponential place map: https://doi.org/10.1121/1.399052 <!--derived-->
- Glasberg and Moore's human auditory-filter measurements support
  equivalent-rectangular-bandwidth channel spacing:
  https://doi.org/10.1016/0378-5955(90)90170-T <!--derived-->
- Kato, Asinof, and Isaacson's awake-cortex recordings show that network-level
  lateral suppression shapes A1 frequency tuning:
  https://doi.org/10.1016/j.neuron.2017.06.019 <!--derived-->
- SciPy's gammatone implementation directly follows the established
  Patterson-Holdsworth/Slaney auditory-filter construction instead of using a
  new hand-written filter design:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.gammatone.html

## Boundary and controls

External sensory transduction contains the digital cochlear filterbank,
half-wave hair-cell rectification, compressive response, low-pass receptor
potential, adaptation, and deterministic refractory auditory-nerve spike
conversion. These are legitimate interface-side approximations of physical
organs that the computer does not possess.

Brain-internal processing is declared as ordinary shared-substrate regions and
pathways: aligned auditory-nerve afferents, local excitatory/PV circuitry, and
flanking GABA_A inhibition. The adapter accepts only a `CochlearFrame`, writes
only to the auditory-nerve regions' external-current slices, and restores those
slices after use. Construction controls require ordered cochlear place, quiet
silence, state-correct streaming chunks, sublinear level growth, strict adapter
isolation, and real-time local CPU processing. No shared-bridge trajectory is
claimed by this no-seed package. No scientific seeds are registered or
executed.

## Honest status

This establishes a runnable sensory boundary for real microphone input and a
bounded shared-substrate A1 construction interface. It does not establish
human cochlear fidelity, calibrated A1 responses, speech perception, phoneme
discovery, or learned auditory concepts.

## Residual scaffolds

- Digital gammatone filters replace traveling-wave mechanics and outer-hair-cell
  feedback.
- Hair-cell adaptation and auditory-nerve firing are reduced phenomenological
  transforms; fiber classes and stochastic release are absent.
- Cochlear nucleus, superior olive, inferior colliculus, and medial geniculate
  are absent.
- A1 population sizes and pathway weights are uncalibrated construction priors;
  proving frequency tuning and inhibitory sharpening on the shared bridge
  requires a separately preregistered seeded gate.
- The adapter turns peripheral auditory-nerve events into current pulses in
  shared-bridge auditory-nerve input neurons. A future bridge-native sensory
  spike clamp could remove this last handoff approximation.
- The input is monaural and not calibrated to physical sound-pressure level, so
  binaural localization and absolute hearing thresholds are out of scope.

The next justified step is sustained live-microphone latency and level testing,
followed by a preregistered shared-bridge A1 calibration with tone-place,
silence, auditory-nerve lesion, and inhibitory-lesion controls. Learned
auditory objects and developmental speech grounding should come only after
that interface preserves timing and level over real room audio.

## Files

- `research/runners/_auditory_cochlea_tonotopic_a1_frontend.py`
- `research/specs/auditory_cochlea_tonotopic_a1_frontend_v1.json`
- `tests/test_auditory_cochlea_tonotopic_a1_frontend.py`

The adjacent construction contract records the boundary, controls, excluded
claims, performance budget, and residual scaffolds.
