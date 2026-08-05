---
status: complete
type: research-finding
lane: gateb-v14-source-constrained-identification
date: 2026-08-05
claim_check: synthesis
artifacts:
  - research/specs/v14_snr_stageB_kinetic_identification_partition_v1.json
  - research/specs/v14_snr_stageB_primary_figure_asset_manifest_v1.json
---

# Kinetic parameter authority: source vectors are points, not ranges

The located primary sources do not establish continuous biological uncertainty
bounds for the microscopic transition constants in the Khaliq, Balbi, Labro,
or Desai models. They provide fitted parameter vectors, aggregate measured
channel properties, and a small number of deliberate perturbations. Treating
componentwise extrema or reported output uncertainty as microscopic parameter
bounds would be an engineering search choice, not a biological claim.

## What the sources authorize

- Khaliq/Raman supplies one exact 13-state vector in ModelDB 48332,
  `rsg.mod`, commit `c96405173a17d18999d2a8d63d40899a76d02bdf`, file SHA-256
  `1a3382714bd0962665ec31f7dfac2aa3a9e403a5e3d23e29851afec232c4543e`.
- Balbi supplies nine complete vectors for human Nav1.1 through Nav1.9 on the
  same six-state graph in ModelDB 230137, commit
  `815a1d7762d0cdccc3a3c6e6bed3a678d15888e4`. These are different isoforms,
  not an uncertainty interval around Nav1.6. The Nav1.6 file SHA-256 is
  `69931ced1587944070edb3169a865e9e3e2a42f715b19a8b7b57e72e831ba71d`.
- Labro supplies one four-state Kv3.1b vector and deliberate beta-l values of
  0.4, 0.6, and 1.8 per ms to demonstrate hooked-tail sensitivity. The
  perturbations are discrete sensitivity conditions, not a confidence range.
  The supplement SHA-256 is
  `d0eb5e8d565d715588543120739fc4d82fff7629f1064b6e5d87d50f9c41a882`.
- Desai supplies one Kv3.3 kinetic vector. The control and PKC current weights
  `(0.23, 0.77)` and `(0.9, 0.1)` are distinct biological conditions, not
  endpoints of a healthy continuous range. The primary XML SHA-256 is
  `4c04b05905e8a976a3590b0d19302e8ec4c03841ff8760a67f07dc39e1da9878`.

Goldfarb's same-family 13-state sodium vector is a useful discrete structural
comparison. Its FHF-knockout modifications are pathological perturbations and
cannot define healthy variability. Primary article:
`https://pmc.ncbi.nlm.nih.gov/articles/PMC2974323/`; retrieved HTML SHA-256
`328ad355464498d913a5bebb924e2a150ee27629910e4461f4f7416f023876ca`.

## Output constraints remain biological

Measured aggregate properties constrain model outputs rather than unique
transition rates. Examples include Labro's gating-charge midpoint and apparent
valence, Desai's inactivation midpoint, recovery half-life, and deactivation
constants, and the Ding SNr activation, inactivation, rise, recovery, and
deactivation statistics already sealed by the project. The project's mean
plus or minus two SEM acceptance convention does not imply that an underlying
microscopic parameter lies in the same interval.

## Prohibited interpretation

The following may be explicitly preregistered as search trust regions or
sensitivity tests, but never labeled biological bounds:

- interpolation between Khaliq and Goldfarb vectors;
- componentwise minima and maxima across Balbi isoforms;
- Labro midpoint, valence, or beta-l intervals;
- any continuous Desai interval;
- temperature or Q10 transfer envelopes;
- profile-likelihood intervals inferred from Ding outputs.

## Search order

Run a discrete source-vector screen before any continuous search: complete
Balbi isoforms, the Goldfarb same-family vector, and the source-declared Labro
beta-l conditions. A later continuous pilot may use a clearly labeled Labro
trust region only after waveform targets and identifiability diagnostics are
available. Any successful result is an SNr-fitted estimate, not a discovered
biological parameter range.

Khaliq, Balbi, and Desai remain discrete until complete-waveform fitting,
profile analysis, multiple independent starts, and held-out protocols show
which parameter combinations are identifiable. Failure of the discrete screen
does not authorize widening ranges until something passes.
