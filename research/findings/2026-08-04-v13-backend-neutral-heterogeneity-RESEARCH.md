---
type: research-finding
status: complete
date: 2026-08-04
mechanism: v13-backend-neutral-gpi-snr-heterogeneity
artifacts:
  - research/findings/raw/v13_tonic_output/replication-numpy.json
  - research/findings/raw/v13_tonic_output/replication-cupy.json
---

# V13 backend-neutral GPi/SNr heterogeneity contract

## Decision

For a paired NumPy-versus-CuPy experiment, one experiment seed must identify
one realized population. The complete initialized population must be
byte-identical before either backend advances it. A shared integer passed to
two different random-number implementations is not the same population and
does not isolate backend behavior.

Biological heterogeneity must remain. The correct design is multiple distinct,
canonically generated populations, with each population replayed on both
backends. Independent backend-native draws can be used as separate biological
replicates, but not as a paired backend comparison.

This contract does not promote the V13 diagnostic observations. The process-
order audit found that seed `1019` was launched from an invalid calibration
selection, so its backend results are procedurally undefined rather than an
earned replication verdict. Seed `1019` remains consumed, seed `1021` remains
sealed, and no observed threshold or current may be retuned without a new
preregistration.

## Why the current comparison is confounded

V13 seeds `cp.random` through the active backend and then calls that backend's
`lognormal` and `normal` functions in `sim/bridge.py`. CuPy documents that its
random implementation differs from NumPy and may change across CuPy major
versions even under the same seed. The repository already established the same
boundary in `2026-07-17-THE-SEED-NEVER-CONTROLLED-THE-SUBSTRATE...md`: fixed
seeds made runs repeatable within a backend, but NumPy and CuPy still produced
different neuron arrays.

At seed `1019`, the realized means differed:

| Effective parameter | NumPy | CuPy |
|---|---:|---:|
| membrane capacitance `C` | `61.3159` | `57.9105` |
| recovery rate `a` | `0.051724` | `0.049449` |
| subthreshold recovery coupling `b` | `2.001880` | `1.982846` |
| post-spike recovery increment `d` | `25.5461` | `24.5923` |

The inhibitory pathway's recorded maximum GABA-A trace was identical, but the
NumPy population fell from `58.7` to `2.0 Hz` (`3.41%`) and the CuPy population
fell from `63.15` to `11.125 Hz` (`17.62%`). Because both the execution backend
and the realized cells changed, these artifacts cannot identify which caused
the suppression difference.

## Biological interpretation of the reduced parameters

The current `IZH2007_GPI_OUTPUT` population is a reduced systems model. Its
heterogeneity distributions are engineering priors around the preset, not
measured GPi/SNr distributions:

- `C` divides the voltage response to all net current. It changes membrane
  timescale and the effect of both intrinsic drive and synaptic inhibition.
- `a` controls how quickly the recovery state follows voltage. It changes
  adaptation and recovery during and after an inhibitory pause.
- `b` controls subthreshold coupling into that recovery state. It changes the
  balance between depolarizing drive and negative feedback before a spike.
- `d` is added after every spike. Lower values generally reduce post-spike
  adaptation and can support more persistent tonic firing.
- The fixed `k`, `vr`, `vt`, `vpeak`, and `c_reset` values also shape threshold,
  gain, reset, and pause recovery. A matched-population artifact must bind them,
  not only the four randomized arrays.

Real SNr tonic firing is produced by interacting channel mechanisms rather than
these four abstract parameters. Direct experiments support a sodium-dependent
background drive, persistent and transient sodium currents, fast Kv3-like
repolarization, and Cav2.2-coupled SK control of the afterhyperpolarization.
NALCN loss substantially slows but does not abolish tonic firing. GABA
suppression also depends on chloride reversal, conductance waveform, input
location, short-term plasticity, each cell's baseline rate, and the phase at
which inhibition arrives. Pallidal and striatal inputs are therefore not
biologically interchangeable.

The repository's existing biophysical fallback finding already concludes that
SNr and GPi/entopeduncular cells need separate explicit-conductance families;
it also records that no primary study supplies one complete adult channel-
density distribution. The present Gaussian/log-normal Izhikevich spread should
remain labeled as a reduced-model test distribution, not biological closure.

Primary evidence:

- [Atherton and Bevan 2005](https://doi.org/10.1523/JNEUROSCI.1475-05.2005) <!--derived-->
  directly measured autonomous SNr firing, persistent sodium drive, and
  Cav2.2-SK control of firing regularity. <!--derived-->
- [Lutas et al. 2016](https://doi.org/10.7554/eLife.15271) found that conditional <!--derived-->
  NALCN loss reduced SNr firing from `21.0` to `11.9 spikes/s`, while cells
  continued to fire. <!--derived-->
- [Connelly et al. 2010](https://doi.org/10.1523/JNEUROSCI.3895-10.2010) showed <!--derived-->
  distinct short-term plasticity at convergent striatonigral and pallidonigral
  inhibitory synapses. <!--derived-->
- [Simmons et al. 2020](https://doi.org/10.1152/jn.00678.2019) showed that <!--derived-->
  pallidonigral conductance barrages alter SNr firing rate and phase, and that
  cells with different baseline rates can respond differently to the same
  inhibition. <!--derived-->

## Executable initialization contract

For every diagnostic or verdict population:

1. Generate the population once with an explicit, version-pinned canonical
   generator. A practical first implementation is NumPy `Generator(PCG64)`.
   NumPy guarantees the fixed-seed PCG64 integer stream, but the generated
   artifact bytes, not regeneration from a seed, are authoritative.
2. Convert once to explicit little-endian, C-contiguous `float32` or integer
   arrays. Store shape, dtype, byte order, generator identity, seed, distribution
   specification, clipping rule, and draw order.
3. Seal and hash all state that can confound the comparison: `C/a/b/d`, all
   fixed Izhikevich parameters, initial `v/u`, intrinsic-current vector, region
   assignments, CSR `data/indices/indptr`, delays, receptor parameters, and the
   complete input schedule.
4. Load the same artifact into NumPy and CuPy. Hash each device array after
   loading and hash a device-to-host round trip before the first simulated
   step. Refuse execution on any mismatch.
5. Record full per-neuron GABA conductance, voltage, recovery, and spike rasters
   for a short diagnostic window. Report the first divergent step and array,
   not only final rates.
6. Use at least two population origins in the diagnostic matrix: a sealed
   NumPy-native draw and a sealed CuPy-native draw, each replayed on both
   backends. If the outcome follows the population, initialization caused the
   V13 difference; if it follows the backend, arithmetic or kernel behavior
   remains causal. A durable verdict protocol should then use one canonical
   generator for all fresh population seeds.

After the diagnostic, a corrected replication needs fresh preregistered seeds.
Each seed should create a different canonical population, but the two backend
arms for that seed must share exactly the same bytes. This preserves biological
variation while keeping backend as the only paired variable.

## Risks of CPU generation and GPU transfer

| Risk | Required control |
|---|---|
| CPU generation becomes a permanent biological shortcut | Restrict it to experiment construction and provenance. Keep it on the scaffold ledger. Learned/developmental heterogeneity in the eventual brain must arise from local growth and homeostatic mechanisms. |
| NumPy or distribution algorithms change | Pin versions and archive the realized bytes. Never reconstruct a verdict population from seed alone. |
| Float64 generation followed by implicit casts changes values | Define one explicit cast to canonical float32 and hash after conversion. |
| Endianness, layout, or partial copies alter arrays | Record dtype/shape/order; require host, device, and round-trip hashes before step zero. |
| Hidden backend RNG still creates topology or initial state | Include every initialized array and CSR component in the artifact; fail if any active path draws after loading. |
| CPU and GPU arithmetic diverge despite identical state | Treat that as the diagnostic result. GPU FMA, math libraries, reductions, and atomics can differ from CPU execution; locate the first divergence before deciding whether exact or tolerance-based trajectory parity is scientifically appropriate. |
| One byte-identical population hides biological variance | Use multiple fresh canonical populations, paired across backends, and report per-population effects. |
| Startup transfer becomes expensive at scale | Cache immutable sealed bundles and transfer once. Performance optimization must preserve the initialization hashes and cannot restore backend-native draws. |

The transfer itself does not make the simulated neurons less biological: the
same parameters and distributions reach the GPU. Its scientific purpose is to
remove a nuisance variable. It becomes architecturally unacceptable only if
host code is later relied on to compute ongoing neural adaptation or
development rather than to construct a controlled experimental specimen.

Implementation references:

- [CuPy random sampling](https://docs.cupy.dev/en/stable/reference/random.html)
  documents backend/version RNG limitations.
- [NumPy PCG64](https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html)
  documents its fixed-seed integer-stream compatibility guarantee.
- [NVIDIA floating-point guidance](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
  explains why CPU/GPU fused operations can round differently.

## Local retrieval record

`python3 tools/rag/check_workflow.py` returned `RAG_WORKFLOW_READY`: both Python
environments, the full index, catalog, repo-relative schema, hooks, refresh
helper, and enabled/active five-minute timer were healthy.

Local RAG was queried before external search:

- `finding`: GPi/SNr heterogeneity, deterministic seeds, backend comparison,
  inhibition, and prior failures. It returned the V13 preregistration and
  replication finding plus the 2026-07-17 unseeded-substrate finding.
- `catalog`: GPi/SNr autonomous firing and inhibitory integration. It returned
  feature catalog A.04, including the local PBR-160 chapter-9 source.
- `paper`: tonic firing, intrinsic currents, and GABA inhibition. It returned
  `TepperAbercrombieBolam-2007-GABAandTheBasalGanglia-PBR160.txt`.
- `kandel`: basal-ganglia output nuclei and inhibitory pathways. It returned
  Kandel 6e chapter 38.

The local sources established the mechanism and exposed the repeated seed
failure. External primary-source search was still needed to verify the direct
channel lesions, inhibitory synapse dynamics, and current software RNG limits.
No registered seed was executed for this research.
