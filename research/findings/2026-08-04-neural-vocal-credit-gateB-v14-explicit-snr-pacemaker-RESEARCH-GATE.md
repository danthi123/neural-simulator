---
type: research-gate
status: selected
date: 2026-08-04
mechanism: neural-vocal-action-credit-v14-explicit-snr-pacemaker
---

# Gate B v14: explicit SNr pacemaking before selector retuning

## Decision

V14 will replace V13's constant GPi/SNr intrinsic-current scaffold with a
region-scoped SNr conductance family. The minimum biological bundle is a
NALCN-like background cation conductance, persistent sodium drive, and
Cav2.2-coupled SK recovery. Ih may be present for hyperpolarized recovery, but
the intact cell may not depend on Ih for its baseline tonic firing.

SNr is the first target because direct causal and quantitative evidence exists
for this family. GPi/entopeduncular neurons remain a separate future family;
V14 will not describe one parameter set as a generic `GPi/SNr` cell.

V13 remains a sealed reduced-model result. Its physiology evidence and
performance NO-GO are historical controls, not evidence for or against V14.

## Role in the whole brain

The SNr population must continuously inhibit downstream motor routes while the
brain is at rest, pause selectively when a direct-pathway action wins, and
recover after inhibition without a host reset or injected tonic current. Its
own membrane conductances must create and regulate the tonic output so later
action eligibility belongs to the route that actually executed.

The cell gate is necessary but not sufficient. Passing it does not establish a
continuous selector, local reward credit, convention learning, language, or
conversation.

## Evidence boundary

The detailed source review is
`2026-08-04-gpi-snr-autonomous-pacemaking-biophysical-fallback-RESEARCH.md`.
Its direct constraints include:

- NALCN deletion reduced juvenile mouse SNr firing from `21.0 +/- 1.3` to
  `11.9 +/- 0.9 Hz`, so NALCN is load-bearing but not the only clock;
- persistent sodium activates below spike threshold and its blockade can stop
  autonomous firing;
- Cav2.2-to-SK coupling controls afterhyperpolarization and regularity; and
- HCN blockade did not significantly change baseline SNr firing in the direct
  preparation, although Ih remains relevant during hyperpolarization.

No located primary study supplies a complete adult SNr density vector for all
of these channels in one cell. V14 must therefore identify a constrained
parameter ensemble, retain preparation metadata, and label model-derived
density ranges as search priors rather than measurements.

## Required implementation boundary

The engine facility must be default-off and population-scoped on the shared
simulation bridge. Conductance maxima and channel state must be device arrays;
the host may construct them before a run and observe them afterward, but may
not calculate or inject the per-step pacemaker current.

The first implementation must provide:

1. a voltage-dependent NALCN-like ohmic current with a separately named passive
   leak;
2. persistent sodium activation and slow inactivation;
3. a Cav2.2-like calcium current, intracellular calcium state, and calcium-
   activated SK current;
4. optional Ih that can be lesioned independently;
5. region-local conductance maxima and state with zero allocation and zero
   active-path work when no region requests the family;
6. checkpoint continuation of every dynamic gate and calcium state;
7. NumPy and CuPy support without per-step device-to-host synchronization; and
8. an explicit guard for any fast path that has not incorporated the bundle.

An Izhikevich cell augmented only with NaP and Ih may be useful as an engine
derisk, but it remains a reduced scaffold: it cannot pass this replacement
gate or close the scaffold-ledger row because it omits the required NALCN and
Cav2.2-SK causal mechanisms.

## Staged gate

### Stage A: equations and state

Unit tests must establish current direction, voltage dependence, bounded gate
state, calcium accumulation and decay, SK activation, finite 1 ms updates,
default-off equivalence, region isolation, malformed-checkpoint rejection, and
exact same-backend checkpoint continuation. GPU tests must establish behavioral
parity and absence of host synchronization.

Stage A uses synthetic voltages and cannot select biological parameters.

### Stage B: preparation-matched SNr ensemble

A sealed calibration may search only the preregistered parameter ranges from
the fallback review. It must fit isolated-slice targets separately from the
later `40-80 Hz` system contract. Calibration must score waveform and causal
features together: tonic rate and regularity, AP/AHP shape, NALCN loss ratio,
NaP lesion, Cav2.2/SK lesion, HCN baseline neutrality, inhibition, release, and
recovery.

Promotion requires an ensemble of passing parameter sets, not one optimum.
Independent compensation that rescues rate while reversing a lesion signature
is a failure.

### Stage C: shared continuous selector

Only a Stage-B family may replace the constant-current output cells in a fresh
continuous-selector construction. The selector must then pass repeated actions,
competitor suppression, autonomous termination, causal pathway lesions,
checkpoint, backend, and consumer-hardware gates before reward learning reopens.

## Stop rules

Stop and return to research if the implementation requires runtime host drive,
one global conductance profile for unrelated regions, hidden burn-in, relaxed
lesion directions, result-selected seeds, or a parameter outside the filed
range. Stop Stage B if no ensemble passes all causal signatures; do not widen
ranges or add channels after seeing verdict data. Stop Stage C if tonic-cell
replacement passes in isolation but does not remain stable in the uninterrupted
selector.

## Next exact action

File and seal the Stage-A implementation preregistration with fresh seed
partitions and source hashes. Then implement the population-scoped channel
state and focused CPU tests. No physiology sweep, GPU calibration, held-out
seed, or V13 Stage-1 seed may be opened by the implementation work.

