---
type: finding
status: live
date: 2026-06-05
mechanism: opponency
---

# (B) opponency linear-glue — rate-coded common-mode removal is an SNR WALL (3 de-risks + analysis) — 2026-06-05

**Capstone of the opponency arc. Verdict: the `onoff(bon−boff)` opponency — common-mode removal of a SMALL signed
difference of strongly-correlated channels — does NOT reach unbind parity in rate-coded spikes, by ANY of three
independent mechanisms, for ONE fundamental reason (the research synthesis's SNR prediction). The escape is
structural (Option A: spiking-phasor FHRR) or the honest boundary (Option D). Both DEEP shortcuts (A cleanup, B
storage) remain CLEARED; this LAST linear-glue op is the only open question.**

## The object (recap)
`CoreSimComposer.bind_fact`: per-role spiking coincidence binds give `(o,f)`; numpy superposition `bon+=o; boff+=f`
(IN-NETWORK FAITHFUL — per-channel cos 0.97); numpy opponency `onoff(bon−boff)` (common-mode removal). The
opponency is the blocker: `bon,boff` are strongly correlated (common-mode cos **0.89**), so the signed difference
`s=bon−boff` is SMALL, and a rate read of two separately-summed non-negative channels loses it. The research
synthesis (`2026-06-05-spiking-opponency-literature-synthesis.md`) reframed it: the ON/OFF split is a TRANSPORT
code; the real object is the signed `s`; **biology removes the common mode in the ANALOG stage BEFORE spiking
because rate codes can't (Kandel Ch 22 p543); the subtraction amplifies noise ~4.3× at ρ=0.89.**

## Three independent rate-coded mechanisms, all NEGATIVE, all the same wall

| mechanism | what it does | signed read | per-role unbind recovery | finding |
|---|---|---|---|---|
| **simple accumulator + lateral-inhibition opponency** | read both channels, subtract | signed cos **0.41** | 0.46–0.69 (min 0.46) | `2026-06-05-B-innetwork-superposition-NEGATIVE.md` |
| **gated NEF signed-value integrator** | subtract in the represented value, linear decode | signed cos **0.90** (aggregate!) | **0.077** (M-INVARIANT) | `2026-06-05-B-nef-opponency-NEGATIVE.md` |
| **per-dim bipolar threshold / WTA** | per-dim ON/OFF winner = sign | sign_agree **0.617** | **0.385** (hardening → WORSE) | this doc |

GATE for all = per-seed unbind recovery == 1.000 (numpy parity). None clears.

## The bipolar / WTA result (the research's recommended CHEAP fix — Option B)
Cheap-first numpy (`_b_bipolar_threshold_numpy_probe.py`, 3 seeds): binarizing the bound vector to a per-dim sign
(`sign(s)`) preserves the VSA unbind at **100% (3/3)** when the sign is the TRUE sign — `sign(s)` is only cos 0.71 to
the graded `s` yet recovers perfectly (the cleanup reads the SIGN PATTERN). **So the VSA tolerates a sign; the only
question is whether a spiking circuit can produce `sign(true s)`.** It cannot, here:

Spiking WTA w_opp-hardening sweep (`_b_bipolar_wta_spiking_probe.py`, seed 42):

| w_opp (mutual inhibition) | bipolar recovery | graded baseline | sign_agree | signed cos |
|---|---|---|---|---|
| 200 (soft) | **0.385** | 0.308 | 0.617 | 0.378 |
| 800 | 0.154 | 0.154 | 0.585 | 0.289 |
| 2000 | 0.077 | 0.000 | 0.359 | 0.133 |
| 5000 (hard) | 0.077 | 0.000 | **0.083** | −0.037 |

Hardening the WTA makes it **monotonically WORSE** — symmetric mutual inhibition, cranked, silences BOTH
populations (the Rutishauser-Douglas-Slotine instability: a stable WTA needs α>1 self-excitation + asymmetric
inhibition, which symmetric soft inhibition lacks). The best point is the soft one, bipolar 0.385.

### Why no WTA architecture can beat 0.385 (the load-bearing analysis — why a "proper" WTA was NOT built)
A per-dimension WTA's output is, by definition, `sign(drive_on[k] − drive_off[k])` — it reports WHICH channel's
(window-integrated) drive is larger. The drives are the accumulator's channel reads `(bon'[k], boff'[k])`. **No WTA
dynamics — self-excitation, asymmetric inhibition, latching — change WHICH channel is larger; they only change how
fast/cleanly the winner settles.** So a stable Rutishauser WTA's sign output == `sign(s_acc)` == the bipolar 0.385
already measured. The only lever that could lift it is more temporal integration of the differential — but the
accumulator already integrates over the run window, the NEF M-sweep is M-INVARIANT (0.903 flat over an 8× neuron
range → a representational wall, not an averaging-N wall), and the in-network late-window read was worse. Building
the self-excitation circuit would re-measure 0.385. **The wall is the differential's SNR, which is fixed by the
representation, not the readout circuit.**

## The single root cause (all three failures)
The signed value `s = bon − boff` is a SMALL difference of two channels carrying a LARGE common mode (cos 0.89).
Every rate-coded read of `bon` and `boff` carries the full common mode + spiking noise; recovering `s` (by
subtraction, NEF decode, or per-dim sign) inherits noise that swamps the small signal. This is the research
synthesis's exact SNR prediction and matches biology's solution: **the retina removes the common mode with GRADED
signals before action potentials (Kandel p543) precisely because spike rates cannot.** It is a real, biology-
translatable boundary, not a tuning miss.

## What this does and does NOT change
- **Both DEEP shortcuts stay CLEARED.** (A) NEF cleanup and (B) Crawford weight-store are GO at D=2048 multi-seed,
  unaffected. The composer's NONLINEAR core (bind, store, unbind, cleanup) is fully spiking.
- **The two LINEAR-glue ops (superposition + opponency) stay numpy, DISCLOSED** (the audit boundary, n=111). The
  superposition is genuine in-network (cos 0.97); only the opponency's small-signal read resists rate-coded spiking.
- **No `sim/` edits.** All probes reuse `CoreSimComposer` / `build_bind_bridge` / the accumulator by import.

## The decision (SURFACED to owner — major-arc milestone)
The research's two remaining options, after all rate-coded mechanisms hit the SNR wall:
- **Option A (structural escape): pivot the bound-vector representation to spiking-phasor FHRR** (Frady-Sommer 2019 /
  Orchard-Jarvis 2023; the repo has the numpy reference). Unit-magnitude, info in PHASE — there is NO common mode and
  NO small signed difference, so the opponency simply does not exist. Readout SNR ≈ 2N/M (a dimension dial). Also
  gives the F=3 two-attribute resonator (which the ±1 scheme provably can't do) for free. **Cost: a big rework — the
  whole bind/store/unbind move from the ±1 Hadamard to phase/timing. Per the standing rule, this is the owner's call,
  not auto-launched.**
- **Option D (honest SNR boundary): accept it.** Rate-coded common-mode removal of a small correlated difference is a
  fundamental spiking boundary (biology does it analog pre-spiking). Both deep shortcuts already cleared; the two
  linear-glue ops stay numpy DISCLOSED; pivot to the higher-value spine item (the fully-grounded run). The honest
  negative IS the scientific deliverable.

## Artifacts
- `research/findings/raw/_b_bipolar_threshold_numpy_probe.py` (+ `.json`) — cheap-first numpy: binarization preserves VSA 100% 3/3
- `research/findings/raw/_b_bipolar_wta_spiking_probe.py` (+ `.json`) — spiking WTA w_opp sweep (NEGATIVE, hardening worse)
- prior: `_b_innetwork_superposition_probe.py`, `_b_nef_opponency_probe.py`; syntheses `2026-06-05-spiking-opponency-literature-synthesis.md`
- Backend: CuPy / RTX 3090. NO `sim/` edits.
