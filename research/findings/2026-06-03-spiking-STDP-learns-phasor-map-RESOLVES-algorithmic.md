# Real-weight spike-timing plasticity learns + composes the phasor map — RESOLVES (algorithmic) — 2026-06-03

**One line:** The load-bearing open question behind unifying the production substrate onto phasor FHRR —
can **real-valued synaptic weights**, shaped by **spike-timing plasticity**, learn the input→phasor-code
map and still **compose**? — RESOLVES at the algorithmic level: at D=512, N=32 concepts, retrieval 1.00 and
learned-code bind/unbind 0.95, with anti-cheat controls at chance.

This is the pre-registered minimal first experiment from the substrate-unification design note
(`docs/plans/2026-06-03-phasor-substrate-unification-design-note.md`). It is **not** the trivial linear
probe: the earlier Hebbian result used a *complex* outer product (`W = code ⊗ cue^H`), but biology gives
**real scalar synapses** — a real weight scales, it cannot rotate phase. So the faithful question is whether
real-weighted synapses with a spike-timing rule can drive each output neuron to an arbitrary target phase,
for many concepts, on one shared weight matrix (interference is the real unknown).

## The faithful (phenomenological) model

- A phase-coded spike at phase φ is the phasor e^{iφ} (Frady-Sommer 2019).
- Output neuron j fires at **phase = angle( Σ_i W[j,i] · e^{i·cue_i} )**, with **W real** — the angle of the
  real-weighted input population vector (a first-order model of the integrate-and-fire spike time).
- Classic asymmetric STDP: for each concept, potentiate synapse i→j when the presynaptic cue phase precedes
  the teacher-forced postsynaptic code phase, depress when it follows. W is the sum over concepts.

## Result (frozen probe, `research/findings/raw/_spiking_stdp_phasor_learn_probe.py`, 5 seeds)

| N | retrieval | learned bind/unbind | untrained control | shuffled-pairing control |
|---|---|---|---|---|
| 8 | 1.00 | 1.00 | 0.10 (chance 0.12) | 0.10 |
| **32** | **1.00** | **0.95** | 0.03 (chance 0.03) | 0.04 |

**Verdict: RESOLVES** — retrieval ≥ 0.90 and learned bind/unbind ≥ 0.80 at N=32, both controls at chance.

### The capacity lever (D), measured

Composition of *learned* (noisy retrieved) codes is dimension-limited, and the D lever closes it exactly as
the SNR analysis predicts:

| D | N=32 retrieval | raw learned-compose | cleanup-then-compose (standard VSA) |
|---|---|---|---|
| 256 | 0.96 | 0.65 | 0.91 |
| 512 | 1.00 | 0.95 | 1.00 |
| 1024 | 1.00 | 1.00 | 1.00 |

At D=256 retrieval already passes but raw composition (0.65) needs either more dimension or the standard
clean-up-then-compose pipeline (0.91). At D≥512 raw learned-code composition passes outright.

### Anti-cheat — the decisive control

The **shuffled-pairing** control is the load-bearing check: train the weights on a *permuted* (cue, code)
assignment, then test the *true* pairing. It decodes at chance (0.04 vs 0.03) — so the weights learned the
*specific* cue→code pairing, not a generic cue/code-structure artifact. The untrained-weights control is also
at chance. Genuine, pairing-specific learning.

## The readout is the genuine resonate-and-fire spiking readout (confirmed)

The "angle of the real-weighted population vector" is **not** an arbitrary proxy: it is exactly the phase a
resonate-and-fire phasor neuron fires at when its weighted phasor inputs sum (Frady-Sommer 2019). Confirmed
directly — the learned-weight drive was converted to spikes and read out through the project's genuine
resonate-and-fire substrate (`rf_resonate` + spiking phasor cleanup against a spiking codebook):

| seed | numpy-angle retrieval | genuine-spiking retrieval (rf substrate) |
|---|---|---|
| 0 | 16/16 | 16/16 |
| 1 | 16/16 | 16/16 |
| 2 | 16/16 | 16/16 |

So the learned representation + readout are spiking-faithful, not merely a numpy convenience.

## Honest scope — what RESOLVES and what does not

**Does:** the *algorithmic* learning question, with a spiking-faithful representation + readout. Real-valued
weights + a spike-timing potentiation rule learn the input→phasor-code map for 32 concepts on a shared
matrix; the learned codes compose (bind/unbind); the readout is realized in the genuine resonate-and-fire
substrate; anti-cheat controls are clean. The biological constraint that broke the naive analogy (real, not
complex, weights) is satisfied.

**Does NOT:** two fidelity details remain for the full implementation. (1) The plasticity here is **computed
from phase/timing differences** (the STDP rule), not run as an **online spike-driven plasticity loop** in the
substrate (it is the same rule, applied in closed form rather than accumulated spike-by-spike). (2) There is
no explicit **membrane ODE / refractory / conductance** dynamics — the resonate-and-fire readout is used in
its steady-state phasor form. Both are implementation fidelity, not open scientific risk: the rule and the
readout are each individually faithful; wiring them into one online spike-driven loop is the **next rung**
(a writing-plans engineering arc).

## Where this leaves substrate unification

Every cheap-first-testable axis is now de-risked (all committed 2026-06-03):

| Axis | Status |
|---|---|
| Diversity (320 concepts) + SVO composition | 1.00 |
| Inter-code correlation (common-mode + clustered) | 1.00 |
| Nesting (multi-modifier, recursive clause; incl. spikes) | RESOLVES |
| Agent at production-diversity scale (120 concepts) | ~96% |
| Linear-Hebbian learning analog | 1.00 |
| **Real-weight spike-timing learning + composition (algorithmic)** | **RESOLVES (this finding)** |

The remaining step is the **full membrane-level spiking implementation** of the learned binding + cleanup
across the production path — a writing-plans engineering arc, with the science de-risked. Whether to invest
in that migration is the owner's strategic call; this finding removes the last *scientific* uncertainty from it.

## Verdict

**RESOLVES (algorithmic).** Real-weight spike-timing plasticity learns and composes the phasor map at N=32
(D=512), anti-cheat clean. The substrate-unification path has no remaining cheap-first scientific blocker;
the open work is the full membrane-level spiking implementation.
