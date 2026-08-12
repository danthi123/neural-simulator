---
type: biology
id: coincidence-binding
mechanism: Binding by coincidence -- a synapse is a supralinear detector of the near-simultaneity of two converging signals, so a conjunction can be written locally
status: established
last_verified: 2026-07-31
current_finding: research/findings/2026-07-21-gap2-spiking-learned-binder-6seed-GO-emergence-bar-close.md
current_status: "A per-fact fast-weight J written by a LOCAL coincidence rule (delta or plain Hebbian outer product over role-key/filler phasor pairs) and read through the spiking RF resonate loop reaches 1.000 at P=1..5 roles/fact, permuted-role control 0.000, identical on all 6 seeds -- matching the fixed exact-inverse FHRR ceiling it replaces. The DENDRITIC two-compartment realization named as the ledger surpass is NOT built."
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "a biochemical detector of the near simultaneity of"
    note: "the primitive stated outright -- an individual synapse detects near-simultaneity of input (EPSP) and output (backpropagating spike)"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "signal is greater than the linear sum of the individual"
    note: "the conjunction is SUPRALINEAR (multiplicative-like), not additive -- via Mg2+ expulsion from the NMDA receptor-channel; this is what makes it a binder rather than a summer"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "when a distal stimulus is paired with"
    note: "the same conjunction at dendrite scale (distal x proximal -> plateau) -- the substrate the ledger row's named surpass would use; see dendritic-plateau-coincidence-burst"
implemented_by:
  - research/runners/_gap2_spiking_deltarule_binder_derisk.py
findings:
  - research/findings/2026-07-21-gap2-spiking-learned-binder-6seed-GO-emergence-bar-close.md
  - research/findings/2026-07-17-gap2-adversarial-verify-CONFIRMED-and-content-addressable-wire-in-GO.md
---

# A conjunction can be written by coincidence, locally, without an algebra

**The claim the code must respect.** When a backpropagating action potential is paired with presynaptic
stimulation, the spine Ca²⁺ "signal is greater than the linear sum of the individual" signals, because
depolarization expels Mg²⁺ from the NMDA receptor-channel. The resulting accumulation is, in Kandel's words, "a
biochemical detector of the near simultaneity of the input (EPSP) and output (backpropagating action potential),"
and it is **specific to the activated spine**. Two signals in, a supralinear conjunction out, written where the two
signals met.

That is the biological content of a bind: it does not require an invertible algebra, a global error, or a
transported weight. It requires two signals arriving together at one place.

**Why this row is tagged with it.** The gap#2-depth wall is that the FHRR exact-inverse algebra is a host
idealization and that multi-attribute bundling was not learnable from scratch on point neurons. The exit is a bind
that is *written by coincidence* instead of *computed by an algebra*.

## What is established, and where the ledger's named surpass still stands open

**Established (6 seeds):** a per-fact complex fast-weight `J` written by a purely LOCAL outer-product rule —
`J += (v − Jk)kᴴ/D`, or plain Hebbian `J += v kᴴ/D`, no backprop and no weight transport — installed as the RF
coupling and read by kicking the role key through the committed resonate loop, recovers the filler at **1.000 for
P = 1…5 roles per fact**, with the **permuted-role anti-cheat at 0.000**, identical across all 6 seeds. This is
the fixed-FHRR ceiling, reached by a learned local rule on the spiking substrate, over 788 correlated
stream-cortex phasor codes.

**NOT established, and it is what the ledger row names:** the surpass on that row is *dendritic multiplicative
binding* — a two-compartment cell with apical = role and basal = filler. Nothing here implements that. The result
above is an outer-product fast weight on the RF slice; it is coincidence-written and local, which is why this
entry claims coincidence binding and not dendritic binding. The dendritic version has its own biology entry
(`dendritic-plateau-coincidence-burst`) and its own separate status.

## What this entry cannot catch

No `constraints_config`. The property that matters — that `J` is written from a product of two *locally available*
vectors, with no transported error — is a property of the update expression, not of any numeric default, so
`biology_check --config` cannot see it. A runner could keep every number here and quietly reintroduce a global
signal. The existing anti-cheat that *does* catch it is the permuted-role control (must read 0.000), which lives
in the runner, not here.
