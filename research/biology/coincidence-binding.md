---
type: biology
id: coincidence-binding
mechanism: Binding by coincidence -- a synapse is a supralinear detector of the near-simultaneity of two converging signals, so a conjunction can be written locally
status: established
last_verified: 2026-08-25
current_finding: research/findings/2026-08-25-gap2-deltarule-binder-production-integration-NOT-WIRED.md
current_status: "A per-fact fast-weight J written by a LOCAL coincidence rule (delta or plain Hebbian outer product over role-key/filler phasor pairs) and read through the spiking RF resonate loop reaches 1.000 at P=1..8 roles/fact (6-seed, full 788-code corpus) and to P=48 in a wider single-seed probe, permuted-role control 0.000 throughout -- matching the fixed exact-inverse FHRR ceiling it replaces, but the delta-vs-additive gap the runner was built to demonstrate NEVER OPENS at any tested scale (delta==additive==1.000 from P=1 to P=48, 12x production's real per-fact P=4). The WRITE remains host-numpy (rf_set_complex_weights installs a host-computed matrix; only the READ is on-substrate). Production has NEVER used this mechanism -- every deployed brain bundle checked (scale787/day_9, day_33) runs composer_kind='rf' (the fixed FHRR bind this mechanism targets), unchanged. A separate, more mature alternative for the same production shortcut (SlotBinderComposer -- genuinely on-bridge WRITE, 6-seed GO, covers embedded clauses) is also unwired; its own finding named the wire-in as the next step and it was never executed. The DENDRITIC two-compartment realization named as the ledger surpass is NOT built."
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
  - research/findings/2026-08-25-gap2-deltarule-binder-production-integration-NOT-WIRED.md
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
P = 1…8 roles per fact** (6-seed, the full 788-code corpus; the original 2026-07-21 6-seed artifact ran P=1..5 on a
300-code cap, corrected here), and to **P=48** in a wider single-seed probe, with the **permuted-role anti-cheat at
0.000** throughout. This is the fixed-FHRR ceiling, reached by a learned local rule on the spiking substrate.
**But the delta-rule's namesake advantage over plain Hebbian bundling never appears**: `delta == additive == 1.000`
at every P tested, 1 through 48 — 12x production's real per-fact role count (measured directly from the live corpus,
`bridges/developed/scale787/day_33/facts.json`: 404/404 facts have exactly 4 populated roles). The WRITE
(`build_W`'s `np.outer`) stays host-numpy; only the READ is on-substrate. **Production has never used this
mechanism** — every deployed brain bundle checked still runs `composer_kind="rf"` (the fixed bind this was built to
replace). See `research/findings/2026-08-25-gap2-deltarule-binder-production-integration-NOT-WIRED.md`.

**NOT established, and it is what the ledger row names:** the surpass on that row is *dendritic multiplicative
binding* — a two-compartment cell with apical = role and basal = filler. Nothing here implements that. The result
above is an outer-product fast weight on the RF slice; it is coincidence-written and local, which is why this
entry claims coincidence binding and not dendritic binding. The dendritic version has its own biology entry
(`dendritic-plateau-coincidence-burst`) and its own separate status.

**A more mature, ALSO-unwired sibling exists for the same production shortcut:** `SlotBinderComposer`
(`research/runners/slotbinder_composer.py`) writes via a genuine on-bridge Hebbian step (not host-numpy), is
6-seed GO with anti-cheats, and covers the full deployed FHRR capability set including depth-1 embedded clauses —
strictly more complete than this entry's mechanism. Its own finding named the production wire-in as the next step;
it was never executed, likely blocked by an unmeasured scale question (its bridge size scales with `max_facts` via
a dense slot→filler pathway). That wire-in, not a WRITE-port of this entry's mechanism, is the higher-value next
rung for retiring the FHRR fixed bind.

## What this entry cannot catch

No `constraints_config`. The property that matters — that `J` is written from a product of two *locally available*
vectors, with no transported error — is a property of the update expression, not of any numeric default, so
`biology_check --config` cannot see it. A runner could keep every number here and quietly reintroduce a global
signal. The existing anti-cheat that *does* catch it is the permuted-role control (must read 0.000), which lives
in the runner, not here.
