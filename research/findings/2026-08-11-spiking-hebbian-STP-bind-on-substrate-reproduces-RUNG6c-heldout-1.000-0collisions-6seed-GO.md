---
type: finding
status: go
date: 2026-08-11
mechanism: the variable-binding fast-weight role->filler BIND realized as HEBBIAN short-term potentiation on a real spiking SimulationBridge — barcode input + K slot pools + shared FS, barcode->slot synapses PLASTIC via the rate-window Hebbian coactivity rule; the bind lives in cp_connections.data (real synapses), NOT a host numpy matrix
lane: emergence engine / working memory (closes the WM-GO residual (c): "the fast-weight BIND is host numpy; its spiking-STP realisation is a banked next rung")
verdict: 6-SEED GO — held-out NOVEL recovery 1.000 / 0.000 collisions every seed (reproduces the RUNG6c host binder GO as a genuine synaptic mechanism); all lesion teeth bite. The literal Mongillo STP facilitation (cp_stp_u/x) is the honest NEGATIVE (frozen-Hebbian arm 0.000-0.031, plus RUNG6d non-selectivity + a verified tau_F mismatch). Scope: slot ALLOCATION is host (as RUNG6c's free-counter); the STORAGE + RETRIEVAL are spiking/synaptic.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_spiking_stp_bind_derisk.py
artifacts:
  - research/findings/raw/_spiking_stp_bind/stp_bind_6seed.json
instrument: a real SimulationBridge (SIM_BACKEND=numpy, 212 neurons: 48 barcode + 4x36 slot + 20 FS). BIND = drive a barcode + CLAMP its host-allocated target slot (the "post" of pre x post) -> the substrate's rate-window Hebbian coactivity (cp_hebb_coactivity_trace) potentiates barcode->slot from init 30 toward 4000 in cp_connections.data. RETRIEVE = re-present the barcode ALONE (learn off), read the winner slot from cp_firing_states. Metric = the RUNG6c held-out-novel recovery + collision metric. Anti-cheats via tools.verdict.Verdict; backend/seed asserted (tools.lab.assert_backend + cfg.seed substrate-seeding verified; hebbian_max_weight clamp checked).
---

# The spiking Hebbian short-term-potentiation BIND reproduces the RUNG6c host binder on a real substrate — held-out NOVEL 1.000 / 0 collisions, 6-seed GO; the literal cp_stp_u/x STP facilitation is the honest negative

The variable-binding working-memory GO
(`2026-08-11-variable-binding-working-memory-gated-slot-surpasses-HTM-heldout-1.000-vs-0.000-6seed-GO.md`) named its
open residual precisely: the content-agnostic Hebbian fast-weight BIND was HOST numpy. RUNG 6d
(`2026-07-13-RUNG6d-spiking-STP-binder-needs-HEBBIAN-not-presynaptic-6seed-GO.md`) resolved the MECHANISM question in a
numpy STP-dynamics model (HEBBIAN pre x post potentiation binds; presynaptic Tsodyks-Markram STP does not) and scoped
the on-substrate build. This de-risk executes that build on a real `SimulationBridge` and answers: does the
spiking-synaptic bind reproduce the RUNG6c GO — 0 collisions / clean recovery on HELD-OUT NOVEL entities minted at test?

## Result — 6-seed GO (`research/findings/raw/_spiking_stp_bind/stp_bind_6seed.json`)

<!--derived-->
Every seed (42/43/44/100/101/102): held-out NOVEL recovery **1.000**, collisions **0.000**; known-pool control 1.000
(no memorisation gap, as expected for a content-agnostic mechanism); chance 0.250. **6/6 GO.** The bind weight moved
from init ~30 to ~71-212 in `cp_connections.data` on every seed (the fast weight is a real synaptic change).

**All lesion / control teeth bite** (the GO rests on these, not the headline):
- **cp_stp_u/x-only (freeze Hebbian, STP still on)** -> recovery **0.000-0.031** (<= chance). The substrate's presynaptic
  STP facilitation ALONE does not store/retrieve the bind — the honest negative for the literal Mongillo variable.
- **permuted-binding** (score against a shuffled slot map) -> **0.156-0.292** ~ chance. The bind is content-specific.
- **MERGE** (identical barcodes) -> recovery **0.26-0.43** ~ chance, collisions **0.51-0.96**. Without individuation the
  retrieval collapses — the bind is carried by the barcode content, not by capacity.

## Is the bind genuinely spiking / synaptic (not host numpy)? Yes.

<!--derived-->
- The stored association lives in **`cp_connections.data`** (real synapse weights), written by the substrate's own
  rate-window Hebbian coactivity as barcode and slot neurons fire coincidentally (`reaches`: before 30 -> after ~150).
- RETRIEVE reads only **`cp_firing_states`** (spikes); the binder object holds **no numpy `W`** (asserted). No host
  fast-weight matrix is used in the read-out.
- backend asserted numpy; `cfg.seed` verified to actually seed the substrate (same seed -> identical firing thresholds);
  the `hebbian_max_weight` (4000) clamp exceeds every design weight (the plasticity-bound trap).

## Honesty / scope (brain-based-only)

<!--derived-->
- **SPIKING + synaptic + load-bearing:** the BIND (write) and RETRIEVE (content-addressable read) — the substrate's
  Hebbian synaptic plasticity does the potentiation; spikes do the read. This de-shortcuts the WM-GO residual (c).
- **HOST (inherited from RUNG6c, unchanged):** slot ALLOCATION (which fresh slot a new entity takes) is a host counter,
  and binding drives the target slot as a teaching clamp (the "post"). RUNG6c's `HebbianBinder` allocates with a `free`
  counter identically; this build does not add a shortcut beyond RUNG6c, it moves the STORAGE onto real synapses.
- **The literal `cp_stp_u`/`cp_stp_x` (Tsodyks-Markram / Mongillo facilitation) is an HONEST NEGATIVE for this role, on
  two counts:** (1) presynaptic non-selectivity (RUNG6d: u rises on ALL barcode->slot synapses the barcode drives ->
  0.999 collisions), reproduced here as the frozen-Hebbian arm's collapse; (2) the WINDOW is too short — Mongillo et al.
  (2008) set the facilitation time constant **tau_F = 1500 ms** (verified at a secondary source, Frontiers fnint
  2022.972055, quoting Mongillo 2008: "tau_d = 200 ms and tau_f = 1,500 ms in agreement with ... Mongillo et al.
  (2008)"), whereas `sim`'s `stp_tau_f` defaults to **50 ms** (config.py:686) — ~30x too short to bridge a WM span.
  Additionally, STP facilitation is a BOUNDED multiplicative transmission gain, structurally unable to lift a
  subthreshold (init-30) synapse to firing, which the Hebbian potentiation (a 133x durable weight range) does. The bind
  therefore uses HEBBIAN short-term potentiation (durable weight, reset per narrative = RUNG6c's default), NOT the
  cp_stp_u/x facilitation variable.
- **Named next rungs (unchanged from the WM-GO ladder):** the ROLE-based (syntactic, not barcode-class) write gate; the
  on-substrate spiking three-factor DA-gated gate; then wire this spiking bind + the D3 slow-NMDA HOLD + the gate into
  the emergence stream (this de-risk isolates the BIND; the D3 hold carries persistence across the span, so this bind
  needs no long tau_F of its own). No `sim/` edit was required — reuse-by-import of the brain-region framework, the
  rate-window Hebbian rule, and FS-WTA.
