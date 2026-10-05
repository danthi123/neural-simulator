# fb-spiking-witness, the synaptic (relay) write: diagnosis and next measurement (Claude, 2026-10-04)

Dev seed 7 only, numpy, Claude's local scans (session 6). None of this is evidence; it fixes what the next build is.
The scan scripts are in `docs/drafts/fb-spiking-witness-relay2/scans/` (run from the repo root with `SIM_BACKEND=numpy python3 docs/drafts/fb-spiking-witness-relay2/scans/grid.py 7`). Context: the registered
relay ops 0-3 failed every case on all 8 dev seeds, so the T18 PASS runs in read-out mode with PARTIAL credit
(RULINGs t18-credit, fb-verify-go). The organ changes below are opt-in constants; the registered GRID is untouched,
and ops 0 and 4 reproduce the pool's committed sweep byte for byte on dev seed 1.

## What the relay-mode failure is made of (from the pool sweep at 40c43d5)

1. **The AND gate leaks.** With presence at 0 spikes, world input alone fires the relay: 56 such writes across ops
   0-3, up to 961 relay spikes. Witnessed and unwitnessed writes differ little (seed 7: about 40 vs 58-65 at op 0).
2. **No written belief survives the hold** (true-belief ends with every slot at 0). In the AMPA relay this is
   confounded by defect 1; see defect 2 below.

## Defect 1, the AND gate: fixed on seed 7 by the dendritic AND plus a bias (mode `relay_nmda`)

`world_k -> relay_k` arrives on the engine's slow, Mg2+-blocked NMDA alone (`exc_receptor="nmda_slow"`, no fast AMPA
onset), so it cannot fire a relay cell at rest; presence's AMPA depolarization relieves the block (Major, Larkum &
Schiller 2013). **Corrected 2026-10-04:** a first scan at no bias read "0-1 spikes unwitnessed vs 185-531 witnessed".
That was presence on vs off, not a coincidence. A presence-alone arm showed presence driving every location's relay
almost equally (124 / 105 / 104 / 106). With the right metric (`scans/and_scan.py`: target vs off-target relay on a
witnessed move, and the target on an unwitnessed one), the AND holds at `RELAY_BIAS_PA -250`, `W_PRES_RELAY 3.0`,
`W_WORLD_RELAY 2.0`. There, a witnessed move fires its target relay 99 spikes against an off-target mean of 34, and
an unwitnessed move fires 0. In the full battery at that point (`W_RELAY_BELIEF 48`, eviction GABA_A), witnessed
moves fire the target relay 179-202 spikes and unwitnessed ones 0-2 (29 in two_unseen_moves). The placing write is
fully selective: world slot 0 only (933 spikes), relay 0 only (94), belief slot 0 only (232).

## Defect 2: a synaptically written belief does not persist (the open problem)

- At that operating point the battery passes 1 of 10 (eviction GABA_B: 0 of 10).
- The write ignites the target slot, but after `HOLD_STEPS` the query reads 0 on every slot.
- Two earlier readings were artifacts of bad operating points, not store defects:
  - "the store spreads to every slot" came from relays that presence alone drove at every location;
  - "the belief cells sit near 0 mV" (`scans/measure_ei.py`) came from relay -> belief weight 1000.
- **Read-out mode never meets the persistence problem, because its write is host-actuated three ways:**
  - `_restore_slice(belief_all)` resets the belief store's neuron state to a quiescent snapshot (a host clear);
  - a 5000 pA helper current is injected into the target slot;
  - the host threshold read sets the transmission gate.
  The fb-spiking-witness finding's caveat now names all three.

## Next measurement (before any new mechanism; dev seeds only)

At the defect-1 operating point, on seed 7 and then seeds 1-8:
1. The belief slot's rate step by step from write onset through `HOLD_STEPS` (`scans/measure_ei.py`, extended to
   the hold), next to the same trace in read-out mode. Does the slot ignite and then decay (too weak to cross the
   attractor's bistability), or is it killed (eviction or the shared FS pool)? Also report the voltages in mV, so an
   invalid state is visible.
2. The same at `W_RELAY_BELIEF` 48, 100, 200 and 400, with `scans/and_scan.py` re-run at each value: the AND must
   hold while the write strengthens.
3. With that answer Claude picks the mechanism. Candidates in biological order: (a) a write that lasts the
   attractor's ignition time (the relay's own NMDA tail, `RELAY_BELIEF_RECEPTOR="nmda_slow"`, is in place for this);
   (b) slot-specific perisomatic inhibition for eviction instead of one shared pool; (c) a disinhibitory write.
   Then the AND point is re-checked on seeds 1-8 and the method goes to a prereg with a new GRID index. The
   registered GRID 0-7 is never edited.

The relay's credit target stays FULL (a write the host neither reads, clears nor drives). Read-out mode keeps its
PARTIAL credit in the meantime.
