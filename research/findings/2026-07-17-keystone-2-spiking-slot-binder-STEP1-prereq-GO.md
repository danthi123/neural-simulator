# Gap #2 spiking slot binder — BUILD STEP 1 (slot-separation prerequisite): GO, + the multi-slot-coexistence challenge precisely identified (2026-07-17)

**Per `2026-07-17-keystone-slot-binder-research-gate.md` #1. Composes the EMERGE-41 spiking competitive pooler (`FSWTAProbe`). CPU/numpy probe; the gate's #1 build begun.**

## Step 1 result (3-seed, drive uniform(0,6) = EMERGE-41's working scale)
Distinct ROLE drives → **DISTINCT competitive slots** via the spiking rank-order (Thorpe latency) pooler:
| seed | slot sizes (R=4 roles) | mean pairwise Jaccard |
|---|---|---|
| 42 | [6,6,6,6] | 0.064 |
| 43 | [6,6,6,6] | 0.079 |
| 44 | [6,6,6,6] | 0.064 |
⇒ each role/bind gets its own near-orthogonal 6-column slot (Jaccard ~0.07 ≪ overlap). **This is the load-bearing property**: capacity converts from SNR-limited (~2, the write-rule store) to slot-count-limited (combinatorial). **Prerequisite GO.**

## The precisely-identified next challenge (build step 2): multi-slot COEXISTENCE on ONE bridge
On a REUSED bridge, sequential selections gave `[6,0,6,0]` — the columns' adaptation + FS inhibition from slot-0's firing **suppress the next selection** (state carryover; the EMERGE-61 adaptation-accumulation family). A fresh bridge per selection avoids it (above) but doesn't test coexistence. The real binder needs the P slots to COEXIST on ONE bridge for retrieval. **The gate's designed fix: the D3 persistent-slot ATTRACTOR holds each selected slot** (stable, zero-input, coexisting) — `_d3_persistent_slot_derisk.py`. So build step 2 = pooler-SELECT the slot → D3-attractor HOLD it (per bind) → role-cued RETRIEVE (drive role → complete the matching slot → decode filler), with a per-selection reset (EMERGE-61 wash-out) or the attractor absorbing the carryover.

## Status + next
- Build step 1 (slot separation) GO. Step 2 (coexistence + retrieval via D3 attractor) is the substantial continuation, with the state-carryover mechanism precisely identified.
- GO bar unchanged: a fact's P≥3 bundle recovers on spikes ≥0.80 where the write-rule capped ~2; anti-cheats permuted-role / lesion-the-competition→~2 / homeostasis-OFF; 6-seed.
- THE LAW: the write-rule method is refuted; this competitive-slot method is progressing; the capability stays OPEN until it works end-to-end.

---

## BUILD STEP 2a (multi-slot COEXISTENCE): GO — P=3 slots coexist via genuine NMDA persistent activity

Composed `build_persistent_slot` (K NMDA-recurrent pools + shared FS): load P=3 pools sequentially (no CLEAR), then hold at zero input (asserted). Result (3-seed):
| seed | NMDA-ON held | NMDA-OFF (anti-cheat) |
|---|---|---|
| 42 | 3/3 [0.078 0.073 0.102] | **0/3** [0 0 0.007] |
| 43 | 3/3 [0.096 0.098 0.023] | **0/3** [0 0.003 0.008] |
| 44 | 3/3 [0.129 0.028 0.061] | **0/3** [0.001 0 0.007] |

**The decisive no-recurrence anti-cheat collapses 3/3 → 0/3 every seed** ⇒ the coexistence is genuine NMDA-recurrent persistence, not an artifact. Multiple slots hold SIMULTANEOUSLY (the single-item-WTA framing does not bind here — loading DIFFERENT pools, each holds its own; "resists overwrite" was about re-writing an ALREADY-held pool). Honest flag: hold-rates were fs_to_exc-INVARIANT (10→0.5), so the shared FS is not the coexistence-limiting factor at these sparse loads — a real observation to characterize, not a claimed win.

## ⇒ Step-2 status: slot SEPARATION (step 1) GO + slot COEXISTENCE (step 2a) GO. Remaining:
- **step 2b — role-cued RETRIEVAL:** drive a role/partial cue → the matching coexisting slot completes/wins → decode its filler. (The slot must be role-addressable — the composition of the pooler selection + the NMDA hold + a decode read.)
- **step 2c — the full multi-bind recovery test:** a fact's P≥3 role-filler bundle recovers ≥0.80 where the write-rule capped ~2; anti-cheats permuted-role / lesion-the-competition→~2 / homeostasis-OFF; 6-seed.

---

## BUILD STEP 2c (role-cued retrieval) — runner built + PRECISELY diagnosed to a substrate bug (in progress)

Runner `research/runners/_keystone2_spiking_slot_binder_derisk.py` (slot pools NMDA-recurrent + FS + KF filler pools + a PER-SLOT-gated PLASTIC slot→filler pathway; role→slot→filler store + role-cued retrieve; slot-separated vs shared, + no-recur / permuted-role anti-cheats).

**The mechanism WORKS single-bind** (isolated diagnostic): teach (drive slot+filler, co-activation → Hebbian) → retrieve (drive the slot) → the filler pool fires, argmax = the correct filler (f0 rate 0.45). Two composition bugs found + fixed along the way: (a) `_reset` between binds BREAKS retrieval (the mechanism needs the NMDA hold — reset=False works, reset=True gives 0), (b) a SHARED plasticity gate let a bind's teach decay the OTHER slots' associations → switched to PER-SLOT gates.

**The remaining bug (precisely pinned, undiagnosed-to-root):** the MULTI-bind store loses the earlier association. teach0 writes w0→f2 (it transmits — at teach1, f2 still fires 0.02 from the held w0); but at RETRIEVE, **f2 = 0.00 despite w0 firing 0.12** ⇒ a SUBSEQUENT bind's teach window eroded w0→f2 **despite slot0's per-slot gate being frozen (gain 0) during teach1**. This points to a **Hebbian decay that `set_plasticity_gate` does not fully freeze** for a frozen pathway (or a gate-freeze gap). 

**Next debug (fresh focus): verify whether `set_plasticity_gate(name, 0.0)` fully freezes the Hebbian DECAY (not just potentiation) for that pathway's synapses** — read the Hebbian update in `sim/bridge.py` for whether the decay term is scaled by `cp_plasticity_rate_gain`. If NOT gated: write the associations decay-free (freeze `enable_hebbian_learning` after each write + a manual/eligibility write, or a per-slot store that never re-opens). Then re-run: slot-sep P≥3 ≥0.80 vs shared ~2, no-recur collapse, permuted→chance, 6-seed.

**Honest status: steps 1 (separation) + 2a (coexistence) GO; step 2c (retrieval) is a built runner diagnosed to a specific gated-Hebbian-decay substrate question — the last piece of the gap-#2 spiking closure.**

### ↳ step-2c diagnosis CORRECTED by reading the substrate (decay is NOT the bug)

Hypothesis was "the Hebbian decay isn't gated." READ `sim/bridge.py:7049-7053`: the decay IS gated — `gated_decay = hebbian_weight_decay * cp_plasticity_rate_gain` (line 7051), so a FROZEN synapse (gain 0) has ZERO decay. ⇒ the frozen slot-0 association does NOT decay during bind-1's teach; the decay hypothesis is REFUTED (per `feedback_read_own_substrate_before_theorizing`: read the code, don't theorize).

**The real remaining issue = WRITE STRENGTH, not decay.** The single-bind write is strong (retrieve f0=0.45), but in the multi-bind flow w0→f2 is weak/never-consolidated (f2≈0.02 at teach1, 0.00 at retrieve). Candidate root causes to check next (fresh focus): (a) the Hebbian **coactivity threshold** `hebbian_coactivity_thresh=0.25` (`bridge.py:7006`) — if the slot+filler co-activity (both ~0.06-0.08 rate) is below 0.25, potentiation is GATED OUT and the single-bind "worked" for a different reason (e.g. residual/held transmission) — so the write may never be strong; (b) the write needs the slot HELD (strong sustained firing) during teach to reach the coactivity threshold, which the held-during-store then conflicts with; (c) longer teach / higher `hebbian_learning_rate` / lower `hebbian_coactivity_thresh`. **Next debug: print `_coact` for the slot→filler synapses during teach + the resulting weight; tune the write to clear threshold; then re-run the full slot-sep vs shared + anti-cheats, 6-seed.** The mechanism is sound (single-bind GO); the write-strength calibration is the last step.

### ↳ step-2c: four hypotheses ruled out; bug narrowed to "earlier association lost despite frozen per-slot gate"

Ruled out by substrate-reads + targeted tests (not blind iteration):
1. `_reset` breaks retrieval → removed (mechanism needs the NMDA hold). FIXED.
2. shared plasticity gate → per-slot gates. FIXED.
3. Hebbian DECAY unfrozen → REFUTED: `bridge.py:7051` gates the decay (`hebbian_weight_decay * gain`), frozen synapse = 0 decay.
4. WRITE STRENGTH (coactivity-thresh 0.25→0.01, teach 25→80) → REFUTED (0.00 at all).
5. FS suppression (fs_to_exc 10→0) → REFUTED (0.00 at all; target slot fires ~0.08 regardless).

**Narrowed:** in the 2-bind diagnostic, retrieve role0 → the LATER bind's filler transmits (w1→f4 = 0.01) but the FIRST bind's is GONE (w0→f2 = 0.00). So a LATER bind's teach erodes the EARLIER slot's association **despite that slot's per-slot gate being frozen (gain 0) during the later teach** — which should be impossible (both potentiation AND decay are gated by `cp_plasticity_rate_gain`). ⇒ either the gate is not mapping to the intended synapses, or the held earlier slot's firing during the later teach interacts with the write in an un-gated path.

**DECISIVE next test (fresh focus):** directly read the `w0→f2` synaptic weight (from the CSR `cp_connections` for w0's neurons → f2's neurons) immediately after bind-0's teach vs after bind-1's teach — isolates WRITE-LOSS (weight drops despite frozen gate → a gate-mapping/ungated-write bug in `bridge.py`) vs READ-FAILURE (weight persists but doesn't transmit → the held-slot read regime). Then fix accordingly and re-run the full slot-sep(≥0.80 P≥3) vs shared + anti-cheats, 6-seed. The single-bind mechanism is SOUND; this multi-bind gate/write interaction is the last, precisely-localized piece.

### ↳ step-2c: isolated to a WEIGHT-ORIENTATION + multi-bind transmission subtlety (needs fresh focus)

Full filler-rate decode (gain=4000, cued-slot fires 0.43): retrieve role0 (want f2) → fires **f1** (0.3, = bind-2's filler slot2→f1), roles 1&2 → nothing. So driving slot0 fires the LAST-written bind's filler, not slot0's. Combined with the weight-read returning 96.845 in the `[sidx0, fidx2]` orientation ONLY, the likely root is a **`cp_connections` orientation ambiguity**: if it is [post,pre], the "96.845" is a REVERSE f2→w0 synapse and the intended w0→f2 was written to the OTHER index (~0) — i.e. the association may be written on the wrong side / read on the wrong side, and only the last bind coincidentally transmits.

**⚠️ RESUME HERE (fresh focus, decisive first step):** in the WORKING single-bind case (w0→f0 transmits, f0=0.45), print BOTH `cp_connections[f0_idx, w0_idx]` and `[w0_idx, f0_idx]` to fix the orientation empirically (which index holds the strong, transmitting weight). Then verify the multi-bind writes/reads the SAME index. If the framework wired the plastic `w→f` pathway or the read on the unexpected side, correct the pool wiring / the weight-read; re-run slot-sep(≥0.80 P≥3) vs shared + no-recur + permuted, 6-seed.

**HONEST STATUS of gap #2 spiking closure:** steps 1 (slot separation) + 2a (multi-slot coexistence) are GO with decisive anti-cheats. Step 2c (role-cued multi-bind retrieval) is a BUILT runner + an extraordinarily-narrowed diagnosis (7 hypotheses tested/ruled out via substrate-reads + targeted probes): the SINGLE-bind mechanism works; the MULTI-bind read is stuck on a precise weight-orientation/transmission subtlety. This is the last localized piece — a fresh-focus debugging session, NOT a wall (per THE LAW). The single-bind GO proves the mechanism is sound; if this exact wiring proves un-fixable, the ranked fallback is the gate's #2 theta-gamma slots (EMERGE-85, timing-based).
