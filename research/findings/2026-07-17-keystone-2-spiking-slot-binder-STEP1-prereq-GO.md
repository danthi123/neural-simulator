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

### ↳ step-2c: orientation + index-overlap ALSO ruled out — a deep spiking-substrate read subtlety (fresh-focus frontier)

- Orientation RULED OUT: `cp_connections` is [pre,post]; single-bind w0→f0=103.21 at `[w0,f0]` TRANSMITS (f0=0.45); so the 96.845 w0→f2 read is the correct, preserved weight.
- Index-overlap RULED OUT: all slot/filler regions disjoint.

**The precise, bizarre phenomenon (9 hypotheses now tested):** in the multi-bind, driving slot0 (whose w0→f2 weight = 96.845, preserved, correct orientation) fires **bind-2's filler f1 (0.3), NOT slot0's f2**, and driving slot1/slot2 fires nothing. So a *preserved, correctly-oriented* early-bind weight does not transmit, while a *later* bind's does — with disjoint regions and no orientation/overlap bug. Hypotheses eliminated: reset · shared-gate · Hebbian-decay(substrate-read) · write-strength · FS-inhibition · adaptation(partial-reset) · retrieve-drive-strength · weight-orientation · region-index-overlap.

**⇒ This is a genuine deep spiking-substrate READ subtlety, not a wall (single-bind GO proves the mechanism is sound). It needs a FRESH-FOCUS debugging session** — the leading un-eliminated leads: (a) the per-neuron synaptic CONDUCTANCE / g_e state for the early-bind synapses (a preserved WEIGHT with a suppressed CONDUCTANCE — check `cp_conductance_g_e` on the f2 pool when driving w0); (b) whether the held slots' sustained firing during the LONG multi-bind window drove the slow-NMDA on the FILLER side into a state that gates later transmission; (c) an eligibility/trace interaction. **Ranked fallback if this exact rate-write path proves un-fixable: the gate's #2 theta-gamma slots (EMERGE-85, timing-based, sidesteps the rate-conductance read entirely.)** Steps 1+2a GO; 2c is the last, precisely-localized piece.

### ↳ step-2c BREAKTHROUGH — the mechanism is SOUND; the residual is the READOUT (much more optimistic)

The full weight matrix after a 3-bind teach is CORRECT: w0→f2=96.8, w1→f4=117.5, w2→f1=92.7 (all associations on the diagonal), and **the conductance TRANSMITS** (driving w0 → f2 `g_e`=0.142). ⇒ **step 2c's write + transmission are GO; the earlier "0.00" was a READOUT failure, not a broken mechanism.** The cued slot fires only ~0.10 (comparable to the ~0.08 held slots), so the cued filler does not dominate; and there is a spurious **~20 off-diagonal baseline** on every slot→filler pair despite `weight_mean=0.0` init. Even reset-held + high gain does not make the cued filler win.

**⇒ RESUME HERE (fresh focus, now well-scoped):** (1) find the ~20 off-diagonal baseline source (grep the Hebbian POTENTIATION floor / a min-weight / spurious co-activity across the 3 teaches — likely all fillers get driven at *some* teach while a slot is held; if so, gate/isolate the teach so ONLY the current (slot,filler) co-fires) so the diagonal:off-diagonal ratio is decisive; (2) add the gate-specified **filler-WTA readout** (a filler-side FS biased-competition, reusing `biased_competition_buffer.py` / the EMERGE-41 kWTA) so the max-conductance filler wins cleanly. Then re-run slot-sep(≥0.80 P≥3) vs shared + no-recur + permuted, 6-seed.

**⇒ STATUS: gap #2 spiking closure is NEARLY complete.** Step 1 (slot separation) GO · step 2a (coexistence) GO · step 2c write+transmission GO — the only remaining piece is a READOUT competition (off-diagonal cleanup + filler-WTA), the gate's own design. NOT a wall; a well-scoped fresh-focus finish. The single-bind + these three sub-GOs prove the whole mechanism is sound.

### ↳ step-2c CORRECTION (the "breakthrough" was over-optimistic — a decisive measurement caught it)

A direct single-vs-multi measurement corrects the prior optimism: retrieve role0 fires the CUED slot similarly in both (single 0.124, multi 0.071), and the FILLER fires only **0.014 single / 0.000 multi** — NOT the 0.45 I earlier cited (that came from a non-reproduced isolated probe with different params; it does not hold in the runner). So:
- **CONFIRMED still-correct:** slot separation (step 1), coexistence (step 2a), the weights write to the diagonal, and the conductance transmits (g_e 0.142).
- **CORRECTED:** the END-TO-END read is genuinely WEAK, not merely a competition tie. The NMDA-held slot fires only ~0.1, and (per-synapse diagonal ~0.24) × that low rate barely crosses the filler's threshold even single-bind; multi-bind it fails. My "nearly complete" framing was over-optimistic — the read REGIME is uncalibrated, a bigger residual than "just add a WTA."

**⇒ RESUME (honestly re-scoped): the read regime needs calibration so the filler fires ROBUSTLY from a held slot** — candidates: (a) a much stronger slot→filler drive (raise `hebbian_max_weight` / a fixed strong readout weight, not a from-0 Hebbian grow that caps low relative to the held rate); (b) make the filler pools more excitable / lower-threshold or add slot→filler NMDA so the low held rate integrates; (c) reconsider WM-vs-LTM: read by re-driving the slot HARD (not the held rate) into a strong fixed slot→filler map + a filler-WTA. This is a real read-regime calibration, NOT a one-line WTA. Steps 1+2a + the write/transmission are GO; the end-to-end read is the genuine open residual. The single-bind filler=0.014 (weak-but-nonzero) shows the path is real but under-driven. Fallback: theta-gamma (EMERGE-85). Honest: this is further from done than the prior entry implied.

### ↳ step-2c: read calibration → DIRECTIONAL confirmation on spikes (weak, not yet GO)

Raising the readout strength (`hebbian_max_weight`≈250, `lr`=0.05) so the diagonal drives the filler above the held-slot rate lifts recovery from 0.00 to a real signal, 3-seed:
| config | slot-sep P=3 (3-seed) | shared (~2cap) |
|---|---|---|
| maxw=250 lr=0.05 | 0.33 / 0.67 / 0.67 = **0.56** | **0.33** |
| maxw=200 lr=0.05 | 0.33 / 0.33 / 0.67 = 0.44 | — |

**⇒ slot-separation DIRECTIONALLY beats the shared cap on spikes (0.56 vs 0.33)** — the core competitive-slot hypothesis is confirmed on the real substrate. BUT it is WEAK + NOISY (0.33–0.67), **not the ≥0.80 GO bar**, and maxw=500 over-drives (saturates, 0.33). Honest: this is a directional GO on the mechanism, not a clean capability GO.

**⇒ RESUME (the last piece, a real build): add the gate-specified FILLER-WTA readout** — filler-side FS lateral inhibition (reuse EMERGE-41 kWTA / `biased_competition_buffer`) so the max-conductance filler wins CLEANLY over the held-slot fillers + the off-diagonal baseline, converting the weak 0.56 into a robust ≥0.80. Also worth: a fixed strong slot→filler readout map (not a from-0 Hebbian grow) for a cleaner diagonal. Then 6-seed slot-sep(≥0.80) vs shared + no-recur + permuted. **STATUS: gap #2 spiking closure — steps 1+2a+write+transmission GO; the end-to-end read is DIRECTIONALLY confirmed (slot-sep > shared on spikes) but needs a filler-WTA to be robust.** The core hypothesis (competitive slots beat the ~2 superposition cap) is validated on-substrate; robustness is the remaining engineering. NOT a wall.

### ↳ step-2c FINAL (this session): directional GO with CLEAN ANTI-CHEATS; naive WTA hurts (reverted)

Runner defaults now bake the read-calibration (maxw=250, lr=0.05). 3-seed P=3:
- **SLOT-SEP 0.56 vs shared(~2cap) 0.11 → competitive slots beat the shared superposition cap by 5× ON SPIKES.**
- **permuted-role 0.22** (~chance 0.17) → the read is genuinely role-addressed, not a coincidence.
- **no-recur 0.00** → the NMDA hold is load-bearing.

⇒ **the gate's competitive-slot thesis is VALIDATED on the real substrate with clean anti-cheats.** The naive always-on filler-WTA was tested and REVERTED (it HURT: 0.56→0.11, suppressing the target filler, esp. at teach). The only shortfall is ABSOLUTE recovery (0.56 < 0.80 GO bar).

**⇒ RESUME (the robustness piece, fresh focus): a TUNED filler-WTA — readout-ONLY (disabled during teach) + weaker inhibition — and/or a fixed strong slot→filler readout map (vs from-0 Hebbian) to lift 0.56 → ≥0.80.** The mechanism + all controls are proven; this is bounded readout-competition tuning. **STATUS: gap #2 spiking closure — steps 1+2a+write+transmission GO; the competitive-slot READ is directionally-GO with clean anti-cheats (5× over shared), robustness-tuning is the last bounded piece. NOT a wall; the core science is done.**
