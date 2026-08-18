---
status: live
type: finding
lane: laneC
date: 2026-08-18
---

# GNW STN->GPi hyperdirect reactive STOP-SIGNAL veto — the neural conflict SENSOR + SELECTIVITY are a GO, the global-clear EFFECTOR is a 6/6 NO-GO (the dense recurrent localist workspace attractor is inhibition-resistant to a GLOBAL clear exactly as it is to Rung-2b/2c EVICTION: a moderate STN->GPi pulse HOLDS the co-ignition, a stronger pulse DESTABILIZES the network UPWARD via the g_i-driving-force reversal, and NO strength drives n_ignited -> 0 — the named next lever is a distributed OVERWRITE workspace where clearing a shared resource de-ignites ALL content uniformly, driven by the conflict sensor built here)

**Date:** 2026-08-18
**Runner:** `research/runners/_gnw_rung_stn_stop_veto_derisk.py` (reuse-by-import of the Rung-1 assembly-loop + the _p1_2 K-slot workspace harness + the Rung-2 competitive harness; the ACC->STN->GPi hyperdirect stop loop is built by EXPLICIT WIRING only — dense frozen ACC/STN/GPi pools — **NO `sim/` edit**).
**Backend:** CPU (numpy). **Seeds:** 42/43/44/100/101/102. **Verdict:** a decisive, fully-anti-cheated **method-negative** on the EFFECTOR arm, with the neural conflict SENSOR and SELECTIVITY passing as GO components on all 6 seeds. This is a verdict on the METHOD (a broad hyperdirect inhibitory brake into a localist self-sufficient attractor), NOT on the CAPABILITY: the reactive global brake is NOT abandoned; the next lever (a distributed overwrite workspace) is named below, and the conflict sensor built here is its ready effector arm.
**Artifact:** `research/findings/raw/_gnw_stn_stop_veto_6seed.json` (+ `.prov.json` sidecar).
**Builds on / cites:** the Rung-2c BOUNDARY `research/findings/2026-08-17-gnw-rung2c-salience-disinhibition-BOUNDARY.md` (the dense recurrent attractor is inhibition-resistant: "co-ignite or self-extinguish; the metastable middle is EMPTY"; and its do-not-retread "active-clear FS quench works but is a HOST shortcut" — this STN->GPi is that quench's brain-based analogue) and the Rung-2b BOUNDARY `research/findings/2026-08-14-gnw-rung2b-sfa-workspace-eviction-BOUNDARY.md`.
Biology: Frank 2006 hyperdirect "hold-your-horses"; Aron & Poldrack 2006 / Wessel & Aron 2017 broad fast STN global reactive stop; Wei-Rubin-Wang 2015 STN-GPe stopping dynamics. `NO-EXTERNAL-NEEDED:` the mechanism, the substrate property, and the named next lever are fixed by the corpus (the Rung-1/2/2b/2c findings) + the cited sources.

## What is DISTINCT from Rung-2b/2c (the intended distinction — confirmed, then the horn transfers)
Rung-2b (SFA) and Rung-2c (salience dis-inhibition) tried to EVICT one attractor to SWAP IN a more-salient challenger (n_ignited stays 1) — needing a metastable tip-point the dense attractor lacks. The STN stop is a GLOBAL reactive brake: it should KILL ALL content (workspace -> EMPTY, n_ignited -> 0), and SELF-EXTINCTION is the DESIRED outcome — so conceptually it sidesteps the metastability horn (it never needs a marginally-stable state to tip). **The de-risk CONFIRMS the conceptual distinction but finds the SAME underlying substrate property blocks the global clear from the other side:** the ignited dense recurrent attractor is not just un-evictable, it is un-CLEARABLE by any graded neural inhibition — it is a robust, deep, self-sufficient bistable basin with no accessible "off" transition under a graded brake.

## The GO components (robust, all 6 seeds) — the neural conflict SENSOR + SELECTIVITY
The ACC conflict unit reads the workspace's OWN spiking ignition MARGIN (winner_rate - runnerup_rate, late-window per-slot), NOT a host "wrong" label. Measured identically on all 6 seeds:
- **Neural margin distinguishes conflict from confident:** a CONFIDENT commit (slot0 strong, rest weak) ignites ONE slot -> margin ~0.33 (single content); a HIGH-CONFLICT commit (two slots co-driven) CO-IGNITES -> margin 0.000 (two contents = the ambiguity). The margin is a genuine spiking read of the ignition state.
- **The pulse is conflict-gated (SIGNAL-DRIVEN, not a host reset):** i_acc = conflict_gain * max(0, MARGIN_REF - margin) * SCALE. i_acc = 0 at the confident (zero-conflict) end and rises monotonically across a conflict sweep (`pulse_zero_at_zero_conflict` and `pulse_scales_with_conflict` both True, 6/6). Conflict i_acc = 3000 pA into the ACC pool.
- **SELECTIVITY (GO gate 2) PASSES:** on the confident commit i_acc = 0 -> GPi silent -> the workspace is NOT perturbed, n_ignited stays 1, and the correct content BROADCASTS (delivered = COMMIT). The veto does not fire on a confident commit, on all 6.
- **NEURAL-SENSOR load-bearing:** a host-margin-SCRAMBLE (feeding the confident margin to the conflict trial) sets i_acc ~ 0 -> no pulse -> the wrong commit is NOT aborted (`scramble_breaks_abort` True, 6/6). The abort depends on the ACTUAL spiking margin, not a ground-truth label.
- **The effector genuinely fires:** ACC->STN->GPi transmits — GPi fires at rate ~0.19-0.21 during the conflict pulse on all 6 (a real, chain-delivered inhibitory volley, effective g_i ~ 30 into the workspace), so the effector failure below is the attractor's resistance, NOT a dead chain.

## The EFFECTOR NO-GO (6/6) — no GPi strength drives n_ignited -> 0 (the residual, quantified)
On the HIGH-conflict commit the ACC->STN->GPi pulse fires (GPi ~0.19), yet the co-ignition does NOT clear: n_ignited 2 -> 2 (the two contents HOLD through the pulse). Sweeping the GPi->workspace strength (the effector arm), IDENTICAL on all 6 seeds:

| gpi_ws_w (per-syn) | n_ignited_post | regime |
|---|---|---|
| 8  | 2 | inhibition-resistant HOLD (co-ignition survives) |
| 16 | 4 | DESTABILIZES UPWARD (weak slots also ignite) |
| 24 | 4 | destabilizes upward |
| 32 | 4 | destabilizes upward |
| 40 | 4 | destabilizes upward |

`min_n_ignited_post_over_sweep = 2`, `effector_ever_empties_workspace = False`, on every seed. There is no GPi strength that reaches n_ignited = 0: below ~g_i 200 the dense self-sufficient attractor HOLDS (a heterogeneity-determined survivor at intermediate weight/aw; both hold at the primary aw=30 point); above it the network DESTABILIZES upward — the accumulated g_i drives V below the GABA-A reversal E_i = -75 mV, so the inhibitory current g_i*(E_i - V) REVERSES to DEPOLARIZING, and a synchronous post-inhibitory rebound re-ignites even the weak slots (n -> 4). The "off" transition simply is not accessible to a graded neural brake on this substrate.

## Anti-cheats (each held — this is a real, clean negative, not an artifact or an instrument failure)
- **SIGNAL-DRIVEN not host reset:** i_acc = f(ACC conflict); `pulse_zero_at_zero_conflict` and `pulse_scales_with_conflict` both True (6/6). A current into a spiking pool timed as a pulse, asserted in code — never a `_restore_state`.
- **NEURAL SENSOR (the load-bearing one):** host-margin-SCRAMBLE breaks the abort (6/6).
- **CONFLICT-OFF reproduces the negative:** conflict_gain = 0 -> i_acc = 0 -> the wrong commit broadcasts uncorrected (6/6).
- **STN-lesion (STN->GPi weight 0):** the pulse's downstream inhibition is removed; the conflict commit broadcasts (`stn_lesion_broadcasts` True, 6/6). (The intact-vs-lesion emptying ATTRIBUTION is UNDEFINED here because the intact arm does not empty either — there is no emptying to attribute; this is honestly reported as `attribution=None`, not a fabricated 100%.)
- **0 `_restore_state` calls:** the CONTINUOUS abort headline (isolate=False) makes ZERO restore calls on all 6 — any workspace change is neural, never a host wash-out.
- **Determinism:** build-twice at one seed -> identical hash of the seed-derived Izhikevich params (heterogeneity seeded from `cfg.seed`, NOT `actual_seed_used`), 6/6.
- **Verdict is a clean NO-GO, not UNDEFINED:** all VALIDITY preconditions hold on all 6 (commit ignites on both trials, the neural margin distinguishes conflict from confident, the pulse is conflict-gated, SELECTIVITY holds, the effector delivers a GPi pulse, the scramble is load-bearing, CONFLICT-OFF reproduces the negative, 0-restore, determinism). The negative is the MEASURED failure to reach n_ignited = 0, NOT a failed precondition — the exact Rung-2b/2c split (encoding the abort outcome as a require would wrongly mark a valid NO-GO UNDEFINED).

## Why the horn transfers to a GLOBAL clear (the diagnosis) — banked exploration
The ignited dense recurrent workspace assembly (weight ~24-30, the validated Rung-1 recipe) is a robust, deep, SELF-SUFFICIENT bistable basin. Across a thorough sweep, NO graded neural manipulation drives it to n_ignited = 0:
1. **Direct hyperpolarizing inhibition** (E_i = -75): weak -> HOLD or a heterogeneity-determined WTA survivor (n -> 1/2, seed-dependent at aw = 24: n_post = {1,0,2,2,2,2}); strong -> driving-force reversal (V < E_i) -> numerical destabilization + synchronous post-inhibitory rebound -> n -> 4. No window in between.
2. **Shunting inhibition** (E_i raised toward rest, -63..-59): high g_i still destabilizes the network upward (n -> 4); no clean silence.
3. **Thalamo-cortical remove-excitation** (a tonic thalamic pool the workspace depends on, GPi silences it): the attractor is either sub-critical (never sustains, even WITH support) or, once ignited, SELF-SUFFICIENT (removing the tonic support does NOT kill it) — there is no "needs external support to sustain" window, and a resumed support RE-IGNITES an emptied slot (wrong hysteresis). Same for a direct shared-bias current.
4. **Attractor-depth tuning** (aw 21-28): aw <= 23 the confident commit does not hold over a long read (self-extinguishes on its own -> invalid); aw = 24 is a knife-edge that tips 2 -> 1 (one heterogeneity survivor), seed-dependent; aw >= 25 fully inhibition-resistant. No frozen point holds the confident commit AND cleanly clears the conflict on all 6.

The **companion-process** question (CLAUDE.md's deepest lesson) resolves it exactly as Rung-2c: we replaced the workspace's DYNAMIC, DISTRIBUTED, shared-resource content representation with K STATIC, SEPARABLE, SELF-SUFFICIENT localist attractors.
A localist self-sufficient basin has no shared resource to withdraw and no marginal stability to tip, so a graded neural brake can only push it around (hold / survivor / destabilize) but never off.

## The named next lever (this launches the next build; it is NOT a wall)
Rung-2c's own suggested substrate, now required for the global clear as well: a **distributed OVERWRITE workspace** — overlapping distributed PATTERNS in ONE recurrent net + divisive normalization / a shared global-gain resource — where content is a shared, normalized resource rather than K separable self-sufficient basins. There, WITHDRAWING the shared gain (the STN->GPi brake acting on the normalization pool) de-ignites ALL content UNIFORMLY (no localist survivor, no separable basin to hold, no reversal-blowup because the mechanism is gain-withdrawal not a huge g_i), and self-extinction is the natural collapse. **The conflict SENSOR arm built and validated here — the neural ignition-margin read -> ACC -> STN -> GPi, conflict-gated and host-margin-scramble-breakable — is the READY effector to drive that substrate;** only the workspace representation needs replacing, not the hyperdirect loop.

## Do-NOT-retread (banked)
On the K-slot localist self-sufficient recurrent workspace: direct hyperdirect STN->GPi inhibition (any E_i -75..-59, any per-synapse weight, any pulse duration to 300 steps, OU on), thalamo-cortical remove-excitation, a shared-bias-current withdrawal, and attractor-depth tuning ALL FAIL to drive n_ignited -> 0 — the ignited basin either holds/leaves a survivor or destabilizes upward via the g_i driving-force reversal + synchronous rebound. The residual is specific: the effector needs a DISTRIBUTED shared-resource workspace (not K localist basins), which no graded inhibition can create.

## Remaining scaffolds (named, not claimed closed)
1. The conflict MARGIN is read by host code from the workspace spikes (the instrument) and injected as the ACC unit's afferent drive — faithful to a conflict monitor's afferents, but the read is host-side until an ACC circuit computes it from synaptic inputs.
2. The workspace slots + ACC/STN/GPi pools are hand-wired dense frozen populations, not self-organized.
3. GPi is modeled near-silent at baseline with a phasic STN-driven burst (the phasic-stop increase), not the full tonic-GPi / thalamic-relay loop.

## Files
Runner: `research/runners/_gnw_rung_stn_stop_veto_derisk.py`. 6-seed artifact: `research/findings/raw/_gnw_stn_stop_veto_6seed.json` (+ `.prov.json`).
