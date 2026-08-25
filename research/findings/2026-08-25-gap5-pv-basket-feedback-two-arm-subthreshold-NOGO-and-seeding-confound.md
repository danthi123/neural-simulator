---
type: finding
status: qualified
lane: gap#5
date: 2026-08-25
---
# gap#5 branch-B: recall CA3 PV-basket feedback (BOTH arms engaged) gives only SUB-GO-THRESHOLD store-dependent events — NO-GO 0/6 — and, found while verifying, the gap5 store build is NOT cfg.seed-reproducible (2026-08-25)

Artifacts (n_ca3=2000, cupy): `research/findings/raw/gap5_r4/fbinhib_6seed_2000.json`
(fb_read x fb_drive x chain_lr, 6 seeds), `fbdrive_controlsuite_6seed_chain05.json` (the AUTHORITATIVE anti-cheat
suite + GO gate), `fbinhib_sweep_s42_diag.json` + `chainlr_s42_2000_diag.json` (seed-42 sweeps). Runners:
`_gap5_ignition_fbinhib_sweep.py` (new); `_gap5_store_seeding_determinism_probe.py` (new, the seeding confound below);
`_gap5_dg_detonator_ignition_derisk.py` (added `fb_read`/`fb_drive`/`chain_btsp_lr` readout knobs, additive
default-None; NO sim/ edit).

## What was tested (board #71 next-rung 2c)
The DG-detonator honest-negative's ranked next-rung: recall-time FEEDBACK inhibition (a CA3 PV-basket pool) so
competitive WTA sharpens the readout to ONE discrete assembly, then transitions forward. Built as TWO additive readout
knobs on the substrate's EXISTING ca3_pv_basket E->I->E loop (`ca3_fb_inhib`=20): `fb_read` sets the recall basket->ca3
(I->E) weight; `fb_drive` scales the ca3->basket (E->I) drive. Both default None => no write (byte-identical);
plasticity FROZEN across every readout (byte-verified); all anti-cheats WIRED AND INVOKED.

## Result — HONEST NEGATIVE, 0/6 on the GO gate (`fbdrive_controlsuite_6seed_chain05.json`)
Two things had to be true and only one is:
1. `fb_read` ALONE is INERT (seed-42 2000 sweep): a0_peak flat across fb_read None..40, basket_mean ~1e-4 -- a sparse
   ~k_det-cell detonation does NOT recruit the FS basket, so re-arming the I->E weight alone does nothing. (This is the
   "wrong lever" that a single-knob read would report.)
2. Engaging the E->I drive (`fb_drive=10`) DOES recruit the basket (basket_mean ~0.10) and it produces GENUINE,
   store-dependent, FORWARD-ORDERED a0->a2 events -- every anti-cheat collapses as designed (6/6 seeds):
   no-detonator SILENT (assembly_rest 0.000); shuffled-detonator, no-encode both -> member_frac 0.000; shuffled-within
   -> the completion collapses. FWD=1.00 / REV=0.00 on the decoupled store. The event exists ONLY with the learned store.
   BUT it is SUB-GO-THRESHOLD: member_frac 0.126-0.221 (min/max of the 6 per-seed member_frac in fbdrive_controlsuite_6seed_chain05.json) <!--derived--> (< the 0.30 discreteness bar), n_multi=1 (a SINGLE a0->a2
   hand-off, not a multi-step sweep), symmetric positive control fires only 2/6. So `discrete_ignition`,
   `assembly_specific`, and `forward_transition` all FALSE => GO 0/6.
So the two-arm-engaged basket produces a genuine (store-dependent) but SUB-THRESHOLD event; it does NOT clear the bar.

## ⛔ CONFOUND found while verifying — the gap5 store build is NOT `cfg.seed`-reproducible (`_gap5_store_seeding_determinism_probe.py`)
Building the SAME seed (42) TWICE IN ONE PROCESS gives DIFFERENT `cp_neuron_firing_thresholds` (e0718ff vs 00a5fe1) AND
DIFFERENT `cp_connections` (1fe4d02 vs fbca341), and the readout flips: member_frac 0.136 vs 0.128 <!--derived-->, **FWD 1.000 vs
0.000**. So `cfg.seed` seeds the cupy heterogeneity but NOT the connectivity draw (an unseeded global RNG that each
build advances -- the CLAUDE.md "each build advances the global RNG" confound, present on THIS `_prepare_sequence`
path). CONSEQUENCE: the per-seed numbers above are 6 samples across 6 DISTINCT non-reproducible substrates, NOT 6
controlled re-seeds; the exact member_frac (0.17 vs the permissive sweep's 0.35) is process-variable, and the specific
FORWARD-ORDER (FWD=1.0) is NOT reproducible (flips to 0.0 on rebuild). ROBUST across all builds: fb_read-alone inert,
fb_drive engages the basket, the anti-cheats collapse (within-run), and GO never reached (0/6). This seeding defect is
almost certainly a load-bearing reason the whole gap5 READOUT arc has been hard to pin -- and is the FIRST thing to fix.

## ⛔ CORRECTION (2026-08-25, on FIXING it) -- the root cause above is WRONG; it is NOT an unseeded RNG
On fixing, the mechanism was measured directly and the "connectivity draw unseeded" diagnosis above did NOT hold. The
connectivity + threshold DRAW is ALREADY fully `cfg.seed`-seeded: `_prepare_sequence(seed, cfg, do_encode=False)` is
BYTE-IDENTICAL across 3 fresh processes (thresholds AND connectivity). The non-reproducibility appears ONLY with the
ENCODE, enters at the FIRST spiking step, and VARIES run-to-run ACROSS processes (build2's hash changed between process
runs) -- i.e. it is NOT a deterministic global-RNG advance. TRUE CAUSE: the per-step synaptic-current SpMV
(`Wᵀ@spikes`) via cupyx/cuSPARSE is BIT-non-reproducible run-to-run on this stack (the identical SpMV returns 6
distinct results over 6 calls -- atomic FP accumulation), and the chaotic/bistable spiking + BTSP plasticity amplify
that per-step jitter into an entirely different store + a flipped readout. So it is a GPU floating-point determinism
bug, not a seeding bug; every build-time RNG here was already correct.

FIX (committed same day): `sim.bridge._deterministic_csr_matvec` computes each per-step transpose SpMV as an explicit
`add.reduceat` segmented reduction (no atomics -> byte-identical every call), wired under the existing
`cfg.deterministic_transpose_matvec` flag (which was itself non-deterministic before -- it still routed through
cuSPARSE `@`); `_build` now sets that flag, so the gap5 store + readout are byte-reproducible at a fixed seed (verified
in-process AND across fresh processes; different seeds still differ). Default (flag off) is byte-identical (9/9 existing
determinism tests pass). Pinned by `tests/test_determinism.py::TestGap5StoreByteReproducible`. This retroactively
un-blocks next-rung (a): the store is now reproducible, so the per-seed readout metrics can be trusted; caveat 3 (only
directions + within-run contrasts trustworthy) is LIFTED for runs built through the fixed `_build`.

## What the diagnostics corrected
- The DG-detonator drives a0 to a ~0.5 PEAK active fraction transiently (direct asm_peaks) -- well above the sparse
  driven set -- but only ~0.17 AVERAGED within the event (member_frac): a BRIEF, NON-SUSTAINED partial ignition, not a
  held burst. The arc's `_detect_events` (smoothed TOTAL-pop > ev_floor*asize ~= 103) reads ev=0 on this un-sharpened
  transient co-fire, so the on-disk "member_frac=0 / no completion" was SUBSTANTIALLY an instrument artifact masking a
  real-but-transient partial ignition. The instrument is part of the emulation.
- The basket suppression is genuine WTA, not global inhibition: the DETONATED a0 (strong drive) survives the shared
  basket while the recurrently-driven a2 is suppressed (a2_peak 0.54->0.18 at chain 0.5) -- a differential, drive-biased
  competition.

## Caveats / residuals (the map for the next rung)
1. a1 EXCLUSION (universal, all 6 seeds, both scales, every chain_lr/fb): asm_peaks = [a0, 0.0, a2] ALWAYS -- the MIDDLE
   assembly never activates; the "forward transition" is a0->a2 (2 of 3), NOT a full A->B->C sweep. The BTSP tau-1000ms
   eligibility writes a0->a2 SKIP-links; the encode does not form a clean sequential adjacency. An ENCODE-structure
   defect upstream of the readout.
2. FWD-order is (i) partly protocol-aided (a0 is detonated first) AND (ii) NOT reproducible (flips 1.0<->0.0 on rebuild,
   per the confound above) -- so "forward-ordered" is not a bankable property of this readout at present.
3. The seeding confound (above) makes every quantitative gap5 readout metric on this path process-variable; only
   directions + within-run control contrasts are trustworthy until it is fixed.

## Verdict + next rung
NO-GO (0/6) for recall PV-basket feedback as the mechanism that reaches discrete-ignition-plus-forward-transition on the
decoupled store. Banks: "fb_read alone is inert"; records: "fb_drive engages the E->I arm -> a genuine store-dependent
but SUB-THRESHOLD event (member_frac ~0.17, n_multi=1), whose forward-order is not reproducible". THE LAW next rung,
in dependency order (all UPSTREAM of feedback inhibition):
(a) FIX THE SEEDING CONFOUND FIRST -- seed the connectivity draw from `cfg.seed` on the `_prepare_sequence`/`_build`
    path so a fixed seed gives a byte-identical store (verify via `_gap5_store_seeding_determinism_probe.py`'s hash compare); without this
    no gap5 readout metric is reproducible and no GO/NO-GO is quantitatively safe;
(b) FIX the a1-EXCLUSION encode defect -- a sequential (theta-compressed) adjacency so A->B->C completes, not an a0->a2
    skip;
(c) RECALIBRATE `_detect_events` for transient single-burst ignition (ev_floor*asize masks real bursts at asize~206) --
    the instrument must see the event before any readout can be judged;
(d) THEN re-test the two-arm basket (and a reverse-detonation directionality control) on the fixed, reproducible substrate.
