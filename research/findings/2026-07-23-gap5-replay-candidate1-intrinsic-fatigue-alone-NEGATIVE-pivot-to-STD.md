# gap#5 replay-transition: candidate #1 (intrinsic fatigue ALONE) does NOT direct the order — pivot to #2 (STD co-driver), as the research gate predicted (2026-07-23)

## Result (seed 42, full anti-cheat mode, center config sr0.12 / a0.025 / d120)
```
INTRINSIC     fwd=0.333 rev=0.000 (ev=4 multi=3 act=[3,3,3] pop=0.079)
ADAPT-LESION  fwd=0.500 (act=[2,2,2])       <- orders AS WELL AS / better than intrinsic
LATCH-ON      fwd=0.333 (act=[3,3,3])
NO-NOISE      multi=0                        <- good (not self-sustaining)
SCRAMBLE      fwd=0.000                      <- good (order doesn't ride a fixed permutation)
NO-ENCODE     multi=0                        <- good (chain is the store)
=> NO GO
```
`INTRINSIC fwd=0.333` is at chance, and the load-bearing control **fails**: `ADAPT-LESION fwd=0.500 > INTRINSIC` means the (weak) forward order rides the stored chain / self-avoidance, **NOT** the spike-frequency adaptation. The `--quick` `fwd=0.500` that looked like a marginal GO earlier was **noise** — the runs have only 2–9 replay events, so `forward_frac` swings 0.333↔1.000 across the config sweep (gcr1.5/gcr2.5/d130/a030/sr0.10/gcr2a030) with no config beating chance robustly.

## Diagnosis (research-gate-aligned, not a surprise)
The 2026-07-23 research gate predicted exactly this: **"somatic adaptation mainly SILENCES; short-term depression DIRECTS."** Candidate #1 (intrinsic dendritic-plateau sustain + somatic Izhikevich adaptation) does its job — it **prevents co-ignition** (INTRINSIC `act≈[2,2,2]` vs the ADAPT-LESION's `[3,3,3]` diffuse co-fire) — but silencing is not the same as **directing** a forward A→B→C order. Nothing in intrinsic fatigue alone breaks the forward/backward symmetry of the transition; the order it does show is inherited from the stored chain (which the lesion has too). So candidate-#1-ALONE being negative **confirms** the research gate's framing rather than refuting the capability.

## Per THE LAW: a verdict on the METHOD (intrinsic-fatigue-ALONE), the capability stays OPEN
The next mechanism is the research gate's **candidate #2: re-enable E→E (ca3→ca3) SHORT-TERM DEPRESSION as the directional co-driver, tested TOGETHER with #1** (Ecker 2022 / Romani-Tsodyks 2015: the just-fired assembly's *outgoing* recurrent synapses deplete → its forward/backward drive weakens → the bump travels forward and cannot immediately recur). Adaptation silences, STD directs.

## Candidate #2 — the precise, scoped build (EXACT NEXT ACTION)
1. **Enable STP on the ca3→ca3 recurrent WITH the mossy carve-out.** The build chain is `_gap5_intrinsic_fatigue_replay_derisk` → `_gap5_sequence_replay_derisk._prepare_sequence` → `_riii_ca3_coincidence_completion_derisk._build` (`:169-177`), which currently sets `cfg.enable_per_type_stp = False`. `_riii` ALREADY has the per-pathway carve-out infra (`RegionPathway.stp_disabled` → `cp_stp_disabled_mask`; the `mossy_stp_disabled` block `:154-162` disables STP on the dg→ca3 detonator so it isn't crushed while the recurrent keeps STP). Add an STP-enable param to `_build` that sets `enable_short_term_plasticity=True` + `enable_per_type_stp=True` AND flips `mossy_stp_disabled` so **mossy detonates STP-off, ca3→ca3 recurrent depresses STP-on** (the "no working window" boundary the `:157` comment names, solved by the per-pathway mask).
2. **Configure the E→E depression:** `stp_U ≈ 0.4–0.6`, `stp_tau_d ≈ 200–400 ms` (Tsodyks-Markram depression on the recurrent), `stp_tau_f` low (pure depression). Per-connection-type STP fields (`stp_U_per_type` / `stp_tau_d_per_type`) target the E→E class.
3. **Test #1+#2 together** via the intrinsic-fatigue replay runner (add a `--stp` toggle threading the enable through `_prepare_sequence` → `_build`): keep the plateau (self_regen 0.12) + gamma-scale adaptation (a 0.025, d 120) AND the recurrent STD. GO gate (6-seed 42/43/44/100/101/102, full anti-cheats already wired): `INTRINSIC fwd >> SCRAMBLE floor` AND `forward > reverse` AND `per_asm_active ~[1,1,1]` AND `ADAPT-LESION` (and now `STD-LESION`) collapse. **Watch for avalanche** (the `:157` note) — the recurrent STP is stabilizing (depression), but verify no runaway; increase `rest_steps`/events to raise the event count above the current noisy 2–9 (a low event count is why #1's forward_frac was unreadable).

## Scope caveat (honest)
This NEGATIVE is seed-42 full + the seed-42 `--quick` sweep (not 6-seed). That is sufficient to **bank the failing METHOD and pivot** (THE LAW: a method-negative launches the next method; 6-seed is required to GENERALIZE a POSITIVE, not to pivot). If candidate #2 (STD) also fails, a full 6-seed characterization of #1-alone + the theta-gamma-timing fallback (research candidate #3, phase-precession encoding) is the next escalation. The static CA3 completion this rides on is CLOSED (intrinsic dendritic bistability, 2026-07-18) and unaffected.
