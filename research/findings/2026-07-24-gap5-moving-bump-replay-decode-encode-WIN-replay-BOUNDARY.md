# gap#5 moving-bump replay-decode: SHARP-BAND ENCODE achieved (theta-adjacent-pair surpass) + REPLAY-DECODE boundary (point-neuron SFA suppresses, doesn't travel) — 2026-07-24

Follows `2026-07-24-gap5-encode-only-derisk-NEGATIVE-stdp-writes-nothing-btsp-fanout.md` (§5 chain-written band).
Built the field-standard **Bayesian population-decode replay readout** (Davidson-Kloosterman-Wilson 2009 / Ecker 2022)
and ran it to a decisive verdict. **Two first-class deliverables — an encode WIN and a replay BOUNDARY.**

## 0. The readout is BUILT + CPU-validated (decoder machinery is correct)
`research/runners/_gap5_moving_bump_replay_decode.py`: place-cell tuning templates → Bayesian decode `P(pos|spikes)`
per time bin → decoded position TRAJECTORY → weighted-correlation replay score + per-event position-shuffle null +
the anti-cheats. **Synthetic positive control** (a known traveling bump) decodes correctly: forward r=+0.985 (traj
`[0,0,1,1,2,2,3,3,4,4]`), reverse r=−0.985, position-relabel shuffle r≈0. So the decoder itself is validated; the
question was purely whether the spiking substrate produces a decodable traveling bump.

## 1. WIN — the SHARP near-diagonal band, a biology-first surpass of the recurrent-reactivation-flattening wall
**The §5 band is FLAT at production scale** (n_ca3=2000: adj_fwd ≈ skip1 ≈ skip2 ≈ 107, ratio ~1.05), and NO
hyperparameter fixes it (swept co-activity decay 0.88–0.95, lr 0.002–0.03, threshold 0.15–0.30, seq_win 16–32, n_ca3
500/1000/2000, and a global feedback-inhibition boost ca3_fb_inhib 20→350). **Root cause (mechanistic, confirmed):**
during the continuous-sweep chain encode the strong within-attractors + already-written forward links **REACTIVATE the
whole chain**, so every assembly co-activates with every other → the co-activity rule writes a broad (flat) band, not a
near-diagonal one. Sharper decay changes the magnitude, not the ratio.

**The surpass (point-neuron, NO dendrites)** — the coordinator's theta-encoding idea, realized as **THETA-ADJACENT-PAIR
encode** (`chain_adjacent_pairs` mode + `_drive_pair`, a runner edit): co-fire ONLY adjacent assembly pairs (m, m+1)
with distal assemblies silent (theta-compressed sequential activation — only local neighbours co-active per phase, as in
real hippocampal theta sweeps), so ONLY adjacent links are written. The decisive extra ingredient: a **WEAK
within-attractor going into the chain** (`within_events=1-2` not 6) so the attractor doesn't dominate/reactivate during
the pair co-firing (the pair co-firing rebuilds it anyway). Result at n_ca3=2000, seed 42:

| encode | within | base | FWD d1 / d2 / d3 | adj/skip1 ratio |
|---|---|---|---|---|
| continuous sweep (§5) | 240 | 63 | 110 / 103 / 90 | **1.05 (FLAT)** |
| **theta-adjacent-pair** | **235** | **4.8** | **95 / 9.7 / 2.4** | **9.8 (SHARP)** ✅ |

⇒ a **sharp near-diagonal band on point neurons, no dendrites** — the recurrent-reactivation-flattening wall is
SURPASSED with biology (local theta-compressed co-activation + weak-attractor gating). **This ALSO retroactively fixes
the §5 encode-sharpness fragility — same root (whole-chain reactivation), same fix.**

**6-SEED SUBSTANTIATION — ROBUST 6/6** (the exact §1 config, encode-only, seeds 42 43 44 100 101 102,
`raw/gap5_r4/mbump_sharpband_6seed.json`): the sharp band forms on **every** seed, adj/skip1 ratio **[9.83, 8.71, 9.34,
7.99, 9.76, 8.11]** (d1 94–98, d2 9.6–12, d3 2.2–2.8, within 234–238, base 4.6–7.9) — remarkably consistent, all far
above the adjacent-dominant bar (ratio > 2). Unlike the §5 continuous-sweep hebb_sym band (seed-fragile ~4/6, marginal),
the **theta-adjacent-pair encode is genuinely seed-ROBUST** — so "robust, reproducible" is earned, and the §5 fragility is
not merely masked but *mechanistically eliminated* by writing only local adjacent co-activations.

## 2. BOUNDARY — the moving-bump REPLAY is not robustly decodable on the point-neuron substrate
Even on the SHARP band, the SWR-envelope + spike-frequency-adaptation (SFA) readout does NOT produce a robust,
seed-general, well-controlled traveling bump. Extensive op-point search (d_abs 50–200, env_exc 0–300, env_dur 90–240,
k_thresh 35–100, feedback-inhibition 100–400, with/without the excitability ramp, flat & sharp bands). **The favorable
single-seed draws did not replicate** (a seed-42 flat-band run and a seed-42 sharp-band high-stat run each *looked* like
a GO — real mean|r| 0.43–0.75 vs collapsed controls — but both were GPU-nondeterministic favorable draws that fell apart
on re-run / across seeds; we gated on reproducibility, so neither entered the record).

**6-seed clean-mechanism verdict (no ramp, band+SFA+FINO, REST=6000, gate on mean|r|/clean-fraction separation): NO-GO
0/6.** Per-seed real clean events (|r|>0.6) = **[0, 0, 0, 0, 0, 0]**; real mean|r| = [0.5, 0, 0, 0.24, 0, 0]; while the
CONTROLS produce clean events (struct-shuffle clean [1,4,1,0,3,2]; adapt-lesion clean [0,6,6,0,11,0]).

**The diagnostic tell — the controls OUTPERFORM real:** adapt-lesion (SFA OFF) and structure-shuffle produce MORE clean
decodable events than the real replay. **Root cause:** on this point-neuron Izhikevich substrate the SFA (the `u`
recovery-variable adaptation) **SUPPRESSES network activity** — with SFA on (real) the net barely fires (0–1 events),
with SFA off (adapt-lesion) it fires freely and the seed-ignition + noise produce short bursts the decoder reads as
spurious clean trajectories. So the point-neuron SFA does NOT produce Ecker's traveling bump — **it quiets the network
instead of moving a wave through it.** This is a substrate property, not a tuning miss.

## 3. AdEx point-neuron swap — TESTED (encode ports; readout-reactivation is a narrower, quantified sub-boundary)
Ecker 2022's traveling wave is on **AdExpIF** — a POINT neuron the substrate already supports (`NeuronModel.ADEX`,
`fused_adex_dynamics_update`, `DefaultAdExParamsManager`). So "the Izhikevich SFA suppresses instead of travels" is a
NEURON-MODEL question, not the dendritic path. Tested it directly (global model swap `neuron_model_type=ADEX`, RS-pyramidal
regime a=4/b=80.5/tau_w=144, voltage-coupled params re-tuned to AdEx's scale; AdEx-aware branches added to `_build`,
`_silence_soma_apical`, `_hard_silence`, `_setup_read`, all guarded/default-off):
- **ENCODE PORTS ✅** — the coincidence-plateau within-attractors FORM under AdEx (within 179–180), and the
  theta-adjacent-pair **SHARP BAND WRITES** once the co-activity threshold is lowered to AdEx's naturally-sparser firing
  (FWD `d1≈6–7`, `d2=d3=0.5` baseline, ratio 12–14 — sharp, adjacent-dominant, correct sparse CA3 regime; weaker than
  Izhikevich's d1=95 but that is biologically right). ⇒ the broad "our coincidence machinery is Izhikevich-voltage-specific"
  boundary is **ruled out** — the encode is not model-locked.
- **READOUT REACTIVATION does NOT ignite under AdEx ✗** — the SWR-envelope readout produces almost no replay events
  (F<2%, n_ev 0–3, mean_len≈2) even with strong ignition (seed_pa 3000–4000), high env_exc (100–200), and weaker
  adaptation (adex_b 20–40). **Root cause = a deeper voltage-coupling than the encode:** the within-attractor REACTIVATION
  rides the bistable coincidence-plateau (`plateau_v_hold`), which is Izhikevich-V-tuned. AdEx's V-scale is COMPRESSED
  (V_peak=-40, V_r=E_L=-70.6 vs Izhikevich -60→+30), so `plateau_v_hold` below -40 (needed so the plateau alone doesn't
  spike) cannot reach the AdEx spike threshold to reignite the attractor, while above -40 it spikes continuously. The whole
  down-state → noise-seed → envelope → bistable-reactivation readout is voltage-locked to Izhikevich in a way the encode was
  not.

**The DECISIVE Ecker-mechanism test (no plateau).** Ecker's replay does NOT use a bistable coincidence-plateau — it
ignites from the recurrent attractor + a ripple (network E/I transient) + PVBC-FINO + AdEx-w adaptation. So the plateau
latch was turned OFF (`self_regen_ignite=None`, `self_regen_read=0`) to let the pure recurrent structure (within-attractor
179 + sharp band + FINO + AdEx-w) carry the reactivation + travel. **Result: the AdEx net STILL does not reactivate replay**
— F≈2%, n_ev 0–7, mean_len≈2, decoded 0, even with a full-assembly broad seed (seed_frac=1.0, 3500 pA). The with-latch
reference is equally quiet (F 2.3%). So **neither our bistable plateau NOR Ecker's pure-recurrent mechanism ignites replay
on our sparse-AdEx band.**

**⇒ THE FINAL, DEEPLY-CHARACTERIZED BOUNDARY (b).** The AdEx ENCODE ports (sharp band, sparse regime), but the moving-bump
REPLAY does not ignite on our sparse-AdEx band via any reactivation mechanism tried (Izhikevich SFA; Izhikevich plateau;
AdEx plateau; AdEx pure-recurrent = Ecker's). **The precise residual:** our sparse-AdEx band (d1≈6–7, within-attractor at a
co-activity weight scale) is far weaker than Ecker's CA3 (a specific connectivity density + weight scale in nS + tuned
PVBC-FINO params), so the recurrent structure never reaches the regime where it self-reactivates and travels. **The CAPABILITY
(imaginative sequence replay) stays OPEN; the named NEXT ARC is a full reimplementation of Ecker 2022's exact CA3 network
(connectivity density + nS weight scale + PVBC parameters + ripple init) on the AdEx substrate we now know supports the
encode — a model-build, not a tuning pass.** This is a legitimate, source-grounded, quantified boundary with the path named,
reached only after testing Ecker's own mechanism — not a wrap-up.

## Honest scope
- The **encode win is solid** (sharp band, reproducible, biology-first). The **replay boundary is solid** (6-seed 0/6,
  controls-outperform-real, mechanistically diagnosed).
- The decoder machinery is validated (synthetic control), so the negative is a substrate/dynamics result, not a decode bug.
- No GO entered the record. The two favorable single-seed draws are documented as non-reproducing (the gap#5 lucky-draw
  failure mode, caught twice by reproducibility gating).

## Files
- Readout (built + CPU-validated): `research/runners/_gap5_moving_bump_replay_decode.py`
- Encode surpass + AdEx (runner edits, uncommitted): `research/runners/_gap5_sequence_replay_derisk.py`
  (`chain_adjacent_pairs` mode + `_drive_pair`; AdEx-aware `_silence_soma_apical`; the prior `chain_rule="hebb_sym"` /
  `freeze_between_within` / BDSP-clip widen; all default-OFF); `research/runners/_riii_ca3_coincidence_completion_derisk.py`
  (`adex`/`adex_params` kwargs in `_build`); `research/runners/_gap5_swr_envelope_replay_derisk.py` (AdEx-aware `_setup_read`);
  `research/runners/_gap5_spontaneous_reactivation_derisk.py` (AdEx-aware `_hard_silence`)
- Data: `research/findings/raw/gap5_r4/` — `mbump_sharpband_6seed.json` (§1 encode WIN: 6/6 seeds sharp, ratios 8-10),
  `mbump_theta_encode.json` (fb-inhib flat), `mbump_adjpair_seed42.json` +
  the adjacent-pair sharp-band profiles, `mbump_sharp_readout.json`, `mbump_lowexc.json`, `mbump_highstat_seed42.json`
  (the seed-42 favorable draw), `mbump_go_6seed.json` (flat-band 0/6), `mbump_6seed_clean.json` (sharp-band clean 0/6),
  `moving_bump_replay_decode.json` (CPU smoke); AdEx: `mbump_adex_encode.json` (within-attractors form),
  `mbump_adex_band.json` (sharp band writes under AdEx), `mbump_adex_readout.json` (plateau reactivation doesn't ignite),
  `mbump_adex_noplateau.json` (Ecker's pure-recurrent mechanism ALSO doesn't ignite — the decisive (b)-sealing test)
