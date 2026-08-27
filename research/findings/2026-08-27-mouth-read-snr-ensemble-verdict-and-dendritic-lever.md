---
type: finding
status: live
date: 2026-08-27
verdict: ensemble (--sub-pop) read lever INERT by construction (honest NO-GO); dendritic (Urbanczik-Senn) lever BUILT + smoke-clean + staged (6-seed eval pending)
mechanism: mouth read-SNR — population/word-pool ensemble read vs two-compartment apical teacher read (gap#4 / #80)
lane: mouth / gap#4-read-regime
artifacts:
  - research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py
  - research/runners/_wkv_mouth_readout_snr_ensemble_dendritic_derisk.py
  - research/findings/raw/_wkv_mouth_readout_snr_ensemble/screen/sub_pop1_screen.json
  - research/findings/raw/_wkv_mouth_readout_snr_ensemble/screen/sub_pop2_screen.json
  - research/findings/raw/_wkv_mouth_readout_dendritic_smoke_on.json
  - research/findings/raw/_wkv_mouth_readout_dendritic_offdiff_baseline.json
  - research/findings/raw/_wkv_mouth_readout_dendritic_offdiff_afteredit.json
---
# Mouth read-SNR (#80 / gap#4): the ENSEMBLE read lever is inert by construction; the DENDRITIC (Urbanczik-Senn) lever is built + staged

## The wall (do not re-derive)
The mouth read-out e-prop learning FORWARD — the graded-conductance substrate read that supplies the per-output
error to the local three-factor rule — plateaus at `sub_learned_recov_mean` ~0.34-0.37 while the copied-weight read
reaches the ~0.98 ceiling and a matched host-linear-proxy forward reaches ~0.86-0.90
(`2026-08-19-mouth-readout-eprop-batched-substrate-marginclean-6seed-NOGO.md`,
`2026-08-19-mouth-substrate-forward-40k-coverage-EXCLUDED-real-credit-limit.md`). Coverage and integration-window are
EXCLUDED. The plan-of-record's one open lever family: raise the effective read SNR by an ENSEMBLE (population average)
or a MULTI-COMPARTMENT / DENDRITIC read (Urbanczik-Senn soma-vs-dendrite).

## STEP 1 — ENSEMBLE (`--sub-pop`) read: INERT BY CONSTRUCTION (honest NO-GO, not a wiring bug)

A prior screen (`research/findings/raw/_wkv_mouth_readout_snr_ensemble/screen/sub_pop1_screen.json` and
`research/findings/raw/_wkv_mouth_readout_snr_ensemble/screen/sub_pop2_screen.json`) swept `--sub-pop` {1,2} and got
`sub_learned_recov_mean` = **0.4977 for BOTH** (bit-identical), go 0.
The question was whether `--sub-pop` reaches the read (a bug) or the graded read is genuinely P-independent (a design
property). **Answer: `--sub-pop` DOES reach the read; the read is genuinely P-invariant by construction — for two
independent reasons — so growing the word-pool cannot lift the read SNR.**

<!--derived-->
Evidence (code + the two cited screen JSONs):
- **`--sub-pop` reaches the substrate.** `batch_margin` pools the P word-pool members' conductance:
  `cp_conductance_g_e[...].reshape(B*V, P).sum(axis=1)`
  (`_wkv_mouth_readout_eprop_batched_substrate_derisk.py:312-313`). The screen confirms it took effect: `gain` doubled
  (sub_pop1 **9.933** -> sub_pop2 **19.769**, `gain_per_seed`), the `seed_hash` differed (`1d90c9..` vs `2660ec..`),
  and elapsed doubled (142s -> 247s). So the P-sum ran on the substrate; this is NOT a "sub_pop never reached the read"
  bug.
- **Yet the read is EXACTLY P-invariant.** The P members of a word-pool receive the SAME presynaptic hidden drive
  (same `Wp`/`hid` block) and the graded read is off the pools' subthreshold `g_e`/`g_i` CONDUCTANCES; OU noise is
  injected as CURRENT, not conductance, and the pools never spike — so the P members' conductances are DETERMINISTIC
  REPLICAS. `sum(P)` scales the margin by ~P, and `_calibrate_gain` measures the gain on the SAME net, so
  `margin_sub / gain` cancels P EXACTLY -> identical logits -> identical error -> identical learned W -> bit-identical
  recovery. The bit-identical 0.4977 is the fingerprint of this exact cancellation.
- **Second, independent reason.** The reported `sub_learned` metric is a DEMO read on a fresh `LearnedReadout(...,
  pop=args.pop, ...)` (line 574) that uses `--pop` (=4), NOT `--sub-pop`. So the screen's headline metric is P-invariant
  regardless of the forward.

**The mechanistic reason (and why this differs from the 2026-08-13 population-coding GO).** Population averaging
reduces read noise ONLY when the members carry INDEPENDENT noise. The 2026-08-13 few-spike GO
(`2026-08-13-gap1-A1-fewspike-...-population-coding-is-the-companion-process.md`) worked because its P candidate-pool
neurons were INDEPENDENT SPIKING SAMPLERS (OU membrane noise made each a stochastic winner -> a genuine sqrt(P) gain).
Here the word-pool members are DOWNSTREAM CONDUCTANCE REPLICAS of ONE shared noisy hidden population -> the limiting
read noise is COMMON-MODE across pool members -> summing + gain-normalizing removes exactly zero of it. A genuine
ensemble at this stage would need P INDEPENDENT HIDDEN populations (each with its own OU noise), which `--sub-pop` does
not create. **The word-pool ensemble lever is therefore inert; banked as an honest negative; escalate to the dendritic
lever (the pre-registered contingency).** (The prior in-code comment "graded read ~P-indep" is now a filed, mechanistic
verdict rather than a terse assertion.)

## STEP 2 — DENDRITIC (Urbanczik-Senn two-compartment) lever: BUILT, smoke-clean, staged

The dendritic contingency stops asking ONE noisy read to carry BOTH the answer and the teaching signal. Implemented in
`_wkv_mouth_readout_eprop_batched_substrate_derisk.py` behind `--dendritic` (default OFF), and invoked by the scaffold's
`_wkv_mouth_readout_snr_ensemble_dendritic_derisk.py --lever dendritic`.

Mechanism (all on the substrate, 0 host matmul on the forward):
- **BASAL** compartment (unchanged): the feedforward hid/hidinh graded-conductance margin — the prediction.
- **APICAL** compartment (new): a SECOND, target-driven population wired onto the SAME word-pools, reusing the
  bias-pop wiring template but TARGET-driven instead of tonic. A labelled-line excitatory teacher per (block, word)
  (`apical_e`, B*V) drives its own word-pool one-hot for the batch target; a tonic inhibitory baseline (`apical_i`)
  suppresses all pools so a non-target reads LOW. `apical_margin` is a SECOND substrate read (feature silent, target
  driven) -> the teacher enters through a spiking top-down pathway, never through the forward answer.
- **APICAL calibration** (once per seed, a real substrate read): measure m_target vs m_nontarget, set a unit map so
  sigma(apical) is a clean +/-spread one-hot regardless of raw drive magnitude (analogous to the conductance->logit
  `gain`).
- **LOCAL ERROR:** `err = sigma(apical) - sigma(basal)` (PER-UNIT sigmoids, target - prediction), NOT a cross-unit
  softmax over the noisy basal read (so read noise in one word no longer corrupts every other word's error). Update
  `W += lr * err @ h` (the delta rule toward the apical-delivered target) + weight decay. No weight transport, no host
  gradient.

### Byte-identical when OFF (verified)
A tiny numpy smoke run WITH the new code but WITHOUT `--dendritic`, diffed against a pre-edit baseline of the same
command: every determinism-bearing result field is identical — `seed_hash` (`a45d2385f84619f0`), `gain` (7.521613),
`sub_learned`/`sub_copied`/`sub_shuffle`, `hostlinear_recov`, `weight_cosine`, `verify_first_ce`, `go`, and the whole
`summary`. Only the two wall-clock fields (`learn_secs`, `shuffle_secs`) differ. All dendritic code is gated behind
`if self.dendritic` / `if args.dendritic`, and dendritic JSON keys are added only when the flag is on.
(Artifacts: `research/findings/raw/_wkv_mouth_readout_dendritic_offdiff_baseline.json`,
`research/findings/raw/_wkv_mouth_readout_dendritic_offdiff_afteredit.json`.)

### Smoke ON (runs end-to-end, anti-cheats wired)
Tiny numpy `--dendritic` smoke (`research/findings/raw/_wkv_mouth_readout_dendritic_smoke_on.json`, 12 gradient
steps — NOT decisive):
`dendritic` true; `host_matmul_on_learning_forward` **0**; `main_apical_reads` **12 == n_grad_steps 12**
(`apical_reads_match_grad_steps` true — the apical substrate read ran EVERY gradient step); apical calibration
m_target **-23.9** > m_nontarget **-38.2** (the teacher raises the target pool's current, correct sign);
`sub_shuffle_recov` **0.0003** (shuffle-teach collapses); `sub_freeze_apical_recov` **~0** (freezing the apical target
at FULL budget collapses -> the teacher is load-bearing, not a sigmoid-error artifact); `verify_first_ok` true. The
smoke's `go` is false and `sub_learned` ~0.39 as expected at 12 steps — efficacy is genuinely open and is decided by
the staged 6-seed run, not this smoke.

### Pre-registered GO-gate (the staged 6-seed eval)
Per seed, dendritic GO = **[ integrated (sub_learned recov >= 0.85 x sub_copied recov) OR decisive lift (sub_learned
recov >= 0.55, the WKV-fewspike midpoint over the ~0.37 plateau) ]** AND anti-cheats collapse (hostlin > 2x floor;
wcos_main > 3x floor and > 0.12; sub_learned > 2x sub_shuffle) AND dendritic anti-cheats (sub_learned > 2x
sub_freeze_apical AND apical_reads == grad_steps) AND forward_is_substrate (host_matmul == 0) AND verify_first_ok.
**Board GO = >= 5/6 seeds.** NO-GO is a verdict on this dendritic METHOD, to be banked with the next lever named
(e.g. P independent hidden ensembles; burst-multiplexed credit), never a wall.

### Staged command (queue AFTER the branch lands on main — the shared checkout the queue uses)
The branch `research/mouth-read-snr-dendritic` carries `--dendritic`; the live gpu_queue runs from the main checkout,
so this must be added AFTER the runner is on main (it was not merged here — topic-branch discipline):
```
bash tools/gpu_queue.sh add 'cd /home/dant123/Projects/sim && SIM_BACKEND=cupy /home/dant123/Projects/sim/.venv/bin/python -u -m research.runners._wkv_mouth_readout_snr_ensemble_dendritic_derisk --lever dendritic --coverage decisive --seeds 42,43,44,100,101,102 --out-dir research/findings/raw/_wkv_mouth_readout_snr_ensemble/dendritic'
```
This runs `_wkv_mouth_readout_eprop_batched_substrate_derisk --dendritic --forward substrate --sub-pop 1` on all 6
seeds at the module's decisive defaults (epochs 8, n-train-pos 9600, batch 48, sub-read-window 120), then aggregates
the >=5/6 verdict.

## Files
- `research/runners/_wkv_mouth_readout_eprop_batched_substrate_derisk.py` — the `--dendritic` mechanism (apical
  population, `apical_margin` read, `calibrate_apical`, the U-S local error, freeze-apical + shuffle-apical anti-cheats,
  dendritic GO-gate). Byte-identical when off.
- `research/runners/_wkv_mouth_readout_snr_ensemble_dendritic_derisk.py` — `--lever dendritic` now invokes it
  (was NotImplementedError); the ensemble-lever docstring updated to record the inert-by-construction verdict.

## Honest blocker
The dendritic mechanism's EFFICACY is unproven — the smoke confirms it runs, is byte-identical off, and is
anti-cheat-clean, but 12 steps cannot show a lift. The decisive 6-seed run (staged) is the verdict. Independent of it,
the ensemble word-pool lever is a genuine structural negative (common-mode noise), and a P-independent-hidden-ensemble
remains an untested alternative if the dendritic method also NO-GOs.
