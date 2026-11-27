# Local sleep rule screen prereg DRAFT (2026-11-27): sleep learning by a local delta rule instead of gradients

STATUS: DRAFT. Not registered until a fresh sonnet pre-data review has passed and the dev-seed settings (below) are frozen.
Milestone: ARCHITECTURE section 4, 2026-12-31 "Sleep learning by a local, biology-style rule instead of gradients; scale to 1.5B."
This is the de-risk screen at 0.1B then 0.4B; 1.5B is a later step and only after a GO here.

## Record brief
- **Knew.** Old track: a transport-free KP/FA delta rule over a fixed reservoir learned the grounded render as well as
  BPTT (0.91 / 0.86 vs 0.86; pure SGD 0.86), but converged slower on general fluency (ppl 63 vs 34 at 10k steps, 35 at 40k)
  (`research/findings/2026-07-20-wkv-cortex-biological-learning-CLOSE-...`). Deep local credit on spikes did not enter the
  learning regime (`2026-08-02-gap4-depth-rescue-untestable...`). New core: gradient sleep gives 3.8/12 hippocampus-off recall
  at 0.1B and 2.3/12 at 0.4B with drift KL 0.02-0.04 (`experiments/results/2026-11-02-sleep-0.1b-six-seeds.md`,
  `...-overnight-0.4b-kl3-st480.md`); SCAFFOLDS lists gradient sleep as the declared trade to replace.
- **New.** The local rule is applied to the existing rank-8 channel-mix adapters of the new core, with no backward pass
  through the network and none through time: output error (softmax minus one-hot; for the keep term, minus the pretrained
  softmax) sent back by fixed feedback pathways, update = pre-synaptic rank activity x post-synaptic error, Kolen-Pollack
  feedback copies, a local decay of B as the keep term.
- **Prerequisites.** Met: sleep consolidation harness with hippocampus-off, shuffled and no-sleep arms; adapters equal the
  pretrained cortex at zero. Missing: credit through the WKV state (this rule treats it as constant); replay is stored
  text (declared); the feedback presets other than "random" carry the error through transposed frozen weights (readout,
  residual identity, value weights), a declared weight-transport scaffold, so only "random" is fully transport-free.
- **Expected.** Weaker than the gradient arm (local rules converge slowly; no temporal credit). About 45% NEGATIVE or
  UNDEFINED, 35% GO on "mixed"/"transport" feedback, 10% GO on strict "random" (my estimate, not measured).

## Method
`brain/experiments/local_sleep.py` (arms and CLI) over the existing `brain/experiments/sleep_probe.py` battery
(`evaluate`: 12 told facts, hippocampus off; 4 never-told questions; drift = KL(on || off) on three ordinary texts;
episodes from `build_arm_episodes`; paraphrase is reported, not gated). One shared adapter initialisation per seed
(`torch.manual_seed(seed)`), same 240 steps, same replay and keep-bank texts in every arm.

## Arms (per model, per seed)
1. `local`: local rule, true episodes, KP feedback, preset "mixed" (readout transpose + residual identity + random hidden).
2. `gradient`: current `brain.sleep.sleep` (Adam lr 5e-4, KL weight 1), true episodes. The reference.
3. `no_sleep`: adapters at zero.
4. `local_shuffled`: the `local` arm on the shuffled episodes (names swapped, same exposure).
5. `local_random` (secondary, not in the GO rule): strict fixed random feedback, nothing transported.
Seeds 42, 43, 44, 100, 101, 102. Models: 0.1B `/home/dant123/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth`, then 0.4B
`rwkv7-g1d-0.4b-20260210-ctx8192.pth` (1.5B `rwkv7-g1l-1.5b-20261031-ctx34304.pth` only after a GO).

## Dev settings (seed 7 only, frozen before the six seeds)
Grid: preset in {mixed, transport, random} x lr in {0.01, 0.03, 0.1, 0.3} x decay in {0, 1e-3} = 24 local sleeps, plus one
reference run (`no_sleep` + `gradient`) on seed 7. Script: `brain/experiments/results/2026-11-27-local-sleep/dev_sweep.sh`
(one config per process, one json each). Pick, separately for each preset, the (lr, decay) with max hippocampus-off recall
on seed 7 subject to drift <= 1.25 x the gradient arm's drift on seed 7. `local` uses the pick of "mixed" or "transport"
(whichever has the higher recall under that constraint; ties go to "mixed", the less transported); `local_random` uses its
own pick from the "random" rows under the same drift constraint, not the `local` pick. If no config of a preset meets the
drift constraint, that preset's arm is run at its lowest-drift config and the fact is reported. No further tuning. A
toy-cortex test shows the rule lowers replay loss only at lr ~0.3 with the transport preset; the real-model lr is unknown,
so the sweep is required.

## Measures (all hippocampus off, per seed, per arm)
Recall of told facts /12 (substring of the key in the greedy answer); false recall /4; generic drift (mean KL); paraphrase
/12 (reported only); adapter norm; final replay loss.

## Registered rule (written before data)
Per model, over the six seeds, using means across seeds and counting seeds:
- **GO** if all hold: (a) mean `local` recall >= max(0.5 x mean `gradient` recall, mean `no_sleep` recall + 2), i.e. the
  effective bar is the larger of the two; the 0.5 x clause binds only when the gradient mean is >= 4 (0.5 g > no_sleep + 2
  with no_sleep ~0). At 0.1B (gradient ~3.8) the "+2" clause binds (bar ~2.0 vs 1.9); at 0.4B (gradient ~2.3) it binds
  too (bar ~2.0 vs 1.2), so at both scales the bar is "2 told facts above no_sleep" unless this run's gradient mean
  reaches 4; (b) `local` recall > `local_shuffled` recall on >= 5 of 6 seeds AND mean `local_shuffled` recall <= 1.0 (chance
  level; the shuffled gradient arm sat at 0.3); (c) mean `local` drift <= 1.25 x mean `gradient` drift; (d) mean `local`
  false recall <= mean `gradient` false recall + 0.5. Reference numbers from the earlier runs: 0.1B gradient ~3.8, 0.4B ~2.3; the bar is recomputed from this run's own `gradient` arm, and is checked against its counted
  maximum (12) before registration.
- **UNDEFINED** if the `gradient` arm's mean recall is < 2 (no reference to compare), or the local arm's adapter norm is
  zero or non-finite on any seed (the rule did not run), or `local_shuffled` recall > 1.0 (the control moved: this gives UNDEFINED intentionally, not NEGATIVE, because a moved control means the recall measure is not specific to the true episodes and the arm comparison cannot be read).
- **NEGATIVE** otherwise (a verdict on this rule and this preset, not on local learning; judge against the missing
  temporal-credit prerequisite before trying another method).
A GO at 0.1B alone does not pass; 0.4B must also GO before the 1.5B scale step. `local_random` is reported next to the
verdict: if `local` GOes and `local_random` does not, the claim is "local rule with transposed-weight feedback", not
"transport-free".

## Compute (estimate, not measured)
CPU fp32, no GPU. The local rule runs two no_grad forwards per step (replay + keep text), no backward: about a third of
a gradient step. Per seed at 0.1B: 4 sleeps + 5 batteries, order of 30-60 min; peak RAM under 4 GB. At 0.4B: 2-4 h per
seed, peak about 8-10 GB (gradient arm; weights 1.6 GB fp32; the 1.5B CPU training peak was 17.7 GB) so pool41 and pool42
(15 GB) fit, run under `tools/memcap.sh` after `tools/mem_ok.sh`. Layout: seeds 42/43/44 on pool41, 100/101/102 on pool42,
0.1B first; measure one smoke seed's time and RSS before the chain. Dev sweep: 24 local sleeps + 1 reference run on seed 7 at 0.1B, ~3-3.5 h on pool40 (~7 min per local config assumed, reference ~15 min; not measured; sequential, resumable).
Run: `python -m brain.experiments.local_sleep --model <pth> --seed <s> --steps 240 --lr <dev> --feedback mixed --out <json>`.
