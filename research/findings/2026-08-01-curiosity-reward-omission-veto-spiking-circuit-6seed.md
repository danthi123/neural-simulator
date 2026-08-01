---
type: finding
status: live
date: 2026-08-01
mechanism: curiosity-seek-learn
runner: research/runners/_curiosity_reward_omission_veto_derisk.py
artifacts:
  - research/findings/raw/lanes/curiosity_omission_veto_6seed_aggregate.json
  - research/findings/raw/lanes/curiosity_omission_veto_s42.json
  - research/findings/raw/lanes/curiosity_omission_veto_s43.json
  - research/findings/raw/lanes/curiosity_omission_veto_s44.json
  - research/findings/raw/lanes/curiosity_omission_veto_s100.json
  - research/findings/raw/lanes/curiosity_omission_veto_s101.json
  - research/findings/raw/lanes/curiosity_omission_veto_s102.json
---

# A spiking reward-omission circuit computes the DR-1 curiosity veto — the host ELP tracker is discharged (6/6 core, 5/6 composite)

## Why this ran

The 2026-08-01 finding
[`..._curiosity-veto-cannot-be-read-off-the-spiking-striosome-value-it-inverts.md`](2026-08-01-curiosity-veto-cannot-be-read-off-the-spiking-striosome-value-it-inverts.md)
established: DR-1 curiosity is GO, but its noisy-concept veto (the honesty anti-cheat — STOP asking about
UNLEARNABLE concepts) is still computed by a HOST Python ELP tracker (a TD low-pass fed by the SNc
paired-subtraction `snc_B - snc_A`). Thresholding the spiking striosome VALUE FAILED 0/6 (it inverts: a noisy
concept reads a HIGHER value because reward-independent STDP drift inflates it). That finding located the clean
spiking separator — the SNc reward burst (learnable 4-31 Hz vs noisy 0.0 Hz every seed) — and named the next
method: a spiking reward-OMISSION detector (SNc dip / lateral-habenula) that gates the ASK pool DOWN for
concepts that yield no reward burst, replacing the host arithmetic with substrate computation.

This runner builds that circuit and tests whether NEURONS reading reward-omission can compute the veto.

## The circuit (all additive — NO `sim/` edit; new BrainRegions + RegionPathways in the runner only)

```
reward_us --(exc)--> rmtg --(gaba_a, inhibitory)--> omit --(exc)--> veto <--(PLASTIC exc)-- cue
                                                     (tonic)                      reward_us --(exc)--> veto
```

- `omit` (lateral-habenula / RMTg reward-absence detector, RS, tonically driven). The delivered reward US drives
  the GABAergic relay `rmtg`, which silences `omit`. So a DELIVERED reward -> omit LOW; a reward ABSENCE (no
  learning-progress US) -> omit HIGH. The "tonic minus reward" subtraction is done by SYNAPTIC INTEGRATION, not
  by a Python `snc_B - snc_A`. Keying on the reward US (not the V-cancelled SNc burst) is what makes it robust:
  a partially-learned concept whose striosome value V has caught up still reads its US, so it is NOT mistaken
  for an omission (the confound that would have re-broken the striosome-value approach).
- `cue -> veto` PLASTIC — the per-concept veto MEMORY, i.e. the substrate replacement for the host ELP tracker.
  An LHb-opponent three-factor rule: veto fires on EITHER ask type (via `omit->veto` on absence, `reward_us->veto`
  on reward), so `cue[c]+veto` STDP eligibility builds every ask; the UPDATE SIGN is the omit read against a
  tonic baseline — `veto_rpe = clip((omit_read - OMIT_MID)/OMIT_RPE_SCALE)`. Reward-absence (omit high) ->
  POTENTIATE; reward (omit low) -> DEPRESS (the protective reserve for concepts that were ever rewarding). A
  NOISY concept (always absence) accumulates `cue->veto`; a learnable one (early reward depresses) does not,
  until it is mastered out by the novelty gate.
- The veto DECISION is a spiking read: drive `cue[c]` alone (a `veto_drive` transmission gate isolates the read
  to `cue->veto`) and read the `veto` pool rate; HIGH -> vetoed. This is thresholded exactly like DR-1's own
  accepted `WANT_FLOOR` / `NOVEL_THRESH` gates.
- Per-pathway `set_plasticity_gate` keeps the striosome critic and the veto accumulator learning in disjoint
  windows, so the DR-1 critic (gates a/b/c) is byte-unaffected.

## Result — the spiking veto works; the host ELP is discharged

6 seeds (42/43/44/100/101/102), numpy backend, DR-1's real config (8 learnable / 4 noisy, 220 turns, 30 asks).
Aggregate: `research/findings/raw/lanes/curiosity_omission_veto_6seed_aggregate.json`.

**Core conversion: 6/6.** Every veto-specific gate passes on every seed — corr(gap,want) >= 0.9; ask-unknown >=
2x known; conf rises above the abstain floor; NOISY STOPS (spiking veto fires) while its gap stays HIGH; moat
holds; curiosity-lesion silences asking; permuted-teacher collapses corr; and the load-bearing dissociation for
THIS conversion — **the OMISSION-DETECTOR LESION collapses the veto on 6/6**.

| seed | reward-absence detector (omit Hz) learn / noisy | spiking veto (Hz) noisy / learn | omit-LESION noisy veto | mastered real / yoked |
|---|---|---|---|---|
| 42  | 9.3 / 73.3  | 45.8 / 25.1 | 2.0 | 7 / 5 |
| 43  | 4.3 / 89.9  | 68.5 / 26.2 | 9.0 | 8 / 1 |
| 44  | 2.4 / 104.2 | 113.0 / 28.1 | 3.0 | 6 / 0 |
| 100 | 9.9 / 72.9  | 47.8 / 27.4 | 5.5 | 8 / 0 |
| 101 | 2.0 / 84.0  | 74.8 / 21.4 | 6.5 | 3 / 4 |
| 102 | 2.8 / 120.1 | 138.2 / 49.9 | 6.0 | 4 / 0 |

The detector separates cleanly on all 6 (reward -> omit 2-10 Hz vs absence -> omit 73-120 Hz), and the veto
tracks it (noisy 45-138 Hz vs learn 21-50 Hz). `tools.lab.attributable_to(noisy-veto real vs omit-lesion)`
reads ~90-97% of the veto firing attributable to the detector (noisy veto 45-138 Hz intact -> 2-9 Hz lesioned):
the veto is genuinely produced by the spiking omission circuit, not by baseline cue drive. This is the exact
opposite of the host-ELP veto, which SURVIVED the GABA_B critic lesion 6/6 — the striosome is now load-bearing
for the decision precisely because the veto reads a detector, not a Python tracker.

**Composite (add the inherited DR-1 yoked-random control): 5/6.** Only seed 101 misses, and it misses ONLY on
`yoked_collapses` (yoked mastered 4 >= real 3), a consequence of the residual below, not of the veto itself.

## Honest scope — the residual, named for the next lever

The veto value OVERLAPS across classes: learnable reads 21-50 Hz, noisy 45-138 Hz — separated per seed, but the
distributions touch (seed 102 learn 49.9 vs seed 42 noisy 45.8). The cause is real and is the hard core of this
problem: a slow-to-master learnable concept that is re-asked while still novel but with ZERO learning-progress
is, per-ask, indistinguishable from an unlearnable one (both deliver no reward US). The protective depression
that should hold such a concept below the floor SATURATES at the excitatory weight floor (~0), so after a few
zero-progress re-asks the veto climbs and the concept is vetoed before it finishes mastering. This drops real
mastery to 3-4 on 2/6 seeds (101, 102) and, on 101, low enough that the (also-degraded) yoked control matches
it. The core honesty capability — noisy STOPS via a brain-computed veto, learnable continues, detector-lesion
collapses it — holds 6/6; the residual is a mastery-efficiency loss on slow learners, not a failure to veto.

The next lever is a deeper / decaying protective reserve (reconsolidation or a sub-baseline inhibitory veto
memory) so a concept that was EVER rewarding is not vetoed on later zero-progress re-asks — the only per-ask
signal cannot separate "mastered-but-still-novel" from "unlearnable"; that separation lives in the HISTORY, and
the history reserve is currently floor-limited.

## What it does and does not mean

- It does NOT touch the DR-1 GO: the striosome critic learns in a frozen, disjoint window (gates a/b/c hold 6/6).
- It DISCHARGES the named shortcut: the per-concept veto value that DR-1 kept in a host Python ELP TD low-pass is
  now the `cue->veto` synaptic weight, written by an opponent three-factor rule gated by a spiking reward-absence
  detector, and read as a spiking pool rate. The residual host ops (reading a pool rate, a baseline-subtract +
  scale, a floor threshold) are the SAME accepted patterns DR-1's own GO relies on (SNC_SCALE, WANT_FLOOR,
  DA-release-proportional-to-firing).
- Backend/config: numpy, DR-1's real config, 6 seeds — the same backend the striosome-value negative used, so the
  comparison is like-for-like.
