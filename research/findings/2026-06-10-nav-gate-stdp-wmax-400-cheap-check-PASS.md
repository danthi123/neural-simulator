# Nav-gate (a) cheap-first check — stdp_w_max=400 does NOT regress navigation (the actor over-grows but inertly) (2026-06-10)

**Roadmap step 2, STEP 2a, navigation acceptance gate (a), the cheapest-first check** (per
`docs/plans/2026-06-10-nav-episode-integration-design.md` §"cheapest-first check"). The #1 could-be-NEGATIVE
risk of merging the conversational populations onto the navigation bridge is that the 5a clip mitigation —
raising `cfg.stdp_w_max` from the navigation's own 150 to 400, to keep the ungated reward clip from moving the
~300-weight frozen parser role-routes — lets the navigation actor (cortex→D1) over-grow and regresses
navigation.

## The check

A single-seed standalone navigation episode at the G v2.5 + K v2 recipe (grid-32, deterministic, seed 42,
1800 steps — the tight ±0.11 band so a regression is visible), run twice via the new additive `--stdp-w-max`
override (`g11_bg_runner.py`, default `None` = the computed 150 = byte-equivalent standalone), with the
NAV-GATE(a) probe recording the max cortex→D1 actor weight.

| | wmax150 (baseline) | wmax400 |
|---|---|---|
| nav score = sum of `final_quarter_mean_distance` over 4 phases | **2.0** | **2.0** |
| per-phase distances | [0.496, 0.504, 0.496, 0.504] | [0.496, 0.504, 0.496, 0.504] (byte-identical) |
| `actor_max_cortex_to_D1_weight` | 150.0 (at the ceiling) | **311.07** |

## Verdict: PASS — navigation is not regressed

**The navigation score is byte-identical at stdp_w_max 150 vs 400** (the deterministic episode selects the
exact same actions). Raising the clip ceiling to protect the frozen conversational weights does not change
navigation behaviour. So the 5a clip mitigation (`stdp_w_max=400` on the merged bridge, above the ~300 frozen
parser role-route) is safe for navigation, and the merge can proceed.

## The honest nuance (documented, not hidden): the actor is ceiling-bound, not soft-bound

The integration design assumed "the actor asymptotes well below its bound by soft-bound design." That is
FALSE here: the actor max weight is exactly 150.0 at the 150 ceiling and grows to 311.07 at the 400 ceiling —
it is **hitting the ceiling**, not soft-bounding below it. So raising the ceiling DOES let the actor grow
more (150 → 311). BUT this over-growth is **inert** for navigation: the spiking winner-take-all
action-selection readout saturates — once the cortex→D1 drive is strong enough to win, more drive (311 vs
150) selects the same action — so the byte-identical score. (311 < 400 stays within the new ceiling, and the
frozen conversational weights at ~300 are plasticity-gated to gain 0, so they never grow regardless.)

This is a real characterization of the navigation actor (ceiling-bound + a saturating readout), not a
problem: the thing that matters for gate (a) — the navigation score — is unchanged. The full 6-seed
navigation gate will confirm the over-growth stays inert across seeds (the byte-identical single-seed score
is strong evidence: a weight difference of 150 vs 311 producing identical per-phase distances means the
readout is robustly in its saturated regime, not coincidentally matching).

## Proceed

Phase 1 PASSES → build Phase 2 (the (C) hybrid integration: the additive `run_moving_goal_episode` params +
the conv-finalization hook, so the navigation episode runs on the merged bridge with the conversational
populations frozen) + the single-seed navigation-on-the-merged-bridge smoke (anti-cheat A2: the frozen
conversational weights stay byte-identical across the episode under the live navigation reward-STDP+dopamine
stressor — the 5a isolation, now in vivo). Then the full 6-seed navigation gate (a).
