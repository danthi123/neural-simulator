# Arousal as gain: plan (workflow wf_561b659e-0f4 synthesis, 11-18; unedited)

**(A) Record brief**

- **Pick:** a graft. The base is **uptake-gain**, which the judges scored highest (5.5 and 4). It is the only design whose gain acts on the cortex's own fast memory. The other two put a host knob on the decoder (temperature) or on forgetting (decay), and the judges flagged both as host decisions.
- **Fix 1, source:** take arousal from the two highest per-token surprise values in the heard turn, not the mean over the whole turn. The surprise read also produces the state for the reply, so it costs no extra forward pass.
- **Fix 2, nulls:** add a futility gate on dev, conversations as the statistical unit, a CONSTANT null with a real bar (+0.10, CI excluding 0) and a CURRENT arm with no integrator, used only to choose the claim's name.
- **Claim scope:** arousal only. Valence and mood stay blocked at 1.5B.
- **Correction:** the own-gain probe tested only the brain's own reply at gains of 1 or less. A heard-turn gain above 1 has never been probed.
- **Most likely failure:** the gain amplifies echo or invented meanings rather than real uptake of the event. The next method is then per-token gain on the surprising tokens only.

**(B) Build plan (smallest first)**

1. **`brain/cortex.py`:** add `Cortex.read_nll(text, state)` after `read` (66-69). It runs one `model.forward(ids, deepcopy(state), full_output=True)`, the same call as `novelty_stage0.item_nll` (72-84), and returns `(last_logits, state, per_token_nll)`.
   - `LiveCortex.forward` (`brain.py` 124) has no full output. While the switch is on, `hear` raises if `self.cortex.active`, so the scope is brains that have not slept. Declare this.
2. **`brain/affect.py`:**
   - `drive_arousal(x)`: `s = clip((x-MU)/SD, 0, 3)/3`, then `arousal += (s - arousal)/TAU_AROUSAL` (TAU 1.5, line 148). It does not touch mood. `update` (309) is not called while this switch is on, so there is one arousal definition at a time.
   - `encode_gain(span, pinned)`: returns `min(1 + span*arousal, 2.0)`, or 1.0 if pinned.
   - Add `surprise`, `encode_gain` and `gain_on` to `readout()` (327).
3. **`brain/brain.py`:**
   - Add switch `Brain(arousal_gain=None)` and `arousal_pin=False` (172-181), documented beside `own_gain` (231).
   - In `hear`, replace line 414 (switch on only):
     - `_, su, nll = raw.read_nll("User: {text}\n\n", active)`, then `drive_arousal(mean(top2(nll[1:])))`.
     - `su_g = add_delta(su, matrix_delta(su, active), g-1)`.
     - `logits, st = read(self._asst, su_g)`.
   - In `commit_exchange` (444-456): the heard-turn part takes the same gain on its matrix slots. The reply part keeps the current rule.
   - Add `info["arousal"] = affect.readout()`.
4. **`brain/SCAFFOLDS.md` row:** "Surprise normalisation MU/SD, arousal-to-encoding-gain map `1+SPAN*a` (ceiling 2), turn-level gain on heard-turn matrix write (`affect.py`, `brain.py hear/commit_exchange`); replace with an LC-like population whose rate sets per-token write gain."
5. **Runner `brain/experiments/arousal_gain.py`:**
   - Subcommands: `smoke`, `trace`, `pilot`, `dev`, `test`.
   - Arms: OFF, ON, LESION, SHUFFLED, CONSTANT, CURRENT.
   - Frozen mechanical measures: carry of the coined token or event noun, `_is_copy`, fluency ratio, questions-back, "User:" leaks.
6. **Smoke (about 10 min, no sampling):**
   - S1: split read versus joint read gives logits within 1e-5, and 20 greedy turns are byte-identical. If not, every arm, OFF included, uses the split path.
   - S2: LESION is byte-identical to OFF.
   - S3: `trace` on the dev bank. Filler turns must have gain ≤ 1.1 on at least 90% of turns.

**(C) Section 19 of `brain/research/2026-11-03-mood-state-design.md`**

**19. Iteration 6 (11-19): arousal as encoding gain. PRE-REGISTERED before any run**

**Method**
- **Drive:** the mean of the two highest per-token NLLs of the heard turn, from the current state.
- **Calibration:** MU and SD come from V5_DEV neutral events plus 12 ordinary sentences read from birth. They are disjoint from the stage-0 items and from both banks.
- **Gain:** `g = min(1 + SPAN*a, 2)` multiplies the heard turn's matrix-memory write, both for this reply and in the committed state. No direction and no words are added. The gain never goes below 1.
- **Conversations:** 4 turns each.
  - T1: an event, either surprising (a coined item or an unexpected event) or mundane (the same frame with a common noun).
  - T2 and T3: fixed mundane fillers.
  - T4: the open probe "How's your day going?".
- **Banks:**
  - Dev: 12 conversations (6 surprising, 6 mundane).
  - Test: 24 conversations (12 matched pairs), with coined words new to every earlier run.
- **Arms** (same sampling seed per conversation):
  - OFF.
  - ON.
  - LESION: g pinned at 1, arousal still computed and logged.
  - SHUFFLED: the g sequence of another conversation at the same seed.
  - CONSTANT: the mean ON g on every turn.
  - CURRENT: TAU = 1, no carry-over. This arm decides only the claim's name.
- **Seeds:** 42/43/44/100/101/102, 2 samples each. Seeds only resample the replies, so **the unit is the conversation**, and the paired bootstrap resamples conversations.
- **Readers:** Haiku and Sonnet, blind to arm and class, with order shuffled. Each sees the T1 text and one reply and gives three labels:
  - (a) takes up the T1 event;
  - (b) echoes or copies the user;
  - (c) invents a meaning.
- **Reader agreement:** Cohen's kappa on (a) must be at least 0.6 over a 40-item overlap. If it is lower, the verdict is INCONCLUSIVE.

**Dev rule** (frozen and committed as 19.1 before test)
- **Pilot (seed 42, SPAN 1; OFF, ON, CONSTANT; about 100 replies, 20 min):**
  - Gate: mechanical carry at T2 and T4 must be ON − CONSTANT ≥ +0.10, with no rise in echo.
  - If it fails, stop with a NO-GO for the method and go to branch (iii).
- **Dev (seeds 42/43/44, SPAN in {0.5, 1, 2}):**
  - Choose the SPAN with the highest mechanical carry of ON minus CONSTANT, subject to fluency ratio ≤ 1.15 and echo rise ≤ 0.05.
  - Futility: if that SPAN gives less than +0.10 pooled, stop.
  - Floor: if LESION's T2 carry on surprising conversations is below 0.15, the bank has failed. Reframe it once and do not tune SPAN.
  - Readers are never used on dev.

**Criteria** (test bank; all must hold except the claim name)

| | Measure | Bar |
|---|---|---|
| C0 | Arousal after T1, surprising vs mundane | AUROC ≥ 0.90; filler gain ≤ 1.1 on ≥ 90% of filler turns |
| C1 | Blind (a) rate at T2+T4, surprising conversations | ON − LESION ≥ +0.15, 95% CI > 0, positive on ≥ 5/6 seeds |
| C2 | Shuffled and constant nulls | ON − SHUFFLED ≥ +0.10 and ON − CONSTANT ≥ +0.10, each with CI > 0 |
| C3 | Selectivity, mundane conversations | \|ON − LESION\| ≤ 0.05 |
| C4 | Lesion changes the REPLY | ≥ 50% of ON replies differ from LESION on turns with g ≥ 1.2; ≤ 10% differ on turns with g < 1.05 |
| C5 | Harms (ON − LESION) | echo (`_is_copy` or reader b) ≤ +0.05; invents (c) ≤ +0.05; questions-back ≤ +0.05; no rise in "User:" leaks; fluency ≤ 1.15 |
| Name | ON vs CURRENT | ON − CURRENT ≥ +0.05: call it "arousal state sets encoding gain". Otherwise "surprise-gated encoding". |

**Decision branches**
- (i) All of C0-C5 hold: GO for "arousal shapes replies (lesion-proven)", under the earned name. Then run land-result with a fresh skeptic. The switch stays OFF.
- (ii) C1 holds but C2 fails: plain heard-turn gain works, but arousal timing is not shown. NO-GO for the arousal claim.
- (iii) C5 fails, or the pilot or dev futility gate fails: NO-GO for flat turn-level gain. Next is per-token gain on the matrix write of the high-NLL tokens only.
- (iv) C0 fails: the drive unit is wrong. Re-calibrate on dev only.

**(D) If the milestone is not met on 11-30, report:**
1. **Read-out works.** Brain-native surprise, through arousal, separates novel from ordinary input (C0 numbers) and habituates.
2. **Mood half is blocked at 1.5B.** The additive valence family failed: mood4 did not pass, and mood5 Stage A had neutral events reading |v| 0.24-0.30 against a bar of 0.15. Only the negative appraisal sign holds (6/6 and 3/3). This is a judgment against scale: at 1.5B the appraisal axis is not separable from topic. It is not a verdict on mood.
3. **Arousal-gain result.** Give the exact branch reached (ii), (iii) or (iv), with the per-seed numbers read from the artifact, and the failure mode (echo, invented meanings, or floor).
4. **Next method and its prerequisite.**
   - Per-token write gain on surprising tokens, which is closer to LC phasic gain.
   - Then NE gain on the per-head decay inside the forward.
   - Then sleep-time consolidation of high-arousal episodes. The biological prerequisite named for this is hippocampal tagging.
   - Re-date the milestone with a scaled timeline.

Files referenced:
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/brain.py`: `hear` 386-438, read at 414, `commit_exchange` 444-456, `LiveCortex.forward` 124, switches 172-181 and 231.
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/cortex.py`: `read` 66, `speak` 76, `matrix_delta` 119, `add_delta` 129.
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/affect.py`: TAU 148, `update` 309-312, `readout` 327.
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/experiments/novelty_stage0.py`: `item_nll` 72-84.
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/SCAFFOLDS.md`.
- `/home/dant123/Projects/sim/_scratch/wt-mixday/brain/research/2026-11-03-mood-state-design.md`: its last existing section is 17.