# Capability: Lifelong learning
# Record: brain/experiments/results/2026-11-20-rank-raw-screen-1.5b.md; 2026-11-21-stage1-rehearsal-s318-dev-1.5b.md
# Expected-fail: none (a research round)
# Hub research round (before-build 7, 2026-11-21): verdict + adversarial critique

The critique (section B) overrides the verdict (section A) where they disagree. Outcome: D1 dropped (no headroom: the 11-13 confidence gate oracle), the decay-clock rule dropped (M-use at margin 0), static CSLS = arm D plus an offset; the hippocampal next step is a D + fire-gate screen through `life_screen.py rank`. The main effort goes to the CORTEX side, where the stock "you're Dan" reply is produced with the hippocampus off from night 10: Stage-2 D2-PR (`2026-11-21-replay-priority-prereg.md`).

## A. Verdict (synthesis agent)

# Hub verdict: research round, 2026-11-21

Scope: the "hub" defect, where one or a few episodes win recall for many unrelated cues over a multi-week life. This is a read-only verdict and no repo files were changed. Sources are the record report, the literature report, and the code as read (`brain/hippocampus.py:122-218`, `brain/brain.py:154-170`, `brain/PREREQUISITES.md` §Memory selection, `brain/ARCHITECTURE.md` §4, `brain/research/2026-11-19-multiweek-life-design.md` §9). No TEST/S1REG results were opened. Claims tagged **[inferred]** are mine and not measured.

---

## 1. What we know

**The defect.** Evidence is DEV seed s318, 15 nights (`experiments/results/2026-11-21-stage1-rehearsal-s318-dev-1.5b.md`):
- Wrong fetches rise to 7-8 of 8-10 direct questions by nights 14-15.
- The top episode takes 0.5-0.6 of live fetches.
- "Do you remember X?" gets "Of course, you're Dan, my friend." for most X. That reply comes from the permanent identity episode.
- A context cue fetches the right episode, but the reply names another told fact.

**Levers on record: 15 in all.**
- 2 levers aimed directly at the hub were built and killed.
- 8 related levers were measured. None fixed hub selection.
- 5 were refuted on paper or never built.

| # | Lever | Verdict | What it taught | Source |
|---|---|---|---|---|
| 1 | **M-use** (use only strengthens if raw sim confirms the winner by 0.02) | KILLED (over-gating: denies 3 of 4 right fetches) | Raw margins fall below 0.02 on about 69% of fetches by night 7. On s310 the weighted winner differs from the raw winner on 100% of wrong fetches. | `2026-11-19-life-screens-1.5b.md` |
| 2 | **R-raw** (rank by raw similarity; strength only gates firing). Arms A/D/B/R | KILLED on K1/K2 | **Arm D (S0 raw) dissolves the hub**: top-1 share 0.50-0.71 drops to ≤0.22 on 4 of 4 hub homes; right fetches 8 -> 14, wrong 23 -> 20. But it fires on 41 open/greet/never turns (K3 fail). Arm R (S_all raw) seldom fires because the bar and key are confounded. | `2026-11-20-rank-raw-screen-1.5b.md` |
| 3 | S_all key + unslept bar | Default kept | After sleep the bar seldom fires. Forced top-1 is right on 0.86. | `2026-11-14-key-default`, `2026-11-18-sall-open-conv`, `-sall-slept-fit` |
| 4 | S_all + slept bar 0.165/0.02 | Selection MET, replies FAIL | In a life it cuts fetches 70 -> 31 and wrong 23 -> 9, but right stays 8 -> 8. It stops firing; it does not choose better. | `2026-11-18-sall-slept-*`, R-raw arm B |
| 5 | Cortex-confidence gate | NARROWED, not built | Told 9.17 vs hippocampus-off 9.33. Best-of-both oracle only 9.67, so little room at the reply level. | `2026-11-13-confidence-gate-1.5b.md` |
| 6 | "Do I know this" signals (cue_response, mean_logp) | Negative | cue_response AUROC 0.84 for same wording, 0.61 reworded. It tracks overlap with replayed wording, not knowledge. | `2026-11-17-know-signal-told-vs-never-1.5b.md` |
| 7 | M1 recall-to-reject | FALSIFIED | AUROC: F 0.70, C_mean 0.62 | `2026-11-17-confab-m1-1.5b.md` |
| 8 | M4 near-miss guard in sleep | Not a lead | Targets false recall in the slept cortex, not selection | `2026-11-14-near-miss-guard-1.5b.md` |
| 9 | Lived key / speaker contrast (+S_all, dose, mixed day) | NO-GO | Limited by routing: S0 fetches the right episode only about half the time. The "aunt -> Oslo" lure comes from routing. | `2026-11-14-lived-key`, `-speaker-contrast-live`, `-sall-contrast-*` |
| 10 | M-scale / SAB / SABE (adapter downscaling) | KILLED | Big gain in slept-fact recall, but wrong fetches unchanged. **H with hippocampus off is about equal to intact H**: a fetch seldom moves the reply. | `2026-11-20-life-mscale-*` |
| 11-15 | M-eff, fan normalisation, decay-clock rule, F1 trace turnover, **D1 disengagement** | Not tested | M-eff cannot run on a slept cortex. Fan normalisation was refuted on paper (fixed point at 0.4-0.6 hub share). Decay-clock was named as a fallback. D1 has no code, no prereg and no result. | `research/2026-11-20-hub-rank-raw-prereg.md` §0; design §9 |

**How recall works now** (`hippocampus.py`):
- Score = centred cosine of the S0 layer-0 key × tanh(effective_strength).
- Fetch if the best score ≥ 0.12 (question) or 0.45 (statement), and its margin over the runner-up ≥ 0.02.
- **Every fetch, right or wrong, does `strength += 0.5`, `uses += 1`, `last_used = now`** (lines 192-196). Resetting `last_used` also resets decay.
- Permanent identity episodes use max(strength, 3.0), so tanh ≈ 1.
- Sleep replays each episode weighted by effective_strength (`brain.py:772-782`), so the hub also gets most of the replay.

## 2. Cause, as far as the data says

The data shows three layers. Only the first has been shown to dissolve.

**Layer 1: strength feedback (measured; it amplifies the hub).**
- s311: the "Bramble" episode reached uses 10, strength 6.0, and was fetched 86 times.
- About 9 of 10 s311 hub fetches, and 100% of s310 wrong fetches, are weighted-only winners.
- Frozen-use top-1 share is 0.31, vs 0.56 when use self-boosts (s310). The probe itself feeds the hub.
- Max non-permanent strength rises from 1.5 (night 1) to 3.5-4.5 (night 7) on the s315-318 runs, in step with wrong fetches.
- R-raw arm D, where strength cannot make a winner, drops top-1 share to ≤0.22.

**Layer 2: key geometry plus the permanent-episode floor (measured; it is the floor).**
- On night 1, with only 8 stored episodes, wrong fetches are already 4 on all four seeds.
- 4 of 5 (s310) and 4 of 7 (s312) hub fetches are also the raw winner, so strength is not needed for them.
- Raw margins fall below 0.02 on about 69% of fetches by night 7. Centred cosine over about 50 near-template episodes ("My X is called Y") stops telling them apart.
- PREREQUISITES §Memory selection already lists this: "layer-0 delta key is lexical", "missing: separation of confusable facts".
- The identity episode wins question-shaped cues at tanh ≈ 1, and nothing limits that weight.

**Layer 3: slept facts leaking into replies (measured; outside the hippocampus).**
- Hippocampus-off H is about equal to intact H in every dev life and M-scale arm.
- 4 of 6 right replies on s310-312 followed a wrong fetch.
- In s318 a right fetch still produced a reply naming another fact ("Patagonia").
- **[inferred]** Fixing selection fixes the "you're Dan" reply and wrong-episode reinstatement. It will not by itself raise H much.

**Measurements that would separate the causes** (all offline, on stored homes; see §4):
1. **Strength vs geometry.** Score the night-14/15 battery three ways: weighted (A), raw (D), and equal strength for all with permanent episodes kept. Wrong fetches that stay wrong under raw ranking are geometry; those that flip to right are strength. Also report the share of the identity episode's wins that remain when its weight is set to the median weight.
2. **Hubness of the key space.** Compute the k-occurrence N_k (k = 5) over the stored keys, using K·Kᵀ with each episode's key as a cue, and its skew. Compare raw S0, centred S0, whitened S0, and S_all. If skew is high and whitening cuts it, geometry is the main fixable part.
3. **Reply headroom.** Run an oracle-fetch probe: reinstate the ledger-correct episode on each H question (one `life.probe` run with generation, per home). If oracle H minus intact H is < 1 per seed, the reply-level gain from any hub lever is capped, and the hub is mainly a wrong-reinstatement and interview defect. That is still worth fixing, but expectations should be set accordingly.

## 3. Ranked candidate methods

### Rank 1: H1, per-episode excitability normalisation (hubness-corrected ranking)

This merges literature candidates 1, 2 and 3. A static fixed-point form is screened first; the online homeostatic form is built only if the static form passes.

- **Mechanism in brain terms.** Each episode (engram) has an excitability offset that grows with how broadly its key matches.
  - Static form: score_i(q) = sim_i(q) − r_i, where r_i is the mean of episode i's top-k similarities to the other stored keys (CSLS). The z-score form uses (sim_i − μ_i)/σ_i over that same reference bank (mutual proximity).
  - Online form: θ_i ← θ_i + η(win_i − p_target) on each competition, relaxing toward 0 in sleep.
  - Strength leaves the ranking (as in R-raw D) and only gates firing.
  - The fire test runs on the normalised score, asking whether this cue is unusually close *for this episode*. **[inferred]** This is what could fix arm D's K3 failure (41 open-turn fires), since a broad-matching episode no longer clears the bar on generic turns.
- **Biology.** Homeostatic intrinsic plasticity and firing-rate homeostasis (Desai et al. 1999; Turrigiano 2008), divisive normalisation (Carandini & Heeger 2012), and adaptation preventing capture by one attractor (Hopfield 2010). The ML counterparts are CSLS (Conneau et al. 2018), mutual proximity (Schnitzer et al. 2012), and conscience learning (DeSieno 1988).
- **Host decision.** None that reads text. The new constants are k, the bar and the margin on the normalised score, and p_target/η for the online form. Declare them in `SCAFFOLDS.md` next to the existing bars, capped as the bars are (one global value, fitted on night-7 DEV, never per-episode). The reference bank is the store's own keys, so no text corpus is needed.
- **Code.**
  - `hippocampus.py`: a class switch beside `RANK_ON_RAW` (lines 183-188), e.g. `RANK_RULE = 'csls'|'mp'`. It computes r_i (or μ_i, σ_i) from the stored keys inside `similarities` (lines 154-164) or a sibling function, and passes the normalised score to `_compete`.
  - Online form: a θ field on `Episode`, persisted in `episodes.json`.
  - `brain.py:443-444`: a `Brain(rank=...)` kwarg; `_hippocampus_for` at lines 448-463 for keyed hippocampi.
  - `life_screen.py`: a `RANK_ARMS` entry.
  - Keep the existing USE_BOOST but take it out of ranking, so strength still decides firing and replay.
- **Prerequisites.** Memory selection is PARTIAL. This lever targets the listed gap ("separation of confusable facts"). The lexical key is still a code substitute, so H1 corrects geometry it cannot change. It needs no consolidation prerequisites. The limit at 1.5B is the key, and scale is not the issue.
- **Falsifiable prediction.** On the night-14/15 DEV homes s315-s317 with frozen use:
  - top-1 share of fetches ≤ 0.30 on all three homes;
  - right fetches ≥ arm D (S0 raw), pooled;
  - open/greet/never fires ≤ arm A;
  - identity-question right rate not below arm A.
  If any one of these fails, H1 is refuted as a method.
- **Cheapest screen.** About 1 CPU-hour of dumps (§4), then a sweep in seconds.

### Rank 2: H2, whitening / top-PC removal of the key space (literature candidate 4)

- **Mechanism.** At sleep, fit a transform on the stored keys and apply it to stored and cue keys alike, at recall only; stored keys stay unchanged. Options:
  - per-dimension standardisation;
  - projecting out the top d principal components (d = 1-8);
  - shrinkage ZCA, (C + λI)^(-1/2).
  Re-fit it each night.
- **Biology.** Decorrelation by anti-Hebbian rules, and subtracting the mean activity at storage (Tsodyks & Feigelman 1988). ML: Mu & Viswanath 2018; Timkey & van Schijndel 2021; Suzuki et al. 2013.
- **Host decision.** None that reads text. d and λ are declared constants.
- **Code.** `Hippocampus.similarities` (lines 159-164 already centre; extend there). Fit in `Brain.sleep` and persist the projection in the home.
- **Prerequisites.** Same as H1.
- **Prediction.** On night-14 homes the share of fetches with raw margin < 0.02 falls from about 69% to < 40%, and N_k skew falls. Right fetches under raw ranking (arm D + whitening) beat arm D. If the margin share does not move, geometry is not the fixable part.
- **Screen.** The same dump as H1, run in the same sweep. **It is not a separate Stage-2 slot**: it is a preprocessing axis crossed with H1.

### Rank 3: H3, sleep strength homeostasis plus a decoupled use boost (literature candidate 9, with the decay-clock fallback)

- **Mechanism.** Nightly after replay, rescale non-permanent strengths to a conserved budget (s_i ← s_i·S/Σs), or compress them (s_i ← s_i/(1+λ s_i)).
  - Reset `last_used` only when a fetch was also the raw winner (the decay-clock rule; this is a reset rule, not M-use's strengthening gate).
  - Cap replay weight per episode per night, so the hub stops taking most of the replay.
- **Biology.** Synaptic homeostasis in sleep (Tononi & Cirelli 2003, 2014); synaptic scaling (Turrigiano 2008); prioritised replay of weak items (Schapiro et al. 2018).
- **Host decision.** None that reads text. The budget S and λ are declared.
- **Code.** `Brain.sleep` (`brain.py:772-782`) for the rescale and the replay cap. `_compete` lines 192-196 for the clock reset.
- **Prerequisites.** Forgetting/strength is PARTIAL and sleep consolidation is PARTIAL, so both are present.
- **Prediction.** In a 15-night life, max non-permanent strength stays ≤ 2.0 at night 14 (vs 3.5-4.5), and wrong fetches at nights 14-15 are ≤ half of baseline.
- **Screen.** Only first-order offline: rescale the stored strengths and re-score. Formation needs a life. **This is the only candidate that targets hub *formation* and replay dilution; H1 and H2 only fix selection at a given state.**

### Rank 4: D1, hippocampal disengagement (cue_response adapters on minus off)

Honest standing:
- **It was not effectively tested as a hub lever.** The 11-13 confidence gate chose per cue between the cortex's reply and the reinstated reply. D1 removes an episode from the competition. Those are different operations.
- **But all of its parts have been tested, and they point against it:**
  1. Its signal is `cue_response` (`brain.py:154-170`). That signal tracks overlap with replayed wording (AUROC 0.61 on rewording, `2026-11-17-know-signal`). An episode's own text is exactly what was replayed, so **[inferred]** almost every slept episode will read as "answered" and disengage. That leaves something close to hippocampus-off for slept facts.
  2. Hippocampus-off H is already about equal to intact H, and the confidence-gate oracle showed only about +0.3 to +0.5 headroom. So D1's best case is roughly known: no H gain. Its main benefit would be fewer wrong reinstatements.
  3. It does not touch the permanent identity episode, which causes the "you're Dan" defect, unless permanent episodes are included. Including them would remove identity recall.
  4. It disengages by consolidation, not by breadth. A hub that is unconsolidated (such as a recent lived turn) stays in.
- Literature candidate 7's gist/detail refinement adds a second signal that does not exist yet.
- **Prerequisites.** Sleep consolidation is PARTIAL. The signal is known to depend on wording (PREREQUISITES §feeling-of-knowing: "both familiarity signals fail to generalise to new question wording"). Per the CLAUDE.md rule, that is a verdict on this signal, and the capability remains open.
- **Prediction (if screened).** Disengaging episodes with cue_response > t leaves top-1 share ≤ 0.30 *and* right fetches ≥ arm A on s315-317. My expectation **[inferred]** is that right fetches fall, because slept right episodes leave the competition.
- **Screen.** About 98 cue_response passes per home on CPU with `cortex_adapters.pt`, then arithmetic. Cheap enough to include as one arm. **It is not a Stage-2 slot unless it beats H1 offline.**

### Rank 5: H4, sleep-trained contrastive key projection (literature candidate 6; RIF/differentiation, candidate 5, as its local-rule form)

- Learn a rank-8-32 projection P on keys at sleep, using hard negatives from the day's confusable pairs. Biology: Hulbert & Norman 2015; Norman et al. 2007.
- It is the strongest fix in principle for layer 2. But it needs a competition log that does not exist, labels ("own cue") that are exact only when cue text equals stored text, and more than 2 days.
- **Deferred.** Revisit if H1+H2 leave geometric wrong fetches above about 30%.

## 4. Offline screen plan (minutes per rule after a one-time dump)

**Homes.** Copies, never opened in place. Read-only ssh; copy to scratch on disk, not /tmp.
- Night-14 `snap_trunk` and night-15 `trunk` of `workLIFE_S1D_s315` (pool40), `_s316` (pool41) and `_s317` (pool42).
- The night-7 trunks of s310-s314 that the R-raw screen used.
- Skip `workLIFE_S1F_s319` and anything adjacent to S1REG.

**Step 0: one dump per home** (new script; about 10 CPU-min per home at 1.5B, same cost as `life_screen rank`):
- Load the copied home as in `life.probe`.
- Ledger: `life.build_life(seed, days, 'dev')`; battery: `life.battery(seed, night, ledger)`. That is 6 H, 2 near-duplicates, 3 never-told and 7 open/greet questions, plus rotating ones.
- For each battery cue: `sims_S0 = hc.similarities(cortex, base, cue)` (lines 154-164), using the same base `Brain.hear` passes at `brain.py:491`. Also `KeyedHippocampus.similarities` for S_all, and the raw uncentred cue key `cortex.key_of(...)`, so whitening can be re-fit offline.
- Per episode: S0 `key` from `episodes/*.pt`; S_all blocks via `hippocampus_keys._delta_layers`; and `strength`, `uses`, `last_used`, `created`, `permanent` and `text` from `episodes.json`.
- For D1 only: `cue_response(ep.text, base)` per episode on the adapters-on cortex (`brain.py:154`).
- Save everything as one `.pt` file per home.

**Step 1: the rule sweep** (pure tensor arithmetic, seconds per rule):
- Each rule is `rule(sims, K, strengths, ages, uses, permanent, question) -> (winner|None)`.
- Competition and firing follow `_compete` (lines 174-196) with `FREEZE_USE`: no feedback, `sims=` passed.
- Arms:
  - A: S0 weighted (baseline)
  - D: S0 raw
  - H1-CSLS for k ∈ {3, 5, 10}
  - H1-MP (z-score)
  - H2 with d ∈ {0, 1, 2, 4, 8} and λ-ZCA, crossed with A, D and H1
  - H3-first-order: rescaled strengths into A
  - D1 disengagement for t on a grid, into A and into H1
  - H1+D1
- Bars: sweep bar and margin per rule on night-7 s310-s312, freeze them, then verify on held-out night-7 s313-s314 and on night-14/15 s315-s317. Do not carry 0.12/0.02 or 0.165/0.02 to a new score scale; R-raw's lesson is that bar and key were confounded.

**Metrics per home** (by ledger identity, using fetch fields as in `brain.eval.life.probe`; no generation needed):
- **Primary:**
  - right-episode top-1 rate on H/direct questions (forced, ignoring the bar);
  - right fetches and wrong fetches (with the bar);
  - **hub top-1 share** (largest single episode's share of all fetches).
- **Guards:**
  - (a) right fetches ≥ arm A, and ≥ 0.9 × arm D;
  - (b) open/greet/never fires ≤ arm A;
  - (c) identity-question right rate ≥ arm A;
  - (d) permanent-episode win rate on non-identity cues reported (target: down);
  - (e) near-duplicate pairs: both members still reachable.
- **Diagnostics:** N_k skew; share of fetches with margin < 0.02; strength-vs-geometry split (§2, item 1).

**Pass to a life.** Every primary and guard holds on all 3 night-14 homes and on the held-out night-7 homes. Right-fetch counts are small (4-8 per screen), so require the direction to hold on every home, not just pooled.

**Reply check, once, for the winner only.** A full `life.probe` with generation on the copies of s315-s317 under the winning rule and the oracle-fetch arm (§2, item 3), to get H-right and open intrusions. About 30-60 CPU-min per home.

**Offline limits.** A frozen probe screens selection at the stored state, not hub formation. The stored strengths come from the old rule, so H3 and online H1 are only first-order screened. The simulator for replaying the heard-turn sequence is not written and is not needed for this round.

## 5. Recommendation

The timeline below is due by 12-08 and feeds the 12-14 multi-week-life milestone.

**Before Stage 2 (11-22 to 11-24, CPU on the pools).** Run the §4 dump and sweep, plus the oracle-fetch probe. Budget: 1 day.

**Stage 2, slot 1: H1, excitability-normalised ranking (CSLS or MP, whichever wins offline, with the best H2 whitening setting as its preprocessing). At most 2 days.**
- **Offline kill (before any life):** kill H1 if any of these fails on any s315-s317 night-14 home:
  - top-1 share ≤ 0.30;
  - right fetches ≥ arm D;
  - open/greet/never fires ≤ arm A;
  - identity right ≥ arm A.
- **Life kill (15 nights, DEV seeds s315-s317 first, then six seeds before any claim):** kill H1 if, against the A arm, any of these holds:
  - wrong fetches at nights 14-15 are not ≤ 50% of baseline;
  - right fetches drop;
  - H-right drops by ≥ 1 per seed (mean);
  - a whole-transcript defect appears (identity loss, name drift; the M-scale kill criteria);
  - "do you remember X?" still gets the identity reply on > 2 of 5 interview items.

**Stage 2, slot 2: H3, sleep strength homeostasis + decay-clock reset only on raw-confirmed fetches + a per-episode replay cap. At most 2 days, as a life arm, with H1 if H1 passed offline.**
- It is the only candidate that targets hub *formation* and replay dilution, which the frozen screen cannot see.
- **Kill H3 if any of these holds:**
  - max non-permanent strength at night 14 is > 2.5;
  - wrong fetches at nights 14-15 are not below the same-rule arm without H3;
  - slept-fact recall with hippocampus off (direct-off) drops by > 10% vs baseline. Replay dilution must not be traded for weaker consolidation.

**D1** gets one offline arm only (about 1 CPU-hour) and no Stage-2 slot. It is promoted only if H1+D1 beats H1 on right fetches with hub share ≤ 0.30. Expected **[inferred]**: it fails, because its signal reads replayed wording.

**H4** (learned contrastive key) is deferred. It opens only if, after H1+H2, more than about 30% of wrong fetches still have a geometric cause (they stay wrong under raw ranking).

**Expectation.** Layer 3 (slept facts leaking into replies) is outside every candidate here. If the oracle-fetch probe shows less than 1 H-right of headroom per seed, report the hub fix as solving wrong reinstatement and the "you're Dan" interview defect, not reply accuracy. Route the reply defect to the sleep/cortex track separately.

## B. Critique (opus, read-only, verified against code and data)

**Corrections to the hub verdict (2026-11-21)**

**(a) Levers that were already killed, under new names**
1. H1-online, the θ win-rate homeostat, is the "fan normalisation (use-share homeostat)" refuted in `research/2026-11-20-hub-rank-raw-prereg.md` §0. That refutation found its fixed point leaves a hub share of 0.4-0.6. The refutation's premise was weighted ranking: 3 of 4 right fetches were weighted-only winners. H1 ranks on raw similarity instead, so the refutation may not carry over. The verdict has to name H1-online as fan normalisation and answer §0. It must not present it as new.
2. The H3 decay-clock rule ("reset last_used only on raw-winner fetches") is M-use with the margin set to 0. In `_compete` (`hippocampus.py:206-216`) one `confirmed` test gates `uses`, `strength` and `last_used` together. The verdict's line "a reset rule, not M-use's strengthening gate" is wrong. It also fails the same way M-use did: on s310-312, 3 of 4 right fetches are weighted-only winners (prereg §0), so the rule withholds the clock reset from most right fetches. Combined with H1 or D the rule does nothing, because every fetch is then the raw winner.
3. Static H1 is R-raw arm D plus a per-episode offset. Every measured number cited for H1 is arm D's. The claim that the offset fixes D's K3 failure (open-turn fires) is untested.

**(b) Host rules and label leaks**
4. H4 trained on ledger "own cue" labels would feed the teacher's ledger into a brain synapse, which breaks the brain-based rule. Its only legitimate form is self-supervised: an episode's own key against cues the cortex itself produces. Say so, or drop H4.
5. Every rule still picks its bar with `question = text.endswith("?")` (`brain.py:491`), a host rule on punctuation. Any new score scale needs its own bar for both question and statement turns. The verdict sweeps only one bar.

**(c) Is the offline screen valid?**
6. The S0 dump approach is sound, and the verdict misses why. The S0 key is the layer-0 attention state (`cortex.key_of`, layer 0), and the adapters only touch FFN key/value (`plastic_cortex.py`). So S0 similarities depend only on the cue text and are unaffected by sleep. S_all is not: stored deltas were read through that day's adapters and cues through tonight's (`hippocampus_keys.py:193-201`). Fitting H2 on S_all therefore mixes in key drift.
7. Do not write a second competition from scratch. Each rule should be a `Hippocampus` subclass that overrides `scores`/`_compete`, driven through `life.probe` → `run_battery` → `Brain.hear` with `FREEZE_USE`, as `life_screen.py rank` (`RANK_ARMS`) already does. If tensors are dumped for speed, arm A built from the dump must first reproduce the logged `row["fetch"]["idx"]` and `fire` of the `intact-frozen` arm exactly, on every home.
8. Write each fire rule out exactly. Under `RANK_ON_RAW` the bar is tested on the weighted score of the raw winner, while the margin is on raw similarity (`hippocampus.py:188`). This is the "strength gates firing" the verdict relies on. Also keep the inquiry path in `Brain.hear`, which is on by default.
9. The baselines are wrong. The headline defect is measured on the self-boosting intact arm, and the battery feeds its own hub: s315-318 night 1 gives top-1 10-11 of 13-14 fetches intact, but 5-7 of 11-12 frozen. Baselines must be frozen-A. In-battery feedback (USE_BOOST 0.5 per fetch) should be reported as a separate driver.
10. The power is far too low. Right fetches are 0-3 per home, so "direction holds on every home" is a coin flip. Because S0 sims are a function of cue text alone, the screen can score every told DEV ledger item's `direct` question and its DEV rewordings, plus all DEV opens. That is about 50+ cues per home instead of 20, at almost no cost. It must never import `life_bank_test` or touch S1REG.
11. I could not confirm that the night-14/15 homes `workLIFE_S1D_s315-317` exist on the pools. The local `rankres/life_s31[5-8]_B.json` files are 7-day lives. Confirm the homes before planning on them.

**(d) Are the claims supported by the record?**
12. The "geometry floor" claim is wrong. "Wrong fetches already 4 at night 1 on all four seeds" is the intact arm. The frozen arm gives 2, 2, 2 and 1 of 6 (`life_s31[5-8]_B.json`).
13. The "about 69% of raw margins below 0.02" and "100% of wrong fetches are weighted-only" figures come from s310 alone, with 16 fetches and 4 wrong (`2026-11-19-life-screens-1.5b.md`). They are not record-wide.
14. H3's mechanism doesn't hold up numerically. tanh is already saturated above a strength of about 2 (tanh 3.5 ≈ 0.998 vs tanh 2 ≈ 0.96), so capping the hub's raw strength barely changes its weight. The weight gap comes from decay of unused facts: a told-once fact at strength 1 is down to about 0.34 after 7 days. Its prediction (max strength ≤ 2.0) can pass while ranking stays unchanged.
15. Open-turn fires were already high under arm A: 32 of 50 open/greet/never turns pooled, against 41 for D. K3 measures a problem A already has, not one D created.
16. The ranking of H1 > H2 > H3 rests on plausibility. The only measured lead is arm D.

**(e) What is missing**
17. **The "you're Dan" interview reply is cortex-side by night 10.** In the s318 interview prompts (`_scratch/stage1/ivread318`), KNOW items get "you're Dan" on:

    | Arm | Night 5 | Night 10 | Night 15 |
    |---|---|---|---|
    | intact | 5/6 | 10/10 | 7/7 |
    | hippocampus-off | 0/6 | 10/10 | 7/7 |
    | both-off | 0 | 0 | 0 |

    From night 10 the slept cortex produces it with no fetch at all. One cause is that sleep replays each permanent self episode at weight ≥ 3.0 (`effective_strength`, `brain.py:782`; `rng.choices(weights=...)` in `sleep.py`). Another is that the KNOW template ("do you remember what I told you about…") overlaps word for word with the self episode "Do you remember my name?". No hippocampal selection lever can fix this reply. H3 as written rescales only non-permanent strengths, so it misses the largest replay share.
18. The hub has two different causes, and the verdict merges them. In the battery the hub is a told fact: Pixel on night 1; "The bus was late again" or Bramble on night 7. In the interview the identity reply comes from the permanent self exchange plus the slept cortex.
19. H falls with the hippocampus off too (5-6/6 down to 1-3/6 by nights 13-15, s318), while the adapter norm grows from 59 to 672. The H decline over the life is not caused by the hub. Under the owner's 11-15 directive (fix the biggest conversational defect, don't optimise our own scripted tests), lowering wrong fetches without moving replies is metric work. The oracle-fetch probe is right, but item 17 already answers most of it.
20. The static H1 bank is the wrong shape. A reference bank of stored keys (mostly "My X is called Y" tellings) measures crowding among stored episodes, not how widely an episode matches incoming cues. CSLS uses the query-side neighbourhood (here, incoming cues). So the static form will not penalise the three question-shaped self episodes, and may not penalise "bus late" either. A cue-side bank has to come from heard history, which is the online homeostat of item 1.
21. Gaming guard: hub share falls whenever a rule simply stops firing (arm R: 4-5 fetches per home). Require a minimum fetch count per home, or report share alongside right fetches as a pair.

**Corrected recommendation**

The record supports one measured selection lead: raw S0 ranking (arm D) dissolves the battery hub, with top-1 ≤ 0.22 on 4/4 homes and right fetches 8 → 14, but it keeps A's already high open-turn firing. The next hippocampal step is therefore a D-plus-fire-gate screen, not three new levers. Build it as `Hippocampus` subclasses in `life_screen.py rank`, through `Brain.hear` with `FREEZE_USE`. Check it against the `intact-frozen` arm's logged fetches. Widen the cues to every told DEV ledger direct question, its rewordings and all DEV opens. Run frozen-A, D, D with a per-episode normalised fire test (static CSLS/MP, labelled as D plus an offset), and, if pursued, an online homeostat explicitly labelled fan normalisation that answers prereg §0. Refit the question and statement bars on s310-312 and hold them out on s313-314, after confirming which night-14 homes exist. Drop the decay-clock rule (it is M-use at margin 0 and does nothing under raw ranking), drop D1, and defer H4 unless it is self-supervised. In the same budget, put the main effort on the cortex side, where the record shows the conversational defect now sits: from night 10 the "you're Dan" reply is produced with the hippocampus off, and H falls with it off too. Test a cap on replay share that includes the permanent self episodes, with the self-identity interview check and the hippocampus-off arm as the reply read. Report any hub fix as a fix to wrong reinstatement only.