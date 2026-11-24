# Capability: Episodic recall / Memory selection (D1, which episode a question fetches) + Dialogue (D2, in-conversation loop onset)
# Record: brain/research/2026-11-21-hub-research-round.md; 2026-11-23-biggest-defect-read.md; results/2026-11-23-m3-margin-screen.md; results/2026-11-24-flip-night-diagnosis.md; results/2026-11-24-recall-voice-dev-read-1.5b.md; results/2026-11-14-lived-key-1.5b.md; results/2026-11-16-reply-monitor-1.5b.md; results/2026-11-16-own-gain-probe.md; results/2026-11-16-onset-baseline.md; 2026-11-24-self-other-tag-prereg-DRAFT.md (BLOCKED); commits 38e2eab5d, 248f95667
# Expected-fail: D1 yes, partly. The key is a lexical layer-0 stand-in (SCAFFOLDS.md:8), and "separation of near-duplicate episodes" is listed MISSING in PREREQUISITES.md, so ranking fixes can only remove cue-shape and strength bias, not separate true near-duplicates. D2: a frozen 1.5B cortex loops in this dialogue format alone (19% of conversations, onset-baseline), so a state-side lever can cut the cascade but not the cortex's own onset rate.

Research note, 2026-11-24. No model was run. Three host-only counts were made for this note (scripts in
`_scratch/route/`). Everything else comes from the files cited.

## 1. Record brief: levers tried and verdicts

**D1 routing.** More than 15 levers. Full table in the hub round (`2026-11-21-hub-research-round.md:24-41`).
- **Fixed S0 key and bars.** Bar 0.12 for questions and 0.45 for statements, margin 0.02, score = centred cosine × tanh(strength) (`hippocampus.py:34-36,166-172,174-215`). This is the default.
- **S_all key, five variants.** No go. It picks right on reworded questions (14/18 vs 1/18) but the replies do not change, and the bar does not transfer (`2026-11-18-sall-slept-gate-1.5b.md`, `2026-11-14-key-default-1.5b.md`).
- **Self-episode question key.** Not a lead: question-shaped keys cluster together.
- **M-use.** Killed for over-gating.
- **R-raw arm D.** It dissolved the hub and right fetches went 8 to 14. It was killed because it fires on open, greet and never-told turns.
- **R arm.** Stops firing.
- **Gates.** None of these worked:
  - the confidence gate (narrowed);
  - the "do I know this" signals (they track wording);
  - M1 recall-to-reject (falsified);
  - the M4 near-miss guard;
  - M-scale and SAB (killed).
- **Lived key and speaker contrast.** No go: right episode only 36/72 unslept.
- **M3 margin-gated gain.** Does not separate: AUROC 0.42-0.44, and the direction flips between seeds.
- **Recall-voice M1 and M2.** Killed (M2: told 105/223 against K-P's 123). The refuter narrowed the cause to routing.
- **Never built:**
  - H1 (CSLS / mutual proximity);
  - H2 (whitening);
  - H3 (sleep strength homeostasis);
  - H4 (contrastive key);
  - the D plus fire-gate screen;
  - a self/other term in `_compete`;
  - inhibition of return.

**D2 cascade.** All four switches are default OFF (`brain.py:221-222`, docs at 323-340), and none is wired into `life.py`. Only `open_conv.py:132-147` passes them.
- **`habituation`** (no-repeat n-gram). No flip; it leaves an "I don't know." loop.
- **`reply_monitor`.** No-worse only. All 24 withheld drafts ended as the default reply, because the redraws from the same state were copies too. On the echo seed the echo became a stored-line loop.
- **`own_gain`.** At full gain a forced onset repeats 14/14. At 0.5 the echo still repeats 14/14. At 0.25 or lower the brain speaks user-like questions (it loses whose turn it is).
- **`own_gain_copies` at g=0.** NO LEAD: 2/6 samples fall into question loops.
- **`carry_penalty` and fake-think.** NO LEAD.
- **Cascade evidence.** P(first person | previous first person) is 0.83 vs 0.17. On pool40 the s320 intact home is a CASCADE (A 4, fresh-state B 0, told-first C 3) and s319 is WEIGHTS (`flip-night-diagnosis.md:31-40`, 38e2eab5d).
- **Battery resets.** `run_battery` resets the state only for `open-*` and `ab` questions (`life.py:265-267`). Told and never rows share one running state.

### New host-only counts (this note)

These come from banked `_scratch/stage2/read/lifeKP_s319-321.json`, trunk, nights 5-15, told rows, with the same join as `m3b.py`.

**(a) The 28 wrong-episode hurt rows** (`_scratch/route/wrong28.py`).
- They name only **6 distinct episodes**. "My dog is called Bramble." accounts for 13 and "The bus was late again this morning." for 8.
- The top-1 share per seed is 0.50, 0.83 and 0.67.
- **26/28** name an episode already fetched earlier in the same night's battery, and **13/28** name the same episode as the immediately preceding fetch.
- All 28 are question cues.

**(b) Intact vs intact-frozen** (`_scratch/route/frozen_vs_intact.py`; the frozen arm has no use boost during the battery).
- Wrong fetches: 50/46/45 intact vs 50/47/44 frozen.
- Right episode: 16/14/14 vs 16/15/14.
- So the boost *during the battery* adds nothing. What causes the repeat is already in place when the battery starts.
- Wrong fetches sit on **141 of 223** told rows and the right episode on 44. The defect is wider than the 28 rows where fetching hurt.

**(c) Key-space hubness** of the stored S0 keys (`_scratch/route/nk.py`). This uses the banked homes `_scratch/kpgate_w/home_KP_s319` and `home_KP_s320` (98 episodes, 16 sleeps). These are replay homes for the gate run, not the exact lifeKP lives.
- N_5 skew is 0.21 and 0.41; N_1 skew is 0.91 and 1.50.
- Bramble's N_5 is 5 and 4; "bus was late" is 6 and 6. The mean is 5, so both are ordinary.
- Strength hubs exist: Philippa has strength 9.0 with 16 uses in s319, and Bramble 8.5 with 15 uses in s320. The median strength is 1.0 and the median use count is 0.
- **Reading:** the wrong episodes are not geometric hubs *among stored keys*. Whatever pulls a question toward them comes from how question-form cue keys sit relative to statement-form stored keys, plus the strength the episode built up during the life. H1 as the hub round specified it (r_i from key-to-key similarity, `hub-research-round.md:86`) is therefore expected to be near-inert.

## 2. External (from the external brief; summaries only, papers not opened)

**For D1:**
- Hubness corrections, CSLS (Conneau 2018) and mutual proximity.
- Retrieval-induced competition (Norman, Newman and Detre 2007, secondhand).
- The CA1 match/mismatch comparator (Hasselmo; Lisman and Otmakhova).
- Temporal context, TCM/CMR (Howard and Kahana 2002).
- kNN-LM: the best keys are deep layers, not shallow ones.
- DG expansion with k-WTA. The old track's DG index was built for speed, not separation (`research/biology/dg-ca3-sparse-index.md`).

**For D2:**
- Self-reinforcement of repetition, which rises with the number of prior repeats in context (Xu et al. 2022).
- Contrastive decoding against an "amateur" (Li et al. 2022).
- State re-priming in RWKV (state tuning).

**Biology for D2:**
- Sensory attenuation of self-produced input: efference copy and corollary discharge damp the trace of one's own speech.

**Contraindicated by count (a):**
- Adding conversation context to the cue (TCM) would pull the fetch toward the previous fetch. That is already the wrong episode in 13/28 rows.

## 3. Ranked methods

### D1 routing

**1. Cue-referenced excitability normalisation (H1 re-aimed).**
- **Mechanism:**
  - Each episode i keeps a running excitability r_i: the mean of its centred similarity over the last N heard *questions*. N is about 50, the variable is an EMA, and it is updated on every cue whether or not a fetch fires.
  - Competition then uses (sim_i − r_i) × tanh(s_i).
  - An episode that is close to every question, because of question shape, is discounted. The cue that is specifically about it still wins.
- **Change:** `Hippocampus.similarities` / `scores` / `_compete` (`hippocampus.py:154-215`), plus one float per episode in `episodes.json`.
- **Status:** brain-native. This is intrinsic-excitability homeostasis, a local per-unit rule, with no host bank of cues.
- **Prerequisite supplied:** "separation of confusable facts" (Memory selection, PARTIAL), on the cue side.
- **Avoids:**
  - H1 key-to-key (count c: low skew);
  - R-raw D (D removed strength and changed the bar; this leaves both as they are);
  - S_all (a key change);
  - M3 (a gate on gain, not a change of choice).

**2. Inhibition of return (winner fatigue within a conversation).**
- **Mechanism:**
  - An episode reinstated on turn t gets a transient penalty f·exp(−Δturns/τ) in `_compete`.
  - The penalty resets on sleep and on a fresh conversation.
  - This is short-term depression of the just-fired CA3 ensemble, the same idea as the old track's D5 winner fatigue (e62113ef8, 6/6 separation GO).
- **Change:** `_compete` plus a per-episode `last_reinstated_turn` held in the Brain's conversation state (`brain.py:552-561`).
- **Status:** brain-native, a synaptic adaptation.
- **Target:** count (a), the 26/28 and 13/28 repeats.
- **Avoids:** M-use and FREEZE. Those change the strength update; count (b) shows that update is not the driver.
- **Risk:** the same item is asked twice in one battery.

**3. Sleep strength homeostasis (H3).**
- **Mechanism:** each night, rescale non-permanent strengths toward a cap or the median, as in the synaptic homeostasis hypothesis.
- **Change:** `sleep.py`, or a night hook in `brain.py`.
- **Status:** brain-native.
- **Target:** hub formation. Strength reaches 8.5-9 against a median of 1, and count (b) puts the cause before the battery.
- **Rank:** third, because its first-order offline form is close to arm D, and the part that is distinct (hub formation) needs a 15-night life.

### D2 cascade

**1. Decaying own-write (attenuated self-trace).**
- **Mechanism:**
  - `commit_exchange` (`brain.py:636-660`) records the delta that writing the brain's own reply makes, using `cortex.matrix_delta`.
  - The next turn reads at full gain, which keeps whose turn it is.
  - On each later turn, β^k of that delta is subtracted from the state, so an early first-person or refusal reply fades instead of persisting.
- **Avoids:** flat `own_gain`. That lost the turn role at ≤0.25 because it also weakened the reply just spoken. Also avoids `own_gain_copies`, whose detector only sees word overlap.
- **Status:** brain-native (efference-copy attenuation).
- **Prerequisite supplied:** "self/other tag on the state write", listed missing under Self and identity.

**2. Readout against the no-own-history track (contrastive).**
- **Mechanism:**
  - Keep a second state with no own replies written, and read logits = (1−α)·l_full + α·l_noOwn.
  - Write the reply on the full track only.
- **Change:** `Brain.hear` / `commit_exchange`. It costs 2× the reads.
- **Status:** a declared scaffold (a host-held second track), to add to SCAFFOLDS.md if it is built.
- **Avoids:** `reply_monitor`. The monitor redraws from the same attractor; this method mixes in a state that lacks the attractor.

**3. Monitor redraw from the attenuated state.**
- **Mechanism:**
  - Keep `reply_monitor` as the trigger.
  - When it withholds a draft, redraw from the state with own-writes removed (method 1 with β = 0), not from the same state.
- **Status:** a scaffold, because `_is_copy` is a host word-overlap test (`brain.py:116`).
- **Rank:** third. Its reach is bounded by the copy detector, which misses refusal and identity loops with varied wording (`flip-night-diagnosis.md`).

## 4. Recommended first screen per defect, with KILL bars fixed now

### D1: offline S0 fetch replay on banked homes, methods 1 and 2 together

**Setup.**
- Pool40, CPU fp32, via `tools/memcap.sh`.
- For each told and never battery cue of nights 5-15 in lifeKP s319-321, compute `hc.similarities(cortex, base, cue)` once on the matching banked home. The same read as the hub-round dump (`hub-research-round.md:163`) is acceptable.
- Score arms A (default), IOR, CUE-NORM and CUE-NORM+IOR from these similarities with home strengths, replaying the battery order. f, τ and N come from a grid fit on s319 only. Then s320 and s321 are read with that setting frozen.
- Caveat: the strengths are the end state of the home, not the battery-time state.

**KILL (on s320 and s321, the held-out seeds), if any of these holds:**
1. Right-episode fetches on told rows rise by fewer than 10 pooled, or not on both seeds.
2. Of the 28 wrong-hurt rows, fewer than 8 become right-episode.
3. New wrong fetches appear on rows that were right-episode in A on 2 or more rows per seed.
4. Fires on never-told and open cues rise above A by 1 or more per seed. This is the reason arm D was killed.

**Pass is a lead only.** Then run a 15-night K-P life on s319-321 (AWS g5, not 12-04..06), and six seeds before any claim.

**Zero-cost pre-check already done:** count (c) predicts that key-to-key H1 is inert, so key-to-key H1 is not screened.

### D2: pool40 cascade harness, s320 intact night-6 home (CASCADE), with s319 (WEIGHTS) as the control

**Setup.**
- Battery in banked order, through the 38e2eab5d harness.
- Arms: A (as banked), DECAY with β ∈ {0.5, 0.25} fitted on s320 only, and CONTRAST with α = 0.5.
- Plus the forced-onset echo (E) and stored-line (B) probes from `2026-11-16-own-gain-probe.md` in `open_conv.py`, one script seed.

**KILL if any of these holds:**
1. s320 banked-order first-person flips are 2/8 or more (fresh state gave 0; A gave 4).
2. Told right on those battery rows falls 2 or more below A.
3. On E, an echo shows on 10 or more of 14 free turns in any sample.
4. User-style question loops appear in 2 or more of 6 samples (the `own_gain_copies` failure).
5. On s319 (WEIGHTS), flips are not within ±1 of A. A move there means the method is not acting on the cascade.

**Pass is a lead only.** Then wire the switch into `life.py` `brain_kw` and run per-night-clustered reads (`flip-night-diagnosis.md:36`).

**Measurement fix owed regardless of method:** report D2 per night, not pooled by row.

## 5. Amendment after two refuters (11-24, before any screen data)
Refuters: counts OK_WITH_FIXES (all load-bearing counts reproduce), feasibility BLOCK. Changes:
- **Re-rank D1.** Strength outranks cue-excitability: in the frozen arm 110 of 141 wrong fetches are weighted-only winners
  (the raw-similarity winner differs, median margin_raw -0.08 vs margin_w +0.087; 20/45 on right fetches), and the hub
  per seed follows strength (Philippa 9.0 in s319, Bramble 8.5 in s320) with geometry held fixed. First: rank by raw
  similarity, gate firing by the weighted score (the hub round's never-run "D plus fire gate"), and a cap or rescale on
  recency-weighted strength (H3). Second: cue-norm H1. Caveats: tanh saturates at strength >= 3; "bus" (strength 1.0, 0 uses
  in the replay homes) is unexplained.
- **Drop** the 26/28 "fetched earlier that night" figure (order-invariant: shuffled null 386 = observed 386) and the
  "previous fetch" framing: the cue is read from `base`, so the battery conversation cannot reach routing except via
  USE_BOOST, which the frozen arm removes.
- Host-only raw-winner read (`_scratch/route/rawwin.py`, frozen arm, nights 5-15): of 119 wrong fetches whose raw winner is
  identifiable, the raw winner is the right episode on 35; identity unknown for 44 wrong and 34 right fetches. A lead,
  not decisive.
- **Both screens as written cannot run**: no per-night homes exist (final trunk = night 15; kpgate/kpid homes are 16 sleeps),
  and night 15 alone has 8 told rows per seed (bars unreachable). Fix: rerun the three K-P lives (`lifeKR_s319-321`,
  `brain/research/recall_voice_launch/jobs_KR_s*.txt`) with `life_kw --night-snapshots` (record-only: episodes.json with
  strength/uses/last_used, all keys, adapters, per trunk morning; stub test shows the life byte-identical). Validity gate:
  each KR life must reproduce `lifeKP_s<seed>.json` row for row (AWS fp16 is byte-deterministic); a mismatch voids it.
  Episode deltas for any night come from the final trunk's episode files (append-only).
- Before any screen data, a screen prereg will fix: bars as fractions of nights 5-15 told rows with denominators; grid,
  objective and tie-break a priori (s320/s321 declared seen for design); r_i seeding rule for H1; fire bar kept raw and
  declared; D2 beta fixed at 0.5 and 0.25 (both reported, no selection), s319 bar "not worse than A and not below B",
  told bar no-worse on s320 with the s319 floor stated; per-night results with n. Fetch-level passes are labelled
  fetch-level only (S_all moved fetches 1/18 -> 14/18 without moving replies). New constants go in SCAFFOLDS.md.
