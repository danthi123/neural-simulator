---
type: design
status: live
date: 2026-09-25
lane: load-bearing
mechanism: prioritized memory (keep what matters, let minor details fade) as the companion of the DA tag-capture + sleep-replay pair -- research and design only, no sim/ or webapp/ change
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_sleep_replay_capture/aggregate.json
  - research/findings/raw/_sleep_replay_capture_r2/aggregate.json
  - research/findings/raw/_awake_replay_capture/aggregate.json
  - research/findings/raw/_sleep_forgetting_interference_smoke/seed42.json
biology:
  - research/biology/importance-tagging-at-encoding.md
  - research/biology/prp-competition-and-locality.md
  - research/biology/prioritized-replay-triage.md
  - research/biology/gist-detail-graded-forgetting.md
  - research/biology/repetition-retrieval-strengthen-same-trace.md
---

# Prioritized memory: remember what matters, let minor details fade (DESIGN)

Research and design, no code. It answers the owner's ruling of 2026-09-25 on the overnight forgetting of the DA
tag-and-capture + sleep-replay pair, maps what the brain already has against what real brains run to prioritize
memory, orders the mechanisms to build, and registers the test battery ("what-matters gates") the mechanisms will be
held to. Terms follow `docs/TERMS.md`: "consolidation" is used only for the biology until a source lesion earns it in
the model, and a verdict word only as a gate's own verdict.

## 0. The owner's ruling (verbatim) and what it changes

> "Wait on fix. To be clear, a certain degree of forgetting is acceptable, as is the case with real brains. What's
> important is that the brain should remember things that matter. Real brains have ways of prioritizing what should
> be remembered and focusing on key details, while forgetting minor details over time that aren't as important. Of
> course, the more that's remembered the better, but it would be dumb to remember every little thing perfectly, or at
> least to spend the maximum effort on remembering all things equally. You can extrapolate from this. But overall yes,
> it's very important to me that the brain actually learns, grows, and remembers. Not just stores info like a RAG
> system."

Three consequences. (1) The pair stays default OFF until the fix lands. (2) The flip question is no longer "is an
ordinary fact kept overnight?" but "does what is kept follow what matters, with minor details fading gradually?". (3)
Storage that is append-only, equal-strength and never forgets is itself a failure, not a safe default.

## 1. The target behaviour, in plain words

- **Important things are kept.** A fact the user flags ("remember this"), one told with feeling, one that surprised
  the brain, one about the thing the conversation is about, one that fits what the brain already knows, one it was
  told twice or has already been asked about: these are still there after a week, key details included.
- **Minor details fade, gradually.** An ordinary aside is still there the same day and usually the next morning, then
  loses its details over days as other things are learned, and is eventually gone. Nothing vanishes at once, nothing
  is kept forever by default, and a lost fact does not come back on its own.
- **The gist survives when the details go.** The brain can say "you told me something about the dog, but I don't
  remember what" -- an honest read of familiarity without recall -- and never fills the gap with an invented detail.
- **More effort on what matters.** The brain's own offline replay and its limited stabilizing resources are spent
  mostly on important and at-risk memories, not spread equally.
- **Not RAG.** A repeat strengthens the memory it already has instead of filing a second copy, recalling something
  strengthens it, related memories support each other, and what is kept moves over nights into slower, integrated
  knowledge. Knowledge is learned into synapses and changes with use.

What "RAG-like" means here, measured: today's production default writes every told fact at full strength and keeps it
forever (no ledger), and a re-told fact is appended as a new block (research/findings/2026-09-25-sleep-forgetting-interference-fi-seed42-smoke.md:
on the re-mention arm the recall matched the day-3 re-mention block while the original fell like an unmentioned
fact's). The pair adds forgetting, but through one threshold that does not look at importance (section 2).

## 2. Why the pair's keep-or-lose is a single threshold

The adversarial review of the pair (research/findings/2026-09-25-da-capture-sleep-replay-pair-verify-go-review.md,
branch `research/pair-verify-go`, finding I-1) measured, across 24 seed x telling cells with both flags on, that the
next-day outcome is a step in one number: R, the store's cleanup margin at the night's one SWR epoch. Every cell at R
0.185 or below was lost, every cell at 0.209 or above was kept. <!--derived-->
Reading the code gives four reasons, each a constant standing where the real system runs a process:

1. **One switch per fact.** The composer writes a block as `complex(g) * zc[k]` with `zc` a unit phasor
   (`research/runners/one_brain_composer.py` `_write_block`), and the ledger sets each synapse's tag to
   `h0 = |inc_k|` (`webapp/da_tag_capture.py` `SynapticTagCaptureLedger.on_store`). Every synapse of a block carries
   the same tag, so every late-phase variable `z_k` follows the same trajectory and the block is captured or lost as
   one unit. The ledger's own summary field `frac_synapses_z_gt_half` reads 0 or 1 by construction (up to rounding),
   and every row of the review's table reads `z` at recall as 0 or 1. Real synapses differ in
   readiness (fewer than half of spines are primed at baseline, Kramar et al. 2012), so a trace is captured in part.
2. **One read, one epoch, no selection.** `SleepReplayCapture` runs one epoch per night and drives every managed
   block once, identically; R is the minimum margin over agent/action/patient (`reactivation_strength`). A strong
   trace reads high and is re-tagged, a weak one is not, whatever it is worth. Real nights have four or five NREM
   cycles (Buzsaki), and replay content is biased by reward, by awake ripples, by expected use and toward weak items
   (Singer & Frank 2009; Yang et al. 2024; Wilhelm et al. 2011; Schapiro et al. 2018).
3. **One global, bottomless PRP pool.** The ledger's plasticity-related-protein pool `p` is one cell-wide scalar that
   capture never consumes. Any tag inside its window is captured by any PRP event, which is why the pair's fake design
   day (prereg Amendment 7, branch `research/pair-production-path-arms`) captured every fact told within the hour
   after a salient telling or an SWR bout. Real PRPs are limited and competed for (Fonseca et al. 2004), local to a
   dendritic branch (Govindarajan et al. 2011), and behavioural tagging reaches related memories only (Dunsmoor et al.
   2015). Without competition, salience cannot win anything: under the pair the salient-vs-neutral next-morning
   contrast is gone (both kept 6/6, review section 4).
4. **One importance channel.** The only thing that marks a fact as worth keeping is the spiking SNc's dopamine,
   driven by the habituation novelty organ plus a host content-word count (`webapp/da_mode_drives_chat.py`,
   `webapp/reward_value_afferent_chat.py` docstring). Arousal, expected use, topic relevance, value and schema fit
   enter nowhere, and repetition and recall do not strengthen the trace.

A fifth constant makes the fact indivisible: R = min over roles, so there is no "core kept, detail lost" state. And
the read margin that decides everything varies across seeds with vocabulary crosstalk (the awake-completion binding's
0.21-0.49 spread of fresh reads), so the threshold sorts facts partly by how their words happen to be coded. <!--derived-->

**The companion processes, answered.** Asked "what does the real system run alongside this that we replaced with a
constant or a single threshold?", the answer is a set, in order of how directly each turns the step into a graded,
importance-ordered curve: (C-1) heterogeneous synapses plus limited, local, competed-for PRPs -- capture becomes a
graded fraction and resources become contested; (C-2) repeated, biased reactivation across the day and several NREM
cycles and nights, with the load-set renormalization (r3) as its brake; (C-3) several importance channels at encoding
that act locally and spread only to related memories; (C-4) component-wise traces (gist and detail separable) plus
repetition and recall that strengthen the same trace. Raising a threshold, retuning gamma or changing the replay-to-DA
map is not a response to this diagnosis.

## 3. What the brain has, has off, or lacks

HAS = on by default; HAS-OFF = built behind a default-off flag or runner-only; LACKS = no mechanism, or no edge from an
existing organ to memory. Biology entries are the five listed in the frontmatter.

| process (biology) | this brain | status | where |
|---|---|---|---|
| novelty / surprise dopamine marks a memory, behavioural tagging (Wang 2010; Moncada & Viola 2007) | spiking SNc driven by the habituation novelty organ + a host content-word count; DA write gain; DA -> spiking D1 pool -> PRP -> per-synapse capture | write gain HAS (`BRAIN_DA_ENCODING` default on); capture HAS-OFF (`BRAIN_DA_TAG_CAPTURE`) | `webapp/da_encoding_drives_chat.py`, `webapp/da_tag_capture.py`; chat-wire GO LTM-off and LTM-on 6/6 |
| prediction-error salience to the SNc | surprise organ's mismatch rate replaces the engagement mix on assertions | HAS-OFF (`BRAIN_REWARD_VALUE_AFFERENT`); its regex assertion gate is a declared shortcut | `webapp/reward_value_afferent_chat.py` |
| signed reward / value (Singer & Frank 2009; Oudiette 2013; Kandel ch.52) | none; the afferent above is unsigned by its own docstring | LACKS | -- |
| arousal: BLA + LC noradrenaline, LC dopamine co-release onto D1/D5 (McGaugh 2004; Cahill 1994; Takeuchi 2016) | LC-like arousal population (runner GO); LC-NE gain swap in the GNW (runner GO); affect organ with arousal rungs on the shared pool (3/6, NOT ALL-GO); interoceptive affect (runner GO) | organs HAS-OFF; the edge to D1 / PRP / write LACKS | `2026-08-13-affect-lc-arousal-population-GO.md`, `2026-09-04-gnw-lc-ne-adaptive-gain-swap-eviction-GO.md`, `2026-09-23-onebrain-affect-ladder-...-NOT-ALL-GO.md` |
| local priority: key detail up, background down (Mather 2016 GANE; Payne 2008) | none | LACKS | -- |
| expected future use, "remember this", directed forgetting (Wilhelm 2011; Stickgold & Walker 2013) | prospective-memory intention latch + Hebbian binding + NMDA facilitation holds an intention; no edge to a fact's tag or replay; no forget cue | latch HAS-OFF (`BRAIN_PMEM_FACILITATION`, load-bearing 6/6); edge LACKS | `2026-09-22-prospective-memory-facilitation-load-bearing-6seed.md` |
| topic / goal relevance (Kandel ch.52; Dunsmoor 2015) | common-ground ledger (NMDA attractor per referent, wired); WM referent focus binding (6/6, 2026-09-25) | organs HAS / HAS-OFF; edge to memory LACKS | `webapp/common_ground_drives_chat.py`, `2026-09-25-wm-referent-focus-bind-GO-6seed.md` |
| schema fit (Tse 2007; van Kesteren 2012) | CA3 superposed-fact attractor (runner, capacity 6/6, no chat write path); the LTM tier is a bulk, teacher-loaded closed-form store | LACKS on the chat path | `research/runners/ca3_superposed_fact_attractor.py`, `research/runners/tiered_fact_store.py` |
| repetition strengthens the same trace; spacing (Kramar 2012; Lee 2008; Cepeda 2006) | a re-telling appends a new block; reconsolidation rewrites only on a prediction error; "restabilize" writes nothing | LACKS (append is the RAG-like pattern) | `OneBrainComposer.update_on_mismatch`; fi seed-42 smoke `fir_lr` |
| retrieval strengthens (Karpicke & Roediger 2008; Sekeres 2016) | reads never write (kept for the read itself: systems-consolidation protocol rule) | LACKS | `research/biology/systems-consolidation.md` |
| graded capture over heterogeneous synapses (Kramar 2012; Govindarajan 2011) | one tag value per block, so one switch | LACKS | section 2 item 1 |
| limited, local, competed-for PRPs (Fonseca 2004; Govindarajan 2011) | one global scalar `p`, never consumed | LACKS | section 2 item 3 |
| replay selection: awake ripples tag sleep content, reward bias, weak items first, several cycles and nights (Yang 2024; Schapiro 2018; Buzsaki) | one epoch per night, every block driven once; awake bout OFF (arc family NO-GO, 5 of 6 seeds passed); pattern completion (branch, dev); a risk-prioritized teacher-loop replay (2026-08-09: beat random, failed coverage at a fixed budget) | HAS-OFF, uniform; selection LACKS | `webapp/sleep_replay_capture.py`, `webapp/awake_replay_capture.py`, `research/awake-replay-completion-r2` |
| brake: the night's renormalization set by the day's learning (Tononi & Cirelli 2014) | r2 constant (NO-GO 0/6); r3 load-dependent `BRAIN_SLEEP_LOAD_RENORM`, fi family 6 seeds running, seed-42 smoke holds every FI gate | HAS-OFF (in flight) | `research/biology/sleep-load-dependent-renormalization.md` |
| gist vs detail kept separately (Payne 2008; Sekeres 2016; Winocur & Moscovitch 2011) | one block per fact, R = min over roles; the episodic organ's topic familiarity is not ledger-managed and keeps every topic | LACKS | `reactivation_strength`; Amendment 7 `wd_epi` prediction |
| replay-written transfer to a slow cortical store, fast for schema-consistent facts (McClelland 1995; Tse 2007) | `promote_buffer_to_ltm()` is a host hook, never auto-invoked | LACKS | `research/runners/tiered_fact_store.py` |
| regulated forgetting (Hardt 2013; Berry 2012; Richards & Frankland 2017) | passive early-phase decay + r2/r3 downscaling | LACKS beyond r3 | -- |

## 4. The mechanism plan, in order

Each step is brain-based: neurons, synapses and neuromodulators decide; host code only for the world (the
conversation, the test questions), the body (the sleep/wake clock) and the clock. The ledger's per-synapse state
equations (tag, PRP, late phase) stay host-integrated synaptic state, the same category as every plasticity rule in
the engine, declared; moving them onto the substrate is the pair's own backlog (review section 5, items 1-3 and 8) and
not part of this plan. Every step is default OFF, byte-identical off, with its own lesion, its own biology binding and
its own prereg committed before any run.

**Step 0 -- the instrument first.** Build the what-matters battery (section 5) as a runner family, with the graded
outcome grader, a per-block read of the late-phase fraction and of each block's reactivation count, and the two
baselines: today's production default and the pair as it stands. No brain change. Its own gate is discriminating
power: both baselines must read NO-GO on the gates predicted in 5.6. If a baseline passes a gate predicted to fail,
that gate is repaired by amendment before any mechanism result is scored. "The instrument is part of the emulation."

**Step 1 -- graded capture over heterogeneous synapses** (C-1; `prp-competition-and-locality`). Give each managed
block's synapses a seeded per-synapse readiness drawn once at the write (a distribution fixed a priori from Kramar's
"fewer than half primed", never fitted to a gate seed), entering the late-phase drive as `gamma * p * h_k * rho_k`.
The `z_k` then cross their unstable point at different drives, the captured fraction becomes a smooth function of the
replay and PRP drive, and the expressed trace `b + (e + z(1 - e)) inc` degrades gradually, so recall margin falls
gradually. Lesion: `rho_k = 1` (today's single switch). Gates first expected: WM5d, WM2a, WM2c, WM4, WM10.

**Step 2 -- limited, local, competed-for PRPs** (C-1; Fonseca 2004; Govindarajan 2011; Dunsmoor 2015). Replace the
global bottomless `p` with (a) compartments: a block is allocated to the compartment whose units its concept codes
overlap most (allocation by excitability overlap, Yiu 2014), so facts about the same entity share one, and a PRP
event reaches other compartments attenuated; (b) consumption: capture draws on the compartment's PRP in proportion to
`gamma * p * h_k * rho_k`, so a strongly tagged trace spends what a weakly tagged neighbour would have used. PRP
synthesis still comes only from the spiking D1 pool. This makes behavioural tagging specific, makes salience
competitive, and is the registered NR response already named in Amendment 7. The compartment allocation rule is a
declared host step until the store has dendritic structure (`sim/dendritic_neuron.py` is the named rung). Lesion: one
global, unconsumed pool. Gates first expected: WM8, WM2c on the multi-fact day, WM6b (reported).

**Step 3 -- one trace per fact: repetition and recall strengthen it** (C-4; `repetition-retrieval-strengthen-same-trace`).
(a) A re-telling is recognized by the composer's own cued read (`_find_cued_block`, the spiking K-way sequencer); a
predicted re-statement, now "restabilize", re-induces early LTP and re-sets the tag on that block through the ledger
(the awake-replay rule `e <- e + R (1 - e)`, reused) instead of appending a copy. (b) A correct recall is a
reactivation of the answered block and gets the same re-induction, applied after the reply so the read never writes
during itself (the systems-consolidation protocol rule is kept). (c) Spacing: synapses left unprimed by an episode
become ready on an hour scale (Kramar's 1-h rule), so a repeat after an hour recruits them and a massed repeat does
not. The host `kb` list must not be the dedupe key. Lesion: the re-induction edge cut (and dedupe must still hold).
Gates first expected: WM5a, WM5b, WM5c, WM1 for Rsp and T, WM7 L-RECON.

**Step 4 -- replay that spends effort where it matters** (C-2; `prioritized-replay-triage`). (a) An importance mark
carried by excitability: at encoding the neuromodulatory mix at the telling (DA now; the Step-5 channels later) raises
the intrinsic excitability of the fact's trigger units, a slow CREB-like variable that decays within a day (the
instruction tag decays faster than the item, Stickgold & Walker 2013). (b) Each SWR burst is a population event in
which the managed triggers compete through lateral inhibition (the WTA the completion lane names as its next rung);
initiation is biased by excitability and by need (a weak trace has more headroom, Schapiro 2018), stochastic under
the arm's seed; the reactivation read is the completed ensemble (the awake-completion branch's `R_c`), not the min
margin. (c) Several bursts per epoch, one epoch per NREM cycle (four or five a night), and awake bursts in rest pauses
whose winners bias the night (Yang 2024). (d) The brake: r3 renormalization, Step 2's consumed PRPs and Step 1's
graded capture; the five-cycle fake-substrate runaway that forced one epoch per night is re-run first and must not
recur. Lesion: uniform selection (every trigger equally likely). Gates first expected: WM6, WM1 and WM3 for S, WM7
L-DA, WM2c under several epochs.

**Step 5 -- the other importance channels** (C-3; `importance-tagging-at-encoding`). Each is a spiking afferent onto
an existing population with its own lesion; none is a host importance score or a keyword test.
- 5a arousal: the affect organ's arousal (the LC-like population) projects onto the D1 pool as dopamine co-release
  (Takeuchi 2016) and as a local gain on the most active trace's tag with suppression of the rest (GANE). Lesion
  L-NE. Cue E.
- 5b expected use: "remember this" is understood by the language route (not a regex; the reward-value afferent's regex
  gate is the cautionary case) and latches the prospective-memory intention assembly, bound by its Hebbian edge to the
  fact's trigger; the latch raises that trace's excitability mark and co-activates with it in replay (Wilhelm 2011).
  "Never mind, forget that" is the complementary edge. Lesion L-REL. Cue F.
- 5c topic and goal: overlap between the fact's concepts and the active common-ground / WM-focus referents at the
  telling raises the mark. Lesion L-TOPIC. Cue G.
- 5d value: a signed outcome (praise, correction) needs a signed value afferent that does not exist yet; named, not
  built here.
Gates first expected: WM1, WM3, WM7 for E, F and G.

**Step 6 -- gist and detail on separable traces** (C-4; `gist-detail-graded-forgetting`). The core predicate and a
peripheral detail are stored on separate synapse sets (separate managed blocks linked by the shared agent code), each
with its own tag, capture and replay read (no min over roles). The Step-5a local gain decides which component a
salient moment favours. The gist that survives a lost detail is carried by the episodic organ's topic familiarity and
the common-ground referent, which get a decay of their own (today the episodic organ keeps every topic, equal and
permanent). The reply for "familiar but not recalled" is a functional read-out (section 8). Gates first expected: WM9,
WM3 peripheral for F.

**Step 7 -- replay-written transfer to a slow cortical store** (`gist-detail-graded-forgetting`; CLS). Replay
interleaves captured, important facts into a slow cortical store (the CA3 superposed-fact attractor or a slow cortical
Hebbian store), fewer replays needed when the fact's concepts already have many stored associates (schema, Tse 2007);
`promote_buffer_to_ltm()` stops being a host hook. The word "consolidation" becomes available only when a source
lesion (the composer block removed) shows the cortical trace answers. Lesion L-SCHEMA. Gates: WM1 and WM7 for K, and
a source-lesion gate in its own prereg. This is the "grows" part of the owner's ruling.

**Step 8 -- regulated forgetting of what is marked unneeded** (Hardt 2013; Berry 2012). A dopamine-dependent
forgetting drive during sleep aimed by a "forget" mark (Step 5b's complementary edge) and by staleness (a superseded
fact; reconsolidation already rewrites on a prediction error). Last, and optional until Steps 1-7 hold.

**Why this order.** The instrument comes first. Steps 1-2 come before Step 5 because feeding more importance channels
into a one-switch store with a bottomless shared PRP pool only moves the cliff: under the pair the salient/neutral
contrast has already vanished by the first morning. Step 3 is early because it is the RAG-like defect the owner named
and it reuses the existing reconsolidation read. Step 4 needs Steps 1-2 as its brake. Step 6 needs Step 1 (graded
traces) and Step 5a (local priority). Step 7 needs Step 4. Steps 0, 1 and 3 touch disjoint code and can be built in
parallel; 5a, 5b and 5c are independent of each other.

## 5. The registered test battery: what-matters gates

This section fixes the battery's design, gates and UNDEFINED rules now, before any code. Step 0's prereg pins the
code, the content lists and the constants, and may tighten but never loosen what is registered here. A gate below is
written so that it can FAIL, and 5.6 says which gates each baseline is predicted to fail.

### 5.1 Protocol

Two conversation groups keep the managed-block count within the composer's capacity (the arms raise `k_max` through
the existing override to at least the block count; P0 below).

| group | cues (facts per cue) | per-fact structure | other tellings |
|---|---|---|---|
| `wa` salience | N neutral (2), S surprise, the `datc` news frame (2), E told inside the user's emotional disclosure (2), F preceded by "please remember this" (2), BT-rel and BT-unrel (1 each) | cued core fact + a plain aside about the same agent in the next turn (the peripheral detail); BT facts core only | 2 unrelated plain facts per later day (interference), never probed |
| `wb` use | N (2), Rsp re-told once 2 h later (2), Rms re-told within the same minute (2), T asked once 10 min after the telling (2), G about the conversation's current topic referent (2), K about an entity with 3 prior facts told earlier that day (2) | as above | the 3 schema-prior facts (core only); interference as above |

- **Day 1 (virtual wall clock, the Amendment-7 seam):** target tellings in a morning block (09:00-10:30) and an
  evening block (18:30-20:00); each cue has one fact in each block, so cues are balanced for time before sleep. A few
  chit-chat turns surround each telling. Between turns the body is AWAKE by the environment's awake mark
  (`da_tag_capture_chat.mark_awake`, the r2 world step, extended to the virtual wall clock in Step 0), so the day's
  idle stretches are quiet wakefulness (awake bursts from Step 4 on), not sleep; two registered 20-min rest pauses
  follow the blocks. BT-rel is told 20 min before the second S fact and shares its agent; BT-unrel is told in the
  adjacent turn about an unrelated agent. The night starts at 24:00 by the body clock. Mornings at 08:00;
  interference facts at 10:00 and 16:00 on each later day. The production trigger (any idle of 5 min counts as sleep)
  is not used here; its effect is the `pp` day's and D8's question (section 6).
- **Content:** a pool of content triples per group rotated across cue slots by seed (a Latin-square shift), so
  vocabulary crosstalk is not confounded with cue; no (agent, action) cue of a target collides with a build-time or
  LTM fact, checked offline against the store's fact list before the run (test construction, not the brain).
- **Probes at the delay, once per arm:** for each target fact, in fixed order, the central question (the core's
  patient), the peripheral question (the aside's patient), then the referential probe ("you mentioned the <agent>",
  read on the episodic organ's `in_memory`). BT facts: central only.
- **Delay arms** (each arm runs the shared prefix and is probed only at its delay, so earlier probes cannot act as
  retrieval practice): `imm` (right after each telling; the P1 arm), `d1h` (21:00, day 1), `d4h` (24:00, awake, no
  sleep yet), `n1`, `n3`, `n7` (08:00 after 1, 3 and 7 nights).
- **Other arms:** `n1_b` (G0 null rebuild of `n1`); lesion arms at `n7` (wa: L-DA as the waking-only DA lesion with
  the SWR edge spared, Amendment 7's knob; L-NE; L-REL; L-PRIO) (wb: L-RECON; L-TOPIC; L-SCHEMA; L-PRIO), where L-PRIO
  cuts every importance edge, the replay bias and the re-induction on repetition and recall, while keeping the
  machinery (one trace per fact, graded capture, consumed local PRPs, the r3 brake); REPORTED `rp` (probed at every delay
  in one arm: the testing effect over the whole protocol) and `n7_vac` (no interference: retention in a vacuum,
  Wixted's case).
- **Env:** production defaults plus the flags of the step under test; LTM on (production), `BRAIN_EPISODIC_STORE=1`
  on numpy so the episodic organ writes; the seed through `BRAIN_CHAT_SEED` to `cfg.seed` (never `actual_seed_used`);
  the D1 reader seeded with the arm's seed and gamma calibrated a priori per seed (review I-4), the production-seed
  reader REPORTED beside it.

### 5.2 Grading

Per probe: correct, abstain, guess (flagged by the brain as a guess), confab (a wrong answer not flagged), undefined.
"Not recalled" is abstain or guess. Per target fact at a delay:

| outcome | condition | score |
|---|---|---|
| kept | central correct and peripheral correct | 2 |
| gist | central correct and peripheral not recalled; or central not recalled, referential probe familiar, and the reply discloses familiarity without content | 1 |
| lost | nothing recalled, not familiar | 0 |
| confab | any unflagged wrong answer on any probe of the fact | 0, and WM4 fails |
| inversion | peripheral correct while central not recalled | counted for WM9a |

`score(X)` is the mean over cue X's facts in the group; the retention curve of a fact is its score over the delay arms.

### 5.3 Gates (per seed and group)

| gate | passes only if | first expected at |
|---|---|---|
| WM1 importance order | for each cue X (wa: S, E, F; wb: Rsp, T, G, K): (a) score(X) >= score(N) at every delay, and (b) score(X) > score(N) at `n7` | per cue: S step 4, Rsp/T step 3, E/F/G step 5, K step 7 |
| WM2 ordinary facts fade gradually | (a) not at once: both N facts kept at `d1h` and mean N score >= 1 at `n1`; (b) not never: mean N score at `n7` below its `d1h` value and at least one peripheral detail in the group lost at `n7`; (c) no resurrection: no fact scored 0 at one delay scores above 0 at a later delay; (d) the group's loss events fall in at least two different delay intervals | step 1 |
| WM3 important kept | wa: every F and E fact's central correct at `n7`, every S fact's central correct at `n3`, every F fact's peripheral correct at `n3`; wb: every G, K, T and Rsp fact's central correct at `n3` | per cue as WM1; F peripheral step 6 |
| WM4 no confabulation | zero unflagged wrong answers on every probe of every gated arm; every familiarity-without-content reply names no content | always |
| WM5 not RAG-like storage | (a) after each re-telling and each practice recall the fact has exactly one managed block (the composer's own count); (b) score(Rsp) >= score(Rms) at `n3` and `n7`, and higher summed over delays; (c) score(T) > score(N) at `n7`; (d) at `n3` at least three target blocks per group express a fraction of their written increment strictly between 0.1 and 0.9 (the ledger's weight factor `e + z(1 - e)` averaged over the block's synapses; 1 by construction with no ledger) | (a-c) step 3; (d) step 1 |
| WM6 effort follows importance | the mean number of replay reactivations won by cued target blocks exceeds that of N blocks over the protocol; REPORTED (6b): PRP consumed per block | step 4 (6b step 2) |
| WM7 lesions remove the prioritization | at `n7`, under each channel lesion the lesioned cue's advantage is gone (score(X) <= score(N)) while at least one other cue's advantage remains; under L-PRIO every cue's advantage is gone while WM2a and WM2b still hold (importance-blind, neither amnesic nor keeping everything) | with each channel |
| WM8 behavioural tagging is specific | score(BT-rel) > score(BT-unrel) at `n3` | step 2 |
| WM9 the gist survives the detail | (a) no inversion for any cued fact at any delay (N inversions REPORTED); (b) at `n3` or `n7` at least one cued fact per group reads gist | step 6 |
| WM10 prioritizing does not mean forgetting more | summed target score at `n7` >= the pair-as-is baseline's at the same seed (the Step-0 row, re-run if the protocol changes); REPORTED against today's default | step 1 |

Verdicts: a group reads GO at a seed only if every gate registered for the step under test holds and no UNDEFINED rule
fires; a step's family verdict is GO iff all six seeds read GO, INCOMPLETE if a seed is missing, NO-GO otherwise. The
one-sided exact sign-flip p over seeds is reported for WM1b per cue, WM5b and WM8. The flip candidate is held to every
gate.

### 5.4 UNDEFINED rules (never scored as a pass or as zero)

- **U0 G0:** `n1` and `n1_b` differ in any outcome, triple, abstain flag, familiarity read, ledger state, sleep record
  or final store.
- **U1 P0 input and capacity:** a telling did not store the registered number of blocks; the managed-block count
  exceeded `k_max`; a scripted re-telling, practice recall or rest pause did not happen; a world step failed.
- **U2 P1 learned first:** in `imm`, any target fact's central or peripheral probe was not correct (the brain cannot
  forget what it never learned). The seed is UNDEFINED for that group.
- **U3 lesion held:** a lesion did not hold on the record at every turn and epoch (docs/TERMS.md "lesion"); that
  lesion's WM7 row is UNDEFINED.
- **U4 run integrity:** a gated arm errs, or a probe reads undefined.
- **U5 constants:** gamma differs across the ledger-on arms of a seed; the clock is not the registered virtual wall
  clock; the epochs did not occur at the registered times for the step's design.
- Cue manipulation checks read on the brain (the D1 read at an S telling, the arousal read at an E telling, the latch
  state after an F cue, the referent state at a G telling) are REPORTED, not UNDEFINED rules: if the brain does not
  register a cue, the resulting failure is the brain's, and the baselines must be able to fail it.

### 5.5 What the gates can and cannot show

They measure behaviour at the reply (what the brain recalls, hedges or invents) and the brain's own records (block
counts, late-phase fractions, replay wins, lesions). They cannot say how a model dose maps onto a human day, and two
facts per cue per seed make each per-seed comparison coarse; the six-seed sign test is the evidence. Probes are single
questions per component, so partial recall inside a component is not graded. The episodic read is topic-level
familiarity only.

### 5.6 Baselines the battery must fail (discriminating power, Step 0)

| config | predicted to fail | predicted to pass |
|---|---|---|
| today's production default (ledger off) | WM2b (nothing fades), WM5a (a re-telling appends), WM5d (no graded traces), WM6 (no replay selection), WM1b for every cue (everything kept, nothing ordered) | WM2a, WM4 |
| the pair as it stands (DA capture + sleep route) | WM5a, WM5d (fractions only 0 or 1), WM6 (every block reactivated once a night), WM8 (the global pool captures both), WM1b for E, F, G, K, T; WM2a likely (a neutral fact told about 4 h before sleep was lost 6/6 in r2) | WM4 |

### 5.7 What it takes to run

Timing basis: the fi family's seed-42 smoke ran nine 7-night arms of up to 20 tellings in 26-36 min each on numpy, one
process per arm, about 0.7 GB per process. This battery's day is longer (about 40 turns of tellings and chit-chat, 14
interference tellings, about 40 probes), so an `n7` arm is estimated at about 45 min before Step 4 and about 90 min
after it (several epochs a night; the per-epoch cost is measured on the dev smoke before the six rows are queued);
shorter delay arms 15-50 min. Estimates only:

- Step 0 baselines: 2 configs x {imm, n1, n7} x 2 groups = 12 arms per seed, about 7 CPU-h per seed, about 42 CPU-h
  for six seeds: about 3.5 h of wall time on the pool at 12 concurrent workers (four pool nodes x 3, the fi layout).
- The full battery (after Step 4): 13 arms per group, 26 per seed, about 27 CPU-h per seed, about 160 CPU-h for six
  seeds: about 13 h on the pool, about 8 h with local cores added under `tools/memcap.sh`.
- The reference-3090 subset (the episodic organ writes by default on cupy): wa `imm`, `n1`, `n7`, L-PRIO for six
  seeds, 24 arms through `tools/gpu_queue.sh`, roughly 12-24 GPU-h.
- On-demand AWS CPU inside the owner's approved daily cap is an overflow option, not needed.
- A seed-7 dev smoke precedes every six-seed set; a full-brain snapshot fork at the branch points (the GNW fork
  instrument generalized) could cut the shared-prefix cost but needs its own fork-equals-rerun check first.
Nothing is queued by this document.

## 6. How this changes the pair's flip criteria (legs b-d)

The flip candidate is no longer the pair alone but the pair plus Steps 1-4 at least, and the pair stays default OFF
until then (the owner's "wait on fix").

- **Leg (b), verify-go review.** B1 (record corrections) is unchanged and still required. B2 (the owner's decision on
  the measured forgetting) is answered in principle by the ruling: losing an ordinary fact is acceptable when the
  loss follows importance, so the 0/6 loss of a neutral fact told 4 h before sleep is not by itself a blocker. What
  blocks is that the loss is importance-blind. B2 becomes a registered criterion: the what-matters gates for the
  channels built (WM1, WM2, WM4, WM5, WM7, WM8, WM10) read GO 6/6 on the production path, wall clock, LTM on. B3 (the
  `pp` wall-clock day) stays; its registered verdicts stand as registered, its WD2 ("ordinary kept") is read as
  REPORTED for the flip decision, and its NR gate becomes binding (it is WM2c on a production day). B4 (`sn`: salient
  vs neutral at long delay in one family, waking-only DA lesion) stays as the first registered instance of WM1 and
  WM7 for the dopamine channel. B5 (the weak telling read at once and with the ledger off) stays; it is the P1 logic.
- **Leg (c), combined no-regression battery.** C1-C3 carry over, run at the revision that carries the prioritization
  steps. Added: every memory faculty that reads the composer store (episodic, source provenance, prospective memory,
  WM binding, common ground) keeps its load-bearing row, because Steps 3 and 6 change what a told fact writes.
- **Leg (d), production-default validation.** D1-D6 carry over. D6 (many managed facts) becomes a behaviour check,
  not only a latency check: the battery is a many-facts session. Added D7: the full battery at production defaults (no
  flags in env), LTM on, the episodic organ writing (cupy on the 3090, or `BRAIN_EPISODIC_STORE=1`). Added D8: a
  normal day with several 5-minute pauses (several epochs inside one early-phase window) raises no minor fact over an
  important one and resurrects nothing (WM1a and WM2c on the `pp` day).

## 7. The in-flight lanes: keep, reshape, supersede

- **fi family (`BRAIN_SLEEP_LOAD_RENORM`, six rows on the pool): keep, unchanged.** It is Step 4's brake and the
  interference half of WM2 (fading follows later learning, not the count of nights). One reading note for its
  verdict: its re-mention protection (FI6) was carried by the re-mention's new block, not by the original (seed 42:
  the original fell to a ratio of 0.406652, like the unmentioned arm's), which is the append pattern Step 3 removes. <!--derived-->
- **Awake-replay completion (`research/awake-replay-completion-r2`, dev): reshape.** Pattern completion is real
  biology and Step 4 needs it as its reactivation read. Its current target, rescuing a neutral fact told 4 h before
  sleep on every seed (the arc family's seed-101 miss), is no longer a requirement under the ruling, and completion
  without importance-weighted competition pushes toward keeping every trace that still selects its items. Recommended:
  stop spending levers on the seed-101 neutral rescue; fold completion into Step 4 with the gates WM6, WM2c and WM1.
- **Production-path arms (`research/pair-production-path-arms`, `pp` / `sn` / `cu`, Amendment 7 registered before
  any run): keep, as registered.** The wall-clock seam, the cupy RNG restore, the waking-only DA lesion knob and the
  episodic-agreement arm are infrastructure this battery reuses. For the flip: WD2 REPORTED, NR binding, SN1 and SN2
  the first WM1 / WM7 instance, `cu` required for D3 and for the episodic organ's gist role. Its registered NR
  response (Fonseca 2004 PRP competition) is Step 2 here: consistent, not superseded.
- **Awake-rest replay capture (`BRAIN_AWAKE_REPLAY_CAPTURE`, arc family NO-GO, 5 of 6 seeds passed): superseded in role** by Step 4's awake
  bursts that bias the night; the code stays, it is not a flip candidate on its own.
- **r2 constant downscaling (NO-GO 0/6): superseded** by r3, already.
- **The pair's flip: on hold** until Steps 1-4 and the gates of section 6 hold.

## 8. The honesty boundary

Every memory self-report is a functional read-out of a measured state, never a claim of experience. Gist:
"You told me something about the dog, but I can't recall what" only when the episodic familiarity read is positive
and the recall abstains. Flagged facts: "you asked me to keep this" only when the relevance latch is on. Fading: "I'm
not sure any more; that was a while ago" only when the recall margin is in the registered uncertain band. The brain
never asserts a detail it cannot recall (WM4), and never says it "feels" that it remembers.

## 9. What this design does not do, and open questions

- It builds nothing, queues nothing and changes no default. Each step is its own build, binding, prereg and review.
- The ledger's host-integrated synaptic equations stay; Step 2's compartment allocation is a declared host step; the
  sleep/wake clock stays the body's host clock.
- A further constant, named here and not addressed: production starts "sleep" after any 5 min of idle
  (`continuous_engine.SLEEP_IDLE_SEC`), where a real brain runs quiet wakefulness with awake ripples and sleeps when
  sleep pressure and circadian phase say so. The battery uses the body's awake mark instead; the production effect is
  D8's measurement, and a body-clock sleep trigger is its own later design.
- Open: the readiness distribution's shape beyond "fewer than half primed" (the Step-1 prereg fixes it from the
  source, not from a gate seed); whether an excitability mark or a separate tag variable best carries importance to
  replay (Step 4 prereg decides, with a lesion either way); how "remember this" reaches the latch without a keyword
  test before the learned language route covers it (a declared scaffold needs an owner waiver); signed value (5d) has
  no afferent yet; self-relevance (facts about the conversation partner) is a plausible further cue, not yet sourced.
- Research record: `bash tools/before_you_build.sh "memory prioritization what matters"` and
  `bash tools/deep_research.sh` run on 2026-09-25; the external sources (PubMed, PubMed Central, Consensus) are in the
  five biology entries and in `research/queue/.external_searches.jsonl`.

## Sources

<!--derived-->

(Block marker above: the numbers in this section are DOIs, not measurements.) External, read 2026-09-25 (full text where noted, otherwise the abstract):
Frey & Morris 1997 (doi 10.1038/385533a0); Redondo & Morris 2011 (doi 10.1038/nrn2963); Wang, Redondo & Morris 2010
(doi 10.1073/pnas.1008638107); Bethus, Tse & Morris 2010 (doi 10.1523/JNEUROSCI.2721-09.2010); Takeuchi et al. 2016
(doi 10.1038/nature19325); McGaugh 2004 (doi 10.1146/annurev.neuro.27.070203.144157); Cahill et al. 1994
(doi 10.1038/371702a0); Mather et al. 2016 (doi 10.1017/S0140525X15000667); Wilhelm et al. 2011
(doi 10.1523/JNEUROSCI.3575-10.2011); Stickgold & Walker 2013, full text (doi 10.1038/nn.3303); Dunsmoor et al. 2015,
full text (doi 10.1038/nature14106); Oudiette et al. 2013 (doi 10.1523/JNEUROSCI.5497-12.2013); van Kesteren et al.
2012 (doi 10.1016/j.tins.2012.02.001); Fonseca et al. 2004 (doi 10.1016/j.neuron.2004.10.033); Govindarajan et al.
2011 (doi 10.1016/j.neuron.2010.12.008); Govindarajan, Kelleher & Tonegawa 2006 (doi 10.1038/nrn1937); Kramar et al.
2012 (doi 10.1073/pnas.1120700109); Rogerson et al. 2014 (doi 10.1038/nrn3667); Yiu et al. 2014
(doi 10.1016/j.neuron.2014.07.017); Yang et al. 2024, Science, via Consensus; Schapiro et al. 2018
(doi 10.1038/s41467-018-06213-1); Singer & Frank 2009 (doi 10.1016/j.neuron.2009.11.016); Ambrose, Pfeiffer &
Foster 2016 (doi 10.1016/j.neuron.2016.07.047); Mattar & Daw 2018 (doi 10.1038/s41593-018-0232-z); Rasch & Born 2013
(doi 10.1152/physrev.00032.2012); Payne et al. 2008, full text (doi 10.1111/j.1467-9280.2008.02157.x); Sekeres et al.
2016 (doi 10.1101/lm.039057.115); Winocur & Moscovitch 2011 (doi 10.1017/S1355617711000683); Reyna et al. 2016
(doi 10.1016/j.jarmac.2015.12.003); Corbin et al. 2015 (doi 10.1016/j.jarmac.2015.09.001); McClelland, McNaughton &
O'Reilly 1995 (doi 10.1037/0033-295X.102.3.419); Tse et al. 2007 (doi 10.1126/science.1135935); Richards & Frankland
2017 (doi 10.1016/j.neuron.2017.04.037); Hardt, Nader & Nadel 2013 (doi 10.1016/j.tics.2013.01.001); Berry et al.
2012 (doi 10.1016/j.neuron.2012.04.007); Karpicke & Roediger 2008 (doi 10.1126/science.1152408); Lee 2008
(doi 10.1038/nn.2205); Cepeda et al. 2006 (doi 10.1037/0033-2909.132.3.354).
Local corpus: Kandel 6e ch.42, ch.52, ch.53, ch.54; Buzsaki, Rhythms of the Brain (2006), cycle 7 and p.123, p.349.
