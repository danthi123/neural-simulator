---
type: design
status: live
date: 2026-09-25
lane: load-bearing
mechanism: prioritized memory (keep what matters, let minor details fade) as the companion of the DA tag-capture + sleep-replay pair, revision 3 -- every review item of b82e7d2 and of fcf3c847d closed, and encoding added after the fi battery's first-morning losses -- research and design only, no sim/ or webapp/ change
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_sleep_replay_capture/aggregate.json
  - research/findings/raw/_sleep_replay_capture_r2/aggregate.json
  - research/findings/raw/_awake_replay_capture/aggregate.json
  - research/findings/raw/_sleep_forgetting_interference_smoke/seed42.json
  - research/findings/raw/_sleep_forgetting_interference/seed42.json
  - research/findings/raw/_sleep_forgetting_interference/seed43.json
  - research/findings/raw/_sleep_forgetting_interference/seed44.json
  - research/findings/raw/_sleep_forgetting_interference/seed100.json
  - research/findings/raw/_sleep_forgetting_interference/seed101.json
  - research/findings/raw/_sleep_forgetting_interference/seed102.json
  - research/findings/raw/_sleep_forgetting_interference/seed*/fiv_lr.json
  - research/findings/raw/_sleep_forgetting_interference/seed*/fis_lr.json
  - research/findings/raw/_sleep_forgetting_interference/seed43/fir_lr.json
  - research/findings/raw/_sleep_forgetting_interference/seed101/fir_lr.json
biology:
  - research/biology/importance-tagging-at-encoding.md
  - research/biology/prp-competition-and-locality.md
  - research/biology/prioritized-replay-triage.md
  - research/biology/gist-detail-graded-forgetting.md
  - research/biology/repetition-retrieval-strengthen-same-trace.md
  - research/biology/encoding-strength-and-allocation.md
  - research/biology/homeostatic-scaling-relative-strength.md
  - research/biology/awake-replay-tag-capture.md
  - research/biology/sleep-load-dependent-renormalization.md
---

# Prioritized memory: remember what matters, let minor details fade (DESIGN, revision 3)

Research and design, no code. It answers the owner's ruling of 2026-09-25 on the overnight forgetting of the DA
tag-and-capture + sleep-replay pair, maps what the brain already has against what real brains run to prioritize
memory, orders the mechanisms to build, and registers the test battery ("what-matters gates") the mechanisms will be
held to. Terms follow `docs/TERMS.md`: "consolidation" is used only for the biology until a source lesion earns it in
the model, and a verdict word only as a gate's own verdict.

Revision 2 (branch `research/prioritized-memory-design-final`) closes all seventeen items of the review of `b82e7d2` (eight MEDIUM)
(section 10 lists each and where it landed) and adds encoding: section 2a reads the fi battery's six seed records and
answers whether the design must address what gets stored and how strongly, not only retention. It must (section 2a,
Steps 0b and 1a, gate WM0). Revision 3 closes the ten items of the independent review of `fcf3c847d` (five MEDIUM;
section 10b): the homeostat's effect on the replay tag and capture with the ledger on, a misread of seed 43's
re-mention record, WM0b's read, WM2a's timing and the per-step gate sets, the cued read's host path, and five LOW
items. The document is ready for a preregistration and a build; it queues nothing.

**Merge order.** Section 2's central measurement (the keep/lose cliff) and section 7's Amendment 7 cite documents
that are on branch `research/pair-docs-final` (which carries `research/pair-verify-go` and
`research/pair-production-path-arms`) and not yet on main: merge that branch before this one, or with it.

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

- **What the user tells the brain is learned.** A plainly told fact can be recalled right after it is told, and how
  strongly it is written depends on what it is worth (attention, feeling, surprise, relevance), never on where in the
  store it happened to land.
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

What "RAG-like" means here, measured: today's production default writes every told fact once, scaled only by the DA
write gain, and appends a re-told fact as a new block (research/findings/2026-09-25-sleep-forgetting-interference-fi-seed42-smoke.md:
on the re-mention arm the recall matched the day-3 re-mention block while the original fell like an unmentioned
fact's). The only later process that changes a stored fact by default is the per-engram homeostatic scaling pass
(section 3): it lifts every weak engram up to one set-point and pulls strong ones part of the way down. It equalizes
and never forgets. The pair adds forgetting, but through one threshold that does not look at importance (section 2).

## 2. Why the pair's keep-or-lose is a single threshold

The adversarial review of the pair (research/findings/2026-09-25-da-capture-sleep-replay-pair-verify-go-review.md,
branch `research/pair-verify-go`, carried by `research/pair-docs-final` and not yet on main, finding I-1) measured,
across 24 seed x telling cells with both flags on, that the next-day outcome is a step in one number: R, the store's
cleanup margin at the night's one SWR epoch. Every cell at R 0.185 or below was lost, every cell at 0.209 or above was <!--derived-->
kept. Reading the code gives four reasons, each a constant standing where the real system runs a process:

1. **One switch per fact.** The composer writes a block as `complex(g) * zc[k]` with `zc` a unit phasor
   (`research/runners/one_brain_composer.py` `_write_block`), and the ledger sets each synapse's tag to
   `h0 = |inc_k|` (`webapp/da_tag_capture.py` `SynapticTagCaptureLedger.on_store`). Every synapse of a block carries
   the same tag, so every late-phase variable `z_k` follows the same trajectory and the block is captured or lost as
   one unit. The ledger's own summary field `frac_synapses_z_gt_half` reads 0 or 1 by construction (up to rounding),
   and every row of the review's table reads `z` at recall as 0 or 1. Real synapses differ in readiness to be
   potentiated at induction (fewer than half of spines are primed at baseline, Kramar et al. 2012), and capture is
   branch-local and competed for (Govindarajan et al. 2011), so a trace is potentiated and captured in part.
2. **One read, one epoch, no selection.** `SleepReplayCapture` runs one epoch per night and drives every managed
   block once, identically; R is the minimum margin over agent/action/patient (`reactivation_strength`). A strong
   trace reads high and is re-tagged, a weak one is not, whatever it is worth. The same R also sets the night's PRP
   supply: the SWR dopamine is a host map of the sum of R over blocks (`swr_da`), so R enters twice. Real nights have
   four or five NREM cycles (Buzsaki), and replay content is biased by reward, by awake ripples, by expected use and
   toward weak items (Singer & Frank 2009; Yang et al. 2024; Wilhelm et al. 2011; Schapiro et al. 2018).
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
importance-ordered curve: (C-0) encoding whose strength is set by the brain's state at the telling, not by the slot the
fact lands in (section 2a); (C-1) heterogeneous synapses plus limited, local, competed-for PRPs -- potentiation and
capture become graded fractions and resources become contested; (C-2) repeated, biased reactivation across the day and
several NREM cycles and nights, with the load-set renormalization (r3) as its brake; (C-3) several importance channels
at encoding that act locally and spread only to related memories; (C-4) component-wise traces (gist and detail
separable) plus repetition and recall that strengthen the same trace. Raising a threshold, retuning gamma or changing
the replay-to-DA map is not a response to this diagnosis.

## 2a. What the fi battery's first mornings add: encoding, not only retention

**What was run.** The forgetting-interference family (prereg Amendment 6; ledger, sleep route and r3 load
renormalization on, LTM off, numpy) tells one fact, "the cat chases the ball", as the WEAK telling (`_DATC_WEAK` in
`research/runners/onebrain_regression_battery.py`: the fact said last of five turns, after its words have habituated
the brain's DA), then runs seven nights with a recall question each morning. The harvested aggregate (in the primary
checkout, not committed: commit `fe1066f64` says why) reads NO-GO, 3 of 6 seeds GO. On seeds 43 and 101 the brain
abstains on the FIRST morning in every arm that told the weak telling -- fiv (nothing else learned, FI1 false), fil,
fih and fir -- because the arms are identical through night 1 (I3 holds on every seed).

**The six records, read.** Per seed, from `research/findings/raw/_sleep_forgetting_interference/seed*.json`: the write
magnitude of the fact's block (`tag0_mean`, the ledger's write tag = the written increment's magnitude), the block's
reactivation read R at the first SWR epoch, and the first-morning outcome, for the weak telling (arm `fiv_lr`) and for
the salient telling of the same fact (arm `fis_lr`, same seed, same block index):

<!-- table below: two-decimal reads of the fi seed records (not claim-checked: under three decimals) -->
| seed | weak: write magnitude | weak: R at first epoch | weak: morning 1 | salient: write magnitude | salient: R | salient: morning 1 |
|---|---|---|---|---|---|---|
| 42 | 1.23 | 0.36 | correct | 2.46 | 0.47 | correct |
| 43 | 1.00 | 0.17 | abstain | 2.20 | 0.39 | correct |
| 44 | 1.00 | 0.31 | correct | 2.00 | 0.50 | correct |
| 100 | 1.00 | 0.32 | correct | 2.33 | 0.46 | correct |
| 101 | 1.00 | 0.04 | abstain | 2.12 | 0.29 | correct |
| 102 | 1.05 | 0.42 | correct | 1.97 | 0.55 | correct |

What the records show:

1. **What gets stored at all did not fail.** Every seed wrote one managed block for the telling
   (`daily_n_managed_blocks` = 1 in `fiv_lr`), and the confabulation gate held everywhere (FI7).
2. **Encoding set the read that the first night acted on.** The first SWR epoch ran five minutes after the telling
   (sleep onset after the last turn), when the early phase was still about 95 % of its written value, so R there is
   the encoded read, not a decayed one. On seeds 43 and 101 it sat below the pair's measured cliff (section 2), the
   late phase never started (`z_mean` about 0 at the first recall), and the early phase then decayed by morning. On
   seed 101 the loss was decided at encoding (a read at noise, below the separator); on seed 43 encoding set a read
   above noise and just under the cliff, and the pair's single threshold decided it (the paragraph after this list).
   With one block in the store the night's SWR dopamine is set by that block's own R (`da_swr` 0.62 and 0.53 against
   0.73-0.81 on the passing seeds), so a weakly encoded lone fact also cannot pay for its own capture: R entered
   twice. <!--derived-->
3. **At the floor write, encoding strength is a lottery.** Four seeds wrote the weak telling at magnitude 1.0, the DA
   write gain's recall-safe floor (`webapp/da_encoding_drives_chat.py`: a below-tonic-DA fact is written at unit
   magnitude), and read 0.17, 0.31, 0.32 and 0.04: same words, same script, same turn, same block index. With the
   ledger on, a gain-1 increment lies on top of a baseline the ledger draws once per (seed, block index) at the same
   expected magnitude (`BETA_BASELINE = 1.0`,
   "increment ~ baseline"), and the read is the minimum over three roles of a D = 128 matched-filter margin
   (`brain_conversational_agent.py` default). Which component owns the spread -- the slot's baseline draw, the seed's
   word codes, or read noise -- is not in the record. The re-mention arm (`fir_lr`, 5 build-time facts, so recall
   `matched_fact_index` 5 is managed block 0 and 10 is managed block 5) tells the same fact again on days 2 and 3,
   each time into a new block. Read at each block's first night, the same content read, on seed 43, 0.168588 (block
   0), 0.008005 (block 1, the day-2 re-mention: captured through the shared PRP pool, yet never the block that
   answered) and 0.362487 (block 5, the day-3 re-mention, which answered every recall from the third on); on seed 101,
   0.035600, 0.282606 (block 1, which answered recalls 2 and 3) and 0.482487 (block 5, recalls 4 to 7)
   (research/findings/raw/_sleep_forgetting_interference/seed43/fir_lr.json,
   research/findings/raw/_sleep_forgetting_interference/seed101/fir_lr.json). Identical content spanned about 0.01 to
   0.48 across slots of one seed, which if anything points toward slot or store state rather than the word codes; but
   here the slot is confounded with the store's load and the write gain (block 5 wrote at 1.12 against 1.0), so the
   record does not attribute it. Step 0b does.
4. **Importance already acts at encoding, through dopamine.** The salient telling of the same fact wrote at about
   twice the magnitude and read 0.29-0.55 on every seed, above the cliff, and was recalled on the first morning on
   seeds 43 and 101 too (seed 101's salient fact was lost later, by day 5: FI5 false, a retention matter). At a fixed
   seed and block index, the stronger write lifted the read on every seed. <!--derived-->
5. **The family's precondition could not see any of this.** Its P1 (`P1_immediate_precondition`) is the arm
   `neu_imm_fi`, which tells a DIFFERENT telling (the neutral script, fact third of five, `_DATC_NEUTRAL`), writes at
   1.27-1.84 and recalls correctly on all six seeds. No arm asked about the weak telling right after it was told, so
   the record cannot say whether seeds 43 and 101 could have answered at once. R = 0.04 on seed 101 is below the
   composer's own clean/noise separator (g = 0.15, `OneBrainComposer._block_role_scores` docstring); if its weakest
   role was a cue role (agent or action), the confidence gate would have abstained even at once. <!--derived-->

The two seeds differ. Seed 101's read (0.04) is below the composer's own separator: the fact was encoded at noise, an
encoding defect. Seed 43's (0.17) is above the separator but below the pair's capture cliff: a weakly encoded trace
meeting the pair's one-threshold retention (section 2). The design addresses both, the first at the write (Step 1a,
WM0c), the second through graded capture and selective replay (Steps 1-4, scored by WM2a). <!--derived-->

**The answer: yes, the design must address encoding.** Not admission -- every telling was stored -- but how strongly
and how cleanly a plainly told fact is written, which today is set by the seed's realization of the store at the floor
write gain and not by anything the fact is worth. The owner's rule allows a minor fact to fade, but not at once and
not by lottery: a fact lost by the first morning in the vacuum arm, at the same write gain at which other seeds kept
it, is not "forgetting minor details over time". The design is changed in four places:

- **Measure it.** The encoding read becomes graded and per arm (Step 0 and 5.1): right after each telling, in every
  arm, the arm's OWN telling is decoded and its written magnitude, its R0 (the sleep route's own reactivation read) and
  its cue-role margins are recorded. A precondition read on a sibling script is not allowed (the fi lesson). R0 is a
  normalized margin and cannot see how strongly a fact was written, so importance at encoding is read on the magnitude
  (WM0b) and noise-level encoding on R0 (WM0c).
- **Score it as the brain's outcome.** A fact that does not encode is scored as lost, under a new gate WM0 (learned,
  and learned by importance, and not by lottery), instead of voiding the seed (5.3, 5.4 U2).
- **Attribute, then build.** Step 0b attributes the spread of R0 at a fixed write gain to slot, content, read noise and
  ensemble size; Step 1a builds the encoding mechanism that the attribution points to, by a decision rule registered
  now (section 4).
- **Importance acts at encoding too.** Every Step-5 channel raises encoding strength as well as the tag (Kandel ch.52:
  encoding is stronger when one is motivated to remember), measured by WM0b.

Not changed: there is no storage gate that refuses ordinary facts. A brain does fail to encode what it does not attend
to (Kandel ch.52, absent-mindedness), and a faithful arousal mechanism may encode a background detail weakly (GANE);
those are brain outcomes, scored as such.

## 3. What the brain has, has off, or lacks

HAS = on by default; HAS-OFF = built behind a default-off flag or runner-only; LACKS = no mechanism, or no edge from an
existing organ to memory. Biology entries are those listed in the frontmatter.

| process (biology) | this brain | status | where |
|---|---|---|---|
| encoding strength set by depth, attention and state; allocation by excitability; separation from stored patterns (Kandel ch.52; Wagner 1998; Han 2007; Yiu 2014; Leutgeb 2007; Hasselmo 2006) | a told fact goes to the next free block at the DA write gain; with the ledger on, onto a seeded baseline of equal expected magnitude; no allocation competition, no separation step on the chat store's write (the composer's DG sparse index is default off), no encoding-state signal besides novelty DA | LACKS (write gain HAS) | section 2a; `one_brain_composer.py` `_write_block`; `da_tag_capture.py` `BETA_BASELINE` |
| novelty / surprise dopamine marks a memory, behavioural tagging (Wang 2010; Moncada & Viola 2007); LC dopamine co-release is a second novelty route (Takeuchi 2016) | spiking SNc driven by the habituation novelty organ + a host content-word count; DA write gain; DA -> spiking D1 pool -> PRP -> per-synapse capture | write gain HAS (`BRAIN_DA_ENCODING` default on); capture HAS-OFF (`BRAIN_DA_TAG_CAPTURE`) | `webapp/da_encoding_drives_chat.py`, `webapp/da_tag_capture.py`; chat-wire GO LTM-off and LTM-on 6/6 |
| homeostatic synaptic scaling: per neuron, over all of its inputs, each in proportion to its strength, over tens of hours (Turrigiano 1998; Kandel ch.49) | `OneBrainComposer.apply_homeostatic_scaling` on an idle tick after new facts: each block scaled alone, a weak engram up to the set-point (at most x4), a strong one part of the way down (`ratio ** 0.25`, at least x0.34) | HAS (`BRAIN_DA_ENCODING` + `BRAIN_DA_ENCODING_SUBSTRATE` default on); per-memory, not per-neuron | `webapp/continuous_engine.py` `consolidate_substrate_homeostasis`; `homeostatic-scaling-relative-strength` |
| prediction-error salience to the SNc | surprise organ's mismatch rate replaces the engagement mix on assertions | HAS-OFF (`BRAIN_REWARD_VALUE_AFFERENT`); its regex assertion gate is a declared shortcut | `webapp/reward_value_afferent_chat.py` |
| signed reward / value (Singer & Frank 2009; Oudiette 2013; Kandel ch.52) | none; the afferent above is unsigned by its own docstring | LACKS | -- |
| emotional arousal: beta-adrenergic noradrenaline through the BLA (McGaugh 2004; Cahill 1994 is the lesion) | LC-like arousal population (runner GO); LC-NE gain swap in the GNW (runner GO); affect organ with arousal rungs on the shared pool (3/6, NOT ALL-GO); interoceptive affect (runner GO) | organs HAS-OFF; the edge to encoding strength / tag LACKS | `2026-08-13-affect-lc-arousal-population-GO.md`, `2026-09-04-gnw-lc-ne-adaptive-gain-swap-eviction-GO.md`, `2026-09-23-onebrain-affect-ladder-...-NOT-ALL-GO.md` |
| local priority: key detail up, background down (Mather 2016 GANE; Payne 2008) | none | LACKS | -- |
| expected future use, "remember this", directed forgetting (Wilhelm 2011; Stickgold & Walker 2013) | prospective-memory intention latch + Hebbian binding + NMDA facilitation holds an intention; no edge to a fact's tag or replay; no forget cue | latch HAS (`BRAIN_PMEM_FACILITATION` default on since 2026-09-23, load-bearing 6/6); edge LACKS | `research/runners/prospective_memory_production_organ.py`; `2026-09-22-prospective-memory-facilitation-load-bearing-6seed.md` |
| topic / goal relevance (Kandel ch.52; Dunsmoor 2015) | common-ground ledger (NMDA attractor per referent, wired); WM referent focus binding (6/6, 2026-09-25) | organs HAS / HAS-OFF; edge to memory LACKS | `webapp/common_ground_drives_chat.py`, `2026-09-25-wm-referent-focus-bind-GO-6seed.md` |
| schema fit (Tse 2007; van Kesteren 2012) | CA3 superposed-fact attractor (runner, capacity 6/6, no chat write path); the LTM tier is a bulk, teacher-loaded closed-form store | LACKS on the chat path | `research/runners/ca3_superposed_fact_attractor.py`, `research/runners/tiered_fact_store.py` |
| repetition strengthens the same trace; spacing (Kramar 2012; Lee 2008; Cepeda 2006) | a re-telling appends a new block; reconsolidation rewrites only on a prediction error; "restabilize" writes nothing | LACKS (append is the RAG-like pattern) | `OneBrainComposer.update_on_mismatch`; fi seed-42 smoke `fir_lr` |
| retrieval strengthens, feedback not needed (Karpicke & Roediger 2008; Roediger & Karpicke 2006; Sekeres 2016) | reads never write (kept for the read itself: systems-consolidation protocol rule) | LACKS | `research/biology/systems-consolidation.md` |
| graded induction over heterogeneous synapses (Kramar 2012) and graded, branch-local capture (Govindarajan 2011) | one tag value per block, so one switch | LACKS | section 2 item 1 |
| limited, local, competed-for PRPs (Fonseca 2004; Govindarajan 2011) | one global scalar `p`, never consumed | LACKS | section 2 item 3 |
| replay selection: awake ripples tag sleep content, reward bias, weak items first, several cycles and nights (Yang 2024; Schapiro 2018; Buzsaki) | one epoch per night, every block driven once, SWR DA a host map of the sum of R; awake bout OFF (arc family NO-GO, 5 of 6 seeds passed); pattern completion (branch, dev); a risk-prioritized teacher-loop replay (2026-08-09: beat random, failed coverage at a fixed budget) | HAS-OFF, uniform; selection LACKS | `webapp/sleep_replay_capture.py`, `webapp/awake_replay_capture.py`, `research/awake-replay-completion-r2` |
| brake: the night's renormalization set by the day's learning (Tononi & Cirelli 2014) | r2 constant (NO-GO 0/6); r3 load-dependent `BRAIN_SLEEP_LOAD_RENORM`, fi family harvested (NO-GO 3/6, section 2a) | HAS-OFF | `research/biology/sleep-load-dependent-renormalization.md` |
| gist vs detail kept separately (Payne 2008; Sekeres 2016; Winocur & Moscovitch 2011) | one block per fact, R = min over roles; the episodic organ's topic familiarity is not ledger-managed and keeps every topic | LACKS | `reactivation_strength`; Amendment 7 `wd_epi` prediction |
| replay-written transfer to a slow cortical store, fast for schema-consistent facts (McClelland 1995; Tse 2007) | `promote_buffer_to_ltm()` is a host hook, never auto-invoked | LACKS | `research/runners/tiered_fact_store.py` |
| regulated forgetting (Hardt 2013; Berry 2012; Richards & Frankland 2017) | passive early-phase decay + r2/r3 downscaling | LACKS beyond r3 | -- |

**The homeostatic pass and the what-matters gates.** The pass is on by default and sits in every arm, so the design
states how it interacts (biology: `homeostatic-scaling-relative-strength`).

- *What it does.* Each block owns its own D readout units (trig+1..trig+D), so the per-neuron rule of the tissue
  becomes a per-memory normalization here: every engram below the set-point is lifted to it, every engram above is
  pulled toward it. In the tissue the same rule preserves the order of the memories a neuron carries (Turrigiano 1998:
  each synapse scaled "in proportion to its initial strength"); here no unit carries more than one memory, so there is
  no order to preserve. It also runs on the first idle tick after a write, where the tissue takes tens of hours.
- *With the ledger off (today's default).* The per-write gain is already floored at the set-point (1.0), so an N fact
  sits at it; an S fact written at about twice the gain is pulled part of the way down (for a 2x engram by
  `0.5 ** 0.25`, about 0.84), so the S-N magnitude difference that the DA write gain created shrinks at the first idle
  tick, before any delay. This works against WM1 for S in the default baseline (not against WM0b, which reads the
  written magnitude right after the telling, before the pass).
- *With the ledger on.* The pass senses the whole block, the drawn baseline plus the increment, so a gain-1 fact whose
  baseline adds to its readout sits ABOVE the set-point and is pulled down too, and a strongly written fact is pulled
  down further. `sync_from_store` (`webapp/da_tag_capture.py`) reads the pass as an external rescale and multiplies the
  block's baseline and increment by its scale s, but not its write tag h0 (the fi records count the passes:
  `external_rescales` is 1 in the one-fact arm `fiv_lr` and 70 in `fih_lr_a`, consistent with a pass after each day's
  new facts that rescales every managed block). The cleanup margin R is scale-invariant, so R itself is unchanged. The
  replay tag is not: the SWR re-tag is h_rep = R |inc| (`webapp/sleep_replay_capture.py` `_epoch`), taken from the
  rescaled increment, so every replay tag carries s. The capture drive gamma p h (h the larger of the decayed write tag
  and the decayed replay tag, both with TAU_TAG_H = 1.5 h) then carries s wherever the replay tag is the larger term:
  a block told more than about 1.5 h x ln(1 / (R s)) before its epoch (about 2 h at R = 0.3), which, for any block
  reading R above about 0.08, is every target of this battery's day-1 blocks (told at least 4 h before the night), and
  every block on every later night. It is asymmetric: the important fact loses
  more. It is also slot-dependent: the sensed magnitude includes the (seed, block) baseline draw, so s is a second
  slot-dependent input to capture, which an attribution of R0 alone cannot see. Also magnitude-dependent: r3's load
  read (dW / W) and the recall's cross-block cue-match competition.
- *What the fi records show.* The bookkeeping is reproduced exactly: at the first recall a block's increment
  magnitude is its write tag x s x the night's r3 scale. Per seed, weak telling (`fiv_lr`) then salient telling
  (`fis_lr`): write tag `tag0_mean`, increment `inc_mag` at the first recall, r3 `shy_scale`, and s derived from them
  (research/findings/raw/_sleep_forgetting_interference/seed*/fiv_lr.json,
  research/findings/raw/_sleep_forgetting_interference/seed*/fis_lr.json):

| seed | weak: tag0 | weak: inc | weak: r3 | weak: s | salient: tag0 | salient: inc | salient: r3 | salient: s | s ratio S:N |
|---|---|---|---|---|---|---|---|---|---|
| 42 | 1.225950 | 0.989153 | 0.893078 | 0.90 | 2.463246 | 1.662990 | 0.858694 | 0.79 | 0.87 |
| 43 | 1.000000 | 0.821410 | 0.881904 | 0.93 | 2.202763 | 1.512364 | 0.848213 | 0.81 | 0.87 |
| 44 | 1.000000 | 0.840662 | 0.901931 | 0.93 | 2.001480 | 1.458259 | 0.883170 | 0.82 | 0.89 |
| 100 | 1.000000 | 0.847601 | 0.902021 | 0.94 | 2.330044 | 1.603520 | 0.860459 | 0.80 | 0.85 |
| 101 | 1.000000 | 0.802122 | 0.863550 | 0.93 | 2.122842 | 1.428693 | 0.829802 | 0.81 | 0.87 |
| 102 | 1.045387 | 0.899379 | 0.912721 | 0.94 | 1.965960 | 1.467779 | 0.894709 | 0.83 | 0.89 |

  The same s is recovered from the replay tag (`tag_rep_mean` / (R x tag0)) on every row. So with the ledger on the
  pass pulls even gain-1 N facts down (s 0.90-0.94 on all six seeds), pulls the salient fact harder (0.79-0.83), and
  shrinks the S:N replay-tag ratio by 11-15 % (mean 13 %) before the first SWR epoch. <!--derived-->
  In fi's first night the write tag, told five minutes earlier, is still the larger term, so first-night capture
  there did not carry s; in this battery it does (above).
- *What it predicts for the gates in ledger-on arms* (the pair baseline, and every step until Step 2(c) replaces
  the pass). It works against WM1 for S (lower capture drive and replay tag for S relative to N at every pass) and
  against WM6 wherever the burst competition's drive is read off the block's weights (the Step-4 prereg states whether
  it is; its excitability mark is not rescaled by the pass). It lowers N's replay tag and capture drive too, by less.
  Predicted direction: `n7_nohomeo` reads score(S) - score(N) and S's replay wins at least as high as `n7`.
- *In the arms.* It stays ON in every arm, because it is production and the flip candidate must hold with it. Step 0
  adds a default-off lesion knob that skips the pass (byte-identical off, asserted in the data), a per-block record of
  every pass's scale s (the composer's `_homeo_scales`, with the pass's time), and a REPORTED arm `n7_nohomeo` per group
  for both baselines and for the step under test, so its cost to WM0b, WM1 and WM6 is measured. Reading `n7_nohomeo`:
  skipping the pass removes the S:N compression AND the pull-down of N AND changes r3's load read, so a difference
  between `n7` and `n7_nohomeo` is not the compression alone; it is read with the per-block s record beside it. Step
  2(c) replaces the pass with a homeostat over units that several engrams share.

## 4. The mechanism plan, in order

Each step is brain-based: neurons, synapses and neuromodulators decide; host code only for the world (the
conversation, the test questions), the body (the sleep/wake clock) and the clock. The ledger's per-synapse state
equations (tag, PRP, late phase) stay host-integrated synaptic state, the same category as every plasticity rule in
the engine, declared; moving them onto the substrate is the pair's own backlog (review section 5, items 1, 3 and 8) and
not part of this plan. Every step is default OFF, byte-identical off (asserted in the data), with its own lesion, its
own biology binding and its own prereg committed before any run.

**Step 0 -- the instrument first.** Build the what-matters battery (section 5) as a runner family, with the graded
outcome grader; the graded encoding read after every telling (the decode of the arm's own telling, the written
magnitude, R0 and the cue-role margins, section 5.1), which saves and restores the substrate state it touches and is
checked by a read-off control (U0b); a per-block read of the late-phase fraction and of each block's reactivation
count; a per-block record of every homeostatic pass's scale s (section 3); the homeostat lesion knob; and the two
baselines: today's production default and the pair as it stands, each at the delay arms {`d1h`, `d4h`, `n1`, `n3`,
`n7`} so that every gate predicted to fail in 5.6 is scored at a delay the baseline actually runs. No brain change.
Its own gate is discriminating power: both baselines must read NO-GO on the gates predicted in 5.6. If a baseline
passes a gate predicted to fail, that gate is repaired by amendment before any mechanism result is scored. If a
baseline FAILS a gate predicted to pass, the failure is diagnosed on the record before any mechanism result is scored:
an instrument cause (grader, probe, protocol, the read's own perturbation) is fixed by amendment and the baseline
re-run; a brain cause stands as a baseline finding, the prediction is corrected by amendment, and the gate stays as
registered for every step that binds it (a prediction can be corrected, a gate never loosened). "The instrument is
part of the emulation."

**Step 0b -- attribute the encoding spread (instrument).** A composer-plus-ledger measurement, no chat: at the six
seeds, (i) the same fact written into eight block slots, (ii) eight different facts written into the same slot index
over rebuilds, (iii) the ledger's baseline on versus off (b = 0), (iv) the read repeated under eight private RNG
streams, (v) D in {128, 256} as the ensemble-size factor, (vi) the homeostatic pass applied once after the writes, as
the idle tick applies it, all at write gain 1.0 and at the salient gain. Output, per seed: the share of the variance
of R0 (and of the cue-role margins) carried by slot, content, read noise and D; the written magnitude of every block;
and the pass's scale s per block with the share of its variance carried by slot (through the baseline draw), content
and D, because s is a second slot-dependent input to capture that R0 cannot see (section 3). No verdict of its own; a
seed-7 dev run precedes the six seeds. **Decision rule for Step 1a, registered now** (exactly one of (1) and (2)
applies, and (3) adds to it): (1) if one of slot, content and read noise carries more than half of R0's variance at
write gain 1.0, its candidate ((a), (b) or (c) below) is built first; (2) otherwise the candidates of the two largest
of the three are built together, so (c) is among them whenever read noise is one of the two largest; (3) if the slot
carries more than half of s's variance, (a) is added, since it removes the drawn baseline that s senses. D is reported
as the change in R0's mean and spread from 128 to 256; a larger D enters candidate (c) only if it lifts every seed's
gain-1 R0 above the composer's separator (g = 0.15) AND the doubled store still fits the one-3090 reference
(`gates/consumer_hardware_reference`).

**Step 1a -- encoding strength set by the brain's state, not by where the fact lands** (C-0;
`encoding-strength-and-allocation`). Candidates, each brain-based:
- (a) *Slot dominates: allocation replaces the drawn baseline.* The ledger's per-block random baseline stands in for
  "strength that belongs to other memories"; the real system's version is the other memories themselves, on units
  chosen by an excitability competition at learning (Han 2007; Yiu 2014). The new block is allocated by a spiking
  competition among free trigger units driven by their excitability, and its pre-existing strength is what the store
  actually holds on those units (Step 2's compartments), so interference is earned by what has been learned, not drawn.
  If (a) is chosen, Step 1a and Step 2(a) are built together. On fresh units the normalized read R0 no longer sees
  the gain through its ratio to a drawn baseline, which is one reason WM0b reads the written magnitude, not R0.
- (b) *Content dominates: separation at the write.* The fact's composite is decorrelated from stored and vocabulary
  codes before it is written (dentate-gyrus pattern separation, Leutgeb 2007), through the composer's DG sparse index
  (`research/biology/dg-ca3-sparse-index.md`), so overlapping word codes stop sharing a readout at the store.
- (c) *Read noise dominates: a read that uses the whole ensemble.* The completed-ensemble read (Step 4's `R_c`) serves
  both the replay and the post-telling encoding read, and the ensemble size D is an operating point measured against
  the one-3090 reference.
- In every branch, (d): an encoding-state signal (acetylcholine raises afferent strength and synaptic modification,
  Hasselmo 2006) as a spiking afferent onto the write, driven by the brain's own attention populations (the
  biased-competition / GNW state), never by a host test of which turns are assertions; and Step 5's channels raising
  encoding strength. The awake rest route (post-encoding rest helps, Tambini 2010; Dewar 2012) cannot rescue a trace encoded at
  noise, because its re-induction is proportional to R (the arc family's seed-101 miss), so it is not a candidate here.
Lesion: the step's mechanism cut (the pre-step write). Gates that bind: 5.3a's Step-1a row (WM0a, WM0b for S, WM0c,
WM4). WM2a is scored and REPORTED here: before Step 2 its same-day keep of the morning N fact can come only from the
global pool's same-block capture (section 2 item 3), a pass by the defect.

**Step 1 -- graded induction over heterogeneous synapses** (C-1; `prp-competition-and-locality`). Give each managed
block's synapses a seeded per-synapse readiness drawn once at the write (primed with a probability below one half,
fixed a priori from Kramar's "fewer than half primed", never fitted to a gate seed). Readiness gates INDUCTION: only
primed synapses take the early increment and the write tag (`h0_k = rho_k |inc_k|`), so one telling potentiates part
of the trace, and unprimed synapses become ready on an hour scale (Step 3c's spacing). Capture of what was potentiated is
graded by Step 2's competition and locality (Govindarajan 2011), not by readiness. Built after Step 1a: potentiating
fewer than half the synapses lowers the encoded read, so the Step-1 prereg first checks, on a dev seed, that the recall
margin at the primed fraction clears the confidence gate at the floor write gain on the Step-1a store. Lesion:
`rho_k = 1` (today's single switch). Gates that bind: 5.3a's Step-1 row (adds WM2b-d, WM5d, WM10); WM2a REPORTED
as at Step 1a.

**Step 2 -- limited, local, competed-for PRPs, and a homeostat over shared units** (C-1; Fonseca 2004; Govindarajan
2011; Dunsmoor 2015; Turrigiano 1998). (a) Compartments: a block is allocated to the compartment whose units its
concept codes overlap most (allocation by excitability overlap, Yiu 2014), so facts about the same entity share one, and
a PRP event reaches other compartments attenuated. (b) Consumption: capture draws on the compartment's PRP in
proportion to `gamma * p * h_k`, so a strongly tagged trace spends what a weakly tagged neighbour would have used. PRP
synthesis still comes only from the spiking D1 pool. (c) The homeostat senses a compartment's shared units and scales
all of their engrams by one factor, so it regulates total drive and keeps the order of what the compartment holds
(replacing the per-engram pass, section 3). This makes behavioural tagging specific, makes salience competitive, and is
the registered NR response already named in Amendment 7. The compartment allocation rule is a declared host step until
the store has dendritic structure (`sim/dendritic_neuron.py` is the named rung). Lesions: one global, unconsumed pool;
the per-engram homeostat restored. Gates that bind: 5.3a's Step-2 row (adds WM8; WM2c now on the multi-fact day).
**Predicted regression, registered now: WM2a fails at Steps 2 and 3.** The morning N fact is about 11 h old at `d1h`
and, with TAU_EARLY_H = 1.5 h, survives only if captured; before Step 2 the global, never-consumed pool fed by the same
block's S, E and F tellings captures it (the non-specific capture of section 2 item 3), Step 2 removes that, and no
registered mechanism keeps an unrelated ordinary morning fact through the day until Step 4(c)'s awake bursts. WM10 is
at risk for the same reason (its comparator, the pair as it stands, keeps what the global pool captured). Both are
scored and REPORTED at Steps 2 and 3 and bind again from Step 4. REPORTED: WM1 for S against `n7_nohomeo` (whether (c)
removed the per-engram pass's S compression, section 3), WM6b.

**Step 3 -- one trace per fact: repetition and recall strengthen it** (C-4;
`repetition-retrieval-strengthen-same-trace`). (a) A re-telling is recognized by the composer's own cued read
(`_find_cued_block`); a predicted re-statement, now "restabilize", re-induces early LTP and re-sets the tag on that
block through the ledger (the awake-replay rule `e <- e + R (1 - e)`, reused) instead of appending a copy. That cued
read is the spiking K-way sequencer only when the composer's integrated loop is on, and production has it OFF
(`webapp/server.py` `_INTEGRATED_LOOP_DEFAULT_ON = False`; `BrainConversationalAgent` defaults
`integrated_loop=False`): then `_seq_block` runs the host first-match loop over the decoded blocks (an agent and
action string compare), and "this is a re-telling" would be a host decision. So Step 3's arms, and the arms of every
later step, run with `BRAIN_INTEGRATED_LOOP=1`, registered in the Step-3 prereg, and the flip candidate and leg (d)
carry it. The flip then includes turning the integrated loop on by default, which has its own open owner decision (the
sequencer's characterized over-abstention at the tiny-demo's small vocabulary, noted at that constant in
`webapp/server.py`); the Step-3 prereg's seed-7 dev run measures that over-abstention on this battery's vocabulary
before any six-seed set, and from Step 3 on WM10's comparator (the pair as it stands) runs with the same selection
path. The prediction error that separates a re-statement from a correction is the spike-resident matched-filter read
(`_patient_prediction_error`, persistent loop on by default); its threshold compare and calibration are host
arithmetic on that read, declared like R. Strengthening acts on the same synapses: an earlier finding here showed
re-encoding a second copy is a random walk in binding quality
(research/findings/2026-05-31-P4-multihop-trace-bimodality-DIAGNOSED-per-pair-per-seed-recall-strength-lottery-actionable.md).
(b) The brain's own retrieval event strengthens what it retrieved: when its cued read (the sequencer's selection, as
in (a)) selected a managed block and the reply did not abstain, that block gets the same re-induction, WHETHER OR NOT
THE ANSWER IS RIGHT. No host grader enters: retrieval practice helps without feedback (Roediger & Karpicke 2006). The
risk side is real and is why WM4 must catch any confabulation that results: wrong answers met at a test can become
later answers (reading more multiple-choice lures raised lure intrusions on a final test, Roediger & Marsh 2005; this
shows exposure to wrong alternatives, not directly that a produced wrong answer is strengthened). The re-induction is
applied after the reply, so the read never writes during itself (the systems-consolidation protocol rule is kept). (c)
Spacing: synapses left unprimed by an episode become ready on an hour scale (Kramar's 1-h rule), so a repeat after an
hour recruits them and a massed repeat does not. The host `kb` list must not be the dedupe key. Lesion L-RECON: the
re-induction edge cut (and dedupe must still hold). Gates that bind: 5.3a's Step-3 row (adds WM5a-c, WM1 and WM3 for
Rsp and T, WM7 L-RECON); WM2a and WM10 REPORTED as at Step 2.

**Step 4 -- replay that spends effort where it matters** (C-2; `prioritized-replay-triage`). (a) An importance mark
carried by excitability: at encoding the neuromodulatory mix at the telling (DA now; the Step-5 channels later) raises
the intrinsic excitability of the fact's trigger units, a slow CREB-like variable that decays within a day (the
instruction tag decays faster than the item, Stickgold & Walker 2013). (b) Each SWR burst is a population event in
which the managed triggers compete through lateral inhibition (the WTA the completion lane names as its next rung);
initiation is biased by excitability and by need (a weak trace has more headroom, Schapiro 2018), stochastic under
the arm's seed; the reactivation read is the completed ensemble (the awake-completion branch's `R_c`), not the min
margin. (c) Several bursts per epoch, one epoch per NREM cycle (four or five a night), and awake bursts in rest pauses
whose winners bias the night (Yang 2024). (d) The brake: r3 renormalization, Step 2's consumed PRPs and Step 1's
graded induction; the five-cycle fake-substrate runaway that forced one epoch per night is re-run first and must not
recur. (e) The replay-to-DA map: the Step-4 prereg replaces the host map `tonic + (DA_SWR_FULL - tonic) * min(1, sum R)`
with the spiking SNc's own output during the burst; if that cannot be built inside the step, the map is declared and a
registered arm pins the burst's DA to tonic while the re-tag stays, so R's two entries (tag and PRP supply) are
measured apart. Lesion L-PRIO-replay: uniform selection (every trigger equally likely). Gates that bind: 5.3a's
Step-4 row (adds WM6, WM1 and WM3 for S, WM7 L-DA and L-PRIO; WM2c now under several epochs), and WM2a and WM10 bind
again: the awake bursts of (c), in the rest pauses and quiet wakefulness after each block, are the registered same-day
maintenance of an ordinary fact.

**Step 5 -- the other importance channels** (C-3; `importance-tagging-at-encoding`). Each is a spiking afferent onto
an existing population with its own lesion; none is a host importance score or a keyword test. Each raises encoding
strength as well as the tag and the Step-4 excitability mark.
- 5a arousal: the affect organ's arousal population drives a BLA-like population whose noradrenergic output (the
  beta-adrenergic route of McGaugh 2004 and Cahill 1994) acts on the units of the currently most active trace -- a
  local gain with suppression of the rest (GANE). L-NE cuts that edge, the model's analogue of propranolol. The LC
  dopamine co-release of Takeuchi 2016 is a novelty route: it is part of cue S, an afferent onto the existing D1 pool,
  and L-DA (not L-NE) cuts it. Cue E.
- 5b expected use: "remember this" is understood by the language route (not a regex; the reward-value afferent's regex
  gate is the cautionary case) and latches the prospective-memory intention assembly (on by default), bound by its
  Hebbian edge to the fact's trigger; the latch raises that trace's encoding strength and excitability mark and
  co-activates with it in replay (Wilhelm 2011). "Never mind, forget that" is the complementary edge. Lesion L-REL.
  Cue F.
- 5c topic and goal: a spiking projection from the referent attractor (the common-ground / WM-focus referent units)
  onto the trigger units of the fact being written; when the referent's sustained firing coincides with the telling,
  the coincidence raises the trace's encoding strength and mark through a Hebbian term at the write. There is no host
  overlap score. Lesion L-TOPIC cuts the projection. Cue G.
- 5d value: a signed outcome (praise, correction) needs a signed value afferent that does not exist yet; named, not
  built here.
Gates that bind: 5.3a's Step-5 row (WM0b, WM1, WM3, WM7 for each channel built).

**Step 6 -- gist and detail on separable traces** (C-4; `gist-detail-graded-forgetting`). The core predicate and a
peripheral detail are stored on separate synapse sets (separate managed blocks linked by the shared agent code), each
with its own tag, capture and replay read (no min over roles). The Step-5a local gain decides which component a
salient moment favours. The gist that survives a lost detail is carried by the episodic organ's topic familiarity and
the common-ground referent, which get a decay of their own (Amendment 7 predicts that the episodic organ, not managed
by the ledger, keeps every topic it formed; its `wd_epi` arm will measure it). The within-fact case is probed with the
composer's attribute role (WM9c). The reply for "familiar but not recalled" is a functional read-out (section 8). Gates
that bind: 5.3a's Step-6 row (WM9a, WM9c, WM3 peripheral for F).

**Step 7 -- replay-written transfer to a slow cortical store** (`gist-detail-graded-forgetting`; CLS). Replay
interleaves captured, important facts into a slow cortical store (the CA3 superposed-fact attractor or a slow cortical
Hebbian store), fewer replays needed when the fact's concepts already have many stored associates (schema, Tse 2007);
`promote_buffer_to_ltm()` stops being a host hook. The word "consolidation" becomes available only when a source
lesion (the composer block removed) shows the cortical trace answers. Lesion L-SCHEMA. Gates that bind: 5.3a's Step-7
row (WM0b, WM1, WM3, WM7 for K) and a source-lesion gate in its own prereg. This is the "grows" part of the owner's
ruling.

**Step 8 -- regulated forgetting of what is marked unneeded** (Hardt 2013; Berry 2012). A dopamine-dependent
forgetting drive during sleep aimed by a "forget" mark (Step 5b's complementary edge) and by staleness (a superseded
fact; reconsolidation already rewrites on a prediction error). Last, and optional until Steps 1-7 hold.

**Why this order.** The instruments come first (Steps 0 and 0b, built in parallel). Step 1a comes next because every
later step reads a trace, and a trace encoded at noise on some seeds makes every later gate partly a lottery. Step 1
follows 1a (it lowers the per-episode encoded read). Steps 1-2 come before Step 5 because feeding more importance
channels into a one-switch store with a bottomless shared PRP pool only moves the cliff: under the pair the
salient/neutral contrast has already vanished by the first morning. Step 3 is early because it is the RAG-like defect
the owner named and it reuses the existing reconsolidation read; it touches code disjoint from Steps 1a and 1 and can be
built alongside them. Step 4 needs Steps 1-2 as its brake. Step 6 needs Step 1 (graded traces) and Step 5a (local
priority). Step 7 needs Step 4. 5a, 5b and 5c are independent of each other.

## 5. The registered test battery: what-matters gates

This section fixes the battery's design, gates and UNDEFINED rules now, before any code. Step 0's prereg pins the
code, the content lists and the constants, and may tighten but never loosen what is registered here. A gate below is
written so that it can FAIL, and 5.6 says which gates each baseline is predicted to fail.

### 5.1 Protocol

Two conversation groups keep the managed-block count within the composer's capacity (the arms raise `k_max` through
the existing override to at least the block count; P0 below).

| group | cues (facts per cue) | per-fact structure | other tellings |
|---|---|---|---|
| `wa` salience | N neutral (2), Nw weak (2: the fact said last of several turns about its own words, the fi pattern), S surprise, the `datc` news frame (2), E told inside the user's emotional disclosure (2), F preceded by "please remember this" (2), BT-rel, BT-rel-N and BT-unrel (1 each) | cued core fact + a plain aside about the same agent in the next turn (the peripheral detail); Nw and BT facts core only | 2 unrelated plain facts per later day (interference), never probed |
| `wb` use | N (2), Rsp spaced (2), Rms massed (2), T asked once 10 min after the telling (2), G about the conversation's current topic referent (2), K about an entity with 3 prior facts told earlier that day (2) | as above | the 3 schema-prior facts (core only); interference as above |

- **Day 1 (virtual wall clock, the Amendment-7 seam):** target tellings in a morning block (09:00-10:30) and an
  evening block (18:30-20:00); each cue has one fact in each block, so cues are balanced for time before sleep. A few
  chit-chat turns surround each telling. Between turns the body is AWAKE by the environment's awake mark
  (`da_tag_capture_chat.mark_awake`, the r2 world step, extended to the virtual wall clock in Step 0), so the day's
  idle stretches are quiet wakefulness (awake bursts from Step 4 on), not sleep; two registered 20-min rest pauses
  follow the blocks. The night starts at 24:00 by the body clock. Mornings at 08:00; interference facts at 10:00 and
  16:00 on each later day. The production trigger (any idle of 5 min counts as sleep) is not used here; its effect is
  the `pp` day's and D8's question (section 6).
- **Behavioural tagging (wa):** BT-rel is told 20 min before the second S fact and shares its agent; BT-rel-N is told
  20 min before the second N fact and shares ITS agent (the control for plain agent-sharing crosstalk); BT-unrel is
  told in the turn after BT-rel, about an unrelated agent.
- **Spacing, the lag design (wb):** each Rsp fact's SECOND telling is at its slot inside the registered block, and
  its first telling 2 h earlier (07:00-08:30 for the morning block, 16:30-18:00 for the evening block); each Rms fact is
  told twice within the same minute, AT THE CLOCK TIME of the matching Rsp fact's second telling. The last exposure of
  both is at the same time and inside a block, so the retention interval from the last exposure to every probe is equal
  (Cepeda 2006: the inter-study interval and the retention interval act jointly, so only the interval between tellings
  may differ), and every telling of every cue is over before `d1h` (21:00), which probes each fact at least an hour
  after its last exposure.
- **Content:** a pool of content triples per group rotated across cue slots by seed (a Latin-square shift), so
  vocabulary crosstalk is not confounded with cue; no (agent, action) cue of a target collides with a build-time or
  LTM fact, checked offline against the store's fact list before the run (test construction, not the brain).
- **The encoding read (P1, graded), in every arm:** right after each telling of each target and aside, (i) the
  composer's own non-writing decode of the new block(s): told roles returned or not; (ii) the written magnitude W0 of
  each new block, taken before any homeostatic pass: the ledger's write tag `tag0` (mean |inc|) with the ledger on,
  and with it off the block's sensed readout A_i / A* (`_measure_block_readout`, the homeostat's own read, linear in
  the write); A_i / A* is REPORTED in both; (iii) R0, the sleep route's own reactivation read of each new block; (iv)
  the cue-role margins of the decode. The reads run under a private RNG stream in every arm, so the shared prefix
  stays identical (U0 checks it). They set the RF bridge's weights and resonate its units mid-day
  (`_block_role_scores`, `_measure_block_readout`), so the read saves and restores every substrate state it touches
  (the bridge's complex weights, membrane state and RNG streams) and asserts, on the data (a hash compare), that the
  substrate state after the read is byte-identical to the state before it; and each config and group runs a read-off
  control `n1_noread` (U0b), because U0 cannot see a perturbation that `n1` and `n1_b` share. They read the arm's OWN
  telling; a precondition read on a sibling script is not allowed (section 2a, item 5). Their outcomes are the
  BRAIN's: a core block whose decode does not return the told roles is scored lost at every delay and fails WM0a; an
  aside that does not encode is scored not recalled at every delay (a faithful arousal mechanism may do exactly that
  to a background detail). A behavioural `imm` arm (a chat probe right after each telling; it diverges after its first
  probe) is REPORTED beside it.
- **Probes at the delay, once per arm:** for each target fact, in fixed order, the central question (the core's
  patient), the peripheral question (the aside's patient), then the referential probe ("you mentioned the <agent>",
  read on the episodic organ's `in_memory`). Nw and BT facts: central only.
- **Delay arms** (each arm runs the shared prefix and is probed only at its delay, so earlier probes cannot act as
  retrieval practice): `d1h` (21:00, day 1), `d4h` (24:00, awake, no sleep yet), `n1`, `n3`, `n7` (08:00 after 1, 3
  and 7 nights).
- **Other arms:** `n1_b` (G0 null rebuild of `n1`); lesion arms at `n7`, each lesion mapped to the cue it must
  remove: wa -- L-DA -> S (the waking-only DA lesion with the SWR edge spared, Amendment 7's knob), L-NE -> E, L-REL ->
  F; wb -- L-RECON -> Rsp and T, L-TOPIC -> G, L-SCHEMA -> K; both groups -- L-PRIO, which cuts every importance edge,
  the replay bias and the re-induction on repetition and recall, while keeping the machinery (one trace per fact,
  graded induction, consumed local PRPs, the r3 brake). REPORTED: `rp` (probed at every delay in one arm: the testing
  effect over the whole protocol), `n7_vac` (no interference: retention in a vacuum, Wixted's case), `n7_nohomeo`
  (the homeostatic pass skipped, section 3). Instrument control: `n1_noread` (the encoding read skipped, every other
  step equal; U0b).
- **Env:** production defaults plus the flags of every step in the stack of the step under test (5.3a), with
  `BRAIN_INTEGRATED_LOOP=1` from Step 3 on (Step 3(a)); the LTM tier OFF in the gated arms (a declared
  deviation, the same one the chat-wire family's LTM-off GO made: an LTM-on build exceeds the 15 GB pool nodes, no gate
  reads the LTM tier, and target content is checked offline against it; LTM on is flip leg D7's job, section 6);
  `BRAIN_EPISODIC_STORE=1` so the episodic organ writes; the seed through `BRAIN_CHAT_SEED` to `cfg.seed` (never
  `actual_seed_used`); the D1 reader seeded with the arm's seed and gamma calibrated a priori per seed (review I-4), the
  production-seed reader REPORTED beside it.

### 5.2 Grading

Per probe: correct, abstain, guess (flagged by the brain as a guess), confab (a wrong answer not flagged), undefined.
"Not recalled" is abstain or guess. Per target fact at a delay:

| outcome | condition | score |
|---|---|---|
| kept | central correct and peripheral correct | 2 |
| gist | central correct and peripheral not recalled; or central not recalled, referential probe familiar, and the reply discloses familiarity without content | 1 |
| lost | nothing recalled, not familiar; or the core did not encode (WM0a) | 0 |
| confab | any unflagged wrong answer on any probe of the fact | 0, and WM4 fails |
| inversion | peripheral correct while central not recalled | counted for WM9a |

`score(X)` is the mean over cue X's facts in the group; the retention curve of a fact is its score over the delay arms.
For Nw and BT facts (core only) kept = 2 on a correct central probe.

### 5.3 Gates (per seed and group)

| gate | passes only if | first binds at (5.3a) |
|---|---|---|
| WM0 what is told is learned, by importance, not by lottery | (a) every target core block's post-telling decode returns the told roles, in every gated arm; (b) for each cue whose importance is present at the telling (wa: S, E, F; wb: G, K), mean written magnitude W0 (5.1) over its core blocks > mean W0 over N's, strictly (at the floor write gain two cues tie, and a tie fails); (c) every N and Nw core block's R0 is at or above the composer's own clean/noise separator g = 0.15 (the confidence gate's constant: a told ordinary fact is written where the brain's own reads can tell it from noise). (b) reads the magnitude because R0 is a normalized margin, (peak - runner_up) / peak (`OneBrainComposer._margin`), which with the ledger off does not depend on the write gain at all and with it on sees the gain only through its ratio to the drawn baseline. REPORTED: every block's W0, A_i / A*, R0 and cue-role margins, the scale s of each homeostatic pass, R0 per cue, the step in outcome against R0 | (a) step 1a (predicted to hold in both baselines); (b) S step 1a, E/F/G step 5, K step 7; (c) step 1a |
| WM1 importance order | for each cue X (wa: S, E, F; wb: Rsp, T, G, K): (a) score(X) >= score(N) at every delay, and (b) score(X) > score(N) at `n7` | per cue: S step 4, Rsp/T step 3, E/F/G step 5, K step 7 |
| WM2 ordinary facts fade gradually | (a) not at once: both N facts kept at `d1h` and mean N score >= 1 at `n1`; (b) not never: mean N score at `n7` below its `d1h` value and at least one peripheral detail in the group lost at `n7`; (c) no resurrection: no fact scored 0 at one delay scores above 0 at a later delay; (d) the group's loss events fall in at least two different delay intervals. Nw facts are REPORTED beside N (a-d). REPORTED (e): whether each N fact's familiarity reaches "lost" by `n7` (the "eventually gone" half; seven nights may be too few to require it) | (b-d) step 1; (a) step 4 (REPORTED at steps 1a-3, predicted to fail at steps 2-3) |
| WM3 important kept | wa: every F and E fact's central correct at `n7`, every S fact's central correct at `n3`, every F fact's peripheral correct at `n3`; wb: every G, K, T and Rsp fact's central correct at `n3` | per cue as WM1; F peripheral step 6 |
| WM4 no confabulation | zero unflagged wrong answers on every probe of every gated arm; every familiarity-without-content reply names no content | step 1a, and every step after |
| WM5 not RAG-like storage | (a) after each re-telling and each retrieval the fact has exactly one managed block (the composer's own count); (b) with the lag design of 5.1, score(Rsp) >= score(Rms) at `n3` and `n7`, and higher summed over delays; (c) score(T) > score(N) at `n7`; (d) at `n3` at least three target blocks per group express a fraction of their written increment strictly between 0.1 and 0.9 (the ledger's weight factor `e + z(1 - e)` averaged over the block's synapses; 1 by construction with no ledger). (d) is an installation check: with readiness below one half fixed a priori, every captured block expresses about the primed fraction, which lies inside (0.1, 0.9) by construction at Step 1, so (d) shows that graded induction is installed, not that retention is graded by importance. REPORTED (e): at `n3`, each cue's mean expressed fraction against N's (the importance order of the fraction; WM1 carries the binding order) | (a-c) step 3; (d) step 1 |
| WM6 effort follows importance | the mean number of replay reactivations won by cued target blocks exceeds that of N blocks over the protocol; REPORTED (6b): PRP consumed per block | step 4 (6b REPORTED from step 2) |
| WM7 lesions remove the prioritization | at `n7`, under each channel lesion its mapped cue's advantage is gone (score(X) <= score(N): wa L-DA -> S, L-NE -> E, L-REL -> F; wb L-RECON -> Rsp and T, L-TOPIC -> G, L-SCHEMA -> K) while at least one other cue's advantage remains; under L-PRIO every cue's advantage is gone while WM2a and WM2b still hold (importance-blind, neither amnesic nor keeping everything) | L-RECON step 3; L-DA and L-PRIO step 4; L-NE, L-REL, L-TOPIC step 5; L-SCHEMA step 7 |
| WM8 behavioural tagging is specific | at `n3`: score(BT-rel) > score(BT-unrel) AND score(BT-rel) > score(BT-rel-N) | step 2 |
| WM9 the gist survives the detail | (a) no inversion for any cued fact at any delay (N inversions REPORTED); (b) REPORTED: at `n3` or `n7` at least one cued fact per group reads gist (the across-turn aside is a separate block, so the pair can pass this without Step 6; it is evidence of nothing on its own); (c) within-fact, in Step 6's arms with an attributed patient ("the cat chases the red ball") written into the composer's attribute role: at `n3` or `n7` at least one cued fact per group reads core correct and attribute not recalled. The role itself is bound and read by default (`BrainConversationalAgent` defaults `enable_attributed=True` and builds the onebrain composer with it, its "default OFF" comments are stale; the fi records decode an attribute role, e.g. seed 43 `fir_recall3`, "fish" at margin 0.08, a noise read of an attribute never told). What production chat lacks is the route that parses an attributed sentence into it: chat acquisition (`_maybe_acquire`) stores a three-word SVO, and the neural attributed parser (`hear_attributed`) has no chat caller. So Step 6's arms route the attributed telling through `hear_attributed`, a declared deviation unless Step 6 wires that route into chat acquisition | (a, c) step 6 |
| WM10 prioritizing does not mean forgetting more | summed target score at `n7` >= the pair-as-is baseline's at the same seed (the Step-0 row, re-run if the protocol changes, and from Step 3 on with the same cue-match selection path, Step 3(a)); REPORTED against today's default | step 1 (REPORTED at steps 2-3, binding again from step 4) |

**Verdicts.** A group reads GO at a seed only if every gate in the binding set of the step under test (5.3a) holds,
and NO-GO if one fails; a seed where any U-rule fires reads UNDEFINED, never GO and never NO-GO. A step's family
verdict is, in this order: INCOMPLETE if a seed is missing; UNDEFINED if any seed is still UNDEFINED after its re-run
budget (the defined seeds' verdicts are then REPORTED beside it); GO iff all six seeds read GO; NO-GO otherwise.
Re-run budget: one re-run per seed for U4 (run integrity) only, at the same pinned revision; U0, U0b, U1, U2, U3 and
U5 are instrument or input failures that need a fix and an amendment before any re-run. The one-sided exact sign-flip
p over seeds is reported for WM0b per cue, WM1b per cue, WM5b and WM8. The flip candidate is held to the binding set
of the highest step it contains (5.3a; section 6).

### 5.3a Which gates bind at which step (registered now)

Each step is tested on its stack: the pair plus every step before it in the order 1a, 1, 2, 3, 4, 5 (5a, 5b and 5c in
any order), 6, 7, with their flags set (a step may be dev-run on a shorter stack; that is REPORTED only). The binding
set is cumulative: at a step, every gate in its row and in every row above binds, except a gate that the step's row
lists as REPORTED. Every other gate is scored at every step and REPORTED. Step 0's gate is discriminating power (5.6);
Step 0b has no verdict.

| step under test | adds to the binding set | scored and REPORTED only, at this step |
|---|---|---|
| 1a | WM0a; WM0b for S; WM0c; WM4 | WM2a: before Step 2 the morning N fact reaches `d1h` only through the global pool's same-block capture, so a pass is the defect's |
| 1 | WM2b, WM2c, WM2d; WM5d; WM10 | WM2a (as at 1a); WM5e |
| 2 | WM8 | WM2a, predicted to FAIL (the registered Step-2 regression, Step 2); WM10, at risk for the same cause; WM1 for S against `n7_nohomeo`; WM6b |
| 3 | WM5a, WM5b, WM5c; WM1 and WM3 for Rsp and T; WM7 L-RECON | WM2a and WM10 (as at 2) |
| 4 | WM2a and WM10 (binding again: Step 4(c) supplies the same-day maintenance); WM6; WM1 and WM3 for S; WM7 L-DA and L-PRIO | -- |
| 5 | for each channel built: WM0b, WM1, WM3 and WM7 for E (L-NE), F (L-REL), G (L-TOPIC) | -- |
| 6 | WM9a, WM9c; WM3 for F's peripheral detail | -- |
| 7 | WM0b, WM1, WM3 and WM7 for K (L-SCHEMA); the source-lesion gate of its own prereg | -- |

### 5.4 UNDEFINED rules (never scored as a pass or as zero)

- **U0 G0:** `n1` and `n1_b` differ in any outcome, triple, abstain flag, familiarity read, encoding read, ledger state,
  sleep record or final store.
- **U0b the instrument does not act on the brain:** the encoding read's state assertion fails at any telling (5.1), or
  `n1` and `n1_noread` differ in any outcome, triple, abstain flag, familiarity read, ledger state, sleep record or
  final store.
- **U1 P0 input and capacity:** a telling did not store the registered number of blocks; the managed-block count
  exceeded `k_max`; a scripted re-telling, retrieval probe or rest pause did not happen; a world step failed.
- **U2 input reached the store:** a target core telling does not decode in the ledger-off baseline arm of the same seed
  and group (Step 0's production-default row runs the same prefix): the telling itself failed (parser, route), which is
  an input failure. In every other arm a telling that fails to decode is the brain's outcome (WM0a), not UNDEFINED.
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
counts, encoding reads, late-phase fractions, replay wins, lesions). They cannot say how a model dose maps onto a human
day, and two facts per cue per seed make each per-seed comparison coarse; the six-seed sign test is the evidence.
Probes are single questions per component, so partial recall inside a component is not graded. The episodic read is
topic-level familiarity only. WM0c's separator is the composer's own constant (the confidence gate's), not a biological
number; it is used because the brain's own reads use it, and R0 is host arithmetic on a substrate read, declared like R.

### 5.6 Baselines the battery must fail (discriminating power, Step 0)

Both baselines run {`d1h`, `d4h`, `n1`, `n3`, `n7`} plus `n1_b`, `n1_noread` and `n7_nohomeo` per group, so every
prediction below is scored at a delay the baseline runs. A baseline that fails a gate predicted to pass is handled by
Step 0's rule (diagnose; an instrument cause is fixed, a brain cause corrects the prediction, never the gate).

| config | predicted to fail | predicted to pass |
|---|---|---|
| today's production default (ledger off) | WM2b (nothing fades), WM5a (a re-telling appends), WM5d (no graded traces), WM6 (no replay selection), WM1b for every cue (everything kept, nothing ordered), WM8 (BT-rel, BT-rel-N and BT-unrel all kept, so no strict order), WM9b REPORTED as absent (nothing lost, no gist state), WM0b for E, F, G, K (no channel: at the floor write gain their W0 ties N's, and a tie fails) | WM2a, WM3 (trivially: everything kept), WM4, WM9a (trivially), WM0a, WM0b for S on the written magnitude (the DA write gain scales the write and W0 is linear in it; fi's salient telling wrote at about twice the weak one's magnitude on every seed). The previous revision's R0-based WM0b-S prediction is withdrawn: with the ledger off R0 does not depend on the write gain. WM0c not predicted (no ledger baseline on this store; REPORTED) |
| the pair as it stands (DA capture + sleep route) | WM5a, WM5d (fractions only 0 or 1), WM6 (every block reactivated once a night), WM8 (the global pool captures all three BT facts alike), WM1b for E, F, G, K, T; WM0b for E, F, G, K (as for the default); WM2a in wb, likely: wb has no S, E or F telling, a plain telling's own D1 read was 0 at every turn of the fi weak and neutral scripts on five seeds (one turn read 0.11 on seed 42), so nothing captures the morning N fact before `d1h`. WM2a in wa is NOT predicted: the same block's S, E and F tellings feed the global, never-consumed pool, which can capture both N facts (the fake design day of Amendment 7 captured every fact told within the hour after a salient telling), so a wa pass would be the defect's, and is REPORTED as such; WM0c on at least one seed (at the floor write the fi weak telling read 0.04 on seed 101, below the separator; seed 43's 0.17 was above it, so the prediction rests on the six N and Nw core tellings per seed (36 over the family) meeting at least one noise-level read, and a pass here is repaired by amendment per Step 0); WM9a on at least one seed (core and aside are separate blocks kept or lost independently on their own R) | WM4, WM0a, WM0b for S on the written magnitude (the DA write gain sets the write tag `tag0`, which fi's salient telling put at about twice the weak one's on every seed) |

### 5.7 What it takes to run

The only measured basis is the fi family: nine 7-night arms of up to 20 tellings in 26-36 min each on numpy, one process
per arm, about 0.7 GB per process, with the LTM tier off and no episodic write. This battery differs in two measured
ways that change the plan (review): (1) the episodic write, which the server documents at about 510 s per topic on
numpy (`webapp/server.py` `_episodic_store_ok`), and an arm here has roughly 25-40 topics, so on numpy it adds about
3.5-5.5 h per arm; (2) an LTM-on build does not fit a 15 GB pool node (the chat-wire prereg measured over 11 GB at build;
the LTM-on GO ran on AWS `r7i.4xlarge` with a 48 GB job cap). The registered placement:

- **Gated arms (LTM off, episodic write on).** On the reference 3090 through `tools/gpu_queue.sh` when the GPU is free
  (cupy: the episodic write takes seconds per topic there, and the 3090 is the consumer reference), or on numpy on the
  mini-PC pool (an LTM-off build fits its nodes; `tools/sweep_pool.sh`) with AWS on-demand CPU inside the owner's daily
  cap as overflow, whichever the seed-7 dev smoke measures cheaper in wall time. Never on the pool nodes with LTM on.
- **Flip leg D7 (LTM on).** AWS `r7i.4xlarge`, one seed per instance job, the memory cap set from a measured seed-42
  peak before the other five are queued (the chat-wire LTM-on convention).
- **Estimates, to be replaced by the smoke's measured per-arm wall time and peak RSS before any six-seed set is
  queued.** Step 0: 2 configs x 8 arms (`d1h`, `d4h`, `n1`, `n3`, `n7`, `n1_b`, `n1_noread`, `n7_nohomeo`) x 2 groups
  = 32 arms per seed, 192 for six seeds. On numpy the episodic write dominates: about 25 topics on day 1 plus up to 14
  interference topics later, at about 510 s each, puts an arm at roughly 3.5-5.5 h, so about 110-175 CPU-h per seed
  and 670-1060 CPU-h for six seeds, about 50-75 instance-hours of one `r7i.4xlarge` at 14 concurrent processes (the
  budget guard converts that to spend and holds the owner's daily cap). On the 3090, where the episodic write takes
  seconds, an arm should cost near the fi basis (20-60 min by delay): about 17-29 GPU-h per seed and 100-170 GPU-h for
  six seeds, one brain process at a time. Step 0b is small (composer-plus-ledger builds, no chat), well under 10 CPU-h
  on local cores under `tools/memcap.sh`. The full battery after Step 4 has about twice Step 0's arm count and longer
  nights (several epochs). <!--derived-->
- A seed-7 dev smoke precedes every six-seed set; a full-brain snapshot fork at the branch points (the GNW fork
  instrument generalized) could cut the shared-prefix cost but needs its own fork-equals-rerun check first.
Nothing is queued by this document.

## 6. How this changes the pair's flip criteria (legs b-d)

The flip candidate is no longer the pair alone but the pair plus Steps 0, 0b, 1a, 1, 2, 3 and 4 at least, and the pair stays default
OFF until then (the owner's "wait on fix").

- **Leg (b), verify-go review.** B1 (record corrections) is unchanged and still required. B2 (the owner's decision on
  the measured forgetting) is answered in principle by the ruling: losing an ordinary fact is acceptable when the loss
  follows importance, so the 0/6 loss of a neutral fact told 4 h before sleep is not by itself a blocker. What blocks
  is that the loss is importance-blind, and that encoding is a lottery at the floor write. B2 becomes a registered
  criterion: every gate in the binding set of the candidate's highest step (5.3a) reads GO 6/6 in the battery (wall
  clock, LTM off as declared in 5.1) and again in D7 (production defaults, LTM on). For the minimum candidate (pair +
  Steps 0, 0b, 1a, 1, 2, 3, 4) that is the cumulative binding set of 5.3a's Step-4 row: WM0a, WM0b for S, WM0c, WM1
  and WM3 for S, Rsp and T, WM2a-d, WM4, WM5a-d, WM6, WM7 (L-DA, L-RECON, L-PRIO), WM8 and WM10; E, F, G and K join
  WM0b, WM1, WM3 and WM7 with Steps 5 and 7, and WM9a and WM9c with Step 6. The candidate carries
  `BRAIN_INTEGRATED_LOOP=1` (Step 3(a)), so the flip includes the integrated loop's own default, which has an open
  owner decision. WM1 is a relative order, so a brain that forgot nearly everything could pass it; WM3 (important
  facts kept, absolute) and WM10 (no more forgetting than the pair as it stands) are what stop that. B3 (the `pp`
  wall-clock day) stays; its registered verdicts stand as registered, its WD2 ("ordinary kept") is read as REPORTED
  for the flip decision, and its NR gate becomes binding (it is WM2c on a production day). B4 (`sn`: salient vs
  neutral at long delay in one family, waking-only DA lesion) stays as the first registered instance of WM1 and WM7
  for the dopamine channel. B5 (the weak telling read at once and with the ledger off) stays; it is the P1 logic, now
  graded (WM0).
- **Leg (c), combined no-regression battery.** C1-C3 carry over, run at the revision that carries the prioritization
  steps. Added: every memory faculty that reads the composer store (episodic, source provenance, prospective memory,
  WM binding, common ground) keeps its load-bearing row, because Steps 1a, 3 and 6 change what a told fact writes.
- **Leg (d), production-default validation.** D1-D6 carry over. D6 (many managed facts) becomes a behaviour check, not
  only a latency check: the battery is a many-facts session. Added D7: the full battery at production defaults (no
  flags in env; the candidate's flags, the integrated loop among them, flipped on by default), LTM on, the episodic
  organ writing (cupy on the 3090, or `BRAIN_EPISODIC_STORE=1`). Added D8: a normal day with several 5-minute pauses
  (several epochs inside one early-phase window) raises no minor fact over an important one and resurrects nothing
  (WM1a and WM2c on the `pp` day).

## 7. The in-flight lanes: keep, reshape, supersede

- **fi family (`BRAIN_SLEEP_LOAD_RENORM`): harvested, NO-GO 3/6 (aggregate not committed).** It stays Step 4's brake
  and the interference half of WM2. Two readings for its finding: the re-mention protection (FI6) was carried by the
  re-mention's new block, not by the original (seed 42: the original fell to a ratio of 0.406652, like the unmentioned
  arm's), which is the append pattern Step 3 removes; and the first-morning losses on seeds 43 and 101 are an encoding
  lottery its P1 could not see, because P1 read a different telling (section 2a). Seed 100's FI6 failure is a
  retention matter. <!--derived-->
- **Awake-replay completion (`research/awake-replay-completion-r2`, dev): reshape.** Pattern completion is real
  biology and Step 4 needs it as its reactivation read, and Step 1a candidate (c) needs it as the encoding read. Its
  current target, rescuing a neutral fact told 4 h before sleep on every seed (the arc family's seed-101 miss), is no
  longer a requirement under the ruling, and completion without importance-weighted competition pushes toward keeping
  every trace that still selects its items. Recommended: stop spending levers on the seed-101 neutral rescue (section
  2a shows the same seed's weak telling is encoded near noise, which Step 1a addresses at the write); fold completion
  into Steps 1a and 4 with the gates WM0c, WM6, WM2c and WM1.
- **Production-path arms (`research/pair-production-path-arms`, `pp` / `sn` / `cu`, Amendment 7 registered before
  any run): keep, as registered.** The wall-clock seam, the cupy RNG restore, the waking-only DA lesion knob and the
  episodic-agreement arm are infrastructure this battery reuses. For the flip: WD2 REPORTED, NR binding, SN1 and SN2
  the first WM1 / WM7 instance, `cu` required for D3 and for the episodic organ's gist role. Its registered NR
  response (Fonseca 2004 PRP competition) is Step 2 here: consistent, not superseded.
- **Awake-rest replay capture (`BRAIN_AWAKE_REPLAY_CAPTURE`, arc family NO-GO, 5 of 6 seeds passed): superseded in
  role** by Step 4's awake bursts that bias the night; the code stays, it is not a flip candidate on its own.
- **r2 constant downscaling (NO-GO 0/6): superseded** by r3, already.
- **The pair's flip: on hold** until the steps and gates of section 6 hold.

## 8. The honesty boundary

Every memory self-report is a functional read-out of a measured state, never a claim of experience. Gist:
"You told me something about the dog, but I can't recall what" only when the episodic familiarity read is positive
and the recall abstains. Flagged facts: "you asked me to keep this" only when the relevance latch is on. Fading: "I'm
not sure" only when the recall margin is in the registered uncertain band; the reply says nothing about how long ago
the fact was told or how sure the brain used to be, because it has no read of a memory's age or of its past
confidence (a recency read would be its own design).
The brain never asserts a detail it cannot recall (WM4), and never says it "feels" that it remembers.

## 9. What this design does not do, and open questions

- It builds nothing, queues nothing and changes no default. Each step is its own build, binding, prereg and review.
- The ledger's host-integrated synaptic equations stay; Step 2's compartment allocation is a declared host step; the
  sleep/wake clock stays the body's host clock; R and R0 are host arithmetic on substrate reads, declared; W0 is the
  ledger's host record of the write with the ledger on and the homeostat's substrate read with it off.
- Step 3 depends on the integrated loop (the spiking cue-match sequencer), whose production default is an open owner
  decision; until it flips, the production cued read is the host first-match (Step 3(a)).
- A further constant, named here and not addressed: production starts "sleep" after any 5 min of idle
  (`continuous_engine.SLEEP_IDLE_SEC`), where a real brain runs quiet wakefulness with awake ripples and sleeps when
  sleep pressure and circadian phase say so. The battery uses the body's awake mark instead; the production effect is
  D8's measurement, and a body-clock sleep trigger is its own later design.
- Open: the readiness distribution's shape beyond "fewer than half primed" (the Step-1 prereg fixes it from the
  source, not from a gate seed); whether an excitability mark or a separate tag variable best carries importance to
  replay (Step 4 prereg decides, with a lesion either way); how "remember this" reaches the latch without a keyword
  test before the learned language route covers it (a declared scaffold needs an owner waiver); signed value (5d) has
  no afferent yet; self-relevance (facts about the conversation partner) is a plausible further cue, not yet sourced;
  which component owns the encoding spread (Step 0b answers it).
- Research record: `bash tools/before_you_build.sh "memory prioritization what matters"` and
  `bash tools/deep_research.sh` on 2026-09-25 for revision 1; for revision 2, `bash tools/before_you_build.sh` on the
  encoding defect (it surfaced the 2026-05-31 per-seed recall-strength lottery finding, read), local-corpus queries
  (Kandel ch.49 and ch.52 read), and PubMed abstracts for every new external source (listed below and in
  `research/queue/.external_searches.jsonl`).

## 10. The review of b82e7d2, closed item by item

| # | review item (severity) | closed in |
|---|---|---|
| 1 | the on-by-default homeostatic pass missing from the map; "never weakens it" wrong (MEDIUM) | section 1 (RAG-like wording), section 3 (HAS row and the interaction paragraph: what it does, ledger off, ledger on, in the arms), Step 2(c), `n7_nohomeo`; biology `homeostatic-scaling-relative-strength` |
| 2 | compute not feasible on the fi timing basis with LTM on and the numpy episodic write (MEDIUM) | 5.1 Env (LTM off in gated arms, declared) and 5.7 (placement on the 3090 or AWS, D7 on AWS, estimates re-derived, smoke-measured before queueing) |
| 3 | Step-0 arms lack `n3`, so WM5d and WM8 cannot show discriminating power (MEDIUM) | Step 0 and 5.6: baselines run `d1h`, `d4h`, `n1`, `n3`, `n7`; cost recomputed in 5.7 |
| 4 | family verdict rule contradicts the UNDEFINED rules; U2 loophole for an aside that fails to encode (MEDIUM) | 5.3 Verdicts (UNDEFINED is never NO-GO, re-run budget), 5.4 U2 (input failure only, read on the ledger-off baseline), 5.1 encoding read and 5.2 (failure to encode scored as the brain's outcome) |
| 5 | B2 leaves out WM3 and WM6; WM1 is relative only (MEDIUM) | section 6 B2 lists every gate of the candidate's steps; 5.3's last line reconciled |
| 6 | arousal bound to Takeuchi's LC dopamine route instead of the beta-adrenergic BLA route (MEDIUM) | Step 5a; section 3 arousal and novelty rows; `importance-tagging-at-encoding` note and body corrected |
| 7 | "a correct recall" could become a host oracle (MEDIUM) | Step 3b: the brain's own retrieval event, right or wrong; biology Roediger & Karpicke 2006, Roediger & Marsh 2005 |
| 8 | WM5b confounded by recency (MEDIUM) | 5.1 lag design (Rms told at Rsp's second-telling time), WM5b; biology Cepeda note |
| 9 | WM9 does not discriminate; the within-fact defect never probed (LOW-MEDIUM) | WM9a predictions in 5.6, WM9b REPORTED, WM9c within-fact with the attribute role; Step 6; `gist-detail-graded-forgetting` |
| 10 | Kramar over-read: readiness is at induction, not capture (LOW-MEDIUM) | Step 1 (readiness gates induction and the write tag; capture graded by Step 2), section 2 item 1; `prp-competition-and-locality` note corrected |
| 11 | WM8 confounded by agent-sharing crosstalk (LOW) | 5.1 BT-rel-N control; WM8 requires BT-rel over both |
| 12 | prospective-memory latch is default on (LOW) | section 3 row: HAS |
| 13 | Yang 2024 locator (LOW) | `prioritized-replay-triage` uses PMC11068097; DOI in Sources |
| 14 | Moncada & Viola's binding missing from the frontmatter (LOW) | frontmatter lists `awake-replay-tag-capture` |
| 15 | wb lesion-to-cue mapping; Step 5c read as a host overlap score; "eventually gone" never gated (LOW) | 5.1 other arms and WM7 (explicit mapping), Step 5c (a spiking projection, Hebbian coincidence at the write), WM2e REPORTED |
| 16 | Step 4 multiplies bursts with the host sum-R DA map untouched (LOW) | Step 4(e), section 2 item 2; `prioritized-replay-triage` binding |
| 17 | "that was a while ago" asserts elapsed time with no read (LOW, honesty) | section 8 |

## 10b. The review of fcf3c847d, closed item by item

The independent review of `fcf3c847d` read SOUND-WITH-ISSUES. It re-derived by hand the numbers this document marks as
derived (all six rows of the 2a table, the first-night SWR dopamine, P1's write range, the rescale counts, the fi
ratio and FI1/FI5/FI6/I3 per seed) and found them exact; the claim checker had verified only one of them, because it
checks only numbers of three or more decimals outside derived markers. The new homeostat and re-mention numbers of
this revision are written at the records' precision beside their paths, so the checker verifies them.

| # | review item (severity) | closed in |
|---|---|---|
| 1 | with the ledger on, the homeostat changes the replay tag and capture (h_rep = R \|inc\| from the rescaled increment; h0 not rescaled), asymmetrically, and is a second slot-dependent input (MEDIUM) | section 3 (ledger-on paragraph corrected; traced table: s 0.90-0.94 N, 0.79-0.83 S, S:N tag ratio down 11-15 %; predictions for WM1-S and WM6; how to read `n7_nohomeo`); Step 0 (per-block s record); Step 0b (s attributed; rule clause 3); biology `homeostatic-scaling-relative-strength` |
| 2 | seed 43's `fir_lr` misread: the recalled block is block 5 (the day-3 re-mention, R 0.36), not block 1 (R 0.008) (MEDIUM) | section 2a item 3 (corrected with the full per-block reads of seeds 43 and 101; the inference dropped, the slot confound stated) |
| 3 | WM0b read R0, a scale-invariant margin blind to the write gain with the ledger off; no rule for a baseline that fails a gate predicted to pass (MEDIUM) | 5.1 (written magnitude W0), WM0b on W0 with a strict order, R0 kept for WM0c; 5.6 (the R0-based ledger-off WM0b-S prediction withdrawn, restated on W0 with its mechanism); Step 0 (rule for an unexpected baseline failure) |
| 4 | WM2a at `d1h` needs same-day capture: before Step 2 only the defect supplies it, Steps 2-3 predictably fail it; the pair prediction ignored same-block capture; which gates bind a step was ambiguous (MEDIUM) | 5.3a (per-step cumulative binding sets); WM2a binds from Step 4 (the awake bursts); the Step-2 regression registered in Step 2, WM10 with it; 5.6 pair row (wb likely fail, wa not predicted, a pass the defect's) |
| 5 | Step 3's cued read is the host first-match with the integrated loop off, as in production (MEDIUM) | Step 3(a, b): `BRAIN_INTEGRATED_LOOP=1` in Step 3's and every later step's arms, in the flip candidate and D7; its over-abstention measured first; WM10's comparator on the same path; section 9 |
| 6 | the attribute role is on by default; the missing piece is the chat parse route (LOW) | WM9c; biology `gist-detail-graded-forgetting`; the stale "default OFF" code comments noted in WM9c |
| 7 | Rsp's second telling and Rms fell after `d1h` in the evening, and outside the morning block (LOW) | 5.1 spacing: first tellings moved 2 h earlier, last exposures inside the blocks, all before `d1h` |
| 8 | WM5d passes by construction at Step 1; Step 0b's rule contradicted itself (LOW) | WM5d (named an installation check; WM5e REPORTED); Step 0b rule (clauses exclusive, (c) reachable) |
| 9 | the per-telling read is invasive and U0 cannot see a shared perturbation (LOW) | 5.1 (save and restore, state assertion), `n1_noread`, U0b; 5.6 and 5.7 arm counts |
| 10 | wording: "decided at encoding" for both seeds; "where it landed"; "not sure any more"; Roediger & Marsh stretched; merge order and evidence strength (LOW) | section 2a items 2 and the answer paragraph; section 8; Step 3(b) and biology `repetition-retrieval-strengthen-same-trace`; the merge-order note after the introduction; this section's first paragraph |


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
(doi 10.1016/j.neuron.2014.07.017); Yang et al. 2024, Science 383:1478 (doi 10.1126/science.adk8261, PMC11068097);
Schapiro et al. 2018 (doi 10.1038/s41467-018-06213-1); Singer & Frank 2009 (doi 10.1016/j.neuron.2009.11.016);
Ambrose, Pfeiffer & Foster 2016 (doi 10.1016/j.neuron.2016.07.047); Mattar & Daw 2018 (doi 10.1038/s41593-018-0232-z);
Rasch & Born 2013 (doi 10.1152/physrev.00032.2012); Payne et al. 2008, full text (doi 10.1111/j.1467-9280.2008.02157.x);
Sekeres et al. 2016 (doi 10.1101/lm.039057.115); Winocur & Moscovitch 2011 (doi 10.1017/S1355617711000683); Reyna et
al. 2016 (doi 10.1016/j.jarmac.2015.12.003); Corbin et al. 2015 (doi 10.1016/j.jarmac.2015.09.001); McClelland,
McNaughton & O'Reilly 1995 (doi 10.1037/0033-295X.102.3.419); Tse et al. 2007 (doi 10.1126/science.1135935); Richards
& Frankland 2017 (doi 10.1016/j.neuron.2017.04.037); Hardt, Nader & Nadel 2013 (doi 10.1016/j.tics.2013.01.001); Berry
et al. 2012 (doi 10.1016/j.neuron.2012.04.007); Karpicke & Roediger 2008 (doi 10.1126/science.1152408); Lee 2008
(doi 10.1038/nn.2205); Cepeda et al. 2006 (doi 10.1037/0033-2909.132.3.354); Moncada & Viola 2007
(doi 10.1523/JNEUROSCI.1083-07.2007). Added in revision 2: Turrigiano et al. 1998 (doi 10.1038/36103); Wagner et al.
1998 (doi 10.1126/science.281.5380.1188); Han et al. 2007 (doi 10.1126/science.1139438); Leutgeb et al. 2007
(doi 10.1126/science.1135801); Hasselmo 2006 (doi 10.1016/j.conb.2006.09.002); Tambini, Ketz & Davachi 2010
(doi 10.1016/j.neuron.2010.01.001); Dewar et al. 2012 (doi 10.1177/0956797612441220); Roediger & Karpicke 2006
(doi 10.1111/j.1467-9280.2006.01693.x); Roediger & Marsh 2005 (doi 10.1037/0278-7393.31.5.1155).
Local corpus: Kandel 6e ch.42, ch.49, ch.52, ch.53, ch.54; Buzsaki, Rhythms of the Brain (2006), cycle 7 and p.123,
p.349.
