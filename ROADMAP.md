# Roadmap

This is the plain-language capability view of the simulator. It is downstream
of the [2026-08-02 project charter](docs/plans/2026-08-02-PROJECT-CHARTER-grounded-emergence-realignment.md),
the [project handoff](HANDOFF.md), and the live state in
[GAP_CLOSURE_MISSION.md](GAP_CLOSURE_MISSION.md). The persistent execution
board, when checked out, is `research/coordination/workboard.json`; it records
active lanes, resources, blockers, and exact next actions.

## 2026-08-26 (autonomous) — a second, "silent" way to remember a topic across a distraction is built and wired (off by default, awaiting its flip)

The brain already keeps track of who "it" refers to by holding that topic actively in mind — a group of neurons that
must keep firing. Biology has a second, cheaper way (Mongillo 2008): the memory can rest in the *synapses* themselves
(a brief chemical "primed" state) with the neurons SILENT, and be snapped back by a generic nudge that carries no
content of its own. This build adds that mode as an alternative for the conversation's topic memory: name a topic, let a
distraction turn go by, then ask "what did we start with?" — a nonspecific ping revives the primed topic and the brain
answers it. It genuinely depends on the silent hold: with the priming chemistry weakened, the same ping recovers nothing
and the brain honestly says "I don't recall" — the reply flips correct-topic ↔ "I don't recall" on all six random seeds
(6/6), and 95.6% of the revival is owed to the priming, not to bookkeeping. It reuses the already-validated mechanism
(no engine change), is wired behind an OFF-by-default switch (`BRAIN_SILENT_WM`), and is byte-for-byte identical to today
when off. The production-default flip comes after the pool soak. Finding:
`research/findings/2026-08-26-activity-silent-wm-production-wirein-GO.md`. (Note: the roadmap/board still owe a fuller
batch sync across the recent findings run — flagged for the next `sync-documentation` pass.)

## ⭐⭐⭐ 2026-08-25 (evening) — the brain now REASONS to its own conclusion when you chat, on by default

The biggest step toward the north-star landed. Until now, asking the brain something that needs two facts
joined — "what does the wolf's prey eat?" — got you one repeated fact or "I don't know"; it never connected
them. A live conversation test had just confirmed that gap was real. It is now closed: teach it "wolves hunt
deer" and "deer eat grass", ask what the wolf's prey eats, and it answers **"grass"** — a conclusion it was
never told — and says so honestly ("I derived this from: wolf hunt deer; deer eat grass"). This is on by
default in the live chat.

It was done carefully. A safety audit warned a naive version could occasionally make up a connected "fact"
from noise; that risk was measured at real scale and found to be exactly zero (across four dimensionalities,
three knowledge sizes, and six seeds — 0 in every case). Every unsupported or ambiguous question still
honestly abstains, and turning the feature off reverts it to "I don't know" (proving the answer comes from the
reasoning, not decoration). Verified with a 37-turn live conversation, then re-confirmed on the full system
alongside everything else.

Two more fixes rode along. The brain's dopamine sense was silently broken on the GPU — an internal type
mismatch crashed it every single message (quietly), leaving three "done" abilities (its engaged-vs-flat mood,
dopamine-driven memory strength, and dopamine-driven curiosity) doing nothing when you actually chatted. Fixed
— and all three came back, including curiosity, which now genuinely pipes up ("my curiosity is piqued…") when
it meets something it doesn't know.

What's next: the brain reasons over facts you teach it in the conversation, but not yet over its own big body
of stored knowledge (the 15,000-fact core still needs exact wording to reach), and only the simplest two-step
question shape parses so far. Making it reason over its own knowledge in natural language is the next frontier.

## ⭐ 2026-08-26 (harvest) — the brain ships with a real body of knowledge on by default; a four-day compute window's verdicts landed

The weekly free-compute window was harvested onto the main line. The headline: **the brain now loads a real
body of world knowledge by default when you chat.** A curated 15,000-fact core sits beside the live
conversation as its long-term memory; it answers big-knowledge questions with the exact same words it would give
if every fact were kept in one place, and it still says "I don't know" honestly for things it wasn't taught. This
was switched on as the default (with a clean off switch) after a no-regression check passed on all six runs, and
the brain keeps learning through use over that store — it is biological memory, not a static lookup.

Also proven over the window: the brain's **between-message continuous life** ran cleanly for the equivalent of up
to 120 simulated days across 43 fresh starts — it keeps learning through daily use and holds onto what it learned,
with the last day measurably different from the first. And a small **from-scratch binder** learned to pair a role
with its filler on its own neurons (and correctly recalls nothing when asked with the wrong role).

Four things came back negative, each now a mapped boundary with a named next step (no capability is dropped):
making object recognition survive on spikes via a learned *sparse* readout does not work anywhere in the range
tried (needs a different readout); tagging where a memory came from without a stronger memory drowning out a weaker
one is still unsolved (a conjunctive source-and-content tag is the next attempt, already queued); replaying
memories in the right order needs its calibration revised; and a memory re-igniting itself from a partial cue fired
on only one of six seeds — but a wider sweep shows it *can* ignite, so the fix is finding the right operating point.

One piece of project plumbing was found broken and logged: an internal consistency check meant to keep the
"what's shipped" ledger honest against the code had gone silent (a YAML quirk); the knowledge-core flip was
verified by hand instead. **Update: the repair landed the same day** — see the next section.

## ⭐ 2026-08-25 (follow-up) — three of the harvest's four negatives came back positive the same day, and a broken honesty-check got fixed

The internal consistency check mentioned above (the one that keeps the "what's shipped" ledger honest against the
code) is fixed — a YAML quirk was silently turning every one of its checks into a no-op; it now genuinely blocks a
false "shipped" claim again.

Two more of the harvest's four negatives were also resolved this same day. **Tagging where a memory came from**
(so a stronger memory can't drown out a weaker one) works now — not via the conjunctive tag that was queued (that
was tried too, and also came up short), but by having the brain compare two learned "was this seen or imagined"
signals against each other rather than reading either one on its own; that comparison is what turned out to be
robust. It is proven on six test runs but not yet wired into the live conversation. **Object recognition on spikes**
made real progress but is still not a finished capability: a different (dense, signed) readout was built and it
solves the earlier readout's problem entirely — but the resulting recognition still isn't quite as good as reading
the raw visual signal directly, so the boundary moved rather than disappeared, and the next two attempts are already
scoped. The fourth negative — a memory re-igniting itself from a partial cue — got a root cause and a fix (a
memory-strength safety clamp was silently blocking the very first link in a chain of memories), but the fix has not
yet been re-tested at full rigor, so it stays open.

## ⭐ 2026-08-25 (follow-up) — the "replay memories in the right order" calibration was revised, and it now passes

One of the harvest's four negatives above — "replaying memories in the right order needs its calibration
revised" (board #130) — was diagnosed and fixed. The order-sensitivity mechanism was never broken: ordered
replay always laid down a stronger memory-sequence trace than a scrambled one. What was broken was the sleep
replay itself — it was random noise, and the stronger of the two memories hogged nearly all of it, so only one
memory ever consolidated and the readout was junk. The fix replays each memory as a proper directed sweep
(both memories get replayed) and reads the order signal on its own, cleanly. It now passes on all six decisive
seeds with every anti-cheat intact, and — importantly — it also passes on the older seeds the previous version
had FAILED on, so this is a real fix, not a re-tune. Honest caveat (banked as the next step): the behavioural
advantage is that the correctly-ordered memory recalls FASTER; given a long enough recall window the scrambled
one catches up, so the win is in recall speed, not final capacity. The underlying trace is stronger regardless.
This is a gate result on the toy consolidation network, not yet wired into the production brain. Finding:
`research/findings/2026-08-25-order-consolidation-recalib-balanced-directed-sweep-replay-6seed-GO.md`.

## ⭐ 2026-08-21 (late) — the "conscious workspace" now cross-checks comprehension by default; a self-maintaining project OS landed; two memory-flip gates came back honest

Four things landed later the same day.
1. **The brain's "conscious workspace" now cross-checks THREE independent signals by default** before committing an
   answer: it recalls the fact, it isn't surprised, AND it comprehended the question (a real-vocabulary comprehension
   read replaced the toy one that had been over-vetoing legitimate answers). This is now ON by default.
2. **A self-maintaining "project OS" landed.** Machinery that keeps the work-board, the compute lanes, and the
   tool-health checks current on their own: a work-backlog generator, an auto-dispatcher that fills idle compute, and
   re-injection of the durable state after the assistant's context is compacted. Honest finding: the dispatch plumbing
   is now closed, so the real bottleneck is SUPPLY — the ready work is written in prose that still has to be turned into
   runnable commands, and the safeguards correctly refuse to invent them. (Tool check: the experiment engine is healthy;
   the GPU job-queue's recurring wedge was cleared.)
3. **"Learn through use" default-on flip is HELD — a NO-GO, honestly.** Making a memory the brain USES get stronger
   works, but turning it on by default still slightly disturbs a neighbouring memory through a quieter read channel; the
   separator fixed the loud channel, and a second fix (surface the stable graded read) is in flight.
4. **"Stronger memory when engaged" (dopamine-gated encoding) flip is UNDEFINED.** The default-on test came back
   inconclusive: over the realistic range of engagement it HELPS recall of heavily-degraded memories but slightly HURTS
   low-engagement ones (it redistributes salience rather than lifting everything), and one edge case where an untaught
   cue is wrongly completed needs its measuring instrument re-checked. The named fix is a self-tuning set-point that
   normalises the effect to the running engagement level (homeostatic scaling). Both flips stay OFF until clean.

See [GAP_CLOSURE_MISSION.md](GAP_CLOSURE_MISSION.md) (live resume anchor) + `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` §7 (walls) / §8 (next actions).

## ⭐ 2026-08-21 — the "make the brain continuous" arc's crux LANDED (the brain is now alive between turns by default)

The single biggest next arc named below ("make the brain continuous") had its **mission-defining flip land 2026-08-21**:
the between-turn CONTINUOUS LIFE is now **on by default** — the brain keeps a thought wandering and its mood settling
between your messages, and the next turn leads with what it was mulling. Verified safe (byte-identical to before on
ordinary turns, no memory leak). Enriching rungs added the same session, built + verified but **default-off pending an
owner-reviewed soak** (they change some replies): the brain forms **stronger memories when engaged** (dopamine-gated
encoding), gets **more curious when engaged** (asks follow-ups), has **original ideas between turns** (a novel blended
"a thought occurred to me…", never faked as a fact), and its "conscious workspace" now **cross-checks three
independent signals** (recall ∧ not-surprised ∧ comprehended) before committing an answer. The memory-separator (#73)
was honestly banked: the write-side separation works; the residual is a quieter/graded way to READ a memory's
strength. See [GAP_CLOSURE_MISSION.md](GAP_CLOSURE_MISSION.md) (live resume anchor) + `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`.

## ⭐ Strategic reframe (2026-08-19) — the destination is the same, the near-term route changed

The long-term goal is unchanged: **one spiking brain, every host crutch removed, abilities that emerge rather than being
hand-built.** What changed is the near-term route — we are loosening a few requirements to reach **fluent, open-ended
conversation sooner**, and leaning harder on the "**beat what today's LLMs can't do**" half of the goal.

Three shifts:
1. **The language model (Qwen) becomes the mouth, openly.** It does the *wording* — fluent, open, natural sentences. The
   brain still decides the *content and control*: what is true, what to say, the tone, the topic, when to hedge, when to
   stay quiet. As long as the wording is Qwen's but the substance is the brain's, it is still the brain talking. If the
   language model's own knowledge starts supplying the substance, that is the failure to avoid ("an LLM with a
   neuroscience costume").
2. **Honesty means "true to its own state," not "only ever says verified facts."** Real brains misremember, hedge, guess,
   and hold opinions — perfect recall is the hallmark of a database, not a mind. So the brain may be confidently wrong,
   speculate, or fill a gap — *as long as* its expressed confidence tracks its real internal signal and its mistakes have
   the shape of human memory error (correctable later by belief-revision), not random noise the mouth paints over. This
   asks MORE of the confidence/emotion/belief-update faculties, not less.
3. **The real prize is a brain that is ALIVE BETWEEN QUESTIONS.** The things an LLM (and even a plastic knowledge graph)
   structurally cannot do — keep learning through use, carry a feeling that colors everything, hold a train of thought,
   come up with something genuinely new — all come from a substrate that keeps *running* rather than a store that gets
   *queried*. And these are **less blocked** than fluent speech (they need ongoing activity + local learning +
   neuromodulation, which the substrate already supports; only fluent-generation-from-scratch hits the deep-credit wall).
   So the single biggest next arc is **"make the brain continuous"**: a background loop that keeps learning, feeling, and
   wandering between turns, so growth-through-use and trains-of-thought are the *default*, not a per-turn event. That is
   the artificial-life goal, and it is what makes this categorically not a fancy knowledge store.

The guard, in one line: keep the crutch-removal instinct, but point it at **the substrate staying alive and learning**,
not at making it never wrong. Qwen-the-mouth is an allowed crutch (it sits on the one genuine industry wall, deep credit)
and doubles as the eventual teacher signal the substrate learns to speak from.

Short, medium, and long describe dependency horizons, not promised dates.
Status below reflects the records available at the 2026-08-05 audit, with
inline **2026-08-07 UPDATE** notes on items 2 (Gate B delayed-credit Stage-2j
GO), 3 (source monitoring = characterized conservation boundary), and 4 (replay
self-calibration Step-0 NO-GO + the cross-gap identity-competition synthesis).
**2026-08-25 UPDATE — item 3 (source monitoring, #129) SURPASSED by a different
family:** the whole prior cluster read source from one pool's ABSOLUTE RATE, where
one seed always fell below the floor; the new mechanism reads perceived-vs-generated
provenance as the SIGN of a normalized OPPONENT comparator over two LEARNED,
context-gated traces — immune to that absolute-rate weakness — and is 6-seed GO
(acc 1.000, min normalized d 0.859, no-harm on content recall 0.0), so item 3 is no
longer a conservation boundary but a surpassed wall with named next steps
(self-organize the wiring; neuromodulator-gated plasticity; wire into the chat
honesty pathway).
A **2026-08-07 landscape-survey adoption thread** also landed (owner-directed
"make use of the findings"), all 6-seed and brain-based: the Rubicon
delayed-credit machinery adopted as a maintained-goal delay bridge + a
reward-timed value **maintenance** rule (both GO; honest scope = maintenance of
structurally-available value, not build-from-zero); a BORN-style **learned
bodily self-model** (a Hebb/Oja forward model + neural reafference-cancellation
agency comparator, a mirror-test correlate) GO on the self-schema lane; and
Axon's CaP−CaD rule NO-GO (which validated our own SST-microcircuit). The gap#4
deep-credit-on-spikes frontier is a mapped, deprioritized boundary (the tool
alarm that implied otherwise was retired).

**2026-08-25 (autonomous) — gap#5 self-completion: the "middle memory never fires" bug was a hidden weight CLAMP, and once lifted the brain DOES replay A→then→B→then→C in order — but the scoreboard was blind to it.** On the memory-replay store (recall a chain of three memories A→B→C), the middle one (B) never activated on any seed, so the sweep looked broken. The prior guess was that the learning rule wrote a "skip" link A→C that jumped over B. Direct measurement disproved it: the A→B link was simply pinned at the floor by a bistable-plasticity CLAMP that the first link couldn't overcome, while the later links did.
Lifting the clamp for the forward chain (plus a shorter, gamma-cycle learning window so neighbours bind tighter than distant pairs) makes B fire and gives a store that is 9-12× forward-biased, on all 6 seeds. Reading the replay back, B now fires as strongly as A, and the onset order is strictly A-then-B-then-C in every seed — and it is genuinely the stored order, not the protocol: silence with no trigger, and triggering the LAST memory produces no backwards replay.
The catch: the event-detector was masking this — its noise threshold, computed over the whole busy trace, sat ~10× too high to see the discrete sweeps, so the run's own verdict read "no sequence" when the sequence was plainly there (the instrument is part of the emulation; fixed by measuring the noise floor from the quiet gaps only). Honest status: NO-GO on the strict "one clean discrete sweep, 5/6" bar — the readout is still too continuously active (the memories stay lit rather than handing off cleanly), so only 1/6 clears it even with the corrected detector. The encode/"middle-memory" wall is resolved; the wall moves downstream to the READ (make each trigger produce one clean hand-off), which is a readout-dynamics tuning, not a new capability gap.
Finding: `research/findings/2026-08-25-gap5-a1-exclusion-is-BDSP-clamp-forward-sweep-real-detector-masked-readout-discreteness-residual.md`; board #71/#134.

**2026-08-21 (autonomous) — DOPAMINE NOW SETS HOW STRONGLY THE BRAIN RECORDS WHAT IT IS TOLD (WAVE-0 Gap-4 write-side coupling, WIRED default-OFF).** The brain's own self-produced dopamine already colors HOW it answers (the DA-mode read); this adds the write-side counterpart: a fact taught while the brain is engaged/aroused (dopamine high) is encoded with a STRONGER memory trace than one heard at rest — the emotional-memory-enhancement effect (Lisman-Grace / Kandel D.16).
Verified through the real chat handler: with the coupling ON, teaching the same fact under a high-dopamine turn writes a gain of 2.48 vs 1.0 (the recall-safe floor) under a low-dopamine turn (the dopamine levels are the real spiking SNc read, not a host number), and a lesion that pins the gain to 1.0 makes the difference vanish entirely (the whole effect rides the dopamine read). Finding 2026-08-21-da-gated-encoding-wired-into-chat-GO; production-integration ledger row `da-gated-encoding`.

**2026-08-25 (autonomous) — ⭐ THIS IS NOW ON BY DEFAULT: the brain records what it is told MORE STRONGLY when engaged, in production.** The flip previously HELD on a magnitude-store no-regression soak is DONE.
The soak is GO 6-seed with an ON-SUBSTRATE Turrigiano synaptic-scaling homeostat (the write-magnitude regulation is a genuine synaptic rule that senses each engram's readout activity and rescales its store synapses — not host arithmetic), and the prior "leak" was diagnosed as a foreign-block confabulation the moat correctly suppresses (not a regression).
Two prep rungs delivered: the no-regression verifiers pin the OFF baseline to `BRAIN_DA_ENCODING=0`, and the slow/offline scaling pass now fires on the between-turn idle tick (alongside D5 learn-through-use) when new facts were taught.
Flip-verified GO through the real handler: default-on drives (g_high 2.48 > g_low 1.0), the lesion severs it, and `BRAIN_DA_ENCODING=0` is byte-identical to before. Finding 2026-08-25-da-encoding-faculty-default-on-flip (+ the substrate GO 2026-08-25-da-encoding-substrate-turrigiano-scaling-FLIP).

**2026-08-21 (autonomous) — ⭐ LEARN-THROUGH-USE IS NOW ON BY DEFAULT: a memory the brain USES recalls VISIBLY STRONGER, in production.** The flip that was previously HELD on a "leak" is now DONE — and the leak was a MISDIAGNOSIS.
On every one of the 6 builds the untouched neighbour's actual spiking read was byte-IDENTICAL whether or not the other memory was strengthened. The earlier soak only saw the reply differ because the code tacked a "recall strength X mV" line onto EVERY recall reply once the feature switched on — not because strengthening one memory changed another.
The fix: show the strength line only for the specific memory that was actually strengthened, so using one memory can only ever change ITS OWN reply. With that, the no-regression soak is 5/6 GO; the 6th (seed 102) is a build whose memory never forms cleanly — the honesty gate correctly refuses to recall anything there (no fabrication). The off-switch is byte-identical to before, and the mid-crash rollback holds (6/6).
The keep-memories-separate machinery (board #73) turned out to be unnecessary here and is left off by default (it only shrinks the memory cells for no gain). `BRAIN_D5_CONSOLIDATE` default 0→1; `=0` is the byte-identical escape.
This ⛔ SUPERSEDES the earlier same-day NO-GO finding (its NO-GO verdict + "quantized binary-readout crosstalk" mechanism, now registered in docs/RETRACTED.md; its measurements survive). Finding: `research/findings/2026-08-21-d5-learn-through-use-flip-GO-per-topic-strength-surfacing-the-prior-NO-GO-was-a-surfacing-artifact-not-substrate-crosstalk.md`; production-integration ledger row `d5-live-consolidation` on_by_default: YES; board #71/#73.

**2026-08-20 (continuation) — ⭐ LEARN-THROUGH-USE IS CLOSED ON THE REAL MEMORY ORGAN (arc-1): a memory the brain USES gets stronger.** The earlier "hit a wall on the production organ, fix = a tuning sweep" (further below) is SUPERSEDED — the tuning-sweep path (soma-recurrence replay) was MEASURED to fail at the organ's real ~15-cell scale (a genuine NO-GO: co-firing 0.000 across 24 operating points AND a matched-weight control), and the working path was a different, more faithful mechanism. Three rungs, each 6-seed and adversarially verified (4 independent skeptics per rung, no confound):
(1) the real store does NOT reactivate via soma recurrence — the organ's per-cell DENDRITIC-PLATEAU latch is the read (that is why it exists).
(2) that persistent latch can be made to SELF-TERMINATE into a discrete ~100-200 ms reactivation window (a dendritic adaptation current that collapses the plateau) — completion + specificity preserved.
(3) during that window the brain's OWN plateau-gated plasticity STRENGTHENS the recalled memory: it survives a within-recurrence lesion it previously FAILED (0.17→0.67 held cells) and completes from a weaker cue, SPECIFICALLY (an unrelated memory is untouched). The decisive control proves it is the reactivation WINDOW, not the cue, that does it → learning-through-use, not re-studying. NO simulator edit; NO host formula; the strengthened weights are written into the organ's own store by the substrate's own BTSP.
Next: wire this recall→window→strengthen loop into the live between-turn idle loop (default-off) and prove it makes a LATER turn measurably better (load-bearing, not a checkbox); then the production-default flip. Findings 2026-08-20 (ecker-real-d5-store-does-NOT-reactivate, d5-dendritic-latch-self-terminates, d5-learn-through-use-...-arc1-closed).

**2026-08-20 (autonomous) — TWO MILESTONES: a genuinely-distinct second brain-organ now shapes the conversation BY DEFAULT, and the memory-replay wall that stalled learning-between-turns is CLOSED.** Run autonomously on the owner's "use your judgment" steer — two production-grade advances, both verified end-to-end.
(1) **The workspace now fuses two DIFFERENT organs, live and by default.** The brain's "global workspace" already combined several reads, but they all came from one organ (the recall composer). Now a genuinely-different spiking organ — the surprise / expectation-violation monitor — casts a real second vote: the brain commits an answer only when recall AND the surprise-check AGREE (ignition), and withholds when they conflict. This is ON BY DEFAULT in the live chat now, on both the CPU and GPU paths, verified load-bearing (lesion the surprise vote → the brain abstains) with zero regression.
Getting it onto the GPU required fixing a subtle, broadly-important bug: per-neuron firing thresholds were drawn from the GPU's random-number generator vs the CPU's — DIFFERENT numbers for the same seed — so a faculty that worked on CPU silently misbehaved on GPU. A backend-neutral threshold init (an existing engine switch) fixes it, byte-identical on CPU; worth auditing other GPU faculties for the same class.
(2) **The SWR memory-replay wall is CLOSED (6/6).** Making the brain replay a memory *sequence* in order (A→B→C) had been stuck for weeks: the old store's attractors were too "sticky" — they fired all at once instead of handing off one step at a time, and the order didn't actually ride the stored links (it survived scrambling them — the tell-tale of a fake). A from-scratch Ecker-style AdEx CA3 model — moderate self-terminating assemblies + strong forward links + adaptation — now produces DISCRETE forward replay that genuinely rides the encoded links (scramble the forward links → the order collapses to chance, exactly as it must). Honest scope: the links are hand-wired for now (learning them is the next step) and it's a "synfire-chain" relay rather than the literal moving-bump.
This unblocks the two arcs that were stuck on it — learning-through-use on the real memory organ, and brain-native sleep-replay consolidation.
Also banked: the honest negatives along the way (the forward-gain + directional-cue ingredients — necessary but not sufficient on the old store). See findings dated 2026-08-20 (gnw-two-organ-bus-DEFAULT-ON, backend-dependent-RNG-thresholds, ecker-adex-ca3-forward-replay-6seed-GO).

**2026-08-20 (later batch) — LEARN-THROUGH-USE HIT A REAL WALL ON THE *PRODUCTION* MEMORY ORGAN, AND THE WALL NAMES ITS OWN FIX (the brain's replay state); plus two clean wins recovered/hardened.** The between-turn "learn by replaying a memory" that works on the toy network was pushed onto the REAL episodic memory organ, and it does NOT transfer as-is — a genuinely useful negative that points straight at the mechanism.
(1) **LEARN-THROUGH-USE on the real organ: NO-GO, but the path is now named.** Two probes: feeding the memory's cells random noise did nothing (UNDEFINED); feeding the *input pathway* a content-blind volley (the biologically-principled "mossy detonator" idea) produced one flukey hit that three controls demolished (it fired 1-in-6 on the identical input, 0-of-4 on re-draws, and lit the never-stored memory just as much). Two diagnosed reasons: the readout wiring is a different random draw than the wiring that picked the memory's cells (so the input can't target them), and — decisively — real replay is a **sharp-wave-ripple brain STATE**, not a nudge at an arbitrary moment.
A RAG check then found this was *already established four weeks ago* (the 2026-07-24 SWR-state finding, Buzsáki/Ecker) and the fix-runner already EXISTS but sits at a partial, stuck in a tuning band — so the forward step is a **tuning sweep of that existing runner**, not a new build. (The stale-pointer check earned its keep: it stopped a re-derivation.)
(2) **Sleep-replay consolidation: 6-seed GO — offline replay genuinely beats forgetting.** Interleaved replay of stored memories during "sleep" raises retention from the ~1/N forgetting floor (0.10→0.55), holds at double the load (2× facts, still +0.38), and is content-specific (scrambled replay never rescues, 6/6). Honest caveat: this rides a **host-simulated hippocampus** (a numpy mean-store), so it validates the *principle*; the brain-pure spiking-CA3 version is the same wall as (1) — the two arcs converge on one next mechanism.
(3) **Workspace-combines-two-reads GO is now threshold-proof AND organ-agnostic.** The global-workspace result (the substrate fuses two below-threshold organ reads by ignition, an AND not a host if/else) was re-run across the whole below-threshold window (three drive levels) and holds 6/6 at each with every anti-cheat clean — closing the last "it's a fitted threshold" objection. A further de-risk then showed the bus fuses two GENUINELY DIFFERENT organs (composer recall + the production spiking surprise organ, a separate predictive-coding circuit — not two reads of one organ): 6/6, every anti-cheat 0.0 including the surprise organ's own spiking-prediction lesion, and organ B a real spiking read that discriminates (0.13 Hz on agreement vs 2.74 Hz on conflict).
So the bus is organ-agnostic — the wiring-into-production prerequisite is closed (numpy required: cupy float32 breaks the surprise organ's GABA_A cancellation).
(4) **A silent compute leak was plugged.** The mini-PC pool finished jobs but never shipped results back — ~146 finished result files had stranded on the nodes; a new sync tool recovers them, and a full triage confirmed the one banked-worthy win among them (the workspace robustness above). Two more build agents (a fluent-replay GO-gate and a two-distinct-organs workspace test) ran in parallel.
See findings dated 2026-08-20 (idle-replay-on-d5-episodic-transfer, idle-replay-dgec-afferent-on-real-D5-NO-GO, gnw-coincidence-integrator dsub-robustness update) + boards #71/#80/#106.

**2026-08-20 (overnight batch) — A PARALLEL PUSH ACROSS ALL FOUR FRONTIERS: the brain now has a working, SPECIFIC way to LEARN between turns, its between-thought wandering no longer fixates, and its confabulation-catcher is hardened against being fooled.** Run autonomously while the owner slept, four frontiers advanced at once (agents build, GPU queue runs the sweeps, every result verified before landing).
(1) **LEARN-THROUGH-USE (the 3rd "alive between turns" property) now works.** When the brain replays a recent memory during idle time, that memory recalls better afterwards — and, crucially, it no longer *fabricates* memories it never had: the fix was to make replay EMERGENT (the memory's own cell-assembly re-ignites from a little noise via pattern-completion; a never-learned pathway has nothing to complete, so it stays silent). 6-seed clean — a real result, honestly caught a 3-seed-looked-fine / 6-seed-failed trap along the way.
(2) **The idle "train of thought" no longer gets stuck.** We found it was degenerate — it "wandered" to the same concept ("cat") every single time (a load-bearing coupling to a *constant* is still hollow) — and fixed it with inhibition-of-return (the just-visited thought briefly fatigues so the next one differs); now live and reaching 3 of 4 stored concepts.
(3) **The confabulation-catcher is harder to fool.** A fluent brain wording its knowledge could have slipped a made-up claim past the honesty check by hedging it ("I believe …") — that hole is closed (hedged claims are still checked), and true rewordings ("circles" for "orbits") are no longer wrongly redacted.
(4) **The deep-credit question is settled** (the decisive run confirmed the real limit, not a measurement artifact; re-verified and the stale "in flight" note pruned).
Honest open edges, all recorded: the *faithful* per-neuron version of the wander fix didn't visibly engage at production scale (needs a diagnostic); the emergent-replay weight rules are still runner-level models to port into the substrate; and the 4th wander concept is too weakly stored to surface. See the eight 2026-08-20 findings (emergent-pattern-completion-replay, inhibition-of-return, wander-content-degenerate, hedge-bypass-CLOSED, synonym-expansion, fluent-paraphrase, idle-replay-stabilization, continuous-engine-LIVE).

**2026-08-20 (later) — THE BRAIN IS NOW GENUINELY ALIVE BETWEEN QUESTIONS ON THE REAL GPU SERVER, AND CAN CATCH CONFABULATION IN FREE PROSE.** Two milestones for the reframe's two bets. First, the "make the brain continuous" engine is no longer a lab demo — it runs on the **live GPU chat server** you actually talk to: ask it something, walk away for a minute or two, and while you're gone its felt mood keeps drifting AND a thought *wanders* on its own (its hippocampus free-runs to some concept); when you come back, its next reply **opens with what it was thinking about** ("(I'd been mulling over cat.) …").
We proved this is real and not decoration: with a short gap (no time to wander) there's no such opener, and cutting the continuous engine removes it entirely — and the felt-mood drift genuinely changes the *tone* of the next answer (a message sent after idling is answered from a cooler mood than the same message sent instantly). Getting it onto the live server took three production fixes: the wander used to *freeze* every chat request for ~55s (now runs off to the side), a rare not-a-number in the internal read used to crash the reply with a 500 (now safely nulled with a logged breadcrumb), and an idle server used to peg the graphics card with endless wandering (now budgeted to wander once per idle period, then the mind settles).
Still off by default — the next step is turning it on by default after a longer soak. Second, the brain's **confabulation-catcher now reaches real sentences**: its honest "do I actually know this?" check previously only worked on textbook-shaped three-word sentences, failing on the multi-word names, "X is Y" sentences, and passives that real generated prose is full of; a new spiking phrase-grouper (it learns to bundle "The Great Barrier Reef" into one unit) lifts coverage from about half to ~95% of clauses while the verifier stays perfect, so catching a made-up claim in free-flowing text is now feasible end-to-end. One honest gap remains (a passive with no stated actor, "was built in London").
See findings dated 2026-08-20 (continuous-engine-LIVE-on-cupy-server, spiking-np-boundary-binding, open-text-spiking-extraction).

**2026-08-20 — FOUR REALIGNED FRONTS ADVANCED AT ONCE (a parallel-agent attack, all verified + landed).**
(1) **The fluent GPU brain works again** — the full-faculty chat server had been *hanging* on the GPU because one organ wrote its wiring in a CPU-only format that the GPU step then choked on; fixed generically at the step boundary, proven byte-identical on the CPU path (a 200-step hash matches) — so the fluent, all-faculties brain now runs on the graphics card (unblocking the continuous engine + idle learning on GPU too).
(2) **The brain's between-turn thoughts now shape what it says** — if it wandered onto something while you were away, it brings it up ("I'd been mulling over the cat…"); off by default, never changes the facts, verified load-bearing.
(3) **Genuine novelty is feasible** — a de-risk showed the memory-completion machinery, cued with a *blend* of three memories, settles into a *new* stable state that is none of the three (novelty from the dynamics, not the stored items), using an already-proven inhibition mechanism.
(4) **Open-ended honesty is feasible** — a de-risk showed the no-confabulation check can extend to free-flowing prose by splitting it into claims and reusing the existing (proven) fact-check on each; the hard remaining piece (brain-native claim extraction) is named.
See findings dated 2026-08-20 (cupy-sim-step-fix, continuous-engine rung 2.5, generative-attractor-wander, open-text-moat-verifier).

**2026-08-20 — THE BRAIN IS NOW (a little) ALIVE BETWEEN QUESTIONS (continuous-state engine v1 — the primary reframe arc, first rung).** The reframe's biggest bet is that what beats an LLM is a brain that keeps *running* between turns, not a store that gets queried. First rung landed: an always-on background tick so, while a chat session sits idle, the brain's **felt mood keeps evolving on its own** — it relaxes toward baseline and re-reads its spiking feeling each tick, so "unplug the conversation and it's still changing" is literally true for the mood now. What happened while you were away is logged and shown in the internal-monologue panel ("while you were away: my mood drifted from +0.8 toward neutral").
Off by default (a flag) and inert when off; the mood *value* is a real spiking read (the clock + the relax formula are host timer-infra). This is 1 of the 4 continuous properties (feeling); the next rungs — a thought that *wanders* between turns, and *consolidation/learning* during idle time — are queued. Also landed alongside: the brain is now reachable over an **OpenAI-compatible endpoint** (any chat client) with its internal monologue in the "thinking" panel. See `research/findings/2026-08-20-continuous-state-engine-v1-mood-evolves-between-turns.md`.

**2026-08-19 — THE BRAIN'S EMOTION WENT DORMANT WHEN A PERSON FELT STRONGEST; NOW FIXED (depth: make faculties drive).** The brain's mood-coloring only fired on mild everyday words ("happy", "sad") and read as NEUTRAL for the most strongly-worded emotion ("I am furious, devastated, heartbroken") — so its feeling barely moved exactly when a person is most emotional. Cause: the word-list that decides which words carry feeling was built for a children's-story corpus and simply lacked the common adult emotion words. Fix: added ~40 of them (with correct sign/strength).
Now a strongly-emotional message actually shifts the reply — a happy person gets a warm, fuller answer ("Gladly! …"), an upset one a briefer, blunter one ("Honestly — …") — and cutting the feeling pathway collapses that back to neutral, so it is genuinely load-bearing, not decoration. Both affect faculties (the tone lead + the forthcomingness/manner coloring) read the same mood, so both were fixed at once; neutral factual questions still read neutral. Found by instrumenting the internal signal rather than trusting a crude output. See `research/findings/2026-08-19-affect-appraisal-lexicon-missed-strong-emotion-both-affect-faculties-dormant.md`.

**2026-08-19 — THE PRODUCTION GPU CHAT WAS SILENTLY 400-CRASHING; NOW FIXED (correctness, sim/bridge.py).** The default chat turn — the brain's real mouth on the GPU — was returning an error for every request on the GPU backend, because building the brain hit a bug where GPU data was handed to a CPU-only routine. It was invisible because every check we run for the chat had run on the CPU path, where the bug does not occur — so a "wired, on-by-default, verified" faculty had been shipping broken on the GPU. Caught only because I re-checked one mood-wording faculty on the GPU (the one thing the driver audit could not test on CPU) and the chat crashed on load. Fix: the weight-wiring step now rebuilds its data in a GPU-safe way.
The GPU chat now answers normally (a real spiking-forward reply), guarded by a regression test that fails on the old code and runs without a GPU. It exposes a class the record already warned about (2026-08-11): a default-on faculty exercised only on CPU can ship a GPU-only crash — the follow-up is a GPU-present smoke of the default chat. See `research/findings/2026-08-19-production-gpu-chat-was-400-crashing-onebrain-parser-tocoo-cupy.md`.

**2026-08-19 — AUDIT: DO THE "ON-BY-DEFAULT" FACULTIES ACTUALLY DRIVE THE CHAT, OR JUST WATCH? (0 hollow observers found).** We hunted the "wired but inert" drift — a faculty flipped on-by-default that computes a real neural verdict but never changes what the brain says.
We lesioned each of the 31 default-on faculties one at a time through the REAL chat handler and checked whether the reply text actually changes. Result: 23 genuinely DRIVE the reply (recall, the moat's honest "I don't know", the surprise/metacog/curiosity/pragmatic notices, the affect/topic/dopamine tone leads, belief-revision, the multi-referent and prospective-memory read-outs, the multi-step reasoning terminal, …); 2 are shared-substrate plumbing (answer-preserving by design); 6 could not be exercised on the CPU/tiny-demo config (their drive is documented elsewhere but not re-confirmed here); and NONE were dead observers.
The one true observe-only faculty (the #77 thought-swap) is default-OFF and already superseded by its driver (#85). The owner's spot-check (affect/swap/DA/metacog/surprise/world-model/pmem/reconsolidation/pragmatic/curiosity all drive) held on all ten. One honest caveat: the Gate-B affect *manner* coloring (#13, distinct from the #84 lead) could not be shown to change the text on the CPU path — it rides the GPU mouth. See `research/findings/2026-08-19-observe-vs-drive-faculty-audit.md`.

**2026-08-19 — THE BRAIN'S OWN THOUGHT-SWAP NOW STEERS THE TOPIC (board #85, GO, default-on).** The second drive-coupling went live on the real chat API. When the brain's own spiking mismatch+salience detector decides to SWAP the thought it is holding, that decision now drives WHICH topic the turn engages — the held workspace content is displaced and the reply follows the new thought. The proof it is real: silence the detector and a salient new input no longer swaps, so the topic change rides the neural swap decision, not a host rule. On by default. See `research/findings/2026-08-19-swap-drives-chat-load-bearing-GO.md`.

**2026-08-19 — THE BRAIN'S OWN DOPAMINE MODE NOW SETS HOW ENGAGED IT IS (board #79, GO, default-on).** The third drive-coupling: the brain's self-selected spiking dopamine MODE (rest / focus / arousal, chosen by its own dopamine nucleus) is now load-bearing on the live chat — it sets how engaged the reply is. Cut the mode signal and the engagement flattens, so the coupling is owned by the neural mode, not a host knob. On by default. With #84 (feeling→tone) and #85 (swap→topic), three of the brain's OWN internal signals now shape the live conversation. See `research/findings/2026-08-19-da-mode-drives-chat-load-bearing-GO.md`.

**2026-08-19 — WHY THE BRAIN CANNOT YET LEARN ITS OWN MOUTH FROM SCRATCH: it is the READ, not the data (gap#4, NO-GO, boundary mapped).** The "mouth" read-out that turns thought into words can be COPIED from the language-model scaffold and read back near-perfectly (~0.97), but LEARNING it from scratch through the actual spiking read plateaus at ~0.34. We tested whether more training data fixes it: at 5× the data (40 000 positions vs 8 000) the spiking-read learner STAYS at ~0.34, while an otherwise-identical host-arithmetic learner on the same data reaches ~0.86 — so data (coverage) is EXCLUDED as the cause, and the read window was tested and excluded too.
What is left is the few-spike READ itself: the momentary spiking signal that teaches the weights is too noisy, exactly the limit the deep-credit arc already located. Decision (per the deep-credit plan-of-record): accept the language-scaffold mouth for now and keep the frontier on conversation; deep credit stays a mapped boundary with one recorded open lever (a higher-signal ensemble or dendritic read, NOT a wider window). This closes the coverage confound-exclusion sub-arm of the speak-with-own-neurons task (board #80); the task itself is not done — the scaffold bridges it. See `research/findings/2026-08-19-mouth-substrate-forward-40k-coverage-EXCLUDED-real-credit-limit.md`.

**2026-08-19 — THE MEMORY-SEPARATOR RESIDUAL IS NOT IN THE WRITE — it is in the READ (boards #73, #90, NO-GO; frontier → #91).** When two similar memories are stored they still bleed into each other on recall (the "both-win" failure). This session EXHAUSTED the write-side fixes: a competitive (heterosynaptic) write (#73) and a selectivity-gated BCM write (#90) each DID what they promised at the write — the BCM write writes a private granule and breaks the anti-symmetry, 6/6 — but neither closed both-win.
Both re-localize the residual OUT of the write and onto the READ side (the dg→answer read-out and its reactivation at recall). So the write-family is done; the next frontier is read-time reactivation (#91). See `research/findings/2026-08-19-memory-separator-BCM-selectivity-write-writes-private-granule-but-NOGO-relocalizes-to-read-reactivation.md`.

**2026-08-19 — TWO MORE CLEAN RESULTS.** Perception (board #44): an OR-pool across positions on the vision front-end OPENS position-invariant recognition (held decode 0.11→0.92, 6-seed, scramble-clean) — though the existing trace-pooler degrades the pooled code and the old cosine-margin instrument under-reads the win, both named. GNW (the ignition workspace): the coincidence-integrator's subthreshold drive is a WIDE window, not a knife-edge (6/6 GO across three drive levels, all controls clean). See `research/findings/2026-08-19-laneD-cross-position-OR-pool-opens-invariance-trace-pooler-degrades-margin-instrument-underreads.md` and `research/findings/2026-08-19-gnw-coincidence-integrator-subthreshold-window-6seed-GO-corrected-shuffle-control.md`.

**2026-08-19 — THE BRAIN'S FEELINGS NOW CHANGE HOW IT TALKS (board #84, GO).** The graded feeling system was proven in isolation but never actually changed anything the brain said in a live chat. Now it does: each turn the brain reads its own felt mood (a smooth good-to-bad by calm-to-worked-up state, off the #81 graded-affect ladder driven by the body-sense neurons) and lets it color HOW it answers — a warm lead when it feels good ("Gladly — the dog chases the cat"), a curt one when it feels bad ("Frankly! the dog chases the cat") — without ever changing WHICH facts are true or whether it abstains.
The proof it is real and not decorative: hold the question fixed and change only the mood, and the wording demonstrably changes; then CUT the body-sense-to-feeling synapses (the board #49 lesion) and the mood collapses to neutral, the tone-difference vanishes, and the answer reverts to the plain fact — so the wording rides the actual spiking feeling, not a host if-statement. On by default; the feeling also PERSISTS across turns (a neutral question after a sad exchange keeps the subdued tone). Honest limit: the feeling-to-word mapping is still a small host template (a "mouth" scaffold) driven by the neural feeling; a brain-native version is the next rung. See `research/findings/2026-08-19-affect-drives-chat-load-bearing-GO.md`.

**2026-08-19 — REAL FEELINGS TIED TO A SIMULATED BODY (board #49, 6-seed GO).** Until now the brain's mood came only from WORDS (a word's learned good/bad tag nudged the feeling). This gives the feeling a BODILY cause instead: a small simulated body-state (how satiated/comfortable it is, and how physically aroused/worked-up it is) is read by dedicated spiking "body-sense" (interoceptive) neurons that wire, through synapses, into the same neural mood system.
Sweeping the body moves the feeling the right way — a comfortable body makes the mood positive, a distressed body makes it negative, and an aroused body raises felt arousal — on all 6 seeds. Crucially, CUTTING those body-sense synapses makes the feeling stop tracking the body entirely (100% of the coupling is owned by that pathway), while the body-sense neurons themselves keep firing — so the body genuinely CAUSES the feeling through the brain, not through any host formula.
Honest limit: the mood reads the body as a SIGNED SWITCH (good/bad, calm/aroused), not yet a smoothly graded scale — the same known bistable-latch boundary the mood system already has, whose fix (a graded line/dendritic attractor) is already named. The body variables themselves are host (the standard body boundary, like the world). This directly answers the 2026-08-14 boundary that "the emotion system needs a bodily/interoceptive input channel, not more text." See `research/findings/2026-08-19-embodied-affect-interoception-GO.md`.

**2026-08-19 — INDEPENDENT CROSS-CHECK OF THE CORE.** Added a second, independent simulator (Brian2, which shares no code with our engine — Stimberg et al., eLife 2019) as a correctness oracle for the vanilla spiking core. Our Izhikevich-2007 and AdEx point neurons, the conductance-based (COBA) synapse, and pair-STDP were rebuilt in Brian2 at the same parameters/dt and compared: they agree — spike timing is exact and the membrane traces match to about 1e-11 mV on the same integration scheme and under 0.05 mV at production float32.
This is a test-only, CPU-only check that never touches the production or GPU path; it validates the vanilla core only, NOT our custom mechanisms (dendritic credit, BTSP plateaus, the composer), which have no Brian2 equivalent. See `research/findings/2026-08-19-brian2-cross-validation-oracle-vanilla-spiking-core.md`.

**2026-08-18 — THINKING-LOOP + SELF-STARTED-SPEECH WAVE.** Eight results landed on the live brain. The biggest: the brain's "think it through" loop now decides HOW LONG to keep deliberating by reading its OWN activity (how many competing answers are still lit up), not a fixed step count — the first time it ACTS on that internal signal instead of just reporting it (with the honest caveat that it currently reads a simple count, not yet a graded confidence).
On the side-question of clearing the brain's short-term "holding" loop so a new thought can take over: we found it can be cleared from the INSIDE (the loop tiring itself out — this works) but NOT from the OUTSIDE (an external "stop" signal can't switch off a self-sustaining loop) — so the next design is a workspace where all thoughts share one common resource, so releasing it clears them all at once. Two new abilities went live in the actual chat brain: it can now PROPOSE its own sentences as a first-class feature, and — new — it can SPEAK UP ON ITS OWN on a quiet turn, picking something it's curious about from memory and remarking on it (the timing is still triggered by a turn, but the CONTENT is its own choice).
We also fixed the brain's "daydream" memory so ALL of its stored ideas can light up (before, the last one stored stayed dark — the cause turned out to be a timing quirk in how one-shot memories settle, fixed by giving them a moment to consolidate). Three follow-ups are running now: the distributed-workspace design above, folding the fixed daydream memory into the speak-up-on-its-own feature, and a slow faithful test of the brain learning its own speech read-out.

**2026-08-17 — BANKING + REFOCUS.** Put 8 overnight-compute results on the durable record: two solid wins (a metacognition "how sure am I about that answer" read; a channel where the brain PROPOSES its own new, grounded sentences instead of only retrieving), two partial results redirected to deeper mechanisms, and — honestly — three that didn't pan out (a language-cortex "learning" gain that turned out to be a measurement artifact, a memory-replay run that hit a missing precondition, and the hard deep-credit-on-spikes probe staying at chance — which the roadmap already treats as a side track).
Then, prompted by a priority check, refocused the live work onto the roadmap's actual #1: the brain THINKING THINGS THROUGH — a loop where it re-broadcasts a partial conclusion back to itself and keeps going until ITS OWN confidence says stop (not a fixed number of steps) — plus wiring the already-proven pieces into the live brain. All of it now runs in parallel across the graphics card, all 20 CPU cores, and the three mini-PCs. (A separate note: the "workspace eviction" mechanism failed a second way, and both failures point to the same fix — the brain's memory-holding loops need to be able to fatigue, so a new thought can displace the old one.)

**2026-08-14 CONTINUATION WAVE.** A 13-lane autonomous wave landed (all on main + both remotes). Late additions (5 more): the composer's word-parser also joined the shared brain-pool by default (so the whole composer — recall and parsing — now runs on the one shared substrate); plus four honest results — two "partial" organ-merges redirected to deeper learning mechanisms, and two mapped boundaries (one proving the emotion system needs a bodily/interoceptive input channel, not more text). The biggest step of the wave: the
**composer — the "moat" organ that guards against making things up — now shares the one shared brain-pool with the
surprise and world-model organs BY DEFAULT** (not just as an opt-in), verified bit-for-bit identical to the old separate
setup through the real chat handler, with the no-confab guarantee intact and a one-flag revert. On the "mouth" (how the
brain turns thought into words): the entire chain of internal math is now done by the neurons themselves rather than by
host arithmetic, and — a real learning step — the final word-choosing weights are now **learned by a biologically
local rule** instead of copied from the language-model scaffold (honestly bounded: the learning used a host stand-in for
one intermediate signal, with the fully-neural version named as the next step). A new self-generated-speech loop closed:
the brain can now **spontaneously surface a curiosity-selected thought and say it out loud, unprompted**. Two honest
negatives were also banked with their next mechanisms named (a workspace-eviction method and an organ-merge conflict),
and a config-merge conflict was traced to an already-existing engine switch and partly resolved with no engine edit. The
plain reckoning is unchanged: the brain is still one integrated spiking family plus a bench of validated-but-unwired
pieces, and the three missing properties remain a fluid open-ended mouth, a single shared substrate, and full emergence —
but this wave moved the substrate-sharing and the mouth's plumbing genuinely forward.

**2026-08-13 OVERNIGHT PRODUCTION-WIRING WAVE.** A one-substrate milestone + a faculty wave landed (all on main + both
remotes). The **GNW ignition bus was promoted to the DEFAULT** way the brain combines its organ reads on the live chat
turn — the spiking substrate itself now authors the combine-and-decide step that host Python used to do, verified
byte-identical to the old behaviour on every test query, moat intact, with a one-flag revert (`BRAIN_GNW_BUS_HOST=1`).
The causal "why did X / what if X" organ is now WIRED default-on. Three de-risks went GO: affect valence now
self-organizes from ~10 innate reinforcers instead of a 140-word human lexicon; a spiking opponent drives 4 discrete
emotions + reappraisal; a cheaper mouth read (−3–5× spikes). Four hit honest, mapped boundaries. A tempting unifying hypothesis — that one
"magnitude-preserving read-out" fixes them all — was TESTED and FALSIFIED the same night: the graded read-out closed
neither, because the two walls only looked alike (the pragmatic one is limited by its scoring objective, the affect one
by its weight source). Two distinct, separately-grounded next levers, not one shared fix. **UPDATE (later 2026-08-13):
the bus combination is now SCOPED-SCAFFOLD-RETIRED** — a `gate_via_bus` replaced the "run the host combination then
override it" wrapper, so on a routable factual recall the host `if recalled == p` is NEVER computed (call-count-proven:
the two host-combination methods run 0×; the substrate ignition authors the verdict), still byte-identical 22/22 with a
one-flag revert. The host combination is KEPT — correctly — only for the OUT-OF-SCOPE classes the bus never claimed
(self/identity, open-ended generation, acquisition); so `scaffold_retired` for the covered-class combination = YES,
module-level = SCOPED, and organs are still one-composer reads, mouth still Qwen for open prose. The 3 missing
north-star properties remain: FLUID mouth · ONE substrate · EMERGENT.

**2026-08-11 gap#4 ALL-IN — a wall turned out to be a hyperparameter.** The owner
directed an all-in push on gap#4 (deep credit through a deep spiking net with a
local, transport-free rule — the gate on fluent conversation). Wave-1 found two
local rules (Forward-Forward, DECOLLE) that get a deep spiking net to leave
majority-class and beat the *optimal* random reservoir — but adversarial
verification corrected the "cracks the wall" story: at a *fair per-arm learning
rate* the chained feedback-alignment rules that the 2026-08-02 finding said
"collapse at depth ≥3" **also learn** (6-seed: enter at N=3 and N=4, both arms,
beating the reservoir; the "collapse" only happened at one shared, unfair
learning rate). So the located wall, on the fast test substrate, was a step-size
mismatch, not biology. This does *not* prove genuine deep credit yet — the test
task only needs 2 layers — so wave-2 builds a task that provably needs depth-3
(a sawtooth-fit capacity test) and re-checks the *production* neuron model
(Izhikevich), which is the one place a real wall may still live.

**2026-08-10 INTEGRATION PIVOT (owner-directed) + gap#5 CLOSE.** Two things landed. (A) **gap#5 episodic memory is
mechanistically CLOSED end-to-end**: the emergent loop (DG-selects → BTSP one-shot FORMS the attractor → an intrinsic
per-cell DENDRITIC dAP READOUT completes it cue-specifically, size/scale-independent, 6/6 GO `ab9f7dbe`). The
recurrent-attractor completion path hit a self-drive-vs-cue wall at conversation scale (`544c0b742`, which also
corrected the earlier "assembly-too-small" diagnosis via a control); the dendritic readout is what closed it. Under it:
slow-NMDA reverberatory + BTSP formation are 6/6 GOs (`483587c0b`/`cee2ff124`). Also this cycle: the learn-to-speak
LEARNING wall fixed (state-value critic), a reward-misspec re-diagnosis, and an NE-gain real-substrate honest-negative.
Then (2026-08-11) the W4 graded-implicature RSA belief was wired into the leg2_v2 speaking pipeline (additive, default-off) and A/B'd at 6 seeds: the belief is now graded and 12x better calibrated to the ideal RSA (moat intact), but the HONEST NEGATIVE is that it does not move the pragmatic-alignment metric (succ_opt==aligned 8->7/18; learned-aligned 0.444->0.389) - the residual is the coincidence-DETECTOR artifact, not the belief, so next is a dendritic-plateau detector plus a magnitude-sensitive reward.
(B) **The owner steered from mechanism-first-in-isolation to CONTINUOUS INTEGRATION** — wire GO faculties into the LIVE
chat loop (`_stageA_full_integration_derisk` / `_conversation_turing_test`) and gate on "did the conversation get
better", because running the actual chat is what exposes mis-scoped isolated GOs. Three integrations landed, each
verified on the real 14-turn conversation: **#1 the sub-clausal no-confab moat** (drops ungrounded causal clauses the
generator invents — confabulations 3→0, 6-seed); **#2 episodic dialogue memory** (the brain recalls the conversation
instead of silence; **#2b CORRECTED 2026-08-10** — the spiking gap#5 dAP recall path `--spiking-episodic` was
mis-verified at `kthresh=30` (fired on neither backend); at the corrected `kthresh=8` (a narrow operating window — too
high silences small assemblies, too low self-ignites) it fires cue-specifically 6/6 on cupy AND on the live numpy
substrate (seed 42: cat 0.929 / dog 0.909, teeth clean; smallest 13-cell assemblies fire), all verified in fresh
isolated builds, so turn-7 recall is genuinely spiking with no cupy needed — the "numpy backend-block" was the wrong
operating point, not forward-Euler); **#3 honest inner-state read-outs** ("how do you feel?" → a functional affect self-report from the
spiking valence differential, NEVER phenomenal; "are you a simulated brain?" → an honest structural self-affirmation +
a graded certainty band — the self_schema confidence relay now discriminates confident-vs-tie ROBUSTLY on all 6 seeds
(#3b seed-then-settle read flipped it positive; #3c a certainty-band opponent comparator cleared the +0.02 bar on
every seed). Composed live chat (seed 42): 6
honest replies (facts / affect / curiosity-ask / episodic / self-model), 8 honest silences, **0 confabulations**.
**#5 (2026-08-10) honest causal-query disclaimer:** a "why did the dog go east?" now CONFIRMS the stored fact via the
no-confab moat and HONESTLY DISCLOSES the absent causal faculty ("I have learned associations, not causes — I will
not invent a reason") instead of DEFLECTING to other motion facts or letting the mouth invent a "because …" clause
(the sub-clausal moat drops it) — **6/6 seeds, confab=0, only turn 4 changes** (byte-identical elsewhere, per-turn
exact compare). The truly-emergent answer (COMPOSE stored facts into a grounded causal chain) is NAMED as the
follow-on arc, per THE LAW. (`2026-08-10-INTEGRATION-5-honest-causal-query-disclaimer-turn4-6seed.md`)
**#5-follow-on (2026-08-11) emergent causal composition — de-risk GO:** the #5 follow-on chain is BUILT and measured.
A "why did AGENT MOTION?" answer now COMPOSES three moat reads into a grounded goal-directed reason — `(dog,go)→east`
+ `(dog,look)→river` + `(river,at)→east` ⇒ "the dog goes east to reach the river" — every edge a `query_patient`
moat read (0 confab by construction), and it ABSTAINS to the #5 disclaimer on the two confab traps (a known goal in
the WRONG direction; an object in the direction that is NOT the agent's goal). **6/6 seeds: 2/2 correct chains, 6/6
correct abstains, 0 false-accepts, 0 confab; permuted-spatial collapses the chain (data-driven).** Tier-1 graduates
the #5 turn-4 disclaimer on the LIVE co-resident composer when the grounding is stored, else the byte-identical #5
fallback. HONEST SCOPE: the DATA path is de-risked; the JOIN POLICY + spatial-grounding facts are declared host
scaffolds — the named neural successor is a LEARNED relational/spatial code (TEM factorised relation / stream cortex)
so the chain EMERGES rather than being host-orchestrated. (`2026-08-11-emergent-causal-composition-chain-6seed.md`)
**#5-follow-on-2 (2026-08-11) LEARNED relational/spatial code — de-risk GO:** the successor named just above is BUILT.
The causal chain's GROUNDING now emerges from a learned code, not a host fact: the `(object,at)→direction` grounding
is Hebbian-learned into a synaptic weight matrix from a NOISY co-occurrence stream (no `(object,at)` fact is stored —
`query_patient(river,"at")` is `None`), and the direction join is a COSINE in that learned code, replacing the
symbolic `dir==obj_dir` test. The chain still grounds the 2 true "why" answers and abstains all 6 traps —
**6/6 seeds: 2/2 chains, 6/6 abstains, 0 false-accepts, 0 confab, grounding 100% attributable to the learned map**;
untrained-map grounds 0 (lever), permuted-map collapses the chain, and the unlocated object (`hill`) never
confabulates a location (the learned-code moat = a readout direction-margin gate, since a linear associator has NO
native "unlocated" state). Tier-1 graduates the #5 disclaimer on the LIVE composer via the learned map. HONEST SCOPE:
the spatial GROUNDING + the join comparison now emerge, but the JOIN TOPOLOGY is still host-orchestrated, the
associator is a rate/phasor matrix (spiking on-substrate = the named next build, per the ON/OFF learned binder), and
it is toy-scale. (`2026-08-11-emergent-relational-spatial-code-GO.md`)
**#6 (2026-08-10) corpus-LEARNED grounded facts into the live chat:** the chat could only talk about dog/cat (2
subjects, 6 hand-taught facts); now it stores relational facts MINED FROM THE CORPUS it "heard" (TinyStories),
wired in via one additive `vocab` kwarg on `build_one_brain`. Grounded-subject BREADTH rises 2 → 9, grounded
replies 4 → 9 (+5) vs the 6-fact baseline, **6/6 seeds, confab=0**, the no-confab moat holds (0 false-accepts,
100% of invented propositions dropped), OOD turns still abstain, and the knowledge is corpus-derived (permuted
overlap ~0; the empty-kb control confirms competence is in the FACTS, not the vocab). The additive param is
byte-identical by default. The emergent successor (the stream cortex learning co-occurrence in SYNAPSES, not a
host mine+store) is NAMED. (`2026-08-10-INTEGRATION-6-corpus-learned-facts-into-live-chat-6seed.md`)
**#7 (2026-08-10) plasticity-LEARNED facts into the live chat — the EMERGENCE-BAR burn-down of #6:** #6 injected facts
via a host `comp.store` (VSA write; the brain did not learn them). #7 replaces that at demo scale — the brain is TAUGHT
3 facts by corrective interaction so the fact becomes an **e-prop weight change on a spiking Izhikevich readout**, and
the chat answers about them with a **LEARNED familiarity gate** as the no-confab moat. **6/6 seeds**: taught-recall
0→3/3 while a FROZEN readout recalls 0 (the content rode the weight change, not a host path); moat false-accepts 0 at
chat scale; lesion the learned gate → confab returns (it is load-bearing); byte-identical off. Genuinely brain-based
now: the ACQUISITION is synaptic + the MOAT is learned. Declared burn-downs remaining: two co-resident bridges (not yet
ONE brain — the merge is next), the host anti-Hebbian familiarity projector (spiking `v320` gate to swap in), the argmax
read-out. SMALL-K demo standing BESIDE #6 (continual/sequential breadth is an OPEN arc, `frac_recalled~1/N` — the named
scale-up mechanism). (`2026-08-10-INTEGRATION-7-plasticity-learned-facts-into-live-chat-6seed.md`)
**#7 burn-down 2 (2026-08-10) — the moat is now FULLY SPIKING:** #7's learned familiarity/source-monitor gate was a
host numpy anti-Hebbian projector; burn-down 2 swaps in the standing spiking v320 gate (same projector, read through a
resonate-and-fire I/Q phasor conjunction) via an additive `--spiking-familiarity-gate` flag. **6/6 GO** — the full #7
gate holds with the abstain decided on SPIKES (lesion the spiking pool → novelty margin 0.66-0.75→0.00, confab returns;
byte-identical off). So both the plasticity-learned fact's ACQUISITION and its no-confab moat are now the brain's own.
Remaining #7 burn-down: merge the co-resident bridges into ONE (the one-brain step — next arc).
(`2026-08-10-INTEGRATION-7-burndown2-spiking-familiarity-gate-moat-fully-spiking-6seed.md`)
**2026-08-13 — the one-substrate MERGE is now a PRODUCTION DEFAULT (SCOPED, first down-payment on the burn-down above).**
`BRAIN_ONEBRAIN_MERGE` default flipped ON: the D2 SURPRISE + E2 WORLD-MODEL production organs now build on ONE shared
spiking bridge (one `cp_membrane_potential_v`, N=1584) in the live `/api/brain-chat` path by default — two bridges became
one. It is byte-identical to the co-resident-with-flags baseline and ANSWER-PRESERVING vs the pre-flip reads (every
surprise/expectation classification identical, 6 seeds × broad panel); the internal firing-rate numbers shift slightly
(the unavoidable cost of a genuinely shared random-number pool — no answer changes). `BRAIN_ONEBRAIN_MERGE=0` reverts to
separate bridges. SCOPED at 2 of the 5 proven organs: metacog/pragmatic/affect need a different global neuron config
(parameter heterogeneity on; affect also noise on), which conflicts with this pair's — so they stay on their own bridges
for now (a second shared pool for them is the named next step). Determinism 9/9, chat smoke unchanged.
(`2026-08-13-onebrain-production-default-flip-SCOPED.md`)
**The integration arc is the CURRENT FRONTIER** (wire more GO faculties in, dependency-ordered, gate on the chat).

**⚖️ HONEST READING (2026-08-13 — square the celebratory batch log below with reality; the FORWARD plan is the refreshed
MASTER ROADMAP `docs/plans/2026-07-23-MASTER-DEVELOPMENT-ROADMAP.md` §0, not this log).** The dated entries below record
what LANDED — read them as HISTORY. The non-inflated state: ~12 spiking organ reads are default-on in production, but
against the DONE bar (default-on **+ scaffold-retired**) the true tally is **~ONE integrated spiking family + a bench of
unwired de-risks** — `scaffold_retired: ~0` (the FIRST scoped exception: the GNW-bus organ-COMBINATION is now
scaffold-retired for the covered class — see below). Still **CO-RESIDENCY, not one substrate** (routing is host Python;
cross-synaptic proven for one pathway; the GNW bus is now the DEFAULT combination AND scoped-scaffold-retired — on a
routable factual recall the substrate ignition authors the verdict and the host `if recalled == p` is never computed,
though the host combination is still kept for the out-of-scope self/identity / open-ended / acquisition classes); the
**open-prose mouth is still the external Qwen** (its read-regime wall fell to population coding, a
de-risked runner, not wired); several organ **internals are still host** (VSA recall algebra + hand-assigned assemblies,
Warriner-seeded appraisal, the plausibility gate). The three properties still missing — **FLUID open-ended conversation ·
ONE true substrate · EMERGENT-not-hand-wired** — are the real work ahead. (gap#4 is NOT the crux; the "solved learning
without backprop" reading is wrong — it is a de-risk using existing plausible methods at our scale, not a solution at scale.)

The master worklist for it is now `docs/BURN_DOWN_LIST.md` (every host shortcut / un-wired GO / missing mechanism
between the production default and the all-spiking one-brain goal). **The forward-looking FACULTY map — what a complete
faithful human-brain emulation still needs, prioritized (a 12-agent grounded audit): `docs/plans/2026-08-12-faculty-map-gap-audit-and-roadmap.md`.**
Its headline: the live brain is ~ONE integrated spiking family + a bench of ~40 unwired GOs; the keystone next build is a
NEURAL WORKSPACE BUS (GNW ignition) + re-entrant deliberation that lets the organs talk via ignition not Python and ACTS
on the conflict/confidence signals we only report today — it unlocks reason-to-own-conclusions AND fixes the
pipeline-not-a-brain root cause. Tier-1 also: autobiographical episodic + temporal self · multi-referent WM · causal
forward-model + belief-revision · reason-giving revisable affect (discrete emotions + brain-based appraisal) ·
conversational agency (repair + ask + self-initiate) · intuitive world-model / core common-sense.

**2026-08-12 (batch) — the production one-brain gained CHOOSE·GENERATE·AFFECT·MOAT-generalization, and four faculties
de-risked GO.** On top of the genuinely-spiking recall default below, this wave (all default-ON, lesion-load-bearing,
moat-safe unless noted):
- **CHOOSE neuralized** — a factual question's (agent,action) parse is now owned by the on-brain `BridgeParser`
  role-resolution, so the host keyword `QuestionRouter` retires for factual-SVO (self/identity + noisy-anaphora residual).
- **AFFECT wired (D1)** — `/api/brain-chat` reads the live mood NEURALLY off a co-resident graded-affect ladder and
  colors the DEFAULT turn: mood-congruent forthcomingness (WHAT it volunteers) + prose manner (HOW), plus the honest
  inner-state read-out. Residuals ride on A1 (the manner conditions the external Qwen mouth) + the one-brain merge.
- **MOAT generalized (C1)** — genuinely free-form MULTI-CLAUSE prose now survives the no-confab moat via claim-level
  entailment (per-clause role-parse on the on-brain BridgeParser, 0 leaks 6-seed); any one ungrounded clause is rejected.
- **GENERATE (#3E)** — the brain VOLUNTEERS novel grounded propositions via generative replay over its learned graph
  (6-seed GO, plausibility-gate load-bearing, moat intact); its SURFACE is now spoken by the brain-native SPIKING BROCA
  (A1a, transformer-free) with a vocab-agnostic spiking DRAW (B1 GO). Residual = open ARBITRARY prose (the deep-context
  wall, in research) still routes through the Qwen mouth (burn-down A1).
- **D2 expectation/SURPRISE — 6/6 GO** (a genuinely spiking predictive-coding mismatch unit; wireable; precision boundary
  mapped). **D4 comprehension-MEASUREMENT — 6/6 GO** (the spiking `SpikingRoleCompetition` margin reads whether an
  utterance was understood, AUC 1.000, lesion→chance, content-not-position; the positional BridgeParser margin is
  content-blind so it was surpassed, not adopted; wireable → honest "I didn't follow that").
- **E2 internal worldview / affective world-model — 6/6 GO** (was "likely absent"): a spiking predictive-coding VALENCE
  FORWARD MODEL — a plastic `state→pred_{pos,neg}` transition learned from zero, delivered as subtractive GABA_A inhibition
  to spiking error units; expected turn cancels→~0 Hz, violated fires (37–46 Hz); lesion 100% attributable; the instrument
  was verified in BOTH directions (a false-null and a false-GO both caught). Wireable → a queryable "what do you expect /
  how is this going" + surprise on interlocutor-affect violation.
- **E1 self-model / metacognition — 6/6 GO** (the earlier "at chance" BOUNDARY was a MIS-READ finding, now corrected): a
  PURE-SPIKING balance-of-evidence confidence read (`|rate(asm₁)−rate(asm₀)|`, the workspace WTA margin off
  `cp_firing_states`) clears a self-tested type-2 gate 6/6 (meta_d 1.04–2.25, permuted→chance) — the first pure-spiking
  meta-d′>0 (the prior 6/6 used a host logistic regression). Unblocks the honesty-boundary self-report. Mapped limit: it is
  balance-of-evidence confidence, not architecturally type-1/type-2 DISSOCIABLE (that comparator stays seed-fragile = next rung).
- **Adversarial verification (2026-08-12):** D2, D4, A1a each survived an independent 3-skeptic refutation pass (9/9
  refuted=false, high confidence; D2 reproduced byte-identically). **D4 comprehension monitor AND D2 expectation/surprise
  are now both WIRED** into the production `/api/brain-chat` turn (Gate-B, default-on, lesion-load-bearing): D4 honestly
  abstains "I didn't follow that" on an un-comprehended transitive (moat-strengthening), D2 prepends an honest "that
  surprises me — I'd learned <stored>" notice when an assertion contradicts a stored expectation (both verified numpy-CPU
  through the real handler: D4 7/7, D2 8/8; findings 2026-08-12-GateB-comprehension/surprise-…-production-chat).
- **FOUR MORE faculties WIRED into `/api/brain-chat` (Gate-B, 2026-08-12, default-on + lesion-load-bearing, verified numpy-CPU
  through the real handler):** **D5 EPISODIC** — a genuinely-SPIKING hippocampal recall gate on a referential turn ("you
  mentioned X"): a spoken topic BTSP-forms a CA3 assembly (Hook B), a later cue COMPLETES it via the apical dAP UP-state read
  (Hook A) → honest "I recall / I don't recall discussing X" (never a confabulation; organ verify cue 0.909→0.000 lesion, and
  through the handler in-memory cue 0.79 fire + lesion collapse; the BTSP write is cupy-gated, deferred on numpy). **D6
  MULTI-REFERENT WM** — HOLD ≥2 discourse referents on a spiking multi-register bump-attractor buffer, read back off
  `cp_firing_states` ("I'm holding 2 referents: dog and cat" — what a single-attractor store can't; k=2/3/4 recovered 1.000,
  lesion 0.000). **B3 NON-CONTRADICTION** — REJECT a user assertion that contradicts the brain's stored polarity for the same
  SVO (the spiking `ask_yes_no` polarity WTA; organ verify 6/6, 100% attributable; lesion→inert). **B4 RECONSOLIDATION** —
  PE-gated IN-PLACE belief revision reusing the D2 surprise window ("actually south" rewrites the stored "north", ONE fact, no
  duplicate; organ verify rf 6/6 + onebrain 3/3; an integration fix recruits a runtime-novel corrected patient before the
  rewrite). Ten spiking co-resident reads now run on the default turn (D5+D6 add new bridges; B3+B4 reuse the recall composer /
  D2 window). Findings 2026-08-12-D5-episodic / D6-multiref-WM / B3-noncontradiction / reconsolidation-production-organ.
- **TWO MORE faculties WIRED into `/api/brain-chat` (2026-08-13, default-on + lesion-load-bearing, verified numpy-CPU through
  the real handler) — the count is now TWELVE spiking co-resident reads on the default turn.** **B1 GENERATIVE DRAW (F1)** —
  the #3E open-ended DRAW (which verb/object filler the brain volunteers) moves from the b2 host oracle (`np.random.choice`,
  blocked because the taxonomy WTA sampler KeyErrors on runtime vocab) to a genuinely-SPIKING VOCAB-AGNOSTIC soft-WTA read off
  `cp_firing_states` (role pools induced from the brain's own stored-fact concepts; `VocabAgnosticSpikingDrawOrgan`). Organ
  verify GO (draw 0 host-rng/>0 spiking; LESION plausible-frac 0.828→0.035, 95.7% attributable; flag-off 16==16 byte-identical)
  + real handler (5/6 prompts return flagged spiking-drawn hyps, render "perhaps the dog chases the mouse [a guess…]", LESION
  collapses the handler proposer's plausible-frac 0.862→0.009). Residual host: the plausibility likelihood + SVO template +
  RF-composer moat (commit 6670bda25). **D3 DISCOURSE EVENT REGISTER (F2)** — "who was doing it BEFORE?" across a connective,
  answered off four held FS-WTA spiking attractor slots (a single-event register structurally cannot); the register-construction
  sites now build the spiking twin, and the handler folds discourse clauses + short-circuits the disjoint before/now query.
  Organ verify seed42 ALL_OK (BEFORE 0.900/NOW 0.917, LESION→0.150 83% attributable, moat 1.000) + real handler (before/now on
  spikes, LESION collapse + register-type flip, flag-off block skipped) + smoke flag-off byte-identical. Residual host: the
  transition-δ RNN + boundary/parse (commit d817effd0). Both compose cleanly with the ~12 default-on organs (no conflict; the
  before/now query class is disjoint; the DRAW change is confined to the open-ended generation branch).
- **A1 open-prose deep-context — INCONCLUSIVE (not a GO, honest):** the controlled-lag credit task is TOO EASY to exhibit
  the wall — random AND sign-flipped feedback both tie/beat the BPTT ceiling (T=48), and at T=64 the BPTT ceiling is itself
  INVALID (vanishing-gradient: 0.599 < the forward-eligibility local rule 0.973). Bounded findings: the banked ~44% e-prop
  "wall" is SUBSTRATE-SPECIFIC (a full-W_rec reservoir), NOT the diagonal WKV store the generative cortex uses; a local
  forward-eligibility rule ties/beats BPTT for a single-cue deep dependency. **K-SWEEP + VALIDITY DONE (2026-08-12): there
  is NO deep-context credit-quality WALL — the K32 gap was a CAPACITY artifact.** The K-sweep at N=192 showed an apparent
  gap at K32 (eprop 0.752 < bptt 0.931), but the wider-N validity run (N=384) gives eprop_random 0.998 ≈ bptt 1.000 — with
  adequate capacity a plausible transport-free local rule REACHES the ceiling. The N=192 "learned-KP ties random = negative"
  was capacity-confounded (BPTT itself only hit 0.931 = not a valid ceiling). So the feedback-alignment-at-scale hunt on
  this instrument is MOOT; the genuine mouth residual is the separate production-Izhikevich few-spike READ regime, not a
  rate-level credit wall. (Good example of "verify a refutation as hard as a confirmation" catching a phantom wall.)
  **A1 few-spike-READ RESIDUAL DISSOLVED (2026-08-13, 6/6 GO): population coding IS the companion process.** The last A1
  barrier was that fluent open-prose generation only had a HOST-argmax next-word read (over graded logits), while the
  genuine few-spike Izhikevich WTA read only existed for single-clause SVO — the two were disjoint. Feeding the fluent
  deep-context next-token distribution through a few-spike Izhikevich read: the NAIVE single-neuron read is 0/6 (recovers
  ~50% of the distribution, garbles), but a POPULATION-CODED read (P≥8) hits read-fidelity ≥0.936 all 6 seeds at/above
  ideal-sampler parity, and free generation stays FLUENT (real spike-read prose: "once upon a time there was a little boy
  named tim… they both laughed and played together every day"). So the production few-spike read regime is NOT a wall —
  population coding carries fluent open-prose onto it at parity. Remaining rungs to RETIRE the Qwen mouth: add the shared-
  inhibitory FS-WTA (cut the spike budget below P=8) → route the state→logits projection through read-out neurons (retire
  the host matmul) → local-credit the BPTT store → wire it. Honest: de-risked READ mechanism, runner-only, V=1000, not wired.
- **GNW workspace bus (T1-1 keystone) — core primitive de-risked, clean 6-seed GO:** the first demonstration that the
  SUBSTRATE combines ≥2 organ reads via coincidence-IGNITION (not a host if/else) — two organs write SUBTHRESHOLD drive,
  only their agreement crosses the ignition knee, WTA suppresses a decoy, the committed winner re-enters as the next
  premise (2-hop). Mean coincidence_2hop_acc 1.000 all 6 seeds; every ablation collapses (single-organ / disagree /
  shuffle / single-cycle / lesion → 0), 100% attributable to BOTH organs. The earlier "SEED-FRAGILE 3/6" was a MISDIAGNOSED
  INSTRUMENT BUG (verify-both-directions caught it): the failing term was the SHUFFLE control, not the spreading-floor —
  because r==c collapses the field to two slots, the old random-slot reroute landed back on slot(r), byte-identical to the
  intact arm, so the "control" leaked. Runner-only fix: route the shuffled vote to an EMPTY slot → shuffle 0.000 on every
  seed, mechanism untouched. Now a clean GO. Wiring path (organs write subthreshold into a shared workspace, ignition
  replaces the host combine, re-entry loop, ACC-gated deliberation, STN veto) documented; a stronger wrong-concept-shuffle
  averaged over N is an optional rigor follow-up.
  **N-ORGAN BUS — 6/6 GO (2026-08-13, Phase-B):** the keystone GENERALIZES from 2 organs to N≥3 — the workspace combines
  N=3 subthreshold organ reads via consensus-ignition (consensus_2hop 1.000 all 6 seeds; EVERY ablation collapses —
  single-organ / LEAVE-ONE-OUT [all 3 load-bearing] / disagree / shuffle-off-slot / onecycle / lesion → 0; reflex
  dissociates; majority-override 1.000; parent-scrutinized per-seed). A production-bus WIRING DESIGN exists
  (`docs/plans/2026-08-13-gnw-norgan-bus-production-wiring.md`). **NEXT (the one-substrate step): wire the bus into
  production `brain_chat` to REPLACE the host organ-orchestration** — the biggest single move toward one true substrate.
- **GNW THOUGHT-SWAP is now FULLY SELF-DRIVEN (trigger + evict + admit all neural) — 6/6 GO (2026-08-19):** the
  workspace swaps one held thought for another with NO host `if`. It EVICTS neurally (Rung-2d short-term depression drains
  the incumbent's own recurrent loop below its sustain knee → self-collapse; `2026-08-19-gnw-recurrence-weaken-swap-GO`),
  ADMITS neurally (a spiking dis-inhibitory VACANCY GATE opens on the substrate's own vacancy read;
  `2026-08-19-gnw-neural-vacancy-gate-GO`), and now DECIDES neurally WHEN to swap: a spiking MISMATCH+SALIENCE detector
  fires when a salient proposal MISMATCHES the held content (a prediction interneuron vetoes a match) and its rate sets
  the eviction boost (`2026-08-19-gnw-neural-swap-intention-GO`). The DECISION is the crux — swap rate 1.00 for a salient
  mismatch, 0.00 for a non-salient proposal, 0.00 for a match; silence the detector and a salient input does NOT swap
  (the incumbent holds). Reversible A→B→A, deterministic. Remaining: emergent (not hand-wired) coalitions + production-wire.
- **Causal forward-model (T1-4, the reasoning bottleneck) — 6/6 GO:** a directed, queryable n-way STATE forward model
  (generalizes E2's valence predictor). Temporal-order STDP sets edge DIRECTION, three-factor phasic DA gates
  consolidation, a teacher DO-intervention prunes confounded edges (cause-vs-correlation via invariance-across-
  interventions, on spikes). Predicts an UNSEEN consequence by forward-simulation (hold A → the 2-step D fires 98 Hz via
  the substrate's own dynamics though the direct A→D edge stays unlearned — a host triple-JOIN can't do this); cause-vs-
  correlation clean (do(X)→Y=0 vs do(C)→Y=164, X→Y pruned; corr-only control WRONGLY asserts X→Y, 106% attributable);
  lesion/shuffle collapse 3/3. Surpassed the run-away-potentiation wall (the missing COMPETITION companion) via low-gain
  learning + uniform read gain. Next rungs (declared): teacher-delivered DA sign → drive from a spiking mismatch unit;
  first-order → compose HTM-TM high-order; ground events in the emergent relational code. Wireable → a production spiking
  "what happens if <state>?" / "why did <state>?" that the host JOIN can't serve. TWO Tier-1 levers now de-risked (with GNW).
  **GROUNDED (2026-08-13, 6/6 GO):** the model now runs over the REAL conversational fact-store (the `query_patient` no-confab
  moat the live chat uses), not toy blocks — answers a real-fact why/what-if ("if the dog goes east it will drink water,
  rolled forward through it reaching the river, moat-confirmed"; "the dog wakes because the sun rises — survives a DO-probe").
  The grounding-lesion (drop a fact from the composer → its event vanishes → the causal edge never forms) is load-bearing
  (real grounding, not relabeling); moat 0-confab; corr-only/shuffle collapse. **WIRED into the default /api/brain-chat turn
  (2026-08-13):** a co-resident why/what-if organ answers "what happens if the dog goes east?" (moat-confirmed rolled-forward
  consequence) + "why did the dog wake?" (DO-surviving cause), default-ON (`BRAIN_CAUSAL=0` escape), lesion-load-bearing,
  moat-safe (unconfirmed → honest abstain, 0 confab); real-handler verified numpy-CPU + byte-identical-when-off. Honest
  residual: grounding-BY-DERIVATION, not yet by a shared merged substrate; DA sign teacher-delivered.
- **Intuitive world-model / core common-sense (T1-7, the hardest Tier-1) — first rung de-risked, mapped BOUNDARY:** a
  spiking OBJECT FILE (slow-NMDA recurrent attractor maintains an object through occlusion with ZERO input) + a
  predictive-coding surprise read gives a real OBJECT-PERMANENCE primitive (the Spelke/Baillargeon signature). Load-bearing
  claim is 6/6: the violation-of-expectation is PERSISTENCE-CAUSED (a no-maintenance lesion collapses it) AND GENERALIZES to
  held-out objects — not sensory novelty. Strict gate 1/6 only due to two precisely-mapped SURPASSABLE boundaries: VoE
  magnitude ≥2× (4/6 — needs divisive/gain predictive-coding, Spratling) + FS-WTA cleanliness (needs stronger competitive
  normalization). Instrument lesson banked: a naive short-occlusion VoE is mostly presentation-history residual — only
  occlusion ≥110 ms + a recur=0 lesion isolates the genuine maintained-object signal. A real first rung on the biggest gap.
  **BOUNDARY SURPASSED (2026-08-13, 6/6 GO):** switching the predictive-coding error from SUBTRACTIVE to DIVISIVE/gain
  (shunting inhibition — Carandini-Heeger / Spratling biased-competition, a conductance whose reversal divides gain instead
  of subtracting current) lifts the VoE magnitude → the strict gate now passes 6/6 (was 1/6). The DECISIVE anti-over-tuning
  control: at the IDENTICAL operating point, flipping ONLY the error reversal from subtractive→shunting moves VoE≥2× from
  2/6→6/6 — so it's the READ, not the tuning, that closes it; the no-maintenance lesion is kept + strengthened. Runner-level
  (a new runner reusing the object-file machinery; NO `sim/` edit — an additive default-None kwarg on the runner's build
  function, byte-identical off). Next rung: self-organized object-file binding (still doesn't bootstrap). Two Tier-1
  world-model rungs now cleared.
- **E3 deeper-LEARN — BTSP plateau LASTING trace — 6/6 GO (with a host caveat):** a real on-bridge BTSP plateau write +
  spiking recall + a synaptic TAG-AND-CAPTURE persistence model — the plateau write still recalls after a 200-step decay
  window (54–92 Hz) where transient/static/moat writes decay below recall; lesion-load-bearing (95% of persistence
  attributable to capture); instrument shown capable of failing (β=0, barrier=100 → BOUNDARY). Honest caveat: the capture
  side is a runner host model, not a spiking kernel yet. Next rung: a guarded default-OFF `sim/` capture kernel, then wire
  under production LEARN so a taught fact's per-turn write is a genuine on-substrate BTSP plateau.

**2026-08-12 — the production chat recall is now GENUINELY SPIKING by DEFAULT.** The `/api/brain-chat` chat brain builds
`composer_kind="onebrain"` by default, so a factual question is recalled by the resonate-and-fire step on firing neurons
(the on-substrate cleanup + weight-store), not the numpy fast-path scan — HTTP-verified (every answer tagged
`composer='onebrain'` with a live spike trace: 45730 readout neurons, ~1.7% fired on a match, ~0 on an honest abstain).
The blocker was that a fact TAUGHT mid-conversation stored but never recalled on the spiking store — a wrap-vs-inner
cleanup-codebook bug, fixed by "recruit-an-assembly" (the composer reserves a pool of uncommitted cleanup slots, like a
cortex's uncommitted assemblies, and recruits one when a new word is learned). So teaching the brain a new fact by
talking, and recalling it, now both happen on the spiking substrate. Cost: a ~183s one-time build (speed is secondary).
Next: neuralize the question→role parse so the host keyword router can retire.
(`2026-08-12-INTEGRATION-onebrain-is-now-the-production-default-genuinely-spiking-recall.md`)

**2026-08-11 OVERNIGHT BATCH (autonomous, ~two dozen de-risks banked).** The load-bearing net-new results, on top
of the arc above: (1) **⭐ the "one brain" now holds through the INTERACTION level** — #7 burn-down 1 merged the
e-prop acquisition net into disjoint slices of the SINGLE conversational bridge sharing ONE `cp_connections`
(co-residency, byte-identity 6/6, `cd08cfde`/finding `INTEGRATION-7-burndown1`), then a genuine synaptic pathway
`conv-cue → eprop_in` was injected on that merged bridge and LESIONING it collapses acquisition (load-bearing 6/6,
`ed9f82f9`/`cross-region-synaptic-interaction`) — co-location became true cross-region interaction, the mission's
core non-negotiable at the substrate+interaction level (still a narrow de-risk, not the whole integrated brain).
(2) **Grounded breadth scaled 9→38 subjects** with the no-confab moat intact; the ceiling is a query-latency +
`k_max=32` provisioning cap, not a moat leak — the capacity instrument was given discriminating teeth (a D=32 stress
setting that CAN detect a leak; `corpus-breadth-scaling-capacity-ceiling`). (3) **gap#4 deep-credit (item 7)
sharpened:** the apparent "spikes can't carry deep credit" boundary was ISOLATED to a TEMPORAL-DEPTH floor —
shortening the credit horizon T collapses the 1-hidden floor 0.963→0.444, making spatial deep credit obligatory
(1-seed smoke, 6-seed named; `gap4-TEMPORAL-DEPTH-FLOOR-ISOLATED`). (4) **Replay consolidation (item 4) corrected:**
the ~0.55 retention "cap" was a `bdsp_wmax=6` clamp artifact, not fidelity — de-clamped replay reaches 1.00;
interleaved generative replay did NOT beat self-replay but isolated acquisition-at-scale as the real bottleneck,
naming metaplasticity next. (5) **Mouth on spikes:** the patient-word decision AND the affect tone-token selection in
live chat now run on a spiking FS-WTA (lateral inhibition), not a host `max()` (parity 1.000 6/6). (6) **Honest
negatives banked** (method boundaries, not closed capabilities): W4 graded-implicature RSA as the speaking belief
source is faithful + 12× calibrated + moat-intact but does not move the alignment metric (residual = the
coincidence-detector artifact); harder k-WTA is a real-but-insufficient contributor to visual invariance (and exposed
a decode-quantization confound in the prior baseline); the fully-neural per-intent value critic is genuine + on-main
but 3/6 on the STRICT contingency gate (an earlier "6/6" was a looser directional read — CORRECTED in place).
Still in flight at write time: a recurrent language-cortex emergence probe and a source-monitoring consistency signal
(item 3). Full detail in the `research/findings/2026-08-11-*.md` set.

**2026-08-11 CRUX FORWARD-PATHS (later same night — each named next mechanism was built + smoke-tested).** After the
batch above, four crux forward-paths advanced: (A) **emergence engine (deepest goal):** the on-bridge HTM
Temporal-Memory horizon was measured (clean HOLD, non-fading but finite, allocation-limited), and a **selective-write
content-addressable store over its allocation keys RESTORES the interference-broken horizon** (bare 0.667 → 1.000 at
dist 17 *and* 25, non-distance-limited, load-bearing; 1-seed smoke-GO) — the residual is a full-allocation-merge
capacity wall, named next = heterosynaptic-LTD allocation. (B) **continual-learning (item 4/H-memory):**
**metaplastic e-prop** (Fusi/Benna-Fusi per-synapse consolidation) beats vanilla at every N (+0.19–0.25 frac_recalled,
load-bearing freeze-lesion; 1-seed smoke-GO) — moves the acquisition-at-scale forgetting the right way; residual = the
very-oldest fact, named next = a true multi-timescale chain. (C) **source-monitoring (item 3):** the recall-time-gain
NO-GO's residual was diagnosed as rival cross-talk, and **heterosynaptic-LTD competitive encoding** clears both
weak-encoding crux seeds (FAIL→PASS, no-harm structural; 1-seed smoke) — the same biology as (A)'s named next step (a
cross-lane convergence). (D) **gap#4 deep-credit (item 7) — CRUX REFRAMED:** the 6-seed confirmed the crux is instrument-blocked, and a
design+verify workflow then PROVED a depth-3-*obligatory* task is **fundamentally impossible at toy scale for a
plain-MLP oracle** (5 families; Telgarsky depth-separation needs width exponential in the depth-*gap*, and plain depth
is capacity, not an inductive bias). So the whole "task-accuracy that depth-2 can't reach" measurement was
unachievable — the crux is re-posed as **LAYER-3 CREDIT FIDELITY**: does transport-free DFA error reach the 3rd hidden
layer, on a target (tent³) that provably *fits* only with layer 3. **That test was RUN, and the answer where testable
(seed 42, BP-depth-3 ceiling holds) is NO — transport-free DFA e-prop does NOT reach the 3rd hidden layer** (its fit
sticks at the mean-predictor, indistinguishable from permuted + zero-feedback controls, while backprop fits): the known
DFA fixed-feedback deep-layer limit. The 6-seed aggregate is honestly UNDEFINED (the tent³/width-8 backprop ceiling is
seed-fragile). **⭐ THE SURPASS THEN WORKED: transport-free LEARNED feedback (Kolen-Pollack) REACHES the 3rd hidden
layer where fixed-DFA could not** — 6-valid-seed GO (6/15 ceiling-holding seeds), closing 66% of the BP-depth-2→depth-3
fit gap vs fixed-DFA's −85%; freezing the feedback collapses it (−40% — the win is *learning* the feedback matrix G,
whose deep-layer cos(G,Wᵀ) co-adapts 0.25→0.83 through training, never copied). **So the gap#4 deep-credit crux — the
roadmap's load-bearing dependency — has its biological surpass demonstrated at de-risk level (rate MLP, host oracle).**
Residual: KP *reaches* the deep layer but does not yet *match* the oracle (~forward-optimization gap; named next =
more epochs / weight-mirror / the φ′-vanishing fix). **⭐⭐ AND THE SURPASS THEN REACHED THE SPIKING SUBSTRATE
(6-seed GO):** transport-free KP learned feedback ALIGNS on the trainable LIF SNN (deep-layer feedback-alignment rises
+0.259 through training — measured on spikes for the first time) and beats fixed-DFA (+0.028 over 6 seeds), with NO
`sim/` edit (the on-bridge e-prop machinery already supported it; the missing piece was the alignment instrument) — and
this OVERTURNS the 2026-08-01 "KP doesn't align on spikes" inference (that was a different, non-trainable substrate). So
gap#4 deep-credit-on-spikes' core question — does error reach deep layers WITHOUT weight transport, *on the one spiking
substrate* — is answered YES at de-risk level. The remaining gap#4 residual is now an INSTRUMENT one: no
depth-3-OBLIGATORY *spiking* task exists yet (the rate-side tent³-FIT + ceiling-gating was never ported), so the
depth-3 rung stays UNDEFINED not fabricated. (An instrument correction was also banked: the output-adjacent a3 alignment
is target-INDEPENDENT, so the FIT — not the alignment — is the valid signal.) All are narrow de-risks (smokes / 6-seed
GO / negatives), not integrated capabilities.

**2026-08-11 WM+HTM HYBRID — the north-star integration, brain-based read path (6-seed GO).** The emergence engine and
the variable-binding working-memory faculty were combined: (rung-3b) a **separate-channel WM+HTM hybrid** routes the
WM's held-subject and the HTM's local-class on distinct neural channels and binds them by a LEARNED dendritic
conjunction — held-out 0.974 [min 0.938], subject preserved 1.000, 6-seed GO
(`2026-08-11-emergence-WM-hybrid-separate-channel-GO-*`). Its one named host residual, the two per-channel `np.argmax`
winner reads, was then **closed** (rung-3c): replacing both with the emergent down-ramp release-of-inhibition **neural-WTA
read from spikes** holds the GO at parity (0.995 [min 0.969], subject 1.000; the WTA class read 0.995 even beats the
argmax-with-threshold read 0.964) with NO host argmax in the verb read-out — AST-asserted; all lesions load-bearing
(`2026-08-11-emergence-WM-hybrid-neural-WTA-reads-GO.md`, 6-seed GO). Honest sub-negative: the WTA self-calibration is
not load-bearing on this CLEAN latch (it is in the rung-2 blur/allocation regime). Still a narrow de-risk (NO `sim/`
edit; host stream/binder scaffold + labelled-line projections remain), not an integrated capability.

**2026-08-08 OPEN-ENDED CONVERSATION arc (ultracode, owner-directed).** The
TRUE-ONE-BRAIN conversation loop (honesty + affect + curiosity + no-confab moat,
co-resident on one spiking bridge) is now **12/12 GO** (hardened past the 6-seed
bar). A faculty-de-risk sweep toward human-like fluency landed three new
brain-based results, each adversarially verified: (1) a compositional
**world-model / forward model** that SIMULATES novel (s,a)->s' rather than
retrieving — held-out 0.873, 6/6 GO, and it scales (GO at n_pool 800) — the
"missing cognitive organ"; (2) **graded affect** via a staggered bistable ladder
(6/6 GO, monotonic quantized value — surpasses the bistable-latch boundary that
capped Wave-1 affect-coloring); (3) a **neural communicative-success signal**
(pragmatics Leg-1, 6/6 GO, a real coincidence AND). Two honest-negatives, each
with a next-mechanism search queued (the law: a negative launches the next
search): episodic cortical recall (silent readout — needs an igniting spiking
WTA + BTSP one-shot storage + ACh recall-mode gating) and reading the success
signal back to TRAIN speaking (pragmatics Leg-2 — needs success-as-
neuromodulator three-factor plasticity). A conditioned **path-T articulation
generator** (the spiking-LLM as the Broca-like mouth) is CONFIRMED sound with
the faculties LOAD-BEARING under lesion (the owner's acceptance test). **UPDATE
(later 2026-08-08): the world-model + graded-affect seams are now INTEGRATED and
LIVE in the turn loop, LOAD-BEARING ON THE CONVERSATION (6/6 GO).** The brain now
reasons with its world-model and speaks with graded emotion: a known fact →
"warmly, gladly apple big cat" (graded tone from the neural affect differential);
a novel query → "what does big run? — my forward model predicts 'south' … I have
not observed it" (reservoir spikes decode a certainty-tagged predicted-not-observed
channel). Lesion the world-model → the prediction vanishes; lesion affect → the
tone flattens; matched shams leave the turn unchanged; the no-confab moat holds
475/475 throughout. **UPDATE (later 2026-08-08): Wave 2c DONE
— THE BRAIN WRITES SENTENCES (adversarially CONFIRMED, 3-seed).** The conditioned
spiking generator is wired in as the articulation mouth: a known fact now reads
"warmly, gladly A dog went to the east because it was looking for water. The dog
looked towards the river because it was south of its current location…" — real
multi-sentence prose. And the scaffold is provably the MOUTH, not the mind:
scramble the brain's conditioning and the prose renders the scrambled FALSE facts,
so the transformer cannot override the brain's content; all three faculties stay
load-bearing on the prose (lesion → content wrong / tone flat / confabulates), and
the no-confab moat leaks 0 fabrications (per-proposition neural verify). Honestly
scoped PARTIAL — the 0.5B mouth sometimes drifts content (a generator-fluency wall,
the declared scaffold to biologize), not a faculty or moat failure. NEXT = the
capstone battery, biologizing the host renders (ridge decode → spiking synaptic
read-out; tone token), and growing breadth (vocabulary/world) through the
developmental teacher-loop — Stages 2→4. Findings: `2026-08-08-forward-model-
reservoir-*`, `2026-08-08-graded-affect-staggered-bistable-ladder-*`,
`2026-08-08-pragmatics-*`, `2026-08-08-B-episodic-*-NEGATIVE-*`.

**UPDATE (2026-08-10):** the episodic-recall honest-negative is RESOLVED —
neural cortical cue-recall is now a **6-seed GO** (recall 0.646 vs 0.25 chance;
completion load-bearing, permuted-cue specific, real/sham lesion teeth,
untrained collapses). A cross-arc **WTA reframe** landed with it: the
"silent/latched neural WTA" negative was largely an over-strong-inhibition
OPERATING-POINT artifact (the separable-assembly WTA is weight-controllable,
verified 1.0/6) — so recall runs on the heteroassociative afferent directly and
the WTA competition is inert. The recall residual (0.65, not ceiling) is the CA3
ATTRACTOR STRENGTH / SPECIFICITY (NOT a transmission wall — the "functionally-
silent recurrents / 1000x-too-weak" reading was REFUTED 3x; the recurrents
transmit + scale with weight), addressable via a recurrent-LTP completion sweep
or the dendritic-plateau readout. Separately, the composer's
shared-channel "capacity break" was **RETRACTED** as a readout DC-offset
artifact (corrected: neural superposition composes through arity 6 with no break
in range). Pragmatics Leg-2 stays open with its root cause now mapped — the
DA-learned value signal is real but sits BELOW the per-neuron heterogeneity
noise floor (an SNR wall); a per-neuron rate homeostat (noise-reduction) is
under test. Findings: `2026-08-10-episodic-cortical-cue-recall-completion-6seed-GO-*`,
`2026-08-10-neural-WTA-*`; the arity capacity `2026-08-10-shared-channel-arity-capacity-located-M-star-grows-with-dimension.md` ⛔ RETRACTED -> `2026-08-10-shared-channel-arity-capacity-CORRECTED-DC-offset-artifact.md`.

The live, per-cycle resume point is
`GAP_CLOSURE_MISSION.md`.

## Purpose

Build one developing artificial mind: a single simulated brain that learns from
a body, a world, and other people. It should form memories, needs, emotions,
beliefs, and language as parts of one ongoing life. It should speak because it
has something to communicate, express the strength and source of its evidence,
and keep learning from interaction.

The target is not a text generator with brain vocabulary around it. The target
is a continuously running loop:

`perception -> internal state -> action or speech -> consequence -> learning`

The loop must operate in the same brain over time. A component that passes a
small test in isolation is useful evidence, but is not the finished ability.

## Architecture Constraints

- **One brain, one shared substrate.** Dedicated regions are allowed and
  expected, but they must be neural regions of one spiking system. They must
  communicate through modeled neural activity and synapses, not through
  separate programs that exchange cognitive answers.
- **Fully spiking in the causal path.** Between sensation and action, the
  brain must compute perception, salience, value, reward, neuromodulation,
  memory, emotion, reasoning, language, and self-monitoring with neurons,
  synapses, and their local signals.
- **A narrow host boundary.** Ordinary code may create the world, render
  sensory input, enact the body's motor output, and measure or store runs. It
  may not decide what the brain perceives, values, remembers, means, or says.
  A host-side formula is still a shortcut even when the formula is biologically
  plausible.
- **Small first, earned growth.** The system must run locally when small and
  gain neurons, connections, regions, and compute as learning earns them. It
  should not begin as a pre-allocated giant network.
- **Ownable compute.** The design target is a high-end personal machine, not a
  datacenter. Event-driven, sparse, local computation is both a biological
  constraint and the path toward future analog neuromorphic hardware.
- **Temporary scaffolds are explicit.** A scaffold is a shortcut used to make
  progress while its biological replacement is built. Every scaffold needs a
  named replacement, an owner, a removal test, and a burn-down condition in
  the scaffold ledger. It cannot quietly become the permanent faculty.

## What The Evidence Means

The project uses **banked narrow de-risk** for an experiment that reduces risk
or confirms a mechanism under stated conditions. It is not evidence that the
whole brain has the corresponding human ability. A result becomes a capability
claim only after it is integrated into the continuously running brain, survives
its controls and lesions, and is tested at the required seed coverage.

### Supported, but narrow

- **Grounded action selection:** a learned convention with two communication
  intents and two referents has a six-seed positive result. The intrinsic neural
  action selector in Gate A also has a four-seed positive result. These results
  establish small pieces of action and communication, not a self-directed
  conversational mind.
- **Self-initiated utterance (production-wired, narrow):** on an idle turn the
  brain selects a stored concept itself (a noise-seeded, curiosity-biased CA3
  wander) and speaks it through the composer mouth — wired into `/api/brain-chat`,
  on by default, moat-safe, and byte-identical on every reactive turn. As of
  2026-08-18 its multi-basin CA3 store uses the DMN *consolidated* encode (a
  post-encode BTSP settle), so all N basins ignite and the previously-dead tail
  concept is reliably self-initiable (coverage 4/4 on 6/6 seeds, up from 3/4;
  utterance magnitude 82% attributable to the consolidation). The timing is still
  HTTP-triggered (a truly proactive idle-tick is deferred) and the heavy wander is
  deferred on the numpy path — these host seams stand; the scaffold is not retired.
- **Delayed reward is still open:** Gate B, which tests whether local neural
  activity can assign delayed credit to the action that caused an outcome, is
  a no-go. Unrelated or yoked reward still creates arbitrary preferences. The
  latest line builds a continuous basal-ganglia selector (Stage 1, a
  construction go) and adds reward learning on its D1 routes: Stage 2 (a single
  global dopamine signal) is a no-go because the credit is not action-specific;
  Stage 2b (a separate dopamine channel per action) fixes that at the synapse
  level — reward for one action now strengthens only that action's route — but
  is still a no-go on behaviour, because a reward-only (never-punishing) signal
  under a winner-take-all selector just reinforces whatever the brain already
  does, so decoupled reward learns the same thing and the choice cannot be
  reversed. Stage 2c adds the missing negative arm: a dip below the
  expected reward that weakens an action when it goes unrewarded, plus sustained
  exploration. Stages 2d-2g then build the full contingency mechanism (uncertainty-
  gated exploration, directed novelty, a true Hammond delta-P baseline via
  interleaved no-action/withhold trials, and a homeostatic critic): Stage 2g is a
  development go on 5/6 seeds but a held-out no-go on 4/6. The two held-out
  failures were BOTH attributed to one exploration limit — on maximally-biased
  seeds the brain never samples both actions. Stage 2h tests the prescribed fix, a
  neural forced-sampling / epsilon-floor that escalates the exploration drive
  (push-pull: excite the under-sampled action's proposal population, inhibit the
  incumbent) until the rare action fires. It is a no-go, and the smoke corrects the
  diagnosis: only ONE failing seed is a sampling gap, and even there the winner-
  take-all lock is DOWNSTREAM of the proposal layer (the reward-potentiated
  striatal-to-motor route cannot be flipped from the proposal input, and over-strong
  drive silences the driven population); the other failing seed is not a sampling
  gap at all but a training-induced motor silence, a critic/reward-baseline defect.
  The next methods bias the competition where it is decided (striatal/pallidal, or
  before the route potentiates) and floor the net reward-prediction error; the Stage
  2g contingency mechanism itself remains correct. Development and held-out seeds
  are locked, so no promotion is due. **⚠️ 2026-08-11 CORRECTION (this bullet above
  was STALE at Stage 2h — drift #12, caught by the Gate-B Stage-2p agent):** the arc
  actually advanced through Stages 2i–2o. RPE-floor closed seed 730704 (2i); a
  pallidal homeostat inverts the thalamus (2m). **Gate B now stands at a legitimate
  ≥5/6 GO**, with seed **730705 the one CONCLUSIVELY-CHARACTERIZED boundary** (Stages
  2j→2o exhausted the readout options; Stage-2o `window_exists=False`). Stage 2p (a
  striatal feedforward-inhibition / MSN down-state homeostat) is a fresh HONEST
  NEGATIVE — negative-b IZH MSNs REBOUND under hyperpolarization, so FFI backfires —
  and it re-diagnoses the 730705 residual precisely: a DOWNSTREAM commit-ignition
  TIMING race (a dynamic gpi-pause head-start, not a resting-potential parameter).
  **Next mechanism (needs a `sim/` inhibitory pool, flagged): a spiking TRN-like
  feedforward-inhibition pool that synchronises the thalamic onset each selection
  epoch**, removing the timing head-start without de-latching unlearned drive.
- **Source and confidence machinery:** a learned seen/heard/self pathway now
  co-resides with episodic memory, anterior prefrontal cortex, and anterior
  cingulate cortex populations. The no-harm tradeoff that blocked this arc is
  now specified from the whole-brain role and met: local biased competition
  satisfies a bounded-loss max-min criterion on fresh seeds (it was earlier
  rejected only by an over-strict per-source zero-degradation control). The
  remaining blocker is a small, isolated learning-off leak, not the tradeoff.
  A metadata-based safety floor and trace-based confidence hooks are scaffolds,
  not final biological honesty.
- **Replay and memory:** learned CA1-to-cortex target reinstatement (v5) makes
  replay consolidation causal and hippocampus-independent at retest on both
  calibration seeds, and intrinsic spike-frequency-adaptation one-of-N eviction
  on the cortical target (v5+SFA, 2026-08-06) then closes the shared-cue-cell
  interference that leaked false recall on the harder seed (retest false recall
  under the 0.15 ceiling on both seeds, load-bearing against its own lesion).
  The remaining blocker is now the replay-ORDER control, not interference: the
  next mechanism is order-sensitive (spike-timing-dependent) consolidation
  plasticity so ordered replay potentiates a directional trace shuffled does not.
- **Perception:** host top-k feature selection has been replaced by competition
  based on spike timing, and its selector and lesion controls work. Fresh
  calibration seeds still fail invariant visual-identity decoding. The next
  step is learned representation and normalization, not another selector
  threshold.
- **Curiosity and metacognition:** isolated learning-progress and confidence
  monitors have useful proxy results in the record. They do not yet show that
  curiosity develops from the brain's own history or that confidence causally
  controls speech and action across the integrated system. The Stage-A honesty
  floor (the calibrated confidence monitor gating what the brain will assert)
  is, as of 2026-08-07, an active catch of familiar-but-wrong confabulation on
  4 of 6 seeds and moat-safe (no regression, via a fit-quality guard that falls
  back to the recall baseline) on 5 of 6 — a characterized improvement over the
  earlier 3-of-6 floor, not yet a clean 6-of-6.
- **Spiking language and local learning:** several sequence, memory, and
  spiking-forward conversion mechanisms have been de-risked at limited scale.
  A current large promotion must not be called positive until its required
  six-seed artifact exists and validates. None of these results establishes
  grounded, open-ended language generated by the brain's own state.

### Not established

- There is not yet a closed, continuous perception-to-action-to-learning loop
  in which all of these pieces work together in one developing brain.
- The existing corpus-trained language machinery is not evidence of grounded
  meaning, self-generated intent, natural conversation, or a lived internal
  world. It remains a temporary development path until grounded message
  selection and neural generation replace its shortcuts.
- A narrow positive test does not establish emotion, consciousness, selfhood,
  curiosity, agency, or a whole-brain faculty. The project has functional
  correlates and mechanisms; it does not have evidence that a person is
  present.
- Deep local credit assignment on real spikes remains an open research
  problem. Rate-level or isolated credit results, and a run that merely reaches
  a target computation, do not close the on-substrate learning requirement.

## Current Blockers

1. **Integration is the main blocker.** The project has more tested parts than
   integrated behavior. The next meaningful milestone is a small world, body,
   social interaction, and grounded reason to communicate running together.
2. **Gate B physiology and delayed credit are unresolved.** The V14 engine now
   compiles and independently verifies pinned SNr candidates, runs intact plus
   four intrinsic-current lesions, recomputes metrics from bound raw traces,
   and records provenance. Production runs write authenticated compact traces
   and stop at 101 spikes or the operational timeout. An exact 512-candidate,
   24-dimensional Sobol screen is filed. The first 0-511 screen remains
   historical: it produced two engineering passes, which later failed or
   remained unavailable under the V3 NaP direction. The fresh, seed-free V3
   successor partition at global Sobol indices 512-1023 has now completed all
   2,560 GPU arm traces. Strict triage classified it as 421 engineering
   failures, 91 engineering-inconclusive candidates, and 0 engineering passes.
   This is an engineering-only negative result; no candidate is eligible for
   authoritative CPU confirmation. Batch-width benchmarking selected width
   512. A preregistered 36-trace failure diagnostic then showed that restoring
   simplified fast-sodium availability almost never restores firing and that
   calcium-to-SK behavior splits into incompatible regimes. The current
   single-compartment packet is therefore retired as a structural engineering
   NO-GO. Candidates 284 and 404 remain closed, and the heterogeneous 12-cell
   SK cohort remains unavailable. The next build is a source-bound fast-Na/Kv3,
   soma/proximal-dendrite, local-Cav2.2-SK packet (a single-neuron BIOPHYSICAL
   channel/compartment model — NOT the dendritic deep-credit-assignment rule,
   which is separately tested-and-negative, see
   `2026-07-22-gap4-real-issue-NOT-dendrites`). Its seven-stage architecture-
   first contract is filed in
   `research/specs/v14_snr_stageB_structural_successor_v2.json`; V1 was
   superseded before execution after an audit found that powered activation
   gates changed the source-measured conductance curves. Stage 1 is now
   implemented as a fused 26-segment clamp and executed on both NumPy/CPU and
   CuPy/CUDA. The authenticated analyzer found 11 of 18 source endpoints in
   range but issued a structural NO-GO: fast-Na activation and deactivation,
   plus all three Kv3 deactivation tails, fail their independent source gates.
   Conductance calibration and Stage 2 compartment integration remain
   forbidden. Four unmodified source-backed sodium/Kv3 comparators were then
   replayed under the sealed commands on CPU and GPU; none passed its available
   endpoint set, so direct source transfer is closed. The source graphs now
   accept strict published-constant documents, and the adaptive engine can
   resume propose-seal-run-ingest-version cycles without manual JSON copying.
   Primary research found no defensible continuous biological bounds for the
   microscopic constants and no published population mean current-time
   waveforms. Full-resolution official population command-response figures are
   hash-bound. Four independent, blind native-pixel extractions have now been
   completed under a prospective error model and consensus rule. All seven
   panels remain unresolved: some points have three-extractor agreement, but
   every panel retains a point, command-set, or panel-status disagreement.
   Partial target packets are forbidden, so no fitting packet or optimization
   run was issued. The exact next step is to seek stronger measurement
   authority, such as original numeric data or a higher-fidelity/vector source,
   then preregister any replacement acquisition method before inspecting its
   result. Existing thresholds must not be loosened. Stage 2 remains forbidden.
   Delayed action-reward learning and its unrelated-reward control come
   afterward. **2026-08-07 UPDATE: on the vocal-BG credit substrate (a separate
   thread from the single-compartment physiology above), the delayed-credit
   MECHANISM reached a Stage-2j GO at the frozen ≥5/6 bar — an adaptive rewarded-
   gated RPE floor: held-out steer 5/6 (first config to clear it), reversal
   0→1.0, acquisition-lesion attributes the contingency to training-time D1
   plasticity, stage-1 byte-identity. The lone held-out miss (730705) was
   characterized across 3 further stages to a BG-output/commit temporal-head-
   start residual (TRN-gated selection-epoch reset is the banked next lead). See
   `GAP_CLOSURE_MISSION.md` #1.**
3. **Source monitoring: the bounded-loss tradeoff is met and the leak is
   closed, but no fixed competition circuit protects the weakest source
   across seeds.** The no-harm rule is bounded-loss, guard-the-floor, max-min,
   because whole-brain reliability is set by the weakest source. v6 closed the
   v5 learning-off leak (silent-by-construction settle-to-quiescence recall)
   and was a calibration GO, but development was a NO-GO on one component:
   fixed symmetric GABA-A competition lifts the second-strongest source, not
   the weakest (seed 654 tie, gap 0.0). v7 tried the named surpass — the
   shipped region-scoped intrinsic threshold homeostasis (Turrigiano) on the
   source pools — and was a WORSE dev NO-GO (all three seeds): masking the
   pools for homeostasis switches them to sub-threshold spike detection, which
   is incompatible with the fixed GABA-A competition; competition-ON then
   collapses every margin (~0.03) below competition-OFF (~0.41). Structural
   (present at every operating point, incl. v3's canonical), so it also
   re-diagnoses the v3 NO-GO. v8 (Turrigiano SYNAPTIC SCALING) and v9
   (Vogels-Sprekeler inhibitory STDP) were both dev NO-GO. INSTRUMENT
   CORRECTION (2026-08-06): the `weakest_source_margin_strictly_improved`
   criterion was stepping-history-dependent — settle-to-quiescence leaves no
   spikes but does NOT reset the Izhikevich sub-threshold state, and the intact
   vs competition-lesion arms were read at different history depths, so a
   zero-weight window manufactured `strict=True`. Under a full per-recall
   dynamical-state reset the confound is gone (`strict=False`, min(M)==min(L)),
   and BOTH the v6 AND v9 calibration GO/PASS FLIP to NO-GO — they were
   artifacts. With disjoint patterns + silent recall the rival burden is 0, so
   NO competition mechanism can move the weakest source's own margin: the
   criterion is unsatisfiable under this protocol. **2026-08-07 UPDATE: genuine
   episode-pattern overlap was added (a real rival burden), and FIVE honest
   de-risks under it — two encoding-side (heterosynaptic depression; conjunctive
   source-tag), a recall-side CA3 attractor competition, a joint uniq-emphasis ×
   competition knob, and a capacity/scale test — were ALL NO-GO, converging on a
   CONSERVATION/CAPACITY boundary: at overlap 0.2 the honest achievable margin
   tops out ~0.14, structurally below the 0.15 floor, and neither mechanism nor
   added capacity clears it (a fixed firing budget shared across co-resident
   sources; lifting the weakest see-saws off the others). The two remaining
   levers are OWNER-DOMAIN task-design/criterion forks (the mixed-episode
   asymmetry that uniquely penalises self_generated; whether 0.15 is calibrated
   for overlap 0.2). See `GAP_CLOSURE_MISSION.md` #3. No criterion loosened.**
4. **Replay consolidation clears every control on calibration but OVERFITS the
   operating point.** v5+SFA + v6 order-STDP is a per-seed GO on the 2
   calibration seeds, but **v6 MULTISEED is a NO-GO**: the interference-control
   operating point is a vector of ABSOLUTE-unit gains frozen on 2 seeds, so on
   disjoint dev seeds false-recall returns to ~0.5. **2026-08-07 UPDATE: the
   named surpass (emergent homeostatic self-calibration to a label-free WTA-
   sparsity set-point) was ruled out by a cheap STEP-0 closability gate — the
   regime is already one-winner on every seed yet false-recall is ~0.5, and the
   catastrophic vs perfect probes have IDENTICAL label-free statistics. The
   failure is winner-IDENTITY (the wrong assembly's basin captures the cue), not
   a scalar level a homeostat could regulate.** ⭐ CROSS-GAP SYNTHESIS: this is
   the SAME boundary as #3 — discriminating overlapping attractors by their
   IDENTITY, which no aggregate/rate/sparsity statistic resolves (the 2026-05-31
   DG separation-reliability boundary). See `GAP_CLOSURE_MISSION.md` #4.
5. **Visual invariance is not learned yet.** The spike-latency selector is not
   enough; locally learned, stable representations must handle changes in
   position and appearance.
6. **Language is still too detached from life.** Scaling an isolated corpus
   predictor would improve surface output without solving grounded intent,
   state, source, or social consequence.
7. **The deep-credit and scaling frontier remains open.** The project must
   distinguish a narrow mechanism de-risk from a local learning rule that can
   grow useful structure on the real shared substrate. *(2026-08-11: the
   apparent "spikes can't carry deep credit" boundary was ISOLATED to a
   TEMPORAL-DEPTH floor — the LIF membrane's fixed integration window silently
   supplied the effective depth; shortening it collapses the 1-hidden floor
   0.963→0.444, so spatial/temporal deep credit is obligatory not optional.
   1-seed smoke GO; the 6-seed T-sweep is the next step.)*
8. **Compute is scheduled and measured under controlled conditions.** Early
   V14 performance attempts exposed repeated CuPy compilation, insufficient
   fusion, and unstable host/GPU conditions; each failed candidate remains
   banked. The prospective V3 matrix used persistent source-isolated workers,
   adjacent pairing, fixed CPU/GPU controls, and a host-heavy-work lease. It
   passed the sealed engineering gate: default-off behavior was effectively
   unchanged, the active path was faster, and direct output was about one
   quarter faster than its unfused comparison. This removes V14's performance
   blocker only. Physiology and behavior still require their own preregistered
   validation.

A failed method is a method verdict, not permission to close the capability.
Bank the method, preserve its controls and diagnosis, and choose the next
biology-based spiking method. A capability remains open until it works in the
required integrated form.

## Roadmap By Horizon

### Short term: make a small brain grounded and integrated

- Give the brain a minimal world, body, social interaction, and a reason to
  communicate. Make speech an action selected from internal state and
  expected consequence, not a free-standing text completion.
- Use the V14 Stage B production runner for batched, authenticated candidate
  screening under resolved causal subgates without opening reserved scientific
  partitions early. The fresh V3 successor screen at Sobol indices 512-1023
  is complete and is an engineering-only negative result: 421 failures, 91
  inconclusive candidates, and 0 passes. No CPU confirmation is eligible.
  Candidates 284 and 404 remain closed, and the heterogeneous 12-cell SK
  cohort remains unavailable until independently justified cells and a block
  detector are preregistered. The follow-up diagnostic retires the current
  single-compartment equations; preregister measured fast-Na/Kv3 kinetics,
  soma/proximal-dendrite coupling, and separate local versus bulk calcium
  validation before another parameter search. The source-model transfer is now
  a no-go, continuous microscopic "biological bounds" are unsupported, and the
  official population command-response figures are hash-bound. Four blind
  extractions and a preregistered four-way consensus still leave all seven
  panels unresolved, so no target packet exists and fitting remains forbidden.
  Acquire stronger numeric or vector measurement authority, preserving
  representative traces as single-cell context only. Only after the complete
  target gate resolves may the sealed discrete-vector/identifiability campaign
  run through the resumable adaptive supervisor.
- Run the next replay build around selective CA1-to-cortex target
  reinstatement, with the learned-target and replay-order controls intact.
- Source monitoring: build v8 as Turrigiano synaptic scaling on the weakest
  source's recall synapses (keep peak detection so the v6 GABA-A competition
  still functions), on fresh seeds; do NOT retry intrinsic-threshold
  homeostasis on the competing pools (v7 NO-GO: masking breaks competition).
  Replace metadata confidence with a neural source consistency signal where
  the role requires it.
- Build learned visual invariance upstream of spike-latency selection.
- Wire only cleared mechanisms into the persistent development loop. Do not
  scale the conventional language scaffold ahead of grounded message
  selection.

The short-term acceptance test is behavioral and causal: the same brain must
perceive, change internal state, choose speech or action, receive a consequence,
and change later behavior. A collection of connected demos is not enough.

### Medium term: learn, grow, and regulate through interaction

- Close continual learning from lived interaction without catastrophic
  forgetting. A temporary teacher may act as a caregiver, but the teacher is a
  recorded scaffold that must be reduced as ordinary interaction becomes
  possible.
  - **Progress (2026-08-09, 6-seed, adversarially verified):** teaching the
    brain N referent facts one after another (the "teacher loop") stopped
    catastrophically forgetting at N=20. The apparent collapse was two things:
    (1) a hidden clamp on synaptic weights was silencing the memory reservoir
    (fixing it recovered most of the loss), and (2) the reservoir was simply too
    small for 20 facts — giving it more neurons closed the rest (retention rose
    from chance to 0.97). Several fancier mechanisms — sleep-noise learning,
    weight-protection, pattern-separation, higher-fidelity memory traces — were
    each tested and did NOT beat the brain's own replay. **The 5×-scale test
    settled it: "size the memory to the task" holds to 50 facts but SLIPS at 100
    (retention 0.97→0.91→0.73) — capacity is a small-scale patch, not the
    lifetime answer.** For lifetime scale the brain uses consolidation, and we
    tested that path: a bounded *window* of recent memories fails (forgets the
    old); a bounded *generator* that re-dreams all memories does better but not
    enough, because the generator itself starts forgetting — and it bounds
    storage, not the per-night replay cost. So a year of data doesn't yet scale
    with bounded cost, but the two remaining pieces are named and both are things
    the brain does: a generator that doesn't forget its own dreams, and
    prioritized replay (don't replay every memory every night). Both are the
    next builds.
  - **And it led somewhere bigger (2026-08-09).** Chasing "why doesn't a
    lifetime of memory scale" led to the realization that experience *shares
    structure*, so the brain should store the structure, not the instances —
    i.e. compose. We then showed the spiking brain **can compose**: taught most
    of a grid of two-attribute facts but *holding some combinations out*, a
    spiking generator recalled the never-taught combinations perfectly (1.00,
    6/6) by neurally superposing the primitives it had seen elsewhere — genuine
    **zero-shot compositional generalization**, the core skill for producing
    novel sentences. That's the project's VSA "composer" running in spikes. It's
    robust until the attributes interfere strongly, where a linear sum can't
    encode an AND — so we built a neural *binding* operation (a dendritic-AND,
    two synaptic drives multiplied on a dendrite), **and it works**: it recovers
    the composition superposition couldn't (6/6 at maximum interference). So the
    spiking brain now has *both* pieces of the composition algebra — bundle and
    bind — the core skill for producing and understanding novel combinations.
    The deepest remaining step is to compute that multiply inside a real spiking
    dendrite rather than as a readout product.
- Grow structure as needed through activity-dependent connections, neuron or
  region growth, pruning, homeostasis, and replay-based consolidation.
- Turn the affect core into graded internal state that changes attention,
  memory, speech, and action. A scalar label or binary mood switch is not an
  emotion claim.
- Make curiosity track learning progress and uncertainty in the brain's own
  experience, rather than rewarding novelty by a host rule.
- Make source, confidence, authorship, and uncertainty influence what the
  shared brain says or withholds. Retire host-side safety floors when their
  neural replacements are verified.
- Use new combinations, lesions, social consequences, and retention tests to
  judge the whole loop rather than collecting more isolated faculty gates.

### Long term: become fluent, deep, and efficient without changing the claim

- Reach open-ended conversation that is genuinely generated by the brain's
  grounded world model, self model, affect, memory, and goals.
- Let the system form and revise beliefs, remember sources, imagine and test
  alternatives, and keep learning after the initial caregiver period.
- Retire the remaining corpus, host-decision, hand-set-structure, and exact
  metadata scaffolds that stand between sensation and action.
- Optimize the same faithful neural mechanisms for the high-end consumer
  hardware envelope. Preserve sparse, event-driven, local computation so the
  design can eventually inform analog neuromorphic hardware.

## Research And RAG Workflow

**RAG** means retrieval-augmented generation: retrieve relevant project records
and scientific sources before proposing or writing a result. In this project,
retrieval prevents redoing refuted work; it does not replace reading the source.

Before building a mechanism:

1. Search the project's findings, plans, biology catalog, and retracted or
   refuted records. Run the local pre-build/corpus check when available.
2. Read the cited biology in depth, then check relevant external engineering,
   machine-learning, and spiking-neuroscience work. A RAG hit is a pointer;
   open and read the load-bearing passage.
3. Write a functional-role specification: what the mechanism must do for the
   whole brain, what a template could fake, and what would count as failure.
4. Produce a ranked set of biology-based, fully spiking, one-brain methods.
   Start with the cheapest rate-level or spike-level de-risk that preserves the
   necessary controls, then move to the real shared substrate.
5. Record every external claim in a structured research packet. External
   evidence may inform a gate only after explicit review and source intake; a
   packet is not automatic permission to call a result solved.

Keep the RAG index fresh on CPU and check both manifest freshness and retrieval
quality, such as labeled top-three hit rate and mean reciprocal rank. Index
maintenance is workflow support, not biological evidence.

## Experiment Engine Workflow

The experiment engine now automates a bounded Stage B screening path, not the
full research loop. It authenticates candidate packets, supports a distinct
authority policy for each candidate, executes readiness traces and four
intrinsic-current lesions, stores compact authenticated traces, binds artifacts
to scoring receipts, and strictly aggregates the five resolved subgates. The
exact 512-point Sobol candidate manifest is deterministic and filed. The GPU
batch path accelerates engineering screening; NumPy/CPU execution remains the
scientific authority.

The exact filed campaign materializes, dispatches, persists, triages, and
confirms survivors end to end. A digest-bound supervisor now authenticates
existing receipts, resumes valid partial progress, advances one deterministic
GPU batch per invocation, and runs strict triage after completion. It is not
yet a fully autonomous research loop. The fresh V3 successor screen completed 2,560 GPU
arm traces and produced no engineering passes, so no CPU confirmation was
launched. Batch width 512 was selected by a separate engineering benchmark.
The controller pins source, dependencies, host assignments, recovery hosts,
artifact sets, and receipts; local collection independently authenticates and
recomputes each score. Generation of a scientifically valid next search
remains blocked on unresolved biological protocols, and arbitrary new
objectives must not be invented from partial evidence. Unspecified subgates
remain fail-closed.

The latest authenticated diagnostic localized the old packet failure to model
structure rather than an unsearched corner of its parameter ranges. Receipt
ingestion and bounded observation generation are now implemented. The new
successor's Stage 1 clamp also executes and analyzes automatically: sealed
CPU/GPU observations are authenticated, all 18 source endpoints are re-fit,
and uncompensated gates assign the verdict. It found seven structural failures.
An authenticated handoff now maps that exact failure set into fixed biological
questions and opens a research gate without manual transcription. It still
cannot independently accept source claims, choose a replacement state family,
append a new design version, preregister its successor, or dispatch it. Those
architecture changes remain independently source-reviewed and prospectively
sealed; the active slice is validating candidate state models before the next
Stage 1 transfer.

A versioned V2 contract resolves the post-spike AHP direction without
pretending the paper specified a medium-AHP window: the scorer recomputes the
median voltage nadir across all 100 complete interspike intervals. The original
V1 files remain byte-identical for historical replay, and both survivors pass
the V2 Cav2.2 and SK directions. V3 adds explicit project-operational NaP and
HCN companion protocols. The controller now generates their raw traces and the
scorer independently recomputes stable-baseline voltage change, post-lesion
spiking, and paired hyperpolarized V-I slopes. HCN passes in both survivors.
NaP removal stops spikes but depolarizes both cells, exposing that the prior
silence-only check selected the wrong quiescent state. These are direction
tests, not claims that simulator lesions reproduce drug protocols. The next
search must screen the NaP voltage direction before CPU confirmation. The
12-cell SK result remains unavailable until independently justified cell
parameterizations exist.

1. **Plan.** Materialize the treatment, controls, lesions, anti-cheats, exact
   variables, seed partitions, expected artifacts, and resource budget. For any
   "it emerged / depends on the learned or structured weights" claim, include the
   distribution-preserving **weight-shuffle dependency control**
   (`tools.lab.dependency_control`, Shiu 2024): a function that survives a
   value-distribution-preserving shuffle of its weights rode on gross statistics,
   not learned structure (added 2026-08-19; demonstrated on the gap#5 WHEN `W_ctx`
   recency pathway — see the finding and the verify-go skill).
2. **Seal.** Freeze the command/configuration and record provenance. Keep
   development and held-out seeds mechanically separate.
3. **Dry-run.** Validate the sealed handoff, arm materialization, control set,
   lesion set, receipts, and held-out gates before dispatch is allowed.
4. **Execute.** The controller, not a short-lived research agent, owns decisive
   multi-seed runs. Each seed runs as an independently identifiable process
   when parallelism is scientifically valid.
5. **Verify.** Read the runner's own verdict and raw artifact, check backend,
   seed, configuration, controls, lesions, and provenance, then use
   independent adversarial checks before calling a result positive.
6. **Record.** Append the finding and update the live state, workboard, and
   roadmap in the same cycle when a status, blocker, or next action changes.

Agents may build or audit, but they do not own long sweeps. Independent work
must run concurrently when resources permit, every lane has a next action, and
every blocker has a recovery action. The controller must not fill hardware with
duplicate, unplanned, or scientifically dependent work.

## Compute And Parallelism Rules

- Use the local RTX 3090 with 24 GB VRAM for large coupled simulations. Set
  `SIM_BACKEND=cupy` explicitly for GPU work; do not infer the backend from
  imports or process mappings.
- Use `SIM_BACKEND=numpy` for tests and tiny smoke checks. A runner's default
  may silently select CPU, so the call site must choose the backend explicitly.
- Use local CPU for tests and bounded calibration. Use `pool40`, `pool41`, and
  `pool42` mini PCs for independent CPU seeds when the dispatcher and source
  provenance checks allow it.
- Fan independent seeds out as separate OS processes rather than looping all
  seeds serially in one process. Do not parallelize arms that share mutable
  state or violate the preregistered design.
- GPU work requires the shared lease and an empty running-queue claim. Check
  lane coverage before stocking a queue: keep independent CPU lanes active and
  do not mistake a full GPU for scientific coverage.
- Keep the local model-offload service stopped during GPU experiments. Use it
  only for bounded conservative work in its isolated fallback clone when the
  lease is free. Its end-to-end edit, local commit, exact-session resume, and
  cleanup path are validated; frontier review remains mandatory.
- Long runs need per-seed or per-day checkpoints, resumable output, provenance,
  and a state-checking heartbeat. This roadmap edit does not launch experiments.

## Acceptance And Honesty Boundary

Use **GO** only for the exact test that passed, and **NO-GO** for the tested
method when its controls fail. Neither label alone means that a human faculty
or the whole mind is complete. Generalization claims normally require the six
canonical seeds 42, 43, 44, 100, 101, and 102, plus matched controls, lesions,
and adversarial verification. A gate that uses a different preregistered seed
set must be reported with that scope, not silently upgraded.

Every claim should say:

- what was tested and what was not;
- which computation was neural and which part was a temporary host scaffold;
- which controls, lesions, seeds, backend, and artifact support the result;
- whether the result is a narrow de-risk, an integrated capability, a failed
  method, or an unresolved blocker; and
- what exact evidence would permit the next promotion.

The system may report functional readings such as, "the familiarity monitor
reports this input as novel" or "the confidence signal is weak." It must not
say or imply that it feels, is conscious, has subjective experience, or has a
person inside it. The project measures functional correlates of self-modeling,
affect, memory, agency, and uncertainty. Phenomenal experience is outside what
the experiments can honestly establish.

## Short Glossary

- **Shared substrate:** the common simulated neural network on which regions
  communicate through modeled activity and synapses.
- **Fully spiking:** the causal computation between sensation and action is
  carried by spiking neurons and synapses, not host-side cognitive formulas.
- **Scaffold:** a temporary shortcut with a named biological replacement and a
  removal test.
- **De-risk:** a small experiment that tests feasibility or a mechanism; it is
  narrower than an integrated capability demonstration.
- **Held-out seed:** a reserved random initialization used only after a design
  is fixed, to test generalization without tuning on it.
- **Local credit assignment:** a neural learning rule that assigns a delayed
  consequence to the synapses and actions that caused it without a host answer
  key or nonlocal backpropagation shortcut.
- **Source monitoring:** distinguishing what was experienced, heard, inferred,
  imagined, or is uncertain about, and using that distinction in behavior.
- **Neuromodulation:** brain-wide or regional chemical-like signals that alter
  learning, attention, motivation, or plasticity in the neural model.
- **RAG:** retrieval-augmented generation; here it means retrieving and then
  reading project and scientific sources before research decisions.
- **CuPy and NumPy:** the GPU and CPU numerical backends used by the simulator.
