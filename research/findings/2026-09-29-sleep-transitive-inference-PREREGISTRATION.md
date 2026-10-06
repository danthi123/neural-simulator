---
type: finding
status: live
date: 2026-09-29
lane: prove-who-owns-the-computation
mechanism: offline-replay associative inference (the plan's sleep transitive inference) on the spiking CA3 replay store (research/runners/sleep_transitive_inference.py; research/biology/sleep-relational-integration.md) -- A->B learned in context X, B->D in context Y, A and D never co-occurring awake; after a night of SWR replay with the engine's own STDP, does A alone in a novel context Z drive D through a direct link, which a replay disruption, a scrambled or reversed replay and crossed chains do not produce, while the premises survive the night; no sim/ or webapp/ change
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/_sleep_transitive_inference/dev/cupy_seed8.json
  - research/findings/raw/_sleep_transitive_inference/dev/cupy_seed9.json
  - research/findings/raw/_sleep_transitive_inference/dev/cupy_seed10.json
  - research/findings/raw/_sleep_transitive_inference/dev/cupy_seed11.json
  - research/findings/raw/_sleep_transitive_inference/dev/q1_s7.json
  - research/findings/raw/_sleep_transitive_inference/dev/q2_s7.json
  - research/findings/raw/_sleep_transitive_inference/dev/q3_s7.json
  - research/findings/raw/_sleep_transitive_inference/dev/exploration_seed7.json
  - research/findings/raw/_sleep_transitive_inference/dev/opcheck.json
  - research/findings/raw/_sleep_transitive_inference/dev/smoke_seed7.json
biology:
  - research/biology/sleep-relational-integration.md
  - research/biology/swr-sequence-replay.md
verdict: PRE-REGISTRATION only. No gate seed has been built, run or queued. Dev seeds only (never 42/43/44/100/101/102).
---

# PRE-REGISTRATION -- offline-replay associative inference (the plan's sleep transitive inference): does replay join A->B and B->D into A->D, and what does the night cost? (plan 2026-09-25, step 5)

**Status at commit time.** Registered before any gate-seed run: no output of this family exists at any of the six gate
seeds, locally, on the pool or on the GPU, and no gate line is queued. The runner refuses a gate seed under `--dev`
and refuses a gate run on any backend but numpy. Everything seen before this registration is disclosed in section 2
(dev seeds only). A mid-flight audit of this build (2026-09-29, seven confirmed issues) was applied before
registration; section 13 lists each item and where it landed. Terms follow `docs/TERMS.md`: "consolidation" is not
used (no source-structure lesion is tested), "lesion" only where the lesion is verified on the record at measurement
time (U2 reads the night's raster and weights, never a host flag), "selective" only with the scrambled control and
raw magnitudes beside it.

**Naming.** The plan calls this "sleep transitive inference". What is tested is associative inference over paired
associates sharing an item (A-B, B-D => A-D; Lau 2010's paradigm), produced by offline replay; it is not ordinal
transitive inference (A>B>C, the 2026-06-27 research gate's capability), and "sleep" here means an offline period of
SWR replay (an awake-rest replay arm is not in this family).

## 1. The question, and who owns the answer

The plan of 2026-09-25 (section 5) asks: A->B is learned in context X, B->D in context Y, and A and D never co-occur
awake; after sleep, presented with A alone in a novel context Z, does the system infer D? It requires at least four
arms (normal replay, wake only, replay with its order scrambled at preserved activity statistics, replay off), a
control that matches marginal activation and co-occurrence statistics while changing the relational chain, before and
after representational reads, and a replay lesion that removes A->D while leaving A->B and B->D intact. The plan
places these gates in the prioritized-memory preregistration: revision 5 of the design
(`research/findings/2026-09-25-prioritized-memory-remember-what-matters-DESIGN.md`) registers the chat-level gate WM11
(binding at Step 7, section 9 here) and this document registers the family that tests the mechanism now, together with
the owner's prioritized-memory questions for this substrate: is what was learned kept through the night, and does
replay spend its effort on what matters or on a minor detail?

**Why the substrate first.** The chat store (the composer's blocks with the DA tag-capture ledger and the sleep route)
has no process that writes a relation between two blocks during sleep: the night re-tags existing blocks. A chat-level
arm today can only read "no inference" (WM11's baseline prediction). The spiking CA3 replay store is where the biology
puts the mechanism (co-reactivation of overlapping memories in SWRs, with plasticity during it), and it is machinery
the project already has (the Ecker 2022 AdEx CA3 store, 6-seed GO for learned forward replay,
`research/findings/2026-08-20-ecker-ca3-forward-band-LEARNED-by-STDP-emergent-swr-replay-6seed-GO.md`; the
replay-driven STDP arm of `research/findings/2026-08-27-ecker-adex-store-learn-through-use-NOGO.md`).

**Host versus brain, for this family (the phase's accounting).**

| step | who does it | declared as |
|---|---|---|
| which assemblies are driven, when, in the waking episodes and the probe | host | the world |
| awake / asleep, the length of the night | host | the body clock |
| the start of each SWR event: a non-specific current on a randomly chosen assembly every `swr_period` steps | host | a declared shortcut (the Ecker GO runner's protocol; without it the store is silent) |
| what an event contains after its start, which assemblies co-activate and in what order | the network | spiking propagation through learned weights, bounded by PV feedback inhibition |
| writing A->B, B->D (awake) and any A->D (asleep) | the engine's STDP | spiking plasticity |
| which assemblies ignite when A and Z are presented | the network | the answer |
| spike reads, weight reads, event counts, the hub silencing, the replay disruption, the surrogate rasters of `yok`/`scr`/`rev` | host | instruments and experimenter interventions |

No host code relates A to D: the runner never looks up, chains or scores a relation. Every relational read is the
fraction of an assembly's cells that the network drove to spike.

## 2. What was seen before this registration (disclosed)

Dev seeds 7, 8, 9, 10 and 11 only, with the runner at this commit's tree or its dev predecessors; artifacts under
`research/findings/raw/_sleep_transitive_inference/dev/`.

All on dev seed 7 unless stated; every number here is transcribed at two decimals from
research/findings/raw/_sleep_transitive_inference/dev/exploration_seed7.json (scratch scripts importing this runner's
dev predecessors; their log lines are stored verbatim there) or from the runner's own dev records
research/findings/raw/_sleep_transitive_inference/dev/q1_s7.json, q2_s7.json and q3_s7.json (cupy, arms `wake`,
`sleep`, `off`, `yok`, `scr`; the minor detail then told on every third lap).

1. **The first read could not see a full volley.** A peak of a 5 ms smoothed active fraction read about 0.06 for an
   assembly whose 48 cued cells had all fired; the read became the recruited fraction (section 3), and replay
   participation "at least 25 % of the cells spiked in the SWR period".
2. **A plastic night with SWR events 32.5 ms apart spreads to every assembly.** At the Ecker GO runner's period (325
   steps), between-pair density 0.5 and no inhibition, 62 events drove every between-pair mean to about 400 pA and a
   hub-silenced A1 cue then recruited all 15 assemblies (`dbg2_d50l30`); at density 0.25 and 185 events the mean
   between-pair weight rose from 39.7 to 266.5 pA (`dbg3_v1`). Events 32.5 ms apart pair with each other inside the
   100 ms STDP window. At one event per second an event recruits exactly the stored chain (`tstep`: started at X1 it
   recruited X1, A1, B1, D1, P1; at Y1, Y1, B1, D1, P1). The registered period, 150 ms, keeps events further apart than
   the STDP window at 6.7 events per second: a declared compression of NREM's roughly one SWR per second, for cost.
3. **At 150 ms the night writes a specific A->D link.** numpy, no inhibition, 120 events (`dbg4_p1l`): the host
   replica reproduced the engine exactly (max |dw| 0); A1->D1 rose from 15 to 224 pA and A2->D2 to 240 pA while
   A1->D2 and A2->D1 stayed at 15; DI 0.35 against a foil read of 0; the scrambled surrogate read DI 0.02 and raised
   the foil read to 0.36. The asymmetric rule symmetrizes under replay (as in
   `research/findings/2026-08-27-ecker-adex-store-learn-through-use-NOGO.md`): B1->D1 fell from 321 to 157 pA while
   D1->B1 rose from 10 to 162 pA, and PBD fell from about 0.45 to 0.12.
4. **Feedback inhibition keeps the premises better and the inference larger.** cupy, 160 events: without inhibition
   (`q1_s7`) wake PAB 0.48, PBD 0.45; after the night DI 0.27, PAB 0.32, PBD 0.10, foil 0; with 40 PV cells (150 pA
   PC->PV, 40 pA PV->PC; `q3_s7`) wake PAB 0.59, PBD 0.52, FI 0.58; after the night DI 0.45, PAB 0.43, PBD 0.41, foil
   0, PBP 0.14 -> 0.44. In both, `off` read exactly as `wake` (no weight changed) and the scramble read DI 0.00 with
   the foil at 0.44 and 0.50: scrambled replay writes links, but not relational ones.
5. **cupy is not exact for the replica.** On cupy the replica differs from the engine by up to 0.01 pA (`q1_s7`:
   13873 synapses above 1e-3 pA); on numpy it is exact. Gate seeds run on numpy (U3's tolerance is 1e-3 pA).
6. **The operating point, chosen on dev seeds before any gate seed, by the section-3 criteria:** PV inhibition on
   (without it the premises erode twice as fast, item 4, and CA3 has it); the between-pair weight 15 pA (the Ecker band
   runner's value; 5 pA was also run: DI 0.15 at 60 events on numpy, `dbg4_p3`, and 0.25 at 160 events on cupy,
   `q2_s7`); 160 events (240000 steps, set before the q runs by the per-seed pool budget); the minor detail on every
   third lap at half drive, as in every dev run above. Disclosed: after the q runs showed the detail replayed as often
   as the core (PM2 failing on dev), the detail was changed once to "told on one lap" and then restored, because that
   change would have tuned the protocol toward a gate; no run at the one-lap setting was graded. The bars of section 6
   were written before any run at this operating point, and the audit's changes (section 13) added arms and gates, not
   looser bars.
7. **The operating point, checked on dev seeds 8-11 at the registered constants** (every arm, cupy through
   `tools/gpu_queue.sh`): the section-3 criteria are applied mechanically by
   `python -m research.runners.sleep_transitive_inference_grade --devcheck research/findings/raw/_sleep_transitive_inference/dev/cupy_seed8.json ... cupy_seed11.json`,
   which writes `research/findings/raw/_sleep_transitive_inference/dev/opcheck.json` and none of which reads DI. This
   document is committed only if every record passes every criterion, and those records are committed, with the rest of
   the dev evidence, before it. A plumbing smoke of every arm on numpy (dev seed 7, 3 laps, a 4-event night,
   research/findings/raw/_sleep_transitive_inference/dev/smoke_seed7.json) ran every arm without error, the replica
   reproduced the engine exactly (max |dw| 0), and the crossed store's premise read flipped sign as designed (PBD -0.30).
   Dev seed 8 (cupy; `cupy_seed8.json`; grades UNDEFINED on U3 alone, the cupy replica mismatch, which is why gate
   seeds run on numpy): waking PAB 0.42, PBD 0.41; after the night DI 0.32, PAB 0.34, PBD 0.16 (the erosion, with PV
   on), foil 0; `dis` wrote no weight change and read as `wake`; the budget-matched raster scramble read PAB 0.07 and
   PBD 0.17 (it wrecks the premises even at the normal night's total plasticity), which moved STI3 onto the weight
   permutation `wperm` before registration; `wperm` kept the premises (PAB 0.33, PBD 0.42, foil 0) and read DI 0.01;
   `rev` read DI 0.13 with WDA above WAD; `cross` read DI -0.38 (the sign flip); `nocomp` kept the inference (DI
   0.32); `noadapt` ran away (every assembly recruited, mean |dw| 374 pA).

## 3. Protocol and constants (fixed at this commit; `DEFAULTS` in the runner)

**Store.** 15 assemblies of 80 AdEx CA3 pyramidal cells (`ADEX_ECKER_CA3_PC`, the Ecker 2022 preset; b = 120 as the
Ecker GO runner) plus 20 unassigned cells: items A1, B1, D1, P1, C1, A2, B2, D2, P2, C2; contexts X1, Y1, X2, Y2; the
novel context Z. Within-assembly synapses (density 0.5, 60 pA) are fixed. Every ordered pair of different assemblies is
connected at density 0.25 with weak, equal, plastic weights (15 pA, the Ecker band runner's initial between-edge
weight): the anatomy allows any link, experience writes it. Feedback inhibition: 40 PV basket cells, every PC exciting
every PV cell (150 pA) and every PV cell inhibiting every PC (40 pA), fixed (the Ecker 2022 model's PC-PVBC loop; the
Ecker replay runner's 150 pA PC->PV). OU background current sigma 100 pA (the engine's `ou_std_current_pA`; section 8,
(8)); dt = 0.1 ms; `cfg.seed` set to the seed. Backend: numpy for every gate seed (U3; the runner refuses others).

**Waking episodes.** The two chains are each other's matched foil. Phase X: for 30 laps, chain 1's episode X1->A1->B1
then chain 2's X2->A2->B2; the minor detail P_k is a fourth slot at half drive on every third lap (10 of 30). Phase Y:
for 30 laps, Y1->B1->D1 then Y2->B2->D2. A slot drives 60 % of its assembly's cells at 9000 pA for 4 ms of its 8 ms;
after each episode 30 ms of silence and the STDP spike clocks cleared, which stands for the minutes between episodes, so
no pairing crosses two episodes. A_k and D_k therefore never co-occur awake, and the pre-sleep weight read checks it
(A->D equal to the foil A->D_other, section 5). STDP: the engine's Bi-Poo soft-bound rule, a+ 0.05, a- 0.06, tau 20 ms,
w_max 900 pA (the Ecker band runner's values).

**Night.** A fresh build carrying the post-waking weights (no transient state crosses the hours between phases).
240000 steps (24 s), an SWR event started every 1500 steps (150 ms; 160 events) by 9000 pA on 60 % of an assembly drawn
uniformly from the 14 that are not the test context (Z never replays: it is never part of experience), for 4 ms; STDP
on, the clock advanced every step, the raster recorded. Events 150 ms apart cannot pair with each other inside the
100 ms STDP window (section 2, item 2).

**Replay disruption (the replay lesion, `dis`).** As the normal night, and from 4.2 ms to 19.2 ms after each initiation
every pyramidal cell receives -4000 pA: the initiating assembly fires (its cue ends at 4 ms) and the event is cut before
it propagates (the closed-loop SWR disruption of Girardeau 2009 and Ego-Stengel & Wilson 2010, as a current). U2 checks
on the raster that it held.

**Probe.** A fresh build carrying the arm's post-night weights, STDP off. Per trial (8 trials, cue-cell subsets drawn
per trial): A_k's cue cells (60 %) and Z's cue cells driven at 9000 pA for 4 ms, 40 ms read, 100 ms silence. Three
probes per chain: free (A_k + Z), hub-silenced (A_k + Z with B_k held at -20000 pA for the whole trial; the membrane
potential this implies is not physiological, only whether B_k spikes is read, U1) and premise (B_k + Z). The read of an
assembly is its recruited fraction: the fraction of its 80 cells that spiked in the trial, averaged over trials; the
onset (steps to 25 % recruited) is recorded. The training-context probe (A_k + X_k, hub-silenced) is REPORTED.

**Operating-point criteria (registered; none reads DI).** The constants above were chosen on dev seeds, before any
gate seed, by criteria that do not depend on the inference read: the waking premises are learned (STI0's bar); the
normal night does not saturate the store (foil and never-encoded reads below 0.05 in the `sleep` arm); SWR events are
further apart than the STDP window; the disruption holds (U2's bars). Section 2 reports the dev seeds against them.

## 4. Arms (per seed; one process per seed; numpy)

| arm | night | role |
|---|---|---|
| `wake` | none: probed straight after the waking episodes | wake only (Ellenbogen's 20-min group, section 8) and the pre-sleep read |
| `sleep` | SWR replay, STDP on | normal replay |
| `off` | STDP on, OU on, clock advanced, no SWR initiation | no offline activity (REPORTED; U2: the raster shows no replay and no weight changes) |
| `dis` | SWR replay, STDP on, each event cut 4.2 ms after its start | the replay lesion (binding: STI2, STI6) |
| `yok` | the `sleep` arm's own raster applied to the post-waking weights by an exact host replica of the engine's STDP step | instrument: must reproduce the `sleep` arm's weights (U3) |
| `wperm` | the `sleep` night's own weight changes on the plastic synapses, permuted among them (seeded) | structure scrambled at exactly matched plasticity (the same multiset of weight changes; binding: STI3; U6 guards its validity) |
| `scr` | the `sleep` raster with every assembly's activity (its cells together; each unassigned cell alone) circularly shifted by its own random whole number of SWR periods, through the same replica, then its weight change scaled to the `sleep` night's total absolute weight change | replay co-reactivation scrambled at matched spike counts and matched total plasticity (REPORTED with its validity: on dev seed 8 it lost the premises, section 2 item 7) |
| `rev` | the `sleep` raster time-reversed inside every SWR period, through the same replica | temporal order only reversed: counts, co-participating sets and pairs kept (binding: STI3b) |
| `cross_pre`, `cross` | a second waking run with Y_k->B_other->D_k in place of Y_k->B_k->D_k, probed before and after the `sleep` night | the relational chain changed at matched marginals: every item keeps its counts (each B in two episodes, each D in one), only which D each A leads to changes (binding: STI4, a sign flip) |
| `nohub_pre`, `nohub` | a waking run with Y_k->C_k->D_k (no shared item), before and after the night | REPORTED (marginals not matched: B in one episode, C in one) |
| `nostdp` | SWR replay, STDP off | instrument: replay without plasticity leaves the store byte-identical |
| `nocomp` | SWR replay, STDP on, within-assembly weights at 0 for the night only (restored for the probe) | pattern-completion lesion (REPORTED; the lesion checked at the night's end) |
| `noadapt` | SWR replay, STDP on, AdEx a = b = 0 for the night only | adaptation lesion (REPORTED) |

The two chains make the within-brain matched-statistics control: D_k and D_other have identical marginal statistics
and differ only in whether they share an item with A_k; every inference read is a difference against it.

## 5. Reads (per arm)

- **DI, direct inference:** mean over the chains of read(D_k | A_k + Z, B_k silenced) - read(D_other | same probe).
  The hub-silenced probe reads only a route from A to D that does not pass through B; retrieval-time chaining through
  B (REMERGE, Kumaran & McClelland 2012) cannot supply it.
- **FI, free inference (REPORTED):** the same with B_k free. Chaining at retrieval can supply it in every arm. The
  onset of D_k in the free probe is REPORTED (a direct link should bring D earlier).
- **PAB, PBD, premises:** read(B_k | A_k + Z) - read(B_other | same) and read(D_k | B_k + Z) - read(D_other | same).
- **PBP, the minor detail:** read(P_k | B_k + Z) - read(P_other | same).
- **Weights (the representation, before and after):** per ordered assembly pair, the mean weight; WAD = mean
  w(A_k->D_k) - mean w(A_k->D_other); WDA = mean w(D_k->A_k) - mean w(D_other->A_k); the between-pair mean; the mean
  absolute weight change over the plastic synapses against the arm's own pre-night store; before (`wake`, `cross_pre`,
  `nohub_pre`) and after each night.
- **Replay (from the night raster):** per assembly, the number of SWR periods in which at least 25 % of its cells
  spiked; the A_k/D_k and A_k/D_other co-participation counts; per event, whether the initiating assembly took part and
  how many assemblies did.

## 6. Gates (per seed) -- bars fixed now (`research/runners/sleep_transitive_inference_grade.py`)

STI gates (the family verdict):

| gate | passes only if |
|---|---|
| STI0 premises learned | in `wake`: PAB >= 0.30 and PBD >= 0.30 (scored as the brain's outcome, never voided) |
| STI1 sleep infers | DI(`sleep`) >= 0.15 and DI(`sleep`) - DI(`wake`) >= 0.15 |
| STI2 replay necessary, premises intact | DI(`dis`) - DI(`wake`) <= 0.05 and DI(`sleep`) - DI(`dis`) >= 0.15, and in `dis`: PAB >= 0.30 and PBD >= 0.30 (the replay lesion removes A->D and leaves A->B and B->D) |
| STI3 structure necessary | DI(`sleep`) - DI(`wperm`) >= 0.5 x (DI(`sleep`) - DI(`wake`)), with that gain > 0; scored only if U6 does not fire |
| STI3b temporal order necessary | DI(`sleep`) - DI(`rev`) >= 0.5 x the gain, and in `rev` WDA > WAD (the link the reversed night writes runs D->A) |
| STI4 the chain decides where the inference goes | DI(`cross`) - DI(`cross_pre`) <= -0.5 x the gain (A_k now reaches D_other: a sign flip), with the crossed store's premises learned (PAB(`cross_pre`) >= 0.30, PBD(`cross_pre`) <= -0.30) |
| STI5 no false inference | under the hub-silenced probe after the normal night, the foil read max_k read(D_other) is at most 0.05 above `wake`'s, and every never-encoded C assembly reads below 0.10 |
| STI6 representation before/after | WAD(`sleep`) - WAD(`wake`) >= 20 pA, and abs(WAD(`dis`) - WAD(`wake`)) <= 0.1 x that rise |
| STI7 the premises survive the night | PAB(`sleep`) >= max(0.30, 0.9 x PAB(`wake`)) and PBD(`sleep`) >= max(0.30, 0.9 x PBD(`wake`)) |

STI7 is in the STI verdict: an A->D link bought by erasing B->D is a lossy rewire, not relational integration, and the
biology binds premise retention to the capability (Ellenbogen 2007: premises kept by every group, over 85 %).

Prioritized-memory gates for this substrate (a separate verdict line; the owner's "remember things that matter... not
just store info like a RAG system"; section 8 says what they can and cannot show):

| gate | passes only if |
|---|---|
| PM1 what matters is kept | STI7 (the same condition) |
| PM2 replay effort follows what matters | the mean replay count of the six core items (A, B, D of both chains) exceeds that of the two minor details (P1, P2) |
| PM3 a minor detail fades over the night | PBP(`sleep`) < PBP(`wake`) |

**UNDEFINED (never a pass, never a zero).** U1 the hub silencing did not hold (B_k recruited 0.10 or more under any
hub-silenced probe of any arm). U2 a lesion did not hold, read on the engine's behaviour: `off`'s raster shows any
replay participation or any weight changed; in `dis` fewer than 90 % of the events have their initiating assembly
taking part, or more than 10 % reach a second assembly; `nostdp` changed any weight; `nocomp`'s within-assembly
weights were not all 0 at the start and the end of its night; `noadapt`'s a or b was not 0 at the start and the end of
its night. U3 the replica did not reproduce the engine (max |w(`yok`) - w(`sleep`)| > 1e-3 pA), a surrogate changed
any cell's spike count, or `wperm` changed the multiset of weight changes. U4 a run error or a missing arm. U5 the `sleep` night initiated no event. U6 the weight-level
control is not valid: in `wperm`, PAB < 0.30 or PBD < 0.30, or its foil read more than 0.05 above `wake`'s, or a
never-encoded C at 0.10 or more (a DI drop there would come from wrecked specificity, not from missing structure). The
same validity is REPORTED for `scr`.
U7 the seed's part records disagree (the seed, the constants or the post-waking weight hash) or an arm appears twice
(section 11).

**Verdicts.** Per seed: UNDEFINED if a U-rule fires; STI GO iff STI0-STI7 all hold, else NO-GO; the PM line likewise
over PM1-PM3. Family: INCOMPLETE if a seed is missing; UNDEFINED if any seed is still UNDEFINED after its re-run budget
(one re-run per seed, for U4 only, at the same pinned revision; the other U-rules need a fix and an amendment first);
GO iff all six seeds read GO; NO-GO otherwise. The grader's `selftest()` runs the designed GO, NO-GO and UNDEFINED
fixtures (28 cases) and passes at this commit. The one-sided exact sign-flip p over seeds is REPORTED for
DI(`sleep`) - DI(`wake`) and for each lesion's gap.

## 7. Which mechanisms are expected to be necessary (the six-seed ablation standard)

Rule, per seed (`necessity()` in the grader, with `tools.lab.attributable_to`): a mechanism is necessary if its lesion
arm keeps less than half of the sleep gain, DI(lesion) - DI(its own pre-night store) < 0.5 x (DI(`sleep`) -
DI(`wake`)). Family label: NECESSARY if necessary on all six seeds, "contributing" on one to five (reported as k of 6,
the arcc lesson: pattern completion was load-bearing on 1 of 6 seeds there), "not necessary" on none.

| mechanism | lesion arm | expected | why |
|---|---|---|---|
| offline activity at all | `off` | NECESSARY (by construction: no spikes, no pairing) | REPORTED; the binding replay lesion is `dis` |
| SWR replay propagation | `dis` | NECESSARY (binding: STI2) | the A_k->D_k pairing needs events that continue past their start |
| where the night's plasticity landed | `wperm` | NECESSARY (binding: STI3) | the inference needs the changes on the A_k->D_k synapses, not the same amount of change elsewhere |
| replay co-reactivation structure | `scr` | NECESSARY, REPORTED | the pairing comes from events that enter B_k from A_k and continue to D_k |
| temporal order within events | `rev` | NECESSARY (binding: STI3b) | under the asymmetric rule the reversed order writes D->A, not A->D |
| a shared item | `nohub` | NECESSARY | without B_k in both episodes no event joins A_k and D_k |
| plasticity during replay | `nostdp` | NECESSARY by construction (the weights cannot change) | kept as the instrument that the replay itself writes nothing else |
| pattern completion (within-assembly recurrence) | `nocomp` | uncertain, REPORTED | the initiation cue already drives 60 % of the assembly |
| spike-frequency adaptation | `noadapt` | uncertain, REPORTED | adaptation segments replay in the Ecker model; PV inhibition may do it here |

Fixed in every arm and therefore NOT tested here (named, not claimed): the STDP rule's form (asymmetric Bi-Poo; the
symmetric broad CA3-CA3 rule of Mishra 2016 is the named follow-up), the waking protocol's context slots, the host SWR
initiation, the PV pool (it was on at every dev point that met the criteria; its lesion is a named follow-up arm).

## 8. Predictions, and what the gates can and cannot show

**Predictions (registered before any gate seed, from the dev evidence of section 2).**

| gate | predicted | basis |
|---|---|---|
| STI0 | holds | dev waking PAB 0.45-0.59, PBD 0.43-0.52 |
| STI1 | holds | dev DI after 160 events 0.25-0.45 (against 0 awake) |
| STI2 | holds if U2 holds | `dis` first runs at the registered constants on dev seeds 8-11 |
| STI3 | holds | the permuted changes are mostly small, so the premises should stay near their post-night values while A_k->D_k loses its change; not yet run on a dev seed (seeds 9-11 carry it) |
| STI3b | holds | the asymmetric rule writes the reversed order as D->A |
| STI4 | holds | the crossed store's premise flips sign by construction; the night should carry A_k to D_other |
| STI5 | holds | foil read 0 after the night with PV on dev |
| STI6 | holds | A->D rose by 216-263 pA on dev |
| STI7 | FAILS | with PV on dev PAB 0.59 -> 0.43 and PBD 0.52 -> 0.41 (0.73 and 0.79 of waking); without PV PBD 0.45 -> 0.10 |
| PM1 | FAILS | the same condition as STI7 |
| PM2 | FAILS | dev replay counts, core items against the detail: 37.0 against 41.0 (`q1_s7`), 42.2 against 45.0 (`q3_s7`) |
| PM3 | FAILS on most seeds | dev PBP rose (0.10 -> 0.12, 0.14 -> 0.44) in two of three configurations and fell (0.11 -> 0.08) in one |

Predicted family verdicts: STI NO-GO (carried by STI7) and PM NO-GO. The prediction that matters most for the owner:
on this substrate the night writes the inference, but it trades premises for it and spends replay on the minor detail
as much as on the core, the opposite of "remember things that matter"; section 10's companions are therefore the next
work whatever STI1 reads. REPORTED alongside: FI is present awake (dev 0.30-0.58), so this store reaches D from A
through B at retrieval without any night; the night adds a direct route that needs no B.


**What they show.** Behaviour at the probe (which assemblies the network drives from A with Z), the store's weights
before and after the night, which manipulations remove the effect, and what the night does to what was learned. The
two-chain foil cancels every marginal statistic of D, so a DI above zero is relational; the crossed store tests that
the chain, not the items' statistics, decides where the inference goes. The hub-silenced read separates offline
integration from retrieval-time chaining.

**What they cannot show.** (1) This is a CA3 store, not the chat brain: WM11 carries the question to the conversation
at Step 7 of the design. (2) The `wake` arm is tested at once; human inference also develops over awake offline time
(Ellenbogen 2007: 12 h awake, over 75 %), so "sleep beats awake rest" is not tested; an awake-rest replay arm is a
named follow-up. (3) The SWR start is a host cue on a uniformly drawn assembly: the host does not choose content by
importance, and nothing in this store does either (PM2 reads the consequence). (4) The rule is asymmetric STDP, a
declared deviation from the symmetric CA3-CA3 rule. (5) `scr` and `rev` are surrogates: their weights are the
engine's own rule applied by an exact host replica to a transformed raster (U3 checks the replica against the engine at
every seed), and `scr`'s change is then scaled to the normal night's total; their outcomes are read on the brain. (6)
"What matters" in this store is only what was experienced more strongly (the core items: 30 laps at full drive; the
detail: 10 at half drive); the design's importance channels (dopamine, arousal, relevance) are chat-level and absent
here, so PM1-PM3 test strength-weighted keeping, not value. (7) One store size, one night length, two chains. (8) The
Ecker runners set a config field the engine never reads (`ou_noise_sigma_pa`), so their documented 40 pA OU noise ran
at the engine default of 100 pA; this family sets `ou_std_current_pA` = 100 pA explicitly (the validated point).

## 9. The chat-level gate WM11 (registered in the design, revision 5)

Group `wt` (design 5.1): two matched chains, A_k->B_k told in a conversation block about topic X_k, B_k->D_k in a later
block about topic Y_k; probed at `d4h` (awake, no night), `n1`, `n7`, a replay-lesioned `n1_rl` and a scrambled
`n1_scr`, with a cue of A_k alone in a block about a never-used topic Z, scored for D_k against D_other. WM11 (design
5.3) requires premises kept, inference after sleep, none before, none without replay or with scrambled replay, no false
inference, and every inferred answer flagged by the brain as an inference, never asserted as a told fact (the honesty
boundary). The read must not pass through the host multi-hop loop (`OneBrainComposer.query_chain`) or the GNW
multistep chase's marker route. It binds at Step 7 (replay-written transfer), the first step that can write a relation
during sleep; both current baselines are predicted to fail its inference clause.

## 10. If it fails (THE LAW, fixed now)

A NO-GO is a verdict on this method. Registered next methods, by failure pattern:

- **STI7 / PM1 (the night erodes a premise):** the wall question first -- what does the real system run beside this
  that the model replaced with nothing? (1) the symmetric broad CA3-CA3 rule (Mishra 2016), under which co-reactivation
  potentiates both directions whatever the order, so replay cannot depress a premise; (2) sleep downscaling as the brake
  on the night's net potentiation (`research/biology/sleep-load-dependent-renormalization.md`); (3) a stronger or
  plastic PV loop (inhibitory STDP). PV feedback is already in the registered store.
- **PM2 (replay spends as much on the detail as on the core):** a neural salience channel that biases which memories
  replay, without a host rule: reward-tagged assemblies are over-represented in SWRs (Singer & Frank 2009; Ambrose,
  Pfeiffer & Foster 2016), here through the engine's reward-modulated plasticity at encoding raising the tagged
  assemblies' excitability (design Step 4(a)); then a four-chain arm (two tagged through a DA population at encoding,
  two neutral, foils matched within class) asking whether the night integrates and keeps the tagged chains first.
- **PM3 (the detail does not fade):** the store has no forgetting process; downscaling (above) is the companion.
- **STI1 fails with STI6 passing (the weight rose, the network does not use it):** the read side: the night's length
  and the PV loop's strength. **STI1 and STI6 both fail:** the symmetric CA3 rule.
- **STI2 or STI4 fails (inference without replay, or not following the chain):** the mechanism is not what this
  document claims: find the route first (encode-time integration, Zeithamova 2012: read the pre-sleep A->D weights).
- **STI3 or STI3b fails (the control also infers):** the effect is activity-level, not structural: test a
  lower-noise store before any claim. **U6 fires (the weight permutation is not a valid control):** permute within
  classes of synapses of equal pre-night weight (so premise synapses exchange changes only with each other) as the
  next control.
- **STI5 fails (sleep also raises the foil):** nonspecific potentiation, the Wang plasticity-plus-noise confound: the
  PV loop and downscaling.

## 11. Compute and queue

**Placement: the pool, numpy.** Gate seeds run on numpy (U3's replica is exact there; the runner refuses any other
backend for a gate run) on the mini-PC pool, at the commit that carries this document, the runner and the grader,
provisioned isolated (`bash tools/pool_provision.sh --revision <sha> --isolated <nodes>`), each line checked by
`bash tools/queue_job_shape_check.sh` and queued with `bash tools/pool_queue.sh add`. Each seed S runs as five jobs
(arm subsets; every hub part re-runs the same deterministic waking episodes, and U7 checks that their post-waking
weight hashes agree):

    cd ~/derisk-pool/revisions/<sha> && env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m research.runners.sleep_transitive_inference --seed S --arms <PART> --out research/findings/raw/_sleep_transitive_inference/parts/seed<S>_<tag>.json

with PART one of `wake sleep yok wperm scr rev` (tag a), `off dis nostdp` (b), `nocomp noadapt` (c), `cross` (x), `nohub`
(n). The seed record is `python -m research.runners.sleep_transitive_inference_grade --merge
research/findings/raw/_sleep_transitive_inference/seed<S>.json research/findings/raw/_sleep_transitive_inference/parts/seed<S>_*.json`
(U7: the parts disagree on the seed, the constants or the waking store, or an arm appears twice), and the family is
`... --family research/findings/raw/_sleep_transitive_inference` (writes `aggregate.json`).

**Cost.** Per seed about 3.0 M simulation steps in total (the waking runs, eight nights of 240000 steps, 13 probe sets
of 6 probes x 8 trials x 1400 steps), three replica passes over a 240000-step raster and about 90 store builds; the
largest part (b) is about 1 M steps. The GPU dev runs of section 2 (item 7) give the wall time on the 3090; on one
unloaded pool core at 3-6 ms per step a part takes about 0.5-1.7 h. Peak memory per job about 1.5 GB (the night raster,
240000 steps x 1260 cells as bytes, and its surrogates).

## 12. Amendments

Any change to a constant, a bar, an arm or the grader after this commit is an amendment, committed with its reason
before any gate-seed run it governs; a change after a gate seed has been seen voids that seed's weight for the changed
item and is disclosed.

## 13. The mid-flight audit of 2026-09-29, applied before registration

| # | audit item (severity) | where it landed |
|---|---|---|
| 1 | the scramble removed specificity, so STI3 could pass for the wrong reason; order was conflated with co-activation (HIGH) | `scr`'s weight change scaled to the normal night's total; on dev seed 8 even so it lost the premises (PAB 0.07, PBD 0.17), so it is REPORTED and STI3 binds on `wperm`, the night's own weight changes permuted among the plastic synapses (exactly matched plasticity); U6 voids STI3 when that control loses the premises or floods the foils; the order-only `rev` arm and STI3b added |
| 2 | the night that creates A->D erodes B->D, and no STI gate checked the premises after sleep (HIGH) | STI7 in the STI verdict (PM1 is the same condition); PV feedback is part of the registered store; section 10 names the companions |
| 3 | the replay-off arm was silent, so STI2 and STI6's lesion clause were vacuous, and U2 read the host's own flag (HIGH) | the disruption arm `dis` is STI2's and STI6's replay lesion; `off` is REPORTED as "no offline activity"; U2 reads the night's raster and weights for every lesion |
| 4 | the prioritized-memory line was diluted and PM2's prediction contradicted the dev evidence (MEDIUM) | PM2 and PM3 predicted FAIL with the dev numbers (section 8); the minor detail's schedule was restored to the value the dev runs used before the PM reads were seen (every third lap), after it had been changed once to "told once" when PM2 had failed on a dev seed, which would have tuned toward the gate; the next method (a neural salience channel for replay) is registered in section 10 |
| 5 | cupy's replica mismatch would make every GPU gate seed UNDEFINED (MEDIUM) | gate seeds run on numpy on the pool; the runner refuses a gate run on any other backend |
| 6 | the operating point was tuned on one dev seed against gate reads; defaults and text were stale; dev evidence sat in RAM (MEDIUM) | registered operating-point criteria that do not read DI (section 3), checked on dev seeds 8-11 (section 2, item 7); the defaults and this text agree; the dev scripts and logs are copied into `research/findings/raw/_sleep_transitive_inference/dev/scripts/` |
| 7 | the no-hub control did not match marginals; a crossed-chain arm would, with a predicted sign flip (MEDIUM) | `cross_pre`/`cross` added and binding (STI4, a sign flip); `nohub` REPORTED and read against its own pre-night store |
| LOW | Z replayed in the night; naming; the OU config trap | Z excluded from SWR initiation; the claim named "offline-replay associative inference"; the OU trap declared (section 8, (8)) and handed to a separate task (a gate for config fields the engine never reads) |
