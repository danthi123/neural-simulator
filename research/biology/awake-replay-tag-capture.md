---
type: biology
id: awake-replay-tag-capture
mechanism: Awake sharp-wave-ripple replay during quiet rest reactivates the day's recently potentiated synapses; the reactivation re-induces early LTP (and so re-sets the synaptic tag) in proportion to how strongly the assembly is reactivated, which keeps a weakly encoded trace readable and tagged until sleep can capture it; it co-releases no DA for neutral content, so it supplies no plasticity-related proteins
status: established
last_verified: 2026-09-25
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "resting periods after recent learning, for example after"
    note: "ch.5 (text of Fig. 5-2): 'Notably, sharp-wave ripples are prominent during resting periods after recent learning, for example after exploration of an environment', and replay decodes 'discrete trajectories through the recently explored environment'. WHEN awake replay happens: quiet rest after learning."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "behavior, the hippocampus enters a different regime"
    note: "ch.5: 'during immobile or resting behavior, the hippocampus enters a different regime in which neural activity is instead dominated by discrete semi-synchronous population bursts termed sharp-wave ripples'. The engine's light idle tick (idle >= IDLE_SEC, below sleep depth) is the model's immobile/resting state."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "thought to represent a form of mental rehearsal by"
    note: "ch.5: 'Replay is thought to represent a form of mental rehearsal by which certain memories are gradually consolidated'. Rehearsal = the reactivation keeps the trace."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "these nontheta states involve consummatory behaviors, such as eating, drinking,"
    note: "p.344: the non-theta states in which sharp waves occur are 'consummatory behaviors, such as eating, drinking, and grooming and immobility, non-REM sleep' -- awake rest, not only sleep."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "because sharp waves are present during such consummatory"
    note: "p.347: the argument about what sharp-wave activity does to the day's synaptic modifications is stated to apply to 'immobility, drinking, and eating following exploratory learning, because sharp waves are present during such consummatory behaviors, as well'; the alternative he develops is that the pathways used and modified in the waking brain are repeatedly replayed."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "The transient but substantial gain in population excitation creates favorable"
    note: "p.347: the SWR's three- to fivefold gain in network excitability 'creates favorable conditions for synaptic plasticity' -- why a reactivation can re-induce LTP."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "activity spreads along the path of the strongest synaptic weights"
    note: "p.349: replay CONTENT follows the weights: assemblies activated most in the experience are 'held together by the strongest synaptic connectivity and become the “burst initiators”'. Basis for selecting by the store's own read-back (a stronger trace reactivates more), not by a host list."
  - path: "PMC4785795"
    anchor: "are able to induce long-term potentiation (LTP) at synapses between CA3 and CA1 cells but only if accompanied by SWR-associated synaptic activity"
    note: "Sadowski, Jones & Mellor 2016, Cell Rep 14:1916 (doi 10.1016/j.celrep.2016.01.061), abstract; full text read via PubMed Central 2026-09-25. Spike trains recorded in the first 5 min of post-run REST (57 SWRs in 300 s) replayed into slices induced test-path LTP of x2.19, x2.42 and x3.36; the change correlated with the number of LTP-competent CA3->CA1 pairings in SWRs (r = 0.89); a poorly bound pair (CA1e) induced none. Discussion: quiescence 'may enable the connectivity of specific spatial engrams to be enhanced prior to sleep'. THE load-bearing claim: awake reactivation re-induces LTP, graded by how strongly the assembly is reactivated. Source of the 5-min bout and of the graded induction e <- e + R(1 - e)."
  - path: "PMC4441285"
    anchor: "We interrupted awake SWRs in animals learning a spatial alternation task"
    note: "Jadhav, Kemere, German & Frank 2012, Science 336:1454 (doi 10.1126/science.1217230), abstract via PubMed 2026-09-25: interrupting AWAKE SWRs gave 'a specific learning and performance deficit' while leaving place fields and post-experience SWR reactivation intact. The behavioural lesion the model's BRAIN_AWAKE_REPLAY_CAPTURE_LESION mirrors."
  - path: "PMC3215304"
    anchor: "occurs frequently in the awake state, particularly during periods of relative immobility"
    note: "Carr, Jadhav & Frank 2011, Nat Neurosci 14:147 (doi 10.1038/nn.2732), abstract via PubMed 2026-09-25: awake replay is frequent in immobility and 'consolidation occurs in both the awake and sleeping animal'."
  - path: "PMC4695386"
    anchor: "many reward responsive (RR) VTA neurons coordinated with quiet wakefulness-associated hippocampal SPW-R events that replayed recent experience"
    note: "Gomperts, Kloosterman & Wilson 2015, eLife 4:e05360 (doi 10.7554/eLife.05360), abstract via PubMed 2026-09-25, rats on APPETITIVE tasks. The DA cells that join awake replay are reward-responsive cells replaying rewarded experience -> no basis for DA co-release when a plainly told fact is replayed -> awake replay supplies NO PRP here. Same abstract: 'coordination between RR neurons and SPW-R events in subsequent slow wave sleep was diminished' -- a TENSION for the sleep route's SWR-coupled DA (see body)."
  - path: "PMC6672624"
    anchor: "can be consolidated into LTM by an exploration to a novel, but not a familiar, environment"
    note: "Moncada & Viola 2007, J Neurosci 27:7476 (doi 10.1523/JNEUROSCI.1083-07.2007), abstract via PubMed 2026-09-25: the PRPs that stabilize a weak trace come from NOVELTY (D1/D5-dependent), not from re-exposure to the familiar -- the PRP source stays the brain's DA (waking salience, the night's SWR-coupled DA)."
  - path: "PMID:22829465"
    anchor: "wakeful resting after new learning allows new memory traces to be consolidated better"
    note: "Dewar, Alber, Butler, Cowan & Della Sala 2012, Psychol Sci 23:955 (doi 10.1177/0956797612441220), abstract via PubMed 2026-09-25: 10 min of wakeful rest after a story (vs a distractor game) improved recall at 15-30 min and at 7 days. The human behavioural anchor: rest after learning helps, and helps long-term."
implemented_by:
  - webapp/awake_replay_capture.py
  - webapp/da_tag_capture_chat.py
  - webapp/da_tag_capture.py
findings:
  - research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md
  - research/findings/2026-09-25-prioritized-memory-remember-what-matters-DESIGN.md
---

# Quiet rest keeps the day's trace alive until sleep can capture it

**The wall this binds to.** With the DA tag-and-capture ledger and the sleep route armed, a plainly told fact that is
4 h old at sleep onset was lost on a real build (seed 42, sleep-replay-capture Amendment 1): its early-phase trace had
decayed, the night's replay read it at 0.0056 (a fresh fact reads 0.36-0.47), so the night neither re-tagged it nor
released DA for it. Kandel ch.54 puts the capture window at 2-3 h, and Gais, Lucas & Born 2006 find the human sleep
benefit largest when sleep follows learning within hours -- but they report a smaller benefit, not total loss.

**What the real hippocampus runs alongside tag-and-capture during WAKE, that the model replaced with a constant.**
Awake sharp-wave-ripple replay. In quiet rest after learning the hippocampus replays recent experience (Kandel ch.5;
Buzsaki p.344-349; Carr et al. 2011); the reactivation re-induces LTP at the replayed synapses, graded by how strongly
the assembly is reactivated (Sadowski et al. 2016); disrupting awake SWRs impairs learning (Jadhav et al. 2012); and in
people a few minutes of wakeful rest after learning improves recall a week later (Dewar et al. 2012). Between turns the
model's early phase did nothing but decay: the constant was "no reactivation while awake".

## How the code binds to it

- A bout runs on the continuous engine's idle tick when the ledger's own clock says the brain is still awake, at most
  once per 5 min (Sadowski's rest window; the sleep route's SWR bout; the v3 capture protocol). A turn never runs one.
- Every managed block is driven and read back by the store's own cleanup (the sleep route's read, R_i). The
  reactivation re-induces early LTP by its share of the headroom, e <- e + R_i (1 - e), decaying with the written
  trace's own 1.5 h; the tag is re-set to the same level (v3's invariant: tag = |inc| x early phase). Late phase z is
  untouched and no PRP is supplied (Gomperts; Moncada & Viola).
- `BRAIN_AWAKE_REPLAY_CAPTURE_LESION` cuts the reactivation's effect with the reads still running (Jadhav's lesion).
- No `constraints_config` / `protocol`: the one scalar (the 5-min bout) is a reuse of existing anchors, and "awake replay
  supplies no PRP" is a code path, not a config value the checker can compare.

## Known limits, recorded so they are not re-derived

**Regrowth of a faint trace (measured on the fake substrate before any brain run,
`research/findings/raw/_awake_replay_capture/design_fake_substrate.json`).** Because the read rises with the expressed
trace, rest is a positive feedback: with a bout every 5 min, a trace already down to 13 % of its expression (3 h without
rest, read ~0.04) regrows to ~0.92 within one hour of rest. In the animal, what brakes it is presumably competition for
the SWR's content among many recent assemblies (the most strongly bound become the burst initiators) and the synapse-by-
synapse reversal of decaying E-LTP (the pattern is lost, not uniformly shrunk). Neither exists with one fact in a store
whose decayed trace is a scaled copy of the full pattern. It is measured on the brain as a REPORTED arm (late rest).
On the seed-42 brain smoke (prereg Amendment 5) the regrowth did NOT happen: the real composer read the 3-h-old trace
at ~0.008 (the fake curve said ~0.043), and one hour of rest held the trace at ~13 % without bringing it back.

**Rest dose.** On the fake substrate a rest bout every 5-30 min keeps the 4-h-old fact capturable and one per hour does
not. The dose that the animal needs is not in any source read here; the brain measurement reports it.

**A tension for the sleep route, recorded here because this research surfaced it.** Gomperts et al. 2015 found the
reward-responsive VTA cells' coordination with SWRs DIMINISHED in slow-wave sleep. The sleep route's SWR-coupled DA
(tonic + span x the night's read) rests on Clopath et al. 2008's modelling assumption and is declared an operating
point in `webapp/sleep_replay_capture.py`; this abstract does not support a DA burst with sleep SWRs for neutral content.
Not resolved here; the night's PRP source is the named next rung for the sleep route.

⚠️ **Provenance honesty.** The Kandel and Buzsaki anchors are in the local corpus. The PMC/PMID sources are not; they were
read through PubMed on 2026-09-25 (Sadowski in full text, the others as abstracts), which is why they are marked
external. The induction law e <- e + R (1 - e) is a declared modelling choice shaped by Sadowski's graded, pairing-count
dependence; no source gives its functional form.
