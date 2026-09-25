---
type: biology
id: awake-replay-pattern-completion
mechanism: A sharp-wave-ripple reactivation (in quiet rest or in NREM sleep) is a population burst of the CA3 recurrent network that starts once firing reaches a threshold; the recurrent connections of the stored assembly then complete it, so a partial trace that still selects the memory reinstates the whole ensemble, and the LTP / tag the replay sets (and the DA it co-releases, in the sleep route's model) scales with the number of co-active pairs in the reinstated ensemble, not with how decisively the partial trace decodes
status: established
last_verified: 2026-09-25
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "The CA3 Region Is Important for Pattern"
    note: "ch.54 section heading (p.1360)."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "stored cell assembly would be sufficient to activate"
    note: "ch.54 (p.1360), Marr's proposal: during retrieval 'the reactivation of a subset of this stored cell assembly would be sufficient to activate the entire original neural ensemble that encodes the memory because of the strong recurrent connections between the cells of the ensemble. This restoration is referred to as pattern completion.' THE load-bearing claim for this module: the reinstated ensemble is the whole assembly, not a copy scaled by the cue."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "platform with fewer spatial cues, their performance"
    note: "ch.54 (p.1360-1361): CA3-specific NMDA-receptor knockout mice find the platform with the full cue set but are impaired 'with fewer spatial cues' -- completion from a partial cue needs the recurrent synapses' LTP. The lesion logic behind BRAIN_REPLAY_COMPLETION_LESION (every read kept, the completion's effect cut)."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "wave emerges in the excitatory recurrent circuits of the CA3 region"
    note: "p.345: 'In the intact brain, the endogenous hippocampal sharp wave emerges in the excitatory recurrent circuits of the CA3 region', from 'the synchronous bursting of CA3 pyramidal cells' -- the replay event IS a recurrent-network burst."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "networks is an especially effective method of recruiting large numbers of neurons"
    note: "p.153: 'Excitatory feedback in recurrent neuronal networks is an especially effective method of recruiting large numbers of neurons within a short time period' -- recruitment by recurrent feedback, not proportional read-out."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "Because of their pattern completion ability, recurrent networks are also known as"
    note: "p.57, note 71: recurrent networks are autoassociators because they complete patterns."
  - path: "PMID:16387645"
    anchor: "population bursts in the disinhibited CA3 region are initiated at a threshold level of population firing"
    note: "de la Prida, Huberfeld, Cohen & Miles 2006, Neuron 49:131 (doi 10.1016/j.neuron.2005.10.034), abstract via PubMed 2026-09-25: population bursts start at a THRESHOLD level of population firing; 'Population synchrony is suppressed when threshold frequencies cannot be reached due to reduced cellular excitability or synaptic efficacy'. Caveat recorded, not hidden: 'Reducing synaptic strength reveals partially synchronous population bursts' -- disinhibited slices; with weak synapses a burst can be PARTIAL. The model's partial reinstatement (some items resolve, some do not) is that regime."
  - path: "PMID:27609885"
    anchor: "Recurrent CA3-CA3 synapses are thought to be the subcellular substrate of pattern completion"
    note: "Guzman, Schloegl, Frotscher & Jonas 2016, Science 353:1117 (doi 10.1126/science.aaf1836), abstract via PubMed 2026-09-25: real-size modelling on measured CA3 connectivity 'robustly generated pattern completion'."
  - path: "PMC2877140"
    anchor: "impaired in retrieving this memory when presented with a fraction of the original cues"
    note: "Nakazawa et al. 2002, Science 297:211 (doi 10.1126/science.1071795), abstract via PubMed 2026-09-25: pattern completion defined as retrieving 'complete memories on the basis of incomplete sets of cues'; the CA3 NMDA knockout the Kandel passage summarizes."
  - path: "PMC4785795"
    anchor: "are able to induce long-term potentiation (LTP) at synapses between CA3 and CA1 cells but only if accompanied by SWR-associated synaptic activity"
    note: "Sadowski, Jones & Mellor 2016, Cell Rep 14:1916 (already bound in research/biology/awake-replay-tag-capture.md, full text read 2026-09-25): the induced LTP correlated with the NUMBER of LTP-competent CA3->CA1 pairings in the replayed SWRs (r = 0.89). The quantity this module computes as R_c: the fraction of the block's synapses whose post cell is reinstated in phase with the synapse's increment."
  - path: "PMC6156217"
    anchor: "Objects that were remembered less well were replayed more during the subsequent rest period"
    note: "Schapiro, McDevitt, Rogers, Mednick & Norman 2018, Nat Commun 9:3920 (doi 10.1038/s41467-018-06213-1), abstract via PubMed 2026-09-25: awake human hippocampal replay PRIORITIZES weakly learned items, and replay predicts later memory. Evidence against the Amendment-4 linear proxy (under it the weakest trace is replayed least and collapses); weighed as the alternative 'replay prioritization' in the body."
implemented_by:
  - webapp/replay_completion.py
  - webapp/awake_replay_capture.py
  - webapp/sleep_replay_capture.py
findings:
  - research/findings/2026-09-25-awake-replay-capture-arc-no-go-6seed.md
  - research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md
---

# A replay event completes the stored ensemble; it does not replay a scaled copy of the decayed trace

**The wall this binds to.** The awake-rest route (`webapp/awake_replay_capture.py`, prereg Amendments 4-5) re-induced
early LTP at each 5-min bout in proportion to the partial trace's cleanup DECODE MARGIN R, the smallest
(peak - runner_up) / peak over agent/action/patient. On one gate seed of six the long-delay fact's R started at
0.207 and fell to 0.031 over 48 bouts (research/findings/2026-09-25-awake-replay-capture-arc-no-go-6seed.md). The
margin is a read-out quantity: it says how close the most similar OTHER vocabulary word comes to the fact's word at
the cleanup. A replay event's LTP depends on something else.

**The same proxy sits in the night's route.** The sleep epoch re-tags with R x |inc| and co-releases DA
tonic + span x sum R. On dev seed 2 the awake completion kept the trace expressed (0.98 at the last bout, all three items
reinstated) but the night read 0.107, the SWR DA was 0.579 and the fact was not captured.

**What the real system runs alongside this that the code replaced with a linear proxy.** CA3 pattern completion. A
replay event, awake or asleep, is a population burst of the CA3 recurrent network (Buzsaki p.345) that starts once population firing
crosses a threshold (de la Prida et al. 2006), and a reactivated subset of a stored assembly activates the whole
ensemble through the assembly's recurrent connections (Kandel ch.54; Nakazawa et al. 2002; Guzman et al. 2016). The
LTP a replay induces grows with the number of co-active pre/post pairs (Sadowski et al. 2016). So once the partial
trace still selects its memory, the burst reinstates the ensemble and the induction is near-full, however close the
nearest competitor word sits; when the trace no longer selects it, nothing of it is reinstated.

## How the code binds to it

- The bout still runs the Amendment-4 read R (kept on the record).
- Per content role, the concept units' matched-filter drive (the store's own cleanup read) drives the composer's
  Izhikevich concept bank (peak-normalized, the feedback-inhibition stand-in already used by the spiking cleanup) at
  its graded operating point; the unit that fires most is the reinstated item; a silent or tied competition
  reinstates nothing.
- The reinstated items are re-bound and bundled on the composer's own resonate-and-fire work registers
  (`_compose_phases`, the op that encoded the fact) and read back as spike phases: the reinstated ensemble.
- R_c = its in-phase coherence with the block's stored increment (Sadowski's pairing count). The bout induces
  e <- e + R_c (1 - e) (`BRAIN_AWAKE_REPLAY_COMPLETION`). The night's SWR epoch, with its own flag
  (`BRAIN_SLEEP_REPLAY_COMPLETION`), uses R_c for its re-tag, its SWR-coupled DA and its downscaling protection.
  `BRAIN_REPLAY_COMPLETION_LESION` keeps every read and uses R in both routes.
- No new constant. The threshold is where the partial trace stops selecting its own items (the decayed increment
  against the baseline and the vocabulary crosstalk), not a set number.
- No `constraints_config` / `protocol`: nothing here is a config scalar the checker can compare.

## Alternatives weighed (and why they are not the fix for this failure)

- **Replay prioritization / tag-dependent selection** (Schapiro et al. 2018; the Buzsaki "burst initiators"). Real,
  and it argues against the linear proxy too (weak items are replayed MORE). But in this scenario there is ONE managed
  fact and every bout already drives it (the arc family's I1 gate held on all six seeds: 48 bouts, a substrate read
  each time). The failure is the induction per replay, not which memory is replayed. Prioritization among several
  recent facts is a separate, unmeasured question (the prereg's declared one-fact scope).
- **Synaptic-tag lifetime.** Each bout re-sets the tag from the re-induced expression, and the night's re-tag and
  SWR-coupled DA are both set by the read at sleep onset. A longer tag lifetime cannot help a trace whose expression
  has collapsed: on the failing seed the read at sleep onset was 0.0305, so the night's DA was near tonic whatever
  the tag.
- **The read's heterogeneity origin.** The per-seed spread of the fresh read (0.21-0.49) is read-out crosstalk (how
  similar the nearest competitor word is on that seed's codes and that block's baseline synapses), not how much of the
  trace is left. That is the argument FOR replacing the margin as the induction quantity, rather than tuning it.

## Known limits, recorded so they are not re-derived

- **Not literal CA3 collaterals.** The composer store has no recurrent collaterals; the model's recurrent path for a
  stored fact is its readout -> unbind -> cleanup -> re-bind loop, whose forward half every recall runs. The repo's
  CA3 superposed-fact attractor (`research/runners/ca3_superposed_fact_attractor.py`, capacity GO 6/6) is a
  standalone binary k-WTA runner with its own EC codes and no chat write path; routing the awake read through it would
  need a second store for every told fact. Not used; named as the rung that would make the completion literally CA3.
- **The competition has no lateral inhibition.** The concept bank's neurons are driven independently (no synapses),
  so the "winner" is the most-firing unit of a peak-normalized drive; spike counts are small (5-8 in 120 steps at the
  graded point) and near-ties resolve as no reinstatement. An equal drive to every unit still produces a winner (the
  bank's heterogeneous thresholds). A lateral-inhibition WTA is the named next rung.
- **One pass.** No multi-cycle settle within the ripple.
- **Wrong items are credited only along the stored increment.** The LTP a wrong reinstated item would write elsewhere
  is dropped by the ledger's bookkeeping (it can only understate confabulation risk); every bout records the items.
- **Polarity is not reinstated.** Its 2-word competition resolves at any trace strength, so it would add a floor
  that carries no evidence about this fact (measured: R_c 0.081 on the bare baseline with polarity in).

⚠️ **Provenance honesty.** The Kandel and Buzsaki anchors are in the local corpus. The PMID/PMC sources are not; they
were read as abstracts through PubMed on 2026-09-25 (Sadowski in full text, as recorded in the sibling binding).
