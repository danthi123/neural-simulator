---
type: biology
id: sleep-load-dependent-renormalization
mechanism: Sleep's synaptic renormalization is the price of the day's plasticity -- its size is set by how much the preceding wake potentiated, it stops when that is paid back, and the traces the night reactivates least pay most -- so an unrehearsed recent memory fades as a function of how much else is learned after it (retroactive interference), not of the number of nights alone
status: established
last_verified: 2026-09-25
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "requiring that some excitatory inputs be reduced"
    note: "ch.44 (Tononi & Cirelli): 'The size of many excitatory synapses is increased during learning, requiring that some excitatory inputs be reduced to avoid overexciting the target neuron.' The depression is REQUIRED BY the day's potentiation -- the basis for setting the night's amplitude from the wake's load."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "found that the size of smaller synapses in motor"
    note: "ch.44: smaller synapses are reduced during sleep, 'resulting in strong inputs being strengthened while competing weaker ones are removed' -- competition, not a uniform per-trace tax."
  - path: "PMC3921176"
    anchor: "positively correlated with the amount of the time spent exploring"
    note: "Tononi & Cirelli 2014, Neuron 81:12 (doi 10.1016/j.neuron.2013.12.025), full text read via PubMed Central 2026-09-25: after an enriched environment the widespread increase in sleep SWA 'is positively correlated with the amount of the time spent exploring and with the cortical induction of BDNF'. Same paper: the decrease in synaptic strength and SWA during sleep 'is exponential and self-limiting'; SHY renormalizes 'based on a comprehensive sampling of its overall knowledge of the environment, rather than being biased by the particular inputs of a particular waking day', because otherwise 'one would remember the new acquaintance and forget old friends' (why old knowledge -- the model's baseline and build-time blocks -- is not depressed); 'new memories that fit less well with previous knowledge are less activated and are competitively down-selected' (the protection by the night's own read). Basis of delta = dW/W and of the old-knowledge exemption."
  - path: "PMID:15184907"
    anchor: "sleep homeostasis indeed has a local component, which can be triggered by a learning task"
    note: "Huber, Ghilardi, Massimini & Tononi 2004, Nature 430:78 (doi 10.1038/nature02663), abstract read via PubMed 2026-09-25: a learning task involving specific regions raises SWA locally in the following sleep. The amount of renormalization follows the amount learned."
  - path: "PMID:14744216"
    anchor: "recently formed memories that have not yet had a chance to consolidate are vulnerable to the interfering force of mental activity and memory formation"
    note: "Wixted 2004, Annu Rev Psychol 55:235 (doi 10.1146/annurev.psych.55.090902.141555), abstract read via PubMed 2026-09-25: everyday forgetting is retroactive interference from later memory formation, 'even if the interfering activity is not similar to the previously learned material'; the account explains why sleep, alcohol and benzodiazepines improve memory for a recently learned list. Why a protocol with no later learning should NOT show forgetting."
  - path: "PMID:11740500"
    anchor: "LTP is normally a persistent process that is actively reversed by NMDA receptor activation"
    note: "Villarreal, Do, Haddad & Derrick 2002, Nat Neurosci 5:48 (doi 10.1038/nn776), abstract read via PubMed 2026-09-25: daily NMDA-receptor blockade after induction blocked LTP decay over a week, including the late phase, and improved spatial memory retention. The decay of a potentiated trace is driven by later plasticity, not by time."
  - path: "PMC6758050"
    anchor: "it was reversed when animals were exposed repeatedly to an enriched environment beginning 14 d post-HFS"
    note: "Abraham, Logan, Greenwood & Dragunow 2002, J Neurosci 22:9626 (doi 10.1523/JNEUROSCI.22-21-09626.2002), abstract read via PubMed 2026-09-25: stable LTP lasting months in the dentate gyrus was reversed by repeated enriched-environment exposure; 'under naturalistic environmental conditions, LTP may normally be retained in the hippocampus for only short periods of time.' New experience is what removes an old potentiation."
  - path: "PMID:9732871"
    anchor: "exploration of a new, non-stressful environment rapidly induces a complete and persistent reversal"
    note: "Xu, Anwyl & Rowan 1998, Nature 394:891 (doi 10.1038/29783), abstract read via PubMed 2026-09-25: novel exploration reversed recently induced early-phase LTP in CA1 but not long-established LTP and not baseline transmission. Wake-time depotentiation by later learning -- a second interference route, NOT built here (named next candidate)."
  - path: "PMID:16824917"
    anchor: "sleep protects declarative memories from subsequent associative interference"
    note: "Ellenbogen, Hulbert, Stickgold, Dinges & Thompson-Schill 2006, Curr Biol 16:1290 (doi 10.1016/j.cub.2006.05.024), abstract read via PubMed 2026-09-25: the human declarative benefit of sleep is unmasked by later interfering learning."
  - path: "PMC8735725"
    anchor: "We changed the 1 week interval of Experiment 2 to 3 days to avoid the floor effects found in said experiment"
    note: "Rivera-Lares, Logie, Baddeley & Della Sala 2022, Mem Cognit 50:1706 (doi 10.3758/s13421-021-01271-1), full text read via PubMed Central 2026-09-25: cued recall of sentences presented two to six times fell to floor by one week, so the authors moved to three days, where recall was above floor. The human anchor for the horizon: a sentence heard once or twice is typically still recalled at 3 days and near floor by a week (while living a normal, interference-filled week)."
  - path: "doi:10.1016/j.jml.2018.05.008"
    anchor: "textbase level memories were retained until about seven days when memory suddenly dropped to around chance levels"
    note: "Fisher & Radvansky 2018, J Mem Lang 102:130 'Patterns of forgetting', abstract read via the Consensus index 2026-09-25 (not in PubMed): propositional (textbase) memory for a narrative held for about a week, then dropped; event-model memory stayed high. A second human anchor for a ~1-week horizon of a once-read proposition."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "data sets available on forgetting in humans"
    note: "p.123: 'the power law function is the best fit to the large data sets available on forgetting in humans' -- retention falls gradually, with no fixed number of days at which a memory is gone."
implemented_by:
  - webapp/sleep_replay_capture.py
  - research/runners/_sleep_load_renorm_design.py
findings:
  - research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md
---

# Sleep renormalization is paid for by the day's learning; later learning is what makes a recent trace fade

**What the model had replaced with a constant.** r2's `BRAIN_SLEEP_DOWNSCALING` multiplies every managed trace by
1 - 0.18 (1 - R) each night. The 0.18 is de Vivo et al. 2017's sleep-versus-wake difference after an ordinary waking
period of a mouse, which is full of experience. Applied as a constant, it charges a night that follows a day with no
learning at all the same 18 %, and it charges it per trace, so a fact's fate depends only on how many nights pass. The
three-night battery (`d3w`) is exactly that case: after the telling, the brain learns nothing for three days.

**What the real system runs alongside.** Two things the constant stood in for, both set by later learning:
- The night's depression is the price of the preceding day's potentiation. It scales with how much was explored or
  learned (Tononi & Cirelli 2014; Huber et al. 2004) and is self-limiting. The traces reactivated least pay most
  (down-selection), and old, integrated knowledge is not charged for the particulars of one day.
- Everyday forgetting of recent memories is retroactive interference from later memory formation, even dissimilar
  (Wixted 2004). At the synapse, LTP decay is actively driven by later NMDA-receptor-dependent plasticity (Villarreal
  et al. 2002) and new experience (Abraham et al. 2002; Xu et al. 1998).

## How the code binds to it (`BRAIN_SLEEP_LOAD_RENORM`, default OFF)

Each night, after the reactivation, the amplitude is read on the store: delta = dW / W, where dW is the expressed
learned strength of the managed blocks written since the previous night and W is the store's total synaptic strength
(every block, build-time ones included). Each managed increment is multiplied by 1 - delta (1 - R_i), r2's protection.
The baseline and the build-time blocks are not depressed. `BRAIN_SLEEP_LOAD_RENORM_LESION` keeps the read and applies
delta = 0. No `constraints_config`: the only new quantity is measured, not set.

## Known limits, recorded so they are not re-derived

- **The store grows.** Each fact gets its own block, so every fact told adds to W as well as to dW. A real brain
  renormalizes its total back each night, so this model's delta falls faster with accumulated knowledge than a real
  brain's would. The direction is towards retention.
- **Interference here acts only through the night's shared renormalization.** Wake-time depotentiation by novel
  experience (Xu 1998) and similarity-dependent overlap on shared synapses are not built. Both are named candidates.
- **The dose is the environment's.** How much a person learns in a day is not something the tiny-demo conversation can
  reproduce, so the protocol reports a dose series; mapping a dose onto a human day is not claimed.

## The three-night criterion (r2 SHY1) is not what the biology predicts

A weak fact told once and followed by three days in which nothing else is learned is the laboratory's minimum-
interference condition. Wixted's account predicts little forgetting there, and in this mechanism the nights after an
empty day cost nothing. Human retention of a sentence heard once or twice is still above floor at three days, and near
floor at about a week, while living an ordinary interference-filled week (Rivera-Lares et al. 2022; Fisher &
Radvansky 2018; the power-law curve in Buzsáki p.123). So "gone after three idle nights" is not a biological
requirement. What the biology does require: retention in a vacuum, loss that grows with later learning, and protection
by salience and by re-mention.
