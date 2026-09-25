---
type: biology
id: sleep-replay-tag-capture
mechanism: Sleep sharp-wave-ripple replay works IN PARALLEL with synaptic tagging-and-capture -- NREM reactivation of the day's modified synapses re-sets their tags and co-triggers plasticity-related-protein synthesis, so a weakly-encoded trace can be captured without waking salience
status: established
last_verified: 2026-09-24
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "replay mechanism could replace the tagging mechanism"
    note: "note 28 (p.346-347): 'The sharp-wave replay mechanism could replace the tagging mechanism or the two processes could work in parallel to ensure input specificity of synaptic modification.' THE load-bearing claim: replay is the companion of the Frey-Morris tag."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "of the same neurons and synapses by the sharp-wave events"
    note: "the hours-long molecular cascade (nucleus, transcription, protein back to the synapse) is guided by 'the selective and repeated activation of the same neurons and synapses by the sharp-wave events'"
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "neuronal pathways used and modi"
    note: "WHAT is replayed: 'the neuronal pathways used and modified in the waking brain' -- selection by what the waking brain changed, read out of the weights, not a list"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "recruits the cAMP and PKA signaling pathway"
    note: "ch.54: late LTP recruits cAMP/PKA -> CREB -> new protein synthesis (the PRP route the capture needs)"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "the dopamine D1/D5 type of receptor, which has been"
    note: "ch.54: stable place fields need D1/D5 activation, 'shown to enhance the formation of late LTP through production of cAMP and activation of PKA' -- why the sleep route's PRP goes through the same D1 edge the lesions cut"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "Although a single tetanus on its own induces"
    note: "ch.54, Frey & Morris: a single (weak) tetanus alone gives only early LTP, but gives late LTP when delivered within 2-3 hours of the strong one -- the capture window the one-epoch design reproduces"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "usually begins with a rapid descent into stage N3 non-"
    note: "ch.44: sleep begins with a rapid descent into N3 -- where the one SWR epoch per night is placed"
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "reduced during sleep, resulting"
    note: "ch.44, Tononi & Cirelli: smaller synapses are reduced during sleep, 'competing weaker ones are removed' -- the downscaling companion NOT modeled, which is why only one epoch runs (see body)"
  - path: "PMC2596310"
    anchor: "is triggered if the total number of set tags is larger than a critical number"
    note: "Clopath, Ziegler, Vasilaki, Busing & Gerstner 2008, PLoS Comput Biol 4:e1000248 (doi 10.1371/journal.pcbi.1000248), Fig.1B legend; read in full text via PubMed Central 2026-09-24. Same paper, Discussion: 'The phasic dopamine signal caused by co-stimulation of dopaminergic input during tagging experiments is assumed to be proportional to the number of tags.' Basis of the SWR-coupled DA = f(reactivated tag mass)."
  - path: "PMC5313037"
    anchor: "The axon-spine interface (ASI) decreased ~18% after sleep compared with wake"
    note: "de Vivo, Bellesi, Marshall, Bushong, Ellisman, Tononi & Cirelli 2017, Science 355:507 (doi 10.1126/science.aah5982), abstract, read via PubMed 2026-09-24: the decrease was proportional to size (scaling) and 'selective, sparing synapses that were large'. Source of SHY_DELTA = 0.18 per night (r2 sleep downscaling)."
  - path: "PMC5873548"
    anchor: "connections contributing to postsynaptic spiking are protected against this synaptic weakening"
    note: "Gonzalez-Rueda, Pedrosa, Feord, Clopath & Paulsen 2018, Neuron 97:1244 (doi 10.1016/j.neuron.2018.01.047), abstract, read via PubMed 2026-09-24: in slow-wave Up states presynaptic activity alone depresses a synapse; inputs that drive the postsynaptic cell are spared. Basis of the per-block protection by the night's own reactivation read R_i."
  - path: "PMC10807868"
    anchor: "declarative memory is enhanced when sleep follows within a few hours of learning"
    note: "Gais, Lucas & Born 2006, Learn Mem 13:259 (doi 10.1101/lm.132106), abstract, read via PubMed 2026-09-24: the human declarative benefit of sleep depends on sleep coming within a few hours of learning. Bears on r2 item 1 (a fact told 4 h before sleep onset). It shows a smaller BENEFIT with delayed sleep, not total loss."
implemented_by:
  - webapp/sleep_replay_capture.py
  - webapp/da_tag_capture_chat.py
  - webapp/da_tag_capture.py
findings:
  - research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md
---

# Replay runs alongside the tag; sleep is the missing PRP supply

**The claim the code must respect.** Buzsáki (p.346-347) names the companion directly: the pathways "used and
modified in the waking brain" are replayed in sharp waves, the "selective and repeated activation of the same neurons
and synapses by the sharp-wave events" keeps the hours-long protein traffic aimed at the synapses that learned, and
(note 28) the replay mechanism "could replace the tagging mechanism or the two processes could work in parallel."
Clopath et al. 2008 give the trigger: protein synthesis starts when enough tags are set, and they model the phasic DA
released with the stimulation as proportional to the tag count. Kandel ch.54 gives the pathway (late LTP via
cAMP/PKA; D1/D5 enhances it through cAMP) and the capture window (a weak tetanus is captured within 2-3 h).

**What this repo replaced with a constant.** The v3 ledger (`webapp/da_tag_capture.py`) had exactly one PRP source,
the waking DA through the D1 pool. The night contributed nothing, so the PRP supply during sleep was the constant zero,
and an ordinary fact (DA at tonic) was gone by morning, by construction.

## How the code binds to it

- One SWR epoch at sleep onset (the engine's sleep-depth criterion). Every managed block is driven and read back by
  the store's own cleanup; the read-back strength R_i decides the replay tag (R_i x the local E-LTP amplitude) and the
  SWR-coupled DA (tonic + span x min(1, sum R)), which drives the SAME spiking D1 pool into the SAME PRP pool. Both
  DA lesions cut this edge; `BRAIN_SLEEP_REPLAY_CAPTURE_LESION` cuts the reactivation edge.
- No `constraints_config`: the constants that matter (5-min bout, D1 ceiling anchor, one epoch) are reuses of existing
  anchors or a design choice recorded below, not a scalar the biology pins to an equality the checker could compare.

## Known limit, recorded so it is not re-derived

**One epoch per night, not one per NREM cycle.** Measured 2026-09-24 on the fake-substrate test (before any brain
run): with 5 cycles 90 min apart, a fact told 8 h before sleep (read-back R = 0.035, at baseline) was captured by
cycle 4, because a sub-threshold late-phase z is expressed in the weight and raises the next cycle's read-back (a
runaway). Real sleep runs a brake alongside, synaptic downscaling (Tononi & Cirelli, Kandel ch.44). Until that
companion exists here, only the first-N3 epoch runs; with it, facts told <= 2 h before sleep are captured and facts told
>= 3 h before are not (R 0.30 -> 0.17 across the boundary), in line with the 2-3 h capture window.

## r2: the downscaling companion (default-OFF `BRAIN_SLEEP_DOWNSCALING`)

The brake named above is now built as a sub-flag, one night at a time: after each night's reactivation, each managed
block's learned increment is multiplied by 1 - 0.18 x (1 - R_i). The 18 % is de Vivo et al. 2017's sleep-vs-wake
axon-spine-interface decrease; the protection by R_i follows González-Rueda et al. 2018 (inputs that drive the
postsynaptic cell during slow-wave Up states are spared). Two modelling choices are declared, not sourced: R_i (the
store's cleanup margin) stands in for "drove postsynaptic spiking", and the baseline synapses are not downscaled
(they stand for other memories' strength; scaling them with the increment would not change the magnitude-invariant
read). r2 also runs one replay epoch per NIGHT (every 24 h of idle) instead of one per idle stretch; later NREM cycles
within a night are still not modeled.

⚠️ **Provenance honesty.** Clopath 2008 is not in the local corpus; its quotes were checked against the PMC full text
(PMC2596310) on 2026-09-24, which is why its source is marked external. The SWR-coupled VTA/SNc burst itself (the
biological source of the sleep-time DA) is represented by a declared operating point in the code, not by a cited
magnitude: no local source gives one.
