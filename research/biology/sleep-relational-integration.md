---
type: biology
id: sleep-relational-integration
mechanism: Offline (sleep) replay links memories that share an element -- A->B learned in one episode and B->D in another come to support an A->D inference that was never experienced, through co-reactivation of the overlapping memories during sharp-wave ripples and plasticity during that co-reactivation; premise memory is kept while the inference develops, and without the offline period (tested at once) the inference is absent
status: established
last_verified: 2026-09-29
current_finding: research/findings/2026-09-29-sleep-transitive-inference-PREREGISTRATION.md
implemented_by:
  - research/runners/sleep_transitive_inference.py
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "by way of the overlapping events, supporting the capacity for"
    note: "Kandel 6e ch.52, Fig. 52-9 (Eichenbaum & Cohen 2014): relational networks are formed via associations between events and episodes by way of the overlapping events, supporting links between indirectly related events. The overlap (the shared item B) is the route by which A and D become related."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "hippocampus binds in memory the separate elements"
    note: "Kandel 6e ch.52: the hippocampus binds the separate elements of experiences, encoding events as relational maps of items within spatial and temporal contexts. Items and contexts are separate elements of one episode (the store's item and context assemblies)."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "activity spreads along the path of the strongest synaptic weights"
    note: "Buzsaki p.349: replay content follows the learned weights. A replay that enters the shared item B continues along B's learned links, which is how an event can join A and D."
  - path: "PMC1863467"
    anchor: "the 20-min group showed no evidence of inferential ability (52%), whereas the 12- and 24-hr groups displayed highly significant relational memory developments"
    note: "Ellenbogen, Hu, Payne, Titone & Walker 2007, PNAS 104:7723 (doi 10.1073/pnas.0700094104), abstract via PubMed 2026-09-29. Premise pairs kept by every group (>85%); inference absent when tested 20 min after learning and present after 12 h awake or asleep; sleep adds a boost for the most distant pair (B>E: sleep 93%, wake 69%). So: inference develops OFFLINE, sleep helps most, and it is absent at once. The model's `wake` arm (tested at once) corresponds to the 20-min group, not to 12 h awake: an awake-rest replay arm is not in this family (declared in the prereg)."
  - path: "PMC7116148"
    anchor: "coactivation of hippocampal cells in sharp-wave/ripples represent inferred relationships"
    note: "Barron et al. 2020, Cell 183:228 (doi 10.1016/j.cell.2020.08.035), abstract via PubMed 2026-09-29: during resting behaviour, hippocampal co-activation in SWRs represents inferred relationships, 'joining-the-dots' between events not observed together. The substrate-level correlate of the mechanism: co-reactivation of A and D in replay events."
  - path: "PMID20176120"
    anchor: "the binding of items that were not directly learned together, reorganizing them for flexible use at a later time"
    note: "Lau, Tucker & Fishbein 2010, Neurobiol Learn Mem 93:554 (doi 10.1016/j.nlm.2010.02.003), abstract via PubMed 2026-09-29: two face-object lists sharing the objects (A-B, C-B); after a NREM nap, better direct-associative retention AND better relational (A-C) memory than an equivalent wake period; only slow-wave sleep predicted relational performance. The paradigm the plan's A->B / B->D design follows."
  - path: "PMC4869174"
    anchor: "The STDP curve is symmetric and broad (half-width"
    note: "Mishra, Kim, Guzman & Jonas 2016, Nat Commun 7:11552 (doi 10.1038/ncomms11552), abstract via PubMed 2026-09-29: CA3-CA3 STDP is LTP independent of temporal order, symmetric and broad (half-width ~150 ms). The store in this family uses the engine's ASYMMETRIC Bi-Poo rule (tau 20 ms), the rule under which the project's CA3 band learned its forward order; that is a declared deviation, and a symmetric-rule arm is a named follow-up."
  - path: "PMC3444305"
    anchor: "recurrent similarity computation, a process that facilitates the discovery of higher-order relationships"
    note: "Kumaran & McClelland 2012, Psychol Rev 119:573 (doi 10.1037/a0028681), abstract via PubMed 2026-09-29 (REMERGE): inference can arise AT RETRIEVAL through recurrence over separately stored episodes, with no offline change. The competing account; the prereg separates it from offline integration with the hub-silenced probe (retrieval-time chaining needs B active; a direct A->D link does not)."
  - path: "PMC3398403"
    anchor: "retrieval-mediated learning, whereby prior event details are reinstated during encoding of related experiences, supports participants' ability to infer relationships"
    note: "Zeithamova, Dominick & Preston 2012, Neuron 75:168 (doi 10.1016/j.neuron.2012.05.010), abstract via PubMed 2026-09-29: integration can also happen AT ENCODING, when learning B->D reinstates A->B. A third route; in this family's waking protocol the B->D episode does not reinstate A (a settle gap with spike clocks cleared separates episodes and the Y episode drives only Y, B, D), so the pre-sleep weight read (A->D at its initial value, equal to the foil) checks that no encoding-time integration occurred."
protocol:
  - rule: "A and D never co-occur awake"
    why: "The capability is an inference about a relation never experienced; a waking co-occurrence would make it recall"
    check: enc_settle gte 300
  - rule: "The test context is novel"
    why: "A test in D's own context (Y) could retrieve D by context association; Z is never encoded"
    check: probe_context eq Z
---

# Sleep links memories that share an element (relational integration, transitive inference)

**What the biology says.** Premise pairs that share an item (A-B, B-C) are learned in separate episodes; tested at
once, people cannot infer the unlearned relation (A-C), though they keep the premises; after an offline period they can,
and sleep helps most (Ellenbogen 2007; Lau 2010, where only slow-wave sleep predicted the relational score). In mice,
hippocampal cells that code the two ends of an inferred relation co-activate in sharp-wave ripples during rest (Barron
2020). The textbook account: the hippocampus binds the elements of each episode, and relational networks form through
the elements episodes share (Kandel ch.52, Fig. 52-9). Two other routes exist and must be told apart from offline
integration: inference computed at retrieval by recurrence over separate episodes (REMERGE, Kumaran & McClelland 2012),
and integration at encoding when a new episode reinstates an old one (Zeithamova 2012).

**What the model does.** `research/runners/sleep_transitive_inference.py`: an AdEx CA3 store (the Ecker 2022 CA3
pyramidal preset) whose assemblies are items and contexts, every ordered pair anatomically connected, weak and plastic.
Episodes X->A->B and Y->B->D are written by the engine's STDP; A and D never share an episode. During the night,
non-specifically started SWR events propagate through what was learned, with STDP on, so an event that enters B from A
continues to D and the A->D synapses see pre-before-post pairings. The test drives A with a novel context Z; the
hub-silenced probe (B held silent) reads only a direct A->D route, which retrieval-time chaining through B cannot supply.

**Declared deviations.** (1) The plasticity rule is the engine's asymmetric Bi-Poo STDP (tau 20 ms), not the symmetric
broad CA3-CA3 rule of Mishra 2016. (2) SWR events are started by a host cue on a randomly chosen assembly (the Ecker GO
runner's protocol; its no-seed control is silent), not by intrinsic CA3 dynamics. (3) The `wake` arm is tested at once
(Ellenbogen's 20-min group); an awake-rest replay arm (Ellenbogen's 12-h wake group) is not in this family.
