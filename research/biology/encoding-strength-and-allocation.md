---
type: biology
id: encoding-strength-and-allocation
mechanism: How well a new memory is formed is decided at encoding, and it is graded -- by how deeply the input is processed (attended and tied to what is already known), by the brain's state and motivation at that moment (neuromodulators such as acetylcholine and dopamine raise the strength of afferent input and of synaptic modification), by which neurons are allocated to the trace (those most excitable just before learning win a competition), and by how well the new pattern is separated from stored ones (dentate gyrus); activity at encoding predicts later remembering, and a brief quiet rest right after learning strengthens the new trace
status: proposed
last_verified: 2026-09-25
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "cally important for determining how well the learned"
    note: "ch.52: encoding is the process by which new information is acquired and processed during the formation of a new memory; the extent of this processing is critically important for how well the material will be remembered."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "plished by attending to the information and associating"
    note: "ch.52, same passage: 'deep' encoding (Craik & Lockhart) is accomplished by attending to the information and associating it with memories already established; encoding is also stronger when one is motivated to remember (emotional or behavioural relevance, or association with something meaningful)."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "remembered items, compared with forgotten items,"
    note: "ch.52 (the subsequent-memory paradigm, Figure 52-6 after Wagner et al. 1998): items later remembered are associated with greater hippocampal activity during encoding than items later forgotten, together with prefrontal, retrosplenial and parietal cortex."
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "Absent-mindedness results from a lack of attention"
    note: "ch.52 (the 'seven sins'): absent-mindedness during encoding -- a lack of attention to the immediate experience -- is a likely source of common memory failures. What is not attended is weakly encoded; that is a normal failure, not an instrument error."
  - path: "PMID:9712582"
    anchor: "the ability to later remember a verbal experience is predicted by the magnitude of activation in left prefrontal and temporal cortices during that experience"
    note: "Wagner et al. 1998, Science 281:1188 (doi 10.1126/science.281.5380.1188), abstract read via PubMed 2026-09-25: the subsequent-memory effect for single words."
  - path: "PMID:17446403"
    anchor: "Changes in CREB function influenced the probability that individual lateral amygdala neurons were recruited into a fear memory trace"
    note: "Han et al. 2007, Science 316:457 (doi 10.1126/science.1139438), abstract read via PubMed 2026-09-25: a competitive model of memory formation; eligible neurons are selected into a trace as a function of their relative CREB activity at the time of learning."
  - path: "PMID:25102562"
    anchor: "neuronal memory allocation is based on relative neuronal excitability immediately before training"
    note: "Yiu et al. 2014, Neuron 83:722 (doi 10.1016/j.neuron.2014.07.017), abstract read via PubMed 2026-09-25 (also bound in prp-competition-and-locality)."
  - path: "PMID:17303747"
    anchor: "signals from the entorhinal cortex can be decorrelated both by changes in coincidence patterns in the dentate gyrus and by recruitment of nonoverlapping cell assemblies in CA3"
    note: "Leutgeb, Leutgeb, Moser & Moser 2007, Science 315:961 (doi 10.1126/science.1135801), abstract read via PubMed 2026-09-25: pattern separation at encoding keeps a new pattern from overlapping stored ones."
  - path: "PMC2659740"
    anchor: "Acetylcholine has been shown to increase the strength of afferent input relative to feedback"
    note: "Hasselmo 2006, Curr Opin Neurobiol 16:710 (doi 10.1016/j.conb.2006.09.002), abstract read via PubMed 2026-09-25: muscarinic and nicotinic receptors take part in encoding new memories; ACh also increases the modification of synapses. An encoding-state signal distinct from novelty dopamine."
  - path: "PMC3287976"
    anchor: "the magnitude of hippocampal-LO correlations during posttask rest predicts individual differences in later associative memory"
    note: "Tambini, Ketz & Davachi 2010, Neuron 65:280 (doi 10.1016/j.neuron.2010.01.001), abstract read via PubMed 2026-09-25: hippocampal-cortical coupling during rest right after encoding relates to later memory."
  - path: "PMID:22829465"
    anchor: "wakeful resting led to significant enhancement of memory after a 15- to 30-min period and also after 7 days"
    note: "Dewar, Alber, Butler, Cowan & Della Sala 2012, Psychol Sci 23:955 (doi 10.1177/0956797612441220), abstract read via PubMed 2026-09-25: 10 min of quiet rest right after hearing a story improved memory a week later, even with no retrievals in between."
findings:
  - research/findings/2026-09-25-prioritized-memory-remember-what-matters-DESIGN.md
---

# How strongly a memory is formed is decided at encoding, and the brain's state decides it

**Why this entry exists (2026-09-25).** The fi battery of the DA tag-capture + sleep-replay pair (prereg Amendment 6,
six seeds) lost a plainly told fact by the first morning on seeds 43 and 101, even in the arm where nothing else was
learned. The design doc listed under `findings` (section 2a) reads the per-seed records: every telling wrote its block,
but the block's own reactivation read five minutes after it was written -- the encoded read -- ranged about 0.04 to
0.32 across the four seeds that wrote it at the same floor gain, and the seeds at the low end lost it. The encoded
strength was set by the seed's realization of the store (the slot's seeded baseline draw, the word codes, the read
noise; which one is not in the record), not by anything the fact was worth.

**The constants this replaces (read in the code).** The composer writes a told fact into the next free block at the
DA write gain (`research/runners/one_brain_composer.py` `_write_block`); with the ledger on, the block's synapses carry
a baseline drawn once per (seed, block index) at the same expected magnitude as a gain-1 increment
(`webapp/da_tag_capture.py`, `BETA_BASELINE = 1.0`, "increment ~ baseline"); the per-block read is a D = 128 matched
filter (`research/runners/brain_conversational_agent.py` default). No allocation competition picks the units, no
separation step acts on the chat store's write, and no encoding-state signal other than novelty dopamine changes the
write.

**What the real system runs.** Encoding strength varies with depth of processing, attention and motivation (Kandel
ch.52); activity during encoding predicts later memory (Wagner 1998; Kandel Figure 52-6); the neurons that hold a trace
are selected by an excitability competition at learning (Han 2007; Yiu 2014); the dentate gyrus decorrelates a new
input from stored ones (Leutgeb 2007); acetylcholine sets an encoding mode that strengthens afferent input and
plasticity (Hasselmo 2006); a quiet rest right after learning strengthens the new trace (Tambini 2010; Dewar 2012).
Failure to encode what was not attended is normal (Kandel ch.52, absent-mindedness), so a weakly encoded background
detail is a brain outcome, not a void run.

## How the code is expected to bind to it (design only, nothing implemented)

Design doc Step 0 (a graded encoding read per telling in every arm), Step 0b (attribute the spread of the encoded
read at a fixed write gain to slot, content, read noise and ensemble size) and Step 1a (encoding strength set by the
brain's state rather than by where the fact lands; the candidate built first is picked by Step 0b's registered
decision rule), gate WM0. No `constraints_config`: the biology constrains encoding strength as a graded dependence, not
a scalar equality.

⚠️ **Provenance honesty.** The Kandel anchors are in the local corpus. The PMID/PMC sources are not; they were read as
abstracts through PubMed on 2026-09-25.
