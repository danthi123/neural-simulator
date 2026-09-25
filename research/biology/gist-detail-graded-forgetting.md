---
type: biology
id: gist-detail-graded-forgetting
mechanism: Forgetting is graded and selective, not all-or-none per experience -- the components of one experience are kept or lost separately (central, emotional or gist elements survive; peripheral detail fades first), gist traces outlast verbatim ones, systems consolidation turns detailed hippocampal episodes into gist-like cortical versions (fast when a schema exists), and a regulated forgetting process removes what is not needed; this transience is adaptive, because the goal of memory is to guide future decisions, not to transmit every detail
status: proposed
last_verified: 2026-09-25
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "a memory system that automatically retains"
    note: "ch.52 (the 'seven sins of memory'): a memory system that automatically retained every detail of every experience could result in an overwhelming clutter of useless trivia (Luria's mnemonist Shereshevski); many memory imperfections have adaptive value."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "data sets available on forgetting in humans"
    note: "p.123: the power-law function is the best fit to the large human forgetting data sets -- retention falls gradually with no fixed day at which a memory is gone (reused anchor, see research/biology/sleep-load-dependent-renormalization.md)."
  - path: "PMC5846336"
    anchor: "sleep may preferentially promote memory for gist over detail"
    note: "Payne, Stickgold, Swanberg & Kensinger 2008, Psychol Sci 19:781 (doi 10.1111/j.1467-9280.2008.02157.x), full text via PubMed Central 2026-09-25: across 12 h of wake both the negative object and its neutral background faded; across 12 h with sleep the emotional object was preserved and the background was not (negative-object general recognition 68% after sleep vs 44% after wake; background 38% vs 38%). The authors propose the components become unbound in sleep so the most salient one is kept."
  - path: "PMC4749834"
    anchor: "memory for naturalistic events (film clips) underwent a time-dependent loss of peripheral details, while memory for central details (the core or gist of events) showed significantly less loss"
    note: "Sekeres et al. 2016, Learn Mem 23:72 (doi 10.1101/lm.039057.115), abstract via PubMed 2026-09-25: over 7 days; reminders reinstated peripheral details (loss is partly retrieval failure); retrieving a memory soon after encoding prevented the loss of both kinds."
  - path: "PMID:21729403"
    anchor: "hippocampally dependent, episodic, or context-specific memories transform into semantic or gist-like versions that are represented in extra-hippocampal structures"
    note: "Winocur & Moscovitch 2011, J Int Neuropsychol Soc 17:766 (doi 10.1017/S1355617711000683), abstract via PubMed 2026-09-25: the transformation hypothesis -- detailed episodes keep needing the hippocampus; their semantic versions do not."
  - path: "PMC4815269"
    anchor: "false memories outlasting true memories"
    note: "Reyna, Corbin, Weldon & Brainerd 2016, J Appl Res Mem Cogn 5:1 (doi 10.1016/j.jarmac.2015.12.003), abstract via PubMed 2026-09-25: fuzzy-trace theory -- independent verbatim and gist traces; gist supports acceptance of meaning-consistent but unpresented sentences. The risk side of gist: it must not be allowed to produce confabulated detail."
  - path: "PMC4671075"
    anchor: "Gist is stable, less sensitive to interference, and easier to manipulate"
    note: "Corbin, Reyna, Weldon & Brainerd 2015, J Appl Res Mem Cogn 4:344 (doi 10.1016/j.jarmac.2015.09.001), abstract via PubMed 2026-09-25: gist captures the functionally significant essence of information."
  - path: "PMID:7624455"
    anchor: "the neocortex learns slowly to discover the structure in ensembles of experiences"
    note: "McClelland, McNaughton & O'Reilly 1995, Psychol Rev 102:419 (doi 10.1037/0033-295X.102.3.419), abstract via PubMed 2026-09-25: complementary learning systems -- rapid hippocampal storage, reinstatement interleaved into slow neocortical learning (already bound as research/biology/cls-interleaved-consolidation.md)."
  - path: "PMID:17412951"
    anchor: "systems consolidation can occur extremely quickly if an associative"
    note: "Tse et al. 2007, Science 316:76 (doi 10.1126/science.1135935), abstract via PubMed 2026-09-25: with a pre-existing neocortical schema, one-trial paired associates became hippocampus-independent within a day or two."
  - path: "PMID:28641107"
    anchor: "the goal of memory is not the transmission of information through time, per se"
    note: "Richards & Frankland 2017, Neuron 94:1071 (doi 10.1016/j.neuron.2017.04.037), abstract via PubMed 2026-09-25: transience enhances flexibility and prevents overfitting to specific past events; 'the goal of memory is to optimize decision-making'."
  - path: "PMID:23369831"
    anchor: "a brain-wide well-regulated decay process, occurring mostly during sleep, systematically removes selected memories"
    note: "Hardt, Nader & Nadel 2013, Trends Cogn Sci 17:111 (doi 10.1016/j.tics.2013.01.001), abstract via PubMed 2026-09-25: active, regulated forgetting; in pattern-separating regions such as the hippocampus, decay rather than interference causes most forgetting."
  - path: "PMC4083655"
    anchor: "the bidirectional modulation of a small subset of dopamine neurons (DANs) after olfactory learning regulates the rate of forgetting"
    note: "Berry, Cervantes-Sandoval, Nicholas & Davis 2012, Neuron 74:530 (doi 10.1016/j.neuron.2012.04.007), abstract via PubMed 2026-09-25: in Drosophila dopamine serves acquisition through one receptor (dDA1) and forgetting through another (DAMB)."
findings:
  - research/findings/2026-09-25-prioritized-memory-remember-what-matters-DESIGN.md
---

# The parts of a memory fade at different rates, and the gist outlasts the detail

**The constants this replaces (read in the code, 2026-09-25).** A told fact is ONE composer block whose synapses carry
agent, action and patient superposed (`research/runners/one_brain_composer.py`), and the sleep route's read R is the
MINIMUM cleanup margin over the three roles (`webapp/sleep_replay_capture.py` `reactivation_strength`). So a fact is
kept or lost whole; there is no state in which the core is kept and a detail is lost. Separately, the episodic organ
(topic-level familiarity keyed on the agent) is not managed by the tag-and-capture ledger and keeps every topic it
formed (prereg Amendment 7's own prediction for its `wd_epi` arm), which is equal, permanent storage at the topic
level.

**What the real system runs.** Components processed separately, emotional/central ones kept and background detail
dropped across sleep (Payne 2008; Sekeres 2016); gist traces more stable than verbatim ones (Corbin 2015) but a source
of false memory if unchecked (Reyna 2016); transformation of detailed episodes into gist-like cortical versions
(Winocur & Moscovitch 2011), fast when a schema exists (Tse 2007; McClelland 1995 for the slow default); and regulated
forgetting (Hardt 2013; Berry 2012), which is adaptive (Richards & Frankland 2017; Kandel ch.52).

## How the code is expected to bind to it (design only, nothing implemented)

Design doc steps 6-8: the core predicate and each peripheral detail on separate synapse sets with their own tags and
reads (no min over roles); the local priority mark of `importance-tagging-at-encoding` deciding which component is
kept; the episodic / common-ground topic trace carrying the gist, with a decay of its own; replay-written transfer to a
slow cortical store for important and schema-consistent facts. The within-fact case (one block carrying the core and a
detail) is probed with the composer's attribute role, because an aside told in its own turn is already a separate
block and cannot show the defect. The role is bound and read by default (`BrainConversationalAgent` defaults
`enable_attributed=True` and builds the onebrain composer with it); what production chat lacks is the route that
parses an attributed sentence into it (chat acquisition stores a three-word SVO; `hear_attributed`, the neural
attributed parser, has no chat caller), so that route is the declared deviation in Step 6's arms and their baseline
rows, unless Step 6 wires it into chat acquisition.
The word "consolidation" is reserved for the model
until a source lesion shows the cortical trace answers without the composer block (docs/TERMS.md). No
`constraints_config`.

⚠️ **Provenance honesty.** The Kandel and Buzsaki anchors are in the local corpus. The rest were read as abstracts
through PubMed on 2026-09-25; Payne et al. 2008 in full text.
