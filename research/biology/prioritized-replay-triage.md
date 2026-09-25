---
type: biology
id: prioritized-replay-triage
mechanism: Offline reactivation is not uniform -- the hippocampus replays some memories far more than others (those marked while awake by reward, salience, expected use and the content of awake ripples, and among them the weakly learned ones that most need it), across many ripples in the four or five NREM cycles of a night and over several nights, and sleep uses those replays to keep, transform or drop each memory (memory triage)
status: proposed
last_verified: 2026-09-25
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "non-REM/REM cycles with a period of"
    note: "Cycle 7: typically four or five non-REM/REM cycles of 70-90 min occur within a night. The model runs one SWR epoch per night (sleep-replay-capture r2)."
  - path: ~/Projects/sim-catalog/references/textbooks/buzsaki-rhythms/Buzsaki-RhythmsOfTheBrain-2006.txt
    anchor: "activity spreads along the path of the strongest synaptic weights"
    note: "p.349: replay content follows the weights; the most-activated assemblies become burst initiators (already bound in research/biology/awake-replay-tag-capture.md). Selection is the substrate's response, not a host list."
  - path: "PMC5826623"
    anchor: "sleep-dependent memory processing is unlikely to be complete after just a single night"
    note: "Stickgold & Walker 2013, Nat Neurosci 16:139 (doi 10.1038/nn.3303), full text via PubMed Central 2026-09-25: 'memory triage' -- sleep selects which memories to keep and which to forget from prior waking salience tags, then stabilizes, integrates or generalizes them; the repeating cycles of sleep stages across the night and across nights matter."
  - path: "https://consensus.app/papers/details/c9a3b8b259e7522087c0fe107c167920/"
    anchor: "SPW-Rs continued to replay those trial blocks that were reactivated most frequently during waking SPW-Rs"
    note: "Yang, Sun, Huszar, Hainmueller, Kiselev & Buzsaki 2024, Science (Selection of experience for memory by hippocampal sharp wave ripples), abstract read via the Consensus index 2026-09-25 (not returned by the PubMed title search): awake ripple content during reward consumption tags which experiences post-experience sleep replays."
  - path: "PMC6156217"
    anchor: "Objects that were remembered less well were replayed more during the subsequent rest period"
    note: "Schapiro, McDevitt, Rogers, Mednick & Norman 2018, Nat Commun 9:3920 (doi 10.1038/s41467-018-06213-1), abstract via PubMed 2026-09-25: rest replay prioritizes weakly learned items; more replay predicted better memory 12 h later, and replay predicted improvement only for participants who slept. Priority includes NEED, not only strength."
  - path: "PMC2807414"
    anchor: "rat hippocampal CA3 principal cells are significantly more active during SWRs following receipt of reward"
    note: "Singer & Frank 2009, Neuron 64:910 (doi 10.1016/j.neuron.2009.11.016), abstract via PubMed 2026-09-25: reward enhances SWR reactivation of the paths that led to it."
  - path: "PMC6013068"
    anchor: "only reverse replays increased their rate at increased reward or decreased their rate at decreased reward"
    note: "Ambrose, Pfeiffer & Foster 2016, Neuron 91:1124 (doi 10.1016/j.neuron.2016.07.047), abstract via PubMed 2026-09-25: replay rate is graded by the change in reward."
  - path: "PMC6203620"
    anchor: "an agent accesses memories of locations sequentially, ordered by utility"
    note: "Mattar & Daw 2018, Nat Neurosci 21:1609 (doi 10.1038/s41593-018-0232-z), abstract via PubMed 2026-09-25: a normative account in which replay priority is gain x need; it unifies planning, learning and consolidation roles of replay. Cited for the gain x need form, not as a mechanism."
  - path: "PMID:23575863"
    anchor: "covert reactivation is a major factor determining the selectivity of memory consolidation"
    note: "Oudiette, Antony, Creery & Paller 2013, J Neurosci 33:6672 (doi 10.1523/JNEUROSCI.5497-12.2013), abstract via PubMed 2026-09-25: low-value associations were forgotten more, and cued reactivation rescued them -- during sleep the whole low-value set, during wake only the cued items."
  - path: "PMC6623736"
    anchor: "Subjects expecting the retrieval displayed a robust increase in slow oscillation activity and sleep spindle count during postlearning slow-wave sleep"
    note: "Wilhelm et al. 2011, J Neurosci 31:1563 (doi 10.1523/JNEUROSCI.3575-10.2011), abstract via PubMed 2026-09-25: expected future use changes the night's own processing."
  - path: "PMC3768102"
    anchor: "Consolidation originates from reactivation of recently encoded neuronal memory representations"
    note: "Rasch & Born 2013, Physiol Rev 93:681 (doi 10.1152/physrev.00032.2012), abstract via PubMed 2026-09-25."
findings:
  - research/findings/2026-09-25-prioritized-memory-remember-what-matters-DESIGN.md
  - research/findings/2026-08-09-prioritized-replay-fixed-k-fails-at-scale-compute-half-NEGATIVE.md
---

# Replay spends its effort unevenly, and the unevenness is the selection

**The constants this replaces (read in the code, 2026-09-25).** `SleepReplayCapture` (`webapp/sleep_replay_capture.py`)
runs ONE SWR epoch per night and drives EVERY managed block exactly once, identically; what differs is only the
store's read-back R of each block. The awake-rest bout (`webapp/awake_replay_capture.py`) does the same. So the replay
has no selection of its own: a strong trace reads high and is re-tagged, a weak one reads low and is not, whatever it
was worth. The module docstring records why only one epoch runs: a five-cycle variant resurrected noise-level traces
on the fake substrate, because the brake (sleep downscaling) was missing. r2 and r3 have since built that brake
(`BRAIN_SLEEP_DOWNSCALING`, `BRAIN_SLEEP_LOAD_RENORM`, default OFF).

**What the real system runs.** Several cycles a night and several nights (Buzsaki; Stickgold & Walker); replay content
biased by reward (Singer & Frank; Ambrose), by what awake ripples reactivated (Yang 2024), by expected use (Wilhelm), by
value (Oudiette), and toward weakly learned items (Schapiro 2018). Mattar & Daw's gain x need form summarizes the
bias: replay what is both worth having and at risk.

**A prior result in this repo to respect.** The 2026-08-09 teacher-loop finding: replay prioritized by a neural
forgetting-risk read beat random selection but failed retention at a fixed small budget as the number of memories
grew (coverage). The design here does not fix a budget: priority biases a stochastic competition across many bursts,
so every trace keeps a chance, and important, at-risk traces get most of the draws.

## How the code is expected to bind to it (design only, nothing implemented)

Design doc step 4: an excitability mark set at encoding by the importance channels and decaying within a day; SWR
bursts in which managed blocks compete (a lateral-inhibition WTA over their triggers) with initiation biased by that
mark and by need; one epoch per NREM cycle and awake bursts in rest pauses; the r3 renormalization, limited local PRPs
and graded capture as the brake. The sleep/wake clock stays the body's (host) clock. No `constraints_config`.

⚠️ **Provenance honesty.** The Buzsaki anchors are in the local corpus. The rest were read as abstracts through PubMed
on 2026-09-25 (Stickgold & Walker 2013 in full text); Yang et al. 2024 through the Consensus index, whose URL is the
locator given.
