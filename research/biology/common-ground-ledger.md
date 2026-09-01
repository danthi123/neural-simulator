---
type: biology
id: common-ground-ledger
mechanism: A conversation's COMMON GROUND — the set of referents mutually established so far — is a PERSISTENT declarative record held in the hippocampal/declarative memory system, updated per grounding act and read at speak-time to drive AUDIENCE DESIGN (a referent already in common ground is REDUCED/pronominalized; a not-yet-grounded referent is INTRODUCED with a full description). Realized here as a bank of self-sustaining NMDA attractors (one bistable store per referent), latched ON by a grounding act and held by recurrence, read by a Namburi-Tye biased-competition reduce/introduce decision biased by a novelty prior.
status: established
last_verified: 2026-08-26
current_finding: research/findings/raw/_learned_common_ground_ledger_6seed.json
current_status: "FUNCTIONAL common-ground/audience-design correlate. De-risk (research/runners/_learned_common_ground_ledger_derisk.py) 1-seed numpy smoke: audience-design acc 1.000 (chance 0.5); PERMUTED grounding history -> 0.500 (tracks WHICH referents were grounded); LESION the ledger recurrence (weight 0) -> 0.500 static (frac-introduce 1.00, the store cannot HOLD); substrate-read evidence rate grounded 0.113 >> ungrounded 0.000 (the read rides real firing, not a host dict). The 6-seed pool run satisfies the project bar. WIRED into the live /api/brain-chat reply path (webapp/common_ground_drives_chat.py + webapp/server.py, DEFAULT-OFF flag BRAIN_CG_DRIVES): a grounded referent's reply LEADS with a reduced/pronominal reference, a first mention is named in full; the neural ledger-recurrence lesion (BRAIN_CG_DRIVES_LESION=1) collapses the reduced lead to static (load-bearing). HONEST: the given/new audience-design axis only; learned conceptual pacts / lexical entrainment / partner-specificity are named follow-ons; the referent extraction + the surface string are host (a comprehension boundary + a conditioned-articulation scaffold), the grounded-vs-ungrounded DECISION is the spiking ledger read (lesion-proven)."
sources:
  - path: "doi:10.1111/j.1756-8765.2011.01152.x (Duff & Brown-Schmidt 2012, Topics in Cognitive Science — 'The hippocampus and the flexible use and processing of language')"
    anchor: "the hippocampal declarative memory system supports the formation and use of common ground in conversation; patients with hippocampal amnesia are impaired at establishing and maintaining common ground"
    note: "EXTERNAL (recorded for local addition). The load-bearing biology: common ground is not a language-module store but a DECLARATIVE/relational-memory record — amnesics with hippocampal damage FAIL to build it across a conversation while their moment-to-moment language is intact. This licenses realizing the ledger as a PERSISTENT hippocampal-style attractor record updated per grounding act, and predicts the lesion result (cut the persistence -> audience design goes static), which the de-risk reproduces (lesion -> 0.500, frac-introduce 1.00)."
  - path: "doi:10.1016/0010-0285(86)90010-7 (Clark & Wilkes-Gibbs 1986, Cognition — 'Referring as a collaborative process') / Clark & Brennan 1991 (grounding in communication)"
    anchor: "speakers and addressees collaborate to ground references, and once a referent is grounded its subsequent mentions are REDUCED (shorter, pronominalized) — the given/new contrast in referring expressions"
    note: "EXTERNAL. The functional read-out the ledger drives: audience design. A referent freshly introduced needs a full description; once it is in common ground the collaborators REDUCE it (a pronoun / a shortened form). Our reduce/introduce biased-competition decision, biased by a novelty prior and won by the ledger's grounded-bit read, is the spiking realization of this given/new choice."
  - path: "doi:10.1016/j.neuron.2013.03.007 (Wang et al. 2013 / Wimmer-Nykamp-Constantinidis-Compte line — bistable NMDA persistent-activity attractors)"
    anchor: "recurrent NMDA-dominated excitation sustains a self-sustaining bump/assembly of persistent activity that HOLDS a discrete item across a delay, and is switched ON by a transient input"
    note: "EXTERNAL. The persistence substrate: each referent's grounded bit is one self-sustaining NMDA attractor (reused from the GNW ignition machinery), latched by the grounding act's transient drive and held by recurrence — the same class of persistent-activity attractor used for working memory, here banked K-fold and independent so many referents stay simultaneously grounded. The recurrence IS the hold: setting it to 0 removes the persistence (the de-risk's lesion)."
implemented_by:
  - research/runners/_learned_common_ground_ledger_derisk.py
  - research/runners/common_ground_ledger_production_organ.py
  - webapp/common_ground_drives_chat.py
findings:
  - research/findings/raw/_learned_common_ground_ledger_6seed.json
---

# Common-ground ledger — a persistent per-referent record drives audience design

**What is measured.** A per-referent COMMON-GROUND LEDGER (K bistable NMDA-attractor stores, one per referent,
latched by grounding acts + held by recurrence) read at speak-time through a gated synapse into a biased-competition
reduce/introduce decision. Audience design (INTRODUCE a full description vs REDUCE/pronominalize) FOLLOWS the actual
grounding history: intact acc 1.000 (chance 0.5); permuting WHICH referents were grounded collapses it to 0.500;
lesioning the ledger's write-persistence (recurrence weight 0) makes it go static (always INTRODUCE, frac-introduce
1.00). The decision is a substrate read — evidence fires on grounded (0.113) but not ungrounded (0.000) targets.

## Why this is brain-based, and where the boundary sits

The referent word -> ledger-slot MAPPING is host (a language-comprehension boundary, the same one the SVO question
parser occupies) and the surface reduce/introduce STRING is a host conditioned-articulation scaffold (the discourse
"mouth"). Everything between is neurons/synapses: a grounding act IGNITES a referent's NMDA store, the recurrence
HOLDS the grounded bit across the rest of the conversation, and at speak-time the gated `led_k -> evidence` synapse
routes that persistent firing into the biased-competition decision — the grounded-vs-ungrounded verdict is never
computed in Python. The lesion proves it: silence the recurrence and the reduced reference disappears even though the
world input (a re-mentioned referent) is unchanged.

## No `constraints_config` (an empirical operating point, not a biology-required constant)

The operating point (per-referent attractor weight ~55, novelty-prior current ~220 pA, grounding/hold/query windows)
is an EMPIRICAL calibration on this substrate — the biology constrains it only as inequalities (the attractor weight
must exceed the self-sustain knee; the grounding window must reach the basin), which the equality matcher would
mis-fire on. The biology that IS pinned is structural + read-time-resolved: the grounded bit must PERSIST via
recurrence (lesion at weight 0 collapses audience design) and the read must ride real firing (grounded >> ungrounded
evidence). Those are enforced by the de-risk's own anti-cheats, not a config equality.
