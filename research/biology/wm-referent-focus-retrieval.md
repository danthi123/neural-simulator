---
type: biology
id: wm-referent-focus-retrieval
mechanism: A working-memory HOLD-and-RETRIEVE mechanism for the D6 multi-referent organ. HOLD: the organ's
  per-referent register state (membrane, recovery, conductances including the slow-NMDA recurrence, firing and
  refractory arrays) persists across turns rather than being re-read from a host dict, so which referent is
  "in mind" is carried by which register's assembly keeps firing (a spiking analogue of prefrontal delay-period
  persistent activity maintaining working-memory content across a delay). RETRIEVE: a later cue (an anaphor) does
  not read a fixed positional slot; each held register's rate drives its own assembly in an N-way lateral-inhibition
  winner-take-all competition (the SAME competitive-selection primitive already established for the affect-marker
  and question-route circuits in this codebase), and the assembly that wins the race names which register --
  and therefore which referent -- the cue retrieves.
status: de-risking
last_verified: 2026-09-24
current_finding: research/findings/2026-09-24-wm-referent-focus-bind-anaphor-probe-PREREGISTRATION.md
current_status: >
  PRE-REGISTRATION filed 2026-09-24 (`LB_WMB_FOCUS_PROBE`, mechanism flag `BRAIN_MULTIREF_FOCUS_BIND`, default
  OFF). Flags-off byte-identity is asserted in data (`research/findings/raw/_wm_focus_bind/offflag_byte_identity_s7.json`,
  6/6 sha256-identical turns against the pinned pre-change SHA). Dev-seed-7 calibration of the retrieval operating
  point (gain/cue-current/race-duration) is committed
  (`research/findings/raw/_wm_focus_bind/calib_dev_seeds.json`): at the shipped operating point a live 2-referent
  buffer resolved 8/8 across 8 dev seeds, with the same register winning under both mention orders on 8/8 and both
  referents still held after retrieval on 8/8; an empty or lesioned (recur=0) buffer resolved 0/8. No evaluation
  seed (42/43/44/100/101/102) has been built and no GO/NO-GO is claimed; this entry records the biology the
  mechanism is grounded in, ahead of that run.
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "during the delay period after the first stimulus ends."
    note: >
      EXTERNAL. Kandel PNS-6e (somatosensory working-memory chapter): "Neurons in these frontal cortical areas
      continue to fire during the delay period after the first stimulus ends" -- the direct precedent for HOLD:
      a memorandum is maintained across a temporal gap by CONTINUED FIRING of the neurons that first represented
      it, not by a re-read of stored symbolic state. This mechanism's per-referent register keeping its assembly's
      membrane/firing state alive across a turn boundary (rather than reloading from the host `_slot_of_ref` dict)
      is the same structure: the memorandum's identity is carried by WHICH population keeps firing, not by a
      lookup.
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "during the delay period of a delayed match-to-sample"
    note: >
      EXTERNAL. Kandel PNS-6e, on prefrontal delay-period activity in a delayed match-to-sample task: "the
      continued firing of many prefrontal neurons during the delay period of a delayed match-to-sample task
      initially represents information about the sample image." Grounds that persistent delay-period firing
      carries SPECIFIC held content (which sample was shown), not merely a generic "something is held" signal --
      the precedent for this mechanism's claim that "what is held is read off that [persisting] state," i.e. the
      live register's identity, not just its liveness, is what a later retrieval reads.
  - path: research/biology/affective-marker-lateral-inhibition-wta.md
    anchor: "mutual/reciprocal lateral inhibition"
    note: >
      LOCAL. The exact N-way competitive-selection primitive this mechanism's retrieval step reuses (via
      `_affect_marker_wta_derisk._build_bridge`, imported not reimplemented, at N=5 register-assemblies instead
      of N=6 affect registers or N=4 route candidates): each competing assembly recruits its own fast-spiking
      sub-pool that cross-inhibits every OTHER assembly, and the assembly whose rate wins the settled race names
      the decision. `question-route-selection-wta.md` already generalizes this SAME primitive once (route
      selection); this entry is a second reuse (which held register a cue retrieves), not a new architecture.
  - path: "doi:10.1207/s15516709cog0000_25 (Lewis & Vasishth 2005, Cognitive Science 29:375 -- 'An activation-based
      model of sentence processing as skilled memory retrieval')"
    anchor: "cue-based retrieval"
    note: >
      EXTERNAL, not locally anchored (no local full text; DOI only, as the prereg document itself already cites
      this work under the SAME caveat). Names the residual this mechanism does NOT yet implement: retrieval here
      is decided by the pools' intrinsic excitability alone, with no encoded discourse-salience cues (subjecthood,
      recency, topic) biasing the race the way cue-based sentence-processing models predict interference between
      similarly-cued memoranda. The prereg's "Declared residual shortcuts" section names this as the next rung
      (a Centering-style salience preference), not a claim already implemented.
implemented_by:
  - research/runners/d6_multiref_wm_production_organ.py
  - research/runners/brain_chat_tui.py
  - webapp/server.py
  - webapp/gnw_bus_shadow.py
  - webapp/gnw_two_organ_bus.py
  - webapp/gnw_three_organ_bus.py
findings:
  - research/findings/2026-09-24-wm-binding-ordinary-content-probe-6seed-NOGO-held-state-does-not-reach-an-ordinary-reply.md
  - research/findings/2026-09-24-wm-referent-focus-bind-anaphor-probe-PREREGISTRATION.md
---

# A referent's hold is persistent delay-period-style firing; its retrieval is a lateral-inhibition race

**What this binds.** `research/findings/2026-09-24-wm-referent-focus-bind-anaphor-probe-PREREGISTRATION.md`
pre-registers `BRAIN_MULTIREF_FOCUS_BIND` (default OFF): the D6 multi-referent working-memory organ's held state
persists across turns as live per-register firing rather than a host dict re-read, and a later anaphor retrieves
one held register by an N-way lateral-inhibition competition instead of a fixed positional read
(`CAND_POOLS[0]`). This entry records the two pieces of biology the mechanism follows, ahead of any evaluation
seed being run (the doc's own verdict field: "PRE-REGISTRATION only. No evaluation seed has been built.").

**HOLD is grounded in persistent delay-period firing.** Kandel PNS-6e documents frontal-cortical neurons that
"continue to fire during the delay period after the first stimulus ends" in order to "preserve a memory of that
response," and that during a delayed match-to-sample task "the continued firing of many prefrontal neurons ...
initially represents information about the sample image." The organ's per-referent register keeping its own
membrane/recovery/conductance/firing state alive across a turn boundary -- so that "a live bump keeps firing; a
dead one stays dead," and "what is held is read off that state" -- is the same structure at the level this
codebase can build it: identity of the memorandum is carried by WHICH assembly's activity persists, not by a
host-side symbolic re-load.

**RETRIEVE is grounded in the project's own established competitive-selection primitive.** Rather than importing
a new competition architecture, the anaphor-resolution step reuses the SAME mutual/reciprocal lateral-inhibition
winner-take-all circuit already banked for affect-marker selection
(`affective-marker-lateral-inhibition-wta.md`) and generalized once already for comprehension-route selection
(`question-route-selection-wta.md`): each candidate (here, a held register) drives its own excitatory assembly
with its own fast-spiking cross-inhibition sub-pool, the assemblies race under a common cue drive, and the
winner's identity is read off settled firing rates, never a host `argmax` over pre-computed scores.

**Declared residual, not claimed as implemented.** The organ's retrieval race is decided by intrinsic
excitability alone; it carries no discourse-salience signal (subjecthood, recency, topic) of the kind cue-based
sentence-processing models (Lewis & Vasishth 2005) predict biases real anaphor resolution under interference.
The prereg names this as the next rung, not a gap this entry papers over.

**Honesty boundary.** Both Kandel anchors describe primate PFC/frontal delay-period activity during instructed
tasks, not the specific referent-tracking-across-a-dialogue-turn paradigm this organ implements; the citation
grounds the PRINCIPLE (persistent firing maintains a memorandum's identity across a temporal gap, and later
processing reads that persisting state) that the organ's turn-to-turn hold structurally follows, not a claim that
this probe replicates a delayed-match-to-sample experiment.
