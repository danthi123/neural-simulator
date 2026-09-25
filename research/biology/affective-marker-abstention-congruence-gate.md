---
type: biology
id: affective-marker-abstention-congruence-gate
mechanism: A conflict-monitoring GATE on the affective EXPRESSION marker (`webapp/affect_drives_chat.expression_lead`):
  before the felt-state marker is allowed to color a reply, it is checked against two INDEPENDENTLY-computed
  brain reads already available on the SAME turn -- the moat/BG speak-vs-abstain decision (`resp["abstained"]`)
  and the co-resident Gate-B spiking affect organ's own valence sign (`resp["affect"]["valence_sign"]`, from
  `affect_production_organ.appraise_text` + `read_differential`). When the marker's register disagrees with
  either read (a warm/positive marker glued onto an abstained turn, or a positive-register marker when Gate-B's
  independent read is negative, or vice-versa), the marker is WITHHELD before it is ever prepended to the answer
  surface -- an honest no-lead turn, not a silently-edited one. Both inputs to the conflict check are neural
  reads; only the comparison itself (a fixed word->sign lookup + a boolean branch) is host control flow, of the
  same kind every other Gate-B-driven surface coupling in `webapp/server.py` already uses (see "Declared host
  step" below).
status: de-risking
last_verified: 2026-09-25
current_finding: research/findings/2026-09-24-affect-marker-settle-flip-criteria-AMENDMENT-PREREG.md
current_status: >
  A2 (abstention-congruence) amendment-2 (2026-09-25) commits the production wiring this entry documents:
  `BRAIN_AFFECT_MARKER_CONGRUENCE` (default OFF) gates `expression_lead`'s output at both production call sites
  (`webapp/server.py`'s rich and single-fact paths) via `webapp/affect_drives_chat.congruence_gate`, which reuses
  the register/sign logic already de-risked in `research/runners/_affect_marker_settle_congruence.py` (2026-09-24)
  rather than re-deriving it. Byte-identical-off is asserted by `tests/test_affect_marker_congruence_gate.py`
  (unit-level, hash-free exact-compare) ahead of any full-brain run; no seed has been scored under this flag yet.
sources:
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "the ACC is involved in cognitive processes involved"
    note: >
      EXTERNAL. Kandel PNS-6e (Ch. 61, Disorders of Mood and Anxiety): "The caudal subdivision of the ACC is
      involved in cognitive processes involved in control of behavior; it has connections with dorsal regions of
      the prefrontal cortex, secondary motor cortex, and posterior cingulate cortex." Names the caudal ACC /
      dorsolateral-PFC "cognitive control network" as the circuit-level precedent for arbitrating a candidate
      expressive/behavioral output against a competing cognitive-appraisal signal, rather than letting the
      affective output surface unchecked -- the general principle this gate implements at the scale this codebase
      can build (a boolean congruence check), not a claim of replicating ACC microcircuitry.
  - path: ~/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "the integration of emotion, cognition, and autonomic"
    note: >
      EXTERNAL, same chapter/page: "[the rostral and ventral ACC] ... are involved in the integration of emotion,
      cognition, and autonomic nervous system function." Grounds the premise that emotion and cognition are
      NORMALLY integrated (reconciled against each other) rather than independent parallel outputs -- i.e. that a
      conflict between an affective read and a cognitive/decision read is exactly the kind of signal a real brain
      circuit resolves, not a defect specific to this codebase's two-organ (Gate-B affect / BG-moat abstain)
      split.
  - path: research/biology/interoceptive-affect.md
    anchor: "interoceptive"
    note: >
      LOCAL. The upstream felt-state read this gate's marker input rides on (`GradedAffectBrain` / `read_body`,
      board #81) is already biology-bound there; this entry does not re-derive it, only the NEW conflict-check
      step added downstream of it.
  - path: research/biology/affective-marker-lateral-inhibition-wta.md
    anchor: "lateral inhibition"
    note: >
      LOCAL. The marker's register/word IDENTITY (what this gate checks for a sign conflict) is itself selected by
      the spiking lateral-inhibition WTA this entry already binds; this gate is a second, independent check
      applied to that circuit's OUTPUT, not a replacement for it.
implemented_by:
  - webapp/affect_drives_chat.py
  - webapp/server.py
  - research/runners/_affect_marker_settle_congruence.py
findings:
  - research/findings/2026-09-24-affect-marker-settle-flip-criteria-AMENDMENT-PREREG.md
---

# The affect marker is withheld, not edited, when it conflicts with an independent brain read

**What this binds.** The A2 flip precondition (abstention-congruence) in
`research/findings/2026-09-24-affect-marker-settle-flip-criteria-AMENDMENT-PREREG.md` named the prior mechanism
(`research/runners/_affect_marker_settle_congruence.apply_policy`) an explicit host shortcut (S6): it edits an
ALREADY-COMPOSED reply string after the fact, is not reachable from the production chat path at all, and a
prefix-only strip left the marker in the reply on real incongruent turns whenever a later stage prepended its own
text. That amendment named the production design this entry documents as the fix: "gating the marker with the
same spiking speak/abstain race that decides abstention" -- i.e., check BEFORE the marker is ever surfaced,
using reads the brain already computed this turn, not a post-hoc string edit.

**The two inputs are both neural reads, computed earlier in the SAME turn.** `resp["abstained"]` is the
moat/BG speak-vs-abstain decision (the no-confab gate) that already runs first and unchanged for every turn.
`resp["affect"]["valence_sign"]` is the Gate-B affect organ's own spiking differential read
(`affect_production_organ.read_differential`), independent of the #81 graded-affect ladder that selects the
marker's word. Neither is computed by this gate; the gate only reads them.

**The biology.** Real conflict between an affective output and a cognitive/behavioral-control signal is resolved
by dedicated circuitry, not left to surface unchecked: Kandel PNS-6e names the caudal ACC / DLPFC "cognitive
control network" as distinct from, but interconnected with, the ACC/insula "emotional salience network" that
computes affect, and states plainly that the rostral/ventral ACC's job is "the integration of emotion, cognition,
and autonomic nervous system function" (Ch. 61). This gate is that integration step at the scale this codebase
can build: a conflict between two already-computed reads (affect-organ valence, moat-organ abstain) is checked
before either is allowed to shape the reply.

**Declared host step (not claimed as neural).** The marker WORD is translated to a fixed sign (`Wonderful` /
`Gladly` / `Sure` -> +1, `Hm` / `Honestly` / `Frankly` -> -1) by a host lookup table -- the SAME table
`webapp/affect_drives_chat._LEAD_WORD` already inverts to select the word, so no new host arithmetic is added --
and the decision to withhold vs. surface the marker is a host `if` branch over two booleans/signs. This is the
identical pattern already used for every other Gate-B-driven surface coupling in `webapp/server.py` (metacog
hedge on `abstained`, curiosity follow-up on `abstained`, surprise/reconsolidation prefixes on their own spiking
reads): the READ is neural, the branch that gates a string operation on a neural boolean is host, and this
project's existing convention treats that branch as legitimate host glue, not a scored shortcut. What would still
be a shortcut, and is NOT done here, is computing either input (abstain or valence sign) from anything other than
its existing spiking organ.

**What this gate deliberately does NOT do.** It does not build a new spiking projection from the abstain/moat
circuit onto the affect-marker WTA's own assemblies (a true single-circuit "speak/abstain race" gating the
marker's spiking selection itself, as the amendment's phrasing evokes literally). That would let an abstain
SUPPRESS the marker WTA's read before it ever fires, rather than checking its output afterward. This entry
records that as the next rung, not a gap papered over: the current gate is a correct, minimal, honestly-declared
host arbitration over two already-neural reads; a same-circuit spiking veto is a larger, separate build.
