# Load-bearing terms — one term, one meaning, one code condition

**Why this file exists.** On 2026-07-28 three of nine retractions in a single session were pure terminology
overclaim, not measurement error. An experiment was called **consolidation** when the replay code never executed,
**compositional** when the representation was localist, and **self-organized** while the host was supplying both
factors of the learning rule. Every measurement in those runs was correct and reproducible. The *words* were wrong,
and each one did real damage because it carried a meaning the code did not implement.

This is the ASD-STE100 discipline (one term, one meaning) applied to the ~10 words in this project that actually
carry load. It is deliberately small. It is not a style guide.

**The rule:** before using a term below in a commit message, findings doc, or board entry, check its CODE
CONDITION. If the condition does not hold, use the fallback wording instead. A term whose condition you have not
checked is a HYPOTHESIS, exactly like a claim in a comment.

---

| Term | May be used ONLY IF | Fallback when it does not hold |
|---|---|---|
| **consolidation** | a replay/reactivation path **actually executes** (verify the call is reached, not merely present — check the branch it sits in), AND the trace survives a lesion of the source structure | "a cortical write", "a store", "an association" |
| **compositional** | the item's representation is **constructed from its constituents**, and the design can DISTINGUISH that from a per-item code (a disjoint-constituent fact set cannot — either constituent alone identifies the item) | "localist", "one-unit-per-item", "an address→value map" |
| **self-organized** | **no host code supplies the answer** — check both factors of the learning rule (pre AND post), the target selection, and the allocation of any slot/unit | "host-supervised", "teacher-driven", "host-selected target" |
| **closed** (a capability) | the end-to-end capability gate passes with its anti-cheat controls at 6 seeds, AND the shipped default path uses it | "de-risked", "validated opt-in", "GO on a proxy" |
| **GO** | the gate's OWN verdict is positive — never a metric lifted out of a run whose verdict was negative | "partial", "LEAD", "indicative" |
| **fully spiking** | every cognitive step between sensation and action is neurons/synapses; host code only for world + body | "spiking with a host read-out", "host-argmax read-out" |
| **byte-identical** | asserted **in the data** (hash or exact compare), never inferred from reading the code | "expected unchanged (unverified)" |
| **lesion** | the manipulation is verified to STILL HOLD at the moment of measurement (a zeroed weight can regrow within steps if plasticity is live) | "attempted lesion (persistence unverified)" |
| **selective** | reported with its permuted/scrambled control AND raw per-item magnitudes; a ratio alone is not selectivity | "ratio X (control not run)" |
| **works / solved** | the capability gate passes, not a proxy of it (a weight read is a proxy; behaviour is the capability) | "the proxy passes", "the write works" |

---

## Notes

- **This does not replace verification.** A correctly-worded claim can still rest on a broken instrument — the
  three terminology retractions above sat alongside six *instrument* failures that no vocabulary would catch. Use
  `.claude/skills/verify-go/SKILL.md` for those.
- **Scope.** Findings docs, commit messages, `GAP_CLOSURE_MISSION.md`, `ROADMAP.md`, `CLAUDE.md`. Not enforced on
  code comments or scratch notes.
- **Adding a term.** Only when a real overclaim has occurred — this file records earned lessons, not anticipated
  ones. Cite the incident.
- **On full ASD-STE100** (Issue 9, Jan 2025; ~53 rules + ~900-word dictionary, free-on-request from STEMG, not
  redistributable): the full specification was assessed and NOT adopted for findings. Its rules target procedural
  maintenance prose, whereas findings are argumentative and causal; its controlled dictionary does not cover
  computational neuroscience. The one-term-one-meaning principle above is the part that maps to this project's
  actual failure mode. It remains a reasonable candidate for public-facing docs (README/CONTRIBUTING).
