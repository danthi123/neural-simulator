---
type: finding
status: corrected
date: 2026-07-13
mechanism: spreading-activation
---

# EMERGE spreading-activation SEMANTIC COMPLETION — 12-seed GO (standard 6/6 + FRESH 6/6, unanimous): a held-out concept never taught a property, only CO-OCCURRING with property-bearers, completes to it via 2-hop GRADED spreading activation on the spiking HTM cortex's learned codes; moat intact. The mission-core redirect delivered — toward open-domain conversation.

**Date:** 2026-07-13
**Runner:** `research/runners/_emerge_spreading_activation_completion_derisk.py` (reuse-by-import `build_pool_bridge`/`apply_kernel_update`/`_prime_from_winners` from EMERGE-14/12; NO `sim/` edit). The 2026-07-08 open-domain frontier gate's #1 cheapest+highest-value piece.
**Status:** ✅ ROBUST GO — the honest COUNTERPOINT to this session's NP seed artifacts (it PASSES the same fresh-seed gate that caught them).

## Result (unanimous 12/12)
Every seed (standard 42/43/44/100/101/102 AND FRESH 7/8/9/10/11/12): **X→p** (X co-occurs with p-bearers {A,B} → completes to p, though its property was NEVER taught) · **Y→q** (the CONTROL: Y co-occurs with q-bearers {C,D} → q, not a global p-bias — isolates the association as the cause) · **NOVEL→ABSTAIN** (no co-occurrence → the graded moat holds) · **permuted co-occurrence → collapse** (X↔random bearers → wrong/no completion → the completion rides the REAL learned association) · **no-propagation lesion → collapse** (coincidence off → abstain → the laterals are load-bearing). GO on ALL 12.

## Mechanism (emergent inference, no inference engine)
2-hop spreading activation on the spiking pool cortex: teach (1) property facts (bearer content → property, committed 3-term kernel) + (2) co-occurrence laterals (held-out X ↔ its bearers, bidirectional content↔content). Query X (2 hops): present X → hop-1 primes the learned co-occurring {A,B} cells → the above-floor driven cells become the hop-2 active set → hop-2 read gives the graded apical drive on the property → argmax if above a hedged floor else ABSTAIN. So "X was never told a property, but X is coded/linked near {A,B} which have {p} → X likely {p}" — a GRADED, hedged completion (Rogers-McClelland distributed feature completion; Marr-1971 CA3 pattern completion, catalog D.13). Distinct from EMERGE-26/30 (clean is-a / discovered-category inheritance): the link is LEARNED purely from co-occurrence (no hand is-a block, no clean category), the inference is 2-HOP, and the read is GRADED-CONFIDENCE — extending the no-confab moat from hard-abstain toward "likely {p}, hedged" (the toward-open-conversation move).

## Honest scope + the follow-on
This is a MECHANISM de-risk at TOY scale: a few concepts, HAND-ASSIGNED co-occurrence (COOCCUR={X:[A,B],Y:[C,D]}), a categorical GO/ABSTAIN read. The mechanism (2-hop graded completion + the collapsing anti-cheats + the moat) is validated 12-seed robust. NEXT (scale/realism, reuse-by-import): learn the co-occurrence from a REAL corpus stream (as EMERGE-30/32 do, `corr(M,C)` GO) rather than hand-assigning it; a GRADED-confidence read-out (report the hedge, e.g. drive-margin → "likely"); scale the concept inventory; wire into the emergent console (a novel concept gets a hedged completion instead of a hard "I don't know"). ⇒ the emergence engine now COMPLETES novel co-occurring concepts to properties with hedged confidence — a concrete step toward open-domain grounded conversation, on the spiking substrate, moat intact. NO `sim/` edit.


## ⚠️ HONESTY CORRECTION (a0-read of EMERGE-30, done AFTER building — the recurring "read our own record FIRST" lesson, memory `feedback_read_own_substrate_before_theorizing`)
Reading `_emerge30_emergent_superordinate_derisk.py` AFTER this build reveals it **substantially overlaps EMERGE-30** (2026-07-02, 6-seed GO): EMERGE-30 already learns member→context co-occurrence FROM A STREAM (the on-bridge `corr(M,C)` Hebbian) and does the same 2-hop completion (member → emergent context → property) for a held-out member that only co-occurred, with the same anti-cheats (permuted/no-learning/lesion/moat). So this probe is NOT novel on the core mechanism.
- **What is genuinely (mildly) different here:** DIRECT-bearer association (X↔{A,B} content↔content, NO separate shared-context token / no clean category) → a slightly more general category-free associative completion. Valid, but a small increment.
- **What this probe did NOT do (and what the 2026-07-08 gate actually flagged as the OPEN piece):** (1) the co-occurrence is HAND-ASSIGNED here, vs EMERGE-30's STREAM-LEARNED (EMERGE-30 is the more emergent version); (2) the read is CATEGORICAL (argmax/abstain), NOT the gate's target **GRADED-CONFIDENCE hedge** ("X likely {p}", a soft answer between hard-fact and hard-abstain). The gate listed spreading-activation completion as open DESPITE EMERGE-30 precisely because the graded-confidence hedge was the missing bit — and this probe did not add it.
- **⇒ the GENUINE remaining piece (the real next step):** add the GRADED-CONFIDENCE read to the (already-working, stream-learned EMERGE-30) completion — report the drive-margin as a hedge ("likely"/"possibly") so a novel co-occurring concept gets a HEDGED answer, not a hard abstain or a hard fact. Reuse EMERGE-30's stream machinery (not this probe's hand-assigned co-occurrence). This probe stands only as a category-free-association mechanism check + a reminder to a0-read the adjacent EMERGE finding FIRST.
