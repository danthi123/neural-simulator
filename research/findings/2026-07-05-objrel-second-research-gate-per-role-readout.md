# Second research gate — the objrel read boundary is the READ-OUT ARCHITECTURE (per-role loci + ridge committee), not another read trick

**Date:** 2026-07-05
**Type:** read-only deep-research survey (catalog + Kandel 6e + literature + bio-research MCP). No code touched.
**Launched by:** the exhausted first-gate ladder — subtraction (see-saw), division (see-saw), 3 learned-signed reads (position-basin), first-to-fire (dt-blocked) — plus the c2 read-out seed-fragility (base canon ≤0.03 on 3/6).
**Launches:** the per-role read-out de-risk (RANK 1 below).

## Verdict — SURPASSABLE; the target is the read-out ARCHITECTURE

All four prior failures share ONE flaw: they kept a **single competitive 3-way winner-take-all (WTA) doing BOTH canonical
(position) AND object-relative (form), read as a rate-max off a sub-1% margin.** That is why they all produced the same
anti-correlated SEE-SAW (lifting objrel regressed canonical) and never touched the seed-fragility. The residual is
**(b) representational competition inside a shared read-out ∧ (c) a non-generalizing decoder**; (a) the WTA's inability to
resolve the linearly-separable sub-1% differential is the visible SYMPTOM of reading a rate-max off that shared, fragile
read-out (a rate-WTA's sensitivity grows only LOGARITHMICALLY with N — Chen-Miller temporal-WTA — so no common-mode trick
on it can reach the margin).

## The biological reframe (triply-grounded — this is how biology does robust non-local role assignment)

1. **Per-role read-out LOCI, not one competition** (Frankland-Greene, lmSTC: a region selective for AGENTS, a separate one
   for PATIENTS; "dog chased cat" vs "cat chased cat" discriminated by WHICH locus holds which filler, across active +
   passive). Binding = which filler drives which role-locus. Split the read-out per role → canonical and objrel stop
   fighting over one competition → the see-saw dissolves at its source.
2. **Dual-route / Competition Model** (Bates-MacWhinney): thematic assignment = multiple cues (word-order, case, animacy,
   agreement) weighted by cue-validity = availability × reliability, combined competitively. Object-relative = the POSITION
   cue is misleading (slot-0 = theme) and a FORM cue (the "that" marker, the 2nd NP, verb-argument structure) must
   OVERRIDE it. A position route AND a form route that COMBINE — not one read-out. (On the project's own standing
   directive `project_conversational_primary_robust_multicue_parser`.)
3. **The PARENT model already does this** (Hinaut-Dominey 2013, PLoS ONE 8:e52946 — the SAME reservoir class): its read-out
   neurons code thematic roles as SEPARATE output units, form-driven, RIDGE-trained, and it generalizes to unlearned
   constructions. **The project's single shared 3-way WTA is the DEVIATION that created this boundary.**
4. **Robust decoders generalize via ridge-regularization + committee/ensemble voting** (the RC-standard closed-form read-out
   the failed delta-rule-from-scratch omitted; MindEye2 shared latent; Ensembles-of-SNNs). Fixes the seed-fragility (c).

## Ranked cheap-first ladder (all: reuse-by-import, no `sim/` edit expected; 6-seed-BLIND dev 42/43/44 → 100/101/102; 4 anti-cheats)

**RANK 1 — PER-ROLE read-out populations + RIDGE-regularized + COMMITTEE-voted decoder. BUILD FIRST.** Replace the single
3-way WTA with N_roles INDEPENDENT read-out pools (agent, theme, [recipient]), each asking "does MY filler bind here?".
Train reservoir→role weights by RIDGE regression (the RC standard the failed learners omitted); COMMITTEE the decode
(K read-outs on K reservoir sub-samples / bootstrap resamples → plurality vote per role). Fixes BOTH residuals at once:
the see-saw (per-role independence — no shared competition to push in opposite directions) AND seed-fragility (ridge +
ensemble). Single-variable: shared-WTA → per-role ridge-committee, else fixed. Anti-cheats: lesion one role locus → only
that role collapses (per-role separability + load-bearing); scramble → chance; canonical-not-regressed; **objrel-recovers on
the BLIND seeds = the acceptance bar.** Cheapest; the project already has multi-pool WTA regions + ridge read-out utilities.

**RANK 2 — per-role OPPONENT ACCUMULATOR pair** (signed difference integrated over the sentence to a bound — drift-diffusion
/ LIP, catalog G.16/G.17; √T×√N gain, post-spike, no shared pedestal). ADD only if Rank 1 leaves an objrel margin residual
on blind seeds. Reuses the Wang-2002/Lo-Wang accumulator machinery.

**RANK 3 — explicit DUAL-ROUTE cue-competition** (position route + form/construction route, combined by learned cue-validity
weights from the corpus statistics the project already mines). For BROADENING past objrel to passive/cleft/wh; later
composes with the existing `transmission_gate` for construction routing.

**RANK 4 (fallback) — re-express the working LINEAR ARGMAX as per-role LEARNED THRESHOLDS** (NEF-cleanup, the project's own
Stewart-Tang-Eliasmith mechanism) — skip a second reservoir (the linear argmax already solves it → nonlinearity is not the
missing piece).

**NOT recommended:** any trick on the single shared WTA (ruled out by the see-saw); a second reservoir / Deep-ESN as a first
move (over-engineering — the linear argmax already works); pure more-population/replay on the rate-WTA (logarithmic scaling).

## Key sources
Catalog G.11/G.12 (dual-stream + Broca, Kandel 6e Ch 55 pp 1380-1387 — the exact objrel-fails validation), G.16/G.17 (LIP
accumulator, Ch 56 pp 1399-1404), E.03 (population coding, Ch 17). Hinaut-Dominey 2013 (PLoS ONE 8:e52946 — per-role
ridge-trained reservoir read-out). Bates-MacWhinney Competition Model. Frankland-Greene (arXiv 2110.12342 — agent/patient
loci). Grodzinsky TDH (object>subject relative deficit). Chen-Miller temporal-WTA (PMC2633619 — rate-WTA logarithmic
scaling). Cross-subject decoding (committee + ridge + shared geometry).
