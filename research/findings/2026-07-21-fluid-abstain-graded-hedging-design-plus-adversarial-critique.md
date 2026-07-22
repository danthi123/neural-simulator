# Fluid graded-hedging (replace the hard binary moat) — DESIGN + an adversarial critique that found a STRUCTURAL flaw: the cheap N-threshold ladder is binary-in-disguise on the current KB; the real graded signal is the cleanup-score S (calibrate) or the trained-LM confidence (scale). GPU-free research-gate Workflow.

**2026-07-21.** Owner priority #3 ("do away with the hard abstain moat in favor of more fluid and natural abstaining
without outright refusing"). A GPU-free research+design+adversarial-critique Workflow (5 agents, run in parallel with the
LM training + gap#5). The critique prevented building the wrong thing — the value of the adversarial pass.

## The DESIGN (sound architecture, per Mielke-Boureau TACL-2022: internal signal → discrete band → controllable hedge)
Replace the single binary moat (`_fluidconv_chat_repl.py::FluidChat._answer:333-347`, `if p is None: "I don't know"`)
with a 4-band ladder over the SAME gate-first seam, driven by confidence the composer ALREADY computes:
- **L0 assertive / L1 "I think" / L2 "I'm not certain, but…" / L3 graceful soft-abstain (surface the candidate p) / MOAT (unchanged hard "I don't know" for genuinely-unknown = `what_does`→None).**
- **Two axes, combined ASYMMETRICALLY:** (1) GROUNDED-fact confidence sets the band (the calibrated Bogacz-Brown
  familiarity novelty N + the RF cleanup match-score S); (2) OPEN-generation confidence (length-normalized token
  log-prob) is DEMOTION-ONLY render-quality (never promotes) — because an 83M model's own verbalized/entropy confidence
  is saturated/orthogonal-to-accuracy (lit: "Wired for Overconfidence", the 3-9B psychometric screen). **The model must
  never self-report; it only renders hedge words conditioned on the gate-chosen band.**
- **Render:** rung-A template prefix (NO fine-tune, on the deployed console NOW); rung-B a learned `<hedgeK>` marker via
  the EMERGE-57 confidence-conditioned fine-tune on the WKV checkpoint (no base retrain). All in `research/runners/`, NO
  `sim/` edit, behind a default-off `enable_hedging` flag (byte-identical default).
- The literature framing (load-bearing): the entire external UQ field spends 5-10× compute (semantic entropy, sampling)
  to RECOVER a graded internal scalar the composer emits natively for free — so multi-band thresholding of that scalar is
  the cheap best-fit, NOT a bolted-on estimator.

## The ADVERSARIAL CRITIQUE — a STRUCTURAL flaw in the cheap path (verified against the code, decisive)
**The primary signal N is BIMODAL, so the graded middle does not exist on the current KB.** `N = ||x||² − xᵀWx` over the
Bogacz-Brown projector: familiar cue → N≈0, novel → N≈0.99. **The cited "+0.98 margin / safe window 0.44-0.52" is the
EMPTY GAP between the two modes — proof there is NO population there.** The console KB (`familiarity_gate_v320_validation.py:148-155`)
FORBIDS duplicate `(agent,action)` (`seen_aa`) → zero relational ambiguity → the graded-middle source (competition between
two learned answers, as in the cited `_emerge_graded_confidence_console_derisk.py` bat=bird∨mammal template) DOES NOT
EXIST here. Consequences (all verified in-repo):
1. **The 4 bands degenerate to L0-or-MOAT — a re-skinned binary.** Every stored fact has N≈0 → lands L0 assertive; L1/L2
   never fire; forcing thresholds into the empty gap makes hedging UNCORRELATED with correctness (random "I think" =
   Yona-2024's exact failure). Bimodal N gives decorative-OR-random, never graded.
2. **L3 graceful soft-abstain is near-vacuous** — it needs "matched candidate p AND high N", but a `what_does` match
   imprints N≈0; "matched and novel" is a near-contradiction on this path → genuine unknowns still hit the hard "I don't
   know" (RENAME, not replace).
3. **N is blind to the one real confident-wrong case** — the composer's false relational match (`_scan_first_match:435-444`)
   reads the SAME `{agent,action}` composite whose coincidental in-span-ness CAUSES the false match; N reads that closeness
   as LOW novelty → L0 confident. N is independent of the *patient* equality but perfectly correlated with the *cue*
   failure that produces the confident-wrong. So the design's headline safety claim ("N catches familiar-but-wrong") is
   FALSE (the hard moat, retained, still covers genuine unknowns — safety isn't regressed, but the claim is).

## The HONEST PATH (the critique inverts the load-bearing decision)
Graded hedging is NOT a cheap N-threshold on this KB. The genuine graded signal must come from one of:
- **(A) The RF cleanup match-score S** (the phase-cos the argmax already computes, `rf_phasor_composer.py:_cleanup_all_scored`)
  — it IS continuous. NEXT de-risk: does S have a CALIBRATED graded structure that separates correct-vs-wrong grounded
  answers (an S safe-window study, like N's 0.44-0.52)? If yes, S (not N) is the primary band splitter. (S currently has
  no calibrated study — the gap to close.)
- **(B) Introduce relational ambiguity** (multi-referent / competing facts) → then a competition-margin has a real graded
  middle (the bat=bird∨mammal structure). This is the mechanism the cited GO template actually used.
- **(C) OPEN-domain hedging ties to MODEL SCALE** — the honest open-conversation hedge signal is the TRAINED LM's own
  confidence, which is usable only once the model is large/well-trained enough (at 83M, token entropy is orthogonal to
  accuracy per the lit). ⇒ open-domain fluid-abstain COLLAPSES INTO the scale axis (#1), like #2/#3.

## Verdict
- **Grounded single-fact graded hedging is achievable but NOT via the bimodal N** — it needs an S-calibration study (A)
  or introduced ambiguity (B). The architecture (bands over the gate-first seam, demotion-only open-gen axis, moat
  retained, EMERGE-57 render fine-tune) is SOUND and reusable once the signal is fixed.
- **Open-domain fluid abstain is scale-bound** (C) — it improves with the trained LM, tracking the same training axis.
- **The hard moat is RETAINED as belt-and-suspenders** on genuinely-unknown cues (no-confab never regresses); the honest
  target is "graded much-better-than-binary," not perfect (perfect calibration = the field's open hallucination problem).
- NEXT (when prioritized): the S-calibration de-risk (A) — does the cleanup match-score separate correct-vs-wrong on
  held-out grounded facts? — is the cheap decisive test for grounded graded hedging.
- Full design/critique: workflow `wf_95e2b35f-2de` journal. NO code built (the critique showed the cheap rung is
  decorative — do NOT build it); the design is banked with its structural caveat.
