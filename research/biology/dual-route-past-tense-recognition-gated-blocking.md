---
type: biology
id: dual-route-past-tense-recognition-gated-blocking
mechanism: >
  Dual-route past-tense inflection (Pinker-Ullman words-and-rules): a REGULAR rule ("-ed", the DEFAULT /
  elsewhere condition, applies to any stem including never-seen pseudo-stems) and a DECLARATIVE store of
  irregular whole-forms (go->went) that BLOCKS the rule ONLY when a stored form is genuinely retrieved. The
  load-bearing companion process is the RECOGNITION THRESHOLD on the blocking pathway: the exception overrides
  the default only on supra-threshold (familiar) retrieval, so an unfamiliar stem leaves the default unopposed.
status: established
last_verified: 2026-08-26
current_finding: research/findings/2026-08-01-E-language-dual-route-morphology-declarative-route-works-procedural-rule-does-not-generalize-6seed-NOGO.md
current_status: >
  Two-pool structural separation realizes the declarative route 6/6 but the RULE generalization to novel stems
  is SEED-FRAGILE (reg_acc 0.25-1.0). Diagnosed 2026-08-26 to a stem-INDEPENDENT went/ran tonic FLOOR (~0.20
  cosine) in the LEX readout. The recognition-THRESHOLD-on-the-interneuron lever was BUILT and REFUTED (1-seed
  numpy smoke seed 43, then CONFIRMED as a 6-seed NO-GO on seeds 42-47: gate_selective 0/6, reg_acc 0.875 and
  irr_acc 0.00 on every seed; see the 2026-08-26 finding below): the pooled blocking interneuron fires ~equally
  for irregular vs novel cues (~1.0x on every seed) at EVERY inh_drive 0.3-6.0, because it sums over all 7
  whole-forms and the tonic floor keeps some whole-form active for every cue -- so there is no genuine-vs-
  spurious rate to threshold on; raising the threshold only breaks blocking (irr 0.857->0.00). The residual is
  UPSTREAM: the tonic floor must be removed AT SOURCE. Next mechanism (not yet built): spike-frequency
  ADAPTATION (AHP/M-current) to quench persistent attractors so entrenched whole-forms fire only transiently on
  their own cue.
sources:
  - path: /home/dant123/Projects/sim-catalog/references/textbooks/kandel-pns-6e/full-book.txt
    anchor: "becomes went rather than goed and break becomes broke"
    note: >
      Kandel 6e Ch 55 (Language), p.1373. Grounds the dual pattern the code must reproduce: regular verbs take a
      rule-governed ending ("play -> played"), irregulars are stored exceptions ("go -> went, not goed"), and the
      child's error "goed"/"breaked" is OVER-REGULARIZATION = the default rule applied when the stored exception
      is not retrieved (the lesion signature this runner reproduces).
constants:
  whole_form_floor_cosine: 0.20
  affix_default_cosine: 0.24
# THEORY (external, not in the local corpus so recorded in prose, not as a resolvable source): Pinker & Ullman
# 2002 (Trends Cogn Sci 6(11):456) words-and-rules -- regulars = a procedural DEFAULT rule, irregulars =
# associative declarative memory, the rule is the ELSEWHERE case that applies whenever lexical lookup fails;
# Marcus et al. 1992 = the blocking + U-shaped over-regularization data.
# ARCHITECTURE the runner uses (a build arg, not a cfg key, so stated here not in constraints_config): blocking
# is a Dale-compliant di-synaptic interneuron (whole-form(exc) -> interneuron(inh) -> affix), build_two_pool(...,
# di_synaptic=True). The tried-and-refuted lever was a spike-threshold on that interneuron.
companion_processes:
  - process: >
      recognition / familiarity threshold on the exception-retrieval-to-blocking pathway (perirhinal/MTL
      recognition + the interneuron's own spike threshold): the declarative override fires ONLY when a stored
      whole-form is retrieved above criterion. A novel stem is unfamiliar -> no supra-threshold retrieval -> no
      blocking -> the default rule proceeds.
    status: proxied_lever_refuted
    proxied_by: >
      inh_drive as a LINEAR whole-form->interneuron weight with no effective threshold: the ~0.20 spurious
      whole-form floor drove the interneuron enough to suppress the default affix for novel stems. The
      threshold-on-the-interneuron FIX was tried and REFUTED (the interneuron pools over all whole-forms, so the
      floor keeps it firing for every cue -- no rate separation to gate). The real fix is source-side.
    proxy_share_measured: >
      the entire novel-stem seed-fragility. With di-synaptic blocking at inh_drive=3.0 the affix score for a
      failing novel stem (kick, seed 43) collapses to 0.162 (vs 0.24 unblocked) and loses to the went floor
      (0.215); the whole-form floor itself is stem-INDEPENDENT (~0.20 even for never-encoded regulars), so the
      failure is the blocking pathway mis-firing, not genuine mis-retrieval.
    why_it_matters: >
      a linear relay makes blocking a graded leak that punishes the default for ANY whole-form activity; biology
      gates blocking on a THRESHOLD (a real interneuron only spikes above criterion input), which is exactly what
      lets the rule be the robust default for the unfamiliar.
implemented_by:
  - research/runners/_productive_morphology_recognition_gated_derisk.py
findings:
  - research/findings/2026-08-01-E-language-dual-route-morphology-declarative-route-works-procedural-rule-does-not-generalize-6seed-NOGO.md
  - research/findings/2026-08-26-E-language-recognition-gated-blocking-interneuron-threshold-REFUTED-6seed-NOGO.md
---

# Dual-route past tense — the regular rule is the DEFAULT, blocked only by RECOGNIZED exceptions

**The claim the code must respect (Kandel 6e Ch 55, p.1373):** regular verbs inflect by a rule ("play -> played")
that applies to any stem; irregulars are stored exceptions ("go -> went, not goed"); a child's "goed" is
**over-regularization** — the default rule applied when the stored exception is not retrieved. The rule is the
*elsewhere* case (Pinker-Ullman words-and-rules): it wins **unless** the declarative store delivers a recognized
whole-form.

**The measured wall this binding closes.** On the two-pool spiking substrate the declarative route is solid 6/6
(blocking, over-regularization under lesion, collapse under permuted binding), but the rule's generalization to
novel stems is seed-fragile (reg_acc 0.25-1.0). Diagnosed 2026-08-26: the most-entrenched irregular attractors
(went/ran) leave a **stem-independent ~0.20 floor** in the readout — present even for held-out regulars that were
never co-encoded with any whole-form — and the **linear** whole-form->interneuron blocking drive lets that floor
suppress the default affix for whichever novel stem happens to overlap it most (kick+PAST: affix 0.24 -> 0.16,
lost to went 0.215). The blocking pathway, not the retrieval, is mis-firing.

**The fix, rooted in the companion process.** A real GABAergic interneuron has a **spike threshold**: it should
fire (and block the default) only on **genuine, supra-threshold** whole-form retrieval — the recognition/
familiarity gate the prior build replaced with a constant linear weight. Below threshold (a novel stem's weak
spurious floor) the interneuron is silent, the default "-ed" proceeds unopposed, and the rule generalizes
robustly; above threshold (an entrenched irregular) the interneuron fires hard and blocks -> "went". This is the
recognition-gated blocking the runner implements and tests.
