---
type: finding
status: contributing
date: 2026-08-12
mechanism: E1 self-model / metacognition WIRED into the DEFAULT /api/brain-chat turn — an honest FUNCTIONAL confidence read-out. The co-resident spiking balance-of-evidence monitor (the workspace WTA margin |rate(asm1)-rate(asm0)| read directly from cp_firing_states, reuse-by-import from the metacog balance de-risk, 6/6 GO) reads the confidence of the answer the brain is about to give (evidence = the brain's own mean role-decode confidence); a LOW-confidence answer gets an honest hedge PREPENDED. Default-ON, moat-safe (only qualifies an already-produced answer, never fabricates content or flips an abstain), lesion-load-bearing, NO sim/ edit.
lane: Gate-B / E1 · Self-model / metacognition (honesty-boundary confidence read-out)
lane_ref: E1
verdict: GO / WIRED (production-integration). Single-process synchronous in-process verify on the real /api/brain-chat handler (SIM_BACKEND=numpy, GPU-free stub renderer, rf composer, rich=False single-fact path — the metacog block runs in BOTH the rich and single-fact paths). 11/11 verify checks pass.
seed-waiver: production-INTEGRATION verify of an already-6/6-GO faculty (the E1 de-risk 2026-08-12-laneC-metacog-INSTRUMENT-FIX-and-balance-of-evidence-pure-spiking-meta-d.md, balance read, meta-d'>0 on all 6 seeds). This doc verifies the deterministic WIRING glue on the real handler (single process, one seed=42 organ); the 6-seed statistical evidence is the cited de-risk. Lesion + flag-off arms are decisive on the single wired seed.
artifacts:
  - research/findings/raw/_gateB_metacog_production_verify.json
---

# Gate-B / E1: an honest functional metacognitive confidence read-out on the default chat turn

**Status:** GO / WIRED. The brain now reads a genuinely-SPIKING confidence of the answer it is about to
give and, when that confidence is LOW, honestly QUALIFIES it ("My decision-margin reads this as
low-confidence, so take it as uncertain: ...") instead of asserting a marginal recall as if it were
certain — a FUNCTIONAL metacognition read-out, never a phenomenal claim.

## The wire (reuse-by-import; NO `sim/` edit)

`research/runners/metacog_production_organ.py` builds ONE co-resident metacog-workspace bridge
(`build_metacog_bridge(confidence_read="balance")`, reused from the E1 de-risk) and calibrates a
confident-vs-uncertain threshold from a synthetic high/low-evidence battery. On the production turn the
brain's OWN mean role-decode confidence (its spiking thematic-role parse certainty for the answer, read off
the composer trace) is the evidence; it is delivered as the graded drive to the workspace WTA and the
settled MARGIN `|rate(asm1)-rate(asm0)|` off `cp_firing_states` IS the confidence. A margin below the
calibrated threshold prepends an honest hedge. `webapp/server.py brain_chat` runs the read AFTER the
gate/moat has already produced the answer (both the rich and the single-fact paths), so it only QUALIFIES;
it never manufactures a fact, changes WHICH answer the recall produced, or flips an abstain.

The evidence DERIVATION (role-decode confidence) is a declared host boundary; the balance MARGIN read + the
threshold decision are the load-bearing SPIKING part — exactly the affect organ's pattern (host appraisal
injection + spiking ladder read-back) and the surprise organ's (host sensory encoding + spiking mismatch read).

## Verify — 11/11 (real handler, numpy-CPU). Artifact `research/findings/raw/_gateB_metacog_production_verify.json`

<!--derived-->

(The numbers below are rounded reads of the cited verify artifact
`research/findings/raw/_gateB_metacog_production_verify.json`, whose full-precision values are the ground truth.)

Measured on the tiny-demo recalls: mean role-decode confidence spans 0.400 ("dog chase cat", the
lowest-confidence recall) .. 0.476 ("brain use spikes"/"what are you", the highest); the normalization
places 0.400 -> evidence 0.29 and 0.476 -> evidence 0.74, straddling the calibrated margin threshold 0.001422.

- **HIGH-confidence recall -> NO hedge.** "what does the brain use" -> "The brain uses spikes." (role-conf
  0.476, evidence 0.738, balance 0.001891 >= threshold -> confident=True; no hedge).
- **LOW-confidence recall -> honest hedge.** "what does the dog chase" -> "My decision-margin reads this as
  low-confidence, so take it as uncertain: The dog chases cat." (role-conf 0.400, evidence 0.295, balance
  0.000953 < threshold -> confident=False).
- **ABSTAIN -> skipped.** "what does the dragon do" -> "I don't know about that." (metacog null — no answer
  to qualify; the moat is untouched).
- **LESION-LOAD-BEARING.** `BRAIN_METACOG_LESION=1` removes the evidence DIFFERENTIAL from the workspace
  (drives both assemblies at base); the SAME high-confidence "what does the brain use" FLIPS to hedged
  (balance 0.000891 -> confident=False). So the confident/uncertain discrimination is caused by the SPIKING
  margin reading the evidence, not by a host threshold on the raw role-conf scalar (the lesion changes only
  the workspace drive, not the role-conf).
- **FLAG-OFF byte-identical.** `BRAIN_METACOG=0` -> metacog null on both recalls; the answers are
  byte-identical to the un-hedged answers ("The brain uses spikes." / "The dog chases cat.").
- **NO-REGRESSION with ALL organs default-ON.** recall ("brain use spikes"), anaphora ("what does it eat"
  -> "The cat eats fish."), abstain (dragon), D2 surprise ("the cat eat grass" fires 5.61 Hz surprised >
  threshold 2.73), D4 comprehension (OOV "blork zonk plemf" -> honest "didn't follow" abstain) — all hold.
- **Canonical `--smoke`** GO on the modified tree (the TUI path does not import server.py or the organs, so
  it is byte-identical by construction).

## Honest residuals (declared — the mission's named next rungs, not faked)

- **EVIDENCE = the parse confidence** (a COMPONENT of answer confidence), not a full
  recall-vs-alternatives balance; the substrate's other recall signals (rf readout magnitude / frac) are
  saturated on the tiny-demo, so the role-decode confidence is the graded signal available. A richer
  recall-margin evidence is the next rung.
- **NOT type-1/type-2 DISSOCIABLE.** Balance-of-evidence confidence is an ENCODING read (the de-risk's mapped
  boundary: loop-ablation does not collapse it); this is a genuine decision-variable read, not a separable
  second-order monitor. The architecturally-dissociable comparator (`margin_abs`) is seed-fragile and is the
  named next rung. The load-bearing lesion here is therefore on the evidence encoding (as the finding predicts).
- **NARROW DYNAMIC RANGE.** The balance margin's absolute range on this operating point is small; the wire
  reliably separates clearly-high from clearly-low evidence, the mid-range is a boundary (the read is averaged
  over 8 jittered reads to denoise). The tiny-demo's role-conf band is likewise narrow (see Verify above).
- **CO-RESIDENT** on its own metacog-workspace bridge ALONGSIDE the recall composer — rides on the one-brain
  merge (burn-down #1), exactly as the affect/surprise/comprehension organs do.

A FUNCTIONAL metacognition CORRELATE, NOT a claim of subjective experience (phenomenal consciousness is OPEN).

## Escape / lesion knobs

```
BRAIN_METACOG=0          # disable -> byte-identical oracle (no confidence read, no hedge)
BRAIN_METACOG_LESION=1   # remove the evidence differential -> the margin collapses -> a confident answer hedges
```
