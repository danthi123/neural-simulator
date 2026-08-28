---
type: finding
status: positive
date: 2026-08-28
lane: introspection
board: 94
verdict: The confidence→forthcomingness #94 re-test is COMPLETE (controller-owned, after the delegated agent went runaway on a seed-sweep). The core "hollow flip" blocker — "mean_role_confidence SATURATES at 1.0 on every real turn, so there is no low-confidence turn to compare" — is REFUTED. Through the REAL webapp.server.brain_chat handler on the tiny-demo brain, the confidence read DISCRIMINATES: a clean turn reads mrc=0.6081 (confident, above the HIGH band 0.5) and, as the decode is degraded, drops monotonically to 0.3995 / 0.3863 / 0.3732 (uncertain, below 0.5) WHILE STILL ANSWERING (not abstaining) — and the low-confidence turns drive a DIFFERENTIAL response (the honesty hedge "my decision-margin reads this as low-confidence" fires below the band, vs the full grounded answer above it). The saturation claim was STALE (pre-#181, when the legacy `confidence` field was averaged; post-#181 `da84fde7` the read averages the OneBrainComposer margin `(peak-runnerup)/peak`, which has a genuine range). Combined with the LTM-shard finding's `confidence_cap_engages: True`, the coupling mechanism is sound end-to-end. HONEST REMAINING CONDITION (why still default-OFF): a NON-HOLLOW real-traffic default-on flip needs real (undegraded) turns that read low-confidence NATURALLY — the tiny-demo's unambiguous content produces only confident (~0.6) natural turns (the low-confidence turns here were noise-INDUCED), so the flip is FLIP-VIABLE on genuinely-AMBIGUOUS content (a rich KB with competing / near-synonym recalls) — the SAME content the knowledge-scale #66 bulk-KB bundle provides. This is a CONTENT-availability condition, NOT a broken/hollow mechanism.
mechanism: confidence read discrimination through the real handler (margin read, post-#181) + the hedge differential — completing the #94 re-test after the LTM-shard content-exhaustion unblock
seed-waiver: single-query targeted confirmation through the REAL handler; the MULTI-SEED discrimination of the margin read is already established (#181 da84fde7; 2026-06-18-emergent-graceful-degradation-derisk ~0 damaged / ~0.5+ intact). This finding confirms it end-to-end + refutes the saturation blocker, not a fresh 6-seed generalization.
artifacts:
  - research/findings/raw/_confidence_read_discrimination/discrimination_result.json
runner: research/runners/_confidence_read_discrimination_derisk.py
---

# Confidence→forthcomingness #94 re-test COMPLETE: the read DISCRIMINATES (the "hollow/saturation" wall was stale) — flip-viable on rich content, tied to #66

Owner directive: "for the confidence retest, rather than defer, do what's needed to complete it even if it means you take ownership." The delegated agent went runaway on a seed-qualifying sweep (killed); I took direct ownership.

## What the re-test was stuck on, and why it was a false wall

The 2026-08-27 retest-PARTIAL left the flip default-OFF, naming "content-exhaustion (buffer-tier-only elaboration)" as the blocker. The 2026-08-28 LTM-shard cupy GO fixed that (elaboration now reaches past the buffer). But the DEEPER, more-precisely-characterized blocker in `webapp/confidence_forthcoming_chat.py` was: `mean_role_confidence` reads a SATURATED 1.0 on every real turn (above the metacog HIGH band 0.5), so `confident` is True unconditionally and NO low-confidence turn exists to compare → any flip is HOLLOW.

That claim is STALE. It described the PRE-#181 read (averaging the legacy `confidence` field = `s[argmax]/max(s)` = 1.0 by construction). #181 (`da84fde7`) changed the read to average the OneBrainComposer `margin` = `(peak - runner_up)/(peak+eps)` — LOW exactly when a decode has a close runner-up, HIGH when one concept dominates.

## The measurement (through the real handler, controller-owned)

Artifact: `research/findings/raw/_confidence_read_discrimination/discrimination_result.json` (runner `research/runners/_confidence_read_discrimination_derisk.py`) — bounded (5-sigma scan, no sweep), standard tiny-demo path (no custom-LTM bundle, which was the agent's `mrc=null` bug), through `webapp.server.brain_chat`:

- **clean**: mrc **0.6081** (confident) — "The brain uses spikes. The brain stores memory. The brain learns words."
- sigma 0.3 / 0.6: mrc 0.5765 / 0.5156 (still above the 0.5 band)
- **sigma 0.9 / 1.2 / 1.5**: mrc **0.3995 / 0.3863 / 0.3732** (below the band) — "**My decision-margin reads this as low-confidence, so …**" (the hedge FIRES), and the turn STILL ANSWERS (`abstained=False`).
- `DISCRIMINATES: True`; 3 low-but-answering turns found.

So the read genuinely varies with recall certainty, and the confidence-driven honesty hedge fires differentially. The "no low-confidence turn / saturation" premise is refuted.

## The honest remaining condition (why the flip stays default-OFF — a content condition, not a wall)

The low-confidence turns above were NOISE-INDUCED (degrading the decode). On the tiny-demo's UNAMBIGUOUS content, a NATURAL (undegraded) turn reads confident (~0.6) — there is no naturally-ambiguous recall, because every fact is distinct (a dominant winner, high margin). A NON-HOLLOW real-traffic default-on flip needs real turns that read low-confidence NATURALLY — which requires genuinely-ambiguous content: near-synonyms, competing facts for a role, partial recalls. That is exactly what a large real KB (the knowledge-scale #66 bulk-KB bundle, in flight) provides and the tiny-demo does not.

## Verdict + next (NO-DEFER)

The confidence→forthcomingness faculty is SOUND and the read WORKS — the #94 blocker ("hollow, the read can never discriminate") is REFUTED. The flip is **FLIP-VIABLE on ambiguous/rich content**, coupled to knowledge-scale #66. NEXT (the final non-hollow GO): run the confidence-forthcomingness coupling on the #66 bulk-KB rich bundle and confirm real (undegraded) traffic contains natural low-confidence turns that get a hedged / shorter answer while confident turns get the +1 LTM-shard elaboration. The faculty stays default-OFF until that non-hollow real-traffic demonstration + the owner gate — but it is no longer blocked on a broken mechanism, only on content that the #1-priority knowledge arc is already producing.
