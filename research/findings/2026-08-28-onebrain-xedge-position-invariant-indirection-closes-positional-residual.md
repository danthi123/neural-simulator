---
type: finding
status: live
date: 2026-08-28
verdict: The one-brain d6-WM->comprehension cross-edge flip's LAST residual (positional-binding, `2026-08-28-onebrain-xedge-reverify-after-crashfix-still-NOGO-positional-binding`) is CLOSED by a position-invariant indirection in the comprehension READ path (`_wm_resolved_role_indirect`). Through the real handler the WM focus is the fixed positional CAND_POOLS[0]=w0 whose grown role is seed-arbitrary, so a fixed-slot read was visible only for the 2/4 seeds where w0 held a grown AGENT role; seeds 101/102 (w0=p_patient) resolved patient == the test item's content role -> no visible change (flagged hollow). The fix READS the substrate's own candidate role-population (probe each candidate pool's balanced amb_read margin) and binds the held topical referent to the AGENT slot INVARIANT to which position (w0/w1/w2) grew it (Kriete 2013 PFC/BG indirection); a held-but-ungrown (ctrl) referent stays correctly inert; the lesion collapses every margin to baseline -> reverts. Numpy organ-level wiring smoke (4 diagnostic seeds through the explicit-wm_focus real-handler path): the previously-hollow patient-position seeds 101/102 now resolve AGENT (visible) + lesion-revert, agent-position seed 42 stays visible, ctrl-position seed 43 stays correctly inert; n_hollow=0, seed_ok=4/4. numpy != cupy -- the DECISIVE 6-seed cupy re-verify through the real handler is STAGED on the primary gpu.queue; the controller flips BRAIN_ONEBRAIN_XEDGE + _LEARN default-ON only on a genuine non-hollow cupy GO.
mechanism: position-invariant indirection (substrate role-population read) routes the held WM referent to the grown AGENT role regardless of candidate position; behind BRAIN_ONEBRAIN_XEDGE (default-OFF), only the READ is position-invariant, edge weights unchanged
lane: onebrain-integration-xedge-flip
artifacts:
  - research/findings/raw/_xedge_flip_verify/indirection_numpy_wiring_smoke.json
runner: research/findings/raw/_xedge_flip_verify/verify_indirection_numpy_smoke.py
---

# One-brain xedge flip — position-invariant indirection closes the positional-binding residual

The prior NO-GO ([`2026-08-28-onebrain-xedge-reverify-after-crashfix-still-NOGO-positional-binding`](2026-08-28-onebrain-xedge-reverify-after-crashfix-still-NOGO-positional-binding.md)) isolated the flip's last residual to a POSITIONAL-BINDING mismatch, not hollowness: through the real `/api/brain-chat` handler the WM focus is the fixed positional `CAND_POOLS[0]=w0`, and its grown role is seed-arbitrary (`_role_assignment`), so a fixed-slot read of that one pool is visible only when the position happens to carry a grown AGENT role.

## The mechanism (what the fix does)

`research/runners/comprehension_production_organ.py` — new `_wm_resolved_role_indirect`, called from `_wm_resolved_role` ONLY when the caller threads `wm_focus` explicitly (the production/real-handler path; the offline self-tests use the legacy ambient-focus fallback and are byte-identical, so their per-referent variation `hold(p_agent)->agent` vs `hold(p_patient)->patient` is preserved):

- READ the substrate's OWN candidate role-population — probe each candidate pool's balanced (content-cancelled) `amb_read` margin (the SAME F2 instrument; only the READ is position-invariant, the edge weights are the learned ones).
- (a) if the positionally-held referent does NOT itself carry a grown role (`|margin(foc) - baseline| < eps` -> an ungrown distractor/control pool), stay inconclusive: the content role stands (a held-but-ungrown referent is CORRECTLY inert, unchanged from before).
- (b) otherwise bind the held (topical/discourse-given) referent to the AGENT slot POSITION-INVARIANTLY — route the drive to whichever candidate pool the substrate shows drives `sel_agent` most (argmax(margin - baseline)), regardless of which position (w0/w1/w2) grew it. The topic/given-referent-as-subject/agent linguistic universal, realised as a Kriete indirection pointer.
- Under the cross-edge LESION every candidate margin collapses to baseline -> branch (a) fires -> reverts to the content role (lesion-attributable).

Grounded in Kriete, Noelle, Cohen & O'Reilly 2013 (PNAS; PMID 24062434 — position-invariant variable binding via a PFC/basal-ganglia INDIRECTION pointer rather than a fixed slot). Additive; still behind `BRAIN_ONEBRAIN_XEDGE` (default-OFF); no flag surface widened.

## Numpy wiring smoke (organ-level, explicit-wm_focus = the real-handler path)

Artifact: `research/findings/raw/_xedge_flip_verify/indirection_numpy_wiring_smoke.json` (runner `verify_indirection_numpy_smoke.py`; test item "the wolf watches the owl", content role = patient; `b_edge=learn` converged edge). Each seed: no-focus content read vs `wm_focus='w0'` held read vs held-under-lesion. <!--derived-->

| seed | w0 role | novisi role | held role | wm_resolved | lesion role | visible | hollow | inert | reverts |
|------|---------|-------------|-----------|-------------|-------------|---------|--------|-------|---------|
| 42   | p_agent   | patient | agent | True  | patient | yes | no | no  | yes |
| 101  | p_patient | patient | agent | True  | patient | yes | no | no  | yes |
| 102  | p_patient | patient | agent | True  | patient | yes | no | no  | yes |
| 43   | p_ctrl    | patient | patient | (None) | patient | no  | no | yes | yes |

`n_hollow=0`, `seed_ok=4/4`. <!--derived--> Seeds 101/102 (the previously-hollow patient-position seeds) now resolve AGENT (visible, `agent != content patient`) and revert under lesion; the agent-position seed 42 stays visible; the ctrl-position seed 43 stays correctly inert.

## Honesty boundary + what is STILL a declared residual

`numpy != cupy` — this is a WIRING smoke, not the production verdict. The DECISIVE 6-seed cupy re-verify through the real handler (`_xedge_flip_production_verify.py`, writes `flip_verify_cupy_6seed_indirection.json` into the `research/findings/raw/_xedge_flip_verify/` dir) is STAGED on the primary `gpu.queue`; the controller flips `BRAIN_ONEBRAIN_XEDGE` + `_LEARN` default-ON only on a genuine non-hollow cupy GO (`n_hollow=0`, all 4 grown-focus visible, byte-identical-off, no-regression). The SEMANTIC referent->pool binding (so DIFFERENT real referents on real traffic resolve DIFFERENT roles, rather than the topical referent defaulting to agent) remains a DECLARED residual — a later rung; the per-referent agent-vs-patient discrimination CAPACITY is proven at the organ level (the unchanged PART-2/PART-3 self-tests).
