# HANDOFF packet: neural-selfid (integration wave 2)

Run the shared flow in `../HANDOFF.md`; this file holds what is specific to this packet. Prereg:
`research/findings/2026-09-30-iw2-neural-selfid-flip-PREREGISTRATION.md`.

**What flips.** The existing anchor `_NEURAL_SELFID_DEFAULT_ON` in `research/runners/brain_chat_tui.py` turns on
`BRAIN_NEURAL_SELFID` ALONE (its 2026-09-05 partner `BRAIN_NEURAL_ANAPHORA_ABSTAIN` stays off): a question that names
the brain itself and carries an action ("what do you use?") goes through the on-brain BridgeParser and the substrate
recall like every other factual question, and a bare identity question ("what are you?") is answered by the substrate
recall; the host QuestionRouter no longer owns either class. Recorded as a sub-rung of ledger row `content-selection`
(whose declared residual names exactly this: "still the self/identity + noisy-anaphora fallback"). Escape after the
flip: `BRAIN_NEURAL_SELFID=0`.

**Why it was chosen.** Rank 3 of the wave (shared prereg). It moves a decision from host code to the brain, which is
what the current phase measures (the host-decision share), and the flip is the proof: once on, the content-selection
row's own parse lesion changes the self-factual reply; while off, the router answers regardless of the lesion.

## Step R for this packet (shared HANDOFF sections 3-5)

Job file `jobs/R_si_<P7>.txt` (12 lines), tag `iw2-R-si-<P7>`: row `neural-selfid` ("what do you use?" in a fresh
session; lesion `BRAIN_SUBSTRATE_PARSE_LESION=1`, the parser's role-conjunction read cut) and control
`selfid-nofact-ctl` ("what do you drink?": no stored self fact, the substrate route must abstain). Score:
`tools/iw2_score.py R --packet neural-selfid ...`.

| R verdict | per-seed detail (`R_si.json`) | do |
|---|---|---|
| PASS | capability 6/6 `regressed` (the intact reply names spikes), control 6/6 `control-ok` | into the flip list (if D0 and D_neural-selfid read PASS / DIFFERS-IN-SCOPE) |
| FAIL | a capability seed `pass`, `intact-wrong` or `lesion-kept` (the lesion reply still names spikes; prereg Amendment 1), or a control `control-violated` (a self fact confabulated for "what do you drink?") | not flipped. Draft `research/findings/<date>-iw2-neural-selfid-reply-level-NO-GO-<n>of6.md` (provisional): per seed, the intact and lesion answers of both rows, and which rule failed |
| UNDEFINED | a cell not DEFINED | shared re-run rule once; still UNDEFINED: not flipped, NEEDS-CLAUDE "iw2 si R UNDEFINED: <seed>: <fails>" |

## Production default (shared steps B and F)

F1: in `iw2-flip-<F7>` the row `neural-selfid` must read reply-level load-bearing on 6/6 (step R's guards). Watch B1
for `content-selection`, `semantic-recall`, `moat-verify`, `spiking-anaphor` and `wm-binding-advanced`: the 2026-09-05
pair flip regressed an anaphora turn at seed 44 (attributed to the other flag); a drop there goes through the shared
attribution and bisect rows like any other.

## Finding template (after the shared section 7 reads PASS)

`research/findings/<date>-iw2-neural-selfid-default-on-flip-validated-PROVISIONAL.md`, frontmatter as in the
false-belief-chat template with `integration_faculty: content-selection`. Sections: (1) Result: R, D0/D/S, B, F1, F2,
F3 (numbers from the verdict JSONs). (2) Host-decision share: which self/identity questions the router answered at P
and the substrate answers at F (the identity arms' `source` field, turns 14-16). (3) Declared residuals from the
prereg. (4) Deviations. (5) Honesty: "the brain's parser comprehends a question about itself"; this is not a claim of
self-awareness (non-negotiable 5). Never write "integrated" or "closed".

## Dev seed 7 (Claude, 2026-09-30)

See the shared HANDOFF section 9.

## NEEDS CLAUDE (packet-specific)

1. The Stage-1 C2 anaphora sub-probe interaction (shared NEEDS-CLAUDE 3).
2. With the flag on, "what are you?" / "who are you?" change from the router's keyword reply ("the brain uses the
   spikes") to "I don't know about that." (dev seed 7, prereg "A known behaviour change"). Step D shows it on identity
   turns 14-15 (as `reply_differing_turns` in `D_neural-selfid.json`; that is expected, not a failure). Decide at merge
   whether an honest abstain on these two questions is acceptable, or whether the flip waits for a class-(b) route
   that answers them from the substrate.
