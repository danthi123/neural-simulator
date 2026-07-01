# Fluid conversation — Phase 7 GO: a NEURAL interrogative parser (burn the host question-parse scaffold)

**2026-07-01 (autonomous night; the brain-based-only / burn-down-shortcuts standard).** The Phase-3/4/console
question comprehension used a HOST rule to detect the wh/aux word → query-type. Per BRAIN-BASED-ONLY (comprehension is
the brain's job), this de-risks the brain-based replacement, reusing VALIDATED mechanisms (zero new mechanism):

- **wh → query-type via the composer:** the wh-word is a LEXICAL cue the BRAIN learns — stored in the validated
  composer as facts (`"what" queries "patient"`, `"who" queries "agent"`, `"does" queries "yesno"`) and recalled
  (`what_does(wh, "queries")`). Genuinely brain-based (the composer that stores all the conversational facts).
- **content → roles via the BridgeParser:** the query-type selects the 3-slot SVO frame (the queried slot = a
  placeholder); the validated position-parser (`BridgeParser.parse`) assigns the content words to roles → the cue.

## Result — GO (3 seeds)
`_fluidconv_phase7_neural_interrog_parser_derisk.py`, held-out questions (what/who/yes-no):

| gate | result (3 seeds) |
|---|---|
| NEURAL match the ground-truth (query-type + cue) on held-out questions | **10/10** all seeds |
| PERMUTED wh→type anti-cheat — the map is scrambled → the what/who cases break | **3/10** (only the yes/no cases, invariant under the permutation, stay correct → the wh→type is load-bearing for the discriminating what/who cases) |
| LESION — do NOT store the wh→type facts → the composer abstains → cannot map | **abstain 10/10** all seeds |

Sample: *"what does the dog eat?" → (patient, [dog, eat]); "who eats meat?" → (agent, [eat, meat]); "does the dog eat
meat?" → (yesno, [dog, eat, meat])*, all via the composer + parser.

## Honest scope (the residual, flagged)
- The **wh → query-type** mapping — the clearest host part of the scaffold — is now **brain-based** (composer recall),
  and load-bearing (permuted breaks it, lesion abstains).
- **Residual (defensible-brain-based, not the burned-down cheat):** (1) identifying which tokens are CONTENT vs
  function words uses the vocab (the brain's known concepts — function words simply aren't concepts); (2) the
  query-type → SVO-frame slot map is a small structural fact (exactly the FrameParser's per-frame position→role
  definitions). Both are lexical/structural cues the brain legitimately holds, not the host wh-detection this closes.
- The permuted control's 3/10 residual is the yes/no cases (my permutation kept `does→yesno` — only one yes/no type,
  so it can't be permuted meaningfully); the discriminating what/who cases (7) all break under permutation,
  confirming the wh→type is load-bearing.

**⇒ the question-parse shortcut is burned down to brain mechanisms** (the composer + the BridgeParser), reusing
validated pieces, NO new mechanism, NO `sim/` edit. Connects to the multicue-parser frontier
(`project_conversational_primary_robust_multicue_parser`): interrogatives are one more comprehension frame the brain
handles. This closes one of the tracked fluid-conversation shortcuts (the others: the ANN generator → spiking-forward,
deferred; growth over pre-allocated codes → the dendritic frontier).

**Artifacts:** `research/runners/_fluidconv_phase7_neural_interrog_parser_derisk.py`; result
`research/findings/raw/_fluidconv_phase7_neural_interrog_parser.json`.
