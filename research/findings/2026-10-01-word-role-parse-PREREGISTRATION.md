---
status: preregistration
date: 2026-10-01
packet: word-role-parse
runner: research/runners/word_role_parser_organ.py
scorer: tools/s2p_score.py
---
# word-role-parse: the self-ID role map reads the words (PRE-REGISTRATION, 2026-10-01)

Written by the cloud Claude session 2 (packet factory) before any gate-seed run. HANDOFF:
`docs/drafts/word-role-parse/HANDOFF.md`. Owner ruling (runbook section 6, `iw2-stage1-credit`): the neural-selfid flip
counts toward Stage 1 as PARTIAL, and is not double-counted, until its role map moves into the spiking substrate.

## 1. The host decision being moved

`ChatBrain._neural_question_parse` presents (content[0], content[1], "__q__") to `BridgeParser.role_of(position)`. That
(position, voice) -> role conjunction never receives the words, so the parse IS the positional map content[0] = agent,
content[1] = action, whatever the words are. Runbook RULING `iw2-verify-go`: neural-selfid is "the content-selection
parser, the same positional role map, reaching self questions". Which word plays which role is decided by position,
which the host fixed.

## 2. The mechanism (code at the pinned revision; default OFF behind `BRAIN_WORD_ROLE_PARSE`)

A spiking parser with one small input assembly per vocabulary word, three position assemblies and three role ensembles
(agent / action / patient) under lateral inhibition. Its word -> role and position -> role projections are Hebbian.
They are trained by replaying the brain's OWN stored sentences: each fact's agent word at position 0 with the agent
role, the action at 1, the patient at 2. Position alone is subthreshold. A word gets a role only if its role ensemble
clears a frozen count and margin, so a word the parser never heard gets no role and is skipped. When the flag is on,
`_neural_question_parse` returns the parser's (agent, action) or declines. `BRAIN_SUBSTRATE_PARSE_LESION` is still
checked first, so the iw2 row keeps its meaning.

Lesion `BRAIN_WORD_ROLE_LESION=1`: the word -> role projection is cut, position alone stays subthreshold, the parse
declines and the route abstains.

Declared residuals: the training stimuli are transcribed from the composer's fact index (a host transcription of heard
sentences). The self-alias -> 'brain' resolution (`QuestionRouter._resolve_self`) stays host and is named as the next
rung. The content-word filter (stop-word strip) is unchanged. The organ is its own bridge (a one-brain debt).

## 3. Step 1: operating point (dev seeds 1-8 only; no verdict)

Every GRID index runs the organ battery on dev seeds 1-8 (pool, `--sweep`). Selection, by `tools/s2p_score.py select`
only: the FIRST index in grid order passing on all 8 dev seeds. If none passes, the method is banked and NEEDS-CLAUDE
follows. Dev record before registration (the build agent, local numpy, 2026-10-01; no gate weight): GRID 0-4 passed all six cases on dev seeds 1-8; GRID 5 passed 3 of 8 (position alone too strong), GRID 6 and 7 passed 6 of 8. At GRID 0 position alone gives at most 7 role spikes against ROLE_MIN 9, a heard word at least 10: a thin margin, so GRID 1-3 are passing neighbours. Known limit: a word taught in two roles is resolved at the right position in 17 of 32 dev readings and otherwise declines, never a wrong role.

## 4. Step O: the organ gate (gate seeds 42/43/44/100/101/102, the selected index)

Training facts (dog chase cat), (cat eat fish), (bird eat worm), (brain use spikes), (brain have neurons), (fish swim
lake). Cases: svo, parse(dog, chase) -> (dog, chase); self, parse(brain, use) -> (brain, use); adverb,
parse(brain, usually, use) -> (brain, use), where "usually" was never heard and gets no role; swapped, parse(chase, dog) ->
(dog, chase), where the positional map fails; novel, parse(zorp, glim) -> decline; lesion_self (lesion),
parse(brain, use) -> decline. **Bar: PASS iff `all_pass` on 6/6 gate seeds**, with cell validity as in the
fb-spiking-witness prereg section 4.

## 5. Step R: reply-level gate (real handler, LB shards, the same six seeds)

Rows `word-role-selfid` (probe "what do you usually use?": intact states the built-in self fact, "spikes"; lesion loses it) and the
control `word-role-nofact-ctl` ("what do you usually drink?": intact abstains, lesion unchanged), in
research/runners/lbf_rows/word_role_parse.py. Both arms set `BRAIN_NEURAL_SELFID=1`, `BRAIN_WORD_ROLE_PARSE=1` and
`BRAIN_WORD_ROLE_OP=<selected>`. **Bar: PASS iff the capability is reply-level load-bearing on 6/6 AND the control
holds 6/6**, under every rule of `tools/iw2_score.py` step R plus the operating-point sidecar check. With the flag off,
the positional parse of this probe reads (brain, usually), so the iw2 route cannot answer it. That makes the capability
cell a test of the word-conditioned map, not of the route.

## 6. What a PASS lets the finding say (fixed sentences)

"The question's roles are assigned by a spiking parser that reads the words. It was trained on the brain's own heard
sentences, and a word it never heard gets no role. The positional map no longer decides which word is the agent.
Self-alias resolution and the stop-word filter stay host (named residuals)." The Stage-1 credit for neural-selfid
becomes FULL only when P4 makes this the default and steps B/F pass there.

## 7. Seeds, cost, files

Dev seeds 1-8 (step 1); gate seeds 42, 43, 44, 100, 101, 102. Step 1: 8 pool lines. Step O: 6 pool lines. Step R: 12 LB
shards, `mem_gb=8`. Outputs go under `research/findings/raw/_s2p/word-role-parse/<P7>/` and
`research/findings/raw/_load_bearing/_shards/s2p-R-wr-<P7>/`.
