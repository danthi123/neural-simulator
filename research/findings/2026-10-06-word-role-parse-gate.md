---
type: finding
status: live
claim_check: measured
lane: lane-E
date: 2026-10-06
mechanism: BRAIN_WORD_ROLE_PARSE=1 at BRAIN_WORD_ROLE_OP=0 (read-out mode; a host count-and-margin rule over the
  spiking role-ensemble counts decides each word's role) with BRAIN_NEURAL_SELFID=1, pin 9ea7411eb, lesion flag
  BRAIN_WORD_ROLE_LESION=1
---

# Word-role parse gate: O PASS + R PASS at op 0 (read-out mode; PARTIAL Stage-1 credit)

> **Claude verify-go (2026-10-06): the PASS stands, narrowed.** Claude re-ran `tools/s2p_score.py` O and R from the
> raw data and got byte-identical verdicts. The preregistration (commit 746f9a3a4, 2026-10-02 00:11 UTC) was
> committed before the step-1 sweep was queued, and only frontmatter changed after it. Every O file and all 48 R arm
> files carry pin 9ea7411eb (not dirty). In each R shard the lesion arm differs from the intact arms by one variable
> (`--env {"BRAIN_WORD_ROLE_LESION": "1"}`), the two intact arms agree (null clean), and the lesion reproduces. The
> intact reply is "the brain uses the spikes" with `recalled_svo` [brain, use, spikes]. The positional map would read
> (brain, usually), so this reply shows the organ ran in the intact arm. Seven caveats follow; they narrow the headline
> and the credit sentence below.
> (1) **Read-out mode.** The spiking network computes role-ensemble spike counts. A frozen host rule turns them into
> the parse: a word gets a role if its top count is at least ROLE_MIN 9 and at least 2x the runner-up, and the host
> then takes the highest-count word per role. On the gate seeds, heard words read 15-34 spikes and the unheard
> "usually" reads 1-7, so the host threshold is what drops "usually". That is the same shape as T18's read-out mode.
> Under the principle of owner RULING `t18-credit`, the credit is **PARTIAL, not FULL**, even after P4. The next method
> is a spiking read-out: an all-or-none commit population per role that replaces the count rule and the argmax.
> (2) **Four host residuals, not two.** Besides the self-alias resolution and the stop-word strip that the
> preregistered sentence names, the word-form to assembly dictionary (including the spare-assembly assignment) is host,
> and the training stimuli are transcribed from the composer's fact index. The runner docstring declares all four.
> (3) **"Never heard, no role" holds by construction.** An unheard word drives a spare assembly with no learned
> synapses, so it reaches the role ensembles only through position. The margin is thin: position alone reads up to 7
> spikes on the gate seeds (seeds 43, 44 and 100), against ROLE_MIN 9.
> (4) **Known limit, measured on the gate seeds.** Words taught in two roles (cat, fish) are read in the right role in
> 13 of 24 diagnostic readings. The others decline (abstain); none read a wrong role. This sits outside the bar. A
> default flip turns some questions about such words from answered (positional map) into abstained, so P4's steps B and
> F1 must check them.
> (5) **What the reply-level gate covers.** The probe's agent is at position 0 under either map, so step R tests the
> action slot only. Choosing the agent by word (the `swapped` case) is shown at organ level only (O, 6/6). The gate
> compares the organ with its own lesion. The flag-off positional reply to this probe is argued from the code
> (prereg section 5) and is not measured here. All six seeds give one deterministic reply template.
> (6) **Weak control.** In the intact control arm the parse declines ("drink" was never heard), the same path the
> lesion takes. So the control shows that the lesion leaves an already-declining parse alone. It does not test
> off-target effects of the lesion on a parse that succeeds.
> (7) **Lesion engagement is inferred, not recorded.** `ChatBrain._neural_question_parse` catches every organ exception
> and declines, and the arm files carry no word-role trace. The lesion arm differs from the intact arm only by the gate
> value read in `_read` (0.0 against 1.0). The intact arms show the organ builds and parses on the same knowledge base
> at the same seed, and O's `lesion_self` case (6/6) shows the closed-gate path runs without error. On that basis a
> crash-only abstain is implausible, but nothing records it. A `word_role_trace` (route ran, lesion flag, swallowed
> error, per-word roles) is owed before P4's F1 lines for this packet.
> **Update (Claude, 2026-10-06, branch claude-review/word-role-trace):** the trace now exists. The reply carries
> `word_role_trace` on every turn where the word-role parse ran, and `tools/s2p_score.py` makes a word-role-parse R
> cell UNDEFINED:engagement-* unless every intact and lesion arm records it. A re-score of the 9ea7411 R data
> therefore now reads UNDEFINED:engagement-no-record, by design; the PASS above stands on this caveat.

**Status: confirmed by Claude (2026-10-06), narrowed as above.** Hermes's own verify-go ran two MiMo Pro refuters
(lens 1 control-integrity/gate-cheat, lens 2 reproducibility/instrument-trust/claim-precision; 2/2 SURVIVES).
Not yet the faculty's default path: `BRAIN_WORD_ROLE_PARSE` is default OFF (`_WORD_ROLE_PARSE_DEFAULT_ON = False`).
Preregistration: `research/findings/2026-10-01-word-role-parse-PREREGISTRATION.md`.
Pin: `9ea7411eb0f219560c83816c84bc75cae4602369` (PINS.tsv P row, 2026-10-02).
Scored 2026-10-06; verdicts: `research/findings/raw/_s2p/word-role-parse/9ea7411/verdicts/O.json`
and `research/findings/raw/_s2p/word-role-parse/9ea7411/verdicts/R.json` (with `.prov.json` siblings), data
commit `99906f6fc` on
`research/word-role-parse`.

## Step O — the organ gate (6 gate seeds, selected index op=0)

From `research/findings/raw/_s2p/word-role-parse/9ea7411/verdicts/O.json`: verdict **PASS**; `n_all_pass=6`, `n_defined=6`, `op=0`;
every gate seed (42, 43, 44, 100, 101, 102) `state=DEFINED`, `all_pass=true`,
`failed_cases=[]`; the one precondition ("organ cells DEFINED on 6/6 seeds") `ok=true`
(`measured=6 expect=6`); `undefined_reasons=[]`.

## Step R — reply-level gate (real handler, 12 LB shards, same six seeds)

From `research/findings/raw/_s2p/word-role-parse/9ea7411/verdicts/R.json`: verdict **PASS**; `op=0`, `at_default=false`;
`n_defined=6`, `n_reply_load_bearing=6` — lesion of the word-role parser changes the
reply on 6/6 seeds (each seed `reply_verdict=regressed`: intact answers
"the brain uses the spikes" with `abstained=false`, lesion regresses to the
abstention "I don't know about that. My curiosity is piqued — I haven't learned about
usually yet: what can you tell me about usually?"; `changed_fields=[abstained, answer,
recalled_svo]`); control row `word-role-nofact-ctl` `n_ok=6`, `n_defined=6`
(6/6 `control-ok`: intact and lesion both abstain identically, `changed_fields=[]`).
Both preconditions ("capability cells DEFINED on 6/6 seeds", "permuted-control cells
DEFINED on 6/6 seeds") `ok=true` (`measured=6 expect=6`); `undefined_reasons=[]`.

This is **trace-level/organ** evidence for step O and **reply-level load-bearing**
evidence for step R (TERMS.md 09-29: unqualified "load-bearing" = reply-level).

## What a PASS lets the finding say (preregistration section 6, fixed sentences)

> The question's roles are assigned by a spiking parser that reads the words. It was
> trained on the brain's own heard sentences, and a word it never heard gets no role.
> The positional map no longer decides which word is the agent. Self-alias resolution
> and the stop-word filter stay host (named residuals). The Stage-1 credit for
> neural-selfid becomes FULL only when P4 makes this the default and steps B/F pass
> there.

Read these sentences with the caveats above. "Assigned by a spiking parser" means spiking counts with a host
read-out (caveat 1). "No longer decides" holds with `BRAIN_WORD_ROLE_PARSE=1`, which is default OFF until P4. The
named residuals are four (caveat 2). The last sentence is superseded: the credit stays PARTIAL in read-out mode
(caveat 1), whatever P4 does.

`at_default=false`: the organ ran enabled-for-the-gate (default OFF behind
`BRAIN_WORD_ROLE_PARSE`), so this gate does not by itself flip any default.

## Reproduction

```
cd .claude/worktrees/word-role-parse
.venv/bin/python tools/s2p_score.py O --packet word-role-parse \
    --dir research/findings/raw/_s2p/word-role-parse/9ea7411/O \
    --selection research/findings/raw/_s2p/word-role-parse/9ea7411/selection.json \
    --pin 9ea7411eb0f219560c83816c84bc75cae4602369 \
    --out research/findings/raw/_s2p/word-role-parse/9ea7411/verdicts/O.json
.venv/bin/python tools/s2p_score.py R --packet word-role-parse --tag s2p-R-wr-9ea7411 \
    --pin 9ea7411eb0f219560c83816c84bc75cae4602369 \
    --selection research/findings/raw/_s2p/word-role-parse/9ea7411/selection.json \
    --out research/findings/raw/_s2p/word-role-parse/9ea7411/verdicts/R.json
```

Both scored `PASS op=0` (rc=0) on 2026-10-06.
