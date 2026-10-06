---
type: finding
status: live
lane: lane-E
date: 2026-10-06
---

# Word-role parse gate: steps O and R at pin 9ea7411eb

**Status: PROVISIONAL** — verify-go ran (RULING verify-go, two MiMo Pro refuters, 2/2 SURVIVES: lens 1
control-integrity/gate-cheat, lens 2 reproducibility/instrument-trust/claim-precision); awaiting Claude's
confirm + merge.
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
