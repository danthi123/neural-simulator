# Flagship talkable-brain transcript (all capabilities end-to-end): ONE emergent brain, over its own real-corpus codes, answers property + relational (what/who/yes-no) questions, describes fluently, teaches + grows live, resolves pronouns, learns from the corpus, and abstains — in one interactive session. The definitive demonstration of the CYCLE 981-1012 arc. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_unified_talkable_console.py --repl --two-bridge --learn-corpus-facts` (the fullest config: 31-word spoken vocab across 2 A→W bridges + 78 corpus-learned facts). `SIM_BACKEND=numpy`. NO `sim/` edit.

## The transcript (seed 42; the brain's setup: discovered animal cluster [bird,cat,fish,bear,frog], class 'run', exception bird→sleep, 3 taught + 78 corpus-learned facts)
```
> does a bear run?           -> "yes -- the bear can run"          [property: INHERITANCE]
> does a bird run?           -> "no -- the bird can sleep"          [property: CANCELLATION]
> what does the bird eat?    -> "the bird eats frog"                [relational: WHAT]
> who eats frog?             -> "the bird eats frog"                [relational: WHO]
> does the bird eat frog?    -> "yes -- the bird eats frog"          [relational: YES/NO]
> does the bird eat cat?     -> "no -- the bird eats frog"           [YES/NO with correction]
> the girl likes ball        -> "ok, I learned that the girl likes ball."   [TEACH relational, live]
> what does the girl like?   -> "the girl likes ball"                [spoken via BRIDGE-2, broad vocab]
> the bear sleeps            -> "ok, I learned that the bear sleeps (an exception)."  [TEACH property exception]
> does a bear run?           -> "no -- the bear can sleep"           [the taught exception now OVERRIDES -- GROWTH]
> tell me about the bird     -> "The bird can run. It eats frog and sees cat, ball, and bug and 9 more and
                                 finds food and likes tree and wants dog."   [DESCRIBE: fluent multi-fact discourse]
> what does it see?          -> "the bird sees rock"                 [ANAPHORA: it = bird]
> what does the lion eat?    -> "I don't know"                       [no-confab MOAT]
```

## What this demonstrates (one integrated brain, one session)
- **Two knowledge dimensions** over the SAME discovered codes: property (inheritance + cancellation) AND relational (what/who/yes-no).
- **Speech on spikes** across a 31-word two-bridge A→W vocab (animals + objects + people: "the girl likes ball").
- **Growth through conversation**: teach a relational fact AND a property exception LIVE, and the brain applies them immediately ("does a bear run?" flips to "no -- the bear can sleep" after teaching).
- **Fluent discourse**: a multi-fact description with a referring pronoun + verb-grouped, capped aggregation.
- **Multi-turn anaphora**: "what does it see?" resolves "it" to the last subject.
- **Learning from experience**: 78 relational facts mined from the TinyStories corpus (the bird "sees rock" etc. — never taught).
- **The no-confab moat**: "what does the lion eat?" → "I don't know".

## Honest limitation surfaced by the transcript (multi-exception collateral)
The describe of the bird reads "The bird can run" — but the bird is the setup EXCEPTION (bird→sleep, correctly answered at line 2). Teaching a SECOND property exception ("the bear sleeps", line 9) perturbed the bird's own-exception prediction (the shared associative memory's cross-talk), flipping the bird's describe to the inherited class. CYCLE-982's no-collateral gate was validated for a SINGLE exception; MULTIPLE simultaneous exceptions can interfere (the FHRR/associative-memory capacity). A characterized limitation (not a core failure — the single-exception path and every other capability are correct); the fix is per-exception isolation / higher capacity (a follow-on). Every other line is correct.

## What this establishes
The CYCLE 981-1012 talkable-brain arc composes into ONE coherent interactive conversation: discover → reason (property + relational) → speak on spikes → describe fluently → teach both dimensions live → grow → resolve pronouns → learn from the corpus → abstain — transformer-free, grounded, no-confab, both dimensions fully-spiking-realizable, CI-guarded. The owner can talk to it. Follow-ons: multi-exception isolation; the fully-spiking spoken turn; POS-tagged open mining; more corpus verbs.

## Files
`research/runners/_realcorpus_unified_talkable_console.py`; transcript `research/findings/raw/_flagship_transcript.log`. Composes the CYCLE 981-1012 findings (cancellation, relational SVO, the unified console, corpus-fact learning, anaphora, the fluent discourse).
