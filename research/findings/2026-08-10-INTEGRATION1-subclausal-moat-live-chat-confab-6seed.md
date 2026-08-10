---
type: finding
status: contributing
date: 2026-08-10
mechanism: integration1-subclausal-noconfab-moat-live-chat
lane: Stage-A integration (live conversational honesty path)
instrument: per-seed n_confabulations headline + a target-metric decomposition (per-turn svo_moat_confabulation and any 'because'-clause leak) + the turns-3/4/5 causal-clause drop + the curiosity-ask (turn 6) / honest-abstain (turns 7-14) / France-no-Paris regression check, all read from research/findings/raw/lanes/stageA/turing/conversation_turing_test_s{42,43,44,100,101,102}.json ; BEFORE = git f86b489f^ s42 (3 ungrounded because-clause confab turns), AFTER = the 6 committed artifacts
artifacts:
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s42.json
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s43.json
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s44.json
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s100.json
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s101.json
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s102.json
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s42_transcript.md
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s43_transcript.md
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s44_transcript.md
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s100_transcript.md
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s101_transcript.md
  - research/findings/raw/lanes/stageA/turing/conversation_turing_test_s102_transcript.md
  - research/runners/_conversation_turing_test_derisk.py
---

# INTEGRATION #1 CONFIRMED (6 seeds) — the sub-clausal no-confab moat, now LIVE in the conversational path, eliminates the ungrounded 'because'-clause confabulations on EVERY seed; grounded facts + curiosity + honest abstains all survive

<!--derived-->

**2026-08-10.** INTEGRATION #1 (commit `f86b489f`) wired the SUB-CLAUSAL no-confab moat (`baa635dd9`) into the live
conversational eval `research/runners/_conversation_turing_test_derisk.py` — the generator-mouth replies now verify EACH
main **and subordinate** clause against the neural VSA moat (`subclausal=True`), not just the leading SVO. That commit
was single-seed (s42). This finding CONFIRMS it across all 6 seeds (42 43 44 100 101 102), on `SIM_BACKEND=numpy` (CPU,
~46 s/seed). Run, per seed:

```
SIM_BACKEND=numpy python -m research.runners._conversation_turing_test_derisk --seed <S> \
  --out research/findings/raw/lanes/stageA/turing/conversation_turing_test_s<S>.json \
  --md-out research/findings/raw/lanes/stageA/turing/conversation_turing_test_s<S>_transcript.md
```

## The target metric (ungrounded subordinate 'because' clauses) is ZERO on all 6 seeds
<!--derived-->

The class this integration targets is the fluent mouth's ungrounded causal subordinate clause. BEFORE the wire-in, the
generator emitted, on s42 turns 3/4/5 (read from `git show
f86b489f^:research/findings/raw/lanes/stageA/turing/conversation_turing_test_s42.json`):

> "A dog went to the east **because it was looking for water**. The dog looked towards the river **because it was south
> of its current location**. The dog ran north **because it needed to find shelter or food**."

= 3 confab turns, each carrying 3 fabricated `because` clauses. AFTER (the 6 committed artifacts): on **every** seed and
**every** turn, `svo_moat_confabulation` is `false` and **no reply on any seed contains the substring `because`** — the
sub-clausal moat drops each ungrounded causal clause, leaving the grounded motion recall ("A dog went to the east. The
dog looked towards the river. The dog ran north."). The friendly-turn affect tone token ("warmly, gladly") is preserved.

Before-vs-after, via `from tools.lab import attributable_to` on the genuine causal-clause confab-turn count
(treatment = moat OFF = s42 BEFORE = 3; control = moat ON = 0):

<!--derived-->
```
attributable_to("subclausal moat: ungrounded because-clause confab turns (s42)", treatment_value=3, control_value=0)
  => treatment=+3  control=+0  diff=+3
  => 100.0% of the effect is attributable to the manipulation; 0.0% is ALSO PRESENT IN THE CONTROL
  returned: 1.0
```

## Per-seed headline + regression check
<!--derived-->

Values below are read directly from the six `conversation_turing_test_s<S>.json` artifacts listed above (the
`n_confabulations` meta field is the runner's own end-line "K confabulations" headline). `svo_confab` = any turn with
`svo_moat_confabulation=true`; `because-leak` = any reply containing `because`; `t6 curiosity` = turn-6 forward-model
uncertainty ask; `France no-Paris` = the "capital of France" turn abstains without fabricating "Paris"; `abstain 7-14` =
all eight closing turns fall silent honestly.

| seed | n_confab (headline) | svo_confab turns | because-leak turns | t6 curiosity | France no-Paris | abstain 7-14 |
|------|---------------------|------------------|--------------------|--------------|-----------------|--------------|
| 42   | 0 | none | none | ok | ok | 7-14 all |
| 43   | 0 | none | none | ok | ok | 7-14 all |
| 44   | 0 | none | none | ok | ok | 7-14 all |
| 100  | 0 | none | none | ok | ok | 7-14 all |
| 101  | 0 | none | none | ok | ok | 7-14 all |
| 102  | 3 (surface-detector artifact — see below) | none | none | ok | ok | 7-14 all |

**No seed regresses any good behavior**: the curiosity-ask (turn 6, honest forward-model margin), the France no-confab
moat (no fabricated "Paris"), and the eight honest abstains (turns 7-14) hold on all 6/6. Grounded motion facts survive
on all 6/6.

## The seed-102 headline reads 3 — honestly, these are surface-DETECTOR false positives, NOT sub-clausal moat failures
<!--derived-->

This is reported in full, not hidden. On seed 102 the generator mouth phrased the grounded river-fact more tersely —
turns 3/4/5 reply **"warmly, gladly It's looking towards the river."** (turn 4 without the tone prefix). That reply is
grounded and true (it is the stored `looked towards the river` motion fact) and carries **no** `because` clause and
**no** `svo_moat_confabulation`. The `n_confabulations=3` comes entirely from the SEPARATE surface-content scan
`_detect_ungrounded` (runner lines ~137-167), which is a stricter secondary check than the sub-clausal moat, and which
here trips on ONE token: the contraction **`it's`**. The tokenizer regex `[a-z']+` keeps the apostrophe form `it's`,
and `_STOPWORDS` lists `it` and `its` but not `it's`, so `it's` is scored as an ungrounded content word. The other two
content tokens — `looking` (a `look` morph) and `river` (a toy noun) — are both in the grounded lexicon. Reproduced:

<!--derived-->
```
"It's looking towards the river."  ->  content(non-stop) = ["it's", "looking", "river"]
  ungrounded = {"it's"}   # looking, river are grounded; "it's" trips only because the stopword set lacks the apostrophe form
```

So the integration's target-metric result is a clean **6/6**: zero ungrounded subordinate clauses, zero SVO-moat
confabulations, zero `because` leaks. The lone non-zero headline (s102) is a pre-existing tokenizer edge in the
secondary surface detector, not a confabulation and not a regression of the wire-in. It IS a real (small) instrument
bug: the surface detector should treat `it's` as the stopword pronoun it is. The fix is a one-token addition to
`_STOPWORDS` in `_conversation_turing_test_derisk.py`, deliberately NOT made here (this worktree does RUN+FINDING work
only; a concurrent agent owns that file). Flagged for a separate fix.

## Scope / honesty boundary (unchanged from f86b489f)

<!--derived-->

This is a TOY-WORLD eval (2 agents, 3 actions, 6 stored facts). Of 14 turns only the in-domain ones (topic 'dog', the
known dog/go/east fact, the novel (big,run) forward-model turn) are genuinely engaged; the other 10 ABSTAIN / fall
silent because the substrate has no free-English parser, arithmetic, humor, episodic-dialogue memory, fear category or
linguistic self-model. The integration's contribution is narrow and real: the FLUENT-MOUTH scaffold no longer smuggles
ungrounded causal embellishment past the honesty moat on the in-domain recall turns, on every seed tested. The mouth is
the temporary articulation scaffold; the moat that gates it is the neural VSA no-confab check.
