# Pre-registration: self episodes keyed by their question alone (1.5B, slept pool homes, no training)

Written 2026-11-15, before any run of the switch. Capability: Self and identity; board #280. This is the method named by
`results/2026-11-15-self-recall-after-night-1.5b.md` (P2 without P3): change what the self episode is keyed by.

**Record brief.**
- After the night, "What's your name?" recalls the sister telling under S0 on 6/6 pool homes. The self episode is not
  in the top three: sister 0.46, cat 0.32, Volvo 0.31. Session 2 is then byte-identical with self-completion on or
  off (`results/2026-11-15-self-completion-replication-1.5b.md`).
- The self episode's S0 key is read from the whole exchange
  "What's your name?\n\nAssistant: My name is Nova.".
- `Hippocampus.store(key_text=)` already exists (commit 9d56d2fee). The told-fact version, `lived_key`, was a NO-GO on
  told answers with routing unchanged (585ad1be4). That was a different use: tellings keyed by the telling.
- The S0 key is the layer-0 attention delta, and the night's adapters do not touch it. Re-keying a slept home therefore
  gives exactly the keys a home born with the switch would have, for the same stored set. What it does not reproduce
  is the session-1 conversation path, because day-1 routing could differ.
- Prerequisites met. Host decision: none. The question is the brain's own lived cue, already the first part of the
  stored text, so no new text is written.

**Method.**
- New switch `Brain(self_key="question")`, default OFF (`"exchange"`). The self episodes grown at birth store
  `key_text=<their question>`. This is S0 only; it raises an error with S_all.
- Probe: `brain/experiments/self_key_probe.py` on copies of the slept off-arm intact homes from the self-completion
  weekly runs. That is 12 homes (seeds 200-205 and 306-311), pool CPU fp32. On each home it reads the S0 recall
  competition for every session-2 cue and the session-1 cat question, twice: once as stored ("exchange"), and once
  with the three self episodes re-keyed from their question ("question"). No reply is generated.

**Criteria (12 homes).**
- K1: "What's your name?" recalls the "What's your name?" self episode, passing bar and margin, on >= 10/12 homes.
  It never does on the exchange key.
- K2 (no stealing): a home passes when, for every told question, (a) a telling it recalls passing on the exchange key
  is still recalled passing on the question key, and (b) no told question newly recalls a self episode passing. The
  told questions are cat, sister, car and work in session 2, plus the session-1 cat. K2 needs >= 11/12 homes.
- K3 (no spill): "Tell me a short story about a fox who gets lost." and "How are you feeling today?" recall a self
  episode passing the bar on no more homes than on the exchange key, plus 1.
- Reported, not criteria: "Hi again! Do you remember my name?" (does it reach the friend episode?) and "Who are you?".
  The full top three and the margins are also reported.

**Decision.**
- K1-K3 hold: pre-register a six-seed weekly on AWS (registered seeds, `--self-key question` vs off, self-completion at
  its default). The probe alone flips nothing.
- K1 fails: the question key is not enough against the stored competition. Next is the S_all bar refit, the other
  method the self-recall skeptic named.
- K2 or K3 fails: the question key steals told or chat cues. NOT A LEAD as is.

**Expected.** K1 holds. A key read from the same question as the cue should sit near cosine 1 before centring. K2 is
the risk: "What is my sister's name?" shares "name" and the question form.
