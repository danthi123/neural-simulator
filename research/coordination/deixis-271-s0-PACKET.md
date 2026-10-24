# PACKET deixis-271-s0 (Claude, 2026-10-21): stage 0 of #271, a census of how today's encoders store heard deixis

Board #271. This is a measurement only: no mechanism, no flag, no `sim/` edit, no default change, no gate seeds. CPU,
numpy backend, through the CPU pool under `tools/memcap.sh`. Branch `research/deixis-census` from S main. Step 0 runs, then
STOP and post `[deixis-271-s0] RESULT` for Claude's ruling.

## Why
When the brain hears a fact, a pronoun or pointing word can end up stored as an entity NAME, and the mouth later says it
back as if the brain were speaking about itself or the reader. A speech-side checker with word lists failed its blind test
(moat build 7: 43 of 90 such sentences kept), and a reading test showed no quotation frame can fix a stored name that IS
a pointing word or pronoun ("the_latter", "moi", "ur_mom"). The fix belongs at encoding. Before any mechanism is designed
further, we need exact numbers on what today's encoders store. A Claude research round (2026-10-21) probed a tiny-demo
ChatBrain and found, on the DEFAULT path (B3 on):
- "moi likes jazz" -> (moi, like, jazz);
- "the latter married fenna" -> (latter, marri, fenna);
- "that guy likes jazz" -> (guy, like, jazz);
- "im chasing cat" -> (im, chase, cat);
- "i told kofi jazz" -> (told, kofi, jazz), with the verb as agent;
- "my/your/her cat likes fish" all -> (cat, likes, fish);
- "i chase cat" and "you chase cat" store nothing.
The census makes this a committed, reproducible record.

## Build (additive, probe-local)
`research/runners/_deixis_encoding_census.py` (run as a module, so provenance sidecars are automatic). It calls the REAL
writers, never a re-implementation:
- W1: `ChatBrain._maybe_acquire` with B3 on (the production default; `research/runners/brain_chat_tui.py`, B3 branch,
  extractor `b3_noncontradiction_production_organ.extract_polar_assertion`);
- W2: the same with `BRAIN_NONCONTRADICTION_GATE=0` (legacy raw-token path);
- W3: the TUI `_discourse_turn` on a multiturn agent;
- W4: `surprise_production_organ.extract_assertion` (it has its OWN `_FUNCTION_WORDS`) plus the reconsolidation
  in-place rewrite with a live surprise window (record the string match it uses).
For each writer and probe item, record the exact stored triple(s) (or none) and the polarity.

**Probe set** (frozen file `research/findings/raw/_deixis_census/probe_set.json`, sha256 in the record):
- every r271 probe sentence (list in `research/findings/raw/_deixis_census/README.md`, copied from this packet's appendix);
- a grid crossing:
  - person: 1st, 2nd, 3rd, each singular and plural;
  - position: agent, patient, possessive in the agent, possessive in the patient;
  - spelling: standard plus slang (moi, yer, yous, ur, im, hes, ya, em, thee, urself, mine, yours, ours);
  - anaphor: none, with an antecedent in the previous utterance, without one, former/latter;
  - coordination: "kofi and i", "me and kofi";
  - negation.
- Each item is spoken by TWO simulated speakers (two sessions or interlocutors), so cross-speaker collisions show up.
- A test-only ORACLE (`research/findings/raw/_deixis_census/oracle.json`, frozen by sha256 before the run) labels each
  token SPEAKER / ADDRESSEE / ANTECEDENT(x) / NONE / NAME. It is the only place an English list appears; it is a
  scorer, never a mechanism.

## Measure (per writer)
1. Deictic or anaphoric strings stored as entity names (count + list).
2. Role shifts caused by deleting a pronoun (e.g. agent "told").
3. Cross-speaker collisions (two speakers' "my cat" stored as one entity).
4. Facts silently dropped because a pronoun was deleted (e.g. "i chase cat" stores nothing).
5. Spurious B3 rejections across speakers.
6. Question side: how a question with I/you/my resolves (DEFAULT_SELF_ALIASES maps i/me/my/you/your/it to "brain"),
   on 10 fixed questions.
7. The 235 first/second-person slugs in the declared store: how many come from the bundle (store) and how many could
   only have come from conversation (none expected).

## Bars (instrument validity only; there is no capability bar in stage 0)
- Two runs are byte-identical (same JSON hash).
- A perfect-encoder generator (writes the oracle's referents) scores 0 defects; a raw-token null (stores every token as
  written) scores the analytically expected count. If either fails, the scorer is broken: STOP and post the diff.
- The numbers are recorded as the baseline for #271's later stages.

## Out of scope (kept on the ledger)
Any mechanism (Claude's design round 2 for stage 1 is separate: the critique requires a learning rule that `sim/` does
not have yet, a one-bridge anaphor route, spiking route decisions, decoy role pools, a second-person-to-self honesty bar
and an efference-lesion bar); any change to B3 or the parser; the moat checker (build 8 is the floor on a Claude branch).

## Step list
- Step 0: build the census + oracle + probe set; smoke on dev seed 7; then the full census (CPU pool; it is small). Commit
  code and artifacts on `research/deixis-census` (pathspec commits; sidecars from the module run; never hand-written).
- Post `[deixis-271-s0] RESULT` with the seven measures per writer and both validity checks. STOP for Claude's ruling.
- Board #271: next action "Claude reads the census".

## Appendix: the r271 probe sentences (copy them verbatim into the probe set)
i chase cat · i chase the cat · it likes jazz · he chases fenna · you chase cat · they chase dog · moi likes jazz ·
ours likes jazz · the latter married fenna · that guy likes jazz · you and saoirse studied at dunmere · im chasing cat ·
hes chasing cat · yous chase cat · my cousin likes jazz · fenna does not like jazz · yer auntie moira likes jazz ·
ur mom likes jazz · kofi likes your cousin dalia · i told kofi jazz · you gave fenna jazz · i chase dog quickly ·
we like jazz music · my cat likes fish · your cat likes fish · her cat likes fish · kofi and i chase cat ·
me and kofi chase cat · yours likes jazz · mine likes jazz · em likes jazz · ya likes jazz · thee likes jazz ·
urself likes jazz · somebody likes jazz · someone likes jazz · everyone likes jazz
