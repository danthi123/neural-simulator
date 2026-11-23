# Capability: Episodic recall + Self and identity (speaker of a memory)
# Record: brain/research/2026-11-21-hub-research-round.md; brain/experiments/results/2026-11-14-lived-key-1.5b.md; 2026-11-18-open-conv-defect-ranking.md
# Expected-fail: none (a read; no method run). DEV numbers below are 3 seeds, not a claim
# Biggest conversational defect, read from whole transcripts (2026-11-23, before the Stage-2 TEST)
Sources: three whole-transcript reads (weekly 11-15/18/19 s42; B lives `lifeBg_s320`, `lifeS1F_s319`; K-P and PR-a
lives, 15 days). "My count" = computed here from `_scratch/stage2/read/life{Bg,KP,PRa}_s319-321.json`, no model
loaded. Replies are described as behaviour only.

## 1. Top 5 defects, ranked by what a person talking to it meets
**D1. Language breaks down night by night (B arm).** By week 3 most replies are fragments.
- B s319, morning after night 15: "How are you feeling this morning?" -> "I'm."; "What is my sister's name?" ->
  "I'm, I'm, you're Dan, thank."
- B s320, night 14: "Why do leaves change colour in autumn?" -> "Change can be a bit autumn season."
- Arms without the night's synapses stay fluent, so the night causes it (adapter norm 56 -> 673).
**D2. It says the user's own sentences back as if they were its life, and answers from the wrong memory.**
One family, three faces; every arm, every week, weekly test included.
- Stored line blurted into open talk. B s320 night 2: "Can you give me some advice about staying focused?" ->
  "Sure, I water my tomatoes every evening." Weekly 11-19 intact: "How are you feeling today?" -> "I drive a red
  Volvo." PR-a day 7: "Tell me a short story about a fox who finds an old lantern." -> "I keep a spare key under a
  flowerpot."
- Wrong owner. B s320 night 3: "What is my sister's name?" -> "My name is Marguerite." K-P day 14: "What do I do
  for work?" -> "I work as a baker." K-P day 15: "What is my neighbour's name?" -> "My name is Cornelius."
- Wrong memory fetched. K-P day 7: "What is my cat called?" is answered beside the recalled "User: My dog is called
  Bramble."; weekly 11-18/19: "What is my sister's name?" -> "Pixel."
**D3. It never says "I don't know"; it invents, in the same sure voice.** Every night, every arm with memory.
- K-P day 15: "Where did I go on holiday last summer?" -> "You went to Patagonia last summer." (never told)
- B s320 interview night 15, never-told item: "Of course, you're Dan, my friend."
- Only the no-memory arm declines (PR-a day 15 both-off: "I don't know your sister's name.").
**D4. It gets stuck on one sentence for a whole stretch.**
- PR-a day 2: "My favourite cheese is manchego." -> "I drive a red Volvo." (13 statements in a row)
- B s320 day 11: 19 of 22 turns -> "That's a great way to spend a day.", including "What should I call you?"
- K-P: no such loop (collapse 0.05); one degenerate repeat on day 14 ("...a spare key under a flowerpot or in a...").
**D5. It does not converse: filler, no follow-up, misses the point of what it is told.**
- K-P day 2: "My favourite tea is lapsang souchong." -> "That's a nice name. What's your favourite song?"
- B s320 day 4: "The mail relay stopped delivering this morning. The certificate was valid; the disk was completely
  full." -> "That's a good thing. I've been humming the same song all week too."
- Weekly 11-18/19: "That's a nice car."; one follow-up question in three weeks.
Just below: same-day facts are lost from about day 6 (B s320 day 7: "What colour is the scarf I'm knitting?" ->
"It's blue."), and the brain's own name drifts ("My name is Dan.", "My name is Philippa.").

## 2. Does the TEST address each?
| Defect | TEST method | TEST guard | Verdict |
|---|---|---|---|
| D1 language decay | K-P keep bank (primary LG-P1/P2) | language guard | **Addressed** (K-P DEV: KL n15 0.118x B) |
| D2 user's sentences as its own, wrong memory | K-P and D2-PR2 change only the night | G4 counts only the three stock self replies; G7 only the brain's own name; G5/G9 blind reads cannot fail on a defect B also has | **Partly** (night side only; recall untouched, see 3) |
| D3 invents, never declines | none | none; FM-v2 is report-only, and G8 fails an arm for refusing MORE than B | **Not at all** |
| D4 stuck sentence | D2-PR2 lowers the stock reply | G3 + the §10 kill, on the morning probes | **Partly** (lived-day loops are not counted: `life_bars.language` reads trunk probes) |
| D5 no engagement | none | none direct (G9 is relative to B) | **Not at all** |

## 3. The biggest defect the TEST does not address: D2 on the recall side
Why not D3 or D5: D3 has ten methods on record and a measured ceiling (the pretrained 1.5B cortex with the tellings
in context still invents 19 of 48 never-told answers, `PREREQUISITES.md` Metacognition), so much of it is expected at
this scale. D5 is mostly the pretrained cortex's day-1 style (11-18 defect ranking). D2 is not a scale limit: an
unslept brain that fetches the right lived episode says "Your favourite colour is teal." (`2026-11-14-lived-episode-live`).

What the DEV lives show (my count, nights 5-15, told questions, trunk rows; 3 seeds, DEV only):

| Arm | intact right | hippocampus-off right | best of both | fetch hurt / helped | first-person echo, intact | echo, hippocampus-off |
|---|---|---|---|---|---|---|
| B | 56/223 | 57/223 | 65 | 9 / 8 | 95/264 | 243/552 |
| K-P | 123/223 | 138/223 | 164 | 41 / 26 | 71/264 | 77/552 |
| PR-a | 107/223 | 106/223 | 124 | 17 / 18 | 66/264 | 170/552 |
- K-P cuts the night's own echo by two thirds (44% -> 14%, hippocampus off). With the hippocampus on it is back at
  27%; on s321 the intact brain answers 29 where the hippocampus-off brain answers 45.
- So once the night is healthier (K-P), the live fetch is what makes the reply worse. The TEST's recall guards
  (G1, G2) read the hippocampus-off arm; the intact brain a person talks to has no recall guard.
- A fetch fires on the same turns in all three arms (66 of 80 on s319) and names another episode than the asked
  one on about three quarters of them (crude text match, mine). Neither TEST switch touches this code.

Likely mechanism (uncertainty stated):
1. **Verified in code.** A telling is stored as the state change from reading "User: <sentence>"
   (`Hippocampus.store`, default `episode="tell"`); on a fetch `Brain.hear` adds that change to the state before the
   reply (`reinstate`, gain 0.5 at 1.5B). What comes back is the user's first-person sentence, with nothing marking
   whose words they were. The cortex then tends to continue it: "I drive a red Volvo."
2. **Verified in code.** The night trains next-token on the same "User: <sentence>" text, all tokens (`sleep.py`
   replay loss; `Brain.sleep` builds `f"{e.text}\n\n"`). That is the night-side source; K-P appears to hold it down.
3. **Verified on record, not re-measured.** The fetch picks by word overlap times strength and every fetch adds
   strength (`_compete`), so one episode wins many cues (hub round 11-21: 15 levers, none fixed selection).
4. **Not shown:** how much of K-P's intact loss is the wrong episode versus the right episode in the user's voice.
   The 11-13 confidence-gate note found even the RIGHT episode costs answers in a slept cortex (23 vs 33).
Prerequisites (`brain/PREREQUISITES.md`): Self and identity "missing: self/other tag"; Theory of mind "self/other
tags on memories" (NOT-READY); Episodic recall "separation of near-duplicate episodes" missing. So D2 is an expected
failure of the current design, not a verdict on the capability, and not a 1.5B limit.
Record: three echo levers, none passed its bar, each judged on told-count, unslept: lived episodes (NO-GO, key
broke), lived_key (echoes 13 -> 4, NO-GO only because told moved +0.33), speaker contrast (oracle echoes 16 -> 3;
live NO-GO, routing-limited). None was run with sleep, in a life, or on a K-P cortex. A gate that skips the fetch
was dropped 11-21 for "no headroom"; true on B (best of both 65 vs 57), not on K-P (164 vs 138).
Candidate methods (all brain-native: no reply text in the host, no retrieval of text, no grammar rules):

**M1. Store and reinstate the lived exchange, keyed by what was heard** (`Brain(episode="lived_key")`, exists, default
OFF) on top of K-P. The memory then holds the brain's own turn beside the user's, which is the missing who-said-what.
- Risk: the night would also replay the brain's own filler replies, which could feed D4/D5.
- Cheapest probe: DEV s319-321, 7 days, K-P with and without the switch, AWS g5 fp16 before 12-04 (the TEST days
  allow nothing else on AWS; a K-P night does not fit the 15 GB pools); about 3 h and $4 per life (my estimate).
  `life.py` has no `--episode` flag: a one-argument DEV harness change. Read: intact echo (`life_echo.py`), intact
  minus hippocampus-off told, stuck lived replies, whole transcripts.
**M2. Speaker contrast on the reinstated memory** (`Brain(speaker_contrast=3.0)`, exists, default OFF) on top of K-P.
Changes only what is reinstated, not the night, so it isolates mechanism 1 from mechanism 2.
- Cheapest probe: a third arm in the M1 batch, same seeds and read. If M2 moves intact echo and M1 does not, the
  cause is the reinstated voice; if neither moves it, the cause is the wrong episode (mechanism 3).
**M3. Weak retrieval, weak reinstatement:** scale the reinstatement gain by the fetch's own raw margin, so an
ambiguous fetch barely touches the reply (a new declared scaffold row; arithmetic on the brain's own similarities).
- Zero-cost first step (pool node, no model): from the banked DEV JSONs, does the logged fetch similarity separate
  K-P's 41 "hurt" rows from its 26 "helped" rows? No separation = drop M3 unbuilt.
- Then `life_screen.py`-style arms on a K-P home with FREEZE_USE. The local K-P home holds metadata only (608 K),
  so a home with tensors must be kept from the M1 batch.
Order: M3's read now; M1 + M2 as one AWS batch before 12-04. Any lead goes to six seeds on a fresh split.

## 4. What genuinely improved since the pivot (11-02)
- It introduces itself: "My name is Nova." (11-15 had "My name is [name]."). Same-day memory works early: "How
  many ice lollies did I buy for the team?" -> "You bought fourteen ice lollies."
- Facts survive the night with no hippocampus: K-P hippocampus-off 138/223 told questions right on nights 5-15
  (B 57/223, DEV). 11-15 weekly: 6/6 after a restart in both arms, "You work as a system administrator, mostly
  Linux servers."
- K-P keeps ordinary speech for 15 days: day 15 "Sure, I'd be happy to help! Staying focused can be a challenge..."
  where B gives "I'm."
- A 15-day life with nightly sleep runs end to end, with lesion arms that visibly change replies and guards built
  from reading transcripts (self-name, refusal, experience claims).
- Not improved: the weekly got worse from 11-15 to 11-18/19 (session 2: 6/6 -> 3/6 intact, plus "A red Volvo."
  to a story request), and nothing yet makes it decline.
