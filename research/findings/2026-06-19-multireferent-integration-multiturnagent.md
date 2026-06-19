# Multi-referent biased competition INTEGRATED into the production MultiTurnAgent (2026-06-19)

**Status: WIRED + CI-guarded, default OFF, byte-identical when off.** The multi-referent pronoun-disambiguation
mechanism — WTA **biased competition** (mutual inhibition between held discourse referents + a small CONTENT bias
from the query verb's selectional restriction × candidate animacy) — de-risked GO on 2026-06-19
(`2026-06-19-multireferent-biased-competition-derisk.md`: GO-arm 5/6 seeds, all anti-cheat controls 6/6) and lived
only in the standalone de-risk runner. It is now folded into the production `MultiTurnAgent` behind a default-OFF
`enable_biased_competition` flag — exactly the pattern that landed conversational #1 (attributed + multi-frame folded
into the production composer).

Reuse-by-import; **no `sim/` edit**. CPU/numpy-runnable (`SIM_BACKEND=numpy`).

---

## What was wired (vs deferred)

**WIRED (this integration):**

1. **The validated mechanism is now production-importable code.** `BiasedCompetitionContextBuffer` +
   `content_bias_target` + `resolve_referent` + the `ANIMACY`/`VERB_SELECTS` feature lexicons were extracted
   byte-faithfully from `_phaseB_biased_competition_derisk.py` into `research/runners/biased_competition_buffer.py`
   (single source of truth). The de-risk runner now imports them verbatim; re-running seed 42 post-extraction
   reproduced the validated result identically (GO-arm 1/1, lesion 1/1, moat 1/1, baselines 1/1) — confirming the
   extraction is byte-faithful.

2. **`MultiTurnAgent(enable_biased_competition=False)`** (default OFF). When ON, a pronoun query over **≥2 held
   discourse referents** routes through the biased-competition buffer instead of the plain single-attractor read:
   - The buffer mirrors the held-referent registry (built **only** when the flag is ON).
   - `_resolve(word, query_verb)` — when the mode is on + the word is an anaphor + ≥2 referents are held — calls
     `_resolve_biased(query_verb)`, which (a) abstains (no-confab moat) when the content is silent (the verb has no
     selectional restriction, or 0 / >1 compatible candidates), and (b) otherwise re-presents the held referents,
     biases the content-favored `sel` pool, and returns the **moat-gated WTA winner**.
   - `what_does` / `who_does` / `is_it_true` / `reason_chain` pass their query verb through; `describe` stays on the
     plain path (no query verb to bias with — correct, there is no content signal to steer).

3. **Agent-owned held-referent registry.** The plain `SpikingLoopContextBuffer` does not track which referents were
   introduced, so the held SET (needed to know when ≥2 referents co-occur) is maintained by the agent
   (`self._referent_history`, appended in `_write_referent`, mirroring exactly what is written into the WM loops).
   *(This was a real bug found + fixed during integration: the first wiring keyed `_held_set()` off the WM buffer's
   internals, which were always empty, silently skipping the biased path.)*

4. **CI test** `tests/test_multireferent_biased_competition.py` (5 tests, CPU/numpy) — see below.

**DEFERRED (honest, named):**

- **The full in-loop live-turn integration is wired and exercised** (the capability test drives the real
  `what_does("it","eat")` turn through the biased path → fact answer). What is NOT changed: production demos
  (`multi_turn_conversation_demo.py`) still default to the plain path; flipping a demo default is a separate, trivial
  follow-on if/when multi-referent dialogue is prioritized. The mechanism + flag + capability test are complete.

## The flag-OFF byte-identity confirmation

- Default `enable_biased_competition=False` ⇒ `self.bcw is None` (the biased-competition buffer is **never even
  constructed**), and `_resolve` takes the plain `held_referent()` path unchanged.
- `tests/test_multi_turn_agent.py` (the existing CI: anaphora-resolves, pronoun-cued-multihop, moat-empty-WM) passes
  **verbatim** (3/3) before and after the integration.
- The new test additionally asserts `a.bcw is None` and that single-referent anaphora answers identically with the
  flag off.

## The capability + moat test result

`tests/test_multireferent_biased_competition.py` — **5/5 pass** (CPU/numpy); 8/8 with the existing 3.

| Test | Asserts | Result |
|---|---|---|
| `..._resolves_content_favored_referent` | held {cat=animate, ball=inanimate}: `'it'+'eat' → cat`; full turn `what_does("it","eat") → fish`; **feature-flip** `'it'+'roll' → ball` | PASS |
| `..._content_not_recency` | write-order flip (ball older, cat recent): `'eat'` still → cat (content beats recency) | PASS |
| `..._moat_empty_wm_abstains` | empty WM → `_resolve_biased` and `what_does` both → **None** | PASS |
| `..._moat_content_silent_abstains` | 2 held + a verb with no selectional restriction (`see`) → **None** | PASS |
| `..._flag_off_buffer_not_built_and_anaphora_unchanged` | flag OFF: `bcw is None`; single-referent anaphora answer unchanged | PASS |

**Anti-cheat strength of the capability test:** BOTH `cat` and `ball` are given an `eat` fact
(`cat eat fish`, `ball eat worm`), so the answer is decided by *which referent the content bias resolves to* —
resolving wrongly would return a different, also-non-None answer (`worm`). The test gets `fish`, i.e. it resolved to
`cat` (the content-favored animate) and not to whichever referent merely has a fact.

**The no-confab moat is preserved everywhere** — empty WM and content-silent queries abstain (None), zero
confabulation. The moat was never weakened.

## The documented host-helper shortcut + its learned-map follow-on (BRAIN-BASED-ONLY)

`content_bias_target` (+ the `ANIMACY` / `VERB_SELECTS` feature lexicons) is **host-side**: given the pronoun's
features and the query verb's selectional restriction, it returns which held referent receives the bias current.
This is **FLAGGED in-module for conversion to a learned synaptic feature-compatibility map** per the project's
BRAIN-BASED-ONLY standard. The **win is brain-based** (the spiking WTA competition + selective inhibition +
the Wong-Wang recurrence amplifying a small content asymmetry into a suppressive winner); the content **scoring** is
the host scaffold. The follow-on neuralizes it: a learned map *pronoun-feature population × candidate-feature
population → bias current*, so the bias itself is computed by neurons/synapses. (An honest NEGATIVE on the
*neural-bias* version would itself map what the substrate can compute about agreement.)

## The two named boundaries (NOT in scope here)

These are the de-risk's pre-registered honest boundaries, carried forward unchanged:

1. **Extreme intrinsic-asymmetry (the seed-100 case): ABSTAINS, not a clean win.** When one referent's intrinsic
   attractor is extreme, a *fixed-magnitude* bias occasionally cannot flip the winner — and the failure mode is an
   **abstention (None), not a confabulation** (the moat held through it). The fix is a **content-graded /
   homeostatically-normalized bias** (within the α<1 WTA-stability envelope). The CI test uses the validated
   `seed=42` (the de-risk's clean 3/3 on 42/43/44); it does not claim seed-100 robustness.
2. **The all-compatible-referent case** (two animate candidates of the same number/gender, where agreement is
   *silent* and only finer role/recency cues decide) needs finer cues composed *on top of* the validated
   competition. Not addressed here.

---

## Files

- `research/runners/biased_competition_buffer.py` — the promoted production mechanism (buffer + content-bias helper
  + `resolve_referent` + the feature lexicons; the BRAIN-BASED-ONLY shortcut flag in-module).
- `research/runners/multi_turn_agent.py` — `enable_biased_competition` flag + the `_resolve_biased` routing + the
  agent-owned `_referent_history` registry.
- `research/runners/_phaseB_biased_competition_derisk.py` — the de-risk runner, now importing the production symbols
  (kept working byte-faithfully).
- `tests/test_multireferent_biased_competition.py` — the capability + moat CI guard.

## Commits (on `main`, pushed to both `origin` + `gitea`)

- `c92fb04f` — promote `BiasedCompetitionContextBuffer` to production module (byte-faithful extraction).
- step-2 flag wiring landed in `d2f02d88` (a concurrent session's commit swept the staged working-tree edit into its
  own commit; the content — `enable_biased_competition` + `_resolve_biased` + all method wiring — is intact and
  complete on `main`/both remotes, just under that commit's message rather than the intended one).
- `cc185eef` — fix the held-referent registry (agent-owned), making the biased path actually fire.
- `c7190d75` — the capability + moat CI test.

## Cited

Desimone & Duncan 1995 (biased competition); Wong & Wang 2006 (attractor WTA amplifying a biased input);
Rutishauser-Douglas-Slotine 2011 (the α<1 WTA-stability condition). Catalog: N.19 (gamma binding-by-synchrony FS
mutual inhibition), B-cluster (MSN lateral-inhibition WTA precedent), H.24/H.25 (the navigation `sel`/`commit`
recipe reused). The two prior NEGATIVEs this converts: `2026-06-17-multireferent-disambiguation-NEGATIVE.md`.
