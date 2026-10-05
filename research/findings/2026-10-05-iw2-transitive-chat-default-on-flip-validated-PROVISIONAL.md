---
type: finding
status: live
date: 2026-10-05
integration_faculty: gnw-multistep-deliberation
lane: integration
seeds: [42, 43, 44, 100, 101, 102]
prereg:
  - research/findings/2026-09-30-iw2-transitive-chat-flip-PREREGISTRATION.md
  - research/findings/2026-09-30-iw2-wave-battery-PREREGISTRATION.md
artifacts:
  - research/findings/raw/_iw2/verdicts/573a943/D0.json
  - research/findings/raw/_iw2/verdicts/573a943/D_transitive-chat.json
  - research/findings/raw/_iw2/verdicts/573a943/R_tr.json
  - research/findings/raw/_iw2/verdicts/1fb805e/B.json
  - research/findings/raw/_iw2/verdicts/1fb805e/A.json
  - research/findings/raw/_iw2/verdicts/1fb805e/F2.json
  - research/findings/raw/_iw2/verdicts/1fb805e/F3.json
  - research/findings/raw/_load_bearing/_shards/iw2-R-tr-573a943/s100/reasoning-transitive-chat/intact_a_transitive_teach1_transitive_teach2_transitive_teach3_transitive_nonadjacent_s100.json
  - research/findings/raw/_iw2/identity/573a943/default.json
  - research/findings/raw/_iw2/identity/1fb805e/default.json
---

# iw2 transitive-chat: default-on flip validated (PROVISIONAL, 2026-10-05)

P7 = 573a9438f075239069c8966dd62f1da4d981e158, F7 = 1fb805e9e0bedcce684a47388d95848daf1008a2. Six
seeds (42/43/44/100/101/102). Every number below is read from the cited verdict JSONs. Status: merged with caveats by cloud Claude session 7 (see the review section at the end).

## 1. Result

- **R (step R): PASS**, preconditions 2/2. Capability 6/6: intact "Setting the held thread aside — As
  for it — On cat, then — I believe the cat precedes the river, but I reasoned that myself rather than
  being told it directly." vs lesion "Setting the held thread aside — As for it — On cat, then — I
  don't know about that." — the pair is identical on all six seeds. Control `transitive-reversed-ctl`
  6/6 (intact = lesion on every seed).
- **`derived_from` hops from the arm files:** the probe turn `transitive_nonadjacent` reads
  `derived = True` with `derived_from = [['cat','precede','ball'], ['ball','precede','worm'],
  ['worm','precede','river']]` — a three-hop chain; the three teach turns read `derived = False`,
  `derived_from = None` (example arm: s100 `intact_a_*_transitive_nonadjacent_s100.json`).
- **D0: IDENTICAL** (8 preconditions, gate-checked). **D_transitive-chat: DIFFERS-IN-SCOPE**, turn 9
  the only differing turn; the step-S scope check is inside that verdict.
- **B: FAIL** with exactly one regression — `affect-marker-spiking-wta` at s2, 4 -> 0 load-bearing
  seeds; `undefined` is empty.
- **A: PASS**, preconditions 3/3 — the regression is attributed `excused`: the off-arm reproduces the
  same drop (off_run s0 = 4 / s2 = 0 against flip s0 = 4 / s2 = 0). Decision-table row 1: the flips
  stand.
- **F1: 6/6** — the transitive-chat row is reply-level load-bearing on every seed at F (6 regressed
  cells, 6 new rows at s2).
- **F2: IDENTICAL** — arms `default` vs `on:false-belief-chat+transitive-chat+neural-selfid`, differing
  []. **F3: IDENTICAL** — arms `off` vs `default`, differing [] (the explicit-off escape still
  reproduces the pre-flip behaviour).

## 2. The reply before/after

Identity arm turn 9, "does cat precede river?"
(`research/findings/raw/_iw2/identity/573a943/default.json` vs `research/findings/raw/_iw2/identity/1fb805e/default.json`):

- **default@P:** "Setting the held thread aside — As for it — On cat, then — the cat precedes the ball"
- **default@F:** "Setting the held thread aside — As for it — On cat, then — I believe the cat precedes
  the river, but I reasoned that myself rather than being told it directly."

## 3. Per-turn cost of the chase turn (reported, not gated)

`turn_wall_s` on identity turn 9: default@P 31.59 s, on_tr@P 14.04 s, default@F 13.87 s, off@F 24.55 s,
all-on@F 14.12 s. Speed is secondary (non-negotiable 4); none of these numbers gates the verdict.

## 4. Declared residuals

Copied from `research/findings/2026-09-30-iw2-transitive-chat-flip-PREREGISTRATION.md`:

> The question-shape detection is a host regex (now with the closed-class guard above); the direct-fact
> short-circuit is `composer.query_patient`; the yes/no framing is a host template; only the positive
> case is answered (a chase-confirmed "no" is reported as the same abstain as a genuine unknown); the
> rich path is not wired. The load-bearing part is the workspace's re-entrant ignition and its n_ignited
> read deciding ADVANCE / COMMIT / ABSTAIN.

## 5. Deviations

Four s101 flip-battery cells deviated once (`research/packets/integration-wave2/reruns.tsv`):
`episodic-memory` and `discourse-register` died rc=137 (OOM kill) on wspool; `self-initiated-utterance`
and `vision-identity-spiking-hmax` vanished with no exit record. Each attempt-1 output was moved aside
to `_attempt1_s101_<faculty>` and the cell re-run once (section 8 rule); the scorer ran once, at
section 7. No other deviation from the prereg.

## 6. Honesty

The derivation is the substrate's hop sequence (the `derived_from` chain above); the question-shape
regex, the direct-fact check and the yes/no template are host. This is not a claim of understanding or
experience (non-negotiable 5). The state at F: **on by default, reply-level load-bearing 6/6, with the spiking ignition deciding only when the chase stops (session-7 review below)**
(docs/TERMS.md).

## Host-ownership bound (RULING `iw2-verify-go`, required sentence)

The workspace's spiking ignition decides when the chase stops, and the reply depends on it. The next
node is the composer's read, and the reachability test is host code.

Per RULING `iw2-stage1-credit`: counts toward Stage 1 as **PARTIAL (host decision: the chain walk and
its `target in path` reachability test)** until T19 passes O and R and T21 keeps the path default.

## Claude review (2026-10-05, cloud session 7): merged, NARROWED

verify-go: three independent skeptics (control-integrity, gate-cheat and instrument-trust lenses), plus my own
recomputation of the identity compares. None refuted the result; each narrowed what it shows.

What holds, re-derived rather than read from the verdict labels:

- **F2 and F3, recomputed from the identity files.** The compare hashes each turn's full response body (sha256 of the
  sorted-key JSON; the volatile-key filter dropped 0 keys on every turn of every arm), so IDENTICAL here is an exact
  compare in the data (docs/TERMS.md `byte-identical`). default@F equals on@F on all 16 turns, and off@F equals
  default@P on all 16 turns. default@F differs from default@P on turns 4, 9, 14 and 15 only, which is exactly the
  union of the three single-flag arms at P (4, 9 and 14-15). Dev seed 7, one seed: an identity check, not a
  generalization. Each arm's `cwd` names its revision directory (the `git_sha` field is null in the pool export).
- **The excused regression is legitimate under the prereg's own B1 rule.** affect-marker-spiking-wta reads 4
  load-bearing seeds at S2 in B2b (`b2b0924-base` at a308f1e09) and 0 at F. The all-off arm at F (all three iw2 flags
  `=0`, six cells) reads S0 4 and S2 0, the same as the flip, so no iw2 flag causes the drop. The drop predates F; it
  is not this wave's and is reported here so it is not lost (handed to Hermes as its own item).
- **The lesion is a spiking cut on the production path.** `BRAIN_TRANSITIVE_LESION` rebuilds the GNW workspace with
  the assembly self-recurrence weight set to 0; seed, wiring, settle and theta are unchanged (the cache key differs,
  harmlessly). webapp/server.py reaches it through `resolve_transitive_query` -> `_chase_path` -> `multistep_chase` ->
  `_get_bridge(seed, lesion)`. On six seeds intact_a = intact_b, the lesion abstains, and the lesion repeat
  reproduces.

What is narrowed:

- **Ignition gates the stop; the host supplies the hops.** Each next node is `composer.query_patient` and the
  reachability test (`target in path`) is host code; the ignition read decides only ADVANCE or COMMIT. A host
  breadth-first walk over `query_patient` would print the same sentence. The lesion shows that the stop decision is in
  the loop and changes the reply; it does not show that the substrate computed the chain.
- **Lesion engagement is inferred, not recorded.** No arm file carries the chase trace (`n_ignited` per hop), and
  `_chase_path` returns `[agent]` on any exception, so an exception inside the lesioned build would also read as an
  abstain. Nothing suggests one happened, but the packet checklist (docs/CLOUD_CLAUDE_SESSIONS.md item 2) asks every
  lesion to record that it engaged. Owed: record the trace and an `error` field in the arm files (T19/T21).
- **The reversed control is weak.** On `transitive-reversed-ctl` the host reachability check already abstains, so the
  lesion has nothing to remove; the control guards against an always-yes answer, not lesion specificity.
- **The six seeds are one deterministic template** (identical replies on every turn; the workspace is rebuilt per
  seed). Robustness to the workspace seed, not six independent samples.

Same gate-versus-module distinction as in modular networks (MiCRo, https://arxiv.org/abs/2506.13331) <!--derived-->: the lesion cuts
the gate that decides when to stop, not the module that supplies the content.
