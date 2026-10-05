---
type: finding
status: provisional
date: 2026-10-05
integration_faculty: content-selection
lane: integration
seeds: [42, 43, 44, 100, 101, 102]
prereg:
  - research/findings/2026-09-30-iw2-neural-selfid-flip-PREREGISTRATION.md
  - research/findings/2026-09-30-iw2-wave-battery-PREREGISTRATION.md
artifacts:
  - research/findings/raw/_iw2/verdicts/573a943/D0.json
  - research/findings/raw/_iw2/verdicts/573a943/D_neural-selfid.json
  - research/findings/raw/_iw2/verdicts/573a943/R_si.json
  - research/findings/raw/_iw2/verdicts/1fb805e/B.json
  - research/findings/raw/_iw2/verdicts/1fb805e/A.json
  - research/findings/raw/_iw2/verdicts/1fb805e/F2.json
  - research/findings/raw/_iw2/verdicts/1fb805e/F3.json
  - research/findings/raw/_iw2/identity/573a943/default.json
  - research/findings/raw/_iw2/identity/1fb805e/default.json
---

# iw2 neural-selfid: default-on flip validated (PROVISIONAL, 2026-10-05)

P7 = 573a9438f075239069c8966dd62f1da4d981e158, F7 = 1fb805e9e0bedcce684a47388d95848daf1008a2. Six
seeds (42/43/44/100/101/102). Every number below is read from the cited verdict JSONs. Status:
provisional pending adversarial review and merge (NEEDS [iw2], 2026-10-05).

## 1. Result

- **R (step R): PASS**, preconditions 2/2. Capability 6/6: intact "the brain uses the spikes" vs
  lesion "I don't know about that." — the pair is identical on all six seeds. Control
  `selfid-nofact-ctl` 6/6 (intact = lesion on every seed).
- **D0: IDENTICAL** (8 preconditions, gate-checked). **D_neural-selfid: DIFFERS-IN-SCOPE**, turns 14
  and 15 the only differing turns; the step-S scope check is inside that verdict.
- **B: FAIL** with exactly one regression — `affect-marker-spiking-wta` at s2, 4 -> 0 load-bearing
  seeds; `undefined` is empty. The B1 watch-list faculties named in the packet HANDOFF
  (`content-selection`, `semantic-recall`, `moat-verify`, `spiking-anaphor`, `wm-binding-advanced`) did
  not regress.
- **A: PASS**, preconditions 3/3 — the regression is attributed `excused`: the off-arm reproduces the
  same drop (off_run s0 = 4 / s2 = 0 against flip s0 = 4 / s2 = 0). Decision-table row 1: the flips
  stand.
- **F1: 6/6** — the neural-selfid row is reply-level load-bearing on every seed at F (6 regressed
  cells, 6 new rows at s2).
- **F2: IDENTICAL** — arms `default` vs `on:false-belief-chat+transitive-chat+neural-selfid`, differing
  []. **F3: IDENTICAL** — arms `off` vs `default`, differing [] (the explicit-off escape still
  reproduces the pre-flip behaviour).

## 2. Host-decision share on the self/identity turns

Identity arms, script turns 14-16 (turn fields 13-15): "what do you use?", "what are you?", "who are
you?" (`research/findings/raw/_iw2/identity/573a943/default.json` vs
`research/findings/raw/_iw2/identity/1fb805e/default.json`). The `source` field reads
`tiny-demo +LTM` on all three turns in both arms; the discriminating fields are `abstained` and
`recalled_svo`:

| turn | message | default@P (router answers) | default@F (substrate answers) |
|---|---|---|---|
| 14 | what do you use? | "the brain uses the spikes", recalled_svo=[brain,use,spikes], abstained=false | same text, same recall, abstained=false (sha identical across pins) |
| 15 | what are you? | "the brain uses the spikes", recalled_svo=[brain,use,spikes], abstained=false | abstained=true, recalled_svo=null |
| 16 | who are you? | "the brain uses the spikes", recalled_svo=[brain,use,spikes], abstained=false | abstained=true, recalled_svo=null |

At P the router's keyword match answered all three with the first brain fact. At F turns 15-16 go down
the substrate route, find no (brain, isa, X) fact, and abstain honestly; turn 14 keeps its answer from
the substrate parse (`recalled_svo` present, byte-identical reply across pins). iw2-selfid's turn
condition holds: turns 15-16 read `abstained` true at F (and in the on-arm at P).

## 3. Declared residuals

Copied from `research/findings/2026-09-30-iw2-neural-selfid-flip-PREREGISTRATION.md`:

> Class (b), the bare identity question, is not the neural parser: it is the same host regex
> comprehension helper 'what is X?' already uses, with the recall itself on the substrate. The
> self-alias resolution ('you' -> 'brain') mirrors the router's own `_resolve_self`. The
> noisy-anaphora fallback stays host-owned (ANAPHORA_ABSTAIN off). Default-on, 'it' is also a self-alias:
> a bare "it" that the discourse WM does not substitute resolves to the brain itself on the substrate
> path (the router did the same on the host path); step S plays one ordinary anaphora turn to check that
> a held referent still wins.

## 4. Deviations

- **The two bare identity questions now abstain under default-on** ("what are you?", "who are you?"):
  the flip trades the router's keyword answer to a *different* question for an honest abstain on these
  two (the prereg's recorded pre-gate behaviour change; RULING `iw2-selfid`). Step D reports both turns
  differing, in scope.
- Four s101 flip-battery cells deviated once (`research/packets/integration-wave2/reruns.tsv`):
  `episodic-memory` and `discourse-register` died rc=137 (OOM kill) on wspool; `self-initiated-utterance`
  and `vision-identity-spiking-hmax` vanished with no exit record. Each attempt-1 output was moved aside
  to `_attempt1_s101_<faculty>` and the cell re-run once (section 8 rule); the scorer ran once, at
  section 7.

## 5. Honesty

The brain's parser comprehends a question about itself — a functional read-out of which route answered
and whether a recall fired. This is not a claim of self-awareness (non-negotiable 5). The state at F:
**on by default, reply-level load-bearing 6/6** (docs/TERMS.md).

## Host-ownership bound (RULING `iw2-verify-go`, required sentence)

This is the content-selection parser, the same positional role map, reaching self questions. The real
change is that the host router no longer answers them. Count this as a reach of the content-selection
mechanism, not a new mechanism.

Per RULING `iw2-stage1-credit`: counts toward Stage 1 as **PARTIAL (host decision: the positional role
map)** until T20 passes O and R and T21 keeps the path default.
