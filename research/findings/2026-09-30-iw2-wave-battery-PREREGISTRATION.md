---
type: preregistration
status: preregistered
date: 2026-09-30
lane: integration (owner 2026-09-29, integration-first; docs/plans/2026-09-29-acceleration-report.md M1)
mechanism: integration wave 2 -- the shared steps of three default-flip packets (false-belief-chat, transitive-chat,
  neural-selfid): the candidate ranking, a pre-flip identity and scope check, a reply-level gate with a permuted
  control per packet, a flip applied by a script, and a production-default no-regression battery at the flipped
  revision F against the reference battery B2b, with attribution re-runs
seeds: [42, 43, 44, 100, 101, 102]
backend: numpy (every gate cell; the scorer's cell-validity rule 3 requires it for comparability with B2b)
verdict: PRE-REGISTRATION only. Filed before any gate-seed job exists. No result is claimed here. Dev seed 7 runs are
  reported in the packet HANDOFFs and carry no weight.
---

# Integration wave 2: selection, shared steps and the no-regression battery (pre-registration)

## Why

The owner's unit of progress (2026-09-29) is a faculty that is on by default and whose lesion changes the reply. Wave
1 (branch `research/integration-wave1`) packaged three such flips. This wave packages three more in the same shape,
with the wave-1 review's lessons built in from the start (named in "Lessons applied" below). The flip bar (memory
`feedback_flip_validated_fixes_without_waiting`, 2026-09-23) has four legs: a six-seed GO, a sound adversarial review,
no regression in a combined battery with the change on, and a production-default validation run. Each packet prereg
fixes its own reply-level gate (step R); this document fixes what the three share, scored by one committed scorer
(`tools/iw2_score.py`).

Packet preregs: `2026-09-30-iw2-false-belief-chat-flip-PREREGISTRATION.md`,
`2026-09-30-iw2-transitive-chat-flip-PREREGISTRATION.md`, `2026-09-30-iw2-neural-selfid-flip-PREREGISTRATION.md`.

## Selection and ranking (recorded before any gate job)

Pool screened: every mechanism in `docs/PRODUCTION_INTEGRATION_LEDGER.yaml`'s rows (or a default-off sub-flag of one)
with a six-seed GO that is not retracted (`docs/RETRACTED.md`) and not on by default at main f5e3732fe. Excluded by
the task: wave 1's three (in-loop-learning's D6 write, wm-binding-advanced's focus bind, worldmodel-forward's C1
edge) and anything Stage 1 C1 covers (the category-inheritance route: rung-3c's member->context pathway).

Scores are 0-3 per axis: (a) the likelihood that the packet's lesion changes the REPLY, (b) readiness (a flag, a
reader that can carry an anchor, a lesion knob, a reply-level probe row, handler-level evidence), (c) relevance to
the Stage 1 or Stage 2 exit criteria (Stage 1: C1-C4 of `2026-09-29-stage1-exit-gate-PREREGISTRATION.md`; Stage 2:
the master roadmap's "grammatical grounded conversation; inference on held-out; curiosity ask-rate; discrete-emotion
read-out; private-to-inner-speech chain"). Total = 3a + 2b + c; ties broken by (a), then (b), then (c).

| rank | candidate (flag) | a | b | c | total | decision |
|---|---|---|---|---|---|---|
| 1 | false-belief chat (`BRAIN_FALSE_BELIEF_CHAT`) | 3: the reply names the stale location; the lesion (witnessing gate forced open) names the true one | 3: wired, reply template, LBF row with its own turn, seed-7 organ smoke + byte-identity-off GO at the post-review commit (`2026-09-25-tom-false-belief-chat-wire-seed7-smoke-GO.md`) | 1: a Stage 3 exit criterion; inference about another's belief | 16 | chosen |
| 2 | transitive yes/no chat route (`BRAIN_TRANSITIVE_CHAT`) | 3: a derived yes vs an honest abstain under the keystone's own verified lesion | 2: wired; the row existed but its turns were unreachable (PARKED); the real-handler seed-7 run was stopped before completion (`2026-09-24-reasoning-transitive-chat-PREREGISTRATION.md`) | 3: Stage 2 "inference on held-out" names transitive inference | 16 | chosen |
| 3 | self/identity comprehension on the brain (`BRAIN_NEURAL_SELFID`, alone) | 2: the parse lesion collapses the self-factual answer 6/6 at organ level (`2026-09-05-rank13-selfid-anaphora-scaffold-derisk-GO-6of6.md`); handler level not yet shown | 2: anchor, reader and lesion knob exist; the 2026-09-05 pair flip was NO-GO, attributed to the OTHER flag of the pair (ANAPHORA_ABSTAIN, seed 44) after SELFID's own seed-43 defect was fixed (`...-CORRECTED-DIAGNOSIS.md`) | 1: the self as a topic of grounded conversation; strongly relevant to the current phase (host-decision share) | 11 | chosen |
| 4 | biased competition, shared inhibitory pool (`competition_mode="shared_pool"`) | 2: changes 3-referent pronoun turns | 1: no production flag; at the shipped brain the bias source is the D3 focus fallback and the row's lesion is `proposed` (hollow) | 2 | 10 | next wave |
| 5 | metacog graded 3-band marker | 2 | 0: a measurement de-risk; no production wire | 1 | 7 | next wave (needs a wire) |
| 6 | pmem operating-point stabilizer (`BRAIN_PMEM_OP_STABILIZER`) | 0: hardens a margin; a calibrated table covers gate seeds only | 3 | 0 | 6 | not chosen |
| 7 | DA encoding sleep/NREM trigger (`BRAIN_DA_ENCODING_SLEEP_TRIGGER`) | 0: the gain colours trace strength, never which fact | 3 | 0 | 6 | not chosen |
| 8 | gendered-pronoun anaphor detection | 1 | 0: detection only; the agreement check is unbuilt | 2 | 5 | not chosen |
| 9 | awake/night replay completion (`BRAIN_AWAKE_REPLAY_COMPLETION`) | 0: necessary on 1 of 6 seeds | 1 | 1 | 4 | not chosen |
| - | DA tag-capture + sleep-replay capture | - | - | - | - | excluded: the pair review found a fact told 4 h before sleep is not kept on 6/6 (a flip would regress today's default) |
| - | learned-referent lexicon; frame-junction lexicon | - | - | - | - | excluded: admits wh-words; the closed-class lane reads NOT READY |
| - | affect learned vocabulary; open-ended gated turn; learned segmenter | - | - | - | - | excluded: NO-GO; NOT-GO/UNDEFINED; 5/6 |
| - | SlotBinder composer; affect-marker SETTLE; 11-organ one-pool flip | - | - | - | - | excluded: scoped to 32 of 404 facts; owner sign-off required; held on a DA-mode regression |

## Revisions

- **P** (pre-flip pin): the head of `research/integration-wave2` when the shared HANDOFF provisions it, and only if
  `origin/main` is an ancestor (the provisioner's stale-source guard); if main has moved, the HANDOFF merges it first
  and P is the merge commit. Every wave-2 anchor reads False at P.
- **F** (flip pin): the commit made by `tools/integration_wave2_flip.py apply <packets>`, for exactly the packets whose
  steps R, D and S passed at P. Nothing else changes between P and F. After a revert, **F'** is the next such commit.

**Interaction with wave 1.** Whatever main holds at P is the production default this wave measures against. If wave
1's flips merged first, they are part of the default at P and F; the wave-2 guard never checks wave-1 anchors, and
attribution (step A) forces only wave-2 flags off, so a drop wave 1 caused is excused here and stays wave 1's.

## Compute lane

numpy on the CPU pool (`tools/pool_queue.sh`), every line under `tools/mem_ok.sh` + `tools/memcap.sh`, declared
`mem_gb` <= 11 and a `max_age_h` in its `--checked` reason. Not the GPU queue: the registered cell-validity rules
(`tools/b2d_score.py` rule 3) require `SIM_BACKEND=numpy` so the flip battery compares like for like with B2b, and
`tools/lb_shard.py` emits numpy lines only.

## Step D: identity and scope at P, dev seed 7 (real handler, `research/runners/iw2_default_identity.py`)

Five fresh-process arms of one 16-turn script in one session (three segments, each opening with `reset` so one brain
is alive at a time; each segment plays its SCOPE turns first): `default`, `off`, `on:false-belief-chat`,
`on:transitive-chat`, `on:neural-selfid`. Per-turn sha256 of the canonical response body (timing keys dropped).

- **D0 (byte-identical-off).** `default` equals `off` on all 16 turns. Required for every packet.
- **D_p (the script reaches p's flag).** `on:p` differs from `default` on at least one turn tagged p; none is
  VACUOUS (UNDEFINED for p, never a pass).
- **S_p (scope).** `on:p` equals `default` on every turn NOT tagged p (the scope turns and the other packets' turns);
  a difference is OUT-OF-SCOPE: the flag changes a reply it was never meant to reach (the lesson of the wave-1
  lexicon screening). Scored together with D_p by `--compare ... --expect differs --packet p`.

## Step R: reply level, per packet, at P (six gate seeds; each packet prereg)

The packet's capability row AND its permuted control row (`research/runners/lbf_rows/iw2_controls.py`), both at
`--repeats 2`, in one job file. A capability cell counts only if DEFINED (b2d rules), S2 `regressed`, and the intact
reply is neither degenerate (empty, absent, abstaining, an echo of the probe: UNDEFINED) nor wrong (lacks the
packet's content token: `intact-wrong`, a FAIL). A control cell holds only if DEFINED, its intact reply meets the
control's prediction, and the lesion leaves its reply unchanged (S2 `pass`). **R_p PASS** = capability 6/6 and
control 6/6; any not-DEFINED cell (after the re-run rule) = UNDEFINED; otherwise FAIL.

## Step B: the no-regression battery, production default at F

One sharded load-bearing battery at F, `iw2-flip-<F7>`: `tools/lb_shard.py jobs --no-fixes` (the adequate probe
set, no fix flag, no BRAIN_* flag), numpy, six gate seeds, one shard per registry faculty and seed, every line
prefixed by `tools/assert_iw2_defaults.py --expect-on=<flipped>` and the node preflight `tools/assert_iw2_node.py`.
At F the lifted rows (`tom-false-belief`, `reasoning-transitive-chat`, `neural-selfid`) are in the registry. The
reference is B2b (`b2b0924-base` at a308f1e09): 210 of 216 cells DEFINED under this scorer (wave-1 ruling
2026-09-30); a reference row that is not DEFINED is reported by name and never counted either way.

- **B1 (no regression, both levels).** For every faculty in both batteries, its number of load-bearing seeds at F is
  not lower than in B2b, at S0 and at S2 -- or, for a faculty that is lower, its attribution run `iw2-alloff-<F7>`
  (its six cells at F with every flipped packet's flag `=0`) is also at or below the F count at both levels.
- **B2 (completeness).** Every coverable cell of `iw2-flip-<F7>` is DEFINED after the re-run rule.
- **F1 (the new rows at the production default).** Each flipped packet's row is reply-level load-bearing 6/6 in
  `iw2-flip-<F7>`, read with step R's guards (degenerate or wrong intact never counts).

## Step F: production-default identity at F, dev seed 7

Arms at F: `default`, `off`, `on:<all flipped packets joined by +>`.
- **F2 (the default IS the capability).** `default@F` equals `on:<flipped>@F` on all 16 turns.
- **F3 (the escape still works).** `off@F` equals `default@P` (step D) on all 16 turns.

## Decision table (applied by the HANDOFF, no judgment)

| outcome | action |
|---|---|
| packet p: R_p PASS, D0 PASS, D_p and S_p PASS (DIFFERS-IN-SCOPE) | p enters the flip at F |
| R_p FAIL (a DEFINED capability seed not load-bearing, an `intact-wrong` seed, or a violated control) | p is not flipped; NO-GO finding (provisional) for p's reply-level integration |
| S_p FAIL (OUT-OF-SCOPE) | p is not flipped; NEEDS-CLAUDE (a scope fix is a code change) |
| R_p, D_p or S_p UNDEFINED after the re-run rule | p is not flipped; NEEDS-CLAUDE |
| D0 FAIL or UNDEFINED | no packet is flipped; NEEDS-CLAUDE (an anchor edit changed the default path) |
| B1, B2, F1, F2, F3 all PASS | the flips stand on the branch; NEEDS-CLAUDE: adversarial review + merge |
| faculty X lower than B2b and its all-off run is higher than F at either level | bisect: X's six cells at F with ONE packet's flag `=0` per run; the one packet whose run is higher than F is reverted, F' committed, steps B and F re-run at F' |
| the bisect shows no single packet, or several | revert every flipped packet; NEEDS-CLAUDE (an interaction) |
| F1 FAIL for p | revert p; re-run steps B and F at F' |
| F2 or F3 FAIL | revert every flipped packet; NEEDS-CLAUDE |
| any criterion UNDEFINED after the re-run rule | nothing merges; NEEDS-CLAUDE |

## UNDEFINED and the re-run rule

A cell that is not DEFINED (missing or unreadable arm, sidecar not pinned to the revision, `null-noise`,
`lesion-noise`, an organ error on the probe turn, UNRELIABLE, a degenerate intact reply) is re-run once, at the same
revision, with the job line unchanged, logged in `research/packets/integration-wave2/reruns.tsv` before its result is
read. A cell still not DEFINED stays UNDEFINED. UNDEFINED never counts as load-bearing, never as a pass of B1, and a
criterion that depends on it is UNDEFINED.

## Lessons applied from the wave-1 review (built in, each pinned by `tests/test_integration_wave2_anchors.py`)

1. Every `--checked` reason names its prereg as `prereg <path> ;` (a space before any `;`), so the pool's GATE-SEED
   GUARD resolves the path; the test resolves each path on disk.
2. Declared `mem_gb` <= 11 (the largest size the dispatcher hands a 15 GB node); the identity arm keeps one brain
   alive at a time and its need is the peak RSS measured on dev seed 7 (recorded in the arm file).
3. `mem_ok.sh` + `memcap.sh` on every line; every LB line also runs `tools/assert_iw2_node.py` first (exit 3 without
   data/corpus/ or a resolvable LTM bundle), and the identity runner refuses the same way itself.
4. The scorer never passes an empty, absent, abstaining or echoed intact reply (UNDEFINED), and never passes a wrong
   one (`intact-wrong`, FAIL); a permuted control is part of every R bar.
5. UNDEFINED never counts; step A can never turn a B that carried undefined criteria into a pass.
6. Every line carries `max_age_h=N` (N = 216 here: the pool held ~450 lines at packet time, three to four days of
   work, with wave-1 and Stage-1 lines ahead of these; the cap is 240).

## Predictions (written before any gate job)

A scratch feasibility run through the real handler at dev seed 7 preceded this prereg (numbers in the packet preregs;
the recorded dev-seed runs are in the shared HANDOFF section 9). It changed two things before any gate data, each
declared as a deviation in its packet prereg: the transitive probe's words (e0..e4 are not words to the acquisition
path) and the transitive route's closed-class scope guard.

- D0 IDENTICAL (the anchors are False; the readers are the pre-anchor expressions, pinned for every env value).
- D and S: DIFFERS-IN-SCOPE for all three packets. neural-selfid's own turns differ at reply level on "what are you?"
  / "who are you?" (the flag turns the router's keyword reply into an abstain; its prereg records this as a known
  behaviour change for the merge decision).
- R: false-belief-chat PASS; neural-selfid PASS; transitive-chat PASS with less margin (a teach turn held back by the
  comprehension monitor, or a chase that does not reach the third hop, on some gate seed would read FAIL).
- B: PASS for the flipped set: each flag acts only on its own disjoint question shape (the S check tests this). The
  most likely B1 exposure is neural-selfid on the anaphora rows (the 2026-09-05 pair flip's regression class).

## Not measured here

- **Latency.** Reported per turn by step D (`turn_wall_s`), not gated (non-negotiable 4). The transitive chase runs a
  multi-cycle spiking simulation per hop; its per-turn cost is reported in the transitive HANDOFF.
- **The Qwen mouth.** Every arm uses the stub renderer, as every battery does.
- **Multi-scenario conversations** (a second false-belief scenario in one session, a second chain), and the rich path
  (neither route is wired into `rich=True`).
