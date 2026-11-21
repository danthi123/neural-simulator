# Capability: Lifelong learning
# Record: brain/research/2026-11-19-multiweek-life-design.md (section 9, D2); _scratch/stage1/hub_verdict.md + hub_critique.md (research round 11-21)
# Expected-fail: none named; at 1.5B the replies may stay flat even if replay share moves (the M-scale lesson)

# Stage-2 method D2-PR: loss-prioritised replay (prereg, 2026-11-21, written before any D2-PR life)

## 0. Record brief (research round 11-21, before a 3rd hub lever)

**Record.** Two hub levers have already failed: R-raw and M-use, both KILLED.
- R-raw arm D (S0 raw ranking) does dissolve the battery hub, but the open-turn fires stay high. It remains the only
  measured selection lead.
- A critique of the research round (opus, read-only, verified against the code and data) found that from night 10 the
  biggest conversational defect is on the CORTEX side, not the hippocampus. In the s318 DEV interview, the stock
  reply "Of course, you're Dan, my friend." answers the KNOW questions:
  - with the hippocampus OFF: 0/6 on night 5, 10/10 on night 10, 7/7 on night 15;
  - with both off: 0 on every night.
- H also falls with the hippocampus off: 5-6/6 on nights 1-2, 1-3/6 by nights 13-15. So no recall-selection lever can
  fix the reply.

**Measured cause** (s316 DEV night-15 trunk home, offline, `_scratch/stage2/replay_loss_check.py`). Replay is
weighted by effective strength (`Brain.sleep`, `brain/brain.py`), and 6 of the 98 episodes take 39% of the night:

| Episode | Replay share | Why it is weighted up |
|---|---|---|
| "The bus was late again this morning." | 11% | 12 uses (each fetch adds USE_BOOST) |
| the "Dan" exchange | 9% | permanent (weight >= 3.0) |
| the two "Nova" exchanges | 12% | permanent |
| sister | 4.5% | use |
| "I stayed up too late watching a documentary about bridges." | 3% | use |

The last one comes back as an asserted experience claim in the interview. The cortex has nearly memorised every one
of these: NLL with adapters on is 0.15-0.41, against 1.3-3.1 pretrained. The highest-loss episodes are the facts told
once and never used (NLL on 1.0-1.8): those are the ones replay should strengthen. This is a rich-get-richer loop. A
fetch raises strength, strength raises replay, replay drills the cortex, and the drilled cortex answers with it.

**Biology.**
- Prioritised replay: replay is weighted by its expected gain or prediction error (Mattar & Daw 2018, Nat Neurosci;
  Schaul et al. 2016, prioritized experience replay).
- Sleep benefits weakly learned items most (Schapiro et al. 2018, Nat Commun; Denis et al. 2020).

**Prerequisites.** Nightly sleep, adapters and per-episode replay texts are MET. The cortex's own NLL on a replay text
is available on the slept cortex through `plastic(ids, base)` (the same path as `life.span_read` and `sleep.py`).
`LiveCortex.read_nll` refuses a slept cortex and is not used.

**Host decision.** None. The weights come from the cortex's own error on each episode. There is no threshold and no
new constant.

## 1. Method

At the start of each night, before replay, compute L_e for each stored episode e that the night replays (the
permanent ones included). L_e is the mean per-token NLL of the episode's replay text (`f"{e.text}\n\n"`) under the
current adapters, with no gradient and no state change.

(Implementation note, 11-21, before any D2-PR life: the loss pass runs inside `Brain.sleep` just before replay, after
the M-scale "before" cut if that switch is on, so L_e is read on the adapters replay starts from; M-scale is off in
this screen, so this is the start of the night.)

Arms, both behind `Brain(replay_priority=...)` (default None = today's weighting, byte-identical):
- **PR-x** (`"strength_x_loss"`): w_e = effective_strength(e) x L_e.
- **PR-l** (`"loss"`): w_e = L_e (pure prioritised replay with alpha = 1; strength no longer weights replay, but it
  still governs recall and decay).

On night 1 the adapters are fresh (zero B), so L_e is the pretrained NLL. Nothing else changes: steps, lr, the KL
guard and the ordinary texts all stay as they are. Each night's weights are logged: per-episode L_e, w_e and the
share by class.

Offline prediction on the s316 night-15 home (engagement, not evidence):

| Arm | Permanent share | Top-6 share | Largest single episode |
|---|---|---|---|
| current | 0.21 | 0.39 | 0.11 |
| PR-x | 0.11 | 0.26 | 0.08 |
| PR-l | 0.01 | 0.13 | 0.04 |

## 2. Screen (DEV only, no registered claim)

- **Seeds.** DEV 319, 320 and 321, paired with the banked all-flags DEV defaults lives of the same seeds (pool runs
  started 11-21, `lifeS1F_s<S>`, B arm).
- **Command.** The §13 command on `--split dev`, plus `--replay-priority {strength_x_loss|loss}`. If capacity forces
  it, the screen may cut to 10 nights with `--full-probe-nights 1,5,10`, `--interview-nights 5,10` and
  `--pruned-a-day 6`. It is then paired against the same nights of B.
- **Compute.** AWS g5 or pool CPU, as capacity allows; the host is recorded.

## 3. Read-outs and bars (pooled over 3 seeds; the direction is also reported per seed)

1. **Engagement (must hold, or the arm is UNDEFINED).** On nights 5 and 10, the permanent replay share and the top-6
   share are lower than B on 3/3 seeds.
2. **Primary, the stock self-reply leak.** The share of probe and interview replies, intact and hippocampus-off,
   whose text contains a permanent episode's Assistant reply ("you're Dan", "My name is Nova", "I'm Nova") on a
   question that is not about names or identity. It is reported on nights 10 and 15. Bar: below B on 3/3 seeds, and
   pooled at least 30% lower. Post-hoc read from the results JSON by `brain/eval/life_leak.py` (written before any D2-PR result). Identity
   questions are the ledger items without a telling (the user's and the brain's names). DEV baseline, s318 (B):
   39/152 = 0.26 on nights 10 and 15, almost all on the interview's KNOW replies.
3. **Guards (any fail = KILL).**
   - H hippocampus-off on nights 5-15 is not lower than B by more than 0.5/6 pooled per seed.
   - Direct hippocampus-off recall on day-1-2 facts is not lower by more than 1/6.
   - Collapse is not worse.
   - Language-guard KL at night 15 is not higher than B by more than 25%.
   - The whole-transcript read (3 sonnet readers plus refuters, the M-scale protocol, or the §7 reader once
     validated) finds no new defect class. A new defect class means identity loss, name overwrite, a marker leak or
     a stuck tail absent in B.
4. **Secondary, report only.**
   - Hippocampus-off recall of once-told E and F facts (prediction: up).
   - Wrong fetches and hub top-1 (a hippocampal read-out; PR does not touch recall, so these should not change, which
     also serves as a check).
   - Interview AUROC and invented-memory rate.
   - Adapter norm (prediction: it grows more slowly).

## 4. Decision

- PR passes 1 + 2 + 3 on DEV: it goes to a registered six-seed test on a FRESH TEST split (§9: any method change
  after a TEST read writes a fresh split), and becomes the default only on that result (land-result).
- 2 fails, 3 holds: NARROWED (replay share moved, replies did not); the cause then sits elsewhere (e.g. the KNOW
  wording overlapping the permanent "Do you remember my name?").
- 3 fails: KILLED.

Timebox: 2 days (to 11-23). The two arms are the whole grid; no third weighting is tried before this read.
