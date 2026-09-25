---
type: finding
status: partial
claim_check: measured
date: 2026-09-25
lane: load-bearing (memory -- a fact told hours before sleep, kept by rest and captured by the night)
mechanism: replay pattern completion (webapp/replay_completion.py; BRAIN_AWAKE_REPLAY_COMPLETION for the awake-rest
  bouts, BRAIN_SLEEP_REPLAY_COMPLETION for the night's SWR epoch, BRAIN_REPLAY_COMPLETION_LESION; all default OFF) --
  a replay event reinstates the stored fact's ensemble through a spiking item competition and the composer's own
  re-bind op, and each route uses the reinstated ensemble's coherence with the block's increment (R_c) where it used
  the partial trace's decode margin (R)
prereg: research/findings/2026-09-24-sleep-replay-capture-PREREGISTRATION.md (Amendment 7, committed after this
  finding and before any gate-seed run of the `arcc` family)
seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
seed-waiver: development seeds only, by design; none of the six gate seeds (42/43/44/100/101/102) was built by this
  arc. This finding makes no generalization claim; the six-seed test is Amendment 7's `arcc` family, prepared and
  not run.
runner: research/runners/_awake_replay_completion_dev.py
artifacts:
  - research/findings/raw/_awake_replay_completion_dev/scan/s*_arcc_scan.json
  - research/findings/raw/_awake_replay_completion_dev/scan/s7_arc.json
  - research/findings/raw/_awake_replay_completion_dev/arms/s2_arcc.json
  - research/findings/raw/_awake_replay_completion_dev/arms/s2_arcc_nocomp.json
  - research/findings/raw/_awake_replay_completion_dev/arms/s7_arcc.json
  - research/findings/raw/_awake_replay_completion_dev/arms_both/s*_*.json
  - research/findings/raw/_awake_replay_completion_dev/arms_both/attribution.json
verdict: DEV CHECK (not a gate row, not a GO): on the two weak dev seeds of fifteen (2 and 13, fresh reads 0.16 and
  0.03, both below the failing gate seed's 0.21) the margin routes lose the long-delay fact and the replay
  completion keeps it (correct on both, captured); on seed 2 the rescue vanishes with the awake edge cut, with no
  rest, with the night's replay edge cut and with the DA-encoding lesion; the normal dev seeds (7, 11) are unchanged;
  no arm confabulated; the completion-lesioned arm reproduces the pure margin route field for field. BUT the rescue
  on both weak seeds is carried by the NIGHT's completion: the awake completion alone did not rescue either seed
  (the night's margin read stayed below the capture point), the night's completion alone rescued both (the awake
  margin route plateaued above the night's ignition point instead of collapsing). Whether the awake completion is
  needed on a seed whose margin route collapses deeper -- as gate seed 101's did -- is not answered here; Amendment 7
  registers it as REPORTED attribution arms.
---

# Replay pattern completion on dev seeds: the night's completion carries the rescue on both weak seeds; the awake completion alone does not

## The wall, and the question asked first

The awake-rest route scored NO-GO 5/6 (`research/findings/2026-09-25-awake-replay-capture-arc-no-go-6seed.md`): on
one gate seed the long-delay fact's read R began at 0.207451321 and fell to 0.030533799 by sleep onset <!--derived-->. Each 5-min
bout re-induces e <- e + R (1 - e), where R is the cleanup's decode margin (peak - runner_up) / peak, and between
bouts e decays by exp(-5/90) = 0.946 <!--derived-->. So a block is held only where R(e) (1 - e) >= ~0.057 e; a block
whose margin is low at every expression has no upper fixed point and rest drives it down (subcritical).

The wall question, before building: what does the real system run alongside this that the code replaced with a
linear proxy? The margin measures READ-OUT confusability (how close the most similar other word sits at the cleanup);
the LTP a replay induces depends on PARTICIPATION (how many of the trace's own pre/post pairs fire together; Sadowski
et al. 2016). In the hippocampus the two come apart through CA3 pattern completion: a replay is a population burst of
the CA3 recurrent network that starts at a threshold (de la Prida et al. 2006), and a reactivated subset of a stored
assembly activates the whole ensemble (Kandel 6e ch.54; Nakazawa et al. 2002; Guzman et al. 2016). The linear proxy
R_eff = margin omits the completion. Binding: `research/biology/awake-replay-pattern-completion.md` (11 sources; the
local Kandel and Buzsaki anchors resolve).

Record checked first: `bash tools/before_you_build.sh "awake replay subcritical collapse weak trace"` (logged) and
the local corpus (`rag_search.py`, corpus all / kandel / paper; the Buzsaki and Kandel passages were read in the
text). The one prior it surfaced is `2026-08-20-idle-replay-trace-stabilization-directional-but-subbar-nonspecific-
UNDEFINED.md`, whose named next lever is exactly this: "pattern-completion reactivates only the ENCODED assembly (a
real trace re-ignites, an unencoded one does not)".

## Alternatives weighed, with the evidence against each as the fix for THIS failure

- **Replay prioritization / tag-dependent selection** (Schapiro et al. 2018: awake human replay favours weakly learned
  items). It argues against the linear proxy too, but in this scenario one fact is managed and every bout already
  drives it (the arc family's I1 held on all six seeds: 48 bouts, a substrate read each). The failure is the size of
  each replay's effect, not which memory is replayed.
- **Synaptic-tag lifetime.** Each bout re-sets the tag from e, and the night's re-tag and DA are set by the read at
  sleep onset (0.030533799 on the failing seed <!--derived-->). A longer tag lifetime cannot rescue a trace whose read collapsed.
- **The read's heterogeneity origin.** Measured here (table below): over 15 dev seeds the fresh read varies about
  17-fold (0.035-0.603 <!--derived-->), set on each seed by the one role whose nearest competitor word sits closest,
  while the completion read R_c at full expression varies 1.5-fold (0.560-0.816 <!--derived-->; 0.708-0.816
  <!--derived--> on the 13 seeds where all resolving items resolve, the two lower values being one tied item). The
  heterogeneity is read-out crosstalk, not how much of the trace is left -- the argument for replacing the margin as
  the reactivation quantity rather than retuning it.
- **The CA3 superposed-fact attractor** (`research/runners/ca3_superposed_fact_attractor.py`, capacity GO 6/6). The
  awake read cannot route through it: it is a standalone binary gamma-cycle k-WTA runner (host argpartition) with its
  own EC codes and no chat write path, so using it would need a second store for every told fact. Not used; named as
  the rung that would make the completion literally CA3.

## What was built (default OFF, byte-identical off)

`webapp/replay_completion.py`: the route's own read R runs unchanged and is recorded; per content role the concept
units' matched-filter drive (the same substrate read) drives the composer's Izhikevich concept bank (peak-normalized,
at its graded operating point `_margin_drive_pA`, for `_cleanup_window` steps); the most-firing unit is the reinstated
item (a silent or tied competition reinstates nothing; a reserved slot never); the reinstated items are re-bound on the
composer's own resonate-and-fire work registers by `_compose_phases`, the op that encoded the fact; R_c is the in-phase
coherence of that reinstated pattern with the block's stored increment. The awake bout induces with R_c
(`BRAIN_AWAKE_REPLAY_COMPLETION`); the night's epoch re-tags, releases SWR DA and protects with R_c
(`BRAIN_SLEEP_REPLAY_COMPLETION`); `BRAIN_REPLAY_COMPLETION_LESION` keeps every read and uses R in both. No new
constant. Declared host shortcuts (module docstring): op dispatch, one pass, peak normalization, the largest-count
read of the winner, R_c arithmetic, a wrong item's LTP dropped by the ledger's bookkeeping; polarity not reinstated.

Tests (`tests/test_awake_replay_completion.py`, 18 pass): with both flags unset the store hash equals main's modules at
`05eba333f` on three fake scenarios (awake rest + night; night only; a very-low-margin block); the completion lesion
writes exactly the margin routes' store; the substrate read on a real D=64 composer equals the composer's own block
read, a fully expressed block reinstates its own three items, the bare baseline does not, and a silenced competition
reinstates nothing. The runner's `--family arcc` selftest passes (GO, NO-GO, UNDEFINED incl. the new I4).

## The design changed once, on a dev seed, before any gate seed

The first build (commit `938ee5ad5`) completed the AWAKE bouts only. On dev seed 2 it kept the trace (expression
0.982736605 after the last bout, all three items reinstated at every bout, R_c 0.754680272) but the fact was still
lost: the night's epoch re-tagged and released DA from the margin read (0.106879619 -> SWR DA 0.579090918, D1
activation 0.119442293, z > 1/2 on 0 synapses). The night route carries the same proxy. A sleep SWR is the same CA3
burst, so the second build (`c1429b20a`) completes the night's epoch too, under its own flag, with one lesion for both.

## Dev-seed scan: the read R and the completion read R_c against the trace's expression

Each dev seed: the `datr` telling through the real `brain_chat` handler, then the fact block rewritten as
base + e x inc for each e (read-only; restored), read by the margin (R) and by the completion (R_c).
Seed 7 comes from the 17-point curve of its awake-completion run at `938ee5ad5`
(`research/findings/raw/_awake_replay_completion_dev/arms/s7_arcc.json`); the others from
`research/findings/raw/_awake_replay_completion_dev/scan/s<N>_arcc_scan.json`.

| seed | R e=1 | R e=0.9 | R e=0.7 | R e=0.5 | weakest role | R_c e=1 | R_c e=0.7 | R_c e=0.5 | R_c e=0.3 | R_c e=0.1 | R_c e=0.065 (0.07 for seed 7) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.433870373 | 0.420027508 | 0.378790872 | 0.297610260 | patient | 0.750280588 | 0.750280588 | 0.593133824 | 0.593133824 | - | - |
| 2 | 0.159776106 | 0.081054535 | 0.125502214 | 0.083666789 | agent | 0.754680272 | 0.439902563 | 0.000000000 | 0.000000000 | - | - |
| 3 | 0.507390572 | 0.483228579 | 0.411788817 | 0.273449191 | patient | 0.764019764 | 0.764019764 | 0.764019764 | 0.352475355 | - | - |
| 4 | 0.514473201 | 0.495900247 | 0.387801308 | 0.173874567 | patient | 0.741958157 | 0.741958157 | 0.598585137 | 0.000000000 | - | - |
| 5 | 0.494308634 | 0.478331314 | 0.433296246 | 0.354458737 | patient | 0.771269013 | 0.771269013 | 0.771269013 | 0.771269013 | - | - |
| 6 | 0.603362074 | 0.572930858 | 0.477139006 | 0.272410834 | patient | 0.781471965 | 0.781471965 | 0.781471965 | 0.000000000 | - | - |
| 7 | 0.575156437 | 0.559967471 | 0.514344447 | 0.295037347 | agent | 0.816382696 | 0.816382696 | 0.816382696 | 0.675708939 | 0.000000000 | 0.000000000 |
| 8 | 0.374530337 | 0.349956863 | 0.281944944 | 0.167155508 | agent | 0.777170765 | 0.777170765 | 0.777170765 | 0.432181725 | - | - |
| 9 | 0.400019459 | 0.364163156 | 0.264215728 | 0.093128870 | patient | 0.768547278 | 0.768547278 | 0.768547278 | 0.601429124 | - | - |
| 10 | 0.358172202 | 0.340456629 | 0.287415953 | 0.181798909 | patient | 0.744667843 | 0.615419723 | 0.615419723 | 0.499538565 | - | - |
| 11 | 0.293037829 | 0.253692205 | 0.133885387 | 0.076282360 | patient | 0.797126177 | 0.797126177 | 0.797126177 | 0.594767800 | - | - |
| 12 | 0.506940082 | 0.491219877 | 0.445974743 | 0.363279645 | action | 0.790950553 | 0.790950553 | 0.790950553 | 0.666858389 | - | - |
| 13 | 0.034828016 | 0.005652979 | 0.019970905 | 0.071584370 | patient | 0.707772701 | 0.707772701 | 0.707772701 | 0.707772701 | 0.000000000 | 0.000000000 |
| 14 | 0.459034849 | 0.432873880 | 0.351388577 | 0.173529761 | patient | 0.648881822 | 0.648881822 | 0.491321668 | 0.000000000 | 0.000000000 | 0.000000000 |
| 15 | 0.429042401 | 0.410403421 | 0.358776947 | 0.271507998 | agent | 0.560172503 | 0.560172503 | 0.473965491 | 0.473965491 | 0.000000000 | 0.000000000 |

Margin route replayed on each measured curve (48 bouts, linear interpolation of R(e); triage, not a run):

<!--derived-->
| seed | predicted R, 1st bout | predicted R, last bout | predicted e after the last bout |
|---|---|---|---|
| 1 | 0.424 | 0.416 | 0.929 |
| 2 | 0.105 | 0.121 | 0.719 |
| 3 | 0.491 | 0.481 | 0.945 |
| 4 | 0.502 | 0.494 | 0.948 |
| 5 | 0.483 | 0.477 | 0.944 |
| 6 | 0.582 | 0.576 | 0.962 |
| 7 | 0.565 | 0.561 | 0.959 |
| 8 | 0.357 | 0.334 | 0.903 |
| 9 | 0.375 | 0.343 | 0.906 |
| 10 | 0.346 | 0.328 | 0.900 |
| 11 | 0.266 | 0.149 | 0.767 |
| 12 | 0.496 | 0.490 | 0.947 |
| 13 | 0.015 | 0.064 | 0.558 |
| 14 | 0.441 | 0.427 | 0.932 |
| 15 | 0.416 | 0.405 | 0.926 |

Seeds 2 and 13 are the weak seeds of the fifteen (fresh reads 0.159776106 and 0.034828016, below the failing gate
seed's 0.207451321 <!--derived-->); seed 11 falls steeply (0.293037829 at e = 1, 0.133885387 at e = 0.7) and was run
as a middle case. The second table replays the margin route's 48 bouts on each seed's measured curve (linear
interpolation; triage only): it predicts the full runs closely where both exist (seed 2: e 0.719 predicted, 0.730 run
<!--derived-->; seed 11: 0.767 vs 0.795 <!--derived-->; seed 13: 0.558 vs 0.557 <!--derived-->) and puts every seed but
2, 11 and 13 above e = 0.9 after the rest, where the margin night already captures. Seeds 1-6 and 8-12 were scanned on a 5-point
grid (1, 0.9, 0.7, 0.5, 0.3) before the grid was densified.

## Dev-seed arms

Each arm is the arc family's `datr` protocol (the neutral telling, 48 quiet-rest ticks of 5 min over 4 h, the
night, the recall question) through the real handler, one fresh tiny-demo brain per arm, numpy, LTM off.
`arms/` = commit `938ee5ad5` (awake completion only: there `arcc` meant `arcc_awake`, and `arcc_nocomp` cut it with the
lesion's former name); `arms_both/` = the two-route design. Arm key: `arc` = the margin routes, no completion flag;
`arcc` = both routes complete; `arcc_awake` / `arcc_sleep` = one route completes; `arcc_nocomp` = both flags +
`BRAIN_REPLAY_COMPLETION_LESION` (every read runs, both routes use R); `arcc_lesion` = both + the awake edge cut;
`arcc_norest` = both, 4 h awake with no idle tick (`datl`); `arcc_dalesion` = both + `BRAIN_DA_ENCODING_LESION`;
`arcc_sleeplesion` = both + the night's replay edge cut; `noarc_c` = the night's completion, no awake route.
"night R" is the margin read at the sleep epoch; "night R_c" the completion read there (recorded whenever the night's
flag is armed, used unless the lesion is on); "z>1/2 at recall" is the fraction of the block's synapses captured.

The completion-lesioned arm reproduces the pure margin-route arm field for field on seed 2 (every
awake bout's R, R_eff and expression, the sleep epoch's R, DA and D1 activation, the ledger blocks at recall and the
outcome; `research/findings/raw/_awake_replay_completion_dev/arms_both/attribution.json` records the check under
`lesion_reproduces_margin_route`; the `938ee5ad5` lesion arm reads the same values in the table): the completion READS are inert when their effect is cut.

| run | outcome | bouts | R 1st bout | R last bout | R_eff last bout | e after last bout | night R | night R_c | SWR DA | D1 a_eff | z>1/2 at recall | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arms/s2_arcc | abstain | 48 | 0.107457145 | 0.106879619 | 0.754680272 | 0.982736605 | 0.106879619 | - | 0.579090918 | 0.119442293 | 0.000000000 | 636.7 |
| arms/s2_arcc_nocomp | abstain | 48 | 0.107457145 | 0.127366640 | 0.127366640 | 0.729794488 | 0.127366667 | - | 0.594251334 | 0.144539117 | 0.000000000 | 658.1 |
| arms/s7_arcc | correct | 48 | 0.565032456 | 0.565680897 | 0.816382696 | 0.987991394 | 0.565678299 | - | 0.918601941 | 0.588613478 | 1.000000000 | 1170.9 |
| arms_both/s11_arcc | correct | 48 | 0.266656712 | 0.267805287 | 0.797126177 | 0.986432929 | 0.267805287 | 0.797126177 | 1.089873371 | 0.811696359 | 1.000000000 | 1070.8 |
| arms_both/s11_arcc_nocomp | correct | 48 | 0.266656712 | 0.172564575 | 0.172564575 | 0.794999654 | 0.172422697 | 0.797126177 | 0.627592796 | 0.195662277 | 1.000000000 | 1009.7 |
| arms_both/s13_arcc | correct | 48 | 0.007662999 | 0.005527226 | 0.707772701 | 0.978174563 | 0.005527226 | 0.707772701 | 1.023751799 | 0.695972115 | 1.000000000 | 999.5 |
| arms_both/s13_arcc_awake | abstain | 48 | 0.007662999 | 0.005527226 | 0.707772701 | 0.978174563 | 0.005527226 | - | 0.504090147 | 0.019984508 | 0.000000000 | 365.5 |
| arms_both/s13_arcc_nocomp | abstain | 48 | 0.007662999 | 0.063591722 | 0.063591722 | 0.556871989 | 0.063591968 | 0.707772701 | 0.547058056 | 0.068783888 | 0.000000000 | 402.3 |
| arms_both/s13_arcc_sleep | correct | 48 | 0.007662999 | 0.063591722 | 0.063591722 | 0.556871989 | 0.063591968 | 0.707772701 | 1.023751799 | 0.695972115 | 1.000000000 | 960.8 |
| arms_both/s2_arc | abstain | 48 | 0.107457145 | 0.127366640 | 0.127366640 | 0.729794488 | 0.127366667 | - | 0.594251334 | 0.144539117 | 0.000000000 | 331.4 |
| arms_both/s2_arcc | correct | 48 | 0.107457145 | 0.106879619 | 0.754680272 | 0.982736605 | 0.106879619 | 0.754680272 | 1.058463401 | 0.752672347 | 1.000000000 | 927.0 |
| arms_both/s2_arcc_dalesion | abstain | 48 | 0.048277557 | 0.047559032 | 0.754680272 | 0.982736605 | 0.047524070 | 0.754680272 | 1.058463401 | 0.029047250 | 0.000000000 | 464.6 |
| arms_both/s2_arcc_lesion | abstain | 48 | 0.107457145 | 0.045007990 | 0.000000000 | 0.068334991 | 0.040532097 | 0.000000000 | 0.500000000 | 0.029047250 | 0.000000000 | 485.6 |
| arms_both/s2_arcc_nocomp | abstain | 48 | 0.107457145 | 0.127366640 | 0.127366640 | 0.729794488 | 0.127366667 | 0.439902563 | 0.594251334 | 0.144539117 | 0.000000000 | 358.2 |
| arms_both/s2_arcc_norest | abstain | 0 | - | - | - | - | 0.040532097 | 0.000000000 | 0.500000000 | 0.029047250 | 0.000000000 | 397.0 |
| arms_both/s2_arcc_sleep | correct | 48 | 0.107457145 | 0.127366640 | 0.127366640 | 0.729794488 | 0.127366667 | 0.439902563 | 0.825527897 | 0.449651433 | 1.000000000 | 910.8 |
| arms_both/s2_arcc_sleeplesion | abstain | 48 | 0.107457145 | 0.106879619 | 0.754680272 | 0.982736605 | 0.106879619 | 0.754680272 | 0.500000000 | 0.029047250 | 0.000000000 | 433.9 |
| arms_both/s2_noarc_c | abstain | 0 | - | - | - | - | 0.040532097 | 0.000000000 | 0.500000000 | 0.029047250 | 0.000000000 | 324.2 |
| arms_both/s7_arcc | correct | 48 | 0.565032456 | 0.565680897 | 0.816382696 | 0.987991394 | 0.565678299 | 0.816382696 | 1.104123195 | 0.834469404 | 1.000000000 | 986.9 |

## What this shows, and what it does not

**On both weak dev seeds the margin routes lose the fact and the completion keeps it.** Seeds 2 and 13 are the two of
fifteen whose fresh read is lowest (0.159776106 and 0.034828016 at e = 1, both below the failing gate seed's
0.207451321 <!--derived-->). With the completion lesioned (both routes on the partial cue's margin) both abstain; with both routes
completing both recall the fact the next day (captured: z > 1/2 on every synapse of the block). No arm confabulated.

**The rescue still vanishes where the prereg says it must** (seed 2, both completion flags armed): with the awake edge
cut the trace decays to 0.068334991 before sleep and the night reinstates no item (R_c 0, SWR DA at tonic); with no
idle tick (4 h awake, the `datl` group) the night again reinstates nothing; with the night's replay edge cut the trace
is held at 0.982736605 through the rest but nothing captures it; with the DA-encoding lesion the D1 pool sees tonic DA
(activation 0.029047250) and nothing is captured. The normal dev seeds (7, 11) recall the fact with the completion as
without it; the completion only moves their reads up (seed 11: R 0.172564575 at the last bout on the margin route,
0.267805287 with the completion).

**But on these dev seeds it is the NIGHT's completion that carries the rescue, not the awake one -- the hypothesis as
first built (awake route only) is not what these data support.** The one-route arms split it:

- awake completion only: seed 2 and seed 13 both still abstain. The trace is kept (0.982736605 and 0.978174563 after
  the last bout) but the night re-tags and releases DA from the margin read (0.106879619 and 0.005527226), below the
  point where the D1 pool captures (the SWR DA must reach the Go boundary 0.62, i.e. a night read of ~0.16 <!--derived-->;
  on seed 11's margin night 0.172422697 was captured, seed 2's 0.127366667 was not);
- night completion only: seed 2 recalls the fact (the awake margin route held the trace at 0.729794488, where the night
  reinstated one item, R_c 0.439902563, SWR DA 0.825527897); seed 13 recalls it too (the awake margin route held the
  trace at 0.556871989, where the night reinstated agent and action, R_c 0.707772701, SWR DA 1.023751799).

The awake margin route did not collapse on these two seeds the way it collapsed on gate seed 101 (R 0.207451321 ->
0.030467672 <!--derived-->): it plateaued (seed 2 at e ~0.73, seed 13 at ~0.56), because the margin read has a FLOOR -- a decayed
block whose cleanup confidently decodes a wrong word or a reserved slot still has a decisive margin (seed 7: R
0.293994549 at e = 0.3 with the agent decoded as a reserved slot; R 0.120892791 on the bare baseline). That floor is an
artifact of using a read-out statistic as the reactivation strength; R_c does not have it (R_c is 0 on the bare
baseline of both 17-point curves, seeds 2 and 7, and at e = 0.1 and 0.065 on every seed measured there). Whether the
awake completion is needed on a seed whose margin route falls below the night's ignition point -- as seed 101's may
have -- cannot be read from these dev seeds; Amendment 7 therefore registers the two one-route arms as REPORTED
attribution arms on the six gate seeds.

**Threshold, not proportionality -- on the substrate as built.** R_c is all-or-none per item: 0.56-0.82 while the
items resolve, 0 once they do not, with partial values when one or two of the three resolve. Across 15 dev seeds R_c at
full expression spans 0.560172503-0.816382696 while the margin spans 0.034828016-0.603362074; at e = 0.065 (the
expression a fact told 4 h before sleep has left) R_c is 0 on every dev seed measured there (7, 13, 14, 15), which is
what keeps ARC1's `lr_noarc` and ARC3's `ln_arc` abstaining with the night's completion armed.

**What it does not show.**
- Not literal CA3: the completion runs through the fact's own readout -> cleanup -> re-bind loop; the concept bank has
  no lateral inhibition (its spike counts are small: single digits in the 120-step window, 5-8 per the binding), so near-ties resolve as "no item"
  and the bank's most excitable unit wins an equal drive. On seeds 14 and 15 one item ties even at full expression
  (R_c 0.648881822 and 0.560172503 there).
- A two-item reinstatement re-potentiates the whole stored increment (seed 13: the patient never resolved, R_c
  0.707772701 from agent + action, and the block was captured and recalled whole). That is what completion means, but
  the credit to the unreinstated component is the ledger's bookkeeping, declared; the LTP a WRONG item would write is
  dropped (it can only understate confabulation risk).
- One fact per conversation, LTM off, numpy CPU, dev seeds only. No gate seed was built. This is a dev check, not a
  gate row and not a GO.

## Next

Amendment 7 of the prereg (committed after this finding, before any gate-seed run) registers the `arcc` family for
the six gate seeds; the six pool lines are below. They are prepared, not queued.

The six pool lines, pinned to the Amendment-7 commit, are added below in the commit that follows it.
