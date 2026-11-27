# Capability: Sleep consolidation
# Record: brain/experiments/results/2026-11-27-local-sleep-01b-read.md

# Local rule v2 prereg (2026-11-27): the local delta rule with its plasticity confined off the keep bank's own subspace

STATUS: DRAFT pending a fresh pre-data review (sonnet). Revised 2026-11-27 after the first pre-data review (14 items;
the response is the last section). Nothing has run. Settings below are frozen only after that review
and the seed-7 dev picks. Milestone: ARCHITECTURE section 4, 2026-12-31, "sleep learning by a local, biology-style rule
instead of gradients; scale to 1.5B". This is the next method after the v1 NEGATIVE; it does not re-open that verdict or
the SHY screen (six seeds still running; read separately by its own rule).

## Record brief
- **Knew.** v1 local rule, 0.1B six seeds: NEGATIVE (recall 0.17/12 vs bar 2; under the drift cap the rule is nearly
  inert, local = shuffled) (`experiments/results/2026-11-27-local-sleep-01b-read.md`). Seed-7 dev: 0.1B transport lr
  0.1-0.3 recall 2 at KL 0.11-0.12 (3.5x gradient, `dev_pick.md`); 0.4B transport lr 0.3 recall 4/12 at KL 0.089 vs
  gradient 5/12 at 0.032 (`dev_s7_04b/`); SHY cut g 0.6 keeps recall 4 at KL 0.035, but the gradient night given the same
  cut falls to 0.017, so (e2) is expected to fail on the six seeds (`shy_dev_pick.md`). The rule CAN learn the replay at
  both scales (loss 2.4 -> 0.25); the binding constraint is drift per unit of learning.
- **Why the v1 keep term fails.** The keep term is a penalty: a feedback-carried KL error on keep text plus a B decay,
  stepped against the replay error. Less drift costs learning step for step, and with transposed or random feedback the
  keep error is carried less accurately than the replay error it competes with.
- **New (picked by the record).** Attack drift at the source instead of penalising it. The adapter's effect on an input x
  at layer i is (x A) B; if A is orthogonal to the span of the layer's keep-bank activity, the delta on keep text is zero
  whatever B learns, and by induction the keep text's later-layer inputs are unchanged. Per layer and path the pretrained
  cortex's own pre-synaptic activity on the keep bank (k_in for the key path, hidden h for the value path) gives an
  uncentred principal subspace U holding a fraction tau of its energy; A starts in and is kept in the complement
  (A <- (I - U U^T) A after every local step). B, feedback, KP copies, clip, decay and the keep-error step are v1's,
  unchanged (`subspace="none"` reproduces v1 step for step; tested). Chosen over the other candidates: a layerwise
  anchor of the adapter output to zero on keep text is still a penalty that trades against learning (the v1 failure
  mode); top-k sparse updates do not target keep text at all; replay interleaving is already in v1 (one keep text per
  step). Method family: orthogonal weight modification / gradient projection memory (Zeng et al. 2019, Saha et al. 2021),
  here with a local rule, no backprop through the network.
- **Biology and scaffolds.** A decorrelating (anti-Hebbian lateral) circuit trained on ordinary activity converges to
  this projector (Oja/Foldiak subspace learning); an SVD of the stored keep activity stands in for the converged circuit
  (declared scaffold; to add to SCAFFOLDS.md on a GO). tau is a host constant (declared). Feedback preset "transport"
  (readout and value-weight transposes, residual identity) is the v1 declared weight-transport scaffold.
- **Prerequisites.** Met: sleep_probe battery with no-sleep and shuffled arms; adapters exactly zero at birth. Missing:
  credit through the WKV state (as v1); the keep bank is 3 short texts (~75 tokens), so U has rank <= ~75 of C = 768
  (0.1B) / 1024 (0.4B): many free directions, but the held-out generalisation of the protection is unknown.
- **Confound handled.** The keep bank (`sleep_probe.GENERIC`) is also the battery's drift text, so a projector fitted on
  it passes the battery's drift by construction. Added held-out drift: KL(on || off) over all positions of
  `mix_night_screen.PROMPTS` (24 ordinary prompts, pinned by sha256 `77e31aa1...d8d6`, the list at commit 4389b39; a
  changed list refuses the run) in the `ask()` frame, for EVERY arm (gradient included). Split: even indices (12) = DEV
  half, read only by the seed-7 pick (`heldout_kl_dev`); odd indices (12) = VERDICT half (`heldout_kl`), gating c2, the
  lesion (f) and the shams (g). Generic drift gates only c1 (v2 must not be worse there) and is reported for the controls,
  never used to credit the projection.
- **Expected (my estimate, not measured).** About 45% that dev finds a config under both caps with recall >= 2 at 0.1B
  (projection kills drift on keep text by construction; recall may drop because fact texts share the chat frame and
  common directions with the keep bank); given dev passes, about 30% GO-V2 at 0.1B six seeds, about 20% for both scales.
  Most likely failing clauses: (a) recall and (c2) held-out drift. Expected-fail: missing temporal credit.

## Method
Module `brain/experiments/local_rule_v2.py` (imports `local_sleep`, does not edit it), tests
`brain/tests/test_local_rule_v2.py`, node script `brain/experiments/results/2026-11-27-local-rule-v2/run_node.sh`.
Battery `sleep_probe.evaluate` (12 told facts hippocampus off, 4 never-told, generic drift on GENERIC, paraphrase reported)
plus `heldout_kl`. Episodes `build_arm_episodes(seed)`; 240 steps; `torch.manual_seed(seed)` before each arm's
PlasticCortex (same adapter init in every arm). Each (tag, seed, arm) is one json unit, atomic write, resumed by skip;
each records host, host_env, commit, device, precision (fp32), lr, tau, subspace ranks per layer and the keep energy
inside the subspace, factor norms, first/final replay loss, elapsed. Guard: POOL_NODE=1 or BRAIN_ALLOW_W3090=1.

## Arms (per model, per seed)
1. `v2`: local rule (KP, transport, decay 1e-3) with the keep-subspace projection at (lr*, tau*).
2. `gradient`: `brain.sleep.sleep` (Adam 5e-4, KL weight 1), true episodes: the reference.
3. `no_sleep`: adapters at zero.
4. `v2_shuffled` (no-contrast control): `v2` on the shuffled episodes (same sentences, another fact's answer).
5. `noproj` (lesion): the same rule at lr* with the projection removed (= v1 at lr*).
6. `sham`: the projection replaced by random orthonormal subspaces of the SAME rank per layer and path (seed + 7919).
7. `sham_tok`: the projection replaced by the top principal subspace, at the SAME rank per layer and path, of the
   pretrained cortex's own activity on random token sequences of the keep bank's lengths (seed + 7920). Real activity,
   including the shared (uncentred) mean direction every input carries, but no keep-text content: asks whether any
   real-activity subspace of that rank does it. The random sham alone removes only about m/D of A (rank <= ~75 of
   768/1024), so it sits close to the lesion; `sham_tok` is the harder control.
Seeds 42, 43, 44, 100, 101, 102. 0.1B `rwkv7-g1d-0.1b-20260129-ctx8192.pth`, then 0.4B `rwkv7-g1d-0.4b-20260210-ctx8192.pth`.

## Dev (seed 7, per model, frozen before that model's six seeds)
Grid: lr in {0.1, 0.3, 1.0} x tau in {0.9, 0.99, 0.999} for `v2` (9 nights), plus `no_sleep` + `gradient` re-run in this
harness on seed 7 (`ref`; needed for the held-out drift; its generic KL is reported next to the pinned v1 refs, 0.1B
0.032 / 0.4B 0.03234). Pick rule (`dev_pick`, tested): qualify if every value is finite AND recall >= the verdict's own
bar, max(0.5 x ref gradient recall, no_sleep + 2) AND generic_kl <= 1.25 x the ref gradient's AND heldout_kl_dev (DEV
half only) <= 1.25 x the ref gradient's AND rank_min >= 4 (every layer and path protects at least 4 directions, more
than the shared mean direction); pick max recall; ties to lower heldout_kl_dev, then lower lr, then lower tau.
None qualifying: **NEGATIVE-DEV** for that model, its six seeds are not run, no grid extension. A non-finite ref value:
**UNDEFINED-DEV** (the scale is not closed; find the cause, re-run). The pick refuses (exit non-zero, no pick.json)
unless the dir holds exactly the two ref units and one unit per (lr, tau) cell, all seed 7, one model, one commit, one
prompt list; the node script stops on any unit that exits non-zero, so an OOM or a bug never becomes NEGATIVE-DEV.
pick.json carries the model id and commit; the six-seed run refuses a pick of another model or commit and takes lr, tau
from it; the score refuses units that differ in model, commit, prompt list, lr (local arms) or tau (projected arms), or
that disagree with the pick. Dev reports m_k / m_v per layer for every cell.
0.4B gets its own dev pick (operating points do not transfer across scale, PREREQUISITES scale note). One dev seed and
discrete recall /12: noisy, disclosed.

## Measures (hippocampus off, per seed, per arm)
Recall /12, false recall /4, generic drift (KL on GENERIC), held-out drift (VERDICT half; DEV half reported), paraphrase
/12 (reported), adapter norm, factor norms, replay loss first/final, subspace ranks and keep energy inside, and the mean
applied (post-projection) step norm of the replay update and of the keep-error update.

## Registered rule (written before data; `verdict()` in the module, tested with each clause failing alone)
Per model, six seeds, means across seeds; `verdict()` refuses (ValueError) unless every one of the 7 arms holds exactly
the six seeds. bar = max(0.5 x mean gradient recall, mean no_sleep recall + 2) (the same bar as the dev floor).
"Held-out" below is the VERDICT half.
- **UNDEFINED** if any recall, false recall, generic or held-out drift of any arm and seed, or any `v2` adapter norm,
  is non-finite; or mean gradient recall < 2; or any `v2` adapter norm is zero; or mean `v2_shuffled` recall > 1.0 (the
  control moved; the recall measure is not specific to the true episodes).
- Core: (a) mean `v2` recall >= bar; (b) `v2` > `v2_shuffled` on >= 5 of 6 seeds AND mean `v2_shuffled` <= 1.0;
  (c1) mean `v2` generic drift <= 1.25 x mean gradient generic drift; (c2) mean `v2` held-out drift <= 1.25 x mean
  gradient held-out drift; (d) mean `v2` false recall <= mean gradient false recall + 0.5.
- (f) lesion, held-out drift only: mean `noproj` held-out > the c2 cap AND >= 1.5 x mean `v2` held-out AND `noproj`
  held-out > `v2` held-out on >= 5 of 6 seeds. (Generic drift is not used: the projection zeroes it by construction.)
- (g) each sham (`sham`, `sham_tok`) fails, by either: drift specificity (mean sham held-out >= 1.5 x mean `v2`
  held-out AND sham > `v2` on >= 5 of 6 seeds) OR recall specificity (mean sham recall < bar AND `v2` recall > sham
  recall on >= 5 of 6 seeds).
- **GO-V2** = core, (f), (g). **GO-PLAIN-LOCAL** = core and not (f) (the projection was not shown to be needed; not
  credited to it). **GO-NONSPECIFIC** = core, (f), not (g) (a rank-matched random or real-activity projection does it;
  the verdict names which sham passed; not credited to the keep subspace).
  **NEGATIVE** = a core clause fails: a verdict on this projection method at this scale, not on local learning.
A 0.1B GO alone does not pass: 0.4B must GO-V2 too before a 1.5B prereg. Report beside the verdict: all arms' means,
per-seed recall and held-out drift, subspace ranks, `noproj` / `v2` held-out ratio and seed wins, each sham's detail,
generic drift of every arm, step norms, replay loss.

## Compute (estimates from the v1 logs, not measured for v2)
CPU fp32, pool nodes, one process per node. v1 0.4B local night + battery 206 s, gradient night 8-14 min, RSS 3.3 GB
local / 7.1 GB gradient (shy_dev). v2 adds one keep read and two small projections per step (SVD of ~75 x 4096 per
layer: seconds) and a 24-prompt held-out drift per battery (~48 short forwards); `sham_tok` adds one more local night
and one more keep-length read per seed. Estimates: 0.1B dev ~25-35 min; 0.1B six seeds ~17 min/seed (3 per node on
pool41/42: ~50 min); 0.4B dev ~50-60 min (pool40); 0.4B six ~35-45 min/seed (3 per node: ~2-2.3 h). Memcap 8 GB (0.1B) /
13 GB (0.4B). Total about 5 node-hours, no GPU, no AWS.
Run (after `tools/mem_ok.sh`, on the node, `cd ~/brain-lane/route_repo`):
`bash brain/experiments/results/2026-11-27-local-rule-v2/run_node.sh 01b dev` (pool40), then
copy pool40's `~/brain-lane/out/local-rule-v2/01b/dev/pick.json` to the same path on pool41/42 (or set PICK=<path>),
`... 01b six 42 43 44` (pool41) and `... 01b six 100 101 102` (pool42), copy both `six/` dirs and the pick to one
machine, `... 01b score`; then the same with `04b`. The run path is `run_node.sh` only: the engine spec
`tools/engine/specs/local_rule_v2.{yaml,py}` (not written by this build, status TODO, so the engine refuses it; its cmd
still names `local_sleep` and `--arms TODO`) must be rewritten from this prereg before anything is queued through the
engine.

## Response to the pre-data review (2026-11-27)
Fixed: (1) (f) and (g) gate on the held-out VERDICT half only; generic is reported. (2) (f) and the drift branch of
(g) need a 1.5x ratio to `v2` and >= 5 of 6 seed wins. (3) any non-finite gated value in any arm -> UNDEFINED; a
non-finite dev ref -> UNDEFINED-DEV; non-finite dev rows never qualify. (4) `verdict()` requires all 7 arms on exactly
the six seeds. (5) the pick requires the full grid and both refs, else exits non-zero without a pick.json; the node
script stops on any non-zero unit. (6) prompts split 12 dev / 12 verdict, pinned by sha256. (7) one bar for dev and
verdict. (8) added `sham_tok` (real activity, rank-matched). (9) declared, and both updates' applied step norms logged.
(10) dev reports m_k / m_v; a config needs rank_min >= 4. (11) pick.json carries model and commit; run and score check
them and one (lr, tau) across the local arms. (13) `${TIMEV[@]+"${TIMEV[@]}"}`. (14) tests added: a tau 0.9 night (A
stays off U, energy >= tau), every non-finite gated field per arm, (f)/(g) on held-out alone, margin and seed count,
the token sham, the incomplete-grid refusal, mixed-unit refusals, and `run_units` (the body of `run`) on the stub
cortex with resume and pick checks.

Disagreed, with reasons:
- (8) "the told-fact episodes" as the sham text: their subspace holds the very directions the replay must move, so that
  sham fails on recall trivially and says nothing about specificity. An unrelated real bank of ordinary text would
  protect the held-out prompts much as the keep bank does, so its passing would not separate "keep-specific" from
  "ordinary-text subspace" (both are the method working). Random tokens give real activity without content; that is the
  contrast (g) needs. The random sham is kept as the floor.
- (9) not dropped and no extra arm: the keep-error step is part of the v1 loop that `noproj` must reproduce exactly (the
  lesion is "v1 at lr*"); removing it in `v2` only would make the lesion differ in two ways. Its cost is declared and its
  applied norm is logged, so a near-zero step is visible in the read.
- (10) activity stays uncentred: the adapter acts on raw x, so the mean direction is the largest part of the keep text's
  input and the biggest drift path; centring would leave it unprotected. The rank floor (>= 4) and the tau grid answer
  the "rank 1 to 3" risk instead.
- (12) the engine spec is another session's untracked file and `tests/test_engine.py` (not mine) pins it as a TODO
  skeleton; this build does not edit it. It is inert (status TODO, the engine refuses it) and flagged above.

