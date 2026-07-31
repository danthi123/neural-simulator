---
type: finding
status: live
lane: audit
date: 2026-07-31
mechanism: audit-lever-efficacy-triage
claim_check: synthesis
---

# TRIAGE of the 11 lever-efficacy artifacts: 2 load-bearing (both anti-cheat claims, both SUSPECT), 9 incidental — and the audit's own "40 hits" is a REPORT CAP, not a count

The follow-up the audit finding asked for. `2026-07-31-audit-lever-efficacy-40-identical-arm-pairs-in-banked-artifacts.md`
§4 said: *"Each of the 11 artifacts needs its owning finding read, and the question asked: was the identical pair
load-bearing for the claim?"* This is those 11 readings. **Not a new experiment** — the only compute spent was
re-executing four already-committed runners for a few steps to confirm, rather than infer, why each pair is identical.

## 0. Evidence, and how every number here can be checked

- **The audit under triage:** `research/findings/raw/_provenance/AUDIT_lever_efficacy.json` (40 hits, 11 artifacts,
  git_sha 9531a345). Gate: `tools/gates/lever_efficacy.py` (failure class 1).
- **The gate reproduces exactly.** `g.check([])` returns a set identical to the banked hits once the artifact-path
  prefix is re-attached, and `g.selftest()` returns no problems (it still catches a dead arm and still does not fire
  on distinct arms, shared floors/ceilings, quantized accuracies, or a single shared number).
- Every figure below is either quoted from a named artifact with its JSON pointer, or produced by a command printed
  in the section that uses it. Nothing here is recalled.
- **This document trips the gate, on purpose.** `g.check(["<this file>"])` returns one hit, on the §2a table — because
  §2a quotes the offending rows verbatim as evidence. That is the exact case the gate's docstring calls out as the
  reason it is NON-BLOCKING: *"the artifact that documents a dead-arm failure ... is numerically indistinguishable
  from one that commits it."* Checks run and reported: `tools/check_docs.py` W1 0 / W2 0; `tools/claim_check.py`
  11 artifacts cited, 0 missing; `tests/test_doc_rules.py` 2 passed; `lever_efficacy.selftest()` no problems.

**Scope flag, reported not worked around:** this session owns only this file, so the aggregates computed here
(the uncapped hit count, the sweep-duplicate count, the per-mode weight-delta checks) were **not banked as a new
`_provenance` JSON**. They are all reproducible from the commands given. Banking them is a follow-up for whoever
owns `research/findings/raw/_provenance/`.

## 1. The verdict table

`LB?` = does the owning finding's claim COMPARE the two identical arms?

| artifact | owning finding | identical pair | LB? | consequence |
|---|---|---|---|---|
| `_emerge6_recurrent_microcircuit_seq.json` (18) | `2026-07-02-emerge6-rung3a-recurrent-microcircuit-sequence-BOUNDARY.md` + `2026-07-02-emerge6b-rung3a-eprop-eligibility-relocalizes-wall-to-generation-stability.md` | `apical_feedback_lesion` = `no_teaching_null` = `untrained`; and `eprop_lesion` = `eprop_null` = `eprop_untrained` (all 3 seeds, both families) | **YES** | **The anti-cheat claim is SUSPECT.** Both findings enumerate lesion AND null as separate evidence; in code they are one branch, and both leave the weights bit-unchanged, so they are also `untrained`. Panel has **1** independent floor where **3** are reported. The headline BOUNDARY is unaffected. A plain re-run reproduces byte-identically — what is needed is a genuinely distinct apical lesion. |
| `_emerge49_graded_read.json` (1) | `2026-07-02-emerge49-graded-read-BOUNDARY.md` | `stacked_graded` = `dap_lesion` (all 4 metrics + every per-seed list) | **YES** | **The claim "dAP-lesion — the coincidence-plateau read is load-bearing" is SUSPECT and needs a re-run.** 3 of the 4 metrics CANNOT respond to the lesion by construction; the 1 that can is 0.00 in both arms, so the test is **UNDEFINED, not passed**. Re-run only at an operating point where the intact arm is off the floor. The headline BOUNDARY is unaffected. |
| `_emerge2_selfsup_burst.json` (1) | `research/findings/AUTONOMOUS_STATE.md` CYCLE 790 (no standalone finding) | `apical_lesion` = `no_teaching_null` (all 3 seeds, not just the 1 the gate reported) | no | The claim compares `deep_burst` to the lesion, not lesion to null. But the GO gate scores `lesion_collapses` and `null_flat` as two independent conditions when they are one arm: **4 real controls, 5 claimed**. Numbers stand. |
| `_gabor_cifar_planeauto42.json` (1) | `2026-07-07-deep-credit-real-task-cifar-fc-vision-wrong-instrument.md` | `test_fixed` = `plain_fa` | no | Verdict is STAGE-0 BOUNDARY, decided before the deep-credit arms are read; the finding's table compares `test-fixed` to the ORACLE, never to `plain_fa`. Documented, intended rate-level identity. **Forward warning below.** |
| `_gabor_cifar_raw_catdog.json` (1) | same | `test_fixed` = `plain_fa` | no | same |
| `_gabor_cifar_raw_k10.json` (1) | same | `test_fixed` = `plain_fa` | no | same |
| `_gabor_cifar_raw_planeauto.json` (1) | same | `test_fixed` = `plain_fa` | no | same |
| `_gabor_cifar_deep_credit_smoke_k3.json` (1) | **NONE** (a smoke run; outside the glob the finding cites) | `test_fixed` = `plain_fa` | no | No claim rests on it. Same structural cause. |
| `_lge_divnorm_multiseed.json` (9) | `2026-06-11-learned-graded-embedding-divnorm-readout-GO.md` | `none_diffuse_only_steps{2,3,4}_..._logclip0` = `..._logclip1` (3 seeds) | no | Grid degeneracy: at `order="diffuse_only"` the divnorm branch is never entered, so `logclip` is inert by construction. The GO's `best_key` is a different arm on all 3 seeds. **Recorded `n_variants` 132 should be 129 distinct.** |
| `_phase1_composer_routeA_smoke_seed42.json` (3) | **NONE** (the GO cites `_phase1_composer_routeA_512_seed42.json`) | `animals` = `foods` = `vehicles` | no | Not arms — shard names. No claim compares them. Real cause: under `--cortex synthetic` all shards get a **byte-identical** codebook. Confined to the synthetic smoke path. |
| `_phase1_composer_routeB_smoke_seed42.json` (3) | **NONE** (same) | `animals` = `foods` = `vehicles` | no | same |

**Tally: 2 load-bearing, 9 incidental.** Both load-bearing cases are ANTI-CHEAT claims, not headline claims — in
both, the headline is a BOUNDARY that rests on arms which are all distinct. That is the single most useful pattern
in this triage: **a degenerate control panel inflates a positive; it cannot rescue a negative.** Neither headline
needs retracting; both anti-cheat paragraphs do.

## 2. The two load-bearing cases

### 2a. `_emerge6_recurrent_microcircuit_seq.json` — "apical lesion" is an alias for "no teaching", and both are "untrained"

The BOUNDARY finding's own table gives three rows the same two numbers:

| arm (as printed in the finding) | recall_heldout | onestep |
|---|---|---|
| apical_feedback_lesion (anti-cheat) | +0.025 | −0.015 |
| no_teaching_null (anti-cheat) | +0.025 | −0.015 |
| untrained (floor) | +0.025 | −0.015 |

Those are the correct 3-seed means of the artifact (`/per_seed[*]`, mean of −0.028961/+0.148089/−0.043893 = +0.0251,
and −0.069824/+0.052007/−0.026856 = −0.0149). The verdict then reads *"the credit signal is real and load-bearing
(lesion + wrong-sign + null all collapse to the floor)"*, and the emerge6b finding repeats the pattern for the e-prop
family: *"Anti-cheats for the e-prop family all intact/load-bearing: eprop-lesion +0.04, eprop-null +0.04"* (artifact
mean 0.0371 for both). **The claim counts the lesion and the null as two pieces of evidence.**

The cause is one line of `research/runners/_emerge6_recurrent_microcircuit_seq_derisk.py`:

```python
if mode in ("apical_feedback_lesion", "no_teaching_null"):
    err = np.zeros(self.N)      # no teaching signal reaches the apical -> no learning
```

Two names, one branch. And zero error means zero `dW`, which means the weights never move, which is definitionally
the `untrained` arm (`train_mode=None`). Confirmed by execution rather than by reading:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0,'.'); import numpy as np, importlib
m = importlib.import_module('research.runners._emerge6_recurrent_microcircuit_seq_derisk')
s, T = m.make_seq_task(42, N=32, T=140)
for mode in ('recurrent_microcircuit','apical_feedback_lesion','no_teaching_null','wrong_sign'):
    for el in ('forward','eprop'):
        n = m.RecurrentMicrocircuitRNN(32, seed=42, alpha=0.7, kappa=(0.7 if el=='eprop' else 0.0), elig=el)
        W0 = n.W.copy(); n.train(s, T, mode, 20, 0.5, 42, free_run=(el=='eprop'))
        print('%-24s %-8s max|dW| = %.3e' % (mode, el, abs(n.W-W0).max()))"
```

```
recurrent_microcircuit   forward  max|dW| = 1.911e+00      apical_feedback_lesion  forward  max|dW| = 0.000e+00
recurrent_microcircuit   eprop    max|dW| = 4.047e-01      apical_feedback_lesion  eprop    max|dW| = 0.000e+00
wrong_sign               forward  max|dW| = 2.380e+01      no_teaching_null        forward  max|dW| = 0.000e+00
wrong_sign               eprop    max|dW| = 5.157e-01      no_teaching_null        eprop    max|dW| = 0.000e+00
```

**Honest reading.** This is not the classic class-1 failure (an arm that silently never ran). Both arms ran; they are
the same manipulation. The measurements are correct and the finding's GO-gate docstring even anticipates part of it
(*"no_teaching_null flat (~untrained)"*). What is overclaimed is the COUNT: there is no test of "apical feedback"
as a manipulation distinct from "no teaching at all", and neither is distinguishable from "never trained", so
"kill the credit path and it collapses to floor" reduces to "training beats not-training". The panel's discriminating
power lives entirely in `wrong_sign`, `shuffled_target` and `hebbian_selforg`, which ARE distinct and did move.

**What a re-run would and would not fix.** Re-running the same code reproduces byte-identically (the arms are
deterministic in the seed). The fix is a genuinely distinct manipulation — e.g. zero only the top-down apical
pathway while a basal teaching path still drives learning — or, if no such distinction exists in this rate model,
merge the two arm names and re-count the panel as one floor.

### 2b. `_emerge49_graded_read.json` — the dAP lesion could not have moved 3 of its 4 reported metrics, and the 4th was UNDEFINED

`2026-07-02-emerge49-graded-read-BOUNDARY.md` §"Anti-cheats (all correctly-behaving)" states: *"**dAP-lesion** — the
coincidence-plateau read is load-bearing."* The artifact says otherwise — `/onsubstrate/stacked_graded` and
`/onsubstrate/dap_lesion` agree on all four metrics AND on all three per-seed lists:

```
held_within 0.004243827160493827   held_cross 0.0019290123456790122   super_acc 0.0   l2_group 0.07366053417251675
```

Reading `research/runners/_emerge49_graded_read_derisk.py` shows why this is not a coin flip. The lesion enters only
via `_build_inherit_bridge(seed, lesion)` → `_build_cells_bridge(..., coincidence=(not lesion))`, which runs AFTER
`self.l2codon` is already built. `held_out_within_cross_overlap()` and `l2_grouping()` read `self.l2codon` and
nothing else — so **three of the four reported metrics are structurally invariant to this lesion**. Only
`held_out_super_acc()` touches the bridge, through `infer_super()`'s apical read.

And that one metric is **0.00 in the intact arm on every seed** (`super_acc_per_seed: [0.0, 0.0, 0.0]`) — with two
superordinates, `infer_super` abstained on every held-out item. A lesion applied to a capability already at zero
cannot demonstrate that anything is load-bearing: the comparison is **UNDEFINED, not passed**. This is the project's
own "every arm below chance is UNDEFINED" rule, applied to a control instead of a treatment.

**The finding's own table hid it.** The markdown reports the dAP row as held-within `0.005` and L2-group `+0.08`,
where the cited artifact gives `0.00424` (→ 0.004) and `0.07366` (→ +0.07) — identical to the `stacked_graded` row
above it. Two of four cells in that row do not exist in the artifact it cites, and the discrepancy is exactly what
makes an identical row read as a distinct one. (A `0.005` held-within for `graded_read` at `ld_wi`=0.005 DOES appear
in the finding's earlier single-seed `--diag` table, which is a plausible mis-copy source — offered as a hypothesis,
not established.)

**Consequence.** The headline BOUNDARY stands: it rests on the permanence histogram (97% near 0, mid-band 0.03) and
on super-acc 0.00 vs the 0.80 bar, neither of which involves the dAP arm, and the gate would have failed on
`acc >= 0.80` alone. What must change is the anti-cheat paragraph and the table row. A re-run is owed, but only at
an operating point where the intact arm is above floor — otherwise the lesion remains untestable by construction.

## 3. The nine incidental cases, with their real causes

**The five `_gabor_cifar_*` (test_fixed = plain_fa).** Four are owned by
`2026-07-07-deep-credit-real-task-cifar-fc-vision-wrong-instrument.md`, which cites them via a brace-glob
(`_gabor_cifar_{smoke42,catdog42,planeauto42,raw_*}.json`) — that is why a literal filename grep finds no owner, and
why the audit finding could not locate them. `_gabor_cifar_deep_credit_smoke_k3.json` falls outside that glob and is
unowned. The identity is **documented and intended**: `FANet`'s own docstring in
`research/runners/_gnw_d1_spiking_bdsp_derisk.py` says *"At the RATE level this IS numerically the microcircuit
credit"*, and `MicrocircuitBDSPNet`'s extra machinery (`W_PI`, `pbar`, the burst readout) never enters `upd`. The
identity is therefore systematic, not incidental to a run: it holds in **9 of 9** gabor artifacts, and so does the
matching `wrong_sign` = `wrong_sign_plain_fa` pair, which the gate did not flag because those arms' accuracies are
low-denominator rationals the calibration excludes.

```bash
.venv/bin/python -c "
import json,glob
for f in sorted(glob.glob('research/findings/raw/_gabor_cifar_*.json')):
    s=json.load(open(f))['per_seed'][0]['stage1_deep_credit']
    print('%-46s tf==pf %s   ws==ws_pf %s' % (f.split('/')[-1],
          s['test_fixed']==s['plain_fa'], s['wrong_sign']==s['wrong_sign_plain_fa']))"
```

No current claim is affected: every one of these runs recorded `STAGE-0 BOUNDARY`, whose text explicitly says the
deep-credit arms were not read, and the owning finding's table compares `test-fixed` held-out against the ORACLE's
linear/1-layer/deep-best columns, never against `plain_fa`. **Forward warning:** the runner's own STAGE-0-PASS
verdict string prints `plain-FA` and `test-fixed` side by side, and its `align_signal` check is
`best_test_deep > pf_deep - 0.02`. For the `test_fixed` arm that comparison is tautological. Any future reading of
these arms as a microcircuit-vs-FA contrast would be void; the real contrast is on the spiking substrate, as the
docstring already says.

**`_lge_divnorm_multiseed.json` (9 hits).** `divnorm_spreading_readout` calls `_apply_divnorm` in the `pre`, `post`
and `interleave` branches only — for `order == "diffuse_only"` it is never called, so `divnorm`, `sigma`, `exponent`
and `log_clip` are all inert and the sweep's full cross-product emits each such row twice. Per seed there are 132
enumerated variants but only **129 distinct metric-tuples**, and the 3 duplicate groups are precisely the 3 hits:

```bash
.venv/bin/python -c "
import json,collections
d=json.load(open('research/findings/raw/_lge_divnorm_multiseed.json'))
for s in ('42','43','44'):
    sw=d['per_seed'][s]['brain_based_divnorm_sweep']['sweep']
    g=collections.defaultdict(list)
    for k,v in sw.items(): g[json.dumps(v,sort_keys=True)].append(k)
    print(s, len(sw), 'variants,', len(g), 'distinct,', sum(len(v)>1 for v in g.values()), 'dup groups')"
```

The GO is untouched: `best_key` is `ch_interleave_steps2_sigma0.001_exp2.0_logclip0` on all three seeds, and the
finding's "raw diffusion (brain, base)" baseline (+0.038 mean) uses one member of a duplicate pair while the other
is simply unused. The only correction owed is the recorded variant count — best-of-132 is really best-of-129.

**The two `_phase1_composer_*_smoke_seed42.json` (6 hits).** `animals`/`foods`/`vehicles` under `graded_stats` are
shard names, not experimental arms; no finding compares them, and the routeA GO cites a different artifact
(`_phase1_composer_routeA_512_seed42.json`, learned cortex, 512 concepts). The cause is nonetheless real:
`build_ensemble_cortices` passes the run's single `seed` to `build_cortex_codebook_synthetic` for every shard, so
in `--cortex synthetic` mode all shards receive a byte-identical code matrix and only the word labels differ.

```bash
SIM_BACKEND=numpy .venv/bin/python -c "
import sys; sys.path.insert(0,'.'); import numpy as np
from research.runners.cortex_conversation_capability_derisk import build_cortex_codebook_synthetic
k=dict(D=96,dg_n_pool=300,dg_pattern_size=30,seed=42,dim=96,residual_frac=0.55)
a=build_cortex_codebook_synthetic([f'animals.c{c}_m{m}' for c in range(2) for m in range(4)],2,4,**k)
b=build_cortex_codebook_synthetic([f'foods.c{c}_m{m}'   for c in range(2) for m in range(4)],2,4,**k)
print('byte-identical codebooks across shards:', np.array_equal(a.codes,b.codes))"
```

This is confined to the synthetic path — the learned builder takes the shard's own corpus. **Forward warning:** any
synthetic-cortex run claiming per-shard independence, or reading a 3-bridge synthetic smoke as three distinct
bridges, would be void.

**`_emerge2_selfsup_burst.json` (1 hit).** `apical_lesion` and `no_teaching_null` are genuinely different code
branches here (`Y[k] := 0` vs `b := 0`), yet both freeze every hidden layer and train only the readout, because
`pbar` initialises at `p0=0.5` and both branches drive `v_api = 0`, so `p = sig(0) = 0.5` and `dev = post*(p-pbar)`
is exactly zero at every layer and every step:

```
burst_linearized  per-layer max|dW| = 3.443e-02 3.642e-02 1.510e-01   hidden frozen = False
apical_lesion     per-layer max|dW| = 0.000e+00 0.000e+00 1.336e-01   hidden frozen = True
no_teaching_null  per-layer max|dW| = 0.000e+00 0.000e+00 1.336e-01   hidden frozen = True
wrong_sign        per-layer max|dW| = 3.999e-01 1.935e-01 1.706e-01   hidden frozen = False
```

Both arms are the same control — a readout on frozen random features — which is a legitimate floor, just one floor
and not two. The recorded claim (CYCLE 790: *"deep_burst generalizes (0.72-0.80) >> ... lesion (~0.60) + linear
(0.585)"*) compares `deep_burst` to the lesion, not lesion to null, so it is incidental; the artifact gives
lesion 0.5947 and linear 0.5848, matching. The consequence is the gate's arithmetic: `lesion_collapses` and
`null_flat` are scored as two independent conditions over one arm. I checked whether they could contradict each
other (they use different reference sets) and they do **not** here — chance is 0.6295, so both bars are 0.6795 and
both pass; the recorded BOUNDARY is driven by `wrong_sign` at 0.7550 alone, exactly as the board says.

## 4. Three defects in the audit itself, found by doing the triage

**(a) "40 hits" is `_MAX_REPORT`, not a count.** `tools/gates/lever_efficacy.py` ends with `return out[:_MAX_REPORT]`
and `_MAX_REPORT = 40`. The banked audit's `total_hits: 40` is exactly the cap. Raising it:

```bash
.venv/bin/python -c "
import sys,collections; sys.path.insert(0,'.')
from tools.gates import lever_efficacy as g
g._MAX_REPORT = 100000; h = g.check([])
print(len(h), 'hits across', len({x.split('.json')[0] for x in h}), 'files')"
```

**123 hits across 80 files**, not 40 across 11. The 11 are simply the files whose hits landed in the first 40 of the
`os.walk` order — and `_emerge6` (18) plus `_lge` (9) consumed 27 of the 40 slots on their own. Unreported families
include `funcint_perception_to_memory_probe.json` (9), `gap4_p32/*` (6), `q2r_gate.json` (3),
`q2_constrained_decode_gate.json`, and ~45 in `gap5_reader/pool/*`. **The audit finding's line "None of the 11
artifacts is one of the headline 6-seed GOs" is true of the 11 but was never a statement about the corpus, and with
the cap lifted the flagged set now reaches gap#4 and gap#5 artifacts.** That is a re-scoping, not a retraction: the
finding did explicitly disclaim clearance. (Two files >2MB are skipped by `_MAX_BYTES`, so 123 is itself a floor.)

**(b) The markdown scanner produced ZERO of the banked 40.** `_scan_md` works — uncapped it returns 17 markdown
hits, including, self-referentially, the emerge6 BOUNDARY table this triage independently caught:
*table rows `['apical_feedback_lesion (anti-cheat)', 'no_teaching_null (anti-cheat)', 'untrained (floor)']` carry
identical numbers `['0.025', '0.015']`*. All 17 were truncated away by the cap, so the audit reported the artifact
side of the corpus only.

**(c) The gate under-counts within a flagged file.** In `_emerge2_selfsup_burst.json` the audit lists one hit
(`per_seed[2]`), but `apical_lesion == no_teaching_null` on **all three** seeds — seeds 42 and 43 were filtered out
by the `>= 2 informative metrics` rule, which is the calibration working as designed. Identity is sufficient
evidence of a dead or duplicated arm, never necessary; the gate's own docstring says so, and this is a concrete
instance. Per-file hit counts are a lower bound on per-file duplication.

## 5. What this does NOT establish

- **It clears nothing.** Nine "incidental" verdicts mean the identical pair is not part of the stated claim — not
  that those artifacts are sound. The gate detects identity, and identity is only the loudest signature of a dead
  lever; a 99%-inert manipulation whose arm still differs by noise passes every check in this document. That
  requires `tools/lab.py::lever`, which asserts the manipulated quantity actually moved.
- **It does not re-verify any headline.** I read each owning finding's claim and each artifact; I did not re-run any
  experiment or re-check any number outside the arms under triage.
- **One number I could not reconcile, reported as unresolved:** CYCLE 790 says *"shallow (~0.58) ... oracle
  0.98-1.0 (task-sane)"*, while `_emerge2_selfsup_burst.json` gives shallow mean 0.5014 and oracle mean 0.8745
  (per-seed 0.662 / 0.984 / 0.977 — seed 42 is far below the quoted band). The board sentence explicitly aggregates
  *"the fixed run + the bottleneck sweep"*, so the figures plausibly come from sibling artifacts I did not locate.
  **Not established either way**; flagged for whoever owns that arc.
- **No new `_provenance` JSON was banked**, per the file-ownership scope in §0.

## 6. Next

1. **Correct two anti-cheat paragraphs** (emerge6 + emerge6b: 3 floors are 1; emerge49: the dAP claim is undefined,
   not passed) and **one table row** (emerge49's dAP row, to the artifact's 0.004 / +0.07). Both headlines stand.
2. **Raise or remove `_MAX_REPORT`** in `tools/gates/lever_efficacy.py` and re-bank the audit — or, if 40 is a
   deliberate readability cap, record the true total beside it so `total_hits` stops reading as a corpus count.
3. **Triage the 69 newly-visible files**, starting with the gap#4 and gap#5 artifacts, which the capped run hid.
4. **Two forward warnings to carry**, neither a current defect: gabor `test_fixed` vs `plain_fa` is tautological at
   the rate level by design, and `--cortex synthetic` gives every shard the same codebook.
