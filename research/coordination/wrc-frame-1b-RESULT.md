# [wrc-frame-1b] RESULT — engagement diagnosis (RULING wrc-frame-1-1024(2))

STATUS: PROVISIONAL (agent finding; the scorer/Claude owns the verdict).
SCOPE: DIAGNOSE-ONLY (one MiMo Pro ruling-model delegate decided scope).
No sim/ edits, no parameter retune, no fix executed.

METHOD. New runner `research/runners/wrc_frame_step1b_diag.py`: measurement-only
class-level wraps (pass-through; answer rules in-module, committed before the
run). Tiers: full per-step instrument seed 7; light per-presentation seeds
1-15; isolated rheobase titration; step-1 re-run UNCHANGED (seeds 1-15, new dir
step1b_rerun). All local under mem_ok 4, SIM_BACKEND=numpy CPU, 2.8 s/seed.

(a) IS T->role IN THE PLASTIC MASK, DOES THE RULE TOUCH IT?
- plan marks it plastic with its own gate: organ:401-403
  `plastic=True, plasticity_gate='wrp_trole_plastic'`.
- registered 2736/2736 expected synapses (exact); the bound check passed
  (bridge:5490 `indices.max() < gain nnz`, gain array 15824) — no silent skip.
- enforcement OFF: `enforce_plastic_mask_in_hebbian` default False (config:472),
  env unset; mask built but not enforced (bridge:1259-1271); reads True over
  the trole range → NOT mask-excluded.
- rule touch (causal pre(t-1)&post(t), bridge:10333-10346, gain at :10341):
  seed 7 = 1 update step out of 5760 train steps, sum|dw|=6.01, 3 wires moved
  at end; seeds 6/9 = 244.7/246.5; 12/15 seeds sum|dw|=0. word->role (same
  rule, same gain): 3595 steps, sum|dw|=18581.7, 1373 wires — robust.
- SUB-CLASS: ACTIVE-SET STARVED. Max 3 active trole edges in one step, mean
  0.00052/step, vs word 4.05/step: the trace pop (pre side) almost never fires
  at t-1 while a role fires at t during training. The rule can reach T->role;
  it essentially never gets the chance to fire on it.
- NOTE vs [wrc-frame-1]'s narrative "0/2736 frozen at 0.05": that count is not
  stored in step-1's json (no moved-count field there), so it is not
  re-verifiable from artifacts; under this instrument T->role is starved, not
  literally immobile (3 moved wires on seed 7).

(b) GATE OPEN AT THE CO-FIRE STEPS? — YES. Open calls at value 1.0 before
  training (organ:421 `_ensure`, organ:481 `_train`); read-back min gain over
  the range = 1.0 at EVERY training presentation, 15/15 seeds; the only close
  is post-train (organ:489, presentation 144/144) — the designed freeze. No
  second closer anywhere. frame params carry no PLAST_GAIN key → 1.0 default.

(c) DISCHARGE DRIVE vs RHEOBASE? — NOT sub-rheobase (the cs-race mention
  lesson does not apply here). Isolated titration: I_rh = 100 pA (0 spikes at
  0/50 pA, first spikes at 100 pA, monotone to 800 pA). Arriving drive =
  commit-gate spikes at t-1 x W_GD 200 pA (organ:404; 288 edges, all 200):
  train max 1200 pA, parse max 800 pA; across seeds 800-1400 pA = 8-14x
  rheobase. Discharge SPIKES: 5770 during training, 1601 during the batched
  parse probe (seed 7). Per-word intact runs (step-1's arm): 1/12 words fire
  (brain@41); batched probe 5/12 — the difference is batching (state carries
  across words in one run_frame call); both fire, so neither arm is silent.

FIX-SITE CLASSIFICATION (per ruling (2)(c): the fix must be WIRING, never a
retuned gain): NOT the mask, NOT the gate, NOT gain, NOT registration, NOT
sub-rheobase drive. FIX-SITE = trace->role active-set starvation: presynaptic
trace cells essentially never fire during training while the taught role
fires, so the causal rule never selects T->role. OPTIONS (NOT executed):
- O1 wiring/drive: strengthen word->trace engagement during TRAIN so the trace
  pop fires while the role is taught (matches the ruling's wiring class).
- O2 follow step-1's own packet fallback: AMPA frame WITH discharge (never
  drop discharge), then re-test engagement.
- O3 presentation timing: align teach onset so a role spike lands at t right
  after trace at t-1 (schedule change — prereg/packet-owned, needs a rule).

IDENTITY RE-RUN (ruling: "re-run step 1's two measurements and post RESULT"):
  step-1 run UNCHANGED (same code, seeds 1-15, new dir step1b_rerun):
  compared all stored metric fields (a_cofire_present, ab_singles, b_gain_mean,
  b_gain_present, c_*) across 15/15 seeds → mismatched=0; verdict reproduced:
  co-firing 15/15, gain 0/15 → FALLBACK (AMPA frame with discharge).

FILES. research/runners/wrc_frame_step1b_diag.py; raw/_wrc_frame/step1b/
  (16 artifacts + prov); raw/_wrc_frame/step1b_rerun/ (16 + prov).

NEXT. [wrc-frame-1b] NEEDS-RESULT posted on the runbook; Claude rules O1-O3,
  or parks word-role frame per batch ruling (9)/(2) after this diagnosis.