# Draft: xedge-prov-surprise-reciprocal — WORKLOG

## 2026-10-04 10:3x — §5 step 1: choice and why (local model)

Section 4 had nothing to move (duties done; all other tasks blocked or landing-bound), so §5 (Idle: self-directed
draft research) applies — ONE draft at a time.

**Chosen topic:** close the missing reciprocal of the surprise <-> source_provenance pair: declare
**source_provenance -> surprise** (the provenance verdict feeding back into the surprise read) as a declarative
`CrossEdge` through the same generic `onebrain_crossedge_gate.run_gate` used by the prior reciprocal
(`2026-09-01-onebrain-crossedge-provenance-to-selfschema-reciprocal-GO.md`).

**Why this one (first-open rule):**
- `GAP_CLOSURE_MISSION.md` CURRENT STATE (10-02) has no idle-draft next-item (its "Next command" = runbook §4),
  so the fallback is the master roadmap's literal "next to queue": `docs/plans/2026-07-23-MASTER-DEVELOPMENT-
  ROADMAP.md:46` — *"NEXT (queued, not yet built): another reciprocal pair from the same connectome
  (surprise->source_provenance is still one-direction only), OR the production wire-in rung"* — the wire-in rung
  is a default flip (Claude-reserved), so the reciprocal pair is the driver-open candidate.
- The one-direction fact verified in code today: `research/runners/_onebrain_integration_surprise_episodic_
  crossedge.py:41` declares `merge_organs([SURPRISE_LITE, source_provenance], wire=True, cross_edges=[surprise
  ->prov...])`; no source_provenance->surprise row exists (grep of all `CrossEdge(` rows).
- Precedent class: functionally-related areas connect reciprocally, not all-to-all (Magrou 2024; Gamanut 2018;
  Theodoni 2020 — quoted in the provenance<->self_schema reciprocal GO's §1); R1 (comprehension<->d6 WM) and R4's
  second half are the two completed pairs before this.

**Preference check (§5 step 1):** integration-first-leaning (a learned edge whose lesion is tested; stays OFF by
default — the ON-by-default production wire-in stays Claude's, recorded). Stage-1 (10-24) relevance: indirect
(source-monitoring/surprise sit beside the C1-C3 seams); marked honestly.

**Record research done (step 2, first pass):** `before_you_build.sh` recorded (corpus log line; pointers:
`_source_monitor_attractor_competition_scoping.md`, `_source_monitor_heterosynaptic_encoding_scoping.md`,
`_route_b_onebrain_seam_scoping.md`, ...); `rag_search.py ... --corpus all` top-5 = the reciprocal-GO finding
(4 hits) + the cross-edge framework gap-analysis (already-closed, no Phase-2 layer to build); read the
reciprocal-GO finding §1 (rationale + the generic-gate declaration recipe: 1-row CrossEdge + train_fn + read_fn +
two conditions) and the phase-DESIGN R3 entry (surprise -> episodic/provenance encoding gate, Lisman-Grace;
`docs/BURN_DOWN_LIST.md:146`).

**Next (step 3):** PLAN.md with one MiMo Pro delegate (question, biology, dev seeds 7 + 1-15 only,
outcome-to-reading table BEFORE any run), then a second delegate tries to refute it against CLAUDE.md's
non-negotiables; revise until it approves. Then step 4 build in `$WT/xedge-prov-surprise-reciprocal` on
`research/draft-xedge-prov-surprise-reciprocal` (new `_xedge_*` runner copy; never edit sim/, a registered runner,
a scorer or a gate), run per RULING compute-placement (pool/workstation, never this server >10 min), step 5 =
PROVISIONAL finding + verify-go + routine-merges + NEEDS "[draft-...] RESULT: ...". A draft never claims a GO;
gate seeds / preregs / scorers / default flips / sim/ stay Claude's.

## 2026-10-04 10:5x — §5 step 3: PLAN drafted, refuter verdict REVISE (8 fixes queued)

MiMo Pro delegate #1 (deleg_15ee3fb0) returned the full PLAN; written to PLAN.md (question, biology, dev seeds
[7,1,4,9,12,15], 7-row outcome-to-reading table, sign constraint: amplification edge prov_perceived -> surprise
+ attenuation as control read only, variant B pre-declared).
MiMo Pro delegate #2 (deleg_dc6a7b54) = refuter verdict **REVISE**, 8 required changes: (1) qualify
'load-bearing' as 'trace-level' (TERMS.md 09-29); (2) row 1 must not spend the governed word 'closed'; (3) fix
miscite Fremaux & Gerstner (three-factor) for the two-factor symmetric rule (grounding: sim/config.py
hebbian_symmetric, Kandel 6e Ch 54); (4) mandate the 2026-09-02 C2 per-neuron-state reset (MergedPool
_PER_NEURON_STATE) between reads + threshold-restored smoke check + counterbalance/order-invariance THIS run;
(5) row 3 must not assert a cause it cannot establish (bank honest negative if instrument clean); (6) add
lesion persistence-at-read check (slots read exactly 0 at measurement, learning OFF); (7) split JHL 1993
confidence tag (loop framing = precedent's extension, not JHL's); (8) pre-declare INTACT_FLOOR rule now
(never set post-smoke). Refuter verified the sign constraint TRUE in code (sim/bridge.py:1240-1257 clip) and
everything else compliant.
Next wake: revise PLAN.md per the 8 fixes, re-run refuter until APPROVES, then step 4 (build). No run yet.

## 2026-10-04 11:2x — §5 step 3: PLAN revised per refuter's 8 fixes

Applied all 8 to PLAN.md: (1) 'load-bearing' qualified 'trace-level' at every occurrence (probe, rows 2-3,
§6 check 3); (2) row 1 no longer spends the governed word 'closed' (reciprocal feedback loop at the synapse
level, NOT a closed capability, default-OFF wire-in=Claude); (3) plasticity citation fixed to the TWO-factor
symmetric rule grounding (sim/config.py hebbian_symmetric ~869-873; Fremaux & Gerstner kept only as the
three-factor family this rule is the cousin of); (4) risk 4 rewritten: MANDATORY Port-A reset via
MergedPool._PER_NEURON_STATE (onebrain_merge_framework.py:246-250) + smoke threshold-restore check +
read-order COUNTERBALANCED THIS RUN; (5) row 3 reworded (only 'not carried by the declared edge'; instrument
checks first, bank honest negative if clean); (6) §6 check 3 gains persistence-at-measurement (slots read
exactly 0 at every lesion re-read, learning OFF); (7) JHL confidence split (loop framing = precedent's
extension, not JHL's); (8) INTACT_FLOOR rule PRE-DECLARED (0.5 x smoke seed-7 delta, frozen before 6-seed
queue, never revisited).
Standing duties: T0 clean (flip 9q+8r draining, c1 54q, b2d 43q, iw1-R 2q+8r, stage2 8q); no new NEEDS/rulings.
Next: refuter re-run until APPROVES, then step 4 build.

## 2026-10-04 11:4x — refuter round 2: REVISE (3 fixes) applied

Round-2 refuter (deleg_a71022d6) confirmed 7/8 original fixes land correctly; 3 remaining: (a) row 5's bare
'load-bearing' -> 'trace-level load-bearing' (TERMS stated-result site); (b) §8 fail-gate didn't cover new
item (e) -> added: (e) failure at smoke = no floor freeze, no queue, report per row 6; (c) risk-4 line span
stale (246-250 -> 254-259, verified by refuter in code). All 3 applied. Next: round-3 refuter for APPROVES.

## 2026-10-04 11:5x — refuter round 3: APPROVES — §5 step 3 COMPLETE

Round-3 refuter (deleg_b1d38151): APPROVES, zero required changes. Verified via git show fa119e951 + live
reads: all 3 round-2 fixes landed (row 5 'trace-level load-bearing'; §8 dual fail-gate coherent with row 6;
_PER_NEURON_STATE span 254-259 = real definition site, 4 fields present). Non-negotiables hold; W2 line-length
clean. One pre-existing cosmetic dangling paren at L179 noted as out-of-scope (non-semantic).
Refutation arc: round 1 REVISE (8 fixes, deleg_dc6a7b54) -> applied -> round 2 REVISE (3 fixes,
deleg_a71022d6) -> applied -> round 3 APPROVES (deleg_b1d38151). Planning delegates: 3 this session
(total session delegates now 9).
STEP 3 DONE. Next wake = step 4: build `research/runners/_xedge_prov_surprise_reciprocal.py` in
$WT/draft-xedge-prov-surprise-reciprocal on research/draft-xedge-prov-surprise-reciprocal (new file copy,
sim//registered/scorers/gates untouched), embed the outcome table in the runner docstring first, then smoke
seed 7 on pool/workstation per compute-placement.

## 2026-10-04 12:3x — §5 STEP 4: BUILD (branch research/draft-xedge-prov-surprise-reciprocal)

Worktree $WT/draft-xedge-prov-surprise-reciprocal off main; branch research/draft-xedge-prov-surprise-reciprocal.
Runner research/runners/_xedge_prov_surprise_reciprocal.py (516 lines): reshaped from the 2026-09-01
reciprocal precedent (_onebrain_crossedge_provenance_to_selfschema — generic GATE_SPEC/run_gate/byte-off
harness) onto template-2's calibrated SURPRISE+source_provenance organ pair (_onebrain_integration_surprise_
episodic_crossedge — mismatch-trial drives, CUE_PA=2000, LR=0.15, HMAX=20, N_EPISODES=150, Port-A
_EXTRA_RESET_ARRAYS). Edge FLIPPED to the draft's declared direction: CrossEdge prov_perceived -> surprise
(gate provper_to_surprise), full-mask (no concept blocks on the prov side). Train = CONTRADICT trial +
ctx_perceived + perceived-exemplar content, Hebbian learning on; read = same trial + verdict HELD through the
window, N_READS=8, surprise-region windowed mean rate; delta = perceived - generated (sign +1).
Docstring = table-first: all 7 outcome rows embedded, MUST-NOT-CLAIM block, sign constraint + variant B,
dev seeds 7,1,4,9,12,15 as the --seeds default (smoke = seed 7 only; gate seeds never run here).
PROVISIONAL-only strings (tags PROVISIONAL-SUPPORTS / NOT-SUPPORTED / smoke indicator; payload key
provisional_supports; probe xedge_prov_surprise_reciprocal). INTACT_FLOOR = 0.005 PLACEHOLDER with the
pre-declared smoke rule (0.5 x smoke delta) in the comment + printed by --smoke. Smoke adds the
order-invariance check (a->b->a, drift must be <=1e-12; FAIL => do NOT queue). Selftest = read-isolation
fails-in-its-failing-direction guard (seed 7). Clean-zero control checked per seed (perc rate < 0.005).
Verified: py_compile OK; import-only check OK (GATE/SPEC/expected/docstring seeds confirmed; no pool build,
no seeds run). NOT YET RUN — smoke/selftest next, and per RULING compute-placement they do NOT run on this
server: queued to the pool (numpy CPU) next wake.

## 2026-10-04 13:3x — §5 step 5 ATTEMPTED: selftest STOP (2nd failure, §2) + pool add refused

Selftest attempt 1 (S, mem_ok 23G avail; memcap refused — no systemd --user, known §2 skip, ran UNCAPPED
accepting OOM risk, job stayed <1G): FAIL — fix-disabled probe DIVERGED=False (both reads 0.00333...) AND
fix-enabled identical=True => ok=False. Root: my teeth design (induction contradict+ctx drive, then two
back-to-back full-trial read_verdict) — with the Port-A restore off, the two reads still agree because the
strong trial drive reproduces the same extra-array end-state each call (contractive), so the probe cannot
fail-in-its-failing-direction.
Fix round (one allowed): attempt 2 = template-2-style WEAK probe (episode+ctx only, edge lesioned, untrained,
no trial): TEETH_MISSING — surprise=0.0 both arms, because surprise ONLY fires on the mismatch trial; degenerate.
SECOND failure on the same issue => §2 "one fix round per issue; a second failure is a STOP". STOP recorded;
runner left as committed @ 62a83bbbbe (its --selftest honestly returns FAIL). Needs Claude: either port
template-2's amb_read margin probe verbatim (needs the ambiguous-pattern machinery dropped from this port) or
rule the read-isolation guard optional for this draft. NEEDS line filed on main.
Pool: before_you_build corpus check ran 13:2x (logged .git/corpus_checks_shared.jsonl); smoke line SHAPE-CHECKED
queue_job_shape_check RC=0; pool_queue add REFUSED twice (ADD_RC=2): "no reachable pool node" — wspool
unreachable (GAME_MODE / workstation paused), pool40 BROKEN (code missing), pool41/42 healthy but not yet
provisioned for revision 62a83bbb.... Touched research/queue/provision.req/62a83bbb... (iw2-provision flow);
/host-status/sim-provision/62a83bbb*.json not present yet — retry the ready-to-add line next wake; node
restore/provisioning beyond the req file is Claude's if it does not land. NEEDS line filed on main.

## 2026-10-04 13:36 — §5 step 5b RETRY OK: smoke QUEUED on the pool

Provisioning landed 13:31 (host-status /opt/host-status/sim-provision/62a83bbb....json; pool41 pool42 pool40 ok,
wspool skipped — GAME_MODE backfill). Retried the shape-checked line verbatim: pool_queue add ADD_RC=0,
"queued (depth now 108)". Job: cd ~/derisk-pool/revisions/62a83bbb... && mem_ok 6 -> memcap 8 ->
.venv/bin/python -u -m research.runners._xedge_prov_surprise_reciprocal --smoke --out
research/findings/raw/_xedge_prov_surprise_reciprocal/smoke_seed7.json (dev seed 7, numpy CPU, --checked record
sentence embedded). Duties: T0 no-hold clean; T9 fresh-queued=43 >= 10 -> no new stub wave; yield retried ->
still refused GAME_MODE (8th; standing NEEDS 06:45). Selftest STOP remains open (NEEDS 13:35, Claude's call) —
smoke outcome still readable per row 1-4/6/7 once the file lands; INTACT_FLOOR freezes ONLY if smoke sign = +1
and the selftest guard question is resolved before the 6-seed queue.

## 2026-10-04 13:44 — §5 smoke attempt 1 FAILED (rc=1, 33s) -> REQUEUE-ONCE with stderr capture

Evidence: /opt/host-status/sim-pool-jobs/exits.json entry {node pool40, ts 1791135275, rc 1, out
research/findings/raw/_xedge_prov_surprise_reciprocal/smoke_seed7.json, rev 62a83bbb..., reason xedge};
claims tail shows dispatch ts 1791135242 -> lifetime 33s = startup failure (not compute). First-attempt
stderr was NOT redirected, so no traceback on S (pool dispatcher logs are remote-only). No output file.
Tried: claims/dispatch/exits/logs inspection (dispatch.log has dispatch only; agent_gateway/dashboard: no
match); /opt/host-status/sim-pool-jobs/ holds exits.json only.
Fix-round 1 for THIS issue: REQUEUED ONCE — same line verbatim + `2> research/findings/raw/
_xedge_prov_surprise_reciprocal/diag_smoke_seed7.txt` (stderr syncs back with raw/ via pool_sync; the CA3
10-03 merge-diag pattern). Shape-check RC=0; add ADD_RC=0, depth 108. Candidates if it fails again: mem_ok
refusal on a busy pool40 (33s ~= two-retry window; mem_ok.sh 6 2), pool-side memcap, or import/arg error —
stderr will discriminate. SECOND failure on this issue = STOP (NEEDS Claude) per §2.

## 2026-10-04 13:56 — §5 smoke attempt 2 FAILED -> SECOND failure = STOP (§2); root cause IDENTIFIED

Attempt 2 (the stderr-capture requeue) also rc=1: exits ts 1791135757 pool40. The diag file landed and syncs:
research/findings/raw/_xedge_prov_surprise_reciprocal/diag_smoke_seed7.txt (1517 bytes, first-attempt stderr —
the file existed from the FIRST run's redirect? No: written 13:42 by attempt 2's dispatch). Traceback:
  run_seed -> run_gate(pool, GATE_SPEC)
  onebrain_crossedge_gate.py:218 cross_edge_masks(pool.bridge, spec.cross_edges)
  AttributeError: 'ProvSurprisePool' object has no attribute 'bridge'
ROOT CAUSE: the generic harness contract is pool.bridge (the merged brain); my port kept template-1's
attribute name pool.b (template-1 ran its own bespoke checks, not run_gate). One-line fix: expose
@property bridge -> self.b (or rename b to bridge throughout). memcap enforced fine on pool (MemoryMax=8G,
pool-sandbox scope) — NOT a memory issue; the 33s/2min lifetimes are import + first pool build then crash.
§2: one fix round used (the diagnostic requeue), second failure => STOP. Runner left as-is on the branch;
needs Claude to land the one-liner (or delegate the fix back to me). NEEDS filed on main. Both smoke
attempts produced NO smoke_seed7.json; INTACT_FLOOR remains unfrozen.

## 2026-10-06 05:4x — RULING xedge-bridge executed (Claude's fix round, §2 count restarts)

- `git apply /home/dant123/Projects/neural-simulator/docs/drafts/xedge-prov-surprise-reciprocal/CLAUDE-2026-10-04-bridge-and-selftest.patch`
  in this worktree: full apply REFUSED on the `research/findings/raw/_provenance/runs.jsonl` hunk (local copy
  diverged — pre-existing ` M runs.jsonl`); re-applied with `--exclude` on that path -> runner hunks applied.
  Verified `self.bridge = self.b` at `_xedge_prov_surprise_reciprocal.py:171` (+ selftest teeth hunks).
- Commit `a8619b8f2` on research/draft-xedge-prov-surprise-reciprocal, pathspec-limited to the runner file
  (§2: never `git add research/findings/raw/_provenance/runs.jsonl` — left modified-uncommitted, as before).
- Next per the ruling: diagnose byte_off False (equal synapse counts 240322, values differ) and clean_zero False
  (prov_perceived 0.01 under "generated", threshold <0.005 at runner line 364) BEFORE any 6-seed queue.

## 2026-10-06 05:5x — provision for the patched tip REFUSED stale; diagnosis progress (code-side)

Provision: touch research/queue/provision.req/0add7f3cb82ebbbae1a55623b006a29e17bbfa1f -> landed 75s,
state="refused", reason verbatim: "stale: does not contain published main 871bc551498b or a main published in
the 60 min before this request, and main changed more than docs since"; hint "Merge or rebase onto main, commit,
and name the new sha right away." reachable_via= research/draft-xedge-prov-surprise-reciprocal (the sha IS on
the branch). Merge/rebase = SECTION-2 RESERVED (same class as [ldc-provision]) -> NEEDS-CLAUDE line filed on
main; smoke line NOT queued (no reachable revision; the 10-04 pattern: add only after the status lands ok).

Diagnosis in progress (the ruling: diagnose BOTH before the 6-seed queue) — code-side, run pending on the
smoke seed-7 JSON:

byte_off (observed False; equal synapse counts 240322, values differ): verify_byte_off
(onebrain_crossedge_gate.py:191-208) excludes the declared (pre,post) slots from the WITH side only and requires
k_with_base == k_without as exact {key: value} maps. The descriptors in this draft have NO explicit_wiring_fn
(grep: only CROSS_EDGES at :114), so the slots' presence in the no-edge pool must come from the merge's base
wiring plan (onebrain_merge_framework.py:433 rm.build_wiring_plan, per_region_seed=True) carrying the
prov_perceived->surprise anatomical pair at base values — cross_edges only OVERWRITES/annotates the same
support (union[ce.key] at :444-445). That fits ALL THREE observations: n_with == n_without (240322), slot
values differ (W0/plastic vs base), and PASS=False (k_without retains the excluded slot keys). HYPOTHESIS,
pending the smoke's n_with/n_without/n_with_base triple to confirm; if confirmed, the fix question (make the
no-edge build drop the anatomical pair, or make verify_byte_off exclude slots from BOTH sides) is Claude's.

clean_zero (observed perc_rate 0.01 >= 0.005, runner :364): read_verdict("generated") drives
patterns["generated"][0] + ctx_generated (:312-313) and reads prov_perceived, and run_seed calls it AFTER
run_gate (:355-359) — i.e. on the trained + interaction-lesioned pool, not a resting one. Candidates: (a)
prov_perceived baseline/spontaneous rate above the 0.005 bar under any drive; (b) content/ctx leakage from the
generated exemplar's drive into prov_perceived; (c) post-lesion state shift. Needs the smoke JSON plus a
per-region rate dump to discriminate; NOT resolvable by code reading alone.

Both checks block the INTACT_FLOOR freeze and any 6-seed queue (PLAN row 7: clean-zero FAIL = leaky control;
row 5: byte-off FAIL = CONFOUND). Next wake after the rebase: provision -> record smoke -> shape-check -> add.

## 2026-10-06 06:1x — sync-branch unblocked the provision; smoke QUEUED

RULING sync-branch applies verbatim here (provisioning refused for published main -> bring main in myself):
`bash tools/agent_sync_branch.sh research/draft-xedge-prov-surprise-reciprocal` in $P -> merged main @
2c5b8a45a as ada33408de961b8480de03c3f3c72e7e6c40b14c (every gate run, RC=0; only tool-health-fresh warn).
Provision: touch research/queue/provision.req/ada33408d... -> running (attempt 1/3, 7095 files 112 MiB) ->
state=ok after ~4 min: "provisioned on pool41, pool42, pool40, wspool".
Smoke line (shape RC=0, ADD_RC=0, depth 19): cd ~/derisk-pool/revisions/ada33408d... && mem_ok 6 -> memcap 8 ->
env SIM_BACKEND=numpy ... python -u -m research.runners._xedge_prov_surprise_reciprocal --smoke --out
research/findings/raw/_xedge_prov_surprise_reciprocal/smoke_seed7.json; checked reason carries max_age_h=24,
the PLAN citation, and both diagnosis targets (byte_off, clean_zero); no gate seeds (dev seed 7 CALIBRATION).
[xedge-provision] NEEDS line deleted on main (condition "once the branch carries main" met). Next wake: read
smoke_seed7.json -> complete the byte_off/clean_zero diagnosis; INTACT_FLOOR freezes ONLY per PLAN rule and only
if the guard question is settled; a second failure of either check remains a STOP (fix-round 1 restarts fresh).

## 2026-10-06 07:1x — round-1 smoke DIAGNOSED (MiMo Pro verdict + empirical confirm); fix landed

Round-1 smoke (pool40, 96.9 s, rc 0): emergence PASS (edge 1.6923, drift 0), interaction PASS (delta 0.005833
> floor 0.005, frac_attributable 1.0), byte_off FAIL, clean_zero FAIL (0.01 vs 0.005) -> provisional_supports
false (correct per PLAN table: instrument first). Round-1 JSON kept as smoke_seed7_round1.json(+.prov).

Diagnosis (ONE MiMo Pro delegate, verified by code, cites checked):
1. byte_off = PIPELINE-STAGE MISMATCH in my run_seed, not an RNG/merge leak: bridge_with was
   ProvSurprisePool(seed).b (class init runs the sp-organ build-time Hebbian encode: episode->prov_perceived/
   prov_generated/content_readout potentiate from W_EP_PROV_INIT=0.0, PERSISTING - read_isolation restores
   per-neuron state, not cp_connections), bridge_without was the bare pre-encode _noedge_bridge. Inject path
   verified verbatim (initial_weights copied unchanged, cross-edge appended last, _cross_edge_dense is a
   deterministic W0 fill) -> EMPIRICALLY CONFIRMED this wake, local, 11 s, mem_ok 28G (memcap unavailable on
   this host - systemd --user missing - ran under mem_ok only per skip rule):
   A: pure-with vs pure-without -> PASS True, n_with_base 240322 == n_without 240322, byte-identical.
   B: ProvSurprisePool(seed).b vs pure-without -> PASS False, SAME counts (240322/240322), values differ =
   reproduces round 1 exactly. Fix = MINE (draft runner): run_seed now compares _build_pool(seed,
   with_edge=True).bridge (like-for-like, both pre-encode).
2. clean_zero = MIS-PREMISED bar + missing record (mine), live leak candidate (laneC graded trace):
   read_verdict's shared drives verified NOT to reach prov_perceived (only episode->prov W plastic,
   ctx_perceived->prov W=55, inh_generated->prov); generated[0] shares overlap_k=3 episode neurons with
   perceived[0] potentiated during encode; 0.005 appears NOWHERE in PLAN (only runner:364); precedent's
   silence bar = NOVEL_MAX_RATE 0.02 under a LIGHTER protocol. run_seed now records gen_rate too (was
   perc-only -> leak-vs-tonic unmeasurable). Bar rule = PLAN gets a PRE-DECLARED calibration rule (INTACT_
   FLOOR 0.5x style, calibrated on dev gen/perc) BEFORE the 6-seed queue, never post-hoc; if the leak is
   then confirmed inside laneC encode machinery = registered = Claude's.
Next: re-queue seed-7 smoke on the fixed head (byte_off should PASS end-to-end; gen_rate lands); write the
PLAN calibration rule from that data; only then INTACT_FLOOR freeze question + any 6-seed queue.

## 2026-10-06 07:2x — fixed head provisioned (00beae8a2) + smoke RE-QUEUED

Provision touch research/queue/provision.req/00beae8a2c67822b9e46264e3c3c33a0dcdd8d06 -> running ~11 min
(busier than ada33408d's 4 min; wave-3 fleet load) -> state=ok, published_main df182050a, "provisioned on
pool41, pool42, pool40, wspool". Requeued the seed-7 smoke line at the fixed tip (shape RC=0, ADD_RC=0,
depth 57): byte_off now compares like-for-like pure builds (locally verified PASS: n_with_base 240322 ==
n_without 240322 byte-identical, +9216 edge slots only) and clean_zero records gen_rate alongside perc_rate.
Hold: INTACT_FLOOR freeze + any 6-seed queue until the re-smoke lands and the PLAN clean-zero calibration
rule is pre-declared from its gen/perc data.
