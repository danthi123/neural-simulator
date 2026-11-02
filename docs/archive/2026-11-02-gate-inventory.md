# Gate inventory, 2026-11-02 (lean-method)

Why: the owner asked whether the many gates slowed the work. Measured on a one-file commit (a new finding with
frontmatter and no numbers): the old pre-commit took **44 s** and BLOCKED it (doc-type wanted a `type:` line); the
lean hook takes **~1.2 s** (about 6 s on a cold cache) and passes. `operating_point`'s selftest alone took 35-54 s, because
the registry runs every selftest on every commit.

Columns. **cost** = selftest s / check s on that commit (the registry runs both). **evidence** = commits since
2026-07-01 naming the gate / commit messages saying it blocked or caught something / FAILURE_LOG mentions. These
are rough grep counts, so a high count can mean "much edited", not "often caught". **unrelated state** = blocks
on something other than the commit's own content.

Decision rule: keep a check only if it is cheap, reads the commit's own content, and guards costly or
irreversible harm or an invalid scientific claim. Most archived gates guard the frozen old track (runners, flips,
pool lanes, boards), block on unrelated state, or have no record of catching anything. Archived code is kept at
`tools/gates/_archive/` (importable as `tools.gates._archive.<name>`, never run); its tests are in
`tests/_archive_gates/` (not collected). To restore a gate: `git mv` it back and show it meets the rule.

## Inline checks in tools/githooks/pre-commit

| check | what | cost | unrelated state | decision |
|---|---|---|---|---|
| agent merge backstop | uid 99 (Hermes) on main may commit only routine classes (`tools/agent_merge_policy.py`) | ~0 s, server only | no | keep |
| G1 `check_docs.py` | doc line length (W2) and retraction marking (W1) over all governed docs | 1.5 s | yes (whole corpus) | removed from pre-commit (still run by `tests/test_doc_rules.py` and `pre-merge-commit`) |
| G2 `claim_check.py` | numbers in a new finding exist in an artifact it cites | 0.05 s | no | keep |
| G3 `biology_check.py` | biology bindings resolve; config does not contradict them | 3.8 s | yes (all bindings) | keep, but only when the commit touches `research/biology/`, `research/runners/` or `sim/` |
| G4 finding status | a new finding declares `status:` | ~0 s | no | keep |
| G5 registry | runs `tools/gates/*.py` | was ~60 s, now ~1 s | n/a | keep |

## Registry gates

| gate | what (docstring start) | cost s | evidence | unrelated state | decision |
|---|---|---|---|---|---|
| `agent_floor_persistent` | CLASS UA — fewer than the agent floor running, PERSISTENTLY. Report-only (why NOT blocking, below). RELATIONSH | 0.0 / 0.0 | 2 / 0 / 1 | agent count | archive  |
| `agent_parallelism` | CLASS AP — PENDING WORK SERIALISED while dispatchable agents sit unused. BLOCKING. WHY (owner, 2026-07-31): "p | 0.0 / 0.0 | 2 / 0 / 3 | workboard state | archive  |
| `attribution_required` | CLASS AT — A TREATMENT/CONTROL PAIR MEASURED, THE DIFFERENCE NEVER ATTRIBUTED. THE DEFECT. A runner computes a | 0.0 / 0.01 | 12 / 2 / 3 | no | archive  |
| `bar_sanity_line` | CLASS BAR — a pre-registered bar nobody checked could be passed (or failed). EVIDENCE (2026-10-06). Two bars c | 0.05 / 0.02 | 2 / 0 / 5 | no | archive  |
| `below_chance` | CLASS BC — EVERY ARM BELOW CHANCE, reported as a NEGATIVE. The verdict is UNDEFINED, not a NO-GO. THE FAILURE. | 0.0 / 0.0 | 18 / 3 / 2 | no | archive  |
| `board_sync_on_status_change` | CLASS BS — a faculty's production STATUS changed but the single-pane-of-glass board was left stale. WHY (owner | 0.0 / 0.01 | 7 / 0 / 1 | board sync marker | archive  |
| `boundary_verdict_external_check` | CLASS BV — A 'FUNDAMENTAL LIMIT' VERDICT BANKED WITHOUT READING THE FIELD THAT REFUTES IT. THE DEFECT, measure | 0.0 / 0.0 | 4 / 0 / 3 | no | archive  |
| `cfg_phantom_field` | CLASS PF -- a runner writes `cfg.<field>` under a name CoreSimConfig never declares, so the engine silently ne | 0.0 / 0.0 | 1 / 0 / 1 | no | archive  |
| `claim_verdict_consistency` | CLASS CVV — A FINDING CLAIMS A GO / SURPASS / CLOSURE WHILE CITING AN ARTIFACT WHOSE OWN VERDICT IS NEGATIVE.  | 0.0 / 0.0 | 1 / 0 / 1 | no | archive  |
| `closure_names_mechanism` | CLASS CM — A CLOSURE CLAIM THAT NAMES NO MECHANISM, SO NOTHING CAN ADJUDICATE IT. THE DEFECT, measured 2026-07 | 0.0 / 0.0 | 2 / 0 / 3 | no | archive  |
| `compute_idle_persistent` | CLASS UC — DEDICATED compute (a pool node / the GPU) reads idle-with-ready-work, PERSISTENTLY. BLOCKING. WHY T | 0.0 / 0.01 | 10 / 0 / 4 | idle GPU/pool | archive  |
| `conditional_sweep` | FAILURE CLASS 10 -- single-axis-sweep-reported-as-absolute (3 incidents). EVIDENCE. A sweep varies one axis wh | 0.0 / 0.0 | 4 / 0 / 0 | no | archive  |
| `consumer_hardware_reference` | CLASS CH — the production brain must BUILD + RUN a chat turn inside a SINGLE CONSUMER RTX 3090's VRAM. WHY (ow | 0.0 / 0.0 | 4 / 1 / 0 | old-substrate config estimate | archive (estimates the frozen spiking substrate, not the new core; repo-state, not commit content) |
| `coverage` | CLASS COV — a NOTICED failure mode that never became a gate. WHY (owner, 2026-07-31: "are we automatically upd | 0.0 / 0.01 | 158 / 15 / 11 | FAILURE_LOG rows | archive (removed by owner rule: FAILURE_LOG no longer needs a gate) |
| `deep_research_at_wall` | DR — DEEP RESEARCH (local record + EXTERNAL literature) NOT DONE WHEN HAMMERING A WALL. THE DEFECT, recurring  | 0.0 / 0.0 | 15 / 0 / 2 | lever log | archive  |
| `device_and_cost` | CLASS DC — AN ARTIFACT CANNOT SAY WHAT DEVICE IT RAN ON, OR BURNED HOURS WITHOUT EVER PROJECTING ITS COST. TWO | 0.0 / 0.0 | 23 / 1 / 2 | no | archive  |
| `discriminating_power` | FAILURE CLASS 4 -- comparison with no discriminating power (7 incidents). EVIDENCE. A place-specificity CONTRO | 0.0 / 0.0 | 11 / 1 / 0 | no | archive  |
| `doc_type` | CLASS D — DOCUMENT TYPE AND PLACEMENT: information recorded where it does not belong. THE EVIDENCE (measured 2 | 0.0 / 0.0 | 19 / 1 / 1 | no | archive  |
| `finding_mechanism_on_main` | CLASS FM — A FINDING CLAIMS A MECHANISM THAT ISN'T ON MAIN. THE FAILURE IT CLOSES (2026-09-23). `research/find | 0.01 / 0.0 | 2 / 0 / 1 | state of main | archive  |
| `flip_offarm_staleness` | flip_offarm_staleness (CLASS OS, BLOCK) — a flip/soak/verify runner's OFF arm must FORCE the flag OFF explicit | 0.18 / 0.57 | 11 / 0 / 7 | no | archive (old-track flips; 1.2 s) |
| `flip_smoke_first` | CLASS SMK -- a FLIP preregistration filed with no dev-seed smoke behind it. EVIDENCE (2026-10-17/18). The word | 0.18 / 0.02 | 1 / 0 / 2 | no | archive  |
| `instrument_required` | CLASS I — INSTRUMENT-REQUIRED: a GO that reports a SIZE without a SOURCE. THE FAILURE CLASS (cause 4 of `2026- | 0.0 / 0.0 | 3 / 0 / 1 | no | archive  |
| `knob_reachable` | CLASS KR — A KNOB THAT CHANGES THE SUBSTRATE BUT CANNOT BE SET FROM THE COMMAND LINE. THE DEFECT, measured 202 | 0.0 / 0.0 | 2 / 0 / 1 | no | archive  |
| `lane_starvation` | CLASS L — CPU LANES STARVED while work continues elsewhere. BLOCKING. WHY THIS IS BLOCKING AND NOT REPORTING ( | 0.01 / 0.07 | 117 / 13 / 13 | idle CPU lanes | archive  |
| `lb_battery_provenance` | CLASS LBP — a load-bearing battery's aggregate.json is committed while its own per-cell provenance was never c | 0.0 / 0.0 | 1 / 0 / 2 | no | archive  |
| `lever_efficacy` | Failure class 1 — MANIPULATION-NEVER-ENGAGED (10 incidents, the most expensive class). EVIDENCE. A 17-agent au | 0.0 / 0.0 | 5 / 1 / 1 | no | archive  |
| `live_state_reinjection` | Failure class 13 — LIVE-STATE re-injection broken / over-cap (compaction-loss + attention-dilution). WHY (owne | 0.0 / 0.0 | 1 / 0 / 0 | live-state file | archive  |
| `operating_point` | CLASS OP — OPERATING POINT: the artifact records a target AND the value that missed it, and nobody looked. THE | 35.12 / 0.0 | 87 / 1 / 3 | no | archive (selftest alone 54 s) |
| `production_integration` | CLASS PI — A GO THAT IS NOT WIRED INTO THE PRODUCTION DEFAULT, CLAIMED AS IF IT WERE. THE DEFECT, named by the | 0.16 / 0.15 | 24 / 1 / 5 | no | archive  |
| `quantity_mismatch` | FAILURE CLASS 6 -- wrong-quantity comparison: two correct numbers of DIFFERENT quantities, compared as if they | 0.0 / 0.0 | 0 / 0 / 1 | no | archive  |
| `refuted_mechanism_reproposal` | refuted-mechanism re-proposal gate (RM). WHY THIS EXISTS. On 2026-08-02 a finding, the board, and BOTH roadmap | 0.0 / 0.0 | 1 / 0 / 1 | no | archive  |
| `retrieval_completeness` | CLASS R — THE RECORD'S OWN RETRIEVAL LAYER CANNOT SEE PART OF THE RECORD. THE DEFECT, measured 2026-07-31. Eve | 0.0 / 0.0 | 1 / 0 / 1 | repo code | archive  |
| `single_draw_null` | CLASS SN — a SINGLE random draw used as a pass/fail NULL control. BLOCKING, for NEWLY ADDED runners. EVIDENCE  | 0.0 / 0.01 | 1 / 0 / 1 | no | archive  |
| `ssh_stdin_in_read_loop` | ssh_stdin_in_read_loop (class SR, BLOCK) -- an `ssh` call, or a call to a repo script that itself runs ssh on  | 0.0 / 0.01 | 6 / 0 / 1 | no | archive  |
| `stale_pointer` | Failure class 8 -- STALE POINTER / UNMAINTAINED REGISTRY (5 recorded incidents). THE EVIDENCE. `docs/RETRACTED | 0.0 / 0.0 | 11 / 0 / 1 | steering docs corpus | archive  |
| `stated_value_mismatch` | CLASS SV — A FINDING STATES A NAMED QUANTITY THAT DISAGREES WITH THE ARTIFACT IT CITES. THE DEFECT, and it is  | 0.0 / 0.0 | 3 / 0 / 2 | no | archive  |
| `summary_doc_freshness` | CLASS SF — THE FORWARD-LOOKING SUMMARY DOCS DRIFTED WHILE FINDINGS PILED UP, AND NOTHING CAUGHT IT. THE DEFECT | 0.0 / 0.11 | 18 / 1 / 2 | summary-doc age | archive  |
| `throughput` | Failure class 7 — LIVENESS MISTAKEN FOR PROGRESS (4 incidents). THE EVIDENCE. `research/findings/2026-07-30-CR | 0.0 / 0.0 | 23 / 1 / 0 | no | archive  |
| `tool_health_fresh` | CLASS TH — a free lane/tool ROTTED (or its readiness went unchecked) and nothing surfaced it. WHY (owner, 2026 | 0.0 / 0.0 | 1 / 0 / 1 | tool health file | archive  |
| `verdict_can_read_nogo` | CLASS VN — a Verdict that can only print GO or UNDEFINED, never NO-GO. BLOCKING, for NEWLY ADDED runners. EVID | 0.0 / 0.01 | 1 / 0 / 1 | no | archive  |
| `vikunja_freshness` | CLASS VF — THE VIKUNJA BOARD DRIFTED WHILE FINDINGS PILED UP, AND NOTHING CAUGHT IT. THE DEFECT, measured 2026 | 0.0 / 0.04 | 15 / 0 / 1 | board sync marker | archive  |
| `artifact_provenance` | CLASS P — ARTIFACT PROVENANCE: a result JSON that cannot say what produced it. THE FAILURE CLASS. 645 of 1330  | 0.19 / 0.01 | 9 / 1 / 6 | no | keep: cheap; a result that cannot say what made it is an invalid claim  |
| `brain_prereq` | brain_prereq (class BP, BLOCK): a new-core experiment or eval that never asked the record or the biology. WHY  | 0.0 / 0.01 | 0 / 0 / 1 | no | NEW (from brain-v2, fixed): new-core tests cite capability + record; grandfathers brain-v2 files  |
| `corpus_check_required` | CLASS CC — AN EXPENSIVE RUN WHOSE QUESTION WAS NEVER CHECKED AGAINST THE RECORD. THE DEFECT, and it is the mos | 0.02 / 0.01 | 9 / 0 / 9 | no | keep + EXTEND: new brain/, docs/plans/, research/findings/ files need a record check in 24 h  |
| `host_decision_declared` | CLASS HD -- a plan that adds HOST code deciding content, filed without saying so, and rebuilt build after buil | 0.2 / 0.02 | 3 / 0 / 1 | no | keep: brain-based rule; host code deciding content must be declared  |
| `prereg_amendment_order` | CLASS PRA — an AMENDMENT to an EXISTING pre-registration, committed TOGETHER WITH run data. THE GAP (opus revi | 0.16 / 0.01 | 5 / 0 / 2 | no | keep: same, for amendments; prereg_before_run imports it  |
| `prereg_before_run` | CLASS PR — a pre-registration committed TOGETHER WITH the run artifacts it governs. EVIDENCE (2026-09-23). Thr | 0.12 / 0.01 | 11 / 2 / 4 | no | keep: prereg landed with its own data is an invalid claim  |
| `secret_scan` | secret_scan (class SEC, BLOCK): a credential committed into the repo. WHY (2026-11-02 lean-method). The repo i | 0.0 / 0.02 | 0 / 0 / 0 | no | NEW: a pushed credential is irreversible; no check existed  |
| `single_seed` | FAILURE CLASS 9 — single-seed headline (4 recorded incidents). EVIDENCE. "The headline was reported all day fr | 0.0 / 0.0 | 75 / 4 / 3 | no | keep: six-seed rule for headline claims  |
| `terminology` | FAILURE CLASS 11 — TERMINOLOGY OVERCLAIM (3 recorded incidents, all on 2026-07-28). THE EVIDENCE. Three of nin | 0.0 / 0.0 | 12 / 1 / 1 | no | keep (report-only): overclaim words  |
| `verdict_preconditions` | CLASS V — AN ARTIFACT ASSERTS A VERDICT WITHOUT CARRYING WHAT EARNED IT. THE GAP THIS CLOSES, and why it neede | 0.0 / 0.0 | 40 / 4 / 7 | no | keep: verdict without its earning preconditions; best catch record of the claim gates  |
| `workflow_cost_tiering` | workflow_cost_tiering (class CT, BLOCK) — a committed workflow script must TIER its agent models. WHY (owner-f | 0.0 / 0.0 | 1 / 0 / 1 | no | keep: untiered agents burn money  |

## Not found

- No secrets check existed; `secret_scan` is new.
- No-verify and force-push protection is not a gate: it lives in the drivers' permission deny lists
  (`tests/test_local_llm_permissions.py`) and the Claude Code hooks. `pre-merge-commit` (doc structure on merges) is
  unchanged by this pass.

## Follow-ups outside the gate files

- `tools/parallel_audit.py:48` imports `gates.compute_idle_persistent` / `gates.lane_starvation`; it now prints
  "waiver surfacing disabled" (by design, the gates no longer block). Point it at `gates._archive` or drop the waiver code.
- `tools/ssh_stdin_gate_replay.py` imports the archived ssh gate by its old path.
- Pre-existing test failures, unchanged by this pass: two `test_gate_corpus_check_required` incident tests (the real
  artifact now carries a stamp), `test_seam_contracts` seams 3 and 4, `test_integration_wave2_anchors` job-line,
  `test_deny_protected_edits_hook` hook-exists.
