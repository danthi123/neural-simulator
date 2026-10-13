# battery-retention (c5-next(b)): what the battery worker kept per session, and the fix

Branch `claude-build/battery-retention` from `s/main` a41082d54. Workstation, numpy CPU, OMP_NUM_THREADS=1,
`BRAIN_CHAT_SEED=7` (dev seed), memcap 14. One brain process at a time. Units: rss_kb from /proc is KiB;
GiB = KiB / 1048576.

## Commits

- fefd940a7: first fix. Release was ON by default at that commit (superseded by 8e0ea8415).
- ebc11c78c: this WORKLOG's first version, with the RSS evidence.
- 8e0ea8415: review fixes (correctness SHIP, science FIX_FIRST). Release is OPT-IN, a release fails loudly, and the
  handler computes its key through the helper. This is instrument B in E1 below.

## What is retained

- `webapp/server.py` `brain_chat` caches one ChatBrain per `(session, brain, renderer)` key in `_BRAIN_CHATS`
  (plus ~20 per-session registries under the same key) and frees them only on `req.reset`. The worker runs every
  turn in one process, so each distinct session stays alive to the end. The step is per session, not per turn kind:
  bc_a and bc_b share session `bc` and add nothing.
- Inside each ChatBrain, the big part is the tiny-demo's long-term-memory tier. `_build_chat_brain` attaches
  `_load_or_build_ltm_store(wikidata_100k)` to every new ChatBrain as `TieredFactStore(composer, ltm)`. With the
  substrate store on (the default since RANK-6/#211), every LTM fact is held as its own 129-neuron
  `SimulationBridge` (`RFPhasorComposer._store_substrate`).
- `ltm_size_probe.py` loads only that LTM store, with no brain build (memcap 9): 395 shards, 78,857 facts, 78,857
  `SimulationBridge` handles. RSS goes up 5,709,276 KiB = 5.445 GiB (72.4 KiB per fact). Deleting the store and
  running gc + malloc_trim gives back 5,372,704 KiB = 5.124 GiB. The battery's emo to bc_a step is
  +5,690,652 KiB = 5.427 GiB (below). So one live session costs about 5.44 GiB, almost all of it this LTM tier.

## The mechanism (as of 8e0ea8415)

- `webapp/server.py`: the `req.reset` teardown is moved word for word into `_evict_brain_chat_session(cache_key)`.
  It is the same pops in the same order, and reset calls it. `brain_chat` computes its key by calling
  `brain_chat_cache_key(req)` (renderer = key[2]), so the helper cannot drift from the handler.
- `research/runners/onebrain_regression_battery.py`: release is OFF by default. That is the keep-everything loop
  every registered instrument ran, with no key computation and no release. `LB_BATTERY_RELEASE_SESSIONS=1` turns it
  on; the value is read case-insensitively (1/true/on/yes), and the LB_ prefix lets the provenance sidecar capture
  it.
- When on, the worker computes each turn's key with the helper BEFORE the call. `session_release_plan` releases a
  session after its last turn, or after the last world step that follows its first turn, whichever is later; world
  steps tick every live session through `_SESSION_MOOD` and `_LAST_REQUEST`.
- A release DROPS that session's ChatBrain and the per-session organs the teardown covers, then runs gc.collect and
  malloc_trim. It fails loudly when no key was recorded or a key was not live. The release is wrapped, so a
  non-live key or a reset-hook exception is recorded and the worker does not crash. The record goes to three places:
  `_battery_release_error` on that turn's response, stderr, and a `battery_release` sidecar record.
- Every `battery_turn` sidecar record carries `"session_release": true/false`.

## Why a worker-side release and not a server LRU

A bounded LRU in `_BRAIN_CHATS` (the 2026-10-12 finding's recommendation) evicts by recency, not by last use. World
steps (`_run_world_step`, `_run_vclock_step`) tick EVERY live session and read the max of `_LAST_REQUEST` across
sessions. An LRU that evicts a session before a later world step changes `n_ticked`, the virtual clock base and
the shared state the tick evolves, and it would do so silently, in every caller. The worker knows its whole turn
list, so it can release exactly after the last read, world steps included. That keeps world-step semantics, and the
server keeps its existing behaviour.

## Evidence so far (fefd940a7, turns emo,bc_a,bc_b; per-turn `battery_turn` rss_kb)

| turn | BEFORE (pristine a41082d54) | AFTER (fefd940a7, release on) |
|---|---|---|
| emo | 6679772 (6.37 GiB) | 6677732 (6.37 GiB) |
| bc_a | 12370424 (11.80 GiB) | 6287396 (6.00 GiB) |
| bc_b | 12370600 (11.80 GiB) | 6287384 (6.00 GiB) |
| process hwm | 12454096 (11.88 GiB) | 6834412 (6.52 GiB) |

Scope of this evidence: byte-identical on emo,bc_a,bc_b, dev seed 7 (whole-record sha), at fefd940a7 only. Both
output JSONs have sha256 `1b6dfe69a2e0ca972c19d6dc58325d4f8e66e21f554458cfee64ae64ac20dd01`, with no fields
excluded. Nothing broader is shown: not selfinit, rich turns, world steps, other seeds, or the 8e0ea8415 code. The
per-turn RSS sidecars are `before_seed7_emo_bc_rss.jsonl` and `after_seed7_emo_bc_rss.jsonl`. The reply JSONs are
not committed; the device-and-cost gate refuses an output without a provenance sidecar, and the runner writes
sidecars only under `research/findings/raw`.

## Open items

- OWNER-LEVEL: production still holds one 5.44 GiB LTM tier per LIVE conversation until reset, with no bound. The
  webapp server builds a full ChatBrain, wikidata_100k substrate store included, per (session, brain, renderer) key
  and never evicts one except on reset. Nothing here changes that. A shared read-only LTM or a bounded cache is a
  production design decision, because the LTM is mutable per session (trace, promote_buffer_to_ltm, codebook cache).
- Pre-existing gaps, noted and NOT fixed here:
  - `common_ground_ledger_production_organ.reset_organ(cache_key)` is missing from the reset teardown (its
    per-session `_ORGANS` entry survives a reset and an eviction).
  - `brain_chat_reset` (the /api/brain-chat/reset handler) is a drifted partial copy of the teardown. It pops only
    `_BRAIN_CHATS`, `_BRAIN_RICH`, `_SESSION_MOOD`, emotion-category and `_SESSION_PMEM`. It skips continuous_engine,
    worldview, multiref, silent-WM, selfinit, idle-replay, discourse, episodic, causal and false-belief, and it still
    builds its key inline.
  - `_prewarm_discard_scratch_session` is another drifted partial copy; it skips idle-replay and false-belief.
- The default may flip to release-on only after E1 passes.
- B must be reachable before it can be provisioned: this branch is not pushed. `tools/pool_provision.sh --revision
  8e0ea8415 --isolated` refuses a revision that does not contain origin/main; if it refuses, rebase or merge first.
  Rebasing changes the SHA, so regenerate the B lines.

## E1 equivalence check (Hermes runs this before the default can flip)

Instruments: A = the provisioned pin f24963bfb05ca535f10f70bf4380fe4b2f68a9ea (old worker, keep-everything).
B = 8e0ea84150e7f7000fe520f4bf8ca86baee44560 (this branch) with `LB_BATTERY_RELEASE_SESSIONS=1`. Dev seeds 7 and 8
only. The arms are OFF (`BRAIN_IDLE_REPLAY=0`), ON (no BRAIN_IDLE_REPLAY token, the shipped default at both pins),
and commit-lesion (`BRAIN_IDLE_REPLAY_COMMIT_LESION=1`). Every line is the registered JOBS line verbatim
(research/coordination/2026-10-12-validation-battery-JOBS.md line 46). That means the same guard chain, mem_ok 21 1,
memcap 22, numpy, OMP_NUM_THREADS=1, the LEAKLOG sidecar, and `--turns emo,bc_a,bc_b,selfinit`. Only these change:
the revision dir, the seed, the arm token, the B-only `LB_BATTERY_RELEASE_SESSIONS=1` token (after the guards, in the
runner's env prefix), and the out dir `research/findings/raw/_regression_battery/battery_retention_equiv/`. The
lines were generated mechanically from line 46 by `gen_e1.py`. A and the branch base a41082d54 have identical code:
`git diff f24963bfb a41082d54` touches only research/findings, research/coordination and docs. So A and B differ
only by fefd940a7 and 8e0ea8415.

Discipline is that of JOBS section 5: one arm in flight globally, and the arms fit only wspool (mem_gb=21). Add one
line at a time with `tools/pool_queue.sh add` through the quoted heredoc of JOBS section 3. Provision both pins
first: `bash tools/pool_provision.sh --revision <sha> --isolated wspool`.

PASS requires all four of the following; anything else is FAIL, and the default stays OFF.

1. For all 6 seed x arm pairs, the whole-record sha256 of `A_s<seed>_<arm>.json` equals that of `B_s<seed>_<arm>.json`.
2. Both ON arms of both instruments (`A_s7_on`, `A_s8_on`, `B_s7_on`, `B_s8_on`) show a `select` record with
   circuits_built > 0 between the `bc_b` and `selfinit` `battery_turn` records (the replay fired at the idle turn).
   If an A ON arm itself shows no select, E1 is UNDEFINED for that seed, not a B failure: the replay never fired, so
   the pair cannot show equivalence there. Replace the seed with another dev seed (9, 10, ...), never a gate seed.
3. Every B sidecar shows release: each `battery_turn` has session_release true. It holds `battery_release` records
   for emo (after emo), bc (after bc_b) and si (after selfinit), all with error null, and no B json carries
   `_battery_release_error`.
4. `Bnorel_s7_on` (B with release unset) has the same sha256 as `A_s7_on`, session_release false on every
   `battery_turn`, no `battery_release` record, and the same RSS staircase as A_s7_on. The staircase test is per turn:
   |rss_kb(Bnorel) - rss_kb(A)| <= 2% of A at emo, bc_a, bc_b and selfinit. The 2% tolerance is this WORKLOG's
   choice, because RSS is not bit-reproducible.

Checker, run in the out dir after all 13 lines record rc=0 (`python3 e1_check.py`):

```
import hashlib, json
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rows = lambda p: [json.loads(l) for l in open(p)]
ok = True
def chk(cond, msg):
    global ok
    ok &= bool(cond); print(("PASS " if cond else "FAIL ") + msg)
for s in (7, 8):
    for a in ("off", "on", "lesion"):
        chk(sha("A_s%d_%s.json" % (s, a)) == sha("B_s%d_%s.json" % (s, a)), "sha A==B s%d %s" % (s, a))
        b = rows("B_s%d_%s_rss.jsonl" % (s, a))
        chk(all(r.get("session_release") is True for r in b if r.get("event") == "battery_turn"), "B stamp s%d %s" % (s, a))
        rel = [(tuple(r["sessions"]), r["turn"], r["error"]) for r in b if r.get("event") == "battery_release"]
        chk(rel == [(("emo",), "emo", None), (("bc",), "bc_b", None), (("si",), "selfinit", None)],
            "B release s%d %s %r" % (s, a, rel))
        chk("_battery_release_error" not in open("B_s%d_%s.json" % (s, a)).read(), "B no error s%d %s" % (s, a))
    for inst in ("A", "B"):
        r = rows("%s_s%d_on_rss.jsonl" % (inst, s))
        i0 = next(i for i, x in enumerate(r) if x.get("event") == "battery_turn" and x["turn"] == "bc_b")
        i1 = next(i for i, x in enumerate(r) if x.get("event") == "battery_turn" and x["turn"] == "selfinit")
        chk(any(x.get("event") == "select" and (x.get("circuits_built") or 0) > 0 for x in r[i0 + 1:i1]),
            "%s s%d on: select circuits_built>0 at selfinit" % (inst, s))
chk(sha("Bnorel_s7_on.json") == sha("A_s7_on.json"), "sha Bnorel == A s7 on")
nr = rows("Bnorel_s7_on_rss.jsonl")
chk(all(r.get("session_release") is False for r in nr if r.get("event") == "battery_turn"), "Bnorel stamp false")
chk(not any(r.get("event") == "battery_release" for r in nr), "Bnorel no release records")
ta = {r["turn"]: r["rss_kb"] for r in rows("A_s7_on_rss.jsonl") if r.get("event") == "battery_turn"}
tb = {r["turn"]: r["rss_kb"] for r in nr if r.get("event") == "battery_turn"}
for t in ("emo", "bc_a", "bc_b", "selfinit"):
    chk(abs(tb[t] - ta[t]) <= 0.02 * ta[t], "Bnorel RSS staircase %s: %d vs A %d" % (t, tb[t], ta[t]))
print("E1", "PASS" if ok else "FAIL")
```

The 13 lines (order: seed 7 then 8; per seed OFF, ON, lesion; A before B within each arm; Bnorel last):

E1.1 `A_s7_off` (instrument A, seed 7, arm off):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=7 BRAIN_IDLE_REPLAY=0 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/A_s7_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/A_s7_off.json > research/findings/raw/_regression_battery/battery_retention_equiv/A_s7_off.log 2>&1; }'
```

E1.2 `B_s7_off` (instrument B, seed 7, arm off):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/8e0ea84150e7f7000fe520f4bf8ca86baee44560 && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=7 BRAIN_IDLE_REPLAY=0 LB_BATTERY_RELEASE_SESSIONS=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/B_s7_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/B_s7_off.json > research/findings/raw/_regression_battery/battery_retention_equiv/B_s7_off.log 2>&1; }'
```

E1.3 `A_s7_on` (instrument A, seed 7, arm on):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=7 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/A_s7_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/A_s7_on.json > research/findings/raw/_regression_battery/battery_retention_equiv/A_s7_on.log 2>&1; }'
```

E1.4 `B_s7_on` (instrument B, seed 7, arm on):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/8e0ea84150e7f7000fe520f4bf8ca86baee44560 && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=7 LB_BATTERY_RELEASE_SESSIONS=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/B_s7_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/B_s7_on.json > research/findings/raw/_regression_battery/battery_retention_equiv/B_s7_on.log 2>&1; }'
```

E1.5 `A_s7_lesion` (instrument A, seed 7, arm lesion):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=7 BRAIN_IDLE_REPLAY_COMMIT_LESION=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/A_s7_lesion_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/A_s7_lesion.json > research/findings/raw/_regression_battery/battery_retention_equiv/A_s7_lesion.log 2>&1; }'
```

E1.6 `B_s7_lesion` (instrument B, seed 7, arm lesion):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/8e0ea84150e7f7000fe520f4bf8ca86baee44560 && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=7 BRAIN_IDLE_REPLAY_COMMIT_LESION=1 LB_BATTERY_RELEASE_SESSIONS=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/B_s7_lesion_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/B_s7_lesion.json > research/findings/raw/_regression_battery/battery_retention_equiv/B_s7_lesion.log 2>&1; }'
```

E1.7 `A_s8_off` (instrument A, seed 8, arm off):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=8 BRAIN_IDLE_REPLAY=0 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/A_s8_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/A_s8_off.json > research/findings/raw/_regression_battery/battery_retention_equiv/A_s8_off.log 2>&1; }'
```

E1.8 `B_s8_off` (instrument B, seed 8, arm off):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/8e0ea84150e7f7000fe520f4bf8ca86baee44560 && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=8 BRAIN_IDLE_REPLAY=0 LB_BATTERY_RELEASE_SESSIONS=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/B_s8_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/B_s8_off.json > research/findings/raw/_regression_battery/battery_retention_equiv/B_s8_off.log 2>&1; }'
```

E1.9 `A_s8_on` (instrument A, seed 8, arm on):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=8 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/A_s8_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/A_s8_on.json > research/findings/raw/_regression_battery/battery_retention_equiv/A_s8_on.log 2>&1; }'
```

E1.10 `B_s8_on` (instrument B, seed 8, arm on):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/8e0ea84150e7f7000fe520f4bf8ca86baee44560 && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=8 LB_BATTERY_RELEASE_SESSIONS=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/B_s8_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/B_s8_on.json > research/findings/raw/_regression_battery/battery_retention_equiv/B_s8_on.log 2>&1; }'
```

E1.11 `A_s8_lesion` (instrument A, seed 8, arm lesion):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=8 BRAIN_IDLE_REPLAY_COMMIT_LESION=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/A_s8_lesion_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/A_s8_lesion.json > research/findings/raw/_regression_battery/battery_retention_equiv/A_s8_lesion.log 2>&1; }'
```

E1.12 `B_s8_lesion` (instrument B, seed 8, arm lesion):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/8e0ea84150e7f7000fe520f4bf8ca86baee44560 && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=8 BRAIN_IDLE_REPLAY_COMMIT_LESION=1 LB_BATTERY_RELEASE_SESSIONS=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/B_s8_lesion_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/B_s8_lesion.json > research/findings/raw/_regression_battery/battery_retention_equiv/B_s8_lesion.log 2>&1; }'
```

E1.13 `Bnorel_s7_on` (instrument B with release UNSET, seed 7, arm on):

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/8e0ea84150e7f7000fe520f4bf8ca86baee44560 && mkdir -p research/findings/raw/_regression_battery/battery_retention_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=7 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/battery_retention_equiv/Bnorel_s7_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/battery_retention_equiv/Bnorel_s7_on.json > research/findings/raw/_regression_battery/battery_retention_equiv/Bnorel_s7_on.log 2>&1; }'
```
