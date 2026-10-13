# 2026-10-12 validation battery JOBS — C5 pair (idle-replay flip), tag c5val_1012

- File: research/findings/2026-10-12-validation-battery-JOBS.md
- Date: 2026-10-12 (job generation for the C5 flip pair; RULING c5-flip-merged-1012 is the operative order)
- Prereg: research/findings/2026-10-12-validation-battery-prereg.md (section 4 job generation; section 5 bars C5/B1/B2)
- Rulings: c5-g6b-accept-1012, battery-prereg-1012, battery-prereg-landed-1012, battery-c5-flip-1012, c5-flip-merged-1012 (operative), c5-off-baseline, flip-hurts-answers, speedup-smoke-first (amended smoke-rss-portable), pool-hold-1012.
- Revision pin: f24963bfb05ca535f10f70bf4380fe4b2f68a9ea (contains c5-flip merge 8c10af86a; idle replay default-ON PROVISIONAL; BRAIN_IDLE_REPLAY=0 stays the byte-identical escape; dev-7 identity smoke 4/4 whole-record byte-identical vs pre-flip default; g6b ON arm circuits_built=1 at selfinit, ON-OFF 6.3 MB against the 2 GB bar).

## 0. Smoke basis (already measured — the C5 pair smokes exist, so these lines are queue-now)

g6b pair (turns emo,bc_a,bc_b,selfinit, fresh worker build per arm): peak 16924040 KB = 16.14 GiB per arm, ~12 min wall per arm (4 turns incl. build). Prereg smoke-first is satisfied for this pair (dev-7 identity 4/4 + g6b circuits_built=1 both landed); the B2d shards still owe their own smoke (see section 7).

## 1. Fixed defaults-guard prefix (prereg section 4; fixed at job generation and committed with this JOBS list)

```
.venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid
```

Composition: inside each job's bash -c body, FIRST in the { ...; } chain (before mem_ok, memcap, the env prefix and the runner), joined with &&, so any guard exit non-zero stops the line before a brain builds. The guards assert NO BRAIN_* in their environment (assert_flipped_defaults rejects any ambient BRAIN_*; assert_iw1/assert_iw2 likewise), so every arm flag (BRAIN_CHAT_SEED, BRAIN_IDLE_REPLAY, BRAIN_IDLE_REPLAY_LEAKLOG) rides the env prefix of the runner command ONLY, after the guards. This doubles as protection of the default-ON arm: an ambient leftover BRAIN_IDLE_REPLAY fails the guard loudly instead of silently measuring the wrong arm.

Job-generation fix (battery-prereg-landed-1012 items (3)/(4): fix at job generation as written): assert_iw2_defaults.py requires its packet list exactly once (argparse error otherwise — tools/assert_iw2_defaults.py lines 81-82 at the pin), so the prereg's "wave anchors at their expected set" is passed POSITIONALLY (the guard's own job-line form per its docstring). Expected set verified in the pinned tree: false-belief-chat (webapp/false_belief_chat.py:72), transitive-chat (webapp/reasoning_transitive_chat.py:65), neural-selfid (research/runners/brain_chat_tui.py:660) all read True; wm-focus-bind reads False (d6_multiref_wm_production_organ.py:249) so iw1's --expect-on d6-hebbian-write,xedge-surprise-worldmodel passes as the prereg writes it. A bare assert_iw2_defaults.py would exit 2 on every line; the list above is the fix.

## 2. Seed mechanism (the runner has no --seed argument)

The worker's brain is seeded by the env var BRAIN_CHAT_SEED (webapp/server.py::_brain_chat_seed: unset -> 42, threaded to the substrate and every organ build; webapp/affect_drives_chat.py._DEFAULT_SEED reads the same var). A single-seed arm is therefore expressed by BRAIN_CHAT_SEED=<N> in the env prefix of the worker command. The --env JSON channel also works in principle (research/queue/pool.queue.claims:1468 used --env {"BRAIN_CHAT_SEED": "7", ...}) but over-escaped under the printf dispatcher and died rc1 (smoke-c5a requeue note: "flag now rides outer env") — so all seed/flag knobs ride the outer env prefix and --env {} stays empty (the proven g6b/identity shape). Seeds 42/43/44/100/101/102 are set EXPLICITLY on all 12 lines even where they equal the default (reproducibility rule: cfg.seed always passed). Each arm is one --worker invocation = one fresh brain build at that seed.

## 3. The queued jobs — C5 pair, 6 gate seeds x OFF + default-ON (12 lines)

Arm env decision (prereg: "OFF (BRAIN_IDLE_REPLAY=0) and default (the flipped ON)"): the OFF arm sets BRAIN_IDLE_REPLAY=0 explicitly (the byte-identical escape, never a pop). The default-ON arm passes NO BRAIN_IDLE_REPLAY token at all — it measures the shipped default at the pin (which contains 8c10af86a). If the default were somehow not ON, the seed reads UNDEFINED loudly through the C5(1) replay-fired precondition (no organ records / circuits_built=0 in the ON sidecar), never a silent pass. Both arms set BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/<tag>_rss.jsonl (flag-independent per-turn battery_turn sidecar; the ON side's file also holds the organ's own records).

Paste discipline (also section 5): ONE `bash tools/pool_queue.sh add` at a time, entry N+1 only after entry N's job records rc=0. Since the cmd text contains single quotes and $-expansions, pass it through a quoted heredoc so it reaches add verbatim:

```
bash tools/pool_queue.sh add "$(cat <<'EOF'
<cmd line from below>
EOF
)" --checked "$(cat <<'EOF'
<checked text from below>
EOF
)"
```

### 3.1 seed 42, OFF
cmd:
```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/c5val_1012 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=42 BRAIN_IDLE_REPLAY=0 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/c5_s42_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/c5val_1012/c5_s42_off.json > research/findings/raw/_regression_battery/c5val_1012/c5_s42_off.log 2>&1; }'
```
checked:
```
c5val_1012 C5 pair OFF arm seed 42: record says dev-7 identity smoke 4/4 whole-record byte-identical off@F vs pre-flip default (RULING c5-flip-merged-1012) and g6b pair peak 16924040 KB / ~12 min wall per 4-turn arm (RULING c5-g6b-accept-1012); BRAIN_IDLE_REPLAY=0 is the byte-identical escape; mem_gb=21 = ceil(1.3 x 16.14 GiB peak); max_age_h=24; one arm at a time globally, ON of seed 42 queued only after this lands rc=0
```

### 3.2 seed 42, default-ON
cmd:
```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/c5val_1012 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=42 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/c5_s42_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/c5val_1012/c5_s42_on.json > research/findings/raw/_regression_battery/c5val_1012/c5_s42_on.log 2>&1; }'
```
checked:
```
c5val_1012 C5 pair default-ON arm seed 42: record says g6b ON fired replay (circuits_built=1) with ON-OFF 6.3 MB vs the 2 GB C5 bar (RULING c5-g6b-accept-1012); arm passes NO BRAIN_IDLE_REPLAY so it measures the flipped default at pin f24963bfb05ca535f10f70bf4380fe4b2f68a9ea (guard chain refuses ambient BRAIN_*); mem_gb=21 = ceil(1.3 x 16.14 GiB peak); max_age_h=24; OFF arm seed 42 landed rc=0 first, one arm at a time globally
```

### 3.3 seed 43, OFF
cmd:
```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/c5val_1012 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=43 BRAIN_IDLE_REPLAY=0 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/c5_s43_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/c5val_1012/c5_s43_off.json > research/findings/raw/_regression_battery/c5val_1012/c5_s43_off.log 2>&1; }'
```
checked:
```
c5val_1012 C5 pair OFF arm seed 43: record says dev-7 identity smoke 4/4 whole-record byte-identical off@F vs pre-flip default (RULING c5-flip-merged-1012) and g6b pair peak 16924040 KB / ~12 min wall per 4-turn arm (RULING c5-g6b-accept-1012); BRAIN_IDLE_REPLAY=0 is the byte-identical escape; mem_gb=21 = ceil(1.3 x 16.14 GiB peak); max_age_h=24; one arm at a time globally, ON of seed 43 queued only after this lands rc=0
```

### 3.4 seed 43, default-ON
cmd:
```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/c5val_1012 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=43 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/c5_s43_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/c5val_1012/c5_s43_on.json > research/findings/raw/_regression_battery/c5val_1012/c5_s43_on.log 2>&1; }'
```
checked:
```
c5val_1012 C5 pair default-ON arm seed 43: record says g6b ON fired replay (circuits_built=1) with ON-OFF 6.3 MB vs the 2 GB C5 bar (RULING c5-g6b-accept-1012); arm passes NO BRAIN_IDLE_REPLAY so it measures the flipped default at pin f24963bfb05ca535f10f70bf4380fe4b2f68a9ea (guard chain refuses ambient BRAIN_*); mem_gb=21 = ceil(1.3 x 16.14 GiB peak); max_age_h=24; OFF arm seed 43 landed rc=0 first, one arm at a time globally
```

### 3.5 seed 44, OFF
cmd:
```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/c5val_1012 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=44 BRAIN_IDLE_REPLAY=0 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/c5_s44_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/c5val_1012/c5_s44_off.json > research/findings/raw/_regression_battery/c5val_1012/c5_s44_off.log 2>&1; }'
```
checked:
```
c5val_1012 C5 pair OFF arm seed 44: record says dev-7 identity smoke 4/4 whole-record byte-identical off@F vs pre-flip default (RULING c5-flip-merged-1012) and g6b pair peak 16924040 KB / ~12 min wall per 4-turn arm (RULING c5-g6b-accept-1012); BRAIN_IDLE_REPLAY=0 is the byte-identical escape; mem_gb=21 = ceil(1.3 x 16.14 GiB peak); max_age_h=24; one arm at a time globally, ON of seed 44 queued only after this lands rc=0
```

### 3.6 seed 44, default-ON
cmd:
```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/c5val_1012 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=44 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/c5_s44_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/c5val_1012/c5_s44_on.json > research/findings/raw/_regression_battery/c5val_1012/c5_s44_on.log 2>&1; }'
```
checked:
```
c5val_1012 C5 pair default-ON arm seed 44: record says g6b ON fired replay (circuits_built=1) with ON-OFF 6.3 MB vs the 2 GB C5 bar (RULING c5-g6b-accept-1012); arm passes NO BRAIN_IDLE_REPLAY so it measures the flipped default at pin f24963bfb05ca535f10f70bf4380fe4b2f68a9ea (guard chain refuses ambient BRAIN_*); mem_gb=21 = ceil(1.3 x 16.14 GiB peak); max_age_h=24; OFF arm seed 44 landed rc=0 first, one arm at a time globally
```

### 3.7 seed 100, OFF
cmd:
```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/c5val_1012 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=100 BRAIN_IDLE_REPLAY=0 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/c5_s100_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/c5val_1012/c5_s100_off.json > research/findings/raw/_regression_battery/c5val_1012/c5_s100_off.log 2>&1; }'
```
checked:
```
c5val_1012 C5 pair OFF arm seed 100: record says dev-7 identity smoke 4/4 whole-record byte-identical off@F vs pre-flip default (RULING c5-flip-merged-1012) and g6b pair peak 16924040 KB / ~12 min wall per 4-turn arm (RULING c5-g6b-accept-1012); BRAIN_IDLE_REPLAY=0 is the byte-identical escape; mem_gb=21 = ceil(1.3 x 16.14 GiB peak); max_age_h=24; one arm at a time globally, ON of seed 100 queued only after this lands rc=0
```

### 3.8 seed 100, default-ON
cmd:
```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/c5val_1012 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=100 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/c5_s100_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/c5val_1012/c5_s100_on.json > research/findings/raw/_regression_battery/c5val_1012/c5_s100_on.log 2>&1; }'
```
checked:
```
c5val_1012 C5 pair default-ON arm seed 100: record says g6b ON fired replay (circuits_built=1) with ON-OFF 6.3 MB vs the 2 GB C5 bar (RULING c5-g6b-accept-1012); arm passes NO BRAIN_IDLE_REPLAY so it measures the flipped default at pin f24963bfb05ca535f10f70bf4380fe4b2f68a9ea (guard chain refuses ambient BRAIN_*); mem_gb=21 = ceil(1.3 x 16.14 GiB peak); max_age_h=24; OFF arm seed 100 landed rc=0 first, one arm at a time globally
```

### 3.9 seed 101, OFF
cmd:
```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/c5val_1012 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=101 BRAIN_IDLE_REPLAY=0 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/c5_s101_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/c5val_1012/c5_s101_off.json > research/findings/raw/_regression_battery/c5val_1012/c5_s101_off.log 2>&1; }'
```
checked:
```
c5val_1012 C5 pair OFF arm seed 101: record says dev-7 identity smoke 4/4 whole-record byte-identical off@F vs pre-flip default (RULING c5-flip-merged-1012) and g6b pair peak 16924040 KB / ~12 min wall per 4-turn arm (RULING c5-g6b-accept-1012); BRAIN_IDLE_REPLAY=0 is the byte-identical escape; mem_gb=21 = ceil(1.3 x 16.14 GiB peak); max_age_h=24; one arm at a time globally, ON of seed 101 queued only after this lands rc=0
```

### 3.10 seed 101, default-ON
cmd:
```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/c5val_1012 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=101 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/c5_s101_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/c5val_1012/c5_s101_on.json > research/findings/raw/_regression_battery/c5val_1012/c5_s101_on.log 2>&1; }'
```
checked:
```
c5val_1012 C5 pair default-ON arm seed 101: record says g6b ON fired replay (circuits_built=1) with ON-OFF 6.3 MB vs the 2 GB C5 bar (RULING c5-g6b-accept-1012); arm passes NO BRAIN_IDLE_REPLAY so it measures the flipped default at pin f24963bfb05ca535f10f70bf4380fe4b2f68a9ea (guard chain refuses ambient BRAIN_*); mem_gb=21 = ceil(1.3 x 16.14 GiB peak); max_age_h=24; OFF arm seed 101 landed rc=0 first, one arm at a time globally
```

### 3.11 seed 102, OFF
cmd:
```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/c5val_1012 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=102 BRAIN_IDLE_REPLAY=0 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/c5_s102_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/c5val_1012/c5_s102_off.json > research/findings/raw/_regression_battery/c5val_1012/c5_s102_off.log 2>&1; }'
```
checked:
```
c5val_1012 C5 pair OFF arm seed 102: record says dev-7 identity smoke 4/4 whole-record byte-identical off@F vs pre-flip default (RULING c5-flip-merged-1012) and g6b pair peak 16924040 KB / ~12 min wall per 4-turn arm (RULING c5-g6b-accept-1012); BRAIN_IDLE_REPLAY=0 is the byte-identical escape; mem_gb=21 = ceil(1.3 x 16.14 GiB peak); max_age_h=24; one arm at a time globally, ON of seed 102 queued only after this lands rc=0
```

### 3.12 seed 102, default-ON
cmd:
```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea && mkdir -p research/findings/raw/_regression_battery/c5val_1012 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=102 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/c5val_1012/c5_s102_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/c5val_1012/c5_s102_on.json > research/findings/raw/_regression_battery/c5val_1012/c5_s102_on.log 2>&1; }'
```
checked:
```
c5val_1012 C5 pair default-ON arm seed 102: record says g6b ON fired replay (circuits_built=1) with ON-OFF 6.3 MB vs the 2 GB C5 bar (RULING c5-g6b-accept-1012); arm passes NO BRAIN_IDLE_REPLAY so it measures the flipped default at pin f24963bfb05ca535f10f70bf4380fe4b2f68a9ea (guard chain refuses ambient BRAIN_*); mem_gb=21 = ceil(1.3 x 16.14 GiB peak); max_age_h=24; OFF arm seed 102 landed rc=0 first, one arm at a time globally
```

Line-shape notes (all 12 identical in shape, proven against pool_queue.sh's gates): first word python3 resolves on PATH (shape gate); the only -m module is research.runners.onebrain_regression_battery and its scoped flags (--worker --env --turns --out) are all in its argparse (command validity gate); the revision pin makes the flag check run against the pinned revision on the node (remote validity gate); .venv/bin/python is the interpreter (interpreter guard); the guard scripts' --expect-on sits BEFORE the -m token so it is outside the flag scope (RULING iw1-stepB scoping). The $PWD inside the single-quoted bash -c body expands on the node after cd, i.e. the revision root (the proven identity-smoke form). The python3 wrapper prints {"rc":..., "maxrss_kb":...} to the dispatcher record (the proven g6b/identity form).

## 4. Sizing table (prereg smoke-first rule: mem_gb = ceil(1.3 x peak); if peak x parallelism > host RAM or projected wall > 24 h, do not launch)

| item | value |
| --- | --- |
| peak RSS per arm (smoke: 4 turns emo,bc_a,bc_b,selfinit, fresh build) | 16924040 KB = 16.14 GiB |
| mem_gb (queue reservation, per line) | 21 = ceil(1.3 x 16.14 GiB) = ceil(20.98) — the value RULING c5-flip-merged-1012 names |
| mem_ok need / memcap kill ceiling (in-line) | 21 1 / 22 (1 GB over the reservation; the proven g6b-ON pair 21/22 completed the identical turn set at 16.1 GiB) |
| max_age_h (stamped in --checked) | 24 |
| estimated wall per arm | ~12 min (4 turns incl. fresh worker build; dev smoke measured) |
| estimated wall, 12 arms strictly sequential | 144 min = 2.4 h |
| peak x parallelism (1) | 16.14 GiB < 46 GiB wspool RAM — pass (wspool only; see correction below) |
| 24 h wall bar | 0.2 h/arm, 2.4 h total — pass |

**Correction 2026-10-13 01:00 (measured `free -g` on pool41): pool40/41/42 total RAM = 15 GB, not the
32 GiB assumed when this list was generated.** A mem_gb=21 arm therefore fits ONLY wspool (46 GB); no pool
node can host even one arm, so line 3/12's `pool_node=pool41` pin was withdrawn (withdrawn.log 2026-10-13
01:01) and the line re-added unpinned — during game-1013 the dispatcher skips unreachable wspool, so it
holds until game off and then dispatches to the only node that fits. The one-arm-at-a-time discipline still
binds on wspool (2 x mem_gb=21 = 42 <= 46 GiB, so the mem math alone would allow co-location).
| NEEDS [aws-run] | none (both prereg launch bars pass; see section 8) |

Unit note: the g6b checked text computed its 22 from decimal GB (16.924 x 1.3 = 22.00); the operative ruling and this sizing use binary GiB (16.14 x 1.3 = 20.98 -> 21), which is what c5-flip-merged-1012 stamps as mem_gb=21. memcap 22 covers either reading.

## 5. Sequential / parallelism discipline (decided explicitly)

DECISION: one arm in flight GLOBALLY, at all times — not per node, not per seed. RULING c5-flip-merged-1012 ("run one arm at a time") read strictly together with RULING c5-off-baseline ("never both at once"): even arms of different seeds never run concurrently. This is the safest reading and it is what is enforced.

How it is enforced: the QUEUE NOW list is pasted ONE tools/pool_queue.sh add at a time; entry N+1 is added only after entry N's job has recorded rc=0 (the proven g6b pattern: "ON arm queued after this lands"). The 12 lines are NOT batch-added.

Why batching is forbidden even though the dispatcher would accept it: with mem_gb=21 per line, two lines sum to 42 — that exceeds pool40/41/42 RAM (measured 15 GB each as of 2026-10-13, not the 32 GiB first assumed; they in fact cannot host even one arm) so only wspool can host arms, and 42 <= 46 GiB means wspool COULD co-locate two. The mem math therefore does not by itself prevent co-location on wspool; discipline does. If a future operator must batch (e.g. unattended overnight), the hardening is to pin pool_node=wspool with mem_gb=26 (wspool-ram-1012: one at a time) so 2 x 26 = 52 > 46 makes co-location physically impossible — that variant is NOT used here because c5-flip-merged-1012 names mem_gb=21 and the paste discipline already guarantees serialization.

Per-seed order: OFF (BRAIN_IDLE_REPLAY=0) first, then default-ON (the g6b shape; C5(3)'s per-turn ON-OFF RSS diff needs the OFF curve first). Seed order: 42, 43, 44, 100, 101, 102. Nodes: no pool_node pin — any node that fits, which after the 2026-10-13 01:00 measurement means wspool only (pool40/41/42 are 15 GB total, below the 16.14 GiB arm peak; see the correction in section 4). Queue adds may run before the f24963b... provision marker is up (the dispatcher holds pinned lines until the revision is provisioned; pool_autodispatch.sh fails closed on the revision check).

## 6. alloff before answer_rate (RULING flip-hurts-answers)

UNMEASURED means not checked: the alloff attribution cells (tag b2d1012-alloff, every flip flag forced =0 at the same revision) are queued BEFORE any answer_rate read. The C5 pair above is not an answer_rate input (answer_rate pairs --on-tag b2d1012 --off-tag b2d1012-alloff per flipped faculty row); the binding ordering lands on the HELD B2d batch (section 7): when their own smoke releases them, the b2d1012-alloff lines queue first (in any case before any tools/answer_rate.py invocation). For each flip row, answer_rate runs per the prereg's paired layout; flags: FLIP-HURTS-ANSWERS on ANY measured seed is REPORTED beside the registered verdict, changes no PASS/FAIL, and requires a NEEDS line and a Claude ruling before that flip stands; UNMEASURED seeds are listed, never read as no harm.

## 7. HELD — B2d shard lines (declared here, NOT queued; pending their own smoke)

Held per RULING c5-flip-merged-1012 ("the B2d shards follow their own smoke"). Release condition (prereg smoke-first): the smallest unit — one probe group, one seed — measured for peak RSS and wall time at the pin; the shard lines carry mem_gb = ceil(1.3 x that peak) and a stamped max_age_h; if projected wall > 24 h or peak x parallelism > host RAM, do not launch — one NEEDS with the numbers.

Generation commands (verbatim from prereg section 4, <F> = f24963bfb05ca535f10f70bf4380fe4b2f68a9ea; run at generation time):

```
.venv/bin/python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag b2d1012 --no-fixes --probe-set adequate --repeats 2 --root ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea --pin f24963bfb05ca535f10f70bf4380fe4b2f68a9ea
.venv/bin/python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 --tag b2d1012-alloff --no-fixes --probe-set adequate --repeats 2 --extra-env <every flip flag =0> --root ~/derisk-pool/revisions/f24963bfb05ca535f10f70bf4380fe4b2f68a9ea --pin f24963bfb05ca535f10f70bf4380fe4b2f68a9ea
```

FLAGGED (genuinely not fixed by the prereg text): the exact --extra-env enumeration for the alloff generation — the six flipped rows' env names forced =0 (partially verified in research/runners/lbf_rows/: e.g. BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL=0 per xedge_surprise_worldmodel.py line 17) — is resolved at generation from the rows' flip flags (the iw2 make_jobs.sh off_env pattern) and committed with the shard JOBS list; it is not invented here. Every generated job line carries the SAME guard prefix from section 1 (guards first, in a BRAIN_*-free env; the alloff flags ride the shard line's runner env after the guards). The b2d1012-alloff lines queue before the b2d1012 lines are read/scored and before any answer_rate run (section 6).

Also still HELD per RULING pool-hold-1012: the 09-30 held _stage1_exit (204) and _stage1_c1_category (54) lines stay unqueued (pre-flip pin; attempt 1 already read). Stage-1 attempt 2 is fresh lines at current main after PLAN step D (C1 Amendment A2) — not this batch.

## 8. NEEDS

none — both prereg launch bars pass: peak 16.14 GiB x parallelism 1 = 16.14 GiB fits every pool node (32/46 GiB), and projected wall is 12 min/arm (2.4 h for all 12 sequential), far under 24 h. No NEEDS [aws-run] is forced.

(W2 note for commit time: the 12 verbatim queue lines are machine lines longer than 800 chars; do not run tools/split_long_doc_lines.py --apply over the fenced command blocks — it would break copy-paste verbatim execution. Register or exempt before commit.)