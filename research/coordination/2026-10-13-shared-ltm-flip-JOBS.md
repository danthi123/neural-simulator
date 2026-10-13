# Shared-LTM flip battery (one-brain memory plan step 1), 2026-10-13 — PREREG + JOBS

Ruled by Claude 2026-10-13 (runbook RULING shared-ltm-flip-1013). Committed BEFORE any of these lines ran; nothing below may change after data.

- **Revision:** main e08895cd3c723028b264d42c3d11d9ed8954e72c (contains the shared LTM 683c181b3 + 50e87d0f6, default OFF, and M0's turn lock e08895cd3).
- **Seeds:** the six gate seeds 42/43/44/100/101/102.
- **Arms:** `off` (production defaults) and `on` (production defaults + `BRAIN_SHARED_LTM=1`). Nothing else differs: idle replay is at its production default in both arms, unlike the C5 template.
- **Template:** registered line 46 of 2026-10-12-validation-battery-JOBS.md, with only the revision, seed, output dir/tag, the C5 `BRAIN_IDLE_REPLAY=0` removed and the flag changed (generator: Claude's sim-rev/sharedltm/gen_flip.py).

**Pass bar (all must hold; byte identity per docs/TERMS.md):**
1. All 12 lines rc=0.
2. For each seed, every per-turn response record in slf_sN_on.json equals slf_sN_off.json after dropping ONLY wall-clock, latency and RSS fields. Any other differing key on any seed = FAIL: report it, do not explain it away.
3. maxrss_kb(on) < maxrss_kb(off) on every seed.

On PASS, Claude flips `_SHARED_LTM_DEFAULT_ON` and adds the ledger row. On FAIL, the flag stays off and the differing turns go to a NEEDS line.

**Sizing:** same caps as the registered template (mem_ok 21, memcap 22). Parallelism per the 2026-10-12 JOBS section 5 rules.

SLF.1 `slf_s42_off`:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/e08895cd3c723028b264d42c3d11d9ed8954e72c && mkdir -p research/findings/raw/_regression_battery/shared_ltm_flip_1013 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=42 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s42_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s42_off.json > research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s42_off.log 2>&1; }'
```

SLF.2 `slf_s42_on`:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/e08895cd3c723028b264d42c3d11d9ed8954e72c && mkdir -p research/findings/raw/_regression_battery/shared_ltm_flip_1013 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=42 BRAIN_SHARED_LTM=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s42_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s42_on.json > research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s42_on.log 2>&1; }'
```

SLF.3 `slf_s43_off`:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/e08895cd3c723028b264d42c3d11d9ed8954e72c && mkdir -p research/findings/raw/_regression_battery/shared_ltm_flip_1013 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=43 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s43_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s43_off.json > research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s43_off.log 2>&1; }'
```

SLF.4 `slf_s43_on`:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/e08895cd3c723028b264d42c3d11d9ed8954e72c && mkdir -p research/findings/raw/_regression_battery/shared_ltm_flip_1013 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=43 BRAIN_SHARED_LTM=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s43_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s43_on.json > research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s43_on.log 2>&1; }'
```

SLF.5 `slf_s44_off`:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/e08895cd3c723028b264d42c3d11d9ed8954e72c && mkdir -p research/findings/raw/_regression_battery/shared_ltm_flip_1013 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=44 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s44_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s44_off.json > research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s44_off.log 2>&1; }'
```

SLF.6 `slf_s44_on`:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/e08895cd3c723028b264d42c3d11d9ed8954e72c && mkdir -p research/findings/raw/_regression_battery/shared_ltm_flip_1013 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=44 BRAIN_SHARED_LTM=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s44_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s44_on.json > research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s44_on.log 2>&1; }'
```

SLF.7 `slf_s100_off`:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/e08895cd3c723028b264d42c3d11d9ed8954e72c && mkdir -p research/findings/raw/_regression_battery/shared_ltm_flip_1013 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=100 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s100_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s100_off.json > research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s100_off.log 2>&1; }'
```

SLF.8 `slf_s100_on`:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/e08895cd3c723028b264d42c3d11d9ed8954e72c && mkdir -p research/findings/raw/_regression_battery/shared_ltm_flip_1013 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=100 BRAIN_SHARED_LTM=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s100_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s100_on.json > research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s100_on.log 2>&1; }'
```

SLF.9 `slf_s101_off`:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/e08895cd3c723028b264d42c3d11d9ed8954e72c && mkdir -p research/findings/raw/_regression_battery/shared_ltm_flip_1013 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=101 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s101_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s101_off.json > research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s101_off.log 2>&1; }'
```

SLF.10 `slf_s101_on`:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/e08895cd3c723028b264d42c3d11d9ed8954e72c && mkdir -p research/findings/raw/_regression_battery/shared_ltm_flip_1013 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=101 BRAIN_SHARED_LTM=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s101_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s101_on.json > research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s101_on.log 2>&1; }'
```

SLF.11 `slf_s102_off`:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/e08895cd3c723028b264d42c3d11d9ed8954e72c && mkdir -p research/findings/raw/_regression_battery/shared_ltm_flip_1013 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=102 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s102_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s102_off.json > research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s102_off.log 2>&1; }'
```

SLF.12 `slf_s102_on`:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/e08895cd3c723028b264d42c3d11d9ed8954e72c && mkdir -p research/findings/raw/_regression_battery/shared_ltm_flip_1013 && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=102 BRAIN_SHARED_LTM=1 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s102_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s102_on.json > research/findings/raw/_regression_battery/shared_ltm_flip_1013/slf_s102_on.log 2>&1; }'
```

