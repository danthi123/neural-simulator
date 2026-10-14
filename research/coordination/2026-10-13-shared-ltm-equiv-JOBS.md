# Shared-LTM equivalence battery (step 1 of the one-brain shared-memory plan), 2026-10-13

Ruled by Claude 2026-10-13 (runbook RULING shared-ltm-equiv-1013). Branch research/shared-ltm-equiv @ bf91d574738912d70d417285a741d685c0e1ee2d (= claude-build/shared-ltm, unmerged). Dev seeds 7 and 8 only. Template: the registered JOBS line 46 of 2026-10-12-validation-battery-JOBS.md, with only the revision, seed, output dir/tag and BRAIN_SHARED_LTM=0 (off arm) changed (generator: Claude's sim-rev/sharedltm/gen.py).

Pass bar (byte identity, docs/TERMS.md): the worker loads the wikidata_100k LTM tier per session (3 sessions: emo, bc, selfinit). For each seed, every per-turn response record in sl_sN_on.json equals sl_sN_off.json after dropping only wall-clock, latency and RSS fields; list every other differing key (any difference = FAIL, report it, do not explain it away). Report maxrss_kb for both arms (expected: on much lower).

SL.1:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/bf91d574738912d70d417285a741d685c0e1ee2d && mkdir -p research/findings/raw/_regression_battery/shared_ltm_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=7 BRAIN_IDLE_REPLAY=0 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_equiv/sl_s7_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_equiv/sl_s7_on.json > research/findings/raw/_regression_battery/shared_ltm_equiv/sl_s7_on.log 2>&1; }'
```

SL.2:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/bf91d574738912d70d417285a741d685c0e1ee2d && mkdir -p research/findings/raw/_regression_battery/shared_ltm_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=7 BRAIN_IDLE_REPLAY=0 BRAIN_SHARED_LTM=0 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_equiv/sl_s7_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_equiv/sl_s7_off.json > research/findings/raw/_regression_battery/shared_ltm_equiv/sl_s7_off.log 2>&1; }'
```

SL.3:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/bf91d574738912d70d417285a741d685c0e1ee2d && mkdir -p research/findings/raw/_regression_battery/shared_ltm_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=8 BRAIN_IDLE_REPLAY=0 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_equiv/sl_s8_on_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_equiv/sl_s8_on.json > research/findings/raw/_regression_battery/shared_ltm_equiv/sl_s8_on.log 2>&1; }'
```

SL.4:

```
python3 -c 'import json,resource,subprocess,sys;r=subprocess.run(sys.argv[1:]);t=resource.getrusage(resource.RUSAGE_CHILDREN);print(json.dumps({"rc":r.returncode,"maxrss_kb":t.ru_maxrss}))' bash -c 'cd ~/derisk-pool/revisions/bf91d574738912d70d417285a741d685c0e1ee2d && mkdir -p research/findings/raw/_regression_battery/shared_ltm_equiv && { .venv/bin/python tools/assert_flipped_defaults.py && .venv/bin/python tools/assert_iw1_defaults.py --expect-on "d6-hebbian-write,xedge-surprise-worldmodel" && .venv/bin/python tools/assert_iw2_defaults.py false-belief-chat,transitive-chat,neural-selfid && bash tools/mem_ok.sh 21 1 && bash tools/memcap.sh 22 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= BRAIN_CHAT_SEED=8 BRAIN_IDLE_REPLAY=0 BRAIN_SHARED_LTM=0 BRAIN_IDLE_REPLAY_LEAKLOG=$PWD/research/findings/raw/_regression_battery/shared_ltm_equiv/sl_s8_off_rss.jsonl .venv/bin/python -u -m research.runners.onebrain_regression_battery --worker --env {} --turns emo,bc_a,bc_b,selfinit --out research/findings/raw/_regression_battery/shared_ltm_equiv/sl_s8_off.json > research/findings/raw/_regression_battery/shared_ltm_equiv/sl_s8_off.log 2>&1; }'
```

