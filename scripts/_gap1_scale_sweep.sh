#!/bin/bash
# gap#1 scale de-risk: does more DATA (d256 100k->200k) + a bigger MODEL (d256->d512) lower the WKV LM's deep-NLL
# toward fluency? Sequential (GPU can't parallelize these cleanly). Each writes its own json + log.
cd /home/dant123/Projects/sim
run() {  # $1=d_model $2=n_sentences $3=max_train_sents $4=name
  SIM_BACKEND=cupy .venv/bin/python3 -m research.runners._emerge_wkv_lm_derisk \
    --seeds 42 --corpus data/corpus/tinystories.txt --vocab 4000 \
    --n-sentences "$2" --max-train-sents "$3" --max-eval-sents 3000 \
    --epochs 10 --d-model "$1" --json "research/findings/raw/_gap1_scale_$4.json" \
    > "research/findings/raw/_gap1_scale_$4.log" 2>&1
  echo "DONE_CONFIG $4 rc=$?"
}
run 256 200000 100000 d256_100k
run 256 400000 200000 d256_200k
run 512 400000 200000 d512_200k
echo "ALL_SCALE_CONFIGS_DONE"
