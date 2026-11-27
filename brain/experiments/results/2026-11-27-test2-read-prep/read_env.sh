# source this from bash, in /home/dant123/Projects/sim/_scratch/wt-mixday (never edit it in place while a read is running)
#   bash -c 'cd /home/dant123/Projects/sim/_scratch/wt-mixday && source brain/experiments/results/2026-11-27-test2-read-prep/read_env.sh && echo ${B[@]}'
export PY=/home/dant123/Projects/sim/.venv/bin/python CUDA_VISIBLE_DEVICES=""   # host scorers only: no GPU, no model
export PYTHONPATH=/home/dant123/Projects/sim/_scratch/wt-mixday
AWSROOT=${AWSROOT:-/home/dant123/Projects/sim/_scratch/wt-test2/_scratch/brainv2/aws}   # pulled batches: <batch>/life<ARM>_s<S>.json.json
PREP=brain/experiments/results/2026-11-27-test2-read-prep
RD=${RD:-brain/experiments/results/2026-11-28-test2-read}     # the real read's output dir (new dir, never the prep dir)
SEEDS="42 43 44 100 101 102"
pick() {  # pick ARM SEED -> the single life json (newest batch if several; warns on stderr)
  local f; f=$(ls -1 "$AWSROOT"/*/life$1_s$2.json.json 2>/dev/null | sort | tail -1)
  [ -n "$f" ] || { echo "MISSING life$1_s$2" >&2; return 1; }
  [ "$(ls -1 "$AWSROOT"/*/life$1_s$2.json.json | wc -l)" -gt 1 ] && echo "WARN several copies of life$1_s$2, using $f" >&2
  echo "$f"
}
B=(); KP=(); PR=()
for s in $SEEDS; do
  f=$(pick B $s) && B+=("$f"); f=$(pick KP $s) && KP+=("$f"); f=$(pick PR $s) && PR+=("$f")
done
echo "B ${#B[@]}/6  KP ${#KP[@]}/6  PR ${#PR[@]}/6"
