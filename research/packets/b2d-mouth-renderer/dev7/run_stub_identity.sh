#!/usr/bin/env bash
# Dev seed 7: the stub (default) arm of the three mouth rows, at the renderer-change revision P and at its parent on
# main M, each from its own exported git-archive tree. Acceptance (B2d prereg, Mouth-channel arm): with the renderer
# left at its default, the arm files are byte-identical before and after the change. Two LB runs at a time.
set -uo pipefail
P=${1:?P sha}; M=${2:?M sha}
ROWS="affect-coloring affect-appraisal-interoceptive affect-drives-response"
FLAGS="LB_AFFECT_DRIVE_PROBE=1 LB_BG_SELECT_DRIVE_PROBE=1 LB_CG_DRIVE_PROBE=1 LB_DISCOURSE_REGISTER_DRIVE_PROBE=1 LB_EPISODIC_DRIVE_PROBE=1 LB_NONCONTRADICTION_DRIVE_PROBE=1 LB_OPEN_ENDED_DISTRIB_PROBE=1 LB_PMEM_DRIVE_PROBE=1 LB_SURPRISE_CONFIRM_PROBE=1"
one() {  # <sha> <row>
  local sha=$1 row=$2 tag="b2d0929-stubid-dev7-${1:0:7}"
  local out="research/findings/raw/_load_bearing/_shards/$tag/s7/$row/lb.json"
  cd "$HOME/derisk-pool/revisions/$sha" && .venv/bin/python tools/assert_flipped_defaults.py && mkdir -p "$(dirname "$out")" \
    && until bash tools/mem_ok.sh 8 4 >/dev/null 2>&1; do sleep 30; done \
    && bash tools/memcap.sh 12 -- env SIM_BACKEND=numpy OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $FLAGS \
       .venv/bin/python -u -m research.runners.load_bearing_fraction --only "$row" --seed 7 --repeats 2 --out "$out" \
       > "$HOME/derisk-pool/revisions/$sha/research/findings/raw/_load_bearing/_shards/$tag/s7_$row.log" 2>&1
  echo "$(date +%T) done sha=${sha:0:7} row=$row rc=$?"
}
for row in $ROWS; do
  one "$P" "$row" & one "$M" "$row" & wait
done
echo "ALL-DONE"
