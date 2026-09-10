#!/usr/bin/env bash
# stock_research_queue.sh — STAGE a deep, pause-safe research queue with ZERO Claude/agent tokens.
#
# WHY THIS EXISTS (2026-09-10, owner): Claude usage is the scarce resource (a weekly cap). This stages days of
# NON-Claude compute — the local 3090 (GPU lane) + the mini-PC pool (CPU lane) — that the owner starts/pauses/stops
# themselves via tools/game.sh, with NO Claude in the loop. See research/queue/QUEUE_GUIDE.md for the owner controls.
#
#   bash tools/stock_research_queue.sh            # stage every not-yet-run cell into both lanes, then report
#   bash tools/stock_research_queue.sh --dry-run  # show what WOULD be staged (queues untouched)
#
# DESIGN (the two requirements the owner set):
#   (1) LASTS ~5 days: two deep sweeps on ALREADY-BUILT runners (no fresh build = no agent tokens) —
#       * GPU: the mouth TOKEN-SCALING sweep (the owner's decided #1 mouth fork) pushed into the large-token
#         "does the curve bend" regime on the now-LOCAL fineweb_edu (4.3GB) + wikitext103, across model sizes/seeds.
#       * POOL: the Rank-7 affect-opponent COMPETITION-STRENGTH gain sweep (the finding's own named next rung).
#   (2) MINIMIZES WASTED COMPUTE ON PAUSE: every queued job is a SINGLE self-contained unit (one seed, or one
#       seed's full scaling curve) writing its OWN artifact. `game.sh on` kills+re-queues only the ONE running unit,
#       so a gaming pause loses at most that unit's progress — never a whole 6-seed battery.
#
# RE-DERIVATION GUARD (the ~94-GPU-hour lesson): a cell whose output artifact already exists is SKIPPED. So this is
# safe to re-run any time the queue looks low — it tops up with only the genuinely-unrun cells and never re-derives.
#
# It does NOT start the dispatcher or unpause — the OWNER starts the queue (that was the explicit ask). It only stages.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
PY="$ROOT/.venv/bin/python"
GPUQ="$ROOT/tools/gpu_queue.sh"
POOLQ="$ROOT/tools/pool_queue.sh"
RAW="$ROOT/research/findings/raw"
FINEWEB="$ROOT/data/corpus/fineweb_edu.txt"
WT103="$ROOT/data/corpus/wikitext103.txt"
SEEDS="42 43 44 100 101 102"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

g_added=0; g_skip=0; p_added=0; p_skip=0
mkdir -p "$RAW/gap4" "$RAW/_gencortex_scaling" "$RAW/_affect_gain_sweep"

# gpu_add "<full command>" "<absolute output path this cell writes>"
# Idempotent two ways: skip if the result already EXISTS, or if the cell is already QUEUED (so re-running to top up
# never duplicates an un-run cell — gpu.queue persists while paused).
gpu_add() {
  local cmd="$1" out="$2"
  if [ -s "$out" ]; then g_skip=$((g_skip+1)); return 0; fi
  if grep -qF "$out" "$ROOT/research/queue/gpu.queue" 2>/dev/null; then g_skip=$((g_skip+1)); return 0; fi
  if [ "$DRY" = 1 ]; then echo "  [GPU stage] ${out##*/}"; g_added=$((g_added+1)); return 0; fi
  bash "$GPUQ" add "$cmd" >/dev/null && g_added=$((g_added+1))
}
# pool_add "<command w/ RELATIVE --out>" "<absolute LOCAL landing path>" "<--checked reason>"
# Pool jobs run from ~/derisk-pool/sim on the remote node, so their --out MUST be a repo-relative path (an absolute
# /home/dant123/... path writes to a non-existent path on the node and never rsyncs back). The skip-check is on the
# ABSOLUTE local path where the result lands after the pool syncs it home.
pool_add() {
  local cmd="$1" out_abs="$2" reason="$3"
  if [ -s "$out_abs" ]; then p_skip=$((p_skip+1)); return 0; fi
  if [ "$DRY" = 1 ]; then echo "  [POOL stage] ${out_abs##*/}"; p_added=$((p_added+1)); return 0; fi
  bash "$POOLQ" add "$cmd" --checked "$reason" >/dev/null && p_added=$((p_added+1))
}

echo "== stock_research_queue $( [ "$DRY" = 1 ] && echo '(DRY-RUN)' ) =="

# ---------------------------------------------------------------------------------------------------------------
# GPU LANE — pausable via game.sh. Enqueued in PRIORITY order (gpu_queue is FIFO / head-of-file first).
# ---------------------------------------------------------------------------------------------------------------

# (0) De-monolith: drop the pre-existing 6-seed gap#4 MONOLITH if present (it wastes ~2-3h on a pause). The 6
#     single-seed splits below replace it, pause-safe. Safe rewrite: the daemon is paused while staging.
if [ "$DRY" = 0 ] && grep -q "selfpredict_inengine_6seed.json" "$ROOT/research/queue/gpu.queue" 2>/dev/null; then
  # `|| true`: when the monolith is the ONLY queued line, grep -v emits nothing and exits 1 — without the guard
  # the `&& mv` is skipped and the monolith SURVIVES (real bug, 2026-09-10). The guard makes an empty result valid.
  { grep -v "selfpredict_inengine_6seed.json" "$ROOT/research/queue/gpu.queue" || true; } > "$ROOT/research/queue/gpu.queue.tmp"
  mv "$ROOT/research/queue/gpu.queue.tmp" "$ROOT/research/queue/gpu.queue"
  echo "  removed the gap#4 6-seed monolith (replaced by 6 single-seed splits below)"
fi

# (1) gap#4 decisive — the in-engine self-predicting-interneuron microcircuit. Single-seed splits, FIRST (decisive,
#     owner has been waiting on this verdict; may legitimately return UNDEFINED under the interpretability gate).
for S in $SEEDS; do
  OUT="$RAW/gap4/selfpredict_inengine_s${S}.json"
  gpu_add "SIM_BACKEND=cupy $PY -u -m research.runners._gap4_selfpredict_interneuron_inengine_derisk --full --seeds $S --arms reservoir fixed_fa micro micro_inengine transport_ceiling --hidden 64 --pool-k 16 --n-hidden-layers 2 --epochs 40 --train-subsample 400 --graded-credit --wpi-init noisy --assert-no-transport --out $OUT" "$OUT"
done

# (2) TOKEN-SCALING sweep (owner's #1 mouth fork). Unit = one SEED's full scaling curve (multi-point, so the
#     runner's descending/margin-grows verdict is well-defined per cell) → one artifact per (corpus,d_model,seed).
#     Grids ordered: fineweb-quality-corpus first (the core "does more QUALITY token supply bend the curve" question),
#     then param sizes, then the wikitext103 baseline. n_sentences set high enough to cover the top token-point.
stage_scaling() {  # $1=corpus_path  $2=corpus_tag  $3=d_model  $4="token points..."
  local corpus="$1" tag="$2" d="$3" pts="$4" S OUT
  [ -f "$corpus" ] || { echo "  (skip $tag d$d — corpus missing: $corpus)"; return 0; }
  for S in $SEEDS; do
    OUT="$RAW/_gencortex_scaling/${tag}_d${d}_s${S}.json"
    gpu_add "SIM_BACKEND=cupy $PY -u -m research.runners._gen_cortex_token_supply_scaling_derisk --seeds $S --token-points $pts --corpus $corpus --d-model $d --vocab 2000 --epochs 6 --max-len 48 --n-sentences 400000 --json $OUT" "$OUT"
  done
}
stage_scaling "$FINEWEB" "fineweb" 96  "48000 96000 192000 384000 768000 1536000"
stage_scaling "$FINEWEB" "fineweb" 192 "96000 192000 384000 768000"
stage_scaling "$FINEWEB" "fineweb" 384 "96000 192000 384000 768000"
stage_scaling "$WT103"   "wt103"   96  "96000 192000 384000 768000"
stage_scaling "$WT103"   "wt103"   192 "192000 384000 768000"

# ---------------------------------------------------------------------------------------------------------------
# POOL LANE — CPU on the mini-PCs; runs THROUGH gaming (never paused). Rank-7 affect-opponent gain sweep.
# The finding's OWN named next rung: TO_FS_W/FS_INH_W were imported from a ridge-fit template, never tuned for this
# convergence; Kang/Watanabe/Pu 2024 PNAS = specialization reliability tracks competition STRENGTH, not topology.
# ---------------------------------------------------------------------------------------------------------------
AFFECT_RUNNER="research.runners._affect_onsubstrate_noise_robust_convergence_derisk"
REASON='Rank-7 affect-opponent BOUNDARY (608a06304, 2026-09-08): 6-seed worst-case recall@FP0 UNCHANGED at DEFAULT gains (TO_FS_W=18/FS_INH_W=15, imported from a ridge-fit template, never tuned for this convergence). The finding names its OWN next rung as a competition-STRENGTH gain sweep (Kang/Watanabe/Pu 2024 PNAS: reliability=competition strength, not topology). Verified no artifact at this (to_fs,fs_inh,seed) tuple exists under raw/_affect_gain_sweep. ON-SUBSTRATE (--spiking) sweep = the actual boundary; runner localizes it to FS_INH_W competition strength + neutral-firing floor, which this grid targets. CPU/pool. Default center (18,15) excluded — already the BOUNDARY artifact.'
# Primary axis: FS_INH_W = the competition strength (source comment), swept off default 15 at default drive TO_FS_W=18.
for FS in 8 12 18 22 27; do
  for S in $SEEDS; do
    REL="research/findings/raw/_affect_gain_sweep/opp_to18_fs${FS}_s${S}.json"   # relative: pool runs from ~/derisk-pool/sim
    pool_add "CUDA_VISIBLE_DEVICES=\"\" SIM_BACKEND=numpy .venv/bin/python -u -m $AFFECT_RUNNER --opponent --spiking --to-fs-w 18 --fs-inh-w $FS --seeds $S --out $REL" "$RAW/_affect_gain_sweep/opp_to18_fs${FS}_s${S}.json" "$REASON"
  done
done
# Secondary axis: TO_FS_W = drive strength, swept off default 18 at default competition FS_INH_W=15.
for TO in 12 27; do
  for S in $SEEDS; do
    REL="research/findings/raw/_affect_gain_sweep/opp_to${TO}_fs15_s${S}.json"
    pool_add "CUDA_VISIBLE_DEVICES=\"\" SIM_BACKEND=numpy .venv/bin/python -u -m $AFFECT_RUNNER --opponent --spiking --to-fs-w $TO --fs-inh-w 15 --seeds $S --out $REL" "$RAW/_affect_gain_sweep/opp_to${TO}_fs15_s${S}.json" "$REASON"
  done
done

# ---------------------------------------------------------------------------------------------------------------
echo
echo "  GPU  lane: staged $g_added new, skipped $g_skip already-run"
echo "  POOL lane: staged $p_added new, skipped $p_skip already-run"
if [ "$DRY" = 0 ]; then
  echo
  bash "$GPUQ" status 2>/dev/null | sed 's/^/  /'
  echo "  pool queue depth: $(bash "$POOLQ" depth 2>/dev/null || echo '?')"
  echo
  echo "  START the queue when ready:   bash tools/gpu_queue.sh start && bash tools/game.sh off"
  echo "  PAUSE for gaming:             bash tools/game.sh on        (RESUME: bash tools/game.sh off)"
  echo "  full owner guide:             research/queue/QUEUE_GUIDE.md"
fi
