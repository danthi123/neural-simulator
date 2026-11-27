#!/bin/bash
# Runs ON a pool node (CPU fp32). Mix-night screen, prereg brain/research/2026-11-27-mix-night-prereg.md (DRAFT: do not launch
# before its pre-data review is recorded there).
#
# Placement: each pool node holds ONE seed's KR snapshots (pool40 s319, pool41 s320, pool42 s321), so the seed is the one
# whose snapshots are on this node: exactly one $C/snaps_s<seed> must exist (SEED=<s> may be given; it must match).
# A5f needs the foreign seed's n13..n15 episodes.json in $C/foreign_s<F> (F = 320/321/319 for 319/320/321); launch_w.sh ships them.
#
#   bash run_node.sh strata          store-only: P2 stratum sizes at n12..n15 (no model, seconds)
#   bash run_node.sh measure         step 0, measure first: A0r (night 13), then A0 and A2 night 13 only (--max-night 13; the
#                                    run resumes them), each under /usr/bin/time -v; then the determinism pair (detcheck,
#                                    A0r vs A0 n13). Writes $O/measure_s<S>.txt (max RSS kB of A0r and of A2 = the worst-case
#                                    arm, wall, threads, det verdict). A determinism MISS exits 3 and writes no measure file:
#                                    the registered fallback (prereg section 5) is a fresh out dir with THREADS=1.
#                                    MEMCAP_GB sets the cap; it is REQUIRED on nodes under 24 GB (pool41/42: MEMCAP_GB=11, so
#                                    mem_ok needs 12 GB free), else the default is MemTotal - 2 GB.
#   THREADS=<n>                      OMP/torch threads (default nproc); recorded in every unit; run must match measure.
#   bash run_node.sh run [arm ...]   the rest (default all arms; finished units skip). Needs the measure file with det ok;
#                                    the cap is ceil(1.3 x max(RSS A0r, RSS A2)) + 1 GB and must fit MemTotal - 1 GB, else it
#                                    refuses. A baseline arm (A0L A0 A0e A0k) failing for any reason stops the node. Each arm
#                                    waits for tools/mem_ok.sh (margin 1 GB, poll 5 min, up to MAX_WAIT_H=12 h) and is itself
#                                    timed with /usr/bin/time -v ($O/time_<arm>_s<S>.txt), so every arm's peak is on record.
# One python process per (arm, seed), sequential; units are written atomically and a rerun resumes (finished units skipped,
# a unit written under another commit / host / device stops the run).
R=~/brain-lane/route_repo; C=$R/_scratch/route/kr
MODE=${1:-}; shift
case $MODE in strata|measure|run) ;; *) echo "usage: run_node.sh strata|measure|run [arm ...]"; exit 2;; esac
COMMIT=$(cat $R/COMMIT 2>/dev/null)
[ -n "$COMMIT" ] || { echo "$R/COMMIT missing or empty: ship the code with its COMMIT file"; exit 2; }
HAVE=$(cd $C 2>/dev/null && ls -d snaps_s3* 2>/dev/null | sed 's/snaps_s//')
[ "$(echo $HAVE | wc -w)" = 1 ] || { echo "$(hostname): expected exactly one snaps_s<seed> in $C, found: '$HAVE'"; exit 2; }
S=${SEED:-$HAVE}
[ "$S" = "$HAVE" ] || { echo "$(hostname) holds snaps_s$HAVE, not s$S: seed $S runs on its own node"; exit 2; }
case $S in 319) F=320;; 320) F=321;; 321) F=319;; *) echo "seed must be 319/320/321, node holds s$S"; exit 2;; esac
OFFFRAME=${OFFFRAME:-$R/brain/eval/mix_night_offframe.json}
OFFFRAME_SHA256=${OFFFRAME_SHA256:-b1a95684b8caeec1f6564493afa35a6cddc280d791f8466a2c73ccbc49e6f268}
PY=~/brain-lane/venv/bin/python; [ -x "$PY" ] || PY=~/derisk-pool/sim/.venv/bin/python
export BRAIN_MODEL_DIR=~/brain-lane/models PYTHONPATH=$R:~/brain-lane/pylib OMP_NUM_THREADS=${THREADS:-$(nproc)} POOL_NODE=1
O=$C/mix_night; mkdir -p $O; cd $R || exit 1
TOTAL_GB=${MN_TEST_TOTAL_GB:-$(awk '/^MemTotal:/{print int($2/1048576)}' /proc/meminfo)}   # override: tests only
TIME=${MN_TEST_TIME:-/usr/bin/time}                                                         # override: tests only

if [ "$MODE" = strata ]; then
  $PY -u -m brain.experiments.mix_night_screen strata --snaps $C/snaps_s$S | tee $O/strata_s$S.json
  exit ${PIPESTATUS[0]}
fi
[ -x "$TIME" ] || { echo "$TIME missing on $(hostname): measure-first needs GNU time (-v)"; exit 2; }

unit() {   # unit <arm> <cap_gb> [extra args]
  local A=$1 CAP=$2 W=0; shift 2
  until bash $R/tools/mem_ok.sh $CAP 1 >> $O/memwait_s$S.log 2>&1; do
    W=$((W + 1)); [ $W -gt $(( ${MAX_WAIT_H:-12} * 12 )) ] && { echo "s$S $A: RAM never free for ${CAP}G; stop"; return 75; }
    sleep 300
  done
  if [ "$A" = A5f ]; then
    for k in 13 14 15; do [ -f $C/foreign_s$F/n$k/episodes.json ] || { echo "s$S A5f: $C/foreign_s$F/n$k/episodes.json missing"; return 2; }; done
  fi
  echo "start $(hostname) $(date -u +%FT%TZ) s$S $A cap ${CAP}G commit $COMMIT" >> $O/done_s$S.txt
  bash $R/tools/memcap.sh $CAP -- $TIME -v -o $O/time_${A}_s$S.txt \
    $PY -u -m brain.experiments.mix_night_screen run --seed $S --arm $A \
    --life $C/lifeKR_s$S.json.json --snaps $C/snaps_s$S --foreign-snaps $C/foreign_s$F \
    --offframe "$OFFFRAME" --offframe-sha256 "$OFFFRAME_SHA256" --model 1.5b --device cpu \
    --out-dir $O --work $O/work_${A}_s$S "$@" >> $O/${A}_s$S.log 2>&1
  local RC=$?
  local RSS=$(awk -F': ' '/Maximum resident set size/{print $2}' $O/time_${A}_s$S.txt 2>/dev/null)
  echo "s$S $A rc $RC maxrss_kB ${RSS:-?} $(date -u +%FT%TZ)" >> $O/done_s$S.txt
  return $RC
}

rss_of() { awk -F': ' '/Maximum resident set size/{print $2}' $O/time_$1_s$S.txt 2>/dev/null; }
if [ "$MODE" = measure ]; then
  [ -f $O/measure_s$S.txt ] && { cat $O/measure_s$S.txt; exit 0; }
  for A in A0r A0 A2; do
    [ -f $O/u_${A}_s${S}_n13.json ] && { echo "u_${A}_s${S}_n13.json exists without a measure file: a resumed unit measures nothing; move $O aside"; exit 2; }
  done
  if [ -z "$MEMCAP_GB" ] && [ "$TOTAL_GB" -lt 24 ]; then
    echo "$(hostname) has ${TOTAL_GB}G: give MEMCAP_GB explicitly (11 on a 15 GB node; mem_ok needs cap + 1 free)"; exit 2
  fi
  CAP=${MEMCAP_GB:-$(( TOTAL_GB - 2 ))}
  T0=$(date +%s)
  unit A0r $CAP; RC=$?
  [ "$RC" = 0 ] && [ -n "$(rss_of A0r)" ] || { echo "measure A0r failed rc $RC (see $O/A0r_s$S.log)"; exit 1; }
  WALL=$(awk -F': ' '/Elapsed \(wall clock\)/{print $2}' $O/time_A0r_s$S.txt 2>/dev/null)
  unit A0 $CAP --max-night 13; RC=$?
  [ "$RC" = 0 ] || { echo "measure A0 n13 failed rc $RC (see $O/A0_s$S.log)"; exit 1; }
  $PY -m brain.experiments.mix_night_screen detcheck --dir $O --seed $S > $O/detcheck_s$S.json 2>&1; DRC=$?
  cat $O/detcheck_s$S.json
  [ "$DRC" = 0 ] || { echo "DETERMINISM MISS on s$S at ${OMP_NUM_THREADS} threads: registered fallback = move $O aside, rerun measure with THREADS=1 (prereg section 5); no measure file written"; exit 3; }
  unit A2 $CAP --max-night 13; RC=$?
  [ "$RC" = 0 ] && [ -n "$(rss_of A2)" ] || { echo "measure A2 n13 failed rc $RC (see $O/A2_s$S.log): A2 does not fit ${CAP}G"; exit 1; }
  printf 'host %s\ncommit %s\nthreads %s\nmaxrss_kB %s\nmaxrss_A0r_kB %s\nmaxrss_A2_kB %s\nwall_A0r %s\nstep0_s %s\ndet ok\ncap_used_GB %s\nmem_total_GB %s\n' \
    "$(hostname)" "$COMMIT" "$OMP_NUM_THREADS" "$(printf '%s\n%s\n' "$(rss_of A0r)" "$(rss_of A2)" | sort -n | tail -1)" \
    "$(rss_of A0r)" "$(rss_of A2)" "$WALL" "$(( $(date +%s) - T0 ))" "$CAP" "$TOTAL_GB" > $O/measure_s$S.txt
  cat $O/measure_s$S.txt
  exit 0
fi

# MODE=run
[ -f $O/measure_s$S.txt ] || { echo "no $O/measure_s$S.txt: run 'bash run_node.sh measure' first"; exit 2; }
grep -qx 'det ok' $O/measure_s$S.txt || { echo "measure_s$S.txt has no 'det ok': the determinism pair did not pass"; exit 2; }
MT=$(awk '/^threads/{print $2}' $O/measure_s$S.txt)
[ "$MT" = "$OMP_NUM_THREADS" ] || { echo "measure ran at $MT threads, this run at $OMP_NUM_THREADS: set THREADS=$MT"; exit 2; }
RSS=$(awk '/^maxrss_kB/{print $2}' $O/measure_s$S.txt)
CAP=${MEMCAP_GB:-$(awk -v r="$RSS" 'BEGIN{g = 1.3 * r / 1048576; c = int(g); if (c < g) c++; print c + 1}')}
[ "$CAP" -le $(( TOTAL_GB - 1 )) ] || { echo "cap ${CAP}G (1.3 x measured max $RSS kB + 1) does not fit $(hostname) (${TOTAL_GB}G); stop"; exit 2; }
ARMS="${*:-A0L A0 A0e A0k A1 A2 A3 A4 A5s A5f}"
for A in $ARMS; do
  unit $A $CAP; RC=$?
  [ $RC = 75 ] && exit 75
  case $A in A0L|A0|A0e|A0k) [ $RC = 0 ] || { echo "s$S baseline $A failed rc $RC: every candidate needs it; stop"; exit $RC; };; esac
done
