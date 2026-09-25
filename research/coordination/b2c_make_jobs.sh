#!/usr/bin/env bash
# b2c_make_jobs.sh -- write Battery B2c's queue source: 516 job lines (43 sharded rows x 6 seeds x 2 arms), one
# base line then its flipcand twin for every (seed, row), seed by seed, all pinned to F2.
#
# Prereg: research/findings/2026-09-25-production-default-battery-B2c-paired-flip-PREREGISTRATION.md
#   F2   = fd29040db19987819461693aaf385977e45840ef (origin/main when the prereg was filed; contains 3bdf8b619,
#          2ac0fb245 and cce3c1dbd)
#   arms = b2c0925-base      (--no-fixes, adequate probes, no BRAIN_* token)
#          b2c0925-flipcand  (the same, plus exactly BRAIN_DA_TAG_CAPTURE=1 BRAIN_DA_TAG_CAPTURE_CLOCK=turn
#                             BRAIN_SLEEP_REPLAY_CAPTURE=1 in the env prefix)
#
# The row list is FROZEN below (F2's registry, 43 rows of lb_shard.MEASURABLE_KINDS, registry order) and passed with
# --faculties, so the output does not depend on the registry of the checkout this runs in. The guard
# `.venv/bin/python tools/assert_flipped_defaults.py && ` goes AFTER the pinned `cd` (it must run inside F2's tree),
# as in B2b's job file.
#
# Usage:  bash research/coordination/b2c_make_jobs.sh [--record-pin] [OUT]
#   OUT defaults to research/coordination/b2c0925_jobs.txt. --record-pin also passes --pin F2 to both `jobs` calls,
#   which writes PIN.txt (and, for flipcand, EXPECT_ENV.txt) under research/findings/raw/_load_bearing/_shards/<tag>/
#   of THIS checkout -- run it that way once, in the primary checkout, just before the first wave is queued.
# Exit: 0 = written and every static check passed; 1 = a check failed (nothing written); 2 = usage.
set -euo pipefail
export LC_ALL=C   # byte-order sort: the env-token comparisons below must not depend on the locale
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

F2=fd29040db19987819461693aaf385977e45840ef
ROOT_DIR="~/derisk-pool/revisions/$F2"     # literal tilde: expanded on the node, not here
SEEDS=(42 43 44 100 101 102)
PAIR_ENV=(BRAIN_DA_TAG_CAPTURE=1 BRAIN_SLEEP_REPLAY_CAPTURE=1 BRAIN_DA_TAG_CAPTURE_CLOCK=turn)
FACS=(one-brain-substrate comprehension-monitor comprehension-learned-animacy-cue comprehension-learned-verb-selects
      noncontradiction-gate affect-coloring affect-drives-response affect-marker-spiking-wta da-mode-drives-response
      da-gated-encoding source-provenance-honesty common-ground-drives swap-drives-response wm-binding-advanced
      prospective-memory pragmatic-implicature surprise-monitor metacog-monitor worldmodel-forward curiosity-followup
      reconsolidation episodic-memory discourse-register gnw-multistep-deliberation self-initiated-utterance
      vision-identity-spiking-hmax bg-action-selection open-ended-generation gnw-deliberation value-driven-choice
      confidence-forthcomingness d5-consolidate sleep-replay affective-tom causal-whatif spiking-anaphor gnw-bus
      multiref-competition affect-appraisal-interoceptive spiking-qroute learned-referent onebrain-xedge
      surprise-salience-snc-afferent)

RECORD_PIN=0
OUT=research/coordination/b2c0925_jobs.txt
for arg in "$@"; do
  case "$arg" in
    --record-pin) RECORD_PIN=1 ;;
    -*) echo "usage: $0 [--record-pin] [OUT]" >&2; exit 2 ;;
    *) OUT=$arg ;;
  esac
done
[ "${#FACS[@]}" -eq 43 ] || { echo "⛔ frozen row list has ${#FACS[@]} rows, want 43" >&2; exit 1; }
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
PIN_ARGS=(); [ "$RECORD_PIN" -eq 1 ] && PIN_ARGS=(--pin "$F2")

gen() {  # $1 = tag, rest = extra env
  local tag=$1; shift
  local extra=(); [ $# -gt 0 ] && extra=(--extra-env "$@")
  SIM_NO_PROVENANCE=1 "$PY" tools/lb_shard.py jobs --seeds "${SEEDS[@]}" --tag "$tag" --no-fixes \
      --probe-set adequate --repeats 2 --root "$ROOT_DIR" --faculties "${FACS[@]}" "${extra[@]}" "${PIN_ARGS[@]}" \
    | sed "s#^cd $ROOT_DIR && #&.venv/bin/python tools/assert_flipped_defaults.py \&\& #"
}

TMP=$(mktemp -d "${TMPDIR:-/tmp}/b2c_jobs.XXXXXX"); trap 'rm -rf "$TMP"' EXIT
gen b2c0925-base > "$TMP/base"
gen b2c0925-flipcand "${PAIR_ENV[@]}" > "$TMP/flip"
paste -d '\n' "$TMP/base" "$TMP/flip" > "$TMP/all"

fail=0
check() { echo "⛔ $*" >&2; fail=1; }
[ "$(wc -l < "$TMP/base")" -eq 258 ] || check "base arm has $(wc -l < "$TMP/base") lines, want 258"
[ "$(wc -l < "$TMP/flip")" -eq 258 ] || check "flipcand arm has $(wc -l < "$TMP/flip") lines, want 258"
HEAD="cd $ROOT_DIR && .venv/bin/python tools/assert_flipped_defaults.py && mkdir -p research/findings/raw/_load_bearing/_shards/"
bad=$(awk -v h="$HEAD" 'index($0,h)!=1 || $0 !~ /\/lb\.json$/' "$TMP/all" | wc -l)
[ "$bad" -eq 0 ] || check "$bad line(s) do not start with the pinned cd + guard or do not end in /lb.json"
grep -qE 'BRAIN_[A-Z0-9_]+=' "$TMP/base" && check "a base line carries a BRAIN_* token"
want=$(printf '%s\n' "${PAIR_ENV[@]}" | sort | tr '\n' ' ')
while IFS= read -r l; do
  got=$(printf '%s\n' "$l" | grep -oE 'BRAIN_[A-Z0-9_]+=[^ ]+' | sort | tr '\n' ' ')
  [ "$got" = "$want" ] || { check "flipcand line carries [$got], want [$want]"; break; }
done < "$TMP/flip"
# twins: a flipcand line minus its three tokens, with its tag renamed, must equal its base line exactly
norm=$(sed -e 's/ BRAIN_DA_TAG_CAPTURE=1//; s/ BRAIN_DA_TAG_CAPTURE_CLOCK=turn//; s/ BRAIN_SLEEP_REPLAY_CAPTURE=1//' \
           -e 's#/b2c0925-flipcand/#/b2c0925-base/#g' "$TMP/flip" | cmp -s - "$TMP/base" && echo same || echo differ)
[ "$norm" = same ] || check "a flipcand line differs from its base twin in more than the tag and the three tokens"
[ "$fail" -eq 0 ] || { echo "⛔ nothing written" >&2; exit 1; }
cp "$TMP/all" "$OUT"
echo "[b2c-jobs] wrote $OUT: $(wc -l < "$OUT") lines (258 base + 258 flipcand, interleaved), pinned to $F2" \
     "$([ "$RECORD_PIN" -eq 1 ] && echo '; PIN.txt/EXPECT_ENV.txt recorded')"
