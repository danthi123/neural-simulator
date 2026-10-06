#!/usr/bin/env bash
# tools/pool_canary.sh -- the POOL CANARY: run ONE queue line on ONE pool node, judge it, and record the verdict
# for its (runner module, revision sha) in research/queue/pool_canaries.tsv (2026-09-30).
#
#   bash tools/pool_canary.sh run <node>|auto '<line>' [--exact] [--wait-s N]   # run it, judge it, record PASS/FAIL
#   bash tools/pool_canary.sh record <module> <sha> PASS|FAIL <node> gate|dev '<evidence>'   # a canary run elsewhere
#   bash tools/pool_canary.sh has <module> <sha>        # exit 0 iff the pair may take gate-seed lines (below)
#   bash tools/pool_canary.sh check-line '<line>'       # the same, keyed from a queue line (pool_queue.sh add uses it)
#   bash tools/pool_canary.sh key '<line>'              # print "<module><TAB><sha>" for a line
#   bash tools/pool_canary.sh list                      # print the records
#
# WHY. On 2026-09-29 about 470 GATE-seed lines were queued for eight runner families after only workstation dev
# checks. Run for real on pool41/pool42 on 2026-09-30, three families died on every line: host_share,
# stage1_exit_gate and stage1_c1_category spawned their '--worker --seed N' without --allow-gate-seed, so each
# worker's own gate-seed guard exited in argparse (the same defect stage2_exit_gate had already fixed), and a
# fourth (iw1_default_identity) died at import because the pool venvs lacked httpx. A dev-seed workstation check
# exercises neither a gate-seed code path nor the pool's venv. `pool_queue.sh add` therefore refuses a gate-seed
# line until a canary of the same runner at the same revision has passed ON A POOL NODE (research/FAILURE_LOG.md,
# 2026-09-30).
#
# THE RULE (`has`, `check-line`): a (module, revision) may take gate-seed lines iff (a) its latest record with
# seed_kind=gate is PASS and (b) its latest record of any kind is not FAIL. A dev-seed canary never satisfies (a):
# the 2026-09-30 defects lived only on the gate-seed path, where a dev seed never goes. A canary's seed_kind is
# `gate` when its line passes a gate seed (42/43/44/100/101/102, detected as pool_queue.sh's gate-seed guard
# detects them). A canary of a gate-seed line is still one of the committed-prereg lines being queued; its result
# JSON goes to _pool_canary/ and only its error fields are read here, never an outcome value.
#
# WHAT `run` DOES. It launches the line on <node> through plain ssh, the same way tools/pool_autodispatch.sh
# launches a queued job (from ~/derisk-pool/sim, setsid, POOL_JOB_ID/POOL_JOB_MEM_GB set so the dispatcher counts
# the canary's memory, thread counts 1), never through the queue. By default every path under
# research/findings/raw/ in the runner's argv (after `-m <module>`) is redirected into
# research/findings/raw/_pool_canary/<id>/, so a canary never writes gate data; --exact runs the line verbatim.
# The job's stdout+stderr go to ~/derisk-pool/canary/<id>.log on the node (outside research/findings/raw, so
# pool-sync does not copy a 100+ MB log back). Verdict:
#   PASS  rc 0, at least one JSON result written under the output paths, and no JSON result carrying an error
#         (a non-empty string `error` or any non-empty `_error` value): a runner that catches an exception per
#         turn or per part still exits 0, so rc alone is not a pass (host_share and sleep_transitive_inference).
#   FAIL  anything else, except a line whose own `tools/mem_ok.sh` refused to start it: that says nothing about
#         the runner, so it is INCONCLUSIVE (exit 3, nothing recorded) -- as is a job still running at --wait-s.
# Exit 0 on PASS, 1 on FAIL, 3 on inconclusive, 2 on a usage error.
#
# SANDBOX. The bubblewrap pool-job sandbox (branch research/pool-job-sandbox, tools/pool_job_sandbox.sh and
# tools/pool_sandbox_selftest.sh) was not merged when this was written, so `run` uses a plain ssh launch. Once
# the dispatcher launches every job sandboxed, `run` should launch through the same render
# (`pool_autodispatch.sh --render-remote-command <job> sandbox`) so the canary runs where the job will run.
# Canaries run through pool_sandbox_selftest.sh --job are entered with `record`, which requires the evidence.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=tools/pool_revision_marker.sh
source "$ROOT/tools/pool_revision_marker.sh"
# shellcheck source=tools/pool_nodes.sh
source "$ROOT/tools/pool_nodes.sh"
# Shared repo root = the parent of the ONE git common dir (the resolution tools/gpu_queue.sh uses), so every
# worktree reads the same queue on whichever host holds the checkout (was the workstation path until the
# 2026-09-30 move of the project home to the server, docs/plans/2026-09-30-home-to-server-MIGRATION.md).
SHARED_ROOT=$ROOT; _GC=$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) && SHARED_ROOT=$(dirname "$_GC")
CANARIES="${POOL_CANARIES_PATH:-$SHARED_ROOT/research/queue/pool_canaries.tsv}"
POOL_SSH_CONFIG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
SSH_F=(); [ -f "$POOL_SSH_CONFIG" ] && SSH_F=(-F "$POOL_SSH_CONFIG")
HEADER=$'# recorded_at\tmodule\tsha\tverdict\tnode\tseed_kind\tvia\tevidence'
GATE_SEEDS_LIST="42 43 44 100 101 102"   # the same list and detection as tools/pool_queue.sh's gate-seed guard

usage() { sed -n '4,9p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }

# bare_line <line> -- accepts a raw queue line too: drops a leading "<epoch><TAB>" and the "  #checked:" reason.
bare_line() {
  local s="$1"
  [[ $s =~ ^[0-9]+$'\t'(.*)$ ]] && s="${BASH_REMATCH[1]}"
  s="${s%%  #checked:*}"
  printf '%s' "$s"
}

# line_module <cmd> -- the Python module the line runs: a research.runners module if any, else the first `-m X`.
line_module() {
  local m
  m=$(printf '%s' "$1" | grep -oE -- '-m +research\.runners\.[A-Za-z0-9_]+' | head -1 | awk '{print $2}')
  [ -n "$m" ] || m=$(printf '%s' "$1" | grep -oE -- '(^|[[:space:]])-m +[A-Za-z_][A-Za-z0-9_.]*' | head -1 | awk '{print $2}')
  printf '%s' "$m"
}

# line_rev <cmd> -- the pinned revision (cd ~/derisk-pool/revisions/<sha>), as written in the line.
line_rev() {
  printf '%s' "$1" | grep -oE 'derisk-pool/revisions/[0-9a-f]{7,40}' | head -1 | sed 's#.*/##'
}

# line_seed_kind <cmd> -- "gate" when any --seed/--seeds value is a gate seed, else "dev".
line_seed_kind() {
  local s g
  for s in $(printf '%s' "$1" | grep -oE -- '--seeds?(=|[[:space:]]+)[0-9]+([,[:space:]]+[0-9]+)*' | grep -oE '[0-9]+'); do
    for g in $GATE_SEEDS_LIST; do [ "$s" = "$g" ] && { printf gate; return; }; done
  done
  printf dev
}

# full_sha <sha> -- the full commit sha when this repo knows it, else the sha as given.
full_sha() {
  local f
  f=$(git -C "$ROOT" rev-parse --verify -q "$1^{commit}" 2>/dev/null) && [ -n "$f" ] && { printf '%s' "$f"; return; }
  printf '%s' "$1"
}

# latest_record <module> <sha> [kind] -- the newest record for the pair, of that seed_kind when given
# ("verdict<TAB>recorded_at<TAB>node<TAB>seed_kind<TAB>evidence"). Shas match when one is a prefix of the other
# (both are >= 7 hex characters by construction).
latest_record() {
  local mod="$1" sha kind="${3:-}"
  sha=$(full_sha "$2")
  [ -f "$CANARIES" ] || return 0
  awk -F'\t' -v m="$mod" -v s="$sha" -v k="$kind" '
    $0 ~ /^[[:space:]]*(#|$)/ { next }
    $2 == m && (index($3, s) == 1 || index(s, $3) == 1) && (k == "" || $6 == k) {
      last = $4 "\t" $1 "\t" $5 "\t" $6 "\t" $8 }
    END { if (last != "") print last }' "$CANARIES"
}

describe() {  # describe <record> -- "FAIL (2026-09-30T.. on pool41, gate seed: rc=1 ...)"
  printf '%s (%s on %s, %s seed: %s)' "$(printf '%s' "$1" | cut -f1)" "$(printf '%s' "$1" | cut -f2)" \
    "$(printf '%s' "$1" | cut -f3)" "$(printf '%s' "$1" | cut -f4)" "$(printf '%s' "$1" | cut -f5 | cut -c1-160)"
}

# gate_ok <module> <sha> -- 0 iff the pair may take gate-seed lines (THE RULE above); prints why not otherwise.
gate_ok() {
  local any gate
  any=$(latest_record "$1" "$2")
  if [ -z "$any" ]; then
    printf 'no pool canary record for (%s, %s) in %s' "$1" "$2" "$CANARIES"; return 1
  fi
  if [ "$(printf '%s' "$any" | cut -f1)" = FAIL ]; then
    printf 'the latest pool canary for (%s, %s) is %s' "$1" "$2" "$(describe "$any")"; return 1
  fi
  gate=$(latest_record "$1" "$2" gate)
  if [ -z "$gate" ]; then
    printf 'no GATE-seed pool canary for (%s, %s): a dev-seed canary never runs the gate-seed code path' "$1" "$2"
    return 1
  fi
  if [ "$(printf '%s' "$gate" | cut -f1)" != PASS ]; then
    printf 'the latest gate-seed pool canary for (%s, %s) is %s' "$1" "$2" "$(describe "$gate")"; return 1
  fi
  return 0
}

# append_record <module> <sha> <verdict> <node> <seed_kind> <via> <evidence> -- one line, under a lock.
append_record() {
  local mod="$1" sha verdict="$3" node="$4" kind="$5" via="$6" ev="$7"
  sha=$(full_sha "$2")
  mkdir -p "$(dirname "$CANARIES")"
  ( flock -w 60 9 || { echo "could not lock $CANARIES" >&2; exit 1; }
    [ -s "$CANARIES" ] || printf '%s\n' "$HEADER" > "$CANARIES"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Iseconds)" "$mod" "$sha" "$verdict" "$node" "$kind" "$via" "$ev" >> "$CANARIES"
  ) 9>"$CANARIES.lock"
}

valid_field() {  # valid_field <value> -- no tab, newline or other control character
  case "$1" in *[![:print:]]*) return 1 ;; esac
  return 0
}

rssh() {  # rssh <node> <remote command> -- the dispatcher's batch-mode ssh
  local node="$1"; shift
  timeout "${POOL_CANARY_SSH_TIMEOUT:-60}" ssh -n "${SSH_F[@]}" -o BatchMode=yes -o ConnectTimeout=8 "$node" "$@"
}

# The on-node result scan: every *.json (not *.prov.json) under the output roots, modified since the start.
read -r -d '' SCAN_PY <<'PY'
import json, os, sys
start = float(sys.argv[1]); roots = sys.argv[2:]
files, bad, unreadable = [], [], []
def walk(o, path, f):
    if isinstance(o, dict):
        for k, v in o.items():
            p = path + "/" + str(k)
            if k == "_error" and v not in (None, "", [], {}, False):
                bad.append("%s:%s=%s" % (f, p, str(v)[:100]))
            elif k == "error" and isinstance(v, str) and v.strip():
                bad.append("%s:%s=%s" % (f, p, v[:100]))
            walk(v, p, f)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, "%s[%d]" % (path, i), f)
def consider(f):
    if not f.endswith(".json") or f.endswith(".prov.json"):
        return
    try:
        if os.path.getmtime(f) < start - 1:
            return
    except OSError:
        return
    files.append(f)
    try:
        with open(f) as fh:
            walk(json.load(fh), "", f)
    except Exception as e:
        unreadable.append("%s: %s" % (f, str(e)[:100]))
for r in roots:
    if os.path.isfile(r):
        consider(r)
    elif os.path.isdir(r):
        for d, _, names in os.walk(r):
            for n in names:
                consider(os.path.join(d, n))
print(json.dumps({"files": len(files), "bad": bad[:20], "n_bad": len(bad), "unreadable": unreadable[:20]}))
PY

# pick_node <rev> <need_gb> -- `run auto`: of the pool nodes (POOL_NODES, else tools/pool_nodes.conf), the one that
# holds <rev> and has the most MemAvailable right now. A canary launches over plain ssh, outside the queue's memory
# accounting, so a busy node can refuse it at the line's own mem_ok; refuse here instead (rc 3, inconclusive) when no
# node has <need_gb> + 2 GB (mem_ok's margin) free. wspool answers exit 75 while the owner games and simply drops out.
pick_node() {
  local rev="$1" need="$2" n gb best="" best_gb=-1 seen=""
  for n in ${POOL_NODES:-$POOL_NODES_DEFAULT}; do
    if ! rssh "$n" "$(revision_marker_probe_cmd "derisk-pool/revisions/$rev")" >/dev/null 2>&1; then
      seen+=" $n:no-revision"; continue
    fi
    gb=$(rssh "$n" "awk '/^MemAvailable:/{print int(\$2/1048576)}' /proc/meminfo" 2>/dev/null) || gb=""
    if ! [[ $gb =~ ^[0-9]+$ ]]; then seen+=" $n:no-answer"; continue; fi
    seen+=" $n:${gb}G"
    if [ "$gb" -gt "$best_gb" ]; then best=$n; best_gb=$gb; fi
  done
  if [ -z "$best" ] || [ "$best_gb" -lt $((need + 2)) ]; then
    echo "⛔ auto: no pool node holds $rev with ${need}+2 GB free (${seen# })" >&2
    return 1
  fi
  echo "auto: picked $best (${seen# })" >&2
  printf '%s' "$best"
}

cmd_run() {
  local node="${1:-}" raw="${2:-}"; shift 2 2>/dev/null || usage
  [ -n "$node" ] && [ -n "$raw" ] || usage
  local exact=0 wait_s="${POOL_CANARY_WAIT_S:-14400}" poll="${POOL_CANARY_POLL_S:-30}"
  while [ $# -gt 0 ]; do
    case "$1" in
      --exact) exact=1 ;;
      --wait-s) wait_s="${2:?--wait-s needs seconds}"; shift ;;
      *) echo "unknown option: $1" >&2; usage ;;
    esac
    shift
  done
  local line mod rev sha kind
  line=$(bare_line "$raw")
  mod=$(line_module "$line"); rev=$(line_rev "$line"); kind=$(line_seed_kind "$line")
  [ -n "$mod" ] || { echo "⛔ the line names no '-m <module>': nothing to key a canary on" >&2; exit 2; }
  [ -n "$rev" ] || { echo "⛔ the line is not pinned (cd ~/derisk-pool/revisions/<sha>): nothing to key a canary on" >&2; exit 2; }
  sha=$(full_sha "$rev")
  valid_field "$line" || { echo "⛔ the line contains a control character" >&2; exit 2; }
  if [ "$node" = auto ]; then
    local need
    need=$(printf '%s' "$raw" | grep -oE 'mem_gb=[0-9]+' | head -1 | cut -d= -f2)
    [ -n "$need" ] || need=$(printf '%s' "$line" | grep -oE 'memcap\.sh [0-9]+' | head -1 | awk '{print $2}')
    node=$(pick_node "$rev" "${need:-2}") || exit 3
  fi
  if ! rssh "$node" "$(revision_marker_probe_cmd "derisk-pool/revisions/$rev")" >/dev/null 2>&1; then
    echo "⛔ $node is unreachable or not provisioned with $rev (bash tools/pool_provision.sh --isolated --revision $sha $node)" >&2
    exit 3
  fi

  local id tok pre post roots=() dirs=() p job est
  id="$(printf '%s' "${mod##*.}" | tr -c 'A-Za-z0-9_' '_')_${sha:0:9}_${node}_$(date +%Y%m%dT%H%M%S)"
  tok=$(printf '%s' "$line" | grep -oE -- "-m +${mod//./\\.}" | head -1)
  pre="${line%%"$tok"*}"; post="${line:${#pre}}"
  if [ "$exact" = 0 ]; then
    post=$(printf '%s' "$post" | sed -E "s#(^|[[:space:]=])research/findings/raw/#\\1research/findings/raw/_pool_canary/$id/#g")
    while IFS= read -r p; do
      [ -n "$p" ] || continue
      roots+=("$p")
      case "${p##*/}" in *.*) dirs+=("$(dirname "$p")") ;; *) dirs+=("$p") ;; esac
    done < <(printf '%s' "$post" | grep -oE "research/findings/raw/_pool_canary/$id/[^[:space:];&|)\"']+")
    [ ${#roots[@]} -gt 0 ] || { roots=("research/findings/raw/_pool_canary/$id"); dirs=("${roots[0]}"); }
    job="cd ~/derisk-pool/revisions/$rev && mkdir -p$(printf ' %q' "${dirs[@]}") && { $pre$post ; }"
  else
    while IFS= read -r p; do [ -n "$p" ] && roots+=("$p"); done \
      < <(printf '%s' "$post" | grep -oE "research/findings/raw/[^[:space:];&|)\"']+")
    job="$line"
  fi
  est=$(printf '%s' "$raw" | grep -oE 'mem_gb=[0-9]+' | head -1 | cut -d= -f2)
  [ -n "$est" ] || est=$(printf '%s' "$line" | grep -oE 'memcap\.sh [0-9]+' | head -1 | awk '{print $2}')
  est="${est:-2}"

  local wrapper job_b64 wrapper_b64 launch
  wrapper='job=$(printf "%s" "$JOB_B64" | base64 -d) || exit 125
mkdir -p ~/derisk-pool/canary
start=$(date +%s)
bash -c "$job" > ~/derisk-pool/canary/"$CANARY_ID".log 2>&1
rc=$?
printf "%s\t%s\t%s\n" "$start" "$(date +%s)" "$rc" > ~/derisk-pool/canary/"$CANARY_ID".status'
  job_b64=$(printf '%s' "$job" | base64 -w0); wrapper_b64=$(printf '%s' "$wrapper" | base64 -w0)
  launch=$(printf "cd ~/derisk-pool/sim && POOL_JOB_ID='canary-%s' POOL_JOB_MEM_GB='%s' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CANARY_ID='%s' JOB_B64='%s' WRAPPER_B64='%s' setsid bash -c 'printf \"%%s\" \"\$WRAPPER_B64\" | base64 -d | bash' </dev/null >/dev/null 2>&1 & exit 0" \
    "$id" "$est" "$id" "$job_b64" "$wrapper_b64")
  echo "canary $id: ($mod, ${sha:0:9}, $kind seed) on $node; log ~/derisk-pool/canary/$id.log"
  [ "$exact" = 1 ] && echo "  --exact: the line runs verbatim and writes its real output paths"
  rssh "$node" "$launch" >/dev/null 2>&1 || { echo "⛔ launch over ssh failed" >&2; exit 3; }

  local status="" waited=0
  while :; do
    status=$(rssh "$node" "cat ~/derisk-pool/canary/$id.status 2>/dev/null" 2>/dev/null)
    [ -n "$status" ] && break
    if awk -v w="$waited" -v m="$wait_s" 'BEGIN{exit !(w >= m)}'; then
      echo "INCONCLUSIVE: still running after ${wait_s}s (not recorded). Its status will land in ~/derisk-pool/canary/$id.status on $node."
      exit 3
    fi
    sleep "$poll"; waited=$(awk -v a="$waited" -v b="$poll" 'BEGIN{print a+b}')
  done
  local t0 t1 rc wall tb memref tail_txt scan nfiles nbad unread verdict ev
  IFS=$'\t' read -r t0 t1 rc <<<"$status"
  wall=$(( t1 - t0 ))
  tb=$(rssh "$node" "grep -c 'Traceback (most recent call last)' ~/derisk-pool/canary/$id.log 2>/dev/null" 2>/dev/null); tb="${tb:-?}"
  memref=$(rssh "$node" "grep -c 'mem_ok.*REFUSE' ~/derisk-pool/canary/$id.log 2>/dev/null" 2>/dev/null); memref="${memref:-0}"
  tail_txt=$(rssh "$node" "tail -5 ~/derisk-pool/canary/$id.log 2>/dev/null" 2>/dev/null | tr '\t\r' '  ' | tr '\n' ' ' | cut -c1-240)
  if [ "$rc" != 0 ] && [ "${memref:-0}" -gt 0 ] 2>/dev/null; then
    echo "INCONCLUSIVE: the line's own mem_ok refused to start it on $node (rc $rc); nothing recorded. Retry when the node has room."
    exit 3
  fi
  scan=$(rssh "$node" "cd ~/derisk-pool/revisions/$rev && printf '%s' $(printf '%s' "$SCAN_PY" | base64 -w0) | base64 -d > ~/derisk-pool/canary/$id.scan.py && python3 ~/derisk-pool/canary/$id.scan.py $t0$(printf ' %q' "${roots[@]}")" 2>/dev/null)
  nfiles=$(printf '%s' "$scan" | python3 -c 'import json,sys; print(json.load(sys.stdin)["files"])' 2>/dev/null || echo "?")
  nbad=$(printf '%s' "$scan" | python3 -c 'import json,sys; print(json.load(sys.stdin)["n_bad"])' 2>/dev/null || echo "?")
  unread=$(printf '%s' "$scan" | python3 -c 'import json,sys; print(len(json.load(sys.stdin)["unreadable"]))' 2>/dev/null || echo "?")
  verdict=FAIL
  if [ "$rc" = 0 ] && [ "$nbad" = 0 ] && [ "$unread" = 0 ] && [ "$nfiles" != "?" ] && [ "$nfiles" -ge 1 ]; then
    verdict=PASS
  fi
  ev="rc=$rc wall=${wall}s json=$nfiles errors=$nbad unreadable=$unread tracebacks=$tb id=$id"
  [ "$verdict" = PASS ] || ev="$ev tail: $tail_txt"
  [ "$exact" = 1 ] && ev="$ev (exact line)"
  append_record "$mod" "$sha" "$verdict" "$node" "$kind" "pool_canary.sh run (plain ssh)" "$ev"
  echo "$verdict ($mod, ${sha:0:9}, $kind seed) on $node: $ev"
  [ "$kind" = dev ] && [ "$verdict" = PASS ] && echo "  (a dev-seed PASS does not open the pair to gate-seed lines; canary a gate-seed line for that)"
  if [ "$verdict" != PASS ] && [ -n "$scan" ]; then
    printf '%s' "$scan" | python3 -c 'import json,sys; d=json.load(sys.stdin); [print("  error:", b) for b in d["bad"]]; [print("  unreadable:", u) for u in d["unreadable"]]' 2>/dev/null
  fi
  [ "$verdict" = PASS ]
}

case "${1:-}" in
  run) shift; cmd_run "$@"; exit $? ;;
  record)
    [ $# -eq 7 ] || usage
    mod="$2"; sha="$3"; verdict="$4"; node="$5"; kind="$6"; ev="$7"
    [[ $mod =~ ^[A-Za-z_][A-Za-z0-9_.]*$ ]] || { echo "⛔ module must look like research.runners.x" >&2; exit 2; }
    [[ $sha =~ ^[0-9a-f]{7,40}$ ]] || { echo "⛔ sha must be 7-40 hex characters" >&2; exit 2; }
    case "$verdict" in PASS|FAIL) ;; *) echo "⛔ verdict must be PASS or FAIL" >&2; exit 2 ;; esac
    [[ $node =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "⛔ node must be a bare hostname" >&2; exit 2; }
    case "$kind" in gate|dev) ;; *) echo "⛔ seed kind must be gate or dev (did the canary line pass a gate seed?)" >&2; exit 2 ;; esac
    valid_field "$ev" || { echo "⛔ evidence must not contain a tab, newline or other control character" >&2; exit 2; }
    [ "${#ev}" -ge 20 ] || { echo "⛔ evidence must say what ran and what it showed (>= 20 characters)" >&2; exit 2; }
    append_record "$mod" "$sha" "$verdict" "$node" "$kind" "recorded (run elsewhere)" "$ev"
    echo "recorded $verdict ($mod, ${sha:0:9}, $kind seed) on $node" ;;
  has)
    [ $# -eq 3 ] || usage
    if why=$(gate_ok "$2" "$3"); then exit 0; else echo "$why"; exit 1; fi ;;
  check-line)
    [ $# -eq 2 ] || usage
    line=$(bare_line "$2"); mod=$(line_module "$line"); rev=$(line_rev "$line")
    [ -n "$mod" ] || { echo "the line names no '-m <module>', so no pool canary can be matched to it"; exit 1; }
    [ -n "$rev" ] || { echo "the line is not pinned to a revision, so no pool canary can be matched to it"; exit 1; }
    if why=$(gate_ok "$mod" "$rev"); then exit 0; else echo "$why"; exit 1; fi ;;
  key)
    [ $# -eq 2 ] || usage
    line=$(bare_line "$2"); mod=$(line_module "$line"); rev=$(line_rev "$line")
    [ -n "$mod" ] && [ -n "$rev" ] || exit 1
    printf '%s\t%s\n' "$mod" "$(full_sha "$rev")" ;;
  list) [ -f "$CANARIES" ] && cat "$CANARIES" ;;
  *) usage ;;
esac
