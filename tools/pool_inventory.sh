#!/usr/bin/env bash
# pool_inventory.sh — make pool40/41/42 uniform for brain/ work (inventory + copy-only sync).
#   tools/pool_inventory.sh inventory   # read-only: probe nodes, write tools/engine/nodes.json (default)
#   tools/pool_inventory.sh sync        # copy-only: KR seed data (relay via workstation) + tools/memcap.sh, mem_ok.sh; then inventory
# Never deletes, never overwrites an existing file that differs (rsync --ignore-existing); differing
# files are REPORTED as CONFLICT. Never touches running processes. Skips a seed copy if target free < MIN_FREE_GB.
set -uo pipefail
MODE="${1:-inventory}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NODES="${POOL_NODES:-pool40 pool41 pool42}"
SEEDS="${KR_SEEDS:-319 320 321}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
RELAY="${KR_RELAY:-$HERE/_scratch/kr_relay}"
OUT="$HERE/tools/engine/nodes.json"
BL=/home/node/brain-lane
KR=$BL/route_repo/_scratch/route/kr
export SSH_AUTH_SOCK=
SSHO="ssh -o IdentitiesOnly=yes -o ConnectTimeout=12"
rs() { $SSHO "$1" "${@:2}"; }

probe() {  # one node -> one JSON object on stdout
  local n="$1"
  rs "$n" "bash -s" <<'REMOTE'
BL=/home/node/brain-lane; KR=$BL/route_repo/_scratch/route/kr
ram=$(free -g | awk '/^Mem:/{print $2}'); ramm=$(free -m | awk '/^Mem:/{print $2}')
free_gb=$(df -BG --output=avail /home/node | tail -1 | tr -dc 0-9)
py=""; pp=""
for cand in "$BL/venv/bin/python|" "$BL/venv/bin/python|$BL/pylib" "/home/node/derisk-pool/sim/.venv/bin/python|$BL/pylib"; do
  p=${cand%%|*}; e=${cand#*|}
  [ -x "$p" ] || continue
  if PYTHONPATH="$e" "$p" -c 'import torch, rwkv' >/dev/null 2>&1; then py=$p; pp=$e; break; fi
done
models=$(ls $BL/models 2>/dev/null | grep '\.pth$' | tr '\n' ' ')
seeds=""
for s in $(ls $KR 2>/dev/null | sed -n 's/^lifeKR_s\([0-9]*\)\.json\.json$/\1/p'); do
  [ -d $KR/snaps_s$s ] && [ -d $KR/trunk_s$s ] && seeds="$seeds $s"
done
keep=$(ls $KR 2>/dev/null | grep -i keep | tr '\n' ' ')
printf 'RAM_GB=%s\nRAM_MB=%s\nFREE_GB=%s\nPY=%s\nPP=%s\nMODELS=%s\nSEEDS=%s\nKEEP=%s\nMEMCAP=%s\nMEMOK=%s\n' \
  "$ram" "$ramm" "$free_gb" "$py" "$pp" "$models" "$seeds" "$keep" \
  "$(md5sum $BL/route_repo/tools/memcap.sh 2>/dev/null | cut -c1-32)" "$(md5sum $BL/route_repo/tools/mem_ok.sh 2>/dev/null | cut -c1-32)"
REMOTE
}

inventory() {
  local tmp; tmp="$(mktemp)"
  for n in $NODES; do probe "$n" | sed "s/^/$n /" >> "$tmp" || echo "WARN: $n unreachable" >&2; done
  mkdir -p "$(dirname "$OUT")"
  python3 - "$tmp" "$OUT" <<'PY'
import sys, json, collections
d = collections.OrderedDict()
for l in open(sys.argv[1]):
    n, kv = l.rstrip("\n").split(" ", 1)
    k, _, v = kv.partition("=")
    d.setdefault(n, {})[k] = v
out = []
for n, m in d.items():
    ram = int(m.get("RAM_GB") or 0)
    out.append({"name": n, "ram_gb": ram, "ram_mb": int(m.get("RAM_MB") or 0),
                "disk_free_gb": int(m.get("FREE_GB") or 0),
                "python": m.get("PY", ""), "pythonpath": m.get("PP", ""),
                "models": m.get("MODELS", "").split(),
                "kr_seeds": [int(s) for s in m.get("SEEDS", "").split()],
                "keep_files": m.get("KEEP", "").split(),
                "memcap_md5": m.get("MEMCAP", ""), "mem_ok_md5": m.get("MEMOK", ""),
                "max_1p5b_cpu": ram >= 22})
json.dump(out, open(sys.argv[2], "w"), indent=2); open(sys.argv[2], "a").write("\n")
for o in out:
    print(f'{o["name"]}: ram={o["ram_gb"]}G free={o["disk_free_gb"]}G py={o["python"] or "NONE"} seeds={o["kr_seeds"]} models={len(o["models"])} 1p5b={o["max_1p5b_cpu"]}')
PY
  rm -f "$tmp"
}

have_seed() { rs "$1" "[ -f $KR/lifeKR_s$2.json.json ] && [ -d $KR/snaps_s$2 ] && [ -d $KR/trunk_s$2 ]" 2>/dev/null; }
free_gb() { rs "$1" "df -BG --output=avail /home/node | tail -1 | tr -dc 0-9"; }
RSY="rsync -a --ignore-existing -e 'ssh -o IdentitiesOnly=yes -o ConnectTimeout=12'"
# differing-size files that --ignore-existing skipped (report only)
conflicts() { # $1 src-spec $2 dst-spec
  eval rsync -rn --size-only --itemize-changes --ignore-missing-args -e "'ssh -o IdentitiesOnly=yes'" "$1" "$2" 2>/dev/null | grep -E '^>f.*s' | head -20; }

sync_seeds() {
  mkdir -p "$RELAY"
  for s in $SEEDS; do
    local src="" tgts=""
    for n in $NODES; do
      if have_seed "$n" "$s"; then [ -z "$src" ] && src="$n"; else tgts="$tgts $n"; fi
    done
    [ -z "$tgts" ] && { echo "seed $s: all nodes have it"; continue; }
    [ -z "$src" ] && { echo "seed $s: NO node has a complete copy; skip"; continue; }
    # disk check first (target free, and workstation relay free)
    local ok=""
    for t in $tgts; do
      f=$(free_gb "$t"); if [ "${f:-0}" -lt "$MIN_FREE_GB" ]; then echo "seed $s -> $t: SKIP free ${f}G < ${MIN_FREE_GB}G"; else ok="$ok $t"; fi
    done
    [ -z "$ok" ] && continue
    local wf; wf=$(df -BG --output=avail "$RELAY" | tail -1 | tr -dc 0-9)
    [ "$wf" -lt "$MIN_FREE_GB" ] && { echo "seed $s: SKIP workstation relay free ${wf}G"; continue; }
    echo "seed $s: $src -> workstation relay -> $ok"
    mkdir -p "$RELAY/s$s"
    for item in lifeKR_s$s.json.json snaps_s$s trunk_s$s; do
      eval $RSY "$src:$KR/$item" "$RELAY/s$s/" || { echo "pull $item from $src FAILED"; continue 2; }
    done
    for t in $ok; do
      rs "$t" "mkdir -p $KR"
      for item in lifeKR_s$s.json.json snaps_s$s trunk_s$s; do
        eval $RSY "$RELAY/s$s/$item" "$t:$KR/" || echo "push $item to $t FAILED"
        c=$(conflicts "$RELAY/s$s/$item" "$t:$KR/"); [ -n "$c" ] && echo "CONFLICT $t $item (existing differs, not overwritten):" && echo "$c"
      done
      have_seed "$t" "$s" && echo "  $t: seed $s complete" || echo "  $t: seed $s INCOMPLETE"
    done
    [ -z "${KEEP_RELAY:-}" ] && rm -rf "$RELAY/s$s"
  done
}

sync_tools() {
  for f in memcap.sh mem_ok.sh; do
    for n in $NODES; do
      local lm rm_; lm=$(md5sum "$HERE/tools/$f" | cut -c1-32)
      rm_=$(rs "$n" "md5sum $BL/route_repo/tools/$f 2>/dev/null | cut -c1-32")
      if [ -z "$rm_" ]; then eval $RSY "$HERE/tools/$f" "$n:$BL/route_repo/tools/" && echo "$n: installed $f"
      elif [ "$rm_" != "$lm" ]; then echo "CONFLICT $n tools/$f differs from workstation (not overwritten)"
      else echo "$n: $f identical"; fi
    done
  done
}

case "$MODE" in
  inventory) inventory ;;
  sync) echo "--- before"; inventory; echo "--- sync"; sync_tools; sync_seeds; echo "--- after"; inventory ;;
  *) echo "usage: $0 [inventory|sync]"; exit 2 ;;
esac
