#!/usr/bin/env bash
# prepare.sh -- build the model bake-off's per-task starting clones (docs: tools/model_bakeoff/README.md).
#
# Runs INSIDE the Hermes container as the agent (uid 99), the one place that can read the full history:
#   docker exec -u 99 -e HOME=/opt/data/home hermes bash /opt/data/bakeoff/tools/prepare.sh [task-id ...]
# It never writes to the workspace. Layout under $BK (= /opt/data/bakeoff here, /bk in the bake-off container):
#   tools/   this directory, copied from the workspace at $REV
#   home/    the bake-off's own HERMES_HOME: production config.yaml minus terminal.cwd
#   bases/<task>/            a clone holding ONLY history up to the task's base commit, setup applied, uncommitted
#   bases/<task>.protected/  copies of the files setup created or changed, restored before grading when the task
#                            protects them (category implement), so editing a given test cannot make it pass
# The temporary bare mirror ($BK/mirror.git, every ref) is deleted at the end: nothing that can see past a task's
# base commit is ever mounted into the bake-off container.
set -euo pipefail
WS=${BAKEOFF_WS:-/home/dant123/Projects/neural-simulator}
BK=${BAKEOFF_DIR:-/opt/data/bakeoff}
REV=${BAKEOFF_REV:-HEAD}
PY=${BAKEOFF_PY:-/opt/hermes/.venv/bin/python}

mkdir -p "$BK"/{tools,home,bases,runs}
rm -rf "$BK/tools.new" && mkdir "$BK/tools.new"
git -C "$WS" archive "$REV" tools/model_bakeoff | tar -x -C "$BK/tools.new" --strip-components=2
rm -rf "$BK/tools" && mv "$BK/tools.new" "$BK/tools"

# The bake-off's HERMES_HOME: production settings (approvals, deny list, command_allowlist, provider_routing with
# data_collection deny, the local model as default), but no terminal.cwd: each run's terminal starts in its clone.
"$PY" - "$BK/home/config.yaml" <<'EOF'
import sys, yaml
cfg = yaml.safe_load(open("/opt/data/config.yaml"))
term = cfg.setdefault("terminal", {})
term.pop("cwd", None)
term["backend"] = "local"
cfg.pop("dashboard", None)
pr = cfg.setdefault("provider_routing", {})
assert pr.get("data_collection") == "deny", "production config lost provider_routing.data_collection: deny"
yaml.safe_dump(cfg, open(sys.argv[1], "w"), sort_keys=False)
print("bake-off config written:", sys.argv[1])
EOF

SUITE="$BK/tools/${BAKEOFF_SUITE:-suite.json}"
[ -f "$SUITE" ] || { echo "no suite at $SUITE"; exit 1; }
mapfile -t IDS < <("$PY" -c 'import json,sys; [print(t["id"]) for t in json.load(open(sys.argv[1]))["tasks"]]' "$SUITE")
[ $# -gt 0 ] && IDS=("$@")

MIRROR="$BK/mirror.git"
rm -rf "$MIRROR"
git clone -q --mirror --no-local "$WS" "$MIRROR"   # --mirror: every ref, incl. the imported origin/research/*
trap 'rm -rf "$MIRROR"' EXIT

for id in "${IDS[@]}"; do
  [[ "$id" =~ ^[a-z0-9][a-z0-9-]{2,80}$ ]] || { echo "refusing task id '$id'"; exit 1; }
  base=$("$PY" -c 'import json,sys; print(next(t["base_commit"] for t in json.load(open(sys.argv[1]))["tasks"] if t["id"]==sys.argv[2]))' "$SUITE" "$id")
  sha=$(git -C "$MIRROR" rev-parse --verify "$base^{commit}")
  git -C "$MIRROR" update-ref "refs/heads/bk-$id" "$sha"
  dst="$BK/bases/$id"
  rm -rf "$dst" "$dst.protected"
  # --single-branch + --no-local: the clone receives ONLY objects reachable from the base commit.
  git clone -q --single-branch --branch "bk-$id" --no-local "$MIRROR" "$dst"
  git -C "$dst" remote remove origin
  git -C "$dst" branch -q -m "bk-$id" main
  echo "== $id at ${sha:0:9}"
  # Setup: the suite's own commands, run in the clone with $SRC = the full mirror (read-only use).
  "$PY" -c 'import json,sys; [print(c) for c in next(t for t in json.load(open(sys.argv[1]))["tasks"] if t["id"]==sys.argv[2]).get("setup", [])]' "$SUITE" "$id" > "$BK/.setup-$id.sh"
  ( cd "$dst" && SRC="$MIRROR" bash -euo pipefail "$BK/.setup-$id.sh" ) || { echo "setup FAILED for $id"; rm -f "$BK/.setup-$id.sh"; exit 1; }
  rm -f "$BK/.setup-$id.sh"
  # Snapshot what setup created or changed (the grader restores it for protected tasks).
  mkdir -p "$dst.protected"
  : > "$dst.protected/.deleted"
  git -C "$dst" status --porcelain=v1 -uall --no-renames | while IFS= read -r line; do
    p=${line:3}
    if [ -f "$dst/$p" ]; then mkdir -p "$dst.protected/$(dirname "$p")" && cp -p "$dst/$p" "$dst.protected/$p"
    else printf '%s\n' "$p" >> "$dst.protected/.deleted"; fi
  done
  git -C "$dst" rev-parse HEAD > "$dst.protected/.base_sha"
  # Leak check: no ref and no object newer than base can exist in the clone.
  n=$(git -C "$dst" for-each-ref | wc -l)
  [ "$n" = 1 ] || { echo "LEAK: $id has $n refs"; exit 1; }
  echo "   ready: $(git -C "$dst" log --oneline -1 | cut -c1-80); setup touched $(find "$dst.protected" -type f ! -name .base_sha ! -name .deleted | wc -l) file(s)"
done
echo "prepared ${#IDS[@]} task(s) in $BK/bases; mirror removed"
