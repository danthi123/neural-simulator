#!/usr/bin/env bash
# ship_runtime_inputs.sh -- top up an ALREADY-provisioned pool revision with the run-time inputs under
# research/findings/raw/ that tools/pool/runtime_raw_inputs.txt names (2026-10-04, cloud Claude session 6).
#
# WHY. The server's provisioner answers `already` for a revision whose nodes carry .provisioned_ok, so a revision
# provisioned before the runtime-input fix never receives them (the Stage-1 gate pin c238ff325: the DR-2 valence map
# was missing on pool41/42 and its C4 arms silently read the raw Warriner norm). This copies each listed path FROM
# THAT REVISION (git archive, never the working tree) into ~/derisk-pool/revisions/<sha>/ on each node, then checks
# every shipped file's sha256 on the node against git. It never touches code or the source manifest.
#
#   bash tools/pool/ship_runtime_inputs.sh <full 40-hex sha> [node ...]     # default nodes: pool41 pool42
#   bash tools/pool/ship_runtime_inputs.sh --dry-run <sha>                  # list what would ship, touch nothing
# Exit 0 = every file verified on every node; 1 = a copy or a check failed; 2 = usage.
set -uo pipefail
DRY=0
[ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }
SHA=${1:-}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "usage: ship_runtime_inputs.sh [--dry-run] <full 40-hex sha> [node ...]" >&2; exit 2; }
shift
NODES=("$@"); [ ${#NODES[@]} -eq 0 ] && NODES=(pool41 pool42)
ROOT=$(git rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2
git cat-file -e "$SHA^{commit}" 2>/dev/null || { echo "unknown commit $SHA" >&2; exit 2; }
LIST=$(git show "$SHA:tools/pool/runtime_raw_inputs.txt" 2>/dev/null || cat tools/pool/runtime_raw_inputs.txt)
mapfile -t PATHS < <(printf '%s\n' "$LIST" | sed 's/#.*//; s/[[:space:]]*$//' | grep -v '^$')
mapfile -t FILES < <(git ls-tree -r --name-only "$SHA" -- "${PATHS[@]}")
[ ${#FILES[@]} -gt 0 ] || { echo "none of the runtime inputs is in $SHA" >&2; exit 1; }
echo "revision $SHA: ${#FILES[@]} file(s) from ${#PATHS[@]} listed path(s) -> ${NODES[*]}"
if [ "$DRY" = 1 ]; then printf '  %s\n' "${FILES[@]}"; exit 0; fi
DEST="derisk-pool/revisions/$SHA"
rc=0
for n in "${NODES[@]}"; do
  if ! ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$n" "test -d ~/$DEST"; then
    echo "  ⛔ $n: ~/$DEST does not exist (provision the revision first)" >&2; rc=1; continue
  fi
  if ! git archive "$SHA" -- "${FILES[@]}" | ssh -o BatchMode=yes -o ConnectTimeout=10 "$n" "tar -x -C ~/$DEST"; then
    echo "  ⛔ $n: copy failed" >&2; rc=1; continue
  fi
  bad=0
  for f in "${FILES[@]}"; do
    want=$(git show "$SHA:$f" | sha256sum | cut -d' ' -f1)
    have=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$n" "sha256sum ~/$DEST/$f 2>/dev/null" | cut -d' ' -f1)
    [ "$want" = "$have" ] || { echo "  ⛔ $n: $f sha256 mismatch (want $want, have ${have:-missing})" >&2; bad=1; }
  done
  [ "$bad" = 0 ] && echo "  ✔ $n: ${#FILES[@]} file(s) verified" || rc=1
done
exit $rc
