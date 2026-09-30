#!/usr/bin/env bash
# Integration wave 1: copy landed results from the primary checkout (where pool_sync lands them) into this worktree,
# so they can be committed on the packet branch. Copies research/findings/raw/_iw1/identity/<PIN7> and every named
# shard tag. Refuses if a named tag is missing (nothing landed yet).
#   bash research/packets/integration-wave1/copy_results.sh <PIN7> <tag> [<tag> ...]
set -euo pipefail
P=/home/dant123/Projects/sim
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PIN7="${1:?pin7}"; shift
d=research/findings/raw/_iw1/identity/$PIN7
if [ -d "$P/$d" ]; then mkdir -p "$W/$d" && cp -r "$P/$d/." "$W/$d/"; echo "copied $d"; fi
for T in "$@"; do
  d=research/findings/raw/_load_bearing/_shards/$T
  [ -d "$P/$d" ] || { echo "copy_results: $P/$d is missing" >&2; exit 1; }
  mkdir -p "$W/$d" && cp -r "$P/$d/." "$W/$d/"; echo "copied $d"
done
