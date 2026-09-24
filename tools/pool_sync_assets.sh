#!/usr/bin/env bash
# pool_sync_assets.sh — sync the MINIMAL brain-build data lake (sim-data/knowledge_bundles/*) to remote
# compute nodes, so a remote brain build resolves the SAME cortical long-term-memory (LTM) bundle a local
# build does, instead of silently degrading to the byte-identical-but-hollow no-LTM path.
#
# WHY (2026-09-23 investigation; see research/FAILURE_LOG.md same date). None of tools/pool_provision.sh,
# tools/aws_provision.sh, tools/aws_cpu_provision.sh ship ANYTHING from `sim-data/` -- it lives OUTSIDE the
# repo entirely (a sibling directory, ~/Projects/sim-data on this machine), so `_default_ltm_bundle_dir()`
# (webapp/server.py) always resolves to None on a freshly-provisioned remote box and the brain silently ships
# with 5 hardcoded facts and no LTM (`source` stays "tiny-demo", never "tiny-demo +LTM"). This is NOT the
# 25GB bridges/ + 11GB data/ haul the existing provisioners already (correctly) exclude -- it is a separate,
# small (~105MB) directory neither script knows exists. This script ships ONLY that.
#
# Usage: bash tools/pool_sync_assets.sh [pool40 pool41 pool42]
#   SIM_DATA_ROOT=<dir>    override the LOCAL source root (default: ~/Projects/sim-data, matching
#                          webapp/server.py::_default_ltm_bundle_dir()'s own 3rd-candidate fallback)
#   REMOTE_SIM_DATA=<dir>  override the REMOTE destination's path component under $HOME (default:
#                          Projects/sim-data, the SAME fallback candidate -- works regardless of where the
#                          code checkout itself lives on that node: derisk-pool/sim for the mini-PC pool vs
#                          ~/sim for AWS, since _default_ltm_bundle_dir() checks $HOME/Projects/sim-data
#                          independently of the checkout location)
set -euo pipefail
cd "$(dirname "$0")/.."

SRC_ROOT="${SIM_DATA_ROOT:-$HOME/Projects/sim-data}"
SRC="$SRC_ROOT/knowledge_bundles"
if [ ! -d "$SRC" ]; then
  echo "⛔ no local knowledge_bundles dir at $SRC -- nothing to sync (set SIM_DATA_ROOT if it lives elsewhere)" >&2
  exit 1
fi

NODES=("${@:-pool40 pool41 pool42}"); NODES=(${NODES[@]})
REMOTE_SUFFIX="${REMOTE_SIM_DATA:-Projects/sim-data}/knowledge_bundles"
# AWS-AS-EXTRA-POOL-NODE (2026-09-23) -- same repo-local, gitignored ssh config pool_provision.sh (which calls
# this script per node) now honours; see its header comment for the full rationale. ABSENT by default, so
# every ssh/rsync call below is unchanged for the existing pool40/41/42 nodes.
ROOT="$(pwd)"
POOL_SSH_CONFIG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
SSH_CMD="ssh"; [ -f "$POOL_SSH_CONFIG" ] && SSH_CMD="ssh -F $POOL_SSH_CONFIG"

# Ship ONLY the two curated bundles _default_ltm_bundle_dir() actually looks for -- wikidata_100k (the
# 2026-09-02 default, ~88M) and wikidata_core_15k (the robustness fallback, ~17M) -- NOT the whole sim-data
# tree (which may hold other, unrelated large scratch data). Same minimal-haul discipline as
# pool_provision.sh's own code archive (git archive an explicit path list, not the whole repo).
BUNDLES=(wikidata_100k wikidata_core_15k)

FAILED_NODES=()
for h in "${NODES[@]}"; do
  echo "=== syncing knowledge bundles to $h:~/${REMOTE_SUFFIX} ==="
  $SSH_CMD -o ConnectTimeout=10 "$h" "mkdir -p ~/${REMOTE_SUFFIX}" || {
    echo "  SSH FAIL $h"
    FAILED_NODES+=("$h:ssh")
    continue
  }
  ok=1
  shipped=0
  for b in "${BUNDLES[@]}"; do
    if [ ! -d "$SRC/$b" ]; then
      echo "  (skip: no local bundle $b under $SRC)"
      continue
    fi
    rsync -az -e "$SSH_CMD" --delete "$SRC/$b/" "$h:~/${REMOTE_SUFFIX}/$b/" || { ok=0; break; }
    shipped=$((shipped + 1))
  done
  if [ "$ok" != 1 ]; then
    echo "  RSYNC FAIL $h" >&2
    FAILED_NODES+=("$h:rsync")
    continue
  fi
  if [ "$shipped" = 0 ]; then
    echo "  ⛔ NOTHING SHIPPED to $h (neither bundle exists locally under $SRC)" >&2
    FAILED_NODES+=("$h:no-bundles")
    continue
  fi
  echo "  done $h ($shipped bundle(s))"
done

if ((${#FAILED_NODES[@]})); then
  printf 'ASSET SYNC FAILED: %s\n' "${FAILED_NODES[*]}" >&2
  exit 1
fi
echo "ALL ASSET SYNC DONE"
