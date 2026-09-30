#!/usr/bin/env bash
# Install requirements-pool.txt into a pool revision's .venv and verify it -- the ONE venv code path.
# tools/pool_provision.sh calls this for every node it provisions; call it directly to bring an ALREADY
# provisioned revision's venv up to the current pinned set without re-syncing its (read-only) source.
#
# Usage:
#   bash tools/pool_venv_deps.sh --revision <sha> [--requirements <file>] [<node> ...]
#       (remote root ~/derisk-pool/revisions/<full sha>; the revision must already be provisioned there)
#   bash tools/pool_venv_deps.sh --remote-root <path under ~> --requirements <file> <node> ...
#       (what pool_provision.sh uses, for both --isolated and the compatibility path)
# Nodes default to tools/pool_nodes.conf.
#
# Requirements file: --requirements wins; else the revision's own requirements-pool.txt (git show <sha>:...);
# else this checkout's requirements-pool.txt (revisions that predate the file get the current pinned set).
# The chosen file is copied to <root>/.venv/pool-requirements.txt (inside the venv, so the source manifest
# ignores it) and stays there as the record of what was installed.
#
# pip only ADDS missing packages or moves a package to its pin; it never touches the revision's source files.
# Failure (pip error, a pin not met, a package or the brain-chat harness failing to import) exits non-zero.
# No sudo, no system packages: python3 -m venv must already work on the node (pool_provision.sh ensures it).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
# shellcheck source=tools/pool_nodes.sh
source "$ROOT/tools/pool_nodes.sh"
REMOTE_ROOT=""
REVISION=""
REQUIREMENTS=""
usage() { sed -n '2,19p' "$0" >&2; exit 2; }
while (( $# )); do
  case "$1" in
    --revision) [[ -n "${2:-}" ]] || usage; REVISION=$2; shift 2 ;;
    --remote-root) [[ -n "${2:-}" ]] || usage; REMOTE_ROOT=$2; shift 2 ;;
    --requirements) [[ -n "${2:-}" ]] || usage; REQUIREMENTS=$2; shift 2 ;;
    -h|--help) usage ;;
    --) shift; break ;;
    -*) echo "unknown option: $1" >&2; usage ;;
    *) break ;;
  esac
done
if [[ -n "$REVISION" && -n "$REMOTE_ROOT" ]] || [[ -z "$REVISION" && -z "$REMOTE_ROOT" ]]; then
  echo "pass exactly one of --revision / --remote-root" >&2; exit 2
fi
TMP_REQ=""
trap 'if [ -n "$TMP_REQ" ]; then rm -f "$TMP_REQ"; fi' EXIT
if [[ -n "$REVISION" ]]; then
  SHA=$(git rev-parse --verify "${REVISION}^{commit}" 2>/dev/null) || { echo "invalid revision: $REVISION" >&2; exit 2; }
  REMOTE_ROOT="derisk-pool/revisions/$SHA"
  if [[ -z "$REQUIREMENTS" ]] && git cat-file -e "$SHA:requirements-pool.txt" 2>/dev/null; then
    TMP_REQ=$(mktemp)
    git show "$SHA:requirements-pool.txt" > "$TMP_REQ"
    REQUIREMENTS=$TMP_REQ
  fi
fi
REQUIREMENTS=${REQUIREMENTS:-$ROOT/requirements-pool.txt}
[[ -f "$REQUIREMENTS" ]] || { echo "requirements file not found: $REQUIREMENTS" >&2; exit 2; }
case "$REMOTE_ROOT" in /*|*..*|"") echo "remote root must be a relative path under ~ without '..': $REMOTE_ROOT" >&2; exit 2 ;; esac
CHECKER="$ROOT/tools/pool/provisioning/pool_venv_check.py"
NODES=("${@:-$POOL_NODES_DEFAULT}"); NODES=(${NODES[@]})
POOL_SSH_CONFIG="${POOL_SSH_CONFIG:-$ROOT/research/queue/.pool_ssh_config}"
SSH_CMD="ssh"; [ -f "$POOL_SSH_CONFIG" ] && SSH_CMD="ssh -F $POOL_SSH_CONFIG"
FAILED=()
for h in "${NODES[@]}"; do
  echo "  venv deps $h:~/$REMOTE_ROOT ($(basename "$REQUIREMENTS"))"
  if ! $SSH_CMD -n -o ConnectTimeout=10 "$h" "test -d ~/$REMOTE_ROOT"; then
    echo "  ⛔ $h: ~/$REMOTE_ROOT does not exist (provision the revision first)" >&2
    FAILED+=("$h:no-root"); continue
  fi
  # venv: reuse a working one, recreate a pip-less leftover (same rule pool_provision.sh always used).
  if ! $SSH_CMD -n "$h" "cd ~/$REMOTE_ROOT && { test -x .venv/bin/python && .venv/bin/python -m pip --version >/dev/null 2>&1 || \
      { rm -rf .venv && python3 -m venv .venv; }; }"; then
    echo "  ⛔ $h: could not create .venv" >&2
    FAILED+=("$h:venv"); continue
  fi
  if ! $SSH_CMD "$h" "cat > ~/$REMOTE_ROOT/.venv/pool-requirements.txt" < "$REQUIREMENTS"; then
    FAILED+=("$h:copy-requirements"); continue
  fi
  $SSH_CMD -n "$h" "cd ~/$REMOTE_ROOT && .venv/bin/python -m pip -q install --upgrade pip >/dev/null 2>&1" || \
    echo "  (warning: pip self-upgrade failed on $h; continuing with the venv's pip)" >&2
  if ! $SSH_CMD -n "$h" "cd ~/$REMOTE_ROOT && .venv/bin/python -m pip install -q --disable-pip-version-check \
      -r .venv/pool-requirements.txt"; then
    echo "  ⛔ $h: pip install -r pool-requirements.txt FAILED" >&2
    FAILED+=("$h:pip"); continue
  fi
  # The checker comes from THIS checkout (piped over stdin), so a revision that predates it is checked identically.
  if ! REPORT=$($SSH_CMD "$h" "cd ~/$REMOTE_ROOT && SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= .venv/bin/python - \
      .venv/pool-requirements.txt --write-env .pool_environment.json" < "$CHECKER"); then
    echo "  ⛔ $h: venv check FAILED: $REPORT" >&2
    FAILED+=("$h:check"); continue
  fi
  echo "  $h ok: $REPORT"
done
if ((${#FAILED[@]})); then
  printf 'POOL VENV DEPS FAILED: %s\n' "${FAILED[*]}" >&2
  exit 1
fi
