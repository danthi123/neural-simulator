#!/usr/bin/env bash
# export_local_revision.sh <commit> -- the WORKSTATION's own isolated revision of <commit>, for the B2d mouth arm.
#
# The mouth arm runs on the local 3090 through tools/gpu_queue.sh, but B2d's validity rules (A1.2 rules 1-3) want
# every sidecar to read `git_sha` = the pin IN FULL, `source_kind` git_archive and a verified source manifest: the
# same record a pool node's `~/derisk-pool/revisions/<sha>` produces. This builds that tree on this machine, the way
# tools/pool_provision.sh builds it on a node (same `git archive` path list, same ancestry attestation, same manifest,
# same .source_revision), at ~/derisk-pool/revisions/<sha>. `.venv` and `data` are symlinks to the primary checkout's
# (the manifest ignores both; data/ is runtime, .venv* is never source).
#
# Idempotent: an existing tree whose manifest verifies is kept; one that does not verify is refused (never patched).
# Refuses a commit that does not contain origin/main (the provisioner's stale-source guard), unless
# B2D_EXPORT_ALLOW_STALE=1 (dev use only: the pre-change identity reference at main's own HEAD needs no override).
# Writes nothing outside ~/derisk-pool/revisions/ (the stage directory is created next to the destination, on disk).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PRIMARY=${B2D_PRIMARY:-/home/dant123/Projects/sim}
REVS=${B2D_LOCAL_REVISIONS:-$HOME/derisk-pool/revisions}
[ -n "${1:-}" ] || { echo "usage: $0 <commit>" >&2; exit 2; }
SHA=$(git rev-parse --verify "${1}^{commit}") || { echo "invalid commit: $1" >&2; exit 2; }
if timeout 15 git fetch -q origin main 2>/dev/null; then
  if ! git merge-base --is-ancestor origin/main "$SHA"; then
    echo "REFUSED: $SHA does not contain origin/main ($(git rev-parse origin/main))" >&2
    [ "${B2D_EXPORT_ALLOW_STALE:-0}" = "1" ] || exit 2
  fi
else
  echo "REFUSED: could not fetch origin/main to check ancestry" >&2; exit 2
fi
DEST="$REVS/$SHA"
verify() {
  local want
  want=$(sed -n 's/^source_manifest_sha256=//p' "$DEST/.source_revision")
  "$PRIMARY/.venv/bin/python" "$DEST/tools/pool/provisioning/source_manifest.py" verify --root "$DEST" \
    --manifest "$DEST/.source_manifest.sha256" --expected-sha256 "$want" >/dev/null
}
if [ -e "$DEST" ]; then
  if [ -f "$DEST/.source_revision" ] && verify; then echo "exists, manifest verified: $DEST"; exit 0; fi
  echo "REFUSED: $DEST exists but its manifest does not verify; move it aside by hand" >&2; exit 1
fi
mkdir -p "$REVS"
STAGE=$(mktemp -d "$REVS/.stage.${SHA:0:12}.XXXXXX")
trap 'rm -rf "$STAGE"' EXIT
# The path list is tools/pool_provision.sh's (tests/test_b2d_mouth_packet.py pins the two copies equal).
git archive "$SHA" sim webapp research/__init__.py research/runners research/specs research/fixtures \
  research/findings ':(exclude)research/findings/raw' experiment tools tests \
  docs CLAUDE.md GAP_CLOSURE_MISSION.md README.md ROADMAP.md requirements.txt requirements-dev.txt \
  | tar -x -C "$STAGE"
python3 "$STAGE/tools/pool/provisioning/ancestry_attestation.py" create \
  --repo . --revision "$SHA" --output "$STAGE/.source_ancestry.json" >/dev/null
ANCESTRY_SHA=$(sha256sum "$STAGE/.source_ancestry.json" | awk '{print $1}')
python3 "$STAGE/tools/pool/provisioning/source_manifest.py" create \
  --root "$STAGE" --output "$STAGE/.source_manifest.sha256" >/dev/null
MANIFEST_SHA=$(sha256sum "$STAGE/.source_manifest.sha256" | awk '{print $1}')
printf 'git_sha=%s\nsource_kind=git_archive\nsource_manifest_sha256=%s\nsource_ancestry_sha256=%s\nexcluded_worktree_paths=0\ncreated_utc=%s\nexported_by=research/packets/b2d-mouth-renderer/export_local_revision.sh\n' \
  "$SHA" "$MANIFEST_SHA" "$ANCESTRY_SHA" "$(date -u +%FT%TZ)" > "$STAGE/.source_revision"
ln -s "$PRIMARY/.venv" "$STAGE/.venv"
ln -s "$PRIMARY/data" "$STAGE/data"
mv "$STAGE" "$DEST"
trap - EXIT
verify || { echo "EXPORT FAIL: manifest does not verify at $DEST" >&2; exit 1; }
[ -f "$DEST/data/corpus/tinystories.txt" ] || { echo "EXPORT FAIL: no data/corpus/tinystories.txt via $DEST/data" >&2; exit 1; }
echo "exported $SHA -> $DEST (manifest $MANIFEST_SHA)"
