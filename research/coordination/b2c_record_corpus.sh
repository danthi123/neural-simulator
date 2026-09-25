#!/usr/bin/env bash
# b2c_record_corpus.sh -- write Battery B2c's corpus-hash record (research/coordination/b2c0925_corpus_sha256.tsv),
# one row for the primary checkout and one per pool node provisioned with F2. Same columns as b2b0924's record.
#
# Prereg: research/findings/2026-09-25-production-default-battery-B2c-paired-flip-PREREGISTRATION.md ("Corpus"). F2
# pins the code, not the data: data/corpus/tinystories.txt is untracked and rsynced by pool_provision.sh. A cell from
# a node whose hash differs from the primary checkout's is class E (did not run F2 as registered). Run from the
# PRIMARY checkout after provisioning, commit the file, and only then start the waves (b2c_queue_next_wave.sh refuses
# without it). Read-only on the nodes (ssh -n: sha256sum, hostname, a marker test).
#
# Usage:  bash research/coordination/b2c_record_corpus.sh [node ...]     (default nodes: pool1 pool2)
# Exit:   0 = every node answered, is provisioned (.provisioned_ok), holds F2 and matches the primary hash -> OUT is
#         written; 1 = otherwise -> the rows go to b2c0925_corpus_sha256.FAILED.tsv instead and OUT is NOT written
#         (the wave script's start gate reads OUT's presence, so a failed record can never start the battery).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
F2=fd29040db19987819461693aaf385977e45840ef
OUT=research/coordination/b2c0925_corpus_sha256.tsv
FILE=data/corpus/tinystories.txt
SSHCFG=research/queue/.pool_ssh_config
SSH=(ssh -n -o ConnectTimeout=15); [ -f "$SSHCFG" ] && SSH=(ssh -n -F "$SSHCFG" -o ConnectTimeout=15)
NODES=("$@"); [ "${#NODES[@]}" -eq 0 ] && NODES=(pool1 pool2)

primary=$(sha256sum "$FILE" 2>/dev/null | cut -d' ' -f1)
[ -n "$primary" ] || { echo "⛔ no $FILE in $(pwd)" >&2; exit 1; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
rc=0
TMPF=$(mktemp "${TMPDIR:-/tmp}/b2c_corpus.XXXXXX"); trap 'rm -f "$TMPF"' EXIT
{
  printf 'node\thost\tfile\tsha256\tsource_git_sha\tprovisioned_ok\trecorded_utc\n'
  printf 'primary-checkout\t%s\t%s\t%s\t-\t-\t%s\n' "$(hostname)" "$FILE" "$primary" "$(utc)"
  for n in "${NODES[@]}"; do
    row=$("${SSH[@]}" "$n" "cd ~/derisk-pool/revisions/$F2 2>/dev/null || exit 3; \
      h=\$(hostname); s=\$(sha256sum $FILE 2>/dev/null | cut -d' ' -f1); r=\$(cat .source_revision 2>/dev/null); \
      [ -f .provisioned_ok ] && p=yes || p=no; printf '%s\t%s\t%s\t%s' \"\$h\" \"\$s\" \"\$r\" \"\$p\"")
    if [ -z "$row" ]; then
      printf '%s\tunreachable-or-not-provisioned\t%s\t-\t-\tno\t%s\n' "$n" "$FILE" "$(utc)"; rc=1; continue
    fi
    IFS=$'\t' read -r host sha rev prov <<< "$row"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$n" "$host" "$FILE" "${sha:--}" "${rev:--}" "$prov" "$(utc)"
    { [ "$sha" = "$primary" ] && [ "$rev" = "$F2" ] && [ "$prov" = yes ]; } || rc=1
  done
} > "$TMPF"
cat "$TMPF"
if [ "$rc" -eq 0 ]; then
  cp "$TMPF" "$OUT"
  echo "[b2c-corpus] every node matches the primary hash, is provisioned, and holds $F2 -> $OUT (commit it)"
else
  cp "$TMPF" "${OUT%.tsv}.FAILED.tsv"
  echo "⛔ [b2c-corpus] a node is unreachable, unprovisioned, off-revision, or holds a different corpus ->" \
       "${OUT%.tsv}.FAILED.tsv; $OUT NOT written" >&2
fi
exit "$rc"
