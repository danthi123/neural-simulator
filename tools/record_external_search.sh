#!/usr/bin/env bash
# record_external_search.sh — the WRITER for research/.last_external_search, a marker the heartbeat's
# workflow_check.sh rule 3 has always READ but that NOTHING ever wrote (FAILURE_LOG 2026-08-01: a
# recognized-but-never-produced marker). Run this after a genuine EXTERNAL-literature read (WebSearch /
# WebFetch / bio-research MCP / a paper PDF) so the heartbeat stops nagging "GO EXTERNAL" after you already did,
# and so the deep-external read leaves a trail — the companion to before_you_build.sh (corpus) and
# research_gate.sh (the research gate). It does NOT satisfy gates/boundary-verdict-external-check: that gate
# reads the FINDING's own citations, deliberately, so the external touch-point travels WITH the claim rather
# than living in a side-marker (a marker can go stale against the doc; an inline arxiv/DOI citation cannot).
#
# Usage: bash tools/record_external_search.sh "<query>" "<key source / url / one-line finding>"
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q="${1:-}"; SRC="${2:-}"
[ -z "$Q" ] && { echo "usage: bash tools/record_external_search.sh \"<query>\" \"<key source/url/author-year>\""; exit 2; }
# A SOURCE IS REQUIRED (2026-08-09, owner-flagged recurrence). Every prior entry had an EMPTY source — i.e. the
# "external search" was logged but no external literature was actually read. gates/deep-research-at-wall only
# accepts a NON-EMPTY source. An honest "none found after searching" is a valid source string ("none-found: <why>").
[ -z "$SRC" ] && { echo "⛔ REFUSED: a real external source is required (paper / arxiv / doi / (Author, YEAR) / 'none-found: <why>')." >&2
                   echo "   Do the external search FIRST (bio-research consensus/pubmed MCP, WebSearch, or a paper), then record what it found." >&2; exit 2; }

MARK="$ROOT/research/.last_external_search"
LOG="$ROOT/research/queue/.external_searches.jsonl"
mkdir -p "$ROOT/research/queue"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
GITSHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"

# touch the marker the heartbeat reads (its mtime is what rule 3 compares against the newest finding)
: > "$MARK"; printf '%s\t%s\n' "$TS" "$Q" >> "$MARK"

# append a durable JSONL record (mirrors before_you_build.sh's .corpus_checks.jsonl)
_json_escape() { python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1" 2>/dev/null || printf '"%s"' "${1//\"/\\\"}"; }
printf '{"ts": %s, "git": %s, "query": %s, "source": %s}\n' \
  "$(_json_escape "$TS")" "$(_json_escape "$GITSHA")" "$(_json_escape "$Q")" "$(_json_escape "$SRC")" >> "$LOG"

echo "  [recorded] external search logged to research/queue/.external_searches.jsonl + touched research/.last_external_search"
echo "  NOTE: to satisfy gates/boundary-verdict-external-check, the CITATION must also appear in the finding itself"
echo "        (an arxiv/doi/Sources reference, an (Author, YEAR) cite, or a NO-EXTERNAL-NEEDED: line)."
