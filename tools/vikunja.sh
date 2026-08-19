#!/bin/bash
# Vikunja API helper for the neural-simulator project.
# Reads the instance URL + token from ~/.claude-config/secrets/vikunja.json
# (OUTSIDE the repo — no committed file contains the secret).
# Human-readable task board that mirrors the roadmap: the owner monitors via the
# Vikunja web app; Claude syncs landings/next-rungs here as a durable source of
# truth that survives context compaction (supplements the RAG/findings record).
#
# Default output is COMPACT (context-thrifty); pass --json for raw JSON.
set -e

SECRETS_FILE="${HOME}/.claude-config/secrets/vikunja.json"
if [ ! -f "$SECRETS_FILE" ]; then
    echo "Error: Vikunja secrets file not found at $SECRETS_FILE" >&2
    echo 'Create it as: {"apiUrl":"https://vikunja.dant123.com","apiToken":"tk_..."}' >&2
    exit 1
fi
TOKEN=$(grep -o '"apiToken" *: *"[^"]*"' "$SECRETS_FILE" | cut -d'"' -f4)
BASE=$(grep -o '"apiUrl" *: *"[^"]*"' "$SECRETS_FILE" | cut -d'"' -f4)
API_URL="${BASE%/}/api/v1"
if [ -z "$TOKEN" ] || [ -z "$BASE" ]; then
    echo "Error: could not read apiToken/apiUrl from $SECRETS_FILE" >&2
    exit 1
fi

FORMAT="compact"
if [[ "$1" == "--json" ]]; then FORMAT="json"; shift; fi

# ---- formatters (all use single-quoted python; double-quote strings, single-quote dict keys) ----
format_tasks() {
    if [[ "$FORMAT" == "json" ]]; then cat; else
        python3 -c '
import sys, json, re
from datetime import datetime
tasks = json.load(sys.stdin)
if not isinstance(tasks, list): tasks = []
for t in sorted(tasks, key=lambda x: (x.get("done", False), -(x.get("priority") or 0), x.get("id", 0))):
    status = "check" if t.get("done") else "open"
    mark = "x" if t.get("done") else " "
    line = "[" + mark + "] #" + str(t["id"]) + " " + str(t.get("title", ""))
    pr = t.get("priority") or 0
    if pr: line += " (p" + str(pr) + ")"
    dd = t.get("due_date", "")
    if dd and dd != "0001-01-01T00:00:00Z":
        line += " (Due: " + datetime.fromisoformat(dd.replace("Z", "+00:00")).strftime("%Y-%m-%d") + ")"
    print(line)
    desc = re.sub("<[^>]+>", "", (t.get("description") or "")).strip()
    if desc:
        for ln in desc.splitlines():
            if ln.strip(): print("    " + ln.strip())
'
    fi
}
format_task() {
    if [[ "$FORMAT" == "json" ]]; then cat; else
        python3 -c '
import sys, json, re
t = json.load(sys.stdin)
status = "DONE" if t.get("done") else "OPEN"
print("Task #" + str(t.get("id")) + " " + status + ": " + str(t.get("title", "")))
desc = re.sub("<[^>]+>", "", (t.get("description") or "")).strip()
if desc: print("  " + desc)
'
    fi
}

case "$1" in
    list-projects)
        curl -s "${API_URL}/projects" -H "Authorization: Bearer $TOKEN" | \
        if [[ "$FORMAT" == "json" ]]; then cat; else
            python3 -c '
import sys, json
for p in json.load(sys.stdin):
    if p["id"] > 0:
        par = p.get("parent_project_id", 0)
        s = "[" + str(p["id"]) + "] " + str(p.get("title", ""))
        if par: s += "  (under " + str(par) + ")"
        print(s)
'
        fi
        ;;

    create-project)
        TITLE="$2"; PARENT="${3:-0}"; DESC="${4:-}"
        if [ -z "$TITLE" ]; then echo 'Usage: create-project "title" [parent_id] ["description"]' >&2; exit 1; fi
        JSON=$(TITLE="$TITLE" PARENT="$PARENT" DESC="$DESC" python3 -c '
import json, os
d = {"title": os.environ["TITLE"]}
if os.environ.get("DESC"): d["description"] = os.environ["DESC"]
p = int(os.environ.get("PARENT") or 0)
if p > 0: d["parent_project_id"] = p
print(json.dumps(d))')
        curl -s -X PUT "${API_URL}/projects" -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" -d "$JSON" | \
        if [[ "$FORMAT" == "json" ]]; then cat; else
            python3 -c 'import sys, json; p = json.load(sys.stdin); print("Created project [" + str(p.get("id")) + "] " + str(p.get("title", "")))'
        fi
        ;;

    list-tasks)
        PROJECT_ID="${2:-all}"
        if [[ "$PROJECT_ID" == "all" ]]; then
            curl -s "${API_URL}/projects" -H "Authorization: Bearer $TOKEN" | \
            TOKEN="$TOKEN" API_URL="$API_URL" python3 -c '
import sys, json, os, urllib.request
token = os.environ["TOKEN"]; api = os.environ["API_URL"]
for p in json.load(sys.stdin):
    if p["id"] <= 0: continue
    try:
        req = urllib.request.Request(api + "/projects/" + str(p["id"]) + "/tasks", headers={"Authorization": "Bearer " + token})
        tasks = json.load(urllib.request.urlopen(req))
    except Exception as e:
        print("  (error " + str(p.get("title")) + ": " + str(e) + ")"); continue
    if not isinstance(tasks, list): tasks = []
    opent = [t for t in tasks if not t.get("done")]
    done = [t for t in tasks if t.get("done")]
    print("\n=== " + str(p.get("title")) + " (" + str(len(opent)) + " open / " + str(len(done)) + " done) ===")
    for t in sorted(opent, key=lambda x: (-(x.get("priority") or 0), x.get("id", 0))):
        line = "[ ] #" + str(t["id"]) + " " + str(t.get("title", ""))
        if t.get("priority"): line += " (p" + str(t["priority"]) + ")"
        print(line)
'
        else
            curl -s "${API_URL}/projects/${PROJECT_ID}/tasks" -H "Authorization: Bearer $TOKEN" | format_tasks
        fi
        ;;

    create-task)
        PROJECT_ID="$2"; TITLE="$3"; DESC="${4:-}"; PRIORITY="${5:-0}"; DUE="${6:-}"
        if [ -z "$PROJECT_ID" ] || [ -z "$TITLE" ]; then echo 'Usage: create-task <project_id> "title" ["desc"] [priority 0-5] [YYYY-MM-DD]' >&2; exit 1; fi
        JSON=$(TITLE="$TITLE" DESC="$DESC" PRIORITY="$PRIORITY" DUE="$DUE" python3 -c '
import json, os
d = {"title": os.environ["TITLE"]}
if os.environ.get("DESC"): d["description"] = os.environ["DESC"]
pr = int(os.environ.get("PRIORITY") or 0)
if pr: d["priority"] = pr
due = os.environ.get("DUE")
if due: d["due_date"] = due + "T23:59:59Z"
print(json.dumps(d))')
        curl -s -X PUT "${API_URL}/projects/${PROJECT_ID}/tasks" -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" -d "$JSON" | format_task
        ;;

    update-task)
        TASK_ID="$2"; DONE="${3:-true}"
        if [ -z "$TASK_ID" ]; then echo 'Usage: update-task <task_id> [true|false]' >&2; exit 1; fi
        curl -s -X POST "${API_URL}/tasks/${TASK_ID}" -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" -d "{\"done\": $DONE}" | format_task
        ;;

    set-desc)
        TASK_ID="$2"; DESC="$3"
        if [ -z "$TASK_ID" ] || [ -z "$DESC" ]; then echo 'Usage: set-desc <task_id> "description"' >&2; exit 1; fi
        JSON=$(DESC="$DESC" python3 -c 'import json, os; print(json.dumps({"description": os.environ["DESC"]}))')
        curl -s -X POST "${API_URL}/tasks/${TASK_ID}" -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" -d "$JSON" | format_task
        ;;

    set-priority)
        TASK_ID="$2"; PRIO="$3"
        if [ -z "$TASK_ID" ] || [ -z "$PRIO" ]; then echo 'Usage: set-priority <task_id> <0-5>' >&2; exit 1; fi
        curl -s -X POST "${API_URL}/tasks/${TASK_ID}" -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" -d "{\"priority\": $PRIO}" | format_task
        ;;

    set-due-date)
        TASK_ID="$2"; DUE="$3"
        if [ -z "$TASK_ID" ] || [ -z "$DUE" ]; then echo 'Usage: set-due-date <task_id> <YYYY-MM-DD>' >&2; exit 1; fi
        curl -s -X POST "${API_URL}/tasks/${TASK_ID}" -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" -d "{\"due_date\": \"${DUE}T23:59:59Z\"}" | format_task
        ;;

    *)
        cat >&2 <<'EOF'
Vikunja helper (neural-simulator). URL+token from ~/.claude-config/secrets/vikunja.json
  tools/vikunja.sh [--json] list-projects
  tools/vikunja.sh [--json] create-project "title" [parent_id] ["description"]
  tools/vikunja.sh [--json] list-tasks [project_id|all]
  tools/vikunja.sh [--json] create-task <project_id> "title" ["desc"] [priority 0-5] [YYYY-MM-DD]
  tools/vikunja.sh [--json] update-task <task_id> [true|false]   # mark done/undone
  tools/vikunja.sh [--json] set-desc <task_id> "description"
  tools/vikunja.sh [--json] set-priority <task_id> <0-5>
  tools/vikunja.sh [--json] set-due-date <task_id> <YYYY-MM-DD>
EOF
        exit 1
        ;;
esac
