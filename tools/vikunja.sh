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

# ---- board-sync RECEIPT ----------------------------------------------------------------
# Every MUTATING board op appends a line to research/coordination/board_sync.json. This is
# the audit trail gates/board_sync_on_status_change.py reads: a commit that advances a
# faculty's status (docs/PRODUCTION_INTEGRATION_LEDGER.yaml) BLOCKS until the board was
# actually synced, and "actually synced" means this receipt grew in the same commit. So the
# receipt is produced by RUNNING the sync (here), never by hand-touching a file.
_record() {
    local op="$1"; shift; local args="$*"
    local root rf
    root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    rf="$root/research/coordination/board_sync.json"
    op="$op" args="$args" rf="$rf" root="$root" python3 - <<'PY' 2>/dev/null || true
import json, os, time, subprocess
rf, root = os.environ["rf"], os.environ["root"]
os.makedirs(os.path.dirname(rf), exist_ok=True)
try:
    d = json.load(open(rf))
    assert isinstance(d, dict) and isinstance(d.get("entries"), list)
except Exception:
    d = {"schema": "board-sync-v1", "entries": []}
try:
    head = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, timeout=5).stdout.strip()
except Exception:
    head = ""
d["entries"].append({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "op": os.environ["op"], "args": os.environ["args"][:240], "head": head})
d["updated_at"] = d["entries"][-1]["at"]
json.dump(d, open(rf, "w"), indent=1)
PY
}

# ---- READ-MODIFY-WRITE update -----------------------------------------------------------
# ⚠️ Vikunja's `POST /tasks/{id}` is a FULL REPLACE: a partial body silently resets EVERY
# unspecified scalar field (start/end/due dates, description, priority all wiped). The old
# per-field curl POSTs therefore clobbered each other when used in sequence. _patch fetches
# the whole task, merges only the field(s) in $MERGE (a JSON object in the env), and posts
# the whole object back — so setting the date never erases the description, and so on.
# Relational fields (labels, relations, …) live on their own endpoints and are dropped from
# the body so they are never reprocessed/reset.
_patch() {
    local tid="$1"
    TID="$tid" API_URL="$API_URL" TOKEN="$TOKEN" python3 - <<'PY'
import json, os, urllib.request
api, token, tid = os.environ["API_URL"], os.environ["TOKEN"], os.environ["TID"]
merge = json.loads(os.environ.get("MERGE") or "{}")
req = urllib.request.Request(api + "/tasks/" + tid, headers={"Authorization": "Bearer " + token})
t = json.load(urllib.request.urlopen(req))
for k in ("labels", "related_tasks", "reactions", "attachments", "assignees", "subscription"):
    t.pop(k, None)
t.update(merge)
body = json.dumps(t).encode()
r = urllib.request.Request(api + "/tasks/" + tid, data=body, method="POST",
                           headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
res = json.load(urllib.request.urlopen(r))
print("Task #%s %s: %s" % (res.get("id"), "DONE" if res.get("done") else "OPEN", res.get("title", "")))
PY
}

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
        # PAGINATED. Vikunja CAPS per_page at 50 regardless of the value sent, so a single
        # request silently truncated the board to its first 50 tasks — the "single pane of
        # glass" was showing HALF the board (102 tasks, 53 open, only 50 returned). We now
        # loop pages until a short/empty page. (Fixed 2026-08-21, research/vikunja-pane-of-glass.)
        PROJECT_ID="${2:-all}"
        if [[ "$PROJECT_ID" == "all" ]]; then
            curl -s "${API_URL}/projects" -H "Authorization: Bearer $TOKEN" | \
            TOKEN="$TOKEN" API_URL="$API_URL" python3 -c '
import sys, json, os, urllib.request
token = os.environ["TOKEN"]; api = os.environ["API_URL"]
def fetch_all(pid):
    out, page = [], 1
    while True:
        req = urllib.request.Request(api + "/projects/" + str(pid) + "/tasks?per_page=50&page=" + str(page),
                                     headers={"Authorization": "Bearer " + token})
        chunk = json.load(urllib.request.urlopen(req))
        if not isinstance(chunk, list) or not chunk: break
        out += chunk
        if len(chunk) < 50: break
        page += 1
        if page > 200: break
    return out
for p in json.load(sys.stdin):
    if p["id"] <= 0: continue
    try:
        tasks = fetch_all(p["id"])
    except Exception as e:
        print("  (error " + str(p.get("title")) + ": " + str(e) + ")"); continue
    opent = [t for t in tasks if not t.get("done")]
    done = [t for t in tasks if t.get("done")]
    print("\n=== " + str(p.get("title")) + " (" + str(len(opent)) + " open / " + str(len(done)) + " done) ===")
    for t in sorted(opent, key=lambda x: (-(x.get("priority") or 0), x.get("id", 0))):
        line = "[ ] #" + str(t["id"]) + " " + str(t.get("title", ""))
        if t.get("priority"): line += " (p" + str(t["priority"]) + ")"
        print(line)
'
        else
            TOKEN="$TOKEN" API_URL="$API_URL" PID="$PROJECT_ID" python3 -c '
import json, os, urllib.request
token = os.environ["TOKEN"]; api = os.environ["API_URL"]; pid = os.environ["PID"]
out, page = [], 1
while True:
    req = urllib.request.Request(api + "/projects/" + pid + "/tasks?per_page=50&page=" + str(page),
                                 headers={"Authorization": "Bearer " + token})
    chunk = json.load(urllib.request.urlopen(req))
    if not isinstance(chunk, list) or not chunk: break
    out += chunk
    if len(chunk) < 50: break
    page += 1
    if page > 200: break
print(json.dumps(out))
' | format_tasks
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
        _record create-task "proj=$PROJECT_ID title=$TITLE"
        ;;

    update-task)
        TASK_ID="$2"; DONE="${3:-true}"
        if [ -z "$TASK_ID" ]; then echo 'Usage: update-task <task_id> [true|false]' >&2; exit 1; fi
        MERGE="{\"done\": $DONE}" _patch "$TASK_ID"
        _record update-task "task=$TASK_ID done=$DONE"
        ;;

    set-desc)
        TASK_ID="$2"; DESC="$3"
        if [ -z "$TASK_ID" ] || [ -z "$DESC" ]; then echo 'Usage: set-desc <task_id> "description"' >&2; exit 1; fi
        MERGE=$(DESC="$DESC" python3 -c 'import json, os; print(json.dumps({"description": os.environ["DESC"]}))') _patch "$TASK_ID"
        _record set-desc "task=$TASK_ID"
        ;;

    append-desc)
        # Idempotent append of a machine/status line onto a task's existing description, preserving it.
        # Used to attach the [lane:…] (Ref:…) machine field without clobbering the human lead.
        TASK_ID="$2"; ADD="$3"
        if [ -z "$TASK_ID" ] || [ -z "$ADD" ]; then echo 'Usage: append-desc <task_id> "text to append"' >&2; exit 1; fi
        CUR=$(curl -s "${API_URL}/tasks/${TASK_ID}" -H "Authorization: Bearer $TOKEN")
        MERGE=$(CUR="$CUR" ADD="$ADD" python3 -c '
import json, os, re
t = json.loads(os.environ["CUR"])
cur = (t.get("description") or "")
add = os.environ["ADD"]
plain = re.sub("<[^>]+>", "", cur)
if add.strip() and add.strip() not in plain:
    sep = "" if cur.endswith(">") or cur == "" else "<br/>"
    cur = cur + sep + add
print(json.dumps({"description": cur}))') _patch "$TASK_ID"
        _record append-desc "task=$TASK_ID"
        ;;

    set-title)
        TASK_ID="$2"; TITLE="$3"
        if [ -z "$TASK_ID" ] || [ -z "$TITLE" ]; then echo 'Usage: set-title <task_id> "new title"' >&2; exit 1; fi
        MERGE=$(TITLE="$TITLE" python3 -c 'import json, os; print(json.dumps({"title": os.environ["TITLE"]}))') _patch "$TASK_ID"
        _record set-title "task=$TASK_ID"
        ;;

    set-priority)
        TASK_ID="$2"; PRIO="$3"
        if [ -z "$TASK_ID" ] || [ -z "$PRIO" ]; then echo 'Usage: set-priority <task_id> <0-5>' >&2; exit 1; fi
        MERGE="{\"priority\": $PRIO}" _patch "$TASK_ID"
        _record set-priority "task=$TASK_ID prio=$PRIO"
        ;;

    set-due-date)
        TASK_ID="$2"; DUE="$3"
        if [ -z "$TASK_ID" ] || [ -z "$DUE" ]; then echo 'Usage: set-due-date <task_id> <YYYY-MM-DD>' >&2; exit 1; fi
        MERGE="{\"due_date\": \"${DUE}T23:59:59Z\"}" _patch "$TASK_ID"
        _record set-due-date "task=$TASK_ID due=$DUE"
        ;;

    set-dates)
        # Set start + end (+ due) so the Gantt renders a real bar. end is the target/deadline.
        TASK_ID="$2"; START="$3"; END="$4"
        if [ -z "$TASK_ID" ] || [ -z "$START" ] || [ -z "$END" ]; then
            echo 'Usage: set-dates <task_id> <start YYYY-MM-DD> <end YYYY-MM-DD>' >&2; exit 1; fi
        MERGE="{\"start_date\": \"${START}T00:00:00Z\", \"end_date\": \"${END}T23:59:59Z\", \"due_date\": \"${END}T23:59:59Z\"}" _patch "$TASK_ID"
        _record set-dates "task=$TASK_ID start=$START end=$END"
        ;;

    add-relation)
        # Make dependencies visible on the Gantt + machine-readable for parallelizability.
        # add-relation <task> <other> [kind]  — default kind 'blocked' => <task> is BLOCKED BY <other>.
        # Vikunja stores the relation from <task>'s perspective; it auto-creates the inverse on <other>.
        TID="$2"; OTHER="$3"; KIND="${4:-blocked}"
        if [ -z "$TID" ] || [ -z "$OTHER" ]; then
            echo 'Usage: add-relation <task_id> <other_task_id> [blocked|blocking|related|precedes|follows]' >&2; exit 1; fi
        JSON="{\"task_id\": $TID, \"other_task_id\": $OTHER, \"relation_kind\": \"$KIND\"}"
        curl -s -X PUT "${API_URL}/tasks/${TID}/relations" -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" -d "$JSON" >/dev/null \
            && echo "relation: #$TID $KIND #$OTHER"
        _record add-relation "task=$TID $KIND other=$OTHER"
        ;;

    del-relation)
        TID="$2"; OTHER="$3"; KIND="${4:-blocked}"
        if [ -z "$TID" ] || [ -z "$OTHER" ]; then echo 'Usage: del-relation <task_id> <other_task_id> [kind]' >&2; exit 1; fi
        curl -s -X DELETE "${API_URL}/tasks/${TID}/relations/${KIND}/${OTHER}" -H "Authorization: Bearer $TOKEN" >/dev/null \
            && echo "deleted relation: #$TID $KIND #$OTHER"
        ;;

    delete-project)
        PID="$2"
        if [ -z "$PID" ]; then echo 'Usage: delete-project <project_id>' >&2; exit 1; fi
        curl -s -X DELETE "${API_URL}/projects/${PID}" -H "Authorization: Bearer $TOKEN" >/dev/null && echo "Deleted project $PID"
        ;;

    delete-task)
        TID="$2"
        if [ -z "$TID" ]; then echo 'Usage: delete-task <task_id>' >&2; exit 1; fi
        curl -s -X DELETE "${API_URL}/tasks/${TID}" -H "Authorization: Bearer $TOKEN" >/dev/null && echo "Deleted task $TID"
        ;;

    list-labels)
        curl -s "${API_URL}/labels" -H "Authorization: Bearer $TOKEN" | \
        if [[ "$FORMAT" == "json" ]]; then cat; else
            python3 -c '
import sys, json
d = json.load(sys.stdin)
for l in (d if isinstance(d, list) else []):
    print("[" + str(l.get("id")) + "] " + str(l.get("title", "")))
'
        fi
        ;;

    create-label)
        TITLE="$2"; COLOR="${3:-}"
        if [ -z "$TITLE" ]; then echo 'Usage: create-label "title" [hexcolor]' >&2; exit 1; fi
        JSON=$(TITLE="$TITLE" COLOR="$COLOR" python3 -c '
import json, os
d = {"title": os.environ["TITLE"]}
if os.environ.get("COLOR"): d["hex_color"] = os.environ["COLOR"].lstrip("#")
print(json.dumps(d))')
        curl -s -X PUT "${API_URL}/labels" -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" -d "$JSON" | \
        if [[ "$FORMAT" == "json" ]]; then cat; else
            python3 -c 'import sys, json; l = json.load(sys.stdin); print("Created label [" + str(l.get("id")) + "] " + str(l.get("title", "")))'
        fi
        ;;

    label-task)
        TID="$2"; LID="$3"
        if [ -z "$TID" ] || [ -z "$LID" ]; then echo 'Usage: label-task <task_id> <label_id>' >&2; exit 1; fi
        curl -s -X PUT "${API_URL}/tasks/${TID}/labels" -H "Authorization: Bearer $TOKEN" \
            -H "Content-Type: application/json" -d "{\"label_id\": $LID}" >/dev/null && echo "Labeled task $TID with $LID"
        _record label-task "task=$TID label=$LID"
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
  tools/vikunja.sh [--json] append-desc <task_id> "text"        # idempotent append (machine field)
  tools/vikunja.sh [--json] set-priority <task_id> <0-5>
  tools/vikunja.sh [--json] set-due-date <task_id> <YYYY-MM-DD>
  tools/vikunja.sh [--json] set-dates <task_id> <start> <end>   # start+end+due (Gantt bar)
  tools/vikunja.sh add-relation <task_id> <other_id> [kind]     # default 'blocked' (task blocked-by other)
  tools/vikunja.sh del-relation <task_id> <other_id> [kind]
  tools/vikunja.sh label-task <task_id> <label_id>
EOF
        exit 1
        ;;
esac
