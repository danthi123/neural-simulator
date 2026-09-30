#!/usr/bin/env bash
# Launch `unsloth run` for the serving bake-off, and extract its auto-minted bearer key into a plaintext file
# that tools/local_llm/bakeoff.py's --api-key-file re-reads (see refresh_api_key_file() there) -- Unsloth Studio
# mints the key only once the model has finished loading (printed in its own banner AND saved to
# ~/.unsloth/studio/auth/agent_api_key.json), which is AFTER bakeoff.py's readiness check already needs to know
# the server is up (see generic_ready()'s 401/403-counts-as-ready handling), so the key cannot be a CLI literal.
#
# Queued as the generic profile's --start-cmd (tools/agent_container/profiles/unsloth-studio.env documents the
# manual equivalent). bakeoff.py owns stopping this: it killpg's THIS script's pid, which is also the `unsloth
# run` child's process group (it is forked without its own setsid, so it inherits ours) -- so one killpg tears
# down both. --stop-cmd (`unsloth studio stop`) is the belt-and-braces second path.
set -uo pipefail
STUDIO_VENV="${UNSLOTH_STUDIO_VENV:-$HOME/.unsloth/studio/unsloth_studio}"
MODEL="${1:-/var/tmp/ollama-import/Qwen3.8-27B-IQ4_NL.gguf}"
PORT="${2:-8888}"
KEYFILE="${3:-/var/tmp/ollama-import/unsloth_bakeoff_key.txt}"
# shellcheck source=/dev/null
source "$STUDIO_VENV/bin/activate"
rm -f "$KEYFILE"

unsloth run --model "$MODEL" --max-seq-length 131072 --gpu-memory-mode manual \
    --speculative-type mtp --spec-draft-n-max 3 --host 0.0.0.0 --port "$PORT" --parallel 1 \
    --api-only --api-key-name bakeoff &
UPID=$!

# Poll the studio's own key store for a key minted against this port (owner 2026-09-29: more robust than
# scraping the run log's banner text, which is redacted in tail views -- see start.py's sk-unsloth- redaction).
python3 - "$PORT" "$KEYFILE" <<'PYEOF' &
import json, os, sys, time
port, keyfile = sys.argv[1], sys.argv[2]
path = os.path.expanduser("~/.unsloth/studio/auth/agent_api_key.json")
deadline = time.time() + 900
while time.time() < deadline:
    try:
        data = json.load(open(path))
        for base, entry in data.get("servers", {}).items():
            if base.rstrip("/").endswith(":" + port):
                minted = entry.get("minted") or []
                if minted:
                    with open(keyfile, "w") as fh:
                        fh.write(minted[-1])
                    os.chmod(keyfile, 0o600)
                    sys.exit(0)
    except Exception:
        pass
    time.sleep(2)
PYEOF

wait "$UPID"
