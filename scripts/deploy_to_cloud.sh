#!/bin/bash
# Cloud deployment script — works on RunPod, Lambda Labs, vast.ai, etc.
#
# Clones, installs, runs a YAML sweep, pushes results back to git.
# Designed for ephemeral cloud instances where storage doesn't persist.
#
# Usage on cloud instance:
#   curl -sL https://raw.githubusercontent.com/danthi123/neural-simulator/main/scripts/deploy_to_cloud.sh | bash -s experiments/bio_b3_validation.yaml
#
# OR (after manual clone):
#   bash scripts/deploy_to_cloud.sh experiments/bio_b3_validation.yaml
#
# Required env vars (set on cloud instance):
#   GIT_USER_EMAIL — for the commit
#   GIT_USER_NAME  — for the commit
#   GIT_PUSH_TOKEN — github personal access token (if using HTTPS)
#                    OR have ssh keys configured
#
# Optional:
#   PARALLELISM_OVERRIDE — override YAML's parallelism (e.g., 12 for H100)

set -e

REPO_URL="${REPO_URL:-https://github.com/danthi123/neural-simulator}"
SWEEP_YAML="${1:-experiments/bio_b3_validation.yaml}"
WORK_DIR="${WORK_DIR:-/workspace}"

echo "=== Cloud deploy: $(date) ==="
echo "Sweep: $SWEEP_YAML"
echo "Repo: $REPO_URL"
echo

# Step 1: clone or update
cd "$WORK_DIR"
if [ ! -d neural-simulator ]; then
    git clone "$REPO_URL"
fi
cd neural-simulator
git pull origin main

# Step 2: install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
# Verify cupy is available
python -c "import cupy; print(f'CuPy {cupy.__version__}, CUDA {cupy.cuda.runtime.runtimeGetVersion()}')"

# Step 3: optionally override parallelism (H100 80GB lets us go higher)
if [ -n "$PARALLELISM_OVERRIDE" ]; then
    echo "Override: parallelism=$PARALLELISM_OVERRIDE"
    python -c "
import yaml, sys
p = '$SWEEP_YAML'
with open(p) as f: d = yaml.safe_load(f)
d['parallelism'] = $PARALLELISM_OVERRIDE
with open(p, 'w') as f: yaml.safe_dump(d, f)
print('YAML parallelism updated')
"
fi

# Step 4: run the sweep
SWEEP_NAME=$(basename "$SWEEP_YAML" .yaml)
echo
echo "=== Running $SWEEP_NAME ==="
python -m research.experiment_runner "$SWEEP_YAML" 2>&1 | tee "$SWEEP_NAME.runlog"

# Step 5: aggregate results
echo
echo "=== Aggregating ==="
RESULTS_MD="research/findings/$(date +%Y-%m-%d)-cloud-${SWEEP_NAME}-results.md"
# Try with the YAML's name as aggregator config (e.g., bio-b3-validation -> bio_b3_validation)
AGG_CONFIG=$(echo "$SWEEP_NAME" | tr '-' '_')
python -m research.result_aggregator --config "$AGG_CONFIG" --out "$RESULTS_MD" || \
    echo "WARN: aggregator config '$AGG_CONFIG' not found; results JSON files still saved"

# Step 6: push results back
echo
echo "=== Pushing results ==="
git config user.email "${GIT_USER_EMAIL:-cloud-runner@neural-sim}"
git config user.name "${GIT_USER_NAME:-Cloud Runner}"

# Stage findings + raw results (json files)
git add research/findings/raw/g11_bg/text_eval_*.json 2>/dev/null || true
git add research/findings/*.md 2>/dev/null || true

if git diff --cached --quiet; then
    echo "No changes to commit"
else
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    git commit -m "cloud(${SWEEP_NAME}): results from ${GPU_INFO}

GPU: ${GPU_INFO}
Date: $(date)
Hostname: $(hostname)
"
    if [ -n "$GIT_PUSH_TOKEN" ]; then
        # HTTPS auth via token
        REPO_URL_AUTH=$(echo "$REPO_URL" | sed "s|https://|https://${GIT_USER_NAME}:${GIT_PUSH_TOKEN}@|")
        git push "$REPO_URL_AUTH" main
    else
        # Assume SSH or pre-configured creds
        git push origin main
    fi
fi

echo
echo "=== Done at $(date) ==="
echo "Findings: $RESULTS_MD"
