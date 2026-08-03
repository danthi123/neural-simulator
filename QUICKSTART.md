# Quickstart

This gets the repo installed and gives you a few safe first runs. The project is
large, so the goal here is orientation, not a full research workflow.

## Requirements

- Python 3.10 or newer.
- Linux, Windows, or macOS for the CPU backend.
- NVIDIA GPU plus a matching CuPy package for GPU runs.
- `git`, `pip`, and enough disk for research artifacts.

The project is developed day to day on Linux with NVIDIA hardware. CPU mode is
slower but is the right default for quick tests and many research runners.

## Install

```bash
git clone https://github.com/danthi123/neural-simulator
cd neural-simulator

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

On Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Choose A Backend

CPU mode:

```bash
SIM_BACKEND=numpy python -m pytest tests/test_strict_step_errors.py -q
```

GPU mode:

```bash
python - <<'PY'
from sim.backend import xp
print(xp.__name__)
PY
```

If the GPU path is active, this should print a CuPy module name. If you want to
force CPU mode for a command, prefix it with `SIM_BACKEND=numpy`.

## First Runs

Run a short CPU conversation demo:

```bash
SIM_BACKEND=numpy python -m research.runners.chat_demo --seed 43
```

Run the GUI on a CUDA-capable machine:

```bash
python neural-simulator.py
```

Run the test suite:

```bash
pytest tests/ -q
```

Run a focused test while developing:

```bash
SIM_BACKEND=numpy python -m pytest tests/test_laneC_self_schema_honesty_wirein.py -q
```

## Working With Research Runs

Most experiments live in `research/runners/` and write raw results into
`research/findings/raw/`. A good research run should usually produce:

- a raw JSON artifact;
- a provenance sidecar;
- a dated finding in `research/findings/`;
- controls or lesion/permutation tests when claiming a mechanism works.

Before starting new research work, use the repo's preflight:

```bash
tools/before_you_build.sh <short_name> "<plain description of the work>"
```

After writing a finding, run the relevant docs and claim gates:

```bash
python tools/check_docs.py
python tools/finding_status.py --check research/findings/<finding>.md
python tools/claim_check.py research/findings/<finding>.md
```

## Troubleshooting

**CuPy import fails.** Install the CuPy package matching your CUDA stack, for
example `cupy-cuda12x` for CUDA 12.

**A run is too slow.** Force CPU only for small tests with `SIM_BACKEND=numpy`,
or use the GPU for large single runs. For independent seeds, prefer parallel CPU
workers.

**CUDA runs out of memory.** Use smaller model/run settings, reduce batch/step
counts, or switch that smoke test to CPU.

**Windows line endings or long paths cause noisy diffs.** Use
`git diff --ignore-cr-at-eol` for inspection and enable long paths if checkout
fails.

## Where To Go Next

- [README.md](README.md) - project overview.
- [ROADMAP.md](ROADMAP.md) - current plan and priorities.
- [docs/CURRENT-STATE.md](docs/CURRENT-STATE.md) - honest capability status.
- [docs/SCAFFOLD-LEDGER.md](docs/SCAFFOLD-LEDGER.md) - temporary shortcuts and
  replacement paths.
- [research/findings/](research/findings/) - detailed evidence record.
