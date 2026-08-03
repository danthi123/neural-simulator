# Quickstart

This guide gets a fresh checkout to a verified CPU or NVIDIA GPU setup. Run all
commands from the repository root; this project does not currently install as a
Python package.

## Requirements

- Python 3.10 or newer. Python 3.11 is used for current development.
- Git and `pip`.
- For GPU use: a supported NVIDIA GPU, working NVIDIA driver, and a CuPy package
  compatible with the CUDA runtime you intend to use.

## Create An Environment

```bash
git clone https://github.com/danthi123/neural-simulator.git
cd neural-simulator
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

## CPU-Only Setup

Do **not** install `requirements.txt` on a CPU-only machine. That file currently
includes `cupy-cuda12x` and is a full NVIDIA/CUDA 12 environment, not a portable
base manifest.

Install the headless CPU dependencies:

```bash
python -m pip install numpy scipy h5py pyyaml
```

Install test dependencies when developing:

```bash
python -m pip install -r requirements-dev.txt
```

Verify that NumPy and SciPy are selected:

```bash
SIM_BACKEND=numpy python - <<'PY'
from sim.backend import get_backend, get_sparse_module

_, name = get_backend()
print("backend:", name)
print("sparse module:", get_sparse_module().__name__)
PY
```

Expected output includes `backend: numpy` and `scipy.sparse`. Then run a focused
CPU smoke test:

```bash
SIM_BACKEND=numpy python -m pytest tests/test_strict_step_errors.py -q
```

The desktop GUI and several older headless launchers still import CuPy directly.
CPU-only users should use backend-compatible tests and research modules, checking
each module's header before running it.

## NVIDIA GPU Setup

First confirm that the driver can see the GPU:

```bash
nvidia-smi
```

Install the simulator and desktop dependencies:

```bash
python -m pip install numpy scipy h5py pyyaml
python -m pip install dearpygui PyOpenGL PyOpenGL-accelerate psutil hdf5plugin
```

Then install the CuPy wheel matching your CUDA environment. The repository's
current full manifest uses the CUDA 12 wheel:

```bash
python -m pip install cupy-cuda12x
```

Use a different official CuPy wheel when your CUDA runtime requires it; do not
install more than one CuPy package in the same environment.

Verify the GPU backend:

```bash
SIM_BACKEND=cupy python - <<'PY'
from sim.backend import get_backend, get_device_properties

_, name = get_backend()
print("backend:", name)
print("device:", get_device_properties().get("name"))
PY
```

Launch the desktop simulator:

```bash
SIM_BACKEND=cupy python neural-simulator.py
```

## Optional Components

Tests and coverage:

```bash
python -m pip install -r requirements-dev.txt
```

Research dashboard:

```bash
python -m pip install -r webapp/requirements.txt
SIM_BACKEND=numpy python -m uvicorn webapp.server:app --port 8765
```

Open `http://127.0.0.1:8765`. Use `SIM_BACKEND=cupy` instead when launching
GPU research runs from the dashboard.

## Backend Rules

- Set `SIM_BACKEND=numpy` for CPU work.
- Set `SIM_BACKEND=cupy` for NVIDIA GPU work.
- Leaving it unset auto-detects CuPy first, but explicit selection makes research
  runs reproducible and prevents an unnoticed device change.
- In PowerShell, set it with `$env:SIM_BACKEND="numpy"` or
  `$env:SIM_BACKEND="cupy"` before the Python command.

## Next Steps

- [User Guide](USER_GUIDE.md): working interfaces and workflows.
- [Current State](docs/CURRENT-STATE.md): what the project can and cannot do.
- [Roadmap](ROADMAP.md): current priorities.
- [Contributing](CONTRIBUTING.md): engineering and research expectations.
