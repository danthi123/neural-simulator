"""CHECKPOINT-RESUME infra for the token-supply LM scaling runner (additive, default-safe, correctness-critical).

WHY (2026-09-18): the decisive extended-d384 token-supply scaling test is ~2.5 GPU-h/seed (~370M tokens). The local
gpu_queue is PAUSED whenever the owner games (`tools/gpu_queue.sh pause --now`), which SIGTERMs the running job's
process group, waits ~2s, then SIGKILLs it, and re-queues the SAME command at the front. Without resume, every pause
throws away up to 2.5h of a seed. This module makes a paused+resumed run reach the SAME final result (wkv_deep_nll
etc., bit-exact on CPU) as an uninterrupted one, wasting at most one checkpoint interval.

TWO GRANULARITIES (both load-bearing):
  1. CELL-LEVEL (`RunProgress`): each (seed, token_point) cell's finished result is written to a sidecar the moment
     the cell completes, so a restart SKIPS already-finished cells and resumes at the first unfinished one.
  2. INTRA-CELL STEP-LEVEL (`IntraCellCheckpoint`): inside the ~2.5h cell, every N optimizer steps (and/or every T
     seconds) the model + optimizer + ALL RNG + the exact data-iteration position are saved; on restart the cell
     LOADS that and continues from that step, not from 0. This is the one that matters for the big cell.
  3. SIGTERM/SIGINT: `install_term_handlers()` arms a flag the training loop checks each step; when set, the loop
     saves a FINAL checkpoint immediately and exits cleanly (raises SystemExit(0)) -- so a `pause --now` loses ~0.

THE CORRECTNESS GUARANTEE it exists to provide: faithful continuation requires restoring EVERY source of randomness
and EVERY piece of optimizer/model/data-position state. `capture_rng`/`restore_rng` snapshot python `random`, numpy's
global RNG, the trainer's LOCAL numpy Generator (which drives the per-epoch data `order` -- the only RNG the training
loop actually consumes), and torch's CPU (+CUDA) RNG. The trainer does NOT use cupy (it is a torch model; SIM_BACKEND
selects the sim substrate, not this LM), so no cupy state is captured -- documented here so the omission is a decision,
not a gap. On CPU the ops are deterministic, so uninterrupted vs resumed is BIT-EXACT (verified in the runner's test).

STALE-CHECKPOINT SAFETY: every checkpoint/sidecar stores a config hash; on load a mismatch is IGNORED (the run starts
clean) rather than silently resuming an incompatible config. This is what makes auto-resume safe to leave ON by default
-- a checkpoint only ever resumes a byte-for-byte identical (seed, token_point, config).
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path

import numpy as np

# ---- SIGTERM / SIGINT flag ------------------------------------------------------------------------------------------
# gpu_queue's `pause --now` sends SIGTERM to the job's process group, waits ~2s, then SIGKILL. We catch SIGTERM/SIGINT,
# set a flag, and let the TRAINING LOOP save at the next safe step boundary (never inside the handler -- saving a torch
# state_dict mid-signal is re-entrancy-unsafe). The ~2s window is ample for the small state_dict this runner produces;
# the periodic checkpoint is the guaranteed fallback if a SIGKILL ever beats the save.
_TERM = {"flag": False, "signum": None}


def install_term_handlers():
    """Arm SIGTERM/SIGINT -> set the module flag. Idempotent; a no-op off the main thread (where signal.signal raises)."""
    def _h(signum, _frame):
        _TERM["flag"] = True
        _TERM["signum"] = signum
    for s in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(s, _h)
        except (ValueError, OSError):
            pass  # not main thread / platform without this signal -- checkpointing still works, just no fast-exit


def term_requested() -> bool:
    return _TERM["flag"]


def request_term_for_test():
    """TEST-ONLY seam: set the flag as if a SIGTERM arrived, so the CPU test can exercise the save-and-exit path
    without actually signalling the process. NEVER called in production (only from the runner's scratchpad test)."""
    _TERM["flag"] = True
    _TERM["signum"] = int(signal.SIGTERM)


def clear_term_for_test():
    _TERM["flag"] = False
    _TERM["signum"] = None


# ---- config hashing -------------------------------------------------------------------------------------------------
def config_hash(fields: dict) -> str:
    """Stable short hash of a config dict; any change to a field that would change the trained result changes the hash,
    which invalidates (ignores) a stale checkpoint/sidecar."""
    return hashlib.sha256(json.dumps(fields, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---- RNG capture / restore ------------------------------------------------------------------------------------------
def capture_rng(device, np_rng) -> dict:
    """Snapshot every RNG a faithful continuation must restore. `np_rng` is the trainer's LOCAL numpy Generator (the
    one driving the per-epoch data `order`); the rest (python random, numpy global, torch cpu/cuda) are captured
    defensively so a future stochastic op cannot silently desync a resume."""
    import torch
    st = {
        "py_random": random.getstate(),
        "np_global": np.random.get_state(),
        "np_local": np_rng.bit_generator.state,     # THE load-bearing one: drives `order = rng.permutation(...)`
        "torch_cpu": torch.get_rng_state(),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st: dict, device, np_rng):
    import torch
    if st.get("py_random") is not None:
        random.setstate(st["py_random"])
    if st.get("np_global") is not None:
        np.random.set_state(st["np_global"])
    if st.get("np_local") is not None:
        np_rng.bit_generator.state = st["np_local"]
    if st.get("torch_cpu") is not None:
        # torch.get_rng_state() returns a CPU ByteTensor; torch.load may hand it back on another device -> force CPU.
        rs = st["torch_cpu"]
        torch.set_rng_state(rs.cpu() if hasattr(rs, "cpu") else rs)
    if st.get("torch_cuda") is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in st["torch_cuda"]])
        except Exception as e:
            print(f"    [ckpt] could not restore CUDA RNG ({e}) -- continuing (CPU/local RNG already restored)", flush=True)


def _atomic_write_bytes(path: Path, write_fn):
    """Write via a temp file + os.replace so a killed process never leaves a truncated/corrupt checkpoint for the next
    run to (silently) load. `write_fn(tmp_path)` does the actual serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".tmp.{path.name}.{os.getpid()}")
    write_fn(tmp)
    os.replace(tmp, path)


# ---- intra-cell step checkpoint -------------------------------------------------------------------------------------
class IntraCellCheckpoint:
    """One (seed, token_point, d_model) cell's mid-training checkpoint. Bound to the trainer's device + local numpy RNG
    inside build_and_train_wkv (which owns them). `enabled=False` makes every method a no-op so the trainer path is
    byte-identical to before."""

    def __init__(self, ckpt_dir, seed, token_point, d_model, cfg_hash,
                 every_steps=500, every_seconds=300.0, enabled=True, fresh=False):
        self.path = Path(ckpt_dir) / f"cell_s{seed}_k{token_point}_d{d_model}.pt"
        self.cfg_hash = cfg_hash
        self.every_steps = int(every_steps)
        self.every_seconds = float(every_seconds)
        self.enabled = bool(enabled)
        self.fresh = bool(fresh)
        self._last_step = 0
        self._last_time = time.time()
        self._device = "cpu"
        self._np_rng = None
        if self.fresh:
            self.clear()

    def bind(self, device, np_rng):
        self._device = device
        self._np_rng = np_rng
        self._last_time = time.time()

    def load(self):
        """Return the saved training-state dict, or None (nothing saved / disabled / stale config / corrupt)."""
        if not self.enabled or self.fresh or not self.path.exists():
            return None
        import torch
        try:
            st = torch.load(self.path, map_location=self._device, weights_only=False)
        except Exception as e:
            print(f"    [ckpt] load failed ({e}) -- ignoring, training this cell from scratch", flush=True)
            return None
        if st.get("cfg_hash") != self.cfg_hash:
            print(f"    [ckpt] STALE checkpoint (hash {st.get('cfg_hash')} != current {self.cfg_hash}) "
                  f"-- ignoring, training this cell from scratch", flush=True)
            return None
        return st

    def should_save(self, global_step) -> bool:
        if not self.enabled:
            return False
        if term_requested():
            return True
        if self.every_steps > 0 and (global_step - self._last_step) >= self.every_steps:
            return True
        if self.every_seconds > 0 and (time.time() - self._last_time) >= self.every_seconds:
            return True
        return False

    def save(self, state: dict, global_step):
        if not self.enabled:
            return
        import torch
        state = dict(state)
        state["cfg_hash"] = self.cfg_hash
        _atomic_write_bytes(self.path, lambda tmp: torch.save(state, tmp))
        self._last_step = int(global_step)
        self._last_time = time.time()

    @staticmethod
    def term_requested_now() -> bool:
        return term_requested()

    @staticmethod
    def raise_exit():
        """Clean, traceback-free exit after a SIGTERM/SIGINT-triggered save. gpu_queue re-queues the SAME command, so
        the next dispatch auto-resumes from the checkpoint just written."""
        raise SystemExit(0)

    def clear(self):
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass


# ---- cell-level progress sidecar ------------------------------------------------------------------------------------
class RunProgress:
    """Durable per-(seed, token_point) cell results. Written incrementally so a restart skips finished cells. Keyed by
    a GLOBAL config hash (everything that would invalidate ALL cells: corpus, d_model, vocab, epochs, batch, max_len,
    n_sentences, max_eval_sents, and the wkv training hyperparams) -- NOT the seeds/token_points lists, so a rerun with
    a subset/superset of seeds or points still reuses the cells that match. A per-seed eval_ids_sha guard (checked by
    the caller) catches any residual pool/seed mismatch."""

    def __init__(self, sidecar_path, cfg_hash, enabled=True, fresh=False):
        self.path = Path(sidecar_path)
        self.cfg_hash = cfg_hash
        self.enabled = bool(enabled)
        self.data = {"cfg_hash": cfg_hash, "per_seed": {}}
        if fresh:
            self.clear()
        elif self.enabled and self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                if d.get("cfg_hash") == cfg_hash:
                    self.data = d
                else:
                    print("    [progress] stale sidecar (config-hash mismatch) -- ignoring", flush=True)
            except Exception as e:
                print(f"    [progress] sidecar load failed ({e}) -- ignoring", flush=True)

    def seed_entry(self, seed):
        return self.data["per_seed"].get(str(seed))

    def cell_done(self, seed, token_point):
        """Return the stored point dict for a finished cell, or None."""
        e = self.data["per_seed"].get(str(seed))
        if not e:
            return None
        for pt in e.get("points", []):
            if pt.get("max_train_sents") == token_point:
                return pt
        return None

    def drop_seed(self, seed):
        self.data["per_seed"].pop(str(seed), None)
        if self.enabled:
            self._flush()

    def record_cell(self, seed, token_point, point, seed_meta=None, gen_samples=None):
        e = self.data["per_seed"].setdefault(str(seed), {"points": []})
        if seed_meta:
            e.update(seed_meta)
        if gen_samples is not None:
            e["gen_samples_top_point"] = gen_samples
        pts = [p for p in e.get("points", []) if p.get("max_train_sents") != token_point]
        pts.append(point)
        pts.sort(key=lambda p: p.get("max_train_sents", 0))
        e["points"] = pts
        if self.enabled:
            self._flush()

    def _flush(self):
        _atomic_write_bytes(self.path, lambda tmp: Path(tmp).write_text(json.dumps(self.data, indent=2)))

    def clear(self):
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass
