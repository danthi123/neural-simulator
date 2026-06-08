"""Host-side per-region activity reduction for live brain-activity streaming.

Frontend-revamp Phase 1 (2026-06-08). See
docs/plans/2026-06-08-frontend-revamp-design.md §3.2.

This module is ADDITIVE and used ONLY when a runner opts in (e.g.
g11_bg_runner --emit-activity). It is NOT imported or touched by the
simulation step loop; the bridge has ZERO knowledge of it. The probe reads
the bridge's already-public state (`cp_firing_states` + `region_manager`)
from the OUTSIDE and produces ~30 host floats per *sampled* frame:

  - per-region mean firing fraction (one boolean-mean reduce per region)
  - per-pathway flux (a cheap proxy: the post-region's mean firing — reuses
    the region reduction, no extra GPU work)

LOAD-BEARING CONSTRAINT (owner directive): the viz must NEVER bottleneck the
sim. This probe is consistent with that because:
  * it is only constructed + sampled when --emit-activity is set; when off,
    nothing here runs and the step loop is byte-identical;
  * `sample()` is called on a THROTTLE (every N steps), not every step;
  * each sample is O(regions + pathways), not O(neurons): ~30 reductions +
    one host transfer of ~30 scalars — negligible vs a full step;
  * the emit (sim/progress.emit_activity) is fire-and-forget: it writes a
    stdout line and returns. The probe never waits on a reader.

The per-region slices are STATIC (region_manager allocates contiguous
index ranges once), so the index arrays are precomputed in __init__ and the
per-frame cost is just the reduction + transfer.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class RegionActivityProbe:
    """Per-region (and per-pathway) activity reduction for live streaming.

    Usage (runner-side, throttled):

        from sim.activity_probe import RegionActivityProbe
        from sim.progress import emit_activity

        probe = RegionActivityProbe(bridge)          # built ONCE, before the loop
        ...
        for step in range(n_steps):
            bridge._run_one_simulation_step()
            if emit_activity_on and (step + 1) % emit_every == 0:
                regions, flux = probe.sample(bridge)
                emit_activity(bridge.runtime_state.current_time_ms,
                              regions, flux, step=step + 1)
    """

    def __init__(
        self,
        bridge,
        region_names: Optional[List[str]] = None,
        ema_alpha: float = 0.5,
    ):
        """Precompute per-region device index arrays from the bridge.

        Args:
            bridge: a SimulationBridge whose region_manager is initialized and
                whose cp_firing_states array exists.
            region_names: optional subset of regions to probe; default = all
                regions the region_manager knows about.
            ema_alpha: host-side display-rate EMA smoothing for the emitted
                rates so the viz isn't strobic. rate_ema = a*rate + (1-a)*ema.
                1.0 disables smoothing; 0.5 is a gentle default. The EMA is
                applied at the (slow) sample cadence, NOT the sim step rate, so
                it never touches the sim.

        Raises:
            RuntimeError if the bridge lacks a region_manager (the probe needs
            the per-region neuron slices) or cp_firing_states.
        """
        rm = getattr(bridge, "region_manager", None)
        if rm is None:
            raise RuntimeError(
                "RegionActivityProbe requires bridge.region_manager "
                "(enable the brain-region framework)."
            )
        if getattr(bridge, "cp_firing_states", None) is None:
            raise RuntimeError(
                "RegionActivityProbe requires bridge.cp_firing_states "
                "(call _initialize_simulation_data first)."
            )

        # Resolve the array module the same way the bridge does, so our index
        # arrays live on the same device (CuPy GPU or NumPy host) as
        # cp_firing_states. Fall back to plain numpy if the backend module is
        # unavailable in this context.
        try:
            from sim.backend import get_backend
            xp, _ = get_backend()
        except Exception:  # pragma: no cover - defensive bootstrap
            import numpy as xp  # type: ignore

        self._xp = xp
        self.ema_alpha = float(ema_alpha)

        names = list(region_names) if region_names is not None \
            else list(rm.region_indices_dict().keys())

        # Precompute one device int64 index array per region (STATIC slices).
        self._region_idx: Dict[str, "xp.ndarray"] = {}
        for name in names:
            idx = rm.indices(name)
            if not idx:
                continue
            self._region_idx[name] = xp.asarray(idx, dtype=xp.int64)

        # Precompute (from_region, to_region) -> pathway-name for flux. Flux is
        # a proxy: the post-region's mean firing (already computed in sample()),
        # so no extra GPU reduce. We only keep pathways whose post-region is
        # being probed. Pathway name matches extract_per_pathway_csrs():
        # "<from>_to_<to>".
        self._pathways: List[Tuple[str, str]] = []  # (pw_name, post_region)
        try:
            for pw in rm.pathways():
                if pw.to_region in self._region_idx:
                    self._pathways.append(
                        (f"{pw.from_region}_to_{pw.to_region}", pw.to_region)
                    )
        except Exception:  # pragma: no cover - pathways() unavailable
            self._pathways = []

        # Host-side EMA state (per region), seeded lazily on first sample.
        self._ema: Dict[str, float] = {}

    @property
    def n_regions(self) -> int:
        return len(self._region_idx)

    @property
    def n_pathways(self) -> int:
        return len(self._pathways)

    def region_names(self) -> List[str]:
        return list(self._region_idx.keys())

    def sample(self, bridge) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute per-region mean firing + per-pathway flux for this frame.

        Returns (regions, flux):
            regions: {region_name: ema-smoothed firing fraction in [0,1]}
            flux:    {pathway_name: post-region firing fraction in [0,1]}

        Cost: O(regions) boolean-mean reductions + one host transfer of ~30
        scalars. Independent of neuron count. Called on the runner's throttle,
        never every step.
        """
        fired = bridge.cp_firing_states  # bool[N] on device
        xp = self._xp

        # One mean-reduce per region. We compute each region's mean on-device
        # then pull the scalar to host. (Batching into a single transfer is a
        # future micro-opt; at ~30 regions the per-scalar .get() cost is
        # already negligible relative to a full sim step.)
        raw: Dict[str, float] = {}
        a = self.ema_alpha
        for name, idx in self._region_idx.items():
            # mean over a boolean slice = fraction of the region's neurons that
            # fired this step.
            val = float(fired[idx].mean())
            prev = self._ema.get(name)
            if prev is None or a >= 1.0:
                ema = val
            else:
                ema = a * val + (1.0 - a) * prev
            self._ema[name] = ema
            raw[name] = round(ema, 4)

        flux: Dict[str, float] = {}
        for pw_name, post_region in self._pathways:
            # Flux proxy = the post-synaptic region's (smoothed) firing. This
            # reuses `raw` so there is NO additional GPU reduction. A weighted
            # version (× mean pathway weight) is a Phase-2 refinement.
            v = raw.get(post_region)
            if v is not None:
                flux[pw_name] = v

        return raw, flux
