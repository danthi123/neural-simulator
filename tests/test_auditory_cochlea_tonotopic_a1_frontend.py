from __future__ import annotations

from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.io import wavfile

from research.runners._auditory_cochlea_tonotopic_a1_frontend import (
    A1_EXC_PREFIX,
    A1_PV_PREFIX,
    AUDITORY_NERVE_PREFIX,
    AuditoryFrontendConfig,
    AuditoryNerveBridgeAdapter,
    CochlearFrame,
    CochlearTransducer,
    TonotopicA1SliceConfig,
    build_tonotopic_a1_slice,
    capture_microphone_alsa,
    construction_smoke,
    erb_spaced_centers,
    load_wav_mono,
    main,
    synthesize_tone,
)


def test_erb_centers_are_ordered_and_bounded():
    cfg = AuditoryFrontendConfig()
    centers = erb_spaced_centers(cfg.low_hz, cfg.high_hz, cfg.channels)
    assert np.all(np.diff(centers) > 0.0)
    assert centers[0] == pytest.approx(cfg.low_hz, rel=1e-5)
    assert centers[-1] == pytest.approx(cfg.high_hz, rel=1e-5)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"channels": 2}, "at least 3"),
        ({"high_hz": 8_000.0}, "Nyquist guard"),
        ({"ihc_compression_power": 0.0}, "compression"),
    ],
)
def test_frontend_config_fails_closed(change, message):
    with pytest.raises(ValueError, match=message):
        AuditoryFrontendConfig(**(AuditoryFrontendConfig().__dict__ | change)).validate()


def test_transducer_rejects_invalid_waveforms():
    transducer = CochlearTransducer(AuditoryFrontendConfig())
    with pytest.raises(ValueError, match="mono"):
        transducer.process(np.zeros((10, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        transducer.process(np.asarray([np.nan], dtype=np.float32))
    with pytest.raises(ValueError, match="normalized"):
        transducer.process(np.asarray([1.1], dtype=np.float32))


def test_streaming_cochlear_chunks_match_whole_buffer():
    cfg = AuditoryFrontendConfig(channels=12, high_hz=6_000.0)
    tone = synthesize_tone(700.0, 0.12, cfg.sample_rate_hz)
    whole = CochlearTransducer(cfg).process(tone)
    chunked = CochlearTransducer(cfg)
    parts = [chunked.process(chunk) for chunk in np.array_split(tone, 7)]
    assert np.array_equal(
        whole.auditory_nerve_spikes,
        np.concatenate([part.auditory_nerve_spikes for part in parts], axis=1),
    )


def test_tones_preserve_ordered_cochlear_place():
    cfg = AuditoryFrontendConfig(channels=20, high_hz=6_500.0)
    centers = erb_spaced_centers(cfg.low_hz, cfg.high_hz, cfg.channels)
    targets = [3, 10, 16]
    peaks = []
    for target in targets:
        frame = CochlearTransducer(cfg).process(
            synthesize_tone(float(centers[target]), 0.25, cfg.sample_rate_hz)
        )
        peaks.append(int(np.argmax(frame.auditory_nerve_spikes.sum(axis=1))))
    assert peaks == sorted(peaks)
    assert max(abs(actual - target) for actual, target in zip(peaks, targets)) <= 2


def test_hair_cell_compression_is_monotonic_but_sublinear():
    cfg = AuditoryFrontendConfig(channels=12, high_hz=6_000.0)
    weak = CochlearTransducer(cfg).process(
        synthesize_tone(1_000.0, 0.20, cfg.sample_rate_hz, amplitude=0.02)
    )
    strong = CochlearTransducer(cfg).process(
        synthesize_tone(1_000.0, 0.20, cfg.sample_rate_hz, amplitude=0.20)
    )
    weak_drive = float(weak.hair_cell_drive.mean())
    strong_drive = float(strong.hair_cell_drive.mean())
    assert strong_drive > weak_drive > 0.0
    assert strong_drive / weak_drive < 10.0


def test_slice_builds_channel_aligned_shared_substrate_regions():
    front = AuditoryFrontendConfig(channels=8, high_hz=6_000.0)
    built = build_tonotopic_a1_slice(front)
    assert len(built.regions) == 24
    names = {region.name for region in built.regions}
    for channel in range(front.channels):
        assert f"{AUDITORY_NERVE_PREFIX}{channel:02d}" in names
        assert f"{A1_EXC_PREFIX}{channel:02d}" in names
        assert f"{A1_PV_PREFIX}{channel:02d}" in names
        same_place = [
            region.coordinate_center
            for region in built.regions
            if region.name.endswith(f"{channel:02d}")
        ]
        assert len(set(same_place)) == 1


def test_a1_receives_only_nerve_or_internal_synaptic_inputs():
    front = AuditoryFrontendConfig(channels=8, high_hz=6_000.0)
    built = build_tonotopic_a1_slice(front)
    a1_names = set(built.excitatory_a1_regions) | set(built.inhibitory_a1_regions)
    allowed_sources = set(built.auditory_nerve_regions) | a1_names
    assert all(
        pathway.from_region in allowed_sources
        for pathway in built.pathways
        if pathway.to_region in a1_names
    )
    assert all(
        pathway.from_region.startswith(AUDITORY_NERVE_PREFIX)
        for pathway in built.pathways
        if pathway.to_region.startswith(A1_EXC_PREFIX)
        and pathway.from_region not in a1_names
    )
    assert not any("wave" in pathway.from_region or "label" in pathway.from_region for pathway in built.pathways)


def test_slice_has_spiking_local_and_flanking_gaba_pathways():
    front = AuditoryFrontendConfig(channels=8, high_hz=6_000.0)
    built = build_tonotopic_a1_slice(front)
    gaba = [path for path in built.pathways if path.from_region.startswith(A1_PV_PREFIX)]
    assert gaba
    assert all(path.receptor == "gaba_a" and path.plastic is False for path in gaba)
    assert any(path.from_region.endswith("03") and path.to_region.endswith("05") for path in gaba)
    assert all(path.graded is False for path in built.pathways)


class _FakeRegionManager:
    def __init__(self, regions):
        self._regions = regions

    def indices(self, name):
        if name not in self._regions:
            raise KeyError(name)
        return self._regions[name]


class _FakeBridge:
    def __init__(self, channels=3, neurons_per_channel=2):
        self.region_manager = _FakeRegionManager(
            {
                f"{AUDITORY_NERVE_PREFIX}{channel:02d}": range(
                    channel * neurons_per_channel, (channel + 1) * neurons_per_channel
                )
                for channel in range(channels)
            }
        )
        self.core_config = type("Config", (), {"dt_ms": 1.0})()
        self.runtime_state = SimpleNamespace(current_time_ms=0.0, current_time_step=0)
        self.cp_external_input_current = np.full(channels * neurons_per_channel + 5, 17.0)
        self.snapshots = []

    def _run_one_simulation_step(self):
        self.snapshots.append(self.cp_external_input_current.copy())


def test_adapter_drives_only_auditory_nerve_regions_and_restores_current():
    front = AuditoryFrontendConfig(channels=3, high_hz=6_000.0)
    slice_cfg = TonotopicA1SliceConfig(auditory_nerve_neurons_per_channel=2)
    bridge = _FakeBridge()
    before = bridge.cp_external_input_current.copy()
    spikes = np.zeros((3, 32), dtype=bool)
    spikes[1, 5] = True
    frame = CochlearFrame(
        erb_spaced_centers(front.low_hz, front.high_hz, front.channels),
        np.zeros((3, 32), dtype=np.float32),
        spikes,
    )
    summary = AuditoryNerveBridgeAdapter(bridge, front, slice_cfg).drive(frame)
    assert summary.simulation_steps == 2
    assert summary.channel_event_counts == (0, 1, 0)
    assert np.array_equal(bridge.cp_external_input_current, before)
    assert bridge.snapshots[0][2:4].tolist() == [slice_cfg.input_drive_pA] * 2
    assert bridge.snapshots[0][:2].tolist() == [0.0, 0.0]
    assert bridge.snapshots[0][4:6].tolist() == [0.0, 0.0]
    assert bridge.snapshots[0][6:].tolist() == before[6:].tolist()
    assert bridge.runtime_state.current_time_ms == 2.0
    assert bridge.runtime_state.current_time_step == 2


def test_adapter_rejects_waveform_or_mismatched_channels():
    front = AuditoryFrontendConfig(channels=3, high_hz=6_000.0)
    adapter = AuditoryNerveBridgeAdapter(_FakeBridge(), front)
    with pytest.raises(TypeError, match="CochlearFrame"):
        adapter.drive(np.zeros(100, dtype=np.float32))
    wrong = CochlearFrame(
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        np.zeros((3, 10), dtype=np.float32),
        np.zeros((3, 10), dtype=bool),
    )
    with pytest.raises(ValueError, match="centers"):
        adapter.drive(wrong)


def test_slice_initializes_on_real_shared_bridge():
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig

    front = AuditoryFrontendConfig(channels=3, high_hz=6_000.0)
    built = build_tonotopic_a1_slice(front)
    config = CoreSimConfig()
    config.enable_brain_region_framework = True
    config.brain_regions = list(built.regions)
    config.region_pathways = list(built.pathways)
    config.seed = config.ou_seed = config.heterogeneity_seed = 123
    runtime = RuntimeState()
    runtime.actual_seed_used = 123
    bridge = SimulationBridge(
        core_config=config,
        viz_config=VisualizationConfig(),
        runtime_state=runtime,
        gpu_config=GPUConfig(),
    )
    try:
        bridge._initialize_simulation_data()
        adapter = AuditoryNerveBridgeAdapter(bridge, front)
        assert len(adapter._nerve_indices) == front.channels
        assert config.num_neurons == sum(region.n_neurons for region in built.regions)
    finally:
        bridge.clear_simulation_state_and_gpu_memory()


def test_wav_loader_normalizes_integer_pcm_and_downmixes(tmp_path: Path):
    path = tmp_path / "stereo.wav"
    stereo = np.asarray([[32767, -32768], [16384, 16384]], dtype=np.int16)
    wavfile.write(path, 16_000, stereo)
    audio = load_wav_mono(path, 16_000)
    assert audio.dtype == np.float32
    assert audio.shape == (2,)
    assert np.max(np.abs(audio)) <= 1.0
    assert audio[1] == pytest.approx(0.5, abs=1e-4)


def test_microphone_capture_uses_pcm_stdout_without_shell(monkeypatch):
    pcm = np.asarray([0, 32767, -32768, 16384], dtype="<i2")
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout=pcm.tobytes(), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    audio = capture_microphone_alsa(0.00025, 16_000, device="test")
    assert seen["argv"][0] == "arecord"
    assert seen["kwargs"]["check"] is True
    assert audio.tolist() == pytest.approx([0.0, 32767 / 32768, -1.0, 0.5])


def test_default_construction_smoke_is_peripheral_and_no_seed():
    result = construction_smoke()
    assert result["kind"] == "cochlear_construction_smoke_no_seed"
    assert result["pass"] is True
    assert all(result["checks"].values())
    assert result["shared_slice"]["brain_internal_host_dynamics"] is False
    assert "A1 functional calibration" in result["claims_excluded"]


def test_optional_summary_output_is_create_only(tmp_path):
    output = tmp_path / "construction.json"
    assert main(["--output", str(output)]) == 0
    with pytest.raises(FileExistsError):
        main(["--output", str(output)])
