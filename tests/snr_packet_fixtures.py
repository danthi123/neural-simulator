"""Compact typed fixtures for authenticated SNr bridge tests."""

from types import MappingProxyType, SimpleNamespace

from sim.snr_packet_parameters import (
    CalciumParameters,
    Cav22Parameters,
    FastHHParameters,
    GeometryParameters,
    HCNParameters,
    IonicEnvironmentParameters,
    NaLCNParameters,
    NaPParameters,
    Q10Factors,
    SKParameters,
    SNrPacketParameters,
)


def runtime_parameters(*, packet_id="snr-test", scale=1.0):
    """Return finite, non-default typed values without bypassing bridge checks."""
    return SNrPacketParameters(
        packet_id=packet_id,
        packet_sha256="a" * 64,
        structural_sha256="b" * 64,
        raw_groups=MappingProxyType({}),
        parsed_values=MappingProxyType({}),
        fast_hh=FastHHParameters(
            1.2, 101.0, 31.0, 0.21, -61.0, -18.0,
            0.12, 0.62, 0.32, 2.1, 1.7, 1.8,
        ),
        nalcn=NaLCNParameters(0.011 * scale),
        nap=NaPParameters(
            0.02, 0.024 * scale, -48.0, 4.0,
            0.08, 0.16, -42.0, 14.0, -14.0,
            -58.0, -5.0, 12.0, 24.0, -34.0, 27.0, -31.0,
            2.0, 16.0,
        ),
        cav22=Cav22Parameters(
            0.03, 0.036 * scale, -26.0, 3.5, 0.7,
            -51.0, -5.5, 21.0, 3.0, 2.0, 16.0,
        ),
        hcn=HCNParameters(
            0.01, 0.012 * scale, -77.0, -6.0, 115.0, 2.0, 16.0,
        ),
        calcium=CalciumParameters(0.08, 55.0, 0.5, 2.0, 16.0),
        sk=SKParameters(
            0.04, 0.048 * scale, 0.45, 4.0, 2.0, 8.0, 2.0, 16.0,
        ),
        geometry=GeometryParameters(1000.0, 200.0),
        ionic_env=IonicEnvironmentParameters(
            52.0, -92.0, -67.0, -8.0, 118.0, -32.0,
            2000.0, 2.0, 36.0,
        ),
        q10_factors=Q10Factors(2.1, 1.7, 1.8, 4.0, 4.0, 4.0, 4.0, 4.0),
        calcium_influx_um_per_ms_per_inward_ua_per_cm2=0.003,
    )


def runtime_binding(*, label="packet", packet_id="snr-test", scale=1.0):
    return SimpleNamespace(
        label=label,
        runtime_parameters=runtime_parameters(packet_id=packet_id, scale=scale),
    )
