import hashlib
import json
from pathlib import Path

from sim.snr_executable_packet import canonical_bytes


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "research/specs/v14_snr_stageB_causal_gates.json"
GATE_V2_PATH = ROOT / "research/specs/v14_snr_stageB_causal_gates_v2.json"


def _spec():
    return json.loads(GATE_PATH.read_text(encoding="utf-8"))


def test_causal_gate_contract_is_canonical_for_authenticated_execution():
    assert GATE_PATH.read_bytes() == canonical_bytes(_spec())


def test_causal_gate_contract_is_bound_to_current_target_packet():
    spec = _spec()
    target = spec["target_packet"]
    assert hashlib.sha256((ROOT / target["path"]).read_bytes()).hexdigest() == target["sha256"]
    assert spec["status"] == "readiness-draft-not-executable"


def test_source_derived_and_project_operational_values_are_not_collapsed():
    spec = _spec()
    gates = {gate["id"]: gate for gate in spec["causal_gates"]}
    assert gates["cav2.2-complete-lesion"]["derived"] == {
        "rate_after_over_before": 13.97 / 11.22,
        "cv_after_over_before": 1.75,
    }
    nap = gates["nap-complete-lesion"]
    assert nap["source"] == "Ding-Wei-Zhou-2011"
    assert "Atherton and Bevan supplied the riluzole result" in nap["forbidden_interpretations"]
    assert nap["hard_gates"][0]["evidence_class"] == "project_operational"
    hcn = gates["hcn-complete-lesion"]
    assert hcn["hard_gates"][1]["value"] == 0.20
    assert hcn["hard_gates"][1]["evidence_class"] == "project_operational"
    assert "not a source equivalence" in hcn["boundary"]


def test_directional_controls_preserve_non_significant_and_descriptive_boundaries():
    gates = {gate["id"]: gate for gate in _spec()["causal_gates"]}
    sk = gates["sk-complete-lesion"]
    assert sk["source_reported"]["rate_p"] == 0.20
    assert "firing-rate increase" in sk["not_a_gate"]
    assert "exact 4-of-12 depolarization-block match" in sk["not_a_gate"]
    pallidal = gates["pallidal-barrage-step-release"]
    assert pallidal["source_reported"]["rate_reduction_low_fraction"] == 0.20
    assert pallidal["source_reported"]["rate_reduction_high_fraction"] == 0.50
    assert "step_over_barrage_overshoot_area_ratio_mean" in pallidal["descriptive_not_hard_interval"]


def test_scope_excludes_unmatched_passive_values_and_tracks_dynamic_calcium_work():
    scope = _spec()["scope_decisions"]
    assert scope["adult_capacitance_and_input_resistance"]["decision"] == "exclude_from_stageB_scoring"
    calcium = scope["calcium_reversal"]
    assert calcium["primary_campaign"].startswith("constant packet E_Ca")
    assert "not yet implemented" in calcium["engineering_status"]


def test_analysis_protocols_preserve_event_count_and_under_specified_boundaries():
    protocols = _spec()["analysis_protocol_boundaries"]
    firing = protocols["atherton_bevan_2005_firing_characteristics"]
    assert "101 spontaneous action potentials" in firing["source_reported"]
    assert "event-count trace" in firing["required_runner_semantics"]
    assert "arbitrary fixed-duration" in firing["required_runner_semantics"]

    assert protocols["medium_ahp"]["status"] == "protocol_under_specified"
    hcn = protocols["hcn_hyperpolarized_input_resistance"]
    assert hcn["status"] == "protocol_under_specified"
    assert "does not by itself define" in hcn["boundary"]

    sk = protocols["sk_depolarization_block"]
    assert "4 of 12" in sk["source_reported"]
    assert "not an exact block-onset target" in sk["boundary"]


def test_v2_contract_adds_directional_total_ahp_assay_without_rewriting_v1():
    v1 = _spec()
    v2 = json.loads(GATE_V2_PATH.read_text(encoding="ascii"))
    assert GATE_V2_PATH.read_bytes() == canonical_bytes(v2)
    assert v2["schema"] == "v14-snr-stageB-causal-gates-v2"
    assert v1["analysis_protocol_boundaries"]["medium_ahp"]["status"] == (
        "protocol_under_specified"
    )
    nadir = v2["analysis_protocol_boundaries"]["post_spike_voltage_nadir"]
    assert nadir["status"] == "production_project_analysis_convention"
    assert "does not define" in nadir["boundary"]
    assert "medium-AHP amplitude" in nadir["boundary"]
    protocol = v2["authorized_analysis_protocol"]
    protocol_path = ROOT / protocol["path"]
    assert hashlib.sha256(protocol_path.read_bytes()).hexdigest() == protocol["sha256"]

    gates = {gate["id"]: gate for gate in v2["causal_gates"]}
    for gate_id in ("cav2.2-complete-lesion", "sk-complete-lesion"):
        metrics = {item["metric"]: item for item in gates[gate_id]["hard_gates"]}
        assay = metrics["median_interspike_voltage_nadir_mV"]
        assert assay["operator"] == "lesion_greater_than_intact"
        assert assay["evidence_class"] == "source_reported_direction"
