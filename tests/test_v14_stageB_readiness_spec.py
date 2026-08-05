import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "research/specs/v14_snr_stageB_readiness.json"
TARGET_PATH = ROOT / "research/specs/v14_snr_stageB_target_packet.json"
FIXTURE_PATH = ROOT / "research/fixtures/v14_snr_stageB_scorer_fixtures.json"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _spec():
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _targets():
    return json.loads(TARGET_PATH.read_text(encoding="utf-8"))


def test_parent_lineage_is_hash_bound():
    spec = _spec()
    for name, record in spec["parent_lineage"].items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        if name == "target_packet":
            continue
        path = ROOT / record["path"]
        assert path.is_file()
        assert _sha256(path) == record["sha256"]


def test_revised_target_packet_is_hash_bound_without_rewriting_readiness():
    readiness_target = _spec()["parent_lineage"]["target_packet"]
    fixture_packet = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture_target = fixture_packet["source_target_packet"]
    assert fixture_target["path"] == readiness_target["path"]
    assert fixture_target["path"] == "research/specs/v14_snr_stageB_target_packet.json"
    assert fixture_target["sha256"] == _sha256(TARGET_PATH)


def test_seed_partitions_are_disjoint_and_readiness_opens_none():
    spec = _spec()
    partitions = spec["partitions"]
    seen = set()
    for name, seeds in partitions.items():
        assert isinstance(seeds, list), name
        assert not (seen & set(seeds)), name
        seen.update(seeds)

    authority = spec["current_authority"]
    assert authority["allowed_partitions"] == ["readiness"]
    assert partitions["readiness"] == []
    assert authority["scientific_parameter_search"] is False
    assert authority["physiology_verdict"] is False
    assert authority["promotion_effect"] == "none"
    assert spec["current_readiness"]["verdict"] == "READINESS_UNDEFINED"
    assert len(spec["current_readiness"]["open_gates"]) == 5
    assert partitions == {
        "readiness": [],
        "calibration": [590297],
        "replication": [979881, 651019, 950955],
        "held_out": [312588, 787884, 625835],
        "future_selector": [790326],
        "historical_v13_forbidden": [1031],
    }


def test_target_packet_separates_pathways_and_records_source_conditions():
    packet = _targets()
    assert packet["status"] == "readiness-draft-not-executable"
    sources = {source["id"]: source for source in packet["sources"]}
    for source_id in (
        "Simmons-et-al-2018",
        "Simmons-et-al-2020",
        "Lutas-et-al-2016",
        "McElvain-et-al-2021",
        "Sitzia-et-al-2022",
        "Thompson-et-al-2025",
        "Atherton-and-Bevan-2005",
        "Ding-Wei-Zhou-2011",
    ):
        source = sources[source_id]
        assert isinstance(source["preparation"], dict)
        assert source["source_locator"]
        assert source["preparation"]["species_age"]
        assert source["preparation"]["recording_modes"]

    assert sources["Atherton-and-Bevan-2005"]["preparation"]["temperature"].startswith("37 C")
    assert sources["Ding-Wei-Zhou-2011"]["preparation"]["temperature"].startswith("30 C")
    lutas_solution = sources["Lutas-et-al-2016"]["preparation"]["solution"]
    assert "2.5 or 4 KCl" in lutas_solution
    assert "not explicitly mapped" in lutas_solution

    transferred = {item["id"]: item for item in packet["transferred_source_observations"]}
    assert transferred["juvenile-rat-atherton-baseline"]["values"]["perforated_cv_mean"] == 0.060
    assert transferred["juvenile-rat-ding-action-potential"]["values"]["base_duration_mean_ms"] == 1.1
    assert "not an adult" in transferred["juvenile-rat-atherton-baseline"]["use"]
    lutas_figure5 = transferred["juvenile-mouse-lutas-figure5-k-unresolved"]
    assert lutas_figure5["bath_KCl_mM"] == "unresolved_2.5_or_4"
    assert lutas_figure5["evidence_label"] == "blocked-unscorable-transfer-evidence"
    assert lutas_figure5["scorable"] is False
    assert "prohibited from executable accepted targets" in lutas_figure5["use"]
    sitzia = transferred["mature-adult-mouse-sitzia-waveform-ahp-regularity"]
    assert sitzia["evidence_label"] == "transfer-only-held-out-context"
    assert sitzia["scorable"] is False
    assert sitzia["values"]["ap_half_width_mean_ms"] == 0.53
    assert "AP amplitude is not AP peak" in sitzia["uncertainty"]

    excluded = {item["id"]: item for item in packet["excluded_source_observations"]}
    thompson = excluded["adult-mouse-thompson-waveform-distributions"]
    assert thompson["evidence_label"] == "plot-only-excluded"
    assert thompson["scorable"] is False
    assert "plot digitization is not preregistered" in thompson["reason"]

    targets = {target["id"]: target for target in packet["accepted_targets"]}
    direct = targets["adult-inhibitory-conductance-support"]
    assert direct["source"] == "Simmons-et-al-2018"
    assert direct["value"]["peak_low"] == 0.35
    assert direct["value"]["peak_high"] == 6.4
    pallidal = targets["adult-pallidonigral-unitary-support"]
    assert pallidal["source"] == "Simmons-et-al-2020"
    assert pallidal["value"]["peak_low"] == 2.4
    assert pallidal["value"]["peak_high"] == 25.1
    barrage = targets["adult-pallidonigral-depressed-barrage"]
    assert barrage["value"] == {
        "peak_low": 1.6,
        "peak_high": 2.4,
        "rise": 0.4,
        "decay": 2.1,
        "event_rate": 90.0,
        "target_rate_reduction_low_percent": 20.0,
        "target_rate_reduction_high_percent": 50.0,
    }
    assert "nalcn-lesion-ratio" not in targets
    nalcn = targets["nalcn-lesion-ratio-4mM"]
    assert nalcn["value"]["intact_mean_hz"] == 30.2
    assert nalcn["value"]["lesion_mean_hz"] == 16.8
    assert nalcn["condition"].endswith("4 mM extracellular KCl")
    assert nalcn["evidence_label"] == "measured-source-statistics-with-model-derived-interval"
    assert nalcn["model_derived_interval"]["status"] == "model-derived-not-source-measured"
    assert "Figure 6-figure supplement 1A" in nalcn["source_locator"]
    mcelvain = targets["young-adult-intrinsic-isi-cv-point"]
    assert mcelvain["value"] == {"mean": 0.10, "sem": 0.10, "n": 120}
    assert mcelvain["acceptance_bound"] is None
    assert "no pass/fail scorer fixture" in mcelvain["use"]
    assert "no pooling" in mcelvain["use"]

    decisions = {item["decision"]: item for item in packet["evidence_scope_decisions"]}
    assert decisions["blocked-unscorable-transfer-only"]["record_id"] == lutas_figure5["id"]
    assert decisions["include-model-derived-transfer-fixture"]["target_id"] == nalcn["id"]
    assert decisions["include-separate-cohort-point-only"]["target_id"] == mcelvain["id"]
    assert decisions["transfer-only-held-out-context"]["record_id"] == sitzia["id"]
    assert decisions["plot-only-excluded"]["record_id"] == thompson["id"]

    unresolved_evidence = " ".join(packet["unresolved_before_executable_spec"])
    assert "Lutas Figure 5" not in unresolved_evidence
    assert "adult intrinsic CV or CV2" not in unresolved_evidence
    hcn = next(value for value in packet["directional_targets"] if value.startswith("HCN"))
    assert "did not show a statistically significant" in hcn
    assert "no equivalence claim" in hcn


def test_readiness_keeps_required_controls_and_parameter_gaps_visible():
    spec = _spec()
    arms = set(spec["required_arms"])
    assert {
        "intact",
        "nalcn-lesion",
        "nap-lesion",
        "cav2.2-lesion",
        "sk-lesion",
        "hcn-baseline-lesion",
        "hcn-hyperpolarization-lesion",
        "inhibition-source-on",
        "inhibition-source-off",
        "rate-matched-compensation-control",
        "wrong-sign-scorer-control",
    } <= arms

    surface = spec["required_parameter_surface"]
    assert {
        "hh_C_m_override",
        "hh_g_Na_max_override",
        "hh_g_K_max_override",
        "hh_g_L_override",
        "hh_E_Na_override",
        "hh_E_K_override",
        "hh_E_L_override",
    } <= set(surface["already_population_scoped"])
    integrated = " ".join(surface["executable_packet_integrated"])
    for mechanism in ("fast sodium", "potassium", "NaP", "Cav2.2", "Ih", "calcium", "SK"):
        assert mechanism in integrated
    assert "not yet wired into the bridge" not in json.dumps(surface)
    assert "packet-backed runner" in " ".join(surface["remaining_runner_integration"])


def test_future_search_budget_and_order_are_fixed():
    spec = _spec()
    budget = spec["future_search_budget"]
    assert budget["sobol_numpy_intrinsic_exact"] == 512
    assert budget["full_population_numpy_exact"] == 64
    assert budget["cupy_calibration_exact"] == 12
    assert budget["frozen_diverse_survivors"] == 4
    assert budget["selection_rule"] == {
        "space": "all filed search parameters normalized to [0,1] by their preregistered low/high bounds; log-transformed parameters are normalized after log transform",
        "distance": "Euclidean distance in normalized parameter space",
        "first": "highest preregistered normalized objective score, candidate sha256 ascending as tie-break",
        "iteration": "choose the candidate maximizing its minimum distance to the already selected set",
        "ties": "preregistered normalized objective score descending, then candidate sha256 ascending",
        "count": 4,
    }
    assert "scientific NO-GO" in budget["early_stop"]
    assert budget["range_widening_after_results"] is False
    assert budget["channel_addition_after_results"] is False
    order = spec["future_stage_order"]
    assert order == [
        "readiness-GO",
        "calibration-seed-590297-NumPy",
        "calibration-seed-590297-CuPy",
        "freeze-four-diverse-survivors",
        "replication-three-seeds-NumPy-on-pool",
        "replication-three-seeds-CuPy-under-GPU-lease",
        "replication-GO",
        "held-out-three-seeds-CuPy-first",
        "held-out-three-seeds-NumPy-second",
        "Stage-B-verdict",
    ]
