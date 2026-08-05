from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sim.snr_executable_packet import PARAMETER_SCHEMA, canonical_bytes
from tools.v14_stageB_packet_compiler import compile_documents


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "research/specs/v14_snr_stageB_packet_template.json"
SPEC_PATH = ROOT / "research/specs/v14_snr_stageB_executable_spec.json"
SCORER_FIXTURES_PATH = ROOT / "research/fixtures/v14_snr_stageB_scorer_fixtures.json"
CANDIDATE_LOW_PATH = ROOT / "research/specs/v14_snr_stageB_readiness_candidate_low.json"
CANDIDATE_HIGH_PATH = ROOT / "research/specs/v14_snr_stageB_readiness_candidate_high.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_bytes())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _leaves(template: dict[str, object]):
    for group, leaves in template["parameter_leaves"].items():
        for parameter, leaf in leaves.items():
            yield group, parameter, leaf


def _candidate(template: dict[str, object], candidate_id: str, endpoint: str):
    parameters = {
        leaf["candidate_key"]: leaf["bounds"][endpoint]
        for _, _, leaf in _leaves(template)
        if leaf["mode"] == "searched"
    }
    return {
        "schema": "sim-adaptive-candidate-v1",
        "candidate_id": candidate_id,
        "parameters": parameters,
    }


def test_template_is_canonical_and_exactly_matches_the_69_leaf_runtime_surface():
    raw = TEMPLATE_PATH.read_bytes()
    template = json.loads(raw)

    assert raw == canonical_bytes(template)
    assert set(template["parameter_leaves"]) == set(PARAMETER_SCHEMA)
    assert sum(len(leaves) for leaves in template["parameter_leaves"].values()) == 69
    for group, schema in PARAMETER_SCHEMA.items():
        assert set(template["parameter_leaves"][group]) == set(schema)
        for parameter, units in schema.items():
            assert template["parameter_leaves"][group][parameter]["unit"] in units


def test_executable_spec_binds_current_canonical_sources_and_template_bytes():
    spec = _load(SPEC_PATH)
    for binding in spec["source_bindings"].values():
        source = ROOT / binding["path"]
        assert source.is_file()
        assert _sha256(source) == binding["sha256"]

    template = _load(TEMPLATE_PATH)
    assert TEMPLATE_PATH.read_bytes() == canonical_bytes(template)
    assert spec["source_bindings"]["packet_template"]["sha256"] == _sha256(
        TEMPLATE_PATH
    )


def test_search_coordinates_and_fixed_priors_preserve_authority_boundaries():
    template = _load(TEMPLATE_PATH)
    spec = _load(SPEC_PATH)
    filed_bounds = spec["parameter_authority"]["search_bounds"]
    candidate_keys = []

    for group, parameter, leaf in _leaves(template):
        if leaf["mode"] == "searched":
            candidate_keys.append(leaf["candidate_key"])
            assert leaf["evidence"] == "derived"
            assert leaf["authority"] == "project_decision"
            assert set(leaf["bounds"]) == {"low", "high"}
            assert leaf["transform"] in {"linear", "log"}
            filed = filed_bounds[leaf["candidate_key"]]
            assert filed["leaf"] == f"{group}.{parameter}"
            assert filed["low"] == leaf["bounds"]["low"]
            assert filed["high"] == leaf["bounds"]["high"]
            assert filed["transform"] == leaf["transform"]
        else:
            assert leaf["mode"] == "fixed"
            assert leaf["evidence"] == "model_prior"
            assert leaf["authority"] == "model_source"

    assert len(candidate_keys) == len(set(candidate_keys))
    assert set(candidate_keys) == set(filed_bounds)
    assert spec["parameter_authority"]["adult_density_claim_from_fit"] is False
    assert spec["parameter_authority"]["adult_q10_claim"] is False


def test_potassium_is_searched_while_confounded_geometry_has_held_out_sensitivity():
    template = _load(TEMPLATE_PATH)
    spec = _load(SPEC_PATH)
    potassium = template["parameter_leaves"]["fast_hh"][
        "potassium_conductance_density"
    ]
    geometry = spec["scientific_boundaries"]["geometry_current_fraction"]

    assert potassium["mode"] == "searched"
    assert potassium["candidate_key"] == "fast_hh_potassium_conductance_density"
    assert potassium["bounds"] == {"low": "2", "high": "8"}
    assert potassium["evidence"] == "derived"
    assert potassium["authority"] == "project_decision"

    assert template["parameter_leaves"]["geometry"]["membrane_area"]["mode"] == "fixed"
    assert template["parameter_leaves"]["geometry"]["accessible_calcium_volume"][
        "mode"
    ] == "fixed"
    assert template["parameter_leaves"]["calcium"]["current_fraction"]["mode"] == "fixed"
    assert "divided by accessible_calcium_volume" in geometry["identifiability"]
    assert {arm["id"] for arm in geometry["held_out_sensitivities"]} == {
        "source-model-shell-depth-0p2um",
        "current-fraction-half-equivalence",
        "area-half-preserve-shell",
        "area-double-preserve-shell",
    }
    assert "not an additional search coordinate" in geometry["current_fraction_boundary"]
    assert "effective_calcium_coupling_sensitivity" in spec["readiness_only_execution"][
        "missing_runner_arms"
    ]


def test_status_honestly_withholds_promotion_while_protocols_and_controls_are_missing():
    spec = _load(SPEC_PATH)
    readiness = _load(ROOT / spec["source_bindings"]["readiness_spec"]["path"])
    execution = spec["readiness_only_execution"]
    fixtures = _load(SCORER_FIXTURES_PATH)
    bounded_fixture_ids = {
        item["id"]
        for item in fixtures["fixtures"]
        if item["score_kind"] == "bounded-interval"
    }

    assert spec["status"] == "READINESS_UNDEFINED_NON_EXECUTABLE"
    assert spec["executable"] is False
    assert readiness["current_readiness"]["verdict"] == "READINESS_UNDEFINED"
    assert execution["scientific_parameter_search"] is False
    assert execution["physiology_verdict"] is False
    assert execution["allowed_partitions"] == ["readiness"]
    assert execution["implemented_runner_arms"] == [
        "intact_autonomous", "nap_lesion", "cav2_2_lesion", "sk_lesion",
        "hcn_baseline_lesion",
    ]
    assert execution["missing_runner_arms"]
    assert execution["missing_scorer_contracts"]
    implemented = set(execution["implemented_scorer_contracts"])
    assert bounded_fixture_ids <= implemented
    assert {
        "nap-complete-lesion-partial", "cav2.2-complete-lesion-partial",
        "sk-complete-lesion-partial", "hcn-complete-lesion-partial",
    } <= implemented
    assert "forbidden" in execution["scientific_seed_material"]


def test_compiler_accepts_two_distinct_in_range_candidates_from_exact_template():
    template = _load(TEMPLATE_PATH)
    template_sha256 = _sha256(TEMPLATE_PATH)
    low = compile_documents(
        template,
        _candidate(template, "stageB-template-low", "low"),
        template_sha256=template_sha256,
    )
    high = compile_documents(
        template,
        _candidate(template, "stageB-template-high", "high"),
        template_sha256=template_sha256,
    )

    low_packet = low["packet.structural.json"]
    high_packet = high["packet.structural.json"]
    assert sum(len(leaves) for leaves in low_packet["groups"].values()) == 69
    assert sum(len(leaves) for leaves in high_packet["groups"].values()) == 69
    assert low_packet["packet_id"] != high_packet["packet_id"]
    assert low["compilation-request.json"]["candidate_sha256"] != high[
        "compilation-request.json"
    ]["candidate_sha256"]


def test_filed_readiness_candidates_are_canonical_exact_template_endpoints():
    template = _load(TEMPLATE_PATH)

    for path, candidate_id, endpoint in (
        (CANDIDATE_LOW_PATH, "v14-stageB-readiness-low", "low"),
        (CANDIDATE_HIGH_PATH, "v14-stageB-readiness-high", "high"),
    ):
        document = _load(path)
        assert path.read_bytes() == canonical_bytes(document)
        assert document["schema"] == "sim-adaptive-candidate-v1"
        assert document["candidate_id"] == candidate_id
        expected_parameters = {
            leaf["candidate_key"]: float(leaf["bounds"][endpoint])
            for _, _, leaf in _leaves(template)
            if leaf["mode"] == "searched"
        }
        assert set(document["parameters"]) == set(expected_parameters)
        assert {
            key: float(value) for key, value in document["parameters"].items()
        } == expected_parameters
        compiled = compile_documents(
            template,
            document,
            template_sha256=_sha256(TEMPLATE_PATH),
        )
        assert compiled["compilation-request.json"]["candidate_sha256"] == _sha256(path)
