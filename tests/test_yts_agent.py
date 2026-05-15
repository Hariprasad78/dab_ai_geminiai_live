"""Tests for the runtime YTS validation agent."""

from pathlib import Path

from vertex_live_dab_agent.yts_agent.decision_engine import decide_yts_response
from vertex_live_dab_agent.yts_agent.expectation_extractor import heuristic_extract_expectation
from vertex_live_dab_agent.yts_agent.memory_store import YtsMemoryStore
from vertex_live_dab_agent.yts_agent.prompt_parser import parse_yts_prompt
from vertex_live_dab_agent.yts_agent.prompt_parser import classify_yts_prompt_kind
from vertex_live_dab_agent.yts_agent.utils import normalize_missing_evidence
from vertex_live_dab_agent.yts_agent.validation_gate import validate_decision_gate
from vertex_live_dab_agent.yts_agent.visual_evidence import build_visual_evidence


def _playback_expectation():
    prompt = parse_yts_prompt(
        "In-app playback validation. Does the visible video match the YTS requirement?\n1: Pass\n2: Fail\n3: Skip",
        ["1", "2", "3"],
    )
    return heuristic_extract_expectation(prompt)


def test_prompt_expectation_extraction_from_generic_yts_prompt():
    prompt = parse_yts_prompt(
        "In-app video playback validation. Confirm the rendered video matches the expected behavior.\n1: Pass\n2: Fail\n3: Skip",
        ["1", "2", "3"],
    )
    expectation = heuristic_extract_expectation(prompt)

    assert expectation["required_app_context"] == "YouTube"
    assert expectation["required_state"] == "video_playback_active"
    assert expectation["allowed_answers"][0]["label"] == "pass"
    assert "positive proof" in expectation["minimum_evidence_policy"].lower()


def test_prompt_expectation_extracts_parenthetical_visual_target_generically():
    prompt = parse_yts_prompt(
        "Does the image on screen render correctly? (Sample Asset WebP)\n1: Yes\n2: No",
        ["1", "2"],
    )
    expectation = heuristic_extract_expectation(prompt)

    assert expectation["expected_visual_target"] == "Sample Asset WebP"


def test_yes_blocked_when_fresh_frame_observed_target_mismatches_expected_asset():
    prompt = parse_yts_prompt(
        "Does the image on screen render correctly? (Expected Asset)\n1: Yes\n2: No",
        ["1", "2"],
    )
    expectation = heuristic_extract_expectation(prompt)
    evidence = build_visual_evidence(
        {
            "prompt_id": 7,
            "prompt_sequence_id": 7,
            "prompt_timestamp": "2026-05-13T10:00:00+00:00",
            "fresh_after_prompt": True,
            "expected_visual_target": "Expected Asset",
            "captured_at": "2026-05-13T10:00:01+00:00",
            "source": "hdmi-capture",
            "summary": "A different asset is visible.",
            "analysis": {
                "summary": "Observed Previous Asset.",
                "confidence": 0.94,
                "requirement_seen": True,
                "recommended_result": "pass",
                "expected_visual_target": "Expected Asset",
                "observed_visual_target": "Previous Asset",
                "target_match": False,
                "prompt_requirement_match": "mismatch",
            },
        }
    )
    gate = validate_decision_gate(expectation, evidence, {"selected_option": "1", "selected_label": "Yes", "confidence": 0.94})

    assert gate["safety_blocked_pass"] is True
    assert "Expected Asset" in " ".join(gate["missing_evidence"])
    assert "Previous Asset" in " ".join(gate["missing_evidence"])


def test_yes_allowed_only_with_prompt_fresh_matching_asset_evidence():
    prompt = parse_yts_prompt(
        "Does the image on screen render correctly? (Expected Asset)\n1: Yes\n2: No",
        ["1", "2"],
    )
    expectation = heuristic_extract_expectation(prompt)
    evidence = build_visual_evidence(
        {
            "prompt_id": 8,
            "prompt_sequence_id": 8,
            "prompt_timestamp": "2026-05-13T10:00:00+00:00",
            "fresh_after_prompt": True,
            "expected_visual_target": "Expected Asset",
            "captured_at": "2026-05-13T10:00:01+00:00",
            "source": "hdmi-capture",
            "summary": "Expected Asset is visible and rendered correctly.",
            "analysis": {
                "summary": "Expected Asset is visible and rendered correctly.",
                "confidence": 0.94,
                "requirement_seen": True,
                "recommended_result": "pass",
                "expected_visual_target": "Expected Asset",
                "observed_visual_target": "Expected Asset",
                "target_match": True,
                "prompt_requirement_match": "match",
            },
        }
    )
    gate = validate_decision_gate(expectation, evidence, {"selected_option": "1", "selected_label": "Yes", "confidence": 0.94})

    assert gate["allowed"] is True
    assert gate["safety_blocked_pass"] is False


def test_yes_blocked_when_frame_timestamp_is_not_after_prompt_timestamp():
    prompt = parse_yts_prompt(
        "Does the image on screen render correctly? (Expected Asset)\n1: Yes\n2: No",
        ["1", "2"],
    )
    expectation = heuristic_extract_expectation(prompt)
    evidence = build_visual_evidence(
        {
            "prompt_timestamp": "2026-05-13T10:00:00+00:00",
            "fresh_after_prompt": False,
            "expected_visual_target": "Expected Asset",
            "captured_at": "2026-05-13T09:59:59+00:00",
            "source": "hdmi-capture",
            "analysis": {
                "summary": "Expected Asset is visible.",
                "confidence": 0.94,
                "requirement_seen": True,
                "expected_visual_target": "Expected Asset",
                "observed_visual_target": "Expected Asset",
                "target_match": True,
            },
        }
    )
    gate = validate_decision_gate(expectation, evidence, {"selected_option": "1", "selected_label": "Yes", "confidence": 0.94})

    assert gate["safety_blocked_pass"] is True
    assert "after the current prompt" in " ".join(gate["missing_evidence"])


def test_pass_blocked_when_current_screen_is_outside_required_app_context():
    expectation = _playback_expectation()
    evidence = build_visual_evidence(
        {
            "timeline": [
                {
                    "captured_at": "2026-05-07T00:00:00+00:00",
                    "source": "hdmi-capture",
                    "summary": "Device home screen launcher with app tiles is visible.",
                    "confidence": 0.9,
                    "recommended_result": "pass",
                }
            ],
            "continuous_frame_count": 12,
        }
    )
    decision = {"selected_option": "1", "selected_label": "Pass", "confidence": 0.9}
    gate = validate_decision_gate(expectation, evidence, decision)

    assert gate["safety_blocked_pass"] is True
    assert "YouTube" in " ".join(gate["missing_evidence"])


def test_pass_blocked_when_playback_test_has_no_active_video():
    expectation = _playback_expectation()
    evidence = build_visual_evidence(
        {
            "timeline": [
                {
                    "captured_at": "2026-05-07T00:00:00+00:00",
                    "source": "hdmi-capture",
                    "summary": "YouTube app menu is visible but no video playback surface is active.",
                    "analysis": {
                        "detected_app_context": "YouTube",
                        "screen_type": "menu",
                        "youtube_active": True,
                        "video_playback_active": False,
                        "confidence": 0.86,
                    },
                }
            ],
            "continuous_frame_count": 12,
        }
    )
    decision = {"selected_option": "1", "selected_label": "Pass", "confidence": 0.86}
    gate = validate_decision_gate(expectation, evidence, decision)

    assert gate["safety_blocked_pass"] is True
    assert "playback" in " ".join(gate["missing_evidence"]).lower()


def test_memory_created_for_every_run(tmp_path: Path):
    store = YtsMemoryStore(tmp_path / "yts_memory")
    store.start_run("run-1", test_name="Example", command="yts test adt-4 Example")

    assert (tmp_path / "yts_memory" / "runs" / "run-1" / "memory.json").exists()
    assert (tmp_path / "yts_memory" / "index.json").exists()


def test_same_test_second_run_loads_previous_expectation_memory(tmp_path: Path):
    store = YtsMemoryStore(tmp_path / "yts_memory")
    prompt = parse_yts_prompt("Validate video playback.\n1: Pass\n2: Fail", ["1", "2"])
    expectation = heuristic_extract_expectation(prompt)
    store.start_run("run-1")
    store.record_prompt(
        run_id="run-1",
        test_key="same-test",
        test_title="Same Test",
        prompt_hash=prompt["prompt_hash"],
        prompt_text=prompt["prompt_text"],
        options=prompt["allowed_answers"],
        extracted_expectation=expectation,
    )

    memory = store.load_test_memory("same-test")

    assert memory["test_key"] == "same-test"
    assert memory["prompt_history"][0]["extracted_expectation"]["required_state"] == "video_playback_active"


def test_memory_does_not_override_current_live_evidence(tmp_path: Path):
    store = YtsMemoryStore(tmp_path / "yts_memory")
    expectation = _playback_expectation()
    store.start_run("run-1")
    store.record_decision(
        run_id="run-1",
        test_key="playback-test",
        decision={"selected_option": "1", "selected_label": "Pass", "confidence": 0.95, "reason": "previous pass"},
    )
    evidence = build_visual_evidence(
        {
            "timeline": [
                {
                    "summary": "Launcher home screen visible.",
                    "confidence": 0.95,
                    "recommended_result": "pass",
                }
            ],
            "continuous_frame_count": 12,
        }
    )

    decision = decide_yts_response(expectation, evidence, {"selected_option": "1", "selected_label": "Pass", "confidence": 0.95})

    assert decision["selected_label"] == "fail"
    assert decision["selected_option"] == "2"


def test_prompt_text_change_creates_new_prompt_hash():
    first = parse_yts_prompt("Validate playback.\n1: Pass\n2: Fail", ["1", "2"])
    second = parse_yts_prompt("Validate playback after seeking.\n1: Pass\n2: Fail", ["1", "2"])

    assert first["prompt_hash"] != second["prompt_hash"]


def test_agent_does_not_choose_pass_when_evidence_is_insufficient():
    expectation = _playback_expectation()
    evidence = build_visual_evidence(
        {
            "timeline": [
                {
                    "summary": "A dark screen is visible; app context and playback are unclear.",
                    "confidence": 0.2,
                }
            ],
            "continuous_frame_count": 12,
        }
    )

    decision = decide_yts_response(expectation, evidence, {"selected_option": "1", "selected_label": "Pass", "confidence": 0.2})

    assert decision["selected_option"] == "2"
    assert decision["safety_blocked_pass"] is True


def test_missing_evidence_string_is_not_split_into_characters():
    assert normalize_missing_evidence("Confirm playback is visible") == ["Confirm playback is visible"]


def test_option_one_done_is_not_treated_as_pass():
    prompt = parse_yts_prompt("Previous Selection: FAILED\n1: Done\n2: Retry", ["1", "2"])
    expectation = heuristic_extract_expectation(prompt)
    evidence = build_visual_evidence({"summary": "No visual validation required.", "continuous_frame_count": 0})
    decision = {"selected_option": "1", "selected_label": "Pass", "missing_evidence": "Confirm result prompt"}

    gate = validate_decision_gate(expectation, evidence, decision)

    assert gate["safety_blocked_pass"] is False
    assert gate["missing_evidence"] == ["Confirm result prompt"]
    assert "done" in gate["reason"].lower()


def test_retry_prompt_is_classified_separately():
    prompt = parse_yts_prompt("FAILED - Marked by user\nPrevious Selection: 2\n1: Done\n2: Retry", ["1", "2"])

    assert prompt["prompt_kind"] == "result_retry_control"
    assert classify_yts_prompt_kind(prompt["prompt_text"]) == "result_retry_control"
    expectation = heuristic_extract_expectation(prompt)
    assert expectation["test_type"] == "control"
    assert expectation["visual_requirements"][0]["evidence_required"] is False
