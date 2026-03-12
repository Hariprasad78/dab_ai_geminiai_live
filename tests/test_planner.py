"""Tests for the Planner."""
import pytest
from pydantic import ValidationError

from vertex_live_dab_agent.planner.planner import Planner, _SUPPORTED_ACTIONS
from vertex_live_dab_agent.planner.schemas import ActionType, PlannedAction


@pytest.fixture
def planner():
    return Planner()


@pytest.mark.asyncio
async def test_planner_no_last_actions(planner):
    """With no prior actions, planner should capture screenshot first."""
    result = await planner.plan(goal="Launch Netflix", last_actions=[])
    assert result.action == ActionType.CAPTURE_SCREENSHOT
    assert result.confidence > 0


@pytest.mark.asyncio
async def test_planner_after_screenshot(planner):
    """After a screenshot, planner should get app state."""
    result = await planner.plan(
        goal="Launch Netflix",
        last_actions=[ActionType.CAPTURE_SCREENSHOT],
    )
    assert result.action == ActionType.GET_STATE


@pytest.mark.asyncio
async def test_planner_max_retries(planner):
    """Planner should return FAILED when retry count hits max steps."""
    from vertex_live_dab_agent.config import get_config
    config = get_config()
    result = await planner.plan(
        goal="Launch Netflix",
        last_actions=["PRESS_OK"] * 5,
        retry_count=config.max_steps_per_run,
    )
    assert result.action == ActionType.FAILED


@pytest.mark.asyncio
async def test_planner_heuristic_after_many_retries(planner):
    """Heuristic planner returns FAILED after >3 retries."""
    result = await planner.plan(
        goal="Launch Netflix",
        last_actions=["PRESS_OK", "GET_STATE", "PRESS_OK"],
        retry_count=4,
    )
    assert result.action == ActionType.FAILED


@pytest.mark.asyncio
async def test_planner_need_better_view(planner):
    """With some actions but not too many retries, get NEED_BETTER_VIEW."""
    result = await planner.plan(
        goal="Find settings",
        last_actions=["PRESS_OK", "GET_STATE"],
        retry_count=1,
    )
    assert result.action == ActionType.NEED_BETTER_VIEW


def test_planner_parse_valid_json(planner):
    """Test that valid JSON is parsed correctly."""
    response = '{"action": "PRESS_UP", "confidence": 0.85, "reason": "Navigate up"}'
    result = planner._parse_action(response)
    assert result.action == "PRESS_UP"
    assert result.confidence == 0.85


def test_planner_parse_markdown_json(planner):
    """Test that JSON wrapped in markdown fences is parsed."""
    response = '```json\n{"action": "PRESS_DOWN", "confidence": 0.7, "reason": "Navigate down"}\n```'
    result = planner._parse_action(response)
    assert result.action == "PRESS_DOWN"


def test_planner_parse_invalid_json(planner):
    """Test that invalid JSON returns FAILED action."""
    result = planner._parse_action("not valid json at all")
    assert result.action == ActionType.FAILED


def test_planner_build_context(planner):
    """Test context builder includes all fields."""
    context = planner._build_context(
        goal="Launch Netflix",
        has_screenshot=True,
        ocr_text="Home screen",
        current_app="launcher",
        current_screen="HOME",
        last_actions=["PRESS_OK"],
        retry_count=2,
    )
    assert "Launch Netflix" in context
    assert "launcher" in context
    assert "HOME" in context
    assert "Home screen" in context
    assert "PRESS_OK" in context
    assert "2" in context
    assert "Has screenshot: True" in context
    assert "Supported actions:" in context


# ---------------------------------------------------------------------------
# New tests: context fields
# ---------------------------------------------------------------------------


def test_planner_context_includes_supported_actions(planner):
    """Context must list all supported action strings."""
    context = planner._build_context(
        goal="test",
        has_screenshot=False,
        ocr_text=None,
        current_app=None,
        current_screen=None,
        last_actions=[],
        retry_count=0,
    )
    assert "Supported actions:" in context
    for action in _SUPPORTED_ACTIONS:
        assert action in context


def test_planner_context_includes_screenshot_reference(planner):
    """Context must indicate whether a screenshot is available."""
    ctx_with = planner._build_context(
        goal="test", has_screenshot=True, ocr_text=None,
        current_app=None, current_screen=None, last_actions=[], retry_count=0,
    )
    ctx_without = planner._build_context(
        goal="test", has_screenshot=False, ocr_text=None,
        current_app=None, current_screen=None, last_actions=[], retry_count=0,
    )
    assert "Has screenshot: True" in ctx_with
    assert "Has screenshot: False" in ctx_without


def test_planner_context_truncates_ocr_at_500_chars(planner):
    """OCR text longer than 500 chars must be truncated."""
    long_text = "A" * 600
    context = planner._build_context(
        goal="test", has_screenshot=False, ocr_text=long_text,
        current_app=None, current_screen=None, last_actions=[], retry_count=0,
    )
    assert "A" * 500 in context
    assert "A" * 501 not in context


def test_planner_context_limits_last_actions_to_5(planner):
    """Context should only include the last 5 actions."""
    actions = [f"PRESS_{i}" for i in range(10)]
    context = planner._build_context(
        goal="test", has_screenshot=False, ocr_text=None,
        current_app=None, current_screen=None, last_actions=actions, retry_count=0,
    )
    # The 5th-from-last should be present, but an earlier one should not
    assert "PRESS_9" in context
    assert "PRESS_0" not in context


# ---------------------------------------------------------------------------
# New tests: schema validation
# ---------------------------------------------------------------------------


def test_schema_rejects_empty_reason():
    """PlannedAction must reject empty reason strings."""
    with pytest.raises(ValidationError, match="reason must not be empty"):
        PlannedAction(action=ActionType.DONE, confidence=1.0, reason="")


def test_schema_rejects_whitespace_only_reason():
    """PlannedAction must reject whitespace-only reason."""
    with pytest.raises(ValidationError, match="reason must not be empty"):
        PlannedAction(action=ActionType.DONE, confidence=1.0, reason="   ")


def test_schema_rejects_launch_app_without_app_id():
    """LAUNCH_APP requires params['app_id']."""
    with pytest.raises(ValidationError, match="app_id"):
        PlannedAction(
            action=ActionType.LAUNCH_APP, confidence=0.9, reason="Launching",
            params={},
        )


def test_schema_rejects_launch_app_with_empty_app_id():
    """LAUNCH_APP rejects empty app_id."""
    with pytest.raises(ValidationError, match="app_id"):
        PlannedAction(
            action=ActionType.LAUNCH_APP, confidence=0.9, reason="Launching",
            params={"app_id": ""},
        )


def test_schema_rejects_wait_without_seconds():
    """WAIT requires params['seconds']."""
    with pytest.raises(ValidationError, match="seconds"):
        PlannedAction(
            action=ActionType.WAIT, confidence=0.8, reason="Waiting",
            params={},
        )


def test_schema_accepts_launch_app_with_valid_app_id():
    """LAUNCH_APP accepts a valid app_id."""
    action = PlannedAction(
        action=ActionType.LAUNCH_APP, confidence=0.9, reason="Launch Netflix",
        params={"app_id": "com.netflix.ninja"},
    )
    assert action.params["app_id"] == "com.netflix.ninja"


def test_schema_accepts_wait_with_seconds():
    """WAIT accepts params with seconds."""
    action = PlannedAction(
        action=ActionType.WAIT, confidence=0.8, reason="Loading",
        params={"seconds": 5},
    )
    assert action.params["seconds"] == 5


# ---------------------------------------------------------------------------
# New tests: _validate_action and _parse_action edge cases
# ---------------------------------------------------------------------------


def test_planner_validate_action_passes_valid(planner):
    """_validate_action must return the same action when valid."""
    action = PlannedAction(action=ActionType.PRESS_OK, confidence=0.9, reason="OK")
    result = planner._validate_action(action)
    assert result.action == ActionType.PRESS_OK.value
    assert result.confidence == 0.9


def test_planner_parse_unknown_action_returns_failed(planner):
    """JSON with an action not in ActionType must return FAILED."""
    response = '{"action": "HACK_DEVICE", "confidence": 0.99, "reason": "Attack"}'
    result = planner._parse_action(response)
    assert result.action == ActionType.FAILED


def test_planner_parse_missing_reason_returns_failed(planner):
    """JSON missing required 'reason' field must return FAILED."""
    response = '{"action": "PRESS_OK", "confidence": 0.8}'
    result = planner._parse_action(response)
    assert result.action == ActionType.FAILED


def test_planner_parse_empty_reason_returns_failed(planner):
    """JSON with empty reason must return FAILED."""
    response = '{"action": "PRESS_OK", "confidence": 0.8, "reason": ""}'
    result = planner._parse_action(response)
    assert result.action == ActionType.FAILED


def test_planner_parse_confidence_out_of_bounds_returns_failed(planner):
    """JSON with out-of-bounds confidence must return FAILED."""
    response = '{"action": "PRESS_OK", "confidence": 1.5, "reason": "too high"}'
    result = planner._parse_action(response)
    assert result.action == ActionType.FAILED


def test_planner_parse_launch_app_without_app_id_returns_failed(planner):
    """JSON with LAUNCH_APP but no app_id must return FAILED."""
    response = '{"action": "LAUNCH_APP", "confidence": 0.9, "reason": "launch", "params": {}}'
    result = planner._parse_action(response)
    assert result.action == ActionType.FAILED


def test_planner_parse_launch_app_with_app_id_succeeds(planner):
    """JSON with LAUNCH_APP and valid app_id must parse correctly."""
    response = (
        '{"action": "LAUNCH_APP", "confidence": 0.95, "reason": "open app",'
        ' "params": {"app_id": "com.netflix.ninja"}}'
    )
    result = planner._parse_action(response)
    assert result.action == ActionType.LAUNCH_APP.value
    assert result.params["app_id"] == "com.netflix.ninja"


def test_planner_parse_done_action(planner):
    """JSON with DONE action must parse cleanly."""
    response = '{"action": "DONE", "confidence": 1.0, "reason": "Goal achieved"}'
    result = planner._parse_action(response)
    assert result.action == ActionType.DONE.value
    assert result.confidence == 1.0


def test_planner_parse_need_better_view(planner):
    """JSON with NEED_BETTER_VIEW must parse cleanly."""
    response = '{"action": "NEED_BETTER_VIEW", "confidence": 0.6, "reason": "Screen unclear"}'
    result = planner._parse_action(response)
    assert result.action == ActionType.NEED_BETTER_VIEW.value


def test_planner_parse_plain_backtick_fence(planner):
    """Plain triple-backtick fences (no 'json' label) should be handled."""
    response = '```\n{"action": "PRESS_HOME", "confidence": 0.8, "reason": "Go home"}\n```'
    result = planner._parse_action(response)
    assert result.action == ActionType.PRESS_HOME.value


@pytest.mark.asyncio
async def test_planner_plan_returns_planned_action_type(planner):
    """plan() must always return a PlannedAction instance."""
    result = await planner.plan(goal="Navigate to settings")
    assert isinstance(result, PlannedAction)
    assert result.reason  # non-empty
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_planner_plan_confidence_in_bounds(planner):
    """Confidence must always be in [0.0, 1.0]."""
    for retry in range(6):
        result = await planner.plan(
            goal="test", last_actions=["PRESS_OK"], retry_count=retry,
        )
        assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_planner_plan_screenshot_b64_is_optional(planner):
    """Planner must work with and without screenshot_b64."""
    r1 = await planner.plan(goal="test", screenshot_b64=None)
    r2 = await planner.plan(goal="test", screenshot_b64="fake_b64_data")
    assert isinstance(r1, PlannedAction)
    assert isinstance(r2, PlannedAction)

