"""Tests for the Planner."""
import pytest

from vertex_live_dab_agent.planner.planner import Planner
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
