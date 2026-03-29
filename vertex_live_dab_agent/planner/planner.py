"""Planning layer: takes observations and produces the next single action.

The planner is the only component allowed to decide what the agent does next.
It accepts all available context (goal, OCR text, screenshot flag, current
app/screen, last actions, retry count) and returns exactly one
:class:`~vertex_live_dab_agent.planner.schemas.PlannedAction`.

Two modes are supported:

* **Heuristic mode** (default, no ``vertex_client``) — rule-based fallback
  that is fully deterministic and requires no external services. Suitable for
  CI and mock tests.
* **Vertex AI mode** — sends a structured prompt to a Gemini model and parses
  the JSON response, with heuristic fall-through on any failure.

Safe fallbacks are always enforced:
* Unclear screen → ``NEED_BETTER_VIEW``
* Repeated failures → ``FAILED``
* Goal reached → ``DONE``
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.planner.schemas import (
    ActionType,
    NavigationPlan,
    PlannedAction,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported actions exposed in the context so the model knows its vocabulary
# ---------------------------------------------------------------------------
_SUPPORTED_ACTIONS: List[str] = [a.value for a in ActionType]

# Keyword-to-canonical-name hints used in heuristic mode only.
# Values are *logical* app names / short IDs — NOT package names.
# Package-style IDs are intentionally avoided here; the runtime AppResolver
# queries DAB's applications/list to turn a name into the real app_id.
_APP_NAME_HINTS: Dict[str, str] = {
    "youtube": "youtube",
    "netflix": "netflix",
    "settings": "settings",
    "prime video": "prime video",
    "amazon prime": "prime video",
    "disney": "disney+",
    "disney plus": "disney+",
    "disney+": "disney+",
    "hulu": "hulu",
    "hbo": "hbo max",
    "hbo max": "hbo max",
    "max": "hbo max",
    "peacock": "peacock",
    "paramount": "paramount+",
    "apple tv": "apple tv",
}

# ---------------------------------------------------------------------------
# System prompt — navigation phases + batching
# ---------------------------------------------------------------------------
_DEFAULT_PLANNER_SYSTEM_PROMPT = """You are a TV UI navigation planner.
Always use current execution state and session history to choose the next grounded step.
Return exactly one JSON object that matches the NavigationPlan schema.
Never return markdown.
"""

_BUNDLED_PROMPT_RELATIVE_PATH = "vertex_live_dab_agent/prompts/planner_system_prompt.txt"

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """Plans next action based on current observations."""

    def __init__(self, vertex_client: Optional[Any] = None) -> None:
        """
        Initialize planner.

        Args:
            vertex_client: Optional Vertex AI client for AI-powered planning.
                          If None, uses deterministic heuristic planning.
        """
        self._vertex_client = vertex_client
        self._config = get_config()
        self._vertex_retry_after_ts: float = 0.0
        self._subplan_max_actions = max(0, int(self._config.planner_subplan_max_actions))
        self._last_nav_parse_error: Optional[str] = None
        self._nav_parse_error_count: int = 0
        self._planner_system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        override_text = str(getattr(self._config, "planner_prompt_override", "") or "").strip()
        if override_text:
            return override_text

        configured_path = str(getattr(self._config, "planner_prompt_path", "") or "").strip()
        candidate_paths: list[Path] = []
        if configured_path:
            candidate_paths.append(Path(configured_path).expanduser())

        repo_root = Path(__file__).resolve().parents[2]
        candidate_paths.append(repo_root / "prompts" / "planner_system_prompt.txt")
        candidate_paths.append(repo_root / _BUNDLED_PROMPT_RELATIVE_PATH)

        for path in candidate_paths:
            try:
                if path.exists() and path.is_file():
                    text = path.read_text(encoding="utf-8").strip()
                    if text:
                        return text
            except Exception as exc:
                logger.warning("Could not load planner prompt from %s: %s", path, exc)

        return _DEFAULT_PLANNER_SYSTEM_PROMPT

    def build_master_plan(self, goal: str) -> List[str]:
        g = (goal or "").lower()
        if self._is_settings_task(g):
            return [
                "launch_settings",
                "navigate_to_target_setting",
                "validate_destination_before_commit",
                "apply_and_verify",
            ]
        target = self._infer_direct_launch_app_id(g)
        if target:
            return ["launch_target_app", "checkpoint_after_launch", "verify_goal"]
        return ["observe", "navigate_in_batches", "checkpoint_and_verify", "recover_if_needed"]

    def _is_settings_task(self, goal: str) -> bool:
        g = (goal or "").lower()
        settings_keywords = (
            "settings",
            "time zone",
            "timezone",
            "date & time",
            "date and time",
            "wifi",
            "network",
            "bluetooth",
            "display",
            "sound",
            "privacy",
            "account",
            "language",
        )
        return any(k in g for k in settings_keywords)

    async def plan_navigation(
        self,
        goal: str,
        screenshot_b64: Optional[str] = None,
        ocr_text: Optional[str] = None,
        current_app: Optional[str] = None,
        current_screen: Optional[str] = None,
        last_actions: Optional[List[str]] = None,
        retry_count: int = 0,
        launch_content: Optional[str] = None,
        execution_state: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        master_plan: Optional[List[str]] = None,
    ) -> NavigationPlan:
        if retry_count >= self._config.max_steps_per_run:
            return NavigationPlan(
                phase="abort",
                intent="retry_limit_reached",
                execution_mode="UI_NAVIGATION_ONLY",
                confidence=1.0,
                starting_assumption="step budget exhausted",
                action_batch=[{"action": ActionType.FAILED.value}],
                checkpoint_required=False,
                validate_before_commit=False,
                expected_result="run terminated",
                fallback_if_failed=None,
                need_screenshot=False,
                done=True,
            )

        context = self._build_context(
            goal=goal,
            has_screenshot=screenshot_b64 is not None,
            ocr_text=ocr_text,
            current_app=current_app,
            current_screen=current_screen,
            last_actions=last_actions or [],
            retry_count=retry_count,
            execution_state=execution_state,
            master_plan=master_plan,
        )

        if self._vertex_client is not None:
            if time.monotonic() < self._vertex_retry_after_ts:
                wait_s = max(0.5, float(self._config.vertex_429_wait_seconds))
                return NavigationPlan(
                    phase="quota_wait",
                    intent="respect_rate_limit",
                    execution_mode="CONTINUE_IN_CURRENT_APP",
                    confidence=0.95,
                    starting_assumption="vertex recently rate limited",
                    action_batch=[{"action": ActionType.WAIT.value, "params": {"seconds": wait_s}}],
                    checkpoint_required=False,
                    validate_before_commit=False,
                    expected_result="retry later",
                    fallback_if_failed={"action": ActionType.PRESS_BACK.value, "params": {}},
                    need_screenshot=False,
                    done=False,
                )
            return await self._plan_navigation_with_vertex(
                context=context,
                screenshot_b64=screenshot_b64,
                goal=goal,
                last_actions=last_actions or [],
                retry_count=retry_count,
                current_app=current_app,
                launch_content=launch_content,
                session_id=session_id,
            )

        return self._plan_navigation_heuristic(
            goal=goal,
            has_screenshot=screenshot_b64 is not None,
            current_app=current_app,
            last_actions=last_actions or [],
            retry_count=retry_count,
            launch_content=launch_content,
        )

    async def plan(
        self,
        goal: str,
        screenshot_b64: Optional[str] = None,
        ocr_text: Optional[str] = None,
        current_app: Optional[str] = None,
        current_screen: Optional[str] = None,
        last_actions: Optional[List[str]] = None,
        retry_count: int = 0,
        launch_content: Optional[str] = None,
    ) -> PlannedAction:
        """Plan the next action.

        Args:
            goal: The testing goal to achieve.
            screenshot_b64: Base64 encoded screenshot if available.
            ocr_text: OCR text extracted from screenshot.
            current_app: Currently active app.
            current_screen: Current screen identifier.
            last_actions: Recent actions taken.
            retry_count: Number of retries so far.

        Returns:
            PlannedAction with action, confidence, reason, and optional params.
        """
        plan = await self.plan_navigation(
            goal=goal,
            screenshot_b64=screenshot_b64,
            ocr_text=ocr_text,
            current_app=current_app,
            current_screen=current_screen,
            last_actions=last_actions,
            retry_count=retry_count,
            launch_content=launch_content,
        )

        if plan.done and not plan.action_batch:
            return PlannedAction(action=ActionType.DONE, confidence=plan.confidence, reason=plan.expected_result or plan.intent)

        batch = plan.action_batch or [{"action": ActionType.CAPTURE_SCREENSHOT.value}]
        primary = batch[0]
        sub = [{"action": b.action, "params": b.params} for b in batch[1: self._subplan_max_actions + 1]]
        return PlannedAction(
            action=primary.action,
            confidence=plan.confidence,
            reason=f"{plan.phase}: {plan.intent}",
            params=primary.params,
            subplan=sub or None,
        )

    def _infer_direct_launch_app_id(self, goal: str) -> Optional[str]:
        """Infer direct app launch target from natural-language goal in heuristic mode.

        Returns the *logical* app id or canonical name.  Package-style IDs are
        never returned here — the runtime AppResolver resolves canonical names
        to real app IDs via DAB's applications/list.
        """
        g = (goal or "").strip().lower()
        if not g:
            return None

        launch_verbs = ("open", "launch", "start")
        if not any(v in g for v in launch_verbs):
            return None

        for name, canonical in _APP_NAME_HINTS.items():
            if name in g:
                if canonical == "youtube":
                    return self._config.youtube_app_id
                return canonical
        return None

    def _build_context(
        self,
        goal: str,
        has_screenshot: bool,
        ocr_text: Optional[str],
        current_app: Optional[str],
        current_screen: Optional[str],
        last_actions: List[str],
        retry_count: int,
        execution_state: Optional[Dict[str, Any]] = None,
        master_plan: Optional[List[str]] = None,
    ) -> str:
        """Build the context string sent to the planner model."""
        parts = [
            f"Goal: {goal}",
            f"Current app: {current_app or 'unknown'}",
            f"Current screen: {current_screen or 'unknown'}",
            f"Has screenshot: {has_screenshot}",
            f"Retry count: {retry_count}",
            f"Supported actions: {', '.join(_SUPPORTED_ACTIONS)}",
        ]
        if has_screenshot:
            parts.append("Use the attached screenshot as the primary visual context.")
        if last_actions:
            parts.append(f"Last actions: {', '.join(str(a) for a in last_actions[-5:])}")
        if master_plan:
            parts.append(f"Master plan phases: {', '.join(master_plan)}")
        if execution_state:
            session_history: dict[str, Any] = {}
            recent_ai = execution_state.get("recent_ai_events")
            if isinstance(recent_ai, list):
                session_history["recent_ai_events"] = recent_ai[-8:]
            recent_dab = execution_state.get("recent_dab_events")
            if isinstance(recent_dab, list):
                session_history["recent_dab_events"] = recent_dab[-6:]
            recent_action_records = execution_state.get("recent_action_records")
            if isinstance(recent_action_records, list):
                session_history["recent_action_records"] = recent_action_records[-8:]

            compact_state = dict(execution_state)
            compact_state.pop("recent_ai_events", None)
            compact_state.pop("recent_dab_events", None)
            compact_state.pop("recent_action_records", None)

            parts.append(f"Execution state: {json.dumps(compact_state, ensure_ascii=False)[:1200]}")
            if session_history:
                parts.append(f"Session history: {json.dumps(session_history, ensure_ascii=False)[:1400]}")
        return "\n".join(parts)

    async def _plan_navigation_with_vertex(
        self,
        context: str,
        screenshot_b64: Optional[str],
        goal: str,
        last_actions: List[str],
        retry_count: int,
        current_app: Optional[str] = None,
        launch_content: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> NavigationPlan:
        try:
            prompt = (
                f"{self._planner_system_prompt}\n\n"
                "Return JSON object with fields exactly as specified. "
                "Use action_batch as array of {action, params}.\n\n"
                f"Current situation:\n{context}"
            )
            try:
                response = await self._vertex_client.generate_content(
                    prompt,
                    screenshot_b64=screenshot_b64,
                    session_id=session_id,
                )
            except TypeError:
                try:
                    response = await self._vertex_client.generate_content(
                        prompt,
                        screenshot_b64=screenshot_b64,
                    )
                except TypeError:
                    response = await self._vertex_client.generate_content(prompt)
            return self._parse_navigation_plan(response)
        except Exception as exc:
            logger.error("Vertex AI planning failed: %s, falling back to heuristic", exc)
            msg = str(exc).lower()
            if (
                "publisher model" in msg
                and ("not found" in msg or "does not have access" in msg)
            ) or ("404" in msg and "model" in msg):
                # Hard-disable Vertex for this Planner instance to avoid per-step
                # retries/log spam when model id/region access is invalid.
                self._vertex_client = None
                logger.warning("Disabling Vertex planner for this process due to model 404/access error")
            if "429" in msg or "resource exhausted" in msg:
                cooldown_s = max(1.0, float(self._config.vertex_429_cooldown_seconds))
                wait_s = max(0.5, float(self._config.vertex_429_wait_seconds))
                self._vertex_retry_after_ts = time.monotonic() + cooldown_s
                return NavigationPlan(
                    phase="quota_wait",
                    intent="respect_rate_limit",
                    execution_mode="CONTINUE_IN_CURRENT_APP",
                    confidence=0.95,
                    starting_assumption="vertex quota exhausted",
                    action_batch=[{"action": ActionType.WAIT.value, "params": {"seconds": wait_s}}],
                    checkpoint_required=False,
                    validate_before_commit=False,
                    expected_result="retry after cooldown",
                        fallback_if_failed={"action": ActionType.PRESS_BACK.value, "params": {}},
                    need_screenshot=False,
                    done=False,
                )
            return self._plan_navigation_heuristic(
                goal=goal,
                has_screenshot=screenshot_b64 is not None,
                current_app=current_app,
                last_actions=last_actions,
                retry_count=retry_count,
                launch_content=launch_content,
            )

    def _plan_navigation_heuristic(
        self,
        goal: str,
        has_screenshot: bool,
        current_app: Optional[str],
        last_actions: List[str],
        retry_count: int,
        launch_content: Optional[str] = None,
    ) -> NavigationPlan:
        g = (goal or "").lower()

        if self._is_settings_task(g) and current_app != "settings" and not last_actions:
            return NavigationPlan(
                phase="launch_settings",
                intent="open settings directly",
                execution_mode="DIRECT_APP_LAUNCH",
                target_app_name="Settings",
                target_app_domain="system_settings",
                target_app_hint="settings",
                confidence=0.95,
                starting_assumption="settings task detected",
                action_batch=[
                    {"action": ActionType.LAUNCH_APP.value, "params": {"app_id": "settings"}},
                    {"action": ActionType.WAIT.value, "params": {"seconds": 1.0}},
                    {"action": ActionType.GET_STATE.value, "params": {"app_id": "settings"}},
                ],
                checkpoint_required=True,
                validate_before_commit=False,
                expected_result="settings app foreground",
                fallback_if_failed={"action": ActionType.PRESS_HOME.value, "params": {}},
                need_screenshot=True,
                done=False,
            )

        target_app_id = self._infer_direct_launch_app_id(g)
        if target_app_id and current_app == target_app_id and last_actions and last_actions[-1] == ActionType.GET_STATE.value:
            if self._requires_in_app_flow(g):
                follow_up_action = (
                    ActionType.NEED_VIDEO_PLAYBACK_CONFIRMED.value
                    if "youtube" in g and ("play" in g or "video" in g)
                    else ActionType.NEED_BETTER_VIEW.value
                )
                return NavigationPlan(
                    phase="in_app_continue",
                    intent="app is open but goal needs in-app steps",
                    execution_mode="CONTINUE_IN_CURRENT_APP",
                    target_app_name=target_app_id,
                    target_app_hint=target_app_id,
                    confidence=0.9,
                    starting_assumption="goal requires more than app foreground",
                    action_batch=[{"action": follow_up_action}],
                    checkpoint_required=True,
                    validate_before_commit=False,
                    expected_result="in-app workflow progresses",
                    fallback_if_failed={"action": ActionType.CAPTURE_SCREENSHOT.value, "params": {}},
                    need_screenshot=True,
                    done=False,
                )
            return NavigationPlan(
                phase="verify",
                intent="target app already active",
                execution_mode="CONTINUE_IN_CURRENT_APP",
                target_app_name=target_app_id,
                target_app_hint=target_app_id,
                confidence=0.95,
                starting_assumption="requested app is foreground",
                action_batch=[],
                checkpoint_required=False,
                validate_before_commit=False,
                expected_result="goal reached",
                fallback_if_failed={"action": ActionType.GET_STATE.value, "params": {}},
                need_screenshot=False,
                done=True,
            )

        if target_app_id and last_actions and last_actions[-1] == ActionType.LAUNCH_APP.value:
            return NavigationPlan(
                phase="verify_launch",
                intent="confirm launched app state",
                execution_mode="CONTINUE_IN_CURRENT_APP",
                target_app_name=target_app_id,
                target_app_hint=target_app_id,
                confidence=0.8,
                starting_assumption="launch request already sent",
                action_batch=[{"action": ActionType.GET_STATE.value, "params": {"app_id": target_app_id}}],
                checkpoint_required=False,
                validate_before_commit=False,
                expected_result="target app in foreground",
                fallback_if_failed={"action": ActionType.PRESS_HOME.value, "params": {}},
                need_screenshot=False,
                done=False,
            )

        if target_app_id and current_app != target_app_id:
            params: Dict[str, Any] = {"app_id": target_app_id}
            if launch_content and str(launch_content).strip():
                params["content"] = str(launch_content).strip()
            return NavigationPlan(
                phase="launch_target",
                intent="known target app",
                execution_mode=(
                    "DIRECT_APP_LAUNCH_WITH_PARAMS"
                    if launch_content and str(launch_content).strip()
                    else "DIRECT_APP_LAUNCH"
                ),
                target_app_name=target_app_id,
                target_app_domain="media",
                target_app_hint=target_app_id,
                launch_parameters={"content": str(launch_content).strip()} if launch_content and str(launch_content).strip() else {},
                confidence=0.94,
                starting_assumption="app id can be launched directly",
                action_batch=[
                    {"action": ActionType.LAUNCH_APP.value, "params": params},
                    {"action": ActionType.WAIT.value, "params": {"seconds": 1.0}},
                    {"action": ActionType.GET_STATE.value, "params": {"app_id": target_app_id}},
                ],
                checkpoint_required=True,
                validate_before_commit=False,
                expected_result="target app foreground",
                fallback_if_failed={"action": ActionType.PRESS_HOME.value, "params": {}},
                need_screenshot=True,
                done=False,
            )

        if not has_screenshot and not last_actions:
            return NavigationPlan(
                phase="checkpoint",
                intent="initial observation",
                execution_mode="UI_NAVIGATION_ONLY",
                confidence=0.9,
                starting_assumption="no screenshot available",
                action_batch=[{"action": ActionType.CAPTURE_SCREENSHOT.value}],
                checkpoint_required=False,
                validate_before_commit=False,
                expected_result="fresh visual context",
                fallback_if_failed={"action": ActionType.GET_STATE.value, "params": {}},
                need_screenshot=True,
                done=False,
            )

        if retry_count > 3:
            if not self._is_settings_task(g):
                return NavigationPlan(
                    phase="abort",
                    intent="multiple retries without progress",
                    execution_mode="UI_NAVIGATION_ONLY",
                    confidence=0.9,
                    starting_assumption="heuristic mode stuck",
                    action_batch=[{"action": ActionType.FAILED.value}],
                    checkpoint_required=False,
                    validate_before_commit=False,
                    expected_result="terminate run",
                    fallback_if_failed=None,
                    need_screenshot=False,
                    done=True,
                )
            return NavigationPlan(
                phase="recovery",
                intent="recover from off-path navigation",
                execution_mode="RECOVERY_RELAUNCH",
                target_app_name="Settings",
                target_app_domain="system_settings",
                target_app_hint="settings",
                confidence=0.85,
                starting_assumption="path likely incorrect",
                action_batch=[
                    {"action": ActionType.PRESS_BACK.value},
                    {"action": ActionType.PRESS_HOME.value},
                    {"action": ActionType.WAIT.value, "params": {"seconds": 0.8}},
                ],
                checkpoint_required=True,
                validate_before_commit=False,
                expected_result="return to stable launcher",
                fallback_if_failed={"action": ActionType.PRESS_HOME.value, "params": {}},
                need_screenshot=True,
                done=False,
            )

        # Backward-compatible fallback when heuristic context is still ambiguous.
        return NavigationPlan(
            phase="ambiguous",
            intent="need better view",
            execution_mode="UI_NAVIGATION_ONLY",
            confidence=0.7,
            starting_assumption="heuristic cannot safely infer destination",
            action_batch=[{"action": ActionType.NEED_BETTER_VIEW.value}],
            checkpoint_required=False,
            validate_before_commit=False,
            expected_result="request improved visual context",
            fallback_if_failed={"action": ActionType.CAPTURE_SCREENSHOT.value, "params": {}},
            need_screenshot=False,
            done=False,
        )

    def _requires_in_app_flow(self, goal: str) -> bool:
        g = (goal or "").lower()
        return any(
            token in g
            for token in (
                "play",
                "video",
                "stats for nerds",
                "gear icon",
                "player control",
                "player settings",
                "toggle",
                "enable",
            )
        )

    async def _plan_with_vertex(
        self,
        context: str,
        screenshot_b64: Optional[str],
        goal: str,
        last_actions: List[str],
        retry_count: int,
        current_app: Optional[str] = None,
        launch_content: Optional[str] = None,
    ) -> PlannedAction:
        """Use Vertex AI to plan next action."""
        try:
            prompt = f"{self._planner_system_prompt}\n\nCurrent situation:\n{context}"
            try:
                response = await self._vertex_client.generate_content(
                    prompt,
                    screenshot_b64=screenshot_b64,
                )
            except TypeError:
                # Backward compatibility for simple test clients.
                response = await self._vertex_client.generate_content(prompt)
            return self._parse_action(response)
        except Exception as exc:
            logger.error("Vertex AI planning failed: %s, falling back to heuristic", exc)
            msg = str(exc).lower()
            if (
                "publisher model" in msg
                and ("not found" in msg or "does not have access" in msg)
            ) or ("404" in msg and "model" in msg):
                self._vertex_client = None
                logger.warning("Disabling Vertex planner for this process due to model 404/access error")
            if "429" in msg or "resource exhausted" in msg:
                cooldown_s = max(1.0, float(self._config.vertex_429_cooldown_seconds))
                wait_s = max(0.5, float(self._config.vertex_429_wait_seconds))
                self._vertex_retry_after_ts = time.monotonic() + cooldown_s
                return PlannedAction(
                    action=ActionType.WAIT,
                    confidence=0.95,
                    reason=f"Vertex quota limit hit; waiting {wait_s:.1f}s before retry",
                    params={"seconds": wait_s},
                    subplan=[{"action": ActionType.CAPTURE_SCREENSHOT.value}],
                )
            return self._plan_heuristic(
                goal,
                last_actions,
                retry_count,
                current_app=current_app,
                launch_content=launch_content,
            )

    def _plan_heuristic(
        self,
        goal: str,
        last_actions: List[str],
        retry_count: int,
        current_app: Optional[str] = None,
        launch_content: Optional[str] = None,
    ) -> PlannedAction:
        """Deterministic heuristic planner for mock/testing mode.

        Decision priority:
        1. Too many retries → FAILED
        2. Direct app intent (e.g. open YouTube) → LAUNCH_APP / DONE / GET_STATE
        3. No prior context → CAPTURE_SCREENSHOT
        4. Last action was a screenshot → GET_STATE
        5. retry_count > 3 → FAILED (likely stuck)
        6. Default → NEED_BETTER_VIEW
        """
        if not last_actions:
            target_app_id = self._infer_direct_launch_app_id(goal)
            if not target_app_id and launch_content:
                target_app_id = self._config.youtube_app_id
            if target_app_id:
                params: Dict[str, Any] = {"app_id": target_app_id}
                if launch_content and str(launch_content).strip():
                    params["content"] = str(launch_content).strip()
                return PlannedAction(
                    action=ActionType.LAUNCH_APP,
                    confidence=0.95,
                    reason=f"Goal asks to open app directly - launching {target_app_id}",
                    params=params,
                    subplan=[{"action": ActionType.GET_STATE.value, "params": {"app_id": target_app_id}}],
                )
            return PlannedAction(
                action=ActionType.CAPTURE_SCREENSHOT,
                confidence=0.9,
                reason="No prior context - capturing screenshot to observe current state",
            )

        target_app_id = self._infer_direct_launch_app_id(goal)
        if not target_app_id and launch_content:
            target_app_id = self._config.youtube_app_id
        if target_app_id:
            if (
                current_app
                and current_app == target_app_id
                and last_actions
                and last_actions[-1] == ActionType.GET_STATE.value
            ):
                return PlannedAction(
                    action=ActionType.DONE,
                    confidence=0.95,
                    reason=f"Target app already active: {target_app_id}",
                )
            if last_actions and last_actions[-1] == ActionType.LAUNCH_APP.value:
                return PlannedAction(
                    action=ActionType.GET_STATE,
                    confidence=0.8,
                    reason="App launch sent - verifying app state",
                    params={"app_id": target_app_id},
                )
            params = {"app_id": target_app_id}
            if launch_content and str(launch_content).strip():
                params["content"] = str(launch_content).strip()
            return PlannedAction(
                action=ActionType.LAUNCH_APP,
                confidence=0.92,
                reason=f"Goal asks to open app directly - launching {target_app_id}",
                params=params,
            )

        # last_actions may contain either ActionType enum members or plain strings
        # (use_enum_values=True means PlannedAction stores strings; external callers
        # may pass enum members).  ActionType inherits from str so == works for both.
        if last_actions[-1] == ActionType.CAPTURE_SCREENSHOT.value:
            return PlannedAction(
                action=ActionType.GET_STATE,
                confidence=0.8,
                reason="Screenshot captured - getting app state",
            )
        if retry_count > 3:
            return PlannedAction(
                action=ActionType.FAILED,
                confidence=0.9,
                reason="Multiple retries without progress in heuristic mode",
            )
        return PlannedAction(
            action=ActionType.NEED_BETTER_VIEW,
            confidence=0.7,
            reason="Heuristic planner cannot determine next action without Vertex AI",
        )

    def _parse_navigation_plan(self, response_text: str) -> NavigationPlan:
        """Parse structured navigation plan JSON from model response."""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                if len(lines) >= 3 and lines[-1].strip() == "```":
                    text = "\n".join(lines[1:-1])
                else:
                    text = "\n".join(lines[1:])
            data = json.loads(text)

            if not isinstance(data, dict):
                raise ValueError("planner response must be a JSON object")

            raw_batch = data.get("action_batch")
            cleaned_batch: List[Dict[str, Any]] = []
            if isinstance(raw_batch, list):
                for item in raw_batch:
                    if not isinstance(item, dict):
                        continue
                    action = str(item.get("action", "")).strip()
                    if not action:
                        continue
                    cleaned_item: Dict[str, Any] = {"action": action}
                    if isinstance(item.get("params"), dict):
                        cleaned_item["params"] = item.get("params")
                    cleaned_batch.append(cleaned_item)
                    if len(cleaned_batch) >= self._subplan_max_actions + 1:
                        break
            data["action_batch"] = cleaned_batch

            if data.get("done") and not cleaned_batch:
                data["action_batch"] = []

            mode = data.get("execution_mode")
            if isinstance(mode, str):
                data["execution_mode"] = mode.strip().upper()
            if not isinstance(data.get("launch_parameters"), dict):
                data["launch_parameters"] = {}

            plan = NavigationPlan.model_validate(data)
            self._last_nav_parse_error = None
            self._nav_parse_error_count = 0
            return plan
        except Exception as exc:
            err = str(exc)
            if self._last_nav_parse_error == err:
                self._nav_parse_error_count += 1
            else:
                self._last_nav_parse_error = err
                self._nav_parse_error_count = 1
            logger.error(
                "Failed to parse navigation plan (count=%d): %s | response: %s",
                self._nav_parse_error_count,
                err,
                response_text,
            )
            if self._nav_parse_error_count >= 3:
                return NavigationPlan(
                    phase="fallback",
                    intent="parse_failure_limit_reached",
                    confidence=0.7,
                    starting_assumption="repeated invalid model output",
                    action_batch=[{"action": ActionType.FAILED.value}],
                    checkpoint_required=False,
                    validate_before_commit=False,
                    expected_result="stop repeated parse failures",
                    fallback_if_failed=None,
                    need_screenshot=False,
                    done=True,
                )
            return NavigationPlan(
                phase="fallback",
                intent="parse_failure",
                confidence=0.6,
                starting_assumption="invalid model output",
                action_batch=[{"action": ActionType.GET_STATE.value}],
                checkpoint_required=False,
                validate_before_commit=False,
                expected_result="recover with deterministic local fallback",
                fallback_if_failed={"action": ActionType.PRESS_HOME.value, "params": {}},
                need_screenshot=False,
                done=False,
            )

    def _validate_action(self, action: PlannedAction) -> PlannedAction:
        """Post-parse validation: re-validate via Pydantic to catch constraint violations.

        Returns the action unchanged if valid, or a FAILED action if the
        parsed result violates any schema constraint. This is a second safety
        net on top of the initial Pydantic parse so that partial JSON objects
        assembled in ``_parse_action`` are always fully validated.
        """
        try:
            # Re-validate by round-tripping through the model
            validated = PlannedAction.model_validate(action.model_dump())
            return validated
        except (ValidationError, Exception) as exc:
            logger.error("Planner action failed validation: %s", exc)
            return PlannedAction(
                action=ActionType.FAILED,
                confidence=0.5,
                reason=f"Action failed validation: {exc}",
            )

    def _parse_action(self, response_text: str) -> PlannedAction:
        """Parse and validate action JSON from model response.

        Handles raw JSON as well as JSON wrapped in markdown code fences
        (e.g. ```json ... ```). After parsing, the result is passed through
        :meth:`_validate_action` to catch constraint violations.
        """
        try:
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                # First line should be ``` or ```json -- skip it
                fence_label = lines[0].strip().lstrip("`").lower()
                if fence_label not in ("", "json"):
                    logger.warning("Unexpected opening fence: %s", lines[0])
                # Strip fences only when the closing fence is present
                if len(lines) >= 3 and lines[-1].strip() == "```":
                    text = "\n".join(lines[1:-1])
                else:
                    text = "\n".join(lines[1:])
            data = json.loads(text)

            if isinstance(data, dict):
                raw_subplan = data.get("subplan")
                if isinstance(raw_subplan, list):
                    cleaned = []
                    for item in raw_subplan:
                        if not isinstance(item, dict):
                            continue
                        a = str(item.get("action", "")).strip()
                        if not a or a in {ActionType.DONE.value, ActionType.FAILED.value}:
                            continue
                        cleaned_item: Dict[str, Any] = {"action": a}
                        if isinstance(item.get("params"), dict):
                            cleaned_item["params"] = item.get("params")
                        cleaned.append(cleaned_item)
                        if len(cleaned) >= self._subplan_max_actions:
                            break
                    data["subplan"] = cleaned

            # Normalize known app package aliases to logical app IDs.
            # Only the youtube_app_id is normalized here because it comes from
            # config; all other aliases are handled by AppResolver at runtime.
            if isinstance(data, dict) and data.get("action") == ActionType.LAUNCH_APP.value:
                params = data.get("params")
                if isinstance(params, dict):
                    raw_app_id = str(params.get("app_id", "")).strip().lower()
                    if raw_app_id and (
                        raw_app_id == self._config.youtube_app_id.lower()
                        or "youtube" in raw_app_id
                    ):
                        params["app_id"] = self._config.youtube_app_id

            action = PlannedAction(**data)
            return self._validate_action(action)
        except (ValidationError, Exception) as exc:
            logger.error("Failed to parse planner response: %s | response: %s", exc, response_text)
            return PlannedAction(
                action=ActionType.FAILED,
                confidence=0.5,
                reason=f"Failed to parse planner response: {exc}",
            )
