"""Main orchestration loop: observe -> plan -> act -> verify -> repeat."""
import asyncio
import logging
from typing import Optional

from vertex_live_dab_agent.artifacts.logger import ArtifactStore
from vertex_live_dab_agent.capture.capture import ScreenCapture
from vertex_live_dab_agent.capture.validator import Validator
from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.dab.client import DABClientBase, create_dab_client
from vertex_live_dab_agent.dab.topics import KEY_MAP
from vertex_live_dab_agent.orchestrator.run_state import RunState, RunStatus
from vertex_live_dab_agent.planner.planner import Planner
from vertex_live_dab_agent.planner.schemas import ActionType, PlannedAction

logger = logging.getLogger(__name__)


class Orchestrator:
    """Runs the observe-plan-act-verify loop and saves artifacts for every run."""

    def __init__(
        self,
        dab_client: Optional[DABClientBase] = None,
        planner: Optional[Planner] = None,
        capture: Optional[ScreenCapture] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        self._config = get_config()
        self._dab = dab_client or create_dab_client()
        self._planner = planner or Planner()
        self._capture = capture or ScreenCapture(self._dab)
        self._validator = Validator()
        self._max_steps = max_steps if max_steps is not None else self._config.max_steps_per_run

    async def run(self, state: RunState) -> RunState:
        """Execute the full run loop, saving artifacts throughout."""
        state.start()
        store = ArtifactStore(state.run_id)
        state.artifacts_dir = str(store.run_dir)

        store.save_metadata({
            "run_id": state.run_id,
            "goal": state.goal,
            "started_at": state.started_at,
            "max_steps": self._max_steps,
            "mock_mode": self._config.dab_mock_mode,
        })
        logger.info("Run started: run_id=%s goal=%r", state.run_id, state.goal)

        try:
            while state.status == RunStatus.RUNNING:
                if state.step_count >= self._max_steps:
                    state.finish(RunStatus.TIMEOUT, "Max steps exceeded")
                    break
                await self._step(state, store)
        except asyncio.CancelledError:
            state.finish(RunStatus.STOPPED, "Run cancelled")
            logger.info("Run cancelled: run_id=%s", state.run_id)
        except Exception as exc:
            state.finish(RunStatus.ERROR, str(exc))
            logger.error("Run error: run_id=%s error=%s", state.run_id, exc, exc_info=True)
        finally:
            store.save_final_summary({
                "run_id": state.run_id,
                "goal": state.goal,
                "status": state.status.value,
                "step_count": state.step_count,
                "retries": state.retries,
                "started_at": state.started_at,
                "finished_at": state.finished_at,
                "error": state.error,
            })
            logger.info("Run finished: run_id=%s status=%s steps=%d", state.run_id, state.status, state.step_count)

        return state

    async def _step(self, state: RunState, store: ArtifactStore) -> None:
        """Execute a single observe-plan-act-verify step."""
        step = state.step_count

        # Observe
        capture_result = await self._capture.capture()
        state.latest_screenshot_b64 = capture_result.image_b64
        state.latest_ocr_text = capture_result.ocr_text
        if capture_result.image_b64:
            store.save_screenshot(capture_result.image_b64, step)

        # Plan
        planned = await self._planner.plan(
            goal=state.goal,
            screenshot_b64=state.latest_screenshot_b64,
            ocr_text=state.latest_ocr_text,
            current_app=state.current_app,
            current_screen=state.current_screen,
            last_actions=state.last_actions,
            retry_count=state.retries,
        )
        logger.info(
            "Planner decision: run_id=%s step=%d action=%s confidence=%.2f reason=%r",
            state.run_id, step, planned.action, planned.confidence, planned.reason,
        )
        store.save_planner_trace(
            {
                "step": step,
                "goal": state.goal,
                "current_app": state.current_app,
                "current_screen": state.current_screen,
                "ocr_text": state.latest_ocr_text,
                "last_actions": state.last_actions[-5:],
                "retry_count": state.retries,
                "planned_action": planned.action,
                "confidence": planned.confidence,
                "reason": planned.reason,
                "params": planned.params,
            },
            step,
        )

        # Handle terminal actions before incrementing step_count
        if planned.action == ActionType.DONE:
            state.record_action(planned.action, planned.params, planned.confidence, planned.reason, "PASS")
            store.save_action(state.action_history[-1].model_dump())
            state.finish(RunStatus.DONE)
            return
        if planned.action == ActionType.FAILED:
            state.record_action(planned.action, planned.params, planned.confidence, planned.reason, "FAIL")
            store.save_action(state.action_history[-1].model_dump())
            state.finish(RunStatus.FAILED, planned.reason)
            return

        # Act
        action_success = await self._execute_action(state, planned)

        # Record
        validation = self._validator.map_action_outcome(action_success, timed_out=False)
        state.record_action(
            planned.action, planned.params, planned.confidence, planned.reason, validation.value
        )
        store.save_action(state.action_history[-1].model_dump())

        if not action_success:
            state.retries += 1

        # Small pause between steps
        await asyncio.sleep(0.5)

    async def _execute_action(self, state: RunState, planned: PlannedAction) -> bool:
        """Execute the planned action via DAB and return whether it succeeded."""
        action = planned.action
        params = planned.params or {}
        try:
            if action in KEY_MAP:
                resp = await self._dab.key_press(KEY_MAP[action])
                return resp.success
            elif action == ActionType.LAUNCH_APP:
                app_id = params.get("app_id", "")
                if not app_id:
                    logger.warning("LAUNCH_APP missing app_id param")
                    return False
                resp = await self._dab.launch_app(app_id)
                if resp.success:
                    state.current_app = app_id
                return resp.success
            elif action == ActionType.GET_STATE:
                app_id = state.current_app or params.get("app_id", "")
                resp = await self._dab.get_app_state(app_id)
                if resp.success:
                    state.current_app = resp.data.get("appId", state.current_app)
                    state.current_screen = resp.data.get("state")
                return resp.success
            elif action in (ActionType.CAPTURE_SCREENSHOT, ActionType.NEED_BETTER_VIEW):
                await self._update_capture(state)
                return state.latest_screenshot_b64 is not None
            elif action == ActionType.WAIT:
                seconds = params.get("seconds", 1)
                await asyncio.sleep(float(seconds))
                return True
            else:
                logger.warning("Unknown action type: %s", action)
                return False
        except Exception as exc:
            logger.error("Action execution failed: action=%s error=%s", action, exc)
            return False

    async def _update_capture(self, state: RunState) -> None:
        """Capture a new screenshot and update state in-place."""
        capture = await self._capture.capture()
        state.latest_screenshot_b64 = capture.image_b64
        state.latest_ocr_text = capture.ocr_text
