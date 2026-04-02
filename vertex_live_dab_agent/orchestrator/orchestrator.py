"""Main orchestration loop: observe -> plan -> act -> verify -> repeat."""
import asyncio
import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from vertex_live_dab_agent.android_timezone import (
    get_setting_via_adb,
    get_timezone_via_adb,
    is_adb_device_online,
    list_timezones_via_adb,
    resolve_timezone_from_supported,
    set_setting_via_adb,
    set_timezone_via_adb,
)
from vertex_live_dab_agent.artifacts.logger import ArtifactStore
from vertex_live_dab_agent.capture.capture import ScreenCapture
from vertex_live_dab_agent.capture.validator import Validator
from vertex_live_dab_agent.config import get_config
from vertex_live_dab_agent.dab.client import DABClientBase, create_dab_client
from vertex_live_dab_agent.dab.topics import KEY_MAP
from vertex_live_dab_agent.hybrid import (
    DeviceProfileRegistry,
    ExperienceQuery,
    HybridPolicyEngine,
    LocalActionRanker,
    LocalTrainingExample,
    TrajectoryMemory,
    extract_local_visual_features,
)
from vertex_live_dab_agent.orchestrator.app_resolver import AppResolver
from vertex_live_dab_agent.orchestrator.run_state import RunState, RunStatus
from vertex_live_dab_agent.planner.planner import Planner
from vertex_live_dab_agent.planner.schemas import ActionType, NavigationBatchAction, PlannedAction, TaskPrePlan
from vertex_live_dab_agent.system_ops.routing import (
    has_android_adb_fallback,
    operation_supported_by_dab,
    resolve_execution_method,
)
from vertex_live_dab_agent.system_ops.device_detection import get_device_platform_info
from vertex_live_dab_agent.system_ops.capabilities import (
    build_capability_snapshot,
    has_key,
    has_operation,
    has_setting,
    normalize_setting_key,
    normalize_setting_value,
)

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
        self._app_resolver = AppResolver(self._dab)
        self._validator = Validator()
        requested_ai_budget = max_steps if max_steps is not None else self._config.max_steps_per_run
        self._max_ai_requests = max(1, min(50, int(requested_ai_budget)))
        self._capture_interval_steps = max(1, int(self._config.planner_capture_interval_steps))
        self._step_delay_seconds = max(0.0, float(self._config.planner_step_delay_seconds))
        self._run_timeout_seconds = max(1.0, float(getattr(self._config, "session_timeout_seconds", 120)))
        self._step_timeout_seconds = max(1.0, float(getattr(self._config, "orchestrator_step_timeout_seconds", 120.0)))
        artifacts_root = Path(str(self._config.artifacts_base_dir or "./artifacts"))
        profiles_dir = Path(str(self._config.device_profiles_dir or "")).expanduser() if str(self._config.device_profiles_dir or "").strip() else artifacts_root / "device_profiles"
        memory_path = Path(str(self._config.trajectory_memory_path or "")).expanduser() if str(self._config.trajectory_memory_path or "").strip() else artifacts_root / "experience" / "trajectories.jsonl"
        ranker_model_path = Path(str(self._config.local_ranker_model_path or "")).expanduser() if str(self._config.local_ranker_model_path or "").strip() else artifacts_root / "models" / "local_action_ranker.json"
        self._device_profiles = DeviceProfileRegistry(profiles_dir)
        self._trajectory_memory = TrajectoryMemory(memory_path)
        self._hybrid_policy = HybridPolicyEngine()
        self._local_ranker = LocalActionRanker(ranker_model_path)

    async def run(self, state: RunState) -> RunState:
        """Execute the full run loop, saving artifacts throughout."""
        state.start()
        self._record_narration(
            state,
            step=0,
            text=f"Goal: {state.goal}",
            category="GOAL",
            priority=80,
            interruptible=False,
        )
        if not state.master_plan:
            try:
                state.master_plan = self._planner.build_master_plan(state.goal)
            except Exception:
                state.master_plan = []
        store = ArtifactStore(state.run_id)
        state.artifacts_dir = str(store.run_dir)

        store.save_metadata({
            "run_id": state.run_id,
            "goal": state.goal,
            "started_at": state.started_at,
            "max_ai_requests": self._max_ai_requests,
            "mock_mode": self._config.dab_mock_mode,
        })
        logger.info("Run started: run_id=%s goal=%r", state.run_id, state.goal)
        run_started_at = asyncio.get_running_loop().time()

        try:
            while state.status == RunStatus.RUNNING:
                if state.ai_request_count >= self._max_ai_requests:
                    state.finish(
                        RunStatus.TIMEOUT,
                        f"Max AI requests exceeded ({self._max_ai_requests})",
                    )
                    break
                elapsed_s = asyncio.get_running_loop().time() - run_started_at
                if elapsed_s >= self._run_timeout_seconds:
                    state.finish(
                        RunStatus.TIMEOUT,
                        (
                            f"Run timed out after {int(self._run_timeout_seconds)} seconds "
                            "while executing navigation and verification steps."
                        ),
                    )
                    state.record_ai_event(
                        {
                            "type": "run-timeout",
                            "step": state.step_count,
                            "timeout_seconds": self._run_timeout_seconds,
                            "reason": "run deadline reached",
                        }
                    )
                    break
                try:
                    await asyncio.wait_for(self._step(state, store), timeout=self._step_timeout_seconds)
                except asyncio.TimeoutError:
                    state.finish(
                        RunStatus.TIMEOUT,
                        (
                            f"A navigation step timed out after {int(self._step_timeout_seconds)} seconds. "
                            "The agent stopped safely."
                        ),
                    )
                    state.record_ai_event(
                        {
                            "type": "step-timeout",
                            "step": state.step_count,
                            "timeout_seconds": self._step_timeout_seconds,
                            "strategy": state.strategy_selected,
                        }
                    )
                    self._record_narration(
                        state,
                        step=state.step_count,
                        text=(
                            f"This step timed out after {int(self._step_timeout_seconds)} seconds. "
                            "Stopping safely."
                        ),
                        category="FAILURE",
                        priority=90,
                        interruptible=False,
                    )
                    break
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
        previous_action = state.last_actions[-1] if state.last_actions else None
        task_preplan = self._build_task_preplan(state)
        state.task_preplan = task_preplan.model_dump()
        state.verification_mode = task_preplan.verification_mode
        await self._bootstrap_capabilities_if_needed(state)
        self._persist_capability_reference_if_needed(state, store)

        should_observe = (
            state.latest_screenshot_b64 is None
            or state.steps_since_observe >= self._capture_interval_steps
            or previous_action in {
                ActionType.CAPTURE_SCREENSHOT.value,
                ActionType.NEED_BETTER_VIEW.value,
                ActionType.GET_STATE.value,
                ActionType.LAUNCH_APP.value,
                ActionType.PRESS_OK.value,
                ActionType.PRESS_BACK.value,
                ActionType.PRESS_HOME.value,
            }
        )

        # Observe (sparse cadence for speed)
        if should_observe:
            capture_result = await self._capture.capture()
            state.latest_screenshot_b64 = capture_result.image_b64
            state.latest_visual_summary = capture_result.ocr_text
            self._refresh_observation_features(state)
            if capture_result.image_b64:
                store.save_screenshot(capture_result.image_b64, step)
            state.steps_since_observe = 0
            state.last_checkpoint = "observe"
        self._refresh_player_context(state)
        self._refresh_hybrid_context(state)

        planned_from_subplan = False
        nav_phase = ""
        nav_intent = ""
        nav_expected_result = ""
        nav_checkpoint_required = False
        nav_validate_before_commit = False
        nav_need_screenshot = False
        nav_action_batch = []
        diagnosis_batch = await self._run_stuck_diagnosis_if_needed(state, step)
        preflight_batch = diagnosis_batch or await self._select_execution_strategy(state)
        if preflight_batch and not self._strategy_matches_task_semantics(state):
            state.record_ai_event(
                {
                    "type": "strategy-blocked",
                    "step": step,
                    "strategy": state.strategy_selected,
                    "reason": "strategy does not match task semantics",
                }
            )
            preflight_batch = []
        if preflight_batch:
            primary = preflight_batch[0]
            planned = PlannedAction(
                action=primary["action"],
                confidence=0.96,
                reason=f"strategy:{state.strategy_selected}",
                params=primary.get("params") or None,
            )
            planned_from_subplan = False
            nav_phase = "strategy"
            nav_intent = f"{state.strategy_selected} preflight"
            nav_action_batch = preflight_batch
            for sub in preflight_batch[1:]:
                state.pending_subplan.append(
                    {
                        "action": sub.get("action"),
                        "params": sub.get("params") or {},
                        "reason": f"Strategy follow-up after {planned.action}",
                    }
                )
            nav_need_screenshot = True
            nav_checkpoint_required = True
            nav_expected_result = "execute direct DAB strategy before UI navigation"
            self._record_narration(
                state,
                step=step,
                text=self._narrate_action(str(primary.get("action", "")), primary.get("params") or {}),
                category="STEP_START",
            )
        elif state.pending_subplan:
            queued = state.pending_subplan.pop(0)
            queued_action = str(queued.get("action", ActionType.WAIT.value))
            queued_params = queued.get("params") if isinstance(queued.get("params"), dict) else {}
            queued_action, queued_params, queued_resolution_error = await self._normalize_launch_action(
                state=state,
                action=queued_action,
                params=queued_params,
                planner_output=None,
            )
            if queued_resolution_error:
                logger.warning("Launch target resolution failed for queued action: %s", queued_resolution_error)
                state.record_ai_event(
                    {
                        "type": "launch-resolution-failed",
                        "step": step,
                        "source": "subplan",
                        "error": queued_resolution_error,
                        "fallback_action": ActionType.NEED_BETTER_VIEW.value,
                    }
                )
                self._record_narration(
                    state,
                    step=step,
                    text="Recovery started because app launch target could not be resolved.",
                    category="RECOVERY",
                    priority=70,
                )
            if self._is_guarded_commit_action(queued_action) and state.steps_since_observe > 0:
                state.pending_subplan.insert(0, queued)
                planned = PlannedAction(
                    action=ActionType.CAPTURE_SCREENSHOT,
                    confidence=0.9,
                    reason="Guarded commit checkpoint before queued commit action",
                )
                planned_from_subplan = True
                nav_phase = "guarded_commit"
                nav_intent = "validate_destination_before_commit"
                nav_action_batch = [{"action": planned.action, "params": {}}]
            else:
                planned = PlannedAction(
                    action=queued_action,
                    confidence=0.85,
                    reason=str(queued.get("reason", "Executing queued sub-plan action")),
                    params=queued_params,
                )
                planned_from_subplan = True
                nav_phase = "subplan"
                nav_intent = "execute queued action"
                nav_action_batch = [{"action": planned.action, "params": planned.params or {}}]
                self._record_narration(
                    state,
                    step=step,
                    text=self._narrate_action(str(planned.action), planned.params or {}),
                    category="STEP_START",
                )
        else:
            platform_name = str(state.device_platform or "Unknown Device").strip() or "Unknown Device"
            os_family = str(state.device_os_family or "unknown").strip() or "unknown"
            exec_state = {
                "platform_name": platform_name,
                "os_family": os_family,
                "app_context": state.current_app_id or state.current_app or "unknown",
                "target_operation": str((state.task_preplan or {}).get("step_type") or "MENU_NAVIGATION"),
                "target_screen": str(state.last_expected_screen or state.current_screen or "unknown"),
                "target_item": str(state.focus_target_guess or "unknown"),
                "navigation_memory": self._build_navigation_memory_summary(state),
                "current_app_guess": state.current_app,
                "current_screen_guess": state.current_screen_guess or state.current_screen,
                "highlighted_item_guess": state.highlighted_item_guess,
                "confidence": state.nav_confidence,
                "failed_paths": state.failed_paths[-8:],
                "last_checkpoint": state.last_checkpoint,
                "supported_operations": state.supported_operations,
                "app_catalog": state.app_catalog,
                "supported_keys": state.supported_keys,
                "supported_settings": state.supported_settings,
                "capability_snapshot": state.capability_snapshot,
                "resolved_apps_cache": state.resolved_apps_cache,
                "strategy_selected": state.strategy_selected,
                "hybrid_policy_mode": state.hybrid_policy_mode,
                "hybrid_policy_rationale": state.hybrid_policy_rationale,
                "device_profile_id": state.device_profile_id,
                "retrieved_experiences": state.retrieved_experiences[-3:],
                "observation_features": state.observation_features,
                "local_action_suggestions": state.local_action_suggestions,
                "local_model_version": state.local_model_version,
                "current_app_state": state.current_app_state,
                "current_app_id": state.current_app_id,
                "last_verified_foreground_app": state.last_verified_foreground_app,
                "is_video_playback_context": state.is_video_playback_context,
                "player_controls_visible": state.player_controls_visible,
                "focus_target_guess": state.focus_target_guess,
                "last_ok_effect": state.last_ok_effect,
                "repeated_commit_count": state.repeated_commit_count,
                "no_progress_count": state.no_progress_count,
                "last_player_phase": state.last_player_phase,
                # World model fields from requirement 2
                "current_subgoal": state.current_subgoal,
                "target_app_name": state.target_app_name,
                "target_app_domain": state.target_app_domain,
                "grounded_screenshot_summary": state.grounded_screenshot_summary,
                "blocked_actions": state.blocked_actions,
                "recent_ai_events": state.ai_transcript[-10:],
                "recent_dab_events": state.dab_transcript[-8:],
                "recent_action_records": [
                    {
                        "step": a.step,
                        "action": a.action,
                        "result": a.result,
                        "reason": a.reason,
                    }
                    for a in state.action_history[-10:]
                ],
            }
            nav_plan = await self._planner.plan_navigation(
                goal=state.goal,
                screenshot_b64=state.latest_screenshot_b64,
                ocr_text=state.latest_visual_summary,
                current_app=state.current_app,
                current_screen=state.current_screen,
                last_actions=state.last_actions,
                retry_count=state.retries,
                launch_content=state.launch_content,
                execution_state=exec_state,
                session_id=state.run_id,
                master_plan=state.master_plan,
            )
            vertex_prompt = str(getattr(self._planner, "last_vertex_prompt", "") or "").strip()
            if vertex_prompt:
                state.ai_request_count += 1
                state.record_ai_event(
                    {
                        "type": "gemini-request",
                        "step": step,
                        "ai_request_count": state.ai_request_count,
                        "session_id": state.run_id,
                        "prompt": vertex_prompt,
                    }
                )
            vertex_response = str(getattr(self._planner, "last_vertex_response", "") or "").strip()
            if vertex_response:
                state.record_ai_event(
                    {
                        "type": "gemini-response",
                        "step": step,
                        "session_id": state.run_id,
                        "response": vertex_response,
                    }
                )
            state.nav_confidence = float(nav_plan.confidence)
            state.current_screen_guess = state.current_screen or state.current_screen_guess
            nav_phase = nav_plan.phase
            nav_intent = nav_plan.intent
            nav_expected_result = nav_plan.expected_result
            if nav_expected_result:
                state.last_expected_screen = nav_expected_result
            nav_checkpoint_required = bool(nav_plan.checkpoint_required)
            nav_validate_before_commit = bool(nav_plan.validate_before_commit)
            nav_need_screenshot = bool(nav_plan.need_screenshot)
            nav_action_batch = [
                {"action": b.action, "params": b.params or {}}
                for b in (nav_plan.action_batch or [])
            ]
            nav_action_batch, launch_resolution_failures = await self._normalize_navigation_batch(
                state=state,
                nav_action_batch=nav_action_batch,
                planner_output=nav_plan.model_dump(),
            )
            nav_batch_models = [NavigationBatchAction.model_validate(s) for s in nav_action_batch]
            if launch_resolution_failures:
                nav_intent = f"{nav_intent} (launch resolution fallback used)".strip()
                self._record_narration(
                    state,
                    step=step,
                    text="Recovery started because app launch details were unclear.",
                    category="RECOVERY",
                    priority=70,
                )

            if str(nav_plan.execution_mode) in {
                "DIRECT_APP_LAUNCH",
                "DIRECT_APP_LAUNCH_WITH_PARAMS",
                "GO_HOME_THEN_LAUNCH",
                "GO_HOME_AND_RECOVER",
                "RECOVERY_RELAUNCH",
                "RELAUNCH_TARGET_APP",
            }:
                has_launch_step = any(
                    str((item or {}).get("action", "")).upper() == ActionType.LAUNCH_APP.value
                    for item in nav_action_batch
                )
                resolved = await self._app_resolver.resolve_target_app(
                    goal=state.goal,
                    planner_output=nav_plan.model_dump(),
                    execution_state={
                        "session_id": state.run_id,
                        "device_id": self._config.dab_device_id,
                    },
                )
                if resolved is not None:
                    # Track resolved target app on state
                    if not state.target_app_name:
                        state.target_app_name = resolved.app_name or nav_plan.target_app_name
                    if not state.target_app_domain:
                        state.target_app_domain = nav_plan.target_app_domain
                if resolved is not None and not has_launch_step:
                    launch_step = self._app_resolver.build_launch_action(
                        resolved_target=resolved,
                        launch_parameters=nav_plan.launch_parameters,
                    )
                    strategy_batch: list[dict] = []
                    if str(nav_plan.execution_mode) in {
                        "GO_HOME_THEN_LAUNCH",
                        "GO_HOME_AND_RECOVER",
                        "RECOVERY_RELAUNCH",
                        "RELAUNCH_TARGET_APP",
                    }:
                        strategy_batch.append({"action": ActionType.PRESS_HOME.value, "params": {}})
                    strategy_batch.append(launch_step)
                    strategy_batch.append({"action": ActionType.WAIT.value, "params": {"seconds": 1.0}})
                    strategy_batch.append({"action": ActionType.GET_STATE.value, "params": {"app_id": resolved.app_id}})
                    nav_action_batch = strategy_batch + nav_action_batch
                    nav_action_batch, _ = await self._normalize_navigation_batch(
                        state=state,
                        nav_action_batch=nav_action_batch,
                        planner_output=nav_plan.model_dump(),
                    )
                    nav_batch_models = [NavigationBatchAction.model_validate(s) for s in nav_action_batch]
                    nav_checkpoint_required = True
                    nav_need_screenshot = True
                    nav_intent = f"{nav_intent} (launch-first via resolver)".strip()

            guarded_commit_idx = self._guarded_commit_index(nav_action_batch)
            if guarded_commit_idx is not None:
                nav_action_batch = nav_action_batch[:guarded_commit_idx]
                nav_batch_models = nav_batch_models[:guarded_commit_idx]
                nav_checkpoint_required = True
                nav_validate_before_commit = True
                nav_need_screenshot = True
                nav_intent = f"{nav_intent} (guarded commit deferred)".strip()

            # checkpoint capture only when explicitly needed by plan/guard
            if nav_need_screenshot and state.steps_since_observe > 0:
                capture_result = await self._capture.capture()
                state.latest_screenshot_b64 = capture_result.image_b64
                state.latest_visual_summary = capture_result.ocr_text
                self._refresh_observation_features(state)
                if capture_result.image_b64:
                    store.save_screenshot(capture_result.image_b64, step)
                state.steps_since_observe = 0
                state.last_checkpoint = "plan-requested"

            if nav_plan.done and not nav_plan.action_batch:
                planned = PlannedAction(
                    action=ActionType.DONE,
                    confidence=nav_plan.confidence,
                    reason=nav_plan.expected_result or nav_plan.intent,
                )
            else:
                primary = nav_batch_models[0] if nav_batch_models else None
                if primary is None:
                    planned = PlannedAction(
                        action=ActionType.CAPTURE_SCREENSHOT,
                        confidence=0.6,
                        reason="No batch action returned; checkpointing",
                    )
                else:
                    planned = PlannedAction(
                        action=primary.action,
                        confidence=nav_plan.confidence,
                        reason=f"{nav_plan.phase}: {nav_plan.intent}",
                        params=primary.params,
                    )

            if (
                not nav_validate_before_commit
                and self._is_guarded_commit_action(str(planned.action))
                and state.steps_since_observe > 0
            ):
                nav_validate_before_commit = True

            if self._config.planner_enable_subplans and nav_batch_models:
                for sub in nav_batch_models[1:]:
                    state.pending_subplan.append(
                        {
                            "action": sub.action,
                            "params": sub.params,
                            "reason": f"Batch follow-up after {planned.action}",
                        }
                    )

            # For short directional batches, force a visual re-check checkpoint.
            if self._requires_batch_recheck(nav_action_batch):
                nav_checkpoint_required = True

            if nav_checkpoint_required:
                if not state.pending_subplan or state.pending_subplan[-1].get("action") != ActionType.CAPTURE_SCREENSHOT.value:
                    state.pending_subplan.append({"action": ActionType.CAPTURE_SCREENSHOT.value, "reason": "Checkpoint"})

            if nav_plan.validate_before_commit and planned.action in {
                ActionType.PRESS_OK.value,
                ActionType.PRESS_HOME.value,
                ActionType.LAUNCH_APP.value,
            } and state.steps_since_observe > 0:
                state.pending_subplan.insert(0, {"action": planned.action, "params": planned.params, "reason": planned.reason})
                planned = PlannedAction(
                    action=ActionType.CAPTURE_SCREENSHOT,
                    confidence=max(0.7, nav_plan.confidence),
                    reason="Pre-commit validation checkpoint",
                )

        planned = self._sanitize_planned_action_for_goal(state, planned)
        logger.info(
            "Planner decision: run_id=%s step=%d action=%s confidence=%.2f reason=%r source=%s queued_subplan=%d",
            state.run_id,
            step,
            planned.action,
            planned.confidence,
            planned.reason,
            "subplan" if planned_from_subplan else "planner",
            len(state.pending_subplan),
        )
        state.record_ai_event(
            {
                "type": "planner-decision",
                "step": step,
                "source": "subplan" if planned_from_subplan else "planner",
                "phase": nav_phase,
                "intent": nav_intent,
                "action": planned.action,
                "confidence": planned.confidence,
                "reason": planned.reason,
                "params": planned.params or {},
                "action_batch": nav_action_batch,
                "expected_result": nav_expected_result,
                "checkpoint_required": nav_checkpoint_required,
                "validate_before_commit": nav_validate_before_commit,
                "need_screenshot": nav_need_screenshot,
                "queued_subplan": len(state.pending_subplan),
                "current_app": state.current_app,
                "current_screen": state.current_screen,
                "retry_count": state.retries,
            }
        )
        store.save_planner_trace(
            {
                "step": step,
                "goal": state.goal,
                "current_app": state.current_app,
                "current_screen": state.current_screen,
                "visual_summary": state.latest_visual_summary,
                "last_actions": state.last_actions[-5:],
                "retry_count": state.retries,
                "planned_action": planned.action,
                "confidence": planned.confidence,
                "reason": planned.reason,
                "params": planned.params,
                "source": "subplan" if planned_from_subplan else "planner",
                "queued_subplan": len(state.pending_subplan),
            },
            step,
        )

        # Handle terminal actions before incrementing step_count
        if planned.action == ActionType.DONE:
            youtube_goal = self._is_youtube_player_task(state.goal)
            youtube_verified = self._is_youtube_goal_verified(state)
            open_app_goal = self._is_open_app_goal(state.goal)
            open_app_verified = self._is_app_goal_verified(state)
            if (youtube_goal and not youtube_verified) or (open_app_goal and not open_app_verified):
                state.record_ai_event(
                    {
                        "type": "terminal-check-blocked",
                        "step": step,
                        "status": "DONE_BLOCKED",
                        "reason": (
                            "YouTube in-app goal not verified"
                            if youtube_goal and not youtube_verified
                            else "App goal not verified in FOREGROUND"
                        ),
                        "current_app": state.current_app_id,
                        "current_app_state": state.current_app_state,
                    }
                )
                planned = PlannedAction(
                    action=ActionType.GET_STATE,
                    confidence=0.85,
                    reason="Verify foreground before DONE",
                    params={
                        "app_id": (
                            str(getattr(self._config, "youtube_app_id", "youtube") or "youtube")
                            if youtube_goal
                            else (
                                state.current_app_id
                                or state.current_app
                                or str(getattr(self._config, "youtube_app_id", "youtube") or "youtube")
                            )
                        )
                    },
                )
            else:
                state.record_ai_event(
                    {
                        "type": "terminal",
                        "step": step,
                        "status": "DONE",
                        "reason": planned.reason,
                    }
                )
                state.record_action(planned.action, planned.params, planned.confidence, planned.reason, "PASS")
                store.save_action(state.action_history[-1].model_dump())
                self._record_narration(
                    state,
                    step=step,
                    text="Success. The test finished as expected.",
                    category="SUCCESS",
                    priority=90,
                    interruptible=False,
                )
                state.finish(RunStatus.DONE)
                return
        if planned.action == ActionType.FAILED:
            state.record_ai_event(
                {
                    "type": "terminal",
                    "step": step,
                    "status": "FAILED",
                    "reason": planned.reason,
                }
            )
            state.record_action(planned.action, planned.params, planned.confidence, planned.reason, "FAIL")
            store.save_action(state.action_history[-1].model_dump())
            self._record_narration(
                state,
                step=step,
                text="The test stopped because recovery did not work.",
                category="FAILURE",
                priority=95,
                interruptible=False,
            )
            state.finish(RunStatus.FAILED, planned.reason)
            return

        visual_summary_before_action = str(state.latest_visual_summary or "")

        # Act
        action_success = await self._execute_action(state, planned)

        # Record
        validation = self._validator.map_action_outcome(action_success, timed_out=False)
        state.record_action(
            planned.action, planned.params, planned.confidence, planned.reason, validation.value
        )
        store.save_action(state.action_history[-1].model_dump())
        self._record_trajectory_experience(
            state=state,
            planned=planned,
            result=validation.value,
            visual_summary_before_action=visual_summary_before_action,
        )

        # Settings goals are single-operation tasks in this workflow.
        # After a successful GET/SET setting action (DAB or Android ADB fallback),
        # stop the run instead of repeating the same action until timeout.
        planned_action_name = str(getattr(planned.action, "value", planned.action) or "").upper()
        setting_key_name = str((planned.params or {}).get("key") or "").strip().lower()
        if (
            action_success
            and planned_action_name in {ActionType.GET_SETTING.value, ActionType.SET_SETTING.value}
            and (
                self._is_settings_goal(state.goal)
                or self._normalize_step_type((state.task_preplan or {}).get("step_type", "")) == "SETTING_CHANGE"
                or setting_key_name in {"timezone", "time_zone", "time-zone", "language", "brightness", "contrast", "screensaver"}
            )
        ):
            state.record_ai_event(
                {
                    "type": "terminal",
                    "step": step,
                    "status": "DONE",
                    "reason": "settings operation completed successfully",
                    "action": planned_action_name,
                    "setting_key": setting_key_name,
                }
            )
            self._record_narration(
                state,
                step=step,
                text="Settings operation completed successfully.",
                category="SUCCESS",
                priority=90,
                interruptible=False,
            )
            state.finish(RunStatus.DONE)
            return

        if (
            str(planned.action).upper() == ActionType.PRESS_OK.value
            and self._is_youtube_player_task(state.goal)
        ):
            await self._update_capture(state)
            self._refresh_player_context(state)
            ok_effect = self._diagnose_press_ok_effect(state, before_visual_summary=visual_summary_before_action)
            state.last_ok_effect = ok_effect
            state.record_ai_event(
                {
                    "type": "ok-effect-analysis",
                    "step": step,
                    "effect": ok_effect,
                    "phase": self._normalized_youtube_phase(state),
                    "repeated_commit_count": state.repeated_commit_count,
                }
            )

        if not action_success:
            state.retries += 1
            state.failed_paths.append(f"{planned.action}:{planned.reason}")
            # Replan after failures rather than continuing stale queued actions.
            state.pending_subplan.clear()
            self._record_narration(
                state,
                step=step,
                text="That step failed. Trying recovery now.",
                category="RECOVERY",
                priority=70,
            )

        if planned.action in (ActionType.CAPTURE_SCREENSHOT, ActionType.NEED_BETTER_VIEW):
            state.steps_since_observe = 0
        else:
            state.steps_since_observe += 1

        self._record_navigation_memory(state, planned, action_success)
        self._update_no_progress_tracking(state, str(planned.action), bool(action_success))

        # Small pause between steps
        if self._step_delay_seconds > 0:
            await asyncio.sleep(self._step_delay_seconds)

    @staticmethod
    def _narrate_action(action: str, params: dict) -> str:
        a = str(action or "").upper()
        if a == ActionType.LAUNCH_APP.value:
            target = params.get("app_id") or params.get("app_name") or "the app"
            return f"Trying to open {target}."
        if a == ActionType.OPEN_CONTENT.value:
            return "Trying to open the requested content."
        if a == ActionType.GET_STATE.value:
            return "Checking what screen is open now."
        if a == ActionType.PRESS_BACK.value:
            return "Pressing Back to recover navigation."
        if a == ActionType.PRESS_OK.value:
            ok_intent = str(params.get("ok_intent", "")).strip()
            if ok_intent:
                return f"Pressing OK with intent {ok_intent.lower().replace('_', ' ')}."
            return "Selecting the highlighted item."
        if a == ActionType.NEED_SETTINGS_GEAR_LOCATION.value:
            return "Looking for the settings gear in player controls."
        if a == ActionType.NEED_VIDEO_PLAYBACK_CONFIRMED.value:
            return "Trying to confirm that a video is playing."
        if a == ActionType.CAPTURE_SCREENSHOT.value:
            return "Taking a screenshot to understand the current screen."
        if a == ActionType.WAIT.value:
            return "Waiting briefly for the screen to update."
        return f"Running {a.lower().replace('_', ' ')}."

    def _record_narration(
        self,
        state: RunState,
        step: int,
        text: str,
        category: str,
        priority: int = 20,
        interruptible: bool = True,
    ) -> None:
        text_n = str(text or "").strip()
        if not text_n:
            return
        # simple dedupe + anti-spam
        last = state.narration_transcript[-1] if state.narration_transcript else None
        if last and str(last.get("tts_text", "")).strip() == text_n and int(last.get("step", -1)) == int(step):
            return
        # broader dedupe for repeated boilerplate narration across nearby steps
        # (prevents repeated GOAL/RECOVERY/STEP_START spam in transcript panel).
        recent = state.narration_transcript[-20:] if state.narration_transcript else []
        if any(str(item.get("tts_text", "")).strip() == text_n for item in recent):
            if category in {"GOAL", "RECOVERY", "STEP_START"}:
                return
        if category == "STEP_START" and last and str(last.get("tts_category", "")) == "STEP_START" and int(step) - int(last.get("step", 0)) <= 0:
            return
        state.record_narration_event(
            {
                "step": int(step),
                "tts_text": text_n,
                "tts_priority": int(priority),
                "tts_category": str(category),
                "tts_should_play": True,
                "tts_interruptible": bool(interruptible),
            }
        )

    @staticmethod
    def _is_directional_action(action: str) -> bool:
        return str(action).upper() in {"PRESS_UP", "PRESS_DOWN", "PRESS_LEFT", "PRESS_RIGHT"}

    @staticmethod
    def _is_guarded_commit_action(action: str) -> bool:
        a = str(action).upper().strip()
        if a in {"PRESS_OK", "PRESS_ENTER"}:
            return True
        return any(token in a for token in ("SELECT", "CONFIRM", "TOGGLE", "APPLY"))

    def _guarded_commit_index(self, batch: list[dict]) -> Optional[int]:
        for idx, item in enumerate(batch):
            action = str((item or {}).get("action", ""))
            if not self._is_guarded_commit_action(action):
                continue
            if idx <= 0:
                return None
            if any(self._is_directional_action(str((b or {}).get("action", ""))) for b in batch[:idx]):
                return idx
            return None
        return None

    def _requires_batch_recheck(self, batch: list[dict]) -> bool:
        if len(batch) < 2:
            return False
        directional_count = sum(
            1
            for step in batch
            if self._is_directional_action(str((step or {}).get("action", "")))
        )
        if directional_count >= 2:
            return True
        has_commit = any(str((step or {}).get("action", "")).upper() in {ActionType.PRESS_OK.value, "PRESS_ENTER"} for step in batch)
        return directional_count >= 1 and has_commit

    async def _run_stuck_diagnosis_if_needed(self, state: RunState, step: int) -> list[dict]:
        if not self._is_stuck_navigation(state):
            return []

        repeat_back_recovery = self._consecutive_recovery_decisions(state, "PRESS_BACK_AND_RECOVER")
        if repeat_back_recovery >= 3:
            state.record_ai_event(
                {
                    "type": "recovery-loop-blocked",
                    "step": step,
                    "reason": "Repeated identical recovery path detected",
                    "decision": "PRESS_BACK_AND_RECOVER",
                    "repeat_count": repeat_back_recovery,
                }
            )
            return [
                {
                    "action": ActionType.FAILED.value,
                    "params": {"reason": "stuck_recovery_loop_detected"},
                }
            ]

        if state.latest_screenshot_b64 is None:
            capture_result = await self._capture.capture()
            state.latest_screenshot_b64 = capture_result.image_b64
            state.latest_visual_summary = capture_result.ocr_text
            self._refresh_observation_features(state)

        ctx = self._build_stuck_context(state)
        state.last_stuck_context = ctx
        state.stuck_diagnosis_count += 1

        decision, reason, batch = await self._decide_recovery_from_context(state, ctx)
        state.strategy_selected = decision
        state.record_recovery(decision=decision, reason=reason, step=step)
        state.record_ai_event(
            {
                "type": "stuck-diagnosis",
                "step": step,
                "decision": decision,
                "reason": reason,
                "context": {
                    "current_app": ctx.get("current_app"),
                    "current_app_state": ctx.get("current_app_state"),
                    "current_screenshot_summary": ctx.get("current_screenshot_summary"),
                    "last_expected_screen": ctx.get("last_expected_screen"),
                    "last_successful_checkpoint": ctx.get("last_successful_checkpoint"),
                    "repeated_failures": ctx.get("repeated_failures"),
                },
                "next_actions": batch,
            }
        )
        self._record_narration(
            state,
            step=step,
            text=f"Recovery started. {reason}.",
            category="RECOVERY",
            priority=75,
        )
        return batch

    def _is_stuck_navigation(self, state: RunState) -> bool:
        if state.retries >= 2:
            return True
        if state.repeated_commit_count >= 2 and state.no_progress_count >= 1:
            return True
        if state.no_progress_count >= 3:
            return True
        recent = [str(a).upper() for a in state.last_actions[-6:]]
        if recent.count("GET_STATE") >= 2 and recent.count("NEED_BETTER_VIEW") >= 1:
            return True
        parse_fail_count = sum(
            1
            for e in state.ai_transcript[-12:]
            if str(e.get("intent", "")).startswith("parse_failure")
            or "parse_failure" in str(e.get("reason", ""))
        )
        return parse_fail_count >= 2

    def _build_stuck_context(self, state: RunState) -> dict:
        latest_decision = next(
            (e for e in reversed(state.ai_transcript) if e.get("type") == "planner-decision"),
            {},
        )
        visual_summary = str(state.latest_visual_summary or "").strip()
        screenshot_summary = visual_summary[:220] if visual_summary else "No visual summary from screenshot"
        # Update grounded screenshot summary on state for Gemini to use
        state.grounded_screenshot_summary = screenshot_summary
        return {
            "test_goal": state.goal,
            "current_subgoal": state.current_subgoal,
            "current_step_goal": (state.task_preplan or {}).get("expected_outcome") or state.goal,
            "current_screenshot_summary": screenshot_summary,
            "current_app": state.current_app_id or state.current_app,
            "current_app_state": state.current_app_state or state.current_screen,
            "last_expected_screen": latest_decision.get("expected_result") or state.last_expected_screen,
            "last_successful_checkpoint": state.last_checkpoint,
            "recent_actions": state.last_actions[-8:],
            "repeated_failures": int(state.retries),
            "available_operations": state.supported_operations,
            "available_apps": state.app_catalog,
            "supported_keys": state.supported_keys,
            "supported_settings": state.supported_settings,
            "is_video_playback_context": state.is_video_playback_context,
            "player_controls_visible": state.player_controls_visible,
            "focus_target_guess": state.focus_target_guess,
            "last_ok_effect": state.last_ok_effect,
            "repeated_commit_count": state.repeated_commit_count,
            "no_progress_count": state.no_progress_count,
            "last_player_phase": state.last_player_phase,
            "target_app_name": state.target_app_name,
            "target_app_domain": state.target_app_domain,
            "recovery_history": state.recovery_history[-5:],
        }

    def _persist_capability_reference_if_needed(self, state: RunState, store: ArtifactStore) -> None:
        if state.capability_reference_path:
            return
        payload = {
            "run_id": state.run_id,
            "goal": state.goal,
            "device_id": self._config.dab_device_id,
            "supported_operations": list(state.supported_operations or []),
            "app_catalog": list(state.app_catalog or []),
            "supported_keys": list(state.supported_keys or []),
            "supported_settings": list(state.supported_settings or []),
            "youtube_app_id": str(getattr(self._config, "youtube_app_id", "youtube") or "youtube"),
        }
        try:
            dest = store.run_dir / "capability_reference.json"
            dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            state.capability_reference = payload
            state.capability_reference_path = str(dest)
            state.record_ai_event(
                {
                    "type": "capability-reference",
                    "step": state.step_count,
                    "path": state.capability_reference_path,
                    "operations": len(payload.get("supported_operations", [])),
                    "apps": len(payload.get("app_catalog", [])),
                    "keys": len(payload.get("supported_keys", [])),
                }
            )
        except Exception as exc:
            logger.warning("Failed to persist capability reference: %s", exc)

    @staticmethod
    def _has_usable_visual_summary(state: RunState) -> bool:
        if state.latest_screenshot_b64:
            # Text extraction can be unavailable in some environments; a fresh screenshot is still
            # a usable visual signal for guarded navigation.
            return True
        visual_summary = str(state.latest_visual_summary or "").strip()
        if not visual_summary:
            return False
        return "no visual summary" not in visual_summary.lower()

    @staticmethod
    def _consecutive_recovery_decisions(state: RunState, decision: str) -> int:
        target = str(decision or "").strip().upper()
        if not target:
            return 0
        count = 0
        for item in reversed(state.recovery_history):
            d = str((item or {}).get("decision", "")).strip().upper()
            if d != target:
                break
            count += 1
        return count

    async def _decide_recovery_from_context(self, state: RunState, ctx: dict) -> tuple[str, str, list[dict]]:
        goal = str(ctx.get("test_goal", "")).lower()
        current_app_state = str(ctx.get("current_app_state", "")).upper()
        screenshot_summary = str(ctx.get("current_screenshot_summary", "")).lower()

        if self._is_youtube_player_task(goal):
            current_app = str(ctx.get("current_app") or "").lower()
            youtube_app_id = str(getattr(self._config, "youtube_app_id", "youtube") or "youtube").lower()
            if current_app != youtube_app_id or current_app_state == "BACKGROUND":
                resolved = await self._app_resolver.resolve_target_app(
                    goal=state.goal,
                    planner_output={"target_app_name": "YouTube", "target_app_hint": "YouTube"},
                    execution_state={"session_id": state.run_id, "device_id": self._config.dab_device_id},
                )
                if resolved is not None:
                    return (
                        "RELAUNCH_TARGET_APP",
                        "Goal needs YouTube player controls, so relaunching YouTube first",
                        [
                            self._app_resolver.build_launch_action(resolved_target=resolved),
                            {"action": ActionType.WAIT.value, "params": {"seconds": 1.0}},
                            {"action": ActionType.GET_STATE.value, "params": {"app_id": resolved.app_id}},
                        ],
                    )

            if not self._has_usable_visual_summary(state):
                wait_streak = self._consecutive_recovery_decisions(state, "YOUTUBE_WAIT_FOR_VISUAL_STABILITY")
                if wait_streak >= 4:
                    return (
                        "FAIL_WITH_GROUNDED_REASON",
                        "Visual summary/screenshot signal stayed unavailable across repeated recovery attempts",
                        [
                            {
                                "action": ActionType.FAILED.value,
                                "params": {"reason": "youtube_visual_signal_missing_for_recovery"},
                            }
                        ],
                    )
                if wait_streak >= 2:
                    resolved = await self._app_resolver.resolve_target_app(
                        goal=state.goal,
                        planner_output={"target_app_name": "YouTube", "target_app_hint": "YouTube"},
                        execution_state={"session_id": state.run_id, "device_id": self._config.dab_device_id},
                    )
                    if resolved is not None:
                        return (
                            "YOUTUBE_RELAUNCH_FOR_VISUAL_RESET",
                            "Visual summary still missing after repeated waits; relaunching YouTube to reset playback surface",
                            [
                                self._app_resolver.build_launch_action(resolved_target=resolved),
                                {"action": ActionType.WAIT.value, "params": {"seconds": 1.2}},
                                {"action": ActionType.GET_STATE.value, "params": {"app_id": resolved.app_id}},
                                {"action": ActionType.CAPTURE_SCREENSHOT.value, "params": {}},
                            ],
                        )

                return (
                    "YOUTUBE_WAIT_FOR_VISUAL_STABILITY",
                    "Playback context is active but visual summary is missing; waiting and recapturing before navigation",
                    [
                        {"action": ActionType.WAIT.value, "params": {"seconds": 1.2}},
                        {"action": ActionType.CAPTURE_SCREENSHOT.value, "params": {}},
                        {"action": ActionType.GET_STATE.value, "params": {"app_id": youtube_app_id}},
                    ],
                )

            if not self._is_probably_video_playback_visible(state):
                return (
                    "YOUTUBE_PHASE_RECOVERY",
                    "YouTube is open but playback is not confirmed; starting any visible video",
                    [
                        {"action": ActionType.PRESS_OK.value, "params": {"ok_intent": "SELECT_FOCUSED_CONTROL"}},
                        {"action": ActionType.WAIT.value, "params": {"seconds": 1.0}},
                        {"action": ActionType.NEED_VIDEO_PLAYBACK_CONFIRMED.value, "params": {}},
                    ],
                )
            if "settings" not in screenshot_summary and "gear" not in screenshot_summary:
                return (
                    "YOUTUBE_PHASE_RECOVERY",
                    "Playback exists but gear/control panel location is unclear; revealing controls",
                    [
                        {"action": ActionType.PRESS_RIGHT.value, "params": {}},
                        {"action": ActionType.PRESS_RIGHT.value, "params": {}},
                        {"action": ActionType.NEED_SETTINGS_GEAR_LOCATION.value, "params": {}},
                    ],
                )
            if "stats for nerds" not in screenshot_summary:
                return (
                    "YOUTUBE_PHASE_RECOVERY",
                    "Player menu likely open but Stats for Nerds toggle not confirmed yet",
                    [{"action": ActionType.NEED_STATS_FOR_NERDS_TOGGLE_CONFIRMATION.value, "params": {}}],
                )

        if self._is_open_app_goal(goal):
            target_name = self._infer_target_app_name_from_goal(goal, state.app_catalog)
            resolved = None
            if target_name:
                resolved = await self._app_resolver.resolve_target_app(
                    goal=state.goal,
                    planner_output={"target_app_name": target_name, "target_app_hint": target_name},
                    execution_state={"session_id": state.run_id, "device_id": self._config.dab_device_id},
                )
            if resolved and (current_app_state == "BACKGROUND" or "home" in screenshot_summary or "launcher" in screenshot_summary):
                return (
                    "RELAUNCH_TARGET_APP",
                    "Current screen is not the target foreground UI; relaunching target app",
                    [
                        {"action": ActionType.PRESS_HOME.value, "params": {}},
                        self._app_resolver.build_launch_action(resolved_target=resolved),
                        {"action": ActionType.WAIT.value, "params": {"seconds": 1.0}},
                        {"action": ActionType.GET_STATE.value, "params": {"app_id": resolved.app_id}},
                    ],
                )

        if self._is_settings_goal(goal):
            setting_key = self._infer_setting_key_from_goal(goal, state.supported_settings)
            setting_value = self._infer_setting_value_from_goal(state.goal)
            if setting_key:
                desired_operation = "system/settings/set" if setting_value is not None else "system/settings/get"
                method, reason = self._resolve_setting_execution_method(state, desired_operation, setting_key)
                if method in {"dab", "adb"}:
                    action = ActionType.SET_SETTING.value if setting_value is not None else ActionType.GET_SETTING.value
                    params = {"key": "timezone" if self._is_timezone_setting_key(setting_key) else setting_key}
                    if setting_value is not None:
                        params["value"] = setting_value
                    return (
                        "USE_DIRECT_DAB_OPERATION",
                        f"Using {method.upper()} settings operation based on goal and supported capabilities",
                        [{"action": action, "params": params}],
                    )
                return (
                    "FAIL_WITH_GROUNDED_REASON",
                    f"Settings operation is unsupported for this device path: {reason}",
                    [
                        {
                            "action": ActionType.FAILED.value,
                            "params": {"reason": f"unsupported_settings_operation: {reason}"},
                        }
                    ],
                )

        if "home" in screenshot_summary or "launcher" in screenshot_summary:
            return (
                "CONTINUE_WITH_ADJUSTED_NAVIGATION",
                "Device appears on Home; applying minimal goal-aware recovery",
                [{"action": ActionType.PRESS_BACK.value, "params": {}}, {"action": ActionType.WAIT.value, "params": {"seconds": 0.6}}],
            )

        if state.retries >= 4:
            return (
                "FAIL_WITH_GROUNDED_REASON",
                "Recovery retries exhausted after screenshot-based diagnosis",
                [{"action": ActionType.FAILED.value, "params": {"reason": "stuck_navigation_grounded_failure"}}],
            )

        return (
            "PRESS_BACK_AND_RECOVER",
            "Navigation drift detected; using back-and-verify recovery",
            [
                {"action": ActionType.PRESS_BACK.value, "params": {}},
                {"action": ActionType.CAPTURE_SCREENSHOT.value, "params": {}},
            ],
        )

    async def _bootstrap_capabilities_if_needed(self, state: RunState) -> None:
        goal_l = (state.goal or "").lower()
        preplan = state.task_preplan or {}

        if not state.device_info and hasattr(self._dab, "get_device_info"):
            try:
                info_resp = await self._dab.get_device_info()
                payload = info_resp.data if isinstance(getattr(info_resp, "data", None), dict) else {}
                if payload:
                    state.device_info = payload
                    self._refresh_device_type_from_info(state, payload)
                    await self._refresh_device_type_from_adb(state)
                    logger.info(
                        "Device type detection: run_id=%s platform=%s is_android=%s is_android_tv=%s connection=%s adb_device_id=%s detection_error=%s",
                        state.run_id,
                        state.device_platform,
                        state.is_android_device,
                        state.is_android_tv_device,
                        state.device_connection_type,
                        state.android_adb_device_id,
                        state.device_detection_error,
                    )
            except Exception as exc:
                logger.warning("Capability bootstrap device/info failed: %s", exc)

        if state.is_android_device is None:
            configured_hint = str(self._config.dab_device_id or "").strip().lower()
            state.is_android_device = "android" in configured_hint or configured_hint.startswith("adb:")
        if not state.android_adb_device_id:
            state.android_adb_device_id = self._infer_adb_device_id_from_value(self._config.dab_device_id)
        adb_hint = str(state.android_adb_device_id or "")
        looks_adb_target = ":" in adb_hint or str(adb_hint).lower().startswith("emulator-") or str(self._config.dab_device_id or "").strip().lower().startswith("adb:")
        if state.android_adb_device_id and (bool(state.is_android_device) or looks_adb_target):
            await self._refresh_device_type_from_adb(state)

        if not state.supported_operations:
            try:
                resp = await self._dab.list_operations()
                ops = (resp.data or {}).get("operations", []) if isinstance(resp.data, dict) else []
                if isinstance(ops, list):
                    state.supported_operations = [str(o).strip() for o in ops if str(o).strip()]
            except Exception as exc:
                logger.warning("Capability bootstrap operations/list failed: %s", exc)

        if not state.app_catalog:
            try:
                catalog = await self._app_resolver.load_app_catalog(device_id=self._config.dab_device_id)
                state.app_catalog = [
                    {
                        "appId": a.app_id,
                        "name": a.name,
                        "friendlyName": a.friendly_name,
                        "packageName": a.package_name,
                    }
                    for a in catalog
                ]
            except Exception as exc:
                logger.warning("Capability bootstrap applications/list failed: %s", exc)

        has_key_list_op = any("input/key/list" in str(o).lower() for o in state.supported_operations)
        if (not state.supported_keys) and has_key_list_op:
            try:
                resp = await self._dab.list_keys()
                keys = (resp.data or {}).get("keys", []) if isinstance(resp.data, dict) else []
                if isinstance(keys, list):
                    state.supported_keys = [str(k).strip() for k in keys if str(k).strip()]
            except Exception as exc:
                logger.warning("Capability bootstrap input/key/list failed: %s", exc)

        has_settings_list_op = any("system/settings/list" in str(o).lower() for o in state.supported_operations)
        if (not state.supported_settings) and has_settings_list_op and hasattr(self._dab, "list_settings"):
            try:
                resp = await self._dab.list_settings()
                settings = (resp.data or {}).get("settings", []) if isinstance(resp.data, dict) else []
                if isinstance(settings, list):
                    state.supported_settings = [s for s in settings if isinstance(s, dict)]
                if not bool(getattr(resp, "success", False)):
                    self._record_direct_setting_failure(state, "system/settings/list", "*", resp)
                warning_text = str((resp.data or {}).get("warning") if isinstance(getattr(resp, "data", None), dict) else "")
                if (
                    bool((resp.data or {}).get("degraded"))
                    and "no shell command implementation" in warning_text.lower()
                ):
                    state.unsupported_direct_operations[self._direct_op_cache_key("system/settings/get", "*")] = {
                        "step": state.step_count,
                        "operation": "system/settings/get",
                        "setting_key": "*",
                        "reason": warning_text or "settings/list degraded due unsupported shell command",
                        "status": int(getattr(resp, "status", 0) or 0),
                        "failure_count": int(state.direct_operation_failures.get(self._direct_op_cache_key("system/settings/get", "*"), 0)) + 1,
                    }
                    state.unsupported_direct_operations[self._direct_op_cache_key("system/settings/set", "*")] = {
                        "step": state.step_count,
                        "operation": "system/settings/set",
                        "setting_key": "*",
                        "reason": warning_text or "settings/list degraded due unsupported shell command",
                        "status": int(getattr(resp, "status", 0) or 0),
                        "failure_count": int(state.direct_operation_failures.get(self._direct_op_cache_key("system/settings/set", "*"), 0)) + 1,
                    }
            except Exception as exc:
                logger.warning("Capability bootstrap system/settings/list failed: %s", exc)

        state.capability_snapshot = build_capability_snapshot(
            supported_operations=state.supported_operations,
            supported_settings=state.supported_settings,
            supported_keys=state.supported_keys,
        )
        state.capability_preflight_done = True
        logger.info(
            "Capability preflight: run_id=%s operations=%d settings=%d keys=%d",
            state.run_id,
            len(state.capability_snapshot.get("supported_operations") or []),
            len(state.capability_snapshot.get("supported_settings") or {}),
            len(state.capability_snapshot.get("supported_keys") or []),
        )

        state.resolved_apps_cache = self._app_resolver.get_session_resolutions(state.run_id)
        if (state.supported_operations or state.app_catalog or state.supported_settings) and hasattr(self, "_device_profiles"):
            profile = self._device_profiles.upsert_from_capabilities(
                device_id=self._config.dab_device_id,
                supported_operations=state.supported_operations,
                supported_keys=state.supported_keys,
                supported_settings=state.supported_settings,
                app_catalog=state.app_catalog,
            )
            state.device_profile_id = profile.profile_id
            state.device_profile_path = str(self._device_profiles.profile_path(self._config.dab_device_id))

    def _refresh_hybrid_context(self, state: RunState) -> None:
        profile = self._device_profiles.load(self._config.dab_device_id)
        similar = self._trajectory_memory.find_similar(
            ExperienceQuery(
                goal=state.goal,
                device_id=self._config.dab_device_id,
                current_app=state.current_app or state.current_app_id or "",
                limit=5,
            )
        )
        recommendation = self._hybrid_policy.recommend(
            goal=state.goal,
            device_profile=profile.model_dump() if profile is not None else None,
            similar_experiences=similar,
        )
        state.device_profile_id = recommendation.device_profile_id or state.device_profile_id
        requested_mode = str(state.hybrid_policy_mode or "").strip()
        config_mode = str(getattr(self._config, "hybrid_policy_mode", "auto")).strip()
        if requested_mode and requested_mode.lower() != "auto":
            state.hybrid_policy_mode = requested_mode
        elif config_mode and config_mode.lower() != "auto":
            state.hybrid_policy_mode = config_mode
        else:
            state.hybrid_policy_mode = recommendation.mode
        state.hybrid_policy_rationale = recommendation.rationale
        state.retrieved_experiences = similar
        state.local_model_version = self._local_ranker.version
        ranked_actions = self._local_ranker.rank(
            goal=state.goal,
            current_app=state.current_app_id or state.current_app or "",
            observation_features=state.observation_features,
            retrieved_experiences=similar,
            top_k=3,
        )
        state.local_action_suggestions = [item.model_dump() for item in ranked_actions]

    def _refresh_observation_features(self, state: RunState) -> None:
        state.observation_features = extract_local_visual_features(
            image_b64=state.latest_screenshot_b64,
            ocr_text=state.latest_visual_summary,
        )

    def _record_trajectory_experience(
        self,
        *,
        state: RunState,
        planned: PlannedAction,
        result: str,
        visual_summary_before_action: str,
    ) -> None:
        example = LocalTrainingExample(
            run_id=state.run_id,
            step=state.step_count,
            goal=state.goal,
            device_id=self._config.dab_device_id,
            device_profile_id=state.device_profile_id,
            hybrid_policy_mode=state.hybrid_policy_mode,
            current_app=state.current_app_id or state.current_app or "",
            current_screen=state.current_screen or "",
            visual_summary_before=visual_summary_before_action,
            visual_summary_after=str(state.latest_visual_summary or ""),
            observation_features=dict(state.observation_features or {}),
            action=str(planned.action),
            params=planned.params or {},
            result=str(result),
            strategy_selected=state.strategy_selected,
            retrieved_actions=[
                str(item.get("action", ""))
                for item in (state.retrieved_experiences or [])
                if isinstance(item, dict) and str(item.get("action", "")).strip()
            ],
            local_ranker_actions=[
                str(item.get("action", ""))
                for item in (state.local_action_suggestions or [])
                if isinstance(item, dict) and str(item.get("action", "")).strip()
            ],
            reason=planned.reason,
        )
        self._trajectory_memory.append(example.model_dump())

    async def _select_execution_strategy(self, state: RunState) -> list[dict]:
        if state.pending_subplan:
            return []

        preplan = state.task_preplan or self._build_task_preplan(state).model_dump()
        step_type = self._normalize_step_type(preplan.get("step_type", "MENU_NAVIGATION"))
        required_action = str(preplan.get("required_action", "")).strip()
        needs_home_first = bool(preplan.get("needs_home_first", False))

        if self._is_youtube_player_task(state.goal):
            state.strategy_selected = "YOUTUBE_PLAYER_WORKFLOW"
            phase = self._youtube_phase(state)
            youtube_app_id = str(getattr(self._config, "youtube_app_id", "youtube") or "youtube")
            if phase == "OPEN_TARGET_APP":
                return [
                    {"action": ActionType.LAUNCH_APP.value, "params": {"app_id": youtube_app_id}},
                    {"action": ActionType.WAIT.value, "params": {"seconds": 1.0}},
                    {"action": ActionType.GET_STATE.value, "params": {"app_id": youtube_app_id}},
                ]
            if phase == "START_ANY_VIDEO":
                return [
                    {"action": ActionType.PRESS_OK.value, "params": {"ok_intent": "SELECT_FOCUSED_CONTROL"}},
                    {"action": ActionType.WAIT.value, "params": {"seconds": 1.0}},
                    {"action": ActionType.NEED_VIDEO_PLAYBACK_CONFIRMED.value, "params": {}},
                ]
            if phase == "REVEAL_PLAYER_CONTROLS":
                return [
                    {"action": ActionType.PRESS_OK.value, "params": {"ok_intent": "REVEAL_PLAYER_CONTROLS"}},
                    {"action": ActionType.WAIT.value, "params": {"seconds": 0.6}},
                    {"action": ActionType.NEED_PLAYER_CONTROLS_VISIBLE.value, "params": {}},
                ]
            if phase == "OPEN_PLAYER_SETTINGS":
                return [
                    {"action": ActionType.PRESS_RIGHT.value, "params": {}},
                    {"action": ActionType.PRESS_RIGHT.value, "params": {}},
                    {"action": ActionType.NEED_SETTINGS_GEAR_LOCATION.value, "params": {}},
                ]
            if phase == "ENABLE_STATS_FOR_NERDS":
                return [
                    {"action": ActionType.PRESS_OK.value, "params": {"ok_intent": "CONFIRM_MENU_ITEM"}},
                    {"action": ActionType.NEED_PLAYER_MENU_CONFIRMATION.value, "params": {}},
                ]
            if phase == "VERIFY_STATS_OVERLAY":
                return [{"action": ActionType.NEED_STATS_FOR_NERDS_TOGGLE_CONFIRMATION.value, "params": {}}]
            return []

        if state.step_count > 0 and not self._is_settings_goal(state.goal):
            return []

        if step_type == "DIRECT_KEY_VALIDATION" and required_action:
            state.strategy_selected = "DIRECT_KEY_VALIDATION"
            batch: list[dict] = []
            if needs_home_first and not self._is_home_context_satisfied(state):
                batch.append({"action": ActionType.PRESS_HOME.value, "params": {}})
            batch.append({"action": required_action, "params": {}})
            batch.append({"action": ActionType.WAIT.value, "params": {"seconds": 0.8}})
            return batch

        ops = [str(o).lower() for o in (state.supported_operations or [])]
        can_launch = any("applications/launch" in o for o in ops)
        can_launch_with_content = any("applications/launch-with-content" in o for o in ops)
        can_content_open = any("content/open" in o for o in ops)

        snapshot = state.capability_snapshot or build_capability_snapshot(
            supported_operations=state.supported_operations,
            supported_settings=state.supported_settings,
            supported_keys=state.supported_keys,
        )
        setting_key = self._infer_setting_key_from_goal(state.goal, state.supported_settings)
        setting_value = self._infer_setting_value_from_goal(state.goal) if setting_key else None
        if setting_key:
            resolved_key = str(setting_key or "").strip()
            key_norm = normalize_setting_key(snapshot, resolved_key)
            if bool(key_norm.get("success")):
                resolved_key = str(key_norm.get("key") or resolved_key)
                if bool(key_norm.get("corrected")):
                    logger.info("Normalized setting key before planning: original=%s normalized=%s", setting_key, resolved_key)
            elif self._is_timezone_setting_key(resolved_key):
                resolved_key = "timezone"
            else:
                state.strategy_selected = "UNSUPPORTED_SETTING_OPERATION"
                return [{"action": ActionType.FAILED.value, "params": {"reason": str(key_norm.get("reason") or "unsupported setting key")}}]
            setting_key = resolved_key

            desired_operation = "system/settings/set" if setting_value is not None else "system/settings/get"
            method, reason = self._resolve_setting_execution_method(state, desired_operation, setting_key)
            logger.info(
                "Settings routing decision: run_id=%s operation=%s key=%s method=%s reason=%s",
                state.run_id,
                desired_operation,
                setting_key,
                method,
                reason,
            )

            if method in {"dab", "adb"}:
                state.strategy_selected = "DIRECT_SETTING_OPERATION"
                if setting_value is not None:
                    resolved_value = setting_value
                    if method == "dab":
                        value_norm = normalize_setting_value(snapshot, setting_key, setting_value)
                        if not bool(value_norm.get("success")):
                            return [{"action": ActionType.FAILED.value, "params": {"reason": str(value_norm.get("reason") or "invalid setting value")}}]
                        resolved_value = value_norm.get("value")
                        if bool(value_norm.get("corrected")):
                            logger.info(
                                "Normalized setting value before planning: key=%s original=%s normalized=%s",
                                setting_key,
                                setting_value,
                                resolved_value,
                            )
                    return [{"action": ActionType.SET_SETTING.value, "params": {"key": setting_key, "value": resolved_value}}]
                return [{"action": ActionType.GET_SETTING.value, "params": {"key": setting_key}}]

            state.strategy_selected = "UNSUPPORTED_SETTING_OPERATION"
            return [
                {
                    "action": ActionType.FAILED.value,
                    "params": {
                        "reason": (
                            f"Unsupported settings operation '{desired_operation}' for key '{setting_key}'. "
                            f"{reason}. UI navigation fallback is disabled for settings operations."
                        )
                    },
                }
            ]

        if self._is_settings_goal(state.goal):
            state.strategy_selected = "UNSUPPORTED_SETTING_OPERATION"
            return [
                {
                    "action": ActionType.FAILED.value,
                    "params": {
                        "reason": (
                            "Settings operation requested but no supported setting key was identified. "
                            "UI navigation fallback is disabled for settings operations."
                        )
                    },
                }
            ]

        target_name = self._infer_target_app_name_from_goal(state.goal, state.app_catalog)
        has_content = bool(str(state.launch_content or "").strip())
        if target_name and can_launch:
            resolved = await self._app_resolver.resolve_target_app(
                goal=state.goal,
                planner_output={
                    "target_app_name": target_name,
                    "target_app_hint": target_name,
                    "target_app_domain": "media" if not self._is_settings_goal(state.goal) else "system_settings",
                    "launch_parameters": {"content": str(state.launch_content).strip()} if has_content else {},
                },
                execution_state={"session_id": state.run_id, "device_id": self._config.dab_device_id},
            )
            if resolved is not None:
                # Track resolved target app on RunState world model
                if not state.target_app_name:
                    state.target_app_name = resolved.app_name or target_name
                if not state.target_app_domain:
                    state.target_app_domain = "media" if not self._is_settings_goal(state.goal) else "system_settings"
                state.strategy_selected = (
                    "DIRECT_APP_LAUNCH_WITH_PARAMS"
                    if has_content and (can_launch_with_content or can_content_open)
                    else "DIRECT_APP_LAUNCH"
                )
                if has_content and can_content_open:
                    return [
                        {
                            "action": "OPEN_CONTENT",
                            "params": {"content": str(state.launch_content).strip(), "app_id": resolved.app_id},
                        }
                    ]
                launch_step = self._app_resolver.build_launch_action(
                    resolved_target=resolved,
                    launch_parameters={"content": str(state.launch_content).strip()} if has_content else {},
                )
                return [
                    launch_step,
                    {"action": ActionType.WAIT.value, "params": {"seconds": 1.0}},
                    {"action": ActionType.GET_STATE.value, "params": {"app_id": resolved.app_id}},
                ]
            state.resolution_failures += 1

        state.strategy_selected = "UI_NAVIGATION_ONLY"
        return []

    @staticmethod
    def _direct_op_cache_key(operation: str, setting_key: Optional[str]) -> str:
        op = str(operation or "").strip().lower()
        key = str(setting_key or "*").strip().lower() or "*"
        return f"{op}:{key}"

    @staticmethod
    def _extract_resp_error_text(resp: Any) -> str:
        data = getattr(resp, "data", {}) or {}
        if isinstance(data, dict):
            return str(data.get("error") or data.get("message") or "").strip()
        return ""

    def _is_known_unsupported_direct_op_response(self, resp: Any) -> bool:
        status = int(getattr(resp, "status", 0) or 0)
        if status < 500:
            return False
        lowered = self._extract_resp_error_text(resp).lower()
        return any(
            marker in lowered
            for marker in (
                "no shell command implementation",
                "getcecenabled",
                "getcurrentsystemsettings",
                "getsystemsettings",
                "listsystemsettings",
                "listsupportedsystemsettings",
            )
        )

    @staticmethod
    def _is_timezone_setting_key(setting_key: Optional[str]) -> bool:
        key = str(setting_key or "").strip().lower()
        return key in {"timezone", "time_zone", "time-zone"}

    @staticmethod
    def _infer_adb_device_id_from_value(raw_device_id: Optional[str]) -> Optional[str]:
        value = str(raw_device_id or "").strip()
        if not value:
            return None
        lower = value.lower()
        if lower.startswith("adb:"):
            tail = value[4:].strip()
            return tail or None
        if value.upper().startswith("DAB/"):
            return None
        return value

    def _refresh_device_type_from_info(self, state: RunState, info: dict[str, Any]) -> None:
        platform = str(
            info.get("platform")
            or info.get("osFamily")
            or info.get("os")
            or info.get("deviceType")
            or ""
        ).strip()
        lower_platform = platform.lower()
        looks_android = any(token in lower_platform for token in ("android", "android-tv", "tvos-android"))
        looks_android_tv = any(token in lower_platform for token in ("android-tv", "google-tv", "television", "leanback"))
        state.device_platform = platform or state.device_platform
        state.device_os_family = lower_platform or state.device_os_family
        state.is_android_device = looks_android if platform else state.is_android_device
        state.is_android_tv_device = looks_android_tv if platform else state.is_android_tv_device

        adb_candidates = [
            info.get("adbDeviceId"),
            info.get("adb_device_id"),
            info.get("adbSerial"),
            info.get("adb_serial"),
            info.get("adb"),
            info.get("serial"),
            info.get("deviceId"),
            info.get("device_id"),
            self._config.dab_device_id,
        ]
        for candidate in adb_candidates:
            resolved = self._infer_adb_device_id_from_value(str(candidate or "").strip())
            if resolved:
                state.android_adb_device_id = resolved
                break

    async def _refresh_device_type_from_adb(self, state: RunState) -> None:
        adb_device_id = str(state.android_adb_device_id or "").strip()
        if not adb_device_id:
            return
        try:
            platform_info = await get_device_platform_info(adb_device_id)
        except Exception as exc:
            state.device_detection_error = str(exc)
            logger.warning("ADB device detection failed: device_id=%s error=%s", adb_device_id, exc)
            return

        state.device_connection_type = str(platform_info.get("connection_type") or state.device_connection_type or "unknown")
        state.device_detection_error = str(platform_info.get("error") or "").strip() or None
        evidence = platform_info.get("evidence")
        if isinstance(evidence, dict):
            state.device_detection_evidence = dict(evidence)

        if bool(platform_info.get("reachable")):
            state.is_android_device = bool(platform_info.get("is_android"))
            state.is_android_tv_device = bool(platform_info.get("is_android_tv"))
            sdk = str(platform_info.get("sdk") or "").strip()
            product = str(platform_info.get("product") or "").strip()
            characteristics = str(platform_info.get("build_characteristics") or "").strip()
            state.device_info.update(
                {
                    "adb_device_id": adb_device_id,
                    "connection_type": state.device_connection_type,
                    "adb_detection": {
                        "sdk": sdk,
                        "product": product,
                        "build_characteristics": characteristics,
                        "tv_features": list(platform_info.get("tv_features") or []),
                    },
                }
            )
            logger.info(
                "ADB device classification: run_id=%s device_id=%s connection=%s is_android=%s is_android_tv=%s evidence=%s",
                state.run_id,
                adb_device_id,
                state.device_connection_type,
                state.is_android_device,
                state.is_android_tv_device,
                state.device_detection_evidence,
            )
        else:
            logger.warning(
                "ADB device classification unreachable: run_id=%s device_id=%s connection=%s error=%s",
                state.run_id,
                adb_device_id,
                state.device_connection_type,
                state.device_detection_error,
            )

    def _is_direct_setting_unavailable_for_fallback(self, state: RunState, resp: Any, operation: str, setting_key: Optional[str]) -> bool:
        if self._is_direct_setting_op_unavailable(state, operation, setting_key):
            return True
        if not any("system/settings/set" in str(o).lower() for o in (state.supported_operations or [])):
            return True

        status = int(getattr(resp, "status", 0) or 0)
        if status in {404, 405, 501}:
            return True
        lowered = self._extract_resp_error_text(resp).lower()
        unavailable_markers = (
            "not supported",
            "unsupported",
            "not implemented",
            "unavailable",
            "no shell command implementation",
            "operation not found",
        )
        return any(marker in lowered for marker in unavailable_markers)

    def _can_attempt_android_timezone_adb_fallback(self, state: RunState) -> bool:
        is_android = bool(state.is_android_device)
        adb_device_id = str(state.android_adb_device_id or "").strip()
        if not is_android:
            return False
        return bool(adb_device_id)

    @staticmethod
    def _normalize_adb_setting_key(setting_key: Optional[str]) -> str:
        key = str(setting_key or "").strip().lower()
        return re.sub(r"[\s\-]+", "_", key)

    def _can_attempt_android_setting_adb_fallback(
        self,
        state: RunState,
        operation: str,
        setting_key: Optional[str],
    ) -> bool:
        if not self._can_attempt_android_timezone_adb_fallback(state):
            return False
        return has_android_adb_fallback(operation, setting_key)

    def _resolve_setting_execution_method(self, state: RunState, operation: str, setting_key: Optional[str]) -> tuple[str, str]:
        snapshot = state.capability_snapshot or build_capability_snapshot(
            supported_operations=state.supported_operations,
            supported_settings=state.supported_settings,
            supported_keys=state.supported_keys,
        )
        input_key = str(setting_key or "").strip()
        normalized_key = input_key
        key_known_in_snapshot = False
        if normalized_key:
            key_norm = normalize_setting_key(snapshot, normalized_key)
            if bool(key_norm.get("success")):
                normalized_key = str(key_norm.get("key") or normalized_key)
                key_known_in_snapshot = has_setting(snapshot, normalized_key)
            elif self._is_timezone_setting_key(normalized_key):
                # Allow Android timezone fallback even when system/settings/list is missing/incomplete.
                normalized_key = "timezone"
                key_known_in_snapshot = has_setting(snapshot, normalized_key)
            else:
                if self._can_attempt_android_setting_adb_fallback(state, operation, normalized_key):
                    normalized_key = self._normalize_adb_setting_key(normalized_key)
                    key_known_in_snapshot = False
                else:
                    return "unsupported", str(key_norm.get("reason") or "setting key is unsupported")

        dab_supported = operation_supported_by_dab(state.supported_operations or [], operation)
        if not has_operation(snapshot, operation):
            dab_supported = False
        # Do not hard-block direct DAB setting operations only because
        # settings/list is incomplete; many devices still accept get/set.
        if normalized_key and not key_known_in_snapshot and not has_operation(snapshot, operation):
            dab_supported = False
        if self._is_direct_setting_op_unavailable(state, operation, normalized_key):
            dab_supported = False
        adb_fallback_available = has_android_adb_fallback(operation, normalized_key)
        decision = resolve_execution_method(
            is_android=bool(state.is_android_device),
            dab_supported=dab_supported,
            adb_fallback_available=adb_fallback_available,
        )
        if decision.method == "unsupported" and state.device_detection_error and adb_fallback_available and not dab_supported:
            return decision.method, f"{decision.reason}; adb detection error: {state.device_detection_error}"
        return decision.method, decision.reason

    def _is_direct_setting_op_unavailable(self, state: RunState, operation: str, setting_key: Optional[str]) -> bool:
        specific = self._direct_op_cache_key(operation, setting_key)
        wildcard = self._direct_op_cache_key(operation, "*")
        return specific in state.unsupported_direct_operations or wildcard in state.unsupported_direct_operations

    def _record_strategy_transition(
        self,
        state: RunState,
        *,
        new_strategy: str,
        reason: str,
        operation: Optional[str] = None,
        setting_key: Optional[str] = None,
    ) -> None:
        previous = str(state.strategy_selected or "UI_NAVIGATION_ONLY")
        if previous == new_strategy:
            return
        transition = {
            "step": state.step_count,
            "from": previous,
            "to": new_strategy,
            "reason": reason,
            "operation": operation,
            "setting_key": setting_key,
        }
        state.strategy_transitions.append(transition)
        if len(state.strategy_transitions) > 100:
            state.strategy_transitions = state.strategy_transitions[-100:]
        state.record_ai_event({"type": "strategy-transition", **transition})
        logger.info(
            "Strategy transition: run_id=%s step=%d from=%s to=%s op=%s key=%s reason=%s",
            state.run_id,
            state.step_count,
            previous,
            new_strategy,
            operation,
            setting_key,
            reason,
        )
        state.strategy_selected = new_strategy

    def _record_direct_setting_failure(self, state: RunState, operation: str, setting_key: Optional[str], resp: Any) -> None:
        cache_key = self._direct_op_cache_key(operation, setting_key)
        count = int(state.direct_operation_failures.get(cache_key, 0)) + 1
        state.direct_operation_failures[cache_key] = count
        error_text = self._extract_resp_error_text(resp)
        known_unsupported = self._is_known_unsupported_direct_op_response(resp)

        if known_unsupported and count >= 1:
            state.unsupported_direct_operations[cache_key] = {
                "step": state.step_count,
                "operation": operation,
                "setting_key": setting_key,
                "reason": error_text or "Unsupported direct DAB operation",
                "status": int(getattr(resp, "status", 0) or 0),
                "failure_count": count,
            }
            state.record_ai_event(
                {
                    "type": "direct-op-unsupported",
                    "step": state.step_count,
                    "operation": operation,
                    "setting_key": setting_key,
                    "status": int(getattr(resp, "status", 0) or 0),
                    "failure_count": count,
                    "reason": error_text,
                }
            )
            blocked_id = f"{operation}:{setting_key or '*'}"
            if blocked_id not in state.blocked_actions:
                state.blocked_actions.append(blocked_id)
            self._record_narration(
                state,
                step=state.step_count,
                text=(
                    "Direct DAB settings access is not supported for this operation. "
                    "Executor will use Android ADB fallback when available."
                ),
                category="RECOVERY",
                priority=75,
            )

    def _build_settings_ui_fallback_batch(self, state: RunState) -> list[dict]:
        keys = {str(k).upper() for k in (state.supported_keys or [])}
        batch: list[dict] = []
        if "KEY_HOME" in keys:
            batch.append({"action": ActionType.PRESS_HOME.value, "params": {}})
            batch.append({"action": ActionType.WAIT.value, "params": {"seconds": 0.8}})
        elif "KEY_BACK" in keys:
            batch.append({"action": ActionType.PRESS_BACK.value, "params": {}})
            batch.append({"action": ActionType.WAIT.value, "params": {"seconds": 0.6}})
        batch.append({"action": ActionType.CAPTURE_SCREENSHOT.value, "params": {}})
        return batch

    def _build_task_preplan(self, state: RunState) -> TaskPrePlan:
        g = (state.goal or "").strip()
        gl = g.lower()

        starting_context = "UNKNOWN"
        if "from the home screen" in gl or "on the home screen" in gl:
            starting_context = "HOME_SCREEN"
        elif "while app is open" in gl:
            starting_context = "APP_OPEN"
        elif "on the settings page" in gl or "in settings" in gl:
            starting_context = "SETTINGS"

        required_action = ""
        key_map = {
            "press the back": ActionType.PRESS_BACK.value,
            "press back": ActionType.PRESS_BACK.value,
            "press the home": ActionType.PRESS_HOME.value,
            "press home": ActionType.PRESS_HOME.value,
            "press ok": ActionType.PRESS_OK.value,
        }
        for phrase, action in key_map.items():
            if phrase in gl:
                required_action = action
                break

        expected_outcome = ""
        if "confirm" in gl:
            idx = gl.find("confirm")
            expected_outcome = g[idx:].strip()
        elif "verify" in gl:
            idx = gl.find("verify")
            expected_outcome = g[idx:].strip()

        step_type = "MENU_NAVIGATION"
        target_domain = "GENERAL_UI"
        target_app = None
        target_ui_context = ""
        required_subgoals: list[str] = []
        verification_condition = ""
        forbidden_detours: list[str] = []
        needs_app_launch = False
        needs_settings_navigation = False
        needs_home_first = starting_context == "HOME_SCREEN"
        verification_mode = "VISUAL"
        minimal_action_path: list[str] = []
        reason = ""

        if self._is_youtube_player_task(g):
            step_type = "MENU_NAVIGATION"
            target_domain = "APP_PLAYER_CONTROLS"
            target_app = "youtube"
            target_ui_context = "YouTube video playback screen with controls visible"
            required_subgoals = [
                "OPEN_YOUTUBE",
                "START_VIDEO",
                "REVEAL_PLAYER_CONTROLS",
                "NAVIGATE_TO_GEAR",
                "OPEN_PLAYER_SETTINGS",
                "ENABLE_STATS_FOR_NERDS",
                "VERIFY_OVERLAY",
            ]
            verification_condition = "Stats for Nerds overlay visible on video"
            forbidden_detours = [
                "do not open Android Settings",
                "do not use repeated BACK loops without screenshot reason",
                "do not use generic NEED_BETTER_VIEW",
            ]
            needs_app_launch = True
            needs_settings_navigation = False
            minimal_action_path = [
                "OPEN_YOUTUBE",
                "PLAY_ANY_VIDEO",
                "SHOW_CONTROLS",
                "OPEN_GEAR",
                "ENABLE_STATS_FOR_NERDS",
                "VERIFY_OVERLAY",
            ]
            reason = "Goal is an in-app YouTube player-controls task"

        if required_action and ("confirm" in gl or "verify" in gl):
            step_type = "DIRECT_KEY_VALIDATION"
            target_domain = "HOME_OR_SYSTEM_UI"
            needs_app_launch = False
            needs_settings_navigation = False
            minimal_action_path = [
                "GO_HOME_IF_NEEDED" if needs_home_first else "",
                required_action,
                "VERIFY_EXPECTED_OUTCOME",
            ]
            minimal_action_path = [s for s in minimal_action_path if s]
            reason = "Step explicitly requests key action + validation from context"
        elif "open" in gl or "launch" in gl:
            step_type = "APP_LAUNCH"
            target_domain = "APP"
            needs_app_launch = True
            minimal_action_path = ["RESOLVE_APP", "LAUNCH_APP", "VERIFY_FOREGROUND"]
            reason = "Step explicitly requests opening an app"
        elif "time zone" in gl or "timezone" in gl or "change" in gl or "set" in gl:
            step_type = "SETTING_CHANGE"
            target_domain = "SYSTEM_SETTINGS"
            needs_settings_navigation = False
            verification_mode = "STATE_OR_VISUAL"
            minimal_action_path = ["DIRECT_SETTING_IF_SUPPORTED", "VERIFY_SETTING", "ANDROID_ADB_FALLBACK_IF_DIRECT_UNSUPPORTED"]
            reason = "Step requires direct settings operation without UI navigation"
        elif "confirm" in gl or "verify" in gl or "check" in gl:
            step_type = "VISUAL_CONFIRMATION"
            verification_mode = "VISUAL"
            minimal_action_path = ["VERIFY_EXPECTED_OUTCOME"]
            reason = "Step is verification-first"

        selected_strategy = {
            "DIRECT_KEY_VALIDATION": "DIRECT_KEY_VALIDATION",
            "APP_LAUNCH": "DIRECT_APP_LAUNCH",
            "SETTING_CHANGE": "DIRECT_SETTING_OPERATION",
        }.get(step_type, "UI_NAVIGATION_ONLY")

        return TaskPrePlan(
            goal=g,
            target_app=target_app,
            target_ui_context=target_ui_context,
            required_subgoals=required_subgoals,
            verification_condition=verification_condition,
            forbidden_detours=forbidden_detours,
            starting_context=starting_context,
            required_action=required_action,
            target_domain=target_domain,
            expected_outcome=expected_outcome,
            verification_mode=verification_mode,
            step_type=step_type,
            needs_app_launch=needs_app_launch,
            needs_settings_navigation=needs_settings_navigation,
            needs_home_first=needs_home_first,
            minimal_action_path=minimal_action_path,
            selected_strategy=selected_strategy,
            reason=reason,
        )

    @staticmethod
    def _is_home_context_satisfied(state: RunState) -> bool:
        current = str(state.current_app_id or state.current_app or "").lower()
        if not current:
            return False
        return current in {"launcher", "home", "com.google.android.tvlauncher"}

    def _strategy_matches_task_semantics(self, state: RunState) -> bool:
        preplan = state.task_preplan or {}
        step_type = self._normalize_step_type(preplan.get("step_type", ""))
        strategy = str(state.strategy_selected or "")
        if step_type == "DIRECT_KEY_VALIDATION" and strategy in {
            "DIRECT_APP_LAUNCH",
            "DIRECT_APP_LAUNCH_WITH_PARAMS",
            "GO_HOME_THEN_LAUNCH",
            "GO_HOME_AND_RECOVER",
            "RECOVERY_RELAUNCH",
            "RELAUNCH_TARGET_APP",
            "DIRECT_SETTING_OPERATION",
            "DIRECT_DAB_OPERATION",
        }:
            return False
        return True

    @staticmethod
    def _normalize_step_type(step_type: Any) -> str:
        raw = str(step_type or "").strip()
        if "." in raw:
            raw = raw.split(".")[-1]
        return raw.upper()

    @staticmethod
    def _is_settings_goal(goal: str) -> bool:
        g = (goal or "").lower()
        if "youtube" in g and any(k in g for k in ("stats for nerds", "gear icon", "player control", "video control")):
            return False
        return any(k in g for k in ("setting", "time zone", "timezone", "brightness", "contrast", "screensaver"))

    @staticmethod
    def _is_youtube_player_task(goal: str) -> bool:
        g = (goal or "").lower()
        return "youtube" in g and any(
            k in g for k in ("stats for nerds", "gear", "video control", "player settings", "play any video", "play video")
        )

    @staticmethod
    def _is_probably_video_playback_visible(state: RunState) -> bool:
        visual_summary = str(state.latest_visual_summary or "").lower()
        return any(k in visual_summary for k in ("pause", "seek", "settings", "stats for nerds", "up next"))

    @staticmethod
    def _visual_summary_fingerprint(visual_summary: str) -> str:
        normalized = " ".join(str(visual_summary or "").lower().split())
        return normalized[:220]

    @staticmethod
    def _focus_guess_from_visual_summary(visual_summary: str) -> Optional[str]:
        visual = str(visual_summary or "").lower()
        if "stats for nerds" in visual:
            return "stats for nerds"
        if "settings" in visual or "gear" in visual:
            return "settings gear"
        if "captions" in visual:
            return "captions"
        if "quality" in visual:
            return "quality"
        return None

    def _normalized_youtube_phase(self, state: RunState) -> str:
        mapping = {
            "OPEN_TARGET_APP": "OPEN_YOUTUBE",
            "START_ANY_VIDEO": "START_VIDEO",
            "REVEAL_PLAYER_CONTROLS": "REVEAL_PLAYER_CONTROLS",
            "OPEN_PLAYER_SETTINGS": "NAVIGATE_TO_GEAR",
            "ENABLE_STATS_FOR_NERDS": "OPEN_PLAYER_SETTINGS",
            "VERIFY_STATS_OVERLAY": "VERIFY_OVERLAY",
            "COMPLETE": "VERIFY_OVERLAY",
        }
        return mapping.get(self._youtube_phase(state), "START_VIDEO")

    def _refresh_player_context(self, state: RunState) -> None:
        visual_summary = str(state.latest_visual_summary or "")
        cfg = getattr(self, "_config", None)
        youtube_app_id = str(getattr(cfg, "youtube_app_id", "youtube") or "youtube").lower()
        state.focus_target_guess = self._focus_guess_from_visual_summary(visual_summary)
        state.player_controls_visible = self._is_player_controls_visible(state)
        state.is_video_playback_context = self._is_youtube_player_task(state.goal) and (
            self._is_probably_video_playback_visible(state)
            or state.player_controls_visible
            or str(state.current_app_id or state.current_app or "").lower() == youtube_app_id
        )
        if self._is_youtube_player_task(state.goal):
            state.last_player_phase = self._normalized_youtube_phase(state)

    def _diagnose_press_ok_effect(self, state: RunState, before_visual_summary: str) -> str:
        before = str(before_visual_summary or "").lower()
        after = str(state.latest_visual_summary or "").lower()
        if ("settings" not in before and "gear" not in before) and ("settings" in after or "gear" in after):
            return "CONTROLS_REVEALED"
        if "stats for nerds" not in before and "stats for nerds" in after:
            return "MENU_OPENED_OR_TOGGLE_VISIBLE"
        if ("pause" in before and "play" in after) or ("play" in before and "pause" in after):
            return "TOGGLED_PLAYBACK"
        if self._visual_summary_fingerprint(before) == self._visual_summary_fingerprint(after):
            return "NO_VISIBLE_CHANGE"
        return "SCREEN_CHANGED"

    def _update_no_progress_tracking(self, state: RunState, action: str, action_success: bool) -> None:
        action_u = str(action or "").upper()
        current_fingerprint = self._visual_summary_fingerprint(state.latest_visual_summary or "")
        phase = self._normalized_youtube_phase(state) if self._is_youtube_player_task(state.goal) else ""
        same_fingerprint = bool(current_fingerprint and state.last_screen_fingerprint == current_fingerprint)
        same_phase = bool(phase and state.last_player_phase == phase)

        if action_u in {ActionType.PRESS_OK.value, "PRESS_ENTER"}:
            state.repeated_commit_count = state.repeated_commit_count + 1 if same_fingerprint and same_phase else 1
        else:
            state.repeated_commit_count = 0

        if action_success and action_u in {ActionType.PRESS_OK.value, "PRESS_ENTER"} and same_fingerprint and same_phase:
            state.no_progress_count += 1
        elif not same_fingerprint or not same_phase:
            state.no_progress_count = 0

        state.last_screen_fingerprint = current_fingerprint or state.last_screen_fingerprint
        if phase:
            state.last_player_phase = phase

    def _build_navigation_memory_summary(self, state: RunState) -> dict[str, Any]:
        recent = list(state.navigation_memory[-8:])
        recent_actions = [str(item.get("action", "")) for item in recent if isinstance(item, dict)]
        no_progress_actions = [
            str(item.get("action", ""))
            for item in recent
            if isinstance(item, dict) and bool(item.get("focus_changed")) is False
        ]
        return {
            "recent_steps": recent,
            "recent_actions": recent_actions,
            "failed_paths": list(state.failed_paths[-8:]),
            "blocked_actions": list(state.blocked_actions[-8:]),
            "repeated_no_progress_actions": no_progress_actions[-5:],
            "last_known_ui_region": str(state.focus_target_guess or state.current_screen_guess or state.current_screen or "unknown"),
            "last_focus_change_detected": state.last_focus_change_detected,
            "strategy_selected": state.strategy_selected,
        }

    def _record_navigation_memory(self, state: RunState, planned: PlannedAction, action_success: bool) -> None:
        prev_fp = str(state.last_screen_fingerprint or "")
        current_fp = self._visual_summary_fingerprint(state.latest_visual_summary or "")
        focus_changed = bool(current_fp and prev_fp and current_fp != prev_fp)
        if not prev_fp:
            focus_changed = action_success
        state.last_focus_change_detected = focus_changed

        entry = {
            "step": state.step_count,
            "action": str(planned.action),
            "reason": str(planned.reason or ""),
            "params": dict(planned.params or {}),
            "success": bool(action_success),
            "focus_changed": bool(focus_changed),
            "ui_region": str(state.focus_target_guess or state.current_screen or "unknown"),
            "screen": str(state.current_screen or state.current_app_state or "unknown"),
        }
        state.navigation_memory.append(entry)
        if len(state.navigation_memory) > 120:
            state.navigation_memory = state.navigation_memory[-120:]

    @staticmethod
    def _is_player_controls_visible(state: RunState) -> bool:
        visual_summary = str(state.latest_visual_summary or "").lower()
        return any(k in visual_summary for k in ("settings", "gear", "captions", "quality", "playback speed", "cc"))

    def _youtube_phase(self, state: RunState) -> str:
        cfg = getattr(self, "_config", None)
        youtube_app_id = str(getattr(cfg, "youtube_app_id", "youtube") or "youtube").lower()
        current_app = str(state.current_app_id or state.current_app or "").lower()
        goal = (state.goal or "").lower()
        visual_summary = str(state.latest_visual_summary or "").lower()

        if current_app != youtube_app_id:
            return "OPEN_TARGET_APP"
        if not self._is_probably_video_playback_visible(state):
            return "START_ANY_VIDEO"
        if not self._is_player_controls_visible(state):
            return "REVEAL_PLAYER_CONTROLS"
        if "stats for nerds" in goal:
            if "stats for nerds" in visual_summary:
                return "COMPLETE"
            if "settings" not in visual_summary and "gear" not in visual_summary:
                return "OPEN_PLAYER_SETTINGS"
            if "stats for nerds" not in visual_summary:
                return "ENABLE_STATS_FOR_NERDS"
            return "VERIFY_STATS_OVERLAY"
        if "play" in goal and "video" in goal:
            return "COMPLETE"
        return "VERIFY_STATS_OVERLAY"

    def _is_youtube_goal_verified(self, state: RunState) -> bool:
        cfg = getattr(self, "_config", None)
        youtube_app_id = str(getattr(cfg, "youtube_app_id", "youtube") or "youtube").lower()
        current_app = str(state.current_app_id or state.current_app or "").lower()
        if current_app != youtube_app_id:
            return False
        if str(state.current_app_state or "").upper() != "FOREGROUND":
            return False
        goal = (state.goal or "").lower()
        visual_summary = str(state.latest_visual_summary or "").lower()
        if "stats for nerds" in goal:
            return "stats for nerds" in visual_summary
        if "play" in goal and "video" in goal:
            return self._is_probably_video_playback_visible(state)
        return True

    def _sanitize_planned_action_for_goal(self, state: RunState, planned: PlannedAction) -> PlannedAction:
        action = str(planned.action).upper()
        params = planned.params or {}

        if self._is_settings_goal(state.goal):
            allowed_actions = {
                ActionType.GET_SETTING.value,
                ActionType.SET_SETTING.value,
                ActionType.FAILED.value,
                ActionType.DONE.value,
            }
            if action not in allowed_actions:
                return PlannedAction(
                    action=ActionType.FAILED,
                    confidence=max(0.85, planned.confidence),
                    reason=(
                        "UI navigation is disabled for settings-related operations; "
                        "only direct settings operations are allowed "
                        "(GET_SETTING/SET_SETTING with DAB or Android ADB fallback)."
                    ),
                )

        # Global ENTER/OK safety gate: never commit with weak confidence.
        if action in {ActionType.PRESS_OK.value, "PRESS_ENTER"} and float(planned.confidence) < 0.72:
            state.record_ai_event(
                {
                    "type": "enter-gate-blocked",
                    "step": state.step_count,
                    "confidence": float(planned.confidence),
                    "reason": "low confidence commit prevented",
                }
            )
            if self._has_usable_visual_summary(state):
                return PlannedAction(
                    action=ActionType.PRESS_RIGHT,
                    confidence=0.72,
                    reason="ENTER blocked at low confidence; making one safe focus move before re-check",
                )
            return PlannedAction(
                action=ActionType.CAPTURE_SCREENSHOT,
                confidence=0.74,
                reason="ENTER blocked at low confidence; capturing screenshot for safer next move",
            )

        # If the same directional action repeatedly produced no visible change,
        # adjust strategy instead of repeating blind moves.
        if action in {ActionType.PRESS_UP.value, ActionType.PRESS_DOWN.value, ActionType.PRESS_LEFT.value, ActionType.PRESS_RIGHT.value}:
            recent = [item for item in state.navigation_memory[-4:] if isinstance(item, dict)]
            same_no_progress = [
                item for item in recent
                if str(item.get("action", "")).upper() == action and bool(item.get("focus_changed")) is False
            ]
            if len(same_no_progress) >= 2:
                self._record_strategy_transition(
                    state,
                    new_strategy="UI_NAVIGATION_FALLBACK",
                    reason="Repeated no-progress directional moves; adapting navigation strategy",
                )
                state.record_ai_event(
                    {
                        "type": "navigation-strategy-adjusted",
                        "step": state.step_count,
                        "blocked_action": action,
                        "reason": "same move produced no progress repeatedly",
                    }
                )
                return PlannedAction(
                    action=ActionType.CAPTURE_SCREENSHOT,
                    confidence=0.78,
                    reason="Repeated directional no-progress detected; re-checking UI before next move",
                )

        if action in {ActionType.GET_SETTING.value, ActionType.SET_SETTING.value}:
            setting_key = str(params.get("key", "")).strip() or None
            operation = "system/settings/get" if action == ActionType.GET_SETTING.value else "system/settings/set"
            method, reason = self._resolve_setting_execution_method(state, operation, setting_key)
            if method in {"dab", "adb"}:
                return planned
            return PlannedAction(
                action=ActionType.FAILED,
                confidence=max(0.8, planned.confidence),
                reason=(
                    f"Unsupported settings action '{operation}' for '{setting_key or 'unknown'}': {reason}. "
                    "UI navigation fallback is disabled."
                ),
            )

        if action == ActionType.NEED_BETTER_VIEW.value:
            repeat = 0
            for a in reversed(state.last_actions):
                if str(a).upper() == ActionType.NEED_BETTER_VIEW.value:
                    repeat += 1
                else:
                    break
            if repeat >= 2:
                state.record_ai_event(
                    {
                        "type": "anti-flake-guard",
                        "step": state.step_count,
                        "reason": "Blocked repeated NEED_BETTER_VIEW loop",
                        "repeat": repeat,
                    }
                )
                return PlannedAction(
                    action=ActionType.GET_STATE,
                    confidence=0.78,
                    reason="Anti-flake guard: repeated NEED_BETTER_VIEW; forcing grounded state probe",
                    params={"app_id": state.current_app_id or state.current_app or self._config.youtube_app_id},
                )

        if self._is_youtube_player_task(state.goal):
            self._refresh_player_context(state)
            if state.is_video_playback_context and action in {ActionType.PRESS_OK.value, "PRESS_ENTER"}:
                ok_intent = str(params.get("ok_intent", "")).strip().upper()
                if not ok_intent:
                    state.record_ai_event(
                        {
                            "type": "commit-guard-blocked",
                            "step": state.step_count,
                            "reason": "Blocked blind OK in playback context",
                            "blocked_action": action,
                            "phase": state.last_player_phase,
                        }
                    )
                    return PlannedAction(
                        action=ActionType.CAPTURE_SCREENSHOT,
                        confidence=0.84,
                        reason="Playback context requires explicit OK intent before commit",
                    )

                if ok_intent == "REVEAL_PLAYER_CONTROLS" and (
                    state.player_controls_visible or state.repeated_commit_count >= 1
                ):
                    state.record_ai_event(
                        {
                            "type": "commit-guard-blocked",
                            "step": state.step_count,
                            "reason": "Blocked repeated OK while revealing player controls",
                            "blocked_action": action,
                            "ok_intent": ok_intent,
                            "repeated_commit_count": state.repeated_commit_count,
                        }
                    )
                    return PlannedAction(
                        action=ActionType.NEED_PLAYER_CONTROLS_VISIBLE,
                        confidence=0.83,
                        reason="One-OK reveal rule enforced; verify controls before next commit",
                    )

                if ok_intent in {"SELECT_FOCUSED_CONTROL", "CONFIRM_MENU_ITEM"}:
                    focus = str(state.focus_target_guess or "").lower()
                    if "settings" not in focus and "stats for nerds" not in focus:
                        state.record_ai_event(
                            {
                                "type": "commit-guard-blocked",
                                "step": state.step_count,
                                "reason": "Focus uncertain before OK commit",
                                "blocked_action": action,
                                "ok_intent": ok_intent,
                                "focus_target_guess": state.focus_target_guess,
                            }
                        )
                        if not self._has_usable_visual_summary(state):
                            return PlannedAction(
                                action=ActionType.CAPTURE_SCREENSHOT,
                                confidence=0.82,
                                reason="Focus-before-select rule: visual summary missing, capture screenshot before navigation",
                            )
                        return PlannedAction(
                            action=ActionType.PRESS_RIGHT,
                            confidence=0.8,
                            reason="Focus-before-select rule: move focus toward settings gear before OK",
                        )

                if state.repeated_commit_count >= 2 or state.no_progress_count >= 2:
                    state.record_ai_event(
                        {
                            "type": "commit-guard-blocked",
                            "step": state.step_count,
                            "reason": "Repeated commit without progress",
                            "blocked_action": action,
                            "ok_intent": ok_intent,
                            "repeated_commit_count": state.repeated_commit_count,
                            "no_progress_count": state.no_progress_count,
                            "last_ok_effect": state.last_ok_effect,
                        }
                    )
                    return PlannedAction(
                        action=ActionType.CAPTURE_SCREENSHOT,
                        confidence=0.86,
                        reason="Repeated OK/ENTER without progress; forcing screenshot-grounded re-plan",
                    )

            if action == ActionType.LAUNCH_APP.value:
                app_id = str(params.get("app_id", "")).lower()
                cfg = getattr(self, "_config", None)
                youtube_app_id = str(getattr(cfg, "youtube_app_id", "youtube") or "youtube").lower()
                if app_id and app_id != youtube_app_id:
                    state.record_ai_event(
                        {
                            "type": "detour-blocked",
                            "step": state.step_count,
                            "reason": "Blocked unrelated app launch for YouTube player task",
                            "blocked_action": action,
                            "blocked_app": app_id,
                        }
                    )
                    return PlannedAction(
                        action=ActionType.NEED_VIDEO_PLAYBACK_CONFIRMED,
                        confidence=0.7,
                        reason="Blocked unrelated app detour; re-focus on YouTube playback",
                    )
            if action == ActionType.PRESS_BACK.value and self._consecutive_back_count(state) >= 2:
                state.record_ai_event(
                    {
                        "type": "detour-blocked",
                        "step": state.step_count,
                        "reason": "Blocked repeated BACK loop without grounded diagnosis",
                        "blocked_action": action,
                    }
                )
                return PlannedAction(
                    action=ActionType.CAPTURE_SCREENSHOT,
                    confidence=0.75,
                    reason="Mandatory screenshot re-evaluation after bounded BACK usage",
                )
        return planned

    @staticmethod
    def _consecutive_back_count(state: RunState) -> int:
        count = 0
        for a in reversed(state.last_actions):
            if str(a).upper() == ActionType.PRESS_BACK.value:
                count += 1
            else:
                break
        return count

    def _infer_target_app_name_from_goal(self, goal: str, app_catalog: list[dict]) -> Optional[str]:
        g = (goal or "").strip().lower()
        if not g:
            return None
        if self._is_settings_goal(goal):
            return "Settings"

        open_words = ("open", "launch", "start")
        if not any(w in g for w in open_words):
            return None
        best_name: Optional[str] = None
        best_score = 0.0
        goal_tokens = {t for t in g.replace("_", " ").split() if t}
        for app in app_catalog or []:
            if not isinstance(app, dict):
                continue
            for key in ("friendlyName", "name", "appId"):
                value = str(app.get(key, "")).strip()
                if not value:
                    continue
                if value.lower() in g:
                    return value
                val_tokens = {t for t in value.lower().replace("_", " ").split() if t}
                if not val_tokens:
                    continue
                overlap = len(goal_tokens.intersection(val_tokens)) / float(len(val_tokens))
                if overlap > best_score:
                    best_score = overlap
                    best_name = value
        if best_score >= 0.5:
            return best_name
        return None

    def _infer_setting_key_from_goal(self, goal: str, supported_settings: list[dict]) -> Optional[str]:
        g = (goal or "").lower()
        candidates: list[str] = []
        aliases: dict[str, str] = {}
        for item in supported_settings or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key", "")).strip()
            friendly = str(item.get("friendlyName", "")).strip().lower()
            if key and (key.lower() in g or (friendly and friendly in g)):
                return key
            if key:
                candidates.append(key)
                aliases[key.lower()] = key
            if friendly and key:
                aliases[friendly] = key
        if "time zone" in g or "timezone" in g:
            return "timezone"
        if "language" in g or "locale" in g:
            return "language"
        if "brightness" in g:
            for key in candidates:
                if "bright" in key.lower():
                    return key

        goal_token = " ".join([tok for tok in re.split(r"[^a-z0-9]+", g) if tok])
        if goal_token and aliases:
            best = difflib.get_close_matches(goal_token, list(aliases.keys()), n=1, cutoff=0.72)
            if best:
                return aliases[best[0]]

            best_key = None
            best_score = 0.0
            for alias_text, resolved_key in aliases.items():
                score = difflib.SequenceMatcher(a=goal_token, b=alias_text).ratio()
                if score > best_score:
                    best_score = score
                    best_key = resolved_key
            if best_key and best_score >= 0.64:
                return best_key
        return None

    @staticmethod
    def _infer_setting_value_from_goal(goal: str) -> Optional[str]:
        g = (goal or "").strip()
        lower = g.lower()
        marker = " to "
        idx = lower.find(marker)
        if idx < 0:
            return None
        return g[idx + len(marker):].strip() or None

    @staticmethod
    def _is_open_app_goal(goal: str) -> bool:
        g = (goal or "").lower()
        explicit_app_intent = any(w in g for w in ("open", "launch", "start")) and "app" in g
        named_app_intent = any(name in g for name in ("youtube", "netflix", "prime video", "settings"))
        return explicit_app_intent or named_app_intent

    @staticmethod
    def _is_app_goal_verified(state: RunState) -> bool:
        app_state = str(state.current_app_state or "").upper()
        if app_state != "FOREGROUND":
            return False
        return bool(state.current_app_id or state.current_app or state.last_verified_foreground_app)

    async def _normalize_navigation_batch(
        self,
        state: RunState,
        nav_action_batch: list[dict],
        planner_output: Optional[dict],
    ) -> tuple[list[dict], int]:
        normalized: list[dict] = []
        failures = 0
        for item in nav_action_batch:
            action = str((item or {}).get("action", "")).strip() or ActionType.WAIT.value
            params = (item or {}).get("params")
            params_dict = params if isinstance(params, dict) else {}
            next_action, next_params, err = await self._normalize_launch_action(
                state=state,
                action=action,
                params=params_dict,
                planner_output=planner_output,
            )
            if err:
                failures += 1
                logger.warning("Launch target resolution failed in planner batch: %s", err)
                state.record_ai_event(
                    {
                        "type": "launch-resolution-failed",
                        "step": state.step_count,
                        "source": "planner",
                        "error": err,
                        "fallback_action": ActionType.NEED_BETTER_VIEW.value,
                    }
                )
            normalized.append({"action": next_action, "params": next_params or {}})
        return normalized, failures

    async def _normalize_launch_action(
        self,
        state: RunState,
        action: str,
        params: dict,
        planner_output: Optional[dict],
    ) -> tuple[str, dict, Optional[str]]:
        if str(action).upper() != ActionType.LAUNCH_APP.value:
            return action, params, None

        app_id = str((params or {}).get("app_id", "")).strip()
        if app_id:
            return action, params, None

        merged_output: dict[str, Any] = dict(planner_output or {})
        merged_output.setdefault("target_app_name", params.get("target_app_name") or params.get("app_name"))
        merged_output.setdefault("target_app_domain", params.get("target_app_domain"))
        merged_output.setdefault("target_app_hint", params.get("target_app_hint") or params.get("app_name"))
        launch_parameters = dict(merged_output.get("launch_parameters") or {})
        for k, v in (params or {}).items():
            if k not in {"app_id", "app_name", "target_app_name", "target_app_domain", "target_app_hint", "launch_mode"}:
                launch_parameters.setdefault(k, v)
        merged_output["launch_parameters"] = launch_parameters

        resolved = await self._app_resolver.resolve_target_app(
            goal=state.goal,
            planner_output=merged_output,
            execution_state={
                "session_id": state.run_id,
                "device_id": self._config.dab_device_id,
            },
        )
        if resolved is None:
            target_debug = (
                params.get("app_name")
                or params.get("target_app_name")
                or merged_output.get("target_app_name")
                or merged_output.get("target_app_hint")
                or "unknown"
            )
            state.resolution_failures += 1
            return self._targeted_ambiguity_action(state), {}, (
                f"Could not resolve launch target '{target_debug}' from applications/list"
            )

        normalized_params = dict(launch_parameters)
        normalized_params["app_id"] = resolved.app_id
        return ActionType.LAUNCH_APP.value, normalized_params, None

    def _targeted_ambiguity_action(self, state: RunState) -> str:
        if self._is_youtube_player_task(state.goal):
            visual_summary = str(state.latest_visual_summary or "").lower()
            if not self._is_probably_video_playback_visible(state):
                return ActionType.NEED_VIDEO_PLAYBACK_CONFIRMED.value
            if "settings" not in visual_summary and "gear" not in visual_summary:
                return ActionType.NEED_SETTINGS_GEAR_LOCATION.value
            if "stats for nerds" not in visual_summary:
                return ActionType.NEED_PLAYER_MENU_CONFIRMATION.value
            return ActionType.NEED_STATS_FOR_NERDS_TOGGLE_CONFIRMATION.value
        return ActionType.NEED_BETTER_VIEW.value

    async def _execute_action(self, state: RunState, planned: PlannedAction) -> bool:
        """Execute the planned action via DAB and return whether it succeeded."""
        action = planned.action
        params = planned.params or {}

        if action in {ActionType.GET_SETTING, ActionType.SET_SETTING}:
            # Hard preflight for settings operations: consult operations/list and
            # settings/list before attempting direct or fallback paths.
            if (not state.capability_preflight_done or not state.supported_operations) and hasattr(self, "_app_resolver"):
                try:
                    await self._bootstrap_capabilities_if_needed(state)
                except Exception as exc:
                    logger.warning("Settings capability preflight failed before execution: %s", exc)

        def _log_req(op: str, payload: dict) -> None:
            state.record_dab_event({
                "type": "request",
                "step": state.step_count,
                "op": op,
                "payload": payload,
            })

        def _log_resp(op: str, resp) -> None:
            state.record_dab_event({
                "type": "response",
                "step": state.step_count,
                "op": op,
                "success": bool(getattr(resp, "success", False)),
                "status": int(getattr(resp, "status", 0) or 0),
                "topic": getattr(resp, "topic", ""),
                "request_id": getattr(resp, "request_id", ""),
                "data": getattr(resp, "data", {}) or {},
            })

        try:
            snapshot = state.capability_snapshot or build_capability_snapshot(
                supported_operations=state.supported_operations,
                supported_settings=state.supported_settings,
                supported_keys=state.supported_keys,
            )
            if action in KEY_MAP:
                key_code = KEY_MAP[action]
                if not has_operation(snapshot, "input/key-press"):
                    logger.warning("Blocked key action before DAB: input/key-press unsupported action=%s", action)
                    return False
                if state.supported_keys and key_code not in set(state.supported_keys):
                    logger.warning("Skipping unsupported key action=%s key_code=%s", action, key_code)
                    return False
                if (snapshot.get("supported_keys") or []) and not has_key(snapshot, key_code):
                    logger.warning("Blocked key action before DAB: key unsupported key_code=%s", key_code)
                    return False
                _log_req("input/key-press", {"keyCode": key_code})
                resp = await self._dab.key_press(key_code)
                _log_resp("input/key-press", resp)
                return resp.success
            elif action == ActionType.LAUNCH_APP:
                if not has_operation(snapshot, "applications/launch"):
                    logger.warning("Blocked launch before DAB: applications/launch unsupported")
                    return False
                app_id = params.get("app_id", "")
                if not app_id:
                    logger.warning("LAUNCH_APP missing app_id param")
                    return False
                launch_parameters = {}
                content = str(params.get("content", "")).strip()
                if content:
                    launch_parameters["content"] = content
                _log_req(
                    "applications/launch",
                    {"appId": app_id, **({"content": content} if content else {})},
                )
                resp = await self._dab.launch_app(app_id, parameters=launch_parameters or None)
                _log_resp("applications/launch", resp)
                if resp.success:
                    state.current_app = app_id
                    state.current_app_id = app_id
                return resp.success
            elif str(action).upper() == "OPEN_CONTENT":
                content = str(params.get("content", "")).strip()
                if not content:
                    return False
                op = "content/open"
                _log_req(op, {"content": content})
                resp = await self._dab.open_content(content, parameters=params)
                _log_resp(op, resp)
                return resp.success
            elif action == ActionType.GET_SETTING:
                setting_key = str(params.get("key", "")).strip()
                if not setting_key:
                    return False
                normalized_key = normalize_setting_key(snapshot, setting_key)
                if bool(normalized_key.get("success")):
                    setting_key = str(normalized_key.get("key") or setting_key)
                elif self._is_timezone_setting_key(setting_key):
                    setting_key = "timezone"
                elif self._can_attempt_android_setting_adb_fallback(state, "system/settings/get", setting_key):
                    setting_key = self._normalize_adb_setting_key(setting_key)
                else:
                    logger.warning("Blocked GET_SETTING before DAB: %s", normalized_key.get("reason"))
                    return False
                operation = "system/settings/get"
                method, reason = self._resolve_setting_execution_method(state, operation, setting_key)
                logger.info(
                    "Requested operation=%s key=%s is_android=%s selected_method=%s reason=%s",
                    operation,
                    setting_key,
                    state.is_android_device,
                    method,
                    reason,
                )

                if method == "dab":
                    _log_req(operation, {"key": setting_key})
                    resp = await self._dab.get_setting(setting_key)
                    _log_resp(operation, resp)
                    if resp.success:
                        return True
                    self._record_direct_setting_failure(state, operation, setting_key, resp)
                    method, reason = self._resolve_setting_execution_method(state, operation, setting_key)
                    if method != "adb":
                        return False

                if method != "adb":
                    logger.warning(
                        "Unsupported setting read: key=%s reason=%s is_android=%s",
                        setting_key,
                        reason,
                        state.is_android_device,
                    )
                    state.record_ai_event(
                        {
                            "type": "settings-operation-unsupported",
                            "step": state.step_count,
                            "operation": operation,
                            "setting_key": setting_key,
                            "reason": reason,
                        }
                    )
                    return False

                adb_device_id = str(state.android_adb_device_id or "").strip()
                online, online_detail = await is_adb_device_online(adb_device_id)
                if not online:
                    logger.warning("ADB read fallback unavailable for %s: %s", adb_device_id, online_detail)
                    return False
                if self._is_timezone_setting_key(setting_key):
                    _log_req(
                        "adb/system/settings/timezone/get",
                        {"device_id": adb_device_id, "key": "timezone"},
                    )
                    fallback_read = await get_timezone_via_adb(adb_device_id)
                    adb_op = "adb/system/settings/timezone/get"
                else:
                    _log_req(
                        "adb/system/settings/get",
                        {"device_id": adb_device_id, "key": setting_key},
                    )
                    fallback_read = await get_setting_via_adb(adb_device_id, setting_key)
                    namespace = str(fallback_read.get("namespace") or "unknown")
                    adb_op = f"adb/system/settings/{namespace}/{setting_key}/get"
                success = bool(fallback_read.get("success"))
                state.record_dab_event(
                    {
                        "type": "response",
                        "step": state.step_count,
                        "op": adb_op,
                        "success": success,
                        "status": 200 if success else 500,
                        "data": fallback_read,
                    }
                )
                return success
            elif action == ActionType.SET_SETTING:
                setting_key = str(params.get("key", "")).strip()
                if not setting_key:
                    return False
                normalized_key = normalize_setting_key(snapshot, setting_key)
                if bool(normalized_key.get("success")):
                    setting_key = str(normalized_key.get("key") or setting_key)
                elif self._is_timezone_setting_key(setting_key):
                    setting_key = "timezone"
                elif self._can_attempt_android_setting_adb_fallback(state, "system/settings/set", setting_key):
                    setting_key = self._normalize_adb_setting_key(setting_key)
                else:
                    logger.warning("Blocked SET_SETTING before DAB: %s", normalized_key.get("reason"))
                    return False

                requested_value = params.get("value")
                operation = "system/settings/set"
                method, reason = self._resolve_setting_execution_method(state, operation, setting_key)
                logger.info(
                    "Requested operation=%s key=%s is_android=%s selected_method=%s reason=%s",
                    operation,
                    setting_key,
                    state.is_android_device,
                    method,
                    reason,
                )

                if method == "dab":
                    if has_setting(snapshot, setting_key):
                        value_norm = normalize_setting_value(snapshot, setting_key, requested_value)
                        if bool(value_norm.get("success")):
                            requested_value = value_norm.get("value")
                    _log_req(operation, {"key": setting_key, "value": requested_value})
                    resp = await self._dab.set_setting(setting_key, requested_value)
                    _log_resp(operation, resp)
                    if resp.success:
                        if self._is_timezone_setting_key(setting_key):
                            logger.info("Timezone setting path used: DAB direct operation succeeded run_id=%s key=%s", state.run_id, setting_key)
                        return True
                    self._record_direct_setting_failure(state, operation, setting_key, resp)
                    method, reason = self._resolve_setting_execution_method(state, operation, setting_key)
                    if method != "adb":
                        return False

                if method != "adb":
                    logger.warning(
                        "Unsupported setting write: key=%s reason=%s is_android=%s",
                        setting_key,
                        reason,
                        state.is_android_device,
                    )
                    state.record_ai_event(
                        {
                            "type": "settings-operation-unsupported",
                            "step": state.step_count,
                            "operation": operation,
                            "setting_key": setting_key,
                            "reason": reason,
                        }
                    )
                    return False

                tz_value = str(requested_value or "").strip()
                if not tz_value:
                    logger.warning("Skipping Android timezone fallback: empty timezone value")
                    return False

                adb_device_id = str(state.android_adb_device_id or "").strip()
                logger.info(
                    "Timezone fallback selected: path=ADB reason=%s run_id=%s adb_device_id=%s",
                    reason,
                    state.run_id,
                    adb_device_id,
                )
                online, online_detail = await is_adb_device_online(adb_device_id)
                if not online:
                    logger.warning(
                        "Skipping Android timezone fallback: adb device offline/unavailable adb_device_id=%s detail=%s",
                        adb_device_id,
                        online_detail,
                    )
                    state.record_ai_event(
                        {
                            "type": "timezone-adb-fallback-skipped",
                            "step": state.step_count,
                            "reason": "adb device offline/unavailable",
                            "adb_device_id": adb_device_id,
                            "detail": online_detail,
                        }
                    )
                    return False

                if not self._is_timezone_setting_key(setting_key):
                    _log_req(
                        "adb/system/settings/set",
                        {
                            "device_id": adb_device_id,
                            "key": setting_key,
                            "value": str(requested_value if requested_value is not None else ""),
                        },
                    )
                    fallback_result = await set_setting_via_adb(adb_device_id, setting_key, requested_value)
                    verified = bool(fallback_result.get("success"))
                    state.record_dab_event(
                        {
                            "type": "response",
                            "step": state.step_count,
                            "op": f"adb/system/settings/{setting_key}",
                            "success": verified,
                            "status": 200 if verified else 500,
                            "data": fallback_result,
                        }
                    )
                    state.record_ai_event(
                        {
                            "type": "setting-adb-fallback-success" if verified else "setting-adb-fallback-failed",
                            "step": state.step_count,
                            "setting_key": setting_key,
                            "requested_value": str(requested_value if requested_value is not None else ""),
                            "observed_value": fallback_result.get("observed_value"),
                            "error": fallback_result.get("error"),
                        }
                    )
                    return verified

                timezone_list = await list_timezones_via_adb(adb_device_id)
                tz_to_apply = tz_value
                if not bool(timezone_list.get("success")):
                    logger.warning(
                        "ADB timezone listing failed but proceeding with direct set: adb_device_id=%s error=%s",
                        adb_device_id,
                        timezone_list.get("error"),
                    )
                else:
                    ai_client = getattr(self._planner, "_vertex_client", None)
                    resolved = await resolve_timezone_from_supported(
                        tz_value,
                        list(timezone_list.get("timezones") or []),
                        ai_client=ai_client,
                    )
                    if not bool(resolved.get("success")):
                        logger.warning(
                            "Requested timezone could not be mapped to supported IDs: requested=%s reason=%s",
                            tz_value,
                            resolved.get("reason"),
                        )
                        state.record_ai_event(
                            {
                                "type": "timezone-adb-fallback-failed",
                                "step": state.step_count,
                                "requested_timezone": tz_value,
                                "error": str(resolved.get("reason") or "unsupported timezone on device"),
                            }
                        )
                        return False
                    tz_to_apply = str(resolved.get("resolved_timezone") or tz_value)
                    logger.info(
                        "Timezone normalization: requested=%s resolved=%s strategy=%s",
                        tz_value,
                        tz_to_apply,
                        resolved.get("reason"),
                    )

                _log_req(
                    "adb/system/settings/timezone",
                    {
                        "device_id": adb_device_id,
                        "key": "timezone",
                        "requested_timezone": tz_value,
                        "resolved_timezone": tz_to_apply,
                    },
                )
                fallback_result = await set_timezone_via_adb(adb_device_id, tz_to_apply)
                verified = bool(fallback_result.get("success"))
                logger.info(
                    "ADB timezone fallback verification: adb_device_id=%s requested=%s observed=%s verified=%s",
                    adb_device_id,
                    fallback_result.get("requested_timezone") or tz_to_apply,
                    fallback_result.get("observed_timezone"),
                    verified,
                )
                state.record_dab_event(
                    {
                        "type": "response",
                        "step": state.step_count,
                        "op": "adb/system/settings/timezone",
                        "success": verified,
                        "status": 200 if verified else 500,
                        "data": fallback_result,
                    }
                )
                state.record_ai_event(
                    {
                        "type": "timezone-adb-fallback-success" if verified else "timezone-adb-fallback-failed",
                        "step": state.step_count,
                        "requested_timezone": tz_value,
                        "observed_timezone": fallback_result.get("observed_timezone"),
                        "error": fallback_result.get("error"),
                    }
                )
                return verified
            elif action == ActionType.GET_STATE:
                app_id = (
                    state.current_app
                    or params.get("app_id", "")
                    or self._config.youtube_app_id
                )
                _log_req("applications/get-state", {"appId": app_id})
                resp = await self._dab.get_app_state(app_id)
                _log_resp("applications/get-state", resp)
                if resp.success:
                    state.current_app = resp.data.get("appId", state.current_app)
                    state.current_app_id = resp.data.get("appId", state.current_app_id)
                    state.current_screen = resp.data.get("state")
                    state.current_app_state = str(resp.data.get("state", ""))
                    if state.current_app_state.upper() == "FOREGROUND" and state.current_app_id:
                        state.last_verified_foreground_app = state.current_app_id
                return resp.success
            elif action in (
                ActionType.CAPTURE_SCREENSHOT,
                ActionType.NEED_BETTER_VIEW,
                ActionType.NEED_PLAYER_CONTROLS_VISIBLE,
                ActionType.NEED_VIDEO_PLAYBACK_CONFIRMED,
                ActionType.NEED_SETTINGS_GEAR_LOCATION,
                ActionType.NEED_PLAYER_MENU_CONFIRMATION,
                ActionType.NEED_STATS_FOR_NERDS_TOGGLE_CONFIRMATION,
            ):
                await self._update_capture(state)
                self._refresh_player_context(state)
                if action == ActionType.CAPTURE_SCREENSHOT:
                    return state.latest_screenshot_b64 is not None
                if action == ActionType.NEED_BETTER_VIEW:
                    return self._has_usable_visual_summary(state)
                if action == ActionType.NEED_PLAYER_CONTROLS_VISIBLE:
                    return self._is_player_controls_visible(state)
                if action == ActionType.NEED_VIDEO_PLAYBACK_CONFIRMED:
                    return self._is_probably_video_playback_visible(state)
                if action == ActionType.NEED_SETTINGS_GEAR_LOCATION:
                    focus = str(state.focus_target_guess or "").lower()
                    return "settings" in focus or "gear" in focus
                if action == ActionType.NEED_PLAYER_MENU_CONFIRMATION:
                    visual_summary = str(state.latest_visual_summary or "").lower()
                    return any(k in visual_summary for k in ("stats for nerds", "quality", "captions", "playback speed"))
                if action == ActionType.NEED_STATS_FOR_NERDS_TOGGLE_CONFIRMATION:
                    visual_summary = str(state.latest_visual_summary or "").lower()
                    return "stats for nerds" in visual_summary
                return False
            elif action == ActionType.WAIT:
                seconds = params.get("seconds", 1)
                await asyncio.sleep(float(seconds))
                return True
            else:
                logger.warning("Unknown action type: %s", action)
                return False
        except Exception as exc:
            logger.error("Action execution failed: action=%s error=%s", action, exc)
            state.record_dab_event({
                "type": "error",
                "step": state.step_count,
                "op": str(action),
                "error": str(exc),
            })
            return False

    async def _update_capture(self, state: RunState) -> None:
        """Capture a new screenshot and update state in-place."""
        capture = await self._capture.capture()
        state.latest_screenshot_b64 = capture.image_b64
        state.latest_visual_summary = capture.ocr_text
        self._refresh_observation_features(state)
