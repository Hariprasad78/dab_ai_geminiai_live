package com.dabcontrol.app.ui.runs

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.api.TaskMacroRequestDto
import com.dabcontrol.app.data.repo.RunsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

@HiltViewModel
class RunsListViewModel @Inject constructor(
    private val runsRepository: RunsRepository
) : ViewModel() {
    private val _uiState = MutableStateFlow(RunsListUiState())
    val uiState: StateFlow<RunsListUiState> = _uiState.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            val runsResult = runsRepository.fetchRuns()
            val devicesResult = runsRepository.fetchDevices()

            val current = _uiState.value
            val next = when (runsResult) {
                is ApiResult.Success -> current.copy(items = runsResult.data)
                else -> current.copy(error = runsResult.errorText())
            }
            _uiState.value = when (devicesResult) {
                is ApiResult.Success -> {
                    val ids = devicesResult.data.devices.mapNotNull { device ->
                        device["id"]?.jsonPrimitive?.contentOrNull
                            ?: device["device_id"]?.jsonPrimitive?.contentOrNull
                            ?: device["serial"]?.jsonPrimitive?.contentOrNull
                    }.distinct()
                    next.copy(
                        isLoading = false,
                        deviceIds = ids,
                        selectedDeviceId = next.selectedDeviceId.ifBlank {
                            devicesResult.data.selected_device_id ?: ids.firstOrNull().orEmpty()
                        }
                    )
                }
                else -> next.copy(isLoading = false, error = next.error ?: devicesResult.errorText())
            }
        }
    }

    fun onAiInstructionChanged(value: String) {
        _uiState.value = _uiState.value.copy(aiInstruction = value)
    }

    fun onDeviceSelected(value: String) {
        _uiState.value = _uiState.value.copy(selectedDeviceId = value)
    }

    fun createAiRunnerJob() {
        val instruction = _uiState.value.aiInstruction.trim()
        val deviceId = _uiState.value.selectedDeviceId.trim()
        if (instruction.isBlank()) {
            _uiState.value = _uiState.value.copy(error = "AI instruction is required.")
            return
        }
        if (deviceId.isBlank()) {
            _uiState.value = _uiState.value.copy(error = "Select a target device before running an AI job.")
            return
        }

        viewModelScope.launch {
            val localJobId = "task-${System.currentTimeMillis()}"
            _uiState.value = _uiState.value.copy(
                isSubmittingAiJob = true,
                aiJobId = localJobId,
                aiJobStatus = "RUNNING",
                aiJobResult = "Creating AI Runner job...",
                aiJobLogs = "Submitting instruction to /task/macro",
                error = null
            )

            val request = TaskMacroRequestDto(
                instruction = instruction,
                execute = true,
                continue_on_error = true,
                control_mode = "DAB",
                device_id = deviceId,
                max_steps = 12
            )

            when (val result = runsRepository.createTaskMacro(request)) {
                is ApiResult.Success -> {
                    val body = result.data
                    val success = body["success"]?.jsonPrimitive?.booleanOrNull == true
                    val plannedCount = body["planned_count"]?.jsonPrimitive?.intOrNull ?: 0
                    val failedExecution = body.executionResults().firstOrNull { item ->
                        item["success"]?.jsonPrimitive?.booleanOrNull == false
                    }
                    val status = when {
                        success -> "PASSED"
                        plannedCount == 0 -> "SKIPPED"
                        else -> "FAILED"
                    }
                    _uiState.value = _uiState.value.copy(
                        isSubmittingAiJob = false,
                        aiJobStatus = status,
                        aiJobResult = failedExecution?.get("error")?.jsonPrimitive?.contentOrNull
                            ?: body["instruction"]?.jsonPrimitive?.contentOrNull
                            ?: instruction,
                        aiJobLogs = body.aiRunnerLog(),
                        error = if (status == "FAILED") failedExecution?.get("error")?.jsonPrimitive?.contentOrNull ?: "AI Runner job failed." else null
                    )
                    refresh()
                }
                else -> {
                    _uiState.value = _uiState.value.copy(
                        isSubmittingAiJob = false,
                        aiJobStatus = "FAILED",
                        aiJobResult = result.errorText(),
                        aiJobLogs = result.errorText(),
                        error = result.errorText()
                    )
                }
            }
        }
    }

    private fun ApiResult<*>.errorText(): String = when (this) {
        is ApiResult.Success<*> -> ""
        is ApiResult.HttpError -> "HTTP ${code}: ${message}"
        is ApiResult.NetworkError -> "Network error: ${throwable.message}"
        is ApiResult.UnknownError -> "Unknown error: ${throwable.message}"
    }

    private fun JsonObject.executionResults(): List<JsonObject> {
        val execution = this["execution"] as? JsonObject ?: return emptyList()
        val results = execution["results"] as? JsonArray ?: return emptyList()
        return results.mapNotNull { it as? JsonObject }
    }

    private fun JsonObject.aiRunnerLog(): String {
        val lines = mutableListOf<String>()
        lines += "Instruction: ${this["instruction"]?.jsonPrimitive?.contentOrNull.orEmpty()}"
        lines += "Planned actions: ${this["planned_count"]?.jsonPrimitive?.contentOrNull ?: "0"}"
        val planned = this["planned_actions"] as? JsonArray
        planned?.forEachIndexed { index: Int, item: JsonElement ->
            val obj = item.jsonObject
            val action = obj["action"]?.jsonPrimitive?.contentOrNull.orEmpty()
            val params = obj["params"]?.toString() ?: "{}"
            lines += "${index + 1}. PLAN $action $params"
        }
        executionResults().forEachIndexed { index, obj ->
            val action = obj["action"]?.jsonPrimitive?.contentOrNull.orEmpty()
            val success = obj["success"]?.jsonPrimitive?.booleanOrNull == true
            val result = obj["result"]?.toString()
            val error = obj["error"]?.jsonPrimitive?.contentOrNull
            lines += "${index + 1}. EXEC $action ${if (success) "PASSED" else "FAILED"}"
            if (!result.isNullOrBlank()) lines += result
            if (!error.isNullOrBlank()) lines += "Error: $error"
        }
        return lines.joinToString("\n").take(5000)
    }
}
