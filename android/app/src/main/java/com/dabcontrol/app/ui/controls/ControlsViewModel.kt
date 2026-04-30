package com.dabcontrol.app.ui.controls

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.api.ManualActionBatchRequestDto
import com.dabcontrol.app.data.api.ManualActionRequestDto
import com.dabcontrol.app.data.api.PlannerDebugRequestDto
import com.dabcontrol.app.data.api.TaskMacroRequestDto
import com.dabcontrol.app.data.api.IrSendRequestDto
import com.dabcontrol.app.data.api.IrTrainRequestDto
import com.dabcontrol.app.data.repo.ControlsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

@HiltViewModel
class ControlsViewModel @Inject constructor(
    private val controlsRepository: ControlsRepository
) : ViewModel() {
    private val _uiState = MutableStateFlow(ControlsUiState())
    val uiState: StateFlow<ControlsUiState> = _uiState.asStateFlow()
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }

    init {
        refreshAll(force = true)
    }

    fun onDeviceSelected(deviceId: String) {
        _uiState.value = _uiState.value.copy(selectedDeviceId = deviceId)
    }

    fun onActionChanged(value: String) {
        _uiState.value = _uiState.value.copy(actionName = value)
    }

    fun onActionParamsChanged(value: String) {
        _uiState.value = _uiState.value.copy(actionParamsJson = value)
    }

    fun onBatchActionsChanged(value: String) {
        _uiState.value = _uiState.value.copy(batchActionsJson = value)
    }

    fun refreshAll(force: Boolean = true) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null, refreshStatus = "Refreshing...")
            val devicesDef = async { controlsRepository.fetchDevices() }
            val devicesRes = devicesDef.await()

            val selected = when (devicesRes) {
                is ApiResult.Success -> {
                    val ids = devicesRes.data.devices.mapNotNull { it["device_id"]?.jsonPrimitive?.contentOrNull }.filter { it.isNotBlank() }
                    val selectedFromApi = devicesRes.data.selected_device_id.orEmpty()
                    val fallback = ids.firstOrNull().orEmpty()
                    val resolved = if (selectedFromApi.isNotBlank()) selectedFromApi else fallback
                    _uiState.value = _uiState.value.copy(
                        deviceIds = ids,
                        selectedDeviceId = if (_uiState.value.selectedDeviceId.isBlank()) resolved else _uiState.value.selectedDeviceId
                    )
                    _uiState.value.selectedDeviceId
                }
                else -> _uiState.value.selectedDeviceId
            }

            val infoDef = async { controlsRepository.fetchDeviceInfo(selected.ifBlank { null }) }
            val capsDef = async { controlsRepository.fetchCapabilityStatus(selected.ifBlank { null }, refresh = force) }
            val opsDef = async { controlsRepository.fetchOperationsGrid(selected.ifBlank { null }, refresh = force) }
            val curDef = async { controlsRepository.fetchCurrentSettings(selected.ifBlank { null }, refresh = force) }
            val irStatusDef = async { controlsRepository.irStatus() }
            val irDevicesDef = async { controlsRepository.irDevices() }

            val infoRes = infoDef.await()
            val capsRes = capsDef.await()
            val opsRes = opsDef.await()
            val curRes = curDef.await()
            val irStatusRes = irStatusDef.await()
            val irDevicesRes = irDevicesDef.await()

            _uiState.value = _uiState.value.copy(
                isLoading = false,
                deviceInfoPreview = preview(infoRes),
                capabilityPreview = preview(capsRes),
                operationsPreview = preview(opsRes),
                currentSettingsPreview = preview(curRes),
                irStatusPreview = preview(irStatusRes),
                irDevicesPreview = preview(irDevicesRes),
                refreshStatus = "Last refreshed at ${java.time.LocalTime.now().withNano(0)}",
                error = firstError(devicesRes, infoRes, capsRes, opsRes, curRes, irStatusRes, irDevicesRes)
            )
        }
    }

    fun executeAction() {
        viewModelScope.launch {
            val params = parseJsonObject(_uiState.value.actionParamsJson) ?: buildJsonObject { }
            val req = ManualActionRequestDto(
                action = _uiState.value.actionName.trim(),
                params = params,
                device_id = _uiState.value.selectedDeviceId.ifBlank { null }
            )
            when (val res = controlsRepository.manualAction(req)) {
                is ApiResult.Success -> _uiState.value = _uiState.value.copy(lastActionResult = json.encodeToString(res.data))
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(lastActionResult = "HTTP ${res.code}: ${res.message}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(lastActionResult = "Network error: ${res.throwable.message}")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(lastActionResult = "Unknown error: ${res.throwable.message}")
            }
        }
    }

    fun executeBatch() {
        viewModelScope.launch {
            val array = parseJsonArray(_uiState.value.batchActionsJson)
            if (array == null) {
                _uiState.value = _uiState.value.copy(lastBatchResult = "Invalid batch JSON array")
                return@launch
            }
            val actions = array.mapNotNull { element ->
                val obj = element as? JsonObject ?: return@mapNotNull null
                val action = obj["action"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
                val params = obj["params"] as? JsonObject
                ManualActionRequestDto(
                    action = action,
                    params = params,
                    device_id = _uiState.value.selectedDeviceId.ifBlank { null }
                )
            }
            val req = ManualActionBatchRequestDto(actions = actions, continue_on_error = true)
            when (val res = controlsRepository.manualBatch(req)) {
                is ApiResult.Success -> _uiState.value = _uiState.value.copy(lastBatchResult = json.encodeToString(res.data))
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(lastBatchResult = "HTTP ${res.code}: ${res.message}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(lastBatchResult = "Network error: ${res.throwable.message}")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(lastBatchResult = "Unknown error: ${res.throwable.message}")
            }
        }
    }

    fun onIrDeviceChanged(value: String) {
        _uiState.value = _uiState.value.copy(irDeviceId = value)
    }

    fun onIrKeyChanged(value: String) {
        _uiState.value = _uiState.value.copy(irKeyName = value)
    }

    fun fetchIrKeys() {
        viewModelScope.launch {
            val deviceId = _uiState.value.irDeviceId.trim()
            if (deviceId.isEmpty()) {
                _uiState.value = _uiState.value.copy(irKeysPreview = "IR device id required")
                return@launch
            }
            _uiState.value = _uiState.value.copy(irKeysPreview = "Loading IR keys...")
            _uiState.value = _uiState.value.copy(irKeysPreview = preview(controlsRepository.irDeviceKeys(deviceId)))
        }
    }

    fun irSend() {
        viewModelScope.launch {
            val req = IrSendRequestDto(
                device_id = _uiState.value.irDeviceId.trim(),
                key_name = _uiState.value.irKeyName.trim()
            )
            _uiState.value = _uiState.value.copy(irLastResult = preview(controlsRepository.irSend(req)))
        }
    }

    fun irTrain() {
        viewModelScope.launch {
            val req = IrTrainRequestDto(
                device_id = _uiState.value.irDeviceId.trim(),
                key_name = _uiState.value.irKeyName.trim(),
                timeout_ms = 6000
            )
            _uiState.value = _uiState.value.copy(irLastResult = preview(controlsRepository.irTrain(req)))
        }
    }

    fun onMacroInstructionChanged(value: String) {
        _uiState.value = _uiState.value.copy(macroInstruction = value)
    }

    fun toggleMacroExecute() {
        _uiState.value = _uiState.value.copy(macroExecute = !_uiState.value.macroExecute)
    }

    fun runMacro() {
        viewModelScope.launch {
            val req = TaskMacroRequestDto(
                instruction = _uiState.value.macroInstruction,
                execute = _uiState.value.macroExecute,
                continue_on_error = true
            )
            _uiState.value = _uiState.value.copy(macroResult = preview(controlsRepository.taskMacro(req)))
        }
    }

    fun onPlannerGoalChanged(value: String) {
        _uiState.value = _uiState.value.copy(plannerGoal = value)
    }

    fun onPlannerAppChanged(value: String) {
        _uiState.value = _uiState.value.copy(plannerCurrentApp = value)
    }

    fun onPlannerScreenChanged(value: String) {
        _uiState.value = _uiState.value.copy(plannerCurrentScreen = value)
    }

    fun onPlannerOcrChanged(value: String) {
        _uiState.value = _uiState.value.copy(plannerOcrText = value)
    }

    fun runPlannerDebug() {
        viewModelScope.launch {
            val req = PlannerDebugRequestDto(
                goal = _uiState.value.plannerGoal,
                device_id = _uiState.value.selectedDeviceId.ifBlank { null },
                ocr_text = _uiState.value.plannerOcrText.ifBlank { null },
                current_app = _uiState.value.plannerCurrentApp.ifBlank { null },
                current_screen = _uiState.value.plannerCurrentScreen.ifBlank { null }
            )
            _uiState.value = _uiState.value.copy(plannerResult = preview(controlsRepository.plannerDebug(req)))
        }
    }

    private fun parseJsonObject(raw: String): JsonObject? {
        return try {
            json.parseToJsonElement(raw).jsonObject
        } catch (_: Throwable) {
            null
        }
    }

    private fun parseJsonArray(raw: String): JsonArray? {
        return try {
            json.parseToJsonElement(raw) as? JsonArray
        } catch (_: Throwable) {
            null
        }
    }

    private fun preview(result: ApiResult<*>): String {
        return when (result) {
            is ApiResult.Success<*> -> {
                val data = result.data
                when (data) {
                    is JsonObject -> json.encodeToString(data).take(3500)
                    else -> data.toString().take(3500)
                }
            }
            is ApiResult.HttpError -> "HTTP ${result.code}: ${result.message}"
            is ApiResult.NetworkError -> "Network error: ${result.throwable.message}"
            is ApiResult.UnknownError -> "Unknown error: ${result.throwable.message}"
        }
    }

    private fun firstError(vararg results: ApiResult<*>): String? {
        for (r in results) {
            when (r) {
                is ApiResult.HttpError -> return "HTTP ${r.code}: ${r.message}"
                is ApiResult.NetworkError -> return "Network error: ${r.throwable.message}"
                is ApiResult.UnknownError -> return "Unknown error: ${r.throwable.message}"
                is ApiResult.Success -> Unit
            }
        }
        return null
    }
}
