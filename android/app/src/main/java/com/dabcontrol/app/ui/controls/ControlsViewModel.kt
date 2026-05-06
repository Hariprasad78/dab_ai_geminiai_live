package com.dabcontrol.app.ui.controls

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.api.IrSendRequestDto
import com.dabcontrol.app.data.api.IrTrainRequestDto
import com.dabcontrol.app.data.api.ManualActionBatchRequestDto
import com.dabcontrol.app.data.api.ManualActionBatchResponseDto
import com.dabcontrol.app.data.api.ManualActionRequestDto
import com.dabcontrol.app.data.api.ManualActionResponseDto
import com.dabcontrol.app.data.api.PlannerDebugRequestDto
import com.dabcontrol.app.data.api.TaskMacroRequestDto
import com.dabcontrol.app.data.preferences.ApiSettingsStore
import com.dabcontrol.app.data.repo.ControlsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import javax.inject.Inject
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.OkHttpClient
import okhttp3.Request

@HiltViewModel
class ControlsViewModel @Inject constructor(
    private val controlsRepository: ControlsRepository,
    private val apiSettingsStore: ApiSettingsStore,
    private val okHttpClient: OkHttpClient
) : ViewModel() {
    private val _uiState = MutableStateFlow(ControlsUiState())
    val uiState: StateFlow<ControlsUiState> = _uiState.asStateFlow()
    private val json = Json { ignoreUnknownKeys = true; isLenient = true }
    private var streamJob: Job? = null

    init {
        viewModelScope.launch {
            apiSettingsStore.apiBaseUrl.collectLatest { baseUrl ->
                _uiState.value = _uiState.value.copy(apiBaseUrl = baseUrl)
            }
        }
        viewModelScope.launch {
            apiSettingsStore.selectedDeviceId.collectLatest { deviceId ->
                if (deviceId.isNotBlank()) {
                    _uiState.value = _uiState.value.copy(selectedDeviceId = deviceId)
                }
            }
        }
        refreshAll(force = true)
    }

    fun onDeviceSelected(deviceId: String) {
        _uiState.value = _uiState.value.copy(selectedDeviceId = deviceId)
        viewModelScope.launch {
            apiSettingsStore.saveSelectedDeviceId(deviceId)
        }
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

    fun toggleStream() {
        if (_uiState.value.isStreaming) {
            stopStream()
        } else {
            startStream()
        }
    }

    fun refreshStream() {
        stopStream(clearFrame = false)
        startStream()
    }

    fun sendRemoteAction(action: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(remoteStatus = "Sending $action...")
            val request = ManualActionRequestDto(
                action = action,
                device_id = _uiState.value.selectedDeviceId.ifBlank { null }
            )
            when (val res = controlsRepository.manualAction(request)) {
                is ApiResult.Success -> {
                    val resultText = json.encodeToString(ManualActionResponseDto.serializer(), res.data)
                    _uiState.value = _uiState.value.copy(
                        remoteStatus = "$action sent",
                        lastActionResult = resultText
                    )
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(
                    remoteStatus = "$action failed: HTTP ${res.code}",
                    lastActionResult = "HTTP ${res.code}: ${res.message}"
                )
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(
                    remoteStatus = "$action failed: network",
                    lastActionResult = "Network error: ${res.throwable.message}"
                )
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(
                    remoteStatus = "$action failed",
                    lastActionResult = "Unknown error: ${res.throwable.message}"
                )
            }
        }
    }

    private fun startStream() {
        streamJob?.cancel()
        _uiState.value = _uiState.value.copy(
            isStreaming = true,
            streamStatus = "Connecting to HDMI stream..."
        )
        streamJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                val baseUrl = apiSettingsStore.apiBaseUrl.first().trimEnd('/')
                val request = Request.Builder()
                    .url("$baseUrl/stream/hdmi?ts=${System.currentTimeMillis()}")
                    .build()
                okHttpClient.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        _uiState.value = _uiState.value.copy(
                            isStreaming = false,
                            streamStatus = "Stream failed: HTTP ${response.code}"
                        )
                        return@use
                    }
                    val body = response.body ?: run {
                        _uiState.value = _uiState.value.copy(
                            isStreaming = false,
                            streamStatus = "Stream failed: empty response body"
                        )
                        return@use
                    }
                    BufferedInputStream(body.byteStream()).use { input ->
                        readMjpegFrames(input)
                    }
                }
            } catch (ce: CancellationException) {
                throw ce
            } catch (t: Throwable) {
                _uiState.value = _uiState.value.copy(
                    isStreaming = false,
                    streamStatus = "Stream failed: ${t.message ?: "unknown error"}"
                )
            }
        }
    }

    private fun stopStream(clearFrame: Boolean = true) {
        streamJob?.cancel()
        streamJob = null
        _uiState.value = _uiState.value.copy(
            isStreaming = false,
            streamFrameBytes = if (clearFrame) null else _uiState.value.streamFrameBytes,
            streamStatus = if (clearFrame) "Stream stopped." else "Reconnecting stream..."
        )
    }

    private suspend fun readMjpegFrames(input: BufferedInputStream) {
        val coroutineContext = currentCoroutineContext()
        val frameBuffer = ByteArrayOutputStream()
        var previous = -1
        var collecting = false
        var firstFrame = true

        while (coroutineContext.isActive) {
            val current = input.read()
            if (current == -1) break

            if (!collecting) {
                if (previous == 0xFF && current == 0xD8) {
                    collecting = true
                    frameBuffer.reset()
                    frameBuffer.write(0xFF)
                    frameBuffer.write(0xD8)
                }
            } else {
                frameBuffer.write(current)
                if (previous == 0xFF && current == 0xD9) {
                    val frame = frameBuffer.toByteArray()
                    _uiState.value = _uiState.value.copy(
                        streamFrameBytes = frame,
                        streamStatus = if (firstFrame) "Live stream connected." else _uiState.value.streamStatus
                    )
                    firstFrame = false
                    collecting = false
                    frameBuffer.reset()
                }
            }
            previous = current
        }

        if (coroutineContext.isActive) {
            _uiState.value = _uiState.value.copy(
                isStreaming = false,
                streamStatus = if (firstFrame) "Stream ended without video frames." else "Stream disconnected."
            )
        }
    }

    override fun onCleared() {
        streamJob?.cancel()
        super.onCleared()
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
                    if (_uiState.value.selectedDeviceId.isNotBlank()) {
                        apiSettingsStore.saveSelectedDeviceId(_uiState.value.selectedDeviceId)
                    }
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
                deviceInfoRows = buildDeviceInfoRows(infoRes),
                capabilityRows = buildCapabilityRows(capsRes),
                operationRows = buildOperationRows(opsRes),
                settingRows = buildSettingRows(curRes),
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
                is ApiResult.Success -> _uiState.value = _uiState.value.copy(
                    lastActionResult = json.encodeToString(ManualActionResponseDto.serializer(), res.data)
                )
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
                is ApiResult.Success -> _uiState.value = _uiState.value.copy(
                    lastBatchResult = json.encodeToString(ManualActionBatchResponseDto.serializer(), res.data)
                )
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
                    is JsonObject -> data.toString().take(3500)
                    else -> data.toString().take(3500)
                }
            }
            is ApiResult.HttpError -> "HTTP ${result.code}: ${result.message}"
            is ApiResult.NetworkError -> "Network error: ${result.throwable.message}"
            is ApiResult.UnknownError -> "Unknown error: ${result.throwable.message}"
        }
    }

    private fun buildDeviceInfoRows(result: ApiResult<*>): List<ControlsInfoRow> {
        val payload = (result as? ApiResult.Success<*>)?.data as? JsonObject ?: return emptyList()
        val body = payload["result"] as? JsonObject ?: payload
        val network = body["networkInterfaces"]?.let { element ->
            (element as? JsonArray)?.firstOrNull()?.jsonObject
        }
        return listOf(
            ControlsInfoRow("Device", body.stringValue("deviceId") ?: payload.stringValue("device_id") ?: "--"),
            ControlsInfoRow("Manufacturer", body.stringValue("manufacturer") ?: "--"),
            ControlsInfoRow("Model", body.stringValue("model") ?: "--"),
            ControlsInfoRow("Firmware", body.stringValue("firmwareVersion") ?: "--"),
            ControlsInfoRow("Build", body.stringValue("firmwareBuild") ?: "--"),
            ControlsInfoRow("Display", "${body.intValue("screenWidthPixels") ?: 0} x ${body.intValue("screenHeightPixels") ?: 0}"),
            ControlsInfoRow("Network", network?.stringValue("type") ?: "--"),
            ControlsInfoRow("IP Address", network?.stringValue("ipAddress") ?: "--")
        )
    }

    private fun buildCapabilityRows(result: ApiResult<*>): List<ControlsInfoRow> {
        val payload = (result as? ApiResult.Success<*>)?.data as? JsonObject ?: return emptyList()
        return listOf(
            ControlsInfoRow("Supported operations", payload.arraySize("supported_operations").toString()),
            ControlsInfoRow("Remote keys", payload.arraySize("supported_keys").toString()),
            ControlsInfoRow("Installed apps", payload.arraySize("installed_applications").toString()),
            ControlsInfoRow("Voice systems", payload.arraySize("supported_voice_systems").toString()),
            ControlsInfoRow("Settings exposed", payload.arraySize("supported_settings").toString()),
            ControlsInfoRow("Last updated", payload.stringValue("last_updated") ?: "--")
        )
    }

    private fun buildOperationRows(result: ApiResult<*>): List<ControlsOperationRow> {
        val payload = (result as? ApiResult.Success<*>)?.data as? JsonObject ?: return emptyList()
        val rows = payload["rows"] as? JsonArray ?: return emptyList()
        return rows.mapNotNull { element ->
            val obj = element as? JsonObject ?: return@mapNotNull null
            ControlsOperationRow(
                operation = obj.stringValue("operation") ?: return@mapNotNull null,
                supported = obj.booleanValue("supported") ?: false,
                defaultAction = obj.stringValue("default_action") ?: "--",
                relatedCount = obj.intValue("related_count") ?: 0
            )
        }
    }

    private fun buildSettingRows(result: ApiResult<*>): List<ControlsSettingRow> {
        val payload = (result as? ApiResult.Success<*>)?.data as? JsonObject ?: return emptyList()
        val rows = payload["current_setting_values"] as? JsonArray ?: return emptyList()
        return rows.mapNotNull { element ->
            val obj = element as? JsonObject ?: return@mapNotNull null
            ControlsSettingRow(
                name = obj.stringValue("friendlyName") ?: obj.stringValue("key") ?: return@mapNotNull null,
                value = formatSettingValue(obj["current_value"]),
                writable = obj.booleanValue("writable") ?: false,
                status = when {
                    obj.booleanValue("read_success") == true -> "Ready"
                    !obj.stringValue("read_error").isNullOrBlank() -> "Missing"
                    else -> "--"
                }
            )
        }
    }

    private fun formatSettingValue(element: kotlinx.serialization.json.JsonElement?): String {
        return when (element) {
            null -> "--"
            is JsonObject -> "${element.size} fields"
            is JsonArray -> "${element.size} items"
            else -> element.jsonPrimitive.contentOrNull
                ?: element.jsonPrimitive.booleanOrNull?.toString()
                ?: element.jsonPrimitive.intOrNull?.toString()
                ?: element.jsonPrimitive.doubleOrNull?.toString()
                ?: element.toString()
        }
    }

    private fun JsonObject.stringValue(key: String): String? = this[key]?.jsonPrimitive?.contentOrNull

    private fun JsonObject.intValue(key: String): Int? = this[key]?.jsonPrimitive?.intOrNull

    private fun JsonObject.booleanValue(key: String): Boolean? = this[key]?.jsonPrimitive?.booleanOrNull

    private fun JsonObject.arraySize(key: String): Int = (this[key] as? JsonArray)?.size ?: 0

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
