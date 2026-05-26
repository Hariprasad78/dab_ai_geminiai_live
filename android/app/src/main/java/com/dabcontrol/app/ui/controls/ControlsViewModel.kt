package com.dabcontrol.app.ui.controls

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.api.AudioSourceResponseDto
import com.dabcontrol.app.data.api.CurrentDeviceContextResponseDto
import com.dabcontrol.app.data.api.DeviceContextDto
import com.dabcontrol.app.data.api.DeviceContextsResponseDto
import com.dabcontrol.app.data.api.IrDeviceKeysResponseDto
import com.dabcontrol.app.data.api.IrDevicesResponseDto
import com.dabcontrol.app.data.api.IrSendRequestDto
import com.dabcontrol.app.data.api.IrTrainRequestDto
import com.dabcontrol.app.data.api.ManualActionBatchRequestDto
import com.dabcontrol.app.data.api.ManualActionBatchResponseDto
import com.dabcontrol.app.data.api.ManualActionRequestDto
import com.dabcontrol.app.data.api.ManualActionResponseDto
import com.dabcontrol.app.data.api.ScrcpyStreamStartRequestDto
import com.dabcontrol.app.data.api.TaskMacroRequestDto
import com.dabcontrol.app.data.api.PlannerDebugRequestDto
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
                _uiState.value = _uiState.value.copy(selectedDeviceId = deviceId)
            }
        }
        refreshAll(force = true)
    }

    fun onDeviceSelected(deviceId: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(refreshStatus = "Applying device context...")
            when (val result = controlsRepository.selectDeviceContext(deviceId, persist = true)) {
                is ApiResult.Success -> {
                    val selected = applyContextState(
                        currentContext = result,
                        contexts = null
                    )
                    if (selected.isNotBlank()) {
                        apiSettingsStore.saveSelectedDeviceId(selected)
                    }
                    refreshAll(force = true)
                }
                is ApiResult.HttpError -> {
                    _uiState.value = _uiState.value.copy(error = "HTTP ${result.code}: ${result.message}")
                }
                is ApiResult.NetworkError -> {
                    _uiState.value = _uiState.value.copy(error = "Network error: ${result.throwable.message}")
                }
                is ApiResult.UnknownError -> {
                    _uiState.value = _uiState.value.copy(error = "Unknown error: ${result.throwable.message}")
                }
            }
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

    fun toggleAudioStream() {
        val currentlyStreaming = _uiState.value.isAudioStreaming
        _uiState.value = _uiState.value.copy(
            isAudioStreaming = !currentlyStreaming,
            audioStatus = if (currentlyStreaming) {
                "Audio stream stopped."
            } else {
                "Starting HDMI audio stream..."
            }
        )
    }

    fun startScrcpyStream() {
        stopStream(clearFrame = false)
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(streamStatus = "Starting Android UI (scrcpy) stream...")
            val req = ScrcpyStreamStartRequestDto(
                device_id = _uiState.value.selectedDeviceId.ifBlank { null },
                persist = false
            )
            when (val res = controlsRepository.startScrcpyStream(req)) {
                is ApiResult.Success -> {
                    startStream()
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(streamStatus = "Scrcpy start failed: HTTP ${res.code}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(streamStatus = "Scrcpy start failed: network")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(streamStatus = "Scrcpy start failed")
            }
        }
    }

    fun onAudioPlaybackReady() {
        _uiState.value = _uiState.value.copy(audioStatus = "HDMI audio stream connected.")
    }

    fun onAudioPlaybackError(message: String) {
        _uiState.value = _uiState.value.copy(
            isAudioStreaming = false,
            audioStatus = message.ifBlank { "Audio stream failed." }
        )
    }

    fun onRemoteModeChanged(mode: ControlsRemoteMode) {
        _uiState.value = _uiState.value.copy(
            remoteMode = mode,
            remoteStatus = if (mode == ControlsRemoteMode.DAB) {
                "DAB remote ready."
            } else {
                "IR remote ready."
            }
        )
        if (mode == ControlsRemoteMode.IR) {
            fetchIrKeys(silent = true)
        }
    }

    fun sendRemoteAction(action: String) {
        if (_uiState.value.remoteMode == ControlsRemoteMode.IR) {
            sendIrRemoteAction(action)
            return
        }
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

    private fun sendIrRemoteAction(action: String) {
        viewModelScope.launch {
            val keyName = resolveIrKeyName(action)
            if (keyName == null) {
                _uiState.value = _uiState.value.copy(
                    remoteStatus = "No IR key mapping found for $action. Load IR keys first."
                )
                return@launch
            }
            _uiState.value = _uiState.value.copy(
                remoteStatus = "Sending IR $keyName...",
                irKeyName = keyName
            )
            val req = IrSendRequestDto(
                device_id = _uiState.value.irDeviceId.trim(),
                key_name = keyName
            )
            when (val result = controlsRepository.irSend(req)) {
                is ApiResult.Success -> {
                    _uiState.value = _uiState.value.copy(
                        remoteStatus = "IR $keyName sent",
                        irLastResult = result.data.toString().take(3500)
                    )
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(remoteStatus = "IR failed: HTTP ${result.code}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(remoteStatus = "IR failed: network")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(remoteStatus = "IR failed")
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
            val currentContextDef = async { controlsRepository.fetchCurrentDeviceContext() }
            val contextsDef = async { controlsRepository.fetchDeviceContexts() }

            val devicesRes = devicesDef.await()
            val currentContextRes = currentContextDef.await()
            val contextsRes = contextsDef.await()
            val selected = applyContextState(currentContextRes, contextsRes)

            val infoDef = async { controlsRepository.fetchDeviceInfo(selected.ifBlank { null }) }
            val capsDef = async { controlsRepository.fetchCapabilityStatus(selected.ifBlank { null }, refresh = force) }
            val opsDef = async { controlsRepository.fetchOperationsGrid(selected.ifBlank { null }, refresh = force) }
            val curDef = async { controlsRepository.fetchCurrentSettings(selected.ifBlank { null }, refresh = force) }
            val audioDef = async { controlsRepository.fetchAudioSource() }
            val irStatusDef = async { controlsRepository.irStatus() }
            val irDevicesDef = async { controlsRepository.irDevices() }

            val infoRes = infoDef.await()
            val capsRes = capsDef.await()
            val opsRes = opsDef.await()
            val curRes = curDef.await()
            val audioRes = audioDef.await()
            val irStatusRes = irStatusDef.await()
            val irDevicesRes = irDevicesDef.await()

            _uiState.value = _uiState.value.copy(
                isLoading = false,
                deviceIds = extractDeviceIds(devicesRes),
                deviceInfoRows = buildDeviceInfoRows(infoRes),
                capabilityRows = buildCapabilityRows(capsRes),
                operationRows = buildOperationRows(opsRes),
                settingRows = buildSettingRows(curRes),
                audioSource = buildAudioSource(audioRes),
                audioStatus = mergeAudioStatus(audioRes, _uiState.value.isAudioStreaming),
                irStatusRows = buildIrStatusRows(irStatusRes),
                irAvailableDevices = buildIrDeviceList(irDevicesRes),
                refreshStatus = "Last refreshed at ${java.time.LocalTime.now().withNano(0)}",
                error = firstError(devicesRes, currentContextRes, contextsRes, infoRes, capsRes, opsRes, curRes, audioRes, irStatusRes, irDevicesRes)
            )
            if (selected.isNotBlank()) {
                apiSettingsStore.saveSelectedDeviceId(selected)
            }
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
        _uiState.value = _uiState.value.copy(irDeviceId = value, irAvailableKeys = emptyList())
    }

    fun onIrKeyChanged(value: String) {
        _uiState.value = _uiState.value.copy(irKeyName = value)
    }

    private fun resolveIrKeyName(action: String): String? {
        val preferred = when (action) {
            "PRESS_UP" -> listOf("UP", "KEY_UP", "CURSOR_UP")
            "PRESS_DOWN" -> listOf("DOWN", "KEY_DOWN", "CURSOR_DOWN")
            "PRESS_LEFT" -> listOf("LEFT", "KEY_LEFT", "CURSOR_LEFT")
            "PRESS_RIGHT" -> listOf("RIGHT", "KEY_RIGHT", "CURSOR_RIGHT")
            "PRESS_OK" -> listOf("ENTER", "OK", "SELECT", "KEY_ENTER")
            "PRESS_BACK" -> listOf("BACK", "RETURN", "EXIT")
            "PRESS_HOME" -> listOf("HOME")
            "PRESS_MENU" -> listOf("MENU", "SETTINGS")
            "PRESS_INFO" -> listOf("INFO", "GUIDE")
            "PRESS_PLAY_PAUSE" -> listOf("PLAYPAUSE", "PLAY_PAUSE", "PLAY", "PAUSE")
            "PRESS_POWER" -> listOf("POWER")
            "PRESS_VOLUME_UP" -> listOf("VOLUMEUP", "VOLUP", "VOLUME_UP")
            "PRESS_VOLUME_DOWN" -> listOf("VOLUMEDOWN", "VOLDOWN", "VOLUME_DOWN")
            "PRESS_MUTE" -> listOf("MUTE")
            else -> listOf(action.removePrefix("PRESS_"))
        }
        val available = _uiState.value.irAvailableKeys
        if (available.isEmpty()) {
            return preferred.firstOrNull()
        }
        preferred.forEach { candidate ->
            available.firstOrNull { it.equals(candidate, ignoreCase = true) }?.let { return it }
        }
        val normalizedAvailable = available.associateBy { normalizeIrToken(it) }
        preferred.firstNotNullOfOrNull { candidate ->
            normalizedAvailable[normalizeIrToken(candidate)]
        }?.let { return it }
        return available.firstOrNull { normalizeIrToken(it).contains(normalizeIrToken(action.removePrefix("PRESS_"))) }
    }

    private fun normalizeIrToken(value: String): String = value.lowercase().replace(Regex("[^a-z0-9]"), "")

    private fun applyContextState(
        currentContext: ApiResult<CurrentDeviceContextResponseDto>,
        contexts: ApiResult<DeviceContextsResponseDto>?
    ): String {
        val currentPayload = (currentContext as? ApiResult.Success)?.data
        val mappedContexts = buildDeviceContexts(contexts, currentPayload)
        val active = mappedContexts.firstOrNull { it.isActive }
            ?: mappedContexts.firstOrNull { it.dabDeviceId == currentPayload?.selected_device_id.orEmpty() }
        val resolvedSelected = active?.dabDeviceId
            ?: currentPayload?.selected_device_id.orEmpty()
            ?: _uiState.value.selectedDeviceId
        val resolvedIrDeviceId = active?.irDeviceId
            ?: currentPayload?.context?.irDeviceId.orEmpty()
            ?: _uiState.value.irDeviceId
        _uiState.value = _uiState.value.copy(
            deviceContexts = mappedContexts,
            selectedDeviceId = resolvedSelected,
            selectedDeviceName = active?.displayName.orEmpty(),
            selectedYtsDeviceId = active?.ytsDeviceId.orEmpty(),
            selectedYtsShortId = active?.ytsShortId.orEmpty(),
            selectedIrDeviceId = active?.irDeviceId.orEmpty(),
            selectedVideoSource = active?.videoSource.orEmpty(),
            selectedContextIssues = active?.issues ?: currentPayload?.validation?.issues.orEmpty(),
            irDeviceId = if (resolvedIrDeviceId.isBlank()) _uiState.value.irDeviceId else resolvedIrDeviceId
        )
        return resolvedSelected
    }

    private fun buildDeviceContexts(
        contexts: ApiResult<DeviceContextsResponseDto>?,
        currentPayload: CurrentDeviceContextResponseDto?
    ): List<ControlsDeviceContext> {
        val payload = (contexts as? ApiResult.Success)?.data?.contexts.orEmpty()
        if (payload.isNotEmpty()) {
            return payload.map { dto ->
                val readiness = dto.readiness
                ControlsDeviceContext(
                    contextId = dto.contextId,
                    displayName = dto.displayName.ifBlank { dto.dabDeviceId },
                    dabDeviceId = dto.dabDeviceId,
                    ytsDeviceId = dto.ytsDeviceId,
                    ytsShortId = readiness?.ytsShortId.orEmpty(),
                    irDeviceId = dto.irDeviceId,
                    videoSource = dto.videoSource.ifBlank { readiness?.videoSource.orEmpty() },
                    isActive = dto.active,
                    isReady = readiness?.valid ?: false,
                    issues = readiness?.issues.orEmpty()
                )
            }
        }

        val current = currentPayload?.context ?: return emptyList()
        return listOf(
            ControlsDeviceContext(
                contextId = current.contextId,
                displayName = current.displayName.ifBlank { current.dabDeviceId },
                dabDeviceId = current.dabDeviceId,
                ytsDeviceId = current.ytsDeviceId,
                ytsShortId = currentPayload.validation?.ytsShortId.orEmpty(),
                irDeviceId = current.irDeviceId,
                videoSource = current.videoSource.ifBlank { currentPayload.validation?.videoSource.orEmpty() },
                isActive = true,
                isReady = currentPayload.validation?.valid ?: false,
                issues = currentPayload.validation?.issues.orEmpty()
            )
        )
    }

    fun fetchIrKeys() = fetchIrKeys(silent = false)

    private fun fetchIrKeys(silent: Boolean) {
        viewModelScope.launch {
            val deviceId = _uiState.value.irDeviceId.trim()
            if (deviceId.isEmpty()) {
                if (!silent) {
                    _uiState.value = _uiState.value.copy(error = "IR device id required")
                }
                return@launch
            }
            if (!silent) {
                _uiState.value = _uiState.value.copy(refreshStatus = "Loading IR keys...")
            }
            when (val result = controlsRepository.irDeviceKeys(deviceId)) {
                is ApiResult.Success -> {
                    _uiState.value = _uiState.value.copy(
                        irAvailableKeys = result.data.keys,
                        refreshStatus = if (silent) _uiState.value.refreshStatus else "Loaded ${result.data.keys.size} IR keys."
                    )
                }
                is ApiResult.HttpError -> if (!silent) _uiState.value = _uiState.value.copy(error = "HTTP ${result.code}: ${result.message}")
                is ApiResult.NetworkError -> if (!silent) _uiState.value = _uiState.value.copy(error = "Network error: ${result.throwable.message}")
                is ApiResult.UnknownError -> if (!silent) _uiState.value = _uiState.value.copy(error = "Unknown error: ${result.throwable.message}")
            }
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

    fun captureScreenshot() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isScreenshotting = true, plannerResult = "Capturing screenshot...")
            when (val res = controlsRepository.captureScreenshot()) {
                is ApiResult.Success -> {
                    val b64 = res.data["image_b64"]?.jsonPrimitive?.content
                    val ocr = res.data["ocr_text"]?.jsonPrimitive?.content
                    _uiState.value = _uiState.value.copy(
                        isScreenshotting = false,
                        capturedScreenshotB64 = b64,
                        plannerOcrText = ocr ?: _uiState.value.plannerOcrText,
                        plannerResult = "Screenshot captured successfully."
                    )
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(
                    isScreenshotting = false,
                    plannerResult = "Screenshot failed: HTTP ${res.code}"
                )
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(
                    isScreenshotting = false,
                    plannerResult = "Screenshot failed: network error"
                )
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(
                    isScreenshotting = false,
                    plannerResult = "Screenshot failed: unknown error"
                )
            }
        }
    }

    fun runPlannerDebug() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(plannerResult = "Analyzing...", isAnalyzing = true)
            val req = PlannerDebugRequestDto(
                goal = _uiState.value.plannerGoal,
                device_id = _uiState.value.selectedDeviceId.ifBlank { null },
                ocr_text = _uiState.value.plannerOcrText.ifBlank { null },
                screenshot_b64 = _uiState.value.capturedScreenshotB64,
                use_live_capture = _uiState.value.capturedScreenshotB64 == null,
                current_app = _uiState.value.plannerCurrentApp.ifBlank { null },
                current_screen = _uiState.value.plannerCurrentScreen.ifBlank { null }
            )
            _uiState.value = _uiState.value.copy(
                plannerResult = preview(controlsRepository.plannerDebug(req)),
                isAnalyzing = false
            )
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
                    is IrDevicesResponseDto -> data.devices.joinToString()
                    is IrDeviceKeysResponseDto -> data.keys.joinToString()
                    else -> data.toString().take(3500)
                }
            }
            is ApiResult.HttpError -> "HTTP ${result.code}: ${result.message}"
            is ApiResult.NetworkError -> "Network error: ${result.throwable.message}"
            is ApiResult.UnknownError -> "Unknown error: ${result.throwable.message}"
        }
    }

    private fun buildIrStatusRows(result: ApiResult<*>): List<ControlsInfoRow> {
        val payload = (result as? ApiResult.Success<*>)?.data as? JsonObject ?: return emptyList()
        return buildList {
            payload.stringValue("service")?.takeIf { it.isNotBlank() }?.let { add(ControlsInfoRow("Service", it)) }
            payload.stringValue("brand")?.takeIf { it.isNotBlank() }?.let { add(ControlsInfoRow("Brand", it)) }
            payload.stringValue("active_device_id")?.takeIf { it.isNotBlank() }?.let { add(ControlsInfoRow("Active IR Profile", it)) }
            payload.stringValue("status")?.takeIf { it.isNotBlank() }?.let { add(ControlsInfoRow("Status", it)) }
        }
    }

    private fun buildAudioSource(result: ApiResult<AudioSourceResponseDto>): ControlsAudioSource? {
        val payload = (result as? ApiResult.Success)?.data ?: return null
        return ControlsAudioSource(
            enabled = payload.enabled,
            ffmpegAvailable = payload.ffmpeg_available,
            device = payload.device.orEmpty(),
            inputFormat = payload.input_format.orEmpty(),
            sampleRate = payload.sample_rate?.toString().orEmpty(),
            channels = payload.channels?.toString().orEmpty()
        )
    }

    private fun mergeAudioStatus(
        result: ApiResult<AudioSourceResponseDto>,
        isPlaying: Boolean
    ): String {
        return when (result) {
            is ApiResult.Success -> {
                val data = result.data
                when {
                    !data.enabled -> "Audio streaming is disabled on the backend."
                    !data.ffmpeg_available -> "Audio backend is missing ffmpeg."
                    isPlaying -> "HDMI audio stream running."
                    else -> "Audio stream ready for the selected device."
                }
            }
            is ApiResult.HttpError -> "Audio failed: HTTP ${result.code}"
            is ApiResult.NetworkError -> "Audio failed: network"
            is ApiResult.UnknownError -> "Audio failed"
        }
    }

    private fun buildIrDeviceList(result: ApiResult<IrDevicesResponseDto>): List<String> {
        val payload = (result as? ApiResult.Success)?.data ?: return emptyList()
        val devices = payload.devices.filter { it.isNotBlank() }
        val activeIr = payload.active_device_id.orEmpty()
        if (activeIr.isNotBlank() && _uiState.value.irDeviceId.isBlank()) {
            _uiState.value = _uiState.value.copy(irDeviceId = activeIr)
        }
        return devices
    }

    private fun extractDeviceIds(result: ApiResult<com.dabcontrol.app.data.api.DabDevicesResponseDto>): List<String> {
        val payload = (result as? ApiResult.Success)?.data ?: return emptyList()
        return payload.devices
            .mapNotNull { it["device_id"]?.jsonPrimitive?.contentOrNull }
            .filter { it.isNotBlank() }
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
