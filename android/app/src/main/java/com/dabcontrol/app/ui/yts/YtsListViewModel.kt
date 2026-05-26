package com.dabcontrol.app.ui.yts

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.api.CurrentDeviceContextResponseDto
import com.dabcontrol.app.data.api.RuntimeModelResponseDto
import com.dabcontrol.app.data.api.ManualActionRequestDto
import com.dabcontrol.app.data.api.YtsLiveCommandRequestDto
import com.dabcontrol.app.data.api.YtsLiveCommandStateDto
import com.dabcontrol.app.data.api.YtsLiveCommandSummaryDto
import com.dabcontrol.app.data.api.YtsTestCatalogItemDto
import com.dabcontrol.app.data.api.YtsResultsAnalysisRequestDto
import com.dabcontrol.app.data.preferences.ApiSettingsStore
import com.dabcontrol.app.data.repo.ControlsRepository
import com.dabcontrol.app.data.repo.YtsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import javax.inject.Inject
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import okhttp3.OkHttpClient
import okhttp3.Request

@HiltViewModel
class YtsListViewModel @Inject constructor(
    private val ytsRepository: YtsRepository,
    private val controlsRepository: ControlsRepository,
    private val apiSettingsStore: ApiSettingsStore,
    private val okHttpClient: OkHttpClient
) : ViewModel() {
    private val _uiState = MutableStateFlow(YtsListUiState())
    val uiState: StateFlow<YtsListUiState> = _uiState.asStateFlow()

    private var pollJob: Job? = null
    private var streamJob: Job? = null

    init {
        viewModelScope.launch {
            apiSettingsStore.apiBaseUrl.collectLatest { baseUrl ->
                _uiState.value = _uiState.value.copy(apiBaseUrl = baseUrl)
            }
        }
        viewModelScope.launch {
            apiSettingsStore.selectedDeviceId.collectLatest { deviceId ->
                _uiState.value = _uiState.value.copy(deviceId = deviceId)
                refreshSharedDeviceContext()
            }
        }
        refresh()
        loadCatalog(refresh = false)
        loadRuntimeModels()
        fetchArtifacts()
        startPolling()
    }

    fun fetchArtifacts() {
        viewModelScope.launch {
            when (val result = ytsRepository.fetchResultArtifacts()) {
                is ApiResult.Success -> {
                    _uiState.value = _uiState.value.copy(
                        artifacts = result.data,
                        analysisStatus = "Loaded ${result.data.size} artifact option${if (result.data.size == 1) "" else "s"}."
                    )
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(
                    analysisStatus = "Artifact load failed: HTTP ${result.code}",
                    error = "HTTP ${result.code}: ${result.message}"
                )
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(
                    analysisStatus = "Artifact load failed: network",
                    error = "Network error: ${result.throwable.message}"
                )
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(
                    analysisStatus = "Artifact load failed",
                    error = "Unknown error: ${result.throwable.message}"
                )
            }
        }
    }

    fun analyzeArtifacts(refs: List<String>, includeZipBase64: Boolean = true) {
        if (refs.isEmpty()) {
            _uiState.value = _uiState.value.copy(analysisStatus = "Select at least one artifact first.")
            return
        }
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(
                analysisLoading = true,
                analysisReportText = "Analyzing selected result JSON, terminal logs, and DAB logs...",
                analysisStatus = "Gemini is analyzing selected artifacts...",
                analysisReportId = "",
                analysisTxtName = "",
                analysisPdfName = ""
            )
            val request = YtsResultsAnalysisRequestDto(
                artifact_refs = refs,
                include_zip_base64 = includeZipBase64,
                analysis_model = "gemini-3.1-pro-preview",
                triage_level = "deep"
            )
            when (val result = ytsRepository.analyzeResultArtifacts(request)) {
                is ApiResult.Success -> {
                    val report = result.data
                    val failedReasons = report.failed_reasons.ifEmpty { listOf("No failed reasons found by heuristic scan.") }
                    val text = buildString {
                        appendLine("Report: ${report.report_id.ifBlank { "-" }}")
                        appendLine("Model: ${report.analysis_model.ifBlank { "gemini-3.1-pro-preview" }}")
                        appendLine("Triage level: ${report.triage_level.ifBlank { "deep" }}")
                        appendLine("Total tests found: ${report.total_tests}")
                        appendLine("Failed tests count: ${report.failed_tests}")
                        appendLine()
                        appendLine("Failed reasons:")
                        failedReasons.forEach { appendLine(it) }
                        appendLine()
                        appendLine("Gemini summary:")
                        appendLine(report.summary.ifBlank { "-" })
                    }
                    _uiState.value = _uiState.value.copy(
                        analysisLoading = false,
                        analysisReportText = text,
                        analysisStatus = "Deep triage complete - ${report.analysis_model.ifBlank { "Gemini Pro" }} - failed tests: ${report.failed_tests}",
                        analysisReportId = report.report_id,
                        analysisTxtName = report.txt_name,
                        analysisPdfName = report.pdf_name
                    )
                    refresh()
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(
                    analysisLoading = false,
                    analysisStatus = "Analysis failed: HTTP ${result.code}",
                    error = "HTTP ${result.code}: ${result.message}"
                )
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(
                    analysisLoading = false,
                    analysisStatus = "Analysis failed: network",
                    error = "Network error: ${result.throwable.message}"
                )
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(
                    analysisLoading = false,
                    analysisStatus = "Analysis failed",
                    error = "Unknown error: ${result.throwable.message}"
                )
            }
        }
    }

    fun refresh() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            when (val result = ytsRepository.fetchLiveCommands()) {
                is ApiResult.Success -> {
                    val activeId = resolveActiveCommandId(
                        items = result.data,
                        current = _uiState.value.activeCommandId
                    )
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        items = result.data,
                        activeCommandId = activeId,
                        sessionStatus = if (activeId == null) "No active YTS session." else _uiState.value.sessionStatus
                    )
                    if (activeId != null) {
                        refreshActiveCommand(activeId)
                    } else {
                        stopStream()
                        _uiState.value = _uiState.value.copy(activeCommand = null)
                    }
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = "HTTP ${result.code}: ${result.message}"
                )
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = "Network error: ${result.throwable.message}"
                )
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    error = "Unknown error: ${result.throwable.message}"
                )
            }
        }
        fetchArtifacts()
    }

    fun selectTab(tab: YtsWorkspaceTab) {
        _uiState.value = _uiState.value.copy(activeTab = tab)
    }

    fun selectCommand(commandId: String) {
        _uiState.value = _uiState.value.copy(
            activeCommandId = commandId,
            activeTab = YtsWorkspaceTab.RUNNING_SESSION,
            sessionStatus = "Loading session $commandId..."
        )
        refreshActiveCommand(commandId)
    }

    fun onPromptInputChanged(value: String) {
        _uiState.value = _uiState.value.copy(promptInput = value)
    }

    fun sendPromptResponse() {
        val commandId = _uiState.value.activeCommandId ?: return
        val text = _uiState.value.promptInput.trim()
        if (text.isEmpty()) return
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(sessionStatus = "Sending Gemini response...")
            when (val result = ytsRepository.respondToPrompt(commandId, text)) {
                is ApiResult.Success -> {
                    _uiState.value = _uiState.value.copy(
                        promptInput = "",
                        sessionStatus = "Response sent to YTS session."
                    )
                    refreshActiveCommand(commandId)
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(error = "Respond failed: HTTP ${result.code}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(error = "Respond failed: network")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(error = "Respond failed")
            }
        }
    }

    fun suggestPromptResponse(sendResponse: Boolean = false) {
        val commandId = _uiState.value.activeCommandId ?: return
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(sessionStatus = "Requesting Gemini suggestion...")
            when (val result = ytsRepository.suggestPrompt(commandId, sendResponse)) {
                is ApiResult.Success -> {
                    val suggestion = result.data["suggested_response"]?.jsonPrimitive?.contentOrNull
                        ?: result.data["response"]?.jsonPrimitive?.contentOrNull
                        ?: result.data["text"]?.jsonPrimitive?.contentOrNull
                    _uiState.value = _uiState.value.copy(
                        promptInput = if (!sendResponse && !suggestion.isNullOrBlank()) suggestion else _uiState.value.promptInput,
                        sessionStatus = if (sendResponse) {
                            "Gemini suggestion sent to the running test."
                        } else {
                            "Gemini suggestion ready for review."
                        }
                    )
                    refreshActiveCommand(commandId)
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(error = "Suggest failed: HTTP ${result.code}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(error = "Suggest failed: network")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(error = "Suggest failed")
            }
        }
    }

    fun stopActiveCommand() {
        val commandId = _uiState.value.activeCommandId ?: return
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(sessionStatus = "Stopping YTS session...")
            ytsRepository.stopLiveCommand(commandId)
            refreshActiveCommand(commandId)
            refresh()
        }
    }

    fun retryStream() {
        stopStream(clearFrame = false)
        startStream()
    }

    fun sendRemoteAction(action: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(remoteStatus = "Sending $action...")
            val request = ManualActionRequestDto(
                action = action,
                device_id = _uiState.value.deviceId.ifBlank { null }
            )
            when (val result = controlsRepository.manualAction(request)) {
                is ApiResult.Success -> {
                    _uiState.value = _uiState.value.copy(remoteStatus = "$action sent")
                }
                is ApiResult.HttpError -> {
                    _uiState.value = _uiState.value.copy(remoteStatus = "$action failed: HTTP ${result.code}")
                }
                is ApiResult.NetworkError -> {
                    _uiState.value = _uiState.value.copy(remoteStatus = "$action failed: network")
                }
                is ApiResult.UnknownError -> {
                    _uiState.value = _uiState.value.copy(remoteStatus = "$action failed")
                }
            }
        }
    }

    fun loadCatalog(refresh: Boolean) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isCatalogLoading = true, error = null)
            val guided = _uiState.value.guidedMode
            val result = if (refresh) ytsRepository.refreshTests(guided) else ytsRepository.fetchTests(guided)
            when (result) {
                is ApiResult.Success -> {
                    val filtered = filterCatalog(
                        catalog = result.data,
                        suite = _uiState.value.suiteFilter,
                        category = _uiState.value.categoryFilter,
                        query = _uiState.value.searchQuery
                    )
                    val validSelections = _uiState.value.selectedTestIds.filter { selectedId ->
                        result.data.any { it.test_id == selectedId }
                    }
                    _uiState.value = _uiState.value.copy(
                        isCatalogLoading = false,
                        catalog = result.data,
                        filteredCatalog = filtered,
                        selectedTestIds = validSelections,
                        startStatus = "Loaded ${result.data.size} ${if (guided) "guided " else ""}tests."
                    )
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(
                    isCatalogLoading = false,
                    error = "HTTP ${result.code}: ${result.message}"
                )
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(
                    isCatalogLoading = false,
                    error = "Network error: ${result.throwable.message}"
                )
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(
                    isCatalogLoading = false,
                    error = "Unknown error: ${result.throwable.message}"
                )
            }
        }
    }

    fun loadRuntimeModels() {
        viewModelScope.launch {
            when (val result = ytsRepository.fetchRuntimeModels()) {
                is ApiResult.Success -> applyRuntimeModelState(result.data)
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(error = "HTTP ${result.code}: ${result.message}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(error = "Network error: ${result.throwable.message}")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(error = "Unknown error: ${result.throwable.message}")
            }
        }
    }

    fun onDeviceIdChanged(value: String) {
        viewModelScope.launch {
            when (val result = controlsRepository.selectDeviceContext(value, persist = true)) {
                is ApiResult.Success -> {
                    val selected = applySharedDeviceContext(result.data)
                    apiSettingsStore.saveSelectedDeviceId(selected)
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(error = "HTTP ${result.code}: ${result.message}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(error = "Network error: ${result.throwable.message}")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(error = "Unknown error: ${result.throwable.message}")
            }
        }
    }

    fun onJsonOutputChanged(value: String) {
        _uiState.value = _uiState.value.copy(jsonOutputFile = value)
    }

    fun onSuiteFilterChanged(value: String) {
        _uiState.value = _uiState.value.copy(
            suiteFilter = value,
            filteredCatalog = filterCatalog(
                catalog = _uiState.value.catalog,
                suite = value,
                category = _uiState.value.categoryFilter,
                query = _uiState.value.searchQuery
            )
        )
    }

    fun onCategoryFilterChanged(value: String) {
        _uiState.value = _uiState.value.copy(
            categoryFilter = value,
            filteredCatalog = filterCatalog(
                catalog = _uiState.value.catalog,
                suite = _uiState.value.suiteFilter,
                category = value,
                query = _uiState.value.searchQuery
            )
        )
    }

    fun onSearchQueryChanged(value: String) {
        _uiState.value = _uiState.value.copy(
            searchQuery = value,
            filteredCatalog = filterCatalog(
                catalog = _uiState.value.catalog,
                suite = _uiState.value.suiteFilter,
                category = _uiState.value.categoryFilter,
                query = value
            )
        )
    }

    fun toggleGuidedMode() {
        _uiState.value = _uiState.value.copy(guidedMode = !_uiState.value.guidedMode)
        loadCatalog(refresh = true)
    }

    fun toggleInteractiveAi() {
        _uiState.value = _uiState.value.copy(interactiveAi = !_uiState.value.interactiveAi)
    }

    fun toggleRecordVideo() {
        val next = !_uiState.value.recordVideo
        _uiState.value = _uiState.value.copy(
            recordVideo = next,
            recordAudio = if (next) _uiState.value.recordAudio else false
        )
    }

    fun toggleRecordAudio() {
        _uiState.value = _uiState.value.copy(recordAudio = !_uiState.value.recordAudio)
    }

    fun onFilterTokensChanged(value: String) {
        _uiState.value = _uiState.value.copy(filterTokensInput = value)
    }

    fun onExtraArgsChanged(value: String) {
        _uiState.value = _uiState.value.copy(extraArgsInput = value)
    }

    fun onPlannerModelChanged(value: String) {
        _uiState.value = _uiState.value.copy(plannerModel = value)
    }

    fun onLiveModelChanged(value: String) {
        _uiState.value = _uiState.value.copy(liveModel = value)
    }

    fun toggleTestSelection(testId: String) {
        val current = _uiState.value.selectedTestIds.toMutableList()
        if (current.contains(testId)) current.remove(testId) else current.add(testId)
        _uiState.value = _uiState.value.copy(selectedTestIds = current)
    }

    fun clearSelectedTests() {
        _uiState.value = _uiState.value.copy(selectedTestIds = emptyList())
    }

    fun applyRuntimeModels() {
        viewModelScope.launch {
            val planner = _uiState.value.plannerModel.trim()
            val live = _uiState.value.liveModel.trim()
            val plannerDef = async { if (planner.isNotBlank()) ytsRepository.updateRuntimeModel(planner, "planner") else null }
            val liveDef = async { if (live.isNotBlank()) ytsRepository.updateRuntimeModel(live, "live") else null }
            val plannerResult = plannerDef.await()
            val liveResult = liveDef.await()
            val finalResult = liveResult ?: plannerResult
            when (finalResult) {
                is ApiResult.Success -> applyRuntimeModelState(finalResult.data)
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(error = "HTTP ${finalResult.code}: ${finalResult.message}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(error = "Network error: ${finalResult.throwable.message}")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(error = "Unknown error: ${finalResult.throwable.message}")
                null -> Unit
            }
        }
    }

    fun startGuidedTestRun() {
        viewModelScope.launch {
            val body = buildTestRequest() ?: return@launch
            val testCount = body.params.drop(1).takeWhile { !it.startsWith("--") }.size
            _uiState.value = _uiState.value.copy(isStarting = true, error = null, startStatus = "Starting YTS run...")
            when (val result = ytsRepository.startLiveCommand(body)) {
                is ApiResult.Success -> {
                    _uiState.value = _uiState.value.copy(
                        isStarting = false,
                        activeTab = YtsWorkspaceTab.RUNNING_SESSION,
                        lastStartedCommand = result.data,
                        activeCommandId = result.data.command_id,
                        commandTestCountHints = _uiState.value.commandTestCountHints + (result.data.command_id to testCount),
                        startStatus = "YTS run started: ${result.data.command_id}",
                        sessionStatus = "YTS session created. Connecting live view..."
                    )
                    refresh()
                    refreshActiveCommand(result.data.command_id)
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(
                    isStarting = false,
                    error = "HTTP ${result.code}: ${result.message}",
                    startStatus = "YTS run failed to start."
                )
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(
                    isStarting = false,
                    error = "Network error: ${result.throwable.message}",
                    startStatus = "YTS run failed to start."
                )
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(
                    isStarting = false,
                    error = "Unknown error: ${result.throwable.message}",
                    startStatus = "YTS run failed to start."
                )
            }
        }
    }

    private fun startPolling() {
        pollJob?.cancel()
        pollJob = viewModelScope.launch {
            while (isActive) {
                val activeId = _uiState.value.activeCommandId
                if (activeId != null) {
                    refreshActiveCommand(activeId)
                } else {
                    refresh()
                }
                delay(2000)
            }
        }
    }

    private fun refreshActiveCommand(commandId: String) {
        viewModelScope.launch {
            when (val result = ytsRepository.fetchLiveCommandState(commandId)) {
                is ApiResult.Success -> {
                    val data = result.data
                    _uiState.value = _uiState.value.copy(
                        activeCommandId = data.command_id,
                        activeCommand = data,
                        sessionStatus = buildSessionStatus(data)
                    )
                    if (data.status in terminalStates) {
                        stopStream()
                        refreshRecentCommands(data.command_id)
                    } else {
                        startStream()
                    }
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(error = "HTTP ${result.code}: ${result.message}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(error = "Network error: ${result.throwable.message}")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(error = "Unknown error: ${result.throwable.message}")
            }
        }
    }

    private fun refreshRecentCommands(commandId: String) {
        viewModelScope.launch {
            when (val result = ytsRepository.fetchLiveCommands()) {
                is ApiResult.Success -> {
                    val activeId = resolveActiveCommandId(result.data, commandId)
                    _uiState.value = _uiState.value.copy(
                        items = result.data,
                        activeCommandId = activeId ?: commandId
                    )
                }
                else -> Unit
            }
        }
    }

    private fun startStream() {
        if (_uiState.value.isStreaming && streamJob != null) return
        streamJob?.cancel()
        _uiState.value = _uiState.value.copy(
            isStreaming = true,
            streamStatus = "Connecting to device live stream..."
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
            streamStatus = if (clearFrame) "Stream idle." else "Reconnecting stream..."
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
                        streamStatus = if (firstFrame) "Live device stream connected." else _uiState.value.streamStatus
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

    private fun buildTestRequest(): YtsLiveCommandRequestDto? {
        val deviceId = _uiState.value.deviceId.trim()
        val runnerDeviceId = _uiState.value.ytsShortId.ifBlank {
            _uiState.value.ytsDeviceId.ifBlank { deviceId }
        }
        if (deviceId.isBlank()) {
            _uiState.value = _uiState.value.copy(error = "Device ID is required.", startStatus = "Device ID is required.")
            return null
        }
        if (runnerDeviceId.isBlank()) {
            _uiState.value = _uiState.value.copy(
                error = "The selected device has no YTS mapping yet.",
                startStatus = "YTS device mapping missing."
            )
            return null
        }
        val selectedIds = _uiState.value.selectedTestIds
        val fallbackIds = if (selectedIds.isEmpty()) {
            _uiState.value.filteredCatalog.map { it.test_id }.take(25)
        } else {
            selectedIds
        }
        val filterTokens = tokenize(_uiState.value.filterTokensInput)
        val extraArgs = tokenize(_uiState.value.extraArgsInput)
        if (fallbackIds.isEmpty() && filterTokens.isEmpty() && !_uiState.value.guidedMode && extraArgs.isEmpty()) {
            _uiState.value = _uiState.value.copy(
                error = "Select tests or provide filter tokens before running.",
                startStatus = "No tests selected."
            )
            return null
        }

        val params = mutableListOf<String>()
        params += runnerDeviceId
        params += fallbackIds
        params += filterTokens
        if (_uiState.value.guidedMode) params += "--guided"
        params += extraArgs
        val jsonOutput = _uiState.value.jsonOutputFile.trim()
        if (jsonOutput.isNotBlank()) {
            params += "--json-output"
            params += jsonOutput
        }

        return YtsLiveCommandRequestDto(
            command = "test",
            params = params,
            global_options = buildGlobalOptions(),
            output_file = jsonOutput.ifBlank { null },
            interactive_ai = _uiState.value.interactiveAi,
            record_video = _uiState.value.recordVideo,
            record_audio = _uiState.value.recordVideo && _uiState.value.recordAudio,
            device_id = runnerDeviceId
        )
    }

    private fun refreshSharedDeviceContext() {
        viewModelScope.launch {
            when (val result = controlsRepository.fetchCurrentDeviceContext()) {
                is ApiResult.Success -> applySharedDeviceContext(result.data)
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(error = "HTTP ${result.code}: ${result.message}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(error = "Network error: ${result.throwable.message}")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(error = "Unknown error: ${result.throwable.message}")
            }
        }
    }

    private fun applySharedDeviceContext(data: CurrentDeviceContextResponseDto): String {
        val context = data.context
        val selected = data.selected_device_id?.takeIf { it.isNotBlank() }
            ?: context?.dabDeviceId.orEmpty()
        _uiState.value = _uiState.value.copy(
            deviceId = selected,
            deviceDisplayName = context?.displayName.orEmpty(),
            ytsDeviceId = context?.ytsDeviceId.orEmpty(),
            ytsShortId = data.validation?.ytsShortId.orEmpty(),
            irDeviceId = context?.irDeviceId.orEmpty(),
            sharedDeviceReady = data.validation?.valid ?: false,
            sharedDeviceIssues = data.validation?.issues.orEmpty()
        )
        return selected
    }

    private fun buildGlobalOptions(): JsonObject = buildJsonObject { }

    private fun applyRuntimeModelState(data: RuntimeModelResponseDto) {
        _uiState.value = _uiState.value.copy(
            plannerModel = data.active_vertex_planner_model,
            liveModel = data.active_vertex_live_model,
            availableModels = data.available_models,
            modelStatus = data.message
        )
    }

    private fun filterCatalog(
        catalog: List<YtsTestCatalogItemDto>,
        suite: String,
        category: String,
        query: String
    ): List<YtsTestCatalogItemDto> {
        val normalizedQuery = query.trim().lowercase()
        return catalog.filter { test ->
            if (suite.isNotBlank() && test.test_suite != suite) return@filter false
            if (category.isNotBlank() && test.test_category != category) return@filter false
            if (normalizedQuery.isBlank()) return@filter true
            listOf(test.test_id, test.test_title, test.test_suite, test.test_category)
                .filterNotNull()
                .any { it.lowercase().contains(normalizedQuery) }
        }
    }

    private fun tokenize(value: String): List<String> {
        return value
            .split(Regex("\\s+"))
            .map { it.trim() }
            .filter { it.isNotBlank() }
    }

    private fun resolveActiveCommandId(
        items: List<YtsLiveCommandSummaryDto>,
        current: String?
    ): String? {
        if (current != null && items.any { it.command_id == current }) return current
        return items.firstOrNull { it.status !in terminalStates }?.command_id
    }

    private fun buildSessionStatus(data: YtsLiveCommandStateDto): String {
        val pieces = mutableListOf("Status: ${data.status}")
        data.updated_at?.let { pieces += "Updated $it" }
        data.returncode?.let { pieces += "Exit $it" }
        if (data.awaiting_input) pieces += "Waiting for Gemini/operator response"
        if (data.video_recording_status != null) pieces += "Video ${data.video_recording_status}"
        return pieces.joinToString("  •  ")
    }

    override fun onCleared() {
        pollJob?.cancel()
        streamJob?.cancel()
        super.onCleared()
    }

    companion object {
        private val terminalStates = setOf("completed", "stopped", "failed")
    }
}
