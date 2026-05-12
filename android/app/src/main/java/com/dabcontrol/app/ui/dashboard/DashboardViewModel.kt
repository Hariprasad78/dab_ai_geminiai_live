package com.dabcontrol.app.ui.dashboard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.api.RuntimeModelResponseDto
import com.dabcontrol.app.data.preferences.ApiSettingsStore
import com.dabcontrol.app.data.repo.ControlsRepository
import com.dabcontrol.app.data.repo.DashboardRepository
import com.dabcontrol.app.data.repo.YtsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.isActive
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonPrimitive

@HiltViewModel
class DashboardViewModel @Inject constructor(
    private val dashboardRepository: DashboardRepository,
    private val controlsRepository: ControlsRepository,
    private val ytsRepository: YtsRepository,
    private val apiSettingsStore: ApiSettingsStore
) : ViewModel() {
    private val _uiState = MutableStateFlow(DashboardUiState())
    val uiState: StateFlow<DashboardUiState> = _uiState.asStateFlow()
    private var autoRefreshJob: Job? = null
    private var refreshJob: Job? = null

    init {
        viewModelScope.launch {
            apiSettingsStore.apiBaseUrl.collectLatest { url ->
                _uiState.value = _uiState.value.copy(apiBaseUrl = url)
            }
        }
        viewModelScope.launch {
            apiSettingsStore.selectedDeviceId.collectLatest { deviceId ->
                _uiState.value = _uiState.value.copy(selectedDeviceId = deviceId)
            }
        }
        refresh(silent = false)
        startAutoRefresh()
    }

    fun onApiBaseUrlChanged(value: String) {
        _uiState.value = _uiState.value.copy(apiBaseUrl = value)
    }

    fun onDeviceSelected(value: String) {
        _uiState.value = _uiState.value.copy(selectedDeviceId = value)
        viewModelScope.launch {
            apiSettingsStore.saveSelectedDeviceId(value)
        }
    }

    fun onPlannerModelChanged(value: String) {
        _uiState.value = _uiState.value.copy(plannerModel = value)
    }

    fun onLiveModelChanged(value: String) {
        _uiState.value = _uiState.value.copy(liveModel = value)
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

    fun saveApiBaseUrl() {
        viewModelScope.launch {
            apiSettingsStore.saveApiBaseUrl(_uiState.value.apiBaseUrl)
            refresh(silent = false)
        }
    }

    fun toggleAutoRefresh() {
        val enabled = !_uiState.value.autoRefreshEnabled
        _uiState.value = _uiState.value.copy(
            autoRefreshEnabled = enabled,
            refreshStateLabel = if (enabled) {
                "Live sync every ${_uiState.value.refreshIntervalSeconds}s"
            } else {
                "Manual refresh"
            }
        )
        if (enabled) {
            startAutoRefresh()
        } else {
            autoRefreshJob?.cancel()
            autoRefreshJob = null
        }
    }

    fun refresh(silent: Boolean = false) {
        if (refreshJob?.isActive == true) return
        refreshJob = viewModelScope.launch {
            if (!silent) {
                _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            }
            val healthDeferred = async { dashboardRepository.fetchHealth() }
            val metricsDeferred = async { dashboardRepository.fetchMetrics() }
            val devicesDeferred = async { controlsRepository.fetchDevices() }
            val modelsDeferred = async { ytsRepository.fetchRuntimeModels() }

            val healthResult = healthDeferred.await()
            val metricsResult = metricsDeferred.await()
            val devicesResult = devicesDeferred.await()
            val modelsResult = modelsDeferred.await()

            val healthStatus = when (healthResult) {
                is ApiResult.Success -> healthResult.data.status
                is ApiResult.HttpError -> "HTTP ${healthResult.code}"
                is ApiResult.NetworkError -> "Network error"
                is ApiResult.UnknownError -> "Unknown error"
            }

            val mode = if (healthResult is ApiResult.Success) healthResult.data.mode ?: "--" else "--"
            val metricsPreview = toMetricsPreview(metricsResult)
            val metrics = extractMetrics(metricsResult)
            val deviceIds = extractDeviceIds(devicesResult)
            val preferred = _uiState.value.selectedDeviceId
            val resolvedSelected = when {
                preferred.isNotBlank() && deviceIds.contains(preferred) -> preferred
                devicesResult is ApiResult.Success && !devicesResult.data.selected_device_id.isNullOrBlank() ->
                    devicesResult.data.selected_device_id.orEmpty()
                else -> deviceIds.firstOrNull().orEmpty()
            }
            if (resolvedSelected.isNotBlank() && resolvedSelected != preferred) {
                apiSettingsStore.saveSelectedDeviceId(resolvedSelected)
            }
            if (modelsResult is ApiResult.Success) {
                applyRuntimeModelState(modelsResult.data)
            }

            _uiState.value = _uiState.value.copy(
                isLoading = false,
                healthStatus = healthStatus,
                mode = mode,
                deviceIds = deviceIds,
                selectedDeviceId = resolvedSelected,
                cpuPercent = metrics?.cpuPercent,
                ramPercent = metrics?.ramPercent,
                load1m = metrics?.load1m,
                cpuTempC = metrics?.cpuTempC,
                cpuCount = metrics?.cpuCount,
                timestamp = metrics?.timestamp ?: _uiState.value.timestamp,
                cpuHistory = appendHistory(_uiState.value.cpuHistory, metrics?.timestampShort, metrics?.cpuPercent),
                ramHistory = appendHistory(_uiState.value.ramHistory, metrics?.timestampShort, metrics?.ramPercent),
                loadHistory = appendHistory(_uiState.value.loadHistory, metrics?.timestampShort, metrics?.load1m),
                tempHistory = appendHistory(_uiState.value.tempHistory, metrics?.timestampShort, metrics?.cpuTempC),
                metricsPreview = metricsPreview,
                backendStatusSummary = buildBackendStatusSummary(
                    healthStatus = healthStatus,
                    mode = mode,
                    deviceCount = deviceIds.size,
                    selectedDeviceId = resolvedSelected,
                    modelsResult = modelsResult
                ),
                refreshStateLabel = buildRefreshStateLabel(
                    timestamp = metrics?.timestampShort,
                    silent = silent
                ),
                error = buildError(healthResult, metricsResult, devicesResult, modelsResult)
            )
        }
    }

    private fun startAutoRefresh() {
        autoRefreshJob?.cancel()
        autoRefreshJob = viewModelScope.launch {
            while (isActive) {
                delay(_uiState.value.refreshIntervalSeconds * 1000L)
                if (_uiState.value.autoRefreshEnabled) {
                    refresh(silent = true)
                }
            }
        }
    }

    private fun applyRuntimeModelState(data: RuntimeModelResponseDto) {
        _uiState.value = _uiState.value.copy(
            plannerModel = data.active_vertex_planner_model,
            liveModel = data.active_vertex_live_model,
            availableModels = data.available_models,
            modelStatus = data.message
        )
    }

    private fun extractDeviceIds(result: ApiResult<com.dabcontrol.app.data.api.DabDevicesResponseDto>): List<String> {
        val payload = (result as? ApiResult.Success)?.data ?: return emptyList()
        return payload.devices
            .mapNotNull { it["device_id"]?.jsonPrimitive?.contentOrNull }
            .filter { it.isNotBlank() }
    }

    private fun extractMetrics(result: ApiResult<JsonObject>): MetricsSnapshot? {
        val payload = (result as? ApiResult.Success)?.data ?: return null
        return MetricsSnapshot(
            timestamp = payload.stringValue("timestamp") ?: "--",
            timestampShort = payload.stringValue("timestamp")
                ?.substringAfter('T')
                ?.substringBefore('.')
                ?: "--",
            cpuPercent = payload.floatValue("cpu_percent"),
            ramPercent = payload.floatValue("ram_percent"),
            load1m = payload.floatValue("load_1m"),
            cpuTempC = payload.floatValue("cpu_temp_c"),
            cpuCount = payload.intValue("cpu_count")
        )
    }

    private fun toMetricsPreview(result: ApiResult<JsonObject>): String {
        return when (result) {
            is ApiResult.Success -> {
                val cpu = result.data.floatValue("cpu_percent")
                val ram = result.data.floatValue("ram_percent")
                val load = result.data.floatValue("load_1m")
                val temp = result.data.floatValue("cpu_temp_c")
                listOfNotNull(
                    cpu?.let { "CPU ${it.format1()}%" },
                    ram?.let { "RAM ${it.format1()}%" },
                    load?.let { "Load ${it.format2()}" },
                    temp?.let { "Temp ${it.format1()}C" }
                ).ifEmpty { listOf("No metrics values") }.joinToString("  |  ")
            }
            is ApiResult.HttpError -> "HTTP ${result.code}"
            is ApiResult.NetworkError -> "Network error"
            is ApiResult.UnknownError -> "Unknown error"
        }
    }

    private fun buildError(
        health: ApiResult<*>,
        metrics: ApiResult<*>,
        devices: ApiResult<*>,
        models: ApiResult<*>
    ): String? {
        val issues = mutableListOf<String>()
        if (health !is ApiResult.Success) issues.add("Health failed")
        if (metrics !is ApiResult.Success) issues.add("Metrics failed")
        if (devices !is ApiResult.Success) issues.add("Devices failed")
        if (models !is ApiResult.Success) issues.add("Gemini models failed")
        return if (issues.isEmpty()) null else issues.joinToString(" · ")
    }

    private fun buildBackendStatusSummary(
        healthStatus: String,
        mode: String,
        deviceCount: Int,
        selectedDeviceId: String,
        modelsResult: ApiResult<*>
    ): String {
        val modelState = if (modelsResult is ApiResult.Success) "models synced" else "model sync issue"
        return listOf(
            "health $healthStatus",
            "mode $mode",
            "$deviceCount device${if (deviceCount == 1) "" else "s"} visible",
            if (selectedDeviceId.isBlank()) "no device selected" else "device $selectedDeviceId",
            modelState
        ).joinToString(" · ")
    }

    private fun buildRefreshStateLabel(
        timestamp: String?,
        silent: Boolean
    ): String {
        val cadence = "every ${_uiState.value.refreshIntervalSeconds}s"
        val sample = timestamp?.takeIf { it.isNotBlank() && it != "--" } ?: "waiting"
        return if (silent) {
            "Live sync $cadence · last sample $sample"
        } else {
            "Manual refresh complete · last sample $sample"
        }
    }

    private fun appendHistory(
        current: List<MetricPoint>,
        label: String?,
        value: Float?
    ): List<MetricPoint> {
        if (label == null || value == null) return current
        return (current + MetricPoint(label, value)).takeLast(HISTORY_LIMIT)
    }

    private fun JsonObject.floatValue(key: String): Float? = get(key)?.asFloat()

    private fun JsonObject.intValue(key: String): Int? = get(key)?.jsonPrimitive?.intOrNull

    private fun JsonObject.stringValue(key: String): String? = get(key)?.jsonPrimitive?.contentOrNull

    private fun JsonElement.asFloat(): Float? = jsonPrimitive.doubleOrNull?.toFloat()

    private fun Float.format1(): String = String.format("%.1f", this)

    private fun Float.format2(): String = String.format("%.2f", this)

    private data class MetricsSnapshot(
        val timestamp: String,
        val timestampShort: String,
        val cpuPercent: Float?,
        val ramPercent: Float?,
        val load1m: Float?,
        val cpuTempC: Float?,
        val cpuCount: Int?
    )

    companion object {
        private const val HISTORY_LIMIT = 12
    }
}
