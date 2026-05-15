package com.dabcontrol.app.ui.deviceinfo

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.preferences.ApiSettingsStore
import com.dabcontrol.app.data.repo.ControlsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive

@HiltViewModel
class DeviceInfoViewModel @Inject constructor(
    private val controlsRepository: ControlsRepository,
    private val apiSettingsStore: ApiSettingsStore
) : ViewModel() {
    private val _uiState = MutableStateFlow(DeviceInfoUiState())
    val uiState: StateFlow<DeviceInfoUiState> = _uiState.asStateFlow()

    private val prettyJson = Json { prettyPrint = true }

    init {
        viewModelScope.launch {
            apiSettingsStore.selectedDeviceId.collectLatest { deviceId ->
                _uiState.value = _uiState.value.copy(selectedDeviceId = deviceId)
            }
        }
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)

            val devicesDeferred = async { controlsRepository.fetchDevices() }
            val contextDeferred = async { controlsRepository.fetchCurrentDeviceContext() }
            val devicesResult = devicesDeferred.await()
            val contextResult = contextDeferred.await()
            val deviceIds = extractDeviceIds(devicesResult)
            val selectedDeviceId = resolveSelectedDeviceId(
                current = _uiState.value.selectedDeviceId,
                deviceIds = deviceIds,
                devicesResult = devicesResult
            )
            val context = (contextResult as? ApiResult.Success)?.data?.context

            if (selectedDeviceId.isBlank()) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    selectedDeviceName = context?.displayName.orEmpty(),
                    selectedYtsDeviceId = context?.ytsDeviceId.orEmpty(),
                    selectedIrDeviceId = context?.irDeviceId.orEmpty(),
                    deviceIds = deviceIds,
                    rows = emptyList(),
                    error = null
                )
                return@launch
            }

            val infoResult = controlsRepository.fetchDeviceInfo(selectedDeviceId)
            _uiState.value = _uiState.value.copy(
                isLoading = false,
                deviceIds = deviceIds,
                selectedDeviceId = selectedDeviceId,
                selectedDeviceName = context?.displayName.orEmpty(),
                selectedYtsDeviceId = context?.ytsDeviceId.orEmpty(),
                selectedIrDeviceId = context?.irDeviceId.orEmpty(),
                rows = buildRows(infoResult),
                error = buildError(infoResult)
            )
        }
    }

    private fun extractDeviceIds(result: ApiResult<com.dabcontrol.app.data.api.DabDevicesResponseDto>): List<String> {
        val payload = (result as? ApiResult.Success)?.data ?: return emptyList()
        return payload.devices
            .mapNotNull { it["device_id"]?.jsonPrimitive?.contentOrNull }
            .filter { it.isNotBlank() }
    }

    private fun resolveSelectedDeviceId(
        current: String,
        deviceIds: List<String>,
        devicesResult: ApiResult<com.dabcontrol.app.data.api.DabDevicesResponseDto>
    ): String {
        return when {
            current.isNotBlank() && deviceIds.contains(current) -> current
            devicesResult is ApiResult.Success && !devicesResult.data.selected_device_id.isNullOrBlank() ->
                devicesResult.data.selected_device_id.orEmpty()
            else -> deviceIds.firstOrNull().orEmpty()
        }
    }

    private fun buildRows(result: ApiResult<JsonObject>): List<DeviceInfoRow> {
        val payload = (result as? ApiResult.Success)?.data ?: return emptyList()
        val rows = mutableListOf<DeviceInfoRow>()
        payload["success"]?.let { rows += DeviceInfoRow("success", prettyValue(it), isStructured = false) }
        payload["device_id"]?.let { rows += DeviceInfoRow("device_id", prettyValue(it), isStructured = false) }
        val resultBody = payload["result"]
        if (resultBody is JsonObject) {
            resultBody.forEach { (key, value) ->
                rows += DeviceInfoRow(
                    field = key,
                    value = prettyValue(value),
                    isStructured = value is JsonObject || value is JsonArray
                )
            }
        } else if (resultBody != null) {
            rows += DeviceInfoRow(
                field = "result",
                value = prettyValue(resultBody),
                isStructured = resultBody is JsonObject || resultBody is JsonArray
            )
        } else {
            payload.forEach { (key, value) ->
                rows += DeviceInfoRow(
                    field = key,
                    value = prettyValue(value),
                    isStructured = value is JsonObject || value is JsonArray
                )
            }
        }
        return rows
    }

    private fun prettyValue(value: JsonElement): String {
        return when (value) {
            is JsonPrimitive -> primitiveValue(value)
            is JsonObject, is JsonArray -> prettyJson.encodeToString(JsonElement.serializer(), value)
        }
    }

    private fun primitiveValue(value: JsonPrimitive): String {
        return value.contentOrNull ?: value.toString()
    }

    private fun buildError(result: ApiResult<JsonObject>): String? {
        return when (result) {
            is ApiResult.Success -> null
            is ApiResult.HttpError -> "HTTP ${result.code}: ${result.message}"
            is ApiResult.NetworkError -> "Network error: ${result.throwable.message}"
            is ApiResult.UnknownError -> "Unknown error: ${result.throwable.message}"
        }
    }
}
