package com.dabcontrol.app.data.repo

import com.dabcontrol.app.data.api.ApiClientFactory
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.api.DabDevicesResponseDto
import com.dabcontrol.app.data.api.IrSendRequestDto
import com.dabcontrol.app.data.api.IrTrainRequestDto
import com.dabcontrol.app.data.api.ManualActionBatchRequestDto
import com.dabcontrol.app.data.api.ManualActionBatchResponseDto
import com.dabcontrol.app.data.api.ManualActionRequestDto
import com.dabcontrol.app.data.api.ManualActionResponseDto
import com.dabcontrol.app.data.api.PlannerDebugRequestDto
import com.dabcontrol.app.data.api.TaskMacroRequestDto
import com.dabcontrol.app.data.api.safeApiCall
import com.dabcontrol.app.data.preferences.ApiSettingsStore
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.JsonObject

@Singleton
class ControlsRepository @Inject constructor(
    private val apiSettingsStore: ApiSettingsStore,
    private val apiClientFactory: ApiClientFactory
) {
    private suspend fun service() = apiClientFactory.create(apiSettingsStore.apiBaseUrl.first())

    suspend fun fetchDevices(): ApiResult<DabDevicesResponseDto> = safeApiCall { service().dabDevices() }

    suspend fun fetchDeviceInfo(deviceId: String?): ApiResult<JsonObject> =
        safeApiCall { service().dabDeviceInfo(deviceId) }

    suspend fun fetchCapabilityStatus(deviceId: String?, refresh: Boolean): ApiResult<JsonObject> =
        safeApiCall { service().dabDeviceCapabilityStatus(deviceId, refresh) }

    suspend fun fetchOperationsGrid(deviceId: String?, refresh: Boolean): ApiResult<JsonObject> =
        safeApiCall { service().dabDeviceOperationsGrid(deviceId, refresh) }

    suspend fun fetchCurrentSettings(deviceId: String?, refresh: Boolean): ApiResult<JsonObject> =
        safeApiCall { service().dabDeviceCurrentSettings(deviceId, refresh) }

    suspend fun manualAction(request: ManualActionRequestDto): ApiResult<ManualActionResponseDto> =
        safeApiCall { service().manualAction(request) }

    suspend fun manualBatch(request: ManualActionBatchRequestDto): ApiResult<ManualActionBatchResponseDto> =
        safeApiCall { service().manualActionsBatch(request) }

    suspend fun irStatus(): ApiResult<JsonObject> = safeApiCall { service().irStatus() }

    suspend fun irDevices(): ApiResult<JsonObject> = safeApiCall { service().irDevices() }

    suspend fun irDeviceKeys(deviceId: String): ApiResult<JsonObject> = safeApiCall { service().irDeviceKeys(deviceId) }

    suspend fun irSend(request: IrSendRequestDto): ApiResult<JsonObject> = safeApiCall { service().irSend(request) }

    suspend fun irTrain(request: IrTrainRequestDto): ApiResult<JsonObject> = safeApiCall { service().irTrain(request) }

    suspend fun taskMacro(request: TaskMacroRequestDto): ApiResult<JsonObject> = safeApiCall { service().taskMacro(request) }

    suspend fun plannerDebug(request: PlannerDebugRequestDto): ApiResult<JsonObject> =
        safeApiCall { service().plannerDebug(request) }
}
