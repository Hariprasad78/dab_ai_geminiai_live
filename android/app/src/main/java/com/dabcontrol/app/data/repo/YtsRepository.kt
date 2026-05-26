package com.dabcontrol.app.data.repo

import com.dabcontrol.app.data.api.ApiClientFactory
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.api.RuntimeModelResponseDto
import com.dabcontrol.app.data.api.RuntimeModelUpdateRequestDto
import com.dabcontrol.app.data.api.YtsLiveCommandStateDto
import com.dabcontrol.app.data.api.YtsLiveCommandRequestDto
import com.dabcontrol.app.data.api.YtsLiveCommandStartResponseDto
import com.dabcontrol.app.data.api.YtsLiveCommandSummaryDto
import com.dabcontrol.app.data.api.YtsPromptResponseRequest
import com.dabcontrol.app.data.api.YtsPromptSuggestRequest
import com.dabcontrol.app.data.api.YtsTestCatalogItemDto
import com.dabcontrol.app.data.api.YtsResultArtifactItemDto
import com.dabcontrol.app.data.api.YtsResultsAnalysisResponseDto
import com.dabcontrol.app.data.api.YtsResultsAnalysisRequestDto
import com.dabcontrol.app.data.api.safeApiCall
import com.dabcontrol.app.data.preferences.ApiSettingsStore
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.JsonObject

@Singleton
class YtsRepository @Inject constructor(
    private val apiSettingsStore: ApiSettingsStore,
    private val apiClientFactory: ApiClientFactory
) {
    private suspend fun service() = apiClientFactory.create(apiSettingsStore.apiBaseUrl.first())

    suspend fun fetchLiveCommands(limit: Int = 100): ApiResult<List<YtsLiveCommandSummaryDto>> {
        return safeApiCall { service().ytsLiveCommands(limit = limit) }
    }

    suspend fun fetchLiveCommandState(commandId: String): ApiResult<YtsLiveCommandStateDto> {
        return safeApiCall { service().ytsLiveCommandState(commandId) }
    }

    suspend fun startLiveCommand(request: YtsLiveCommandRequestDto): ApiResult<YtsLiveCommandStartResponseDto> {
        return safeApiCall { service().startYtsLiveCommand(request) }
    }

    suspend fun stopLiveCommand(commandId: String): ApiResult<JsonObject> {
        return safeApiCall { service().stopYtsLiveCommand(commandId) }
    }

    suspend fun respondToPrompt(commandId: String, response: String): ApiResult<JsonObject> {
        return safeApiCall { service().respondYtsLiveCommand(commandId, YtsPromptResponseRequest(response)) }
    }

    suspend fun suggestPrompt(commandId: String, sendResponse: Boolean): ApiResult<JsonObject> {
        return safeApiCall { service().suggestYtsLiveCommand(commandId, YtsPromptSuggestRequest(send_response = sendResponse)) }
    }

    suspend fun fetchResultArtifacts(limit: Int = 100): ApiResult<List<YtsResultArtifactItemDto>> {
        return safeApiCall { service().ytsResultsArtifacts(limit) }
    }

    suspend fun analyzeResultArtifacts(request: YtsResultsAnalysisRequestDto): ApiResult<YtsResultsAnalysisResponseDto> {
        return safeApiCall { service().analyzeYtsResultArtifacts(request) }
    }

    suspend fun fetchTests(guided: Boolean): ApiResult<List<YtsTestCatalogItemDto>> {
        return safeApiCall { service().ytsTests(guided) }
    }

    suspend fun refreshTests(guided: Boolean): ApiResult<List<YtsTestCatalogItemDto>> {
        return safeApiCall { service().refreshYtsTests(guided) }
    }

    suspend fun fetchRuntimeModels(): ApiResult<RuntimeModelResponseDto> {
        return safeApiCall { service().runtimeModelSummary() }
    }

    suspend fun updateRuntimeModel(model: String, target: String): ApiResult<RuntimeModelResponseDto> {
        return safeApiCall { service().updateRuntimeModel(RuntimeModelUpdateRequestDto(model = model, target = target)) }
    }
}
