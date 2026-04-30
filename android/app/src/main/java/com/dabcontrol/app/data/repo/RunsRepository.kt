package com.dabcontrol.app.data.repo

import com.dabcontrol.app.data.api.ActionHistoryResponseDto
import com.dabcontrol.app.data.api.ApiClientFactory
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.api.FriendlyRunExplanationResponseDto
import com.dabcontrol.app.data.api.NarrationResponseDto
import com.dabcontrol.app.data.api.RunStatusResponseDto
import com.dabcontrol.app.data.api.RunSummaryItemDto
import com.dabcontrol.app.data.api.TranscriptResponseDto
import com.dabcontrol.app.data.api.safeApiCall
import com.dabcontrol.app.data.preferences.ApiSettingsStore
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.first

@Singleton
class RunsRepository @Inject constructor(
    private val apiSettingsStore: ApiSettingsStore,
    private val apiClientFactory: ApiClientFactory
) {
    suspend fun fetchRuns(): ApiResult<List<RunSummaryItemDto>> {
        val service = apiClientFactory.create(apiSettingsStore.apiBaseUrl.first())
        return safeApiCall { service.runs() }
    }

    suspend fun fetchRunStatus(runId: String): ApiResult<RunStatusResponseDto> {
        val service = apiClientFactory.create(apiSettingsStore.apiBaseUrl.first())
        return safeApiCall { service.runStatus(runId) }
    }

    suspend fun fetchRunHistory(runId: String): ApiResult<ActionHistoryResponseDto> {
        val service = apiClientFactory.create(apiSettingsStore.apiBaseUrl.first())
        return safeApiCall { service.runHistory(runId) }
    }

    suspend fun fetchAiTranscript(runId: String): ApiResult<TranscriptResponseDto> {
        val service = apiClientFactory.create(apiSettingsStore.apiBaseUrl.first())
        return safeApiCall { service.runAiTranscript(runId) }
    }

    suspend fun fetchDabTranscript(runId: String): ApiResult<TranscriptResponseDto> {
        val service = apiClientFactory.create(apiSettingsStore.apiBaseUrl.first())
        return safeApiCall { service.runDabTranscript(runId) }
    }

    suspend fun fetchExplain(runId: String): ApiResult<FriendlyRunExplanationResponseDto> {
        val service = apiClientFactory.create(apiSettingsStore.apiBaseUrl.first())
        return safeApiCall { service.runExplain(runId) }
    }

    suspend fun fetchNarration(runId: String): ApiResult<NarrationResponseDto> {
        val service = apiClientFactory.create(apiSettingsStore.apiBaseUrl.first())
        return safeApiCall { service.runNarration(runId) }
    }
}
