package com.dabcontrol.app.data.repo

import com.dabcontrol.app.data.api.ApiClientFactory
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.api.YtsLiveCommandStateDto
import com.dabcontrol.app.data.api.YtsLiveCommandSummaryDto
import com.dabcontrol.app.data.api.YtsPromptResponseRequest
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

    suspend fun stopLiveCommand(commandId: String): ApiResult<JsonObject> {
        return safeApiCall { service().stopYtsLiveCommand(commandId) }
    }

    suspend fun respondToPrompt(commandId: String, response: String): ApiResult<JsonObject> {
        return safeApiCall { service().respondYtsLiveCommand(commandId, YtsPromptResponseRequest(response)) }
    }
}
