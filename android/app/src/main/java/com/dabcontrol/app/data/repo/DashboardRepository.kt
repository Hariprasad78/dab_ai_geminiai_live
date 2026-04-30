package com.dabcontrol.app.data.repo

import com.dabcontrol.app.data.api.ApiClientFactory
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.api.HealthResponse
import com.dabcontrol.app.data.api.safeApiCall
import com.dabcontrol.app.data.preferences.ApiSettingsStore
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.first
import kotlinx.serialization.json.JsonObject

@Singleton
class DashboardRepository @Inject constructor(
    private val apiSettingsStore: ApiSettingsStore,
    private val apiClientFactory: ApiClientFactory
) {
    suspend fun fetchHealth(): ApiResult<HealthResponse> {
        val baseUrl = apiSettingsStore.apiBaseUrl.first()
        return safeApiCall { apiClientFactory.create(baseUrl).health() }
    }

    suspend fun fetchMetrics(): ApiResult<JsonObject> {
        val baseUrl = apiSettingsStore.apiBaseUrl.first()
        return safeApiCall { apiClientFactory.create(baseUrl).systemMetrics() }
    }
}
