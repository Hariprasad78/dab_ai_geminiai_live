package com.dabcontrol.app.ui.dashboard

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.preferences.ApiSettingsStore
import com.dabcontrol.app.data.repo.DashboardRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.async
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.serialization.json.JsonObject

@HiltViewModel
class DashboardViewModel @Inject constructor(
    private val dashboardRepository: DashboardRepository,
    private val apiSettingsStore: ApiSettingsStore
) : ViewModel() {
    private val _uiState = MutableStateFlow(DashboardUiState())
    val uiState: StateFlow<DashboardUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            val url = apiSettingsStore.apiBaseUrl.first()
            _uiState.value = _uiState.value.copy(apiBaseUrl = url)
            refresh()
        }
    }

    fun onApiBaseUrlChanged(value: String) {
        _uiState.value = _uiState.value.copy(apiBaseUrl = value)
    }

    fun saveApiBaseUrl() {
        viewModelScope.launch {
            apiSettingsStore.saveApiBaseUrl(_uiState.value.apiBaseUrl)
            refresh()
        }
    }

    fun refresh() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            val healthDeferred = async { dashboardRepository.fetchHealth() }
            val metricsDeferred = async { dashboardRepository.fetchMetrics() }

            val healthResult = healthDeferred.await()
            val metricsResult = metricsDeferred.await()

            val healthStatus = when (healthResult) {
                is ApiResult.Success -> healthResult.data.status
                is ApiResult.HttpError -> "HTTP ${healthResult.code}"
                is ApiResult.NetworkError -> "Network error"
                is ApiResult.UnknownError -> "Unknown error"
            }

            val mode = if (healthResult is ApiResult.Success) healthResult.data.mode ?: "--" else "--"
            val metricsPreview = toMetricsPreview(metricsResult)
            val error = buildError(healthResult, metricsResult)

            _uiState.value = _uiState.value.copy(
                isLoading = false,
                healthStatus = healthStatus,
                mode = mode,
                metricsPreview = metricsPreview,
                error = error
            )
        }
    }

    private fun toMetricsPreview(result: ApiResult<JsonObject>): String {
        return when (result) {
            is ApiResult.Success -> {
                val keys = result.data.keys.take(6)
                if (keys.isEmpty()) "No metrics keys" else keys.joinToString(", ")
            }
            is ApiResult.HttpError -> "HTTP ${result.code}"
            is ApiResult.NetworkError -> "Network error"
            is ApiResult.UnknownError -> "Unknown error"
        }
    }

    private fun buildError(health: ApiResult<*>, metrics: ApiResult<*>): String? {
        val issues = mutableListOf<String>()
        if (health !is ApiResult.Success) issues.add("Health failed")
        if (metrics !is ApiResult.Success) issues.add("Metrics failed")
        return if (issues.isEmpty()) null else issues.joinToString(" · ")
    }
}
