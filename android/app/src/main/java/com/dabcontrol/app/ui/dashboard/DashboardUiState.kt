package com.dabcontrol.app.ui.dashboard

data class DashboardUiState(
    val isLoading: Boolean = false,
    val healthStatus: String = "--",
    val mode: String = "--",
    val metricsPreview: String = "--",
    val error: String? = null,
    val apiBaseUrl: String = ""
)
