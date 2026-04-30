package com.dabcontrol.app.ui.dashboard

data class MetricPoint(
    val label: String,
    val value: Float
)

data class DashboardUiState(
    val isLoading: Boolean = false,
    val healthStatus: String = "--",
    val mode: String = "--",
    val apiBaseUrl: String = "",
    val deviceIds: List<String> = emptyList(),
    val selectedDeviceId: String = "",
    val plannerModel: String = "",
    val liveModel: String = "",
    val availableModels: List<String> = emptyList(),
    val modelStatus: String = "--",
    val cpuPercent: Float? = null,
    val ramPercent: Float? = null,
    val load1m: Float? = null,
    val cpuTempC: Float? = null,
    val cpuCount: Int? = null,
    val timestamp: String = "--",
    val cpuHistory: List<MetricPoint> = emptyList(),
    val ramHistory: List<MetricPoint> = emptyList(),
    val loadHistory: List<MetricPoint> = emptyList(),
    val tempHistory: List<MetricPoint> = emptyList(),
    val metricsPreview: String = "--",
    val error: String? = null,
)
