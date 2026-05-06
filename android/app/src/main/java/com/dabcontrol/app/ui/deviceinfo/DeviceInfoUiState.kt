package com.dabcontrol.app.ui.deviceinfo

data class DeviceInfoRow(
    val field: String,
    val value: String,
    val isStructured: Boolean = false
)

data class DeviceInfoUiState(
    val isLoading: Boolean = false,
    val selectedDeviceId: String = "",
    val deviceIds: List<String> = emptyList(),
    val rows: List<DeviceInfoRow> = emptyList(),
    val error: String? = null
)
