package com.dabcontrol.app.ui.controls

data class ControlsInfoRow(
    val label: String,
    val value: String
)

data class ControlsAudioSource(
    val enabled: Boolean,
    val ffmpegAvailable: Boolean,
    val device: String,
    val inputFormat: String,
    val sampleRate: String,
    val channels: String
)

enum class ControlsRemoteMode {
    DAB,
    IR
}

data class ControlsDeviceContext(
    val contextId: String,
    val displayName: String,
    val dabDeviceId: String,
    val ytsDeviceId: String,
    val ytsShortId: String,
    val irDeviceId: String,
    val videoSource: String,
    val isActive: Boolean,
    val isReady: Boolean,
    val issues: List<String>
)

data class ControlsOperationRow(
    val operation: String,
    val supported: Boolean,
    val defaultAction: String,
    val relatedCount: Int
)

data class ControlsSettingRow(
    val name: String,
    val value: String,
    val writable: Boolean,
    val status: String
)

data class ControlsUiState(
    val isLoading: Boolean = false,
    val apiBaseUrl: String = "",
    val deviceIds: List<String> = emptyList(),
    val deviceContexts: List<ControlsDeviceContext> = emptyList(),
    val selectedDeviceId: String = "",
    val selectedDeviceName: String = "",
    val selectedYtsDeviceId: String = "",
    val selectedYtsShortId: String = "",
    val selectedIrDeviceId: String = "",
    val selectedVideoSource: String = "",
    val selectedContextIssues: List<String> = emptyList(),
    val isStreaming: Boolean = false,
    val streamFrameBytes: ByteArray? = null,
    val streamStatus: String = "Stream stopped.",
    val isAudioStreaming: Boolean = false,
    val audioStatus: String = "Audio stream stopped.",
    val audioSource: ControlsAudioSource? = null,
    val remoteMode: ControlsRemoteMode = ControlsRemoteMode.DAB,
    val remoteStatus: String = "No remote actions yet.",
    val actionName: String = "PRESS_HOME",
    val actionParamsJson: String = "{}",
    val batchActionsJson: String = """[{"action":"PRESS_HOME"},{"action":"WAIT","params":{"seconds":1.0}}]""",
    val deviceInfoRows: List<ControlsInfoRow> = emptyList(),
    val capabilityRows: List<ControlsInfoRow> = emptyList(),
    val operationRows: List<ControlsOperationRow> = emptyList(),
    val settingRows: List<ControlsSettingRow> = emptyList(),
    val lastActionResult: String = "--",
    val lastBatchResult: String = "--",
    val irDeviceId: String = "samsung_tv_default",
    val irKeyName: String = "POWER",
    val irStatusRows: List<ControlsInfoRow> = emptyList(),
    val irAvailableDevices: List<String> = emptyList(),
    val irAvailableKeys: List<String> = emptyList(),
    val irLastResult: String = "--",
    val macroInstruction: String = "Go home, wait, open YouTube",
    val macroExecute: Boolean = false,
    val macroResult: String = "--",
    val plannerGoal: String = "Launch YouTube and verify home screen",
    val plannerCurrentApp: String = "",
    val plannerCurrentScreen: String = "",
    val plannerOcrText: String = "",
    val plannerResult: String = "--",
    val refreshStatus: String = "Not refreshed yet.",
    val error: String? = null
)
