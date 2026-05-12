package com.dabcontrol.app.ui.controls

data class ControlsInfoRow(
    val label: String,
    val value: String
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
    val selectedDeviceId: String = "",
    val isStreaming: Boolean = false,
    val streamFrameBytes: ByteArray? = null,
    val streamStatus: String = "Stream stopped.",
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
    val irStatusPreview: String = "--",
    val irDevicesPreview: String = "--",
    val irKeysPreview: String = "--",
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
