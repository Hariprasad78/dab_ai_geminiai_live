package com.dabcontrol.app.ui.yts

import com.dabcontrol.app.data.api.YtsLiveCommandStartResponseDto
import com.dabcontrol.app.data.api.YtsLiveCommandStateDto
import com.dabcontrol.app.data.api.YtsLiveCommandSummaryDto
import com.dabcontrol.app.data.api.YtsTestCatalogItemDto

enum class YtsWorkspaceTab {
    CREATE_JOB,
    RUNNING_SESSION,
    PAST_RESULTS
}

data class YtsListUiState(
    val isLoading: Boolean = false,
    val isCatalogLoading: Boolean = false,
    val isStarting: Boolean = false,
    val activeTab: YtsWorkspaceTab = YtsWorkspaceTab.CREATE_JOB,
    val apiBaseUrl: String = "",
    val items: List<YtsLiveCommandSummaryDto> = emptyList(),
    val activeCommandId: String? = null,
    val commandTestCountHints: Map<String, Int> = emptyMap(),
    val activeCommand: YtsLiveCommandStateDto? = null,
    val promptInput: String = "",
    val sessionStatus: String = "No active YTS session.",
    val remoteStatus: String = "Remote idle.",
    val isStreaming: Boolean = false,
    val streamFrameBytes: ByteArray? = null,
    val streamStatus: String = "Stream idle.",
    val catalog: List<YtsTestCatalogItemDto> = emptyList(),
    val filteredCatalog: List<YtsTestCatalogItemDto> = emptyList(),
    val selectedTestIds: List<String> = emptyList(),
    val suiteFilter: String = "",
    val categoryFilter: String = "",
    val searchQuery: String = "",
    val deviceId: String = "",
    val deviceDisplayName: String = "",
    val ytsDeviceId: String = "",
    val ytsShortId: String = "",
    val irDeviceId: String = "",
    val sharedDeviceReady: Boolean = false,
    val sharedDeviceIssues: List<String> = emptyList(),
    val jsonOutputFile: String = "/tmp/yts-results.json",
    val guidedMode: Boolean = false,
    val interactiveAi: Boolean = true,
    val recordVideo: Boolean = false,
    val recordAudio: Boolean = true,
    val filterTokensInput: String = "",
    val extraArgsInput: String = "",
    val plannerModel: String = "",
    val liveModel: String = "",
    val availableModels: List<String> = emptyList(),
    val modelStatus: String = "--",
    val startStatus: String = "Configure a YTS run.",
    val lastStartedCommand: YtsLiveCommandStartResponseDto? = null,
    val error: String? = null
)

data class YtsDetailUiState(
    val isLoading: Boolean = false,
    val data: YtsLiveCommandStateDto? = null,
    val promptInput: String = "",
    val error: String? = null
)
