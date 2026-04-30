package com.dabcontrol.app.ui.yts

import com.dabcontrol.app.data.api.YtsLiveCommandStateDto
import com.dabcontrol.app.data.api.YtsLiveCommandSummaryDto

data class YtsListUiState(
    val isLoading: Boolean = false,
    val items: List<YtsLiveCommandSummaryDto> = emptyList(),
    val error: String? = null
)

data class YtsDetailUiState(
    val isLoading: Boolean = false,
    val data: YtsLiveCommandStateDto? = null,
    val promptInput: String = "",
    val error: String? = null
)
