package com.dabcontrol.app.ui.runs

import com.dabcontrol.app.data.api.ActionRecordItemDto
import com.dabcontrol.app.data.api.FriendlyStepItemDto
import com.dabcontrol.app.data.api.NarrationEventItemDto
import com.dabcontrol.app.data.api.RunStatusResponseDto
import com.dabcontrol.app.data.api.RunSummaryItemDto
import kotlinx.serialization.json.JsonObject

data class RunsListUiState(
    val isLoading: Boolean = false,
    val items: List<RunSummaryItemDto> = emptyList(),
    val error: String? = null
)

data class RunDetailUiState(
    val isLoading: Boolean = false,
    val status: RunStatusResponseDto? = null,
    val actions: List<ActionRecordItemDto> = emptyList(),
    val aiEvents: List<JsonObject> = emptyList(),
    val dabEvents: List<JsonObject> = emptyList(),
    val explainTimeline: List<FriendlyStepItemDto> = emptyList(),
    val diagnosisSummary: String? = null,
    val narrationEvents: List<NarrationEventItemDto> = emptyList(),
    val error: String? = null
)
