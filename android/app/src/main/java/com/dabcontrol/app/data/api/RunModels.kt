package com.dabcontrol.app.data.api

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

@Serializable
data class RunSummaryItemDto(
    val run_id: String,
    val goal: String,
    val status: String,
    val step_count: Int,
    val started_at: String? = null
)

@Serializable
data class RunStatusResponseDto(
    val run_id: String,
    val status: String,
    val goal: String,
    val step_count: Int,
    val retries: Int = 0,
    val current_app: String? = null,
    val current_screen: String? = null,
    val ai_log_count: Int = 0,
    val dab_log_count: Int = 0
)

@Serializable
data class ActionRecordItemDto(
    val step: Int,
    val action: String,
    val reason: String,
    val result: String,
    val timestamp: String? = null
)

@Serializable
data class ActionHistoryResponseDto(
    val run_id: String,
    val goal: String,
    val action_count: Int = 0,
    val actions: List<ActionRecordItemDto> = emptyList()
)

@Serializable
data class TranscriptResponseDto(
    val run_id: String,
    val goal: String,
    val count: Int = 0,
    val events: List<JsonObject> = emptyList()
)

@Serializable
data class FriendlyStepItemDto(
    val step: Int,
    val title: String,
    val simple_action: String,
    val what_happened: String,
    val result: String,
    val simple_status: String
)

@Serializable
data class FinalDiagnosisDto(
    val final_summary: String,
    val root_cause: String,
    val user_friendly_reason: String
)

@Serializable
data class FriendlyRunExplanationResponseDto(
    val run_id: String,
    val goal: String,
    val status: String,
    val timeline: List<FriendlyStepItemDto> = emptyList(),
    val diagnosis: FinalDiagnosisDto? = null
)

@Serializable
data class NarrationEventItemDto(
    val idx: Int,
    val step: Int,
    val tts_text: String,
    val tts_category: String
)

@Serializable
data class NarrationResponseDto(
    val run_id: String,
    val goal: String,
    val count: Int = 0,
    val events: List<NarrationEventItemDto> = emptyList()
)
