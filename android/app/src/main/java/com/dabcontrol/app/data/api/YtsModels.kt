package com.dabcontrol.app.data.api

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject

@Serializable
data class YtsPromptResponseRequest(
    val response: String = ""
)

@Serializable
data class YtsPromptSuggestRequest(
    val send_response: Boolean = false
)

@Serializable
data class YtsTestCatalogItemDto(
    val test_id: String = "",
    val test_title: String? = null,
    val test_suite: String? = null,
    val test_category: String? = null
)

@Serializable
data class RuntimeModelResponseDto(
    val success: Boolean = false,
    val active_vertex_planner_model: String = "",
    val configured_vertex_planner_model: String = "",
    val active_vertex_live_model: String = "",
    val configured_vertex_live_model: String = "",
    val available_models: List<String> = emptyList(),
    val message: String = ""
)

@Serializable
data class RuntimeModelUpdateRequestDto(
    val model: String = "",
    val target: String = "planner"
)

@Serializable
data class YtsLiveCommandRequestDto(
    val command: String = "",
    val params: List<String> = emptyList(),
    val global_options: JsonObject = JsonObject(emptyMap()),
    val output_file: String? = null,
    val interactive_ai: Boolean = false,
    val record_video: Boolean = false,
    val record_audio: Boolean = true,
    val device_id: String? = null
)

@Serializable
data class YtsLiveCommandStartResponseDto(
    val command_id: String = "",
    val status: String = "",
    val device_id: String? = null
)

@Serializable
data class YtsLiveCommandSummaryDto(
    val command_id: String = "",
    val status: String = "",
    val command: String? = null,
    val updated_at: String? = null,
    val result_file_name: String? = null,
    val report_pdf_name: String? = null,
    val report_html_name: String? = null,
    val record_video: Boolean = false,
    val video_recording_status: String? = null,
    val video_file_name: String? = null
)

@Serializable
data class YtsLiveCommandStateDto(
    val command_id: String = "",
    val status: String = "",
    val command: String? = null,
    val updated_at: String? = null,
    val returncode: Int? = null,
    val stdout: String? = null,
    val stderr: String? = null,
    val awaiting_input: Boolean = false,
    val pending_prompt: JsonObject? = null,
    val logs: JsonArray? = null,
    val result_file_name: String? = null,
    val report_pdf_name: String? = null,
    val report_html_name: String? = null,
    val record_video: Boolean = false,
    val video_recording_status: String? = null,
    val video_file_name: String? = null
)

@Serializable
data class YtsResultArtifactItemDto(
    val command_id: String = "",
    val command: String? = null,
    val status: String? = null,
    val updated_at: String? = null,
    val result_summary: JsonElement? = null,
    val ref: String = "",
    val type: String = "",
    val label: String = "",
    val available: Boolean = false
)

@Serializable
data class YtsResultsAnalysisRequestDto(
    val artifact_refs: List<String> = emptyList(),
    val include_zip_base64: Boolean = true,
    val analysis_model: String? = null,
    val triage_level: String = "deep"
)

@Serializable
data class YtsResultsAnalysisResponseDto(
    val report_id: String = "",
    val txt_name: String = "",
    val pdf_name: String = "",
    val artifact_refs: List<String> = emptyList(),
    val created_at: String = "",
    val summary: String = "",
    val analysis_model: String = "",
    val triage_level: String = "",
    val total_tests: Int = 0,
    val failed_tests: Int = 0,
    val failed_reasons: List<String> = emptyList()
)
