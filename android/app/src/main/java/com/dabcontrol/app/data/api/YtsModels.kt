package com.dabcontrol.app.data.api

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject

@Serializable
data class YtsPromptResponseRequest(
    val response: String
)

@Serializable
data class YtsLiveCommandSummaryDto(
    val command_id: String,
    val status: String,
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
    val command_id: String,
    val status: String,
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
