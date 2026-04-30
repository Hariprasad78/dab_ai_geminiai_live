package com.dabcontrol.app.data.api

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

@Serializable
data class DabDevicesResponseDto(
    val success: Boolean = false,
    val devices: List<JsonObject> = emptyList(),
    val selected_device_id: String? = null,
    val warning: String? = null
)

@Serializable
data class ManualActionRequestDto(
    val action: String,
    val params: JsonObject? = null,
    val device_id: String? = null,
    val control_mode: String? = null,
    val ir_device_id: String? = null
)

@Serializable
data class ManualActionResponseDto(
    val success: Boolean = false,
    val action: String,
    val result: JsonObject? = null,
    val error: String? = null
)

@Serializable
data class ManualActionBatchRequestDto(
    val actions: List<ManualActionRequestDto>,
    val continue_on_error: Boolean = true
)

@Serializable
data class ManualActionBatchResponseDto(
    val success: Boolean = false,
    val total: Int = 0,
    val results: List<ManualActionResponseDto> = emptyList()
)

@Serializable
data class IrSendRequestDto(
    val device_id: String,
    val key_name: String
)

@Serializable
data class IrTrainRequestDto(
    val device_id: String,
    val key_name: String,
    val timeout_ms: Int = 6000
)

@Serializable
data class TaskMacroRequestDto(
    val instruction: String,
    val execute: Boolean = false,
    val continue_on_error: Boolean = true
)

@Serializable
data class PlannerDebugRequestDto(
    val goal: String,
    val device_id: String? = null,
    val ocr_text: String? = null,
    val current_app: String? = null,
    val current_screen: String? = null
)
