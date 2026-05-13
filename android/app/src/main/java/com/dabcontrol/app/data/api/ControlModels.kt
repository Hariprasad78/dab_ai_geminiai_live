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
data class DeviceContextValidationDto(
    val valid: Boolean = false,
    val dabReady: Boolean = false,
    val ytsReady: Boolean = false,
    val ytsShortId: String = "",
    val irReady: Boolean = false,
    val cameraReady: Boolean = false,
    val videoSource: String? = null,
    val issues: List<String> = emptyList()
)

@Serializable
data class DeviceContextDto(
    val contextId: String = "",
    val displayName: String = "",
    val dabDeviceId: String = "",
    val ytsDeviceId: String = "",
    val adbDeviceId: String = "",
    val irDeviceId: String = "",
    val cameraId: String = "",
    val videoSource: String = "",
    val cameraPath: String = "",
    val audioDevice: String = "",
    val active: Boolean = false,
    val readiness: DeviceContextValidationDto? = null
)

@Serializable
data class DeviceContextsResponseDto(
    val success: Boolean = false,
    val activeContextId: String = "",
    val contexts: List<DeviceContextDto> = emptyList()
)

@Serializable
data class CurrentDeviceContextResponseDto(
    val selected_device_id: String? = null,
    val configured_device_id: String? = null,
    val context: DeviceContextDto? = null,
    val validation: DeviceContextValidationDto? = null,
    val warning: String? = null
)

@Serializable
data class DeviceContextSelectRequestDto(
    val device_id: String? = null,
    val contextId: String? = null,
    val persist: Boolean = true
)

@Serializable
data class AudioSourceResponseDto(
    val enabled: Boolean = false,
    val follow_active_video: Boolean = false,
    val ffmpeg_available: Boolean = false,
    val arecord_available: Boolean = false,
    val user_in_audio_group: Boolean? = null,
    val input_format: String? = null,
    val device: String? = null,
    val guessed_device: String? = null,
    val active_video_device: String? = null,
    val strict_match_blocked: Boolean = false,
    val selected_video_device: String? = null,
    val sample_rate: Int? = null,
    val channels: Int? = null,
    val bitrate: String? = null
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
data class IrDevicesResponseDto(
    val brand: String? = null,
    val active_device_id: String? = null,
    val devices: List<String> = emptyList()
)

@Serializable
data class IrDeviceKeysResponseDto(
    val device_id: String = "",
    val keys: List<String> = emptyList()
)

@Serializable
data class TaskMacroRequestDto(
    val instruction: String,
    val execute: Boolean = false,
    val continue_on_error: Boolean = true,
    val control_mode: String? = null,
    val ir_device_id: String? = null,
    val device_id: String? = null,
    val max_steps: Int = 8
)

@Serializable
data class PlannerDebugRequestDto(
    val goal: String,
    val device_id: String? = null,
    val ocr_text: String? = null,
    val current_app: String? = null,
    val current_screen: String? = null
)
