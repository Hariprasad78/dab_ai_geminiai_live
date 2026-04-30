package com.dabcontrol.app.data.api

import kotlinx.serialization.json.JsonObject
import retrofit2.Response
import retrofit2.http.GET
import retrofit2.http.POST
import retrofit2.http.Path
import retrofit2.http.Body
import retrofit2.http.Query

interface DabApiService {
    @GET("health")
    suspend fun health(): Response<HealthResponse>

    @GET("system/metrics")
    suspend fun systemMetrics(): Response<JsonObject>

    @GET("runs")
    suspend fun runs(): Response<List<RunSummaryItemDto>>

    @GET("run/{runId}/status")
    suspend fun runStatus(@Path("runId") runId: String): Response<RunStatusResponseDto>

    @GET("run/{runId}/history")
    suspend fun runHistory(@Path("runId") runId: String): Response<ActionHistoryResponseDto>

    @GET("run/{runId}/ai-transcript")
    suspend fun runAiTranscript(@Path("runId") runId: String): Response<TranscriptResponseDto>

    @GET("run/{runId}/dab-transcript")
    suspend fun runDabTranscript(@Path("runId") runId: String): Response<TranscriptResponseDto>

    @GET("run/{runId}/explain")
    suspend fun runExplain(@Path("runId") runId: String): Response<FriendlyRunExplanationResponseDto>

    @GET("run/{runId}/narration")
    suspend fun runNarration(@Path("runId") runId: String): Response<NarrationResponseDto>

    @GET("yts/command/live")
    suspend fun ytsLiveCommands(
        @Query("limit") limit: Int = 100,
        @Query("active_only") activeOnly: Boolean = false
    ): Response<List<YtsLiveCommandSummaryDto>>

    @GET("yts/command/live/{commandId}")
    suspend fun ytsLiveCommandState(@Path("commandId") commandId: String): Response<YtsLiveCommandStateDto>

    @POST("yts/command/live/{commandId}/stop")
    suspend fun stopYtsLiveCommand(@Path("commandId") commandId: String): Response<JsonObject>

    @POST("yts/command/live/{commandId}/respond")
    suspend fun respondYtsLiveCommand(
        @Path("commandId") commandId: String,
        @Body request: YtsPromptResponseRequest
    ): Response<JsonObject>

    @GET("dab/devices")
    suspend fun dabDevices(): Response<DabDevicesResponseDto>

    @GET("dab/device-info")
    suspend fun dabDeviceInfo(@Query("device_id") deviceId: String? = null): Response<JsonObject>

    @GET("dab/device-capability-status")
    suspend fun dabDeviceCapabilityStatus(
        @Query("device_id") deviceId: String? = null,
        @Query("refresh") refresh: Boolean = false
    ): Response<JsonObject>

    @GET("dab/device-operations-grid")
    suspend fun dabDeviceOperationsGrid(
        @Query("device_id") deviceId: String? = null,
        @Query("refresh") refresh: Boolean = false
    ): Response<JsonObject>

    @GET("dab/device-current-settings")
    suspend fun dabDeviceCurrentSettings(
        @Query("device_id") deviceId: String? = null,
        @Query("refresh") refresh: Boolean = false
    ): Response<JsonObject>

    @POST("action")
    suspend fun manualAction(@Body request: ManualActionRequestDto): Response<ManualActionResponseDto>

    @POST("actions/batch")
    suspend fun manualActionsBatch(@Body request: ManualActionBatchRequestDto): Response<ManualActionBatchResponseDto>

    @GET("ir/status")
    suspend fun irStatus(): Response<JsonObject>

    @GET("ir/devices")
    suspend fun irDevices(): Response<JsonObject>

    @GET("ir/device/{deviceId}/keys")
    suspend fun irDeviceKeys(@Path("deviceId") deviceId: String): Response<JsonObject>

    @POST("ir/send")
    suspend fun irSend(@Body request: IrSendRequestDto): Response<JsonObject>

    @POST("ir/train")
    suspend fun irTrain(@Body request: IrTrainRequestDto): Response<JsonObject>

    @POST("task/macro")
    suspend fun taskMacro(@Body request: TaskMacroRequestDto): Response<JsonObject>

    @POST("planner/debug")
    suspend fun plannerDebug(@Body request: PlannerDebugRequestDto): Response<JsonObject>
}
