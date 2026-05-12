package com.dabcontrol.app.ui.navigation

object NavRoutes {
    const val DASHBOARD = "dashboard"
    const val LIVE_CONTROL = "live_control"
    const val DEVICE_INFO = "device_info"
    const val AI_RUNNER = "ai_runner"
    const val YTS_RUNNER = "yts_runner"
    const val YTS_RESULTS = "yts_results"
    const val RUN_DETAIL = "run_detail/{runId}"
    const val YTS_DETAIL = "yts_detail/{commandId}"
    const val YTS_REPORT = "yts_report/{commandId}"
    const val YTS_ARTIFACT = "yts_artifact/{commandId}/{artifactType}"

    const val CONTROLS = LIVE_CONTROL
    const val DEVICE = DEVICE_INFO
    const val RUNS = AI_RUNNER
    const val YTS = YTS_RUNNER

    fun runDetail(runId: String): String = "run_detail/$runId"
    fun ytsDetail(commandId: String): String = "yts_detail/$commandId"
    fun ytsReport(commandId: String): String = "yts_report/$commandId"
    fun ytsArtifact(commandId: String, artifactType: String): String = "yts_artifact/$commandId/$artifactType"
}
