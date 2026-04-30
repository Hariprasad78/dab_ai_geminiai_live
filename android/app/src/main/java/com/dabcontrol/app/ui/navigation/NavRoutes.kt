package com.dabcontrol.app.ui.navigation

object NavRoutes {
    const val DASHBOARD = "dashboard"
    const val CONTROLS = "controls"
    const val RUNS = "runs"
    const val RUN_DETAIL = "run_detail/{runId}"
    const val YTS = "yts"
    const val YTS_DETAIL = "yts_detail/{commandId}"
    const val YTS_REPORT = "yts_report/{commandId}"

    fun runDetail(runId: String): String = "run_detail/$runId"
    fun ytsDetail(commandId: String): String = "yts_detail/$commandId"
    fun ytsReport(commandId: String): String = "yts_report/$commandId"
}
