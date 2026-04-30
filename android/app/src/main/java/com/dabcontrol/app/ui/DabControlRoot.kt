package com.dabcontrol.app.ui

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.dabcontrol.app.ui.navigation.NavRoutes
import com.dabcontrol.app.ui.controls.ControlsScreen
import com.dabcontrol.app.ui.dashboard.DashboardScreen
import com.dabcontrol.app.ui.runs.RunDetailScreen
import com.dabcontrol.app.ui.runs.RunsListScreen
import com.dabcontrol.app.ui.yts.YtsDetailScreen
import com.dabcontrol.app.ui.yts.YtsListScreen
import com.dabcontrol.app.ui.yts.YtsReportScreen

@Composable
fun DabControlRoot() {
    val navController = rememberNavController()
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("DAB Control") },
                actions = { TopNavActions(navController) }
            )
        }
    ) { inner ->
        NavHost(
            navController = navController,
            startDestination = NavRoutes.DASHBOARD,
            modifier = Modifier.fillMaxSize().padding(inner)
        ) {
            composable(NavRoutes.DASHBOARD) {
                DashboardScreen(modifier = Modifier.fillMaxSize())
            }
            composable(NavRoutes.CONTROLS) {
                ControlsScreen(modifier = Modifier.fillMaxSize())
            }
            composable(NavRoutes.RUNS) {
                RunsListScreen(
                    modifier = Modifier.fillMaxSize(),
                    onOpenRun = { runId -> navController.navigate(NavRoutes.runDetail(runId)) }
                )
            }
            composable(NavRoutes.RUN_DETAIL) {
                RunDetailScreen(modifier = Modifier.fillMaxSize())
            }
            composable(NavRoutes.YTS) {
                YtsListScreen(
                    modifier = Modifier.fillMaxSize(),
                    onOpenCommand = { commandId -> navController.navigate(NavRoutes.ytsDetail(commandId)) }
                )
            }
            composable(NavRoutes.YTS_DETAIL) {
                YtsDetailScreen(
                    modifier = Modifier.fillMaxSize(),
                    onOpenReport = { commandId -> navController.navigate(NavRoutes.ytsReport(commandId)) }
                )
            }
            composable(NavRoutes.YTS_REPORT) {
                YtsReportScreen(modifier = Modifier.fillMaxSize())
            }
        }
    }
}

@Composable
private fun TopNavActions(navController: NavHostController) {
    Row(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Button(onClick = { navController.navigate(NavRoutes.DASHBOARD) }) { Text("Dashboard") }
        Button(onClick = { navController.navigate(NavRoutes.CONTROLS) }) { Text("Controls") }
        Button(onClick = { navController.navigate(NavRoutes.RUNS) }) { Text("Runs") }
        Button(onClick = { navController.navigate(NavRoutes.YTS) }) { Text("YTS") }
    }
}
