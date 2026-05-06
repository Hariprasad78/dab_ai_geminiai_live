package com.dabcontrol.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.NavigationDrawerItemDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.safeDrawing
import androidx.navigation.NavDestination.Companion.hierarchy
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.dabcontrol.app.ui.controls.ControlsScreen
import com.dabcontrol.app.ui.common.PremiumPanel
import com.dabcontrol.app.ui.dashboard.DashboardScreen
import com.dabcontrol.app.ui.deviceinfo.DeviceInfoScreen
import com.dabcontrol.app.ui.navigation.NavRoutes
import com.dabcontrol.app.ui.runs.RunDetailScreen
import com.dabcontrol.app.ui.runs.RunsListScreen
import com.dabcontrol.app.ui.yts.YtsArtifactScreen
import com.dabcontrol.app.ui.yts.YtsDetailScreen
import com.dabcontrol.app.ui.yts.YtsListScreen
import com.dabcontrol.app.ui.yts.YtsReportScreen
import com.dabcontrol.app.ui.yts.YtsResultsScreen
import kotlinx.coroutines.launch

private data class PrimaryDestination(
    val route: String,
    val label: String,
    val title: String,
    val subtitle: String,
    val icon: @Composable () -> Unit
)

private val primaryDestinations = listOf(
    PrimaryDestination(
        route = NavRoutes.DASHBOARD,
        label = "Dashboard",
        title = "System Dashboard",
        subtitle = "Backend URL, health, CPU, RAM, load, and thermal overview",
        icon = { Text("D") }
    ),
    PrimaryDestination(
        route = NavRoutes.DEVICE_INFO,
        label = "Device Info",
        title = "Device Info",
        subtitle = "Structured device info, supported operations, and current settings",
        icon = { Text("I") }
    ),
    PrimaryDestination(
        route = NavRoutes.LIVE_CONTROL,
        label = "Live",
        title = "Live Control",
        subtitle = "HDMI preview, D-pad, remote keys, and direct control",
        icon = { Text("L") }
    ),
    PrimaryDestination(
        route = NavRoutes.AI_RUNNER,
        label = "AI Runner",
        title = "AI Task Runner",
        subtitle = "Automation runs, execution history, and detailed run state",
        icon = { Text("A") }
    ),
    PrimaryDestination(
        route = NavRoutes.YTS_RUNNER,
        label = "YTS",
        title = "YTS",
        subtitle = "Live YouTube certification sessions, prompts, and command state",
        icon = { Text("Y") }
    ),
    PrimaryDestination(
        route = NavRoutes.YTS_RESULTS,
        label = "Results",
        title = "YTS Results",
        subtitle = "Generated reports, result sessions, and result-focused review",
        icon = { Text("R") }
    )
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DabControlRoot() {
    val navController = rememberNavController()
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val scope = rememberCoroutineScope()
    val navBackStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = navBackStackEntry?.destination?.route
    val currentPrimary = primaryDestinations.firstOrNull { destination ->
        navBackStackEntry?.destination?.hierarchy?.any { it.route == destination.route } == true
    }

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet(
                drawerContainerColor = MaterialTheme.colorScheme.background,
                drawerShape = MaterialTheme.shapes.extraLarge
            ) {
                Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 14.dp)) {
                    PremiumPanel(
                        accentColor = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.padding(bottom = 12.dp)
                    ) {
                        Text(
                            text = "Navigation",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.primary
                        )
                        Text(
                            text = currentPrimary?.title ?: "DAB Control",
                            style = MaterialTheme.typography.headlineSmall
                        )
                        Text(
                            text = currentPrimary?.subtitle ?: "Live operations, device visibility, and automation control.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.6f))
                    primaryDestinations.forEach { destination ->
                        val selected = navBackStackEntry?.destination?.hierarchy?.any { it.route == destination.route } == true
                        NavigationDrawerItem(
                            label = {
                                Text(
                                    destination.label,
                                    fontWeight = if (selected) FontWeight.SemiBold else FontWeight.Medium
                                )
                            },
                            selected = selected,
                            icon = {
                                Surface(
                                    modifier = Modifier
                                        .size(34.dp)
                                        .clip(MaterialTheme.shapes.small),
                                    color = if (selected) {
                                        MaterialTheme.colorScheme.primary.copy(alpha = 0.16f)
                                    } else {
                                        MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f)
                                    }
                                ) {
                                    Box {
                                        Text(
                                            text = destination.label.first().toString(),
                                            modifier = Modifier.padding(8.dp),
                                            color = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                                            style = MaterialTheme.typography.labelLarge
                                        )
                                    }
                                }
                            },
                            onClick = {
                                navController.navigatePrimary(destination.route)
                                scope.launch { drawerState.close() }
                            },
                            colors = NavigationDrawerItemDefaults.colors(
                                selectedContainerColor = MaterialTheme.colorScheme.surface,
                                unselectedContainerColor = Color.Transparent
                            ),
                            modifier = Modifier.padding(NavigationDrawerItemDefaults.ItemPadding)
                        )
                    }
                }
            }
        }
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            NavHost(
                navController = navController,
                startDestination = NavRoutes.DASHBOARD,
                modifier = Modifier
                    .fillMaxSize()
                    .windowInsetsPadding(WindowInsets.safeDrawing)
                    .padding(top = 52.dp)
            ) {
                composable(NavRoutes.DASHBOARD) {
                    DashboardScreen(modifier = Modifier.fillMaxSize())
                }
                composable(NavRoutes.LIVE_CONTROL) {
                    ControlsScreen(
                        modifier = Modifier.fillMaxSize(),
                        onOpenDeviceInfo = { navController.navigatePrimary(NavRoutes.DEVICE_INFO) }
                    )
                }
                composable(NavRoutes.DEVICE_INFO) {
                    DeviceInfoScreen(modifier = Modifier.fillMaxSize())
                }
                composable(NavRoutes.AI_RUNNER) {
                    RunsListScreen(
                        modifier = Modifier.fillMaxSize(),
                        onOpenRun = { runId -> navController.navigate(NavRoutes.runDetail(runId)) }
                    )
                }
                composable(NavRoutes.RUN_DETAIL) {
                    RunDetailScreen(modifier = Modifier.fillMaxSize())
                }
                composable(NavRoutes.YTS_RUNNER) {
                    YtsListScreen(
                        modifier = Modifier.fillMaxSize(),
                        onOpenCommand = { commandId -> navController.navigate(NavRoutes.ytsDetail(commandId)) },
                        onOpenReport = { commandId -> navController.navigate(NavRoutes.ytsReport(commandId)) },
                        onOpenArtifact = { commandId, artifactType ->
                            navController.navigate(NavRoutes.ytsArtifact(commandId, artifactType))
                        }
                    )
                }
                composable(NavRoutes.YTS_RESULTS) {
                    YtsResultsScreen(
                        modifier = Modifier.fillMaxSize(),
                        onOpenCommand = { commandId -> navController.navigate(NavRoutes.ytsDetail(commandId)) },
                        onOpenReport = { commandId -> navController.navigate(NavRoutes.ytsReport(commandId)) },
                        onOpenArtifact = { commandId, artifactType ->
                            navController.navigate(NavRoutes.ytsArtifact(commandId, artifactType))
                        }
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
                composable(NavRoutes.YTS_ARTIFACT) {
                    YtsArtifactScreen(modifier = Modifier.fillMaxSize())
                }
            }
            Surface(
                modifier = Modifier
                    .statusBarsPadding()
                    .padding(start = 14.dp, top = 18.dp),
                shape = MaterialTheme.shapes.large,
                color = Color.Transparent,
                tonalElevation = 0.dp,
                shadowElevation = 6.dp
            ) {
                Box(
                    modifier = Modifier
                        .clip(MaterialTheme.shapes.large)
                        .background(
                            Brush.verticalGradient(
                                listOf(
                                    MaterialTheme.colorScheme.surface.copy(alpha = 0.94f),
                                    MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.78f)
                                )
                            )
                        )
                ) {
                    if (currentPrimary != null) {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Text("≡", style = MaterialTheme.typography.titleLarge)
                        }
                    } else if (navController.previousBackStackEntry != null) {
                        IconButton(onClick = { navController.popBackStack() }) {
                            Text("‹", style = MaterialTheme.typography.titleLarge)
                        }
                    }
                }
            }
        }
    }
}

private fun NavHostController.navigatePrimary(route: String) {
    navigate(route) {
        launchSingleTop = true
        restoreState = true
        popUpTo(graph.startDestinationId) {
            saveState = true
        }
    }
}
