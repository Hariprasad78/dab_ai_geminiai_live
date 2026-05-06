package com.dabcontrol.app.ui.controls

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.dabcontrol.app.ui.common.PremiumBackdrop
import com.dabcontrol.app.ui.common.SectionLabel

private enum class DiagnosticsTab(val label: String) {
    OVERVIEW("Overview"),
    OPERATIONS("Operations"),
    SETTINGS("Settings")
}

private enum class SettingsFilter(val label: String) {
    ALL("All settings"),
    WRITABLE("Writable only"),
    ISSUES("Issues only")
}

@Composable
fun ControlsScreen(
    modifier: Modifier = Modifier,
    onOpenDeviceInfo: () -> Unit = {},
    viewModel: ControlsViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val scrollState = rememberScrollState()
    var floatingStreamHidden by rememberSaveable { mutableStateOf(false) }
    val shouldOfferFloatingStream = state.isStreaming && scrollState.value > 420

    Box(modifier = modifier.fillMaxSize()) {
        PremiumBackdrop {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .verticalScroll(scrollState)
                    .padding(start = 16.dp, end = 16.dp, top = 20.dp, bottom = 16.dp),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                SectionLabel(
                    eyebrow = "Live Control",
                    title = "Remote and Stream Workspace",
                    subtitle = "Focused HDMI preview, remote actions, IR tools, and automation without diagnostics clutter."
                )

                HeaderCard(
                    isLoading = state.isLoading,
                    refreshStatus = state.refreshStatus,
                    error = state.error,
                    onRefresh = { viewModel.refreshAll(force = true) },
                    onOpenDeviceInfo = onOpenDeviceInfo
                )

                DeviceSelectionCard(
                    selectedDeviceId = state.selectedDeviceId,
                    deviceIds = state.deviceIds,
                    onDeviceSelected = viewModel::onDeviceSelected
                )

                LiveControlCard(
                    isStreaming = state.isStreaming,
                    streamFrameBytes = state.streamFrameBytes,
                    streamStatus = state.streamStatus,
                    remoteStatus = state.remoteStatus,
                    onToggleStream = viewModel::toggleStream,
                    onRefreshStream = viewModel::refreshStream,
                    onSendRemoteAction = viewModel::sendRemoteAction
                )

                ActionWorkbenchCard(
                    actionName = state.actionName,
                    actionParamsJson = state.actionParamsJson,
                    batchActionsJson = state.batchActionsJson,
                    lastActionResult = state.lastActionResult,
                    lastBatchResult = state.lastBatchResult,
                    onActionChanged = viewModel::onActionChanged,
                    onActionParamsChanged = viewModel::onActionParamsChanged,
                    onBatchActionsChanged = viewModel::onBatchActionsChanged,
                    onExecuteAction = viewModel::executeAction,
                    onExecuteBatch = viewModel::executeBatch
                )

                IrControlsCard(
                    irDeviceId = state.irDeviceId,
                    irKeyName = state.irKeyName,
                    irStatusPreview = state.irStatusPreview,
                    irDevicesPreview = state.irDevicesPreview,
                    irKeysPreview = state.irKeysPreview,
                    irLastResult = state.irLastResult,
                    onIrDeviceChanged = viewModel::onIrDeviceChanged,
                    onIrKeyChanged = viewModel::onIrKeyChanged,
                    onFetchIrKeys = viewModel::fetchIrKeys,
                    onIrSend = viewModel::irSend,
                    onIrTrain = viewModel::irTrain
                )

                AutomationLabCard(
                    macroInstruction = state.macroInstruction,
                    macroExecute = state.macroExecute,
                    macroResult = state.macroResult,
                    plannerGoal = state.plannerGoal,
                    plannerCurrentApp = state.plannerCurrentApp,
                    plannerCurrentScreen = state.plannerCurrentScreen,
                    plannerOcrText = state.plannerOcrText,
                    plannerResult = state.plannerResult,
                    onMacroInstructionChanged = viewModel::onMacroInstructionChanged,
                    onToggleMacroExecute = viewModel::toggleMacroExecute,
                    onRunMacro = viewModel::runMacro,
                    onPlannerGoalChanged = viewModel::onPlannerGoalChanged,
                    onPlannerAppChanged = viewModel::onPlannerAppChanged,
                    onPlannerScreenChanged = viewModel::onPlannerScreenChanged,
                    onPlannerOcrChanged = viewModel::onPlannerOcrChanged,
                    onRunPlannerDebug = viewModel::runPlannerDebug
                )
            }
        }

        if (!state.isStreaming) {
            floatingStreamHidden = false
        }

        if (shouldOfferFloatingStream && floatingStreamHidden) {
            OutlinedButton(
                onClick = { floatingStreamHidden = false },
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(12.dp)
            ) {
                Text("Reopen Stream")
            }
        } else if (shouldOfferFloatingStream) {
            FloatingStreamOverlay(
                streamFrameBytes = state.streamFrameBytes,
                streamStatus = state.streamStatus,
                onHide = { floatingStreamHidden = true },
                modifier = Modifier.align(Alignment.BottomEnd)
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class, ExperimentalMaterial3Api::class)
@Composable
fun DeviceInfoScreen(
    modifier: Modifier = Modifier,
    viewModel: ControlsViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    var selectedTab by rememberSaveable { mutableStateOf(DiagnosticsTab.OVERVIEW) }
    var settingsFilter by rememberSaveable { mutableStateOf(SettingsFilter.ALL) }
    var settingsExpanded by remember { mutableStateOf(false) }
    val filteredSettings = remember(state.settingRows, settingsFilter) {
        when (settingsFilter) {
            SettingsFilter.ALL -> state.settingRows
            SettingsFilter.WRITABLE -> state.settingRows.filter { it.writable }
            SettingsFilter.ISSUES -> state.settingRows.filter { it.status != "Ready" }
        }
    }

    PremiumBackdrop(modifier = modifier) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(start = 16.dp, end = 16.dp, top = 20.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            SectionLabel(
                eyebrow = "Device Info",
                title = "Structured Backend Device Snapshot",
                subtitle = "Live backend data arranged into grouped tables from the real device-info, operations-grid, and current-settings payloads."
            )

            HeaderCard(
                isLoading = state.isLoading,
                refreshStatus = state.refreshStatus,
                error = state.error,
                onRefresh = { viewModel.refreshAll(force = true) },
                onOpenDeviceInfo = {}
            )

            DeviceSelectionCard(
                selectedDeviceId = state.selectedDeviceId,
                deviceIds = state.deviceIds,
                onDeviceSelected = viewModel::onDeviceSelected
            )

            SectionCard(
                title = "Device Snapshot",
                subtitle = "The tables below are built from the live backend JSON keys, not printed raw."
            ) {
                ScrollableTabRow(
                    selectedTabIndex = DiagnosticsTab.entries.indexOf(selectedTab),
                    edgePadding = 0.dp
                ) {
                    DiagnosticsTab.entries.forEach { tab ->
                        Tab(
                            selected = tab == selectedTab,
                            onClick = { selectedTab = tab },
                            text = { Text(tab.label) }
                        )
                    }
                }

                when (selectedTab) {
                    DiagnosticsTab.OVERVIEW -> {
                        FlowRow(
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            state.capabilityRows.forEach { row ->
                                StatusStrip(label = row.label, value = row.value)
                            }
                        }
                        InfoGroupTable(
                            title = "Identity",
                            rows = state.deviceInfoRows.filter { it.label in listOf("Manufacturer", "Model", "Device", "Firmware", "Build") }
                        )
                        InfoGroupTable(
                            title = "Display",
                            rows = state.deviceInfoRows.filter { it.label in listOf("Display") }
                        )
                        InfoGroupTable(
                            title = "Connectivity",
                            rows = state.deviceInfoRows.filter { it.label in listOf("Network", "IP Address") }
                        )
                    }
                    DiagnosticsTab.OPERATIONS -> {
                        TableHeaderRow(columns = listOf("Operation", "Action", "Status", "Count"))
                        state.operationRows.forEach { row ->
                            Surface(
                                modifier = Modifier.fillMaxWidth(),
                                tonalElevation = 2.dp,
                                shape = MaterialTheme.shapes.medium
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(12.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(row.operation, modifier = Modifier.weight(1.5f), style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                                    Text(row.defaultAction, modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text(if (row.supported) "Supported" else "Unavailable", modifier = Modifier.weight(0.9f), style = MaterialTheme.typography.bodySmall)
                                    Text(row.relatedCount.toString(), modifier = Modifier.weight(0.4f), style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
                                }
                            }
                        }
                    }
                    DiagnosticsTab.SETTINGS -> {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text("Current Settings", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                            Box {
                                OutlinedButton(onClick = { settingsExpanded = true }) {
                                    Text(settingsFilter.label)
                                }
                                DropdownMenu(
                                    expanded = settingsExpanded,
                                    onDismissRequest = { settingsExpanded = false }
                                ) {
                                    SettingsFilter.entries.forEach { option ->
                                        DropdownMenuItem(
                                            text = { Text(option.label) },
                                            onClick = {
                                                settingsFilter = option
                                                settingsExpanded = false
                                            }
                                        )
                                    }
                                }
                            }
                        }
                        TableHeaderRow(columns = listOf("Setting", "Value", "Access", "State"))
                        filteredSettings.forEach { row ->
                            Surface(
                                modifier = Modifier.fillMaxWidth(),
                                tonalElevation = 2.dp,
                                shape = MaterialTheme.shapes.medium
                            ) {
                                Row(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .padding(12.dp),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(row.name, modifier = Modifier.weight(1.2f), style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.Medium)
                                    Text(row.value, modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                    Text(if (row.writable) "Writable" else "Read only", modifier = Modifier.weight(0.7f), style = MaterialTheme.typography.bodySmall)
                                    Text(row.status, modifier = Modifier.weight(0.5f), style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun HeaderCard(
    isLoading: Boolean,
    refreshStatus: String,
    error: String?,
    onRefresh: () -> Unit,
    onOpenDeviceInfo: () -> Unit
) {
    ElevatedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text("Control Deck", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
            Text(
                "Live stream, DAB remote control, IR tools, and automation in one focused workspace.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                FilledTonalButton(onClick = onRefresh) { Text("Refresh Everything") }
                OutlinedButton(onClick = onOpenDeviceInfo) { Text("Open Device Info") }
                if (isLoading) {
                    CircularProgressIndicator(modifier = Modifier.height(20.dp))
                }
            }
            StatusStrip(label = "Sync", value = refreshStatus)
            StatusStrip(label = "Device Data", value = "Moved to dedicated Device Info page")
            error?.let {
                StatusStrip(label = "Issue", value = it, emphasized = true)
            }
        }
    }
}

@Composable
private fun DeviceSelectionCard(
    selectedDeviceId: String,
    deviceIds: List<String>,
    onDeviceSelected: (String) -> Unit
) {
    SectionCard(
        title = "Target Device",
        subtitle = "Choose the active device before sending DAB or IR actions."
    ) {
        OutlinedTextField(
            value = selectedDeviceId,
            onValueChange = onDeviceSelected,
            label = { Text("Selected Device ID") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )
        if (deviceIds.isNotEmpty()) {
            ActionRow {
                deviceIds.forEach { id ->
                    OutlinedButton(onClick = { onDeviceSelected(id) }) {
                        Text(id)
                    }
                }
            }
        }
    }
}

@Composable
private fun LiveControlCard(
    isStreaming: Boolean,
    streamFrameBytes: ByteArray?,
    streamStatus: String,
    remoteStatus: String,
    onToggleStream: () -> Unit,
    onRefreshStream: () -> Unit,
    onSendRemoteAction: (String) -> Unit
) {
    val frameBitmap = streamFrameBytes?.let { bytes ->
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size)?.asImageBitmap()
    }

    SectionCard(
        title = "Live Stream & Remote",
        subtitle = "Operate the device while keeping the HDMI feed visible."
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            FilledTonalButton(onClick = onToggleStream) {
                Text(if (isStreaming) "Stop Stream" else "Start Stream")
            }
            OutlinedButton(onClick = onRefreshStream) {
                Text("Reconnect")
            }
        }

        Card(modifier = Modifier.fillMaxWidth()) {
            if (frameBitmap != null) {
                Image(
                    bitmap = frameBitmap,
                    contentDescription = "Live HDMI stream",
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(16f / 9f),
                    contentScale = ContentScale.Fit
                )
            } else {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .aspectRatio(16f / 9f),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        if (isStreaming) "Waiting for HDMI video..." else "HDMI stream is stopped.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        StatusStrip(label = "Stream", value = streamStatus)
        HorizontalDivider()
        RemotePad(onSendRemoteAction = onSendRemoteAction)
        StatusStrip(label = "Remote", value = remoteStatus)
    }
}

@Composable
private fun FloatingStreamOverlay(
    streamFrameBytes: ByteArray?,
    streamStatus: String,
    onHide: () -> Unit,
    modifier: Modifier = Modifier
) {
    val frameBitmap = streamFrameBytes?.let { bytes ->
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size)?.asImageBitmap()
    }
    var offsetX by remember { mutableFloatStateOf(0f) }
    var offsetY by remember { mutableFloatStateOf(0f) }

    ElevatedCard(
        modifier = modifier
            .padding(12.dp)
            .offset { IntOffset(offsetX.toInt(), offsetY.toInt()) }
            .pointerInput(Unit) {
                detectDragGestures { change, dragAmount ->
                    change.consume()
                    offsetX += dragAmount.x
                    offsetY += dragAmount.y
                }
            }
    ) {
        Column(
            modifier = Modifier.padding(8.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Row(
                modifier = Modifier.width(300.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Live", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
                OutlinedButton(onClick = onHide) {
                    Text("Hide")
                }
            }
            Card(modifier = Modifier.width(300.dp)) {
                if (frameBitmap != null) {
                    Image(
                        bitmap = frameBitmap,
                        contentDescription = "Floating live stream",
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(16f / 9f),
                        contentScale = ContentScale.Fit
                    )
                } else {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(16f / 9f),
                        contentAlignment = Alignment.Center
                    ) {
                        Text("Waiting...", color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
            }
            Text(streamStatus, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun RemotePad(
    onSendRemoteAction: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            FilledTonalButton(onClick = { onSendRemoteAction("PRESS_UP") }) { Text("Up") }
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            OutlinedButton(onClick = { onSendRemoteAction("PRESS_LEFT") }) { Text("Left") }
            FilledTonalButton(onClick = { onSendRemoteAction("PRESS_OK") }) { Text("OK") }
            OutlinedButton(onClick = { onSendRemoteAction("PRESS_RIGHT") }) { Text("Right") }
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            FilledTonalButton(onClick = { onSendRemoteAction("PRESS_DOWN") }) { Text("Down") }
        }
        ActionRow {
            listOf(
                "Back" to "PRESS_BACK",
                "Home" to "PRESS_HOME",
                "Menu" to "PRESS_MENU",
                "Info" to "PRESS_INFO",
                "Play/Pause" to "PRESS_PLAY_PAUSE",
                "Power" to "PRESS_POWER"
            ).forEach { (label, action) ->
                OutlinedButton(onClick = { onSendRemoteAction(action) }) {
                    Text(label)
                }
            }
        }
    }
}

@Composable
private fun ActionWorkbenchCard(
    actionName: String,
    actionParamsJson: String,
    batchActionsJson: String,
    lastActionResult: String,
    lastBatchResult: String,
    onActionChanged: (String) -> Unit,
    onActionParamsChanged: (String) -> Unit,
    onBatchActionsChanged: (String) -> Unit,
    onExecuteAction: () -> Unit,
    onExecuteBatch: () -> Unit
) {
    SectionCard(
        title = "Action Workbench",
        subtitle = "Send direct manual actions or stage multi-step batches for advanced control."
    ) {
        Text("Single Action", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Medium)
        OutlinedTextField(
            value = actionName,
            onValueChange = onActionChanged,
            label = { Text("Action") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )
        OutlinedTextField(
            value = actionParamsJson,
            onValueChange = onActionParamsChanged,
            label = { Text("Action Params JSON") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 3
        )
        FilledTonalButton(onClick = onExecuteAction) { Text("Run Action") }
        PreviewBlock(title = "Last Action Result", value = lastActionResult)

        HorizontalDivider()

        Text("Batch Actions", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Medium)
        OutlinedTextField(
            value = batchActionsJson,
            onValueChange = onBatchActionsChanged,
            label = { Text("Batch JSON Array") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 5
        )
        FilledTonalButton(onClick = onExecuteBatch) { Text("Run Batch") }
        PreviewBlock(title = "Last Batch Result", value = lastBatchResult)
    }
}

@Composable
private fun InfoGroupTable(
    title: String,
    rows: List<ControlsInfoRow>
) {
    if (rows.isEmpty()) return
    Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
    CompactTable(rows = rows)
}

@Composable
private fun CompactTable(rows: List<ControlsInfoRow>) {
    rows.forEach { row ->
        Surface(
            modifier = Modifier.fillMaxWidth(),
            tonalElevation = 2.dp,
            shape = MaterialTheme.shapes.medium
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(12.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    row.label,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    row.value,
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }
    }
}

@Composable
private fun TableHeaderRow(columns: List<String>) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f),
        shape = MaterialTheme.shapes.medium
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp)
        ) {
            columns.forEach { column ->
                Text(
                    text = column,
                    modifier = Modifier.weight(1f),
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun IrControlsCard(
    irDeviceId: String,
    irKeyName: String,
    irStatusPreview: String,
    irDevicesPreview: String,
    irKeysPreview: String,
    irLastResult: String,
    onIrDeviceChanged: (String) -> Unit,
    onIrKeyChanged: (String) -> Unit,
    onFetchIrKeys: () -> Unit,
    onIrSend: () -> Unit,
    onIrTrain: () -> Unit
) {
    SectionCard(
        title = "IR Tools",
        subtitle = "Fallback infrared controls for devices or commands not covered by DAB."
    ) {
        OutlinedTextField(
            value = irDeviceId,
            onValueChange = onIrDeviceChanged,
            label = { Text("IR Device ID") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )
        OutlinedTextField(
            value = irKeyName,
            onValueChange = onIrKeyChanged,
            label = { Text("IR Key Name") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )
        ActionRow {
            FilledTonalButton(onClick = onFetchIrKeys) { Text("Load Keys") }
            OutlinedButton(onClick = onIrSend) { Text("Send Key") }
            OutlinedButton(onClick = onIrTrain) { Text("Train Key") }
        }
        PreviewBlock(title = "IR Status", value = irStatusPreview)
        PreviewBlock(title = "IR Devices", value = irDevicesPreview)
        PreviewBlock(title = "IR Keys", value = irKeysPreview)
        PreviewBlock(title = "IR Last Result", value = irLastResult)
    }
}

@Composable
private fun AutomationLabCard(
    macroInstruction: String,
    macroExecute: Boolean,
    macroResult: String,
    plannerGoal: String,
    plannerCurrentApp: String,
    plannerCurrentScreen: String,
    plannerOcrText: String,
    plannerResult: String,
    onMacroInstructionChanged: (String) -> Unit,
    onToggleMacroExecute: () -> Unit,
    onRunMacro: () -> Unit,
    onPlannerGoalChanged: (String) -> Unit,
    onPlannerAppChanged: (String) -> Unit,
    onPlannerScreenChanged: (String) -> Unit,
    onPlannerOcrChanged: (String) -> Unit,
    onRunPlannerDebug: () -> Unit
) {
    SectionCard(
        title = "Automation Lab",
        subtitle = "Prototype macros and planner requests without leaving the control surface."
    ) {
        Text("Task Macro", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Medium)
        OutlinedTextField(
            value = macroInstruction,
            onValueChange = onMacroInstructionChanged,
            label = { Text("Instruction") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 2
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Execute immediately")
            Switch(checked = macroExecute, onCheckedChange = { onToggleMacroExecute() })
        }
        FilledTonalButton(onClick = onRunMacro) { Text("Run Macro") }
        PreviewBlock(title = "Macro Result", value = macroResult)

        HorizontalDivider()

        Text("Planner Debug", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Medium)
        OutlinedTextField(
            value = plannerGoal,
            onValueChange = onPlannerGoalChanged,
            label = { Text("Goal") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 2
        )
        OutlinedTextField(
            value = plannerCurrentApp,
            onValueChange = onPlannerAppChanged,
            label = { Text("Current App (optional)") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )
        OutlinedTextField(
            value = plannerCurrentScreen,
            onValueChange = onPlannerScreenChanged,
            label = { Text("Current Screen (optional)") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )
        OutlinedTextField(
            value = plannerOcrText,
            onValueChange = onPlannerOcrChanged,
            label = { Text("OCR Text (optional)") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 3
        )
        FilledTonalButton(onClick = onRunPlannerDebug) { Text("Run Planner Debug") }
        PreviewBlock(title = "Planner Result", value = plannerResult)
    }
}

@Composable
private fun SectionCard(
    title: String,
    subtitle: String,
    content: @Composable () -> Unit
) {
    ElevatedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text(
                subtitle,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            content()
        }
    }
}

@Composable
private fun PreviewBlock(
    title: String,
    value: String
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(title, style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Medium)
        Surface(
            modifier = Modifier.fillMaxWidth(),
            tonalElevation = 2.dp,
            shape = MaterialTheme.shapes.medium
        ) {
            SelectionContainer {
                Text(
                    text = value,
                    modifier = Modifier.padding(12.dp),
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace
                )
            }
        }
    }
}

@Composable
private fun StatusStrip(
    label: String,
    value: String,
    emphasized: Boolean = false
) {
    Surface(
        color = if (emphasized) {
            MaterialTheme.colorScheme.errorContainer
        } else {
            MaterialTheme.colorScheme.secondaryContainer
        },
        shape = MaterialTheme.shapes.medium
    ) {
        Text(
            text = "$label: $value",
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            style = MaterialTheme.typography.bodyMedium,
            color = if (emphasized) {
                MaterialTheme.colorScheme.onErrorContainer
            } else {
                MaterialTheme.colorScheme.onSecondaryContainer
            }
        )
    }
}

@Composable
private fun ActionRow(
    content: @Composable RowScope.() -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        content = content
    )
}
