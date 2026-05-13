package com.dabcontrol.app.ui.controls

import android.graphics.BitmapFactory
import androidx.compose.foundation.layout.Spacer
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
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
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
import androidx.media3.common.MediaItem
import androidx.media3.common.Player
import androidx.media3.exoplayer.ExoPlayer
import androidx.compose.ui.platform.LocalContext
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
                    selectedDeviceName = state.selectedDeviceName,
                    selectedYtsDeviceId = state.selectedYtsDeviceId,
                    selectedYtsShortId = state.selectedYtsShortId,
                    selectedIrDeviceId = state.selectedIrDeviceId,
                    selectedVideoSource = state.selectedVideoSource,
                    selectedContextIssues = state.selectedContextIssues,
                    deviceContexts = state.deviceContexts,
                    deviceIds = state.deviceIds,
                    onDeviceSelected = viewModel::onDeviceSelected
                )

                LiveControlCard(
                    apiBaseUrl = state.apiBaseUrl,
                    selectedDeviceName = state.selectedDeviceName,
                    selectedDeviceId = state.selectedDeviceId,
                    selectedIrDeviceId = state.selectedIrDeviceId,
                    isAudioStreaming = state.isAudioStreaming,
                    audioStatus = state.audioStatus,
                    audioSource = state.audioSource,
                    remoteMode = state.remoteMode,
                    isStreaming = state.isStreaming,
                    streamFrameBytes = state.streamFrameBytes,
                    streamStatus = state.streamStatus,
                    remoteStatus = state.remoteStatus,
                    irAvailableKeys = state.irAvailableKeys,
                    onToggleAudioStream = viewModel::toggleAudioStream,
                    onAudioPlaybackReady = viewModel::onAudioPlaybackReady,
                    onAudioPlaybackError = viewModel::onAudioPlaybackError,
                    onRemoteModeChanged = viewModel::onRemoteModeChanged,
                    onToggleStream = viewModel::toggleStream,
                    onRefreshStream = viewModel::refreshStream,
                    onSendRemoteAction = viewModel::sendRemoteAction
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
                selectedDeviceName = state.selectedDeviceName,
                selectedYtsDeviceId = state.selectedYtsDeviceId,
                selectedYtsShortId = state.selectedYtsShortId,
                selectedIrDeviceId = state.selectedIrDeviceId,
                selectedVideoSource = state.selectedVideoSource,
                selectedContextIssues = state.selectedContextIssues,
                deviceContexts = state.deviceContexts,
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
    selectedDeviceName: String,
    selectedYtsDeviceId: String,
    selectedYtsShortId: String,
    selectedIrDeviceId: String,
    selectedVideoSource: String,
    selectedContextIssues: List<String>,
    deviceContexts: List<ControlsDeviceContext>,
    deviceIds: List<String>,
    onDeviceSelected: (String) -> Unit
) {
    SectionCard(
        title = "Target Device",
        subtitle = "Choose one unified device context so DAB, YTS, IR, and video all stay routed together."
    ) {
        if (selectedDeviceId.isNotBlank()) {
            Surface(
                modifier = Modifier.fillMaxWidth(),
                tonalElevation = 2.dp,
                shape = MaterialTheme.shapes.medium
            ) {
                Column(
                    modifier = Modifier.padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        selectedDeviceName.ifBlank { selectedDeviceId },
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                    CompactTable(
                        rows = listOf(
                            ControlsInfoRow("DAB", selectedDeviceId),
                            ControlsInfoRow("YTS Config", selectedYtsDeviceId.ifBlank { "--" }),
                            ControlsInfoRow("IR", selectedIrDeviceId.ifBlank { "--" }),
                            ControlsInfoRow("Video", selectedVideoSource.ifBlank { "--" })
                        )
                    )
                }
            }
        }
        if (deviceContexts.isNotEmpty()) {
            Text("Available contexts", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Medium)
            ActionRow {
                deviceContexts.forEach { context ->
                    val selected = context.dabDeviceId == selectedDeviceId
                    FilledTonalButton(
                        onClick = { onDeviceSelected(context.dabDeviceId) },
                        enabled = !selected
                    ) {
                        Text(context.displayName)
                    }
                }
            }
            deviceContexts.firstOrNull { it.dabDeviceId == selectedDeviceId }?.let { context ->
                StatusStrip(
                    label = "Context readiness",
                    value = if (context.isReady) "Mapped for DAB, YTS, IR, and capture" else "Needs attention"
                )
            }
            if (selectedYtsShortId.isNotBlank()) {
                Text(
                    "YTS runtime mapping is resolved automatically in the background.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        } else if (deviceIds.isNotEmpty()) {
            ActionRow {
                deviceIds.forEach { id ->
                    OutlinedButton(onClick = { onDeviceSelected(id) }) { Text(id) }
                }
            }
        }
        if (selectedContextIssues.isNotEmpty()) {
            selectedContextIssues.forEach { issue ->
                StatusStrip(label = "Context issue", value = issue, emphasized = true)
            }
        }
    }
}

@Composable
private fun LiveControlCard(
    apiBaseUrl: String,
    selectedDeviceName: String,
    selectedDeviceId: String,
    selectedIrDeviceId: String,
    isAudioStreaming: Boolean,
    audioStatus: String,
    audioSource: ControlsAudioSource?,
    remoteMode: ControlsRemoteMode,
    isStreaming: Boolean,
    streamFrameBytes: ByteArray?,
    streamStatus: String,
    remoteStatus: String,
    irAvailableKeys: List<String>,
    onToggleAudioStream: () -> Unit,
    onAudioPlaybackReady: () -> Unit,
    onAudioPlaybackError: (String) -> Unit,
    onRemoteModeChanged: (ControlsRemoteMode) -> Unit,
    onToggleStream: () -> Unit,
    onRefreshStream: () -> Unit,
    onSendRemoteAction: (String) -> Unit
) {
    val context = LocalContext.current
    val frameBitmap = streamFrameBytes?.let { bytes ->
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size)?.asImageBitmap()
    }
    val audioUrl = rememberAudioStreamUrl(apiBaseUrl)
    val audioPlayer = remember(audioUrl) {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(audioUrl))
            prepare()
        }
    }

    DisposableEffect(audioPlayer) {
        val listener = object : Player.Listener {
            override fun onPlaybackStateChanged(playbackState: Int) {
                if (playbackState == Player.STATE_READY && audioPlayer.playWhenReady) {
                    onAudioPlaybackReady()
                }
            }

            override fun onPlayerError(error: androidx.media3.common.PlaybackException) {
                onAudioPlaybackError(error.message ?: "Audio stream failed.")
            }
        }
        audioPlayer.addListener(listener)
        onDispose {
            audioPlayer.removeListener(listener)
            audioPlayer.release()
        }
    }

    LaunchedEffect(isAudioStreaming, audioUrl) {
        if (isAudioStreaming) {
            audioPlayer.setMediaItem(MediaItem.fromUri(audioUrl))
            audioPlayer.prepare()
            audioPlayer.playWhenReady = true
        } else {
            audioPlayer.playWhenReady = false
            audioPlayer.stop()
        }
    }

    SectionCard(
        title = "Live Stream & Remote",
        subtitle = "Operate the device while keeping the HDMI video and audio streams available for the selected device."
    ) {
        if (selectedDeviceId.isNotBlank()) {
            StatusStrip(
                label = if (remoteMode == ControlsRemoteMode.DAB) "DAB Route" else "IR Route",
                value = if (remoteMode == ControlsRemoteMode.DAB) {
                    "${selectedDeviceName.ifBlank { selectedDeviceId }} · ${selectedDeviceId}"
                } else {
                    "${selectedDeviceName.ifBlank { selectedDeviceId }} · ${selectedIrDeviceId.ifBlank { "auto IR profile" }}"
                }
            )
        }
        ActionRow {
            FilledTonalButton(
                onClick = { onRemoteModeChanged(ControlsRemoteMode.DAB) },
                enabled = remoteMode != ControlsRemoteMode.DAB
            ) {
                Text("DAB Remote")
            }
            FilledTonalButton(
                onClick = { onRemoteModeChanged(ControlsRemoteMode.IR) },
                enabled = remoteMode != ControlsRemoteMode.IR
            ) {
                Text("IR Remote")
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            FilledTonalButton(onClick = onToggleStream) {
                Text(if (isStreaming) "Stop Stream" else "Start Stream")
            }
            FilledTonalButton(
                onClick = onToggleAudioStream,
                enabled = audioSource?.enabled != false && audioSource?.ffmpegAvailable != false
            ) {
                Text(if (isAudioStreaming) "Stop Audio" else "Start Audio")
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
        StatusStrip(label = "Audio", value = audioStatus)
        audioSource?.let { source ->
            CompactTable(
                rows = listOf(
                    ControlsInfoRow("Input", source.inputFormat.ifBlank { "--" }),
                    ControlsInfoRow("Device", source.device.ifBlank { "Auto-selected" }),
                    ControlsInfoRow("Sample Rate", source.sampleRate.ifBlank { "--" }),
                    ControlsInfoRow("Channels", source.channels.ifBlank { "--" })
                )
            )
        }
        StatusStrip(
            label = "Remote Mode",
            value = if (remoteMode == ControlsRemoteMode.DAB) {
                "Direct DAB control for the selected device"
            } else {
                "IR control auto-configured from the selected device context"
            }
        )
        if (remoteMode == ControlsRemoteMode.IR && irAvailableKeys.isNotEmpty()) {
            StatusStrip(
                label = "IR Profile",
                value = "${irAvailableKeys.size} mapped keys loaded for ${selectedIrDeviceId.ifBlank { "current profile" }}"
            )
        }
        HorizontalDivider()
        RemotePad(
            remoteMode = remoteMode,
            onSendRemoteAction = onSendRemoteAction
        )
        StatusStrip(label = "Remote", value = remoteStatus)
    }
}

@Composable
private fun rememberAudioStreamUrl(apiBaseUrl: String): String {
    return remember(apiBaseUrl) {
        "${apiBaseUrl.trimEnd('/')}/stream/audio?ts=${System.currentTimeMillis()}"
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
    remoteMode: ControlsRemoteMode,
    onSendRemoteAction: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text(
            if (remoteMode == ControlsRemoteMode.DAB) "Precision D-pad for direct DAB input" else "IR remote mode uses the same pad with IR key routing",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            RemoteDirectionalButton(label = "Up", onClick = { onSendRemoteAction("PRESS_UP") })
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            RemoteDirectionalButton(label = "Left", onClick = { onSendRemoteAction("PRESS_LEFT") })
            RemoteCenterButton(label = "OK", onClick = { onSendRemoteAction("PRESS_OK") })
            RemoteDirectionalButton(label = "Right", onClick = { onSendRemoteAction("PRESS_RIGHT") })
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center
        ) {
            RemoteDirectionalButton(label = "Down", onClick = { onSendRemoteAction("PRESS_DOWN") })
        }
        ActionRow {
            listOf(
                "Back" to "PRESS_BACK",
                "Home" to "PRESS_HOME",
                "Menu" to "PRESS_MENU",
                "Info" to "PRESS_INFO",
                "Play/Pause" to "PRESS_PLAY_PAUSE",
                "Vol +" to "PRESS_VOLUME_UP",
                "Vol -" to "PRESS_VOLUME_DOWN",
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
private fun RemoteDirectionalButton(
    label: String,
    onClick: () -> Unit
) {
    FilledTonalButton(
        onClick = onClick,
        modifier = Modifier
            .width(96.dp)
            .height(52.dp)
    ) {
        Text(label)
    }
}

@Composable
private fun RemoteCenterButton(
    label: String,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        modifier = Modifier
            .width(104.dp)
            .height(56.dp)
    ) {
        Text(label)
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
