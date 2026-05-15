package com.dabcontrol.app.ui.dashboard

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.dabcontrol.app.ui.common.PremiumBackdrop
import com.dabcontrol.app.ui.common.PremiumPanel
import com.dabcontrol.app.ui.common.SectionLabel
import kotlin.math.roundToInt

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun DashboardScreen(
    modifier: Modifier = Modifier,
    viewModel: DashboardViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()

    PremiumBackdrop(modifier = modifier) {
        Column(
            modifier = Modifier
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            SectionLabel(
                eyebrow = "Operations Center",
                title = "Advanced Device and Backend Dashboard",
                subtitle = "Auto-refreshing health, performance, model, and device telemetry in one operator view."
            )

            DashboardCommandDeck(
                state = state,
                onUrlChanged = viewModel::onApiBaseUrlChanged,
                onSaveUrl = viewModel::saveApiBaseUrl,
                onRefresh = viewModel::refresh,
                onToggleAutoRefresh = viewModel::toggleAutoRefresh,
                onPlannerModelChanged = viewModel::onPlannerModelChanged,
                onLiveModelChanged = viewModel::onLiveModelChanged,
                onApplyModels = viewModel::applyRuntimeModels,
                onSelectDevice = viewModel::onDeviceSelected
            )

            KpiRow(state = state)

            CombinedStatusCard(state = state)

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                maxItemsInEachRow = 2
            ) {
                MetricCard(
                    title = "CPU",
                    value = state.cpuPercent?.let { "${format1(it)}%" } ?: "--",
                    subtitle = "Processor utilization",
                    accent = Color(0xFF6366F1),
                    history = state.cpuHistory,
                    maxValue = 100f,
                    modifier = Modifier.fillMaxWidth()
                )
                MetricCard(
                    title = "Memory",
                    value = state.ramPercent?.let { "${format1(it)}%" } ?: "--",
                    subtitle = "RAM pressure",
                    accent = Color(0xFF8B5CF6),
                    history = state.ramHistory,
                    maxValue = 100f,
                    modifier = Modifier.fillMaxWidth()
                )
                MetricCard(
                    title = "Load",
                    value = state.load1m?.let { format2(it) } ?: "--",
                    subtitle = "1 minute system load",
                    accent = Color(0xFF3B82F6),
                    history = state.loadHistory,
                    maxValue = (state.loadHistory.maxOfOrNull { it.value } ?: 1f).coerceAtLeast(1f),
                    modifier = Modifier.fillMaxWidth()
                )
                MetricCard(
                    title = "Temperature",
                    value = state.cpuTempC?.let { "${format1(it)}C" } ?: "--",
                    subtitle = "Thermal signal",
                    accent = Color(0xFF818CF8),
                    history = state.tempHistory,
                    maxValue = (state.tempHistory.maxOfOrNull { it.value } ?: 100f).coerceAtLeast(50f),
                    modifier = Modifier.fillMaxWidth()
                )
            }

            BackendEvidenceCard(state = state)
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DashboardCommandDeck(
    state: DashboardUiState,
    onUrlChanged: (String) -> Unit,
    onSaveUrl: () -> Unit,
    onRefresh: () -> Unit,
    onToggleAutoRefresh: () -> Unit,
    onPlannerModelChanged: (String) -> Unit,
    onLiveModelChanged: (String) -> Unit,
    onApplyModels: () -> Unit,
    onSelectDevice: (String) -> Unit
) {
    PremiumPanel(
        modifier = Modifier.fillMaxWidth(),
        accentColor = MaterialTheme.colorScheme.primary
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Control Deck", style = MaterialTheme.typography.titleLarge)
                    Text(
                        "Keep the backend endpoint, shared device, and Gemini runtime aligned from one place.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                if (state.isLoading && state.cpuHistory.isEmpty()) {
                    CircularProgressIndicator(modifier = Modifier.size(22.dp), strokeWidth = 2.5.dp)
                }
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
                maxItemsInEachRow = 3
            ) {
                StatusPill("Health", state.healthStatus, state.healthStatus.equals("ok", ignoreCase = true))
                StatusPill("Mode", state.mode, true)
                StatusPill("Refresh", state.refreshStateLabel, state.autoRefreshEnabled)
            }

            UrlPresetPicker(currentUrl = state.apiBaseUrl, onSelected = onUrlChanged)
            OutlinedTextField(
                value = state.apiBaseUrl,
                onValueChange = onUrlChanged,
                label = { Text("Backend API Base URL") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                FilledTonalButton(onClick = onSaveUrl) {
                    Text("Save Endpoint")
                }
                OutlinedButton(onClick = onRefresh) {
                    Text("Refresh Now")
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Switch(
                        checked = state.autoRefreshEnabled,
                        onCheckedChange = { onToggleAutoRefresh() }
                    )
                    Text("Auto refresh", style = MaterialTheme.typography.bodyMedium)
                }
            }

            DevicePickerCard(
                deviceContexts = state.deviceContexts,
                selectedDeviceId = state.selectedDeviceId,
                onSelect = onSelectDevice
            )

            HorizontalDivider()

            Text("Gemini Runtime Models", style = MaterialTheme.typography.titleMedium)
            Text(state.modelStatus, color = MaterialTheme.colorScheme.onSurfaceVariant)
            ModelDropdownField(
                title = "Planner model",
                currentValue = state.plannerModel,
                availableModels = state.availableModels,
                onValueChanged = onPlannerModelChanged
            )
            ModelDropdownField(
                title = "Live/visual model",
                currentValue = state.liveModel,
                availableModels = state.availableModels,
                onValueChanged = onLiveModelChanged
            )
            FilledTonalButton(onClick = onApplyModels) {
                Text("Apply Runtime Models")
            }

            state.error?.let { ErrorStrip(it) }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun KpiRow(state: DashboardUiState) {
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        maxItemsInEachRow = 4
    ) {
        KpiCard("Backend", state.healthStatus, "Health signal", Modifier.fillMaxWidth())
        KpiCard("Device", state.selectedDeviceName.ifBlank { state.selectedDeviceId.ifBlank { "--" } }, "Selected target", Modifier.fillMaxWidth())
        KpiCard("Cores", state.cpuCount?.toString() ?: "--", "CPU count", Modifier.fillMaxWidth())
        KpiCard("Sample", state.timestamp.substringAfter('T', state.timestamp).substringBefore('.'), "Last refresh", Modifier.fillMaxWidth())
    }
}

@Composable
private fun KpiCard(title: String, value: String, subtitle: String, modifier: Modifier = Modifier) {
    PremiumPanel(
        modifier = modifier,
        accentColor = MaterialTheme.colorScheme.tertiary
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(title, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(value, style = MaterialTheme.typography.headlineSmall)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun CombinedStatusCard(state: DashboardUiState) {
    PremiumPanel(
        modifier = Modifier.fillMaxWidth(),
        accentColor = MaterialTheme.colorScheme.tertiary
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Unified Status Graph", style = MaterialTheme.typography.titleLarge)
            Text(
                "One operator graph for CPU, memory, load, and thermal behavior. This is the fastest way to see whether the backend is stable or drifting.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            CombinedStatusGraph(
                cpuHistory = state.cpuHistory,
                ramHistory = state.ramHistory,
                loadHistory = state.loadHistory,
                tempHistory = state.tempHistory
            )
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                LegendPill("CPU", Color(0xFF6366F1))
                LegendPill("Memory", Color(0xFF8B5CF6))
                LegendPill("Load", Color(0xFF3B82F6))
                LegendPill("Temp", Color(0xFF818CF8))
            }
            Text(
                state.backendStatusSummary,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun CombinedStatusGraph(
    cpuHistory: List<MetricPoint>,
    ramHistory: List<MetricPoint>,
    loadHistory: List<MetricPoint>,
    tempHistory: List<MetricPoint>
) {
    val outlineColor = MaterialTheme.colorScheme.outlineVariant
    val graphSurface = Brush.verticalGradient(
        listOf(
            MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.22f),
            MaterialTheme.colorScheme.surface
        )
    )
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = Color.Transparent,
        tonalElevation = 2.dp
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .background(graphSurface, MaterialTheme.shapes.medium)
                .padding(12.dp)
        ) {
            if (cpuHistory.size < 2 && ramHistory.size < 2 && loadHistory.size < 2 && tempHistory.size < 2) {
                Text(
                    "Collecting live telemetry samples...",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            } else {
                Canvas(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(240.dp)
                ) {
                    val baseline = size.height
                    val maxPoints = listOf(cpuHistory.size, ramHistory.size, loadHistory.size, tempHistory.size).maxOrNull() ?: 0
                    val stepX = if (maxPoints <= 1) size.width else size.width / (maxPoints - 1)
                    repeat(4) { index ->
                        val y = baseline - (baseline * (index / 3f))
                        drawLine(
                            color = outlineColor,
                            start = Offset(0f, y),
                            end = Offset(size.width, y),
                            strokeWidth = 1.2f
                        )
                    }
                    drawRoundRect(
                        color = outlineColor.copy(alpha = 0.28f),
                        style = Stroke(width = 2f),
                        cornerRadius = CornerRadius(18f, 18f)
                    )
                    drawSeries(cpuHistory, stepX, baseline, 100f, Color(0xFF6366F1))
                    drawSeries(ramHistory, stepX, baseline, 100f, Color(0xFF8B5CF6))
                    drawSeries(loadHistory, stepX, baseline, (loadHistory.maxOfOrNull { it.value } ?: 1f).coerceAtLeast(1f), Color(0xFF3B82F6))
                    drawSeries(tempHistory, stepX, baseline, (tempHistory.maxOfOrNull { it.value } ?: 100f).coerceAtLeast(50f), Color(0xFF818CF8))
                }
            }
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawSeries(
    points: List<MetricPoint>,
    stepX: Float,
    baseline: Float,
    maxValue: Float,
    color: Color
) {
    if (points.size < 2) return
    val path = Path()
    points.forEachIndexed { index, point ->
        val x = stepX * index
        val y = baseline - ((point.value / maxValue).coerceIn(0f, 1f) * baseline)
        if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
    }
    drawPath(path = path, color = color, style = Stroke(width = 6f, cap = StrokeCap.Round))
    points.forEachIndexed { index, point ->
        val x = stepX * index
        val y = baseline - ((point.value / maxValue).coerceIn(0f, 1f) * baseline)
        drawCircle(color = color, radius = 5f, center = Offset(x, y))
    }
}

@Composable
private fun LegendPill(label: String, color: Color) {
    Surface(color = color.copy(alpha = 0.12f), shape = MaterialTheme.shapes.small) {
        Row(
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(modifier = Modifier.size(10.dp).background(color, CircleShape))
            Text(label, style = MaterialTheme.typography.bodySmall)
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun UrlPresetPicker(
    currentUrl: String,
    onSelected: (String) -> Unit
) {
    val presets = listOf(
        "Local Lab" to "http://10.99.57.66:8081",
        "Cloud Tunnel" to "https://creative-airline-maintaining-manufacturers.trycloudflare.com"
    )
    var expanded by remember(currentUrl) { mutableStateOf(false) }
    val selectedLabel = presets.firstOrNull { it.second == currentUrl }?.first ?: "Custom Endpoint"

    ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = !expanded }) {
        OutlinedTextField(
            value = selectedLabel,
            onValueChange = {},
            readOnly = true,
            label = { Text("Endpoint Preset") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier.menuAnchor().fillMaxWidth()
        )
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            presets.forEach { (label, url) ->
                DropdownMenuItem(
                    text = { Text("$label: $url") },
                    onClick = {
                        onSelected(url)
                        expanded = false
                    }
                )
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun DevicePickerCard(
    deviceContexts: List<DashboardDeviceContext>,
    selectedDeviceId: String,
    onSelect: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("Shared Device", style = MaterialTheme.typography.titleMedium)
        Text(
            "The selected device is reused across dashboard, live control, and YTS actions.",
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        if (deviceContexts.isEmpty()) {
            Text("No devices loaded yet.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else {
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                deviceContexts.forEach { context ->
                    val selected = context.dabDeviceId == selectedDeviceId
                    Surface(
                        color = if (selected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant,
                        shape = MaterialTheme.shapes.small,
                        modifier = Modifier.clickable { onSelect(context.dabDeviceId) }
                    ) {
                        Text(
                            text = context.displayName,
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
                            color = if (selected) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurface
                        )
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class, ExperimentalMaterial3Api::class)
@Composable
private fun ModelDropdownField(
    title: String,
    currentValue: String,
    availableModels: List<String>,
    onValueChanged: (String) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    var query by remember(currentValue) { mutableStateOf(currentValue) }
    val filteredModels = remember(query, availableModels) {
        availableModels.filter { model ->
            query.isBlank() || model.contains(query.trim(), ignoreCase = true)
        }
    }
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        ExposedDropdownMenuBox(
            expanded = expanded,
            onExpandedChange = { expanded = !expanded }
        ) {
            OutlinedTextField(
                value = query,
                onValueChange = {
                    query = it
                    expanded = true
                    onValueChanged(it)
                },
                label = { Text(title) },
                modifier = Modifier
                    .menuAnchor()
                    .fillMaxWidth(),
                singleLine = true,
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                supportingText = {
                    Text(
                        if (filteredModels.isEmpty()) "No matching models" else "${filteredModels.size} model options"
                    )
                },
                colors = TextFieldDefaults.colors()
            )
            DropdownMenu(
                expanded = expanded,
                onDismissRequest = { expanded = false }
            ) {
                filteredModels.take(12).forEach { model ->
                    DropdownMenuItem(
                        text = {
                            Text(
                                model,
                                maxLines = 1,
                                overflow = TextOverflow.Ellipsis
                            )
                        },
                        onClick = {
                            query = model
                            onValueChanged(model)
                            expanded = false
                        }
                    )
                }
            }
        }
        if (currentValue.isNotBlank()) {
            StatusPill("Selected", currentValue, positive = true)
        }
    }
}

@Composable
private fun MetricCard(
    title: String,
    value: String,
    subtitle: String,
    accent: Color,
    history: List<MetricPoint>,
    maxValue: Float,
    modifier: Modifier = Modifier
) {
    PremiumPanel(
        modifier = modifier,
        accentColor = accent
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(title, style = MaterialTheme.typography.titleLarge)
            Text(value, style = MaterialTheme.typography.headlineMedium, color = accent)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            SparklineChart(points = history, accent = accent, maxValue = maxValue)
            if (history.isNotEmpty()) {
                Text(
                    "Trend ${history.takeLast(4).joinToString("  ") { "${it.label} ${it.value.roundToInt()}" }}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun SparklineChart(points: List<MetricPoint>, accent: Color, maxValue: Float) {
    val outlineColor = MaterialTheme.colorScheme.outlineVariant
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f),
        shape = MaterialTheme.shapes.medium
    ) {
        if (points.size < 2) {
            Text(
                "Waiting for more samples...",
                modifier = Modifier.padding(12.dp),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        } else {
            Canvas(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(130.dp)
                    .padding(12.dp)
            ) {
                val stepX = size.width / (points.size - 1).coerceAtLeast(1)
                val baseline = size.height
                drawLine(
                    color = outlineColor,
                    start = Offset(0f, baseline),
                    end = Offset(size.width, baseline),
                    strokeWidth = 2f
                )
                drawSeries(points, stepX, baseline, maxValue.coerceAtLeast(1f), accent)
            }
        }
    }
}

@Composable
private fun BackendEvidenceCard(state: DashboardUiState) {
    PremiumPanel(
        modifier = Modifier.fillMaxWidth(),
        accentColor = MaterialTheme.colorScheme.secondary
    ) {
        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Backend Evidence Panel", style = MaterialTheme.typography.titleLarge)
            Text(
                "This panel makes it obvious that the application is talking to the backend and receiving live values, not showing placeholder content.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            StatusTable(
                rows = listOf(
                    "Health status" to state.healthStatus,
                    "Mode" to state.mode,
                    "Selected device" to state.selectedDeviceId.ifBlank { "--" },
                    "Timestamp" to state.timestamp,
                    "Metrics summary" to state.metricsPreview
                )
            )
            HorizontalDivider()
            Text("Metrics Table", style = MaterialTheme.typography.titleMedium)
            MetricsTable(state = state)
        }
    }
}

@Composable
private fun StatusTable(rows: List<Pair<String, String>>) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        rows.forEach { (label, value) ->
            Surface(
                color = MaterialTheme.colorScheme.surface,
                tonalElevation = 1.dp,
                shape = MaterialTheme.shapes.small
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(label, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text(value, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                }
            }
        }
    }
}

@Composable
private fun MetricsTable(state: DashboardUiState) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        MetricTableRow("CPU", state.cpuPercent?.let { "${format1(it)}%" } ?: "--", "Live processor usage")
        MetricTableRow("RAM", state.ramPercent?.let { "${format1(it)}%" } ?: "--", "Memory pressure")
        MetricTableRow("Load", state.load1m?.let { format2(it) } ?: "--", "1 minute average")
        MetricTableRow("Temperature", state.cpuTempC?.let { "${format1(it)}C" } ?: "--", "Thermal reading")
    }
}

@Composable
private fun MetricTableRow(name: String, value: String, detail: String) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.44f),
        shape = MaterialTheme.shapes.small
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 12.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(name, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                Text(detail, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Text(value, style = MaterialTheme.typography.bodyMedium)
        }
    }
}

@Composable
private fun StatusPill(label: String, value: String, positive: Boolean) {
    val background = if (positive) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant
    val textColor = if (positive) MaterialTheme.colorScheme.onPrimaryContainer else MaterialTheme.colorScheme.onSurface
    Surface(color = background, shape = MaterialTheme.shapes.small) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
            Text(label, style = MaterialTheme.typography.labelSmall, color = textColor.copy(alpha = 0.8f))
            Text(value, style = MaterialTheme.typography.bodySmall, color = textColor)
        }
    }
}

@Composable
private fun ErrorStrip(message: String) {
    Surface(
        color = MaterialTheme.colorScheme.errorContainer,
        shape = MaterialTheme.shapes.medium
    ) {
        Text(
            text = message,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            color = MaterialTheme.colorScheme.onErrorContainer,
            style = MaterialTheme.typography.bodyMedium
        )
    }
}

private fun format1(value: Float): String = String.format("%.1f", value)

private fun format2(value: Float): String = String.format("%.2f", value)
