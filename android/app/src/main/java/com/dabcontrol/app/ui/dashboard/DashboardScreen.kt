package com.dabcontrol.app.ui.dashboard

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.clickable
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun DashboardScreen(
    modifier: Modifier = Modifier,
    viewModel: DashboardViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()

    Column(
        modifier = modifier
            .verticalScroll(rememberScrollState())
            .padding(start = 16.dp, end = 16.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp)
    ) {
        ElevatedCard(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text("Operations Dashboard", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
                Text(
                    "Configure the backend URL, confirm device health, and watch CPU, memory, load, and temperature trends.",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                UrlPresetPicker(
                    currentUrl = state.apiBaseUrl,
                    onSelected = viewModel::onApiBaseUrlChanged
                )
                OutlinedTextField(
                    value = state.apiBaseUrl,
                    onValueChange = viewModel::onApiBaseUrlChanged,
                    label = { Text("API Base URL") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )
                DevicePickerCard(
                    deviceIds = state.deviceIds,
                    selectedDeviceId = state.selectedDeviceId,
                    onSelect = viewModel::onDeviceSelected
                )
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    FilledTonalButton(onClick = viewModel::saveApiBaseUrl) {
                        Text("Save URL")
                    }
                    OutlinedButton(onClick = viewModel::refresh) {
                        Text("Refresh")
                    }
                    if (state.isLoading) {
                        CircularProgressIndicator(modifier = Modifier.height(20.dp))
                    }
                }
                state.error?.let { ErrorStrip(it) }
            }
        }

        ElevatedCard(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text("Gemini Runtime Models", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
                Text(state.modelStatus, color = MaterialTheme.colorScheme.onSurfaceVariant)
                ModelPicker(
                    title = "Planner model",
                    currentValue = state.plannerModel,
                    availableModels = state.availableModels,
                    onValueChanged = viewModel::onPlannerModelChanged
                )
                ModelPicker(
                    title = "Live/visual model",
                    currentValue = state.liveModel,
                    availableModels = state.availableModels,
                    onValueChanged = viewModel::onLiveModelChanged
                )
                FilledTonalButton(onClick = viewModel::applyRuntimeModels) {
                    Text("Apply Gemini Models")
                }
            }
        }

        SummaryCard(
            healthStatus = state.healthStatus,
            mode = state.mode,
            cpuCount = state.cpuCount,
            timestamp = state.timestamp,
            selectedDeviceId = state.selectedDeviceId
        )

        MetricCardRow(
            title = "CPU",
            value = state.cpuPercent?.let { "${format1(it)}%" } ?: "--",
            subtitle = "Processor usage",
            accent = Color(0xFF0F766E),
            history = state.cpuHistory,
            maxValue = 100f
        )
        MetricCardRow(
            title = "Memory",
            value = state.ramPercent?.let { "${format1(it)}%" } ?: "--",
            subtitle = "RAM consumption",
            accent = Color(0xFF2563EB),
            history = state.ramHistory,
            maxValue = 100f
        )
        MetricCardRow(
            title = "System Load",
            value = state.load1m?.let { format2(it) } ?: "--",
            subtitle = "1 minute average",
            accent = Color(0xFFD97706),
            history = state.loadHistory,
            maxValue = (state.loadHistory.maxOfOrNull { it.value } ?: 1f).coerceAtLeast(1f)
        )
        MetricCardRow(
            title = "CPU Temperature",
            value = state.cpuTempC?.let { "${format1(it)}C" } ?: "--",
            subtitle = "Thermal signal",
            accent = Color(0xFFDC2626),
            history = state.tempHistory,
            maxValue = (state.tempHistory.maxOfOrNull { it.value } ?: 100f).coerceAtLeast(50f)
        )

        ElevatedCard(modifier = Modifier.fillMaxWidth()) {
            Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                Text("Metrics Snapshot", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                Text(
                    state.metricsPreview,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                HorizontalDivider()
                Surface(
                    tonalElevation = 2.dp,
                    shape = MaterialTheme.shapes.medium
                ) {
                    SelectionContainer {
                        Text(
                            text = buildRawSummary(state),
                            modifier = Modifier.padding(12.dp),
                            style = MaterialTheme.typography.bodySmall,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }
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
        "Local" to "http://10.99.57.66:8081",
        "Public" to "https://creative-airline-maintaining-manufacturers.trycloudflare.com"
    )
    var expanded by remember(currentUrl) { mutableStateOf(false) }
    val selectedLabel = presets.firstOrNull { it.second == currentUrl }?.first ?: "Custom"

    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = !expanded }
    ) {
        OutlinedTextField(
            value = selectedLabel,
            onValueChange = {},
            readOnly = true,
            label = { Text("URL Preset") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .menuAnchor()
                .fillMaxWidth()
        )
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
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

@Composable
private fun SummaryCard(
    healthStatus: String,
    mode: String,
    cpuCount: Int?,
    timestamp: String,
    selectedDeviceId: String
) {
    ElevatedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text("Backend Health", style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text("Status: $healthStatus")
            Text("Mode: $mode")
            Text("Selected device: ${selectedDeviceId.ifBlank { "--" }}")
            Text("CPU cores: ${cpuCount ?: "--"}")
            Text("Last sample: $timestamp")
        }
    }
}

@Composable
private fun DevicePickerCard(
    deviceIds: List<String>,
    selectedDeviceId: String,
    onSelect: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("Shared Device", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
        Text(
            "The selected device here is reused by YTS, live control, and DAB operations.",
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        if (deviceIds.isEmpty()) {
            Text("No devices loaded yet.", color = MaterialTheme.colorScheme.onSurfaceVariant)
        } else {
            deviceIds.forEach { deviceId ->
                Surface(
                    tonalElevation = if (deviceId == selectedDeviceId) 4.dp else 1.dp,
                    shape = MaterialTheme.shapes.small,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onSelect(deviceId) }
                ) {
                    Text(
                        text = deviceId,
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(10.dp),
                        color = if (deviceId == selectedDeviceId) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        }
    }
}

@Composable
private fun ModelPicker(
    title: String,
    currentValue: String,
    availableModels: List<String>,
    onValueChanged: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        OutlinedTextField(
            value = currentValue,
            onValueChange = onValueChanged,
            label = { Text(title) },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )
        availableModels.take(8).forEach { model ->
            Surface(
                tonalElevation = if (model == currentValue) 4.dp else 1.dp,
                shape = MaterialTheme.shapes.small,
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { onValueChanged(model) }
            ) {
                Text(
                    text = model,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(10.dp),
                    color = if (model == currentValue) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                )
            }
        }
    }
}

@Composable
private fun MetricCardRow(
    title: String,
    value: String,
    subtitle: String,
    accent: Color,
    history: List<MetricPoint>,
    maxValue: Float
) {
    ElevatedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text(value, style = MaterialTheme.typography.headlineMedium, color = accent)
            Text(subtitle, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            SparklineChart(
                points = history,
                accent = accent,
                maxValue = maxValue
            )
            if (history.isNotEmpty()) {
                Text(
                    "Recent: ${history.takeLast(4).joinToString("  ") { "${it.label} ${format1(it.value)}" }}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun SparklineChart(
    points: List<MetricPoint>,
    accent: Color,
    maxValue: Float
) {
    val outlineColor = MaterialTheme.colorScheme.outlineVariant
    Surface(
        modifier = Modifier.fillMaxWidth(),
        tonalElevation = 2.dp,
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
                    .height(140.dp)
                    .padding(12.dp)
            ) {
                val usableMax = maxValue.coerceAtLeast(1f)
                val stepX = size.width / (points.size - 1).coerceAtLeast(1)
                val baseline = size.height

                drawLine(
                    color = outlineColor,
                    start = Offset(0f, baseline),
                    end = Offset(size.width, baseline),
                    strokeWidth = 2f
                )

                for (index in 0 until points.lastIndex) {
                    val first = points[index]
                    val second = points[index + 1]
                    val firstOffset = Offset(
                        x = index * stepX,
                        y = baseline - ((first.value / usableMax).coerceIn(0f, 1f) * size.height)
                    )
                    val secondOffset = Offset(
                        x = (index + 1) * stepX,
                        y = baseline - ((second.value / usableMax).coerceIn(0f, 1f) * size.height)
                    )
                    drawLine(
                        color = accent,
                        start = firstOffset,
                        end = secondOffset,
                        strokeWidth = 6f,
                        cap = StrokeCap.Round
                    )
                }

                points.forEachIndexed { index, point ->
                    drawCircle(
                        color = accent,
                        radius = 6f,
                        center = Offset(
                            x = index * stepX,
                            y = baseline - ((point.value / usableMax).coerceIn(0f, 1f) * size.height)
                        )
                    )
                }

                drawRect(
                    color = accent.copy(alpha = 0.08f),
                    style = Stroke(width = 0f)
                )
            }
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

private fun buildRawSummary(state: DashboardUiState): String {
    return listOf(
        "health=${state.healthStatus}",
        "mode=${state.mode}",
        "timestamp=${state.timestamp}",
        "cpu_percent=${state.cpuPercent?.let(::format1) ?: "--"}",
        "ram_percent=${state.ramPercent?.let(::format1) ?: "--"}",
        "load_1m=${state.load1m?.let(::format2) ?: "--"}",
        "cpu_temp_c=${state.cpuTempC?.let(::format1) ?: "--"}",
        "cpu_count=${state.cpuCount ?: "--"}"
    ).joinToString("\n")
}

private fun format1(value: Float): String = String.format("%.1f", value)

private fun format2(value: Float): String = String.format("%.2f", value)
