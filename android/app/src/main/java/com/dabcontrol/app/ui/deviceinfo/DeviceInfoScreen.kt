package com.dabcontrol.app.ui.deviceinfo

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.dabcontrol.app.ui.common.PremiumBackdrop
import com.dabcontrol.app.ui.common.PremiumPanel
import com.dabcontrol.app.ui.common.SectionLabel

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun DeviceInfoScreen(
    modifier: Modifier = Modifier,
    viewModel: DeviceInfoViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val structuredRows = state.rows.count { it.isStructured }
    val simpleRows = state.rows.size - structuredRows

    PremiumBackdrop(modifier = modifier) {
        LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding = androidx.compose.foundation.layout.PaddingValues(
                start = 16.dp,
                end = 16.dp,
                top = 20.dp,
                bottom = 28.dp
            ),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            item {
                SectionLabel(
                    eyebrow = "Device Info",
                    title = "Live Device Inspector",
                    subtitle = "A clean table view of the live device metadata returned by the backend, with readable handling for structured values."
                )
            }
            item {
                PremiumPanel(
                    modifier = Modifier.fillMaxWidth(),
                    accentColor = MaterialTheme.colorScheme.primary
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(
                            modifier = Modifier.weight(1f),
                            verticalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            Text("Selected Device", style = MaterialTheme.typography.titleLarge)
                            Text(
                                state.selectedDeviceName.ifBlank { state.selectedDeviceId.ifBlank { "No device selected" } },
                                style = MaterialTheme.typography.bodyLarge,
                                color = MaterialTheme.colorScheme.onSurface
                            )
                            if (state.selectedDeviceId.isNotBlank()) {
                                Text(
                                    "DAB ${state.selectedDeviceId} · YTS ${state.selectedYtsDeviceId.ifBlank { "--" }} · IR ${state.selectedIrDeviceId.ifBlank { "--" }}",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                            Text(
                                if (state.deviceIds.isEmpty()) {
                                    "Waiting for available devices from the backend."
                                } else {
                                    "Using the shared selection from the existing app state."
                                },
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                        OutlinedButton(onClick = viewModel::refresh) {
                            Text("Refresh")
                        }
                    }
                    if (state.deviceIds.isNotEmpty()) {
                        FlowRow(
                            modifier = Modifier.padding(top = 16.dp),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            DeviceStatChip("Devices", state.deviceIds.size.toString())
                            DeviceStatChip("Fields", state.rows.size.toString())
                            DeviceStatChip("Simple", simpleRows.toString())
                            DeviceStatChip("Structured", structuredRows.toString())
                        }
                        Text(
                            text = "Available devices: ${state.deviceIds.joinToString()}",
                            modifier = Modifier.padding(top = 12.dp),
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
            item {
                PremiumPanel(
                    modifier = Modifier.fillMaxWidth(),
                    accentColor = MaterialTheme.colorScheme.tertiary
                ) {
                    Text("Device Info Table", style = MaterialTheme.typography.titleLarge)
                    Text(
                        "Every available field is rendered as readable rows. Objects and arrays stay formatted in-place instead of being dumped as raw text.",
                        modifier = Modifier.padding(top = 4.dp, bottom = 14.dp),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    when {
                        state.isLoading -> {
                            InfoStatePanel("Loading device info...") {
                                CircularProgressIndicator(strokeWidth = 2.6.dp)
                            }
                        }
                        state.selectedDeviceId.isBlank() -> {
                            InfoStatePanel("Please select a device to view device info.")
                        }
                        state.error != null -> {
                            InfoStatePanel(
                                state.error.orEmpty(),
                                emphasisColor = MaterialTheme.colorScheme.error
                            )
                        }
                        state.rows.isEmpty() -> {
                            InfoStatePanel("No device info available.")
                        }
                        else -> {
                            TableHeader()
                            Column(verticalArrangement = Arrangement.spacedBy(0.dp)) {
                                state.rows.forEach { row ->
                                DeviceInfoTableRow(row)
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
private fun DeviceStatChip(label: String, value: String) {
    Surface(
        color = MaterialTheme.colorScheme.primary.copy(alpha = 0.1f),
        shape = MaterialTheme.shapes.small
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
            Text(
                text = label.uppercase(),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary
            )
            Text(
                text = value,
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurface
            )
        }
    }
}

@Composable
private fun InfoStatePanel(
    message: String,
    emphasisColor: androidx.compose.ui.graphics.Color = MaterialTheme.colorScheme.onSurfaceVariant,
    leading: @Composable (() -> Unit)? = null
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.medium,
        shadowElevation = 2.dp
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            if (leading != null) {
                leading()
            }
            Text(
                text = message,
                style = MaterialTheme.typography.bodyMedium,
                color = emphasisColor
            )
        }
    }
}

@Composable
private fun TableHeader() {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant,
        shape = MaterialTheme.shapes.medium,
        shadowElevation = 1.dp
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text(
                text = "Field",
                modifier = Modifier.weight(0.8f),
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = "Value",
                modifier = Modifier.weight(1.2f),
                style = MaterialTheme.typography.labelLarge,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

@Composable
private fun DeviceInfoTableRow(row: DeviceInfoRow) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 10.dp),
        color = MaterialTheme.colorScheme.surface,
        shape = MaterialTheme.shapes.medium,
        shadowElevation = 2.dp
    ) {
        Column(modifier = Modifier.padding(horizontal = 14.dp, vertical = 14.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.Top
            ) {
                Text(
                    text = row.field,
                    modifier = Modifier.weight(0.8f),
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 3,
                    overflow = TextOverflow.Ellipsis
                )
                if (row.isStructured) {
                    Surface(
                        modifier = Modifier.weight(1.2f),
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        shape = MaterialTheme.shapes.small,
                        tonalElevation = 1.dp
                    ) {
                        Text(
                            text = row.value,
                            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
                            style = MaterialTheme.typography.bodySmall,
                            fontFamily = FontFamily.Monospace,
                            color = MaterialTheme.colorScheme.onSurface,
                            softWrap = true
                        )
                    }
                } else {
                    Text(
                        text = row.value,
                        modifier = Modifier.weight(1.2f),
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
            HorizontalDivider(
                modifier = Modifier.padding(top = 12.dp),
                color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.55f)
            )
        }
    }
}
