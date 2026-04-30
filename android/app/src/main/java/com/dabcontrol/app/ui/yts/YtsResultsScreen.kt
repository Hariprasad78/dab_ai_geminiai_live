package com.dabcontrol.app.ui.yts

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun YtsResultsScreen(
    onOpenCommand: (String) -> Unit,
    onOpenReport: (String) -> Unit,
    onOpenArtifact: (String, String) -> Unit,
    modifier: Modifier = Modifier,
    viewModel: YtsListViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(start = 16.dp, end = 16.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("YTS Results", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
        Text(
            "Open completed sessions, inspect command output, and jump straight to generated reports.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        FilledTonalButton(onClick = viewModel::refresh) {
            Text("Refresh Results")
        }
        if (state.isLoading) CircularProgressIndicator()
        state.error?.let { Text("Error: $it") }

        LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            items(state.items, key = { it.command_id }) { item ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onOpenCommand(item.command_id) }
                ) {
                    Column(
                        modifier = Modifier.padding(14.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text(item.command_id, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Medium)
                        Text("Status: ${item.status}")
                        Text("Updated: ${item.updated_at ?: "--"}")
                        Text(item.command ?: "--", style = MaterialTheme.typography.bodySmall)
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            Button(onClick = { onOpenCommand(item.command_id) }) {
                                Text("Open Session")
                            }
                            if (!item.result_file_name.isNullOrBlank()) {
                                OutlinedButton(onClick = { onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_RESULT) }) {
                                    Text("View Result")
                                }
                            }
                            if (!item.report_html_name.isNullOrBlank() || !item.report_pdf_name.isNullOrBlank()) {
                                FilledTonalButton(onClick = { onOpenReport(item.command_id) }) {
                                    Text("Open Report")
                                }
                            }
                            if (!item.report_pdf_name.isNullOrBlank()) {
                                OutlinedButton(onClick = { onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_REPORT_PDF) }) {
                                    Text("PDF")
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
