package com.dabcontrol.app.ui.runs

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
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun RunsListScreen(
    onOpenRun: (String) -> Unit,
    modifier: Modifier = Modifier,
    viewModel: RunsListViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()

    Column(
        modifier = modifier.fillMaxSize().padding(start = 16.dp, end = 16.dp, bottom = 16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("Create AI Runner Job")
                Text("Enter a natural-language instruction. Gemini converts it into validated executable actions on the selected device.")
                OutlinedTextField(
                    value = state.aiInstruction,
                    onValueChange = viewModel::onAiInstructionChanged,
                    label = { Text("AI instruction") },
                    placeholder = { Text("Set time zone to India Kolkata") },
                    modifier = Modifier.fillMaxWidth(),
                    minLines = 3
                )
                OutlinedTextField(
                    value = state.selectedDeviceId,
                    onValueChange = viewModel::onDeviceSelected,
                    label = { Text("Target device") },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true
                )
                if (state.deviceIds.isNotEmpty()) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        state.deviceIds.take(4).forEach { id ->
                            OutlinedButton(onClick = { viewModel.onDeviceSelected(id) }) {
                                Text(id)
                            }
                        }
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = viewModel::createAiRunnerJob,
                        enabled = !state.isSubmittingAiJob
                    ) {
                        Text(if (state.isSubmittingAiJob) "Running..." else "Run AI Job")
                    }
                    Button(onClick = viewModel::refresh, enabled = !state.isLoading) {
                        Text("Refresh")
                    }
                    if (state.isSubmittingAiJob) CircularProgressIndicator()
                }
                Text("Job ID: ${state.aiJobId ?: "--"}")
                Text("Status: ${state.aiJobStatus}")
                Text("Result: ${state.aiJobResult}")
                Text("Logs:\n${state.aiJobLogs}")
            }
        }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = viewModel::refresh) { Text("Refresh Runs") }
        }
        if (state.isLoading) CircularProgressIndicator()
        state.error?.let { Text("Error: $it") }

        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(state.items, key = { it.run_id }) { item ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onOpenRun(item.run_id) }
                ) {
                    Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("Run: ${item.run_id}")
                        Text("Status: ${item.status}")
                        Text("Goal: ${item.goal}")
                        Text("Steps: ${item.step_count}")
                    }
                }
            }
        }
    }
}
