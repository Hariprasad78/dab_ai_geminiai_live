package com.dabcontrol.app.ui.yts

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun YtsDetailScreen(
    onOpenReport: (String) -> Unit,
    modifier: Modifier = Modifier,
    viewModel: YtsDetailViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val data = state.data

    LazyColumn(
        modifier = modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        item {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = viewModel::refresh) { Text("Refresh") }
                Button(onClick = viewModel::stopCommand) { Text("Stop") }
                if (data?.command_id != null) {
                    Button(onClick = { onOpenReport(data.command_id) }) { Text("Report") }
                }
            }
        }
        item {
            if (state.isLoading) CircularProgressIndicator()
            state.error?.let { Text("Error: $it") }
        }
        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Command ID: ${data?.command_id ?: "--"}")
                    Text("Status: ${data?.status ?: "--"}")
                    Text("Updated: ${data?.updated_at ?: "--"}")
                    Text("Exit code: ${data?.returncode?.toString() ?: "--"}")
                    Text("Command: ${data?.command ?: "--"}")
                }
            }
        }
        item {
            val promptText = data?.pending_prompt?.get("text")?.toString().orEmpty().trim('"')
            if (data?.awaiting_input == true || promptText.isNotBlank()) {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text("Interactive Prompt")
                        Text(if (promptText.isNotBlank()) promptText else "(Waiting for prompt text)")
                        OutlinedTextField(
                            modifier = Modifier.fillMaxWidth(),
                            value = state.promptInput,
                            onValueChange = viewModel::onPromptInputChange,
                            label = { Text("Response") }
                        )
                        Button(onClick = viewModel::sendPromptResponse) { Text("Send Response") }
                    }
                }
            }
        }
        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("Live Log")
                    Text(data?.logs?.toString() ?: "(no logs)")
                }
            }
        }
        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("STDOUT")
                    Text(data?.stdout ?: "(empty)")
                    Text("STDERR")
                    Text(data?.stderr ?: "(empty)")
                }
            }
        }
    }
}
