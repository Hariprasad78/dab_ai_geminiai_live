package com.dabcontrol.app.ui.runs

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun RunDetailScreen(
    modifier: Modifier = Modifier,
    viewModel: RunDetailViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val status = state.status

    LazyColumn(
        modifier = modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        item {
            Button(onClick = viewModel::refreshAll) { Text("Refresh Detail") }
        }
        item {
            if (state.isLoading) CircularProgressIndicator()
            state.error?.let { Text("Error: $it") }
        }
        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("Run: ${status?.run_id ?: "--"}")
                    Text("Status: ${status?.status ?: "--"}")
                    Text("Goal: ${status?.goal ?: "--"}")
                    Text("Steps: ${status?.step_count ?: 0}")
                    Text("Retries: ${status?.retries ?: 0}")
                    Text("App: ${status?.current_app ?: "--"}")
                    Text("Screen: ${status?.current_screen ?: "--"}")
                    Text("AI Logs: ${status?.ai_log_count ?: 0} | DAB Logs: ${status?.dab_log_count ?: 0}")
                }
            }
        }
        item {
            Text("Action History")
        }
        items(state.actions) { action ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text("#${action.step} ${action.action} (${action.result})")
                    Text(action.reason)
                }
            }
        }
        item { Text("Explain Timeline") }
        items(state.explainTimeline) { step ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(10.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text("Step ${step.step}: ${step.title}")
                    Text("Action: ${step.simple_action}")
                    Text("What happened: ${step.what_happened}")
                    Text("Result: ${step.result} (${step.simple_status})")
                }
            }
        }
        item {
            Text("Diagnosis: ${state.diagnosisSummary ?: "--"}")
        }
        item {
            Text("Narration (${state.narrationEvents.size})")
        }
        items(state.narrationEvents) { ev ->
            Text("${ev.step}. [${ev.tts_category}] ${ev.tts_text}")
        }
        item { Text("AI Transcript Events: ${state.aiEvents.size}") }
        item { Text("DAB Transcript Events: ${state.dabEvents.size}") }
    }
}
