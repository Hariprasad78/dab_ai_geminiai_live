package com.dabcontrol.app.ui.yts

import androidx.compose.foundation.clickable
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
fun YtsListScreen(
    onOpenCommand: (String) -> Unit,
    modifier: Modifier = Modifier,
    viewModel: YtsListViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    Column(
        modifier = modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Button(onClick = viewModel::refresh) { Text("Refresh Jobs") }
        if (state.isLoading) CircularProgressIndicator()
        state.error?.let { Text("Error: $it") }

        LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(state.items, key = { it.command_id }) { item ->
                Card(
                    modifier = Modifier.fillMaxWidth().clickable { onOpenCommand(item.command_id) }
                ) {
                    Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text("Command: ${item.command_id}")
                        Text("Status: ${item.status}")
                        Text("Updated: ${item.updated_at ?: "--"}")
                        Text(item.command ?: "--")
                    }
                }
            }
        }
    }
}
