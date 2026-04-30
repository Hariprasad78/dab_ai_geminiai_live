package com.dabcontrol.app.ui.controls

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
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
fun ControlsScreen(
    modifier: Modifier = Modifier,
    viewModel: ControlsViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()

    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = { viewModel.refreshAll(force = true) }) { Text("Refresh All") }
            if (state.isLoading) CircularProgressIndicator()
        }
        Text(state.refreshStatus)
        state.error?.let { Text("Error: $it") }

        OutlinedTextField(
            value = state.selectedDeviceId,
            onValueChange = viewModel::onDeviceSelected,
            label = { Text("Selected Device ID") },
            modifier = Modifier.fillMaxWidth()
        )
        if (state.deviceIds.isNotEmpty()) {
            Row(
                modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                state.deviceIds.forEach { id ->
                    Button(onClick = { viewModel.onDeviceSelected(id) }) { Text(id) }
                }
            }
        }

        Text("Single Action")
        OutlinedTextField(
            value = state.actionName,
            onValueChange = viewModel::onActionChanged,
            label = { Text("Action (e.g. PRESS_HOME, GET_STATE)") },
            modifier = Modifier.fillMaxWidth()
        )
        OutlinedTextField(
            value = state.actionParamsJson,
            onValueChange = viewModel::onActionParamsChanged,
            label = { Text("Action Params JSON") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 3
        )
        Button(onClick = viewModel::executeAction) { Text("Execute Action") }
        Text("Last Action Result")
        Text(state.lastActionResult)

        Text("Batch Actions")
        OutlinedTextField(
            value = state.batchActionsJson,
            onValueChange = viewModel::onBatchActionsChanged,
            label = { Text("Batch JSON Array") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 5
        )
        Button(onClick = viewModel::executeBatch) { Text("Execute Batch") }
        Text("Last Batch Result")
        Text(state.lastBatchResult)

        Text("Device Info")
        Text(state.deviceInfoPreview)
        Text("Capability Status")
        Text(state.capabilityPreview)
        Text("Operations Grid")
        Text(state.operationsPreview)
        Text("Current Settings")
        Text(state.currentSettingsPreview)

        Text("IR Controls")
        OutlinedTextField(
            value = state.irDeviceId,
            onValueChange = viewModel::onIrDeviceChanged,
            label = { Text("IR Device ID") },
            modifier = Modifier.fillMaxWidth()
        )
        OutlinedTextField(
            value = state.irKeyName,
            onValueChange = viewModel::onIrKeyChanged,
            label = { Text("IR Key Name") },
            modifier = Modifier.fillMaxWidth()
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = viewModel::fetchIrKeys) { Text("Load IR Keys") }
            Button(onClick = viewModel::irSend) { Text("IR Send") }
            Button(onClick = viewModel::irTrain) { Text("IR Train") }
        }
        Text("IR Status")
        Text(state.irStatusPreview)
        Text("IR Devices")
        Text(state.irDevicesPreview)
        Text("IR Keys")
        Text(state.irKeysPreview)
        Text("IR Last Result")
        Text(state.irLastResult)

        Text("Task Macro")
        OutlinedTextField(
            value = state.macroInstruction,
            onValueChange = viewModel::onMacroInstructionChanged,
            label = { Text("Instruction") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 2
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Checkbox(checked = state.macroExecute, onCheckedChange = { viewModel.toggleMacroExecute() })
            Text("Execute Now")
        }
        Button(onClick = viewModel::runMacro) { Text("Run Macro") }
        Text("Macro Result")
        Text(state.macroResult)

        Text("Planner Debug")
        OutlinedTextField(
            value = state.plannerGoal,
            onValueChange = viewModel::onPlannerGoalChanged,
            label = { Text("Goal") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 2
        )
        OutlinedTextField(
            value = state.plannerCurrentApp,
            onValueChange = viewModel::onPlannerAppChanged,
            label = { Text("Current App (optional)") },
            modifier = Modifier.fillMaxWidth()
        )
        OutlinedTextField(
            value = state.plannerCurrentScreen,
            onValueChange = viewModel::onPlannerScreenChanged,
            label = { Text("Current Screen (optional)") },
            modifier = Modifier.fillMaxWidth()
        )
        OutlinedTextField(
            value = state.plannerOcrText,
            onValueChange = viewModel::onPlannerOcrChanged,
            label = { Text("OCR Text (optional)") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 3
        )
        Button(onClick = viewModel::runPlannerDebug) { Text("Run Planner Debug") }
        Text("Planner Result")
        Text(state.plannerResult)
    }
}
