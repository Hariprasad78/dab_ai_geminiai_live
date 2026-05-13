package com.dabcontrol.app.ui.yts

import android.graphics.BitmapFactory
import android.content.res.Configuration
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ElevatedCard
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalConfiguration
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.dabcontrol.app.data.api.YtsLiveCommandStateDto
import com.dabcontrol.app.data.api.YtsLiveCommandSummaryDto
import com.dabcontrol.app.data.api.YtsTestCatalogItemDto
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject

@Composable
fun YtsListScreen(
    onOpenCommand: (String) -> Unit,
    onOpenReport: (String) -> Unit,
    onOpenArtifact: (String, String) -> Unit,
    modifier: Modifier = Modifier,
    viewModel: YtsListViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val configuration = LocalConfiguration.current
    val isLandscape = configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
    var showCommandCenter by rememberSaveable(isLandscape) { mutableStateOf(!isLandscape) }
    var runnerSplit by rememberSaveable { mutableStateOf(0.56f) }
    var floatingStreamHidden by rememberSaveable { mutableStateOf(false) }
    val createListState = rememberLazyListState()
    val runnerListState = rememberLazyListState()
    val resultsListState = rememberLazyListState()
    val autoShowHero by remember(state.activeTab, createListState, runnerListState, resultsListState) {
        derivedStateOf {
            val currentState = when (state.activeTab) {
                YtsWorkspaceTab.CREATE_JOB -> createListState
                YtsWorkspaceTab.RUNNING_SESSION -> runnerListState
                YtsWorkspaceTab.PAST_RESULTS -> resultsListState
            }
            currentState.firstVisibleItemIndex == 0 && currentState.firstVisibleItemScrollOffset < 20
        }
    }
    val suites = state.catalog.mapNotNull { it.test_suite }.distinct().sorted()
    val categories = state.catalog
        .filter { state.suiteFilter.isBlank() || it.test_suite == state.suiteFilter }
        .mapNotNull { it.test_category }
        .distinct()
        .sorted()

    Box(modifier = modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(start = 16.dp, end = 16.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (state.activeTab != YtsWorkspaceTab.RUNNING_SESSION && autoShowHero) {
                HeroHeader(
                    state = state,
                    compact = isLandscape,
                    expanded = true
                )
            }

            WorkspaceTabs(
                activeTab = state.activeTab,
                onSelect = viewModel::selectTab
            )

            Box(modifier = Modifier.fillMaxWidth().weight(1f, fill = true)) {
                when (state.activeTab) {
                    YtsWorkspaceTab.CREATE_JOB -> CreateJobWindow(
                        state = state,
                        suites = suites,
                        categories = categories,
                        onDeviceIdChanged = viewModel::onDeviceIdChanged,
                        onJsonOutputChanged = viewModel::onJsonOutputChanged,
                        onGuidedModeToggle = viewModel::toggleGuidedMode,
                        onInteractiveAiToggle = viewModel::toggleInteractiveAi,
                        onRecordVideoToggle = viewModel::toggleRecordVideo,
                        onRecordAudioToggle = viewModel::toggleRecordAudio,
                        onFilterTokensChanged = viewModel::onFilterTokensChanged,
                        onExtraArgsChanged = viewModel::onExtraArgsChanged,
                        onRefreshSessions = viewModel::refresh,
                        onLoadCatalog = { refresh -> viewModel.loadCatalog(refresh) },
                        onStop = viewModel::stopActiveCommand,
                        listState = createListState,
                        compactControls = isLandscape,
                        showCommandCenter = showCommandCenter,
                        onToggleCommandCenter = { showCommandCenter = !showCommandCenter },
                        onSuiteFilterChanged = viewModel::onSuiteFilterChanged,
                        onCategoryFilterChanged = viewModel::onCategoryFilterChanged,
                        onSearchQueryChanged = viewModel::onSearchQueryChanged,
                        onToggleTest = viewModel::toggleTestSelection,
                        onClearTests = viewModel::clearSelectedTests,
                        onStart = viewModel::startGuidedTestRun
                    )
                    YtsWorkspaceTab.RUNNING_SESSION -> RunningSessionWindow(
                        state = state,
                        onRefresh = viewModel::refresh,
                        onRetryStream = viewModel::retryStream,
                        onStop = viewModel::stopActiveCommand,
                        onSendRemoteAction = viewModel::sendRemoteAction,
                        onPromptChanged = viewModel::onPromptInputChanged,
                        onSendPrompt = viewModel::sendPromptResponse,
                        onSuggestDraft = { viewModel.suggestPromptResponse(sendResponse = false) },
                        onSuggestAutoReply = { viewModel.suggestPromptResponse(sendResponse = true) },
                        listState = runnerListState,
                        runnerSplit = runnerSplit,
                        onRunnerSplitChanged = { runnerSplit = it },
                        onOpenDetail = { state.activeCommandId?.let(onOpenCommand) },
                        onSelectCommand = viewModel::selectCommand
                    )
                    YtsWorkspaceTab.PAST_RESULTS -> ResultsWindow(
                        state = state,
                        onRefresh = viewModel::refresh,
                        onMonitor = viewModel::selectCommand,
                        onStop = viewModel::stopActiveCommand,
                        listState = resultsListState,
                        compactControls = isLandscape,
                        showCommandCenter = showCommandCenter,
                        onToggleCommandCenter = { showCommandCenter = !showCommandCenter },
                        onOpenDetail = onOpenCommand,
                        onOpenReport = onOpenReport,
                        onOpenArtifact = onOpenArtifact
                    )
                }
            }
        }

        val shouldOfferFloatingStream =
            state.activeTab == YtsWorkspaceTab.RUNNING_SESSION &&
            state.isStreaming &&
            (runnerListState.firstVisibleItemIndex > 0 || runnerListState.firstVisibleItemScrollOffset > 420)

        if (!state.isStreaming || state.activeTab != YtsWorkspaceTab.RUNNING_SESSION) {
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
            FloatingYtsStreamOverlay(
                frameBytes = state.streamFrameBytes,
                streamStatus = state.streamStatus,
                onHide = { floatingStreamHidden = true },
                modifier = Modifier.align(Alignment.BottomEnd)
            )
        }
    }
}

@Composable
private fun WorkspaceTabs(
    activeTab: YtsWorkspaceTab,
    onSelect: (YtsWorkspaceTab) -> Unit
) {
    val tabs = listOf(
        YtsWorkspaceTab.CREATE_JOB to "Create Job",
        YtsWorkspaceTab.RUNNING_SESSION to "Test Runner",
        YtsWorkspaceTab.PAST_RESULTS to "Past Results"
    )
    ScrollableTabRow(
        selectedTabIndex = tabs.indexOfFirst { it.first == activeTab },
        containerColor = Color(0xFFF4EDE3),
        contentColor = Color(0xFF111827),
        edgePadding = 4.dp,
        divider = {}
    ) {
        tabs.forEach { (tab, label) ->
            Tab(
                selected = activeTab == tab,
                onClick = { onSelect(tab) },
                modifier = Modifier
                    .defaultMinSize(minHeight = 36.dp)
                    .padding(horizontal = 2.dp),
                text = {
                    Text(
                        label,
                        style = MaterialTheme.typography.labelLarge,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        textAlign = TextAlign.Center
                    )
                }
            )
        }
    }
}

@Composable
private fun CreateJobWindow(
    state: YtsListUiState,
    suites: List<String>,
    categories: List<String>,
    onDeviceIdChanged: (String) -> Unit,
    onJsonOutputChanged: (String) -> Unit,
    onGuidedModeToggle: () -> Unit,
    onInteractiveAiToggle: () -> Unit,
    onRecordVideoToggle: () -> Unit,
    onRecordAudioToggle: () -> Unit,
    onFilterTokensChanged: (String) -> Unit,
    onExtraArgsChanged: (String) -> Unit,
    onRefreshSessions: () -> Unit,
    onLoadCatalog: (Boolean) -> Unit,
    onStop: () -> Unit,
    listState: LazyListState,
    compactControls: Boolean,
    showCommandCenter: Boolean,
    onToggleCommandCenter: () -> Unit,
    onSuiteFilterChanged: (String) -> Unit,
    onCategoryFilterChanged: (String) -> Unit,
    onSearchQueryChanged: (String) -> Unit,
    onToggleTest: (String) -> Unit,
    onClearTests: () -> Unit,
    onStart: () -> Unit
) {
    LazyColumn(
        state = listState,
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            YtsControlDeck(
                activeCommandId = state.activeCommandId,
                onRefreshSessions = onRefreshSessions,
                onRefreshCatalog = { onLoadCatalog(true) },
                onStop = onStop,
                compact = compactControls,
                expanded = showCommandCenter,
                onToggle = onToggleCommandCenter
            )
        }
        item {
            if (compactControls) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Box(modifier = Modifier.weight(0.46f)) {
                        JobBlueprintCard(
                            state = state,
                            onDeviceIdChanged = onDeviceIdChanged,
                            onJsonOutputChanged = onJsonOutputChanged,
                            onGuidedModeToggle = onGuidedModeToggle,
                            onInteractiveAiToggle = onInteractiveAiToggle,
                            onRecordVideoToggle = onRecordVideoToggle,
                            onRecordAudioToggle = onRecordAudioToggle,
                            onFilterTokensChanged = onFilterTokensChanged,
                            onExtraArgsChanged = onExtraArgsChanged,
                            onStart = onStart
                        )
                    }
                    Box(modifier = Modifier.weight(0.54f)) {
                        TestCatalogCard(
                            state = state,
                            suites = suites,
                            categories = categories,
                            onLoadCatalog = onLoadCatalog,
                            onSuiteFilterChanged = onSuiteFilterChanged,
                            onCategoryFilterChanged = onCategoryFilterChanged,
                            onSearchQueryChanged = onSearchQueryChanged,
                            onToggleTest = onToggleTest,
                            onClearTests = onClearTests,
                            onStart = onStart
                        )
                    }
                }
            } else {
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    JobBlueprintCard(
                        state = state,
                        onDeviceIdChanged = onDeviceIdChanged,
                        onJsonOutputChanged = onJsonOutputChanged,
                        onGuidedModeToggle = onGuidedModeToggle,
                        onInteractiveAiToggle = onInteractiveAiToggle,
                        onRecordVideoToggle = onRecordVideoToggle,
                        onRecordAudioToggle = onRecordAudioToggle,
                        onFilterTokensChanged = onFilterTokensChanged,
                        onExtraArgsChanged = onExtraArgsChanged,
                        onStart = onStart
                    )
                    TestCatalogCard(
                        state = state,
                        suites = suites,
                        categories = categories,
                        onLoadCatalog = onLoadCatalog,
                        onSuiteFilterChanged = onSuiteFilterChanged,
                        onCategoryFilterChanged = onCategoryFilterChanged,
                        onSearchQueryChanged = onSearchQueryChanged,
                        onToggleTest = onToggleTest,
                        onClearTests = onClearTests,
                        onStart = onStart
                    )
                }
            }
        }
    }
}

@Composable
private fun JobBlueprintCard(
    state: YtsListUiState,
    onDeviceIdChanged: (String) -> Unit,
    onJsonOutputChanged: (String) -> Unit,
    onGuidedModeToggle: () -> Unit,
    onInteractiveAiToggle: () -> Unit,
    onRecordVideoToggle: () -> Unit,
    onRecordAudioToggle: () -> Unit,
    onFilterTokensChanged: (String) -> Unit,
    onExtraArgsChanged: (String) -> Unit,
    onStart: () -> Unit
) {
    SectionCard("Job Blueprint") {
        val hasYtsRoute = state.ytsShortId.isNotBlank() || state.ytsDeviceId.isNotBlank()
        val canStart = !state.isStarting && state.deviceId.isNotBlank() && hasYtsRoute
        Text(state.startStatus, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Surface(
            tonalElevation = 2.dp,
            shape = MaterialTheme.shapes.medium,
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text("Shared device", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                Text(
                    state.deviceDisplayName.ifBlank { state.deviceId.ifBlank { "Select the device from Dashboard or Live Control first." } },
                    color = if (state.deviceId.isBlank()) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary
                )
                Text(
                    "YTS now follows the unified device context from the updated backend mapping.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
        OperatorSummaryStrip(
            entries = listOf(
                "Device" to state.deviceDisplayName.ifBlank { state.deviceId.ifBlank { "--" } },
                "YTS Config" to state.ytsDeviceId.ifBlank { "--" },
                "IR" to state.irDeviceId.ifBlank { "--" }
            )
        )
        if (state.sharedDeviceIssues.isNotEmpty()) {
            state.sharedDeviceIssues.take(2).forEach { issue ->
                StatusLine(
                    label = "Mapping issue",
                    value = issue,
                    tone = MaterialTheme.colorScheme.error
                )
            }
        }
        OutlinedTextField(
            value = state.deviceDisplayName.ifBlank { state.deviceId },
            onValueChange = {},
            label = { Text("Selected Device") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            readOnly = true,
            supportingText = { Text("Select by device name from Dashboard or Live Control. YTS runtime IDs are resolved automatically from that active device context.") }
        )
        OutlinedTextField(
            value = state.jsonOutputFile,
            onValueChange = onJsonOutputChanged,
            label = { Text("Result Bundle Output Path") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            supportingText = {
                Text("Used for structured result export while reports and artifacts stay accessible from the Results workspace.")
            }
        )
        ToggleRow("Guided mode", state.guidedMode, onGuidedModeToggle)
        ToggleRow("Use Gemini for interactive responses", state.interactiveAi, onInteractiveAiToggle)
        ToggleRow("Record TV video", state.recordVideo, onRecordVideoToggle)
        ToggleRow("Record audio with video", state.recordAudio, onRecordAudioToggle)
        OutlinedTextField(
            value = state.filterTokensInput,
            onValueChange = onFilterTokensChanged,
            label = { Text("Filter Tokens") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 2
        )
        OutlinedTextField(
            value = state.extraArgsInput,
            onValueChange = onExtraArgsChanged,
            label = { Text("Additional YTS Args") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 2
        )
        OperatorSummaryStrip(
            entries = listOf(
                "Guided" to if (state.guidedMode) "On" else "Off",
                "Interactive AI" to if (state.interactiveAi) "Enabled" else "Manual",
                "Video" to if (state.recordVideo) "Recording" else "Off",
                "Selected" to state.selectedTestIds.size.toString()
            )
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilledTonalButton(onClick = onStart, enabled = canStart) {
                Text(if (state.isStarting) "Starting..." else "Run Selected Tests")
            }
            OutlinedButton(onClick = onStart, enabled = canStart && state.selectedTestIds.isEmpty()) {
                Text("Run Filtered Set")
            }
        }
    }
}

@Composable
private fun TestCatalogCard(
    state: YtsListUiState,
    suites: List<String>,
    categories: List<String>,
    onLoadCatalog: (Boolean) -> Unit,
    onSuiteFilterChanged: (String) -> Unit,
    onCategoryFilterChanged: (String) -> Unit,
    onSearchQueryChanged: (String) -> Unit,
    onToggleTest: (String) -> Unit,
    onClearTests: () -> Unit,
    onStart: () -> Unit
) {
    SectionCard("Test Catalog") {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilledTonalButton(onClick = { onLoadCatalog(false) }) {
                Text("Load Catalog")
            }
            OutlinedButton(onClick = { onLoadCatalog(true) }) {
                Text("Refresh Catalog")
            }
            if (state.isCatalogLoading) {
                CircularProgressIndicator()
            }
        }
        Text("Loaded ${state.catalog.size} tests. Matching ${state.filteredCatalog.size}.")
        SmartTestSelector(
            tests = state.catalog,
            suiteValue = state.suiteFilter,
            categoryValue = state.categoryFilter,
            titleValue = state.searchQuery,
            onSuiteSelected = onSuiteFilterChanged,
            onCategorySelected = onCategoryFilterChanged,
            onTitleSelected = onSearchQueryChanged,
            selectedTestIds = state.selectedTestIds,
            onToggleTest = onToggleTest
        )
        Surface(
            tonalElevation = 2.dp,
            shape = MaterialTheme.shapes.medium,
            modifier = Modifier.fillMaxWidth()
        ) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                Text("Selection Summary", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
                Text(
                    "Selected tests: ${state.selectedTestIds.size}",
                    color = if (state.selectedTestIds.isNotEmpty()) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    state.selectedTestIds.take(6).joinToString("\n").ifBlank { "Tap tests below to add them to the run." },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = onClearTests) {
                        Text("Clear Selected")
                    }
                    FilledTonalButton(onClick = onStart, enabled = !state.isStarting) {
                        Text("Run Job")
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SmartTestSelector(
    tests: List<YtsTestCatalogItemDto>,
    suiteValue: String,
    categoryValue: String,
    titleValue: String,
    onSuiteSelected: (String) -> Unit,
    onCategorySelected: (String) -> Unit,
    onTitleSelected: (String) -> Unit,
    selectedTestIds: List<String>,
    onToggleTest: (String) -> Unit
) {
    val suiteOptions = tests.mapNotNull { it.test_suite?.takeIf(String::isNotBlank) }.distinct().sorted()
    val suiteScopedTests = tests.filter { suiteValue.isBlank() || it.test_suite == suiteValue }
    val categoryOptions = suiteScopedTests.mapNotNull { it.test_category?.takeIf(String::isNotBlank) }.distinct().sorted()
    val categoryScopedTests = suiteScopedTests.filter { categoryValue.isBlank() || it.test_category == categoryValue }
    val titleOptions = categoryScopedTests.mapNotNull { it.test_title?.takeIf(String::isNotBlank) }.distinct().sorted()
    val visibleTests = categoryScopedTests.filter { titleValue.isBlank() || it.test_title == titleValue }
    val selectedTest = visibleTests.firstOrNull() ?: categoryScopedTests.firstOrNull()
    val selectedTestId = selectedTest?.test_id.orEmpty()

    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text("Test Selector", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.SemiBold)
        Text(
            "Filter the YTS catalog by suite, category, and test title, then add the matching test case.",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        SmartDropdown(
            label = "Test Suite",
            value = suiteValue,
            options = suiteOptions,
            onSelected = {
                onSuiteSelected(it)
                onCategorySelected("")
                onTitleSelected("")
            }
        )
        SmartDropdown(
            label = "Test Category",
            value = categoryValue,
            options = categoryOptions,
            onSelected = {
                onCategorySelected(it)
                onTitleSelected("")
            }
        )
        SmartDropdown(
            label = "Test Title",
            value = titleValue,
            options = titleOptions,
            onSelected = onTitleSelected
        )
        if (selectedTest != null) {
            Surface(
                tonalElevation = 1.dp,
                shape = MaterialTheme.shapes.small,
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(
                    modifier = Modifier.padding(10.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(selectedTest.test_title ?: "--", fontWeight = FontWeight.SemiBold)
                    Text("ID: ${selectedTest.test_id}", style = MaterialTheme.typography.bodySmall)
                    Text("Suite: ${selectedTest.test_suite ?: "--"}", style = MaterialTheme.typography.bodySmall)
                    Text("Category: ${selectedTest.test_category ?: "--"}", style = MaterialTheme.typography.bodySmall)
                }
            }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilledTonalButton(
                onClick = { if (selectedTestId.isNotBlank()) onToggleTest(selectedTestId) },
                enabled = selectedTestId.isNotBlank()
            ) {
                Text(if (selectedTestIds.contains(selectedTestId)) "Remove Test" else "Add Test")
            }
            OutlinedButton(
                onClick = {
                    visibleTests.take(8).forEach { test ->
                        if (!selectedTestIds.contains(test.test_id)) onToggleTest(test.test_id)
                    }
                },
                enabled = visibleTests.isNotEmpty()
            ) {
                Text("Add Matching Tests")
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SmartDropdown(
    label: String,
    value: String,
    options: List<String>,
    onSelected: (String) -> Unit
) {
    var expanded by remember(options, value) { mutableStateOf(false) }
    ExposedDropdownMenuBox(
        expanded = expanded,
        onExpandedChange = { expanded = !expanded }
    ) {
        OutlinedTextField(
            value = value,
            onValueChange = {},
            readOnly = true,
            label = { Text(label) },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier
                .menuAnchor()
                .fillMaxWidth()
        )
        DropdownMenu(
            expanded = expanded,
            onDismissRequest = { expanded = false }
        ) {
            options.forEach { option ->
                DropdownMenuItem(
                    text = { Text(option) },
                    onClick = {
                        onSelected(option)
                        expanded = false
                    }
                )
            }
        }
    }
}

@Composable
private fun RunningSessionWindow(
    state: YtsListUiState,
    onRefresh: () -> Unit,
    onRetryStream: () -> Unit,
    onStop: () -> Unit,
    onSendRemoteAction: (String) -> Unit,
    onPromptChanged: (String) -> Unit,
    onSendPrompt: () -> Unit,
    onSuggestDraft: () -> Unit,
    onSuggestAutoReply: () -> Unit,
    listState: LazyListState,
    runnerSplit: Float,
    onRunnerSplitChanged: (Float) -> Unit,
    onOpenDetail: () -> Unit,
    onSelectCommand: (String) -> Unit
) {
    val activeItems = state.items.filter { it.status !in terminalStates }
    val data = state.activeCommand
    val promptText = remember(data) { extractPromptText(data?.pending_prompt) }
    val timelineLines = remember(data) { buildTimelineLines(data?.logs) }
    val geminiLines = remember(data) { buildGeminiLines(data) }
    val consoleText = remember(data) { buildConsoleText(data) }

    LazyColumn(
        state = listState,
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            SectionCard("Session Queue") {
                Text(state.sessionStatus, color = MaterialTheme.colorScheme.onSurfaceVariant)
                OperatorSummaryStrip(
                    entries = listOf(
                        "Active jobs" to activeItems.size.toString(),
                        "Current" to (state.activeCommandId ?: "--"),
                        "Stream" to state.streamStatus
                    )
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilledTonalButton(onClick = onRefresh) {
                        Text("Update Sessions")
                    }
                    OutlinedButton(onClick = onRetryStream, enabled = data != null) {
                        Text("Reload Stream")
                    }
                    OutlinedButton(onClick = onOpenDetail, enabled = data != null) {
                        Text("Open Detail")
                    }
                    OutlinedButton(onClick = onStop, enabled = data != null && data.status !in terminalStates) {
                        Text("Stop")
                    }
                }
                SplitRatioSelector(
                    selected = runnerSplit,
                    onSelected = onRunnerSplitChanged
                )
                if (activeItems.isEmpty()) {
                    Text("No running YTS session detected.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                } else {
                    activeItems.take(6).forEach { item ->
                        SessionQueueRow(item = item, selected = item.command_id == state.activeCommandId) {
                            onSelectCommand(item.command_id)
                        }
                    }
                }
            }
        }

        item {
            if (data == null) {
                SectionCard(null) {
                    Text(
                        "Select a running session from the queue or start a new job from the Create Job window.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            } else {
                BoxWithConstraints(modifier = Modifier.fillMaxWidth()) {
                    val wideLayout = maxWidth >= 900.dp
                    if (wideLayout) {
                        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                            SessionSummary(data)
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(12.dp)
                            ) {
                                Column(
                                    modifier = Modifier.weight(1f - runnerSplit),
                                    verticalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    SignalCard("Execution Log") {
                                        LogText(timelineLines.ifBlank { "No structured run events yet." })
                                    }
                                    PromptPanel(
                                        promptText = promptText,
                                        promptInput = state.promptInput,
                                        onPromptChanged = onPromptChanged,
                                        onSendPrompt = onSendPrompt,
                                        onSuggestDraft = onSuggestDraft,
                                        onSuggestAutoReply = onSuggestAutoReply
                                    )
                                }
                                Column(
                                    modifier = Modifier.weight(runnerSplit),
                                    verticalArrangement = Arrangement.spacedBy(12.dp)
                                ) {
                                    FullStreamPanel(
                                        frameBytes = state.streamFrameBytes,
                                        streamStatus = state.streamStatus
                                    )
                                    RunnerRemotePanel(
                                        remoteStatus = state.remoteStatus,
                                        onSendRemoteAction = onSendRemoteAction
                                    )
                                }
                            }
                            SignalCard("Gemini Log") {
                                LogText(geminiLines.ifBlank { "No Gemini-specific log lines detected yet." })
                            }
                            SignalCard("Console Log") {
                                LogText(consoleText)
                            }
                        }
                    } else {
                        Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                            SessionSummary(data)
                            FullStreamPanel(
                                frameBytes = state.streamFrameBytes,
                                streamStatus = state.streamStatus
                            )
                            SignalCard("Execution Log") {
                                LogText(timelineLines.ifBlank { "No structured run events yet." })
                            }
                            PromptPanel(
                                promptText = promptText,
                                promptInput = state.promptInput,
                                onPromptChanged = onPromptChanged,
                                onSendPrompt = onSendPrompt,
                                onSuggestDraft = onSuggestDraft,
                                onSuggestAutoReply = onSuggestAutoReply
                            )
                            RunnerRemotePanel(
                                remoteStatus = state.remoteStatus,
                                onSendRemoteAction = onSendRemoteAction
                            )
                            SignalCard("Gemini Log") {
                                LogText(geminiLines.ifBlank { "No Gemini-specific log lines detected yet." })
                            }
                            SignalCard("Console Log") {
                                LogText(consoleText)
                            }
                        }
                    }
                }
            }
        }

        if (data != null) {
            item {
                ProgressPanel(
                    data = data,
                    requestedTestCount = state.commandTestCountHints[state.activeCommandId] ?: state.selectedTestIds.size
                )
            }
        }
    }
}

@Composable
private fun SplitRatioSelector(
    selected: Float,
    onSelected: (Float) -> Unit
) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        listOf(
            0.5f to "50/50",
            0.6f to "40/60",
            0.7f to "30/70"
        ).forEach { (ratio, label) ->
            OutlinedButton(onClick = { onSelected(ratio) }) {
                Text(if (selected == ratio) "[$label]" else label)
            }
        }
    }
}

@Composable
private fun ResultsWindow(
    state: YtsListUiState,
    onRefresh: () -> Unit,
    onMonitor: (String) -> Unit,
    onStop: () -> Unit,
    listState: LazyListState,
    compactControls: Boolean,
    showCommandCenter: Boolean,
    onToggleCommandCenter: () -> Unit,
    onOpenDetail: (String) -> Unit,
    onOpenReport: (String) -> Unit,
    onOpenArtifact: (String, String) -> Unit
) {
    val completedItems = state.items.filter { it.status in terminalStates }
    LazyColumn(
        state = listState,
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            YtsControlDeck(
                activeCommandId = state.activeCommandId,
                onRefreshSessions = onRefresh,
                onRefreshCatalog = onRefresh,
                onStop = onStop,
                compact = compactControls,
                expanded = showCommandCenter,
                onToggle = onToggleCommandCenter
            )
        }
        item {
            SectionCard("Past Results") {
                Text(
                    "Completed and stopped sessions stay here so the runner window stays focused on what is active now.",
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                FilledTonalButton(onClick = onRefresh) {
                    Text("Refresh Results")
                }
            }
        }
        items(completedItems, key = { it.command_id }) { item ->
            ElevatedCard(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(14.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(item.command ?: item.command_id, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Medium)
                    Text("Status: ${item.status}")
                    Text("Updated: ${item.updated_at ?: "--"}")
                    Text(item.command_id, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        FilledTonalButton(onClick = { onOpenDetail(item.command_id) }) {
                            Text("Open Detail")
                        }
                        if (!item.report_html_name.isNullOrBlank() || !item.report_pdf_name.isNullOrBlank()) {
                            OutlinedButton(onClick = { onOpenReport(item.command_id) }) {
                                Text("View Summary")
                            }
                        }
                        if (!item.result_file_name.isNullOrBlank()) {
                            OutlinedButton(onClick = { onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_RESULT) }) {
                                Text("View Result")
                            }
                        }
                        if (!item.report_pdf_name.isNullOrBlank()) {
                            OutlinedButton(onClick = { onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_REPORT_PDF) }) {
                                Text("Download PDF")
                            }
                        }
                        if (!item.report_html_name.isNullOrBlank()) {
                            OutlinedButton(onClick = { onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_REPORT_HTML) }) {
                                Text("Download HTML")
                            }
                        }
                        OutlinedButton(onClick = { onMonitor(item.command_id) }) {
                            Text("Load In Runner")
                        }
                    }
                    Text(
                        buildResultSummaryText(item),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
private fun SessionQueueRow(
    item: YtsLiveCommandSummaryDto,
    selected: Boolean,
    onSelect: () -> Unit
) {
    Surface(
        tonalElevation = if (selected) 4.dp else 1.dp,
        shape = MaterialTheme.shapes.medium,
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onSelect)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(item.command ?: item.command_id, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text("Status: ${item.status}")
            Text("Updated: ${item.updated_at ?: "--"}", style = MaterialTheme.typography.bodySmall)
        }
    }
}

@Composable
private fun PromptPanel(
    promptText: String,
    promptInput: String,
    onPromptChanged: (String) -> Unit,
    onSendPrompt: () -> Unit,
    onSuggestDraft: () -> Unit,
    onSuggestAutoReply: () -> Unit
) {
    SignalCard("Prompt Response") {
        Text(
            if (promptText.isNotBlank()) {
                promptText
            } else {
                "Use this input feed to send manual replies or trigger Gemini suggestions while the test is running."
            }
        )
        OutlinedTextField(
            value = promptInput,
            onValueChange = onPromptChanged,
            label = { Text("Reply to running test") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 3
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            FilledTonalButton(onClick = onSendPrompt) {
                Text("Send Reply")
            }
            OutlinedButton(onClick = onSuggestDraft) {
                Text("Gemini Draft")
            }
            OutlinedButton(onClick = onSuggestAutoReply) {
                Text("Auto Reply")
            }
        }
    }
}

@Composable
private fun RunnerRemotePanel(
    remoteStatus: String,
    onSendRemoteAction: (String) -> Unit
) {
    SignalCard("TV Remote") {
        Text(remoteStatus, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilledTonalButton(onClick = { onSendRemoteAction("PRESS_UP") }) { Text("Up") }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { onSendRemoteAction("PRESS_LEFT") }) { Text("Left") }
                FilledTonalButton(onClick = { onSendRemoteAction("PRESS_OK") }) { Text("OK") }
                OutlinedButton(onClick = { onSendRemoteAction("PRESS_RIGHT") }) { Text("Right") }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilledTonalButton(onClick = { onSendRemoteAction("PRESS_DOWN") }) { Text("Down") }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf(
                    "Back" to "PRESS_BACK",
                    "Home" to "PRESS_HOME",
                    "Menu" to "PRESS_MENU",
                    "Play" to "PRESS_PLAY_PAUSE",
                    "Power" to "PRESS_POWER"
                ).forEach { (label, action) ->
                    OutlinedButton(onClick = { onSendRemoteAction(action) }) {
                        Text(label)
                    }
                }
            }
        }
    }
}

@Composable
private fun ProgressPanel(
    data: YtsLiveCommandStateDto,
    requestedTestCount: Int
) {
    val progress = remember(data, requestedTestCount) { deriveTestProgress(data, requestedTestCount) }
    val milestones = buildList {
        add("Session created")
        if (progress.completed > 0) add("${progress.completed} tests completed")
        if (!data.stdout.isNullOrBlank() || !data.stderr.isNullOrBlank()) add("Console output received")
        if (data.awaiting_input || extractPromptText(data.pending_prompt).isNotBlank()) add("Interactive prompt issued")
        if (!data.report_html_name.isNullOrBlank() || !data.report_pdf_name.isNullOrBlank()) add("Report generated")
        if (data.status in terminalStates) add("Run finished")
    }

    SectionCard("Test Progress") {
        Text("Current state: ${data.status}", fontWeight = FontWeight.SemiBold)
        Text(
            "Completed ${progress.completed} of ${progress.total} tests",
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        LinearProgressIndicator(
            progress = { progress.progressValue },
            modifier = Modifier.fillMaxWidth()
        )
        OperatorSummaryStrip(
            entries = listOf(
                "Completed" to progress.completed.toString(),
                "Passed" to progress.passed.toString(),
                "Failed" to progress.failed.toString()
            )
        )
        milestones.forEach { item ->
            Text("• $item", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun SessionSummary(data: YtsLiveCommandStateDto) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        tonalElevation = 3.dp,
        shape = MaterialTheme.shapes.medium
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(data.command ?: "(starting...)", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text("Command ID: ${data.command_id}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("Status: ${data.status}")
            Text("Updated: ${data.updated_at ?: "--"}")
            Text("Exit code: ${data.returncode?.toString() ?: "--"}")
            if (data.video_recording_status != null) {
                Text("Video: ${data.video_recording_status}")
            }
        }
    }
}

@Composable
private fun FullStreamPanel(
    modifier: Modifier = Modifier,
    frameBytes: ByteArray?,
    streamStatus: String
) {
    SignalCard("Live Streaming", modifier = modifier) {
        val bitmap = remember(frameBytes) {
            frameBytes?.let { bytes ->
                BitmapFactory.decodeByteArray(bytes, 0, bytes.size)?.asImageBitmap()
            }
        }
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(16f / 9f)
                .background(Brush.linearGradient(listOf(Color(0xFF0F172A), Color(0xFF1E293B)))),
            contentAlignment = Alignment.Center
        ) {
            if (bitmap != null) {
                Image(
                    bitmap = bitmap,
                    contentDescription = "Device live stream",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
                )
            } else {
                Text(
                    "Video preview will appear here when frames arrive.",
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
        Text(streamStatus, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun FloatingYtsStreamOverlay(
    frameBytes: ByteArray?,
    streamStatus: String,
    onHide: () -> Unit,
    modifier: Modifier = Modifier
) {
    val bitmap = frameBytes?.let { bytes ->
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
                modifier = Modifier.width(320.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Live", style = MaterialTheme.typography.labelLarge, fontWeight = FontWeight.SemiBold)
                OutlinedButton(onClick = onHide) {
                    Text("Hide")
                }
            }
            Surface(shape = MaterialTheme.shapes.medium, modifier = Modifier.width(320.dp)) {
                if (bitmap != null) {
                    Image(
                        bitmap = bitmap,
                        contentDescription = "Floating YTS live stream",
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
private fun SectionCard(
    title: String?,
    content: @Composable () -> Unit
) {
    ElevatedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            verticalArrangement = Arrangement.spacedBy(0.dp)
        ) {
            if (!title.isNullOrBlank()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(
                            Brush.horizontalGradient(
                                listOf(Color(0xFFD97706), Color(0xFFEA580C), Color(0xFFBE123C))
                            )
                        )
                        .padding(horizontal = 14.dp, vertical = 10.dp)
                ) {
                    Text(
                        title,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = Color.White
                    )
                }
            }
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                content()
            }
        }
    }
}

@Composable
private fun SignalCard(
    title: String,
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit
) {
    Surface(
        modifier = modifier.fillMaxWidth(),
        tonalElevation = 2.dp,
        shape = MaterialTheme.shapes.medium
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            content()
        }
    }
}

@Composable
private fun ToggleRow(
    label: String,
    value: Boolean,
    onToggle: () -> Unit
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(label)
        Switch(checked = value, onCheckedChange = { onToggle() })
    }
}

@Composable
private fun ChoiceChips(
    title: String,
    options: List<String>,
    selected: String,
    onSelected: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(title, style = MaterialTheme.typography.titleSmall)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            OutlinedButton(onClick = { onSelected("") }) {
                Text(if (selected.isBlank()) "All" else "Clear")
            }
        }
        LazyColumn(
            modifier = Modifier.heightIn(max = 120.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            items(options) { option ->
                Surface(
                    tonalElevation = if (selected == option) 4.dp else 1.dp,
                    shape = MaterialTheme.shapes.small,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onSelected(option) }
                ) {
                    Text(
                        option,
                        modifier = Modifier.padding(10.dp),
                        color = if (selected == option) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        }
    }
}

@Composable
private fun ModelChips(
    title: String,
    availableModels: List<String>,
    currentValue: String,
    onSelected: (String) -> Unit
) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Text(title, style = MaterialTheme.typography.titleSmall)
        OutlinedTextField(
            value = currentValue,
            onValueChange = onSelected,
            label = { Text(title) },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true
        )
        LazyColumn(
            modifier = Modifier.heightIn(max = 150.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            items(availableModels) { model ->
                Surface(
                    tonalElevation = if (model == currentValue) 4.dp else 1.dp,
                    shape = MaterialTheme.shapes.small,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { onSelected(model) }
                ) {
                    Text(
                        model,
                        modifier = Modifier.padding(10.dp),
                        color = if (model == currentValue) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface
                    )
                }
            }
        }
    }
}

@Composable
private fun LogText(text: String) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .background(Color(0xFF09111F), RoundedCornerShape(16.dp))
            .padding(12.dp)
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.bodySmall,
            fontFamily = FontFamily.Monospace,
            color = Color(0xFFD9F99D)
        )
    }
}

@Composable
private fun HeroHeader(
    state: YtsListUiState,
    compact: Boolean,
    expanded: Boolean
) {
    val scrollState = rememberScrollState()
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(18.dp),
        shadowElevation = 6.dp
    ) {
        Box(
            modifier = Modifier
                .background(
                    Brush.linearGradient(
                        listOf(Color(0xFF7C2D12), Color(0xFF1D4ED8), Color(0xFF0F766E))
                    )
                )
                .padding(horizontal = 12.dp, vertical = 8.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(max = if (expanded) 96.dp else 40.dp)
                    .verticalScroll(scrollState),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.Start,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            "YTS Command Center",
                            style = if (compact) MaterialTheme.typography.titleSmall else MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            color = Color.White
                        )
                        if (!compact && expanded) {
                            Text(
                                "Create jobs, update YTS catalog data, stop live sessions, and review results from a single operator workspace.",
                                style = MaterialTheme.typography.bodySmall,
                                color = Color(0xFFE5EEF9)
                            )
                        }
                    }
                }
                if (expanded) {
                    OperatorSummaryStrip(
                        entries = listOf(
                            "Device" to state.deviceId.ifBlank { "--" },
                            "Selected Tests" to state.selectedTestIds.size.toString(),
                            "Active Session" to (state.activeCommandId ?: "--")
                        ),
                        dark = true
                    )
                    state.error?.let {
                        Text(it, color = Color(0xFFFECACA))
                    }
                }
            }
        }
    }
}

@Composable
private fun YtsControlDeck(
    activeCommandId: String?,
    onRefreshSessions: () -> Unit,
    onRefreshCatalog: () -> Unit,
    onStop: () -> Unit,
    compact: Boolean,
    expanded: Boolean,
    onToggle: () -> Unit
) {
    val scrollState = rememberScrollState()
    Surface(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        tonalElevation = 3.dp,
        color = Color(0xFFFFF7ED)
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("YTS Controls", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                OutlinedButton(onClick = onToggle) {
                    Text(if (expanded) "Collapse" else "Expand")
                }
            }
            if (expanded) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = if (compact) 84.dp else 108.dp)
                        .verticalScroll(scrollState),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    if (!compact) {
                        Text(
                            "Surface the available runner operations clearly: update sessions, update catalog, and stop the active run.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        FilledTonalButton(onClick = onRefreshSessions) {
                            Text(if (compact) "Sessions" else "Update YTS Sessions")
                        }
                        OutlinedButton(onClick = onRefreshCatalog) {
                            Text(if (compact) "Catalog" else "Update Test Catalog")
                        }
                        OutlinedButton(onClick = onStop, enabled = activeCommandId != null) {
                            Text(if (compact) "Stop" else "Stop Active YTS")
                        }
                    }
                    if (scrollState.maxValue > 0) {
                        Text(
                            "Scroll inside this panel to access the remaining controls.",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun OperatorSummaryStrip(
    entries: List<Pair<String, String>>,
    dark: Boolean = false
) {
    val chipColor = if (dark) Color(0x33FFFFFF) else Color(0xFFF3F4F6)
    val textColor = if (dark) Color.White else Color(0xFF111827)
    Row(
        modifier = Modifier.horizontalScroll(rememberScrollState()),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        entries.take(4).forEach { (label, value) ->
            Surface(
                color = chipColor,
                shape = RoundedCornerShape(14.dp),
                modifier = Modifier.width(110.dp)
            ) {
                Column(modifier = Modifier.padding(horizontal = 10.dp, vertical = 8.dp)) {
                    Text(label, style = MaterialTheme.typography.labelSmall, color = textColor.copy(alpha = 0.8f))
                    Text(value, style = MaterialTheme.typography.bodySmall, color = textColor, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }
        }
    }
}

@Composable
private fun StatusLine(
    label: String,
    value: String,
    tone: Color = MaterialTheme.colorScheme.primary
) {
    Surface(
        color = tone.copy(alpha = 0.12f),
        shape = RoundedCornerShape(14.dp)
    ) {
        Text(
            "$label: $value",
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
            style = MaterialTheme.typography.bodySmall,
            color = tone
        )
    }
}

private fun extractPromptText(prompt: JsonObject?): String {
    return prompt?.get("text")?.toString().orEmpty().trim('"')
}

private fun deriveTestProgress(
    data: YtsLiveCommandStateDto,
    requestedTestCount: Int
): TestProgressInfo {
    val text = buildString {
        appendLine(data.stdout.orEmpty())
        appendLine(data.stderr.orEmpty())
        appendLine(data.logs?.joinToString("\n") { it.toString() }.orEmpty())
    }
    val directMatch = Regex("""(?i)(\d+)\s*(?:/|of)\s*(\d+)\s*tests?""").findAll(text).lastOrNull()
    val completedFromDirect = directMatch?.groupValues?.getOrNull(1)?.toIntOrNull() ?: 0
    val totalFromDirect = directMatch?.groupValues?.getOrNull(2)?.toIntOrNull() ?: 0
    val passed = Regex("""(?i)\bpass(?:ed)?\b""").findAll(text).count()
    val failed = Regex("""(?i)\bfail(?:ed)?\b|\berror\b""").findAll(text).count()
    val completed = maxOf(completedFromDirect, passed + failed)
    val total = listOf(totalFromDirect, requestedTestCount, completed).maxOrNull()?.coerceAtLeast(1) ?: 1
    return TestProgressInfo(
        completed = completed.coerceAtMost(total),
        total = total,
        passed = passed.coerceAtMost(total),
        failed = failed.coerceAtMost(total)
    )
}

private fun buildResultSummaryText(item: YtsLiveCommandSummaryDto): String {
    return buildString {
        append("Result: ")
        append(item.result_file_name ?: "not generated")
        append("  |  HTML: ")
        append(item.report_html_name ?: "none")
        append("  |  PDF: ")
        append(item.report_pdf_name ?: "none")
    }
}

private data class TestProgressInfo(
    val completed: Int,
    val total: Int,
    val passed: Int,
    val failed: Int
) {
    val progressValue: Float
        get() = if (total <= 0) 0f else completed.toFloat() / total.toFloat()
}

private fun buildTimelineLines(logs: JsonArray?): String {
    return logs
        ?.takeLast(18)
        ?.joinToString("\n") { element -> shortenLine(element.toString()) }
        .orEmpty()
}

private fun buildGeminiLines(data: YtsLiveCommandStateDto?): String {
    if (data == null) return ""
    val promptLine = extractPromptText(data.pending_prompt)
    val geminiEvents = data.logs
        ?.map { it.toString() }
        ?.filter { line ->
            val normalized = line.lowercase()
            listOf("gemini", "vertex", "prompt", "vision", "planner", "model", "response", "ai").any { token ->
                normalized.contains(token)
            }
        }
        ?.takeLast(12)
        .orEmpty()
    return buildList {
        if (promptLine.isNotBlank()) add("Pending prompt: ${shortenLine(promptLine)}")
        addAll(geminiEvents.map(::shortenLine))
    }.joinToString("\n")
}

private fun buildConsoleText(data: YtsLiveCommandStateDto?): String {
    if (data == null) return "No active console output."
    val stdout = data.stdout?.trim().orEmpty()
    val stderr = data.stderr?.trim().orEmpty()
    return buildString {
        appendLine("STDOUT")
        appendLine(if (stdout.isBlank()) "(empty)" else stdout.takeLast(2200))
        appendLine()
        appendLine("STDERR")
        append(if (stderr.isBlank()) "(empty)" else stderr.takeLast(2200))
    }.trim()
}

private fun shortenLine(value: String, max: Int = 220): String {
    return if (value.length <= max) value else value.take(max) + "..."
}

private val terminalStates = setOf("completed", "stopped", "failed")
