package com.dabcontrol.app.ui.yts

import android.content.Intent
import android.net.Uri
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Checkbox
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
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.material3.Surface
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.dabcontrol.app.data.api.YtsResultArtifactItemDto
import com.dabcontrol.app.data.api.YtsLiveCommandSummaryDto
import com.dabcontrol.app.ui.common.PremiumBackdrop
import com.dabcontrol.app.ui.common.SectionLabel

private enum class ResultsTab(val label: String) {
    OVERVIEW("Overview"),
    SESSION_TABLE("Session Table"),
    AI_ANALYSIS("AI Analysis"),
    FILES("Files & Artifacts")
}

private enum class ResultsStatusFilter(val label: String) {
    ALL("All statuses"),
    PASSED("Passed"),
    FAILED("Failed"),
    STOPPED("Stopped"),
    COMPLETED("Completed")
}

@Composable
fun YtsResultsScreen(
    onOpenCommand: (String) -> Unit,
    onOpenReport: (String) -> Unit,
    onOpenArtifact: (String, String) -> Unit,
    modifier: Modifier = Modifier,
    viewModel: YtsListViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val terminalItems = remember(state.items) { state.items.filter { it.status.lowercase() !in setOf("running", "queued", "starting") } }

    var activeTab by rememberSaveable { mutableStateOf(ResultsTab.OVERVIEW) }
    var statusFilter by rememberSaveable { mutableStateOf(ResultsStatusFilter.ALL) }
    var searchQuery by rememberSaveable { mutableStateOf("") }
    val filteredItems = remember(terminalItems, statusFilter, searchQuery) {
        terminalItems.filter { item ->
            matchesStatus(item, statusFilter) && matchesSearch(item, searchQuery)
        }
    }
    val selectedItem = filteredItems.firstOrNull()

    PremiumBackdrop(modifier = modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp, vertical = 20.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            SectionLabel(
                eyebrow = "Results Workspace",
                title = "Backend-Linked Reports and Sessions",
                subtitle = "A proper analyst view for session status, generated reports, and output artifacts."
            )

            ResultsToolbar(
                state = state,
                statusFilter = statusFilter,
                onStatusFilterChanged = { statusFilter = it },
                searchQuery = searchQuery,
                onSearchChanged = { searchQuery = it },
                onRefresh = viewModel::refresh
            )

            ResultsTabs(activeTab = activeTab, onTabSelected = { activeTab = it })

            when (activeTab) {
                ResultsTab.OVERVIEW -> ResultsOverviewTab(
                    items = filteredItems,
                    selectedItem = selectedItem,
                    onOpenCommand = onOpenCommand,
                    onOpenReport = onOpenReport,
                    onOpenArtifact = onOpenArtifact
                )
                ResultsTab.SESSION_TABLE -> ResultsTableTab(
                    items = filteredItems,
                    onOpenCommand = onOpenCommand,
                    onOpenReport = onOpenReport,
                    onOpenArtifact = onOpenArtifact
                )
                ResultsTab.AI_ANALYSIS -> YtsResultsAnalysisPanel(
                    artifacts = state.artifacts,
                    analysisReportText = state.analysisReportText,
                    analysisStatus = state.analysisStatus,
                    analysisReportId = state.analysisReportId,
                    analysisTxtName = state.analysisTxtName,
                    analysisPdfName = state.analysisPdfName,
                    apiBaseUrl = state.apiBaseUrl,
                    analysisLoading = state.analysisLoading,
                    onAnalyze = viewModel::analyzeArtifacts,
                    onRefreshArtifacts = viewModel::fetchArtifacts
                )
                ResultsTab.FILES -> ResultsFilesTab(
                    items = filteredItems,
                    onOpenArtifact = onOpenArtifact
                )
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ResultsToolbar(
    state: YtsListUiState,
    statusFilter: ResultsStatusFilter,
    onStatusFilterChanged: (ResultsStatusFilter) -> Unit,
    searchQuery: String,
    onSearchChanged: (String) -> Unit,
    onRefresh: () -> Unit
) {
    ElevatedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text("Result Control Deck", style = MaterialTheme.typography.titleLarge)
                    Text(
                        "Proof that the app is connected: session rows, report assets, and backend-driven timestamps update here.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                if (state.isLoading) {
                    CircularProgressIndicator()
                }
            }

            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                SummaryChip("Sessions", state.items.size.toString())
                SummaryChip("Active", state.activeCommandId ?: "None")
                SummaryChip("Backend", state.apiBaseUrl.ifBlank { "Not configured" })
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                FilledTonalButton(onClick = onRefresh) {
                    Text("Refresh Workspace")
                }
                StatusFilterDropdown(selected = statusFilter, onSelected = onStatusFilterChanged)
            }

            OutlinedTextField(
                value = searchQuery,
                onValueChange = onSearchChanged,
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Search by command, ID, report, or artifact") },
                singleLine = true
            )
            state.error?.let {
                Surface(
                    color = MaterialTheme.colorScheme.errorContainer,
                    shape = MaterialTheme.shapes.medium
                ) {
                    Text(
                        text = it,
                        modifier = Modifier.padding(horizontal = 12.dp, vertical = 10.dp),
                        color = MaterialTheme.colorScheme.onErrorContainer
                    )
                }
            }
        }
    }
}

@Composable
private fun ResultsTabs(activeTab: ResultsTab, onTabSelected: (ResultsTab) -> Unit) {
    val tabs = ResultsTab.entries
    ScrollableTabRow(
        selectedTabIndex = tabs.indexOf(activeTab),
        containerColor = Color.Transparent,
        contentColor = MaterialTheme.colorScheme.onSurface,
        edgePadding = 0.dp,
        divider = {}
    ) {
        tabs.forEach { tab ->
            Tab(
                selected = tab == activeTab,
                onClick = { onTabSelected(tab) },
                text = {
                    Text(
                        tab.label,
                        style = MaterialTheme.typography.labelLarge,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ResultsOverviewTab(
    items: List<YtsLiveCommandSummaryDto>,
    selectedItem: YtsLiveCommandSummaryDto?,
    onOpenCommand: (String) -> Unit,
    onOpenReport: (String) -> Unit,
    onOpenArtifact: (String, String) -> Unit
) {
    LazyColumn(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        item {
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                maxItemsInEachRow = 3
            ) {
                OverviewCard("Ready Reports", items.count { hasReport(it) }.toString(), "HTML or PDF generated")
                OverviewCard("Result Files", items.count { !it.result_file_name.isNullOrBlank() }.toString(), "Structured outputs available")
                OverviewCard("Video Assets", items.count { !it.video_file_name.isNullOrBlank() }.toString(), "Recorded validation evidence")
            }
        }
        item {
            ElevatedCard(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text("Backend Data Snapshot", style = MaterialTheme.typography.titleLarge)
                    Text(
                        "This window is populated from live command summaries returned by the backend, not from a static file on device.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    selectedItem?.let { item ->
                        KeyValueTable(
                            rows = listOf(
                                "Command ID" to item.command_id,
                                "Status" to item.status,
                                "Updated" to (item.updated_at ?: "--"),
                                "Result" to (item.result_file_name ?: "--"),
                                "HTML report" to (item.report_html_name ?: "--"),
                                "PDF report" to (item.report_pdf_name ?: "--")
                            )
                        )
                    } ?: Text("No matching result sessions found.")
                }
            }
        }
        items(items.take(5), key = { it.command_id }) { item ->
            ResultSpotlightCard(
                item = item,
                onOpenCommand = onOpenCommand,
                onOpenReport = onOpenReport,
                onOpenArtifact = onOpenArtifact
            )
        }
    }
}

@Composable
private fun ResultsTableTab(
    items: List<YtsLiveCommandSummaryDto>,
    onOpenCommand: (String) -> Unit,
    onOpenReport: (String) -> Unit,
    onOpenArtifact: (String, String) -> Unit
) {
    LazyColumn(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        item {
            ElevatedCard(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("Session Table", style = MaterialTheme.typography.titleLarge)
                    Text("Comprehensive list of all past execution sessions.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
        items(items, key = { it.command_id }) { item ->
            SessionTableRow(item = item, onOpenCommand = onOpenCommand, onOpenReport = onOpenReport, onOpenArtifact = onOpenArtifact)
        }
    }
}

@Composable
internal fun YtsResultsAnalysisPanel(
    artifacts: List<YtsResultArtifactItemDto>,
    analysisReportText: String,
    analysisStatus: String,
    analysisReportId: String,
    analysisTxtName: String,
    analysisPdfName: String,
    apiBaseUrl: String,
    analysisLoading: Boolean,
    onAnalyze: (List<String>, Boolean) -> Unit,
    onRefreshArtifacts: () -> Unit
) {
    var selectedRefs by rememberSaveable { mutableStateOf(setOf<String>()) }
    var includeZipBase64 by rememberSaveable { mutableStateOf(true) }
    val groupedArtifacts = remember(artifacts) { artifacts.groupBy { it.command_id.ifBlank { "unknown" } } }
    val latestCommandId = artifacts.firstOrNull { it.command_id.isNotBlank() }?.command_id
    val context = LocalContext.current
    val baseUrl = apiBaseUrl.trimEnd('/')

    LazyColumn(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        item {
            ElevatedCard(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("YTS Results Gemini Analysis", style = MaterialTheme.typography.titleLarge)
                    Text(
                        "Select result artifacts and ask the backend Gemini analysis engine to correlate JSON results, terminal logs, and DAB/device evidence into a triage report.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        style = MaterialTheme.typography.bodyMedium
                    )

                    Surface(
                        modifier = Modifier.fillMaxWidth(),
                        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
                        shape = MaterialTheme.shapes.small
                    ) {
                        Text(
                            text = analysisStatus,
                            modifier = Modifier.padding(12.dp),
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }

                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Checkbox(
                            checked = includeZipBase64,
                            onCheckedChange = { includeZipBase64 = it }
                        )
                        Text("Include DAB ZIP base64 in Gemini prompt")
                    }

                    Row(
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        FilledTonalButton(
                            onClick = { onAnalyze(selectedRefs.toList(), includeZipBase64) },
                            enabled = !analysisLoading && selectedRefs.isNotEmpty(),
                            modifier = Modifier.weight(1f)
                        ) {
                            Text(if (analysisLoading) "Analyzing..." else "Analyze ${selectedRefs.size} Selected")
                        }
                        OutlinedButton(onClick = onRefreshArtifacts) {
                            Text("Refresh")
                        }
                    }

                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(
                            enabled = latestCommandId != null,
                            onClick = {
                                latestCommandId?.let { commandId ->
                                    selectedRefs = coreArtifactRefs(groupedArtifacts[commandId].orEmpty())
                                }
                            }
                        ) { Text("Latest Core") }
                        OutlinedButton(onClick = { selectedRefs = emptySet() }) {
                            Text("Clear")
                        }
                    }

                    if (analysisReportText.isNotBlank()) {
                        Surface(
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(min = 200.dp),
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
                            shape = MaterialTheme.shapes.medium
                        ) {
                            Text(
                                text = analysisReportText,
                                modifier = Modifier.padding(12.dp),
                                fontFamily = FontFamily.Monospace,
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }

                    if (analysisReportId.isNotBlank() && baseUrl.isNotBlank()) {
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(
                                onClick = {
                                    context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(analysisDownloadUrl(baseUrl, analysisReportId, "txt"))))
                                }
                            ) {
                                Text(analysisTxtName.ifBlank { "Open TXT" })
                            }
                            OutlinedButton(
                                onClick = {
                                    context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(analysisDownloadUrl(baseUrl, analysisReportId, "pdf"))))
                                }
                            ) {
                                Text(analysisPdfName.ifBlank { "Open PDF" })
                            }
                        }
                    }
                }
            }
        }

        item {
            Text(
                "Available Artifacts (${artifacts.size}) - ${selectedRefs.size} selected",
                style = MaterialTheme.typography.titleMedium,
                modifier = Modifier.padding(start = 4.dp)
            )
        }

        if (artifacts.isEmpty()) {
            item {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
                    shape = MaterialTheme.shapes.medium
                ) {
                    Text(
                        "No result artifacts found yet. Run a YTS session, then refresh artifacts.",
                        modifier = Modifier.padding(16.dp),
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }

        groupedArtifacts.forEach { (commandId, commandArtifacts) ->
            item(key = "job-$commandId") {
                ElevatedCard(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                        val first = commandArtifacts.firstOrNull()
                        Text(commandId, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                        Text(
                            listOfNotNull(first?.status, first?.updated_at, first?.command).filter { it.isNotBlank() }.joinToString(" - ").ifBlank { "YTS result artifacts" },
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            style = MaterialTheme.typography.bodySmall,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            OutlinedButton(onClick = { selectedRefs = coreArtifactRefs(commandArtifacts) }) {
                                Text("Core")
                            }
                            OutlinedButton(onClick = { selectedRefs = commandArtifacts.map { it.ref }.toSet() }) {
                                Text("All")
                            }
                        }
                    }
                }
            }
            items(commandArtifacts, key = { it.ref }) { artifact ->
                AnalysisArtifactRow(
                    artifact = artifact,
                    selected = selectedRefs.contains(artifact.ref),
                    onSelectedChanged = { selected ->
                        selectedRefs = if (selected) selectedRefs + artifact.ref else selectedRefs - artifact.ref
                    }
                )
            }
        }
    }
}

@Composable
private fun AnalysisArtifactRow(
    artifact: YtsResultArtifactItemDto,
    selected: Boolean,
    onSelectedChanged: (Boolean) -> Unit
) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onSelectedChanged(!selected) },
        color = if (selected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
        shape = MaterialTheme.shapes.small
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Checkbox(checked = selected, onCheckedChange = onSelectedChanged)
            Column(modifier = Modifier.weight(1f)) {
                Text(artifact.label.ifBlank { artifact.ref }, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                Text(
                    "${artifact.type.ifBlank { "artifact" }} - ${artifact.status ?: "--"}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    summarizeArtifact(artifact),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@Composable
private fun ResultsFilesTab(
    items: List<YtsLiveCommandSummaryDto>,
    onOpenArtifact: (String, String) -> Unit
) {
    LazyColumn(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        item {
            ElevatedCard(modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(18.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("Artifact Explorer", style = MaterialTheme.typography.titleLarge)
                    Text("Direct access to generated files and output evidence from recent sessions.", color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
            }
        }
        items(items, key = { it.command_id }) { item ->
            ResultArtifactsList(item = item, onOpenArtifact = onOpenArtifact)
        }
    }
}

@Composable
private fun ResultArtifactsList(
    item: YtsLiveCommandSummaryDto,
    onOpenArtifact: (String, String) -> Unit
) {
    ElevatedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Text(item.command ?: item.command_id, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
            Text(item.command_id, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            ArtifactStatusRow(
                label = "Saved Result",
                value = item.result_file_name,
                onOpen = { onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_RESULT) }
            )
            ArtifactStatusRow(
                label = "HTML Report",
                value = item.report_html_name,
                onOpen = { onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_REPORT_HTML) }
            )
            ArtifactStatusRow(
                label = "PDF Report",
                value = item.report_pdf_name,
                onOpen = { onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_REPORT_PDF) }
            )
            ArtifactStatusRow(
                label = "Video Evidence",
                value = item.video_file_name,
                onOpen = null
            )
        }
    }
}

@Composable
private fun ResultSpotlightCard(
    item: YtsLiveCommandSummaryDto,
    onOpenCommand: (String) -> Unit,
    onOpenReport: (String) -> Unit,
    onOpenArtifact: (String, String) -> Unit
) {
    ElevatedCard(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(item.command ?: item.command_id, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.SemiBold)
                    Text(item.updated_at ?: "--", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                StatusBadge(item.status)
            }
            Text(
                "Reports: ${if (hasReport(item)) "available" else "pending"}  |  Result bundle: ${item.result_file_name ?: "--"}",
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilledTonalButton(onClick = { onOpenCommand(item.command_id) }) {
                    Text("Open Detail")
                }
                if (hasReport(item)) {
                    OutlinedButton(onClick = { onOpenReport(item.command_id) }) {
                        Text("Open Report")
                    }
                }
                if (!item.result_file_name.isNullOrBlank()) {
                    OutlinedButton(onClick = { onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_RESULT) }) {
                        Text("Open Result")
                    }
                }
            }
        }
    }
}

@Composable
private fun TableHeaderRow() {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.7f),
        shape = MaterialTheme.shapes.medium
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 14.dp, vertical = 10.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            TableHeader("Session")
            TableHeader("Status")
            TableHeader("Reports")
            TableHeader("Artifacts")
            TableHeader("Actions")
        }
    }
}

@Composable
private fun SessionTableRow(
    item: YtsLiveCommandSummaryDto,
    onOpenCommand: (String) -> Unit,
    onOpenReport: (String) -> Unit,
    onOpenArtifact: (String, String) -> Unit
) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onOpenCommand(item.command_id) },
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 1.dp,
        shape = MaterialTheme.shapes.medium
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1.2f)) {
                    Text(item.command ?: item.command_id, fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(item.command_id, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                Box(modifier = Modifier.weight(0.7f), contentAlignment = Alignment.CenterStart) {
                    StatusBadge(item.status)
                }
                Box(modifier = Modifier.weight(0.9f), contentAlignment = Alignment.CenterStart) {
                    Text(
                        when {
                            hasReport(item) -> "Ready"
                            else -> "Pending"
                        },
                        style = MaterialTheme.typography.bodySmall
                    )
                }
                Box(modifier = Modifier.weight(1f), contentAlignment = Alignment.CenterStart) {
                    Text(
                        listOfNotNull(
                            item.result_file_name?.let { "Result" },
                            item.video_file_name?.let { "Video" }
                        ).ifEmpty { listOf("None") }.joinToString(" · "),
                        style = MaterialTheme.typography.bodySmall
                    )
                }
            }
            HorizontalDivider()
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilledTonalButton(onClick = { onOpenCommand(item.command_id) }) {
                    Text("Session")
                }
                if (hasReport(item)) {
                    OutlinedButton(onClick = { onOpenReport(item.command_id) }) {
                        Text("Report")
                    }
                }
                if (!item.result_file_name.isNullOrBlank()) {
                    OutlinedButton(onClick = { onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_RESULT) }) {
                        Text("Result File")
                    }
                }
            }
        }
    }
}

@Composable
private fun ArtifactStatusRow(label: String, value: String?, onOpen: (() -> Unit)? = null) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
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
                Text(label, style = MaterialTheme.typography.bodyMedium)
                Text(value ?: "Not generated", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            if (value != null && onOpen != null) {
                OutlinedButton(onClick = onOpen) {
                    Text("Open")
                }
            }
        }
    }
}

@Composable
private fun KeyValueTable(rows: List<Pair<String, String>>) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        rows.forEach { (label, value) ->
            Surface(
                modifier = Modifier.fillMaxWidth(),
                color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.45f),
                shape = MaterialTheme.shapes.small
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 10.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    Text(value, fontWeight = FontWeight.SemiBold)
                }
            }
        }
    }
}

@Composable
private fun OverviewCard(title: String, value: String, subtitle: String) {
    ElevatedCard {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(title, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(value, style = MaterialTheme.typography.headlineSmall)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun SummaryChip(label: String, value: String) {
    Surface(
        color = MaterialTheme.colorScheme.primaryContainer,
        shape = MaterialTheme.shapes.small
    ) {
        Column(modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp)) {
            Text(label, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f))
            Text(value, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onPrimaryContainer)
        }
    }
}

@Composable
private fun StatusBadge(status: String) {
    val normalized = status.lowercase()
    val background = when {
        "pass" in normalized || "complete" in normalized -> Color(0x1A3B82F6)
        "fail" in normalized || "error" in normalized -> Color(0x1AEF4444)
        else -> Color(0x1A6366F1)
    }
    val foreground = when {
        "pass" in normalized || "complete" in normalized -> Color(0xFF3B82F6)
        "fail" in normalized || "error" in normalized -> Color(0xFFEF4444)
        else -> Color(0xFF818CF8)
    }
    Surface(color = background, shape = MaterialTheme.shapes.small) {
        Text(
            text = status,
            modifier = Modifier.padding(horizontal = 10.dp, vertical = 6.dp),
            color = foreground,
            style = MaterialTheme.typography.bodySmall,
            fontWeight = FontWeight.SemiBold
        )
    }
}

@Composable
private fun TableHeader(text: String) {
    Text(text, style = MaterialTheme.typography.labelLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun StatusFilterDropdown(
    selected: ResultsStatusFilter,
    onSelected: (ResultsStatusFilter) -> Unit
) {
    var expanded by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = !expanded }) {
        OutlinedTextField(
            value = selected.label,
            onValueChange = {},
            readOnly = true,
            label = { Text("Status filter") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
            modifier = Modifier.menuAnchor().fillMaxWidth()
        )
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            ResultsStatusFilter.entries.forEach { option ->
                DropdownMenuItem(
                    text = { Text(option.label) },
                    onClick = {
                        onSelected(option)
                        expanded = false
                    }
                )
            }
        }
    }
}

private fun matchesStatus(item: YtsLiveCommandSummaryDto, filter: ResultsStatusFilter): Boolean {
    val normalized = item.status.lowercase()
    return when (filter) {
        ResultsStatusFilter.ALL -> true
        ResultsStatusFilter.PASSED -> "pass" in normalized
        ResultsStatusFilter.FAILED -> "fail" in normalized || "error" in normalized
        ResultsStatusFilter.STOPPED -> "stop" in normalized
        ResultsStatusFilter.COMPLETED -> "complete" in normalized || "finished" in normalized
    }
}

private fun matchesSearch(item: YtsLiveCommandSummaryDto, query: String): Boolean {
    if (query.isBlank()) return true
    val haystack = listOfNotNull(
        item.command_id,
        item.command,
        item.status,
        item.result_file_name,
        item.report_html_name,
        item.report_pdf_name,
        item.video_file_name
    ).joinToString(" ").lowercase()
    return haystack.contains(query.trim().lowercase())
}

private fun hasReport(item: YtsLiveCommandSummaryDto): Boolean {
    return !item.report_html_name.isNullOrBlank() || !item.report_pdf_name.isNullOrBlank()
}

private fun coreArtifactRefs(artifacts: List<YtsResultArtifactItemDto>): Set<String> {
    val coreTypes = setOf("result-json", "terminal-log", "dab-log-summary")
    return artifacts
        .filter { it.type in coreTypes || it.ref.substringAfterLast(":") in coreTypes }
        .map { it.ref }
        .toSet()
        .ifEmpty { artifacts.take(1).map { it.ref }.toSet() }
}

private fun summarizeArtifact(artifact: YtsResultArtifactItemDto): String {
    val summary = artifact.result_summary?.toString().orEmpty()
    return summary.ifBlank { artifact.ref }
}

private fun analysisDownloadUrl(baseUrl: String, reportId: String, kind: String): String {
    return "$baseUrl/yts/results/analysis/${Uri.encode(reportId)}/$kind"
}
