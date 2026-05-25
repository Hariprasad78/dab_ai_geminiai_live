package com.dabcontrol.app.ui.yts

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.lazy.items
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.dabcontrol.app.data.api.YtsLiveCommandSummaryDto
import com.dabcontrol.app.ui.common.PremiumBackdrop
import com.dabcontrol.app.ui.common.SectionLabel

private enum class ResultsTab(val label: String) {
    OVERVIEW("Overview"),
    SESSION_TABLE("Session Table"),
    ARTIFACT_MATRIX("Artifacts"),
    ANALYSIS("Analysis")
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
                ResultsTab.ARTIFACT_MATRIX -> ArtifactMatrixTab(
                    items = filteredItems,
                    onOpenCommand = onOpenCommand,
                    onOpenReport = onOpenReport,
                    onOpenArtifact = onOpenArtifact
                )
                ResultsTab.ANALYSIS -> AnalysisTab(
                    state = state,
                    onAnalyze = viewModel::analyzeArtifacts
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
    LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item {
            TableHeaderRow()
        }
        items(items, key = { it.command_id }) { item ->
            SessionTableRow(
                item = item,
                onOpenCommand = onOpenCommand,
                onOpenReport = onOpenReport,
                onOpenArtifact = onOpenArtifact
            )
        }
    }
}

@Composable
private fun ArtifactMatrixTab(
    items: List<YtsLiveCommandSummaryDto>,
    onOpenCommand: (String) -> Unit,
    onOpenReport: (String) -> Unit,
    onOpenArtifact: (String, String) -> Unit
) {
    LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item {
            ElevatedCard(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text("Artifact Matrix", style = MaterialTheme.typography.titleLarge)
                    Text(
                        "Each row shows whether the backend has produced the expected evidence bundle for that session.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
        items(items, key = { it.command_id }) { item ->
            ElevatedCard(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(item.command ?: item.command_id, fontWeight = FontWeight.SemiBold)
                            Text(item.command_id, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                        StatusBadge(item.status)
                    }
                    ArtifactStatusRow("Structured Result", item.result_file_name) {
                        onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_RESULT)
                    }
                    ArtifactStatusRow("HTML Summary", item.report_html_name) {
                        onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_REPORT_HTML)
                    }
                    ArtifactStatusRow("PDF Summary", item.report_pdf_name) {
                        onOpenArtifact(item.command_id, YtsArtifactViewModel.TYPE_REPORT_PDF)
                    }
                    ArtifactStatusRow("Session Video", item.video_file_name)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        FilledTonalButton(onClick = { onOpenCommand(item.command_id) }) {
                            Text("Open Session")
                        }
                        if (hasReport(item)) {
                            OutlinedButton(onClick = { onOpenReport(item.command_id) }) {
                                Text("Open Report")
                            }
                        }
                    }
                }
            }
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

@Composable
private fun AnalysisTab(
    state: YtsListUiState,
    onAnalyze: (List<String>) -> Unit
) {
    var selectedRefs by rememberSaveable { mutableStateOf(setOf<String>()) }
    
    LazyColumn(verticalArrangement = Arrangement.spacedBy(14.dp)) {
        item {
            ElevatedCard(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Text("Result Analysis Studio", style = MaterialTheme.typography.titleLarge)
                    Text(
                        "Select past result artifacts to generate an AI-driven triage report. This uses the backend's result analysis engine.",
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    
                    if (state.artifacts.isEmpty()) {
                        Text("No artifacts available for analysis.", style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
        
        if (state.artifacts.isNotEmpty()) {
            items(state.artifacts, key = { it.ref }) { artifact ->
                val isSelected = selectedRefs.contains(artifact.ref)
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable {
                            selectedRefs = if (isSelected) {
                                selectedRefs - artifact.ref
                            } else {
                                selectedRefs + artifact.ref
                            }
                        },
                    color = if (isSelected) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                    shape = MaterialTheme.shapes.small
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            Text(artifact.label, style = MaterialTheme.typography.bodyMedium, fontWeight = FontWeight.SemiBold)
                            Text("${artifact.command_id} • ${artifact.type}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
        }
        
        item {
            ElevatedCard(modifier = Modifier.fillMaxWidth()) {
                Column(
                    modifier = Modifier.padding(18.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.End
                    ) {
                        FilledTonalButton(
                            enabled = selectedRefs.isNotEmpty() && !state.analysisLoading,
                            onClick = { onAnalyze(selectedRefs.toList()) }
                        ) {
                            if (state.analysisLoading) {
                                CircularProgressIndicator(modifier = Modifier.padding(end = 8.dp))
                            }
                            Text("Analyze Selected (${selectedRefs.size})")
                        }
                    }
                    
                    if (state.analysisReportText.isNotBlank()) {
                        HorizontalDivider()
                        Text("Analysis Result", style = MaterialTheme.typography.titleMedium)
                        Surface(
                            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
                            shape = MaterialTheme.shapes.medium,
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                text = state.analysisReportText,
                                modifier = Modifier.padding(12.dp),
                                style = MaterialTheme.typography.bodyMedium
                            )
                        }
                    }
                }
            }
        }
    }
}
