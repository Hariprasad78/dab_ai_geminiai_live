package com.dabcontrol.app.ui.yts

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.FileProvider
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import java.io.File

@Composable
fun YtsArtifactScreen(
    modifier: Modifier = Modifier,
    viewModel: YtsArtifactViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text(state.title, style = MaterialTheme.typography.headlineSmall)
        Text("Command: ${state.commandId}", color = MaterialTheme.colorScheme.onSurfaceVariant)
        Text(state.status, color = MaterialTheme.colorScheme.onSurfaceVariant)
        state.error?.let { Text(it, color = MaterialTheme.colorScheme.error) }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = viewModel::refresh) { Text("Redownload") }
            OutlinedButton(
                enabled = state.localPath.isNotBlank(),
                onClick = {
                    val file = File(state.localPath)
                    val uri = FileProvider.getUriForFile(
                        context,
                        "${context.packageName}.fileprovider",
                        file
                    )
                    val intent = Intent(Intent.ACTION_VIEW).apply {
                        setDataAndType(uri, state.mimeType)
                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    context.startActivity(intent)
                }
            ) { Text("Open File") }
            OutlinedButton(
                enabled = state.remoteUrl.isNotBlank(),
                onClick = {
                    context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(state.remoteUrl)))
                }
            ) { Text("Open Remote") }
        }

        when (state.artifactType) {
            YtsArtifactViewModel.TYPE_RESULT -> {
                Surface(modifier = Modifier.fillMaxWidth().weight(1f, true)) {
                    Column(
                        modifier = Modifier
                            .fillMaxSize()
                            .verticalScroll(rememberScrollState())
                            .padding(12.dp)
                    ) {
                        if (state.isLoading) {
                            CircularProgressIndicator()
                        } else {
                            Text(
                                text = state.textContent.ifBlank { "(empty result file)" },
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }
                }
            }
            YtsArtifactViewModel.TYPE_REPORT_HTML -> {
                AndroidView(
                    modifier = Modifier.fillMaxWidth().weight(1f, true),
                    factory = { ctx ->
                        WebView(ctx).apply {
                            settings.javaScriptEnabled = true
                            webViewClient = WebViewClient()
                        }
                    },
                    update = { webView ->
                        if (state.htmlContent.isNotBlank()) {
                            webView.loadDataWithBaseURL(
                                state.remoteUrl,
                                state.htmlContent,
                                "text/html",
                                "utf-8",
                                null
                            )
                        }
                    }
                )
            }
            else -> {
                Surface(modifier = Modifier.fillMaxWidth().weight(1f, true)) {
                    PdfArtifactPanel(
                        localPath = state.localPath,
                        isLoading = state.isLoading
                    )
                }
            }
        }
    }
}

@Composable
private fun PdfArtifactPanel(
    localPath: String,
    isLoading: Boolean
) {
    val pages by produceState<List<Bitmap>>(initialValue = emptyList(), localPath) {
        value = if (localPath.isBlank()) {
            emptyList()
        } else {
            renderPdfPages(localPath)
        }
    }

    when {
        isLoading -> {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                CircularProgressIndicator()
                Text("Rendering PDF report...")
            }
        }
        pages.isEmpty() -> {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text("No PDF preview available yet.")
                Text(localPath.ifBlank { "No local file yet." }, fontFamily = FontFamily.Monospace)
            }
        }
        else -> {
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                itemsIndexed(pages) { index, bitmap ->
                    Column(
                        modifier = Modifier.padding(horizontal = 12.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        Text("Page ${index + 1}", style = MaterialTheme.typography.titleSmall)
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = "PDF page ${index + 1}",
                            modifier = Modifier.fillMaxWidth()
                        )
                    }
                }
            }
        }
    }
}

private fun renderPdfPages(localPath: String): List<Bitmap> {
    val file = File(localPath)
    if (!file.exists()) return emptyList()

    val descriptor = ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY)
    val renderer = PdfRenderer(descriptor)
    return try {
        buildList {
            for (pageIndex in 0 until renderer.pageCount) {
                renderer.openPage(pageIndex).use { page ->
                    val width = page.width * 2
                    val height = page.height * 2
                    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
                    bitmap.eraseColor(android.graphics.Color.WHITE)
                    page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                    add(bitmap)
                }
            }
        }
    } finally {
        renderer.close()
        descriptor.close()
    }
}
