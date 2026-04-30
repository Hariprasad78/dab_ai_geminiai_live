package com.dabcontrol.app.ui.yts

import android.content.Intent
import android.net.Uri
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.weight
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@Composable
fun YtsReportScreen(
    modifier: Modifier = Modifier,
    viewModel: YtsReportViewModel = hiltViewModel()
) {
    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val context = LocalContext.current

    Column(
        modifier = modifier.fillMaxSize().padding(10.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = viewModel::refresh) { Text("Refresh View") }
            Button(
                enabled = state.url.isNotBlank(),
                onClick = {
                    val openUrl = "${state.url}?t=${System.currentTimeMillis()}"
                    context.startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(openUrl)))
                }
            ) { Text("Open Browser") }
        }
        Text(if (state.url.isBlank()) "Preparing report URL..." else state.url)

        AndroidView(
            modifier = Modifier.fillMaxWidth().weight(1f, fill = true),
            factory = { ctx ->
                WebView(ctx).apply {
                    settings.javaScriptEnabled = true
                    webViewClient = WebViewClient()
                }
            },
            update = { webView ->
                if (state.url.isNotBlank()) {
                    webView.loadUrl("${state.url}?t=${state.refreshToken}")
                }
            }
        )
    }
}
