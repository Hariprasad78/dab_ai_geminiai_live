package com.dabcontrol.app.ui.yts

import android.content.Context
import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.preferences.ApiSettingsStore
import dagger.hilt.android.lifecycle.HiltViewModel
import dagger.hilt.android.qualifiers.ApplicationContext
import java.io.File
import java.net.URLEncoder
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import okhttp3.OkHttpClient
import okhttp3.Request

data class YtsArtifactUiState(
    val isLoading: Boolean = false,
    val title: String = "YTS Artifact",
    val commandId: String = "",
    val artifactType: String = "result",
    val fileName: String = "",
    val localPath: String = "",
    val mimeType: String = "text/plain",
    val textContent: String = "",
    val htmlContent: String = "",
    val remoteUrl: String = "",
    val status: String = "Preparing artifact...",
    val error: String? = null
)

@HiltViewModel
class YtsArtifactViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val apiSettingsStore: ApiSettingsStore,
    private val okHttpClient: OkHttpClient,
    @ApplicationContext private val appContext: Context
) : ViewModel() {
    private val commandId: String = checkNotNull(savedStateHandle["commandId"])
    private val artifactType: String = checkNotNull(savedStateHandle["artifactType"])

    private val _uiState = MutableStateFlow(
        YtsArtifactUiState(
            commandId = commandId,
            artifactType = artifactType,
            title = titleFor(artifactType)
        )
    )
    val uiState: StateFlow<YtsArtifactUiState> = _uiState.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null, status = "Downloading artifact...")
            val baseUrl = apiSettingsStore.apiBaseUrl.first().trimEnd('/')
            val remoteUrl = "$baseUrl/yts/command/live/${URLEncoder.encode(commandId, "UTF-8")}/${endpointFor(artifactType)}"
            try {
                val request = Request.Builder().url(remoteUrl).build()
                okHttpClient.newCall(request).execute().use { response ->
                    if (!response.isSuccessful) {
                        _uiState.value = _uiState.value.copy(
                            isLoading = false,
                            remoteUrl = remoteUrl,
                            error = "HTTP ${response.code}",
                            status = "Artifact download failed."
                        )
                        return@use
                    }
                    val bytes = response.body?.bytes() ?: ByteArray(0)
                    val fileName = buildFileName(commandId, artifactType, response.header("Content-Type"))
                    val targetFile = File(appContext.cacheDir, fileName)
                    targetFile.writeBytes(bytes)
                    val text = if (artifactType == TYPE_RESULT) bytes.toString(Charsets.UTF_8) else ""
                    val html = if (artifactType == TYPE_REPORT_HTML) bytes.toString(Charsets.UTF_8) else ""
                    _uiState.value = _uiState.value.copy(
                        isLoading = false,
                        fileName = fileName,
                        localPath = targetFile.absolutePath,
                        mimeType = mimeTypeFor(artifactType, response.header("Content-Type")),
                        textContent = text,
                        htmlContent = html,
                        remoteUrl = remoteUrl,
                        status = "Artifact ready."
                    )
                }
            } catch (t: Throwable) {
                _uiState.value = _uiState.value.copy(
                    isLoading = false,
                    remoteUrl = remoteUrl,
                    error = t.message ?: "Unknown error",
                    status = "Artifact download failed."
                )
            }
        }
    }

    private fun buildFileName(commandId: String, artifactType: String, contentType: String?): String {
        val suffix = when (artifactType) {
            TYPE_RESULT -> "result.json"
            TYPE_REPORT_HTML -> "report.html"
            TYPE_REPORT_PDF -> "report.pdf"
            else -> "artifact.bin"
        }
        return "yts-$commandId-$suffix"
    }

    private fun mimeTypeFor(artifactType: String, responseType: String?): String {
        return when (artifactType) {
            TYPE_RESULT -> "application/json"
            TYPE_REPORT_HTML -> "text/html"
            TYPE_REPORT_PDF -> "application/pdf"
            else -> responseType ?: "application/octet-stream"
        }
    }

    private fun endpointFor(artifactType: String): String {
        return when (artifactType) {
            TYPE_RESULT -> "result"
            TYPE_REPORT_HTML -> "report-html"
            TYPE_REPORT_PDF -> "report"
            else -> "result"
        }
    }

    private fun titleFor(artifactType: String): String {
        return when (artifactType) {
            TYPE_RESULT -> "YTS Result"
            TYPE_REPORT_HTML -> "YTS Summary"
            TYPE_REPORT_PDF -> "YTS PDF"
            else -> "YTS Artifact"
        }
    }

    companion object {
        const val TYPE_RESULT = "result"
        const val TYPE_REPORT_HTML = "report_html"
        const val TYPE_REPORT_PDF = "report_pdf"
    }
}
