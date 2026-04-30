package com.dabcontrol.app.ui.yts

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.preferences.ApiSettingsStore
import dagger.hilt.android.lifecycle.HiltViewModel
import java.net.URLEncoder
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

data class YtsReportUiState(
    val url: String = "",
    val refreshToken: Long = System.currentTimeMillis()
)

@HiltViewModel
class YtsReportViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val apiSettingsStore: ApiSettingsStore
) : ViewModel() {
    private val commandId: String = checkNotNull(savedStateHandle["commandId"])
    private val _uiState = MutableStateFlow(YtsReportUiState())
    val uiState: StateFlow<YtsReportUiState> = _uiState.asStateFlow()

    init {
        viewModelScope.launch {
            val base = apiSettingsStore.apiBaseUrl.first().trimEnd('/')
            _uiState.value = _uiState.value.copy(
                url = "$base/yts/command/live/${URLEncoder.encode(commandId, "UTF-8")}/report-view"
            )
        }
    }

    fun refresh() {
        _uiState.value = _uiState.value.copy(refreshToken = System.currentTimeMillis())
    }
}
