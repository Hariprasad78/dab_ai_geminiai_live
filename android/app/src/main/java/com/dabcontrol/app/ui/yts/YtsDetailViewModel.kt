package com.dabcontrol.app.ui.yts

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.repo.YtsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

@HiltViewModel
class YtsDetailViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val ytsRepository: YtsRepository
) : ViewModel() {
    private val commandId: String = checkNotNull(savedStateHandle["commandId"])
    private val _uiState = MutableStateFlow(YtsDetailUiState())
    val uiState: StateFlow<YtsDetailUiState> = _uiState.asStateFlow()
    private var pollJob: Job? = null

    init {
        refresh()
        startPolling()
    }

    fun onPromptInputChange(value: String) {
        _uiState.value = _uiState.value.copy(promptInput = value)
    }

    fun refresh() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            when (val result = ytsRepository.fetchLiveCommandState(commandId)) {
                is ApiResult.Success -> _uiState.value = _uiState.value.copy(isLoading = false, data = result.data)
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(isLoading = false, error = "HTTP ${result.code}: ${result.message}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(isLoading = false, error = "Network error: ${result.throwable.message}")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(isLoading = false, error = "Unknown error: ${result.throwable.message}")
            }
        }
    }

    fun sendPromptResponse() {
        val text = _uiState.value.promptInput.trim()
        if (text.isEmpty()) return
        viewModelScope.launch {
            when (val result = ytsRepository.respondToPrompt(commandId, text)) {
                is ApiResult.Success -> {
                    _uiState.value = _uiState.value.copy(promptInput = "")
                    refresh()
                }
                is ApiResult.HttpError -> _uiState.value = _uiState.value.copy(error = "Respond failed: HTTP ${result.code}")
                is ApiResult.NetworkError -> _uiState.value = _uiState.value.copy(error = "Respond failed: network")
                is ApiResult.UnknownError -> _uiState.value = _uiState.value.copy(error = "Respond failed")
            }
        }
    }

    fun stopCommand() {
        viewModelScope.launch {
            ytsRepository.stopLiveCommand(commandId)
            refresh()
        }
    }

    private fun startPolling() {
        pollJob?.cancel()
        pollJob = viewModelScope.launch {
            while (isActive) {
                val status = _uiState.value.data?.status
                if (status != null && status !in terminalStates) {
                    refresh()
                }
                delay(2000)
            }
        }
    }

    override fun onCleared() {
        pollJob?.cancel()
        super.onCleared()
    }

    companion object {
        private val terminalStates = setOf("completed", "stopped", "failed")
    }
}
