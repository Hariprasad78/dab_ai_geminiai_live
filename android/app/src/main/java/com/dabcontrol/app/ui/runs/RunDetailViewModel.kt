package com.dabcontrol.app.ui.runs

import androidx.lifecycle.SavedStateHandle
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.repo.RunsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.Job
import kotlinx.coroutines.async
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch

@HiltViewModel
class RunDetailViewModel @Inject constructor(
    savedStateHandle: SavedStateHandle,
    private val runsRepository: RunsRepository
) : ViewModel() {
    private val runId: String = checkNotNull(savedStateHandle["runId"])
    private val _uiState = MutableStateFlow(RunDetailUiState())
    val uiState: StateFlow<RunDetailUiState> = _uiState.asStateFlow()
    private var pollJob: Job? = null

    init {
        refreshAll()
        startPolling()
    }

    fun refreshAll() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            val statusDef = async { runsRepository.fetchRunStatus(runId) }
            val historyDef = async { runsRepository.fetchRunHistory(runId) }
            val aiDef = async { runsRepository.fetchAiTranscript(runId) }
            val dabDef = async { runsRepository.fetchDabTranscript(runId) }
            val explainDef = async { runsRepository.fetchExplain(runId) }
            val narrationDef = async { runsRepository.fetchNarration(runId) }

            val status = statusDef.await()
            val history = historyDef.await()
            val ai = aiDef.await()
            val dab = dabDef.await()
            val explain = explainDef.await()
            val narration = narrationDef.await()

            _uiState.value = _uiState.value.copy(
                isLoading = false,
                status = (status as? ApiResult.Success)?.data,
                actions = (history as? ApiResult.Success)?.data?.actions.orEmpty(),
                aiEvents = (ai as? ApiResult.Success)?.data?.events.orEmpty(),
                dabEvents = (dab as? ApiResult.Success)?.data?.events.orEmpty(),
                explainTimeline = (explain as? ApiResult.Success)?.data?.timeline.orEmpty(),
                diagnosisSummary = (explain as? ApiResult.Success)?.data?.diagnosis?.final_summary,
                narrationEvents = (narration as? ApiResult.Success)?.data?.events.orEmpty(),
                error = firstError(status, history, ai, dab, explain, narration)
            )
        }
    }

    private fun startPolling() {
        pollJob?.cancel()
        pollJob = viewModelScope.launch {
            while (isActive) {
                val statusResult = runsRepository.fetchRunStatus(runId)
                if (statusResult is ApiResult.Success) {
                    _uiState.value = _uiState.value.copy(status = statusResult.data)
                    if (statusResult.data.status in terminalStatuses) {
                        refreshAll()
                        break
                    }
                }
                delay(2000)
            }
        }
    }

    override fun onCleared() {
        pollJob?.cancel()
        super.onCleared()
    }

    private fun firstError(vararg results: ApiResult<*>): String? {
        for (result in results) {
            when (result) {
                is ApiResult.HttpError -> return "HTTP ${result.code}: ${result.message}"
                is ApiResult.NetworkError -> return "Network error: ${result.throwable.message}"
                is ApiResult.UnknownError -> return "Unknown error: ${result.throwable.message}"
                is ApiResult.Success -> Unit
            }
        }
        return null
    }

    companion object {
        private val terminalStatuses = setOf("DONE", "FAILED", "TIMEOUT", "ERROR", "STOPPED")
    }
}
