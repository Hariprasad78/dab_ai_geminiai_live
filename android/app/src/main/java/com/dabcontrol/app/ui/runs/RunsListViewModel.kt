package com.dabcontrol.app.ui.runs

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.repo.RunsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

@HiltViewModel
class RunsListViewModel @Inject constructor(
    private val runsRepository: RunsRepository
) : ViewModel() {
    private val _uiState = MutableStateFlow(RunsListUiState())
    val uiState: StateFlow<RunsListUiState> = _uiState.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            when (val result = runsRepository.fetchRuns()) {
                is ApiResult.Success -> _uiState.value = RunsListUiState(items = result.data)
                is ApiResult.HttpError -> _uiState.value = RunsListUiState(error = "HTTP ${result.code}: ${result.message}")
                is ApiResult.NetworkError -> _uiState.value = RunsListUiState(error = "Network error: ${result.throwable.message}")
                is ApiResult.UnknownError -> _uiState.value = RunsListUiState(error = "Unknown error: ${result.throwable.message}")
            }
        }
    }
}
