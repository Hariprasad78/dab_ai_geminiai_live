package com.dabcontrol.app.ui.yts

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.dabcontrol.app.data.api.ApiResult
import com.dabcontrol.app.data.repo.YtsRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

@HiltViewModel
class YtsListViewModel @Inject constructor(
    private val ytsRepository: YtsRepository
) : ViewModel() {
    private val _uiState = MutableStateFlow(YtsListUiState())
    val uiState: StateFlow<YtsListUiState> = _uiState.asStateFlow()

    init {
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isLoading = true, error = null)
            when (val result = ytsRepository.fetchLiveCommands()) {
                is ApiResult.Success -> _uiState.value = YtsListUiState(items = result.data)
                is ApiResult.HttpError -> _uiState.value = YtsListUiState(error = "HTTP ${result.code}: ${result.message}")
                is ApiResult.NetworkError -> _uiState.value = YtsListUiState(error = "Network error: ${result.throwable.message}")
                is ApiResult.UnknownError -> _uiState.value = YtsListUiState(error = "Unknown error: ${result.throwable.message}")
            }
        }
    }
}
