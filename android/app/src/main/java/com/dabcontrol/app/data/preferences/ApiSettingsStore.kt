package com.dabcontrol.app.data.preferences

import android.content.Context
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

@Singleton
class ApiSettingsStore @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val sharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val apiBaseUrlState = MutableStateFlow(
        normalizeBaseUrl(
            sharedPreferences.getString(KEY_API_BASE_URL, DEFAULT_API_BASE_URL) ?: DEFAULT_API_BASE_URL
        )
    )
    private val selectedDeviceIdState = MutableStateFlow(
        sharedPreferences.getString(KEY_SELECTED_DEVICE_ID, "")?.trim().orEmpty()
    )

    val apiBaseUrl: Flow<String> = apiBaseUrlState.asStateFlow()
    val selectedDeviceId: Flow<String> = selectedDeviceIdState.asStateFlow()

    suspend fun saveApiBaseUrl(value: String) {
        val normalized = normalizeBaseUrl(value)
        sharedPreferences.edit()
            .putString(KEY_API_BASE_URL, normalized)
            .apply()
        apiBaseUrlState.value = normalized
    }

    suspend fun saveSelectedDeviceId(value: String) {
        val normalized = value.trim()
        sharedPreferences.edit()
            .putString(KEY_SELECTED_DEVICE_ID, normalized)
            .apply()
        selectedDeviceIdState.value = normalized
    }

    private fun normalizeBaseUrl(value: String): String {
        val trimmed = value.trim()
        if (trimmed.isEmpty()) return DEFAULT_API_BASE_URL
        val withScheme = if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
            trimmed
        } else {
            "http://$trimmed"
        }
        return withScheme.trimEnd('/')
    }

    companion object {
        private const val PREFS_NAME = "dab_control_settings"
        private const val KEY_API_BASE_URL = "api_base_url"
        private const val KEY_SELECTED_DEVICE_ID = "selected_device_id"
        private const val DEFAULT_API_BASE_URL = "http://10.99.57.66:8081"
    }
}
