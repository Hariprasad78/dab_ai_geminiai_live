package com.dabcontrol.app.data.preferences

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import dagger.hilt.android.qualifiers.ApplicationContext
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore(name = "dab_control_settings")

@Singleton
class ApiSettingsStore @Inject constructor(
    @ApplicationContext private val context: Context
) {
    private val apiBaseUrlKey: Preferences.Key<String> = stringPreferencesKey("api_base_url")

    val apiBaseUrl: Flow<String> = context.dataStore.data.map { prefs ->
        prefs[apiBaseUrlKey] ?: "http://10.0.2.2:8000"
    }

    suspend fun saveApiBaseUrl(value: String) {
        context.dataStore.edit { prefs ->
            prefs[apiBaseUrlKey] = value.trim().trimEnd('/')
        }
    }
}
