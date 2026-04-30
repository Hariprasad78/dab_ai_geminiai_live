package com.dabcontrol.app.data.api

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject

@Serializable
data class HealthResponse(
    val status: String = "unknown",
    val mode: String? = null,
    val mock_mode: Boolean? = null,
    val version: String? = null
)

@Serializable
data class MetricsResponse(
    val metrics: JsonObject = JsonObject(emptyMap())
)

@Serializable
data class ApiErrorResponse(
    val detail: String? = null,
    val message: String? = null
)

@Serializable
data class GenericObjectResponse(
    val data: Map<String, JsonElement> = emptyMap()
)
