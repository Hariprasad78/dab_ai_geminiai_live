package com.dabcontrol.app.data.api

sealed interface ApiResult<out T> {
    data class Success<T>(val data: T) : ApiResult<T>
    data class HttpError(val code: Int, val message: String, val body: String?) : ApiResult<Nothing>
    data class NetworkError(val throwable: Throwable) : ApiResult<Nothing>
    data class UnknownError(val throwable: Throwable) : ApiResult<Nothing>
}
