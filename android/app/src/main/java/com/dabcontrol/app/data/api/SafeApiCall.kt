package com.dabcontrol.app.data.api

import java.io.IOException
import retrofit2.Response

suspend fun <T> safeApiCall(block: suspend () -> Response<T>): ApiResult<T> {
    return try {
        val response = block()
        if (response.isSuccessful) {
            val body = response.body()
            if (body != null) {
                ApiResult.Success(body)
            } else {
                ApiResult.HttpError(
                    code = response.code(),
                    message = "Response body is null",
                    body = response.errorBody()?.string()
                )
            }
        } else {
            ApiResult.HttpError(
                code = response.code(),
                message = response.message(),
                body = response.errorBody()?.string()
            )
        }
    } catch (io: IOException) {
        ApiResult.NetworkError(io)
    } catch (t: Throwable) {
        ApiResult.UnknownError(t)
    }
}
