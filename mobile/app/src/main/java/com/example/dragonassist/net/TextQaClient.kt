package com.example.dragonassist.net

import android.util.Log
import org.json.JSONObject
import java.io.Closeable

/**
 * An answer from the laptop, with the routing decision that produced it.
 *
 * [route] is what makes the system legible: the same question can be answered in ~40 ms by
 * DistilBERT on the X Elite NPU, or escalated to Llama-3.1-8B in the cloud. Surfacing it
 * turns an invisible optimisation into something the user can see.
 */
data class TextAnswer(
    val answer: String,
    val route: String?,
    val sources: List<String>,
    val requestId: String,
)

/**
 * Asks questions about a session's accumulated context, over `/v1/phone`.
 *
 * Unlike the board's VLM socket, this is request/response rather than token streaming, and
 * results are *durable*: the server stores each answer against its `request_id`. If the
 * connection dies while the laptop is still thinking, reconnecting and sending
 * `query_status` with the same id retrieves the stored result.
 *
 * That matters here more than it sounds. Tailscale on the phone has already been observed
 * reporting "connected" while silently out of sync, and a three-minute answer was lost to
 * a client timeout with no way to recover it. This protocol makes that recoverable.
 */
class TextQaClient(
    private val url: String,
    private val token: String,
    private val deviceId: String,
) : Closeable {

    private var socket: JsonWebSocket? = null

    /** Connects and completes the handshake. Idempotent. */
    suspend fun connect() {
        if (socket != null) return
        val ws = JsonWebSocket.connect(url)
        socket = ws
        ws.send(
            JSONObject()
                .put("type", "phone_start")
                .put("protocol", PROTOCOL_VERSION)
                .put("token", token)
                .put("device_id", deviceId)
        )
        ws.await("phone_ready")
        Log.i(TAG, "phone channel ready as $deviceId")
    }

    /**
     * Submits [question] against [sessionId]'s context and waits for the answer.
     *
     * @param requestId must be stable across retries of the *same* question — that is what
     *   makes resubmission idempotent rather than queueing the work twice.
     */
    suspend fun ask(
        sessionId: String,
        question: String,
        requestId: String,
        onQueued: () -> Unit = {},
    ): TextAnswer {
        require(question.isNotBlank()) { "question is blank" }
        connect()
        val ws = socket ?: throw TransportException("Not connected", "closed")

        ws.send(
            JSONObject()
                .put("type", "query")
                .put("request_id", requestId)
                .put("video_id", sessionId)
                .put("question", question)
        )

        val ack = ws.await("query_ack")
        if (ack.optString("status") == "pending") onQueued()
        Log.i(TAG, "query $requestId acknowledged (${ack.optString("status")})")

        return parse(ws.await("query_result"))
    }

    /**
     * Retrieves a previously submitted answer after a reconnect.
     *
     * The recovery path for a dropped connection: reconnect, ask by id, get the stored
     * result rather than re-running the model.
     */
    suspend fun retrieve(requestId: String): TextAnswer {
        connect()
        val ws = socket ?: throw TransportException("Not connected", "closed")
        ws.send(JSONObject().put("type", "query_status").put("request_id", requestId))
        return parse(ws.await("query_result", ignore = setOf("query_ack")))
    }

    private fun parse(result: JSONObject): TextAnswer {
        val status = result.optString("status")
        if (status != "complete") {
            throw TransportException(
                result.optString("error").ifEmpty { "Query $status" },
                status.ifEmpty { "failed" },
            )
        }
        val sources = result.optJSONArray("sources")?.let { array ->
            (0 until array.length()).mapNotNull { array.optString(it).ifEmpty { null } }
        }.orEmpty()

        return TextAnswer(
            answer = result.optString("answer"),
            route = result.optString("route").ifEmpty { null },
            sources = sources,
            requestId = result.optString("request_id"),
        )
    }

    override fun close() {
        runCatching { socket?.close() }
        socket = null
    }

    private companion object {
        const val TAG = "TextQaClient"
        const val PROTOCOL_VERSION = 1
    }
}
