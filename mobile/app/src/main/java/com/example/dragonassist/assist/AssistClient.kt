package com.example.dragonassist.assist

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import org.json.JSONObject
import java.io.Closeable
import java.util.UUID
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeout

/** The service refused the request, or the connection failed. */
class AssistException(message: String, val code: String = "client") : Exception(message)

/** A stored answer for one question. */
data class AssistAnswer(
    val requestId: String,
    val videoId: String,
    val answer: String,
    /** Which backend produced it — "cloud", "npu", "mock". */
    val route: String,
    /**
     * Citations, as raw JSON objects.
     *
     * The laptop types these as objects but nothing populates them yet, so this is
     * reliably empty today and the element shape is not settled. Kept as raw JSON
     * rather than guessing at keys that may never exist.
     */
    val sources: List<String>,
)

/**
 * Client for the laptop's phone query service (`stream_transport`, `/v1/phone`).
 *
 * Unlike [com.example.dragonassist.vlm.VlmClient], nothing is uploaded here: the laptop
 * already holds the video's accumulated text, published from the board as it watches. A
 * question names a `videoId` and comes back answered from that context.
 *
 * The important property is durability. The laptop persists every query and its result
 * before acknowledging, so an answer survives the phone losing its connection — which on
 * a tailnet over mobile data is routine, and a question can take a while to answer. That
 * makes [askDurable] the method to prefer: it submits, and if the socket dies while the
 * laptop is thinking, it reconnects and collects the stored result by `requestId` rather
 * than asking again. Reusing a `requestId` is explicitly safe; the laptop treats a repeat
 * submission as the same query, not a new one.
 */
class AssistClient(private val config: AssistConfig) : Closeable {

    private val http = OkHttpClient.Builder()
        // Answers involve a cloud model behind a router and are not quick; the socket
        // must not time out underneath one. Pings keep the link alive meanwhile.
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .connectTimeout(15, TimeUnit.SECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build()

    private var socket: WebSocket? = null
    private var inbox = Channel<Frame>(Channel.UNLIMITED)

    private sealed interface Frame {
        data class Text(val json: JSONObject) : Frame
        data class Failure(val cause: Throwable) : Frame
        data class Closed(val reason: String) : Frame
    }

    /** Connects and completes the `phone_start` handshake. */
    suspend fun connect(deviceId: String): JSONObject = withTimeout(CONNECT_TIMEOUT_MS) {
        // Bind this connection's listener to *this* channel rather than to the property.
        // A socket abandoned by a reconnect can still deliver onFailure/onClosed after the
        // replacement is live; without this it would inject a stale failure into the new
        // connection's inbox and knock over the retry it was meant to enable.
        val channel = Channel<Frame>(Channel.UNLIMITED)
        inbox = channel
        val request = Request.Builder().url(config.url).build()

        suspendCancellableCoroutine { cont ->
            val ws = http.newWebSocket(request, object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    if (cont.isActive) cont.resume(Unit)
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    runCatching { JSONObject(text) }
                        .onSuccess { channel.trySend(Frame.Text(it)) }
                        .onFailure { Log.w(TAG, "unparseable frame: ${text.take(120)}") }
                }

                override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                    // The service is JSON-only; ignore rather than fail.
                    Log.w(TAG, "unexpected binary frame (${bytes.size} bytes)")
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    channel.trySend(Frame.Failure(t))
                    if (cont.isActive) cont.resumeWithException(
                        AssistException("Could not reach ${config.url}: ${t.message}", "connect")
                    )
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    channel.trySend(Frame.Closed(reason.ifEmpty { "closed ($code)" }))
                }
            })
            socket = ws
            cont.invokeOnCancellation { ws.cancel() }
        }

        send(
            JSONObject()
                .put("type", "phone_start")
                .put("protocol", PROTOCOL_VERSION)
                .put("device_id", safeId(deviceId))
        )
        val ready = await("phone_ready")
        Log.i(TAG, "ready as ${ready.optString("device_id")}")
        ready
    }

    /**
     * Submits a question and returns immediately with the server's acknowledgement.
     *
     * The answer arrives later, pushed on this socket — [awaitResult] collects it. Splitting
     * submit from collect is what lets a caller survive a disconnect in between.
     *
     * @param requestId stable id for this question; reuse it verbatim when retrying.
     */
    suspend fun submit(requestId: String, videoId: String, question: String): String {
        require(question.isNotBlank()) { "question is blank" }
        send(
            JSONObject()
                .put("type", "query")
                .put("request_id", safeId(requestId))
                .put("video_id", safeId(videoId))
                .put("question", question)
        )
        return await("query_ack").optString("status", "pending")
    }

    /** Waits for the pushed result of an already-submitted [requestId]. */
    suspend fun awaitResult(requestId: String): AssistAnswer {
        while (true) {
            val frame = receive()
            when (frame.optString("type")) {
                "query_result" ->
                    if (frame.optString("request_id") == requestId) return frame.toAnswer()
                "error" -> throw AssistException(
                    frame.optString("message"), frame.optString("code", "query"),
                )
            }
        }
    }

    /**
     * Asks for the current state of a request, which is how a reconnected phone collects an
     * answer computed while it was away. Returns null if the laptop is still working on it.
     */
    suspend fun poll(requestId: String): AssistAnswer? {
        send(JSONObject().put("type", "query_status").put("request_id", safeId(requestId)))
        while (true) {
            val frame = receive()
            when (frame.optString("type")) {
                "query_result" -> return frame.toAnswer()
                "query_ack" -> return null      // still pending or claimed
                "error" -> throw AssistException(
                    frame.optString("message"), frame.optString("code", "query"),
                )
            }
        }
    }

    /**
     * Submits a question and returns its answer, reconnecting across dropped sockets.
     *
     * This is the method to use. The laptop's durability guarantee only pays off if the
     * client actually reconnects and asks again by id, which is what this does: on a
     * connection failure it reopens the socket and polls [requestId], and because the id is
     * stable the laptop returns the original answer rather than starting over.
     *
     * @param onPending invoked each time the answer is not ready yet, for UI that wants to
     *   say so rather than sit silent through a long generation.
     */
    suspend fun askDurable(
        deviceId: String,
        videoId: String,
        question: String,
        requestId: String = newRequestId(),
        attempts: Int = RECONNECT_ATTEMPTS,
        onPending: () -> Unit = {},
    ): AssistAnswer {
        var submitted = false
        var lastFailure: AssistException? = null

        repeat(attempts) { attempt ->
            try {
                if (attempt > 0) {
                    closeSocket()
                    connect(deviceId)
                    // The answer may already be waiting from before the drop.
                    poll(requestId)?.let { return it }
                    onPending()
                }
                if (!submitted) {
                    submit(requestId, videoId, question)
                    submitted = true
                    onPending()
                }
                return awaitResult(requestId)
            } catch (exc: AssistException) {
                // A refusal is final; only connection trouble is worth retrying.
                if (exc.code != "connect") throw exc
                Log.w(TAG, "attempt ${attempt + 1} lost the connection: ${exc.message}")
                lastFailure = exc
            }
        }
        throw lastFailure ?: AssistException("Gave up after $attempts attempts", "connect")
    }

    /** Round-trips a ping and returns the latency in milliseconds. */
    suspend fun ping(): Long {
        val started = System.nanoTime()
        send(JSONObject().put("type", "ping"))
        await("pong")
        return (System.nanoTime() - started) / 1_000_000
    }

    private fun JSONObject.toAnswer(): AssistAnswer {
        if (optString("status") == "failed") {
            throw AssistException(optString("error").ifEmpty { "The laptop could not answer" }, "failed")
        }
        val sources = optJSONArray("sources")
        return AssistAnswer(
            requestId = optString("request_id"),
            videoId = optString("video_id"),
            answer = optString("answer"),
            route = optString("route"),
            sources = buildList {
                for (i in 0 until (sources?.length() ?: 0)) {
                    add(sources!!.opt(i)?.toString().orEmpty())
                }
            },
        )
    }

    private fun send(payload: JSONObject) {
        val ws = socket ?: throw AssistException("Not connected", "connect")
        if (!ws.send(payload.toString())) {
            throw AssistException("Could not queue message; socket is closing", "connect")
        }
    }

    /** Waits for a frame of [type], surfacing `error` frames as exceptions. */
    private suspend fun await(type: String): JSONObject {
        while (true) {
            val frame = receive()
            when (val kind = frame.optString("type")) {
                type -> return frame
                "error" -> throw AssistException(
                    frame.optString("message"), frame.optString("code", "query"),
                )
                else -> Log.d(TAG, "ignoring '$kind' while waiting for '$type'")
            }
        }
    }

    private suspend fun receive(): JSONObject =
        withTimeout(REPLY_TIMEOUT_MS) {
            when (val frame = inbox.receive()) {
                is Frame.Text -> frame.json
                is Frame.Failure -> throw AssistException(
                    "Connection failed: ${frame.cause.message}", "connect",
                )
                is Frame.Closed -> throw AssistException(
                    "Server closed the connection: ${frame.reason}", "connect",
                )
            }
        }

    private fun closeSocket() {
        socket?.close(1000, "reconnecting")
        socket = null
        inbox.close()
    }

    override fun close() {
        closeSocket()
        http.dispatcher.executorService.shutdown()
    }

    companion object {
        private const val TAG = "AssistClient"
        const val PROTOCOL_VERSION = 1
        const val CONNECT_TIMEOUT_MS = 20_000L

        // A question goes through a router and then a cloud model; far longer than a
        // typical network timeout is normal, not a symptom.
        const val REPLY_TIMEOUT_MS = 180_000L

        const val RECONNECT_ATTEMPTS = 3

        /** Fresh id for a new question. Stable across retries of the *same* question. */
        fun newRequestId(): String = "q-" + UUID.randomUUID().toString()

        /**
         * Coerces an id into what the service accepts.
         *
         * Ids name files and database rows on the laptop, so it enforces
         * `[A-Za-z0-9][A-Za-z0-9._-]{0,127}` and closes the socket on anything else.
         * Android device and model names routinely contain spaces, so sanitising here
         * turns a whole class of confusing disconnects into a harmless rewrite.
         */
        fun safeId(raw: String): String {
            val cleaned = raw.map { if (it.isLetterOrDigit() || it in "._-") it else '-' }
                .joinToString("")
                .trimStart('.', '-', '_')
                .take(128)
            return cleaned.ifEmpty { "device" }
        }
    }
}
