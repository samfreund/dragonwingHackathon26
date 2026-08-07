package com.example.dragonassist.net

import android.util.Log
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import org.json.JSONObject
import java.io.Closeable
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeout

/** A transport-level failure, or a refusal from the far end. */
class TransportException(message: String, val code: String = "transport") : Exception(message)

/**
 * A JSON-over-WebSocket connection with a suspending, request/reply shape.
 *
 * Both DragonAssist services speak the same *style* — JSON text frames, replies identified
 * by a `type` field — while differing in their handshakes and message names. This holds
 * the part that is genuinely common: OkHttp's callback threading funnelled into a channel
 * so callers can simply `await("ack")` without dealing with listener callbacks.
 *
 * `VlmClient` predates this and keeps its own copy; it is tested against the real board and
 * not worth destabilising. New clients should build on this.
 */
class JsonWebSocket private constructor(
    private val http: OkHttpClient,
    private val socket: WebSocket,
    private val inbox: Channel<Frame>,
    private val replyTimeoutMs: Long,
    private val label: String,
) : Closeable {

    sealed interface Frame {
        data class Text(val json: JSONObject) : Frame
        data class Failure(val cause: Throwable) : Frame
        data class Closed(val reason: String) : Frame
    }

    /** Bytes queued by OkHttp but not yet written — used for upload backpressure. */
    fun queueSize(): Long = socket.queueSize()

    fun send(payload: JSONObject) {
        Wire.out(label, payload.toString())
        if (!socket.send(payload.toString())) {
            throw TransportException("Socket is closing; message not queued", "closed")
        }
    }

    fun sendBinary(bytes: ByteString) {
        if (!socket.send(bytes)) {
            throw TransportException("Socket is closing; bytes not queued", "closed")
        }
    }

    /** Next frame of any type, surfacing failures and closes as exceptions. */
    suspend fun receive(): JSONObject = withTimeout(replyTimeoutMs) {
        when (val frame = inbox.receive()) {
            is Frame.Text -> frame.json
            is Frame.Failure -> throw TransportException(
                "Connection failed: ${frame.cause.message}", "connect",
            )
            is Frame.Closed -> throw TransportException(
                "Server closed the connection: ${frame.reason}", "closed",
            )
        }
    }

    /**
     * Waits for a frame whose `type` is [type].
     *
     * Frames named in [ignore] are dropped silently — progress updates and the like.
     * Anything else is logged and skipped rather than treated as an error, since both
     * protocols are free to add message types we do not know about yet. An `error` frame
     * always throws, whatever we were waiting for.
     */
    suspend fun await(type: String, ignore: Set<String> = emptySet()): JSONObject {
        while (true) {
            val frame = receive()
            when (val kind = frame.optString("type")) {
                type -> return frame
                "error" -> throw TransportException(
                    frame.optString("message").ifEmpty { "Server reported an error" },
                    frame.optString("code", "error"),
                )
                in ignore -> Unit
                else -> Log.d(TAG, "ignoring '$kind' while waiting for '$type'")
            }
        }
    }

    override fun close() {
        runCatching { socket.close(1000, "bye") }
        inbox.close()
        runCatching { http.dispatcher.executorService.shutdown() }
    }

    companion object {
        private const val TAG = "JsonWebSocket"

        suspend fun connect(
            url: String,
            label: String = url.substringAfterLast('/'),
            connectTimeoutMs: Long = 20_000,
            replyTimeoutMs: Long = 120_000,
            maxMessageBytes: Long = 1024 * 1024,
        ): JsonWebSocket {
            val http = OkHttpClient.Builder()
                // No read timeout: an answer may take minutes when a query is queued
                // behind another, or escalates to a cloud model. Pings detect a dead peer.
                .readTimeout(0, TimeUnit.MILLISECONDS)
                .connectTimeout(15, TimeUnit.SECONDS)
                .pingInterval(20, TimeUnit.SECONDS)
                .build()

            val inbox = Channel<Frame>(Channel.UNLIMITED)

            val socket = withTimeout(connectTimeoutMs) {
                suspendCancellableCoroutine { cont ->
                    val ws = http.newWebSocket(
                        Request.Builder().url(url).build(),
                        object : WebSocketListener() {
                            override fun onOpen(webSocket: WebSocket, response: Response) {
                                if (cont.isActive) cont.resume(webSocket)
                            }

                            override fun onMessage(webSocket: WebSocket, text: String) {
                                Wire.inbound(label, text)
                                runCatching { JSONObject(text) }
                                    .onSuccess { inbox.trySend(Frame.Text(it)) }
                                    .onFailure {
                                        Log.w(TAG, "unparseable frame: ${text.take(120)}")
                                    }
                            }

                            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                                Log.w(TAG, "unexpected binary frame (${bytes.size} bytes)")
                            }

                            override fun onFailure(
                                webSocket: WebSocket,
                                t: Throwable,
                                response: Response?,
                            ) {
                                inbox.trySend(Frame.Failure(t))
                                if (cont.isActive) cont.resumeWithException(
                                    TransportException(
                                        "Could not reach $url: ${t.message}", "connect",
                                    )
                                )
                            }

                            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                                inbox.trySend(Frame.Closed(reason.ifEmpty { "closed ($code)" }))
                            }
                        },
                    )
                    cont.invokeOnCancellation { ws.cancel() }
                }
            }

            require(maxMessageBytes > 0)
            return JsonWebSocket(http, socket, inbox, replyTimeoutMs, label)
        }
    }
}
