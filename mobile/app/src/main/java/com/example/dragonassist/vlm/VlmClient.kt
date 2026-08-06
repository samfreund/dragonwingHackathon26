package com.example.dragonassist.vlm

import android.util.Base64
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

/** The server refused, or the connection failed. */
class VlmException(message: String, val code: String = "client") : Exception(message)

/** A completed answer, with the server's own timing. */
data class VlmAnswer(
    val text: String,
    val latencySeconds: Double,
    val completionTokens: Int?,
)

/**
 * Client for the vlm-qa WebSocket server running on the IQ-9075.
 *
 * One connection is one conversation about one piece of media: upload an image, then ask
 * as many questions as you like. Answers stream back token by token — on this board the
 * first token lands seconds before the last, so [ask] reports them as they arrive rather
 * than making the user watch a spinner.
 *
 * Incoming frames are funnelled into a [Channel] so the suspending API can await specific
 * message types without the caller dealing with OkHttp's callback threading.
 */
class VlmClient(private val config: VlmConfig) : Closeable {

    private val http = OkHttpClient.Builder()
        // The board answers in ~10s and can be queued behind another request, so the
        // read timeout must be generous. Pings keep the socket alive across that.
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .connectTimeout(15, TimeUnit.SECONDS)
        .pingInterval(20, TimeUnit.SECONDS)
        .build()

    private var socket: WebSocket? = null
    private val inbox = Channel<Frame>(Channel.UNLIMITED)

    private sealed interface Frame {
        data class Text(val json: JSONObject) : Frame
        data class Failure(val cause: Throwable) : Frame
        data class Closed(val reason: String) : Frame
    }

    /** Connects, waits for `ready`, and authenticates if the server asks. */
    suspend fun connect(): JSONObject = withTimeout(CONNECT_TIMEOUT_MS) {
        val request = Request.Builder().url(config.url).build()

        suspendCancellableCoroutine { cont ->
            val ws = http.newWebSocket(request, object : WebSocketListener() {
                override fun onOpen(webSocket: WebSocket, response: Response) {
                    if (cont.isActive) cont.resume(Unit)
                }

                override fun onMessage(webSocket: WebSocket, text: String) {
                    runCatching { JSONObject(text) }
                        .onSuccess { inbox.trySend(Frame.Text(it)) }
                        .onFailure { Log.w(TAG, "unparseable frame: ${text.take(120)}") }
                }

                override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                    // The server never sends binary; ignore rather than fail.
                    Log.w(TAG, "unexpected binary frame (${bytes.size} bytes)")
                }

                override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                    inbox.trySend(Frame.Failure(t))
                    if (cont.isActive) cont.resumeWithException(
                        VlmException("Could not reach ${config.url}: ${t.message}", "connect")
                    )
                }

                override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                    inbox.trySend(Frame.Closed(reason.ifEmpty { "closed ($code)" }))
                }
            })
            socket = ws
            cont.invokeOnCancellation { ws.cancel() }
        }

        val ready = await("ready")
        Log.i(TAG, "ready: model=${ready.optString("model")} protocol=${ready.optInt("protocol")}")

        if (ready.optBoolean("auth_required")) {
            if (config.token.isEmpty()) {
                throw VlmException("Server requires a token but vlm.properties has none", "auth")
            }
            send(JSONObject().put("type", "auth").put("token", config.token))
            await("auth")
            Log.i(TAG, "authenticated")
        }
        ready
    }

    /**
     * Uploads an image inline as base64 and waits for the server to prepare it.
     *
     * Inline is the right choice for photos: the protocol's binary path exists for
     * multi-hundred-megabyte video, and a JPEG well under the 16 MB message cap costs
     * only base64's 33% overhead, which is cheaper than managing chunked framing.
     */
    suspend fun uploadImage(bytes: ByteArray, name: String = "photo.jpg"): JSONObject {
        require(bytes.isNotEmpty()) { "image is empty" }
        if (bytes.size > MAX_INLINE_BYTES) {
            throw VlmException(
                "Image is ${bytes.size / 1024} KB; inline upload is capped at " +
                    "${MAX_INLINE_BYTES / 1024} KB. Downscale it first.",
                "media",
            )
        }
        send(
            JSONObject()
                .put("type", "upload")
                .put("name", name)
                .put("data", Base64.encodeToString(bytes, Base64.NO_WRAP))
        )
        val media = await("media", ignore = setOf("progress"))
        Log.i(TAG, "media ready: ${media.optString("description")}")
        return media
    }

    /**
     * Asks a question about the uploaded media.
     *
     * @param onToken called for each streamed fragment, in arrival order.
     * @param onQueued called if the board is busy serving someone else — the NPU handles
     *   one request at a time, so this is normal rather than an error.
     */
    suspend fun ask(
        prompt: String,
        onQueued: () -> Unit = {},
        onToken: (String) -> Unit = {},
    ): VlmAnswer {
        require(prompt.isNotBlank()) { "prompt is blank" }
        send(JSONObject().put("type", "ask").put("prompt", prompt).put("stream", true))

        val builder = StringBuilder()
        while (true) {
            val frame = receive()
            when (frame.optString("type")) {
                "queued" -> onQueued()
                "token" -> frame.optString("text").let { builder.append(it); onToken(it) }
                "answer" -> return VlmAnswer(
                    // Prefer the server's assembled text; the streamed pieces are a
                    // preview and can lag the final detokenisation.
                    text = frame.optString("text").ifEmpty { builder.toString() },
                    latencySeconds = frame.optDouble("latency_s", 0.0),
                    completionTokens = frame.opt("completion_tokens") as? Int,
                )
                "error" -> throw VlmException(
                    frame.optString("message"), frame.optString("code", "vlm"),
                )
            }
        }
    }

    /** Server health and which models are loaded upstream. */
    suspend fun status(): JSONObject {
        send(JSONObject().put("type", "status"))
        return await("status")
    }

    /**
     * Round-trips a ping and returns the latency in milliseconds.
     *
     * Cheap enough to run before a demo: it proves the socket, the token and the route
     * across the tailnet without waking the NPU or queueing behind someone else's
     * inference. A slow pong points at the network — most likely a DERP relay rather
     * than a direct peer connection.
     */
    suspend fun ping(): Long {
        val started = System.nanoTime()
        send(JSONObject().put("type", "ping"))
        await("pong")
        return (System.nanoTime() - started) / 1_000_000
    }

    private fun send(payload: JSONObject) {
        val ws = socket ?: throw VlmException("Not connected", "connect")
        if (!ws.send(payload.toString())) {
            throw VlmException("Could not queue message; socket is closing", "connect")
        }
    }

    /** Waits for a frame of [type], surfacing `error` frames as exceptions. */
    private suspend fun await(type: String, ignore: Set<String> = emptySet()): JSONObject {
        while (true) {
            val frame = receive()
            when (val kind = frame.optString("type")) {
                type -> return frame
                "error" -> throw VlmException(
                    frame.optString("message"), frame.optString("code", "vlm"),
                )
                in ignore -> Unit
                else -> Log.d(TAG, "ignoring '$kind' while waiting for '$type'")
            }
        }
    }

    private suspend fun receive(): JSONObject =
        withTimeout(REPLY_TIMEOUT_MS) {
            when (val frame = inbox.receive()) {
                is Frame.Text -> frame.json
                is Frame.Failure -> throw VlmException(
                    "Connection failed: ${frame.cause.message}", "connect",
                )
                is Frame.Closed -> throw VlmException(
                    "Server closed the connection: ${frame.reason}", "connect",
                )
            }
        }

    override fun close() {
        socket?.close(1000, "bye")
        socket = null
        inbox.close()
        http.dispatcher.executorService.shutdown()
    }

    private companion object {
        const val TAG = "VlmClient"
        const val CONNECT_TIMEOUT_MS = 20_000L

        // Inference on the board is ~10s and a queued request waits behind another,
        // so this is deliberately far longer than a typical network timeout.
        const val REPLY_TIMEOUT_MS = 180_000L

        // Server caps a message at 16 MB; base64 inflates by 4/3, so keep the raw
        // image comfortably below that.
        const val MAX_INLINE_BYTES = 8 * 1024 * 1024
    }
}
