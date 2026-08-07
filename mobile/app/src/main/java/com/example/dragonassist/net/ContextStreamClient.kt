package com.example.dragonassist.net

import android.util.Log
import org.json.JSONObject
import java.io.Closeable

/**
 * Appends text to a session's context file on the laptop, over `/v1/iq9`.
 *
 * The endpoint was written for the IQ-9075 to stream video narration, but it authenticates
 * on a shared token rather than on identity — so the phone can write context directly.
 * That is what lets the whole system be wired together without deploying anything new to
 * the board.
 *
 * Two sources feed it: what Whisper heard (translated on the phone's NPU) and what the
 * VLM saw. Both become material the laptop's extractive reader can answer from later,
 * which is why a repeat question costs ~40 ms on a laptop NPU instead of seconds of vision.
 *
 * Sequence numbers are per session and must be contiguous. The server replies to `start`
 * with `next_sequence`, so a reconnect resumes exactly where it stopped rather than
 * duplicating or losing text.
 */
class ContextStreamClient(
    private val url: String,
    private val token: String,
) : Closeable {

    private var socket: JsonWebSocket? = null
    private var sequence = 0
    private var sessionId: String? = null

    val isOpen: Boolean get() = socket != null

    /**
     * Opens the stream for [sessionId], returning the sequence the server expects next.
     *
     * Safe to call again after a disconnect: the returned value resynchronises the local
     * counter with what the server already has on disk.
     */
    suspend fun open(sessionId: String): Int {
        close()
        val ws = JsonWebSocket.connect(url)
        socket = ws
        this.sessionId = sessionId

        ws.send(
            JSONObject()
                .put("type", "start")
                .put("protocol", PROTOCOL_VERSION)
                .put("token", token)
                .put("video_id", sessionId)
        )
        val started = ws.await("started")
        sequence = started.optInt("next_sequence", 1).coerceAtLeast(1) - 1
        Log.i(TAG, "context stream open for $sessionId, resuming at ${sequence + 1}")
        return sequence + 1
    }

    /**
     * Appends [text] verbatim and waits for it to be durable.
     *
     * The receiver acknowledges only after the bytes are on disk, so returning normally
     * means the text will survive a crash — worth waiting for, since context that silently
     * failed to write would produce confidently wrong answers later.
     */
    suspend fun append(text: String) {
        if (text.isBlank()) return
        val ws = socket ?: throw TransportException("Context stream is not open", "closed")
        val id = sessionId ?: throw TransportException("No session", "closed")

        for (chunk in text.chunked(CHUNK_CHARS)) {
            val seq = ++sequence
            ws.send(
                JSONObject()
                    .put("type", "text")
                    .put("video_id", id)
                    .put("sequence", seq)
                    .put("text", chunk)
            )
            val ack = ws.await("ack")
            val acked = ack.optInt("sequence", -1)
            if (acked != seq) {
                // A mismatch means our counter has drifted from the server's; continuing
                // would append text at the wrong offset or be rejected outright.
                throw TransportException(
                    "Out of sync: sent sequence $seq but the server acked $acked", "sequence",
                )
            }
        }
    }

    /** Marks the session complete. Best-effort: the context is already durable without it. */
    suspend fun end() {
        val ws = socket ?: return
        val id = sessionId ?: return
        runCatching {
            ws.send(
                JSONObject()
                    .put("type", "end")
                    .put("video_id", id)
                    .put("sequence", sequence)
            )
        }.onFailure { Log.w(TAG, "could not send end for $id", it) }
    }

    override fun close() {
        runCatching { socket?.close() }
        socket = null
    }

    private companion object {
        const val TAG = "ContextStream"
        const val PROTOCOL_VERSION = 1

        /** Server caps a frame at 1 MB; this keeps each append comfortably small. */
        const val CHUNK_CHARS = 4096
    }
}
