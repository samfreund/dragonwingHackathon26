package com.example.dragonassist.net

import android.util.Log

/**
 * Logs every WebSocket frame the phone sends and receives.
 *
 * Without this, a protocol problem is only visible by correlating logs on three machines
 * — which is how a token mismatch and a session-id mismatch each took far longer to find
 * than they should have. `adb logcat -s WsWire` now shows the whole conversation.
 *
 * Payloads are truncated hard: an inline photo upload is a few hundred kilobytes of
 * base64, and logging it whole would push everything else out of the ring buffer.
 */
object Wire {

    private const val TAG = "WsWire"
    private const val MAX = 400

    fun out(label: String, payload: String) = Log.i(TAG, "$label >>> ${trim(payload)}")

    fun inbound(label: String, payload: String) = Log.i(TAG, "$label <<< ${trim(payload)}")

    fun binary(label: String, bytes: Int) = Log.i(TAG, "$label >>> <$bytes binary bytes>")

    /**
     * Keeps the head and tail: the head carries the message type and ids, the tail carries
     * whatever a long field was hiding at the end.
     */
    private fun trim(payload: String): String = if (payload.length <= MAX) {
        payload
    } else {
        val head = payload.take(MAX - 60)
        val tail = payload.takeLast(40)
        "$head …[${payload.length - MAX + 20} chars]… $tail"
    }
}
