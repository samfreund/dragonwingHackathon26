package com.example.dragonassist.vlm

import android.content.Context
import java.io.File
import java.util.Properties

/**
 * Where the board's Q&A server is and how to authenticate to it.
 *
 * Read from `vlm.properties` in the app's external files directory rather than compiled
 * in, for two reasons: the token is a shared secret that must not reach a public repo,
 * and the board's address changes between the venue and anywhere else. Push it alongside
 * the Whisper models:
 *
 *   adb push vlm.properties /sdcard/Android/data/com.example.dragonassist/files/
 *
 *   host=iq9
 *   port=8765
 *   token=<VLMQA_WS_TOKEN from /etc/vlmqa/vlmqa.env on the board>
 */
data class VlmConfig(
    val host: String,
    val port: Int,
    val token: String,
) {
    val url: String get() = "ws://$host:$port"

    companion object {
        const val FILE_NAME = "vlm.properties"
        const val DEFAULT_PORT = 8765

        fun file(context: Context): File =
            File(context.getExternalFilesDir(null), FILE_NAME)

        fun isConfigured(context: Context): Boolean = file(context).let { it.isFile && it.length() > 0 }

        /** Returns null when the file is absent, so callers can degrade rather than crash. */
        fun load(context: Context): VlmConfig? {
            val source = file(context)
            if (!source.isFile) return null
            val props = Properties().apply { source.inputStream().use { load(it) } }
            val host = props.getProperty("host")?.trim().orEmpty()
            if (host.isEmpty()) return null
            return VlmConfig(
                host = host,
                port = props.getProperty("port")?.trim()?.toIntOrNull() ?: DEFAULT_PORT,
                token = props.getProperty("token")?.trim().orEmpty(),
            )
        }
    }
}
