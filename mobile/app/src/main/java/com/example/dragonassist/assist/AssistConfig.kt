package com.example.dragonassist.assist

import android.content.Context
import java.io.File
import java.util.Properties

/**
 * Where the laptop's query service is.
 *
 * Distinct from [com.example.dragonassist.vlm.VlmConfig], which points at the board's VLM
 * on port 8765. This one points at `stream_transport` on the laptop, which answers from
 * accumulated video context rather than from images — so both can be configured at once
 * and used for different questions.
 *
 * Read from `assist.properties` in the app's external files directory, because the
 * laptop's address changes between the venue and anywhere else:
 *
 *   adb push assist.properties /sdcard/Android/data/com.example.dragonassist/files/
 *
 *   host=qcworkshop3
 *   port=8001
 *
 * There is no token: the service has no application-level authentication and relies on
 * the tailnet for access control, so nothing secret belongs in this file.
 */
data class AssistConfig(
    val host: String,
    val port: Int,
) {
    val url: String get() = "ws://$host:$port/v1/phone"

    companion object {
        const val FILE_NAME = "assist.properties"
        const val DEFAULT_PORT = 8001

        fun file(context: Context): File =
            File(context.getExternalFilesDir(null), FILE_NAME)

        fun isConfigured(context: Context): Boolean = file(context).let { it.isFile && it.length() > 0 }

        /** Returns null when the file is absent, so callers can degrade rather than crash. */
        fun load(context: Context): AssistConfig? {
            val source = file(context)
            if (!source.isFile) return null
            val props = Properties().apply { source.inputStream().use { load(it) } }
            val host = props.getProperty("host")?.trim().orEmpty()
            if (host.isEmpty()) return null
            return AssistConfig(
                host = host,
                port = props.getProperty("port")?.trim()?.toIntOrNull() ?: DEFAULT_PORT,
            )
        }
    }
}
