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
 *   laptop_host=qcworkshop3
 *   laptop_port=8001
 *   stream_token=<DRAGONASSIST_STREAM_TOKEN>
 *   phone_token=<DRAGONASSIST_PHONE_TOKEN>
 */
data class VlmConfig(
    val host: String,
    val port: Int,
    val token: String,
    /** The laptop running stream_transport; blank when it isn't deployed. */
    val laptopHost: String = "",
    val laptopPort: Int = DEFAULT_LAPTOP_PORT,
    /** Writes context via /v1/iq9. */
    val streamToken: String = "",
    /** Asks questions via /v1/phone. Deliberately a different secret from streamToken. */
    val phoneToken: String = "",
) {
    val url: String get() = "ws://$host:$port"

    val hasLaptop: Boolean get() = laptopHost.isNotEmpty()

    /** `/v1/iq9` — the context write endpoint. */
    val contextUrl: String get() = "ws://$laptopHost:$laptopPort/v1/iq9"

    /** `/v1/phone` — the question endpoint. */
    val textQaUrl: String get() = "ws://$laptopHost:$laptopPort/v1/phone"

    companion object {
        const val FILE_NAME = "vlm.properties"
        const val DEFAULT_PORT = 8765
        const val DEFAULT_LAPTOP_PORT = 8001

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
                laptopHost = props.getProperty("laptop_host")?.trim().orEmpty(),
                laptopPort = props.getProperty("laptop_port")?.trim()?.toIntOrNull()
                    ?: DEFAULT_LAPTOP_PORT,
                streamToken = props.getProperty("stream_token")?.trim().orEmpty(),
                phoneToken = props.getProperty("phone_token")?.trim().orEmpty(),
            )
        }
    }
}
