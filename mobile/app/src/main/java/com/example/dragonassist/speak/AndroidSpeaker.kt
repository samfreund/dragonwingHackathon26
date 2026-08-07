package com.example.dragonassist.speak

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.Voice
import android.util.Log
import java.util.Locale
import kotlin.coroutines.resume
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull

/**
 * Android's built-in speech synthesiser.
 *
 * Runs on the phone with no network once voice data is installed, which keeps the
 * "nothing leaves the device" property of the rest of the pipeline intact. It is not,
 * however, running on the Hexagon NPU — that would need a neural model such as Piper,
 * which is why this sits behind [Speaker] rather than being called directly.
 */
class AndroidSpeaker(private val context: Context) : Speaker {

    override val name: String
        get() = if (offline) "Android TTS (offline)" else "Android TTS"

    override var isReady = false
        private set

    private var engine: TextToSpeech? = null
    private var offline = false
    private var utterance = 0L

    /**
     * `TextToSpeech` reports readiness through a callback, so construction alone is not
     * enough — speaking before `onInit` silently drops the text.
     */
    override suspend fun prepare() {
        if (isReady) return

        val status = withTimeoutOrNull(INIT_TIMEOUT_MS) {
            suspendCancellableCoroutine { cont ->
                val tts = TextToSpeech(context) { code ->
                    if (cont.isActive) cont.resume(code)
                }
                engine = tts
                cont.invokeOnCancellation { runCatching { tts.shutdown() } }
            }
        }

        if (status != TextToSpeech.SUCCESS) {
            Log.w(TAG, "TextToSpeech init failed or timed out (status=$status)")
            close()
            return
        }

        val tts = engine ?: return
        when (tts.setLanguage(Locale.US)) {
            TextToSpeech.LANG_MISSING_DATA, TextToSpeech.LANG_NOT_SUPPORTED -> {
                // The engine exists but has no en-US voice data installed. Better to
                // stay silent than to speak in whatever the default locale happens to be.
                Log.w(TAG, "no en-US voice data installed")
                close()
                return
            }
        }

        // Prefer a voice that needs no network, so speech keeps working on a flaky
        // tailnet — the whole point of the on-device pipeline.
        offline = runCatching {
            tts.voices
                ?.firstOrNull { it.locale == Locale.US && !it.isNetworkConnectionRequired }
                ?.also { tts.voice = it } != null
        }.getOrDefault(false)

        tts.setSpeechRate(SPEECH_RATE)
        isReady = true
        Log.i(TAG, "ready; offline voice=$offline")
    }

    override fun say(sentence: String) {
        val tts = engine ?: return
        if (!isReady || sentence.isBlank()) return
        // QUEUE_ADD, not QUEUE_FLUSH: sentences arrive one at a time as the answer
        // streams, and flushing would cut off the previous one mid-word.
        tts.speak(sentence, TextToSpeech.QUEUE_ADD, null, "answer-${utterance++}")
    }

    override fun stop() {
        runCatching { engine?.stop() }
    }

    override fun close() {
        runCatching { engine?.stop() }
        runCatching { engine?.shutdown() }
        engine = null
        isReady = false
    }

    private companion object {
        const val TAG = "AndroidSpeaker"
        const val INIT_TIMEOUT_MS = 5_000L

        /** Slightly quicker than default; the default reads oddly slowly for short answers. */
        const val SPEECH_RATE = 1.05f
    }
}
