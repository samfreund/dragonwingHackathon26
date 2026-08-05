package com.example.dragonassist.transcribe

import com.example.dragonassist.audio.Recording
import kotlinx.coroutines.delay

/**
 * Placeholder so the whole app runs end to end before Whisper exists.
 *
 * It deliberately reports what the microphone actually captured, which makes it a real
 * diagnostic and not just a mock: if peak amplitude is near zero here, no amount of
 * model work will produce a transcript, and the bug is in permissions or the mic.
 */
class StubTranscriber : Transcriber {

    override val name = "Stub (no model yet)"

    override var isReady = false
        private set

    override suspend fun prepare() {
        isReady = true
    }

    override suspend fun transcribe(recording: Recording): String {
        if (recording.isEmpty) {
            throw TranscriptionException("Nothing was recorded.")
        }

        // Stand in for inference latency so the UI's loading state gets exercised.
        delay(400)

        val seconds = "%.1f".format(recording.durationSeconds)
        val peak = "%.3f".format(recording.peak)

        return if (recording.peak < 0.01f) {
            "Captured $seconds s but the signal is silent (peak $peak). " +
                "Check the mic permission and that nothing else holds the microphone."
        } else {
            "Captured $seconds s of audio at ${recording.sampleRate} Hz, " +
                "${recording.samples.size} samples, peak $peak.\n\n" +
                "Audio capture works. Swap in WhisperTranscriber to get real text."
        }
    }

    override fun close() {
        isReady = false
    }
}
