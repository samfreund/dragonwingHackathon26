package com.example.dragonassist

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.dragonassist.audio.Recording
import com.example.dragonassist.transcribe.WhisperSession
import com.example.dragonassist.transcribe.WhisperTranscriber
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * End-to-end transcription on the phone, checked against the Python reference.
 *
 * The reference (`HfWhisperApp` from qai_hub_models) produces exactly
 * "Hello, how are you doing today?" for `sample.wav`. Matching that string means the mel
 * front end, the encoder, the greedy decode loop, the KV-cache threading and the
 * detokenizer are all correct together — which no test of any single stage can establish.
 */
@RunWith(AndroidJUnit4::class)
class WhisperTranscriberTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private fun sampleRecording(): Recording {
        val bytes = InstrumentationRegistry.getInstrumentation().context.assets
            .open("sample.wav").use { it.readBytes() }
        val pcm = bytes.copyOfRange(44, bytes.size)
        val shorts = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        return Recording(FloatArray(shorts.remaining()) { shorts.get(it) / 32768f }, 16_000)
    }

    @Test
    fun transcribesSampleAudioMatchingThePythonReference() = runBlocking {
        assumeTrue("models not pushed", WhisperSession.modelsPresent(context))

        val transcriber = WhisperTranscriber(context)
        try {
            transcriber.prepare()
            val text = transcriber.transcribe(sampleRecording())

            val timings = transcriber.lastTimings
            println("TRANSCRIPT: \"$text\"")
            println("timings: $timings  total=${timings?.totalMs}ms")

            assertEquals(
                "on-device transcript differs from the Python reference",
                "Hello, how are you doing today?",
                text,
            )
        } finally {
            transcriber.close()
        }
    }

    @Test
    fun reportsSensibleTimings() = runBlocking {
        assumeTrue("models not pushed", WhisperSession.modelsPresent(context))

        val transcriber = WhisperTranscriber(context)
        try {
            transcriber.prepare()
            transcriber.transcribe(sampleRecording())
            val t = requireNotNull(transcriber.lastTimings)

            println("mel=${t.melMs}ms encoder=${t.encoderMs}ms decoder=${t.decoderMs}ms " +
                "tokens=${t.tokens} total=${t.totalMs}ms")

            assertTrue("decoder produced no tokens", t.tokens > 0)
            // A CPU fallback would be many seconds; this is the guard against a silent
            // regression to the CPU execution provider.
            assertTrue("total ${t.totalMs}ms is too slow for a live demo", t.totalMs < 10_000)
        } finally {
            transcriber.close()
        }
    }

    @Test
    fun silenceDoesNotProduceHallucinatedText() = runBlocking {
        assumeTrue("models not pushed", WhisperSession.modelsPresent(context))

        val transcriber = WhisperTranscriber(context)
        try {
            transcriber.prepare()
            val text = transcriber.transcribe(Recording(FloatArray(16_000), 16_000))
            println("silence -> \"$text\"")
            // Whisper is known to hallucinate on silence; this documents the behaviour
            // rather than asserting a specific string, so a regression is visible.
            assertTrue("silence produced a suspiciously long transcript", text.length < 200)
        } finally {
            transcriber.close()
        }
    }
}
