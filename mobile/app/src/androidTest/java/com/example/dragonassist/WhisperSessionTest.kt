package com.example.dragonassist

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import ai.onnxruntime.OrtEnvironment
import com.example.dragonassist.audio.MelSpectrogram
import com.example.dragonassist.transcribe.WhisperSession
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs

/**
 * Phase 1 probe: does ONNX Runtime actually load the precompiled QNN graphs and bind them
 * to the Hexagon NPU on this device?
 *
 * These must run on hardware (`./gradlew connectedDebugAndroidTest`), not on the JVM.
 * A silent fallback to CPU would still "work" and be far too slow to demo, so the
 * provider is asserted rather than assumed.
 */
@RunWith(AndroidJUnit4::class)
class WhisperSessionTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun modelFilesArePresentOnDevice() {
        val dir = WhisperSession.modelDir(context)
        println("model dir: ${dir.absolutePath}")
        dir.listFiles()?.forEach { println("  ${it.name}  ${it.length()} bytes") }
        assertTrue(
            "push the precompiled_qnn_onnx bundle to ${dir.absolutePath}",
            WhisperSession.modelsPresent(context),
        )
    }

    @Test
    fun float16RoundTripsWithinHalfPrecision() {
        val probes = floatArrayOf(0f, 1f, -1f, 0.5f, -0.895080f, 1.104920f, 3.14159f, 1e-4f)
        for (v in probes) {
            val back = WhisperSession.halfToFloat(WhisperSession.floatToHalf(v))
            // fp16 has ~3 decimal digits; tolerance scales with magnitude.
            val tolerance = maxOf(1e-3f, abs(v) * 1e-3f)
            assertTrue("fp16 round trip $v -> $back", abs(v - back) < tolerance)
        }
    }

    @Test
    fun sessionsLoadAndBindToTheNpu() {
        assumeTrue("models not pushed", WhisperSession.modelsPresent(context))

        WhisperSession.create(context).use { session ->
            println("available providers: ${session.providersUsed}")
            println("encoder inputs:  ${session.encoder.inputNames}")
            println("encoder outputs: ${session.encoder.outputNames.size} tensors")
            println("decoder inputs:  ${session.decoder.inputNames.size} tensors")
            println("decoder outputs: ${session.decoder.outputNames.size} tensors")

            assertEquals("encoder should take one input", 1, session.encoder.inputNames.size)
            assertEquals("encoder should emit 12 cross caches", 12, session.encoder.outputNames.size)
            assertEquals("decoder should take 27 inputs", 27, session.decoder.inputNames.size)
            assertEquals("decoder should emit 13 outputs", 13, session.decoder.outputNames.size)

            assertTrue(
                "QNN did not bind — inference would silently run on CPU. Providers: " +
                    "${session.providersUsed}",
                session.isOnNpu,
            )
        }
    }

    @Test
    fun encoderRunsOnSilenceAndProducesCrossCaches() {
        assumeTrue("models not pushed", WhisperSession.modelsPresent(context))

        val env = OrtEnvironment.getEnvironment()
        WhisperSession.create(context).use { session ->
            val mel = FloatArray(MelSpectrogram.N_MELS * MelSpectrogram.N_FRAMES)
            val input = WhisperSession.float16Tensor(
                env, mel, longArrayOf(1, 80, 3000),
            )

            val started = System.nanoTime()
            input.use { tensor ->
                session.encoder.run(mapOf(session.encoder.inputNames.first() to tensor))
                    .use { results ->
                        val elapsedMs = (System.nanoTime() - started) / 1_000_000
                        println("encoder forward pass: ${elapsedMs}ms")
                        assertEquals(12, results.size())
                        assertTrue("encoder should be well under a second on NPU",
                            elapsedMs < 5_000)
                    }
            }
        }
    }
}
