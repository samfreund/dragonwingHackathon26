package com.example.dragonassist

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.dragonassist.audio.MelSpectrogram
import com.example.dragonassist.transcribe.WhisperSession
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * Everything upstream of the decoder loop, running on the phone: WAV to mel to encoder.
 *
 * Two things this measures that nothing else has. The mel front end uses an O(n^2) direct
 * DFT — correct, but its cost on real hardware was never established. And the encoder had
 * only been exercised on a silent tensor, which is not proof that real input produces
 * meaningful cross-attention caches.
 */
@RunWith(AndroidJUnit4::class)
class WhisperPipelineTest {

    /** The app under test — owns mel_filters.bin and the model files. */
    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    /**
     * The test APK — owns the fixtures. These are two different asset managers; reading
     * fixtures from the app context throws FileNotFoundException.
     */
    private fun asset(name: String) =
        InstrumentationRegistry.getInstrumentation().context.assets.open(name)

    private fun sampleAudio(): FloatArray {
        val bytes = asset("sample.wav").use { it.readBytes() }
        val pcm = bytes.copyOfRange(44, bytes.size)
        val shorts = ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        return FloatArray(shorts.remaining()) { shorts.get(it) / 32768f }
    }

    private fun mel(): MelSpectrogram =
        MelSpectrogram(MelSpectrogram.readFilters(context.assets.open(MelSpectrogram.FILTERS_ASSET)))

    @Test
    fun melMatchesReferenceOnDevice() {
        val actual = mel().compute(sampleAudio())
        val expected = asset("ref_mel.bin").use {
            val buf = ByteBuffer.wrap(it.readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            FloatArray(buf.remaining()).also { a -> buf.get(a) }
        }
        assertEquals(expected.size, actual.size)

        var worst = 0f
        for (i in expected.indices) worst = maxOf(worst, abs(expected[i] - actual[i]))
        println("on-device mel max abs err: $worst")
        assertTrue("mel differs from reference on device: $worst", worst < 1e-2f)
    }

    @Test
    fun melIsFastEnoughForADemo() {
        val audio = sampleAudio()
        val m = mel()
        m.compute(audio) // warm up JIT before timing

        val started = System.nanoTime()
        m.compute(audio)
        val elapsedMs = (System.nanoTime() - started) / 1_000_000
        println("mel front end: ${elapsedMs}ms for a 30s window")

        // Generous: the point is to catch an O(n^2) DFT being unusably slow, not to
        // set a performance target. Anything under ~2s is fine for record-then-transcribe.
        assertTrue("mel took ${elapsedMs}ms — replace the direct DFT with Bluestein",
            elapsedMs < 3_000)
    }

    @Test
    fun encoderProducesRealCrossCachesFromRealAudio() {
        assumeTrue("models not pushed", WhisperSession.modelsPresent(context))

        val melValues = mel().compute(sampleAudio())
        val env = OrtEnvironment.getEnvironment()

        WhisperSession.create(context).use { session ->
            assertTrue("QNN did not bind", session.isOnNpu)

            val input = WhisperSession.float16Tensor(env, melValues, longArrayOf(1, 80, 3000))
            val started = System.nanoTime()
            input.use { tensor ->
                session.encoder.run(mapOf(session.encoder.inputNames.first() to tensor))
                    .use { results ->
                        val elapsedMs = (System.nanoTime() - started) / 1_000_000
                        println("encoder on real audio: ${elapsedMs}ms")
                        assertEquals(12, results.size())

                        // Silence in, silence out would pass a shape check. Verify the
                        // caches actually carry signal: finite, and not all zeros.
                        val first = results.get(0) as OnnxTensor
                        val halves = first.shortBuffer
                        var nonZero = 0
                        var maxAbs = 0f
                        var checked = 0
                        while (halves.hasRemaining() && checked < 20_000) {
                            val v = WhisperSession.halfToFloat(halves.get())
                            assertTrue("cross cache contains NaN/Inf", v.isFinite())
                            if (v != 0f) nonZero++
                            maxAbs = maxOf(maxAbs, abs(v))
                            checked++
                        }
                        println("cross cache: $nonZero/$checked non-zero, max abs $maxAbs")
                        assertTrue("cross cache is all zeros — encoder saw nothing",
                            nonZero > checked / 10)
                    }
            }
        }
    }

    @Test
    fun sessionsCanBeCreatedAndClosedRepeatedly() {
        assumeTrue("models not pushed", WhisperSession.modelsPresent(context))
        // 192 MB of graphs: a leak here would surface as an OOM part-way through a demo.
        repeat(3) { i ->
            val started = System.nanoTime()
            WhisperSession.create(context).use { assertTrue(it.isOnNpu) }
            println("session ${i + 1} create+close: ${(System.nanoTime() - started) / 1_000_000}ms")
        }
    }
}
