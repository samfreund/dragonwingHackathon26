package com.example.dragonassist

import com.example.dragonassist.audio.MelSpectrogram
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * Checks the Kotlin mel front end against HuggingFace `WhisperFeatureExtractor`.
 *
 * The reference tensor in `ref_mel.bin` was produced by `dump_reference_mel.py` from the
 * same `sample.wav`. If this test passes, a bad transcript is the model's or the decoder
 * loop's fault, not the front end's — which is the whole reason the test exists.
 */
class MelSpectrogramTest {

    private fun resource(name: String) =
        checkNotNull(javaClass.classLoader?.getResourceAsStream(name)) { "missing fixture $name" }

    /** Reads 16-bit mono PCM WAV to floats, mirroring what AudioRecorder produces. */
    private fun readWav(): FloatArray {
        val bytes = resource("sample.wav").use { it.readBytes() }
        val data = bytes.copyOfRange(44, bytes.size) // skip the 44-byte canonical header
        val shorts = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        return FloatArray(shorts.remaining()) { shorts.get(it) / 32768f }
    }

    private fun readReference(): FloatArray {
        val bytes = resource("ref_mel.bin").use { it.readBytes() }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun mel(): MelSpectrogram =
        MelSpectrogram(MelSpectrogram.readFilters(resource("mel_filters.bin")))

    @Test
    fun `produces the shape whisper's encoder requires`() {
        val out = mel().compute(readWav())
        assertEquals(MelSpectrogram.N_MELS * MelSpectrogram.N_FRAMES, out.size)
        assertEquals(80, MelSpectrogram.N_MELS)
        assertEquals(3000, MelSpectrogram.N_FRAMES)
    }

    @Test
    fun `matches the huggingface reference tensor`() {
        val actual = mel().compute(readWav())
        val expected = readReference()
        assertEquals("reference size", expected.size, actual.size)

        var worst = 0f
        var worstIndex = -1
        var sum = 0.0
        for (i in expected.indices) {
            val diff = abs(expected[i] - actual[i])
            sum += diff
            if (diff > worst) {
                worst = diff
                worstIndex = i
            }
        }
        val mean = sum / expected.size

        println("mel comparison: mean abs err=%.6f  max abs err=%.6f at %d (mel %d, frame %d)"
            .format(mean, worst, worstIndex,
                worstIndex / MelSpectrogram.N_FRAMES, worstIndex % MelSpectrogram.N_FRAMES))

        // Float32 accumulation order differs between numpy and this loop, so exact
        // equality is not expected. Anything near 1e-3 means the algorithm matches.
        assertTrue("max abs error $worst too large — front end does not match reference",
            worst < 1e-2f)
        assertTrue("mean abs error $mean too large", mean < 1e-3f)
    }

    @Test
    fun `silence produces the clamped floor everywhere`() {
        val out = mel().compute(FloatArray(16_000))
        val first = out[0]
        assertTrue("silence should sit at the dynamic-range floor, got $first", first < -0.5f)
        assertTrue("silence should be uniform", out.all { abs(it - first) < 1e-5f })
    }
}
