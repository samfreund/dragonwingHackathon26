package com.example.dragonassist.audio

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sin

/**
 * Whisper's log-mel front end: audio samples in, the `(80, 3000)` tensor its encoder expects out.
 *
 * Every constant here matches HuggingFace `WhisperFeatureExtractor` for `openai/whisper-base`.
 * They are not tunable — a mismatch anywhere produces a spectrogram that still *looks* plausible
 * and makes the model emit fluent, confident, wrong text. Verify against
 * `dump_reference_mel.py` rather than against intuition.
 *
 * The mel filterbank is loaded from `assets/mel_filters.bin` instead of being reconstructed,
 * because Whisper's slaney-scale filter geometry is easy to get subtly wrong.
 */
class MelSpectrogram(
    /** Row-major `(N_FREQS, N_MELS)` filterbank, as written by the export script. */
    private val filters: FloatArray,
) {

    init {
        require(filters.size == N_FREQS * N_MELS) {
            "expected ${N_FREQS * N_MELS} filter values, got ${filters.size}"
        }
    }

    // Periodic Hann, matching torch.hann_window(400) — note the divisor is N, not N-1.
    private val window = FloatArray(N_FFT) { n ->
        (0.5 - 0.5 * cos(2.0 * Math.PI * n / N_FFT)).toFloat()
    }

    // Twiddle tables for a direct real-input DFT. N_FFT is 400, which is not a power of
    // two, so a radix-2 FFT cannot be used without changing the result. A 201-bin direct
    // transform is O(n^2) but runs in well under a second for a 30 s clip, and it is
    // obviously correct. Optimise to Bluestein only if profiling says to.
    private val cosTable = FloatArray(N_FREQS * N_FFT)
    private val sinTable = FloatArray(N_FREQS * N_FFT)

    init {
        for (k in 0 until N_FREQS) {
            for (n in 0 until N_FFT) {
                val angle = -2.0 * Math.PI * k * n / N_FFT
                cosTable[k * N_FFT + n] = cos(angle).toFloat()
                sinTable[k * N_FFT + n] = sin(angle).toFloat()
            }
        }
    }

    /**
     * Computes the log-mel spectrogram.
     *
     * @param samples mono 16 kHz audio in [-1, 1]. Padded with silence or trimmed to
     *   exactly 30 s, which is the only length Whisper's encoder accepts.
     * @return `N_MELS * N_FRAMES` floats in mel-major order: `mel[m * N_FRAMES + t]`.
     */
    fun compute(samples: FloatArray): FloatArray {
        val padded = FloatArray(N_SAMPLES)
        System.arraycopy(samples, 0, padded, 0, minOf(samples.size, N_SAMPLES))

        // center=True: reflect-pad by n_fft/2 so frame t is centred on sample t*hop.
        val reflected = reflectPad(padded, N_FFT / 2)

        val out = FloatArray(N_MELS * N_FRAMES)
        val power = FloatArray(N_FREQS)
        val frame = FloatArray(N_FFT)

        var maxLog = Float.NEGATIVE_INFINITY

        for (t in 0 until N_FRAMES) {
            val offset = t * HOP_LENGTH
            for (n in 0 until N_FFT) {
                frame[n] = reflected[offset + n] * window[n]
            }

            for (k in 0 until N_FREQS) {
                var re = 0f
                var im = 0f
                val base = k * N_FFT
                for (n in 0 until N_FFT) {
                    val s = frame[n]
                    re += s * cosTable[base + n]
                    im += s * sinTable[base + n]
                }
                power[k] = re * re + im * im
            }

            // mel[m] = sum_k power[k] * filters[k][m]
            for (m in 0 until N_MELS) {
                var acc = 0f
                for (k in 0 until N_FREQS) {
                    acc += power[k] * filters[k * N_MELS + m]
                }
                val logged = log10(max(acc, LOG_FLOOR)).toFloat()
                out[m * N_FRAMES + t] = logged
                if (logged > maxLog) maxLog = logged
            }
        }

        // Whisper clamps to 8 decades below the peak, then rescales to roughly [-1, 1].
        val floor = maxLog - DYNAMIC_RANGE
        for (i in out.indices) {
            out[i] = (max(out[i], floor) + 4f) / 4f
        }
        return out
    }

    /** Mirrors `numpy.pad(mode="reflect")`: edge sample is not repeated. */
    private fun reflectPad(input: FloatArray, pad: Int): FloatArray {
        val out = FloatArray(input.size + 2 * pad)
        System.arraycopy(input, 0, out, pad, input.size)
        for (i in 0 until pad) {
            out[pad - 1 - i] = input[i + 1]
            out[pad + input.size + i] = input[input.size - 2 - i]
        }
        return out
    }

    companion object {
        const val SAMPLE_RATE = 16_000
        const val N_FFT = 400
        const val HOP_LENGTH = 160
        const val N_MELS = 80
        const val CHUNK_SECONDS = 30

        /** Real-input DFT yields n_fft/2 + 1 usable bins. */
        const val N_FREQS = N_FFT / 2 + 1          // 201
        const val N_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS  // 480000
        const val N_FRAMES = N_SAMPLES / HOP_LENGTH        // 3000

        private const val LOG_FLOOR = 1e-10f
        private const val DYNAMIC_RANGE = 8f

        const val FILTERS_ASSET = "mel_filters.bin"

        /** Reads the little-endian float32 filterbank produced by `dump_reference_mel.py`. */
        fun readFilters(stream: InputStream): FloatArray = stream.use { input ->
            val bytes = input.readBytes()
            require(bytes.size == N_FREQS * N_MELS * 4) {
                "mel_filters.bin should be ${N_FREQS * N_MELS * 4} bytes, got ${bytes.size}"
            }
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
            FloatArray(buffer.remaining()).also { buffer.get(it) }
        }
    }
}
