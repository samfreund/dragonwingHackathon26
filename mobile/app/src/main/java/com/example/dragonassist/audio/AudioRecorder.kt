package com.example.dragonassist.audio

import android.Manifest
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.annotation.RequiresPermission
import java.io.ByteArrayOutputStream
import kotlin.math.abs
import kotlin.math.min

/**
 * Captures 16 kHz mono 16-bit PCM — the exact format Whisper's encoder expects, so
 * nothing has to be resampled later.
 *
 * Samples are held in memory. A 30 s clip is under 1 MB, and Whisper's window is 30 s,
 * so there is no reason to stream to disk while recording.
 */
class AudioRecorder(val sampleRate: Int = SAMPLE_RATE) {

    private var record: AudioRecord? = null
    private var reader: Thread? = null
    private val pcm = ByteArrayOutputStream()

    @Volatile
    private var active = false

    val isRecording: Boolean get() = active

    /**
     * Begins capture on a background thread. [onLevel] is called continuously with a
     * 0..1 loudness value so the UI can show the mic is actually hearing something —
     * a silent recording is the single most common way an ASR demo fails.
     */
    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun start(onLevel: (Float) -> Unit) {
        if (active) return

        val minBuffer = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        require(minBuffer > 0) { "AudioRecord rejected ${sampleRate}Hz mono PCM16" }

        // A generous buffer: small ones drop samples when the UI thread stalls.
        val bufferBytes = minBuffer * 4

        val recorder = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferBytes,
        )
        check(recorder.state == AudioRecord.STATE_INITIALIZED) {
            "AudioRecord failed to initialise — is RECORD_AUDIO granted?"
        }

        pcm.reset()
        record = recorder
        active = true
        recorder.startRecording()

        reader = Thread {
            val chunk = ByteArray(bufferBytes)
            while (active) {
                val read = recorder.read(chunk, 0, chunk.size)
                if (read > 0) {
                    pcm.write(chunk, 0, read)
                    onLevel(peakOf(chunk, read))
                }
            }
        }.apply {
            priority = Thread.MAX_PRIORITY
            start()
        }
    }

    /** Stops capture and returns everything recorded since [start]. */
    fun stop(): Recording {
        if (!active) return Recording(FloatArray(0), sampleRate)

        active = false
        reader?.join(500)
        reader = null

        record?.run {
            try {
                stop()
            } catch (_: IllegalStateException) {
                // Already stopped; nothing to do.
            }
            release()
        }
        record = null

        return Recording(toFloatSamples(pcm.toByteArray()), sampleRate)
    }

    /** Releases the mic without producing a result. Safe to call when idle. */
    fun cancel() {
        active = false
        reader?.join(500)
        reader = null
        record?.run {
            try {
                stop()
            } catch (_: IllegalStateException) {
            }
            release()
        }
        record = null
        pcm.reset()
    }

    /** Peak amplitude of a little-endian PCM16 chunk, normalised to 0..1. */
    private fun peakOf(bytes: ByteArray, length: Int): Float {
        var peak = 0
        var i = 0
        while (i + 1 < length) {
            val sample = (bytes[i].toInt() and 0xFF) or (bytes[i + 1].toInt() shl 8)
            peak = maxOf(peak, abs(sample.toShort().toInt()))
            i += 2
        }
        return min(1f, peak / Short.MAX_VALUE.toFloat())
    }

    /** Little-endian PCM16 bytes to the [-1, 1] floats Whisper wants. */
    private fun toFloatSamples(bytes: ByteArray): FloatArray {
        val out = FloatArray(bytes.size / 2)
        var i = 0
        while (i < out.size) {
            val lo = bytes[i * 2].toInt() and 0xFF
            val hi = bytes[i * 2 + 1].toInt() shl 8
            out[i] = (lo or hi).toShort() / 32768f
            i++
        }
        return out
    }

    companion object {
        /** Whisper is trained at 16 kHz. Do not change this without resampling. */
        const val SAMPLE_RATE = 16_000
    }
}

/** A finished capture, normalised and ready for a [com.example.dragonassist.transcribe.Transcriber]. */
data class Recording(
    val samples: FloatArray,
    val sampleRate: Int,
) {
    val durationSeconds: Float get() = samples.size / sampleRate.toFloat()

    val isEmpty: Boolean get() = samples.isEmpty()

    /** Loudest sample. Near zero means the mic captured silence. */
    val peak: Float get() = samples.maxOfOrNull { abs(it) } ?: 0f

    // Generated because the class holds an array.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Recording) return false
        return sampleRate == other.sampleRate && samples.contentEquals(other.samples)
    }

    override fun hashCode(): Int = 31 * samples.contentHashCode() + sampleRate
}
