package com.example.dragonassist.audio

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Writes a [Recording] as a 16-bit PCM WAV.
 *
 * This exists for debugging, and it earns its keep: when the on-device transcript looks
 * wrong, pull the WAV off the phone and run it through desktop Whisper. If desktop gets
 * it right, the bug is in the mel pipeline or the model wiring, not the audio capture.
 *
 *   adb exec-out run-as com.example.dragonassist cat files/last_recording.wav > out.wav
 */
object WavWriter {

    fun write(recording: Recording, target: File): File {
        val pcm = toPcm16(recording.samples)
        FileOutputStream(target).use { out ->
            out.write(header(pcm.size, recording.sampleRate))
            out.write(pcm)
        }
        return target
    }

    private fun toPcm16(samples: FloatArray): ByteArray {
        val buffer = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (sample in samples) {
            val clamped = sample.coerceIn(-1f, 1f)
            buffer.putShort((clamped * Short.MAX_VALUE).toInt().toShort())
        }
        return buffer.array()
    }

    private fun header(dataBytes: Int, sampleRate: Int): ByteArray {
        val channels = 1
        val bitsPerSample = 16
        val byteRate = sampleRate * channels * bitsPerSample / 8

        return ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray())
            putInt(36 + dataBytes)
            put("WAVE".toByteArray())
            put("fmt ".toByteArray())
            putInt(16)                                   // PCM header size
            putShort(1)                                  // PCM, uncompressed
            putShort(channels.toShort())
            putInt(sampleRate)
            putInt(byteRate)
            putShort((channels * bitsPerSample / 8).toShort())
            putShort(bitsPerSample.toShort())
            put("data".toByteArray())
            putInt(dataBytes)
        }.array()
    }
}
