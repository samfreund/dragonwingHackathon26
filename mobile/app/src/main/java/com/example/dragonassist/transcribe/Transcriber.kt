package com.example.dragonassist.transcribe

import com.example.dragonassist.audio.Recording

/**
 * Speech to text. The seam that lets the Whisper/QNN work land without touching
 * capture or UI.
 *
 * Implementations must be safe to call from a background dispatcher and must not
 * assume the recording is any particular length — Whisper's 30 s window is the
 * model's constraint, not the recorder's, so padding and chunking belong here.
 */
interface Transcriber {

    /** Shown in the UI so it's obvious which engine produced a given transcript. */
    val name: String

    /** True once any model assets are loaded and [transcribe] can run. */
    val isReady: Boolean

    /** Loads model assets. Called off the main thread; safe to call more than once. */
    suspend fun prepare()

    /** Returns the transcript, or throws [TranscriptionException] with something readable. */
    suspend fun transcribe(recording: Recording): String

    /** Frees model assets. */
    fun close()
}

class TranscriptionException(message: String, cause: Throwable? = null) :
    Exception(message, cause)
