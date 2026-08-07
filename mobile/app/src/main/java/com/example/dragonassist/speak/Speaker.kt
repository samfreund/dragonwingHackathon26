package com.example.dragonassist.speak

import java.io.Closeable

/**
 * Speaks the board's answer aloud.
 *
 * Deliberately the same shape as `Transcriber`: the app talks to this interface, so
 * swapping Android's built-in engine for an on-device neural model (Piper on the Hexagon
 * NPU) is a one-class change rather than a rewrite of the calling code.
 *
 * Implementations queue rather than interrupt — [say] is called once per sentence as the
 * answer streams in, and the sentences must come out in order.
 */
interface Speaker : Closeable {

    /** Shown in the UI so it is obvious which engine is talking. */
    val name: String

    val isReady: Boolean

    /** Initialises the engine. Safe to call more than once. */
    suspend fun prepare()

    /** Queues one sentence behind anything already speaking. */
    fun say(sentence: String)

    /** Abandons the queue and stops mid-word — used when a new question starts. */
    fun stop()
}

/** Used when speech is unavailable or switched off, so callers need no null checks. */
object SilentSpeaker : Speaker {
    override val name = "Speech off"
    override val isReady = true
    override suspend fun prepare() = Unit
    override fun say(sentence: String) = Unit
    override fun stop() = Unit
    override fun close() = Unit
}
