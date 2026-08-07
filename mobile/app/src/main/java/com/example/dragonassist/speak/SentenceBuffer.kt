package com.example.dragonassist.speak

/**
 * Accumulates streamed tokens and releases them a sentence at a time.
 *
 * The board streams the answer token by token, and tokens are fragments — `"A "`,
 * `"person"`, `" in"`. Waiting for the whole answer before speaking wastes the streaming
 * entirely: on a ten-second response the user would hear nothing for ten seconds. Speaking
 * each fragment as it arrives is worse still, since a synthesiser given `" in"` produces
 * staccato nonsense.
 *
 * Releasing on sentence boundaries gets speech started about a second after the first
 * token while still giving the synthesiser enough context for natural prosody.
 */
class SentenceBuffer(
    /** Release anyway past this length, so a run-on answer still gets spoken. */
    private val maxChars: Int = 160,
) {

    private val pending = StringBuilder()

    /** Appends a streamed fragment and returns any sentences now complete. */
    fun append(piece: String): List<String> {
        pending.append(piece)
        val out = mutableListOf<String>()

        while (true) {
            val cut = boundaryIn(pending) ?: break
            val sentence = pending.substring(0, cut).trim()
            pending.delete(0, cut)
            if (sentence.isNotEmpty()) out += sentence
        }
        return out
    }

    /** Whatever is left at the end of the answer — the last sentence usually lacks a space after it. */
    fun flush(): String? {
        val rest = pending.toString().trim()
        pending.setLength(0)
        return rest.ifEmpty { null }
    }

    fun clear() = pending.setLength(0)

    /**
     * Index just past a sentence end, or null.
     *
     * A terminator only counts when followed by whitespace: `"3.9"` and `"e.g."` would
     * otherwise be split mid-number and mid-abbreviation. Since tokens stream in, a
     * terminator at the very end of the buffer is left alone until the next fragment
     * reveals whether a space follows.
     */
    private fun boundaryIn(text: CharSequence): Int? {
        for (i in text.indices) {
            if (text[i] !in TERMINATORS) continue
            // Consume any run of terminators and closing quotes: `?"` or `!!`
            var end = i
            while (end + 1 < text.length && (text[end + 1] in TERMINATORS || text[end + 1] in CLOSERS)) {
                end++
            }
            if (end + 1 >= text.length) return null   // wait for more
            if (text[end + 1].isWhitespace()) return end + 1
        }
        return if (text.length >= maxChars) lastSpaceBefore(text, maxChars) else null
    }

    /** Break on a word boundary rather than mid-word when forcing a long buffer out. */
    private fun lastSpaceBefore(text: CharSequence, limit: Int): Int {
        for (i in (limit - 1) downTo 1) if (text[i].isWhitespace()) return i
        return limit
    }

    private companion object {
        val TERMINATORS = charArrayOf('.', '!', '?')
        val CLOSERS = charArrayOf('"', '\'', ')', ']')
    }
}
