package com.example.dragonassist

import com.example.dragonassist.speak.SentenceBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The buffer decides when a streamed answer has enough text to speak naturally.
 *
 * Getting it wrong is audible rather than visible: split too eagerly and the synthesiser
 * reads "3." and "9" as separate utterances; split too late and speech doesn't start
 * until the answer has finished, wasting the streaming entirely.
 */
class SentenceBufferTest {

    /** Feeds text the way the board does — in fragments, not whole words. */
    private fun stream(buffer: SentenceBuffer, text: String, chunk: Int = 3): List<String> {
        val spoken = mutableListOf<String>()
        var i = 0
        while (i < text.length) {
            val piece = text.substring(i, minOf(i + chunk, text.length))
            spoken += buffer.append(piece)
            i += chunk
        }
        return spoken
    }

    @Test
    fun `releases each sentence as it completes`() {
        val buffer = SentenceBuffer()
        val spoken = stream(buffer, "A red circle. The word DRAGON is below it. ")
        assertEquals(
            listOf("A red circle.", "The word DRAGON is below it."),
            spoken,
        )
    }

    @Test
    fun `holds the final sentence until flushed`() {
        val buffer = SentenceBuffer()
        // No trailing space, which is how a streamed answer actually ends.
        val spoken = stream(buffer, "There is a laptop on the desk.")
        assertTrue("should not release before the boundary is confirmed", spoken.isEmpty())
        assertEquals("There is a laptop on the desk.", buffer.flush())
    }

    @Test
    fun `does not split decimals`() {
        val buffer = SentenceBuffer()
        val spoken = stream(buffer, "It took 3.9 seconds to answer. ")
        assertEquals(listOf("It took 3.9 seconds to answer."), spoken)
    }

    @Test
    fun `keeps closing punctuation with its sentence`() {
        val buffer = SentenceBuffer()
        val spoken = stream(buffer, "Is that a laptop? Yes! ")
        assertEquals(listOf("Is that a laptop?", "Yes!"), spoken)
    }

    @Test
    fun `forces a break on a run-on answer`() {
        val buffer = SentenceBuffer(maxChars = 40)
        val text = "a ".repeat(60) // no punctuation at all
        val spoken = stream(buffer, text)
        assertTrue("a run-on answer must still be spoken", spoken.isNotEmpty())
        assertTrue(
            "forced breaks should land on word boundaries",
            spoken.none { it.endsWith(" a") && it.length > 40 },
        )
    }

    @Test
    fun `flush returns null when nothing is pending`() {
        val buffer = SentenceBuffer()
        stream(buffer, "Done. ")
        assertNull(buffer.flush())
    }

    @Test
    fun `clear discards pending text`() {
        val buffer = SentenceBuffer()
        buffer.append("An unfinished thought")
        buffer.clear()
        assertNull(buffer.flush())
    }

    @Test
    fun `reassembles exactly what arrived`() {
        val buffer = SentenceBuffer()
        val answer = "A person sits at a desk. A screen shows a pricing page. It is 3.9 m away."
        val spoken = stream(buffer, answer, chunk = 2) + listOfNotNull(buffer.flush())
        assertEquals(
            "no text may be lost or duplicated between sentences",
            answer.replace(" ", ""),
            spoken.joinToString("").replace(" ", ""),
        )
    }
}
