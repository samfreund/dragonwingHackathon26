package com.example.dragonassist

import com.example.dragonassist.transcribe.WhisperVocab
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Checks the Kotlin detokenizer against HuggingFace's, using sequences produced by
 * `WhisperTokenizer` itself. The accented and emoji cases matter most: they span
 * multiple tokens per character, so they catch the classic mistake of decoding UTF-8
 * per token instead of once over the whole byte stream.
 */
class WhisperVocabTest {

    private fun resource(name: String) =
        checkNotNull(javaClass.classLoader?.getResourceAsStream(name)) { "missing fixture $name" }

    private fun vocab(): WhisperVocab {
        val tokens = WhisperVocab.readTokens(resource(WhisperVocab.TOKENS_ASSET))
        val meta = resource(WhisperVocab.META_ASSET).use { it.readBytes().decodeToString() }
        return WhisperVocab(tokens, WhisperVocab.readSpecial(meta))
    }

    /** Minimal parse of the fixture — avoids adding a JSON dependency for one test. */
    private fun cases(): List<Pair<List<Int>, String>> {
        val json = resource("vocab_cases.json").use { it.readBytes().decodeToString() }
        val idBlocks = Regex("\"ids\"\\s*:\\s*\\[([^]]*)]").findAll(json).map { m ->
            m.groupValues[1].split(",").mapNotNull { it.trim().toIntOrNull() }
        }.toList()
        val texts = Regex("\"text\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"").findAll(json).map { m ->
            m.groupValues[1]
                .replace("\\\"", "\"")
                .replace("\\\\", "\\")
                .replace("\\n", "\n")
        }.toList()
        assertEquals("fixture parse mismatch", idBlocks.size, texts.size)
        return idBlocks.zip(texts)
    }

    @Test
    fun `vocab size matches the decoder logits dimension`() {
        assertEquals(51865, vocab().vocabSize)
    }

    @Test
    fun `special token ids are the ones whisper expects`() {
        val s = vocab().special
        assertEquals(50258, s.sot)
        assertEquals(50257, s.eot)
        assertEquals(50259, s.langEn)
        assertEquals(50359, s.transcribe)
        assertEquals(50363, s.noTimestamps)
        assertEquals(200, s.meanDecodeLen)
    }

    @Test
    fun `decodes every reference sequence identically to huggingface`() {
        val v = vocab()
        val all = cases()
        assertTrue("expected fixtures", all.isNotEmpty())
        for ((ids, expected) in all) {
            assertEquals("decoding $ids", expected, v.decode(ids))
        }
    }

    @Test
    fun `control tokens contribute no text`() {
        val v = vocab()
        val s = v.special
        for (id in listOf(s.sot, s.eot, s.langEn, s.transcribe, s.noTimestamps)) {
            assertEquals("token $id should emit nothing", 0, v.bytesOf(id).size)
        }
        assertEquals("", v.decode(v.initialTokens().toList()))
    }

    @Test
    fun `initial prompt is sot language task timestamps`() {
        val v = vocab()
        val s = v.special
        assertArrayEquals(
            intArrayOf(s.sot, s.langEn, s.transcribe, s.noTimestamps),
            v.initialTokens(),
        )
    }

    private fun assertArrayEquals(expected: IntArray, actual: IntArray) =
        assertEquals(expected.toList(), actual.toList())
}
