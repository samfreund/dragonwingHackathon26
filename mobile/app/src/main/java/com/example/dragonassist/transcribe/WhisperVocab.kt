package com.example.dragonassist.transcribe

import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

/**
 * Whisper's detokenizer, backed by the byte table exported by `export_tokenizer.py`.
 *
 * Whisper uses byte-level BPE, so a token maps to raw *bytes* rather than text and a
 * single UTF-8 character can span several tokens. Decoding therefore concatenates bytes
 * across the whole sequence and decodes UTF-8 once at the end — decoding per token would
 * corrupt any non-ASCII output.
 *
 * Control tokens (`<|startoftranscript|>`, `<|en|>`, timestamps, …) carry empty byte
 * strings in the table, so they drop out of the text without needing a special case.
 */
class WhisperVocab(
    private val tokens: Array<ByteArray>,
    val special: SpecialTokens,
) {

    val vocabSize: Int get() = tokens.size

    /** Decodes ids to text, skipping control tokens. */
    fun decode(ids: List<Int>): String {
        val out = java.io.ByteArrayOutputStream()
        for (id in ids) {
            if (id !in tokens.indices) continue
            out.write(tokens[id])
        }
        return String(out.toByteArray(), StandardCharsets.UTF_8).trim()
    }

    /** Raw bytes for one token; empty for control tokens. Exposed for tests. */
    fun bytesOf(id: Int): ByteArray = tokens[id]

    /**
     * The prompt Whisper expects before it will emit text: start-of-transcript, then
     * language, task, and timestamp mode. Getting this wrong is a common cause of
     * empty or garbled output.
     */
    fun initialTokens(): IntArray = intArrayOf(
        special.sot,
        special.langEn,
        special.transcribe,
        special.noTimestamps,
    )

    data class SpecialTokens(
        val sot: Int,
        val eot: Int,
        val transcribe: Int,
        val translate: Int,
        val noTimestamps: Int,
        val langEn: Int,
        val meanDecodeLen: Int,
    )

    companion object {
        const val TOKENS_ASSET = "whisper_tokens.bin"
        const val META_ASSET = "whisper_tokens.json"

        /** Reads `[u32 count]( [u16 len][bytes] ) * count`. */
        fun readTokens(stream: InputStream): Array<ByteArray> = stream.use { input ->
            val bytes = input.readBytes()
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val count = buffer.int
            require(count in 1..1_000_000) { "implausible vocab size $count" }
            Array(count) {
                val len = buffer.short.toInt() and 0xFFFF
                ByteArray(len).also { b -> buffer.get(b) }
            }
        }

        fun readSpecial(json: String): SpecialTokens {
            fun field(name: String): Int =
                Regex("\"$name\"\\s*:\\s*(-?\\d+)").find(json)
                    ?.groupValues?.get(1)?.toInt()
                    ?: error("missing \"$name\" in tokenizer metadata")

            return SpecialTokens(
                sot = field("sot"),
                eot = field("eot"),
                transcribe = field("transcribe"),
                translate = field("translate"),
                noTimestamps = field("no_timestamps"),
                langEn = field("lang_en"),
                meanDecodeLen = field("mean_decode_len"),
            )
        }
    }
}
