package com.example.dragonassist.transcribe

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import android.content.Context
import android.util.Log
import com.example.dragonassist.audio.MelSpectrogram
import com.example.dragonassist.audio.Recording
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.IntBuffer

/**
 * Whisper on the Hexagon NPU: audio in, text out.
 *
 * The pipeline is
 *   samples -> [MelSpectrogram] -> (1,80,3000) -> [encoder] -> 12 cross-attention caches
 *           -> [decoder loop, greedy] -> token ids -> [WhisperVocab] -> text
 *
 * The decoder is autoregressive with a KV cache: each step consumes the previous token,
 * an attention mask, 12 self-attention caches and the 12 cross-attention caches, and
 * returns logits plus 12 updated self caches. Ported from `HfWhisperApp` in
 * qai_hub_models, which is the reference for exact behaviour.
 */
class WhisperTranscriber(private val context: Context) : Transcriber {

    override val name = "Whisper-Base · Hexagon NPU"

    override var isReady = false
        private set

    private var session: WhisperSession? = null
    private var mel: MelSpectrogram? = null
    private var vocab: WhisperVocab? = null

    /** Set after a transcription; useful for the UI and for spotting CPU fallback. */
    var lastTimings: Timings? = null
        private set

    data class Timings(
        val melMs: Long,
        val encoderMs: Long,
        val decoderMs: Long,
        val tokens: Int,
    ) {
        val totalMs: Long get() = melMs + encoderMs + decoderMs
    }

    override suspend fun prepare() {
        if (isReady) return

        check(WhisperSession.modelsPresent(context)) {
            "Whisper models are not on the device. Push the AI Hub precompiled_qnn_onnx " +
                "bundle to ${WhisperSession.modelDir(context).absolutePath}"
        }

        mel = MelSpectrogram(
            MelSpectrogram.readFilters(context.assets.open(MelSpectrogram.FILTERS_ASSET))
        )
        vocab = WhisperVocab(
            WhisperVocab.readTokens(context.assets.open(WhisperVocab.TOKENS_ASSET)),
            WhisperVocab.readSpecial(
                context.assets.open(WhisperVocab.META_ASSET).use { it.readBytes().decodeToString() }
            ),
        )
        session = WhisperSession.create(context).also {
            if (!it.isOnNpu) {
                Log.w(TAG, "QNN did not bind — running on CPU, this will be very slow")
            }
        }
        isReady = true
    }

    override suspend fun transcribe(recording: Recording): String {
        if (!isReady) prepare()
        if (recording.isEmpty) throw TranscriptionException("Nothing was recorded.")

        val session = session ?: throw TranscriptionException("Session not initialised")
        val mel = mel ?: throw TranscriptionException("Mel front end not initialised")
        val vocab = vocab ?: throw TranscriptionException("Vocabulary not initialised")
        val env = OrtEnvironment.getEnvironment()

        val melStarted = System.nanoTime()
        val melValues = mel.compute(recording.samples)
        val melMs = (System.nanoTime() - melStarted) / 1_000_000

        val encoderStarted = System.nanoTime()
        val melTensor = WhisperSession.float16Tensor(env, melValues, longArrayOf(1, 80, 3000))

        return melTensor.use { input ->
            session.encoder.run(mapOf(session.encoder.inputNames.first() to input))
                .use { encoded ->
                    val encoderMs = (System.nanoTime() - encoderStarted) / 1_000_000

                    // Cross caches stay alive for the whole loop — 18 MB that would
                    // otherwise be copied 199 times.
                    val crossCaches = buildMap {
                        for (outputName in session.encoder.outputNames) {
                            put(outputName, encoded.get(outputName).get() as OnnxTensor)
                        }
                    }

                    val decoderStarted = System.nanoTime()
                    val ids = decode(env, session, vocab, crossCaches)
                    val decoderMs = (System.nanoTime() - decoderStarted) / 1_000_000

                    lastTimings = Timings(melMs, encoderMs, decoderMs, ids.size)
                    Log.i(
                        TAG,
                        "mel ${melMs}ms · encoder ${encoderMs}ms · " +
                            "decoder ${decoderMs}ms for ${ids.size} tokens",
                    )

                    vocab.decode(ids).ifBlank {
                        "(no speech detected)"
                    }
                }
        }
    }

    /**
     * Greedy autoregressive decode. Mirrors `_transcribe_single_chunk`: start from
     * `<|startoftranscript|>` and let the model emit its own language and task tokens,
     * revealing one more attention position per step, until end-of-transcript.
     */
    private fun decode(
        env: OrtEnvironment,
        session: WhisperSession,
        vocab: WhisperVocab,
        crossCaches: Map<String, OnnxTensor>,
    ): List<Int> {
        val special = vocab.special
        val steps = special.meanDecodeLen - 1        // 199 cache slots

        // Self-attention caches, reused across steps to avoid churning ~2.4 MB per token.
        val selfK = Array(LAYERS) { directBuffer(HEADS * HEAD_DIM * steps) }
        val selfV = Array(LAYERS) { directBuffer(HEADS * steps * HEAD_DIM) }

        // Everything is masked initially; one position is revealed per step.
        val maskBuffer = directBuffer(special.meanDecodeLen)
        val maskedOut = WhisperSession.floatToHalf(MASK_NEG)
        for (i in 0 until special.meanDecodeLen) maskBuffer.putShort(i * 2, maskedOut)

        val ids = mutableListOf(special.sot)
        var current = special.sot

        val kShape = longArrayOf(HEADS.toLong(), 1, HEAD_DIM.toLong(), steps.toLong())
        val vShape = longArrayOf(HEADS.toLong(), 1, steps.toLong(), HEAD_DIM.toLong())
        val maskShape = longArrayOf(1, 1, 1, special.meanDecodeLen.toLong())

        for (step in 0 until steps) {
            // Reveal the position this token occupies. Indexed from the end, matching
            // the reference: mask[mean_decode_len - step - 1] = 0.
            maskBuffer.putShort((special.meanDecodeLen - step - 1) * 2, ZERO_HALF)

            val tensors = mutableListOf<OnnxTensor>()
            fun track(t: OnnxTensor): OnnxTensor = t.also { tensors.add(it) }

            try {
                val inputs = HashMap<String, OnnxTensor>(27)
                inputs["input_ids"] = track(
                    OnnxTensor.createTensor(
                        env, IntBuffer.wrap(intArrayOf(current)), longArrayOf(1, 1),
                    )
                )
                inputs["attention_mask"] = track(
                    OnnxTensor.createTensor(env, maskBuffer.duplicate(), maskShape, FP16)
                )
                inputs["position_ids"] = track(
                    OnnxTensor.createTensor(
                        env, IntBuffer.wrap(intArrayOf(step)), longArrayOf(1),
                    )
                )
                for (layer in 0 until LAYERS) {
                    inputs["k_cache_self_${layer}_in"] = track(
                        OnnxTensor.createTensor(env, selfK[layer].duplicate(), kShape, FP16)
                    )
                    inputs["v_cache_self_${layer}_in"] = track(
                        OnnxTensor.createTensor(env, selfV[layer].duplicate(), vShape, FP16)
                    )
                }
                inputs.putAll(crossCaches)

                session.decoder.run(inputs).use { out ->
                    val logits = out.get("logits").get() as OnnxTensor
                    val next = argmaxHalf(logits.shortBuffer, vocab.vocabSize)

                    if (next == special.eot) return ids
                    ids.add(next)
                    current = next

                    // Carry the updated caches into the next step.
                    for (layer in 0 until LAYERS) {
                        copyInto(selfK[layer], out.get("k_cache_self_${layer}_out").get() as OnnxTensor)
                        copyInto(selfV[layer], out.get("v_cache_self_${layer}_out").get() as OnnxTensor)
                    }
                }
            } finally {
                // Cross caches are owned by the encoder result — never close them here.
                tensors.forEach { runCatching { it.close() } }
            }
        }
        return ids
    }

    private fun directBuffer(halfCount: Int): ByteBuffer =
        ByteBuffer.allocateDirect(halfCount * 2).order(ByteOrder.LITTLE_ENDIAN)

    private fun copyInto(target: ByteBuffer, source: OnnxTensor) {
        target.clear()
        target.put(source.byteBuffer)
        target.clear()
    }

    /** Argmax over fp16 logits without materialising 51,865 floats. */
    private fun argmaxHalf(logits: java.nio.ShortBuffer, count: Int): Int {
        var bestIndex = 0
        var best = Float.NEGATIVE_INFINITY
        for (i in 0 until count) {
            val value = WhisperSession.halfToFloat(logits.get(i))
            if (value > best) {
                best = value
                bestIndex = i
            }
        }
        return bestIndex
    }

    override fun close() {
        session?.close()
        session = null
        mel = null
        vocab = null
        isReady = false
    }

    private companion object {
        const val TAG = "WhisperTranscriber"
        const val LAYERS = 6          // config.decoder_layers
        const val HEADS = 8           // config.decoder_attention_heads
        const val HEAD_DIM = 64       // d_model / heads = 512 / 8
        const val MASK_NEG = -100.0f  // config.mask_neg
        val FP16: OnnxJavaType = OnnxJavaType.FLOAT16
        val ZERO_HALF: Short = 0
    }
}
