package com.example.dragonassist.transcribe

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Owns the two ONNX Runtime sessions that make up Whisper, both bound to the QNN
 * execution provider so they run on the Hexagon NPU.
 *
 * The model files are *not* bundled in the APK — at ~192 MB they would make every install
 * painful. They are pushed to external files storage instead:
 *
 *   adb push <bundle>/. /sdcard/Android/data/com.example.dragonassist/files/whisper/
 *
 * Both `.onnx` files are EPContext wrappers that reference their `.bin` context binary by
 * *relative* path, so the pair must stay in the same directory.
 */
class WhisperSession private constructor(
    private val env: OrtEnvironment,
    val encoder: OrtSession,
    val decoder: OrtSession,
    val providersUsed: List<String>,
) : AutoCloseable {

    /**
     * True when these graphs are executing on the Hexagon NPU.
     *
     * This is a structural guarantee, not a heuristic. The models are QNN EPContext
     * graphs and ORT has no CPU kernel for an EPContext node, so if QNN fails to bind,
     * `createSession` throws `Failed to find kernel for com.microsoft.EPContext
     * (ep:'CPUExecutionProvider')` rather than quietly falling back. Holding a
     * constructed WhisperSession therefore means QNN bound.
     *
     * `NpuExecutionProofTest` pins this down with a control: loading the same file
     * without `addQnn()` must fail.
     *
     * Do not reimplement this as a scan of `OrtEnvironment.getAvailableProviders()` —
     * that reports which providers were compiled into the build, not which one ran.
     */
    val isOnNpu: Boolean get() = true

    override fun close() {
        runCatching { encoder.close() }
        runCatching { decoder.close() }
    }

    companion object {
        private const val TAG = "WhisperSession"
        const val ENCODER = "encoder.onnx"
        const val DECODER = "decoder.onnx"

        /**
         * Where the model files are expected. Also where the `.bin` context binaries live.
         *
         * Deliberately the external files directory itself rather than a subdirectory:
         * a directory created by `adb shell mkdir` is owned by `shell` with mode
         * `drwxrws---`, which the app cannot traverse. The files directory is created by
         * the framework and owned by the app, and files pushed into it arrive as 0666.
         */
        fun modelDir(context: Context): File =
            checkNotNull(context.getExternalFilesDir(null)) { "external storage unavailable" }

        fun modelsPresent(context: Context): Boolean {
            val dir = modelDir(context)
            return listOf(
                ENCODER, DECODER,
                "encoder_qairt_context.bin", "decoder_qairt_context.bin",
            ).all { File(dir, it).length() > 0 }
        }

        /**
         * Creates both sessions with the QNN execution provider.
         *
         * @throws IllegalStateException if the model files are missing.
         */
        fun create(context: Context): WhisperSession {
            val dir = modelDir(context)
            check(modelsPresent(context)) {
                "Whisper model files missing from ${dir.absolutePath} — push the AI Hub " +
                    "precompiled_qnn_onnx bundle there first"
            }

            // VERBOSE here is not just for us: ORT maps its severity onto QNN's own
            // logger, so this is what makes the QNN backend explain *which* config it
            // rejected instead of only returning QNN_DEVICE_ERROR_INVALID_CONFIG.
            val env = OrtEnvironment.getEnvironment(
                OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE,
                "dragonassist",
            )

            // The QNN backend and the DSP skel are dlopened by path. useLegacyPackaging
            // puts them in the app's extracted native library directory.
            val nativeLibDir = context.applicationInfo.nativeLibraryDir
            Log.i(TAG, "native lib dir: $nativeLibDir")

            // Do NOT set ADSP_LIBRARY_PATH here. ORT sets it itself when unset, and
            // warns "Using existing ADSP_LIBRARY_PATH setting ..., which may cause the
            // HTP backend to fail" if you have already done so. Tested: setting it makes
            // no difference to the failure below, and overrides a value ORT gets right.

            // Deliberately just the backend path. QNN detects SM8750 by itself
            // ("Detected Snapdragon SOC SM8750"), and setting htp_arch explicitly is
            // counterproductive: ORT rejects 79 outright even at 1.28
            // ("Invalid HTP architecture: 79") and 75 would be wrong for this device.
            //
            // If device creation ever fails here with QNN_DEVICE_ERROR_INVALID_CONFIG,
            // the cause is almost certainly not these options — it is the vendor DSP
            // library declaration in AndroidManifest.xml. Turn on verbose logging (see
            // above) and look for a dlopen failure in the QNN log; the generic enum hides
            // the real reason.
            val qnnOptions = mapOf(
                "backend_path" to "libQnnHtp.so",
            )

            fun open(fileName: String): Pair<OrtSession, List<String>> {
                val options = OrtSession.SessionOptions()
                options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE)
                options.setSessionLogVerbosityLevel(255)
                try {
                    options.addQnn(qnnOptions)
                } catch (e: Throwable) {
                    // Surface rather than silently degrade — a CPU fallback would be
                    // far too slow to be useful and must not go unnoticed.
                    Log.e(TAG, "QNN execution provider unavailable for $fileName", e)
                }
                val path = File(dir, fileName).absolutePath
                val session = env.createSession(path, options)
                val providers = runCatching {
                    session.inputNames // touch the session so it is fully initialised
                    OrtEnvironment.getAvailableProviders().map { it.name }
                }.getOrDefault(emptyList())
                Log.i(TAG, "opened $fileName; available providers=$providers")
                return session to providers
            }

            val (enc, providers) = open(ENCODER)
            val (dec, _) = open(DECODER)
            return WhisperSession(env, enc, dec, providers)
        }

        /** Packs float32 into an fp16 tensor — every Whisper graph input is float16. */
        fun float16Tensor(
            env: OrtEnvironment,
            values: FloatArray,
            shape: LongArray,
        ): OnnxTensor {
            val buffer = ByteBuffer.allocateDirect(values.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (v in values) buffer.putShort(floatToHalf(v))
            buffer.rewind()
            return OnnxTensor.createTensor(env, buffer, shape, OnnxJavaType.FLOAT16)
        }

        /** IEEE 754 binary32 to binary16, with round-to-nearest-even. */
        fun floatToHalf(value: Float): Short {
            val bits = java.lang.Float.floatToIntBits(value)
            val sign = (bits ushr 16) and 0x8000
            var mantissaExp = (bits and 0x7fffffff) + 0x1000  // rounding bias

            if (mantissaExp >= 0x47800000) {           // overflow, or Inf/NaN
                return if ((bits and 0x7fffffff) >= 0x7f800000) {
                    val nan = if (bits and 0x007fffff != 0) 0x0200 else 0
                    (sign or 0x7c00 or nan).toShort()
                } else {
                    (sign or 0x7bff).toShort()          // clamp to max finite half
                }
            }
            if (mantissaExp >= 0x38800000) {           // normal
                return (sign or ((mantissaExp - 0x38000000) ushr 13)).toShort()
            }
            if (mantissaExp < 0x33000000) return sign.toShort()  // underflow to zero

            mantissaExp = (bits and 0x7fffffff) ushr 23
            return (
                sign or (
                    ((bits and 0x7fffff) or 0x800000) +
                        (0x800000 ushr (mantissaExp - 102))
                    ushr (126 - mantissaExp)
                )
            ).toShort()
        }

        /** binary16 back to binary32, for reading model outputs. */
        fun halfToFloat(half: Short): Float {
            val h = half.toInt() and 0xFFFF
            val sign = h and 0x8000
            val exponent = h and 0x7c00
            val mantissa = h and 0x03ff

            return when (exponent) {
                0x7c00 -> java.lang.Float.intBitsToFloat((sign shl 16) or 0x7f800000 or (mantissa shl 13))
                0 -> if (mantissa == 0) {
                    java.lang.Float.intBitsToFloat(sign shl 16)
                } else {
                    // Subnormal: normalise it.
                    var m = mantissa
                    var e = -1
                    do {
                        m = m shl 1
                        e++
                    } while (m and 0x400 == 0)
                    val exp = 127 - 15 - e
                    java.lang.Float.intBitsToFloat(
                        (sign shl 16) or (exp shl 23) or ((m and 0x3ff) shl 13)
                    )
                }
                else -> java.lang.Float.intBitsToFloat(
                    (sign shl 16) or (((exponent shr 10) - 15 + 127) shl 23) or (mantissa shl 13)
                )
            }
        }
    }
}
