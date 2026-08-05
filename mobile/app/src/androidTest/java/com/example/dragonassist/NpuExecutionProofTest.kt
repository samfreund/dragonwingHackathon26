package com.example.dragonassist

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.dragonassist.transcribe.WhisperSession
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Evidence that inference really runs on the Hexagon NPU rather than silently on the CPU.
 *
 * Three independent arguments, strongest first:
 *
 *  1. A control experiment. The models are QNN EPContext graphs, and no CPU kernel exists
 *     for an EPContext node. Loading one *without* the QNN provider must therefore fail.
 *     If it ever succeeds, something is executing these graphs that is not QNN, and every
 *     other claim here is void.
 *  2. Timing. Whisper-Base's encoder on eight Arm cores is seconds, not ~160 ms.
 *  3. QNN's own log says `QnnDevice_create done ... status 0x0` and initialises the HTP.
 */
@RunWith(AndroidJUnit4::class)
class NpuExecutionProofTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun withoutQnnTheGraphCannotRunAtAll() {
        assumeTrue("models not pushed", WhisperSession.modelsPresent(context))

        val env = OrtEnvironment.getEnvironment()
        val path = File(WhisperSession.modelDir(context), WhisperSession.ENCODER).absolutePath

        try {
            // Default options: CPU execution provider only, no addQnn().
            env.createSession(path, OrtSession.SessionOptions()).use {
                fail(
                    "Encoder loaded WITHOUT the QNN provider. An EPContext graph has no CPU " +
                        "kernel, so this should be impossible — the NPU claim is unsound."
                )
            }
        } catch (e: OrtException) {
            println("control: CPU-only load correctly failed -> ${e.message?.take(160)}")
            assertTrue(
                "expected a missing-kernel error, got: ${e.message}",
                e.message?.contains("EPContext") == true ||
                    e.message?.contains("kernel", ignoreCase = true) == true,
            )
        }
    }

    @Test
    fun withQnnTheSameGraphLoadsAndRunsFast() {
        assumeTrue("models not pushed", WhisperSession.modelsPresent(context))

        WhisperSession.create(context).use { session ->
            val env = OrtEnvironment.getEnvironment()
            val mel = FloatArray(80 * 3000)

            // Warm up: the first run includes graph setup on the DSP.
            WhisperSession.float16Tensor(env, mel, longArrayOf(1, 80, 3000)).use { warm ->
                session.encoder.run(mapOf("input_features" to warm)).close()
            }

            val runs = 5
            val started = System.nanoTime()
            repeat(runs) {
                WhisperSession.float16Tensor(env, mel, longArrayOf(1, 80, 3000)).use { t ->
                    session.encoder.run(mapOf("input_features" to t)).close()
                }
            }
            val perRunMs = (System.nanoTime() - started) / 1_000_000 / runs
            println("encoder steady-state: ${perRunMs}ms/run over $runs runs")

            // Whisper-Base's encoder is ~20 GFLOPs per 30 s window. Sustaining that in a
            // few hundred ms is not achievable on this CPU; it is NPU-class throughput.
            assertTrue(
                "encoder averaged ${perRunMs}ms — too slow to be running on the NPU",
                perRunMs < 1_000,
            )
        }
    }
}
