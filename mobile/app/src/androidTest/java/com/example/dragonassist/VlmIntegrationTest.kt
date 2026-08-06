package com.example.dragonassist

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.dragonassist.vlm.VlmClient
import com.example.dragonassist.vlm.VlmConfig
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * The phone talking to the VLM server on the IQ-9075, over Tailscale.
 *
 * Needs `vlm.properties` pushed to the app's external files directory — the token is a
 * shared secret and is deliberately not in the repo. Skips rather than fails when absent,
 * so the suite still runs on a machine without board access.
 */
@RunWith(AndroidJUnit4::class)
class VlmIntegrationTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    /** A shape the model cannot be vague about, so a wrong answer is obviously wrong. */
    private fun testImage(): ByteArray {
        val bitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
        Canvas(bitmap).apply {
            drawColor(Color.WHITE)
            drawCircle(320f, 200f, 120f, Paint().apply { color = Color.RED; isAntiAlias = true })
            drawText(
                "DRAGON", 170f, 420f,
                Paint().apply {
                    color = Color.BLACK
                    textSize = 72f
                    isAntiAlias = true
                },
            )
        }
        val jpeg = ByteArrayOutputStream().also {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it)
        }.toByteArray()

        // Keep a copy on disk. Otherwise the only artefacts of a run are log lines, and
        // "did the model actually see what I think it saw?" is unanswerable.
        runCatching {
            File(context.getExternalFilesDir(null), "probe_sent.jpg").writeBytes(jpeg)
        }
        return jpeg
    }

    /** Appends a Q&A pair to a transcript file so runs can be reviewed afterwards. */
    private fun record(question: String, answer: String, detail: String = "") {
        runCatching {
            File(context.getExternalFilesDir(null), "vlm_answers.txt").appendText(
                buildString {
                    appendLine("Q: $question")
                    appendLine("A: $answer")
                    if (detail.isNotEmpty()) appendLine("   ($detail)")
                    appendLine()
                }
            )
        }
    }

    @Test
    fun connectsAuthenticatesAndReportsServerStatus() = runBlocking {
        assumeTrue("vlm.properties not pushed", VlmConfig.isConfigured(context))
        val config = requireNotNull(VlmConfig.load(context))
        println("connecting to ${config.url}")

        VlmClient(config).use { client ->
            val ready = client.connect()
            println("ready: model=${ready.optString("model")} protocol=${ready.optInt("protocol")}")

            val status = client.status()
            println("status: up=${status.optBoolean("up")} models=${status.optJSONArray("models")}")
            assertTrue("server reports itself down", status.optBoolean("up"))
        }
    }

    @Test
    fun uploadsAnImageAndGetsAStreamedAnswer() = runBlocking {
        assumeTrue("vlm.properties not pushed", VlmConfig.isConfigured(context))
        val config = requireNotNull(VlmConfig.load(context))

        VlmClient(config).use { client ->
            client.connect()

            val jpeg = testImage()
            println("uploading ${jpeg.size} bytes")
            val media = client.uploadImage(jpeg, "probe.jpg")
            println("media: ${media.optString("description")}")

            var firstTokenMs = -1L
            val started = System.currentTimeMillis()
            val answer = client.ask(
                prompt = "Describe this image in one sentence. " +
                    "What shape and colour is it, and what word is written?",
                onQueued = { println("queued behind another request") },
                onToken = {
                    if (firstTokenMs < 0) {
                        firstTokenMs = System.currentTimeMillis() - started
                        println("first token after ${firstTokenMs}ms")
                    }
                },
            )

            println("ANSWER: ${answer.text}")
            record(
                "Describe this image in one sentence.",
                answer.text,
                "first token ${firstTokenMs}ms, server ${answer.latencySeconds}s",
            )
            println("latency: ${answer.latencySeconds}s (server), " +
                "${System.currentTimeMillis() - started}ms wall")

            assertTrue("empty answer", answer.text.isNotBlank())
            assertTrue(
                "answer never mentioned red — the model may not have seen the image: " +
                    answer.text,
                answer.text.contains("red", ignoreCase = true),
            )
            assertTrue("no tokens streamed; the UI would show a spinner instead of text",
                firstTokenMs >= 0)
        }
    }

    @Test
    fun aSecondQuestionReusesTheSameUploadedImage() = runBlocking {
        assumeTrue("vlm.properties not pushed", VlmConfig.isConfigured(context))
        val config = requireNotNull(VlmConfig.load(context))

        // One connection is one conversation: uploading once and asking twice is the
        // shape the app will actually use.
        VlmClient(config).use { client ->
            client.connect()
            client.uploadImage(testImage(), "probe.jpg")

            val first = client.ask("What colour is the shape?")
            println("Q1 -> ${first.text}")
            record("What colour is the shape?", first.text)
            val second = client.ask("What word is written in the image?")
            println("Q2 -> ${second.text}")
            record("What word is written in the image?", second.text)

            assertTrue("second question failed", second.text.isNotBlank())
        }
    }

    @Test
    fun pingRoundTripsQuickly() = runBlocking {
        assumeTrue("vlm.properties not pushed", VlmConfig.isConfigured(context))
        val config = requireNotNull(VlmConfig.load(context))

        VlmClient(config).use { client ->
            client.connect()
            // Three pings: the first can include TCP/TLS warm-up on the tailnet path.
            val samples = (1..3).map { client.ping() }
            println("ping round trips: ${samples.joinToString("ms, ")}ms")
            assertTrue("no pong received", samples.isNotEmpty())
            assertTrue(
                "ping took ${samples.min()}ms — the tailnet path is probably relayed",
                samples.min() < 2_000,
            )
        }
    }
}
