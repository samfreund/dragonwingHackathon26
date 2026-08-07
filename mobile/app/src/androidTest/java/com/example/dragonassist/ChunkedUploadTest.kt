package com.example.dragonassist

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.dragonassist.vlm.VlmClient
import com.example.dragonassist.vlm.VlmConfig
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Exercises the chunked binary upload path against the real board.
 *
 * Deliberately uses a photo rather than a video: the mechanics that can go wrong are the
 * protocol ones — declaring an exact `size`, framing the body, honouring backpressure,
 * and ending on `media` — and none of those care what the bytes contain. The server
 * classifies by content via ffprobe, so a JPEG sent down the binary path is prepared as
 * an image and answers the same questions.
 *
 * Video adds only ffmpeg frame sampling on the board, which is Sam's code and already
 * exercised by his own tests.
 */
@RunWith(AndroidJUnit4::class)
class ChunkedUploadTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private fun capture(): File = File(context.filesDir, "captures/capture.jpg")

    @Test
    fun streamsAFileInChunksAndReportsProgress() = runBlocking {
        assumeTrue("vlm.properties not pushed", VlmConfig.isConfigured(context))
        val source = capture()
        assumeTrue("no capture.jpg — take a photo in the app first", source.isFile)

        val config = requireNotNull(VlmConfig.load(context))
        val expectedChunks = (source.length() + VlmClient.CHUNK_BYTES - 1) / VlmClient.CHUNK_BYTES
        println("streaming ${source.length()} bytes in ~$expectedChunks chunks")

        VlmClient(config).use { client ->
            client.connect()

            val progress = mutableListOf<Pair<Long, Long>>()
            val started = System.currentTimeMillis()
            val media = client.uploadFile(source, name = "chunked.jpg") { sent, total ->
                progress += sent to total
            }
            val elapsed = System.currentTimeMillis() - started

            val throughput = source.length() / 1024.0 / 1024.0 / (elapsed / 1000.0)
            println("uploaded in ${elapsed}ms (%.2f MB/s), ${progress.size} progress callbacks"
                .format(throughput))
            println("media: ${media.optString("description")}")

            assertTrue("no progress reported", progress.isNotEmpty())

            // Progress must be monotonic and finish exactly at the declared size, since
            // that is the only thing telling the server the upload is complete.
            var previous = -1L
            for ((sent, total) in progress) {
                assertEquals("total changed mid-upload", source.length(), total)
                assertTrue("progress went backwards: $previous -> $sent", sent >= previous)
                previous = sent
            }
            assertEquals("final progress must equal the file size", source.length(), previous)
        }
    }

    @Test
    fun aChunkedUploadCanThenBeQuestioned() = runBlocking {
        assumeTrue("vlm.properties not pushed", VlmConfig.isConfigured(context))
        val source = capture()
        assumeTrue("no capture.jpg", source.isFile)

        val config = requireNotNull(VlmConfig.load(context))
        VlmClient(config).use { client ->
            client.connect()
            client.uploadFile(source, name = "chunked.jpg")

            val answer = client.ask("Describe this image in one short sentence.")
            println("ANSWER: ${answer.text}")
            assertTrue("empty answer after a chunked upload", answer.text.isNotBlank())
        }
    }
}
