package com.example.dragonassist

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.dragonassist.net.ContextStreamClient
import com.example.dragonassist.net.TextQaClient
import com.example.dragonassist.net.TransportException
import com.example.dragonassist.vlm.VlmConfig
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.UUID

/**
 * The phone against Krishna's `stream_transport` service.
 *
 * Exercises both endpoints on the one port: `/v1/iq9` to write session context, and
 * `/v1/phone` to ask questions about it. Skips when no laptop is configured, so the suite
 * still runs with only the board available.
 */
@RunWith(AndroidJUnit4::class)
class LaptopTransportTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private fun config(): VlmConfig? =
        VlmConfig.load(context)?.takeIf { it.hasLaptop }

    private fun session() = "test-${UUID.randomUUID().toString().take(8)}"

    @Test
    fun writesContextAndResumesFromTheServersSequence() = runBlocking {
        val cfg = config()
        assumeTrue("no laptop_host in vlm.properties", cfg != null)
        requireNotNull(cfg)

        val id = session()
        ContextStreamClient(cfg.contextUrl, cfg.streamToken).use { stream ->
            assertEquals("a fresh session starts at sequence 1", 1, stream.open(id))
            stream.append("The patient says their right knee has hurt for three days. ")
            stream.append("The pain is worse when climbing stairs. ")
            stream.end()
        }

        // Reopening must resume, not restart — otherwise a reconnect would duplicate
        // everything already written.
        ContextStreamClient(cfg.contextUrl, cfg.streamToken).use { stream ->
            val next = stream.open(id)
            println("resumed at sequence $next")
            assertTrue("server should have advanced past 1, got $next", next > 1)
        }
    }

    @Test
    fun asksAQuestionAndGetsAnAnswerWithItsRoute() = runBlocking {
        val cfg = config()
        assumeTrue("no laptop_host in vlm.properties", cfg != null)
        requireNotNull(cfg)

        val id = session()
        ContextStreamClient(cfg.contextUrl, cfg.streamToken).use { stream ->
            stream.open(id)
            stream.append(
                "The patient says their right knee has hurt for three days. " +
                    "The pain is worse when climbing stairs. No fall or injury."
            )
            stream.end()
        }

        TextQaClient(cfg.textQaUrl, cfg.phoneToken, "instrumented-test").use { qa ->
            val requestId = "req-${UUID.randomUUID()}"
            var queued = false
            val started = System.currentTimeMillis()
            val answer = qa.ask(id, "How long has the knee hurt?", requestId) { queued = true }
            val elapsed = System.currentTimeMillis() - started

            println("answer: ${answer.answer}")
            println("route: ${answer.route}  sources: ${answer.sources}  ${elapsed}ms  queued=$queued")

            assertTrue("empty answer", answer.answer.isNotBlank())
            assertEquals("request id must round-trip", requestId, answer.requestId)
        }
    }

    @Test
    fun aStoredAnswerSurvivesLosingTheConnection() = runBlocking {
        val cfg = config()
        assumeTrue("no laptop_host in vlm.properties", cfg != null)
        requireNotNull(cfg)

        val id = session()
        ContextStreamClient(cfg.contextUrl, cfg.streamToken).use { stream ->
            stream.open(id)
            stream.append("The medication is amoxicillin 500mg, three times a day.")
            stream.end()
        }

        val requestId = "req-${UUID.randomUUID()}"

        // Ask on one connection...
        TextQaClient(cfg.textQaUrl, cfg.phoneToken, "instrumented-test").use { qa ->
            qa.ask(id, "What is the dosage?", requestId)
        }

        // ...and retrieve it on a completely new one. This is the recovery path for the
        // network dropping mid-answer, which has already cost us a real result once.
        TextQaClient(cfg.textQaUrl, cfg.phoneToken, "instrumented-test").use { qa ->
            val recovered = qa.retrieve(requestId)
            println("recovered after reconnect: ${recovered.answer}")
            assertEquals(requestId, recovered.requestId)
            assertTrue("recovered answer was empty", recovered.answer.isNotBlank())
        }
    }

    @Test
    fun theWrongTokenIsRefused() = runBlocking {
        val cfg = config()
        assumeTrue("no laptop_host in vlm.properties", cfg != null)
        requireNotNull(cfg)

        // The two endpoints have deliberately different secrets: the stream token can
        // write context, the phone token can only ask. Using one for the other must fail.
        TextQaClient(cfg.textQaUrl, cfg.streamToken, "instrumented-test").use { qa ->
            try {
                qa.connect()
                fail("the stream token should not be accepted on /v1/phone")
            } catch (e: TransportException) {
                println("correctly refused: ${e.code} — ${e.message}")
            }
        }
    }
}
