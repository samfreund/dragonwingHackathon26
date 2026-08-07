package com.example.dragonassist.net

import android.util.Log
import com.example.dragonassist.vlm.VlmClient
import com.example.dragonassist.vlm.VlmConfig
import java.util.UUID

/**
 * Checks every hop the system depends on, and names the one that is broken.
 *
 * dragonAssist spans four devices, and a partial failure is worse than a total one: if the
 * phone quietly answered from the board alone it would look like a working demo while the
 * laptop and cloud were doing nothing. This exists so that never has to be inferred from
 * latency or guessed at from logs.
 *
 * Each link is checked in the order the data actually flows, and a failure stops the chain
 * — there is no point testing whether the laptop can answer if nothing can write to it.
 */
object SystemHealth {

    enum class State { Ok, Failed, Skipped }

    data class Link(
        val name: String,
        val state: State,
        val detail: String,
    ) {
        val ok: Boolean get() = state == State.Ok
    }

    data class Report(val links: List<Link>) {
        val healthy: Boolean get() = links.all { it.ok }

        /** The first broken link — the one worth fixing. */
        val firstFailure: Link? get() = links.firstOrNull { it.state == State.Failed }

        fun summary(): String = links.joinToString("\n") { link ->
            val mark = when (link.state) {
                State.Ok -> "OK"
                State.Failed -> "FAIL"
                State.Skipped -> "--"
            }
            "$mark  ${link.name}: ${link.detail}"
        }
    }

    private const val TAG = "SystemHealth"

    /**
     * Runs the full preflight.
     *
     * The last check submits a real question against a throwaway session. That is the only
     * way to prove the laptop *worker* is running: the server acknowledges queries whether
     * or not anything is there to answer them, so a `query_ack` alone proves nothing.
     */
    suspend fun check(config: VlmConfig): Report {
        val links = mutableListOf<Link>()

        // 1. The board — vision, and the source of narration.
        val boardOk = runCatching {
            VlmClient(config).use { client ->
                val ready = client.connect()
                val status = client.status()
                require(status.optBoolean("up")) { "GenieX reports itself down" }
                ready.optString("model").ifEmpty { "model unknown" }
            }
        }
        links += if (boardOk.isSuccess) {
            Link("IQ-9075 vision", State.Ok, boardOk.getOrNull().orEmpty())
        } else {
            Link("IQ-9075 vision", State.Failed, boardOk.exceptionOrNull()?.message.orEmpty())
        }

        if (!config.hasLaptop) {
            links += Link("Laptop", State.Failed, "no laptop_host in vlm.properties")
            return Report(links)
        }

        // 2. The context endpoint — where narration has to land.
        val probeSession = "health-${UUID.randomUUID().toString().take(8)}"
        val contextOk = runCatching {
            ContextStreamClient(config.contextUrl, config.streamToken).use { it.open(probeSession) }
        }
        links += if (contextOk.isSuccess) {
            Link("Laptop context write", State.Ok, "/v1/iq9 accepting")
        } else {
            Link("Laptop context write", State.Failed, contextOk.exceptionOrNull()?.message.orEmpty())
        }

        if (contextOk.isFailure) {
            links += Link("Laptop answers", State.Skipped, "context endpoint unreachable")
            return Report(links)
        }

        // 3. End to end: write known text, ask about it, require an answer back. This is
        //    the check that catches a running server with no worker behind it.
        val answerOk = runCatching {
            ContextStreamClient(config.contextUrl, config.streamToken).use { stream ->
                stream.open(probeSession)
                stream.append(CANARY_TEXT)
                stream.end()
            }
            TextQaClient(config.textQaUrl, config.phoneToken, "health-check").use { qa ->
                val answer = qa.ask(probeSession, CANARY_QUESTION, "health-$probeSession")
                answer.route?.let { "answered via $it" } ?: "answered"
            }
        }
        links += if (answerOk.isSuccess) {
            Link("Laptop answers", State.Ok, answerOk.getOrNull().orEmpty())
        } else {
            val message = answerOk.exceptionOrNull()?.message.orEmpty()
            Link(
                "Laptop answers",
                State.Failed,
                // The overwhelmingly common cause, and invisible from the protocol alone.
                if (message.contains("timeout", true) || message.contains("Timed out", true)) {
                    "no result — is laptop_worker running?"
                } else {
                    message
                },
            )
        }

        return Report(links).also { Log.i(TAG, "\n" + it.summary()) }
    }

    private const val CANARY_TEXT =
        "The dragonAssist preflight canary phrase is amber lantern."
    private const val CANARY_QUESTION =
        "What is the dragonAssist preflight canary phrase?"
}
