package com.example.dragonassist

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.dragonassist.net.SystemHealth
import com.example.dragonassist.vlm.VlmConfig
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Runs the preflight against whatever is actually deployed and prints the per-link report.
 *
 * This is deliberately not an assertion that the system is healthy — it is the tool for
 * finding out which of the four devices is letting the side down, and it must be as useful
 * when things are broken as when they are not.
 */
@RunWith(AndroidJUnit4::class)
class SystemHealthTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun reportsEveryLinkInTheChain() = runBlocking {
        val config = VlmConfig.load(context)
        assumeTrue("no vlm.properties on the device", config != null)
        requireNotNull(config)

        val report = SystemHealth.check(config)
        println("=== dragonAssist preflight ===")
        println(report.summary())
        println("healthy: ${report.healthy}")
        report.firstFailure?.let { println("first broken link: ${it.name} — ${it.detail}") }

        assertTrue("the report must cover every hop", report.links.size >= 2)
    }
}
