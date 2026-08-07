package com.example.dragonassist.vlm

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.MediaStore
import androidx.activity.result.contract.ActivityResultContract

/**
 * Video capture with duration and size limits.
 *
 * `ActivityResultContracts.CaptureVideo` offers no way to constrain the recording. That
 * matters here because the board samples six stills and downscales them to 896 px, so
 * everything above ~720p is uploaded and then discarded — a 4K clip costs 62 MB to send
 * and produces model input identical to a 12 MB one.
 *
 * Duration is the meaningful limit: it decides how much of the world the six frames cover
 * (10 s gives one frame per ~1.7 s). Size is only a backstop against someone leaving the
 * camera on 4K — as a primary limit it truncates unpredictably, since the same byte count
 * is 5 s at 4K but 26 s at 720p.
 *
 * Both are *hints*. Camera apps usually honour them but are not required to, so callers
 * must still check the resulting file.
 */
class CaptureVideoLimited(
    private val maxSeconds: Int = DEFAULT_MAX_SECONDS,
    private val maxBytes: Long = DEFAULT_MAX_BYTES,
) : ActivityResultContract<Uri, Boolean>() {

    override fun createIntent(context: Context, input: Uri): Intent =
        Intent(MediaStore.ACTION_VIDEO_CAPTURE)
            .putExtra(MediaStore.EXTRA_OUTPUT, input)
            .putExtra(MediaStore.EXTRA_DURATION_LIMIT, maxSeconds)
            .putExtra(MediaStore.EXTRA_SIZE_LIMIT, maxBytes)
            .addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)

    override fun parseResult(resultCode: Int, intent: Intent?): Boolean =
        resultCode == Activity.RESULT_OK

    companion object {
        const val DEFAULT_MAX_SECONDS = 10

        /**
         * Never fires at 1080p30 (~21 MB for 10 s), which is what this phone is set to.
         * It exists so a 4K recording truncates to ~15 s of upload instead of ~30 s.
         */
        const val DEFAULT_MAX_BYTES = 32L * 1024 * 1024
    }
}
