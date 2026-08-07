package com.example.dragonassist

import android.graphics.BitmapFactory
import android.net.Uri
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.dragonassist.vlm.ImageScaler
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Runs the scaler against a photo actually taken by the camera on this device.
 *
 * A synthetic bitmap would not have caught the bug this test was written for: with
 * `inJustDecodeBounds`, `BitmapFactory.decodeStream` returns null on success, which made
 * a valid 4000x3000 Samsung JPEG report "could not open". Real camera output also carries
 * EXIF rotation, which synthetic images do not.
 */
@RunWith(AndroidJUnit4::class)
class ImageScalerTest {

    private val context get() = InstrumentationRegistry.getInstrumentation().targetContext

    private fun lastCapture(): File = File(context.filesDir, "captures/capture.jpg")

    @Test
    fun scalesARealCameraPhoto() {
        val source = lastCapture()
        assumeTrue("no capture.jpg — take a photo in the app first", source.isFile)

        val originalBounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(source.absolutePath, originalBounds)
        println("original: ${originalBounds.outWidth}x${originalBounds.outHeight}, " +
            "${source.length() / 1024} KB")

        val jpeg = ImageScaler.scaleToJpeg(context, Uri.fromFile(source))

        val scaledBounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, scaledBounds)
        println("scaled:   ${scaledBounds.outWidth}x${scaledBounds.outHeight}, " +
            "${jpeg.size / 1024} KB")

        assertTrue("output is not decodable", scaledBounds.outWidth > 0)
        assertEquals(
            "longest edge should be the requested max",
            ImageScaler.DEFAULT_MAX_EDGE,
            maxOf(scaledBounds.outWidth, scaledBounds.outHeight),
        )
        assertTrue(
            "scaled file (${jpeg.size / 1024} KB) should be far smaller than the " +
                "original (${source.length() / 1024} KB)",
            jpeg.size < source.length() / 4,
        )
        assertTrue(
            "must stay under the inline upload cap",
            jpeg.size < 8 * 1024 * 1024,
        )
    }

    @Test
    fun appliesExifRotation() {
        val source = lastCapture()
        assumeTrue("no capture.jpg", source.isFile)

        val raw = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(source.absolutePath, raw)
        val exif = android.media.ExifInterface(source.absolutePath)
            .getAttributeInt(android.media.ExifInterface.TAG_ORIENTATION, 1)
        println("exif orientation: $exif, raw ${raw.outWidth}x${raw.outHeight}")

        val jpeg = ImageScaler.scaleToJpeg(context, Uri.fromFile(source))
        val out = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, out)
        println("after rotation: ${out.outWidth}x${out.outHeight}")

        // Orientations 5-8 are the quarter turns; those swap the aspect ratio. Anything
        // else must preserve it. Getting this wrong sends a sideways scene to the model.
        val quarterTurn = exif in 5..8
        val rawLandscape = raw.outWidth > raw.outHeight
        val outLandscape = out.outWidth > out.outHeight
        assertEquals(
            "aspect orientation wrong for EXIF $exif",
            if (quarterTurn) !rawLandscape else rawLandscape,
            outLandscape,
        )
    }
}
