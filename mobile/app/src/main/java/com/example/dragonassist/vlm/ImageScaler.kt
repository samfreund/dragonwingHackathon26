package com.example.dragonassist.vlm

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.net.Uri
import android.util.Log
import java.io.ByteArrayOutputStream

/**
 * Shrinks a captured photo before it goes over the wire.
 *
 * The board downscales everything to 896 px on the longest edge anyway
 * (`VLMQA_MAX_IMAGE_EDGE`), so sending a full-resolution S25 Ultra frame means uploading
 * ~12 MB for the server to throw away. Scaling here produces byte-identical model input
 * from roughly 150 KB.
 *
 * Slightly larger than 896 by default: the server re-encodes, and giving it a little
 * headroom avoids compounding two lossy resizes.
 */
object ImageScaler {

    const val DEFAULT_MAX_EDGE = 1280
    const val DEFAULT_QUALITY = 85

    private const val TAG = "ImageScaler"

    /**
     * Decodes [uri], corrects its orientation, scales it down and JPEG-encodes it.
     *
     * Decoding is two-pass: the first only reads the header, so a large image is never
     * fully materialised at full resolution just to be shrunk.
     */
    fun scaleToJpeg(
        context: Context,
        uri: Uri,
        maxEdge: Int = DEFAULT_MAX_EDGE,
        quality: Int = DEFAULT_QUALITY,
    ): ByteArray {
        // Note: with inJustDecodeBounds, decodeStream returns null *on success* — it only
        // populates the options. So the stream's nullability must be checked separately
        // from the decode result, or a perfectly good file looks unopenable.
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        val header = context.contentResolver.openInputStream(uri)
            ?: error("Could not open $uri")
        header.use { BitmapFactory.decodeStream(it, null, bounds) }

        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "Not a decodable image: $uri" }

        val options = BitmapFactory.Options().apply {
            inSampleSize = sampleSizeFor(bounds.outWidth, bounds.outHeight, maxEdge)
        }
        val body = context.contentResolver.openInputStream(uri)
            ?: error("Could not open $uri")
        val decoded = body.use { BitmapFactory.decodeStream(it, null, options) }
            ?: error("Could not decode $uri (${bounds.outWidth}x${bounds.outHeight})")

        val rotated = applyExifRotation(context, uri, decoded)
        val scaled = scaleToFit(rotated, maxEdge)

        val jpeg = ByteArrayOutputStream().also {
            scaled.compress(Bitmap.CompressFormat.JPEG, quality, it)
        }.toByteArray()

        Log.i(
            TAG,
            "scaled ${bounds.outWidth}x${bounds.outHeight} -> ${scaled.width}x${scaled.height}, " +
                "${jpeg.size / 1024} KB",
        )

        if (scaled !== rotated) scaled.recycle()
        if (rotated !== decoded) rotated.recycle()
        decoded.recycle()
        return jpeg
    }

    /** Largest power-of-two subsample that still leaves the image above [maxEdge]. */
    private fun sampleSizeFor(width: Int, height: Int, maxEdge: Int): Int {
        var sample = 1
        var longest = maxOf(width, height)
        while (longest / 2 >= maxEdge) {
            longest /= 2
            sample *= 2
        }
        return sample
    }

    /**
     * Phone cameras usually record orientation in EXIF rather than rotating the pixels.
     * Re-encoding drops that tag, so without this a portrait photo arrives sideways and
     * the model describes a rotated scene.
     */
    private fun applyExifRotation(context: Context, uri: Uri, bitmap: Bitmap): Bitmap {
        val orientation = runCatching {
            context.contentResolver.openInputStream(uri)?.use {
                ExifInterface(it).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,
                    ExifInterface.ORIENTATION_NORMAL,
                )
            } ?: ExifInterface.ORIENTATION_NORMAL
        }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)

        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            else -> return bitmap
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun scaleToFit(bitmap: Bitmap, maxEdge: Int): Bitmap {
        val longest = maxOf(bitmap.width, bitmap.height)
        if (longest <= maxEdge) return bitmap
        val ratio = maxEdge.toFloat() / longest
        return Bitmap.createScaledBitmap(
            bitmap,
            (bitmap.width * ratio).toInt().coerceAtLeast(1),
            (bitmap.height * ratio).toInt().coerceAtLeast(1),
            true,
        )
    }
}
