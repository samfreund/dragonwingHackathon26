package com.example.dragonassist.vlm

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import java.io.File

/**
 * What the user captured, and how it should reach the board.
 *
 * The two kinds take different upload paths for a concrete reason: a downscaled photo is
 * ~150 KB and fits in one inline message, while a 10 s clip is ~21 MB — past the ~11.9 MB
 * inline ceiling — and needs the chunked binary path with a progress bar.
 */
sealed interface CapturedMedia {

    val file: File
    val sizeBytes: Long

    /** Downscaled JPEG bytes, ready to send inline. */
    data class Photo(
        override val file: File,
        val jpeg: ByteArray,
    ) : CapturedMedia {
        override val sizeBytes: Long get() = jpeg.size.toLong()

        // ByteArray breaks data-class equality; identity is what callers actually mean.
        override fun equals(other: Any?) = this === other
        override fun hashCode() = System.identityHashCode(this)
    }

    /** Left on disk and streamed in chunks — never read into memory whole. */
    data class Video(
        override val file: File,
        val durationMs: Long,
        override val sizeBytes: Long = file.length(),
    ) : CapturedMedia

    companion object {
        /**
         * Reads duration from the recorded file.
         *
         * Returns 0 rather than throwing: a missing duration is worth reporting in the UI
         * but is no reason to refuse an otherwise valid upload, since the board re-probes
         * the file with ffprobe anyway.
         */
        fun videoDurationMs(file: File): Long = withRetriever(file) {
            it.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull()
        } ?: 0L

        /** A representative frame for the UI, taken from the middle of the clip. */
        fun videoThumbnail(file: File): Bitmap? = withRetriever(file) {
            val middleUs = (videoDurationMs(file) * 1000) / 2
            it.getFrameAtTime(middleUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
        }

        /**
         * MediaMetadataRetriever only became AutoCloseable in API 29, so `use` would be a
         * lint violation against this module's minSdk. Release explicitly instead.
         */
        private fun <T> withRetriever(file: File, block: (MediaMetadataRetriever) -> T?): T? {
            val retriever = MediaMetadataRetriever()
            return try {
                retriever.setDataSource(file.absolutePath)
                block(retriever)
            } catch (_: Exception) {
                null
            } finally {
                runCatching { retriever.release() }
            }
        }
    }
}
