package com.example.dragonassist

import android.Manifest
import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.annotation.RequiresPermission
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.dragonassist.audio.AudioRecorder
import com.example.dragonassist.audio.Recording
import com.example.dragonassist.audio.WavWriter
import com.example.dragonassist.speak.AndroidSpeaker
import com.example.dragonassist.speak.SentenceBuffer
import com.example.dragonassist.speak.Speaker
import com.example.dragonassist.transcribe.StubTranscriber
import com.example.dragonassist.transcribe.Transcriber
import com.example.dragonassist.transcribe.TranscriptionException
import com.example.dragonassist.transcribe.WhisperSession
import com.example.dragonassist.transcribe.WhisperTranscriber
import com.example.dragonassist.vlm.CaptureVideoLimited
import com.example.dragonassist.vlm.CapturedMedia
import com.example.dragonassist.vlm.ImageScaler
import com.example.dragonassist.vlm.VlmClient
import com.example.dragonassist.vlm.VlmConfig
import com.example.dragonassist.vlm.VlmException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

enum class Phase { Idle, Recording, Transcribing, Uploading, Answering }

enum class MediaKind { Photo, Video }

data class RecordUiState(
    val phase: Phase = Phase.Idle,
    val level: Float = 0f,
    val transcript: String = "",
    val answer: String = "",
    val mediaKind: MediaKind? = null,
    /** Path of the image to show — the photo itself, or a frame from the video. */
    val previewPath: String? = null,
    val mediaLabel: String = "",
    /** 0..1 while a video is streaming up; ignored for photos. */
    val uploadProgress: Float = 0f,
    val warning: String? = null,
    val error: String? = null,
    val engine: String = "",
    val vlmModel: String = "",
    val queued: Boolean = false,
    val lastAnswerSeconds: Double = 0.0,
    val speechEnabled: Boolean = true,
    val speaker: String = "",
) {
    val hasMedia: Boolean get() = mediaKind != null
}

class RecordViewModel(app: Application) : AndroidViewModel(app) {

    private val recorder = AudioRecorder()

    private var transcriber: Transcriber =
        if (WhisperSession.modelsPresent(app)) WhisperTranscriber(app) else StubTranscriber()

    /** Held open across questions: one connection is one conversation about one media. */
    private var vlm: VlmClient? = null

    private var media: CapturedMedia? = null

    /** What the server currently holds, so repeat questions skip the upload entirely. */
    private var uploadedMedia: CapturedMedia? = null

    private var pendingCapture: File? = null

    private val speaker: Speaker = AndroidSpeaker(app)

    /** Releases the streamed answer a sentence at a time, so speech starts early. */
    private val sentences = SentenceBuffer()

    private val _state = MutableStateFlow(RecordUiState())
    val state: StateFlow<RecordUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { transcriber.prepare() }
            _state.update { it.copy(engine = transcriber.name) }
            speaker.prepare()
            _state.update { it.copy(speaker = speaker.name) }
        }
    }

    // ------------------------------------------------------------------ capture

    fun newPhotoTarget(): Uri = newCaptureTarget("capture.jpg")

    fun newVideoTarget(): Uri = newCaptureTarget("capture.mp4")

    private fun newCaptureTarget(fileName: String): Uri {
        val context = getApplication<Application>()
        val dir = File(context.filesDir, "captures").apply { mkdirs() }
        // Remove any previous file: the camera app appends to an existing target on some
        // devices, which would produce a corrupt clip.
        val target = File(dir, fileName).also { it.delete() }
        pendingCapture = target
        return FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", target)
    }

    fun onPhotoCaptured() {
        val file = pendingCapture ?: return
        viewModelScope.launch {
            try {
                val jpeg = withContext(Dispatchers.IO) {
                    ImageScaler.scaleToJpeg(getApplication(), Uri.fromFile(file))
                }
                adopt(
                    CapturedMedia.Photo(file, jpeg),
                    kind = MediaKind.Photo,
                    previewPath = file.absolutePath,
                    label = "Photo · ${jpeg.size / 1024} KB",
                    warning = null,
                )
            } catch (e: Exception) {
                Log.e(TAG, "photo scaling failed for $file", e)
                _state.update { it.copy(error = "Could not read the photo: ${e.message}") }
            }
        }
    }

    fun onVideoCaptured() {
        val file = pendingCapture ?: return
        viewModelScope.launch {
            try {
                val prepared = withContext(Dispatchers.IO) {
                    val duration = CapturedMedia.videoDurationMs(file)
                    val thumb = CapturedMedia.videoThumbnail(file)?.let { saveThumbnail(it) }
                    Triple(CapturedMedia.Video(file, duration), thumb, duration)
                }
                val (video, thumbPath, durationMs) = prepared
                val megabytes = video.sizeBytes / (1024.0 * 1024.0)

                adopt(
                    video,
                    kind = MediaKind.Video,
                    previewPath = thumbPath,
                    label = "Video · %.1fs · %.1f MB".format(durationMs / 1000.0, megabytes),
                    // At ~2 MB/s over the relayed tailnet, anything past this is a long
                    // wait — almost always because the camera is on 4K rather than FHD.
                    warning = if (video.sizeBytes > SLOW_UPLOAD_BYTES) {
                        "%.0f MB will take roughly %.0f s to upload. Set the camera to FHD."
                            .format(megabytes, megabytes / 2.1)
                    } else null,
                )
            } catch (e: Exception) {
                Log.e(TAG, "video preparation failed for $file", e)
                _state.update { it.copy(error = "Could not read the video: ${e.message}") }
            }
        }
    }

    private fun adopt(
        captured: CapturedMedia,
        kind: MediaKind,
        previewPath: String?,
        label: String,
        warning: String?,
    ) {
        media = captured
        // New media means the server's copy is stale and the conversation starts over.
        uploadedMedia = null
        _state.update {
            it.copy(
                mediaKind = kind,
                previewPath = previewPath,
                mediaLabel = label,
                transcript = "",
                answer = "",
                uploadProgress = 0f,
                warning = warning,
                error = null,
            )
        }
    }

    private fun saveThumbnail(bitmap: Bitmap): String {
        val target = File(getApplication<Application>().filesDir, "captures/thumb.jpg")
        FileOutputStream(target).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 85, it) }
        bitmap.recycle()
        return target.absolutePath
    }

    // ------------------------------------------------------------------ recording

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun startRecording() {
        if (_state.value.phase != Phase.Idle) return
        try {
            recorder.start { level -> _state.update { it.copy(level = level) } }
            _state.update {
                it.copy(phase = Phase.Recording, error = null, answer = "", level = 0f)
            }
        } catch (e: Exception) {
            _state.update {
                it.copy(phase = Phase.Idle, error = e.message ?: "Could not start recording")
            }
        }
    }

    fun stopAndAsk() {
        if (_state.value.phase != Phase.Recording) return

        val recording = recorder.stop()
        speaker.stop()
        _state.update { it.copy(phase = Phase.Transcribing, level = 0f) }

        viewModelScope.launch {
            withContext(Dispatchers.IO) { saveForDebugging(recording) }

            val question = try {
                withContext(Dispatchers.Default) { transcriber.transcribe(recording) }
            } catch (e: TranscriptionException) {
                _state.update { it.copy(phase = Phase.Idle, error = e.message) }
                return@launch
            } catch (e: Exception) {
                _state.update {
                    it.copy(phase = Phase.Idle, error = "Transcription failed: ${e.message}")
                }
                return@launch
            }

            _state.update { it.copy(transcript = question) }

            if (media == null) {
                _state.update {
                    it.copy(phase = Phase.Idle, error = "Capture a photo or video first.")
                }
                return@launch
            }
            ask(question)
        }
    }

    // ------------------------------------------------------------------ the board

    private suspend fun ask(question: String) {
        val config = VlmConfig.load(getApplication())
        if (config == null) {
            _state.update {
                it.copy(
                    phase = Phase.Idle,
                    error = "No board configured. Push vlm.properties to " +
                        VlmConfig.file(getApplication<Application>()).absolutePath,
                )
            }
            return
        }

        try {
            val client = vlm ?: VlmClient(config).also { fresh ->
                val ready = fresh.connect()
                vlm = fresh
                _state.update { it.copy(vlmModel = ready.optString("model")) }
            }

            val current = media ?: return
            if (uploadedMedia !== current) {
                _state.update { it.copy(phase = Phase.Uploading, uploadProgress = 0f) }
                when (current) {
                    is CapturedMedia.Photo ->
                        client.uploadImage(current.jpeg, "capture.jpg")

                    is CapturedMedia.Video ->
                        client.uploadFile(
                            file = current.file,
                            name = "capture.mp4",
                            frames = VIDEO_FRAMES,
                            strategy = VIDEO_STRATEGY,
                        ) { sent, total ->
                            _state.update {
                                it.copy(uploadProgress = sent.toFloat() / total.coerceAtLeast(1))
                            }
                        }
                }
                uploadedMedia = current
            }

            _state.update { it.copy(phase = Phase.Answering, answer = "", queued = false) }
            speaker.stop()
            sentences.clear()

            val answer = client.ask(
                prompt = question,
                onQueued = { _state.update { it.copy(queued = true) } },
                onToken = { piece ->
                    _state.update { it.copy(answer = it.answer + piece, queued = false) }
                    if (_state.value.speechEnabled) {
                        sentences.append(piece).forEach(speaker::say)
                    }
                },
            )
            // The final sentence usually has no trailing space, so it never triggers a
            // boundary while streaming and has to be flushed explicitly.
            if (_state.value.speechEnabled) sentences.flush()?.let(speaker::say)
            _state.update {
                it.copy(
                    phase = Phase.Idle,
                    answer = answer.text,
                    lastAnswerSeconds = answer.latencySeconds,
                )
            }
        } catch (e: VlmException) {
            closeVlm()
            _state.update {
                it.copy(phase = Phase.Idle, error = describe(e))
            }
        } catch (e: Exception) {
            closeVlm()
            Log.e(TAG, "board request failed", e)
            _state.update { it.copy(phase = Phase.Idle, error = "Board error: ${e.message}") }
        }
    }

    /** Connection failures are almost always Tailscale, not the board — say so. */
    private fun describe(e: VlmException): String = when (e.code) {
        "connect" -> "Can't reach the board. Check Tailscale is connected on this phone " +
            "and that iq9 appears in its device list. (${e.message})"
        "auth" -> "The board rejected the token in vlm.properties."
        else -> "${e.code}: ${e.message}"
    }

    fun cancelRecording() {
        recorder.cancel()
        _state.update { it.copy(phase = Phase.Idle, level = 0f) }
    }

    fun setSpeechEnabled(enabled: Boolean) {
        if (!enabled) {
            speaker.stop()
            sentences.clear()
        }
        _state.update { it.copy(speechEnabled = enabled) }
    }

    fun dismissError() = _state.update { it.copy(error = null, warning = null) }

    private fun saveForDebugging(recording: Recording): String? = try {
        if (recording.isEmpty) null else {
            val target = File(getApplication<Application>().filesDir, LAST_RECORDING)
            WavWriter.write(recording, target).absolutePath
        }
    } catch (_: Exception) {
        null
    }

    private fun closeVlm() {
        runCatching { vlm?.close() }
        vlm = null
        uploadedMedia = null
    }

    override fun onCleared() {
        recorder.cancel()
        transcriber.close()
        speaker.close()
        closeVlm()
        super.onCleared()
    }

    companion object {
        private const val TAG = "RecordViewModel"
        const val LAST_RECORDING = "last_recording.wav"

        /** Server default; 6 frames over a 10 s clip is one every ~1.7 s. */
        const val VIDEO_FRAMES = 6

        /**
         * Tile the frames into a single contact sheet rather than sending them as
         * separate images. Measured on this board with a 4.3 s clip:
         *
         *   6 frames, "sheet"   ->    3.9 s, coherent answer
         *   3 frames, "frames"  ->  172.2 s, degenerated into repetition then "000000…"
         *   6 frames, "frames"  ->  183.8 s, same failure
         *
         * So `"frames"` is not merely slower — multiple images in one prompt push
         * Qwen3-VL-4B past something it does not recover from, and the answer is
         * unusable. `"sheet"` is ~47x faster *and* correct.
         */
        const val VIDEO_STRATEGY = "sheet"

        /** Roughly 15 s of upload at the ~2.1 MB/s measured over the relayed tailnet. */
        const val SLOW_UPLOAD_BYTES = 32L * 1024 * 1024
    }
}
