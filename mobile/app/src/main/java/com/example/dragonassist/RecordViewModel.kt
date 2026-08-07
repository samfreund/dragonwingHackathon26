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
import com.example.dragonassist.ui.theme.ThemeMode
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
import com.example.dragonassist.net.ContextStreamClient
import com.example.dragonassist.net.SystemHealth
import com.example.dragonassist.net.TextQaClient
import com.example.dragonassist.net.TransportException
import com.example.dragonassist.vlm.VlmException
import java.util.UUID
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

enum class Phase { Idle, Recording, Transcribing, Uploading, Narrating, Answering }

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
    val themeMode: ThemeMode = ThemeMode.System,
    /** Which backend produced [answer]; shown so the source is never guessed at. */
    val answerRoute: String? = null,
    val health: String = "",
    val healthy: Boolean = false,
    /** True once the board has the media; ASK stays disabled until then. */
    val mediaSent: Boolean = false,
) {
    val hasMedia: Boolean get() = mediaKind != null
}

class RecordViewModel(app: Application) : AndroidViewModel(app) {

    private val recorder = AudioRecorder()

    private var transcriber: Transcriber =
        if (WhisperSession.modelsPresent(app)) WhisperTranscriber(app) else StubTranscriber()

    /**
     * Used to deliver media to the board and nothing else — no prompt, no question.
     * The board has its own system prompt, and narration reaching the laptop is handled
     * further down the stack.
     *
     * Nothing the board says is ever shown, spoken or stored, so a board-only setup
     * cannot masquerade as a working full-stack demo.
     */
    private var vlm: VlmClient? = null

    private var textQa: TextQaClient? = null

    /** One consultation, used as the transport's `video_id`. */
    private val sessionId: String = "session-${UUID.randomUUID().toString().take(8)}"

    private var media: CapturedMedia? = null

    /** What the server currently holds, so repeat questions skip the upload entirely. */
    private var uploadedMedia: CapturedMedia? = null

    private var pendingCapture: File? = null

    /** Guards against two uploads racing after a quick retake. */
    @Volatile private var uploading = false

    private val speaker: Speaker = AndroidSpeaker(app)

    /** Releases the streamed answer a sentence at a time, so speech starts early. */
    private val sentences = SentenceBuffer()

    /** Survives process death, so the choice sticks between demo runs. */
    private val prefs = app.getSharedPreferences("dragonassist", Application.MODE_PRIVATE)

    private val _state = MutableStateFlow(
        RecordUiState(
            themeMode = runCatching {
                ThemeMode.valueOf(prefs.getString(KEY_THEME, null) ?: ThemeMode.System.name)
            }.getOrDefault(ThemeMode.System),
        )
    )
    val state: StateFlow<RecordUiState> = _state.asStateFlow()

    init {
        // The single string the phone and the board must agree on: it names the upload
        // and the context file the laptop answers from.
        Log.i(TAG, "session id: $sessionId")
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
                val (jpeg, previewFile) = withContext(Dispatchers.IO) {
                    val scaled = ImageScaler.scaleToJpeg(getApplication(), Uri.fromFile(file))
                    // Preview the *scaled* bytes, not the original: BitmapFactory ignores
                    // EXIF orientation, so previewing capture.jpg shows a sideways photo
                    // while an upright one is sent. Same pixels in both places.
                    val preview = File(getApplication<Application>().filesDir, "captures/preview.jpg")
                    preview.writeBytes(scaled)
                    scaled to preview
                }
                adopt(
                    CapturedMedia.Photo(file, jpeg),
                    kind = MediaKind.Photo,
                    previewPath = previewFile.absolutePath,
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
                answerRoute = null,
                uploadProgress = 0f,
                warning = warning,
                error = null,
            )
        }
        // Send it immediately. The user is about to spend several seconds thinking of a
        // question and speaking it; the board can be looking at the image throughout,
        // instead of the whole pipeline starting only once they stop talking. It also
        // takes a 21 MB video upload off the critical path.
        uploadMedia(captured)
    }

    /** Delivers media to the board in the background, guarding against a double send. */
    private fun uploadMedia(captured: CapturedMedia) {
        if (uploading) return
        val config = VlmConfig.load(getApplication()) ?: return
        uploading = true
        viewModelScope.launch {
            _state.update { it.copy(mediaSent = false, uploadProgress = 0f, error = null) }
            try {
                val client = vlm ?: VlmClient(config).also { fresh ->
                    val ready = fresh.connect()
                    vlm = fresh
                    _state.update { it.copy(vlmModel = ready.optString("model")) }
                }
                when (captured) {
                    is CapturedMedia.Photo ->
                        client.uploadImage(captured.jpeg, "$sessionId.jpg")

                    is CapturedMedia.Video ->
                        client.uploadFile(
                            file = captured.file,
                            name = "$sessionId.mp4",
                            frames = VIDEO_FRAMES,
                            strategy = VIDEO_STRATEGY,
                        ) { sent, total ->
                            _state.update {
                                it.copy(uploadProgress = sent.toFloat() / total.coerceAtLeast(1))
                            }
                        }
                }
                _state.update { it.copy(uploadProgress = 1f, phase = Phase.Narrating) }

                // Ask the board to describe what it received.
                //
                // Runs here rather than at question time so it overlaps with the user
                // thinking of and speaking their question, instead of adding to the wait.
                val description = client.ask(NARRATION_PROMPT).text
                Log.i(TAG, "description (${description.length} chars): ${description.take(120)}")

                // Relay it to the laptop ourselves, under the same session id we will
                // later ask about. The board publishes under a fixed `phone-live`, which
                // never matches the id in our query — routing it through the phone keeps
                // the write and the read addressed to the same context file.
                ContextStreamClient(config.contextUrl, config.streamToken).use { stream ->
                    stream.open(sessionId)
                    stream.append(description.trim() + "\n")
                    stream.end()
                }
                Log.i(TAG, "context written for $sessionId")

                uploadedMedia = captured
                _state.update { it.copy(mediaSent = true, phase = Phase.Idle) }
            } catch (e: Exception) {
                Log.e(TAG, "media upload or narration failed", e)
                _state.update {
                    it.copy(
                        phase = Phase.Idle,
                        mediaSent = false,
                        error = if (e is VlmException) describeVlm(e)
                        else "Could not send the media: ${e.message}",
                    )
                }
            } finally {
                uploading = false
            }
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

    /**
     * Answers a question by going through the whole stack.
     *
     * The board is used to *look*, never to reply: its narration is forwarded to the
     * laptop by the bridge, and the laptop answers. Nothing the board says is displayed,
     * so there is no path by which a board-only setup can look like a working demo.
     *
     * Fails loudly and names the broken link rather than degrading to a direct answer.
     */
    private suspend fun ask(question: String) {
        val config = VlmConfig.load(getApplication())
        if (config == null || !config.hasLaptop) {
            _state.update {
                it.copy(
                    phase = Phase.Idle,
                    error = "The laptop is not configured. Set laptop_host in " +
                        VlmConfig.file(getApplication<Application>()).absolutePath,
                )
            }
            return
        }

        try {
            // Media is sent at capture time, not here. If it is still in flight, wait
            // rather than asking about something the board has not received.
            while (uploading) delay(100)
            if (media != null && uploadedMedia == null) {
                _state.update {
                    it.copy(phase = Phase.Idle, error = "The media didn't reach the board.")
                }
                return
            }

            // 2. Ask the laptop. This is the only source of anything the user sees.
            _state.update {
                it.copy(phase = Phase.Answering, answer = "", answerRoute = null, queued = false)
            }
            speaker.stop()
            sentences.clear()

            val qa = textQa ?: TextQaClient(
                config.textQaUrl, config.phoneToken, "galaxy-s25-ultra",
            ).also { it.connect(); textQa = it }

            val answer = qa.ask(
                sessionId = sessionId,
                question = question,
                requestId = "q-${UUID.randomUUID()}",
                onQueued = { _state.update { it.copy(queued = true) } },
            )

            _state.update {
                it.copy(
                    phase = Phase.Idle,
                    answer = answer.answer,
                    answerRoute = answer.route,
                )
            }
            if (_state.value.speechEnabled) {
                sentences.append(answer.answer).forEach(speaker::say)
                sentences.flush()?.let(speaker::say)
            }
        } catch (e: TransportException) {
            closeAll()
            _state.update { it.copy(phase = Phase.Idle, error = describeTransport(e)) }
        } catch (e: VlmException) {
            closeAll()
            _state.update { it.copy(phase = Phase.Idle, error = describeVlm(e)) }
        } catch (e: Exception) {
            closeAll()
            Log.e(TAG, "request failed", e)
            _state.update { it.copy(phase = Phase.Idle, error = "Failed: ${e.message}") }
        }
    }

    /** Runs the preflight and publishes a per-link report. */
    fun runHealthCheck() {
        viewModelScope.launch {
            _state.update { it.copy(health = "checking…", healthy = false) }
            val config = VlmConfig.load(getApplication())
            if (config == null) {
                _state.update { it.copy(health = "no vlm.properties on the device") }
                return@launch
            }
            val report = withContext(Dispatchers.IO) { SystemHealth.check(config) }
            _state.update { it.copy(health = report.summary(), healthy = report.healthy) }
        }
    }

    private fun describeTransport(e: TransportException): String = when (e.code) {
        "connect", "closed" ->
            "Can't reach the laptop. Check Tailscale, and that stream_transport is " +
                "running on qcworkshop3. (${e.message})"
        "auth" -> "The laptop rejected the token in vlm.properties."
        "sequence" -> "Context stream out of sync: ${e.message}"
        else -> "Laptop: ${e.message}"
    }

    private fun describeVlm(e: VlmException): String = when (e.code) {
        "connect" -> "Can't reach the board. Check Tailscale and that iq9 appears in it."
        "auth" -> "The board rejected the token in vlm.properties."
        else -> "Board: ${e.code} — ${e.message}"
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

    /** Cycles Auto → Light → Dark, which needs no menu and no extra screen. */
    fun cycleTheme() {
        val next = _state.value.themeMode.next()
        prefs.edit().putString(KEY_THEME, next.name).apply()
        _state.update { it.copy(themeMode = next) }
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

    private fun closeAll() {
        runCatching { vlm?.close() }
        runCatching { textQa?.close() }
        vlm = null
        textQa = null
        uploadedMedia = null
    }

    override fun onCleared() {
        recorder.cancel()
        transcriber.close()
        speaker.close()
        closeAll()
        super.onCleared()
    }

    companion object {
        private const val TAG = "RecordViewModel"
        const val LAST_RECORDING = "last_recording.wav"
        private const val KEY_THEME = "theme_mode"

        /**
         * Sent with the media so the board produces a description for the laptop to
         * answer from. The board is never asked the user's question — only this.
         *
         * Deliberately terse. An earlier "describe this in detail, list everything"
         * prompt generated a 512-token answer and took **161 s** on this board; a bounded
         * request comes back in seconds. Short labelled facts are also better material
         * for the extractive reader downstream, which answers by copying a span out of
         * this text rather than reasoning over it.
         */
        const val NARRATION_PROMPT =
            "Describe what is in this media. Include any visible text, and name objects " +
                "and people. At most six short lines. No preamble."

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
