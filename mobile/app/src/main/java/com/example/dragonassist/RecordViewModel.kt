package com.example.dragonassist

import android.Manifest
import android.app.Application
import android.net.Uri
import androidx.annotation.RequiresPermission
import androidx.core.content.FileProvider
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.dragonassist.audio.AudioRecorder
import com.example.dragonassist.audio.Recording
import com.example.dragonassist.audio.WavWriter
import com.example.dragonassist.transcribe.StubTranscriber
import com.example.dragonassist.transcribe.Transcriber
import com.example.dragonassist.transcribe.TranscriptionException
import com.example.dragonassist.transcribe.WhisperSession
import com.example.dragonassist.transcribe.WhisperTranscriber
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

enum class Phase { Idle, Recording, Transcribing, Uploading, Answering }

data class RecordUiState(
    val phase: Phase = Phase.Idle,
    val level: Float = 0f,
    /** What the user said, from Whisper. */
    val transcript: String = "",
    /** What the board answered, appended token by token. */
    val answer: String = "",
    val photoPath: String? = null,
    val photoKb: Int = 0,
    val error: String? = null,
    val engine: String = "",
    val vlmModel: String = "",
    val queued: Boolean = false,
    val lastAnswerSeconds: Double = 0.0,
) {
    val hasPhoto: Boolean get() = photoPath != null
}

class RecordViewModel(app: Application) : AndroidViewModel(app) {

    private val recorder = AudioRecorder()

    private var transcriber: Transcriber =
        if (WhisperSession.modelsPresent(app)) WhisperTranscriber(app) else StubTranscriber()

    /** Held open across questions: one connection is one conversation about one photo. */
    private var vlm: VlmClient? = null

    /** The photo currently loaded on the server, so repeat questions skip the upload. */
    private var uploadedPhoto: String? = null

    private var photoBytes: ByteArray? = null

    private val _state = MutableStateFlow(RecordUiState())
    val state: StateFlow<RecordUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { transcriber.prepare() }
            _state.update { it.copy(engine = transcriber.name) }
        }
    }

    // ------------------------------------------------------------------ capture

    /**
     * Creates the destination the system camera app will write into, and returns a
     * content:// URI it is allowed to use.
     */
    fun newPhotoTarget(): Uri {
        val context = getApplication<Application>()
        val dir = File(context.filesDir, "captures").apply { mkdirs() }
        val target = File(dir, "capture.jpg")
        pendingCapture = target
        return FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", target)
    }

    private var pendingCapture: File? = null

    /** Called once the camera app reports success. Scales the photo and resets the Q&A. */
    fun onPhotoCaptured() {
        val file = pendingCapture ?: return
        viewModelScope.launch {
            try {
                val jpeg = withContext(Dispatchers.IO) {
                    ImageScaler.scaleToJpeg(getApplication(), Uri.fromFile(file))
                }
                photoBytes = jpeg
                // A new photo means the server's copy is stale and the conversation is over.
                uploadedPhoto = null
                _state.update {
                    it.copy(
                        photoPath = file.absolutePath,
                        photoKb = jpeg.size / 1024,
                        transcript = "",
                        answer = "",
                        error = null,
                    )
                }
            } catch (e: Exception) {
                // Log the stack too: the message alone rarely says which stage failed.
                android.util.Log.e("RecordViewModel", "photo scaling failed for $file", e)
                _state.update { it.copy(error = "Could not read the photo: ${e.message}") }
            }
        }
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

    /** Stops recording, transcribes on the NPU, then asks the board about the photo. */
    fun stopAndAsk() {
        if (_state.value.phase != Phase.Recording) return

        val recording = recorder.stop()
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

            if (photoBytes == null) {
                _state.update {
                    it.copy(phase = Phase.Idle, error = "Take a photo first, then ask about it.")
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

            // Upload only when the server doesn't already hold this photo.
            val photo = photoBytes ?: return
            if (uploadedPhoto != _state.value.photoPath) {
                _state.update { it.copy(phase = Phase.Uploading) }
                client.uploadImage(photo, "capture.jpg")
                uploadedPhoto = _state.value.photoPath
            }

            _state.update { it.copy(phase = Phase.Answering, answer = "", queued = false) }
            val answer = client.ask(
                prompt = question,
                onQueued = { _state.update { it.copy(queued = true) } },
                onToken = { piece ->
                    _state.update { it.copy(answer = it.answer + piece, queued = false) }
                },
            )
            _state.update {
                it.copy(
                    phase = Phase.Idle,
                    answer = answer.text,
                    lastAnswerSeconds = answer.latencySeconds,
                )
            }
        } catch (e: VlmException) {
            // The socket is likely unusable now; drop it so the next question reconnects.
            closeVlm()
            _state.update { it.copy(phase = Phase.Idle, error = "${e.code}: ${e.message}") }
        } catch (e: Exception) {
            closeVlm()
            _state.update { it.copy(phase = Phase.Idle, error = "Board error: ${e.message}") }
        }
    }

    fun cancelRecording() {
        recorder.cancel()
        _state.update { it.copy(phase = Phase.Idle, level = 0f) }
    }

    fun dismissError() = _state.update { it.copy(error = null) }

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
        uploadedPhoto = null
    }

    override fun onCleared() {
        recorder.cancel()
        transcriber.close()
        closeVlm()
        super.onCleared()
    }

    companion object {
        const val LAST_RECORDING = "last_recording.wav"
    }
}
