package com.example.dragonassist

import android.Manifest
import android.app.Application
import androidx.annotation.RequiresPermission
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.dragonassist.audio.AudioRecorder
import com.example.dragonassist.audio.Recording
import com.example.dragonassist.audio.WavWriter
import com.example.dragonassist.transcribe.StubTranscriber
import com.example.dragonassist.transcribe.WhisperSession
import com.example.dragonassist.transcribe.WhisperTranscriber
import com.example.dragonassist.transcribe.Transcriber
import com.example.dragonassist.transcribe.TranscriptionException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

enum class Phase { Idle, Recording, Transcribing }

data class RecordUiState(
    val phase: Phase = Phase.Idle,
    val level: Float = 0f,
    val transcript: String = "",
    val error: String? = null,
    val engine: String = "",
    val lastDurationSeconds: Float = 0f,
    val savedWavPath: String? = null,
)

class RecordViewModel(app: Application) : AndroidViewModel(app) {

    private val recorder = AudioRecorder()

    /**
     * Whisper when its models are on the device, the stub otherwise.
     *
     * The 192 MB of graphs are pushed with adb rather than bundled in the APK, so a fresh
     * install has no models until that happens. Falling back keeps the app usable and
     * makes the reason visible in the UI instead of crashing at startup.
     */
    private var transcriber: Transcriber =
        if (WhisperSession.modelsPresent(app)) WhisperTranscriber(app) else StubTranscriber()

    private val _state = MutableStateFlow(RecordUiState())
    val state: StateFlow<RecordUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { transcriber.prepare() }
            _state.update { it.copy(engine = transcriber.name) }
        }
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun startRecording() {
        if (_state.value.phase != Phase.Idle) return

        try {
            recorder.start { level ->
                _state.update { it.copy(level = level) }
            }
            _state.update {
                it.copy(phase = Phase.Recording, error = null, transcript = "", level = 0f)
            }
        } catch (e: Exception) {
            _state.update {
                it.copy(phase = Phase.Idle, error = e.message ?: "Could not start recording")
            }
        }
    }

    fun stopAndTranscribe() {
        if (_state.value.phase != Phase.Recording) return

        val recording = recorder.stop()
        _state.update {
            it.copy(
                phase = Phase.Transcribing,
                level = 0f,
                lastDurationSeconds = recording.durationSeconds,
            )
        }

        viewModelScope.launch {
            val wavPath = withContext(Dispatchers.IO) { saveForDebugging(recording) }
            try {
                val text = withContext(Dispatchers.Default) { transcriber.transcribe(recording) }
                _state.update {
                    it.copy(phase = Phase.Idle, transcript = text, savedWavPath = wavPath)
                }
            } catch (e: TranscriptionException) {
                _state.update {
                    it.copy(phase = Phase.Idle, error = e.message, savedWavPath = wavPath)
                }
            } catch (e: Exception) {
                _state.update {
                    it.copy(
                        phase = Phase.Idle,
                        error = "Transcription failed: ${e.message}",
                        savedWavPath = wavPath,
                    )
                }
            }
        }
    }

    fun cancelRecording() {
        recorder.cancel()
        _state.update { it.copy(phase = Phase.Idle, level = 0f) }
    }

    fun dismissError() = _state.update { it.copy(error = null) }

    /** Keeps the most recent take on disk so it can be compared against desktop Whisper. */
    private fun saveForDebugging(recording: Recording): String? = try {
        if (recording.isEmpty) {
            null
        } else {
            val target = File(getApplication<Application>().filesDir, LAST_RECORDING)
            WavWriter.write(recording, target).absolutePath
        }
    } catch (_: Exception) {
        null // Debug convenience only — never fail a transcription over it.
    }

    override fun onCleared() {
        recorder.cancel()
        transcriber.close()
        super.onCleared()
    }

    companion object {
        const val LAST_RECORDING = "last_recording.wav"
    }
}
