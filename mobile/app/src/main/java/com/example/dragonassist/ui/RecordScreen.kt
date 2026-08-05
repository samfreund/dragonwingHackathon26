package com.example.dragonassist.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.dragonassist.Phase
import com.example.dragonassist.RecordViewModel

@Composable
fun RecordScreen(
    viewModel: RecordViewModel,
    modifier: Modifier = Modifier,
) {
    val state by viewModel.state.collectAsState()
    val context = LocalContext.current

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        )
    }

    val requestPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasPermission = granted
        if (granted) viewModel.startRecording()
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = "dragonAssist",
            style = MaterialTheme.typography.headlineMedium,
        )
        Text(
            text = state.engine.ifEmpty { "loading engine…" },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Spacer(Modifier.height(32.dp))

        MicButton(
            phase = state.phase,
            level = state.level,
            onClick = {
                when (state.phase) {
                    Phase.Idle ->
                        if (hasPermission) {
                            viewModel.startRecording()
                        } else {
                            requestPermission.launch(Manifest.permission.RECORD_AUDIO)
                        }

                    Phase.Recording -> viewModel.stopAndTranscribe()
                    Phase.Transcribing -> Unit
                }
            },
        )

        Spacer(Modifier.height(16.dp))

        Text(
            text = when (state.phase) {
                Phase.Idle -> if (hasPermission) "Tap to record" else "Tap to allow the microphone"
                Phase.Recording -> "Listening — tap to stop"
                Phase.Transcribing -> "Transcribing…"
            },
            style = MaterialTheme.typography.bodyLarge,
        )

        if (state.phase == Phase.Recording) {
            Spacer(Modifier.height(24.dp))
            LevelMeter(level = state.level)
        }

        Spacer(Modifier.height(32.dp))

        state.error?.let { message ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text(
                        text = "Error",
                        style = MaterialTheme.typography.titleSmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(message, style = MaterialTheme.typography.bodyMedium)
                    Spacer(Modifier.height(8.dp))
                    Button(onClick = viewModel::dismissError) { Text("Dismiss") }
                }
            }
            Spacer(Modifier.height(16.dp))
        }

        if (state.transcript.isNotEmpty()) {
            Card(Modifier.fillMaxWidth()) {
                Column(
                    Modifier
                        .padding(16.dp)
                        .verticalScroll(rememberScrollState()),
                ) {
                    Text(
                        text = "Transcript · ${"%.1f".format(state.lastDurationSeconds)} s",
                        style = MaterialTheme.typography.titleSmall,
                    )
                    Spacer(Modifier.height(8.dp))
                    Text(
                        text = state.transcript,
                        style = MaterialTheme.typography.bodyMedium,
                        fontFamily = FontFamily.Monospace,
                    )
                }
            }
        }
    }
}

@Composable
private fun MicButton(
    phase: Phase,
    level: Float,
    onClick: () -> Unit,
) {
    // The button breathes with input level, so it's obvious the mic is live.
    val pulse by animateFloatAsState(
        targetValue = if (phase == Phase.Recording) 1f + (level * 0.25f) else 1f,
        label = "mic-pulse",
    )

    val container = when (phase) {
        Phase.Idle -> MaterialTheme.colorScheme.primary
        Phase.Recording -> MaterialTheme.colorScheme.error
        Phase.Transcribing -> MaterialTheme.colorScheme.secondary
    }

    Box(
        modifier = Modifier
            .size(140.dp)
            .scale(pulse),
        contentAlignment = Alignment.Center,
    ) {
        Button(
            onClick = onClick,
            enabled = phase != Phase.Transcribing,
            shape = CircleShape,
            colors = ButtonDefaults.buttonColors(containerColor = container),
            modifier = Modifier.fillMaxSize(),
        ) {
            if (phase == Phase.Transcribing) {
                CircularProgressIndicator(modifier = Modifier.size(36.dp))
            } else {
                Text(
                    text = if (phase == Phase.Recording) "STOP" else "RECORD",
                    style = MaterialTheme.typography.titleMedium,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}

@Composable
private fun LevelMeter(level: Float) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        LinearProgressIndicator(
            progress = { level },
            modifier = Modifier
                .fillMaxWidth()
                .height(8.dp),
        )
        Spacer(Modifier.height(8.dp))
        Text(
            text = if (level < 0.01f) "no signal" else "input ${(level * 100).toInt()}%",
            style = MaterialTheme.typography.labelSmall,
            color = if (level < 0.01f) MaterialTheme.colorScheme.error else Color.Unspecified,
        )
    }
}
