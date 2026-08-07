package com.example.dragonassist.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.example.dragonassist.MediaKind
import com.example.dragonassist.Phase
import com.example.dragonassist.RecordViewModel

@Composable
fun RecordScreen(
    viewModel: RecordViewModel,
    modifier: Modifier = Modifier,
) {
    val state by viewModel.state.collectAsState()
    val context = LocalContext.current

    var hasMicPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED
        )
    }

    val requestMic = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasMicPermission = granted
        if (granted) viewModel.startRecording()
    }

    // The system camera app writes into our sandbox through the FileProvider, so no
    // CAMERA permission is needed here — the camera app holds it.
    val takePhoto = rememberLauncherForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { saved -> if (saved) viewModel.onPhotoCaptured() }

    val recordVideo = rememberLauncherForActivityResult(
        com.example.dragonassist.vlm.CaptureVideoLimited()
    ) { saved -> if (saved) viewModel.onVideoCaptured() }

    Column(
        modifier = modifier
            .fillMaxSize()
            .padding(20.dp)
            .verticalScroll(rememberScrollState()),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("dragonAssist", style = MaterialTheme.typography.headlineMedium)
        Text(
            text = listOfNotNull(
                state.engine.ifEmpty { null },
                state.vlmModel.substringAfterLast('/').ifEmpty { null },
                state.speaker.ifEmpty { null },
            ).joinToString(" · ").ifEmpty { "starting…" },
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
        )

        Spacer(Modifier.height(16.dp))

        MediaPanel(
            previewPath = state.previewPath,
            label = state.mediaLabel,
            isVideo = state.mediaKind == MediaKind.Video,
            uploading = state.phase == Phase.Uploading,
            uploadProgress = state.uploadProgress,
            onPhoto = { takePhoto.launch(viewModel.newPhotoTarget()) },
            onVideo = { recordVideo.launch(viewModel.newVideoTarget()) },
        )

        state.warning?.let { warning ->
            Spacer(Modifier.height(8.dp))
            Card(
                Modifier.fillMaxWidth(),
                colors = androidx.compose.material3.CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.tertiaryContainer,
                ),
            ) {
                Text(warning, Modifier.padding(12.dp), style = MaterialTheme.typography.bodySmall)
            }
        }

        Spacer(Modifier.height(20.dp))

        MicButton(
            phase = state.phase,
            level = state.level,
            enabled = state.hasMedia,
            onClick = {
                when (state.phase) {
                    Phase.Idle ->
                        if (hasMicPermission) viewModel.startRecording()
                        else requestMic.launch(Manifest.permission.RECORD_AUDIO)

                    Phase.Recording -> viewModel.stopAndAsk()
                    else -> Unit
                }
            },
        )

        Spacer(Modifier.height(12.dp))

        Text(
            text = when {
                !state.hasMedia -> "Capture a photo or video first"
                state.phase == Phase.Idle && !hasMicPermission -> "Tap to allow the microphone"
                state.phase == Phase.Idle -> "Hold a question in mind, then tap"
                state.phase == Phase.Recording -> "Listening — tap to ask"
                state.phase == Phase.Transcribing -> "Transcribing on the NPU…"
                state.phase == Phase.Uploading -> "Sending to the board…"
                state.phase == Phase.Answering ->
                    if (state.queued) "Board is busy — queued" else "Thinking…"
                else -> ""
            },
            style = MaterialTheme.typography.bodyLarge,
            textAlign = TextAlign.Center,
        )

        if (state.phase == Phase.Recording) {
            Spacer(Modifier.height(16.dp))
            LinearProgressIndicator(
                progress = { state.level },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(6.dp),
            )
        }

        Spacer(Modifier.height(20.dp))

        state.error?.let { message ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(16.dp)) {
                    Text(
                        "Error",
                        style = MaterialTheme.typography.titleSmall,
                        color = MaterialTheme.colorScheme.error,
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(message, style = MaterialTheme.typography.bodyMedium)
                    Spacer(Modifier.height(8.dp))
                    OutlinedButton(onClick = viewModel::dismissError) { Text("Dismiss") }
                }
            }
            Spacer(Modifier.height(12.dp))
        }

        if (state.transcript.isNotEmpty()) {
            Bubble(
                label = "You asked",
                body = state.transcript,
                container = MaterialTheme.colorScheme.secondaryContainer,
            )
            Spacer(Modifier.height(10.dp))
        }

        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Speak answers", style = MaterialTheme.typography.labelMedium)
            Spacer(Modifier.size(8.dp))
            Switch(
                checked = state.speechEnabled,
                onCheckedChange = viewModel::setSpeechEnabled,
            )
        }

        if (state.answer.isNotEmpty()) {
            Bubble(
                label = buildString {
                    append("dragonAssist")
                    if (state.lastAnswerSeconds > 0) {
                        append(" · %.1fs".format(state.lastAnswerSeconds))
                    }
                },
                body = state.answer,
                container = MaterialTheme.colorScheme.primaryContainer,
            )
        }
    }
}

@Composable
private fun MediaPanel(
    previewPath: String?,
    label: String,
    isVideo: Boolean,
    uploading: Boolean,
    uploadProgress: Float,
    onPhoto: () -> Unit,
    onVideo: () -> Unit,
) {
    Card(Modifier.fillMaxWidth()) {
        Column(
            Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            if (previewPath != null) {
                // Keyed on the path so a retake re-decodes rather than showing the old frame.
                val bitmap = remember(previewPath, label) {
                    android.graphics.BitmapFactory.decodeFile(previewPath)
                }
                if (bitmap != null) {
                    Box {
                        Image(
                            bitmap = bitmap.asImageBitmap(),
                            contentDescription = if (isVideo) "Video frame" else "Captured photo",
                            contentScale = ContentScale.Fit,
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(200.dp)
                                .clip(RoundedCornerShape(8.dp)),
                        )
                        if (isVideo) {
                            Text(
                                text = "▶ video",
                                style = MaterialTheme.typography.labelSmall,
                                modifier = Modifier.padding(8.dp),
                            )
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                }
            }

            // Only meaningful for video: a photo goes up in a single inline message.
            if (uploading && isVideo) {
                LinearProgressIndicator(
                    progress = { uploadProgress },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp),
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    "Uploading ${(uploadProgress * 100).toInt()}%",
                    style = MaterialTheme.typography.labelSmall,
                )
                Spacer(Modifier.height(8.dp))
            }

            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = label.ifEmpty { "Nothing captured yet" },
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = onVideo, enabled = !uploading) { Text("Video") }
                    Button(onClick = onPhoto, enabled = !uploading) { Text("Photo") }
                }
            }
        }
    }
}

@Composable
private fun MicButton(phase: Phase, level: Float, enabled: Boolean, onClick: () -> Unit) {
    val pulse by animateFloatAsState(
        targetValue = if (phase == Phase.Recording) 1f + (level * 0.25f) else 1f,
        label = "mic-pulse",
    )
    val busy = phase == Phase.Transcribing || phase == Phase.Uploading || phase == Phase.Answering

    val container = when {
        !enabled -> MaterialTheme.colorScheme.surfaceVariant
        phase == Phase.Recording -> MaterialTheme.colorScheme.error
        busy -> MaterialTheme.colorScheme.secondary
        else -> MaterialTheme.colorScheme.primary
    }

    Box(
        modifier = Modifier
            .size(130.dp)
            .scale(pulse),
        contentAlignment = Alignment.Center,
    ) {
        Button(
            onClick = onClick,
            enabled = enabled && !busy,
            shape = CircleShape,
            colors = ButtonDefaults.buttonColors(containerColor = container),
            modifier = Modifier.fillMaxSize(),
        ) {
            if (busy) {
                CircularProgressIndicator(modifier = Modifier.size(32.dp))
            } else {
                Text(
                    text = if (phase == Phase.Recording) "STOP" else "ASK",
                    style = MaterialTheme.typography.titleMedium,
                )
            }
        }
    }
}

@Composable
private fun Bubble(label: String, body: String, container: androidx.compose.ui.graphics.Color) {
    Card(
        Modifier.fillMaxWidth(),
        colors = androidx.compose.material3.CardDefaults.cardColors(containerColor = container),
    ) {
        Column(Modifier.padding(14.dp)) {
            Text(label, style = MaterialTheme.typography.labelMedium)
            Spacer(Modifier.height(6.dp))
            Text(body, style = MaterialTheme.typography.bodyLarge)
        }
    }
}
