# mobile — on-device speech to text on the phone's NPU

Android app for dragonAssist. Records audio and transcribes it with **Whisper-Base running
on the Snapdragon Hexagon NPU** — no network, no cloud, no round trip to the board.

Verified end to end on a Galaxy S25 Ultra (SM-S938U1, Snapdragon 8 Elite / SM8750,
Android 16). The reference clip transcribes to *"Hello, how are you doing today?"* —
character-identical to the Python reference implementation.

| Stage | Time |
|---|---|
| Mel spectrogram | ~390 ms per 30 s window |
| Encoder (NPU) | ~165 ms |
| Decoder (NPU, 12 tokens) | ~400 ms |
| **Total** | **~1 s** |

## The stack

| Layer | Choice | Why |
|---|---|---|
| Model | `whisper_base`, `precompiled_qnn_onnx`, float | Qualcomm AI Hub publishes it precompiled for `snapdragon_8_elite_for_galaxy` — no export step |
| Runtime | ONNX Runtime `onnxruntime-android-qnn` + QNN EP | Real Kotlin API, so no JNI; QNN pins execution to the Hexagon NPU |
| Front end | `MelSpectrogram.kt` | Whisper's log-mel, verified against HuggingFace to 6×10⁻⁶ |
| Decoder | `WhisperTranscriber.kt` | Greedy autoregressive loop with KV cache, ported from `qai_hub_models` |
| Tokenizer | `WhisperVocab.kt` + exported byte table | Byte-level BPE without porting BPE to Kotlin |
| UI | Jetpack Compose | |

## Setup

Two things are deliberately **not** in this repo: Qualcomm's runtime libraries (proprietary)
and the model bundle (192 MB). Both are one-time fetches.

### 1. QNN runtime libraries

Download the **QAIRT SDK 2.45.x** from
[Qualcomm Software Center](https://softwarecenter.qualcomm.com/catalog/item/Qualcomm_AI_Runtime_Community),
then copy four files into `app/src/main/jniLibs/arm64-v8a/`:

```bash
SDK=~/Downloads/qairt/2.45.41.260507
JNI=app/src/main/jniLibs/arm64-v8a
mkdir -p "$JNI"
cp "$SDK"/lib/aarch64-android/{libQnnHtp.so,libQnnSystem.so,libQnnHtpV79Stub.so} "$JNI/"
cp "$SDK"/lib/hexagon-v79/unsigned/libQnnHtpV79Skel.so "$JNI/"
```

`V79` is the Hexagon revision in Snapdragon 8 Elite. A different SoC needs the matching
`V**` stub and skel. `libQnnHtpPrepare.so` (84 MB) is **not** needed — the models are
precompiled context binaries, so nothing is compiled at runtime.

### 2. Whisper model bundle

From [Qualcomm AI Hub](https://aihub.qualcomm.com/mobile/models/whisper_base), download
`whisper_base` with runtime **ONNX Runtime** (`precompiled_qnn_onnx`), precision float,
device **Snapdragon 8 Elite for Galaxy**. Push all four files to the app's external files
directory:

```bash
adb push <bundle>/encoder.onnx               /sdcard/Android/data/com.example.dragonassist/files/
adb push <bundle>/decoder.onnx               /sdcard/Android/data/com.example.dragonassist/files/
adb push <bundle>/encoder_qairt_context.bin  /sdcard/Android/data/com.example.dragonassist/files/
adb push <bundle>/decoder_qairt_context.bin  /sdcard/Android/data/com.example.dragonassist/files/
```

Push into that directory **directly** — not a subdirectory. A directory created by
`adb shell mkdir` is owned by `shell` with mode `drwxrws---`, which the app cannot
traverse, and the files then look absent.

Without the models the app falls back to `StubTranscriber` and says so in the UI, rather
than crashing.

### 3. Build and run

```bash
./gradlew installDebug
```

## Tests

```bash
./gradlew testDebugUnitTest
```

Nine JVM tests: mel accuracy against a HuggingFace reference tensor, and tokenizer
round-trips including accented text and emoji.

On-device tests need the models present. `connectedAndroidTest` reinstalls the app and
wipes external storage, deleting them — so install first, push second, run third:

```bash
./gradlew installDebug installDebugAndroidTest
# ...push the model files...
adb shell am instrument -w -r com.example.dragonassist.test/androidx.test.runner.AndroidJUnitRunner
```

Fourteen on-device tests cover mel accuracy and speed on real hardware, encoder output,
session lifecycle, full transcription against the Python reference, and proof of NPU
execution.

## How do we know it's on the NPU and not the CPU?

`NpuExecutionProofTest` settles this with a control experiment. The models are QNN
**EPContext** graphs, and ONNX Runtime has no CPU kernel for an `EPContext` node — loading
one without the QNN provider fails with `Failed to find kernel for com.microsoft.EPContext
(ep:'CPUExecutionProvider')`. So a silent CPU fallback is impossible: either QNN binds and
the Hexagon NPU executes, or session creation throws. Corroborated by ~165 ms encoder runs
and QNN's own log (`Detected Snapdragon SOC SM8750`, `Initializing HtpProvider`,
`QnnDevice_create done ... status 0x0`).

## Gotchas worth knowing

**The vendor library declaration is mandatory.** `AndroidManifest.xml` contains:

```xml
<uses-native-library android:name="libcdsprpc.so" android:required="true" />
```

`libQnnHtpV79Stub.so` links against the vendor DSP RPC library, and apps targeting API 31+
must declare vendor libraries or `dlopen` fails inside the app's linker namespace. Without
it, the failure surfaces far downstream and misleadingly as
`QNN_DEVICE_ERROR_INVALID_CONFIG` from `QnnDevice_create`. This one line cost hours.

**Debug QNN with verbose logging.** ORT maps its log severity onto QNN's own logger, so
`OrtEnvironment.getEnvironment(ORT_LOGGING_LEVEL_VERBOSE, id)` is what makes QNN print the
real cause instead of a generic enum. Turn it on *first*.

**Don't set `htp_arch`.** ORT rejects `79` even at 1.28 (`Invalid HTP architecture: 79`),
and QNN detects SM8750 correctly on its own.

**Don't set `ADSP_LIBRARY_PATH`.** ORT sets it when unset, and warns that a pre-existing
value may make the HTP backend fail.

**ORT ≥ 1.28 is required** to load AI Hub's models directly; 1.22 rejects their IR version 13.
Check `maven-metadata.xml` for versions — the Maven Central search API reports stale ones.

## Layout

```
app/src/main/java/com/example/dragonassist/
  MainActivity.kt              Compose entry point
  RecordViewModel.kt           Idle → Recording → Transcribing
  audio/AudioRecorder.kt       16 kHz mono PCM capture, level metering
  audio/MelSpectrogram.kt      Whisper log-mel front end
  audio/WavWriter.kt           Debug dump of each take
  transcribe/Transcriber.kt    The seam
  transcribe/WhisperSession.kt ORT + QNN sessions, fp16 helpers
  transcribe/WhisperTranscriber.kt  Encoder + greedy decode loop
  transcribe/WhisperVocab.kt   Byte-level BPE detokenizer
  transcribe/StubTranscriber.kt     Fallback when models are absent
  ui/RecordScreen.kt           Record button, level meter, transcript
```

`app/src/main/assets/` holds `mel_filters.bin` (Whisper's mel filterbank, exported rather
than reconstructed) and `whisper_tokens.bin` / `.json` (the byte table and special token
ids). Both are generated by scripts kept outside this repo; they are stable and need no
regeneration unless the model changes.

## Debugging a bad transcript

Every take is written to `files/last_recording.wav`. Pull it and run it through desktop
Whisper — if desktop gets it right, the bug is on the device side, not in audio capture:

```bash
adb exec-out run-as com.example.dragonassist cat files/last_recording.wav > out.wav
```

Timings are logged per transcription:

```bash
adb logcat -s WhisperTranscriber
```

## Not done yet

The transcript currently goes no further than the screen. Wiring it to `vlm-qa`'s
OpenAI-compatible endpoint on the IQ-9075 is the next step — note that server binds to
`127.0.0.1:18181`, so it needs either a different bind address or an SSH tunnel before the
phone can reach it over Tailscale.
