import os
import subprocess

# Core modules
from core.audioextractor import extract_audio
from core.separator import separate_audio
from core.transcribe import transcribe_audio
from core.translator import Translator
from core.dubbing import generate_dubbed_audio

# ============================================================
# CONFIGURATION
# ============================================================
# Target language for dubbing (change this to switch languages)
# Supported: hi (Hindi), ta (Tamil), te (Telugu), kn (Kannada), 
#            ml (Malayalam), mr (Marathi), bn (Bengali), gu (Gujarati), pa (Punjabi)
TARGET_LANGUAGE = "hi"

# Paths
VIDEO_PATH = "input/sample.mp4"
ORIGINAL_AUDIO = "audio/original.wav"
DUBBED_AUDIO = "audio/dubbed_final.aac"
OUTPUT_VIDEO = "output/dubbed.mp4"

os.makedirs("audio", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Language name mapping for display
from core.translator import SUPPORTED_LANGUAGES
LANGUAGE_NAME = SUPPORTED_LANGUAGES.get(TARGET_LANGUAGE, "Hindi")

print("=" * 50)
print(f"🎬 VIDEO DUBBING PIPELINE")
print(f"   Target Language: {LANGUAGE_NAME} ({TARGET_LANGUAGE})")
print("=" * 50)

# ============================================================
# STEP 1: Extract Audio from Video
# ============================================================
print("=" * 50)
print("STEP 1: Extracting audio from video")
print("=" * 50)
extract_audio(VIDEO_PATH, ORIGINAL_AUDIO)
print(f"✅ Audio extracted: {ORIGINAL_AUDIO}")

# ============================================================
# STEP 2: Separate Vocals + Background (Demucs)
# ============================================================
vocals_path, background_path = separate_audio(ORIGINAL_AUDIO)

# ============================================================
# STEP 3: Transcribe Vocals (Deepgram)
# ============================================================
print("=" * 50)
print("STEP 3: Transcribing vocals (Deepgram with diarization)")
print("=" * 50)
utterances = transcribe_audio(vocals_path)  # Transcribe CLEAN vocals
print(f"✅ Got {len(utterances)} utterances")

for utt in utterances[:3]:  # Show first 3
    print(f"  [Speaker {utt.get('speaker', 0)}] [{utt['start']:.1f}s - {utt['end']:.1f}s]: {utt['transcript'][:50]}...")

# ============================================================
# STEP 4: Translate to Target Language
# ============================================================
print("=" * 50)
print(f"STEP 4: Translating to {LANGUAGE_NAME}")
print("=" * 50)
translator = Translator(target_language=TARGET_LANGUAGE)
translated_segments = translator.translate_segments(utterances)
print(f"✅ Translated {len(translated_segments)} segments")

# ============================================================
# STEP 5 & 6: Generate Hindi TTS + Mix with Background
# ============================================================
generate_dubbed_audio(background_path, translated_segments, DUBBED_AUDIO)

# ============================================================
# STEP 7: Merge Audio with Video
# ============================================================
print("=" * 50)
print("STEP 7: Merging dubbed audio with video")
print("=" * 50)
subprocess.run([
    "ffmpeg", "-y",
    "-i", VIDEO_PATH,
    "-i", DUBBED_AUDIO,
    "-map", "0:v:0",
    "-map", "1:a:0",
    "-c:v", "copy",
    "-c:a", "copy",
    "-shortest",
    OUTPUT_VIDEO
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("=" * 50)
print("🎬 DUBBING COMPLETE!")
print(f"   Output: {OUTPUT_VIDEO}")
print("=" * 50)
