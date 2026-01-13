import os
import subprocess

# Core modules
from core.audioextractor import extract_audio
from core.separator import separate_audio
from core.transcribe import transcribe_audio
from core.translator import Translator, SUPPORTED_LANGUAGES
from core.dubbing import generate_dubbed_audio

# ============================================================
# CONFIGURATION - EDIT THESE VALUES
# ============================================================
# Source language of the video (for transcription)
# Use "multi" for auto-detect (recommended for mixed language videos)
# Supported: multi (auto), en, hi, ta, te, kn, ml, mr, bn, gu, pa, or, as
SOURCE_LANGUAGE = "multi"

# Target language for dubbing (for translation + TTS)
# Supported: en, hi, ta, te, kn, ml, mr, bn, gu, pa, or, as
TARGET_LANGUAGE = "hi"

# Input video path - change this to your video file
VIDEO_PATH = "input/sample.mp4"

# Output paths (auto-generated based on input filename and target language)
video_basename = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
ORIGINAL_AUDIO = f"audio/{video_basename}_original.wav"
DUBBED_AUDIO = f"audio/{video_basename}_dubbed_{TARGET_LANGUAGE}.aac"
OUTPUT_VIDEO = f"output/{video_basename}_{TARGET_LANGUAGE}.mp4"

os.makedirs("audio", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Display config
TARGET_LANG_NAME = SUPPORTED_LANGUAGES.get(TARGET_LANGUAGE, "Unknown")

print("=" * 50)
print(f"🎬 VIDEO DUBBING PIPELINE")
print(f"   Input:  {VIDEO_PATH}")
print(f"   Source: {SOURCE_LANGUAGE} → Target: {TARGET_LANG_NAME} ({TARGET_LANGUAGE})")
print(f"   Output: {OUTPUT_VIDEO}")
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
# STEP 3: Transcribe Vocals (Google Cloud Speech-to-Text)
# ============================================================
print("=" * 50)
print(f"STEP 3: Transcribing vocals with Google STT ({SOURCE_LANGUAGE})")
print("=" * 50)
utterances = transcribe_audio(vocals_path, source_language=SOURCE_LANGUAGE)
print(f"✅ Got {len(utterances)} utterances")

for utt in utterances[:3]:  # Show first 3
    print(f"  [Speaker {utt.get('speaker', 0)}] [{utt['start']:.1f}s - {utt['end']:.1f}s]: {utt['transcript'][:50]}...")

# ============================================================
# STEP 4: Translate to Target Language
# ============================================================
print("=" * 50)
print(f"STEP 4: Translating to {TARGET_LANG_NAME}")
print("=" * 50)
translator = Translator(target_language=TARGET_LANGUAGE)
translated_segments = translator.translate_segments(utterances)
print(f"✅ Translated {len(translated_segments)} segments")

# ============================================================
# STEP 5 & 6: Generate TTS + Mix with Background
# ============================================================
generate_dubbed_audio(background_path, translated_segments, DUBBED_AUDIO, language=TARGET_LANGUAGE)

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
