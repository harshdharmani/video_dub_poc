import time
import os
import shutil
from typing import Dict, Any, List

# Core modules (assuming these exist from previous Context)
from core.audioextractor import extract_audio
from core.separator import separate_audio
from core.transcribe import transcribe_audio
from core.translator import Translator, SUPPORTED_LANGUAGES
from core.dubbing import generate_dubbed_audio

def process_video(video_path: str, source_lang: str, target_lang: str) -> Dict[str, Any]:
    """
    Orchestrates the video dubbing process with timing.
    
    Returns:
        Dict containing:
        - output_video_path (str)
        - transcription (str or list)
        - timings (dict)
    """
    timings = {}
    
    # Generate paths
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    
    # Ensure directories exist
    os.makedirs("audio", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    original_audio = f"audio/{video_basename}_original.wav"
    dubbed_audio = f"audio/{video_basename}_dubbed_{target_lang}.aac"
    output_video = f"output/{video_basename}_{target_lang}.mp4"
    
    start_total = time.time()
    
    # STEP 1: Extract Audio
    print(f"--- Step 1: Extracting Audio ---")
    t0 = time.time()
    extract_audio(video_path, original_audio)
    timings["extract_audio"] = time.time() - t0
    
    # STEP 2: Separate Audio
    print(f"--- Step 2: Separating Audio ---")
    # We allow this to run but maybe separation isn't strictly needing independent timing 
    # if it's not a primary "blocking" user step? But let's track it implicitly or explicitly.
    # The original main.py separated vocals/bg. Let's include that in "extract_audio" or separate.
    # I'll add a 'separation' timing for clarity if desired, or bundle it.
    # The separator.py uses demucs.
    t0 = time.time()
    vocals_path, background_path = separate_audio(original_audio)
    timings["separation"] = time.time() - t0
    
    # STEP 3: Transcribe
    print(f"--- Step 3: Transcribing ---")
    t0 = time.time()
    utterances = transcribe_audio(vocals_path, source_language=source_lang)
    timings["transcribe"] = time.time() - t0
    
    # Prepare transcription text for return
    full_transcript = []
    for utt in utterances:
        full_transcript.append(f"[Speaker {utt.get('speaker', 0)}] {utt['transcript']}")
    transcription_text = "\n".join(full_transcript)

    # STEP 4: Translate
    print(f"--- Step 4: Translating ---")
    t0 = time.time()
    # Translator expects target_language in constructor or method?
    # Checking previous main.py: Translator(target_language=TARGET_LANGUAGE)
    translator = Translator(target_language=target_lang)
    translated_segments = translator.translate_segments(utterances)
    timings["translate"] = time.time() - t0
    
    # STEP 5: Synthesize & Mix
    print(f"--- Step 5: Synthesizing & Mixing ---")
    t0 = time.time()
    # dubbing.py: generate_dubbed_audio(background_path, segments, output_path, language=...)
    generate_dubbed_audio(background_path, translated_segments, dubbed_audio, language=target_lang)
    timings["synthesize"] = time.time() - t0
    
    # STEP 6: Merge Video
    print(f"--- Step 6: Merging Video ---")
    t0 = time.time()
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", dubbed_audio,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        output_video
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    timings["merge_video"] = time.time() - t0
    
    timings["total_dubbing"] = time.time() - start_total
    
    return {
        "output_video_path": output_video,
        "transcription": transcription_text,
        "timings": timings
    }
