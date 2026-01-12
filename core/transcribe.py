import os
import subprocess
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from deepgram import DeepgramClient

# Load .env from project root safely
load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

# Language to Deepgram model mapping
# Nova-3 supports: English, Hindi, and many European languages
# Whisper supports: 50+ languages including all Indian regional languages
LANGUAGE_MODEL_MAP = {
    # Languages supported by Nova-3 (faster, more accurate for these)
    "en": {"model": "nova-3", "code": "en"},
    "hi": {"model": "nova-3", "code": "hi"},
    
    # Indian regional languages - use Whisper (supports these natively)
    "ta": {"model": "whisper-large", "code": "ta"},   # Tamil
    "te": {"model": "whisper-large", "code": "te"},   # Telugu
    "kn": {"model": "whisper-large", "code": "kn"},   # Kannada
    "ml": {"model": "whisper-large", "code": "ml"},   # Malayalam
    "mr": {"model": "whisper-large", "code": "mr"},   # Marathi
    "bn": {"model": "whisper-large", "code": "bn"},   # Bengali
    "gu": {"model": "whisper-large", "code": "gu"},   # Gujarati
    "pa": {"model": "whisper-large", "code": "pa"},   # Punjabi
    "or": {"model": "whisper-large", "code": "or"},   # Odia
    "as": {"model": "whisper-large", "code": "as"},   # Assamese
    
    # Auto-detect (for mixed language videos)
    "multi": {"model": "nova-3", "code": "multi"},
}

def get_supported_languages() -> List[str]:
    """Returns list of supported source language codes."""
    return list(LANGUAGE_MODEL_MAP.keys())


def _extract_audio_chunk(audio_path: str, start: float, end: float, output_path: str) -> str:
    """Extracts a chunk of audio using FFmpeg."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ss", str(start),
        "-t", str(duration),
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        output_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def _diarize_audio(audio_path: str, api_key: str) -> List[Dict[str, Any]]:
    """
    PASS 1: Get speaker diarization using Nova-3 with multi-language support.
    Returns speaker segments with timestamps (no transcription yet).
    """
    print("  📢 Pass 1: Diarization with Nova-3 (language=multi)")
    
    deepgram = DeepgramClient(api_key=api_key)
    
    with open(audio_path, "rb") as audio_file:
        response = deepgram.listen.v1.media.transcribe_file(
            request=audio_file.read(),
            model="nova-3",
            language="multi",  # Multi-language for mixed content
            diarize=True,
            utterances=True,
            smart_format=True,
            punctuate=True
        )
    
    segments = []
    if response.results and response.results.utterances:
        for utt in response.results.utterances:
            segments.append({
                "start": float(utt.start),
                "end": float(utt.end),
                "speaker": int(utt.speaker) if hasattr(utt, 'speaker') and utt.speaker is not None else 0,
                "transcript": utt.transcript.strip()  # Nova-3 transcription (may be rough for non-en/hi)
            })
    
    speakers = set(seg["speaker"] for seg in segments)
    print(f"  ✅ Found {len(segments)} segments with {len(speakers)} speaker(s)")
    return segments


def _transcribe_chunk(audio_path: str, api_key: str, language: str = "en") -> str:
    """
    PASS 2: Transcribe a single audio chunk with the appropriate model.
    For regional languages, uses Whisper which auto-detects if needed.
    """
    if language not in LANGUAGE_MODEL_MAP:
        language = "en"
    
    config = LANGUAGE_MODEL_MAP[language]
    model = config["model"]
    lang_code = config["code"]
    
    deepgram = DeepgramClient(api_key=api_key)
    
    with open(audio_path, "rb") as audio_file:
        response = deepgram.listen.v1.media.transcribe_file(
            request=audio_file.read(),
            model=model,
            language=lang_code,
            smart_format=True,
            punctuate=True
        )
    
    if response.results and response.results.channels:
        alternatives = response.results.channels[0].alternatives
        if alternatives:
            return alternatives[0].transcript.strip()
    
    return ""


def transcribe_audio(
    audio_path: str, 
    source_language: str = "en",
    enable_diarization: bool = True,
    use_two_pass: bool = True,
    temp_dir: str = "temp_chunks"
) -> List[Dict[str, Any]]:
    """
    Transcribes audio with speaker diarization support for any language.
    
    TWO-PASS APPROACH (for regional languages with diarization):
    - Pass 1: Nova-3 with diarize=true, language=multi → get speaker timestamps
    - Pass 2: For each speaker segment, transcribe with appropriate model
    
    Args:
        audio_path: Path to the audio file
        source_language: ISO 639-1 language code (en, hi, ta, te, kn, ml, etc.)
        enable_diarization: Enable speaker diarization
        use_two_pass: Use two-pass method for regional languages (recommended)
        temp_dir: Directory for temporary audio chunks

    Output format:
    [
        {"start": 0.0, "end": 2.5, "transcript": "Hello", "speaker": 0},
        ...
    ]
    """
    print(f"Transcribing audio with Deepgram...")
    print(f"  → Source language: {source_language}")
    
    # --- Validate API Key ---
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY not found. Check your .env file.")
    
    # Determine if we need two-pass (for regional languages that need diarization)
    is_regional = source_language in ["ta", "te", "kn", "ml", "mr", "bn", "gu", "pa", "or", "as"]
    needs_two_pass = use_two_pass and enable_diarization and is_regional
    
    if needs_two_pass:
        print(f"  → Using TWO-PASS method (diarization + regional language)")
        return _two_pass_transcribe(audio_path, api_key, source_language, temp_dir)
    else:
        print(f"  → Using SINGLE-PASS method")
        return _single_pass_transcribe(audio_path, api_key, source_language, enable_diarization)


def _single_pass_transcribe(
    audio_path: str, 
    api_key: str, 
    source_language: str,
    enable_diarization: bool
) -> List[Dict[str, Any]]:
    """Single-pass transcription (for en/hi or when diarization not needed)."""
    
    if source_language not in LANGUAGE_MODEL_MAP:
        print(f"⚠️ Language '{source_language}' not in map, defaulting to English")
        source_language = "en"
    
    config = LANGUAGE_MODEL_MAP[source_language]
    model = config["model"]
    lang_code = config["code"]
    
    print(f"  → Model: {model}")
    
    # Whisper doesn't support diarization
    use_diarization = enable_diarization and not model.startswith("whisper")
    if enable_diarization and model.startswith("whisper"):
        print(f"  ⚠️ Diarization not available for Whisper model")
    if use_diarization:
        print(f"  → Speaker diarization: ENABLED")

    deepgram = DeepgramClient(api_key=api_key)

    with open(audio_path, "rb") as audio_file:
        response = deepgram.listen.v1.media.transcribe_file(
            request=audio_file.read(),
            model=model,
            language=lang_code,
            smart_format=True,
            punctuate=True,
            utterances=True,
            diarize=use_diarization
        )

    segments: List[Dict[str, Any]] = []

    if response.results and response.results.utterances:
        for utt in response.results.utterances:
            segment = {
                "start": float(utt.start),
                "end": float(utt.end),
                "transcript": utt.transcript.strip(),
                "speaker": int(utt.speaker) if hasattr(utt, 'speaker') and utt.speaker is not None else 0
            }
            segments.append(segment)
        
        speakers = set(seg["speaker"] for seg in segments)
        print(f"  ✅ Found {len(segments)} segments with {len(speakers)} speaker(s)")
        return segments

    # Fallback for no utterances
    if response.results and response.results.channels:
        alternatives = response.results.channels[0].alternatives
        if alternatives and alternatives[0].transcript:
            return [{
                "start": 0.0,
                "end": 0.0,
                "transcript": alternatives[0].transcript.strip(),
                "speaker": 0
            }]

    return []


def _two_pass_transcribe(
    audio_path: str, 
    api_key: str, 
    source_language: str,
    temp_dir: str
) -> List[Dict[str, Any]]:
    """
    Two-pass transcription for regional languages with diarization.
    Pass 1: Diarization with Nova-3 (multi) → speaker timestamps
    Pass 2: Transcribe each chunk with Whisper → accurate transcription
    """
    import shutil
    
    os.makedirs(temp_dir, exist_ok=True)
    
    # PASS 1: Get diarization (speaker timestamps)
    diarized_segments = _diarize_audio(audio_path, api_key)
    
    if not diarized_segments:
        print("  ⚠️ No segments found in diarization pass")
        return []
    
    # PASS 2: Transcribe each segment with appropriate model
    print(f"  📝 Pass 2: Transcribing {len(diarized_segments)} chunks with Whisper")
    
    final_segments = []
    for i, seg in enumerate(diarized_segments):
        chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
        
        # Extract audio chunk
        _extract_audio_chunk(audio_path, seg["start"], seg["end"], chunk_path)
        
        # Transcribe chunk with language-specific model
        if os.path.exists(chunk_path):
            transcript = _transcribe_chunk(chunk_path, api_key, source_language)
            
            # Use Whisper transcription if available, else fall back to Nova-3
            if transcript:
                seg["transcript"] = transcript
            
            final_segments.append(seg)
            print(f"    [{seg['start']:.1f}s] Speaker {seg['speaker']}: {seg['transcript'][:40]}...")
    
    # Cleanup temp chunks
    try:
        shutil.rmtree(temp_dir)
        print(f"  🧹 Cleaned up temp chunks")
    except:
        pass
    
    print(f"  ✅ Two-pass transcription complete: {len(final_segments)} segments")
    return final_segments


