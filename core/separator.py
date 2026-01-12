import os
import sys
import subprocess
from typing import Tuple

def separate_audio(audio_path: str, output_dir: str = "audio/separated", force: bool = False) -> Tuple[str, str]:
    """
    Separates audio into vocals and background using Demucs.
    If the separation has already been performed and the output files exist,
    the function will skip re-running Demucs unless `force=True`.

    Returns:
        Tuple of (vocals_path, background_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine expected output paths (Demucs creates .mp3 files)
    audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
    demucs_output = os.path.join(output_dir, "htdemucs", audio_basename)
    vocals_path = os.path.join(demucs_output, "vocals.mp3")
    background_path = os.path.join(demucs_output, "no_vocals.mp3")

    # If files already exist and not forced, skip processing
    if not force and os.path.exists(vocals_path) and os.path.exists(background_path):
        print(f"✅ Skipping Demucs – existing separation found:")
        print(f"   Vocals: {vocals_path}")
        print(f"   Background: {background_path}")
        return vocals_path, background_path

    print("=" * 50)
    print("STEP 2: Separating vocals from background (Demucs)")
    print("=" * 50)
    print(f"Input: {audio_path}")
    print("⏳ This may take several minutes...")

    # Run Demucs with MP3 output (bypasses torchaudio WAV issue on Python 3.13)
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "--mp3",  # Save as MP3 to bypass torchcodec
        "-o", output_dir,
        audio_path
    ]

    subprocess.run(cmd)

    # Verify files exist after processing
    if os.path.exists(vocals_path) and os.path.exists(background_path):
        print(f"✅ Vocals extracted: {vocals_path}")
        print(f"✅ Background extracted: {background_path}")
        return vocals_path, background_path
    else:
        print("❌ Separation failed. Files not found.")
        print(f"   Expected vocals: {vocals_path}")
        print(f"   Expected background: {background_path}")
        if os.path.exists(demucs_output):
            print(f"   Found files: {os.listdir(demucs_output)}")
        raise FileNotFoundError("Demucs separation failed")
