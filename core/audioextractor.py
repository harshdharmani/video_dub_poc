import subprocess
import os

def extract_audio(video_path: str, output_audio_path: str):
    """
    Extracts audio from a video file using FFmpeg.
    """
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        output_audio_path
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    if not os.path.exists(output_audio_path):
        raise RuntimeError("Audio extraction failed")

    return output_audio_path
