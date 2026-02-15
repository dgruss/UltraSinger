"""Convert audio to other formats"""

import os
import subprocess

import librosa
import soundfile as sf

from modules.console_colors import ULTRASINGER_HEAD
from modules.ffmpeg_helper import get_ffmpeg_and_ffprobe_paths, is_ffmpeg_available


def convert_audio_to_mono_wav(input_file_path: str, output_file_path: str) -> None:
    """Convert audio to mono wav"""
    print(f"{ULTRASINGER_HEAD} Converting audio for AI")
    y, sr = librosa.load(input_file_path, mono=True, sr=None)
    sf.write(output_file_path, y, sr)


def convert_wav_to_mp3(input_file_path: str, output_file_path: str) -> None:
    """Convert wav to mp3"""
    print(f"{ULTRASINGER_HEAD} Converting wav to mp3. -> {output_file_path}")

    if not is_ffmpeg_available():
        raise FileNotFoundError("FFmpeg is required to convert audio formats.")

    ffmpeg_path, _ = get_ffmpeg_and_ffprobe_paths()
    ext = os.path.splitext(output_file_path)[1].lower()

    base_cmd = [ffmpeg_path, "-y", "-i", input_file_path, "-vn"]
    codec_args: list[str] = []
    filter_chain: list[str] = []

    if ext in {".m4a", ".mp4"}:
        filter_chain.append("alimiter=limit=0.95")

    if ext in {".m4a", ".mp4"}:
        filter_chain.append("aresample=44100:resampler=soxr")
        codec_args.extend(["-ar", "44100"])

    if filter_chain:
        base_cmd.extend(["-af", ",".join(filter_chain)])

    if ext in {".m4a", ".mp4"}:
        codec_args.extend(["-c:a", "aac", "-b:a", "128k", "-profile:a", "aac_low"])
    elif ext == ".mp3":
        codec_args.extend(["-c:a", "libmp3lame", "-b:a", "128k"])
    elif ext == ".ogg":
        codec_args.extend(["-c:a", "libvorbis", "-q:a", "4"])
    elif ext == ".flac":
        codec_args.extend(["-c:a", "flac"])
    elif ext in {".wav", ".wave"}:
        codec_args.extend(["-c:a", "pcm_s16le"])
    else:
        codec_args.extend(["-c:a", "aac", "-b:a", "128k", "-profile:a", "aac_low"])

    cmd = base_cmd + codec_args + [output_file_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
