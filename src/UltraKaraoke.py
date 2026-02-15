"""UltraKaraoke: simple Demucs-based karaoke generator.

Usage:
    python UltraKaraoke.py -i input.m4a -o output.m4a
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from modules.Audio.separation import DemucsModel, separate_audio
from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    green_highlighted,
    red_highlighted,
)
from modules.ffmpeg_helper import get_ffmpeg_and_ffprobe_paths, is_ffmpeg_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a karaoke (no-vocals) version of an audio file using Demucs."
    )
    parser.add_argument("-i", "--input", required=True, help="Input audio file (e.g., .m4a, .mp3, .wav)")
    parser.add_argument("-o", "--output", help="Output audio file (e.g., .m4a, .mp3, .wav)")
    parser.add_argument(
        "--model",
        default=DemucsModel.HTDEMUCS.value,
        choices=[m.value for m in DemucsModel],
        help="Demucs model to use (default: htdemucs)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device for Demucs (e.g., cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--temp-dir",
        help="Optional temp directory for separation cache (default: auto temp)",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary separation files",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    if not output_arg:
        return input_path.with_name(f"{input_path.stem} [Karaoke]{input_path.suffix}")

    output_path = Path(output_arg)
    if output_path.is_dir():
        return output_path / f"{input_path.stem} [Karaoke]{input_path.suffix}"
    return output_path


def build_ffmpeg_command(input_wav: Path, output_path: Path) -> list[str]:
    ffmpeg_path, _ = get_ffmpeg_and_ffprobe_paths()
    ext = output_path.suffix.lower()

    base_cmd = [ffmpeg_path, "-y", "-i", str(input_wav), "-vn"]

    codec_args: list[str] = []
    if ext in {".m4a", ".mp4"}:
        codec_args = ["-c:a", "aac", "-b:a", "128k"]
    elif ext == ".mp3":
        codec_args = ["-c:a", "libmp3lame", "-b:a", "128k"]
    elif ext == ".ogg":
        codec_args = ["-c:a", "libvorbis", "-q:a", "4"]
    elif ext == ".flac":
        codec_args = ["-c:a", "flac"]

    return base_cmd + codec_args + [str(output_path)]


def create_karaoke(
    input_path: Path,
    output_path: Path,
    model: DemucsModel,
    device: str,
    temp_dir: Path,
    keep_temp: bool,
    created_temp: bool,
) -> None:
    print(
        f"{ULTRASINGER_HEAD} Using model {blue_highlighted(model.value)} on device {blue_highlighted(device)}"
    )

    separate_audio(str(input_path), str(temp_dir), model, device=device)

    separated_path = temp_dir / "separated" / model.value / input_path.stem
    no_vocals_path = separated_path / "no_vocals.wav"

    if not no_vocals_path.exists():
        raise FileNotFoundError(
            f"Expected Demucs output not found: {no_vocals_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in {".wav", ".wave"}:
        shutil.copyfile(no_vocals_path, output_path)
    else:
        if not is_ffmpeg_available():
            raise FileNotFoundError("FFmpeg is required to convert output audio formats.")

        cmd = build_ffmpeg_command(no_vocals_path, output_path)
        print(f"{ULTRASINGER_HEAD} FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")

    print(f"{ULTRASINGER_HEAD} {green_highlighted('Karaoke created')} -> {output_path}")

    if created_temp and not keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> int:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Input file not found')}: {input_path}")
        return 2

    output_path = resolve_output_path(input_path, args.output)
    model = DemucsModel(args.model)

    created_temp = False
    if args.temp_dir:
        temp_dir = Path(args.temp_dir).expanduser().resolve()
        temp_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = Path(tempfile.mkdtemp(prefix="ultrakaraoke_"))
        created_temp = True

    try:
        create_karaoke(
            input_path=input_path,
            output_path=output_path,
            model=model,
            device=args.device,
            temp_dir=temp_dir,
            keep_temp=args.keep_temp,
            created_temp=created_temp,
        )
    except Exception as exc:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Error')}: {exc}")
        if created_temp and not args.keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())