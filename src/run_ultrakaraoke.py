from pathlib import Path
import subprocess
import shutil
import os
import sys

base_dir = Path("/mnt/d/games/usdx/Songs")

dirs = base_dir.iterdir()
dirs = list(dirs)
dirs.insert(0, base_dir)

for outer in dirs:
    if not outer.is_dir():
        continue
    for inner in outer.iterdir():
        print(f"Handle {inner}?")
        if not inner.is_dir():
            continue
        marker = inner / "processing.tmp"
        if marker.exists():
            print(f"Already processing {inner}")
            continue
        marker.write_text("Temp marker")
        for m4a in inner.glob("*.m4a"):
            if "[Karaoke]" in m4a.name:
                continue
            karaoke_m4a = inner / f"{m4a.stem} [Karaoke].m4a"
            if karaoke_m4a.exists():
                continue
            # Call UltraKaraoke
            print(f"Processing {m4a}")
            subprocess.run([
                "python3.11",
                "/mnt/d/games/usdxwork/tools/UltraSinger/src/UltraKaraoke.py",
                "-i", str(m4a),
                "--whisper", "tiny"
            ])
            # convert resulting wav to m4a
            # e.g. for d:\games\usdx\Songs\NeedVideo\Aradszky László\Aradszky László - Nem Csak A Húszéveseké A Világ.m4a
            # it becomes: d:\games\usdx\Songs\NeedVideo\Aradszky László\output\Aradszky László - Nem Csak A Húszéveseké A Világ\cache\separated\htdemucs\Aradszky László - Nem Csak A Húszéveseké A Világ\no_vocals.wav
            wav_path = inner / "output" / m4a.stem / "cache" / "separated" / "htdemucs" / f"{m4a.stem}/no_vocals.wav"
            if not wav_path.exists():
                print(f"Error: expected wav file not found: {wav_path}")
                marker.unlink()
                continue
            subprocess.run([
                "ffmpeg",
                "-i", str(wav_path),
                "-c:a", "aac",
                "-b:a", "128k",
                "-y",
                str(karaoke_m4a)
            ])
            # Move result, clean up
            output_dir = inner / "output"
            # if output_dir folder exists, remove it
            if output_dir.is_dir():
                shutil.rmtree(output_dir, ignore_errors=True)
        marker.unlink()
