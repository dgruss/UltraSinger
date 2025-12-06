"""UltraTimer aligns Ultrastar timing to audio by optimizing GAP and BPM."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Optional, Sequence, Tuple

import librosa
import numpy as np
from matplotlib import pyplot as plt

from modules.Audio.convert_audio import convert_audio_to_mono_wav
from modules.Audio.denoise import denoise_vocal_audio
from modules.Audio.separation import DemucsModel, separate_vocal_from_audio
from modules.Audio.silence_processing import mute_no_singing_parts
from modules.DeviceDetection.device_detection import check_gpu_support
from modules.os_helper import create_folder


@dataclass(slots=True)
class UltraTimerConfig:
    """Configuration values collected from the command line."""

    audio_path: Path
    ultrastar_txt_path: Path
    output_dir: Path
    cache_dir: Optional[Path]
    pytorch_device: Optional[str]
    demucs_model: Optional[DemucsModel]
    skip_cache_separation: bool
    no_denoise: bool
    sample_rate: int
    envelope_window_ms: float
    bpm_search_range: Tuple[float, float]
    bpm_search_step: float
    gap_search_range_ms: Tuple[int, int]
    gap_search_step_ms: int
    only_gap: bool
    correlation_metric: str
    binary_threshold: Optional[float]
    preprocess: str
    verbose: bool
    generate_debug_plot: bool
    debug_plot_path: Optional[Path]
    weight_early: float
    threads: int
    plot_pitch: bool
    use_pitch_correlation: bool
    moving_average_window: int
    ignore_gaps_below_ms: float


@dataclass(slots=True)
class PreparedAudioArtifacts:
    """File paths produced during audio preparation."""

    cache_dir: Path
    separation_dir: Path
    vocals_path: Path
    instrumental_path: Path
    mono_vocals_path: Path
    muted_vocals_path: Path
    waveform: np.ndarray
    sample_rate: int
    debug_plot_path: Optional[Path]
    envelope: np.ndarray
    binary_trace: Optional[np.ndarray]


@dataclass(slots=True)
class UltrastarNote:
    """Minimal representation of a note line in an Ultrastar txt file."""

    type: str
    start_beats: int
    length_beats: int
    pitch: int
    text: str


@dataclass(slots=True)
class UltrastarSong:
    """Parsed data required for timing alignment from an Ultrastar txt file."""

    bpm: float
    gap_ms: float
    headers: dict[str, str]
    notes: list[UltrastarNote]


@dataclass(slots=True)
class AlignmentResult:
    bpm: float
    gap_ms: float
    correlation: float
    metric: str
    simulated_activity: Optional[np.ndarray]


def _existing_file(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path_str}")
    return path


def _existing_dir(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Directory not found: {path_str}")
    return path


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return number


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return number


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0")
    return number


def _non_negative_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value} is not a valid float") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0")
    return number


def _range_pair(value: str, cast) -> Tuple[float, float]:
    try:
        low_str, high_str = value.split(",", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated pair, e.g. 110,150"
        ) from exc
    low = cast(low_str)
    high = cast(high_str)
    if low >= high:
        raise argparse.ArgumentTypeError("Lower bound must be < upper bound")
    return low, high


def _select_pytorch_device(config: UltraTimerConfig) -> str:
    if config.pytorch_device:
        return config.pytorch_device
    _, detected_device = check_gpu_support()
    return detected_device


def _normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(waveform))
    if peak <= np.finfo(float).eps:
        return waveform
    return waveform / peak


def compute_energy_envelope(
    waveform: np.ndarray, sample_rate: int, window_ms: float
) -> np.ndarray:
    window_samples = max(1, int(round(window_ms * sample_rate / 1000.0)))
    absolute = np.abs(waveform)
    if window_samples <= 1:
        return absolute.astype(np.float32, copy=False)
    kernel = np.ones(window_samples, dtype=np.float32) / window_samples
    envelope = np.convolve(absolute, kernel, mode="same")
    return envelope.astype(np.float32, copy=False)


def binarize_signal(signal: np.ndarray, threshold: float) -> np.ndarray:
    binary = np.zeros_like(signal, dtype=np.float32)
    binary[signal >= threshold] = 1.0
    return binary


def compute_rms(signal_in: np.ndarray, window_ms: float, sample_rate: int) -> np.ndarray:
    window_samples = max(1, int(round(window_ms * sample_rate / 1000.0)))
    pad = window_samples // 2
    squared = np.square(signal_in)
    kernel = np.ones(window_samples, dtype=np.float32) / window_samples
    rms = np.sqrt(np.convolve(squared, kernel, mode="same"))
    return rms.astype(np.float32, copy=False)


def save_waveform_debug_plot(
    audio_waveform: np.ndarray,
    simulated_waveform: np.ndarray,
    sample_rate: int,
    output_path: Path,
    title: str = "Waveform Alignment Debug",
    secondary_waveform: Optional[np.ndarray] = None,
    secondary_label: str = "Reference signal",
    pitch_overlay: Optional[tuple[np.ndarray, Optional[np.ndarray]]] = None,
) -> None:
    if output_path is None:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        (audio_waveform, "Muted vocal waveform", "tab:blue", "Normalized amplitude"),
    ]
    if secondary_waveform is not None:
        panels.append((secondary_waveform, secondary_label, "tab:green", "Value"))
    panels.append((simulated_waveform, "Simulated note activity", "tab:orange", "Activity"))

    num_panels = len(panels) + (1 if pitch_overlay is not None else 0)
    fig, axes = plt.subplots(num_panels, 1, sharex=False, figsize=(12, 3 * num_panels))
    if num_panels == 1:
        axes = [axes]
    fig.suptitle(title)

    plotted_axes = axes if pitch_overlay is None else axes[:-1]

    for axis, (waveform, label, color, ylabel) in zip(plotted_axes, panels):
        duration = len(waveform) / sample_rate if len(waveform) else 0
        time_axis = np.linspace(0, duration, num=len(waveform), endpoint=False)
        unique_values = np.unique(waveform)
        if len(unique_values) <= 2 and np.all((unique_values == 0) | (unique_values == 1)):
            axis.step(time_axis, waveform, where="post", color=color, linewidth=0.6)
        else:
            axis.plot(time_axis, waveform, color=color, linewidth=0.6)
        axis.set_ylabel(ylabel)
        axis.set_title(label)
        axis.grid(True, linestyle=":", linewidth=0.3)

    if pitch_overlay is not None:
        note_trace, detected_trace = pitch_overlay
        axis = axes[-1]
        duration = len(note_trace) / sample_rate if len(note_trace) else 0
        time_axis = np.linspace(0, duration, num=len(note_trace), endpoint=False)
        axis.step(time_axis, note_trace, where="post", color="tab:red", linewidth=0.6, label="Ultrastar notes")
        if detected_trace is not None:
            axis.plot(time_axis, detected_trace, color="tab:purple", linewidth=0.6, label="Detected pitch")
        axis.set_ylabel("Frequency (Hz)")
        axis.set_title("Pitch contour")
        axis.grid(True, linestyle=":", linewidth=0.3)
        axis.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


NOTE_LINE_PATTERN = re.compile(
    r"^(?P<type>.)(?P<body>\s+\d+\s+\d+\s+-?\d+\s?.*)$"
)


def parse_ultrastar_song(file_path: Path) -> UltrastarSong:
    headers: dict[str, str] = {}
    notes: list[UltrastarNote] = []

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                key, value = _parse_header_line(line)
                headers[key] = value
                continue

            match = NOTE_LINE_PATTERN.match(line)
            if match is None:
                continue
            note_tokens = match.group("body").split(maxsplit=3)
            if len(note_tokens) < 3:
                continue
            type_symbol = match.group("type")
            start_beats = int(note_tokens[0])
            length_beats = int(note_tokens[1])
            pitch = int(note_tokens[2]) if len(note_tokens) >= 3 else 0
            lyric = note_tokens[3] if len(note_tokens) == 4 else ""
            notes.append(
                UltrastarNote(
                    type=type_symbol,
                    start_beats=start_beats,
                    length_beats=length_beats,
                    pitch=pitch,
                    text=lyric,
                )
            )

    bpm = float(headers.get("BPM", "0"))
    if bpm <= 0:
        raise ValueError("Ultrastar txt is missing a valid #BPM header.")

    gap_ms = float(headers.get("GAP", "0"))

    return UltrastarSong(bpm=bpm, gap_ms=gap_ms, headers=headers, notes=notes)


def _parse_header_line(line: str) -> tuple[str, str]:
    try:
        key, value = line[1:].split(":", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"Malformed header line: {line}") from exc
    return key.strip().upper(), value.strip()


def beats_to_seconds(beats: float, bpm: float) -> float:
    if bpm <= 0:
        raise ValueError("BPM must be positive to convert beats to seconds.")
    return beats * 60.0 / (bpm * 4.0)


def generate_simulated_activity(
    song: UltrastarSong,
    sample_rate: int,
    target_duration_seconds: Optional[float] = None,
    bpm_override: Optional[float] = None,
    gap_ms_override: Optional[float] = None,
    verbose: bool = False,
    weight_early: float = 0.0,
) -> np.ndarray:
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    bpm = bpm_override if bpm_override is not None else song.bpm
    gap_ms = gap_ms_override if gap_ms_override is not None else song.gap_ms

    gap_seconds = gap_ms / 1000.0
    beat_duration = beats_to_seconds(1.0, bpm)

    intervals: list[tuple[float, float, int, int, str]] = []
    max_time = target_duration_seconds or 0.0

    for note in song.notes:
        if note.length_beats <= 0:
            continue
        start_seconds = gap_seconds + note.start_beats * beat_duration
        end_seconds = start_seconds + note.length_beats * beat_duration
        if end_seconds <= start_seconds:
            continue
        intervals.append((start_seconds, end_seconds, note.start_beats, note.length_beats, note.text))
        max_time = max(max_time, end_seconds)

    if max_time <= 0:
        return np.zeros(1, dtype=np.float32)

    num_samples = max(1, int(np.ceil(max_time * sample_rate)))
    activity = np.zeros(num_samples, dtype=np.float32)

    for (start, end, sb, lb, txt) in intervals:
        start_idx = int(np.floor(start * sample_rate))
        end_idx = int(np.ceil(end * sample_rate))
        clipped = False
        if start_idx < 0:
            start_idx = 0
            clipped = True
        if end_idx > num_samples:
            end_idx = num_samples
            clipped = True
        if end_idx > start_idx:
            activity[start_idx:end_idx] = 1.0
        # if verbose:
        #     print(
        #         f"  interval beat_start={sb} len_beats={lb} text={txt!r} -> start={start:.3f}s end={end:.3f}s "
        #         f"samples=({start_idx},{end_idx}) length={max(0,end_idx-start_idx)} clipped={clipped}"
        #     )

    # if verbose:
    #     total = int(np.count_nonzero(activity))
    #     print(f"  generated simulated activity: samples={num_samples} active_samples={total}")

    # Apply linear weighting to earlier samples if requested
    if weight_early > 0 and len(activity) > 1:
        N = len(activity)
        # Weight: 1 + k * (1 - i/N), i=0..N-1
        idx = np.arange(N)
        weights = 1.0 + weight_early * (1.0 - idx / (N - 1))
        activity = activity * weights.astype(np.float32)
    return activity


def pitch_value_to_frequency(pitch_value: int) -> float:
    """Convert Ultrastar pitch value (0 == C3) to frequency in Hz."""
    midi_note = pitch_value + 60
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))


def generate_note_frequency_trace(
    song: UltrastarSong,
    sample_rate: int,
    target_length: int,
    bpm: float,
    gap_ms: float,
) -> np.ndarray:
    """Generate an array of note frequencies aligned to the simulated timeline."""
    if target_length <= 0 or sample_rate <= 0:
        return np.zeros(max(target_length, 1), dtype=np.float32)

    trace = np.zeros(target_length, dtype=np.float32)
    beat_duration = beats_to_seconds(1.0, bpm)
    gap_seconds = gap_ms / 1000.0

    for note in song.notes:
        if note.length_beats <= 0:
            continue
        frequency = pitch_value_to_frequency(note.pitch)
        start_seconds = gap_seconds + note.start_beats * beat_duration
        end_seconds = start_seconds + note.length_beats * beat_duration
        start_idx = int(np.floor(start_seconds * sample_rate))
        end_idx = int(np.ceil(end_seconds * sample_rate))
        start_idx = max(0, min(start_idx, target_length))
        end_idx = max(0, min(end_idx, target_length))
        if end_idx > start_idx:
            trace[start_idx:end_idx] = frequency

    return trace


def detect_pitch_curve(
    audio_path: Path,
    target_sample_rate: int,
    target_length: int,
    fmin: float = librosa.note_to_hz("C2"),
    fmax: float = librosa.note_to_hz("C6"),
) -> np.ndarray:
    """Detect vocal pitch curve and resample to match the target timeline."""
    if target_length <= 0:
        return np.zeros(1, dtype=np.float32)

    try:
        y, sr = librosa.load(str(audio_path), sr=None)
    except Exception:
        return np.zeros(target_length, dtype=np.float32)

    detection_sr = sr if sr and sr >= 4000 else 22050
    if detection_sr != sr:
        y, detection_sr = librosa.load(str(audio_path), sr=detection_sr)

    hop_length = max(256, detection_sr // 200)

    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=fmin,
            fmax=fmax,
            sr=detection_sr,
            hop_length=hop_length,
        )
    except Exception:
        try:
            f0 = librosa.yin(
                y,
                fmin=fmin,
                fmax=fmax,
                sr=detection_sr,
                frame_length=2048,
                hop_length=hop_length,
            )
        except Exception:
            return np.zeros(target_length, dtype=np.float32)

    if f0 is None or len(f0) == 0:
        return np.zeros(target_length, dtype=np.float32)

    times = librosa.times_like(f0, sr=detection_sr, hop_length=hop_length)
    f0 = np.nan_to_num(f0, nan=0.0)

    if len(times) > len(f0):
        times = times[: len(f0)]
    elif len(f0) > len(times):
        f0 = f0[: len(times)]

    if len(times) == 0:
        return np.zeros(target_length, dtype=np.float32)

    duration = target_length / target_sample_rate if target_sample_rate > 0 else times[-1]
    target_times = np.linspace(0, duration, num=target_length, endpoint=False)
    interpolated = np.interp(target_times, times, f0, left=0.0, right=0.0)
    return interpolated.astype(np.float32)


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(signal) == 0:
        return signal
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(signal, kernel, mode="same")
    return smoothed.astype(np.float32)


def _fill_short_gaps(signal: np.ndarray, max_gap_samples: int, epsilon: float = 1e-6) -> np.ndarray:
    if max_gap_samples <= 0 or len(signal) == 0:
        return signal
    result = signal.copy()
    n = len(signal)
    i = 0
    while i < n:
        if abs(signal[i]) <= epsilon:
            start = i
            while i < n and abs(signal[i]) <= epsilon:
                i += 1
            end = i
            gap_len = end - start
            if (
                gap_len > 0
                and gap_len <= max_gap_samples
                and start > 0
                and end < n
                and abs(signal[start - 1]) > epsilon
                and abs(signal[end]) > epsilon
            ):
                left_val = signal[start - 1]
                right_val = signal[end]
                for offset in range(gap_len):
                    alpha = (offset + 1) / float(gap_len + 1)
                    result[start + offset] = float(left_val + (right_val - left_val) * alpha)
        else:
            i += 1
    return result.astype(np.float32)


def preprocess_signal(
    signal: np.ndarray,
    sample_rate: int,
    moving_average_window: int,
    ignore_gaps_below_ms: float,
) -> np.ndarray:
    if len(signal) == 0:
        return signal
    processed = signal.astype(np.float32, copy=True)
    if ignore_gaps_below_ms and sample_rate > 0:
        max_gap_samples = int(round(ignore_gaps_below_ms * sample_rate / 1000.0))
        if max_gap_samples > 0:
            processed = _fill_short_gaps(processed, max_gap_samples)
    if moving_average_window and moving_average_window > 1:
        processed = _moving_average(processed, moving_average_window)
    return processed


def get_simulated_intervals(
    song: UltrastarSong,
    bpm_override: Optional[float] = None,
    gap_ms_override: Optional[float] = None,
) -> list[tuple[float, float, int, int, str]]:
    """Return list of (start_sec, end_sec, start_beats, length_beats, text) for the song using optional overrides."""
    bpm = bpm_override if bpm_override is not None else song.bpm
    gap_ms = gap_ms_override if gap_ms_override is not None else song.gap_ms
    beat_duration = beats_to_seconds(1.0, bpm)
    gap_seconds = gap_ms / 1000.0

    intervals: list[tuple[float, float, int, int, str]] = []
    for note in song.notes:
        if note.length_beats <= 0:
            continue
        start_seconds = gap_seconds + note.start_beats * beat_duration
        end_seconds = start_seconds + note.length_beats * beat_duration
        intervals.append((start_seconds, end_seconds, note.start_beats, note.length_beats, note.text))
    return intervals


def _match_signal_lengths(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(reference) == len(candidate):
        return reference, candidate

    if len(candidate) < len(reference):
        pad_width = len(reference) - len(candidate)
        candidate = np.pad(candidate, (0, pad_width), mode="constant", constant_values=0.0)
    elif len(candidate) > len(reference):
        candidate = candidate[: len(reference)]

    if len(reference) < len(candidate):
        pad_width = len(candidate) - len(reference)
        reference = np.pad(reference, (0, pad_width), mode="constant", constant_values=0.0)
    elif len(reference) > len(candidate):
        reference = reference[: len(candidate)]

    return reference, candidate


def compute_alignment_score(
    reference: np.ndarray,
    candidate: np.ndarray,
    metric: str,
    epsilon: float = 1e-9,
) -> float:
    ref, cand = _match_signal_lengths(reference, candidate)
    if metric == "pearson":
        ref_centered = ref - ref.mean()
        cand_centered = cand - cand.mean()
        numerator = float(np.dot(ref_centered, cand_centered))
        denominator = float(np.linalg.norm(ref_centered) * np.linalg.norm(cand_centered))
        if denominator <= epsilon:
            return 0.0
        return numerator / denominator
    if metric == "cosine":
        numerator = float(np.dot(ref, cand))
        denominator = float(np.linalg.norm(ref) * np.linalg.norm(cand))
        if denominator <= epsilon:
            return 0.0
        return numerator / denominator
    if metric == "pointbiserial":
        # candidate expected to be binary
        if not np.array_equal(np.unique(cand), np.array([0.0])) and not np.array_equal(np.unique(cand), np.array([1.0])):
            # allow cand to be continuous but thresholded
            cand = (cand >= 0.5).astype(float)
        # compute point-biserial: difference in means divided by pooled std * sqrt(p*q)
        mask = cand == 1.0
        if mask.sum() == 0 or mask.sum() == len(cand):
            return 0.0
        m1 = ref[mask].mean()
        m0 = ref[~mask].mean()
        s = ref.std(ddof=0)
        p = float(mask.mean())
        q = 1.0 - p
        if s <= epsilon or p <= 0 or q <= 0:
            return 0.0
        r_pb = (m1 - m0) / s * np.sqrt(p * q)
        return float(r_pb)
    if metric == "ccf":
        # return maximum normalized cross-correlation at any lag using numpy
        xr = ref - ref.mean()
        yr = cand - cand.mean()
        corr = np.correlate(xr, yr, mode="full")
        denom = np.linalg.norm(xr) * np.linalg.norm(yr)
        if denom <= epsilon:
            return 0.0
        return float(np.max(corr) / denom)
    raise ValueError(f"Unknown correlation metric: {metric}")


def frange(start: float, stop: float, step: float):
    value = start
    while value <= stop + step / 2:
        yield round(value, 10)
        value += step


def _symmetric_candidates(
    base: float,
    lower: float,
    upper: float,
    step: float,
) -> list[float]:
    if step <= 0:
        raise ValueError("Step must be positive for symmetric search.")

    candidates: list[float] = []
    seen: set[float] = set()

    def _add(value: float) -> None:
        rounded = round(value, 10)
        if rounded not in seen and lower <= rounded <= upper:
            candidates.append(rounded)
            seen.add(rounded)

    if lower <= base <= upper:
        _add(base)

    offset = 1
    while True:
        positive = base + step * offset
        negative = base - step * offset
        added = False
        if positive <= upper:
            _add(positive)
            added = True
        if negative >= lower:
            _add(negative)
            added = True
        if not added:
            break
        offset += 1

    return candidates


def _gap_candidates_from_range(
    base: float,
    lower: float,
    upper: float,
    step: float,
) -> list[float]:
    if lower > upper:
        lower, upper = upper, lower

    values = list(_symmetric_candidates(base, lower, upper, step))
    if values:
        return values

    count = int((upper - lower) / step) + 1
    return [round(lower + i * step, 10) for i in range(count)]


def _adjust_range_around_value(range_min: float, range_max: float, value: float) -> tuple[float, float]:
    if range_min <= value <= range_max:
        return range_min, range_max
    span = range_max - range_min
    half_span = span / 2.0
    new_min = value - half_span
    new_max = value + half_span
    if new_min > new_max:
        new_min, new_max = new_max, new_min
    return new_min, new_max


def search_alignment(
    song: UltrastarSong,
    envelope: np.ndarray,
    sample_rate: int,
    bpm_range: tuple[float, float],
    bpm_step: float,
    gap_range_ms: tuple[int, int],
    gap_step_ms: int,
    metric: str,
    only_gap: bool = False,
    verbose: bool = False,
    weight_early: float = 0.0,
    threads: int = 1,
    use_pitch_correlation: bool = False,
    moving_average_window: int = 0,
    ignore_gaps_below_ms: float = 0.0,
) -> AlignmentResult:
    best_result = AlignmentResult(
            bpm=song.bpm,
            gap_ms=song.gap_ms,
            correlation=-np.inf,
            metric=metric,
            simulated_activity=None,
    )

    target_duration = len(envelope) / sample_rate
    target_length = len(envelope)

    bpm_min, bpm_max = _adjust_range_around_value(bpm_range[0], bpm_range[1], song.bpm)
    gap_min_f, gap_max_f = _adjust_range_around_value(gap_range_ms[0], gap_range_ms[1], song.gap_ms)
    gap_min, gap_max = int(round(gap_min_f)), int(round(gap_max_f))

    bpm_candidates = [song.bpm] if only_gap else _symmetric_candidates(song.bpm, bpm_min, bpm_max, bpm_step)
    gap_candidates = _gap_candidates_from_range(song.gap_ms, gap_min, gap_max, gap_step_ms)

    candidates = [(bpm, gap_ms) for bpm in bpm_candidates or frange(bpm_min, bpm_max, bpm_step) for gap_ms in gap_candidates]

    def build_signal(bpm_value: float, gap_value: float) -> np.ndarray:
        if use_pitch_correlation:
            base = generate_note_frequency_trace(
                song,
                sample_rate,
                target_length,
                bpm_value,
                gap_value,
            )
        else:
            base = generate_simulated_activity(
                song,
                sample_rate,
                target_duration_seconds=target_duration,
                bpm_override=bpm_value,
                gap_ms_override=gap_value,
                weight_early=weight_early,
            )
        return preprocess_signal(base, sample_rate, moving_average_window, ignore_gaps_below_ms)
    results = []
    if threads > 1:
        import concurrent.futures
        def score_candidate(args):
            bpm, gap_ms = args
            simulated = build_signal(bpm, gap_ms)
            correlation = compute_alignment_score(envelope, simulated, metric)
            return (bpm, gap_ms, correlation)
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for bpm, gap_ms, correlation in executor.map(score_candidate, candidates):
                if verbose:
                    print(f"  sweep bpm={bpm:.3f} gap={gap_ms}ms -> corr={correlation:.4f} -- best values so far: bpm={best_result.bpm:.3f} gap={best_result.gap_ms:.0f}ms corr={best_result.correlation:.4f}")
                if correlation > best_result.correlation:
                    best_result.bpm = bpm
                    best_result.gap_ms = float(gap_ms)
                    best_result.correlation = correlation
    else:
        for bpm, gap_ms in candidates:
            simulated = build_signal(bpm, gap_ms)
            correlation = compute_alignment_score(envelope, simulated, metric)
            if verbose:
                print(f"  sweep bpm={bpm:.3f} gap={gap_ms}ms -> corr={correlation:.4f} -- best values so far: bpm={best_result.bpm:.3f} gap={best_result.gap_ms:.0f}ms corr={best_result.correlation:.4f}")
            if correlation > best_result.correlation:
                best_result.bpm = bpm
                best_result.gap_ms = float(gap_ms)
                best_result.correlation = correlation
    # if the best correlation is at the edges (e.g. +5000 or -5000 ms), return the original values
    if best_result.gap_ms in (gap_min, gap_max):
        best_result.gap_ms = song.gap_ms

    if best_result.bpm in (bpm_min, bpm_max):
        best_result.bpm = song.bpm

    # regenerate simulated activity for the winning parameters (verbose prints intervals)
    if use_pitch_correlation:
        winning_sim = build_signal(best_result.bpm, best_result.gap_ms)
        if verbose:
            print("Generated pitch-based simulated trace for winning parameters.")
    else:
        raw_sim = generate_simulated_activity(
            song,
            sample_rate,
            target_duration_seconds=target_duration,
            bpm_override=best_result.bpm,
            gap_ms_override=best_result.gap_ms,
            verbose=verbose,
            weight_early=weight_early,
        )
        winning_sim = preprocess_signal(raw_sim, sample_rate, moving_average_window, ignore_gaps_below_ms)
    best_result.simulated_activity = winning_sim
    return best_result


def prepare_audio(config: UltraTimerConfig) -> PreparedAudioArtifacts:
    """Generate cached audio artifacts (vocals, muted) ready for analysis."""

    cache_dir = config.cache_dir
    create_folder(str(cache_dir))

    pytorch_device = _select_pytorch_device(config)
    demucs_model = config.demucs_model or DemucsModel.HTDEMUCS

    separation_dir_str = separate_vocal_from_audio(
        str(cache_dir),
        str(config.audio_path),
        True,
        False,
        pytorch_device,
        demucs_model,
        config.skip_cache_separation,
    )
    separation_dir = Path(separation_dir_str)
    vocals_path = separation_dir / "vocals.wav"
    instrumental_path = separation_dir / "no_vocals.wav"

    if not vocals_path.exists():
        raise FileNotFoundError(
            "Vocal stem not found after separation. Expected at " f"{vocals_path}"
        )

    audio_stem = config.audio_path.stem

    # Optionally denoise (can introduce latency); allow skipping.
    source_for_mono = vocals_path
    if not config.no_denoise:
        try:
            denoised_vocals_path = cache_dir / f"{audio_stem}_denoised.wav"
            # Denose if not skipping; reuse cache via our wrapper
            from modules.os_helper import check_file_exists
            if not check_file_exists(str(denoised_vocals_path)):
                denoise_vocal_audio(str(vocals_path), str(denoised_vocals_path), skip_cache=False)
            source_for_mono = denoised_vocals_path
        except Exception as e:
            # Fall back silently if denoise fails
            source_for_mono = vocals_path

    mono_vocals_path = cache_dir / f"{audio_stem}_mono.wav"
    convert_audio_to_mono_wav(str(source_for_mono), str(mono_vocals_path))

    muted_vocals_path = cache_dir / f"{audio_stem}_mute.wav"
    mute_no_singing_parts(str(mono_vocals_path), str(muted_vocals_path))

    waveform, sample_rate = librosa.load(
        str(muted_vocals_path), sr=config.sample_rate, mono=True
    )
    waveform = _normalize_waveform(waveform)
    envelope = compute_energy_envelope(waveform, sample_rate, config.envelope_window_ms)
    envelope = _normalize_waveform(envelope)
    binary_trace = None
    if config.binary_threshold is not None:
        threshold = float(config.binary_threshold)
        binary_trace = binarize_signal(envelope, threshold)

    return PreparedAudioArtifacts(
        cache_dir=cache_dir,
        separation_dir=separation_dir,
        vocals_path=vocals_path,
        instrumental_path=instrumental_path,
        mono_vocals_path=mono_vocals_path,
        muted_vocals_path=muted_vocals_path,
        waveform=waveform,
        sample_rate=sample_rate,
        debug_plot_path=config.debug_plot_path,
        envelope=envelope,
        binary_trace=binary_trace,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> UltraTimerConfig:
    parser = argparse.ArgumentParser(
        description="Find optimal GAP and BPM to align an Ultrastar txt with vocals",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for parallel candidate scoring (default: 1).",
    )
    parser.add_argument(
        "--weight-early",
        type=float,
        default=0.0,
        help="Linearly increase weight for earlier notes in simulated activity (0=off, >0=stronger).",
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=_existing_file,
        required=True,
        help="Path to the source audio file (mp3, m4a, wav, etc.).",
    )
    parser.add_argument(
        "--txt",
        "-t",
        type=_existing_file,
        required=True,
        help="Path to the Ultrastar txt file to analyze.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path.cwd(),
        help="Directory where results (updated txt, diagnostics) will be written.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Override cache directory. Defaults to <output>/cache.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override torch device (e.g. cuda:0, cpu). If omitted, auto-detect.",
    )
    parser.add_argument(
        "--demucs",
        type=str,
        choices=[model.value for model in DemucsModel],
        default=None,
        help="Select Demucs model variant for vocal separation.",
    )
    parser.add_argument(
        "--skip-cache-separation",
        action="store_true",
        help="Recompute vocal separation even if cached files exist.",
    )
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip denoise step (use separated vocals directly).",
    )
    parser.add_argument(
        "--sample-rate",
        type=_positive_int,
        default=10000,
        help="Analysis sample rate in Hz (after resampling). Lower = smaller traces. Example: 200 Hz ~ 5 ms.",
    )
    parser.add_argument(
        "--resolution-ms",
        type=_positive_float,
        default=None,
        help="Desired time resolution in milliseconds (overrides --sample-rate). Example: 5 -> ~200 Hz.",
    )
    parser.add_argument(
        "--envelope-window-ms",
        type=_positive_float,
        default=20.0,
        help="Window size in milliseconds for computing the energy envelope.",
    )
    parser.add_argument(
        "--moving-average",
        type=_non_negative_int,
        default=0,
        help="Apply a moving average with this window size (in samples) to traces before correlation/plots.",
    )
    parser.add_argument(
        "--ignore-gaps-below-ms",
        type=_non_negative_float,
        default=0.0,
        help="Fill gaps shorter than this duration (ms) by connecting neighboring values before correlation/plots.",
    )
    parser.add_argument(
        "--bpm-range",
        type=lambda s: _range_pair(s, float),
        default=(-0.05, 0.05),
        help="Coarse BPM search range (inclusive) as 'min,max'.",
    )
    parser.add_argument(
        "--bpm-step",
        type=_positive_float,
        default=0.05,
        help="Step size for BPM sweep during coarse search.",
    )
    parser.add_argument(
        "--gap-range-ms",
        type=lambda s: _range_pair(s, int),
        default=(-5000, 5000),
        help="GAP search range in milliseconds as 'min,max'.",
    )
    parser.add_argument(
        "--gap-step-ms",
        type=_positive_int,
        default=20,
        help="Step size in milliseconds for GAP sweep during coarse search.",
    )
    parser.add_argument(
        "--only-gap",
        action="store_true",
        help="Only optimize GAP (keep BPM fixed to the txt's BPM).",
    )
    parser.add_argument(
        "--correlation",
        choices=["pearson", "cosine", "pointbiserial", "ccf"],
        default="cosine",
        help="Correlation metric to evaluate alignment quality.",
    )
    parser.add_argument(
        "--preprocess",
        choices=["envelope", "rms", "none"],
        default="envelope",
        help="Preprocessing to apply to audio before correlation: envelope or rms (root-mean-square) or none.",
    )
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=None,
        help="If provided, binarize audio envelope at this threshold (0-1) before correlation.",
    )
    parser.add_argument(
        "--debug-plot",
        type=Path,
        default=None,
        help="Optional path for saving waveform debug image. Defaults to <output>/<audio>_waveform.png.",
    )
    parser.add_argument(
        "--no-debug-plot",
        action="store_true",
        help="Disable waveform debug image generation.",
    )
    parser.add_argument(
        "--plot-pitch",
        action="store_true",
        help="Overlay Ultrastar note pitch and detected vocal pitch on debug plots.",
    )
    parser.add_argument(
        "--pitch-correlation",
        action="store_true",
        help="Use pitch traces (Ultrastar notes vs detected pitch) for correlation instead of binary activity.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging.",
    )

    args = parser.parse_args(argv)

    output_dir = args.output.expanduser().resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    if args.cache is None:
        cache = output_dir / "cache"
    else:
        cache = args.cache.expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)

    demucs_model = DemucsModel(args.demucs) if args.demucs else None

    generate_debug_plot = not args.no_debug_plot
    if generate_debug_plot:
        if args.debug_plot:
            debug_plot_path = args.debug_plot.expanduser().resolve()
            debug_plot_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            default_name = f"{args.audio.stem}_waveform_debug.png"
            debug_plot_path = output_dir / default_name
    else:
        debug_plot_path = None

    # If a target time resolution is provided, derive an effective analysis sample rate
    effective_sample_rate = args.sample_rate
    if args.resolution_ms is not None:
        try:
            eff_sr = int(round(1000.0 / float(args.resolution_ms)))
            effective_sample_rate = max(1, eff_sr)
        except Exception:
            effective_sample_rate = args.sample_rate

    config = UltraTimerConfig(
        audio_path=args.audio,
        ultrastar_txt_path=args.txt,
        output_dir=output_dir,
        cache_dir=cache,
        pytorch_device=args.device,
        demucs_model=demucs_model,
        skip_cache_separation=args.skip_cache_separation,
        no_denoise=args.no_denoise,
        sample_rate=effective_sample_rate,
        envelope_window_ms=args.envelope_window_ms,
        bpm_search_range=args.bpm_range,
        bpm_search_step=args.bpm_step,
        gap_search_range_ms=args.gap_range_ms,
        gap_search_step_ms=args.gap_step_ms,
        only_gap=args.only_gap,
            correlation_metric=args.correlation,
            preprocess=args.preprocess,
        binary_threshold=args.binary_threshold,
        verbose=args.verbose,
        generate_debug_plot=generate_debug_plot,
        debug_plot_path=debug_plot_path,
        weight_early=args.weight_early,
        threads=args.threads,
        plot_pitch=args.plot_pitch,
        use_pitch_correlation=args.pitch_correlation,
        moving_average_window=args.moving_average,
        ignore_gaps_below_ms=args.ignore_gaps_below_ms,
    )
    return config


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    if config.verbose:
        print("UltraTimer configuration:")
        for field in dataclass_fields(config):
            value = getattr(config, field.name)
            print(f"  {field.name}: {value}")
    song_data = parse_ultrastar_song(config.ultrastar_txt_path)
    if config.verbose:
        print("Parsed Ultrastar song:")
        print(f"  bpm: {song_data.bpm}")
        print(f"  gap_ms: {song_data.gap_ms}")
        print(f"  note_count: {len(song_data.notes)}")
    audio_artifacts = prepare_audio(config)
    if config.verbose:
        print("Prepared audio artifacts:")
        for field in dataclass_fields(audio_artifacts):
            value = getattr(audio_artifacts, field.name)
            if field.name == "waveform":
                summary = f"array(shape={value.shape}, dtype={value.dtype})"
                print(f"  {field.name}: {summary}")
            elif field.name == "envelope":
                summary = f"array(shape={value.shape}, dtype={value.dtype})"
                print(f"  {field.name}: {summary}")
            elif field.name == "binary_trace" and value is not None:
                summary = f"array(shape={value.shape}, dtype={value.dtype})"
                print(f"  {field.name}: {summary}")
            else:
                print(f"  {field.name}: {value}")
    reference_signal = audio_artifacts.binary_trace if audio_artifacts.binary_trace is not None else audio_artifacts.envelope

    use_pitch_correlation = config.use_pitch_correlation
    need_pitch_trace = use_pitch_correlation or (config.plot_pitch and config.generate_debug_plot)
    pitch_detection_trace: Optional[np.ndarray] = None
    if need_pitch_trace:
        pitch_detection_trace = detect_pitch_curve(
            audio_artifacts.muted_vocals_path,
            audio_artifacts.sample_rate,
            len(reference_signal),
        )
        if use_pitch_correlation:
            if pitch_detection_trace is None or len(pitch_detection_trace) == 0:
                print("Warning: pitch detection failed; falling back to envelope for correlation.")
                use_pitch_correlation = False
            else:
                reference_signal = pitch_detection_trace

    reference_signal = preprocess_signal(
        reference_signal,
        audio_artifacts.sample_rate,
        config.moving_average_window,
        config.ignore_gaps_below_ms,
    )

    if pitch_detection_trace is not None and len(pitch_detection_trace) > 0:
        pitch_detection_trace = preprocess_signal(
            pitch_detection_trace,
            audio_artifacts.sample_rate,
            config.moving_average_window,
            config.ignore_gaps_below_ms,
        )

    if config.generate_debug_plot and config.debug_plot_path is not None:
        target_duration = len(reference_signal) / audio_artifacts.sample_rate
        if use_pitch_correlation:
            baseline_simulated = generate_note_frequency_trace(
                song_data,
                audio_artifacts.sample_rate,
                len(reference_signal),
                song_data.bpm,
                song_data.gap_ms,
            )
        else:
            baseline_simulated = generate_simulated_activity(
                song_data,
                audio_artifacts.sample_rate,
                target_duration_seconds=target_duration,
                weight_early=config.weight_early,
            )
        baseline_note_trace: Optional[np.ndarray] = None
        if config.plot_pitch:
            baseline_note_trace = generate_note_frequency_trace(
                song_data,
                audio_artifacts.sample_rate,
                len(baseline_simulated),
                song_data.bpm,
                song_data.gap_ms,
            )
        baseline_simulated = preprocess_signal(
            baseline_simulated,
            audio_artifacts.sample_rate,
            config.moving_average_window,
            config.ignore_gaps_below_ms,
        )
        if baseline_note_trace is not None:
            baseline_note_trace = preprocess_signal(
                baseline_note_trace,
                audio_artifacts.sample_rate,
                config.moving_average_window,
                config.ignore_gaps_below_ms,
            )
        # if config.verbose:
        #     print("Simulated intervals (initial):")
        #     for s, e, sb, lb, txt in get_simulated_intervals(song_data):
        #         print(f"  start={s:.3f}s end={e:.3f}s beats={sb} len={lb} text={txt}")
        # if config.verbose:
        #     print(
        #         "Baseline simulated activity samples:",
        #         int(np.count_nonzero(baseline_simulated)),
        #     )
        initial_plot_path = config.debug_plot_path.with_name(
            f"{config.debug_plot_path.stem}_initial{config.debug_plot_path.suffix}"
        )
        save_waveform_debug_plot(
            audio_artifacts.waveform,
            baseline_simulated,
            audio_artifacts.sample_rate,
            initial_plot_path,
            title="Initial waveform vs simulated activity",
            secondary_waveform=reference_signal,
            secondary_label="Reference signal",
            pitch_overlay=(baseline_note_trace, pitch_detection_trace)
            if config.plot_pitch and baseline_note_trace is not None
            else None,
        )
        if config.verbose:
            print(f"Initial debug plot written to {initial_plot_path}")

    alignment_result = search_alignment(
        song_data,
        reference_signal,
        audio_artifacts.sample_rate,
        config.bpm_search_range,
        config.bpm_search_step,
        config.gap_search_range_ms,
        config.gap_search_step_ms,
        config.correlation_metric,
        only_gap=config.only_gap,
        verbose=config.verbose,
        weight_early=config.weight_early,
        threads=config.threads,
        use_pitch_correlation=use_pitch_correlation,
        moving_average_window=config.moving_average_window,
        ignore_gaps_below_ms=config.ignore_gaps_below_ms,
    )
    print(
        f"Best alignment -> BPM: {alignment_result.bpm:.3f}, GAP: {alignment_result.gap_ms:.0f} ms, "
        f"correlation ({alignment_result.metric}): {alignment_result.correlation:.4f}"
    )
    # if config.verbose:
    #     print("Simulated intervals (aligned):")
    #     for s, e, sb, lb, txt in get_simulated_intervals(song_data, bpm_override=alignment_result.bpm, gap_ms_override=alignment_result.gap_ms):
    #         print(f"  start={s:.3f}s end={e:.3f}s beats={sb} len={lb} text={txt}")
    if config.generate_debug_plot:
        # if config.verbose:
        #     print(
        #         "Aligned simulated activity samples:",
        #         int(np.count_nonzero(alignment_result.simulated_activity)),
        #     )
        aligned_note_trace: Optional[np.ndarray] = None
        if config.plot_pitch:
            aligned_note_trace = generate_note_frequency_trace(
                song_data,
                audio_artifacts.sample_rate,
                len(alignment_result.simulated_activity),
                alignment_result.bpm,
                alignment_result.gap_ms,
            )
            aligned_note_trace = preprocess_signal(
                aligned_note_trace,
                audio_artifacts.sample_rate,
                config.moving_average_window,
                config.ignore_gaps_below_ms,
            )
        save_waveform_debug_plot(
            audio_artifacts.waveform,
            alignment_result.simulated_activity,
            audio_artifacts.sample_rate,
            config.debug_plot_path,
            title="Waveform vs. simulated activity",
            secondary_waveform=reference_signal,
            secondary_label="Reference signal",
            pitch_overlay=(aligned_note_trace, pitch_detection_trace)
            if config.plot_pitch and aligned_note_trace is not None
            else None,
        )
        # if config.verbose:
        #     print(f"Waveform debug plot written to {config.debug_plot_path}")
        #     print(
        #         "Simulated trace generated from Ultrastar timing with current BPM/GAP."  # noqa: E501
        #     )
    # round the gap to the next 10 ms for Ultrastar
    rounded_gap = int(round(alignment_result.gap_ms / 10.0)) * 10
    print(f"Updating Ultrastar txt file at {config.ultrastar_txt_path}")
    print(f"Original BPM: {song_data.bpm:.3f} and GAP: {song_data.gap_ms:.0f} ms")
    print(f" Optimal BPM: {alignment_result.bpm:.3f} and GAP: {rounded_gap:.0f} ms")
    # if the gap is different now, update the txt file
    rounded_bpm = round(alignment_result.bpm, 2)
    if (rounded_bpm > 80 and rounded_bpm < 800 and abs(round(song_data.bpm,2) - rounded_bpm) >= 0.05) or (rounded_gap > 0 and abs(int(round(song_data.gap_ms)) - rounded_gap) >= 10):
        # create a .bak file first
        backup_path = config.ultrastar_txt_path.with_suffix(config.ultrastar_txt_path.suffix + ".bak")
        shutil.copy2(config.ultrastar_txt_path, backup_path)
        if config.verbose:
            print(f"Backup of original txt created at {backup_path}")
        gap_updated = False
        bpm_updated = False
        # update file in place
        with config.ultrastar_txt_path.open("r+", encoding="utf-8") as handle:
            lines = handle.readlines()
            handle.seek(0)
            for line in lines:
                if line.strip().startswith("#GAP:") and (rounded_gap > 0 and abs(int(round(song_data.gap_ms)) - rounded_gap) >= 10):
                    handle.write(f"#GAP:{rounded_gap}\n")
                    gap_updated = True
                elif line.strip().startswith("#BPM:") and (rounded_bpm > 80 and rounded_bpm < 800 and abs(round(song_data.bpm,2) - rounded_bpm) >= 0.05):
                    handle.write(f"#BPM:{rounded_bpm:.2f}\n")
                    bpm_updated = True
                else:
                    handle.write(line)
            handle.truncate()
    else:
        print("GAP and BPM unchanged (or no better GAP/BPM found); no update to txt file needed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
