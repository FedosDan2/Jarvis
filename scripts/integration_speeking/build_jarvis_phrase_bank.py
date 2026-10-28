#!/usr/bin/env python3
"""
Строит банк фраз Джарвиса из папки с аудио.

Вход:
    raw_jarvis_voice/*.wav|mp3|ogg|m4a

Выход:
    models/tts/jarvis/clips/*.wav
    models/tts/jarvis/phrase_index.json

Каждая фраза:
    - конвертируется в WAV 16 kHz mono;
    - транскрибируется через SpeechToText;
    - сохраняется в индекс для поиска по тексту.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


import sys
from pathlib import Path

# Корень проекта: scripts/integration_speeking/file.py -> parents[2] = Jarvis
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.integration_sound_detect.utils.audio_preprocessor import AudioPreprocessor
from scripts.integration_sound_detect.utils.speech_to_text import SpeechToText

TARGET_SR = 16000
MIN_DUR = 0.5
MAX_DUR = 15.0

AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".opus", ".m4a", ".flac", ".webm"}


def speech_score(wav: np.ndarray, sr: int = TARGET_SR) -> float:
    """Простая метрика: доля энергичных кадров."""
    if wav.size == 0:
        return 0.0

    frame = max(1, int(sr * 0.025))
    hop = max(1, int(sr * 0.010))
    n = (len(wav) - frame) // hop + 1

    if n <= 0:
        rms = float(np.sqrt(np.mean(wav ** 2) + 1e-12))
        return 1.0 if rms > 1e-4 else 0.0

    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    frames = wav[idx]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    thr = max(float(rms.max()) * 0.1, 1e-4)
    return float((rms > thr).mean())


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\sа-яё]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="raw_jarvis_voice")
    ap.add_argument("--output-dir", default="models/tts/jarvis")
    ap.add_argument("--language", default=None, help="ru / en / None=auto")
    ap.add_argument("--stt-model", default="small")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "phrase_index.json"

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print("Loading AudioPreprocessor...")
    preprocessor = AudioPreprocessor(device=device)

    print("Loading SpeechToText...")
    stt = SpeechToText(
        model_size=args.stt_model,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        beam_size=3,
        vad_filter=True,
    )

    files = []
    for ext in AUDIO_EXTS:
        files.extend(input_dir.rglob(f"*{ext}"))

    files = sorted(set(files))

    if not files:
        raise SystemExit(f"No audio files found in {input_dir}")

    print(f"Found files: {len(files)}")

    records = []

    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path.name}")

        try:
            tensor = preprocessor.preprocess(str(path))
        except Exception as e:
            print(f"  skip preprocess: {e}")
            continue

        wav = tensor.cpu().numpy()
        if wav.ndim == 2:
            wav = wav.squeeze(0)

        duration = len(wav) / TARGET_SR

        if duration < MIN_DUR or duration > MAX_DUR:
            print(f"  skip duration={duration:.2f}s")
            continue

        score = speech_score(wav)

        if score < 0.15:
            print(f"  skip low speech_score={score:.2f}")
            continue

        clip_name = f"jarvis_{i:04d}.wav"
        clip_path = clips_dir / clip_name

        sf.write(str(clip_path), wav, TARGET_SR, subtype="PCM_16")

        try:
            stt_result = stt.transcribe(str(clip_path), language=args.language)
            raw_text = stt_result.text
            language = stt_result.language
        except Exception as e:
            print(f"  STT failed: {e}")
            raw_text = ""
            language = args.language or "unknown"

        norm_text = normalize_text(raw_text)

        records.append({
            "id": f"jarvis_{i:04d}",
            "file": str(clip_path.relative_to(output_dir)),
            "source_file": str(path),
            "text": raw_text,
            "normalized_text": norm_text,
            "language": language,
            "duration": round(duration, 3),
            "speech_score": round(score, 3),
        })

    with index_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Index: {index_path}")
    print(f"Clips: {clips_dir}")
    print(f"Phrases: {len(records)}")


if __name__ == "__main__":
    main()