#!/usr/bin/env python3
"""
Нарекает свои сессии (только RU) с VAD.
Вход: /home/fedosdan2/Music/session_*.wav
Выход: data/own/ru/*.wav + data/own/metadata.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf

TARGET_SR = 16000
MIN_SEG_SEC = 1.5
MAX_SEG_SEC = 8.0
CUT_SEC = 6.0
TAIL_MIN_SEC = 2.0

FIELDS = ["file", "speaker_type", "client_id", "language", "duration", "sentence", "session"]


def load_mono_16k(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0)


def cut_long_segment(start: int, end: int, sr: int) -> list[tuple[int, int]]:
    dur = (end - start) / sr
    if dur <= MAX_SEG_SEC:
        return [(start, end)]
    
    pieces = []
    step = int(CUT_SEC * sr)
    cur = start
    while cur < end:
        nxt = min(cur + step, end)
        pieces.append([cur, nxt])
        cur = nxt
    
    if len(pieces) >= 2:
        last_dur = (pieces[-1][1] - pieces[-1][0]) / sr
        if last_dur < TAIL_MIN_SEC:
            pieces[-2][1] = pieces[-1][1]
            pieces.pop()
            
    return [tuple(p) for p in pieces]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="/home/fedosdan2/Music")
    ap.add_argument("--out-dir", default="data/own")
    ap.add_argument("--speaker-id", default="OWNER_001")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    ru_out_dir = out_dir / "ru"
    ru_out_dir.mkdir(parents=True, exist_ok=True)

    # Берем ВСЕ session_*.wav, предполагая, что теперь нас интересует только RU контекст
    # Или можно фильтровать: if 'en' not in name.lower()
    wav_files = sorted(raw_dir.glob("session_*.wav"))
    
    if not wav_files:
        raise SystemExit(f"Не найдено session_*.wav в {raw_dir}")

    print(f"Найдено сессий для обработки (RU only): {len(wav_files)}")
    for w in wav_files:
        print(f"  {w.name}")

    print("\nЗагружаю Silero VAD...")
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
        onnx=False,
    )
    get_speech_timestamps = utils[0]
    print("VAD готов.\n")

    rows = []
    per_session = {}

    for wav_path in wav_files:
        # Принудительно считаем всё русским, так как EN убрали из задачи
        lang = "ru" 
        session = wav_path.stem 

        try:
            wav = load_mono_16k(wav_path)
        except Exception as e:
            print(f"Ошибка загрузки {wav_path}: {e}")
            continue

        ts = get_speech_timestamps(
            wav, vad_model,
            sampling_rate=TARGET_SR,
            threshold=0.5,
            min_speech_duration_ms=250,
            max_speech_duration_s=30,
            min_silence_duration_ms=100,
            speech_pad_ms=30,
        )

        kept = 0
        for seg in ts:
            for (s, e) in cut_long_segment(seg["start"], seg["end"], TARGET_SR):
                dur = (e - s) / TARGET_SR
                if dur < MIN_SEG_SEC:
                    continue
                
                chunk = wav[s:e].numpy().astype(np.float32)
                chunk = np.clip(chunk, -1.0, 1.0)

                fname = f"{lang}_{session}_{kept:03d}.wav"
                fpath = ru_out_dir / fname
                sf.write(fpath, chunk, TARGET_SR, subtype="PCM_16")

                rows.append({
                    "file": fname,
                    "speaker_type": "owner",
                    "client_id": args.speaker_id,
                    "language": lang,
                    "duration": f"{dur:.3f}",
                    "sentence": "",
                    "session": session,
                })
                kept += 1

        per_session[session] = kept
        print(f"  {session}: речевых сегментов = {kept}")

    meta = out_dir / "metadata.csv"
    with meta.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\nГотово. Всего сегментов RU: {len(rows)}")
    print(f"Сессий: {len(per_session)}")
    print(f"Metadata: {meta}")


if __name__ == "__main__":
    main()