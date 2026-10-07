#!/usr/bin/env python3
"""
Шаг 1 пайплайна: audio -> ECAPA-TDNN embedding (+ proxy-качество сегмента).
Читает data/manifests/manifest.csv, пишет data/embeddings/embeddings.npz.

Схема манифеста (актуальная):
    audio_path, speaker_type, client_id, language, duration,
    sentence, session, split

Зачем speech_score: 
    Твои own-окна режутся VAD'ом, но impostor из Common Voice может содержать 
    тишину/обрезки. speech_score = доля энергичных кадров, чтобы на этапе профиля 
    не пускать мёртвые сегменты в эталоны владельца.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm

TARGET_SR = 16000
FRAME_MS = 25
HOP_MS = 10


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    
def speech_score(wav: np.ndarray, sr: int) -> float:
    """
    Доля кадров с энергией выше порога. 
    Тишина -> ~0, речь -> 0.5..0.9.
    Это дешевый фильтр качества аудио перед отправкой в модель.
    """
    frame = max(1, int(sr * FRAME_MS / 1000))
    hop = max(1, int(sr * HOP_MS / 1000))
    
    n_frames = (len(wav) - frame) // hop + 1

    idx = np.arange(frame)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = wav[idx]
    
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    thr = max(rms.max() * 0.1, 1e-4)  # порог относительно пика записи
    
    return float((rms > thr).mean())


def load_mono_16k(path: Path) -> np.ndarray:
    """Приводим любой WAV к float32 mono 16k."""
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass # Если уже установлен или нет такого метода, игнорируем
        
    wav, sr = torchaudio.load(str(path))
    
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        
    return wav.squeeze(0).numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/manifest.csv")
    ap.add_argument("--project", default=".")
    ap.add_argument("--out", default="data/embeddings/embeddings.npz")
    ap.add_argument("--model", default="speechbrain/spkrec-ecapa-voxceleb")
    ap.add_argument("--limit", type=int, default=0, help="0 = все строки")
    ap.add_argument("--smoke", action="store_true", help="режим быстрой проверки")
    args = ap.parse_args()

    project = Path(args.project).expanduser().resolve()
    manifest = Path(args.manifest).expanduser()
    if not manifest.is_absolute():
        manifest = project / manifest
        
    if not manifest.exists():
        raise SystemExit(f"Манифест не найден: {manifest}")

    device = pick_device()
    print(f"Device: {device}")

    # Загружаем предобученную модель один раз
    print("Loading SpeechBrain ECAPA-TDNN...")
    model = EncoderClassifier.from_hparams(
        source=args.model, 
        savedir="pretrained_models/ecapa"
    ).to(device)
    model.eval()
    print("Model ready.\n")

    with manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        
    if args.limit:
        rows = rows[:args.limit]
        print(f"LIMIT MODE: processing only first {len(rows)} rows\n")

    embs, paths, stypes, cids, langs, sess, splits, sscores, durs = (
        [], [], [], [], [], [], [], [], []
    )

    iterator = tqdm(rows, desc="Extracting embeddings", disable=args.smoke)
    
    for r in iterator:
        p_raw = (r.get("audio_path") or "").strip()
        if not p_raw:
            continue
            
        p = Path(p_raw).expanduser()
        if not p.is_absolute():
            p = project / p
        p = p.resolve(strict=False)
          
        try:
            wav_np = load_mono_16k(p)
        except Exception as e:
            print(f"\nLoad fail {p}: {e}")
            continue

        # 1. Считаем качество сигнала (для фильтрации мусора позже)
        score = speech_score(wav_np, TARGET_SR)
        
        # 2. Получаем эмбеддинг через модель
        wav_t = torch.from_numpy(wav_np).unsqueeze(0).to(device)  # (1, T)
        
        with torch.no_grad():
            emb = model.encode_batch(wav_t)  # -> (1, 1, D), обычно D=192
        
        emb = emb.squeeze()                  # -> (D,)
        emb = F.normalize(emb, dim=0)        # нормализуем для cosine similarity

        # Сохраняем метаданные рядом с вектором
        embs.append(emb.cpu().numpy().astype(np.float32))
        
        # Путь сохраняем относительным, чтобы манифест был переносимым
        try:
            rel_path = str(p.relative_to(project))
        except ValueError:
            rel_path = str(p)
            
        paths.append(rel_path)
        stypes.append((r.get("speaker_type") or "").strip().lower())
        cids.append((r.get("client_id") or "").strip())
        langs.append((r.get("language") or "").strip().lower())
        sess.append((r.get("session") or "").strip())
        splits.append((r.get("split") or "").strip())
        sscores.append(score)
        durs.append(len(wav_np) / TARGET_SR)

    if not embs:
        raise SystemExit("No valid embeddings extracted! Check paths in manifest.")

    embs_arr = np.stack(embs)
    
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = project / out_path
        
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Сохраняем всё в один .npz архив
    np.savez_compressed(
        out_path,
        embeddings=embs_arr,          # (N, 192) float32
        paths=np.array(paths),        # (N,) string
        speaker_type=np.array(stypes),# (N,) string
        client_id=np.array(cids),     # (N,) string
        language=np.array(langs),     # (N,) string
        session=np.array(sess),       # (N,) string
        split=np.array(splits),       # (N,) string
        speech_score=np.array(sscores, dtype=np.float32), # (N,) float
        duration=np.array(durs, dtype=np.float32),        # (N,) float
    )

    print(f"\nSaved {embs_arr.shape[0]} embeddings to {out_path}")
    print(f"Embedding dimension: {embs_arr.shape[1]}")
    
    # Быстрая статистика по владельцу
    own_mask = np.array(stypes) == "owner"
    if own_mask.any():
        mean_ss = np.mean(np.array(sscores)[own_mask])
        print(f"Owner speech_score mean: {mean_ss:.2f}")
        
    if args.smoke:
        print("\nSMOKE TEST OK. You can run full extraction without --smoke/--limit.")


if __name__ == "__main__":
    main()